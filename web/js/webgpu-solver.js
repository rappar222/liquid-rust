// WebGPU Compute Engine for LiquidRust
// Massively parallel PBF solver running 100% on GPU compute shaders
// Includes Akinci free-surface cohesion, flow-gated vorticity confinement,
// and single-encoder / single-pass substep batching for minimal driver overhead.

import { Vec3, Aabb } from './math.js?v=20261113e';
import { ScenePreset } from './pbf-solver.js?v=20261113e';
import { PBF_COMPUTE_WGSL } from './pbf-shader.js?v=20261113e';

export class WebGpuSolver {
    constructor(device) {
        this.device = device;
        this.maxParticles = 160000;
        this.particleCount = 0;
        this.capacity = 160000;

        // Physics parameters (default: 2D Planar Mode, Draft lattice)
        this.is2D = true;
        this.quality = 'draft';
        this.gravity = new Vec3(0, -9.81, 0);
        this.particleRadius = 0.024;
        this.spacing = 0.040;
        this.h = this.particleRadius * 3.1;
        this.h2 = this.h * this.h;
        this.restDensity = 1000.0;
        this.invRestDensity = 1.0 / this.restDensity;
        this.viscosity = 0.035;
        this.vorticity = 0.045;
        this.surfaceTension = 0.030;
        this.substeps = 2;
        this.solverIterations = 3;
        this.tensileK = 0.001;
        this.restitution = 0.02;
        this.boundaryFriction = 0.95;
        this.maxVelocity = 12.0;

        // Chamber bounds (meters)
        this.bounds = new Aabb(new Vec3(-0.9, 0.0, -0.04), new Vec3(0.9, 1.2, 0.04));

        // Interactive obstacle
        this.obstacleEnabled = false;
        this.obstaclePos = new Vec3(0.0, 0.35, 0.0);
        this.obstacleRadius = 0.22;

        // Mouse interaction
        this.mouseRay = null;
        this.mouseForceMode = 0;
        this.mouseStrength = 25.0;

        // Grid parameters
        this.cellSize = this.h;
        this.gridDim = [1, 1, 1];
        this.totalCells = 1;

        // Pre-allocated uniform staging buffer (976 bytes = 244 x 32-bit words, zero GC per frame)
        this.uBuf = new ArrayBuffer(976);
        this.uF32 = new Float32Array(this.uBuf);
        this.uU32 = new Uint32Array(this.uBuf);

        // Compute pipelines & GPU buffers
        this.initPipelines();
        this.allocateBuffers(this.capacity);
        this.setQuality('draft');
    }

    initPipelines() {
        const device = this.device;
        const shaderModule = device.createShaderModule({
            label: 'PBF Compute Shader Module',
            code: PBF_COMPUTE_WGSL,
        });

        // 8 Bindings: uniforms (0), particles (1), pred_pos (2), old_pos (3), lambdas (4), delta_pos (5), grid_heads (6), grid_next (7)
        this.bindGroupLayout = device.createBindGroupLayout({
            label: 'PBF Compute BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });

        const pipelineLayout = device.createPipelineLayout({
            label: 'PBF Compute Pipeline Layout',
            bindGroupLayouts: [this.bindGroupLayout],
        });

        const makePipeline = (entryPoint, label) => device.createComputePipeline({
            label,
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint },
        });

        this.pipePredict = makePipeline('predict_positions', 'PBF Predict Pipeline');
        this.pipeClearGrid = makePipeline('clear_grid', 'PBF Clear Grid Pipeline');
        this.pipeInsert = makePipeline('insert_particles', 'PBF Insert Pipeline');
        this.pipeDensityLambda = makePipeline('compute_density_lambda', 'PBF Density Pipeline');
        this.pipeDeltaPos = makePipeline('compute_delta_positions', 'PBF Delta Pipeline');
        this.pipeApplyDelta = makePipeline('apply_delta_and_collide', 'PBF Apply Delta Pipeline');
        this.pipeUpdateVelocity = makePipeline('update_velocity_and_xsph', 'PBF Velocity Pipeline');
        this.pipeVorticity = makePipeline('apply_vorticity', 'PBF Vorticity Pipeline');
    }

    computeGridExtents() {
        const bSize = this.bounds.size();
        this.cellSize = this.h;
        this.gridDim[0] = Math.max(1, Math.ceil(bSize.x / this.cellSize));
        this.gridDim[1] = Math.max(1, Math.ceil(bSize.y / this.cellSize));
        this.gridDim[2] = Math.max(1, Math.ceil(Math.max(0.08, bSize.z) / this.cellSize));
        this.totalCells = this.gridDim[0] * this.gridDim[1] * this.gridDim[2];
    }

    rebuildBindGroup() {
        this.bindGroup = this.device.createBindGroup({
            label: 'PBF Compute BindGroup',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuf } },
                { binding: 1, resource: { buffer: this.particleBuf } },
                { binding: 2, resource: { buffer: this.predPosBuf } },
                { binding: 3, resource: { buffer: this.oldPosBuf } },
                { binding: 4, resource: { buffer: this.lambdasBuf } },
                { binding: 5, resource: { buffer: this.deltaPosBuf } },
                { binding: 6, resource: { buffer: this.gridHeadsBuf } },
                { binding: 7, resource: { buffer: this.gridNextBuf } },
            ],
        });
    }

    updateGridDimensions() {
        this.computeGridExtents();
        const requiredBytes = Math.max(4096, this.totalCells * 4);
        if (this.gridHeadsBuf && this.gridHeadsBuf.size < requiredBytes) {
            this.gridHeadsBuf.destroy();
            this.gridHeadsBuf = this.device.createBuffer({
                label: 'PBF Grid Heads Buffer',
                size: requiredBytes * 2,
                usage: GPUBufferUsage.STORAGE,
            });
            this.rebuildBindGroup();
        }
    }

    allocateBuffers(capacity) {
        const device = this.device;
        this.capacity = capacity;
        this.computeGridExtents();

        if (this.uniformBuf) this.uniformBuf.destroy();
        if (this.particleBuf) this.particleBuf.destroy();
        if (this.predPosBuf) this.predPosBuf.destroy();
        if (this.oldPosBuf) this.oldPosBuf.destroy();
        if (this.lambdasBuf) this.lambdasBuf.destroy();
        if (this.deltaPosBuf) this.deltaPosBuf.destroy();
        if (this.gridHeadsBuf) this.gridHeadsBuf.destroy();
        if (this.gridNextBuf) this.gridNextBuf.destroy();

        // 1. Uniform Buffer (1024 bytes, >= 976 bytes struct)
        this.uniformBuf = device.createBuffer({
            label: 'PBF Uniform Buffer',
            size: 1024,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // 2. Particle Buffer (pos_type: vec4 + vel_dens: vec4 = 32 bytes per particle)
        this.particleBuf = device.createBuffer({
            label: 'PBF Particle Buffer',
            size: capacity * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        // 3. Predicted Positions Buffer (vec4 = 16 bytes)
        this.predPosBuf = device.createBuffer({
            label: 'PBF Pred Pos Buffer',
            size: capacity * 16,
            usage: GPUBufferUsage.STORAGE,
        });

        // 4. Old Positions Buffer (vec4 = 16 bytes)
        this.oldPosBuf = device.createBuffer({
            label: 'PBF Old Pos Buffer',
            size: capacity * 16,
            usage: GPUBufferUsage.STORAGE,
        });

        // 5. Lambdas Buffer (f32 = 4 bytes)
        this.lambdasBuf = device.createBuffer({
            label: 'PBF Lambdas Buffer',
            size: capacity * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // 6. Delta Positions / Curl Buffer (vec4 = 16 bytes)
        this.deltaPosBuf = device.createBuffer({
            label: 'PBF Delta Pos Buffer',
            size: capacity * 16,
            usage: GPUBufferUsage.STORAGE,
        });

        // 7. Grid Heads (atomic<u32> = 4 bytes per cell)
        this.gridHeadsBuf = device.createBuffer({
            label: 'PBF Grid Heads Buffer',
            size: Math.max(16384, this.totalCells * 4 * 2),
            usage: GPUBufferUsage.STORAGE,
        });

        // 8. Grid Next (u32 = 4 bytes per particle)
        this.gridNextBuf = device.createBuffer({
            label: 'PBF Grid Next Buffer',
            size: capacity * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.rebuildBindGroup();
    }

    setQuality(quality) {
        this.quality = quality;
        if (this.is2D) {
            switch (quality) {
                case 'draft':
                    this.particleRadius = 0.024;
                    this.spacing = 0.040;
                    break;
                case 'medium':
                    this.particleRadius = 0.016;
                    this.spacing = 0.027;
                    break;
                case 'high':
                    this.particleRadius = 0.010;
                    this.spacing = 0.017;
                    break;
                case 'ultra':
                    this.particleRadius = 0.007;
                    this.spacing = 0.012;
                    break;
                case 'extreme':
                    this.particleRadius = 0.005;
                    this.spacing = 0.0085;
                    break;
                default:
                    this.particleRadius = 0.024;
                    this.spacing = 0.040;
            }
        } else {
            switch (quality) {
                case 'draft':
                    this.particleRadius = 0.030;
                    this.spacing = 0.052;
                    break;
                case 'medium':
                    this.particleRadius = 0.022;
                    this.spacing = 0.038;
                    break;
                case 'high':
                    this.particleRadius = 0.015;
                    this.spacing = 0.025;
                    break;
                case 'ultra':
                    this.particleRadius = 0.011;
                    this.spacing = 0.018;
                    break;
                case 'extreme':
                    this.particleRadius = 0.008;
                    this.spacing = 0.013;
                    break;
                default:
                    this.particleRadius = 0.022;
                    this.spacing = 0.038;
            }
        }

        this.h = this.particleRadius * 3.1;
        this.h2 = this.h * this.h;
        this.cellSize = this.h;
        this.updateGridDimensions();
        this.calibrateRestDensity();
    }

    calibrateRestDensity() {
        const s = this.spacing;
        const s2 = s * s;
        const range = Math.ceil(this.h / s);
        const poly6Factor = 315.0 / (64.0 * Math.PI * Math.pow(this.h, 9));

        let sum = 0.0;
        if (this.is2D) {
            for (let dx = -range; dx <= range; dx++) {
                for (let dy = -range; dy <= range; dy++) {
                    const r2 = (dx * dx + dy * dy) * s2;
                    if (r2 < this.h2) {
                        const diff = this.h2 - r2;
                        sum += poly6Factor * diff * diff * diff;
                    }
                }
            }
        } else {
            for (let dx = -range; dx <= range; dx++) {
                for (let dy = -range; dy <= range; dy++) {
                    for (let dz = -range; dz <= range; dz++) {
                        const r2 = (dx * dx + dy * dy + dz * dz) * s2;
                        if (r2 < this.h2) {
                            const diff = this.h2 - r2;
                            sum += poly6Factor * diff * diff * diff;
                        }
                    }
                }
            }
        }

        this.restDensity = Math.max(10.0, sum);
        this.invRestDensity = 1.0 / this.restDensity;
    }

    set2D(is2D) {
        this.is2D = is2D;
        if (is2D) {
            this.bounds.min.z = -0.04;
            this.bounds.max.z = 0.04;
            this.obstaclePos.z = 0.0;
        } else {
            this.bounds.min.z = -0.45;
            this.bounds.max.z = 0.45;
        }
        this.setQuality(this.quality);
    }

    loadPreset(preset) {
        this.obstacleEnabled = (preset === ScenePreset.OBSTACLE);
        const s = this.spacing;
        const r = this.particleRadius;
        const particles = [];

        switch (preset) {
            case ScenePreset.DAM_BREAK: {
                const bMin = this.bounds.min;
                const bMax = this.bounds.max;
                const width = (bMax.x - bMin.x) * 0.42;
                const height = (bMax.y - bMin.y) * 0.75;
                const depth = this.is2D ? 0 : (bMax.z - bMin.z) * 0.85;

                const startX = bMin.x + r * 1.5;
                const endX = startX + width;
                const startY = bMin.y + r * 1.5;
                const endY = startY + height;
                const startZ = this.is2D ? 0 : bMin.z + r * 1.5;
                const endZ = this.is2D ? 0 : startZ + depth;

                for (let y = startY; y <= endY; y += s) {
                    for (let x = startX; x <= endX; x += s) {
                        if (this.is2D) {
                            particles.push(x, y, 0.0, 0.0, 0, 0, 0, this.restDensity);
                        } else {
                            for (let z = startZ; z <= endZ; z += s) {
                                particles.push(x, y, z, 0.0, 0, 0, 0, this.restDensity);
                            }
                        }
                    }
                }
                break;
            }

            case ScenePreset.DOUBLE_DAM: {
                const bMin = this.bounds.min;
                const bMax = this.bounds.max;
                const width = (bMax.x - bMin.x) * 0.28;
                const height = (bMax.y - bMin.y) * 0.70;

                // Left column
                for (let y = bMin.y + r * 1.5; y <= bMin.y + height; y += s) {
                    for (let x = bMin.x + r * 1.5; x <= bMin.x + width; x += s) {
                        if (this.is2D) {
                            particles.push(x, y, 0.0, 0.0, 0, 0, 0, this.restDensity);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                particles.push(x, y, z, 0.0, 0, 0, 0, this.restDensity);
                            }
                        }
                    }
                }

                // Right column
                for (let y = bMin.y + r * 1.5; y <= bMin.y + height; y += s) {
                    for (let x = bMax.x - width; x <= bMax.x - r * 1.5; x += s) {
                        if (this.is2D) {
                            particles.push(x, y, 0.0, 0.0, 0, 0, 0, this.restDensity);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                particles.push(x, y, z, 0.0, 0, 0, 0, this.restDensity);
                            }
                        }
                    }
                }
                break;
            }

            case ScenePreset.SPLASH: {
                const bMin = this.bounds.min;
                const bMax = this.bounds.max;
                const poolH = (bMax.y - bMin.y) * 0.25;

                for (let y = bMin.y + r * 1.5; y <= bMin.y + poolH; y += s) {
                    for (let x = bMin.x + r * 1.5; x <= bMax.x - r * 1.5; x += s) {
                        if (this.is2D) {
                            particles.push(x, y, 0.0, 0.0, 0, 0, 0, this.restDensity);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                particles.push(x, y, z, 0.0, 0, 0, 0, this.restDensity);
                            }
                        }
                    }
                }

                const dropCenter = new Vec3(0, bMin.y + (bMax.y - bMin.y) * 0.72, 0);
                const dropRadius = 0.22;
                for (let y = dropCenter.y - dropRadius; y <= dropCenter.y + dropRadius; y += s) {
                    for (let x = dropCenter.x - dropRadius; x <= dropCenter.x + dropRadius; x += s) {
                        if (this.is2D) {
                            const d2 = (x - dropCenter.x) ** 2 + (y - dropCenter.y) ** 2;
                            if (d2 <= dropRadius * dropRadius) {
                                particles.push(x, y, 0.0, 0.0, 0, -1.8, 0, this.restDensity);
                            }
                        } else {
                            for (let z = dropCenter.z - dropRadius; z <= dropCenter.z + dropRadius; z += s) {
                                const d2 = (x - dropCenter.x) ** 2 + (y - dropCenter.y) ** 2 + (z - dropCenter.z) ** 2;
                                if (d2 <= dropRadius * dropRadius) {
                                    particles.push(x, y, z, 0.0, 0, -1.8, 0, this.restDensity);
                                }
                            }
                        }
                    }
                }
                break;
            }

            case ScenePreset.OBSTACLE: {
                this.obstacleEnabled = true;
                const bMin = this.bounds.min;
                const bMax = this.bounds.max;
                const poolH = (bMax.y - bMin.y) * 0.65;
                const width = (bMax.x - bMin.x) * 0.55;

                for (let y = bMin.y + (bMax.y - bMin.y) * 0.45; y <= bMin.y + poolH; y += s) {
                    for (let x = -width * 0.5; x <= width * 0.5; x += s) {
                        if (this.is2D) {
                            particles.push(x, y, 0.0, 0.0, 0, 0, 0, this.restDensity);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                particles.push(x, y, z, 0.0, 0, 0, 0, this.restDensity);
                            }
                        }
                    }
                }
                break;
            }

            case ScenePreset.VORTEX: {
                const center = new Vec3(0, 0.55, 0);
                const radius = 0.35;
                for (let y = center.y - radius; y <= center.y + radius; y += s) {
                    for (let x = center.x - radius; x <= center.x + radius; x += s) {
                        if (this.is2D) {
                            const d2 = (x - center.x) ** 2 + (y - center.y) ** 2;
                            if (d2 <= radius * radius) {
                                const angle = Math.atan2(y - center.y, x - center.x);
                                const speed = 2.4;
                                particles.push(x, y, 0.0, 0.0, -Math.sin(angle) * speed, Math.cos(angle) * speed, 0, this.restDensity);
                            }
                        } else {
                            for (let z = center.z - radius; z <= center.z + radius; z += s) {
                                const d2 = (x - center.x) ** 2 + (y - center.y) ** 2 + (z - center.z) ** 2;
                                if (d2 <= radius * radius) {
                                    const angle = Math.atan2(z - center.z, x - center.x);
                                    const speed = 2.4;
                                    particles.push(x, y, z, 0.0, -Math.sin(angle) * speed, 0, Math.cos(angle) * speed, this.restDensity);
                                }
                            }
                        }
                    }
                }
                break;
            }

            case ScenePreset.ZERO_G: {
                const center = new Vec3(0, 0.6, 0);
                const blobR = 0.28;
                for (let y = center.y - blobR; y <= center.y + blobR; y += s) {
                    for (let x = center.x - blobR; x <= center.x + blobR; x += s) {
                        if (this.is2D) {
                            const d2 = (x - center.x) ** 2 + (y - center.y) ** 2;
                            if (d2 <= blobR * blobR) {
                                particles.push(x, y, 0.0, 0.0, (Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4, 0, this.restDensity);
                            }
                        } else {
                            for (let z = center.z - blobR; z <= center.z + blobR; z += s) {
                                const d2 = (x - center.x) ** 2 + (y - center.y) ** 2 + (z - center.z) ** 2;
                                if (d2 <= blobR * blobR) {
                                    particles.push(x, y, z, 0.0, (Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4, this.restDensity);
                                }
                            }
                        }
                    }
                }
                break;
            }
        }

        this.particleCount = Math.floor(particles.length / 8);

        if (
            preset === ScenePreset.DAM_BREAK ||
            preset === ScenePreset.DOUBLE_DAM ||
            preset === ScenePreset.OBSTACLE
        ) {
            const amp = s * 0.08;
            const pad = r * 1.02;
            const bMin = this.bounds.min;
            const bMax = this.bounds.max;
            for (let i = 0; i < this.particleCount; i++) {
                const i8 = i * 8;
                let h1 = Math.imul(i ^ 0x9E3779B9, 0x85EBCA6B);
                h1 = Math.imul(h1 ^ (h1 >>> 13), 0xC2B2AE35);
                const h2 = Math.imul(h1 ^ (h1 >>> 16), 0x27D4EB2D);
                const h3 = Math.imul(h2 ^ (h2 >>> 15), 0x165667B1);

                const jx = (((h1 >>> 8) & 0xFFFF) / 32767.5 - 1.0) * amp;
                const jy = (((h2 >>> 8) & 0xFFFF) / 32767.5 - 1.0) * amp;
                const jz = (((h3 >>> 8) & 0xFFFF) / 32767.5 - 1.0) * amp;

                particles[i8] = Math.min(bMax.x - pad, Math.max(bMin.x + pad, particles[i8] + jx));
                particles[i8 + 1] = Math.min(bMax.y - pad, Math.max(bMin.y + pad, particles[i8 + 1] + jy));
                if (!this.is2D) {
                    particles[i8 + 2] = Math.min(bMax.z - pad, Math.max(bMin.z + pad, particles[i8 + 2] + jz));
                }
            }
        }

        if (this.particleCount > this.capacity) {
            this.allocateBuffers(Math.max(this.capacity * 2, this.particleCount + 1024));
        }

        // Upload initial particle state to GPU
        const pData = new Float32Array(particles);
        this.device.queue.writeBuffer(this.particleBuf, 0, pData);
    }

    step(dt, speedMultiplier = 1.0) {
        const n = this.particleCount;
        if (n === 0) return;

        const device = this.device;
        const subDt = (dt * speedMultiplier) / this.substeps;

        // PBF Kernel factors
        const poly6Factor = 315.0 / (64.0 * Math.PI * Math.pow(this.h, 9));
        const spikyFactor = 45.0 / (Math.PI * Math.pow(this.h, 6));
        const wQ = Math.max(1e-12, poly6Factor * Math.pow(this.h2 - Math.pow(0.15 * this.h, 2), 3));

        const dtRef = 1.0 / 60.0;
        const dtRatio = Math.max(1.0, dtRef / Math.max(1e-5, subDt));
        const cfmScale = dtRatio * dtRatio;
        const dtFactor = Math.min(1.0, Math.max(0.0001, subDt / dtRef));
        const relaxationEpsilon = 1200.0 * Math.pow(0.075 / this.h, 2) * cfmScale;
        const maxDisplacement = Math.max(this.particleRadius * 0.45, this.h * 0.25) * dtFactor;
        const stCoeff = (this.surfaceTension * this.spacing * 0.085) / cfmScale;

        const f32 = this.uF32;
        const u32 = this.uU32;

        // 0: bounds_min [x, y, z, cell_size]
        f32[0] = this.bounds.min.x;
        f32[1] = this.bounds.min.y;
        f32[2] = this.bounds.min.z;
        f32[3] = this.cellSize;

        // 4: bounds_max [x, y, z, particle_radius]
        f32[4] = this.bounds.max.x;
        f32[5] = this.bounds.max.y;
        f32[6] = this.bounds.max.z;
        f32[7] = this.particleRadius;

        // 8: gravity [gx, gy, gz, dt]
        f32[8] = this.gravity.x;
        f32[9] = this.gravity.y;
        f32[10] = this.is2D ? 0.0 : this.gravity.z;
        f32[11] = subDt;

        // 12: grid_dim [dimX, dimY, dimZ, totalCells]
        u32[12] = this.gridDim[0];
        u32[13] = this.gridDim[1];
        u32[14] = this.gridDim[2];
        u32[15] = this.totalCells;

        // 16..23: mouse ray interaction
        if (this.mouseRay && this.mouseForceMode !== 0) {
            f32[16] = this.mouseRay.origin.x;
            f32[17] = this.mouseRay.origin.y;
            f32[18] = this.mouseRay.origin.z;
            f32[19] = this.mouseStrength;

            f32[20] = this.mouseRay.direction.x;
            f32[21] = this.mouseRay.direction.y;
            f32[22] = this.mouseRay.direction.z;
            f32[23] = this.mouseForceMode;
        } else {
            f32[19] = 0.0;
            f32[23] = 0.0;
        }

        // 24: collider_pos [ox, oy, oz, radius]
        f32[24] = this.obstaclePos.x;
        f32[25] = this.obstaclePos.y;
        f32[26] = this.is2D ? 0.0 : this.obstaclePos.z;
        f32[27] = this.obstacleRadius;

        // 28: kernel_params [h, h2, inv_rho0, rho0]
        f32[28] = this.h;
        f32[29] = this.h2;
        f32[30] = this.invRestDensity;
        f32[31] = this.restDensity;

        // 32: kernel_factors [poly6, spiky, w_q, tensile_k]
        f32[32] = poly6Factor;
        f32[33] = spikyFactor;
        f32[34] = wQ;
        f32[35] = this.tensileK;

        // 36: solver_params [relax_eps, max_disp, rest, friction]
        f32[36] = relaxationEpsilon;
        f32[37] = maxDisplacement;
        f32[38] = this.restitution;
        f32[39] = this.boundaryFriction;

        // 40: fluid_params [visc, surface_tension_coeff, max_vel, obs_enabled]
        f32[40] = this.viscosity;
        f32[41] = stCoeff;
        f32[42] = this.maxVelocity;
        f32[43] = this.obstacleEnabled ? 1.0 : 0.0;

        // 44: counts [particle_count, solver_iterations, substeps, colliders_flags]
        u32[44] = n;
        u32[45] = this.solverIterations;
        u32[46] = this.substeps;
        let colliderFlag = this.obstacleEnabled ? 1 : 0;
        if (this.is2D) colliderFlag |= 0x80000000;
        u32[47] = colliderFlag >>> 0;

        // 48..239: 16 GpuColliders (each 12 x 32-bit words = 48 bytes)
        if (this.obstacleEnabled) {
            u32[48] = 1; // sphere kind
            u32[49] = 1; // enabled
            u32[50] = 0;
            u32[51] = 0;
            f32[52] = this.obstaclePos.x;
            f32[53] = this.obstaclePos.y;
            f32[54] = this.is2D ? 0.0 : this.obstaclePos.z;
            f32[55] = this.obstacleRadius;
        } else {
            u32[48] = 0;
            u32[49] = 0;
        }

        // 240..243: chamber_info [shape, vorticity, spacing, pad]
        f32[240] = 0.0;
        f32[241] = this.vorticity;
        f32[242] = this.spacing;
        f32[243] = 0.0;

        // Single uniform upload per frame
        device.queue.writeBuffer(this.uniformBuf, 0, this.uBuf);

        const wgParticles = Math.ceil(n / 256);
        const wgCells = Math.ceil(this.totalCells / 256);
        const useVorticity = this.vorticity > 1e-4;

        // Single CommandEncoder + Single ComputePass across all substeps & PBF iterations
        const encoder = device.createCommandEncoder({ label: 'PBF Frame Compute Encoder' });
        const pass = encoder.beginComputePass({ label: 'PBF Batched Substeps Pass' });
        pass.setBindGroup(0, this.bindGroup);

        for (let sub = 0; sub < this.substeps; sub++) {
            // 1. Predict Positions
            pass.setPipeline(this.pipePredict);
            pass.dispatchWorkgroups(wgParticles, 1, 1);

            // 2. Clear Uniform Spatial Grid
            pass.setPipeline(this.pipeClearGrid);
            pass.dispatchWorkgroups(wgCells, 1, 1);

            // 3. Insert Predicted Particles into Grid
            pass.setPipeline(this.pipeInsert);
            pass.dispatchWorkgroups(wgParticles, 1, 1);

            // 4. Incompressibility + Akinci Cohesion Iterations
            for (let iter = 0; iter < this.solverIterations; iter++) {
                pass.setPipeline(this.pipeDensityLambda);
                pass.dispatchWorkgroups(wgParticles, 1, 1);

                pass.setPipeline(this.pipeDeltaPos);
                pass.dispatchWorkgroups(wgParticles, 1, 1);

                pass.setPipeline(this.pipeApplyDelta);
                pass.dispatchWorkgroups(wgParticles, 1, 1);
            }

            // 5. Update Velocity, XSPH Viscosity & Fused Curl
            pass.setPipeline(this.pipeUpdateVelocity);
            pass.dispatchWorkgroups(wgParticles, 1, 1);

            // 6. Flow-Gated Vorticity Confinement
            if (useVorticity) {
                pass.setPipeline(this.pipeVorticity);
                pass.dispatchWorkgroups(wgParticles, 1, 1);
            }
        }

        pass.end();
        device.queue.submit([encoder.finish()]);
    }
}
