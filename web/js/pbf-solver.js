// Position-Based Fluids (PBF) Engine for LiquidRust Web
// Implements Macklin & Müller (2013) + Akinci Cohesion + Flow-Gated Vorticity Confinement
// Zero per-frame heap allocations for smooth 60 FPS execution

import { Vec3, Aabb } from './math.js?v=20261112c';

export const ScenePreset = {
    DAM_BREAK: 'dam_break',
    DOUBLE_DAM: 'double_dam',
    SPLASH: 'splash',
    OBSTACLE: 'obstacle',
    VORTEX: 'vortex',
    ZERO_G: 'zero_g'
};

export class PbfSolver {
    constructor() {
        this.maxParticles = 8000;
        this.particleCount = 0;

        // Pre-allocated TypedArrays for zero-GC simulation stepping
        this.positions = new Float32Array(this.maxParticles * 3);
        this.predicted = new Float32Array(this.maxParticles * 3);
        this.velocities = new Float32Array(this.maxParticles * 3);
        this.densities = new Float32Array(this.maxParticles);
        this.pressures = new Float32Array(this.maxParticles);
        this.lambdas = new Float32Array(this.maxParticles);
        this.deltaPositions = new Float32Array(this.maxParticles * 3);
        this.curls = new Float32Array(this.maxParticles * 3);
        this.curlMags = new Float32Array(this.maxParticles);
        this.gradBuf = new Float32Array(3);

        // Simulation parameters (default: 2D Planar Mode, Draft quality)
        this.is2D = true;
        this.gravity = new Vec3(0, -9.81, 0);
        this.particleRadius = 0.032;
        this.spacing = 0.054;
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
        this.mouseForceMode = 0; // 0: none, 1: attract, -1: repel
        this.mouseStrength = 22.0;

        // Power-of-two Spatial Hashing Grid (fast bitmask lookup)
        this.gridCellSize = this.h;
        this.gridTableSize = 16384;
        this.gridMask = this.gridTableSize - 1;
        this.gridHeads = new Int32Array(this.gridTableSize);
        this.gridNext = new Int32Array(this.maxParticles);

        // Kernel prefactors
        this.poly6Factor = 315.0 / (64.0 * Math.PI * Math.pow(this.h, 9));
        this.spikyFactor = 45.0 / (Math.PI * Math.pow(this.h, 6));
        this.wQ = this.poly6Sq(0.0225 * this.h2);

        this.setQuality('draft');
    }

    setQuality(quality) {
        if (this.is2D) {
            switch (quality) {
                case 'draft':
                    this.particleRadius = 0.032;
                    this.spacing = 0.054;
                    break;
                case 'medium':
                    this.particleRadius = 0.024;
                    this.spacing = 0.042;
                    break;
                case 'high':
                    this.particleRadius = 0.016;
                    this.spacing = 0.028;
                    break;
                case 'ultra':
                case 'extreme':
                    this.particleRadius = 0.011;
                    this.spacing = 0.019;
                    break;
                default:
                    this.particleRadius = 0.024;
                    this.spacing = 0.042;
            }
        } else {
            switch (quality) {
                case 'draft':
                    this.particleRadius = 0.034;
                    this.spacing = 0.058;
                    break;
                case 'medium':
                    this.particleRadius = 0.025;
                    this.spacing = 0.043;
                    break;
                case 'high':
                    this.particleRadius = 0.018;
                    this.spacing = 0.031;
                    break;
                case 'ultra':
                case 'extreme':
                    this.particleRadius = 0.014;
                    this.spacing = 0.024;
                    break;
                default:
                    this.particleRadius = 0.025;
                    this.spacing = 0.043;
            }
        }

        this.h = this.particleRadius * 3.1;
        this.h2 = this.h * this.h;
        this.gridCellSize = this.h;
        this.poly6Factor = 315.0 / (64.0 * Math.PI * Math.pow(this.h, 9));
        this.spikyFactor = 45.0 / (Math.PI * Math.pow(this.h, 6));
        this.wQ = Math.max(1e-12, this.poly6Sq(0.0225 * this.h2));
        this.calibrateRestDensity();
    }

    calibrateRestDensity() {
        let sum = 0.0;
        const s = this.spacing;
        const s2 = s * s;
        const range = Math.ceil(this.h / s);

        if (this.is2D) {
            for (let dx = -range; dx <= range; dx++) {
                for (let dy = -range; dy <= range; dy++) {
                    const r2 = (dx * dx + dy * dy) * s2;
                    if (r2 < this.h2) {
                        sum += this.poly6Sq(r2);
                    }
                }
            }
        } else {
            for (let dx = -range; dx <= range; dx++) {
                for (let dy = -range; dy <= range; dy++) {
                    for (let dz = -range; dz <= range; dz++) {
                        const r2 = (dx * dx + dy * dy + dz * dz) * s2;
                        if (r2 < this.h2) {
                            sum += this.poly6Sq(r2);
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
        this.calibrateRestDensity();
    }

    clearParticles() {
        this.particleCount = 0;
    }

    addParticle(x, y, z, vx = 0, vy = 0, vz = 0) {
        if (this.particleCount >= this.maxParticles) return;
        const idx = this.particleCount;
        const p3 = idx * 3;

        this.positions[p3] = x;
        this.positions[p3 + 1] = y;
        this.positions[p3 + 2] = this.is2D ? 0.0 : z;

        this.velocities[p3] = vx;
        this.velocities[p3 + 1] = vy;
        this.velocities[p3 + 2] = this.is2D ? 0.0 : vz;

        this.densities[idx] = this.restDensity;
        this.pressures[idx] = 0.0;

        this.particleCount++;
    }

    loadPreset(preset) {
        this.clearParticles();
        this.obstacleEnabled = (preset === ScenePreset.OBSTACLE);
        const s = this.spacing;
        const r = this.particleRadius;

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
                            this.addParticle(x, y, 0);
                        } else {
                            for (let z = startZ; z <= endZ; z += s) {
                                this.addParticle(x, y, z);
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

                for (let y = bMin.y + r * 1.5; y <= bMin.y + height; y += s) {
                    for (let x = bMin.x + r * 1.5; x <= bMin.x + width; x += s) {
                        if (this.is2D) {
                            this.addParticle(x, y, 0);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                this.addParticle(x, y, z);
                            }
                        }
                    }
                }

                for (let y = bMin.y + r * 1.5; y <= bMin.y + height; y += s) {
                    for (let x = bMax.x - width; x <= bMax.x - r * 1.5; x += s) {
                        if (this.is2D) {
                            this.addParticle(x, y, 0);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                this.addParticle(x, y, z);
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
                            this.addParticle(x, y, 0);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                this.addParticle(x, y, z);
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
                                this.addParticle(x, y, 0, 0, -2.5, 0);
                            }
                        } else {
                            for (let z = dropCenter.z - dropRadius; z <= dropCenter.z + dropRadius; z += s) {
                                const d2 = (x - dropCenter.x) ** 2 + (y - dropCenter.y) ** 2 + (z - dropCenter.z) ** 2;
                                if (d2 <= dropRadius * dropRadius) {
                                    this.addParticle(x, y, z, 0, -2.5, 0);
                                }
                            }
                        }
                    }
                }
                break;
            }

            case ScenePreset.OBSTACLE: {
                this.obstacleEnabled = true;
                this.obstaclePos.set(0.0, 0.35, 0.0);
                const bMin = this.bounds.min;
                const bMax = this.bounds.max;
                const colWidth = (bMax.x - bMin.x) * 0.40;
                for (let y = 0.55; y <= 1.15; y += s) {
                    for (let x = -colWidth * 0.5; x <= colWidth * 0.5; x += s) {
                        if (this.is2D) {
                            this.addParticle(x, y, 0);
                        } else {
                            for (let z = -0.3; z <= 0.3; z += s) {
                                this.addParticle(x, y, z);
                            }
                        }
                    }
                }
                break;
            }

            case ScenePreset.VORTEX: {
                const bMin = this.bounds.min;
                const bMax = this.bounds.max;
                const poolH = (bMax.y - bMin.y) * 0.45;
                for (let y = bMin.y + r * 1.5; y <= bMin.y + poolH; y += s) {
                    for (let x = bMin.x + r * 1.5; x <= bMax.x - r * 1.5; x += s) {
                        if (this.is2D) {
                            const angle = Math.atan2(y - 0.3, x);
                            const speed = 1.9;
                            this.addParticle(x, y, 0, -Math.sin(angle) * speed, Math.cos(angle) * speed, 0);
                        } else {
                            for (let z = bMin.z + r * 1.5; z <= bMax.z - r * 1.5; z += s) {
                                const angle = Math.atan2(z, x);
                                const speed = 2.2;
                                this.addParticle(x, y, z, -Math.sin(angle) * speed, 0, Math.cos(angle) * speed);
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
                                this.addParticle(x, y, 0, (Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4, 0);
                            }
                        } else {
                            for (let z = center.z - blobR; z <= center.z + blobR; z += s) {
                                const d2 = (x - center.x) ** 2 + (y - center.y) ** 2 + (z - center.z) ** 2;
                                if (d2 <= blobR * blobR) {
                                    this.addParticle(x, y, z, (Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4);
                                }
                            }
                        }
                    }
                }
                break;
            }
        }
    }

    hashCoords(cx, cy, cz) {
        return ((Math.imul(cx, 73856093) ^ Math.imul(cy, 19349663) ^ Math.imul(cz, 83492791)) & this.gridMask);
    }

    buildSpatialHash() {
        this.gridHeads.fill(-1);
        const invH = 1.0 / this.gridCellSize;
        const n = this.particleCount;
        const is2D = this.is2D;

        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            const cx = Math.floor(this.predicted[i3] * invH);
            const cy = Math.floor(this.predicted[i3 + 1] * invH);
            const cz = is2D ? 0 : Math.floor(this.predicted[i3 + 2] * invH);

            const hash = this.hashCoords(cx, cy, cz);
            this.gridNext[i] = this.gridHeads[hash];
            this.gridHeads[hash] = i;
        }
    }

    poly6Sq(r2) {
        if (r2 < this.h2 && r2 >= 0.0) {
            const diff = this.h2 - r2;
            return this.poly6Factor * diff * diff * diff;
        }
        return 0.0;
    }

    poly6(r) {
        return this.poly6Sq(r * r);
    }

    spikyGradient(rx, ry, rz, r, out) {
        if (r < this.h && r > 1e-6) {
            const diff = this.h - r;
            const factor = (-this.spikyFactor * diff * diff) / r;
            out[0] = rx * factor;
            out[1] = ry * factor;
            out[2] = rz * factor;
            return true;
        }
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        return false;
    }

    step(dt, speedMultiplier = 1.0) {
        if (this.particleCount === 0) return;

        const effectiveDt = (dt * speedMultiplier) / this.substeps;
        for (let sub = 0; sub < this.substeps; sub++) {
            this.substep(effectiveDt);
        }
    }

    substep(dt) {
        const n = this.particleCount;
        const invH = 1.0 / this.gridCellSize;
        const is2D = this.is2D;

        // 1. External forces & predicted positions
        const gx = this.gravity.x * dt;
        const gy = this.gravity.y * dt;
        const gz = is2D ? 0 : this.gravity.z * dt;
        const hasMouse = this.mouseRay && this.mouseForceMode !== 0;

        for (let i = 0; i < n; i++) {
            const i3 = i * 3;

            this.velocities[i3] += gx;
            this.velocities[i3 + 1] += gy;
            this.velocities[i3 + 2] += gz;

            if (hasMouse) {
                const px = this.positions[i3];
                const py = this.positions[i3 + 1];
                const pz = this.positions[i3 + 2];

                const ox = this.mouseRay.origin.x;
                const oy = this.mouseRay.origin.y;
                const oz = this.mouseRay.origin.z;
                const dx = this.mouseRay.direction.x;
                const dy = this.mouseRay.direction.y;
                const dz = this.mouseRay.direction.z;

                const vx = px - ox;
                const vy = py - oy;
                const vz = pz - oz;
                const proj = vx * dx + vy * dy + vz * dz;

                if (proj > 0) {
                    const rx = ox + dx * proj - px;
                    const ry = oy + dy * proj - py;
                    const rz = oz + dz * proj - pz;
                    const distSq = rx * rx + ry * ry + rz * rz;

                    if (distSq < 0.2025) {
                        const dist = Math.sqrt(distSq) + 0.001;
                        const strength = this.mouseStrength * (1.0 - dist / 0.45) * dt * this.mouseForceMode;
                        this.velocities[i3] += (rx / dist) * strength;
                        this.velocities[i3 + 1] += (ry / dist) * strength;
                        if (!is2D) this.velocities[i3 + 2] += (rz / dist) * strength;
                    }
                }
            }

            this.predicted[i3] = this.positions[i3] + this.velocities[i3] * dt;
            this.predicted[i3 + 1] = this.positions[i3 + 1] + this.velocities[i3 + 1] * dt;
            this.predicted[i3 + 2] = is2D ? 0.0 : this.positions[i3 + 2] + this.velocities[i3 + 2] * dt;
        }

        // 2. Spatial Hash
        this.buildSpatialHash();

        // 3. Iterative Constraint Solver + Akinci Free-Surface Cohesion
        const gradBuf = this.gradBuf;
        const dtRef = 1.0 / 60.0;
        const dtRatio = Math.max(1.0, dtRef / Math.max(1e-5, dt));
        const cfmScale = dtRatio * dtRatio;
        const dtFactor = Math.min(1.0, Math.max(0.0001, dt / dtRef));
        const relaxationEpsilon = 1200.0 * Math.pow(0.075 / this.h, 2) * cfmScale;
        const maxDisplacement = Math.max(this.particleRadius * 0.45, this.h * 0.25) * dtFactor;
        const h = this.h;
        const h2 = this.h2;
        const invRho0 = this.invRestDensity;
        const rho0 = this.restDensity;
        const wQ = this.wQ;
        const selfDensity = this.poly6Sq(0.0);
        const stCoeff = (this.surfaceTension * this.spacing * 0.085) / cfmScale;
        const zMin = is2D ? 0 : -1;
        const zMax = is2D ? 0 : 1;

        for (let iter = 0; iter < this.solverIterations; iter++) {
            for (let i = 0; i < n; i++) {
                const i3 = i * 3;
                const px = this.predicted[i3];
                const py = this.predicted[i3 + 1];
                const pz = this.predicted[i3 + 2];

                const cx = Math.floor(px * invH);
                const cy = Math.floor(py * invH);
                const cz = is2D ? 0 : Math.floor(pz * invH);

                let density = selfDensity;
                let gradIx = 0.0, gradIy = 0.0, gradIz = 0.0;
                let sumGradSq = 0.0;

                for (let dz = zMin; dz <= zMax; dz++) {
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            let j = this.gridHeads[this.hashCoords(cx + dx, cy + dy, cz + dz)];
                            while (j !== -1) {
                                if (i !== j) {
                                    const j3 = j * 3;
                                    const rx = px - this.predicted[j3];
                                    const ry = py - this.predicted[j3 + 1];
                                    const rz = is2D ? 0 : pz - this.predicted[j3 + 2];
                                    const r2 = rx * rx + ry * ry + rz * rz;

                                    if (r2 < h2 && r2 > 1e-12) {
                                        density += this.poly6Sq(r2);
                                        const r = Math.sqrt(r2);
                                        if (this.spikyGradient(rx, ry, rz, r, gradBuf)) {
                                            const gvx = gradBuf[0] * invRho0;
                                            const gvy = gradBuf[1] * invRho0;
                                            const gvz = gradBuf[2] * invRho0;
                                            gradIx += gvx;
                                            gradIy += gvy;
                                            gradIz += gvz;
                                            sumGradSq += gvx * gvx + gvy * gvy + gvz * gvz;
                                        }
                                    }
                                }
                                j = this.gridNext[j];
                            }
                        }
                    }
                }

                sumGradSq += gradIx * gradIx + gradIy * gradIy + gradIz * gradIz;
                this.densities[i] = density;

                const constraint = Math.max(0.0, density * invRho0 - 1.0);
                this.pressures[i] = constraint * rho0;
                this.lambdas[i] = -constraint / (sumGradSq + relaxationEpsilon);
            }

            for (let i = 0; i < n; i++) {
                const i3 = i * 3;
                const px = this.predicted[i3];
                const py = this.predicted[i3 + 1];
                const pz = this.predicted[i3 + 2];
                const lambdaI = this.lambdas[i];
                const rhoI = this.densities[i];

                const cx = Math.floor(px * invH);
                const cy = Math.floor(py * invH);
                const cz = is2D ? 0 : Math.floor(pz * invH);

                let delX = 0.0, delY = 0.0, delZ = 0.0;

                for (let dz = zMin; dz <= zMax; dz++) {
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            let j = this.gridHeads[this.hashCoords(cx + dx, cy + dy, cz + dz)];
                            while (j !== -1) {
                                if (i !== j) {
                                    const j3 = j * 3;
                                    const rx = px - this.predicted[j3];
                                    const ry = py - this.predicted[j3 + 1];
                                    const rz = is2D ? 0 : pz - this.predicted[j3 + 2];
                                    const r2 = rx * rx + ry * ry + rz * rz;

                                    if (r2 < h2 && r2 > 1e-12) {
                                        const r = Math.sqrt(r2);
                                        if (this.spikyGradient(rx, ry, rz, r, gradBuf)) {
                                            const lambdaSum = lambdaI + this.lambdas[j];
                                            let sCorr = 0.0;
                                            if (lambdaSum < -1e-6) {
                                                const ratio = Math.min(this.poly6Sq(r2) / wQ, 1.5);
                                                const r4 = ratio * ratio * ratio * ratio;
                                                sCorr = (-this.tensileK * r4 * h2) / cfmScale;
                                            }
                                            const coeff = (lambdaSum + sCorr) * invRho0;
                                            delX += gradBuf[0] * coeff;
                                            delY += gradBuf[1] * coeff;
                                            delZ += gradBuf[2] * coeff;
                                        }

                                        // Akinci bell-spline free-surface cohesion
                                        if (stCoeff > 0.0 && r > this.spacing * 0.45) {
                                            const q = r / h;
                                            const spline = q * q * q * (1.0 - q) * (1.0 - q) * (1.0 - q) * 64.0;
                                            const kij = (2.0 * rho0) / Math.max(rhoI + this.densities[j], rho0 * 0.5);
                                            const pull = (stCoeff * kij * spline) / r;
                                            delX -= rx * pull;
                                            delY -= ry * pull;
                                            delZ -= rz * pull;
                                        }
                                    }
                                }
                                j = this.gridNext[j];
                            }
                        }
                    }
                }

                const deltaLen = Math.hypot(delX, delY, delZ);
                if (deltaLen > maxDisplacement) {
                    const scale = maxDisplacement / deltaLen;
                    delX *= scale;
                    delY *= scale;
                    delZ *= scale;
                }

                this.deltaPositions[i3] = delX;
                this.deltaPositions[i3 + 1] = delY;
                this.deltaPositions[i3 + 2] = is2D ? 0 : delZ;
            }

            const omega = 0.65;
            for (let i = 0; i < n; i++) {
                const i3 = i * 3;
                this.predicted[i3] += this.deltaPositions[i3] * omega;
                this.predicted[i3 + 1] += this.deltaPositions[i3 + 1] * omega;
                this.predicted[i3 + 2] += this.deltaPositions[i3 + 2] * omega;
                this.resolveCollisions(i3);
            }
        }

        // 4. Update velocities & inelastic boundary damping
        const invDt = 1.0 / dt;
        const r = this.particleRadius;
        const bMin = this.bounds.min;
        const bMax = this.bounds.max;
        const rest = this.restitution;
        const fric = this.boundaryFriction;
        const restThreshold = 0.10;

        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            let vx = (this.predicted[i3] - this.positions[i3]) * invDt;
            let vy = (this.predicted[i3 + 1] - this.positions[i3 + 1]) * invDt;
            let vz = is2D ? 0 : (this.predicted[i3 + 2] - this.positions[i3 + 2]) * invDt;

            if (this.predicted[i3] <= bMin.x + r + 1e-4) {
                if (vx < 0) vx = (vx < -restThreshold) ? -vx * rest : 0;
                vy *= fric;
                vz *= fric;
            } else if (this.predicted[i3] >= bMax.x - r - 1e-4) {
                if (vx > 0) vx = (vx > restThreshold) ? -vx * rest : 0;
                vy *= fric;
                vz *= fric;
            }

            if (this.predicted[i3 + 1] <= bMin.y + r + 1e-4) {
                if (vy < 0) {
                    vy = (vy < -restThreshold) ? -vy * rest : 0;
                } else if (vy > 0 && vy < 0.20 && this.predicted[i3 + 1] <= bMin.y + r + 0.001) {
                    vy *= 0.1;
                }
                vx *= fric;
                vz *= fric;
            } else if (this.predicted[i3 + 1] >= bMax.y - r - 1e-4) {
                if (vy > 0) vy = (vy > restThreshold) ? -vy * rest : 0;
                vx *= fric;
                vz *= fric;
            }

            if (!is2D) {
                if (this.predicted[i3 + 2] <= bMin.z + r + 1e-4) {
                    if (vz < 0) vz = (vz < -restThreshold) ? -vz * rest : 0;
                    vx *= fric;
                    vy *= fric;
                } else if (this.predicted[i3 + 2] >= bMax.z - r - 1e-4) {
                    if (vz > 0) vz = (vz > restThreshold) ? -vz * rest : 0;
                    vx *= fric;
                    vy *= fric;
                }
            }

            const speed = Math.hypot(vx, vy, vz);
            if (speed > this.maxVelocity) {
                const s = this.maxVelocity / speed;
                vx *= s;
                vy *= s;
                vz *= s;
            }

            this.velocities[i3] = vx;
            this.velocities[i3 + 1] = vy;
            this.velocities[i3 + 2] = vz;
        }

        // 5. Fused XSPH Viscosity + Velocity Curl Evaluation
        const doVort = this.vorticity > 1e-4;
        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            const px = this.predicted[i3];
            const py = this.predicted[i3 + 1];
            const pz = this.predicted[i3 + 2];

            const vx = this.velocities[i3];
            const vy = this.velocities[i3 + 1];
            const vz = this.velocities[i3 + 2];

            const cx = Math.floor(px * invH);
            const cy = Math.floor(py * invH);
            const cz = is2D ? 0 : Math.floor(pz * invH);

            let viscX = 0, viscY = 0, viscZ = 0;
            let curlX = 0, curlY = 0, curlZ = 0;

            for (let dz = zMin; dz <= zMax; dz++) {
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        let j = this.gridHeads[this.hashCoords(cx + dx, cy + dy, cz + dz)];
                        while (j !== -1) {
                            if (i !== j) {
                                const j3 = j * 3;
                                const rx = px - this.predicted[j3];
                                const ry = py - this.predicted[j3 + 1];
                                const rz = is2D ? 0 : pz - this.predicted[j3 + 2];
                                const r2 = rx * rx + ry * ry + rz * rz;

                                if (r2 < h2 && r2 > 1e-12) {
                                    const w = this.poly6Sq(r2);
                                    const dvx = this.velocities[j3] - vx;
                                    const dvy = this.velocities[j3 + 1] - vy;
                                    const dvz = this.velocities[j3 + 2] - vz;
                                    viscX += dvx * w;
                                    viscY += dvy * w;
                                    viscZ += dvz * w;

                                    if (doVort) {
                                        const dist = Math.sqrt(r2);
                                        if (this.spikyGradient(rx, ry, rz, dist, gradBuf)) {
                                            // v_ij x grad_W
                                            curlX += (dvy * gradBuf[2] - dvz * gradBuf[1]) * invRho0;
                                            curlY += (dvz * gradBuf[0] - dvx * gradBuf[2]) * invRho0;
                                            curlZ += (dvx * gradBuf[1] - dvy * gradBuf[0]) * invRho0;
                                        }
                                    }
                                }
                            }
                            j = this.gridNext[j];
                        }
                    }
                }
            }

            // Store XSPH-updated velocity into deltaPositions temporarily so neighbor reads stay symmetric
            this.deltaPositions[i3] = vx + viscX * this.viscosity * invRho0;
            this.deltaPositions[i3 + 1] = vy + viscY * this.viscosity * invRho0;
            this.deltaPositions[i3 + 2] = is2D ? 0 : vz + viscZ * this.viscosity * invRho0;

            if (doVort) {
                this.curls[i3] = curlX;
                this.curls[i3 + 1] = curlY;
                this.curls[i3 + 2] = curlZ;
                this.curlMags[i] = Math.hypot(curlX, curlY, curlZ);
            }
        }

        // 6. Flow-Gated Vorticity Confinement & Position Commit
        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            let vx = this.deltaPositions[i3];
            let vy = this.deltaPositions[i3 + 1];
            let vz = this.deltaPositions[i3 + 2];

            if (doVort && this.curlMags[i] > 1e-3) {
                const speed = Math.hypot(vx, vy, vz);
                // Gate at rest so settled pools remain mirror-calm
                if (speed > 0.08) {
                    const px = this.predicted[i3];
                    const py = this.predicted[i3 + 1];
                    const pz = this.predicted[i3 + 2];
                    const cx = Math.floor(px * invH);
                    const cy = Math.floor(py * invH);
                    const cz = is2D ? 0 : Math.floor(pz * invH);

                    const wMagI = this.curlMags[i];
                    let etaX = 0, etaY = 0, etaZ = 0;

                    for (let dz = zMin; dz <= zMax; dz++) {
                        for (let dy = -1; dy <= 1; dy++) {
                            for (let dx = -1; dx <= 1; dx++) {
                                let j = this.gridHeads[this.hashCoords(cx + dx, cy + dy, cz + dz)];
                                while (j !== -1) {
                                    if (i !== j) {
                                        const j3 = j * 3;
                                        const rx = px - this.predicted[j3];
                                        const ry = py - this.predicted[j3 + 1];
                                        const rz = is2D ? 0 : pz - this.predicted[j3 + 2];
                                        const r2 = rx * rx + ry * ry + rz * rz;
                                        if (r2 < h2 && r2 > 1e-12) {
                                            const dist = Math.sqrt(r2);
                                            if (this.spikyGradient(rx, ry, rz, dist, gradBuf)) {
                                                const dw = (this.curlMags[j] - wMagI) * invRho0;
                                                etaX += gradBuf[0] * dw;
                                                etaY += gradBuf[1] * dw;
                                                etaZ += gradBuf[2] * dw;
                                            }
                                        }
                                    }
                                    j = this.gridNext[j];
                                }
                            }
                        }
                    }

                    const etaLen = Math.hypot(etaX, etaY, etaZ);
                    if (etaLen > 1e-5) {
                        const nx = etaX / etaLen;
                        const ny = etaY / etaLen;
                        const nz = etaZ / etaLen;
                        const wx = this.curls[i3];
                        const wy = this.curls[i3 + 1];
                        const wz = this.curls[i3 + 2];

                        const flowGate = Math.min(1.0, (speed - 0.08) / 0.35);
                        const gain = this.vorticity * this.spacing * 2.2 * flowGate * dt;
                        let dvx = (ny * wz - nz * wy) * gain;
                        let dvy = (nz * wx - nx * wz) * gain;
                        let dvz = is2D ? 0 : (nx * wy - ny * wx) * gain;

                        const dvLen = Math.hypot(dvx, dvy, dvz);
                        const maxDv = speed * 0.22;
                        if (dvLen > maxDv && dvLen > 1e-6) {
                            const s = maxDv / dvLen;
                            dvx *= s;
                            dvy *= s;
                            dvz *= s;
                        }
                        vx += dvx;
                        vy += dvy;
                        vz += dvz;
                    }
                }
            }

            this.velocities[i3] = vx;
            this.velocities[i3 + 1] = vy;
            this.velocities[i3 + 2] = is2D ? 0 : vz;

            this.positions[i3] = this.predicted[i3];
            this.positions[i3 + 1] = this.predicted[i3 + 1];
            this.positions[i3 + 2] = this.predicted[i3 + 2];
        }
    }

    resolveCollisions(i3) {
        const r = this.particleRadius;
        const bMin = this.bounds.min;
        const bMax = this.bounds.max;

        if (this.predicted[i3] < bMin.x + r) {
            this.predicted[i3] = bMin.x + r;
        } else if (this.predicted[i3] > bMax.x - r) {
            this.predicted[i3] = bMax.x - r;
        }

        if (this.predicted[i3 + 1] < bMin.y + r) {
            this.predicted[i3 + 1] = bMin.y + r;
        } else if (this.predicted[i3 + 1] > bMax.y - r) {
            this.predicted[i3 + 1] = bMax.y - r;
        }

        if (this.is2D) {
            this.predicted[i3 + 2] = 0.0;
        } else {
            if (this.predicted[i3 + 2] < bMin.z + r) {
                this.predicted[i3 + 2] = bMin.z + r;
            } else if (this.predicted[i3 + 2] > bMax.z - r) {
                this.predicted[i3 + 2] = bMax.z - r;
            }
        }

        if (this.obstacleEnabled) {
            const ox = this.obstaclePos.x;
            const oy = this.obstaclePos.y;
            const oz = this.is2D ? 0.0 : this.obstaclePos.z;
            const totalR = this.obstacleRadius + r;

            const dx = this.predicted[i3] - ox;
            const dy = this.predicted[i3 + 1] - oy;
            const dz = this.is2D ? 0.0 : this.predicted[i3 + 2] - oz;
            const d2 = dx * dx + dy * dy + dz * dz;

            if (d2 < totalR * totalR && d2 > 1e-7) {
                const dist = Math.sqrt(d2);
                const push = totalR - dist;
                this.predicted[i3] += (dx / dist) * push;
                this.predicted[i3 + 1] += (dy / dist) * push;
                if (!this.is2D) this.predicted[i3 + 2] += (dz / dist) * push;
            }
        }
    }
}
