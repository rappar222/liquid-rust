package com.liquidrust.mobile;

import android.content.res.AssetManager;
import android.opengl.GLES30;
import android.opengl.GLES31;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Hardware-accelerated Position-Based Fluids (PBF) GPU Compute Solver for OpenGL ES 3.1 / 3.2.
 *
 * Mobile GPU Optimizations:
 * - Workgroup size = 64 (optimal occupancy across Qualcomm Adreno fibers & ARM Mali warps).
 * - Strictly <= 4 SSBO bindings per shader (100% compliant with minimum GLES 3.1 spec).
 * - Combined grid_heads + grid_next buffer, read via L1-cached readonly non-atomic loads after insertion.
 * - Fused Jacobi solve + Akinci cohesion + chamber/obstacle collision pass.
 * - Zero-copy pointer swap at end of substep + direct GL_ARRAY_BUFFER instanced vertex fetch.
 * - Zero Java heap allocations during per-frame stepping.
 */
public final class GpuPbfSolver {

    private static final int WORKGROUP_SIZE = 64;
    private static final int UBO_BYTES = 208; // 13 * vec4/ivec4

    private int progPredict;
    private int progClearGrid;
    private int progInsertGrid;
    private int progDensity;
    private int progSolveCollide;
    private int progVelocityXsph;
    private int progCurl;
    private int progVorticity;

    private int uboSim;
    private int ssboPos;
    private int ssboVel;
    private int ssboPredA;
    private int ssboPredB;
    private int ssboGrid;

    private int allocatedParticles = 0;
    private int allocatedGridWords = 0;

    // Pre-allocated native UBO staging buffer (zero GC per frame)
    private final ByteBuffer uboByteBuf;
    private final FloatBuffer uboFloatView;
    private final IntBuffer uboIntView;

    // Simulation domain & kernel parameters
    public float minX = -0.90f, minY = 0.0f, minZ = -0.05f;
    public float maxX =  0.90f, maxY = 1.15f, maxZ =  0.05f;

    private int particleCount = 0;
    private boolean is2D = true;
    private float spacing = 0.028f;
    private float particleRadius = 0.015f;
    private float h = 0.056f;
    private float h2 = h * h;
    private float poly6Factor;
    private float spikyFactor;
    private float wQ;
    private float selfDensity;
    private float restDensity = 1000.0f;
    private float invRestDensity = 0.001f;
    private int gridNx = 32, gridNy = 22, gridNz = 1, totalCells = 704;
    private int solverIterations = 3;
    private int substeps = 2;

    // Staging arrays used only during scene load / preset reset
    private float[] initPos = new float[4 * 4096];
    private float[] initVel = new float[4 * 4096];

    public GpuPbfSolver(AssetManager assets) {
        uboByteBuf = ByteBuffer.allocateDirect(UBO_BYTES).order(ByteOrder.nativeOrder());
        uboFloatView = uboByteBuf.asFloatBuffer();
        uboIntView = uboByteBuf.asIntBuffer();

        progPredict = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_predict.comp");
        progClearGrid = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_clear_grid.comp");
        progInsertGrid = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_insert_grid.comp");
        progDensity = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_density.comp");
        progSolveCollide = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_solve_collide.comp");
        progVelocityXsph = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_velocity_xsph.comp");
        progCurl = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_curl.comp");
        progVorticity = ShaderUtils.buildComputeProgram(assets, "shaders/pbf_vorticity.comp");

        bindUniformBlock(progPredict);
        bindUniformBlock(progClearGrid);
        bindUniformBlock(progInsertGrid);
        bindUniformBlock(progDensity);
        bindUniformBlock(progSolveCollide);
        bindUniformBlock(progVelocityXsph);
        bindUniformBlock(progCurl);
        bindUniformBlock(progVorticity);

        int[] bufs = new int[6];
        GLES30.glGenBuffers(6, bufs, 0);
        uboSim = bufs[0];
        ssboPos = bufs[1];
        ssboVel = bufs[2];
        ssboPredA = bufs[3];
        ssboPredB = bufs[4];
        ssboGrid = bufs[5];

        GLES30.glBindBuffer(GLES30.GL_UNIFORM_BUFFER, uboSim);
        GLES30.glBufferData(GLES30.GL_UNIFORM_BUFFER, UBO_BYTES, null, GLES30.GL_DYNAMIC_DRAW);
        GLES30.glBindBuffer(GLES30.GL_UNIFORM_BUFFER, 0);
    }

    private static void bindUniformBlock(int prog) {
        int idx = GLES30.glGetUniformBlockIndex(prog, "SimUniforms");
        if (idx != GLES30.GL_INVALID_INDEX) {
            GLES30.glUniformBlockBinding(prog, idx, 0);
        }
    }

    public int getPosBufferId() { return ssboPos; }
    public int getVelBufferId() { return ssboVel; }
    public int getParticleCount() { return particleCount; }
    public float getParticleRadius() { return particleRadius; }
    public float getInvRestDensity() { return invRestDensity; }
    public float getSmoothingLength() { return h; }
    public boolean is2D() { return is2D; }

    public void loadScene(SimulationConfig cfg) {
        this.is2D = cfg.is2D;
        int q = cfg.quality;

        if (is2D) {
            if (cfg.isPortrait) {
                minX = -0.60f; maxX = 0.60f;
                minY =  0.00f; maxY = 1.86f;
            } else {
                minX = -0.96f; maxX = 0.96f;
                minY =  0.00f; maxY = 1.12f;
            }
            minZ = -0.05f; maxZ = 0.05f;
            switch (q) {
                case SimulationConfig.QUALITY_DRAFT:   spacing = 0.0280f; solverIterations = 3; break;
                case SimulationConfig.QUALITY_MEDIUM:  spacing = 0.0170f; solverIterations = 3; break;
                case SimulationConfig.QUALITY_HIGH:    spacing = 0.0120f; solverIterations = 3; break;
                case SimulationConfig.QUALITY_ULTRA:   spacing = 0.0085f; solverIterations = 4; break;
                default:                               spacing = 0.0062f; solverIterations = 4; break;
            }
            h = spacing * 2.05f;
        } else {
            minX = -0.65f; maxX = 0.65f;
            minY =  0.00f; maxY = 1.05f;
            minZ = -0.55f; maxZ = 0.55f;
            switch (q) {
                case SimulationConfig.QUALITY_DRAFT:   spacing = 0.0520f; solverIterations = 3; break;
                case SimulationConfig.QUALITY_MEDIUM:  spacing = 0.0360f; solverIterations = 3; break;
                case SimulationConfig.QUALITY_HIGH:    spacing = 0.0280f; solverIterations = 3; break;
                case SimulationConfig.QUALITY_ULTRA:   spacing = 0.0225f; solverIterations = 3; break;
                default:                               spacing = 0.0185f; solverIterations = 4; break;
            }
            h = spacing * 1.95f;
        }

        particleRadius = spacing * 0.52f;
        h2 = h * h;
        poly6Factor = (float) (315.0 / (64.0 * Math.PI * Math.pow(h, 9)));
        spikyFactor = (float) (45.0 / (Math.PI * Math.pow(h, 6)));
        selfDensity = poly6Factor * h2 * h2 * h2;
        float qDist = 0.3f * h;
        float diffQ = h2 - qDist * qDist;
        wQ = poly6Factor * diffQ * diffQ * diffQ;

        restDensity = computeLatticeRestDensity(is2D, h, h2, poly6Factor, spacing);
        invRestDensity = 1.0f / restDensity;

        gridNx = Math.max(4, (int) Math.ceil((maxX - minX) / h) + 1);
        gridNy = Math.max(4, (int) Math.ceil((maxY - minY) / h) + 1);
        gridNz = is2D ? 1 : Math.max(4, (int) Math.ceil((maxZ - minZ) / h) + 1);
        totalCells = gridNx * gridNy * gridNz;

        particleCount = 0;
        int preset = cfg.preset;
        cfg.obstacleEnabled = (preset == SimulationConfig.PRESET_OBSTACLE);
        if (preset == SimulationConfig.PRESET_ZERO_G) {
            cfg.gravityMag = 0.0f;
        } else if (cfg.gravityMag < 0.1f) {
            cfg.gravityMag = 9.81f;
        }

        spawnPresetParticles(preset);
        uploadInitialBuffers();
    }

    private static float computeLatticeRestDensity(boolean is2D, float h, float h2, float poly6, float s) {
        int reach = (int) Math.ceil(h / s) + 1;
        float sum = 0.0f;
        int zRange = is2D ? 0 : reach;
        for (int dz = -zRange; dz <= zRange; dz++) {
            for (int dy = -reach; dy <= reach; dy++) {
                for (int dx = -reach; dx <= reach; dx++) {
                    float rx = dx * s;
                    float ry = dy * s;
                    float rz = dz * s;
                    float r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 < h2) {
                        float diff = h2 - r2;
                        sum += poly6 * diff * diff * diff;
                    }
                }
            }
        }
        return Math.max(sum * 0.96f, 1.0f);
    }

    private void addParticle(float x, float y, float z, float vx, float vy, float vz) {
        int idx4 = particleCount * 4;
        if (idx4 + 4 > initPos.length) {
            int newLen = initPos.length * 2;
            float[] np = new float[newLen];
            float[] nv = new float[newLen];
            System.arraycopy(initPos, 0, np, 0, idx4);
            System.arraycopy(initVel, 0, nv, 0, idx4);
            initPos = np;
            initVel = nv;
        }
        initPos[idx4] = x;
        initPos[idx4 + 1] = y;
        initPos[idx4 + 2] = is2D ? 0.0f : z;
        initPos[idx4 + 3] = restDensity;

        initVel[idx4] = vx;
        initVel[idx4 + 1] = vy;
        initVel[idx4 + 2] = is2D ? 0.0f : vz;
        initVel[idx4 + 3] = 0.0f;
        particleCount++;
    }

    private void spawnBlock(float x0, float y0, float z0, float x1, float y1, float z1,
                            float vx, float vy, float vz) {
        for (float y = y0; y <= y1; y += spacing) {
            for (float x = x0; x <= x1; x += spacing) {
                if (is2D) {
                    addParticle(x, y, 0.0f, vx, vy, vz);
                } else {
                    for (float z = z0; z <= z1; z += spacing) {
                        addParticle(x, y, z, vx, vy, vz);
                    }
                }
            }
        }
    }

    private void spawnSphere(float cx, float cy, float cz, float rad, float vx, float vy, float vz) {
        float r2 = rad * rad;
        int steps = (int) Math.ceil(rad / spacing);
        int zRange = is2D ? 0 : steps;
        for (int iz = -zRange; iz <= zRange; iz++) {
            for (int iy = -steps; iy <= steps; iy++) {
                for (int ix = -steps; ix <= steps; ix++) {
                    float ox = ix * spacing;
                    float oy = iy * spacing;
                    float oz = iz * spacing;
                    if (ox * ox + oy * oy + oz * oz <= r2) {
                        addParticle(cx + ox, cy + oy, cz + oz, vx, vy, vz);
                    }
                }
            }
        }
    }

    private void spawnPresetParticles(int preset) {
        float pad = particleRadius * 1.6f;
        float w = maxX - minX;
        float hBox = maxY - minY;
        float z0 = minZ + pad;
        float z1 = maxZ - pad;

        switch (preset) {
            case SimulationConfig.PRESET_DOUBLE_DAM: {
                float colW = w * 0.27f;
                float colH = hBox * 0.72f;
                spawnBlock(minX + pad, minY + pad, z0, minX + colW, minY + colH, z1, 0f, 0f, 0f);
                spawnBlock(maxX - colW, minY + pad, z0, maxX - pad, minY + colH, z1, 0f, 0f, 0f);
                break;
            }
            case SimulationConfig.PRESET_SPLASH: {
                float poolH = hBox * 0.24f;
                spawnBlock(minX + pad, minY + pad, z0, maxX - pad, minY + poolH, z1, 0f, 0f, 0f);
                spawnSphere(0.0f, minY + hBox * 0.76f, 0.0f, Math.min(w, hBox) * 0.17f, 0f, -2.4f, 0f);
                break;
            }
            case SimulationConfig.PRESET_OBSTACLE: {
                float blockW = w * 0.56f;
                spawnBlock(-blockW * 0.5f, minY + hBox * 0.54f, z0 * 0.75f,
                            blockW * 0.5f, minY + hBox * 0.92f, z1 * 0.75f, 0f, -0.5f, 0f);
                break;
            }
            case SimulationConfig.PRESET_VORTEX: {
                float rOuter = Math.min(w, hBox) * 0.36f;
                float cy = minY + hBox * 0.52f;
                int steps = (int) Math.ceil(rOuter / spacing);
                int zRange = is2D ? 0 : (int) Math.ceil(((z1 - z0) * 0.42f) / spacing);
                for (int iz = -zRange; iz <= zRange; iz++) {
                    for (int iy = -steps; iy <= steps; iy++) {
                        for (int ix = -steps; ix <= steps; ix++) {
                            float ox = ix * spacing;
                            float oy = iy * spacing;
                            float oz = iz * spacing;
                            float d2 = ox * ox + oy * oy;
                            if (d2 <= rOuter * rOuter) {
                                float d = (float) Math.sqrt(d2) + 1e-4f;
                                float swirl = 3.2f * (d / rOuter);
                                if (is2D) {
                                    addParticle(ox, cy + oy, 0f, -oy / d * swirl, ox / d * swirl, 0f);
                                } else {
                                    float dr = (float) Math.sqrt(ox * ox + oz * oz) + 1e-4f;
                                    addParticle(ox, cy + oy * 0.7f, oz, -oz / dr * swirl, 0.2f, ox / dr * swirl);
                                }
                            }
                        }
                    }
                }
                break;
            }
            case SimulationConfig.PRESET_ZERO_G: {
                float rad = Math.min(w, hBox) * 0.20f;
                float cy = minY + hBox * 0.52f;
                spawnSphere(-w * 0.24f, cy + 0.04f, 0f, rad,  1.45f, -0.15f, 0f);
                spawnSphere( w * 0.24f, cy - 0.04f, 0f, rad, -1.45f,  0.15f, 0f);
                break;
            }
            case SimulationConfig.PRESET_DAM_BREAK:
            default: {
                boolean tall = (hBox > w * 1.1f);
                float damW = w * (tall ? 0.52f : 0.42f);
                float damH = hBox * (tall ? 0.58f : 0.76f);
                spawnBlock(minX + pad, minY + pad, z0, minX + pad + damW, minY + pad + damH, z1, 0f, 0f, 0f);
                break;
            }
        }
    }

    private void uploadInitialBuffers() {
        int n = Math.max(1, particleCount);
        int neededParticles = Math.max(n, allocatedParticles);
        if (n > allocatedParticles) {
            allocatedParticles = (int) (n * 1.35f) + 256;
            neededParticles = allocatedParticles;
            int byteCap = neededParticles * 16;

            allocBuffer(ssboPos, byteCap, GLES30.GL_DYNAMIC_COPY);
            allocBuffer(ssboVel, byteCap, GLES30.GL_DYNAMIC_COPY);
            allocBuffer(ssboPredA, byteCap, GLES30.GL_DYNAMIC_COPY);
            allocBuffer(ssboPredB, byteCap, GLES30.GL_DYNAMIC_COPY);
        }

        int neededGridWords = totalCells + neededParticles;
        if (neededGridWords > allocatedGridWords) {
            allocatedGridWords = (int) (neededGridWords * 1.25f) + 512;
            allocBuffer(ssboGrid, allocatedGridWords * 4, GLES30.GL_DYNAMIC_COPY);
        }

        int uploadBytes = particleCount * 16;
        if (uploadBytes > 0) {
            ByteBuffer stagePos = ByteBuffer.allocateDirect(uploadBytes).order(ByteOrder.nativeOrder());
            stagePos.asFloatBuffer().put(initPos, 0, particleCount * 4);
            stagePos.position(0);

            ByteBuffer stageVel = ByteBuffer.allocateDirect(uploadBytes).order(ByteOrder.nativeOrder());
            stageVel.asFloatBuffer().put(initVel, 0, particleCount * 4);
            stageVel.position(0);

            GLES30.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, ssboPos);
            GLES30.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, uploadBytes, stagePos);

            GLES30.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, ssboPredA);
            GLES30.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, uploadBytes, stagePos);

            GLES30.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, ssboVel);
            GLES30.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, uploadBytes, stageVel);

            GLES30.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0);
        }
    }

    private static void allocBuffer(int bufId, int bytes, int usage) {
        GLES30.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, bufId);
        GLES30.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, bytes, null, usage);
        GLES30.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0);
    }

    public void step(float frameDt, SimulationConfig cfg) {
        if (particleCount <= 0) return;

        float speed = cfg.simSpeed;
        float totalDt = Math.min(frameDt, 1.0f / 45.0f) * speed;
        if (totalDt <= 1e-6f) return;

        float subDt = totalDt / (float) substeps;
        float dtFactor = Math.max(0.02f, Math.min(1.0f, subDt / (0.016f / substeps)));
        float cfmScale = Math.max(1.0f, 1.0f / (dtFactor * dtFactor));

        float gMag = cfg.gravityMag;
        float gx, gy, gz;
        if (cfg.tiltGravityEnabled) {
            gx = cfg.tiltDirX * gMag;
            gy = cfg.tiltDirY * gMag;
            gz = is2D ? 0.0f : (cfg.tiltDirZ * gMag);
        } else {
            gx = 0.0f;
            gy = -gMag;
            gz = 0.0f;
        }

        float relaxEps = (float) (1200.0 * Math.pow(0.075 / h, 2.0) * cfmScale);
        float maxDisp = Math.max(particleRadius * 0.45f, h * 0.25f) * dtFactor;
        float stCoeff = (cfg.surfaceTension * spacing * 0.085f) / cfmScale;

        // Pack 52 words (208 bytes) into SimUniforms UBO without allocations
        FloatBuffer f = uboFloatView;
        IntBuffer u = uboIntView;

        // 0..3: u_bounds_min (xyz, cell_size)
        f.put(0, minX); f.put(1, minY); f.put(2, minZ); f.put(3, h);
        // 4..7: u_bounds_max (xyz, particle_radius)
        f.put(4, maxX); f.put(5, maxY); f.put(6, maxZ); f.put(7, particleRadius);
        // 8..11: u_gravity_dt
        f.put(8, gx);   f.put(9, gy);   f.put(10, gz);  f.put(11, subDt);
        // 12..15: u_grid_dim (ivec4)
        u.put(12, gridNx); u.put(13, gridNy); u.put(14, gridNz); u.put(15, totalCells);
        // 16..19: u_touch_orig (xyz, strength)
        f.put(16, cfg.rayOrigX); f.put(17, cfg.rayOrigY); f.put(18, cfg.rayOrigZ); f.put(19, 26.0f);
        // 20..23: u_touch_dir (xyz, mode)
        f.put(20, cfg.rayDirX);  f.put(21, cfg.rayDirY);  f.put(22, cfg.rayDirZ);  f.put(23, (float) cfg.activeTouchForceMode);
        // 24..27: u_obstacle (xyz, radius)
        f.put(24, cfg.obstacleX); f.put(25, cfg.obstacleY); f.put(26, 0.0f); f.put(27, cfg.obstacleR);
        // 28..31: u_kernel (h, h2, inv_rho0, rho0)
        f.put(28, h); f.put(29, h2); f.put(30, invRestDensity); f.put(31, restDensity);
        // 32..35: u_factors (poly6, spiky, w_q, tensile_k)
        f.put(32, poly6Factor); f.put(33, spikyFactor); f.put(34, wQ); f.put(35, 0.00025f);
        // 36..39: u_solver (eps, max_disp, restitution, friction)
        f.put(36, relaxEps); f.put(37, maxDisp); f.put(38, 0.12f); f.put(39, 0.04f);
        // 40..43: u_fluid (viscosity, st_coeff, max_vel, obs_enabled)
        f.put(40, cfg.viscosity); f.put(41, stCoeff); f.put(42, 8.5f); f.put(43, cfg.obstacleEnabled ? 1.0f : 0.0f);
        // 44..47: u_counts (particle_count, is_2d, 0, 0)
        u.put(44, particleCount); u.put(45, is2D ? 1 : 0); u.put(46, 0); u.put(47, 0);
        // 48..51: u_extra (vorticity, spacing, cfm_scale, self_density)
        f.put(48, cfg.vorticity); f.put(49, spacing); f.put(50, cfmScale); f.put(51, selfDensity);

        uboByteBuf.position(0);
        GLES30.glBindBuffer(GLES30.GL_UNIFORM_BUFFER, uboSim);
        GLES30.glBufferSubData(GLES30.GL_UNIFORM_BUFFER, 0, UBO_BYTES, uboByteBuf);
        GLES30.glBindBufferBase(GLES30.GL_UNIFORM_BUFFER, 0, uboSim);

        int pGroups = (particleCount + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        int cGroups = (totalCells + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        boolean doVorticity = (cfg.vorticity > 1e-4f);

        for (int sub = 0; sub < substeps; sub++) {
            // 1. Predict positions
            GLES30.glUseProgram(progPredict);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboPos);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, ssboVel);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, ssboPredA);
            GLES31.glDispatchCompute(pGroups, 1, 1);

            // 2. Clear grid heads (independent of ssboPredA, can overlap before barrier!)
            GLES30.glUseProgram(progClearGrid);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboGrid);
            GLES31.glDispatchCompute(cGroups, 1, 1);
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);

            // 3. Insert predicted particles into spatial grid
            GLES30.glUseProgram(progInsertGrid);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboPredA);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, ssboGrid);
            GLES31.glDispatchCompute(pGroups, 1, 1);
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);

            // 4 & 5. Jacobi Pressure + Cohesion + Collision Iterations
            for (int iter = 0; iter < solverIterations; iter++) {
                GLES30.glUseProgram(progDensity);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboPredA);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, ssboGrid);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, ssboPredB);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 3, ssboPos);
                GLES31.glDispatchCompute(pGroups, 1, 1);
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);

                GLES30.glUseProgram(progSolveCollide);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboPredB);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, ssboPos);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, ssboGrid);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 3, ssboPredA);
                GLES31.glDispatchCompute(pGroups, 1, 1);
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);
            }

            // 6. Derive velocity + boundary friction/restitution + XSPH viscosity
            GLES30.glUseProgram(progVelocityXsph);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboPredA);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, ssboPos);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, ssboGrid);
            GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 3, ssboVel);
            GLES31.glDispatchCompute(pGroups, 1, 1);
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);

            // Zero-copy swap: ssboPredA (new_pos.xyz, rho) becomes ssboPos!
            int tmp = ssboPos;
            ssboPos = ssboPredA;
            ssboPredA = tmp;

            // 7 & 8. Flow-gated Vorticity Confinement
            if (doVorticity) {
                GLES30.glUseProgram(progCurl);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboPos);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, ssboVel);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, ssboGrid);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 3, ssboPredB);
                GLES31.glDispatchCompute(pGroups, 1, 1);
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);

                GLES30.glUseProgram(progVorticity);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, ssboPos);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, ssboPredB);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, ssboGrid);
                GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 3, ssboVel);
                GLES31.glDispatchCompute(pGroups, 1, 1);
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }

        // Ensure updated SSBO data is visible to hardware instanced vertex attribute fetch
        GLES31.glMemoryBarrier(GLES31.GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        GLES30.glUseProgram(0);
    }
}
