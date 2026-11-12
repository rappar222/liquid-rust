package com.liquidrust.mobile;

import android.opengl.GLES30;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Multi-threaded, zero-allocation flat-array CPU PBF fallback solver for OpenGL ES 3.0 devices.
 * Uses pre-spawned worker threads across CPU cores and direct NIO FloatBuffers for VBO streaming.
 */
public final class CpuPbfSolver {

    private static final int HASH_SIZE = 16384;
    private static final int HASH_MASK = HASH_SIZE - 1;

    private final int numThreads;
    private final ExecutorService pool;

    private int vboPos;
    private int vboVel;
    private int vboCapacity = 0;
    private ByteBuffer posByteBuf;
    private FloatBuffer posFloatBuf;
    private ByteBuffer velByteBuf;
    private FloatBuffer velFloatBuf;

    public float minX = -0.90f, minY = 0.0f, minZ = -0.05f;
    public float maxX =  0.90f, maxY = 1.15f, maxZ =  0.05f;

    private int particleCount = 0;
    private boolean is2D = true;
    private float spacing = 0.028f;
    private float particleRadius = 0.015f;
    private float h = 0.056f;
    private float h2 = h * h;
    private float invH = 1.0f / h;
    private float poly6Factor;
    private float spikyFactor;
    private float wQ;
    private float selfDensity;
    private float restDensity = 1000.0f;
    private float invRestDensity = 0.001f;

    private int capacity = 4096;
    private float[] pos = new float[capacity * 4];  // xyz, density
    private float[] vel = new float[capacity * 4];  // xyz, pres_norm
    private float[] oldP = new float[capacity * 3];
    private float[] pred = new float[capacity * 3];
    private float[] delta = new float[capacity * 3];
    private float[] lambdas = new float[capacity];
    private final int[] gridHeads = new int[HASH_SIZE];
    private int[] gridNext = new int[capacity];

    public CpuPbfSolver() {
        this.numThreads = Math.max(1, Math.min(6, Runtime.getRuntime().availableProcessors()));
        this.pool = Executors.newFixedThreadPool(numThreads, r -> {
            Thread t = new Thread(r, "PbfWorker");
            t.setDaemon(true);
            return t;
        });

        int[] bufs = new int[2];
        GLES30.glGenBuffers(2, bufs, 0);
        vboPos = bufs[0];
        vboVel = bufs[1];
    }

    public int getPosBufferId() { return vboPos; }
    public int getVelBufferId() { return vboVel; }
    public int getParticleCount() { return particleCount; }
    public float getParticleRadius() { return particleRadius; }
    public float getInvRestDensity() { return invRestDensity; }
    public float getSmoothingLength() { return h; }
    public boolean is2D() { return is2D; }

    public void shutdown() {
        pool.shutdownNow();
    }

    private void ensureCapacity(int n) {
        if (n > capacity) {
            capacity = Math.max(n + 512, capacity * 2);
            pos = Arrays.copyOf(pos, capacity * 4);
            vel = Arrays.copyOf(vel, capacity * 4);
            oldP = new float[capacity * 3];
            pred = new float[capacity * 3];
            delta = new float[capacity * 3];
            lambdas = new float[capacity];
            gridNext = new int[capacity];
        }
        if (n > vboCapacity) {
            vboCapacity = Math.max(n + 512, vboCapacity * 2);
            int bytes = vboCapacity * 16;
            posByteBuf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
            posFloatBuf = posByteBuf.asFloatBuffer();
            velByteBuf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
            velFloatBuf = velByteBuf.asFloatBuffer();

            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboPos);
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, bytes, null, GLES30.GL_DYNAMIC_DRAW);
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboVel);
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, bytes, null, GLES30.GL_DYNAMIC_DRAW);
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
        }
    }

    public void loadScene(SimulationConfig cfg) {
        this.is2D = cfg.is2D;
        int q = Math.min(cfg.quality, SimulationConfig.QUALITY_HIGH); // Cap CPU fallback for smooth FPS

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
                case SimulationConfig.QUALITY_DRAFT:  spacing = 0.028f; break;
                case SimulationConfig.QUALITY_MEDIUM: spacing = 0.020f; break;
                default:                              spacing = 0.015f; break;
            }
            h = spacing * 2.05f;
        } else {
            minX = -0.65f; maxX = 0.65f;
            minY =  0.00f; maxY = 1.05f;
            minZ = -0.55f; maxZ = 0.55f;
            switch (q) {
                case SimulationConfig.QUALITY_DRAFT:  spacing = 0.056f; break;
                case SimulationConfig.QUALITY_MEDIUM: spacing = 0.044f; break;
                default:                              spacing = 0.036f; break;
            }
            h = spacing * 1.95f;
        }

        particleRadius = spacing * 0.52f;
        h2 = h * h;
        invH = 1.0f / h;
        poly6Factor = (float) (315.0 / (64.0 * Math.PI * Math.pow(h, 9)));
        spikyFactor = (float) (45.0 / (Math.PI * Math.pow(h, 6)));
        selfDensity = poly6Factor * h2 * h2 * h2;
        float qDist = 0.3f * h;
        float diffQ = h2 - qDist * qDist;
        wQ = poly6Factor * diffQ * diffQ * diffQ;

        int reach = (int) Math.ceil(h / spacing) + 1;
        float sum = 0f;
        int zR = is2D ? 0 : reach;
        for (int dz = -zR; dz <= zR; dz++) {
            for (int dy = -reach; dy <= reach; dy++) {
                for (int dx = -reach; dx <= reach; dx++) {
                    float r2 = (dx * dx + dy * dy + dz * dz) * spacing * spacing;
                    if (r2 < h2) {
                        float d = h2 - r2;
                        sum += poly6Factor * d * d * d;
                    }
                }
            }
        }
        restDensity = Math.max(1.0f, sum * 0.96f);
        invRestDensity = 1.0f / restDensity;

        particleCount = 0;
        int preset = cfg.preset;
        cfg.obstacleEnabled = (preset == SimulationConfig.PRESET_OBSTACLE);
        if (preset == SimulationConfig.PRESET_ZERO_G) cfg.gravityMag = 0.0f;
        else if (cfg.gravityMag < 0.1f) cfg.gravityMag = 9.81f;

        float pad = particleRadius * 1.6f;
        float w = maxX - minX;
        float hBox = maxY - minY;
        float z0 = minZ + pad, z1 = maxZ - pad;

        switch (preset) {
            case SimulationConfig.PRESET_DOUBLE_DAM:
                spawnBlock(minX + pad, minY + pad, z0, minX + w * 0.27f, minY + hBox * 0.72f, z1, 0, 0, 0);
                spawnBlock(maxX - w * 0.27f, minY + pad, z0, maxX - pad, minY + hBox * 0.72f, z1, 0, 0, 0);
                break;
            case SimulationConfig.PRESET_SPLASH:
                spawnBlock(minX + pad, minY + pad, z0, maxX - pad, minY + hBox * 0.24f, z1, 0, 0, 0);
                spawnSphere(0f, minY + hBox * 0.76f, 0f, Math.min(w, hBox) * 0.17f, 0f, -2.4f, 0f);
                break;
            case SimulationConfig.PRESET_OBSTACLE:
                spawnBlock(-w * 0.28f, minY + hBox * 0.54f, z0 * 0.75f, w * 0.28f, minY + hBox * 0.92f, z1 * 0.75f, 0f, -0.5f, 0f);
                break;
            case SimulationConfig.PRESET_VORTEX:
                spawnSphere(0f, minY + hBox * 0.52f, 0f, Math.min(w, hBox) * 0.34f, 1.8f, 0f, 0f);
                break;
            case SimulationConfig.PRESET_ZERO_G:
                spawnSphere(-w * 0.24f, minY + hBox * 0.54f, 0f, Math.min(w, hBox) * 0.20f,  1.45f, -0.15f, 0f);
                spawnSphere( w * 0.24f, minY + hBox * 0.48f, 0f, Math.min(w, hBox) * 0.20f, -1.45f,  0.15f, 0f);
                break;
            default: {
                boolean tall = (hBox > w * 1.1f);
                float damW = w * (tall ? 0.52f : 0.42f);
                float damH = hBox * (tall ? 0.58f : 0.76f);
                spawnBlock(minX + pad, minY + pad, z0, minX + pad + damW, minY + pad + damH, z1, 0, 0, 0);
                break;
            }
        }
        ensureCapacity(Math.max(1, particleCount));
        uploadVbos();
    }

    private void addP(float x, float y, float z, float vx, float vy, float vz) {
        ensureCapacity(particleCount + 1);
        int i4 = particleCount * 4;
        pos[i4] = x; pos[i4 + 1] = y; pos[i4 + 2] = is2D ? 0f : z; pos[i4 + 3] = restDensity;
        vel[i4] = vx; vel[i4 + 1] = vy; vel[i4 + 2] = is2D ? 0f : vz; vel[i4 + 3] = 0f;
        particleCount++;
    }

    private void spawnBlock(float x0, float y0, float z0, float x1, float y1, float z1, float vx, float vy, float vz) {
        for (float y = y0; y <= y1; y += spacing) {
            for (float x = x0; x <= x1; x += spacing) {
                if (is2D) addP(x, y, 0f, vx, vy, vz);
                else for (float z = z0; z <= z1; z += spacing) addP(x, y, z, vx, vy, vz);
            }
        }
    }

    private void spawnSphere(float cx, float cy, float cz, float rad, float vx, float vy, float vz) {
        float r2 = rad * rad;
        int steps = (int) Math.ceil(rad / spacing);
        int zR = is2D ? 0 : steps;
        for (int iz = -zR; iz <= zR; iz++) {
            for (int iy = -steps; iy <= steps; iy++) {
                for (int ix = -steps; ix <= steps; ix++) {
                    float ox = ix * spacing, oy = iy * spacing, oz = iz * spacing;
                    if (ox * ox + oy * oy + oz * oz <= r2) addP(cx + ox, cy + oy, cz + oz, vx, vy, vz);
                }
            }
        }
    }

    private static int hash(int cx, int cy, int cz) {
        return ((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) & HASH_MASK;
    }

    private interface RangeTask {
        void run(int start, int end);
    }

    private void parallelFor(int total, RangeTask task) {
        if (total < 512 || numThreads <= 1) {
            task.run(0, total);
            return;
        }
        int chunk = (total + numThreads - 1) / numThreads;
        CountDownLatch latch = new CountDownLatch(numThreads);
        for (int t = 0; t < numThreads; t++) {
            final int s = t * chunk;
            final int e = Math.min(total, s + chunk);
            pool.execute(() -> {
                try {
                    if (s < e) task.run(s, e);
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
    }

    public void step(float frameDt, SimulationConfig cfg) {
        final int n = particleCount;
        if (n <= 0) return;

        final float totalDt = Math.min(frameDt, 1.0f / 45.0f) * cfg.simSpeed;
        if (totalDt <= 1e-6f) return;

        final int substeps = 2;
        final float dt = totalDt / substeps;
        final float invDt = 1.0f / dt;
        final float dtFactor = Math.max(0.02f, Math.min(1.0f, dt / 0.008f));
        final float cfmScale = Math.max(1.0f, 1.0f / (dtFactor * dtFactor));
        final float relaxEps = (float) (1200.0 * Math.pow(0.075 / h, 2.0) * cfmScale);
        final float maxDisp = Math.max(particleRadius * 0.45f, h * 0.25f) * dtFactor;
        final float stCoeff = (cfg.surfaceTension * spacing * 0.085f) / cfmScale;

        final float gMag = cfg.gravityMag;
        final float gx = cfg.tiltGravityEnabled ? cfg.tiltDirX * gMag : 0f;
        final float gy = cfg.tiltGravityEnabled ? cfg.tiltDirY * gMag : -gMag;
        final float gz = (cfg.tiltGravityEnabled && !is2D) ? cfg.tiltDirZ * gMag : 0f;

        final int tMode = cfg.activeTouchForceMode;
        final float rOx = cfg.rayOrigX, rOy = cfg.rayOrigY, rOz = cfg.rayOrigZ;
        final float rDx = cfg.rayDirX,  rDy = cfg.rayDirY,  rDz = cfg.rayDirZ;
        final boolean obsOn = cfg.obstacleEnabled;
        final float obsX = cfg.obstacleX, obsY = cfg.obstacleY, obsR = cfg.obstacleR;

        final float bMinX = minX + particleRadius, bMaxX = maxX - particleRadius;
        final float bMinY = minY + particleRadius, bMaxY = maxY - particleRadius;
        final float bMinZ = is2D ? 0f : (minZ + particleRadius);
        final float bMaxZ = is2D ? 0f : (maxZ - particleRadius);
        final int zMin = is2D ? 0 : -1, zMax = is2D ? 0 : 1;

        for (int sub = 0; sub < substeps; sub++) {
            parallelFor(n, (s, e) -> {
                for (int i = s; i < e; i++) {
                    int i4 = i * 4, i3 = i * 3;
                    float px = pos[i4], py = pos[i4 + 1], pz = pos[i4 + 2];
                    float vx = vel[i4], vy = vel[i4 + 1], vz = vel[i4 + 2];
                    oldP[i3] = px; oldP[i3 + 1] = py; oldP[i3 + 2] = pz;

                    float fx = 0f, fy = 0f, fz = 0f;
                    if (tMode != 0) {
                        float tpx = px - rOx, tpy = py - rOy, tpz = pz - rOz;
                        float proj = tpx * rDx + tpy * rDy + tpz * rDz;
                        if (proj > 0f) {
                            float dfx = (rOx + rDx * proj) - px;
                            float dfy = (rOy + rDy * proj) - py;
                            float dfz = is2D ? 0f : ((rOz + rDz * proj) - pz);
                            float d = (float) Math.sqrt(dfx * dfx + dfy * dfy + dfz * dfz);
                            if (d < 0.52f && d > 1e-4f) {
                                float falloff = 1.0f - d / 0.52f;
                                float str = 26.0f * falloff / d;
                                if (tMode > 0) {
                                    fx = dfx * str - vx * 2.2f * falloff;
                                    fy = dfy * str - vy * 2.2f * falloff;
                                    fz = dfz * str - vz * 2.2f * falloff;
                                } else {
                                    fx = -dfx * str * 1.25f;
                                    fy = -dfy * str * 1.25f;
                                    fz = -dfz * str * 1.25f;
                                }
                            }
                        }
                    }

                    vx += (gx + fx) * dt;
                    vy += (gy + fy) * dt;
                    vz += (gz + fz) * dt;

                    pred[i3]     = Math.max(bMinX, Math.min(bMaxX, px + vx * dt));
                    pred[i3 + 1] = Math.max(bMinY, Math.min(bMaxY, py + vy * dt));
                    pred[i3 + 2] = is2D ? 0f : Math.max(bMinZ, Math.min(bMaxZ, pz + vz * dt));
                }
            });

            Arrays.fill(gridHeads, -1);
            for (int i = 0; i < n; i++) {
                int i3 = i * 3;
                int cx = (int) Math.floor(pred[i3] * invH);
                int cy = (int) Math.floor(pred[i3 + 1] * invH);
                int cz = is2D ? 0 : (int) Math.floor(pred[i3 + 2] * invH);
                int b = hash(cx, cy, cz);
                gridNext[i] = gridHeads[b];
                gridHeads[b] = i;
            }

            for (int iter = 0; iter < 3; iter++) {
                parallelFor(n, (s, e) -> {
                    for (int i = s; i < e; i++) {
                        int i3 = i * 3;
                        float px = pred[i3], py = pred[i3 + 1], pz = pred[i3 + 2];
                        int cx = (int) Math.floor(px * invH);
                        int cy = (int) Math.floor(py * invH);
                        int cz = is2D ? 0 : (int) Math.floor(pz * invH);

                        float dens = selfDensity;
                        float gix = 0f, giy = 0f, giz = 0f, sumSq = 0f;

                        for (int dz = zMin; dz <= zMax; dz++) {
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int j = gridHeads[hash(cx + dx, cy + dy, cz + dz)];
                                    while (j != -1) {
                                        if (i != j) {
                                            int j3 = j * 3;
                                            float rx = px - pred[j3], ry = py - pred[j3 + 1], rz = is2D ? 0f : (pz - pred[j3 + 2]);
                                            float r2 = rx * rx + ry * ry + rz * rz;
                                            if (r2 < h2 && r2 > 1e-12f) {
                                                float diff2 = h2 - r2;
                                                dens += poly6Factor * diff2 * diff2 * diff2;
                                                float r = (float) Math.sqrt(r2);
                                                float hm = h - r;
                                                float sc = (-spikyFactor * hm * hm / r) * invRestDensity;
                                                float gxJ = rx * sc, gyJ = ry * sc, gzJ = rz * sc;
                                                gix += gxJ; giy += gyJ; giz += gzJ;
                                                sumSq += gxJ * gxJ + gyJ * gyJ + gzJ * gzJ;
                                            }
                                        }
                                        j = gridNext[j];
                                    }
                                }
                            }
                        }

                        float dFloor = py - bMinY;
                        if (dFloor < h && dens < restDensity * 0.98f) {
                            float rw = Math.max(spacing * 0.92f, dFloor + spacing * 0.92f);
                            float df = Math.max(0f, h2 - rw * rw);
                            dens = Math.min(restDensity * 0.98f, dens + poly6Factor * df * df * df * (is2D ? 2.2f : 4.2f));
                        }

                        sumSq += gix * gix + giy * giy + giz * giz;
                        pos[i * 4 + 3] = dens;
                        float cI = Math.max(0f, dens * invRestDensity - 1.0f);
                        lambdas[i] = -cI / (sumSq + relaxEps);
                    }
                });

                parallelFor(n, (s, e) -> {
                    for (int i = s; i < e; i++) {
                        int i3 = i * 3;
                        float px = pred[i3], py = pred[i3 + 1], pz = pred[i3 + 2];
                        float lamI = lambdas[i];
                        float rhoI = pos[i * 4 + 3];
                        int cx = (int) Math.floor(px * invH);
                        int cy = (int) Math.floor(py * invH);
                        int cz = is2D ? 0 : (int) Math.floor(pz * invH);

                        float dlx = 0f, dly = 0f, dlz = 0f;
                        for (int dz = zMin; dz <= zMax; dz++) {
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int j = gridHeads[hash(cx + dx, cy + dy, cz + dz)];
                                    while (j != -1) {
                                        if (i != j) {
                                            int j3 = j * 3;
                                            float rx = px - pred[j3], ry = py - pred[j3 + 1], rz = is2D ? 0f : (pz - pred[j3 + 2]);
                                            float r2 = rx * rx + ry * ry + rz * rz;
                                            if (r2 < h2 && r2 > 1e-12f) {
                                                float r = (float) Math.sqrt(r2);
                                                float lamSum = lamI + lambdas[j];
                                                float sCorr = 0f;
                                                if (lamSum < -1e-6f && r > spacing * 0.80f) {
                                                    float df = h2 - r2;
                                                    float rat = Math.min(1.5f, (poly6Factor * df * df * df) / wQ);
                                                    float r4 = rat * rat * rat * rat;
                                                    sCorr = (-0.00025f * r4 * h2) / cfmScale;
                                                }
                                                float hm = h - r;
                                                float sc = (-spikyFactor * hm * hm / r) * ((lamSum + sCorr) * invRestDensity);
                                                dlx += rx * sc; dly += ry * sc; dlz += rz * sc;

                                                if (stCoeff > 0f && r > spacing * 0.48f) {
                                                    float q = r / h, omq = 1f - q;
                                                    float sp = q * q * q * omq * omq * omq * 64.0f;
                                                    float kij = (2.0f * restDensity) / Math.max(rhoI + pos[j * 4 + 3], restDensity * 0.5f);
                                                    float pull = (stCoeff * kij * sp) / r;
                                                    dlx -= rx * pull; dly -= ry * pull; dlz -= rz * pull;
                                                }
                                            }
                                        }
                                        j = gridNext[j];
                                    }
                                }
                            }
                        }

                        float dLen = (float) Math.sqrt(dlx * dlx + dly * dly + dlz * dlz);
                        if (dLen > maxDisp && dLen > 1e-7f) {
                            float sc = maxDisp / dLen;
                            dlx *= sc; dly *= sc; dlz *= sc;
                        }
                        delta[i3] = dlx; delta[i3 + 1] = dly; delta[i3 + 2] = dlz;
                    }
                });

                parallelFor(n, (s, e) -> {
                    for (int i = s; i < e; i++) {
                        int i3 = i * 3;
                        float nx = Math.max(bMinX, Math.min(bMaxX, pred[i3] + delta[i3] * 0.65f));
                        float ny = Math.max(bMinY, Math.min(bMaxY, pred[i3 + 1] + delta[i3 + 1] * 0.65f));
                        float nz = is2D ? 0f : Math.max(bMinZ, Math.min(bMaxZ, pred[i3 + 2] + delta[i3 + 2] * 0.65f));

                        if (obsOn) {
                            float dx = nx - obsX, dy = ny - obsY, dz = is2D ? 0f : nz;
                            float d = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
                            float minD = obsR + particleRadius;
                            if (d < minD && d > 1e-5f) {
                                float sc = minD / d;
                                nx = Math.max(bMinX, Math.min(bMaxX, obsX + dx * sc));
                                ny = Math.max(bMinY, Math.min(bMaxY, obsY + dy * sc));
                                nz = is2D ? 0f : Math.max(bMinZ, Math.min(bMaxZ, dz * sc));
                            }
                        }
                        pred[i3] = nx; pred[i3 + 1] = ny; pred[i3 + 2] = nz;
                    }
                });
            }

            parallelFor(n, (s, e) -> {
                for (int i = s; i < e; i++) {
                    int i3 = i * 3, i4 = i * 4;
                    float nx = pred[i3], ny = pred[i3 + 1], nz = pred[i3 + 2];
                    float vx = (nx - oldP[i3]) * invDt;
                    float vy = (ny - oldP[i3 + 1]) * invDt;
                    float vz = is2D ? 0f : ((nz - oldP[i3 + 2]) * invDt);

                    if (ny <= bMinY + 0.0015f) {
                        if (vy < 0f) vy = (vy < -0.12f) ? -vy * 0.12f : 0f;
                        else if (vy < 0.22f) vy *= 0.08f;
                        vx *= 0.98f; vz *= 0.98f;
                    }
                    float sp = (float) Math.sqrt(vx * vx + vy * vy + vz * vz);
                    if (sp > 8.5f) {
                        float sc = 8.5f / sp;
                        vx *= sc; vy *= sc; vz *= sc;
                    }
                    pos[i4] = nx; pos[i4 + 1] = ny; pos[i4 + 2] = nz;
                    vel[i4] = vx; vel[i4 + 1] = vy; vel[i4 + 2] = vz;
                    vel[i4 + 3] = Math.max(0f, Math.min(1f, (pos[i4 + 3] * invRestDensity - 1.0f) / 0.45f));
                }
            });
        }

        uploadVbos();
    }

    private void uploadVbos() {
        int words = particleCount * 4;
        if (words <= 0) return;

        posFloatBuf.position(0);
        posFloatBuf.put(pos, 0, words);
        posByteBuf.position(0);

        velFloatBuf.position(0);
        velFloatBuf.put(vel, 0, words);
        velByteBuf.position(0);

        int bytes = words * 4;
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboPos);
        GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, bytes, posByteBuf);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboVel);
        GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, bytes, velByteBuf);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
    }
}
