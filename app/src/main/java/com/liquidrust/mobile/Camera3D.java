package com.liquidrust.mobile;

import android.opengl.Matrix;

/**
 * Zero-allocation 2D Planar & 3D Orbital Camera with exact mobile safe-area framing
 * and unproject screen-ray casting for touch fluid forces.
 */
public final class Camera3D {

    private static final float FOV_Y_DEG = 44.0f;

    public final float[] viewMatrix = new float[16];
    public final float[] projMatrix = new float[16];
    public final float[] viewProjMatrix = new float[16];
    private final float[] invViewProj = new float[16];
    private final float[] nearVec = new float[4];
    private final float[] farVec = new float[4];
    private final float[] outNear = new float[4];
    private final float[] outFar = new float[4];

    private float boundMinX = -0.60f, boundMaxX = 0.60f;
    private float boundMinY =  0.00f, boundMaxY = 1.86f;
    private float boundMinZ = -0.05f, boundMaxZ = 0.05f;

    private float targetX = 0.0f;
    private float targetY = 0.90f;
    private float targetZ = 0.0f;

    private float yaw = 0.56f;
    private float pitch = 0.34f;
    private float distance = 3.20f;
    private boolean pendingAutoFit = true;

    private int viewportWidth = 1080;
    private int viewportHeight = 2340;

    public Camera3D() {
        reset(true);
    }

    public synchronized void setChamberBounds(float minX, float maxX,
                                              float minY, float maxY,
                                              float minZ, float maxZ) {
        this.boundMinX = minX;
        this.boundMaxX = maxX;
        this.boundMinY = minY;
        this.boundMaxY = maxY;
        this.boundMinZ = minZ;
        this.boundMaxZ = maxZ;
    }

    public synchronized void reset(boolean is2D) {
        targetX = (boundMinX + boundMaxX) * 0.5f;
        // Slight downward target shift visually lifts the tank slightly above the bottom floating dock
        float hSpan = boundMaxY - boundMinY;
        targetY = (boundMinY + boundMaxY) * 0.5f - (is2D ? hSpan * 0.015f : 0.0f);
        targetZ = (boundMinZ + boundMaxZ) * 0.5f;

        if (is2D) {
            yaw = 0.0f;
            pitch = 0.0f;
        } else {
            yaw = 0.58f;
            pitch = 0.35f;
        }
        distance = computeIdealDistance(is2D);
        pendingAutoFit = false;
    }

    private float computeIdealDistance(boolean is2D) {
        float aspect = (float) Math.max(1, viewportWidth) / (float) Math.max(1, viewportHeight);
        float tanHalfY = (float) Math.tan(Math.toRadians(FOV_Y_DEG * 0.5));
        float tanHalfX = Math.max(0.12f, tanHalfY * aspect);

        if (is2D) {
            float wPad = (boundMaxX - boundMinX) * 1.10f;
            float hPad = (boundMaxY - boundMinY) * 1.28f; // Reserves clean room for top chip + bottom dock
            float distW = (wPad * 0.5f) / tanHalfX;
            float distH = (hPad * 0.5f) / tanHalfY;
            return Math.max(distW, distH);
        } else {
            float spanXZ = (float) Math.hypot(boundMaxX - boundMinX, boundMaxZ - boundMinZ) * 1.12f;
            float spanY = (boundMaxY - boundMinY) * 1.38f;
            float distW = (spanXZ * 0.5f) / tanHalfX;
            float distH = (spanY * 0.5f) / tanHalfY;
            return Math.max(distW, distH);
        }
    }

    public synchronized void setViewport(int width, int height, boolean is2D) {
        boolean changed = (this.viewportWidth != width || this.viewportHeight != height);
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        if (changed || pendingAutoFit) {
            reset(is2D);
        }
    }

    public synchronized void orbit(float dYaw, float dPitch) {
        yaw += dYaw;
        pitch = Math.max(-1.35f, Math.min(1.35f, pitch + dPitch));
    }

    public synchronized void pan(float dxPx, float dyPx) {
        float scale = distance / Math.max(320.0f, Math.min(viewportWidth, viewportHeight));
        float cosYaw = (float) Math.cos(yaw);
        float sinYaw = (float) Math.sin(yaw);

        targetX -= dxPx * scale * cosYaw;
        targetZ += dxPx * scale * sinYaw;
        targetY += dyPx * scale;
    }

    public synchronized void zoomByFactor(float scaleFactor) {
        if (scaleFactor <= 0.01f) return;
        distance = Math.max(0.80f, Math.min(8.50f, distance / scaleFactor));
    }

    public synchronized void updateMatrices(boolean is2D) {
        float aspect = (float) viewportWidth / (float) viewportHeight;
        Matrix.perspectiveM(projMatrix, 0, FOV_Y_DEG, aspect, 0.08f, 32.0f);

        float eyeX, eyeY, eyeZ;
        if (is2D) {
            eyeX = targetX;
            eyeY = targetY;
            eyeZ = distance;
        } else {
            float cosP = (float) Math.cos(pitch);
            eyeX = targetX + distance * cosP * (float) Math.sin(yaw);
            eyeY = targetY + distance * (float) Math.sin(pitch);
            eyeZ = targetZ + distance * cosP * (float) Math.cos(yaw);
        }

        Matrix.setLookAtM(viewMatrix, 0,
            eyeX, eyeY, eyeZ,
            targetX, targetY, targetZ,
            0.0f, 1.0f, 0.0f);

        Matrix.multiplyMM(viewProjMatrix, 0, projMatrix, 0, viewMatrix, 0);
    }

    /**
     * Unprojects screen touch coordinates (xPx, yPx) into SimulationConfig's world ray without allocations.
     * Returns true if the cast ray intersects the simulation chamber bounding box.
     */
    public synchronized boolean computeTouchRay(float xPx, float yPx, boolean is2D, SimulationConfig cfg) {
        updateMatrices(is2D);
        if (!Matrix.invertM(invViewProj, 0, viewProjMatrix, 0)) {
            return false;
        }

        float ndcX = (xPx / (float) viewportWidth) * 2.0f - 1.0f;
        float ndcY = 1.0f - (yPx / (float) viewportHeight) * 2.0f;

        nearVec[0] = ndcX; nearVec[1] = ndcY; nearVec[2] = -1.0f; nearVec[3] = 1.0f;
        farVec[0] = ndcX;  farVec[1] = ndcY;  farVec[2] =  1.0f; farVec[3] = 1.0f;

        Matrix.multiplyMV(outNear, 0, invViewProj, 0, nearVec, 0);
        Matrix.multiplyMV(outFar, 0, invViewProj, 0, farVec, 0);

        float invWn = (Math.abs(outNear[3]) > 1e-6f) ? (1.0f / outNear[3]) : 1.0f;
        float invWf = (Math.abs(outFar[3]) > 1e-6f) ? (1.0f / outFar[3]) : 1.0f;

        float ox = outNear[0] * invWn;
        float oy = outNear[1] * invWn;
        float oz = outNear[2] * invWn;

        float dx = outFar[0] * invWf - ox;
        float dy = outFar[1] * invWf - oy;
        float dz = outFar[2] * invWf - oz;

        float len = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len <= 1e-6f) {
            return false;
        }

        float invL = 1.0f / len;
        float ndx = dx * invL;
        float ndy = dy * invL;
        float ndz = dz * invL;

        cfg.rayOrigX = ox;
        cfg.rayOrigY = oy;
        cfg.rayOrigZ = oz;
        cfg.rayDirX = ndx;
        cfg.rayDirY = ndy;
        cfg.rayDirZ = ndz;

        return rayIntersectsAABB(ox, oy, oz, ndx, ndy, ndz,
            boundMinX, boundMaxX, boundMinY, boundMaxY, boundMinZ, boundMaxZ);
    }

    private static boolean rayIntersectsAABB(float ox, float oy, float oz,
                                             float dx, float dy, float dz,
                                             float minX, float maxX,
                                             float minY, float maxY,
                                             float minZ, float maxZ) {
        float tMin = 0.0f;
        float tMax = 1e6f;

        if (Math.abs(dx) < 1e-6f) {
            if (ox < minX || ox > maxX) return false;
        } else {
            float inv = 1.0f / dx;
            float t1 = (minX - ox) * inv;
            float t2 = (maxX - ox) * inv;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = Math.max(tMin, t1);
            tMax = Math.min(tMax, t2);
            if (tMin > tMax) return false;
        }

        if (Math.abs(dy) < 1e-6f) {
            if (oy < minY || oy > maxY) return false;
        } else {
            float inv = 1.0f / dy;
            float t1 = (minY - oy) * inv;
            float t2 = (maxY - oy) * inv;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = Math.max(tMin, t1);
            tMax = Math.min(tMax, t2);
            if (tMin > tMax) return false;
        }

        if (Math.abs(dz) < 1e-6f) {
            if (oz < minZ || oz > maxZ) return false;
        } else {
            float inv = 1.0f / dz;
            float t1 = (minZ - oz) * inv;
            float t2 = (maxZ - oz) * inv;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = Math.max(tMin, t1);
            tMax = Math.min(tMax, t2);
            if (tMin > tMax) return false;
        }

        return tMax >= Math.max(tMin, 0.0f);
    }
}
