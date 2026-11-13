package com.liquidrust.mobile;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Lock-free shared configuration between Android UI/Sensor threads and the OpenGL Render thread.
 * Zero allocations during per-frame reads.
 */
public final class SimulationConfig {

    // Scene Presets
    public static final int PRESET_DAM_BREAK = 0;
    public static final int PRESET_DOUBLE_DAM = 1;
    public static final int PRESET_SPLASH = 2;
    public static final int PRESET_OBSTACLE = 3;
    public static final int PRESET_VORTEX = 4;
    public static final int PRESET_ZERO_G = 5;

    public static final String[] PRESET_NAMES = {
        "Dam Break",
        "Double Dam",
        "Droplet Splash",
        "Obstacle Drop",
        "Vortex Swirl",
        "Zero-G Blobs"
    };

    // Lattice Quality Levels (Default: DRAFT + 2D Planar)
    public static final int QUALITY_DRAFT = 0;
    public static final int QUALITY_MEDIUM = 1;
    public static final int QUALITY_HIGH = 2;
    public static final int QUALITY_ULTRA = 3;
    public static final int QUALITY_EXTREME = 4;

    public static final String[] QUALITY_NAMES = {
        "Draft",
        "Balanced",
        "High",
        "Ultra",
        "Extreme"
    };

    // Single-finger Touch Tool
    public static final int TOOL_ATTRACT = 0;
    public static final int TOOL_REPEL = 1;
    public static final int TOOL_CAMERA = 2;

    // Optical Material Presets
    public static final class MaterialPreset {
        public final String name;
        public final int shallowColor;
        public final int deepColor;
        public final float sR, sG, sB;
        public final float dR, dG, dB;
        public final float ior, roughness, specular, refraction, metallic, foam;

        public MaterialPreset(String name, int shallowHex, int deepHex,
                              float ior, float roughness, float specular,
                              float refraction, float metallic, float foam) {
            this.name = name;
            this.shallowColor = 0xFF000000 | shallowHex;
            this.deepColor = 0xFF000000 | deepHex;
            this.sR = ((shallowHex >> 16) & 0xFF) / 255.0f;
            this.sG = ((shallowHex >> 8) & 0xFF) / 255.0f;
            this.sB = (shallowHex & 0xFF) / 255.0f;
            this.dR = ((deepHex >> 16) & 0xFF) / 255.0f;
            this.dG = ((deepHex >> 8) & 0xFF) / 255.0f;
            this.dB = (deepHex & 0xFF) / 255.0f;
            this.ior = ior;
            this.roughness = roughness;
            this.specular = specular;
            this.refraction = refraction;
            this.metallic = metallic;
            this.foam = foam;
        }
    }

    public static final MaterialPreset[] MATERIALS = {
        new MaterialPreset("Clear Water",   0x38BDF8, 0x0C3B8A, 1.333f, 0.045f, 1.35f, 0.42f, 0.00f, 0.35f),
        new MaterialPreset("Deep Ocean",    0x2199D6, 0x031733, 1.340f, 0.065f, 1.30f, 0.38f, 0.00f, 0.55f),
        new MaterialPreset("Lagoon",        0x3DE8DB, 0x055261, 1.335f, 0.040f, 1.40f, 0.48f, 0.00f, 0.25f),
        new MaterialPreset("Molten Amber",  0xFCB82E, 0x5C1403, 1.540f, 0.055f, 1.45f, 0.52f, 0.00f, 0.12f),
        new MaterialPreset("Crimson Wine",  0xE12D58, 0x2B0410, 1.360f, 0.050f, 1.30f, 0.32f, 0.00f, 0.10f),
        new MaterialPreset("Emerald Gel",   0x33EB85, 0x03381F, 1.520f, 0.042f, 1.50f, 0.48f, 0.00f, 0.15f),
        new MaterialPreset("Creamy Milk",   0xFAFCFF, 0xCBD5E1, 1.350f, 0.180f, 0.85f, 0.08f, 0.00f, 0.05f),
        new MaterialPreset("Liquid Chrome", 0xE6ECF5, 0x5A677D, 2.400f, 0.025f, 2.30f, 0.00f, 0.94f, 0.00f)
    };

    // Transport & Mode
    public volatile boolean isPlaying = true;
    public volatile boolean is2D = true;
    public volatile boolean isPortrait = true;
    public volatile int preset = PRESET_DAM_BREAK;
    public volatile int quality = QUALITY_DRAFT;
    public volatile float simSpeed = 1.0f;
    public volatile int touchTool = TOOL_ATTRACT;

    // Physics
    public volatile float vorticity = 0.045f;
    public volatile float surfaceTension = 0.030f;
    public volatile float viscosity = 0.035f;
    public volatile float gravityMag = 9.81f;

    // Hardware Tilt Gravity Sensor
    public volatile boolean tiltGravityEnabled = false;
    public volatile float tiltDirX = 0.0f;
    public volatile float tiltDirY = -1.0f;
    public volatile float tiltDirZ = 0.0f;

    // Sphere Collider
    public volatile boolean obstacleEnabled = false;
    public volatile float obstacleX = 0.0f;
    public volatile float obstacleY = 0.45f;
    public volatile float obstacleR = 0.20f;

    // Active Touch Ray
    public volatile int activeTouchForceMode = 0; // -1 repel, 0 none, +1 attract
    public volatile float rayOrigX = 0f, rayOrigY = 0f, rayOrigZ = 2f;
    public volatile float rayDirX = 0f, rayDirY = 0f, rayDirZ = -1f;

    // Live Optical Shading & Material
    public volatile int colorMode = 3; // 3: Optical Material, 0: Velocity, 1: Pressure, 2: Density
    public volatile int colorMap = 0;  // 0: Ocean, 1: Turbo, 2: Viridis, 3: Fire
    public volatile boolean litSpheres = true;
    public volatile int materialIndex = 0;
    public volatile float shallowR = 0.22f, shallowG = 0.74f, shallowB = 0.97f;
    public volatile float deepR = 0.05f, deepG = 0.23f, deepB = 0.54f;
    public volatile int texMode = 1;   // 0: None, 1: Ripples, 2: Caustics, 3: Foam, 4: Grid
    public volatile float texStrength = 0.55f;
    public volatile float foamAmount = 0.35f;
    public volatile float ior = 1.333f;
    public volatile float roughness = 0.045f;
    public volatile float specular = 1.35f;
    public volatile float refraction = 0.42f;
    public volatile float metallic = 0.00f;

    // Cross-thread commands
    public final AtomicBoolean reloadSceneRequested = new AtomicBoolean(false);
    public final AtomicBoolean stepOnceRequested = new AtomicBoolean(false);
    public final AtomicBoolean resetCameraRequested = new AtomicBoolean(false);

    public SimulationConfig() {
        applyMaterialPreset(0);
    }

    public void applyMaterialPreset(int idx) {
        if (idx < 0 || idx >= MATERIALS.length) return;
        MaterialPreset m = MATERIALS[idx];
        this.materialIndex = idx;
        this.colorMode = 3;
        this.shallowR = m.sR;
        this.shallowG = m.sG;
        this.shallowB = m.sB;
        this.deepR = m.dR;
        this.deepG = m.dG;
        this.deepB = m.dB;
        this.ior = m.ior;
        this.roughness = m.roughness;
        this.specular = m.specular;
        this.refraction = m.refraction;
        this.metallic = m.metallic;
        this.foamAmount = m.foam;
    }
}
