import Foundation
import SwiftUI
import simd

struct MaterialPreset: Identifiable {
    let id: Int
    let name: String
    let shallowHex: UInt32
    let deepHex: UInt32
    let ior: Float
    let roughness: Float
    let specular: Float
    let refraction: Float
    let metallic: Float
    let foam: Float

    var shallowSIMD: SIMD3<Float> {
        SIMD3<Float>(
            Float((shallowHex >> 16) & 0xFF) / 255.0,
            Float((shallowHex >> 8) & 0xFF) / 255.0,
            Float(shallowHex & 0xFF) / 255.0
        )
    }

    var deepSIMD: SIMD3<Float> {
        SIMD3<Float>(
            Float((deepHex >> 16) & 0xFF) / 255.0,
            Float((deepHex >> 8) & 0xFF) / 255.0,
            Float(deepHex & 0xFF) / 255.0
        )
    }

    var shallowColor: Color {
        Color(
            red: Double((shallowHex >> 16) & 0xFF) / 255.0,
            green: Double((shallowHex >> 8) & 0xFF) / 255.0,
            blue: Double(shallowHex & 0xFF) / 255.0
        )
    }

    var deepColor: Color {
        Color(
            red: Double((deepHex >> 16) & 0xFF) / 255.0,
            green: Double((deepHex >> 8) & 0xFF) / 255.0,
            blue: Double(deepHex & 0xFF) / 255.0
        )
    }
}

final class SimulationConfig: ObservableObject {

    static let presetNames: [String] = [
        "Dam Break",
        "Double Dam",
        "Droplet Splash",
        "Obstacle Drop",
        "Vortex Swirl",
        "Zero-G Blobs"
    ]

    static let qualityNames: [String] = [
        "Draft",
        "Balanced",
        "High",
        "Ultra",
        "Extreme"
    ]

    static let materials: [MaterialPreset] = [
        MaterialPreset(id: 0, name: "Clear Water",   shallowHex: 0x38BDF8, deepHex: 0x0C3B8A, ior: 1.333, roughness: 0.045, specular: 1.35, refraction: 0.42, metallic: 0.00, foam: 0.35),
        MaterialPreset(id: 1, name: "Deep Ocean",    shallowHex: 0x2199D6, deepHex: 0x031733, ior: 1.340, roughness: 0.065, specular: 1.30, refraction: 0.38, metallic: 0.00, foam: 0.55),
        MaterialPreset(id: 2, name: "Lagoon",        shallowHex: 0x3DE8DB, deepHex: 0x055261, ior: 1.335, roughness: 0.040, specular: 1.40, refraction: 0.48, metallic: 0.00, foam: 0.25),
        MaterialPreset(id: 3, name: "Molten Amber",  shallowHex: 0xFCB82E, deepHex: 0x5C1403, ior: 1.540, roughness: 0.055, specular: 1.45, refraction: 0.52, metallic: 0.00, foam: 0.12),
        MaterialPreset(id: 4, name: "Crimson Wine",  shallowHex: 0xE12D58, deepHex: 0x2B0410, ior: 1.360, roughness: 0.050, specular: 1.30, refraction: 0.32, metallic: 0.00, foam: 0.10),
        MaterialPreset(id: 5, name: "Emerald Gel",   shallowHex: 0x33EB85, deepHex: 0x03381F, ior: 1.520, roughness: 0.042, specular: 1.50, refraction: 0.48, metallic: 0.00, foam: 0.15),
        MaterialPreset(id: 6, name: "Creamy Milk",   shallowHex: 0xFAFCFF, deepHex: 0xCBD5E1, ior: 1.350, roughness: 0.180, specular: 0.85, refraction: 0.08, metallic: 0.00, foam: 0.05),
        MaterialPreset(id: 7, name: "Liquid Chrome", shallowHex: 0xE6ECF5, deepHex: 0x5A677D, ior: 2.400, roughness: 0.025, specular: 2.30, refraction: 0.00, metallic: 0.94, foam: 0.00)
    ]

    // Transport & Scene
    @Published var isPlaying: Bool = true
    @Published var is2D: Bool = true
    @Published var preset: Int = 0
    @Published var quality: Int = 0
    @Published var simSpeed: Float = 1.0
    @Published var touchTool: Int = 0 // 0: Pull, 1: Push, 2: Orbit

    // Physics
    @Published var tiltGravityEnabled: Bool = false
    @Published var vorticity: Float = 0.045
    @Published var surfaceTension: Float = 0.030
    @Published var viscosity: Float = 0.035
    @Published var gravityMag: Float = 9.81

    // Sphere Obstacle
    @Published var obstacleEnabled: Bool = false
    @Published var obstacleX: Float = 0.0
    @Published var obstacleY: Float = 0.45
    @Published var obstacleR: Float = 0.20

    // Optical Shading & Procedural Texture
    @Published var materialIndex: Int = 0
    @Published var colorMode: Int = 3 // 3: Optical Liquid, 0: Velocity, 1: Pressure, 2: Density
    @Published var colorMap: Int = 0  // 0: Ocean, 1: Turbo, 2: Viridis, 3: Magma
    @Published var litSpheres: Bool = true
    @Published var texMode: Int = 1   // 0: Smooth, 1: Ripples, 2: Caustics, 3: Foam, 4: Grid
    @Published var texStrength: Float = 0.28
    @Published var foamAmount: Float = 0.35
    @Published var ior: Float = 1.333
    @Published var roughness: Float = 0.045
    @Published var specular: Float = 1.35
    @Published var refraction: Float = 0.42
    @Published var metallic: Float = 0.00

    var shallowRGB: SIMD3<Float> = SIMD3<Float>(0.22, 0.74, 0.97)
    var deepRGB: SIMD3<Float> = SIMD3<Float>(0.05, 0.23, 0.54)

    // Telemetry published back to SwiftUI HUD
    @Published var gpuName: String = "Apple GPU"
    @Published var fps: Int = 120
    @Published var particleCount: Int = 0
    @Published var stepMs: Float = 0.4
    @Published var smoothingH: Float = 0.057

    // Non-published render-loop flags & rays
    var isPortrait: Bool = true
    var reloadRequested: Bool = true
    var stepOnceRequested: Bool = false
    var resetCameraRequested: Bool = false

    var activeTouchForceMode: Int32 = 0
    var rayOrig: SIMD3<Float> = SIMD3<Float>(0, 0, 2)
    var rayDir: SIMD3<Float> = SIMD3<Float>(0, 0, -1)
    var tiltDir: SIMD3<Float> = SIMD3<Float>(0, -1, 0)

    init() {
        applyMaterialPreset(0)
    }

    func applyMaterialPreset(_ index: Int) {
        guard index >= 0 && index < Self.materials.count else { return }
        let m = Self.materials[index]
        materialIndex = index
        colorMode = 3
        shallowRGB = m.shallowSIMD
        deepRGB = m.deepSIMD
        ior = m.ior
        roughness = m.roughness
        specular = m.specular
        refraction = m.refraction
        metallic = m.metallic
        foamAmount = m.foam
    }

    func selectPreset(_ idx: Int) {
        preset = idx
        let hasObs = (idx == 3)
        obstacleEnabled = hasObs
        if idx == 5 {
            gravityMag = 0.0
        } else if gravityMag < 0.1 {
            gravityMag = 9.81
        }
        reloadRequested = true
    }

    func toggleDimension(_ mode2D: Bool) {
        guard is2D != mode2D else { return }
        is2D = mode2D
        reloadRequested = true
        resetCameraRequested = true
    }
}
