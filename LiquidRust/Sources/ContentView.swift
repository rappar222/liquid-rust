import SwiftUI

struct ContentView: View {
    @StateObject private var cfg = SimulationConfig()
    @State private var showInspectorSheet: Bool = false
    @State private var expandTelemetry: Bool = false
    @State private var selectedSheetTab: Int = 0

    var body: some View {
        ZStack {
            // Full-bleed 120Hz Apple Metal Fluid Simulation
            MetalFluidViewport(cfg: cfg)
                .ignoresSafeArea()

            // Floating Liquid Glass HUD
            VStack(spacing: 0) {
                topLiquidGlassBar
                    .padding(.horizontal, 16)
                    .padding(.top, 6)

                Spacer(minLength: 0)

                if !showInspectorSheet {
                    bottomLiquidGlassDock
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .animation(.spring(response: 0.34, dampingFraction: 0.80), value: showInspectorSheet)
        }
        .preferredColorScheme(.dark)
        .onChange(of: cfg.tiltGravityEnabled) { enabled in
            OrientationController.setLocked(enabled)
        }
        .sheet(isPresented: $showInspectorSheet) {
            LiquidGlassInspectorSheet(cfg: cfg, selectedTab: $selectedSheetTab)
                .presentationDetents([.fraction(0.54), .large])
                .presentationDragIndicator(.visible)
                .presentationCornerRadius(34)
                .presentationBackground(.ultraThinMaterial)
                .presentationBackgroundInteraction(.enabled(upThrough: .fraction(0.54)))
        }
    }

    // MARK: - Top Floating Liquid Glass Bar

    private var topLiquidGlassBar: some View {
        HStack(spacing: 10) {
            // Refractive Telemetry Capsule (tap to expand GPU detail)
            Button {
                GlassHaptics.light()
                withAnimation(.spring(response: 0.30, dampingFraction: 0.78)) {
                    expandTelemetry.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Circle()
                        .fill(Color(red: 0.32, green: 0.95, blue: 0.62))
                        .frame(width: 8, height: 8)
                        .shadow(color: Color.green.opacity(0.7), radius: 5)

                    Text(telemetryLabel)
                        .font(.system(size: 12.5, weight: .semibold, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(Color.white.opacity(0.95))
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                .padding(.horizontal, 14)
                .frame(height: 40)
                .liquidGlass(cornerRadius: 20)
            }
            .buttonStyle(LiquidGlassPressStyle())

            Spacer(minLength: 4)

            // 2D / 3D Liquid Glass Capsule
            Button {
                GlassHaptics.selection()
                cfg.toggleDimension(!cfg.is2D)
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: cfg.is2D ? "square.dashed" : "cube.transparent")
                        .font(.system(size: 13.5, weight: .semibold))
                    Text(cfg.is2D ? "2D" : "3D")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 13)
                .frame(height: 40)
                .liquidGlass(cornerRadius: 20, isActive: !cfg.is2D)
            }
            .buttonStyle(LiquidGlassPressStyle())

            // Hardware Gyro / Gravity Tilt Orb
            LiquidGlassOrbButton(
                systemImage: "gyroscope",
                size: 40,
                isActive: cfg.tiltGravityEnabled
            ) {
                cfg.tiltGravityEnabled.toggle()
            }

            // Reset Scene Orb
            LiquidGlassOrbButton(
                systemImage: "arrow.counterclockwise",
                size: 40
            ) {
                cfg.reloadRequested = true
            }
        }
    }

    private var telemetryLabel: String {
        let pts = cfg.particleCount >= 1000
            ? String(format: "%.1fk", Double(cfg.particleCount) / 1000.0)
            : "\(cfg.particleCount)"
        if expandTelemetry {
            return String(format: "%@ · %d FPS · %@ · %.1fms", cfg.gpuName, cfg.fps, pts, cfg.stepMs)
        } else {
            return "\(cfg.fps) FPS · \(pts)"
        }
    }

    // MARK: - Bottom Thumb-Zone Liquid Glass Dock + Luminous Play Orb

    private var bottomLiquidGlassDock: some View {
        HStack(spacing: 10) {
            LiquidGlassSegmentedControl(
                items: [
                    ("dot.radiowaves.right", "Pull"),
                    ("burst", "Push"),
                    ("move.3d", "Orbit")
                ],
                selection: $cfg.touchTool
            )

            // Open Liquid Glass Inspector Sheet Orb
            LiquidGlassOrbButton(
                systemImage: "slider.horizontal.3",
                size: 50,
                isActive: showInspectorSheet
            ) {
                showInspectorSheet = true
            }

            // Primary Luminous Liquid Glass Play / Pause Orb
            Button {
                GlassHaptics.medium()
                cfg.isPlaying.toggle()
            } label: {
                Image(systemName: cfg.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 54, height: 54)
                    .liquidGlass(
                        cornerRadius: 27,
                        isActive: cfg.isPlaying,
                        tint: Color(red: 0.22, green: 0.76, blue: 1.0)
                    )
            }
            .buttonStyle(LiquidGlassPressStyle())
        }
    }
}

// MARK: - Interactive Translucent Liquid Glass Bottom Sheet

private struct LiquidGlassInspectorSheet: View {
    @ObservedObject var cfg: SimulationConfig
    @Binding var selectedTab: Int
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 14) {
            // Sheet Header + Quick Transport Pills
            HStack(spacing: 10) {
                Text("Liquid Rust")
                    .font(.system(size: 19, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)

                Spacer()

                // Step Single Frame
                Button {
                    GlassHaptics.light()
                    cfg.isPlaying = false
                    cfg.stepOnceRequested = true
                } label: {
                    Label("Step", systemImage: "forward.frame.fill")
                        .font(.system(size: 12.5, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .frame(height: 34)
                        .liquidGlass(cornerRadius: 17)
                }
                .buttonStyle(LiquidGlassPressStyle())

                // Recenter Camera
                Button {
                    GlassHaptics.light()
                    cfg.resetCameraRequested = true
                } label: {
                    Label("Fit", systemImage: "viewfinder")
                        .font(.system(size: 12.5, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .frame(height: 34)
                        .liquidGlass(cornerRadius: 17)
                }
                .buttonStyle(LiquidGlassPressStyle())

                LiquidGlassOrbButton(systemImage: "chevron.down", size: 34) {
                    dismiss()
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 14)

            // Sliding Glass Tab Lens
            LiquidGlassSegmentedControl(
                items: [
                    ("sparkles", "Scene"),
                    ("waveform.path", "Physics"),
                    ("drop.halffull", "Appearance")
                ],
                selection: $selectedTab
            )
            .padding(.horizontal, 18)

            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 14) {
                    if selectedTab == 0 {
                        scenePane
                    } else if selectedTab == 1 {
                        physicsPane
                    } else {
                        appearancePane
                    }
                }
                .padding(.horizontal, 18)
                .padding(.bottom, 28)
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: Tab 1 — Scene

    private var scenePane: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Scene Preset")
                .font(.system(size: 12.5, weight: .semibold, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.65))
                .padding(.leading, 4)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(SimulationConfig.presetNames.indices, id: \.self) { idx in
                        LiquidGlassChip(
                            title: SimulationConfig.presetNames[idx],
                            isSelected: cfg.preset == idx
                        ) {
                            cfg.selectPreset(idx)
                        }
                    }
                }
                .padding(.vertical, 2)
            }

            LiquidGlassCard("Simulation Space & Quality") {
                HStack(spacing: 8) {
                    LiquidGlassChip(title: "2D Planar", isSelected: cfg.is2D) {
                        cfg.toggleDimension(true)
                    }
                    LiquidGlassChip(title: "3D Volumetric", isSelected: !cfg.is2D) {
                        cfg.toggleDimension(false)
                    }
                }

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(SimulationConfig.qualityNames.indices, id: \.self) { idx in
                            LiquidGlassChip(
                                title: SimulationConfig.qualityNames[idx],
                                isSelected: cfg.quality == idx
                            ) {
                                cfg.quality = idx
                                cfg.reloadRequested = true
                            }
                        }
                    }
                    .padding(.vertical, 2)
                }

                Text(String(format: "Kernel smoothing radius h: %.3f m", cfg.smoothingH))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.55))
            }

            LiquidGlassCard("Time Scale") {
                HStack(spacing: 8) {
                    LiquidGlassChip(title: "0.25× Slow", isSelected: abs(cfg.simSpeed - 0.25) < 0.02) {
                        cfg.simSpeed = 0.25
                    }
                    LiquidGlassChip(title: "0.5× Smooth", isSelected: abs(cfg.simSpeed - 0.50) < 0.02) {
                        cfg.simSpeed = 0.50
                    }
                    LiquidGlassChip(title: "1.0× Real", isSelected: abs(cfg.simSpeed - 1.00) < 0.02) {
                        cfg.simSpeed = 1.00
                    }
                }
                LiquidGlassSlider(label: "Playback Speed", value: $cfg.simSpeed, range: 0.05...2.0, format: "%.2f×")
            }

            LiquidGlassCard {
                Toggle(isOn: $cfg.obstacleEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Sphere Collider")
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white)
                        Text("Interactive solid obstacle inside chamber")
                            .font(.system(size: 12, design: .rounded))
                            .foregroundStyle(Color.white.opacity(0.58))
                    }
                }
                .tint(Color(red: 0.30, green: 0.80, blue: 1.0))

                if cfg.obstacleEnabled {
                    LiquidGlassSlider(label: "Horizontal X", value: $cfg.obstacleX, range: -0.55...0.55, format: "%.2f m")
                    LiquidGlassSlider(label: "Vertical Y", value: $cfg.obstacleY, range: 0.15...1.40, format: "%.2f m")
                    LiquidGlassSlider(label: "Sphere Radius", value: $cfg.obstacleR, range: 0.08...0.40, format: "%.2f m")
                }
            }
        }
    }

    // MARK: Tab 2 — Physics

    private var physicsPane: some View {
        VStack(spacing: 14) {
            LiquidGlassCard {
                Toggle(isOn: $cfg.tiltGravityEnabled) {
                    HStack(spacing: 10) {
                        Image(systemName: "gyroscope")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(Color(red: 0.36, green: 0.84, blue: 1.0))
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Motion Tilt Gravity")
                                .font(.system(size: 14.5, weight: .semibold, design: .rounded))
                                .foregroundStyle(.white)
                            Text("Tilt your iPhone or iPad to pour and slosh liquid")
                                .font(.system(size: 12, design: .rounded))
                                .foregroundStyle(Color.white.opacity(0.60))
                        }
                    }
                }
                .tint(Color(red: 0.30, green: 0.80, blue: 1.0))
            }

            LiquidGlassCard("Fluid Dynamics") {
                LiquidGlassSlider(label: "Vorticity Swirl", value: $cfg.vorticity, range: 0.0...0.18, format: "%.3f")
                LiquidGlassSlider(label: "Surface Cohesion", value: $cfg.surfaceTension, range: 0.0...0.15, format: "%.3f")
                LiquidGlassSlider(label: "Viscosity (XSPH)", value: $cfg.viscosity, range: 0.0...0.30, format: "%.3f")
                LiquidGlassSlider(label: "Gravity Strength", value: $cfg.gravityMag, range: 0.0...25.0, format: "%.1f m/s²")
            }
        }
    }

    // MARK: Tab 3 — Appearance

    private var appearancePane: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Liquid Material")
                .font(.system(size: 12.5, weight: .semibold, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.65))
                .padding(.leading, 4)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(SimulationConfig.materials) { mat in
                        LiquidDropletSwatch(
                            preset: mat,
                            isSelected: cfg.materialIndex == mat.id && cfg.colorMode == 3
                        ) {
                            cfg.applyMaterialPreset(mat.id)
                        }
                    }
                }
                .padding(.vertical, 2)
            }

            LiquidGlassCard("Shading Mode") {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        LiquidGlassChip(title: "Optical Liquid", isSelected: cfg.colorMode == 3) { cfg.colorMode = 3 }
                        LiquidGlassChip(title: "Velocity", isSelected: cfg.colorMode == 0) { cfg.colorMode = 0 }
                        LiquidGlassChip(title: "Pressure", isSelected: cfg.colorMode == 1) { cfg.colorMode = 1 }
                        LiquidGlassChip(title: "Density", isSelected: cfg.colorMode == 2) { cfg.colorMode = 2 }
                    }
                }

                if cfg.colorMode != 3 {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            LiquidGlassChip(title: "Ocean", isSelected: cfg.colorMap == 0) { cfg.colorMap = 0 }
                            LiquidGlassChip(title: "Turbo", isSelected: cfg.colorMap == 1) { cfg.colorMap = 1 }
                            LiquidGlassChip(title: "Viridis", isSelected: cfg.colorMap == 2) { cfg.colorMap = 2 }
                            LiquidGlassChip(title: "Magma", isSelected: cfg.colorMap == 3) { cfg.colorMap = 3 }
                        }
                    }
                }

                Toggle("Fresnel Sky-Rim & Sun Specular", isOn: $cfg.litSpheres)
                    .font(.system(size: 13.5, weight: .medium, design: .rounded))
                    .foregroundStyle(.white)
                    .tint(Color(red: 0.30, green: 0.80, blue: 1.0))
            }

            LiquidGlassCard("Surface Texture") {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        LiquidGlassChip(title: "Ripples", isSelected: cfg.texMode == 1) { cfg.texMode = 1 }
                        LiquidGlassChip(title: "Caustics", isSelected: cfg.texMode == 2) { cfg.texMode = 2 }
                        LiquidGlassChip(title: "Foam", isSelected: cfg.texMode == 3) { cfg.texMode = 3 }
                        LiquidGlassChip(title: "Grid", isSelected: cfg.texMode == 4) { cfg.texMode = 4 }
                        LiquidGlassChip(title: "Smooth", isSelected: cfg.texMode == 0) { cfg.texMode = 0 }
                    }
                }
                LiquidGlassSlider(label: "Pattern Depth", value: $cfg.texStrength, range: 0.0...1.0, format: "%.2f")
                LiquidGlassSlider(label: "Crest Whitecaps", value: $cfg.foamAmount, range: 0.0...1.0, format: "%.2f")
            }

            LiquidGlassCard("Optical Light Behavior") {
                LiquidGlassSlider(label: "Fresnel IOR", value: $cfg.ior, range: 1.01...2.50, format: "%.3f")
                LiquidGlassSlider(label: "GGX Roughness", value: $cfg.roughness, range: 0.015...0.80, format: "%.3f")
                LiquidGlassSlider(label: "Sun Specular", value: $cfg.specular, range: 0.0...3.0, format: "%.2f")
                LiquidGlassSlider(label: "Subsurface Glow", value: $cfg.refraction, range: 0.0...1.2, format: "%.2f")
                LiquidGlassSlider(label: "Metallic", value: $cfg.metallic, range: 0.0...1.0, format: "%.2f")
            }
        }
    }
}
