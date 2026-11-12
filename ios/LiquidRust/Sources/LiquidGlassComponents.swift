import SwiftUI
import UIKit

enum GlassHaptics {
    static func light() {
        let gen = UIImpactFeedbackGenerator(style: .light)
        gen.impactOccurred()
    }

    static func medium() {
        let gen = UIImpactFeedbackGenerator(style: .medium)
        gen.impactOccurred()
    }

    static func selection() {
        let gen = UISelectionFeedbackGenerator()
        gen.selectionChanged()
    }
}

/// Tactile spring scale + specular flare button style for Liquid Glass controls
struct LiquidGlassPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.94 : 1.0)
            .brightness(configuration.isPressed ? 0.07 : 0.0)
            .animation(.spring(response: 0.26, dampingFraction: 0.68), value: configuration.isPressed)
    }
}

/// Multi-layered Apple Liquid Glass Capsule / Rounded-Rect Surface Modifier
struct LiquidGlassCapsule: ViewModifier {
    var cornerRadius: CGFloat = 28
    var isActive: Bool = false
    var tintColor: Color = Color(red: 0.30, green: 0.80, blue: 1.0)

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        content
            .background(
                ZStack {
                    shape.fill(.ultraThinMaterial)

                    // Inner refractive glaze
                    shape.fill(
                        LinearGradient(
                            colors: isActive
                                ? [tintColor.opacity(0.38), tintColor.opacity(0.16), Color.white.opacity(0.10)]
                                : [Color.white.opacity(0.13), Color.white.opacity(0.03), tintColor.opacity(0.05)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                }
            )
            .overlay(
                // Dual-angle specular refraction rim
                shape.strokeBorder(
                    LinearGradient(
                        stops: [
                            .init(color: Color.white.opacity(isActive ? 0.78 : 0.52), location: 0.0),
                            .init(color: Color.white.opacity(0.12), location: 0.42),
                            .init(color: tintColor.opacity(isActive ? 0.65 : 0.28), location: 1.0)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: isActive ? 1.25 : 0.9
                )
            )
            .shadow(
                color: isActive ? tintColor.opacity(0.28) : Color.black.opacity(0.34),
                radius: isActive ? 14 : 12,
                x: 0,
                y: 6
            )
    }
}

extension View {
    func liquidGlass(cornerRadius: CGFloat = 28, isActive: Bool = false, tint: Color = Color(red: 0.30, green: 0.80, blue: 1.0)) -> some View {
        modifier(LiquidGlassCapsule(cornerRadius: cornerRadius, isActive: isActive, tintColor: tint))
    }
}

/// Circular Liquid Glass Orb Icon Button
struct LiquidGlassOrbButton: View {
    let systemImage: String
    var size: CGFloat = 42
    var isActive: Bool = false
    var tint: Color = Color(red: 0.30, green: 0.80, blue: 1.0)
    let action: () -> Void

    var body: some View {
        Button {
            GlassHaptics.light()
            action()
        } label: {
            Image(systemName: systemImage)
                .font(.system(size: size * 0.40, weight: .semibold))
                .foregroundStyle(isActive ? Color.white : Color.white.opacity(0.92))
                .frame(width: size, height: size)
                .liquidGlass(cornerRadius: size * 0.5, isActive: isActive, tint: tint)
        }
        .buttonStyle(LiquidGlassPressStyle())
    }
}

/// Sliding Refractive Lens Segmented Control
struct LiquidGlassSegmentedControl: View {
    let items: [(icon: String?, title: String)]
    @Binding var selection: Int
    @Namespace private var lensNamespace

    var body: some View {
        HStack(spacing: 4) {
            ForEach(items.indices, id: \.self) { idx in
                let active = (selection == idx)
                Button {
                    guard selection != idx else { return }
                    GlassHaptics.selection()
                    withAnimation(.spring(response: 0.32, dampingFraction: 0.76)) {
                        selection = idx
                    }
                } label: {
                    HStack(spacing: 5) {
                        if let ic = items[idx].icon {
                            Image(systemName: ic)
                                .font(.system(size: 13, weight: .semibold))
                        }
                        Text(items[idx].title)
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .lineLimit(1)
                    }
                    .foregroundStyle(active ? Color.white : Color.white.opacity(0.68))
                    .frame(maxWidth: .infinity)
                    .frame(height: 40)
                    .background {
                        if active {
                            Capsule(style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: [
                                            Color.white.opacity(0.28),
                                            Color(red: 0.25, green: 0.78, blue: 1.0).opacity(0.28)
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .overlay(
                                    Capsule(style: .continuous)
                                        .strokeBorder(
                                            LinearGradient(
                                                colors: [Color.white.opacity(0.75), Color.white.opacity(0.15)],
                                                startPoint: .top,
                                                endPoint: .bottom
                                            ),
                                            lineWidth: 1.0
                                        )
                                )
                                .shadow(color: Color.cyan.opacity(0.22), radius: 8, x: 0, y: 2)
                                .matchedGeometryEffect(id: "glass_lens", in: lensNamespace)
                        }
                    }
                }
                .buttonStyle(.plain)
            }
        }
        .padding(5)
        .liquidGlass(cornerRadius: 26)
    }
}

/// Liquid Glass Choice Pill
struct LiquidGlassChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button {
            GlassHaptics.selection()
            action()
        } label: {
            HStack(spacing: 5) {
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                }
                Text(title)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .lineLimit(1)
            }
            .foregroundStyle(isSelected ? Color.white : Color.white.opacity(0.82))
            .padding(.horizontal, isSelected ? 13 : 15)
            .frame(height: 36)
            .liquidGlass(cornerRadius: 18, isActive: isSelected)
        }
        .buttonStyle(LiquidGlassPressStyle())
    }
}

/// Optical Liquid Droplet Swatch Pill
struct LiquidDropletSwatch: View {
    let preset: MaterialPreset
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button {
            GlassHaptics.selection()
            action()
        } label: {
            HStack(spacing: 8) {
                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [preset.shallowColor, preset.deepColor],
                                center: .topLeading,
                                startRadius: 2,
                                endRadius: 18
                            )
                        )
                        .frame(width: 20, height: 20)
                    Circle()
                        .fill(Color.white.opacity(0.78))
                        .frame(width: 5, height: 5)
                        .offset(x: -3.5, y: -3.5)
                    Circle()
                        .strokeBorder(Color.white.opacity(0.45), lineWidth: 0.8)
                        .frame(width: 20, height: 20)
                }

                Text(preset.name)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(isSelected ? Color.white : Color.white.opacity(0.85))
                    .lineLimit(1)
            }
            .padding(.leading, 10)
            .padding(.trailing, 14)
            .frame(height: 40)
            .liquidGlass(cornerRadius: 20, isActive: isSelected, tint: preset.shallowColor)
        }
        .buttonStyle(LiquidGlassPressStyle())
    }
}

/// Frosted Glass Card Container inside the Liquid Glass Sheet
struct LiquidGlassCard<Content: View>: View {
    let title: String?
    @ViewBuilder let content: Content

    init(_ title: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let title = title {
                Text(title)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.92))
            }
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Color.white.opacity(0.065))
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.32), Color.white.opacity(0.06)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 0.85
                        )
                )
        )
    }
}

/// Clean Liquid Glass Parameter Slider Row
struct LiquidGlassSlider: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var format: String = "%.2f"

    var body: some View {
        VStack(spacing: 5) {
            HStack {
                Text(label)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.84))
                Spacer()
                Text(String(format: format, value))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Color(red: 0.65, green: 0.90, blue: 1.0))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.white.opacity(0.10)))
            }
            Slider(
                value: Binding(
                    get: { Double(value) },
                    set: { value = Float($0) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound)
            )
            .tint(Color(red: 0.36, green: 0.82, blue: 1.0))
        }
    }
}
