# Liquid Rust — iOS Metal + Liquid Glass Edition (`.ipa`)

Native **Apple Metal GPU Compute** + **SwiftUI Liquid Glass** real-time 2D/3D Position-Based Fluids (PBF) simulator for iPhone and iPad (`iOS 16.4+`, `120Hz ProMotion`).

## Built IPA Package

- **Compiled iOS IPA**: `dist/liquid-rust-ios.ipa`
- **Bundle ID**: `com.liquidrust.ios`
- **Display Name**: `Liquid Rust`

---

## Sideloading on Arch Linux (`AltServer-Linux`)

`AltServer-Linux` (`~/.local/bin/altserver`) and `libimobiledevice` (`idevice_id`) are already installed in user-space.

When you connect your iPhone or iPad via USB:

1. **Ensure `usbmuxd` USB multiplexer service is installed & active** (one-time Arch setup):
   ```bash
   sudo pacman -S --needed usbmuxd libimobiledevice
   sudo systemctl start usbmuxd
   ```
2. **Plug in your iPhone/iPad**, unlock it, and tap **Trust This Computer**.
3. **Run the sideloader script** (signs `dist/liquid-rust-ios.ipa` with your free Apple ID via Anisette and installs directly onto your device):
   ```bash
   ./sideload_ipa.sh
   ```
4. On your iPhone/iPad, open **Settings → General → VPN & Device Management**, trust your Apple ID developer profile, and enable **Developer Mode** under **Settings → Privacy & Security → Developer Mode** (iOS 16+).

---

## Architecture Highlights

- **Apple Metal 8-Pass GPU Compute (`LiquidRust/Shaders/PbfShaders.metal`)**:
  - 64-thread workgroups (`2 × 32` Apple GPU SIMD-groups) over unified `MTLResourceStorageModeShared` memory (`0` CPU↔GPU copy overhead).
  - Lock-free `atomic_exchange_explicit` spatial hash grid, analytical mirror-floor boundary density support, Akinci bell-spline surface cohesion, XSPH viscosity, and Macklin & Müller vorticity confinement.
  - Dedicated **2D Early-Z** (`particle_frag_2d`) and **3D Spherical Depth** (`particle_frag_3d` in `[0, 1]` Metal NDC) render pipelines + `120Hz ProMotion` (`CADisableMinimumFrameDurationOnPhone`).
- **Liquid Glass UI (`LiquidRust/Sources/LiquidGlassComponents.swift`, `ContentView.swift`)**:
  - Multi-layered `.ultraThinMaterial` capsules & orbs with dual-angle specular refraction rims, spring press physics, SF Symbols, haptics, sliding refractive lens tabs (`matchedGeometryEffect`), and a live-interactive translucent frosted sheet (`presentationBackgroundInteraction(.enabled)`).
  - `CoreMotion` (`CMMotionManager`) hardware tilt gravity + multi-touch pinch-zoom, two-finger pan, and single-finger `Pull` / `Push` / `Orbit`.
