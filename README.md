# 🌊 LiquidRust

A high-performance, real-time 3D fluid simulation engine written in Rust using Position-Based Fluids (PBF), **GPU compute acceleration via Vulkan (`wgpu`)**, hardware-accelerated 3D OpenGL visualization via `eframe` and `glow`, multi-threaded CPU fallback with `rayon`, ASMR slow-motion simulation controls, direct CLI MP4 video rendering, and Stanford PLY sequence export for 3D rendering in Blender.

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3%20%7C%201.4-red.svg)](https://www.vulkan.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ⚡ Quick Install

Install `liquid-rust` (both GUI & CLI) with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/rappar222/liquid-rust/main/install.sh | bash
```

Or download the latest release bundle from [GitHub Releases](https://github.com/rappar222/liquid-rust/releases), extract and run `./install.sh`.

---

## ✨ Features

- **⚡ GPU Acceleration via Vulkan Compute (`wgpu`)**:
  - Full parallel Position-Based Fluids (PBF) pipeline running on GPU compute shaders.
  - 3D uniform spatial hashing and neighbor search directly in WGSL compute passes.
  - Multi-million particle-steps per second on modern GPUs (e.g. AMD Radeon, NVIDIA GeForce, Intel Arc).
  - Automatic detection and seamless fallback to multi-threaded CPU when GPU is not available.

- **⏱ ASMR Slow-Motion & Speed Control**:
  - Real-time simulation speed slider (0.05x to 2.0x) with quick presets (`🎬 0.25x ASMR`, `0.5x Smooth`, `1.0x Realtime`).
  - Enables hypnotic, ultra-smooth fluid splash visualization with high display refresh rates.

- **🛡 Graceful Ctrl+C Video Finalization**:
  - Hitting `Ctrl+C` during CLI MP4 video generation immediately interrupts cleanly and finalizes the MP4 at current progress.
  - Never deletes or corrupts your generated frames—the video is immediately saved and playable.

- **Position-Based Fluids (PBF) Physics**:
  - Momentum-conserving density constraint projection (Macklin & Müller).
  - Dynamic rest density calibration matching hexagonal close-packed fluid lattice.
  - Tensile instability correction and cohesive surface tension formulation.
  - Inelastic wall collisions with normal velocity elimination (no artificial trampoline bounce).
  - Normalized XSPH artificial viscosity.

- **High Performance & Multi-Threading**:
  - $O(N)$ 3D uniform spatial hashing grid with compact linked-list storage.
  - Multi-threaded density, pressure, force, and integration loops with `rayon`.
  - Simulates 10,000+ particles at over 100–250+ FPS on modern CPUs.

- **Interactive 3D Hardware Viewport**:
  - Point sprite sphere rendering with fragment-shader screen-space normal reconstruction and specular Phong lighting.
  - Smooth 3D Orbit Camera (orbit, pan, zoom, view-reset).
  - Multiple color visualization modes:
    - **Velocity Magnitude**
    - **Pressure Field**
    - **Density Deviation**
    - **Height (Y)**
    - **Kinetic Energy**
    - **Uniform Aqua**
  - Perceptual colormaps: **Ocean**, **Turbo (Google Rainbow)**, **Viridis**, **Fire / Magma**.

- **Interactive Fluid Manipulation**:
  - Hold `Shift` + Left Drag to **attract** fluid towards the mouse ray.
  - Hold `Shift` + Right Drag to **repel** fluid.
  - Add and interactively move solid obstacles (spheres, boxes) in real time.

- **Blender / VFX Point Cloud & MP4 Video Export**:
  - Direct CLI export to **H.264 MP4 videos** (`--export-mp4 simulation.mp4`) via hardware-buffered software rasterizer and FFmpeg pipe.
  - Exports single frames or entire animation sequences in standard binary/ASCII **Stanford PLY format** (`.ply`).
  - Includes positions, velocities, density, pressure, and RGB color fields.
  - Easily importable into **Blender** (Geometry Nodes / Point Cloud), MeshLab, Houdini, or CloudCompare for photorealistic fluid meshing and raytracing.

- **Configurable Particle Quality & Smoothness**:
  - Dynamically adjust particle size and density in both GUI and CLI:
    - **Draft**: ~2,500 particles (fastest)
    - **Medium**: ~5,500 particles (balanced default)
    - **High**: ~18,000 particles (smooth, fine liquid details)
    - **Ultra**: ~70,000–120,000 particles (hyper-realistic, fine water)
    - **Cinematic**: ~160,000+ particles (VFX film quality)
    - **Extreme**: 1,000,000+ particles (1M+ GPU mega-simulation with freeze warning modal)
  - ASMR & Slow-Motion controls (16x slow, 4x slow, 2x slow, fine speed slider down to 0.01x).
  - Custom particle radius and lattice spacing sliders for precision control.

- **Headless & Benchmark CLI Modes**:
  - Run simulations in headless server environments without opening a display window.
  - Built-in benchmarking for measuring step times and throughput (`particle-steps/sec`).

---

## 🚀 Installation & Quick Start

### 📦 APT (Ubuntu / Debian / Linux Mint / Pop!_OS)

#### Option 1: Native APT Repository
Add the repository to receive automatic updates via `apt upgrade`:
```bash
curl -fsSL https://rappar222.github.io/liquid-rust/setup-apt.sh | sudo bash
sudo apt update
sudo apt install liquid-rust
```

#### Option 2: Download `.deb` Package
```bash
curl -LO https://github.com/rappar222/liquid-rust/releases/latest/download/liquid-rust_0.3.0_amd64.deb
sudo apt install ./liquid-rust_0.3.0_amd64.deb
```

---

### 🚀 Portable AppImage (Runs on ANY Linux Distro)
No installation required:
```bash
curl -LO https://github.com/rappar222/liquid-rust/releases/latest/download/liquid-rust-x86_64.AppImage
chmod +x liquid-rust-x86_64.AppImage
./liquid-rust-x86_64.AppImage
```

---

### 🏹 Arch Linux (AUR)
Ready-to-use `PKGBUILD` scripts are provided in [`packaging/aur/`](packaging/aur/):
```bash
# Clone the binary package or build from source
cd packaging/aur/liquid-rust-bin
makepkg -si
```

---

### 🦀 Build from Source (Cargo)

Ensure you have Rust and Cargo installed (1.75+):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone the repository and run in release mode for maximum simulation performance:
```bash
git clone https://github.com/rappar222/liquid-rust.git
cd liquid-rust
cargo run --release
```

---

## 🕹 Viewport & Controls

| Action | Control |
| :--- | :--- |
| **Orbit Camera** | Left-Click + Drag in Viewport |
| **Pan Camera** | Right-Click / Middle-Click + Drag |
| **Zoom In / Out** | Scroll Wheel |
| **Attract Fluid** | `Shift` + Left-Click Drag |
| **Repel Fluid** | `Shift` + Right-Click Drag |
| **Play / Pause** | Top Toolbar Button or Space |
| **Step Single Frame** | `⏭ Step` Button |
| **Reset Simulation** | `🔄 Reset` Button |

---

## 📦 Preset Scenarios

Select presets directly from the GUI dropdown or launch them via the CLI:

1. **Dam Break (Classic)**: A column of fluid crashing against the walls of an aquarium.
2. **Double Dam Break (Collision)**: Two fluid blocks colliding head-on in the center.
3. **Droplet Splash into Pool**: High-speed spherical water drop penetrating a resting fluid reservoir.
4. **Fluid onto Obstacle**: Fluid pouring over an interactive solid sphere collider.
5. **Swirling Vortex / Tornado**: Rotational velocity field creating a whirlpool funnel.
6. **Zero-G Droplet Coalescence**: Multiple floating fluid spheres in microgravity merging via surface tension.

---

## 💻 Headless & CLI Usage

`liquid-rust` includes a full CLI powered by `clap`:

```bash
# Export 4 seconds of hyper-realistic Ultra quality fluid at 60 FPS to MP4
cargo run --release -- --export-mp4 realistic_fluid.mp4 --preset dam-break --time 4.0 --fps 60 --quality ultra

# Export 3 seconds of high-speed splash at 60 FPS
cargo run --release -- --export-mp4 splash.mp4 --preset splash --time 3.0 --fps 60 --quality high

# Launch interactive 3D GUI directly with Ultra quality
cargo run --release -- --quality ultra

# Headless benchmark on Dam Break preset with ultra quality
cargo run --release -- --headless --benchmark --frames 100 --quality ultra

# Simulate Droplet Splash and export PLY files to a folder
cargo run --release -- --headless --preset splash --frames 150 --export-ply ./renders

# Run with custom particle radius and spacing for ultra-fine liquid
cargo run --release -- --particle-radius 0.008 --spacing 0.014 --time 2.5 --fps 60
```

### Available CLI Flags

```
Options:
      --headless                Run in headless mode without opening GUI window
      --preset <NAME>           Scene preset: dam-break, double-dam, splash, obstacle, tornado, zero-g [default: dam-break]
  -q, --quality <QUALITY>       Particle quality: draft, medium, high, ultra, cinematic [default: medium]
      --particle-radius <R>     Custom particle radius override (smaller = finer fluid)
      --spacing <S>             Custom particle spacing override
  -t, --time <SECONDS>          Total simulation duration in seconds (e.g. --time 4.0). Automatically calculates total frames = time * fps
      --fps <FPS>               Framerate for video export and simulation stepping [default: 60]
      --frames <FRAMES>         Number of simulation frames to run if --time is not set [default: 120]
      --dt <DT>                 Custom time step dt per frame (defaults to 1.0 / fps)
      --export-mp4 <PATH>       Export simulation video to MP4 file via ffmpeg (e.g. output.mp4)
      --video-width <WIDTH>     Video export width in pixels [default: 1280]
      --video-height <HEIGHT>   Video export height in pixels [default: 720]
      --export-ply <DIR>        Directory to export PLY point cloud frames
      --benchmark               Print performance benchmark statistics
  -h, --help                    Print help
  -V, --version                 Print version
```

---

## 🎬 Importing PLY Sequences into Blender

1. In Blender, enable **Import-Export: Stanford PLY format**.
2. Go to **File > Import > Stanford PLY (.ply)**.
3. Use a **Mesh to Volume** or **Geometry Nodes (Points to Volume -> Volume to Mesh)** modifier chain to generate high-resolution water surfaces with glass/water shaders and cycles caustics!

---

## 🏗 Architecture

```
liquid-rust/
├── src/
│   ├── app.rs            # egui + glow interactive UI application
│   ├── lib.rs            # Library exports and public API
│   ├── main.rs           # CLI argument parsing and entrypoint
│   ├── math.rs           # 3D math, AABB, ray casting, camera unprojection
│   ├── export/
│   │   └── ply.rs        # High-performance Stanford PLY writer (binary & ascii)
│   ├── render/
│   │   ├── camera.rs     # 3D smooth orbit camera
│   │   ├── color_map.rs  # Perceptual colormaps (Ocean, Turbo, Viridis, Fire)
│   │   └── gl_renderer.rs# OpenGL point sprite shader with 3D normal reconstruction
│   └── sph/
│       ├── collider.rs   # Dynamic solid colliders (Sphere, Box, Cylinder)
│       ├── kernel.rs     # Poly6, Spiky, Viscosity & Wendland C2 kernels
│       ├── particle.rs   # Particle state representations
│       ├── scene.rs      # Preset scenario generators
│       ├── solver.rs     # Weakly Compressible SPH solver with Rayon parallelism
│       └── spatial.rs    # O(N) 3D spatial hash grid with 27-neighbor traversal
└── tests/
    └── fluid_tests.rs    # Unit and physics verification test suite
```

---

## 🧪 Testing

Run the test suite to verify kernel behaviors, spatial hashing, collision response, and PLY exports:

```bash
cargo test
```

---

## 📄 License

Licensed under the [MIT License](LICENSE).
