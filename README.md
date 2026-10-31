# 🌊 LiquidRust

A high-performance, real-time 3D fluid simulation engine written in Rust using Smoothed Particle Hydrodynamics (SPH), hardware-accelerated OpenGL visualization via `eframe` and `glow`, multi-threaded CPU physics with `rayon`, and Stanford PLY sequence export for 3D rendering in Blender.

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ✨ Features

- **Weakly Compressible SPH (WCSPH)**:
  - Momentum-conserving symmetric pressure gradient.
  - Tait equation of state (EOS) and linear compressibility models.
  - Desbrun/Müller spiky smoothing kernels to prevent particle clustering.
  - Akinci/Morris cohesive surface tension formulation for droplet formation.
  - Dynamic shear viscosity and artificial dissipation.
  - Symplectic Euler integration with configurable substepping.

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
    - **High**: ~16,000 particles (smooth, fine liquid details)
    - **Ultra**: ~40,000+ particles (fine film-quality water)
  - Custom particle radius and lattice spacing sliders for precision control.

- **Headless & Benchmark CLI Modes**:
  - Run simulations in headless server environments without opening a display window.
  - Built-in benchmarking for measuring step times and throughput (`particle-steps/sec`).

---

## 🚀 Quick Start

### Prerequisites

Ensure you have Rust and Cargo installed (1.75+):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Installation & Launch

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
# Export simulation video directly to MP4
cargo run --release -- --export-mp4 water_splash.mp4 --preset splash --frames 180 --quality high

# Run high-resolution simulation with fine, smooth particles
cargo run --release -- --quality high

# Headless benchmark on Dam Break preset with ultra quality
cargo run --release -- --headless --benchmark --frames 100 --quality ultra

# Simulate Droplet Splash and export PLY files to a folder
cargo run --release -- --headless --preset splash --frames 150 --export-ply ./renders

# Run with custom particle radius and spacing for ultra-fine liquid
cargo run --release -- --particle-radius 0.014 --spacing 0.024 --frames 100
```

### Available CLI Flags

```
Options:
      --headless                Run in headless mode without opening GUI window
      --preset <NAME>           Scene preset: dam-break, double-dam, splash, obstacle, tornado, zero-g [default: dam-break]
  -q, --quality <QUALITY>       Particle quality: draft, medium, high, ultra [default: medium]
      --particle-radius <R>     Custom particle radius override (smaller = finer fluid)
      --spacing <S>             Custom particle spacing override
      --frames <FRAMES>         Number of simulation frames to run in headless mode [default: 120]
      --dt <DT>                 Time step dt per frame in seconds [default: 0.016]
      --export-mp4 <PATH>       Export simulation video to MP4 file via ffmpeg (e.g. output.mp4)
      --video-width <WIDTH>     Video export width in pixels [default: 1280]
      --video-height <HEIGHT>   Video export height in pixels [default: 720]
      --fps <FPS>               Video export framerate in FPS [default: 60]
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
