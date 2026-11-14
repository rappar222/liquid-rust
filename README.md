# 🌊 LiquidRust

A high-performance, real-time 3D fluid simulation engine written in Rust using Position-Based Fluids (PBF), **GPU compute acceleration via Vulkan (`wgpu`)**, hardware-accelerated 3D OpenGL visualization via `eframe` and `glow`, multi-threaded CPU fallback with `rayon`, ASMR slow-motion simulation controls, direct CLI MP4 video rendering, and Stanford PLY sequence export for 3D rendering in Blender.

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3%20%7C%201.4-red.svg)](https://www.vulkan.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ⚡ Quick Install (`/bin/liquid-rust`)

Install `liquid-rust` system-wide into `/bin` (both GUI & CLI, runnable from anywhere) with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/rappar222/liquid-rust/main/install.sh | bash
```

Once installed, run `liquid-rust` from any directory — and update anytime via the built-in Studio GUI updater or CLI (`liquid-rust --update`).

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

- **Interactive 3D Studio Viewport & Environments**:
  - Point sprite sphere rendering with world-anchored sun lighting, screen-space normal reconstruction, and aerated whitewater crest highlights.
  - Reconstructed 3D Marching-Cubes **Surface Mesh** mode with Beer-Lambert absorption, Fresnel refraction, caustics, and procedural ripples.
  - Studio sky & lighting presets: **Studio Dark**, **Daylight Sky**, **Golden Sunset**, **Arctic Lab**, and **Abyssal Deep**, plus exposure, floor grid, wireframe, FOV, and turntable orbit controls.
  - Color visualization modes: **Velocity**, **Pressure**, **Density**, **Height (Y)**, **Kinetic Energy**, **Foam / Whitewater**, **Vorticity / Swirl**, and **Uniform Aqua**.
  - Perceptual colormaps: **Ocean**, **Turbo**, **Viridis**, **Magma / Fire**, **Inferno**, and **Ice**.
  - Font-independent vector iconography and progressive `Advanced …` drawers so novice and expert controls stay clean and organized.

- **Substance Physics Presets & Hydraulic Wave Maker**:
  - One-click physical substance profiles (`--fluid-preset`): **Pure Water**, **Ocean Surf**, **Light Oil**, **Thick Honey**, **Glycerin**, **Liquid Mercury**, **Molten Lava**, and **Zero-G Globules**.
  - Built-in sinusoidal **Wave Maker** (`--wave-amp`, `--wave-freq`), sparse airborne spray drag (`--air-drag`), and per-obstacle bounce/slip controls.

- **Portable Studio Scene JSON, PNG Screenshots, MP4 Video & Blender Export**:
  - Save and reload complete studio setups (chamber, obstacles, faucets, drains, physics, materials, lighting, camera) as portable `.json` files (`--scene` / `--save-scene` or GUI toolbar).
  - Instant studio **PNG frame / screenshot** export (`--export-png snapshot.png` or `P` / `F12` in GUI) with up to 4x4 SSAA.
  - Direct CLI & GUI export to **H.264 MP4 videos** (`--export-mp4 simulation.mp4`), **Wavefront `.OBJ` liquid meshes** (`--export-obj`), and binary/ASCII **Stanford PLY** point clouds (`--export-ply`).

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
curl -LO https://github.com/rappar222/liquid-rust/releases/latest/download/liquid-rust_0.3.11_amd64.deb
sudo apt install ./liquid-rust_0.3.11_amd64.deb
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

## 🕹 Viewport & Keyboard Shortcuts

| Action | Control |
| :--- | :--- |
| **Attract Fluid** | Left-Click + Drag in Viewport |
| **Repel Fluid** | `Shift` + Left-Click + Drag |
| **Pan Camera** | Middle-Click + Drag |
| **Orbit Camera (3D)** | Right-Click + Drag |
| **Zoom In / Out** | Scroll Wheel |
| **Play / Pause** | `Space` or Top Toolbar Button |
| **Step Single Frame** | `.` (Period) or `Step` Button |
| **Reset Simulation** | `R` |
| **Reset Camera View** | `C` |
| **Toggle Spheres / Surface Mesh** | `M` |
| **Toggle Floor Grid / Wireframe** | `G` / `B` |
| **Toggle Turntable Auto-Orbit** | `O` |
| **Save Studio PNG Screenshot** | `P` or `F12` |
| **Keyboard Shortcuts Reference** | `?` or `F1` |

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

# Export an ultra-detailed CLI-only offline render (Surface Mesh + SSAA + 4K + H.264 CRF 11)
cargo run --release -- --export-mp4 master_water.mp4 --preset dam-break --quality studio --render-quality master --res 4k --surface-mesh --time 4.0 --fps 60

# Export a high-SSAA studio PNG snapshot with Honey physics, Golden Sunset sky, and wave maker
cargo run --release -- --export-png honey_waves.png --preset splash --fluid-preset honey --env-preset sunset --wave-amp 0.15 --render-quality studio --time 1.8

# Save a customized CLI setup to a portable JSON scene file, then launch or render from it
cargo run --release -- --preset obstacle --fluid-preset mercury --chamber-shape bathtub --save-scene mercury_tub.json
cargo run --release -- --scene mercury_tub.json --export-mp4 mercury_tub.mp4 --time 3.0 --fps 60

# Run with custom particle radius and spacing for ultra-fine liquid
cargo run --release -- --particle-radius 0.008 --spacing 0.014 --time 2.5 --fps 60
```

### Key CLI Flags (`liquid-rust --help` for full grouped reference)

```
Scene & Container:
      --preset <NAME>               Scene preset: dam-break, double-dam, splash, obstacle, tornado, zero-g, empty [default: dam-break]
      --scene <FILE.json>           Load complete studio scene configuration (.json); CLI flags override JSON values
      --save-scene <FILE.json>      Save effective scene, physics, lighting & camera configuration to portable .json
      --2d                          Run simulation in 2D planar mode (Z = 0)
      --chamber-shape <TYPE>        Bounding box type: box, sphere, bathtub (aliases: --box-type, --bounds-type)
      --chamber-size <W,H[,D]>      Bounding box size in meters: D, W,H (2D), or W,H,D (aliases: --box-size, --bounds-size)
      --chamber-scale <SCALE>       Bounding box scale: uniform, X,Y, or X,Y,Z
      --faucet <SPEC>               Add emitter: tap:x,y,z[:radius[:rate[:vx,vy,vz]]], stream:..., circle/shower/waterfall
      --drain <SPEC>                Add drain: floor:radius, floor:x,z:radius, <face>:radius, or custom:x,y,z:radius
      --obstacle <SPEC>             Add collider: sphere:x,y,z:r[:bounce[:slip]], box:..., cylinder:..., or obj:path:x,y,z:scale[:rx,ry,rz]

Resolution & Timeline:
  -q, --quality <QUALITY>           Interactive: draft..extreme; CLI-only: studio (~2.5M), film (~4.9M), master (~8.2M) [default: medium]
      --particles <COUNT>           Target initial particle count override
      --particle-radius <R>         Custom particle radius override
      --spacing <S>                 Custom particle spacing override
  -t, --time <SECONDS>              Total simulation duration in seconds
      --fps <FPS>                   Framerate for export and simulation stepping [default: 60]
      --sim-speed <MULT>            Simulation speed / ASMR slow-motion multiplier (e.g. 0.25) [default: 1.0]

Fluid Physics & Solver:
      --fluid-preset <NAME>         Substance profile: water, ocean, oil, honey, glycerin, mercury, lava, zero-g
      --viscosity <V>               XSPH artificial viscosity override
      --surface-tension <ST>        Macroscopic cohesive surface tension override
      --vorticity <VORT>            Swirl / vorticity confinement strength override
      --air-drag <DRAG>             Aerodynamic spray drag on isolated airborne droplets
      --gravity <Y|GX,GY,GZ>        Gravity override (scalar Y such as -9.81, or 3D vector gx,gy,gz)
      --restitution <BOUNCE>        Chamber wall bounce / restitution [0..1]
      --friction <SLIP>             Chamber wall tangential friction [0..1]
      --substeps <N>                Physics substeps per frame override
      --iterations <N>              PBF pressure solver iterations per substep override
      --wave-amp <AMP>              Sinusoidal wave-maker amplitude (0 = off)
      --wave-freq <HZ>              Wave-maker oscillation frequency in Hz [default: 0.85]

Visuals, Environment, Camera & Export:
      --surface-mesh                Enable reconstructed 3D liquid surface mesh + PBR refraction/caustics
      --color-mode <MODE>           Particle coloring: velocity, pressure, density, height, energy, foam, vorticity, solid
      --color-map <MAP>             Colormap: ocean, turbo, viridis, magma, inferno, ice
      --env-preset <ENV>            Studio sky & lighting: studio, daylight, sunset, arctic, abyssal
      --exposure <EXP>              Filmic camera exposure multiplier [default: 1.0]
      --camera-orbit <DEG_PER_SEC>  Continuous 3D turntable orbit speed in deg/s
      --fov <DEGREES>               Vertical camera field of view in degrees [default: 45]
      --render-quality <MODE>       Offline render AA & optics: standard (1x), studio (2x2), cinema (3x3), master (4x4, CRF 11)
      --res, --resolution <RES>     Video/PNG resolution preset: 480p..8k, optional -vertical, or WxH
      --export-mp4 <FILE.mp4>       Export H.264 MP4 video via FFmpeg
      --export-png <FILE.png>       Render simulation and save final studio PNG frame
      --export-obj <DIR>            Export reconstructed Wavefront .OBJ surface mesh sequence
      --export-ply <DIR>            Export Stanford .PLY point cloud sequence
      --headless / --benchmark      Run without GUI window / print throughput statistics
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
