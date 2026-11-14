use clap::Parser;
use eframe::egui;
use glam::Vec3;
use liquid_rust::export::{Mp4Encoder, PlyExporter, PngExporter, SoftwareRenderer};
use liquid_rust::gpu::GpuSphSolver;
use liquid_rust::render::{
    Camera, CliRenderQuality, ColorMap, ColorMapType, ColorMode, EnvironmentPreset,
    FluidMaterialPreset, MeshTextureMode, RenderMode, SurfaceMesh, SurfaceMeshConfig,
};
use liquid_rust::scene_io::{
    CameraDto, ChamberDto, ColliderDto, DrainDto, FaucetDto, SceneConfigFile, SolverConfigDto,
    SurfaceMeshDto, VisualsDto,
};
use liquid_rust::sph::{
    compute_lattice_rest_density, compute_lattice_rest_density_2d, Chamber, ChamberShape,
    Collider, ColliderShape, CubeFace, Drain, Faucet, FluidPhysicsPreset, SceneGenerator,
    ScenePreset, SimulationQuality, SolverConfig, SphSolver,
};
use liquid_rust::LiquidApp;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "liquid-rust",
    author = "Radu <rappar222>",
    version,
    about = "Studio-grade 2D/3D SPH Fluid Simulator in Rust with Surface Meshing, PBR Optics, Real-Time Viewport & MP4/PNG/OBJ/PLY Export"
)]
struct Args {
    // ======================== SCENE & CONTAINER ========================
    /// Load a portable LiquidRust studio scene (.json); explicit CLI flags override loaded fields
    #[arg(long, help_heading = "Scene & Container")]
    scene: Option<PathBuf>,

    /// Save the configured studio scene (preset, physics, chamber, objects, visuals) to a .json file
    #[arg(long, help_heading = "Scene & Container")]
    save_scene: Option<PathBuf>,

    /// Scene preset: dam-break, double-dam, splash, obstacle, tornado, zero-g, empty
    #[arg(long, help_heading = "Scene & Container")]
    preset: Option<String>,

    /// Run simulation in 2D planar mode (GUI also has a live 3D / 2D toggle button)
    #[arg(long = "2d", alias = "two-d", help_heading = "Scene & Container")]
    mode_2d: bool,

    /// Bounding box / chamber type: box (cuboid), sphere (globe), bathtub (tub) [default: box]
    #[arg(
        long,
        visible_aliases = ["chamber-type", "box-type", "bounds-type", "bounding-box-type"],
        help_heading = "Scene & Container"
    )]
    chamber_shape: Option<String>,

    /// Bounding box / chamber scale factor: uniform (1.5), 2D (1.5,1.2), or 3D (1.5,1.2,1.0)
    #[arg(
        long,
        visible_aliases = ["box-scale", "bounds-scale", "bounding-box-scale"],
        help_heading = "Scene & Container"
    )]
    chamber_scale: Option<String>,

    /// Bounding box / chamber dimensions in meters: D, W,H (2D), or W,H,D (e.g. 2.0,1.5,1.0 or 2.0x1.5x1.0)
    #[arg(
        long,
        visible_aliases = ["box-size", "bounds-size", "bounding-box-size"],
        help_heading = "Scene & Container"
    )]
    chamber_size: Option<String>,

    /// Bathtub backrest slope ratio (relative to length, 0.05..0.5; default 0.28)
    #[arg(long, help_heading = "Scene & Container")]
    tub_slope: Option<f32>,

    /// Bathtub bottom fillet radius ratio (relative to width, 0.01..0.45; default 0.16)
    #[arg(long, help_heading = "Scene & Container")]
    tub_fillet: Option<f32>,

    /// Oscillating hydraulic wave-maker paddle amplitude in meters along the left wall (0.0 = off)
    #[arg(long, alias = "wave-amplitude", help_heading = "Scene & Container")]
    wave_amp: Option<f32>,

    /// Wave-maker oscillation frequency in Hz [default: 0.85]
    #[arg(long, alias = "wave-frequency", help_heading = "Scene & Container")]
    wave_freq: Option<f32>,

    /// Add obstacle: 'sphere:x,y,z:r[:bounce[:slip]]', 'box:x,y,z:w,h,d', 'cylinder:x,y,z:r:h', or 'obj:path:x,y,z:scale[:rx,ry,rz]'
    #[arg(long = "obstacle", action = clap::ArgAction::Append, help_heading = "Scene & Container")]
    obstacles: Vec<String>,

    /// Add faucet: 'waterfall:x,y,z:width[:rate[:depth]]', 'shower:x,y,z:radius[:rate[:spread]]', 'circle:x,y,z:radius[:rate[:vx,vy,vz[:spread]]]', 'tap:x,y,z[:radius[:rate[:vx,vy,vz]]]', 'stream:x,y,z:vx,vy,vz[:rate[:spread[:radius]]]'
    #[arg(long = "faucet", action = clap::ArgAction::Append, help_heading = "Scene & Container")]
    faucets: Vec<String>,

    /// Add drain: 'floor:radius' or 'floor:x,z:radius', '<face>:radius' (floor|ceiling|left|right|back|front), or 'custom:x,y,z:radius'
    #[arg(long = "drain", action = clap::ArgAction::Append, help_heading = "Scene & Container")]
    drains: Vec<String>,

    // ======================== RESOLUTION & TIMELINE ========================
    /// Particle quality tier: draft, medium [default], high, ultra, cinematic, extreme, or CLI-only: studio (~2.5M), film (~4.9M), master (~8.2M)
    #[arg(long, short = 'q', help_heading = "Resolution & Timeline")]
    quality: Option<String>,

    /// Custom particle radius override in meters (smaller = finer fluid)
    #[arg(long, help_heading = "Resolution & Timeline")]
    particle_radius: Option<f32>,

    /// Custom initial particle lattice spacing override in meters
    #[arg(long, help_heading = "Resolution & Timeline")]
    spacing: Option<f32>,

    /// Target number of particles to spawn (automatically computes optimal spacing & radius)
    #[arg(long, alias = "particle-count", help_heading = "Resolution & Timeline")]
    particles: Option<usize>,

    /// Total simulation duration in seconds (e.g. --time 4.0). Overrides --frames
    #[arg(long, short = 't', alias = "duration", help_heading = "Resolution & Timeline")]
    time: Option<f32>,

    /// Number of simulation frames to run when --time is not specified [default: 120]
    #[arg(long, help_heading = "Resolution & Timeline")]
    frames: Option<usize>,

    /// Framerate (FPS) for video export and simulation stepping
    #[arg(long, default_value_t = 60, help_heading = "Resolution & Timeline")]
    fps: u32,

    /// Simulation time-scale speed multiplier (e.g. 0.25 for 4x slow-mo, 1.0 realtime)
    #[arg(long, alias = "sim-speed", help_heading = "Resolution & Timeline")]
    speed: Option<f32>,

    /// Custom base timestep dt per frame in seconds (defaults to 1.0 / fps * speed)
    #[arg(long, help_heading = "Resolution & Timeline")]
    dt: Option<f32>,

    // ======================== FLUID PHYSICS & SOLVER ========================
    /// Physical substance preset: water, ocean, oil, honey, glycerin, mercury, lava, zero-g
    #[arg(long, alias = "substance", help_heading = "Fluid Physics & Solver")]
    fluid_preset: Option<String>,

    /// Gravity acceleration in m/s² as vertical scalar (e.g. -9.81 or 0) or 3D vector 'gx,gy,gz'
    #[arg(long, allow_hyphen_values = true, help_heading = "Fluid Physics & Solver")]
    gravity: Option<String>,

    /// XSPH shear viscosity coefficient [0.0..1.0]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    viscosity: Option<f32>,

    /// Akinci free-surface cohesion / surface tension strength [0.0..0.5]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    surface_tension: Option<f32>,

    /// Macklin & Müller vorticity confinement swirl strength [0.0..0.5]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    vorticity: Option<f32>,

    /// Aerodynamic drag on sparse airborne splash droplets [0.0..5.0; default: 0.35]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    air_drag: Option<f32>,

    /// Boundary wall collision restitution / bounce [0.0..1.0; default: 0.0]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    restitution: Option<f32>,

    /// Boundary wall slip factor [0.0..1.0; 1.0 = free slip, lower = higher drag]
    #[arg(long, visible_alias = "boundary-friction", help_heading = "Fluid Physics & Solver")]
    friction: Option<f32>,

    /// Number of PBF physics substeps per frame [1..32]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    substeps: Option<usize>,

    /// Number of PBF pressure-projection solver iterations per substep [1..32]
    #[arg(long, visible_alias = "solver-iterations", help_heading = "Fluid Physics & Solver")]
    iterations: Option<usize>,

    /// Monaghan tensile anti-clumping correction factor [0.0..0.05]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    tensile_k: Option<f32>,

    /// Safety maximum particle speed clamp in m/s [default: 15.0]
    #[arg(long, help_heading = "Fluid Physics & Solver")]
    max_velocity: Option<f32>,

    // ======================== VISUALS, ENVIRONMENT & CAMERA ========================
    /// Fluid rendering representation: 'particles' (SPH spheres) or 'mesh' (3D Marching Cubes surface mesh) [default: particles]
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    render_mode: Option<String>,

    /// Enable continuous 3D surface meshing for renders & viewport (shortcut for --render-mode mesh)
    #[arg(long, alias = "mesh", help_heading = "Visuals, Environment & Camera")]
    surface_mesh: bool,

    /// Particle color field: velocity [default], foam, vorticity, pressure, density, height, energy, single
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    color_mode: Option<String>,

    /// Particle colormap gradient: ocean [default], turbo, viridis, plasma, inferno, fire, ice
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    color_map: Option<String>,

    /// Visual sphere radius multiplier for particle mode [default: 1.25]
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    particle_scale: Option<f32>,

    /// Disable dynamic aerated whitewater crest highlights in particle sphere mode
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    no_particle_foam: bool,

    /// Render flat unlit discs instead of 3D Fresnel-lit spheres in particle mode
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    unlit_spheres: bool,

    /// Studio lighting & sky backdrop preset: studio [default], daylight, sunset, arctic, abyssal
    #[arg(long, visible_alias = "environment", help_heading = "Visuals, Environment & Camera")]
    env_preset: Option<String>,

    /// Filmic camera exposure multiplier [0.2..3.5; default: 1.0]
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    exposure: Option<f32>,

    /// Hide the 3D studio floor reference grid
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    no_grid: bool,

    /// Hide chamber, obstacle & emitter wireframe overlays
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    no_wireframe: bool,

    /// Directional key sun vector 'x,y,z' (e.g. '0.42,0.82,0.38')
    #[arg(long, allow_hyphen_values = true, help_heading = "Visuals, Environment & Camera")]
    light_dir: Option<String>,

    /// Key directional sun radiance multiplier [default: 1.25]
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    light_intensity: Option<f32>,

    /// Hemisphere ambient light fill intensity [default: 0.32]
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    ambient: Option<f32>,

    /// Disable colored liquid floor shadows & underwater caustics in surface-mesh renders
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    no_shadows: bool,

    /// Camera horizontal orbit azimuth angle in degrees
    #[arg(long, allow_hyphen_values = true, help_heading = "Visuals, Environment & Camera")]
    camera_azimuth: Option<f32>,

    /// Camera vertical orbit elevation angle in degrees
    #[arg(long, allow_hyphen_values = true, help_heading = "Visuals, Environment & Camera")]
    camera_elevation: Option<f32>,

    /// Camera orbit distance from target in meters
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    camera_distance: Option<f32>,

    /// Camera look-at target 'x,y,z'
    #[arg(long, allow_hyphen_values = true, help_heading = "Visuals, Environment & Camera")]
    camera_target: Option<String>,

    /// Camera vertical field-of-view in degrees [15..110; default: 45]
    #[arg(long, help_heading = "Visuals, Environment & Camera")]
    fov: Option<f32>,

    /// Automatic 360° turntable camera orbit angular speed in degrees/sec [default: 0]
    #[arg(long, allow_hyphen_values = true, help_heading = "Visuals, Environment & Camera")]
    camera_orbit: Option<f32>,

    // ======================== SURFACE MESH & PBR OPTICS ========================
    /// Surface mesh material preset: water [default], ocean, lagoon, amber, wine, emerald, milk, chrome, custom
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_material: Option<String>,

    /// Shallow / thin-film surface liquid color as hex '#RRGGBB' or 'R,G,B'
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_color: Option<String>,

    /// Deep volume Beer-Lambert absorption color as hex '#RRGGBB' or 'R,G,B'
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_deep_color: Option<String>,

    /// Base liquid surface opacity [0.05..1.0]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_opacity: Option<f32>,

    /// Beer-Lambert optical volume absorption density [0.0..10.0]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_absorption: Option<f32>,

    /// Aerated whitewater wave-crest foam intensity [0.0..1.5]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_foam: Option<f32>,

    /// Procedural triplanar surface texture: none, ripples [default], caustics, foam, grid
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_texture: Option<String>,

    /// Spatial frequency scale for procedural surface texture [default: 6.0]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_texture_scale: Option<f32>,

    /// Bump & pattern strength for procedural surface texture [0.0..1.0; default: 0.28]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_texture_strength: Option<f32>,

    /// Optical Index of Refraction (IOR) for Schlick Fresnel (e.g. 1.333 water, 1.52 glass, 2.40 chrome)
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_ior: Option<f32>,

    /// GGX microfacet surface roughness [0.01..1.0] (smaller = mirror-sharp highlights)
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_roughness: Option<f32>,

    /// Direct sun specular highlight intensity multiplier
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_specular: Option<f32>,

    /// Screen-space refraction background distortion strength [0.0..2.0]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_refraction: Option<f32>,

    /// Subsurface wave-crest back-scatter glow intensity [0.0..3.0]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_sss: Option<f32>,

    /// Studio sky & horizon Fresnel reflection intensity [0.0..4.0]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_reflection: Option<f32>,

    /// Metallic reflectance factor [0.0..1.0]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_metallic: Option<f32>,

    /// Marching Cubes voxel size multiplier relative to particle radius [default: 0.72]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_scale: Option<f32>,

    /// Isosurface density threshold [0.10..0.90; default: 0.42]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_iso: Option<f32>,

    /// Volume-preserving Taubin mesh smoothing iterations [default: 5]
    #[arg(long, help_heading = "Surface Mesh & PBR Optics")]
    mesh_smooth: Option<usize>,

    // ======================== EXPORT, COMPUTE & UPDATES ========================
    /// Run in headless CLI mode without opening an interactive window
    #[arg(long, help_heading = "Export, Compute & Updates")]
    headless: bool,

    /// Export simulation video to MP4 file via ffmpeg (e.g. output.mp4)
    #[arg(long, help_heading = "Export, Compute & Updates")]
    export_mp4: Option<PathBuf>,

    /// Export the final simulated frame (or frame 0 with --frames 0) as a studio PNG image
    #[arg(long, visible_alias = "screenshot", help_heading = "Export, Compute & Updates")]
    export_png: Option<PathBuf>,

    /// Directory to export binary PLY point-cloud frames
    #[arg(long, help_heading = "Export, Compute & Updates")]
    export_ply: Option<PathBuf>,

    /// Directory to export reconstructed Wavefront .OBJ surface mesh sequence
    #[arg(long, help_heading = "Export, Compute & Updates")]
    export_obj: Option<PathBuf>,

    /// Offline render quality preset (SSAA, high-res mesh, soft shadows, dispersion, CRF): standard, studio, cinema, master
    #[arg(long, alias = "rq", help_heading = "Export, Compute & Updates")]
    render_quality: Option<String>,

    /// Output image/video resolution preset: 480p, 720p, 1080p, 1440p, 4k, 8k, optional -vertical/-portrait, or WxH
    #[arg(long, visible_alias = "res", help_heading = "Export, Compute & Updates")]
    resolution: Option<String>,

    /// Video / PNG export width in pixels
    #[arg(long, default_value_t = 1280, help_heading = "Export, Compute & Updates")]
    video_width: usize,

    /// Video / PNG export height in pixels
    #[arg(long, default_value_t = 720, help_heading = "Export, Compute & Updates")]
    video_height: usize,

    /// Force CPU multi-threaded solver instead of GPU acceleration
    #[arg(long, help_heading = "Export, Compute & Updates")]
    cpu: bool,

    /// Explicitly enable GPU acceleration (default: auto-detected Vulkan)
    #[arg(long, help_heading = "Export, Compute & Updates")]
    gpu: bool,

    /// Enable particle frustum and viewport culling during rendering [default: true]
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, help_heading = "Export, Compute & Updates")]
    culling: bool,

    /// Disable frustum and viewport particle culling (renders all particles)
    #[arg(long, alias = "no-cull", help_heading = "Export, Compute & Updates")]
    no_culling: bool,

    /// Print performance benchmark statistics
    #[arg(long, help_heading = "Export, Compute & Updates")]
    benchmark: bool,

    /// Check GitHub Releases for a newer LiquidRust version and exit
    #[arg(long, help_heading = "Export, Compute & Updates")]
    check_update: bool,

    /// Download and install the latest LiquidRust release from GitHub in-place
    #[arg(long, alias = "self-update", help_heading = "Export, Compute & Updates")]
    update: bool,

    /// Force reinstalling the latest GitHub release even if version matches
    #[arg(long, help_heading = "Export, Compute & Updates")]
    force_update: bool,
}

fn format_time(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    let mins = s / 60;
    let secs = s % 60;
    if mins > 0 {
        format!("{}m {:02}s", mins, secs)
    } else {
        format!("{:.1}s", seconds.max(0.0))
    }
}

fn parse_preset(name: Option<&str>, has_faucets: bool) -> ScenePreset {
    match name {
        Some(s) => match s.trim().to_lowercase().as_str() {
            "dam-break" | "dambreak" | "dam" => ScenePreset::DamBreak,
            "double-dam" | "doubledam" | "double" | "double-dam-break" | "doubledambreak" => {
                ScenePreset::DoubleDamBreak
            }
            "splash" | "droplet" | "dropletsplash" | "droplet-splash" => ScenePreset::DropletSplash,
            "obstacle" | "drop" | "obstacledrop" | "obstacle-drop" => ScenePreset::ObstacleDrop,
            "tornado" | "vortex" | "vortex-tornado" => ScenePreset::VortexTornado,
            "zero-g" | "zerog" | "blobs" | "zero-gravity" => ScenePreset::ZeroGravityBlobs,
            "empty" | "none" | "basin" | "faucet" => ScenePreset::Empty,
            _ => {
                eprintln!("Unknown preset '{}', defaulting to dam-break", s);
                ScenePreset::DamBreak
            }
        },
        None => {
            if has_faucets {
                ScenePreset::Empty
            } else {
                ScenePreset::DamBreak
            }
        }
    }
}

fn parse_quality(name: &str) -> SimulationQuality {
    match name.trim().to_lowercase().as_str() {
        "draft" | "low" | "fast" => SimulationQuality::Draft,
        "medium" | "med" | "normal" => SimulationQuality::Medium,
        "high" | "hi" | "smooth" => SimulationQuality::High,
        "ultra" | "realistic" => SimulationQuality::Ultra,
        "cinematic" | "vfx" => SimulationQuality::Cinematic,
        "extreme" | "1m" | "insane" => SimulationQuality::Extreme,
        "studio" | "photoreal" => SimulationQuality::Studio,
        "film" | "vfx-max" => SimulationQuality::Film,
        "master" | "hyper" | "insane-cli" => SimulationQuality::Master,
        "custom" => SimulationQuality::Custom,
        _ => {
            eprintln!("Unknown quality '{}', defaulting to 'medium'", name);
            SimulationQuality::Medium
        }
    }
}

fn resolve_cli_render_quality(args: &Args, sim_quality: SimulationQuality) -> CliRenderQuality {
    if let Some(ref rq) = args.render_quality {
        if let Some(parsed) = CliRenderQuality::from_slug(rq) {
            return parsed;
        }
        eprintln!(
            "Unknown --render-quality '{}' (expected standard, studio, cinema, master), defaulting to 'standard'",
            rq
        );
        return CliRenderQuality::Standard;
    }

    match sim_quality {
        SimulationQuality::Studio => CliRenderQuality::Studio,
        SimulationQuality::Film => CliRenderQuality::Cinema,
        SimulationQuality::Master => CliRenderQuality::Master,
        _ => CliRenderQuality::Standard,
    }
}

fn parse_resolution_spec(spec: &str) -> Option<(usize, usize)> {
    let normalized = spec.trim().to_ascii_lowercase().replace('_', "-");
    let (base, vertical) = if let Some(b) = normalized.strip_suffix("-vertical") {
        (b, true)
    } else if let Some(b) = normalized.strip_suffix("-portrait") {
        (b, true)
    } else {
        (normalized.as_str(), false)
    };

    let (w, h) = match base {
        "360p" => (640, 360),
        "480p" | "sd" => (854, 480),
        "540p" | "qhd-half" => (960, 540),
        "720p" | "hd" => (1280, 720),
        "900p" | "hd+" | "hdplus" => (1600, 900),
        "1k" | "1080p" | "fhd" | "fullhd" | "full-hd" => (1920, 1080),
        "dci-2k" | "2k-dci" => (2048, 1080),
        "2k" | "1440p" | "qhd" | "wqhd" => (2560, 1440),
        "3k" | "1800p" => (3200, 1800),
        "4k" | "2160p" | "uhd" | "4k-uhd" => (3840, 2160),
        "dci-4k" | "4k-dci" => (4096, 2160),
        "5k" | "2880p" => (5120, 2880),
        "6k" | "3456p" => (6144, 3456),
        "8k" | "4320p" | "8k-uhd" => (7680, 4320),
        other => {
            let (lw, lh) = other
                .split_once('x')
                .or_else(|| other.split_once('*'))
                .or_else(|| other.split_once(','))?;
            let pw = lw.trim().parse::<usize>().ok()?;
            let ph = lh.trim().parse::<usize>().ok()?;
            if pw < 16 || ph < 16 {
                return None;
            }
            (pw, ph)
        }
    };

    Some(if vertical { (h, w) } else { (w, h) })
}

fn resolve_video_dimensions(args: &Args) -> (usize, usize) {
    let (mut w, mut h) = if let Some(ref spec) = args.resolution {
        if let Some(dims) = parse_resolution_spec(spec) {
            dims
        } else {
            eprintln!(
                "Warning: unknown --res/--resolution '{}' (try 480p, 720p, 1k, 1080p, 2k, 1440p, 3k, 4k, 5k, 6k, 8k, or WxH), using {}x{}",
                spec, args.video_width, args.video_height
            );
            (args.video_width, args.video_height)
        }
    } else {
        (args.video_width, args.video_height)
    };

    if args.resolution.is_some() {
        if args.video_width != 1280 {
            w = args.video_width;
        }
        if args.video_height != 720 {
            h = args.video_height;
        }
    }

    // H.264 yuv420p requires even dimensions >= 16
    let w_even = (w.max(16) + 1) & !1;
    let h_even = (h.max(16) + 1) & !1;
    (w_even, h_even)
}

fn parse_color_mode(s: &str) -> ColorMode {
    if let Some(cm) = ColorMode::from_slug(s) {
        cm
    } else {
        eprintln!("Warning: unknown --color-mode '{}', defaulting to 'velocity'", s);
        ColorMode::Velocity
    }
}

fn parse_color_map(s: &str) -> ColorMapType {
    if let Some(cm) = ColorMapType::from_slug(s) {
        cm
    } else {
        eprintln!("Warning: unknown --color-map '{}', defaulting to 'ocean'", s);
        ColorMapType::Ocean
    }
}

fn parse_gravity_arg(raw: &str) -> Result<Vec3, String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c| c == '(' || c == ')' || c == '[' || c == ']');
    if !trimmed.contains(',') {
        let gy = trimmed
            .parse::<f32>()
            .map_err(|e| format!("invalid gravity scalar '{}': {}", raw, e))?;
        if !gy.is_finite() {
            return Err("gravity must be finite".to_string());
        }
        return Ok(Vec3::new(0.0, gy, 0.0));
    }
    parse_vec3(trimmed)
}

fn build_mesh_config(args: &Args, base_scene: Option<&SceneConfigFile>) -> SurfaceMeshConfig {
    let mut cfg = if let Some(sc) = base_scene {
        sc.visuals.mesh.to_config()
    } else {
        let mut def = SurfaceMeshConfig::default();
        FluidMaterialPreset::ClearWater.apply(&mut def);
        def
    };

    if let Some(ref mat) = args.mesh_material {
        let preset = FluidMaterialPreset::from_slug(mat);
        preset.apply(&mut cfg);
    }

    if let Some(ref c) = args.mesh_color {
        if let Some(rgb) = SurfaceMeshConfig::parse_rgb(c) {
            cfg.shallow_color = rgb;
            if args
                .mesh_material
                .as_deref()
                .map_or(false, |m| m.eq_ignore_ascii_case("custom"))
            {
                cfg.preset = FluidMaterialPreset::Custom;
            }
        } else {
            eprintln!("Warning: invalid --mesh-color '{}', expected #RRGGBB or R,G,B", c);
        }
    }
    if let Some(ref c) = args.mesh_deep_color {
        if let Some(rgb) = SurfaceMeshConfig::parse_rgb(c) {
            cfg.deep_color = rgb;
        } else {
            eprintln!(
                "Warning: invalid --mesh-deep-color '{}', expected #RRGGBB or R,G,B",
                c
            );
        }
    }
    if let Some(v) = args.mesh_opacity {
        cfg.opacity = v.clamp(0.05, 1.0);
    }
    if let Some(v) = args.mesh_absorption {
        cfg.absorption = v.clamp(0.0, 10.0);
    }
    if let Some(v) = args.mesh_foam {
        cfg.foam_intensity = v.clamp(0.0, 1.5);
    }

    if let Some(ref tex) = args.mesh_texture {
        cfg.texture_mode = MeshTextureMode::from_slug(tex);
    }
    if let Some(v) = args.mesh_texture_scale {
        cfg.texture_scale = v.clamp(0.2, 40.0);
    }
    if let Some(v) = args.mesh_texture_strength {
        cfg.texture_strength = v.clamp(0.0, 1.0);
    }

    if let Some(v) = args.mesh_ior {
        cfg.ior = v.clamp(1.01, 3.0);
    }
    if let Some(v) = args.mesh_roughness {
        cfg.roughness = v.clamp(0.01, 1.0);
    }
    if let Some(v) = args.mesh_specular {
        cfg.specular_intensity = v.clamp(0.0, 5.0);
    }
    if let Some(v) = args.mesh_refraction {
        cfg.refraction_strength = v.clamp(0.0, 2.0);
    }
    if let Some(v) = args.mesh_sss {
        cfg.subsurface_scattering = v.clamp(0.0, 3.0);
    }
    if let Some(v) = args.mesh_reflection {
        cfg.env_reflection = v.clamp(0.0, 4.0);
    }
    if let Some(v) = args.mesh_metallic {
        cfg.metallic = v.clamp(0.0, 1.0);
    }
    if let Some(v) = args.mesh_scale {
        cfg.voxel_scale = v.clamp(0.30, 2.0);
    }
    if let Some(v) = args.mesh_iso {
        cfg.iso_value = v.clamp(0.10, 0.90);
    }
    if let Some(v) = args.mesh_smooth {
        cfg.smoothing_passes = v.min(24);
    }
    if let Some(ref ep) = args.env_preset {
        if let Some(parsed_ep) = EnvironmentPreset::from_slug(ep) {
            cfg.env_preset = parsed_ep;
        } else {
            eprintln!(
                "Warning: unknown --env-preset '{}' (expected studio, daylight, sunset, arctic, abyssal)",
                ep
            );
        }
    }
    if let Some(exp) = args.exposure {
        cfg.exposure = exp.clamp(0.2, 3.5);
    }
    if args.no_grid {
        cfg.show_grid = false;
    }
    if args.no_wireframe {
        cfg.show_wireframe = false;
    }
    if let Some(ref ldir) = args.light_dir {
        if let Ok(v) = parse_vec3(ldir) {
            if v.length_squared() > 1e-4 {
                cfg.light_dir = v.normalize();
            }
        } else {
            eprintln!("Warning: invalid --light-dir '{}', expected x,y,z", ldir);
        }
    }
    if let Some(v) = args.light_intensity {
        cfg.light_intensity = v.clamp(0.0, 5.0);
    }
    if let Some(v) = args.ambient {
        cfg.ambient_intensity = v.clamp(0.0, 2.0);
    }
    if args.no_shadows {
        cfg.cast_shadows = false;
    }

    cfg
}

fn build_camera(args: &Args, base_scene: Option<&SceneConfigFile>, is_2d: bool) -> (Camera, f32) {
    let (mut cam, mut orbit_deg) = if let Some(sc) = base_scene {
        (sc.camera.to_camera(), sc.camera.orbit_speed_deg)
    } else {
        (Camera::default(), 0.0)
    };
    if is_2d {
        cam.azimuth = 0.0;
        cam.elevation = 0.0;
        if base_scene.is_none() {
            cam.target = Vec3::new(0.0, 0.6, 0.0);
            cam.distance = 2.4;
        }
        orbit_deg = 0.0;
    }
    if let Some(az_deg) = args.camera_azimuth {
        cam.azimuth = az_deg.to_radians();
    }
    if let Some(el_deg) = args.camera_elevation {
        cam.elevation = el_deg.to_radians().clamp(-1.5, 1.5);
    }
    if let Some(dist) = args.camera_distance {
        cam.distance = dist.clamp(0.3, 25.0);
    }
    if let Some(ref tgt) = args.camera_target {
        if let Ok(v) = parse_vec3(tgt) {
            cam.target = v;
        }
    }
    if let Some(fov_deg) = args.fov {
        cam.fov_y = fov_deg.clamp(15.0, 110.0).to_radians();
    }
    if let Some(orb) = args.camera_orbit {
        orbit_deg = if is_2d { 0.0 } else { orb.clamp(-360.0, 360.0) };
    }
    (cam, orbit_deg)
}

fn apply_cli_physics_overrides(
    args: &Args,
    config: &mut SolverConfig,
    fluid_preset_out: &mut FluidPhysicsPreset,
) {
    if let Some(ref fp_str) = args.fluid_preset {
        if let Some(fp) = FluidPhysicsPreset::from_slug(fp_str) {
            fp.apply(config);
            *fluid_preset_out = fp;
        } else {
            eprintln!(
                "Warning: unknown --fluid-preset '{}' (expected water, ocean, oil, honey, glycerin, mercury, lava, zero-g)",
                fp_str
            );
        }
    }
    if let Some(ref g_str) = args.gravity {
        match parse_gravity_arg(g_str) {
            Ok(g) => config.gravity = g,
            Err(e) => eprintln!("Warning: invalid --gravity '{}': {}", g_str, e),
        }
    }
    if let Some(visc) = args.viscosity {
        config.viscosity = visc.clamp(0.0, 1.0);
    }
    if let Some(st) = args.surface_tension {
        config.surface_tension = st.clamp(0.0, 0.5);
    }
    if let Some(vort) = args.vorticity {
        config.vorticity = vort.clamp(0.0, 0.5);
    }
    if let Some(drag) = args.air_drag {
        config.air_drag = drag.clamp(0.0, 5.0);
    }
    if let Some(rest) = args.restitution {
        config.restitution = rest.clamp(0.0, 1.0);
    }
    if let Some(fric) = args.friction {
        config.boundary_friction = fric.clamp(0.0, 1.0);
    }
    if let Some(sub) = args.substeps {
        config.substeps = sub.clamp(1, 32);
    }
    if let Some(iters) = args.iterations {
        config.solver_iterations = iters.clamp(1, 32);
    }
    if let Some(tk) = args.tensile_k {
        config.tensile_k = tk.clamp(0.0, 0.05);
    }
    if let Some(mv) = args.max_velocity {
        config.max_velocity = mv.clamp(1.0, 100.0);
    }
    if let Some(wa) = args.wave_amp {
        config.wave_amplitude = wa.clamp(0.0, 2.0);
    }
    if let Some(wf) = args.wave_freq {
        config.wave_frequency = wf.clamp(0.05, 10.0);
    }
}

fn clean_spec(spec: &str) -> Vec<String> {
    let s = spec.replace("[:", ":").replace('[', "").replace(']', "");
    s.split(':')
        .map(|part| {
            part.trim()
                .trim_matches(|c| c == '(' || c == ')' || c == '{' || c == '}')
                .trim()
                .to_string()
        })
        .filter(|part| !part.is_empty())
        .collect()
}

fn parse_vec3(s: &str) -> Result<glam::Vec3, String> {
    let clean = s
        .trim()
        .trim_matches(|c| c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}');
    let parts: Vec<&str> = clean.split(',').collect();
    if parts.len() != 3 {
        return Err(format!(
            "expected 3 comma-separated floats 'x,y,z', got '{}'",
            s
        ));
    }
    let x = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("invalid float '{}': {}", parts[0], e))?;
    let y = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("invalid float '{}': {}", parts[1], e))?;
    let z = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("invalid float '{}': {}", parts[2], e))?;
    Ok(glam::Vec3::new(x, y, z))
}

fn parse_obstacle(spec: &str, index: usize) -> Result<Collider, String> {
    let parts = clean_spec(spec);
    if parts.is_empty() {
        return Err("empty obstacle specification".to_string());
    }
    let kind = parts[0].to_lowercase();
    let name = format!("Obstacle #{} ({})", index + 1, kind);

    let (mut col, extra_start) = match kind.as_str() {
        "sphere" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid sphere spec '{}'. Format: sphere:x,y,z:radius[:restitution[:friction]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
            (Collider::sphere(pos, radius), 3usize)
        }
        "box" | "cube" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid box spec '{}'. Format: box:x,y,z:w,h,d[:restitution[:friction]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let size = parse_vec3(&parts[2])?;
            let half = size * 0.5;
            (Collider::box_obstacle(pos - half, pos + half), 3usize)
        }
        "cylinder" => {
            if parts.len() < 4 {
                return Err(format!(
                    "Invalid cylinder spec '{}'. Format: cylinder:x,y,z:radius:height[:restitution[:friction]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
            let height = parse_f32_checked(&parts[3], "height", Some(0.0))?;
            (Collider::cylinder(pos, radius, height), 4usize)
        }
        "obj" | "mesh" => {
            if parts.len() < 4 {
                return Err(format!(
                    "Invalid obj spec '{}'. Format: obj:filepath:x,y,z:scale[:rx,ry,rz]",
                    spec
                ));
            }
            let path = parts[1].trim();
            let pos = parse_vec3(&parts[2])?;
            let scale = parse_f32_checked(&parts[3], "scale", Some(0.0))?;
            let rot_rad = if parts.len() >= 5 {
                let deg = parse_vec3(&parts[4])?;
                glam::Vec3::new(deg.x.to_radians(), deg.y.to_radians(), deg.z.to_radians())
            } else {
                glam::Vec3::ZERO
            };
            let mut c = Collider::load_obj_transformed(path, pos, scale, rot_rad)
                .map_err(|e| format!("failed to load OBJ '{}': {}", path, e))?;
            c.name = format!("OBJ #{} ({})", index + 1, path);
            c.id = (index + 1) as u32;
            return Ok(c);
        }
        _ => {
            return Err(format!(
                "Unknown obstacle type '{}'. Expected 'sphere', 'box', 'cylinder', or 'obj'",
                kind
            ))
        }
    };

    col.name = name;
    col.id = (index + 1) as u32;
    if parts.len() > extra_start {
        col.restitution = parse_f32_checked(&parts[extra_start], "restitution", None)?.clamp(0.0, 1.0);
    }
    if parts.len() > extra_start + 1 {
        col.friction =
            parse_f32_checked(&parts[extra_start + 1], "friction", None)?.clamp(0.0, 1.0);
    }
    Ok(col)
}

fn parse_f32_checked(raw: &str, label: &str, min_exclusive: Option<f32>) -> Result<f32, String> {
    let val = raw
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("invalid {}: {}", label, e))?;
    if !val.is_finite() {
        return Err(format!("invalid {}: must be finite", label));
    }
    if let Some(min) = min_exclusive {
        if val <= min {
            return Err(format!("invalid {}: must be > {}", label, min));
        }
    } else if val < 0.0 {
        return Err(format!("invalid {}: must be >= 0", label));
    }
    Ok(val)
}

fn parse_faucet(spec: &str, index: usize) -> Result<Faucet, String> {
    let parts = clean_spec(spec);
    if parts.is_empty() {
        return Err("empty faucet specification".to_string());
    }
    let kind = parts[0].to_lowercase();
    let name = format!("Faucet #{} ({})", index + 1, kind);

    let mut faucet = match kind.as_str() {
        "waterfall" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid waterfall spec '{}'. Format: waterfall:x,y,z:width[:flow_rate[:depth]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let width = parse_f32_checked(&parts[2], "width", Some(0.0))?;
            let flow_rate = if parts.len() > 3 {
                parse_f32_checked(&parts[3], "flow_rate", None)?
            } else {
                600.0
            };
            let depth = if parts.len() > 4 {
                parse_f32_checked(&parts[4], "depth", Some(0.0))?
            } else {
                0.08
            };
            Faucet::new_waterfall(name, pos, width, depth, flow_rate)
        }
        "shower" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid shower spec '{}'. Format: shower:x,y,z:radius[:flow_rate[:spread]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
            let flow_rate = if parts.len() > 3 {
                parse_f32_checked(&parts[3], "flow_rate", None)?
            } else {
                500.0
            };
            let spread = if parts.len() > 4 {
                parse_f32_checked(&parts[4], "spread", None)?
            } else {
                0.22
            };
            Faucet::new_shower(name, pos, radius, flow_rate, spread)
        }
        "circle" | "pipe" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid circle spec '{}'. Format: circle:x,y,z:radius[:flow_rate[:vx,vy,vz[:spread]]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
            let flow_rate = if parts.len() > 3 {
                parse_f32_checked(&parts[3], "flow_rate", None)?
            } else {
                500.0
            };
            let vel = if parts.len() > 4 {
                parse_vec3(&parts[4])?
            } else {
                glam::Vec3::new(0.0, -2.5, 0.0)
            };
            let spread = if parts.len() > 5 {
                parse_f32_checked(&parts[5], "spread", None)?
            } else {
                0.0
            };
            let mut f = Faucet::new_circle(name, pos, vel, radius, flow_rate);
            f.spray_spread = spread;
            f
        }
        "tap" | "point" => {
            if parts.len() < 2 {
                return Err(format!(
                    "Invalid {} spec '{}'. Format: {}:x,y,z[:radius[:flow_rate[:vx,vy,vz[:spread]]]]",
                    kind, spec, kind
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let default_radius = if kind == "tap" { 0.05 } else { 0.0 };
            let (radius, flow_rate) = if parts.len() == 2 {
                (default_radius, 400.0)
            } else if parts.len() == 3 {
                let val = parse_f32_checked(&parts[2], "radius_or_rate", None)?;
                if kind == "tap" && val <= 2.0 {
                    (val, 400.0)
                } else {
                    (default_radius, val)
                }
            } else {
                let radius = parse_f32_checked(&parts[2], "radius", None)?;
                let flow_rate = parse_f32_checked(&parts[3], "flow_rate", None)?;
                (radius, flow_rate)
            };
            let vel = if parts.len() > 4 {
                parse_vec3(&parts[4])?
            } else {
                glam::Vec3::new(0.0, -2.0, 0.0)
            };
            let spread = if parts.len() > 5 {
                parse_f32_checked(&parts[5], "spread", None)?
            } else {
                0.0
            };
            let mut f = if radius <= 1e-6 {
                Faucet::new_point(name, pos, vel, flow_rate)
            } else {
                Faucet::new_circle(name, pos, vel, radius, flow_rate)
            };
            f.spray_spread = spread;
            f
        }
        "stream" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid stream spec '{}'. Format: stream:x,y,z:vx,vy,vz[:flow_rate[:spread[:radius]]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let vel = parse_vec3(&parts[2])?;
            let flow_rate = if parts.len() > 3 {
                parse_f32_checked(&parts[3], "flow_rate", None)?
            } else {
                500.0
            };
            let spread = if parts.len() > 4 {
                parse_f32_checked(&parts[4], "spread", None)?
            } else {
                0.05
            };
            let radius = if parts.len() > 5 {
                parse_f32_checked(&parts[5], "radius", None)?
            } else {
                0.04
            };
            let mut f = if radius <= 1e-6 {
                Faucet::new_point(name, pos, vel, flow_rate)
            } else {
                Faucet::new_circle(name, pos, vel, radius, flow_rate)
            };
            f.spray_spread = spread;
            f
        }
        _ => {
            return Err(format!(
                "Unknown faucet shape '{}'. Expected 'waterfall', 'shower', 'circle', 'tap', or 'stream'",
                kind
            ))
        }
    };
    faucet.id = (index + 1) as u32;
    Ok(faucet)
}

fn parse_drain(spec: &str, index: usize) -> Result<Drain, String> {
    let parts = clean_spec(spec);
    if parts.is_empty() {
        return Err("empty drain specification".to_string());
    }
    let kind = parts[0].to_lowercase();
    let name = format!("Drain #{} ({})", index + 1, kind);

    let (face, pos, radius) = match kind.as_str() {
        "floor" | "bottom" => {
            if parts.len() == 2 {
                let r = parse_f32_checked(&parts[1], "radius", Some(0.0))?;
                (CubeFace::Floor, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xz: Vec<&str> = parts[1].split(',').collect();
                if xz.len() != 2 {
                    return Err(format!(
                        "Expected 2 coords (x,z) for floor drain offset, got '{}'",
                        parts[1]
                    ));
                }
                let x = xz[0]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid x: {}", e))?;
                let z = xz[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid z: {}", e))?;
                let r = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
                (CubeFace::Floor, glam::Vec3::new(x, 0.0, z), r)
            } else {
                return Err(format!(
                    "Invalid floor drain spec '{}'. Format: floor:radius or floor:x,z:radius",
                    spec
                ));
            }
        }
        "ceiling" | "top" => {
            if parts.len() == 2 {
                let r = parse_f32_checked(&parts[1], "radius", Some(0.0))?;
                (CubeFace::Ceiling, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xz: Vec<&str> = parts[1].split(',').collect();
                if xz.len() != 2 {
                    return Err(format!(
                        "Expected 2 coords (x,z) for ceiling drain offset, got '{}'",
                        parts[1]
                    ));
                }
                let x = xz[0]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid x: {}", e))?;
                let z = xz[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid z: {}", e))?;
                let r = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
                (CubeFace::Ceiling, glam::Vec3::new(x, 0.0, z), r)
            } else {
                return Err(format!(
                    "Invalid ceiling drain spec '{}'. Format: ceiling:radius or ceiling:x,z:radius",
                    spec
                ));
            }
        }
        "left" => {
            if parts.len() == 2 {
                let r = parse_f32_checked(&parts[1], "radius", Some(0.0))?;
                (CubeFace::Left, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let yz: Vec<&str> = parts[1].split(',').collect();
                if yz.len() != 2 {
                    return Err(format!(
                        "Expected 2 coords (y,z) for left wall drain offset, got '{}'",
                        parts[1]
                    ));
                }
                let y = yz[0]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid y: {}", e))?;
                let z = yz[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid z: {}", e))?;
                let r = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
                (CubeFace::Left, glam::Vec3::new(0.0, y, z), r)
            } else {
                return Err(format!(
                    "Invalid left drain spec '{}'. Format: left:radius or left:y,z:radius",
                    spec
                ));
            }
        }
        "right" => {
            if parts.len() == 2 {
                let r = parse_f32_checked(&parts[1], "radius", Some(0.0))?;
                (CubeFace::Right, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let yz: Vec<&str> = parts[1].split(',').collect();
                if yz.len() != 2 {
                    return Err(format!(
                        "Expected 2 coords (y,z) for right wall drain offset, got '{}'",
                        parts[1]
                    ));
                }
                let y = yz[0]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid y: {}", e))?;
                let z = yz[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid z: {}", e))?;
                let r = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
                (CubeFace::Right, glam::Vec3::new(0.0, y, z), r)
            } else {
                return Err(format!(
                    "Invalid right drain spec '{}'. Format: right:radius or right:y,z:radius",
                    spec
                ));
            }
        }
        "back" => {
            if parts.len() == 2 {
                let r = parse_f32_checked(&parts[1], "radius", Some(0.0))?;
                (CubeFace::Back, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xy: Vec<&str> = parts[1].split(',').collect();
                if xy.len() != 2 {
                    return Err(format!(
                        "Expected 2 coords (x,y) for back wall drain offset, got '{}'",
                        parts[1]
                    ));
                }
                let x = xy[0]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid x: {}", e))?;
                let y = xy[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid y: {}", e))?;
                let r = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
                (CubeFace::Back, glam::Vec3::new(x, y, 0.0), r)
            } else {
                return Err(format!(
                    "Invalid back drain spec '{}'. Format: back:radius or back:x,y:radius",
                    spec
                ));
            }
        }
        "front" => {
            if parts.len() == 2 {
                let r = parse_f32_checked(&parts[1], "radius", Some(0.0))?;
                (CubeFace::Front, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xy: Vec<&str> = parts[1].split(',').collect();
                if xy.len() != 2 {
                    return Err(format!(
                        "Expected 2 coords (x,y) for front wall drain offset, got '{}'",
                        parts[1]
                    ));
                }
                let x = xy[0]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid x: {}", e))?;
                let y = xy[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid y: {}", e))?;
                let r = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
                (CubeFace::Front, glam::Vec3::new(x, y, 0.0), r)
            } else {
                return Err(format!(
                    "Invalid front drain spec '{}'. Format: front:radius or front:x,y:radius",
                    spec
                ));
            }
        }
        "custom" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid custom drain spec '{}'. Format: custom:x,y,z:radius",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let r = parse_f32_checked(&parts[2], "radius", Some(0.0))?;
            (CubeFace::Custom, pos, r)
        }
        _ => {
            return Err(format!(
                "Unknown drain face '{}'. Expected 'floor', 'ceiling', 'left', 'right', 'back', 'front', or 'custom'",
                kind
            ))
        }
    };

    let mut d = Drain::new(name, face, pos, radius);
    d.id = (index + 1) as u32;
    Ok(d)
}

fn parse_chamber_floats(raw: &str, flag_name: &str) -> Result<Vec<f32>, String> {
    let cleaned = raw.replace(['[', ']', '(', ')'], "");
    let parts: Vec<&str> = cleaned.split([',', 'x', 'X']).collect();
    let mut vals = Vec::with_capacity(parts.len());
    for part in parts {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(format!("Invalid {} spec '{}': empty value", flag_name, raw));
        }
        let v = parse_f32_checked(trimmed, flag_name, Some(0.0))?;
        vals.push(v);
    }
    Ok(vals)
}

fn parse_chamber_with_base(
    args: &Args,
    base_scene: Option<&SceneConfigFile>,
    is_2d: bool,
) -> Result<Chamber, String> {
    let mut chamber = if let Some(sc) = base_scene {
        sc.chamber.to_chamber()
    } else {
        Chamber::new(ChamberShape::Box)
    };

    if let Some(ref shape_raw) = args.chamber_shape {
        chamber.shape = match shape_raw.trim().to_lowercase().as_str() {
            "box" | "cuboid" | "rect" | "rectangle" | "" => ChamberShape::Box,
            "sphere" | "globe" | "ball" | "circle" => ChamberShape::Sphere,
            "bathtub" | "tub" | "bath" => ChamberShape::Bathtub,
            other => {
                return Err(format!(
                    "Unknown bounding box / chamber type '{}'. Valid options: box, sphere, bathtub",
                    other
                ))
            }
        };
    }

    if let Some(ref size_str) = args.chamber_size {
        let vals = parse_chamber_floats(size_str, "chamber-size")?;
        chamber.base_size = match vals.as_slice() {
            [s] => {
                if is_2d {
                    Vec3::new(*s, *s, 0.1)
                } else {
                    Vec3::splat(*s)
                }
            }
            [w, h] => Vec3::new(*w, *h, if is_2d { 0.1 } else { 0.9 }),
            [w, h, d] => Vec3::new(*w, *h, *d),
            _ => {
                return Err(format!(
                    "Invalid chamber size '{}'. Expected D, W,H (2D), or W,H,D (e.g. 2.0,1.5,1.0)",
                    size_str
                ))
            }
        };
        chamber.custom_size = true;
    }

    if let Some(ref scale_str) = args.chamber_scale {
        let vals = parse_chamber_floats(scale_str, "chamber-scale")?;
        chamber.scale = match vals.as_slice() {
            [s] => Vec3::splat(*s),
            [x, y] => Vec3::new(*x, *y, 1.0),
            [x, y, z] => Vec3::new(*x, *y, *z),
            _ => {
                return Err(format!(
                    "Invalid chamber scale '{}'. Expected S, X,Y, or X,Y,Z (e.g. 1.5 or 1.5,1.2,1.0)",
                    scale_str
                ))
            }
        };
    }

    if let Some(slope) = args.tub_slope {
        chamber.tub_slope_ratio = slope.clamp(0.05, 0.5);
    }
    if let Some(fillet) = args.tub_fillet {
        chamber.tub_fillet_ratio = fillet.clamp(0.01, 0.45);
    }

    Ok(chamber)
}

#[cfg(test)]
fn parse_chamber(args: &Args) -> Result<Chamber, String> {
    parse_chamber_with_base(args, None, args.mode_2d)
}

#[allow(clippy::too_many_arguments)]
fn run_headless(
    args: Args,
    loaded_scene: Option<SceneConfigFile>,
    is_2d: bool,
    preset: ScenePreset,
    quality: SimulationQuality,
    chamber: Chamber,
    parsed_colliders: Vec<Collider>,
    parsed_faucets: Vec<Faucet>,
    parsed_drains: Vec<Drain>,
) {
    let cli_render_quality = resolve_cli_render_quality(&args, quality);
    let (video_width, video_height) = resolve_video_dimensions(&args);
    let fps = args.fps.max(1);
    let sim_speed = args
        .speed
        .unwrap_or_else(|| loaded_scene.as_ref().map_or(1.0, |s| s.sim_speed))
        .clamp(0.05, 4.0);

    let render_mode = if args.surface_mesh {
        RenderMode::SurfaceMesh
    } else if let Some(ref rm) = args.render_mode {
        RenderMode::from_slug(rm)
    } else if let Some(ref sc) = loaded_scene {
        sc.visuals.parsed_render_mode()
    } else {
        RenderMode::Particles
    };

    let mut mesh_cfg = build_mesh_config(&args, loaded_scene.as_ref());
    cli_render_quality.apply_to_mesh_config(
        &mut mesh_cfg,
        args.mesh_smooth.is_some(),
        args.mesh_scale.is_some(),
    );
    let (mut camera, orbit_deg) = build_camera(&args, loaded_scene.as_ref(), is_2d);
    let orbit_rad_per_sec = orbit_deg.to_radians();
    let wall_frame_dt = 1.0 / fps as f32;

    let color_mode = args
        .color_mode
        .as_deref()
        .map(parse_color_mode)
        .unwrap_or_else(|| {
            loaded_scene
                .as_ref()
                .map_or(ColorMode::Velocity, |s| s.visuals.parsed_color_mode())
        });
    let color_map = args
        .color_map
        .as_deref()
        .map(parse_color_map)
        .unwrap_or_else(|| {
            loaded_scene
                .as_ref()
                .map_or(ColorMapType::Ocean, |s| s.visuals.parsed_color_map())
        });
    let particle_scale = args
        .particle_scale
        .unwrap_or_else(|| {
            loaded_scene
                .as_ref()
                .map_or(1.25, |s| s.visuals.particle_scale)
        })
        .clamp(0.3, 3.0);
    let particle_foam = if args.no_particle_foam {
        false
    } else {
        loaded_scene
            .as_ref()
            .map_or(true, |s| s.visuals.particle_foam)
    };
    let lit_spheres = if args.unlit_spheres {
        false
    } else {
        loaded_scene
            .as_ref()
            .map_or(true, |s| s.visuals.lit_spheres)
    };
    let use_culling = if args.no_culling {
        false
    } else {
        args.culling
            && loaded_scene
                .as_ref()
                .map_or(true, |s| s.visuals.enable_culling)
    };

    let save_scene_only = args.save_scene.is_some()
        && !args.headless
        && !args.benchmark
        && args.export_mp4.is_none()
        && args.export_png.is_none()
        && args.export_ply.is_none()
        && args.export_obj.is_none()
        && args.time.is_none()
        && args.frames.is_none();

    let (total_frames, dt) = if let Some(t) = args.time {
        let f = (t * fps as f32).round().max(0.0) as usize;
        let delta_t = (1.0 / fps as f32) * sim_speed;
        (f, delta_t)
    } else {
        let f = args.frames.unwrap_or(if save_scene_only { 0 } else { 120 });
        let base_dt = args.dt.unwrap_or(1.0 / fps as f32);
        let delta_t = base_dt * sim_speed;
        (f, delta_t)
    };

    let bounds = chamber.compute_bounds();
    let mut config = if let Some(ref sc) = loaded_scene {
        sc.solver.to_config()
    } else {
        SolverConfig::default()
    };
    let mut fluid_preset = loaded_scene
        .as_ref()
        .map_or(FluidPhysicsPreset::Water, |s| s.parsed_fluid_preset());

    if loaded_scene.is_none() || args.quality.is_some() {
        quality.apply(&mut config, is_2d);
    }

    if let Some(target_n) = args.particles {
        let approx_vol = match preset {
            ScenePreset::DamBreak => 0.415f32,
            ScenePreset::DoubleDamBreak => 0.694f32,
            ScenePreset::DropletSplash => 0.35f32,
            ScenePreset::ObstacleDrop => 0.38f32,
            ScenePreset::VortexTornado => 0.30f32,
            ScenePreset::ZeroGravityBlobs => 0.25f32,
            ScenePreset::Empty => 0.40f32,
        };
        let s = (approx_vol / (target_n as f32).max(100.0)).cbrt();
        config.spacing = s;
        config.particle_radius = s * 0.58;
        config.smoothing_length = config.particle_radius * 3.1;
    }

    if let Some(r) = args.particle_radius {
        config.particle_radius = r;
        config.smoothing_length = r * 3.1;
    }
    if let Some(s) = args.spacing {
        config.spacing = s;
    }
    config.rest_density = if is_2d {
        compute_lattice_rest_density_2d(config.smoothing_length, config.spacing)
    } else {
        compute_lattice_rest_density(config.smoothing_length, config.spacing)
    };

    let loaded_solver_override = loaded_scene.as_ref().map(|_| config.clone());

    let mut solver = SphSolver::new(bounds, config);
    if is_2d {
        solver.set_2d(true);
    }
    solver.set_chamber(chamber);
    SceneGenerator::setup_scene(&mut solver, preset);

    // Preserve loaded JSON solver parameters if no preset override was requested, then apply explicit CLI physics flags
    if let Some(sc_cfg) = loaded_solver_override {
        if args.preset.is_none() {
            solver.config = sc_cfg;
        }
    }
    apply_cli_physics_overrides(&args, &mut solver.config, &mut fluid_preset);

    if let Some(ref sc) = loaded_scene {
        if args.preset.is_none() {
            solver.colliders.clear();
        }
        for (i, c_dto) in sc.colliders.iter().enumerate() {
            if let Some(col) = c_dto.to_collider((i + 1) as u32) {
                solver.colliders.push(col);
            }
        }
        for (i, f_dto) in sc.faucets.iter().enumerate() {
            solver.faucets.push(f_dto.to_faucet((i + 1) as u32));
        }
        for (i, d_dto) in sc.drains.iter().enumerate() {
            solver.drains.push(d_dto.to_drain((i + 1) as u32));
        }
    }

    let base_col_id = solver.colliders.len() as u32;
    for (idx, mut col) in parsed_colliders.into_iter().enumerate() {
        col.id = base_col_id + idx as u32 + 1;
        solver.colliders.push(col);
    }
    let base_faucet_id = solver.faucets.len() as u32;
    for (idx, mut faucet) in parsed_faucets.into_iter().enumerate() {
        faucet.id = base_faucet_id + idx as u32 + 1;
        solver.faucets.push(faucet);
    }
    let base_drain_id = solver.drains.len() as u32;
    for (idx, mut drain) in parsed_drains.into_iter().enumerate() {
        drain.id = base_drain_id + idx as u32 + 1;
        solver.drains.push(drain);
    }

    // Save portable .json scene if --save-scene was requested
    if let Some(ref save_path) = args.save_scene {
        let scene_out = SceneConfigFile {
            version: 1,
            preset: preset.slug().to_string(),
            quality: quality.slug().to_string(),
            fluid_preset: fluid_preset.slug().to_string(),
            is_2d,
            sim_speed,
            mouse_strength: solver.mouse_force_strength,
            solver: SolverConfigDto::from_config(&solver.config),
            chamber: ChamberDto::from_chamber(&solver.chamber),
            colliders: solver
                .colliders
                .iter()
                .map(ColliderDto::from_collider)
                .collect(),
            faucets: solver
                .faucets
                .iter()
                .map(FaucetDto::from_faucet)
                .collect(),
            drains: solver.drains.iter().map(DrainDto::from_drain).collect(),
            camera: CameraDto::from_camera(&camera, orbit_deg),
            visuals: VisualsDto {
                render_mode: render_mode.slug().to_string(),
                color_mode: color_mode.slug().to_string(),
                color_map: color_map.slug().to_string(),
                particle_scale,
                particle_foam,
                lit_spheres,
                enable_culling: use_culling,
                mesh: SurfaceMeshDto::from_config(&mesh_cfg),
            },
        };
        match scene_out.save_to_path(save_path) {
            Ok(_) => println!("Saved studio scene JSON to: {}", save_path.display()),
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        if save_scene_only {
            return;
        }
    }

    println!("=== liquid-rust studio render / headless mode ===");
    println!("Preset:      {} ({:?})", preset.name(), preset);
    println!("Substance:   {}", fluid_preset.name());
    println!("Quality:     {} ({:?})", quality.name_for(is_2d), quality);
    println!("Render Mode: {} | Env: {}", render_mode.name(), mesh_cfg.env_preset.name());
    if cli_render_quality != CliRenderQuality::Standard
        || args.export_mp4.is_some()
        || args.export_png.is_some()
        || args.export_obj.is_some()
    {
        println!(
            "CLI Optics:  {} | SSAA {}x | Shadow {}px | Mesh max {:.0} cells | H.264 {} CRF {}",
            cli_render_quality.name(),
            cli_render_quality.ssaa_factor(),
            cli_render_quality.shadow_res(),
            mesh_cfg.max_grid_cells,
            cli_render_quality.ffmpeg_preset(),
            cli_render_quality.ffmpeg_crf(),
        );
    }
    if render_mode == RenderMode::SurfaceMesh {
        println!(
            "Mesh Optics: Material={} | Shallow={} | Deep={} | Texture={} (scale {:.1}, str {:.2}) | IOR={:.3} | Rough={:.3}",
            mesh_cfg.preset.name(),
            SurfaceMeshConfig::format_hex(mesh_cfg.shallow_color),
            SurfaceMeshConfig::format_hex(mesh_cfg.deep_color),
            mesh_cfg.texture_mode.name(),
            mesh_cfg.texture_scale,
            mesh_cfg.texture_strength,
            mesh_cfg.ior,
            mesh_cfg.roughness,
        );
    }
    if (sim_speed - 1.0).abs() > 1e-4 {
        println!("Sim Speed:   {:.2}x", sim_speed);
    }
    println!(
        "Timeline:    {} frames ({:.2}s @ {} fps, dt={:.5}s)",
        total_frames,
        total_frames as f32 / fps as f32,
        fps,
        dt
    );

    let bsz = solver.bounds.size();
    println!(
        "Bounds:      {} ({:.2} x {:.2} x {:.2} m)",
        solver.chamber.shape.name(),
        bsz.x,
        bsz.y,
        bsz.z
    );
    for col in &solver.colliders {
        println!("  + Obstacle: {} (id: {})", col.name, col.id);
    }
    for faucet in &solver.faucets {
        println!(
            "  + Faucet: {} (id: {}, rate: {:.0} p/s)",
            faucet.name, faucet.id, faucet.flow_rate
        );
    }
    for drain in &solver.drains {
        println!(
            "  + Drain: {} (id: {}, face: {}, radius: {:.2})",
            drain.name,
            drain.id,
            drain.face.name(),
            drain.radius
        );
    }

    let particle_count = solver.particle_count();
    println!(
        "Spawned {} particles (radius: {:.4}m, spacing: {:.4}m)",
        particle_count, solver.config.particle_radius, solver.config.spacing
    );

    let mut gpu_solver = if !args.cpu && total_frames > 0 {
        match GpuSphSolver::new(solver.bounds, &solver.config, &solver.particles) {
            Ok(g) => {
                println!("GPU Acceleration: ACTIVE [{}]", g.device_name);
                Some(g)
            }
            Err(e) => {
                println!(
                    "GPU compute unavailable ({}), using multi-threaded Rayon CPU solver",
                    e
                );
                None
            }
        }
    } else if args.cpu {
        println!("Using multi-threaded Rayon CPU solver (--cpu specified)");
        None
    } else {
        None
    };

    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = Arc::clone(&interrupted);
    let _ = ctrlc::set_handler(move || {
        interrupted_clone.store(true, Ordering::SeqCst);
    });

    let mut mp4_encoder = if let Some(ref path) = args.export_mp4 {
        println!(
            "Encoding MP4 video ({}x{} @ {} fps, {}, {}) to: {}",
            video_width,
            video_height,
            fps,
            render_mode.slug(),
            cli_render_quality.slug(),
            path.display()
        );
        match Mp4Encoder::new_with_encoding(
            path,
            video_width,
            video_height,
            fps,
            cli_render_quality.ffmpeg_preset(),
            cli_render_quality.ffmpeg_crf(),
        ) {
            Ok(enc) => Some(enc),
            Err(e) => {
                eprintln!("Error starting MP4 encoder: {}", e);
                None
            }
        }
    } else {
        None
    };

    let mut software_renderer = if mp4_encoder.is_some() || args.export_png.is_some() {
        let mut rend =
            SoftwareRenderer::new_with_quality(video_width, video_height, cli_render_quality);
        rend.enable_culling = use_culling;
        rend.lit_spheres = lit_spheres;
        rend.particle_foam = particle_foam;
        rend.env_preset = mesh_cfg.env_preset;
        rend.exposure = mesh_cfg.exposure;
        rend.show_grid = mesh_cfg.show_grid;
        rend.show_wireframe = mesh_cfg.show_wireframe;
        rend.light_dir = mesh_cfg.light_dir;
        rend.light_intensity = mesh_cfg.light_intensity;
        rend.ambient_intensity = mesh_cfg.ambient_intensity;

        if rend.ssaa_factor != cli_render_quality.ssaa_factor() {
            let (rw, rh) = rend.internal_dimensions();
            println!(
                "  (SSAA automatically capped to {}x [{}x{}] at {}x{} to bound RAM within 36M internal pixels)",
                rend.ssaa_factor, rw, rh, video_width, video_height
            );
        }
        Some(rend)
    } else {
        None
    };

    if let Some(ref dir) = args.export_ply {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Failed to create PLY directory {}: {}", dir.display(), e);
            return;
        }
        println!("Exporting PLY point-cloud sequence to: {}", dir.display());
    }
    if let Some(ref dir) = args.export_obj {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Failed to create OBJ directory {}: {}", dir.display(), e);
            return;
        }
        println!("Exporting Wavefront .OBJ surface mesh sequence to: {}", dir.display());
    }

    let start_total = Instant::now();
    let mut total_sim_time = 0.0f64;
    let mut total_mesh_render_time = 0.0f64;
    let mut last_progress_print = Instant::now();
    let mut completed_frames = 0;
    let mut anim_time = 0.0f32;

    for frame in 0..total_frames {
        if interrupted.load(Ordering::SeqCst) {
            println!(
                "\n\n[Ctrl+C] Gracefully finalizing render at frame ({}/{})...",
                completed_frames, total_frames
            );
            break;
        }

        let t0 = Instant::now();
        if let Some(ref mut gpu) = gpu_solver {
            let prev_len = solver.particles.len();
            solver.emit_faucets(dt);
            if solver.particles.len() != prev_len {
                gpu.sync_particles(solver.bounds, &solver.config, &solver.particles, true);
            }
            gpu.step(
                solver.bounds,
                &solver.chamber,
                &solver.config,
                &solver.colliders,
                solver.mouse_ray,
                solver.mouse_force_strength,
                solver.mouse_force_mode,
                dt,
                is_2d,
            );
            gpu.read_particles(&mut solver.particles);

            let has_mesh_col = solver
                .colliders
                .iter()
                .any(|c| c.enabled && matches!(c.shape, ColliderShape::Mesh(_)));
            if has_mesh_col {
                use rayon::prelude::*;
                let radius = solver.config.particle_radius;
                let colliders = &solver.colliders;
                let hit = solver
                    .particles
                    .par_iter_mut()
                    .map(|p| {
                        let mut p_hit = false;
                        for col in colliders {
                            if let ColliderShape::Mesh(_) = &col.shape {
                                p_hit |= col.resolve(&mut p.position, &mut p.velocity, radius);
                            }
                        }
                        if p_hit && is_2d {
                            p.position.z = 0.0;
                            p.velocity.z = 0.0;
                        }
                        p_hit
                    })
                    .reduce(|| false, |a, b| a || b);
                if hit {
                    gpu.sync_particles(solver.bounds, &solver.config, &solver.particles, false);
                }
            }

            let drained = solver.drain_particles();
            if drained > 0 {
                gpu.sync_particles(solver.bounds, &solver.config, &solver.particles, false);
            }
        } else {
            solver.step(dt);
        }
        anim_time += dt;
        total_sim_time += t0.elapsed().as_secs_f64();

        // Build surface mesh if needed for MP4 surface-mesh rendering or .OBJ export
        let t_r = Instant::now();
        let need_mesh = (render_mode == RenderMode::SurfaceMesh && mp4_encoder.is_some())
            || args.export_obj.is_some();
        let built_mesh = if need_mesh {
            Some(SurfaceMesh::from_particles(
                &solver.particles,
                solver.config.particle_radius,
                solver.config.smoothing_length,
                &mesh_cfg,
            ))
        } else {
            None
        };

        if let (Some(ref mut enc), Some(ref mut rend)) = (&mut mp4_encoder, &mut software_renderer)
        {
            match render_mode {
                RenderMode::Particles => {
                    rend.render_particles_studio(
                        &solver.particles,
                        solver.config.particle_radius * particle_scale,
                        &solver.chamber,
                        solver.bounds,
                        &solver.colliders,
                        &camera,
                        color_mode,
                        color_map,
                        solver.config.rest_density,
                        is_2d,
                    );
                }
                RenderMode::SurfaceMesh => {
                    if let Some(ref mesh) = built_mesh {
                        rend.render_surface_mesh_frame(
                            mesh,
                            &mesh_cfg,
                            &camera,
                            &solver.chamber,
                            solver.bounds,
                            &solver.colliders,
                            anim_time,
                            is_2d,
                        );
                    }
                }
            }
            if let Err(e) = enc.write_frame(&rend.rgb_buffer) {
                eprintln!("Error writing video frame {}: {}", frame, e);
            }
        }

        camera.azimuth += orbit_rad_per_sec * wall_frame_dt;

        if let (Some(ref dir), Some(ref mesh)) = (&args.export_obj, &built_mesh) {
            let obj_path = dir.join(format!("fluid_surface_{:05}.obj", frame));
            if let Err(e) = mesh.save_obj(&obj_path) {
                eprintln!("Error exporting OBJ frame {}: {}", frame, e);
            }
        }
        total_mesh_render_time += t_r.elapsed().as_secs_f64();

        if let Some(ref dir) = args.export_ply {
            let path = dir.join(format!("fluid_{:05}.ply", frame));
            let rest_rho = solver.config.rest_density;
            let bnds = solver.bounds;
            let colors: Vec<[u8; 3]> = solver
                .particles
                .iter()
                .map(|p| {
                    ColorMap::particle_color(
                        p,
                        color_mode,
                        color_map,
                        rest_rho,
                        &bnds,
                        particle_foam,
                    )
                })
                .collect();
            if let Err(e) = PlyExporter::save_binary(&path, &solver.particles, &colors) {
                eprintln!("Error exporting PLY frame {}: {}", frame, e);
            }
        }

        completed_frames = frame + 1;
        let elapsed = start_total.elapsed().as_secs_f64();
        let should_print = frame == 0
            || completed_frames == total_frames
            || last_progress_print.elapsed().as_millis() >= 60;

        if should_print {
            let avg_time = elapsed / completed_frames as f64;
            let eta_secs = (total_frames - completed_frames) as f64 * avg_time;
            let percent = (completed_frames as f64 / total_frames as f64) * 100.0;
            let bar_len = 24;
            let filled =
                ((completed_frames as f64 / total_frames as f64) * bar_len as f64).round() as usize;
            let bar = format!(
                "[{}{}]",
                "=".repeat(filled.min(bar_len)),
                " ".repeat(bar_len.saturating_sub(filled))
            );
            let fps_current = completed_frames as f64 / elapsed.max(1e-4);

            use std::io::Write;
            print!(
                "\r{} {:5.1}% | Frame {:4}/{} | ETA: {:<7} | Elapsed: {:<6} | {:4.1} fps",
                bar,
                percent,
                completed_frames,
                total_frames,
                format_time(eta_secs),
                format_time(elapsed),
                fps_current,
            );
            let _ = std::io::stdout().flush();
            last_progress_print = Instant::now();
        }
    }
    if total_frames > 0 {
        println!();
    }

    // Export studio PNG screenshot of the final simulated state if --export-png was requested
    if let (Some(ref png_path), Some(ref mut rend)) = (&args.export_png, &mut software_renderer) {
        let t_png = Instant::now();
        match render_mode {
            RenderMode::Particles => {
                rend.render_particles_studio(
                    &solver.particles,
                    solver.config.particle_radius * particle_scale,
                    &solver.chamber,
                    solver.bounds,
                    &solver.colliders,
                    &camera,
                    color_mode,
                    color_map,
                    solver.config.rest_density,
                    is_2d,
                );
            }
            RenderMode::SurfaceMesh => {
                let mesh = SurfaceMesh::from_particles(
                    &solver.particles,
                    solver.config.particle_radius,
                    solver.config.smoothing_length,
                    &mesh_cfg,
                );
                rend.render_surface_mesh_frame(
                    &mesh,
                    &mesh_cfg,
                    &camera,
                    &solver.chamber,
                    solver.bounds,
                    &solver.colliders,
                    anim_time,
                    is_2d,
                );
            }
        }
        total_mesh_render_time += t_png.elapsed().as_secs_f64();
        match PngExporter::save_rgb(png_path, video_width, video_height, &rend.rgb_buffer) {
            Ok(_) => println!(
                "Saved {}x{} studio PNG image to: {}",
                video_width,
                video_height,
                png_path.display()
            ),
            Err(e) => {
                eprintln!("Error saving PNG {}: {}", png_path.display(), e);
                std::process::exit(1);
            }
        }
    }

    if let Some(enc) = mp4_encoder {
        println!(
            "Finalizing MP4 video encoding (saving {} frames)...",
            completed_frames
        );
        match enc.finish() {
            Ok(_) => println!("MP4 video saved successfully!"),
            Err(e) => eprintln!("Error finalizing MP4 video: {}", e),
        }
    }

    let total_elapsed = start_total.elapsed().as_secs_f64();
    let effective_frames = completed_frames.max(1);
    println!("--------------------------------------------------");
    println!("Render / Simulation complete!");
    println!("Total wall time:        {:.3}s", total_elapsed);
    println!("Frames completed:       {} / {}", completed_frames, total_frames);
    if completed_frames > 0 {
        println!(
            "Avg physics step:       {:.2} ms/frame",
            (total_sim_time / effective_frames as f64) * 1000.0
        );
    }
    if total_mesh_render_time > 0.001 {
        println!(
            "Avg mesh + rasterize:   {:.2} ms/frame",
            (total_mesh_render_time / effective_frames as f64) * 1000.0
        );
    }
    if completed_frames > 0 {
        println!(
            "Throughput:             {:.0} particle-steps/sec",
            (solver.particle_count() * completed_frames) as f64 / total_sim_time.max(1e-5)
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gui(
    args: Args,
    loaded_scene: Option<SceneConfigFile>,
    is_2d: bool,
    initial_preset: ScenePreset,
    initial_quality: SimulationQuality,
    initial_chamber: Chamber,
    parsed_colliders: Vec<Collider>,
    parsed_faucets: Vec<Faucet>,
    parsed_drains: Vec<Drain>,
) -> Result<(), eframe::Error> {
    let window_title = "LiquidRust Studio — 2D/3D SPH & Surface Meshing";

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1380.0, 860.0])
            .with_min_inner_size([940.0, 560.0])
            .with_title(window_title),
        renderer: eframe::Renderer::Glow,
        depth_buffer: 24,
        ..Default::default()
    };

    let render_mode_override = if args.surface_mesh {
        Some(RenderMode::SurfaceMesh)
    } else {
        args.render_mode.as_deref().map(RenderMode::from_slug)
    };
    let mesh_cfg = build_mesh_config(&args, loaded_scene.as_ref());
    let (cam_override, orbit_override) = build_camera(&args, loaded_scene.as_ref(), is_2d);
    let has_cam_override = loaded_scene.is_some()
        || args.camera_azimuth.is_some()
        || args.camera_elevation.is_some()
        || args.camera_distance.is_some()
        || args.camera_target.is_some()
        || args.fov.is_some()
        || args.camera_orbit.is_some();

    eframe::run_native(
        "liquid-rust",
        native_options,
        Box::new(move |cc| {
            let mut app = LiquidApp::new(cc);

            if let Some(ref sc) = loaded_scene {
                app.apply_scene_config_file(sc);
            } else {
                if is_2d {
                    app.set_2d(true);
                }
                app.solver.set_chamber(initial_chamber.clone());
                app.load_preset(initial_preset);
                if initial_quality != SimulationQuality::Medium {
                    app.set_quality(initial_quality);
                }
            }

            if args.mode_2d && !app.is_2d {
                app.set_2d(true);
            }
            if args.chamber_shape.is_some()
                || args.chamber_size.is_some()
                || args.chamber_scale.is_some()
                || args.tub_slope.is_some()
                || args.tub_fillet.is_some()
            {
                app.solver.set_chamber(initial_chamber);
            }
            if args.preset.is_some() {
                app.load_preset(initial_preset);
            }
            if args.quality.is_some() {
                app.set_quality(initial_quality);
            }
            if let Some(spd) = args.speed {
                app.sim_speed = spd.clamp(0.05, 4.0);
            }
            if args.cpu {
                app.use_gpu = false;
            }

            let mut fp = app.fluid_preset;
            apply_cli_physics_overrides(&args, &mut app.solver.config, &mut fp);
            app.fluid_preset = fp;

            if let Some(rm) = render_mode_override {
                app.render_mode = rm;
            }
            app.mesh_config = mesh_cfg;
            app.sync_sun_angles_from_dir();
            if let Some(ref cm) = args.color_mode {
                app.color_mode = parse_color_mode(cm);
            }
            if let Some(ref cmap) = args.color_map {
                app.color_map = parse_color_map(cmap);
            }
            if let Some(ps) = args.particle_scale {
                app.display_size_factor = ps.clamp(0.3, 3.0);
            }
            if args.no_particle_foam {
                app.particle_foam = false;
            }
            if args.unlit_spheres {
                app.lit_spheres = false;
            }
            if args.no_culling || !args.culling {
                app.enable_culling = false;
            }

            let (vw, vh) = resolve_video_dimensions(&args);
            app.video_width = vw;
            app.video_height = vh;
            if has_cam_override {
                app.camera = cam_override;
                app.camera_orbit_speed_deg = orbit_override;
            }

            for mut c in parsed_colliders {
                c.id = app.next_collider_id;
                app.next_collider_id += 1;
                app.solver.colliders.push(c);
            }
            for mut f in parsed_faucets {
                f.id = app.next_faucet_id;
                app.next_faucet_id += 1;
                app.solver.faucets.push(f);
            }
            for mut d in parsed_drains {
                d.id = app.next_drain_id;
                app.next_drain_id += 1;
                app.solver.drains.push(d);
            }

            if let Some(ref save_path) = args.save_scene {
                app.save_scene_json(&save_path.to_string_lossy());
            }

            Ok(Box::new(app))
        }),
    )
}

fn main() {
    liquid_rust::updater::cleanup_old_binary();
    let args = Args::parse();

    if args.check_update {
        std::process::exit(liquid_rust::updater::run_cli_check_update());
    }
    if args.update {
        std::process::exit(liquid_rust::updater::run_cli_self_update(
            args.force_update,
            false,
        ));
    }

    let loaded_scene = if let Some(ref scene_path) = args.scene {
        match SceneConfigFile::load_from_path(scene_path) {
            Ok(sc) => Some(sc),
            Err(e) => {
                eprintln!("Error loading --scene '{}': {}", scene_path.display(), e);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    let mut parsed_colliders = Vec::new();
    for (i, spec) in args.obstacles.iter().enumerate() {
        match parse_obstacle(spec, i) {
            Ok(c) => parsed_colliders.push(c),
            Err(e) => {
                eprintln!("Error parsing --obstacle '{}': {}", spec, e);
                std::process::exit(1);
            }
        }
    }

    let mut parsed_faucets = Vec::new();
    for (i, spec) in args.faucets.iter().enumerate() {
        match parse_faucet(spec, i) {
            Ok(f) => parsed_faucets.push(f),
            Err(e) => {
                eprintln!("Error parsing --faucet '{}': {}", spec, e);
                std::process::exit(1);
            }
        }
    }

    let mut parsed_drains = Vec::new();
    for (i, spec) in args.drains.iter().enumerate() {
        match parse_drain(spec, i) {
            Ok(d) => parsed_drains.push(d),
            Err(e) => {
                eprintln!("Error parsing --drain '{}': {}", spec, e);
                std::process::exit(1);
            }
        }
    }

    let is_2d = args.mode_2d || loaded_scene.as_ref().map_or(false, |s| s.is_2d);
    let preset = if let Some(ref p_str) = args.preset {
        parse_preset(Some(p_str), !parsed_faucets.is_empty())
    } else if let Some(ref sc) = loaded_scene {
        sc.parsed_preset()
    } else {
        parse_preset(None, !parsed_faucets.is_empty())
    };
    let quality = if let Some(ref q_str) = args.quality {
        parse_quality(q_str)
    } else if let Some(ref sc) = loaded_scene {
        sc.parsed_quality()
    } else {
        SimulationQuality::Medium
    };

    let chamber = match parse_chamber_with_base(&args, loaded_scene.as_ref(), is_2d) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error parsing bounding box / chamber flags: {}", e);
            std::process::exit(1);
        }
    };

    if args.headless
        || args.benchmark
        || args.export_mp4.is_some()
        || args.export_png.is_some()
        || args.export_ply.is_some()
        || args.export_obj.is_some()
        || args.save_scene.is_some()
    {
        run_headless(
            args,
            loaded_scene,
            is_2d,
            preset,
            quality,
            chamber,
            parsed_colliders,
            parsed_faucets,
            parsed_drains,
        );
    } else {
        let mut initial_quality = quality;
        if initial_quality.is_cli_only() {
            eprintln!(
                "Note: Quality '{}' is a high-density rendering-CLI-only mode; clamping interactive GUI to 'extreme'. Use --headless, --export-mp4, or --export-png for offline CLI tiers.",
                initial_quality.slug()
            );
            initial_quality = SimulationQuality::Extreme;
        }
        if let Err(e) = run_gui(
            args,
            loaded_scene,
            is_2d,
            preset,
            initial_quality,
            chamber,
            parsed_colliders,
            parsed_faucets,
            parsed_drains,
        ) {
            eprintln!("GUI Error: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_readable_cli_resolutions() {
        assert_eq!(parse_resolution_spec("480p"), Some((854, 480)));
        assert_eq!(parse_resolution_spec("720p"), Some((1280, 720)));
        assert_eq!(parse_resolution_spec("hd"), Some((1280, 720)));
        assert_eq!(parse_resolution_spec("1k"), Some((1920, 1080)));
        assert_eq!(parse_resolution_spec("1080p"), Some((1920, 1080)));
        assert_eq!(parse_resolution_spec("fhd"), Some((1920, 1080)));
        assert_eq!(parse_resolution_spec("2k"), Some((2560, 1440)));
        assert_eq!(parse_resolution_spec("1440p"), Some((2560, 1440)));
        assert_eq!(parse_resolution_spec("3k"), Some((3200, 1800)));
        assert_eq!(parse_resolution_spec("4k"), Some((3840, 2160)));
        assert_eq!(parse_resolution_spec("2160p"), Some((3840, 2160)));
        assert_eq!(parse_resolution_spec("uhd"), Some((3840, 2160)));
        assert_eq!(parse_resolution_spec("5k"), Some((5120, 2880)));
        assert_eq!(parse_resolution_spec("6k"), Some((6144, 3456)));
        assert_eq!(parse_resolution_spec("8k"), Some((7680, 4320)));
        assert_eq!(parse_resolution_spec("4320p"), Some((7680, 4320)));
        assert_eq!(parse_resolution_spec("1080p-vertical"), Some((1080, 1920)));
        assert_eq!(parse_resolution_spec("4K_Portrait"), Some((2160, 3840)));
        assert_eq!(parse_resolution_spec("2560x1080"), Some((2560, 1080)));
        assert_eq!(parse_resolution_spec("bad-res"), None);
    }

    #[test]
    fn test_cli_bounding_box_flags_and_setup_scene() {
        let args = Args::try_parse_from([
            "liquid-rust",
            "--box-type",
            "bathtub",
            "--box-size",
            "2.4x1.5x1.1",
            "--box-scale",
            "1.0",
            "--tub-slope",
            "0.35",
            "--tub-fillet",
            "0.20",
        ])
        .expect("bounding box flags should parse");
        let chamber = parse_chamber(&args).expect("chamber should parse");
        assert_eq!(chamber.shape, ChamberShape::Bathtub);
        assert!(chamber.custom_size);
        assert!((chamber.base_size - Vec3::new(2.4, 1.5, 1.1)).length() < 1e-5);
        assert!((chamber.tub_slope_ratio - 0.35).abs() < 1e-5);
        assert!((chamber.tub_fillet_ratio - 0.20).abs() < 1e-5);

        let mut solver = SphSolver::new(chamber.compute_bounds(), SolverConfig::default());
        solver.set_chamber(chamber);
        SceneGenerator::setup_scene(&mut solver, ScenePreset::Empty);
        let sz = solver.bounds.size();
        assert!((sz - Vec3::new(2.4, 1.5, 1.1)).length() < 1e-4);

        // 2D bounding box with W,H shorthand
        let args_2d = Args::try_parse_from([
            "liquid-rust",
            "--2d",
            "--bounds-type",
            "box",
            "--bounds-size",
            "3.0,1.6",
        ])
        .expect("2D bounds flags should parse");
        let ch_2d = parse_chamber(&args_2d).expect("2D chamber should parse");
        let mut solver_2d = SphSolver::new(ch_2d.compute_bounds(), SolverConfig::default());
        solver_2d.set_2d(true);
        solver_2d.set_chamber(ch_2d);
        SceneGenerator::setup_scene(&mut solver_2d, ScenePreset::DamBreak);
        let sz_2d = solver_2d.bounds.size();
        assert!((sz_2d.x - 3.0).abs() < 1e-4);
        assert!((sz_2d.y - 1.6).abs() < 1e-4);
        assert!((sz_2d.z - 0.1).abs() < 1e-4);

        // Invalid box type errors cleanly
        let bad_args = Args::try_parse_from(["liquid-rust", "--box-type", "pyramid"]).unwrap();
        assert!(parse_chamber(&bad_args).is_err());
    }

    #[test]
    fn test_cli_faucet_and_tap_radius() {
        use liquid_rust::sph::FaucetShape;
        let tap_with_radius = parse_faucet("tap:-0.5,0.9,0:0.12:550", 0).unwrap();
        assert_eq!(tap_with_radius.shape, FaucetShape::Circle { radius: 0.12 });
        assert!((tap_with_radius.flow_rate - 550.0).abs() < 1e-4);

        let tap_shorthand_radius = parse_faucet("tap:0,0.8,0:0.09", 1).unwrap();
        assert_eq!(tap_shorthand_radius.shape, FaucetShape::Circle { radius: 0.09 });
        assert!((tap_shorthand_radius.flow_rate - 400.0).abs() < 1e-4);

        // Legacy tap:x,y,z:flow_rate stays backward compatible
        let tap_legacy_rate = parse_faucet("tap:0,0.8,0:650", 2).unwrap();
        assert_eq!(tap_legacy_rate.shape, FaucetShape::Circle { radius: 0.05 });
        assert!((tap_legacy_rate.flow_rate - 650.0).abs() < 1e-4);

        let tap_point = parse_faucet("tap:0,0.8,0:0:300", 3).unwrap();
        assert_eq!(tap_point.shape, FaucetShape::Point);
        assert!((tap_point.flow_rate - 300.0).abs() < 1e-4);

        let stream_custom_r =
            parse_faucet("stream:-0.5,0.9,0:1.8,-1.0,0:450:0.02:0.115", 4).unwrap();
        assert_eq!(stream_custom_r.shape, FaucetShape::Circle { radius: 0.115 });
        assert!((stream_custom_r.flow_rate - 450.0).abs() < 1e-4);
        assert!((stream_custom_r.spray_spread - 0.02).abs() < 1e-4);
    }

    #[test]
    fn test_cli_substance_physics_and_studio_flags() {
        let args = Args::try_parse_from([
            "liquid-rust",
            "--fluid-preset",
            "mercury",
            "--gravity",
            "1.2,-3.71,0.0",
            "--air-drag",
            "0.60",
            "--restitution",
            "0.15",
            "--friction",
            "0.80",
            "--substeps",
            "5",
            "--iterations",
            "6",
            "--tensile-k",
            "0.0035",
            "--max-velocity",
            "22.0",
            "--wave-amp",
            "0.18",
            "--wave-freq",
            "1.25",
            "--color-mode",
            "foam",
            "--color-map",
            "inferno",
            "--env-preset",
            "sunset",
            "--exposure",
            "1.35",
            "--no-grid",
            "--no-wireframe",
            "--fov",
            "55.0",
            "--camera-orbit",
            "-24.0",
            "--obstacle",
            "cylinder:0,0.3,0:0.18:0.45:0.25:0.70",
        ])
        .expect("extended studio CLI flags should parse");

        let mut cfg = SolverConfig::default();
        let mut fp = FluidPhysicsPreset::Water;
        apply_cli_physics_overrides(&args, &mut cfg, &mut fp);
        assert_eq!(fp, FluidPhysicsPreset::Mercury);
        assert!((cfg.gravity - Vec3::new(1.2, -3.71, 0.0)).length() < 1e-5);
        assert!((cfg.air_drag - 0.60).abs() < 1e-5);
        assert!((cfg.restitution - 0.15).abs() < 1e-5);
        assert!((cfg.boundary_friction - 0.80).abs() < 1e-5);
        assert_eq!(cfg.substeps, 5);
        assert_eq!(cfg.solver_iterations, 6);
        assert!((cfg.tensile_k - 0.0035).abs() < 1e-6);
        assert!((cfg.max_velocity - 22.0).abs() < 1e-5);
        assert!((cfg.wave_amplitude - 0.18).abs() < 1e-5);
        assert!((cfg.wave_frequency - 1.25).abs() < 1e-5);

        assert_eq!(parse_color_mode(args.color_mode.as_deref().unwrap()), ColorMode::Foam);
        assert_eq!(parse_color_map(args.color_map.as_deref().unwrap()), ColorMapType::Inferno);

        let mcfg = build_mesh_config(&args, None);
        assert_eq!(mcfg.env_preset, EnvironmentPreset::GoldenSunset);
        assert!((mcfg.exposure - 1.35).abs() < 1e-5);
        assert!(!mcfg.show_grid);
        assert!(!mcfg.show_wireframe);

        let (cam, orb) = build_camera(&args, None, false);
        assert!((cam.fov_y.to_degrees() - 55.0).abs() < 1e-4);
        assert!((orb - -24.0).abs() < 1e-4);

        let cyl = parse_obstacle(&args.obstacles[0], 0).unwrap();
        assert!((cyl.restitution - 0.25).abs() < 1e-5);
        assert!((cyl.friction - 0.70).abs() < 1e-5);
    }
}
