use clap::Parser;
use eframe::egui;
use glam::Vec3;
use liquid_rust::export::{Mp4Encoder, PlyExporter, SoftwareRenderer};
use liquid_rust::gpu::GpuSphSolver;
use liquid_rust::render::{
    Camera, ColorMapType, ColorMode, FluidMaterialPreset, MeshTextureMode, RenderMode, SurfaceMesh,
    SurfaceMeshConfig,
};
use liquid_rust::sph::{
    compute_lattice_rest_density, Chamber, ChamberShape, Collider, ColliderShape, CubeFace, Drain,
    Faucet, SceneGenerator, ScenePreset, SimulationQuality, SolverConfig, SphSolver,
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
    about = "Studio-grade 2D/3D SPH Fluid Simulator in Rust with Surface Meshing, PBR Optics, Real-Time Viewport & MP4/OBJ/PLY Export"
)]
struct Args {
    /// Run in headless mode without opening GUI window
    #[arg(long)]
    headless: bool,

    /// Run CLI headless rendering in 2D mode (GUI uses the built-in 3D / 2D switch button)
    #[arg(long = "2d", alias = "two-d")]
    mode_2d: bool,

    /// Scene preset: dam-break, double-dam, splash, obstacle, tornado, zero-g, empty
    #[arg(long)]
    preset: Option<String>,

    /// Particle quality / resolution: draft, medium, high, ultra, cinematic, extreme (1M+)
    #[arg(long, short = 'q', default_value = "medium")]
    quality: String,

    /// Custom particle radius override (smaller = higher quality & finer fluid)
    #[arg(long)]
    particle_radius: Option<f32>,

    /// Custom particle spacing override (smaller = denser fluid lattice)
    #[arg(long)]
    spacing: Option<f32>,

    /// Target number of particles to spawn (automatically calculates optimal spacing and radius)
    #[arg(long, alias = "particle-count")]
    particles: Option<usize>,

    /// Total simulation duration in seconds (e.g. --time 4.0 for a 4-second video). Overrides --frames.
    #[arg(long, short = 't', alias = "duration")]
    time: Option<f32>,

    /// Number of simulation frames to run (used if --time is not specified)
    #[arg(long, default_value_t = 120)]
    frames: usize,

    /// Framerate (FPS) for video export and simulation stepping
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Simulation speed multiplier (e.g. 0.5 for 2x slow-mo, 0.25 for 4x slow-mo ASMR, 1.0 realtime)
    #[arg(long, alias = "sim-speed", default_value_t = 1.0)]
    speed: f32,

    /// Custom time step dt per frame in seconds (defaults to 1.0 / fps * speed)
    #[arg(long)]
    dt: Option<f32>,

    // === Selectable Fluid Render Mode & Surface Meshing ===
    /// Fluid rendering representation: 'particles' (SPH spheres) or 'mesh' (3D Marching Cubes surface mesh)
    #[arg(long, default_value = "particles")]
    render_mode: String,

    /// Enable continuous 3D surface meshing for renders & viewport (shortcut for --render-mode mesh)
    #[arg(long, alias = "mesh")]
    surface_mesh: bool,

    /// Surface mesh material preset: water, ocean, lagoon, amber, wine, emerald, milk, chrome, custom
    #[arg(long, default_value = "water")]
    mesh_material: String,

    /// Shallow / thin-film surface liquid color as hex '#RRGGBB' or 'R,G,B' (0..1 or 0..255)
    #[arg(long)]
    mesh_color: Option<String>,

    /// Deep volume Beer-Lambert absorption color as hex '#RRGGBB' or 'R,G,B'
    #[arg(long)]
    mesh_deep_color: Option<String>,

    /// Base liquid surface opacity [0.05..1.0]
    #[arg(long)]
    mesh_opacity: Option<f32>,

    /// Beer-Lambert optical volume absorption density [0.0..8.0]
    #[arg(long)]
    mesh_absorption: Option<f32>,

    /// Aerated whitewater wave-crest foam intensity [0.0..1.0]
    #[arg(long)]
    mesh_foam: Option<f32>,

    /// Procedural triplanar surface texture: none, ripples, caustics, foam, grid
    #[arg(long, default_value = "ripples")]
    mesh_texture: String,

    /// Spatial frequency scale for procedural surface texture (default: 6.0)
    #[arg(long)]
    mesh_texture_scale: Option<f32>,

    /// Bump & pattern strength for procedural surface texture [0.0..1.0] (default: 0.28)
    #[arg(long)]
    mesh_texture_strength: Option<f32>,

    /// Optical Index of Refraction (IOR) for Schlick Fresnel (e.g. 1.333 water, 1.52 glass, 2.40 chrome)
    #[arg(long)]
    mesh_ior: Option<f32>,

    /// GGX microfacet surface roughness [0.015..1.0] (smaller = mirror-sharp highlights)
    #[arg(long)]
    mesh_roughness: Option<f32>,

    /// Direct sun specular highlight intensity multiplier
    #[arg(long)]
    mesh_specular: Option<f32>,

    /// Screen-space refraction background distortion strength [0.0..1.5]
    #[arg(long)]
    mesh_refraction: Option<f32>,

    /// Subsurface wave-crest back-scatter glow intensity [0.0..2.0]
    #[arg(long)]
    mesh_sss: Option<f32>,

    /// Studio sky & horizon Fresnel reflection intensity [0.0..2.5]
    #[arg(long)]
    mesh_reflection: Option<f32>,

    /// Metallic reflectance factor [0.0..1.0]
    #[arg(long)]
    mesh_metallic: Option<f32>,

    /// Marching Cubes voxel size multiplier relative to particle radius (smaller = finer mesh, default: 0.72)
    #[arg(long)]
    mesh_scale: Option<f32>,

    /// Isosurface density threshold [0.15..0.80] (default: 0.42)
    #[arg(long)]
    mesh_iso: Option<f32>,

    /// Volume-preserving Taubin mesh smoothing iterations (default: 3)
    #[arg(long)]
    mesh_smooth: Option<usize>,

    /// Directional key sun vector 'x,y,z' (e.g. '0.42,0.82,0.38')
    #[arg(long, allow_hyphen_values = true)]
    light_dir: Option<String>,

    /// Key directional sun radiance multiplier (default: 1.25)
    #[arg(long)]
    light_intensity: Option<f32>,

    /// Hemisphere ambient light fill intensity (default: 0.32)
    #[arg(long)]
    ambient: Option<f32>,

    /// Disable colored liquid floor shadows & underwater caustics in surface-mesh renders
    #[arg(long)]
    no_shadows: bool,

    // === Physics & Particle Shading Overrides ===
    /// Vorticity confinement swirl strength (default: 0.045)
    #[arg(long)]
    vorticity: Option<f32>,

    /// Akinci free-surface cohesion / surface tension strength (default: 0.025)
    #[arg(long)]
    surface_tension: Option<f32>,

    /// XSPH artificial viscosity damping (default: 0.025)
    #[arg(long)]
    viscosity: Option<f32>,

    /// Particle color mode (for --render-mode particles): velocity, pressure, density, height, energy, single
    #[arg(long, default_value = "velocity")]
    color_mode: String,

    /// Particle colormap gradient (for --render-mode particles): ocean, turbo, viridis, fire
    #[arg(long, default_value = "ocean")]
    color_map: String,

    // === Camera Framing Overrides ===
    /// Camera horizontal orbit azimuth angle in degrees
    #[arg(long, allow_hyphen_values = true)]
    camera_azimuth: Option<f32>,

    /// Camera vertical orbit elevation angle in degrees
    #[arg(long, allow_hyphen_values = true)]
    camera_elevation: Option<f32>,

    /// Camera orbit distance from target in meters
    #[arg(long)]
    camera_distance: Option<f32>,

    /// Camera look-at target 'x,y,z'
    #[arg(long, allow_hyphen_values = true)]
    camera_target: Option<String>,

    // === Scene Objects ===
    /// Add custom obstacle: 'sphere:x,y,z:r', 'box:x,y,z:w,h,d', 'cylinder:x,y,z:r:h', or 'obj:path:x,y,z:scale'
    #[arg(long = "obstacle", action = clap::ArgAction::Append)]
    obstacles: Vec<String>,

    /// Add custom faucet: 'waterfall:x,y,z:width[:rate]', 'shower:x,y,z:radius[:rate[:spread]]', 'circle:x,y,z:radius[:rate]', 'tap:x,y,z[:rate]', 'stream:x,y,z:vx,vy,vz[:rate[:spread]]'
    #[arg(long = "faucet", action = clap::ArgAction::Append)]
    faucets: Vec<String>,

    /// Add custom drain: 'floor:radius' or 'floor:x,z:radius', '<face>:radius' (floor|ceiling|left|right|back|front), or 'custom:x,y,z:radius'
    #[arg(long = "drain", action = clap::ArgAction::Append)]
    drains: Vec<String>,

    /// Simulation chamber shape: box, sphere, bathtub
    #[arg(long, default_value = "box")]
    chamber_shape: String,

    /// Chamber scale factor: uniform (e.g. 1.5) or per-axis (e.g. 1.5,1.2,1.0)
    #[arg(long)]
    chamber_scale: Option<String>,

    /// Chamber bounding dimensions: width,height,depth (e.g. 2.0,1.5,1.0)
    #[arg(long)]
    chamber_size: Option<String>,

    // === Outputs & Compute ===
    /// Export simulation video to MP4 file via ffmpeg (e.g. output.mp4)
    #[arg(long)]
    export_mp4: Option<PathBuf>,

    /// Video export width in pixels
    #[arg(long, default_value_t = 1280)]
    video_width: usize,

    /// Video export height in pixels
    #[arg(long, default_value_t = 720)]
    video_height: usize,

    /// Directory to export PLY point cloud frames (optional)
    #[arg(long)]
    export_ply: Option<PathBuf>,

    /// Directory to export reconstructed Wavefront .OBJ surface mesh sequence (optional)
    #[arg(long)]
    export_obj: Option<PathBuf>,

    /// Force CPU multi-threaded solver instead of GPU acceleration
    #[arg(long)]
    cpu: bool,

    /// Explicitly enable GPU acceleration (default: auto-detected Vulkan)
    #[arg(long)]
    gpu: bool,

    /// Enable particle frustum and viewport culling during rendering (default: enabled)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    culling: bool,

    /// Disable frustum and viewport particle culling (renders all particles)
    #[arg(long, alias = "no-cull")]
    no_culling: bool,

    /// Print performance benchmark statistics
    #[arg(long)]
    benchmark: bool,

    /// Check GitHub Releases for a newer LiquidRust version and exit
    #[arg(long)]
    check_update: bool,

    /// Download and install the latest LiquidRust release from GitHub in-place
    #[arg(long, alias = "self-update")]
    update: bool,

    /// Force reinstalling the latest GitHub release even if version matches
    #[arg(long)]
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
        Some(s) => match s.to_lowercase().as_str() {
            "dam-break" | "dambreak" | "dam" => ScenePreset::DamBreak,
            "double-dam" | "doubledam" | "double" | "double-dam-break" | "doubledambreak" => {
                ScenePreset::DoubleDamBreak
            }
            "splash" | "droplet" | "dropletsplash" => ScenePreset::DropletSplash,
            "obstacle" | "drop" | "obstacledrop" => ScenePreset::ObstacleDrop,
            "tornado" | "vortex" => ScenePreset::VortexTornado,
            "zero-g" | "zerog" | "blobs" => ScenePreset::ZeroGravityBlobs,
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
    match name.to_lowercase().as_str() {
        "draft" | "low" | "fast" => SimulationQuality::Draft,
        "medium" | "med" | "normal" => SimulationQuality::Medium,
        "high" | "hi" | "smooth" => SimulationQuality::High,
        "ultra" | "realistic" => SimulationQuality::Ultra,
        "cinematic" | "vfx" => SimulationQuality::Cinematic,
        "extreme" | "1m" | "insane" => SimulationQuality::Extreme,
        _ => {
            eprintln!("Unknown quality '{}', defaulting to 'medium'", name);
            SimulationQuality::Medium
        }
    }
}

fn parse_color_mode(s: &str) -> ColorMode {
    match s.trim().to_ascii_lowercase().as_str() {
        "pressure" => ColorMode::Pressure,
        "density" => ColorMode::Density,
        "height" | "y" => ColorMode::Height,
        "energy" | "kinetic" => ColorMode::KineticEnergy,
        "single" | "solid" | "flat" => ColorMode::SingleColor,
        _ => ColorMode::Velocity,
    }
}

fn parse_color_map(s: &str) -> ColorMapType {
    match s.trim().to_ascii_lowercase().as_str() {
        "turbo" | "rainbow" => ColorMapType::Turbo,
        "viridis" => ColorMapType::Viridis,
        "fire" | "magma" | "lava" => ColorMapType::Fire,
        _ => ColorMapType::Ocean,
    }
}

fn build_mesh_config(args: &Args) -> SurfaceMeshConfig {
    let mut cfg = SurfaceMeshConfig::default();
    let preset = FluidMaterialPreset::from_slug(&args.mesh_material);
    preset.apply(&mut cfg);

    if let Some(ref c) = args.mesh_color {
        if let Some(rgb) = SurfaceMeshConfig::parse_rgb(c) {
            cfg.shallow_color = rgb;
            if args.mesh_material.eq_ignore_ascii_case("custom") {
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

    cfg.texture_mode = MeshTextureMode::from_slug(&args.mesh_texture);
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
        cfg.voxel_scale = v.clamp(0.35, 2.0);
    }
    if let Some(v) = args.mesh_iso {
        cfg.iso_value = v.clamp(0.10, 0.90);
    }
    if let Some(v) = args.mesh_smooth {
        cfg.smoothing_passes = v.min(12);
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

fn build_camera(args: &Args) -> Camera {
    let mut cam = Camera::default();
    if args.mode_2d {
        cam.azimuth = 0.0;
        cam.elevation = 0.0;
        cam.target = Vec3::new(0.0, 0.6, 0.0);
        cam.distance = 2.4;
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
    cam
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

    match kind.as_str() {
        "sphere" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid sphere spec '{}'. Format: sphere:x,y,z:radius",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid radius: {}", e))?;
            let mut col = Collider::sphere(pos, radius);
            col.name = name;
            col.id = (index + 1) as u32;
            Ok(col)
        }
        "box" | "cube" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid box spec '{}'. Format: box:x,y,z:w,h,d",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let size = parse_vec3(&parts[2])?;
            let half = size * 0.5;
            let mut col = Collider::box_obstacle(pos - half, pos + half);
            col.name = name;
            col.id = (index + 1) as u32;
            Ok(col)
        }
        "cylinder" => {
            if parts.len() < 4 {
                return Err(format!(
                    "Invalid cylinder spec '{}'. Format: cylinder:x,y,z:radius:height",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid radius: {}", e))?;
            let height = parts[3]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid height: {}", e))?;
            let mut col = Collider::cylinder(pos, radius, height);
            col.name = name;
            col.id = (index + 1) as u32;
            Ok(col)
        }
        "obj" | "mesh" => {
            if parts.len() < 4 {
                return Err(format!(
                    "Invalid obj spec '{}'. Format: obj:filepath:x,y,z:scale",
                    spec
                ));
            }
            let path = parts[1].trim();
            let pos = parse_vec3(&parts[2])?;
            let scale = parts[3]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid scale: {}", e))?;
            let mut col = Collider::load_obj(path, pos, scale)
                .map_err(|e| format!("failed to load OBJ '{}': {}", path, e))?;
            col.name = format!("OBJ #{} ({})", index + 1, path);
            col.id = (index + 1) as u32;
            Ok(col)
        }
        _ => Err(format!(
            "Unknown obstacle type '{}'. Expected 'sphere', 'box', 'cylinder', or 'obj'",
            kind
        )),
    }
}

fn parse_faucet(spec: &str, index: usize) -> Result<Faucet, String> {
    let parts = clean_spec(spec);
    if parts.is_empty() {
        return Err("empty faucet specification".to_string());
    }
    let kind = parts[0].to_lowercase();
    let name = format!("Faucet #{} ({})", index + 1, kind);

    match kind.as_str() {
        "waterfall" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid waterfall spec '{}'. Format: waterfall:x,y,z:width[:flow_rate]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let width = parts[2]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid width: {}", e))?;
            let flow_rate = if parts.len() > 3 {
                parts[3]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                600.0
            };
            let mut f = Faucet::new_waterfall(name, pos, width, 0.08, flow_rate);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "shower" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid shower spec '{}'. Format: shower:x,y,z:radius[:flow_rate[:spread]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid radius: {}", e))?;
            let flow_rate = if parts.len() > 3 {
                parts[3]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                500.0
            };
            let spread = if parts.len() > 4 {
                parts[4]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid spread: {}", e))?
            } else {
                0.22
            };
            let mut f = Faucet::new_shower(name, pos, radius, flow_rate, spread);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "circle" | "pipe" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid circle spec '{}'. Format: circle:x,y,z:radius[:flow_rate]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid radius: {}", e))?;
            let flow_rate = if parts.len() > 3 {
                parts[3]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                500.0
            };
            let mut f =
                Faucet::new_circle(name, pos, glam::Vec3::new(0.0, -2.5, 0.0), radius, flow_rate);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "tap" | "point" => {
            if parts.len() < 2 {
                return Err(format!(
                    "Invalid tap spec '{}'. Format: tap:x,y,z[:flow_rate]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let flow_rate = if parts.len() > 2 {
                parts[2]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                400.0
            };
            let mut f = Faucet::new_point(name, pos, glam::Vec3::new(0.0, -2.0, 0.0), flow_rate);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "stream" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Invalid stream spec '{}'. Format: stream:x,y,z:vx,vy,vz[:flow_rate[:spread]]",
                    spec
                ));
            }
            let pos = parse_vec3(&parts[1])?;
            let vel = parse_vec3(&parts[2])?;
            let flow_rate = if parts.len() > 3 {
                parts[3]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                500.0
            };
            let spread = if parts.len() > 4 {
                parts[4]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid spread: {}", e))?
            } else {
                0.05
            };
            let mut f = Faucet::new_circle(name, pos, vel, 0.04, flow_rate);
            f.spray_spread = spread;
            f.id = (index + 1) as u32;
            Ok(f)
        }
        _ => Err(format!(
            "Unknown faucet shape '{}'. Expected 'waterfall', 'shower', 'circle', 'tap', or 'stream'",
            kind
        )),
    }
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
                let r = parts[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[2]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[2]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[2]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[2]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[2]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[1]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
                let r = parts[2]
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| format!("invalid radius: {}", e))?;
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
            let r = parts[2]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid radius: {}", e))?;
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

fn parse_chamber(args: &Args) -> Chamber {
    let chamber_shape = match args.chamber_shape.to_lowercase().as_str() {
        "sphere" | "globe" | "ball" => ChamberShape::Sphere,
        "bathtub" | "tub" | "bath" => ChamberShape::Bathtub,
        _ => ChamberShape::Box,
    };
    let mut chamber = Chamber::new(chamber_shape);

    if let Some(ref scale_str) = args.chamber_scale {
        let cleaned = scale_str.replace(['[', ']', '(', ')'], "");
        let parts: Vec<&str> = cleaned.split(',').collect();
        if parts.len() == 1 {
            if let Ok(s) = parts[0].trim().parse::<f32>() {
                chamber.scale = Vec3::splat(s);
            }
        } else if parts.len() == 3 {
            if let (Ok(x), Ok(y), Ok(z)) = (
                parts[0].trim().parse::<f32>(),
                parts[1].trim().parse::<f32>(),
                parts[2].trim().parse::<f32>(),
            ) {
                chamber.scale = Vec3::new(x, y, z);
            }
        }
    }

    if let Some(ref size_str) = args.chamber_size {
        let cleaned = size_str.replace(['[', ']', '(', ')'], "");
        let parts: Vec<&str> = cleaned.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(w), Ok(h), Ok(d)) = (
                parts[0].trim().parse::<f32>(),
                parts[1].trim().parse::<f32>(),
                parts[2].trim().parse::<f32>(),
            ) {
                chamber.base_size = Vec3::new(w, h, d);
                chamber.scale = Vec3::ONE;
            }
        }
    }

    chamber
}

fn run_headless(
    args: Args,
    parsed_colliders: Vec<Collider>,
    parsed_faucets: Vec<Faucet>,
    parsed_drains: Vec<Drain>,
) {
    let preset = parse_preset(args.preset.as_deref(), !parsed_faucets.is_empty());
    let quality = parse_quality(&args.quality);
    let fps = args.fps.max(1);
    let render_mode = if args.surface_mesh {
        RenderMode::SurfaceMesh
    } else {
        RenderMode::from_slug(&args.render_mode)
    };
    let mesh_cfg = build_mesh_config(&args);
    let camera = build_camera(&args);
    let color_mode = parse_color_mode(&args.color_mode);
    let color_map = parse_color_map(&args.color_map);

    let (total_frames, dt) = if let Some(t) = args.time {
        let f = (t * fps as f32).round().max(1.0) as usize;
        let delta_t = (1.0 / fps as f32) * args.speed;
        (f, delta_t)
    } else {
        let base_dt = args.dt.unwrap_or(1.0 / fps as f32);
        let delta_t = base_dt * args.speed;
        (args.frames, delta_t)
    };

    println!("=== liquid-rust studio render / headless mode ===");
    println!("Preset:      {:?}", preset);
    println!("Quality:     {} ({:?})", quality.name_for(args.mode_2d), quality);
    println!("Render Mode: {}", render_mode.name());
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
    if (args.speed - 1.0).abs() > 1e-4 {
        println!("Sim Speed:   {:.2}x", args.speed);
    }
    println!(
        "Timeline:    {} frames ({:.2}s @ {} fps, dt={:.5}s)",
        total_frames,
        total_frames as f32 / fps as f32,
        fps,
        dt
    );

    let chamber = parse_chamber(&args);
    let bounds = chamber.compute_bounds();
    let mut config = SolverConfig::default();
    quality.apply(&mut config, args.mode_2d);

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
    if let Some(vort) = args.vorticity {
        config.vorticity = vort.clamp(0.0, 0.5);
    }
    if let Some(st) = args.surface_tension {
        config.surface_tension = st.clamp(0.0, 0.5);
    }
    if let Some(visc) = args.viscosity {
        config.viscosity = visc.clamp(0.0, 1.0);
    }
    config.rest_density = compute_lattice_rest_density(config.smoothing_length, config.spacing);

    let mut solver = SphSolver::new(bounds, config);
    if args.mode_2d {
        solver.set_2d(true);
    }
    solver.set_chamber(chamber);
    SceneGenerator::setup_scene(&mut solver, preset);

    for col in parsed_colliders {
        println!("  + Obstacle: {} (id: {})", col.name, col.id);
        solver.colliders.push(col);
    }
    for faucet in parsed_faucets {
        println!(
            "  + Faucet: {} (id: {}, rate: {:.0} p/s)",
            faucet.name, faucet.id, faucet.flow_rate
        );
        solver.faucets.push(faucet);
    }
    for drain in parsed_drains {
        println!(
            "  + Drain: {} (id: {}, face: {}, radius: {:.2})",
            drain.name,
            drain.id,
            drain.face.name(),
            drain.radius
        );
        solver.drains.push(drain);
    }

    let particle_count = solver.particle_count();
    println!(
        "Spawned {} particles (radius: {:.4}m, spacing: {:.4}m)",
        particle_count, solver.config.particle_radius, solver.config.spacing
    );

    let mut gpu_solver = if !args.cpu {
        match GpuSphSolver::new(solver.bounds, &solver.config, &solver.particles) {
            Ok(g) => {
                println!("⚡ GPU Acceleration: ACTIVE [{}]", g.device_name);
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
    } else {
        println!("Using multi-threaded Rayon CPU solver (--cpu specified)");
        None
    };

    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = Arc::clone(&interrupted);
    let _ = ctrlc::set_handler(move || {
        interrupted_clone.store(true, Ordering::SeqCst);
    });

    let mut mp4_encoder = if let Some(ref path) = args.export_mp4 {
        println!(
            "Encoding MP4 video ({}x{} @ {} fps, {}) to: {}",
            args.video_width,
            args.video_height,
            fps,
            render_mode.slug(),
            path.display()
        );
        match Mp4Encoder::new(path, args.video_width, args.video_height, fps) {
            Ok(enc) => Some(enc),
            Err(e) => {
                eprintln!("Error starting MP4 encoder: {}", e);
                None
            }
        }
    } else {
        None
    };

    let use_culling = args.culling && !args.no_culling;
    let mut software_renderer = if mp4_encoder.is_some() {
        let mut rend = SoftwareRenderer::new(args.video_width, args.video_height);
        rend.enable_culling = use_culling;
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
                gpu.sync_particles(solver.bounds, &solver.config, &solver.particles, false);
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
                args.mode_2d,
            );
            gpu.read_particles(&mut solver.particles);

            let has_mesh_col = solver
                .colliders
                .iter()
                .any(|c| c.enabled && matches!(c.shape, ColliderShape::Mesh(_)));
            if has_mesh_col {
                let radius = solver.config.particle_radius;
                let mut hit = false;
                for col in &solver.colliders {
                    if let ColliderShape::Mesh(_) = &col.shape {
                        for p in &mut solver.particles {
                            hit |= col.resolve(&mut p.position, &mut p.velocity, radius);
                        }
                    }
                }
                if hit {
                    gpu.sync_particles(solver.bounds, &solver.config, &solver.particles, true);
                }
            }

            let drained = solver.drain_particles();
            if drained > 0 {
                gpu.sync_particles(solver.bounds, &solver.config, &solver.particles, true);
            }
        } else {
            solver.step(dt);
        }
        anim_time += dt;
        total_sim_time += t0.elapsed().as_secs_f64();

        // Build surface mesh if needed for MP4 surface-mesh rendering or .OBJ export
        let t_r = Instant::now();
        let need_mesh =
            (render_mode == RenderMode::SurfaceMesh && mp4_encoder.is_some()) || args.export_obj.is_some();
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
                    rend.render_frame(
                        &solver.particles,
                        solver.config.particle_radius * 1.25,
                        &solver.chamber,
                        solver.bounds,
                        &camera,
                        color_mode,
                        color_map,
                        solver.config.rest_density,
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
                            args.mode_2d,
                        );
                    }
                }
            }
            if let Err(e) = enc.write_frame(&rend.rgb_buffer) {
                eprintln!("Error writing video frame {}: {}", frame, e);
            }
        }

        if let (Some(ref dir), Some(ref mesh)) = (&args.export_obj, &built_mesh) {
            let obj_path = dir.join(format!("fluid_surface_{:05}.obj", frame));
            if let Err(e) = mesh.save_obj(&obj_path) {
                eprintln!("Error exporting OBJ frame {}: {}", frame, e);
            }
        }
        total_mesh_render_time += t_r.elapsed().as_secs_f64();

        if let Some(ref dir) = args.export_ply {
            let path = dir.join(format!("fluid_{:05}.ply", frame));
            let colors: Vec<[u8; 3]> = solver
                .particles
                .iter()
                .map(|p| {
                    let speed_ratio = (p.speed() / 2.4).clamp(0.0, 1.0);
                    [
                        (speed_ratio * 255.0) as u8,
                        120,
                        ((1.0 - speed_ratio * 0.7) * 255.0) as u8,
                    ]
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
    println!();

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
    println!(
        "Avg physics step:       {:.2} ms/frame",
        (total_sim_time / effective_frames as f64) * 1000.0
    );
    if total_mesh_render_time > 0.001 {
        println!(
            "Avg mesh + rasterize:   {:.2} ms/frame",
            (total_mesh_render_time / effective_frames as f64) * 1000.0
        );
    }
    println!(
        "Throughput:             {:.0} particle-steps/sec",
        (solver.particle_count() * completed_frames) as f64 / total_sim_time.max(1e-5)
    );
}

#[allow(clippy::too_many_arguments)]
fn run_gui(
    args: Args,
    initial_preset: ScenePreset,
    initial_quality: SimulationQuality,
    initial_chamber: Chamber,
    parsed_colliders: Vec<Collider>,
    parsed_faucets: Vec<Faucet>,
    parsed_drains: Vec<Drain>,
) -> Result<(), eframe::Error> {
    if args.mode_2d {
        eprintln!("Note: --2d is for headless CLI rendering only; switch between 3D and 2D directly using the toolbar button.");
    }
    let window_title = "LiquidRust Studio — 2D/3D SPH & Surface Meshing";

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 840.0])
            .with_min_inner_size([920.0, 560.0])
            .with_title(window_title),
        renderer: eframe::Renderer::Glow,
        depth_buffer: 24,
        ..Default::default()
    };

    let render_mode = if args.surface_mesh {
        RenderMode::SurfaceMesh
    } else {
        RenderMode::from_slug(&args.render_mode)
    };
    let mesh_cfg = build_mesh_config(&args);
    let cam_override = build_camera(&args);
    let has_cam_override = args.camera_azimuth.is_some()
        || args.camera_elevation.is_some()
        || args.camera_distance.is_some()
        || args.camera_target.is_some();

    eframe::run_native(
        "liquid-rust",
        native_options,
        Box::new(move |cc| {
            let mut app = LiquidApp::new(cc);
            if (args.speed - 1.0).abs() > 1e-4 {
                app.sim_speed = args.speed;
            }
            if args.cpu {
                app.use_gpu = false;
            }
            app.solver.set_chamber(initial_chamber);
            app.load_preset(initial_preset);
            if initial_quality != SimulationQuality::Medium {
                app.set_quality(initial_quality);
            }
            if let Some(vort) = args.vorticity {
                app.solver.config.vorticity = vort.clamp(0.0, 0.5);
            }
            if let Some(st) = args.surface_tension {
                app.solver.config.surface_tension = st.clamp(0.0, 0.5);
            }
            if let Some(visc) = args.viscosity {
                app.solver.config.viscosity = visc.clamp(0.0, 1.0);
            }

            app.render_mode = render_mode;
            app.mesh_config = mesh_cfg;
            app.sync_sun_angles_from_dir();
            app.color_mode = parse_color_mode(&args.color_mode);
            app.color_map = parse_color_map(&args.color_map);
            app.video_width = args.video_width;
            app.video_height = args.video_height;
            if has_cam_override {
                app.camera = cam_override;
            }

            for c in parsed_colliders {
                app.next_collider_id = app.next_collider_id.max(c.id + 1);
                app.solver.colliders.push(c);
            }
            for f in parsed_faucets {
                app.next_faucet_id = app.next_faucet_id.max(f.id + 1);
                app.solver.faucets.push(f);
            }
            for d in parsed_drains {
                app.next_drain_id = app.next_drain_id.max(d.id + 1);
                app.solver.drains.push(d);
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

    let preset = parse_preset(args.preset.as_deref(), !parsed_faucets.is_empty());
    let chamber = parse_chamber(&args);

    if args.headless
        || args.benchmark
        || args.export_mp4.is_some()
        || args.export_ply.is_some()
        || args.export_obj.is_some()
    {
        run_headless(args, parsed_colliders, parsed_faucets, parsed_drains);
    } else {
        let initial_quality = parse_quality(&args.quality);
        if let Err(e) = run_gui(
            args,
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
