use clap::Parser;
use eframe::egui;
use glam::Vec3;
use liquid_rust::export::{Mp4Encoder, PlyExporter, SoftwareRenderer};
use liquid_rust::gpu::GpuSphSolver;
use liquid_rust::render::{Camera, ColorMapType, ColorMode};
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
    version = "0.1.8",
    about = "High-performance 3D SPH Fluid Simulator in Rust with Real-time 3D Viewport, MP4 Video & PLY Export"
)]
struct Args {
    /// Run in headless mode without opening GUI window
    #[arg(long)]
    headless: bool,

    /// Scene preset to simulate: dam-break, double-dam, splash, obstacle, tornado, zero-g, empty. If omitted, no preset is added.
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

    /// Simulation speed multiplier (e.g. 0.5 for 2x slow-mo, 0.2 for 5x slow-mo ASMR, 2.0 for fast)
    #[arg(long, alias = "sim-speed", default_value_t = 1.0)]
    speed: f32,

    /// Custom time step dt per frame in seconds (defaults to 1.0 / fps * speed)
    #[arg(long)]
    dt: Option<f32>,

    /// Add custom obstacle: 'sphere:x,y,z:r', 'box:x,y,z:w,h,d', 'cylinder:x,y,z:r,h', or 'obj:path:x,y,z:scale'
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

fn parse_preset(name: Option<&str>) -> ScenePreset {
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
                eprintln!("Unknown preset '{}', defaulting to empty", s);
                ScenePreset::Empty
            }
        },
        None => ScenePreset::Empty,
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

fn clean_spec(spec: &str) -> Vec<String> {
    let s = spec.replace("[:", ":").replace('[', "").replace(']', "");
    s.split(':')
        .map(|part| part.trim().trim_matches(|c| c == '(' || c == ')' || c == '{' || c == '}').trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

fn parse_vec3(s: &str) -> Result<glam::Vec3, String> {
    let clean = s.trim().trim_matches(|c| c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}');
    let parts: Vec<&str> = clean.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("expected 3 comma-separated floats 'x,y,z', got '{}'", s));
    }
    let x = parts[0].trim().parse::<f32>().map_err(|e| format!("invalid float '{}': {}", parts[0], e))?;
    let y = parts[1].trim().parse::<f32>().map_err(|e| format!("invalid float '{}': {}", parts[1], e))?;
    let z = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid float '{}': {}", parts[2], e))?;
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
            // sphere:x,y,z:radius
            if parts.len() < 3 {
                return Err(format!("Invalid sphere spec '{}'. Format: sphere:x,y,z:radius", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
            let mut col = Collider::sphere(pos, radius);
            col.name = name;
            col.id = (index + 1) as u32;
            Ok(col)
        }
        "box" | "cube" => {
            // box:x,y,z:w,h,d
            if parts.len() < 3 {
                return Err(format!("Invalid box spec '{}'. Format: box:x,y,z:w,h,d", spec));
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
            // cylinder:x,y,z:radius:height
            if parts.len() < 4 {
                return Err(format!("Invalid cylinder spec '{}'. Format: cylinder:x,y,z:radius:height", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
            let height = parts[3].trim().parse::<f32>().map_err(|e| format!("invalid height: {}", e))?;
            let mut col = Collider::cylinder(pos, radius, height);
            col.name = name;
            col.id = (index + 1) as u32;
            Ok(col)
        }
        "obj" | "mesh" => {
            // obj:path:x,y,z:scale
            if parts.len() < 4 {
                return Err(format!("Invalid obj spec '{}'. Format: obj:filepath:x,y,z:scale", spec));
            }
            let path = parts[1].trim();
            let pos = parse_vec3(&parts[2])?;
            let scale = parts[3].trim().parse::<f32>().map_err(|e| format!("invalid scale: {}", e))?;
            let mut col = Collider::load_obj(path, pos, scale).map_err(|e| format!("failed to load OBJ '{}': {}", path, e))?;
            col.name = format!("OBJ #{} ({})", index + 1, path);
            col.id = (index + 1) as u32;
            Ok(col)
        }
        _ => Err(format!("Unknown obstacle type '{}'. Expected 'sphere', 'box', 'cylinder', or 'obj'", kind)),
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
            // waterfall:x,y,z:width[:flow_rate]
            if parts.len() < 3 {
                return Err(format!("Invalid waterfall spec '{}'. Format: waterfall:x,y,z:width[:flow_rate]", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let width = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid width: {}", e))?;
            let flow_rate = if parts.len() > 3 {
                parts[3].trim().parse::<f32>().map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                600.0
            };
            let mut f = Faucet::new_waterfall(name, pos, width, 0.08, flow_rate);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "shower" => {
            // shower:x,y,z:radius[:flow_rate[:spread]]
            if parts.len() < 3 {
                return Err(format!("Invalid shower spec '{}'. Format: shower:x,y,z:radius[:flow_rate[:spread]]", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
            let flow_rate = if parts.len() > 3 {
                parts[3].trim().parse::<f32>().map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                500.0
            };
            let spread = if parts.len() > 4 {
                parts[4].trim().parse::<f32>().map_err(|e| format!("invalid spread: {}", e))?
            } else {
                0.22
            };
            let mut f = Faucet::new_shower(name, pos, radius, flow_rate, spread);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "circle" | "pipe" => {
            // circle:x,y,z:radius[:flow_rate]
            if parts.len() < 3 {
                return Err(format!("Invalid circle spec '{}'. Format: circle:x,y,z:radius[:flow_rate]", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let radius = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
            let flow_rate = if parts.len() > 3 {
                parts[3].trim().parse::<f32>().map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                500.0
            };
            let mut f = Faucet::new_circle(name, pos, glam::Vec3::new(0.0, -2.5, 0.0), radius, flow_rate);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "tap" | "point" => {
            // tap:x,y,z[:flow_rate]
            if parts.len() < 2 {
                return Err(format!("Invalid tap spec '{}'. Format: tap:x,y,z[:flow_rate]", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let flow_rate = if parts.len() > 2 {
                parts[2].trim().parse::<f32>().map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                400.0
            };
            let mut f = Faucet::new_point(name, pos, glam::Vec3::new(0.0, -2.0, 0.0), flow_rate);
            f.id = (index + 1) as u32;
            Ok(f)
        }
        "stream" => {
            // stream:x,y,z:vx,vy,vz[:flow_rate[:spread]]
            if parts.len() < 3 {
                return Err(format!("Invalid stream spec '{}'. Format: stream:x,y,z:vx,vy,vz[:flow_rate[:spread]]", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let vel = parse_vec3(&parts[2])?;
            let flow_rate = if parts.len() > 3 {
                parts[3].trim().parse::<f32>().map_err(|e| format!("invalid flow_rate: {}", e))?
            } else {
                500.0
            };
            let spread = if parts.len() > 4 {
                parts[4].trim().parse::<f32>().map_err(|e| format!("invalid spread: {}", e))?
            } else {
                0.05
            };
            let mut f = Faucet::new_circle(name, pos, vel, 0.04, flow_rate);
            f.spray_spread = spread;
            f.id = (index + 1) as u32;
            Ok(f)
        }
        _ => Err(format!("Unknown faucet shape '{}'. Expected 'waterfall', 'shower', 'circle', 'tap', or 'stream'", kind)),
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
                let r = parts[1].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Floor, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xz: Vec<&str> = parts[1].split(',').collect();
                if xz.len() != 2 {
                    return Err(format!("Expected 2 coords (x,z) for floor drain offset, got '{}'", parts[1]));
                }
                let x = xz[0].trim().parse::<f32>().map_err(|e| format!("invalid x: {}", e))?;
                let z = xz[1].trim().parse::<f32>().map_err(|e| format!("invalid z: {}", e))?;
                let r = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Floor, glam::Vec3::new(x, 0.0, z), r)
            } else {
                return Err(format!("Invalid floor drain spec '{}'. Format: floor:radius or floor:x,z:radius", spec));
            }
        }
        "ceiling" | "top" => {
            if parts.len() == 2 {
                let r = parts[1].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Ceiling, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xz: Vec<&str> = parts[1].split(',').collect();
                if xz.len() != 2 {
                    return Err(format!("Expected 2 coords (x,z) for ceiling drain offset, got '{}'", parts[1]));
                }
                let x = xz[0].trim().parse::<f32>().map_err(|e| format!("invalid x: {}", e))?;
                let z = xz[1].trim().parse::<f32>().map_err(|e| format!("invalid z: {}", e))?;
                let r = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Ceiling, glam::Vec3::new(x, 0.0, z), r)
            } else {
                return Err(format!("Invalid ceiling drain spec '{}'. Format: ceiling:radius or ceiling:x,z:radius", spec));
            }
        }
        "left" => {
            if parts.len() == 2 {
                let r = parts[1].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Left, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let yz: Vec<&str> = parts[1].split(',').collect();
                if yz.len() != 2 {
                    return Err(format!("Expected 2 coords (y,z) for left wall drain offset, got '{}'", parts[1]));
                }
                let y = yz[0].trim().parse::<f32>().map_err(|e| format!("invalid y: {}", e))?;
                let z = yz[1].trim().parse::<f32>().map_err(|e| format!("invalid z: {}", e))?;
                let r = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Left, glam::Vec3::new(0.0, y, z), r)
            } else {
                return Err(format!("Invalid left drain spec '{}'. Format: left:radius or left:y,z:radius", spec));
            }
        }
        "right" => {
            if parts.len() == 2 {
                let r = parts[1].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Right, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let yz: Vec<&str> = parts[1].split(',').collect();
                if yz.len() != 2 {
                    return Err(format!("Expected 2 coords (y,z) for right wall drain offset, got '{}'", parts[1]));
                }
                let y = yz[0].trim().parse::<f32>().map_err(|e| format!("invalid y: {}", e))?;
                let z = yz[1].trim().parse::<f32>().map_err(|e| format!("invalid z: {}", e))?;
                let r = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Right, glam::Vec3::new(0.0, y, z), r)
            } else {
                return Err(format!("Invalid right drain spec '{}'. Format: right:radius or right:y,z:radius", spec));
            }
        }
        "back" => {
            if parts.len() == 2 {
                let r = parts[1].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Back, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xy: Vec<&str> = parts[1].split(',').collect();
                if xy.len() != 2 {
                    return Err(format!("Expected 2 coords (x,y) for back wall drain offset, got '{}'", parts[1]));
                }
                let x = xy[0].trim().parse::<f32>().map_err(|e| format!("invalid x: {}", e))?;
                let y = xy[1].trim().parse::<f32>().map_err(|e| format!("invalid y: {}", e))?;
                let r = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Back, glam::Vec3::new(x, y, 0.0), r)
            } else {
                return Err(format!("Invalid back drain spec '{}'. Format: back:radius or back:x,y:radius", spec));
            }
        }
        "front" => {
            if parts.len() == 2 {
                let r = parts[1].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Front, glam::Vec3::ZERO, r)
            } else if parts.len() >= 3 {
                let xy: Vec<&str> = parts[1].split(',').collect();
                if xy.len() != 2 {
                    return Err(format!("Expected 2 coords (x,y) for front wall drain offset, got '{}'", parts[1]));
                }
                let x = xy[0].trim().parse::<f32>().map_err(|e| format!("invalid x: {}", e))?;
                let y = xy[1].trim().parse::<f32>().map_err(|e| format!("invalid y: {}", e))?;
                let r = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
                (CubeFace::Front, glam::Vec3::new(x, y, 0.0), r)
            } else {
                return Err(format!("Invalid front drain spec '{}'. Format: front:radius or front:x,y:radius", spec));
            }
        }
        "custom" => {
            // custom:x,y,z:radius
            if parts.len() < 3 {
                return Err(format!("Invalid custom drain spec '{}'. Format: custom:x,y,z:radius", spec));
            }
            let pos = parse_vec3(&parts[1])?;
            let r = parts[2].trim().parse::<f32>().map_err(|e| format!("invalid radius: {}", e))?;
            (CubeFace::Custom, pos, r)
        }
        _ => return Err(format!("Unknown drain face '{}'. Expected 'floor', 'ceiling', 'left', 'right', 'back', 'front', or 'custom'", kind)),
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
    let preset = parse_preset(args.preset.as_deref());
    let quality = parse_quality(&args.quality);
    let fps = args.fps.max(1);

    let (total_frames, dt) = if let Some(t) = args.time {
        let f = (t * fps as f32).round().max(1.0) as usize;
        let delta_t = (1.0 / fps as f32) * args.speed;
        (f, delta_t)
    } else {
        let base_dt = args.dt.unwrap_or(1.0 / fps as f32);
        let delta_t = base_dt * args.speed;
        (args.frames, delta_t)
    };

    println!("=== liquid-rust headless mode ===");
    println!("Preset: {:?}", preset);
    println!("Quality: {} ({:?})", quality.name(), quality);
    if args.speed != 1.0 {
        println!("Speed multiplier: {:.2}x", args.speed);
    }
    println!("Total Frames: {} ({:.2}s @ {} fps)", total_frames, total_frames as f32 / fps as f32, fps);
    println!("Time step dt: {:.5}s", dt);

    let chamber = parse_chamber(&args);
    let bounds = chamber.compute_bounds();
    let mut config = SolverConfig::default();
    quality.apply(&mut config);

    // Apply target particle count if requested
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

    // Apply custom overrides if given
    if let Some(r) = args.particle_radius {
        config.particle_radius = r;
        config.smoothing_length = r * 3.1;
    }
    if let Some(s) = args.spacing {
        config.spacing = s;
    }
    config.rest_density = compute_lattice_rest_density(config.smoothing_length, config.spacing);

    let mut solver = SphSolver::new(bounds, config);
    solver.set_chamber(chamber);
    SceneGenerator::setup_scene(&mut solver, preset);

    let ch_size = solver.bounds.size();
    println!("Chamber: {} (Scale: ({:.2}, {:.2}, {:.2}), Bounds: {:.2} x {:.2} x {:.2} m)",
        solver.chamber.shape.name(),
        solver.chamber.scale.x, solver.chamber.scale.y, solver.chamber.scale.z,
        ch_size.x, ch_size.y, ch_size.z);

    for col in parsed_colliders {
        println!("  + Obstacle: {} (id: {})", col.name, col.id);
        solver.colliders.push(col);
    }
    for faucet in parsed_faucets {
        println!("  + Faucet: {} (id: {}, rate: {:.0} p/s, pos: ({:.2}, {:.2}, {:.2}))",
            faucet.name, faucet.id, faucet.flow_rate, faucet.position.x, faucet.position.y, faucet.position.z);
        if faucet.position.x < solver.bounds.min.x - 0.2 || faucet.position.x > solver.bounds.max.x + 0.2
            || faucet.position.y < solver.bounds.min.y - 0.2 || faucet.position.y > solver.bounds.max.y + 0.8
            || faucet.position.z < solver.bounds.min.z - 0.2 || faucet.position.z > solver.bounds.max.z + 0.2
        {
            eprintln!("  ⚠️ Note: Faucet position ({:.1}, {:.1}, {:.1}) is outside the fluid basin boundaries ({:.2}..{:.2}, {:.2}..{:.2}, {:.2}..{:.2}). Particles will fall outside the container!",
                faucet.position.x, faucet.position.y, faucet.position.z,
                solver.bounds.min.x, solver.bounds.max.x,
                solver.bounds.min.y, solver.bounds.max.y,
                solver.bounds.min.z, solver.bounds.max.z);
        }
        solver.faucets.push(faucet);
    }
    for drain in parsed_drains {
        println!("  + Drain: {} (id: {}, face: {}, radius: {:.2})", drain.name, drain.id, drain.face.name(), drain.radius);
        solver.drains.push(drain);
    }

    let particle_count = solver.particle_count();
    println!("Spawned {} particles (particle radius: {:.4}, spacing: {:.4})",
        particle_count, solver.config.particle_radius, solver.config.spacing);

    // Initialize GPU acceleration (Vulkan) unless --cpu is explicitly requested
    let mut gpu_solver = if !args.cpu {
        match GpuSphSolver::new(bounds, &solver.config, &solver.particles) {
            Ok(g) => {
                println!("⚡ GPU Acceleration: ACTIVE [{}]", g.device_name);
                Some(g)
            }
            Err(e) => {
                println!("GPU compute unavailable ({}), using multi-threaded CPU solver", e);
                None
            }
        }
    } else {
        println!("Using CPU multi-threaded solver (--cpu flag specified)");
        None
    };

    // Setup graceful Ctrl+C interruption handler so video is preserved at current progress
    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = Arc::clone(&interrupted);
    let _ = ctrlc::set_handler(move || {
        interrupted_clone.store(true, Ordering::SeqCst);
    });

    // Setup MP4 Video Encoder if requested
    let mut mp4_encoder = if let Some(ref path) = args.export_mp4 {
        println!("Encoding MP4 video ({}x{} @ {} fps) to: {}",
            args.video_width, args.video_height, fps, path.display());
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
        println!("Particle Culling: {}", if use_culling { "ENABLED (Frustum & Viewport)" } else { "DISABLED (Draw all particles)" });
        let mut rend = SoftwareRenderer::new(args.video_width, args.video_height);
        rend.enable_culling = use_culling;
        Some(rend)
    } else {
        None
    };

    let camera = Camera::default();

    // Setup PLY directory if requested
    if let Some(ref dir) = args.export_ply {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Failed to create export directory {}: {}", dir.display(), e);
            return;
        }
        println!("Exporting PLY sequence to: {}", dir.display());
    }

    let start_total = Instant::now();
    let mut total_sim_time = 0.0f64;
    let mut last_progress_print = Instant::now();
    let mut completed_frames = 0;

    for frame in 0..total_frames {
        if interrupted.load(Ordering::SeqCst) {
            println!(
                "\n\n[Ctrl+C] Interruption caught! Gracefully finalizing video at current frame ({}/{})...",
                completed_frames, total_frames
            );
            break;
        }

        let t0 = Instant::now();
        if let Some(ref mut gpu) = gpu_solver {
            solver.emit_faucets(dt);
            if gpu.particle_count != solver.particles.len() || solver.particles.len() > gpu.particle_capacity {
                gpu.init_buffers(solver.bounds, &solver.config, &solver.particles);
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
            );
            gpu.read_particles(&mut solver.particles);

            // Mesh colliders (.obj) resolution on CPU
            let radius = solver.config.particle_radius;
            for col in &solver.colliders {
                if let ColliderShape::Mesh(_) = &col.shape {
                    for p in &mut solver.particles {
                        col.resolve(&mut p.position, &mut p.velocity, radius);
                    }
                }
            }

            let drained = solver.drain_particles();
            if drained > 0 {
                gpu.init_buffers(solver.bounds, &solver.config, &solver.particles);
            }
        } else {
            solver.step(dt);
        }
        let step_time = t0.elapsed().as_secs_f64();
        total_sim_time += step_time;

        // Render & stream to MP4
        if let (Some(ref mut enc), Some(ref mut rend)) = (&mut mp4_encoder, &mut software_renderer) {
            rend.render_frame(
                &solver.particles,
                solver.config.particle_radius,
                &solver.chamber,
                solver.bounds,
                &camera,
                ColorMode::Velocity,
                ColorMapType::Ocean,
                solver.config.rest_density,
            );
            if let Err(e) = enc.write_frame(&rend.rgb_buffer) {
                eprintln!("Error writing video frame {}: {}", frame, e);
            }
        }

        // Export PLY frame
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
        let completed = completed_frames;
        let elapsed = start_total.elapsed().as_secs_f64();
        let should_print = frame == 0
            || completed == total_frames
            || last_progress_print.elapsed().as_millis() >= 60;

        if should_print {
            let avg_time_per_frame = elapsed / completed as f64;
            let remaining_frames = total_frames - completed;
            let eta_secs = remaining_frames as f64 * avg_time_per_frame;

            let percent = (completed as f64 / total_frames as f64) * 100.0;
            let bar_len = 24;
            let filled =
                ((completed as f64 / total_frames as f64) * bar_len as f64).round() as usize;
            let bar = format!(
                "[{}{}]",
                "=".repeat(filled.min(bar_len)),
                " ".repeat(bar_len.saturating_sub(filled))
            );

            let fps_current = completed as f64 / elapsed.max(1e-4);

            use std::io::Write;
            print!(
                "\r{} {:5.1}% | Frame {:4}/{} | ETA: {:<7} | Elapsed: {:<6} | {:4.1} fps",
                bar,
                percent,
                completed,
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

    // Finalize MP4
    if let Some(enc) = mp4_encoder {
        println!("Finalizing MP4 video encoding (saving {} frames)...", completed_frames);
        match enc.finish() {
            Ok(_) => println!("MP4 video saved successfully at current progress!"),
            Err(e) => eprintln!("Error finalizing MP4 video: {}", e),
        }
    }

    let total_elapsed = start_total.elapsed().as_secs_f64();
    let effective_frames = completed_frames.max(1);
    println!("--------------------------------------------------");
    println!("Simulation finished!");
    println!("Total time: {:.3}s", total_elapsed);
    println!("Frames rendered: {} / {}", completed_frames, total_frames);
    println!("Average time per frame: {:.2}ms", (total_sim_time / effective_frames as f64) * 1000.0);
    println!("Effective simulation FPS: {:.1}", effective_frames as f64 / total_sim_time.max(1e-5));
    println!("Throughput: {:.0} particle-steps/sec", (particle_count * completed_frames) as f64 / total_sim_time.max(1e-5));
}

fn run_gui(
    initial_preset: ScenePreset,
    initial_quality: SimulationQuality,
    initial_speed: f32,
    initial_chamber: Chamber,
    parsed_colliders: Vec<Collider>,
    parsed_faucets: Vec<Faucet>,
    parsed_drains: Vec<Drain>,
) -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("liquid-rust - 3D SPH Fluid Simulator"),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "liquid-rust",
        native_options,
        Box::new(move |cc| {
            let mut app = LiquidApp::new(cc);
            if (initial_speed - 1.0).abs() > 1e-4 {
                app.sim_speed = initial_speed;
            }
            app.solver.set_chamber(initial_chamber);
            app.load_preset(initial_preset);
            if initial_quality != SimulationQuality::Medium {
                app.set_quality(initial_quality);
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
    let args = Args::parse();

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

    let preset = parse_preset(args.preset.as_deref());
    let chamber = parse_chamber(&args);

    if args.headless || args.benchmark || args.export_mp4.is_some() {
        run_headless(args, parsed_colliders, parsed_faucets, parsed_drains);
    } else {
        let initial_quality = parse_quality(&args.quality);
        let initial_speed = args.speed;
        if let Err(e) = run_gui(
            preset,
            initial_quality,
            initial_speed,
            chamber,
            parsed_colliders,
            parsed_faucets,
            parsed_drains,
        ) {
            eprintln!("GUI Error: {}", e);
        }
    }
}
