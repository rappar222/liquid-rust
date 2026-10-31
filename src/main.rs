use clap::Parser;
use eframe::egui;
use liquid_rust::export::{Mp4Encoder, PlyExporter, SoftwareRenderer};
use liquid_rust::math::Aabb;
use liquid_rust::render::{Camera, ColorMapType, ColorMode};
use liquid_rust::sph::{
    compute_lattice_rest_density, SceneGenerator, ScenePreset, SimulationQuality, SolverConfig,
    SphSolver,
};
use liquid_rust::LiquidApp;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "liquid-rust",
    author = "Radu <rappar222>",
    version = "0.1.0",
    about = "High-performance 3D SPH Fluid Simulator in Rust with Real-time 3D Viewport, MP4 Video & PLY Export"
)]
struct Args {
    /// Run in headless mode without opening GUI window
    #[arg(long)]
    headless: bool,

    /// Scene preset to simulate: dam-break, double-dam, splash, obstacle, tornado, zero-g
    #[arg(long, default_value = "dam-break")]
    preset: String,

    /// Particle quality / resolution: draft, medium, high, ultra, cinematic
    #[arg(long, short = 'q', default_value = "medium")]
    quality: String,

    /// Custom particle radius override (smaller = higher quality & finer fluid)
    #[arg(long)]
    particle_radius: Option<f32>,

    /// Custom particle spacing override (smaller = denser fluid lattice)
    #[arg(long)]
    spacing: Option<f32>,

    /// Total simulation duration in seconds (e.g. --time 4.0 for a 4-second video). Overrides --frames.
    #[arg(long, short = 't', alias = "duration")]
    time: Option<f32>,

    /// Number of simulation frames to run (used if --time is not specified)
    #[arg(long, default_value_t = 120)]
    frames: usize,

    /// Framerate (FPS) for video export and simulation stepping
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Custom time step dt per frame in seconds (defaults to 1.0 / fps)
    #[arg(long)]
    dt: Option<f32>,

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

    /// Print performance benchmark statistics
    #[arg(long)]
    benchmark: bool,
}

fn parse_preset(name: &str) -> ScenePreset {
    match name.to_lowercase().as_str() {
        "dam-break" | "dambreak" | "dam" => ScenePreset::DamBreak,
        "double-dam" | "doubledam" | "double" | "double-dam-break" | "doubledambreak" => {
            ScenePreset::DoubleDamBreak
        }
        "splash" | "droplet" | "dropletsplash" => ScenePreset::DropletSplash,
        "obstacle" | "drop" | "obstacledrop" => ScenePreset::ObstacleDrop,
        "tornado" | "vortex" => ScenePreset::VortexTornado,
        "zero-g" | "zerog" | "blobs" => ScenePreset::ZeroGravityBlobs,
        _ => {
            eprintln!("Unknown preset '{}', defaulting to 'dam-break'", name);
            ScenePreset::DamBreak
        }
    }
}

fn parse_quality(name: &str) -> SimulationQuality {
    match name.to_lowercase().as_str() {
        "draft" | "low" | "fast" => SimulationQuality::Draft,
        "medium" | "med" | "normal" => SimulationQuality::Medium,
        "high" | "hi" | "smooth" => SimulationQuality::High,
        "ultra" | "realistic" => SimulationQuality::Ultra,
        "cinematic" | "vfx" | "film" => SimulationQuality::Cinematic,
        _ => {
            eprintln!("Unknown quality '{}', defaulting to 'medium'", name);
            SimulationQuality::Medium
        }
    }
}

fn run_headless(args: Args) {
    let preset = parse_preset(&args.preset);
    let quality = parse_quality(&args.quality);
    let fps = args.fps.max(1);

    let (total_frames, dt) = if let Some(t) = args.time {
        let f = (t * fps as f32).round().max(1.0) as usize;
        let delta_t = 1.0 / fps as f32;
        (f, delta_t)
    } else {
        let delta_t = args.dt.unwrap_or(1.0 / fps as f32);
        (args.frames, delta_t)
    };

    println!("=== liquid-rust headless mode ===");
    println!("Preset: {:?}", preset);
    println!("Quality: {} ({:?})", quality.name(), quality);
    println!("Total Frames: {} ({:.2}s @ {} fps)", total_frames, total_frames as f32 / fps as f32, fps);
    println!("Time step dt: {:.5}s", dt);

    let bounds = Aabb::new(glam::Vec3::new(-0.8, 0.0, -0.45), glam::Vec3::new(0.8, 1.2, 0.45));
    let mut config = SolverConfig::default();
    quality.apply(&mut config);

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
    SceneGenerator::setup_scene(&mut solver, preset);

    let particle_count = solver.particle_count();
    println!("Spawned {} particles (particle radius: {:.4}, spacing: {:.4})",
        particle_count, solver.config.particle_radius, solver.config.spacing);

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

    let mut software_renderer = if mp4_encoder.is_some() {
        Some(SoftwareRenderer::new(args.video_width, args.video_height))
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

    for frame in 0..total_frames {
        let t0 = Instant::now();
        solver.step(dt);
        let step_time = t0.elapsed().as_secs_f64();
        total_sim_time += step_time;

        // Render & stream to MP4
        if let (Some(ref mut enc), Some(ref mut rend)) = (&mut mp4_encoder, &mut software_renderer) {
            rend.render_frame(
                &solver.particles,
                solver.config.particle_radius,
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

        if frame % 20 == 0 || frame == total_frames - 1 {
            println!(
                "Frame {:4}/{} ({:.1}s) | Step time: {:.2}ms | Avg FPS: {:.1}",
                frame + 1,
                total_frames,
                (frame + 1) as f32 / fps as f32,
                step_time * 1000.0,
                (frame + 1) as f64 / total_sim_time
            );
        }
    }

    // Finalize MP4
    if let Some(enc) = mp4_encoder {
        println!("Finalizing MP4 video encoding...");
        match enc.finish() {
            Ok(_) => println!("MP4 video saved successfully!"),
            Err(e) => eprintln!("Error finalizing MP4 video: {}", e),
        }
    }

    let total_elapsed = start_total.elapsed().as_secs_f64();
    println!("--------------------------------------------------");
    println!("Simulation finished!");
    println!("Total time: {:.3}s", total_elapsed);
    println!("Average time per frame: {:.2}ms", (total_sim_time / total_frames as f64) * 1000.0);
    println!("Effective simulation FPS: {:.1}", total_frames as f64 / total_sim_time);
    println!("Throughput: {:.0} particle-steps/sec", (particle_count * total_frames) as f64 / total_sim_time);
}

fn run_gui(initial_quality: SimulationQuality) -> Result<(), eframe::Error> {
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
            if initial_quality != SimulationQuality::Medium {
                app.set_quality(initial_quality);
            }
            Ok(Box::new(app))
        }),
    )
}

fn main() {
    let args = Args::parse();

    if args.headless || args.benchmark || args.export_mp4.is_some() {
        run_headless(args);
    } else {
        let initial_quality = parse_quality(&args.quality);
        if let Err(e) = run_gui(initial_quality) {
            eprintln!("GUI Error: {}", e);
        }
    }
}
