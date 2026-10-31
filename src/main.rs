use clap::Parser;
use eframe::egui;
use liquid_rust::export::{Mp4Encoder, PlyExporter, SoftwareRenderer};
use liquid_rust::gpu::GpuSphSolver;
use liquid_rust::math::Aabb;
use liquid_rust::render::{Camera, ColorMapType, ColorMode};
use liquid_rust::sph::{
    compute_lattice_rest_density, SceneGenerator, ScenePreset, SimulationQuality, SolverConfig,
    SphSolver,
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
    version = "0.1.1",
    about = "High-performance 3D SPH Fluid Simulator in Rust with Real-time 3D Viewport, MP4 Video & PLY Export"
)]
struct Args {
    /// Run in headless mode without opening GUI window
    #[arg(long)]
    headless: bool,

    /// Scene preset to simulate: dam-break, double-dam, splash, obstacle, tornado, zero-g
    #[arg(long, default_value = "dam-break")]
    preset: String,

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
        "cinematic" | "vfx" => SimulationQuality::Cinematic,
        "extreme" | "1m" | "insane" => SimulationQuality::Extreme,
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

    let bounds = Aabb::new(glam::Vec3::new(-0.8, 0.0, -0.45), glam::Vec3::new(0.8, 1.2, 0.45));
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
    SceneGenerator::setup_scene(&mut solver, preset);

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
            gpu.step(
                solver.bounds,
                &solver.config,
                &solver.colliders,
                solver.mouse_ray,
                solver.mouse_force_strength,
                solver.mouse_force_mode,
                dt,
            );
            gpu.read_particles(&mut solver.particles);
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

fn run_gui(initial_preset: ScenePreset, initial_quality: SimulationQuality, initial_speed: f32) -> Result<(), eframe::Error> {
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
            if initial_preset != ScenePreset::DamBreak {
                app.load_preset(initial_preset);
            }
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
        let preset = parse_preset(&args.preset);
        let initial_quality = parse_quality(&args.quality);
        let initial_speed = args.speed;
        if let Err(e) = run_gui(preset, initial_quality, initial_speed) {
            eprintln!("GUI Error: {}", e);
        }
    }
}
