use clap::Parser;
use eframe::egui;
use liquid_rust::export::PlyExporter;
use liquid_rust::math::Aabb;
use liquid_rust::sph::{SceneGenerator, ScenePreset, SolverConfig, SphSolver};
use liquid_rust::LiquidApp;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "liquid-rust",
    author = "Radu <rappar222>",
    version = "0.1.0",
    about = "High-performance 3D SPH Fluid Simulator in Rust with Real-time 3D Viewport & PLY Export"
)]
struct Args {
    /// Run in headless mode without opening GUI window
    #[arg(long)]
    headless: bool,

    /// Scene preset to simulate: dam-break, double-dam, splash, obstacle, tornado, zero-g
    #[arg(long, default_value = "dam-break")]
    preset: String,

    /// Number of simulation frames to run in headless mode
    #[arg(long, default_value_t = 120)]
    frames: usize,

    /// Time step dt per frame in seconds
    #[arg(long, default_value_t = 0.016)]
    dt: f32,

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
        "double-dam" | "doubledam" | "double" => ScenePreset::DoubleDamBreak,
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

fn run_headless(args: Args) {
    let preset = parse_preset(&args.preset);
    println!("=== liquid-rust headless mode ===");
    println!("Preset: {:?}", preset);
    println!("Frames to simulate: {}", args.frames);
    println!("Delta t: {}s", args.dt);

    let bounds = Aabb::new(glam::Vec3::new(-0.8, 0.0, -0.45), glam::Vec3::new(0.8, 1.2, 0.45));
    let config = SolverConfig::default();
    let mut solver = SphSolver::new(bounds, config);

    SceneGenerator::setup_scene(&mut solver, preset);
    let particle_count = solver.particle_count();
    println!("Spawned {} particles", particle_count);

    if let Some(ref dir) = args.export_ply {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Failed to create export directory {}: {}", dir.display(), e);
            return;
        }
        println!("Exporting PLY sequence to: {}", dir.display());
    }

    let start_total = Instant::now();
    let mut total_sim_time = 0.0f64;

    for frame in 0..args.frames {
        let t0 = Instant::now();
        solver.step(args.dt);
        let step_time = t0.elapsed().as_secs_f64();
        total_sim_time += step_time;

        if let Some(ref dir) = args.export_ply {
            let path = dir.join(format!("fluid_{:05}.ply", frame));
            let colors: Vec<[u8; 3]> = solver
                .particles
                .iter()
                .map(|p| {
                    let speed_ratio = (p.speed() / 4.0).clamp(0.0, 1.0);
                    [
                        (speed_ratio * 255.0) as u8,
                        100,
                        ((1.0 - speed_ratio) * 255.0) as u8,
                    ]
                })
                .collect();
            if let Err(e) = PlyExporter::save_binary(&path, &solver.particles, &colors) {
                eprintln!("Error exporting PLY frame {}: {}", frame, e);
            }
        }

        if frame % 20 == 0 || frame == args.frames - 1 {
            println!(
                "Frame {:4}/{} | Step time: {:.2}ms | Avg FPS: {:.1}",
                frame + 1,
                args.frames,
                step_time * 1000.0,
                (frame + 1) as f64 / total_sim_time
            );
        }
    }

    let total_elapsed = start_total.elapsed().as_secs_f64();
    println!("--------------------------------------------------");
    println!("Simulation finished!");
    println!("Total time: {:.3}s", total_elapsed);
    println!("Average time per frame: {:.2}ms", (total_sim_time / args.frames as f64) * 1000.0);
    println!("Effective simulation FPS: {:.1}", args.frames as f64 / total_sim_time);
    println!("Throughput: {:.0} particle-steps/sec", (particle_count * args.frames) as f64 / total_sim_time);
}

fn run_gui() -> Result<(), eframe::Error> {
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
        Box::new(|cc| Ok(Box::new(LiquidApp::new(cc)))),
    )
}

fn main() {
    let args = Args::parse();

    if args.headless || args.benchmark {
        run_headless(args);
    } else {
        if let Err(e) = run_gui() {
            eprintln!("GUI Error: {}", e);
        }
    }
}
