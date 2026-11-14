use glam::Vec3;
use liquid_rust::export::PlyExporter;
use liquid_rust::math::{Aabb, Ray};
use liquid_rust::render::{ColorMap, ColorMapType};
use liquid_rust::sph::collider::Collider;
use liquid_rust::sph::kernel::SphKernels;
use liquid_rust::sph::particle::Particle;
use liquid_rust::sph::scene::{SceneGenerator, ScenePreset};
use liquid_rust::sph::solver::{compute_lattice_rest_density, SolverConfig, SphSolver};
use liquid_rust::sph::spatial::SpatialHashGrid;

#[test]
fn test_aabb_containment_and_clamp() {
    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0));
    assert!(bounds.contains(Vec3::new(0.0, 1.0, 0.0)));
    assert!(!bounds.contains(Vec3::new(2.0, 1.0, 0.0)));

    let clamped = bounds.clamp_point(Vec3::new(5.0, -2.0, 0.5), 0.1);
    assert_eq!(clamped.x, 0.9);
    assert_eq!(clamped.y, 0.1);
    assert_eq!(clamped.z, 0.5);
}

#[test]
fn test_ray_plane_intersection() {
    let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
    let t = ray.intersect_plane(Vec3::ZERO, Vec3::Y);
    assert!(t.is_some());
    assert!((t.unwrap() - 5.0).abs() < 1e-5);
    assert_eq!(ray.at(t.unwrap()), Vec3::ZERO);
}

#[test]
fn test_sph_kernels_compact_support() {
    let h = 0.1;
    let kernels = SphKernels::new(h);

    let w_center = kernels.poly6(0.0);
    assert!(w_center > 0.0);

    let w_mid = kernels.poly6(0.05 * 0.05);
    assert!(w_mid > 0.0);
    assert!(w_center > w_mid);

    let w_outside = kernels.poly6(0.15 * 0.15);
    assert_eq!(w_outside, 0.0);

    let grad_inside = kernels.spiky_gradient(Vec3::new(0.05, 0.0, 0.0), 0.05);
    assert!(grad_inside.x < 0.0);

    let grad_outside = kernels.spiky_gradient(Vec3::new(0.15, 0.0, 0.0), 0.15);
    assert_eq!(grad_outside, Vec3::ZERO);
}

#[test]
fn test_spatial_grid_neighbor_search() {
    let cell_size = 0.1;
    let mut grid = SpatialHashGrid::new(cell_size, 16);

    let positions = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.05, 0.0, 0.0),
        Vec3::new(0.08, 0.05, 0.0),
        Vec3::new(2.0, 2.0, 2.0),
    ];

    grid.build(&positions);

    let mut neighbors_0 = Vec::new();
    grid.for_each_neighbor(positions[0], |idx| {
        neighbors_0.push(idx);
    });

    assert!(neighbors_0.contains(&0));
    assert!(neighbors_0.contains(&1));
    assert!(!neighbors_0.contains(&3));
}

#[test]
fn test_rest_density_computation() {
    let h = 0.075;
    let spacing = 0.042;
    let rho0 = compute_lattice_rest_density(h, spacing);
    assert!(rho0 > 1000.0);
}

#[test]
fn test_density_vs_rest_density() {
    use liquid_rust::sph::SimulationQuality;
    let mut config = SolverConfig::default();
    SimulationQuality::Ultra.apply(&mut config, false);
    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.45), Vec3::new(1.0, 1.2, 0.45));
    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DoubleDamBreak);
    
    let center_pos = Vec3::new(-0.7, 0.4, 0.0);
    let (_, p) = cpu_solver.particles.iter().enumerate().min_by(|(_, a), (_, b)| {
        a.position.distance_squared(center_pos).partial_cmp(&b.position.distance_squared(center_pos)).unwrap()
    }).unwrap();
    println!("Center particle pos: {:?}, target rest_density: {}", p.position, config.rest_density);
    
    let kernels = SphKernels::new(config.smoothing_length);
    let mut d = kernels.poly6(0.0);
    let mut grad_ci = Vec3::ZERO;
    let mut sum_grad_sq = 0.0f32;
    let inv_rho0 = 1.0 / config.rest_density;
    for other in &cpu_solver.particles {
        if other.id != p.id {
            let r_vec = p.position - other.position;
            let r2 = r_vec.length_squared();
            if r2 < kernels.h2 && r2 > 1e-12 {
                let r = r2.sqrt();
                d += kernels.poly6(r2);
                let grad_w = kernels.spiky_gradient(r_vec, r);
                let grad_j = grad_w * inv_rho0;
                grad_ci += grad_j;
                sum_grad_sq += grad_j.length_squared();
            }
        }
    }
    sum_grad_sq += grad_ci.length_squared();
    let relaxation_eps = 1200.0 * (0.075 / config.smoothing_length).powi(2);
    println!("Actual computed density: {}, diff: {}, ratio: {}", d, d - config.rest_density, d / config.rest_density);
    println!("sum_grad_sq = {}, relaxation_eps = {}", sum_grad_sq, relaxation_eps);
}

#[test]
fn test_sphere_collider_response() {
    let collider = Collider::sphere(Vec3::new(0.0, 0.0, 0.0), 0.5);
    let mut pos = Vec3::new(0.4, 0.0, 0.0);
    let mut vel = Vec3::new(-2.0, 0.0, 0.0);
    let p_radius = 0.05;

    let hit = collider.resolve(&mut pos, &mut vel, p_radius);
    assert!(hit);
    assert!((pos.x - 0.55).abs() < 1e-4);
    assert!(vel.x > 0.0);
}

#[test]
fn test_box_collider_response() {
    let collider = Collider::box_obstacle(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
    let mut pos = Vec3::new(0.0, 0.48, 0.0);
    let mut vel = Vec3::new(0.0, -1.0, 0.0);
    let p_radius = 0.05;

    let hit = collider.resolve(&mut pos, &mut vel, p_radius);
    assert!(hit);
    assert!(pos.y >= 0.55 - 1e-5);
    assert!(vel.y >= 0.0);
}

#[test]
fn test_pbf_solver_step_stability() {
    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
    let config = SolverConfig::default();
    let mut solver = SphSolver::new(bounds, config);

    SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);
    assert!(solver.particle_count() > 0);

    let initial_count = solver.particle_count();
    let solver_bounds = solver.bounds;

    // Step simulation 30 times
    for _ in 0..30 {
        solver.step(0.016);
    }

    assert_eq!(solver.particle_count(), initial_count);

    // Verify all particles stay within simulation bounds and speeds remain reasonable
    for p in &solver.particles {
        assert!(p.position.x >= solver_bounds.min.x - 0.01);
        assert!(p.position.x <= solver_bounds.max.x + 0.01);
        assert!(p.position.y >= solver_bounds.min.y - 0.01);
        assert!(p.position.y <= solver_bounds.max.y + 0.01);
        assert!(p.position.z >= solver_bounds.min.z - 0.01);
        assert!(p.position.z <= solver_bounds.max.z + 0.01);
        assert!(!p.velocity.is_nan());
        assert!(!p.position.is_nan());
        assert!(p.speed() < 15.0); // No explosions!
    }
}

#[test]
fn test_ply_export_binary_and_ascii() {
    let p1 = Particle::new(Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.0, 0.0, 0.0), 1.0, 0);
    let p2 = Particle::new(Vec3::new(-0.1, 0.5, 0.2), Vec3::new(0.0, -1.0, 0.0), 1.0, 1);
    let particles = vec![p1, p2];
    let colors = vec![[255, 0, 0], [0, 255, 0]];

    let dir = std::env::temp_dir().join("liquid_rust_test_export_pbf");
    let _ = std::fs::create_dir_all(&dir);

    let bin_path = dir.join("test_bin.ply");
    let ascii_path = dir.join("test_ascii.ply");

    PlyExporter::save_binary(&bin_path, &particles, &colors).expect("Save binary PLY");
    PlyExporter::save_ascii(&ascii_path, &particles, &colors).expect("Save ascii PLY");

    assert!(bin_path.exists());
    assert!(ascii_path.exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_color_maps() {
    for &cmap in &[
        ColorMapType::Ocean,
        ColorMapType::Turbo,
        ColorMapType::Viridis,
        ColorMapType::Fire,
        ColorMapType::Plasma,
        ColorMapType::Ice,
        ColorMapType::Chrome,
    ] {
        let c0 = ColorMap::sample(cmap, 0.0);
        let c2 = ColorMap::sample(cmap, 1.0);
        assert_ne!(c0, c2);
    }
    assert_eq!(ColorMapType::from_slug("chrome"), Some(ColorMapType::Chrome));
    assert_eq!(ColorMapType::from_slug("mercury"), Some(ColorMapType::Chrome));
}

#[test]
fn test_double_dam_collision_splash() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::{SceneGenerator, ScenePreset, SimulationQuality, SolverConfig, SphSolver};

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.05), Vec3::new(1.0, 1.2, 0.05));
    let config = SolverConfig::default();
    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    cpu_solver.set_2d(true);
    cpu_solver.apply_quality(SimulationQuality::Draft);
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DoubleDamBreak);

    let mut gpu = match GpuSphSolver::new(cpu_solver.bounds, &cpu_solver.config, &cpu_solver.particles) {
        Ok(g) => g,
        Err(e) => {
            panic!("GPU failed: {}", e);
        }
    };

    eprintln!("\n=== 2D DOUBLE DAM COLLISION TEST ===");
    for f in 1..=60 {
        gpu.step(cpu_solver.bounds, &cpu_solver.chamber, &cpu_solver.config, &[], None, 0.0, 0, 1.0 / 60.0, true);
        if f % 15 == 0 {
            let mut readback = Vec::new();
            gpu.read_particles(&mut readback);
            let mut center_count = 0;
            let mut max_y_center = 0.0f32;
            for p in &readback {
                if p.position.x.abs() < 0.15 {
                    center_count += 1;
                    max_y_center = max_y_center.max(p.position.y);
                }
            }
            eprintln!("Frame {:02}: Center (-0.15..0.15) particle count: {}, max_y: {}", f, center_count, max_y_center);
        }
    }
}

#[test]
fn test_gpu_slow_motion_stability() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::SimulationQuality;

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.45), Vec3::new(1.0, 1.2, 0.45));
    let mut config = SolverConfig::default();
    SimulationQuality::Ultra.apply(&mut config, false);

    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DoubleDamBreak);
    let bounds = cpu_solver.bounds;

    let mut gpu_solver = match GpuSphSolver::new(bounds, &config, &cpu_solver.particles) {
        Ok(s) => s,
        Err(_) => return,
    };

    let slow_dt = (1.0 / 60.0) * 0.0625; // 16x slow
    let initial_max_y = cpu_solver.particles.iter().map(|p| p.position.y).fold(0.0f32, f32::max);
    println!("Initial max y: {}", initial_max_y);

    for frame in 0..100 {
        gpu_solver.step(bounds, &cpu_solver.chamber, &config, &[], None, 0.0, 0, slow_dt, false);
        let mut readback = Vec::new();
        gpu_solver.read_particles(&mut readback);
        let max_y = readback.iter().map(|p| p.position.y).fold(0.0f32, f32::max);
        let max_speed = readback.iter().map(|p| p.speed()).fold(0.0f32, f32::max);
        if frame % 20 == 0 {
            println!("GPU Frame {}: max y = {:.4}, max speed = {:.4}", frame, max_y, max_speed);
        }
        assert!(max_y < initial_max_y + 0.05, "Particles lifting off: max y = {} (initial was {})", max_y, initial_max_y);
        assert!(max_speed < 8.0, "Particles exploding: max speed = {}", max_speed);
    }
}

#[test]
fn test_ultra_quality_gpu_stability() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::SimulationQuality;

    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
    let mut config = SolverConfig::default();
    SimulationQuality::Ultra.apply(&mut config, false);

    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DamBreak);

    let mut gpu_solver = match GpuSphSolver::new(bounds, &config, &cpu_solver.particles) {
        Ok(s) => s,
        Err(_) => return,
    };

    for _ in 0..10 {
        gpu_solver.step(bounds, &cpu_solver.chamber, &config, &[], None, 0.0, 0, 1.0 / 60.0, false);
    }

    let mut readback = Vec::new();
    gpu_solver.read_particles(&mut readback);

    for p in &readback {
        assert!(!p.position.is_nan());
        assert!(!p.velocity.is_nan());
        assert!(p.speed() < 15.0, "Ultra quality must be completely stable, got speed {}", p.speed());
    }
}

#[test]
fn test_extreme_quality_particle_count() {
    use liquid_rust::sph::SimulationQuality;

    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
    let mut config = SolverConfig::default();
    SimulationQuality::Extreme.apply(&mut config, false);

    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DamBreak);

    // Extreme quality must spawn over 1,000,000 particles
    assert!(
        cpu_solver.particles.len() >= 1_000_000,
        "Extreme quality must spawn at least 1,000,000 particles, got {}",
        cpu_solver.particles.len()
    );
}

#[test]
fn test_triangle_mesh_obj_parser_and_collision() {
    use liquid_rust::sph::{Collider, TriangleMesh};

    // Simple quad (2 triangles) forming a tilted ramp facing upward (+Y)
    let obj_data = "\
v -0.5 0.2 -0.5
v 0.5 0.2 -0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 3 2
f 1 4 3
";
    let mut mesh = TriangleMesh::from_obj_str(obj_data, "Ramp")
        .expect("Valid OBJ string should parse cleanly");
    mesh.position = Vec3::ZERO;
    mesh.scale = 1.0;
    mesh.update_transform();
    assert_eq!(mesh.triangles.len(), 2);

    let collider = Collider::mesh(std::sync::Arc::new(mesh));

    // Test particle penetrating ramp
    let mut pos = Vec3::new(0.0, 0.34, 0.0);
    let mut vel = Vec3::new(0.0, -2.0, 0.0);
    let radius = 0.05;

    let hit = collider.resolve(&mut pos, &mut vel, radius);
    assert!(hit, "Particle should collide with ramp surface");
    assert!(pos.y > 0.35, "Particle should be pushed above the ramp surface");
    assert!(vel.y > -0.2, "Downward velocity should be absorbed on inelastic collision");
    assert!(vel.z < 0.0, "Fluid particle should slide down the tilted ramp");
}

#[test]
fn test_cylinder_collider_response() {
    use liquid_rust::sph::Collider;

    let cylinder = Collider::cylinder(Vec3::new(0.0, 0.5, 0.0), 0.2, 0.6);

    // Particle inside the cylinder radius
    let mut pos = Vec3::new(0.05, 0.5, 0.0);
    let mut vel = Vec3::new(1.0, 0.0, 0.0);
    let radius = 0.02;

    let hit = cylinder.resolve(&mut pos, &mut vel, radius);
    assert!(hit, "Particle should collide with cylinder wall");
    let dist_xz = (pos.x * pos.x + pos.z * pos.z).sqrt();
    assert!(dist_xz >= 0.2 + radius - 1e-4, "Particle should be pushed outside cylinder radius");
}

#[test]
fn test_faucet_particle_emission() {
    use liquid_rust::sph::{Faucet, FaucetShape};

    let mut id = 100;
    // Waterfall faucet emitting 1000 p/s
    let mut waterfall = Faucet::new_waterfall("Waterfall", Vec3::new(0.0, 1.0, 0.0), 0.4, 0.05, 1000.0);
    let particles = waterfall.emit(0.05, 0.02, &mut id); // 0.05s * 1000 p/s = 50 particles
    assert_eq!(particles.len(), 50);
    for p in &particles {
        assert_eq!(p.position.y, 1.0);
        assert!(p.position.x >= -0.21 && p.position.x <= 0.21);
        assert!(p.velocity.y < -1.0, "Velocity should be directed downwards");
    }

    // Shower faucet with spray spread cone
    let mut shower = Faucet::new_shower("Shower", Vec3::new(0.0, 1.0, 0.0), 0.1, 500.0, 0.2);
    let shower_particles = shower.emit(0.1, 0.02, &mut id); // 0.1s * 500 p/s = 50 particles
    assert_eq!(shower_particles.len(), 50);
    assert_eq!(shower.shape, FaucetShape::Circle { radius: 0.1 });

    // Point faucet / tap
    let mut tap = Faucet::new_point("Tap", Vec3::new(0.1, 0.8, 0.2), Vec3::new(1.0, -1.0, 0.0), 200.0);
    let tap_particles = tap.emit(0.05, 0.02, &mut id); // 10 particles
    assert_eq!(tap_particles.len(), 10);
    for p in &tap_particles {
        assert_eq!(p.position, Vec3::new(0.1, 0.8, 0.2));
    }
}

#[test]
fn test_drain_particle_elimination() {
    use liquid_rust::sph::{CubeFace, Drain, Particle};

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let floor_drain = Drain::new("Floor Drain", CubeFace::Floor, Vec3::new(0.0, 0.0, 0.0), 0.25);
    let wall_drain = Drain::new("Wall Drain", CubeFace::Left, Vec3::new(0.0, 0.3, 0.0), 0.2);

    let radius = 0.02;

    // Particle on the floor inside the drain aperture
    let p_drain = Particle::new(Vec3::new(0.05, 0.01, 0.05), Vec3::ZERO, 1.0, 1);
    assert!(floor_drain.should_drain(p_drain.position, radius, &bounds));

    // Particle on the floor OUTSIDE the drain aperture
    let p_safe = Particle::new(Vec3::new(0.5, 0.01, 0.5), Vec3::ZERO, 1.0, 2);
    assert!(!floor_drain.should_drain(p_safe.position, radius, &bounds));

    // Particle high in the air above drain (should not drain until it touches floor)
    let p_air = Particle::new(Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO, 1.0, 3);
    assert!(!floor_drain.should_drain(p_air.position, radius, &bounds));

    // Particle touching left wall inside wall drain
    let p_wall = Particle::new(Vec3::new(-0.99, 0.32, 0.05), Vec3::ZERO, 1.0, 4);
    assert!(wall_drain.should_drain(p_wall.position, radius, &bounds));
}

#[test]
fn test_solver_faucets_and_drains_cycle() {
    use liquid_rust::sph::{CubeFace, Drain, Faucet};

    let bounds = Aabb::new(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5));
    let config = SolverConfig::default();
    let mut solver = SphSolver::new(bounds, config);

    // Start with empty particles
    assert_eq!(solver.particle_count(), 0);

    // Add faucet at top
    solver.faucets.push(Faucet::new_waterfall(
        "Top Faucet",
        Vec3::new(0.0, 0.9, 0.0),
        0.2,
        0.05,
        500.0,
    ));

    // Add drain at bottom
    solver.drains.push(Drain::new(
        "Floor Drain",
        CubeFace::Floor,
        Vec3::new(0.0, 0.0, 0.0),
        0.15,
    ));

    // Run steps: faucet should inject particles into solver
    for _ in 0..10 {
        solver.step(0.016);
    }
    assert!(solver.particle_count() > 0, "Faucet should have emitted particles into solver");

    // Clear particles and test drain removing fallen particles
    solver.clear_particles();
    solver.particles.push(liquid_rust::sph::Particle::new(
        Vec3::new(0.02, 0.01, 0.02),
        Vec3::ZERO,
        1.0,
        1,
    ));
    assert_eq!(solver.particle_count(), 1);
    solver.drain_particles();
    assert_eq!(solver.particle_count(), 0, "Drain should eliminate particles over the aperture");
}

#[test]
fn test_cli_command_generator() {
    use liquid_rust::app::{CliCommandMode, LiquidApp};
    use liquid_rust::sph::{ChamberShape, CubeFace, Drain, Faucet, ScenePreset, SimulationQuality};

    let mut app = LiquidApp::new_headless();

    // Default GUI mode
    let cmd = app.generate_cli_command();
    assert!(cmd.starts_with("liquid-rust"));
    assert!(cmd.contains("--preset dam-break"));
    assert!(cmd.contains("--quality medium"));

    // Switch to MP4 Video export mode
    app.cli_mode = CliCommandMode::Mp4Video;
    app.cli_duration = 3.5;
    app.cli_fps = 60;
    app.cli_mp4_output = "waterfall_test.mp4".to_string();
    app.load_preset(ScenePreset::Empty);
    app.current_quality = SimulationQuality::Cinematic;
    app.sim_speed = 0.5;

    // Chamber bathtub shape and scaling
    app.solver.chamber.shape = ChamberShape::Bathtub;
    app.solver.chamber.scale = glam::Vec3::new(1.5, 1.2, 1.0);

    // Add faucet and drain
    app.solver.faucets.push(Faucet::new_waterfall("Waterfall", glam::Vec3::new(0.0, 0.8, 0.0), 0.3, 0.05, 750.0));
    app.solver.drains.push(Drain::new("Drain", CubeFace::Floor, glam::Vec3::new(0.1, 0.0, -0.1), 0.12));

    let cmd_mp4 = app.generate_cli_command();
    assert!(cmd_mp4.contains("--export-mp4 waterfall_test.mp4"));
    assert!(cmd_mp4.contains("--time 3.5"));
    assert!(cmd_mp4.contains("--fps 60"));
    assert!(cmd_mp4.contains("--preset empty"));
    assert!(cmd_mp4.contains("--quality cinematic"));
    assert!(cmd_mp4.contains("--sim-speed 0.50"));
    assert!(cmd_mp4.contains("--chamber-shape bathtub"));
    assert!(cmd_mp4.contains("--chamber-scale 1.50,1.20,1.00"));
    assert!(cmd_mp4.contains("--faucet \"waterfall:0.00,0.80,0.00:0.30:750\""));
    assert!(cmd_mp4.contains("--drain \"floor:0.10,-0.10:0.12\""));

    // Switch to Headless mode
    app.cli_mode = CliCommandMode::Headless;
    let cmd_headless = app.generate_cli_command();
    assert!(cmd_headless.contains("--headless"));

    // Switch to 2D mode
    app.set_2d(true);
    let cmd_2d = app.generate_cli_command();
    assert!(cmd_2d.contains("--2d"), "Command should include --2d flag");

    // Custom chamber size & tap/stream radius round-trip
    app.set_2d(false);
    app.solver.chamber.custom_size = true;
    app.solver.chamber.base_size = glam::Vec3::new(2.2, 1.4, 1.1);
    app.solver.faucets.push(Faucet::new_circle(
        "Tap #1",
        glam::Vec3::new(-0.5, 0.9, 0.0),
        glam::Vec3::new(1.8, -1.0, 0.0),
        0.115,
        450.0,
    ));
    let cmd_custom = app.generate_cli_command();
    assert!(cmd_custom.contains("--chamber-size 2.20,1.40,1.10"));
    assert!(
        cmd_custom.contains("--faucet \"stream:-0.50,0.90,0.00:1.80,-1.00,0.00:450:0.00:0.115\""),
        "Stream/Tap CLI should preserve nozzle radius: {}",
        cmd_custom
    );
}

#[test]
fn test_cpu_2d_flow() {
    use liquid_rust::sph::{SceneGenerator, ScenePreset, SimulationQuality, SolverConfig, SphSolver};

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.05), Vec3::new(1.0, 1.2, 0.05));
    let config = SolverConfig::default();
    let mut solver = SphSolver::new(bounds, config);
    solver.set_2d(true);
    solver.apply_quality(SimulationQuality::Draft);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DoubleDamBreak);

    eprintln!("\n=== CPU 2D DOUBLE DAM FLOW TEST ===");
    for frame in 1..=60 {
        solver.step(1.0 / 60.0);
        if frame % 10 == 0 {
            let mut min_x_left = 999.0f32;
            let mut max_x_left = -999.0f32;
            let mut min_x_right = 999.0f32;
            let mut max_x_right = -999.0f32;
            let mut avg_speed = 0.0f32;
            for p in &solver.particles {
                if p.position.x < 0.0 {
                    min_x_left = min_x_left.min(p.position.x);
                    max_x_left = max_x_left.max(p.position.x);
                } else {
                    min_x_right = min_x_right.min(p.position.x);
                    max_x_right = max_x_right.max(p.position.x);
                }
                avg_speed += p.velocity.length();
            }
            avg_speed /= solver.particles.len() as f32;
            eprintln!("CPU Frame {:02}: Left X [{:.3}, {:.3}], Right X [{:.3}, {:.3}], avg_speed: {:.3}", 
                frame, min_x_left, max_x_left, min_x_right, max_x_right, avg_speed);
        }
    }
}
#[test]
fn test_gpu_2d_collision() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::{SceneGenerator, ScenePreset, SimulationQuality, SolverConfig, SphSolver};

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.05), Vec3::new(1.0, 1.2, 0.05));
    let config = SolverConfig::default();
    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    cpu_solver.set_2d(true);
    cpu_solver.apply_quality(SimulationQuality::Draft);
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DoubleDamBreak);

    let mut gpu = match GpuSphSolver::new(cpu_solver.bounds, &cpu_solver.config, &cpu_solver.particles) {
        Ok(g) => g,
        Err(e) => {
            panic!("GPU failed: {}", e);
        }
    };
    eprintln!("\n=== 2D DOUBLE DAM BREAK LONG RUN DIAGNOSTIC ===");
    for frame in 1..=180 {
        gpu.step(cpu_solver.bounds, &cpu_solver.chamber, &cpu_solver.config, &[], None, 0.0, 0, 1.0 / 60.0, true);
        if frame % 30 == 0 {
            let mut readback = Vec::new();
            gpu.read_particles(&mut readback);
            let mut min_x = 999.0f32;
            let mut max_x = -999.0f32;
            let mut max_y = -999.0f32;
            let mut avg_speed = 0.0f32;
            for p in &readback {
                min_x = min_x.min(p.position.x);
                max_x = max_x.max(p.position.x);
                max_y = max_y.max(p.position.y);
                avg_speed += p.velocity.length();
            }
            avg_speed /= readback.len() as f32;
            eprintln!("Frame {:03}: X [{:.3}, {:.3}], max_y: {:.3}, avg_speed: {:.3}", 
                frame, min_x, max_x, max_y, avg_speed);
        }
    }
}
#[test]
fn test_2d_simulation_planar_physics() {
    use liquid_rust::sph::{SceneGenerator, ScenePreset, SolverConfig, SphSolver};

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.05), Vec3::new(1.0, 1.2, 0.05));
    let config = SolverConfig::default();
    let mut solver = SphSolver::new(bounds, config);
    solver.set_2d(true);

    assert!(solver.is_2d);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);

    assert!(solver.particle_count() > 0);

    // Initial particles must all be in Z = 0 plane
    for p in &solver.particles {
        assert_eq!(p.position.z, 0.0, "Initial particle Z must be 0.0");
        assert_eq!(p.velocity.z, 0.0, "Initial velocity Z must be 0.0");
    }

    // Step the simulation in 2D mode
    for _ in 0..60 {
        solver.step(0.016);
    }

    // After 60 steps of fluid motion, all particles must strictly remain at Z = 0
    for p in &solver.particles {
        assert!(p.position.z.abs() < 1e-6, "Particle Z must remain 0.0 after steps, got {}", p.position.z);
        assert!(p.velocity.z.abs() < 1e-6, "Velocity Z must remain 0.0 after steps, got {}", p.velocity.z);
        assert!(!p.position.is_nan());
        assert!(!p.velocity.is_nan());
    }
}

#[test]
fn test_2d_quality_resolutions_and_cli() {
    use liquid_rust::app::LiquidApp;
    use liquid_rust::sph::SimulationQuality;

    let mut app = LiquidApp::new_headless();
    app.set_2d(true);

    // In 2D mode, Extreme should NOT trigger 3D extreme warning modal
    assert!(!app.show_extreme_modal);
    app.set_quality(SimulationQuality::Extreme);
    assert!(!app.show_extreme_modal);
    assert_eq!(app.current_quality, SimulationQuality::Extreme);
    assert!(app.generate_cli_command().contains("--quality extreme"));

    app.set_quality(SimulationQuality::Draft);
    assert_eq!(app.current_quality, SimulationQuality::Draft);
    assert!(app.generate_cli_command().contains("--quality draft"));

    app.set_quality(SimulationQuality::Cinematic);
    assert_eq!(app.current_quality, SimulationQuality::Cinematic);
    assert!(app.generate_cli_command().contains("--quality cinematic"));
}

#[test]
fn test_2d_objects_zero_z() {
    use liquid_rust::sph::{Collider, ColliderShape, Faucet, Drain, CubeFace};

    let b = Collider::box_obstacle(Vec3::new(-0.2, 0.05, -0.05), Vec3::new(0.2, 0.35, 0.05));
    if let ColliderShape::Box { aabb } = b.shape {
        assert_eq!(aabb.center().z, 0.0);
    } else {
        panic!("Expected box collider");
    }

    let f = Faucet::new_circle("Tap".to_string(), Vec3::new(0.1, 0.8, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.05, 100.0);
    assert_eq!(f.position.z, 0.0);
    assert_eq!(f.direction.z, 0.0);

    let d = Drain::new("Drain".to_string(), CubeFace::Floor, Vec3::new(0.2, 0.0, 0.0), 0.1);
    assert_eq!(d.position.z, 0.0);
}


#[test]
fn test_marching_cubes_tables_valid() {
    use liquid_rust::render::surface_mesh::{MC_EDGE_CORNERS, MC_EDGE_TABLE, MC_TRI_TABLE};

    assert_eq!(MC_EDGE_TABLE[0], 0);
    assert_eq!(MC_EDGE_TABLE[255], 0);
    assert_eq!(MC_TRI_TABLE[0][0], -1);
    assert_eq!(MC_TRI_TABLE[255][0], -1);

    for &(c0, c1) in &MC_EDGE_CORNERS {
        assert!(c0 < 8 && c1 < 8 && c0 != c1);
    }

    for cube_idx in 0..256 {
        let edge_mask = MC_EDGE_TABLE[cube_idx];
        assert!(edge_mask <= 0x0FFF, "Cube {} mask out of 12-edge range", cube_idx);

        let row = &MC_TRI_TABLE[cube_idx];
        let mut count = 0usize;
        while count < 16 && row[count] != -1 {
            let e = row[count];
            assert!(
                (0..12).contains(&e),
                "Cube {} has invalid edge index {}",
                cube_idx,
                e
            );
            assert!(
                (edge_mask & (1u16 << (e as u16))) != 0,
                "Cube {} references edge {} not enabled in MC_EDGE_TABLE",
                cube_idx,
                e
            );
            count += 1;
        }
        assert_eq!(
            count % 3,
            0,
            "Cube {} triangle list length {} is not a multiple of 3",
            cube_idx,
            count
        );
    }
}

#[test]
fn test_surface_mesh_reconstruction_and_obj_export() {
    use liquid_rust::render::{SurfaceMesh, SurfaceMeshConfig};
    use liquid_rust::sph::{SimulationQuality, TriangleMesh};

    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
    let mut config = SolverConfig::default();
    SimulationQuality::Draft.apply(&mut config, false);
    let mut solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);

    for _ in 0..8 {
        solver.step(1.0 / 60.0);
    }

    let mesh_cfg = SurfaceMeshConfig::default();
    let mesh = SurfaceMesh::from_particles(
        &solver.particles,
        solver.config.particle_radius,
        solver.config.smoothing_length,
        &mesh_cfg,
    );

    assert!(!mesh.is_empty(), "Reconstructed fluid surface mesh must not be empty");
    assert!(mesh.vertices.len() > 50, "Expected > 50 welded vertices, got {}", mesh.vertices.len());
    assert!(mesh.triangle_count() > 80, "Expected > 80 triangles, got {}", mesh.triangle_count());

    for v in &mesh.vertices {
        let p = Vec3::from_array(v.position);
        let n = Vec3::from_array(v.normal);
        assert!(p.is_finite(), "Mesh vertex position must be finite");
        assert!(n.is_finite(), "Mesh vertex normal must be finite");
        assert!(
            (n.length() - 1.0).abs() < 0.05,
            "Mesh vertex normal must be normalized, got len {}",
            n.length()
        );
    }

    let obj_path = std::env::temp_dir().join("liquid_rust_test_surface_mesh.obj");
    mesh.save_obj(&obj_path).expect("OBJ export should succeed");
    let raw_obj = std::fs::read_to_string(&obj_path).expect("Read exported OBJ");
    let parsed = TriangleMesh::from_obj_str(&raw_obj, "ExportedFluid")
        .expect("Exported fluid OBJ should parse cleanly as valid geometry");
    assert_eq!(parsed.triangles.len(), mesh.triangle_count());
    let _ = std::fs::remove_file(&obj_path);
}

#[test]
fn test_surface_mesh_software_render_materials_and_textures() {
    use liquid_rust::export::SoftwareRenderer;
    use liquid_rust::render::{
        Camera, FluidMaterialPreset, MeshTextureMode, SurfaceMesh, SurfaceMeshConfig,
    };
    use liquid_rust::sph::SimulationQuality;

    let bounds = Aabb::new(Vec3::new(-0.9, 0.0, -0.45), Vec3::new(0.9, 1.2, 0.45));
    let mut config = SolverConfig::default();
    SimulationQuality::Draft.apply(&mut config, false);
    let mut solver = SphSolver::new(bounds, config);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::ObstacleDrop);
    for _ in 0..6 {
        solver.step(1.0 / 60.0);
    }

    let mut mesh_cfg = SurfaceMeshConfig::default();
    let mesh = SurfaceMesh::from_particles(
        &solver.particles,
        solver.config.particle_radius,
        solver.config.smoothing_length,
        &mesh_cfg,
    );
    assert!(!mesh.is_empty());

    let mut cam = Camera::default();
    cam.reset();
    let mut renderer = SoftwareRenderer::new(160, 120);

    // 1. Render ClearWater with Ripples
    FluidMaterialPreset::ClearWater.apply(&mut mesh_cfg);
    mesh_cfg.texture_mode = MeshTextureMode::Ripples;
    renderer.render_surface_mesh_frame(
        &mesh,
        &mesh_cfg,
        &cam,
        &solver.chamber,
        solver.bounds,
        &solver.colliders,
        0.25,
        false,
    );
    let water_sum: (u64, u64, u64) = renderer.rgb_buffer.chunks_exact(3).fold((0, 0, 0), |acc, px| {
        (acc.0 + px[0] as u64, acc.1 + px[1] as u64, acc.2 + px[2] as u64)
    });
    assert!(
        water_sum.2 > water_sum.0,
        "ClearWater render should have stronger blue channel than red: {:?}",
        water_sum
    );

    // 2. Render MoltenAmber with Caustics
    FluidMaterialPreset::MoltenAmber.apply(&mut mesh_cfg);
    mesh_cfg.texture_mode = MeshTextureMode::Caustics;
    renderer.render_surface_mesh_frame(
        &mesh,
        &mesh_cfg,
        &cam,
        &solver.chamber,
        solver.bounds,
        &solver.colliders,
        0.25,
        false,
    );
    let amber_sum: (u64, u64, u64) = renderer.rgb_buffer.chunks_exact(3).fold((0, 0, 0), |acc, px| {
        (acc.0 + px[0] as u64, acc.1 + px[1] as u64, acc.2 + px[2] as u64)
    });
    assert!(
        amber_sum.0 > water_sum.0,
        "MoltenAmber render should increase warm red energy compared to ClearWater: amber={:?} water={:?}",
        amber_sum,
        water_sum
    );

    // 3. Verify remaining texture modes evaluate without NaNs or panics
    for &tex in &[MeshTextureMode::Foam, MeshTextureMode::Grid, MeshTextureMode::None] {
        FluidMaterialPreset::LiquidChrome.apply(&mut mesh_cfg);
        mesh_cfg.texture_mode = tex;
        renderer.render_surface_mesh_frame(
            &mesh,
            &mesh_cfg,
            &cam,
            &solver.chamber,
            solver.bounds,
            &solver.colliders,
            0.5,
            false,
        );
        assert!(renderer.rgb_buffer.iter().any(|&v| v > 60));
    }
}

#[test]
fn test_vorticity_and_calm_pool_stability() {
    use liquid_rust::sph::SimulationQuality;

    // Settled pool in 2D should remain calm with flow-gated vorticity enabled
    let bounds = Aabb::new(Vec3::new(-0.9, 0.0, -0.05), Vec3::new(0.9, 1.2, 0.05));
    let mut config = SolverConfig::default();
    config.vorticity = 0.065;
    config.surface_tension = 0.035;
    let mut solver = SphSolver::new(bounds, config);
    solver.set_2d(true);
    solver.apply_quality(SimulationQuality::Draft);

    // Create a flat resting pool on the chamber floor
    let s = solver.config.spacing;
    let r = solver.config.particle_radius;
    let mut id = 0u32;
    let mut y = bounds.min.y + r * 1.2;
    while y <= bounds.min.y + 0.28 {
        let mut x = bounds.min.x + r * 1.5;
        while x <= bounds.max.x - r * 1.5 {
            solver.particles.push(Particle::new(Vec3::new(x, y, 0.0), Vec3::ZERO, 1.0, id));
            id += 1;
            x += s;
        }
        y += s;
    }

    for _ in 0..90 {
        solver.step(1.0 / 60.0);
    }

    let mut avg_speed = 0.0f32;
    let mut max_y = 0.0f32;
    for p in &solver.particles {
        assert!(p.position.is_finite() && p.velocity.is_finite());
        avg_speed += p.velocity.length();
        max_y = max_y.max(p.position.y);
    }
    avg_speed /= solver.particles.len() as f32;
    assert!(
        avg_speed < 0.15,
        "Hydrostatic pool with flow-gated vorticity must settle calmly, got avg_speed={}",
        avg_speed
    );
    assert!(
        max_y < 0.36,
        "Resting pool particles must not pop upward, got max_y={}",
        max_y
    );
}

#[test]
fn test_cli_surface_mesh_command_generation() {
    use liquid_rust::app::{CliCommandMode, LiquidApp};
    use liquid_rust::render::{FluidMaterialPreset, MeshTextureMode, RenderMode};

    let mut app = LiquidApp::new_headless();
    app.cli_mode = CliCommandMode::Mp4Video;
    app.cli_mp4_output = "studio_amber.mp4".to_string();
    app.render_mode = RenderMode::SurfaceMesh;
    FluidMaterialPreset::MoltenAmber.apply(&mut app.mesh_config);
    app.mesh_config.shallow_color = [0.95, 0.55, 0.12];
    app.mesh_config.texture_mode = MeshTextureMode::Caustics;
    app.mesh_config.ior = 1.58;
    app.mesh_config.roughness = 0.03;

    let cmd = app.generate_cli_command();
    assert!(cmd.contains("--surface-mesh"), "Expected --surface-mesh in {}", cmd);
    assert!(cmd.contains("--mesh-material amber"), "Expected --mesh-material amber in {}", cmd);
    assert!(cmd.contains("--mesh-color"), "Expected custom --mesh-color override in {}", cmd);
    assert!(cmd.contains("--mesh-texture caustics"), "Expected --mesh-texture caustics in {}", cmd);
    assert!(cmd.contains("--mesh-ior 1.580"), "Expected --mesh-ior 1.580 in {}", cmd);
    assert!(cmd.contains("--mesh-roughness 0.030"), "Expected --mesh-roughness 0.030 in {}", cmd);
}

#[test]
fn test_boundary_contribution_prevents_edge_compression_and_wall_sticking() {
    use liquid_rust::sph::{Chamber, ChamberShape, SimulationQuality};

    let chamber = Chamber::new(ChamberShape::Box);
    let bounds = chamber.compute_bounds();
    let mut config = SolverConfig::default();
    SimulationQuality::Draft.apply(&mut config, false);

    // 1. Verify analytical boundary support at floor, 2-wall edge, and 3-wall corner
    let r = config.particle_radius;
    let s = config.spacing;
    let h = config.smoothing_length;
    let (psi_floor, grad_floor) = chamber.boundary_contribution(
        Vec3::new(0.0, bounds.min.y + r, 0.0),
        r,
        s,
        h,
        &bounds,
        false,
    );
    let (psi_edge, _) = chamber.boundary_contribution(
        Vec3::new(bounds.min.x + r, bounds.min.y + r, 0.0),
        r,
        s,
        h,
        &bounds,
        false,
    );
    let (psi_corner, _) = chamber.boundary_contribution(
        Vec3::new(bounds.min.x + r, bounds.min.y + r, bounds.min.z + r),
        r,
        s,
        h,
        &bounds,
        false,
    );
    assert!(psi_floor > 0.12 && psi_floor < 0.65, "psi_floor={}", psi_floor);
    assert!(psi_edge > psi_floor, "edge must get stronger support than single wall: {} vs {}", psi_edge, psi_floor);
    assert!(psi_corner > psi_edge, "3D corner must get stronger support than 2-wall edge: {} vs {}", psi_corner, psi_edge);
    assert!(grad_floor.y < 0.0, "grad_psi_b on floor must point into solid floor (-Y), got {:?}", grad_floor);

    // 2. Verify high-energy DoubleDamBreak splash leaves zero stuck particles high on walls
    let mut solver = SphSolver::new(bounds, config);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DoubleDamBreak);
    for _ in 0..260 {
        solver.step(1.0 / 60.0);
    }

    let stuck_high = solver
        .particles
        .iter()
        .filter(|p| p.position.y > 0.52)
        .count();
    assert_eq!(
        stuck_high, 0,
        "No particles should stick high on walls/ceiling after splash settles, got {}",
        stuck_high
    );
}

#[test]
fn test_high_poly_obj_bvh_normalization_and_transforms() {
    use liquid_rust::sph::{Collider, ColliderShape};
    use std::io::Write;
    use std::time::Instant;

    // Build a 32,000-triangle UV sphere in giant millimeter coordinates (-2500..+2500)
    // plus degenerate zero-area faces to verify crash-proof BVH + auto-normalization.
    let rings = 128usize;
    let segs = 128usize;
    let mm_radius = 2500.0f32;
    let mm_offset = Vec3::new(12_000.0, -8_500.0, 4_200.0);

    let tmp_path = std::env::temp_dir().join("liquid_rust_bvh_stress_test.obj");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp_path).unwrap());
        writeln!(f, "# High-poly millimeter OBJ stress test").unwrap();
        for r in 0..=rings {
            let phi = std::f32::consts::PI * (r as f32 / rings as f32);
            for s in 0..=segs {
                let theta = std::f32::consts::TAU * (s as f32 / segs as f32);
                let v = mm_offset
                    + Vec3::new(
                        mm_radius * phi.sin() * theta.cos(),
                        mm_radius * phi.cos(),
                        mm_radius * phi.sin() * theta.sin(),
                    );
                writeln!(f, "v {:.4} {:.4} {:.4}", v.x, v.y, v.z).unwrap();
            }
        }
        let stride = segs + 1;
        for r in 0..rings {
            for s in 0..segs {
                let i0 = r * stride + s + 1;
                let i1 = i0 + 1;
                let i2 = (r + 1) * stride + s + 1;
                let i3 = i2 + 1;
                writeln!(f, "f {} {} {}", i0, i2, i1).unwrap();
                writeln!(f, "f {} {} {}", i1, i2, i3).unwrap();
            }
        }
        // Degenerate collinear/duplicate faces
        writeln!(f, "f 1 1 2").unwrap();
        writeln!(f, "f 2 2 2").unwrap();
    }

    let mut col = Collider::load_obj(&tmp_path, Vec3::new(0.0, 0.5, 0.0), 0.45)
        .expect("High-poly millimeter OBJ must load cleanly");
    let _ = std::fs::remove_file(&tmp_path);

    if let ColliderShape::Mesh(ref mut mesh_arc) = col.shape {
        let mesh = std::sync::Arc::make_mut(mesh_arc);
        let size = mesh.world_bounds.size();
        assert!(
            (size.max_element() - 0.45).abs() < 0.02,
            "5000mm OBJ should normalize to ~0.45m world diameter, got {:?}",
            size
        );
        assert!(
            mesh.wireframe_segments.len() <= 24_000,
            "Wireframe segments must be capped for smooth viewport FPS"
        );

        // Rotate, scale, and move via transform API
        mesh.set_transform(
            Vec3::new(0.15, 0.42, -0.10),
            0.60,
            Vec3::new(35.0f32.to_radians(), -50.0f32.to_radians(), 20.0f32.to_radians()),
        );
    }

    // Benchmark 20,000 fluid particles resolving against ~32k triangles via BVH
    let t0 = Instant::now();
    let mut hits = 0usize;
    for i in 0..20_000 {
        let fx = ((i % 25) as f32 / 24.0) * 1.2 - 0.6;
        let fy = (((i / 25) % 32) as f32 / 31.0) * 1.0;
        let fz = ((i / 800) as f32 / 24.0) * 1.2 - 0.6;
        let mut p = Vec3::new(fx, fy, fz);
        let mut v = Vec3::new(0.0, -1.5, 0.0);
        if col.resolve(&mut p, &mut v, 0.015) {
            hits += 1;
        }
        assert!(p.is_finite() && v.is_finite(), "Collision must never produce NaN");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(hits > 0, "Particles near transformed sphere surface must collide");
    assert!(
        elapsed_ms < 250.0,
        "BVH mesh collision for 20k particles x 32k tris took {:.2} ms (too slow!)",
        elapsed_ms
    );
}

#[test]
fn test_cli_only_simulation_qualities_excluded_from_gui() {
    use liquid_rust::sph::SimulationQuality;

    // Interactive GUI list must NOT expose offline CLI-only quality presets
    assert!(
        SimulationQuality::ALL.iter().all(|q| !q.is_cli_only()),
        "SimulationQuality::ALL must only contain interactive GUI presets"
    );
    assert_eq!(SimulationQuality::CLI_ONLY.len(), 3);
    for &q in SimulationQuality::CLI_ONLY {
        assert!(q.is_cli_only());
    }

    let mut cfg_extreme = SolverConfig::default();
    let mut cfg_studio = SolverConfig::default();
    let mut cfg_film = SolverConfig::default();
    let mut cfg_master = SolverConfig::default();
    SimulationQuality::Extreme.apply(&mut cfg_extreme, false);
    SimulationQuality::Studio.apply(&mut cfg_studio, false);
    SimulationQuality::Film.apply(&mut cfg_film, false);
    SimulationQuality::Master.apply(&mut cfg_master, false);

    assert!(cfg_studio.spacing < cfg_extreme.spacing);
    assert!(cfg_film.spacing < cfg_studio.spacing);
    assert!(cfg_master.spacing < cfg_film.spacing);
    assert!(cfg_studio.substeps >= cfg_extreme.substeps);
    assert!(cfg_master.substeps >= cfg_film.substeps);
}

#[test]
fn test_cli_render_quality_and_ssaa_software_renderer() {
    use liquid_rust::export::SoftwareRenderer;
    use liquid_rust::render::{
        Camera, CliRenderQuality, ColorMapType, ColorMode, FluidMaterialPreset, SurfaceMesh,
        SurfaceMeshConfig,
    };
    use liquid_rust::sph::SimulationQuality;

    // 1. Verify CliRenderQuality progression and mesh ceiling upgrades
    let mut mesh_std = SurfaceMeshConfig::default();
    let mut mesh_master = SurfaceMeshConfig::default();
    CliRenderQuality::Standard.apply_to_mesh_config(&mut mesh_std, false, false);
    CliRenderQuality::Master.apply_to_mesh_config(&mut mesh_master, false, false);

    assert!(!CliRenderQuality::Standard.is_cli_only());
    assert!(CliRenderQuality::Studio.is_cli_only());
    assert!(CliRenderQuality::Cinema.is_cli_only());
    assert!(CliRenderQuality::Master.is_cli_only());
    assert!(mesh_master.max_grid_cells > mesh_std.max_grid_cells);
    assert!(mesh_master.min_voxel_size < mesh_std.min_voxel_size);
    assert!(mesh_master.max_smoothing_passes > mesh_std.max_smoothing_passes);
    assert!(mesh_master.smoothing_passes > mesh_std.smoothing_passes);
    assert!(CliRenderQuality::Master.ffmpeg_crf() < CliRenderQuality::Standard.ffmpeg_crf());

    // 2. Verify 4K internal SSAA pixel memory cap prevents runaway allocation
    let rend_4k = SoftwareRenderer::new_with_quality(3840, 2160, CliRenderQuality::Master);
    assert_eq!(rend_4k.width, 3840);
    assert_eq!(rend_4k.height, 2160);
    assert_eq!(rend_4k.rgb_buffer.len(), 3840 * 2160 * 3);
    assert_eq!(
        rend_4k.ssaa_factor, 2,
        "4K Master should auto-cap SSAA to 2x (33.2M internal px <= 36M limit)"
    );

    // 3. Render both Particles & SurfaceMesh through Studio (2x2 SSAA) and Master (4x4 SSAA)
    let mut solver = SphSolver::new(
        Aabb::new(Vec3::new(-0.6, 0.0, -0.6), Vec3::new(0.6, 1.0, 0.6)),
        SolverConfig::default(),
    );
    solver.apply_quality(SimulationQuality::Draft);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DropletSplash);
    solver.step(1.0 / 60.0);

    let cam = Camera::default();
    let mut cfg = SurfaceMeshConfig::default();
    FluidMaterialPreset::ClearWater.apply(&mut cfg);
    CliRenderQuality::Studio.apply_to_mesh_config(&mut cfg, false, false);
    let mesh = SurfaceMesh::from_particles(
        &solver.particles,
        solver.config.particle_radius,
        solver.config.smoothing_length,
        &cfg,
    );
    assert!(!mesh.is_empty());

    for &rq in &[CliRenderQuality::Studio, CliRenderQuality::Master] {
        let mut rend = SoftwareRenderer::new_with_quality(160, 120, rq);
        assert_eq!(rend.ssaa_factor, rq.ssaa_factor());

        rend.render_frame(
            &solver.particles,
            solver.config.particle_radius,
            &solver.chamber,
            solver.bounds,
            &cam,
            ColorMode::Velocity,
            ColorMapType::Ocean,
            solver.config.rest_density,
        );
        assert_eq!(rend.rgb_buffer.len(), 160 * 120 * 3);
        assert!(rend.rgb_buffer.iter().any(|&v| v > 80));

        rend.render_surface_mesh_frame(
            &mesh,
            &cfg,
            &cam,
            &solver.chamber,
            solver.bounds,
            &solver.colliders,
            0.25,
            false,
        );
        assert_eq!(rend.rgb_buffer.len(), 160 * 120 * 3);
        assert!(rend.rgb_buffer.iter().any(|&v| v > 90));
    }
}

#[test]
fn test_vertical_wall_wave_runup_drains_without_lingering_lip() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::SimulationQuality;

    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
    let mut solver = SphSolver::new(bounds, SolverConfig::default());
    solver.apply_quality(SimulationQuality::High);
    // Exercise high-substep / high-iteration regime (matching Studio/Film CLI tiers)
    solver.config.substeps = 6;
    solver.config.solver_iterations = 5;
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);

    let mut gpu = GpuSphSolver::new(solver.bounds, &solver.config, &solver.particles).ok();
    let dt = 1.0 / 60.0;
    let r = solver.config.particle_radius;
    let h = solver.config.smoothing_length;
    let x_wall_l = solver.bounds.min.x + r;

    for frame in 1..=180 {
        if let Some(ref mut g) = gpu {
            g.step(
                solver.bounds,
                &solver.chamber,
                &solver.config,
                &solver.colliders,
                None,
                0.0,
                0,
                dt,
                false,
            );
        } else {
            solver.step(dt);
        }

        if frame == 30 {
            if let Some(ref g) = gpu {
                g.read_particles(&mut solver.particles);
            }
            let max_left_initial_drop = solver
                .particles
                .iter()
                .filter(|p| p.position.x < x_wall_l + h)
                .map(|p| p.position.y)
                .fold(0.0f32, f32::max);
            assert!(
                max_left_initial_drop < 0.86,
                "Initial DamBreak release must drain smoothly from left wall without back-trench jet, got max_y={:.3}",
                max_left_initial_drop
            );
        }
    }

    if let Some(ref g) = gpu {
        g.read_particles(&mut solver.particles);
    }

    let mut wall_ys: Vec<f32> = solver
        .particles
        .iter()
        .filter(|p| p.position.x < x_wall_l + 0.8 * h)
        .map(|p| p.position.y)
        .collect();
    wall_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let l_wall_max = wall_ys.last().copied().unwrap_or(0.0);
    let l_wall_p98 = if wall_ys.is_empty() {
        0.0
    } else {
        wall_ys[wall_ys.len() * 98 / 100]
    };

    let mut near_ys: Vec<f32> = solver
        .particles
        .iter()
        .filter(|p| p.position.x >= -0.68 && p.position.x <= -0.45)
        .map(|p| p.position.y)
        .collect();
    near_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let l_near_p98 = near_ys[near_ys.len() * 98 / 100];

    assert!(
        l_wall_p98 < 0.34 && l_wall_p98 <= l_near_p98 + 0.07 && l_wall_max < 0.48,
        "Left vertical wall fluid must drain cleanly without lingering meniscus: l_wall_max={:.3}, l_wall_p98={:.3}, l_near_p98={:.3}",
        l_wall_max,
        l_wall_p98,
        l_near_p98
    );
}

#[test]
fn test_cylinder_top_cap_collision_cpu_and_gpu() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::{Collider, SimulationQuality};

    let col = Collider::cylinder(Vec3::new(0.0, 0.25, 0.0), 0.22, 0.30);
    let r = 0.015f32;
    let top_y = 0.25 + 0.15; // 0.40

    // 1. Direct CPU penetration just below top cap near cylinder axis must push +Y, NOT sideways to r=0.22
    let mut pos = Vec3::new(0.02, top_y - 0.005, -0.01);
    let mut vel = Vec3::new(0.0, -1.2, 0.0);
    assert!(col.resolve(&mut pos, &mut vel, r));
    assert!(
        pos.y >= top_y + r - 1e-4,
        "Top-cap hit should project upward onto cylinder top cap, got pos={:?}",
        pos
    );
    assert!(
        pos.x.hypot(pos.z) < 0.05,
        "Top-cap hit must NOT teleport particle sideways to cylinder rim, got pos={:?}",
        pos
    );
    assert!(vel.y >= -1e-4, "Downward normal velocity must be removed on top cap, got vel={:?}", vel);

    // 2. Full solver step (CPU & GPU if available) with a particle resting over cylinder top center
    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.8), Vec3::new(0.8, 1.2, 0.8));
    let mut solver = SphSolver::new(bounds, SolverConfig::default());
    solver.apply_quality(SimulationQuality::Draft);
    let pr = solver.config.particle_radius;
    solver.colliders.push(col.clone());
    solver.particles.push(Particle::new(
        Vec3::new(0.0, top_y + pr + 0.02, 0.0),
        Vec3::new(0.0, -0.8, 0.0),
        1.0,
        0,
    ));

    let mut gpu = GpuSphSolver::new(solver.bounds, &solver.config, &solver.particles).ok();
    for _ in 0..20 {
        solver.step(1.0 / 60.0);
        if let Some(ref mut g) = gpu {
            g.step(
                solver.bounds,
                &solver.chamber,
                &solver.config,
                &solver.colliders,
                None,
                0.0,
                0,
                1.0 / 60.0,
                false,
            );
        }
    }

    let cpu_p = solver.particles[0];
    assert!(
        cpu_p.position.y >= top_y + pr - 2e-3 && cpu_p.position.x.hypot(cpu_p.position.z) < 0.08,
        "CPU particle should rest on cylinder top cap, got {:?}",
        cpu_p.position
    );

    if let Some(ref g) = gpu {
        let mut gpu_particles = Vec::new();
        g.read_particles(&mut gpu_particles);
        let gp = gpu_particles[0];
        assert!(
            gp.position.y >= top_y + pr - 2e-3 && gp.position.x.hypot(gp.position.z) < 0.08,
            "GPU particle should rest on cylinder top cap, got {:?}",
            gp.position
        );
    }
}

#[test]
fn test_2d_splash_and_zero_g_non_overlapping_discs() {
    use liquid_rust::sph::SimulationQuality;

    for &preset in &[ScenePreset::DropletSplash, ScenePreset::ZeroGravityBlobs] {
        let bounds = Aabb::new(Vec3::new(-0.9, 0.0, -0.05), Vec3::new(0.9, 1.4, 0.05));
        let mut solver = SphSolver::new(bounds, SolverConfig::default());
        solver.set_2d(true);
        solver.apply_quality(SimulationQuality::Draft);
        SceneGenerator::setup_scene(&mut solver, preset);

        assert!(!solver.particles.is_empty());
        let min_allowed_dist = solver.config.spacing * 0.45;
        for i in 0..solver.particles.len() {
            let pi = solver.particles[i].position;
            assert!(pi.z.abs() < 1e-6, "2D particle must lie on z=0 plane, got {:?}", pi);
            for j in (i + 1)..solver.particles.len() {
                let pj = solver.particles[j].position;
                let d = pi.distance(pj);
                assert!(
                    d >= min_allowed_dist,
                    "2D {:?} spawned overlapping particles {} and {} at dist={:.6} (spacing={:.4})",
                    preset,
                    i,
                    j,
                    d,
                    solver.config.spacing
                );
            }
        }

        for _ in 0..30 {
            solver.step(1.0 / 60.0);
        }
        for p in &solver.particles {
            assert!(p.position.is_finite() && p.velocity.is_finite());
            assert!(
                p.velocity.length() < solver.config.max_velocity + 0.1,
                "2D {:?} velocity should remain bounded, got {:?}",
                preset,
                p.velocity
            );
        }
    }

    // Also verify 3D ZeroGravityBlobs fits cleanly above floor y >= 0.0
    let mut solver_3d = SphSolver::new(
        Aabb::new(Vec3::new(-0.8, 0.0, -0.8), Vec3::new(0.8, 1.4, 0.8)),
        SolverConfig::default(),
    );
    solver_3d.apply_quality(SimulationQuality::Draft);
    SceneGenerator::setup_scene(&mut solver_3d, ScenePreset::ZeroGravityBlobs);
    for p in &solver_3d.particles {
        assert!(
            p.position.y >= solver_3d.bounds.min.y,
            "3D ZeroGravityBlobs particles must spawn above chamber floor y>=0, got {:?}",
            p.position
        );
    }
}

#[test]
fn test_fluid_physics_presets_air_drag_and_wave_maker() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::{FluidPhysicsPreset, SimulationQuality};

    // 1. Honey must damp motion significantly faster than Water
    let run_preset_speed = |preset: FluidPhysicsPreset| -> f32 {
        let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.05), Vec3::new(0.8, 1.2, 0.05));
        let mut solver = SphSolver::new(bounds, SolverConfig::default());
        solver.set_2d(true);
        solver.apply_quality(SimulationQuality::Draft);
        SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);
        preset.apply(&mut solver.config);
        for _ in 0..45 {
            solver.step(1.0 / 60.0);
        }
        solver.particles.iter().map(|p| p.velocity.length()).sum::<f32>()
            / solver.particles.len() as f32
    };
    let speed_water = run_preset_speed(FluidPhysicsPreset::Water);
    let speed_honey = run_preset_speed(FluidPhysicsPreset::Honey);
    assert!(
        speed_honey < speed_water * 0.85,
        "Honey ({:.3}) should flow noticeably slower than Water ({:.3})",
        speed_honey,
        speed_water
    );

    // 2. Airborne droplet drag slows isolated fast spray compared to zero drag
    let run_droplet_vx = |drag: f32| -> f32 {
        let bounds = Aabb::new(Vec3::new(-2.0, 0.0, -0.5), Vec3::new(2.0, 2.0, 0.5));
        let mut cfg = SolverConfig::default();
        cfg.air_drag = drag;
        let mut solver = SphSolver::new(bounds, cfg);
        solver.particles.push(Particle::new(
            Vec3::new(-1.0, 1.2, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            1.0,
            0,
        ));
        for _ in 0..20 {
            solver.step(1.0 / 60.0);
        }
        solver.particles[0].velocity.x
    };
    let vx_no_drag = run_droplet_vx(0.0);
    let vx_with_drag = run_droplet_vx(1.2);
    assert!(
        vx_with_drag < vx_no_drag - 0.15,
        "Air drag should decelerate isolated airborne spray: with_drag={:.3} vs no_drag={:.3}",
        vx_with_drag,
        vx_no_drag
    );

    // 3. Hydraulic wave maker drives sloshing stably on CPU & GPU
    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.4), Vec3::new(0.8, 1.2, 0.4));
    let mut solver = SphSolver::new(bounds, SolverConfig::default());
    solver.apply_quality(SimulationQuality::Draft);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DropletSplash);
    solver.config.wave_amplitude = 0.20;
    solver.config.wave_frequency = 1.1;

    let mut gpu = GpuSphSolver::new(solver.bounds, &solver.config, &solver.particles).ok();
    for _ in 0..35 {
        solver.step(1.0 / 60.0);
        if let Some(ref mut g) = gpu {
            g.step(
                solver.bounds,
                &solver.chamber,
                &solver.config,
                &solver.colliders,
                None,
                0.0,
                0,
                1.0 / 60.0,
                false,
            );
        }
    }
    let avg_abs_vx = solver.particles.iter().map(|p| p.velocity.x.abs()).sum::<f32>()
        / solver.particles.len() as f32;
    assert!(avg_abs_vx > 0.01, "Wave maker should induce horizontal sloshing, got {}", avg_abs_vx);
    assert!(solver.particles.iter().all(|p| p.position.is_finite() && p.velocity.is_finite()));
    if let Some(ref g) = gpu {
        let mut gp = Vec::new();
        g.read_particles(&mut gp);
        let gpu_abs_vx = gp.iter().map(|p| p.velocity.x.abs()).sum::<f32>() / gp.len() as f32;
        assert!(gpu_abs_vx > 0.01, "GPU wave maker should induce sloshing, got {}", gpu_abs_vx);
        assert!(gp.iter().all(|p| p.position.is_finite() && p.velocity.is_finite()));
    }
}

#[test]
fn test_png_exporter_and_studio_particle_renderer() {
    use liquid_rust::export::{PngExporter, SoftwareRenderer};
    use liquid_rust::render::{
        Camera, CliRenderQuality, ColorMapType, ColorMode, EnvironmentPreset, SurfaceMesh,
        SurfaceMeshConfig,
    };
    use liquid_rust::sph::{Collider, SimulationQuality};

    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.8), Vec3::new(0.8, 1.2, 0.8));
    let mut solver = SphSolver::new(bounds, SolverConfig::default());
    solver.apply_quality(SimulationQuality::Draft);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::ObstacleDrop);
    solver.colliders.push(Collider::cylinder(Vec3::new(0.25, 0.2, 0.0), 0.12, 0.35));
    for _ in 0..10 {
        solver.step(1.0 / 60.0);
    }

    let cam = Camera::default();
    let mut mesh_cfg = SurfaceMeshConfig::default();
    mesh_cfg.env_preset = EnvironmentPreset::GoldenSunset;
    mesh_cfg.exposure = 1.15;
    mesh_cfg.show_grid = true;
    mesh_cfg.show_wireframe = true;

    let mut rend = SoftwareRenderer::new_with_quality(192, 108, CliRenderQuality::Studio);
    rend.env_preset = mesh_cfg.env_preset;
    rend.exposure = mesh_cfg.exposure;
    rend.show_grid = mesh_cfg.show_grid;
    rend.show_wireframe = mesh_cfg.show_wireframe;
    rend.particle_foam = true;
    rend.lit_spheres = true;
    rend.render_particles_studio(
        &solver.particles,
        solver.config.particle_radius,
        &solver.chamber,
        solver.bounds,
        &solver.colliders,
        &cam,
        ColorMode::Foam,
        ColorMapType::Inferno,
        solver.config.rest_density,
        false,
    );
    assert_eq!(rend.rgb_buffer.len(), 192 * 108 * 3);
    assert!(rend.rgb_buffer.iter().any(|&v| v > 120));

    let tmp_png = std::env::temp_dir().join("liquid_rust_studio_test.png");
    PngExporter::save_rgb(&tmp_png, rend.width, rend.height, &rend.rgb_buffer)
        .expect("Studio PNG export must succeed");
    let bytes = std::fs::read(&tmp_png).expect("Saved PNG must be readable");
    let _ = std::fs::remove_file(&tmp_png);

    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n", "Valid PNG signature required");
    assert_eq!(&bytes[12..16], b"IHDR");
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    assert_eq!((w, h), (192, 108));
    assert!(bytes.windows(4).any(|w| w == b"IDAT"), "PNG must contain IDAT chunk");
    assert!(bytes.windows(4).any(|w| w == b"IEND"), "PNG must contain IEND chunk");

    // Also exercise Vorticity + Ice colormap and SurfaceMesh with ArcticLab environment
    mesh_cfg.env_preset = EnvironmentPreset::ArcticLab;
    rend.env_preset = EnvironmentPreset::ArcticLab;
    rend.particle_foam = false;
    rend.lit_spheres = false;
    rend.render_particles_studio(
        &solver.particles,
        solver.config.particle_radius,
        &solver.chamber,
        solver.bounds,
        &solver.colliders,
        &cam,
        ColorMode::Vorticity,
        ColorMapType::Ice,
        solver.config.rest_density,
        false,
    );
    let mesh = SurfaceMesh::from_particles(
        &solver.particles,
        solver.config.particle_radius,
        solver.config.smoothing_length,
        &mesh_cfg,
    );
    rend.render_surface_mesh_frame(
        &mesh,
        &mesh_cfg,
        &cam,
        &solver.chamber,
        solver.bounds,
        &solver.colliders,
        0.4,
        false,
    );
    assert!(rend.rgb_buffer.iter().any(|&v| v > 140));
}

#[test]
fn test_scene_json_roundtrip_and_cli_command_flags() {
    use liquid_rust::app::{CliCommandMode, LiquidApp};
    use liquid_rust::render::EnvironmentPreset;
    use liquid_rust::scene_io::SceneConfigFile;
    use liquid_rust::sph::{ChamberShape, Collider, CubeFace, Drain, Faucet, FluidPhysicsPreset};

    let mut app = LiquidApp::new_headless();
    app.fluid_preset = FluidPhysicsPreset::Honey;
    FluidPhysicsPreset::Honey.apply(&mut app.solver.config);
    app.solver.config.wave_amplitude = 0.16;
    app.solver.config.wave_frequency = 1.25;
    app.solver.chamber.shape = ChamberShape::Bathtub;
    app.solver.chamber.base_size = Vec3::new(2.2, 1.4, 1.6);
    app.solver.chamber.custom_size = true;
    app.solver.bounds = app.solver.chamber.compute_bounds();

    let mut col = Collider::sphere(Vec3::new(0.15, 0.35, -0.1), 0.22);
    col.restitution = 0.45;
    col.friction = 0.28;
    app.solver.colliders.push(col);
    app.solver.faucets.push(Faucet::new_circle(
        "Tap #1",
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.5, 0.0),
        0.09,
        650.0,
    ));
    let drain = Drain::new("Drain #1", CubeFace::Right, Vec3::new(0.0, -0.25, 0.0), 0.18);
    app.solver.drains.push(drain);

    app.mesh_config.env_preset = EnvironmentPreset::ArcticLab;
    app.mesh_config.exposure = 1.25;
    app.camera_orbit_speed_deg = 12.5;
    app.cli_mode = CliCommandMode::Mp4Video;

    let cmd = app.generate_cli_command();
    assert!(cmd.contains("--fluid-preset honey"), "Expected --fluid-preset honey in {}", cmd);
    assert!(cmd.contains("--wave-amp 0.160"), "Expected --wave-amp in {}", cmd);
    assert!(cmd.contains("--wave-freq 1.25"), "Expected --wave-freq in {}", cmd);
    assert!(cmd.contains("--chamber-shape bathtub"), "Expected --chamber-shape bathtub in {}", cmd);
    assert!(cmd.contains("--chamber-size 2.20,1.40,1.60"), "Expected --chamber-size in {}", cmd);
    assert!(cmd.contains("--env-preset arctic"), "Expected --env-preset arctic in {}", cmd);
    assert!(cmd.contains("--exposure 1.25"), "Expected --exposure 1.25 in {}", cmd);
    assert!(cmd.contains("--camera-orbit 12.5"), "Expected --camera-orbit 12.5 in {}", cmd);
    assert!(cmd.contains("--faucet \"stream:"), "Expected --faucet stream: in {}", cmd);
    assert!(cmd.contains("--drain "), "Expected --drain in {}", cmd);
    assert!(cmd.contains(":0.45:0.28"), "Expected obstacle bounce:slip suffix in {}", cmd);

    let tmp_json = std::env::temp_dir().join("liquid_rust_scene_roundtrip.json");
    app.to_scene_config_file()
        .save_to_path(&tmp_json)
        .expect("Scene JSON save must succeed");
    let loaded = SceneConfigFile::load_from_path(&tmp_json).expect("Scene JSON load must succeed");
    let _ = std::fs::remove_file(&tmp_json);

    let mut app2 = LiquidApp::new_headless();
    app2.apply_scene_config_file(&loaded);
    assert_eq!(app2.fluid_preset, FluidPhysicsPreset::Honey);
    assert!((app2.solver.config.wave_amplitude - 0.16).abs() < 1e-5);
    assert!((app2.solver.config.wave_frequency - 1.25).abs() < 1e-5);
    assert_eq!(app2.solver.chamber.shape, ChamberShape::Bathtub);
    assert!(app2.solver.chamber.custom_size);
    assert_eq!(app2.solver.colliders.len(), 1);
    assert!((app2.solver.colliders[0].restitution - 0.45).abs() < 1e-5);
    assert!((app2.solver.colliders[0].friction - 0.28).abs() < 1e-5);
    assert_eq!(app2.solver.faucets.len(), 1);
    assert_eq!(app2.solver.drains.len(), 1);
    assert_eq!(app2.mesh_config.env_preset, EnvironmentPreset::ArcticLab);
    assert!((app2.mesh_config.exposure - 1.25).abs() < 1e-5);
    assert!((app2.camera_orbit_speed_deg - 12.5).abs() < 1e-5);
}

#[test]
fn test_gpu_surface_field_parity_with_cpu_marching_cubes() {
    use liquid_rust::gpu::{GpuSphSolver, GpuSurfaceField};
    use liquid_rust::render::{SurfaceMesh, SurfaceMeshConfig};
    use liquid_rust::sph::{SceneGenerator, ScenePreset, SimulationQuality, SolverConfig, SphSolver};
    use std::sync::Arc;

    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.5), Vec3::new(0.8, 1.0, 0.5));
    let mut solver = SphSolver::new(bounds, SolverConfig::default());
    solver.apply_quality(SimulationQuality::Medium);
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);

    let Ok(mut gpu) = GpuSphSolver::new(solver.bounds, &solver.config, &solver.particles) else {
        return;
    };
    for _ in 0..8 {
        gpu.step(
            solver.bounds,
            &solver.chamber,
            &solver.config,
            &solver.colliders,
            solver.mouse_ray,
            solver.mouse_force_strength,
            solver.mouse_force_mode,
            1.0 / 60.0,
            false,
        );
    }
    gpu.read_particles(&mut solver.particles);

    let mut gpu_field = GpuSurfaceField::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue))
        .expect("GpuSurfaceField creation");
    let cfg = SurfaceMeshConfig::default();

    let cpu_mesh = SurfaceMesh::from_particles(
        &solver.particles,
        solver.config.particle_radius,
        solver.config.smoothing_length,
        &cfg,
    );
    let gpu_mesh = SurfaceMesh::from_particles_with_gpu(
        &solver.particles,
        solver.config.particle_radius,
        solver.config.smoothing_length,
        &cfg,
        &mut gpu_field,
    );

    assert!(!cpu_mesh.is_empty());
    assert_eq!(cpu_mesh.vertices.len(), gpu_mesh.vertices.len());
    assert_eq!(cpu_mesh.indices.len(), gpu_mesh.indices.len());

    // Bounding box parity check across vertex positions
    let mut max_pos_diff = 0.0f32;
    for (cv, gv) in cpu_mesh.vertices.iter().zip(gpu_mesh.vertices.iter()) {
        let d = (Vec3::from_array(cv.position) - Vec3::from_array(gv.position)).length();
        max_pos_diff = max_pos_diff.max(d);
    }
    assert!(
        max_pos_diff < 1e-4,
        "GPU surface field vertex positions should match CPU within 1e-4, got {}",
        max_pos_diff
    );
}

#[test]
fn test_gpu_renderer_particles_and_mesh_visual_parity() {
    use liquid_rust::export::SoftwareRenderer;
    use liquid_rust::gpu::{GpuRenderer, GpuSphSolver};
    use liquid_rust::render::{
        Camera, CliRenderQuality, ColorMapType, ColorMode, FluidMaterialPreset, SurfaceMesh,
        SurfaceMeshConfig,
    };
    use liquid_rust::sph::{
        Collider, SceneGenerator, ScenePreset, SimulationQuality, SolverConfig, SphSolver,
    };
    use std::sync::Arc;

    let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.5), Vec3::new(0.8, 1.0, 0.5));
    let mut solver = SphSolver::new(bounds, SolverConfig::default());
    solver.apply_quality(SimulationQuality::Medium);
    solver.colliders.push(Collider::sphere(Vec3::new(0.15, 0.25, 0.0), 0.14));
    SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);

    let Ok(mut gpu) = GpuSphSolver::new(solver.bounds, &solver.config, &solver.particles) else {
        return;
    };
    for _ in 0..5 {
        gpu.step(
            solver.bounds,
            &solver.chamber,
            &solver.config,
            &solver.colliders,
            solver.mouse_ray,
            solver.mouse_force_strength,
            solver.mouse_force_mode,
            1.0 / 60.0,
            false,
        );
    }
    gpu.read_particles(&mut solver.particles);

    let (w, h) = (320usize, 180usize);
    let cam = Camera::default();
    let mut gpu_rend = GpuRenderer::new_with_device(
        Arc::clone(&gpu.device),
        Arc::clone(&gpu.queue),
        gpu.device_name.clone(),
        w,
        h,
        CliRenderQuality::Studio,
    )
    .expect("GpuRenderer init");
    let mut cpu_rend = SoftwareRenderer::new_with_quality(w, h, CliRenderQuality::Studio);

    // 1. Direct GPU-buffer particle rendering (zero CPU readback path)
    let p_buf = gpu.particle_buffer().expect("particle buffer");
    gpu_rend.render_particles_from_gpu_buffer(
        p_buf,
        gpu.particle_count,
        solver.config.particle_radius,
        &solver.chamber,
        solver.bounds,
        &solver.colliders,
        &cam,
        ColorMode::Velocity,
        ColorMapType::Ocean,
        solver.config.rest_density,
        false,
    );
    cpu_rend.render_particles_studio(
        &solver.particles,
        solver.config.particle_radius,
        &solver.chamber,
        solver.bounds,
        &solver.colliders,
        &cam,
        ColorMode::Velocity,
        ColorMapType::Ocean,
        solver.config.rest_density,
        false,
    );

    let mean_luma = |buf: &[u8]| -> f64 {
        buf.iter().map(|&b| b as f64).sum::<f64>() / buf.len().max(1) as f64
    };
    let l_gpu_p = mean_luma(&gpu_rend.rgb_buffer);
    let l_cpu_p = mean_luma(&cpu_rend.rgb_buffer);
    assert!(
        (l_gpu_p - l_cpu_p).abs() < 12.0,
        "GPU vs CPU particle luma mismatch: gpu={:.2} cpu={:.2}",
        l_gpu_p,
        l_cpu_p
    );

    // 2. SurfaceMesh ClearWater + LiquidChrome (Mercury)
    let mut mesh_cfg = SurfaceMeshConfig::default();
    let mesh = SurfaceMesh::from_particles(
        &solver.particles,
        solver.config.particle_radius,
        solver.config.smoothing_length,
        &mesh_cfg,
    );
    for preset in [FluidMaterialPreset::ClearWater, FluidMaterialPreset::LiquidChrome] {
        preset.apply(&mut mesh_cfg);
        gpu_rend.render_surface_mesh_frame(
            &mesh,
            &mesh_cfg,
            &cam,
            &solver.chamber,
            solver.bounds,
            &solver.colliders,
            0.25,
            false,
        );
        cpu_rend.render_surface_mesh_frame(
            &mesh,
            &mesh_cfg,
            &cam,
            &solver.chamber,
            solver.bounds,
            &solver.colliders,
            0.25,
            false,
        );
        let lg = mean_luma(&gpu_rend.rgb_buffer);
        let lc = mean_luma(&cpu_rend.rgb_buffer);
        assert!(
            (lg - lc).abs() < 15.0,
            "{:?} GPU vs CPU luma mismatch: gpu={:.2} cpu={:.2}",
            preset,
            lg,
            lc
        );
    }
}

#[test]
fn test_gui_render_tab_still_and_animation_end_result() {
    use liquid_rust::{
        AnimationStartSource, CentralView, CliRenderQuality, InspectorTab, LiquidApp, RenderMode,
        SimulationQuality, StudioRenderKind,
    };
    use std::time::{Duration, Instant};

    let mut app = LiquidApp::new_headless();
    app.set_quality(SimulationQuality::Draft);
    app.active_tab = InspectorTab::Render;
    app.central_view = CentralView::RenderResult;
    app.use_gpu = true;
    app.render_mode = RenderMode::SurfaceMesh;
    app.video_width = 360;
    app.video_height = 240;
    app.gui_render_quality = CliRenderQuality::Studio;

    // 1. Trigger Studio Still Frame render and verify full-res end result
    app.start_gui_still_render(None, false);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let running = app.bg_render.lock().unwrap().is_running;
        if !running {
            break;
        }
        assert!(Instant::now() < deadline, "Still render timed out");
        std::thread::sleep(Duration::from_millis(15));
    }

    {
        let g = app.bg_render.lock().unwrap();
        assert_eq!(g.kind, StudioRenderKind::StillFrame);
        assert_eq!(g.completed_frames, 1);
        let still = g.full_res_frame.as_ref().expect("full_res_frame must exist");
        assert_eq!(still.width, 360);
        assert_eq!(still.height, 240);
        assert_eq!(still.rgb.len(), 360 * 240 * 3);
        let luma: f64 =
            still.rgb.iter().map(|&b| b as f64).sum::<f64>() / still.rgb.len() as f64;
        assert!(luma > 5.0, "Rendered studio frame should not be black, luma={}", luma);
        assert!(!g.backend_label.is_empty());
    }

    // 2. Save rendered result PNG to temporary folder and verify valid PNG output
    let tmp_dir = std::env::temp_dir().join("liquid_rust_gui_render_tab_test");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    app.export_path = tmp_dir.to_string_lossy().into_owned();
    app.save_rendered_result_png();
    let saved_png = tmp_dir.join("render_result_00000.png");
    let png_bytes = std::fs::read(&saved_png).expect("saved PNG must exist");
    assert_eq!(&png_bytes[0..8], b"\x89PNG\r\n\x1a\n");
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // 3. Trigger in-GUI Video Animation render (CurrentState, 5 frames) and verify progressive sequence
    app.render_kind = StudioRenderKind::Animation;
    app.render_anim_start = AnimationStartSource::CurrentState;
    app.render_save_mp4 = false;
    app.cli_fps = 20;
    app.cli_duration = 0.25; // 5 frames
    app.start_gui_mp4_render_with_ctx(None);

    let deadline2 = Instant::now() + Duration::from_secs(30);
    loop {
        let running = app.bg_render.lock().unwrap().is_running;
        if !running {
            break;
        }
        assert!(Instant::now() < deadline2, "Animation render timed out");
        std::thread::sleep(Duration::from_millis(15));
    }

    {
        let g = app.bg_render.lock().unwrap();
        assert_eq!(g.kind, StudioRenderKind::Animation);
        assert_eq!(g.completed_frames, 5);
        assert_eq!(g.anim_frames.len(), 5);
        assert!(g.just_finished_anim);
        assert!(g.full_res_frame.is_some());
        assert_ne!(
            &*g.anim_frames[0].rgb,
            &*g.anim_frames[4].rgb,
            "Animation frames should evolve across physics timesteps"
        );
    }
}

#[test]
fn test_gui_cli_grade_offline_quality_presets_and_timed_photo() {
    use liquid_rust::{
        CliRenderQuality, GuidedRenderPreset, LiquidApp, RenderAppearance, RenderMode,
        SimulationQuality, StudioRenderKind,
    };
    use std::time::{Duration, Instant};

    let mut app = LiquidApp::new_headless();
    // Live 3D editor stays lightweight at Draft + Particles so interactive framing is effortless
    app.set_quality(SimulationQuality::Draft);
    app.render_mode = RenderMode::Particles;
    let live_draft_pts = app.solver.particle_count();
    assert!(live_draft_pts > 0);

    // 1. Verify one-click GuidedRenderPreset configurations & CLI command equivalence
    app.apply_guided_render_preset(GuidedRenderPreset::RecommendedHd);
    assert_eq!(app.render_kind, StudioRenderKind::Animation);
    assert_eq!(app.render_appearance, RenderAppearance::SmoothLiquid);
    assert_eq!(app.effective_render_mode(), RenderMode::SurfaceMesh);
    assert_eq!(app.render_sim_quality, Some(SimulationQuality::Ultra));
    assert_eq!(app.video_width, 1920);
    assert_eq!(app.video_height, 1080);
    let cli_hd = app.generate_render_tab_cli_command();
    assert!(cli_hd.contains("--export-mp4"));
    assert!(cli_hd.contains("--quality ultra"));
    assert!(cli_hd.contains("--surface-mesh"));
    assert!(cli_hd.contains("--render-quality studio"));

    app.apply_guided_render_preset(GuidedRenderPreset::StudioPhoto);
    assert_eq!(app.render_kind, StudioRenderKind::StillFrame);
    assert_eq!(app.render_sim_quality, Some(SimulationQuality::Cinematic));
    assert!(app.render_still_simulate);
    let cli_photo = app.generate_render_tab_cli_command();
    assert!(cli_photo.contains("--export-png"));
    assert!(cli_photo.contains("--quality cinematic"));
    assert!(cli_photo.contains("--surface-mesh"));

    // 2. Run offline GUI video render at HIGHER particle detail (Medium > Draft) + SmoothLiquid + OBJ/PLY export
    // while keeping live editor untouched at Draft + Particles!
    let tmp_dir = std::env::temp_dir().join("liquid_rust_gui_cli_offline_test");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    app.export_path = tmp_dir.to_string_lossy().into_owned();
    app.render_kind = StudioRenderKind::Animation;
    app.render_appearance = RenderAppearance::SmoothLiquid;
    app.render_sim_quality = Some(SimulationQuality::Medium);
    app.gui_render_quality = CliRenderQuality::Standard;
    app.video_width = 320;
    app.video_height = 240;
    app.cli_fps = 20;
    app.cli_duration = 0.20; // 4 frames
    app.render_save_mp4 = false;
    app.render_also_export_obj = true;
    app.render_also_export_ply = true;
    app.use_gpu = true;

    app.start_gui_mp4_render_with_ctx(None);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if !app.bg_render.lock().unwrap().is_running {
            break;
        }
        assert!(Instant::now() < deadline, "Offline higher-quality GUI video render timed out");
        std::thread::sleep(Duration::from_millis(15));
    }

    // Live 3D editor remained completely untouched at Draft + Particles!
    assert_eq!(app.solver.particle_count(), live_draft_pts);
    assert_eq!(app.render_mode, RenderMode::Particles);

    {
        let g = app.bg_render.lock().unwrap();
        assert_eq!(g.completed_frames, 4);
        assert_eq!(g.anim_frames.len(), 4);
        assert!(
            g.sim_particles_count > live_draft_pts,
            "Offline render should spawn higher Medium particle count ({} vs live Draft {})",
            g.sim_particles_count,
            live_draft_pts
        );
    }
    assert!(
        tmp_dir.join("fluid_surface_00000.obj").exists(),
        "fluid_surface_00000.obj must be exported"
    );
    assert!(
        tmp_dir.join("fluid_00000.ply").exists(),
        "fluid_00000.ply must be exported"
    );

    // 3. Run Timed Still Photo render (simulating offline from t=0 -> 0.15s at Medium density)
    app.render_also_export_obj = false;
    app.render_also_export_ply = false;
    app.render_kind = StudioRenderKind::StillFrame;
    app.render_still_simulate = true;
    app.render_still_time = 0.15; // 3 steps @ 20 FPS + 1 render step = 4 total steps
    app.start_gui_still_render(None, false);

    let deadline2 = Instant::now() + Duration::from_secs(30);
    loop {
        if !app.bg_render.lock().unwrap().is_running {
            break;
        }
        assert!(Instant::now() < deadline2, "Timed offline photo render timed out");
        std::thread::sleep(Duration::from_millis(15));
    }

    {
        let g = app.bg_render.lock().unwrap();
        assert_eq!(g.kind, StudioRenderKind::StillFrame);
        assert_eq!(g.total_frames, 4);
        assert_eq!(g.completed_frames, 4);
        assert!(g.sim_particles_count > live_draft_pts);
        let still = g.full_res_frame.as_ref().expect("Timed photo must exist");
        let luma: f64 = still.rgb.iter().map(|&b| b as f64).sum::<f64>() / still.rgb.len() as f64;
        assert!(luma > 5.0, "Timed offline photo should not be black, luma={}", luma);
    }

    let custom_png = tmp_dir.join("custom_save_as_photo.png");
    app.save_rendered_result_png_to_path(&custom_png);
    assert!(custom_png.exists(), "Save Image As path must exist");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}


