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
    ] {
        let c0 = ColorMap::sample(cmap, 0.0);
        let c2 = ColorMap::sample(cmap, 1.0);
        assert_ne!(c0, c2);
    }
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
