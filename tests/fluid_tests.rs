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
    SimulationQuality::Ultra.apply(&mut config);
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
fn test_gpu_sph_solver() {
    use liquid_rust::gpu::GpuSphSolver;

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.6), Vec3::new(1.0, 2.0, 0.6));
    let mut config = SolverConfig::default();
    config.spacing = 0.06;
    config.particle_radius = 0.03;
    config.smoothing_length = 0.09;
    config.rest_density = compute_lattice_rest_density(config.smoothing_length, config.spacing);

    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DamBreak);

    let mut gpu_solver = match GpuSphSolver::new(bounds, &config, &cpu_solver.particles) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping GPU test (no Vulkan adapter available): {}", e);
            return;
        }
    };

    println!("Testing GPU SPH Solver on: {}", gpu_solver.device_name);

    for _ in 0..10 {
        gpu_solver.step(bounds, &config, &[], None, 0.0, 0, 1.0 / 60.0);
    }

    let mut readback_particles = Vec::new();
    gpu_solver.read_particles(&mut readback_particles);

    assert_eq!(readback_particles.len(), cpu_solver.particles.len());
    for p in &readback_particles {
        assert!(!p.position.is_nan(), "GPU position should not be NaN");
        assert!(!p.velocity.is_nan(), "GPU velocity should not be NaN");
        assert!(p.speed() < 20.0, "GPU fluid should be stable and bounded");
        assert!(bounds.contains(p.position), "GPU particle inside bounds");
    }
}

#[test]
fn test_gpu_slow_motion_stability() {
    use liquid_rust::gpu::GpuSphSolver;
    use liquid_rust::sph::SimulationQuality;

    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.45), Vec3::new(1.0, 1.2, 0.45));
    let mut config = SolverConfig::default();
    SimulationQuality::Ultra.apply(&mut config);

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
        gpu_solver.step(bounds, &config, &[], None, 0.0, 0, slow_dt);
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
    SimulationQuality::Ultra.apply(&mut config);

    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DamBreak);

    let mut gpu_solver = match GpuSphSolver::new(bounds, &config, &cpu_solver.particles) {
        Ok(s) => s,
        Err(_) => return,
    };

    for _ in 0..10 {
        gpu_solver.step(bounds, &config, &[], None, 0.0, 0, 1.0 / 60.0);
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
    SimulationQuality::Extreme.apply(&mut config);

    let mut cpu_solver = SphSolver::new(bounds, config.clone());
    SceneGenerator::setup_scene(&mut cpu_solver, ScenePreset::DamBreak);

    // Extreme quality must spawn over 1,000,000 particles
    assert!(
        cpu_solver.particles.len() >= 1_000_000,
        "Extreme quality must spawn at least 1,000,000 particles, got {}",
        cpu_solver.particles.len()
    );
}
