use glam::Vec3;
use liquid_rust::export::PlyExporter;
use liquid_rust::math::{Aabb, Ray};
use liquid_rust::render::{ColorMap, ColorMapType};
use liquid_rust::sph::collider::Collider;
use liquid_rust::sph::kernel::SphKernels;
use liquid_rust::sph::particle::Particle;
use liquid_rust::sph::scene::{SceneGenerator, ScenePreset};
use liquid_rust::sph::solver::{SolverConfig, SphSolver};
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

    // Within support radius
    let w_center = kernels.poly6(0.0);
    assert!(w_center > 0.0);

    let w_mid = kernels.poly6(0.05 * 0.05);
    assert!(w_mid > 0.0);
    assert!(w_center > w_mid);

    // Outside support radius
    let w_outside = kernels.poly6(0.15 * 0.15);
    assert_eq!(w_outside, 0.0);

    // Spiky gradient
    let grad_inside = kernels.spiky_gradient(Vec3::new(0.05, 0.0, 0.0), 0.05);
    assert!(grad_inside.x < 0.0); // Points away from neighbor

    let grad_outside = kernels.spiky_gradient(Vec3::new(0.15, 0.0, 0.0), 0.15);
    assert_eq!(grad_outside, Vec3::ZERO);

    // Viscosity laplacian
    let lap_inside = kernels.viscosity_laplacian(0.05);
    assert!(lap_inside > 0.0);
    let lap_outside = kernels.viscosity_laplacian(0.15);
    assert_eq!(lap_outside, 0.0);
}

#[test]
fn test_spatial_grid_neighbor_search() {
    let cell_size = 0.1;
    let mut grid = SpatialHashGrid::new(cell_size, 16);

    let positions = vec![
        Vec3::new(0.0, 0.0, 0.0),      // 0
        Vec3::new(0.05, 0.0, 0.0),     // 1: within cell_size of 0
        Vec3::new(0.08, 0.05, 0.0),    // 2: within cell_size of 1
        Vec3::new(2.0, 2.0, 2.0),      // 3: far away
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
fn test_sphere_collider_response() {
    let collider = Collider::sphere(Vec3::new(0.0, 0.0, 0.0), 0.5);
    let mut pos = Vec3::new(0.4, 0.0, 0.0); // Inside sphere + particle_radius
    let mut vel = Vec3::new(-2.0, 0.0, 0.0); // Moving into sphere center
    let p_radius = 0.05;

    let hit = collider.resolve(&mut pos, &mut vel, p_radius);
    assert!(hit);
    // Position should be pushed out to radius + p_radius = 0.55
    assert!((pos.x - 0.55).abs() < 1e-4);
    // Velocity should be reflected outward
    assert!(vel.x > 0.0);
}

#[test]
fn test_box_collider_response() {
    let collider = Collider::box_obstacle(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
    let mut pos = Vec3::new(0.0, 0.48, 0.0); // Near top face
    let mut vel = Vec3::new(0.0, -1.0, 0.0); // Moving downward into box
    let p_radius = 0.05;

    let hit = collider.resolve(&mut pos, &mut vel, p_radius);
    assert!(hit);
    // Pushed above top face + radius
    assert!(pos.y >= 0.55 - 1e-5);
    assert!(vel.y >= 0.0);
}

#[test]
fn test_solver_step_stability() {
    let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.6), Vec3::new(1.0, 1.8, 0.6));
    let config = SolverConfig::default();
    let mut solver = SphSolver::new(bounds, config);

    SceneGenerator::setup_scene(&mut solver, ScenePreset::DamBreak);
    assert!(solver.particle_count() > 0);

    let initial_count = solver.particle_count();
    let solver_bounds = solver.bounds;

    // Step simulation 20 times
    for _ in 0..20 {
        solver.step(0.016);
    }

    assert_eq!(solver.particle_count(), initial_count);

    // Verify all particles stay within simulation bounds
    for p in &solver.particles {
        assert!(p.position.x >= solver_bounds.min.x - 0.01);
        assert!(p.position.x <= solver_bounds.max.x + 0.01);
        assert!(p.position.y >= solver_bounds.min.y - 0.01);
        assert!(p.position.y <= solver_bounds.max.y + 0.01);
        assert!(p.position.z >= solver_bounds.min.z - 0.01);
        assert!(p.position.z <= solver_bounds.max.z + 0.01);
        assert!(!p.velocity.is_nan());
        assert!(!p.position.is_nan());
    }
}

#[test]
fn test_ply_export_binary_and_ascii() {
    let p1 = Particle::new(Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.0, 0.0, 0.0), 1.0, 0);
    let p2 = Particle::new(Vec3::new(-0.1, 0.5, 0.2), Vec3::new(0.0, -1.0, 0.0), 1.0, 1);
    let particles = vec![p1, p2];
    let colors = vec![[255, 0, 0], [0, 255, 0]];

    let dir = std::env::temp_dir().join("liquid_rust_test_export");
    let _ = std::fs::create_dir_all(&dir);

    let bin_path = dir.join("test_bin.ply");
    let ascii_path = dir.join("test_ascii.ply");

    PlyExporter::save_binary(&bin_path, &particles, &colors).expect("Save binary PLY");
    PlyExporter::save_ascii(&ascii_path, &particles, &colors).expect("Save ascii PLY");

    assert!(bin_path.exists());
    assert!(ascii_path.exists());

    let ascii_content = std::fs::read_to_string(&ascii_path).unwrap();
    assert!(ascii_content.contains("format ascii 1.0"));
    assert!(ascii_content.contains("element vertex 2"));

    let bin_content = std::fs::read(&bin_path).unwrap();
    assert!(bin_content.starts_with(b"ply\nformat binary_little_endian 1.0\n"));

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
