use crate::math::Aabb;
use crate::sph::chamber::ChamberShape;
use crate::sph::collider::Collider;
use crate::sph::particle::Particle;
use crate::sph::solver::{compute_lattice_rest_density, compute_lattice_rest_density_2d, SphSolver};
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenePreset {
    DamBreak,
    DoubleDamBreak,
    DropletSplash,
    ObstacleDrop,
    VortexTornado,
    ZeroGravityBlobs,
    Empty,
}

impl ScenePreset {
    pub const ALL: &'static [ScenePreset] = &[
        ScenePreset::DamBreak,
        ScenePreset::DoubleDamBreak,
        ScenePreset::DropletSplash,
        ScenePreset::ObstacleDrop,
        ScenePreset::VortexTornado,
        ScenePreset::ZeroGravityBlobs,
        ScenePreset::Empty,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ScenePreset::DamBreak => "Dam Break (Classic)",
            ScenePreset::DoubleDamBreak => "Double Dam Break (Collision)",
            ScenePreset::DropletSplash => "Droplet Splash into Pool",
            ScenePreset::ObstacleDrop => "Fluid onto Obstacle",
            ScenePreset::VortexTornado => "Swirling Vortex / Tornado",
            ScenePreset::ZeroGravityBlobs => "Zero-G Droplet Coalescence",
            ScenePreset::Empty => "Empty Basin (Faucets / Inflow)",
        }
    }
}

pub struct SceneGenerator;

impl SceneGenerator {
    /// Spawn regular lattice block of particles (supports both 3D volumes and 2D planar sheets)
    pub fn spawn_block(
        min: Vec3,
        max: Vec3,
        spacing: f32,
        initial_vel: Vec3,
        start_id: &mut u32,
    ) -> Vec<Particle> {
        let mut particles = Vec::new();
        let nx = (((max.x - min.x) / spacing).floor() as usize).max(1);
        let ny = (((max.y - min.y) / spacing).floor() as usize).max(1);
        let is_2d_slab = (max.z - min.z).abs() < 1e-4;
        let nz = if is_2d_slab {
            1
        } else {
            (((max.z - min.z) / spacing).floor() as usize).max(1)
        };

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let pos = Vec3::new(
                        min.x + (ix as f32 + 0.5) * spacing,
                        min.y + (iy as f32 + 0.5) * spacing,
                        if is_2d_slab { min.z } else { min.z + (iz as f32 + 0.5) * spacing },
                    );

                    particles.push(Particle::new(pos, initial_vel, 1.0, *start_id));
                    *start_id += 1;
                }
            }
        }
        particles
    }

    /// Spawn spherical cluster of particles on lattice
    pub fn spawn_sphere(
        center: Vec3,
        radius: f32,
        spacing: f32,
        initial_vel: Vec3,
        start_id: &mut u32,
    ) -> Vec<Particle> {
        let mut particles = Vec::new();
        let r2 = radius * radius;
        let steps = (radius / spacing).ceil() as i32;

        for ix in -steps..=steps {
            for iy in -steps..=steps {
                for iz in -steps..=steps {
                    let offset = Vec3::new(
                        ix as f32 * spacing,
                        iy as f32 * spacing,
                        iz as f32 * spacing,
                    );
                    if offset.length_squared() <= r2 {
                        let pos = center + offset;
                        particles.push(Particle::new(pos, initial_vel, 1.0, *start_id));
                        *start_id += 1;
                    }
                }
            }
        }
        particles
    }

    /// Setup solver for specified preset
    pub fn setup_scene(solver: &mut SphSolver, preset: ScenePreset) {
        solver.clear_particles();
        solver.colliders.clear();
        let mut id = 0;

        let spacing = solver.config.spacing;

        if solver.is_2d {
            solver.config.rest_density =
                compute_lattice_rest_density_2d(solver.config.smoothing_length, spacing);

            match preset {
                ScenePreset::DamBreak => {
                    solver.bounds = Aabb::new(Vec3::new(-0.9, 0.0, -0.05), Vec3::new(0.9, 1.2, 0.05));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.08;

                    let particles = Self::spawn_block(
                        Vec3::new(-0.85, 0.02, 0.0),
                        Vec3::new(-0.25, 0.95, 0.0),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for mut p in particles {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }
                }
                ScenePreset::DoubleDamBreak => {
                    solver.bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.05), Vec3::new(1.0, 1.2, 0.05));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.08;

                    let p1 = Self::spawn_block(
                        Vec3::new(-0.95, 0.02, 0.0),
                        Vec3::new(-0.45, 0.95, 0.0),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for mut p in p1 {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }

                    let p2 = Self::spawn_block(
                        Vec3::new(0.45, 0.02, 0.0),
                        Vec3::new(0.95, 0.95, 0.0),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for mut p in p2 {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }
                }
                ScenePreset::DropletSplash => {
                    solver.bounds = Aabb::new(Vec3::new(-0.85, 0.0, -0.05), Vec3::new(0.85, 1.2, 0.05));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.05;

                    let pool = Self::spawn_block(
                        Vec3::new(-0.80, 0.02, 0.0),
                        Vec3::new(0.80, 0.35, 0.0),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for mut p in pool {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }

                    let drop = Self::spawn_sphere(
                        Vec3::new(0.0, 0.85, 0.0),
                        0.18,
                        spacing,
                        Vec3::new(0.0, -3.5, 0.0),
                        &mut id,
                    );
                    for mut p in drop {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }
                }
                ScenePreset::ObstacleDrop => {
                    solver.bounds = Aabb::new(Vec3::new(-0.85, 0.0, -0.05), Vec3::new(0.85, 1.3, 0.05));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.08;

                    let mut col = Collider::sphere(Vec3::new(0.0, 0.45, 0.0), 0.22);
                    col.name = "Central Circle".to_string();
                    solver.colliders.push(col);

                    let block = Self::spawn_block(
                        Vec3::new(-0.45, 0.80, 0.0),
                        Vec3::new(0.45, 1.25, 0.0),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for mut p in block {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }
                }
                ScenePreset::VortexTornado => {
                    solver.bounds = Aabb::new(Vec3::new(-0.85, 0.0, -0.05), Vec3::new(0.85, 1.2, 0.05));
                    solver.config.gravity = Vec3::ZERO;
                    solver.config.viscosity = 0.04;

                    let center = Vec3::new(0.0, 0.6, 0.0);
                    let r_max = 0.45;
                    let steps = (r_max / spacing).ceil() as i32;

                    for ix in -steps..=steps {
                        for iy in -steps..=steps {
                            let x = ix as f32 * spacing;
                            let y = iy as f32 * spacing;
                            let r = (x * x + y * y).sqrt();
                            if r <= r_max && r > 0.04 {
                                let tangent = Vec3::new(-y, x, 0.0).normalize();
                                let speed = 2.6 * (r / r_max);
                                let p = Particle::new(
                                    center + Vec3::new(x, y, 0.0),
                                    tangent * speed,
                                    1.0,
                                    id,
                                );
                                id += 1;
                                solver.add_particle(p);
                            }
                        }
                    }
                }
                ScenePreset::ZeroGravityBlobs => {
                    solver.bounds = Aabb::new(Vec3::new(-0.9, 0.0, -0.05), Vec3::new(0.9, 1.2, 0.05));
                    solver.config.gravity = Vec3::ZERO;
                    solver.config.viscosity = 0.10;
                    solver.config.tensile_k = 0.005;

                    let b1 = Self::spawn_sphere(
                        Vec3::new(-0.35, 0.6, 0.0),
                        0.20,
                        spacing,
                        Vec3::new(0.6, 0.0, 0.0),
                        &mut id,
                    );
                    for mut p in b1 {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }

                    let b2 = Self::spawn_sphere(
                        Vec3::new(0.35, 0.65, 0.0),
                        0.16,
                        spacing,
                        Vec3::new(-0.6, 0.0, 0.0),
                        &mut id,
                    );
                    for mut p in b2 {
                        p.position.z = 0.0;
                        solver.add_particle(p);
                    }
                }
                ScenePreset::Empty => {
                    solver.bounds = Aabb::new(Vec3::new(-0.9, 0.0, -0.05), Vec3::new(0.9, 1.2, 0.05));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.05;
                }
            }
        } else {
            solver.config.rest_density =
                compute_lattice_rest_density(solver.config.smoothing_length, spacing);

            match preset {
                ScenePreset::DamBreak => {
                    solver.bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.08;

                    // Fluid column on left side
                    let particles = Self::spawn_block(
                        Vec3::new(-0.76, 0.02, -0.41),
                        Vec3::new(-0.15, 0.85, 0.41),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for p in particles {
                        solver.add_particle(p);
                    }
                }
                ScenePreset::DoubleDamBreak => {
                    solver.bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.45), Vec3::new(1.0, 1.2, 0.45));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.08;

                    // Left column
                    let p1 = Self::spawn_block(
                        Vec3::new(-0.96, 0.02, -0.41),
                        Vec3::new(-0.45, 0.85, 0.41),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for p in p1 {
                        solver.add_particle(p);
                    }

                    // Right column
                    let p2 = Self::spawn_block(
                        Vec3::new(0.45, 0.02, -0.41),
                        Vec3::new(0.96, 0.85, 0.41),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for p in p2 {
                        solver.add_particle(p);
                    }
                }
                ScenePreset::DropletSplash => {
                    solver.bounds = Aabb::new(Vec3::new(-0.65, 0.0, -0.65), Vec3::new(0.65, 1.2, 0.65));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.05;

                    // Resting reservoir at bottom
                    let pool = Self::spawn_block(
                        Vec3::new(-0.61, 0.02, -0.61),
                        Vec3::new(0.61, 0.32, 0.61),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for p in pool {
                        solver.add_particle(p);
                    }

                    // High-speed spherical drop falling from above
                    let drop = Self::spawn_sphere(
                        Vec3::new(0.0, 0.85, 0.0),
                        0.18,
                        spacing,
                        Vec3::new(0.0, -3.5, 0.0),
                        &mut id,
                    );
                    for p in drop {
                        solver.add_particle(p);
                    }
                }
                ScenePreset::ObstacleDrop => {
                    solver.bounds = Aabb::new(Vec3::new(-0.65, 0.0, -0.65), Vec3::new(0.65, 1.3, 0.65));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.08;

                    // Solid sphere obstacle in center
                    solver
                        .colliders
                        .push(Collider::sphere(Vec3::new(0.0, 0.45, 0.0), 0.22));

                    // Fluid block above obstacle
                    let block = Self::spawn_block(
                        Vec3::new(-0.32, 0.75, -0.32),
                        Vec3::new(0.32, 1.25, 0.32),
                        spacing,
                        Vec3::ZERO,
                        &mut id,
                    );
                    for p in block {
                        solver.add_particle(p);
                    }
                }
                ScenePreset::VortexTornado => {
                    solver.bounds = Aabb::new(Vec3::new(-0.7, 0.0, -0.7), Vec3::new(0.7, 1.2, 0.7));
                    solver.config.gravity = Vec3::new(0.0, -6.0, 0.0);
                    solver.config.viscosity = 0.04;

                    let r_max = 0.55;
                    let steps = (r_max / spacing).ceil() as i32;
                    let height_steps = (0.7 / spacing).ceil() as usize;

                    for iy in 0..height_steps {
                        let y = 0.02 + iy as f32 * spacing;
                        for ix in -steps..=steps {
                            for iz in -steps..=steps {
                                let x = ix as f32 * spacing;
                                let z = iz as f32 * spacing;
                                let r = (x * x + z * z).sqrt();
                                if r <= r_max && r > 0.08 {
                                    let theta = z.atan2(x);
                                    let speed = 2.4 * (r / r_max);
                                    let vx = -speed * theta.sin();
                                    let vz = speed * theta.cos();
                                    let p = Particle::new(
                                        Vec3::new(x, y, z),
                                        Vec3::new(vx, 0.0, vz),
                                        1.0,
                                        id,
                                    );
                                    id += 1;
                                    solver.add_particle(p);
                                }
                            }
                        }
                    }
                }
                ScenePreset::ZeroGravityBlobs => {
                    solver.bounds = Aabb::new(Vec3::new(-0.8, -0.8, -0.8), Vec3::new(0.8, 0.8, 0.8));
                    solver.config.gravity = Vec3::ZERO;
                    solver.config.viscosity = 0.12;
                    solver.config.tensile_k = 0.005;

                    // Two floating liquid spheres moving towards each other
                    let b1 = Self::spawn_sphere(
                        Vec3::new(-0.35, 0.0, 0.0),
                        0.20,
                        spacing,
                        Vec3::new(0.5, 0.0, 0.0),
                        &mut id,
                    );
                    for p in b1 {
                        solver.add_particle(p);
                    }

                    let b2 = Self::spawn_sphere(
                        Vec3::new(0.35, 0.05, 0.0),
                        0.16,
                        spacing,
                        Vec3::new(-0.5, 0.0, 0.0),
                        &mut id,
                    );
                    for p in b2 {
                        solver.add_particle(p);
                    }
                }
                ScenePreset::Empty => {
                    solver.bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
                    solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                    solver.config.viscosity = 0.05;
                }
            }
        }

        // If the chamber has a custom shape or scale, re-apply the chamber bounds
        if solver.chamber.shape != ChamberShape::Box || solver.chamber.scale != Vec3::ONE {
            solver.bounds = solver.chamber.compute_bounds();
        } else {
            solver.chamber.base_size = solver.bounds.size();
        }
    }
}
