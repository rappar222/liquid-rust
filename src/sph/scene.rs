use crate::math::Aabb;
use crate::sph::chamber::ChamberShape;
use crate::sph::collider::Collider;
use crate::sph::particle::Particle;
use crate::sph::solver::{compute_lattice_rest_density, SphSolver};
use crate::sph::substance::MaterialType;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenePreset {
    DamBreak,
    DoubleDamBreak,
    DropletSplash,
    ObstacleDrop,
    VortexTornado,
    ZeroGravityBlobs,
    GasFireBoil,
    HoneyCascade,
    MilkAndOil,
    FlammableAlcohol,
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
        ScenePreset::GasFireBoil,
        ScenePreset::HoneyCascade,
        ScenePreset::MilkAndOil,
        ScenePreset::FlammableAlcohol,
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
            ScenePreset::GasFireBoil => "🔥 Gas, Fire & Boiling Steam",
            ScenePreset::HoneyCascade => "🍯 Viscous Honey Cascade",
            ScenePreset::MilkAndOil => "🥛 Milk & Oil Layering",
            ScenePreset::FlammableAlcohol => "🍸 Volatile Flammable Alcohol",
            ScenePreset::Empty => "Empty Basin (Faucets / Inflow)",
        }
    }
}

pub struct SceneGenerator;

impl SceneGenerator {
    /// Spawn regular lattice block of particles
    pub fn spawn_block(
        min: Vec3,
        max: Vec3,
        spacing: f32,
        initial_vel: Vec3,
        start_id: &mut u32,
    ) -> Vec<Particle> {
        let mut particles = Vec::new();
        let nx = ((max.x - min.x) / spacing).floor() as usize;
        let ny = ((max.y - min.y) / spacing).floor() as usize;
        let nz = ((max.z - min.z) / spacing).floor() as usize;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let pos = Vec3::new(
                        min.x + (ix as f32 + 0.5) * spacing,
                        min.y + (iy as f32 + 0.5) * spacing,
                        min.z + (iz as f32 + 0.5) * spacing,
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

    /// Spawn regular lattice block of particles with specific material & temperature
    pub fn spawn_block_mat(
        min: Vec3,
        max: Vec3,
        spacing: f32,
        initial_vel: Vec3,
        material: MaterialType,
        temperature: f32,
        start_id: &mut u32,
    ) -> Vec<Particle> {
        let mut particles = Vec::new();
        let nx = ((max.x - min.x) / spacing).floor() as usize;
        let ny = ((max.y - min.y) / spacing).floor() as usize;
        let nz = ((max.z - min.z) / spacing).floor() as usize;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let pos = Vec3::new(
                        min.x + (ix as f32 + 0.5) * spacing,
                        min.y + (iy as f32 + 0.5) * spacing,
                        min.z + (iz as f32 + 0.5) * spacing,
                    );

                    particles.push(Particle::new_with_material_and_temp(
                        pos,
                        initial_vel,
                        material.density_factor(),
                        *start_id,
                        material,
                        temperature,
                    ));
                    *start_id += 1;
                }
            }
        }
        particles
    }

    /// Spawn spherical cluster of particles with specific material & temperature
    pub fn spawn_sphere_mat(
        center: Vec3,
        radius: f32,
        spacing: f32,
        initial_vel: Vec3,
        material: MaterialType,
        temperature: f32,
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
                        particles.push(Particle::new_with_material_and_temp(
                            pos,
                            initial_vel,
                            material.density_factor(),
                            *start_id,
                            material,
                            temperature,
                        ));
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
                                let vx = -speed * theta.cos();
                                let vz = speed * theta.sin();
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
            ScenePreset::GasFireBoil => {
                solver.bounds = Aabb::new(Vec3::new(-0.7, 0.0, -0.45), Vec3::new(0.7, 1.4, 0.45));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.viscosity = 0.05;

                // 1. Water basin at bottom (20°C)
                let water = Self::spawn_block_mat(
                    Vec3::new(-0.65, 0.02, -0.41),
                    Vec3::new(0.65, 0.35, 0.41),
                    spacing,
                    Vec3::ZERO,
                    MaterialType::Water,
                    20.0,
                    &mut id,
                );
                for p in water {
                    solver.add_particle(p);
                }

                // 2. Flammable gas pocket in the middle (25°C)
                let gas = Self::spawn_block_mat(
                    Vec3::new(-0.45, 0.45, -0.32),
                    Vec3::new(0.45, 0.85, 0.32),
                    spacing,
                    Vec3::ZERO,
                    MaterialType::Gas,
                    25.0,
                    &mut id,
                );
                for p in gas {
                    solver.add_particle(p);
                }

                // 3. Fire spark / flame cluster ready to ignite the gas
                let fire = Self::spawn_sphere_mat(
                    Vec3::new(0.0, 0.42, 0.0),
                    0.09,
                    spacing,
                    Vec3::new(0.0, 0.8, 0.0),
                    MaterialType::Fire,
                    750.0,
                    &mut id,
                );
                for p in fire {
                    solver.add_particle(p);
                }
            }
            ScenePreset::HoneyCascade => {
                solver.bounds = Aabb::new(Vec3::new(-0.7, 0.0, -0.45), Vec3::new(0.7, 1.3, 0.45));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.viscosity = 0.38; // Rich, slow-flowing honey

                // Shelf obstacle 1 (top-right)
                solver.colliders.push(Collider::box_obstacle(
                    Vec3::new(0.05, 0.70, -0.45),
                    Vec3::new(0.70, 0.75, 0.45),
                ));

                // Shelf obstacle 2 (mid-left)
                solver.colliders.push(Collider::box_obstacle(
                    Vec3::new(-0.70, 0.38, -0.45),
                    Vec3::new(-0.05, 0.43, 0.45),
                ));

                // Honey reservoir at top
                let honey = Self::spawn_block_mat(
                    Vec3::new(0.1, 0.80, -0.38),
                    Vec3::new(0.55, 1.22, 0.38),
                    spacing,
                    Vec3::ZERO,
                    MaterialType::Honey,
                    22.0,
                    &mut id,
                );
                for p in honey {
                    solver.add_particle(p);
                }
            }
            ScenePreset::MilkAndOil => {
                solver.bounds = Aabb::new(Vec3::new(-0.65, 0.0, -0.45), Vec3::new(0.65, 1.25, 0.45));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.viscosity = 0.12;

                // Dense milk layer at the bottom
                let milk = Self::spawn_block_mat(
                    Vec3::new(-0.60, 0.02, -0.40),
                    Vec3::new(0.60, 0.30, 0.40),
                    spacing,
                    Vec3::ZERO,
                    MaterialType::Milk,
                    15.0,
                    &mut id,
                );
                for p in milk {
                    solver.add_particle(p);
                }

                // Lighter oil layer dropped from above
                let oil = Self::spawn_block_mat(
                    Vec3::new(-0.45, 0.50, -0.35),
                    Vec3::new(0.45, 0.85, 0.35),
                    spacing,
                    Vec3::new(0.0, -1.2, 0.0),
                    MaterialType::Oil,
                    20.0,
                    &mut id,
                );
                for p in oil {
                    solver.add_particle(p);
                }
            }
            ScenePreset::FlammableAlcohol => {
                solver.bounds = Aabb::new(Vec3::new(-0.65, 0.0, -0.45), Vec3::new(0.65, 1.25, 0.45));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.viscosity = 0.02;

                // Alcohol pool at bottom
                let alcohol = Self::spawn_block_mat(
                    Vec3::new(-0.60, 0.02, -0.40),
                    Vec3::new(0.60, 0.32, 0.40),
                    spacing,
                    Vec3::ZERO,
                    MaterialType::Alcohol,
                    20.0,
                    &mut id,
                );
                for p in alcohol {
                    solver.add_particle(p);
                }

                // Flammable vapor cloud above pool
                let vapor = Self::spawn_block_mat(
                    Vec3::new(-0.50, 0.36, -0.32),
                    Vec3::new(0.50, 0.58, 0.32),
                    spacing,
                    Vec3::ZERO,
                    MaterialType::Gas,
                    35.0,
                    &mut id,
                );
                for p in vapor {
                    solver.add_particle(p);
                }

                // Falling fire spark from ceiling
                let spark = Self::spawn_sphere_mat(
                    Vec3::new(0.0, 0.95, 0.0),
                    0.06,
                    spacing,
                    Vec3::new(0.0, -2.0, 0.0),
                    MaterialType::Fire,
                    800.0,
                    &mut id,
                );
                for p in spark {
                    solver.add_particle(p);
                }
            }
            ScenePreset::Empty => {
                solver.bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.viscosity = 0.05;
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
