use crate::math::Aabb;
use crate::sph::collider::Collider;
use crate::sph::particle::Particle;
use crate::sph::solver::SphSolver;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenePreset {
    DamBreak,
    DoubleDamBreak,
    DropletSplash,
    ObstacleDrop,
    VortexTornado,
    ZeroGravityBlobs,
}

impl ScenePreset {
    pub const ALL: &'static [ScenePreset] = &[
        ScenePreset::DamBreak,
        ScenePreset::DoubleDamBreak,
        ScenePreset::DropletSplash,
        ScenePreset::ObstacleDrop,
        ScenePreset::VortexTornado,
        ScenePreset::ZeroGravityBlobs,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ScenePreset::DamBreak => "Dam Break (Classic)",
            ScenePreset::DoubleDamBreak => "Double Dam Break (Collision)",
            ScenePreset::DropletSplash => "Droplet Splash into Pool",
            ScenePreset::ObstacleDrop => "Fluid onto Obstacle",
            ScenePreset::VortexTornado => "Swirling Vortex / Tornado",
            ScenePreset::ZeroGravityBlobs => "Zero-G Droplet Coalescence",
        }
    }
}

pub struct SceneGenerator;

impl SceneGenerator {
    /// Helper to fill a box volume with particles spaced by spacing
    pub fn spawn_block(
        min: Vec3,
        max: Vec3,
        spacing: f32,
        initial_vel: Vec3,
        start_id: &mut u32,
    ) -> Vec<Particle> {
        let mut particles = Vec::new();
        let jitter = spacing * 0.05;

        let nx = ((max.x - min.x) / spacing).floor() as usize;
        let ny = ((max.y - min.y) / spacing).floor() as usize;
        let nz = ((max.z - min.z) / spacing).floor() as usize;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let rx = (ix as f32 % 3.0 - 1.0) * jitter;
                    let ry = (iy as f32 % 3.0 - 1.0) * jitter;
                    let rz = (iz as f32 % 3.0 - 1.0) * jitter;

                    let pos = Vec3::new(
                        min.x + (ix as f32 + 0.5) * spacing + rx,
                        min.y + (iy as f32 + 0.5) * spacing + ry,
                        min.z + (iz as f32 + 0.5) * spacing + rz,
                    );

                    particles.push(Particle::new(pos, initial_vel, 1.0, *start_id));
                    *start_id += 1;
                }
            }
        }
        particles
    }

    /// Helper to spawn a spherical cluster of particles
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
        let spacing = solver.config.particle_radius * 1.6;

        match preset {
            ScenePreset::DamBreak => {
                solver.bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.6), Vec3::new(1.0, 1.8, 0.6));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.surface_tension = 0.02;

                // Column on left side
                let particles = Self::spawn_block(
                    Vec3::new(-0.95, 0.05, -0.55),
                    Vec3::new(-0.2, 1.4, 0.55),
                    spacing,
                    Vec3::ZERO,
                    &mut id,
                );
                for p in particles {
                    solver.add_particle(p);
                }
            }
            ScenePreset::DoubleDamBreak => {
                solver.bounds = Aabb::new(Vec3::new(-1.2, 0.0, -0.6), Vec3::new(1.2, 1.6, 0.6));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.surface_tension = 0.03;

                // Left column
                let p1 = Self::spawn_block(
                    Vec3::new(-1.15, 0.05, -0.55),
                    Vec3::new(-0.5, 1.3, 0.55),
                    spacing,
                    Vec3::new(0.5, 0.0, 0.0),
                    &mut id,
                );
                for p in p1 {
                    solver.add_particle(p);
                }

                // Right column
                let p2 = Self::spawn_block(
                    Vec3::new(0.5, 0.05, -0.55),
                    Vec3::new(1.15, 1.3, 0.55),
                    spacing,
                    Vec3::new(-0.5, 0.0, 0.0),
                    &mut id,
                );
                for p in p2 {
                    solver.add_particle(p);
                }
            }
            ScenePreset::DropletSplash => {
                solver.bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.8), Vec3::new(0.8, 1.8, 0.8));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.surface_tension = 0.08;

                // Pool at bottom
                let pool = Self::spawn_block(
                    Vec3::new(-0.75, 0.05, -0.75),
                    Vec3::new(0.75, 0.45, 0.75),
                    spacing,
                    Vec3::ZERO,
                    &mut id,
                );
                for p in pool {
                    solver.add_particle(p);
                }

                // Droplet falling from above with downward velocity
                let drop = Self::spawn_sphere(
                    Vec3::new(0.0, 1.3, 0.0),
                    0.24,
                    spacing,
                    Vec3::new(0.0, -4.5, 0.0),
                    &mut id,
                );
                for p in drop {
                    solver.add_particle(p);
                }
            }
            ScenePreset::ObstacleDrop => {
                solver.bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.8), Vec3::new(0.8, 1.8, 0.8));
                solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                solver.config.surface_tension = 0.04;

                // Spherical obstacle in the middle
                solver
                    .colliders
                    .push(Collider::sphere(Vec3::new(0.0, 0.6, 0.0), 0.28));

                // Fluid block above the obstacle
                let block = Self::spawn_block(
                    Vec3::new(-0.45, 1.0, -0.45),
                    Vec3::new(0.45, 1.7, 0.45),
                    spacing,
                    Vec3::ZERO,
                    &mut id,
                );
                for p in block {
                    solver.add_particle(p);
                }
            }
            ScenePreset::VortexTornado => {
                solver.bounds = Aabb::new(Vec3::new(-0.9, 0.0, -0.9), Vec3::new(0.9, 1.8, 0.9));
                solver.config.gravity = Vec3::new(0.0, -5.0, 0.0);
                solver.config.surface_tension = 0.03;

                let mut cylinder_particles = Vec::new();
                let r_max = 0.7;
                let steps = (r_max / spacing).ceil() as i32;
                let height_steps = (1.2 / spacing).ceil() as usize;

                for iy in 0..height_steps {
                    let y = 0.05 + iy as f32 * spacing;
                    for ix in -steps..=steps {
                        for iz in -steps..=steps {
                            let x = ix as f32 * spacing;
                            let z = iz as f32 * spacing;
                            let r = (x * x + z * z).sqrt();
                            if r <= r_max && r > 0.05 {
                                let theta = z.atan2(x);
                                let speed = 3.5 * (r / r_max);
                                let vx = -speed * theta.cos();
                                let vz = speed * theta.sin();
                                let p = Particle::new(
                                    Vec3::new(x, y, z),
                                    Vec3::new(vx, 0.0, vz),
                                    1.0,
                                    id,
                                );
                                id += 1;
                                cylinder_particles.push(p);
                            }
                        }
                    }
                }
                for p in cylinder_particles {
                    solver.add_particle(p);
                }
            }
            ScenePreset::ZeroGravityBlobs => {
                solver.bounds = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
                solver.config.gravity = Vec3::ZERO;
                solver.config.surface_tension = 0.25; // high surface tension

                // Two droplets moving toward each other
                let b1 = Self::spawn_sphere(
                    Vec3::new(-0.45, 0.0, 0.0),
                    0.28,
                    spacing,
                    Vec3::new(0.8, 0.0, 0.0),
                    &mut id,
                );
                for p in b1 {
                    solver.add_particle(p);
                }

                let b2 = Self::spawn_sphere(
                    Vec3::new(0.45, 0.1, 0.0),
                    0.22,
                    spacing,
                    Vec3::new(-0.8, 0.0, 0.0),
                    &mut id,
                );
                for p in b2 {
                    solver.add_particle(p);
                }
            }
        }
    }
}
