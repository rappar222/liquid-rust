use crate::math::{Aabb, Ray};
use crate::sph::collider::Collider;
use crate::sph::kernel::SphKernels;
use crate::sph::particle::{Particle, ParticleType};
use crate::sph::spatial::SpatialHashGrid;
use glam::Vec3;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SolverConfig {
    pub rest_density: f32,
    pub stiffness: f32,          // Bulk modulus k for EOS: P = k * ((rho/rho0)^gamma - 1)
    pub exponent: f32,           // Tait exponent (typically 7.0 for water, or 1.0 for linear)
    pub viscosity: f32,          // Dynamic viscosity coefficient mu
    pub surface_tension: f32,    // Cohesive surface tension coefficient
    pub smoothing_length: f32,   // SPH kernel radius h
    pub particle_radius: f32,    // Physical radius for particle-particle / particle-wall collision
    pub gravity: Vec3,
    pub boundary_restitution: f32,
    pub boundary_friction: f32,
    pub max_velocity: f32,
    pub substeps: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        let particle_radius = 0.035;
        let smoothing_length = particle_radius * 2.2;
        Self {
            rest_density: 1000.0,
            stiffness: 1500.0,
            exponent: 1.0,
            viscosity: 0.15,
            surface_tension: 0.05,
            smoothing_length,
            particle_radius,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            boundary_restitution: 0.2,
            boundary_friction: 0.98,
            max_velocity: 40.0,
            substeps: 3,
        }
    }
}

pub struct SphSolver {
    pub particles: Vec<Particle>,
    pub bounds: Aabb,
    pub colliders: Vec<Collider>,
    pub config: SolverConfig,
    pub kernels: SphKernels,
    pub grid: SpatialHashGrid,
    // Working buffers to avoid per-frame reallocations
    positions_cache: Vec<Vec3>,
    densities_cache: Vec<f32>,
    pressures_cache: Vec<f32>,
    velocities_cache: Vec<Vec3>,
    // Mouse interaction
    pub mouse_ray: Option<Ray>,
    pub mouse_force_strength: f32,
    pub mouse_force_mode: i32, // 1: attract, -1: repel, 0: off
}

impl SphSolver {
    pub fn new(bounds: Aabb, config: SolverConfig) -> Self {
        let kernels = SphKernels::new(config.smoothing_length);
        let grid = SpatialHashGrid::new(config.smoothing_length, 1024);

        Self {
            particles: Vec::new(),
            bounds,
            colliders: Vec::new(),
            config,
            kernels,
            grid,
            positions_cache: Vec::new(),
            densities_cache: Vec::new(),
            pressures_cache: Vec::new(),
            velocities_cache: Vec::new(),
            mouse_ray: None,
            mouse_force_strength: 50.0,
            mouse_force_mode: 0,
        }
    }

    pub fn set_smoothing_length(&mut self, h: f32) {
        self.config.smoothing_length = h;
        self.kernels = SphKernels::new(h);
        self.grid = SpatialHashGrid::new(h, self.particles.len().max(1024));
    }

    pub fn add_particle(&mut self, p: Particle) {
        self.particles.push(p);
    }

    pub fn clear_particles(&mut self) {
        self.particles.clear();
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Single simulation time step decomposed into substeps
    pub fn step(&mut self, dt: f32) {
        if self.particles.is_empty() {
            return;
        }

        let substeps = self.config.substeps.max(1);
        let sub_dt = dt / substeps as f32;

        for _ in 0..substeps {
            self.substep(sub_dt);
        }
    }

    fn substep(&mut self, dt: f32) {
        let n = self.particles.len();

        // 1. Sync position cache and rebuild spatial grid
        if self.positions_cache.len() != n {
            self.positions_cache.resize(n, Vec3::ZERO);
            self.densities_cache.resize(n, 1000.0);
            self.pressures_cache.resize(n, 0.0);
            self.velocities_cache.resize(n, Vec3::ZERO);
        }

        for (i, p) in self.particles.iter().enumerate() {
            self.positions_cache[i] = p.position;
            self.velocities_cache[i] = p.velocity;
        }

        self.grid.build(&self.positions_cache);

        let positions = &self.positions_cache;
        let velocities = &self.velocities_cache;
        let grid = &self.grid;
        let kernels = &self.kernels;
        let config = &self.config;

        // 2. Parallel Density and Pressure computation
        let rho0 = config.rest_density;
        let k = config.stiffness;
        let exponent = config.exponent;
        let h2 = kernels.h2;

        let self_density_term = kernels.poly6(0.0);

        self.densities_cache
            .par_iter_mut()
            .zip(self.pressures_cache.par_iter_mut())
            .enumerate()
            .for_each(|(i, (density_out, pressure_out))| {
                let pos_i = positions[i];
                let mut density = self_density_term; // particle self-contribution

                grid.for_each_neighbor(pos_i, |j| {
                    if i != j {
                        let r2 = (pos_i - positions[j]).length_squared();
                        if r2 < h2 {
                            density += kernels.poly6(r2);
                        }
                    }
                });

                // Multiply by particle mass (approximated uniform mass)
                let mass = 1.0;
                density *= mass;

                // Density should not drop to zero
                let density = density.max(rho0 * 0.1);
                *density_out = density;

                // Pressure calculation: Tait EOS or linear EOS
                let pressure = if (exponent - 1.0).abs() < 1e-3 {
                    // Linear EOS
                    (k * (density - rho0)).max(0.0)
                } else {
                    // Tait EOS
                    let ratio = density / rho0;
                    let p = (k / exponent) * (ratio.powf(exponent) - 1.0);
                    p.max(0.0)
                };

                *pressure_out = pressure;
            });

        let densities = &self.densities_cache;
        let pressures = &self.pressures_cache;

        // 3. Parallel Force evaluation (Pressure, Viscosity, Surface Tension, Gravity, Mouse)
        let mu = config.viscosity;
        let sigma = config.surface_tension;
        let gravity = config.gravity;
        let mouse_ray = self.mouse_ray;
        let mouse_mode = self.mouse_force_mode;
        let mouse_strength = self.mouse_force_strength;

        self.particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, p)| {
                p.density = densities[i];
                p.pressure = pressures[i];

                if p.particle_type == ParticleType::Boundary {
                    p.acceleration = Vec3::ZERO;
                    return;
                }

                let pos_i = positions[i];
                let vel_i = velocities[i];
                let rho_i = densities[i];
                let pres_i = pressures[i];

                let mut f_pressure = Vec3::ZERO;
                let mut f_viscosity = Vec3::ZERO;
                let mut f_surface = Vec3::ZERO;

                grid.for_each_neighbor(pos_i, |j| {
                    if i != j {
                        let pos_j = positions[j];
                        let r_vec = pos_i - pos_j;
                        let r2 = r_vec.length_squared();

                        if r2 < h2 && r2 > 1e-12 {
                            let r = r2.sqrt();
                            let rho_j = densities[j];
                            let pres_j = pressures[j];
                            let vel_j = velocities[j];

                            // Symmetric pressure gradient force
                            // f_p = - m * (P_i / rho_i^2 + P_j / rho_j^2) * grad_W
                            let p_factor = pres_i / (rho_i * rho_i) + pres_j / (rho_j * rho_j);
                            let grad_w = kernels.spiky_gradient(r_vec, r);
                            f_pressure -= grad_w * (p_factor * rho0);

                            // Viscosity force
                            // f_v = mu * (v_j - v_i) / rho_j * lap_W
                            let lap_w = kernels.viscosity_laplacian(r);
                            f_viscosity += (vel_j - vel_i) * (lap_w / rho_j);

                            // Surface tension / cohesion force (Akinci cohesion kernel)
                            if sigma > 0.0 {
                                let cohesion_w = kernels.poly6(r2);
                                f_surface -= (r_vec / r) * (cohesion_w * sigma * 1000.0);
                            }
                        }
                    }
                });

                f_viscosity *= mu;

                // User mouse interaction force (attractor or repeller)
                let mut f_user = Vec3::ZERO;
                if mouse_mode != 0 {
                    if let Some(ray) = mouse_ray {
                        let to_p = pos_i - ray.origin;
                        let proj = to_p.dot(ray.direction);
                        if proj > 0.0 {
                            let closest = ray.origin + ray.direction * proj;
                            let diff = closest - pos_i;
                            let dist = diff.length();
                            if dist < 0.6 && dist > 1e-4 {
                                let dir = diff / dist;
                                let force_mag = (1.0 - dist / 0.6) * mouse_strength;
                                f_user = dir * (force_mag * mouse_mode as f32);
                            }
                        }
                    }
                }

                // Sum accelerations (f / rho + g + user)
                let a = (f_pressure + f_viscosity + f_surface) / rho_i + gravity + f_user;
                p.acceleration = a;
            });

        // 4. Parallel Integration & Collision Response
        let max_vel = config.max_velocity;
        let bounds = self.bounds;
        let radius = config.particle_radius;
        let restitution = config.boundary_restitution;
        let friction = config.boundary_friction;
        let colliders = &self.colliders;

        self.particles.par_iter_mut().for_each(|p| {
            if p.particle_type == ParticleType::Boundary {
                return;
            }

            // Symplectic Euler integration
            p.velocity += p.acceleration * dt;

            // Velocity clamping
            let speed = p.velocity.length();
            if speed > max_vel {
                p.velocity = (p.velocity / speed) * max_vel;
            }

            p.position += p.velocity * dt;

            // 5. Collision with Bounding Box
            // X min/max
            if p.position.x < bounds.min.x + radius {
                p.position.x = bounds.min.x + radius;
                if p.velocity.x < 0.0 {
                    p.velocity.x = -p.velocity.x * restitution;
                    p.velocity.y *= friction;
                    p.velocity.z *= friction;
                }
            } else if p.position.x > bounds.max.x - radius {
                p.position.x = bounds.max.x - radius;
                if p.velocity.x > 0.0 {
                    p.velocity.x = -p.velocity.x * restitution;
                    p.velocity.y *= friction;
                    p.velocity.z *= friction;
                }
            }

            // Y min/max (floor & ceiling)
            if p.position.y < bounds.min.y + radius {
                p.position.y = bounds.min.y + radius;
                if p.velocity.y < 0.0 {
                    p.velocity.y = -p.velocity.y * restitution;
                    p.velocity.x *= friction;
                    p.velocity.z *= friction;
                }
            } else if p.position.y > bounds.max.y - radius {
                p.position.y = bounds.max.y - radius;
                if p.velocity.y > 0.0 {
                    p.velocity.y = -p.velocity.y * restitution;
                    p.velocity.x *= friction;
                    p.velocity.z *= friction;
                }
            }

            // Z min/max
            if p.position.z < bounds.min.z + radius {
                p.position.z = bounds.min.z + radius;
                if p.velocity.z < 0.0 {
                    p.velocity.z = -p.velocity.z * restitution;
                    p.velocity.x *= friction;
                    p.velocity.y *= friction;
                }
            } else if p.position.z > bounds.max.z - radius {
                p.position.z = bounds.max.z - radius;
                if p.velocity.z > 0.0 {
                    p.velocity.z = -p.velocity.z * restitution;
                    p.velocity.x *= friction;
                    p.velocity.y *= friction;
                }
            }

            // 6. Collision with active colliders
            for collider in colliders {
                collider.resolve(&mut p.position, &mut p.velocity, radius);
            }
        });
    }
}
