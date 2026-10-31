use crate::math::{Aabb, Ray};
use crate::sph::chamber::Chamber;
use crate::sph::collider::Collider;
use crate::sph::kernel::SphKernels;
use crate::sph::particle::{Particle, ParticleType};
use crate::sph::spatial::SpatialHashGrid;
use glam::Vec3;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationQuality {
    Draft,
    Medium,
    High,
    Ultra,
    Cinematic,
    Extreme,
    Custom,
}

impl SimulationQuality {
    pub const ALL: &'static [SimulationQuality] = &[
        SimulationQuality::Draft,
        SimulationQuality::Medium,
        SimulationQuality::High,
        SimulationQuality::Ultra,
        SimulationQuality::Cinematic,
        SimulationQuality::Extreme,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            SimulationQuality::Draft => "Draft (Fast, ~2.5k particles)",
            SimulationQuality::Medium => "Medium (Balanced, ~5.5k particles)",
            SimulationQuality::High => "High (Smooth, ~18k particles)",
            SimulationQuality::Ultra => "Ultra (High-Res, ~50k particles)",
            SimulationQuality::Cinematic => "Cinematic (~120k particles)",
            SimulationQuality::Extreme => "⚠️ Extreme (1,000,000+ particles, Freeze Warning!)",
            SimulationQuality::Custom => "Custom",
        }
    }

    pub fn apply(&self, config: &mut SolverConfig) {
        match self {
            SimulationQuality::Draft => {
                config.particle_radius = 0.032;
                config.spacing = 0.054;
                config.smoothing_length = 0.095;
                config.substeps = 2;
                config.solver_iterations = 3;
            }
            SimulationQuality::Medium => {
                config.particle_radius = 0.024;
                config.spacing = 0.042;
                config.smoothing_length = 0.075;
                config.substeps = 2;
                config.solver_iterations = 3;
            }
            SimulationQuality::High => {
                config.particle_radius = 0.016;
                config.spacing = 0.028;
                config.smoothing_length = 0.050;
                config.substeps = 3;
                config.solver_iterations = 3;
            }
            SimulationQuality::Ultra => {
                config.particle_radius = 0.012;
                config.spacing = 0.020;
                config.smoothing_length = 0.038;
                config.substeps = 3;
                config.solver_iterations = 4;
            }
            SimulationQuality::Cinematic => {
                config.particle_radius = 0.009;
                config.spacing = 0.015;
                config.smoothing_length = 0.028;
                config.substeps = 4;
                config.solver_iterations = 4;
            }
            SimulationQuality::Extreme => {
                config.particle_radius = 0.0042;
                config.spacing = 0.0070;
                config.smoothing_length = 0.014;
                config.substeps = 4;
                config.solver_iterations = 4;
            }
            SimulationQuality::Custom => {}
        }
        config.rest_density = compute_lattice_rest_density(config.smoothing_length, config.spacing);
    }
}

#[derive(Clone, Debug)]
pub struct SolverConfig {
    pub particle_radius: f32,    // Physical radius for collisions
    pub spacing: f32,            // Initial lattice spacing
    pub smoothing_length: f32,   // SPH kernel radius h
    pub rest_density: f32,       // Target rest density rho0
    pub solver_iterations: usize,// PBF solver iterations per substep (typically 3-4)
    pub substeps: usize,         // Number of substeps per frame (typically 2-3)
    pub viscosity: f32,          // Normalized XSPH viscosity [0.0, 1.0]
    pub surface_tension: f32,    // Surface tension cohesion coefficient
    pub restitution: f32,        // Boundary restitution (0.0 for inelastic water, no bouncing)
    pub boundary_friction: f32,  // Wall friction
    pub gravity: Vec3,
    pub tensile_k: f32,          // Tensile instability correction coefficient
    pub max_velocity: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        let particle_radius = 0.024;
        let spacing = 0.042;
        let smoothing_length = 0.075;
        let rest_density = compute_lattice_rest_density(smoothing_length, spacing);

        Self {
            particle_radius,
            spacing,
            smoothing_length,
            rest_density,
            solver_iterations: 3,
            substeps: 2,
            viscosity: 0.15,
            surface_tension: 0.05,
            restitution: 0.02, // Inelastic fluid impact: practically 0 bounce
            boundary_friction: 0.95,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            tensile_k: 0.001,
            max_velocity: 12.0,
        }
    }
}

/// Compute exact rest density on regular cubic lattice with given spacing
pub fn compute_lattice_rest_density(h: f32, spacing: f32) -> f32 {
    let kernels = SphKernels::new(h);
    let mut rho0 = kernels.poly6(0.0);
    let steps = (h / spacing).ceil() as i32;
    let h2 = h * h;

    for ix in -steps..=steps {
        for iy in -steps..=steps {
            for iz in -steps..=steps {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let r2 = (ix as f32 * spacing).powi(2)
                    + (iy as f32 * spacing).powi(2)
                    + (iz as f32 * spacing).powi(2);
                if r2 < h2 {
                    rho0 += kernels.poly6(r2);
                }
            }
        }
    }
    rho0
}

pub struct SphSolver {
    pub particles: Vec<Particle>,
    pub bounds: Aabb,
    pub chamber: Chamber,
    pub colliders: Vec<Collider>,
    pub faucets: Vec<crate::sph::faucet::Faucet>,
    pub drains: Vec<crate::sph::faucet::Drain>,
    pub next_particle_id: u32,
    pub config: SolverConfig,
    pub kernels: SphKernels,
    pub grid: SpatialHashGrid,

    // Working buffers for PBF (Position-Based Fluids)
    predicted_positions: Vec<Vec3>,
    old_positions: Vec<Vec3>,
    lambdas: Vec<f32>,
    delta_positions: Vec<Vec3>,
    densities: Vec<f32>,
    xsph_velocities: Vec<Vec3>,

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
            chamber: Chamber::default(),
            colliders: Vec::new(),
            faucets: Vec::new(),
            drains: Vec::new(),
            next_particle_id: 0,
            config,
            kernels,
            grid,
            predicted_positions: Vec::new(),
            old_positions: Vec::new(),
            lambdas: Vec::new(),
            delta_positions: Vec::new(),
            densities: Vec::new(),
            xsph_velocities: Vec::new(),
            mouse_ray: None,
            mouse_force_strength: 60.0,
            mouse_force_mode: 0,
        }
    }

    pub fn set_smoothing_length(&mut self, h: f32) {
        self.config.smoothing_length = h;
        self.config.rest_density = compute_lattice_rest_density(h, self.config.spacing);
        self.kernels = SphKernels::new(h);
        self.grid = SpatialHashGrid::new(h, self.particles.len().max(1024));
    }

    pub fn set_chamber(&mut self, chamber: Chamber) {
        self.bounds = chamber.compute_bounds();
        self.chamber = chamber;
    }

    pub fn apply_quality(&mut self, quality: SimulationQuality) {
        quality.apply(&mut self.config);
        self.kernels = SphKernels::new(self.config.smoothing_length);
        self.grid = SpatialHashGrid::new(self.config.smoothing_length, self.particles.len().max(1024));
    }

    pub fn add_particle(&mut self, p: Particle) {
        self.particles.push(p);
    }

    pub fn clear_particles(&mut self) {
        self.particles.clear();
        self.next_particle_id = 0;
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Emit new fluid particles from active faucets for dt
    pub fn emit_faucets(&mut self, dt: f32) -> usize {
        let mut count = 0;
        let mut new_particles = Vec::new();
        for faucet in &mut self.faucets {
            let spawned = faucet.emit(dt, self.config.spacing, &mut self.next_particle_id);
            count += spawned.len();
            new_particles.extend(spawned);
        }
        if !new_particles.is_empty() {
            self.particles.extend(new_particles);
        }
        count
    }

    /// Eliminate particles entering active drains
    pub fn drain_particles(&mut self) -> usize {
        if self.drains.is_empty() {
            return 0;
        }
        let bounds = self.bounds;
        let radius = self.config.particle_radius;
        let before = self.particles.len();
        self.particles.retain(|p| {
            !self.drains.iter().any(|d| d.should_drain(p.position, radius, &bounds))
        });
        before - self.particles.len()
    }

    /// Single simulation time step decomposed into substeps
    pub fn step(&mut self, dt: f32) {
        self.emit_faucets(dt);

        if self.particles.is_empty() {
            return;
        }

        let substeps = self.config.substeps.max(1);
        let sub_dt = dt / substeps as f32;

        for _ in 0..substeps {
            self.pbf_substep(sub_dt);
        }

        self.drain_particles();
    }

    fn pbf_substep(&mut self, dt: f32) {
        let n = self.particles.len();

        // 1. Resize working buffers if necessary
        if self.predicted_positions.len() != n {
            self.predicted_positions.resize(n, Vec3::ZERO);
            self.old_positions.resize(n, Vec3::ZERO);
            self.lambdas.resize(n, 0.0);
            self.delta_positions.resize(n, Vec3::ZERO);
            self.densities.resize(n, self.config.rest_density);
            self.xsph_velocities.resize(n, Vec3::ZERO);
        }

        let gravity = self.config.gravity;
        let bounds = self.bounds;
        let radius = self.config.particle_radius;
        let mouse_ray = self.mouse_ray;
        let mouse_mode = self.mouse_force_mode;
        let mouse_strength = self.mouse_force_strength;

        // 2. Predict positions with gravity & mouse interaction
        for (i, p) in self.particles.iter_mut().enumerate() {
            self.old_positions[i] = p.position;

            if p.particle_type == ParticleType::Boundary {
                self.predicted_positions[i] = p.position;
                continue;
            }

            // Mouse force
            let mut f_user = Vec3::ZERO;
            if mouse_mode != 0 {
                if let Some(ray) = mouse_ray {
                    let to_p = p.position - ray.origin;
                    let proj = to_p.dot(ray.direction);
                    if proj > 0.0 {
                        let closest = ray.origin + ray.direction * proj;
                        let diff = closest - p.position;
                        let dist = diff.length();
                        if dist < 0.6 && dist > 1e-4 {
                            let dir = diff / dist;
                            let factor = (1.0 - dist / 0.6) * mouse_strength;
                            f_user = dir * (factor * mouse_mode as f32);
                        }
                    }
                }
            }

            // Predict position
            let accel = gravity + f_user;
            p.velocity += accel * dt;

            let mut pred = p.position + p.velocity * dt;
            // Prevent leaving container
            self.chamber.resolve_collision(&mut pred, radius, &bounds);
            self.predicted_positions[i] = pred;
        }

        // 3. Rebuild spatial grid on predicted positions
        self.grid.build(&self.predicted_positions);

        let grid = &self.grid;
        let kernels = &self.kernels;
        let rho0 = self.config.rest_density;
        let h2 = kernels.h2;
        let h = kernels.h;
        let tensile_k = self.config.tensile_k;
        let surface_tension = self.config.surface_tension;
        let w_q = kernels.poly6((0.15 * h).powi(2)).max(1e-12);
        let inv_rho0 = 1.0 / rho0;
        let dt_ref = 1.0 / 60.0;
        let dt_ratio = (dt_ref / dt.max(1e-5)).max(1.0);
        let cfm_scale = dt_ratio * dt_ratio;
        let dt_factor = (dt / dt_ref).clamp(0.0001, 1.0);
        let relaxation_epsilon = 1200.0 * (0.075 / h).powi(2) * cfm_scale;
        let max_displacement = radius * 0.35 * dt_factor; // Scales with time step to prevent slow-mo explosions

        // 4. PBF Incompressibility Solver Iterations
        for _ in 0..self.config.solver_iterations {
            let pred_pos = &self.predicted_positions;

            // 4a. Parallel computation of density and constraint lambda_i
            self.lambdas
                .par_iter_mut()
                .zip(self.densities.par_iter_mut())
                .enumerate()
                .for_each(|(i, (lambda_out, density_out))| {
                    let pos_i = pred_pos[i];
                    let mut density = kernels.poly6(0.0);
                    let mut grad_ci = Vec3::ZERO;
                    let mut sum_grad_sq = 0.0f32;

                    grid.for_each_neighbor(pos_i, |j| {
                        if i != j {
                            let pos_j = pred_pos[j];
                            let r_vec = pos_i - pos_j;
                            let r2 = r_vec.length_squared();

                            if r2 < h2 && r2 > 1e-12 {
                                let r = r2.sqrt();
                                density += kernels.poly6(r2);

                                let grad_w = kernels.spiky_gradient(r_vec, r);
                                let grad_j = grad_w * inv_rho0;
                                grad_ci += grad_j;
                                sum_grad_sq += grad_j.length_squared();
                            }
                        }
                    });

                    sum_grad_sq += grad_ci.length_squared();
                    *density_out = density;

                    // Density constraint C_i = rho / rho0 - 1.0
                    let c_i = (density * inv_rho0 - 1.0).max(0.0);
                    *lambda_out = -c_i / (sum_grad_sq + relaxation_epsilon);
                });

            let lambdas = &self.lambdas;

            // 4b. Parallel computation of position displacement delta_p_i
            self.delta_positions
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, delta_out)| {
                    let pos_i = pred_pos[i];
                    let lambda_i = lambdas[i];
                    let mut delta = Vec3::ZERO;

                    grid.for_each_neighbor(pos_i, |j| {
                        if i != j {
                            let pos_j = pred_pos[j];
                            let r_vec = pos_i - pos_j;
                            let r2 = r_vec.length_squared();

                            if r2 < h2 && r2 > 1e-12 {
                                let r = r2.sqrt();
                                let lambda_j = lambdas[j];
                                let lambda_sum = lambda_i + lambda_j;

                                // Tensile instability correction (only active during compression)
                                let w_r = kernels.poly6(r2);
                                let s_corr = if lambda_sum < -1e-6 {
                                    (-tensile_k * (w_r / w_q).powi(4) * (h * h)) / cfm_scale
                                } else {
                                    0.0
                                };

                                let grad_w = kernels.spiky_gradient(r_vec, r);
                                delta += grad_w * ((lambda_sum + s_corr) * inv_rho0);

                                if surface_tension > 0.0 {
                                    let cohesion_factor = (w_r * surface_tension * 0.05) / (rho0 * cfm_scale);
                                    delta -= (r_vec / r) * cohesion_factor;
                                }
                            }
                        }
                    });

                    // Anti-trampoline: clamp displacement magnitude per iteration
                    let delta_len = delta.length();
                    if delta_len > max_displacement {
                        delta *= max_displacement / delta_len;
                    }

                    *delta_out = delta;
                });

            // 4c. Apply displacements and project collisions onto predicted positions
            let colliders = &self.colliders;
            let chamber = &self.chamber;
            for i in 0..n {
                self.predicted_positions[i] += self.delta_positions[i];

                let p_pos = &mut self.predicted_positions[i];

                // Chamber collision projection
                chamber.resolve_collision(p_pos, radius, &bounds);

                // Colliders (sphere / box obstacles)
                let mut dummy_vel = Vec3::ZERO;
                for collider in colliders {
                    collider.resolve(p_pos, &mut dummy_vel, radius);
                }
            }
        }

        // 5. Update velocities from position displacement with INELASTIC boundary response
        let inv_dt = 1.0 / dt;
        let max_vel = self.config.max_velocity;
        let restitution = self.config.restitution;
        let friction = self.config.boundary_friction;

        for (i, p) in self.particles.iter_mut().enumerate() {
            if p.particle_type == ParticleType::Boundary {
                continue;
            }

            let mut new_pos = self.predicted_positions[i];
            let old_pos = self.old_positions[i];

            let mut vel = (new_pos - old_pos) * inv_dt;

            // Inelastic chamber boundary response
            self.chamber.resolve_velocity(
                new_pos,
                old_pos,
                &mut vel,
                dt,
                radius,
                &bounds,
                restitution,
                friction,
            );

            // Obstacle Colliders (Sphere, Box, Cylinder, OBJ Mesh)
            for collider in &self.colliders {
                collider.resolve(&mut new_pos, &mut vel, radius);
            }

            // Velocity clamping
            let speed = vel.length();
            if speed > max_vel {
                vel = (vel / speed) * max_vel;
            }

            p.velocity = vel;
            p.position = new_pos;
            p.density = self.densities[i];
            p.pressure = (self.densities[i] - rho0).max(0.0);
        }

        // 6. Normalized XSPH Viscosity (smooth moving average of local velocities)
        let visc_c = self.config.viscosity.clamp(0.0, 0.95);
        if visc_c > 0.0 {
            let particles_slice = &self.particles;

            self.xsph_velocities
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, xsph_out)| {
                    let pos_i = particles_slice[i].position;
                    let vel_i = particles_slice[i].velocity;

                    let mut sum_w = kernels.poly6(0.0);
                    let mut weighted_v = vel_i * sum_w;

                    grid.for_each_neighbor(pos_i, |j| {
                        if i != j {
                            let pos_j = particles_slice[j].position;
                            let r2 = (pos_i - pos_j).length_squared();
                            if r2 < h2 {
                                let w = kernels.poly6(r2);
                                sum_w += w;
                                weighted_v += particles_slice[j].velocity * w;
                            }
                        }
                    });

                    let avg_v = weighted_v / sum_w;
                    *xsph_out = vel_i.lerp(avg_v, visc_c);
                });

            for (i, p) in self.particles.iter_mut().enumerate() {
                p.velocity = self.xsph_velocities[i];
            }
        }
    }
}
