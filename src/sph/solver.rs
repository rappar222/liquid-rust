use crate::math::{Aabb, Ray};
use crate::sph::chamber::Chamber;
use crate::sph::collider::Collider;
use crate::sph::kernel::SphKernels;
use crate::sph::particle::{Particle, ParticleType};
use crate::sph::spatial::SpatialHashGrid;
use crate::sph::substance::MaterialType;
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

            // Predict position with material buoyancy & gas/fire upward draft
            let mut accel = gravity + f_user;
            if p.material.is_gas() {
                accel.y += p.material.buoyancy();
            }
            if p.material == MaterialType::Fire {
                let seed = p.id.wrapping_mul(1664525).wrapping_add(1013904223);
                let fx = ((seed & 0xFF) as f32 / 255.0 - 0.5) * 6.0;
                let fz = (((seed >> 8) & 0xFF) as f32 / 255.0 - 0.5) * 6.0;
                accel.x += fx;
                accel.z += fz;
            }
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
        let dt_factor = (dt / 0.016).clamp(0.001, 1.0);
        let relaxation_epsilon = 1200.0 * (0.075 / h).powi(2);
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
                                    -tensile_k * (w_r / w_q).powi(4) * (h * h) * dt_factor
                                } else {
                                    0.0
                                };

                                let grad_w = kernels.spiky_gradient(r_vec, r);
                                delta += grad_w * ((lambda_sum + s_corr) * inv_rho0);

                                // Surface tension cohesion: scales with dt^2 so slow-motion doesn't over-amplify it into cubes
                                if surface_tension > 0.0 {
                                    let cohesion_factor = (w_r * surface_tension * 0.05 * dt_factor * dt_factor) / rho0;
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

        // 7. Thermodynamics, Combustion, Boiling & Phase Changes
        self.step_thermodynamics(dt);
    }

    /// Step heat conduction, combustion (lighting gas into fire), and water boiling into steam
    pub fn step_thermodynamics(&mut self, dt: f32) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }

        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        self.grid.build(&positions);

        let h2 = self.kernels.h2;
        let w_zero = self.kernels.poly6(0.0).max(1e-12);

        // Handle torch / lighter mouse mode (mode 2)
        if self.mouse_force_mode == 2 {
            self.apply_torch(self.mouse_ray, 0.45);
        }

        let mut new_temps = Vec::with_capacity(n);
        let mut new_materials = Vec::with_capacity(n);
        let mut new_lifes = Vec::with_capacity(n);
        let mut blast_impulses = vec![Vec3::ZERO; n];

        for i in 0..n {
            let p = &self.particles[i];
            let pos_i = p.position;
            let mut temp_i = p.temperature;
            let mut mat_i = p.material;
            let mut life_i = p.life;

            // Ambient heat dissipation towards 20°C
            let ambient_rate = match mat_i {
                MaterialType::Steam => 0.12,
                MaterialType::Smoke => 0.08,
                MaterialType::Fire => 0.06,
                _ => 0.02,
            };
            temp_i += (20.0 - temp_i) * (ambient_rate * dt).min(0.2);

            // Heat conduction with neighbors & fire thermal radiation
            self.grid.for_each_neighbor(pos_i, |j| {
                if i != j {
                    let pos_j = self.particles[j].position;
                    let r2 = (pos_i - pos_j).length_squared();
                    if r2 < h2 {
                        let temp_j = self.particles[j].temperature;
                        let mat_j = self.particles[j].material;
                        let w = self.kernels.poly6(r2) / w_zero;

                        let conduct_k = if mat_j == MaterialType::Fire {
                            16.0 // powerful radiative heating from flames!
                        } else if mat_i == MaterialType::Water && mat_j == MaterialType::Fire {
                            24.0 // rapid boiling heat absorption!
                        } else {
                            2.5  // regular thermal diffusion
                        };

                        let delta_t = (temp_j - temp_i) * (conduct_k * w * dt).clamp(-0.4, 0.4);
                        temp_i += delta_t;
                    }
                }
            });

            // Phase changes, Combustion & Boiling
            match mat_i {
                MaterialType::Gas => {
                    // Ignition: Gas exposed to heat or flame ignites violently!
                    if temp_i >= 220.0 {
                        mat_i = MaterialType::Fire;
                        temp_i = 780.0;
                        life_i = 1.3 + (p.id % 40) as f32 * 0.01;
                        // Explosive upward / outward blast
                        let blast_up = 3.5 + (p.id % 20) as f32 * 0.1;
                        blast_impulses[i] = Vec3::new(0.0, blast_up, 0.0);
                    }
                }
                MaterialType::Water => {
                    // Boiling: Water heated to >= 100°C boils into Steam!
                    if temp_i >= 100.0 {
                        mat_i = MaterialType::Steam;
                        temp_i = 108.0;
                        life_i = 3.8 + (p.id % 50) as f32 * 0.02;
                        blast_impulses[i] = Vec3::new(0.0, 2.5, 0.0);
                    }
                }
                MaterialType::Alcohol => {
                    // Flammable alcohol ignites into fire when heated
                    if temp_i >= 240.0 {
                        mat_i = MaterialType::Fire;
                        temp_i = 680.0;
                        life_i = 1.5;
                        blast_impulses[i] = Vec3::new(0.0, 2.0, 0.0);
                    }
                }
                MaterialType::Fire => {
                    life_i -= dt;
                    temp_i = 750.0 * (life_i / 1.3).clamp(0.2, 1.0);
                    if life_i <= 0.0 {
                        // Extinguished fire becomes rising smoke
                        mat_i = MaterialType::Smoke;
                        temp_i = 110.0;
                        life_i = 3.0;
                    }
                }
                MaterialType::Steam => {
                    life_i -= dt;
                    if life_i <= 0.0 || temp_i < 38.0 {
                        // Condensation: steam condenses back into liquid water
                        mat_i = MaterialType::Water;
                        temp_i = 35.0;
                    }
                }
                MaterialType::Smoke => {
                    life_i -= dt;
                }
                _ => {}
            }

            new_temps.push(temp_i);
            new_materials.push(mat_i);
            new_lifes.push(life_i);
        }

        for i in 0..n {
            self.particles[i].temperature = new_temps[i];
            self.particles[i].material = new_materials[i];
            self.particles[i].life = new_lifes[i];
            self.particles[i].velocity += blast_impulses[i];
        }
    }

    /// User spark: Instantly ignite all flammable gas and alcohol into roaring flames!
    pub fn ignite_all_gas(&mut self) {
        for p in &mut self.particles {
            if p.material == MaterialType::Gas || p.material == MaterialType::Alcohol {
                p.material = MaterialType::Fire;
                p.temperature = 780.0;
                p.life = 1.4;
                p.velocity.y += 4.5;
            }
        }
    }

    /// User heat source: Heat water to boil directly into rising steam clouds!
    pub fn boil_water(&mut self) {
        for p in &mut self.particles {
            if p.material == MaterialType::Water {
                p.material = MaterialType::Steam;
                p.temperature = 115.0;
                p.life = 4.0;
                p.velocity.y += 2.5;
            }
        }
    }

    /// User cooling: Cool all fluid to ambient 20°C and condense all steam
    pub fn cool_all(&mut self) {
        for p in &mut self.particles {
            p.temperature = 20.0;
            if p.material == MaterialType::Steam || p.material == MaterialType::Fire {
                p.material = MaterialType::Water;
            }
        }
    }

    /// Torch tool: Apply heat and ignition near cursor ray
    pub fn apply_torch(&mut self, ray: Option<Ray>, radius: f32) {
        if let Some(r) = ray {
            for p in &mut self.particles {
                let to_p = p.position - r.origin;
                let proj = to_p.dot(r.direction);
                if proj > 0.0 {
                    let closest = r.origin + r.direction * proj;
                    let dist = (closest - p.position).length();
                    if dist < radius {
                        p.temperature = 750.0;
                        if p.material == MaterialType::Gas || p.material == MaterialType::Alcohol {
                            p.material = MaterialType::Fire;
                            p.life = 1.3;
                            p.velocity.y += 4.0;
                        } else if p.material == MaterialType::Water {
                            p.material = MaterialType::Steam;
                            p.life = 3.5;
                            p.velocity.y += 2.5;
                        }
                    }
                }
            }
        }
    }
}
