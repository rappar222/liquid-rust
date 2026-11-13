use crate::math::{Aabb, Ray};
use crate::sph::chamber::Chamber;
use crate::sph::collider::Collider;
use crate::sph::kernel::SphKernels;
use crate::sph::particle::{Particle, ParticleType};
use crate::sph::spatial::SpatialHashGrid;
use glam::Vec3;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 384;

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
        self.name_for(false)
    }

    pub fn name_for(&self, is_2d: bool) -> &'static str {
        match self {
            SimulationQuality::Draft => {
                if is_2d {
                    "Draft (Fast, ~300 particles)"
                } else {
                    "Draft (Fast, ~2.5k particles)"
                }
            }
            SimulationQuality::Medium => {
                if is_2d {
                    "Medium (Balanced, ~500 particles)"
                } else {
                    "Medium (Balanced, ~5.5k particles)"
                }
            }
            SimulationQuality::High => {
                if is_2d {
                    "High (Smooth, ~1.5k particles)"
                } else {
                    "High (Smooth, ~18k particles)"
                }
            }
            SimulationQuality::Ultra => {
                if is_2d {
                    "Ultra (High-Res, ~4k particles)"
                } else {
                    "Ultra (High-Res, ~50k particles)"
                }
            }
            SimulationQuality::Cinematic => {
                if is_2d {
                    "Cinematic (~10k particles)"
                } else {
                    "Cinematic (~120k particles)"
                }
            }
            SimulationQuality::Extreme => {
                if is_2d {
                    "Extreme (~30k particles)"
                } else {
                    "Extreme (1,000,000+ particles, GPU Max)"
                }
            }
            SimulationQuality::Custom => "Custom",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            SimulationQuality::Draft => "draft",
            SimulationQuality::Medium => "medium",
            SimulationQuality::High => "high",
            SimulationQuality::Ultra => "ultra",
            SimulationQuality::Cinematic => "cinematic",
            SimulationQuality::Extreme => "extreme",
            SimulationQuality::Custom => "custom",
        }
    }

    pub fn apply(&self, config: &mut SolverConfig, is_2d: bool) {
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
        if is_2d {
            config.rest_density =
                compute_lattice_rest_density_2d(config.smoothing_length, config.spacing);
        } else {
            config.rest_density =
                compute_lattice_rest_density(config.smoothing_length, config.spacing);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolverConfig {
    pub particle_radius: f32,     // Physical radius for collisions
    pub spacing: f32,             // Initial lattice spacing
    pub smoothing_length: f32,    // SPH kernel radius h
    pub rest_density: f32,        // Target rest density rho0
    pub solver_iterations: usize, // PBF solver iterations per substep (typically 3-4)
    pub substeps: usize,          // Number of substeps per frame (typically 2-3)
    pub viscosity: f32,           // Normalized XSPH viscosity [0.0, 1.0]
    pub surface_tension: f32,     // Free-surface cohesion coefficient
    pub vorticity: f32,           // Vorticity confinement strength for natural turbulent eddies
    pub restitution: f32,         // Boundary restitution (0.0 for inelastic water, no bouncing)
    pub boundary_friction: f32,   // Wall friction
    pub gravity: Vec3,
    pub tensile_k: f32,           // Tensile instability correction coefficient
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
            viscosity: 0.12,
            surface_tension: 0.06,
            vorticity: 0.045,
            restitution: 0.02, // Inelastic fluid impact: practically 0 bounce
            boundary_friction: 0.95,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            tensile_k: 0.001,
            max_velocity: 12.0,
        }
    }
}

/// Compute exact rest density on regular cubic lattice with given spacing (3D)
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

/// Compute exact rest density on regular planar square lattice with given spacing (2D)
pub fn compute_lattice_rest_density_2d(h: f32, spacing: f32) -> f32 {
    let kernels = SphKernels::new(h);
    let mut rho0 = kernels.poly6(0.0);
    let steps = (h / spacing).ceil() as i32;
    let h2 = h * h;

    for ix in -steps..=steps {
        for iy in -steps..=steps {
            if ix == 0 && iy == 0 {
                continue;
            }
            let r2 = (ix as f32 * spacing).powi(2) + (iy as f32 * spacing).powi(2);
            if r2 < h2 {
                rho0 += kernels.poly6(r2);
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
    pub is_2d: bool,

    // Working buffers for PBF (Position-Based Fluids)
    predicted_positions: Vec<Vec3>,
    old_positions: Vec<Vec3>,
    lambdas: Vec<f32>,
    delta_positions: Vec<Vec3>,
    densities: Vec<f32>,
    xsph_velocities: Vec<Vec3>,
    vorticities: Vec<Vec3>,

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
            is_2d: false,
            predicted_positions: Vec::new(),
            old_positions: Vec::new(),
            lambdas: Vec::new(),
            delta_positions: Vec::new(),
            densities: Vec::new(),
            xsph_velocities: Vec::new(),
            vorticities: Vec::new(),
            mouse_ray: None,
            mouse_force_strength: 60.0,
            mouse_force_mode: 0,
        }
    }

    pub fn set_2d(&mut self, is_2d: bool) {
        self.is_2d = is_2d;
        if is_2d {
            self.config.rest_density =
                compute_lattice_rest_density_2d(self.config.smoothing_length, self.config.spacing);
            for p in &mut self.particles {
                p.position.z = 0.0;
                p.velocity.z = 0.0;
            }
        } else {
            self.config.rest_density =
                compute_lattice_rest_density(self.config.smoothing_length, self.config.spacing);
        }
    }

    pub fn set_smoothing_length(&mut self, h: f32) {
        self.config.smoothing_length = h;
        if self.is_2d {
            self.config.rest_density = compute_lattice_rest_density_2d(h, self.config.spacing);
        } else {
            self.config.rest_density = compute_lattice_rest_density(h, self.config.spacing);
        }
        self.kernels = SphKernels::new(h);
        self.grid.set_cell_size(h);
    }

    pub fn set_chamber(&mut self, chamber: Chamber) {
        self.bounds = chamber.compute_bounds();
        self.chamber = chamber;
    }

    pub fn apply_quality(&mut self, quality: SimulationQuality) {
        quality.apply(&mut self.config, self.is_2d);
        self.kernels = SphKernels::new(self.config.smoothing_length);
        self.grid.set_cell_size(self.config.smoothing_length);
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
        for faucet in &mut self.faucets {
            let spawned = faucet.emit(dt, self.config.spacing, &mut self.next_particle_id);
            count += spawned.len();
            if self.is_2d {
                for mut p in spawned {
                    p.position.z = 0.0;
                    p.velocity.z = 0.0;
                    self.particles.push(p);
                }
            } else {
                self.particles.extend(spawned);
            }
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
            !self
                .drains
                .iter()
                .any(|d| d.should_drain(p.position, radius, &bounds))
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
        let use_parallel = n >= PAR_THRESHOLD;

        // 1. Resize working buffers if necessary (zero heap allocations after first frame)
        if self.predicted_positions.len() != n {
            self.predicted_positions.resize(n, Vec3::ZERO);
            self.old_positions.resize(n, Vec3::ZERO);
            self.lambdas.resize(n, 0.0);
            self.delta_positions.resize(n, Vec3::ZERO);
            self.densities.resize(n, self.config.rest_density);
            self.xsph_velocities.resize(n, Vec3::ZERO);
            self.vorticities.resize(n, Vec3::ZERO);
        }

        let gravity = self.config.gravity;
        let bounds = self.bounds;
        let radius = self.config.particle_radius;
        let mouse_ray = self.mouse_ray;
        let mouse_mode = self.mouse_force_mode;
        let mouse_strength = self.mouse_force_strength;
        let is_2d = self.is_2d;
        let chamber = &self.chamber;

        // Precompute planar mouse hit in 2D mode
        let mouse_hit_2d = if is_2d && mouse_mode != 0 {
            mouse_ray.and_then(|ray| ray.intersect_plane(Vec3::ZERO, Vec3::Z).map(|t| ray.at(t)))
        } else {
            None
        };

        // 2. Predict positions with gravity & mouse interaction
        let predict_one = |(p, (old_out, pred_out)): (&mut Particle, (&mut Vec3, &mut Vec3))| {
            *old_out = p.position;

            if p.particle_type == ParticleType::Boundary {
                *pred_out = p.position;
                return;
            }

            let mut f_user = Vec3::ZERO;
            if mouse_mode != 0 {
                if is_2d {
                    if let Some(hit) = mouse_hit_2d {
                        let diff_2d = Vec3::new(hit.x - p.position.x, hit.y - p.position.y, 0.0);
                        let dist = diff_2d.length();
                        if dist < 0.8 && dist > 1e-4 {
                            let dir = diff_2d / dist;
                            let factor = (1.0 - dist / 0.8) * mouse_strength;
                            f_user = dir * (factor * mouse_mode as f32);
                        }
                    }
                } else if let Some(ray) = mouse_ray {
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

            let accel = gravity + f_user;
            p.velocity += accel * dt;

            let mut pred = p.position + p.velocity * dt;
            if is_2d {
                pred.z = 0.0;
                p.velocity.z = 0.0;
            }
            chamber.resolve_collision(&mut pred, radius, &bounds);
            if is_2d {
                pred.z = 0.0;
            }
            *pred_out = pred;
        };

        if use_parallel {
            self.particles
                .par_iter_mut()
                .zip(
                    self.old_positions
                        .par_iter_mut()
                        .zip(self.predicted_positions.par_iter_mut()),
                )
                .for_each(predict_one);
        } else {
            self.particles
                .iter_mut()
                .zip(
                    self.old_positions
                        .iter_mut()
                        .zip(self.predicted_positions.iter_mut()),
                )
                .for_each(predict_one);
        }

        // 3. Rebuild spatial grid on predicted positions (2D-aware zero-alloc hash grid)
        self.grid
            .build_from_positions(&self.predicted_positions, is_2d);

        let grid = &self.grid;
        let kernels = &self.kernels;
        let rho0 = self.config.rest_density;
        let h2 = kernels.h2;
        let h = kernels.h;
        let inv_h = 1.0 / h;
        let tensile_k = self.config.tensile_k;
        let surface_tension = self.config.surface_tension;
        let w_q = kernels.poly6((0.15 * h).powi(2)).max(1e-12);
        let inv_rho0 = 1.0 / rho0;
        let dt_ref = 1.0 / 60.0;
        let dt_ratio = (dt_ref / dt.max(1e-5)).max(1.0);
        let cfm_scale = dt_ratio * dt_ratio;
        let dt_factor = (dt / dt_ref).clamp(0.0001, 1.0);
        let relaxation_epsilon = 1200.0 * (0.075 / h).powi(2) * cfm_scale;
        let max_displacement = (radius * 0.45).max(h * 0.25) * dt_factor;
        let spacing = self.config.spacing;
        let colliders = &self.colliders;

        // 4. PBF Incompressibility Solver Iterations
        for _ in 0..self.config.solver_iterations {
            let pred_pos = &self.predicted_positions;

            // 4a. Computation of density and constraint lambda_i
            let compute_lambda = |(i, (lambda_out, density_out)): (usize, (&mut f32, &mut f32))| {
                let pos_i = pred_pos[i];
                let mut density = kernels.poly6(0.0);
                let mut grad_ci = Vec3::ZERO;
                let mut sum_grad_sq = 0.0f32;

                grid.for_each_neighbor_mode(pos_i, is_2d, |j| {
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

                let (mut psi_b, mut grad_psi_b) =
                    chamber.boundary_contribution(pos_i, radius, spacing, h, &bounds, is_2d);
                for collider in colliders {
                    collider.fold_boundary_contribution(
                        pos_i,
                        radius,
                        spacing,
                        h,
                        is_2d,
                        &mut psi_b,
                        &mut grad_psi_b,
                    );
                }
                if is_2d {
                    grad_psi_b.z = 0.0;
                }

                density += rho0 * psi_b;
                grad_ci += grad_psi_b;
                sum_grad_sq += grad_psi_b.length_squared() + grad_ci.length_squared();
                *density_out = density;

                let c_i = (density * inv_rho0 - 1.0).max(0.0);
                *lambda_out = -c_i / (sum_grad_sq + relaxation_epsilon);
            };

            if use_parallel {
                self.lambdas
                    .par_iter_mut()
                    .zip(self.densities.par_iter_mut())
                    .enumerate()
                    .for_each(compute_lambda);
            } else {
                self.lambdas
                    .iter_mut()
                    .zip(self.densities.iter_mut())
                    .enumerate()
                    .for_each(compute_lambda);
            }

            let lambdas = &self.lambdas;
            let densities = &self.densities;

            // 4b. Computation of position displacement delta_p_i + Akinci free-surface cohesion
            let compute_delta = |(i, delta_out): (usize, &mut Vec3)| {
                let pos_i = pred_pos[i];
                let lambda_i = lambdas[i];
                let rho_i = densities[i];
                let mut delta = Vec3::ZERO;

                let (mut psi_b, mut grad_psi_b) =
                    chamber.boundary_contribution(pos_i, radius, spacing, h, &bounds, is_2d);
                for collider in colliders {
                    collider.fold_boundary_contribution(
                        pos_i,
                        radius,
                        spacing,
                        h,
                        is_2d,
                        &mut psi_b,
                        &mut grad_psi_b,
                    );
                }
                if is_2d {
                    grad_psi_b.z = 0.0;
                }
                let wall_cohesion_scale = (1.0 - psi_b * 3.5).clamp(0.08, 1.0);

                grid.for_each_neighbor_mode(pos_i, is_2d, |j| {
                    if i != j {
                        let pos_j = pred_pos[j];
                        let r_vec = pos_i - pos_j;
                        let r2 = r_vec.length_squared();

                        if r2 < h2 && r2 > 1e-12 {
                            let r = r2.sqrt();
                            let lambda_j = lambdas[j];
                            let lambda_sum = lambda_i + lambda_j;

                            // Tensile instability correction (only active during compression)
                            let s_corr = if lambda_sum < -1e-6 {
                                let w_r = kernels.poly6(r2);
                                let ratio = (w_r / w_q).min(1.5);
                                (-tensile_k * ratio.powi(4) * (h * h)) / cfm_scale
                            } else {
                                0.0
                            };

                            let grad_w = kernels.spiky_gradient(r_vec, r);
                            delta += grad_w * ((lambda_sum + s_corr) * inv_rho0);

                            // Scale-invariant Akinci bell-spline cohesion weighted toward free surface
                            if surface_tension > 0.0 {
                                let q = r * inv_h;
                                let one_minus_q = 1.0 - q;
                                let c_q = 64.0 * (q * q * q) * (one_minus_q * one_minus_q * one_minus_q);
                                let surf_weight =
                                    (2.0 - (rho_i + densities[j]) * inv_rho0).clamp(0.15, 1.40);
                                let cohesion = (c_q
                                    * surf_weight
                                    * surface_tension
                                    * wall_cohesion_scale
                                    * 0.016
                                    * h)
                                    / cfm_scale;
                                delta -= (r_vec / r) * cohesion;
                            }
                        }
                    }
                });

                // Mirror hydrostatic boundary pressure repulsion (prevents pinning/clustering on walls & floors)
                if lambda_i < -1e-7 {
                    delta += grad_psi_b * (2.0 * lambda_i);
                }

                // Anti-trampoline: clamp displacement magnitude per iteration
                let delta_len = delta.length();
                if delta_len > max_displacement {
                    delta *= max_displacement / delta_len;
                }

                *delta_out = delta;
            };

            if use_parallel {
                self.delta_positions
                    .par_iter_mut()
                    .enumerate()
                    .for_each(compute_delta);
            } else {
                self.delta_positions
                    .iter_mut()
                    .enumerate()
                    .for_each(compute_delta);
            }

            // 4c. Apply displacements and project collisions onto predicted positions
            let deltas = &self.delta_positions;
            let relaxation_omega = 0.65;
            let apply_collide = |(i, p_pos): (usize, &mut Vec3)| {
                *p_pos += deltas[i] * relaxation_omega;
                if is_2d {
                    p_pos.z = 0.0;
                }

                chamber.resolve_collision(p_pos, radius, &bounds);

                let mut dummy_vel = Vec3::ZERO;
                for collider in colliders {
                    collider.resolve(p_pos, &mut dummy_vel, radius);
                }

                if is_2d {
                    p_pos.z = 0.0;
                }
            };

            if use_parallel {
                self.predicted_positions
                    .par_iter_mut()
                    .enumerate()
                    .for_each(apply_collide);
            } else {
                self.predicted_positions
                    .iter_mut()
                    .enumerate()
                    .for_each(apply_collide);
            }
        }

        // 5. Update velocities from position displacement with INELASTIC boundary response
        let inv_dt = 1.0 / dt;
        let max_vel = self.config.max_velocity;
        let restitution = self.config.restitution;
        let friction = self.config.boundary_friction;
        let pred_pos = &self.predicted_positions;
        let old_pos_slice = &self.old_positions;
        let densities = &self.densities;

        let update_vel = |(i, p): (usize, &mut Particle)| {
            if p.particle_type == ParticleType::Boundary {
                return;
            }

            let mut new_pos = pred_pos[i];
            let old_pos = old_pos_slice[i];
            let mut vel = (new_pos - old_pos) * inv_dt;

            chamber.resolve_velocity(
                new_pos,
                old_pos,
                &mut vel,
                dt,
                radius,
                &bounds,
                restitution,
                friction,
            );

            for collider in colliders {
                collider.resolve(&mut new_pos, &mut vel, radius);
            }

            if is_2d {
                new_pos.z = 0.0;
                vel.z = 0.0;
            }

            let speed = vel.length();
            if speed > max_vel {
                vel = (vel / speed) * max_vel;
            }

            p.velocity = vel;
            p.position = new_pos;
            p.density = densities[i];
            p.pressure = (densities[i] - rho0).max(0.0);
        };

        if use_parallel {
            self.particles
                .par_iter_mut()
                .enumerate()
                .for_each(update_vel);
        } else {
            self.particles.iter_mut().enumerate().for_each(update_vel);
        }

        // 6. Fused Normalized XSPH Viscosity + Velocity Curl (Vorticity) Pass
        let visc_c = self.config.viscosity.clamp(0.0, 0.95);
        let vort_eps = self.config.vorticity.max(0.0);

        if visc_c > 0.0 || vort_eps > 1e-5 {
            let particles_slice = &self.particles;

            let compute_xsph_and_curl =
                |(i, (xsph_out, vort_out)): (usize, (&mut Vec3, &mut Vec3))| {
                    let pos_i = particles_slice[i].position;
                    let vel_i = particles_slice[i].velocity;

                    let mut sum_w = kernels.poly6(0.0);
                    let mut weighted_v = vel_i * sum_w;
                    let mut curl = Vec3::ZERO;

                    grid.for_each_neighbor_mode(pos_i, is_2d, |j| {
                        if i != j {
                            let pos_j = particles_slice[j].position;
                            let r_vec = pos_i - pos_j;
                            let r2 = r_vec.length_squared();
                            if r2 < h2 && r2 > 1e-12 {
                                let vel_j = particles_slice[j].velocity;
                                if visc_c > 0.0 {
                                    let w = kernels.poly6(r2);
                                    sum_w += w;
                                    weighted_v += vel_j * w;
                                }
                                if vort_eps > 1e-5 {
                                    let r = r2.sqrt();
                                    let grad_w = kernels.spiky_gradient(r_vec, r);
                                    curl += (vel_i - vel_j).cross(grad_w) * inv_rho0;
                                }
                            }
                        }
                    });

                    *xsph_out = if visc_c > 0.0 {
                        let avg_v = weighted_v / sum_w;
                        vel_i.lerp(avg_v, visc_c)
                    } else {
                        vel_i
                    };
                    *vort_out = curl;
                };

            if use_parallel {
                self.xsph_velocities
                    .par_iter_mut()
                    .zip(self.vorticities.par_iter_mut())
                    .enumerate()
                    .for_each(compute_xsph_and_curl);
            } else {
                self.xsph_velocities
                    .iter_mut()
                    .zip(self.vorticities.iter_mut())
                    .enumerate()
                    .for_each(compute_xsph_and_curl);
            }

            // 7. Vorticity Confinement Force (Macklin & Müller 2013) + XSPH Commit
            if vort_eps > 1e-5 {
                let vorts = &self.vorticities;
                let xsph_vels = &self.xsph_velocities;
                let particles_ro = &self.particles;

                // Reuse delta_positions buffer temporarily to hold final velocities
                let commit_vort = |(i, v_final_out): (usize, &mut Vec3)| {
                    let mut v_base = xsph_vels[i];
                    let speed = v_base.length();
                    // Smooth flow gate: resting pools (speed < 0.08 m/s) receive 0 vorticity noise
                    let flow_gate = ((speed - 0.08) / 0.35).clamp(0.0, 1.0);

                    if flow_gate > 0.0 {
                        let pos_i = particles_ro[i].position;
                        let omega_i = vorts[i];
                        let omega_i_mag = omega_i.length();

                        if omega_i_mag > 1e-4 {
                            let mut eta = Vec3::ZERO;
                            grid.for_each_neighbor_mode(pos_i, is_2d, |j| {
                                if i != j {
                                    let r_vec = pos_i - particles_ro[j].position;
                                    let r2 = r_vec.length_squared();
                                    if r2 < h2 && r2 > 1e-12 {
                                        let r = r2.sqrt();
                                        let omega_j_mag = vorts[j].length();
                                        let grad_w = kernels.spiky_gradient(r_vec, r);
                                        eta += grad_w * ((omega_j_mag - omega_i_mag) * inv_rho0);
                                    }
                                }
                            });

                            let eta_len = eta.length();
                            if eta_len > 1e-5 {
                                let n_dir = eta / eta_len;
                                let f_vort = n_dir.cross(omega_i) * (vort_eps * h * 8.0 * flow_gate);
                                let dv = (f_vort * dt).clamp_length_max(speed * 0.22);
                                v_base += dv;
                            }
                        }
                    }

                    if is_2d {
                        v_base.z = 0.0;
                    }
                    let final_speed = v_base.length();
                    if final_speed > max_vel {
                        v_base *= max_vel / final_speed;
                    }
                    *v_final_out = v_base;
                };

                if use_parallel {
                    self.delta_positions
                        .par_iter_mut()
                        .enumerate()
                        .for_each(commit_vort);
                } else {
                    self.delta_positions
                        .iter_mut()
                        .enumerate()
                        .for_each(commit_vort);
                }

                for (i, p) in self.particles.iter_mut().enumerate() {
                    if p.particle_type != ParticleType::Boundary {
                        p.velocity = self.delta_positions[i];
                    }
                }
            } else {
                for (i, p) in self.particles.iter_mut().enumerate() {
                    if p.particle_type != ParticleType::Boundary {
                        p.velocity = self.xsph_velocities[i];
                        if is_2d {
                            p.velocity.z = 0.0;
                        }
                    }
                }
            }
        }
    }
}
