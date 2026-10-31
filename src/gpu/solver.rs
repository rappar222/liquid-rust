use crate::math::{Aabb, Ray};
use crate::sph::collider::Collider;
use crate::sph::particle::{Particle, ParticleType};
use crate::sph::solver::SolverConfig;
use glam::Vec3;
use std::f32::consts::PI;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuUniforms {
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub gravity: [f32; 4],
    pub grid_dim: [u32; 4],
    pub mouse_origin: [f32; 4],
    pub mouse_dir: [f32; 4],
    pub collider_pos: [f32; 4],
    pub kernel_params: [f32; 4],
    pub kernel_factors: [f32; 4],
    pub solver_params: [f32; 4],
    pub fluid_params: [f32; 4],
    pub counts: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub pos_type: [f32; 4], // xyz = position, w = type (0.0 fluid, 1.0 boundary)
    pub vel_dens: [f32; 4], // xyz = velocity, w = density
}

pub struct GpuSphSolver {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub device_name: String,

    pipeline_predict: wgpu::ComputePipeline,
    pipeline_clear_grid: wgpu::ComputePipeline,
    pipeline_insert_particles: wgpu::ComputePipeline,
    pipeline_density_lambda: wgpu::ComputePipeline,
    pipeline_delta_positions: wgpu::ComputePipeline,
    pipeline_apply_delta: wgpu::ComputePipeline,
    pipeline_update_velocity: wgpu::ComputePipeline,

    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,

    uniform_buf: wgpu::Buffer,
    particle_buf: Option<wgpu::Buffer>,
    pred_pos_buf: Option<wgpu::Buffer>,
    old_pos_buf: Option<wgpu::Buffer>,
    lambdas_buf: Option<wgpu::Buffer>,
    delta_pos_buf: Option<wgpu::Buffer>,
    grid_heads_buf: Option<wgpu::Buffer>,
    grid_next_buf: Option<wgpu::Buffer>,
    readback_buf: Option<wgpu::Buffer>,

    pub particle_count: usize,
    particle_capacity: usize,
    pub total_cells: u32,
    pub grid_dim: [u32; 3],
    pub cell_size: f32,
}

impl GpuSphSolver {
    pub fn new(bounds: Aabb, config: &SolverConfig, initial_particles: &[Particle]) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "Failed to find Vulkan / GPU adapter".to_string())?;

        let adapter_info = adapter.get_info();
        let device_name = format!("{} ({:?})", adapter_info.name, adapter_info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SPH Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .map_err(|e| format!("Failed to create GPU device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Load WGSL shader
        let shader_code = include_str!("shaders/pbf.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBF Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        // Create BindGroupLayout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PBF Bind Group Layout"),
            entries: &[
                // 0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: particles
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: predicted_positions
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: old_positions
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: lambdas
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: delta_positions
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: grid_heads
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: grid_next
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBF Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_predict = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Predict Positions"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("predict_positions"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_clear_grid = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Clear Grid"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clear_grid"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_insert_particles = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Insert Particles"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("insert_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_density_lambda = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Density Lambda"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_density_lambda"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_delta_positions = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Delta Positions"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_delta_positions"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_apply_delta = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Apply Delta"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_delta_and_collide"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_update_velocity = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Update Velocity"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_velocity_and_xsph"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Uniform Buffer"),
            size: std::mem::size_of::<GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut solver = Self {
            device,
            queue,
            device_name,
            pipeline_predict,
            pipeline_clear_grid,
            pipeline_insert_particles,
            pipeline_density_lambda,
            pipeline_delta_positions,
            pipeline_apply_delta,
            pipeline_update_velocity,
            bind_group_layout,
            bind_group: None,
            uniform_buf,
            particle_buf: None,
            pred_pos_buf: None,
            old_pos_buf: None,
            lambdas_buf: None,
            delta_pos_buf: None,
            grid_heads_buf: None,
            grid_next_buf: None,
            readback_buf: None,
            particle_count: 0,
            particle_capacity: 0,
            total_cells: 0,
            grid_dim: [0, 0, 0],
            cell_size: config.smoothing_length,
        };

        if !initial_particles.is_empty() {
            solver.init_buffers(bounds, config, initial_particles);
        }

        Ok(solver)
    }

    pub fn init_buffers(&mut self, bounds: Aabb, config: &SolverConfig, particles: &[Particle]) {
        let count = particles.len();
        self.particle_count = count;
        let capacity = (count * 2).max(1024);
        self.particle_capacity = capacity;

        let h = config.smoothing_length;
        self.cell_size = h;
        let size = bounds.size();
        let dim_x = ((size.x / h).ceil() as u32).max(1);
        let dim_y = ((size.y / h).ceil() as u32).max(1);
        let dim_z = ((size.z / h).ceil() as u32).max(1);
        self.grid_dim = [dim_x, dim_y, dim_z];
        self.total_cells = dim_x * dim_y * dim_z;

        // Convert CPU particles to GPU particles
        let mut gpu_particles: Vec<GpuParticle> = Vec::with_capacity(capacity);
        for p in particles {
            let p_type = if p.particle_type == ParticleType::Boundary { 1.0f32 } else { 0.0f32 };
            gpu_particles.push(GpuParticle {
                pos_type: [p.position.x, p.position.y, p.position.z, p_type],
                vel_dens: [p.velocity.x, p.velocity.y, p.velocity.z, p.density],
            });
        }
        // Pad to capacity
        while gpu_particles.len() < capacity {
            gpu_particles.push(GpuParticle {
                pos_type: [0.0; 4],
                vel_dens: [0.0; 4],
            });
        }

        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PBF Particle Buffer"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let pred_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Predicted Positions Buffer"),
            size: (capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let old_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Old Positions Buffer"),
            size: (capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let lambdas_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Lambdas Buffer"),
            size: (capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let delta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Delta Positions Buffer"),
            size: (capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let grid_heads_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Grid Heads Buffer"),
            size: (self.total_cells as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let grid_next_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Grid Next Buffer"),
            size: (capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBF Readback Buffer"),
            size: (capacity * std::mem::size_of::<GpuParticle>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBF Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pred_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: old_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lambdas_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: delta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_heads_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grid_next_buf.as_entire_binding(),
                },
            ],
        });

        self.particle_buf = Some(p_buf);
        self.pred_pos_buf = Some(pred_buf);
        self.old_pos_buf = Some(old_buf);
        self.lambdas_buf = Some(lambdas_buf);
        self.delta_pos_buf = Some(delta_buf);
        self.grid_heads_buf = Some(grid_heads_buf);
        self.grid_next_buf = Some(grid_next_buf);
        self.readback_buf = Some(readback_buf);
        self.bind_group = Some(bind_group);
    }

    pub fn step(
        &mut self,
        bounds: Aabb,
        config: &SolverConfig,
        colliders: &[Collider],
        mouse_ray: Option<Ray>,
        mouse_force_strength: f32,
        mouse_force_mode: i32,
        dt: f32,
    ) {
        if self.particle_count == 0 || self.bind_group.is_none() {
            return;
        }

        let substeps = config.substeps.max(1);
        let sub_dt = dt / substeps as f32;

        let h = config.smoothing_length;
        let h2 = h * h;
        let h3 = h2 * h;
        let h6 = h3 * h3;
        let h9 = h6 * h3;

        let poly6_factor = 315.0 / (64.0 * PI * h9);
        let spiky_factor = 45.0 / (PI * h6);
        let rho0 = config.rest_density;
        let inv_rho0 = 1.0 / rho0;
        let w_q = {
            let r2 = (0.15 * h).powi(2);
            if r2 < h2 {
                let diff = h2 - r2;
                (poly6_factor * diff * diff * diff).max(1e-12)
            } else {
                1e-12
            }
        };

        let mut collider_pos = [0.0; 4];
        let mut collider_enabled = 0.0f32;
        if let Some(col) = colliders.iter().find(|c| c.enabled) {
            match col.shape {
                crate::sph::collider::ColliderShape::Sphere { center, radius } => {
                    collider_enabled = 1.0;
                    collider_pos = [center.x, center.y, center.z, radius];
                }
                _ => {}
            }
        }

        let (m_orig, m_dir) = if let Some(ray) = mouse_ray {
            ([ray.origin.x, ray.origin.y, ray.origin.z, mouse_force_strength],
             [ray.direction.x, ray.direction.y, ray.direction.z, mouse_force_mode as f32])
        } else {
            ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        };

        let dt_factor = (sub_dt / 0.016).clamp(0.05, 1.0);
        let relaxation_epsilon = 1200.0 * (0.075 / h).powi(2);
        let max_displacement = config.particle_radius * 0.35 * dt_factor;

        let uniforms = GpuUniforms {
            bounds_min: [bounds.min.x, bounds.min.y, bounds.min.z, self.cell_size],
            bounds_max: [bounds.max.x, bounds.max.y, bounds.max.z, config.particle_radius],
            gravity: [config.gravity.x, config.gravity.y, config.gravity.z, sub_dt],
            grid_dim: [self.grid_dim[0], self.grid_dim[1], self.grid_dim[2], self.total_cells],
            mouse_origin: m_orig,
            mouse_dir: m_dir,
            collider_pos,
            kernel_params: [h, h2, inv_rho0, rho0],
            kernel_factors: [poly6_factor, spiky_factor, w_q, config.tensile_k],
            solver_params: [
                relaxation_epsilon,
                max_displacement,
                config.restitution,
                config.boundary_friction,
            ],
            fluid_params: [
                config.viscosity,
                config.surface_tension,
                config.max_velocity,
                collider_enabled,
            ],
            counts: [
                self.particle_count as u32,
                config.solver_iterations as u32,
                substeps as u32,
                0,
            ],
        };

        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let workgroups_particles = ((self.particle_count + 255) / 256) as u32;
        let workgroups_cells = ((self.total_cells as usize + 255) / 256) as u32;

        let bind_group = self.bind_group.as_ref().unwrap();

        for _ in 0..substeps {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PBF Substep Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PBF Compute Pass"),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bind_group, &[]);

                // 1. Predict positions
                cpass.set_pipeline(&self.pipeline_predict);
                cpass.dispatch_workgroups(workgroups_particles, 1, 1);

                // 2. Clear uniform grid
                cpass.set_pipeline(&self.pipeline_clear_grid);
                cpass.dispatch_workgroups(workgroups_cells, 1, 1);

                // 3. Insert particles into grid
                cpass.set_pipeline(&self.pipeline_insert_particles);
                cpass.dispatch_workgroups(workgroups_particles, 1, 1);

                // 4. PBF Solver Iterations
                for _ in 0..config.solver_iterations {
                    // 4a. Compute density & lambda
                    cpass.set_pipeline(&self.pipeline_density_lambda);
                    cpass.dispatch_workgroups(workgroups_particles, 1, 1);

                    // 4b. Compute displacement delta
                    cpass.set_pipeline(&self.pipeline_delta_positions);
                    cpass.dispatch_workgroups(workgroups_particles, 1, 1);

                    // 4c. Apply delta & boundary collision
                    cpass.set_pipeline(&self.pipeline_apply_delta);
                    cpass.dispatch_workgroups(workgroups_particles, 1, 1);
                }

                // 5. Update velocity & XSPH viscosity
                cpass.set_pipeline(&self.pipeline_update_velocity);
                cpass.dispatch_workgroups(workgroups_particles, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));
        }
    }

    /// Read particle data back from GPU to CPU
    pub fn read_particles(&self, out: &mut Vec<Particle>) {
        if self.particle_count == 0 || self.particle_buf.is_none() || self.readback_buf.is_none() {
            return;
        }

        let p_buf = self.particle_buf.as_ref().unwrap();
        let r_buf = self.readback_buf.as_ref().unwrap();

        let bytes_to_copy = (self.particle_count * std::mem::size_of::<GpuParticle>()) as u64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PBF Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(p_buf, 0, r_buf, 0, bytes_to_copy);
        self.queue.submit(Some(encoder.finish()));

        let slice = r_buf.slice(0..bytes_to_copy);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let gpu_slice: &[GpuParticle] = bytemuck::cast_slice(&data[..]);

            out.resize(self.particle_count, Particle::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0));
            for (i, p) in out.iter_mut().enumerate() {
                let gp = &gpu_slice[i];
                p.position = Vec3::new(gp.pos_type[0], gp.pos_type[1], gp.pos_type[2]);
                p.velocity = Vec3::new(gp.vel_dens[0], gp.vel_dens[1], gp.vel_dens[2]);
                p.density = gp.vel_dens[3];
                p.id = i as u32;
                p.particle_type = if gp.pos_type[3] > 0.5 {
                    ParticleType::Boundary
                } else {
                    ParticleType::Fluid
                };
            }
            drop(data);
            r_buf.unmap();
        }
    }
}
