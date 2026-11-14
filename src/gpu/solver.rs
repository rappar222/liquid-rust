use crate::math::{Aabb, Ray};
use crate::sph::collider::{Collider, ColliderShape};
use crate::sph::particle::{Particle, ParticleType};
use crate::sph::solver::SolverConfig;
use glam::Vec3;
use std::f32::consts::PI;
use std::sync::Arc;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuColliderData {
    pub info: [u32; 4],    // x: kind (1: sphere, 2: box, 3: cylinder), y: enabled, zw: pad
    pub pos_rad: [f32; 4], // xyz: center, w: radius
    pub extents: [f32; 4], // xyz: half extents, w: height
}

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
    pub colliders: [GpuColliderData; 16],
    pub chamber_info: [f32; 4],
    pub boundary_params: [f32; 4],
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
    pipeline_vorticity: wgpu::ComputePipeline,

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
    pub particle_capacity: usize,
    pub total_cells: u32,
    pub grid_dim: [u32; 3],
    pub cell_size: f32,
    pub sim_time: f32,
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

        let pipeline_vorticity = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline Apply Vorticity"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_vorticity"),
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
            pipeline_vorticity,
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
            sim_time: 0.0,
        };

        if !initial_particles.is_empty() {
            solver.try_init_buffers(bounds, config, initial_particles)?;
        }

        Ok(solver)
    }

    pub fn init_buffers(&mut self, bounds: Aabb, config: &SolverConfig, particles: &[Particle]) {
        if let Err(err) = self.try_init_buffers(bounds, config, particles) {
            eprintln!("GPU buffer allocation warning: {}", err);
        }
    }

    pub fn try_init_buffers(
        &mut self,
        bounds: Aabb,
        config: &SolverConfig,
        particles: &[Particle],
    ) -> Result<(), String> {
        let count = particles.len();
        let is_2d = bounds.size().z < 0.15;
        let capacity = if count > 1_000_000 {
            count + (count / 20).max(10_000)
        } else {
            (count * 2).max(1024)
        };
        let particle_buf_bytes = (capacity * std::mem::size_of::<GpuParticle>()) as u64;
        let limits = self.device.limits();
        if particle_buf_bytes > limits.max_storage_buffer_binding_size as u64
            || particle_buf_bytes > limits.max_buffer_size
        {
            return Err(format!(
                "Particle count {} requires {:.1} MiB storage buffer, exceeding GPU adapter limits (max binding {:.1} MiB, max buffer {:.1} MiB)",
                count,
                particle_buf_bytes as f64 / (1024.0 * 1024.0),
                limits.max_storage_buffer_binding_size as f64 / (1024.0 * 1024.0),
                limits.max_buffer_size as f64 / (1024.0 * 1024.0),
            ));
        }

        self.particle_count = count;
        self.particle_capacity = capacity;
        self.sim_time = 0.0;

        let mut cell_size = config.smoothing_length.max(1e-5);
        let size = bounds.size();
        let max_cells = 8_000_000u64; // Max 32MB grid heads buffer
        let mut dim_x = ((size.x / cell_size).ceil() as u64).max(1);
        let mut dim_y = ((size.y / cell_size).ceil() as u64).max(1);
        let mut dim_z = ((size.z / cell_size).ceil() as u64).max(1);
        if dim_x * dim_y * dim_z > max_cells {
            let exponent = if is_2d { 0.5 } else { 0.3333333 };
            let scale = ((dim_x * dim_y * dim_z) as f64 / max_cells as f64).powf(exponent);
            cell_size *= scale as f32;
            dim_x = ((size.x / cell_size).ceil() as u64).max(1);
            dim_y = ((size.y / cell_size).ceil() as u64).max(1);
            dim_z = ((size.z / cell_size).ceil() as u64).max(1);
        }
        self.cell_size = cell_size;
        self.grid_dim = [dim_x as u32, dim_y as u32, dim_z as u32];
        self.total_cells = (dim_x * dim_y * dim_z) as u32;

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
        Ok(())
    }

    /// Incrementally synchronize CPU particle changes (e.g. faucet emissions or drain removals)
    /// without reallocating GPU buffers whenever `particles.len() <= self.particle_capacity`.
    pub fn sync_particles(
        &mut self,
        bounds: Aabb,
        config: &SolverConfig,
        particles: &[Particle],
        append_only: bool,
    ) {
        let new_count = particles.len();
        if self.particle_buf.is_none() || new_count > self.particle_capacity {
            self.init_buffers(bounds, config, particles);
            return;
        }
        if new_count == 0 {
            self.particle_count = 0;
            return;
        }

        let p_buf = self.particle_buf.as_ref().unwrap();
        if append_only && new_count > self.particle_count {
            let start = self.particle_count;
            let added = &particles[start..new_count];
            let mut gpu_added = Vec::with_capacity(added.len());
            for p in added {
                let p_type = if p.particle_type == ParticleType::Boundary {
                    1.0f32
                } else {
                    0.0f32
                };
                gpu_added.push(GpuParticle {
                    pos_type: [p.position.x, p.position.y, p.position.z, p_type],
                    vel_dens: [p.velocity.x, p.velocity.y, p.velocity.z, p.density],
                });
            }
            let byte_offset = (start * std::mem::size_of::<GpuParticle>()) as u64;
            self.queue
                .write_buffer(p_buf, byte_offset, bytemuck::cast_slice(&gpu_added));
            self.particle_count = new_count;
        } else {
            let mut gpu_all = Vec::with_capacity(new_count);
            for p in particles {
                let p_type = if p.particle_type == ParticleType::Boundary {
                    1.0f32
                } else {
                    0.0f32
                };
                gpu_all.push(GpuParticle {
                    pos_type: [p.position.x, p.position.y, p.position.z, p_type],
                    vel_dens: [p.velocity.x, p.velocity.y, p.velocity.z, p.density],
                });
            }
            self.queue
                .write_buffer(p_buf, 0, bytemuck::cast_slice(&gpu_all));
            self.particle_count = new_count;
        }
    }

    pub fn step(
        &mut self,
        bounds: Aabb,
        chamber: &crate::sph::chamber::Chamber,
        config: &SolverConfig,
        colliders: &[Collider],
        mouse_ray: Option<Ray>,
        mouse_force_strength: f32,
        mouse_force_mode: i32,
        dt: f32,
        is_2d: bool,
    ) {
        if self.particle_count == 0 || self.bind_group.is_none() {
            return;
        }

        // Keep physical substep dt <= (1/60s)/config.substeps even at lower video frame rates
        // (e.g. default --fps 30 or --fps 24) so simulation crispness is frame-rate independent.
        let fps_substep_mult = ((dt * 60.0 - 1e-4).ceil() as usize).clamp(1, 8);
        let substeps = config.substeps.max(1) * fps_substep_mult;
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
        let mut gpu_colliders = [GpuColliderData {
            info: [0; 4],
            pos_rad: [0.0; 4],
            extents: [0.0; 4],
        }; 16];
        let mut num_colliders = 0u32;

        for col in colliders.iter().filter(|c| c.enabled) {
            if num_colliders as usize >= 16 {
                break;
            }
            let rest_bits = col.restitution.to_bits();
            let fric_bits = col.friction.to_bits();
            match &col.shape {
                ColliderShape::Sphere { center, radius } => {
                    if collider_enabled == 0.0 {
                        collider_enabled = 1.0;
                        collider_pos = [center.x, center.y, center.z, *radius];
                    }
                    gpu_colliders[num_colliders as usize] = GpuColliderData {
                        info: [1, 1, rest_bits, fric_bits],
                        pos_rad: [center.x, center.y, center.z, *radius],
                        extents: [0.0; 4],
                    };
                    num_colliders += 1;
                }
                ColliderShape::Box { aabb } => {
                    let center = aabb.center();
                    let half_size = aabb.size() * 0.5;
                    gpu_colliders[num_colliders as usize] = GpuColliderData {
                        info: [2, 1, rest_bits, fric_bits],
                        pos_rad: [center.x, center.y, center.z, 0.0],
                        extents: [half_size.x, half_size.y, half_size.z, 0.0],
                    };
                    num_colliders += 1;
                }
                ColliderShape::Cylinder { center, radius, height } => {
                    gpu_colliders[num_colliders as usize] = GpuColliderData {
                        info: [3, 1, rest_bits, fric_bits],
                        pos_rad: [center.x, center.y, center.z, *radius],
                        extents: [0.0, 0.0, 0.0, *height],
                    };
                    num_colliders += 1;
                }
                ColliderShape::Mesh(_) => {
                    // Triangle meshes are resolved on CPU
                }
            }
        }

        let (m_orig, m_dir) = if let Some(ray) = mouse_ray {
            (
                [ray.origin.x, ray.origin.y, ray.origin.z, mouse_force_strength],
                [ray.direction.x, ray.direction.y, ray.direction.z, mouse_force_mode as f32],
            )
        } else {
            ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        };

        // Keep full PBF stiffness across normal real-time multi-substep tiers (2..8 substeps at
        // 60..120 fps), while scaling compliance smoothly in true slow-motion (frame_dt << 1/90s).
        let frame_dt_ref = 1.0 / 90.0;
        let slow_ratio = (frame_dt_ref / dt.max(1e-5)).max(1.0);
        let cfm_scale = slow_ratio * slow_ratio;
        let slow_factor = (dt / frame_dt_ref).clamp(0.0001, 1.0);
        let relaxation_epsilon = 1000.0 * (0.075 / h).powi(2) * cfm_scale;
        let max_displacement = (config.particle_radius * 0.40)
            .max(h * 0.18)
            .min(2.8 * sub_dt.max(1e-5))
            * slow_factor;

        // Force-like terms applied inside each PBF solver iteration scale with sub_dt^2 / iters
        // so surface tension and tensile anti-clumping produce identical physical acceleration
        // across 2..8 substeps, 3..6 solver iterations, and slow-motion playback.
        let iter_scale = 3.0 / (config.solver_iterations.max(1) as f32);
        let force_dt_scale = ((sub_dt / (1.0 / 60.0)).powi(2)).clamp(1e-6, 1.0) * iter_scale;

        // Substep-independent XSPH viscosity factor (calibrated to 120 Hz reference substep)
        let visc_base = config.viscosity.clamp(0.0, 0.95);
        let visc_substep = if visc_base > 0.0 {
            1.0 - (1.0 - visc_base).powf((sub_dt * 120.0).clamp(0.001, 2.0))
        } else {
            0.0
        };

        self.sim_time += dt;
        let wave_accel_x = if config.wave_amplitude > 1e-4 {
            let omega = std::f32::consts::TAU * config.wave_frequency.clamp(0.05, 8.0);
            config.wave_amplitude * (omega * omega).clamp(1.0, 42.0) * (omega * self.sim_time).sin()
        } else {
            0.0
        };
        let wave_zone_w = ((bounds.max.x - bounds.min.x) * 0.45).max(0.1);

        let mut ch_info = chamber.gpu_info(&bounds);
        ch_info[1] = config.vorticity.max(0.0);

        let uniforms = GpuUniforms {
            bounds_min: [bounds.min.x, bounds.min.y, bounds.min.z, self.cell_size],
            bounds_max: [bounds.max.x, bounds.max.y, bounds.max.z, config.particle_radius],
            gravity: [config.gravity.x, config.gravity.y, config.gravity.z, sub_dt],
            grid_dim: [self.grid_dim[0], self.grid_dim[1], self.grid_dim[2], self.total_cells],
            mouse_origin: m_orig,
            mouse_dir: m_dir,
            collider_pos,
            kernel_params: [h, h2, inv_rho0, rho0],
            kernel_factors: [poly6_factor, spiky_factor, w_q, config.tensile_k * force_dt_scale],
            solver_params: [
                relaxation_epsilon,
                max_displacement,
                config.restitution,
                config.boundary_friction,
            ],
            fluid_params: [
                visc_substep,
                config.surface_tension * force_dt_scale,
                config.max_velocity,
                collider_enabled,
            ],
            counts: [
                self.particle_count as u32,
                config.solver_iterations as u32,
                substeps as u32,
                num_colliders | (if is_2d { 0x80000000 } else { 0 }),
            ],
            colliders: gpu_colliders,
            chamber_info: ch_info,
            boundary_params: [config.spacing, config.air_drag.max(0.0), wave_accel_x, wave_zone_w],
        };

        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let workgroups_particles = ((self.particle_count + 255) / 256) as u32;
        let workgroups_cells = ((self.total_cells as usize + 255) / 256) as u32;
        let (disp_p_x, disp_p_y) = if workgroups_particles <= 65535 {
            (workgroups_particles, 1)
        } else {
            let wg_x = 65535;
            let wg_y = (workgroups_particles + 65534) / 65535;
            (wg_x, wg_y)
        };

        let bind_group = self.bind_group.as_ref().unwrap();
        let has_vorticity = config.vorticity > 1e-5;

        let submit_dispatches = |label: &str, jobs: &[(&wgpu::ComputePipeline, u32, u32)], sync_wait: bool| {
            let mut encoder =
                self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(label),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, bind_group, &[]);
                for &(pipeline, wx, wy) in jobs {
                    cpass.set_pipeline(pipeline);
                    cpass.dispatch_workgroups(wx, wy, 1);
                }
            }
            self.queue.submit(Some(encoder.finish()));
            if sync_wait {
                self.device.poll(wgpu::Maintain::Wait);
            }
        };

        // Multi-million-particle CLI runs can exceed the ~2s OS/driver GPU TDR watchdog if all
        // substeps and solver iterations are queued into one giant command buffer.
        if self.particle_count > 1_200_000 {
            for _ in 0..substeps {
                submit_dispatches(
                    "PBF Substep Setup",
                    &[
                        (&self.pipeline_predict, disp_p_x, disp_p_y),
                        (&self.pipeline_clear_grid, workgroups_cells, 1),
                        (&self.pipeline_insert_particles, disp_p_x, disp_p_y),
                    ],
                    true,
                );
                for _ in 0..config.solver_iterations {
                    submit_dispatches(
                        "PBF Solver Iteration",
                        &[
                            (&self.pipeline_density_lambda, disp_p_x, disp_p_y),
                            (&self.pipeline_delta_positions, disp_p_x, disp_p_y),
                            (&self.pipeline_apply_delta, disp_p_x, disp_p_y),
                        ],
                        true,
                    );
                }
                if has_vorticity {
                    submit_dispatches(
                        "PBF Substep Finalize + Vorticity",
                        &[
                            (&self.pipeline_update_velocity, disp_p_x, disp_p_y),
                            (&self.pipeline_vorticity, disp_p_x, disp_p_y),
                        ],
                        true,
                    );
                } else {
                    submit_dispatches(
                        "PBF Substep Finalize",
                        &[(&self.pipeline_update_velocity, disp_p_x, disp_p_y)],
                        true,
                    );
                }
            }
        } else {
            let mut jobs: Vec<(&wgpu::ComputePipeline, u32, u32)> =
                Vec::with_capacity(5 + config.solver_iterations * 3);
            let per_substep = self.particle_count > 400_000;
            for _ in 0..substeps {
                jobs.push((&self.pipeline_predict, disp_p_x, disp_p_y));
                jobs.push((&self.pipeline_clear_grid, workgroups_cells, 1));
                jobs.push((&self.pipeline_insert_particles, disp_p_x, disp_p_y));
                for _ in 0..config.solver_iterations {
                    jobs.push((&self.pipeline_density_lambda, disp_p_x, disp_p_y));
                    jobs.push((&self.pipeline_delta_positions, disp_p_x, disp_p_y));
                    jobs.push((&self.pipeline_apply_delta, disp_p_x, disp_p_y));
                }
                jobs.push((&self.pipeline_update_velocity, disp_p_x, disp_p_y));
                if has_vorticity {
                    jobs.push((&self.pipeline_vorticity, disp_p_x, disp_p_y));
                }
                if per_substep {
                    submit_dispatches("PBF Substep Pass", &jobs, true);
                    jobs.clear();
                }
            }
            if !per_substep && !jobs.is_empty() {
                submit_dispatches("PBF Batched Frame Pass", &jobs, false);
            }
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
            out.par_iter_mut().enumerate().for_each(|(i, p)| {
                let gp = &gpu_slice[i];
                let pos = Vec3::new(gp.pos_type[0], gp.pos_type[1], gp.pos_type[2]);
                let vel = Vec3::new(gp.vel_dens[0], gp.vel_dens[1], gp.vel_dens[2]);
                let dens = gp.vel_dens[3];
                p.position = if pos.is_finite() { pos } else { Vec3::ZERO };
                p.velocity = if vel.is_finite() { vel } else { Vec3::ZERO };
                p.density = if dens.is_finite() { dens } else { 1000.0 };
                p.id = i as u32;
                p.particle_type = if gp.pos_type[3] > 0.5 {
                    ParticleType::Boundary
                } else {
                    ParticleType::Fluid
                };
            });
            drop(data);
            r_buf.unmap();
        }
    }

    /// Direct handle to the live GPU particle storage buffer (`array<GpuParticle>`).
    pub fn particle_buffer(&self) -> Option<&wgpu::Buffer> {
        self.particle_buf.as_ref()
    }

    /// Whether the active GPU adapter permits storage buffer bindings in vertex shaders.
    pub fn supports_vertex_storage(&self) -> bool {
        self.device.limits().max_storage_buffers_per_shader_stage >= 1
    }
}
