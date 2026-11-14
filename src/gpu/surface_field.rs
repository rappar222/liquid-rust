use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSurfaceParams {
    pub grid_min_voxel: [f32; 4],
    pub coarse_min_inv_r2: [f32; 4],
    pub splat_params: [f32; 4],
    pub c_inv_size: [f32; 4],
    pub u_dims: [u32; 4],
    pub u_coarse: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSurfaceParticle {
    pub pos_type: [f32; 4],
    pub vel_dens: [f32; 4],
}

const CORNERS_PER_BRICK: usize = 729; // (8 + 1)^3
const MAX_COARSE_AXIS: usize = 48;

/// GPU-accelerated narrow-band SPH surface scalar & velocity field evaluator.
///
/// Uses a workgroup-cooperative shared-memory Vulkan compute shader to evaluate
/// 9x9x9 corner densities for all active bricks simultaneously, compacting and
/// transferring back only surface-crossing bricks for watertight triangle welding.
pub struct GpuSurfaceField {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    particles_buf: Option<wgpu::Buffer>,
    cell_offsets_buf: Option<wgpu::Buffer>,
    active_bricks_buf: Option<wgpu::Buffer>,
    counters_buf: wgpu::Buffer,
    counters_read_buf: wgpu::Buffer,
    crossing_ids_buf: Option<wgpu::Buffer>,
    crossing_ids_read_buf: Option<wgpu::Buffer>,
    out_corners_buf: Option<wgpu::Buffer>,
    out_corners_read_buf: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,

    particle_capacity: usize,
    cell_offset_capacity: usize,
    active_brick_capacity: usize,
    crossing_capacity: usize,

    // Reusable CPU scratch buffers (zero per-frame heap churn after warm-up)
    coarse_counts: Vec<u32>,
    coarse_offsets: Vec<u32>,
    coarse_cursor: Vec<u32>,
    particle_cells: Vec<u32>,
    sorted_particles: Vec<GpuSurfaceParticle>,
    brick_packed: Vec<[u32; 4]>,

    pub crossing_brick_ids: Vec<u32>,
    pub crossing_corners: Vec<[f32; 4]>,
}

impl GpuSurfaceField {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<Self, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Surface Field Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/surface_field.wgsl").into()),
        });

        let ro_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Surface Field Bind Group Layout"),
                entries: &[
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
                    ro_storage(1),
                    ro_storage(2),
                    ro_storage(3),
                    rw_storage(4),
                    rw_storage(5),
                    rw_storage(6),
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Surface Field Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Surface Field Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evaluate_bricks"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Field Params"),
            size: std::mem::size_of::<GpuSurfaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counters_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Field Counters"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counters_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Field Counters Readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            params_buf,
            particles_buf: None,
            cell_offsets_buf: None,
            active_bricks_buf: None,
            counters_buf,
            counters_read_buf,
            crossing_ids_buf: None,
            crossing_ids_read_buf: None,
            out_corners_buf: None,
            out_corners_read_buf: None,
            bind_group: None,
            particle_capacity: 0,
            cell_offset_capacity: 0,
            active_brick_capacity: 0,
            crossing_capacity: 0,
            coarse_counts: Vec::new(),
            coarse_offsets: Vec::new(),
            coarse_cursor: Vec::new(),
            particle_cells: Vec::new(),
            sorted_particles: Vec::new(),
            brick_packed: Vec::new(),
            crossing_brick_ids: Vec::new(),
            crossing_corners: Vec::new(),
        })
    }

    fn ensure_buffers(
        &mut self,
        num_particles: usize,
        num_cell_offsets: usize,
        num_active_bricks: usize,
    ) -> bool {
        let max_bind = self.device.limits().max_storage_buffer_binding_size as u64;
        let mut rebind = self.bind_group.is_none();

        if num_particles > self.particle_capacity || self.particles_buf.is_none() {
            let cap = num_particles.max(1024).next_power_of_two();
            let size = (cap * std::mem::size_of::<GpuSurfaceParticle>()) as u64;
            if size > max_bind {
                return false;
            }
            self.particles_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Surface Sorted Particles"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.particle_capacity = cap;
            rebind = true;
        }

        if num_cell_offsets > self.cell_offset_capacity || self.cell_offsets_buf.is_none() {
            let cap = num_cell_offsets.max(4096).next_power_of_two();
            let size = (cap * std::mem::size_of::<u32>()) as u64;
            if size > max_bind {
                return false;
            }
            self.cell_offsets_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Surface Coarse Cell Offsets"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cell_offset_capacity = cap;
            rebind = true;
        }

        if num_active_bricks > self.active_brick_capacity || self.active_bricks_buf.is_none() {
            let cap = num_active_bricks.max(256).next_power_of_two();
            let size = (cap * 16) as u64;
            if size > max_bind {
                return false;
            }
            self.active_bricks_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Surface Active Bricks"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.active_brick_capacity = cap;
            rebind = true;
        }

        if num_active_bricks > self.crossing_capacity
            || self.crossing_ids_buf.is_none()
            || self.out_corners_buf.is_none()
        {
            let cap = num_active_bricks.max(256).next_power_of_two();
            let corners_size = (cap * CORNERS_PER_BRICK * 16) as u64;
            if corners_size > max_bind {
                return false;
            }
            let ids_size = (cap * 4) as u64;
            self.crossing_ids_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Surface Crossing Brick IDs"),
                size: ids_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.crossing_ids_read_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Surface Crossing Brick IDs Readback"),
                size: ids_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.out_corners_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Surface Crossing Corners"),
                size: corners_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.out_corners_read_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Surface Crossing Corners Readback"),
                size: corners_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.crossing_capacity = cap;
            rebind = true;
        }

        if rebind {
            self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Surface Field Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.particles_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cell_offsets_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.active_bricks_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.counters_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.crossing_ids_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.out_corners_buf.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            }));
        }

        true
    }

    /// Evaluate corner densities & velocities on GPU for all `active_brick_coords` and
    /// populate `self.crossing_brick_ids` + `self.crossing_corners` exclusively with
    /// bricks whose field crosses `iso`. Returns `false` if GPU limits require CPU fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_crossing_bricks(
        &mut self,
        fluid_positions: &[Vec3],
        fluid_velocities: &[Vec3],
        grid_min: Vec3,
        grid_max: Vec3,
        voxel_size: f32,
        splat_radius: f32,
        iso: f32,
        nx: usize,
        ny: usize,
        nz: usize,
        active_brick_coords: &[[usize; 3]],
    ) -> bool {
        self.crossing_brick_ids.clear();
        self.crossing_corners.clear();
        if fluid_positions.is_empty() || active_brick_coords.is_empty() {
            return true;
        }

        let extent = (grid_max - grid_min).max(Vec3::splat(1e-4));
        let sr = splat_radius.max(1e-5);
        let cgx = ((extent.x / sr).ceil() as usize).clamp(1, MAX_COARSE_AXIS);
        let cgy = ((extent.y / sr).ceil() as usize).clamp(1, MAX_COARSE_AXIS);
        let cgz = ((extent.z / sr).ceil() as usize).clamp(1, MAX_COARSE_AXIS);
        let total_cells = cgx * cgy * cgz;

        let c_inv_size = Vec3::new(
            cgx as f32 / extent.x,
            cgy as f32 / extent.y,
            cgz as f32 / extent.z,
        );

        let n = fluid_positions.len();
        self.coarse_counts.clear();
        self.coarse_counts.resize(total_cells, 0);
        self.particle_cells.resize(n, 0);

        for i in 0..n {
            let rel = (fluid_positions[i] - grid_min) * c_inv_size;
            let cx = (rel.x.floor() as isize).clamp(0, cgx as isize - 1) as usize;
            let cy = (rel.y.floor() as isize).clamp(0, cgy as isize - 1) as usize;
            let cz = (rel.z.floor() as isize).clamp(0, cgz as isize - 1) as usize;
            let cid = (cz * cgy + cy) * cgx + cx;
            self.particle_cells[i] = cid as u32;
            self.coarse_counts[cid] += 1;
        }

        self.coarse_offsets.resize(total_cells + 1, 0);
        let mut running = 0u32;
        for c in 0..total_cells {
            self.coarse_offsets[c] = running;
            running += self.coarse_counts[c];
        }
        self.coarse_offsets[total_cells] = running;

        self.coarse_cursor.clear();
        self.coarse_cursor
            .extend_from_slice(&self.coarse_offsets[..total_cells]);

        self.sorted_particles.resize(
            n,
            GpuSurfaceParticle {
                pos_type: [0.0; 4],
                vel_dens: [0.0; 4],
            },
        );
        for i in 0..n {
            let cid = self.particle_cells[i] as usize;
            let dst = self.coarse_cursor[cid] as usize;
            self.coarse_cursor[cid] += 1;
            let p = fluid_positions[i];
            let v = fluid_velocities[i];
            self.sorted_particles[dst] = GpuSurfaceParticle {
                pos_type: [p.x, p.y, p.z, 0.0],
                vel_dens: [v.x, v.y, v.z, 0.0],
            };
        }

        let num_bricks = active_brick_coords.len();
        self.brick_packed.clear();
        self.brick_packed.extend(
            active_brick_coords
                .iter()
                .map(|&[bx, by, bz]| [bx as u32, by as u32, bz as u32, 0]),
        );

        if !self.ensure_buffers(n, self.coarse_offsets.len(), num_bricks) {
            return false;
        }

        let splat_r2 = splat_radius * splat_radius;
        let inv_splat_r2 = 1.0 / splat_r2.max(1e-10);
        let params = GpuSurfaceParams {
            grid_min_voxel: [grid_min.x, grid_min.y, grid_min.z, voxel_size],
            coarse_min_inv_r2: [grid_min.x, grid_min.y, grid_min.z, inv_splat_r2],
            splat_params: [splat_radius, splat_r2, iso, voxel_size * 8.0],
            c_inv_size: [c_inv_size.x, c_inv_size.y, c_inv_size.z, 0.0],
            u_dims: [nx as u32, ny as u32, nz as u32, num_bricks as u32],
            u_coarse: [
                cgx as u32,
                cgy as u32,
                cgz as u32,
                self.crossing_capacity as u32,
            ],
        };

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.counters_buf,
            0,
            bytemuck::cast_slice(&[0u32; 4]),
        );
        self.queue.write_buffer(
            self.particles_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.sorted_particles),
        );
        self.queue.write_buffer(
            self.cell_offsets_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.coarse_offsets),
        );
        self.queue.write_buffer(
            self.active_bricks_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.brick_packed),
        );

        let (dispatch_x, dispatch_y) = if num_bricks <= 256 {
            (num_bricks as u32, 1u32)
        } else {
            (256u32, ((num_bricks + 255) / 256) as u32)
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surface Field Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Surface Field Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        encoder.copy_buffer_to_buffer(&self.counters_buf, 0, &self.counters_read_buf, 0, 16);
        self.queue.submit(Some(encoder.finish()));

        // Read back the compacted count of surface-crossing bricks (typically ~15-25% of active bricks)
        let cnt_slice = self.counters_read_buf.slice(0..16);
        let (tx, rx) = std::sync::mpsc::channel();
        cnt_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        let crossing_count = if let Ok(Ok(())) = rx.recv() {
            let mapped = cnt_slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&mapped);
            let cnt = words[0] as usize;
            drop(mapped);
            self.counters_read_buf.unmap();
            if cnt > self.crossing_capacity {
                return false;
            }
            cnt
        } else {
            return false;
        };

        if crossing_count == 0 {
            return true;
        }

        let ids_bytes = (crossing_count * 4) as u64;
        let corners_bytes = (crossing_count * CORNERS_PER_BRICK * 16) as u64;

        let mut copy_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surface Field Copy Encoder"),
            });
        copy_enc.copy_buffer_to_buffer(
            self.crossing_ids_buf.as_ref().unwrap(),
            0,
            self.crossing_ids_read_buf.as_ref().unwrap(),
            0,
            ids_bytes,
        );
        copy_enc.copy_buffer_to_buffer(
            self.out_corners_buf.as_ref().unwrap(),
            0,
            self.out_corners_read_buf.as_ref().unwrap(),
            0,
            corners_bytes,
        );
        self.queue.submit(Some(copy_enc.finish()));

        let ids_buf = self.crossing_ids_read_buf.as_ref().unwrap();
        let corners_buf = self.out_corners_read_buf.as_ref().unwrap();
        let ids_slice = ids_buf.slice(0..ids_bytes);
        let corners_slice = corners_buf.slice(0..corners_bytes);

        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        ids_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx1.send(r);
        });
        corners_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx2.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);

        match (rx1.recv(), rx2.recv()) {
            (Ok(Ok(())), Ok(Ok(()))) => {
                let id_view = ids_slice.get_mapped_range();
                let id_words: &[u32] = bytemuck::cast_slice(&id_view);
                self.crossing_brick_ids
                    .extend_from_slice(&id_words[..crossing_count]);
                drop(id_view);
                ids_buf.unmap();

                let c_view = corners_slice.get_mapped_range();
                let c_vecs: &[[f32; 4]] = bytemuck::cast_slice(&c_view);
                self.crossing_corners
                    .extend_from_slice(&c_vecs[..crossing_count * CORNERS_PER_BRICK]);
                drop(c_view);
                corners_buf.unmap();
                true
            }
            _ => {
                ids_buf.unmap();
                corners_buf.unmap();
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{SurfaceMesh, SurfaceMeshConfig};
    use crate::sph::particle::{Particle, ParticleType};

    #[test]
    fn gpu_surface_field_matches_cpu_mesh() {
        let Ok((device, queue, _)) = crate::gpu::create_headless_wgpu_device() else {
            return;
        };
        let mut gpu_field = GpuSurfaceField::new(device, queue).expect("create GpuSurfaceField");

        let mut particles = Vec::new();
        let r = 0.025f32;
        let spacing = r * 1.9;
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let mut p = Particle::new(
                        Vec3::new(x as f32 * spacing, y as f32 * spacing, z as f32 * spacing),
                        Vec3::new(0.4, 0.1, -0.2),
                        1.0,
                        particles.len() as u32,
                    );
                    p.particle_type = ParticleType::Fluid;
                    particles.push(p);
                }
            }
        }

        let cfg = SurfaceMeshConfig::default();
        let cpu_mesh = SurfaceMesh::from_particles(&particles, r, r * 2.5, &cfg);
        let gpu_mesh = SurfaceMesh::from_particles_with_gpu(
            &particles,
            r,
            r * 2.5,
            &cfg,
            &mut gpu_field,
        );

        assert!(!cpu_mesh.is_empty());
        assert_eq!(cpu_mesh.vertices.len(), gpu_mesh.vertices.len());
        assert_eq!(cpu_mesh.indices.len(), gpu_mesh.indices.len());
        assert!(!gpu_field.crossing_brick_ids.is_empty());
    }
}

