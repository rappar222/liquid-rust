use crate::math::Aabb;
use crate::render::{
    Camera, ColorMap, ColorMapType, ColorMode, SurfaceMesh, SurfaceMeshConfig,
};
use crate::sph::collider::{Collider, ColliderShape};
use crate::sph::particle::Particle;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};

struct ProjectedParticle {
    px: f32,
    py: f32,
    r_px: f32,
    z_dist: f32,
    particle_radius: f32,
    x_min: i32,
    x_max: i32,
    y_min: i32,
    y_max: i32,
    base_rgb: [u8; 3],
}

#[derive(Clone, Copy)]
struct ProjectedTri {
    sx0: f32,
    sy0: f32,
    sx1: f32,
    sy1: f32,
    sx2: f32,
    sy2: f32,
    inv_w0: f32,
    inv_w1: f32,
    inv_w2: f32,
    z0: f32,
    z1: f32,
    z2: f32,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    spd0: f32,
    spd1: f32,
    spd2: f32,
    fm0: f32,
    fm1: f32,
    fm2: f32,
    inv_area: f32,
    is_front: bool,
    x_min: i32,
    x_max: i32,
    y_min: i32,
    y_max: i32,
}

#[derive(Clone, Copy, Default)]
struct SurfaceHit {
    pos: [f32; 3],
    normal: [f32; 3],
    speed: f32,
    foam: f32,
}

/// Compact 2D projected height/density map along key light for soft liquid floor shadows & caustics.
struct FluidShadowGrid {
    origin_xz: Vec2,
    inv_size_xz: Vec2,
    sun_tan_xz: Vec2,
    cells: Vec<f32>,
    res: usize,
}

impl FluidShadowGrid {
    fn build(mesh: &SurfaceMesh, bounds: Aabb, light_dir: Vec3, floor_y: f32) -> Option<Self> {
        if mesh.vertices.is_empty() || light_dir.y <= 0.08 {
            return None;
        }
        let res = 128usize;
        let pad = 0.6;
        let min_x = bounds.min.x - pad;
        let max_x = bounds.max.x + pad;
        let min_z = bounds.min.z - pad;
        let max_z = bounds.max.z + pad;
        let span_x = (max_x - min_x).max(0.1);
        let span_z = (max_z - min_z).max(0.1);
        let sun_tan_xz = Vec2::new(light_dir.x / light_dir.y, light_dir.z / light_dir.y);

        let mut cells = vec![0.0f32; res * res];
        let inv_sx = res as f32 / span_x;
        let inv_sz = res as f32 / span_z;

        // Splat vertex density onto the floor along -light_dir
        let weight_per_vert = (64.0 / (mesh.vertices.len() as f32).sqrt().max(8.0)).clamp(0.08, 0.55);
        for v in &mesh.vertices {
            let h = (v.position[1] - floor_y).max(0.0);
            let fx = v.position[0] - sun_tan_xz.x * h;
            let fz = v.position[2] - sun_tan_xz.y * h;
            let gx = ((fx - min_x) * inv_sx) as i32;
            let gz = ((fz - min_z) * inv_sz) as i32;
            if gx >= 1 && gx < (res as i32 - 1) && gz >= 1 && gz < (res as i32 - 1) {
                let idx = gz as usize * res + gx as usize;
                cells[idx] = (cells[idx] + weight_per_vert).min(2.5);
            }
        }

        // Fast separable 3x3 box blur for soft penumbra
        let mut blurred = vec![0.0f32; res * res];
        for z in 1..(res - 1) {
            for x in 1..(res - 1) {
                let mut sum = 0.0;
                for dz in -1..=1 {
                    let row = (z as isize + dz) as usize * res;
                    for dx in -1..=1 {
                        sum += cells[row + (x as isize + dx) as usize];
                    }
                }
                blurred[z * res + x] = sum * (1.0 / 9.0);
            }
        }

        Some(Self {
            origin_xz: Vec2::new(min_x, min_z),
            inv_size_xz: Vec2::new(1.0 / span_x, 1.0 / span_z),
            sun_tan_xz,
            cells: blurred,
            res,
        })
    }

    #[inline]
    fn sample(&self, x: f32, z: f32) -> f32 {
        let u = (x - self.origin_xz.x) * self.inv_size_xz.x * (self.res as f32 - 1.0);
        let v = (z - self.origin_xz.y) * self.inv_size_xz.y * (self.res as f32 - 1.0);
        if u < 0.0 || v < 0.0 || u >= (self.res - 1) as f32 || v >= (self.res - 1) as f32 {
            return 0.0;
        }
        let x0 = u as usize;
        let z0 = v as usize;
        let fx = u - x0 as f32;
        let fz = v - z0 as f32;
        let r0 = z0 * self.res;
        let r1 = (z0 + 1) * self.res;
        let c00 = self.cells[r0 + x0];
        let c10 = self.cells[r0 + x0 + 1];
        let c01 = self.cells[r1 + x0];
        let c11 = self.cells[r1 + x0 + 1];
        let a = c00 + (c10 - c00) * fx;
        let b = c01 + (c11 - c01) * fx;
        let _ = self.sun_tan_xz;
        a + (b - a) * fz
    }
}

/// High-performance multi-threaded software 3D rasterizer for offline/headless MP4 video rendering.
/// Supports both glossy SPH Sphere rendering and physically-based Surface Mesh rendering.
pub struct SoftwareRenderer {
    pub width: usize,
    pub height: usize,
    pub rgb_buffer: Vec<u8>,
    pub depth_buffer: Vec<f32>,
    pub enable_culling: bool,
    // Persistent scratch buffers for zero-allocation surface mesh rendering
    bg_rgb: Vec<u8>,
    front_depth: Vec<f32>,
    back_depth: Vec<f32>,
    surf_hits: Vec<SurfaceHit>,
}

impl SoftwareRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgb_buffer: vec![0; width * height * 3],
            depth_buffer: vec![f32::INFINITY; width * height],
            enable_culling: true,
            bg_rgb: Vec::new(),
            front_depth: Vec::new(),
            back_depth: Vec::new(),
            surf_hits: Vec::new(),
        }
    }

    fn ensure_mesh_buffers(&mut self) {
        let px_count = self.width * self.height;
        if self.bg_rgb.len() != px_count * 3 {
            self.bg_rgb.resize(px_count * 3, 0);
        }
        if self.front_depth.len() != px_count {
            self.front_depth.resize(px_count, f32::INFINITY);
        }
        if self.back_depth.len() != px_count {
            self.back_depth.resize(px_count, 0.0);
        }
        if self.surf_hits.len() != px_count {
            self.surf_hits.resize(px_count, SurfaceHit::default());
        }
    }

    /// Clear buffer with studio dark gradient background (parallel across rows)
    pub fn clear(&mut self) {
        let w = self.width;
        let h = self.height.max(1);

        self.rgb_buffer
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let t = y as f32 / h as f32;
                let r = (12.0 + t * 14.0) as u8;
                let g = (15.0 + t * 18.0) as u8;
                let b = (22.0 + t * 26.0) as u8;

                for px in row.chunks_exact_mut(3) {
                    px[0] = r;
                    px[1] = g;
                    px[2] = b;
                }
            });

        self.depth_buffer.par_iter_mut().for_each(|d| *d = f32::INFINITY);
    }

    /// Render SPH particle simulation frame with multi-threaded band-bucketed rasterization
    pub fn render_frame(
        &mut self,
        particles: &[Particle],
        particle_radius: f32,
        chamber: &crate::sph::chamber::Chamber,
        bounds: Aabb,
        camera: &Camera,
        color_mode: ColorMode,
        color_map: ColorMapType,
        rest_density: f32,
    ) {
        self.clear();

        let width = self.width;
        let height = self.height;
        if width == 0 || height == 0 {
            return;
        }
        let aspect = width as f32 / height as f32;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let view_proj = proj * view;

        // 1. Draw Chamber Wireframe
        self.draw_chamber_wireframe(chamber, &bounds, view_proj);

        // 2. Parallel projection of visible particles
        let fov_y = camera.fov_y;
        let tan_half_fov = (fov_y * 0.5).tan();
        let w_f = width as f32;
        let h_f = height as f32;
        let w_i = width as i32;
        let h_i = height as i32;
        let enable_culling = self.enable_culling;

        let projected_particles: Vec<ProjectedParticle> = particles
            .par_iter()
            .filter_map(|p| {
                let p_world = Vec4::new(p.position.x, p.position.y, p.position.z, 1.0);
                let p_cam = view * p_world;

                if p_cam.z >= -0.05 {
                    return None;
                }

                let z_dist = -p_cam.z;
                let p_clip = proj * p_cam;

                if p_clip.w <= 0.0 {
                    return None;
                }

                let ndc_x = p_clip.x / p_clip.w;
                let ndc_y = p_clip.y / p_clip.w;

                let px = (ndc_x * 0.5 + 0.5) * w_f;
                let py = (1.0 - (ndc_y * 0.5 + 0.5)) * h_f;

                let r_px = (particle_radius / (z_dist * tan_half_fov)) * (h_f * 0.5);
                if enable_culling && r_px < 0.5 {
                    return None;
                }

                let r_px_ceil = r_px.ceil().max(1.0) as i32;
                let x_min = (px.floor() as i32 - r_px_ceil).max(0);
                let x_max = (px.ceil() as i32 + r_px_ceil).min(w_i - 1);
                let y_min = (py.floor() as i32 - r_px_ceil).max(0);
                let y_max = (py.ceil() as i32 + r_px_ceil).min(h_i - 1);

                if enable_culling && (x_min > x_max || y_min > y_max) {
                    return None;
                }

                let base_rgb =
                    Self::calc_particle_color(p, color_mode, color_map, rest_density, &bounds);

                Some(ProjectedParticle {
                    px,
                    py,
                    r_px,
                    z_dist,
                    particle_radius,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    base_rgb,
                })
            })
            .collect();

        // 3. Bucket particles into horizontal screen bands so each band only visits overlapping particles
        let band_height = 32usize.max((height + 63) / 64);
        let num_bands = (height + band_height - 1) / band_height;
        let mut band_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_bands];
        for (idx, pp) in projected_particles.iter().enumerate() {
            let b0 = (pp.y_min.max(0) as usize) / band_height;
            let b1 = ((pp.y_max.max(0) as usize) / band_height).min(num_bands.saturating_sub(1));
            for b in b0..=b1 {
                band_buckets[b].push(idx as u32);
            }
        }

        let light_dir = Vec3::new(0.42, 0.82, 0.38).normalize();
        let half_dir = (light_dir + Vec3::new(0.0, 0.0, 1.0)).normalize();

        let rgb_chunks = self.rgb_buffer.chunks_mut(band_height * width * 3);
        let depth_chunks = self.depth_buffer.chunks_mut(band_height * width);

        rgb_chunks
            .zip(depth_chunks)
            .zip(band_buckets.iter())
            .enumerate()
            .par_bridge()
            .for_each(|(band_idx, ((rgb_band, depth_band), bucket))| {
                let band_y_start = (band_idx * band_height) as i32;
                let band_rows = rgb_band.len() / (width * 3);
                let band_y_end = band_y_start + band_rows as i32;

                for &p_idx in bucket {
                    let pp = &projected_particles[p_idx as usize];
                    let inv_r_px = 1.0 / pp.r_px;
                    let local_y_min = pp.y_min.max(band_y_start);
                    let local_y_max = pp.y_max.min(band_y_end - 1);

                    let base_r = pp.base_rgb[0] as f32;
                    let base_g = pp.base_rgb[1] as f32;
                    let base_b = pp.base_rgb[2] as f32;

                    for y in local_y_min..=local_y_max {
                        let dy = (y as f32 + 0.5 - pp.py) * inv_r_px;
                        let dy2 = dy * dy;
                        if dy2 > 1.0 {
                            continue;
                        }

                        let band_local_y = (y - band_y_start) as usize;
                        let row_offset = band_local_y * width;

                        for x in pp.x_min..=pp.x_max {
                            let dx = (x as f32 + 0.5 - pp.px) * inv_r_px;
                            let r2 = dx * dx + dy2;

                            if r2 <= 1.0 {
                                let z_norm = (1.0 - r2).sqrt();
                                let depth = pp.z_dist - z_norm * pp.particle_radius;

                                let pixel_idx = row_offset + x as usize;
                                if depth < depth_band[pixel_idx] {
                                    depth_band[pixel_idx] = depth;

                                    let normal = Vec3::new(dx, -dy, z_norm);
                                    let diff = normal.dot(light_dir).max(0.0);
                                    let hemi = 0.5 + 0.5 * normal.y;
                                    let spec = normal.dot(half_dir).max(0.0).powi(36);
                                    let fresnel = 0.025 + 0.975 * (1.0 - z_norm).powi(4);

                                    let shade = 0.28 + 0.16 * hemi + 0.62 * diff;
                                    let lit_r = (base_r * shade + 145.0 * spec + fresnel * 95.0)
                                        .clamp(0.0, 255.0) as u8;
                                    let lit_g = (base_g * shade + 150.0 * spec + fresnel * 135.0)
                                        .clamp(0.0, 255.0) as u8;
                                    let lit_b = (base_b * shade + 160.0 * spec + fresnel * 175.0)
                                        .clamp(0.0, 255.0) as u8;

                                    let rgb_idx = pixel_idx * 3;
                                    rgb_band[rgb_idx] = lit_r;
                                    rgb_band[rgb_idx + 1] = lit_g;
                                    rgb_band[rgb_idx + 2] = lit_b;
                                }
                            }
                        }
                    }
                }
            });
    }

    /// Render reconstructed 3D liquid surface mesh with physically-based optics, Beer-Lambert
    /// volume absorption, procedural triplanar textures, soft floor shadows/caustics, and obstacles.
    pub fn render_surface_mesh_frame(
        &mut self,
        mesh: &SurfaceMesh,
        cfg: &SurfaceMeshConfig,
        camera: &Camera,
        chamber: &crate::sph::chamber::Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        anim_time: f32,
        is_2d: bool,
    ) {
        let width = self.width;
        let height = self.height;
        if width == 0 || height == 0 {
            return;
        }
        self.ensure_mesh_buffers();

        let aspect = width as f32 / height as f32;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let eye_pos = camera.eye_position();

        let cam_fwd = (camera.target - eye_pos).normalize_or_zero();
        let cam_fwd = if cam_fwd.length_squared() > 0.5 {
            cam_fwd
        } else {
            Vec3::NEG_Z
        };
        let cam_right = cam_fwd.cross(Vec3::Y).normalize_or_zero();
        let cam_right = if cam_right.length_squared() > 0.5 {
            cam_right
        } else {
            Vec3::X
        };
        let cam_up = cam_right.cross(cam_fwd).normalize_or_zero();
        let tan_half_fov = (camera.fov_y * 0.5).tan();

        let light_dir = {
            let ld = cfg.light_dir.normalize_or_zero();
            if ld.length_squared() > 0.5 {
                ld
            } else {
                Vec3::new(0.42, 0.82, 0.38).normalize()
            }
        };

        let floor_y = bounds.min.y - 0.002;
        let shadow_grid = if cfg.cast_shadows && !is_2d {
            FluidShadowGrid::build(mesh, bounds, light_dir, floor_y)
        } else {
            None
        };

        let shallow_srgb = Vec3::from_array(cfg.shallow_color).clamp(Vec3::ZERO, Vec3::ONE);
        let deep_srgb = Vec3::from_array(cfg.deep_color).clamp(Vec3::ZERO, Vec3::ONE);
        let shallow_lin = shallow_srgb.powf(2.2);
        let deep_lin = deep_srgb.powf(2.2);

        // 1. Render Studio Sky, 3D Perspective Floor Grid (+ Soft Fluid Shadows/Caustics), and Analytic Obstacles
        let w_f = width as f32;
        let h_f = height as f32;
        self.rgb_buffer
            .par_chunks_mut(width * 3)
            .zip(self.depth_buffer.par_chunks_mut(width))
            .enumerate()
            .for_each(|(y, (rgb_row, depth_row))| {
                let ndc_y = 1.0 - 2.0 * ((y as f32 + 0.5) / h_f);
                let sy = ndc_y * tan_half_fov;

                for x in 0..width {
                    let ndc_x = 2.0 * ((x as f32 + 0.5) / w_f) - 1.0;
                    let sx = ndc_x * aspect * tan_half_fov;
                    let ray_dir = (cam_fwd + cam_right * sx + cam_up * sy).normalize();

                    // Studio gradient background (linear space)
                    let sky_t = (0.5 + 0.5 * ray_dir.y).clamp(0.0, 1.0);
                    let mut col_lin = Vec3::new(0.012, 0.016, 0.026)
                        .lerp(Vec3::new(0.036, 0.052, 0.082), sky_t);
                    let mut best_dist = f32::INFINITY;

                    // Studio floor plane in 3D mode
                    if !is_2d && ray_dir.y < -1e-4 {
                        let t_floor = (floor_y - eye_pos.y) / ray_dir.y;
                        if t_floor > 0.05 {
                            let hit = eye_pos + ray_dir * t_floor;
                            let margin = 1.6;
                            if hit.x >= bounds.min.x - margin
                                && hit.x <= bounds.max.x + margin
                                && hit.z >= bounds.min.z - margin
                                && hit.z <= bounds.max.z + margin
                            {
                                // Distance fade outside chamber footprint
                                let dx = (bounds.min.x - hit.x)
                                    .max(0.0)
                                    .max(hit.x - bounds.max.x);
                                let dz = (bounds.min.z - hit.z)
                                    .max(0.0)
                                    .max(hit.z - bounds.max.z);
                                let edge_fade = (1.0 - (dx * dx + dz * dz).sqrt() / margin)
                                    .clamp(0.0, 1.0);

                                if edge_fade > 0.001 {
                                    let cell_size = 0.25f32;
                                    let gx = (hit.x / cell_size).floor() as i32;
                                    let gz = (hit.z / cell_size).floor() as i32;
                                    let checker = ((gx + gz) & 1) as f32;
                                    let mut floor_albedo = Vec3::new(0.042, 0.050, 0.066)
                                        + Vec3::splat(checker * 0.012);

                                    // Anti-aliased minor/major grid lines
                                    let fx = ((hit.x / cell_size).fract() + 1.0).fract();
                                    let fz = ((hit.z / cell_size).fract() + 1.0).fract();
                                    let dist_minor = fx.min(1.0 - fx).min(fz.min(1.0 - fz));
                                    let line_w = (0.012 + t_floor * 0.0025).min(0.08);
                                    let minor_line = (1.0 - dist_minor / line_w).clamp(0.0, 1.0);
                                    floor_albedo += Vec3::new(0.030, 0.052, 0.085) * minor_line * 0.65;

                                    let mut sun_vis = Vec3::ONE;
                                    let mut caustic_glow = Vec3::ZERO;
                                    if let Some(ref sg) = shadow_grid {
                                        let dens = sg.sample(hit.x, hit.z);
                                        if dens > 0.01 {
                                            // Colored Beer-Lambert shadow tint
                                            let trans = Vec3::new(
                                                (-dens * (1.5 - shallow_lin.x) * 1.35).exp(),
                                                (-dens * (1.5 - shallow_lin.y) * 1.35).exp(),
                                                (-dens * (1.5 - shallow_lin.z) * 1.35).exp(),
                                            );
                                            sun_vis = Vec3::splat(0.22)
                                                + trans * (0.78 * (1.0 - cfg.opacity * 0.55));

                                            // Underwater floor caustics inside translucent shadow
                                            if cfg.refraction_strength > 0.05 && cfg.metallic < 0.5 {
                                                let cp = Vec2::new(hit.x, hit.z) * 5.5
                                                    + Vec2::new(anim_time * 0.7, -anim_time * 0.5);
                                                let w1 = (cp.x * 2.3 + (cp.y * 1.9).sin()).sin();
                                                let w2 = (cp.y * 2.7 - (cp.x * 2.1).cos()).sin();
                                                let focus = ((w1 * w2 + 1.0) * 0.5).powf(5.0);
                                                let caust_mask =
                                                    (dens * 1.4).clamp(0.0, 1.0) * (1.0 - cfg.opacity * 0.65);
                                                caustic_glow = shallow_lin
                                                    * focus
                                                    * caust_mask
                                                    * cfg.refraction_strength
                                                    * 0.55;
                                            }
                                        }
                                    }

                                    let ndotl = light_dir.y.max(0.0);
                                    let floor_lit = floor_albedo
                                        * (Vec3::splat(cfg.ambient_intensity * 0.95)
                                            + sun_vis * (ndotl * cfg.light_intensity * 0.72))
                                        + caustic_glow;

                                    col_lin = col_lin.lerp(floor_lit, edge_fade);
                                    best_dist = (t_floor * ray_dir.dot(cam_fwd)).max(0.05);
                                }
                            }
                        }
                    }

                    // Analytic ray-intersection against enabled Sphere, Box, and Cylinder colliders
                    for col in colliders {
                        if !col.enabled {
                            continue;
                        }
                        if let Some((t_hit, hit_n)) =
                            ray_intersect_analytic_collider(&col.shape, eye_pos, ray_dir)
                        {
                            let z_cam = t_hit * ray_dir.dot(cam_fwd);
                            if z_cam > 0.05 && z_cam < best_dist {
                                best_dist = z_cam;
                                col_lin = shade_solid_obstacle(
                                    hit_n,
                                    -ray_dir,
                                    light_dir,
                                    cfg.light_intensity,
                                    cfg.ambient_intensity,
                                );
                            }
                        }
                    }

                    depth_row[x] = best_dist;
                    let srgb = tonemap_to_srgb(col_lin);
                    let idx = x * 3;
                    rgb_row[idx] = srgb[0];
                    rgb_row[idx + 1] = srgb[1];
                    rgb_row[idx + 2] = srgb[2];
                }
            });

        // Rasterize any custom TriangleMesh colliders into the solid background buffer
        for col in colliders {
            if !col.enabled {
                continue;
            }
            if let ColliderShape::Mesh(ref tri_mesh) = col.shape {
                self.rasterize_solid_triangle_mesh(
                    tri_mesh,
                    view,
                    proj,
                    eye_pos,
                    light_dir,
                    cfg.light_intensity,
                    cfg.ambient_intensity,
                );
            }
        }

        // Snapshot background color buffer so refracted liquid rays can sample distorted background!
        self.bg_rgb.copy_from_slice(&self.rgb_buffer);

        // 2. Project & Rasterize Reconstructed Fluid Surface Mesh
        if !mesh.is_empty() {
            let w_i = width as i32;
            let h_i = height as i32;
            let tri_count = mesh.indices.len() / 3;

            let projected_tris: Vec<ProjectedTri> = (0..tri_count)
                .into_par_iter()
                .filter_map(|ti| {
                    let i0 = mesh.indices[ti * 3] as usize;
                    let i1 = mesh.indices[ti * 3 + 1] as usize;
                    let i2 = mesh.indices[ti * 3 + 2] as usize;

                    let v0 = &mesh.vertices[i0];
                    let v1 = &mesh.vertices[i1];
                    let v2 = &mesh.vertices[i2];

                    let p0 = Vec3::from_array(v0.position);
                    let p1 = Vec3::from_array(v1.position);
                    let p2 = Vec3::from_array(v2.position);

                    let c0_cam = view * p0.extend(1.0);
                    let c1_cam = view * p1.extend(1.0);
                    let c2_cam = view * p2.extend(1.0);

                    // Cull triangles crossing behind camera near plane
                    if c0_cam.z >= -0.04 || c1_cam.z >= -0.04 || c2_cam.z >= -0.04 {
                        return None;
                    }

                    let clip0 = proj * c0_cam;
                    let clip1 = proj * c1_cam;
                    let clip2 = proj * c2_cam;
                    if clip0.w <= 1e-4 || clip1.w <= 1e-4 || clip2.w <= 1e-4 {
                        return None;
                    }

                    let inv_w0 = 1.0 / clip0.w;
                    let inv_w1 = 1.0 / clip1.w;
                    let inv_w2 = 1.0 / clip2.w;

                    let sx0 = ((clip0.x * inv_w0) * 0.5 + 0.5) * w_f;
                    let sy0 = (1.0 - ((clip0.y * inv_w0) * 0.5 + 0.5)) * h_f;
                    let sx1 = ((clip1.x * inv_w1) * 0.5 + 0.5) * w_f;
                    let sy1 = (1.0 - ((clip1.y * inv_w1) * 0.5 + 0.5)) * h_f;
                    let sx2 = ((clip2.x * inv_w2) * 0.5 + 0.5) * w_f;
                    let sy2 = (1.0 - ((clip2.y * inv_w2) * 0.5 + 0.5)) * h_f;

                    // Signed 2D area in screen space (y-down)
                    let area = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0);
                    if area.abs() < 1e-4 {
                        return None;
                    }

                    let x_min = sx0.min(sx1).min(sx2).floor() as i32;
                    let x_max = sx0.max(sx1).max(sx2).ceil() as i32;
                    let y_min = sy0.min(sy1).min(sy2).floor() as i32;
                    let y_max = sy0.max(sy1).max(sy2).ceil() as i32;

                    if x_max < 0 || x_min >= w_i || y_max < 0 || y_min >= h_i {
                        return None;
                    }

                    // Determine front/back orientation using average vertex normal vs view vector
                    let n0 = Vec3::from_array(v0.normal);
                    let n1 = Vec3::from_array(v1.normal);
                    let n2 = Vec3::from_array(v2.normal);
                    let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
                    let avg_n = n0 + n1 + n2;
                    let is_front = avg_n.dot(eye_pos - centroid) >= 0.0;

                    Some(ProjectedTri {
                        sx0,
                        sy0,
                        sx1,
                        sy1,
                        sx2,
                        sy2,
                        inv_w0,
                        inv_w1,
                        inv_w2,
                        z0: -c0_cam.z,
                        z1: -c1_cam.z,
                        z2: -c2_cam.z,
                        p0,
                        p1,
                        p2,
                        n0,
                        n1,
                        n2,
                        spd0: v0.speed,
                        spd1: v1.speed,
                        spd2: v2.speed,
                        fm0: v0.foam,
                        fm1: v1.foam,
                        fm2: v2.foam,
                        inv_area: 1.0 / area,
                        is_front,
                        x_min: x_min.max(0),
                        x_max: x_max.min(w_i - 1),
                        y_min: y_min.max(0),
                        y_max: y_max.min(h_i - 1),
                    })
                })
                .collect();

            // Bucket projected triangles into horizontal bands
            let band_height = 32usize.max((height + 63) / 64);
            let num_bands = (height + band_height - 1) / band_height;
            let mut band_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_bands];
            for (idx, tri) in projected_tris.iter().enumerate() {
                let b0 = (tri.y_min as usize) / band_height;
                let b1 = ((tri.y_max as usize) / band_height).min(num_bands.saturating_sub(1));
                for b in b0..=b1 {
                    band_buckets[b].push(idx as u32);
                }
            }

            let bg_rgb_ref = &self.bg_rgb;
            let rgb_chunks = self.rgb_buffer.chunks_mut(band_height * width * 3);
            let solid_depth_chunks = self.depth_buffer.chunks_mut(band_height * width);
            let front_chunks = self.front_depth.chunks_mut(band_height * width);
            let back_chunks = self.back_depth.chunks_mut(band_height * width);
            let hit_chunks = self.surf_hits.chunks_mut(band_height * width);

            rgb_chunks
                .zip(solid_depth_chunks)
                .zip(front_chunks.zip(back_chunks).zip(hit_chunks))
                .zip(band_buckets.iter())
                .enumerate()
                .par_bridge()
                .for_each(
                    |(
                        band_idx,
                        (((rgb_band, solid_depth_band), ((front_band, back_band), hit_band)), bucket),
                    )| {
                        if bucket.is_empty() {
                            return;
                        }
                        let band_y_start = (band_idx * band_height) as i32;
                        let band_rows = rgb_band.len() / (width * 3);
                        let band_y_end = band_y_start + band_rows as i32;

                        front_band.fill(f32::INFINITY);
                        back_band.fill(0.0);

                        // Pass A: Rasterize front & back fluid depths + interpolated surface attributes
                        for &t_idx in bucket {
                            let tri = &projected_tris[t_idx as usize];
                            let ly_min = tri.y_min.max(band_y_start);
                            let ly_max = tri.y_max.min(band_y_end - 1);

                            for y in ly_min..=ly_max {
                                let py = y as f32 + 0.5;
                                let row_off = (y - band_y_start) as usize * width;

                                for x in tri.x_min..=tri.x_max {
                                    let px = x as f32 + 0.5;

                                    let w0 = ((tri.sx1 - px) * (tri.sy2 - py)
                                        - (tri.sy1 - py) * (tri.sx2 - px))
                                        * tri.inv_area;
                                    if w0 < -1e-5 {
                                        continue;
                                    }
                                    let w1 = ((tri.sx2 - px) * (tri.sy0 - py)
                                        - (tri.sy2 - py) * (tri.sx0 - px))
                                        * tri.inv_area;
                                    if w1 < -1e-5 {
                                        continue;
                                    }
                                    let w2 = 1.0 - w0 - w1;
                                    if w2 < -1e-5 {
                                        continue;
                                    }

                                    // Perspective-correct barycentric interpolation
                                    let pw0 = w0.max(0.0) * tri.inv_w0;
                                    let pw1 = w1.max(0.0) * tri.inv_w1;
                                    let pw2 = w2.max(0.0) * tri.inv_w2;
                                    let pw_sum = pw0 + pw1 + pw2;
                                    if pw_sum <= 1e-12 {
                                        continue;
                                    }
                                    let inv_sum = 1.0 / pw_sum;
                                    let b0 = pw0 * inv_sum;
                                    let b1 = pw1 * inv_sum;
                                    let b2 = pw2 * inv_sum;

                                    let z_dist = b0 * tri.z0 + b1 * tri.z1 + b2 * tri.z2;
                                    let pix = row_off + x as usize;
                                    let solid_z = solid_depth_band[pix];

                                    if z_dist >= solid_z {
                                        continue;
                                    }

                                    if tri.is_front {
                                        if z_dist < front_band[pix] {
                                            front_band[pix] = z_dist;
                                            let wp = tri.p0 * b0 + tri.p1 * b1 + tri.p2 * b2;
                                            let wn = tri.n0 * b0 + tri.n1 * b1 + tri.n2 * b2;
                                            hit_band[pix] = SurfaceHit {
                                                pos: wp.to_array(),
                                                normal: wn.to_array(),
                                                speed: b0 * tri.spd0 + b1 * tri.spd1 + b2 * tri.spd2,
                                                foam: b0 * tri.fm0 + b1 * tri.fm1 + b2 * tri.fm2,
                                            };
                                        }
                                    } else {
                                        if z_dist > back_band[pix] {
                                            back_band[pix] = z_dist;
                                        }
                                        // Fallback if camera views an open thin sheet from behind
                                        if front_band[pix].is_infinite() {
                                            front_band[pix] = z_dist;
                                            let wp = tri.p0 * b0 + tri.p1 * b1 + tri.p2 * b2;
                                            let wn = -(tri.n0 * b0 + tri.n1 * b1 + tri.n2 * b2);
                                            hit_band[pix] = SurfaceHit {
                                                pos: wp.to_array(),
                                                normal: wn.to_array(),
                                                speed: b0 * tri.spd0 + b1 * tri.spd1 + b2 * tri.spd2,
                                                foam: b0 * tri.fm0 + b1 * tri.fm1 + b2 * tri.fm2,
                                            };
                                        }
                                    }
                                }
                            }
                        }

                        // Pass B: Physically-Based Optical Liquid Shading
                        let ior = cfg.ior.max(1.01);
                        let base_f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
                        let rough = cfg.roughness.clamp(0.02, 1.0);
                        let alpha = rough * rough;
                        let alpha2 = alpha * alpha;

                        for ly in 0..band_rows {
                            let y = band_y_start as usize + ly;
                            let row_off = ly * width;

                            for x in 0..width {
                                let pix = row_off + x;
                                let front_z = front_band[pix];
                                let solid_z = solid_depth_band[pix];
                                if front_z >= solid_z {
                                    continue;
                                }

                                let hit = &hit_band[pix];
                                let world_pos = Vec3::from_array(hit.pos);
                                let mut base_n = Vec3::from_array(hit.normal).normalize_or_zero();
                                if base_n.length_squared() < 0.5 {
                                    base_n = Vec3::Y;
                                }

                                let v_dir = (eye_pos - world_pos).normalize_or_zero();
                                if base_n.dot(v_dir) < 0.0 {
                                    base_n = -base_n;
                                }

                                // Evaluate triplanar procedural texture (bump, caustics, foam, grid)
                                let tex = cfg.evaluate_texture(
                                    world_pos,
                                    base_n,
                                    hit.speed,
                                    anim_time,
                                );
                                let mut n = tex.normal;
                                if n.dot(v_dir) < 0.02 {
                                    n = (n + v_dir * 0.15).normalize_or_zero();
                                }

                                let ndotv = n.dot(v_dir).clamp(0.001, 1.0);
                                let ndotl = n.dot(light_dir).max(0.0);

                                // Exact ray-marched / rasterized optical liquid thickness
                                let back_z = back_band[pix].min(solid_z);
                                let thickness = if back_z > front_z + 0.004 {
                                    (back_z - front_z).min(2.8)
                                } else {
                                    0.07 + (1.0 - ndotv) * 0.25
                                };

                                // Schlick Fresnel + Metallic blend
                                let shallow_tex = (shallow_lin * tex.albedo_scale).clamp(Vec3::ZERO, Vec3::splat(2.0));
                                let deep_tex = (deep_lin * tex.albedo_scale).clamp(Vec3::ZERO, Vec3::splat(2.0));
                                let f0_vec = Vec3::splat(base_f0).lerp(shallow_tex, cfg.metallic);
                                let fresnel = f0_vec
                                    + (Vec3::ONE - f0_vec) * (1.0 - ndotv).powi(5);

                                // Screen-Space Refraction + Beer-Lambert Volume Absorption
                                let refr_scale = cfg.refraction_strength
                                    * (1.0 - cfg.metallic)
                                    * (height as f32 * 0.032)
                                    * (thickness / (thickness + 0.25));
                                let rx = (x as f32 - n.x * refr_scale)
                                    .round()
                                    .clamp(0.0, (width - 1) as f32) as usize;
                                let ry = (y as f32 + n.y * refr_scale)
                                    .round()
                                    .clamp(0.0, (height - 1) as f32) as usize;

                                let bg_idx = (ry * width + rx) * 3;
                                let bg_srgb = [
                                    bg_rgb_ref[bg_idx],
                                    bg_rgb_ref[bg_idx + 1],
                                    bg_rgb_ref[bg_idx + 2],
                                ];
                                let bg_lin = srgb_to_linear(bg_srgb);

                                // Beer-Lambert exponential absorption across fluid thickness (in linear optical space)
                                let opt_depth = thickness * cfg.absorption;
                                let depth_mix = 1.0 - (-opt_depth * 0.90).exp();
                                let body_tint = shallow_tex.lerp(deep_tex, depth_mix);
                                let srgb_tint = shallow_srgb.lerp(deep_srgb, depth_mix);

                                let absorb_sigma = (Vec3::splat(1.04) - srgb_tint * 0.92)
                                    .max(Vec3::splat(0.04))
                                    * (cfg.absorption * 1.85);
                                let transmittance = Vec3::new(
                                    (-absorb_sigma.x * thickness).exp(),
                                    (-absorb_sigma.y * thickness).exp(),
                                    (-absorb_sigma.z * thickness).exp(),
                                );

                                // Subsurface volume in-scattering + Beer-Lambert filtered background refraction
                                let hemi = 0.5 + 0.5 * n.y;
                                let body_scatter = body_tint
                                    * (cfg.ambient_intensity * (0.42 + 0.58 * hemi)
                                        + ndotl * cfg.light_intensity * 0.58);

                                let refracted_col = if cfg.metallic > 0.85 {
                                    body_scatter * 0.12
                                } else {
                                    let filtered_bg = bg_lin * transmittance;
                                    // Clear/translucent liquids (< 0.88 opacity) transmit crisp refracted floor/obstacles,
                                    // while turbid liquids (e.g. Milk >= 0.90) scatter throughout their volume.
                                    let turbidity = if cfg.opacity >= 0.88 {
                                        cfg.opacity.clamp(0.0, 1.0)
                                    } else {
                                        (cfg.opacity * cfg.opacity * 0.34).clamp(0.02, 0.45)
                                    };
                                    let volume_inscatter =
                                        body_scatter * (Vec3::ONE - transmittance) * 0.55;
                                    (filtered_bg + volume_inscatter).lerp(body_scatter, turbidity)
                                };

                                // Studio Environment Reflection (blurred by roughness)
                                let refl_dir = (-v_dir) - n * (2.0 * (-v_dir).dot(n));
                                let env_col = sample_studio_sky(refl_dir, light_dir, rough)
                                    * cfg.env_reflection;

                                // GGX Direct Sun Specular
                                let h_vec = (v_dir + light_dir).normalize_or_zero();
                                let ndoth = n.dot(h_vec).max(0.0);
                                let denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
                                let d_ggx = alpha2 / (std::f32::consts::PI * denom * denom + 1e-6);
                                let k = (rough + 1.0).powi(2) * 0.125;
                                let g_v = ndotv / (ndotv * (1.0 - k) + k);
                                let g_l = ndotl / (ndotl * (1.0 - k) + k);
                                let spec_brdf = (d_ggx * g_v * g_l * fresnel)
                                    / (4.0 * ndotv * ndotl + 1e-4);
                                let sun_spec = spec_brdf
                                    * ndotl
                                    * cfg.light_intensity
                                    * cfg.specular_intensity;

                                // Subsurface Wave-Crest Scattering (backlit thin sheets glow)
                                let back_light = (-light_dir + n * 0.45)
                                    .normalize_or_zero()
                                    .dot(v_dir)
                                    .max(0.0)
                                    .powi(3);
                                let thin_factor = (-thickness * 2.2).exp();
                                let sss_col = shallow_tex
                                    * back_light
                                    * thin_factor
                                    * cfg.subsurface_scattering
                                    * cfg.light_intensity
                                    * 0.85;

                                // Combine Dielectric / Metallic Reflection + Transmission + SSS + Caustic
                                let mut final_lin = refracted_col * (Vec3::ONE - fresnel)
                                    + env_col * fresnel
                                    + sun_spec
                                    + sss_col
                                    + shallow_tex * tex.emissive_caustic * cfg.light_intensity;

                                // Aerated Whitewater Crest Foam
                                let foam_amount = ((hit.foam * 0.85 + tex.foam_boost)
                                    * cfg.foam_intensity)
                                    .clamp(0.0, 0.92);
                                if foam_amount > 0.005 {
                                    let foam_lit = Vec3::new(0.94, 0.97, 1.0)
                                        * (cfg.ambient_intensity * 0.85
                                            + (0.35 + 0.65 * ndotl) * cfg.light_intensity * 0.68);
                                    final_lin = final_lin.lerp(foam_lit, foam_amount);
                                }

                                solid_depth_band[pix] = front_z;
                                let out_srgb = tonemap_to_srgb(final_lin);
                                let rgb_idx = pix * 3;
                                rgb_band[rgb_idx] = out_srgb[0];
                                rgb_band[rgb_idx + 1] = out_srgb[1];
                                rgb_band[rgb_idx + 2] = out_srgb[2];
                            }
                        }
                    },
                );
        }

        // 3. Draw subtle depth-aware Chamber Wireframe on top
        self.draw_chamber_wireframe_depth(chamber, &bounds, view, proj, [78, 136, 205]);
    }

    fn rasterize_solid_triangle_mesh(
        &mut self,
        tri_mesh: &crate::sph::mesh::TriangleMesh,
        view: Mat4,
        proj: Mat4,
        eye_pos: Vec3,
        light_dir: Vec3,
        light_intensity: f32,
        ambient_intensity: f32,
    ) {
        let rot = Mat3::from_euler(
            glam::EulerRot::YXZ,
            tri_mesh.rotation.y,
            tri_mesh.rotation.x,
            tri_mesh.rotation.z,
        );
        let s = tri_mesh.scale;
        let pos = tri_mesh.position;
        let w_f = self.width as f32;
        let h_f = self.height as f32;
        let w_i = self.width as i32;
        let h_i = self.height as i32;

        for tri in &tri_mesh.triangles {
            let p0 = pos + rot * (tri.a * s);
            let p1 = pos + rot * (tri.b * s);
            let p2 = pos + rot * (tri.c * s);
            let wn = (rot * tri.normal).normalize_or_zero();

            let c0 = view * p0.extend(1.0);
            let c1 = view * p1.extend(1.0);
            let c2 = view * p2.extend(1.0);
            if c0.z >= -0.04 || c1.z >= -0.04 || c2.z >= -0.04 {
                continue;
            }
            let cl0 = proj * c0;
            let cl1 = proj * c1;
            let cl2 = proj * c2;
            if cl0.w <= 1e-4 || cl1.w <= 1e-4 || cl2.w <= 1e-4 {
                continue;
            }
            let sx0 = ((cl0.x / cl0.w) * 0.5 + 0.5) * w_f;
            let sy0 = (1.0 - ((cl0.y / cl0.w) * 0.5 + 0.5)) * h_f;
            let sx1 = ((cl1.x / cl1.w) * 0.5 + 0.5) * w_f;
            let sy1 = (1.0 - ((cl1.y / cl1.w) * 0.5 + 0.5)) * h_f;
            let sx2 = ((cl2.x / cl2.w) * 0.5 + 0.5) * w_f;
            let sy2 = (1.0 - ((cl2.y / cl2.w) * 0.5 + 0.5)) * h_f;

            let area = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0);
            if area.abs() < 1e-4 {
                continue;
            }
            let inv_area = 1.0 / area;

            let x_min = (sx0.min(sx1).min(sx2).floor() as i32).max(0);
            let x_max = (sx0.max(sx1).max(sx2).ceil() as i32).min(w_i - 1);
            let y_min = (sy0.min(sy1).min(sy2).floor() as i32).max(0);
            let y_max = (sy0.max(sy1).max(sy2).ceil() as i32).min(h_i - 1);

            let centroid = (p0 + p1 + p2) * (1.0 / 3.0);
            let v_dir = (eye_pos - centroid).normalize_or_zero();
            let face_n = if wn.dot(v_dir) < 0.0 { -wn } else { wn };
            let srgb = tonemap_to_srgb(shade_solid_obstacle(
                face_n,
                v_dir,
                light_dir,
                light_intensity,
                ambient_intensity,
            ));

            for y in y_min..=y_max {
                let py = y as f32 + 0.5;
                let row = y as usize * self.width;
                for x in x_min..=x_max {
                    let px = x as f32 + 0.5;
                    let w0 = ((sx1 - px) * (sy2 - py) - (sy1 - py) * (sx2 - px)) * inv_area;
                    let w1 = ((sx2 - px) * (sy0 - py) - (sy2 - py) * (sx0 - px)) * inv_area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        let z_dist = -(w0 * c0.z + w1 * c1.z + w2 * c2.z);
                        let pix = row + x as usize;
                        if z_dist < self.depth_buffer[pix] {
                            self.depth_buffer[pix] = z_dist;
                            let idx = pix * 3;
                            self.rgb_buffer[idx] = srgb[0];
                            self.rgb_buffer[idx + 1] = srgb[1];
                            self.rgb_buffer[idx + 2] = srgb[2];
                        }
                    }
                }
            }
        }
    }

    fn calc_particle_color(
        p: &Particle,
        color_mode: ColorMode,
        color_map: ColorMapType,
        rest_density: f32,
        bounds: &Aabb,
    ) -> [u8; 3] {
        let t = match color_mode {
            ColorMode::Velocity => (p.speed() / 2.4).clamp(0.0, 1.0),
            ColorMode::Pressure => (p.pressure / (rest_density * 0.4)).clamp(0.0, 1.0),
            ColorMode::Density => {
                let dev = (p.density - rest_density).abs() / rest_density;
                (dev * 4.0).clamp(0.0, 1.0)
            }
            ColorMode::Height => {
                ((p.position.y - bounds.min.y) / (bounds.max.y - bounds.min.y)).clamp(0.0, 1.0)
            }
            ColorMode::KineticEnergy => (p.kinetic_energy() / 4.0).clamp(0.0, 1.0),
            ColorMode::SingleColor => return [30, 140, 225],
        };
        ColorMap::sample(color_map, t)
    }

    fn draw_chamber_wireframe(
        &mut self,
        chamber: &crate::sph::chamber::Chamber,
        bounds: &Aabb,
        view_proj: Mat4,
    ) {
        let color = [80, 140, 210];
        for (p0, p1) in chamber.generate_wireframe_lines(bounds) {
            self.draw_3d_line(p0, p1, view_proj, color);
        }
    }

    fn draw_chamber_wireframe_depth(
        &mut self,
        chamber: &crate::sph::chamber::Chamber,
        bounds: &Aabb,
        view: Mat4,
        proj: Mat4,
        color: [u8; 3],
    ) {
        for (p0, p1) in chamber.generate_wireframe_lines(bounds) {
            let c0_cam = view * p0.extend(1.0);
            let c1_cam = view * p1.extend(1.0);
            if c0_cam.z >= -0.04 || c1_cam.z >= -0.04 {
                continue;
            }
            let c0 = proj * c0_cam;
            let c1 = proj * c1_cam;
            if c0.w <= 0.0 || c1.w <= 0.0 {
                continue;
            }
            let w = self.width as f32;
            let h = self.height as f32;
            let x0 = ((c0.x / c0.w) * 0.5 + 0.5) * w;
            let y0 = (1.0 - ((c0.y / c0.w) * 0.5 + 0.5)) * h;
            let x1 = ((c1.x / c1.w) * 0.5 + 0.5) * w;
            let y1 = (1.0 - ((c1.y / c1.w) * 0.5 + 0.5)) * h;
            self.draw_depth_line(
                x0 as i32,
                y0 as i32,
                -c0_cam.z,
                x1 as i32,
                y1 as i32,
                -c1_cam.z,
                color,
            );
        }
    }

    fn draw_depth_line(
        &mut self,
        mut x0: i32,
        mut y0: i32,
        z0: f32,
        x1: i32,
        y1: i32,
        z1: f32,
        color: [u8; 3],
    ) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let steps = dx.max(-dy).max(1) as f32;
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let w = self.width as i32;
        let h = self.height as i32;
        let mut step = 0.0f32;

        loop {
            if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
                let t = (step / steps).clamp(0.0, 1.0);
                let z = z0 + (z1 - z0) * t;
                let pix = y0 as usize * self.width + x0 as usize;
                let idx = pix * 3;
                let alpha = if z <= self.depth_buffer[pix] + 0.015 {
                    0.55f32
                } else {
                    0.12f32
                };
                self.rgb_buffer[idx] =
                    (self.rgb_buffer[idx] as f32 * (1.0 - alpha) + color[0] as f32 * alpha) as u8;
                self.rgb_buffer[idx + 1] = (self.rgb_buffer[idx + 1] as f32 * (1.0 - alpha)
                    + color[1] as f32 * alpha) as u8;
                self.rgb_buffer[idx + 2] = (self.rgb_buffer[idx + 2] as f32 * (1.0 - alpha)
                    + color[2] as f32 * alpha) as u8;
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            step += 1.0;
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    fn draw_3d_line(&mut self, p0: Vec3, p1: Vec3, view_proj: Mat4, color: [u8; 3]) {
        let c0 = view_proj * Vec4::new(p0.x, p0.y, p0.z, 1.0);
        let c1 = view_proj * Vec4::new(p1.x, p1.y, p1.z, 1.0);

        if c0.w <= 0.0 || c1.w <= 0.0 {
            return;
        }

        let w = self.width as f32;
        let h = self.height as f32;

        let x0 = ((c0.x / c0.w) * 0.5 + 0.5) * w;
        let y0 = (1.0 - ((c0.y / c0.w) * 0.5 + 0.5)) * h;
        let x1 = ((c1.x / c1.w) * 0.5 + 0.5) * w;
        let y1 = (1.0 - ((c1.y / c1.w) * 0.5 + 0.5)) * h;

        self.draw_bresenham_line(x0 as i32, y0 as i32, x1 as i32, y1 as i32, color);
    }

    fn draw_bresenham_line(
        &mut self,
        mut x0: i32,
        mut y0: i32,
        x1: i32,
        y1: i32,
        color: [u8; 3],
    ) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let w = self.width as i32;
        let h = self.height as i32;

        loop {
            if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
                let idx = (y0 as usize * self.width + x0 as usize) * 3;
                self.rgb_buffer[idx] = color[0];
                self.rgb_buffer[idx + 1] = color[1];
                self.rgb_buffer[idx + 2] = color[2];
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }
}

#[inline]
fn srgb_to_linear(c: [u8; 3]) -> Vec3 {
    let r = (c[0] as f32 / 255.0).powf(2.2);
    let g = (c[1] as f32 / 255.0).powf(2.2);
    let b = (c[2] as f32 / 255.0).powf(2.2);
    Vec3::new(r, g, b)
}

#[inline]
fn tonemap_to_srgb(col: Vec3) -> [u8; 3] {
    // ACES-inspired filmic curve + gamma 1/2.2
    let x = col.max(Vec3::ZERO);
    let mapped = (x * (2.51 * x + Vec3::splat(0.03)))
        / (x * (2.43 * x + Vec3::splat(0.59)) + Vec3::splat(0.14));
    let srgb = mapped.clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2);
    [
        (srgb.x * 255.0 + 0.5) as u8,
        (srgb.y * 255.0 + 0.5) as u8,
        (srgb.z * 255.0 + 0.5) as u8,
    ]
}

#[inline]
fn sample_studio_sky(refl_dir: Vec3, light_dir: Vec3, roughness: f32) -> Vec3 {
    let r = refl_dir.normalize_or_zero();
    let up = r.y.clamp(-1.0, 1.0);

    // Horizon-to-zenith studio sky + dark floor bounce
    let horizon = Vec3::new(0.24, 0.36, 0.52);
    let zenith = Vec3::new(0.08, 0.16, 0.30);
    let ground = Vec3::new(0.025, 0.032, 0.045);

    let mut sky = if up >= 0.0 {
        horizon.lerp(zenith, up.powf(0.65))
    } else {
        horizon.lerp(ground, (-up).powf(0.45))
    };

    // Overhead studio softbox panel reflection (gives liquid that realistic glossy studio look)
    let softbox_dir = Vec3::new(-0.25, 0.92, 0.30).normalize();
    let soft_dot = r.dot(softbox_dir).max(0.0);
    let sharpness = (1.0 - roughness).clamp(0.05, 1.0);
    let softbox = soft_dot.powf(4.0 + sharpness * 18.0) * (0.45 + 0.55 * sharpness);
    sky += Vec3::new(0.65, 0.78, 0.95) * softbox;

    // Warm sun bloom in sky
    let sun_dot = r.dot(light_dir).max(0.0);
    sky += Vec3::new(1.0, 0.95, 0.86) * sun_dot.powf(8.0 + sharpness * 32.0) * 0.45;

    sky
}

#[inline]
fn shade_solid_obstacle(
    normal: Vec3,
    view_dir: Vec3,
    light_dir: Vec3,
    light_intensity: f32,
    ambient_intensity: f32,
) -> Vec3 {
    let albedo = Vec3::new(0.28, 0.31, 0.37);
    let ndotl = normal.dot(light_dir).max(0.0);
    let hemi = 0.5 + 0.5 * normal.y;
    let h = (view_dir + light_dir).normalize_or_zero();
    let spec = normal.dot(h).max(0.0).powi(28) * 0.22;
    let rim = (1.0 - normal.dot(view_dir).max(0.0)).powi(4) * 0.08;
    albedo * (ambient_intensity * (0.55 + 0.45 * hemi) + ndotl * light_intensity * 0.78)
        + Vec3::splat(spec * light_intensity)
        + Vec3::new(0.12, 0.22, 0.35) * rim
}

#[inline]
fn ray_intersect_analytic_collider(
    shape: &ColliderShape,
    ro: Vec3,
    rd: Vec3,
) -> Option<(f32, Vec3)> {
    match shape {
        ColliderShape::Sphere { center, radius } => {
            let oc = ro - *center;
            let b = oc.dot(rd);
            let c = oc.length_squared() - radius * radius;
            let disc = b * b - c;
            if disc < 0.0 {
                return None;
            }
            let s = disc.sqrt();
            let t = -b - s;
            if t > 0.05 {
                let hit = ro + rd * t;
                Some((t, (hit - *center) / *radius))
            } else {
                None
            }
        }
        ColliderShape::Box { aabb } => {
            let inv_d = Vec3::new(
                if rd.x.abs() > 1e-8 { 1.0 / rd.x } else { 1e8 },
                if rd.y.abs() > 1e-8 { 1.0 / rd.y } else { 1e8 },
                if rd.z.abs() > 1e-8 { 1.0 / rd.z } else { 1e8 },
            );
            let t0 = (aabb.min - ro) * inv_d;
            let t1 = (aabb.max - ro) * inv_d;
            let tmin_v = t0.min(t1);
            let tmax_v = t0.max(t1);
            let t_enter = tmin_v.x.max(tmin_v.y).max(tmin_v.z);
            let t_exit = tmax_v.x.min(tmax_v.y).min(tmax_v.z);
            if t_enter <= t_exit && t_enter > 0.05 {
                let hit = ro + rd * t_enter;
                let c = aabb.center();
                let half = aabb.size() * 0.5;
                let rel = (hit - c) / half.max(Vec3::splat(1e-5));
                let ar = rel.abs();
                let n = if ar.x >= ar.y && ar.x >= ar.z {
                    Vec3::new(rel.x.signum(), 0.0, 0.0)
                } else if ar.y >= ar.x && ar.y >= ar.z {
                    Vec3::new(0.0, rel.y.signum(), 0.0)
                } else {
                    Vec3::new(0.0, 0.0, rel.z.signum())
                };
                Some((t_enter, n))
            } else {
                None
            }
        }
        ColliderShape::Cylinder {
            center,
            radius,
            height,
        } => {
            let half_h = *height * 0.5;
            let y_min = center.y - half_h;
            let y_max = center.y + half_h;
            let ox = ro.x - center.x;
            let oz = ro.z - center.z;
            let a = rd.x * rd.x + rd.z * rd.z;
            let mut best: Option<(f32, Vec3)> = None;

            if a > 1e-8 {
                let b = ox * rd.x + oz * rd.z;
                let c = ox * ox + oz * oz - radius * radius;
                let disc = b * b - a * c;
                if disc >= 0.0 {
                    let t = (-b - disc.sqrt()) / a;
                    if t > 0.05 {
                        let y = ro.y + rd.y * t;
                        if y >= y_min && y <= y_max {
                            let hit = ro + rd * t;
                            let n = Vec3::new(hit.x - center.x, 0.0, hit.z - center.z).normalize_or_zero();
                            best = Some((t, n));
                        }
                    }
                }
            }

            if rd.y.abs() > 1e-6 {
                for &(cap_y, ny) in &[(y_max, 1.0f32), (y_min, -1.0f32)] {
                    let t = (cap_y - ro.y) / rd.y;
                    if t > 0.05 && best.map_or(true, |(bt, _)| t < bt) {
                        let hx = ox + rd.x * t;
                        let hz = oz + rd.z * t;
                        if hx * hx + hz * hz <= radius * radius {
                            best = Some((t, Vec3::new(0.0, ny, 0.0)));
                        }
                    }
                }
            }
            best
        }
        ColliderShape::Mesh(_) => None,
    }
}

/// Streams rawvideo frames into ffmpeg to encode H.264 MP4
pub struct Mp4Encoder {
    child: Child,
    frame_bytes: usize,
}

impl Mp4Encoder {
    pub fn new<P: AsRef<Path>>(
        output_path: P,
        width: usize,
        height: usize,
        fps: u32,
    ) -> Result<Self, String> {
        let output_str = output_path
            .as_ref()
            .to_str()
            .ok_or_else(|| "Invalid output path".to_string())?;

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y", // Overwrite output file
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", width, height),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn ffmpeg: {}. Ensure ffmpeg is installed.", e))?;

        Ok(Self {
            child,
            frame_bytes: width * height * 3,
        })
    }

    pub fn write_frame(&mut self, rgb_buffer: &[u8]) -> Result<(), String> {
        if rgb_buffer.len() != self.frame_bytes {
            return Err("Frame buffer size mismatch".to_string());
        }

        if let Some(ref mut stdin) = self.child.stdin {
            stdin
                .write_all(rgb_buffer)
                .map_err(|e| format!("Failed to write frame to ffmpeg: {}", e))?;
            Ok(())
        } else {
            Err("ffmpeg stdin not available".to_string())
        }
    }

    pub fn finish(mut self) -> Result<(), String> {
        drop(self.child.stdin.take());
        let status = self
            .child
            .wait()
            .map_err(|e| format!("Failed to wait on ffmpeg process: {}", e))?;

        if status.success() {
            Ok(())
        } else {
            Err(format!("ffmpeg exited with status: {}", status))
        }
    }
}
