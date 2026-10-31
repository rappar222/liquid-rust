use crate::math::Aabb;
use crate::render::{Camera, ColorMap, ColorMapType, ColorMode};
use crate::sph::particle::Particle;
use glam::{Mat4, Vec3, Vec4};
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};

/// High-performance software 3D rasterizer for offline/headless frame rendering
pub struct SoftwareRenderer {
    pub width: usize,
    pub height: usize,
    pub rgb_buffer: Vec<u8>,
    pub depth_buffer: Vec<f32>,
}

impl SoftwareRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgb_buffer: vec![0; width * height * 3],
            depth_buffer: vec![f32::INFINITY; width * height],
        }
    }

    /// Clear buffer with dark gradient background
    pub fn clear(&mut self) {
        let w = self.width;
        let h = self.height;

        for y in 0..h {
            let t = y as f32 / h as f32;
            let r = (18.0 + t * 14.0) as u8;
            let g = (20.0 + t * 16.0) as u8;
            let b = (26.0 + t * 20.0) as u8;

            let row_start = y * w * 3;
            for x in 0..w {
                let idx = row_start + x * 3;
                self.rgb_buffer[idx] = r;
                self.rgb_buffer[idx + 1] = g;
                self.rgb_buffer[idx + 2] = b;
            }
        }

        self.depth_buffer.fill(f32::INFINITY);
    }

    /// Render simulation frame (bounding box and 3D shaded spheres)
    pub fn render_frame(
        &mut self,
        particles: &[Particle],
        particle_radius: f32,
        bounds: Aabb,
        camera: &Camera,
        color_mode: ColorMode,
        color_map: ColorMapType,
        rest_density: f32,
    ) {
        self.clear();

        let aspect = self.width as f32 / self.height as f32;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let view_proj = proj * view;

        // 1. Draw Bounding Box Wireframe
        self.draw_box_wireframe(&bounds, view_proj);

        // 2. Render 3D Lit Spheres
        let fov_y = camera.fov_y;
        let tan_half_fov = (fov_y * 0.5).tan();
        let light_dir = Vec3::new(0.4, 0.8, 0.5).normalize();
        let half_dir = (light_dir + Vec3::new(0.0, 0.0, 1.0)).normalize();

        let w = self.width as f32;
        let h = self.height as f32;
        let w_i = self.width as i32;
        let h_i = self.height as i32;

        for p in particles {
            // Transform to camera eye space
            let p_world = Vec4::new(p.position.x, p.position.y, p.position.z, 1.0);
            let p_cam = view * p_world;

            if p_cam.z >= -0.1 {
                continue; // Behind camera near plane
            }

            let z_dist = -p_cam.z;
            let p_clip = proj * p_cam;

            if p_clip.w <= 0.0 {
                continue;
            }

            let ndc_x = p_clip.x / p_clip.w;
            let ndc_y = p_clip.y / p_clip.w;

            let px = (ndc_x * 0.5 + 0.5) * w;
            let py = (1.0 - (ndc_y * 0.5 + 0.5)) * h;

            // Projected radius in pixels
            let r_px = (particle_radius / (z_dist * tan_half_fov)) * (h * 0.5);
            if r_px < 1.0 {
                continue;
            }

            let r_px_ceil = r_px.ceil() as i32;
            let x_min = (px.floor() as i32 - r_px_ceil).max(0);
            let x_max = (px.ceil() as i32 + r_px_ceil).min(w_i - 1);
            let y_min = (py.floor() as i32 - r_px_ceil).max(0);
            let y_max = (py.ceil() as i32 + r_px_ceil).min(h_i - 1);

            let inv_r_px = 1.0 / r_px;

            // Compute particle color
            let base_rgb = self.compute_particle_color(p, color_mode, color_map, rest_density, &bounds);
            let base_r = base_rgb[0] as f32;
            let base_g = base_rgb[1] as f32;
            let base_b = base_rgb[2] as f32;

            for y in y_min..=y_max {
                let dy = (y as f32 + 0.5 - py) * inv_r_px;
                let dy2 = dy * dy;
                if dy2 > 1.0 {
                    continue;
                }

                let row_idx = y as usize * self.width;

                for x in x_min..=x_max {
                    let dx = (x as f32 + 0.5 - px) * inv_r_px;
                    let r2 = dx * dx + dy2;

                    if r2 <= 1.0 {
                        let z_norm = (1.0 - r2).sqrt();
                        let depth = z_dist - z_norm * particle_radius;

                        let pixel_idx = row_idx + x as usize;
                        if depth < self.depth_buffer[pixel_idx] {
                            self.depth_buffer[pixel_idx] = depth;

                            let normal = Vec3::new(dx, -dy, z_norm);
                            let diff = normal.dot(light_dir).max(0.0);
                            let spec = normal.dot(half_dir).max(0.0).powi(24);
                            let rim = (1.0 - z_norm).powi(3) * 0.25;

                            let lit_r = ((base_r * (0.35 + 0.65 * diff) + 128.0 * spec + rim * 255.0)
                                .clamp(0.0, 255.0)) as u8;
                            let lit_g = ((base_g * (0.35 + 0.65 * diff) + 128.0 * spec + rim * 255.0)
                                .clamp(0.0, 255.0)) as u8;
                            let lit_b = ((base_b * (0.35 + 0.65 * diff) + 128.0 * spec + rim * 255.0)
                                .clamp(0.0, 255.0)) as u8;

                            let rgb_idx = pixel_idx * 3;
                            self.rgb_buffer[rgb_idx] = lit_r;
                            self.rgb_buffer[rgb_idx + 1] = lit_g;
                            self.rgb_buffer[rgb_idx + 2] = lit_b;
                        }
                    }
                }
            }
        }
    }

    fn compute_particle_color(
        &self,
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
            ColorMode::Height => ((p.position.y - bounds.min.y) / (bounds.max.y - bounds.min.y)).clamp(0.0, 1.0),
            ColorMode::KineticEnergy => (p.kinetic_energy() / 4.0).clamp(0.0, 1.0),
            ColorMode::SingleColor => return [30, 140, 225],
        };
        ColorMap::sample(color_map, t)
    }

    fn draw_box_wireframe(&mut self, bounds: &Aabb, view_proj: Mat4) {
        let corners = bounds.corners();
        let indices = Aabb::edge_indices();
        let color = [80, 140, 210];

        for (i0, i1) in indices {
            self.draw_3d_line(corners[i0], corners[i1], view_proj, color);
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

    fn draw_bresenham_line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
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

        let child = Command::new("ffmpeg")
            .args([
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
                output_str,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
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
