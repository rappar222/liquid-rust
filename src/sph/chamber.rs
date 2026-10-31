use glam::Vec3;
use std::f32::consts::PI;
use crate::math::Aabb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChamberShape {
    Box,
    Sphere,
    Bathtub,
}

impl ChamberShape {
    pub const ALL: [ChamberShape; 3] = [
        ChamberShape::Box,
        ChamberShape::Sphere,
        ChamberShape::Bathtub,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ChamberShape::Box => "Box (Cuboid)",
            ChamberShape::Sphere => "Sphere (Globe)",
            ChamberShape::Bathtub => "Bathtub (Tub)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Chamber {
    pub shape: ChamberShape,
    /// Scale factor along (X, Y, Z) - default (1.0, 1.0, 1.0)
    pub scale: Vec3,
    /// Base size for standard box / tub: default (1.6, 1.2, 0.9)
    pub base_size: Vec3,
    /// Bathtub backrest slope horizontal offset ratio (relative to length): default 0.28
    pub tub_slope_ratio: f32,
    /// Bathtub bottom fillet radius ratio (relative to width): default 0.16
    pub tub_fillet_ratio: f32,
}

impl Default for Chamber {
    fn default() -> Self {
        Self::new(ChamberShape::Box)
    }
}

impl Chamber {
    pub fn new(shape: ChamberShape) -> Self {
        Self {
            shape,
            scale: Vec3::ONE,
            base_size: Vec3::new(1.6, 1.2, 0.9),
            tub_slope_ratio: 0.28,
            tub_fillet_ratio: 0.16,
        }
    }

    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    pub fn set_shape(&mut self, shape: ChamberShape) {
        self.shape = shape;
    }

    pub fn compute_bounds(&self) -> Aabb {
        let size = self.base_size * self.scale;
        match self.shape {
            ChamberShape::Box => {
                let half_x = size.x * 0.5;
                let half_z = size.z * 0.5;
                Aabb::new(
                    Vec3::new(-half_x, 0.0, -half_z),
                    Vec3::new(half_x, size.y, half_z),
                )
            }
            ChamberShape::Sphere => {
                let r = (size.y * 0.5).max(0.2);
                Aabb::new(
                    Vec3::new(-r, 0.0, -r),
                    Vec3::new(r, 2.0 * r, r),
                )
            }
            ChamberShape::Bathtub => {
                let half_x = size.x * 0.5;
                let half_z = size.z * 0.5;
                Aabb::new(
                    Vec3::new(-half_x, 0.0, -half_z),
                    Vec3::new(half_x, size.y, half_z),
                )
            }
        }
    }

    pub fn gpu_info(&self, bounds: &Aabb) -> [f32; 4] {
        let h = (bounds.max.y - bounds.min.y).max(1e-4);
        let l = (bounds.max.x - bounds.min.x).max(1e-4);
        let w = (bounds.max.z - bounds.min.z).max(1e-4);
        match self.shape {
            ChamberShape::Box => [0.0, 0.0, 0.0, 0.0],
            ChamberShape::Sphere => {
                let r = h * 0.5;
                [1.0, r, 0.0, 0.0]
            }
            ChamberShape::Bathtub => {
                let slope_offset = (l * self.tub_slope_ratio).min(l * 0.45);
                let fillet = (w * self.tub_fillet_ratio).min(w * 0.4);
                [2.0, h, fillet, slope_offset]
            }
        }
    }

    pub fn resolve_collision(&self, pos: &mut Vec3, radius: f32, bounds: &Aabb) {
        match self.shape {
            ChamberShape::Box => {
                pos.x = pos.x.clamp(bounds.min.x + radius, bounds.max.x - radius);
                pos.y = pos.y.clamp(bounds.min.y + radius, bounds.max.y - radius);
                pos.z = pos.z.clamp(bounds.min.z + radius, bounds.max.z - radius);
            }
            ChamberShape::Sphere => {
                let r_chamber = (bounds.max.y - bounds.min.y) * 0.5;
                let center = Vec3::new(
                    (bounds.min.x + bounds.max.x) * 0.5,
                    bounds.min.y + r_chamber,
                    (bounds.min.z + bounds.max.z) * 0.5,
                );
                let to_p = *pos - center;
                let dist = to_p.length();
                let max_d = (r_chamber - radius).max(0.01);
                if dist > max_d && dist > 1e-6 {
                    *pos = center + (to_p / dist) * max_d;
                }
            }
            ChamberShape::Bathtub => {
                let b_min = bounds.min + Vec3::splat(radius);
                let b_max = bounds.max - Vec3::splat(radius);
                let h = (bounds.max.y - bounds.min.y).max(1e-4);
                let l = bounds.max.x - bounds.min.x;
                let w = bounds.max.z - bounds.min.z;
                let slope_offset = (l * self.tub_slope_ratio).min(l * 0.45);
                let fillet = (w * self.tub_fillet_ratio).min(w * 0.4);

                // 1. Y floor and ceiling
                pos.y = pos.y.clamp(b_min.y, b_max.y);

                // 2. Sloped backrest at -X and front wall at +X
                let norm_y = ((pos.y - bounds.min.y) / h).clamp(0.0, 1.0);
                let x_back = b_min.x + (1.0 - norm_y) * slope_offset;
                let x_front = b_max.x;
                pos.x = pos.x.clamp(x_back, x_front);

                // 3. Side walls at Z
                let half_w = w * 0.5;
                let cz = (bounds.min.z + bounds.max.z) * 0.5;
                let z_rel = pos.z - cz;
                let z_clamped = z_rel.clamp(-half_w + radius, half_w - radius);
                pos.z = cz + z_clamped;

                // 4. Fillet curves where floor meets side walls
                if fillet > radius {
                    let eff_fillet = fillet - radius;
                    let dy = (pos.y - bounds.min.y) - fillet;
                    let abs_z = (pos.z - cz).abs();
                    let dz = abs_z - (half_w - fillet);
                    if dy < 0.0 && dz > 0.0 {
                        let d = (dy * dy + dz * dz).sqrt();
                        if d > eff_fillet && d > 1e-6 {
                            let factor = eff_fillet / d;
                            pos.y = bounds.min.y + fillet + dy * factor;
                            let new_z = (half_w - fillet) + dz * factor;
                            pos.z = cz + (pos.z - cz).signum() * new_z;
                        }
                    }
                }
            }
        }
    }

    pub fn resolve_velocity(
        &self,
        new_pos: Vec3,
        _old_pos: Vec3,
        vel: &mut Vec3,
        _dt: f32,
        radius: f32,
        bounds: &Aabb,
        restitution: f32,
        friction: f32,
    ) {
        let margin = 0.002;
        match self.shape {
            ChamberShape::Box => {
                if new_pos.y <= bounds.min.y + radius + margin {
                    if vel.y < 0.0 {
                        vel.y = -vel.y * restitution;
                    } else if vel.y < 0.15 {
                        vel.y = 0.0;
                    }
                    vel.x *= friction;
                    vel.z *= friction;
                }
                if new_pos.y >= bounds.max.y - radius - margin && vel.y > 0.0 {
                    vel.y = -vel.y * restitution;
                    vel.x *= friction;
                    vel.z *= friction;
                }
                if new_pos.x <= bounds.min.x + radius + margin && vel.x < 0.0 {
                    vel.x = -vel.x * restitution;
                    vel.y *= friction;
                    vel.z *= friction;
                }
                if new_pos.x >= bounds.max.x - radius - margin && vel.x > 0.0 {
                    vel.x = -vel.x * restitution;
                    vel.y *= friction;
                    vel.z *= friction;
                }
                if new_pos.z <= bounds.min.z + radius + margin && vel.z < 0.0 {
                    vel.z = -vel.z * restitution;
                    vel.x *= friction;
                    vel.y *= friction;
                }
                if new_pos.z >= bounds.max.z - radius - margin && vel.z > 0.0 {
                    vel.z = -vel.z * restitution;
                    vel.x *= friction;
                    vel.y *= friction;
                }
            }
            ChamberShape::Sphere => {
                let r_chamber = (bounds.max.y - bounds.min.y) * 0.5;
                let center = Vec3::new(
                    (bounds.min.x + bounds.max.x) * 0.5,
                    bounds.min.y + r_chamber,
                    (bounds.min.z + bounds.max.z) * 0.5,
                );
                let to_p = new_pos - center;
                let dist = to_p.length();
                if dist >= r_chamber - radius - margin && dist > 1e-6 {
                    let normal = to_p / dist;
                    let vn = vel.dot(normal);
                    if vn > 0.0 {
                        let normal_vel = normal * vn;
                        let tangent_vel = *vel - normal_vel;
                        *vel = tangent_vel * friction - normal_vel * restitution;
                    }
                }
            }
            ChamberShape::Bathtub => {
                let h = (bounds.max.y - bounds.min.y).max(1e-4);
                let l = bounds.max.x - bounds.min.x;
                let slope_offset = (l * self.tub_slope_ratio).min(l * 0.45);
                let norm_y = ((new_pos.y - bounds.min.y) / h).clamp(0.0, 1.0);
                let x_back = bounds.min.x + (1.0 - norm_y) * slope_offset;

                let back_dist = (new_pos.x - x_back).abs();
                if back_dist <= radius + margin {
                    let back_normal = Vec3::new(h, slope_offset, 0.0).normalize();
                    let vn = vel.dot(-back_normal);
                    if vn < 0.0 {
                        let n_v = -back_normal * vn;
                        let t_v = *vel - n_v;
                        *vel = t_v * friction - n_v * restitution;
                    }
                }

                if new_pos.y <= bounds.min.y + radius + margin {
                    if vel.y < 0.0 {
                        vel.y = -vel.y * restitution;
                    } else if vel.y < 0.15 {
                        vel.y = 0.0;
                    }
                    vel.x *= friction;
                    vel.z *= friction;
                }
                if new_pos.y >= bounds.max.y - radius - margin && vel.y > 0.0 {
                    vel.y = -vel.y * restitution;
                    vel.x *= friction;
                    vel.z *= friction;
                }
                if new_pos.x >= bounds.max.x - radius - margin && vel.x > 0.0 {
                    vel.x = -vel.x * restitution;
                    vel.y *= friction;
                    vel.z *= friction;
                }
                if new_pos.z <= bounds.min.z + radius + margin && vel.z < 0.0 {
                    vel.z = -vel.z * restitution;
                    vel.x *= friction;
                    vel.z *= friction;
                }
                if new_pos.z >= bounds.max.z - radius - margin && vel.z > 0.0 {
                    vel.z = -vel.z * restitution;
                    vel.x *= friction;
                    vel.y *= friction;
                }
            }
        }
    }

    pub fn generate_wireframe_lines_2d(&self, bounds: &Aabb) -> Vec<(Vec3, Vec3)> {
        let mut lines = Vec::new();
        match self.shape {
            ChamberShape::Box => {
                let p0 = Vec3::new(bounds.min.x, bounds.min.y, 0.0);
                let p1 = Vec3::new(bounds.max.x, bounds.min.y, 0.0);
                let p2 = Vec3::new(bounds.max.x, bounds.max.y, 0.0);
                let p3 = Vec3::new(bounds.min.x, bounds.max.y, 0.0);
                lines.push((p0, p1));
                lines.push((p1, p2));
                lines.push((p2, p3));
                lines.push((p3, p0));
            }
            ChamberShape::Sphere => {
                let r = (bounds.max.y - bounds.min.y) * 0.5;
                let c = Vec3::new(
                    (bounds.min.x + bounds.max.x) * 0.5,
                    bounds.min.y + r,
                    0.0,
                );
                let segments = 64;
                let step = (2.0 * PI) / segments as f32;
                for i in 0..segments {
                    let a0 = i as f32 * step;
                    let a1 = (i + 1) as f32 * step;
                    let p0 = c + Vec3::new(a0.cos() * r, a0.sin() * r, 0.0);
                    let p1 = c + Vec3::new(a1.cos() * r, a1.sin() * r, 0.0);
                    lines.push((p0, p1));
                }
            }
            ChamberShape::Bathtub => {
                let h = bounds.max.y - bounds.min.y;
                let l = bounds.max.x - bounds.min.x;
                let y_top = bounds.max.y;
                let y_bot = bounds.min.y;
                let slope_offset = (l * self.tub_slope_ratio).min(l * 0.45);
                let x_back_bot = bounds.min.x + slope_offset;
                let bot_corner_r = (h * self.tub_fillet_ratio).min(h * 0.35);

                // Top rim
                lines.push((Vec3::new(bounds.min.x, y_top, 0.0), Vec3::new(bounds.max.x, y_top, 0.0)));
                // Front vertical wall
                lines.push((Vec3::new(bounds.max.x, y_top, 0.0), Vec3::new(bounds.max.x, y_bot + bot_corner_r, 0.0)));
                // Front bottom fillet curve
                let segs = 12;
                for i in 0..segs {
                    let a0 = (i as f32 / segs as f32) * (PI * 0.5);
                    let a1 = ((i + 1) as f32 / segs as f32) * (PI * 0.5);
                    let p0 = Vec3::new(bounds.max.x - bot_corner_r + a0.sin() * bot_corner_r, y_bot + bot_corner_r - a0.cos() * bot_corner_r, 0.0);
                    let p1 = Vec3::new(bounds.max.x - bot_corner_r + a1.sin() * bot_corner_r, y_bot + bot_corner_r - a1.cos() * bot_corner_r, 0.0);
                    lines.push((p0, p1));
                }
                // Flat bottom floor
                lines.push((Vec3::new(x_back_bot, y_bot, 0.0), Vec3::new(bounds.max.x - bot_corner_r, y_bot, 0.0)));
                // Sloped back rest
                lines.push((Vec3::new(bounds.min.x, y_top, 0.0), Vec3::new(x_back_bot, y_bot, 0.0)));
            }
        }
        lines
    }

    pub fn generate_wireframe_lines(&self, bounds: &Aabb) -> Vec<(Vec3, Vec3)> {
        let mut lines = Vec::new();

        match self.shape {
            ChamberShape::Box => {
                let corners = bounds.corners();
                for (i0, i1) in Aabb::edge_indices() {
                    lines.push((corners[i0], corners[i1]));
                }
            }
            ChamberShape::Sphere => {
                let r = (bounds.max.y - bounds.min.y) * 0.5;
                let c = Vec3::new(
                    (bounds.min.x + bounds.max.x) * 0.5,
                    bounds.min.y + r,
                    (bounds.min.z + bounds.max.z) * 0.5,
                );

                let segments = 40;
                let step = (2.0 * PI) / segments as f32;

                // Equator circle (XZ)
                for i in 0..segments {
                    let a0 = i as f32 * step;
                    let a1 = (i + 1) as f32 * step;
                    let p0 = c + Vec3::new(a0.cos() * r, 0.0, a0.sin() * r);
                    let p1 = c + Vec3::new(a1.cos() * r, 0.0, a1.sin() * r);
                    lines.push((p0, p1));
                }

                // Great circle (XY)
                for i in 0..segments {
                    let a0 = i as f32 * step;
                    let a1 = (i + 1) as f32 * step;
                    let p0 = c + Vec3::new(a0.cos() * r, a0.sin() * r, 0.0);
                    let p1 = c + Vec3::new(a1.cos() * r, a1.sin() * r, 0.0);
                    lines.push((p0, p1));
                }

                // Great circle (YZ)
                for i in 0..segments {
                    let a0 = i as f32 * step;
                    let a1 = (i + 1) as f32 * step;
                    let p0 = c + Vec3::new(0.0, a0.cos() * r, a0.sin() * r);
                    let p1 = c + Vec3::new(0.0, a1.cos() * r, a1.sin() * r);
                    lines.push((p0, p1));
                }

                // Latitude rings at y = +-0.5 * r
                let lat_r = r * 0.866; // cos(30 deg)
                for &dy in &[0.5 * r, -0.5 * r] {
                    for i in 0..segments {
                        let a0 = i as f32 * step;
                        let a1 = (i + 1) as f32 * step;
                        let p0 = c + Vec3::new(a0.cos() * lat_r, dy, a0.sin() * lat_r);
                        let p1 = c + Vec3::new(a1.cos() * lat_r, dy, a1.sin() * lat_r);
                        lines.push((p0, p1));
                    }
                }
            }
            ChamberShape::Bathtub => {
                let _h = bounds.max.y - bounds.min.y;
                let l = bounds.max.x - bounds.min.x;
                let w = bounds.max.z - bounds.min.z;
                let slope_offset = (l * self.tub_slope_ratio).min(l * 0.45);
                let fillet = (w * self.tub_fillet_ratio).min(w * 0.4);

                let cz = (bounds.min.z + bounds.max.z) * 0.5;
                let half_w = w * 0.5;
                let y_top = bounds.max.y;
                let y_bot = bounds.min.y;

                // 1. Top rim perimeter
                let rim_p0 = Vec3::new(bounds.min.x, y_top, cz - half_w);
                let rim_p1 = Vec3::new(bounds.max.x, y_top, cz - half_w);
                let rim_p2 = Vec3::new(bounds.max.x, y_top, cz + half_w);
                let rim_p3 = Vec3::new(bounds.min.x, y_top, cz + half_w);
                lines.push((rim_p0, rim_p1));
                lines.push((rim_p1, rim_p2));
                lines.push((rim_p2, rim_p3));
                lines.push((rim_p3, rim_p0));

                // 2. Outer bathtub lip (slightly flared)
                let lip_flare = 0.035;
                let y_lip = y_top + 0.015;
                let lip_p0 = Vec3::new(bounds.min.x - lip_flare, y_lip, cz - half_w - lip_flare);
                let lip_p1 = Vec3::new(bounds.max.x + lip_flare, y_lip, cz - half_w - lip_flare);
                let lip_p2 = Vec3::new(bounds.max.x + lip_flare, y_lip, cz + half_w + lip_flare);
                let lip_p3 = Vec3::new(bounds.min.x - lip_flare, y_lip, cz + half_w + lip_flare);
                lines.push((lip_p0, lip_p1));
                lines.push((lip_p1, lip_p2));
                lines.push((lip_p2, lip_p3));
                lines.push((lip_p3, lip_p0));

                // Connect rim to lip corners
                lines.push((rim_p0, lip_p0));
                lines.push((rim_p1, lip_p1));
                lines.push((rim_p2, lip_p2));
                lines.push((rim_p3, lip_p3));

                // 3. Bottom floor perimeter (taking into account slope at backrest)
                let x_back_bot = bounds.min.x + slope_offset;
                let floor_w = half_w - fillet * 0.4;
                let bot_p0 = Vec3::new(x_back_bot, y_bot, cz - floor_w);
                let bot_p1 = Vec3::new(bounds.max.x, y_bot, cz - floor_w);
                let bot_p2 = Vec3::new(bounds.max.x, y_bot, cz + floor_w);
                let bot_p3 = Vec3::new(x_back_bot, y_bot, cz + floor_w);
                lines.push((bot_p0, bot_p1));
                lines.push((bot_p1, bot_p2));
                lines.push((bot_p2, bot_p3));
                lines.push((bot_p3, bot_p0));

                // 4. Sloped backrest ribs
                lines.push((rim_p0, bot_p0));
                lines.push((rim_p3, bot_p3));
                let rim_mid_back = Vec3::new(bounds.min.x, y_top, cz);
                let bot_mid_back = Vec3::new(x_back_bot, y_bot, cz);
                lines.push((rim_mid_back, bot_mid_back));

                // 5. Front uprights
                lines.push((rim_p1, bot_p1));
                lines.push((rim_p2, bot_p2));

                // 6. Fillet / cross ribs at 1/3 and 2/3 along the tub length
                for &frac in &[0.35, 0.70] {
                    let rx = x_back_bot + (bounds.max.x - x_back_bot) * frac;
                    let p_left_top = Vec3::new(rx, y_top, cz - half_w);
                    let p_left_bot = Vec3::new(rx, y_bot, cz - floor_w);
                    let p_right_top = Vec3::new(rx, y_top, cz + half_w);
                    let p_right_bot = Vec3::new(rx, y_bot, cz + floor_w);
                    lines.push((p_left_top, p_left_bot));
                    lines.push((p_right_top, p_right_bot));
                    lines.push((p_left_bot, p_right_bot));
                }

                // 7. Small drain ring on the floor
                let drain_x = bounds.max.x - 0.25 * l;
                let drain_c = Vec3::new(drain_x, y_bot + 0.002, cz);
                let drain_r = 0.06 * w;
                let segs = 16;
                let d_step = (2.0 * PI) / segs as f32;
                for i in 0..segs {
                    let a0 = i as f32 * d_step;
                    let a1 = (i + 1) as f32 * d_step;
                    let p0 = drain_c + Vec3::new(a0.cos() * drain_r, 0.0, a0.sin() * drain_r);
                    let p1 = drain_c + Vec3::new(a1.cos() * drain_r, 0.0, a1.sin() * drain_r);
                    lines.push((p0, p1));
                }
            }
        }

        lines
    }
}
