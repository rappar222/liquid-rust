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
    /// True when base_size was explicitly overridden (e.g. via CLI --chamber-size / --box-size)
    pub custom_size: bool,
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
            custom_size: false,
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
        dt: f32,
        radius: f32,
        bounds: &Aabb,
        restitution: f32,
        friction: f32,
    ) {
        let margin = (radius * 0.20).max(0.002);
        let slip_floor = (1.0 - (1.0 - friction).max(0.0) * dt * 1.5).clamp(0.985, 1.0);
        let slip_wall = 1.0f32; // True free-slip smooth container walls
        let rest_threshold = 0.12; // m/s: below this normal speed, restitution is zero (inelastic resting)
        match self.shape {
            ChamberShape::Box => {
                if new_pos.y <= bounds.min.y + radius + margin {
                    if vel.y < 0.0 {
                        if vel.y < -rest_threshold {
                            vel.y = -vel.y * restitution;
                        } else {
                            vel.y = 0.0;
                        }
                    }
                    vel.x *= slip_floor;
                    vel.z *= slip_floor;
                }
                if new_pos.y >= bounds.max.y - radius - margin && vel.y > 0.0 {
                    if vel.y > rest_threshold {
                        vel.y = -vel.y * restitution;
                    } else {
                        vel.y = 0.0;
                    }
                    vel.x *= slip_wall;
                    vel.z *= slip_wall;
                }
                if new_pos.x <= bounds.min.x + radius + margin && vel.x < 0.0 {
                    if vel.x < -rest_threshold {
                        vel.x = -vel.x * restitution;
                    } else {
                        vel.x = 0.0;
                    }
                    vel.y *= slip_wall;
                    vel.z *= slip_wall;
                }
                if new_pos.x >= bounds.max.x - radius - margin && vel.x > 0.0 {
                    if vel.x > rest_threshold {
                        vel.x = -vel.x * restitution;
                    } else {
                        vel.x = 0.0;
                    }
                    vel.y *= slip_wall;
                    vel.z *= slip_wall;
                }
                if new_pos.z <= bounds.min.z + radius + margin && vel.z < 0.0 {
                    if vel.z < -rest_threshold {
                        vel.z = -vel.z * restitution;
                    } else {
                        vel.z = 0.0;
                    }
                    vel.x *= slip_wall;
                    vel.y *= slip_wall;
                }
                if new_pos.z >= bounds.max.z - radius - margin && vel.z > 0.0 {
                    if vel.z > rest_threshold {
                        vel.z = -vel.z * restitution;
                    } else {
                        vel.z = 0.0;
                    }
                    vel.x *= slip_wall;
                    vel.y *= slip_wall;
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
                        let eff_rest = if vn > rest_threshold { restitution } else { 0.0 };
                        *vel = tangent_vel * slip_floor - normal_vel * eff_rest;
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
                        let eff_rest = if vn < -rest_threshold { restitution } else { 0.0 };
                        *vel = t_v * slip_wall - n_v * eff_rest;
                    }
                }

                if new_pos.y <= bounds.min.y + radius + margin {
                    if vel.y < 0.0 {
                        vel.y = if vel.y < -rest_threshold { -vel.y * restitution } else { 0.0 };
                    }
                    vel.x *= slip_floor;
                    vel.z *= slip_floor;
                }
                if new_pos.y >= bounds.max.y - radius - margin && vel.y > 0.0 {
                    vel.y = if vel.y > rest_threshold { -vel.y * restitution } else { 0.0 };
                    vel.x *= slip_wall;
                    vel.z *= slip_wall;
                }
                if new_pos.x >= bounds.max.x - radius - margin && vel.x > 0.0 {
                    vel.x = if vel.x > rest_threshold { -vel.x * restitution } else { 0.0 };
                    vel.y *= slip_wall;
                    vel.z *= slip_wall;
                }
                if new_pos.z <= bounds.min.z + radius + margin && vel.z < 0.0 {
                    vel.z = if vel.z < -rest_threshold { -vel.z * restitution } else { 0.0 };
                    vel.x *= slip_wall;
                    vel.y *= slip_wall;
                }
                if new_pos.z >= bounds.max.z - radius - margin && vel.z > 0.0 {
                    vel.z = if vel.z > rest_threshold { -vel.z * restitution } else { 0.0 };
                    vel.x *= slip_wall;
                    vel.y *= slip_wall;
                }
            }
        }
    }

    /// Evaluate static, zero-mean, curl-free spatial micro-texture `xi` and its dimensionless
    /// normalized gradient `grad_xi = s * grad(xi)` to frustrate epitaxial wall crystallization
    /// strictly conservatively without injecting kinetic energy or wall convection.
    #[inline]
    pub(crate) fn wall_micro_texture(pos: Vec3, spacing: f32, is_2d: bool) -> (f32, Vec3) {
        let inv_s = 1.0 / spacing.max(1e-4);
        let px = pos.x * inv_s;
        let py = pos.y * inv_s;
        let pz = pos.z * inv_s;

        let a1 = 4.71 * px + 2.31 * pz;
        let a2 = 3.83 * py - 4.19 * px;
        let a3 = if is_2d {
            3.13 * px + 4.47 * py
        } else {
            5.17 * pz - 3.41 * py
        };

        let xi = (a1.sin() + a2.sin() + a3.sin()) * (1.0 / 3.0);
        let c1 = a1.cos();
        let c2 = a2.cos();
        let c3 = a3.cos();
        let grad_xi = if is_2d {
            Vec3::new(
                (4.71 * c1 - 4.19 * c2 + 3.13 * c3) * (1.0 / 3.0),
                (3.83 * c2 + 4.47 * c3) * (1.0 / 3.0),
                0.0,
            )
        } else {
            Vec3::new(
                (4.71 * c1 - 4.19 * c2) * (1.0 / 3.0),
                (3.83 * c2 - 3.41 * c3) * (1.0 / 3.0),
                (2.31 * c1 + 5.17 * c3) * (1.0 / 3.0),
            )
        };
        (xi, grad_xi)
    }

    /// Fold one analytical wall boundary plane into `(psi, grad)` using inclusion-exclusion.
    #[inline]
    pub(crate) fn fold_wall_plane(
        psi: &mut f32,
        grad: &mut Vec3,
        d_clearance: f32,
        grad_d: Vec3,
        d_offset: f32,
        h: f32,
        inv_h: f32,
        gamma_w: f32,
    ) {
        let d_eff = (d_clearance.max(0.0) + d_offset).max(0.25 * h);
        if d_eff < h {
            let q = d_eff * inv_h;
            let omq2 = 1.0 - q * q;
            let omq4 = (omq2 * omq2) * (omq2 * omq2);
            let psi_w = (gamma_w * omq4).min(0.65);

            let omq = 1.0 - q;
            let grad_mag = gamma_w * (6.1 * inv_h) * (omq * omq);
            let g_w = -grad_d * grad_mag;

            *grad = *grad * (1.0 - psi_w) + g_w * (1.0 - *psi);
            *psi = *psi + psi_w - *psi * psi_w;
        }
    }

    /// Smooth wall-neutrality weight in `[0.0, 1.0]` for symmetric free-surface cohesion:
    /// returns `0.0` adjacent to vertical/ceiling container walls (90° neutral contact angle,
    /// zero capillary meniscus climbing or asymmetric wall suction) and `1.0` in the interior.
    #[inline]
    pub fn wall_cohesion_weight(
        &self,
        pos: Vec3,
        radius: f32,
        h: f32,
        bounds: &Aabb,
        is_2d: bool,
    ) -> f32 {
        let h_eff = h.max(1e-5);
        let inv_h = 1.0 / h_eff;
        let mut d_floor = (pos.y - bounds.min.y) - radius;
        let mut d_wall = 1e6f32;

        match self.shape {
            ChamberShape::Sphere => {
                let r_chamber = (bounds.max.y - bounds.min.y) * 0.5;
                let center = Vec3::new(
                    (bounds.min.x + bounds.max.x) * 0.5,
                    bounds.min.y + r_chamber,
                    (bounds.min.z + bounds.max.z) * 0.5,
                );
                let to_p = pos - center;
                let dist = to_p.length();
                let n_in_y = if dist > 1e-6 { -to_p.y / dist } else { 1.0 };
                let d_sph = (r_chamber - radius) - dist;
                if n_in_y >= 0.70 {
                    d_floor = d_sph;
                } else {
                    d_wall = d_sph;
                }
            }
            ChamberShape::Bathtub => {
                let h_tub = (bounds.max.y - bounds.min.y).max(1e-4);
                let l_tub = bounds.max.x - bounds.min.x;
                let slope_offset = (l_tub * self.tub_slope_ratio).min(l_tub * 0.45);
                let back_n = Vec3::new(h_tub, slope_offset, 0.0).normalize();
                let p_back_bot = Vec3::new(bounds.min.x + slope_offset, bounds.min.y, 0.0);
                let d_back = (pos - p_back_bot).dot(back_n) - radius;
                d_wall = d_back.min((bounds.max.x - pos.x).min(bounds.max.y - pos.y) - radius);
                if !is_2d {
                    d_wall = d_wall.min((pos.z - bounds.min.z).min(bounds.max.z - pos.z) - radius);
                }
            }
            ChamberShape::Box => {
                d_wall = (pos.x - bounds.min.x)
                    .min(bounds.max.x - pos.x)
                    .min(bounds.max.y - pos.y)
                    - radius;
                if !is_2d {
                    d_wall = d_wall.min((pos.z - bounds.min.z).min(bounds.max.z - pos.z) - radius);
                }
            }
        }

        let t_floor = (d_floor * inv_h).clamp(0.0, 1.0);
        let s_floor = t_floor * t_floor * (3.0 - 2.0 * t_floor);
        let w_floor = 0.25 + 0.75 * s_floor;

        let t_side = ((d_wall - 0.55 * h_eff) / (1.20 * h_eff)).clamp(0.0, 1.0);
        let w_side = t_side * t_side * (3.0 - 2.0 * t_side);

        w_floor * w_side
    }

    /// Analytical Akinci boundary volume fraction `psi_b` (dimensionless fraction of `rho0`)
    /// and its spatial gradient `grad_psi_b` (pointing into the solid boundary, matching
    /// `spiky_gradient * inv_rho0` of virtual boundary particles).
    /// Eliminates SPH boundary density deficiency, edge/floor caterpillar packing,
    /// wall meniscus sticking, and planar hexagonal crystallization.
    #[inline]
    pub fn boundary_contribution(
        &self,
        pos: Vec3,
        radius: f32,
        spacing: f32,
        h: f32,
        bounds: &Aabb,
        is_2d: bool,
    ) -> (f32, Vec3) {
        let s = spacing.max(1e-4);
        let inv_h = 1.0 / h.max(1e-5);
        let (xi, grad_xi) = Self::wall_micro_texture(pos, s, is_2d);
        let rough_amp = if is_2d { 0.025 } else { 0.055 };
        let gamma_w = (if is_2d { 1.42 } else { 1.20 }) * (s * inv_h);

        let mut psi = 0.0f32;
        let mut grad = Vec3::ZERO;

        let mut fold_wall = |d_clearance: f32, n_inward: Vec3| {
            let floor_w = n_inward.y.clamp(0.0, 1.0);
            let eff_amp = rough_amp * floor_w;
            let d_offset = (1.0 - eff_amp * xi) * s;
            let grad_bump = -grad_xi * eff_amp;
            Self::fold_wall_plane(
                &mut psi,
                &mut grad,
                d_clearance,
                n_inward + grad_bump,
                d_offset,
                h,
                inv_h,
                gamma_w,
            );
        };

        match self.shape {
            ChamberShape::Box => {
                fold_wall((pos.y - bounds.min.y) - radius, Vec3::Y);
                fold_wall((bounds.max.y - pos.y) - radius, Vec3::NEG_Y);
                fold_wall((pos.x - bounds.min.x) - radius, Vec3::X);
                fold_wall((bounds.max.x - pos.x) - radius, Vec3::NEG_X);
                if !is_2d {
                    fold_wall((pos.z - bounds.min.z) - radius, Vec3::Z);
                    fold_wall((bounds.max.z - pos.z) - radius, Vec3::NEG_Z);
                }
            }
            ChamberShape::Sphere => {
                let r_chamber = (bounds.max.y - bounds.min.y) * 0.5;
                let center = Vec3::new(
                    (bounds.min.x + bounds.max.x) * 0.5,
                    bounds.min.y + r_chamber,
                    (bounds.min.z + bounds.max.z) * 0.5,
                );
                let to_p = pos - center;
                let dist = to_p.length();
                let n_inward = if dist > 1e-6 { -to_p / dist } else { Vec3::Y };
                let d_clearance = (r_chamber - radius) - dist;
                fold_wall(d_clearance, n_inward);
            }
            ChamberShape::Bathtub => {
                let h_tub = (bounds.max.y - bounds.min.y).max(1e-4);
                let l_tub = bounds.max.x - bounds.min.x;
                let slope_offset = (l_tub * self.tub_slope_ratio).min(l_tub * 0.45);
                let back_n = Vec3::new(h_tub, slope_offset, 0.0).normalize();
                let p_back_bot = Vec3::new(bounds.min.x + slope_offset, bounds.min.y, 0.0);
                let d_back = (pos - p_back_bot).dot(back_n) - radius;

                fold_wall((pos.y - bounds.min.y) - radius, Vec3::Y);
                fold_wall((bounds.max.y - pos.y) - radius, Vec3::NEG_Y);
                fold_wall(d_back, back_n);
                fold_wall((bounds.max.x - pos.x) - radius, Vec3::NEG_X);
                if !is_2d {
                    fold_wall((pos.z - bounds.min.z) - radius, Vec3::Z);
                    fold_wall((bounds.max.z - pos.z) - radius, Vec3::NEG_Z);
                }
            }
        }

        (psi, grad)
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
