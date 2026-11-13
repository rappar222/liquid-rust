use crate::math::Aabb;
use crate::sph::mesh::TriangleMesh;
use glam::Vec3;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum ColliderShape {
    Sphere { center: Vec3, radius: f32 },
    Box { aabb: Aabb },
    Cylinder { center: Vec3, radius: f32, height: f32 },
    Mesh(Arc<TriangleMesh>),
}

#[derive(Clone, Debug)]
pub struct Collider {
    pub id: u32,
    pub name: String,
    pub shape: ColliderShape,
    pub velocity: Vec3,
    pub restitution: f32,
    pub friction: f32,
    pub enabled: bool,
}

impl Collider {
    pub fn sphere(center: Vec3, radius: f32) -> Self {
        Self {
            id: 0,
            name: "Sphere Obstacle".to_string(),
            shape: ColliderShape::Sphere { center, radius },
            velocity: Vec3::ZERO,
            restitution: 0.02,
            friction: 0.95,
            enabled: true,
        }
    }

    pub fn box_obstacle(min: Vec3, max: Vec3) -> Self {
        Self {
            id: 0,
            name: "Box Obstacle".to_string(),
            shape: ColliderShape::Box {
                aabb: Aabb::new(min, max),
            },
            velocity: Vec3::ZERO,
            restitution: 0.02,
            friction: 0.95,
            enabled: true,
        }
    }

    pub fn cylinder(center: Vec3, radius: f32, height: f32) -> Self {
        Self {
            id: 0,
            name: "Cylinder Obstacle".to_string(),
            shape: ColliderShape::Cylinder { center, radius, height },
            velocity: Vec3::ZERO,
            restitution: 0.02,
            friction: 0.95,
            enabled: true,
        }
    }

    pub fn mesh(mesh: Arc<TriangleMesh>) -> Self {
        let name = format!("Model ({})", mesh.name);
        Self {
            id: 0,
            name,
            shape: ColliderShape::Mesh(mesh),
            velocity: Vec3::ZERO,
            restitution: 0.02,
            friction: 0.95,
            enabled: true,
        }
    }

    pub fn load_obj(path: impl AsRef<Path>, position: Vec3, scale: f32) -> Result<Self, String> {
        Self::load_obj_transformed(path, position, scale, Vec3::ZERO)
    }

    pub fn load_obj_transformed(
        path: impl AsRef<Path>,
        position: Vec3,
        scale: f32,
        rotation: Vec3,
    ) -> Result<Self, String> {
        let mut mesh = TriangleMesh::load_obj_normalized(path)?;
        mesh.set_transform(position, scale, rotation);
        Ok(Self::mesh(Arc::new(mesh)))
    }

    /// Resolve collision with particle at `pos` with radius `r` and velocity `vel`.
    pub fn resolve(&self, pos: &mut Vec3, vel: &mut Vec3, particle_radius: f32) -> bool {
        if !self.enabled {
            return false;
        }

        match &self.shape {
            ColliderShape::Sphere { center, radius } => {
                let diff = *pos - *center;
                let dist_sq = diff.length_squared();
                let effective_r = radius + particle_radius;

                if dist_sq < effective_r * effective_r && dist_sq > 1e-10 {
                    let dist = dist_sq.sqrt();
                    let normal = diff / dist;
                    let penetration = effective_r - dist;

                    *pos += normal * penetration;

                    let rel_vel = *vel - self.velocity;
                    let vn = rel_vel.dot(normal);
                    if vn < 0.0 {
                        let eff_restitution = if vn < -0.10 { self.restitution } else { 0.0 };
                        let normal_impulse = -(1.0 + eff_restitution) * vn * normal;
                        let mut new_rel = rel_vel + normal_impulse;
                        let vt = new_rel - normal * new_rel.dot(normal);
                        new_rel -= vt * ((1.0 - self.friction) * 0.1);
                        *vel = self.velocity + new_rel;
                    }
                    return true;
                }
            }
            ColliderShape::Box { aabb } => {
                let inflated_min = aabb.min - Vec3::splat(particle_radius);
                let inflated_max = aabb.max + Vec3::splat(particle_radius);

                if pos.x >= inflated_min.x
                    && pos.x <= inflated_max.x
                    && pos.y >= inflated_min.y
                    && pos.y <= inflated_max.y
                    && pos.z >= inflated_min.z
                    && pos.z <= inflated_max.z
                {
                    let d_left = (pos.x - inflated_min.x).abs();
                    let d_right = (inflated_max.x - pos.x).abs();
                    let d_bottom = (pos.y - inflated_min.y).abs();
                    let d_top = (inflated_max.y - pos.y).abs();
                    let d_back = (pos.z - inflated_min.z).abs();
                    let d_front = (inflated_max.z - pos.z).abs();

                    let min_dist = d_left
                        .min(d_right)
                        .min(d_bottom)
                        .min(d_top)
                        .min(d_back)
                        .min(d_front);

                    let normal = if min_dist == d_left {
                        pos.x = inflated_min.x;
                        Vec3::new(-1.0, 0.0, 0.0)
                    } else if min_dist == d_right {
                        pos.x = inflated_max.x;
                        Vec3::new(1.0, 0.0, 0.0)
                    } else if min_dist == d_bottom {
                        pos.y = inflated_min.y;
                        Vec3::new(0.0, -1.0, 0.0)
                    } else if min_dist == d_top {
                        pos.y = inflated_max.y;
                        Vec3::new(0.0, 1.0, 0.0)
                    } else if min_dist == d_back {
                        pos.z = inflated_min.z;
                        Vec3::new(0.0, 0.0, -1.0)
                    } else {
                        pos.z = inflated_max.z;
                        Vec3::new(0.0, 0.0, 1.0)
                    };

                    let rel_vel = *vel - self.velocity;
                    let vn = rel_vel.dot(normal);
                    if vn < 0.0 {
                        let eff_restitution = if vn < -0.10 { self.restitution } else { 0.0 };
                        let normal_impulse = -(1.0 + eff_restitution) * vn * normal;
                        let mut new_rel = rel_vel + normal_impulse;
                        let vt = new_rel - normal * new_rel.dot(normal);
                        new_rel -= vt * ((1.0 - self.friction) * 0.1);
                        *vel = self.velocity + new_rel;
                    }
                    return true;
                }
            }
            ColliderShape::Cylinder { center, radius, height } => {
                let dx = pos.x - center.x;
                let dz = pos.z - center.z;
                let r2 = dx * dx + dz * dz;
                let effective_r = radius + particle_radius;
                let half_h = height * 0.5;
                let y_min = center.y - half_h - particle_radius;
                let y_max = center.y + half_h + particle_radius;

                if r2 < effective_r * effective_r && pos.y >= y_min && pos.y <= y_max {
                    let r = r2.sqrt();
                    let normal = if r > 1e-6 {
                        Vec3::new(dx / r, 0.0, dz / r)
                    } else {
                        Vec3::X
                    };
                    pos.x = center.x + normal.x * effective_r;
                    pos.z = center.z + normal.z * effective_r;

                    let rel_vel = *vel - self.velocity;
                    let vn = rel_vel.dot(normal);
                    if vn < 0.0 {
                        let eff_restitution = if vn < -0.10 { self.restitution } else { 0.0 };
                        let normal_impulse = -(1.0 + eff_restitution) * vn * normal;
                        let mut new_rel = rel_vel + normal_impulse;
                        let vt = new_rel - normal * new_rel.dot(normal);
                        new_rel -= vt * ((1.0 - self.friction) * 0.1);
                        *vel = self.velocity + new_rel;
                    }
                    return true;
                }
            }
            ColliderShape::Mesh(mesh) => {
                return mesh.resolve_collision(pos, vel, particle_radius, self.restitution, self.friction);
            }
        }
        false
    }

    /// Fold analytical boundary volume fraction & gradient from this solid obstacle into `(psi, grad)`.
    #[inline]
    pub fn fold_boundary_contribution(
        &self,
        pos: Vec3,
        particle_radius: f32,
        spacing: f32,
        h: f32,
        is_2d: bool,
        psi: &mut f32,
        grad: &mut Vec3,
    ) {
        if !self.enabled {
            return;
        }
        let s = spacing.max(1e-4);
        let inv_h = 1.0 / h.max(1e-5);
        let gamma_w = (if is_2d { 1.42 } else { 1.20 }) * (s * inv_h);
        let d_offset = s;

        let mut apply = |d_clearance: f32, n_outward: Vec3| {
            crate::sph::Chamber::fold_wall_plane(
                psi,
                grad,
                d_clearance,
                n_outward,
                d_offset,
                h,
                inv_h,
                gamma_w,
            );
        };

        match &self.shape {
            ColliderShape::Sphere { center, radius } => {
                let diff = pos - *center;
                let dist = diff.length();
                if dist > 1e-6 {
                    let d_c = dist - (*radius + particle_radius);
                    if d_c < h {
                        apply(d_c, diff / dist);
                    }
                }
            }
            ColliderShape::Box { aabb } => {
                let inf_min = aabb.min - Vec3::splat(particle_radius);
                let inf_max = aabb.max + Vec3::splat(particle_radius);
                let clamped = pos.clamp(inf_min, inf_max);
                let diff = pos - clamped;
                let dist = diff.length();
                if dist > 1e-6 && dist < h {
                    apply(dist, diff / dist);
                }
            }
            ColliderShape::Cylinder { center, radius, height } => {
                let half_h = *height * 0.5 + particle_radius;
                let eff_r = *radius + particle_radius;
                if pos.y >= center.y - half_h - h && pos.y <= center.y + half_h + h {
                    let dx = pos.x - center.x;
                    let dz = pos.z - center.z;
                    let r_xz = (dx * dx + dz * dz).sqrt();
                    if r_xz > 1e-6 {
                        let dr = (r_xz - eff_r).max(0.0);
                        let dy = ((pos.y - center.y).abs() - half_h).max(0.0);
                        let dist = (dr * dr + dy * dy).sqrt();
                        if dist < h {
                            let mut n = Vec3::new(dx / r_xz, 0.0, dz / r_xz);
                            if dy > dr && dy > 1e-5 {
                                n = Vec3::new(0.0, (pos.y - center.y).signum(), 0.0);
                            }
                            apply(dist, n);
                        }
                    }
                }
            }
            ColliderShape::Mesh(_) => {}
        }
    }
}
