use crate::math::Aabb;
use glam::Vec3;

#[derive(Clone, Debug)]
pub enum ColliderShape {
    Sphere { center: Vec3, radius: f32 },
    Box { aabb: Aabb },
    Cylinder { center: Vec3, radius: f32, height: f32 },
}

#[derive(Clone, Debug)]
pub struct Collider {
    pub shape: ColliderShape,
    pub velocity: Vec3,
    pub restitution: f32,
    pub friction: f32,
    pub enabled: bool,
}

impl Collider {
    pub fn sphere(center: Vec3, radius: f32) -> Self {
        Self {
            shape: ColliderShape::Sphere { center, radius },
            velocity: Vec3::ZERO,
            restitution: 0.02,
            friction: 0.95,
            enabled: true,
        }
    }

    pub fn box_obstacle(min: Vec3, max: Vec3) -> Self {
        Self {
            shape: ColliderShape::Box {
                aabb: Aabb::new(min, max),
            },
            velocity: Vec3::ZERO,
            restitution: 0.02,
            friction: 0.95,
            enabled: true,
        }
    }

    /// Resolve collision with particle at `pos` with radius `r` and velocity `vel`.
    /// Returns (new_pos, new_vel) if collision occurred, or None.
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
                        let normal_impulse = -(1.0 + self.restitution) * vn * normal;
                        let mut new_rel = rel_vel + normal_impulse;
                        // Apply friction to tangential velocity
                        let vt = new_rel - normal * new_rel.dot(normal);
                        new_rel -= vt * (1.0 - self.friction);
                        *vel = self.velocity + new_rel;
                    }
                    return true;
                }
            }
            ColliderShape::Box { aabb } => {
                // Check if inside box inflated by particle radius
                let inflated_min = aabb.min - Vec3::splat(particle_radius);
                let inflated_max = aabb.max + Vec3::splat(particle_radius);

                if pos.x >= inflated_min.x
                    && pos.x <= inflated_max.x
                    && pos.y >= inflated_min.y
                    && pos.y <= inflated_max.y
                    && pos.z >= inflated_min.z
                    && pos.z <= inflated_max.z
                {
                    // Find closest face
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
                        let normal_impulse = -(1.0 + self.restitution) * vn * normal;
                        let mut new_rel = rel_vel + normal_impulse;
                        let vt = new_rel - normal * new_rel.dot(normal);
                        new_rel -= vt * (1.0 - self.friction);
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
                        let normal_impulse = -(1.0 + self.restitution) * vn * normal;
                        *vel = self.velocity + rel_vel + normal_impulse;
                    }
                    return true;
                }
            }
        }
        false
    }
}
