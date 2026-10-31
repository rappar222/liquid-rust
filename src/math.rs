use glam::{Mat4, Vec3};

/// Axis-Aligned Bounding Box
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Clamp a point to inside the AABB with an optional margin
    pub fn clamp_point(&self, p: Vec3, margin: f32) -> Vec3 {
        Vec3::new(
            p.x.clamp(self.min.x + margin, self.max.x - margin),
            p.y.clamp(self.min.y + margin, self.max.y - margin),
            p.z.clamp(self.min.z + margin, self.max.z - margin),
        )
    }

    /// Get the 8 corners of the bounding box
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
        ]
    }

    /// 12 line segments (pairs of vertex indices) representing the edges of the box
    pub fn edge_indices() -> [(usize, usize); 12] {
        [
            // Bottom face
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            // Top face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            // Connecting vertical edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
    }
}

/// 3D Ray for mouse picking and interaction
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
        }
    }

    /// Point along ray at distance t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Distance from ray to a point in 3D
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        let v = point - self.origin;
        let proj = v.dot(self.direction);
        if proj <= 0.0 {
            v.length()
        } else {
            let closest = self.origin + self.direction * proj;
            (point - closest).length()
        }
    }

    /// Intersect with plane defined by point and normal
    pub fn intersect_plane(&self, plane_p: Vec3, plane_n: Vec3) -> Option<f32> {
        let denom = self.direction.dot(plane_n);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (plane_p - self.origin).dot(plane_n) / denom;
        if t >= 0.0 {
            Some(t)
        } else {
            None
        }
    }
}

/// Screen to world ray unprojection
pub fn unproject_mouse_ray(
    screen_pos: glam::Vec2,
    screen_size: glam::Vec2,
    view: Mat4,
    proj: Mat4,
) -> Ray {
    let ndc_x = (2.0 * screen_pos.x) / screen_size.x - 1.0;
    let ndc_y = 1.0 - (2.0 * screen_pos.y) / screen_size.y;

    let inv_pv = (proj * view).inverse();

    let near_h = inv_pv.project_point3(Vec3::new(ndc_x, ndc_y, -1.0));
    let far_h = inv_pv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));

    let dir = (far_h - near_h).normalize_or_zero();
    Ray::new(near_h, dir)
}
