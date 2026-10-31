use crate::math::Aabb;
use glam::Vec3;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub normal: Vec3,
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let edge1 = b - a;
        let edge2 = c - a;
        let cross = edge1.cross(edge2);
        let normal = if cross.length_squared() > 1e-12 {
            cross.normalize()
        } else {
            Vec3::Y
        };
        Self { a, b, c, normal }
    }

    /// Computes closest point on triangle (a, b, c) to point p
    /// Standard Voronoi region method (Christer Ericson, Real-Time Collision Detection)
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        let a = self.a;
        let b = self.b;
        let c = self.c;

        // Check if P in vertex region outside A
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return a; // barycentric coordinates (1,0,0)
        }

        // Check if P in vertex region outside B
        let bp = p - b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 {
            return b; // barycentric coordinates (0,1,0)
        }

        // Check if P in edge region of AB, if so return projection of P onto AB
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return a + ab * v; // barycentric coordinates (1-v,v,0)
        }

        // Check if P in vertex region outside C
        let cp = p - c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 {
            return c; // barycentric coordinates (0,0,1)
        }

        // Check if P in edge region of AC, if so return projection of P onto AC
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return a + ac * w; // barycentric coordinates (1-w,0,w)
        }

        // Check if P in edge region of BC, if so return projection of P onto BC
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + (c - b) * w; // barycentric coordinates (0,1-w,w)
        }

        // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        a + ab * v + ac * w
    }
}

/// Triangle Mesh obstacle loaded from Wavefront .obj file or generated procedurally
#[derive(Clone, Debug)]
pub struct TriangleMesh {
    pub name: String,
    pub source_path: Option<String>,
    pub triangles: Vec<Triangle>,
    pub local_bounds: Aabb,
    pub world_bounds: Aabb,
    pub position: Vec3,
    pub scale: f32,
    pub rotation: Vec3, // Euler angles (pitch, yaw, roll) in radians
}

impl TriangleMesh {
    pub fn new(name: impl Into<String>, triangles: Vec<Triangle>) -> Self {
        let name = name.into();
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for tri in &triangles {
            for v in [tri.a, tri.b, tri.c] {
                min = min.min(v);
                max = max.max(v);
            }
        }

        if min.x.is_infinite() {
            min = Vec3::ZERO;
            max = Vec3::ZERO;
        }

        let local_bounds = Aabb::new(min, max);
        let mut mesh = Self {
            name,
            source_path: None,
            triangles,
            local_bounds,
            world_bounds: local_bounds,
            position: Vec3::ZERO,
            scale: 1.0,
            rotation: Vec3::ZERO,
        };
        mesh.update_transform();
        mesh
    }

    /// Load Wavefront .obj file from disk
    pub fn from_obj_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| format!("Cannot open {}: {}", path_ref.display(), e))?;
        let reader = BufReader::new(file);

        let mut vertices: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<Triangle> = Vec::new();

        for (line_num, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| format!("Read error at line {}: {}", line_num + 1, e))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    vertices.push(Vec3::new(x, y, z));
                }
                Some("f") => {
                    let mut face_indices: Vec<usize> = Vec::new();
                    for token in parts {
                        // Handle v, v/vt, v//vn, v/vt/vn
                        let v_str = token.split('/').next().unwrap_or("");
                        if let Ok(idx) = v_str.parse::<i32>() {
                            let zero_idx = if idx > 0 {
                                (idx - 1) as usize
                            } else {
                                (vertices.len() as i32 + idx) as usize
                            };
                            if zero_idx < vertices.len() {
                                face_indices.push(zero_idx);
                            }
                        }
                    }

                    // Triangulate n-gon via triangle fan
                    if face_indices.len() >= 3 {
                        for i in 1..(face_indices.len() - 1) {
                            let a = vertices[face_indices[0]];
                            let b = vertices[face_indices[i]];
                            let c = vertices[face_indices[i + 1]];
                            triangles.push(Triangle::new(a, b, c));
                        }
                    }
                }
                _ => {}
            }
        }

        if triangles.is_empty() {
            return Err(format!("No valid faces found in {}", path_ref.display()));
        }

        let file_stem = path_ref.file_stem().and_then(|s| s.to_str()).unwrap_or("obj_mesh");
        let mut mesh = Self::new(file_stem, triangles);
        mesh.source_path = Some(path_ref.to_string_lossy().to_string());
        Ok(mesh)
    }

    /// Load Wavefront .obj data from in-memory string
    pub fn from_obj_str(data: &str, name: impl Into<String>) -> Result<Self, String> {
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<Triangle> = Vec::new();

        for line in data.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    vertices.push(Vec3::new(x, y, z));
                }
                Some("f") => {
                    let mut face_indices: Vec<usize> = Vec::new();
                    for token in parts {
                        let v_str = token.split('/').next().unwrap_or("");
                        if let Ok(idx) = v_str.parse::<i32>() {
                            let zero_idx = if idx > 0 {
                                (idx - 1) as usize
                            } else {
                                (vertices.len() as i32 + idx) as usize
                            };
                            if zero_idx < vertices.len() {
                                face_indices.push(zero_idx);
                            }
                        }
                    }

                    if face_indices.len() >= 3 {
                        for i in 1..(face_indices.len() - 1) {
                            let a = vertices[face_indices[0]];
                            let b = vertices[face_indices[i]];
                            let c = vertices[face_indices[i + 1]];
                            triangles.push(Triangle::new(a, b, c));
                        }
                    }
                }
                _ => {}
            }
        }

        if triangles.is_empty() {
            return Err("No faces found in OBJ data".to_string());
        }

        Ok(Self::new(name, triangles))
    }

    /// Center the mesh around its local origin (0, 0, 0)
    pub fn center_origin(&mut self) {
        let center = self.local_bounds.center();
        for tri in &mut self.triangles {
            tri.a -= center;
            tri.b -= center;
            tri.c -= center;
        }
        self.local_bounds.min -= center;
        self.local_bounds.max -= center;
        self.update_transform();
    }

    /// Scale the mesh to fit within target bounding diameter
    pub fn normalize_scale(&mut self, target_size: f32) {
        let extents = self.local_bounds.size();
        let max_dim = extents.x.max(extents.y).max(extents.z);
        if max_dim > 1e-6 {
            let factor = target_size / max_dim;
            for tri in &mut self.triangles {
                tri.a *= factor;
                tri.b *= factor;
                tri.c *= factor;
            }
            self.local_bounds.min *= factor;
            self.local_bounds.max *= factor;
            self.update_transform();
        }
    }

    pub fn set_transform(&mut self, position: Vec3, scale: f32, rotation: Vec3) {
        self.position = position;
        self.scale = scale.max(1e-4);
        self.rotation = rotation;
        self.update_transform();
    }

    pub fn update_transform(&mut self) {
        let rot_mat = glam::Mat3::from_euler(glam::EulerRot::YXZ, self.rotation.y, self.rotation.x, self.rotation.z);
        let s = self.scale;

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for tri in &self.triangles {
            for v in [tri.a, tri.b, tri.c] {
                let transformed = self.position + (rot_mat * (v * s));
                min = min.min(transformed);
                max = max.max(transformed);
            }
        }

        self.world_bounds = Aabb::new(min, max);
    }

    /// Resolve particle collision against this mesh
    pub fn resolve_collision(
        &self,
        pos: &mut Vec3,
        vel: &mut Vec3,
        particle_radius: f32,
        restitution: f32,
        friction: f32,
    ) -> bool {
        // Fast AABB rejection: if outside inflated world bounds, no collision
        let inflated_min = self.world_bounds.min - Vec3::splat(particle_radius);
        let inflated_max = self.world_bounds.max + Vec3::splat(particle_radius);
        if pos.x < inflated_min.x || pos.x > inflated_max.x
            || pos.y < inflated_min.y || pos.y > inflated_max.y
            || pos.z < inflated_min.z || pos.z > inflated_max.z
        {
            return false;
        }

        // Transform particle into mesh local space for fast evaluation
        let rot_mat = glam::Mat3::from_euler(glam::EulerRot::YXZ, self.rotation.y, self.rotation.x, self.rotation.z);
        let inv_rot = rot_mat.transpose();
        let inv_s = 1.0 / self.scale;

        let local_p = inv_rot * (*pos - self.position) * inv_s;
        let local_r = particle_radius * inv_s;
        let local_r2 = local_r * local_r;

        let mut collided = false;
        let mut closest_dist_sq = local_r2;
        let mut best_normal = Vec3::ZERO;
        let mut best_correction = Vec3::ZERO;

        // Test triangles
        for tri in &self.triangles {
            let closest = tri.closest_point(local_p);
            let diff = local_p - closest;
            let dist_sq = diff.length_squared();

            if dist_sq < closest_dist_sq {
                let dist = dist_sq.sqrt();
                let (normal, pen) = if dist > 1e-7 {
                    let n = diff / dist;
                    if n.dot(tri.normal) < 0.0 {
                        // Particle penetrated behind triangle face; push outward along triangle normal
                        (tri.normal, local_r + dist)
                    } else {
                        (n, local_r - dist)
                    }
                } else {
                    (tri.normal, local_r)
                };

                best_correction = normal * pen;
                best_normal = normal;
                closest_dist_sq = dist_sq;
                collided = true;
            }
        }

        if collided {
            // Transform correction back to world space
            let world_correction = (rot_mat * best_correction) * self.scale;
            let world_normal = (rot_mat * best_normal).normalize();

            *pos += world_correction;

            // Inelastic velocity response
            let vn = vel.dot(world_normal);
            if vn < 0.0 {
                let normal_impulse = -(1.0 + restitution) * vn * world_normal;
                let mut new_vel = *vel + normal_impulse;
                let vt = new_vel - world_normal * new_vel.dot(world_normal);
                new_vel -= vt * (1.0 - friction);
                *vel = new_vel;
            }
            return true;
        }

        false
    }
}
