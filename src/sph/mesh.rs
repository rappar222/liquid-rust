use crate::math::Aabb;
use glam::Vec3;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

const MAX_WIREFRAME_SEGMENTS: usize = 24_000;
const BVH_LEAF_TRIANGLES: usize = 6;

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
        let len_sq = cross.length_squared();
        let normal = if len_sq > 1e-16 && len_sq.is_finite() {
            cross / len_sq.sqrt()
        } else {
            Vec3::Y
        };
        Self { a, b, c, normal }
    }

    #[inline]
    fn try_valid(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return None;
        }
        let edge1 = b - a;
        let edge2 = c - a;
        let cross = edge1.cross(edge2);
        let len_sq = cross.length_squared();
        if !len_sq.is_finite() || len_sq <= 1e-16 {
            return None;
        }
        Some(Self {
            a,
            b,
            c,
            normal: cross / len_sq.sqrt(),
        })
    }

    /// Computes closest point on triangle (a, b, c) to point p
    /// Standard Voronoi region method (Christer Ericson, Real-Time Collision Detection)
    #[inline]
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
            return a;
        }

        // Check if P in vertex region outside B
        let bp = p - b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 {
            return b;
        }

        // Check if P in edge region of AB, if so return projection of P onto AB
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let denom = d1 - d3;
            if denom.abs() > 1e-12 {
                return a + ab * (d1 / denom);
            }
            return a;
        }

        // Check if P in vertex region outside C
        let cp = p - c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 {
            return c;
        }

        // Check if P in edge region of AC, if so return projection of P onto AC
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let denom = d2 - d6;
            if denom.abs() > 1e-12 {
                return a + ac * (d2 / denom);
            }
            return a;
        }

        // Check if P in edge region of BC, if so return projection of P onto BC
        let va = d3 * d6 - d5 * d4;
        let d43 = d4 - d3;
        let d56 = d5 - d6;
        if va <= 0.0 && d43 >= 0.0 && d56 >= 0.0 {
            let denom = d43 + d56;
            if denom.abs() > 1e-12 {
                return b + (c - b) * (d43 / denom);
            }
            return b;
        }

        // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
        let sum = va + vb + vc;
        if sum.abs() <= 1e-12 || !sum.is_finite() {
            return a;
        }
        let denom = 1.0 / sum;
        let v = vb * denom;
        let w = vc * denom;
        a + ab * v + ac * w
    }
}

/// 32-byte cache-aligned flat BVH node in local mesh coordinates.
#[derive(Clone, Copy, Debug)]
struct BvhNode {
    min: Vec3,
    left_or_first: u32,
    max: Vec3,
    /// High bit (`0x8000_0000`) marks a leaf with `(count_or_right & 0x7FFF_FFFF)` triangles;
    /// otherwise stores the right child node index.
    count_or_right: u32,
}

#[derive(Clone, Debug, Default)]
struct MeshBvh {
    nodes: Vec<BvhNode>,
}

impl MeshBvh {
    fn build(triangles: &mut [Triangle]) -> Self {
        let n = triangles.len();
        if n == 0 {
            return Self { nodes: Vec::new() };
        }

        let mut tri_mins = Vec::with_capacity(n);
        let mut tri_maxs = Vec::with_capacity(n);
        let mut centroids = Vec::with_capacity(n);
        for tri in triangles.iter() {
            let mn = tri.a.min(tri.b).min(tri.c);
            let mx = tri.a.max(tri.b).max(tri.c);
            tri_mins.push(mn);
            tri_maxs.push(mx);
            centroids.push((tri.a + tri.b + tri.c) * (1.0 / 3.0));
        }

        let mut order: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity((n / 3).max(2) * 2);

        Self::build_recursive(
            &mut nodes,
            &mut order,
            &tri_mins,
            &tri_maxs,
            &centroids,
            0,
            n,
        );

        let reordered: Vec<Triangle> = order.into_iter().map(|i| triangles[i]).collect();
        triangles.copy_from_slice(&reordered);

        Self { nodes }
    }

    fn build_recursive(
        nodes: &mut Vec<BvhNode>,
        order: &mut [usize],
        tri_mins: &[Vec3],
        tri_maxs: &[Vec3],
        centroids: &[Vec3],
        start: usize,
        end: usize,
    ) -> u32 {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            min: Vec3::ZERO,
            left_or_first: 0,
            max: Vec3::ZERO,
            count_or_right: 0,
        });

        let mut b_min = Vec3::splat(f32::INFINITY);
        let mut b_max = Vec3::splat(f32::NEG_INFINITY);
        let mut c_min = Vec3::splat(f32::INFINITY);
        let mut c_max = Vec3::splat(f32::NEG_INFINITY);

        for &idx in &order[start..end] {
            b_min = b_min.min(tri_mins[idx]);
            b_max = b_max.max(tri_maxs[idx]);
            let c = centroids[idx];
            c_min = c_min.min(c);
            c_max = c_max.max(c);
        }

        let count = end - start;
        if count <= BVH_LEAF_TRIANGLES {
            nodes[node_idx as usize] = BvhNode {
                min: b_min,
                left_or_first: start as u32,
                max: b_max,
                count_or_right: 0x8000_0000 | (count as u32),
            };
            return node_idx;
        }

        let c_ext = c_max - c_min;
        let axis = if c_ext.x >= c_ext.y && c_ext.x >= c_ext.z {
            0
        } else if c_ext.y >= c_ext.z {
            1
        } else {
            2
        };

        let mid = start + count / 2;
        if c_ext[axis] > 1e-7 {
            order[start..end].select_nth_unstable_by(mid - start, |&i, &j| {
                centroids[i][axis].total_cmp(&centroids[j][axis])
            });
        }

        let left = Self::build_recursive(nodes, order, tri_mins, tri_maxs, centroids, start, mid);
        let right = Self::build_recursive(nodes, order, tri_mins, tri_maxs, centroids, mid, end);

        nodes[node_idx as usize] = BvhNode {
            min: b_min,
            left_or_first: left,
            max: b_max,
            count_or_right: right,
        };
        node_idx
    }
}

fn build_wireframe_segments(triangles: &[Triangle], local_bounds: Aabb) -> Vec<[Vec3; 2]> {
    if triangles.is_empty() {
        return Vec::new();
    }

    let span = local_bounds
        .size()
        .max_element()
        .max(1e-5);
    let quant_scale = 100_000.0 / span;

    #[inline]
    fn quantize(v: Vec3, scale: f32) -> [i32; 3] {
        [
            (v.x * scale).round() as i32,
            (v.y * scale).round() as i32,
            (v.z * scale).round() as i32,
        ]
    }

    #[derive(Clone, Copy)]
    struct HalfEdge {
        k0: [i32; 3],
        k1: [i32; 3],
        p0: Vec3,
        p1: Vec3,
        normal: Vec3,
    }

    let mut half_edges = Vec::with_capacity(triangles.len() * 3);
    for tri in triangles {
        for (p0, p1) in [(tri.a, tri.b), (tri.b, tri.c), (tri.c, tri.a)] {
            let qa = quantize(p0, quant_scale);
            let qb = quantize(p1, quant_scale);
            if qa == qb {
                continue;
            }
            let (k0, k1, v0, v1) = if qa < qb {
                (qa, qb, p0, p1)
            } else {
                (qb, qa, p1, p0)
            };
            half_edges.push(HalfEdge {
                k0,
                k1,
                p0: v0,
                p1: v1,
                normal: tri.normal,
            });
        }
    }

    half_edges.sort_unstable_by(|a, b| (a.k0, a.k1).cmp(&(b.k0, b.k1)));

    let mut feature_edges: Vec<[Vec3; 2]> = Vec::new();
    let mut smooth_edges: Vec<[Vec3; 2]> = Vec::new();

    let mut i = 0;
    while i < half_edges.len() {
        let first = &half_edges[i];
        let mut j = i + 1;
        let mut is_feature = false;
        while j < half_edges.len()
            && half_edges[j].k0 == first.k0
            && half_edges[j].k1 == first.k1
        {
            if first.normal.dot(half_edges[j].normal) < 0.88 {
                is_feature = true;
            }
            j += 1;
        }
        if j - i == 1 {
            // Open boundary / silhouette edge
            is_feature = true;
        }
        if is_feature {
            feature_edges.push([first.p0, first.p1]);
        } else {
            smooth_edges.push([first.p0, first.p1]);
        }
        i = j;
    }

    if feature_edges.len() + smooth_edges.len() <= MAX_WIREFRAME_SEGMENTS {
        feature_edges.extend(smooth_edges);
        return feature_edges;
    }

    let mut out = Vec::with_capacity(MAX_WIREFRAME_SEGMENTS);
    if feature_edges.len() >= MAX_WIREFRAME_SEGMENTS {
        let step = feature_edges.len() as f64 / MAX_WIREFRAME_SEGMENTS as f64;
        for k in 0..MAX_WIREFRAME_SEGMENTS {
            let idx = ((k as f64 * step) as usize).min(feature_edges.len() - 1);
            out.push(feature_edges[idx]);
        }
        return out;
    }

    out.extend_from_slice(&feature_edges);
    let remaining = MAX_WIREFRAME_SEGMENTS - out.len();
    if remaining > 0 && !smooth_edges.is_empty() {
        let step = smooth_edges.len() as f64 / remaining as f64;
        for k in 0..remaining {
            let idx = ((k as f64 * step) as usize).min(smooth_edges.len() - 1);
            out.push(smooth_edges[idx]);
        }
    }
    out
}

fn compute_triangles_bounds(triangles: &[Triangle]) -> Aabb {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for tri in triangles {
        for v in [tri.a, tri.b, tri.c] {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.x.is_finite() {
        min = Vec3::ZERO;
        max = Vec3::ZERO;
    }
    Aabb::new(min, max)
}

/// Triangle Mesh obstacle loaded from Wavefront .obj file or generated procedurally.
/// Geometry, BVH acceleration tree, and deduplicated wireframe segments are wrapped in `Arc`
/// so cloning or updating position/rotation/scale sliders is $O(1)$ instant.
#[derive(Clone, Debug)]
pub struct TriangleMesh {
    pub name: String,
    pub source_path: Option<String>,
    pub triangles: Arc<Vec<Triangle>>,
    bvh: Arc<MeshBvh>,
    pub wireframe_segments: Arc<Vec<[Vec3; 2]>>,
    pub local_bounds: Aabb,
    pub world_bounds: Aabb,
    pub position: Vec3,
    pub scale: f32,
    pub rotation: Vec3, // Euler angles (pitch X, yaw Y, roll Z) in radians
}

impl TriangleMesh {
    pub fn new(name: impl Into<String>, mut triangles: Vec<Triangle>) -> Self {
        let name = name.into();
        let bvh = MeshBvh::build(&mut triangles);
        let local_bounds = compute_triangles_bounds(&triangles);
        let wireframe = build_wireframe_segments(&triangles, local_bounds);

        let mut mesh = Self {
            name,
            source_path: None,
            triangles: Arc::new(triangles),
            bvh: Arc::new(bvh),
            wireframe_segments: Arc::new(wireframe),
            local_bounds,
            world_bounds: local_bounds,
            position: Vec3::ZERO,
            scale: 1.0,
            rotation: Vec3::ZERO,
        };
        mesh.update_transform();
        mesh
    }

    fn parse_obj_reader<R: BufRead>(mut reader: R) -> Result<Vec<Triangle>, String> {
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<Triangle> = Vec::new();
        let mut face_indices: Vec<usize> = Vec::with_capacity(8);
        let mut raw_line: Vec<u8> = Vec::with_capacity(256);
        let mut line_num = 0usize;

        loop {
            raw_line.clear();
            let bytes_read = reader
                .read_until(b'\n', &mut raw_line)
                .map_err(|e| format!("Read error at line {}: {}", line_num + 1, e))?;
            if bytes_read == 0 {
                break;
            }
            line_num += 1;

            let line = String::from_utf8_lossy(&raw_line);
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
                    if x.is_finite() && y.is_finite() && z.is_finite() {
                        vertices.push(Vec3::new(x, y, z));
                    } else {
                        vertices.push(Vec3::ZERO);
                    }
                }
                Some("f") => {
                    face_indices.clear();
                    let v_len = vertices.len() as isize;
                    for token in parts {
                        let v_str = token.split('/').next().unwrap_or("");
                        if let Ok(idx) = v_str.parse::<isize>() {
                            let resolved = if idx > 0 { idx - 1 } else { v_len + idx };
                            if resolved >= 0 && (resolved as usize) < vertices.len() {
                                face_indices.push(resolved as usize);
                            }
                        }
                    }

                    // Triangulate n-gon via triangle fan, skipping zero-area degenerate slivers
                    if face_indices.len() >= 3 {
                        let i0 = face_indices[0];
                        for i in 1..(face_indices.len() - 1) {
                            let i1 = face_indices[i];
                            let i2 = face_indices[i + 1];
                            if i0 == i1 || i1 == i2 || i0 == i2 {
                                continue;
                            }
                            if let Some(tri) =
                                Triangle::try_valid(vertices[i0], vertices[i1], vertices[i2])
                            {
                                triangles.push(tri);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(triangles)
    }

    /// Load Wavefront .obj file from disk (preserving original file coordinates)
    pub fn from_obj_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)
            .map_err(|e| format!("Cannot open {}: {}", path_ref.display(), e))?;
        let reader = BufReader::with_capacity(64 * 1024, file);

        let triangles = Self::parse_obj_reader(reader)?;
        if triangles.is_empty() {
            return Err(format!("No valid faces found in {}", path_ref.display()));
        }

        let file_stem = path_ref
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("obj_mesh");
        let mut mesh = Self::new(file_stem, triangles);
        mesh.source_path = Some(path_ref.to_string_lossy().to_string());
        Ok(mesh)
    }

    /// Load Wavefront .obj file from disk, centering its local origin and normalizing its
    /// longest local bounding axis to `1.0` in a single pass before building the BVH.
    /// `scale` then directly represents the obstacle's longest physical dimension in meters.
    pub fn load_obj_normalized(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)
            .map_err(|e| format!("Cannot open {}: {}", path_ref.display(), e))?;
        let reader = BufReader::with_capacity(64 * 1024, file);

        let mut triangles = Self::parse_obj_reader(reader)?;
        if triangles.is_empty() {
            return Err(format!("No valid faces found in {}", path_ref.display()));
        }

        let raw_bounds = compute_triangles_bounds(&triangles);
        let center = raw_bounds.center();
        let max_dim = raw_bounds.size().max_element();
        let inv_dim = if max_dim > 1e-6 && max_dim.is_finite() {
            1.0 / max_dim
        } else {
            1.0
        };

        for tri in &mut triangles {
            tri.a = (tri.a - center) * inv_dim;
            tri.b = (tri.b - center) * inv_dim;
            tri.c = (tri.c - center) * inv_dim;
        }

        let file_stem = path_ref
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("obj_mesh");
        let mut mesh = Self::new(file_stem, triangles);
        mesh.source_path = Some(path_ref.to_string_lossy().to_string());
        Ok(mesh)
    }

    /// Load Wavefront .obj data from in-memory string
    pub fn from_obj_str(data: &str, name: impl Into<String>) -> Result<Self, String> {
        let triangles = Self::parse_obj_reader(std::io::Cursor::new(data.as_bytes()))?;
        if triangles.is_empty() {
            return Err("No faces found in OBJ data".to_string());
        }
        Ok(Self::new(name, triangles))
    }

    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    fn rebuild_geometry(&mut self, mut tris: Vec<Triangle>) {
        let bvh = MeshBvh::build(&mut tris);
        self.local_bounds = compute_triangles_bounds(&tris);
        self.wireframe_segments = Arc::new(build_wireframe_segments(&tris, self.local_bounds));
        self.triangles = Arc::new(tris);
        self.bvh = Arc::new(bvh);
        self.update_transform();
    }

    /// Center the mesh around its local origin (0, 0, 0)
    pub fn center_origin(&mut self) {
        let center = self.local_bounds.center();
        let mut tris = (*self.triangles).clone();
        for tri in &mut tris {
            tri.a -= center;
            tri.b -= center;
            tri.c -= center;
        }
        self.rebuild_geometry(tris);
    }

    /// Scale the mesh to fit within target bounding diameter
    pub fn normalize_scale(&mut self, target_size: f32) {
        let extents = self.local_bounds.size();
        let max_dim = extents.x.max(extents.y).max(extents.z);
        if max_dim > 1e-6 {
            let factor = target_size / max_dim;
            let mut tris = (*self.triangles).clone();
            for tri in &mut tris {
                tri.a *= factor;
                tri.b *= factor;
                tri.c *= factor;
            }
            self.rebuild_geometry(tris);
        }
    }

    pub fn set_transform(&mut self, position: Vec3, scale: f32, rotation: Vec3) {
        self.position = position;
        self.scale = scale.max(1e-4);
        self.rotation = rotation;
        self.update_transform();
    }

    /// $O(1)$ world AABB update from the 8 corners of `local_bounds`.
    pub fn update_transform(&mut self) {
        self.scale = self.scale.max(1e-4);
        let rot_mat = glam::Mat3::from_euler(
            glam::EulerRot::YXZ,
            self.rotation.y,
            self.rotation.x,
            self.rotation.z,
        );
        let s = self.scale;

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for corner in self.local_bounds.corners() {
            let transformed = self.position + (rot_mat * (corner * s));
            min = min.min(transformed);
            max = max.max(transformed);
        }

        if !min.x.is_finite() {
            min = self.position;
            max = self.position;
        }
        self.world_bounds = Aabb::new(min, max);
    }

    /// Resolve particle collision against this mesh using local-space BVH traversal.
    pub fn resolve_collision(
        &self,
        pos: &mut Vec3,
        vel: &mut Vec3,
        particle_radius: f32,
        restitution: f32,
        friction: f32,
    ) -> bool {
        if !pos.is_finite() || self.bvh.nodes.is_empty() {
            return false;
        }

        // Fast world-AABB rejection
        let inflated_min = self.world_bounds.min - Vec3::splat(particle_radius);
        let inflated_max = self.world_bounds.max + Vec3::splat(particle_radius);
        if pos.x < inflated_min.x
            || pos.x > inflated_max.x
            || pos.y < inflated_min.y
            || pos.y > inflated_max.y
            || pos.z < inflated_min.z
            || pos.z > inflated_max.z
        {
            return false;
        }

        // Transform particle into mesh local space
        let rot_mat = glam::Mat3::from_euler(
            glam::EulerRot::YXZ,
            self.rotation.y,
            self.rotation.x,
            self.rotation.z,
        );
        let inv_rot = rot_mat.transpose();
        let inv_s = 1.0 / self.scale.max(1e-4);

        let local_p = inv_rot * (*pos - self.position) * inv_s;
        let local_r = particle_radius * inv_s;
        let mut closest_dist_sq = local_r * local_r;
        let mut query_rad = local_r;

        let mut collided = false;
        let mut best_normal = Vec3::ZERO;
        let mut best_correction = Vec3::ZERO;

        let nodes = &self.bvh.nodes;
        let tris = &*self.triangles;

        let mut stack = [0u32; 64];
        let mut sp = 1usize;

        while sp > 0 {
            sp -= 1;
            let node = &nodes[stack[sp] as usize];

            if local_p.x + query_rad < node.min.x
                || local_p.x - query_rad > node.max.x
                || local_p.y + query_rad < node.min.y
                || local_p.y - query_rad > node.max.y
                || local_p.z + query_rad < node.min.z
                || local_p.z - query_rad > node.max.z
            {
                continue;
            }

            if (node.count_or_right & 0x8000_0000) != 0 {
                let first = node.left_or_first as usize;
                let count = (node.count_or_right & 0x7FFF_FFFF) as usize;
                for tri in &tris[first..first + count] {
                    let closest = tri.closest_point(local_p);
                    let diff = local_p - closest;
                    let dist_sq = diff.length_squared();

                    if dist_sq < closest_dist_sq {
                        let dist = dist_sq.sqrt();
                        let (normal, pen) = if dist > 1e-7 {
                            let n = diff / dist;
                            if n.dot(tri.normal) < 0.0 {
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
                        query_rad = dist;
                        collided = true;
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.count_or_right;
                stack[sp + 1] = node.left_or_first;
                sp += 2;
            }
        }

        if collided {
            let world_correction = (rot_mat * best_correction) * self.scale;
            let world_normal = (rot_mat * best_normal).normalize_or_zero();
            if world_correction.is_finite() && world_normal.length_squared() > 0.5 {
                *pos += world_correction;

                let vn = vel.dot(world_normal);
                if vn < 0.0 {
                    let normal_impulse = -(1.0 + restitution) * vn * world_normal;
                    let mut new_vel = *vel + normal_impulse;
                    let vt = new_vel - world_normal * new_vel.dot(world_normal);
                    new_vel -= vt * (1.0 - friction);
                    if new_vel.is_finite() {
                        *vel = new_vel;
                    }
                }
                return true;
            }
        }

        false
    }
}
