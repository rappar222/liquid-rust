use crate::math::Aabb;
use crate::sph::collider::{Collider, ColliderShape};
use crate::sph::faucet::{CubeFace, Drain, Faucet, FaucetShape};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use glow::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ParticleVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
}

pub struct GlRenderer {
    particle_program: NativeProgram,
    line_program: NativeProgram,
    particle_vao: NativeVertexArray,
    particle_vbo: NativeBuffer,
    line_vao: NativeVertexArray,
    line_vbo: NativeBuffer,
}

impl GlRenderer {
    pub fn new(gl: &Arc<glow::Context>) -> Result<Self, String> {
        unsafe {
            // 1. Compile particle shader
            let p_vs_src = r#"#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec4 a_color;

uniform mat4 u_view_proj;
uniform float u_point_radius;
uniform float u_viewport_height;
uniform float u_tan_half_fov;

out vec4 v_color;

void main() {
    v_color = a_color;
    vec4 clip_pos = u_view_proj * vec4(a_pos, 1.0);
    gl_Position = clip_pos;
    float dist = max(clip_pos.w, 0.001);
    float size_px = (u_point_radius / (dist * u_tan_half_fov)) * u_viewport_height;
    gl_PointSize = clamp(size_px, 2.0, 128.0);
}
"#;

            let p_fs_src = r#"#version 330 core
in vec4 v_color;
uniform vec3 u_light_dir;
uniform int u_lit_spheres;

out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }

    if (u_lit_spheres != 0) {
        float z = sqrt(1.0 - r2);
        vec3 normal = vec3(coord.x, -coord.y, z);

        float diff = max(dot(normal, u_light_dir), 0.0);
        vec3 half_dir = normalize(u_light_dir + vec3(0.0, 0.0, 1.0));
        float spec = pow(max(dot(normal, half_dir), 0.0), 32.0);

        vec3 rgb = v_color.rgb * (0.45 + 0.55 * diff) + vec3(0.35) * spec;
        FragColor = vec4(rgb, 1.0);
    } else {
        FragColor = vec4(v_color.rgb, 1.0);
    }
}
"#;

            let particle_program = create_shader_program(gl, p_vs_src, p_fs_src)?;

            // 2. Compile line shader
            let l_vs_src = r#"#version 330 core
layout (location = 0) in vec3 a_pos;
uniform mat4 u_view_proj;

void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}
"#;

            let l_fs_src = r#"#version 330 core
uniform vec4 u_line_color;
out vec4 FragColor;

void main() {
    FragColor = u_line_color;
}
"#;

            let line_program = create_shader_program(gl, l_vs_src, l_fs_src)?;

            // 3. Create Particle VAO & VBO
            let particle_vao = gl.create_vertex_array()?;
            let particle_vbo = gl.create_buffer()?;

            gl.bind_vertex_array(Some(particle_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(particle_vbo));

            let stride = std::mem::size_of::<ParticleVertex>() as i32;

            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);

            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(
                1,
                4,
                glow::FLOAT,
                false,
                stride,
                (3 * std::mem::size_of::<f32>()) as i32,
            );

            // 4. Create Line VAO & VBO
            let line_vao = gl.create_vertex_array()?;
            let line_vbo = gl.create_buffer()?;

            gl.bind_vertex_array(Some(line_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(line_vbo));

            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(
                0,
                3,
                glow::FLOAT,
                false,
                std::mem::size_of::<LineVertex>() as i32,
                0,
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            Ok(Self {
                particle_program,
                line_program,
                particle_vao,
                particle_vbo,
                line_vao,
                line_vbo,
            })
        }
    }

    /// Render particles and simulation bounds
    pub fn render(
        &mut self,
        gl: &Arc<glow::Context>,
        view_proj: Mat4,
        viewport_size: [f32; 2],
        particle_radius: f32,
        fov_y: f32,
        particles_data: &[ParticleVertex],
        lit_spheres: bool,
        chamber: &crate::sph::chamber::Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        faucets: &[Faucet],
        drains: &[Drain],
        is_2d: bool,
    ) {
        unsafe {
            if is_2d {
                gl.disable(glow::DEPTH_TEST);
                gl.depth_mask(false);
            } else {
                gl.enable(glow::DEPTH_TEST);
                gl.depth_func(glow::LEQUAL);
                gl.depth_mask(true);
                // Clear depth buffer for this frame's 3D geometry so previous frames' depth is not stale
                gl.disable(glow::SCISSOR_TEST);
                gl.clear(glow::DEPTH_BUFFER_BIT);
                gl.enable(glow::SCISSOR_TEST);
            }
            gl.disable(glow::BLEND);
            gl.enable(glow::PROGRAM_POINT_SIZE);

            // 1. Draw Chamber, Colliders, Faucets, and Drains
            let mut line_vertices = Vec::new();
            let chamber_wireframe = if is_2d {
                chamber.generate_wireframe_lines_2d(&bounds)
            } else {
                chamber.generate_wireframe_lines(&bounds)
            };
            for (p0, p1) in chamber_wireframe {
                line_vertices.push(LineVertex { pos: p0.to_array() });
                line_vertices.push(LineVertex { pos: p1.to_array() });
            }
            for collider in colliders {
                if collider.enabled {
                    if is_2d {
                        match &collider.shape {
                            ColliderShape::Sphere { center, radius } => {
                                Self::generate_circle_lines_xy(*center, *radius, 32, &mut line_vertices);
                            }
                            ColliderShape::Box { aabb } => {
                                let p0 = Vec3::new(aabb.min.x, aabb.min.y, 0.0);
                                let p1 = Vec3::new(aabb.max.x, aabb.min.y, 0.0);
                                let p2 = Vec3::new(aabb.max.x, aabb.max.y, 0.0);
                                let p3 = Vec3::new(aabb.min.x, aabb.max.y, 0.0);
                                line_vertices.push(LineVertex { pos: p0.to_array() });
                                line_vertices.push(LineVertex { pos: p1.to_array() });
                                line_vertices.push(LineVertex { pos: p1.to_array() });
                                line_vertices.push(LineVertex { pos: p2.to_array() });
                                line_vertices.push(LineVertex { pos: p2.to_array() });
                                line_vertices.push(LineVertex { pos: p3.to_array() });
                                line_vertices.push(LineVertex { pos: p3.to_array() });
                                line_vertices.push(LineVertex { pos: p0.to_array() });
                            }
                            ColliderShape::Cylinder { center, radius, .. } => {
                                Self::generate_circle_lines_xy(*center, *radius, 32, &mut line_vertices);
                            }
                            ColliderShape::Mesh(mesh) => {
                                Self::generate_mesh_lines(mesh, &mut line_vertices);
                            }
                        }
                    } else {
                        match &collider.shape {
                            ColliderShape::Sphere { center, radius } => {
                                Self::generate_sphere_lines(*center, *radius, 24, &mut line_vertices);
                            }
                            ColliderShape::Box { aabb } => {
                                Self::generate_box_lines(aabb, &mut line_vertices);
                            }
                            ColliderShape::Cylinder { center, radius, height } => {
                                Self::generate_cylinder_lines(*center, *radius, *height, 16, &mut line_vertices);
                            }
                            ColliderShape::Mesh(mesh) => {
                                Self::generate_mesh_lines(mesh, &mut line_vertices);
                            }
                        }
                    }
                }
            }

            for faucet in faucets {
                if faucet.enabled {
                    Self::generate_faucet_lines(faucet, &mut line_vertices);
                }
            }

            for drain in drains {
                if drain.enabled {
                    Self::generate_drain_lines(drain, &bounds, &mut line_vertices);
                }
            }

            if !line_vertices.is_empty() {
                gl.use_program(Some(self.line_program));

                let u_vp = gl.get_uniform_location(self.line_program, "u_view_proj");
                let u_col = gl.get_uniform_location(self.line_program, "u_line_color");

                gl.uniform_matrix_4_f32_slice(u_vp.as_ref(), false, &view_proj.to_cols_array());
                gl.uniform_4_f32(u_col.as_ref(), 0.35, 0.65, 0.95, 0.85);

                gl.bind_vertex_array(Some(self.line_vao));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.line_vbo));
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&line_vertices),
                    glow::DYNAMIC_DRAW,
                );

                gl.line_width(1.5);
                gl.draw_arrays(glow::LINES, 0, line_vertices.len() as i32);
            }

            // 2. Draw Particles
            if !particles_data.is_empty() {
                gl.use_program(Some(self.particle_program));

                let u_vp = gl.get_uniform_location(self.particle_program, "u_view_proj");
                let u_pr = gl.get_uniform_location(self.particle_program, "u_point_radius");
                let u_vh = gl.get_uniform_location(self.particle_program, "u_viewport_height");
                let u_tf = gl.get_uniform_location(self.particle_program, "u_tan_half_fov");
                let u_ld = gl.get_uniform_location(self.particle_program, "u_light_dir");
                let u_lit = gl.get_uniform_location(self.particle_program, "u_lit_spheres");

                let tan_half_fov = (fov_y * 0.5).tan();

                gl.uniform_matrix_4_f32_slice(u_vp.as_ref(), false, &view_proj.to_cols_array());
                gl.uniform_1_f32(u_pr.as_ref(), particle_radius);
                gl.uniform_1_f32(u_vh.as_ref(), viewport_size[1]);
                gl.uniform_1_f32(u_tf.as_ref(), tan_half_fov);

                let light_dir = Vec3::new(0.4, 0.8, 0.5).normalize();
                gl.uniform_3_f32(u_ld.as_ref(), light_dir.x, light_dir.y, light_dir.z);
                gl.uniform_1_i32(u_lit.as_ref(), if lit_spheres { 1 } else { 0 });

                gl.bind_vertex_array(Some(self.particle_vao));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.particle_vbo));
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(particles_data),
                    glow::DYNAMIC_DRAW,
                );

                gl.draw_arrays(glow::POINTS, 0, particles_data.len() as i32);
            }

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);

            // Clean up OpenGL state for egui
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(true);
            gl.disable(glow::PROGRAM_POINT_SIZE);
            gl.enable(glow::BLEND);
        }
    }

    fn generate_circle_lines_xy(
        center: Vec3,
        radius: f32,
        segments: usize,
        lines: &mut Vec<LineVertex>,
    ) {
        let d_theta = std::f32::consts::TAU / segments as f32;
        for i in 0..segments {
            let t0 = i as f32 * d_theta;
            let t1 = (i + 1) as f32 * d_theta;
            lines.push(LineVertex {
                pos: (center + Vec3::new(t0.cos() * radius, t0.sin() * radius, 0.0)).to_array(),
            });
            lines.push(LineVertex {
                pos: (center + Vec3::new(t1.cos() * radius, t1.sin() * radius, 0.0)).to_array(),
            });
        }
    }

    fn generate_box_lines(box_bounds: &Aabb, lines: &mut Vec<LineVertex>) {
        let corners = box_bounds.corners();
        let indices = Aabb::edge_indices();
        for (i0, i1) in indices {
            lines.push(LineVertex {
                pos: corners[i0].to_array(),
            });
            lines.push(LineVertex {
                pos: corners[i1].to_array(),
            });
        }
    }

    fn generate_sphere_lines(
        center: Vec3,
        radius: f32,
        segments: usize,
        lines: &mut Vec<LineVertex>,
    ) {
        let d_theta = std::f32::consts::TAU / segments as f32;

        // XY circle
        for i in 0..segments {
            let t0 = i as f32 * d_theta;
            let t1 = (i + 1) as f32 * d_theta;
            lines.push(LineVertex {
                pos: (center + Vec3::new(t0.cos() * radius, t0.sin() * radius, 0.0)).to_array(),
            });
            lines.push(LineVertex {
                pos: (center + Vec3::new(t1.cos() * radius, t1.sin() * radius, 0.0)).to_array(),
            });
        }
        // XZ circle
        for i in 0..segments {
            let t0 = i as f32 * d_theta;
            let t1 = (i + 1) as f32 * d_theta;
            lines.push(LineVertex {
                pos: (center + Vec3::new(t0.cos() * radius, 0.0, t0.sin() * radius)).to_array(),
            });
            lines.push(LineVertex {
                pos: (center + Vec3::new(t1.cos() * radius, 0.0, t1.sin() * radius)).to_array(),
            });
        }
        // YZ circle
        for i in 0..segments {
            let t0 = i as f32 * d_theta;
            let t1 = (i + 1) as f32 * d_theta;
            lines.push(LineVertex {
                pos: (center + Vec3::new(0.0, t0.cos() * radius, t0.sin() * radius)).to_array(),
            });
            lines.push(LineVertex {
                pos: (center + Vec3::new(0.0, t1.cos() * radius, t1.sin() * radius)).to_array(),
            });
        }
    }

    fn generate_cylinder_lines(
        center: Vec3,
        radius: f32,
        height: f32,
        segments: usize,
        lines: &mut Vec<LineVertex>,
    ) {
        let half_h = height * 0.5;
        let d_theta = std::f32::consts::TAU / segments as f32;

        let top_center = center + Vec3::new(0.0, half_h, 0.0);
        let bot_center = center - Vec3::new(0.0, half_h, 0.0);

        for i in 0..segments {
            let t0 = i as f32 * d_theta;
            let t1 = (i + 1) as f32 * d_theta;

            let p0_top = top_center + Vec3::new(t0.cos() * radius, 0.0, t0.sin() * radius);
            let p1_top = top_center + Vec3::new(t1.cos() * radius, 0.0, t1.sin() * radius);
            lines.push(LineVertex { pos: p0_top.to_array() });
            lines.push(LineVertex { pos: p1_top.to_array() });

            let p0_bot = bot_center + Vec3::new(t0.cos() * radius, 0.0, t0.sin() * radius);
            let p1_bot = bot_center + Vec3::new(t1.cos() * radius, 0.0, t1.sin() * radius);
            lines.push(LineVertex { pos: p0_bot.to_array() });
            lines.push(LineVertex { pos: p1_bot.to_array() });

            if i % (segments / 4).max(1) == 0 {
                lines.push(LineVertex { pos: p0_top.to_array() });
                lines.push(LineVertex { pos: p0_bot.to_array() });
            }
        }
    }

    fn generate_mesh_lines(
        mesh: &crate::sph::mesh::TriangleMesh,
        lines: &mut Vec<LineVertex>,
    ) {
        let rot_mat = glam::Mat3::from_euler(glam::EulerRot::YXZ, mesh.rotation.y, mesh.rotation.x, mesh.rotation.z);
        let s = mesh.scale;
        let pos = mesh.position;

        for tri in &mesh.triangles {
            let v0 = pos + (rot_mat * (tri.a * s));
            let v1 = pos + (rot_mat * (tri.b * s));
            let v2 = pos + (rot_mat * (tri.c * s));

            lines.push(LineVertex { pos: v0.to_array() });
            lines.push(LineVertex { pos: v1.to_array() });

            lines.push(LineVertex { pos: v1.to_array() });
            lines.push(LineVertex { pos: v2.to_array() });

            lines.push(LineVertex { pos: v2.to_array() });
            lines.push(LineVertex { pos: v0.to_array() });
        }
    }

    fn generate_faucet_lines(
        faucet: &Faucet,
        lines: &mut Vec<LineVertex>,
    ) {
        let pos = faucet.position;
        let flow_dir = if faucet.direction.length_squared() > 1e-6 {
            faucet.direction.normalize()
        } else {
            -Vec3::Y
        };

        // Draw nozzle aperture
        match faucet.shape {
            FaucetShape::Circle { radius } => {
                let segments = 16;
                let d_theta = std::f32::consts::TAU / segments as f32;
                let up = if flow_dir.y.abs() < 0.95 { Vec3::Y } else { Vec3::X };
                let right = flow_dir.cross(up).normalize();
                let forward = right.cross(flow_dir).normalize();

                for i in 0..segments {
                    let t0 = i as f32 * d_theta;
                    let t1 = (i + 1) as f32 * d_theta;
                    let p0 = pos + right * (t0.cos() * radius) + forward * (t0.sin() * radius);
                    let p1 = pos + right * (t1.cos() * radius) + forward * (t1.sin() * radius);
                    lines.push(LineVertex { pos: p0.to_array() });
                    lines.push(LineVertex { pos: p1.to_array() });
                }
            }
            FaucetShape::Rectangle { width, depth } => {
                let up = if flow_dir.y.abs() < 0.95 { Vec3::Y } else { Vec3::X };
                let right = flow_dir.cross(up).normalize() * (width * 0.5);
                let forward = right.cross(flow_dir).normalize() * (depth * 0.5);

                let c0 = pos - right - forward;
                let c1 = pos + right - forward;
                let c2 = pos + right + forward;
                let c3 = pos - right + forward;

                lines.push(LineVertex { pos: c0.to_array() });
                lines.push(LineVertex { pos: c1.to_array() });
                lines.push(LineVertex { pos: c1.to_array() });
                lines.push(LineVertex { pos: c2.to_array() });
                lines.push(LineVertex { pos: c2.to_array() });
                lines.push(LineVertex { pos: c3.to_array() });
                lines.push(LineVertex { pos: c3.to_array() });
                lines.push(LineVertex { pos: c0.to_array() });
            }
            FaucetShape::Point => {
                let s = 0.03;
                lines.push(LineVertex { pos: (pos - Vec3::X * s).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::X * s).to_array() });
                lines.push(LineVertex { pos: (pos - Vec3::Z * s).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::Z * s).to_array() });
            }
        }

        // Draw directional flow arrow
        let arrow_len = 0.12;
        let arrow_tip = pos + flow_dir * arrow_len;
        lines.push(LineVertex { pos: pos.to_array() });
        lines.push(LineVertex { pos: arrow_tip.to_array() });
    }

    fn generate_drain_lines(
        drain: &Drain,
        bounds: &Aabb,
        lines: &mut Vec<LineVertex>,
    ) {
        let segments = 20;
        let d_theta = std::f32::consts::TAU / segments as f32;
        let radius = drain.radius;
        let pos = match drain.face {
            CubeFace::Floor => Vec3::new(drain.position.x, bounds.min.y + 0.001, drain.position.z),
            CubeFace::Ceiling => Vec3::new(drain.position.x, bounds.max.y - 0.001, drain.position.z),
            CubeFace::Left => Vec3::new(bounds.min.x + 0.001, drain.position.y, drain.position.z),
            CubeFace::Right => Vec3::new(bounds.max.x - 0.001, drain.position.y, drain.position.z),
            CubeFace::Back => Vec3::new(drain.position.x, drain.position.y, bounds.min.z + 0.001),
            CubeFace::Front => Vec3::new(drain.position.x, drain.position.y, bounds.max.z - 0.001),
            CubeFace::Custom => drain.position,
        };

        // Draw ring according to orientation
        for i in 0..segments {
            let t0 = i as f32 * d_theta;
            let t1 = (i + 1) as f32 * d_theta;

            let (p0, p1) = match drain.face {
                CubeFace::Floor | CubeFace::Ceiling => (
                    pos + Vec3::new(t0.cos() * radius, 0.0, t0.sin() * radius),
                    pos + Vec3::new(t1.cos() * radius, 0.0, t1.sin() * radius),
                ),
                CubeFace::Left | CubeFace::Right => (
                    pos + Vec3::new(0.0, t0.cos() * radius, t0.sin() * radius),
                    pos + Vec3::new(0.0, t1.cos() * radius, t1.sin() * radius),
                ),
                CubeFace::Back | CubeFace::Front | CubeFace::Custom => (
                    pos + Vec3::new(t0.cos() * radius, t0.sin() * radius, 0.0),
                    pos + Vec3::new(t1.cos() * radius, t1.sin() * radius, 0.0),
                ),
            };

            lines.push(LineVertex { pos: p0.to_array() });
            lines.push(LineVertex { pos: p1.to_array() });
        }

        // Draw crossbars / grate across the drain
        match drain.face {
            CubeFace::Floor | CubeFace::Ceiling => {
                lines.push(LineVertex { pos: (pos - Vec3::X * radius).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::X * radius).to_array() });
                lines.push(LineVertex { pos: (pos - Vec3::Z * radius).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::Z * radius).to_array() });
            }
            CubeFace::Left | CubeFace::Right => {
                lines.push(LineVertex { pos: (pos - Vec3::Y * radius).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::Y * radius).to_array() });
                lines.push(LineVertex { pos: (pos - Vec3::Z * radius).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::Z * radius).to_array() });
            }
            CubeFace::Back | CubeFace::Front | CubeFace::Custom => {
                lines.push(LineVertex { pos: (pos - Vec3::X * radius).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::X * radius).to_array() });
                lines.push(LineVertex { pos: (pos - Vec3::Y * radius).to_array() });
                lines.push(LineVertex { pos: (pos + Vec3::Y * radius).to_array() });
            }
        }
    }

    pub fn destroy(&self, gl: &Arc<glow::Context>) {
        unsafe {
            gl.delete_program(self.particle_program);
            gl.delete_program(self.line_program);
            gl.delete_vertex_array(self.particle_vao);
            gl.delete_buffer(self.particle_vbo);
            gl.delete_vertex_array(self.line_vao);
            gl.delete_buffer(self.line_vbo);
        }
    }
}

unsafe fn create_shader_program(
    gl: &Arc<glow::Context>,
    vs_src: &str,
    fs_src: &str,
) -> Result<NativeProgram, String> {
    let vs = gl.create_shader(glow::VERTEX_SHADER)?;
    gl.shader_source(vs, vs_src);
    gl.compile_shader(vs);
    if !gl.get_shader_compile_status(vs) {
        let err = gl.get_shader_info_log(vs);
        gl.delete_shader(vs);
        return Err(format!("Vertex shader compilation failed: {}", err));
    }

    let fs = gl.create_shader(glow::FRAGMENT_SHADER)?;
    gl.shader_source(fs, fs_src);
    gl.compile_shader(fs);
    if !gl.get_shader_compile_status(fs) {
        let err = gl.get_shader_info_log(fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        return Err(format!("Fragment shader compilation failed: {}", err));
    }

    let program = gl.create_program()?;
    gl.attach_shader(program, vs);
    gl.attach_shader(program, fs);
    gl.link_program(program);

    gl.delete_shader(vs);
    gl.delete_shader(fs);

    if !gl.get_program_link_status(program) {
        let err = gl.get_program_info_log(program);
        gl.delete_program(program);
        return Err(format!("Shader program linking failed: {}", err));
    }

    Ok(program)
}
