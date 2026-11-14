use crate::math::Aabb;
use crate::render::surface_mesh::{EnvironmentPreset, MeshVertex, SurfaceMesh, SurfaceMeshConfig};
use crate::sph::collider::{Collider, ColliderShape};
use crate::sph::faucet::{CubeFace, Drain, Faucet, FaucetShape};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
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
    mesh_program: NativeProgram,
    particle_vao: NativeVertexArray,
    particle_vbo: NativeBuffer,
    line_vao: NativeVertexArray,
    line_vbo: NativeBuffer,
    mesh_vao: NativeVertexArray,
    mesh_vbo: NativeBuffer,
    mesh_ibo: NativeBuffer,
    line_scratch: Vec<LineVertex>,
    floor_scratch: Vec<LineVertex>,
    pub show_wireframe: bool,
    pub show_grid: bool,
    pub exposure: f32,
    pub env_preset: EnvironmentPreset,
    pub particle_light_dir: Vec3,
    pub particle_light_intensity: f32,
    pub particle_ambient_intensity: f32,
}

impl GlRenderer {
    pub fn new(gl: &Arc<glow::Context>) -> Result<Self, String> {
        unsafe {
            // 1. Compile glossy wet particle shader
            let p_vs_src = r#"#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec4 a_color;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_point_radius;
uniform float u_viewport_height;
uniform float u_tan_half_fov;

out vec4 v_color;
out vec3 v_pos_eye;

void main() {
    v_color = a_color;
    vec4 eye_pos = u_view * vec4(a_pos, 1.0);
    v_pos_eye = eye_pos.xyz;
    gl_Position = u_proj * eye_pos;
    float dist = max(-eye_pos.z, 0.001);
    float size_px = (u_point_radius / (dist * u_tan_half_fov)) * u_viewport_height;
    gl_PointSize = clamp(size_px, 2.0, 128.0);
}
"#;

            let p_fs_src = r#"#version 330 core
in vec4 v_color;
in vec3 v_pos_eye;

uniform mat4 u_proj;
uniform float u_point_radius;
uniform vec3 u_light_dir;
uniform vec3 u_light_params_p; // x: light_intensity, y: ambient_intensity, z: exposure
uniform vec3 u_rim_sky_col;
uniform int u_lit_spheres;
uniform int u_is_2d;

out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }

    float z = sqrt(max(1.0 - r2, 0.0));
    vec3 normal = vec3(coord.x, -coord.y, z);

    if (u_is_2d == 0) {
        vec3 sphere_pos_eye = v_pos_eye + vec3(coord.x * u_point_radius, -coord.y * u_point_radius, z * u_point_radius);
        vec4 clip_pos = u_proj * vec4(sphere_pos_eye, 1.0);
        gl_FragDepth = clamp(((clip_pos.z / clip_pos.w) + 1.0) * 0.5, 0.0, 1.0);
    }

    float exposure = max(u_light_params_p.z, 0.05);
    if (u_lit_spheres != 0) {
        vec3 L = normalize(u_light_dir);
        float diff = max(dot(normal, L), 0.0);
        vec3 view_dir = vec3(0.0, 0.0, 1.0);
        vec3 half_dir = normalize(L + view_dir);
        float spec = pow(max(dot(normal, half_dir), 0.0), 48.0) * u_light_params_p.x * 0.38;
        float fresnel = pow(1.0 - max(normal.z, 0.0), 3.2);
        vec3 rim_sky = u_rim_sky_col * fresnel * 0.30;

        float amb = u_light_params_p.y * 1.15;
        vec3 rgb = (v_color.rgb * (amb + 0.50 * u_light_params_p.x * diff) + vec3(0.92, 0.96, 1.0) * spec + rim_sky) * exposure;
        rgb = rgb / (vec3(1.0) + rgb * 0.14);
        FragColor = vec4(rgb, 1.0);
    } else {
        FragColor = vec4(v_color.rgb * exposure, 1.0);
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

            // 3. Compile physically-based 3D Surface Mesh shader
            let m_vs_src = r#"#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_speed_foam;

uniform mat4 u_view_proj;

out vec3 v_world_pos;
out vec3 v_normal;
out float v_speed;
out float v_foam;

void main() {
    v_world_pos = a_pos;
    v_normal = a_normal;
    v_speed = a_speed_foam.x;
    v_foam = a_speed_foam.y;
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}
"#;

            let m_fs_src = r#"#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in float v_speed;
in float v_foam;

uniform vec3 u_eye_pos;
uniform vec3 u_shallow_color;
uniform vec3 u_deep_color;
uniform vec4 u_mat_params;   // x: opacity, y: absorption, z: foam_intensity, w: metallic
uniform vec4 u_light_params; // x: ior, y: roughness, z: specular_intensity, w: sss
uniform vec4 u_env_params;   // x: env_reflection, y: refraction_strength, z: light_intensity, w: ambient_intensity
uniform vec3 u_light_dir;
uniform vec4 u_tex_params;   // x: mode (int), y: scale, z: strength, w: time
uniform vec3 u_sky_horizon;
uniform vec3 u_sky_zenith;
uniform vec3 u_sky_ground;
uniform float u_exposure;
uniform int u_backface_pass;

out vec4 FragColor;

vec3 hash3(ivec3 p) {
    uvec3 q = uvec3(p) * uvec3(1597334677u, 3812015801u, 2798796415u);
    uint n = (q.x ^ q.y ^ q.z) * 0x27d4eb2du;
    float r1 = float(n & 0xFFFFu) / 65535.0;
    n = ((n >> 11u) | (n << 21u)) * 0x165667b1u;
    float r2 = float(n & 0xFFFFu) / 65535.0;
    n = ((n >> 13u) | (n << 19u)) * 0x27d4eb2du;
    float r3 = float(n & 0xFFFFu) / 65535.0;
    return vec3(r1, r2, r3);
}

float value_noise(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float c000 = hash3(i + ivec3(0,0,0)).x;
    float c100 = hash3(i + ivec3(1,0,0)).x;
    float c010 = hash3(i + ivec3(0,1,0)).x;
    float c110 = hash3(i + ivec3(1,1,0)).x;
    float c001 = hash3(i + ivec3(0,0,1)).x;
    float c101 = hash3(i + ivec3(1,0,1)).x;
    float c011 = hash3(i + ivec3(0,1,1)).x;
    float c111 = hash3(i + ivec3(1,1,1)).x;
    float x00 = mix(c000, c100, u.x);
    float x10 = mix(c010, c110, u.x);
    float x01 = mix(c001, c101, u.x);
    float x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

float fbm(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int k = 0; k < 4; k++) {
        s += a * value_noise(p);
        p *= 2.03;
        a *= 0.5;
    }
    return s;
}

float voronoi_ridge(vec3 p) {
    ivec3 b = ivec3(floor(p));
    vec3 f = fract(p);
    float d1 = 8.0;
    float d2 = 8.0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ivec3 o = ivec3(dx, dy, dz);
                vec3 r = vec3(o) + hash3(b + o) - f;
                float d = dot(r, r);
                if (d < d1) { d2 = d1; d1 = d; }
                else if (d < d2) { d2 = d; }
            }
        }
    }
    return 1.0 - clamp((sqrt(d2) - sqrt(d1)) * 2.2, 0.0, 1.0);
}

float grid_line(vec2 uv) {
    vec2 f = abs(fract(uv) - 0.5);
    float d = min(0.5 - f.x, 0.5 - f.y);
    return 1.0 - clamp(d / 0.045, 0.0, 1.0);
}

vec3 sample_studio_sky(vec3 dir, float rough, float metallic) {
    vec3 r = normalize(dir);
    float y = clamp(r.y, -1.0, 1.0);
    float sharp = clamp(1.0 - rough, 0.05, 1.0);

    vec3 horizon = u_sky_horizon;
    vec3 zenith = u_sky_zenith;
    vec3 ground = u_sky_ground;

    // Neutralize excessive environmental saturation for silver liquid metals
    // so mercury reflects crisp metallic silver/graphite with subtle ambient tinting.
    float env_desat = metallic * 0.72;
    vec3 luma_w = vec3(0.2126, 0.7152, 0.0722);
    horizon = mix(horizon, vec3(dot(horizon, luma_w)), env_desat);
    zenith = mix(zenith, vec3(dot(zenith, luma_w)), env_desat);
    ground = mix(ground, vec3(dot(ground, luma_w)), env_desat);

    // Smooth C1 transition across y=0 (eliminates artificial contour steps on bevels)
    vec3 upper_raw = mix(horizon, zenith, pow(max(y, 0.0), 0.58));
    // Give liquid metals a luminous studio cyclorama sky dome overhead so wave tops
    // gleam as true silver/quicksilver rather than crushing to black in dark themes.
    vec3 metal_sky_dome = mix(vec3(0.26, 0.28, 0.31), vec3(0.14, 0.16, 0.19), pow(max(y, 0.0), 0.70));
    vec3 upper_sky = mix(upper_raw, max(upper_raw, metal_sky_dome), metallic);

    // Lower hemisphere reflects a silky-smooth studio cyclorama sweep floor (no high-frequency
    // binary edges, preventing moire contours on curved bevels and flat container walls)
    vec3 floor_dark = mix(horizon * 0.35, ground * mix(1.0, 0.55, metallic), 1.0 - smoothstep(-0.22, 0.04, y));
    float up_mix = smoothstep(-0.08, 0.12, y);
    vec3 sky = mix(floor_dark, upper_sky, up_mix);

    // Smooth studio horizon rim line + dark studio negative-fill band above horizon for chrome S-curve contrast
    float horizon_width = mix(0.15, 0.065, metallic);
    float horizon_band = exp(-pow(y / horizon_width, 2.0));
    float neg_fill = 1.0 - (0.30 + 0.40 * metallic) * exp(-pow((y - 0.25) / 0.16, 2.0));
    sky = sky * neg_fill
        + mix(vec3(0.20, 0.22, 0.25), vec3(0.58, 0.61, 0.66), metallic) * horizon_band * (0.28 + 0.45 * metallic);

    // Broad, smooth directional studio floor bounce on the lower hemisphere so vertical walls
    // show natural studio depth and tonal gradation across left/right wall angles
    if (y < 0.0) {
        vec3 r_xz = normalize(vec3(r.x, 0.0, r.z) + vec3(1e-5, 0.0, 0.0));
        float key_bounce = 0.5 + 0.5 * dot(r_xz, normalize(vec3(u_light_dir.x, 0.0, u_light_dir.z) + vec3(1e-5, 0.0, 0.0)));
        float floor_sheen = mix(0.025, 0.095, key_bounce) * exp(-pow((y + 0.32) / 0.38, 2.0));
        float rim_floor = 0.055 * pow(clamp(1.0 + y, 0.0, 1.0), 3.0) * (0.5 + 0.5 * r_xz.z);
        sky += vec3(0.88, 0.92, 0.98) * (floor_sheen + rim_floor) * (0.20 + 0.75 * metallic);
    }

    // Overhead studio softbox panel + secondary diffused fill panel for crisp silver highlights
    vec3 sb_dir = normalize(vec3(-0.22, 0.84, 0.34));
    vec3 sb_u = normalize(cross(vec3(0.0, 0.0, 1.0), sb_dir));
    vec3 sb_v = normalize(cross(sb_dir, sb_u));
    float su = abs(dot(r, sb_u));
    float sv = abs(dot(r, sb_v));
    float sf = max(dot(r, sb_dir), 0.0);
    float feather = 0.05 + (1.0 - sharp) * 0.30;
    float softbox_rect = (1.0 - smoothstep(0.36, 0.36 + feather, su))
                       * (1.0 - smoothstep(0.24, 0.24 + feather, sv))
                       * smoothstep(0.2, 0.55, sf);
    vec3 softbox_col = mix(vec3(0.85, 0.91, 1.0), vec3(1.15, 1.18, 1.22), metallic);
    sky += softbox_col * softbox_rect * (0.85 + 0.55 * metallic);

    // Narrow studio kicker/rim strips positioned at side-back grazing angles
    // so curved bevels & droplets catch bright silver edges without washing out flat front walls
    float strip_y = exp(-pow((y - 0.14) / 0.26, 2.0));
    float strip_pow = mix(4.0, 16.0, metallic) + sharp * mix(10.0, 42.0, metallic);
    vec3 r_horiz = normalize(vec3(r.x, 0.0, r.z) + vec3(1e-5, 0.0, 0.0));
    float strip_l = pow(clamp(dot(r_horiz, normalize(vec3(-0.96, 0.0, -0.28))), 0.0, 1.0), strip_pow);
    float strip_r = pow(clamp(dot(r_horiz, normalize(vec3(0.95, 0.0, -0.31))), 0.0, 1.0), strip_pow);
    float strip_front = pow(clamp(dot(r_horiz, normalize(vec3(-0.25, 0.0, 0.97))), 0.0, 1.0), strip_pow * 1.4);
    sky += mix(vec3(0.55, 0.68, 0.85), vec3(0.95, 0.98, 1.02), metallic) * strip_y * (strip_l + 0.80 * strip_r + 0.55 * strip_front) * (0.35 + 0.70 * metallic);

    // Crisp sun/key light bloom in sky
    float sun_dot = max(dot(r, u_light_dir), 0.0);
    float sun_spot = pow(sun_dot, 12.0 + sharp * 72.0);
    vec3 sun_col = mix(vec3(1.25, 1.18, 1.05), vec3(1.45, 1.44, 1.42), metallic);
    return sky + sun_col * sun_spot;
}

void main() {
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    vec3 V = normalize(u_eye_pos - v_world_pos);
    vec3 L = normalize(u_light_dir);

    int tex_mode = int(u_tex_params.x + 0.5);
    float tex_scale = max(u_tex_params.y, 0.2);
    float tex_strength = clamp(u_tex_params.z, 0.0, 1.0);
    float t_anim = u_tex_params.w;

    vec3 albedo_mod = vec3(1.0);
    float caustic_glow = 0.0;
    float foam_extra = 0.0;

    if (tex_mode > 0 && tex_strength > 0.001) {
        vec3 p = v_world_pos * tex_scale;
        if (tex_mode == 1) {
            // Capillary Ripples
            float t = t_anim * 3.2;
            vec3 k1 = vec3(3.1, 0.7, 2.4);
            vec3 k2 = vec3(-2.3, 1.1, 3.3);
            vec3 k3 = vec3(1.5, -2.9, -2.1);
            float ph1 = dot(p, k1) - t * 1.3;
            float ph2 = dot(p, k2) + t * 1.7;
            float ph3 = dot(p, k3 * 1.6) - t * 2.3;
            vec3 grad = k1 * cos(ph1) * 0.45 + k2 * cos(ph2) * 0.35 + k3 * cos(ph3) * 0.20;
            vec3 tang = grad - N * dot(grad, N);
            N = normalize(N + tang * (tex_strength * 0.22));
            albedo_mod = vec3(1.0 + sin(ph1) * sin(ph2) * tex_strength * 0.08);
        } else if (tex_mode == 2) {
            // Voronoi Caustics
            vec3 q = p * 0.85 + vec3(t_anim * 0.55, -t_anim * 0.35, t_anim * 0.45);
            float c1 = voronoi_ridge(q);
            float c2 = voronoi_ridge(q * 1.7 + vec3(3.7));
            float caustic = pow(c1 * 0.65 + c2 * 0.35, 2.2) * tex_strength;
            float cx = voronoi_ridge(q + vec3(0.06, 0.0, 0.0)) - c1;
            float cz = voronoi_ridge(q + vec3(0.0, 0.0, 0.06)) - c1;
            vec3 bump = vec3(cx, 0.0, cz) * (tex_strength * 1.8);
            N = normalize(N + (bump - N * dot(bump, N)));
            albedo_mod = vec3(1.0) + vec3(0.25, 0.45, 0.55) * caustic;
            caustic_glow = caustic * 0.55;
        } else if (tex_mode == 3) {
            // FBM Foam Streaks
            vec3 q = p * 0.9 + vec3(t_anim * 0.3, -t_anim * 0.2, t_anim * 0.25);
            float f = fbm(q);
            float sf = clamp(v_speed / 2.2, 0.15, 1.0);
            foam_extra = clamp((f - 0.42) * 2.4, 0.0, 1.0) * tex_strength * sf * 0.85;
            vec3 bump = vec3(f - 0.5, 0.0, fbm(q + vec3(0.0, 0.0, 1.3)) - 0.5) * (tex_strength * 0.30);
            N = normalize(N + (bump - N * dot(bump, N)));
        } else if (tex_mode == 4) {
            // Studio UV Grid
            vec3 w = pow(abs(N), vec3(4.0));
            w /= max(w.x + w.y + w.z, 1e-5);
            float line = grid_line(p.yz) * w.x + grid_line(p.xz) * w.y + grid_line(p.xy) * w.z;
            int chk = (int(floor(p.x)) + int(floor(p.y)) + int(floor(p.z))) & 1;
            float base_m = 1.0 - float(chk) * 0.14 * tex_strength;
            float lb = line * tex_strength * 0.65;
            albedo_mod = vec3(base_m + lb);
            caustic_glow = lb * 0.25;
            foam_extra = lb * 0.35;
        }
    }

    float ndotv = max(dot(N, V), 0.001);
    float ndotl = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float ndoth = max(dot(N, H), 0.0);

    vec3 shallow_lin = pow(max(u_shallow_color, vec3(0.0)), vec3(2.2));
    vec3 deep_lin = pow(max(u_deep_color, vec3(0.0)), vec3(2.2));

    float ior = max(u_light_params.x, 1.0);
    float f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    float metallic = clamp(u_mat_params.w, 0.0, 1.0);
    vec3 f0 = mix(vec3(f0_dielectric), shallow_lin * albedo_mod, metallic);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, 5.0);

    // Beer-Lambert optical thickness approximation from grazing angle & height
    float optical_depth = clamp((1.0 - ndotv * 0.65) * u_mat_params.y * 0.55, 0.0, 1.0);
    vec3 body_color = mix(shallow_lin, deep_lin, optical_depth) * albedo_mod;

    if (u_backface_pass != 0) {
        if (metallic >= 0.85 || u_mat_params.x >= 0.99) {
            discard;
        }
        // Subtle interior volume tint pass
        float back_alpha = clamp(u_mat_params.x * 0.42, 0.05, 0.65);
        vec3 back_col = deep_lin * (u_env_params.w + 0.35 * ndotl);
        back_col = pow(clamp(back_col, 0.0, 1.0), vec3(1.0 / 2.2));
        FragColor = vec4(back_col, back_alpha);
        return;
    }

    // Smith GGX Microfacet Specular
    float rough = clamp(u_light_params.y, 0.02, 1.0);
    float alpha = rough * rough;
    float alpha2 = alpha * alpha;
    float denom = (ndoth * ndoth * (alpha2 - 1.0) + 1.0);
    float d_ggx = alpha2 / max(3.14159265 * denom * denom, 1e-6);
    float k_ggx = (rough + 1.0) * (rough + 1.0) * 0.125;
    float g_v = ndotv / (ndotv * (1.0 - k_ggx) + k_ggx);
    float g_l = ndotl / (ndotl * (1.0 - k_ggx) + k_ggx);
    vec3 spec_brdf = (d_ggx * g_v * g_l * fresnel) / max(4.0 * ndotv * ndotl, 0.05);
    vec3 spec = spec_brdf * ndotl * u_light_params.z * u_env_params.z;

    // 360-degree Studio HDRI reflection
    vec3 R = reflect(-V, N);
    vec3 env_col = sample_studio_sky(R, rough, metallic) * u_env_params.x;

    // Subsurface back-scattered wave crest glow (dielectrics only)
    float back_light = pow(clamp(dot(V, -L + N * 0.55), 0.0, 1.0), 3.0);
    vec3 sss = shallow_lin * back_light * u_light_params.w * u_env_params.z * (1.0 - metallic);

    // Diffuse + Ambient (zeroed for pure conductors; faint multi-scatter cavity fill keeps chrome lush)
    float amb = u_env_params.w;
    float diff = ndotl * u_env_params.z;
    float hemi = 0.5 + 0.5 * N.y;
    vec3 fluid_lit = body_color * (amb + diff) * (1.0 - metallic);
    vec3 metal_cavity_bounce = f0 * (0.005 + 0.010 * hemi) * amb * metallic;

    // Blend transmission body with physical Fresnel reflection + GGX sun highlight
    vec3 color = fluid_lit * (vec3(1.0) - fresnel)
        + env_col * fresnel
        + metal_cavity_bounce
        + spec
        + sss
        + shallow_lin * caustic_glow * (1.0 - metallic * 0.65);

    // Aerated whitewater foam on energetic crests
    float foam = clamp(v_foam * u_mat_params.z + foam_extra, 0.0, 1.0);
    vec3 foam_col = vec3(0.88, 0.93, 1.0) * (amb + diff * 0.85 + 0.25);
    color = mix(color, foam_col, foam);

    // Exposure + ACES Filmic Tone Mapping + Linear-to-sRGB output encoding
    color *= max(u_exposure, 0.05);
    color = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    color = pow(color, vec3(1.0 / 2.2));

    float out_alpha = metallic >= 0.85
        ? 1.0
        : clamp(mix(u_mat_params.x, 1.0, max(fresnel.r, foam)), 0.12, 1.0);
    FragColor = vec4(color, out_alpha);
}
"#;

            let mesh_program = create_shader_program(gl, m_vs_src, m_fs_src)?;

            // 4. Create Particle VAO & VBO
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

            // 5. Create Line VAO & VBO
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

            // 6. Create Surface Mesh VAO, VBO & IBO
            let mesh_vao = gl.create_vertex_array()?;
            let mesh_vbo = gl.create_buffer()?;
            let mesh_ibo = gl.create_buffer()?;

            gl.bind_vertex_array(Some(mesh_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(mesh_vbo));
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(mesh_ibo));

            let m_stride = std::mem::size_of::<MeshVertex>() as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, m_stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(
                1,
                3,
                glow::FLOAT,
                false,
                m_stride,
                (3 * std::mem::size_of::<f32>()) as i32,
            );
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(
                2,
                2,
                glow::FLOAT,
                false,
                m_stride,
                (6 * std::mem::size_of::<f32>()) as i32,
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);

            Ok(Self {
                particle_program,
                line_program,
                mesh_program,
                particle_vao,
                particle_vbo,
                line_vao,
                line_vbo,
                mesh_vao,
                mesh_vbo,
                mesh_ibo,
                line_scratch: Vec::with_capacity(512),
                floor_scratch: Vec::with_capacity(128),
                show_wireframe: true,
                show_grid: true,
                exposure: 1.0,
                env_preset: EnvironmentPreset::StudioDark,
                particle_light_dir: Vec3::new(0.42, 0.82, 0.38).normalize(),
                particle_light_intensity: 1.25,
                particle_ambient_intensity: 0.32,
            })
        }
    }

    unsafe fn draw_scene_wireframes(
        &mut self,
        gl: &Arc<glow::Context>,
        view_proj: Mat4,
        chamber: &crate::sph::chamber::Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        faucets: &[Faucet],
        drains: &[Drain],
        is_2d: bool,
    ) {
        // Subtle 3D studio reference floor grid beneath the chamber (scales dynamically with bounds)
        if !is_2d && self.show_grid {
            self.floor_scratch.clear();
            let y_floor = bounds.min.y - 0.002;
            let b_size = bounds.size();
            let grid_half = (b_size.x.max(b_size.z) * 0.85 + 0.8).max(1.6);
            let steps = ((grid_half * 2.0 / 0.25).round() as usize).clamp(16, 48);
            let step = (grid_half * 2.0) / steps as f32;
            for i in 0..=steps {
                let v = -grid_half + i as f32 * step;
                self.floor_scratch.push(LineVertex {
                    pos: [v, y_floor, -grid_half],
                });
                self.floor_scratch.push(LineVertex {
                    pos: [v, y_floor, grid_half],
                });
                self.floor_scratch.push(LineVertex {
                    pos: [-grid_half, y_floor, v],
                });
                self.floor_scratch.push(LineVertex {
                    pos: [grid_half, y_floor, v],
                });
            }

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.use_program(Some(self.line_program));
            let u_vp = gl.get_uniform_location(self.line_program, "u_view_proj");
            let u_col = gl.get_uniform_location(self.line_program, "u_line_color");
            gl.uniform_matrix_4_f32_slice(u_vp.as_ref(), false, &view_proj.to_cols_array());
            gl.uniform_4_f32(u_col.as_ref(), 0.18, 0.25, 0.36, 0.45);

            gl.bind_vertex_array(Some(self.line_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.line_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&self.floor_scratch),
                glow::DYNAMIC_DRAW,
            );
            gl.line_width(1.0);
            gl.draw_arrays(glow::LINES, 0, self.floor_scratch.len() as i32);
        }

        if !self.show_wireframe {
            return;
        }

        self.line_scratch.clear();
        let chamber_wireframe = if is_2d {
            chamber.generate_wireframe_lines_2d(&bounds)
        } else {
            chamber.generate_wireframe_lines(&bounds)
        };
        for (p0, p1) in chamber_wireframe {
            self.line_scratch.push(LineVertex { pos: p0.to_array() });
            self.line_scratch.push(LineVertex { pos: p1.to_array() });
        }

        for collider in colliders {
            if collider.enabled {
                if is_2d {
                    match &collider.shape {
                        ColliderShape::Sphere { center, radius } => {
                            Self::generate_circle_lines_xy(
                                *center,
                                *radius,
                                36,
                                &mut self.line_scratch,
                            );
                        }
                        ColliderShape::Box { aabb } => {
                            let p0 = Vec3::new(aabb.min.x, aabb.min.y, 0.0);
                            let p1 = Vec3::new(aabb.max.x, aabb.min.y, 0.0);
                            let p2 = Vec3::new(aabb.max.x, aabb.max.y, 0.0);
                            let p3 = Vec3::new(aabb.min.x, aabb.max.y, 0.0);
                            self.line_scratch.push(LineVertex { pos: p0.to_array() });
                            self.line_scratch.push(LineVertex { pos: p1.to_array() });
                            self.line_scratch.push(LineVertex { pos: p1.to_array() });
                            self.line_scratch.push(LineVertex { pos: p2.to_array() });
                            self.line_scratch.push(LineVertex { pos: p2.to_array() });
                            self.line_scratch.push(LineVertex { pos: p3.to_array() });
                            self.line_scratch.push(LineVertex { pos: p3.to_array() });
                            self.line_scratch.push(LineVertex { pos: p0.to_array() });
                        }
                        ColliderShape::Cylinder { center, radius, .. } => {
                            Self::generate_circle_lines_xy(
                                *center,
                                *radius,
                                36,
                                &mut self.line_scratch,
                            );
                        }
                        ColliderShape::Mesh(mesh) => {
                            Self::generate_mesh_lines(mesh, &mut self.line_scratch);
                        }
                    }
                } else {
                    match &collider.shape {
                        ColliderShape::Sphere { center, radius } => {
                            Self::generate_sphere_lines(
                                *center,
                                *radius,
                                28,
                                &mut self.line_scratch,
                            );
                        }
                        ColliderShape::Box { aabb } => {
                            Self::generate_box_lines(aabb, &mut self.line_scratch);
                        }
                        ColliderShape::Cylinder {
                            center,
                            radius,
                            height,
                        } => {
                            Self::generate_cylinder_lines(
                                *center,
                                *radius,
                                *height,
                                20,
                                &mut self.line_scratch,
                            );
                        }
                        ColliderShape::Mesh(mesh) => {
                            Self::generate_mesh_lines(mesh, &mut self.line_scratch);
                        }
                    }
                }
            }
        }

        for faucet in faucets {
            if faucet.enabled {
                Self::generate_faucet_lines(faucet, &mut self.line_scratch);
            }
        }

        for drain in drains {
            if drain.enabled {
                Self::generate_drain_lines(drain, &bounds, &mut self.line_scratch);
            }
        }

        if !self.line_scratch.is_empty() {
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.use_program(Some(self.line_program));

            let u_vp = gl.get_uniform_location(self.line_program, "u_view_proj");
            let u_col = gl.get_uniform_location(self.line_program, "u_line_color");

            gl.uniform_matrix_4_f32_slice(u_vp.as_ref(), false, &view_proj.to_cols_array());
            gl.uniform_4_f32(u_col.as_ref(), 0.34, 0.68, 0.96, 0.78);

            gl.bind_vertex_array(Some(self.line_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.line_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&self.line_scratch),
                glow::DYNAMIC_DRAW,
            );

            gl.line_width(1.5);
            gl.draw_arrays(glow::LINES, 0, self.line_scratch.len() as i32);
        }
    }

    /// Render particles and simulation bounds
    pub fn render(
        &mut self,
        gl: &Arc<glow::Context>,
        view: Mat4,
        proj: Mat4,
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
                gl.clear_depth_f32(1.0);
                gl.clear(glow::DEPTH_BUFFER_BIT);
            }
            gl.enable(glow::PROGRAM_POINT_SIZE);

            let view_proj = proj * view;

            // 1. Draw Studio Grid, Chamber, Colliders, Faucets, and Drains
            self.draw_scene_wireframes(
                gl, view_proj, chamber, bounds, colliders, faucets, drains, is_2d,
            );

            gl.disable(glow::BLEND);

            // 2. Draw Particles
            if !particles_data.is_empty() {
                gl.use_program(Some(self.particle_program));

                let u_view = gl.get_uniform_location(self.particle_program, "u_view");
                let u_proj = gl.get_uniform_location(self.particle_program, "u_proj");
                let u_pr = gl.get_uniform_location(self.particle_program, "u_point_radius");
                let u_vh = gl.get_uniform_location(self.particle_program, "u_viewport_height");
                let u_tf = gl.get_uniform_location(self.particle_program, "u_tan_half_fov");
                let u_ld = gl.get_uniform_location(self.particle_program, "u_light_dir");
                let u_lpp = gl.get_uniform_location(self.particle_program, "u_light_params_p");
                let u_rim = gl.get_uniform_location(self.particle_program, "u_rim_sky_col");
                let u_lit = gl.get_uniform_location(self.particle_program, "u_lit_spheres");
                let u_is_2d = gl.get_uniform_location(self.particle_program, "u_is_2d");

                let tan_half_fov = (fov_y * 0.5).tan();

                gl.uniform_matrix_4_f32_slice(u_view.as_ref(), false, &view.to_cols_array());
                gl.uniform_matrix_4_f32_slice(u_proj.as_ref(), false, &proj.to_cols_array());
                gl.uniform_1_f32(u_pr.as_ref(), particle_radius);
                gl.uniform_1_f32(u_vh.as_ref(), viewport_size[1]);
                gl.uniform_1_f32(u_tf.as_ref(), tan_half_fov);

                // Transform world-space sun direction into view/eye space so orbiting camera keeps lighting world-anchored
                let world_ld = if self.particle_light_dir.length_squared() > 0.25 {
                    self.particle_light_dir.normalize()
                } else {
                    Vec3::new(0.42, 0.82, 0.38).normalize()
                };
                let eye_ld = (Mat3::from_mat4(view) * world_ld).normalize_or_zero();
                let light_dir = if eye_ld.length_squared() > 0.25 {
                    eye_ld
                } else {
                    Vec3::new(0.42, 0.82, 0.38).normalize()
                };
                let (sky_hor, _, _) = self.env_preset.sky_reflection_linear();
                let rim_srgb = sky_hor.clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2);

                gl.uniform_3_f32(u_ld.as_ref(), light_dir.x, light_dir.y, light_dir.z);
                gl.uniform_3_f32(
                    u_lpp.as_ref(),
                    self.particle_light_intensity,
                    self.particle_ambient_intensity,
                    self.exposure,
                );
                gl.uniform_3_f32(u_rim.as_ref(), rim_srgb.x, rim_srgb.y, rim_srgb.z);
                gl.uniform_1_i32(u_lit.as_ref(), if lit_spheres { 1 } else { 0 });
                gl.uniform_1_i32(u_is_2d.as_ref(), if is_2d { 1 } else { 0 });

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

    /// Render reconstructed 3D liquid surface mesh in real time with configured texture, color, and lighting.
    pub fn render_surface_mesh(
        &mut self,
        gl: &Arc<glow::Context>,
        view: Mat4,
        proj: Mat4,
        eye_pos: Vec3,
        mesh: &SurfaceMesh,
        mesh_cfg: &SurfaceMeshConfig,
        anim_time: f32,
        chamber: &crate::sph::chamber::Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        faucets: &[Faucet],
        drains: &[Drain],
        is_2d: bool,
    ) {
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);

            let view_proj = proj * view;
            self.show_wireframe = mesh_cfg.show_wireframe;
            self.show_grid = mesh_cfg.show_grid;
            self.exposure = mesh_cfg.exposure;
            self.env_preset = mesh_cfg.env_preset;

            // 1. Draw Studio Grid, Chamber, Colliders, Faucets, and Drains first so translucent liquid blends over them
            self.draw_scene_wireframes(
                gl, view_proj, chamber, bounds, colliders, faucets, drains, is_2d,
            );

            // 2. Draw Surface Mesh (two-pass back/front translucent optical shading)
            if !mesh.is_empty() {
                gl.use_program(Some(self.mesh_program));

                let u_vp = gl.get_uniform_location(self.mesh_program, "u_view_proj");
                let u_eye = gl.get_uniform_location(self.mesh_program, "u_eye_pos");
                let u_shallow = gl.get_uniform_location(self.mesh_program, "u_shallow_color");
                let u_deep = gl.get_uniform_location(self.mesh_program, "u_deep_color");
                let u_mat = gl.get_uniform_location(self.mesh_program, "u_mat_params");
                let u_lgt = gl.get_uniform_location(self.mesh_program, "u_light_params");
                let u_env = gl.get_uniform_location(self.mesh_program, "u_env_params");
                let u_ldir = gl.get_uniform_location(self.mesh_program, "u_light_dir");
                let u_tex = gl.get_uniform_location(self.mesh_program, "u_tex_params");
                let u_sky_h = gl.get_uniform_location(self.mesh_program, "u_sky_horizon");
                let u_sky_z = gl.get_uniform_location(self.mesh_program, "u_sky_zenith");
                let u_sky_g = gl.get_uniform_location(self.mesh_program, "u_sky_ground");
                let u_exp = gl.get_uniform_location(self.mesh_program, "u_exposure");
                let u_back = gl.get_uniform_location(self.mesh_program, "u_backface_pass");

                let ld = mesh_cfg.light_dir.normalize_or_zero();
                let ld = if ld.length_squared() > 0.5 {
                    ld
                } else {
                    Vec3::new(0.42, 0.82, 0.38).normalize()
                };
                let (sky_hor_l, sky_zen_l, sky_gnd_l) = mesh_cfg.env_preset.sky_reflection_linear();
                let sky_hor = sky_hor_l.max(Vec3::ZERO);
                let sky_zen = sky_zen_l.max(Vec3::ZERO);
                let sky_gnd = sky_gnd_l.max(Vec3::ZERO);

                gl.uniform_matrix_4_f32_slice(u_vp.as_ref(), false, &view_proj.to_cols_array());
                gl.uniform_3_f32(u_eye.as_ref(), eye_pos.x, eye_pos.y, eye_pos.z);
                gl.uniform_3_f32(
                    u_shallow.as_ref(),
                    mesh_cfg.shallow_color[0],
                    mesh_cfg.shallow_color[1],
                    mesh_cfg.shallow_color[2],
                );
                gl.uniform_3_f32(
                    u_deep.as_ref(),
                    mesh_cfg.deep_color[0],
                    mesh_cfg.deep_color[1],
                    mesh_cfg.deep_color[2],
                );
                gl.uniform_4_f32(
                    u_mat.as_ref(),
                    mesh_cfg.opacity,
                    mesh_cfg.absorption,
                    mesh_cfg.foam_intensity,
                    mesh_cfg.metallic,
                );
                gl.uniform_4_f32(
                    u_lgt.as_ref(),
                    mesh_cfg.ior,
                    mesh_cfg.roughness,
                    mesh_cfg.specular_intensity,
                    mesh_cfg.subsurface_scattering,
                );
                gl.uniform_4_f32(
                    u_env.as_ref(),
                    mesh_cfg.env_reflection,
                    mesh_cfg.refraction_strength,
                    mesh_cfg.light_intensity,
                    mesh_cfg.ambient_intensity,
                );
                gl.uniform_3_f32(u_ldir.as_ref(), ld.x, ld.y, ld.z);
                gl.uniform_4_f32(
                    u_tex.as_ref(),
                    mesh_cfg.texture_mode.gpu_id() as f32,
                    mesh_cfg.texture_scale,
                    mesh_cfg.texture_strength,
                    anim_time,
                );
                gl.uniform_3_f32(u_sky_h.as_ref(), sky_hor.x, sky_hor.y, sky_hor.z);
                gl.uniform_3_f32(u_sky_z.as_ref(), sky_zen.x, sky_zen.y, sky_zen.z);
                gl.uniform_3_f32(u_sky_g.as_ref(), sky_gnd.x, sky_gnd.y, sky_gnd.z);
                gl.uniform_1_f32(u_exp.as_ref(), mesh_cfg.exposure);

                gl.bind_vertex_array(Some(self.mesh_vao));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.mesh_vbo));
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&mesh.vertices),
                    glow::DYNAMIC_DRAW,
                );
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.mesh_ibo));
                gl.buffer_data_u8_slice(
                    glow::ELEMENT_ARRAY_BUFFER,
                    bytemuck::cast_slice(&mesh.indices),
                    glow::DYNAMIC_DRAW,
                );

                gl.enable(glow::BLEND);
                gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
                gl.enable(glow::CULL_FACE);

                // Pass A: Backfaces (interior volume absorption tint, depth write off)
                if mesh_cfg.opacity < 0.98 && mesh_cfg.metallic < 0.85 {
                    gl.depth_mask(false);
                    gl.cull_face(glow::FRONT);
                    gl.uniform_1_i32(u_back.as_ref(), 1);
                    gl.draw_elements(
                        glow::TRIANGLES,
                        mesh.indices.len() as i32,
                        glow::UNSIGNED_INT,
                        0,
                    );
                }

                // Pass B: Frontfaces (optical Fresnel + GGX specular + texture, depth write on)
                gl.depth_mask(true);
                gl.cull_face(glow::BACK);
                gl.uniform_1_i32(u_back.as_ref(), 0);
                gl.draw_elements(
                    glow::TRIANGLES,
                    mesh.indices.len() as i32,
                    glow::UNSIGNED_INT,
                    0,
                );

                gl.disable(glow::CULL_FACE);
            }

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            gl.use_program(None);

            // Restore state for egui
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(true);
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
            lines.push(LineVertex {
                pos: p0_top.to_array(),
            });
            lines.push(LineVertex {
                pos: p1_top.to_array(),
            });

            let p0_bot = bot_center + Vec3::new(t0.cos() * radius, 0.0, t0.sin() * radius);
            let p1_bot = bot_center + Vec3::new(t1.cos() * radius, 0.0, t1.sin() * radius);
            lines.push(LineVertex {
                pos: p0_bot.to_array(),
            });
            lines.push(LineVertex {
                pos: p1_bot.to_array(),
            });

            if i % (segments / 4).max(1) == 0 {
                lines.push(LineVertex {
                    pos: p0_top.to_array(),
                });
                lines.push(LineVertex {
                    pos: p0_bot.to_array(),
                });
            }
        }
    }

    fn generate_mesh_lines(mesh: &crate::sph::mesh::TriangleMesh, lines: &mut Vec<LineVertex>) {
        let rot_mat = glam::Mat3::from_euler(
            glam::EulerRot::YXZ,
            mesh.rotation.y,
            mesh.rotation.x,
            mesh.rotation.z,
        );
        let s = mesh.scale;
        let pos = mesh.position;

        lines.reserve(mesh.wireframe_segments.len() * 2);
        for &[a, b] in mesh.wireframe_segments.iter() {
            let v0 = pos + (rot_mat * (a * s));
            let v1 = pos + (rot_mat * (b * s));
            lines.push(LineVertex { pos: v0.to_array() });
            lines.push(LineVertex { pos: v1.to_array() });
        }
    }

    fn generate_faucet_lines(faucet: &Faucet, lines: &mut Vec<LineVertex>) {
        let pos = faucet.position;
        let flow_dir = if faucet.direction.length_squared() > 1e-6 {
            faucet.direction.normalize()
        } else {
            -Vec3::Y
        };

        match faucet.shape {
            FaucetShape::Circle { radius } => {
                let segments = 16;
                let d_theta = std::f32::consts::TAU / segments as f32;
                let up = if flow_dir.y.abs() < 0.95 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
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
                let up = if flow_dir.y.abs() < 0.95 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
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
                lines.push(LineVertex {
                    pos: (pos - Vec3::X * s).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::X * s).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos - Vec3::Z * s).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::Z * s).to_array(),
                });
            }
        }

        let arrow_len = 0.12;
        let arrow_tip = pos + flow_dir * arrow_len;
        lines.push(LineVertex {
            pos: pos.to_array(),
        });
        lines.push(LineVertex {
            pos: arrow_tip.to_array(),
        });
    }

    fn generate_drain_lines(drain: &Drain, bounds: &Aabb, lines: &mut Vec<LineVertex>) {
        let segments = 20;
        let d_theta = std::f32::consts::TAU / segments as f32;
        let radius = drain.radius;
        let pos = match drain.face {
            CubeFace::Floor => Vec3::new(drain.position.x, bounds.min.y + 0.001, drain.position.z),
            CubeFace::Ceiling => {
                Vec3::new(drain.position.x, bounds.max.y - 0.001, drain.position.z)
            }
            CubeFace::Left => Vec3::new(bounds.min.x + 0.001, drain.position.y, drain.position.z),
            CubeFace::Right => Vec3::new(bounds.max.x - 0.001, drain.position.y, drain.position.z),
            CubeFace::Back => Vec3::new(drain.position.x, drain.position.y, bounds.min.z + 0.001),
            CubeFace::Front => Vec3::new(drain.position.x, drain.position.y, bounds.max.z - 0.001),
            CubeFace::Custom => drain.position,
        };

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

        match drain.face {
            CubeFace::Floor | CubeFace::Ceiling => {
                lines.push(LineVertex {
                    pos: (pos - Vec3::X * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::X * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos - Vec3::Z * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::Z * radius).to_array(),
                });
            }
            CubeFace::Left | CubeFace::Right => {
                lines.push(LineVertex {
                    pos: (pos - Vec3::Y * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::Y * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos - Vec3::Z * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::Z * radius).to_array(),
                });
            }
            CubeFace::Back | CubeFace::Front | CubeFace::Custom => {
                lines.push(LineVertex {
                    pos: (pos - Vec3::X * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::X * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos - Vec3::Y * radius).to_array(),
                });
                lines.push(LineVertex {
                    pos: (pos + Vec3::Y * radius).to_array(),
                });
            }
        }
    }

    pub fn destroy(&self, gl: &Arc<glow::Context>) {
        unsafe {
            gl.delete_program(self.particle_program);
            gl.delete_program(self.line_program);
            gl.delete_program(self.mesh_program);
            gl.delete_vertex_array(self.particle_vao);
            gl.delete_buffer(self.particle_vbo);
            gl.delete_vertex_array(self.line_vao);
            gl.delete_buffer(self.line_vbo);
            gl.delete_vertex_array(self.mesh_vao);
            gl.delete_buffer(self.mesh_vbo);
            gl.delete_buffer(self.mesh_ibo);
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
