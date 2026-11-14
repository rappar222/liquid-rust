struct GpuColliderItem {
    min_or_center: vec4<f32>, // Sphere/Cyl: xyz=center, w=radius; Box: xyz=min
    max_or_ext: vec4<f32>,    // Box: xyz=max; Cyl: x=height
    kind_pad: vec4<u32>,      // x: 0=Sphere, 1=Box, 2=Cylinder
}

struct RenderUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    shadow_vp: mat4x4<f32>,

    eye_pos: vec4<f32>,       // xyz = eye_pos, w = tan_half_fov
    cam_fwd: vec4<f32>,       // xyz = cam_fwd, w = aspect
    cam_right: vec4<f32>,     // xyz = cam_right, w = near
    cam_up: vec4<f32>,        // xyz = cam_up, w = far

    light_dir: vec4<f32>,     // xyz = world light_dir, w = light_intensity
    eye_light_dir: vec4<f32>, // xyz = eye-space light_dir, w = ambient_intensity
    bounds_min: vec4<f32>,    // xyz = bounds.min, w = floor_y
    bounds_max: vec4<f32>,    // xyz = bounds.max, w = floor_margin

    bg_low: vec4<f32>,        // xyz = bg_low, w = exposure
    bg_high: vec4<f32>,       // xyz = bg_high, w = anim_time
    sky_horizon: vec4<f32>,   // xyz = sky_horizon, w = particle_radius
    sky_zenith: vec4<f32>,    // xyz = sky_zenith, w = rest_density
    sky_ground: vec4<f32>,    // xyz = sky_ground, w = dispersion

    shallow_lin: vec4<f32>,   // xyz = shallow_lin, w = opacity
    deep_lin: vec4<f32>,      // xyz = deep_lin, w = absorption
    shallow_srgb: vec4<f32>,  // xyz = shallow_srgb, w = ior
    deep_srgb: vec4<f32>,     // xyz = deep_srgb, w = roughness
    mat_opt1: vec4<f32>,      // x = spec_int, y = refr_str, z = sss, w = env_refl
    mat_opt2: vec4<f32>,      // x = metallic, y = foam_int, z = tex_scale, w = tex_str
    wire_color: vec4<f32>,    // xyz = wire_color_srgb, w = alpha_boost

    dims: vec4<u32>,          // x = rw, y = rh, z = out_w, w = out_h
    flags0: vec4<u32>,        // x = ssaa, y = shadow_res, z = num_colliders, w = is_2d
    flags1: vec4<u32>,        // x = show_grid, y = env_preset_id, z = tex_mode, w = lit_spheres
    flags2: vec4<u32>,        // x = color_mode, y = color_map, z = particle_foam, w = has_shadow_fluid
}

struct GpuParticle {
    pos_type: vec4<f32>,
    vel_dens: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var<storage, read> colliders: array<GpuColliderItem>;

// Pass-specific resources (unique binding slots per pass layout)
@group(1) @binding(0) var shadow_tex: texture_2d<f32>;
@group(1) @binding(1) var bg_tex: texture_2d<f32>;
@group(1) @binding(2) var thickness_tex: texture_2d<f32>;
@group(1) @binding(3) var depth_tex: texture_depth_2d;
@group(1) @binding(4) var<storage, read> particles_buf: array<GpuParticle>;
@group(1) @binding(5) var<storage, read> color_lut: array<vec4<f32>>;
@group(1) @binding(6) var shadow_blur_dst: texture_storage_2d<rgba16float, write>;

// Resolve compute pass (@group(2))
@group(2) @binding(0) var hdr_resolve_tex: texture_2d<f32>;
@group(2) @binding(1) var<storage, read_write> out_rgba_buf: array<u32>;

const PI: f32 = 3.14159265359;

fn aces_to_display_linear(col: vec3<f32>) -> vec3<f32> {
    let x = max(col, vec3<f32>(0.0));
    let mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn view_dist_to_ndc_depth(dist: f32) -> f32 {
    let c = u.proj * vec4<f32>(0.0, 0.0, -dist, 1.0);
    return clamp(c.z / max(c.w, 1e-6), 0.0, 0.99999);
}

fn ndc_depth_to_view_dist(z_ndc: f32) -> f32 {
    if z_ndc >= 0.99998 {
        return 1e6;
    }
    return u.proj[3][2] / max(z_ndc + u.proj[2][2], 1e-6);
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if len2 > 1e-12 {
        return v * inverseSqrt(len2);
    }
    return fallback;
}

fn shade_solid_obstacle(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    let light_int = u.light_dir.w;
    let amb_int = u.eye_light_dir.w;
    let albedo = vec3<f32>(0.28, 0.31, 0.37);
    let ndotl = max(dot(normal, light_dir), 0.0);
    let hemi = 0.5 + 0.5 * normal.y;
    let h = safe_normalize(view_dir + light_dir, vec3<f32>(0.0, 1.0, 0.0));
    let spec = pow(max(dot(normal, h), 0.0), 28.0) * 0.22;
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 4.0) * 0.08;
    return albedo * (amb_int * (0.55 + 0.45 * hemi) + ndotl * light_int * 0.78)
        + vec3<f32>(spec * light_int)
        + vec3<f32>(0.12, 0.22, 0.35) * rim;
}

struct HitResult {
    hit: bool,
    t: f32,
    n: vec3<f32>,
}

fn ray_intersect_collider(idx: u32, ro: vec3<f32>, rd: vec3<f32>) -> HitResult {
    let col = colliders[idx];
    let kind = col.kind_pad.x;
    var res = HitResult(false, 1e9, vec3<f32>(0.0, 1.0, 0.0));

    if kind == 0u {
        // Sphere
        let center = col.min_or_center.xyz;
        let radius = col.min_or_center.w;
        let oc = ro - center;
        let b = dot(oc, rd);
        let c = dot(oc, oc) - radius * radius;
        let disc = b * b - c;
        if disc >= 0.0 {
            let t = -b - sqrt(disc);
            if t > 0.05 {
                let hp = ro + rd * t;
                res = HitResult(true, t, (hp - center) / max(radius, 1e-5));
            }
        }
    } else if kind == 1u {
        // AABB Box
        let bmin = col.min_or_center.xyz;
        let bmax = col.max_or_ext.xyz;
        let inv_d = vec3<f32>(
            select(1e8, 1.0 / rd.x, abs(rd.x) > 1e-8),
            select(1e8, 1.0 / rd.y, abs(rd.y) > 1e-8),
            select(1e8, 1.0 / rd.z, abs(rd.z) > 1e-8),
        );
        let t0 = (bmin - ro) * inv_d;
        let t1 = (bmax - ro) * inv_d;
        let tmin_v = min(t0, t1);
        let tmax_v = max(t0, t1);
        let t_enter = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
        let t_exit = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
        if t_enter <= t_exit && t_enter > 0.05 {
            let hp = ro + rd * t_enter;
            let ctr = 0.5 * (bmin + bmax);
            let half = max(0.5 * (bmax - bmin), vec3<f32>(1e-5));
            let rel = (hp - ctr) / half;
            let ar = abs(rel);
            var n = vec3<f32>(0.0, 0.0, sign(rel.z));
            if ar.x >= ar.y && ar.x >= ar.z {
                n = vec3<f32>(sign(rel.x), 0.0, 0.0);
            } else if ar.y >= ar.x && ar.y >= ar.z {
                n = vec3<f32>(0.0, sign(rel.y), 0.0);
            }
            res = HitResult(true, t_enter, n);
        }
    } else if kind == 2u {
        // Vertical Cylinder
        let center = col.min_or_center.xyz;
        let radius = col.min_or_center.w;
        let half_h = col.max_or_ext.x * 0.5;
        let y_min = center.y - half_h;
        let y_max = center.y + half_h;
        let ox = ro.x - center.x;
        let oz = ro.z - center.z;
        let a = rd.x * rd.x + rd.z * rd.z;
        if a > 1e-8 {
            let b = ox * rd.x + oz * rd.z;
            let c = ox * ox + oz * oz - radius * radius;
            let disc = b * b - a * c;
            if disc >= 0.0 {
                let t = (-b - sqrt(disc)) / a;
                if t > 0.05 {
                    let y = ro.y + rd.y * t;
                    if y >= y_min && y <= y_max {
                        let hp = ro + rd * t;
                        let n = safe_normalize(vec3<f32>(hp.x - center.x, 0.0, hp.z - center.z), vec3<f32>(1.0, 0.0, 0.0));
                        res = HitResult(true, t, n);
                    }
                }
            }
        }
        if abs(rd.y) > 1e-6 {
            let t_top = (y_max - ro.y) / rd.y;
            if t_top > 0.05 && (!res.hit || t_top < res.t) {
                let hx = ox + rd.x * t_top;
                let hz = oz + rd.z * t_top;
                if hx * hx + hz * hz <= radius * radius {
                    res = HitResult(true, t_top, vec3<f32>(0.0, 1.0, 0.0));
                }
            }
            let t_bot = (y_min - ro.y) / rd.y;
            if t_bot > 0.05 && (!res.hit || t_bot < res.t) {
                let hx = ox + rd.x * t_bot;
                let hz = oz + rd.z * t_bot;
                if hx * hx + hz * hz <= radius * radius {
                    res = HitResult(true, t_bot, vec3<f32>(0.0, -1.0, 0.0));
                }
            }
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Pass 1A: Studio Sky, 3D Floor Grid (+ Volumetric Shadows/Caustics), Obstacles
// ---------------------------------------------------------------------------
@vertex
fn bg_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.99999, 1.0);
}

struct FragDepthOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn bg_fs(@builtin(position) pos: vec4<f32>) -> FragDepthOut {
    let ndc_x = 2.0 * (pos.x / f32(u.dims.x)) - 1.0;
    let ndc_y = 1.0 - 2.0 * (pos.y / f32(u.dims.y));
    let tan_half_fov = u.eye_pos.w;
    let aspect = u.cam_fwd.w;
    let sx = ndc_x * aspect * tan_half_fov;
    let sy = ndc_y * tan_half_fov;

    let cam_fwd = u.cam_fwd.xyz;
    let eye_pos = u.eye_pos.xyz;
    let light_dir = u.light_dir.xyz;
    let light_int = u.light_dir.w;
    let amb_int = u.eye_light_dir.w;
    let exposure = u.bg_low.w;
    let anim_time = u.bg_high.w;
    let ray_dir = normalize(cam_fwd + u.cam_right.xyz * sx + u.cam_up.xyz * sy);

    let sky_t = clamp(0.5 + 0.5 * ray_dir.y, 0.0, 1.0);
    var col_lin = mix(u.bg_low.xyz, u.bg_high.xyz, sky_t);
    var best_dist = 1e6;

    let is_2d = u.flags0.w != 0u;
    let show_grid = u.flags1.x != 0u;
    let floor_y = u.bounds_min.w;
    let margin = u.bounds_max.w;

    if !is_2d && show_grid && ray_dir.y < -1e-4 {
        let t_floor = (floor_y - eye_pos.y) / ray_dir.y;
        if t_floor > 0.05 {
            let hit = eye_pos + ray_dir * t_floor;
            if hit.x >= u.bounds_min.x - margin && hit.x <= u.bounds_max.x + margin
                && hit.z >= u.bounds_min.z - margin && hit.z <= u.bounds_max.z + margin
            {
                let dx = max(max(u.bounds_min.x - hit.x, 0.0), hit.x - u.bounds_max.x);
                let dz = max(max(u.bounds_min.z - hit.z, 0.0), hit.z - u.bounds_max.z);
                let edge_fade = clamp(1.0 - sqrt(dx * dx + dz * dz) / margin, 0.0, 1.0);
                if edge_fade > 0.001 {
                    let cell_size = 0.25;
                    let gx = i32(floor(hit.x / cell_size));
                    let gz = i32(floor(hit.z / cell_size));
                    let checker = f32((gx + gz) & 1);
                    var floor_albedo = vec3<f32>(0.042, 0.050, 0.066) + vec3<f32>(checker * 0.012);

                    let fx = fract(hit.x / cell_size);
                    let fz = fract(hit.z / cell_size);
                    let dist_minor = min(min(fx, 1.0 - fx), min(fz, 1.0 - fz));
                    let line_w = min(0.012 + t_floor * 0.0025, 0.08);
                    let minor_line = clamp(1.0 - dist_minor / line_w, 0.0, 1.0);
                    floor_albedo += vec3<f32>(0.030, 0.052, 0.085) * minor_line * 0.65;

                    var sun_vis = vec3<f32>(1.0);
                    var caustic_glow = vec3<f32>(0.0);

                    if u.flags2.w != 0u {
                        let s_clip = u.shadow_vp * vec4<f32>(hit, 1.0);
                        let suv = vec2<f32>(s_clip.x * 0.5 + 0.5, 0.5 - s_clip.y * 0.5);
                        if suv.x >= 0.0 && suv.x <= 1.0 && suv.y >= 0.0 && suv.y <= 1.0 {
                            let sres = i32(u.flags0.y);
                            let uv_f = suv * f32(sres) - vec2<f32>(0.5);
                            let sc0 = clamp(
                                vec2<i32>(floor(uv_f)),
                                vec2<i32>(0),
                                vec2<i32>(max(sres - 2, 0)),
                            );
                            let sc1 = min(sc0 + vec2<i32>(1), vec2<i32>(max(sres - 1, 0)));
                            let sf = clamp(uv_f - vec2<f32>(sc0), vec2<f32>(0.0), vec2<f32>(1.0));
                            let s00 = textureLoad(shadow_tex, sc0, 0).rg;
                            let s10 = textureLoad(shadow_tex, vec2<i32>(sc1.x, sc0.y), 0).rg;
                            let s01 = textureLoad(shadow_tex, vec2<i32>(sc0.x, sc1.y), 0).rg;
                            let s11 = textureLoad(shadow_tex, sc1, 0).rg;
                            let sv = mix(mix(s00, s10, sf.x), mix(s01, s11, sf.x), sf.y);
                            let avg_thick = max(sv.r, 0.0);
                            let avg_cov = max(sv.g, 0.0);
                            let dens = clamp(avg_thick * 2.4 + min(avg_cov * 0.35, 0.65), 0.0, 2.5);
                            if dens > 0.01 {
                                let shallow = u.shallow_lin.xyz;
                                let opacity = u.shallow_lin.w;
                                let refr_str = u.mat_opt1.y;
                                let metallic = u.mat_opt2.x;
                                let trans = exp(-dens * (vec3<f32>(1.5) - shallow) * 1.35);
                                sun_vis = vec3<f32>(0.22) + trans * (0.78 * (1.0 - opacity * 0.55));

                                if refr_str > 0.05 && metallic < 0.5 {
                                    let cp = vec2<f32>(hit.x, hit.z) * 5.5 + vec2<f32>(anim_time * 0.7, -anim_time * 0.5);
                                    let w1 = sin(cp.x * 2.3 + sin(cp.y * 1.9));
                                    let w2 = sin(cp.y * 2.7 - cos(cp.x * 2.1));
                                    let focus = pow((w1 * w2 + 1.0) * 0.5, 5.0);
                                    let caust_mask = clamp(dens * 1.4, 0.0, 1.0) * (1.0 - opacity * 0.65);
                                    caustic_glow = shallow * focus * caust_mask * refr_str * 0.55;
                                }
                            }
                        }
                    }

                    let ndotl = max(light_dir.y, 0.0);
                    let floor_lit = floor_albedo * (vec3<f32>(amb_int * 0.95) + sun_vis * (ndotl * light_int * 0.72)) + caustic_glow;
                    col_lin = mix(col_lin, floor_lit, edge_fade);
                    best_dist = max(t_floor * dot(ray_dir, cam_fwd), 0.05);
                }
            }
        }
    }

    let n_cols = u.flags0.z;
    for (var i = 0u; i < n_cols; i += 1u) {
        let hr = ray_intersect_collider(i, eye_pos, ray_dir);
        if hr.hit {
            let z_cam = hr.t * dot(ray_dir, cam_fwd);
            if z_cam > 0.05 && z_cam < best_dist {
                best_dist = z_cam;
                col_lin = shade_solid_obstacle(hr.n, -ray_dir, light_dir);
            }
        }
    }

    let disp_lin = aces_to_display_linear(col_lin * exposure);
    let out_d = select(1.0, view_dist_to_ndc_depth(best_dist), best_dist < 1e5);
    return FragDepthOut(vec4<f32>(disp_lin, best_dist), out_d);
}

// ---------------------------------------------------------------------------
// Pass 1B: Custom TriangleMesh Colliders
// ---------------------------------------------------------------------------
struct ObstacleVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) view_dist: f32,
}

@vertex
fn obstacle_vs(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>) -> ObstacleVsOut {
    let vp = u.view * vec4<f32>(pos, 1.0);
    return ObstacleVsOut(u.proj * vp, pos, normal, -vp.z);
}

@fragment
fn obstacle_fs(in: ObstacleVsOut) -> @location(0) vec4<f32> {
    let v_dir = safe_normalize(u.eye_pos.xyz - in.world_pos, vec3<f32>(0.0, 0.0, 1.0));
    var n = safe_normalize(in.normal, vec3<f32>(0.0, 1.0, 0.0));
    if dot(n, v_dir) < 0.0 {
        n = -n;
    }
    let col = shade_solid_obstacle(n, v_dir, u.light_dir.xyz);
    return vec4<f32>(aces_to_display_linear(col * u.bg_low.w), in.view_dist);
}

// ---------------------------------------------------------------------------
// Pass 2: Directional Volumetric Shadow Map (SurfaceMesh)
// ---------------------------------------------------------------------------
struct ShadowVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) light_dist: f32,
}

@vertex
fn shadow_vs(@location(0) pos: vec3<f32>) -> ShadowVsOut {
    let d = -dot(pos, u.light_dir.xyz);
    return ShadowVsOut(u.shadow_vp * vec4<f32>(pos, 1.0), d);
}

@fragment
fn shadow_fs(in: ShadowVsOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let s = select(in.light_dist, -in.light_dist, is_front);
    return vec4<f32>(s, 0.35, 0.0, 0.0);
}

// Separable Gaussian penumbra blur for smooth circular floor shadows (eliminates box/texel artifacts)
fn shadow_blur_axis(gid: vec3<u32>, axis: vec2<i32>) {
    let sres = i32(u.flags0.y);
    if i32(gid.x) >= sres || i32(gid.y) >= sres {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let w = array<f32, 4>(0.2960, 0.2255, 0.1008, 0.0257);
    var acc = max(textureLoad(shadow_tex, coord, 0).rg, vec2<f32>(0.0)) * w[0];
    let max_c = vec2<i32>(sres - 1);
    for (var i = 1; i <= 3; i += 1) {
        let p = clamp(coord + axis * i, vec2<i32>(0), max_c);
        let n = clamp(coord - axis * i, vec2<i32>(0), max_c);
        let sp = max(textureLoad(shadow_tex, p, 0).rg, vec2<f32>(0.0));
        let sn = max(textureLoad(shadow_tex, n, 0).rg, vec2<f32>(0.0));
        acc += (sp + sn) * w[i];
    }
    textureStore(shadow_blur_dst, coord, vec4<f32>(acc, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn shadow_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    shadow_blur_axis(gid, vec2<i32>(1, 0));
}

@compute @workgroup_size(16, 16, 1)
fn shadow_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    shadow_blur_axis(gid, vec2<i32>(0, 1));
}

// ---------------------------------------------------------------------------
// Pass 3: Camera-Space Optical Fluid Thickness (SurfaceMesh)
// ---------------------------------------------------------------------------
struct ThickVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) view_dist: f32,
}

@vertex
fn thickness_vs(@location(0) pos: vec3<f32>) -> ThickVsOut {
    let vp = u.view * vec4<f32>(pos, 1.0);
    return ThickVsOut(u.proj * vp, -vp.z);
}

@fragment
fn thickness_fs(in: ThickVsOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if is_front {
        return vec4<f32>(-in.view_dist, 1.0, 0.0, 0.0);
    }
    return vec4<f32>(in.view_dist, -1.0, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Pass 4A: Physically-Based Liquid Surface Mesh Shading
// ---------------------------------------------------------------------------
struct FluidVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) speed: f32,
    @location(3) foam: f32,
}

@vertex
fn fluid_vs(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) speed: f32,
    @location(3) foam: f32,
) -> FluidVsOut {
    return FluidVsOut(u.view_proj * vec4<f32>(pos, 1.0), pos, normal, speed, foam);
}

fn rotr(x: u32, k: u32) -> u32 {
    return (x >> k) | (x << (32u - k));
}

fn hash3(p: vec3<i32>) -> vec3<f32> {
    var x = (u32(p.x) * 1597334677u) ^ (u32(p.y) * 3812015801u) ^ (u32(p.z) * 2798796415u);
    x = x * 0x27d4eb2du;
    let r1 = f32(x & 0xFFFFu) / 65535.0;
    x = rotr(x, 11u) * 0x165667b1u;
    let r2 = f32(x & 0xFFFFu) / 65535.0;
    x = rotr(x, 13u) * 0x27d4eb2du;
    let r3 = f32(x & 0xFFFFu) / 65535.0;
    return vec3<f32>(r1, r2, r3);
}

fn value_noise_3d(p: vec3<f32>) -> f32 {
    let fl = floor(p);
    let i = vec3<i32>(fl);
    let f = p - fl;
    let n = f * f * (vec3<f32>(3.0) - 2.0 * f);

    let c000 = hash3(i + vec3<i32>(0, 0, 0)).x;
    let c100 = hash3(i + vec3<i32>(1, 0, 0)).x;
    let c010 = hash3(i + vec3<i32>(0, 1, 0)).x;
    let c110 = hash3(i + vec3<i32>(1, 1, 0)).x;
    let c001 = hash3(i + vec3<i32>(0, 0, 1)).x;
    let c101 = hash3(i + vec3<i32>(1, 0, 1)).x;
    let c011 = hash3(i + vec3<i32>(0, 1, 1)).x;
    let c111 = hash3(i + vec3<i32>(1, 1, 1)).x;

    let x00 = mix(c000, c100, n.x);
    let x10 = mix(c010, c110, n.x);
    let x01 = mix(c001, c101, n.x);
    let x11 = mix(c011, c111, n.x);
    let y0 = mix(x00, x10, n.y);
    let y1 = mix(x01, x11, n.y);
    return mix(y0, y1, n.z);
}

fn fbm_3d(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    var sum = 0.0;
    var amp = 0.5;
    for (var i = 0; i < 4; i += 1) {
        sum += amp * value_noise_3d(p);
        p *= 2.03;
        amp *= 0.5;
    }
    return sum;
}

fn voronoi_ridge_3d(p: vec3<f32>) -> f32 {
    let fl = floor(p);
    let base = vec3<i32>(fl);
    let frac = p - fl;
    var d1 = 8.0;
    var d2 = 8.0;

    for (var dz = -1; dz <= 1; dz += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            for (var dx = -1; dx <= 1; dx += 1) {
                let offset = vec3<i32>(dx, dy, dz);
                let jitter = hash3(base + offset);
                let r = vec3<f32>(f32(dx), f32(dy), f32(dz)) + jitter - frac;
                let d = dot(r, r);
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                } else if d < d2 {
                    d2 = d;
                }
            }
        }
    }
    return 1.0 - clamp((sqrt(d2) - sqrt(d1)) * 2.2, 0.0, 1.0);
}

fn grid_line_2d(uv: vec2<f32>) -> f32 {
    let f = abs(uv - floor(uv) - vec2<f32>(0.5));
    let dist_to_edge = min(0.5 - f.x, 0.5 - f.y);
    return 1.0 - clamp(dist_to_edge / 0.045, 0.0, 1.0);
}

struct TexSample {
    normal: vec3<f32>,
    albedo_scale: vec3<f32>,
    emissive_caustic: f32,
    foam_boost: f32,
}

fn evaluate_texture(world_pos: vec3<f32>, base_normal: vec3<f32>, speed: f32, time: f32) -> TexSample {
    let mode = u.flags1.z;
    let strength = clamp(u.mat_opt2.w, 0.0, 1.0);
    if mode == 0u || strength <= 1e-4 {
        return TexSample(base_normal, vec3<f32>(1.0), 0.0, 0.0);
    }

    let scale = max(u.mat_opt2.z, 0.2);
    let p = world_pos * scale;

    if mode == 1u {
        // Ripples
        let t = time * 3.2;
        let k1 = vec3<f32>(3.1, 0.7, 2.4);
        let k2 = vec3<f32>(-2.3, 1.1, 3.3);
        let k3 = vec3<f32>(1.5, -2.9, -2.1);

        let ph1 = dot(p, k1) - t * 1.3;
        let ph2 = dot(p, k2) + t * 1.7;
        let ph3 = dot(p, k3 * 1.6) - t * 2.3;

        let grad = k1 * cos(ph1) * 0.45 + k2 * cos(ph2) * 0.35 + k3 * cos(ph3) * 0.20;
        let tangent_grad = grad - base_normal * dot(grad, base_normal);
        let n = safe_normalize(base_normal + tangent_grad * (strength * 0.22), base_normal);
        let shimmer = 1.0 + (sin(ph1) * sin(ph2)) * strength * 0.08;
        return TexSample(n, vec3<f32>(shimmer), 0.0, 0.0);
    } else if mode == 2u {
        // Caustics
        let q = p * 0.85 + vec3<f32>(time * 0.55, -time * 0.35, time * 0.45);
        let c1 = voronoi_ridge_3d(q);
        let c2 = voronoi_ridge_3d(q * 1.7 + vec3<f32>(3.7));
        let caustic = pow(c1 * 0.65 + c2 * 0.35, 2.2) * strength;
        let eps = 0.06;
        let cx = voronoi_ridge_3d(q + vec3<f32>(eps, 0.0, 0.0)) - c1;
        let cz = voronoi_ridge_3d(q + vec3<f32>(0.0, 0.0, eps)) - c1;
        let bump = vec3<f32>(cx, 0.0, cz) * (strength * 1.8);
        let tangent_bump = bump - base_normal * dot(bump, base_normal);
        let n = safe_normalize(base_normal + tangent_bump, base_normal);
        return TexSample(n, vec3<f32>(1.0) + vec3<f32>(0.25, 0.45, 0.55) * caustic, caustic * 0.55, 0.0);
    } else if mode == 3u {
        // Foam streaks
        let q = p * 0.9 + vec3<f32>(time * 0.3, -time * 0.2, time * 0.25);
        let f = fbm_3d(q);
        let speed_factor = clamp(speed / 2.2, 0.15, 1.0);
        let streaks = clamp((f - 0.42) * 2.4, 0.0, 1.0) * strength * speed_factor;
        let bump = vec3<f32>(f - 0.5, 0.0, fbm_3d(q + vec3<f32>(0.0, 0.0, 1.3)) - 0.5) * (strength * 0.30);
        let tangent_bump = bump - base_normal * dot(bump, base_normal);
        let n = safe_normalize(base_normal + tangent_bump, base_normal);
        return TexSample(n, vec3<f32>(1.0), 0.0, streaks * 0.85);
    } else {
        // Triplanar Grid
        let w = pow(abs(base_normal), vec3<f32>(4.0));
        let w_sum = max(w.x + w.y + w.z, 1e-5);
        let weights = w / w_sum;
        let gx = grid_line_2d(vec2<f32>(p.y, p.z));
        let gy = grid_line_2d(vec2<f32>(p.x, p.z));
        let gz = grid_line_2d(vec2<f32>(p.x, p.y));
        let line = gx * weights.x + gy * weights.y + gz * weights.z;
        let checker = f32((i32(floor(p.x)) + i32(floor(p.y)) + i32(floor(p.z))) & 1);
        let base_mod = 1.0 - checker * 0.14 * strength;
        let line_boost = line * strength * 0.65;
        return TexSample(base_normal, vec3<f32>(base_mod + line_boost), line_boost * 0.25, line_boost * 0.35);
    }
}

fn sample_studio_sky(refl_dir: vec3<f32>, light_dir: vec3<f32>, roughness: f32, metallic: f32) -> vec3<f32> {
    let r = safe_normalize(refl_dir, vec3<f32>(0.0, 1.0, 0.0));
    let up = clamp(r.y, -1.0, 1.0);
    let sharp = clamp(1.0 - roughness, 0.05, 1.0);

    var horizon = u.sky_horizon.xyz;
    var zenith = u.sky_zenith.xyz;
    var ground = u.sky_ground.xyz;

    let env_desat = metallic * 0.72;
    let luma_w = vec3<f32>(0.2126, 0.7152, 0.0722);
    horizon = mix(horizon, vec3<f32>(dot(horizon, luma_w)), env_desat);
    zenith = mix(zenith, vec3<f32>(dot(zenith, luma_w)), env_desat);
    ground = mix(ground, vec3<f32>(dot(ground, luma_w)), env_desat);

    let upper_raw = mix(horizon, zenith, pow(max(up, 0.0), 0.58));
    let metal_sky_dome = mix(vec3<f32>(0.26, 0.28, 0.31), vec3<f32>(0.14, 0.16, 0.19), pow(max(up, 0.0), 0.70));
    let upper_sky = mix(upper_raw, max(upper_raw, metal_sky_dome), metallic);

    let floor_dark = mix(
        horizon * 0.35,
        ground * (1.0 - metallic + 0.55 * metallic),
        1.0 - smoothstep(-0.22, 0.04, up),
    );
    let up_mix = smoothstep(-0.08, 0.12, up);
    var sky = mix(floor_dark, upper_sky, up_mix);

    let horizon_width = 0.15 * (1.0 - metallic) + 0.065 * metallic;
    let horizon_band = exp(-pow(up / horizon_width, 2.0));
    let neg_fill = 1.0 - (0.30 + 0.40 * metallic) * exp(-pow((up - 0.25) / 0.16, 2.0));
    let cove_col = mix(vec3<f32>(0.20, 0.22, 0.25), vec3<f32>(0.58, 0.61, 0.66), metallic);
    sky = sky * neg_fill + cove_col * horizon_band * (0.28 + 0.45 * metallic);

    if up < 0.0 {
        let r_xz = safe_normalize(vec3<f32>(r.x, 0.0, r.z), vec3<f32>(1.0, 0.0, 0.0));
        let l_xz = safe_normalize(vec3<f32>(light_dir.x, 0.0, light_dir.z), vec3<f32>(0.0, 0.0, 1.0));
        let key_bounce = 0.5 + 0.5 * dot(r_xz, l_xz);
        let floor_sheen = (0.025 + (0.095 - 0.025) * key_bounce) * exp(-pow((up + 0.32) / 0.38, 2.0));
        let rim_floor = 0.055 * pow(clamp(1.0 + up, 0.0, 1.0), 3.0) * (0.5 + 0.5 * r_xz.z);
        sky += vec3<f32>(0.88, 0.92, 0.98) * (floor_sheen + rim_floor) * (0.20 + 0.75 * metallic);
    }

    let sb_dir = normalize(vec3<f32>(-0.22, 0.84, 0.34));
    let sb_u = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), sb_dir));
    let sb_v = normalize(cross(sb_dir, sb_u));
    let su = abs(dot(r, sb_u));
    let sv = abs(dot(r, sb_v));
    let sf = max(dot(r, sb_dir), 0.0);
    let feather = 0.05 + (1.0 - sharp) * 0.30;
    let softbox_rect = (1.0 - smoothstep(0.36, 0.36 + feather, su))
        * (1.0 - smoothstep(0.24, 0.24 + feather, sv))
        * smoothstep(0.2, 0.55, sf);
    let softbox_col = mix(vec3<f32>(0.85, 0.91, 1.0), vec3<f32>(1.15, 1.18, 1.22), metallic);
    sky += softbox_col * softbox_rect * (0.85 + 0.55 * metallic);

    let strip_y = exp(-pow((up - 0.14) / 0.26, 2.0));
    let r_horiz_dir = safe_normalize(vec3<f32>(r.x, 0.0, r.z), vec3<f32>(1.0, 0.0, 0.0));
    let strip_pow = (4.0 * (1.0 - metallic) + 16.0 * metallic)
        + sharp * (10.0 * (1.0 - metallic) + 42.0 * metallic);
    let strip_l = pow(clamp(dot(r_horiz_dir, normalize(vec3<f32>(-0.96, 0.0, -0.28))), 0.0, 1.0), strip_pow);
    let strip_r = pow(clamp(dot(r_horiz_dir, normalize(vec3<f32>(0.95, 0.0, -0.31))), 0.0, 1.0), strip_pow);
    let strip_front = pow(clamp(dot(r_horiz_dir, normalize(vec3<f32>(-0.25, 0.0, 0.97))), 0.0, 1.0), strip_pow * 1.4);
    let strip_col = mix(vec3<f32>(0.55, 0.68, 0.85), vec3<f32>(0.95, 0.98, 1.02), metallic);
    sky += strip_col * strip_y * (strip_l + 0.80 * strip_r + 0.55 * strip_front) * (0.35 + 0.70 * metallic);

    let sun_dot = max(dot(r, light_dir), 0.0);
    let sun_col = mix(vec3<f32>(1.0, 0.95, 0.86), vec3<f32>(1.42, 1.42, 1.42), metallic);
    sky += sun_col * pow(sun_dot, 12.0 + sharp * 64.0) * (0.55 + 0.70 * metallic);

    return sky;
}

fn sample_bg_bilinear(fx_in: f32, fy_in: f32) -> vec3<f32> {
    let max_x = f32(max(i32(u.dims.x) - 1, 0));
    let max_y = f32(max(i32(u.dims.y) - 1, 0));
    let fx = clamp(fx_in, 0.0, max_x);
    let fy = clamp(fy_in, 0.0, max_y);
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = min(x0 + 1, i32(u.dims.x) - 1);
    let y1 = min(y0 + 1, i32(u.dims.y) - 1);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);

    let c00 = textureLoad(bg_tex, vec2<i32>(x0, y0), 0).rgb;
    let c10 = textureLoad(bg_tex, vec2<i32>(x1, y0), 0).rgb;
    let c01 = textureLoad(bg_tex, vec2<i32>(x0, y1), 0).rgb;
    let c11 = textureLoad(bg_tex, vec2<i32>(x1, y1), 0).rgb;
    return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

@fragment
fn fluid_fs(in: FluidVsOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let pix = clamp(vec2<i32>(in.clip.xy), vec2<i32>(0), vec2<i32>(i32(u.dims.x) - 1, i32(u.dims.y) - 1));
    let bg_sample = textureLoad(bg_tex, pix, 0);
    let solid_view_dist = bg_sample.a;
    let th_sample = textureLoad(thickness_tex, pix, 0);

    let v_dir = safe_normalize(u.eye_pos.xyz - in.world_pos, vec3<f32>(0.0, 0.0, 1.0));
    var base_n = safe_normalize(in.normal, vec3<f32>(0.0, 1.0, 0.0));
    if !is_front {
        base_n = -base_n;
    }
    if dot(base_n, v_dir) < 0.0 {
        base_n = -base_n;
    }

    let anim_time = u.bg_high.w;
    let tex = evaluate_texture(in.world_pos, base_n, in.speed, anim_time);
    var n = tex.normal;
    if dot(n, v_dir) < 0.02 {
        n = safe_normalize(n + v_dir * 0.15, base_n);
    }

    let light_dir = u.light_dir.xyz;
    let light_int = u.light_dir.w;
    let amb_int = u.eye_light_dir.w;
    let ndotv = clamp(dot(n, v_dir), 0.001, 1.0);
    let ndotl = max(dot(n, light_dir), 0.0);

    var raw_thick = th_sample.r;
    if th_sample.g > 0.5 && solid_view_dist < 1e4 {
        raw_thick += th_sample.g * solid_view_dist;
    }
    let thickness = select(min(raw_thick, 2.8), 0.07 + (1.0 - ndotv) * 0.25, raw_thick <= 0.004);

    let ior = max(u.shallow_srgb.w, 1.01);
    let base_f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let rough = clamp(u.deep_srgb.w, 0.02, 1.0);
    let alpha = rough * rough;
    let alpha2 = alpha * alpha;
    let metallic = clamp(u.mat_opt2.x, 0.0, 1.0);

    let shallow_tex = clamp(u.shallow_lin.xyz * tex.albedo_scale, vec3<f32>(0.0), vec3<f32>(2.0));
    let deep_tex = clamp(u.deep_lin.xyz * tex.albedo_scale, vec3<f32>(0.0), vec3<f32>(2.0));
    let f0_vec = mix(vec3<f32>(base_f0), shallow_tex, metallic);
    let fresnel = f0_vec + (vec3<f32>(1.0) - f0_vec) * pow(1.0 - ndotv, 5.0);

    let px_f = in.clip.x - 0.5;
    let py_f = in.clip.y - 0.5;
    let dispersion = u.sky_ground.w;
    let refr_str = u.mat_opt1.y;
    let refr_scale = refr_str * (1.0 - metallic) * (f32(u.dims.y) * 0.032) * (thickness / (thickness + 0.25));

    var bg_lin: vec3<f32>;
    if dispersion > 0.0 {
        let r_col = sample_bg_bilinear(px_f - n.x * refr_scale * (1.0 - dispersion), py_f + n.y * refr_scale * (1.0 - dispersion));
        let g_col = sample_bg_bilinear(px_f - n.x * refr_scale, py_f + n.y * refr_scale);
        let b_col = sample_bg_bilinear(px_f - n.x * refr_scale * (1.0 + dispersion), py_f + n.y * refr_scale * (1.0 + dispersion));
        bg_lin = vec3<f32>(r_col.x, g_col.y, b_col.z);
    } else {
        bg_lin = sample_bg_bilinear(px_f - n.x * refr_scale, py_f + n.y * refr_scale);
    }

    let absorption = u.deep_lin.w;
    let opacity = u.shallow_lin.w;
    let opt_depth = thickness * absorption;
    let depth_mix = 1.0 - exp(-opt_depth * 0.90);
    let body_tint = mix(shallow_tex, deep_tex, depth_mix);
    let srgb_tint = mix(u.shallow_srgb.xyz, u.deep_srgb.xyz, depth_mix);

    let absorb_sigma = max(vec3<f32>(1.04) - srgb_tint * 0.92, vec3<f32>(0.04)) * (absorption * 1.85);
    let transmittance = exp(-absorb_sigma * thickness);

    let hemi = 0.5 + 0.5 * n.y;
    let body_scatter = body_tint * (amb_int * (0.42 + 0.58 * hemi) + ndotl * light_int * 0.58);
    let filtered_bg = bg_lin * transmittance;
    let turbidity = select(
        clamp(opacity * opacity * 0.34, 0.02, 0.45),
        clamp(opacity, 0.0, 1.0),
        opacity >= 0.88,
    );
    let volume_inscatter = body_scatter * (vec3<f32>(1.0) - transmittance) * 0.55;
    let dielectric_refr = mix(filtered_bg + volume_inscatter, body_scatter, turbidity);
    let refracted_col = dielectric_refr * (1.0 - metallic);

    let refl_dir = (-v_dir) - n * (2.0 * dot(-v_dir, n));
    let env_col = sample_studio_sky(refl_dir, light_dir, rough, metallic) * u.mat_opt1.w;

    let h_vec = safe_normalize(v_dir + light_dir, vec3<f32>(0.0, 1.0, 0.0));
    let ndoth = max(dot(n, h_vec), 0.0);
    let denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
    let d_ggx = alpha2 / (PI * denom * denom + 1e-6);
    let k = pow(rough + 1.0, 2.0) * 0.125;
    let g_v = ndotv / (ndotv * (1.0 - k) + k);
    let g_l = ndotl / (ndotl * (1.0 - k) + k);
    let spec_brdf = (d_ggx * g_v * g_l * fresnel) / (4.0 * ndotv * ndotl + 1e-4);
    let sun_spec = spec_brdf * ndotl * light_int * u.mat_opt1.x;

    let back_light = pow(max(dot(safe_normalize(-light_dir + n * 0.45, -light_dir), v_dir), 0.0), 3.0);
    let thin_factor = exp(-thickness * 2.2);
    let sss_col = shallow_tex * back_light * thin_factor * u.mat_opt1.z * light_int * 0.85 * (1.0 - metallic);

    let metal_cavity_bounce = shallow_tex * (0.005 + 0.010 * hemi) * amb_int * metallic;

    var final_lin = refracted_col * (vec3<f32>(1.0) - fresnel)
        + env_col * fresnel
        + metal_cavity_bounce
        + sun_spec
        + sss_col
        + shallow_tex * tex.emissive_caustic * light_int * (1.0 - metallic * 0.65);

    let foam_amount = clamp((in.foam * 0.85 + tex.foam_boost) * u.mat_opt2.y, 0.0, 0.92);
    if foam_amount > 0.005 {
        let foam_lit = vec3<f32>(0.94, 0.97, 1.0) * (amb_int * 0.85 + (0.35 + 0.65 * ndotl) * light_int * 0.68);
        final_lin = mix(final_lin, foam_lit, foam_amount);
    }

    let exposure = u.bg_low.w;
    return vec4<f32>(aces_to_display_linear(final_lin * exposure), 1.0);
}

// ---------------------------------------------------------------------------
// Pass 4B: GPU Particle Spheres (Direct from Live PBF Particle Buffer)
// ---------------------------------------------------------------------------
struct ParticleVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) view_z: f32,
    @location(2) base_srgb: vec3<f32>,
}

fn sample_color_lut(cmap_id: u32, t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0) * 255.0;
    let i0 = min(u32(floor(t)), 255u);
    let i1 = min(i0 + 1u, 255u);
    let f = t - floor(t);
    let base = (cmap_id & 7u) * 256u;
    return mix(color_lut[base + i0].rgb, color_lut[base + i1].rgb, f);
}

fn eval_particle_color(pos: vec3<f32>, vel: vec3<f32>, dens: f32) -> vec3<f32> {
    let mode = u.flags2.x;
    let cmap = u.flags2.y;
    let foam_on = u.flags2.z != 0u;
    let rho0 = max(u.sky_zenith.w, 1e-4);
    let speed = length(vel);

    var rgb: vec3<f32>;
    if mode == 7u {
        rgb = vec3<f32>(30.0, 140.0, 225.0) / 255.0;
    } else {
        var t = 0.0;
        if mode == 0u {
            t = speed / 2.6;
        } else if mode == 1u {
            t = max(dens - rho0, 0.0) / (rho0 * 0.14);
        } else if mode == 2u {
            t = (abs(dens - rho0) / rho0) * 4.0;
        } else if mode == 3u {
            let h = max(u.bounds_max.y - u.bounds_min.y, 1e-4);
            t = (pos.y - u.bounds_min.y) / h;
        } else if mode == 4u {
            t = (0.5 * dot(vel, vel)) / 4.0;
        } else if mode == 5u {
            let rho_def = clamp(1.0 - dens / rho0, 0.0, 1.0);
            t = rho_def * 1.45 * clamp(speed / 2.2, 0.15, 1.2);
        } else {
            let horiz = length(vec2<f32>(vel.x, vel.z));
            t = (horiz * 0.65 + abs(vel.y) * 0.85) / 2.8;
        }
        rgb = sample_color_lut(cmap, t);
    }

    if foam_on && mode != 7u {
        let rho_ratio = dens / rho0;
        if speed > 1.15 && rho_ratio < 0.78 {
            let spd_f = clamp((speed - 1.15) / 2.6, 0.0, 1.0);
            let air_f = clamp((0.78 - rho_ratio) / 0.36, 0.0, 1.0);
            let foam = clamp(spd_f * air_f * 0.85, 0.0, 0.85);
            rgb = mix(rgb, vec3<f32>(244.0, 250.0, 255.0) / 255.0, foam);
        }
    }
    return rgb;
}

@vertex
fn particle_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> ParticleVsOut {
    let gp = particles_buf[iid];
    if gp.pos_type.w > 0.5 {
        return ParticleVsOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), vec2<f32>(0.0), 1e6, vec3<f32>(0.0));
    }

    let pos = gp.pos_type.xyz;
    let vel = gp.vel_dens.xyz;
    let dens = gp.vel_dens.w;

    let view_pos = u.view * vec4<f32>(pos, 1.0);
    let z_dist = -view_pos.z;
    if z_dist <= 0.04 {
        return ParticleVsOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), vec2<f32>(0.0), 1e6, vec3<f32>(0.0));
    }

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let q = corners[vid];

    let clip_center = u.proj * view_pos;
    let radius = u.sky_horizon.w;
    let tan_half_fov = u.eye_pos.w;
    let aspect = u.cam_fwd.w;

    let r_ndc_y = radius / (z_dist * tan_half_fov);
    let r_ndc_x = r_ndc_y / aspect;
    let ndc_xy = clip_center.xy / clip_center.w + q * vec2<f32>(r_ndc_x, r_ndc_y);
    let out_clip = vec4<f32>(ndc_xy * clip_center.w, clip_center.z, clip_center.w);

    let base_srgb = eval_particle_color(pos, vel, dens);
    return ParticleVsOut(out_clip, vec2<f32>(q.x, -q.y), z_dist, base_srgb);
}

@fragment
fn particle_fs(in: ParticleVsOut) -> FragDepthOut {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if r2 > 1.0 {
        discard;
    }
    let z_norm = sqrt(1.0 - r2);
    let radius = u.sky_horizon.w;
    let depth_dist = max(in.view_z - z_norm * radius, u.cam_right.w + 1e-4);
    let exposure = u.bg_low.w;

    var lit_srgb: vec3<f32>;
    if u.flags1.w != 0u {
        let normal = vec3<f32>(in.quad_uv.x, -in.quad_uv.y, z_norm);
        let ld = u.eye_light_dir.xyz;
        let light_int = u.light_dir.w;
        let amb_int = u.eye_light_dir.w;
        let half_dir = safe_normalize(ld + vec3<f32>(0.0, 0.0, 1.0), ld);

        let diff = max(dot(normal, ld), 0.0);
        let hemi = 0.5 + 0.5 * normal.y;
        let spec = pow(max(dot(normal, half_dir), 0.0), 36.0) * light_int * 0.85;
        let fresnel = 0.025 + 0.975 * pow(1.0 - z_norm, 4.0);
        let shade = amb_int * (0.70 + 0.45 * hemi) + 0.52 * light_int * diff;

        let rim_srgb = pow(clamp(u.sky_horizon.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
        let spec_col = vec3<f32>(150.0, 156.0, 165.0) / 255.0;
        lit_srgb = clamp((in.base_srgb * shade + spec_col * spec + rim_srgb * (fresnel * 0.65)) * exposure, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        lit_srgb = clamp(in.base_srgb * exposure, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    let disp_lin = pow(lit_srgb, vec3<f32>(2.2));
    return FragDepthOut(vec4<f32>(disp_lin, depth_dist), view_dist_to_ndc_depth(depth_dist));
}

// ---------------------------------------------------------------------------
// Pass 5: Depth-Aware Wireframe Overlay
// ---------------------------------------------------------------------------
struct WireVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) view_dist: f32,
}

@vertex
fn wireframe_vs(@location(0) pos: vec3<f32>) -> WireVsOut {
    let vp = u.view * vec4<f32>(pos, 1.0);
    return WireVsOut(u.proj * vp, -vp.z);
}

@fragment
fn wireframe_fs(in: WireVsOut) -> @location(0) vec4<f32> {
    let pix = clamp(vec2<i32>(in.clip.xy), vec2<i32>(0), vec2<i32>(i32(u.dims.x) - 1, i32(u.dims.y) - 1));
    let scene_ndc_z = textureLoad(depth_tex, pix, 0);
    let scene_dist = ndc_depth_to_view_dist(scene_ndc_z);
    let base_alpha = select(0.12, 0.55, in.view_dist <= scene_dist + 0.015);
    let alpha = min(base_alpha * u.wire_color.w, 0.95);
    let lin_col = pow(u.wire_color.xyz, vec3<f32>(2.2));
    return vec4<f32>(lin_col, alpha);
}

// ---------------------------------------------------------------------------
// Pass 6: SSAA Linear Box Resolve + Gamma + Interleaved Gradient Noise Dither
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16, 1)
fn resolve_ssaa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_w = u.dims.z;
    let out_h = u.dims.w;
    if gid.x >= out_w || gid.y >= out_h {
        return;
    }

    let ssaa = max(u.flags0.x, 1u);
    var acc = vec3<f32>(0.0);
    let base_x = i32(gid.x * ssaa);
    let base_y = i32(gid.y * ssaa);

    for (var dy = 0u; dy < ssaa; dy += 1u) {
        for (var dx = 0u; dx < ssaa; dx += 1u) {
            let sc = vec2<i32>(base_x + i32(dx), base_y + i32(dy));
            acc += textureLoad(hdr_resolve_tex, sc, 0).rgb;
        }
    }

    let inv_samples = 1.0 / f32(ssaa * ssaa);
    let lin = clamp(acc * inv_samples, vec3<f32>(0.0), vec3<f32>(1.0));
    let ign = fract(fract(0.06711056 * f32(gid.x) + 0.00583715 * f32(gid.y)) * 52.9829189) - 0.5;
    let dither = ign * 0.75;

    let r = u32(clamp(pow(lin.r, 1.0 / 2.2) * 255.0 + 0.5 + dither, 0.0, 255.0));
    let g = u32(clamp(pow(lin.g, 1.0 / 2.2) * 255.0 + 0.5 + dither, 0.0, 255.0));
    let b = u32(clamp(pow(lin.b, 1.0 / 2.2) * 255.0 + 0.5 + dither, 0.0, 255.0));

    out_rgba_buf[gid.y * out_w + gid.x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
