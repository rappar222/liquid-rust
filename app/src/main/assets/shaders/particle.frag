#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
in vec3 v_pos_eye;
in vec3 v_world_pos;
in float v_metric_t;
in float v_speed_t;
in float v_surf_mask;
in vec2 v_vel_xy;

uniform mat4 u_proj;
uniform float u_radius;
uniform vec3 u_light_dir;
uniform int u_lit_spheres;
uniform int u_color_mode;   // 0: vel, 1: pres, 2: dens, 3: optical material
uniform int u_color_map;    // 0: ocean, 1: turbo, 2: viridis, 3: fire
uniform int u_tex_mode;     // 0: none, 1: ripples, 2: caustics, 3: foam, 4: grid
uniform float u_tex_strength;
uniform float u_anim_time;
uniform vec3 u_shallow_color;
uniform vec3 u_deep_color;
uniform vec4 u_optics;      // x: ior, y: roughness, z: specular, w: metallic
uniform float u_refraction;
uniform float u_foam_amount;

out vec4 fragColor;

vec3 colormap_ocean(float t) {
    vec3 c0 = vec3(0.035, 0.140, 0.360);
    vec3 c1 = vec3(0.060, 0.440, 0.780);
    vec3 c2 = vec3(0.180, 0.780, 0.950);
    vec3 c3 = vec3(0.920, 0.980, 1.000);
    if (t < 0.38) return mix(c0, c1, t / 0.38);
    if (t < 0.76) return mix(c1, c2, (t - 0.38) / 0.38);
    return mix(c2, c3, (t - 0.76) / 0.24);
}

vec3 colormap_turbo(float t) {
    const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

    float x = clamp(t, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(
        dot(v4, kRedVec4) + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4) + dot(v2, kBlueVec2)
    ), 0.0, 1.0);
}

vec3 colormap_viridis(float t) {
    vec3 c0 = vec3(0.267, 0.005, 0.329);
    vec3 c1 = vec3(0.191, 0.407, 0.556);
    vec3 c2 = vec3(0.208, 0.718, 0.472);
    vec3 c3 = vec3(0.993, 0.906, 0.144);
    if (t < 0.33) return mix(c0, c1, t / 0.33);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
    return mix(c2, c3, (t - 0.66) / 0.34);
}

vec3 colormap_fire(float t) {
    vec3 c0 = vec3(0.08, 0.02, 0.12);
    vec3 c1 = vec3(0.72, 0.12, 0.22);
    vec3 c2 = vec3(0.98, 0.56, 0.08);
    vec3 c3 = vec3(1.00, 0.96, 0.72);
    if (t < 0.35) return mix(c0, c1, t / 0.35);
    if (t < 0.72) return mix(c1, c2, (t - 0.35) / 0.37);
    return mix(c2, c3, (t - 0.72) / 0.28);
}

vec2 hash22(ivec2 p) {
    uvec2 q = uvec2(p) * uvec2(1597334677u, 3812015801u);
    uint n = (q.x ^ q.y) * 2798796415u;
    uint n2 = ((n >> 13u) ^ n) * 2246822519u;
    return vec2(float(n & 65535u), float((n2 >> 8u) & 65535u)) * (1.0 / 65535.0);
}

float worley_caustic(vec2 u, float t_anim) {
    ivec2 ip = ivec2(floor(u));
    vec2 fp = u - vec2(ip);
    float d1 = 8.0;
    float d2 = 8.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 o = ivec2(dx, dy);
            vec2 j = hash22(ip + o);
            vec2 r = vec2(o) + 0.5 + 0.36 * vec2(
                sin(t_anim * 1.5 + 6.2831853 * j.x),
                cos(t_anim * 1.3 + 6.2831853 * j.y)
            ) - fp;
            float d = dot(r, r);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    float edge = sqrt(d2) - sqrt(d1);
    return pow(clamp(1.0 - edge * 2.4, 0.0, 1.0), 2.3);
}

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) {
        discard;
    }

    float z = sqrt(max(1.0 - r2, 0.0));
    vec3 n = vec3(v_uv.x, v_uv.y, z);

#ifdef MODE_3D
    vec3 sphere_pos_eye = v_pos_eye + vec3(v_uv * u_radius, z * u_radius);
    vec4 clip_pos = u_proj * vec4(sphere_pos_eye, 1.0);
    gl_FragDepth = clamp(((clip_pos.z / clip_pos.w) + 1.0) * 0.5, 0.0, 1.0);
#endif

    vec3 surf_pos = v_world_pos + n * (u_radius * 1.35);
    float tex_str = clamp(u_tex_strength, 0.0, 1.0);
    float t_anim = u_anim_time;

    float caustic_boost = 0.0;
    float tex_shade = 1.0;
    float foam_tex = 0.0;
    float flow_surf = clamp(v_surf_mask + v_speed_t * 0.65, 0.0, 1.0);

    if (u_tex_mode != 0 && tex_str > 0.005) {
        if (u_tex_mode == 1) {
            // Silky Undulating Water Ripples: gentle wave currents + capillary droplet shimmer + sunlit surface crests
            float warp = 0.55 * sin(surf_pos.x * 6.5 + surf_pos.z * 5.0 - t_anim * 2.8)
                       + 0.30 * cos(surf_pos.x * 12.5 - surf_pos.z * 9.5 + t_anim * 2.1);
            float wave_band = sin(surf_pos.y * 24.0 + warp * 2.2 - t_anim * 1.8);
            float wave_fine = sin(surf_pos.y * 52.0 - warp * 3.1 + surf_pos.x * 18.0 + t_anim * 3.4);
            float dx_w = cos(surf_pos.x * 52.0 + surf_pos.y * 18.0 - t_anim * 4.2) * 0.45
                       + sin(surf_pos.x * 31.0 - surf_pos.y * 44.0 + t_anim * 3.3) * 0.35;
            float dy_w = wave_band * 0.55 + wave_fine * 0.35;
            float bump_gain = tex_str * (0.24 + 0.26 * flow_surf);
            n = normalize(n + vec3(dx_w, dy_w, 0.0) * bump_gain);
            tex_shade = 1.0 + (wave_band * 0.11 + wave_fine * 0.04) * tex_str;
            float crest = smoothstep(0.25, 0.95, wave_band * 0.65 + wave_fine * 0.35);
            caustic_boost = crest * tex_str * (0.16 + 0.36 * flow_surf);
        } else if (u_tex_mode == 2) {
            // Cellular Voronoi Pool Caustics: authentic curved light web
            vec2 uv_c = vec2(
                surf_pos.x * 5.6 + surf_pos.y * 1.1 + surf_pos.z * 2.6 + sin(surf_pos.y * 4.0 + t_anim * 1.2) * 0.18,
                surf_pos.y * 5.6 - surf_pos.x * 0.8 + surf_pos.z * 3.1 + cos(surf_pos.x * 4.0 - t_anim * 1.0) * 0.18
            );
            float c_main = worley_caustic(uv_c, t_anim);
            float c_fine = worley_caustic(uv_c * 1.85 + vec2(3.7, -2.1), t_anim * 1.35);
            float caustic = c_main * 0.74 + c_fine * 0.26;
            caustic_boost = caustic * tex_str * 0.78;
            n = normalize(n + vec3(c_main - 0.35, c_fine - 0.35, 0.0) * (tex_str * 0.20));
        } else if (u_tex_mode == 3) {
            // Natural Aerated Seafoam: buoyant frothy whitecap bubbles on free surface & splashing waves
            vec3 bp = surf_pos * 36.0 + vec3(t_anim * 1.1, -t_anim * 0.7, t_anim * 0.9);
            float bub = (sin(bp.x) * cos(bp.y) + sin(bp.y * 1.6 + bp.z) + cos(bp.z * 2.1 - bp.x)) * 0.333 + 0.5;
            float micro = smoothstep(0.22, 0.72, bub);
            float aeration = clamp(v_surf_mask * 0.90 + smoothstep(0.16, 0.72, v_speed_t) * 0.95, 0.0, 1.0);
            foam_tex = micro * aeration * clamp(tex_str * 1.35, 0.0, 1.0);
            n = normalize(n + vec3(bub - 0.5, 0.5 - bub, 0.0) * (foam_tex * 0.35));
        } else if (u_tex_mode == 4) {
            // Refracted Pool-Tile Grid: background optical grid dynamically refracted through water droplets & flow
            vec2 refr_xy = v_world_pos.xy + n.xy * (u_radius * 2.1) - v_vel_xy * 0.035 + vec2(
                0.012 * sin(v_world_pos.y * 10.0 - t_anim * 2.2),
                0.012 * cos(v_world_pos.x * 10.0 + t_anim * 1.9)
            );
            float freq = 5.5;
            float gx = 0.5 - abs(fract(refr_xy.x * freq) - 0.5);
            float gy = 0.5 - abs(fract(refr_xy.y * freq) - 0.5);
            float line_d = min(gx, gy);
#ifdef MODE_3D
            float rz = v_world_pos.z + n.z * (u_radius * 1.2);
            float gz = 0.5 - abs(fract(rz * freq) - 0.5);
            line_d = min(line_d, gz);
#endif
            float line_mask = 1.0 - smoothstep(0.018, 0.062, line_d);
            float checker = mod(floor(refr_xy.x * freq) + floor(refr_xy.y * freq), 2.0);
            tex_shade = 1.0 - tex_str * (0.09 * checker);
            caustic_boost = line_mask * (0.55 * tex_str);
        }
    }

    float metallic = clamp(u_optics.w, 0.0, 1.0);
    vec3 base_color;
    if (u_color_mode == 3) {
        float rim = 1.0 - z;
        float shallow_mix = clamp(0.28 + 0.48 * v_speed_t + 0.42 * rim, 0.0, 1.0);
        base_color = mix(u_deep_color, u_shallow_color, shallow_mix);
        float crest_foam = smoothstep(0.62, 0.98, v_speed_t) * u_foam_amount * (1.0 - metallic);
        base_color = mix(base_color, vec3(0.94, 0.98, 1.0), crest_foam * 0.72);
    } else {
        if (u_color_map == 1) base_color = colormap_turbo(v_metric_t);
        else if (u_color_map == 2) base_color = colormap_viridis(v_metric_t);
        else if (u_color_map == 3) base_color = colormap_fire(v_metric_t);
        else base_color = colormap_ocean(v_metric_t);
    }

    vec3 boost_col = mix(u_shallow_color, vec3(0.92, 0.99, 1.0), 0.28) * caustic_boost;
    base_color = base_color * tex_shade + boost_col;
    if (foam_tex > 0.005) {
        base_color = mix(base_color, vec3(0.94, 0.98, 1.0), clamp(foam_tex, 0.0, 0.88));
    }

    vec3 rgb = base_color;
    if (u_lit_spheres != 0) {
        float ior = max(u_optics.x, 1.01);
        float rough = clamp(u_optics.y, 0.015, 0.95);
        float spec_mult = max(u_optics.z, 0.0);
        float refr_glow = clamp(u_refraction, 0.0, 1.5);

        float f0_diel = pow((ior - 1.0) / (ior + 1.0), 2.0);
        vec3 f0 = mix(vec3(f0_diel), base_color, metallic);
        float ndotv = max(n.z, 0.001);
        vec3 fresnel = f0 + (vec3(1.0) - f0) * pow(1.0 - ndotv, 4.2);

        float ndotl = max(dot(n, u_light_dir), 0.0);
        float hemi = 0.5 + 0.5 * n.y;
        vec3 half_dir = normalize(u_light_dir + vec3(0.0, 0.0, 1.0));
        float spec_pow = clamp(1.8 / (rough * rough + 0.003), 8.0, 420.0);
        float spec_energy = (0.28 + 0.72 * (1.0 - rough)) * spec_mult;
        vec3 spec = pow(max(dot(n, half_dir), 0.0), spec_pow) * spec_energy * fresnel;

        float refl_y = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
        vec3 sky_env = mix(vec3(0.08, 0.14, 0.24), vec3(0.55, 0.78, 0.98), pow(refl_y, 1.5));
        vec3 body_lit = base_color * (0.28 + 0.18 * hemi + 0.62 * ndotl * (1.0 - 0.65 * metallic));

        float sss_rim = pow(1.0 - ndotv, 2.2) * refr_glow * (1.0 - metallic);
        vec3 sss_tint = u_shallow_color * (0.35 + 0.65 * (0.5 + 0.5 * dot(n, -u_light_dir))) * sss_rim;

        rgb = mix(body_lit, sky_env * mix(vec3(1.0), base_color, metallic), clamp(fresnel * (0.75 + 0.25 * metallic), 0.0, 0.96))
            + spec
            + sss_tint
            + boost_col * 0.22;
    }

    fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
