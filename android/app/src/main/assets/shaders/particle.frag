#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
in vec3 v_pos_eye;
in vec3 v_world_pos;
in float v_metric_t;
in float v_speed_t;

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

    vec3 surf_pos = v_world_pos + n * u_radius;
    float tex_str = clamp(u_tex_strength, 0.0, 1.0);
    float t_anim = u_anim_time;

    float caustic_boost = 0.0;
    float foam_tex = 0.0;
    float grid_mod = 1.0;

    if (u_tex_mode != 0 && tex_str > 0.005) {
        if (u_tex_mode == 1) {
            float w1 = cos(surf_pos.x * 28.0 + surf_pos.y * 19.0 - t_anim * 3.8);
            float w2 = sin(surf_pos.z * 26.0 - surf_pos.x * 16.0 + t_anim * 3.1);
            n = normalize(n + vec3(w1, w2, 0.0) * (tex_str * 0.42));
            caustic_boost = max(w1 * w2, 0.0) * tex_str * 0.20;
        } else if (u_tex_mode == 2) {
            vec3 p = surf_pos * 18.0;
            float c1 = abs(sin(p.x + sin(p.y * 1.3 + t_anim * 2.4)) * cos(p.y + cos(p.z * 1.4 - t_anim * 1.9)));
            float web = pow(1.0 - clamp(c1, 0.0, 1.0), 3.0);
            caustic_boost = web * tex_str * 0.85;
            n = normalize(n + vec3(sin(p.x * 1.7), cos(p.y * 1.7), 0.0) * (tex_str * 0.25));
        } else if (u_tex_mode == 3) {
            vec3 p = surf_pos * 24.0 + vec3(t_anim * 1.5, -t_anim * 0.8, t_anim * 1.2);
            float f = (sin(p.x) * cos(p.y) + sin(p.y * 1.7 + p.z) + cos(p.z * 2.3 - p.x)) * 0.333 + 0.5;
            foam_tex = clamp((f - 0.38) * 2.6, 0.0, 1.0) * tex_str * (0.28 + 0.72 * v_speed_t);
            n = normalize(n + vec3(f - 0.5, (1.0 - f) - 0.5, 0.0) * (tex_str * 0.22));
        } else if (u_tex_mode == 4) {
            float gx = abs(fract(surf_pos.x * 8.0) - 0.5);
            float gy = abs(fract(surf_pos.y * 8.0) - 0.5);
            float gz = abs(fract(surf_pos.z * 8.0) - 0.5);
            float line = max(1.0 - min(gx, gy) * 14.0, 1.0 - gz * 14.0);
            float line_c = clamp(line, 0.0, 1.0) * tex_str;
            grid_mod = 1.0 - tex_str * 0.14;
            caustic_boost = line_c * 0.45;
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

    base_color = base_color * grid_mod + mix(u_shallow_color, vec3(0.96, 0.99, 1.0), 0.35) * caustic_boost;
    if (foam_tex > 0.005) {
        base_color = mix(base_color, vec3(0.95, 0.98, 1.0), clamp(foam_tex, 0.0, 0.85));
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
            + sss_tint;
    }

    fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
