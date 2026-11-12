#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec4 a_pos_rho;  // xyz: world pos, w: density
layout(location = 1) in vec4 a_vel_pres; // xyz: velocity, w: pressure_norm [0..1]

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_radius;
uniform float u_inv_rho0;
uniform int u_color_mode; // 0: velocity, 1: pressure, 2: density, 3: optical material

out vec2 v_uv;
out vec3 v_pos_eye;
out vec3 v_world_pos;
out float v_metric_t;
out float v_speed_t;

void main() {
    const vec2 corners[4] = vec2[4](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );

    vec2 uv = corners[gl_VertexID];
    v_uv = uv;
    v_world_pos = a_pos_rho.xyz;

    float speed = length(a_vel_pres.xyz);
    v_speed_t = min(speed / 2.8, 1.0);

    if (u_color_mode == 0) {
        v_metric_t = min(speed / 2.5, 1.0);
    } else if (u_color_mode == 1) {
        v_metric_t = clamp(a_vel_pres.w, 0.0, 1.0);
    } else if (u_color_mode == 2) {
        float dev = abs(a_pos_rho.w * u_inv_rho0 - 1.0);
        v_metric_t = min(dev * 4.0, 1.0);
    } else {
        v_metric_t = v_speed_t;
    }

    vec4 eye_center = u_view * vec4(a_pos_rho.xyz, 1.0);
    v_pos_eye = eye_center.xyz;

    vec3 quad_eye = eye_center.xyz + vec3(uv * u_radius, 0.0);
    gl_Position = u_proj * vec4(quad_eye, 1.0);
}
