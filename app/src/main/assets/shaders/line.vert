#version 300 es
precision highp float;

layout(location = 0) in vec3 a_pos;
uniform mat4 u_view_proj;

void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}
