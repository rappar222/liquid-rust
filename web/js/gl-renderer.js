// WebGL 2.0 Hardware-Accelerated Studio Renderer for LiquidRust Web
// High-performance point sprite spheres with glossy Fresnel sky-rim, specular highlights,
// true 3D spherical depth, studio floor grid, and zero per-frame JS heap allocations.

import { ColorMap } from './color-map.js?v=20261113b';
import { Mat4 } from './math.js?v=20261113b';

export class GlRenderer {
    constructor(gl) {
        this.gl = gl;
        this.viewProjScratch = new Mat4();
        this.lineScratch = new Float32Array(4096);
        this.initShaders();
        this.initBuffers();
    }

    initShaders() {
        const gl = this.gl;

        // 1. Glossy Liquid Particle Shader (WebGL 2.0 / GLSL 3.00 ES)
        const pVsSource = `#version 300 es
        precision highp float;

        layout(location = 0) in vec3 a_pos;
        layout(location = 1) in vec4 a_color;
        layout(location = 2) in vec3 a_surf_vel;

        uniform mat4 u_view;
        uniform mat4 u_proj;
        uniform float u_point_radius;
        uniform float u_viewport_height;
        uniform float u_tan_half_fov;

        out vec4 v_color;
        out vec3 v_pos_eye;
        out vec3 v_world_pos;
        out float v_surf_mask;
        out vec2 v_vel_xy;

        void main() {
            v_color = a_color;
            v_world_pos = a_pos;
            v_surf_mask = a_surf_vel.x;
            v_vel_xy = a_surf_vel.yz;
            vec4 eye_pos = u_view * vec4(a_pos, 1.0);
            v_pos_eye = eye_pos.xyz;
            gl_Position = u_proj * eye_pos;

            float dist = max(-eye_pos.z, 0.001);
            float size_px = (u_point_radius / (dist * u_tan_half_fov)) * u_viewport_height;
            gl_PointSize = clamp(size_px, 2.0, 128.0);
        }
        `;

        const pFsSource = `#version 300 es
        precision highp float;

        in vec4 v_color;
        in vec3 v_pos_eye;
        in vec3 v_world_pos;
        in float v_surf_mask;
        in vec2 v_vel_xy;

        uniform mat4 u_proj;
        uniform float u_point_radius;
        uniform vec3 u_light_dir;
        uniform int u_lit_spheres;
        uniform int u_is_2d;
        uniform int u_color_mode;
        uniform int u_tex_mode;
        uniform float u_tex_strength;
        uniform float u_anim_time;
        uniform vec3 u_shallow_color;
        uniform vec3 u_deep_color;
        uniform vec4 u_optics;
        uniform float u_refraction;
        uniform float u_foam_amount;

        out vec4 fragColor;

        vec2 hash22(ivec2 p) {
            uvec2 q = uvec2(p) * uvec2(1597334677u, 3812015801u);
            uint n = (q.x ^ q.y) * 2798796415u;
            uint n2 = ((n >> 13u) ^ n) * 2246822519u;
            return vec2(float(n & 65535u), float((n2 >> 8u) & 65535u)) * (1.0 / 65535.0);
        }

        float worley_caustic(vec2 u_coord, float t_anim) {
            ivec2 ip = ivec2(floor(u_coord));
            vec2 fp = u_coord - vec2(ip);
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
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(coord, coord);
            if (r2 > 1.0) {
                discard;
            }

            float z = sqrt(max(1.0 - r2, 0.0));
            vec3 n = vec3(coord.x, -coord.y, z);
            vec3 surf_pos = v_world_pos + n * (u_point_radius * 1.35);

            if (u_is_2d == 0) {
                vec3 sphere_pos_eye = v_pos_eye + vec3(coord.x * u_point_radius, -coord.y * u_point_radius, z * u_point_radius);
                vec4 clip_pos = u_proj * vec4(sphere_pos_eye, 1.0);
                gl_FragDepth = clamp(((clip_pos.z / clip_pos.w) + 1.0) * 0.5, 0.0, 1.0);
            }

            float speed_t = v_color.a;
            float tex_str = clamp(u_tex_strength, 0.0, 1.0);
            float t_anim = u_anim_time;

            float caustic_boost = 0.0;
            float foam_tex = 0.0;
            float tex_shade = 1.0;
            float flow_surf = clamp(v_surf_mask + speed_t * 0.65, 0.0, 1.0);

            if (u_tex_mode != 0 && tex_str > 0.005) {
                if (u_tex_mode == 1) {
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
                    vec3 bp = surf_pos * 36.0 + vec3(t_anim * 1.1, -t_anim * 0.7, t_anim * 0.9);
                    float bub = (sin(bp.x) * cos(bp.y) + sin(bp.y * 1.6 + bp.z) + cos(bp.z * 2.1 - bp.x)) * 0.333 + 0.5;
                    float micro = smoothstep(0.22, 0.72, bub);
                    float aeration = clamp(v_surf_mask * 0.90 + smoothstep(0.16, 0.72, speed_t) * 0.95, 0.0, 1.0);
                    foam_tex = micro * aeration * clamp(tex_str * 1.35, 0.0, 1.0);
                    n = normalize(n + vec3(bub - 0.5, 0.5 - bub, 0.0) * (foam_tex * 0.35));
                } else if (u_tex_mode == 4) {
                    vec2 refr_xy = v_world_pos.xy + n.xy * (u_point_radius * 2.1) - v_vel_xy * 0.035 + vec2(
                        0.012 * sin(v_world_pos.y * 10.0 - t_anim * 2.2),
                        0.012 * cos(v_world_pos.x * 10.0 + t_anim * 1.9)
                    );
                    float freq = 5.5;
                    float gx = 0.5 - abs(fract(refr_xy.x * freq) - 0.5);
                    float gy = 0.5 - abs(fract(refr_xy.y * freq) - 0.5);
                    float line_d = min(gx, gy);
                    if (u_is_2d == 0) {
                        float rz = v_world_pos.z + n.z * (u_point_radius * 1.2);
                        float gz = 0.5 - abs(fract(rz * freq) - 0.5);
                        line_d = min(line_d, gz);
                    }
                    float line_mask = 1.0 - smoothstep(0.018, 0.062, line_d);
                    float checker = mod(floor(refr_xy.x * freq) + floor(refr_xy.y * freq), 2.0);
                    tex_shade = 1.0 - tex_str * (0.09 * checker);
                    caustic_boost = line_mask * (0.55 * tex_str);
                }
            }

            float metallic = clamp(u_optics.w, 0.0, 1.0);
            vec3 base_color = v_color.rgb;
            if (u_color_mode == 3) {
                float rim = 1.0 - z;
                float shallow_mix = clamp(0.28 + 0.48 * speed_t + 0.42 * rim, 0.0, 1.0);
                base_color = mix(u_deep_color, u_shallow_color, shallow_mix);
                float crest_foam = smoothstep(0.62, 0.98, speed_t) * u_foam_amount * (1.0 - metallic);
                base_color = mix(base_color, vec3(0.94, 0.98, 1.0), crest_foam * 0.72);
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
        `;

        this.particleProgram = this.createProgram(pVsSource, pFsSource);

        this.uView = gl.getUniformLocation(this.particleProgram, 'u_view');
        this.uProj = gl.getUniformLocation(this.particleProgram, 'u_proj');
        this.uPointRadius = gl.getUniformLocation(this.particleProgram, 'u_point_radius');
        this.uViewportHeight = gl.getUniformLocation(this.particleProgram, 'u_viewport_height');
        this.uTanHalfFov = gl.getUniformLocation(this.particleProgram, 'u_tan_half_fov');
        this.uLightDir = gl.getUniformLocation(this.particleProgram, 'u_light_dir');
        this.uLitSpheres = gl.getUniformLocation(this.particleProgram, 'u_lit_spheres');
        this.uIs2D = gl.getUniformLocation(this.particleProgram, 'u_is_2d');
        this.uColorMode = gl.getUniformLocation(this.particleProgram, 'u_color_mode');
        this.uTexMode = gl.getUniformLocation(this.particleProgram, 'u_tex_mode');
        this.uTexStrength = gl.getUniformLocation(this.particleProgram, 'u_tex_strength');
        this.uAnimTime = gl.getUniformLocation(this.particleProgram, 'u_anim_time');
        this.uShallowColor = gl.getUniformLocation(this.particleProgram, 'u_shallow_color');
        this.uDeepColor = gl.getUniformLocation(this.particleProgram, 'u_deep_color');
        this.uOptics = gl.getUniformLocation(this.particleProgram, 'u_optics');
        this.uRefraction = gl.getUniformLocation(this.particleProgram, 'u_refraction');
        this.uFoamAmount = gl.getUniformLocation(this.particleProgram, 'u_foam_amount');

        // 2. Line Shader (Studio floor grid, chamber box, and colliders)
        const lVsSource = `#version 300 es
        precision highp float;

        layout(location = 0) in vec3 a_pos;
        uniform mat4 u_view_proj;

        void main() {
            gl_Position = u_view_proj * vec4(a_pos, 1.0);
        }
        `;

        const lFsSource = `#version 300 es
        precision highp float;

        uniform vec4 u_line_color;
        out vec4 fragColor;

        void main() {
            fragColor = u_line_color;
        }
        `;

        this.lineProgram = this.createProgram(lVsSource, lFsSource);
        this.uViewProj = gl.getUniformLocation(this.lineProgram, 'u_view_proj');
        this.uLineColor = gl.getUniformLocation(this.lineProgram, 'u_line_color');
    }

    createProgram(vsSource, fsSource) {
        const gl = this.gl;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSource);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            const err = gl.getShaderInfoLog(vs);
            gl.deleteShader(vs);
            throw new Error('Vertex shader compile error: ' + err);
        }

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSource);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            const err = gl.getShaderInfoLog(fs);
            gl.deleteShader(vs);
            gl.deleteShader(fs);
            throw new Error('Fragment shader compile error: ' + err);
        }

        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);

        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            const err = gl.getProgramInfoLog(prog);
            gl.deleteProgram(prog);
            throw new Error('Shader program link error: ' + err);
        }

        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return prog;
    }

    initBuffers() {
        const gl = this.gl;

        this.particleVao = gl.createVertexArray();
        this.particleVbo = gl.createBuffer();

        gl.bindVertexArray(this.particleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.particleVbo);

        const stride = 40;
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);

        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 12);

        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 28);

        this.lineVao = gl.createVertexArray();
        this.lineVbo = gl.createBuffer();

        gl.bindVertexArray(this.lineVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.lineVbo);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);

        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        this.particleVertexCapacity = 8000;
        this.particleVertexData = new Float32Array(this.particleVertexCapacity * 10);
    }

    render(solver, camera, viewportWidth, viewportHeight, opts = {}) {
        const gl = this.gl;

        gl.viewport(0, 0, viewportWidth, viewportHeight);
        gl.clearColor(9.0 / 255.0, 12.0 / 255.0, 18.0 / 255.0, 1.0);
        gl.clearDepth(1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const aspect = Math.max(0.1, viewportWidth / Math.max(1.0, viewportHeight));
        const viewMat = camera.viewMatrix();
        const projMat = camera.projectionMatrix(aspect);

        this.viewProjScratch.elements.set(projMat.elements);
        const viewProjMat = this.viewProjScratch.multiply(viewMat);

        this.renderWireframes(solver, viewProjMat);
        this.renderParticles(solver, camera, viewMat, projMat, viewportHeight, opts);
    }

    pushSegment(idx, ax, ay, az, bx, by, bz) {
        if (idx + 6 > this.lineScratch.length) {
            const grown = new Float32Array(this.lineScratch.length * 2);
            grown.set(this.lineScratch);
            this.lineScratch = grown;
        }
        const s = this.lineScratch;
        s[idx] = ax; s[idx + 1] = ay; s[idx + 2] = az;
        s[idx + 3] = bx; s[idx + 4] = by; s[idx + 5] = bz;
        return idx + 6;
    }

    renderWireframes(solver, viewProjMat) {
        const gl = this.gl;
        const bMin = solver.bounds.min;
        const bMax = solver.bounds.max;
        let idx = 0;

        if (!solver.is2D) {
            const yFloor = bMin.y - 0.001;
            const divs = 8;
            for (let i = 1; i < divs; i++) {
                const t = i / divs;
                const gx = bMin.x + (bMax.x - bMin.x) * t;
                const gz = bMin.z + (bMax.z - bMin.z) * t;
                idx = this.pushSegment(idx, gx, yFloor, bMin.z, gx, yFloor, bMax.z);
                idx = this.pushSegment(idx, bMin.x, yFloor, gz, bMax.x, yFloor, gz);
            }
        }
        const gridVerts = idx / 3;
        const wireStart = idx;

        if (solver.is2D) {
            idx = this.pushSegment(idx, bMin.x, bMin.y, 0, bMax.x, bMin.y, 0);
            idx = this.pushSegment(idx, bMax.x, bMin.y, 0, bMax.x, bMax.y, 0);
            idx = this.pushSegment(idx, bMax.x, bMax.y, 0, bMin.x, bMax.y, 0);
            idx = this.pushSegment(idx, bMin.x, bMax.y, 0, bMin.x, bMin.y, 0);

            if (solver.obstacleEnabled) {
                const ox = solver.obstaclePos.x;
                const oy = solver.obstaclePos.y;
                const r = solver.obstacleRadius;
                const segments = 36;
                for (let i = 0; i < segments; i++) {
                    const t0 = (i / segments) * Math.PI * 2;
                    const t1 = ((i + 1) / segments) * Math.PI * 2;
                    idx = this.pushSegment(
                        idx,
                        ox + Math.cos(t0) * r, oy + Math.sin(t0) * r, 0,
                        ox + Math.cos(t1) * r, oy + Math.sin(t1) * r, 0
                    );
                }
            }
        } else {
            idx = this.pushSegment(idx, bMin.x, bMin.y, bMin.z, bMax.x, bMin.y, bMin.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMin.z, bMax.x, bMax.y, bMin.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMin.z, bMin.x, bMax.y, bMin.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMin.z, bMin.x, bMin.y, bMin.z);

            idx = this.pushSegment(idx, bMin.x, bMin.y, bMax.z, bMax.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMax.z, bMax.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMax.z, bMin.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMax.z, bMin.x, bMin.y, bMax.z);

            idx = this.pushSegment(idx, bMin.x, bMin.y, bMin.z, bMin.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMin.y, bMin.z, bMax.x, bMin.y, bMax.z);
            idx = this.pushSegment(idx, bMax.x, bMax.y, bMin.z, bMax.x, bMax.y, bMax.z);
            idx = this.pushSegment(idx, bMin.x, bMax.y, bMin.z, bMin.x, bMax.y, bMax.z);

            if (solver.obstacleEnabled) {
                const o = solver.obstaclePos;
                const r = solver.obstacleRadius;
                const segments = 28;

                for (let i = 0; i < segments; i++) {
                    const t0 = (i / segments) * Math.PI * 2;
                    const t1 = ((i + 1) / segments) * Math.PI * 2;
                    const cos0 = Math.cos(t0) * r, sin0 = Math.sin(t0) * r;
                    const cos1 = Math.cos(t1) * r, sin1 = Math.sin(t1) * r;

                    idx = this.pushSegment(idx, o.x + cos0, o.y + sin0, o.z, o.x + cos1, o.y + sin1, o.z);
                    idx = this.pushSegment(idx, o.x + cos0, o.y, o.z + sin0, o.x + cos1, o.y, o.z + sin1);
                    idx = this.pushSegment(idx, o.x, o.y + cos0, o.z + sin0, o.x, o.y + cos1, o.z + sin1);
                }
            }
        }

        const wireVerts = (idx - wireStart) / 3;
        if (idx > 0) {
            gl.useProgram(this.lineProgram);
            gl.uniformMatrix4fv(this.uViewProj, false, viewProjMat.elements);

            gl.bindVertexArray(this.lineVao);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.lineVbo);
            gl.bufferData(gl.ARRAY_BUFFER, this.lineScratch.subarray(0, idx), gl.DYNAMIC_DRAW);

            gl.disable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            if (gridVerts > 0) {
                gl.uniform4f(this.uLineColor, 0.17, 0.24, 0.35, 0.45);
                gl.drawArrays(gl.LINES, 0, gridVerts);
            }
            if (wireVerts > 0) {
                gl.uniform4f(this.uLineColor, 0.30, 0.66, 0.96, 0.80);
                gl.drawArrays(gl.LINES, gridVerts, wireVerts);
            }
            gl.disable(gl.BLEND);
        }
    }

    renderParticles(solver, camera, viewMat, projMat, viewportHeight, opts = {}) {
        const gl = this.gl;
        const n = solver.particleCount;
        if (n === 0) return;

        const colorMode = opts.colorMode ?? 'material';
        const colorMapType = opts.colorMapType ?? 'ocean';
        const litSpheres = opts.litSpheres ?? true;
        const shallow = opts.shallowColor ?? [0.22, 0.74, 0.97];
        const deep = opts.deepColor ?? [0.05, 0.24, 0.54];
        const texMode = opts.texMode ?? 1;
        const texStrength = opts.texStrength ?? 0.50;
        const ior = opts.ior ?? 1.333;
        const roughness = opts.roughness ?? 0.045;
        const specular = opts.specular ?? 1.35;
        const refraction = opts.refraction ?? 0.42;
        const metallic = opts.metallic ?? 0.0;
        const foamAmount = opts.foamAmount ?? 0.35;
        const animTime = opts.animTime ?? 0.0;

        let cMode = 0;
        if (colorMode === 'pressure') cMode = 1;
        else if (colorMode === 'density') cMode = 2;
        else if (colorMode === 'material' || colorMode === 'single') cMode = 3;

        if (n > this.particleVertexCapacity) {
            this.particleVertexCapacity = n * 2;
            this.particleVertexData = new Float32Array(this.particleVertexCapacity * 10);
        }

        const data = this.particleVertexData;
        const pos = solver.positions;
        const vel = solver.velocities;
        const dens = solver.densities;
        const pres = solver.pressures;
        const invRho0 = 1.0 / Math.max(solver.restDensity, 1.0);

        for (let i = 0; i < n; i++) {
            const i3 = i * 3;
            const i10 = i * 10;

            data[i10] = pos[i3];
            data[i10 + 1] = pos[i3 + 1];
            data[i10 + 2] = pos[i3 + 2];

            const vx = vel[i3], vy = vel[i3 + 1], vz = vel[i3 + 2];
            const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            const speedT = Math.min(1.0, speed / 2.8);

            if (cMode === 3) {
                data[i10 + 3] = shallow[0];
                data[i10 + 4] = shallow[1];
                data[i10 + 5] = shallow[2];
            } else {
                let t = 0.0;
                if (cMode === 0) {
                    t = Math.min(1.0, speed / 2.5);
                } else if (cMode === 1) {
                    t = Math.min(1.0, pres[i] / (solver.restDensity * 0.45));
                } else {
                    const dev = Math.abs(dens[i] - solver.restDensity) / solver.restDensity;
                    t = Math.min(1.0, dev * 4.0);
                }
                const rgb = ColorMap.sample(colorMapType, t);
                data[i10 + 3] = rgb[0];
                data[i10 + 4] = rgb[1];
                data[i10 + 5] = rgb[2];
            }
            data[i10 + 6] = speedT;

            const rhoRatio = dens[i] * invRho0;
            const st = Math.max(0.0, Math.min(1.0, (rhoRatio - 0.62) / (0.89 - 0.62)));
            const surfMask = 1.0 - st * st * (3.0 - 2.0 * st);
            data[i10 + 7] = surfMask;
            data[i10 + 8] = vx;
            data[i10 + 9] = vy;
        }

        gl.useProgram(this.particleProgram);

        gl.uniformMatrix4fv(this.uView, false, viewMat.elements);
        gl.uniformMatrix4fv(this.uProj, false, projMat.elements);
        gl.uniform1f(this.uPointRadius, solver.particleRadius * 1.28);
        gl.uniform1f(this.uViewportHeight, viewportHeight);
        gl.uniform1f(this.uTanHalfFov, Math.tan(camera.fovY * 0.5));

        const lx = 0.42, ly = 0.82, lz = 0.48;
        const invLen = 1.0 / Math.hypot(lx, ly, lz);
        gl.uniform3f(this.uLightDir, lx * invLen, ly * invLen, lz * invLen);

        gl.uniform1i(this.uLitSpheres, litSpheres ? 1 : 0);
        gl.uniform1i(this.uIs2D, solver.is2D ? 1 : 0);
        gl.uniform1i(this.uColorMode, cMode);
        gl.uniform1i(this.uTexMode, texMode | 0);
        gl.uniform1f(this.uTexStrength, texStrength);
        gl.uniform1f(this.uAnimTime, animTime);
        gl.uniform3f(this.uShallowColor, shallow[0], shallow[1], shallow[2]);
        gl.uniform3f(this.uDeepColor, deep[0], deep[1], deep[2]);
        gl.uniform4f(this.uOptics, ior, roughness, specular, metallic);
        gl.uniform1f(this.uRefraction, refraction);
        gl.uniform1f(this.uFoamAmount, foamAmount);

        if (solver.is2D) {
            gl.disable(gl.DEPTH_TEST);
            gl.depthMask(false);
        } else {
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(gl.LEQUAL);
            gl.depthMask(true);
        }

        gl.bindVertexArray(this.particleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.particleVbo);
        gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, n * 10), gl.DYNAMIC_DRAW);

        gl.drawArrays(gl.POINTS, 0, n);

        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.useProgram(null);
    }
}
