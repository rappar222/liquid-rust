#include <metal_stdlib>
using namespace metal;

struct SimUniforms {
    float4 bounds_min;   // xyz: min, w: cell_size (h)
    float4 bounds_max;   // xyz: max, w: particle_radius
    float4 gravity_dt;   // xyz: gravity, w: dt
    int4   grid_dim;     // xyz: nx, ny, nz, w: total_cells
    float4 touch_orig;   // xyz: ray origin, w: strength
    float4 touch_dir;    // xyz: ray dir, w: mode (-1 repel, 0 off, +1 attract)
    float4 obstacle;     // xyz: sphere pos, w: sphere radius
    float4 kernel_p;     // x: h, y: h2, z: inv_rho0, w: rho0
    float4 factors;      // x: poly6, y: spiky, z: w_q, w: tensile_k
    float4 solver;       // x: eps, y: max_disp, z: restitution, w: friction
    float4 fluid;        // x: viscosity, y: st_coeff, z: max_vel, w: obs_enabled
    int4   counts;       // x: particle_count, y: is_2d, z: 0, w: 0
    float4 extra;        // x: vorticity, y: spacing, z: cfm_scale, w: self_density
};

struct RenderUniforms {
    float4x4 view;
    float4x4 proj;
    float4x4 view_proj;
    float4   params;        // x: radius, y: inv_rho0, z: tex_strength, w: anim_time
    int4     modes;         // x: color_mode, y: color_map, z: lit_spheres, w: tex_mode
    float4   light_foam;    // xyz: light_dir, w: foam_amount
    float4   shallow_refr;  // xyz: shallow_color, w: refraction
    float4   deep_pad;      // xyz: deep_color, w: 0
    float4   optics;        // x: ior, y: roughness, z: specular, w: metallic
    float4   line_color;    // rgba wireframe color
};

inline int3 cell_coords(float3 p, constant SimUniforms& u) {
    float inv_h = 1.0f / u.bounds_min.w;
    int3 c = int3(floor((p - u.bounds_min.xyz) * inv_h));
    return clamp(c, int3(0), u.grid_dim.xyz - int3(1));
}

inline int cell_index(int3 c, int3 dim) {
    return (c.z * dim.y + c.y) * dim.x + c.x;
}

inline float bell_cohesion(float r, float h) {
    if (r <= 0.0001f || r >= h) return 0.0f;
    float q = r / h;
    if (q < 0.5f) {
        return 2.0f * pow(1.0f - q, 3.0f) * pow(q, 3.0f) - (1.0f / 64.0f);
    }
    return pow(1.0f - q, 3.0f) * pow(q, 3.0f);
}

// ============================================================================
// PASS 1: Predict Positions + External Gravity / Tilt + Spring Touch Ray
// ============================================================================
kernel void pbf_predict(
    const device float4* pos_buf [[buffer(0)]],
    const device float4* vel_buf [[buffer(1)]],
    device float4* pred_buf      [[buffer(2)]],
    constant SimUniforms& u      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.counts.x) return;

    float4 pos_rho = pos_buf[gid];
    float3 p_pos = pos_rho.xyz;
    float3 vel = vel_buf[gid].xyz;
    float dt = u.gravity_dt.w;
    bool is_2d = (u.counts.y != 0);

    float3 f_user = float3(0.0f);
    int touch_mode = int(rint(u.touch_dir.w));
    if (touch_mode != 0) {
        float3 orig = u.touch_orig.xyz;
        float3 dir = u.touch_dir.xyz;
        float3 to_p = p_pos - orig;
        float proj = dot(to_p, dir);
        if (proj > 0.0f) {
            float3 closest = orig + dir * proj;
            float3 diff = closest - p_pos;
            if (is_2d) diff.z = 0.0f;
            float dist = length(diff);
            const float interact_r = 0.52f;
            if (dist < interact_r && dist > 1e-4f) {
                float3 n_dir = diff / dist;
                float falloff = 1.0f - (dist / interact_r);
                float strength = u.touch_orig.w * falloff;
                if (touch_mode > 0) {
                    f_user = n_dir * strength - vel * (2.2f * falloff);
                } else {
                    f_user = -n_dir * (strength * 1.25f);
                }
            }
        }
    }

    vel += (u.gravity_dt.xyz + f_user) * dt;
    float3 pred = p_pos + vel * dt;

    float rad = u.bounds_max.w;
    float3 b_min = u.bounds_min.xyz + float3(rad, rad, is_2d ? 0.0f : rad);
    float3 b_max = u.bounds_max.xyz - float3(rad, rad, is_2d ? 0.0f : rad);
    pred = clamp(pred, b_min, b_max);
    if (is_2d) pred.z = 0.0f;

    pred_buf[gid] = float4(pred, pos_rho.w);
}

// ============================================================================
// PASS 2: Clear Spatial Hash Grid Heads
// ============================================================================
kernel void pbf_clear_grid(
    device int* grid_heads  [[buffer(0)]],
    constant SimUniforms& u [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.grid_dim.w) return;
    grid_heads[gid] = -1;
}

// ============================================================================
// PASS 3: Lock-Free Atomic Spatial Grid Insertion
// ============================================================================
kernel void pbf_insert_grid(
    const device float4* pred_buf [[buffer(0)]],
    device atomic_int* grid_heads [[buffer(1)]],
    device int* grid_next         [[buffer(2)]],
    constant SimUniforms& u       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.counts.x) return;
    int3 c = cell_coords(pred_buf[gid].xyz, u);
    int c_idx = cell_index(c, u.grid_dim.xyz);
    int prev = atomic_exchange_explicit(&grid_heads[c_idx], int(gid), memory_order_relaxed);
    grid_next[gid] = prev;
}

// ============================================================================
// PASS 4: Poly6 Density + Analytical Floor Support + Spiky Lambda
// ============================================================================
kernel void pbf_density(
    const device float4* pred_in  [[buffer(0)]],
    const device int* grid_heads  [[buffer(1)]],
    const device int* grid_next   [[buffer(2)]],
    device float4* pred_out       [[buffer(3)]],
    device float4* pos_buf        [[buffer(4)]],
    constant SimUniforms& u       [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.counts.x) return;

    float3 p_i = pred_in[gid].xyz;
    float h = u.kernel_p.x;
    float h2 = u.kernel_p.y;
    float inv_rho0 = u.kernel_p.z;
    float poly6 = u.factors.x;
    float spiky = u.factors.y;
    bool is_2d = (u.counts.y != 0);

    int3 dim = u.grid_dim.xyz;
    int3 ci = cell_coords(p_i, u);
    int3 c_min = max(ci - int3(1, 1, is_2d ? 0 : 1), int3(0));
    int3 c_max = min(ci + int3(1, 1, is_2d ? 0 : 1), dim - int3(1));

    float rho = u.extra.w;
    float3 grad_i = float3(0.0f);
    float sum_grad2 = 0.0f;

    for (int cz = c_min.z; cz <= c_max.z; ++cz) {
        for (int cy = c_min.y; cy <= c_max.y; ++cy) {
            for (int cx = c_min.x; cx <= c_max.x; ++cx) {
                int j = grid_heads[cell_index(int3(cx, cy, cz), dim)];
                int guard = 0;
                while (j != -1 && guard < 96) {
                    ++guard;
                    if (j != int(gid)) {
                        float3 r_vec = p_i - pred_in[j].xyz;
                        float r2 = dot(r_vec, r_vec);
                        if (r2 < h2 && r2 > 1e-12f) {
                            float diff = h2 - r2;
                            rho += poly6 * diff * diff * diff;

                            float r = sqrt(r2);
                            float hr = h - r;
                            float3 grad_j = (r_vec * (spiky * hr * hr / r)) * inv_rho0;
                            grad_i += grad_j;
                            sum_grad2 += dot(grad_j, grad_j);
                        }
                    }
                    j = grid_next[j];
                }
            }
        }
    }

    // Analytical mirror-floor kernel support prevents bottom crowding & popping
    float floor_y = u.bounds_min.y + u.bounds_max.w;
    float dy_floor = p_i.y - floor_y;
    if (dy_floor < h) {
        float r_wall = max(dy_floor + u.extra.y * 0.85f, u.extra.y * 0.5f);
        if (r_wall < h) {
            float diff_w = h2 - r_wall * r_wall;
            rho += 0.55f * poly6 * diff_w * diff_w * diff_w;
        }
    }

    sum_grad2 += dot(grad_i, grad_i);
    float c_val = max(rho * inv_rho0 - 1.0f, -0.04f);
    float lambda_i = -c_val / (sum_grad2 + u.solver.x);

    pred_out[gid] = float4(p_i, lambda_i);
    float3 old_p = pos_buf[gid].xyz;
    pos_buf[gid] = float4(old_p, rho);
}

// ============================================================================
// PASS 5: Fused Jacobi Position Solve + Akinci Cohesion + Obstacles
// ============================================================================
kernel void pbf_solve_collide(
    const device float4* pred_in  [[buffer(0)]],
    const device float4* pos_buf  [[buffer(1)]],
    const device int* grid_heads  [[buffer(2)]],
    const device int* grid_next   [[buffer(3)]],
    device float4* pred_out       [[buffer(4)]],
    constant SimUniforms& u       [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.counts.x) return;

    float4 self_pl = pred_in[gid];
    float3 p_i = self_pl.xyz;
    float lambda_i = self_pl.w;

    float h = u.kernel_p.x;
    float h2 = u.kernel_p.y;
    float inv_rho0 = u.kernel_p.z;
    float poly6 = u.factors.x;
    float spiky = u.factors.y;
    float inv_wq = 1.0f / max(u.factors.z, 1e-6f);
    float tensile_k = u.factors.w;
    float st_coeff = u.fluid.y;
    bool is_2d = (u.counts.y != 0);

    int3 dim = u.grid_dim.xyz;
    int3 ci = cell_coords(p_i, u);
    int3 c_min = max(ci - int3(1, 1, is_2d ? 0 : 1), int3(0));
    int3 c_max = min(ci + int3(1, 1, is_2d ? 0 : 1), dim - int3(1));

    float3 dp = float3(0.0f);
    float3 coh = float3(0.0f);

    for (int cz = c_min.z; cz <= c_max.z; ++cz) {
        for (int cy = c_min.y; cy <= c_max.y; ++cy) {
            for (int cx = c_min.x; cx <= c_max.x; ++cx) {
                int j = grid_heads[cell_index(int3(cx, cy, cz), dim)];
                int guard = 0;
                while (j != -1 && guard < 96) {
                    ++guard;
                    if (j != int(gid)) {
                        float4 nb = pred_in[j];
                        float3 r_vec = p_i - nb.xyz;
                        float r2 = dot(r_vec, r_vec);
                        if (r2 < h2 && r2 > 1e-12f) {
                            float r = sqrt(r2);
                            float lambda_sum = lambda_i + nb.w;

                            float s_corr = 0.0f;
                            if (lambda_sum < 0.0f) {
                                float diff = h2 - r2;
                                float w_ij = poly6 * diff * diff * diff;
                                float ratio = w_ij * inv_wq;
                                float ratio2 = ratio * ratio;
                                s_corr = -tensile_k * ratio2 * ratio2;
                            }

                            float hr = h - r;
                            float3 dir = r_vec / r;
                            float3 grad_w = dir * (spiky * hr * hr);
                            dp += grad_w * (lambda_sum + s_corr);

                            if (st_coeff > 1e-6f) {
                                coh -= dir * bell_cohesion(r, h);
                            }
                        }
                    }
                    j = grid_next[j];
                }
            }
        }
    }

    float3 total_dp = dp * (inv_rho0 * 0.65f) + coh * st_coeff;
    float disp_len = length(total_dp);
    if (disp_len > u.solver.y) {
        total_dp *= (u.solver.y / disp_len);
    }

    float3 new_p = p_i + total_dp;
    if (is_2d) new_p.z = 0.0f;

    // Sphere Obstacle Projection
    if (u.fluid.w > 0.5f) {
        float3 obs_c = u.obstacle.xyz;
        float obs_r = u.obstacle.w + u.bounds_max.w;
        float3 d_obs = new_p - obs_c;
        if (is_2d) d_obs.z = 0.0f;
        float dist = length(d_obs);
        if (dist < obs_r) {
            float3 n = (dist > 1e-5f) ? (d_obs / dist) : float3(0.0f, 1.0f, 0.0f);
            new_p = obs_c + n * obs_r;
            if (is_2d) new_p.z = 0.0f;
        }
    }

    float rad = u.bounds_max.w;
    float3 b_min = u.bounds_min.xyz + float3(rad, rad, is_2d ? 0.0f : rad);
    float3 b_max = u.bounds_max.xyz - float3(rad, rad, is_2d ? 0.0f : rad);
    new_p = clamp(new_p, b_min, b_max);
    if (is_2d) new_p.z = 0.0f;

    pred_out[gid] = float4(new_p, pos_buf[gid].w);
}

// ============================================================================
// PASS 6: Derive Velocity + Wall Slip + XSPH Viscosity
// ============================================================================
kernel void pbf_velocity_xsph(
    const device float4* new_pos_buf [[buffer(0)]],
    const device float4* old_pos_buf [[buffer(1)]],
    const device int* grid_heads     [[buffer(2)]],
    const device int* grid_next      [[buffer(3)]],
    device float4* vel_buf           [[buffer(4)]],
    constant SimUniforms& u          [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.counts.x) return;

    float4 np = new_pos_buf[gid];
    float3 new_p = np.xyz;
    float rho = np.w;
    float3 old_p = old_pos_buf[gid].xyz;
    float inv_dt = 1.0f / max(u.gravity_dt.w, 1e-6f);
    bool is_2d = (u.counts.y != 0);

    float3 vel = (new_p - old_p) * inv_dt;

    float rad = u.bounds_max.w;
    float eps_w = 0.0015f;
    float rest = u.solver.z;
    float slip = 1.0f - u.solver.w;

    float3 b_min = u.bounds_min.xyz + float3(rad, rad, is_2d ? 0.0f : rad);
    float3 b_max = u.bounds_max.xyz - float3(rad, rad, is_2d ? 0.0f : rad);

    if (new_p.x <= b_min.x + eps_w && vel.x < 0.0f) { vel.x = (vel.x > -0.25f) ? 0.0f : -vel.x * rest; vel.yz *= slip; }
    if (new_p.x >= b_max.x - eps_w && vel.x > 0.0f) { vel.x = (vel.x <  0.25f) ? 0.0f : -vel.x * rest; vel.yz *= slip; }
    if (new_p.y <= b_min.y + eps_w && vel.y < 0.0f) { vel.y = (vel.y > -0.35f) ? 0.0f : -vel.y * rest; vel.xz *= slip; }
    if (new_p.y >= b_max.y - eps_w && vel.y > 0.0f) { vel.y = -vel.y * rest; vel.xz *= slip; }
    if (!is_2d) {
        if (new_p.z <= b_min.z + eps_w && vel.z < 0.0f) { vel.z = (vel.z > -0.25f) ? 0.0f : -vel.z * rest; vel.xy *= slip; }
        if (new_p.z >= b_max.z - eps_w && vel.z > 0.0f) { vel.z = (vel.z <  0.25f) ? 0.0f : -vel.z * rest; vel.xy *= slip; }
    }

    // XSPH Viscosity
    float visc = u.fluid.x;
    if (visc > 1e-5f) {
        float h2 = u.kernel_p.y;
        float poly6 = u.factors.x;
        int3 dim = u.grid_dim.xyz;
        int3 ci = cell_coords(new_p, u);
        int3 c_min = max(ci - int3(1, 1, is_2d ? 0 : 1), int3(0));
        int3 c_max = min(ci + int3(1, 1, is_2d ? 0 : 1), dim - int3(1));

        float3 xsph = float3(0.0f);
        for (int cz = c_min.z; cz <= c_max.z; ++cz) {
            for (int cy = c_min.y; cy <= c_max.y; ++cy) {
                for (int cx = c_min.x; cx <= c_max.x; ++cx) {
                    int j = grid_heads[cell_index(int3(cx, cy, cz), dim)];
                    int guard = 0;
                    while (j != -1 && guard < 96) {
                        ++guard;
                        if (j != int(gid)) {
                            float4 nb_p = new_pos_buf[j];
                            float3 r_vec = new_p - nb_p.xyz;
                            float r2 = dot(r_vec, r_vec);
                            if (r2 < h2) {
                                float3 vj = (nb_p.xyz - old_pos_buf[j].xyz) * inv_dt;
                                float diff = h2 - r2;
                                float w_ij = poly6 * diff * diff * diff;
                                xsph += (vj - vel) * (w_ij / max(nb_p.w, u.kernel_p.w * 0.25f));
                            }
                        }
                        j = grid_next[j];
                    }
                }
            }
        }
        vel += xsph * visc;
    }

    if (is_2d) vel.z = 0.0f;
    float spd = length(vel);
    if (spd > u.fluid.z) {
        vel *= (u.fluid.z / spd);
    }

    float pres_norm = clamp((rho * u.kernel_p.z - 1.0f) * 3.5f, 0.0f, 1.0f);
    vel_buf[gid] = float4(vel, pres_norm);
}

// ============================================================================
// PASS 7: Compute Velocity Curl (Vorticity)
// ============================================================================
kernel void pbf_curl(
    const device float4* pos_buf [[buffer(0)]],
    const device float4* vel_buf [[buffer(1)]],
    const device int* grid_heads [[buffer(2)]],
    const device int* grid_next  [[buffer(3)]],
    device float4* curl_buf      [[buffer(4)]],
    constant SimUniforms& u      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.counts.x) return;

    float3 p_i = pos_buf[gid].xyz;
    float3 v_i = vel_buf[gid].xyz;
    float h = u.kernel_p.x;
    float h2 = u.kernel_p.y;
    float spiky = u.factors.y;
    float inv_rho0 = u.kernel_p.z;
    bool is_2d = (u.counts.y != 0);

    int3 dim = u.grid_dim.xyz;
    int3 ci = cell_coords(p_i, u);
    int3 c_min = max(ci - int3(1, 1, is_2d ? 0 : 1), int3(0));
    int3 c_max = min(ci + int3(1, 1, is_2d ? 0 : 1), dim - int3(1));

    float3 omega = float3(0.0f);
    for (int cz = c_min.z; cz <= c_max.z; ++cz) {
        for (int cy = c_min.y; cy <= c_max.y; ++cy) {
            for (int cx = c_min.x; cx <= c_max.x; ++cx) {
                int j = grid_heads[cell_index(int3(cx, cy, cz), dim)];
                int guard = 0;
                while (j != -1 && guard < 96) {
                    ++guard;
                    if (j != int(gid)) {
                        float3 r_vec = p_i - pos_buf[j].xyz;
                        float r2 = dot(r_vec, r_vec);
                        if (r2 < h2 && r2 > 1e-12f) {
                            float r = sqrt(r2);
                            float hr = h - r;
                            float3 grad_w = (r_vec * (spiky * hr * hr / r)) * inv_rho0;
                            float3 v_ij = vel_buf[j].xyz - v_i;
                            omega += cross(v_ij, grad_w);
                        }
                    }
                    j = grid_next[j];
                }
            }
        }
    }
    curl_buf[gid] = float4(omega, length(omega));
}

// ============================================================================
// PASS 8: Flow-Gated Macklin & Mueller Vorticity Confinement
// ============================================================================
kernel void pbf_vorticity(
    const device float4* pos_buf  [[buffer(0)]],
    const device float4* curl_buf [[buffer(1)]],
    const device int* grid_heads  [[buffer(2)]],
    const device int* grid_next   [[buffer(3)]],
    device float4* vel_buf        [[buffer(4)]],
    constant SimUniforms& u       [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= u.counts.x) return;

    float4 vp = vel_buf[gid];
    float3 vel = vp.xyz;
    float speed = length(vel);
    if (speed < 0.08f) return;

    float4 self_c = curl_buf[gid];
    float3 omega = self_c.xyz;
    if (self_c.w < 1e-4f) return;

    float3 p_i = pos_buf[gid].xyz;
    float h = u.kernel_p.x;
    float h2 = u.kernel_p.y;
    float spiky = u.factors.y;
    bool is_2d = (u.counts.y != 0);

    int3 dim = u.grid_dim.xyz;
    int3 ci = cell_coords(p_i, u);
    int3 c_min = max(ci - int3(1, 1, is_2d ? 0 : 1), int3(0));
    int3 c_max = min(ci + int3(1, 1, is_2d ? 0 : 1), dim - int3(1));

    float3 eta = float3(0.0f);
    for (int cz = c_min.z; cz <= c_max.z; ++cz) {
        for (int cy = c_min.y; cy <= c_max.y; ++cy) {
            for (int cx = c_min.x; cx <= c_max.x; ++cx) {
                int j = grid_heads[cell_index(int3(cx, cy, cz), dim)];
                int guard = 0;
                while (j != -1 && guard < 96) {
                    ++guard;
                    if (j != int(gid)) {
                        float3 r_vec = p_i - pos_buf[j].xyz;
                        float r2 = dot(r_vec, r_vec);
                        if (r2 < h2 && r2 > 1e-12f) {
                            float r = sqrt(r2);
                            float hr = h - r;
                            float3 grad_w = r_vec * (-spiky * hr * hr / r);
                            eta += grad_w * curl_buf[j].w;
                        }
                    }
                    j = grid_next[j];
                }
            }
        }
    }

    float eta_len = length(eta);
    if (eta_len > 1e-5f) {
        float3 n = eta / eta_len;
        float flow_gate = smoothstep(0.08f, 0.42f, speed);
        float3 dv = cross(n, omega) * (u.extra.x * flow_gate * u.gravity_dt.w);
        if (is_2d) dv.z = 0.0f;
        float max_dv = speed * 0.22f;
        float dv_len = length(dv);
        if (dv_len > max_dv && dv_len > 1e-6f) {
            dv *= (max_dv / dv_len);
        }
        vel_buf[gid] = float4(vel + dv, vp.w);
    }
}

// ============================================================================
// ZERO-COPY INSTANCED PARTICLE & WIREFRAME RENDER SHADERS
// ============================================================================

struct ParticleVertOut {
    float4 position [[position]];
    float2 uv;
    float3 pos_eye;
    float3 world_pos;
    float  metric_t;
    float  speed_t;
};

vertex ParticleVertOut particle_vert(
    const device float4* pos_buf [[buffer(0)]],
    const device float4* vel_buf [[buffer(1)]],
    constant RenderUniforms& u   [[buffer(2)]],
    uint vid [[vertex_id]],
    uint iid [[instance_id]]
) {
    const float2 corners[4] = {
        float2(-1.0f, -1.0f),
        float2( 1.0f, -1.0f),
        float2(-1.0f,  1.0f),
        float2( 1.0f,  1.0f)
    };

    float4 pos_rho = pos_buf[iid];
    float4 vel_pres = vel_buf[iid];

    ParticleVertOut out;
    float2 uv = corners[vid];
    out.uv = uv;
    out.world_pos = pos_rho.xyz;

    float speed = length(vel_pres.xyz);
    out.speed_t = min(speed / 2.8f, 1.0f);

    int color_mode = u.modes.x;
    if (color_mode == 0) {
        out.metric_t = min(speed / 2.5f, 1.0f);
    } else if (color_mode == 1) {
        out.metric_t = saturate(vel_pres.w);
    } else if (color_mode == 2) {
        float dev = abs(pos_rho.w * u.params.y - 1.0f);
        out.metric_t = min(dev * 4.0f, 1.0f);
    } else {
        out.metric_t = out.speed_t;
    }

    float4 eye_center = u.view * float4(pos_rho.xyz, 1.0f);
    out.pos_eye = eye_center.xyz;

    float3 quad_eye = eye_center.xyz + float3(uv * u.params.x, 0.0f);
    out.position = u.proj * float4(quad_eye, 1.0f);
    return out;
}

inline float3 colormap_ocean(float t) {
    float3 c0 = float3(0.035f, 0.140f, 0.360f);
    float3 c1 = float3(0.060f, 0.440f, 0.780f);
    float3 c2 = float3(0.180f, 0.780f, 0.950f);
    float3 c3 = float3(0.920f, 0.980f, 1.000f);
    if (t < 0.38f) return mix(c0, c1, t / 0.38f);
    if (t < 0.76f) return mix(c1, c2, (t - 0.38f) / 0.38f);
    return mix(c2, c3, (t - 0.76f) / 0.24f);
}

inline float3 colormap_turbo(float t) {
    const float4 kR4 = float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
    const float4 kG4 = float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
    const float4 kB4 = float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
    const float2 kR2 = float2(-152.94239396f, 59.28637943f);
    const float2 kG2 = float2(4.27729857f, 2.82956604f);
    const float2 kB2 = float2(-89.90310912f, 27.34824973f);
    float x = saturate(t);
    float4 v4 = float4(1.0f, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;
    return saturate(float3(
        dot(v4, kR4) + dot(v2, kR2),
        dot(v4, kG4) + dot(v2, kG2),
        dot(v4, kB4) + dot(v2, kB2)
    ));
}

inline float3 colormap_viridis(float t) {
    float3 c0 = float3(0.267f, 0.005f, 0.329f);
    float3 c1 = float3(0.191f, 0.407f, 0.556f);
    float3 c2 = float3(0.208f, 0.718f, 0.472f);
    float3 c3 = float3(0.993f, 0.906f, 0.144f);
    if (t < 0.33f) return mix(c0, c1, t / 0.33f);
    if (t < 0.66f) return mix(c1, c2, (t - 0.33f) / 0.33f);
    return mix(c2, c3, (t - 0.66f) / 0.34f);
}

inline float3 colormap_fire(float t) {
    float3 c0 = float3(0.08f, 0.02f, 0.12f);
    float3 c1 = float3(0.72f, 0.12f, 0.22f);
    float3 c2 = float3(0.98f, 0.56f, 0.08f);
    float3 c3 = float3(1.00f, 0.96f, 0.72f);
    if (t < 0.35f) return mix(c0, c1, t / 0.35f);
    if (t < 0.72f) return mix(c1, c2, (t - 0.35f) / 0.37f);
    return mix(c2, c3, (t - 0.72f) / 0.28f);
}

inline float4 shade_particle_core(ParticleVertOut in, float z, constant RenderUniforms& u, bool is_3d_mode) {
    float radius = u.params.x;
    float tex_str = saturate(u.params.z);
    float t_anim = u.params.w;

    int color_mode = u.modes.x;
    int color_map = u.modes.y;
    int lit_spheres = u.modes.z;
    int tex_mode = u.modes.w;

    float3 n = float3(in.uv.x, in.uv.y, z);
    float3 surf_pos = in.world_pos + n * (radius * 1.35f);

    float caustic_boost = 0.0f;
    float foam_tex = 0.0f;
    float tex_shade = 1.0f;

    if (tex_mode != 0 && tex_str > 0.005f) {
        if (tex_mode == 1) {
            // Capillary Ripples
            float w1 = sin(surf_pos.x * 22.0f + surf_pos.y * 15.0f - t_anim * 4.2f);
            float w2 = cos(surf_pos.z * 20.0f - surf_pos.x * 14.0f + surf_pos.y * 12.0f + t_anim * 3.4f);
            float ripple = 0.5f * (w1 + w2);
            n = normalize(n + float3(w1, w2, 0.0f) * (tex_str * 0.85f));
            tex_shade = 1.0f + ripple * (0.45f * tex_str);
            caustic_boost = pow(max(ripple, 0.0f), 1.8f) * (0.95f * tex_str);
        } else if (tex_mode == 2) {
            // Luminous Refractive Caustic Web
            float3 p = surf_pos * 13.5f;
            float c1 = abs(sin(p.x + sin(p.y * 1.4f + t_anim * 2.6f)) * cos(p.y + cos(p.z * 1.5f - t_anim * 2.1f)));
            float web = pow(1.0f - saturate(c1), 2.4f);
            tex_shade = 1.0f - tex_str * 0.42f;
            caustic_boost = web * (1.35f * tex_str);
            n = normalize(n + float3(sin(p.x * 1.9f), cos(p.y * 1.9f), 0.0f) * (tex_str * 0.45f));
        } else if (tex_mode == 3) {
            // Aerated Seafoam
            float3 p = surf_pos * 19.0f + float3(t_anim * 1.8f, -t_anim * 0.9f, t_anim * 1.4f);
            float f = (sin(p.x) * cos(p.y) + sin(p.y * 1.7f + p.z) + cos(p.z * 2.3f - p.x)) * 0.333f + 0.5f;
            float cell = smoothstep(0.32f, 0.72f, f);
            foam_tex = cell * saturate(tex_str * 1.35f) * (0.55f + 0.45f * in.speed_t);
            n = normalize(n + float3(f - 0.5f, (1.0f - f) - 0.5f, 0.0f) * (tex_str * 0.40f));
        } else if (tex_mode == 4) {
            // Spatial Lagrangian Grid
            float freq = 5.5f;
            float gx = abs(fract(surf_pos.x * freq) - 0.5f);
            float gy = abs(fract(surf_pos.y * freq) - 0.5f);
            float line_xy = max(smoothstep(0.42f, 0.49f, gx), smoothstep(0.42f, 0.49f, gy));
            if (is_3d_mode) {
                float gz = abs(fract(surf_pos.z * freq) - 0.5f);
                line_xy = max(line_xy, smoothstep(0.42f, 0.49f, gz));
            }
            float checker = float((int(floor(surf_pos.x * freq)) + int(floor(surf_pos.y * freq))) & 1);
            tex_shade = 1.0f - tex_str * (0.28f + 0.22f * checker);
            caustic_boost = line_xy * (1.25f * tex_str);
        }
    }

    float3 shallow_col = u.shallow_refr.xyz;
    float3 deep_col = u.deep_pad.xyz;
    float metallic = saturate(u.optics.w);

    float3 base_color;
    if (color_mode == 3) {
        float rim = 1.0f - z;
        float shallow_mix = saturate(0.28f + 0.48f * in.speed_t + 0.42f * rim);
        base_color = mix(deep_col, shallow_col, shallow_mix);
        float crest_foam = smoothstep(0.62f, 0.98f, in.speed_t) * u.light_foam.w * (1.0f - metallic);
        base_color = mix(base_color, float3(0.94f, 0.98f, 1.0f), crest_foam * 0.72f);
    } else {
        if (color_map == 1) base_color = colormap_turbo(in.metric_t);
        else if (color_map == 2) base_color = colormap_viridis(in.metric_t);
        else if (color_map == 3) base_color = colormap_fire(in.metric_t);
        else base_color = colormap_ocean(in.metric_t);
    }

    float3 boost_col = mix(shallow_col, float3(0.96f, 0.99f, 1.0f), 0.55f) * caustic_boost;
    base_color = base_color * tex_shade + boost_col;
    if (foam_tex > 0.005f) {
        base_color = mix(base_color, float3(0.96f, 0.99f, 1.0f), clamp(foam_tex, 0.0f, 0.90f));
    }

    float3 rgb = base_color;
    if (lit_spheres != 0) {
        float3 light_dir = u.light_foam.xyz;
        float ior = max(u.optics.x, 1.01f);
        float rough = clamp(u.optics.y, 0.015f, 0.95f);
        float spec_mult = max(u.optics.z, 0.0f);
        float refr_glow = clamp(u.shallow_refr.w, 0.0f, 1.5f);

        float f0_diel = pow((ior - 1.0f) / (ior + 1.0f), 2.0f);
        float3 f0 = mix(float3(f0_diel), base_color, metallic);
        float ndotv = max(n.z, 0.001f);
        float3 fresnel = f0 + (float3(1.0f) - f0) * pow(1.0f - ndotv, 4.2f);

        float ndotl = max(dot(n, light_dir), 0.0f);
        float hemi = 0.5f + 0.5f * n.y;
        float3 half_dir = normalize(light_dir + float3(0.0f, 0.0f, 1.0f));
        float spec_pow = clamp(1.8f / (rough * rough + 0.003f), 8.0f, 420.0f);
        float spec_energy = (0.28f + 0.72f * (1.0f - rough)) * spec_mult;
        float3 spec = pow(max(dot(n, half_dir), 0.0f), spec_pow) * spec_energy * fresnel;

        float refl_y = saturate(0.5f + 0.5f * n.y);
        float3 sky_env = mix(float3(0.08f, 0.14f, 0.24f), float3(0.55f, 0.78f, 0.98f), pow(refl_y, 1.5f));
        float3 body_lit = base_color * (0.28f + 0.18f * hemi + 0.62f * ndotl * (1.0f - 0.65f * metallic));

        float sss_rim = pow(1.0f - ndotv, 2.2f) * refr_glow * (1.0f - metallic);
        float3 sss_tint = shallow_col * (0.35f + 0.65f * (0.5f + 0.5f * dot(n, -light_dir))) * sss_rim;

        rgb = mix(body_lit, sky_env * mix(float3(1.0f), base_color, metallic), clamp(fresnel * (0.75f + 0.25f * metallic), 0.0f, 0.96f))
            + spec
            + sss_tint
            + boost_col * 0.45f;
    }

    return float4(saturate(rgb), 1.0f);
}

// 2D Planar Fragment Shader (preserves Apple TBDR Early-Z / HSR)
fragment float4 particle_frag_2d(
    ParticleVertOut in         [[stage_in]],
    constant RenderUniforms& u [[buffer(0)]]
) {
    float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0f) {
        discard_fragment();
    }
    float z = sqrt(max(1.0f - r2, 0.0f));
    return shade_particle_core(in, z, u, false);
}

struct Frag3DOut {
    float4 color [[color(0)]];
    float  depth [[depth(any)]];
};

// 3D Volumetric Fragment Shader (exact spherical depth in [0, 1] Metal NDC)
fragment Frag3DOut particle_frag_3d(
    ParticleVertOut in         [[stage_in]],
    constant RenderUniforms& u [[buffer(0)]]
) {
    float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0f) {
        discard_fragment();
    }
    float z = sqrt(max(1.0f - r2, 0.0f));

    Frag3DOut out;
    out.color = shade_particle_core(in, z, u, true);

    float radius = u.params.x;
    float3 sphere_eye = in.pos_eye + float3(in.uv * radius, z * radius);
    float4 clip_pos = u.proj * float4(sphere_eye, 1.0f);
    out.depth = saturate(clip_pos.z / clip_pos.w);
    return out;
}

// Wireframe Chamber / Floor Grid / Sphere Obstacle Shader
struct LineVertOut {
    float4 position [[position]];
};

vertex LineVertOut line_vert(
    const device float4* verts  [[buffer(0)]],
    constant RenderUniforms& u  [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    LineVertOut out;
    out.position = u.view_proj * float4(verts[vid].xyz, 1.0f);
    return out;
}

fragment float4 line_frag(
    LineVertOut in             [[stage_in]],
    constant RenderUniforms& u [[buffer(0)]]
) {
    return u.line_color;
}
