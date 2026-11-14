// Position-Based Fluids (PBF) Compute Shader for Liquid-Rust
// Implements Macklin & Müller (2013) with tensile instability correction and XSPH viscosity

struct GpuCollider {
    info: vec4<u32>,        // x: kind (1: sphere, 2: box, 3: cylinder), y: enabled, zw: pad
    pos_rad: vec4<f32>,     // xyz: center, w: radius
    extents: vec4<f32>,     // xyz: half extents, w: height
};

struct SimulationUniforms {
    bounds_min: vec4<f32>,       // xyz = min, w = cell_size
    bounds_max: vec4<f32>,       // xyz = max, w = particle_radius
    gravity: vec4<f32>,          // xyz = gravity, w = dt
    grid_dim: vec4<u32>,         // xyz = dim, w = total_cells
    
    mouse_origin: vec4<f32>,     // xyz = origin, w = mouse_strength
    mouse_dir: vec4<f32>,        // xyz = dir, w = f32(mouse_mode)
    
    collider_pos: vec4<f32>,     // xyz = pos, w = collider_radius
    
    // [h, h2, inv_rho0, rho0]
    kernel_params: vec4<f32>,
    // [poly6_factor, spiky_factor, w_q, tensile_k]
    kernel_factors: vec4<f32>,
    // [relaxation_epsilon, max_displacement, restitution, boundary_friction]
    solver_params: vec4<f32>,
    // [viscosity, surface_tension, max_velocity, collider_enabled]
    fluid_params: vec4<f32>,
    
    // [particle_count, solver_iterations, substeps, num_colliders]
    counts: vec4<u32>,
    colliders: array<GpuCollider, 16>,
    chamber_info: vec4<f32>,
    boundary_params: vec4<f32>,
};

struct Particle {
    pos_type: vec4<f32>, // xyz = position, w = particle_type (0 = fluid, 1 = boundary)
    vel_dens: vec4<f32>, // xyz = velocity, w = density
};

@group(0) @binding(0) var<uniform> uniforms: SimulationUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> predicted_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> old_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(5) var<storage, read_write> delta_positions: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> grid_heads: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> grid_next: array<u32>;

fn poly6(r2: f32, h2: f32, poly6_factor: f32) -> f32 {
    if (r2 < h2 && r2 >= 0.0) {
        let diff = h2 - r2;
        return poly6_factor * diff * diff * diff;
    }
    return 0.0;
}

fn spiky_gradient(r_vec: vec3<f32>, r: f32, h: f32, spiky_factor: f32) -> vec3<f32> {
    if (r < h && r > 1e-6) {
        let diff = h - r;
        let factor = -spiky_factor * diff * diff / r;
        return r_vec * factor;
    }
    return vec3<f32>(0.0);
}

fn safe_clamp(val: f32, lo: f32, hi: f32) -> f32 {
    return select(0.5 * (lo + hi), clamp(val, lo, hi), lo <= hi);
}

fn resolve_chamber_gpu(pos_in: vec3<f32>, radius: f32) -> vec3<f32> {
    var pos = pos_in;
    let shape = u32(uniforms.chamber_info.x + 0.5);
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    
    if (shape == 1u) {
        // Sphere Chamber
        let r_chamber = (b_max.y - b_min.y) * 0.5;
        let center = vec3<f32>((b_min.x + b_max.x) * 0.5, b_min.y + r_chamber, (b_min.z + b_max.z) * 0.5);
        let to_p = pos - center;
        let dist = length(to_p);
        let max_d = max(r_chamber - radius, 0.01);
        if (dist > max_d && dist > 1e-6) {
            pos = center + (to_p / dist) * max_d;
        }
    } else if (shape == 2u) {
        // Bathtub Chamber
        let b_min_eff = b_min + vec3<f32>(radius);
        let b_max_eff = b_max - vec3<f32>(radius);
        let H = max(b_max.y - b_min.y, 1e-4);
        let slope_offset = uniforms.chamber_info.w;
        let fillet = uniforms.chamber_info.z;
        
        // 1. Floor and ceiling
        pos.y = safe_clamp(pos.y, b_min_eff.y, b_max_eff.y);
        
        // 2. Sloped backrest at -X and front wall at +X
        let norm_y = clamp((pos.y - b_min.y) / H, 0.0, 1.0);
        let x_back = b_min_eff.x + (1.0 - norm_y) * slope_offset;
        let x_front = b_max_eff.x;
        pos.x = safe_clamp(pos.x, x_back, x_front);
        
        if (!is_2d) {
            // 3. Side walls at Z
            let half_w = (b_max.z - b_min.z) * 0.5;
            let cz = (b_min.z + b_max.z) * 0.5;
            let z_rel = pos.z - cz;
            let z_clamped = safe_clamp(z_rel, -half_w + radius, half_w - radius);
            pos.z = cz + z_clamped;
            
            // 4. Fillets where floor meets side walls
            if (fillet > radius) {
                let eff_fillet = fillet - radius;
                let dy = (pos.y - b_min.y) - fillet;
                let abs_z = abs(pos.z - cz);
                let dz = abs_z - (half_w - fillet);
                if (dy < 0.0 && dz > 0.0) {
                    let d = sqrt(dy * dy + dz * dz);
                    if (d > eff_fillet && d > 1e-6) {
                        let factor = eff_fillet / d;
                        pos.y = b_min.y + fillet + dy * factor;
                        let new_z = (half_w - fillet) + dz * factor;
                        pos.z = cz + sign(pos.z - cz) * new_z;
                    }
                }
            }
        }
    } else {
        // Box Chamber
        pos.x = safe_clamp(pos.x, b_min.x + radius, b_max.x - radius);
        pos.y = safe_clamp(pos.y, b_min.y + radius, b_max.y - radius);
        if (!is_2d) {
            pos.z = safe_clamp(pos.z, b_min.z + radius, b_max.z - radius);
        }
    }
    return pos;
}

struct BoundaryContrib {
    psi: f32,
    grad: vec3<f32>,
};

fn fold_wall_gpu(
    bc_in: BoundaryContrib,
    d_clearance: f32,
    grad_d: vec3<f32>,
    d_offset: f32,
    h: f32,
    inv_h: f32,
    gamma_w: f32,
) -> BoundaryContrib {
    var bc = bc_in;
    let d_eff = max(max(d_clearance, 0.0) + d_offset, 0.25 * h);
    if (d_eff < h) {
        let q = d_eff * inv_h;
        let omq2 = 1.0 - q * q;
        let omq4 = (omq2 * omq2) * (omq2 * omq2);
        let psi_w = min(gamma_w * omq4, 0.65);

        let omq = 1.0 - q;
        let grad_mag = gamma_w * (6.1 * inv_h) * (omq * omq);
        let g_w = -grad_d * grad_mag;

        bc.grad = bc.grad * (1.0 - psi_w) + g_w * (1.0 - bc.psi);
        bc.psi = bc.psi + psi_w - bc.psi * psi_w;
    }
    return bc;
}

fn fold_chamber_wall_gpu(
    bc_in: BoundaryContrib,
    d_clearance: f32,
    n_inward: vec3<f32>,
    s: f32,
    xi: f32,
    grad_xi: vec3<f32>,
    rough_amp: f32,
    h: f32,
    inv_h: f32,
    gamma_w: f32,
) -> BoundaryContrib {
    // Restrict anti-crystallization micro-texture to upward-facing floor surfaces so vertical
    // container walls remain hydrodynamically smooth with zero vertical washboard drag.
    let floor_w = clamp(n_inward.y, 0.0, 1.0);
    let eff_amp = rough_amp * floor_w;
    let d_offset = (1.0 - eff_amp * xi) * s;
    let grad_bump = -grad_xi * eff_amp;
    return fold_wall_gpu(bc_in, d_clearance, n_inward + grad_bump, d_offset, h, inv_h, gamma_w);
}

fn wall_cohesion_weight_gpu(pos: vec3<f32>) -> f32 {
    let radius = uniforms.bounds_max.w;
    let h = max(uniforms.kernel_params.x, 1e-5);
    let inv_h = 1.0 / h;
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    let shape = u32(uniforms.chamber_info.x + 0.5);

    var d_floor = (pos.y - b_min.y) - radius;
    var d_wall = 1e6;

    if (shape == 1u) {
        let r_chamber = (b_max.y - b_min.y) * 0.5;
        let center = vec3<f32>((b_min.x + b_max.x) * 0.5, b_min.y + r_chamber, (b_min.z + b_max.z) * 0.5);
        let to_p = pos - center;
        let dist = length(to_p);
        let n_in_y = select(1.0, -to_p.y / dist, dist > 1e-6);
        let d_sph = (r_chamber - radius) - dist;
        if (n_in_y >= 0.70) {
            d_floor = d_sph;
        } else {
            d_wall = d_sph;
        }
    } else if (shape == 2u) {
        let H_tub = max(b_max.y - b_min.y, 1e-4);
        let slope_offset = uniforms.chamber_info.w;
        let back_n = normalize(vec3<f32>(H_tub, slope_offset, 0.0));
        let p_back_bot = vec3<f32>(b_min.x + slope_offset, b_min.y, 0.0);
        let d_back = dot(pos - p_back_bot, back_n) - radius;
        d_wall = min(d_back, min(b_max.x - pos.x, b_max.y - pos.y) - radius);
        if (!is_2d) {
            d_wall = min(d_wall, min(pos.z - b_min.z, b_max.z - pos.z) - radius);
        }
    } else {
        d_wall = min(min(pos.x - b_min.x, b_max.x - pos.x), b_max.y - pos.y) - radius;
        if (!is_2d) {
            d_wall = min(d_wall, min(pos.z - b_min.z, b_max.z - pos.z) - radius);
        }
    }

    let t_floor = clamp(d_floor * inv_h, 0.0, 1.0);
    let w_floor = mix(0.25, 1.0, t_floor * t_floor * (3.0 - 2.0 * t_floor));

    let t_side = clamp((d_wall - 0.55 * h) / (1.20 * h), 0.0, 1.0);
    let w_side = t_side * t_side * (3.0 - 2.0 * t_side);

    return w_floor * w_side;
}

fn boundary_contribution_gpu(pos: vec3<f32>) -> BoundaryContrib {
    let radius = uniforms.bounds_max.w;
    let h = max(uniforms.kernel_params.x, 1e-5);
    let inv_h = 1.0 / h;
    let s = max(uniforms.boundary_params.x, 1e-4);
    let inv_s = 1.0 / s;
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;

    let px = pos.x * inv_s;
    let py = pos.y * inv_s;
    let pz = pos.z * inv_s;
    let a1 = 4.71 * px + 2.31 * pz;
    let a2 = 3.83 * py - 4.19 * px;
    let a3 = select(5.17 * pz - 3.41 * py, 3.13 * px + 4.47 * py, is_2d);
    let xi = (sin(a1) + sin(a2) + sin(a3)) * (1.0 / 3.0);
    let c1 = cos(a1);
    let c2 = cos(a2);
    let c3 = cos(a3);
    var grad_xi = vec3<f32>(0.0);
    if (is_2d) {
        grad_xi = vec3<f32>(
            (4.71 * c1 - 4.19 * c2 + 3.13 * c3) * (1.0 / 3.0),
            (3.83 * c2 + 4.47 * c3) * (1.0 / 3.0),
            0.0,
        );
    } else {
        grad_xi = vec3<f32>(
            (4.71 * c1 - 4.19 * c2) * (1.0 / 3.0),
            (3.83 * c2 - 3.41 * c3) * (1.0 / 3.0),
            (2.31 * c1 + 5.17 * c3) * (1.0 / 3.0),
        );
    }
    let rough_amp = select(0.055, 0.025, is_2d);
    let gamma_w = select(1.20, 1.42, is_2d) * (s * inv_h);

    var bc = BoundaryContrib(0.0, vec3<f32>(0.0));
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    let shape = u32(uniforms.chamber_info.x + 0.5);

    if (shape == 1u) {
        let r_chamber = (b_max.y - b_min.y) * 0.5;
        let center = vec3<f32>((b_min.x + b_max.x) * 0.5, b_min.y + r_chamber, (b_min.z + b_max.z) * 0.5);
        let to_p = pos - center;
        let dist = length(to_p);
        let n_in = select(vec3<f32>(0.0, 1.0, 0.0), -to_p / max(dist, 1e-6), dist > 1e-6);
        let d_c = (r_chamber - radius) - dist;
        bc = fold_chamber_wall_gpu(bc, d_c, n_in, s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
    } else if (shape == 2u) {
        let H_tub = max(b_max.y - b_min.y, 1e-4);
        let slope_offset = uniforms.chamber_info.w;
        let back_n = normalize(vec3<f32>(H_tub, slope_offset, 0.0));
        let p_back_bot = vec3<f32>(b_min.x + slope_offset, b_min.y, 0.0);
        let d_back = dot(pos - p_back_bot, back_n) - radius;

        bc = fold_chamber_wall_gpu(bc, (pos.y - b_min.y) - radius, vec3<f32>(0.0, 1.0, 0.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        bc = fold_chamber_wall_gpu(bc, (b_max.y - pos.y) - radius, vec3<f32>(0.0, -1.0, 0.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        bc = fold_chamber_wall_gpu(bc, d_back, back_n, s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        bc = fold_chamber_wall_gpu(bc, (b_max.x - pos.x) - radius, vec3<f32>(-1.0, 0.0, 0.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        if (!is_2d) {
            bc = fold_chamber_wall_gpu(bc, (pos.z - b_min.z) - radius, vec3<f32>(0.0, 0.0, 1.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
            bc = fold_chamber_wall_gpu(bc, (b_max.z - pos.z) - radius, vec3<f32>(0.0, 0.0, -1.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        }
    } else {
        bc = fold_chamber_wall_gpu(bc, (pos.y - b_min.y) - radius, vec3<f32>(0.0, 1.0, 0.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        bc = fold_chamber_wall_gpu(bc, (b_max.y - pos.y) - radius, vec3<f32>(0.0, -1.0, 0.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        bc = fold_chamber_wall_gpu(bc, (pos.x - b_min.x) - radius, vec3<f32>(1.0, 0.0, 0.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        bc = fold_chamber_wall_gpu(bc, (b_max.x - pos.x) - radius, vec3<f32>(-1.0, 0.0, 0.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        if (!is_2d) {
            bc = fold_chamber_wall_gpu(bc, (pos.z - b_min.z) - radius, vec3<f32>(0.0, 0.0, 1.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
            bc = fold_chamber_wall_gpu(bc, (b_max.z - pos.z) - radius, vec3<f32>(0.0, 0.0, -1.0), s, xi, grad_xi, rough_amp, h, inv_h, gamma_w);
        }
    }

    let d_obs_offset = s;
    let num_colliders = min(uniforms.counts.w & 0x7FFFFFFFu, 16u);
    for (var c = 0u; c < num_colliders; c++) {
        let col = uniforms.colliders[c];
        if (col.info.y == 0u) { continue; }
        let kind = col.info.x;
        if (kind == 1u) {
            let to_p = pos - col.pos_rad.xyz;
            let dist = length(to_p);
            if (dist > 1e-6) {
                let d_c = dist - (col.pos_rad.w + radius);
                if (d_c < h) {
                    bc = fold_wall_gpu(bc, d_c, to_p / dist, d_obs_offset, h, inv_h, gamma_w);
                }
            }
        } else if (kind == 2u) {
            let h_size = col.extents.xyz + vec3<f32>(radius);
            let inf_min = col.pos_rad.xyz - h_size;
            let inf_max = col.pos_rad.xyz + h_size;
            let cl = clamp(pos, inf_min, inf_max);
            let diff = pos - cl;
            let dist = length(diff);
            if (dist > 1e-6 && dist < h) {
                bc = fold_wall_gpu(bc, dist, diff / dist, d_obs_offset, h, inv_h, gamma_w);
            }
        } else if (kind == 3u) {
            let c_center = col.pos_rad.xyz;
            let eff_r = col.pos_rad.w + radius;
            let half_h = col.extents.w * 0.5 + radius;
            if (pos.y >= c_center.y - half_h - h && pos.y <= c_center.y + half_h + h) {
                let dx = pos.x - c_center.x;
                let dz = pos.z - c_center.z;
                let r_xz = sqrt(dx * dx + dz * dz);
                if (r_xz > 1e-6) {
                    let dr = max(r_xz - eff_r, 0.0);
                    let dy = max(abs(pos.y - c_center.y) - half_h, 0.0);
                    let dist = sqrt(dr * dr + dy * dy);
                    if (dist < h) {
                        var n = vec3<f32>(dx / r_xz, 0.0, dz / r_xz);
                        if (dy > dr && dy > 1e-5) {
                            n = vec3<f32>(0.0, sign(pos.y - c_center.y), 0.0);
                        }
                        bc = fold_wall_gpu(bc, dist, n, d_obs_offset, h, inv_h, gamma_w);
                    }
                }
            }
        }
    }

    if (is_2d) {
        bc.grad.z = 0.0;
    }
    return bc;
}

// 1. Predict positions with gravity & user interaction
@compute @workgroup_size(256)
fn predict_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    let p_pos = particles[i].pos_type.xyz;
    let p_type = particles[i].pos_type.w;
    var vel = particles[i].vel_dens.xyz;
    let dt = uniforms.gravity.w;
    
    old_positions[i] = vec4<f32>(p_pos, 0.0);
    
    if (p_type > 0.5) {
        // Boundary particle: fixed
        predicted_positions[i] = vec4<f32>(p_pos, 0.0);
        return;
    }
    
    // Mouse force
    var f_user = vec3<f32>(0.0);
    let mouse_mode = i32(uniforms.mouse_dir.w);
    if (mouse_mode != 0) {
        let m_orig = uniforms.mouse_origin.xyz;
        let m_dir = uniforms.mouse_dir.xyz;
        let to_p = p_pos - m_orig;
        let proj = dot(to_p, m_dir);
        if (proj > 0.0) {
            let closest = m_orig + m_dir * proj;
            let diff = closest - p_pos;
            let dist = length(diff);
            if (dist < 0.6 && dist > 1e-4) {
                let dir = diff / dist;
                let factor = (1.0 - dist / 0.6) * uniforms.mouse_origin.w;
                f_user = dir * (factor * f32(mouse_mode));
            }
        }
    }
    
    let wave_accel_x = uniforms.boundary_params.z;
    var f_wave = vec3<f32>(0.0);
    if (abs(wave_accel_x) > 1e-5) {
        let zone_w = max(uniforms.boundary_params.w, 0.1);
        let paddle_w = 0.30 + 0.70 * clamp(1.0 - (p_pos.x - uniforms.bounds_min.x) / zone_w, 0.0, 1.0);
        f_wave.x = wave_accel_x * paddle_w;
    }

    let accel = uniforms.gravity.xyz + f_user + f_wave;
    vel += accel * dt;
    var pred = p_pos + vel * dt;
    
    let radius = uniforms.bounds_max.w;
    pred = resolve_chamber_gpu(pred, radius);
    
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    if (is_2d) {
        pred.z = 0.0;
        vel.z = 0.0;
    }

    predicted_positions[i] = vec4<f32>(pred, wall_cohesion_weight_gpu(pred));
    particles[i].vel_dens = vec4<f32>(vel, particles[i].vel_dens.w);
}

// 2. Clear uniform grid heads
@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if (cell_idx < uniforms.grid_dim.w) {
        atomicStore(&grid_heads[cell_idx], 0u);
    }
}

// 3. Insert predicted particle positions into 3D uniform grid
@compute @workgroup_size(256)
fn insert_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x + id.y * 16776960u;
    if (p_idx >= uniforms.counts.x) {
        return;
    }
    
    let pos = predicted_positions[p_idx].xyz;
    let cell_size = uniforms.bounds_min.w;
    let b_min = uniforms.bounds_min.xyz;
    let dim = uniforms.grid_dim.xyz;
    
    let cell = clamp(
        vec3<i32>((pos - b_min) / cell_size),
        vec3<i32>(0),
        vec3<i32>(dim) - vec3<i32>(1)
    );
    let cell_idx = u32(cell.x + cell.y * i32(dim.x) + cell.z * i32(dim.x * dim.y));
    
    let next_p = atomicExchange(&grid_heads[cell_idx], p_idx + 1u);
    grid_next[p_idx] = next_p;
}

// 4. Compute density and lambda_i constraint
@compute @workgroup_size(256)
fn compute_density_lambda(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    let pos_i = predicted_positions[i].xyz;
    let h = uniforms.kernel_params.x;
    let h2 = uniforms.kernel_params.y;
    let inv_rho0 = uniforms.kernel_params.z;
    let poly6_factor = uniforms.kernel_factors.x;
    let spiky_factor = uniforms.kernel_factors.y;
    let relaxation_eps = uniforms.solver_params.x;
    
    let cell_size = uniforms.bounds_min.w;
    let b_min = uniforms.bounds_min.xyz;
    let dim = uniforms.grid_dim.xyz;
    
    let cell = clamp(
        vec3<i32>((pos_i - b_min) / cell_size),
        vec3<i32>(0),
        vec3<i32>(dim) - vec3<i32>(1)
    );
    
    var density = poly6(0.0, h2, poly6_factor);
    var grad_ci = vec3<f32>(0.0);
    var sum_grad_sq = 0.0;
    
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    let z_min = select(-1, 0, is_2d);
    let z_max = select(1, 0, is_2d);
    
    for (var dz = z_min; dz <= z_max; dz++) {
        let cz = cell.z + dz;
        if (cz < 0 || cz >= i32(dim.z)) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let cy = cell.y + dy;
            if (cy < 0 || cy >= i32(dim.y)) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                let cx = cell.x + dx;
                if (cx < 0 || cx >= i32(dim.x)) { continue; }
                
                let c_idx = u32(cx + cy * i32(dim.x) + cz * i32(dim.x * dim.y));
                var curr = atomicLoad(&grid_heads[c_idx]);
                var iter_count = 0u;
                while (curr != 0u && iter_count < 256u) {
                    let j = curr - 1u;
                    curr = grid_next[j];
                    iter_count++;
                    
                    if (i != j) {
                        let pos_j = predicted_positions[j].xyz;
                        let r_vec = pos_i - pos_j;
                        let r2 = dot(r_vec, r_vec);
                        
                        if (r2 < h2 && r2 > 1e-12) {
                            let r = sqrt(r2);
                            density += poly6(r2, h2, poly6_factor);
                            let grad_w = spiky_gradient(r_vec, r, h, spiky_factor);
                            let grad_j = grad_w * inv_rho0;
                            grad_ci += grad_j;
                            sum_grad_sq += dot(grad_j, grad_j);
                        }
                    }
                }
            }
        }
    }
    
    let bc = boundary_contribution_gpu(pos_i);
    let rho0 = uniforms.kernel_params.w;
    density += rho0 * bc.psi;
    grad_ci += bc.grad;
    sum_grad_sq += dot(bc.grad, bc.grad) + dot(grad_ci, grad_ci);
    particles[i].vel_dens.w = density;
    
    let c_i = max(density * inv_rho0 - 1.0, 0.0);
    lambdas[i] = -c_i / (sum_grad_sq + relaxation_eps);
}

// 5. Compute displacement delta_p_i
@compute @workgroup_size(256)
fn compute_delta_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    let pred_i = predicted_positions[i];
    let pos_i = pred_i.xyz;
    let wall_w_i = pred_i.w;
    let lambda_i = lambdas[i];
    let h = uniforms.kernel_params.x;
    let h2 = uniforms.kernel_params.y;
    let inv_rho0 = uniforms.kernel_params.z;
    let rho0 = uniforms.kernel_params.w;
    let poly6_factor = uniforms.kernel_factors.x;
    let spiky_factor = uniforms.kernel_factors.y;
    let w_q = uniforms.kernel_factors.z;
    let tensile_k = uniforms.kernel_factors.w;
    let max_disp = uniforms.solver_params.y;
    let surface_tension = uniforms.fluid_params.y;
    
    let cell_size = uniforms.bounds_min.w;
    let b_min = uniforms.bounds_min.xyz;
    let dim = uniforms.grid_dim.xyz;
    
    let cell = clamp(
        vec3<i32>((pos_i - b_min) / cell_size),
        vec3<i32>(0),
        vec3<i32>(dim) - vec3<i32>(1)
    );
    
    var delta = vec3<f32>(0.0);
    let bc = boundary_contribution_gpu(pos_i);
    
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    let z_min = select(-1, 0, is_2d);
    let z_max = select(1, 0, is_2d);
    
    for (var dz = z_min; dz <= z_max; dz++) {
        let cz = cell.z + dz;
        if (cz < 0 || cz >= i32(dim.z)) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let cy = cell.y + dy;
            if (cy < 0 || cy >= i32(dim.y)) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                let cx = cell.x + dx;
                if (cx < 0 || cx >= i32(dim.x)) { continue; }
                
                let c_idx = u32(cx + cy * i32(dim.x) + cz * i32(dim.x * dim.y));
                var curr = atomicLoad(&grid_heads[c_idx]);
                var iter_count = 0u;
                while (curr != 0u && iter_count < 256u) {
                    let j = curr - 1u;
                    curr = grid_next[j];
                    iter_count++;
                    
                    if (i != j) {
                        let pred_j = predicted_positions[j];
                        let pos_j = pred_j.xyz;
                        let r_vec = pos_i - pos_j;
                        let r2 = dot(r_vec, r_vec);
                        
                        if (r2 < h2 && r2 > 1e-12) {
                            let r = sqrt(r2);
                            let lambda_j = lambdas[j];
                            let lambda_sum = lambda_i + lambda_j;
                            
                            var s_corr = 0.0;
                            if (lambda_sum < -1e-6) {
                                let w_r = poly6(r2, h2, poly6_factor);
                                let ratio = min(w_r / w_q, 1.5);
                                s_corr = -tensile_k * (ratio * ratio * ratio * ratio) * (h * h);
                            }
                            
                            let grad_w = spiky_gradient(r_vec, r, h, spiky_factor);
                            delta += grad_w * ((lambda_sum + s_corr) * inv_rho0);
                            
                            let pair_wall_scale = min(wall_w_i, pred_j.w);
                            if (surface_tension > 0.0 && pair_wall_scale > 1e-4) {
                                let q = r / h;
                                let omq = 1.0 - q;
                                let c_q = 64.0 * (q * q * q) * (omq * omq * omq);
                                let rho_i = particles[i].vel_dens.w;
                                let rho_j = particles[j].vel_dens.w;
                                let surf_w = clamp(2.0 - (rho_i + rho_j) * inv_rho0, 0.15, 1.40);
                                let cohesion = c_q * surf_w * surface_tension * pair_wall_scale * 0.016 * h;
                                delta -= (r_vec / r) * cohesion;
                            }
                        }
                    }
                }
            }
        }
    }

    if (lambda_i < -1e-7) {
        delta += bc.grad * (2.0 * lambda_i);
    }
    
    let d_len = length(delta);
    if (d_len > max_disp && d_len > 1e-7) {
        delta *= (max_disp / d_len);
    }
    
    delta_positions[i] = vec4<f32>(delta, 0.0);
}

// 6. Apply delta displacement and project boundary collisions
@compute @workgroup_size(256)
fn apply_delta_and_collide(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    var pos = predicted_positions[i].xyz + delta_positions[i].xyz * 0.65;
    let radius = uniforms.bounds_max.w;
    
    // Chamber bounds collision
    pos = resolve_chamber_gpu(pos, radius);
    
    // Dynamic Obstacles Collision Resolution (Spheres, Boxes, Cylinders)
    let num_colliders = min(uniforms.counts.w & 0x7FFFFFFFu, 16u);
    for (var c = 0u; c < num_colliders; c++) {
        let col = uniforms.colliders[c];
        if (col.info.y == 0u) { continue; }
        let kind = col.info.x;
        if (kind == 1u) {
            // Sphere
            let c_pos = col.pos_rad.xyz;
            let c_rad = col.pos_rad.w;
            let to_p = pos - c_pos;
            let dist = length(to_p);
            let min_d = c_rad + radius;
            if (dist < min_d && dist > 1e-5) {
                pos = c_pos + (to_p / dist) * min_d;
            }
        } else if (kind == 2u) {
            // Box
            let c_center = col.pos_rad.xyz;
            let h_size = col.extents.xyz + vec3<f32>(radius);
            let box_min = c_center - h_size;
            let box_max = c_center + h_size;
            if (pos.x >= box_min.x && pos.x <= box_max.x &&
                pos.y >= box_min.y && pos.y <= box_max.y &&
                pos.z >= box_min.z && pos.z <= box_max.z) {
                let d_left = abs(pos.x - box_min.x);
                let d_right = abs(box_max.x - pos.x);
                let d_bottom = abs(pos.y - box_min.y);
                let d_top = abs(box_max.y - pos.y);
                let d_back = abs(pos.z - box_min.z);
                let d_front = abs(box_max.z - pos.z);
                let m = min(min(min(d_left, d_right), min(d_bottom, d_top)), min(d_back, d_front));
                if (m == d_left) { pos.x = box_min.x; }
                else if (m == d_right) { pos.x = box_max.x; }
                else if (m == d_bottom) { pos.y = box_min.y; }
                else if (m == d_top) { pos.y = box_max.y; }
                else if (m == d_back) { pos.z = box_min.z; }
                else { pos.z = box_max.z; }
            }
        } else if (kind == 3u) {
            // Cylinder (minimum-penetration projection across top cap, bottom cap, and radial side)
            let c_center = col.pos_rad.xyz;
            let c_rad = col.pos_rad.w + radius;
            let half_h = col.extents.w * 0.5 + radius;
            let y_min = c_center.y - half_h;
            let y_max = c_center.y + half_h;
            let dx = pos.x - c_center.x;
            let dz = pos.z - c_center.z;
            let r2 = dx * dx + dz * dz;
            if (r2 < c_rad * c_rad && pos.y >= y_min && pos.y <= y_max) {
                let r = sqrt(r2);
                let d_side = c_rad - r;
                let d_top = y_max - pos.y;
                let d_bot = pos.y - y_min;
                if (d_top <= d_side && d_top <= d_bot) {
                    pos.y = y_max;
                } else if (d_bot <= d_side && d_bot <= d_top) {
                    pos.y = y_min;
                } else {
                    let r_safe = max(r, 1e-5);
                    let rx = select(1.0, dx / r_safe, r > 1e-5);
                    let rz = select(0.0, dz / r_safe, r > 1e-5);
                    pos.x = c_center.x + rx * c_rad;
                    pos.z = c_center.z + rz * c_rad;
                }
            }
        }
    }
    
    // Legacy single collider fallback if num_colliders is 0
    if (num_colliders == 0u && uniforms.fluid_params.w > 0.5) {
        let obs_pos = uniforms.collider_pos.xyz;
        let obs_rad = uniforms.collider_pos.w;
        let to_p = pos - obs_pos;
        let dist = length(to_p);
        let min_d = obs_rad + radius;
        if (dist < min_d && dist > 1e-5) {
            let n = to_p / dist;
            pos = obs_pos + n * min_d;
        }
    }
    
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    if (is_2d) {
        pos.z = 0.0;
    }

    predicted_positions[i] = vec4<f32>(pos, wall_cohesion_weight_gpu(pos));
}

// 7. Update velocity and apply XSPH artificial viscosity
@compute @workgroup_size(256)
fn update_velocity_and_xsph(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    let p_type = particles[i].pos_type.w;
    if (p_type > 0.5) {
        return;
    }
    
    let new_pos = predicted_positions[i].xyz;
    let old_pos = old_positions[i].xyz;
    let dt = uniforms.gravity.w;
    var vel = (new_pos - old_pos) / dt;
    
    let radius = uniforms.bounds_max.w;
    let restitution = uniforms.solver_params.z;
    let friction = uniforms.solver_params.w;
    let margin = max(radius * 0.20, 0.002);
    let no_slip_rate = max(1.0 - friction, 0.0) * 18.0;
    let slip_floor = clamp(1.0 / (1.0 + no_slip_rate * dt), 0.50, 1.0);
    let slip_wall = 1.0;
    let rest_threshold = 0.12;
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;

    // Chamber velocity response
    let shape = u32(uniforms.chamber_info.x + 0.5);
    if (shape == 1u) {
        let r_chamber = (b_max.y - b_min.y) * 0.5;
        let center = vec3<f32>((b_min.x + b_max.x) * 0.5, b_min.y + r_chamber, (b_min.z + b_max.z) * 0.5);
        let to_p = new_pos - center;
        let dist = length(to_p);
        if (dist >= r_chamber - radius - margin && dist > 1e-6) {
            let n = to_p / dist;
            let vn = dot(vel, n);
            if (vn > 0.0) {
                let n_vel = n * vn;
                let t_vel = vel - n_vel;
                let eff_rest = select(0.0, restitution, vn > rest_threshold);
                vel = t_vel * slip_floor - n_vel * eff_rest;
            }
        }
    } else if (shape == 2u) {
        let H = max(b_max.y - b_min.y, 1e-4);
        let slope_offset = uniforms.chamber_info.w;
        let norm_y = clamp((new_pos.y - b_min.y) / H, 0.0, 1.0);
        let x_back = b_min.x + (1.0 - norm_y) * slope_offset;
        let back_dist = abs(new_pos.x - x_back);
        if (back_dist <= radius + margin) {
            let back_normal = normalize(vec3<f32>(H, slope_offset, 0.0));
            let vn = dot(vel, -back_normal);
            if (vn < 0.0) {
                let n_v = -back_normal * vn;
                let t_v = vel - n_v;
                let eff_rest = select(0.0, restitution, vn < -rest_threshold);
                vel = t_v * slip_wall - n_v * eff_rest;
            }
        }
        if (new_pos.y <= b_min.y + radius + margin) {
            if (vel.y < 0.0) {
                vel.y = select(0.0, -vel.y * restitution, vel.y < -rest_threshold);
            }
            vel.x *= slip_floor;
            vel.z *= slip_floor;
        }
        if (new_pos.y >= b_max.y - radius - margin && vel.y > 0.0) {
            vel.y = select(0.0, -vel.y * restitution, vel.y > rest_threshold);
            vel.x *= slip_wall;
            vel.z *= slip_wall;
        }
        if (new_pos.x >= b_max.x - radius - margin && vel.x > 0.0) {
            vel.x = select(0.0, -vel.x * restitution, vel.x > rest_threshold);
            vel.y *= slip_wall;
            vel.z *= slip_wall;
        }
        if (!is_2d) {
            if (new_pos.z <= b_min.z + radius + margin && vel.z < 0.0) {
                vel.z = select(0.0, -vel.z * restitution, vel.z < -rest_threshold);
                vel.x *= slip_wall;
                vel.y *= slip_wall;
            }
            if (new_pos.z >= b_max.z - radius - margin && vel.z > 0.0) {
                vel.z = select(0.0, -vel.z * restitution, vel.z > rest_threshold);
                vel.x *= slip_wall;
                vel.y *= slip_wall;
            }
        }
    } else {
        if (new_pos.y <= b_min.y + radius + margin) {
            if (vel.y < 0.0) {
                if (vel.y < -rest_threshold) {
                    vel.y = -vel.y * restitution;
                } else {
                    vel.y = 0.0;
                }
            }
            vel.x *= slip_floor;
            vel.z *= slip_floor;
        }
        if (new_pos.y >= b_max.y - radius - margin && vel.y > 0.0) {
            if (vel.y > rest_threshold) {
                vel.y = -vel.y * restitution;
            } else {
                vel.y = 0.0;
            }
            vel.x *= slip_wall;
            vel.z *= slip_wall;
        }
        if (new_pos.x <= b_min.x + radius + margin && vel.x < 0.0) {
            if (vel.x < -rest_threshold) {
                vel.x = -vel.x * restitution;
            } else {
                vel.x = 0.0;
            }
            vel.y *= slip_wall;
            vel.z *= slip_wall;
        }
        if (new_pos.x >= b_max.x - radius - margin && vel.x > 0.0) {
            if (vel.x > rest_threshold) {
                vel.x = -vel.x * restitution;
            } else {
                vel.x = 0.0;
            }
            vel.y *= slip_wall;
            vel.z *= slip_wall;
        }
        if (!is_2d) {
            if (new_pos.z <= b_min.z + radius + margin && vel.z < 0.0) {
                if (vel.z < -rest_threshold) {
                    vel.z = -vel.z * restitution;
                } else {
                    vel.z = 0.0;
                }
                vel.x *= slip_wall;
                vel.y *= slip_wall;
            }
            if (new_pos.z >= b_max.z - radius - margin && vel.z > 0.0) {
                if (vel.z > rest_threshold) {
                    vel.z = -vel.z * restitution;
                } else {
                    vel.z = 0.0;
                }
                vel.x *= slip_wall;
                vel.y *= slip_wall;
            }
        }
    }

    // Dynamic obstacle velocity resolution (Spheres, Boxes, Cylinders)
    let num_colliders = min(uniforms.counts.w & 0x7FFFFFFFu, 16u);
    for (var c = 0u; c < num_colliders; c++) {
        let col = uniforms.colliders[c];
        if (col.info.y == 0u) { continue; }
        let kind = col.info.x;
        let c_rest = bitcast<f32>(col.info.z);
        let c_fric = bitcast<f32>(col.info.w);
        var hit = false;
        var n_obs = vec3<f32>(0.0, 1.0, 0.0);

        if (kind == 1u) {
            let to_p = new_pos - col.pos_rad.xyz;
            let dist = length(to_p);
            if (dist <= col.pos_rad.w + radius + margin && dist > 1e-5) {
                n_obs = to_p / dist;
                hit = true;
            }
        } else if (kind == 2u) {
            let c_center = col.pos_rad.xyz;
            let h_size = col.extents.xyz + vec3<f32>(radius + margin);
            let box_min = c_center - h_size;
            let box_max = c_center + h_size;
            if (new_pos.x >= box_min.x && new_pos.x <= box_max.x &&
                new_pos.y >= box_min.y && new_pos.y <= box_max.y &&
                (is_2d || (new_pos.z >= box_min.z && new_pos.z <= box_max.z))) {
                let d_left = abs(new_pos.x - box_min.x);
                let d_right = abs(box_max.x - new_pos.x);
                let d_bottom = abs(new_pos.y - box_min.y);
                let d_top = abs(box_max.y - new_pos.y);
                let d_back = select(abs(new_pos.z - box_min.z), 1e6, is_2d);
                let d_front = select(abs(box_max.z - new_pos.z), 1e6, is_2d);
                let m = min(min(min(d_left, d_right), min(d_bottom, d_top)), min(d_back, d_front));
                if (m <= margin * 2.5) {
                    hit = true;
                    if (m == d_left) { n_obs = vec3<f32>(-1.0, 0.0, 0.0); }
                    else if (m == d_right) { n_obs = vec3<f32>(1.0, 0.0, 0.0); }
                    else if (m == d_bottom) { n_obs = vec3<f32>(0.0, -1.0, 0.0); }
                    else if (m == d_top) { n_obs = vec3<f32>(0.0, 1.0, 0.0); }
                    else if (m == d_back) { n_obs = vec3<f32>(0.0, 0.0, -1.0); }
                    else { n_obs = vec3<f32>(0.0, 0.0, 1.0); }
                }
            }
        } else if (kind == 3u) {
            let c_center = col.pos_rad.xyz;
            let c_rad = col.pos_rad.w + radius + margin;
            let half_h = col.extents.w * 0.5 + radius + margin;
            let dx = new_pos.x - c_center.x;
            let dz = new_pos.z - c_center.z;
            let r = sqrt(dx * dx + dz * dz);
            if (r <= c_rad && new_pos.y >= c_center.y - half_h && new_pos.y <= c_center.y + half_h) {
                let d_side = c_rad - r;
                let d_top = (c_center.y + half_h) - new_pos.y;
                let d_bot = new_pos.y - (c_center.y - half_h);
                let m = min(d_side, min(d_top, d_bot));
                if (m <= margin * 2.5) {
                    hit = true;
                    if (d_top <= d_side && d_top <= d_bot) {
                        n_obs = vec3<f32>(0.0, 1.0, 0.0);
                    } else if (d_bot <= d_side && d_bot <= d_top) {
                        n_obs = vec3<f32>(0.0, -1.0, 0.0);
                    } else if (r > 1e-5) {
                        n_obs = vec3<f32>(dx / r, 0.0, dz / r);
                    }
                }
            }
        }

        if (hit) {
            let vn = dot(vel, n_obs);
            if (vn < 0.0) {
                let eff_rest = select(0.0, c_rest, vn < -0.10);
                vel -= (1.0 + eff_rest) * vn * n_obs;
                let vt = vel - n_obs * dot(vel, n_obs);
                vel -= vt * clamp((1.0 - c_fric) * 0.1, 0.0, 0.5);
            }
        }
    }
    // Fused XSPH Viscosity + Vorticity Curl
    let viscosity = uniforms.fluid_params.x;
    let vort_eps = uniforms.chamber_info.y;
    var curl = vec3<f32>(0.0);
    if (viscosity > 0.0 || vort_eps > 1e-5) {
        let h = uniforms.kernel_params.x;
        let h2 = uniforms.kernel_params.y;
        let inv_rho0 = uniforms.kernel_params.z;
        let poly6_factor = uniforms.kernel_factors.x;
        let spiky_factor = uniforms.kernel_factors.y;
        let cell_size = uniforms.bounds_min.w;
        let dim = uniforms.grid_dim.xyz;
        
        let cell = clamp(
            vec3<i32>((new_pos - b_min) / cell_size),
            vec3<i32>(0),
            vec3<i32>(dim) - vec3<i32>(1)
        );
        
        var sum_v_diff = vec3<f32>(0.0);
        var sum_weights = 0.0;
        
        let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
        let z_min = select(-1, 0, is_2d);
        let z_max = select(1, 0, is_2d);
        
        for (var dz = z_min; dz <= z_max; dz++) {
            let cz = cell.z + dz;
            if (cz < 0 || cz >= i32(dim.z)) { continue; }
            for (var dy = -1; dy <= 1; dy++) {
                let cy = cell.y + dy;
                if (cy < 0 || cy >= i32(dim.y)) { continue; }
                for (var dx = -1; dx <= 1; dx++) {
                    let cx = cell.x + dx;
                    if (cx < 0 || cx >= i32(dim.x)) { continue; }
                    
                    let c_idx = u32(cx + cy * i32(dim.x) + cz * i32(dim.x * dim.y));
                    var curr = atomicLoad(&grid_heads[c_idx]);
                    var iter_count = 0u;
                    while (curr != 0u && iter_count < 128u) {
                        let j = curr - 1u;
                        curr = grid_next[j];
                        iter_count++;
                        
                        if (i != j) {
                            let pos_j = predicted_positions[j].xyz;
                            let r_vec = new_pos - pos_j;
                            let r2 = dot(r_vec, r_vec);
                            if (r2 < h2 && r2 > 1e-12) {
                                let v_j = (pos_j - old_positions[j].xyz) / dt;
                                if (viscosity > 0.0) {
                                    let w_ij = poly6(r2, h2, poly6_factor);
                                    sum_v_diff += (v_j - vel) * w_ij;
                                    sum_weights += w_ij;
                                }
                                if (vort_eps > 1e-5) {
                                    let r = sqrt(r2);
                                    let grad_w = spiky_gradient(r_vec, r, h, spiky_factor);
                                    curl += cross(vel - v_j, grad_w) * inv_rho0;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (viscosity > 0.0 && sum_weights > 1e-6) {
            let rho0 = uniforms.kernel_params.w;
            let w_self = poly6(0.0, h2, poly6_factor);
            let denom = max(sum_weights + w_self, rho0);
            vel += (sum_v_diff / denom) * viscosity;
        }
        let inv_exp = 1.0 / clamp(dt * 120.0, 0.001, 2.0);
        let visc_raw = 1.0 - pow(max(1.0 - clamp(viscosity, 0.0, 0.9999), 1e-6), inv_exp);
        let creep = clamp((visc_raw - 0.25) / 0.70, 0.0, 1.0);
        if (creep > 0.0) {
            let floor_prox = clamp(1.0 - ((new_pos.y - b_min.y) / (6.0 * h)), 0.0, 1.0);
            let c2 = creep * creep;
            let h_damp = 1.0 / (1.0 + c2 * (2.0 + 10.0 * floor_prox) * dt);
            let v_damp = 1.0 / (1.0 + c2 * 1.2 * dt);
            vel.x *= h_damp;
            vel.z *= h_damp;
            vel.y *= v_damp;
        }
    }
    
    // Aerodynamic drag on sparse airborne splash droplets (bulk liquid unaffected)
    let air_drag = uniforms.boundary_params.y;
    if (air_drag > 1e-4) {
        let rho_ratio = particles[i].vel_dens.w * uniforms.kernel_params.z;
        let exposure = clamp((0.58 - rho_ratio) / 0.42, 0.0, 1.0);
        let cur_spd = length(vel);
        if (exposure > 0.0 && cur_spd > 0.20) {
            let drag_rate = air_drag * exposure * (0.60 + 0.35 * cur_spd);
            vel *= 1.0 / (1.0 + drag_rate * dt);
        }
    }

    // Clamp velocity to maximum speed
    let max_v = uniforms.fluid_params.z;
    let speed = length(vel);
    if (speed > max_v && speed > 1e-6) {
        vel *= (max_v / speed);
    }
    
    if (is_2d) {
        vel.z = 0.0;
    }

    delta_positions[i] = vec4<f32>(curl, length(curl));
    particles[i].pos_type = vec4<f32>(new_pos, p_type);
    particles[i].vel_dens = vec4<f32>(vel, particles[i].vel_dens.w);
}

// 8. Vorticity Confinement Pass (Macklin & Müller 2013)
@compute @workgroup_size(256)
fn apply_vorticity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    if (particles[i].pos_type.w > 0.5) {
        return;
    }

    var vel = particles[i].vel_dens.xyz;
    let speed = length(vel);
    let flow_gate = clamp((speed - 0.08) / 0.35, 0.0, 1.0);
    let rho_ratio = particles[i].vel_dens.w * uniforms.kernel_params.z;
    let bulk_gate = clamp((rho_ratio - 0.72) / 0.22, 0.0, 1.0);
    let wall_gate = 0.20 + 0.80 * predicted_positions[i].w;
    let eff_gate = flow_gate * bulk_gate * wall_gate;
    if (eff_gate <= 1e-4) {
        return;
    }

    let omega_data = delta_positions[i];
    let omega_i = omega_data.xyz;
    let omega_i_mag = omega_data.w;
    if (omega_i_mag <= 1e-4) {
        return;
    }

    let pos_i = particles[i].pos_type.xyz;
    let h = uniforms.kernel_params.x;
    let h2 = uniforms.kernel_params.y;
    let inv_rho0 = uniforms.kernel_params.z;
    let spiky_factor = uniforms.kernel_factors.y;
    let cell_size = uniforms.bounds_min.w;
    let b_min = uniforms.bounds_min.xyz;
    let dim = uniforms.grid_dim.xyz;
    let dt = uniforms.gravity.w;
    let vort_eps = uniforms.chamber_info.y;

    let cell = clamp(
        vec3<i32>((pos_i - b_min) / cell_size),
        vec3<i32>(0),
        vec3<i32>(dim) - vec3<i32>(1)
    );

    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    let z_min = select(-1, 0, is_2d);
    let z_max = select(1, 0, is_2d);

    var eta = vec3<f32>(0.0);
    for (var dz = z_min; dz <= z_max; dz++) {
        let cz = cell.z + dz;
        if (cz < 0 || cz >= i32(dim.z)) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let cy = cell.y + dy;
            if (cy < 0 || cy >= i32(dim.y)) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                let cx = cell.x + dx;
                if (cx < 0 || cx >= i32(dim.x)) { continue; }

                let c_idx = u32(cx + cy * i32(dim.x) + cz * i32(dim.x * dim.y));
                var curr = atomicLoad(&grid_heads[c_idx]);
                var iter_count = 0u;
                while (curr != 0u && iter_count < 128u) {
                    let j = curr - 1u;
                    curr = grid_next[j];
                    iter_count++;

                    if (i != j) {
                        let pos_j = predicted_positions[j].xyz;
                        let r_vec = pos_i - pos_j;
                        let r2 = dot(r_vec, r_vec);
                        if (r2 < h2 && r2 > 1e-12) {
                            let r = sqrt(r2);
                            let omega_j_mag = delta_positions[j].w;
                            let grad_w = spiky_gradient(r_vec, r, h, spiky_factor);
                            eta += grad_w * ((omega_j_mag - omega_i_mag) * inv_rho0);
                        }
                    }
                }
            }
        }
    }

    let eta_len = length(eta);
    if (eta_len > 1e-5) {
        let n_dir = eta / eta_len;
        let f_vort = cross(n_dir, omega_i) * (vort_eps * h * 8.0 * eff_gate);
        var dv = f_vort * dt;
        let dv_len = length(dv);
        let max_dv = min(speed * 0.22, speed * 26.0 * dt);
        if (dv_len > max_dv && dv_len > 1e-6) {
            dv *= (max_dv / dv_len);
        }
        vel += dv;
        if (is_2d) {
            vel.z = 0.0;
        }
        let max_v = uniforms.fluid_params.z;
        let final_speed = length(vel);
        if (final_speed > max_v && final_speed > 1e-6) {
            vel *= (max_v / final_speed);
        }
        particles[i].vel_dens = vec4<f32>(vel, particles[i].vel_dens.w);
    }
}
