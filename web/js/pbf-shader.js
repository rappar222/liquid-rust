// WGSL Shaders for LiquidRust WebGPU
// Includes:
// 1. PBF Compute Shaders (predict, clear grid, insert, density/lambda, delta pos + Akinci cohesion, collide, velocity/xsph + curl, vorticity confinement)
// 2. Studio Glossy Particle Render Shader (instanced spherical billboards with Fresnel sky-rim, GGX highlights, and all 4 color modes)
// 3. Wireframe & Studio Floor Grid Render Shader

export const PBF_COMPUTE_WGSL = `
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
    
    kernel_params: vec4<f32>,    // [h, h2, inv_rho0, rho0]
    kernel_factors: vec4<f32>,   // [poly6_factor, spiky_factor, w_q, tensile_k]
    solver_params: vec4<f32>,    // [relaxation_epsilon, max_displacement, restitution, boundary_friction]
    fluid_params: vec4<f32>,     // [viscosity, surface_tension, max_velocity, collider_enabled]
    
    counts: vec4<u32>,           // [particle_count, solver_iterations, substeps, num_colliders | (is_2d ? 0x80000000 : 0)]
    colliders: array<GpuCollider, 16>,
    chamber_info: vec4<f32>,     // x: shape, y: vorticity, z: spacing, w: pad
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

fn resolve_chamber_gpu(pos_in: vec3<f32>, radius: f32) -> vec3<f32> {
    var pos = pos_in;
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;

    let b_min_eff = b_min + vec3<f32>(radius, radius, select(radius, 0.0, is_2d));
    let b_max_eff = b_max - vec3<f32>(radius, radius, select(radius, 0.0, is_2d));
    pos = clamp(pos, b_min_eff, b_max_eff);
    if (is_2d) {
        pos.z = 0.0;
    }
    return pos;
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
        predicted_positions[i] = vec4<f32>(p_pos, 0.0);
        return;
    }
    
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
    
    let accel = uniforms.gravity.xyz + f_user;
    vel += accel * dt;
    var pred = p_pos + vel * dt;
    
    let radius = uniforms.bounds_max.w;
    pred = resolve_chamber_gpu(pred, radius);
    
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    if (is_2d) {
        pred.z = 0.0;
        vel.z = 0.0;
    }

    predicted_positions[i] = vec4<f32>(pred, 0.0);
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
    
    sum_grad_sq += dot(grad_ci, grad_ci);
    particles[i].vel_dens.w = density;
    
    let c_i = max(density * inv_rho0 - 1.0, 0.0);
    lambdas[i] = -c_i / (sum_grad_sq + relaxation_eps);
}

// 5. Compute displacement delta_p_i + Akinci free-surface cohesion
@compute @workgroup_size(256)
fn compute_delta_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    let pos_i = predicted_positions[i].xyz;
    let lambda_i = lambdas[i];
    let rho_i = particles[i].vel_dens.w;
    let h = uniforms.kernel_params.x;
    let h2 = uniforms.kernel_params.y;
    let inv_rho0 = uniforms.kernel_params.z;
    let rho0 = uniforms.kernel_params.w;
    let poly6_factor = uniforms.kernel_factors.x;
    let spiky_factor = uniforms.kernel_factors.y;
    let w_q = uniforms.kernel_factors.z;
    let tensile_k = uniforms.kernel_factors.w;
    let max_disp = uniforms.solver_params.y;
    let st_coeff = uniforms.fluid_params.y;
    let spacing = max(uniforms.chamber_info.z, h * 0.5);
    
    let cell_size = uniforms.bounds_min.w;
    let b_min = uniforms.bounds_min.xyz;
    let dim = uniforms.grid_dim.xyz;
    
    let cell = clamp(
        vec3<i32>((pos_i - b_min) / cell_size),
        vec3<i32>(0),
        vec3<i32>(dim) - vec3<i32>(1)
    );
    
    var delta = vec3<f32>(0.0);
    
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
                            
                            if (st_coeff > 0.0 && r > spacing * 0.45) {
                                let q = r / h;
                                let omq = 1.0 - q;
                                let spline = (q * q * q) * (omq * omq * omq) * 64.0;
                                let rho_j = particles[j].vel_dens.w;
                                let k_ij = (2.0 * rho0) / max(rho_i + rho_j, rho0 * 0.5);
                                delta -= (r_vec / r) * (st_coeff * k_ij * spline);
                            }
                        }
                    }
                }
            }
        }
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
    
    pos = resolve_chamber_gpu(pos, radius);
    
    let num_colliders = min(uniforms.counts.w & 0x7FFFFFFFu, 16u);
    for (var c = 0u; c < num_colliders; c++) {
        let col = uniforms.colliders[c];
        if (col.info.y == 0u) { continue; }
        let kind = col.info.x;
        if (kind == 1u) {
            let c_pos = col.pos_rad.xyz;
            let c_rad = col.pos_rad.w;
            let to_p = pos - c_pos;
            let dist = length(to_p);
            let min_d = c_rad + radius;
            if (dist < min_d && dist > 1e-5) {
                pos = c_pos + (to_p / dist) * min_d;
            }
        }
    }
    
    if (num_colliders == 0u && uniforms.fluid_params.w > 0.5) {
        let obs_pos = uniforms.collider_pos.xyz;
        let obs_rad = uniforms.collider_pos.w;
        let to_p = pos - obs_pos;
        let dist = length(to_p);
        let min_d = obs_rad + radius;
        if (dist < min_d && dist > 1e-5) {
            pos = obs_pos + (to_p / dist) * min_d;
        }
    }
    
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    if (is_2d) {
        pos.z = 0.0;
    }

    predicted_positions[i] = vec4<f32>(pos, 0.0);
}

// 7. Update velocity, apply XSPH artificial viscosity, and compute curl for vorticity confinement
@compute @workgroup_size(256)
fn update_velocity_and_xsph(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    let p_type = particles[i].pos_type.w;
    if (p_type > 0.5) {
        delta_positions[i] = vec4<f32>(0.0);
        return;
    }
    
    let new_pos = predicted_positions[i].xyz;
    let old_pos = old_positions[i].xyz;
    let dt = uniforms.gravity.w;
    var vel = (new_pos - old_pos) / dt;
    
    let radius = uniforms.bounds_max.w;
    let restitution = uniforms.solver_params.z;
    let friction = uniforms.solver_params.w;
    let margin = 0.001;
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    
    let slip = clamp(1.0 - (1.0 - friction) * dt * 3.0, 0.95, 1.0);
    let rest_threshold = 0.10;
    if (new_pos.y <= b_min.y + radius + margin) {
        if (vel.y < 0.0) {
            vel.y = select(0.0, -vel.y * restitution, vel.y < -rest_threshold);
        } else if (vel.y > 0.0 && vel.y < 0.20 && new_pos.y <= b_min.y + radius + 0.001) {
            vel.y *= 0.1;
        }
        vel.x *= slip;
        vel.z *= slip;
    }
    if (new_pos.y >= b_max.y - radius - margin && vel.y > 0.0) {
        vel.y = select(0.0, -vel.y * restitution, vel.y > rest_threshold);
        vel.x *= slip;
        vel.z *= slip;
    }
    if (new_pos.x <= b_min.x + radius + margin && vel.x < 0.0) {
        vel.x = select(0.0, -vel.x * restitution, vel.x < -rest_threshold);
        vel.y *= slip;
        vel.z *= slip;
    }
    if (new_pos.x >= b_max.x - radius - margin && vel.x > 0.0) {
        vel.x = select(0.0, -vel.x * restitution, vel.x > rest_threshold);
        vel.y *= slip;
        vel.z *= slip;
    }
    if (!is_2d) {
        if (new_pos.z <= b_min.z + radius + margin && vel.z < 0.0) {
            vel.z = select(0.0, -vel.z * restitution, vel.z < -rest_threshold);
            vel.x *= slip;
            vel.y *= slip;
        }
        if (new_pos.z >= b_max.z - radius - margin && vel.z > 0.0) {
            vel.z = select(0.0, -vel.z * restitution, vel.z > rest_threshold);
            vel.x *= slip;
            vel.y *= slip;
        }
    }
    
    let viscosity = uniforms.fluid_params.x;
    let vorticity = uniforms.chamber_info.y;
    var curl = vec3<f32>(0.0);

    if (viscosity > 0.0 || vorticity > 1e-4) {
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
                                let v_diff = v_j - vel;
                                if (viscosity > 0.0) {
                                    let w_ij = poly6(r2, h2, poly6_factor);
                                    sum_v_diff += v_diff * w_ij;
                                    sum_weights += w_ij;
                                }
                                if (vorticity > 1e-4) {
                                    let r = sqrt(r2);
                                    let grad_w = spiky_gradient(r_vec, r, h, spiky_factor);
                                    curl += cross(v_diff, grad_w) * inv_rho0;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (sum_weights > 1e-6) {
            vel += (sum_v_diff / sum_weights) * viscosity;
        }
    }
    
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

// 8. Flow-Gated Vorticity Confinement (Macklin & Müller 2013)
@compute @workgroup_size(256)
fn apply_vorticity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= uniforms.counts.x) {
        return;
    }

    let vorticity = uniforms.chamber_info.y;
    if (vorticity <= 1e-4 || particles[i].pos_type.w > 0.5) {
        return;
    }

    let curl_info = delta_positions[i];
    let w_mag_i = curl_info.w;
    if (w_mag_i <= 1e-3) {
        return;
    }

    var vel = particles[i].vel_dens.xyz;
    let speed = length(vel);
    if (speed <= 0.08) {
        return;
    }

    let pos_i = predicted_positions[i].xyz;
    let h = uniforms.kernel_params.x;
    let h2 = uniforms.kernel_params.y;
    let inv_rho0 = uniforms.kernel_params.z;
    let spiky_factor = uniforms.kernel_factors.y;
    let cell_size = uniforms.bounds_min.w;
    let b_min = uniforms.bounds_min.xyz;
    let dim = uniforms.grid_dim.xyz;
    let is_2d = (uniforms.counts.w & 0x80000000u) != 0u;
    let spacing = max(uniforms.chamber_info.z, h * 0.5);

    let cell = clamp(
        vec3<i32>((pos_i - b_min) / cell_size),
        vec3<i32>(0),
        vec3<i32>(dim) - vec3<i32>(1)
    );

    var eta = vec3<f32>(0.0);
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
                        let r_vec = pos_i - predicted_positions[j].xyz;
                        let r2 = dot(r_vec, r_vec);
                        if (r2 < h2 && r2 > 1e-12) {
                            let r = sqrt(r2);
                            let grad_w = spiky_gradient(r_vec, r, h, spiky_factor);
                            let dw = (delta_positions[j].w - w_mag_i) * inv_rho0;
                            eta += grad_w * dw;
                        }
                    }
                }
            }
        }
    }

    let eta_len = length(eta);
    if (eta_len > 1e-5) {
        let n_dir = eta / eta_len;
        let flow_gate = clamp((speed - 0.08) / 0.35, 0.0, 1.0);
        let dt = uniforms.gravity.w;
        var dv = cross(n_dir, curl_info.xyz) * (vorticity * spacing * 2.2 * flow_gate * dt);
        if (is_2d) {
            dv.z = 0.0;
        }
        let dv_len = length(dv);
        let max_dv = speed * 0.22;
        if (dv_len > max_dv && dv_len > 1e-6) {
            dv *= (max_dv / dv_len);
        }
        vel += dv;
        particles[i].vel_dens = vec4<f32>(vel, particles[i].vel_dens.w);
    }
}
`;

// WebGPU Studio Optical Fluid Rendering Shader
export const PBF_RENDER_WGSL = `
struct RenderUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    params: vec4<f32>,        // x: particle_radius, y: refraction, z: is_2d, w: lit_spheres
    light_dir: vec4<f32>,     // xyz: light_dir, w: rest_density
    color_mode: vec4<u32>,    // x: mode (0:vel, 1:pres, 2:dens, 3:material), y: colormap, z: tex_mode, w: pad
    shallow_color: vec4<f32>, // xyz: shallow tint, w: tex_strength
    deep_color: vec4<f32>,    // xyz: deep tint, w: anim_time
    optics: vec4<f32>,        // x: ior, y: roughness, z: specular, w: metallic
    foam_params: vec4<f32>,   // x: crest_foam_intensity, yzw: pad
};

struct Particle {
    pos_type: vec4<f32>,
    vel_dens: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: RenderUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) metric_t: f32,
    @location(2) pos_eye: vec3<f32>,
    @location(3) world_pos: vec3<f32>,
    @location(4) speed_t: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let corner = corners[vid];
    let p = particles[iid];
    let world_pos = p.pos_type.xyz;
    let eye_center = camera.view * vec4<f32>(world_pos, 1.0);
    let radius = camera.params.x;

    let eye_pos = eye_center.xyz + vec3<f32>(corner * radius, 0.0);

    var out: VertexOutput;
    out.position = camera.proj * vec4<f32>(eye_pos, 1.0);
    out.uv = corner;
    out.pos_eye = eye_pos;
    out.world_pos = world_pos;

    let speed = length(p.vel_dens.xyz);
    let speed_t = clamp(speed / 2.5, 0.0, 1.0);
    out.speed_t = speed_t;

    let mode = camera.color_mode.x;
    let rho0 = max(camera.light_dir.w, 10.0);
    let rho = p.vel_dens.w;

    if (mode == 1u) {
        let constraint = max(rho / rho0 - 1.0, 0.0);
        out.metric_t = clamp(constraint * 2.2, 0.0, 1.0);
    } else if (mode == 2u) {
        let dev = abs(rho - rho0) / rho0;
        out.metric_t = clamp(dev * 4.0, 0.0, 1.0);
    } else {
        out.metric_t = speed_t;
    }

    return out;
}

fn colormap_ocean(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.04, 0.16, 0.38);
    let c1 = vec3<f32>(0.03, 0.56, 0.88);
    let c2 = vec3<f32>(0.36, 0.88, 0.96);
    let c3 = vec3<f32>(0.97, 0.99, 1.00);
    if (t < 0.33) {
        return mix(c0, c1, t / 0.33);
    } else if (t < 0.75) {
        return mix(c1, c2, (t - 0.33) / 0.42);
    } else {
        return mix(c2, c3, (t - 0.75) / 0.25);
    }
}

fn colormap_turbo(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.19, 0.07, 0.23);
    let c1 = vec3<f32>(0.13, 0.57, 0.95);
    let c2 = vec3<f32>(0.55, 0.85, 0.25);
    let c3 = vec3<f32>(0.98, 0.65, 0.10);
    let c4 = vec3<f32>(0.85, 0.15, 0.10);
    if (t < 0.25) {
        return mix(c0, c1, t / 0.25);
    } else if (t < 0.5) {
        return mix(c1, c2, (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        return mix(c2, c3, (t - 0.5) / 0.25);
    } else {
        return mix(c3, c4, (t - 0.75) / 0.25);
    }
}

fn colormap_viridis(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.27, 0.00, 0.33);
    let c1 = vec3<f32>(0.19, 0.41, 0.56);
    let c2 = vec3<f32>(0.12, 0.63, 0.53);
    let c3 = vec3<f32>(0.99, 0.91, 0.14);
    if (t < 0.33) {
        return mix(c0, c1, t / 0.33);
    } else if (t < 0.66) {
        return mix(c1, c2, (t - 0.33) / 0.33);
    } else {
        return mix(c2, c3, (t - 0.66) / 0.34);
    }
}

fn colormap_fire(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.05, 0.0, 0.0);
    let c1 = vec3<f32>(0.8, 0.15, 0.0);
    let c2 = vec3<f32>(1.0, 0.65, 0.0);
    let c3 = vec3<f32>(1.0, 1.0, 0.85);
    if (t < 0.33) {
        return mix(c0, c1, t / 0.33);
    } else if (t < 0.75) {
        return mix(c1, c2, (t - 0.33) / 0.42);
    } else {
        return mix(c2, c3, (t - 0.75) / 0.25);
    }
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }

    let z = sqrt(max(1.0 - r2, 0.0));
    var n = vec3<f32>(in.uv.x, in.uv.y, z);
    let tex_mode = camera.color_mode.z;
    let tex_str = clamp(camera.shallow_color.w, 0.0, 1.0);
    let t_anim = camera.deep_color.w;
    let surf_pos = in.world_pos + n * camera.params.x;

    var caustic_boost = 0.0;
    var foam_tex = 0.0;
    var grid_mod = 1.0;

    if (tex_mode != 0u && tex_str > 0.005) {
        if (tex_mode == 1u) {
            // Capillary Ripples
            let w1 = cos(surf_pos.x * 28.0 + surf_pos.y * 19.0 - t_anim * 3.8);
            let w2 = sin(surf_pos.z * 26.0 - surf_pos.x * 16.0 + t_anim * 3.1);
            let bump = vec3<f32>(w1, w2, 0.0) * (tex_str * 0.42);
            n = normalize(n + bump);
            caustic_boost = max(w1 * w2, 0.0) * tex_str * 0.20;
        } else if (tex_mode == 2u) {
            // Shimmering Voronoi-style interference caustics
            let p = surf_pos * 18.0;
            let c1 = abs(sin(p.x + sin(p.y * 1.3 + t_anim * 2.4)) * cos(p.y + cos(p.z * 1.4 - t_anim * 1.9)));
            let web = pow(1.0 - clamp(c1, 0.0, 1.0), 3.0);
            caustic_boost = web * tex_str * 0.85;
            n = normalize(n + vec3<f32>(sin(p.x * 1.7), cos(p.y * 1.7), 0.0) * (tex_str * 0.25));
        } else if (tex_mode == 3u) {
            // Cellular Crest Foam
            let p = surf_pos * 24.0 + vec3<f32>(t_anim * 1.5, -t_anim * 0.8, t_anim * 1.2);
            let f = (sin(p.x) * cos(p.y) + sin(p.y * 1.7 + p.z) + cos(p.z * 2.3 - p.x)) * 0.333 + 0.5;
            foam_tex = clamp((f - 0.38) * 2.6, 0.0, 1.0) * tex_str * (0.28 + 0.72 * in.speed_t);
            n = normalize(n + vec3<f32>(f - 0.5, (1.0 - f) - 0.5, 0.0) * (tex_str * 0.22));
        } else if (tex_mode == 4u) {
            // Studio UV Contour Grid
            let gx = abs(fract(surf_pos.x * 8.0) - 0.5);
            let gy = abs(fract(surf_pos.y * 8.0) - 0.5);
            let gz = abs(fract(surf_pos.z * 8.0) - 0.5);
            let line = max(1.0 - min(gx, gy) * 14.0, 1.0 - gz * 14.0);
            let line_c = clamp(line, 0.0, 1.0) * tex_str;
            grid_mod = 1.0 - tex_str * 0.14;
            caustic_boost = line_c * 0.45;
        }
    }

    let shallow = camera.shallow_color.xyz;
    let deep = camera.deep_color.xyz;
    let metallic = clamp(camera.optics.w, 0.0, 1.0);
    let color_mode = camera.color_mode.x;

    var base_color: vec3<f32>;
    if (color_mode == 3u) {
        // Optical Material: shallow surface edge -> deep liquid interior
        let rim = 1.0 - z;
        let shallow_mix = clamp(0.28 + 0.48 * in.speed_t + 0.42 * rim, 0.0, 1.0);
        base_color = mix(deep, shallow, shallow_mix);
        let crest_foam = smoothstep(0.62, 0.98, in.speed_t) * camera.foam_params.x * (1.0 - metallic);
        base_color = mix(base_color, vec3<f32>(0.94, 0.98, 1.0), crest_foam * 0.72);
    } else {
        let map_type = camera.color_mode.y;
        if (map_type == 1u) {
            base_color = colormap_turbo(in.metric_t);
        } else if (map_type == 2u) {
            base_color = colormap_viridis(in.metric_t);
        } else if (map_type == 3u) {
            base_color = colormap_fire(in.metric_t);
        } else {
            base_color = colormap_ocean(in.metric_t);
        }
    }

    base_color = base_color * grid_mod + mix(shallow, vec3<f32>(0.96, 0.99, 1.0), 0.35) * caustic_boost;
    if (foam_tex > 0.005) {
        base_color = mix(base_color, vec3<f32>(0.95, 0.98, 1.0), clamp(foam_tex, 0.0, 0.85));
    }

    let is_lit = camera.params.w > 0.5;
    var rgb = base_color;
    if (is_lit) {
        let l_dir = camera.light_dir.xyz;
        let ior = max(camera.optics.x, 1.01);
        let rough = clamp(camera.optics.y, 0.015, 0.95);
        let spec_mult = max(camera.optics.z, 0.0);
        let refr_glow = clamp(camera.params.y, 0.0, 1.5);

        let f0_diel = pow((ior - 1.0) / (ior + 1.0), 2.0);
        let f0 = mix(vec3<f32>(f0_diel), base_color, metallic);
        let ndotv = max(n.z, 0.001);
        let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - ndotv, 4.2);

        let ndotl = max(dot(n, l_dir), 0.0);
        let hemi = 0.5 + 0.5 * n.y;
        let half_dir = normalize(l_dir + vec3<f32>(0.0, 0.0, 1.0));
        let spec_pow = clamp(1.8 / (rough * rough + 0.003), 8.0, 420.0);
        let spec_energy = (0.28 + 0.72 * (1.0 - rough)) * spec_mult;
        let spec = pow(max(dot(n, half_dir), 0.0), spec_pow) * spec_energy * fresnel;

        // Studio overhead sky / horizon reflection
        let refl_y = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
        let sky_env = mix(vec3<f32>(0.08, 0.14, 0.24), vec3<f32>(0.55, 0.78, 0.98), pow(refl_y, 1.5));
        let body_lit = base_color * (0.28 + 0.18 * hemi + 0.62 * ndotl * (1.0 - 0.65 * metallic));

        // Subsurface refraction glow near thin silhouettes
        let sss_rim = pow(1.0 - ndotv, 2.2) * refr_glow * (1.0 - metallic);
        let sss_tint = shallow * (0.35 + 0.65 * (0.5 + 0.5 * dot(n, -l_dir))) * sss_rim;

        rgb = mix(body_lit, sky_env * mix(vec3<f32>(1.0), base_color, metallic), clamp(fresnel * (0.75 + 0.25 * metallic), vec3<f32>(0.0), vec3<f32>(0.96)))
            + spec
            + sss_tint;
    }

    var out: FragmentOutput;
    out.color = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);

    let is_2d = camera.params.z > 0.5;
    if (!is_2d) {
        let radius = camera.params.x;
        let sphere_pos_eye = in.pos_eye + vec3<f32>(0.0, 0.0, z * radius);
        let clip_pos = camera.proj * vec4<f32>(sphere_pos_eye, 1.0);
        out.depth = clamp(clip_pos.z / clip_pos.w, 0.0, 1.0);
    } else {
        out.depth = in.position.z;
    }

    return out;
}
`;

export const WIREFRAME_WGSL = `
struct WireframeUniforms {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: WireframeUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return uniforms.color;
}
`;
