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
};

struct Particle {
    pos_type: vec4<f32>, // xyz = position, w = particle_type (0 = fluid, 1 = boundary)
    vel_dens: vec4<f32>, // xyz = velocity, w = density
    temp_mat: vec4<f32>, // x = temp, y = mat_code, z = life, w = pad
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
    let shape = u32(uniforms.chamber_info.x + 0.5);
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    
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
        pos.y = clamp(pos.y, b_min_eff.y, b_max_eff.y);
        
        // 2. Sloped backrest at -X and front wall at +X
        let norm_y = clamp((pos.y - b_min.y) / H, 0.0, 1.0);
        let x_back = b_min_eff.x + (1.0 - norm_y) * slope_offset;
        let x_front = b_max_eff.x;
        pos.x = clamp(pos.x, x_back, x_front);
        
        // 3. Side walls at Z
        let half_w = (b_max.z - b_min.z) * 0.5;
        let cz = (b_min.z + b_max.z) * 0.5;
        let z_rel = pos.z - cz;
        let z_clamped = clamp(z_rel, -half_w + radius, half_w - radius);
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
    } else {
        // Box Chamber
        let b_min_eff = b_min + vec3<f32>(radius);
        let b_max_eff = b_max - vec3<f32>(radius);
        pos = clamp(pos, b_min_eff, b_max_eff);
    }
    return pos;
}

// 1. Predict positions with gravity & user interaction
@compute @workgroup_size(256)
fn predict_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
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
    
    // Material buoyancy:
    // 5 = Gas (+14.5 m/s^2), 6 = Fire (+25.0 m/s^2), 7 = Steam (+22.0 m/s^2), 8 = Smoke (+12.0 m/s^2)
    let mat = u32(particles[i].temp_mat.y + 0.5);
    var buoy = 0.0;
    if (mat == 5u) {
        buoy = 14.5;
    } else if (mat == 6u) {
        buoy = 25.0;
    } else if (mat == 7u) {
        buoy = 22.0;
    } else if (mat == 8u) {
        buoy = 12.0;
    }

    let accel = uniforms.gravity.xyz + vec3<f32>(0.0, buoy, 0.0) + f_user;
    vel += accel * dt;
    var pred = p_pos + vel * dt;
    
    let radius = uniforms.bounds_max.w;
    pred = resolve_chamber_gpu(pred, radius);
    
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
    let p_idx = id.x;
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
    let i = id.x;
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
    
    for (var dz = -1; dz <= 1; dz++) {
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

// 5. Compute displacement delta_p_i
@compute @workgroup_size(256)
fn compute_delta_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    let pos_i = predicted_positions[i].xyz;
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
    
    for (var dz = -1; dz <= 1; dz++) {
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
                            
                            let dt = uniforms.gravity.w;
                            let dt_factor = clamp(dt / 0.016, 0.001, 1.0);
                            
                            var s_corr = 0.0;
                            if (lambda_sum < -1e-6) {
                                let w_r = poly6(r2, h2, poly6_factor);
                                let ratio = w_r / w_q;
                                s_corr = -tensile_k * (ratio * ratio * ratio * ratio) * (h * h);
                            }
                            
                            let grad_w = spiky_gradient(r_vec, r, h, spiky_factor);
                            delta += grad_w * ((lambda_sum + s_corr) * inv_rho0);
                            
                            if (surface_tension > 0.0) {
                                let w_r = poly6(r2, h2, poly6_factor);
                                let cohesion = (w_r * surface_tension * 0.05) / rho0;
                                delta -= (r_vec / r) * cohesion;
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
    let i = id.x;
    if (i >= uniforms.counts.x) {
        return;
    }
    
    var pos = predicted_positions[i].xyz + delta_positions[i].xyz;
    let radius = uniforms.bounds_max.w;
    
    // Chamber bounds collision
    pos = resolve_chamber_gpu(pos, radius);
    
    // Dynamic Obstacles Collision Resolution (Spheres, Boxes, Cylinders)
    let num_colliders = min(uniforms.counts.w, 16u);
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
            // Cylinder
            let c_center = col.pos_rad.xyz;
            let c_rad = col.pos_rad.w + radius;
            let half_h = col.extents.w * 0.5 + radius;
            let dx = pos.x - c_center.x;
            let dz = pos.z - c_center.z;
            let r2 = dx * dx + dz * dz;
            if (r2 < c_rad * c_rad && pos.y >= c_center.y - half_h && pos.y <= c_center.y + half_h) {
                let r = max(sqrt(r2), 1e-5);
                pos.x = c_center.x + (dx / r) * c_rad;
                pos.z = c_center.z + (dz / r) * c_rad;
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
    
    predicted_positions[i] = vec4<f32>(pos, 0.0);
}

// 7. Update velocity and apply XSPH artificial viscosity
@compute @workgroup_size(256)
fn update_velocity_and_xsph(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
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
    let margin = 0.001;
    let b_min = uniforms.bounds_min.xyz;
    let b_max = uniforms.bounds_max.xyz;
    
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
                vel = t_vel * friction - n_vel * restitution;
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
                vel = t_v * friction - n_v * restitution;
            }
        }
        if (new_pos.y <= b_min.y + radius + margin) {
            if (vel.y < 0.0) {
                vel.y = -vel.y * restitution;
            } else if (vel.y < 0.15) {
                vel.y = 0.0;
            }
            vel.x *= friction;
            vel.z *= friction;
        }
        if (new_pos.y >= b_max.y - radius - margin && vel.y > 0.0) {
            vel.y = -vel.y * restitution;
            vel.x *= friction;
            vel.z *= friction;
        }
        if (new_pos.x >= b_max.x - radius - margin && vel.x > 0.0) {
            vel.x = -vel.x * restitution;
            vel.y *= friction;
            vel.z *= friction;
        }
        if (new_pos.z <= b_min.z + radius + margin && vel.z < 0.0) {
            vel.z = -vel.z * restitution;
            vel.x *= friction;
            vel.y *= friction;
        }
        if (new_pos.z >= b_max.z - radius - margin && vel.z > 0.0) {
            vel.z = -vel.z * restitution;
            vel.x *= friction;
            vel.z *= friction;
        }
    } else {
        if (new_pos.y <= b_min.y + radius + margin) {
            if (vel.y < 0.0) {
                vel.y = -vel.y * restitution;
            } else if (vel.y < 0.15) {
                vel.y = 0.0;
            }
            vel.x *= friction;
            vel.z *= friction;
        }
        if (new_pos.y >= b_max.y - radius - margin && vel.y > 0.0) {
            vel.y = -vel.y * restitution;
            vel.x *= friction;
            vel.z *= friction;
        }
        if (new_pos.x <= b_min.x + radius + margin && vel.x < 0.0) {
            vel.x = -vel.x * restitution;
            vel.y *= friction;
            vel.z *= friction;
        }
        if (new_pos.x >= b_max.x - radius - margin && vel.x > 0.0) {
            vel.x = -vel.x * restitution;
            vel.y *= friction;
            vel.z *= friction;
        }
        if (new_pos.z <= b_min.z + radius + margin && vel.z < 0.0) {
            vel.z = -vel.z * restitution;
            vel.x *= friction;
            vel.y *= friction;
        }
        if (new_pos.z >= b_max.z - radius - margin && vel.z > 0.0) {
            vel.z = -vel.z * restitution;
            vel.x *= friction;
            vel.z *= friction;
        }
    }
    // XSPH Viscosity
    let viscosity = uniforms.fluid_params.x;
    if (viscosity > 0.0) {
        let h2 = uniforms.kernel_params.y;
        let poly6_factor = uniforms.kernel_factors.x;
        let cell_size = uniforms.bounds_min.w;
        let dim = uniforms.grid_dim.xyz;
        
        let cell = clamp(
            vec3<i32>((new_pos - b_min) / cell_size),
            vec3<i32>(0),
            vec3<i32>(dim) - vec3<i32>(1)
        );
        
        var sum_v_diff = vec3<f32>(0.0);
        var sum_weights = 0.0;
        
        for (var dz = -1; dz <= 1; dz++) {
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
                                let w_ij = poly6(r2, h2, poly6_factor);
                                let v_j = (predicted_positions[j].xyz - old_positions[j].xyz) / dt;
                                sum_v_diff += (v_j - vel) * w_ij;
                                sum_weights += w_ij;
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
    
    // Clamp velocity to maximum speed
    let max_v = uniforms.fluid_params.z;
    let speed = length(vel);
    if (speed > max_v && speed > 1e-6) {
        vel *= (max_v / speed);
    }
    
    particles[i].pos_type = vec4<f32>(new_pos, p_type);
    particles[i].vel_dens = vec4<f32>(vel, particles[i].vel_dens.w);
}
