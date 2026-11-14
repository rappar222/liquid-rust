struct SurfaceParams {
    // xyz = grid_min, w = voxel_size
    grid_min_voxel: vec4<f32>,
    // xyz = coarse_min, w = inv_splat_r2
    coarse_min_inv_r2: vec4<f32>,
    // x = splat_radius, y = splat_r2, z = iso, w = brick_world_size
    splat_params: vec4<f32>,
    // xyz = 1.0 / coarse_cell_size, w = 0.0
    c_inv_size: vec4<f32>,
    // x = nx, y = ny, z = nz, w = num_active_bricks
    u_dims: vec4<u32>,
    // x = cgx, y = cgy, z = cgz, w = max_crossing_bricks
    u_coarse: vec4<u32>,
}

struct SurfaceParticle {
    pos_type: vec4<f32>,
    vel_dens: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: SurfaceParams;
@group(0) @binding(1) var<storage, read> sorted_particles: array<SurfaceParticle>;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> active_bricks: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> crossing_brick_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_corners: array<vec4<f32>>;

var<workgroup> sh_pos: array<vec3<f32>, 256>;
var<workgroup> sh_vel: array<vec3<f32>, 256>;
var<workgroup> sh_any_in: atomic<u32>;
var<workgroup> sh_any_out: atomic<u32>;
var<workgroup> sh_slot: u32;

@compute @workgroup_size(256, 1, 1)
fn evaluate_bricks(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let brick_list_idx = wg_id.y * 256u + wg_id.x;
    if brick_list_idx >= u.u_dims.w {
        return;
    }

    if tid == 0u {
        atomicStore(&sh_any_in, 0u);
        atomicStore(&sh_any_out, 0u);
        sh_slot = 0xffffffffu;
    }
    workgroupBarrier();

    let bc = active_bricks[brick_list_idx];
    let gx0 = bc.x * 8u;
    let gy0 = bc.y * 8u;
    let gz0 = bc.z * 8u;

    let cells_x = min(8u, u.u_dims.x - gx0);
    let cells_y = min(8u, u.u_dims.y - gy0);
    let cells_z = min(8u, u.u_dims.z - gz0);

    let voxel = u.grid_min_voxel.w;
    let g_min = u.grid_min_voxel.xyz;
    let b_min = g_min + vec3<f32>(f32(gx0), f32(gy0), f32(gz0)) * voxel;
    let b_max = b_min + vec3<f32>(f32(cells_x), f32(cells_y), f32(cells_z)) * voxel;

    let splat_r = u.splat_params.x;
    let splat_r2 = u.splat_params.y;
    let iso = u.splat_params.z;
    let inv_splat_r2 = u.coarse_min_inv_r2.w;

    let c_min_f = floor((b_min - vec3<f32>(splat_r) - u.coarse_min_inv_r2.xyz) * u.c_inv_size.xyz);
    let c_max_f = floor((b_max + vec3<f32>(splat_r) - u.coarse_min_inv_r2.xyz) * u.c_inv_size.xyz);

    let cx0 = u32(clamp(i32(c_min_f.x), 0, i32(u.u_coarse.x) - 1));
    let cy0 = u32(clamp(i32(c_min_f.y), 0, i32(u.u_coarse.y) - 1));
    let cz0 = u32(clamp(i32(c_min_f.z), 0, i32(u.u_coarse.z) - 1));
    let cx1 = u32(clamp(i32(c_max_f.x), 0, i32(u.u_coarse.x) - 1));
    let cy1 = u32(clamp(i32(c_max_f.y), 0, i32(u.u_coarse.y) - 1));
    let cz1 = u32(clamp(i32(c_max_f.z), 0, i32(u.u_coarse.z) - 1));

    // Each thread evaluates up to 3 corners (729 total across 256 threads)
    let idx0 = tid;
    let idx1 = tid + 256u;
    let idx2 = tid + 512u;

    var p0 = vec3<f32>(0.0);
    var p1 = vec3<f32>(0.0);
    var p2 = vec3<f32>(0.0);
    var v0 = false;
    var v1 = false;
    var v2 = false;

    {
        let lz = idx0 / 81u;
        let rem = idx0 - lz * 81u;
        let ly = rem / 9u;
        let lx = rem - ly * 9u;
        if lx <= cells_x && ly <= cells_y && lz <= cells_z {
            v0 = true;
            p0 = g_min + vec3<f32>(f32(gx0 + lx), f32(gy0 + ly), f32(gz0 + lz)) * voxel;
        }
    }
    {
        let lz = idx1 / 81u;
        let rem = idx1 - lz * 81u;
        let ly = rem / 9u;
        let lx = rem - ly * 9u;
        if lx <= cells_x && ly <= cells_y && lz <= cells_z {
            v1 = true;
            p1 = g_min + vec3<f32>(f32(gx0 + lx), f32(gy0 + ly), f32(gz0 + lz)) * voxel;
        }
    }
    if idx2 < 729u {
        let lz = idx2 / 81u;
        let rem = idx2 - lz * 81u;
        let ly = rem / 9u;
        let lx = rem - ly * 9u;
        if lx <= cells_x && ly <= cells_y && lz <= cells_z {
            v2 = true;
            p2 = g_min + vec3<f32>(f32(gx0 + lx), f32(gy0 + ly), f32(gz0 + lz)) * voxel;
        }
    }

    var s0 = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    var wv0 = vec3<f32>(0.0);
    var wv1 = vec3<f32>(0.0);
    var wv2 = vec3<f32>(0.0);

    // Stream particles along contiguous X spans inside the dilated coarse-grid box
    for (var cz = cz0; cz <= cz1; cz += 1u) {
        for (var cy = cy0; cy <= cy1; cy += 1u) {
            let row_base = (cz * u.u_coarse.y + cy) * u.u_coarse.x;
            let r_start = cell_offsets[row_base + cx0];
            let r_end = cell_offsets[row_base + cx1 + 1u];

            var base = r_start;
            while base < r_end {
                let count = min(256u, r_end - base);
                if tid < count {
                    let gp = sorted_particles[base + tid];
                    sh_pos[tid] = gp.pos_type.xyz;
                    sh_vel[tid] = gp.vel_dens.xyz;
                }
                workgroupBarrier();

                for (var k = 0u; k < count; k += 1u) {
                    let q = sh_pos[k];
                    let qv = sh_vel[k];
                    if v0 {
                        let d = p0 - q;
                        let d2 = dot(d, d);
                        if d2 < splat_r2 {
                            let t = 1.0 - d2 * inv_splat_r2;
                            let w = t * t * t;
                            s0 += w;
                            wv0 += qv * w;
                        }
                    }
                    if v1 {
                        let d = p1 - q;
                        let d2 = dot(d, d);
                        if d2 < splat_r2 {
                            let t = 1.0 - d2 * inv_splat_r2;
                            let w = t * t * t;
                            s1 += w;
                            wv1 += qv * w;
                        }
                    }
                    if v2 {
                        let d = p2 - q;
                        let d2 = dot(d, d);
                        if d2 < splat_r2 {
                            let t = 1.0 - d2 * inv_splat_r2;
                            let w = t * t * t;
                            s2 += w;
                            wv2 += qv * w;
                        }
                    }
                }
                workgroupBarrier();
                base += 256u;
            }
        }
    }

    var phi0 = 0.0;
    var phi1 = 0.0;
    var phi2 = 0.0;
    var vel0 = vec3<f32>(0.0);
    var vel1 = vec3<f32>(0.0);
    var vel2 = vec3<f32>(0.0);

    if v0 {
        phi0 = s0 / (1.0 + s0 * 0.35);
        if s0 > 1e-6 {
            vel0 = wv0 / s0;
        }
        if phi0 >= iso {
            atomicStore(&sh_any_in, 1u);
        } else {
            atomicStore(&sh_any_out, 1u);
        }
    }
    if v1 {
        phi1 = s1 / (1.0 + s1 * 0.35);
        if s1 > 1e-6 {
            vel1 = wv1 / s1;
        }
        if phi1 >= iso {
            atomicStore(&sh_any_in, 1u);
        } else {
            atomicStore(&sh_any_out, 1u);
        }
    }
    if v2 {
        phi2 = s2 / (1.0 + s2 * 0.35);
        if s2 > 1e-6 {
            vel2 = wv2 / s2;
        }
        if phi2 >= iso {
            atomicStore(&sh_any_in, 1u);
        } else {
            atomicStore(&sh_any_out, 1u);
        }
    }
    workgroupBarrier();

    if tid == 0u {
        if atomicLoad(&sh_any_in) != 0u && atomicLoad(&sh_any_out) != 0u {
            let slot = atomicAdd(&counters[0], 1u);
            if slot < u.u_coarse.w {
                crossing_brick_indices[slot] = brick_list_idx;
                sh_slot = slot;
            }
        }
    }
    workgroupBarrier();

    let slot = sh_slot;
    if slot != 0xffffffffu {
        let out_base = slot * 729u;
        if v0 {
            out_corners[out_base + idx0] = vec4<f32>(vel0, phi0);
        }
        if v1 {
            out_corners[out_base + idx1] = vec4<f32>(vel1, phi1);
        }
        if v2 {
            out_corners[out_base + idx2] = vec4<f32>(vel2, phi2);
        }
    }
}
