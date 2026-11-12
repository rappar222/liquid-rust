use glam::Vec3;

const TABLE_BITS: usize = 16;
const TABLE_SIZE: usize = 1 << TABLE_BITS; // 65,536 buckets (fits comfortably in L2 cache)
const TABLE_MASK: usize = TABLE_SIZE - 1;

/// Ultra-fast, zero-allocation Spatial Hash Grid for uniform radius neighbor queries.
/// Uses a generation-stamped open bucket table + intrusive linked list (`heads` + `next`)
/// with exact cell coordinate verification, eliminating per-substep heap allocations and SipHash overhead.
pub struct SpatialHashGrid {
    cell_size: f32,
    inv_cell_size: f32,
    current_gen: u32,
    bucket_gen: Vec<u32>,
    heads: Vec<i32>,
    next: Vec<i32>,
    particle_cells: Vec<[i32; 3]>,
    active_buckets: usize,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f32, initial_capacity: usize) -> Self {
        assert!(cell_size > 0.0, "Cell size must be strictly positive");
        let cap = initial_capacity.max(256);
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            current_gen: 1,
            bucket_gen: vec![0; TABLE_SIZE],
            heads: vec![-1; TABLE_SIZE],
            next: Vec::with_capacity(cap),
            particle_cells: Vec::with_capacity(cap),
            active_buckets: 0,
        }
    }

    /// Build spatial grid from 3D particle positions.
    #[inline]
    pub fn build(&mut self, positions: &[Vec3]) {
        self.build_from_positions(positions, false);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.bucket_gen.fill(0);
            self.current_gen = 1;
        }
        self.next.clear();
        self.particle_cells.clear();
        self.active_buckets = 0;
    }

    pub fn set_cell_size(&mut self, cell_size: f32) {
        self.cell_size = cell_size;
        self.inv_cell_size = 1.0 / cell_size;
    }

    #[inline(always)]
    pub fn cell_coords(&self, pos: Vec3) -> (i32, i32, i32) {
        (
            (pos.x * self.inv_cell_size).floor() as i32,
            (pos.y * self.inv_cell_size).floor() as i32,
            (pos.z * self.inv_cell_size).floor() as i32,
        )
    }

    #[inline(always)]
    fn hash_coords(cx: i32, cy: i32, cz: i32) -> usize {
        let ux = cx as u32;
        let uy = cy as u32;
        let uz = cz as u32;
        let h = ux
            .wrapping_mul(73_856_093)
            ^ uy.wrapping_mul(19_349_663)
            ^ uz.wrapping_mul(83_492_791);
        (h as usize) & TABLE_MASK
    }

    /// Insert a single particle index at world position `pos`.
    #[inline]
    pub fn insert(&mut self, particle_idx: usize, pos: Vec3) {
        let (cx, cy, cz) = self.cell_coords(pos);
        self.insert_coords(particle_idx, cx, cy, cz);
    }

    #[inline(always)]
    fn insert_coords(&mut self, particle_idx: usize, cx: i32, cy: i32, cz: i32) {
        if particle_idx >= self.next.len() {
            let new_len = particle_idx + 1;
            self.next.resize(new_len, -1);
            self.particle_cells.resize(new_len, [0, 0, 0]);
        }

        let bucket = Self::hash_coords(cx, cy, cz);
        let prev_head = if self.bucket_gen[bucket] == self.current_gen {
            self.heads[bucket]
        } else {
            self.bucket_gen[bucket] = self.current_gen;
            self.active_buckets += 1;
            -1
        };

        self.next[particle_idx] = prev_head;
        self.particle_cells[particle_idx] = [cx, cy, cz];
        self.heads[bucket] = particle_idx as i32;
    }

    /// Batch-build spatial grid from contiguous particle positions with zero allocations after warmup.
    pub fn build_from_positions(&mut self, positions: &[Vec3], is_2d: bool) {
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.bucket_gen.fill(0);
            self.current_gen = 1;
        }
        self.active_buckets = 0;

        let n = positions.len();
        self.next.resize(n, -1);
        self.particle_cells.resize(n, [0, 0, 0]);

        let inv_h = self.inv_cell_size;
        let gen = self.current_gen;

        for (i, &pos) in positions.iter().enumerate() {
            let cx = (pos.x * inv_h).floor() as i32;
            let cy = (pos.y * inv_h).floor() as i32;
            let cz = if is_2d {
                0
            } else {
                (pos.z * inv_h).floor() as i32
            };

            let bucket = Self::hash_coords(cx, cy, cz);
            let prev_head = if self.bucket_gen[bucket] == gen {
                self.heads[bucket]
            } else {
                self.bucket_gen[bucket] = gen;
                self.active_buckets += 1;
                -1
            };

            self.next[i] = prev_head;
            self.particle_cells[i] = [cx, cy, cz];
            self.heads[bucket] = i as i32;
        }
    }

    /// Query all particles within the 27 neighboring 3D cells (3x3x3 around `pos`).
    #[inline]
    pub fn for_each_neighbor<F>(&self, pos: Vec3, f: F)
    where
        F: FnMut(usize),
    {
        self.for_each_neighbor_mode(pos, false, f);
    }

    /// Query all particles within neighboring cells (9 cells in 2D planar mode, 27 cells in 3D mode).
    #[inline]
    pub fn for_each_neighbor_mode<F>(&self, pos: Vec3, is_2d: bool, mut f: F)
    where
        F: FnMut(usize),
    {
        let cx = (pos.x * self.inv_cell_size).floor() as i32;
        let cy = (pos.y * self.inv_cell_size).floor() as i32;
        let cz = if is_2d {
            0
        } else {
            (pos.z * self.inv_cell_size).floor() as i32
        };

        let (dz_min, dz_max) = if is_2d { (0, 0) } else { (-1, 1) };
        let gen = self.current_gen;

        for dz in dz_min..=dz_max {
            let tz = cz + dz;
            for dy in -1..=1 {
                let ty = cy + dy;
                for dx in -1..=1 {
                    let tx = cx + dx;
                    let bucket = Self::hash_coords(tx, ty, tz);
                    if self.bucket_gen[bucket] == gen {
                        let mut curr = self.heads[bucket];
                        while curr >= 0 {
                            let idx = curr as usize;
                            let pc = self.particle_cells[idx];
                            if pc[0] == tx && pc[1] == ty && pc[2] == tz {
                                f(idx);
                            }
                            curr = self.next[idx];
                        }
                    }
                }
            }
        }
    }

    pub fn active_cell_count(&self) -> usize {
        self.active_buckets
    }
}
