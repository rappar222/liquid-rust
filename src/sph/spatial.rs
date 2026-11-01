use glam::Vec3;

const P1: i32 = 73856093;
const P2: i32 = 19349663;
const P3: i32 = 83492791;

/// High-performance Spatial Hash Grid for 3D SPH neighbor queries
#[derive(Clone, Debug)]
pub struct SpatialHashGrid {
    pub cell_size: f32,
    pub inv_cell_size: f32,
    pub table_size: usize,
    pub mask: usize,
    pub head: Vec<u32>,
    pub next: Vec<u32>,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f32, capacity: usize) -> Self {
        let min_table = (capacity * 2).max(4096);
        let table_size = min_table.next_power_of_two();
        let mask = table_size - 1;

        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            table_size,
            mask,
            head: vec![u32::MAX; table_size],
            next: vec![u32::MAX; capacity],
        }
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
    pub fn hash_coords(&self, ix: i32, iy: i32, iz: i32) -> usize {
        let h = (ix.wrapping_mul(P1) ^ iy.wrapping_mul(P2) ^ iz.wrapping_mul(P3)) as usize;
        h & self.mask
    }

    /// Rebuild the spatial grid from particle positions in O(N)
    pub fn build(&mut self, positions: &[Vec3]) {
        let n = positions.len();

        // Resize buffers if necessary
        if self.next.len() < n {
            self.next.resize(n, u32::MAX);
        }
        let target_table_size = (n * 2).max(4096).next_power_of_two();
        if target_table_size != self.table_size {
            self.table_size = target_table_size;
            self.mask = target_table_size - 1;
            self.head.resize(target_table_size, u32::MAX);
        }

        self.head.fill(u32::MAX);

        for (i, &pos) in positions.iter().enumerate() {
            let (ix, iy, iz) = self.cell_coords(pos);
            let cell_hash = self.hash_coords(ix, iy, iz);
            self.next[i] = self.head[cell_hash];
            self.head[cell_hash] = i as u32;
        }
    }

    /// Query all neighbor particle indices within radius h
    /// Calls visitor closure for each candidate neighbor: visitor(neighbor_idx)
    #[inline(always)]
    pub fn for_each_neighbor<F>(&self, pos: Vec3, mut visitor: F)
    where
        F: FnMut(usize),
    {
        let (cx, cy, cz) = self.cell_coords(pos);

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let hash = self.hash_coords(cx + dx, cy + dy, cz + dz);
                    let mut curr = self.head[hash];
                    while curr != u32::MAX {
                        let j = curr as usize;
                        visitor(j);
                        curr = self.next[j];
                    }
                }
            }
        }
    }
}
