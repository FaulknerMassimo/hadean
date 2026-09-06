//! The voxel grid.
//!
//! Axis convention: `x` and `y` are horizontal, `z` is depth. **`z = 0` is the
//! water surface and `z` increases downward**, so light enters at `z = 0` and
//! thermal vents sit at `z = nz - 1`. Keep this straight; every field in L2
//! depends on it.
//!
//! Boundaries are closed (no-flux) on all six faces. Transport is written in
//! face-flux form, which means a closed boundary conserves mass exactly rather
//! than approximately -- important for the audit in Phase 1.

use serde::{Deserialize, Serialize};

/// Linear index into a field array.
pub type VoxelId = u32;

/// Grid geometry. Cheap to copy; passed everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    /// Voxel edge length, metres.
    pub dx: f32,
}

impl Grid {
    pub fn new(nx: u32, ny: u32, nz: u32, dx: f32) -> Self {
        assert!(nx > 0 && ny > 0 && nz > 0, "grid must be non-degenerate");
        assert!(dx > 0.0, "voxel size must be positive");
        Self { nx, ny, nz, dx }
    }

    /// Total voxel count.
    #[inline]
    pub fn len(&self) -> usize {
        self.nx as usize * self.ny as usize * self.nz as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Voxels per horizontal layer.
    #[inline]
    pub fn layer(&self) -> usize {
        self.nx as usize * self.ny as usize
    }

    /// Volume of one voxel, m^3.
    #[inline]
    pub fn voxel_volume(&self) -> f32 {
        self.dx * self.dx * self.dx
    }

    /// Area of one voxel face, m^2.
    #[inline]
    pub fn face_area(&self) -> f32 {
        self.dx * self.dx
    }

    /// Total world volume, m^3.
    #[inline]
    pub fn world_volume(&self) -> f64 {
        self.len() as f64 * self.voxel_volume() as f64
    }

    /// World extent in metres, `(x, y, z)`.
    #[inline]
    pub fn extent(&self) -> (f32, f32, f32) {
        (
            self.nx as f32 * self.dx,
            self.ny as f32 * self.dx,
            self.nz as f32 * self.dx,
        )
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32, z: u32) -> usize {
        debug_assert!(x < self.nx && y < self.ny && z < self.nz);
        x as usize + self.nx as usize * (y as usize + self.ny as usize * z as usize)
    }

    #[inline]
    pub fn coords(&self, i: usize) -> (u32, u32, u32) {
        let x = i % self.nx as usize;
        let rest = i / self.nx as usize;
        let y = rest % self.ny as usize;
        let z = rest / self.ny as usize;
        (x as u32, y as u32, z as u32)
    }

    /// Depth index of voxel `i`, without recovering x and y.
    #[inline]
    pub fn depth_of(&self, i: usize) -> u32 {
        (i / self.layer()) as u32
    }

    /// Centre of voxel `i` in world coordinates, metres.
    #[inline]
    pub fn centre(&self, i: usize) -> [f32; 3] {
        let (x, y, z) = self.coords(i);
        [
            (x as f32 + 0.5) * self.dx,
            (y as f32 + 0.5) * self.dx,
            (z as f32 + 0.5) * self.dx,
        ]
    }

    /// Voxel containing a world position, clamped to the grid.
    #[inline]
    pub fn voxel_at(&self, p: [f32; 3]) -> usize {
        let x = ((p[0] / self.dx) as i64).clamp(0, self.nx as i64 - 1) as u32;
        let y = ((p[1] / self.dx) as i64).clamp(0, self.ny as i64 - 1) as u32;
        let z = ((p[2] / self.dx) as i64).clamp(0, self.nz as i64 - 1) as u32;
        self.idx(x, y, z)
    }

    /// Index of the neighbour one step along `axis` in `dir` (+1 / -1), or
    /// `None` at a closed boundary.
    #[inline]
    pub fn neighbour(&self, i: usize, axis: Axis, dir: i32) -> Option<usize> {
        let (x, y, z) = self.coords(i);
        let (nx, ny, nz) = (self.nx as i64, self.ny as i64, self.nz as i64);
        let (mut cx, mut cy, mut cz) = (x as i64, y as i64, z as i64);
        match axis {
            Axis::X => cx += dir as i64,
            Axis::Y => cy += dir as i64,
            Axis::Z => cz += dir as i64,
        }
        if cx < 0 || cy < 0 || cz < 0 || cx >= nx || cy >= ny || cz >= nz {
            None
        } else {
            Some(self.idx(cx as u32, cy as u32, cz as u32))
        }
    }

    /// Stride between successive indices along `axis`.
    #[inline]
    pub fn stride(&self, axis: Axis) -> usize {
        match axis {
            Axis::X => 1,
            Axis::Y => self.nx as usize,
            Axis::Z => self.layer(),
        }
    }

    /// Number of voxels along `axis`.
    #[inline]
    pub fn count(&self, axis: Axis) -> u32 {
        match axis {
            Axis::X => self.nx,
            Axis::Y => self.ny,
            Axis::Z => self.nz,
        }
    }

    /// Component of `i`'s coordinate along `axis`.
    #[inline]
    pub fn coord_on(&self, i: usize, axis: Axis) -> u32 {
        let (x, y, z) = self.coords(i);
        match axis {
            Axis::X => x,
            Axis::Y => y,
            Axis::Z => z,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g() -> Grid {
        Grid::new(8, 5, 3, 25.0e-6)
    }

    #[test]
    fn index_round_trips() {
        let g = g();
        for i in 0..g.len() {
            let (x, y, z) = g.coords(i);
            assert_eq!(g.idx(x, y, z), i);
            assert_eq!(g.depth_of(i), z);
        }
    }

    #[test]
    fn neighbours_stop_at_closed_boundaries() {
        let g = g();
        assert!(g.neighbour(g.idx(0, 0, 0), Axis::X, -1).is_none());
        assert!(g.neighbour(g.idx(7, 4, 2), Axis::Z, 1).is_none());
        assert_eq!(
            g.neighbour(g.idx(0, 0, 0), Axis::X, 1),
            Some(g.idx(1, 0, 0))
        );
        assert_eq!(
            g.neighbour(g.idx(3, 2, 1), Axis::Z, -1),
            Some(g.idx(3, 2, 0))
        );
    }

    #[test]
    fn strides_agree_with_indexing() {
        let g = g();
        let i = g.idx(2, 2, 1);
        for axis in Axis::ALL {
            let j = g.neighbour(i, axis, 1).unwrap();
            assert_eq!(j - i, g.stride(axis));
        }
    }

    #[test]
    fn voxel_lookup_clamps() {
        let g = g();
        assert_eq!(g.voxel_at([-1.0, -1.0, -1.0]), g.idx(0, 0, 0));
        assert_eq!(g.voxel_at([1.0, 1.0, 1.0]), g.idx(7, 4, 2));
        let c = g.centre(g.idx(3, 1, 2));
        assert_eq!(g.voxel_at(c), g.idx(3, 1, 2));
    }
}
