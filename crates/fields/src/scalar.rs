//! Field containers.
//!
//! Chemical amounts are stored **compound-major**: all voxels for compound 0,
//! then all voxels for compound 1, and so on. That is the layout diffusion
//! wants -- each compound's stencil pass walks one contiguous slice -- and
//! diffusion is the dominant cost. It is the wrong layout for per-voxel
//! chemistry, which needs every compound at one voxel, so that pass works in
//! tiles: gather a block of voxels into a small voxel-major scratch buffer,
//! react it, scatter it back. See [`ChemField::tile`].

use hadean_core::hash::{HashState, StateHasher};
use hadean_core::Grid;
use serde::{Deserialize, Serialize};

/// Voxels per pass of the transpose. Chosen so one block of every compound
/// stays resident in L1.
const TRANSPOSE_BLOCK: usize = 64;

/// A single scalar per voxel (temperature, pressure, a morphogen).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarField {
    pub data: Vec<f32>,
}

impl ScalarField {
    pub fn new(grid: &Grid, value: f32) -> Self {
        Self {
            data: vec![value; grid.len()],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn get(&self, i: usize) -> f32 {
        self.data[i]
    }

    #[inline]
    pub fn set(&mut self, i: usize, v: f32) {
        self.data[i] = v;
    }

    #[inline]
    pub fn add(&mut self, i: usize, v: f32) {
        self.data[i] += v;
    }

    /// Sum in `f64`, in index order. Deterministic regardless of thread count.
    pub fn total(&self) -> f64 {
        self.data.iter().map(|&x| x as f64).sum()
    }

    pub fn mean(&self) -> f64 {
        self.total() / self.data.len().max(1) as f64
    }

    pub fn min_max(&self) -> (f32, f32) {
        self.data
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            })
    }
}

impl HashState for ScalarField {
    fn hash_state(&self, h: &mut StateHasher) {
        h.f32_slice(&self.data);
    }
}

/// Particle counts for every compound in every voxel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemField {
    pub n_compounds: usize,
    pub n_voxels: usize,
    /// `data[compound * n_voxels + voxel]`.
    pub data: Vec<f32>,
}

impl ChemField {
    pub fn new(grid: &Grid, n_compounds: usize) -> Self {
        let n_voxels = grid.len();
        Self {
            n_compounds,
            n_voxels,
            data: vec![0.0; n_compounds * n_voxels],
        }
    }

    #[inline]
    pub fn index(&self, compound: usize, voxel: usize) -> usize {
        debug_assert!(compound < self.n_compounds && voxel < self.n_voxels);
        compound * self.n_voxels + voxel
    }

    #[inline]
    pub fn get(&self, compound: usize, voxel: usize) -> f32 {
        self.data[self.index(compound, voxel)]
    }

    #[inline]
    pub fn set(&mut self, compound: usize, voxel: usize, v: f32) {
        let i = self.index(compound, voxel);
        self.data[i] = v;
    }

    #[inline]
    pub fn add(&mut self, compound: usize, voxel: usize, v: f32) {
        let i = self.index(compound, voxel);
        self.data[i] += v;
    }

    /// Move `delta` particles into one voxel, carrying whatever `f32` cannot
    /// represent in `residual`. Returns what the field-and-residual pair
    /// actually took, which is what a caller must book.
    ///
    /// A voxel holds around 1e11 particles, where an `f32` step is 16384. A
    /// corpse handing back its last few thousand particles, or a cell pushing
    /// a waste product out through its membrane, is asking for a change below
    /// half a step: the addition rounds to nothing, the transfer silently
    /// fails, and -- because both sides check that they only give away what
    /// actually landed -- it fails again on every subsequent tick. Corpses
    /// stop decomposing and sit in the world for ever. Deferring the
    /// remainder in the residual, exactly as diffusion and the reaction step
    /// do, means a transfer that is too small to see today is paid tomorrow.
    #[inline]
    pub fn settle(
        &mut self,
        residual: &mut ChemField,
        compound: usize,
        voxel: usize,
        delta: f64,
    ) -> f64 {
        let before = self.get(compound, voxel);
        let carry = residual.get(compound, voxel);
        let exact = before as f64 + carry as f64 + delta;
        let after = exact as f32;
        let new_carry = (exact - after as f64) as f32;
        self.set(compound, voxel, after);
        residual.set(compound, voxel, new_carry);
        (after as f64 + new_carry as f64) - (before as f64 + carry as f64)
    }

    /// All voxels of one compound, contiguous.
    #[inline]
    pub fn plane(&self, compound: usize) -> &[f32] {
        let start = compound * self.n_voxels;
        &self.data[start..start + self.n_voxels]
    }

    #[inline]
    pub fn plane_mut(&mut self, compound: usize) -> &mut [f32] {
        let start = compound * self.n_voxels;
        &mut self.data[start..start + self.n_voxels]
    }

    /// Total amount of one compound across the world, summed in `f64`.
    pub fn total_of(&self, compound: usize) -> f64 {
        self.plane(compound).iter().map(|&x| x as f64).sum()
    }

    /// Copy `count` voxels starting at `first` into a voxel-major buffer of
    /// `count * n_compounds` floats, so per-voxel chemistry runs on
    /// contiguous data.
    ///
    /// Done in blocks. A straight transpose reads one compound plane
    /// sequentially but writes with a stride of `n_compounds` floats, touching
    /// a fresh cache line every element; blocking keeps the working set of
    /// each pass inside L1 and roughly halves the cost.
    pub fn tile(&self, first: usize, count: usize, out: &mut Vec<f32>) {
        out.clear();
        out.resize(count * self.n_compounds, 0.0);
        for block in (0..count).step_by(TRANSPOSE_BLOCK) {
            let n = TRANSPOSE_BLOCK.min(count - block);
            for c in 0..self.n_compounds {
                let base = c * self.n_voxels + first + block;
                let src = &self.data[base..base + n];
                for (k, &v) in src.iter().enumerate() {
                    out[(block + k) * self.n_compounds + c] = v;
                }
            }
        }
    }

    /// Write a tile produced by [`ChemField::tile`] back into the field.
    pub fn untile(&mut self, first: usize, count: usize, tile: &[f32]) {
        for block in (0..count).step_by(TRANSPOSE_BLOCK) {
            let n = TRANSPOSE_BLOCK.min(count - block);
            for c in 0..self.n_compounds {
                let base = c * self.n_voxels + first + block;
                for k in 0..n {
                    self.data[base + k] = tile[(block + k) * self.n_compounds + c];
                }
            }
        }
    }
}

impl HashState for ChemField {
    fn hash_state(&self, h: &mut StateHasher) {
        h.usize(self.n_compounds);
        h.f32_slice(&self.data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_transfer_too_small_for_f32_is_deferred_rather_than_dropped() {
        let grid = Grid::new(2, 2, 2, 25.0e-6);
        let mut amounts = ChemField::new(&grid, 2);
        let mut residual = ChemField::new(&grid, 2);
        // A voxel near the pond's working concentration: one f32 step here is
        // about 16384 particles.
        amounts.set(0, 3, 2.0e11);
        // Not 2e11: one f32 step up here is 16384 particles, so the literal
        // itself lands 4096 short of what was asked for.
        let start = amounts.get(0, 3) as f64;

        let crumb = 1.0e3;
        let mut handed_over = 0.0;
        for _ in 0..100 {
            handed_over += amounts.settle(&mut residual, 0, 3, crumb);
        }

        assert!(
            (handed_over - 100.0 * crumb).abs() < 1.0,
            "the field claimed {handed_over} of {} particles",
            100.0 * crumb
        );
        let held = amounts.get(0, 3) as f64 + residual.get(0, 3) as f64;
        assert!(
            (held - (start + 100.0 * crumb)).abs() < 1.0,
            "field plus residual holds {held}, expected {}",
            start + 100.0 * crumb
        );
        assert!(
            amounts.get(0, 3) as f64 > start,
            "not one whole step ever landed"
        );
    }

    #[test]
    fn settling_leaves_untouched_voxels_alone() {
        let grid = Grid::new(2, 2, 2, 25.0e-6);
        let mut amounts = ChemField::new(&grid, 2);
        let mut residual = ChemField::new(&grid, 2);
        amounts.settle(&mut residual, 1, 5, 7.0);
        assert_eq!(amounts.get(1, 5), 7.0);
        assert_eq!(amounts.total_of(0), 0.0);
        assert_eq!(residual.total_of(1), 0.0);
    }

    fn grid() -> Grid {
        Grid::new(4, 3, 2, 25.0e-6)
    }

    #[test]
    fn tile_round_trips() {
        let g = grid();
        let mut f = ChemField::new(&g, 5);
        for c in 0..5 {
            for v in 0..g.len() {
                f.set(c, v, (c * 100 + v) as f32);
            }
        }
        let original = f.clone();
        let mut buf = Vec::new();
        f.tile(2, 6, &mut buf);
        assert_eq!(buf.len(), 6 * 5);
        // Voxel-major inside the tile.
        assert_eq!(buf[3], original.get(3, 2));
        assert_eq!(buf[4 * 5 + 1], original.get(1, 6));
        f.untile(2, 6, &buf);
        assert_eq!(f, original);
    }

    #[test]
    fn planes_are_contiguous_and_independent() {
        let g = grid();
        let mut f = ChemField::new(&g, 3);
        f.plane_mut(1).fill(7.0);
        assert_eq!(f.total_of(0), 0.0);
        assert_eq!(f.total_of(1), 7.0 * g.len() as f64);
        assert_eq!(f.total_of(2), 0.0);
    }

    #[test]
    fn scalar_statistics() {
        let g = grid();
        let mut s = ScalarField::new(&g, 2.0);
        s.set(0, 10.0);
        assert_eq!(s.min_max(), (2.0, 10.0));
        assert_eq!(s.total(), 2.0 * (g.len() - 1) as f64 + 10.0);
    }
}
