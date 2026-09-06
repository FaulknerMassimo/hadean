//! Diffusion and advection, both in conservative face-flux form.
//!
//! Every transport step is written as an exchange across voxel faces rather
//! than as a divergence evaluated per voxel. The reason is conservation: the
//! flux one voxel computes across a face is the exact IEEE negation of what
//! its neighbour computes across the same face, because the only difference is
//! the order of a subtraction. Nothing is created or destroyed at a face, and
//! a closed boundary simply has no faces to cross.
//!
//! Both passes read a previous buffer and write a new one, so each output
//! voxel depends only on inputs. That makes them safe to parallelise and, more
//! importantly, makes the result independent of how the work is split.
//! Per-voxel accumulation is done in `f64` so a voxel rounds once per step
//! rather than once per face.
//!
//! Each operation comes in two forms. The serial one is for callers that
//! already have parallelism to spend -- the chemical field runs one compound
//! per thread, which is better granularity than splitting every compound's
//! sweep across every core and paying the dispatch each time. The `_parallel`
//! one is for a lone field with nothing else to overlap with, such as heat.
//!
//! # Compensation
//!
//! Exact face fluxes are not enough on their own. A voxel holds around 1e11
//! particles, where an `f32` step is about 16000, and once a field has
//! smoothed out its net flux per step falls *below half a step* -- so the
//! update rounds back to where it started and the flux is silently dropped.
//! The dropped amounts do not cancel between voxels, and the result is a
//! steady, one-directional leak: measured against ticks it grows as a straight
//! line, which is exactly the signature of a bug rather than of noise.
//!
//! Every operation therefore carries a per-voxel `residual`: what the last
//! rounding threw away is added back before the next one. Nothing is
//! discarded, only deferred. With it, `sum(values) + sum(residual)` is
//! conserved to `f64` precision indefinitely, and the residual stays bounded
//! at under half a step per voxel instead of accumulating.

use std::sync::OnceLock;

use hadean_core::{Axis, Grid};
use rayon::prelude::*;

/// The explicit diffusion stability limit in 3D: `D dt / dx^2 <= 1/6`.
const STABILITY: f32 = 1.0 / 6.0;

/// How many substeps a diffusion pass needs to stay stable.
///
/// The explicit stencil blows up above the limit, so a fast-diffusing species
/// is stepped several times per tick rather than once. Hydrogen sets the pace.
pub fn substeps(d: f32, dt: f64, dx: f32) -> u32 {
    let k = d as f64 * dt / (dx as f64 * dx as f64);
    (k / STABILITY as f64).ceil().max(1.0) as u32
}

/// Store a compensated update, returning the value to write and the new
/// residual.
///
/// `delta` is the exact change this voxel should undergo. Adding the carried
/// residual before rounding is what stops sub-ULP updates from vanishing.
#[inline]
fn settle(centre: f32, delta: f64, residual: f32) -> (f32, f32) {
    let exact = centre as f64 + delta + residual as f64;
    let rounded = exact as f32;
    (rounded, (exact - rounded as f64) as f32)
}

/// Diffuse one scalar field in place, on this thread.
///
/// `residual` carries rounding forward between calls and must persist for the
/// life of the field; `scratch` is a reusable double buffer and need not.
pub fn diffuse(
    grid: &Grid,
    values: &mut [f32],
    residual: &mut [f32],
    d: f32,
    dt: f64,
    scratch: &mut Vec<f32>,
) {
    sweep_diffusion(grid, values, residual, d, dt, scratch, false);
}

/// Diffuse one scalar field, splitting the sweep across the thread pool.
pub fn diffuse_parallel(
    grid: &Grid,
    values: &mut [f32],
    residual: &mut [f32],
    d: f32,
    dt: f64,
    scratch: &mut Vec<f32>,
) {
    sweep_diffusion(grid, values, residual, d, dt, scratch, true);
}

fn sweep_diffusion(
    grid: &Grid,
    values: &mut [f32],
    residual: &mut [f32],
    d: f32,
    dt: f64,
    scratch: &mut Vec<f32>,
    parallel: bool,
) {
    if d <= 0.0 {
        return;
    }
    debug_assert_eq!(values.len(), residual.len());
    let n = substeps(d, dt, grid.dx);
    let sub_dt = dt / n as f64;
    let k = (d as f64 * sub_dt / (grid.dx as f64 * grid.dx as f64)) as f32;
    for _ in 0..n {
        scratch.clear();
        scratch.extend_from_slice(values);
        let old: &[f32] = scratch;
        if parallel {
            values
                .par_iter_mut()
                .zip(residual.par_iter_mut())
                .enumerate()
                .for_each(|(i, (out, carry))| {
                    let (v, r) = settle(old[i], diffusion_delta(grid, old, i, k), *carry);
                    *out = v;
                    *carry = r;
                });
        } else {
            for (i, (out, carry)) in values.iter_mut().zip(residual.iter_mut()).enumerate() {
                let (v, r) = settle(old[i], diffusion_delta(grid, old, i, k), *carry);
                *out = v;
                *carry = r;
            }
        }
    }
}

/// Net change at one voxel of an explicit diffusion sweep,
/// `k * sum(neighbour - self)`.
///
/// The two voxels sharing a face evaluate the identical product with opposite
/// signs, so a face's contribution cancels exactly and the sum over the grid
/// is zero to the last bit.
#[inline]
fn diffusion_delta(grid: &Grid, old: &[f32], i: usize, k: f32) -> f64 {
    let (nx, ny, nz) = (grid.nx as usize, grid.ny as usize, grid.nz as usize);
    let layer = nx * ny;
    let x = i % nx;
    let y = (i / nx) % ny;
    let z = i / layer;
    let centre = old[i];
    let mut acc = 0.0f64;
    if x > 0 {
        acc += (k * (old[i - 1] - centre)) as f64;
    }
    if x + 1 < nx {
        acc += (k * (old[i + 1] - centre)) as f64;
    }
    if y > 0 {
        acc += (k * (old[i - nx] - centre)) as f64;
    }
    if y + 1 < ny {
        acc += (k * (old[i + nx] - centre)) as f64;
    }
    if z > 0 {
        acc += (k * (old[i - layer] - centre)) as f64;
    }
    if z + 1 < nz {
        acc += (k * (old[i + layer] - centre)) as f64;
    }
    acc
}

/// Staggered face velocities for a flow field, m/s.
///
/// `u[axis]` holds the velocity on the faces normal to that axis. There is one
/// more face than voxel along the axis in question, and the outermost faces
/// are the closed world boundary, where the velocity is always zero.
#[derive(Debug, Clone)]
pub struct FaceVelocity {
    u: [Vec<f32>; 3],
    grid: Grid,
    /// Static flows are applied to many compounds, so do the O(voxels) CFL
    /// reduction only once. `set` invalidates this during flow construction.
    max_outflow_speed: OnceLock<f32>,
}

impl PartialEq for FaceVelocity {
    fn eq(&self, other: &Self) -> bool {
        // The cache is derived state and must not affect physical equality.
        self.u == other.u && self.grid == other.grid
    }
}

impl FaceVelocity {
    pub fn zero(grid: &Grid) -> Self {
        let (nx, ny, nz) = (grid.nx as usize, grid.ny as usize, grid.nz as usize);
        Self {
            u: [
                vec![0.0; (nx + 1) * ny * nz],
                vec![0.0; nx * (ny + 1) * nz],
                vec![0.0; nx * ny * (nz + 1)],
            ],
            grid: *grid,
            max_outflow_speed: OnceLock::new(),
        }
    }

    /// Index of the face at `(x, y, z)` normal to `axis`, where the face lies
    /// on the low side of that voxel.
    #[inline]
    pub fn face(&self, axis: Axis, x: usize, y: usize, z: usize) -> usize {
        let (nx, ny) = (self.grid.nx as usize, self.grid.ny as usize);
        match axis {
            Axis::X => x + (nx + 1) * (y + ny * z),
            Axis::Y => x + nx * (y + (ny + 1) * z),
            Axis::Z => x + nx * (y + ny * z),
        }
    }

    #[inline]
    pub fn get(&self, axis: Axis, x: usize, y: usize, z: usize) -> f32 {
        self.u[axis as usize][self.face(axis, x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, axis: Axis, x: usize, y: usize, z: usize, v: f32) {
        let i = self.face(axis, x, y, z);
        self.u[axis as usize][i] = v;
        self.max_outflow_speed.take();
    }

    /// Largest speed on any face.
    pub fn max_speed(&self) -> f32 {
        self.u
            .iter()
            .flat_map(|a| a.iter())
            .fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    /// Net outflow from each voxel, per unit volume. Should be zero
    /// everywhere for an incompressible field.
    pub fn max_divergence(&self) -> f32 {
        let g = self.grid;
        let (nx, ny, nz) = (g.nx as usize, g.ny as usize, g.nz as usize);
        let mut worst = 0.0f32;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let d = (self.get(Axis::X, x + 1, y, z) - self.get(Axis::X, x, y, z))
                        + (self.get(Axis::Y, x, y + 1, z) - self.get(Axis::Y, x, y, z))
                        + (self.get(Axis::Z, x, y, z + 1) - self.get(Axis::Z, x, y, z));
                    worst = worst.max(d.abs());
                }
            }
        }
        worst / g.dx
    }

    /// Largest total outward speed from any voxel.
    ///
    /// Donor-cell advection stays positive only when the *sum* of the
    /// Courant numbers on every outgoing face is at most one. Looking at the
    /// fastest individual face is insufficient in multiple dimensions: a
    /// voxel can empty through several faces at once.
    pub fn max_outflow_speed(&self) -> f32 {
        *self
            .max_outflow_speed
            .get_or_init(|| self.compute_max_outflow_speed())
    }

    fn compute_max_outflow_speed(&self) -> f32 {
        let (nx, ny, nz) = (
            self.grid.nx as usize,
            self.grid.ny as usize,
            self.grid.nz as usize,
        );
        let mut worst = 0.0f32;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let outward = (-self.get(Axis::X, x, y, z)).max(0.0)
                        + self.get(Axis::X, x + 1, y, z).max(0.0)
                        + (-self.get(Axis::Y, x, y, z)).max(0.0)
                        + self.get(Axis::Y, x, y + 1, z).max(0.0)
                        + (-self.get(Axis::Z, x, y, z)).max(0.0)
                        + self.get(Axis::Z, x, y, z + 1).max(0.0);
                    worst = worst.max(outward);
                }
            }
        }
        worst
    }

    /// Substeps needed to keep every voxel's total outward Courant number at
    /// or below one.
    pub fn substeps(&self, dt: f64, dx: f32) -> u32 {
        let courant = self.max_outflow_speed() as f64 * dt / dx as f64;
        courant.ceil().max(1.0) as u32
    }
}

/// Advect a scalar field with donor-cell upwinding.
///
/// First-order and diffusive, but unconditionally conservative, which is what
/// matters here: the mass audit must not have to make allowances for the
/// transport scheme.
pub fn advect(
    grid: &Grid,
    values: &mut [f32],
    residual: &mut [f32],
    velocity: &FaceVelocity,
    dt: f64,
    scratch: &mut Vec<f32>,
) {
    debug_assert_eq!(values.len(), residual.len());
    let n = velocity.substeps(dt, grid.dx);
    let sub_dt = dt / n as f64;
    let c = (sub_dt / grid.dx as f64) as f32;
    for _ in 0..n {
        scratch.clear();
        scratch.extend_from_slice(values);
        let old: &[f32] = scratch;
        for (i, (out, carry)) in values.iter_mut().zip(residual.iter_mut()).enumerate() {
            let (v, r) = settle(old[i], upwind_delta(grid, old, i, velocity, c), *carry);
            // Upwinding cannot legitimately empty a voxel past zero, but the
            // carried residual can push a nearly-empty one slightly negative;
            // clamp and keep the difference in the residual so it is not
            // quietly turned into new material.
            if v < 0.0 {
                *carry = r + v;
                *out = 0.0;
            } else {
                *out = v;
                *carry = r;
            }
        }
    }
}

/// Net change at one voxel from donor-cell advection.
#[inline]
fn upwind_delta(grid: &Grid, old: &[f32], i: usize, vel: &FaceVelocity, c: f32) -> f64 {
    let (nx, ny, nz) = (grid.nx as usize, grid.ny as usize, grid.nz as usize);
    let layer = nx * ny;
    {
        let x = i % nx;
        let y = (i / nx) % ny;
        let z = i / layer;
        let mut acc = 0.0f64;

        // Face velocities are positive in the +axis direction. Across the low
        // face that carries material in; across the high face it carries it
        // out. In both cases the amount crossing is the velocity times the
        // *donor* voxel's content -- that is what makes the scheme stable.
        //
        // The two voxels sharing a face evaluate the same product and add it
        // with opposite signs, so a face neither creates nor destroys
        // anything, exactly.
        let here = old[i];
        let low_face = |u: f32, there: f32| {
            let donor = if u > 0.0 { there } else { here };
            (c * u * donor) as f64
        };
        let high_face = |u: f32, there: f32| {
            let donor = if u > 0.0 { here } else { there };
            -((c * u * donor) as f64)
        };

        acc += low_face(
            vel.get(Axis::X, x, y, z),
            if x > 0 { old[i - 1] } else { 0.0 },
        );
        acc += high_face(
            vel.get(Axis::X, x + 1, y, z),
            if x + 1 < nx { old[i + 1] } else { 0.0 },
        );
        acc += low_face(
            vel.get(Axis::Y, x, y, z),
            if y > 0 { old[i - nx] } else { 0.0 },
        );
        acc += high_face(
            vel.get(Axis::Y, x, y + 1, z),
            if y + 1 < ny { old[i + nx] } else { 0.0 },
        );
        acc += low_face(
            vel.get(Axis::Z, x, y, z),
            if z > 0 { old[i - layer] } else { 0.0 },
        );
        acc += high_face(
            vel.get(Axis::Z, x, y, z + 1),
            if z + 1 < nz { old[i + layer] } else { 0.0 },
        );

        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid::new(12, 10, 8, 25.0e-6)
    }

    fn total(v: &[f32]) -> f64 {
        v.iter().map(|&x| x as f64).sum()
    }

    /// Values plus carried residual: the quantity that is actually conserved.
    fn conserved(v: &[f32], r: &[f32]) -> f64 {
        total(v) + total(r)
    }

    #[test]
    fn diffusion_conserves_mass() {
        let g = grid();
        let mut v = vec![0.0f32; g.len()];
        let mut r = vec![0.0f32; g.len()];
        v[g.idx(6, 5, 4)] = 1.0e9;
        let before = conserved(&v, &r);
        let mut scratch = Vec::new();
        for _ in 0..200 {
            diffuse(&g, &mut v, &mut r, 2.0e-9, 0.01, &mut scratch);
        }
        let after = conserved(&v, &r);
        assert!(
            (after - before).abs() / before < 1e-9,
            "mass drifted from {before:e} to {after:e}"
        );
    }

    #[test]
    fn a_smoothed_field_does_not_leak_over_a_long_run() {
        // The case that motivates compensation. A voxel holding 2e11 particles
        // has an f32 step of 16384; once the field flattens, a step's net flux
        // is smaller than that and would round away entirely. Dropped flux
        // does not cancel between voxels, so without compensation this drifts
        // in one direction, linearly in ticks.
        let g = Grid::new(16, 16, 10, 25.0e-6);
        let mut v: Vec<f32> = (0..g.len())
            .map(|i| 2.0e11 * (1.0 + 0.25 * ((i % 7) as f32 / 7.0 - 0.5)))
            .collect();
        let mut r = vec![0.0f32; g.len()];
        let before = conserved(&v, &r);
        let mut scratch = Vec::new();
        for _ in 0..20_000 {
            diffuse(&g, &mut v, &mut r, 2.3e-9, 0.01, &mut scratch);
        }
        let after = conserved(&v, &r);
        let drift = (after - before).abs() / before;
        assert!(drift < 1e-12, "drifted by {drift:e} over 20k steps");

        // And the residual must stay bounded rather than accumulating.
        let worst = r.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(worst < 65_536.0, "residual grew to {worst}");
    }

    #[test]
    fn diffusion_spreads_and_flattens() {
        let g = grid();
        let mut v = vec![0.0f32; g.len()];
        let mut r = vec![0.0f32; g.len()];
        let centre = g.idx(6, 5, 4);
        v[centre] = 1.0e9;
        let mut scratch = Vec::new();
        for _ in 0..500 {
            diffuse(&g, &mut v, &mut r, 2.0e-9, 0.01, &mut scratch);
        }
        let mean = total(&v) / g.len() as f64;
        assert!(v[centre] as f64 > mean, "peak should still be the peak");
        assert!(v[g.idx(0, 0, 0)] > 0.0, "should have reached the corner");
        assert!(v.iter().all(|&x| x.is_finite() && x >= 0.0));
    }

    #[test]
    fn diffusion_reaches_a_flat_steady_state() {
        let g = Grid::new(6, 6, 6, 25.0e-6);
        let mut v = vec![0.0f32; g.len()];
        let mut r = vec![0.0f32; g.len()];
        v[0] = 1.0e6;
        let mut scratch = Vec::new();
        for _ in 0..20_000 {
            diffuse(&g, &mut v, &mut r, 2.0e-9, 0.01, &mut scratch);
        }
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, 0.0f32), |(a, b), &x| (a.min(x), b.max(x)));
        assert!((hi - lo) / hi < 1e-3, "did not flatten: {lo} .. {hi}");
    }

    #[test]
    fn substep_count_tracks_the_stability_limit() {
        let dx = 25.0e-6f32;
        let dt = 0.01;
        let limit = STABILITY * dx * dx / dt as f32;
        // Just inside the limit: one substep. Just outside: more than one.
        assert_eq!(substeps(limit * 0.99, dt, dx), 1);
        assert!(substeps(limit * 1.01, dt, dx) > 1);
        // The substep count must always restore stability.
        for factor in [0.5f32, 1.0, 3.7, 40.0] {
            let d = limit * factor;
            let n = substeps(d, dt, dx);
            let k = d as f64 * (dt / n as f64) / (dx as f64 * dx as f64);
            assert!(
                k <= STABILITY as f64 + 1e-12,
                "d = {d:e} still unstable at k = {k}"
            );
        }
        assert_eq!(substeps(0.0, dt, dx), 1);
    }

    #[test]
    fn advection_conserves_mass_and_moves_material() {
        let g = grid();
        let mut vel = FaceVelocity::zero(&g);
        // Uniform flow in +x, with the boundary faces left at zero so the
        // world stays closed.
        for z in 0..g.nz as usize {
            for y in 0..g.ny as usize {
                for x in 1..g.nx as usize {
                    vel.set(Axis::X, x, y, z, 5.0e-4);
                }
            }
        }
        let mut v = vec![0.0f32; g.len()];
        let mut r = vec![0.0f32; g.len()];
        v[g.idx(2, 5, 4)] = 1.0e9;
        let before = conserved(&v, &r);
        let mut scratch = Vec::new();
        for _ in 0..100 {
            advect(&g, &mut v, &mut r, &vel, 0.01, &mut scratch);
        }
        let after = conserved(&v, &r);
        assert!(
            (after - before).abs() / before < 1e-9,
            "{before:e} -> {after:e}"
        );
        let downstream: f64 = (3..g.nx).map(|x| v[g.idx(x, 5, 4)] as f64).sum();
        assert!(downstream > 0.0, "nothing moved");
        assert!(
            v.iter().all(|&x| x >= 0.0),
            "advection produced a negative amount"
        );
    }

    #[test]
    fn advection_cfl_accounts_for_simultaneous_multiaxis_outflow() {
        let g = Grid::new(3, 3, 3, 1.0);
        let mut vel = FaceVelocity::zero(&g);
        let (x, y, z) = (1, 1, 1);
        assert_eq!(vel.max_outflow_speed(), 0.0);

        // Each face is individually below CFL 1, but together the three
        // faces remove 1.8 times the centre voxel's contents in one step.
        // The old max-single-face check selected one step and relied on the
        // negative-value clamp; the multidimensional bound must select two.
        vel.set(Axis::X, x + 1, y, z, 0.6);
        vel.set(Axis::Y, x, y + 1, z, 0.6);
        vel.set(Axis::Z, x, y, z + 1, 0.6);
        assert_eq!(vel.max_speed(), 0.6);
        assert!((vel.max_outflow_speed() - 1.8).abs() < 1.0e-6);
        assert_eq!(vel.substeps(1.0, g.dx), 2);

        let centre = g.idx(x as u32, y as u32, z as u32);
        let mut values = vec![0.0; g.len()];
        let mut residual = vec![0.0; g.len()];
        values[centre] = 1.0;
        let before = conserved(&values, &residual);
        let mut scratch = Vec::new();
        advect(&g, &mut values, &mut residual, &vel, 1.0, &mut scratch);

        assert!(values[centre] > 0.0, "the source voxel was overdrawn");
        assert!(values.iter().all(|&amount| amount >= 0.0));
        let after = conserved(&values, &residual);
        assert!((after - before).abs() < 1.0e-7, "{before:e} -> {after:e}");
    }

    #[test]
    fn advection_with_no_flow_changes_nothing() {
        let g = grid();
        let vel = FaceVelocity::zero(&g);
        let mut v: Vec<f32> = (0..g.len()).map(|i| (i % 17) as f32).collect();
        let mut r = vec![0.0f32; g.len()];
        let original = v.clone();
        let mut scratch = Vec::new();
        advect(&g, &mut v, &mut r, &vel, 0.01, &mut scratch);
        assert_eq!(v, original);
        assert!(r.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn transport_is_independent_of_thread_count() {
        // The double-buffered form must give bit-identical results however
        // rayon happens to split the work.
        let g = grid();
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut v = vec![0.0f32; g.len()];
                let mut r = vec![0.0f32; g.len()];
                v[g.idx(6, 5, 4)] = 1.0e9;
                let mut scratch = Vec::new();
                for _ in 0..50 {
                    diffuse_parallel(&g, &mut v, &mut r, 2.0e-9, 0.01, &mut scratch);
                }
                (v, r)
            })
        };
        assert_eq!(run(1), run(7));
    }

    #[test]
    fn serial_and_parallel_diffusion_agree_exactly() {
        let g = grid();
        let seed = |i: usize| 1.0e11 * (1.0 + 0.3 * ((i % 5) as f32 / 5.0 - 0.5));
        let mut va: Vec<f32> = (0..g.len()).map(seed).collect();
        let mut vb = va.clone();
        let mut ra = vec![0.0f32; g.len()];
        let mut rb = vec![0.0f32; g.len()];
        let mut scratch = Vec::new();
        for _ in 0..40 {
            diffuse(&g, &mut va, &mut ra, 2.0e-9, 0.01, &mut scratch);
            diffuse_parallel(&g, &mut vb, &mut rb, 2.0e-9, 0.01, &mut scratch);
        }
        assert_eq!(va, vb);
        assert_eq!(ra, rb);
    }
}
