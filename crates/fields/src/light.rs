//! The light field.
//!
//! Not a wave solver. Light is transported by band-limited radiative transfer:
//! eight spectral bands, attenuated down each vertical column by Beer-Lambert,
//! in a single pass. It costs one sweep of the grid and gives the world its
//! most important structure -- a vertical resource gradient that exists from
//! the first tick.
//!
//! Eight bands is the right number because it is enough for absorption spectra
//! to differentiate meaningfully, cheap to propagate, and maps directly onto
//! rendering. When a lineage later evolves a pigment that absorbs strongly in
//! band 2, the cell changes colour in the viewer because the renderer reads the
//! same eight numbers the physics does. Evolution becomes visible with no
//! instrumentation.
//!
//! Scattering is skipped, per the design. Every joule that enters the top face
//! is deposited somewhere -- absorbed in the column, or, if it reaches the
//! bottom, into the floor voxel. Nothing leaves through the base. That makes
//! the incident flux the world's entire radiative input and gives the audit a
//! single number to check against.

use hadean_chem::chemistry::{Chemistry, CompoundId, N_BANDS};
use hadean_core::units::Joules;
use hadean_core::Grid;
use serde::{Deserialize, Serialize};

use crate::scalar::ChemField;

/// A compound is worth attenuating for if its cross-section reaches this, in
/// m^2. Below it, a voxel's worth of the stuff changes nothing, and the
/// medium's own extinction covers it.
const OPTICALLY_ACTIVE: f32 = 1.0e-26;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LightConfig {
    /// Downward irradiance at the surface at local noon, W/m^2 per band.
    pub irradiance: [f32; N_BANDS],
    /// Attenuation by the medium itself, 1/m per band. The pond is only a
    /// millimetre or two deep, so a physically realistic value would produce
    /// no gradient at all; this is deliberately dense enough for depth to
    /// mean something.
    pub medium_extinction: [f32; N_BANDS],
    /// Day length in seconds. Zero holds the sun at noon, which is what you
    /// want while debugging an audit.
    pub day_length: f64,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            // Roughly a clear day, weighted toward the middle of the range.
            irradiance: [40.0, 70.0, 95.0, 110.0, 105.0, 85.0, 55.0, 30.0],
            // Blue is attenuated hardest, so depth also shifts the spectrum
            // and not just the brightness -- two gradients for the price of one.
            medium_extinction: [220.0, 260.0, 300.0, 350.0, 420.0, 520.0, 650.0, 820.0],
            day_length: 600.0,
        }
    }
}

impl LightConfig {
    /// Fraction of noon irradiance at elapsed time `t`.
    pub fn daylight(&self, t: f64) -> f32 {
        if self.day_length <= 0.0 {
            return 1.0;
        }
        let phase = (t / self.day_length) * std::f64::consts::TAU;
        phase.sin().max(0.0) as f32
    }

    /// Total noon irradiance across all bands, W/m^2.
    pub fn total_irradiance(&self) -> f32 {
        self.irradiance.iter().sum()
    }
}

/// What one propagation step let into the world.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Insolation {
    /// Energy incident on the surface, J. The physical quantity.
    pub incident: Joules,
    /// Energy actually deposited in voxels, J, summed from the per-voxel
    /// values the chemistry will read.
    ///
    /// This is what the audit must use. `incident` is computed in `f64` from
    /// the configured irradiance, while what the world receives is the sum of
    /// per-voxel `f32` deposits, and the two differ by a rounding of about
    /// 1e-7 relative. Booking the intended figure and delivering the rounded
    /// one leaves a small, *systematic* discrepancy every tick -- which is
    /// exactly the shape of leak that a drift trend catches and a drift
    /// threshold does not.
    pub absorbed: Joules,
}

/// Per-voxel light state, band-major.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightField {
    pub n_voxels: usize,
    /// Downward irradiance passing through each voxel, W/m^2. What a
    /// photoreceptor reads.
    pub intensity: Vec<f32>,
    /// Energy absorbed in each voxel this step, J. What chemistry consumes.
    pub absorbed: Vec<f32>,
    /// Scratch: downward flux still in flight, per column and band. Lives here
    /// so a step allocates nothing.
    #[serde(skip)]
    flux: Vec<f32>,
}

impl LightField {
    pub fn new(grid: &Grid) -> Self {
        let n = grid.len();
        Self {
            n_voxels: n,
            intensity: vec![0.0; n * N_BANDS],
            absorbed: vec![0.0; n * N_BANDS],
            flux: Vec::new(),
        }
    }

    #[inline]
    pub fn index(&self, band: usize, voxel: usize) -> usize {
        band * self.n_voxels + voxel
    }

    #[inline]
    pub fn intensity_at(&self, band: usize, voxel: usize) -> f32 {
        self.intensity[self.index(band, voxel)]
    }

    #[inline]
    pub fn absorbed_at(&self, band: usize, voxel: usize) -> f32 {
        self.absorbed[self.index(band, voxel)]
    }

    /// All bands absorbed at one voxel, as the chemistry step wants them.
    pub fn absorbed_bands(&self, voxel: usize) -> [f32; N_BANDS] {
        let mut out = [0.0f32; N_BANDS];
        for (b, o) in out.iter_mut().enumerate() {
            *o = self.absorbed[b * self.n_voxels + voxel];
        }
        out
    }

    /// Total intensity over all bands at a voxel, W/m^2.
    pub fn brightness(&self, voxel: usize) -> f32 {
        (0..N_BANDS).map(|b| self.intensity_at(b, voxel)).sum()
    }

    /// Total energy absorbed everywhere this step, J.
    pub fn total_absorbed(&self) -> Joules {
        self.absorbed.iter().map(|&x| x as f64).sum()
    }
}

/// Marches light down the columns.
///
/// Holds the list of compounds actually worth attenuating for. Most of a
/// generated chemistry is transparent, and skipping it turns the inner loop
/// from "every compound" into "the handful of pigments".
#[derive(Debug, Clone)]
pub struct LightSolver {
    pub config: LightConfig,
    active: Vec<CompoundId>,
}

impl LightSolver {
    pub fn new(chem: &Chemistry, config: LightConfig) -> Self {
        let active = chem
            .compounds
            .iter()
            .filter(|c| c.absorption.iter().any(|&s| s >= OPTICALLY_ACTIVE))
            .map(|c| c.id)
            .collect();
        Self { config, active }
    }

    /// Compounds dense enough optically to matter.
    pub fn active_compounds(&self) -> &[CompoundId] {
        &self.active
    }

    /// March light down every column, returning what entered and what was
    /// actually deposited.
    ///
    /// Iteration is depth-outermost rather than column-outermost, which needs
    /// one row of in-flight flux per column but makes every read and write
    /// sequential: the compound planes, the intensity planes and the absorbed
    /// planes are all walked in order. Column-outermost strides through all
    /// three by a whole layer per step, and on a pond-sized grid that costs
    /// more than the rest of the tick put together.
    ///
    /// Optical depth for all eight bands is accumulated in a single pass over
    /// the pigments, so a voxel's amounts are read once rather than once per
    /// band.
    pub fn propagate(
        &self,
        grid: &Grid,
        chem: &Chemistry,
        amounts: &ChemField,
        elapsed: f64,
        dt: f64,
        out: &mut LightField,
    ) -> Insolation {
        let daylight = self.config.daylight(elapsed);
        let face_area = grid.face_area() as f64;
        let layer = grid.layer();
        let nz = grid.nz as usize;
        // Optical depth contributed by one particle in one voxel, per band.
        let per_particle = 1.0 / (grid.dx * grid.dx);

        out.intensity.fill(0.0);
        out.absorbed.fill(0.0);

        if daylight <= 0.0 {
            return Insolation::default();
        }

        // Downward flux in flight, per column and band, W/m^2.
        let flux = &mut out.flux;
        flux.clear();
        flux.resize(layer * N_BANDS, 0.0);
        let mut incident = 0.0f64;
        for column in 0..layer {
            for band in 0..N_BANDS {
                let flux_in = self.config.irradiance[band] * daylight;
                flux[column * N_BANDS + band] = flux_in;
                incident += flux_in as f64 * face_area * dt;
            }
        }

        let medium_tau: [f32; N_BANDS] =
            std::array::from_fn(|b| self.config.medium_extinction[b] * grid.dx);
        let scale = (face_area * dt) as f32;

        for z in 0..nz {
            let base = z * layer;
            let bottom = z + 1 == nz;
            for column in 0..layer {
                let voxel = base + column;

                let mut tau = medium_tau;
                for &c in &self.active {
                    let n = amounts.get(c as usize, voxel);
                    if n > 0.0 {
                        let sigma = &chem.compounds[c as usize].absorption;
                        let k = n * per_particle;
                        for b in 0..N_BANDS {
                            tau[b] += k * sigma[b];
                        }
                    }
                }

                let row = column * N_BANDS;
                for band in 0..N_BANDS {
                    let here = flux[row + band];
                    if here <= 0.0 {
                        continue;
                    }
                    out.intensity[band * out.n_voxels + voxel] = here;

                    let mut absorbed = here * (1.0 - (-tau[band]).exp());
                    let mut remaining = here - absorbed;

                    // The floor absorbs whatever reaches it, so no energy
                    // leaves through the base and the audit has one input to
                    // account for rather than two.
                    if bottom {
                        absorbed += remaining;
                        remaining = 0.0;
                    }

                    flux[row + band] = remaining;
                    out.absorbed[band * out.n_voxels + voxel] = absorbed * scale;
                }
            }
        }

        Insolation {
            incident,
            absorbed: out.total_absorbed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hadean_chem::{generate, ChemParams};

    fn setup() -> (Grid, Chemistry, ChemField) {
        let grid = Grid::new(6, 5, 12, 25.0e-6);
        let chem = generate(17, ChemParams::default());
        let amounts = ChemField::new(&grid, chem.n_compounds());
        (grid, chem, amounts)
    }

    #[test]
    fn every_joule_that_enters_is_deposited() {
        // The audit depends on this exactly: light in must equal light
        // absorbed, with nothing leaking out of the bottom.
        let (grid, chem, mut amounts) = setup();
        for c in 0..chem.n_compounds() {
            amounts.plane_mut(c).fill(1.0e9);
        }
        let solver = LightSolver::new(
            &chem,
            LightConfig {
                day_length: 0.0,
                ..Default::default()
            },
        );
        let mut light = LightField::new(&grid);
        let sun = solver.propagate(&grid, &chem, &amounts, 0.0, 0.01, &mut light);
        assert!(sun.incident > 0.0);
        assert_eq!(sun.absorbed, light.total_absorbed());
        // Nothing escapes: what lands must match what arrived, up to the f32
        // rounding of the per-voxel deposits.
        assert!(
            (sun.absorbed - sun.incident).abs() / sun.incident < 1e-5,
            "incident {:e}, absorbed {:e}",
            sun.incident,
            sun.absorbed
        );
    }

    #[test]
    fn intensity_falls_with_depth() {
        let (grid, chem, amounts) = setup();
        let solver = LightSolver::new(
            &chem,
            LightConfig {
                day_length: 0.0,
                ..Default::default()
            },
        );
        let mut light = LightField::new(&grid);
        solver.propagate(&grid, &chem, &amounts, 0.0, 0.01, &mut light);
        let top = light.brightness(grid.idx(3, 2, 0));
        let bottom = light.brightness(grid.idx(3, 2, grid.nz - 1));
        assert!(top > bottom, "no vertical gradient: {top} -> {bottom}");
        for z in 1..grid.nz {
            let above = light.brightness(grid.idx(3, 2, z - 1));
            let here = light.brightness(grid.idx(3, 2, z));
            assert!(here <= above, "intensity rose with depth at z = {z}");
        }
    }

    #[test]
    fn depth_reddens_the_spectrum() {
        // Blue is attenuated hardest, so the deep world is not merely dimmer.
        let (grid, chem, amounts) = setup();
        let solver = LightSolver::new(
            &chem,
            LightConfig {
                day_length: 0.0,
                ..Default::default()
            },
        );
        let mut light = LightField::new(&grid);
        solver.propagate(&grid, &chem, &amounts, 0.0, 0.01, &mut light);
        let ratio = |z: u32| {
            let v = grid.idx(3, 2, z);
            light.intensity_at(7, v) / light.intensity_at(0, v).max(f32::MIN_POSITIVE)
        };
        assert!(
            ratio(grid.nz - 1) < ratio(0),
            "spectrum did not shift with depth"
        );
    }

    #[test]
    fn pigment_shades_what_is_below_it() {
        let (grid, chem, mut amounts) = setup();
        let solver = LightSolver::new(
            &chem,
            LightConfig {
                day_length: 0.0,
                ..Default::default()
            },
        );
        assert!(
            !solver.active_compounds().is_empty(),
            "chemistry has no pigments"
        );
        let mut light = LightField::new(&grid);

        solver.propagate(&grid, &chem, &amounts, 0.0, 0.01, &mut light);
        let clear = light.brightness(grid.idx(3, 2, grid.nz - 1));

        // Put a dense pigment layer just under the surface.
        let pigment = solver.active_compounds()[0] as usize;
        for column in 0..grid.layer() {
            amounts.set(pigment, column + grid.layer(), 4.0e11);
        }
        solver.propagate(&grid, &chem, &amounts, 0.0, 0.01, &mut light);
        let shaded = light.brightness(grid.idx(3, 2, grid.nz - 1));

        assert!(
            shaded < clear,
            "pigment cast no shadow: {clear} -> {shaded}"
        );
    }

    #[test]
    fn night_is_dark() {
        let (grid, chem, amounts) = setup();
        let cfg = LightConfig {
            day_length: 100.0,
            ..Default::default()
        };
        let solver = LightSolver::new(&chem, cfg);
        let mut light = LightField::new(&grid);
        // Half a day in, the sun is below the horizon.
        let sun = solver.propagate(&grid, &chem, &amounts, 75.0, 0.01, &mut light);
        assert_eq!(sun.incident, 0.0);
        assert_eq!(sun.absorbed, 0.0);
        assert_eq!(light.total_absorbed(), 0.0);
        assert!(light.intensity.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn daylight_peaks_at_midday() {
        let cfg = LightConfig {
            day_length: 100.0,
            ..Default::default()
        };
        assert!((cfg.daylight(25.0) - 1.0).abs() < 1e-6);
        assert_eq!(cfg.daylight(75.0), 0.0);
        assert!(cfg.daylight(0.0).abs() < 1e-6);
        // A fixed sun is what you want while chasing an audit failure.
        let fixed = LightConfig {
            day_length: 0.0,
            ..Default::default()
        };
        assert_eq!(fixed.daylight(12345.0), 1.0);
    }
}
