//! The heat field.
//!
//! Temperature diffuses, exothermic chemistry warms it, endothermic chemistry
//! cools it, and the surface loses heat to the sky. Every reaction rate
//! depends on it through the Arrhenius term, which closes a real feedback
//! loop: metabolism warms the water, warm water speeds metabolism.
//!
//! **The field stores a deviation from a reference temperature, not an
//! absolute one.** This is not a stylistic choice. A voxel is 1.6e-14 m^3, so
//! its heat capacity is 6.5e-8 J/K and a typical reaction deposits a few
//! picojoules -- microkelvins. At 300 K an `f32` resolves about 3e-5 K, so
//! storing absolute temperature would round most of the world's chemistry away
//! and the energy audit would bleed by tens of percent. Around zero the same
//! `f32` resolves 1e-45. Diffusion is linear and the world is closed, so a
//! constant offset transports identically; nothing is lost by moving the
//! origin.
//!
//! The thermal diffusivity of water is about sixty times the molecular
//! diffusivity of a small solute, so heat needs far more substeps than
//! chemistry does. That is a real cost and the first thing to make implicit
//! when it becomes the bottleneck.

use hadean_core::hash::{HashState, StateHasher};
use hadean_core::units::{Joules, Kelvin, HEAT_CAPACITY_VOL, THERMAL_DIFFUSIVITY};
use hadean_core::Grid;
use serde::{Deserialize, Serialize};

use crate::scalar::ScalarField;
use crate::transport::diffuse_parallel;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HeatConfig {
    /// Starting temperature everywhere, K. Also the reference the field
    /// stores deviations from.
    pub ambient: Kelvin,
    /// Temperature the surface exchanges heat with, K.
    pub sky: Kelvin,
    /// Heat transfer coefficient across the surface, W/(m^2 K). Zero makes
    /// the world adiabatic, which is how you isolate an audit failure.
    ///
    /// This is a property of the water-air boundary, not of the mesh, so the
    /// equilibrium temperature does not move when the grid is refined. It
    /// converts to a per-second relaxation rate for the top layer by dividing
    /// by the layer's heat capacity per unit area, `rho c dx`.
    ///
    /// The default is 60, which is high next to the 10-25 W/(m^2 K) of bare
    /// convection because an open water surface loses most of its heat as
    /// latent heat of evaporation, not as sensible heat. It matters: the pond
    /// is half a millimetre deep and holds 3 mJ/K, so it takes in roughly its
    /// entire heat capacity every four seconds of sunlight. Under-couple the
    /// surface and the world does not run warm, it boils -- at 20 W/(m^2 K)
    /// the equilibrium is 46 C, and at the original rate-based 0.02/s there
    /// was no equilibrium below boiling at all.
    pub surface_transfer: f32,
    /// Heat injected by each vent, W.
    pub vent_power: f32,
    /// Number of vents along the floor.
    pub vents: u32,
}

impl Default for HeatConfig {
    fn default() -> Self {
        Self {
            ambient: hadean_core::units::T_AMBIENT,
            sky: hadean_core::units::T_AMBIENT - 4.0,
            surface_transfer: 60.0,
            vent_power: 2.0e-9,
            vents: 3,
        }
    }
}

/// Temperature, stored as a deviation from [`HeatField::reference`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeatField {
    /// The origin the deviations are measured from, K.
    pub reference: Kelvin,
    /// `T - reference`, per voxel.
    pub deviation: ScalarField,
    /// Rounding carried forward by the diffusion pass. See
    /// [`crate::transport`]; without it, a smoothed field leaks steadily.
    pub residual: ScalarField,
}

impl HeatField {
    pub fn new(grid: &Grid, reference: Kelvin) -> Self {
        Self {
            reference,
            deviation: ScalarField::new(grid, 0.0),
            residual: ScalarField::new(grid, 0.0),
        }
    }

    /// Absolute temperature of a voxel, K.
    #[inline]
    pub fn temperature(&self, voxel: usize) -> Kelvin {
        self.reference + self.deviation.get(voxel)
    }

    #[inline]
    pub fn set_temperature(&mut self, voxel: usize, t: Kelvin) {
        self.deviation.set(voxel, t - self.reference);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.deviation.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deviation.is_empty()
    }

    /// Deposit energy into one voxel as heat, returning what actually landed.
    ///
    /// Any `f32` rounding is retained in [`Self::residual`], so even a sub-ULP
    /// deposit remains owned by the field. Callers crossing the world boundary
    /// -- vents, say -- must still book the return value rather than what they
    /// asked for, keeping accounting tied to representable state.
    #[inline]
    pub fn deposit(&mut self, grid: &Grid, voxel: usize, joules: Joules) -> Joules {
        let before = self.deviation.get(voxel);
        let carry = self.residual.get(voxel);
        let capacity = voxel_heat_capacity(grid);
        let exact = before as f64 + carry as f64 + joules / capacity;
        let after = exact as f32;
        let new_carry = (exact - after as f64) as f32;
        self.deviation.set(voxel, after);
        self.residual.set(voxel, new_carry);
        ((after as f64 + new_carry as f64) - (before as f64 + carry as f64)) * capacity
    }

    /// Total thermal energy relative to the reference temperature, J.
    ///
    /// Relative, not absolute: the absolute thermal energy of the medium is
    /// enormous next to the chemical energy the audit cares about, and would
    /// bury it in rounding.
    ///
    /// Includes the residual from transport and deposits: energy the field
    /// owns but has not yet been able to represent. Leaving it out would make
    /// the audit report a leak that is really just deferred rounding.
    pub fn energy(&self, grid: &Grid) -> Joules {
        (self.deviation.total() + self.residual.total()) * voxel_heat_capacity(grid)
    }

    /// Mean absolute temperature, K.
    pub fn mean_temperature(&self) -> f64 {
        self.reference as f64 + self.deviation.mean()
    }

    /// Coldest and warmest absolute temperatures, K.
    pub fn range(&self) -> (Kelvin, Kelvin) {
        let (lo, hi) = self.deviation.min_max();
        (self.reference + lo, self.reference + hi)
    }
}

impl HashState for HeatField {
    fn hash_state(&self, h: &mut StateHasher) {
        h.f32(self.reference);
        self.deviation.hash_state(h);
        self.residual.hash_state(h);
    }
}

/// Heat capacity of one voxel, J/K.
#[inline]
pub fn voxel_heat_capacity(grid: &Grid) -> f64 {
    HEAT_CAPACITY_VOL * grid.voxel_volume() as f64
}

/// Temperature change produced by depositing `joules` into one voxel.
#[inline]
pub fn temperature_delta(joules: Joules, grid: &Grid) -> f32 {
    (joules / voxel_heat_capacity(grid)) as f32
}

/// What the heat step exchanged with the outside world.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HeatExchange {
    /// Energy lost through the surface, J. Positive means the world cooled.
    pub radiated: Joules,
    /// Energy injected by vents, J.
    pub vented: Joules,
}

/// Advance the heat field: vents in, diffusion through, surface loss out.
pub fn step(
    grid: &Grid,
    heat: &mut HeatField,
    cfg: &HeatConfig,
    dt: f64,
    scratch: &mut Vec<f32>,
) -> HeatExchange {
    let mut exchange = HeatExchange::default();

    // Vents: point sources along the floor. Together with the light gradient
    // from above, these give the world two independent ways to make a living,
    // which is what produces divergence rather than a monoculture.
    if cfg.vents > 0 && cfg.vent_power != 0.0 {
        let energy = cfg.vent_power as f64 * dt;
        for v in vent_voxels(grid, cfg.vents) {
            exchange.vented += heat.deposit(grid, v, energy);
        }
    }

    diffuse_parallel(
        grid,
        &mut heat.deviation.data,
        &mut heat.residual.data,
        THERMAL_DIFFUSIVITY,
        dt,
        scratch,
    );

    // Newton cooling across the top face.
    //
    // The exchange is a flux through an area, so it has to be turned into a
    // rate for the layer that carries it. A thinner surface voxel holds less
    // heat behind the same square metre of sky and therefore relaxes faster;
    // dividing by `rho c dx` is what keeps the pond's equilibrium temperature
    // a fact about the water rather than about the mesh.
    if cfg.surface_transfer > 0.0 {
        let sky_deviation = cfg.sky - heat.reference;
        let relaxation = cfg.surface_transfer as f64 / (HEAT_CAPACITY_VOL * grid.dx as f64);
        let k = (relaxation * dt).min(1.0) as f32;
        let mut lost = 0.0f64;
        for i in 0..grid.layer() {
            let before = heat.deviation.get(i);
            let after = before - k * (before - sky_deviation);
            heat.deviation.set(i, after);
            lost += (before - after) as f64;
        }
        exchange.radiated = lost * voxel_heat_capacity(grid);
    }

    exchange
}

/// Evenly spaced voxels along the floor where vents sit.
pub fn vent_voxels(grid: &Grid, vents: u32) -> Vec<usize> {
    let z = grid.nz - 1;
    let y = grid.ny / 2;
    (0..vents)
        .map(|v| {
            let x = ((v as u64 + 1) * grid.nx as u64 / (vents as u64 + 1)) as u32;
            grid.idx(x.min(grid.nx - 1), y, z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid::new(16, 8, 10, 25.0e-6)
    }

    fn quiet() -> HeatConfig {
        HeatConfig {
            surface_transfer: 0.0,
            vents: 0,
            ..Default::default()
        }
    }

    #[test]
    fn deposited_energy_reappears_as_temperature() {
        let g = grid();
        let mut h = HeatField::new(&g, 293.15);
        let joules = 5.0e-9;
        h.deposit(&g, 3, joules);
        let measured = h.energy(&g);
        assert!(
            (measured - joules).abs() / joules < 1e-6,
            "{measured:e} vs {joules:e}"
        );
        assert!(h.temperature(3) > 293.15);
        assert_eq!(h.temperature(4), 293.15);
    }

    #[test]
    fn diffusion_alone_conserves_thermal_energy() {
        // This is the test that a field of absolute temperatures fails: the
        // increments are far below the f32 resolution of 300 K.
        let g = grid();
        let mut h = HeatField::new(&g, 293.15);
        h.deposit(&g, g.idx(8, 4, 5), 1.0e-8);
        let before = h.energy(&g);
        let mut scratch = Vec::new();
        for _ in 0..200 {
            let x = step(&g, &mut h, &quiet(), 0.01, &mut scratch);
            assert_eq!(x.radiated, 0.0);
            assert_eq!(x.vented, 0.0);
        }
        let after = h.energy(&g);
        assert!(
            (after - before).abs() / before < 1e-4,
            "{before:e} -> {after:e}"
        );
    }

    #[test]
    fn tiny_reaction_scale_deposits_are_not_lost() {
        // A single reaction in a voxel releases picojoules. Ten thousand of
        // them must add up to what they should.
        let g = grid();
        let mut h = HeatField::new(&g, 293.15);
        let quantum = 1.0e-15;
        let mut landed = 0.0;
        for i in 0..10_000 {
            landed += h.deposit(&g, i % g.len(), quantum);
        }
        let expected = quantum * 10_000.0;
        let measured = h.energy(&g);
        assert!(
            (measured - expected).abs() / expected < 1e-4,
            "{measured:e} vs {expected:e}"
        );
        assert!(
            (landed - expected).abs() / expected < 1e-6,
            "reported {landed:e} vs {expected:e}"
        );
    }

    #[test]
    fn surface_cooling_is_a_flux_through_an_area_not_a_mesh_rate() {
        // The same pond, meshed twice: 400 x 400 x 200 um, once at 50 um and
        // once at 25 um. A boundary condition written as a per-second rate on
        // the top layer halves its cooling when the layer is halved, and the
        // pond's equilibrium temperature becomes a property of the mesh. As a
        // flux it does not.
        let cfg = HeatConfig {
            sky: 283.15,
            surface_transfer: 20.0,
            vents: 0,
            vent_power: 0.0,
            ..Default::default()
        };
        let lost_from = |nx: u32, dx: f32| -> f64 {
            let g = Grid::new(nx, nx, nx / 2, dx);
            let mut h = HeatField::new(&g, cfg.ambient);
            let mut scratch = Vec::new();
            let mut radiated = 0.0;
            for _ in 0..100 {
                radiated += step(&g, &mut h, &cfg, 0.01, &mut scratch).radiated;
            }
            radiated
        };

        let coarse = lost_from(8, 50.0e-6);
        let fine = lost_from(16, 25.0e-6);
        // Newton cooling of the pond's 10 K excess over the sky, through
        // 1.6e-7 m^2, for one second. Conduction through 200 um of water
        // takes about 0.3 s, so the column tracks its surface and the excess
        // is still very nearly 10 K at the end.
        let expected = 20.0 * 1.6e-7 * (cfg.ambient - cfg.sky) as f64 * 1.0;
        for (label, measured) in [("coarse", coarse), ("fine", fine)] {
            assert!(
                (measured - expected).abs() / expected < 0.1,
                "{label} mesh lost {measured:e} J, expected about {expected:e}"
            );
        }
        assert!(
            (coarse - fine).abs() / fine < 0.1,
            "refining the mesh changed the heat loss: {coarse:e} vs {fine:e}"
        );
    }

    #[test]
    fn the_surface_loses_exactly_what_it_reports() {
        let g = grid();
        let cfg = HeatConfig {
            sky: 290.0,
            surface_transfer: 20.0,
            vents: 0,
            ..Default::default()
        };
        let mut h = HeatField::new(&g, cfg.ambient);
        let mut scratch = Vec::new();
        let before = h.energy(&g);
        let mut radiated = 0.0;
        for _ in 0..100 {
            radiated += step(&g, &mut h, &cfg, 0.01, &mut scratch).radiated;
        }
        let expected = before - h.energy(&g);
        assert!(radiated > 0.0);
        assert!(
            (radiated - expected).abs() / expected.abs() < 1e-4,
            "reported {radiated:e}, actually lost {expected:e}"
        );
    }

    #[test]
    fn vents_warm_the_floor_and_the_surface_stays_cooler() {
        let g = grid();
        let cfg = HeatConfig {
            vents: 3,
            vent_power: 1.0e-8,
            ..Default::default()
        };
        let mut h = HeatField::new(&g, cfg.ambient);
        let mut scratch = Vec::new();
        for _ in 0..2000 {
            step(&g, &mut h, &cfg, 0.01, &mut scratch);
        }
        let layer = g.layer();
        let mean = |base: usize| -> f64 {
            (0..layer)
                .map(|i| h.temperature(base + i) as f64)
                .sum::<f64>()
                / layer as f64
        };
        let floor = mean(layer * (g.nz as usize - 1));
        let surface = mean(0);
        assert!(
            floor > surface,
            "no vertical thermal gradient: {floor} vs {surface}"
        );
    }

    #[test]
    fn vents_report_what_they_inject() {
        let g = grid();
        let cfg = HeatConfig {
            vents: 3,
            vent_power: 1.0e-8,
            surface_transfer: 0.0,
            ..Default::default()
        };
        let mut h = HeatField::new(&g, cfg.ambient);
        let mut scratch = Vec::new();
        let mut vented = 0.0;
        for _ in 0..500 {
            vented += step(&g, &mut h, &cfg, 0.01, &mut scratch).vented;
        }
        let measured = h.energy(&g);
        assert!(
            (measured - vented).abs() / vented < 1e-4,
            "{measured:e} vs {vented:e}"
        );
    }

    #[test]
    fn vent_placement_is_on_the_floor_and_distinct() {
        let g = grid();
        let vents = vent_voxels(&g, 3);
        assert_eq!(vents.len(), 3);
        let mut seen = vents.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 3, "vents must not coincide");
        for v in vents {
            assert_eq!(g.depth_of(v), g.nz - 1);
        }
    }
}
