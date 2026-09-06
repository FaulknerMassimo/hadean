//! The energy and mass audit.
//!
//! Every tick the world's total energy is compared against what it should be:
//! the baseline plus everything that entered, minus everything that left. The
//! design is blunt about why this matters, and it is right -- if the audit
//! drifts upward, organisms will find the leak and evolve into perpetual
//! motion machines, usually within an hour of wall-clock time. A failing audit
//! is a build failure, not a warning.
//!
//! Three things make the audit meaningful rather than decorative:
//!
//! * Reaction enthalpies are differences of a state function, so no cycle of
//!   reactions can net energy (see `hadean_chem`).
//! * Transport is in face-flux form, and both transport and reaction rounding
//!   are carried as conserved residuals, so neither can destroy material.
//! * The kinetics step derives heat from the *measured* change in chemical
//!   energy rather than from intended extents, so clamping and rounding land
//!   in the heat term instead of vanishing.
//!
//! Everything is accumulated in `f64`, in a fixed order, so the numbers do not
//! depend on thread count.

use hadean_cell::{add_element_totals, Population};
use hadean_chem::element::N_ELEMENTS;
use hadean_chem::Chemistry;
use hadean_core::units::{Joules, Kelvin};
use hadean_core::Grid;
use hadean_fields::heat::HeatField;
use hadean_fields::scalar::ChemField;
use serde::{Deserialize, Serialize};

/// A measurement of where the world's energy is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyState {
    /// Sum of `amount * formation enthalpy` over every compound, J.
    pub chemical: Joules,
    /// Thermal energy relative to the reference temperature, J.
    pub thermal: Joules,
    /// Chemical contents and captured free energy held inside cells, J.
    pub cellular: Joules,
}

impl EnergyState {
    pub fn total(&self) -> Joules {
        self.chemical + self.thermal + self.cellular
    }
}

/// Cumulative flows across the world boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    /// Radiant energy that entered through the surface, J.
    pub light_in: Joules,
    /// Heat injected by vents, J.
    pub vent_heat_in: Joules,
    /// Chemical energy carried in by vent-injected matter, J. Negative,
    /// because a bound compound sits below the free-atom zero -- injecting
    /// matter lowers the world's chemical energy total even as it raises its
    /// usable free energy.
    pub vent_chemical_in: Joules,
    /// Heat lost through the surface, J.
    pub radiated_out: Joules,
}

impl Ledger {
    pub fn net_in(&self) -> Joules {
        self.light_in + self.vent_heat_in + self.vent_chemical_in - self.radiated_out
    }

    /// Total energy that has crossed the boundary in either direction. The
    /// denominator for a drift ratio that means something.
    pub fn throughput(&self) -> Joules {
        self.light_in.abs()
            + self.vent_heat_in.abs()
            + self.vent_chemical_in.abs()
            + self.radiated_out.abs()
    }
}

/// One audit reading.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    pub tick: u64,
    pub energy: EnergyState,
    /// What the total should be, from the baseline and the ledger.
    pub expected: Joules,
    /// Measured minus expected. Positive means the world is making energy.
    pub drift: Joules,
    /// Drift as a fraction of the world's total energy.
    pub relative: f64,
    /// Drift as a fraction of everything that has crossed the boundary.
    /// Undefined, and reported as zero, before anything has.
    pub relative_to_flow: f64,
    /// Worst per-element mass drift, as a fraction.
    pub mass_drift: f64,
}

impl AuditReport {
    /// Does this reading pass at the given relative tolerance?
    pub fn passes(&self, tolerance: f64) -> bool {
        self.relative.abs() <= tolerance && self.mass_drift.abs() <= tolerance
    }
}

/// Tracks the baseline and the ledger, and produces readings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub reference: Kelvin,
    pub baseline: EnergyState,
    pub baseline_elements: [f64; N_ELEMENTS],
    pub ledger: Ledger,
    /// Elements introduced by vents since the start.
    pub elements_in: [f64; N_ELEMENTS],
    pub last: Option<AuditReport>,
}

impl Audit {
    /// Start an audit from the world's current state.
    pub fn new(
        reference: Kelvin,
        grid: &Grid,
        chem: &Chemistry,
        amounts: &ChemField,
        residual: &ChemField,
        heat: &HeatField,
        cells: &Population,
    ) -> Self {
        Self {
            reference,
            baseline: measure(grid, chem, amounts, residual, heat, cells),
            baseline_elements: element_totals(chem, amounts, residual, cells),
            ledger: Ledger::default(),
            elements_in: [0.0; N_ELEMENTS],
            last: None,
        }
    }

    /// Record matter injected at the boundary.
    pub fn record_injection(&mut self, chem: &Chemistry, compound: u16, amount: f64) {
        let c = chem.compound(compound);
        self.ledger.vent_chemical_in += amount * c.h_f;
        for e in 0..N_ELEMENTS {
            self.elements_in[e] += amount * c.formula[e] as f64;
        }
    }

    /// Take a reading.
    #[allow(clippy::too_many_arguments)]
    pub fn report(
        &mut self,
        tick: u64,
        grid: &Grid,
        chem: &Chemistry,
        amounts: &ChemField,
        residual: &ChemField,
        heat: &HeatField,
        cells: &Population,
    ) -> AuditReport {
        let energy = measure(grid, chem, amounts, residual, heat, cells);
        let expected = self.baseline.total() + self.ledger.net_in();
        let drift = energy.total() - expected;

        let scale = energy.total().abs().max(self.baseline.total().abs());
        let relative = if scale > 0.0 { drift / scale } else { 0.0 };
        let flow = self.ledger.throughput();
        let relative_to_flow = if flow > 0.0 { drift / flow } else { 0.0 };

        let elements = element_totals(chem, amounts, residual, cells);
        let mut mass_drift = 0.0f64;
        for (e, &measured) in elements.iter().enumerate() {
            let expected_e = self.baseline_elements[e] + self.elements_in[e];
            if expected_e > 0.0 {
                let d = (measured - expected_e) / expected_e;
                if d.abs() > mass_drift.abs() {
                    mass_drift = d;
                }
            }
        }

        let report = AuditReport {
            tick,
            energy,
            expected,
            drift,
            relative,
            relative_to_flow,
            mass_drift,
        };
        self.last = Some(report);
        report
    }
}

/// Where the world's energy is right now.
pub fn measure(
    grid: &Grid,
    chem: &Chemistry,
    amounts: &ChemField,
    residual: &ChemField,
    heat: &HeatField,
    cells: &Population,
) -> EnergyState {
    // Sum each compound's amount across the world first, then multiply by its
    // enthalpy once. Summing `n * h_f` per voxel instead would multiply
    // millions of tiny products and lose precision for no reason.
    let chemical = (0..chem.n_compounds())
        .map(|c| (amounts.total_of(c) + residual.total_of(c)) * chem.compounds[c].h_f)
        .sum();
    EnergyState {
        chemical,
        thermal: heat.energy(grid),
        cellular: cells.stored_energy(chem),
    }
}

/// Total count of each element in the world.
pub fn element_totals(
    chem: &Chemistry,
    amounts: &ChemField,
    residual: &ChemField,
    cells: &Population,
) -> [f64; N_ELEMENTS] {
    let mut out = [0.0f64; N_ELEMENTS];
    for c in 0..chem.n_compounds() {
        let total = amounts.total_of(c) + residual.total_of(c);
        if total == 0.0 {
            continue;
        }
        let formula = &chem.compounds[c].formula;
        for e in 0..N_ELEMENTS {
            if formula[e] > 0 {
                out[e] += total * formula[e] as f64;
            }
        }
    }
    add_element_totals(cells, chem, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hadean_chem::{generate, ChemParams};

    fn setup() -> (Grid, Chemistry, ChemField, HeatField, Population) {
        let grid = Grid::new(8, 8, 6, 25.0e-6);
        let chem = generate(31, ChemParams::default());
        let mut amounts = ChemField::new(&grid, chem.n_compounds());
        for c in 0..chem.n_compounds() {
            amounts.plane_mut(c).fill(1.0e9 * (1 + c % 4) as f32);
        }
        let heat = HeatField::new(&grid, 293.15);
        // No cells: this is the audit of a lifeless world.
        let cells = Population::empty();
        (grid, chem, amounts, heat, cells)
    }

    #[test]
    fn an_untouched_world_does_not_drift() {
        let (grid, chem, amounts, heat, cells) = setup();
        let residual = ChemField::new(&grid, chem.n_compounds());
        let mut audit = Audit::new(293.15, &grid, &chem, &amounts, &residual, &heat, &cells);
        let r = audit.report(100, &grid, &chem, &amounts, &residual, &heat, &cells);
        assert_eq!(r.drift, 0.0);
        assert_eq!(r.mass_drift, 0.0);
        assert!(r.passes(1e-12));
    }

    #[test]
    fn unrecorded_energy_is_caught() {
        let (grid, chem, amounts, mut heat, cells) = setup();
        let residual = ChemField::new(&grid, chem.n_compounds());
        let mut audit = Audit::new(293.15, &grid, &chem, &amounts, &residual, &heat, &cells);
        // Heat appearing from nowhere is exactly the bug this exists to find.
        heat.deposit(&grid, 5, 1.0e-6);
        let r = audit.report(1, &grid, &chem, &amounts, &residual, &heat, &cells);
        assert!(r.drift > 0.0, "phantom energy went unnoticed");
        assert!(!r.passes(1e-6));
    }

    #[test]
    fn recorded_energy_balances() {
        let (grid, chem, amounts, mut heat, cells) = setup();
        let residual = ChemField::new(&grid, chem.n_compounds());
        let mut audit = Audit::new(293.15, &grid, &chem, &amounts, &residual, &heat, &cells);
        let joules = 1.0e-6;
        heat.deposit(&grid, 5, joules);
        audit.ledger.light_in += joules;
        let r = audit.report(1, &grid, &chem, &amounts, &residual, &heat, &cells);
        assert!(r.relative.abs() < 1e-9, "drift {:e}", r.drift);
        assert!(r.passes(1e-6));
    }

    #[test]
    fn unrecorded_matter_is_caught() {
        let (grid, chem, mut amounts, heat, cells) = setup();
        let residual = ChemField::new(&grid, chem.n_compounds());
        let mut audit = Audit::new(293.15, &grid, &chem, &amounts, &residual, &heat, &cells);
        amounts.add(0, 7, 1.0e12);
        let r = audit.report(1, &grid, &chem, &amounts, &residual, &heat, &cells);
        assert!(r.mass_drift.abs() > 1e-6, "phantom matter went unnoticed");
    }

    #[test]
    fn recorded_injection_balances_both_mass_and_energy() {
        let (grid, chem, mut amounts, heat, cells) = setup();
        let residual = ChemField::new(&grid, chem.n_compounds());
        let mut audit = Audit::new(293.15, &grid, &chem, &amounts, &residual, &heat, &cells);
        let fuel = chem.vent_fuel[0];
        let amount = 5.0e11f64;
        amounts.add(fuel as usize, 3, amount as f32);
        audit.record_injection(&chem, fuel, amount);
        let r = audit.report(1, &grid, &chem, &amounts, &residual, &heat, &cells);
        assert!(r.relative.abs() < 1e-6, "energy drift {:e}", r.relative);
        assert!(r.mass_drift.abs() < 1e-6, "mass drift {:e}", r.mass_drift);
    }

    #[test]
    fn transport_residual_is_conserved_matter_and_energy() {
        let (grid, chem, mut amounts, heat, cells) = setup();
        let mut residual = ChemField::new(&grid, chem.n_compounds());
        let mut audit = Audit::new(293.15, &grid, &chem, &amounts, &residual, &heat, &cells);

        let compound = 3;
        let carried = 16_384.0;
        amounts.set(compound, 7, amounts.get(compound, 7) - carried);
        residual.set(compound, 7, carried);

        let r = audit.report(1, &grid, &chem, &amounts, &residual, &heat, &cells);
        assert_eq!(r.drift, 0.0);
        assert_eq!(r.mass_drift, 0.0);
    }

    #[test]
    fn the_ledger_nets_out() {
        let l = Ledger {
            light_in: 10.0,
            vent_heat_in: 2.0,
            vent_chemical_in: -3.0,
            radiated_out: 4.0,
        };
        assert_eq!(l.net_in(), 5.0);
        assert_eq!(l.throughput(), 19.0);
    }
}
