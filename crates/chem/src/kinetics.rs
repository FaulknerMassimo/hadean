//! Running the reaction network inside one voxel.
//!
//! Two invariants govern this module, and both are enforced by construction
//! rather than by hoping the arithmetic works out.
//!
//! **Mass.** Extents are applied through each reaction's stoichiometry, and
//! stoichiometry came from a graph rewrite. Independent `f32` rounding would
//! still break that balance numerically, so every update carries its rounding
//! residual into the next reaction and tick.
//!
//! **Energy.** The heat released is computed as `absorbed_light - change in
//! chemical energy`, measured from the amounts before and after the step. It
//! is not computed from the intended extents. That distinction matters: when
//! an extent is clamped, or an amount rounds in `f32`, the heat term absorbs
//! the difference and the global audit stays flat. Deriving heat from intended
//! extents instead would leak energy every time a clamp fired.
//!
//! Reactions are applied sequentially in id order, each clamped against the
//! amounts left by the previous one. Sequential application is order-dependent
//! -- but the order is fixed, so it is reproducible, and it makes a negative
//! amount impossible.
//!
//! # Why there is a [`Network`] and not just a [`Chemistry`]
//!
//! [`Chemistry`] is the readable form: molecules, named compounds, reactions
//! with `Vec` sides. Stepping it directly means chasing two pointers per
//! reaction per voxel and evaluating two exponentials for the Arrhenius
//! factors -- on a pond-sized grid, eight million `exp` calls per tick, which
//! measured as three quarters of the whole simulation. [`Network`] is the same
//! chemistry flattened for the inner loop, with rate constants precomputed
//! across temperature. Build it once per world; step it every tick.

use crate::chemistry::{band_energy, Chemistry, Drive, N_BANDS};
use hadean_core::units::{arrhenius, Joules, Kelvin, N_REF};

/// The largest fraction of any reactant a single reaction may consume in one
/// step. Keeps fast reactions stable without an implicit solver.
const MAX_CONSUMED_FRACTION: f32 = 0.2;

/// Amounts below this are treated as absent.
const NEGLIGIBLE: f32 = 1.0e-3;

/// Temperature range the rate table covers, K. Outside it, rates are clamped
/// to the nearest end -- the world should never get there, and a wrong rate at
/// 600 K is better than a branch in the inner loop.
const TABLE_MIN: Kelvin = 250.0;
const TABLE_MAX: Kelvin = 450.0;
const TABLE_ROWS: usize = 2048;

/// What one voxel's chemistry did this step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VoxelStep {
    /// Heat released into the voxel, J. Negative if the net chemistry was
    /// endothermic and drew heat from the medium.
    pub heat: Joules,
    /// Light energy stored in chemical bonds, J. This is the world's primary
    /// energy input and is tracked separately for the audit.
    pub photo_stored: Joules,
    /// Change in chemical energy, J.
    pub chem_delta: Joules,
}

/// Rate constants precomputed over temperature.
///
/// Rows are temperature-major so one voxel reads a contiguous run of
/// reactions. Values are interpolated between rows, so the table is smooth in
/// temperature and the metabolism-warms-the-water feedback loop stays
/// continuous rather than stepping.
#[derive(Debug, Clone)]
pub struct RateTable {
    t_min: Kelvin,
    inv_step: f32,
    rows: usize,
    n_reactions: usize,
    forward: Vec<f32>,
    reverse: Vec<f32>,
}

impl RateTable {
    pub fn new(chem: &Chemistry) -> Self {
        let n_reactions = chem.n_reactions();
        let rows = TABLE_ROWS;
        let step = (TABLE_MAX - TABLE_MIN) / (rows - 1) as f32;
        let mut forward = vec![0.0f32; rows * n_reactions];
        let mut reverse = vec![0.0f32; rows * n_reactions];
        for row in 0..rows {
            let t = TABLE_MIN + row as f32 * step;
            for (r, reaction) in chem.reactions.iter().enumerate() {
                forward[row * n_reactions + r] = reaction.k_forward(t, 0.0);
                reverse[row * n_reactions + r] = reaction.k_reverse(t, 0.0);
            }
        }
        Self {
            t_min: TABLE_MIN,
            inv_step: 1.0 / step,
            rows,
            n_reactions,
            forward,
            reverse,
        }
    }

    /// Row index and interpolation weight for a temperature.
    #[inline]
    fn locate(&self, t: Kelvin) -> (usize, usize, f32) {
        let x = ((t - self.t_min) * self.inv_step).clamp(0.0, (self.rows - 1) as f32);
        let lo = x.floor() as usize;
        let hi = (lo + 1).min(self.rows - 1);
        (lo, hi, x - lo as f32)
    }

    /// Forward and reverse rate constants at `t`, as interpolated slices.
    #[inline]
    fn constants(&self, t: Kelvin) -> (&[f32], &[f32], &[f32], &[f32], f32) {
        let (lo, hi, frac) = self.locate(t);
        let n = self.n_reactions;
        (
            &self.forward[lo * n..lo * n + n],
            &self.forward[hi * n..hi * n + n],
            &self.reverse[lo * n..lo * n + n],
            &self.reverse[hi * n..hi * n + n],
            frac,
        )
    }
}

/// A chemistry flattened for stepping.
#[derive(Debug, Clone)]
pub struct Network {
    pub n_compounds: usize,
    pub n_reactions: usize,
    /// Reactant terms then product terms, indexed by the offsets below.
    terms: Vec<(u16, u8)>,
    reactant_at: Vec<u32>,
    reactant_len: Vec<u8>,
    product_at: Vec<u32>,
    product_len: Vec<u8>,
    dh: Vec<f64>,
    ea_f: Vec<f64>,
    ea_r: Vec<f64>,
    rate: Vec<f32>,
    /// Band a reaction is driven by, or -1 when it is thermal.
    photo_band: Vec<i8>,
    /// Precomputed strongest absorber among a photo reaction's reactants.
    photo_pigment: Vec<u16>,
    /// `absorption[compound * N_BANDS + band]`, m^2.
    absorption: Vec<f32>,
    h_f: Vec<f64>,
    photo_yield: f32,
    rates: RateTable,
}

impl Network {
    pub fn new(chem: &Chemistry) -> Self {
        let mut terms = Vec::new();
        let (mut reactant_at, mut reactant_len) = (Vec::new(), Vec::new());
        let (mut product_at, mut product_len) = (Vec::new(), Vec::new());
        let (mut dh, mut ea_f, mut ea_r, mut rate) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut photo_band, mut photo_pigment) = (Vec::new(), Vec::new());

        for r in &chem.reactions {
            reactant_at.push(terms.len() as u32);
            reactant_len.push(r.reactants.len() as u8);
            terms.extend_from_slice(&r.reactants);
            product_at.push(terms.len() as u32);
            product_len.push(r.products.len() as u8);
            terms.extend_from_slice(&r.products);

            dh.push(r.dh);
            ea_f.push(r.ea_f);
            ea_r.push(r.ea_r);
            rate.push(r.rate);

            match r.drive {
                Drive::Thermal => {
                    photo_band.push(-1);
                    photo_pigment.push(0);
                }
                Drive::Photo { band } => {
                    photo_band.push(band as i8);
                    // The pigment is whichever reactant absorbs most strongly
                    // in this band. Fixed for the life of the world, so it is
                    // resolved here rather than per voxel.
                    let pigment = r
                        .reactants
                        .iter()
                        .map(|&(c, _)| c)
                        .max_by(|&a, &b| {
                            chem.compounds[a as usize].absorption[band as usize]
                                .partial_cmp(&chem.compounds[b as usize].absorption[band as usize])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0);
                    photo_pigment.push(pigment);
                }
            }
        }

        let mut absorption = vec![0.0f32; chem.n_compounds() * N_BANDS];
        for c in &chem.compounds {
            for b in 0..N_BANDS {
                absorption[c.id as usize * N_BANDS + b] = c.absorption[b];
            }
        }

        Self {
            n_compounds: chem.n_compounds(),
            n_reactions: chem.n_reactions(),
            terms,
            reactant_at,
            reactant_len,
            product_at,
            product_len,
            dh,
            ea_f,
            ea_r,
            rate,
            photo_band,
            photo_pigment,
            absorption,
            h_f: chem.compounds.iter().map(|c| c.h_f).collect(),
            photo_yield: chem.params.photo_yield,
            rates: RateTable::new(chem),
        }
    }

    #[inline]
    fn reactants(&self, r: usize) -> &[(u16, u8)] {
        let at = self.reactant_at[r] as usize;
        &self.terms[at..at + self.reactant_len[r] as usize]
    }

    #[inline]
    fn products(&self, r: usize) -> &[(u16, u8)] {
        let at = self.product_at[r] as usize;
        &self.terms[at..at + self.product_len[r] as usize]
    }

    /// Advance the reaction network in one voxel.
    ///
    /// * `amounts` -- particle counts per compound, modified in place.
    /// * `residual` -- conserved per-compound rounding carried across field
    ///   reactions and transport.
    /// * `absorbed` -- light energy absorbed in this voxel this step, per band, J.
    /// * `catalysis` -- per-reaction activation energy reduction, J. Empty in
    ///   Phase 1; from L4 this is where enzymes act. A catalyst lowers both
    ///   barriers equally, so it can speed a reaction up but never move its
    ///   equilibrium. Only catalysed reactions pay for an exponential; the
    ///   rest read the table.
    pub fn step_voxel(
        &self,
        amounts: &mut [f32],
        residual: &mut [f32],
        temperature: Kelvin,
        dt: f64,
        absorbed: &[f32; N_BANDS],
        catalysis: &[f64],
    ) -> VoxelStep {
        debug_assert_eq!(amounts.len(), self.n_compounds);
        debug_assert_eq!(residual.len(), self.n_compounds);

        let chemical_before: f64 = amounts
            .iter()
            .zip(residual.iter())
            .zip(self.h_f.iter())
            .map(|((&amount, &carry), &enthalpy)| (amount as f64 + carry as f64) * enthalpy)
            .sum();

        let absorbed_total: f64 = absorbed.iter().map(|&a| a as f64).sum();

        // Share of each band's absorbed energy that landed on each compound.
        // Needed to work out how much light a given pigment actually captured.
        let mut band_weight = [0.0f64; N_BANDS];
        if absorbed_total > 0.0 {
            for (c, &n) in amounts.iter().enumerate() {
                if n <= NEGLIGIBLE {
                    continue;
                }
                let sigma = &self.absorption[c * N_BANDS..c * N_BANDS + N_BANDS];
                for b in 0..N_BANDS {
                    band_weight[b] += n as f64 * sigma[b] as f64;
                }
            }
        }

        let (f_lo, f_hi, r_lo, r_hi, frac) = self.rates.constants(temperature);
        let mut photo_stored = 0.0f64;

        for r in 0..self.n_reactions {
            let reduction = catalysis.get(r).copied().unwrap_or(0.0);

            let (k_f, k_r) = if reduction > 0.0 {
                // A catalyst moves this reaction off the table; it is the only
                // case that pays for exponentials.
                (
                    (self.rate[r] as f64
                        * arrhenius((self.ea_f[r] - reduction).max(0.0), temperature))
                        as f32,
                    (self.rate[r] as f64
                        * arrhenius((self.ea_r[r] - reduction).max(0.0), temperature))
                        as f32,
                )
            } else {
                (
                    f_lo[r] + frac * (f_hi[r] - f_lo[r]),
                    r_lo[r] + frac * (r_hi[r] - r_lo[r]),
                )
            };

            let band = self.photo_band[r];
            let forward = if band < 0 {
                mass_action(amounts, self.reactants(r)) * k_f as f64
            } else {
                self.photon_driven(r, band as usize, amounts, absorbed, &band_weight)
            };
            // The reverse of a photochemical reaction is ordinary downhill
            // chemistry, so it always runs thermally.
            let reverse = mass_action(amounts, self.products(r)) * k_r as f64;

            let mut extent = (forward - reverse) * dt;
            if extent == 0.0 || !extent.is_finite() {
                continue;
            }

            extent = self.clamp_extent(amounts, r, extent);
            if extent == 0.0 {
                continue;
            }

            for &(c, n) in self.reactants(r) {
                settle_amount(
                    &mut amounts[c as usize],
                    &mut residual[c as usize],
                    -extent * n as f64,
                );
            }
            for &(c, n) in self.products(r) {
                settle_amount(
                    &mut amounts[c as usize],
                    &mut residual[c as usize],
                    extent * n as f64,
                );
            }

            if band >= 0 && extent > 0.0 {
                photo_stored += extent * self.dh[r];
            }
        }

        // Chemical energy change measured from the state, not from the extents.
        let chemical_after: f64 = amounts
            .iter()
            .zip(residual.iter())
            .zip(self.h_f.iter())
            .map(|((&amount, &carry), &enthalpy)| (amount as f64 + carry as f64) * enthalpy)
            .sum();
        let chem_delta = chemical_after - chemical_before;

        VoxelStep {
            heat: absorbed_total - chem_delta,
            photo_stored,
            chem_delta,
        }
    }

    /// Forward extent of a photochemical reaction, in particles per second.
    ///
    /// The reaction can consume only the light its own pigment absorbed, which
    /// is its share of the band's total absorption in this voxel. Everything
    /// else thermalises. One photon drives at most one turnover, scaled by the
    /// quantum yield.
    fn photon_driven(
        &self,
        r: usize,
        band: usize,
        amounts: &[f32],
        absorbed: &[f32; N_BANDS],
        band_weight: &[f64; N_BANDS],
    ) -> f64 {
        let energy = absorbed[band] as f64;
        if energy <= 0.0 || band_weight[band] <= 0.0 {
            return 0.0;
        }
        let pigment = self.photo_pigment[r] as usize;
        let captured = amounts[pigment] as f64 * self.absorption[pigment * N_BANDS + band] as f64;
        if captured <= 0.0 {
            return 0.0;
        }
        let share = (captured / band_weight[band]).clamp(0.0, 1.0);
        let photons = energy * share / band_energy(band);
        photons * self.photo_yield as f64 * self.rate[r] as f64
    }

    /// Limit an extent so nothing is over-consumed and nothing goes negative.
    fn clamp_extent(&self, amounts: &[f32], r: usize, extent: f64) -> f64 {
        let consumed = if extent > 0.0 {
            self.reactants(r)
        } else {
            self.products(r)
        };
        let mut limit = extent.abs();
        for &(c, n) in consumed {
            let available = amounts[c as usize] * MAX_CONSUMED_FRACTION;
            if available <= 0.0 {
                return 0.0;
            }
            limit = limit.min(available as f64 * inv_coefficient(n));
        }
        if limit <= 0.0 {
            0.0
        } else if extent > 0.0 {
            limit
        } else {
            -limit
        }
    }
}

/// Apply a reaction delta to an `f32` amount without discarding the part that
/// the field cannot represent. The carried value belongs to the compound just
/// as much as the rounded amount does, so stoichiometric mass remains exact
/// across arbitrarily many reaction steps.
#[inline]
fn settle_amount(amount: &mut f32, residual: &mut f32, delta: f64) {
    let exact = *amount as f64 + *residual as f64 + delta;
    let rounded = exact as f32;
    if rounded < 0.0 {
        *amount = 0.0;
        *residual = exact as f32;
    } else {
        *amount = rounded;
        *residual = (exact - rounded as f64) as f32;
    }
}

/// Reciprocal of the reference amount, so mass action multiplies instead of
/// dividing. This loop runs a few million times a tick and a division is
/// twenty cycles.
const INV_N_REF: f64 = 1.0 / N_REF as f64;

/// Reciprocals of small stoichiometric coefficients, for the same reason.
const INV_COEFFICIENT: [f64; 5] = [1.0, 1.0, 0.5, 1.0 / 3.0, 0.25];

#[inline]
fn inv_coefficient(n: u8) -> f64 {
    INV_COEFFICIENT
        .get(n as usize)
        .copied()
        .unwrap_or_else(|| 1.0 / n as f64)
}

/// Mass-action concentration product, normalised by [`N_REF`] so that a rate
/// constant is in inverse seconds regardless of reaction order.
#[inline]
fn mass_action(amounts: &[f32], side: &[(u16, u8)]) -> f64 {
    let mut acc = N_REF as f64;
    for &(c, n) in side {
        let x = amounts[c as usize] as f64 * INV_N_REF;
        if x <= 0.0 {
            return 0.0;
        }
        acc *= match n {
            1 => x,
            2 => x * x,
            _ => x.powi(n as i32),
        };
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::N_ELEMENTS;
    use crate::generate::generate;
    use crate::molecule::Bond;
    use crate::{ChemParams, Compound, CompoundId, Molecule, Reaction, KEY_DIM};
    use hadean_core::units::{kt, T_AMBIENT};

    fn element_totals(chem: &Chemistry, amounts: &[f32], residual: &[f32]) -> [f64; N_ELEMENTS] {
        let mut out = [0.0f64; N_ELEMENTS];
        for (c, &n) in amounts.iter().enumerate() {
            for (e, total) in out.iter_mut().enumerate() {
                *total += (n as f64 + residual[c] as f64) * chem.compounds[c].formula[e] as f64;
            }
        }
        out
    }

    fn seeded_amounts(chem: &Chemistry) -> Vec<f32> {
        (0..chem.n_compounds())
            .map(|i| 1.0e9 * (1.0 + (i % 5) as f32))
            .collect()
    }

    #[test]
    fn a_step_conserves_every_element() {
        let chem = generate(4, ChemParams::default());
        let net = Network::new(&chem);
        let mut amounts = seeded_amounts(&chem);
        let mut residual = vec![0.0; chem.n_compounds()];
        let before = element_totals(&chem, &amounts, &residual);
        for _ in 0..50 {
            net.step_voxel(
                &mut amounts,
                &mut residual,
                T_AMBIENT,
                0.01,
                &[0.0; N_BANDS],
                &[],
            );
        }
        let after = element_totals(&chem, &amounts, &residual);
        for e in 0..N_ELEMENTS {
            if before[e] == 0.0 {
                continue;
            }
            let drift = (after[e] - before[e]).abs() / before[e];
            assert!(drift < 1e-5, "element {e} drifted by {drift:e}");
        }
    }

    #[test]
    fn energy_is_conserved_exactly() {
        let chem = generate(6, ChemParams::default());
        let net = Network::new(&chem);
        let mut amounts = seeded_amounts(&chem);
        let mut carried = vec![0.0; chem.n_compounds()];
        let absorbed = [1.0e-12f32; N_BANDS];
        for _ in 0..40 {
            let step = net.step_voxel(&mut amounts, &mut carried, T_AMBIENT, 0.01, &absorbed, &[]);
            let input: f64 = absorbed.iter().map(|&a| a as f64).sum();
            // heat + chemical energy change must equal the light let in.
            let energy_error = step.heat + step.chem_delta - input;
            assert!(
                energy_error.abs() <= input.abs() * 1e-12 + 1e-30,
                "energy residual {energy_error:e} against input {input:e}"
            );
        }
    }

    #[test]
    fn amounts_never_go_negative() {
        let chem = generate(8, ChemParams::default());
        let net = Network::new(&chem);
        let mut amounts = vec![0.0f32; chem.n_compounds()];
        amounts[chem.water as usize] = 5.0e10;
        amounts[0] = 1.0e10;
        let mut residual = vec![0.0; chem.n_compounds()];
        for _ in 0..500 {
            net.step_voxel(
                &mut amounts,
                &mut residual,
                340.0,
                0.05,
                &[0.0; N_BANDS],
                &[],
            );
            assert!(amounts.iter().all(|&n| n >= 0.0 && n.is_finite()));
        }
    }

    /// A two-compound world with a single isomerisation, built by hand so the
    /// equilibrium can be checked against theory.
    fn toy_chemistry(dh: f64, ea_f: f64) -> Chemistry {
        let mk = |id: CompoundId, name: &str, h_f: f64| Compound {
            id,
            name: name.into(),
            molecule: Molecule::new(vec![0, 0], vec![Bond::new(0, 1, 1)]),
            formula: {
                let mut f = [0u16; N_ELEMENTS];
                f[0] = 2;
                f
            },
            mass: 1.0e-26,
            h_f,
            diffusion: 1.0e-9,
            permeability: 1.0e-6,
            absorption: [0.0; N_BANDS],
            emission: [0.0; N_BANDS],
            charge: 0,
            polarity: 0.0,
            key: [0.0; KEY_DIM],
        };
        let reaction = Reaction {
            id: 0,
            reactants: vec![(0, 1)],
            products: vec![(1, 1)],
            dh,
            ea_f,
            ea_r: ea_f - dh,
            rate: 1.0e3,
            drive: Drive::Thermal,
            key: [0.0; KEY_DIM],
        };
        Chemistry {
            compounds: vec![mk(0, "A", 0.0), mk(1, "B", dh)],
            reactions: vec![reaction],
            params: ChemParams::default(),
            seed: 0,
            water: 0,
            photo: vec![],
            vent_fuel: vec![],
            involving: vec![vec![0], vec![0]],
        }
    }

    #[test]
    fn equilibrium_matches_the_boltzmann_ratio() {
        // A catalyst may change how fast equilibrium arrives, never where it
        // is. Both cases must land on exp(-dh / kT). The catalysed path also
        // bypasses the rate table, so this checks the two agree.
        let dh = -8.0e-21;
        let chem = toy_chemistry(dh, 4.0e-20);
        let net = Network::new(&chem);
        let expected = (-dh / kt(T_AMBIENT)).exp();

        for catalysis in [vec![], vec![2.0e-20f64]] {
            let mut amounts = vec![1.0e10, 1.0e10];
            let mut residual = vec![0.0; chem.n_compounds()];
            for _ in 0..200_000 {
                net.step_voxel(
                    &mut amounts,
                    &mut residual,
                    T_AMBIENT,
                    0.01,
                    &[0.0; N_BANDS],
                    &catalysis,
                );
            }
            let ratio = amounts[1] as f64 / amounts[0] as f64;
            let error = (ratio - expected).abs() / expected;
            assert!(
                error < 0.02,
                "catalysis {catalysis:?}: ratio {ratio:.4} vs expected {expected:.4}"
            );
        }
    }

    #[test]
    fn the_rate_table_tracks_the_exact_arrhenius_law() {
        let chem = generate(12, ChemParams::default());
        let table = RateTable::new(&chem);
        for &t in &[273.0f32, 293.15, 310.0, 350.0] {
            let (f_lo, f_hi, _, _, frac) = table.constants(t);
            for (r, reaction) in chem.reactions.iter().enumerate() {
                let exact = reaction.k_forward(t, 0.0);
                let looked_up = f_lo[r] + frac * (f_hi[r] - f_lo[r]);
                let scale = exact.abs().max(1e-30);
                assert!(
                    (looked_up - exact).abs() / scale < 1e-3,
                    "reaction {r} at {t} K: table {looked_up:e} vs exact {exact:e}"
                );
            }
        }
    }

    #[test]
    fn the_rate_table_clamps_outside_its_range() {
        let chem = generate(12, ChemParams::default());
        let table = RateTable::new(&chem);
        let (lo, hi, frac) = table.locate(50.0);
        assert_eq!((lo, hi), (0, 1));
        assert_eq!(frac, 0.0);
        let (lo, hi, _) = table.locate(9999.0);
        assert_eq!((lo, hi), (table.rows - 1, table.rows - 1));
    }

    #[test]
    fn light_drives_chemistry_uphill() {
        let chem = generate(3, ChemParams::default());
        let net = Network::new(&chem);
        assert!(!chem.photo.is_empty());
        let run = |absorbed: [f32; N_BANDS]| -> f64 {
            let mut amounts = seeded_amounts(&chem);
            let mut residual = vec![0.0; chem.n_compounds()];
            let mut stored = 0.0;
            for _ in 0..200 {
                stored += net
                    .step_voxel(&mut amounts, &mut residual, T_AMBIENT, 0.01, &absorbed, &[])
                    .photo_stored;
            }
            stored
        };

        let dark = run([0.0; N_BANDS]);
        let lit = run([2.0e-11; N_BANDS]);
        assert_eq!(dark, 0.0, "chemistry stored energy with no light");
        assert!(lit > 0.0, "light drove no photochemistry");
    }

    #[test]
    fn fast_chemistry_burns_out_and_leaves_a_slow_tail() {
        // An unlit soup does relax: barrierless reactions run themselves out
        // in seconds. What it does not do is finish. The reactions with real
        // barriers keep creeping for as long as you watch, and that residual
        // disequilibrium is precisely what a cell makes a living on -- an
        // enzyme is worth having only because the uncatalysed rate is not zero
        // but is far too slow to matter.
        let chem = generate(2, ChemParams::default());
        let net = Network::new(&chem);
        let mut amounts = seeded_amounts(&chem);
        let mut residual = vec![0.0; chem.n_compounds()];

        let mut activity_per_step = |amounts: &mut Vec<f32>, steps: usize| -> f64 {
            let start = amounts.clone();
            for _ in 0..steps {
                net.step_voxel(amounts, &mut residual, 275.0, 0.01, &[0.0; N_BANDS], &[]);
            }
            let moved: f64 = amounts
                .iter()
                .zip(&start)
                .map(|(a, b)| (a - b).abs() as f64)
                .sum();
            moved / steps as f64
        };

        let early = activity_per_step(&mut amounts, 4_000);
        activity_per_step(&mut amounts, 40_000); // let the easy chemistry finish
        let late = activity_per_step(&mut amounts, 32_000);
        assert!(early > 0.0, "nothing happened at all");
        assert!(
            late < early * 0.02,
            "chemistry did not settle: {late:e} after {early:e}"
        );
        assert!(
            late > 0.0,
            "world froze completely; nothing left to exploit"
        );
    }

    #[test]
    fn a_catalyst_cannot_create_energy() {
        let chem = generate(9, ChemParams::default());
        let net = Network::new(&chem);
        let catalysis: Vec<f64> = chem.reactions.iter().map(|r| r.ea_f * 0.9).collect();
        let mut amounts = seeded_amounts(&chem);
        let mut carried = vec![0.0; chem.n_compounds()];
        let mut total_heat = 0.0;
        let mut total_chem = 0.0;
        for _ in 0..2_000 {
            let s = net.step_voxel(
                &mut amounts,
                &mut carried,
                T_AMBIENT,
                0.01,
                &[0.0; N_BANDS],
                &catalysis,
            );
            total_heat += s.heat;
            total_chem += s.chem_delta;
        }
        // With no light in, heat out must exactly balance chemical energy lost.
        let residual = total_heat + total_chem;
        let scale = total_heat.abs().max(total_chem.abs()).max(1e-30);
        assert!(
            residual.abs() / scale < 1e-9,
            "catalysed run leaked {residual:e} J"
        );
    }
}
