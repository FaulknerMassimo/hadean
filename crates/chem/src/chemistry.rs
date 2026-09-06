//! Compounds, reactions, and the generated chemistry as a whole.

use crate::element::N_ELEMENTS;
use crate::molecule::Molecule;
use hadean_core::hash::{HashState, StateHasher};
use hadean_core::units::{arrhenius, ev, Joules, Kelvin};
use serde::{Deserialize, Serialize};

pub type CompoundId = u16;
pub type ReactionId = u32;

/// Spectral bands. Eight is enough for absorption spectra to differentiate
/// meaningfully, cheap to propagate, and maps directly onto rendering: a cell
/// that evolves a pigment absorbing in band 2 literally changes colour.
pub const N_BANDS: usize = 8;

/// Photon energy per band, in electronvolts. Band 0 is the reddest.
pub const BAND_ENERGY_EV: [f64; N_BANDS] = [1.55, 1.85, 2.10, 2.30, 2.55, 2.80, 3.10, 3.45];

/// Photon energy per band, in joules.
pub fn band_energy(band: usize) -> Joules {
    ev(BAND_ENERGY_EV[band])
}

/// Dimension of the key vectors used for all affinity matching in the
/// simulation -- enzyme to reaction, transporter to compound, and later
/// transcription factor to promoter.
pub const KEY_DIM: usize = 8;

/// A compound: one molecular species, with everything the rest of the
/// simulation needs to know about it precomputed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Compound {
    pub id: CompoundId,
    pub name: String,
    pub molecule: Molecule,
    pub formula: [u16; N_ELEMENTS],
    /// Molecular mass, kg.
    pub mass: f32,
    /// Formation enthalpy, J per particle. Negative for a stable compound.
    pub h_f: Joules,
    /// Free-solution diffusion coefficient at reference temperature, m^2/s.
    pub diffusion: f32,
    /// Passive membrane permeability, m/s. Used from L4 onward.
    pub permeability: f32,
    /// Absorption cross-section per band, m^2.
    pub absorption: [f32; N_BANDS],
    /// Emission strength per band, arbitrary units. Nonzero only for the few
    /// fluorescent species.
    pub emission: [f32; N_BANDS],
    pub charge: i8,
    /// Mean bond ionic character, 0..1.
    pub polarity: f32,
    /// Structural descriptor used to build affinity keys.
    pub key: [f32; KEY_DIM],
}

impl Compound {
    /// Enthalpy per atom -- a rough "how much energy is stored here" measure,
    /// used to pick which compounds the thermal vents inject.
    pub fn enthalpy_per_atom(&self) -> f64 {
        self.h_f / self.molecule.n_atoms() as f64
    }

    pub fn n_atoms(&self) -> usize {
        self.molecule.n_atoms()
    }
}

/// What drives a reaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Drive {
    /// Ordinary thermally activated chemistry, rate set by Arrhenius.
    Thermal,
    /// Photochemistry: the rate is proportional to light absorbed in `band`,
    /// and the reaction runs uphill using that energy. This is how energy
    /// enters the world at the surface.
    Photo { band: u8 },
}

/// A reversible reaction.
///
/// Forward and reverse activation energies are linked by `ea_r = ea_f - dh`,
/// which is what enforces detailed balance: the equilibrium constant is then
/// `exp(-dh / kT)` no matter what `ea_f` is, so a catalyst can speed a
/// reaction up but can never shift where it settles. Without this, an enzyme
/// would be a free-energy pump.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reaction {
    pub id: ReactionId,
    /// `(compound, stoichiometric coefficient)`, coefficients >= 1.
    pub reactants: Vec<(CompoundId, u8)>,
    pub products: Vec<(CompoundId, u8)>,
    /// Enthalpy change per unit of extent, J. Negative is exothermic.
    pub dh: Joules,
    /// Forward activation energy, J. Always >= max(0, dh).
    pub ea_f: Joules,
    /// Reverse activation energy, J. Always >= 0.
    pub ea_r: Joules,
    /// Pre-exponential factor, per second.
    pub rate: f32,
    pub drive: Drive,
    /// Affinity key. An enzyme lowers `ea_f` for reactions whose key is close
    /// to its own, which is what makes catalysis evolvable rather than a
    /// lookup table.
    pub key: [f32; KEY_DIM],
}

impl Reaction {
    /// Forward rate constant at temperature `t`, per second.
    #[inline]
    pub fn k_forward(&self, t: Kelvin, ea_reduction: f64) -> f32 {
        let ea = (self.ea_f - ea_reduction).max(0.0);
        (self.rate as f64 * arrhenius(ea, t)) as f32
    }

    /// Reverse rate constant at temperature `t`, per second.
    ///
    /// A catalyst lowers both barriers by the same amount, preserving
    /// equilibrium.
    #[inline]
    pub fn k_reverse(&self, t: Kelvin, ea_reduction: f64) -> f32 {
        let ea = (self.ea_r - ea_reduction).max(0.0);
        (self.rate as f64 * arrhenius(ea, t)) as f32
    }

    /// Total stoichiometric order of the forward direction.
    pub fn order(&self) -> u32 {
        self.reactants.iter().map(|(_, n)| *n as u32).sum()
    }

    pub fn is_exothermic(&self) -> bool {
        self.dh < 0.0
    }

    /// Net change in the amount of `c` per unit extent.
    pub fn net(&self, c: CompoundId) -> i32 {
        let out: i32 = self
            .products
            .iter()
            .filter(|(i, _)| *i == c)
            .map(|(_, n)| *n as i32)
            .sum();
        let inp: i32 = self
            .reactants
            .iter()
            .filter(|(i, _)| *i == c)
            .map(|(_, n)| *n as i32)
            .sum();
        out - inp
    }
}

/// Tunables for chemistry generation. All of it comes from config, and the
/// whole chemistry is a pure function of these plus the seed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChemParams {
    /// Target number of compounds. The plan's advice is to start at 32: enough
    /// for interesting metabolism, small enough to fit fields in VRAM later.
    pub n_compounds: usize,
    /// Upper bound on the number of reactions. Not a target: the network is
    /// as large as the compound pool supports, which at 32 compounds is
    /// typically a few hundred. Lower it to shrink the per-voxel chemistry
    /// cost during long evolutionary runs.
    pub n_reactions: usize,
    /// Maximum non-hydrogen atoms in a generated molecule.
    pub max_skeleton: usize,
    /// Activation energy range, in zeptojoules (1e-21 J). For scale, kT at
    /// 293 K is 4.05 zJ, so a 100 zJ barrier means a reaction that never runs
    /// without a catalyst.
    pub ea_min_zj: f32,
    pub ea_max_zj: f32,
    /// Pre-exponential factor range, per second (log-uniform).
    pub rate_min: f32,
    pub rate_max: f32,
    /// Fraction of generated reactions attempted as additions rather than
    /// exchanges.
    pub addition_fraction: f32,
    /// Multiplier on all absorption cross-sections. The pond is only 1.5 mm
    /// deep, so real cross-sections would give no vertical light gradient at
    /// all; this makes the medium optically dense enough for depth to matter.
    pub absorption_scale: f32,
    /// Quantum yield of photochemical reactions, 0..1.
    pub photo_yield: f32,
    /// How many photochemical reactions to designate.
    pub n_photo: usize,
}

impl Default for ChemParams {
    fn default() -> Self {
        Self {
            n_compounds: 32,
            n_reactions: 240,
            max_skeleton: 4,
            ea_min_zj: 6.0,
            ea_max_zj: 160.0,
            rate_min: 1.0e-1,
            rate_max: 1.0e4,
            addition_fraction: 0.3,
            absorption_scale: 4.0e3,
            photo_yield: 0.35,
            n_photo: 2,
        }
    }
}

/// A generated chemistry: the compound table, the reaction network, and the
/// indices the rest of the simulation looks things up by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chemistry {
    pub compounds: Vec<Compound>,
    pub reactions: Vec<Reaction>,
    pub params: ChemParams,
    pub seed: u64,
    /// The solvent. Always present; used as the default exchange partner.
    pub water: CompoundId,
    /// Indices into `reactions` of the light-driven ones.
    pub photo: Vec<ReactionId>,
    /// High-enthalpy reduced compounds, injected by thermal vents. Sorted by
    /// descending enthalpy per atom, so `vent_fuel[0]` is the most energetic.
    pub vent_fuel: Vec<CompoundId>,
    /// Reactions each compound takes part in, for fast per-voxel iteration.
    pub involving: Vec<Vec<ReactionId>>,
}

impl Chemistry {
    pub fn n_compounds(&self) -> usize {
        self.compounds.len()
    }

    pub fn n_reactions(&self) -> usize {
        self.reactions.len()
    }

    pub fn compound(&self, id: CompoundId) -> &Compound {
        &self.compounds[id as usize]
    }

    pub fn reaction(&self, id: ReactionId) -> &Reaction {
        &self.reactions[id as usize]
    }

    /// Look a compound up by name, e.g. `"H2O"`.
    pub fn by_name(&self, name: &str) -> Option<CompoundId> {
        self.compounds.iter().find(|c| c.name == name).map(|c| c.id)
    }

    /// Total element counts on each side of a reaction, for verification.
    pub fn reaction_formula(&self, r: &Reaction, products: bool) -> [u32; N_ELEMENTS] {
        let side = if products { &r.products } else { &r.reactants };
        let mut f = [0u32; N_ELEMENTS];
        for &(c, n) in side {
            let cf = self.compound(c).formula;
            for e in 0..N_ELEMENTS {
                f[e] += cf[e] as u32 * n as u32;
            }
        }
        f
    }

    /// Every reaction conserves every element.
    ///
    /// This can only fail if the rewrite engine is broken, which is the point:
    /// it is a structural invariant, not a property of the generator's luck.
    pub fn verify_mass_balance(&self) -> Result<(), String> {
        for r in &self.reactions {
            let lhs = self.reaction_formula(r, false);
            let rhs = self.reaction_formula(r, true);
            if lhs != rhs {
                return Err(format!(
                    "reaction {} is unbalanced: {:?} -> {:?}",
                    r.id, lhs, rhs
                ));
            }
        }
        Ok(())
    }

    /// No cycle of reactions can produce free energy.
    ///
    /// Rather than searching the reaction graph for positive cycles, this
    /// checks the stronger and cheaper property that makes such cycles
    /// impossible: every reaction's `dh` is exactly the difference of a
    /// *potential function* over compounds (their formation enthalpies). Any
    /// cycle returns to the same compound multiset, so the potentials
    /// telescope to zero. If this holds, no positive-energy cycle can exist.
    pub fn verify_no_free_energy(&self) -> Result<(), String> {
        for r in &self.reactions {
            let lhs: f64 = r
                .reactants
                .iter()
                .map(|&(c, n)| self.compound(c).h_f * n as f64)
                .sum();
            let rhs: f64 = r
                .products
                .iter()
                .map(|&(c, n)| self.compound(c).h_f * n as f64)
                .sum();
            let expected = rhs - lhs;
            let tol = 1e-24 + expected.abs() * 1e-9;
            if (r.dh - expected).abs() > tol {
                return Err(format!(
                    "reaction {} has dh {:e} but its compounds imply {:e}",
                    r.id, r.dh, expected
                ));
            }
            if r.ea_f < 0.0 || r.ea_r < 0.0 {
                return Err(format!("reaction {} has a negative barrier", r.id));
            }
            // Detailed balance: the two barriers must differ by exactly dh.
            let slack = (r.ea_f - r.ea_r - r.dh).abs();
            if slack > 1e-24 + r.ea_f.abs() * 1e-9 {
                return Err(format!(
                    "reaction {} violates detailed balance by {:e} J",
                    r.id, slack
                ));
            }
        }
        Ok(())
    }

    /// Run every structural check. Called at startup; a failure is fatal.
    pub fn verify(&self) -> Result<(), String> {
        self.verify_mass_balance()?;
        self.verify_no_free_energy()?;
        if self.compounds.len() > CompoundId::MAX as usize {
            return Err("too many compounds for CompoundId".into());
        }
        for (i, c) in self.compounds.iter().enumerate() {
            if c.id as usize != i {
                return Err(format!("compound {i} has mismatched id {}", c.id));
            }
            if !c.molecule.is_valid() {
                return Err(format!("compound {} has an invalid molecule", c.name));
            }
            if c.diffusion <= 0.0 || !c.diffusion.is_finite() {
                return Err(format!(
                    "compound {} has a bad diffusion coefficient",
                    c.name
                ));
            }
        }
        Ok(())
    }

    /// Largest diffusion coefficient, which sets the diffusion substep count.
    pub fn max_diffusion(&self) -> f32 {
        self.compounds
            .iter()
            .map(|c| c.diffusion)
            .fold(0.0, f32::max)
    }

    /// How many compounds take part in at least one reaction.
    /// Which compounds this network can actually make, starting from
    /// `seeds`.
    ///
    /// Reactions are reversible, so a reaction hands over its products once
    /// every reactant is present, and its reactants once every product is.
    /// Iterated to a fixed point.
    ///
    /// This exists because a one-hop answer is wrong in a way that is easy to
    /// miss. Asking "is this compound the product of a photochemical
    /// reaction?" says yes for a compound whose photoreaction has no
    /// substrates in this world -- and a protocell handed that compound as
    /// food sits in the pond with a metabolism it can never run, alive and
    /// doing nothing until it starves. The pond reports a food supply of
    /// exactly zero, forever, and nothing else looks wrong.
    ///
    /// Reachability is necessary, not sufficient: a compound at the far end
    /// of a strongly uphill chain is reachable and still vanishingly rare. It
    /// rules out the impossible, which is what it is for.
    pub fn reachable_from(&self, seeds: &[CompoundId]) -> Vec<bool> {
        let mut present = vec![false; self.n_compounds()];
        for &seed in seeds {
            present[seed as usize] = true;
        }
        loop {
            let mut grew = false;
            let mut spread = |from: &[(CompoundId, u8)], to: &[(CompoundId, u8)]| {
                if from.iter().all(|&(c, _)| present[c as usize]) {
                    for &(c, _) in to {
                        if !present[c as usize] {
                            present[c as usize] = true;
                            grew = true;
                        }
                    }
                }
            };
            for r in &self.reactions {
                spread(&r.reactants, &r.products);
                spread(&r.products, &r.reactants);
            }
            if !grew {
                return present;
            }
        }
    }

    pub fn reactive_compounds(&self) -> usize {
        self.involving.iter().filter(|v| !v.is_empty()).count()
    }
}

impl HashState for Chemistry {
    fn hash_state(&self, h: &mut StateHasher) {
        h.u64(self.seed);
        h.usize(self.compounds.len());
        for c in &self.compounds {
            h.str(&c.name);
            h.f64(c.h_f);
            h.f32(c.mass);
            h.f32(c.diffusion);
            h.f32(c.permeability);
            h.f32_slice(&c.absorption);
        }
        h.usize(self.reactions.len());
        for r in &self.reactions {
            for &(c, n) in r.reactants.iter().chain(r.products.iter()) {
                h.u32(c as u32);
                h.byte(n);
            }
            h.f64(r.dh);
            h.f64(r.ea_f);
            h.f32(r.rate);
        }
    }
}
