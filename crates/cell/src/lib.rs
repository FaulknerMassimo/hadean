//! **L4 cells, with or without an L3 genome.**
//!
//! Two cells live here and they share every mechanism below the top.
//!
//! The **pre-genome protocell** is the Phase 2 ancestor: every cell has the
//! same membrane and catalyses the same generated reaction, and what it
//! inherits is one scalar. It is kept because it is the control. Its result --
//! a complete material lifecycle, where compounds cross a membrane, an
//! exergonic reaction charges a reserve, maintenance drains it, cells divide,
//! and dead cells return every particle and joule to the pond -- is what every
//! claim about the genome is measured against.
//!
//! The **genome cell** (`cells.genome = true`) replaces that scalar with a
//! byte string. It does not add a lifecycle; it changes where the numbers in
//! one come from. What a cell takes up, what it catalyses and what it costs
//! are read out of [`genome`], and every one of them can mutate.
//!
//! The two paths meet at [`Expression`], which is the only thing the lifecycle
//! functions below read. A pre-genome cell fills it from its traits and a
//! genome cell from its proteome, and neither `exchange`, `metabolize` nor
//! `maintain` knows which it was handed. That is deliberate: it keeps the
//! pre-genome path arithmetically identical to what it was, so the control is
//! a real control and not a reimplementation of one.

use std::f32::consts::PI;
use std::sync::Arc;

use hadean_chem::{Chemistry, Drive, ReactionId};
use hadean_core::hash::{HashState, StateHasher};
use hadean_core::rng::Purpose;
use hadean_core::units::{Joules, KB, VISCOSITY};
use hadean_core::{Axis, Counter, Grid, Schedule};
use hadean_fields::heat::HeatField;
use hadean_fields::scalar::ChemField;
use hadean_fields::transport::FaceVelocity;
use serde::{Deserialize, Serialize};

pub mod genome;

pub use genome::{Genome, MutationRates, ProteinClass};

/// Tunables for the hand-authored ancestor used before genomes exist.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CellConfig {
    pub enabled: bool,
    pub initial_count: usize,
    pub population_cap: usize,
    /// Radius just after division, metres.
    pub birth_radius: f32,
    /// Multiplier on physical passive membrane permeability.
    pub membrane_scale: f32,
    /// Fraction of a limiting substrate catalysed per second.
    pub metabolic_rate: f32,
    /// Target fraction of released chemical energy retained by the cell.
    pub capture_efficiency: f32,
    /// Continuous energetic cost per cell, W.
    pub maintenance_power: f64,
    /// Reserve at which a cell divides, J.
    pub division_reserve: f64,
    /// Seconds of unpaid maintenance that accumulate into lethal damage.
    ///
    /// This is the population's tolerance for famine, and it has to be read
    /// against the length of the famines the world actually produces. The
    /// pond's night is three hundred seconds long and its photochemistry
    /// stops dead in the dark; a cohort that cannot outlast the dark does not
    /// have a bad night, it goes extinct by morning.
    pub starvation_time: f32,
    /// Mean lifespan in simulated seconds. Ageing makes death inevitable.
    pub maximum_age: f32,
    /// Fractional spread of individual lifespans about `maximum_age`.
    ///
    /// Zero gives every cell the same lifespan to the tick, which sounds like
    /// a simplification and is really a mechanism. Cells here are clones in a
    /// shared voxel, so a cohort that is born together arrives at `maximum_age`
    /// together and dies as one -- and a population that has settled at its
    /// break-even concentration has no surplus to divide on, so nothing is
    /// being born to replace them. That was the whole of the Phase 2 crash:
    /// not a famine, an actuarial table with one row in it.
    ///
    /// The draw is a pure function of the cell's id, so a cell's allotted span
    /// is fixed at birth, survives a snapshot, and costs no state.
    pub lifespan_spread: f32,
    /// Fractional spread of the heritable traits -- see [`Traits`].
    ///
    /// One number doing two jobs, because they are the same job: it is the
    /// standing variation the founding cohort arrives with, and the size of
    /// the mutational kick a daughter gets at every division. Zero makes every
    /// cell an exact clone of the ancestor for ever, which is what the cell
    /// layer did before traits existed and is the control this is measured
    /// against.
    ///
    /// **How big it has to be is not a matter of taste.** A population eats
    /// down until its marginal cell breaks even, so with
    /// `income(u) = k u C e` and `bill(u) = m (1 - c + c u)`, that cell fixes
    /// the water at `k C e = m (1 - c + c u_m) / u_m`. A cell a fraction `d`
    /// better off than it then earns
    ///
    /// ```text
    ///     surplus = m (1 - c + c u_m)(1 + d) - m (1 - c) - m c u_m (1 + d)
    ///             = m (1 - c) d
    /// ```
    ///
    /// Everything but `d` and the two dials cancels. So the best cell in the
    /// pond funds a division in
    /// `division_reserve / (m (1 - c) d)` seconds, and it has to do that
    /// inside a lifetime:
    ///
    /// ```text
    ///     division_reserve  <  maintenance_power (1 - trait_cost) d maximum_age
    /// ```
    ///
    /// Fail that and the plateau cannot turn over however varied the
    /// population is, because no cell in it can afford a daughter before it
    /// dies. Measured at 0.2, the living spread reached `d = 0.32` and the
    /// right-hand side came to 3.8e-9 J against a `division_reserve` of 1e-8:
    /// short by a factor of two and a half, and the run duly recorded zero
    /// births after the peak, exactly as the clone control did.
    ///
    /// The inequality is generous, so clearing it is necessary and not
    /// sufficient. `income = k u C e` holds only while the membrane is the
    /// bottleneck, and a cell that can take up faster than it can metabolise
    /// gets less and less out of each extra transporter: at `u = 1.5` the
    /// measured income is 1.38 times the marginal cell's, not 1.5. So a real
    /// population needs more spread than the arithmetic asks for.
    pub trait_spread: f32,
    /// Fraction of a cell's upkeep that scales with the machinery it carries.
    ///
    /// Transporters are not free: a cell with twice the membrane machinery
    /// pays to keep twice the machinery. The rest of the bill is the fixed
    /// cost of being a cell at all, and that split is what decides whether
    /// variation in [`Traits::uptake`] means anything.
    ///
    /// A cell breaks even at food concentration
    ///
    /// ```text
    ///     C*(u) = m (1 - c + c u) / (k u e)
    /// ```
    ///
    /// so at `c = 1` -- upkeep entirely proportional to machinery -- the `u`
    /// cancels and every cell in the population breaks even at exactly the
    /// same place no matter what it carries. That is the freeze this whole
    /// mechanism exists to break, reintroduced by the back door. At `c = 0`
    /// uptake is free, nothing bounds it, and a lineage evolves towards
    /// stripping the pond to nothing -- the failure `membrane_scale = 6000`
    /// produced by hand. In between, `C*` falls with `u` towards a floor of
    /// `c m / (k e)`: evolution makes the population hungrier, and `c` sets
    /// how much food is still in the water when it has finished.
    ///
    /// Neutral at `uptake = 1`, so it does not change what
    /// `maintenance_power` means for a population of clones.
    pub trait_cost: f64,
    /// Fraction of working maintenance power a dormant cell pays.
    ///
    /// This is the dial the night bill actually responds to. A population at
    /// its carrying capacity eats what the world produces, so across a night
    /// -- when photochemistry produces nothing -- it has to live on `P x T`
    /// particles of standing stock, and every per-cell term cancels out of
    /// that. Lowering `maintenance_power` does not help, because it raises the
    /// standing population by exactly the factor it lowers each cell's bill.
    /// Dormancy is outside that cancellation: it lowers the bill of the cells
    /// that are shut down without raising how many of them the day supports.
    ///
    /// 1.0 disables dormancy, in the sense that a shut-down cell pays the same
    /// as a working one and immediately wakes again.
    pub dormancy_power_fraction: f64,
    /// Seconds of working maintenance a dormant cell must bank before it wakes.
    ///
    /// Shutting down happens the moment a cell cannot pay its full upkeep;
    /// waking is deliberately a much higher bar, so a cell sitting on the
    /// margin does not flicker between the two states every tick and average
    /// back into having no dormancy at all.
    pub dormancy_exit: f32,
    /// Fraction of a corpse returned to the field per second.
    pub decomposition_rate: f32,
    /// Simulated seconds of lifeless chemistry before the ancestors appear.
    ///
    /// A generated chemistry starts with only its primordials, so at tick zero
    /// nothing photochemical exists yet -- including, in most seeds, the
    /// compound the ancestor eats. Cells introduced then starve in an empty
    /// larder before the sun has made the first meal. Letting the pond run
    /// first is both the fix and the more honest picture: the world predates
    /// life in it, and what life finds is a chemistry already at its
    /// photostationary state. Zero introduces them at tick zero.
    pub seed_delay: f32,

    // ---- L3 ----------------------------------------------------------------
    /// Give cells a genome instead of the hardcoded traits.
    ///
    /// Off by default, and that is not timidity. Everything this project
    /// believes about its own population came out of A/B runs against a
    /// control, and a genome that quietly replaced the protocell everywhere
    /// would destroy the control in the same commit that needed it. With this
    /// off the cell layer is arithmetically what it was; with it on, `traits`
    /// is dead weight and [`Genome`] decides.
    pub genome: bool,
    /// Mutation operator rates, applied at every division. See
    /// [`MutationRates`].
    pub mutation: MutationRates,
    /// Width of an enzyme's recognition of a reaction, in key-space distance.
    ///
    /// The most consequential number in the genome layer, because it sets what
    /// one protein can do at once. Too small and an enzyme catalyses its own
    /// reaction and nothing else; too large and it catalyses half the network,
    /// which is not a metabolism, it is a fire.
    ///
    /// Measured, not chosen. `hadean chem --keys` reports the spread of the
    /// chemistry's keys in the normalised space the genome matches in, and on
    /// `gate.toml` the thermal reactions have a median nearest-neighbour
    /// distance of 0.102. 0.08 puts about 4.6 reactions inside one enzyme's
    /// reach -- its own plus a few weak neighbours, which is a specialist with
    /// the promiscuity `PLAN.md` wants duplication to act on. 0.2 would reach
    /// 39 of 78 and 0.35 would reach 72.
    ///
    /// Reach is not the same as what a lineage can *become*, and confusing the
    /// two is the way to get this wrong in the generous direction. A width
    /// this narrow still leaves the whole network reachable, because mutation
    /// moves the key itself: what the width decides is how much a protein does
    /// at once, not where its descendants can go.
    pub enzyme_sigma: f32,
    /// Width of a transporter's recognition of a compound.
    ///
    /// Wider than `enzyme_sigma` because compounds are further apart than
    /// reactions are -- a reaction key is the mean of its participants', and
    /// averaging pulls them together. Median nearest neighbour is 0.229
    /// against the reactions' 0.102, and 0.15 puts about three compounds
    /// inside a transporter's reach.
    pub transport_sigma: f32,
    /// Multiplier on `starvation_time` bought by a full structural complement.
    ///
    /// What structural protein is *for*, and the reason it is a class rather
    /// than a constant: it buys famine tolerance and charges upkeep for it, so
    /// how much to carry is a trade a lineage makes against how hungry its
    /// pond is, not a number in this file.
    pub structural_benefit: f32,
}

/// Total standing protein a cell carrying the hand-written ancestor's genome
/// expresses, summed over its genes.
///
/// A cell's upkeep is charged against this, so the ancestor pays very close to
/// `maintenance_power` and means the same thing it meant before genomes
/// existed. A lineage that doubles its proteome doubles the machinery half of
/// its bill, which is what stops a genome from being a free capability store.
pub const PROTEOME_REFERENCE: f32 = 4.0;

impl Default for CellConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_count: 24,
            population_cap: 60_000,
            birth_radius: 4.0e-6,
            membrane_scale: 2000.0,
            metabolic_rate: 20.0,
            capture_efficiency: 0.60,
            maintenance_power: 4.0e-12,
            division_reserve: 2.0e-10,
            starvation_time: 120.0,
            maximum_age: 600.0,
            lifespan_spread: 0.4,
            trait_spread: 0.2,
            trait_cost: 0.5,
            dormancy_power_fraction: 0.05,
            dormancy_exit: 60.0,
            decomposition_rate: 0.8,
            seed_delay: 150.0,
            genome: false,
            mutation: MutationRates::default(),
            enzyme_sigma: 0.08,
            transport_sigma: 0.15,
            structural_benefit: 2.0,
        }
    }
}

impl CellConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.initial_count > self.population_cap {
            return Err("initial cell count exceeds the population cap".into());
        }
        if self.birth_radius <= 0.0 || !self.birth_radius.is_finite() {
            return Err("cell birth radius must be positive".into());
        }
        if self.membrane_scale < 0.0 || self.metabolic_rate < 0.0 {
            return Err("cell transport and metabolic rates cannot be negative".into());
        }
        if !(0.0..=1.0).contains(&self.capture_efficiency) {
            return Err("cell capture efficiency must be in 0..1".into());
        }
        if !(0.0..=1.0).contains(&self.dormancy_power_fraction) {
            return Err("cell dormancy power fraction must be in 0..1".into());
        }
        if self.dormancy_exit < 0.0 || !self.dormancy_exit.is_finite() {
            return Err("cell dormancy exit must be a finite number of seconds".into());
        }
        if !(0.0..1.0).contains(&self.lifespan_spread) {
            return Err("cell lifespan spread must be in 0..1".into());
        }
        if self.trait_spread < 0.0 || !self.trait_spread.is_finite() {
            return Err("cell trait spread must be a finite non-negative fraction".into());
        }
        if !(0.0..=1.0).contains(&self.trait_cost) {
            return Err("cell trait cost must be in 0..1".into());
        }
        if self.seed_delay < 0.0 || !self.seed_delay.is_finite() {
            return Err("cell seed delay must be a finite number of seconds".into());
        }
        if self.maintenance_power < 0.0
            || self.division_reserve <= 0.0
            || self.starvation_time <= 0.0
            || self.maximum_age <= 0.0
            || self.decomposition_rate <= 0.0
        {
            return Err("cell lifecycle rates and thresholds must be positive".into());
        }
        self.mutation.validate()?;
        if self.enzyme_sigma <= 0.0 || self.transport_sigma <= 0.0 {
            return Err("genome recognition widths must be positive".into());
        }
        if self.structural_benefit < 1.0 || !self.structural_benefit.is_finite() {
            return Err("structural benefit is a multiplier and cannot be below one".into());
        }
        Ok(())
    }

    /// The largest `division_reserve` a settled population could still divide
    /// on, J.
    ///
    /// A population eats down until its marginal cell breaks even, and a cell
    /// a fraction `d` better off than that one is then left with
    /// `maintenance_power (1 - trait_cost) d` to bank -- see
    /// [`CellConfig::trait_spread`] for where that comes from. Give it a
    /// lifetime to do so and that is a budget: a `division_reserve` above it
    /// cannot be met by anything in the pond, so the plateau has no births in
    /// it, and without births there is no recovery leg to the Phase 2 gate
    /// however healthy the curve looks on the way up.
    ///
    /// `trait_spread` stands in for `d` here, which is right for a pre-genome
    /// population -- it is exactly the kick every daughter gets -- and is only
    /// a guess for a genome one, whose variation is whatever its mutation
    /// operators happen to produce. Use
    /// [`TraitSummary::relative_spread`] against a running population for the
    /// measured version. So this is one standard deviation's worth of
    /// advantage. The best cell of a few hundred is two
    /// or three of those out, and against that the real ceiling is a small
    /// multiple of this -- while the sublinearity noted on `trait_spread`
    /// pushes the other way. It is an order-of-magnitude check, and it is
    /// worth making before spending forty minutes on a run: a configuration
    /// an order below its budget will not turn over, and none of the other
    /// dials can rescue it.
    pub fn turnover_budget(&self) -> Joules {
        self.maintenance_power
            * (1.0 - self.trait_cost)
            * self.trait_spread as f64
            * self.maximum_age as f64
    }
}

/// Discriminants are part of the replay contract: they are hashed and written
/// into snapshots as `state as u8`. Append new states, never reorder these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellState {
    Alive,
    Decomposing,
    /// Alive, but shut down to a fraction of working power to sit out a
    /// famine. Its membrane and its catalysed reaction keep running -- those
    /// are chemistry, not decisions the cell gets to make -- so a dormant cell
    /// still takes up whatever the water offers, and banks it instead of
    /// burning it. What it stops doing is paying full upkeep and growing.
    Dormant,
}

/// The heritable part of a cell: one number, and the place L3 plugs in.
///
/// This is a vestigial genome and it is here because of a measurement rather
/// than a plan. Before it, every cell in the pond was an exact clone in a
/// shared voxel, so the whole population broke even at the same food
/// concentration and arrived there together. What looked like a carrying
/// capacity was a hundred and thirty-six identical cells freezing at once: ten
/// runs across every dial the cell layer had, and in every one of them
/// `births` came out exactly equal to the peak population. Nothing was ever
/// born after the boom, and a population that does not breed cannot recover,
/// which is the leg of the Phase 2 gate that was missing.
///
/// Variation is what unfreezes it. When cells differ in what they need, a
/// shortage stops being simultaneous: the pond settles at a concentration
/// where the poorest cells are already below their line and the best still
/// have a surplus to divide on, so the plateau turns over -- births and deaths
/// both non-zero -- instead of standing still.
///
/// A fixed struct of scalars cannot grow complexity, which is exactly why
/// `PLAN.md` specifies a variable-length byte string for the real genome. This
/// is not that and does not pretend to be. It is the smallest thing that
/// carries inheritance, mutation and selection through the tick, the audit,
/// the digest and the snapshot, so that when the genome arrives it replaces a
/// mechanism that already works rather than introducing one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Traits {
    /// Multiplier on this cell's membrane permeability.
    ///
    /// How much transporter the cell carries. It sets both what the cell can
    /// take up and, through [`CellConfig::trait_cost`], what it costs to keep,
    /// so it is a trade-off and not a free dial: a hungrier cell is a more
    /// expensive cell, and which of those wins depends on how much food is in
    /// the water.
    pub uptake: f32,
}

impl Default for Traits {
    /// The hand-written ancestor: exactly the config, unmodified.
    fn default() -> Self {
        Self { uptake: 1.0 }
    }
}

/// How far a lineage's traits may drift from the ancestor's, either way.
///
/// Not a tuning dial. A membrane is a physical object and there is a limit to
/// how much of one a cell this size can carry; without a bound, a long run
/// walks a lineage off to values where the trade-off this is built on stops
/// meaning anything.
const TRAIT_LIMIT: f32 = 8.0;

/// Apply one multiplicative mutational kick.
///
/// Log-normal, so the kick is symmetric in the thing that matters -- halving
/// and doubling are the same size of step -- and a trait can never be pushed
/// through zero into a negative membrane.
fn mutate(value: f32, spread: f32, kick: f32) -> f32 {
    if spread <= 0.0 {
        return value;
    }
    (value * (spread * kick).exp()).clamp(1.0 / TRAIT_LIMIT, TRAIT_LIMIT)
}

impl Traits {
    /// The ancestor's traits, as the founding cohort draws them.
    ///
    /// A pure function of the cell's id, like [`lifespan`], so a founder's
    /// starting point does not depend on when or in what order it was made.
    fn founder(cfg: &CellConfig, rng: &Counter, id: u64) -> Self {
        Self {
            uptake: mutate(
                1.0,
                cfg.trait_spread,
                rng.normal(0, id, Purpose::Mutation, 0),
            ),
        }
    }

    /// What a daughter inherits from this cell.
    fn inherit(self, cfg: &CellConfig, rng: &Counter, tick: u64, parent: u64) -> Self {
        Self {
            uptake: mutate(
                self.uptake,
                cfg.trait_spread,
                rng.normal(tick, parent, Purpose::Mutation, 0),
            ),
        }
    }
}

impl HashState for Traits {
    fn hash_state(&self, h: &mut StateHasher) {
        h.f32(self.uptake);
    }
}

/// A single protocell. Contents use `f64` because crossing a membrane between
/// the pond's large `f32` amounts and a tiny compartment must not lose mass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub id: u64,
    pub parent: Option<u64>,
    pub pos: [f32; 3],
    pub radius: f32,
    pub contents: Vec<f64>,
    /// Captured chemical free energy, J. It remains part of the global audit.
    pub reserve: Joules,
    pub damage: f32,
    pub age: f32,
    pub state: CellState,
    pub generation: u32,
    /// What this cell inherited before genomes. See [`Traits`]. Ignored when
    /// the cell carries a `genome`.
    pub traits: Traits,
    /// L3. `None` when the world runs the pre-genome protocell.
    ///
    /// Shared: a daughter that inherits her mother's bytes unchanged -- which
    /// at the default rates is most of them -- shares the allocation and the
    /// decode with her, so a colony of clones costs one genome between them.
    pub genome: Option<Arc<Genome>>,
    /// Concentration of each gene's protein, 0..1, indexed by gene.
    ///
    /// This is where the same genome becomes two different cells, and it is
    /// the reason the proteome is per-cell state while the genome is shared.
    /// Nothing reads it yet that varies between siblings; what it is for is
    /// L5, where position and signal drive expression and one genome has to
    /// produce a body with different tissues in it.
    pub proteome: Vec<f32>,
}

impl Cell {
    /// Living, whether or not it is currently working. A dormant cell is a
    /// cell: it occupies the population, it can wake, and it is not a corpse.
    pub fn is_alive(&self) -> bool {
        matches!(self.state, CellState::Alive | CellState::Dormant)
    }

    pub fn is_dormant(&self) -> bool {
        self.state == CellState::Dormant
    }

    pub fn voxel(&self, grid: &Grid) -> usize {
        grid.voxel_at(self.pos)
    }

    /// The membrane permeability multiplier this cell actually runs.
    pub fn membrane_scale(&self, cfg: &CellConfig) -> f64 {
        cfg.membrane_scale as f64 * self.uptake() as f64
    }

    /// How much transport machinery this cell carries, ancestor-relative.
    ///
    /// One number out of what is, for a genome cell, a per-compound profile,
    /// and it exists so that the trait charts and the population summary keep
    /// reading the same column across both cell types. A genome cell's real
    /// uptake is not a scalar -- see [`Expression::uptake`].
    pub fn uptake(&self) -> f32 {
        match &self.genome {
            None => self.traits.uptake,
            Some(g) => {
                let mut total = 0.0;
                for (gene, &conc) in g.genes.iter().zip(&self.proteome) {
                    if gene.class == ProteinClass::Transporter {
                        total += conc * gene.strength();
                    }
                }
                total
            }
        }
    }

    /// Standing machinery, ancestor-relative. The thing upkeep is charged on.
    pub fn machinery(&self) -> f64 {
        match &self.genome {
            None => self.traits.uptake as f64,
            // Every protein, not just the useful ones. A cell expressing a
            // receptor that nothing reads is still paying to keep it, which is
            // what makes an inert gene a cost a lineage can shed.
            Some(_) => {
                (self.proteome.iter().sum::<f32>() / PROTEOME_REFERENCE) as f64
            }
        }
    }

    /// What staying alive costs this cell, W.
    ///
    /// Part fixed cost of being a cell, part the machinery it carries; see
    /// [`CellConfig::trait_cost`] for why the split is the whole point.
    pub fn maintenance(&self, cfg: &CellConfig) -> f64 {
        cfg.maintenance_power * (1.0 - cfg.trait_cost + cfg.trait_cost * self.machinery())
    }

    /// How long this cell can go unpaid before the damage is lethal, seconds.
    ///
    /// Structural protein is what buys the difference.
    pub fn famine_tolerance(&self, cfg: &CellConfig) -> f32 {
        let Some(g) = &self.genome else {
            return cfg.starvation_time;
        };
        let mut structure = 0.0;
        for (gene, &conc) in g.genes.iter().zip(&self.proteome) {
            if gene.class == ProteinClass::Structural {
                structure += conc * gene.strength();
            }
        }
        cfg.starvation_time * (1.0 + (cfg.structural_benefit - 1.0) * structure.min(1.0))
    }

    /// Genes, for the run log. Zero for a pre-genome cell.
    pub fn gene_count(&self) -> usize {
        self.genome.as_ref().map_or(0, |g| g.genes.len())
    }

    pub fn genome_len(&self) -> usize {
        self.genome.as_ref().map_or(0, |g| g.len())
    }

    pub fn chemical_energy(&self, chem: &Chemistry) -> Joules {
        self.contents
            .iter()
            .zip(&chem.compounds)
            .map(|(&n, c)| n * c.h_f)
            .sum()
    }
}

impl HashState for Cell {
    fn hash_state(&self, h: &mut StateHasher) {
        h.u64(self.id);
        h.u64(self.parent.unwrap_or(u64::MAX));
        for &x in &self.pos {
            h.f32(x);
        }
        h.f32(self.radius);
        h.usize(self.contents.len());
        for &x in &self.contents {
            h.f64(x);
        }
        h.f64(self.reserve);
        h.f32(self.damage);
        h.f32(self.age);
        h.byte(self.state as u8);
        h.u32(self.generation);
        self.traits.hash_state(h);
        // A genome is heritable state and belongs in the digest, but a
        // pre-genome cell must hash exactly as it did before this existed --
        // otherwise every recorded digest in the project changes for a
        // mechanism that is switched off.
        if let Some(g) = &self.genome {
            g.hash_state(h);
            h.usize(self.proteome.len());
            for &c in &self.proteome {
                h.f32(c);
            }
        }
    }
}

/// The living and decomposing agents in a world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Population {
    pub cells: Vec<Cell>,
    pub next_id: u64,
    /// Generated reaction catalysed by the hardcoded protocell.
    pub metabolic_reaction: Option<ReactionId>,
    pub births: u64,
    pub deaths: u64,
}

impl Population {
    /// An empty population, with no metabolism chosen yet.
    pub fn empty() -> Self {
        Self {
            cells: Vec::new(),
            next_id: 0,
            metabolic_reaction: None,
            births: 0,
            deaths: 0,
        }
    }

    /// Put the founding cohort into the world, and settle what it eats.
    ///
    /// `abundance` is the particle count of every compound in the pond right
    /// now; see [`choose_metabolism`] for why the choice is made here, from
    /// measurements, rather than at construction from the reaction graph.
    /// Once chosen it never changes -- until L3, when a genome will decide.
    ///
    /// Ancestors arrive with nothing: no contents and no reserve. Nothing is
    /// created, so this is invisible to the energy and mass audit and can
    /// happen at any tick.
    pub fn introduce(
        &mut self,
        cfg: &CellConfig,
        chem: &Chemistry,
        grid: &Grid,
        rng: &Counter,
        abundance: &[f64],
    ) {
        if !cfg.enabled || !self.cells.is_empty() {
            return;
        }
        if self.metabolic_reaction.is_none() {
            self.metabolic_reaction = choose_metabolism(chem, abundance, grid.len());
        }
        if self.metabolic_reaction.is_none() {
            return;
        }
        // The hand-written ancestor, matched to this world's chemistry. Built
        // once: the founding cohort are siblings, and they differ from each
        // other by the same mutation operators that will act on every division
        // afterwards, rather than by a separate founder-only mechanism.
        let founder = cfg.genome.then(|| {
            let reaction = self.metabolic_reaction.expect("checked above");
            let mut a = genome::ancestor(chem, reaction, rng, 0);
            a.bind(chem, cfg.enzyme_sigma, cfg.transport_sigma);
            Arc::new(a)
        });

        let (ex, ey, ez) = grid.extent();
        for id in 0..cfg.initial_count as u64 {
            // Start throughout the water column. The ancestor's preferred
            // pathway consumes a photochemical product, so shallow cells
            // are initially favoured without making depth destiny.
            let x = rng.range(0, id, Purpose::Placement, 20, 0.05 * ex, 0.95 * ex);
            let y = rng.range(0, id, Purpose::Placement, 21, 0.05 * ey, 0.95 * ey);
            let z = rng.range(0, id, Purpose::Placement, 22, 0.05 * ez, 0.95 * ez);
            self.cells.push(Cell {
                id,
                parent: None,
                pos: [x, y, z],
                radius: cfg.birth_radius,
                contents: vec![0.0; chem.n_compounds()],
                reserve: 0.0,
                damage: 0.0,
                age: 0.0,
                state: CellState::Alive,
                generation: 0,
                traits: Traits::founder(cfg, rng, id),
                // Tick zero, entity `id`: a founder's mutations are a pure
                // function of which founder it is, like its lifespan and its
                // position, so the cohort is settled before the world starts
                // rather than by the order they were pushed. Founders whose
                // draw came up empty share the ancestor's allocation with each
                // other, which is what makes them one clone line rather than
                // sixteen identical ones.
                genome: founder
                    .as_ref()
                    .map(|a| inherit_genome(a, cfg, chem, rng, 0, id)),
                proteome: Vec::new(),
            });
        }
        self.next_id = self.cells.len() as u64;
        self.births += self.cells.len() as u64;
    }

    /// A population with its founding cohort already in it, eating whatever
    /// `abundance` says the pond holds.
    pub fn seed(
        cfg: &CellConfig,
        chem: &Chemistry,
        grid: &Grid,
        rng: &Counter,
        abundance: &[f64],
    ) -> Self {
        let mut population = Self::empty();
        population.introduce(cfg, chem, grid, rng, abundance);
        population
    }

    pub fn alive(&self) -> usize {
        self.cells.iter().filter(|c| c.is_alive()).count()
    }

    pub fn decomposing(&self) -> usize {
        self.cells.len() - self.alive()
    }

    /// Living cells currently shut down. Counted inside [`alive`](Self::alive),
    /// and reported separately because a population that has gone quiet and one
    /// that is working draw the same line on a population chart.
    pub fn dormant(&self) -> usize {
        self.cells.iter().filter(|c| c.is_dormant()).count()
    }

    /// Where the living population's heritable traits currently sit.
    ///
    /// Two numbers, and both are needed. The mean says which way selection is
    /// pushing; the spread says whether there is anything left for it to push
    /// on. A spread that collapses to nothing is the freeze coming back --
    /// the population has become clones again, by a different route.
    pub fn trait_summary(&self) -> TraitSummary {
        let mut n = 0.0;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut genes = 0.0;
        let mut bytes = 0.0;
        // Distinct genomes, counted by content.
        //
        // Pointer identity would be cheaper and is what an `Arc` already
        // tracks -- it is shared exactly when a lineage replicated without
        // mutating -- but an address freed by the last cell of a dead lineage
        // can be handed straight back to a new one, and then two unrelated
        // genomes count as one. A metric that quietly undercounts diversity is
        // worse than a slower one.
        let mut lineages: Vec<u64> = Vec::new();
        for cell in self.cells.iter().filter(|c| c.is_alive()) {
            let u = cell.uptake() as f64;
            n += 1.0;
            sum += u;
            sum_sq += u * u;
            genes += cell.gene_count() as f64;
            bytes += cell.genome_len() as f64;
            if let Some(g) = &cell.genome {
                let mut h = StateHasher::new();
                g.hash_state(&mut h);
                let id = h.finish();
                if let Err(i) = lineages.binary_search(&id) {
                    lineages.insert(i, id);
                }
            }
        }
        if n == 0.0 {
            return TraitSummary::default();
        }
        let mean = sum / n;
        TraitSummary {
            mean_uptake: mean,
            uptake_spread: (sum_sq / n - mean * mean).max(0.0).sqrt(),
            mean_genes: genes / n,
            mean_genome_bytes: bytes / n,
            distinct_genomes: lineages.len(),
        }
    }

    /// What the living population currently catalyses, and how many cells
    /// catalyse each thing.
    ///
    /// The reading the genome exists to make possible. A pre-genome population
    /// returns one row for ever, because one reaction is all any of them can
    /// ever run; a genome population's rows are its diet, and a second row
    /// appearing partway through a run is a lineage that found a different
    /// living. Sorted by reaction id, so the same population always reports in
    /// the same order.
    pub fn diet(&self) -> Vec<(ReactionId, usize)> {
        let mut rows: Vec<(ReactionId, usize)> = Vec::new();
        let mut bump = |id: ReactionId| match rows.binary_search_by_key(&id, |r| r.0) {
            Ok(i) => rows[i].1 += 1,
            Err(i) => rows.insert(i, (id, 1)),
        };
        for cell in self.cells.iter().filter(|c| c.is_alive()) {
            match &cell.genome {
                None => {
                    if let Some(id) = self.metabolic_reaction {
                        bump(id);
                    }
                }
                Some(g) => {
                    // The reaction each enzyme runs hardest. A promiscuous
                    // enzyme touches several weakly and reporting all of them
                    // would drown the row that pays the bills.
                    for ((gene, targets), &conc) in
                        g.genes.iter().zip(&g.targets).zip(&cell.proteome)
                    {
                        if gene.class != ProteinClass::Enzyme || conc <= 0.05 {
                            continue;
                        }
                        if let Some(&(id, _)) = targets
                            .reactions
                            .iter()
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        {
                            bump(id);
                        }
                    }
                }
            }
        }
        rows
    }

    pub fn stored_energy(&self, chem: &Chemistry) -> Joules {
        self.cells
            .iter()
            .map(|c| c.chemical_energy(chem) + c.reserve)
            .sum()
    }

    pub fn total_reserve(&self) -> Joules {
        self.cells.iter().map(|c| c.reserve).sum()
    }

    /// Advance every cell, in order of the voxel it occupies.
    ///
    /// The order matters twice. It has to be *fixed*, because cells clamp
    /// against the amounts the previous cell left in their shared voxel, so
    /// the result depends on who goes first. And it is worth making it
    /// *spatial*: a cell reads one voxel out of every compound plane, and
    /// those planes are hundreds of kilobytes apart, so a cell costs about
    /// thirty cache misses. Two cells in neighbouring voxels want the same
    /// thirty cache lines, and sorting is what puts them next to each other.
    ///
    /// New daughters are appended only after the current population has
    /// finished its tick, so a cell born this tick does not also act this
    /// tick.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        cfg: &CellConfig,
        grid: &Grid,
        chem: &Chemistry,
        flow: &FaceVelocity,
        rng: &Counter,
        tick: u64,
        dt: f64,
        growth_interval: u64,
        amounts: &mut ChemField,
        residual: &mut ChemField,
        heat: &mut HeatField,
    ) {
        // Walking `chem.compounds` per cell drags a molecule, a name and two
        // spectra through the cache to read two floats. Flatten them once.
        let table = CompoundTable::new(chem);
        let mut expression = Expression::new(chem);
        let mut levels = Vec::new();
        let mut daughters = Vec::new();
        // Corpses are retained while their conserved contents cross back into
        // the fields, but they must not consume a slot in the *living*
        // population cap. Otherwise a synchronous die-off blocks every
        // survivor from reproducing until decomposition finishes and can turn
        // an ordinary crash into an artificial extinction.
        let mut division_slots = cfg.population_cap.saturating_sub(self.alive());
        let metabolism = self.metabolic_reaction;
        for index in voxel_order(&self.cells, grid) {
            let cell = &mut self.cells[index as usize];
            match cell.state {
                // Dormancy changes what a cell spends, not what physics does
                // to it, so a shut-down cell drifts, exchanges and catalyses
                // exactly as a working one does. The two differences are in
                // `maintain`, which sets the bill and the state, and in
                // division below.
                CellState::Alive | CellState::Dormant => {
                    move_cell(
                        cell,
                        grid,
                        flow,
                        rng,
                        tick,
                        dt,
                        heat.temperature(cell.voxel(grid)),
                    );
                    // What this cell's heritable state comes to today. For a
                    // pre-genome cell this is its one trait and its one
                    // reaction; for a genome cell it is whatever its proteome
                    // currently expresses, which changed since last tick and
                    // will change again.
                    transcribe(cell, &mut levels, dt);
                    expression.read(cell, metabolism);
                    exchange(cell, cfg, grid, &table, &expression, amounts, residual, dt);
                    metabolize(cell, cfg, grid, chem, &expression, &table, heat, dt);
                    maintain(cell, cfg, grid, heat, dt);
                    cell.age += dt as f32;
                    cell.radius = growth_radius(cfg, cell.reserve);

                    // Only a cell paying its way divides. Dormancy is what
                    // a cell does instead of growing, and `dormancy_exit` sits
                    // far below `division_reserve` anyway, so a cell with the
                    // reserve to split has long since woken up.
                    if cell.state == CellState::Alive
                        && division_slots > 0
                        && Schedule::due_for(tick, cell.id, growth_interval)
                        && cell.reserve >= cfg.division_reserve
                    {
                        let child = divide(cell, cfg, chem, grid, rng, tick, self.next_id);
                        self.next_id += 1;
                        self.births += 1;
                        division_slots -= 1;
                        daughters.push(child);
                    }

                    if cell.damage >= 1.0 || cell.age >= lifespan(cfg, rng, cell.id) {
                        cell.state = CellState::Decomposing;
                        self.deaths += 1;
                    }
                }
                CellState::Decomposing => decompose(cell, cfg, grid, amounts, residual, heat, dt),
            }
        }
        self.cells.extend(daughters);
        // A corpse stays until every conserved quantity has actually crossed
        // into an f32 field; this can take extra ticks when a transfer is below
        // the field's current ULP.
        self.cells.retain(|c| {
            c.is_alive()
                || c.reserve.abs() > 1.0e-25
                || c.contents.iter().any(|&n| n.abs() > 1.0e-6)
        });
    }
}

impl HashState for Population {
    fn hash_state(&self, h: &mut StateHasher) {
        h.u64(self.next_id);
        h.u32(self.metabolic_reaction.unwrap_or(u32::MAX));
        h.u64(self.births);
        h.u64(self.deaths);
        h.usize(self.cells.len());
        for cell in &self.cells {
            cell.hash_state(h);
        }
    }
}

/// Where a population's heritable traits sit, for the run log.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TraitSummary {
    pub mean_uptake: f64,
    /// Standard deviation of `uptake` across the living population.
    pub uptake_spread: f64,
    /// Mean decoded genes per living cell. Zero without genomes.
    ///
    /// `PLAN.md` asks for this column by name: a healthy run shows genome
    /// length growing and then stabilising, and monotonic shrinkage means the
    /// upkeep charged per protein is too harsh for anything to be worth
    /// keeping.
    pub mean_genes: f64,
    /// Mean genome length in bytes, junk included.
    pub mean_genome_bytes: f64,
    /// Distinct genome allocations among the living -- clone lines.
    ///
    /// One means the population has become a single clone, which is the freeze
    /// that individual traits were introduced to break, arriving by a longer
    /// road. It is the number to watch beside the population.
    pub distinct_genomes: usize,
}

impl TraitSummary {
    /// Standing variation as a fraction of the mean -- the `d` in the turnover
    /// budget, measured off the population instead of assumed.
    ///
    /// [`CellConfig::turnover_budget`] takes `trait_spread` as its stand-in
    /// for this, which is right for a pre-genome population because
    /// `trait_spread` *is* the mutational kick every daughter gets. It is
    /// wrong for a genome population, where variation is whatever the genome
    /// happens to be producing, and wrong by enough to matter: the first
    /// `evolve.toml` run settled at a mean uptake of 2.428 with a spread of
    /// 0.242, so `d` was 0.0997 against the 0.5 the pre-run check assumed. The
    /// check said the budget was short by a factor of 1.7 when it was short by
    /// a factor of 8.4.
    pub fn relative_spread(&self) -> f64 {
        if self.mean_uptake > 0.0 {
            self.uptake_spread / self.mean_uptake
        } else {
            0.0
        }
    }
}

/// Mean particles per voxel a compound needs before it counts as food at all.
///
/// `N_REF` is the pond's working concentration, about a hundred micromolar, so
/// this is a ten-thousandth of that: present, if thinly. It is a viability
/// floor, not a preference -- preference is what the payoff below is for.
const FOOD_FLOOR: f64 = hadean_core::units::N_REF as f64 * 1.0e-4;

/// Pick the ancestor's metabolism: the reaction offering the largest living,
/// out of what the pond has actually accumulated.
///
/// "Largest living" is joules on the table -- how many turnovers the scarcest
/// substrate allows, times what each turnover releases. Both halves are
/// needed. Ranking on `dh` alone picks a marginally stronger reaction running
/// on a substrate the pond holds three parts per million of, and the founding
/// cohort starves in a pond with a thousand times more food in it that they
/// cannot eat.
///
/// The measurement is the other half of the point. Two earlier versions
/// reasoned about the reaction network instead -- "is this substrate one hop
/// from sunlight?", then "can the network reach it at all?" -- and both handed
/// some seeds a metabolism their world never feeds. Reachability is a real
/// improvement on one-hop and still not enough: a compound at the end of a
/// strongly uphill chain is reachable, and its steady-state abundance is
/// nothing. On seed 4 the founding cohort drew `H2 + CH4O2N2`, and after a
/// hundred and fifty seconds the pond had made no CH4O2N2 at all.
///
/// So the choice is made from `abundance`, the particle count of every
/// compound in the pond at the moment life appears -- after the world has run
/// lifeless long enough for its photochemistry to reach a steady state. That
/// is also the more honest story: life does not arrive with a plan, it arrives
/// to a pond that already contains something worth eating.
///
/// `None` when no downhill reaction runs on anything the pond holds. Not every
/// generated world offers a living, and saying so is better than handing over
/// a metabolism that never turns over.
pub fn choose_metabolism(chem: &Chemistry, abundance: &[f64], voxels: usize) -> Option<ReactionId> {
    let floor = FOOD_FLOOR * voxels.max(1) as f64;
    let held = |c: u16| abundance.get(c as usize).copied().unwrap_or(0.0);

    chem.reactions
        .iter()
        .filter(|r| r.drive == Drive::Thermal && r.dh < 0.0)
        .filter_map(|r| {
            // Turnovers the scarcest substrate allows, and the energy they
            // would release.
            let turnovers = r
                .reactants
                .iter()
                .map(|&(c, n)| {
                    if held(c) < floor {
                        0.0
                    } else {
                        held(c) / n.max(1) as f64
                    }
                })
                .fold(f64::INFINITY, f64::min);
            (turnovers > 0.0).then(|| (r.id, turnovers * -r.dh))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id)
}

/// What a cell's heritable state comes to, this tick, in the units the
/// lifecycle actually uses.
///
/// Both cell types produce one of these and nothing downstream can tell them
/// apart. That is the point of the type: `exchange`, `metabolize` and
/// `maintain` are the audited, measured code from Phase 2, and they keep
/// working on a genome cell without being rewritten to understand one.
///
/// Reused across cells rather than allocated per cell per tick -- there are
/// two vectors here the length of the compound and reaction pools, and a
/// population of a few hundred would otherwise allocate them a few hundred
/// times a tick.
pub struct Expression {
    /// Membrane permeability multiplier, per compound.
    ///
    /// A genome cell's is `1 + transporters`: the one is the lipid bilayer,
    /// which passes small nonpolar molecules whatever the cell would prefer,
    /// and the rest is machinery it evolved. So a cell can specialise on its
    /// food without ever being able to seal itself off, which is both the
    /// physics and the thing that stops a lineage from mutating into a sealed
    /// box that starves.
    pub uptake: Vec<f32>,
    /// Catalysed rate per reaction, in multiples of `metabolic_rate`.
    pub catalysis: Vec<f32>,
    /// Reactions with a nonzero rate, ascending. Applied in id order, which is
    /// the same convention the field chemistry uses and for the same reason:
    /// the result depends on the order, so the order has to be fixed.
    pub active: Vec<ReactionId>,
}

impl Expression {
    pub fn new(chem: &Chemistry) -> Self {
        Self {
            uptake: vec![0.0; chem.n_compounds()],
            catalysis: vec![0.0; chem.reactions.len()],
            active: Vec::new(),
        }
    }

    /// Read one cell's heritable state into this scratch.
    pub fn read(&mut self, cell: &Cell, metabolism: Option<ReactionId>) {
        for r in self.active.drain(..) {
            self.catalysis[r as usize] = 0.0;
        }

        let Some(g) = &cell.genome else {
            // The pre-genome protocell: one multiplier for every compound and
            // one reaction for every cell.
            self.uptake.fill(cell.traits.uptake);
            if let Some(r) = metabolism {
                self.catalysis[r as usize] = 1.0;
                self.active.push(r);
            }
            return;
        };

        // An unbound genome has no targets, so the zip below yields nothing
        // and the cell silently does nothing at all -- which is exactly what
        // the clamped-key bug looked like from the outside, and it took a
        // four-hundred-second run to notice. Every path that hands a genome to
        // a cell binds it first; this is here so that a path that forgets says
        // so immediately.
        debug_assert_eq!(
            g.genes.len(),
            g.targets.len(),
            "a cell is carrying a genome that was never bound to the chemistry"
        );

        self.uptake.fill(1.0);
        for ((gene, targets), &conc) in g.genes.iter().zip(&g.targets).zip(&cell.proteome) {
            if conc <= 0.0 {
                continue;
            }
            let amount = conc * gene.strength();
            match gene.class {
                ProteinClass::Enzyme => {
                    for &(r, a) in &targets.reactions {
                        let slot = &mut self.catalysis[r as usize];
                        if *slot == 0.0 {
                            self.active.push(r);
                        }
                        *slot += amount * a;
                    }
                }
                ProteinClass::Transporter => {
                    for &(c, a) in &targets.compounds {
                        self.uptake[c as usize] += amount * a;
                    }
                }
                _ => {}
            }
        }
        self.active.sort_unstable();
    }
}

/// The two per-compound numbers the cell layer reads in its inner loops.
///
/// A [`Compound`](hadean_chem::Compound) carries a name, an atom graph, two
/// eight-band spectra and a descriptor key. Membrane exchange wants one `f32`
/// from it and metabolism wants another, so iterating the compound vector per
/// cell per tick moves kilobytes to use bytes. Rebuilt each step, which costs
/// two passes over a few dozen floats and needs no state to keep in sync.
struct CompoundTable {
    permeability: Vec<f64>,
    /// Formation enthalpy, J per particle.
    h_f: Vec<f64>,
}

impl CompoundTable {
    fn new(chem: &Chemistry) -> Self {
        Self {
            permeability: chem
                .compounds
                .iter()
                .map(|c| c.permeability as f64)
                .collect(),
            h_f: chem.compounds.iter().map(|c| c.h_f).collect(),
        }
    }

    fn len(&self) -> usize {
        self.h_f.len()
    }

    /// Chemical energy held in a set of contents, J.
    fn energy_of(&self, contents: &[f64]) -> Joules {
        contents.iter().zip(&self.h_f).map(|(&n, &h)| n * h).sum()
    }
}

/// Copy a genome for a daughter and match the copy to the chemistry.
///
/// [`genome::replicate`] returns `None` when no operator fired, which at the
/// default rates is most divisions. That is worth keeping distinct from "the
/// daughter is identical anyway": when nothing fired the caller shares the
/// mother's `Arc` and pays for neither the copy nor the decode nor the
/// rebinding, and a colony of clones costs one genome between all of them.
/// This function is the path for when something *did* fire.
fn inherit_genome(
    parent: &Arc<Genome>,
    cfg: &CellConfig,
    chem: &Chemistry,
    rng: &Counter,
    tick: u64,
    id: u64,
) -> Arc<Genome> {
    match genome::replicate(parent, &cfg.mutation, rng, tick, id) {
        Some(mut child) => {
            child.bind(chem, cfg.enzyme_sigma, cfg.transport_sigma);
            Arc::new(child)
        }
        None => Arc::clone(parent),
    }
}

/// Indices into `cells`, ordered by the voxel each one occupies.
///
/// Ties break on index, which is ascending in cell id, so the order is a pure
/// function of the state and the replay stays bit-identical. Sorting packed
/// `(voxel, index)` pairs as one `u64` keeps the comparison a single integer
/// compare.
fn voxel_order(cells: &[Cell], grid: &Grid) -> Vec<u32> {
    let mut keys: Vec<u64> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| ((c.voxel(grid) as u64) << 32) | i as u64)
        .collect();
    keys.sort_unstable();
    keys.into_iter().map(|k| k as u32).collect()
}

fn cell_volume(radius: f32) -> f64 {
    (4.0 / 3.0 * PI * radius * radius * radius) as f64
}

#[allow(clippy::too_many_arguments)]
fn exchange(
    cell: &mut Cell,
    cfg: &CellConfig,
    grid: &Grid,
    table: &CompoundTable,
    expression: &Expression,
    amounts: &mut ChemField,
    residual: &mut ChemField,
    dt: f64,
) {
    let voxel = cell.voxel(grid);
    let area = (4.0 * PI * cell.radius * cell.radius) as f64;
    let cell_v = cell_volume(cell.radius);
    let voxel_v = grid.voxel_volume() as f64;
    for c in 0..table.len() {
        // Per compound now, because a genome cell's membrane is not one
        // number. Grouped exactly as it was so a cell with a flat profile --
        // which is every pre-genome cell -- gets the same f64 it always got.
        let scale = (cfg.membrane_scale as f64 * expression.uptake[c] as f64) * area * dt;
        let outside = amounts.get(c, voxel) as f64;
        let inside = cell.contents[c];
        let gradient = outside / voxel_v - inside / cell_v;
        let requested = table.permeability[c] * scale * gradient;
        // A cell is a ten-thousandth of a voxel by volume, so almost every
        // transfer it makes is far below an f32 step of the field it is
        // drawing on. `settle` carries the difference forward rather than
        // rounding it away; what it returns is what the field really gave up
        // or took on, and that is what the cell books.
        if requested > 0.0 {
            // Never take more than a fifth of the voxel in one step: the
            // membrane is a rate, not an instantaneous equilibration.
            let take = requested.min(outside * 0.2);
            cell.contents[c] -= amounts.settle(residual, c, voxel, -take);
        } else if requested < 0.0 && inside > 0.0 {
            let give = (-requested).min(inside * 0.2);
            cell.contents[c] -= amounts.settle(residual, c, voxel, give);
        }
    }
}

/// Run every reaction this cell has an enzyme for, in id order.
///
/// A pre-genome cell has exactly one and this is the Phase 2 function with a
/// loop of length one around it. A genome cell may have several, and that is
/// the substantive difference the genome makes: a lineage can hold two enzymes
/// and run a *pathway*, feeding one reaction's product into the next, which no
/// amount of tuning could give the hardcoded protocell.
///
/// Direction is decided by the reaction's own enthalpy, not by which way it
/// was written down: an enzyme runs its reaction downhill, because there is no
/// other direction to get energy from. Free energy cannot be manufactured here
/// for the same structural reason it cannot be anywhere else in this project --
/// what the cell banks is the *measured* fall in the chemical energy of its
/// own contents, and that is a difference of state functions.
///
/// What this does not model is equilibrium. The extent is a fraction of the
/// limiting substrate with no reverse flux, so a catalysed reaction runs to
/// completion rather than to its equilibrium position. That is inherited from
/// the pre-genome cell and is the largest simplification in the metabolism.
#[allow(clippy::too_many_arguments)]
fn metabolize(
    cell: &mut Cell,
    cfg: &CellConfig,
    grid: &Grid,
    chem: &Chemistry,
    expression: &Expression,
    table: &CompoundTable,
    heat: &mut HeatField,
    dt: f64,
) {
    let before = table.energy_of(&cell.contents);
    let mut ran = false;

    for &id in &expression.active {
        let reaction = chem.reaction(id);
        // Downhill, whichever way that is.
        let (from, to) = if reaction.dh < 0.0 {
            (&reaction.reactants, &reaction.products)
        } else {
            (&reaction.products, &reaction.reactants)
        };

        let rate = cfg.metabolic_rate as f64 * expression.catalysis[id as usize] as f64;
        let fraction = 1.0 - (-rate * dt).exp();
        let mut extent = f64::INFINITY;
        for &(c, n) in from {
            extent = extent.min(cell.contents[c as usize] / n as f64);
        }
        extent *= fraction;
        if !extent.is_finite() || extent <= 0.0 {
            continue;
        }

        for &(c, n) in from {
            cell.contents[c as usize] -= extent * n as f64;
        }
        for &(c, n) in to {
            cell.contents[c as usize] += extent * n as f64;
        }
        ran = true;
    }

    if !ran {
        return;
    }

    let after = table.energy_of(&cell.contents);
    let released = (before - after).max(0.0);
    let desired_heat = released * (1.0 - cfg.capture_efficiency as f64);
    let landed = heat.deposit(grid, cell.voxel(grid), desired_heat);
    // Whatever did not actually land as heat remains stored. This includes
    // normal heat-deposit rounding, keeping the audit exact at the boundary.
    cell.reserve += released - landed;
}

/// Advance one cell's proteome.
///
/// Concentration relaxes towards the gene's transcription level at the
/// protein's own turnover rate: `dc/dt = stability (level - c)`. Two things
/// fall out of writing it that way rather than as separate synthesis and decay
/// terms. Concentration is bounded in `0..1` by construction, so no mutation
/// can produce a cell holding a thousandfold of anything; and `stability`
/// means one thing -- how fast this protein tracks its gene -- rather than
/// being half of a ratio that sets the steady state.
///
/// The regulatory sum is what makes this a *network* and not a lookup: a gene
/// with binding sites is driven by whatever regulator proteins are present,
/// and regulators are themselves gene products. So a mutation in one gene's
/// key can retune the expression of every gene whose promoter it now
/// recognises, which is the mechanism L5 differentiation will need.
///
/// Stepped every tick rather than on `schedule.expression`. At a few genes per
/// cell it costs less than the staggering would, and the schedule exists to
/// save work, not to be obeyed.
/// `scratch` holds the transcription levels while they are computed. It is a
/// parameter rather than a local because the regulatory network has to be read
/// off the proteome the cell had at the *start* of the step, not off one that
/// is half updated -- otherwise a gene's own product feeds back into its own
/// level within a single tick -- and because a population of a few hundred
/// would otherwise allocate a vector per cell per tick.
fn transcribe(cell: &mut Cell, scratch: &mut Vec<f32>, dt: f64) {
    let Some(g) = cell.genome.as_ref() else {
        return;
    };
    let n = g.genes.len();
    if n == 0 {
        cell.proteome.clear();
        return;
    }
    scratch.clear();
    scratch.reserve(n);
    for (gene, targets) in g.genes.iter().zip(&g.targets) {
        let mut drive = gene.basal;
        for (site, bound_by) in gene.sites.iter().zip(&targets.sites) {
            let mut bound = 0.0;
            for &(j, a) in bound_by {
                bound += cell.proteome.get(j).copied().unwrap_or(0.0) * a;
            }
            drive += site.weight * bound.min(1.0);
        }
        scratch.push(drive.clamp(0.0, 1.0));
    }

    // A daughter whose genome mutated has a proteome the wrong length; she
    // starts her new genes from nothing and expresses them up.
    if cell.proteome.len() != n {
        cell.proteome.resize(n, 0.0);
    }
    for (i, c) in cell.proteome.iter_mut().enumerate() {
        *c += g.genes[i].stability * (scratch[i] - *c) * dt as f32;
    }
}

/// Pay for staying alive, and shut down rather than starve if that fails.
///
/// A cell that cannot meet its full bill used to keep trying out of a reserve
/// it did not have, accruing damage at a fixed rate until it died -- and it
/// kept eating the whole time, which is what stopped a crashed population from
/// ever recovering: the survivors consumed the trickle that would have funded
/// the comeback. A real cell facing famine stops instead. It drops its
/// discretionary spending, holds what it has, and waits.
///
/// Note what dormancy does *not* switch off. Passive membrane exchange is
/// diffusion down a gradient, and a catalyst the cell has already built goes
/// on catalysing; neither is a decision. So a dormant cell keeps taking up
/// what the water offers and banks it as reserve. Since its bill is
/// `dormancy_power_fraction` of the working one, it breaks even at that much
/// lower a food concentration -- which is exactly the refuge the population
/// has never had, and the reason it can outlast a night that would kill it
/// working.
fn maintain(cell: &mut Cell, cfg: &CellConfig, grid: &Grid, heat: &mut HeatField, dt: f64) {
    let working = cell.maintenance(cfg) * dt;
    if cell.state == CellState::Alive && cell.reserve < working {
        cell.state = CellState::Dormant;
    }
    let due = if cell.state == CellState::Dormant {
        working * cfg.dormancy_power_fraction
    } else {
        working
    };

    // What a cell can endure unpaid. Constant before genomes; for a genome
    // cell it is what its structural protein bought, and that is a real trade
    // because the same protein is on the bill above.
    let tolerance = cell.famine_tolerance(cfg);
    if cell.reserve >= due {
        let landed = heat.deposit(grid, cell.voxel(grid), due);
        cell.reserve -= landed;
        cell.damage = (cell.damage - dt as f32 / tolerance).max(0.0);
    } else {
        // Below even the dormant bill there is nothing left to cut, and
        // `starvation_time` measures what it always did: how long a cell that
        // cannot pay at all takes to die of it.
        cell.damage += dt as f32 / tolerance;
    }

    // Waking is a much higher bar than shutting down was. Without the gap a
    // cell on the margin flickers between the two every tick and pays
    // something close to the working bill on average, which is no dormancy at
    // all.
    if cell.state == CellState::Dormant
        && cell.reserve >= cell.maintenance(cfg) * cfg.dormancy_exit as f64
    {
        cell.state = CellState::Alive;
    }
}

/// How long this particular cell gets, in simulated seconds.
///
/// A pure function of the cell's id, so it is settled at birth, unchanged by a
/// snapshot, and costs no per-cell state. `Purpose::Death` has been reserved
/// for exactly this since L0 and is spent here.
fn lifespan(cfg: &CellConfig, rng: &Counter, id: u64) -> f32 {
    if cfg.lifespan_spread <= 0.0 {
        return cfg.maximum_age;
    }
    let spread = cfg.lifespan_spread.clamp(0.0, 1.0);
    cfg.maximum_age * rng.range(0, id, Purpose::Death, 0, 1.0 - spread, 1.0 + spread)
}

fn growth_radius(cfg: &CellConfig, reserve: Joules) -> f32 {
    let progress = (reserve / cfg.division_reserve).clamp(0.0, 1.0) as f32;
    cfg.birth_radius * (1.0 + (2.0f32.cbrt() - 1.0) * progress)
}

#[allow(clippy::too_many_arguments)]
fn divide(
    parent: &mut Cell,
    cfg: &CellConfig,
    chem: &Chemistry,
    grid: &Grid,
    rng: &Counter,
    tick: u64,
    child_id: u64,
) -> Cell {
    let fraction = rng.range(tick, parent.id, Purpose::Partition, 0, 0.47, 0.53) as f64;
    let mut child_contents = Vec::with_capacity(parent.contents.len());
    for n in &mut parent.contents {
        let child_n = *n * fraction;
        *n -= child_n;
        child_contents.push(child_n);
    }
    let child_reserve = parent.reserve * fraction;
    parent.reserve -= child_reserve;
    parent.radius = growth_radius(cfg, parent.reserve);

    let angle = rng.range(
        tick,
        parent.id,
        Purpose::Division,
        0,
        0.0,
        std::f32::consts::TAU,
    );
    let separation = cfg.birth_radius * 1.1;
    let mut child_pos = parent.pos;
    child_pos[0] += separation * angle.cos();
    child_pos[1] += separation * angle.sin();
    clamp_position(&mut child_pos, grid, cfg.birth_radius);
    parent.pos[0] -= separation * angle.cos();
    parent.pos[1] -= separation * angle.sin();
    clamp_position(&mut parent.pos, grid, cfg.birth_radius);

    // Replication. A daughter gets her mother's cytoplasm whether or not the
    // genome changed: a cell divides its protein along with everything else,
    // and a mutant that had to re-express its whole proteome from nothing
    // would be handicapped by the fact of having mutated rather than by what
    // the mutation did. `transcribe` truncates or pads the copy to the
    // daughter's own gene count, so a point substitution -- which leaves the
    // gene list the same length and in the same order -- carries across
    // exactly, and only genuinely new genes start at zero.
    let (child_genome, child_proteome) = match &parent.genome {
        None => (None, Vec::new()),
        Some(mine) => (
            Some(inherit_genome(mine, cfg, chem, rng, tick, parent.id)),
            parent.proteome.clone(),
        ),
    };

    Cell {
        id: child_id,
        parent: Some(parent.id),
        pos: child_pos,
        radius: growth_radius(cfg, child_reserve),
        contents: child_contents,
        reserve: child_reserve,
        damage: parent.damage,
        age: 0.0,
        state: CellState::Alive,
        generation: parent.generation + 1,
        traits: parent.traits.inherit(cfg, rng, tick, parent.id),
        genome: child_genome,
        proteome: child_proteome,
    }
}

/// Particle counts below one are not a quantity of anything, so a corpse
/// holding less than this has finished returning that compound.
const ONE_PARTICLE: f64 = 1.0;

/// The same idea for the energy reserve: far below the energy of a single
/// chemical bond, there is nothing left to give back.
const LAST_JOULE: Joules = 1.0e-24;

fn decompose(
    cell: &mut Cell,
    cfg: &CellConfig,
    grid: &Grid,
    amounts: &mut ChemField,
    residual: &mut ChemField,
    heat: &mut HeatField,
    dt: f64,
) {
    let fraction = 1.0 - (-cfg.decomposition_rate as f64 * dt).exp();
    let voxel = cell.voxel(grid);
    for c in 0..cell.contents.len() {
        let held = cell.contents[c];
        if held <= 0.0 {
            continue;
        }
        // Exponential decay never reaches zero, so a corpse whose remainder
        // is smaller than a single particle would be carried, and stepped,
        // for ever. Below that there is nothing left to decompose.
        let give = if held < ONE_PARTICLE {
            held
        } else {
            held * fraction
        };
        cell.contents[c] -= amounts.settle(residual, c, voxel, give);
    }
    if cell.reserve != 0.0 {
        let requested = if cell.reserve.abs() < LAST_JOULE {
            cell.reserve
        } else {
            cell.reserve * fraction
        };
        let landed = heat.deposit(grid, voxel, requested);
        cell.reserve -= landed;
    }
    cell.radius *= (1.0 - 0.2 * fraction as f32).max(0.0);
}

fn move_cell(
    cell: &mut Cell,
    grid: &Grid,
    flow: &FaceVelocity,
    rng: &Counter,
    tick: u64,
    dt: f64,
    temperature: f32,
) {
    let voxel = cell.voxel(grid);
    let (x, y, z) = grid.coords(voxel);
    let (x, y, z) = (x as usize, y as usize, z as usize);
    let velocity = [
        0.5 * (flow.get(Axis::X, x, y, z) + flow.get(Axis::X, x + 1, y, z)),
        0.5 * (flow.get(Axis::Y, x, y, z) + flow.get(Axis::Y, x, y + 1, z)),
        0.5 * (flow.get(Axis::Z, x, y, z) + flow.get(Axis::Z, x, y, z + 1)),
    ];
    let diffusion = KB * temperature as f64
        / (6.0 * std::f64::consts::PI * VISCOSITY as f64 * cell.radius as f64);
    let sigma = (2.0 * diffusion * dt).sqrt() as f32;
    for (axis, &flow_velocity) in velocity.iter().enumerate() {
        let brownian = rng.normal(tick, cell.id, Purpose::Brownian, axis as u64) * sigma;
        cell.pos[axis] += flow_velocity * dt as f32 + brownian;
    }
    clamp_position(&mut cell.pos, grid, cell.radius);
}

fn clamp_position(pos: &mut [f32; 3], grid: &Grid, radius: f32) {
    let extent = grid.extent();
    for axis in 0..3 {
        let hi = [extent.0, extent.1, extent.2][axis];
        let margin = radius.min(0.49 * hi);
        pos[axis] = pos[axis].clamp(margin, hi - margin);
    }
}

/// Add the cellular contribution to per-element totals.
pub fn add_element_totals(
    population: &Population,
    chem: &Chemistry,
    totals: &mut [f64; hadean_chem::element::N_ELEMENTS],
) {
    for cell in &population.cells {
        for (c, &amount) in cell.contents.iter().enumerate() {
            if amount == 0.0 {
                continue;
            }
            for (e, total) in totals.iter_mut().enumerate() {
                *total += amount * chem.compounds[c].formula[e] as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hadean_chem::{generate, ChemParams};
    use hadean_fields::flow::{convection_roll, FlowConfig};
    use hadean_fields::heat::HeatField;

    /// A small lit pond with two ancestors in it, plus the rounding residual
    /// every field transfer settles into.
    #[allow(clippy::type_complexity)]
    fn setup() -> (
        CellConfig,
        Grid,
        Chemistry,
        Counter,
        ChemField,
        ChemField,
        HeatField,
        FaceVelocity,
    ) {
        let cfg = CellConfig {
            initial_count: 2,
            ..Default::default()
        };
        let grid = Grid::new(8, 8, 6, 25.0e-6);
        let chem = generate(1, ChemParams::default());
        let rng = Counter::new(1);
        let mut amounts = ChemField::new(&grid, chem.n_compounds());
        for c in 0..chem.n_compounds() {
            amounts.plane_mut(c).fill(2.0e9);
        }
        let residual = ChemField::new(&grid, chem.n_compounds());
        let heat = HeatField::new(&grid, 293.15);
        let flow = convection_roll(
            &grid,
            &FlowConfig {
                speed: 0.0,
                rolls: 1,
            },
        );
        (cfg, grid, chem, rng, amounts, residual, heat, flow)
    }

    /// Particles of every compound in a field, as `introduce` wants them.
    /// The [`Expression`] a pre-genome cell reads out. Tests below poke at
    /// `exchange` directly, and it now wants one.
    fn flat(chem: &Chemistry, cell: &Cell, metabolism: Option<ReactionId>) -> Expression {
        let mut e = Expression::new(chem);
        e.read(cell, metabolism);
        e
    }

    fn abundance(chem: &Chemistry, amounts: &ChemField) -> Vec<f64> {
        (0..chem.n_compounds())
            .map(|c| amounts.total_of(c))
            .collect()
    }

    #[test]
    fn generated_world_has_a_metabolism() {
        let (cfg, grid, chem, rng, amounts, ..) = setup();
        let p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        let r = chem.reaction(p.metabolic_reaction.expect("reaction"));
        assert!(r.dh < 0.0);
        assert_eq!(r.drive, Drive::Thermal);
    }

    #[test]
    fn a_metabolism_is_never_chosen_for_food_the_pond_does_not_have() {
        // This is the failure that moved the choice from the reaction graph to
        // a measurement. On seed 5 the most exergonic downhill reaction ate a
        // compound whose only source was a photoreaction with no substrates;
        // on seed 4 it ate one that is reachable in principle and never
        // accumulates. Both cohorts sat there with a food supply of exactly
        // zero.
        let (_, grid, chem, _, amounts, ..) = setup();
        let full = abundance(&chem, &amounts);
        let chosen = choose_metabolism(&chem, &full, grid.len()).expect("a metabolism");

        // Take away one of its substrates and it must pick something else.
        let mut starved = full.clone();
        let missing = chem.reaction(chosen).reactants[0].0;
        starved[missing as usize] = 0.0;
        let next = choose_metabolism(&chem, &starved, grid.len());
        assert_ne!(next, Some(chosen), "kept a metabolism with no substrate");
        if let Some(next) = next {
            for &(c, _) in &chem.reaction(next).reactants {
                assert!(starved[c as usize] > 0.0, "chose an absent substrate");
            }
        }

        // A trace is not a food supply either.
        let mut trace = full.clone();
        trace[missing as usize] = 1.0;
        assert_ne!(
            choose_metabolism(&chem, &trace, grid.len()),
            Some(chosen),
            "a single particle counted as a larder"
        );

        // And an empty pond offers no living at all.
        assert_eq!(
            choose_metabolism(&chem, &vec![0.0; chem.n_compounds()], grid.len()),
            None
        );
    }

    #[test]
    fn a_thousandfold_larder_beats_a_slightly_better_reaction() {
        // Ranking on enthalpy alone picks the strongest reaction present at
        // all, which in a real pond is routinely one running on parts per
        // million. The cohort then starves surrounded by food it does not eat.
        let (_, grid, chem, _, amounts, ..) = setup();
        let mut pond = abundance(&chem, &amounts);
        let chosen = choose_metabolism(&chem, &pond, grid.len()).expect("a metabolism");

        // Leave the chosen reaction's substrates barely above the floor and
        // everything else as it was. It must lose its place.
        let floor = FOOD_FLOOR * grid.len() as f64;
        for &(c, _) in &chem.reaction(chosen).reactants {
            pond[c as usize] = floor * 1.5;
        }
        let next = choose_metabolism(&chem, &pond, grid.len()).expect("a metabolism");
        assert_ne!(
            next, chosen,
            "kept a metabolism whose larder had dropped by orders of magnitude"
        );
    }

    #[test]
    fn membrane_transfer_conserves_particle_counts() {
        let (cfg, grid, chem, rng, mut amounts, mut residual, ..) = setup();
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        let before: f64 = amounts.data.iter().map(|&x| x as f64).sum();
        assert_eq!(residual.data.iter().map(|&x| x as f64).sum::<f64>(), 0.0);
        for cell in &mut p.cells {
            let e = flat(&chem, cell, p.metabolic_reaction);
            exchange(
                cell,
                &cfg,
                &grid,
                &CompoundTable::new(&chem),
                &e,
                &mut amounts,
                &mut residual,
                0.01,
            );
        }
        // Conservation is over the field, its deferred rounding, and the
        // cells -- the residual is state, not scratch.
        let after_field: f64 = amounts.data.iter().map(|&x| x as f64).sum();
        let after_residual: f64 = residual.data.iter().map(|&x| x as f64).sum();
        let after_cells: f64 = p.cells.iter().flat_map(|c| &c.contents).sum();
        assert_eq!(before, after_field + after_residual + after_cells);
    }

    #[test]
    fn a_fed_cell_charges_its_reserve() {
        let (cfg, grid, chem, rng, mut amounts, mut residual, mut heat, flow) = setup();
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        for tick in 0..200 {
            p.step(
                &cfg,
                &grid,
                &chem,
                &flow,
                &rng,
                tick,
                0.01,
                20,
                &mut amounts,
                &mut residual,
                &mut heat,
            );
        }
        assert!(p.total_reserve() > 0.0);
    }

    #[test]
    fn division_preserves_contents_and_reserve() {
        let (cfg, grid, chem, rng, amounts, ..) = setup();
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        let parent = &mut p.cells[0];
        parent.contents.fill(123_456.0);
        parent.reserve = cfg.division_reserve;
        let before_contents: f64 = parent.contents.iter().sum();
        let before_reserve = parent.reserve;
        let child = divide(parent, &cfg, &chem, &grid, &rng, 20, 99);
        assert_eq!(
            before_contents,
            parent.contents.iter().sum::<f64>() + child.contents.iter().sum::<f64>()
        );
        assert_eq!(before_reserve, parent.reserve + child.reserve);
        assert_eq!(child.parent, Some(parent.id));
    }

    #[test]
    fn a_starved_cell_dies_and_returns_its_contents() {
        let (mut cfg, grid, chem, rng, mut amounts, mut residual, mut heat, flow) = setup();
        cfg.initial_count = 1;
        cfg.membrane_scale = 0.0;
        cfg.starvation_time = 0.01;
        cfg.decomposition_rate = 100.0;
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        p.metabolic_reaction = None;
        amounts.data.fill(0.0);
        p.cells[0].contents[chem.water as usize] = 1.0e6;
        let before =
            amounts.total_of(chem.water as usize) + p.cells[0].contents[chem.water as usize];

        for tick in 0..20 {
            p.step(
                &cfg,
                &grid,
                &chem,
                &flow,
                &rng,
                tick,
                0.01,
                20,
                &mut amounts,
                &mut residual,
                &mut heat,
            );
        }

        let in_field =
            amounts.total_of(chem.water as usize) + residual.total_of(chem.water as usize);
        let in_cells: f64 = p
            .cells
            .iter()
            .map(|c| c.contents[chem.water as usize])
            .sum();
        assert_eq!(p.deaths, 1);
        assert!(in_field > 0.0, "corpse returned no matter");
        assert_eq!(before, in_field + in_cells);
    }

    #[test]
    fn a_corpse_decomposes_completely_into_a_full_pond() {
        // The failure this guards against is quiet: a corpse's last few
        // thousand particles are below half an f32 step of a voxel holding
        // 2e11, so the hand-back rounds to nothing and repeats that failure
        // for ever. The population then carries every cell that ever died,
        // and their matter never reaches the survivors.
        let (mut cfg, grid, chem, rng, mut amounts, mut residual, mut heat, flow) = setup();
        cfg.initial_count = 1;
        cfg.membrane_scale = 0.0;
        cfg.metabolic_rate = 0.0;
        for c in 0..chem.n_compounds() {
            amounts.plane_mut(c).fill(2.0e11);
        }

        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        p.cells[0].state = CellState::Decomposing;
        p.cells[0].contents.fill(1.0e3);
        let corpse: f64 = p.cells[0].contents.iter().sum();

        let before = amounts.data.iter().map(|&x| x as f64).sum::<f64>()
            + residual.data.iter().map(|&x| x as f64).sum::<f64>();

        // At 0.8 per second, a thousand particles takes about nine seconds to
        // fall below the last one.
        for tick in 0..1_500 {
            p.step(
                &cfg,
                &grid,
                &chem,
                &flow,
                &rng,
                tick,
                0.01,
                20,
                &mut amounts,
                &mut residual,
                &mut heat,
            );
        }

        assert!(
            p.cells.is_empty(),
            "{} corpses never finished; one still holds {:?}",
            p.cells.len(),
            p.cells.first().map(|c| c.contents.iter().sum::<f64>())
        );
        let after = amounts.data.iter().map(|&x| x as f64).sum::<f64>()
            + residual.data.iter().map(|&x| x as f64).sum::<f64>();
        assert!(
            (after - before - corpse).abs() < 1.0e-3,
            "the pond gained {} of the corpse's {corpse} particles",
            after - before
        );
    }

    #[test]
    fn corpses_feed_the_survivors() {
        // The Phase 2 gate says corpses visibly feed the living, so this is
        // the mechanism behind that claim, isolated: an empty pond, one
        // starving cell, and one corpse in the same voxel holding a meal.
        // The only route from the corpse's contents to the survivor's reserve
        // is decomposition into the field and uptake back out of it.
        let feed = |decomposition_rate: f32| -> (f64, f64) {
            let (mut cfg, grid, chem, rng, mut amounts, mut residual, mut heat, flow) = setup();
            cfg.initial_count = 2;
            cfg.maintenance_power = 0.0;
            cfg.division_reserve = 1.0e30;
            cfg.decomposition_rate = decomposition_rate;

            // Choose the metabolism from the stocked pond, then empty it, so
            // that the only food left in this world is inside the corpse.
            let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
            amounts.data.fill(0.0);
            let reaction = chem.reaction(p.metabolic_reaction.expect("metabolism"));
            let meal: Vec<(usize, f64)> = reaction
                .reactants
                .iter()
                .map(|&(c, n)| (c as usize, 1.0e8 * n as f64))
                .collect();

            // Put both cells in one voxel so the corpse's return lands where
            // the survivor can reach it.
            let position = p.cells[0].pos;
            p.cells[1].pos = position;
            p.cells[1].state = CellState::Decomposing;
            for &(c, amount) in &meal {
                p.cells[1].contents[c] = amount;
            }

            // Long enough for the corpse's hundred million particles to
            // decay past the last one at 0.8 per second.
            for tick in 0..3_000 {
                p.step(
                    &cfg,
                    &grid,
                    &chem,
                    &flow,
                    &rng,
                    tick,
                    0.01,
                    20,
                    &mut amounts,
                    &mut residual,
                    &mut heat,
                );
            }
            let survivor = p.cells.iter().find(|c| c.is_alive()).expect("survivor");
            let corpse_left: f64 = p
                .cells
                .iter()
                .filter(|c| !c.is_alive())
                .flat_map(|c| &c.contents)
                .sum();
            (survivor.reserve, corpse_left)
        };

        let (fed, left_over) = feed(0.8);
        // Decomposition so slow it is effectively off for this run: the meal
        // stays locked in the corpse.
        let (unfed, locked) = feed(1.0e-6);

        assert!(fed > 0.0, "the survivor got nothing from the corpse");
        assert!(
            fed > 100.0 * unfed.max(f64::MIN_POSITIVE),
            "recycling barely mattered: {fed:e} J fed vs {unfed:e} J unfed"
        );
        assert!(left_over < 1.0, "the corpse kept {left_over} particles");
        assert!(locked > 1.0e7, "the control corpse decomposed anyway");
    }

    #[test]
    fn a_corpse_does_not_block_a_living_population_slot() {
        let (mut cfg, grid, chem, rng, mut amounts, mut residual, mut heat, flow) = setup();
        cfg.population_cap = 2;
        cfg.membrane_scale = 0.0;
        cfg.metabolic_rate = 0.0;
        cfg.maintenance_power = 0.0;

        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        p.cells[0].state = CellState::Decomposing;
        p.cells[1].reserve = cfg.division_reserve;

        p.step(
            &cfg,
            &grid,
            &chem,
            &flow,
            &rng,
            0,
            0.01,
            1,
            &mut amounts,
            &mut residual,
            &mut heat,
        );

        assert_eq!(p.births, 3);
        assert_eq!(p.alive(), 2);
    }

    /// One cell against a fixed bill and a fixed income, stepped through
    /// `maintain` alone.
    ///
    /// `income` is a multiple of the working maintenance bill, which is the
    /// unit that matters: a pond offering less than 1.0 cannot keep a working
    /// cell alive, and the question dormancy answers is how far below 1.0 a
    /// cell can still persist.
    fn dormancy_bench(
        cfg: &CellConfig,
        opening_seconds: f64,
        income: f64,
        ticks: usize,
    ) -> (Cell, HeatField, Joules) {
        let grid = Grid::new(2, 2, 2, 25.0e-6);
        let mut heat = HeatField::new(&grid, 293.15);
        let mut cell = Cell {
            id: 0,
            parent: None,
            pos: [25.0e-6, 25.0e-6, 25.0e-6],
            radius: cfg.birth_radius,
            contents: vec![0.0; 4],
            reserve: cfg.maintenance_power * opening_seconds,
            damage: 0.0,
            age: 0.0,
            state: CellState::Alive,
            generation: 0,
            traits: Traits::default(),
            genome: None,
            proteome: Vec::new(),
        };
        let dt = 0.01;
        let opening = cell.reserve;
        let mut earned = 0.0;
        for _ in 0..ticks {
            let pay = cfg.maintenance_power * dt * income;
            cell.reserve += pay;
            earned += pay;
            maintain(&mut cell, cfg, &grid, &mut heat, dt);
        }
        (cell, heat, opening + earned)
    }

    #[test]
    fn dormancy_lets_a_cell_live_on_a_trickle_that_would_starve_it_working() {
        // This is the whole mechanism in one assertion. A cell breaks even at
        // a fixed food concentration, so before this there was no refuge: when
        // the pond fell below break-even it fell below break-even for every
        // cell at once, and the population died as one. A cell that shuts down
        // pays `dormancy_power_fraction` of its bill, so it breaks even that
        // much lower down -- and a fifth of a living, which starves a working
        // cell, is four times over what a dormant one needs.
        let mut cfg = CellConfig::default();
        cfg.dormancy_power_fraction = 0.05;
        cfg.starvation_time = 100.0;

        let (cell, ..) = dormancy_bench(&cfg, 10.0, 0.2, 100_000);
        assert!(cell.is_alive(), "state {:?}", cell.state);
        assert_eq!(cell.damage, 0.0, "a cell inside its dormant means took damage");

        // The control is the same thousand seconds with dormancy disabled,
        // which is what the cell layer did before: pay in full, or accrue
        // damage against a reserve you do not have until it kills you.
        let mut off = cfg;
        off.dormancy_power_fraction = 1.0;
        off.dormancy_exit = 0.0;
        let (control, ..) = dormancy_bench(&off, 10.0, 0.2, 100_000);
        assert!(
            control.damage >= 1.0,
            "control should have starved on a fifth of a living, damage {}",
            control.damage
        );
    }

    #[test]
    fn dormancy_spends_from_the_same_ledger() {
        // A smaller bill is still a bill. Whatever leaves the reserve has to
        // arrive in the heat field, or dormancy is a hole in the energy audit
        // rather than a cheaper way to live.
        let mut cfg = CellConfig::default();
        cfg.dormancy_power_fraction = 0.1;
        let grid = Grid::new(2, 2, 2, 25.0e-6);
        let (cell, heat, taken_in) = dormancy_bench(&cfg, 1.0, 0.0, 5_000);
        assert_eq!(cell.state, CellState::Dormant);

        let spent = taken_in - cell.reserve;
        let landed = heat.energy(&grid);
        assert!(
            (spent - landed).abs() <= 1.0e-24,
            "reserve fell {spent} J, heat rose {landed} J"
        );
    }

    #[test]
    fn shutting_down_comes_before_taking_damage() {
        // Order matters. A cell must shut down while it can still pay, and
        // start accruing damage only once even the dormant bill is beyond it.
        // The reverse -- damage first, dormancy as a death rattle -- would buy
        // the population nothing.
        //
        // With nothing at all coming in this cell still dies, and that is
        // correct: dormancy is a cheaper way to live off a trickle, not a way
        // to live off nothing.
        let mut cfg = CellConfig::default();
        cfg.dormancy_power_fraction = 0.05;
        cfg.starvation_time = 1000.0;
        let grid = Grid::new(2, 2, 2, 25.0e-6);
        let mut heat = HeatField::new(&grid, 293.15);
        let mut cell = Cell {
            id: 0,
            parent: None,
            pos: [25.0e-6, 25.0e-6, 25.0e-6],
            radius: cfg.birth_radius,
            contents: vec![0.0; 4],
            reserve: cfg.maintenance_power,
            damage: 0.0,
            age: 0.0,
            state: CellState::Alive,
            generation: 0,
            traits: Traits::default(),
            genome: None,
            proteome: Vec::new(),
        };

        let mut shut_down_at = None;
        for tick in 0..1_000 {
            maintain(&mut cell, &cfg, &grid, &mut heat, 0.01);
            if shut_down_at.is_none() && cell.state == CellState::Dormant {
                shut_down_at = Some(tick);
                assert_eq!(cell.damage, 0.0, "took damage before shutting down");
            }
        }
        let shut_down_at = shut_down_at.expect("never shut down");
        assert!(cell.damage > 0.0, "never started starving after shutting down");
        // One second of upkeep in the bank, spent at the working rate.
        assert!(
            (95..=105).contains(&shut_down_at),
            "shut down at tick {shut_down_at}, expected about 100"
        );
    }

    #[test]
    fn waking_is_a_higher_bar_than_shutting_down_was() {
        // Without the gap a cell on the margin flickers in and out every tick
        // and pays something close to the working bill on average, which is no
        // dormancy at all.
        let mut cfg = CellConfig::default();
        cfg.dormancy_power_fraction = 0.05;
        cfg.dormancy_exit = 60.0;
        let (mut cell, mut heat, _) = dormancy_bench(&cfg, 0.5, 0.0, 100);
        assert_eq!(cell.state, CellState::Dormant);
        let grid = Grid::new(2, 2, 2, 25.0e-6);

        // A windfall short of the waking threshold leaves it shut down.
        cell.reserve = cfg.maintenance_power * 59.0;
        maintain(&mut cell, &cfg, &grid, &mut heat, 0.01);
        assert_eq!(cell.state, CellState::Dormant);

        // One that clears it puts the cell back to work.
        cell.reserve = cfg.maintenance_power * 61.0;
        maintain(&mut cell, &cfg, &grid, &mut heat, 0.01);
        assert_eq!(cell.state, CellState::Alive);
    }

    #[test]
    fn a_dormant_cell_does_not_divide() {
        // Reserve alone is not licence to split: a cell that is shut down is
        // shut down, and dividing on the way past the waking threshold would
        // let a famine produce offspring.
        let (mut cfg, grid, chem, rng, mut amounts, mut residual, mut heat, flow) = setup();
        cfg.membrane_scale = 0.0;
        cfg.metabolic_rate = 0.0;
        // Enough reserve to divide on, and still short of the waking
        // threshold, so the cell is genuinely dormant when the check runs.
        cfg.dormancy_exit = 10_000.0;
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        for cell in &mut p.cells {
            cell.state = CellState::Dormant;
            cell.reserve = cfg.division_reserve * 10.0;
        }
        assert!(
            cfg.division_reserve * 10.0 < cfg.maintenance_power * cfg.dormancy_exit as f64,
            "the test must keep the cell below its waking threshold"
        );
        let before = p.births;
        p.step(
            &cfg, &grid, &chem, &flow, &rng, 0, 0.01, 1, &mut amounts, &mut residual, &mut heat,
        );
        assert_eq!(p.births, before, "a dormant cell divided");
    }

    #[test]
    fn every_cell_gets_its_own_allotted_span() {
        // A shared lifespan is not a simplification, it is a synchroniser: a
        // cohort born together dies together, and a population sitting at
        // break-even has no surplus to breed replacements out of. The spread
        // is what turns a plateau from a freeze into a turnover.
        let mut cfg = CellConfig::default();
        cfg.maximum_age = 1000.0;
        cfg.lifespan_spread = 0.4;
        let rng = Counter::new(7);

        let spans: Vec<f32> = (0..500).map(|id| lifespan(&cfg, &rng, id)).collect();
        for &span in &spans {
            assert!(
                (600.0..=1400.0).contains(&span),
                "span {span} outside the configured spread"
            );
        }
        let lowest = spans.iter().cloned().fold(f32::INFINITY, f32::min);
        let highest = spans.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            highest - lowest > 500.0,
            "spans cover only {} s, which will not desynchronise a cohort",
            highest - lowest
        );

        // Settled at birth: asking twice must give the same answer, or a
        // cell's death would depend on when the question was asked.
        assert_eq!(spans[42], lifespan(&cfg, &rng, 42));

        // And zero spread is still the old fixed-age world, exactly.
        cfg.lifespan_spread = 0.0;
        assert_eq!(lifespan(&cfg, &rng, 42), cfg.maximum_age);
    }

    #[test]
    fn a_daughter_inherits_its_parent_with_a_kick() {
        // Inheritance with variation, which is the whole of the mechanism.
        // Identical daughters give back the frozen plateau; daughters
        // unrelated to their parents give a random walk with nothing for
        // selection to accumulate.
        let (mut cfg, grid, chem, rng, amounts, ..) = setup();
        cfg.trait_spread = 0.2;
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        let parent = &mut p.cells[0];
        parent.traits.uptake = 2.0;
        parent.reserve = cfg.division_reserve;

        let children: Vec<f32> = (0..64)
            .map(|tick| divide(parent, &cfg, &chem, &grid, &rng, tick, 1_000 + tick).traits.uptake)
            .collect();

        let mean = children.iter().sum::<f32>() / children.len() as f32;
        assert!(
            (mean - 2.0).abs() < 0.4,
            "daughters averaged {mean}, which is not their parent's 2.0"
        );
        assert!(
            children.iter().any(|&u| u != 2.0),
            "every daughter was an exact copy"
        );
        assert!(
            children.iter().all(|&u| u > 0.0),
            "a mutation pushed a trait through zero"
        );

        // And no spread is the old world exactly: clones, for ever.
        cfg.trait_spread = 0.0;
        assert_eq!(
            divide(parent, &cfg, &chem, &grid, &rng, 7, 2_000).traits.uptake,
            parent.traits.uptake
        );
    }

    #[test]
    fn founders_vary_and_their_traits_are_settled_at_birth() {
        let mut cfg = CellConfig {
            trait_spread: 0.2,
            ..Default::default()
        };
        let rng = Counter::new(11);

        let drawn: Vec<f32> = (0..400).map(|id| Traits::founder(&cfg, &rng, id).uptake).collect();
        let lowest = drawn.iter().cloned().fold(f32::INFINITY, f32::min);
        let highest = drawn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            highest / lowest > 2.0,
            "founders span only {lowest}..{highest}, which is one cell repeated"
        );
        assert_eq!(drawn[42], Traits::founder(&cfg, &rng, 42).uptake);

        cfg.trait_spread = 0.0;
        assert_eq!(Traits::founder(&cfg, &rng, 42), Traits::default());
    }

    #[test]
    fn a_lineage_cannot_drift_off_to_infinity() {
        // Log-normal kicks compound, so a long lineage under steady selection
        // would otherwise walk to values where a membrane stops meaning
        // anything physical.
        let (mut cfg, grid, chem, rng, amounts, ..) = setup();
        cfg.trait_spread = 0.5;
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        let cell = &mut p.cells[0];
        for tick in 0..500 {
            // Keep the luckiest of each pair, which is selection at its most
            // ruthless: straight up the gradient every generation.
            let child = divide(cell, &cfg, &chem, &grid, &rng, tick, 3_000 + tick);
            if child.traits.uptake > cell.traits.uptake {
                cell.traits = child.traits;
            }
        }
        assert_eq!(cell.traits.uptake, TRAIT_LIMIT);
    }

    #[test]
    fn a_hungrier_cell_earns_more_and_costs_more() {
        // The trade-off, isolated. Uptake that came free would ratchet up
        // until the population stripped the pond bare -- which is the failure
        // `membrane_scale = 6000` produced by hand -- so the machinery has to
        // be worth keeping rather than merely worth having.
        let (mut cfg, grid, chem, rng, mut amounts, mut residual, ..) = setup();
        cfg.trait_cost = 0.5;
        // Well below the membrane's own ceiling. `exchange` will never hand
        // over more than a fifth of a voxel in a step, and at the configured
        // scale a full pond puts both of these cells hard against that clamp,
        // where the trait cannot show because nothing is limited by it.
        cfg.membrane_scale = 1.0;
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        let table = CompoundTable::new(&chem);

        // Uptake is a permeability, so it is the *flux* it multiplies, not the
        // concentration the cell eventually reaches. Left alone long enough
        // both of these equilibrate with the same water and hold the same
        // amount; the difference that matters to a cell is how fast it gets
        // there, because metabolism is spending the inside the whole time.
        p.cells[0].traits.uptake = 0.5;
        p.cells[1].traits.uptake = 2.0;
        for cell in p.cells.iter_mut() {
            let e = flat(&chem, cell, None);
            exchange(cell, &cfg, &grid, &table, &e, &mut amounts, &mut residual, 0.01);
        }

        let taken: Vec<f64> = p.cells.iter().map(|c| c.contents.iter().sum()).collect();
        assert!(
            taken[1] > 3.0 * taken[0],
            "the hungrier cell took {:e} against {:e}",
            taken[1],
            taken[0]
        );
        assert!(
            p.cells[1].maintenance(&cfg) > p.cells[0].maintenance(&cfg),
            "and paid no more for it"
        );

        // Neutral at the ancestor's value, so `trait_cost` does not quietly
        // redefine `maintenance_power` for a population of clones.
        let mut clone = p.cells[0].clone();
        clone.traits = Traits::default();
        assert_eq!(clone.maintenance(&cfg), cfg.maintenance_power);
    }

    /// Two cells of the given uptakes, each in its own voxel, in water held at
    /// a fixed concentration for `seconds`. Returns each cell and how many
    /// ticks it spent shut down.
    ///
    /// A chemostat rather than a world. Every claim below is about what one
    /// water level does to two different cells, and in a closed voxel each of
    /// them would quietly eat its way down to a different one.
    fn chemostat(
        uptakes: [f32; 2],
        trait_cost: f64,
        bill: f64,
        opening: f64,
        seconds: f64,
    ) -> Vec<(Cell, u64)> {
        let (mut cfg, grid, chem, rng, mut amounts, mut residual, mut heat, flow) = setup();
        cfg.initial_count = 2;
        cfg.trait_cost = trait_cost;
        cfg.maintenance_power = bill;
        // Never divide, never age out: this is about one cell's books.
        cfg.division_reserve = 1.0e30;
        cfg.maximum_age = 1.0e9;

        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        for (cell, uptake) in p.cells.iter_mut().zip(uptakes) {
            cell.traits.uptake = uptake;
            cell.reserve = opening;
        }
        p.cells[0].pos = [1.5 * grid.dx, 1.5 * grid.dx, 1.5 * grid.dx];
        p.cells[1].pos = [5.5 * grid.dx, 5.5 * grid.dx, 1.5 * grid.dx];

        let mut dormant = vec![0u64; 2];
        for tick in 0..(seconds / 0.01) as u64 {
            for c in 0..chem.n_compounds() {
                amounts.plane_mut(c).fill(2.0e7);
            }
            p.step(
                &cfg,
                &grid,
                &chem,
                &flow,
                &rng,
                tick,
                0.01,
                20,
                &mut amounts,
                &mut residual,
                &mut heat,
            );
            for (i, cell) in p.cells.iter().enumerate() {
                dormant[i] += cell.is_dormant() as u64;
            }
        }
        p.cells.iter().cloned().zip(dormant).collect()
    }

    #[test]
    fn variation_gives_the_best_cells_a_surplus_where_the_worst_are_starving() {
        // The reason traits exist. A population of clones breaks even at one
        // food concentration, so when the pond falls below it every cell falls
        // below it at once: the plateau freezes with no surplus anywhere to
        // divide on, and ten runs of the gate reported `births` exactly equal
        // to the peak population. With variation the same water is a living
        // for some cells and not for others, which is what a carrying capacity
        // is supposed to feel like from the inside.
        const UPTAKES: [f32; 2] = [0.6, 2.5];
        const COST: f64 = 0.5;
        let seconds = 300.0;

        // What each of them can earn out of this water with no bill to pay.
        // Measured rather than derived: the point is that there is a band of
        // upkeeps between the two, and the band has to be found before it can
        // be aimed at.
        let free = chemostat(UPTAKES, COST, 0.0, 0.0, seconds);
        let income: Vec<f64> = free.iter().map(|(c, _)| c.reserve / seconds).collect();
        assert!(
            income[1] > 2.0 * income[0],
            "the two cells earn {:e} and {:e} W, which is not a band to aim at",
            income[0],
            income[1]
        );

        // A bill inside that band: a living for one of them and not the other.
        // They open with ten seconds of it banked, which is enough to start
        // the run awake and not so much that it ends before the poorer cell
        // has spent it.
        let bill = (income[0] * income[1]).sqrt();
        let opening = bill * 10.0;
        let out = chemostat(UPTAKES, COST, bill, opening, seconds);
        let (poor, poor_dormant) = &out[0];
        let (rich, rich_dormant) = &out[1];

        assert!(
            rich.reserve > opening,
            "the better-equipped cell went backwards: {:e} J from {:e} J",
            rich.reserve,
            opening
        );
        assert!(
            poor.reserve < opening,
            "the poorer cell paid its way after all: {:e} J from {:e} J",
            poor.reserve,
            opening
        );
        assert_eq!(*rich_dormant, 0, "the cell with a surplus shut down anyway");
        assert!(*poor_dormant > 0, "the cell below its line never shut down");
    }

    #[test]
    fn a_better_cells_surplus_is_the_bill_times_what_is_fixed_about_it() {
        // The arithmetic the whole mechanism turns on, checked against the
        // cell layer rather than against itself.
        //
        // A population eats down until its marginal cell breaks even, which
        // fixes the water at `k C e = m (1 - c + c u_m) / u_m`. A cell a
        // fraction `d` better off than that one then earns
        //
        //     surplus = m (1 - c + c u_m)(1 + d) - m (1 - c) - m c u_m (1 + d)
        //             = m (1 - c) d
        //
        // and everything but `d` and the two dials cancels. It matters because
        // it says what a plateau needs in order to turn over: the best cell
        // has to fund a whole `division_reserve` out of `m (1 - c) d`, inside
        // one lifetime. At `trait_cost = 1` the surplus is zero for every `d`,
        // and the population is back to being clones as far as its books are
        // concerned.
        //
        // The closed form is an *upper bound*, and this test is also where
        // that shows. It assumes income is proportional to uptake, which holds
        // only while the membrane is the bottleneck; once a cell can take up
        // faster than it can metabolise, more transporter buys less and less.
        // A better cell's real advantage is therefore smaller than `d`, so a
        // plateau needs more variation to turn over than the formula asks for,
        // not less.
        const COST: f64 = 0.5;
        const ADVANTAGE: f64 = 0.5;
        let uptakes = [1.0, 1.0 + ADVANTAGE as f32];
        let seconds = 300.0;

        // Put the marginal cell exactly at break-even by measuring what it
        // earns and handing it a bill of the same size. At `u_m = 1` the bill
        // is `m` whatever the cost split is, so `m` is that income.
        let free = chemostat(uptakes, COST, 0.0, 0.0, seconds);
        let income: Vec<f64> = free.iter().map(|(c, _)| c.reserve / seconds).collect();
        let m = income[0];

        let opening = m * 100.0;
        let out = chemostat(uptakes, COST, m, opening, seconds);
        let measured = (out[1].0.reserve - opening) / seconds;

        // What the cell layer's own books say it should have banked: what it
        // earned, less what a cell carrying that much machinery is charged.
        let booked = income[1] - m * (1.0 - COST + COST * uptakes[1] as f64);
        assert!(
            (measured - booked).abs() < 0.1 * booked.abs(),
            "banked {measured:e} W against {booked:e} W of income less upkeep"
        );

        // And the closed form above it, which the sublinearity makes generous.
        let ceiling = m * (1.0 - COST) * ADVANTAGE;
        assert!(
            measured > 0.0 && measured <= ceiling,
            "surplus {measured:e} W is not inside (0, {ceiling:e}] -- the formula \
             is supposed to bound it from above"
        );

        // The marginal cell, by construction, banks nothing.
        let marginal = (out[0].0.reserve - opening) / seconds;
        assert!(
            marginal.abs() < 0.1 * ceiling,
            "the break-even cell moved by {marginal:e} W"
        );
    }

    #[test]
    fn a_populations_traits_are_reported_from_the_living() {
        let (mut cfg, grid, chem, rng, amounts, ..) = setup();
        cfg.trait_spread = 0.0;
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        assert_eq!(p.trait_summary().mean_uptake, 1.0);
        assert_eq!(p.trait_summary().uptake_spread, 0.0);

        p.cells[0].traits.uptake = 3.0;
        assert_eq!(p.trait_summary().mean_uptake, 2.0);
        assert_eq!(p.trait_summary().uptake_spread, 1.0);

        // A corpse has no traits to select on. Counting it would let a die-off
        // move the mean on its own and read as evolution.
        p.cells[0].state = CellState::Decomposing;
        assert_eq!(p.trait_summary().mean_uptake, 1.0);
    }

    #[test]
    fn dormant_cells_count_as_living() {
        // They occupy the population and the safety cap, and they are not
        // corpses. `decomposing` is derived from `alive`, so getting this
        // wrong reports a sleeping pond as a mass grave.
        let (cfg, grid, chem, rng, amounts, ..) = setup();
        let mut p = Population::seed(&cfg, &chem, &grid, &rng, &abundance(&chem, &amounts));
        p.cells[0].state = CellState::Dormant;
        assert_eq!(p.alive(), 2);
        assert_eq!(p.dormant(), 1);
        assert_eq!(p.decomposing(), 0);
    }
}
