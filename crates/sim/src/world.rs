//! The world: every layer, assembled and stepped.
//!
//! One tick, in order:
//!
//! 1. **Light** marches down the columns, depositing absorbed energy per band.
//! 2. **Chemistry** runs in every voxel, consuming that light where a
//!    photochemical reaction can use it and releasing the rest as heat.
//! 3. **Diffusion** and **advection** move compounds, in conservative form.
//! 4. **Heat** takes vent input, diffuses, and loses energy at the surface.
//! 5. **Vents** inject reduced compounds at the floor.
//! 6. **Cells** move, exchange compounds, metabolise, maintain themselves,
//!    divide, die, and decompose.
//! 7. The **audit** checks that all of that added up.
//!
//! Any fixed order would do; what matters is that it is fixed, and that every
//! boundary crossing is recorded in the ledger as it happens.
//!
//! Chemistry is the one phase that wants a voxel-major layout while everything
//! else wants compound-major. Rather than compromise, the step transposes into
//! a scratch buffer, runs the reactions there in parallel over voxels, and
//! transposes back. Two copies of the field per tick is far cheaper than
//! either phase working against its grain.

use hadean_cell::Population;
use hadean_chem::chemistry::N_BANDS;
use hadean_chem::kinetics::{Network, VoxelStep};
use hadean_chem::{generate, Chemistry};
use hadean_core::hash::{HashState, StateHasher};
use hadean_core::rng::Purpose;
use hadean_core::units::Joules;
use hadean_core::{Clock, Counter, Grid, Schedule};
use hadean_fields::flow::convection_roll;
use hadean_fields::heat::{self, HeatField};
use hadean_fields::light::{LightField, LightSolver};
use hadean_fields::scalar::ChemField;
use hadean_fields::transport::{advect, diffuse, FaceVelocity};
use rayon::prelude::*;

use crate::audit::{Audit, AuditReport};
use crate::config::WorldConfig;

/// A complete simulated world.
pub struct World {
    pub config: WorldConfig,
    pub grid: Grid,
    pub clock: Clock,
    pub rng: Counter,
    pub chem: Chemistry,
    /// The chemistry flattened for the inner loop, built once.
    pub network: Network,
    pub amounts: ChemField,
    /// Rounding the reaction and transport passes have carried forward, one
    /// per voxel per compound. Part of the world's conserved mass; see
    /// [`hadean_fields::transport`].
    pub residual: ChemField,
    pub heat: HeatField,
    pub light: LightField,
    pub solver: LightSolver,
    pub flow: FaceVelocity,
    pub audit: Audit,
    pub cells: Population,
    /// Voxel-major transpose buffer for the chemistry phase.
    transpose: Vec<f32>,
    /// Voxel-major transpose of the chemical rounding residuals.
    residual_transpose: Vec<f32>,
    /// Per-voxel results of the chemistry phase.
    outcomes: Vec<VoxelStep>,
    /// Double buffer for transport.
    scratch: Vec<f32>,
}

impl World {
    /// Build a world from a configuration. Deterministic in the config alone.
    pub fn new(config: WorldConfig) -> anyhow::Result<Self> {
        config
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid configuration: {e}"))?;

        let grid = config.grid();
        let chem = generate(config.seed, config.chemistry);
        chem.verify()
            .map_err(|e| anyhow::anyhow!("generated chemistry is invalid: {e}"))?;

        let mut amounts = ChemField::new(&grid, chem.n_compounds());
        let heat = HeatField::new(&grid, config.heat.ambient);
        let rng = Counter::new(config.seed);
        seed_soup(&config, &grid, &chem, &rng, &mut amounts);

        // Checked here rather than where it is used: `introduce` runs at
        // t = seed_delay, and a config that names a compound this chemistry
        // does not have should not cost two and a half minutes of world time
        // before it says so.
        if let Some(spec) = &config.cells.metabolism {
            hadean_cell::named_metabolism(&chem, spec)
                .map_err(|e| anyhow::anyhow!("cells.metabolism: {e}"))?;
        }

        let solver = LightSolver::new(&chem, config.light);
        let light = LightField::new(&grid);
        let flow = convection_roll(&grid, &config.flow);
        // The population is created empty and introduced on schedule, so a
        // world whose ancestors arrive later still audits its own lifeless
        // chemistry from tick zero.
        let mut cells = Population::empty();
        if seed_tick(&config) == 0 {
            let abundance = abundance_of(&chem, &amounts);
            cells.introduce(&config.cells, &chem, &grid, &rng, &abundance);
        }
        let residual = ChemField::new(&grid, amounts.n_compounds);
        let audit = Audit::new(
            config.heat.ambient,
            &grid,
            &chem,
            &amounts,
            &residual,
            &heat,
            &cells,
        );

        Ok(Self {
            clock: Clock::new(config.dt),
            grid,
            rng,
            network: Network::new(&chem),
            chem,
            residual,
            amounts,
            heat,
            light,
            solver,
            flow,
            audit,
            cells,
            transpose: Vec::new(),
            residual_transpose: Vec::new(),
            outcomes: Vec::new(),
            config,
            scratch: Vec::new(),
        })
    }

    pub fn tick(&self) -> u64 {
        self.clock.tick
    }

    /// Simulated seconds elapsed.
    pub fn elapsed(&self) -> f64 {
        self.clock.elapsed()
    }

    /// Advance one tick.
    pub fn step(&mut self) {
        let dt = self.config.dt;

        let sun = self.solver.propagate(
            &self.grid,
            &self.chem,
            &self.amounts,
            self.clock.elapsed(),
            dt,
            &mut self.light,
        );
        // Book what was deposited, not what arrived. The chemistry consumes
        // the per-voxel f32 figures, so those are what the world actually
        // received.
        self.audit.ledger.light_in += sun.absorbed;

        self.react(dt);
        self.transport(dt);

        let exchange = heat::step(
            &self.grid,
            &mut self.heat,
            &self.config.heat,
            dt,
            &mut self.scratch,
        );
        self.audit.ledger.vent_heat_in += exchange.vented;
        self.audit.ledger.radiated_out += exchange.radiated;

        self.vent_matter(dt);

        if self.clock.tick == seed_tick(&self.config) {
            // What the ancestors will eat is decided here, from the pond as
            // the lifeless world has actually left it.
            let abundance = abundance_of(&self.chem, &self.amounts);
            self.cells.introduce(
                &self.config.cells,
                &self.chem,
                &self.grid,
                &self.rng,
                &abundance,
            );
        }

        // What a receptor can see of the world beyond the cell layer's own
        // fields. Light only, so far.
        let sensorium = hadean_cell::Sensorium {
            light: Some(&self.light),
            full_sun: self.config.light.total_irradiance(),
        };
        self.cells.step(
            &self.config.cells,
            &self.grid,
            &self.chem,
            &self.flow,
            &self.rng,
            self.clock.tick,
            dt,
            &self.config.schedule,
            &sensorium,
            &mut self.amounts,
            &mut self.residual,
            &mut self.heat,
        );

        self.clock.advance();
    }

    /// Run the reaction network in every voxel.
    fn react(&mut self, dt: f64) {
        let n_voxels = self.grid.len();
        let n_compounds = self.chem.n_compounds();

        self.amounts.tile(0, n_voxels, &mut self.transpose);
        self.residual
            .tile(0, n_voxels, &mut self.residual_transpose);

        let network = &self.network;
        let light = &self.light;
        let heat = &self.heat;

        // Every voxel is independent and draws no randomness, so this is
        // deterministic however rayon splits it. `collect_into_vec` preserves
        // index order, which the sequential heat application below relies on.
        self.transpose
            .par_chunks_mut(n_compounds)
            .zip(self.residual_transpose.par_chunks_mut(n_compounds))
            .enumerate()
            .map(|(voxel, (amounts, residual))| {
                let absorbed = light.absorbed_bands(voxel);
                network.step_voxel(
                    amounts,
                    residual,
                    heat.temperature(voxel),
                    dt,
                    &absorbed,
                    &[],
                )
            })
            .collect_into_vec(&mut self.outcomes);

        self.amounts.untile(0, n_voxels, &self.transpose);
        self.residual.untile(0, n_voxels, &self.residual_transpose);

        for (voxel, outcome) in self.outcomes.iter().enumerate() {
            if outcome.heat != 0.0 {
                self.heat.deposit(&self.grid, voxel, outcome.heat);
            }
        }
    }

    /// Diffuse and advect every compound.
    ///
    /// Parallelised over compounds rather than over voxels. Each compound's
    /// plane is contiguous and independent, so one compound per thread gives
    /// every worker a large, cache-local job -- much better than splitting
    /// each of several dozen sweeps across every core and paying the dispatch
    /// each time.
    fn transport(&mut self, dt: f64) {
        let moving = self.flow.max_speed() > 0.0;
        let grid = &self.grid;
        let flow = &self.flow;
        let compounds = &self.chem.compounds;
        let n_voxels = self.amounts.n_voxels;

        self.amounts
            .data
            .par_chunks_mut(n_voxels)
            .zip(self.residual.data.par_chunks_mut(n_voxels))
            .enumerate()
            .for_each_init(Vec::new, |scratch, (c, (plane, residual))| {
                // Skip a compound that is not present anywhere. Early in a run
                // most of the pool is empty, and this is most of the saving.
                if plane.iter().all(|&x| x == 0.0) && residual.iter().all(|&x| x == 0.0) {
                    return;
                }
                diffuse(grid, plane, residual, compounds[c].diffusion, dt, scratch);
                if moving {
                    advect(grid, plane, residual, flow, dt, scratch);
                }
            });
    }

    /// Inject reduced compounds at the vents.
    ///
    /// This is the world's second energy tap. Vent fuel is chosen by the
    /// chemistry generator as the least stable small compounds it produced --
    /// the ones that can still release energy by reacting with what is already
    /// dissolved.
    fn vent_matter(&mut self, dt: f64) {
        let cfg = self.config.vents;
        if cfg.fuel_rate <= 0.0 || cfg.fuel_species == 0 {
            return;
        }
        let species = cfg.fuel_species.min(self.chem.vent_fuel.len());
        if species == 0 {
            return;
        }
        let per_step = (cfg.fuel_rate as f64 * dt) as f32;
        let vents = heat::vent_voxels(&self.grid, self.config.heat.vents.max(1));
        for &voxel in &vents {
            for k in 0..species {
                let fuel = self.chem.vent_fuel[k];
                let compound = fuel as usize;
                let before = self.amounts.get(compound, voxel);
                let after = before + per_step;
                self.amounts.set(compound, voxel, after);
                // Book the representable field change, not the requested
                // injection. At large values an f32 addition can round by
                // thousands of particles, and that discrepancy otherwise
                // accumulates once per vent and tick.
                let actual = after as f64 - before as f64;
                self.audit.record_injection(&self.chem, fuel, actual);
            }
        }
    }

    /// Restart the audit from the world as it stands now.
    ///
    /// The audit measures the world against a baseline plus a ledger of what
    /// has crossed the boundary. Anything that reaches into the fields from
    /// outside that ledger -- a test laying out food, a tool injecting a blob
    /// of compound to watch it diffuse -- is by construction a violation, and
    /// the audit is right to say so. This is how such a caller says "that was
    /// me": it makes the new state the thing to conserve from here on.
    ///
    /// Not for use inside a run. A world that re-baselines whenever it drifts
    /// has no audit at all.
    pub fn rebaseline_audit(&mut self) {
        self.audit = Audit::new(
            self.config.heat.ambient,
            &self.grid,
            &self.chem,
            &self.amounts,
            &self.residual,
            &self.heat,
            &self.cells,
        );
    }

    /// Take an audit reading.
    pub fn audit_now(&mut self) -> AuditReport {
        self.audit.report(
            self.clock.tick,
            &self.grid,
            &self.chem,
            &self.amounts,
            &self.residual,
            &self.heat,
            &self.cells,
        )
    }

    /// Is an audit due this tick?
    pub fn audit_due(&self) -> bool {
        Schedule::due(self.clock.tick, self.config.schedule.audit)
    }

    /// Run `ticks` steps, auditing on schedule. Returns the last reading.
    pub fn run(&mut self, ticks: u64) -> Option<AuditReport> {
        let mut last = None;
        for _ in 0..ticks {
            self.step();
            if self.audit_due() {
                last = Some(self.audit_now());
            }
        }
        last.or_else(|| Some(self.audit_now()))
    }

    /// Digest of the full simulation state. Two runs of the same config for
    /// the same number of ticks must produce the same value.
    pub fn state_digest(&self) -> u64 {
        let mut h = StateHasher::new();
        h.u64(self.clock.tick);
        h.u64(self.config.digest());
        self.chem.hash_state(&mut h);
        self.amounts.hash_state(&mut h);
        self.residual.hash_state(&mut h);
        self.heat.hash_state(&mut h);
        self.cells.hash_state(&mut h);
        h.f64(self.audit.ledger.light_in);
        h.f64(self.audit.ledger.vent_heat_in);
        h.f64(self.audit.ledger.vent_chemical_in);
        h.f64(self.audit.ledger.radiated_out);
        h.finish()
    }

    /// Total light energy the world has taken in, J.
    pub fn light_absorbed(&self) -> Joules {
        self.light.total_absorbed()
    }

    /// Mean temperature at each depth, for reporting the vertical gradient.
    pub fn temperature_profile(&self) -> Vec<f64> {
        let layer = self.grid.layer();
        (0..self.grid.nz as usize)
            .map(|z| {
                let base = z * layer;
                (0..layer)
                    .map(|i| self.heat.temperature(base + i) as f64)
                    .sum::<f64>()
                    / layer as f64
            })
            .collect()
    }

    /// Mean irradiance at each depth, W/m^2.
    pub fn light_profile(&self) -> Vec<f64> {
        let layer = self.grid.layer();
        (0..self.grid.nz as usize)
            .map(|z| {
                let base = z * layer;
                (0..layer)
                    .map(|i| self.light.brightness(base + i) as f64)
                    .sum::<f64>()
                    / layer as f64
            })
            .collect()
    }

    /// Compounds ranked by abundance, as `(name, particles)`.
    pub fn abundances(&self) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = (0..self.chem.n_compounds())
            .map(|c| {
                (
                    self.chem.compounds[c].name.clone(),
                    self.amounts.total_of(c),
                )
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// The compounds the *ancestral* metabolism eats, with what is dissolved
    /// in the pond right now, as `(name, particles, stoichiometric
    /// coefficient)`.
    ///
    /// This is the population's food supply. Reading it alongside the cell
    /// count is what turns "the population fell" into "the population ate
    /// itself out of house and home".
    ///
    /// In a genome world it is the ancestor's food, not necessarily the
    /// population's: a lineage that has evolved a different enzyme is eating
    /// something this does not report. That is a limitation of the column and
    /// not a bug in it -- a population with several diets does not have "a"
    /// food supply, and `Population::diet` is what says who is eating what.
    /// The two read together are the interesting picture: the ancestral
    /// substrate collapsing while the population holds up means something in
    /// the pond has moved off it.
    pub fn metabolic_substrates(&self) -> Vec<(String, f64, u8)> {
        let Some(id) = self.cells.metabolic_reaction else {
            return Vec::new();
        };
        self.chem
            .reaction(id)
            .reactants
            .iter()
            .map(|&(c, n)| {
                (
                    self.chem.compound(c).name.clone(),
                    self.amounts.total_of(c as usize) + self.cells_holding(c),
                    n,
                )
            })
            .collect()
    }

    /// How much of one compound the population is holding inside membranes.
    fn cells_holding(&self, compound: u16) -> f64 {
        self.cells
            .cells
            .iter()
            .map(|c| c.contents[compound as usize])
            .sum()
    }

    /// Particles of the scarcest metabolic substrate, divided by how many of
    /// it each turn of the reaction consumes.
    ///
    /// One number for "how many more meals are left in the pond". Infinite
    /// when there is no metabolism, so a cell-free world never looks starved.
    pub fn limiting_substrate(&self) -> f64 {
        self.metabolic_substrates()
            .iter()
            .map(|&(_, amount, n)| amount / n.max(1) as f64)
            .fold(f64::INFINITY, f64::min)
    }

    /// Particles of every compound in the pond, indexed by compound id.
    ///
    /// The same vector `introduce` chooses a metabolism from, which is why it
    /// is public: a pre-run check that wants to say what that choice will cost
    /// has to read the pond the same way the choice does.
    pub fn abundance(&self) -> Vec<f64> {
        abundance_of(&self.chem, &self.amounts)
    }

    /// Take every particle of one compound out of the water, and say how many
    /// that was.
    ///
    /// This is a measuring instrument, not a mechanism: nothing in a normal
    /// run calls it. It exists because the question "what can this pond feed a
    /// population" is a *rate*, and the only honest way to read a rate off a
    /// world like this one is to take the compound away and watch what the
    /// chemistry does about it. Held at zero, the network runs its production
    /// of that compound as fast as it can, and what has to be removed each
    /// second to keep it there is the largest harvest the pond will ever
    /// support. See `hadean supply`.
    ///
    /// The removal crosses the world boundary, so it is booked in the same
    /// ledger a vent injection is, with the sign the other way round -- the
    /// audit stays exact through a probe and a failing one still means what it
    /// meant. Rounding residuals are left alone: they are already conserved
    /// and they are parts of a particle.
    pub fn harvest(&mut self, compound: u16) -> f64 {
        let c = compound as usize;
        if c >= self.amounts.n_compounds {
            return 0.0;
        }
        let plane = self.amounts.plane_mut(c);
        let mut taken = 0.0;
        for v in plane.iter_mut() {
            taken += *v as f64;
            *v = 0.0;
        }
        self.audit.record_injection(&self.chem, compound, -taken);
        taken
    }

    /// Run one reaction forward as hard as the water will allow, leaving every
    /// atom in the pond, and say how many turnovers that was.
    ///
    /// The mirror of [`World::harvest`], and the difference between them is
    /// the whole question. `harvest` takes a compound *out* of the world, so
    /// it measures what the pond can export -- which bounds a probe and bounds
    /// nothing that lives here, because a cell exports nothing. A cell turns
    /// substrates into products and leaves them in the water. What actually
    /// bounds a population, then, is whether the chemistry and the light can
    /// drive those products back round to the substrate, and holding the
    /// substrate at zero *with the products returned* is the way to ask.
    ///
    /// It is an upper bound and not a simulation of a population: a real cell
    /// is limited by its own membrane and its own enzymes as well as by the
    /// water, and this is limited only by the water. That is the point. A
    /// living no infinitely capable cell could make here is a living no cell
    /// can make here.
    ///
    /// Matter is conserved exactly -- the same atoms leave the reactants and
    /// arrive in the products, in the same voxel -- so nothing is booked in
    /// the element ledger, unlike a harvest. The enthalpy released goes into
    /// the water as heat, which is where it goes when the reaction runs on its
    /// own; a cell would keep `capture_efficiency` of it instead, and that
    /// difference belongs to the cell layer rather than to the supply.
    ///
    /// The heat is accumulated from the deltas [`ChemField::settle`] actually
    /// applied, not from the intended extent, so a clamp or an `f32` rounding
    /// is absorbed by the heat term and the audit stays flat. It is
    /// deliberately *not* taken as a difference of the voxel's before and
    /// after chemical totals, which is how the reaction step does it: that
    /// works there because a step moves a visible fraction of the voxel, and
    /// it fails here because the interesting reactions in this world limit on
    /// a trace substrate. Differencing 1e-8 particles against the 1e12 sitting
    /// beside them in the same sum gives zero in `f64`, and the joules would
    /// be silently dropped rather than deposited.
    pub fn turn_over(&mut self, reaction: u32) -> Turnover {
        let Some(r) = self.chem.reactions.get(reaction as usize) else {
            return Turnover::default();
        };
        // Cloned so the borrow of the chemistry ends before the fields move.
        let reactants = r.reactants.clone();
        let products = r.products.clone();
        let h_f = |c: u16| self.chem.compound(c).h_f;
        let enthalpy: Vec<f64> = reactants
            .iter()
            .chain(products.iter())
            .map(|&(c, _)| h_f(c))
            .collect();

        let mut out = Turnover::default();
        for voxel in 0..self.grid.len() {
            // What this voxel's water can pay for. Amount plus residual,
            // because the residual is material the field owns and cannot yet
            // represent -- the same reason the reaction step carries it.
            let mut extent = f64::INFINITY;
            for &(c, n) in &reactants {
                let have = self.amounts.get(c as usize, voxel) as f64
                    + self.residual.get(c as usize, voxel) as f64;
                extent = extent.min(have.max(0.0) / n as f64);
            }
            // Shrunk by a couple of ulps before anything is applied. `extent`
            // is `have / n` minimised over the reactants, and `have / n * n`
            // is not always `have` in `f64` -- for n = 3 it can land an ulp
            // above, which would settle that amount to a small negative and
            // leave a compound count below zero in the field.
            //
            // The shrink is done here rather than by clamping each reactant as
            // it is consumed, and the difference matters: clamping a reactant
            // while still adding the products at the full extent would create
            // atoms. Mass balance in this project is structural, and a probe
            // is not the place to start making it approximate.
            extent *= 1.0 - 2.0 * f64::EPSILON;
            // Both halves are load-bearing. A voxel with nothing in it gives
            // zero and there is no work to do; a reaction with no reactants at
            // all -- which the generator cannot produce, but this does not
            // depend on that -- would leave `extent` at the infinity it starts
            // from and consume nothing while creating products without end.
            if !extent.is_finite() || extent <= 0.0 {
                continue;
            }
            let mut chemical = 0.0;
            for (i, &(c, n)) in reactants.iter().enumerate() {
                let applied = self.amounts.settle(
                    &mut self.residual,
                    c as usize,
                    voxel,
                    -extent * n as f64,
                );
                chemical += applied * enthalpy[i];
            }
            for (i, &(c, n)) in products.iter().enumerate() {
                let applied =
                    self.amounts
                        .settle(&mut self.residual, c as usize, voxel, extent * n as f64);
                chemical += applied * enthalpy[reactants.len() + i];
            }
            if chemical != 0.0 {
                self.heat.deposit(&self.grid, voxel, -chemical);
            }
            out.turnovers += extent;
            out.released -= chemical;
        }
        out
    }
}

/// What one sweep of [`World::turn_over`] took out of the water.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Turnover {
    /// Extent, in reaction turnovers.
    pub turnovers: f64,
    /// Chemical energy the water gave up, J. Positive for an exergonic
    /// reaction. Measured from the amounts that were actually moved rather
    /// than from `turnovers * -dh`, so it is what the pond paid and not what
    /// it was asked for.
    pub released: Joules,
}

/// Particles of every compound in the world.
fn abundance_of(chem: &Chemistry, amounts: &ChemField) -> Vec<f64> {
    (0..chem.n_compounds())
        .map(|c| amounts.total_of(c))
        .collect()
}

/// The tick the founding cohort appears on.
///
/// Derived from the configuration rather than stored, so a snapshot taken
/// before the ancestors arrive still introduces them at the right moment.
fn seed_tick(config: &WorldConfig) -> u64 {
    (config.cells.seed_delay as f64 / config.dt).round() as u64
}

/// Fill the world with its starting soup.
///
/// Water everywhere, a little of each primordial compound, and multiplicative
/// noise. The noise matters more than it looks: a perfectly uniform world has
/// no niches, and the design is explicit that a uniform environment is one of
/// the main reasons evolution stalls.
fn seed_soup(
    config: &WorldConfig,
    grid: &Grid,
    chem: &Chemistry,
    rng: &Counter,
    amounts: &mut ChemField,
) {
    const PRIMORDIALS: [&str; 8] = ["H2", "O2", "N2", "H2S", "CO2", "H3N", "CH4", "OM2"];

    let jitter = |voxel: usize, stream: u64| -> f32 {
        if config.initial.noise <= 0.0 {
            return 1.0;
        }
        let u = rng.unit(0, voxel as u64, Purpose::Placement, stream);
        1.0 + config.initial.noise * (2.0 * u - 1.0)
    };

    for voxel in 0..grid.len() {
        amounts.set(
            chem.water as usize,
            voxel,
            config.initial.water * jitter(voxel, 0),
        );
    }
    for (k, name) in PRIMORDIALS.iter().enumerate() {
        let Some(id) = chem.by_name(name) else {
            continue;
        };
        for voxel in 0..grid.len() {
            amounts.set(
                id as usize,
                voxel,
                config.initial.primordial * jitter(voxel, k as u64 + 1),
            );
        }
    }
}

/// Bands the light field carries. Re-exported so callers do not have to reach
/// into the chemistry crate for it.
pub const LIGHT_BANDS: usize = N_BANDS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GridConfig;
    use hadean_chem::chemistry::Reaction;

    fn small() -> WorldConfig {
        WorldConfig {
            seed: 7,
            grid: GridConfig {
                nx: 10,
                ny: 8,
                nz: 8,
                dx: 25.0e-6,
            },
            ..Default::default()
        }
    }

    #[test]
    fn a_world_builds_and_steps() {
        let mut w = World::new(small()).expect("builds");
        assert_eq!(w.tick(), 0);
        w.step();
        assert_eq!(w.tick(), 1);
        assert!(w.amounts.data.iter().all(|&x| x.is_finite() && x >= 0.0));
    }

    #[test]
    fn the_starting_soup_is_seeded_and_varied() {
        let w = World::new(small()).expect("builds");
        let water = w.amounts.total_of(w.chem.water as usize);
        assert!(water > 0.0);
        // Noise must actually produce variation.
        let plane = w.amounts.plane(w.chem.water as usize);
        let (lo, hi) = plane
            .iter()
            .fold((f32::MAX, 0.0f32), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(hi > lo, "soup is perfectly uniform");
    }

    #[test]
    fn light_reaches_the_world_and_falls_off_with_depth() {
        let mut w = World::new(WorldConfig {
            light: hadean_fields::light::LightConfig {
                day_length: 0.0,
                ..Default::default()
            },
            ..small()
        })
        .expect("builds");
        w.step();
        let profile = w.light_profile();
        assert!(profile[0] > 0.0);
        assert!(
            profile[0] > *profile.last().unwrap(),
            "no light gradient: {profile:?}"
        );
    }

    #[test]
    fn a_probe_stays_inside_the_audit() {
        // `harvest` takes matter out of the world, which is exactly the kind
        // of thing that quietly breaks a conservation gate. It is booked in
        // the same ledger a vent injection is, with the sign the other way
        // round, so a probe must leave the audit as flat as it found it -- and
        // a failing audit during a measurement is a build failure like any
        // other.
        let mut w = World::new(small()).expect("builds");
        w.run(200);
        let before = w.audit_now();
        assert!(before.passes(1.0e-6), "drifting before the probe");

        let compound = w.chem.vent_fuel[0];
        let mut taken = 0.0;
        for _ in 0..300 {
            w.step();
            taken += w.harvest(compound);
        }
        assert!(taken > 0.0, "held a vent fuel at zero and nothing came back");
        assert!(
            w.amounts.total_of(compound as usize) == 0.0,
            "the probe left some behind"
        );

        let after = w.audit_now();
        assert!(
            after.passes(1.0e-6),
            "the probe broke the audit: energy {:e}, mass {:e}",
            after.relative,
            after.mass_drift
        );
    }

    #[test]
    fn a_probe_recovers_the_flux_the_config_already_knows() {
        // The calibration. A vent fuel's resupply is the one rate in this
        // world that is not in doubt -- `vents` times `fuel_rate`, straight
        // out of the config -- so a probe that measures anything else is
        // measuring itself. Run blind over the whole chemistry on `gate.toml`
        // this recovered 1.206e10/s against a known 1.2e10/s.
        //
        // The tolerance is wide on purpose: the chemistry makes and consumes
        // the fuel too, and the point of the check is that the instrument is
        // reading the right order of magnitude of the right quantity, not that
        // the pond is inert.
        let cfg = WorldConfig {
            // Dark, so the photochemistry is not also making the fuel and
            // the only source left is the one the config states.
            light: hadean_fields::light::LightConfig {
                irradiance: [0.0; hadean_chem::chemistry::N_BANDS],
                ..Default::default()
            },
            ..small()
        };
        let expected = cfg.vents.fuel_rate as f64 * cfg.heat.vents.max(1) as f64;
        let dt = cfg.dt;
        let mut w = World::new(cfg).expect("builds");
        w.run(200);

        let compound = w.chem.vent_fuel[0];
        w.harvest(compound);
        let ticks = 2000;
        let mut taken = 0.0;
        for _ in 0..ticks {
            w.step();
            taken += w.harvest(compound);
        }
        let measured = taken / (ticks as f64 * dt);
        let ratio = measured / expected;
        assert!(
            (0.5..2.0).contains(&ratio),
            "probe measured {measured:.3e}/s against a vent flux of {expected:.3e}/s"
        );
    }

    /// The reaction the return-leg probe should be driven on: exergonic,
    /// thermal, and the one this pond can pay for the most of.
    ///
    /// "Has some of every substrate" is not enough, and choosing that way is
    /// how these tests first passed while measuring nothing. Ranked that way
    /// the first hit here limits on a compound the pond holds 1.1e-8 particles
    /// of; ranking on the limiting substrate instead does not help, because
    /// *every* exergonic thermal reaction in this world limits on a trace. The
    /// abundant compounds are not food -- which is the finding this project
    /// has spent eight runs on, showing up in a unit test.
    fn drivable(w: &World) -> u32 {
        w.chem
            .reactions
            .iter()
            .filter(|r| r.drive == hadean_chem::chemistry::Drive::Thermal && r.dh < 0.0)
            .max_by(|a, b| {
                let limit = |r: &Reaction| {
                    r.reactants
                        .iter()
                        .map(|&(c, n)| w.amounts.total_of(c as usize) / n as f64)
                        .fold(f64::INFINITY, f64::min)
                };
                limit(a)
                    .partial_cmp(&limit(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.id)
            .expect("no exergonic thermal reaction with substrates in the water")
    }

    /// Put enough of a reaction's substrates in every voxel that driving it
    /// moves a measurable fraction of the water.
    ///
    /// This is a fixture, not a mechanism: it wrecks the audit baseline, so
    /// only tests that do not read the audit may use it. It exists because
    /// every exergonic reaction in this pond limits on a trace substrate, and
    /// a test of where the joules went cannot be run on a turnover of 1e-8
    /// particles -- there is nothing there to see.
    fn stock_up(w: &mut World, reaction: u32, per_voxel: f32) {
        let reactants = w.chem.reactions[reaction as usize].reactants.clone();
        for &(c, _) in &reactants {
            for voxel in 0..w.grid.len() {
                w.amounts.set(c as usize, voxel, per_voxel);
                w.residual.set(c as usize, voxel, 0.0);
            }
        }
    }

    #[test]
    fn the_return_leg_keeps_every_atom_in_the_pond() {
        // The one thing that distinguishes this probe from `harvest` is that
        // nothing leaves. A harvest is booked at the boundary; a turnover is
        // not booked at all, because there is nothing to book -- so if it ever
        // moved an atom across the boundary the mass audit would say so and
        // nothing else would.
        let mut w = World::new(small()).expect("builds");
        w.run(200);
        let before = w.audit_now();
        assert!(before.passes(1.0e-6), "drifting before the probe");
        let ledger = w.audit.elements_in;

        let reaction = drivable(&w);
        let mut turnovers = 0.0;
        for _ in 0..300 {
            w.step();
            turnovers += w.turn_over(reaction).turnovers;
        }
        assert!(turnovers > 0.0, "the pond never returned any substrate");

        // Vents keep injecting during the probe, so the ledger moves; what
        // must not happen is the probe moving it.
        for e in 0..hadean_chem::element::N_ELEMENTS {
            assert!(
                w.audit.elements_in[e] >= ledger[e],
                "element {e} left the world through a turnover"
            );
        }
        let after = w.audit_now();
        assert!(
            after.passes(1.0e-6),
            "the return leg broke the audit: energy {:e}, mass {:e}",
            after.relative,
            after.mass_drift
        );
    }

    /// The enthalpy has to land somewhere. `dh` is negative, so driving the
    /// reaction forward lowers the pond's chemical energy, and the only reason
    /// the audit stays flat is that exactly those joules arrive in the water
    /// as heat -- measured from the state either side, not from the intended
    /// extent, so a clamp or an `f32` rounding is absorbed rather than lost.
    ///
    /// The comparison is made on the reaction's own participants rather than
    /// on the audit's `chemical` total. That total is a sum over every
    /// compound in every voxel, and one turnover of one reaction moves it by
    /// far less than an ulp of it: the world-level reading cannot see this at
    /// all, which is why the flat-audit test above is a check on conservation
    /// and not on where the energy went.
    #[test]
    fn the_return_leg_pays_its_enthalpy_into_the_water() {
        let mut w = World::new(small()).expect("builds");
        w.run(200);
        let reaction = drivable(&w);
        let participants: Vec<(usize, f64)> = {
            let r = &w.chem.reactions[reaction as usize];
            r.reactants
                .iter()
                .chain(r.products.iter())
                .map(|&(c, _)| (c as usize, w.chem.compound(c).h_f))
                .collect()
        };
        let chemical = |w: &World| -> f64 {
            participants
                .iter()
                .map(|&(c, h)| (w.amounts.total_of(c) + w.residual.total_of(c)) * h)
                .sum()
        };

        stock_up(&mut w, reaction, 1.0e10);
        let chem_before = chemical(&w);
        let thermal_before = w.audit_now().energy.thermal;
        let out = w.turn_over(reaction);
        assert!(out.turnovers > 0.0, "nothing to drive");
        let released = chem_before - chemical(&w);
        let warmed = w.audit_now().energy.thermal - thermal_before;

        assert!(
            released > 0.0,
            "an exergonic turnover did not lower the participants' chemical energy"
        );
        // Three readings of the same joules, and they have to agree: what the
        // participants lost, what `turn_over` says it took, and what the water
        // gained. Any one of them alone would be self-reported.
        assert!(
            (out.released - released).abs() <= 1.0e-6 * released,
            "turn_over reported {:e} J, the participants lost {released:e} J",
            out.released
        );
        assert!(
            (warmed - released).abs() <= 1.0e-6 * released,
            "released {released:e} J of chemistry and the water gained {warmed:e} J"
        );
    }

    /// A turnover must not leave a negative amount behind. `extent` is
    /// `have / n` minimised over the reactants, and `have / n * n` is not
    /// always `have` in `f64` -- for `n = 3` it can land an ulp above -- so
    /// the extent is shrunk by a couple of ulps before anything is applied.
    /// Driven hard for three hundred ticks, nothing anywhere in the field may
    /// go below zero.
    #[test]
    fn a_turnover_never_leaves_a_negative_amount() {
        let mut w = World::new(small()).expect("builds");
        w.run(200);
        let reaction = drivable(&w);
        for _ in 0..300 {
            w.step();
            w.turn_over(reaction);
            assert!(
                w.amounts.data.iter().all(|&x| x >= 0.0),
                "a turnover drove a compound below zero"
            );
        }
    }

    /// And the shrink must be applied to the extent, not to the consumption.
    /// Clamping a reactant as it is consumed while still adding the products
    /// at the full extent would create atoms, quietly and below every
    /// tolerance in the project. Driven at a coefficient of three, which is
    /// where `have / n * n` overshoots, the element ledger must not move.
    #[test]
    fn a_turnover_creates_no_atoms_even_when_the_extent_rounds() {
        let mut w = World::new(small()).expect("builds");
        w.run(200);
        // Every exergonic thermal reaction, so a coefficient above one is in
        // the set whatever the seed generated.
        let reactions: Vec<u32> = w
            .chem
            .reactions
            .iter()
            .filter(|r| r.drive == hadean_chem::chemistry::Drive::Thermal && r.dh < 0.0)
            .map(|r| r.id)
            .collect();
        let before = w.audit_now();
        assert!(before.passes(1.0e-6), "drifting before the probe");
        for _ in 0..50 {
            w.step();
            for &r in &reactions {
                w.turn_over(r);
            }
        }
        let after = w.audit_now();
        assert!(
            after.mass_drift.abs() <= 1.0e-9,
            "turnovers moved mass by {:e}",
            after.mass_drift
        );
    }

    /// A turnover consumes the reaction's scarcest substrate outright, the way
    /// a harvest empties a compound, and that is what makes the reading a
    /// ceiling rather than a guess about kinetics.
    #[test]
    fn a_turnover_empties_the_scarcest_substrate() {
        let mut w = World::new(small()).expect("builds");
        w.run(200);
        let reaction = drivable(&w);
        let reactants = w.chem.reactions[reaction as usize].reactants.clone();
        assert!(w.turn_over(reaction).turnovers > 0.0, "nothing to drive");
        let left: f64 = reactants
            .iter()
            .map(|&(c, n)| w.amounts.total_of(c as usize) / n as f64)
            .fold(f64::INFINITY, f64::min);
        // Every voxel gave up as much as it could, so the scarcest substrate
        // is gone from all of them; what is left is the rounding the residual
        // carries, not a stock.
        assert!(
            left <= 1.0e-3 * w.grid.len() as f64,
            "the scarcest substrate survived the turnover: {left:e} particles"
        );
    }

    /// `returns` restores every probe from one shared snapshot and runs them
    /// on separate threads, so a turnover has to be a pure function of the
    /// state it is handed. If it were not, `--jobs` would change the answer
    /// and the whole table would be a measurement of the scheduler.
    #[test]
    fn a_turnover_is_the_same_through_a_snapshot_and_a_copy() {
        let mut w = World::new(small()).expect("builds");
        w.run(200);
        let reaction = drivable(&w);
        let bytes = crate::snapshot::save(&w).expect("saves");

        let run = |w: &mut World| -> Vec<(f64, f64)> {
            (0..20)
                .map(|_| {
                    w.step();
                    let out = w.turn_over(reaction);
                    (out.turnovers, out.released)
                })
                .collect()
        };

        let mut a = crate::snapshot::load(&bytes).expect("loads");
        let mut b = crate::snapshot::load(&bytes).expect("loads");
        assert_eq!(run(&mut a), run(&mut b), "two copies disagreed");
        // And bit-identical against the world it was saved from, which is the
        // property `verify`'s replay gate asserts for `step` alone.
        assert_eq!(
            run(&mut crate::snapshot::load(&bytes).expect("loads")),
            run(&mut w),
            "the snapshot and its origin disagreed"
        );
        assert_eq!(a.state_digest(), b.state_digest());
    }

    #[test]
    fn vents_warm_the_floor() {
        let mut w = World::new(small()).expect("builds");
        w.run(400);
        let profile = w.temperature_profile();
        assert!(
            profile[profile.len() - 1] > profile[0],
            "no thermal gradient: {profile:?}"
        );
    }

    #[test]
    fn vents_inject_matter_and_it_is_recorded() {
        let mut w = World::new(small()).expect("builds");
        let fuel = w.chem.vent_fuel[0] as usize;
        let before = w.amounts.total_of(fuel);
        w.run(100);
        assert!(w.amounts.total_of(fuel) > before, "vents injected nothing");
        assert!(w.audit.elements_in.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn zero_vents_and_dark_world_is_closed() {
        // With every tap shut off, the world should conserve energy on its own.
        let cfg = WorldConfig {
            light: hadean_fields::light::LightConfig {
                irradiance: [0.0; N_BANDS],
                ..Default::default()
            },
            heat: hadean_fields::heat::HeatConfig {
                surface_transfer: 0.0,
                vents: 0,
                vent_power: 0.0,
                ..Default::default()
            },
            vents: crate::config::VentConfig {
                fuel_rate: 0.0,
                fuel_species: 0,
            },
            ..small()
        };
        let mut w = World::new(cfg).expect("builds");
        let report = w.run(200).expect("audited");
        assert_eq!(w.audit.ledger.throughput(), 0.0);
        assert!(
            report.relative.abs() < 1e-6,
            "closed world drifted by {:e} ({:e} J)",
            report.relative,
            report.drift
        );
    }

    #[test]
    fn the_state_digest_changes_as_the_world_evolves() {
        let mut w = World::new(small()).expect("builds");
        let start = w.state_digest();
        w.run(20);
        assert_ne!(start, w.state_digest());
    }
}
