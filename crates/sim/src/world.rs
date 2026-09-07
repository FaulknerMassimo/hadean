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
