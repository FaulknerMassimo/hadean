//! **L4 -- hardcoded protocells.**
//!
//! This is deliberately the pre-genome cell from Phase 2. Every cell has the
//! same membrane and catalyses the same generated reaction. The important
//! result is a complete material lifecycle: compounds cross a membrane, an
//! exergonic reaction charges a reserve, maintenance drains it, cells divide,
//! and dead cells return every particle and joule to the pond.

use std::f32::consts::PI;

use hadean_chem::{Chemistry, Drive, Reaction, ReactionId};
use hadean_core::hash::{HashState, StateHasher};
use hadean_core::rng::Purpose;
use hadean_core::units::{Joules, KB, VISCOSITY};
use hadean_core::{Axis, Counter, Grid, Schedule};
use hadean_fields::heat::HeatField;
use hadean_fields::scalar::ChemField;
use hadean_fields::transport::FaceVelocity;
use serde::{Deserialize, Serialize};

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
    /// Maximum lifespan in simulated seconds. Ageing makes death inevitable.
    pub maximum_age: f32,
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
}

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
            decomposition_rate: 0.8,
            seed_delay: 150.0,
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
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellState {
    Alive,
    Decomposing,
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
}

impl Cell {
    pub fn is_alive(&self) -> bool {
        self.state == CellState::Alive
    }

    pub fn voxel(&self, grid: &Grid) -> usize {
        grid.voxel_at(self.pos)
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
                CellState::Alive => {
                    move_cell(
                        cell,
                        grid,
                        flow,
                        rng,
                        tick,
                        dt,
                        heat.temperature(cell.voxel(grid)),
                    );
                    exchange(cell, cfg, grid, &table, amounts, residual, dt);
                    if let Some(id) = metabolism {
                        metabolize(cell, cfg, grid, chem.reaction(id), &table, heat, dt);
                    }
                    maintain(cell, cfg, grid, heat, dt);
                    cell.age += dt as f32;
                    cell.radius = growth_radius(cfg, cell.reserve);

                    if division_slots > 0
                        && Schedule::due_for(tick, cell.id, growth_interval)
                        && cell.reserve >= cfg.division_reserve
                    {
                        let child = divide(cell, cfg, grid, rng, tick, self.next_id);
                        self.next_id += 1;
                        self.births += 1;
                        division_slots -= 1;
                        daughters.push(child);
                    }

                    if cell.damage >= 1.0 || cell.age >= cfg.maximum_age {
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
            c.state == CellState::Alive
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

fn exchange(
    cell: &mut Cell,
    cfg: &CellConfig,
    grid: &Grid,
    table: &CompoundTable,
    amounts: &mut ChemField,
    residual: &mut ChemField,
    dt: f64,
) {
    let voxel = cell.voxel(grid);
    let area = (4.0 * PI * cell.radius * cell.radius) as f64;
    let cell_v = cell_volume(cell.radius);
    let voxel_v = grid.voxel_volume() as f64;
    let scale = cfg.membrane_scale as f64 * area * dt;
    for c in 0..table.len() {
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

fn metabolize(
    cell: &mut Cell,
    cfg: &CellConfig,
    grid: &Grid,
    reaction: &Reaction,
    table: &CompoundTable,
    heat: &mut HeatField,
    dt: f64,
) {
    let fraction = 1.0 - (-cfg.metabolic_rate as f64 * dt).exp();
    let mut extent = f64::INFINITY;
    for &(c, n) in &reaction.reactants {
        extent = extent.min(cell.contents[c as usize] / n as f64);
    }
    extent *= fraction;
    if !extent.is_finite() || extent <= 0.0 {
        return;
    }

    let before = table.energy_of(&cell.contents);
    for &(c, n) in &reaction.reactants {
        cell.contents[c as usize] -= extent * n as f64;
    }
    for &(c, n) in &reaction.products {
        cell.contents[c as usize] += extent * n as f64;
    }
    let after = table.energy_of(&cell.contents);
    let released = (before - after).max(0.0);
    let desired_heat = released * (1.0 - cfg.capture_efficiency as f64);
    let landed = heat.deposit(grid, cell.voxel(grid), desired_heat);
    // Whatever did not actually land as heat remains stored. This includes
    // normal heat-deposit rounding, keeping the audit exact at the boundary.
    cell.reserve += released - landed;
}

fn maintain(cell: &mut Cell, cfg: &CellConfig, grid: &Grid, heat: &mut HeatField, dt: f64) {
    let due = cfg.maintenance_power * dt;
    if cell.reserve >= due {
        let landed = heat.deposit(grid, cell.voxel(grid), due);
        cell.reserve -= landed;
        cell.damage = (cell.damage - dt as f32 / cfg.starvation_time).max(0.0);
    } else {
        cell.damage += dt as f32 / cfg.starvation_time;
    }
}

fn growth_radius(cfg: &CellConfig, reserve: Joules) -> f32 {
    let progress = (reserve / cfg.division_reserve).clamp(0.0, 1.0) as f32;
    cfg.birth_radius * (1.0 + (2.0f32.cbrt() - 1.0) * progress)
}

fn divide(
    parent: &mut Cell,
    cfg: &CellConfig,
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
            exchange(
                cell,
                &cfg,
                &grid,
                &CompoundTable::new(&chem),
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
        let child = divide(parent, &cfg, &grid, &rng, 20, 99);
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
}
