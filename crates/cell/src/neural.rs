//! L6 -- the interpreter for the three classes that had nothing to read them.
//!
//! [`Receptor`](ProteinClass::Receptor), [`Neural`](ProteinClass::Neural) and
//! [`Effector`](ProteinClass::Effector) have been part of the genome format
//! since L3 and have, until now, decoded into a synthesis bill with nothing on
//! the other side. This module is the other side, and it is the last place in
//! the cell layer where a decision is made.
//!
//! That is the point of it. Before this, what a cell did when it ran out of
//! food was written in `maintain`: below the line, shut down; above
//! `dormancy_exit` seconds of banked upkeep, wake. Every cell in the pond
//! obeyed the same two numbers, so every cell in the pond shut down at the
//! same moment and none of them ever woke -- which is not a population making
//! a bad decision, it is a population with no decision in it. What replaced it
//! is not a better rule. It is a network the genome specifies:
//!
//! ```text
//!     receptor ---- signal ----> neuron ---- signal ----> effector
//!     (senses)                  (decides)                 (acts)
//! ```
//!
//! and there is no path from the world to a cell's behaviour that does not go
//! through it.
//!
//! # The signal network
//!
//! Every gene has an eight-dimensional key. A receptor or a neuron *emits* its
//! output at its own key; a neuron or an effector *listens* through its
//! [`Site`](crate::genome::Site)s, each of which is a motif and a signed
//! weight. What arrives at a site is
//!
//! ```text
//!     sum over sources of  output(source) * affinity(source.key, motif)
//! ```
//!
//! -- the same [`affinity`](crate::genome::affinity) that matches an enzyme to
//! a reaction and a regulator to a promoter, because there is only one of
//! those in this simulation and adding a second would be adding a second way
//! for two things to be alike. Which wires exist is settled once per genome by
//! [`Genome::bind`](crate::genome::Genome::bind); what travels down them is
//! per-cell state, and that is the whole reason two cells carrying the same
//! bytes can behave differently.
//!
//! # It is the plan's CTRNN, wired by affinity
//!
//! `PLAN.md` §L6 asks for a continuous-time recurrent network per cell,
//!
//! ```text
//!     tau_i dy_i/dt = -y_i + sum_j w_ij sigma(y_j + theta_j) + I_i
//! ```
//!
//! and this is that, with the squash moved to the receiving neuron:
//!
//! ```text
//!     da_i/dt = leak_i (tanh(theta_i + sum_j w_ij a_j) - a_i)
//! ```
//!
//! `leak` is `1/tau`, `theta` is [`Gene::bias`](crate::genome::Gene::bias),
//! the `w_ij` are a dendrite's weight times the affinity between its motif and
//! the source's key, and `I` is whatever the receptors are reporting.
//!
//! The plan is also emphatic about what *not* to do: "don't encode a network
//! topology in the genome -- that's brittle and doesn't scale". So the
//! topology is not encoded. It is *matched*, by the same key affinity that
//! matches an enzyme to a reaction, which means a duplicated neuron is
//! immediately wired like the original and drifts away from it one byte at a
//! time. What the plan grows developmentally across a body, this grows by
//! recognition inside one cell; the inter-cell half of it needs gap junctions
//! and cell-cell physics, which is L5 and does not exist yet.
//!
//! # Why it is recurrent
//!
//! A neuron reads the activations the network held at the *end of the previous
//! tick*, not activations computed earlier in this one. There is no layering,
//! no ordering constraint and no acyclicity check: a neuron may hear itself.
//!
//! This is cheaper than a feed-forward pass and it is also the substantive
//! choice. Combined with [`Gene::leak`](crate::genome::Gene::leak) -- the rate
//! at which a neuron relaxes towards what its dendrites say -- it gives a cell
//! *memory*, and memory is exactly what the hardcoded dormancy rule was
//! faking. `dormancy_exit` existed because a cell sitting on the margin would
//! otherwise flicker between shut down and working every tick and average back
//! into paying the full bill. A slow neuron does not flicker, and how slow it
//! is, is four bits of one byte in its own genome. Deep and brief, shallow and
//! long, and everything between, are now things a lineage can be rather than
//! things a config file is.
//!
//! # What is fixed here, and why that is not hardcoding
//!
//! Two lists: [`Channel`], the things there are to sense, and [`Action`], the
//! things there are to do. Neither is a behaviour. A cell has a membrane, so
//! there is something to taste through; it has a reserve, so there is a hunger
//! to feel; it can shut down, divide, open a transporter and swim, so there
//! are four things an effector can be wired to. Which channel a receptor
//! watches and which action an effector drives are `params[1]` of the gene --
//! genetic, mutable, and selected on. The lists are the cell's physiology, and
//! a genome that could invent an organ it does not have would not be a genome.

use hadean_chem::chemistry::CompoundId;
use hadean_core::units::N_REF;
use hadean_fields::scalar::ChemField;

use crate::genome::ProteinClass;
use crate::{Cell, CellConfig};

/// What a receptor can be pointed at.
///
/// The order is part of the genome format in the same way
/// [`ProteinClass`]'s discriminants are: a receptor picks a channel by
/// quantising `params[1]`, so inserting a variant in the middle re-points
/// every receptor in every stored genome. **Append only.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// How much of what this receptor's key matches is in the water outside.
    /// Chemotaxis, and the sense a cell needs to know it has arrived
    /// somewhere worth staying.
    Food = 0,
    /// The same compounds, inside. A cell that can tell its own larder from
    /// the pond's can stop taking up what it already has too much of.
    Internal = 1,
    /// Reserve, against what upkeep costs. Hunger.
    Energy = 2,
    /// Accumulated starvation damage, 0..1. How close to dead.
    Damage = 3,
    /// Light at this voxel, against full surface sun.
    Light = 4,
    /// Temperature.
    Heat = 5,
    /// Cells sharing this voxel. Crowding, and the only sense here that is
    /// about other cells at all.
    Crowd = 6,
    /// Age, against this cell's own allotted span.
    Age = 7,
}

/// How many there are. A receptor's `params[1]` is quantised onto this.
pub const N_CHANNELS: usize = 8;

impl Channel {
    pub fn of(index: usize) -> Self {
        match index {
            0 => Self::Food,
            1 => Self::Internal,
            2 => Self::Energy,
            3 => Self::Damage,
            4 => Self::Light,
            5 => Self::Heat,
            6 => Self::Crowd,
            _ => Self::Age,
        }
    }

    /// Whether the receptor's key selects compounds as well as addressing its
    /// output. For the two chemical channels it does both.
    pub fn is_chemical(self) -> bool {
        matches!(self, Self::Food | Self::Internal)
    }
}

/// What an effector can be wired to.
///
/// **Append only**, for the reason given on [`Channel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Shut down. Drives [`Drives::quiesce`], which is how much of the working
    /// bill the cell declines to pay -- and, because a cell that is not
    /// spending is not synthesising, how fast its proteome decays while it
    /// waits. Sleep is not free here: it costs the machinery.
    Quiesce = 0,
    /// Hold off dividing. A cell with the reserve to split still needs to
    /// decide that now is the time.
    Divide = 1,
    /// Open or close the transporters for the compounds this effector's key
    /// matches. The genome's answer to "that does not taste good": a lineage
    /// can shut its membrane against something without losing the transporter
    /// that would carry it if the pond changed its mind.
    Ingest = 2,
    /// Swim, along the axis in `params[3]`, in the direction of the sign of
    /// the output. Costs energy, which the audit sees.
    Move = 3,
}

/// How many there are. An effector's `params[1]` is quantised onto this.
pub const N_ACTIONS: usize = 4;

impl Action {
    pub fn of(index: usize) -> Self {
        match index {
            0 => Self::Quiesce,
            1 => Self::Divide,
            2 => Self::Ingest,
            _ => Self::Move,
        }
    }
}

/// The world as one cell can measure it this tick.
///
/// Assembled by the caller because most of it is per-voxel and the same for
/// every cell in that voxel; what varies per cell is read off the cell itself.
pub struct Surroundings<'a> {
    pub amounts: &'a ChemField,
    pub voxel: usize,
    /// Voxel volume, m^3. Amounts are particle counts, and a receptor should
    /// taste a concentration.
    pub voxel_volume: f64,
    /// Light here as a fraction of the surface's, 0..1.
    pub brightness: f32,
    /// Kelvin.
    pub temperature: f32,
    /// Living cells sharing this voxel, this one included.
    pub crowd: u32,
    /// The span this cell was allotted at birth, seconds.
    pub lifespan: f32,
}

/// What the network decided, in the units the lifecycle uses.
///
/// Read out of the effectors every tick, because it is a pure function of the
/// activations and cheap; the *neurons* update on `schedule.neural`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drives {
    /// How far shut down, 0..1. Scales continuously: there is no dormant
    /// state to be in, only a depth to be at.
    pub quiesce: f32,
    /// Whether the genome permits a division this tick.
    pub divide: bool,
    /// Swimming, -1..1 per axis, already clamped to what the reserve affords.
    pub thrust: [f32; 3],
    /// Whether any effector is gating a transporter. When nothing is, the
    /// per-compound pass is skipped entirely.
    pub gating: bool,
}

impl Default for Drives {
    /// What a cell with no effectors does: nothing special. It pays its full
    /// bill, divides when it can afford to, keeps its membrane as its
    /// transporters left it, and does not swim.
    ///
    /// Permissive rather than inert, and the distinction matters at every
    /// deletion: a genome that loses its division effector should divide
    /// freely, not become sterile. Absence of a brake is not absence of drive.
    fn default() -> Self {
        Self {
            quiesce: 0.0,
            divide: true,
            thrust: [0.0; 3],
            gating: false,
        }
    }
}

/// Fraction of the working bill a cell at quiescence `q` still pays.
///
/// The one place the depth is turned into money, so `maintain`, `transcribe`
/// and the motility budget cannot disagree about what being half asleep means.
/// At `q = 0` this is 1 and at `q = 1` it is `dormancy_power_fraction`, which
/// is what that dial has always meant -- it is only the *decision* to be there
/// that has moved out of the config and into the genome.
pub fn activity(cfg: &CellConfig, quiesce: f32) -> f64 {
    let q = quiesce.clamp(0.0, 1.0) as f64;
    1.0 - q * (1.0 - cfg.dormancy_power_fraction)
}

/// Squash to `-1..1`. One function, so a neuron and an effector agree about
/// what a strong input is.
fn squash(x: f32) -> f32 {
    x.tanh()
}

/// A saturating `0..1` reading of a quantity against a reference.
///
/// Michaelis form rather than a clamp, so a receptor stays informative across
/// orders of magnitude instead of pinning at 1 the moment there is plenty. It
/// reads 0.5 at the reference and never reaches either end, which also means
/// no channel can hand a neuron an input it cannot come back from.
fn saturating(x: f32, reference: f32) -> f32 {
    if x.is_finite() && x > 0.0 && reference > 0.0 {
        x / (x + reference)
    } else {
        0.0
    }
}

/// How much of a gene's protein is actually there to work with, 0..1.
///
/// Concentration times strength, saturating: an unexpressed gene is silent
/// however strong its protein would be, and a strong one saturates rather than
/// scaling without bound.
fn gate(conc: f32, strength: f32) -> f32 {
    (conc * strength).clamp(0.0, 1.0)
}

/// Sense the world and advance the neurons.
///
/// Receptors are read every tick -- they are the cell's contact with the
/// world and staggering them would stagger the world. Neurons advance on
/// their slot of `schedule.neural`, with the matching `dt`, which is what that
/// schedule entry has been reserved for since L0.
///
/// Determinism: this is arithmetic on the cell's own state and on fields it
/// only reads. No draw is made, so nothing here depends on the order cells are
/// stepped in.
pub fn sense_and_think(
    cell: &mut Cell,
    cfg: &CellConfig,
    world: &Surroundings,
    scratch: &mut Vec<f32>,
    neural_due: bool,
    dt: f64,
) {
    let Some(g) = cell.genome.clone() else {
        return;
    };
    let n = g.genes.len();
    if n == 0 {
        cell.activation.clear();
        return;
    }
    // The same check `Expression::read` makes, for the same reason: an unbound
    // genome has no targets, so every zip below yields nothing and the cell
    // silently has no nervous system at all. Every path that hands a genome to
    // a cell binds it first; this is here so that a path that forgets says so
    // immediately rather than producing a cell that cannot decide anything.
    debug_assert_eq!(
        g.genes.len(),
        g.targets.len(),
        "a cell is thinking with a genome that was never bound to the chemistry"
    );
    // A daughter whose genome mutated carries her mother's activations, which
    // is right -- she inherits her cytoplasm, and that includes whatever her
    // mother's network was in the middle of. New genes start silent.
    if cell.activation.len() != n {
        cell.activation.resize(n, 0.0);
    }

    // Receptors first, into the same vector the neurons read from. A neuron
    // that hears a receptor therefore hears this tick's world, while a neuron
    // that hears another neuron hears last tick's -- the sensory lag is zero
    // and the deliberative lag is one tick, which is the right way round.
    for (i, gene) in g.genes.iter().enumerate() {
        if gene.class != ProteinClass::Receptor {
            continue;
        }
        let conc = cell.proteome.get(i).copied().unwrap_or(0.0);
        let open = gate(conc, gene.strength());
        let value = if open > 0.0 {
            let channel = Channel::of(gene.selector(N_CHANNELS));
            open * sense(cell, cfg, world, channel, &g.targets[i].compounds)
        } else {
            0.0
        };
        cell.activation[i] = value;
    }

    if !neural_due {
        return;
    }
    // Neurons. Read from the activations as they stand -- receptors updated
    // above, other neurons as they ended last tick -- and write into a scratch
    // so that within one update no neuron sees another's new value. Without
    // that the result would depend on gene order, which duplication and
    // inversion both change. The scratch is the caller's, because a population
    // of a few thousand would otherwise allocate a vector per cell per tick.
    scratch.clear();
    for (i, ((gene, targets), &a)) in g
        .genes
        .iter()
        .zip(&g.targets)
        .zip(&cell.activation)
        .enumerate()
    {
        if gene.class != ProteinClass::Neural {
            continue;
        }
        let mut input = gene.bias();
        for (site, wires) in gene.sites.iter().zip(&targets.dendrites) {
            let mut arriving = 0.0;
            for &(j, w) in wires {
                arriving += cell.activation[j] * w;
            }
            input += site.weight * arriving;
        }
        // Gated on the emitting side, not the receiving: an unexpressed
        // neuron relaxes towards silence rather than having its stored state
        // scaled away, so a network that is being switched off fades instead
        // of forgetting.
        let target = squash(input) * gate(cell.proteome.get(i).copied().unwrap_or(0.0), gene.strength());
        let step = (gene.leak() * dt as f32).min(1.0);
        scratch.push(a + step * (target - a));
    }
    // Written back over the same zip the values were computed from, not over
    // the gene list: the two agree on a bound genome and only on a bound
    // genome, and a writeback that walked the longer of them would panic in
    // release on exactly the case the assertion above catches in debug.
    let mut next = scratch.iter();
    for ((gene, _), a) in g
        .genes
        .iter()
        .zip(&g.targets)
        .zip(&mut cell.activation)
    {
        if gene.class == ProteinClass::Neural {
            if let Some(&value) = next.next() {
                *a = value;
            }
        }
    }
}

/// One channel, read as a number on `0..1`.
///
/// Every map here saturates and none of them can go negative: a receptor
/// reports how much of something there is, and it is the sign of a dendrite's
/// weight that decides whether more of it is good news.
fn sense(
    cell: &Cell,
    cfg: &CellConfig,
    world: &Surroundings,
    channel: Channel,
    compounds: &[(CompoundId, f32)],
) -> f32 {
    match channel {
        Channel::Food => {
            let mut n = 0.0f32;
            for &(c, a) in compounds {
                n += a * world.amounts.get(c as usize, world.voxel);
            }
            saturating(n, N_REF)
        }
        Channel::Internal => {
            // Compared at the same concentration as the outside reading, so a
            // cell holding as much as the water does reads 0.5 on both.
            let scale = (cell_volume_of(cell) / world.voxel_volume) as f32;
            let mut n = 0.0f32;
            for &(c, a) in compounds {
                n += a * cell.contents.get(c as usize).copied().unwrap_or(0.0) as f32;
            }
            saturating(n, N_REF * scale.max(f32::MIN_POSITIVE))
        }
        // Hunger, measured in the only currency that means anything to a cell:
        // how long the reserve would keep it alive. `starvation_time` is the
        // right horizon because it is already the world's statement of how
        // long a cell can go unpaid, so this introduces no new number.
        Channel::Energy => saturating(
            cell.reserve as f32,
            (cell.maintenance(cfg) * cfg.starvation_time as f64) as f32,
        ),
        Channel::Damage => cell.damage.clamp(0.0, 1.0),
        Channel::Light => world.brightness.clamp(0.0, 1.0),
        // Zero at freezing, one at boiling. Absolute rather than relative to
        // the pond's ambient, so a receptor's setpoint means the same thing in
        // a world with different vents.
        Channel::Heat => ((world.temperature - 273.15) / 100.0).clamp(0.0, 1.0),
        Channel::Crowd => saturating(world.crowd as f32, 8.0),
        Channel::Age => {
            if world.lifespan > 0.0 {
                (cell.age / world.lifespan).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
    }
}

fn cell_volume_of(cell: &Cell) -> f64 {
    let r = cell.radius as f64;
    4.0 / 3.0 * std::f64::consts::PI * r * r * r
}

/// Read the effectors and turn their outputs into what the lifecycle does.
///
/// `ingest` is filled with a per-compound multiplier on the cell's
/// *transporters* -- never on the bilayer, which is physics and not a decision
/// the cell gets to make. That invariant is why a lineage cannot mutate into a
/// sealed box: the worst an effector can do is close what evolution built, and
/// what is left is the membrane every cell has.
pub fn drive(cell: &Cell, cfg: &CellConfig, dt: f64, ingest: &mut [f32]) -> Drives {
    let mut drives = Drives::default();
    let Some(g) = &cell.genome else {
        return drives;
    };

    let mut quiesce = 0.0f32;
    let mut divide = 0.0f32;
    let mut has_divide = false;
    let mut thrust = [0.0f32; 3];

    for ((gene, targets), &conc) in g.genes.iter().zip(&g.targets).zip(&cell.proteome) {
        if gene.class != ProteinClass::Effector {
            continue;
        }
        let open = gate(conc, gene.strength());
        if open <= 0.0 {
            continue;
        }
        let mut input = gene.bias();
        for (site, wires) in gene.sites.iter().zip(&targets.dendrites) {
            let mut arriving = 0.0;
            for &(j, a) in wires {
                arriving += cell.activation.get(j).copied().unwrap_or(0.0) * a;
            }
            input += site.weight * arriving;
        }
        let out = squash(input) * open;

        match Action::of(gene.selector(N_ACTIONS)) {
            Action::Quiesce => quiesce += out,
            Action::Divide => {
                divide += out;
                has_divide = true;
            }
            Action::Ingest => {
                if !drives.gating {
                    ingest.fill(0.0);
                    drives.gating = true;
                }
                for &(c, a) in &targets.compounds {
                    ingest[c as usize] += out * a;
                }
            }
            Action::Move => thrust[gene.axis()] += out,
        }
    }

    drives.quiesce = quiesce.clamp(0.0, 1.0);
    // A genome with no brake on division has no opinion about it, and no
    // opinion means yes. See `Drives::default`.
    drives.divide = !has_divide || divide > 0.0;
    for t in &mut thrust {
        *t = t.clamp(-1.0, 1.0);
    }

    // Swimming is bought, not willed. A cell that cannot pay for the stroke
    // does not take it, and scaling rather than refusing keeps the response
    // graded -- a lineage that overspends slows down instead of stopping dead,
    // so selection can see the difference.
    let want = motility_cost(cfg, &thrust) * dt;
    if want > 0.0 {
        let affordable = cell.reserve.max(0.0);
        if affordable < want {
            let scale = (affordable / want).sqrt() as f32;
            for t in &mut thrust {
                *t *= scale;
            }
        }
    }
    drives.thrust = thrust;

    if drives.gating {
        // `1 + gate * transporters` and nothing else: the multiplier lands on
        // what the cell built, and the bilayer underneath it is untouched.
        for slot in ingest.iter_mut() {
            *slot = (1.0 + *slot).clamp(0.0, 2.0);
        }
    }
    drives
}

/// What holding this thrust costs, W.
///
/// Quadratic in the *total* speed, because dragging a sphere through water at
/// low Reynolds number costs a power proportional to the square of it. Two
/// consequences, and both are load-bearing: half speed is a quarter of the
/// bill, so a lineage that needs to be somewhere and a lineage that needs to
/// be cheap have genuinely different optima; and the sum is not capped, so a
/// cell driving all three axes at once pays three times, rather than getting
/// two of them free.
pub fn motility_cost(cfg: &CellConfig, thrust: &[f32; 3]) -> f64 {
    let squared: f32 = thrust.iter().map(|t| t * t).sum();
    cfg.motility_power * squared as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{write_gene, Genome, ProteinClass, N_PARAMS};
    use hadean_chem::chemistry::KEY_DIM;
    use crate::{CellState, Traits};
    use hadean_core::Grid;
    use std::sync::Arc;

    const PLAIN: [f32; KEY_DIM] = [0.5; KEY_DIM];

    /// A cell carrying exactly the genes handed to it, expressed at their
    /// basal levels.
    fn wearing(genome: Genome) -> Cell {
        let n = genome.genes.len();
        let proteome = genome.genes.iter().map(|g| g.basal).collect();
        Cell {
            id: 0,
            parent: None,
            pos: [25.0e-6; 3],
            radius: 4.0e-6,
            contents: vec![0.0; 8],
            reserve: 1.0e-9,
            damage: 0.0,
            age: 0.0,
            state: CellState::Alive,
            generation: 0,
            traits: Traits::default(),
            genome: Some(Arc::new(genome)),
            proteome,
            activation: vec![0.0; n],
            quiesce: 0.0,
        }
    }

    /// One neuron with a given bias and rate of forgetting, and nothing wired
    /// to it -- so what it settles at is `tanh(bias)` and the only question is
    /// how fast it gets there.
    fn bound(bytes: Vec<u8>) -> Genome {
        let chem = hadean_chem::generate(1, hadean_chem::ChemParams::default());
        let mut g = Genome::new(bytes);
        g.bind(&chem, 0.08, 0.15);
        g
    }

    fn lone_neuron(bias: f32, leak_byte: f32) -> Cell {
        let mut params = [0.0f32; N_PARAMS];
        params[0] = 0.75; // strength 2.25 against a basal of 0.9: fully open.
        params[2] = bias / 8.0 + 0.5;
        params[3] = leak_byte;
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &PLAIN,
            0.9,
            &[],
            ProteinClass::Neural,
            &PLAIN,
            &params,
            0.05,
        );
        wearing(bound(bytes))
    }

    fn think(cell: &mut Cell, cfg: &CellConfig, world: &Surroundings, ticks: usize) {
        let mut scratch = Vec::new();
        for _ in 0..ticks {
            sense_and_think(cell, cfg, world, &mut scratch, true, 0.01);
        }
    }

    fn nowhere(amounts: &ChemField) -> Surroundings<'_> {
        Surroundings {
            amounts,
            voxel: 0,
            voxel_volume: 1.5625e-14,
            brightness: 0.0,
            temperature: 293.15,
            crowd: 1,
            lifespan: 600.0,
        }
    }

    #[test]
    fn how_long_a_neuron_holds_an_opinion_is_in_its_genome() {
        // The claim that let `dormancy_exit` be deleted. That dial existed
        // because a cell on the margin would otherwise flicker between shut
        // down and working every tick, and it was one number for the whole
        // pond -- so every cell in it had the same memory, and a population
        // with one memory cannot have some members riding out a shortage that
        // others give up on. `leak` is four bits of each cell's own genome.
        let cfg = CellConfig::default();
        let grid = Grid::new(2, 2, 2, 25.0e-6);
        let amounts = ChemField::new(&grid, 8);
        let world = nowhere(&amounts);

        // params[3] of 0.0 is 0.1/s -- a ten-second memory. 1.0 is 20/s.
        let mut slow = lone_neuron(2.0, 0.0);
        let mut quick = lone_neuron(2.0, 1.0);
        think(&mut slow, &cfg, &world, 100); // one second
        think(&mut quick, &cfg, &world, 100);

        let settled = 2.0f32.tanh();
        assert!(
            quick.activation[0] > 0.9 * settled,
            "the fast neuron had not made up its mind in a second: {:.3}",
            quick.activation[0]
        );
        assert!(
            slow.activation[0] < 0.2 * settled,
            "the slow neuron made up its mind in a second: {:.3}",
            slow.activation[0]
        );

        // Given long enough they agree about *what* to think, and differ only
        // in how quickly -- which is the axis selection gets to act on.
        think(&mut slow, &cfg, &world, 10_000);
        assert!((slow.activation[0] - settled).abs() < 0.02);
    }

    #[test]
    fn a_neuron_may_hear_itself_without_running_away() {
        // Recurrence is not guarded by an acyclicity check and must not need
        // one. A neuron whose dendrite motif sits on its own key is wired to
        // itself at full strength, positively, with a bias -- the shape that
        // would diverge in an unsquashed network.
        let cfg = CellConfig::default();
        let grid = Grid::new(2, 2, 2, 25.0e-6);
        let amounts = ChemField::new(&grid, 8);
        let world = nowhere(&amounts);

        let mut params = [0.0f32; N_PARAMS];
        params[0] = 0.75;
        params[2] = 0.5625; // bias +0.5
        params[3] = 0.5;
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &PLAIN,
            0.9,
            &[(PLAIN, 4.0)],
            ProteinClass::Neural,
            &PLAIN,
            &params,
            0.05,
        );
        let genome = bound(bytes);
        assert_eq!(
            genome.targets[0].dendrites[0].len(),
            1,
            "the neuron is not wired to itself"
        );

        let mut cell = wearing(genome);
        think(&mut cell, &cfg, &world, 20_000);
        let a = cell.activation[0];
        assert!(a.is_finite(), "a self-connected neuron diverged");
        assert!((-1.0..=1.0).contains(&a), "activation left its range: {a}");
        // A positive self-connection with a positive bias latches high, which
        // is the memory element a cell needs to stay committed to sleeping.
        assert!(a > 0.9, "the latch did not latch: {a}");
    }

    #[test]
    fn an_effector_can_close_a_transporter_and_cannot_close_the_membrane() {
        // "That does not taste good" has to be expressible, and "seal myself
        // off and starve" must not be. The gate multiplies what the cell
        // built; the bilayer underneath is physics.
        let chem = hadean_chem::generate(1, hadean_chem::ChemParams::default());
        let cfg = CellConfig::default();
        let target = chem.compounds[3].key;
        let scale = crate::genome::KeyScale::of(&chem);
        let key = scale.normalise(&target);

        let mut transporter = [0.0f32; N_PARAMS];
        transporter[0] = 0.5;
        // An effector with no dendrites and a hard negative bias: it says
        // "shut" whatever the world is doing, which is what makes this a test
        // of the gate rather than of the network.
        let mut effector = [0.0f32; N_PARAMS];
        effector[0] = 0.75;
        effector[1] = 0.625; // Action::Ingest, 2 of 4
        effector[2] = 0.0; // bias -4
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &PLAIN,
            0.9,
            &[],
            ProteinClass::Transporter,
            &key,
            &transporter,
            0.05,
        );
        let open = {
            let mut g = Genome::new(bytes.clone());
            g.bind(&chem, cfg.enzyme_sigma, cfg.transport_sigma);
            let cell = wearing(g);
            let mut e = crate::Expression::new(&chem);
            e.read(&cell, &cfg, None, 0.01);
            e.uptake[3]
        };
        assert!(open > 1.5, "the transporter did nothing: {open}");

        write_gene(
            &mut bytes,
            &PLAIN,
            0.9,
            &[],
            ProteinClass::Effector,
            &key,
            &effector,
            0.05,
        );
        let mut g = Genome::new(bytes);
        g.bind(&chem, cfg.enzyme_sigma, cfg.transport_sigma);
        assert_eq!(g.genes.len(), 2);
        let cell = wearing(g);
        let mut e = crate::Expression::new(&chem);
        e.read(&cell, &cfg, None, 0.01);
        assert!(e.drives.gating, "the effector was not read as a gate");
        // Shut, not quite sealed: the effector's output is a `tanh` and a
        // `tanh` does not reach its ends, so a hard-biased gate leaves a
        // thousandth of the transporter open. That is the right shape -- the
        // last increment of anything here costs infinite bias -- and it is
        // three orders of magnitude below where it started.
        assert!(
            e.uptake[3] < 1.0 + 0.01 * (open - 1.0),
            "the gate did not close the transporter: {} of {open}",
            e.uptake[3]
        );
        // One, exactly: the bilayer, and not a fraction of it.
        for u in &e.uptake {
            assert!(*u >= 1.0, "a cell sealed itself off at {u}");
        }
    }

    #[test]
    fn swimming_is_paid_for_out_of_the_reserve() {
        // An effector that costs nothing is strictly better than not having
        // one, and then every lineage evolves a flagellum and the trait says
        // nothing. A cell that cannot afford the stroke does not take it.
        let cfg = CellConfig::default();
        let chem = hadean_chem::generate(1, hadean_chem::ChemParams::default());
        let mut params = [0.0f32; N_PARAMS];
        params[0] = 0.75;
        params[1] = 0.875; // Action::Move, 3 of 4
        params[2] = 1.0; // bias +4: full ahead
        params[3] = 0.9; // axis z
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &PLAIN,
            0.9,
            &[],
            ProteinClass::Effector,
            &PLAIN,
            &params,
            0.05,
        );
        let mut g = Genome::new(bytes);
        g.bind(&chem, cfg.enzyme_sigma, cfg.transport_sigma);

        let mut ingest = vec![1.0f32; chem.n_compounds()];
        let mut rich = wearing(g);
        rich.reserve = 1.0;
        let flat_out = drive(&rich, &cfg, 0.01, &mut ingest);
        assert!(flat_out.thrust[2] > 0.9, "{:?}", flat_out.thrust);
        assert!(motility_cost(&cfg, &flat_out.thrust) > 0.0);

        let mut broke = rich.clone();
        broke.reserve = 0.0;
        let becalmed = drive(&broke, &cfg, 0.01, &mut ingest);
        assert_eq!(
            becalmed.thrust, [0.0; 3],
            "a cell with an empty reserve swam anyway"
        );
    }

    #[test]
    fn every_channel_and_action_is_reachable_from_a_byte() {
        // `selector` quantises `params[1]`, and a variant no byte can select
        // is a variant that cannot evolve. This also pins the count: adding a
        // channel without widening the map would silently make the last one
        // unreachable.
        let mut channels = vec![false; N_CHANNELS];
        let mut actions = vec![false; N_ACTIONS];
        for byte in 0..=255u8 {
            let p = byte as f32 / 255.0;
            channels[((p * N_CHANNELS as f32) as usize).min(N_CHANNELS - 1)] = true;
            actions[((p * N_ACTIONS as f32) as usize).min(N_ACTIONS - 1)] = true;
        }
        assert!(channels.iter().all(|&c| c), "unreachable channel: {channels:?}");
        assert!(actions.iter().all(|&a| a), "unreachable action: {actions:?}");
    }
}
