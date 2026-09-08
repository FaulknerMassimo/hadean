//! L3 -- the genome.
//!
//! A cell's heritable state is a **linear byte string**, not a struct of
//! scalars. That choice is the whole of this module's reason for existing, and
//! `PLAN.md` argues it at length: a fixed struct supports only point mutation,
//! and point mutation alone cannot grow complexity. Duplication followed by
//! divergence is how biological complexity actually arose, and a
//! representation that cannot duplicate a gene will plateau no matter how the
//! rates are tuned.
//!
//! What replaced [`Traits`](crate::Traits) here is not a bigger struct. It is
//! the *indirection*: bytes decode into proteins, proteins act on the world,
//! and mutation edits the bytes. Nothing in the cell layer below reads a
//! genome field directly, so the set of things a lineage can become is bounded
//! by the protein classes rather than by a list of named dials.
//!
//! # Layout
//!
//! ```text
//!   ... junk ... [MARKER][motif][control][sites...][class][key][params][stab] ... junk ...
//! ```
//!
//! Decoding is a linear scan for the two-byte promoter marker. Everything
//! between genes is junk and is *not* compacted away: mutations there are
//! neutral, and neutral drift is what lets a population explore genotype space
//! without paying for it. It is also where new genes come from -- a duplication
//! that lands a marker in junk creates one.
//!
//! # The key vector
//!
//! Every matching operation in the simulation is a similarity between eight-
//! dimensional key vectors: enzyme to reaction, transporter to compound,
//! regulator to promoter. [`affinity`] is that function and there is only one
//! of it. Two properties follow, and both are load-bearing:
//!
//! 1. **Graded response.** A substitution usually moves a key component by a
//!    few 255ths of its range, which moves affinity a little, so the fitness
//!    landscape is traversable instead of a field of cliffs. This is a
//!    property of the *operator* as much as of the mapping -- see
//!    [`Draws::substitute`], where getting it wrong the first time cost the
//!    scheme the only thing it is for.
//! 2. **Promiscuity.** A protein binds several targets weakly. Duplicate it,
//!    let the copies drift, and each specialises -- which is how real protein
//!    families arise.
//!
//! # Two things a site can be
//!
//! Every gene carries up to three [`Site`]s: a motif, and a signed weight.
//! What a site *is* depends on the class of the gene that carries it, and
//! there is one rule:
//!
//! * on a gene whose protein does chemistry -- enzyme, transporter,
//!   structural, regulator -- a site is a **promoter binding site**. What
//!   binds there is a transcription factor, and the weight is how much that
//!   binding drives the gene. This is [`Targets::sites`], and it is the
//!   regulatory network.
//! * on a `Neural` or `Effector` gene, a site is a **dendrite**. What arrives
//!   there is the signal emitted by receptors and neurons, and the weight is
//!   synaptic. This is [`Targets::dendrites`], and it is the nervous system.
//!
//! One structure, two networks, and a mutation that flips a gene's class byte
//! moves its inputs from one to the other. The cost of the rule is that a
//! neuron's *expression* cannot itself be transcriptionally regulated -- it
//! runs at its basal level -- and that is the price of not extending the byte
//! format, which would have made every genome written before this
//! undecodable. A neuron's gain is still genomic: `basal` is six bits of the
//! control byte and `params[0]` is its strength.
//!
//! # What is not here
//!
//! `Adhesion` decodes and costs upkeep but does nothing: there is no cell-cell
//! physics for it to act through yet. It is not a stub to be filled in with
//! something else later -- its class code is part of the genome format, and
//! reserving it means an L5 genome stays readable by this decoder. Until then
//! it is exactly what a nonfunctional protein is in a real cell: a synthesis
//! bill with nothing on the other side, and therefore selected against.

use hadean_chem::chemistry::{Chemistry, CompoundId, Drive, ReactionId, KEY_DIM};
use hadean_core::hash::{HashState, StateHasher};
use hadean_core::rng::{Counter, Purpose};
use serde::{Deserialize, Serialize};

/// The two bytes that mark the start of a gene.
///
/// Chosen to be a pattern junk does not fall into often: one in 65536
/// positions, so a 600-byte genome carries about 0.01 spurious genes. Rare
/// enough that junk is really junk, common enough that a duplication or an
/// inversion can create a promoter where there was none.
const MARKER: [u8; 2] = [0xA5, 0x5A];

/// Bytes of a gene after its variable-length regulatory region.
const CODING_LEN: usize = 1 + KEY_DIM + N_PARAMS + 1;

/// Class-specific parameters carried by every protein.
pub const N_PARAMS: usize = 8;

/// Regulatory binding sites a single gene may carry.
///
/// Two bits of the control byte, so the bound is structural rather than a
/// check: a mutation cannot produce a gene claiming four thousand sites and
/// make the decoder walk off the end of the genome.
const MAX_SITES: usize = 3;

/// A genome may not grow past this. Not a biological statement -- a runaway
/// duplication chain would otherwise allocate the machine to death, exactly
/// as `population_cap` exists to stop a runaway population doing.
pub const MAX_GENOME: usize = 16_384;

/// What a protein does. Three bits, from the first byte of the coding region.
///
/// Discriminants are part of the genome format: a byte decodes to a class by
/// its low three bits, so **these may be appended to but never reordered**,
/// exactly like [`CellState`](crate::CellState) in a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProteinClass {
    /// Lowers the barrier for reactions whose key is close to its own. The
    /// class that decides what a cell eats.
    Enzyme = 0,
    /// Raises membrane permeability for compounds whose key is close to its
    /// own. What the old `Traits::uptake` was, made specific to a compound.
    Transporter = 1,
    /// Membrane and cytoskeleton. Buys famine tolerance at an upkeep cost.
    Structural = 2,
    /// L5. Binds cells with matching keys.
    Adhesion = 3,
    /// Senses one channel of the world or of the cell itself and emits what it
    /// finds as a signal at its own key. `params[1]` picks the channel; for
    /// the chemical channels the key also picks the compound, exactly as a
    /// transporter's does. See [`crate::neural::Channel`].
    Receptor = 4,
    /// Acts. `params[1]` picks which of the things a cell can physically do it
    /// drives -- shutting down, dividing, opening a transporter, swimming --
    /// and its dendrites decide how hard. See [`crate::neural::Action`].
    Effector = 5,
    /// Transcription factor: binds promoters, so genes can regulate genes.
    Regulator = 6,
    /// A neuron. Sums its dendrites, squashes, and emits the result at its own
    /// key, carrying a fraction of last tick's activation forward -- so a
    /// network of them has memory, and hysteresis is something a lineage
    /// evolves rather than something a config file sets.
    Neural = 7,
}

impl ProteinClass {
    fn from_byte(b: u8) -> Self {
        match b & 0b111 {
            0 => Self::Enzyme,
            1 => Self::Transporter,
            2 => Self::Structural,
            3 => Self::Adhesion,
            4 => Self::Receptor,
            5 => Self::Effector,
            6 => Self::Regulator,
            _ => Self::Neural,
        }
    }

    /// Whether this class has anything to act on in the world as it stands.
    /// An inert protein still costs its synthesis and upkeep.
    pub fn is_active(self) -> bool {
        self != Self::Adhesion
    }

    /// Whether this class puts a value onto the signal network. Receptors emit
    /// what they sense and neurons emit what they compute; nothing else is
    /// heard by a dendrite.
    pub fn emits_signal(self) -> bool {
        matches!(self, Self::Receptor | Self::Neural)
    }

    /// Whether this protein is part of the cell's nervous system rather than
    /// its metabolism -- something it decides *with* rather than something it
    /// lives *on*.
    pub fn is_signal(self) -> bool {
        matches!(self, Self::Receptor | Self::Neural | Self::Effector)
    }

    /// Whether this gene's [`Site`]s are dendrites rather than promoter
    /// binding sites. See the module docs: it is the same structure read two
    /// ways, and the class byte is what decides which.
    pub fn listens(self) -> bool {
        matches!(self, Self::Neural | Self::Effector)
    }
}

/// One regulatory binding site: a motif a transcription factor recognises, and
/// what binding there does. Positive activates, negative represses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Site {
    pub motif: [f32; KEY_DIM],
    pub weight: f32,
}

/// One decoded gene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gene {
    /// Where the gene's marker sits in the byte string. Mutation operators
    /// work on spans, so duplication and deletion need this.
    pub at: usize,
    pub len: usize,
    /// The promoter motif regulators bind to.
    pub promoter: [f32; KEY_DIM],
    pub sites: Vec<Site>,
    /// Expression with no regulator bound, 0..1.
    pub basal: f32,
    pub class: ProteinClass,
    pub key: [f32; KEY_DIM],
    pub params: [f32; N_PARAMS],
    /// Fraction of the standing protein lost per second.
    pub stability: f32,
}

impl Gene {
    /// The first class parameter, on a useful scale rather than 0..1.
    ///
    /// Every active class reads `params[0]` as "how strongly", so the mapping
    /// lives in one place. Quadratic, so most of the byte range sits at low
    /// strength and a strong protein is something mutation has to find.
    pub fn strength(&self) -> f32 {
        let p = self.params[0];
        4.0 * p * p
    }

    /// `params[1]`, quantised onto `0..n`. What a receptor senses and what an
    /// effector does are both a choice out of a fixed list of things a cell
    /// physically has -- eyes, a membrane, a flagellum -- and this is how the
    /// genome makes it. The *list* is physiology and lives in
    /// [`crate::neural`]; which entry a gene picks is genetic.
    ///
    /// A byte spans `n` selections, so a substitution steps between adjacent
    /// choices only near a boundary. That is deliberate: a receptor should not
    /// change what it looks at every time a base flips.
    pub fn selector(&self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.params[1] * n as f32) as usize).min(n - 1)
    }

    /// `params[2]` as a signed bias on `-4..4`: what a neuron or an effector
    /// does with no input at all. The same range as a [`Site`] weight, so a
    /// single dendrite can exactly cancel a bias.
    pub fn bias(&self) -> f32 {
        (self.params[2] - 0.5) * 8.0
    }

    /// `params[3]` as a neuron's rate of forgetting, per second, logarithmic
    /// over `0.1..20`.
    ///
    /// This is where sleeping deeply-but-briefly and lightly-but-long come
    /// from, and the reason a genome cell does not read
    /// [`CellConfig::dormancy_exit`](crate::CellConfig::dormancy_exit) --
    /// which is one hysteresis for the whole pond, and a pond with one
    /// hysteresis cannot have some cells riding out a shortage that others
    /// give up on. A neuron relaxes towards what its dendrites say at
    /// this rate, so a slow one holds its state across a shortage the way a
    /// fast one cannot. Logarithmic because the useful range spans a time
    /// constant of ten seconds to fifty milliseconds, and a linear map would
    /// spend nine tenths of the byte range on the fast end.
    pub fn leak(&self) -> f32 {
        0.1 * 200.0f32.powf(self.params[3])
    }

    /// `params[3]` as an axis, for the effectors that need one.
    pub fn axis(&self) -> usize {
        ((self.params[3] * 3.0) as usize).min(2)
    }
}

/// What one gene's protein can act on in this particular chemistry.
///
/// Resolved once per genome rather than per cell per tick. A cell steps a
/// hundred times a second and its genome does not change between divisions,
/// so matching every enzyme against every reaction in the inner loop would be
/// paying an exponential per reaction per cell for an answer that is the same
/// every time. Daughters share their mother's tables through the same
/// [`Arc`](std::sync::Arc) that carries her bytes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Targets {
    /// Reactions an enzyme catalyses, and how well.
    pub reactions: Vec<(ReactionId, f32)>,
    /// Compounds a transporter carries, and how well.
    pub compounds: Vec<(CompoundId, f32)>,
    /// For each of this gene's regulatory sites, which genes' regulator
    /// proteins bind there and how strongly, by gene index, ascending.
    ///
    /// Static in the genome, exactly like the two above: which regulator
    /// recognises which promoter is a fact about the byte string and not about
    /// the cell. Computing it per cell per tick makes transcription quadratic
    /// in gene count with an exponential inside, which is affordable at the
    /// ancestor's six genes and is not the point -- a genome that duplicates
    /// its way to thirty would be paying it a thousand times a second per
    /// cell.
    ///
    /// Empty on a gene whose sites are dendrites -- see the module docs.
    pub sites: Vec<Vec<(usize, f32)>>,
    /// The other reading of the same sites: for each dendrite of a `Neural` or
    /// `Effector` gene, which receptors and neurons it hears and how loudly,
    /// by gene index, ascending.
    ///
    /// Resolved here for the same reason `sites` is -- who is wired to whom is
    /// a fact about the byte string, not about the cell, and a cell steps a
    /// hundred times a second. What changes per tick is the *activation*
    /// travelling down these wires, and that is per-cell state.
    ///
    /// Empty on a gene whose sites are promoter binding sites.
    pub dendrites: Vec<Vec<(usize, f32)>>,
}

/// Affinities below this are not worth carrying in a target list. A protein
/// with a hundredth of its peak activity on some distant reaction is not
/// catalysing it in any sense the population will ever notice.
const AFFINITY_FLOOR: f32 = 0.01;

/// A byte string, and what it decodes to.
///
/// Genes are decoded once, when the genome is built, and never again: a cell
/// steps a hundred times a second and its genome does not change between
/// divisions. Daughters share their mother's allocation through an
/// [`Arc`](std::sync::Arc) and only pay for a decode when a mutation actually
/// lands, which for most divisions it does not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub bytes: Vec<u8>,
    /// Decoded on construction. Not serialised: it is a pure function of
    /// `bytes`, and writing it into a snapshot would be a second copy of the
    /// same information that could disagree with the first.
    #[serde(skip)]
    pub genes: Vec<Gene>,
    /// One entry per gene, in the same order. Empty until [`bind`](Self::bind)
    /// has matched the proteins against a chemistry.
    #[serde(skip)]
    pub targets: Vec<Targets>,
}

impl Genome {
    /// Decode a byte string. The proteins are known; what they act on is not
    /// until [`bind`](Self::bind) is given a chemistry.
    pub fn new(bytes: Vec<u8>) -> Self {
        let genes = decode(&bytes);
        Self {
            bytes,
            genes,
            targets: Vec::new(),
        }
    }

    /// Decode and match against a chemistry in one step. This is what the
    /// cell layer uses; [`new`](Self::new) exists for decoding on its own.
    pub fn expressed(
        bytes: Vec<u8>,
        chem: &Chemistry,
        enzyme_sigma: f32,
        transport_sigma: f32,
    ) -> Self {
        let mut g = Self::new(bytes);
        g.bind(chem, enzyme_sigma, transport_sigma);
        g
    }

    /// How hard this genome's own proteins push its replication fidelity,
    /// on `-1..1`.
    ///
    /// Negative is more faithful, positive is more error-prone. What an octave
    /// of that is worth in actual rate is
    /// [`CellConfig::mutator_range`](crate::CellConfig::mutator_range), which
    /// is physiology; how far the cell pushes is genetic, and this is it.
    ///
    /// A `Regulator` whose `selector(2)` is 1 is a **replication factor** as
    /// well as a transcription factor -- it still binds promoters, exactly as
    /// before, and additionally leans on the polymerase. Additionally rather
    /// than instead, because taking half the regulators out of the regulatory
    /// network to pay for this would be a change to gene regulation dressed up
    /// as a change to mutation. A transcription factor that also upregulates
    /// an error-prone polymerase is what an SOS response is.
    ///
    /// The direction and the magnitude both come from [`Gene::bias`], and
    /// nothing else, so there is one number in the genome to select on. The
    /// sum is clamped rather than averaged: two mutator alleles pushing the
    /// same way should push harder than one, up to the physiological limit.
    ///
    /// A genome with no replication factors returns zero and therefore
    /// replicates at exactly the configured rates, which is what every run
    /// before this mechanism existed did.
    pub fn mutator_drive(&self, proteome: &[f32]) -> f32 {
        let mut drive = 0.0;
        for (i, gene) in self.genes.iter().enumerate() {
            if gene.class != ProteinClass::Regulator || gene.selector(2) == 0 {
                continue;
            }
            // `bias` spans -4..4 and a concentration spans 0..1, so one
            // saturated allele at full expression covers the whole range.
            drive += proteome.get(i).copied().unwrap_or(0.0) * gene.bias() * 0.25;
        }
        drive.clamp(-1.0, 1.0)
    }

    /// Genes that lean on the polymerase. For telemetry and for tests.
    pub fn mutator_genes(&self) -> usize {
        self.genes
            .iter()
            .filter(|g| g.class == ProteinClass::Regulator && g.selector(2) == 1)
            .count()
    }

    /// Work out what each protein acts on, and how strongly.
    ///
    /// Enzymes match reactions and transporters match compounds, both through
    /// the same [`affinity`]. Receptors and effectors match compounds too --
    /// a chemoreceptor tastes what a transporter would carry, and by the same
    /// rule, so a lineage that evolves a transporter for something has already
    /// evolved most of a sensor for it. Targets are sorted by id, not by
    /// affinity: the cell layer applies reactions in id order, and that order
    /// is part of the replay contract.
    ///
    /// Then two passes that cannot happen in the same loop, because both are
    /// gene-to-gene and need every gene decoded first: promoter binding, and
    /// the wiring of the signal network.
    pub fn bind(&mut self, chem: &Chemistry, enzyme_sigma: f32, transport_sigma: f32) {
        let scale = KeyScale::of(chem);
        self.targets = self
            .genes
            .iter()
            .map(|gene| {
                let mut t = Targets::default();
                match gene.class {
                    ProteinClass::Enzyme => {
                        for r in &chem.reactions {
                            // A photochemical reaction is driven by a photon,
                            // not by a barrier an enzyme could lower, and a
                            // thermoneutral one has nothing to give a cell.
                            if r.drive != Drive::Thermal || r.dh == 0.0 {
                                continue;
                            }
                            let a = affinity(&gene.key, &scale.normalise(&r.key), enzyme_sigma);
                            if a >= AFFINITY_FLOOR {
                                t.reactions.push((r.id, a));
                            }
                        }
                        t.reactions.sort_by_key(|&(id, _)| id);
                    }
                    ProteinClass::Transporter | ProteinClass::Receptor | ProteinClass::Effector => {
                        for c in &chem.compounds {
                            let a = affinity(&gene.key, &scale.normalise(&c.key), transport_sigma);
                            if a >= AFFINITY_FLOOR {
                                t.compounds.push((c.id, a));
                            }
                        }
                        t.compounds.sort_by_key(|&(id, _)| id);
                    }
                    _ => {}
                }
                t
            })
            .collect();

        // Promoter binding and neural wiring. Both are gene-to-gene, so
        // neither can be resolved in the pass above: each needs every gene
        // decoded before any gene's sites can be matched against them.
        //
        // A gene's sites go to exactly one of the two, decided by its class.
        // A wire whose affinity is below the floor is not carried at all --
        // the same rule as an enzyme too far from a reaction to be catalysing
        // it, and the thing that keeps a thirty-gene genome from paying for a
        // dense matrix a hundred times a second.
        let regulators: Vec<(usize, &Gene)> = self
            .genes
            .iter()
            .enumerate()
            .filter(|(_, g)| g.class == ProteinClass::Regulator)
            .collect();
        let sources: Vec<(usize, &Gene)> = self
            .genes
            .iter()
            .enumerate()
            .filter(|(_, g)| g.class.emits_signal())
            .collect();
        for (gene, targets) in self.genes.iter().zip(&mut self.targets) {
            let wire = |from: &[(usize, &Gene)], motif: &[f32; KEY_DIM], sigma: f32| {
                from.iter()
                    .filter_map(|&(j, g)| {
                        let a = affinity(&g.key, motif, sigma);
                        (a >= AFFINITY_FLOOR).then_some((j, a))
                    })
                    .collect::<Vec<_>>()
            };
            if gene.class.listens() {
                targets.dendrites = gene
                    .sites
                    .iter()
                    .map(|site| wire(&sources, &site.motif, SIGNAL_SIGMA))
                    .collect();
            } else {
                targets.sites = gene
                    .sites
                    .iter()
                    .map(|site| wire(&regulators, &site.motif, REGULATOR_SIGMA))
                    .collect();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Genes that do something in the world as it stands.
    pub fn active_genes(&self) -> usize {
        self.genes.iter().filter(|g| g.class.is_active()).count()
    }

    /// Rebuild the decode after a load. Snapshots carry bytes only.
    pub fn redecode(&mut self, chem: &Chemistry, enzyme_sigma: f32, transport_sigma: f32) {
        self.genes = decode(&self.bytes);
        self.bind(chem, enzyme_sigma, transport_sigma);
    }
}

impl HashState for Genome {
    fn hash_state(&self, h: &mut StateHasher) {
        h.usize(self.bytes.len());
        for &b in &self.bytes {
            h.byte(b);
        }
    }
}

/// Scan a byte string for promoter markers and decode what follows each one.
///
/// A gene that runs off the end of the string is simply not there, which is
/// what a truncating deletion does to a real one. The scan resumes *after* a
/// decoded gene rather than inside it, so a marker pattern that happens to
/// occur within a key cannot spawn a second overlapping gene -- overlapping
/// reading frames are a real phenomenon but they make mutation effects
/// non-local, which is the opposite of what the graded landscape needs.
fn decode(bytes: &[u8]) -> Vec<Gene> {
    let mut genes = Vec::new();
    let mut i = 0usize;
    while i + 2 <= bytes.len() {
        if bytes[i] != MARKER[0] || bytes[i + 1] != MARKER[1] {
            i += 1;
            continue;
        }
        let start = i;
        let mut at = i + 2;

        // Promoter motif.
        let Some(promoter) = read_key(bytes, at) else {
            break;
        };
        at += KEY_DIM;

        // Control byte: two bits of site count, six of basal expression.
        let Some(&control) = bytes.get(at) else {
            break;
        };
        at += 1;
        let n_sites = (control & 0b11) as usize;
        debug_assert!(n_sites <= MAX_SITES);
        let basal = (control >> 2) as f32 / 63.0;

        // Regulatory region.
        let mut sites = Vec::with_capacity(n_sites);
        let mut truncated = false;
        for _ in 0..n_sites {
            let Some(motif) = read_key(bytes, at) else {
                truncated = true;
                break;
            };
            at += KEY_DIM;
            let Some(&w) = bytes.get(at) else {
                truncated = true;
                break;
            };
            at += 1;
            sites.push(Site {
                motif,
                // Signed: half the byte range represses.
                weight: (w as f32 / 255.0) * 8.0 - 4.0,
            });
        }
        if truncated || at + CODING_LEN > bytes.len() {
            break;
        }

        let class = ProteinClass::from_byte(bytes[at]);
        at += 1;
        let key = read_key(bytes, at).expect("length checked above");
        at += KEY_DIM;
        let mut params = [0.0f32; N_PARAMS];
        for (p, slot) in params.iter_mut().enumerate() {
            *slot = bytes[at + p] as f32 / 255.0;
        }
        at += N_PARAMS;
        // Never zero: a protein that never degrades would make expression a
        // ratchet, and a cell could not stop making something it had started.
        let stability = 0.002 + (bytes[at] as f32 / 255.0) * 0.2;
        at += 1;

        genes.push(Gene {
            at: start,
            len: at - start,
            promoter,
            sites,
            basal,
            class,
            key,
            params,
            stability,
        });
        i = at;
    }
    genes
}

/// Eight bytes to a key vector on `0..1`.
///
/// Normalised space, not chemistry space: see [`KeyScale`] for what happens
/// when a genome assumes it knows the range of the thing it is matching
/// against. One byte is 1/255 of a component's range whatever that range is,
/// so a substitution means the same size of step in every dimension.
fn read_key(bytes: &[u8], at: usize) -> Option<[f32; KEY_DIM]> {
    let slice = bytes.get(at..at + KEY_DIM)?;
    let mut key = [0.0f32; KEY_DIM];
    for (k, slot) in key.iter_mut().enumerate() {
        *slot = slice[k] as f32 / 255.0;
    }
    Some(key)
}

/// The inverse, for writing an ancestor whose proteins match a real target.
fn write_key(out: &mut Vec<u8>, key: &[f32; KEY_DIM]) {
    for &k in key {
        out.push((k.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
}

/// The map between a genome's key bytes and a chemistry's key vectors.
///
/// This exists because the first version of this module did not have it, and
/// was wrong. Bytes were mapped onto a fixed `[-2, 2]`, on the reasoning that
/// [`structural_key`](hadean_chem) is built out of tanh-squashed ratios and
/// small counts. Seven of its eight components are indeed inside that. The
/// eighth is formation enthalpy per atom over 1e-19 J, and on `gate.toml` it
/// runs from -9.04 to -1.49 -- so every ancestor's enzyme key was clamped to
/// -2, sat seven units away from the reaction it was written to catalyse, and
/// matched nothing. Sixteen founders arrived in a full pond and ate not one
/// particle.
///
/// The fix is to stop assuming and read the range off the chemistry. Every
/// component is mapped onto `0..1` by the span that chemistry actually uses,
/// which does two things: a key byte means the same fraction of the available
/// range in every dimension, and affinity stops being dominated by whichever
/// component happens to have the widest units. A dimension the chemistry never
/// varies -- ring count is zero for every compound on this seed -- collapses
/// to a constant and drops out of the distance instead of contributing noise.
///
/// A pure function of the chemistry, which is a pure function of the seed, so
/// this is as reproducible as everything else here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyScale {
    lo: [f32; KEY_DIM],
    span: [f32; KEY_DIM],
}

impl KeyScale {
    pub fn of(chem: &Chemistry) -> Self {
        let mut lo = [f32::INFINITY; KEY_DIM];
        let mut hi = [f32::NEG_INFINITY; KEY_DIM];
        let keys = chem
            .compounds
            .iter()
            .map(|c| &c.key)
            .chain(chem.reactions.iter().map(|r| &r.key));
        for key in keys {
            for k in 0..KEY_DIM {
                lo[k] = lo[k].min(key[k]);
                hi[k] = hi[k].max(key[k]);
            }
        }
        let mut span = [1.0f32; KEY_DIM];
        for k in 0..KEY_DIM {
            if lo[k].is_finite() && hi[k] > lo[k] {
                span[k] = hi[k] - lo[k];
            } else {
                // A component this chemistry does not vary. Pin it so every
                // key normalises to the same value and it drops out.
                lo[k] = 0.0;
                span[k] = 1.0;
            }
        }
        Self { lo, span }
    }

    /// A chemistry key in the genome's own `0..1` space.
    pub fn normalise(&self, key: &[f32; KEY_DIM]) -> [f32; KEY_DIM] {
        let mut out = [0.0f32; KEY_DIM];
        for k in 0..KEY_DIM {
            out[k] = ((key[k] - self.lo[k]) / self.span[k]).clamp(0.0, 1.0);
        }
        out
    }
}

/// Width of a transcription factor's recognition of a promoter.
///
/// Not a config dial, deliberately. Enzyme and transporter widths are, because
/// they decide what a lineage can reach in the chemistry and that is a
/// question about the world; this one only decides how sharply a regulator
/// picks out one promoter from another, and nothing in the world as it stands
/// turns on the answer. It becomes a dial when there is a body to
/// differentiate.
pub const REGULATOR_SIGMA: f32 = 0.15;

/// Width of a dendrite's recognition of a signal.
///
/// A not-a-config-dial for the same reason [`REGULATOR_SIGMA`] is: it decides
/// how sharply one neuron picks out another from the crowd, and nothing about
/// the world turns on the answer. It is the same width, and deliberately so --
/// a promoter site and a dendrite are the same eight bytes read twice, and
/// giving them different widths would mean a gene's inputs silently rewired
/// themselves when a mutation changed its class.
pub const SIGNAL_SIGMA: f32 = REGULATOR_SIGMA;

/// The one matching function in the simulation.
///
/// `exp(-||a - b||^2 / sigma^2)`. `sigma` is the width of a protein's
/// recognition: small is a specialist that binds one target, large is a
/// generalist that binds many weakly.
pub fn affinity(a: &[f32; KEY_DIM], b: &[f32; KEY_DIM], sigma: f32) -> f32 {
    let mut d2 = 0.0f32;
    for k in 0..KEY_DIM {
        let d = a[k] - b[k];
        d2 += d * d;
    }
    (-d2 / (sigma * sigma).max(1.0e-6)).exp()
}

// ---------------------------------------------------------------------------
// Writing an ancestor
// ---------------------------------------------------------------------------

/// Assemble the bytes of one gene.
///
/// Public because [`ancestor`] is one caller and not the only legitimate one:
/// this is how a genome is *written* rather than evolved, and the tests that
/// check an interpreter need to be able to hand it a protein of a known shape
/// rather than mutating until one appears.
#[allow(clippy::too_many_arguments)]
pub fn write_gene(
    out: &mut Vec<u8>,
    promoter: &[f32; KEY_DIM],
    basal: f32,
    sites: &[([f32; KEY_DIM], f32)],
    class: ProteinClass,
    key: &[f32; KEY_DIM],
    params: &[f32; N_PARAMS],
    stability: f32,
) {
    assert!(sites.len() <= MAX_SITES);
    out.extend_from_slice(&MARKER);
    write_key(out, promoter);
    let basal_bits = (basal.clamp(0.0, 1.0) * 63.0).round() as u8;
    out.push((basal_bits << 2) | sites.len() as u8);
    for (motif, weight) in sites {
        write_key(out, motif);
        out.push((((weight + 4.0) / 8.0).clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    out.push(class as u8);
    write_key(out, key);
    for &p in params {
        out.push((p.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    out.push((((stability - 0.002) / 0.2).clamp(0.0, 1.0) * 255.0).round() as u8);
}

/// Junk. Neutral sequence between genes, drawn from the world seed so it is
/// part of the replay contract like everything else.
fn write_junk(out: &mut Vec<u8>, rng: &Counter, entity: u64, stream: u64, n: usize) {
    for j in 0..n {
        let bits = rng.bits(0, entity, Purpose::Mutation, stream + j as u64);
        let b = (bits >> 24) as u8;
        // Never emit a marker by accident: junk at t = 0 should be junk.
        out.push(if b == MARKER[0] { b ^ 0x0F } else { b });
    }
}

/// The hand-written ancestor, matched to the world it is dropped into.
///
/// `PLAN.md` is emphatic that a random byte string will not decode into
/// anything that lives, and every digital-evolution system that worked seeded
/// an ancestor by hand. This is that ancestor. What is hand-written is the
/// *structure* -- which genes exist and what each is for. The keys cannot be,
/// because the chemistry is generated per seed: an enzyme's key is copied from
/// the reaction it is meant to run, and a transporter's from the compound it
/// is meant to take up. So the ancestor arrives already able to make the
/// living that the hardcoded protocell made, and mutation starts from a cell
/// that works rather than from noise.
///
/// The metabolism:
///
/// * one **enzyme** on `reaction`, the ancestral metabolism;
/// * one **transporter** per substrate of that reaction, so it can get the
///   food in;
/// * one **transporter** for the product, so it can get the waste out;
/// * one **structural** protein, so famine tolerance is a thing a lineage can
///   trade against rather than a constant;
/// * one **regulator**, expressed constitutively and bound to the enzyme's
///   promoter. It does nothing useful on day one and is deliberately included:
///   a regulatory network cannot evolve out of nothing, and this is the seed
///   crystal for one.
///
/// And the nervous system -- one arc, three synapses long:
///
/// * a **receptor** on [`Channel::Energy`](crate::neural::Channel::Energy),
///   emitting at `hunger`. How much reserve there is, as a fraction of what
///   would see the cell through a famine.
/// * a **neuron** listening at `hunger` with a weight of -4 and a bias of
///   +0.6, emitting at `sleep`. It says *shut down*, and it says it when the
///   reserve falls below about fifty seconds of upkeep. Its rate of forgetting
///   is a third of a second's worth per second, so it holds an opinion for a
///   few seconds rather than re-forming one every tick -- which is the job
///   `dormancy_exit` used to do, done by something heritable.
/// * an **effector** on `Action::Quiesce` listening at `sleep`, weight +4.
/// * an **effector** on `Action::Divide` listening at `sleep`, weight -4, bias
///   +1: divide freely, unless the same signal that shuts the cell down says
///   otherwise.
///
/// This circuit is *approximately* the rule it replaces -- shut down around
/// fifty seconds of banked upkeep, which is where `dormancy_exit = 60` put the
/// old wake bar -- and that is on purpose. The claim being tested is not that
/// a hand-written network is cleverer than a hand-written threshold. It is
/// that a network is made of parts that mutate: a weight, a bias, a rate of
/// forgetting, a channel, and which effector hears what. Sixteen founders of
/// it are sixteen different opinions about when to sleep and how deeply, and
/// the pond keeps whichever were right.
///
/// `hunger` and `sleep` are two arbitrary fixed addresses in key space, chosen
/// about two units apart so that at [`SIGNAL_SIGMA`] neither wire hears the
/// other's traffic. Nothing stops a mutation from moving one until they do.
pub fn ancestor(chem: &Chemistry, reaction: ReactionId, rng: &Counter, entity: u64) -> Genome {
    let r = chem.reaction(reaction);
    let scale = KeyScale::of(chem);
    let mut out = Vec::with_capacity(512);
    let mut stream = 1000u64;

    // A promoter motif the regulator will recognise. Any vector will do; this
    // one is arbitrary and fixed, so the ancestor is the same shape on every
    // seed.
    let regulated: [f32; KEY_DIM] = [0.7, 0.2, 0.6, 0.1, 0.4, 0.9, 0.3, 0.55];
    let plain: [f32; KEY_DIM] = [0.5; KEY_DIM];
    // The two signal addresses. Far apart in key space -- eight components at
    // 0.7 of separation each -- so `affinity` between them is exp(-174) and a
    // dendrite on one hears nothing of the other.
    let hunger: [f32; KEY_DIM] = [0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55];
    let sleep: [f32; KEY_DIM] = [0.85, 0.15, 0.75, 0.25, 0.65, 0.35, 0.55, 0.45];

    write_junk(&mut out, rng, entity, stream, 24);
    stream += 24;

    // The metabolism. Expressed hard, and weakly activated by the regulator.
    let mut enzyme_params = [0.0f32; N_PARAMS];
    enzyme_params[0] = 0.5; // strength() = 1.0: exactly `metabolic_rate`.
    write_gene(
        &mut out,
        &regulated,
        0.9,
        &[(regulated, 1.0)],
        ProteinClass::Enzyme,
        &scale.normalise(&r.key),
        &enzyme_params,
        0.02,
    );

    write_junk(&mut out, rng, entity, stream, 16);
    stream += 16;

    // Transporters: one per substrate, one for the first product. Their keys
    // are the compounds' own, so affinity is 1 for the target and falls away
    // for everything else -- the ancestor is a specialist, and generalists are
    // something a lineage has to evolve into.
    let mut carried: Vec<CompoundId> = r.reactants.iter().map(|&(c, _)| c).collect();
    if let Some(&(product, _)) = r.products.first() {
        carried.push(product);
    }
    let mut transporter_params = [0.0f32; N_PARAMS];
    transporter_params[0] = 0.5;
    for c in carried {
        write_gene(
            &mut out,
            &plain,
            0.8,
            &[],
            ProteinClass::Transporter,
            &scale.normalise(&chem.compounds[c as usize].key),
            &transporter_params,
            0.02,
        );
        write_junk(&mut out, rng, entity, stream, 12);
        stream += 12;
    }

    // Structure. Modest: enough to matter, not enough to carry the cell.
    let mut structural_params = [0.0f32; N_PARAMS];
    structural_params[0] = 0.35;
    write_gene(
        &mut out,
        &plain,
        0.6,
        &[],
        ProteinClass::Structural,
        &plain,
        &structural_params,
        0.01,
    );

    write_junk(&mut out, rng, entity, stream, 16);
    stream += 16;

    // The seed of a regulatory network.
    write_gene(
        &mut out,
        &plain,
        0.5,
        &[],
        ProteinClass::Regulator,
        &regulated,
        &[0.0; N_PARAMS],
        0.05,
    );

    write_junk(&mut out, rng, entity, stream, 16);
    stream += 16;

    // ---- the nervous system ------------------------------------------------
    //
    // Params are positioned, not named, and the positions are the same for
    // every class: [0] is strength, [1] is which channel or which action, [2]
    // is bias, [3] is a rate or an axis. See `Gene::selector`, `Gene::bias`
    // and `Gene::leak` for the maps; the constants below are their inverses.

    // Hunger. `Channel::Energy` is 2 of 8, so params[1] sits in the middle of
    // the third eighth. Strength 1.5 against a basal of 0.7 opens it fully.
    let mut receptor_params = [0.0f32; N_PARAMS];
    receptor_params[0] = 0.61;
    receptor_params[1] = 0.3125;
    write_gene(
        &mut out,
        &plain,
        0.7,
        &[],
        ProteinClass::Receptor,
        &hunger,
        &receptor_params,
        0.05,
    );

    write_junk(&mut out, rng, entity, stream, 12);
    stream += 12;

    // The decision. bias +0.6, so tanh(0.6 - 4 * hunger) crosses zero at a
    // reserve of about fifty seconds of upkeep; leak 0.3/s, so it takes a
    // few seconds to change its mind.
    let mut neuron_params = [0.0f32; N_PARAMS];
    neuron_params[0] = 0.61;
    neuron_params[2] = 0.575;
    neuron_params[3] = 0.21;
    write_gene(
        &mut out,
        &plain,
        0.7,
        &[(hunger, -4.0)],
        ProteinClass::Neural,
        &sleep,
        &neuron_params,
        0.05,
    );

    write_junk(&mut out, rng, entity, stream, 12);
    stream += 12;

    // Shut down when the neuron says so.
    let mut quiesce_params = [0.0f32; N_PARAMS];
    quiesce_params[0] = 0.56;
    quiesce_params[1] = 0.125;
    quiesce_params[2] = 0.5;
    write_gene(
        &mut out,
        &plain,
        0.8,
        &[(sleep, 4.0)],
        ProteinClass::Effector,
        &plain,
        &quiesce_params,
        0.05,
    );

    write_junk(&mut out, rng, entity, stream, 12);
    stream += 12;

    // And do not spend a reserve on a daughter while doing it.
    let mut divide_params = [0.0f32; N_PARAMS];
    divide_params[0] = 0.56;
    divide_params[1] = 0.375;
    divide_params[2] = 0.625;
    write_gene(
        &mut out,
        &plain,
        0.8,
        &[(sleep, -4.0)],
        ProteinClass::Effector,
        &plain,
        &divide_params,
        0.05,
    );

    write_junk(&mut out, rng, entity, stream, 32);

    Genome::new(out)
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

/// Per-replication mutation rates. `PLAN.md` gives the table; these are its
/// defaults, and every one of them is meant to be turned.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MutationRates {
    /// Substitutions per byte. Fine-tuning, and the operator that makes the
    /// landscape graded.
    pub point: f32,
    /// Insertions or deletions of one to four bytes, per byte. Inside a gene
    /// this shifts the reading frame and usually destroys it, which is what a
    /// frameshift does.
    pub indel: f32,
    /// Duplications per gene. **The engine of complexity**: a duplicate is
    /// free to drift while the original goes on doing its job, and that is how
    /// a lineage gets a second enzyme without losing the first.
    pub duplication: f32,
    /// Deletions per gene. Streamlining -- the pressure that removes a protein
    /// whose upkeep is no longer earning.
    pub deletion: f32,
    /// Inversions per genome. Regulatory rewiring, and the operator most
    /// likely to create a promoter where there was none.
    pub inversion: f32,
    /// Whole-genome duplications per genome. Rare and huge.
    pub genome_duplication: f32,
}

impl Default for MutationRates {
    fn default() -> Self {
        Self {
            point: 1.0e-3,
            indel: 1.0e-5,
            duplication: 1.0e-3,
            deletion: 1.0e-3,
            inversion: 1.0e-5,
            genome_duplication: 1.0e-6,
        }
    }
}

impl MutationRates {
    /// Every operator off. A lineage under these is a clone line for ever,
    /// which is the control every claim about evolution here is measured
    /// against.
    pub fn none() -> Self {
        Self {
            point: 0.0,
            indel: 0.0,
            duplication: 0.0,
            deletion: 0.0,
            inversion: 0.0,
            genome_duplication: 0.0,
        }
    }

    /// Every operator scaled by one factor, clamped where `validate` would
    /// have refused the result.
    ///
    /// This is how a mutator allele acts: it moves the whole table together
    /// rather than one operator, because a cell's replication fidelity is one
    /// property of one polymerase and not six independent dials. Zero stays
    /// zero under any factor, so [`none`](Self::none) is still a clone line
    /// however hard a genome leans on it -- which keeps the control a control.
    pub fn scaled(&self, factor: f32) -> Self {
        let f = factor.max(0.0);
        // The same ceiling `validate` applies, so a genome cannot mutate its
        // way to a rate the config layer would have rejected outright.
        let per_byte = |r: f32| (r * f).min(0.5);
        Self {
            point: per_byte(self.point),
            indel: per_byte(self.indel),
            duplication: self.duplication * f,
            deletion: self.deletion * f,
            inversion: self.inversion * f,
            genome_duplication: self.genome_duplication * f,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let all = [
            self.point,
            self.indel,
            self.duplication,
            self.deletion,
            self.inversion,
            self.genome_duplication,
        ];
        if all.iter().any(|r| !r.is_finite() || *r < 0.0) {
            return Err("mutation rates must be finite and non-negative".into());
        }
        if self.point > 0.5 || self.indel > 0.5 {
            return Err("per-byte mutation rates above 0.5 are an error, not a fast run".into());
        }
        Ok(())
    }

    fn any(&self) -> bool {
        self.point > 0.0
            || self.indel > 0.0
            || self.duplication > 0.0
            || self.deletion > 0.0
            || self.inversion > 0.0
            || self.genome_duplication > 0.0
    }
}

/// A counter-based draw stream for one replication event.
///
/// Mutation needs a variable number of draws and the count itself is random,
/// which is exactly the situation [`Counter`] exists to keep deterministic:
/// every draw is a pure function of `(tick, parent, Purpose::Mutation, n)`, so
/// the same division produces the same daughter whatever order the population
/// was stepped in or how many threads were involved.
struct Draws<'a> {
    rng: &'a Counter,
    tick: u64,
    parent: u64,
    n: u64,
}

impl Draws<'_> {
    fn unit(&mut self) -> f32 {
        self.n += 1;
        self.rng.unit(self.tick, self.parent, Purpose::Mutation, self.n)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        self.n += 1;
        let bits = self.rng.bits(self.tick, self.parent, Purpose::Mutation, self.n);
        (bits % n as u64) as usize
    }

    fn byte(&mut self) -> u8 {
        self.n += 1;
        (self.rng.bits(self.tick, self.parent, Purpose::Mutation, self.n) >> 24) as u8
    }

    /// One point substitution: usually a step, occasionally anything.
    ///
    /// A uniform redraw was the first version and it is wrong, for the reason
    /// `PLAN.md` gives when it argues for this representation at all: a single
    /// byte change has to *nudge* the key, so that affinity moves a little and
    /// the landscape is traversable. Redrawing a byte moves one of eight key
    /// components to a uniformly random value, which is a jump. A lineage
    /// whose enzyme jumps usually lands nowhere -- it stops catalysing what it
    /// was catalysing without starting on anything else -- so there is no
    /// gradient for selection to climb and the whole key-vector scheme is
    /// wasted.
    ///
    /// Eighty per cent of substitutions move the byte by one to eight, so a
    /// key component drifts by up to 8/255 of its range and the protein stays
    /// recognisably itself. The remaining fifth is a full redraw, which is
    /// what changes a protein's *class*, breaks a promoter, or makes one out
    /// of junk -- rare, large, and necessary.
    ///
    /// Steps reflect off the ends rather than clamping. Clamping would pile
    /// lineages up at 0 and 255, which are the extremes of a key component and
    /// the last place a drifting protein should preferentially sit.
    fn substitute(&mut self, old: u8) -> u8 {
        if self.unit() >= 0.8 {
            return self.byte();
        }
        let step = 1 + self.below(8) as i16;
        let moved = if self.unit() < 0.5 {
            old as i16 - step
        } else {
            old as i16 + step
        };
        (if moved < 0 {
            -moved
        } else if moved > 255 {
            510 - moved
        } else {
            moved
        }) as u8
    }

    /// How many events happen, given `n` opportunities each of probability
    /// `p`. Knuth's Poisson method on the mean, which is the right
    /// approximation here because `n p` is far below one for every operator
    /// in the table and costs one draw when nothing happens.
    fn count(&mut self, n: usize, p: f32) -> usize {
        let lambda = n as f32 * p;
        if lambda <= 0.0 {
            return 0;
        }
        // Guard against a misconfigured rate turning this into a long loop.
        let limit = 64;
        let l = (-lambda).exp();
        let mut k = 0;
        let mut prod = 1.0f32;
        loop {
            prod *= self.unit();
            if prod <= l || k >= limit {
                return k;
            }
            k += 1;
        }
    }
}

/// Copy a genome for a daughter, with mutation.
///
/// Returns `None` when nothing happened, which is the common case: the caller
/// then shares the mother's allocation instead of paying for a copy and a
/// decode. At the default rates a 600-byte genome mutates at about one
/// division in two, so half of all divisions cost nothing at all.
pub fn replicate(
    genome: &Genome,
    rates: &MutationRates,
    rng: &Counter,
    tick: u64,
    parent: u64,
) -> Option<Genome> {
    if !rates.any() || genome.is_empty() {
        return None;
    }
    let mut d = Draws {
        rng,
        tick,
        parent,
        n: 0,
    };

    let n_bytes = genome.len();
    let n_genes = genome.genes.len();

    let points = d.count(n_bytes, rates.point);
    let indels = d.count(n_bytes, rates.indel);
    let duplications = d.count(n_genes, rates.duplication);
    let deletions = d.count(n_genes, rates.deletion);
    let inversions = d.count(1, rates.inversion);
    let whole = d.count(1, rates.genome_duplication);

    if points + indels + duplications + deletions + inversions + whole == 0 {
        return None;
    }

    let mut bytes = genome.bytes.clone();

    // Gene-level operators first, while the decoded spans still describe the
    // string. Applying them from the back means an edit never moves the span
    // of an operator that has not run yet.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for _ in 0..duplications {
        if n_genes > 0 {
            spans.push((d.below(n_genes), 1));
        }
    }
    for _ in 0..deletions {
        if n_genes > 0 {
            spans.push((d.below(n_genes), 0));
        }
    }
    // Back to front, so an edit never moves the span of an operator that has
    // not run yet, and at most one operator per gene -- a gene that drew both
    // a duplication and a deletion would otherwise have the second of them
    // splicing at an offset the first had already moved.
    spans.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    spans.dedup_by_key(|s| s.0);
    for (gene, duplicate) in spans {
        let g = &genome.genes[gene];
        let (at, len) = (g.at, g.len);
        if at + len > bytes.len() {
            continue;
        }
        if duplicate == 1 {
            if bytes.len() + len > MAX_GENOME {
                continue;
            }
            let copy: Vec<u8> = bytes[at..at + len].to_vec();
            bytes.splice(at + len..at + len, copy);
        } else {
            bytes.drain(at..at + len);
        }
    }

    // Whole-genome duplication. Rare, and the one operator that can double a
    // lineage's whole repertoire in a single division.
    for _ in 0..whole {
        if bytes.len() * 2 <= MAX_GENOME {
            let copy = bytes.clone();
            bytes.extend_from_slice(&copy);
        }
    }

    // Inversion: reverse a segment. Regulatory rewiring, and the cheapest way
    // for a promoter to appear where there was none.
    for _ in 0..inversions {
        if bytes.len() < 8 {
            break;
        }
        let a = d.below(bytes.len());
        let span = 4 + d.below(bytes.len() / 4 + 1);
        let b = (a + span).min(bytes.len());
        bytes[a..b].reverse();
    }

    // Indels. Small, so a deletion inside a gene shifts the frame rather than
    // removing the gene cleanly -- which is the point: most indels in coding
    // sequence should be bad.
    for _ in 0..indels {
        if bytes.is_empty() {
            break;
        }
        let at = d.below(bytes.len());
        let n = 1 + d.below(4);
        if d.unit() < 0.5 {
            if bytes.len() + n <= MAX_GENOME {
                let filler: Vec<u8> = (0..n).map(|_| d.byte()).collect();
                bytes.splice(at..at, filler);
            }
        } else {
            let end = (at + n).min(bytes.len());
            bytes.drain(at..end);
        }
    }

    // Point substitutions last, so they land on the string the daughter
    // actually inherits rather than on offsets the operators above have moved.
    for _ in 0..points {
        if bytes.is_empty() {
            break;
        }
        let at = d.below(bytes.len());
        bytes[at] = d.substitute(bytes[at]);
    }

    if bytes.is_empty() {
        return None;
    }
    Some(Genome::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Counter {
        Counter::new(1)
    }

    #[test]
    fn a_written_gene_decodes_back_to_what_was_written() {
        let key = [0.5, 0.0, 0.25, 1.0, 0.75, 0.125, 0.9, 0.33];
        let mut params = [0.0f32; N_PARAMS];
        params[0] = 0.5;
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &[0.0; KEY_DIM],
            0.9,
            &[],
            ProteinClass::Enzyme,
            &key,
            &params,
            0.02,
        );
        let g = Genome::new(bytes);
        assert_eq!(g.genes.len(), 1);
        let gene = &g.genes[0];
        assert_eq!(gene.class, ProteinClass::Enzyme);
        for (decoded, written) in gene.key.iter().zip(&key) {
            // One byte of resolution over the whole normalised range.
            assert!((decoded - written).abs() < 1.0 / 255.0);
        }
        assert!((gene.strength() - 1.0).abs() < 0.05);
    }

    #[test]
    fn junk_carries_no_genes() {
        let mut bytes = Vec::new();
        write_junk(&mut bytes, &rng(), 0, 0, 4000);
        assert!(
            Genome::new(bytes).genes.is_empty(),
            "junk decoded into a gene"
        );
    }

    #[test]
    fn a_gene_running_off_the_end_is_not_a_gene() {
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &[0.0; KEY_DIM],
            0.5,
            &[],
            ProteinClass::Enzyme,
            &[0.0; KEY_DIM],
            &[0.5; N_PARAMS],
            0.02,
        );
        bytes.truncate(bytes.len() - 3);
        assert!(Genome::new(bytes).genes.is_empty());
    }

    #[test]
    fn affinity_is_one_at_the_target_and_falls_away() {
        let a = [0.5; KEY_DIM];
        let b = [0.5; KEY_DIM];
        let far = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        assert!((affinity(&a, &b, 0.15) - 1.0).abs() < 1.0e-6);
        assert!(affinity(&a, &far, 0.15) < 1.0e-6);
    }

    #[test]
    fn one_byte_changes_affinity_by_a_little_and_not_a_lot() {
        // The graded-landscape claim, checked rather than asserted. A single
        // substitution must move affinity enough for selection to see it and
        // not so much that the landscape is a cliff.
        let key = [0.5f32; KEY_DIM];
        let mut nudged = key;
        nudged[0] += 1.0 / 255.0;
        let a = affinity(&key, &key, 0.15);
        let b = affinity(&nudged, &key, 0.15);
        let change = (a - b).abs();
        assert!(change > 1.0e-5, "a byte moved affinity by {change}");
        assert!(change < 0.2, "a byte moved affinity by {change}");
    }

    #[test]
    fn the_ancestor_catalyses_the_reaction_it_was_written_for() {
        // The test that was missing when the key range was assumed rather than
        // measured. Sixteen founders arrived in a full pond, ate nothing, and
        // every unit test passed: nothing here had ever checked that a written
        // key and a chemistry key end up in the same space.
        let params = hadean_chem::ChemParams {
            n_compounds: 24,
            n_reactions: 120,
            ..Default::default()
        };
        let chem = hadean_chem::generate(1, params);
        let reaction = chem
            .reactions
            .iter()
            .find(|r| r.drive == Drive::Thermal && r.dh < 0.0)
            .expect("an exergonic reaction")
            .id;
        let mut g = ancestor(&chem, reaction, &rng(), 0);
        g.bind(&chem, 0.12, 0.2);

        let enzyme = g
            .genes
            .iter()
            .position(|gene| gene.class == ProteinClass::Enzyme)
            .expect("the ancestor has an enzyme");
        let (best, strength) = g.targets[enzyme]
            .reactions
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .expect("the enzyme binds something");
        assert_eq!(best, reaction, "the enzyme's best match is not its reaction");
        assert!(
            strength > 0.98,
            "the enzyme runs its own reaction at {strength}"
        );

        // And it can get its substrates across the membrane.
        for &(c, _) in &chem.reaction(reaction).reactants {
            assert!(
                g.genes
                    .iter()
                    .zip(&g.targets)
                    .any(|(gene, t)| gene.class == ProteinClass::Transporter
                        && t.compounds.iter().any(|&(id, a)| id == c && a > 0.98)),
                "no transporter for substrate {}",
                chem.compounds[c as usize].name
            );
        }
    }

    /// One regulator gene written by hand, so a replication factor can be
    /// asked for rather than hoped for. `params[1]` picks the role and
    /// `params[2]` becomes the bias, which is both the direction and the size
    /// of the push.
    fn regulator(role: f32, bias: f32) -> Genome {
        let mut params = [0.0f32; N_PARAMS];
        params[1] = role;
        params[2] = bias;
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &[0.0; KEY_DIM],
            0.9,
            &[],
            ProteinClass::Regulator,
            &[0.5; KEY_DIM],
            &params,
            0.05,
        );
        Genome::new(bytes)
    }

    /// The ancestor carries no replication factor, so it replicates at exactly
    /// the configured rates and a mutator allele is something the pond has to
    /// select into a lineage rather than something it is handed. That is what
    /// `PLAN.md` asks for, and it is a property of the written ancestor rather
    /// than of the mechanism, so it is worth pinning.
    #[test]
    fn the_ancestor_carries_no_replication_factor() {
        let chem = hadean_chem::generate::generate(1, Default::default());
        let reaction = chem
            .reactions
            .iter()
            .find(|r| r.drive == Drive::Thermal && r.dh < 0.0)
            .expect("a chemistry with something to eat")
            .id;
        let g = ancestor(&chem, reaction, &rng(), 0);
        assert_eq!(g.mutator_genes(), 0);
        let full = vec![1.0f32; g.genes.len()];
        assert_eq!(g.mutator_drive(&full), 0.0);
    }

    /// A regulator that is not a replication factor is invisible here however
    /// hard its bias pushes, and one that is pushes in the direction of its
    /// bias. Both directions, because a lineage in stasis wants the other one.
    #[test]
    fn a_replication_factor_pushes_in_the_direction_of_its_bias() {
        // params[1] = 0.0 selects role 0: a transcription factor and nothing
        // else, whatever its bias says.
        let plain = regulator(0.0, 1.0);
        assert_eq!(plain.mutator_genes(), 0);
        assert_eq!(plain.mutator_drive(&[1.0]), 0.0);

        // params[2] = 1.0 is a bias of +4, which at full expression is the
        // whole range: maximally error-prone.
        let sloppy = regulator(1.0, 1.0);
        assert_eq!(sloppy.mutator_genes(), 1);
        assert_eq!(sloppy.mutator_drive(&[1.0]), 1.0);
        // Half expressed is half the push. This is the gradient selection
        // climbs; a step function would give it nothing to climb.
        assert!((sloppy.mutator_drive(&[0.5]) - 0.5).abs() < 1.0e-6);
        // Not expressed at all is no push, which is what makes the allele
        // something a cell can carry and not use.
        assert_eq!(sloppy.mutator_drive(&[0.0]), 0.0);

        // params[2] = 0.0 is a bias of -4: maximally faithful.
        let careful = regulator(1.0, 0.0);
        assert_eq!(careful.mutator_drive(&[1.0]), -1.0);
    }

    /// A clone line stays a clone line. The rates are *scaled*, and zero
    /// scales to zero however hard a genome leans on it -- so the control
    /// every claim about evolution in this project is measured against cannot
    /// be broken by a mutation.
    #[test]
    fn no_mutator_allele_can_break_a_clone_line() {
        let none = MutationRates::none();
        assert!(!none.scaled(1.0e6).any());
        assert!(!none.scaled(4.0).any());
    }

    /// And a mutator cannot reach a rate the config layer would have refused.
    /// `validate` rejects a per-byte rate above 0.5 as an error rather than a
    /// fast run, and a genome must not be able to get there by the back door.
    #[test]
    fn a_mutator_cannot_scale_past_what_validate_accepts() {
        let hot = MutationRates {
            point: 0.4,
            indel: 0.3,
            ..MutationRates::default()
        };
        let scaled = hot.scaled(1000.0);
        assert_eq!(scaled.point, 0.5);
        assert_eq!(scaled.indel, 0.5);
        assert!(scaled.validate().is_ok());
        // The gene-level operators are counts per gene, not probabilities per
        // byte, so they are free to go up and `validate` does not cap them.
        assert!(scaled.duplication > hot.duplication);
    }

    /// Scaling by one is the identity, which is what makes `mutator_range = 0`
    /// bit-identical to every run taken before this mechanism existed.
    #[test]
    fn scaling_by_one_changes_nothing() {
        let base = MutationRates::default();
        assert_eq!(base.scaled(1.0), base);
    }

    #[test]
    fn a_substitution_usually_nudges_a_key_and_sometimes_does_not() {
        // The mapping being graded is not enough; the *operator* has to be.
        // The first version redrew the byte uniformly, which moves a key
        // component by about a third of its range on average -- three times
        // the median gap between two reactions -- so a drifting enzyme would
        // leave its own reaction without arriving at another.
        let mut d = Draws {
            rng: &rng(),
            tick: 3,
            parent: 11,
            n: 0,
        };
        let start = 128u8;
        let mut nudges = 0;
        let mut jumps = 0;
        let n = 4000;
        for _ in 0..n {
            let moved = d.substitute(start);
            if (moved as i16 - start as i16).unsigned_abs() <= 8 {
                nudges += 1;
            } else {
                jumps += 1;
            }
        }
        // Eighty per cent nudge by construction, and a redraw lands inside the
        // nudge band about a sixteenth of the time, so the split is not exact.
        assert!(
            nudges > n * 3 / 4,
            "only {nudges} of {n} substitutions were small"
        );
        assert!(jumps > n / 10, "only {jumps} of {n} substitutions were large");
    }

    #[test]
    fn one_substitution_usually_moves_affinity_a_little() {
        // `PLAN.md`'s claim, stated as the measurement that would falsify it:
        // "a single byte change nudges the key slightly, which nudges affinity
        // slightly". The first operator here redrew the byte uniformly and did
        // not have this property at all -- a substitution moved affinity from
        // 1.0 to nothing most of the time, which is the field of cliffs the
        // plan warns the whole scheme exists to avoid.
        //
        // A fifth of substitutions still do exactly that, and should: a
        // radical change at a key residue destroys a protein in real cells
        // too, and it is the same operator that changes a protein's class or
        // makes a promoter out of junk. What matters is that it is the tail
        // and not the body.
        let mut d = Draws {
            rng: &rng(),
            tick: 1,
            parent: 2,
            n: 0,
        };
        let start = [128u8; KEY_DIM];
        let original: [f32; KEY_DIM] = std::array::from_fn(|k| start[k] as f32 / 255.0);

        let n = 2000;
        let mut changes: Vec<f32> = Vec::with_capacity(n);
        for trial in 0..n {
            let mut bytes = start;
            let at = trial % KEY_DIM;
            bytes[at] = d.substitute(bytes[at]);
            let key: [f32; KEY_DIM] = std::array::from_fn(|k| bytes[k] as f32 / 255.0);
            changes.push(1.0 - affinity(&key, &original, 0.15));
        }
        changes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = changes[n / 2];
        assert!(
            median < 0.35,
            "the median substitution cost {median:.3} of a unit of affinity"
        );
        assert!(
            changes[n / 10] < 0.05,
            "even the gentlest tenth of substitutions moved affinity by {:.3}",
            changes[n / 10]
        );
        assert!(
            changes[n - n / 20] > 0.9,
            "no substitution was radical: the worst twentieth moved {:.3}",
            changes[n - n / 20]
        );
    }

    #[test]
    fn mutation_is_a_pure_function_of_tick_and_parent() {
        let mut bytes = Vec::new();
        write_junk(&mut bytes, &rng(), 0, 0, 600);
        let g = Genome::new(bytes);
        let rates = MutationRates {
            point: 0.05,
            ..MutationRates::default()
        };
        let a = replicate(&g, &rates, &rng(), 77, 3);
        let b = replicate(&g, &rates, &rng(), 77, 3);
        assert_eq!(a, b, "the same division produced two different daughters");
    }

    #[test]
    fn no_rates_means_no_copy_at_all() {
        let mut bytes = Vec::new();
        write_junk(&mut bytes, &rng(), 0, 0, 200);
        let g = Genome::new(bytes);
        assert!(replicate(&g, &MutationRates::none(), &rng(), 1, 1).is_none());
    }

    #[test]
    fn duplication_grows_a_genome_and_its_gene_count() {
        let mut bytes = Vec::new();
        write_gene(
            &mut bytes,
            &[0.0; KEY_DIM],
            0.5,
            &[],
            ProteinClass::Enzyme,
            &[0.0; KEY_DIM],
            &[0.5; N_PARAMS],
            0.02,
        );
        let g = Genome::new(bytes);
        assert_eq!(g.genes.len(), 1);
        let rates = MutationRates {
            duplication: 1.0,
            ..MutationRates::none()
        };
        // Whatever the draw, a duplication either happened or it did not; run
        // a few parents until one lands, then check what it did.
        let grown = (0..32)
            .filter_map(|p| replicate(&g, &rates, &rng(), 0, p))
            .find(|d| d.len() > g.len())
            .expect("a duplication in thirty-two divisions at rate one");
        assert_eq!(grown.genes.len(), 2, "duplicate did not decode as a gene");
        assert_eq!(grown.genes[0].key, grown.genes[1].key);
    }

    #[test]
    fn a_genome_cannot_grow_without_bound() {
        let mut bytes = Vec::new();
        write_junk(&mut bytes, &rng(), 0, 0, MAX_GENOME - 100);
        let mut g = Genome::new(bytes);
        let rates = MutationRates {
            genome_duplication: 1.0,
            duplication: 1.0,
            ..MutationRates::none()
        };
        for p in 0..40 {
            if let Some(next) = replicate(&g, &rates, &rng(), 0, p) {
                g = next;
            }
            assert!(g.len() <= MAX_GENOME, "genome reached {}", g.len());
        }
    }
}
