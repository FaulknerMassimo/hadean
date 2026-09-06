# Life Simulator — Design and Build Plan

A bottom-up artificial life simulation: digital chemistry → genomes → proteins → cells → multicellular organisms → nervous systems → ecosystems, rendered in real-time 3D.

---

## 1. Your spec, restated

From the notes, the system needs:

1. Organisms carry a **machine-readable genetic code** that defines what they are.
2. Reproduction is **sexual or asexual with inheritance and mutation**.
3. The genome can encode a **neural network** that makes decisions.
4. The genome codes for **proteins**, and proteins build the organism (membrane, internal machinery).
5. **Multicellularity emerges** through evolution, then cell specialization and organs.
6. The environment obeys **physics**: gravity, particles, wave phenomena.
7. **Sound and light** exist as frequency phenomena; organisms can sense them and emit them.
8. Organisms **eat**, and at the start only **basic compounds** exist. No pre-authored "apple."
9. **Death is inevitable**; corpses decompose into compounds that re-enter the environment.
10. Living organisms can **become food** if something evolves the means to eat them.

That is a coherent and genuinely good design. The core insight — that you can't hand-author "food," you have to let it fall out of a chemistry — is the thing most hobby life-sims get wrong.

### Three places I'm going to push back

**Physics at Earth fidelity is not achievable and not desirable.** Real light is Maxwell's equations, real sound is the wave equation in a compressible fluid, and real chemistry is quantum mechanics. Simulating any of those at molecular resolution for even one cell would consume a supercomputer. What you actually need is a **synthetic physics that is internally consistent and conserves energy**, because energy conservation is what forces organisms to genuinely earn their existence. Fidelity to Earth is optional; fidelity to thermodynamics is not.

**Sound and light are not the same phenomenon at different frequencies.** Sound is a mechanical pressure wave in a medium; light is electromagnetic. They don't sit on one spectrum. But your underlying intuition — that the engine should treat "information carried by waves" as one unified abstraction — is right, and I've built the design around a generic **wave channel** system with different propagation rules per channel. You get the design elegance you were after without the physics being wrong.

**Cells in a multicellular organism don't have their own genetic code for their job.** They all carry the *same* genome and differ in which genes are *expressed*. This matters enormously for the design: differentiation has to come from a **gene regulatory network** responding to position and signals, not from per-cell genomes. This is actually better news than it sounds — it's the mechanism that makes complex bodies evolvable at all, because one mutation can retune an entire tissue.

---

## 2. The decision that determines everything: spatial scale

You cannot have a cubic-metre world *and* microscopic cells. A 1 m³ world at 10 µm resolution is 10¹⁵ voxels. It is not a matter of optimization; it's a factor of 10⁹ off.

So pick a scale, and design a path to widen it later.

### Recommended: start as a pond

| Parameter | Value |
|---|---|
| World volume | ~5 mm × 5 mm × 1.5 mm |
| Voxel size (`dx`) | 25 µm |
| Grid | 200 × 200 × 60 ≈ 2.4 M voxels |
| Cell diameter | 5–30 µm (0.2–1.2 voxels) |
| Target population | 10⁴ – 10⁶ cells |
| Timestep (`dt`) | 10 ms sim time |
| Target throughput | 100–1000 ticks/s headless |

This gives you a world where a cell is roughly voxel-sized, which is exactly the right relationship: the chemistry field has structure at the scale organisms care about. It is also a world you can **actually see** — zoom out and it's a pond of drifting colonies, zoom in and you watch a single cell's membrane transporters firing.

Depth matters here. A vertical light gradient from the surface plus a thermal gradient from the bottom gives you two orthogonal environmental axes immediately, which is enough to drive niche differentiation on day one.

### Scaling up later

Two mechanisms, both Phase 8+:

- **Chunked world.** The grid becomes a sparse set of streamed chunks. Regions with no cells and flat chemistry get frozen and coarsened.
- **Simulation LOD.** Not rendering LOD — *simulation* LOD. A stable colony of 10,000 identical cells far from the camera collapses into a single aggregate agent with a population count and a mean genome, stepped by a coarse ODE. It re-expands to individual cells if a mutation event or a disturbance hits it. This is hard and easy to get wrong; do not attempt it before Phase 8.

---

## 3. Architecture overview

```
┌──────────────────────────────────────────────────────────────┐
│  L7  VIEWER          camera, volumetrics, inspection, graphs  │
├──────────────────────────────────────────────────────────────┤
│  L6  ECOLOGY         death, decomposition, predation, stats   │
├──────────────────────────────────────────────────────────────┤
│  L5  ORGANISM        adhesion, development, differentiation,  │
│                      neural graph, behaviour                  │
├──────────────────────────────────────────────────────────────┤
│  L4  CELL            membrane, transport, metabolism,         │
│                      expression, division, motility           │
├──────────────────────────────────────────────────────────────┤
│  L3  GENOME          bytes → genes → proteins → functions     │
├──────────────────────────────────────────────────────────────┤
│  L2  FIELDS          light bands, heat, flow, wave channels   │
├──────────────────────────────────────────────────────────────┤
│  L1  CHEMISTRY       compounds, reaction network, diffusion   │
├──────────────────────────────────────────────────────────────┤
│  L0  SUBSTRATE       units, grid, time, RNG, determinism      │
└──────────────────────────────────────────────────────────────┘
```

Each layer only talks to the one below it through a narrow interface. That discipline is what lets you swap the chemistry engine in Phase 6 without rewriting cells.

---

## 4. Layer designs

### L0 — Substrate

**Units.** Define a unit system up front and put it in one header. Length in metres, time in seconds, energy in joules, amount in "particles" (not moles — you're at scales where a voxel might hold 10⁶ molecules, so use raw counts as `f32` and accept the shot-noise abstraction). Write the unit constants down; you will regret it if you don't.

**Time.** Fixed timestep, always. Never couple `dt` to frame time. The viewer runs at its own rate and interpolates.

**Multi-rate integration** — this is one of your biggest performance levers. Different processes have different natural timescales:

| Process | Every N ticks | Why |
|---|---|---|
| Physics (forces, motion) | 1 | Stability |
| Chemistry + diffusion | 1 | Coupled to physics |
| Membrane transport | 1 | Fast |
| Neural update | 1–2 | Behaviour needs responsiveness |
| Gene expression | 10 | Transcription is genuinely slow |
| Growth / division check | 20 | Slow |
| Development / morphogenesis | 50 | Slower still |
| Phylogeny bookkeeping | 500 | Analysis only |

Stagger the phases across entities (`entity_id % N == tick % N`) so you don't get a sawtooth CPU load every tenth tick.

**Determinism.** Non-negotiable for a project like this — you *will* need to replay a run where something interesting happened. Requirements:

- **Counter-based RNG.** Never a stateful global RNG. Use PCG or Philox seeded by `hash(world_seed, tick, entity_id, purpose_tag)`. This makes randomness reproducible regardless of thread scheduling or evaluation order.
- **Deterministic reduction order.** Parallel float sums are not associative. Where you must reduce (total energy, per-voxel contributions from many cells), either sort contributions by entity ID before summing, or accumulate in fixed-point integers. The per-voxel scatter from cells is the classic offender — use atomic integer accumulation into fixed-point, then convert.
- **No wall-clock, no uninitialized memory, no iteration over hash maps whose order depends on pointer values.**

Test it: run 10,000 ticks twice, hash the full world state, assert equality. Put that in CI.

**Snapshots and replay.** `(world_seed, config_hash, tick, user_event_log)` should be sufficient to reproduce any state. Also write full binary snapshots every ~100k ticks so you don't have to replay 40 hours to look at something.

---

### L1 — Chemistry

This is the foundation of everything. Get it right and metabolism, food webs, and decomposition all fall out for free.

#### Version 1: fixed compound set (build this first)

At world generation, procedurally generate a **chemistry** from the seed:

- **Elements:** 6 types with valences. Think of them as roughly {carbon-like (4), hydrogen-like (1), oxygen-like (2), nitrogen-like (3), sulfur-like (2), metal ion (1)}.
- **Compounds:** N = 128 to 256, each generated as a small valence-satisfying assembly of 1–12 atoms.
- **Per-compound derived properties:**
  - `formation_enthalpy` (J per particle) — the energy spine
  - `mass`
  - `diffusion_coefficient` (falls with mass)
  - `membrane_permeability` (function of size and polarity)
  - `absorption_spectrum[8]` — per light band
  - `emission_spectrum[8]` — for the few that fluoresce
  - `charge`, `polarity`
- **Reaction network:** generate M ≈ 2000 reactions of the form `aA + bB ⇌ cC + dD`, mass-balanced by construction. `ΔH` is computed from formation enthalpies, so **you cannot accidentally create free energy**. Each reaction has an intrinsic activation energy `Ea` and a base rate.

Uncatalyzed rate: `k = A · exp(-Ea / (kB · T))`. At ambient temperature most reactions run essentially at zero. **Enzymes work by lowering `Ea`.** That single mechanism is the entire reason cells matter — a cell is a bag that makes thermodynamically favourable reactions actually happen, at a place and time of its choosing.

Intern compounds as integer IDs. Store per-voxel concentrations as a dense `f32[N]` array, structure-of-arrays, one texture/buffer per compound. At N=128 and 2.4M voxels that's 1.2 GB in f32 — too much. Two fixes:

- Use `f16` for concentration (0.6 GB), or
- **Sparse compounds.** Most compounds are absent from most voxels. Keep 16 "bulk" compounds dense and the rest in a per-chunk sparse list. This is more code; do it only when you hit the wall.

Start with **N = 32 compounds** for Phases 1–4. You do not need 256 to get interesting metabolism, and 32 fits comfortably in VRAM at full resolution.

#### Version 2: graph-rewrite chemistry (Phase 6+, optional)

Compounds become small labelled graphs; reactions become graph rewrite rules matched by subgraph isomorphism; new compounds are *discovered* rather than pre-enumerated. This gives you truly open-ended chemistry — organisms can evolve molecules the world generator never imagined. It is also 50× more expensive and needs a canonical-hashing scheme to intern molecules. Design your L1 interface (`react(voxel, enzyme_set, dt)`) so this is swappable, but do not build it early.

#### Transport

Per timestep, per voxel: **diffusion** (`∂c/∂t = D∇²c`, standard 7-point stencil, explicit Euler as long as `D·dt/dx² < 1/6`, otherwise implicit or sub-step), plus **advection** by the flow field, plus **buoyancy/settling** for dense compounds.

Diffusion is the single most GPU-friendly part of this project. It's a stencil op on a 3D texture and will run at hundreds of Hz.

#### Primordial energy input

Food starts scarce, per your notes. Energy enters the world through:

1. **Photochemistry.** Light drives one or two specific reactions abiotically at low yield, producing a moderately high-energy compound near the surface. This is your primordial soup, and it establishes a *vertical resource gradient*.
2. **Thermal vents.** A few point sources at the bottom of the world inject heat and reduced compounds. This creates a second, independent niche, so early life has two ways to make a living and you get divergence rather than a monoculture.
3. Nothing else. No respawning pellets, no "food particles." Everything organisms eat must come from these two taps or from other organisms.

**Global energy audit:** every tick, sum all energy — chemical (Σ concentration × formation enthalpy), thermal, kinetic, stored in cells — and compare to (previous total + inputs − radiative losses). This should be conserved to within float precision. Log the drift. If it starts growing, you have a bug that will let organisms evolve into perpetual motion machines, and they absolutely will find it. Treat a broken energy audit as a build-breaking failure.

---

### L2 — Fields

#### Light

Not a wave solver. Use **band-limited radiative transfer**.

- **8 spectral bands.** Sunlight enters the top face with a spectrum that varies over a day/night cycle.
- **Attenuation** by Beer–Lambert along the vertical column: `I(z+dz) = I(z) · exp(-Σ_i c_i · σ_i(band) · dz)`, where `σ_i(band)` is compound *i*'s absorption cross-section in that band. This is a single prefix-sum pass down the Z axis — extremely cheap.
- **Emission.** Bioluminescent cells and fluorescent compounds inject into bands as point sources. For these, don't do full transport — use a bounded-radius scatter with `1/r²` falloff and local attenuation. Fine at pond scale.
- **Scattering** — skip it, or approximate with a cheap ambient term. Not worth the cost early.

**Why 8 bands is the right choice:** it's enough for absorption spectra to differentiate meaningfully, it's cheap, and it maps directly to rendering. When a lineage evolves a pigment protein that absorbs strongly in band 2, the cell *literally changes colour in the viewer* because you're mapping the same spectrum to RGB. Evolution becomes visible without any instrumentation.

Two proteins consume this field:
- **Photoreceptors** — read intensity and directional gradient in a band range set by the genome. Vision.
- **Pigments** — catalyze a photochemical reaction with rate ∝ absorbed flux. Photosynthesis.

Both have a genome-encoded absorption peak and width. Pigments and eyes evolve on the same substrate, which is a nice touch.

#### Heat

Scalar field. Diffuses. Exothermic reactions add to it, endothermic subtract. Radiates from the top boundary. Feeds back into every reaction rate through the Arrhenius term. This closes a loop: metabolism warms the water, warm water speeds metabolism, which is a real and interesting instability.

#### Flow

Coarser grid (e.g. 4× the voxel size). Options in increasing cost:

1. **Static circulation.** Author a slow convection roll from the thermal gradient. Sufficient for Phases 1–5.
2. **Stable Fluids** (Stam's semi-Lagrangian solver). Unconditionally stable, easy, GPU-friendly. Add buoyancy from the heat field. This is the sweet spot.
3. Full Navier–Stokes with proper viscosity. At micrometre scale the Reynolds number is around 10⁻³ — you're in the Stokes regime where inertia is irrelevant and swimming is genuinely weird (the "scallop theorem": reciprocal motion produces no net movement). Physically fascinating, but it makes flagellar propulsion a research project rather than a feature. **Recommendation: fake it.** Give cells simple drag-plus-thrust and note the caveat in your docs.

#### Wave channels — unifying your "frequency" idea

Here's the abstraction that captures what you wrote without being physically wrong. Define a generic interface:

```
trait WaveChannel {
    fn propagate(&mut self, dt: f32);
    fn emit(&mut self, pos: Vec3, band: u8, amplitude: f32);
    fn sample(&self, pos: Vec3, band_range: (u8, u8)) -> (f32, Vec3);  // intensity, gradient
}
```

Three implementations, three physics:

| Channel | Propagation | Speed | Cost | Sense |
|---|---|---|---|---|
| Chemical | Diffusion (L1) | Very slow | Already paid | Smell / taste |
| Light | Beer–Lambert columns | Instant | Very cheap | Sight |
| Mechanical | Event-based (below) | Fast | Cheap | Hearing / touch |

Sensor and effector proteins are written against the trait, not the implementation. So a genome that evolves "emit into channel X at band B" doesn't care whether X is light or sound. That's your unified frequency system, and it's honest.

**On acoustics specifically.** A grid-based FDTD acoustic solver is not viable here. At `dx` = 25 µm and a realistic speed of sound, the CFL condition demands `dt` ≈ 10⁻⁸ s — a million substeps per chemistry tick. Two ways out:

- **Redefine the speed of sound.** It's your universe. Pick `c` such that `c·dt/dx ≤ 1/√3`. Everything stays self-consistent, just slower than water.
- **Better: don't grid it.** At pond scale, propagation delay is nanoseconds — physically irrelevant. What matters is *amplitude, frequency content, and direction at the receiver*. Model emitters as events with `(position, band, amplitude, tick)`, and have each receiver integrate contributions with `1/r` attenuation, exponential absorption, and optional occlusion from a coarse ray query. Cost is O(emitters × nearby receivers), which with a spatial hash is trivial.

At micrometre scale, the mechanically meaningful signal is actually **local flow shear and contact force**, not far-field sound. So: Phase 6 gives you mechanoreception from the flow field and contact forces. Real far-field acoustics becomes interesting only once organisms are millimetres across and can emit at useful amplitudes. Build the channel interface early; light it up in Phase 8.

---

### L3 — Genome

The single most important design decision in the project, because it determines whether evolution actually goes anywhere.

#### Representation

A **linear byte string**, variable length, 500 bytes to ~1 MB. Not a tree, not a fixed struct, not a direct neural network encoding.

Why a linear string: it supports the mutation operators that actually generate complexity — **duplication**, insertion, deletion, inversion, translocation. Fixed-size structs support only point mutation, and point mutation alone cannot grow complexity. Gene duplication followed by divergence is *the* mechanism by which biological complexity arose. If your representation can't duplicate a gene, your organisms will plateau.

#### Structure

```
... junk ... [PROMOTER][REGULATORY REGION][START][ CODING REGION ][STOP] ... junk ...
```

- **Promoter:** an 8-byte recognition motif. Transcription factors bind here with affinity = a similarity function over the motif.
- **Regulatory region:** a list of binding sites, each `(motif, weight)`. Positive weight = activator, negative = repressor.
- **Coding region:** decoded in fixed-width codons into a **protein descriptor**.

Decoding is a linear scan for promoter motifs. Note that mutations in the *junk* regions are neutral, which is important: neutral drift lets populations explore genotype space without paying a fitness cost, and is a well-documented enabler of later innovation. Don't compact junk away.

#### Protein descriptor

```rust
struct Protein {
    class: ProteinClass,     // 3 bits, from the first codon
    key: [f32; 8],           // "shape" — used for all matching
    params: [f32; 12],       // class-specific
    stability: f32,          // degradation rate
}
```

**The key vector is the central trick.** Every matching operation in the entire simulation — enzyme to reaction, transporter to compound, TF to promoter, adhesion to adhesion, gap junction to gap junction — is a **similarity function between key vectors**:

```
affinity(a, b) = exp(-||a.key - b.key||² / σ²)
```

This gives you two things you desperately need:

1. **Graded mutation response.** A single byte change nudges the key slightly, which nudges affinity slightly. Fitness landscapes become traversable instead of a field of cliffs. This is the difference between evolution working and evolution not working.
2. **Promiscuity and neofunctionalization.** A protein weakly binds several targets. Duplicate it, let the copies drift, and each specializes. That's exactly how real protein families arise.

**Protein classes:**

| Class | Function | Key params |
|---|---|---|
| `Enzyme` | Lowers `Ea` for reactions matching `key` | efficiency, cofactor requirement |
| `Transporter` | Moves compounds across membrane | direction, gating (ligand/voltage), ATP cost |
| `Structural` | Membrane integrity, cytoskeleton, stiffness | contribution, target compartment |
| `Adhesion` | Binds cells with matching keys | bond strength, break force |
| `Receptor` | Senses a channel/compound → internal signal | channel, band range, sensitivity, gain |
| `Effector` | Thrust, luminescence, secretion, lysis | magnitude, direction, energy cost |
| `Regulator` | Transcription factor — binds promoters | (uses key only) |
| `Neural` | Declares a neuron with connection prefs | activation, bias, time constant |

Note `Effector::lysis` — a protein that breaks down another cell's structural proteins into free compounds. **This is where predation comes from.** You don't implement "eating." You implement a secreted enzyme that degrades membrane proteins, plus transporters for the resulting compounds. If a lineage evolves both, it is a predator. Your notes called for exactly this ("organisms can be used as food if some organisms find how to use them as food") and this is the mechanism that delivers it without you scripting anything.

#### Mutation operators

Applied at replication, each with its own rate (make them all config-tunable and log them):

| Operator | Typical rate | Why it matters |
|---|---|---|
| Point substitution | 10⁻⁴ / byte | Fine-tuning |
| Small indel | 10⁻⁵ / byte | Frame shifts, key drift |
| **Gene duplication** | 10⁻³ / gene | **The engine of complexity** |
| Gene deletion | 10⁻³ / gene | Streamlining |
| Segment inversion | 10⁻⁵ / genome | Regulatory rewiring |
| Transposon copy | 10⁻⁴ / genome | Bulk shuffling |
| Whole-genome duplication | 10⁻⁶ / genome | Rare, huge |
| Horizontal transfer | on lysis | Massive early-evolution accelerator |

**Mutator alleles.** Let a `Regulator`-class protein modulate the organism's own mutation rate. Mutation rate then evolves, and you'll see it rise during environmental change and fall during stasis. This is real biology and costs you almost nothing to implement.

**Horizontal gene transfer** deserves emphasis. When a cell lyses, its genome fragments enter the voxel as a compound-like entity. Nearby cells with a "competence" protein can integrate fragments. Early prokaryotic evolution ran largely on HGT, and in a simulation it enormously accelerates the search — good ideas propagate laterally instead of waiting for a lineage to reinvent them.

#### Sexual reproduction

Add later (Phase 5). Requires: diploidy (two genome copies), a meiosis operator (crossover at aligned homologous regions, alignment found by motif matching), and a mate-recognition mechanism (a surface protein pair — which then also gives you speciation by mating incompatibility for free). Do not put this in v1; asexual replication with mutation is enough to get evolution running and is far simpler to debug.

#### Seeding — read this before you skip it

**Do not start with random genomes.** The probability that a random byte string decodes into a self-replicating metabolism is effectively zero. Tierra, Avida, and every other successful digital evolution system seeds a **hand-written ancestor**: a minimal organism that can take up a compound, run one exergonic reaction to charge its energy carrier, maintain a membrane, and divide. Maybe 15 genes. Write it by hand, verify it survives in isolation, then let mutation loose on it.

If you want abiogenesis itself as a research target, that's a separate (very hard) project. Seed the ancestor.

---

### L4 — Cell

```rust
struct Cell {
    // Physics
    pos: Vec3, vel: Vec3, radius: f32, orientation: Quat,
    // Genetics
    genome: Arc<Genome>,        // shared, copy-on-write — clones are cheap
    expression: Vec<f32>,       // per-gene expression level
    proteome: Vec<(ProteinId, f32)>,   // sparse: protein → concentration
    // State
    internal: [f32; N_COMPOUNDS],
    energy: f32,
    volume: f32,
    damage: f32,
    age: u32,
    // Multicellular
    bonds: SmallVec<[BondId; 8]>,
    polarity: Vec3,
    neural_state: [f32; MAX_NEURONS],
}
```

`Arc<Genome>` matters: a colony of 10,000 clones shares one genome allocation. Copy-on-write at mutation.

**Per-tick cell update:**

1. **Sense** — receptors sample the local voxel (compounds, light, heat, shear) and contact forces.
2. **Neural** — one CTRNN step (below).
3. **Transport** — passive diffusion across the membrane (rate ∝ permeability × concentration gradient × surface area), plus active transport by transporter proteins (costs energy carrier, can run against a gradient).
4. **Metabolize** — run the reaction network on `internal` with enzymes lowering `Ea`. Exergonic reactions can be coupled to charge the energy carrier; endergonic reactions consume it. Waste heat into the voxel.
5. **Express** — every 10 ticks. For each gene: `expression = σ(Σ_TF affinity(TF.key, site.motif) · TF.conc · site.weight)`. Then produce protein at a rate proportional to expression, minus degradation. Both cost energy and precursor compounds.
6. **Maintain** — structural proteins degrade. Replacing them costs energy. If maintenance isn't paid, `damage` accumulates. **This is why death is inevitable**, per your notes: staying alive has a continuous, unavoidable energetic cost.
7. **Act** — effectors fire: thrust (costs energy), secretion, luminescence, lysis enzymes.
8. **Grow / divide** — every 20 ticks. If volume > threshold and energy reserve is adequate: copy the genome with mutation, partition internal contents (with asymmetry controlled by polarity — this is the seed of differentiation), split into two cells.
9. **Die** — if `energy < 0` or `damage > threshold` or the membrane fails. Convert the entire proteome and internal contents back into their constituent compounds, deposit them in the voxel. **Decomposition is not a special system; it's just cells dissolving back into chemistry.**

**Physics:** soft-sphere repulsion on overlap, adhesion springs from matched adhesion proteins, Stokes drag, buoyancy from density difference, Brownian noise scaled by temperature. Spatial hash for neighbour queries. Semi-implicit Euler.

---

### L5 — Multicellularity and development

#### How it starts

Multicellularity needs one mutation: **adhesion proteins that fail to release after division**. Two daughters stay stuck. That's it — that's the whole origin, and it happens in real life readily.

Why would it be selected for? Provide the conditions:

- **Predation.** Once lysis-based predators exist, size is protection. Clusters are harder to eat.
- **Shear.** Flow strips single cells out of good spots; clusters resist better.
- **Metabolic division of labour.** Two coupled reactions run better in separate compartments (one produces an intermediate that inhibits the other).

If you build the world with those pressures present, multicellularity emerges. If you don't, it won't, no matter how you fiddle with the genome format. **Selection pressure is a level design problem, not a code problem.** That's worth writing on a sticky note.

#### Differentiation

Same genome, different expression. Three sources of asymmetry:

1. **Polarity-driven asymmetric division.** A dividing cell partitions transcription factors unevenly along its polarity axis. Daughters start with different TF concentrations, land in different expression basins.
2. **Morphogen gradients.** Cells secrete diffusible compounds; concentration falls with distance from the source; a cell's position in the gradient sets which genes cross their activation thresholds. Classic French-flag model, and it produces stable spatial patterning.
3. **Juxtacrine signalling.** Direct contact between adjacent cells via matched surface proteins. This gives you lateral inhibition (Notch/Delta-style), which produces regular spaced patterns — the mechanism behind evenly spaced bristles, sensory cells, and so on.

Implement all three. They're each about 100 lines and they interact to produce genuinely surprising morphologies.

#### Differential adhesion → tissue sorting

If adhesion strength depends on key similarity, a mixed clump of two cell types will **spontaneously sort into layers**, with the more strongly adhesive type inside. This is Steinberg's differential adhesion hypothesis, it's real, it works in simulation, and it is startling to watch. It gives you the germ-layer structure that everything else in body plan evolution builds on, for free, from a mechanism you were already implementing.

#### Organs

Don't implement organs. An "organ" is what you *name* a spatially coherent cluster of cells sharing an expression profile and a function. Give the analysis tools the ability to detect and label such clusters (k-means on expression vectors + spatial connectivity) and you'll be discovering organs rather than authoring them. That's the version worth having.

---

### L6 — Nervous system

#### Neurons as cells

Each cell runs a small **continuous-time recurrent neural network** (CTRNN):

```
τᵢ · dyᵢ/dt = -yᵢ + Σⱼ wᵢⱼ · σ(yⱼ + θⱼ) + Iᵢ
```

with `τ`, `w`, `θ` from `Neural`-class proteins, and `I` from receptor proteins. Cap at ~16 neurons per cell. CTRNNs are the right choice over feedforward nets: they have internal dynamics, so organisms can have memory, oscillators (locomotion!), and state without you designing any of it.

#### Growing a nervous system

Here's the part that answers "the genome contains a code for a neural network" properly. Don't encode a network topology in the genome — that's brittle and doesn't scale. Instead, **grow the network developmentally**:

- Cells expressing `GapJunction` proteins connect to adjacent cells with **matching keys**. Connection weight = affinity × expression product.
- Cells expressing `AxonGuidance` proteins extend a process that follows a morphogen gradient (up or down, per the protein's params) until it reaches a cell expressing a matching receptor, then forms a long-range connection.

The organism's full nervous system is the union graph of all inter-cell connections. Its topology is an emergent consequence of body shape and gene expression, exactly as in real development. Retune a morphogen gradient with one mutation and the wiring changes globally. That's evolvability.

Store the graph as a CSR adjacency structure rebuilt every ~50 ticks (development is slow), then run the CTRNN over it as a sparse matrix-vector product every tick. Cheap.

#### Learning (optional, Phase 7)

Add a Hebbian-style plasticity rule whose parameters (learning rate, decay, whether it's applied at all per-synapse) come from the genome. Then **whether an organism learns within its lifetime becomes an evolved trait**. Watching a lineage discover lifetime learning is one of the more remarkable things this kind of system can produce.

---

## 5. Software architecture

### Language and stack

| | **Rust + wgpu** | **C++20 + CUDA** |
|---|---|---|
| GPU portability | Vulkan/Metal/DX12 | NVIDIA only |
| Raw compute ceiling | Good | Best |
| Refactoring safety | Excellent | Manual |
| Ecosystem for this | Bevy ECS, egui, rayon | Thrust, CUB, ImGui |
| Ramp-up from your background | Moderate | Immediate |
| Compile times | Painful | Painful |

**Recommendation: Rust + wgpu.** This project will be refactored a dozen times over its life, and the compiler catching your aliasing mistakes across a 40,000-line simulation is worth more than the last 20% of CUDA throughput. `wgpu` compute shaders will get you within striking distance of CUDA for the stencil work that dominates your cost. Bevy's ECS is a genuinely good fit for the agent layer.

Pick C++/CUDA instead if you want maximum particle count and you're certain you'll only ever run on your RTX 3060. That's the path ALIEN took, and it's a legitimate choice.

### Crate layout

```
lifesim/
├── crates/
│   ├── core/            # L0: units, RNG, fixed-point, determinism harness
│   ├── chem/            # L1: compounds, reactions, chemistry generation
│   ├── fields/          # L2: diffusion, light, heat, flow, wave channels
│   ├── genome/          # L3: encoding, decoding, mutation operators
│   ├── cell/            # L4: cell update, metabolism, expression
│   ├── organism/        # L5: adhesion, development, neural graph
│   ├── sim/             # orchestration, scheduling, snapshots
│   ├── analysis/        # phylogeny, diversity, energy audit, export
│   ├── headless/        # CLI runner — this is what runs on the server
│   └── viewer/          # 3D renderer, egui inspector
├── shaders/             # WGSL compute + render
├── configs/             # world presets (TOML)
├── ancestors/           # hand-written seed genomes
└── tests/               # determinism, conservation, unit
```

**Headless core with a separate viewer is essential.** Long runs go on the server; the viewer attaches over a socket or opens snapshots. Your Proxmox host with the 3060 passed through is exactly the right home for the headless runner — start a run Friday, look at it Monday.

### CPU/GPU split

| Work | Where |
|---|---|
| Diffusion, heat, light, flow | GPU compute |
| Reaction network in voxels | GPU compute |
| Cell physics integration | GPU compute (or CPU + rayon at <50k cells) |
| Cell metabolism + transport | GPU compute |
| Gene expression | GPU if genomes are flattened to GPU-friendly arrays; CPU otherwise |
| Genome decode, mutation, division | CPU (branchy, allocating) |
| Development, neural graph rebuild | CPU |
| CTRNN step | GPU (sparse matvec) |
| Analysis, phylogeny | CPU, off the hot path, on a worker thread |

The CPU↔GPU boundary is the thing to design around. Ping-ponging every tick will destroy you. Target: **one upload and one download per tick**, batched. Division and death events accumulate into a compacted buffer on the GPU and get drained by the CPU once per tick.

### Persistence and telemetry

- **Snapshots:** custom binary, zstd-compressed. Include world config, RNG state, all fields, all cells, all genomes.
- **Time series:** Parquet or SQLite. Population, per-clade counts, mean genome length, diversity, energy audit, reaction flux by ID.
- **Phylogeny:** every division records `(parent_id, child_id, tick, mutations)`. This grows without bound — periodically **prune extinct branches** (retain only lineages with living descendants, plus a downsampled fossil record). Without pruning you'll be writing gigabytes a day.
- **Dashboards:** push metrics to Prometheus, chart in Grafana. You already run this stack; watching population dynamics on a live dashboard while the sim runs headless is genuinely useful, not a gimmick.

---

## 6. Viewer

### Rendering

- **Cells:** GPU instanced spheres, or billboard impostors with analytic sphere raytracing in the fragment shader (cheaper, perfect silhouettes, millions feasible). Depth-write from the analytic intersection.
- **Bonds:** instanced capsules between adhered cells, or skip below a zoom threshold.
- **Chemical fields:** volumetric raymarching through a 3D texture, front-to-back with early-out. Selectable compound, adjustable transfer function.
- **Light field:** a screen-space tint plus optional god rays. Cheap and it sells the depth gradient.
- **Flow:** GPU particle advection, streamlines fading over time.

### Colour modes (a toggle, not a setting to bury)

| Mode | Shows |
|---|---|
| **Pigment** | Cell colour from its actual absorption spectrum → RGB. *The default.* Evolution of photosynthesis is visible with no instrumentation. |
| Clade | Colour by phylogenetic cluster. Speciation events look like the tree branching in space. |
| Energy | Blue→red by reserve. Starvation waves are visible. |
| Expression | Colour by a chosen gene's expression. Watch differentiation happen in a growing body. |
| Age | Newborns vs. old cells. |
| Neural | Activation of a chosen neuron. |

### Inspector

Click a cell, get a panel:
- Genome view with genes highlighted, mutations vs. parent diffed
- Live proteome with concentrations
- Reaction flux (which reactions are actually running, and how fast)
- Energy budget: intake, expenditure by category, net
- Neural graph with live activations
- Lineage: parent chain back N generations, sibling count, descendant count
- "Pin and follow" so the camera tracks it

### Global tools

- **Time control:** pause, step, 0.1× to 1000×. Above ~10× rendering becomes the bottleneck — decouple, render every Nth tick.
- **Phylogeny viewer:** interactive tree; click a node, jump to those organisms.
- **Population graphs:** stacked area by clade over time.
- **Intervention tools:** inject compound, change temperature in a region, kill a region, add a barrier, drop a "meteor." Disturbance is a research tool — punctuated equilibrium needs punctuation.
- **Genome editor:** hand-edit a genome and inject the organism. Essential for building your ancestor and for debugging.

Log every intervention into the replay event stream, or you lose determinism.

---

## 7. Roadmap

Each phase produces something you can actually look at. That matters for a project this long.

### Phase 0 — Skeleton (2–3 weeks)
Workspace, units, counter-based RNG, fixed timestep, determinism test harness. Viewer with camera and a raymarched empty volume. Config loading, snapshot round-trip.
**Done when:** 10k ticks of an empty world hash identically across two runs and across a save/reload.

### Phase 1 — Dead world (3–4 weeks)
Chemistry generation, reaction network, diffusion, heat, light columns, basic flow. Volumetric rendering of a chosen compound.
**Done when:** you can inject a blob of compound, watch it diffuse and react, and the global energy audit stays flat over 1M ticks. **Do not proceed until the energy audit is clean.**

### Phase 2 — Protocell (2–3 weeks)
Hardcoded cell, no genome. Membrane, transport, one metabolic pathway, growth, division, death, decomposition. Physics and rendering of cells.
**Done when:** a hand-tuned protocell population grows to fill its resource supply, crashes, and recovers — a logistic curve you didn't write. Corpses visibly feed the survivors.

### Phase 3 — Evolution ⭐ (6–8 weeks)
This is the big one. Genome encoding, decoding, proteins, mutation, inheritance. Replace the hardcoded protocell with a genome-driven one. Write the ancestor by hand. Phylogeny recording, diversity metrics.
**Done when:** you seed one ancestor, run overnight, and the population has measurably diverged — different metabolic strategies in different regions, and a lineage with a genuinely better energy efficiency than the ancestor that you did not design.

That result is the whole project's proof of concept. Everything after is elaboration.

### Phase 4 — Regulation and behaviour (4–6 weeks)
Gene regulatory network. Receptors, effectors, motility. Chemotaxis. Photoreceptors and pigments.
**Done when:** chemotaxis evolves on its own, and you can prove it — measure mean cell displacement projected onto the local resource gradient over generations, and see it climb from zero.

### Phase 5 — Multicellularity (6–8 weeks)
Adhesion, bonds, polarity, asymmetric division, morphogen gradients, juxtacrine signalling, differential adhesion. Optionally sexual reproduction.
**Done when:** clusters appear, persist across generations, and show differentiated expression by position. Watch a clump sort into layers.

### Phase 6 — Nervous systems (5–7 weeks)
CTRNN per cell, gap junctions, axon guidance, graph assembly, sparse neural step. Mechanoreception. Neural visualization.
**Done when:** a multicellular organism exhibits coordinated locomotion — cells firing in a travelling wave, not independently.

### Phase 7 — Ecology (4–5 weeks)
Lysis enzymes and predation. Day/night and seasonal cycles. Disturbance events. Full food-web analysis tooling.
**Done when:** you can plot a food web with three or more trophic levels and see predator-prey population oscillations.

### Phase 8 — Scale (ongoing)
Chunked streaming world, simulation LOD, larger organisms, far-field acoustics, deeper analysis. This phase never ends, which is the point.

**Realistic total to Phase 5:** 12–18 months of solid part-time work. Phase 3 alone will take longer than you expect, because the first six ancestor designs won't survive.

---

## 8. How to know it's working

Build these as automated checks, not eyeball tests.

**Hard invariants (build-breaking if violated):**
- Energy conservation to float precision
- Mass conservation per element
- Bit-identical replay from seed
- No cell's energy production exceeds the thermodynamic maximum of its reaction set

**Evolution is actually happening:**
- **Fitness proxy climbing:** mean offspring-per-lifetime, or biomass throughput per unit energy. Plot vs. generation.
- **Price equation decomposition:** split observed change into selection and transmission components. If transmission dominates, you're watching drift, not selection.
- **Genome length distribution:** healthy runs show it growing then stabilizing. Monotonic shrinkage means your mutation costs are too harsh.
- **Neutrality test:** re-run with selection disabled (all cells reproduce regardless of energy). If your fitness metric climbs anyway, your metric is measuring an artifact.
- **Ancestor replay:** resurrect the seed ancestor into a late-stage world. If it's outcompeted, adaptation is real.

**Ecosystem health:**
- Shannon diversity over clades — should be nonzero and fluctuating, not collapsing to one
- Trophic level count over time
- Spatial autocorrelation of clades — real ecosystems are patchy, uniform mixing means your spatial structure isn't doing anything

---

## 9. Failure modes

These are the ones that actually kill projects like this.

**Evolution stalls after 50 generations.** By far the most common outcome. Causes and fixes:
- *Fitness landscape too rugged.* Make the key-vector affinity functions smoother (raise σ). Every mutation should usually produce a small effect.
- *No gene duplication.* Check the operator is firing and duplicates aren't being immediately deleted for their cost.
- *Population too small.* Evolution's fuel is population × generations. 1,000 organisms is not enough. Get to 100,000.
- *Uniform environment.* No gradients, no niches, no divergence. Add spatial heterogeneity and periodic disturbance.
- *Mutation rate wrong.* Too low and nothing happens; too high and you get error catastrophe where the population can't retain adaptations. Sweep it; the good window is usually narrower than you'd guess.

**Free energy exploit.** Organisms find a reaction cycle that nets positive energy, and the world fills with them in twenty minutes. Prevented entirely by generating reactions from formation enthalpies so cycles sum to zero by construction. Add a startup check that walks the reaction graph looking for positive-energy cycles.

**Everything dies immediately.** Usually maintenance cost exceeds achievable metabolic yield. Instrument the ancestor's full energy budget by hand before you turn mutation on.

**One clade takes over and nothing else ever happens.** Add: spatial isolation (barriers), disturbance events that clear regions, frequency-dependent effects (dense monocultures get a pathogen-like penalty), and resource heterogeneity.

**The simulation is too slow to evolve anything.** Profile early. Diffusion is likely 60%+ of your time — it's also the most optimizable. Consider dropping to 16 compounds and a smaller grid during Phase 3, when you need generations more than you need detail.

**Scope death.** You have a plan with eight phases and any one of them could absorb a year. The discipline that saves this project: **each phase must produce something you'd want to show someone.** If a phase's deliverable isn't demoable, it's specified wrong.

---

## 10. Prior art worth studying

- **ALIEN** (`chrxh/alien`) — <cite index="3-1">a CUDA-powered artificial life simulator built on a specialized 2D particle engine, where each body is a network of particles that can be upgraded with higher-level functions from information processing to sensors, muscles, weapons and constructors, orchestrated by neural networks, with blueprints stored in genomes and passed to offspring</cite>. <cite index="10-1">It's optimized for large-scale real-time simulation with millions of particles, driven by the goal of understanding conditions for prebiotic evolution and growing biological complexity.</cite> This is the closest existing thing to your vision. It's 2D and particle-based rather than chemistry-based, which is exactly the axis where your design goes further. **Read its source before Phase 3.**
- **Avida** — the rigorous digital evolution platform. Study its ancestor-seeding approach and its instruction-set design for evolvability.
- **Tierra** — the original. Short, readable, and the lessons about parasites emerging unbidden are directly relevant to your predation goals.
- **PhysiCell / Morpheus** — serious agent-based multicellular biology simulators. Their cell mechanics and diffusion solvers are what you want to imitate for L4.
- **Framsticks** — evolved 3D morphology plus neural control. Good reference for the body-brain co-evolution problem.
- **The Bibites** — a polished hobbyist ecosystem sim. Useful mostly for UX and for what makes this genre *watchable*.
- **Lenia** — continuous cellular automata. Different approach entirely, but the "emergent self-organizing forms from simple rules" results are worth internalizing.
- **Papers:** Ray, "An Approach to the Synthesis of Life" (1991); Ofria & Wilke on Avida; Steinberg on differential adhesion; Beer on CTRNN agents; Wolpert's French flag model.

---

## 11. Hardware

Your RTX 3060 (12 GB) is well suited to this. A rough budget at Phase 3 scale:

| Resource | Estimate |
|---|---|
| Chemistry fields, 32 compounds @ 2.4M voxels, f16 | ~150 MB |
| Light (8 bands) + heat + flow | ~120 MB |
| 200k cells × ~600 B state | ~120 MB |
| Genomes (shared, ~5k unique × 8 KB) | ~40 MB |
| Render buffers, double-buffering | ~1 GB |
| **Total** | **~1.5 GB — comfortable** |

You're compute-bound, not memory-bound, which is the good problem. Headroom to go to 128 compounds and a 400×400×100 grid later.

For the headless runner on the Proxmox host: pass the GPU through to a single Linux VM, run in a container, expose Prometheus metrics, and attach the viewer over the network. Long runs are where the interesting results live — the good stuff shows up after generation 10,000, not generation 100.

---

## 12. If you only do one thing first

Build **Phase 1 and Phase 2** and nothing else. A dead world with correct, conserved chemistry, plus a hand-coded protocell that eats, grows, divides, and dissolves back into its compounds when it dies.

That's maybe six weeks. And when you have a boom-bust population curve emerging from a system where you never wrote the word "population," you'll know the foundation is right — and every layer after that is building on rock instead of sand.