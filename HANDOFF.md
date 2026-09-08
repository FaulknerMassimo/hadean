# Hadean — build log and handoff

Implementation of `PLAN.md` (a bottom-up artificial life simulation).
Last updated during the Phase 2 population-tuning pass.

**Status: Phase 0 complete, Phase 1 substantially complete, Phase 2 underway,
L3 genome implemented, L6 decision layer implemented.**
The first hardcoded protocell is integrated: membrane transport, catalysed
metabolism, energy reserve and maintenance, Brownian/flow motion, division,
death, and conservative decomposition. Cell state participates in audits,
hashes, snapshots, and CSV telemetry. Cells now carry a heritable trait and the
pond selects on it, which is the first evolution here. Since L3 they also carry
a genome, and since mutator alleles the mutation rate itself is heritable --
off by default; see "Mutator alleles, which the pond may now switch on for
itself".

**"The population is mining, not grazing" is still the diagnosis**, and the
gate was never blocked by the cell layer -- it was blocked by a metabolism
chosen off the wrong column. Three columns have now been wrong in a row, each
one more sophisticated than the last, so read the instruments before the
findings:

* "What the pond can resupply, which is not what it holds" -- rate, not amount.
* "The larder that held its rate" -- why `holding` endorsed a stock, and the
  element ledger that catches it.
* "The sweep, and a settle you pay for once" -- why one reading at one moment
  cannot tell a supply from a pond on its way down, and the settle cache that
  makes the second reading affordable.
* "The return leg" -- why none of the above bounds a *population*, and
  `hadean returns`, which does. Read its control-pond section: the failure it
  guards against passes the energy audit perfectly.

**Then read "`configs/vent5.toml`, and the confound in it", because it is the
first good news this question has produced.** With `vents.fuel_species = 5` the
vents inject oxygen, and at a 2200 s settle HO2M reads 1.192e10/s **renewed**
and **100% vent-fed** where `renew.toml` reads 4.543e4/s and `stock - no O
enters this pond`. Four livings come back fully funded, all paying a newborn
hundreds of times its upkeep, with the cliff two and a half orders below them
exactly where the carbon chemistry starts. It is the first diet in this project
to survive all three checks that have caught one -- rate, ledger, and sweep.

Two things it does not settle, both stated where the numbers are: the
magnitudes are inflated by a matter confound, and `supply` bounds a probe
rather than a population.

Finally, "The living the pond makes and the cell cannot reach" before tuning
anything.

```
mise exec -- cargo test --release --workspace          # 257 tests
mise exec -- cargo run -p hadean-headless --release -- verify --ticks 2000
  determinism ......... ok
  snapshot replay ..... ok
  energy audit ........ ok  (drift -1.277e-13 relative)
  drift trend ......... ok
```

---

## Environment

There was no Rust toolchain on this machine. One was installed **project-locally**
via mise (`mise.toml` pins `rust = "latest"`, resolved to 1.98.1). Nothing global
was changed.

**Every cargo command must be prefixed with `mise exec --`**, or run `mise install`
and let the shims take over.

There is one commit. The dormancy and lifespan-spread work described below is
uncommitted in the working tree.

No GPU is available in this environment (`nvidia-smi` fails), so everything is the
CPU reference implementation. `wgpu` has not been introduced yet.

---

## What exists

```
hadean/
├── PLAN.md                 the design (unmodified)
├── HANDOFF.md              this file
├── configs/pond.toml       the world: a pond under a ten-minute day
├── configs/vent5.toml      the same pond with oxygen coming into it
└── crates/
    ├── core/       L0  units, RNG, grid, clock, state hashing
    ├── chem/       L1  elements, molecules, reactions, kinetics
    ├── fields/     L2  transport, heat, light, flow, wave channels
    ├── cell/       L4  membrane, metabolism, lifecycle
    ├── sim/            config, world, audit, snapshots
    ├── analysis/       time series, drift trends, population curves, charts
    └── headless/       the `hadean` CLI
```

### L0 — `crates/core`

* `units.rs` — the single home for physical constants. Metres, seconds, joules,
  **particle counts** (not moles), kelvin. `N_REF = 1e9` particles per voxel is
  about 100 µM at 25 µm, so the normal operating band is 1e7–1e11.
* `rng.rs` — `Counter` is the counter-based generator: a pure function of
  `(seed, tick, entity, purpose, stream)`, so draws do not depend on thread
  scheduling or evaluation order. `SeqRng` is a stateful stream, legal **only**
  in single-threaded set-up code. `Purpose` values are part of the replay
  contract — never renumber an existing one.
* `grid.rs` — `z = 0` is the surface and z increases downward. Closed boundaries.
* `time.rs` — fixed timestep, plus `Schedule` with multi-rate intervals and
  `due_for` staggering by entity id.
* `hash.rs` — bitwise `f32` hashing with NaN canonicalised.

### L1 — `crates/chem`

Two properties are structural, not checked:

1. **Mass balance is impossible to violate.** Reactions are generated by *graph
   rewrites* on real atom graphs (`rewrite.rs`). An unbalanced reaction cannot
   be expressed.
2. **Free energy is impossible to manufacture.** A compound's enthalpy is the
   negative of its total bond energy — a function of structure alone — so every
   reaction's `dh` is a difference of potentials and any cycle sums to exactly
   zero.

Forward and reverse barriers are linked by `ea_r = ea_f - dh`, enforcing
detailed balance, so a catalyst can speed a reaction up but never move its
equilibrium.

Generation is a pure function of `(seed, ChemParams)`: primordials → random
assembly → reaction discovery that grows the rest of the pool → `close_network`.

`kinetics.rs` has two forms. `Chemistry` is the readable one; `Network` is the
same chemistry flattened for the inner loop, with a `RateTable` of rate
constants precomputed across temperature.

**Heat is derived from the measured change in chemical energy, not from
intended extents.** Naming is Hill notation, so ammonia reads `H3N`.

`Chemistry::reachable_from` walks the network to a fixed point from a set of
seed compounds, in both reaction directions. It is used for diagnosis rather
than for choosing anything — see "Choosing a metabolism" below for why.

### L2 — `crates/fields`

* `transport.rs` — diffusion and donor-cell advection in **face-flux form**, so
  the flux one voxel computes across a face is the exact IEEE negation of its
  neighbour's. **Carries a per-voxel residual** shared with reaction rounding.
* `heat.rs` — `HeatField` stores a **deviation from a reference temperature**,
  never absolute kelvin. At 300 K an `f32` resolves 3e-5 K while a reaction
  deposits microkelvins.
* `light.rs` — band-limited radiative transfer, 8 bands, Beer–Lambert down
  columns, depth-outermost. Returns `Insolation { incident, absorbed }`; the
  audit books `absorbed`.
* `flow.rs` — static convection roll from a stream function sampled at voxel
  corners, so discrete divergence cancels identically.
* `wave.rs` — the plan's unified `WaveChannel` trait. Built now, used properly
  from Phase 6.
* `scalar.rs` — `ChemField::settle` is the residual-carrying deposit; see "The
  leak" below.

### `crates/sim`

`WorldConfig` (TOML, `deny_unknown_fields` so typos are loud), `World` with the
tick order, `Audit`, and zstd snapshots. Snapshots do **not** store the
chemistry — it is regenerated from the seed and its digest checked.

`World::rebaseline_audit` exists for callers that legitimately reach into the
fields from outside the ledger (a test laying out food, a tool injecting a blob
to watch it diffuse). It is not for use inside a run.

`cargo run -p hadean-sim --release --example leak` is kept, not leftover: it
runs a closed world across several seeds and prints `drift / t` beside
`drift / sqrt(t)`, which is how a linear leak is told apart from a random walk.
Its header comment explains how to read it.

### L4 — `crates/cell`

The Phase 2 ancestor is intentionally hardcoded. It exchanges generated
compounds through a passive membrane, captures energy from one exergonic
pathway, pays continuous maintenance, moves with flow plus Brownian motion, and
divides after accumulating enough reserve. Age or starvation kills it;
decomposition returns its compounds and reserve to the fields.

A **pre-genome** cell that cannot pay its upkeep enters `CellState::Dormant`
rather than running up damage against a reserve it does not have: it drops to
`dormancy_power_fraction` of its bill and does not divide, but keeps its
membrane and its catalyst, because passive diffusion and a catalyst already
built are not decisions a cell gets to make. It wakes once it has banked
`dormancy_exit` seconds of working upkeep, a deliberately higher bar than
shutting down was -- without the gap a cell on the margin flickers every tick
and averages back into having no dormancy at all.

A **genome** cell does none of that. It arrives at `maintain` with a quiescence
*depth* on 0..1 that its own network chose, and `maintain` bills it; there is
no threshold, no hysteresis dial and no floor under a cell that decides badly.
`CellState::Dormant` survives only as a label applied to a cell more than half
shut down, so the counters and the snapshot keep meaning what they meant. See
"L6: the decision layer" below.

Dormant cells count as alive everywhere: in `alive()`, in the safety cap, and
in the population column, with a separate `dormant` column beside it because a
cohort sitting out a night and one working through it draw the same line. The
CSV also carries `mean_quiescence`, which is what the dormant count is a
threshold on: a pond at a mean depth of 0.95 and one at 0.55 report the same
dormant count and are not the same pond.

Lifespans are per cell. `lifespan()` draws on `Purpose::Death` as a pure
function of the cell's id, so a cell's allotted span is settled at birth,
survives a snapshot and costs no state. `lifespan_spread = 0` restores the old
shared lifespan exactly, which is worth knowing is a *mechanism* and not a
simplification -- see "What the crash actually was".

Cells carry **heritable traits**. Today that is one number, `uptake`, a
multiplier on the cell's membrane permeability; a daughter inherits her
mother's with a log-normal kick of `trait_spread`, and the founding cohort is
drawn the same way. `trait_cost` is what stops it being a free dial: it is the
fraction of a cell's upkeep that scales with the machinery it carries, so a
hungrier cell is also a more expensive one. At `uptake = 1` the bill is exactly
`maintenance_power` whatever `trait_cost` is, so a population of clones is
unaffected by either dial and `trait_spread = 0` is the old cell layer exactly.

This is a vestigial genome and it is deliberately not the one `PLAN.md`
specifies -- a fixed struct of scalars cannot duplicate a gene, and gene
duplication is the whole reason that plan calls for a byte string. What it is
for is to carry inheritance, mutation and selection through the tick, the
audit, the digest and the snapshot now, so that L3 replaces a mechanism that
already works rather than introducing one. It is also *state*: unlike a
lifespan, a cell's traits are its ancestry and cannot be re-derived from its
id, so they are written into snapshots. The snapshot format is at **6**: 5
added the genome bytes and the proteome, 6 adds the network's activations,
which are per-cell state for the same reason the proteome is -- two sisters
carrying identical bytes carry neither identical protein nor identical
opinions. `Cell::quiesce` is deliberately *not* in it, or in the digest: it is
a readout recomputed at the top of every tick, and writing it down would be a
second copy of something already there.

`neural.rs` is the interpreter for `Receptor`, `Neural` and `Effector`, and it
is where every decision a genome cell makes is taken. `sense_and_think` reads
the receptors and advances the neurons; `Expression::read` runs the effectors
and hands `exchange`, `move_cell`, `maintain` and the division check what they
decided. `Sensorium` is the part of the world a receptor can see that the cell
layer does not already hold -- light, today -- and `Sensorium::dark()` is the
honest answer for a test that has no sun. See "L6: the decision layer".

Cells are stepped in **voxel order**, not birth order. The order has to be
fixed, because cells clamp against what the previous cell left in their shared
voxel; making it spatial is what keeps it fast. A cell reads one voxel out of
every compound plane, and the planes are hundreds of kilobytes apart, so a cell
costs about thirty cache misses — two cells in neighbouring voxels want the
same thirty cache lines.

### `crates/analysis`

* `series.rs` — the CSV row. `Row` derives `Default` so tests name the columns
  they care about; the schema grows without touching them.
* `drift.rs` — the energy-audit trend test. See "Judging a drift" below.
* `population.rs` — reads boom, bust and recovery out of a population column.
* `chart.rs` — terminal charts, because the Phase 2 gate is a claim about a
  shape and there is no viewer yet.

### `crates/headless`

```
hadean config                      # print the default config as TOML
hadean chem [-r] [-l N]            # describe the generated chemistry
hadean run --ticks N [--csv F] [--out SNAP] [--resume SNAP] [--progress N]
hadean verify --ticks N            # determinism, replay, conservation
hadean ecology --ticks N           # the population gate: boom, bust, recovery
hadean profile --ticks N [-l N]    # vertical structure and abundances
hadean bench --ticks N             # throughput
hadean supply  [--settle A,B,C] [--probe S] [--settle-cache DIR] [--only NAME]
hadean returns [--settle A,B,C] [--probe S] [--settle-cache DIR]
               [--reaction ID | --metabolism "HO2M + HM"]
```

`supply` and `returns` are the two food instruments and they answer different
questions -- see "What the pond can resupply" and "The return leg" below. They
share the settled-world cache: the pond they settle is the same lifeless world,
so a depth one of them pays for is free for the other.

`hadean verify` is the conservation gate and `hadean ecology` is the population
gate. Both exit non-zero on failure; both belong in CI.

---

## The leak (read this before touching fields, chemistry, or cells)

The most consequential class of bug found, and the reason `verify` has a
*trend* check and not just a threshold.

A voxel holds ~2e11 particles, where an `f32` step is 16384. Once a field
smooths out, a step's net diffusive flux falls *below half a step*, so
`new = centre + acc` rounds back to `centre` and the flux is silently dropped.
The dropped amounts do not cancel between voxels.

Fix: every transport operation carries a per-voxel `residual` — what the last
rounding threw away is added back before the next one. Nothing is discarded,
only deferred. Drift fell from 7.9e-8 to 7.0e-11 and the trend went flat. The
same treatment was then needed for reaction stoichiometry and heat deposits.

**The same bug then turned up in the cell layer, wearing a different hat.** A
cell is a ten-thousandth of a voxel by volume, so nearly every transfer it
makes is far below an f32 step of the field it is drawing on. Worse, both sides
were written to only give away what actually landed — a correct instinct that
turns a dropped transfer into a *permanent* one, because the same too-small
amount is requested again next tick and fails again. Corpses stopped
decomposing partway and sat in the world for ever: a 42000-tick run finished
holding 4284 corpses that had been "decomposing" for eight hundred seconds,
their matter never reaching the survivors. `ChemField::settle` now carries the
remainder in the residual for membrane exchange and decomposition too.

Decomposition also needed a floor. Exponential decay never reaches zero, so a
corpse whose remainder is smaller than a single particle would be carried, and
stepped, for ever; below `ONE_PARTICLE` it hands over the rest in one go.

**Consequences to respect:**

* `residual` is **conserved state, not scratch**. It is in the state digest, in
  snapshots, and in both chemical and thermal audit totals. Anything that adds
  a new transported or rounded field needs its own residual.
* "Book what actually happened, not what was intended" applies at every
  boundary: `HeatField::deposit` and `ChemField::settle` both return what
  landed, and the light solver reports what was *actually* absorbed.

---

## Judging a drift

Once the residuals were in, the remaining drift stopped being white noise. It
became a smooth, bounded wander — f32 accumulation tracking a world that is
itself evolving smoothly — and a smooth curve fits a line with almost no
scatter. The old trend test asked only "is the slope bigger than the scatter?",
so it fired on every run, and the sign of the "leak" flipped depending on how
long you watched: +2.1e-19 J/tick over 500 ticks, -9.8e-20 over 2000.

`Trend::is_growing` now asks two questions. Is the slope real (significance),
**and** would it matter — extrapolate to the million ticks the Phase 1 gate
names and compare against the audit's own tolerance. The leak this module was
built to catch projects to about 1e-5 relative over that horizon. What is there
now projects to 1e-10, five orders clear. Both cases are regression-tested in
`drift.rs`.

---

## The pond used to boil

`surface_coupling` was a Newton cooling **rate** in 1/s applied to the top
layer. Two things were wrong with that.

It was a property of the mesh, not of the water: the same pond meshed at 25 µm
instead of 50 µm has half the heat capacity behind each square metre of sky, so
halving `dx` halved the cooling and moved the equilibrium temperature. There is
now a test that refines the mesh and checks the heat loss does not move.

And at 0.02/s it was far too weak. The pond is half a millimetre deep and holds
3 mJ/K, so under 590 W/m² it takes in roughly its entire heat capacity every
four seconds. It had no equilibrium below boiling: a run passed 320 K at 160
seconds and was still climbing 0.25 K/s.

`surface_transfer` is now a heat transfer coefficient in W/(m²·K), converted to
a layer relaxation rate by dividing by `rho c dx`. The default is 60, which is
high next to the 10–25 of bare convection because an open water surface loses
most of its heat as latent heat of evaporation. The pond now sits between 291
and 299 K across its day.

---

## Life arrives after the world does

`cells.seed_delay` (seconds) holds the founding cohort back while the pond
runs. This is not a convenience. A generated chemistry starts with only its
primordials, so at tick zero nothing photochemical exists — including, in most
seeds, the compound the ancestor eats. Cells introduced then starve in an empty
larder before the sun has made the first meal. It is also the more honest
picture: the world predates life in it.

Ancestors arrive with no contents and no reserve, so seeding is invisible to
the audit and can happen at any tick. The seed tick is derived from the config
rather than stored, so a snapshot taken before the cohort arrives still
introduces it at the right moment.

## Choosing a metabolism

`choose_metabolism` picks the most exergonic thermal reaction whose substrates
**the pond has actually accumulated**, measured at the moment life appears.

It took two wrong versions to get there, and both failed quietly. The first
asked "is this substrate a product of a photochemical reaction?" — one hop, and
on seed 5 it chose a compound whose photoreaction has no substrates in that
world either. The second asked the reaction network properly, with
`reachable_from` — and on seed 4 chose a compound that is reachable through a
strongly uphill chain and never accumulates at all. In both cases the founding
cohort sat in the pond, alive, with a food supply of exactly zero, until it
starved. Nothing else looked wrong.

Measurement settles it. It also means **not every generated world offers a
living**, and that is the right answer rather than a bug: seed 4's pond
populates eleven of its thirty-two compounds and none of its downhill reactions
runs on what is there. `hadean ecology` says so and exits non-zero.

That seed 4 populates only eleven compounds is worth a look before Phase 3. A
world where two-thirds of the chemistry is unreachable from the primordials is
a thin substrate for evolution, and the plan's third generation stage
(discovery growing the pool) is supposed to prevent exactly that.

---

## Performance

`hadean bench` on this machine (22 threads, 48×48×20 = 46080 voxels, 32
compounds, 88 reactions): **~49 ticks/s, 20 ms/tick** with no cells.

What the 5× from the original 9.7 ticks/s came from, in order of size: light
(79 → 6 ms, transparent compounds were being given an absorption floor);
chemistry (`Network` + `RateTable` removed ~8M `exp()` calls per tick);
transport parallelised over *compounds* rather than voxels; blocked transpose.

Cells are memory-bound, not compute-bound, and only in the large world: at
5760 voxels the whole field is 737 KB and four thousand cells cost 0.3 ms/tick,
while at 46080 voxels the planes are 184 KB each and ten thousand cells cost
55 ms/tick. Hence voxel-ordered iteration and the flattened `CompoundTable`.
**The speedup has not been measured at full scale yet** — the tuning runs that
would have shown it were all on the small world.

**Small worlds do not want all the threads.** Parameter sweeps run on
development-sized ponds, and a 1440-voxel world across 22 threads is 65 voxels
per worker — mostly dispatch. Measured on that world: 165 ticks/s on one
thread, 277 on two, 388 on four, 417 on eight. So set `RAYON_NUM_THREADS` to
about eight and run three sweeps side by side rather than one at a time; it is
roughly 2.5x the total throughput.

Chemistry generation is 80–250 ms at 32 compounds and grows fast with pool size
(~950 ms at 64). The `n_reactions` config field is a **cap, not a target**.

---

## Phase 2: where the population tuning stands

This is the live piece of work. The gate is: *a hand-tuned protocell population
grows to fill its resource supply, crashes, and recovers — a logistic curve you
didn't write. Corpses visibly feed the survivors.*

**What is done.** The machinery to judge it, and to judge it honestly:

* `analysis/population.rs` finds the boom, the bust and the recovery in a run's
  population column, and refuses two failures that look like success from a
  distance. A population sitting on its `population_cap` has not found its
  carrying capacity, it has found a number in a config file. A population that
  booms and goes to zero has a boom and a bust and no ecosystem.
* Every sample gets a turn as the peak. Taking the highest one reads a cycling
  world from wherever its best cycle fell; if that is the last one, there is no
  room left in the run for the crash and recovery that already happened twice.
* `hadean ecology` runs a world, draws population, corpses and food, and exits
  non-zero unless the shape is there.
* The cell layer has the corpses-feed-the-survivors mechanism isolated as a
  unit test: an empty pond, one starving cell, one corpse in the same voxel,
  and the only route from the corpse's contents to the survivor's reserve is
  decomposition into the field and uptake back out of it. With decomposition
  effectively off, the survivor gets nothing.

**The energetics were out by ten orders of magnitude.** The original ancestor
divided on 5e-18 J, about fifty reaction turnovers, against a voxel holding 1e9
substrate particles. The population was limited by nothing but the safety cap,
which it reached in twenty seconds from twenty-four founders. It is now sized
against the pond's actual photochemical output. `configs/gate.toml` carries
where that got to: `maintenance_power` 1e-11 W, `division_reserve` 1e-8 J,
`membrane_scale` 6000.

**What the world turned out to be like**, measured rather than assumed:

* Seed 1's ancestor eats `H2 + HO3M`, and HO3M is photochemical (`O2 + HOM`,
  203 kJ/mol uphill, band 7). So the food chain is genuinely light-driven, and
  it stops dead at sunset.
* A cell breaks even at a fixed food **concentration** `C*`, and it is the
  population that adjusts, not the concentration. Cells therefore have no
  refuge from a shortage: when food falls below break-even it falls below
  break-even for every cell at once, and mortality is a step function rather
  than something graded that could thin the population gently.
* The lifeless pond accumulates food to something like twenty times `C*`
  before anything is eating it, so a population introduced into that larder
  has a great deal of room to overshoot.
* The carrying capacity is around 200–500 cells in the 24×24×10 gate world —
  0.03 to 0.09 per voxel, which would be one to two thousand in the full pond.
  **Measure it, do not derive it.** Reasoning forward from irradiance and
  photon yield overstates it by more than an order of magnitude, because most
  of the light never reaches the one compound the ancestor can eat. Set
  `initial_count` to a candidate and watch the food column: at the carrying
  capacity it holds roughly level, above it the column falls.

**Where that leaves the tuning.** Every populated run so far has produced a
textbook boom, a textbook bust, and then extinction. Four things had to be
measured before that made sense, and the fourth overturned the first
explanation.

1. *There is no refuge.* A cell breaks even at a fixed food **concentration**,
   and it is the population that adjusts, not the concentration. So a shortage
   is a shortage for every cell at once, and mortality is a step function
   rather than something graded that could thin the population gently.
2. *Famine tolerance alone does not help.* Raising `starvation_time` from 120 s
   to 400 s and then 800 s only delayed a synchronised extinction: a population
   that hangs on keeps consuming whatever trickles in, and so prevents the
   recovery that would have saved it.
3. *The overshoot is the thing to damp.* The population's response time has to
   be long next to the resource's. `division_reserve` is that dial —
   `response ≈ division_reserve / surplus power`. At 3e-10 J the response time
   is fifteen seconds and the population overshoots its carrying capacity
   seventeenfold; at 3e-9 it was past sixfold and still climbing; 1e-8 brings
   the two timescales within a factor of two.
4. **The sun has to move.** This one invalidated a decision and a whole
   preset. `configs/bloom.toml` fixed the sun in the sky, on the reasoning that
   under a moving sun a crash is ambiguous between "ate its food" and "night
   fell". The control that should have been run first — one cell, with
   `division_reserve` set to 1e30 so it can never divide, so that nothing is
   meaningfully eating — shows that under a fixed sun the food supply is not a
   steady state at all. It is a startup transient: HO3M peaks at 1.27e12
   particles around 250 s and then decays to **forty particles by 1500 s, and
   one by 1750 s**. The photoreaction drains its own precursor and nothing
   regenerates it.

   Under the ten-minute day the same pond keeps its food: it rebuilds to
   3.8e12 on the second morning and is still holding 1.9e12 at the end of the
   second night. The night is when the thermal network puts the precursor
   back. **In this world the day/night cycle is not a confound in the
   experiment, it is the renewable resource the experiment is about.**
   `bloom.toml` has been deleted; the gate runs on a pond with a day in it.

That also re-reads the earlier failures. The population was not defeating a
slow nutrient cycle; it was stripping, in a single afternoon, a larder that the
night would have refilled — 3579 cells by 251 s, food down from 2.0e12 to
2.1e11 by 301 s, and nothing left to live on when the sun went down. Damp the
boom so the population never strips the day's food, and it should meet the
night with the larder the night is able to keep.

**What that fixed.** The diurnal pond at 24×24×10, `starvation_time` 300 (the
length of the night), `maximum_age` 1200, and the division cost as the dial:

| `division_reserve` | peak | outcome |
| --- | --- | --- |
| 2e-10 (old default) | 4284 | stripped the day's food; extinct on the first night |
| 3e-9 | 498 | survived three nights; extinct on the fourth |
| 1e-8 | 199 | survived three nights; 199 → 82 → 28 → 4 over the fourth and fifth |

Damping works, and it is not enough. Each step slows the collapse and pushes it
a night further out; none of them stops it. The last of these is checked in as
`configs/gate.toml` — `pond.toml` in every respect but the grid size and three
lifecycle values, with a test enforcing that, because the last time this
project grew a second preset it quietly changed the physics.

**Why none of it is enough.** A population at its carrying capacity consumes
exactly what the world produces — that is what carrying capacity means. Across
a night, when photochemistry produces nothing, it therefore has to eat

```
    night bill = production rate x length of the night
```

particles out of whatever is left standing at dusk. Write it out and every
per-cell term cancels:

```
    N* = P e c / m                     cells the production supports
    bill = N* x m x T / (e c) = P T    particles they need to see morning
```

`m` is gone. **Lowering `maintenance_power` does not help**, because it raises
the standing population by exactly the factor it lowers each cell's bill. Nor
does damping the boom, which only decides which night the population dies on.
The pond produces on the order of 1e10 particles a second in the gate world, so
one night costs about 3e12 — and the population had drawn the pond down to
between 3e9 and 4e10 by dusk. One to two orders of magnitude short.

What *does* move is the standing stock the population leaves behind. A cell
breaks even at a concentration

```
    C* = m / (k e c)          k = permeability x membrane_scale x area
```

and the population eats down to `C*` and stops. `membrane_scale` is therefore
the dial that sets how much food is still in the water at dusk: efficient cells
strip the pond bare, less efficient ones cannot be bothered with the last of
it. It was raised from 20 to 6000 early in this tuning to make cells hungry
enough to matter, and that is what let them strip the pond to nothing. The
arithmetic wants `C* x volume` to be about a night's production, which puts
`membrane_scale` near 1000, and that is what `gate.toml` now carries.

It is the dial that moved the curve furthest. At 6000 the population reached
199, held three nights and then unravelled — 82, 28, 4, gone. At 1000 and at
400 it settles at 135–140 and holds three nights with **no deaths at all**,
then takes a real die-off on the fourth rather than a collapse:

| `membrane_scale` | plateau | fourth night | fifth day |
| --- | --- | --- | --- |
| 6000 | 199 | 82, then 28 | 4 |
| 1000 | 136 | 26, and 110 corpses | 0 |
| 400 | 140 | 60, and 80 corpses | 1 |

So it is much closer and it is still not the gate. What the pond now does is
grow into its food supply, hold there for three days and nights, and then die
in two nights instead of one. The die-off itself is the right shape — 136 cells
down to 26 with a hundred and ten corpses decomposing is a bust with survivors,
which nothing before this produced — but the survivors do not come back.

`gate.toml` carries 1000, as the best-understood configuration rather than a
passing one. **The gate is not met.**

The other way out is a mechanism rather than a dial, and it is the honest one.
A cell that cannot pay maintenance currently keeps trying, out of a reserve it
does not have, accruing damage at a fixed rate until it dies. A real cell
facing famine stops: it drops to a small fraction of its working power and
waits. That makes the night bill small directly, rather than by arranging for
leftovers, and it removes the perverse dynamic where a starving population
keeps eating the trickle and so prevents the recovery that would save it.
`PLAN.md` files behaviour under Phase 4; this world needs it in Phase 2,
because this world has nights.

---

## What the crash actually was

Everything above this line is the tuning as it stood, and the diagnosis in it
is wrong. It is kept because the measurements are real and because the way it
was wrong is worth knowing.

Dormancy went in as the mechanism the section above asks for, and it works: a
cell that cannot pay its full upkeep shuts down to `dormancy_power_fraction` of
it, keeps its membrane and its catalyst running because neither is a decision
it gets to make, and so breaks even at a food concentration that much lower.
The unit tests hold it to that. Then it was run against the gate, with a
control that disables it, and the control reproduced `configs/gate.toml`'s
documented behaviour exactly -- 136 cells, then 26 with 110 corpses, then zero.

**Dormancy made almost no difference, and `dormancy_power_fraction` turned out
not to matter at all.** 0.05, 0.005 and 0.0005 -- a four-thousandfold range --
produce the same trajectory to the cell. A dial that inert is not a weak dial,
it is a dial attached to nothing, and it says the population was not dying of
its maintenance bill.

Two experiments found what it was dying of.

**One: the crash was ageing, not famine.** `maximum_age` was 1200 s and the
die-off lands at t = 1250-1500. Setting `maximum_age` to 1e9 -- or merely
doubling it to 2400 -- makes the die-off *vanish*: 136 cells and zero corpses
straight through the night that had been killing a hundred and ten of them.
Every cell died at exactly the same age, so the cohort that boomed together
aged out together. The night was innocent. So was the night bill, and so is the
whole `P x T` argument above as an account of *this* crash -- the arithmetic is
right, it just was not what was happening.

That run is also where dormancy finally shows itself doing its job: with ageing
out of the way, 103 of 136 cells sit shut down while the pond is stripped to
4.6e6 particles, and not one of them dies. Under the old cell layer that
population was dead.

**Two: nothing is ever born after the boom.** Across all six runs -- control,
dormancy at three fractions, and both ageing variants -- the counters read
`births 136, deaths 136` against a peak of 136. Not one division after
t = 790 s, in any configuration.

That is the missing recovery leg, and it is arithmetic rather than ecology: a
population cannot recover without births. A cell divides on `division_reserve`
= 1e-8 J, which is about five hundred seconds of accumulating at full surplus,
and a population that has settled at its break-even concentration has by
definition no surplus at all. The plateau is not a carrying capacity. It is a
hundred and thirty-six identical clones arriving at break-even together and
freezing there.

Giving every cell its own lifespan (`lifespan_spread`, a draw on `Purpose::Death`
that is a pure function of the cell's id) desynchronises the die-off exactly as
intended -- deaths become a trickle from t = 1000 rather than a wall at 1400 --
and it is still not enough on its own. The food one corpse returns is a
hundred-and-thirty-sixth of the standing stock, spread across the field for
everyone, and no survivor can bank 1e-8 J out of a transient that small.

**So the live tension is between two jobs `division_reserve` is doing at once.**
It has to be large to damp the boom -- that is the finding above, and it holds --
and it has to be small enough that a cell can divide on the surplus a death
frees, or the plateau has no turnover and there is nothing to recover with.

That pairing was then run, staggered mortality with a cheaper division, and it
does not resolve it. Lower division costs buy a bigger boom, not a turnover:

| `division_reserve` | peak | after the boom |
| --- | --- | --- |
| 1e-8 | 136 | 0 births, gradual decline to extinction |
| 3e-9 | 432 | 0 births, strips the pond, 100% dormant by t = 1000 |
| 1e-9 | 1031 | 0 births, strips the pond to 4.6e3, 100% dormant |

The population overshoots until the pond is stripped, shuts down *entirely*,
and then declines. It never divides again at any setting.

Dormancy also introduces a failure mode of its own, worth knowing before
reaching for it: **a permanently sleeping population.** From t = 1000 the whole
population is dormant, and it cannot clear `dormancy_exit` to wake because the
pond never recovers far enough while they are all still drawing on it. The
obvious fix is the wrong one -- dropping `dormancy_exit` from 60 to 5 makes the
run worse, not better, because cheap waking puts cells back onto full spending
in a pond that cannot support it, and both populations then go extinct outright
rather than declining. The high bar was doing useful work.

**Where that leaves it.** Ten runs, and `births` equals the peak population in
every one of them. The gate's missing leg is not survival, tolerance, or
energetics; it is that nothing is ever born after the boom. Whatever comes next
has to make a plateau that turns over -- births and deaths both nonzero at
carrying capacity -- and no combination of the existing dials produces one. The
population regulates by every clone simultaneously arriving at break-even and
stopping, which is a freeze, not a carrying capacity, and the freeze is what
has to go. Candidates, in the order I would try them:

* **Variance between cells in what they need**, not just in how long they live.
  Identical clones in a shared voxel is what makes break-even collective. Some
  heritable spread in `membrane_scale` or `metabolic_rate` would mean the
  population thins from the bottom while the best cells keep a surplus -- and
  it is L3's job anyway, which is an argument for letting the genome arrive
  before the gate rather than after it.
* **A refuge**, so a shortage is not simultaneous everywhere. The pond is
  0.25 mm deep and well mixed; the light gradient is the only structure in it.
* **Density-dependent division** rather than a fixed reserve threshold, so the
  cost of dividing falls as the population thins.

What each dial does, so the next person does not re-derive it:

* `division_reserve` sets how fast the population can respond, and it has to be
  slow next to the resource. Everything else follows from getting this wrong.
* `starvation_time` has to exceed the night. It cannot substitute for the
  above: at 400 s and 800 s with an undamped population it only delayed a
  synchronised extinction.
* `maintenance_power` sets the standing population, and cancels out of the
  night bill entirely. It is not the dial it looks like.
* `membrane_scale` sets the concentration the population stops eating at, and
  so how much food is left standing when the sun goes down.
* `maximum_age` was doing far more than ageing. With `lifespan_spread` at zero
  it is a synchroniser: it decides the date on which an entire cohort dies at
  once, and that -- not the night -- was the Phase 2 crash.
* `lifespan_spread` breaks that synchrony. It converts a cliff into a trickle,
  which is necessary for turnover and not sufficient for it.
* `dormancy_power_fraction` sets how far below break-even a shut-down cell can
  still live. It is inert while something else is doing the killing, which is
  how the ageing diagnosis was found.

---

## Traits, and what variance is actually worth

The section above ends by asking for variance between cells in what they need,
and notes that it is L3's job anyway. That is what went in next: a `Traits`
struct on the cell, one field, `uptake`, a multiplier on membrane permeability;
a log-normal kick of `trait_spread` at every division, and the founding cohort
drawn the same way. `trait_cost` is the brake -- the fraction of a cell's
upkeep that scales with the machinery it carries -- and it is neutral at
`uptake = 1`, so `trait_spread = 0` is the old cell layer to the bit.

**The control says so.** `gate.toml` with `trait_spread = 0` reproduces the
documented run exactly: peak 136 at tick 79400, births 136, deaths 136,
extinct. Nothing about the new code perturbs a population of clones, and the
conservation gate is unmoved -- `verify` still reports `-1.277e-13` relative
drift, to the digit.

**And at `trait_spread = 0.2` it changed almost nothing.** Peak 140 instead of
136, and `births` 140 against a peak of 140: still not one cell born after the
boom. The variation was really there -- the founding cohort opened at a spread
of 0.19 and mutation carried the living population to 0.32 by the plateau --
and it bought two divisions at t = 750 and none afterwards.

### Why, in closed form

This is the part worth keeping. Write a cell's income and its bill:

```
    income(u) = k u C e            bill(u) = m (1 - c + c u)
```

where `u` is the uptake trait, `c` is `trait_cost` and `m` is
`maintenance_power`. A population eats down until its *marginal* cell breaks
even, and that pins the water at `k C e = m (1 - c + c u_m) / u_m`. Now put a
cell a fraction `d` better off than that one into the same water:

```
    surplus = m (1 - c + c u_m)(1 + d) - m (1 - c) - m c u_m (1 + d)
            = m (1 - c) d
```

Everything but `d` and the two dials cancels. So the best cell in the pond
funds a division in `division_reserve / (m (1 - c) d)` seconds, and it has to
do that inside a lifetime:

```
    division_reserve  <  maintenance_power (1 - trait_cost) d maximum_age
```

For the gate at `d = 0.32`: `1e-11 x 0.5 x 0.32 x 2400 = 3.8e-9` J against a
`division_reserve` of `1e-8`. **Short by a factor of two and a half**, which is
exactly why nothing was born. The cells were varied; none of them could afford
a daughter before it died of old age. `CellConfig::turnover_budget` computes
this and `hadean ecology` prints it before the run, because it is forty minutes
you can decline to spend.

Two things follow that are not obvious from the earlier analysis:

* **`maintenance_power` does not cancel here.** It cancels out of the night
  bill -- that argument is in the section above and it still holds -- but a
  bigger bill means a bigger income at break-even, so the same *fractional*
  advantage is a bigger *absolute* surplus. It is the one place where making
  cells more expensive makes the population more alive.
* **The inequality is generous.** `income = k u C e` holds only while the
  membrane is the bottleneck, and a cell that can take up faster than it can
  metabolise gets less from each extra transporter: measured, `u = 1.5` earns
  1.38 times the marginal cell rather than 1.5. A real population needs more
  spread than the arithmetic asks for, not less. Against that, the best cell of
  a few hundred is two or three standard deviations out rather than one, which
  pushes back the other way.


### What raising the spread bought

Three runs at 250000 ticks on `gate.toml`, against the clone control:

| run | `trait_spread` | `trait_cost` | `division_reserve` | peak | trough | finish | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0 | -- | 1e-8 | 136 | 0 | 0 | extinct |
| gate | 0.2 | 0.5 | 1e-8 | 140 | 0 | 0 | extinct |
| A | 0.5 | 0.5 | 1e-8 | 156 | 5 | 5 | no recovery |
| B | 0.5 | 0.25 | 1e-8 | 142 | 0 | 0 | extinct |
| C | 0.5 | 0.25 | 3e-9 | 420 | 8 | 8 | no recovery |

A and C are the first runs in this project that are **not extinct** at the end,
and A's plateau is 156 against the control's 136 on the same food. But
`births_after_peak` is zero in all five, so the recovery leg is still missing
and the gate is still not met.

**Selection is visible, and it runs both ways.** In A, `trait_cost = 0.5`, the
five survivors have a mean uptake of 0.33 against the ancestor's 1.0: when the
pond has been stripped, the cell that outlives the others is the *cheap* one.
In C, `trait_cost = 0.25`, machinery costs less and the eight survivors come in
at 1.63. Same mechanism, opposite direction, decided by which side of the
trade-off the world is rewarding. That is the first evolution this simulation
has done, and the dial that decides its direction is `trait_cost`.

---

## The population is mining, not grazing

Then the food column was read properly, and it says something that reframes all
of the above.

Take the largest amount of food standing in the pond during each five-minute
window of a run:

| window | control (clones) | A (spread 0.5) |
| --- | --- | --- |
| 0-300 s | 1.15e13 | 1.15e13 |
| 300-600 | 6.01e12 | 6.46e12 |
| 600-900 | 1.19e12 | 1.15e12 |
| 900-1200 | 2.30e11 | 1.91e11 |
| 1200-1500 | 8.61e10 | 8.08e10 |
| 1500-1800 | 7.01e9 | 5.76e9 |
| 1800-2100 | 6.37e9 | 6.75e9 |
| 2100-2400 | 1.03e9 | 1.37e9 |

**Four orders of magnitude, and the two runs track each other**, though one
carries 136 cells and the other 156. Every night's rebuild reaches roughly a
tenth of the last one. There is no plateau to turn over because there is no
carrying capacity: the pond is not producing at a rate the population lives
off, it is a larder being emptied.

The control that settles it is a pond with `membrane_scale = 0.001` and one
cell that cannot eat -- the same trick that caught the fixed sun. **It holds
1.15e13 flat through the entire run.** So the decline is the population's
doing, not the pond running itself down.

And the chemistry says why. On seed 1 the ancestor eats

```
    H2 + CH2O2S  ->  H2O + CH2OS          the metabolism, -82 kJ/mol
    H2S + CO2    ->  CH2O2S               band 2, +81 kJ/mol, the only source
```

The food is made photochemically out of H2S and CO2, and **CO2 takes part in
exactly one reaction in this chemistry** -- that one. So the pond's carbon has
one way in to the food chain and, once the cells have turned CH2O2S into
CH2OS, no way back out. The route home does exist and is downhill --
`O2 + CH2OS -> CH2O3S` then `H2 + CH2O3S -> H2O + CH2O2S`, paid for by burning
hydrogen -- but the first step has a rate constant of 8e-15 at 20 C, which is
no route at all. Nothing in this world empties the sink. The waste is a dead
end, and every turnover takes one more carbon out of circulation for good.

That is not a cell-layer problem and no dial in `CellConfig` reaches it. **The
population is not failing to regulate; it is mining a finite pool and the pool
is running out.** A boom, a bust and a recovery need a renewable resource, and
seed 1 does not have one for this metabolism.

### What that means for choosing a metabolism

`choose_metabolism` has now been wrong three times in the same direction, each
time by trusting something that looked like food:

1. one hop from sunlight -- picked a compound whose photoreaction had no
   substrates;
2. reachable in the network -- picked one at the end of an uphill chain that
   never accumulates;
3. **measured standing stock -- picks one the pond holds a great deal of and
   cannot make any more of.**

A large larder is not a living either. The measurement it wants next is a
*rate*, not an amount: what the world can resupply per second, which is what a
carrying capacity is made of. Two candidate readings, and neither is a graph
walk:

* **The vents are a known flux.** `fuel_rate x vents` particles per second of
  each of the top `fuel_species` compounds, straight out of the config. A
  metabolism running on vent fuel has a renewal rate that is not in doubt.
* **A perturbation.** Take some of a candidate substrate out of the settled
  lifeless pond and watch how fast it comes back. That is the honest general
  answer and it costs a short extra run at set-up.

Other seeds do offer vent-fed livings, which is the encouraging half of this.
Probed at 16000 ticks:

| seed | metabolism | substrates |
| --- | --- | --- |
| 2 | H2 + O2 -> H2O2 | both vent fuels |
| 3 | O2 + HM -> HO2M | both vent fuels |
| 5 | O2 + H2S -> H2O2S | both vent fuels |
| 6 | H2S + CH3M | one vent fuel |
| 7 | -- | no living at all |
| 8 | H2S + OM2 | one vent fuel |

`vents.fuel_species` is 2, so only the first two of the ranked list (HM, H2)
are actually injected. Raising it to 5 puts O2 and H2S in the water at a known
rate, and on seeds 2, 3 and 5 that makes the ancestor's whole diet renewable --
a chemosynthetic vent community, which is what the vents were put in the world
for.

**That was tried, and on its own it is not enough.** Seeds 2 and 5 at
`fuel_species = 5`, everything else `gate.toml`:

* Seed 5's larder does what a renewed one should for a while -- it *rises*,
  2.6e12 at t = 250 to 7.5e12 at t = 1000, against seed 1's monotonic decline.
  Then it falls to 6e10 by t = 1500 and stays there, with only sixteen cells in
  the pond, which is far too few to have eaten it. Something in the chemistry
  is consuming the vent fuel faster than the vents deliver it.
* Seed 2 collapses outright: 1.5e8 by t = 750 and 8e4 by t = 1250 on 69 cells.
  It peaks at 69, draws its food down 13.3x, and is extinct by t = 1951.
* Neither population divides much. The lifecycle numbers are tuned to seed 1's
  reaction and its enthalpy, and on a different metabolism they are simply the
  wrong size -- seed 5's cohort of sixteen never divides once.

So the question is quantitative, not structural: `fuel_species` decides *what*
is renewed and `fuel_rate` decides *how fast*, and 4e9 particles per second per
vent is evidently not fast enough against what the network does with it. The
measurement that settles it is the one asked for above -- the substrate's
resupply rate, in a pond with the population in it -- and it should be taken
before any more seeds are tried.

---

## L3: the genome

The population had one heritable number and every cell in the pond catalysed
the same reaction. `Population::metabolic_reaction` was a single
`Option<ReactionId>`, chosen once at introduction, shared by every cell that
would ever live. That is the mechanism underneath "the population is mining,
not grazing": the cells were not failing to regulate and they were not badly
tuned, they could only ever eat one thing, and on seed 1 that thing is made
from a carbon pool with one way in and no way out.

A genome does not fix that by being a better dial. It removes the "one thing".

### What is in it

`crates/cell/src/genome.rs`, and it is `PLAN.md`'s design rather than a
convenient subset of it. A **linear byte string**, variable length, scanned for
a two-byte promoter marker; junk between genes is neutral and is not compacted
away. Each gene decodes to a promoter motif, a list of `(motif, weight)`
regulatory binding sites, a protein class, an eight-component key, class
parameters and a stability. Every match in the system -- enzyme to reaction,
transporter to compound, regulator to promoter -- is the same Gaussian on key
distance, which is what makes the fitness landscape graded rather than a field
of cliffs.

Seven of the eight protein classes act on the world as it stands:

| class | what it does | where it lands |
| --- | --- | --- |
| `Enzyme` | catalyses reactions near its key, in the exergonic direction | `metabolize` |
| `Transporter` | raises membrane permeability for compounds near its key | `exchange` |
| `Structural` | buys famine tolerance, and is charged for it | `maintain` |
| `Regulator` | binds promoters, so genes regulate genes | `transcribe` |
| `Receptor` | senses one channel and emits it as a signal | `neural::sense_and_think` |
| `Neural` | sums its dendrites, squashes, remembers | `neural::sense_and_think` |
| `Effector` | shuts down, divides, gates a transporter, swims | `neural::drive` |

`Adhesion` decodes, costs upkeep and does nothing. It is not a placeholder to
be replaced -- its class code is part of the genome format, and reserving it
means an L5 genome stays readable by this decoder. Until then it is what a
nonfunctional protein is in a real cell: a bill with nothing on the other side,
and therefore selected against.

**A gene's sites mean one of two things, and its class byte decides which.** On
a gene whose protein does chemistry they are promoter binding sites and what
binds there is a transcription factor. On a `Neural` or `Effector` gene they
are dendrites and what arrives there is a signal. One structure, two networks,
and a mutation that flips a class byte moves a gene's inputs from one to the
other. The cost is that a neuron's own expression cannot be transcriptionally
regulated -- it runs at its basal level -- and that is the price of not
extending the byte format, which would have made every genome written before it
undecodable.

Mutation operators are the plan's table: point substitution, small indel, gene
duplication, gene deletion, segment inversion, whole-genome duplication, all
config-tunable and all drawn from `Counter` so a division is a pure function of
`(tick, parent)`. Horizontal transfer is not there; it needs genome fragments
to exist as field entities, which is a change to the mass audit and not a
change to this module.

### Two things that are deliberately not what they look like

**The old cell layer is not gone and is not a legacy path.** `cells.genome`
is a switch, default off, and with it off the arithmetic is *identical* --
checked, not asserted: a 20000-tick `--csv` of `gate.toml` is byte-for-byte the
same before and after this work, and was checked again the same way after L6
went in (every shared column identical; the two new ones are appended, so
compare with `cut -d, -f1-29`). Everything this project believes about its
population came out of A/B runs against a control, and a genome that quietly
replaced the protocell everywhere would have destroyed the control in the same
commit that needed it most. Both cells produce an `Expression` and nothing
downstream can tell them apart.

**Protein is an energetic burden, not a material one.** Synthesis is not
modelled as consuming amino acids, because there are no amino acids; what a
proteome costs is upkeep, charged through the existing `trait_cost` split
against `PROTEOME_REFERENCE`. So mass conservation is untouched and a genome
world audits exactly as a pre-genome one does. It also means junk *DNA* is
free while junk *protein* is not, which is the selection pressure that keeps
genomes from bloating into a free capability store.

### The bug worth reading, because it passed every test

The first version mapped key bytes onto a fixed `[-2, 2]`, reasoning that
`structural_key` is built from tanh-squashed ratios, polarities and small
counts. Seven of its eight components are inside that. The eighth is formation
enthalpy per atom over 1e-19 J and runs from **-9.04 to -1.49**.

So every ancestor's enzyme key was clamped to -2, sat seven units from the
reaction it was written to catalyse, and matched nothing. Sixteen founders
arrived in a pond holding 1.15e13 particles of their own food and ate not one
of them -- and the larder trace was flat at 1.15e13, which is exactly what the
lifeless control looks like. Every unit test passed the whole time, because
nothing had ever checked that a written key and a chemistry key end up in the
same space.

Keys are normalised against the chemistry now (`genome::KeyScale`), which also
stops affinity being dominated by whichever component happens to have the
widest units and drops dimensions the chemistry never varies -- ring count is
zero for every compound on this seed. The test that would have caught it is
`the_ancestor_catalyses_the_reaction_it_was_written_for`, and it asserts the
thing the assumption was hiding: the ancestor's best match is its own reaction,
at affinity above 0.98.

This is the fourth time in this project that something was chosen by reasoning
about a representation instead of measuring it. `choose_metabolism` has three
entries in that column already.

### The second thing that had to be a nudge

Mutation's point operator redrew the byte uniformly at first. That is wrong for
the same reason the key range was, and it is the more consequential of the two
because it would not have shown up as a broken run -- it would have shown up as
evolution simply not going anywhere, which is much harder to attribute.

`PLAN.md` is explicit about why the representation is bytes and keys at all:

> Graded mutation response. A single byte change nudges the key slightly, which
> nudges affinity slightly. Fitness landscapes become traversable instead of a
> field of cliffs. **This is the difference between evolution working and
> evolution not working.**

A uniform redraw moves one of eight key components to a uniformly random value.
Measured against `enzyme_sigma`, a single redraw is enough to take a protein
from full activity on its reaction to none, so a drifting lineage leaves its
own metabolism without arriving at another and there is no slope to climb.

Substitutions are now a step: four fifths move the byte by one to eight,
reflecting off the ends rather than clamping, and one fifth is still a full
redraw. The tail matters as much as the body -- a redraw is what changes a
protein's class, breaks a promoter, or makes one out of junk, and a radical
substitution at a key residue destroys a real protein too. What changed is that
it is now the tail. `one_substitution_usually_moves_affinity_a_little` measures
the distribution: the median substitution costs less than 0.35 of a unit of
affinity, the gentlest tenth less than 0.05, and the worst twentieth more than
0.9.

Point mutations can also land on a gene's two-byte promoter marker, which
deletes the gene outright. That is not a defect: a promoter mutation is a
knockout, and at the ancestor's 309 bytes it happens in about one division in
eighty.

### Recognition width was measured, not chosen

`hadean chem --keys` reports the nearest-neighbour spread of reaction and
compound keys in the normalised space the genome matches in, and what one
protein reaches at each width. On `gate.toml`:

```
  nearest neighbour   min 0.008  p25 0.062  median 0.102  p75 0.134  max 0.333
  sigma 0.05          an enzyme reaches    2.0 reactions
  sigma 0.08          an enzyme reaches    4.6 reactions   <- configured
  sigma 0.12          an enzyme reaches   10.3 reactions
  sigma  0.2          an enzyme reaches   39.0 reactions
```

`enzyme_sigma = 0.08` puts about 4.6 of the 78 thermal reactions inside one
enzyme's reach: its own, plus a few weak neighbours for a duplicate to drift
onto. The first guess, before this existed, was 0.35 -- which reaches 72 of 78,
and is not a metabolism.

Reach is not the same as what a lineage can *become*, and confusing the two is
how to get this wrong in the generous direction. A narrow width still leaves
the whole network reachable, because mutation moves the key itself. What the
width decides is how much one protein does at once.

### What the genome is being asked to find

The route out of seed 1's carbon dead end **exists and is downhill**:

```
    O2 + CH2OS   -> CH2O3S           k(20C) = 8e-15   -- no route at all
    H2 + CH2O3S  -> H2O + CH2O2S     downhill, and this remakes the food
```

The first step is unused because its barrier is too high. Lowering a barrier is
exactly and only what an enzyme does. A lineage that evolves onto that reaction
is a decomposer, and a pond with a decomposer in it recycles its carbon instead
of retiring it -- which is how real ecosystems avoid running their own larder
down, and it is not something any dial in `CellConfig` could ever have reached.

That is the mechanism. It is not a prediction that the run finds it.

### What the first run actually did

`evolve.toml` against its own control -- the same file with every mutation rate
at zero, which is a clone line carrying a genome and isolates what the genome's
machinery costs from what its mutation buys. 250000 ticks, four days of world
time, alongside the pre-genome `gate.toml` figures for scale:

| | pre-genome | genome, no mutation | genome, mutating |
| --- | --- | --- | --- |
| peak | 140 | 144 | 141 |
| births | 140 | 144 | 141 |
| **births after the peak** | **0** | **0** | **0** |
| clone lines | -- | 1 throughout | 4 -> 49 -> 0 |
| reactions eaten | 1 | 1 | 1 |
| food drawn down | 10.6x | 22.4x | 19.1x |
| finish | extinct | extinct | extinct |

The machinery works and it changed nothing. Forty-nine lineages against the
control's one is the mutation operators doing exactly what they are for; the
population is otherwise indistinguishable from a pond of clones.

The CSV says why, and it is not subtle. Every column freezes at t = 480 s and
does not move again for eight hundred seconds: population 140, lineages 49,
mean uptake 2.428, spread 0.242, genes 6.09. The last birth in the run is at
**t = 839 s of 2500**. A hundred and forty-one births from sixteen founders is
one generation and a bit.

**You cannot evolve anything in one generation.** No births means no new
genomes, no new genomes means no selection, and the genome spends the rest of
the run as an expensive way to store a constant. Every negative in the table
above follows from that one fact, and none of it is evidence about whether a
lineage could have found a different living -- the population never got to
look.

### The pre-run check was lying, and by a factor of five

`hadean ecology` prints the turnover budget before it starts, precisely so a
run like that can be declined rather than waited for. It printed:

```
turnover  division_reserve 1.00e-8 J against a budget of 6.00e-9 J
          -- only the far tail of the population can fund a daughter, if anything can
```

Short by a factor of 1.7, which reads like a run worth trying. It was short by
a factor of **8.4**.

[`CellConfig::turnover_budget`] takes `trait_spread` as the population's
standing variation `d`. For a pre-genome population that is right and is not an
approximation: `trait_spread` *is* the kick every daughter gets. For a genome
population it is a guess about a mechanism that is switched off. The measured
variation in this run was a mean uptake of 2.428 with a spread of 0.242 --
`d = 0.0997`, a fifth of the 0.5 the check assumed -- so the real budget was
1.2e-9 J against a `division_reserve` of 1e-8.

The genome's standing variation is *narrower* than the traits it replaced, and
that is not a defect in either: `Traits::founder` draws a log-normal with sigma
0.5 directly onto the trait, while a genome has to arrive at the same variation
through substitutions that nudge a key by a few 255ths at a time. It is the
difference between setting a number and evolving one, and it means a genome
world needs its lifecycle numbers re-measured rather than inherited -- which
was already item 6 on the list below, and is now a measurement rather than a
worry.

`ecology` now reports both: the configured budget before the run, flagged as an
assumption when the genome is on, and the budget at the population's measured
spread *at its peak* against the curve at the end. The peak, not the finish --
a run that ends extinct ends with no variation at all, and a budget computed
from that only says that everything is dead.

### The engine of complexity fired 0.86 times

There is a catch-22 underneath all of this, and it is worth stating separately
because no amount of tuning removes it.

In a pond with one food source, an enzyme that drifts off that food source is
lethal. A lineage exploring the reaction network dies before it arrives
anywhere, so selection actively suppresses the exploration that finding a
second living requires. This is not a flaw in the implementation; it is the
reason real evolution does not work that way either.

The biological escape is **gene duplication**: keep the original enzyme doing
its job, and let the copy drift. `PLAN.md` calls it "the engine of complexity"
and it is the one operator in the table with a bold entry. It is implemented
here and it works -- `duplication_grows_a_genome_and_its_gene_count` checks
that a duplicate decodes as a second gene with the same key.

At the rates in `evolve.toml` it is also, in a run like the first one, a
mechanism that never fires:

```
    births x genes x duplication_rate  =  141 x 6.09 x 1e-3  =  0.86
```

**Under one duplication event in four days of world time.** The engine of
complexity was expected to turn over less than once. Everything the genome
could have discovered was gated behind an operator that, at this population's
birth rate, statistically did not happen.

That is the same finding as "one generation" seen from the other end. The
obvious move is to buy more divisions by lowering `division_reserve` until it
satisfies the turnover budget, and that was written here as the fix before it
was tried.

### It was tried, and buying births does not buy generations

`division_reserve` at 4e-10 -- comfortably under the 1.2e-9 the measured spread
allows -- on both `evolve.toml` and its clone control:

| | genome, no mutation | genome, mutating |
| --- | --- | --- |
| peak | 2089 at t = 227 s | 2088 at t = 237 s |
| births | 2089 | 2088 |
| **births after the peak** | **0** | **0** |
| clone lines | 1 throughout | 617 at the peak |
| uptake spread at the end | 0.000 | 0.357 |
| finish | extinct | extinct |

Fifteen times the population and fifteen times the births, and **still not one
birth after the peak**. The last division in the mutating run is at t = 237 s
of 2500. What lowering `division_reserve` bought was a bigger, faster boom, not
a second generation.

The CSV says exactly what happened, and it is worth reading as a sequence:

```
t=176   pop=153    dormant=0     food=1.116e13
t=201   pop=1801   dormant=1     food=5.270e12
t=326   pop=2088   dormant=2088  food=3.288e10
t=676   pop=2075   dormant=2075  food=2.059e11     <- food recovered 6x
t=1176  pop=1888   dormant=1888  food=2.814e03
```

The population overshoots to 2088 in ninety seconds, strips the pond by a
factor of 340, and goes **100% dormant at t = 326 s -- and never wakes**. Not
one cell wakes even at t = 676, when the night has put six times as much food
back in the water. Two thousand cells sharing a recovered larder is still less
per cell than the wake-up bar, so they sit shut down until starvation damage
kills them one at a time over the following twenty minutes.

Dormant cells do not divide. So the plateau is not a population choosing not to
breed, it is a population that has switched itself off, and `division_reserve`
does not reach that.

**This was predicted here before it was run**, in the `gate.toml` note on
`division_reserve`: "At 2e-10 the cells double every fifteen seconds, strip a
day's food by mid-afternoon, and meet the night with nothing. At 3e-9 they
still overshoot and die on the fourth night." 4e-10 is squarely in the regime
that note describes, and the run did precisely what it says.

### Where that leaves it

Both ends of the dial give the same answer for opposite reasons:

* At `division_reserve = 1e-8`, no cell at the margin can afford a daughter, so
  the plateau has no births in it. One generation.
* At `4e-10`, the response is fast enough to overshoot into total dormancy, so
  the plateau has no *waking cells* in it. One generation.

There may be a window between them and it is worth one sweep. But the reason
both ends collapse to one generation is the same reason, and it is not in
`CellConfig`: **a population with a finite larder has no steady state to have
generations in.** Births balancing deaths is what a carrying capacity *is*, and
this pond does not have one for this metabolism -- which is the finding from
"The population is mining, not grazing", arrived at from a third direction.

So the genome is not the thing that is blocked; it is blocked behind the thing
that was already blocking everything. It cannot be evaluated on seed 1 as
configured, because a genome is a mechanism for adapting across generations and
this world affords one. The next measurement is still the renewable-living one,
and the genome is what will make use of it when it exists.

What the run does establish, and it is not nothing: the machinery is sound
under real load. 617 lineages against a control's 1, standing variation of
0.357 against a control's exactly 0.000, gene counts moving, energy audit flat
at -4.7e-11 relative across two thousand cells' worth of transcription,
transport and catalysis. When there is a living to adapt to, the thing that
would do the adapting works.

---

## L6: the decision layer

### The reading that prompted it

Every genome run so far ended the same way, and the CSV said it in one line:

```
t=326   pop=2088   dormant=2088   food=3.288e10
t=676   pop=2075   dormant=2075   food=2.059e11     <- food recovered 6x
t=1176  pop=1888   dormant=1888   food=2.814e03
```

A hundred per cent dormant, and not one cell wakes even when the night has put
six times as much food back in the water. That was read here as an overshoot
problem -- two thousand cells sharing a recovered larder is still less per cell
than the wake-up bar -- and it is, but that is the symptom rather than the
disease.

The disease is that **there was no decision in the pond to be wrong**. When to
shut down and when to wake were two numbers in `CellConfig`, read by every cell
in the world. A population like that cannot disagree with itself. It shuts down
at the same instant, it waits for the same bar, and when the bar is
unreachable it is unreachable for all of them at once -- so there is no
survivor to select, no variant to be right, and nothing for the genome layer to
work on. Sixteen founders one division from the ancestor made it worse: four
distinct lineages out of sixteen, and a standing variation five times *narrower*
than the single scalar trait the genome had replaced.

That is the same shape as the Phase 2 crash ("What the crash actually was"),
which was a shared lifespan, and as the freeze that `Traits` was introduced to
break, which was a shared uptake. Third time: **every time this project has
found a population dying as one, the cause has been a number that should have
been per-cell and was per-config.** Dormancy was the last big one.

### What replaced it

`crates/cell/src/neural.rs`. Three of the four remaining protein classes now
have an interpreter, and there is no path from the world to a cell's behaviour
that does not go through it:

```
    receptor ---- signal ----> neuron ---- signal ----> effector
    (senses)                  (decides)                 (acts)
```

* A **receptor** reads one [`Channel`] -- food outside, the same compounds
  inside, its own reserve, its damage, light, heat, crowding, age -- and emits
  what it finds at its own key. `params[1]` picks the channel; for the two
  chemical channels the key also picks the compounds, by the same affinity a
  transporter uses, so a lineage with a transporter for something is most of
  the way to a sensor for it.
* A **neuron** sums its dendrites, adds a bias, squashes, and relaxes towards
  the result at its own `leak` rate. It reads the activations as they stood at
  the end of the previous tick, so the network is recurrent, may contain
  cycles, and needs no ordering or acyclicity check.
* An **effector** does the same sum and drives one [`Action`]: `Quiesce`,
  `Divide`, `Ingest` (open or close the transporters its key matches), `Move`
  (swim along an axis, at a cost the audit sees).

Wiring is `affinity` on key distance -- the same function that matches an
enzyme to a reaction, because there is only one way for two things to be alike
in this simulation. Which wires exist is resolved once per genome in `bind`;
what travels down them is per-cell state, which is why two cells carrying
identical bytes in different corners of the pond behave differently.

### Recurrence is not a shortcut, it is the hysteresis

`dormancy_exit` existed for one reason: a cell on the margin would otherwise
flicker between shut down and working every tick and average back into paying
the full bill. That is a memory problem, and the config dial was a shared,
population-wide answer to it.

A neuron has memory of its own. `Gene::leak` is four bits of `params[3]`,
logarithmic over 0.1..20 per second, so a time constant anywhere from ten
seconds to fifty milliseconds. A slow neuron holds an opinion through a
shortage; a fast one tracks the water. **Sleeping deeply but briefly, lightly
but long, and everything in between are now things a lineage is rather than
things a config file is** -- which is the whole of what was asked for, and it
falls out of one byte rather than out of a new mechanism.

### Sleep has to cost something, and what it costs is the machinery

The first version of `transcribe` scaled every gene's transcription level by
`activity` -- the fraction of the working bill the cell is currently paying.
That is the mechanism that stops quiescence being a free lunch. Without it a
shut-down cell pays five per cent of its upkeep and goes on taking up and
catalysing at full rate, which is strictly better than staying awake, so the
only stable strategy is permanent sleep. Which is very nearly what the run
above shows.

With it, a cell that shuts down stops synthesising, its proteome decays at each
protein's own `stability`, and within a minute or two it has lost the
transporters and enzymes it was living on. Cheaper *and* poorer, with a real
spin-up time on the way back. That is a spore.

**The nervous system is exempt from the throttle, and it took a failing test to
see why.** Scaling everything closes the arithmetic on itself: shutting down
decays the receptor that reads the reserve and the effector that acts on it,
which lowers the depth, which restores them. The ancestor settled at a
quiescence of 0.471 and could not go deeper however hungry it got -- a cell
physically unable to commit to sleeping, because the organ it sleeps with was
the first thing it switched off. Nothing could have woken it either. Receptors,
neurons and effectors therefore transcribe at their full level whatever the
depth, which is also what a real spore does, and they stay on the upkeep bill
through `machinery()` -- so a lineage that evolves a large brain pays for it in
every famine it sits through.

### The ancestor now has a nervous system, and it is deliberately dull

`genome::ancestor` writes four more genes: a receptor on `Channel::Energy`
emitting at `hunger`; a neuron listening at `hunger` with weight -4 and bias
+0.6, emitting at `sleep`; a `Quiesce` effector listening at `sleep` with
weight +4; and a `Divide` effector listening at the same address with weight -4
and bias +1, so the signal that shuts the cell down also stops it spending a
reserve on a daughter.

That circuit is *approximately the rule it replaces* -- it shuts down around
fifty seconds of banked upkeep, which is where `dormancy_exit = 60` put the old
wake bar. On purpose. The claim being tested is not that a hand-written network
beats a hand-written threshold; it is that **a network is made of parts that
mutate**. A weight, a bias, a rate of forgetting, a channel, an action, and
which effector hears what. Sixteen founders of it are sixteen different
opinions about when to sleep and how deeply.

`PROTEOME_REFERENCE` moved from 4.0 to 7.0 with it. Four more proteins to keep
is a third again on the upkeep bill, and leaving it would have quietly charged
every genome cell for the privilege of being able to decide anything.

### Founders are relatives now, not copies

`founder_divergence` (default 20) runs each founder through that many rounds of
the *same* mutation operators every division uses. Nothing founder-specific:
the variation the cohort arrives with is drawn from exactly the distribution
its descendants will go on exploring, and it places the ancestor some
generations in the past rather than at t = 0 -- which is the more honest
picture anyway. Life does not arrive at a pond having just been invented.

The measured effect at t = 200 s on `evolve.toml`, sixteen founders:

| | before | after |
| --- | --- | --- |
| distinct lineages | 4 | **16** |
| genes per cell | 6.1 | 9.9, of which 4.1 signal |
| dormant at t = 200 s | all or none | **4 of 16, at a mean depth of 0.22** |

The last row is the one that matters and it is the point of the whole change: a
quarter of the pond has decided to shut down and three quarters have not, in
the same water, at the same moment. That never happened before, because it
could not.

### What the run actually did

`evolve.toml`, 250000 ticks, everything as before except the four new genes and
`founder_divergence = 20`. Against the same file's previous run, which is the
table under "What the first run actually did":

| | before L6 | with L6 |
| --- | --- | --- |
| peak | 141 | **157** at t = 824 s |
| births | 141 | 161 |
| deaths | -- | 154 |
| **births after the peak** | **0** | **0** |
| clone lines at the peak | 49 | **75** |
| signal genes per cell | -- | 4.1 rising to **5.14** |
| reactions eaten | 1 | 1 |
| finish | extinct | 7 living, extinct trajectory |
| energy audit | -4.7e-11 | **-3.9e-12** relative |

And the reading the whole change was for, at t = 1539 s:

```
    pop=156   dormant=78   mean_quiescence=0.453
```

**Exactly half the pond shut down, in the same water, at the same moment**, at
a mean depth of 0.45. Before this, every reading of that column was 0% or
100%. The pond can now disagree with itself about whether to sleep, which is
the entire mechanism, and the disagreement is heritable.

Two other things in that table are worth reading. The nervous system is *not*
being shed: signal genes per cell rise from 4.1 to 5.14 as the population is
culled from 157 to 7, so the cells that survived longest were on average the
ones carrying more of it -- upkeep with no metabolic return, kept anyway. And
the energy audit is an order of magnitude *tighter* than the pre-L6 run at
-3.9e-12 relative, across a layer that added two new ways for a cell to spend
(a graded upkeep and a motility bill) and a third of a joule's worth of
protein. Both new spends go through `heat.deposit` and the reserve, like every
other.

### What it did not do, and why that was predictable

**Still zero births after the peak.** The last division in the run is at
t = 824 s of 2500, against 839 s before it. Dormancy was not what was stopping
them, and the run says so in its own summary:

```
turnover  3.11e-9 J budget at the peak's spread of 0.259
          -- nothing at the margin could fund a daughter
```

`division_reserve` is 1e-8 J and the population's measured spread will fund
3.11e-9. Short by a factor of three, which is item 0 on the outstanding list
and was item 0 before any of this. A cell that has decided not to sleep still
cannot afford a daughter it has no surplus for.

So: the dormancy fault is fixed as a *mechanism* and the pond still fails the
Phase 2 gate, for the reason it was already failing it. Those are two findings
and they should not be reported as one. What has changed is that the failure is
now a clean boom and bust -- 157 to 7, 154 deaths -- rather than a population
freezing at 140 and standing still, and that a population which is dying now
does so at a range of depths instead of all at once.

### The 4e-10 regime, re-run

The run above is at `division_reserve = 1e-8`. The other end of the dial is the
one this change was aimed at, because it is where the pond went 100% dormant
and stayed there. Same file with `division_reserve = 4.0e-10`, which is the
configuration the table in "It was tried, and buying births does not buy
generations" was measured on:

| | before L6 | with L6 |
| --- | --- | --- |
| peak | 2088 at t = 237 s | 2212 at t = 260 s |
| births | 2088 | 2215 |
| **births after the peak** | **0** | **3** |
| clone lines at the peak | 617 | **867** |
| uptake spread at the peak | 0.357 | **0.542** |
| turnover budget | short | **6.50e-9 J: a cell above the margin could fund a daughter** |
| dormant at t = 250 s | -- | 1414 of 2207 (64%) at depth **0.51** |
| dormant at t = 326 s | **2088 of 2088 (100%)** | -- |
| dormant at t = 676 s | **2075 of 2075 (100%)**, food recovered 6x | -- |
| dormant at t = 750 s | -- | 1857 of 2095 (89%): **238 awake**, food recovered 100x |
| finish | extinct | 2 living |

Two things in there are new and one is not.

**The sentence that was true of every earlier run at this setting -- "goes 100%
dormant and never wakes" -- is not true of this one.** At t = 750, with the
night having put a hundred times as much food back in the water, two hundred
and thirty eight cells are working through a shortage that has shut the other
1857 down. Half the pond at t = 250 was awake at a mean depth of 0.51.

**Three births after the peak**, which is a small number and is the first
non-zero value that column has ever had in this project. The mechanism behind
it is visible in the row above it: the population's standing variation at the
peak went from 0.357 to 0.542, which is enough to flip `ecology`'s own pre-run
check from "nothing at the margin could fund a daughter" to "a cell above the
margin could". More variation is exactly what `founder_divergence` and a
heritable dormancy strategy were expected to buy, and the budget line is the
project's own arithmetic agreeing that they bought it.

**It still ends extinct**, at 2 cells from a peak of 2212, having eaten its
larder down 15.8x with 0.00x back. That is not dormancy and it is not
variation. It is item 1: the pond has no renewable living, so a population that
overshoots has nothing to come back to. Three births is a mechanism working at
the very bottom of its range, not a carrying capacity.

### What is still hardcoded, and why it is not behaviour

Two lists, and neither is a decision: `Channel`, the things there are to sense,
and `Action`, the things there are to do. A cell has a membrane so there is
something to taste through; it can shut down, divide, open a transporter and
swim, so there are four things an effector can be wired to. Which channel a
receptor watches and which action an effector drives are `params[1]` of the
gene -- genetic, mutable, selected on. The lists are the cell's *physiology*,
and a genome that could invent an organ the cell does not have would not be a
genome.

The variant order of both is part of the genome format for the same reason
`ProteinClass`'s discriminants are: a gene picks by quantising a byte, so
inserting a variant in the middle re-points every receptor in every stored
genome. **Append only.**

---

## The `division_reserve` sweep, and what it closed

Item 0 asked for the sweep between the two ends that had already been run, on
the grounds that one of them had stopped failing for the reason that used to
mask everything else. It is done, five points on `evolve.toml`, everything but
`division_reserve` identical:

| `division_reserve` | peak | births | after the peak | food eaten | finish |
| --- | --- | --- | --- | --- | --- |
| 4e-10 | 2212 | 2215 | 3 | 15.8x | 2 |
| 1e-9 | 979 | 979 | 0 | 16.9x | 4 |
| 2e-9 | 590 | 590 | 0 | 24.5x | 5 |
| 4e-9 | 330 | 330 | 0 | 9.6x | 2 |
| 1e-8 | 157 | 161 | 0 | 15.1x | 7 |

**`division_reserve` buys peak population and nothing else.** The peak scales
almost exactly inversely with it -- the product is 0.9e-6 to 1.6e-6 across four
and a half octaves -- and at every point `births` comes out equal to the peak
to within four. That is one generation, five times over. Births after the peak
are zero at four of the five points and three at the fifth, food is eaten down
by ten to twenty-five fold at every point, and every run finishes within a
handful of cells of extinction.

So the dial is not the variable, and the two ends were not two different
failures with a shared cause -- they were one failure seen at two
magnifications. The sweep was worth running because it is the last thing that
could have been true about the cell layer, and it is not true. Whatever is
wrong is upstream of `CellConfig`, which is what item 1 has said all along.

---

## What the pond can resupply, which is not what it holds

`choose_metabolism` had been wrong three times, each time by trusting something
that looked like food, and the third correction -- rank by measured standing
stock -- is the one every run in this project has been built on. It is also
wrong, and the section above is what it costs.

A standing stock is a larder. A living is a *rate*. Nothing here measured one.

### The instrument

`hadean supply` settles a lifeless pond, takes one compound out of it
entirely, and then keeps taking every particle the chemistry makes of it, for
as long as the probe runs. Held at zero, the network runs its production of
that compound as fast as it can, so what has to be removed each second to keep
it there is **the largest harvest the pond will ever support**. The standing
stock is taken out first and deliberately not counted: that is the larder, and
being fooled by the larder is the entire point.

Three things make it honest rather than decorative.

* **The removal is booked in the audit.** `World::harvest` records a negative
  injection in the same ledger a vent uses, so a probe cannot quietly
  manufacture the answer -- `a_probe_stays_inside_the_audit` runs one and
  checks the energy and mass drift afterwards.
* **Each probe is its own world**, restored from one shared settled snapshot,
  so `--jobs` changes the wall clock and nothing else.
* **It is cut into windows.** A pond that hands over as much in the last window
  as the first is producing the compound; one whose rate has collapsed between
  them was handing over a stock. That ratio is the `holding` column, and it is
  what separates the two readings that had been conflated.

### It agrees with the one rate that was never in doubt

`gate.toml` has `vents = 3` and `fuel_rate = 4.0e9`, so a vent fuel is
resupplied at exactly 1.2e10 particles per second and the config says so
outright. Run blind over the whole chemistry, the probe measures HM at
**1.206e10/s** and H2 at 1.214e10/s. That is the instrument calibrating itself
against a number it was not told, and it is a test:
`a_probe_recovers_the_flux_the_config_already_knows`.

### What it found on seed 1

```
  compound       standing      opening    sustained  holding
  HM              4.621e11   1.208e10/s   1.206e10/s    0.998  renewed
  H2             1.333e13   1.215e10/s   1.214e10/s    0.999  renewed
  HO2M            1.854e10    6.913e9/s    8.512e9/s    1.231  renewed
  HO3M            1.306e12    8.465e9/s    6.867e9/s    0.811  renewed
  CH2O2S          1.150e13   3.048e9/s    1.027e7/s    0.003  larder
  H2O             1.152e15   1.988e6/s    2.031e4/s    0.010  larder
  H3N             1.152e13   1.516e4/s    0.000e0/s    0.000  dead end
```

`CH2O2S` is the ancestor's food. It is the second-largest stock in the pond and
the pond rebuilds **ten million particles a second** of it -- three orders
below the compounds either side of it in that table, and a factor of three
hundred below its own opening rate, because the opening rate is the last of the
CO2 going through. The mining diagnosis was argued from the food column and a
lifeless control; this measures it directly.

And the pond has a renewable living in it that nothing has ever eaten:

```
    sustains   population  per newborn  reaction
   4.554e-9 W      455 cells   HO2M + HM <=> 2 HOM
  8.387e-13 W        0 cells   H2 + CH2O2S <=> H2O + CH2OS
```

The pond holds six hundred times more CH2O2S than HO2M and rebuilds eight
hundred times less of it. **Ranked by amount the ancestor gets the first;
ranked by rate it should have had the second, which is worth five thousand
times the sustainable power.** That is the mechanism under "the population is
mining, not grazing", and it is not a badly tuned cell -- it is a metabolism
chosen off the wrong column.

### What changed in the code

* `choose_metabolism` takes `supply: Option<&[f64]>`. With it the ranking is a
  **power in watts**, directly comparable against `maintenance_power`, so the
  quotient is a carrying capacity in cells -- the number every run before this
  one lacked. Without it the standing-amount rule is exactly as it was, because
  it is the pre-genome control's arithmetic and has to stay bit-identical.
* `cells.metabolism` names a measured living by its substrates,
  `"HO2M + HM"`. Measuring resupply costs one settled lifeless world per
  candidate substrate, which a run cannot afford at the moment its ancestors
  arrive, so the measurement is taken once with the instrument and written into
  the config -- exactly as `enzyme_sigma` was measured with `chem --keys` and
  written into `evolve.toml`. It is checked at world construction, so a bad
  name costs nothing rather than two and a half minutes of world time.
* `configs/renew.toml` is `gate.toml` with that one line changed.

---

## The living the pond makes and the cell cannot reach

`renew.toml` on the carried-over lifecycle numbers: sixteen ancestors, dormant
on the tick they arrived, dead of starvation by t = 600. Item 2 has been
predicting that since before there was a second food chain -- the numbers were
measured against a reaction releasing 82 kJ/mol on a substrate standing at
1.15e13, and this one releases about 537 kJ/mol on a substrate standing at
1.85e10.

So it was swept. `membrane_scale` over thirty-fold, 1e4 to 3e5. Then
`metabolic_rate` over twenty-fold, 500 to 10000. Eight runs, all sixteen cells
shut down in all eight, and:

* the four `metabolic_rate` runs are **byte-identical** in every logged column,
  a twenty-fold change in the dial making no difference of any size at all;
* the four `membrane_scale` runs differ only in the *fourth* digit of the food
  column -- 1.379e10, 1.378e10, 1.377e10, 1.376e10 at t = 600 -- and not at all
  in anything about the cells. Thirty times the membrane moved the pond by a
  tenth of a percent.

Two dials that inert are not a tuning problem. They mean the quantity being
tuned is not the one that binds, and the arithmetic says which one is:

```
  CH2O2S (the larder diet)  1.997e9/voxel -> 3.425e7 in a cell -> 5.590e-11 W  (5.590 x upkeep)
  HO2M   (the renewed diet) 3.212e6/voxel -> 5.511e4 in a cell -> 5.899e-13 W  (0.059 x upkeep)
```

**A passive membrane equilibrates; it does not concentrate.** At the steady
state of `exchange`, `inside / cell_volume == outside / voxel_volume`, so a
cell holds its own volume's share of whatever the water around it holds and *no
value of `membrane_scale` changes that number*. All a bigger membrane buys is
arriving at the same equilibrium sooner, which is why thirty times more of it
moved nothing. `metabolic_rate` then saturates against what the membrane
delivers, which is why twenty times more of that moved nothing either.

Six hundred times less substrate in the water is six hundred times less inside
the cell, and six and a half times more energy per turnover does not cover it.
Ninety-five times short, on a diet that is five thousand times better *per unit
of pond*.

That is a third distinct sense of "food" and the pond-wide rate does not imply
it: **`supply` measures what the world produces, and a cell is a point consumer
in a dilute field.** Both bounds are real and a living has to clear both.
`hadean_cell::subsistence` is the second one -- what one newborn earns as a
multiple of its upkeep -- and it now prints from `ecology` the moment the
metabolism is settled, and as the `per newborn` column of `supply`. It answers
in a microsecond what those eight runs took half an hour to say.

Confirmed against the world rather than asserted: holding everything else, at
`maintenance_power` 1e-11 all sixteen cells are dormant, at 1e-12 the pond is
mixed and flickering between the two, and at 1e-13 and below nothing is dormant
at all. The break-even sits just under 1e-12 W against a predicted 5.9e-13.

### Re-tuning against the diet, which is what item 2 asked for

Scaling the whole cell energy budget down by the two orders the measurement
calls for -- `maintenance_power` 1e-11 -> 1e-13, `division_reserve` 1e-8 ->
3e-11, against a `turnover_budget` of 6e-11 -- and the ancestors divide. Two
runs of 250000 ticks, both on `renew.toml`'s pond, differing only in
`division_reserve`:

| | `renew.toml` (3e-11) | B (1e-10) | `evolve.toml`, best of five |
| --- | --- | --- | --- |
| peak | **17333** at tick 187000 | 10196 at tick 196600 | 2212 at tick 26000 |
| births | 17756 | 10241 | 2215 |
| **after the peak** | **193** | 8 | 3 |
| deaths | 8903 | 5481 | 2213 |
| finish | 8853 | 4760 | 2 |
| food eaten | 14.9x | 8.5x | 15.8x |
| energy drift | 7.6e-12 | 1.3e-11 | -3.6e-11 |

**One hundred and ninety-three births after the peak.** Every run in this
project's history has recorded zero there, except the one that recorded three.
That column is what a population turning over looks like, and it is the leg of
the Phase 2 gate that has been missing since the beginning.

Four other things in that table are worth reading.

* **The peak is seven times higher and arrives seven times later.** On the
  larder the population sprints through the food and peaks at tick 26000; here
  it grows into a supply and peaks at 187000. A renewable pond has a
  fundamentally longer timescale, and the gate's 250000-tick run length was
  calibrated against the other one.
* **Neither run got to show a recovery, because both were still crashing when
  the clock ran out.** The trough is reported at tick 250000 in both -- that is
  the last tick, not a turning point. This is a run-length result and not an
  ecological one, and it is the single most misreadable thing in this section.
* **The food comes back.** Under the larder every night's rebuild reached a
  tenth of the last one, four orders of magnitude down and never up. Here the
  food column falls to 2.4e3 at tick 240000 and is at 5.2e6 ten thousand ticks
  later. That is the difference between a stock and a supply, showing up in the
  one column that had been flat about it for the whole project.
* **The uptake trait moved.** Mean 5.8 with a spread of 2.2, against 0.139 and
  0.029 on the larder. The population is under real selection pressure on a
  resource it can exhaust locally, which is what was wanted from the trait in
  the first place.

`renewB` doubles as the control showing the pre-run checks read the right
world: at `division_reserve` 1e-10 against the same 6e-11 budget, `ecology`
says *before* the run that only the far tail can fund a daughter, and the
population duly sits at exactly sixteen for four hundred seconds before it
starts to move.

**The gate is still not met.** What has changed is what it is now waiting on: a
longer horizon, rather than a mechanism that does not exist. An 800000-tick run
of `renew.toml` is the next thing, and it is running.

## The larder that held its rate, and the element that was never coming back

`hadean supply` exists because `choose_metabolism` had been wrong three times
by reading an amount where a rate was wanted. It was wrong a fourth time, in
its own column, and the shape of that error is worth more than the fix is.

The 250000-tick `renew.toml` runs ended with the population entirely dormant
and still falling. The last line of `renewA` says why:

```
food        1 HO2M (3.417e6 left) + 1 HM (2.428e12 left)
```

HM, the vent fuel, is untouched at 2.4e12. HO2M is gone. Between tick 220000
and 250000 the pond is **100% dormant** -- 16275 of 16277 cells asleep, nothing
eating -- and HO2M still falls, 4.2e4 to 5.4e3. Against the measured 8.512e9/s,
three hundred seconds of a sleeping pond should have rebuilt 2.5e12 particles.
It rebuilt none. `limiting_substrate` already adds `cells_holding`, so it is
not hiding inside membranes either.

### The chemistry says it in one line

```
vent fuel (most energetic first):  HM, H2, H2S, HO3M, O2, H2NM
vents.fuel_species = 2                    injected: HM, H2 only

  82  O2 + HM    <=> HO2M       the only route that makes HO2M
   8  O2 + CH2OS <=> CH2O3S     consumes O2
  83  O2 + HOM   <=> HO3M       consumes O2, and the band-7 photon drives it this way
```

`rank_vent_fuel` takes the six most energetic molecules of six atoms or fewer,
so that list is the whole list rather than a truncation. At `fuel_species = 2`
the vents inject HM and H2, `Audit::elements_in` therefore only ever
accumulates **H and M**, and HO2M is built from an element that does not enter
this pond. Not slowly -- at no rate at all. No vent fuel on this seed contains
carbon at any `fuel_species`, so the pond's entire carbon chemistry is a closed
pool permanently.

### Why `holding` endorsed it, which is the part to remember

`holding` is sustained-over-opening, and it was put in specifically to catch a
larder. It gave `CH2O2S` 0.003 and caught that one. It gave HO2M **1.231** --
not merely holding its rate but *accelerating* -- and that reading is what the
whole diet change was built on.

A stock drained at a constant speed holds its rate perfectly across a window,
because holding its rate is what a constant speed **is**. `holding` cannot
distinguish a supply from a large pool being tapped harder, and over a 600 s
window against a crash at 2000 s it never had a chance to. The column was not
insufficiently sensitive. It was measuring the wrong thing confidently.

### The provenance column

Matter crosses this world's boundary in exactly two places -- `vent_matter`
puts it in at `world.rs:324`, `harvest` takes it out at `world.rs:539` -- and
both book into `Audit::elements_in`. So the vents' contribution to a probe
window is recoverable exactly: what the ledger gained, plus what the probe took
back out. `Harvest` now carries the compound's `formula` and that per-window
inflow, and three readings come off it:

* **`funded(w)`** -- the largest harvest the vents could have *paid for*,
  particles/s. A probe exports matter, so a sustained harvest is bounded by the
  rate the world gains the atoms it is made of, and only a vent brings an atom
  in; a photon carries energy, not matter, and can rearrange an atom but cannot
  deliver one. The bound is deliberately **generous**: it credits one compound
  with every atom the vents delivered, as though nothing else in the pond
  wanted any, and ignores standing stock entirely. A compound that still reads
  zero under that accounting is not being resupplied.
* **`provenance()`** -- that over `sustained`, capped at one.
* **`starved()`** -- which element binds, so the verdict names it.

`verdict` asks provenance *before* it looks at the shape of the curve, and
`report_inflow` prints the world's matter budget before the table, because that
budget decides most of the table in advance.

On `renew.toml`, settle 150, probe 60:

```
inflow    the vents deliver H 3.600e10, M 1.200e10 -- atoms per second, off the audit's element ledger
          nothing brings C, O, N, S into this pond, so a harvest of anything containing one of them is stock

  compound       standing      opening    sustained       funded  holding
  HOM             8.246e8   1.573e10/s   1.397e10/s    0.000e0/s    0.888  stock - no O enters this pond
  H2             1.333e13   1.215e10/s   1.213e10/s   1.800e10/s    0.998  renewed
  HM             4.621e11   1.208e10/s   1.205e10/s   1.200e10/s    0.998  renewed
  HO2M           1.854e10    6.920e9/s    8.817e9/s    0.000e0/s    1.274  stock - no O enters this pond
  CH2OS          4.123e10    4.630e9/s    5.276e9/s    0.000e0/s    1.139  stock - no C enters this pond
  H2NM            2.254e7    1.824e5/s    1.570e5/s    0.000e0/s    0.861  stock - no N enters this pond
```

**Two of twenty-two candidates come back renewed, and they are the two the
vents inject.** The inflow line is checkable by hand -- `vents = 3` at
`fuel_rate = 4.0e9` gives H at 1.2e10 from HM plus 2.4e10 from H2, M at 1.2e10
-- and `funded` follows from it arithmetically: HM at 1.200e10, H2 at 3.6e10/2
= 1.800e10. The instrument now checks itself against the config's stated flux
in a second column as well as the first.

### The settle sweep, which confirmed it from the other side

The same probe on the same lifeless config at `--settle 2200` instead of 150:

| | settle 150 s | settle 2200 s | factor |
| --- | --- | --- | --- |
| HO2M standing | 1.854e10 | **1.037e4** | 1.8e6 |
| HO2M sustained | 8.817e9/s | **4.543e4/s** | **1.94e5** |
| O2 standing | 9.577e12 | 2.293e9 | 4.2e3 |
| O2 sustained | 3.285e3/s | **0.000e0/s** | dead end |
| `HO2M + HM` living | 4.718e-9 W, 47176 cells | **2.431e-14 W, 0 cells** | 1.94e5 |

**The pond retires its own oxygen in 2200 s with nothing living in it.** The
population was never needed to explain the crash; it only arrived in time to be
blamed for it. The sustained rate and the sustainable power fall by the same
1.94e5, from two independent columns.

Two further things in that run:

* **The reading was taken at the pond's most flattering instant, and not by
  accident.** `supply` defaults `--settle` to `seed_delay`, which is exactly
  when `choose_metabolism` picks the ancestors' diet. Both instruments read the
  same transient at the same moment and neither could see it was one.
* **At a late settle `holding` does not merely fail, it inverts.** `HO3M`
  reports `holding 113230009667.186 -> renewed` on a standing stock of 3778
  particles, because its opening window was 3.255e-7/s. `CH2O2S` reads
  `renewed` at 27.8, `HO2M` at 5.075. Half that table says "renewed" about
  compounds that are essentially gone.

### The bug the first live run exposed, in the new arithmetic

Three compounds came back with **negative** funded rates -- H2O at -1.595e-1/s
-- and `-0%` in the vent-fed column. A negative "largest harvest the vents
could pay for" is not a small number, it is a wrong one, and `O2` read
`part stock - O inflow pays 0% of it` off 4.4e-4/s of the same noise.

`elements_in` is differenced across a window at the magnitude of every atom of
that element that has ever crossed the boundary -- once the standing stock is
harvested, the pond's whole inventory, 1.152e15 for H2O's oxygen. An ulp there
is 0.125, each tick books one, and a window has 1200 ticks. The residue was
rounding. `vent_inflow` now applies a floor of `ticks * EPSILON * ledger
magnitude` and reports below it as the zero it is;
`ledger_rounding_is_not_reported_as_inflow` and
`the_floor_does_not_swallow_a_real_inflow` are the pair that pin it, the second
asserting a genuine 1.2e11 survives being measured against a 1e15 baseline.

### What is deliberately *not* done

The livings table gets a `vent-fed` column but **is not re-ranked on
`funded`**. Bounding a living by matter inflow would be the mirror image of the
error just caught: a *cell* exports nothing -- it turns its substrate into
products and leaves every atom in the pond -- so a living can be sustained by
matter that never enters at all, provided photochemistry drives the products
back. On this chemistry there is such a route,
`HOM --(band 7, +O2)--> HO3M --(+H2, rxn 30)--> HO2M`.

So `funded` bounds the **probe**, which exports matter, and not a population,
which does not. The table prints both and says which question each answers.
Ranking honestly on a number that answers the wrong question beats ranking
confidently on one bent to answer the right one.

### What this costs the runs that came before

Nothing in the cell layer is invalidated: `subsistence`, the lifecycle
re-tuning, the 193 births after the peak all stand as measurements of what
those cells did in that pond. What is invalidated is **the reason `renew.toml`
gives for its own existence**. Its header quotes HO2M at 8.512e9/s and
`holding 1.231` as the renewable living the pond had all along, and that is a
transient read at t = 150 in a pond whose oxygen chemistry is still on its way
down. The 17333-cell boom was a larger larder, not a supply.

### Threads are free, and the machine was being wasted

Small worlds do not want all 22 threads and the performance section already
said so, but three jobs at a default rayon pool each is worse than that: it is
22 threads three times over. `verify --ticks 500` at 1, 4, 8 and 22 threads
gives the same digest every time, with identical drift -- so
`RAYON_NUM_THREADS=8` per job, several jobs side by side, costs nothing and is
about 2.5x the total throughput.

(The digest itself was `84f89bb98d14f9d4` when that was measured and is
`79783137c9105341` now. Nothing about the physics moved: `WorldConfig`'s state
hash is taken over the serialised TOML, so adding `cells.mutator_range` changed
every digest in the project. A recorded digest is a check that two runs agree
with *each other*, not a constant -- and one quoted across a config change is
worse than none.)


## The sweep, and a settle you pay for once

Items 1b and 1c, and they are one piece of work because the first is
unaffordable without the second.

The finding above was assembled by hand: `supply` at `--settle 150`, then the
same command again at `--settle 2200`, then the two tables compared by eye.
That is the reading that would have caught both wrong diets **without the
element ledger at all**, which makes it worth having as an instrument rather
than as a procedure -- two checks that agree are worth more than either when
neither is derived from the other.

`--settle` now takes a list. `--settle 150,600,2200` settles one world through
all three depths, snapshots at each, and probes every candidate at each. The
sweep table reports the sustained rate at every depth, the factor from the
shallowest to the deepest, and a word for it:

* `>= 10x` down -- **transient**, and the shallow read is the one to throw away
* `2..10x` -- falling, still on its way somewhere
* `0.5..2x` -- steady across the sweep
* below that -- climbing, which is a pond still filling rather than draining

Ten is where the line is drawn because the reading this exists to catch missed
it by 1.9e5. The standing stock gets the same treatment in a second block, and
it is a genuinely independent column: on `renew.toml` HO2M's rate fell by
1.94e5 and its standing stock by 1.8e6 between the same two depths.

The summary sentence under each table is computed from the same three buckets
the rows are, which sounds obvious and was not: the first version counted only
transients and then announced that everything "held its rate" on a table whose
one row said "climbing 12x". A climb is its own finding -- the pond had not
finished making that substrate when the shallow probe ran, so the shallow
number is a **floor** rather than a transient -- and the summary now says which
of the three it is seeing. `the_bucket_and_the_word_agree` pins the two against
each other so they cannot drift apart again.

A sweep costs the **deepest** settle rather than the sum of the depths, because
a deep settle passes through every shallow one on its way. That was the point
of doing it in one pass.

Each depth's table is printed as its probes land, and each probe prints a line
as it finishes. Both commands used to print nothing until everything was done,
and a three-depth sweep at a 600 s probe is hours: thirty-five minutes in which
a run that had died and a run that was working looked exactly alike. The
per-probe line names which candidate finished, which is also how you find out
that one of them is taking as long as the other twenty-two together.

### The cache, which is what makes any of this routine

`--settle-cache DIR` keeps each settled world. The settle was 84% of a probe's
wall clock -- 1281 s of the 1531 s `--settle 2200` run -- and it is the same
lifeless pond every time, so paying for it once turns re-probing a config from
a thing you budget for into a thing you do. Both probe commands share the
directory.

Files are named `settle-<config digest>-<ticks>.snap` and **both halves are
checked on load**, not just used for lookup. `snapshot::load` rebuilds the
world from the *snapshot's own* config rather than from the one on the command
line, so a mismatched file would quietly probe a different pond and report it
under this config's name. The digest is taken from the config `supply` actually
settles -- cells switched off -- which is not the digest of the file on disk.

The two ways a cache entry can be wrong get opposite treatment, deliberately.

* **Unreadable** -- an older build wrote it, which happens whenever a field is
  added to the config, since that changes every digest and bumps the snapshot
  format. Re-settling from scratch gives exactly the right answer, so it prints
  what it is doing and does it. Killing a two-hour probe over a stale cache is
  a papercut that teaches people not to use the cache.
* **Readable but from a different config or a different tick** -- this is the
  dangerous one, because the result would be a real measurement of the wrong
  pond, reported under the right pond's name. It bails.

The wall-clock figure on each `settled` line is that depth's own, not the
cumulative one, because what a cache is worth is exactly the difference between
those two. On the second run of the same sweep it reads:

```
settled       20 s, 2000 ticks, read from cache, 0.1 s wall, snapshot 838 KiB
settled       60 s, 6000 ticks, read from cache, 0.1 s wall, snapshot 821 KiB
```

and the second of those two lines was written by `returns`, from a cache
`supply` filled.


## The return leg

Item 1d, and the honest version of the sentence that item ended with: *until
this exists, no number in this project bounds a population's food supply.*

`supply` asks how fast the pond replaces a compound **removed from the world**.
No organism ever asks that. A probe exports matter, and that is why `funded`
bounds it -- a sustained export cannot outrun the rate atoms enter, and only a
vent brings an atom in. A cell exports nothing. It turns its substrate into
products and leaves every atom exactly where it was, in the same voxel. So what
bounds a population is not whether the vents can deliver the atoms, which are
already here, but whether the light and the chemistry can drive the products
back round to the substrate.

`hadean returns` is the same perturbation reached the other way. It holds the
substrate at zero by **eating** it rather than by taking it away, and puts the
products where a cell would put them.

`World::turn_over(reaction)` is the mechanism, and it is `harvest`'s mirror:
in every voxel it consumes as much of the reaction's scarcest substrate as the
water holds and deposits the products beside it. Three things about it are
worth knowing.

* **Nothing is booked in the element ledger, because nothing crosses the
  boundary.** That is the difference from `harvest`, and it is what
  `the_return_leg_keeps_every_atom_in_the_pond` checks: the ledger may only
  move in the direction the vents move it.
* **It is a pure function of the state it is handed**, which
  `a_turnover_is_the_same_through_a_snapshot_and_a_copy` pins across a
  save/load and against the world the snapshot came from. `returns` restores
  every probe from one shared snapshot and runs them on separate threads, so
  without that property `--jobs` would change the answer and the whole table
  would be a measurement of the scheduler.
* **The enthalpy goes into the water as heat**, which is where it goes when the
  reaction runs on its own. A cell would keep `capture_efficiency` of it
  instead; that difference belongs to the cell layer, not to the supply.
* **The extent is shrunk by a couple of ulps before anything is applied**, and
  the *place* that is done is the point. `extent` is `have / n` minimised over
  the reactants, and `have / n * n` is not always `have` in `f64` -- at `n = 3`
  it can land an ulp above, which settles that amount to a small negative and
  leaves a compound count below zero in the field. The obvious fix is to clamp
  each reactant against its own stock as it is consumed. That fix is wrong: the
  products are still added at the full extent, so it **creates atoms**, quietly
  and far below every tolerance in the project. Mass balance here is
  structural, and a probe is not the place to start making it approximate.
  `a_turnover_creates_no_atoms_even_when_the_extent_rounds` drives every
  exergonic reaction at once for fifty ticks and holds the element ledger to
  1e-9.
* **The heat is accumulated from the deltas `ChemField::settle` actually
  applied**, not from `extent * dh` and *not* from the voxel's chemical total
  before and after. The reaction step takes the difference of the totals and is
  right to, because a step moves a visible fraction of the voxel. Here it is
  wrong: every exergonic reaction in this pond limits on a trace substrate, and
  differencing 1e-8 particles against the 1e12 sitting beside them in the same
  `f64` sum gives exactly zero. The first version did it that way and silently
  dropped the joules. Three readings of the same energy have to agree in
  `the_return_leg_pays_its_enthalpy_into_the_water` -- what the participants
  lost, what `turn_over` reports, and what the water gained.

That trace-substrate fact is worth reading twice, because it turned up as a
unit test rather than as a run. `drivable`, the test's own helper, could not
find an exergonic thermal reaction with an abundant limiting substrate at any
ranking -- **the abundant compounds in this pond are not food**, which is the
finding eight runs were spent on, reproducing itself in half a second.

### The failure mode it has to guard against, and the audit cannot

The probe drives the reaction forward and lets the water put the substrate
back. If the water does that through the **reverse of the same reaction**, the
loop is a futile cycle running on the heat the forward leg just deposited, and
a cell sitting in it would be extracting work from an ambient thermal bath.

**The energy audit stays perfectly flat through that**, which is the point
worth remembering. Nothing is created: the joules go from chemical to heat and
back, and `verify` has no opinion about it. Detailed balance is not violated
either -- `ea_r = ea_f - dh` holds, and the reverse flux is honest mass action
off the product pile the probe itself creates. Conservation and equilibrium are
both satisfied by a reading that is nonetheless a Maxwell demon.

Two things catch it, and they are independent.

**A control world, which is the measurement.** Each depth runs one extra world
from the same settled bytes with *nothing* driven in it, recording the pond's
chemical energy at each window boundary. A probe's `net` is how much further
its chemical total fell than the control's did. In a futile cycle the chemical
energy goes down on the forward leg and straight back up on the reverse, so
`net` is zero however large the turnover; a loop the light or the vents
actually feed shows up as a real drawdown. The table prints `gross` -- the
forward leg alone, which is what a naive reading would have reported -- beside
`sustains`, and their ratio is the diagnosis. Rows are ordered by `sustains`,
because ordering by turnover would put a futile cycle on top.

The subtraction is not exact and cannot be: a probe heats its own water, heat
moves the rate constants, and the two worlds' chemistry therefore walks apart
on its own. So the control's *own* window-to-window spread is carried as a
noise floor, and a net figure inside it prints `unresolved` rather than a
number. "This probe could not tell" is a different statement from "this living
is worth nothing", and every wrong turn in this project has come of collapsing
two statements like those into one. If the floor is itself above the pond's
sunlight, nothing can resolve at that depth and the output says so once rather
than printing `unresolved` twenty-three times and letting a reader conclude the
pond is empty.

`sustains` is **signed**. A metabolism can leave the pond holding more chemical
energy than the control does -- consuming a substrate can unblock a
photochemical route that stores more than the reaction released -- and that is
a real reading about this world rather than a small positive one. Clamping it
at zero would print the most interesting row in the table as nothing. A
negative row gets no head count, because a living that costs the pond nothing
is not a population.

**The pond's light budget, which is the sanity check.** Sunlight is the one
unambiguous source of low-entropy energy here and the ledger holds it exactly,
so a sustained power above `light_in / settle` cannot be a living whatever the
control says, and the output says `DISCARD IT` in those words with the ratio.
Vent chemistry is the other source of free energy in this world and this ledger
does not hold *its* free energy -- only the formation enthalpy the matter
carried -- so that bound is stated for reading alongside rather than enforced
as a filter. Making it exact needs a free-energy figure for the vent flux,
which does not exist yet and is the obvious next thing if a row ever lands
between the two.

### The first reading, which is a "could not tell"

`returns` on `renew.toml`, reaction 74 -- `HO2M + HM <=> 2 HOM`, the diet that
file names -- at settles 20 and 60 with a 20 s probe:

```
budget    the sun puts 6.455e-5 W into this pond, averaged over the settle

what the pond keeps paying for after 60 s of settling
    sustained  holding   larder s        gross     sustains   population  per newborn
    4.908e9/s    2.334    7.29e-1   2.626e-9 W   unresolved            -     28.509x

  (a net reading below the control pond's own drift of 1.110e-9 W reads `unresolved`)
```

Three things in that line are worth having.

* **`larder s` is 0.73.** The standing stock is worth less than a second of the
  turnover, so essentially none of what the probe took was a larder -- the
  water was handing it back. That is a different situation from every earlier
  reading in this project and it is what a return leg looks like when there is
  one.
* **`sustains` is `unresolved`, not zero.** Gross is 2.6e-9 W against a control
  drift of 1.1e-9 W, so the net is inside the noise. The run does not say the
  living is worthless; it says a 20 s probe cannot tell, and the closing line
  says exactly that and names the fix -- a longer `--probe` narrows the
  control's drift.
* **The floor falls as the pond settles**: 5.9e-8 W at 20 s, 1.1e-9 W at 60 s.
  The control drifts because the pond is still moving, so a deeper settle is
  also a more sensitive probe. That the two agree in that direction is the
  instrument checking itself.

Gross is four orders below the 6.5e-5 W of sunlight, so the `DISCARD IT` branch
does not fire and nothing here is a Maxwell demon on the light budget. What is
still open is the finer question the control asks, and that needs a longer
probe than the smoke test used.

### What it does not claim

It is an upper bound and it is meant to be. The water is the only limit in it;
a real cell is limited by its membrane and its enzymes as well. So a living
that appears in the table may still be out of reach -- which is what the `per
newborn` column, `subsistence` at the pond's concentrations, is there to say --
but a living that does **not** appear is a living no cell of any design could
make in this pond.

The table drops `opening` and carries `larder s` instead: the standing stock
divided by the sustained turnover, in seconds. That is the larder measured in
the one unit that makes it comparable to an income, and a pond holding a
hundred thousand seconds of its own resupply will feed a boom that looks like a
carrying capacity for as long as anyone watches. `holding` already carries
`opening`, since it is sustained over opening.

`per newborn` is `subsistence` at the pond's concentrations and is therefore
the same in every depth's table -- it reads the water, and the world is left
standing at the deepest settle. The output says so rather than leaving a reader
to assume each row was evaluated where its turnovers were.


## `configs/vent5.toml`, and the confound in it

Item 1's "second route", set up but not yet judged. `renew.toml` with
`vents.fuel_species = 5` instead of 2, so O2 -- fifth in this seed's vent fuel
list -- is injected and `82  O2 + HM <=> HO2M` has both reactants vent-fed. The
inflow line confirms it in one line:

```
renew.toml   the vents deliver H 3.600e10, M 1.200e10
             nothing brings C, O, N, S into this pond
vent5.toml   the vents deliver H 7.200e10, O 6.000e10, S 1.200e10, M 2.400e10
             nothing brings C, N into this pond
```

**It is not a single-variable change and the file says so at the top.**
`fuel_rate` is per species per vent -- `vent_matter` loops the species and
injects `fuel_rate * dt` of each into each vent voxel -- so five species inject
two and a half times the matter of two, not the same matter differently
distributed. Every standing stock in this pond will be larger than
`renew.toml`'s for that reason alone, so a bigger number here is not by itself
evidence that the new elements did anything.

The controlled version is `fuel_rate = 1.6e9`, holding total matter inflow
where `renew.toml` had it and changing only which elements it is made of. That
is the run to do **if** this one says the diet works, because "it works" and
"it works because oxygen now enters" are different claims and only the second
is the reason the file exists. Carbon still never enters at any
`fuel_species` on this seed.

The diet line in it is `renew.toml`'s, inherited so the file runs, and is
marked provisional in the header. It is what the measurement is supposed to
replace.

`the_vent5_preset_is_the_renew_pond_with_oxygen_coming_into_it` is its parity
test, and it does one thing the other three presets' tests do not. The change
here is *inside* `[vents]`, so parity cannot be asserted by comparing that
block; it names `fuel_species` as the one field that may differ. It then
asserts the mechanism rather than assuming it -- that the vent fuels at
`fuel_species = 5` carry oxygen and those at 2 do not -- and, finally, that
**no** `fuel_species` on this seed brings carbon in. That last one turns the
handoff's most load-bearing claim about seed 1 into something that fails loudly
if the chemistry generator ever changes, instead of quietly measuring a
different pond.

### What the sweep found, which is the first clean answer this question has had

`hadean supply --config configs/vent5.toml --settle 150,600,2200 --probe 600`,
two hours nine minutes of wall clock, at the deepest settle:

```
inflow    the vents deliver H 7.200e10, O 6.000e10, S 1.200e10, M 2.400e10
          nothing brings C, N into this pond

  compound       standing      opening    sustained       funded  holding
  HOM            4.768e12   1.800e10/s   1.756e10/s   2.400e10/s    0.976  renewed
  HO3M           4.782e13   1.200e10/s   1.243e10/s   2.000e10/s    1.036  renewed
  H2             3.793e13   1.201e10/s   1.201e10/s   3.600e10/s    1.000  renewed
  O2             3.268e12   1.200e10/s   1.200e10/s   3.000e10/s    1.000  renewed
  HM             5.075e10   1.200e10/s   1.200e10/s   2.400e10/s    1.000  renewed
  HO2M           1.640e11   1.157e10/s   1.192e10/s   2.400e10/s    1.031  renewed
```

**HO2M is renewed at 1.192e10/s, and every particle of it is paid for.** On
`renew.toml` at the same depth it was 4.543e4/s and `stock - no O enters this
pond`. That is a factor of **2.6e5**, and the column that was zero is now
2.4e10/s.

The `funded` figure is checkable by hand and checks out: HO2M is `H O2 M`, so
the inflow bound is `min(H/1, O/2, M/1) = min(7.2e10, 3.0e10, 2.4e10) =
2.4e10`. The instrument is doing the arithmetic the config states, on a ledger
it read rather than a number it was told.

### All three instruments agree, which is the point of having three

This is the first diet in this project to survive every check that has ever
caught one:

* **A rate rather than an amount.** 1.192e10/s of resupply, not 1.15e13
  particles standing in the water.
* **The element ledger.** `funded` 2.4e10/s, `vent-fed` 100%. The matter is
  delivered from outside; the pond is not being emptied.
* **The settle sweep.** 1.450e10 -> 1.199e10 -> 1.192e10 across 150, 600 and
  2200 s: `1.22x`, **steady across the sweep**. `renew.toml`'s HO2M fell by
  1.94e5 over the same two depths.

And the sweep calibrates itself in the same table: H2, HM and O2 all read
1.200e10/s at *every* depth, which is `vents = 3` times `fuel_rate = 4.0e9`
exactly, three separate times.

### Four livings, not one

```
     sustains   population  per newborn  vent-fed  reaction
   6.380e-9 W    63800 cells   404.420x      100%  HO2M + HM <=> 2 HOM
   4.116e-9 W    41156 cells  193597.702x    100%  H2 + HO3M <=> H2O + HO2M
   4.087e-9 W    40870 cells   837.053x      100%  H2 + HO2M <=> H2O + HOM
   3.993e-9 W    39933 cells   251.539x      100%  O2 + HM <=> HO2M
   1.814e-11 W     181 cells     0.088x        0%  O2 + CH2OS <=> CH2O3S
```

Every earlier version of this table had one candidate at the top and a cliff
under it. This one has **four livings within a factor of 1.6 of each other,
all fully vent-fed, all far above a newborn's upkeep** -- and then the cliff,
two and a half orders down, exactly where the carbon chemistry starts.

That matters beyond the top line. A pond with one living can only ever have one
diet; a pond with four that are all reachable is a pond where a lineage that
mutates off its enzyme has somewhere to land, which is what `evolve.toml`
exists to test and has never had.

`HO2M + HM <=> 2 HOM` is the best of them, which is the diet `vent5.toml`
already names -- inherited on trust from `renew.toml`, and now correct for a
different reason than it was chosen for. That is luck, not method, and the file
says so.

### The carbon pool is still draining, and now it is visible

Eight of twenty-three compounds are transients at the deep read, and they are
the carbon ones. CH4 falls by 1.2e3 between 150 s and 2200 s; CH2O2S -- the
compound every run before `renew.toml` was fed on -- reads 7.499e2/s at depth
against a standing stock of 1.154e13, which is a larder holding fifteen billion
seconds of its own resupply.

This is not a new finding, it is the old one finally in a column where it
cannot be missed: **carbon does not enter this pond at any `fuel_species`**, so
every carbon compound here is a closed pool being emptied, and the only
question was how long it took to look like one.

### What this does not establish

The numbers are inflated by the confound the config header names: five fuel
species inject 2.5x the matter of two, so 63800 cells is not 63800 cells at
`renew.toml`'s matter budget. **The mechanism is not in doubt** -- 4.543e4/s
with `funded` at exactly zero does not become 1.192e10/s with `funded` at
2.4e10/s because of 2.5x more of the same matter -- but the magnitudes are, and
`fuel_rate = 1.6e9` is the run that settles them.

And `supply` bounds a probe, not a population. `hadean returns` is the reading
that bounds a population and it has not been taken on this config yet.


## Mutator alleles, which the pond may now switch on for itself

Item 8, and the plan is right that it costs almost nothing: `replicate` already
took its rates as a parameter rather than reading a constant, so the whole
mechanism is a scale factor and where it comes from.

A `Regulator` whose `selector(2)` is 1 is a **replication factor** as well as a
transcription factor. Its [`Gene::bias`] -- one number, -4..4 -- is both the
direction and the size of the push, weighted by how much of the protein the
cell is holding, summed over the alleles and clamped to -1..1. That is
`Genome::mutator_drive`. What an octave of it is worth is
`CellConfig::mutator_range`, in octaves: at 2.0 a lineage can reach a quarter
of the configured rates or four times them.

Four decisions in that are worth the words.

* **Additionally, not instead.** A replication factor still binds promoters.
  Making `params[1]` choose between the two jobs would have taken half the
  regulators out of the regulatory network, which is a change to gene
  regulation dressed up as a change to mutation. A transcription factor that
  also upregulates an error-prone polymerase is what an SOS response is.
  `params[1]` was unused for this class, so reading it costs no format change
  and no existing byte decodes differently.
* **`mutator_range` defaults to zero and the mechanism is then entirely off.**
  Every run in this project was taken at a fixed rate, and a mechanism that
  quietly retunes the most consequential dial in the genome layer is not
  something to have on by default in the config that reproduces them. Two tests
  pin it: off by default, and turning the range up changes nothing for a genome
  that carries no allele.
* **The rates are scaled, so zero stays zero.** `MutationRates::none()` is
  still a clone line however hard a genome leans on it, which keeps the control
  a control -- `no_mutator_allele_can_break_a_clone_line`. And the per-byte
  rates are capped where `validate` caps them, so a genome cannot reach by the
  back door a rate the config layer would have refused at the front.
* **It is the mother's proteome.** Her polymerase does the copying; the
  daughter inherits the consequence and gets her own say at her own division,
  which is the loop that lets the rate evolve. `diverge_founder` passes an
  empty proteome, so a founding cohort always diverges at the configured rates
  -- a mutator allele is something a lineage is selected into, not something it
  is handed. The written ancestor carries none, and
  `the_ancestor_carries_no_replication_factor` says so.

The column is `mean_mutation_factor`, a multiplier rather than a rate because
all six operators move together and six columns saying the same thing is not
telemetry. One means the population replicates at what the config states.
`PLAN.md` predicts it rises while the environment moves and falls in stasis,
and this is the column that would show it.

**It has not been run.** The mechanism, its arithmetic and its off-switch are
tested; what a pond does with it is not measured, and the honest place to
measure it is a world whose food supply is not itself in question. That is item
1, not this.

### One consequence outside the genome layer

Adding a field to `CellConfig` changes the digest of every config, because
`WorldConfig::hash_state` hashes the serialised TOML. So **every snapshot and
every settle-cache file written by an earlier build is unreadable**, and
`snapshot::FORMAT` is bumped to 7 to say so in a sentence a reader can act on
rather than as "config does not match its own digest". The settle cache names
files by digest, so a stale file is never silently used; the error now names
the fix.


## What is left

### Immediately outstanding

0. ~~The `division_reserve` sweep.~~ **Done, and it closed rather than
   opened.** Five points from 4e-10 to 1e-8: the peak scales inversely with the
   dial, `births` equals the peak at every one of them, and births after the
   peak are zero at four of five. One generation, five times over. See "The
   `division_reserve` sweep, and what it closed". Nothing in `CellConfig` was
   ever going to fix this.
1. **Give the population a renewable living.** ~~Still the one that matters~~
   -- **and it now has an answer that survives every check that caught the
   previous two.** `configs/vent5.toml`: HO2M renewed at 1.192e10/s, 100%
   vent-fed, steady across a 150/600/2200 s sweep, with four livings above a
   newborn's upkeep rather than one. See "What the sweep found". What is left
   on this item is no longer "find a living" but "confirm it", in two specific
   ways, both listed at 1h below.
   The history is kept because the shape of the three wrong answers is worth
   more than the right one:
   * `hadean supply` measures a rate rather than an amount, and it agrees with
     the vents' known flux to within 0.5%. That much stands. What it does not
     do on its own is tell a rate from a stock being drained at a steady speed,
     because `holding` cannot: it gave HO2M **1.231**, accelerating, on a
     compound built from an element that does not enter this pond. See "The
     larder that held its rate".
   * **The `funded` column is in.** Provenance off the audit's element ledger,
     which is horizon-independent in the way a longer window never could be.
     On `renew.toml` two of twenty-two candidates come back renewed and they
     are the two the vents inject.
   * **The settle sweep is in**, and it catches the same class of error from
     the other side without touching the ledger, which makes the two checks
     independent rather than one restating the other. Cheap now that the
     settle is cached.
   * **`hadean returns` is in**, and it is the one that actually bounds a
     population -- `funded` bounds the probe and says so. Its own hazard is a
     futile cycle that passes the energy audit perfectly, which a control pond
     catches; see "The return leg". Nothing has been run on it at a probe long
     enough to resolve, and the first thing to do with it is exactly that.
   * **`configs/renew.toml`'s justification is void, though its numbers are
     not.** The 17333-cell boom and the 193 births after the peak happened; the
     diet they happened on is a larger larder, not a supply. Do not treat that
     header comment as a measurement to build on. The lifecycle numbers in it
     were honestly re-measured *against that diet* and will have to be
     re-measured again against whatever replaces it -- see item 2, which has
     now been right three times.
   * **The second route is built and being measured.** O2 is *fifth* in this
     seed's vent fuel list, so at `fuel_species = 5` the vents inject it,
     `O2 + HM -> HO2M` has both reactants vent-fed, and `HO2M + HM -> 2 HOM`
     could become a living paid for from outside the pond. That is
     `configs/vent5.toml`, and the element ledger confirms the mechanism in one
     line: O now enters at 6.000e10 atoms a second where `renew.toml` brought
     in none at all. **It is not a single-variable change** -- `fuel_rate` is
     per species, so five species is 2.5x the matter of two -- and the
     controlled follow-up is `fuel_rate = 1.6e9`. Read the section on it before
     reading its numbers.
     `runs/vent2.log` and `vent5.log` are **not** a test of this -- they are
     seed 5 on the old lifecycle numbers, where the pre-run check already said
     only the far tail could fund a daughter. Note the unfortunate name
     collision: those predate `configs/vent5.toml` and are unrelated to it.
   * Carbon never has an inflow on this seed at any `fuel_species`, so no
     carbon-based living is sustainable in this pond however the vents are set.
     That is a fact about seed 1's chemistry and worth checking on others
     before a seed is chosen to build on.
1b. ~~A settle-time sweep in `supply`.~~ **Done.** `--settle 150,600,2200`
   probes one pond at every depth in one pass and reports how each reading
   moved, with the standing stock swept beside the rate as an independent
   column. See "The sweep, and a settle you pay for once".
1c. ~~`--settle-out` / `--settle-in`.~~ **Done, as `--settle-cache DIR`**, which
   is one flag doing both directions because a sweep has to name the files
   itself anyway. Keyed by config digest and tick, and *both* are checked on
   load rather than merely used for lookup -- `snapshot::load` rebuilds from
   the snapshot's own config, so an unchecked cache hit would probe a different
   pond and report it under this config's name. `supply` and `returns` share
   the directory.
1d. ~~The return-leg probe.~~ **Done, as `hadean returns`.** It holds the
   substrate at zero by *eating* it rather than by taking it away, and leaves
   the products in the water where a cell would leave them, so what it measures
   is bounded by whether the light and the chemistry can drive them back --
   here `HOM --(band 7, +O2)--> HO3M --(+H2, rxn 30)--> HO2M`. `World::turn_over`
   is the mechanism and it is `harvest`'s mirror: matter never crosses the
   boundary, so there is no `funded` column and there should not be. See "The
   return leg", including why the heat had to be accumulated from the applied
   deltas rather than differenced from the voxel's totals.
1h. **Two things confirm the vent5 living, and neither is done.**
   * **`hadean returns` on it.** `supply` bounds a probe, which exports matter;
     a population does not. This is the only reading that bounds a population
     and it has never been taken at a probe long enough to resolve against the
     control pond's drift. Run it at `--settle 2200 --probe 600` -- the same
     depth the sweep settled on, so the cache serves both.
   * **The controlled matter budget, `fuel_rate = 1.6e9`.** Five fuel species
     inject 2.5x the matter of two, so 63800 cells is not 63800 cells at
     `renew.toml`'s budget. The mechanism is not in doubt -- 4.543e4/s at
     `funded` zero does not become 1.192e10/s at `funded` 2.4e10/s from 2.5x
     more of the same matter -- but the magnitudes are, and every lifecycle
     number re-measured against the wrong magnitudes would have to be measured
     again. Do this **before** item 2, not after.
1e. **`supply` has been run on two configs.** `configs/vent5.toml` is the
   `fuel_species = 5` arm and its three-depth sweep is the run in progress; see
   "`configs/vent5.toml`, and the confound in it" for what it is and what it is
   not. Still to do: `pond.toml` at full depth, and the other seeds. Cheap next
   to the runs it replaces, and cheaper again now the settle is cached -- the
   deep settle is paid once and every later probe of that config starts from
   it. `returns` should be run wherever `supply` is, on the same cache.
1f. **`renew.toml` is not the finished preset and should not be treated as
   one.** `membrane_scale` and `metabolic_rate` are set where the sweeps showed
   them to be inert, which is a floor and not a measurement, and
   `maintenance_power` and `division_reserve` were scaled by the two orders the
   subsistence arithmetic called for rather than tuned. What justifies each is
   in the file. `population_cap` is the thing to watch: `renewC` was heading
   for five figures within two hundred seconds, and a run that touches the cap
   proves nothing by construction.
1g. The 800000-tick `renew.toml` run was started and **killed at tick 180000**,
   deliberately, once the element ledger made its ending known: it was tracking
   `renewA` exactly and `renewA` ends with the food's oxygen gone. There is no
   longer a reason to run it. `runs/renew-long-killed.log` is the earlier
   attempt that died on its own at tick 300000, and it is the record of the
   pond going 100% dormant and continuing to fall. The 3000 s single-compound
   probe was also started and killed before it printed, so the "probe for
   longer" arm is untested -- the settle sweep answered the same question
   better.
2. The lifecycle numbers were tuned against seed 1's food chain. If the food
   chain changes, `division_reserve`, `membrane_scale` and `maximum_age` all
   have to be re-measured against it. Do not carry them over on trust; the
   dials' *meanings* are documented in `CellConfig` and hold, their *values* do
   not.
3. `configs/pond.toml` carries the code's defaults, which are the *untuned*
   lifecycle numbers — the tuned ones live in `configs/gate.toml`. Once the
   gate passes, decide whether the defaults should move with it. They should
   probably move; a default that goes extinct on its first night is a poor
   first impression.
4. The tuned numbers have only been run at 24×24×10. The full pond is twice as
   deep, which changes both the light gradient and the volume behind each
   square metre of surface, so they may not carry over unchanged.
5. `Traits` holds one field and is now the pre-genome path only. It is kept
   because it is the control, not because it is expected to grow: a trait to
   add belongs in a protein class.
5b. **The L6 dials have not been measured.** `swim_speed` and `motility_power`
   are reasoned from the pond's flow speed and a cell's upkeep, not measured
   against a run, and no lineage has yet been observed to keep a motility gene.
   `Channel::Heat` maps 0..100 C onto 0..1, which is most of the byte range
   spent on temperatures this pond never reaches. `Channel::Crowd` saturates at
   eight cells to a voxel, which was picked to be the order of a crowded voxel
   and not measured. None of these are wrong in a way that would show up as a
   bug; they are wrong in the way an unmeasured dial always is.

   One consequence of the transcription throttle was not designed for and is
   worth knowing about: `Structural` is not a signal class, so it decays under
   quiescence like the rest of the metabolism, and a deeply shut-down cell
   therefore has *less* famine tolerance than a working one. A spore ought to
   be tougher. It is second-order -- quiescence lowers the bill by up to twenty
   times, so a shut-down cell rarely reaches the branch where tolerance is
   consulted at all -- but if a run shows deep sleepers dying faster than
   shallow ones, this is where to look first.
6. **The lifecycle numbers have not been re-measured for a genome cell**, and
   should not be trusted on `evolve.toml`. A genome cell's membrane is
   `1 + transporters` rather than one flat multiplier, so at the same
   `membrane_scale` it takes up its food roughly twice as fast, and everything
   that turns on the break-even concentration has moved. The dials' meanings
   hold; their values were measured against a different cell.
6b. **`Action::Ingest` has almost no gradient to climb in this world.** An
   effector that closes a transporter is a real mechanism and it works, but
   nothing in the pond is *harmful* to take up: contents that are not
   metabolised simply sit there, and upkeep is charged on standing protein
   rather than on what is inside the cell. So closing a membrane against
   something saves nothing and selection cannot see it. The gate is worth
   having because it is what "that does not taste good" has to be made of, and
   it will start to matter the moment a compound can hurt -- which is a change
   to the chemistry, not to `neural.rs`.
7. **Horizontal gene transfer is not implemented** and is the largest missing
   piece of `PLAN.md`'s L3. It needs a lysing cell's genome to enter the voxel
   as a compound-like entity, which is a change to the mass audit rather than
   to `genome.rs`. The plan is emphatic about what it buys: good ideas
   propagate laterally instead of waiting for a lineage to reinvent them.
8. ~~Mutator alleles.~~ **Built, tested, and not yet run.** A `Regulator`
   whose `selector(2)` is 1 leans on its own polymerase; `mutator_range`
   defaults to zero so the mechanism is off until a config asks for it, and
   `mean_mutation_factor` is the column that would show a pond turning it up.
   See "Mutator alleles, which the pond may now switch on for itself". What a
   population does with it is unmeasured, and measuring it wants a world whose
   food supply is not itself the open question -- which is item 1.

### Phase 1 remainder

* **The 1M-tick audit run has not been done.** At 49 ticks/s that is ~5.7
  hours — worth starting in the background with `--csv` and checking the trend
  afterwards. The trend test is now calibrated for exactly this horizon.
* **Volumetric rendering of a compound** — Phase 1's demoable deliverable.
  There is no viewer at all. `hadean profile` and the `ecology` charts print
  text, which is a stand-in, not a substitute.
* **Prometheus metrics** for the headless runner (plan §5) are not implemented.

### Phase 2 remainder and beyond

Cell collision/soft-sphere physics and rendering are still absent. L5
multicellularity, L6 nervous systems, and L7 viewer have not started.

L3 is in — see "L3: the genome" above. One of the two hooks that were waiting
for it is spent and one is not:

* Every compound carries a `key: [f32; 8]` structural descriptor and every
  reaction carries the mean of its participants' keys. **This is what the
  genome matches against**, through `genome::KeyScale`, and it worked as the
  plan intended.
* `Network::step_voxel` still takes an empty `catalysis: &[f64]`. A genome
  cell's enzymes act on its *own contents*, in `metabolize`, not on the voxel
  it sits in — which is right for an intracellular enzyme and keeps the cell's
  f64 exactness. That slice is now for **secreted** enzymes, which is a
  different thing and needs `Effector` to do something: a protein a cell
  releases into the water, which is where extracellular digestion and
  predation come from.

---

## Judgement calls worth knowing about

* **CPU-first, not GPU.** There is no GPU in this environment, and a correct,
  deterministic, auditable CPU implementation is the right thing to port later.
* **`absorption_scale` is deliberately unphysical.** The pond is 0.5–1.5 mm
  deep, where real absorption cross-sections would give no vertical light
  gradient at all. Documented at the config field.
* **Default grid is 48×48×20**, not the plan's 200×200×60. That is a
  development-sized pond for a CPU implementation.
* **Molecules cap at 14 atoms** (`MAX_ATOMS`) with a 20000-node budget on
  canonical labelling. Exceeding the budget cannot break mass balance.
* **Reaction application is sequential in id order**, clamped against the
  amounts the previous reaction left. Order-dependent, but fixed, so
  reproducible, and it makes a negative amount impossible.
* **Advection stability is multidimensional.** `FaceVelocity::substeps` uses
  the largest sum of outward speeds over all six faces of a voxel.
* **`population_cap` is a safety cap, not an ecological parameter.** It is
  there so a runaway does not allocate the machine to death. Any run that
  touches it is reported as proving nothing, which is the point.
