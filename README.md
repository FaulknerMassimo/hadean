# Hadean

Hadean is a bottom-up artificial-life simulation written in Rust. A generated,
mass-balanced chemistry lives in a lit, heated, flowing pond; protocells take
those compounds across a membrane, catalyse a downhill reaction, grow, divide,
die, and decompose back into the same chemistry.

The simulation is deterministic and energy-audited. There are no food tokens or
untracked life points: matter and energy remain part of the world throughout a
cell's lifecycle.

## Current state

- L0 substrate: units, counter-based randomness, fixed time, state hashing
- L1 chemistry: atom graphs, generated reversible reactions, catalysis
- L2 fields: conservative diffusion/advection, heat, light, static flow
- L3 genome: a linear byte string that decodes into proteins — enzymes that
  decide what a cell eats, transporters that decide what it can take up,
  structural protein that decides what it can endure, and regulators that
  decide how much of each it makes. Duplication, indels, inversion and
  whole-genome duplication act on it at every division
- L4 protocell: membrane transport, metabolism, maintenance, Brownian motion,
  division, dormancy, individual lifespans, death, decomposition
- Headless CLI, compressed snapshots, replay verification, CSV telemetry
- Population-curve analysis: boom, bust and recovery read out of a run rather
  than asserted

The current milestone is the Phase 2 population gate: a population that grows
into its food supply, crashes, and recovers. The machinery to judge it is in
place and the energetics are sized against the pond's measured photochemical
output. **The gate is not met**, and the reason has moved three times.

It was thought to be a famine, and it was ageing: every cell died at exactly
the same age, so the cohort that boomed together aged out together. Individual
lifespans fixed that. Then it was that nothing was ever born after the boom —
a hundred and thirty-six identical clones in a shared pond all break even at
the same concentration and freeze there, and a population that never breeds
cannot recover. Cells now carry a heritable trait, so they differ in what they
need and the pond can select between them. It helps: the plateau rises from 136
to 156 and the first runs finish with survivors rather than extinct. It is not
enough on its own, and how much variation would be enough turns out to be an
inequality you can check before starting a run.

Underneath both, the food column says something neither of those explanations
reached. Over four days the larder falls from 1.15e13 particles to 1e9, in
every configuration, at almost exactly the same rate whether the pond carries
136 cells or 156 — while a pond containing one cell that cannot eat holds it
flat. This population is not failing to regulate. Its food is made from CO2,
CO2 takes part in exactly one reaction in this chemistry, and the waste it
excretes has no way back: it is **mining a finite pool**. A boom, a bust and a
recovery need a renewable resource, and this world does not have one for this
metabolism. HANDOFF.md has the measurements and what to do about it.

So the genome arrived early, out of order, because the diagnosis pointed at it.
The pre-genome cell has one heritable number and *every cell in the pond
catalyses the same reaction*, chosen once and fixed for the run — which is the
mechanism underneath the mining. A population that can only ever eat one thing
cannot respond to that thing running out. With a genome, what a cell eats is an
enzyme, an enzyme is eight bytes matched against the reaction network, and
daughters differ from their mothers.

The pond has a way out and does not use it: `O2 + CH2OS -> CH2O3S` is downhill
and has a rate constant of 8e-15, which is no route at all. Lowering a barrier
is precisely and only what an enzyme does, so a lineage that evolves onto that
reaction is a decomposer, and a pond with a decomposer in it recycles its
carbon instead of retiring it. That is the mechanism.

**No run has found it, and the runs so far cannot.** A genome adapts across
generations, and on seed 1 the population only ever gets one: it booms into the
larder, and then either no cell at the margin can afford a daughter or the
whole plateau is shut down dormant, depending on which way `division_reserve`
is turned. Both ends give a single generation, because births balancing deaths
is what a carrying capacity is and this pond has none. The genome is not
blocked by anything in the genome; it is queued behind the renewable living
that was already the outstanding problem. What the runs do show is that the
machinery works under load — 617 lineages against a clone control's 1, standing
variation of 0.357 against exactly 0.000, and an energy audit flat to 5e-11
across two thousand cells.

`configs/evolve.toml` is the gate pond with the genome switched on; a test
enforces that the two differ in nothing but the cell. `cells.genome` is off by
default and the pre-genome path is arithmetically unchanged — checked by
diffing 20 000 ticks of telemetry, not asserted — because every claim this
project makes came out of an A/B against a control.

## Run it

The Rust toolchain is pinned locally with [mise](https://mise.jdx.dev/):

```sh
mise install
mise exec -- cargo run -p hadean-headless --release -- run \
  --config configs/pond.toml --ticks 10000 --progress 1000
```

Useful commands:

```sh
mise exec -- cargo run -p hadean-headless --release -- config
mise exec -- cargo run -p hadean-headless --release -- chem --reactions
mise exec -- cargo run -p hadean-headless --release -- profile --ticks 2000
mise exec -- cargo run -p hadean-headless --release -- verify --ticks 2000
mise exec -- cargo run -p hadean-headless --release -- ecology \
  --config configs/gate.toml --ticks 250000
mise exec -- cargo run -p hadean-headless --release -- ecology \
  --config configs/evolve.toml --ticks 250000
mise exec -- cargo run -p hadean-headless --release -- chem --keys
mise exec -- cargo test --release --workspace
```

`verify` is the conservation gate: bit-identical replay, a clean save/reload,
and an energy audit that does not drift. `ecology` is the population gate: it
runs a world, draws its population and food curves, and exits non-zero unless
the population grew into its food supply, overshot it, crashed, and recovered
without ever leaning on the safety cap.

Runs can emit audit/population telemetry with `--csv runs/pond.csv` and resume
bit-identically through `--out runs/pond.snap` / `--resume runs/pond.snap`.

`chem --keys` is the measurement behind the genome's recognition widths: it
reports how far apart the chemistry's affinity keys actually are, and how many
reactions one enzyme reaches at each width. The first widths in this project
were guessed, and were wrong by a factor that made every ancestor's enzyme
match nothing at all.

`configs/pond.toml` is the world at full size; `configs/gate.toml` is the same
world at an eighth of the volume, which is where the population gate is tuned
and where it runs in twenty minutes rather than an afternoon. Everything but
the grid and the cell lifecycle numbers is identical, and a test enforces that.
`configs/evolve.toml` is `gate.toml` with the genome on and nothing else
changed, enforced by another.

## The sun has to move

`configs/pond.toml` runs a ten-minute day. That is not decoration. There was
briefly a second preset that fixed the sun in the sky, on the theory that it
made a population crash unambiguously about resources rather than about
nightfall. It made the pond uninhabitable: under a fixed sun the compound the
ancestor eats builds to a peak and then decays to nothing, because the
photoreaction drains its own precursor and nothing regenerates it. The night is
what lets the chemistry recover. In this world the day/night cycle is not a
confound in the experiment, it is the experiment's renewable resource.

That still holds, and it is not the whole story. The lifeless pond under a
moving sun holds its food at 1.15e13 particles indefinitely, so the sun really
does keep the chemistry turning over. What it cannot do is put back an atom the
population has retired: the day/night cycle renews the *state* of the
chemistry, not its *stock*. Both have to be renewable, and only one of them is.

See [PLAN.md](PLAN.md) for the full design and [HANDOFF.md](HANDOFF.md) for
implementation notes and conservation details.

## License

MIT — see [LICENSE](LICENSE).
