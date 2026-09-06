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
- L4 protocell (pre-genome): membrane transport, metabolism, maintenance,
  Brownian motion, division, dormancy, individual lifespans, death,
  decomposition
- Heritable traits: cells differ in how much membrane they carry, daughters
  inherit their mother's with a mutational kick, and the pond selects — the
  first evolution in the project
- Headless CLI, compressed snapshots, replay verification, CSV telemetry
- Population-curve analysis: boom, bust and recovery read out of a run rather
  than asserted

The current milestone is the Phase 2 population gate: a population that grows
into its food supply, crashes, and recovers. The machinery to judge it is in
place and the energetics are sized against the pond's measured photochemical
output. **The gate is not met**, and the reason has moved twice.

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

After Phase 2 comes the L3 genome, which replaces the protocell's hardcoded
traits with a real one.

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
mise exec -- cargo test --release --workspace
```

`verify` is the conservation gate: bit-identical replay, a clean save/reload,
and an energy audit that does not drift. `ecology` is the population gate: it
runs a world, draws its population and food curves, and exits non-zero unless
the population grew into its food supply, overshot it, crashed, and recovered
without ever leaning on the safety cap.

Runs can emit audit/population telemetry with `--csv runs/pond.csv` and resume
bit-identically through `--out runs/pond.snap` / `--resume runs/pond.snap`.

`configs/pond.toml` is the world at full size; `configs/gate.toml` is the same
world at an eighth of the volume, which is where the population gate is tuned
and where it runs in twenty minutes rather than an afternoon. Everything but
the grid and the cell lifecycle numbers is identical, and a test enforces that.

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
