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
- L6 decision layer: receptors that sense the water, the cell's own reserve,
  light, heat, crowding and age; neurons that sum them, squash and remember;
  effectors that shut the cell down, hold off a division, open or close a
  transporter, or swim. Wiring, weights, biases and time constants are all in
  the genome, so *nothing* about what a cell does is written in the cell layer
  any more — only the interpreter that reads it
- Headless CLI, compressed snapshots, replay verification, CSV telemetry
- Population-curve analysis: boom, bust and recovery read out of a run rather
  than asserted
- `hadean supply`: what the pond can *resupply*, measured rather than guessed
  — it settles a lifeless world, takes a compound out of it, and keeps taking
  every particle the chemistry makes of it, so what comes back is the largest
  harvest the world will sustain

The current milestone is the Phase 2 population gate: a population that grows
into its food supply, crashes, and recovers. The machinery to judge it is in
place. **The gate is not met**, and the reason has moved four times — the first
three were the cell, and the fourth was not.

It was thought to be a famine, and it was ageing: every cell died at exactly
the same age, so the cohort that boomed together aged out together. Individual
lifespans fixed that. Then it was that nothing was ever born after the boom —
a hundred and thirty-six identical clones in a shared pond all break even at
the same concentration and freeze there, and a population that never breeds
cannot recover. Cells now carry a heritable trait, and then a genome, so they
differ in what they need and the pond can select between them. Then it was
dormancy: when to shut down and when to wake were two numbers in `CellConfig`
read by every cell in the pond, so the population could not disagree with
itself — it shut down as one and waited for a bar that was unreachable for all
of it at once. Those are now a receptor, a neuron and an effector in each
cell's own genome, and half a pond can sleep while the other half works.

Each of those was real and none of them was it. The thing they were all
downstream of is that **the pond had no carrying capacity, because the
ancestors were eating the wrong thing.**

`choose_metabolism` ranked candidate livings by how much of the substrate the
pond was holding. That is a larder, not an income, and nothing here had ever
measured an income. `hadean supply` does: it settles a lifeless pond, takes one
compound out of it, and then keeps taking every particle the chemistry makes of
it, so what has to be removed each second is the largest harvest the world will
ever support. Held against the vents — whose flux the config states outright at
1.2e10 particles per second — the instrument measures 1.206e10/s, which is it
checking itself against the one rate in this world that was never in doubt.

On seed 1 it says:

```
  compound     standing      opening    sustained  holding
  HM            4.621e11   1.208e10/s   1.206e10/s    0.998  renewed
  HO2M          1.854e10    6.913e9/s    8.512e9/s    1.231  renewed
  CH2O2S        1.150e13   3.048e9/s    1.027e7/s    0.003  larder
```

`CH2O2S` is what every run in this project was handed. The pond holds six
hundred times more of it than of `HO2M` and rebuilds eight hundred times less.
Ranked by amount the ancestor gets a stock of ten million meals with nothing
behind it; ranked by rate it should have had `HO2M + HM -> 2 HOM`, which
sustains 4.554e-9 W where the ancestral reaction sustains 8.4e-13 — **a factor
of five thousand in the living the pond can actually pay.**

That also settles the `division_reserve` question underneath it. Swept over
four and a half octaves, the dial buys peak population and nothing else: the
peak scales inversely with it, `births` comes out equal to the peak at every
point, and births after the peak are zero at four of five. One generation, five
times over, because births balancing deaths is what a carrying capacity is and
there wasn't one.

There is a third sense of "food" and it caught the first attempt to eat the
renewable one. `configs/renew.toml` is `gate.toml` with that single line
changed, and on the carried-over lifecycle numbers the cohort shut down on the
tick it arrived. `membrane_scale` was swept thirty-fold and `metabolic_rate`
twenty-fold — eight runs, sixteen cells shut down in every one, the four
`metabolic_rate` runs byte-identical and the four `membrane_scale` runs
differing only in the fourth digit of the food column.
Two dials that inert mean the quantity being tuned is not the one that binds:
**a passive membrane equilibrates, it does not concentrate**, so a cell holds
its own volume's share of what the water holds and no amount of membrane
changes that number. Six hundred times less substrate in the water is six
hundred times less inside the cell, and a newborn earned 0.059 of its own
upkeep. `hadean_cell::subsistence` is that arithmetic, and `ecology` now prints
it the moment the metabolism is settled — a microsecond for what those eight
runs took half an hour to say.

Re-tuned against the diet rather than carried over — the cell's whole energy
budget down two orders, which HANDOFF.md has been warning a change of food
chain would demand — the ancestors divide. Sixteen cells to a peak of 17 333,
and then **193 births after that peak**. Every run in this project's history
had recorded zero in that column, except one that recorded three; it is the
leg of the gate that has been missing from the start. The food column changes
character with it: under the larder every night's rebuild reached a tenth of
the last one, four orders down and never up, and here it falls to 2.4e3 and is
back at 5.2e6 ten thousand ticks later.

**The gate is still not met**, and the reason is now dull rather than deep:
both runs were *still crashing when the clock ran out*. Growing into a supply
takes seven times longer than sprinting through a stock, so the peak lands at
tick 187 000 where the larder's landed at 26 000, and 250 000 ticks — a run
length calibrated against the other pond — does not reach the far side. A
longer horizon is the next thing, rather than a mechanism that does not exist.

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
mise exec -- cargo run -p hadean-headless --release -- supply \
  --config configs/gate.toml --probe 600
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

`supply` is the measurement behind what the ancestors are given to eat. Every
earlier answer to that question was an amount — is this compound reachable, is
it one hop from sunlight, is there a lot of it — and all three picked a pool
the pond cannot refill. This one is a rate, and it is taken by perturbing the
world rather than by reading its graph: take the compound away, keep taking it,
and see what the chemistry does about it. Its answer goes into a config as
`cells.metabolism`, the same way `chem --keys` sets `enzyme_sigma`.

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
changed, enforced by another. `configs/renew.toml` is `gate.toml` eating the
living the pond can actually resupply, which is one line, and its lifecycle
numbers re-measured against that diet, which is four more.

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
