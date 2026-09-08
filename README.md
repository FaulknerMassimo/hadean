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
  whole-genome duplication act on it at every division — and a regulator can
  lean on the cell's own polymerase, so **the mutation rate is itself
  heritable** and a lineage can choose to change faster or slower than the
  config says. Off by default, because every run so far was taken at a fixed
  rate
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
  harvest the world will sustain; and, since a probe exports matter and only a
  vent brings any in, how much of that harvest the world's inflow could
  actually pay for. Probe at several settle depths in one pass and it reports
  how each rate moved between them, which is the reading that separates a
  supply from a pond still on its way down
- `hadean returns`: what a *population* could live on, which is a different
  question and until now an unmeasured one. A probe exports matter; a cell does
  not. This one holds the substrate at zero by eating it rather than by taking
  it away, leaves the products in the water where a cell would leave them, and
  measures the turnover the pond will keep paying for — against a control pond
  with nothing driven in it, so a metabolism whose substrate comes back round
  the reverse of its own reaction reads as the thermal cycle it is

The current milestone is the Phase 2 population gate: a population that grows
into its food supply, crashes, and recovers. The machinery to judge it is in
place. **The gate is not met**, and the reason has moved five times — the first
three were the cell, the fourth was what the cells were given to eat, and the
fifth was the instrument that chose it. The fifth is the interesting one,
because it has happened three times in a row: an amount, then a rate read at
one moment, then a rate whose matter could not have been paid for. Each
correction was right and each left something for the next one to find, which is
why most of the recent work here is instruments rather than tuning.

**Those instruments have now found a living that survives all three of them.**
`configs/vent5.toml` sets the vents to inject oxygen, and at a 2200-second
settle `HO2M` comes back renewed at 1.192e10 particles a second with every one
of them paid for by matter entering the world — against 4.543e4/s and *"stock —
no O enters this pond"* on the config before it. Four separate livings come
back fully vent-fed, each paying a newborn hundreds of times its upkeep, and
the cliff below them is two and a half orders down, exactly where the carbon
chemistry starts. What that does *not* yet establish is the magnitudes: five
fuel species inject two and a half times the matter of two, which is a confound
the config says so at the top of itself.

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

`CH2O2S` is what every run in this project was handed, and ranked by amount the
ancestor gets a stock of ten million meals with nothing behind it. Ranked by
rate it was given `HO2M + HM -> 2 HOM` instead, worth five thousand times the
sustainable power.

**That was a larder too, and the same column that caught the first one
endorsed it.** `holding` is a rate held across a window, and a stock drained at
a constant speed holds its rate perfectly — holding its rate is what a constant
speed *is*. HO2M scored 1.231, accelerating, and it is built from oxygen, which
`vents.fuel_species = 2` does not deliver at any rate at all. The pond retires
its own oxygen in 2200 seconds with nothing living in it: probed at that settle
rather than at the 150 seconds `choose_metabolism` reads, the same lifeless
world reports HO2M's resupply down by a factor of 1.94e5 and O2 as a dead end.

So `supply` now reports a `funded` column beside the measured rate — the
largest harvest the vents' matter inflow could pay for, taken off the audit's
element ledger. Two of twenty-two candidates on this pond come back renewed,
and they are the two the vents inject:

```
inflow    the vents deliver H 3.600e10, M 1.200e10 atoms per second

  compound     standing      opening    sustained       funded  holding
  H2           1.333e13   1.215e10/s   1.213e10/s   1.800e10/s    0.998  renewed
  HM           4.621e11   1.208e10/s   1.205e10/s   1.200e10/s    0.998  renewed
  HO2M         1.854e10    6.920e9/s    8.817e9/s    0.000e0/s    1.274  stock - no O enters this pond
```

The bound is on the *probe*, which exports matter, and not on a population,
which does not — a cell turns its substrate into products and leaves every atom
in the pond, so a living can be sustained by matter that never enters at all
provided photochemistry drives the products back. `supply` says so in its own
output rather than leaving it to be inferred, and the measurement that would
settle it does not exist yet.

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

**The gate is still not met.** Both runs were still crashing when the clock ran
out, and for a while that looked like the whole of it — growing into a supply
takes seven times longer than sprinting through a stock, so the peak lands at
tick 187 000 where the larder's landed at 26 000. It was not the whole of it.
`renewA` ends with its population entirely dormant, nothing eating, and the
food still falling; the substrate's oxygen has no way into this pond and was
never coming back. The 17 333-cell boom and the 193 births are real
measurements of what those cells did. They were done on a larger larder.

What is next is not a longer horizon. It is `vents.fuel_species = 5`, which
puts O2 in the water and makes that diet vent-fed on both sides — one line, and
`supply`'s new column can now judge it before a run is spent on it.

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
mise exec -- cargo run -p hadean-headless --release -- supply \
  --config configs/renew.toml --settle 150,600,2200 --probe 600 \
  --settle-cache runs/settle
mise exec -- cargo run -p hadean-headless --release -- returns \
  --config configs/vent5.toml --settle 150,600,2200 --probe 600 \
  --settle-cache runs/settle
mise exec -- cargo run -p hadean-headless --release -- returns \
  --config configs/renew.toml --metabolism "HO2M + HM" --probe 3000
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

It reports two numbers for that rate and both matter. `sustained` is what the
chemistry actually handed over in the last window. `funded` is the most the
world's *matter inflow* could have paid for, off the audit's element ledger:
a probe exports matter, and only a vent brings any in — a photon carries
energy, not matter, and can rearrange an atom but cannot deliver one. The
bound is deliberately generous, crediting one compound with every atom the
vents delivered, so a compound reading zero there is not being resupplied at
all. Read `--settle` as part of the measurement rather than as setup: it
defaults to `seed_delay`, which is the instant the ancestors choose a diet, and
on `renew.toml` that instant is a transient. Give it a list — `--settle
150,600,2200` — and the same pond is probed at each depth in one pass, with a
sweep table saying which readings moved. That catches a transient without the
element ledger at all, which makes the two checks independent rather than one
restating the other. `--settle-cache DIR` keeps the settled worlds; the settle
was 84% of a probe's wall clock, a deep settle passes through every shallower
one, and both probe commands share the cache, so re-probing a config is free.

`returns` is the other half of that question, and the half nothing here
measured until now. `supply` asks how fast the pond replaces a compound
*removed from the world*, and no organism ever asks that: a probe exports
matter and a cell does not. A cell turns its substrate into products and leaves
every atom in the pond, so what bounds a population is not whether the vents
can deliver the atoms — they are already here — but whether the light and the
chemistry can drive the products back round. Same perturbation, reached the
other way: hold the substrate at zero by *eating* it, and put the products
where a cell would. There is no vent-fed column in its output and there should
not be, because nothing crosses the boundary for the element ledger to see.

What replaces it is a **control pond**. If the water puts the substrate back
through the reverse of the very reaction being driven, the loop runs on the
heat the forward leg just deposited — a cell living in it would be a heat
engine on an ambient bath, and the energy audit stays perfectly flat through
that, because nothing is created. So each depth also runs one world with
nothing driven in it, and a probe's net is how much further the pond's chemical
energy fell than the control's. `gross` and `sustains` are printed side by
side: far apart means the joules went round in a circle. A net reading inside
the control's own drift prints `unresolved` rather than a number — "the probe
could not tell" and "there is nothing there" are different findings, and
collapsing them is how this project has gone wrong before.

It is an upper bound and says so: the water is the only limit in it, where a
real cell is also limited by its membrane and its enzymes, so a living that
does not appear in that table is a living no cell of any design could make
here.

`--metabolism "HO2M + HM"` names one diet in exactly the spelling a config's
`cells.metabolism` uses, so an answer carries from the instrument to the config
without going through a reaction id that changes with the seed.

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
numbers re-measured against that diet, which is four more — and the diet it
argues for is now known to have been a stock, which its own header says at the
top. `configs/vent5.toml` is that pond with the vents set to inject oxygen, so
the compound in question is built from matter that enters the world; a fourth
parity test enforces that nothing else moved with it, and its header is candid
that injecting five fuel species rather than two also injects two and a half
times the matter, which is a confound and not a control.

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
