//! The headless runner.
//!
//! Long runs belong on a server, not behind a window. This binary is what
//! actually runs a world: it takes a config, steps it, logs the audit, and
//! writes snapshots the viewer can open later.
//!
//! `hadean verify` is the important one. It runs the gates the build plan sets
//! for Phase 0 and Phase 1 -- bit-identical replay, a clean save/reload, and a
//! flat energy audit -- and exits non-zero if any of them fails. That is the
//! command to put in CI.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use hadean_analysis::population::{population_curve, PopulationCurve, Thresholds};
use hadean_analysis::{analyse, chart, Series};
use hadean_chem::chemistry::Drive;
use hadean_chem::generate::kj;
use hadean_sim::{snapshot, World, WorldConfig};

#[derive(Parser)]
#[command(
    name = "hadean",
    version,
    about = "A bottom-up artificial life simulation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a world forward.
    Run {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Ticks to simulate.
        #[arg(short, long, default_value_t = 10_000)]
        ticks: u64,
        /// Resume from a snapshot instead of starting fresh.
        #[arg(long, value_name = "FILE")]
        resume: Option<PathBuf>,
        /// Write a snapshot here when the run finishes.
        #[arg(short, long, value_name = "FILE")]
        out: Option<PathBuf>,
        /// Write the audit time series here.
        #[arg(long, value_name = "FILE")]
        csv: Option<PathBuf>,
        /// Print a progress line every this many ticks. Zero disables it.
        #[arg(long, default_value_t = 0)]
        progress: u64,
    },
    /// Describe the chemistry a config generates.
    Chem {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Also list the reaction network.
        #[arg(short, long)]
        reactions: bool,
        /// Report the spread of affinity keys, which is what sets how far a
        /// mutated enzyme can reach.
        #[arg(short, long)]
        keys: bool,
        /// How many entries to list.
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
    },
    /// Run the Phase 0 and Phase 1 gates. Exits non-zero on failure.
    Verify {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Ticks to run for each check.
        #[arg(short, long, default_value_t = 2_000)]
        ticks: u64,
        /// Relative tolerance for the energy and mass audit.
        #[arg(long, default_value_t = 1.0e-6)]
        tolerance: f64,
    },
    /// Run a world and read its population curve against the Phase 2 gate.
    ///
    /// The gate is a shape, not a number: grow into the food supply, overshoot
    /// it, crash, and come back. Exits non-zero when the run does not show all
    /// three, or when it only stopped growing because it hit the safety cap.
    Ecology {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        #[arg(short, long, default_value_t = 20_000)]
        ticks: u64,
        /// Write the time series here.
        #[arg(long, value_name = "FILE")]
        csv: Option<PathBuf>,
        /// Print a progress line every this many ticks. Zero disables it.
        #[arg(long, default_value_t = 0)]
        progress: u64,
    },
    /// Print the world's vertical structure.
    Profile {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        #[arg(short, long, default_value_t = 2_000)]
        ticks: u64,
        /// How many compounds to list.
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
    },
    /// Measure throughput.
    Bench {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        #[arg(short, long, default_value_t = 500)]
        ticks: u64,
    },
    /// Measure what the pond can *resupply*, rather than what it holds.
    ///
    /// A large standing stock is not a living. This settles a lifeless world,
    /// takes one compound out of it entirely, and then keeps taking every
    /// particle the chemistry makes of it -- so what is reported is the
    /// largest harvest the pond will sustain, in particles per second, which
    /// is what a carrying capacity is made of. The standing stock is thrown
    /// away and not counted: that is the larder, and being fooled by the
    /// larder is what this command exists to stop.
    Supply {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Seconds of lifeless world before probing. Defaults to the config's
        /// `seed_delay`, so the pond is in the state the ancestors would have
        /// found it in.
        #[arg(long)]
        settle: Option<f64>,
        /// Seconds to hold each compound at zero for.
        #[arg(long, default_value_t = 600.0)]
        probe: f64,
        /// Probe only this compound, by name.
        #[arg(long, value_name = "NAME")]
        only: Option<String>,
        /// Probes to run at once. Each is an independent world, so this
        /// changes the wall clock and nothing else.
        #[arg(long, default_value_t = 4)]
        jobs: usize,
    },
    /// Print the default configuration as TOML.
    Config,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run {
            config,
            ticks,
            resume,
            out,
            csv,
            progress,
        } => run(config, ticks, resume, out, csv, progress),
        Command::Chem {
            config,
            reactions,
            keys,
            limit,
        } => chem(config, reactions, keys, limit),
        Command::Verify {
            config,
            ticks,
            tolerance,
        } => verify(config, ticks, tolerance),
        Command::Ecology {
            config,
            ticks,
            csv,
            progress,
        } => ecology(config, ticks, csv, progress),
        Command::Profile {
            config,
            ticks,
            limit,
        } => profile(config, ticks, limit),
        Command::Bench { config, ticks } => bench(config, ticks),
        Command::Supply {
            config,
            settle,
            probe,
            only,
            jobs,
        } => supply(config, settle, probe, only, jobs),
        Command::Config => {
            print!("{}", WorldConfig::default().to_toml());
            Ok(())
        }
    }
}

fn load_config(path: Option<PathBuf>) -> anyhow::Result<WorldConfig> {
    match path {
        Some(p) => WorldConfig::load(&p).with_context(|| format!("loading {}", p.display())),
        None => Ok(WorldConfig::default()),
    }
}

fn build(config: Option<PathBuf>) -> anyhow::Result<World> {
    World::new(load_config(config)?)
}

fn run(
    config: Option<PathBuf>,
    ticks: u64,
    resume: Option<PathBuf>,
    out: Option<PathBuf>,
    csv: Option<PathBuf>,
    progress: u64,
) -> anyhow::Result<()> {
    let mut world = match resume {
        Some(path) => {
            let w = snapshot::read_file(&path)
                .with_context(|| format!("resuming from {}", path.display()))?;
            println!("resumed {} at tick {}", w.config.name, w.tick());
            w
        }
        None => build(config)?,
    };

    describe(&world);
    let start_tick = world.tick();
    let mut series = Series::new();
    let began = Instant::now();

    for _ in 0..ticks {
        world.step();
        if world.audit_due() {
            let report = world.audit_now();
            series.record(&world, &report);
            if progress > 0 && report.tick % progress == 0 {
                println!(
                    "  tick {:>10}  t = {:>9.1} s  cells {:>6}  drift {:>11.3e} J ({:>9.2e} rel)  mean T {:.3} K",
                    report.tick,
                    world.elapsed(),
                    world.cells.alive(),
                    report.drift,
                    report.relative,
                    world.heat.mean_temperature(),
                );
            }
        }
    }

    let report = world.audit_now();
    series.record(&world, &report);
    let elapsed = began.elapsed().as_secs_f64();

    println!();
    println!(
        "ran {ticks} ticks in {elapsed:.2} s ({:.0} ticks/s)",
        ticks as f64 / elapsed
    );
    report_audit(&world, &series, world.tick() - start_tick);

    if let Some(path) = csv {
        series.write_csv(&path)?;
        println!("time series -> {}", path.display());
    }
    if let Some(path) = out {
        let size = snapshot::write_file(&world, &path)?;
        println!(
            "snapshot    -> {} ({:.1} MiB)",
            path.display(),
            size as f64 / (1024.0 * 1024.0)
        );
    }
    Ok(())
}

fn describe(world: &World) {
    let g = &world.grid;
    let (ex, ey, ez) = g.extent();
    println!(
        "world     {}  seed {}",
        world.config.name, world.config.seed
    );
    println!(
        "grid      {} x {} x {} = {} voxels at {:.0} um  ({:.2} x {:.2} x {:.2} mm)",
        g.nx,
        g.ny,
        g.nz,
        g.len(),
        g.dx * 1e6,
        ex * 1e3,
        ey * 1e3,
        ez * 1e3
    );
    println!(
        "chemistry {} compounds, {} reactions, {} photochemical, dt = {} s",
        world.chem.n_compounds(),
        world.chem.n_reactions(),
        world.chem.photo.len(),
        world.config.dt
    );
    if !world.config.cells.enabled {
        println!("cells     disabled");
    } else if let Some(reaction) = world.cells.metabolic_reaction {
        println!(
            "cells     {} alive, hardcoded metabolism: {}",
            world.cells.alive(),
            equation(&world.chem, reaction)
        );
    } else {
        println!(
            "cells     {} ancestors due at t = {} s; what they eat is chosen then, from \
             whatever the pond has made by that point",
            world.config.cells.initial_count, world.config.cells.seed_delay
        );
    }
}

fn report_audit(world: &World, series: &Series, span: u64) {
    let Some(report) = world.audit.last else {
        return;
    };
    let trend = analyse(series);
    println!();
    println!("energy audit");
    println!("  chemical      {:>14.6e} J", report.energy.chemical);
    println!("  thermal       {:>14.6e} J", report.energy.thermal);
    println!("  cellular      {:>14.6e} J", report.energy.cellular);
    println!("  total         {:>14.6e} J", report.energy.total());
    println!("  expected      {:>14.6e} J", report.expected);
    println!(
        "  drift         {:>14.6e} J  ({:.3e} relative)",
        report.drift, report.relative
    );
    println!("  mass drift    {:>14.6e}", report.mass_drift);
    println!("  light in      {:>14.6e} J", world.audit.ledger.light_in);
    println!(
        "  vent heat in  {:>14.6e} J",
        world.audit.ledger.vent_heat_in
    );
    println!(
        "  vent matter   {:>14.6e} J",
        world.audit.ledger.vent_chemical_in
    );
    println!(
        "  radiated out  {:>14.6e} J",
        world.audit.ledger.radiated_out
    );
    println!(
        "  trend         {} ({:.3e} J/tick over {} samples)",
        trend.verdict(span),
        trend.slope,
        trend.samples
    );
    println!();
    println!("cell population");
    println!("  alive         {:>14}", world.cells.alive());
    println!("  dormant       {:>14}", world.cells.dormant());
    println!("  decomposing   {:>14}", world.cells.decomposing());
    println!("  births        {:>14}", world.cells.births);
    println!("  deaths        {:>14}", world.cells.deaths);
    println!("  reserve       {:>14.6e} J", world.cells.total_reserve());
    if world.config.cells.enabled {
        println!("  curve         {}", curve_of(world, series).verdict());
    }
}

fn curve_of(world: &World, series: &Series) -> PopulationCurve {
    population_curve(
        series,
        world.config.cells.population_cap as u64,
        Thresholds::default(),
    )
}

/// Run a world and judge its population curve.
fn ecology(
    config: Option<PathBuf>,
    ticks: u64,
    csv: Option<PathBuf>,
    progress: u64,
) -> anyhow::Result<()> {
    let mut world = build(config)?;
    if !world.config.cells.enabled {
        bail!("this config has cells disabled; there is no population to judge");
    }
    describe(&world);

    // Said before the run rather than after it: a population whose best cell
    // cannot fund a division inside a lifetime has no recovery leg available,
    // and that is forty minutes you can decline to spend.
    let cells = &world.config.cells;
    let budget = cells.turnover_budget();
    println!(
        "turnover  division_reserve {:.2e} J against a budget of {:.2e} J -- {}",
        cells.division_reserve,
        budget,
        if cells.division_reserve <= budget {
            "a cell a standard deviation above the margin can fund a daughter"
        } else {
            "only the far tail of the population can fund a daughter, if anything can"
        }
    );
    if cells.genome {
        // The budget above uses `trait_spread` for the population's standing
        // variation, and in a genome world `trait_spread` is not what produces
        // it -- the mutation operators are, and they produced about a fifth as
        // much. Said here rather than buried, because acting on the wrong one
        // of these two numbers cost a whole run.
        println!(
            "          the budget above assumes a spread of {:.2}; a genome population's is \
whatever\n          its mutation operators produce, and is reported against the curve below",
            cells.trait_spread
        );
    }

    let mut series = Series::new();
    let began = Instant::now();
    // Said as soon as the pond has settled what the ancestors eat, which is
    // the first moment it *can* be said when the metabolism is chosen rather
    // than named. A cohort that cannot pay its own upkeep on the food in the
    // water shuts down on the tick it arrives and starves without ever
    // dividing, and no dial in the cell layer moves that -- eight runs
    // sweeping `membrane_scale` and `metabolic_rate` came back with the same
    // sixteen cells shut down in every one of them, four of the eight
    // byte-identical, before this line existed.
    let mut said = false;
    for _ in 0..ticks {
        world.step();
        if !said {
            if let Some(reaction) = world.cells.metabolic_reaction {
                let ratio = hadean_cell::subsistence(
                    &world.config.cells,
                    &world.chem,
                    reaction,
                    &world.abundance(),
                    &world.grid,
                );
                println!(
                    "subsist   {} earns a newborn {:.3}x its upkeep at equilibrium with the \
water -- {}",
                    equation(&world.chem, reaction),
                    ratio,
                    if ratio >= 1.0 {
                        "it can pay its way"
                    } else {
                        "it cannot pay its way, and nothing downstream of this will rescue it"
                    }
                );
                said = true;
            }
        }
        if world.audit_due() {
            let report = world.audit_now();
            series.record(&world, &report);
            if progress > 0 && report.tick % progress == 0 {
                let food = world.limiting_substrate();
                if world.config.cells.genome {
                    let traits = world.cells.trait_summary();
                    println!(
                        "  tick {:>10}  t = {:>9.1} s  cells {:>6} ({:>5} dormant, \
depth {:>4.2})  corpses {:>6}  food {:>10}  genes {:>4.1} ({:>3.1} signal)  \
lines {:>4}  diets {:>3}",
                        report.tick,
                        world.elapsed(),
                        world.cells.alive(),
                        world.cells.dormant(),
                        traits.mean_quiescence,
                        world.cells.decomposing(),
                        if food.is_finite() {
                            format!("{food:.3e}")
                        } else {
                            "-".into()
                        },
                        traits.mean_genes,
                        traits.mean_signal_genes,
                        traits.distinct_genomes,
                        world.cells.diet().len(),
                    );
                    continue;
                }
                println!(
                    "  tick {:>10}  t = {:>9.1} s  cells {:>6} ({:>5} dormant)  \
corpses {:>6}  food {:>10}",
                    report.tick,
                    world.elapsed(),
                    world.cells.alive(),
                    world.cells.dormant(),
                    world.cells.decomposing(),
                    // Infinite before the ancestors arrive: nothing eats yet,
                    // so nothing is scarce.
                    if food.is_finite() {
                        format!("{food:.3e}")
                    } else {
                        "-".into()
                    },
                );
            }
        }
    }
    let report = world.audit_now();
    series.record(&world, &report);

    let Some(reaction) = world.cells.metabolic_reaction else {
        bail!(
            "this chemistry offers no exergonic reaction running on anything the pond makes, \
             so there is no living for an ancestor here -- try another seed"
        );
    };
    println!();
    if world.config.cells.genome {
        // A genome population does not have "a metabolism": it has whatever
        // its lineages currently catalyse, which is the point of the layer and
        // is a list, not a line. The ancestral reaction is still printed
        // because it is where every one of them started.
        println!("ancestral   {}", equation(&world.chem, reaction));
        let diet = world.cells.diet();
        if diet.is_empty() {
            println!("diet        -- nothing living");
        } else {
            for (id, cells) in &diet {
                println!(
                    "diet        {:>4} cells  {}",
                    cells,
                    equation(&world.chem, *id)
                );
            }
        }
        let traits = world.cells.trait_summary();
        println!(
            "genomes     {:.1} genes, {:.0} bytes, {} clone lines among {} living",
            traits.mean_genes,
            traits.mean_genome_bytes,
            traits.distinct_genomes,
            world.cells.alive()
        );
        println!(
            "nervous     {:.1} signal genes per cell, mean quiescence {:.3}",
            traits.mean_signal_genes, traits.mean_quiescence
        );
    } else {
        println!("metabolism  {}", equation(&world.chem, reaction));
    }
    println!(
        "food        {}",
        world
            .metabolic_substrates()
            .iter()
            .map(|(name, amount, n)| format!("{n} {name} ({amount:.3e} left)"))
            .collect::<Vec<_>>()
            .join(" + ")
    );

    // Chart the populated run. The lifeless opening has no population to plot
    // and no food supply either -- nothing is eating, so nothing is scarce --
    // and leaving it in squeezes the part being judged into the right-hand
    // third of every chart.
    let alive = &series.rows[series
        .rows
        .iter()
        .position(|r| r.population > 0)
        .unwrap_or(0)..];
    let populations: Vec<f64> = alive.iter().map(|r| r.population as f64).collect();
    let substrate: Vec<f64> = alive.iter().map(|r| r.substrate).collect();
    let corpses: Vec<f64> = alive.iter().map(|r| r.decomposing as f64).collect();
    let dormant: Vec<f64> = alive.iter().map(|r| r.dormant as f64).collect();
    println!();
    println!(
        "{ticks} ticks ({:.1} s of world time) in {:.1} s",
        world.elapsed(),
        began.elapsed().as_secs_f64()
    );
    println!();
    print!("{}", chart("living cells", &populations, 72, 12));
    println!();
    // Plotted next to the population because it is the half of it that the
    // population column cannot show: a flat line of cells sitting out a night
    // and a flat line of cells working through one look identical up there.
    print!("{}", chart("of which dormant", &dormant, 72, 5));
    println!();
    print!("{}", chart("corpses decomposing", &corpses, 72, 5));
    println!();
    print!("{}", chart("food: scarcest substrate", &substrate, 72, 8));
    println!();
    // The population's evolutionary state. A flat line here is a population of
    // clones, which is the configuration that could never recover.
    let uptake: Vec<f64> = alive.iter().map(|r| r.mean_uptake).collect();
    print!("{}", chart("mean uptake trait", &uptake, 72, 5));

    if world.config.cells.genome {
        // Two curves the pre-genome world cannot draw. Clone lines is the one
        // to read first: it is the population's standing variation, and a
        // collapse to one is the freeze arriving by a longer road -- a
        // population that has become clones again cannot thin from the bottom
        // while its best cells still divide.
        println!();
        let lineages: Vec<f64> = alive.iter().map(|r| r.distinct_genomes as f64).collect();
        print!("{}", chart("clone lines", &lineages, 72, 5));
        println!();
        // And how many different livings are being made in the pond at once.
        // A second row here is a lineage that found one the ancestor did not
        // have, which is the whole purpose of the layer.
        let diets: Vec<f64> = alive.iter().map(|r| r.diet_breadth as f64).collect();
        print!("{}", chart("reactions being eaten", &diets, 72, 5));
    }

    let curve = curve_of(&world, &series);
    println!();
    println!("population curve");
    println!("  start         {:>10}", curve.start);
    println!(
        "  peak          {:>10}  at tick {}",
        curve.peak.population, curve.peak.tick
    );
    println!(
        "  trough        {:>10}  at tick {}",
        curve.trough.population, curve.trough.tick
    );
    println!(
        "  recovery      {:>10}  at tick {}",
        curve.recovery.population, curve.recovery.tick
    );
    println!("  finish        {:>10}", curve.finish);
    println!("  births        {:>10}", world.cells.births);
    println!("  deaths        {:>10}", world.cells.deaths);
    // Whether the plateau turns over. Without births after the peak there is
    // no recovery leg available at all, however healthy the curve looks.
    println!("  of which after the peak {:>3}", curve.births_after_peak);
    println!(
        "  food eaten    {:>10.2}x by the peak, {:.2}x back by the trough",
        curve.substrate_drawdown, curve.substrate_rebound
    );
    let traits = world.cells.trait_summary();
    if world.config.cells.genome {
        // The same check as the pre-run line, against the variation the
        // population actually carried rather than the variation the config
        // hoped for.
        //
        // Read at the peak, not at the finish. A run that ends extinct ends
        // with no variation at all, and a budget computed from that says only
        // that everything is dead. The peak is where the population had
        // settled into the pond and where the question -- can anything here
        // still afford a daughter? -- is the one that decides the rest of the
        // run.
        let cells = &world.config.cells;
        let at_peak = series
            .rows
            .iter()
            .min_by_key(|r| r.tick.abs_diff(curve.peak.tick))
            .map(|r| {
                if r.mean_uptake > 0.0 {
                    r.uptake_spread / r.mean_uptake
                } else {
                    0.0
                }
            })
            .unwrap_or_else(|| traits.relative_spread());
        let measured =
            cells.maintenance_power * (1.0 - cells.trait_cost) * at_peak * cells.maximum_age as f64;
        println!(
            "  turnover      {:>10.2e} J budget at the peak's spread of {:.3} -- {}",
            measured,
            at_peak,
            if cells.division_reserve <= measured {
                "a cell above the margin could fund a daughter"
            } else {
                "nothing at the margin could fund a daughter"
            }
        );
    }
    println!(
        "  uptake trait  {:>10.3}  mean, spread {:.3}",
        traits.mean_uptake, traits.uptake_spread
    );
    println!("  energy audit  {:>10.3e} relative drift", report.relative);
    println!();
    println!("  {}", curve.verdict());

    if let Some(path) = csv {
        series.write_csv(&path)?;
        println!();
        println!("time series -> {}", path.display());
    }

    if curve.passes() {
        Ok(())
    } else {
        bail!("the population did not demonstrate boom, bust and recovery")
    }
}

/// How far apart the chemistry's affinity keys are, and what that means for
/// the genome's recognition widths.
///
/// `enzyme_sigma` decides what a mutated enzyme can reach and there is no way
/// to pick it by taste. Set it far below the distance from one reaction to its
/// nearest neighbour and every reaction is an island: the ancestor catalyses
/// its own and no accessible mutation reaches a second. Set it far above the
/// spread of the whole network and one enzyme catalyses everything at once,
/// which is not a metabolism.
///
/// So the number to print is the distribution of nearest-neighbour distances
/// in key space, and the number to choose sits inside it: large enough that a
/// reaction's neighbours are reachable, small enough that the far side of the
/// network is not. The line at the bottom reports what the configured widths
/// actually buy, in reactions per enzyme, which is the quantity that matters.
/// How many slices a probe is cut into.
///
/// The first and the last are the two numbers that matter: a pond that hands
/// over as much in the last window as it did in the first is *producing* the
/// compound, and one whose rate has collapsed between them was handing over a
/// stock it cannot replace. Five is enough to see which, and few enough that
/// the table fits on a line.
const SUPPLY_WINDOWS: usize = 5;

/// What holding one compound at zero took out of the pond.
struct Harvest {
    compound: u16,
    /// Particles standing in the water when the probe began. Thrown away and
    /// deliberately not counted -- this is the larder, and mistaking it for a
    /// living is the error the whole command is built around.
    standing: f64,
    /// Particles the chemistry rebuilt in each window, in order.
    windows: [f64; SUPPLY_WINDOWS],
    /// Seconds of world time the whole probe covered.
    seconds: f64,
}

impl Harvest {
    fn rate(&self, w: usize) -> f64 {
        self.windows[w] / (self.seconds / SUPPLY_WINDOWS as f64)
    }

    /// What the pond delivered before it had to work for it.
    fn opening(&self) -> f64 {
        self.rate(0)
    }

    /// What it still delivered at the end, which is the only rate a population
    /// gets to plan around.
    fn sustained(&self) -> f64 {
        self.rate(SUPPLY_WINDOWS - 1)
    }

    /// Sustained over opening. Near one is a supply; near zero is the tail of
    /// a stock running out.
    fn holding(&self) -> f64 {
        let opening = self.opening();
        if opening > 0.0 {
            self.sustained() / opening
        } else {
            0.0
        }
    }
}

/// Measure the pond's resupply rate, compound by compound.
///
/// `choose_metabolism` has now been wrong three times in the same direction,
/// each time by trusting something that looked like food -- most recently by
/// picking the compound the pond held the most of, which turned out to be a
/// pool with one way in and no way out. The correction is not a better graph
/// walk. It is to stop reading an amount and read a rate, and the honest way
/// to read a rate off a world with this much coupling in it is to take the
/// compound away and watch what the chemistry does about it.
fn supply(
    config: Option<PathBuf>,
    settle: Option<f64>,
    probe: f64,
    only: Option<String>,
    jobs: usize,
) -> anyhow::Result<()> {
    let mut cfg = load_config(config)?;
    let settle = settle.unwrap_or(cfg.cells.seed_delay as f64);
    if probe <= 0.0 || settle < 0.0 {
        bail!("settle must be non-negative and probe must be positive");
    }
    // A probe measures the world, not the population. Leave the cells in and
    // the reading is the pond's production *minus whatever they were eating*,
    // which is the exact confound this command exists to remove.
    let cells = cfg.cells.clone();
    cfg.cells.enabled = false;
    let dt = cfg.dt;

    let mut world = World::new(cfg)?;
    describe(&world);
    println!(
        "probe     settle {settle:.0} s, then hold one compound at zero for {probe:.0} s, \
in {SUPPLY_WINDOWS} windows"
    );
    if cells.enabled {
        println!("          cells are switched off for the probe; the pond is measured empty");
    }

    // Every compound something could make a living on: the substrates of the
    // exergonic thermal reactions, which is the same filter the cell layer
    // uses when it picks a metabolism.
    let mut candidates: Vec<u16> = world
        .chem
        .reactions
        .iter()
        .filter(|r| r.drive == Drive::Thermal && r.dh < 0.0)
        .flat_map(|r| r.reactants.iter().map(|&(c, _)| c))
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    if let Some(name) = &only {
        let want = world
            .chem
            .by_name(name)
            .with_context(|| format!("no compound named {name} in this chemistry"))?;
        candidates.retain(|&c| c == want);
        if candidates.is_empty() {
            bail!("{name} is not a substrate of any exergonic thermal reaction here");
        }
    }
    println!(
        "          {} substrates to probe, at {:.0} s of world time each",
        candidates.len(),
        probe
    );

    let began = Instant::now();
    let settle_ticks = (settle / dt).round() as u64;
    for _ in 0..settle_ticks {
        world.step();
    }
    let base = snapshot::save(&world)?;
    println!(
        "settled   {} ticks, {:.1} s wall, snapshot {} KiB\n",
        settle_ticks,
        began.elapsed().as_secs_f64(),
        base.len() / 1024
    );

    let window_ticks = ((probe / dt).round() as u64 / SUPPLY_WINDOWS as u64).max(1);
    let seconds = (window_ticks * SUPPLY_WINDOWS as u64) as f64 * dt;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()?;
    let harvests: anyhow::Result<Vec<Harvest>> = pool.install(|| {
        candidates
            .par_iter()
            .map(|&compound| {
                // Each probe is its own world, restored from the same settled
                // state, so nothing about the result depends on `--jobs`.
                let mut w = snapshot::load(&base)?;
                let standing = w.harvest(compound);
                let mut windows = [0.0f64; SUPPLY_WINDOWS];
                for window in windows.iter_mut() {
                    for _ in 0..window_ticks {
                        w.step();
                        *window += w.harvest(compound);
                    }
                }
                Ok(Harvest {
                    compound,
                    standing,
                    windows,
                    seconds,
                })
            })
            .collect()
    });
    let mut harvests = harvests?;
    harvests.sort_by(|a, b| {
        b.sustained()
            .partial_cmp(&a.sustained())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("resupply, particles per second, holding each compound at zero");
    println!(
        "  {:<10} {:>12}  {:>11}  {:>11}  {:>7}  {}",
        "compound", "standing", "opening", "sustained", "holding", ""
    );
    for h in &harvests {
        println!(
            "  {:<10} {:>12.3e}  {:>9.3e}/s  {:>9.3e}/s  {:>7.3}  {}",
            world.chem.compound(h.compound).name,
            h.standing,
            h.opening(),
            h.sustained(),
            h.holding(),
            verdict(h),
        );
    }

    let rate: Vec<f64> = {
        let mut v = vec![0.0; world.chem.n_compounds()];
        for h in &harvests {
            v[h.compound as usize] = h.sustained();
        }
        v
    };

    // What that buys, in cells. A reaction can only turn over as fast as its
    // scarcest substrate is resupplied, and a cell keeps `capture_efficiency`
    // of what a turnover releases and spends `maintenance_power` staying
    // alive. The quotient is a carrying capacity, and it is the number this
    // project has been missing: everything before it was measured in joules
    // standing in the water, which is a larder and not an income.
    // A reaction's sustainable turnover is set by its *scarcest* substrate, so
    // the table below needs a measured rate for all of them. With `--only` the
    // rest are zero, which would not read as "not measured" -- it would read
    // as "resupplied at nothing", and the one reaction whose substrates all
    // happen to be the probed compound would be crowned on a rate the run
    // never took. Say nothing rather than that.
    if only.is_some() {
        println!(
            "\nno livings table: `--only` measured one substrate, and a living is bounded by \
its scarcest.\nrun without it to rank what this pond can feed."
        );
        println!("\n{:.1} s wall", began.elapsed().as_secs_f64());
        return Ok(());
    }

    let settled = world.abundance();
    let mut livings: Vec<(f64, f64, f64, u32)> = world
        .chem
        .reactions
        .iter()
        .filter(|r| r.drive == Drive::Thermal && r.dh < 0.0)
        .filter_map(|r| {
            let turnovers = r
                .reactants
                .iter()
                .map(|&(c, n)| rate[c as usize] / n.max(1) as f64)
                .fold(f64::INFINITY, f64::min);
            (turnovers > 0.0).then(|| {
                let power = turnovers * -r.dh * cells.capture_efficiency as f64;
                let reach =
                    hadean_cell::subsistence(&cells, &world.chem, r.id, &settled, &world.grid);
                (power, power / cells.maintenance_power, reach, r.id)
            })
        })
        .collect();
    livings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!(
        "the livings this pond can sustain, at capture_efficiency {:.2} and \
maintenance_power {:.2e} W",
        cells.capture_efficiency, cells.maintenance_power
    );
    // Two different bounds, and a living needs to clear both. `cells` is what
    // the *pond* can feed: total sustainable watts over one cell's upkeep. The
    // last column is what one cell can *reach*: a passive membrane holds its
    // own volume's share of what the water holds, so a dilute substrate is
    // dilute inside the cell too, however much of the pond is producing it.
    // The first column being large tells you nothing if the last one is below
    // one -- which on this chemistry is exactly the case that cost eight runs.
    println!(
        "  {:>11}  {:>11}  {:>10}  {}",
        "sustains", "population", "per newborn", "reaction"
    );
    if livings.is_empty() {
        println!("  none: no exergonic thermal reaction has all of its substrates resupplied");
    }
    for &(power, capacity, reach, id) in livings.iter().take(10) {
        println!(
            "  {:>9.3e} W  {:>7.0} cells  {:>8.3}x  {}",
            power,
            capacity,
            reach,
            equation(&world.chem, id)
        );
    }

    println!();
    let reachable = livings.iter().find(|l| l.2 >= 1.0);
    match (livings.first(), reachable) {
        (Some(&(power, capacity, reach, id)), _) if reach >= 1.0 => {
            println!(
                "the best living here is {} at {:.3e} W, which keeps {:.0} cells alive, \
and a newborn earns {:.2}x its upkeep on it",
                equation(&world.chem, id),
                power,
                capacity,
                reach
            );
        }
        (Some(&(power, capacity, reach, id)), best) => {
            println!(
                "the best living here is {} at {:.3e} W and {:.0} cells -- but a newborn \
earns only {:.3}x its upkeep on it,",
                equation(&world.chem, id),
                power,
                capacity,
                reach
            );
            println!(
                "so the pond produces a living no cell of this design can reach. That is a \
cell-layer problem, not a chemistry one."
            );
            match best {
                Some(&(p, c, reach, id)) => println!(
                    "the best living a newborn *can* pay its way on is {} at {:.3e} W \
and {:.0} cells ({:.2}x upkeep)",
                    equation(&world.chem, id),
                    p,
                    c,
                    reach
                ),
                None => println!(
                    "no reaction in this pond pays a newborn its upkeep at the concentrations \
the water holds"
                ),
            }
        }
        (None, _) => println!("this pond has no renewable living in it at all"),
    }
    println!("\n{:.1} s wall", began.elapsed().as_secs_f64());
    Ok(())
}

/// A word for what a probe found, so the table can be read without doing
/// arithmetic in your head.
fn verdict(h: &Harvest) -> &'static str {
    if h.sustained() <= 0.0 {
        "dead end - nothing comes back"
    } else if h.holding() >= 0.5 {
        "renewed"
    } else if h.holding() >= 0.05 {
        "draining"
    } else if h.standing / h.sustained() > 1.0e5 {
        "larder - a big stock on a trickle"
    } else {
        "draining"
    }
}

fn report_keys(chem: &hadean_chem::Chemistry, cfg: &hadean_cell::CellConfig) {
    fn distance(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    // Normalised space, which is where the genome's keys live and therefore
    // where a recognition width means anything. See `genome::KeyScale`.
    let scale = hadean_cell::genome::KeyScale::of(chem);
    let thermal: Vec<[f32; 8]> = chem
        .reactions
        .iter()
        .filter(|r| r.drive == Drive::Thermal && r.dh != 0.0)
        .map(|r| scale.normalise(&r.key))
        .collect();
    if thermal.len() < 2 {
        return;
    }

    let mut nearest: Vec<f32> = Vec::with_capacity(thermal.len());
    for (i, r) in thermal.iter().enumerate() {
        let mut best = f32::INFINITY;
        for (j, other) in thermal.iter().enumerate() {
            if i != j {
                best = best.min(distance(r, other));
            }
        }
        nearest.push(best);
    }
    nearest.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = |q: f32| nearest[((nearest.len() - 1) as f32 * q) as usize];

    // What the configured width actually reaches, averaged over the network.
    let reach = |sigma: f32| -> f64 {
        let floor = 0.01f32;
        let mut total = 0usize;
        for r in &thermal {
            for other in &thermal {
                let d = distance(r, other);
                if (-d * d / (sigma * sigma)).exp() >= floor {
                    total += 1;
                }
            }
        }
        total as f64 / thermal.len() as f64
    };

    let compounds: Vec<[f32; 8]> = chem
        .compounds
        .iter()
        .map(|c| scale.normalise(&c.key))
        .collect();
    let mut compound_nearest: Vec<f32> = Vec::with_capacity(compounds.len());
    for (i, c) in compounds.iter().enumerate() {
        let mut best = f32::INFINITY;
        for (j, other) in compounds.iter().enumerate() {
            if i != j {
                best = best.min(distance(c, other));
            }
        }
        compound_nearest.push(best);
    }
    compound_nearest.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let compound_reach = |sigma: f32| -> f64 {
        let mut total = 0usize;
        for c in &compounds {
            for other in &compounds {
                let d = distance(c, other);
                if (-d * d / (sigma * sigma)).exp() >= 0.01 {
                    total += 1;
                }
            }
        }
        total as f64 / compounds.len() as f64
    };

    println!();
    println!("affinity keys ({} thermal reactions):", thermal.len());
    println!(
        "  nearest neighbour   min {:.3}  p25 {:.3}  median {:.3}  p75 {:.3}  max {:.3}",
        nearest[0],
        quantile(0.25),
        quantile(0.5),
        quantile(0.75),
        nearest[nearest.len() - 1]
    );
    for sigma in [0.05f32, 0.08, 0.12, 0.2, 0.35] {
        println!(
            "  sigma {:>4}          an enzyme reaches {:>6.1} reactions{}",
            sigma,
            reach(sigma),
            if (sigma - cfg.enzyme_sigma).abs() < 1.0e-6 {
                "   <- configured"
            } else {
                ""
            }
        );
    }
    println!();
    println!("             ({} compounds):", chem.compounds.len());
    println!(
        "  nearest neighbour   min {:.3}  median {:.3}  max {:.3}",
        compound_nearest[0],
        compound_nearest[compound_nearest.len() / 2],
        compound_nearest[compound_nearest.len() - 1]
    );
    for sigma in [0.05f32, 0.08, 0.12, 0.2, 0.35] {
        println!(
            "  sigma {:>4}          a transporter reaches {:>4.1} compounds{}",
            sigma,
            compound_reach(sigma),
            if (sigma - cfg.transport_sigma).abs() < 1.0e-6 {
                "   <- configured"
            } else {
                ""
            }
        );
    }
    println!();
    println!(
        "  a point mutation moves one key component by {:.4}, so a lineage walks",
        1.0 / 255.0
    );
    println!("  the median nearest-neighbour gap in about {} substitutions -- which is",
        (compound_nearest[compound_nearest.len() / 2] / (1.0 / 255.0)).ceil() as u32);
    println!("  how an enzyme reaches a reaction its width does not already cover.");
}

fn chem(
    config: Option<PathBuf>,
    reactions: bool,
    keys: bool,
    limit: usize,
) -> anyhow::Result<()> {
    let cell_cfg = load_config(config.clone())?.cells;
    let config = load_config(config)?;
    let chem = hadean_chem::generate(config.seed, config.chemistry);
    chem.verify()
        .map_err(|e| anyhow::anyhow!("chemistry failed verification: {e}"))?;

    println!(
        "{} compounds, {} reactions ({} reactive compounds), seed {}",
        chem.n_compounds(),
        chem.n_reactions(),
        chem.reactive_compounds(),
        chem.seed
    );
    println!();
    println!(
        "{:<10} {:>8} {:>10} {:>10} {:>8} {:>6} {:>7}  reactions",
        "compound", "amu", "hf kJ/mol", "D m2/s", "perm", "charge", "peak"
    );
    for c in chem.compounds.iter().take(limit) {
        let peak = c
            .absorption
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(b, _)| b)
            .unwrap_or(0);
        println!(
            "{:<10} {:>8.1} {:>10.0} {:>10.2e} {:>8.1e} {:>6} {:>7} {:>10}",
            c.name,
            c.mass as f64 / hadean_core::units::AMU,
            kj(c.h_f),
            c.diffusion,
            c.permeability,
            c.charge,
            peak,
            chem.involving[c.id as usize].len(),
        );
    }
    if chem.n_compounds() > limit {
        println!("... and {} more", chem.n_compounds() - limit);
    }

    if keys {
        report_keys(&chem, &cell_cfg);
    }

    println!();
    println!("vent fuel (most energetic first):");
    for &f in &chem.vent_fuel {
        let c = chem.compound(f);
        println!(
            "  {:<10} {:>8.0} kJ/mol per atom",
            c.name,
            kj(c.enthalpy_per_atom())
        );
    }

    println!();
    println!("photochemistry:");
    for &id in &chem.photo {
        let r = chem.reaction(id);
        let Drive::Photo { band } = r.drive else {
            continue;
        };
        println!(
            "  band {band}  {:>7.0} kJ/mol uphill   {}",
            kj(r.dh),
            equation(&chem, id)
        );
    }

    if reactions {
        println!();
        println!("reactions (first {limit}):");
        for r in chem.reactions.iter().take(limit) {
            println!(
                "  {:>4}  dH {:>8.0}  Ea {:>7.0} kJ/mol  k(20C) {:>9.2e}  {}",
                r.id,
                kj(r.dh),
                kj(r.ea_f),
                r.k_forward(hadean_core::units::T_AMBIENT, 0.0),
                equation(&chem, r.id)
            );
        }
        if chem.n_reactions() > limit {
            println!("  ... and {} more", chem.n_reactions() - limit);
        }
    }
    Ok(())
}

fn equation(chem: &hadean_chem::Chemistry, id: u32) -> String {
    let r = chem.reaction(id);
    let side = |s: &[(u16, u8)]| {
        s.iter()
            .map(|&(c, n)| {
                if n == 1 {
                    chem.compound(c).name.clone()
                } else {
                    format!("{n} {}", chem.compound(c).name)
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
    };
    format!("{} <=> {}", side(&r.reactants), side(&r.products))
}

fn verify(config: Option<PathBuf>, ticks: u64, tolerance: f64) -> anyhow::Result<()> {
    let config = load_config(config)?;
    let mut failures = Vec::new();

    println!("verifying '{}' over {ticks} ticks", config.name);

    // Phase 0: the same config run twice must be bit-identical.
    print!("  determinism ......... ");
    let mut a = World::new(config.clone())?;
    let mut b = World::new(config.clone())?;
    a.run(ticks);
    b.run(ticks);
    if a.state_digest() == b.state_digest() {
        println!("ok  (digest {:016x})", a.state_digest());
    } else {
        println!("FAILED");
        failures.push("two identical runs diverged");
    }

    // Phase 0: saving and reloading must be invisible.
    print!("  snapshot replay ..... ");
    let bytes = snapshot::save(&a)?;
    let mut restored = snapshot::load(&bytes)?;
    if restored.state_digest() != a.state_digest() {
        println!("FAILED (reload differs)");
        failures.push("a reloaded snapshot did not match");
    } else {
        a.run(ticks / 2);
        restored.run(ticks / 2);
        if a.state_digest() == restored.state_digest() {
            println!("ok  ({} KiB)", bytes.len() / 1024);
        } else {
            println!("FAILED (diverged after reload)");
            failures.push("a reloaded world diverged from one that never stopped");
        }
    }

    // Phase 1: the energy and mass audit must stay flat.
    print!("  energy audit ........ ");
    let mut world = World::new(config.clone())?;
    let mut series = Series::new();
    for _ in 0..ticks {
        world.step();
        if world.audit_due() {
            let report = world.audit_now();
            series.record(&world, &report);
        }
    }
    let report = world.audit_now();
    series.record(&world, &report);
    if report.passes(tolerance) {
        println!("ok  (drift {:.3e} relative)", report.relative);
    } else {
        println!(
            "FAILED (drift {:.3e} relative, mass {:.3e})",
            report.relative, report.mass_drift
        );
        failures.push("the energy or mass audit drifted beyond tolerance");
    }

    print!("  drift trend ......... ");
    let trend = analyse(&series);
    if trend.is_growing(ticks) {
        println!("FAILED ({:.3e} J/tick)", trend.slope);
        failures.push("drift is growing, not just jittering");
    } else {
        println!("ok  ({})", trend.verdict(ticks));
    }

    println!();
    if failures.is_empty() {
        println!("all checks passed");
        Ok(())
    } else {
        for f in &failures {
            println!("  - {f}");
        }
        bail!("{} check(s) failed", failures.len())
    }
}

fn profile(config: Option<PathBuf>, ticks: u64, limit: usize) -> anyhow::Result<()> {
    let mut world = build(config)?;
    describe(&world);
    world.run(ticks);

    let temperature = world.temperature_profile();
    let light = world.light_profile();
    println!();
    println!(
        "vertical structure after {ticks} ticks ({:.1} s)",
        world.elapsed()
    );
    println!(
        "{:>6} {:>10} {:>14} {:>14}",
        "depth", "z (um)", "T (K)", "light (W/m2)"
    );
    for z in 0..world.grid.nz as usize {
        println!(
            "{:>6} {:>10.0} {:>14.4} {:>14.3}",
            z,
            (z as f32 + 0.5) * world.grid.dx * 1e6,
            temperature[z],
            light[z]
        );
    }

    println!();
    println!("most abundant compounds:");
    for (name, amount) in world.abundances().into_iter().take(limit) {
        println!(
            "  {name:<10} {amount:>12.4e} particles  ({:>10.3e} per voxel)",
            amount / world.grid.len() as f64
        );
    }
    Ok(())
}

fn bench(config: Option<PathBuf>, ticks: u64) -> anyhow::Result<()> {
    let mut world = build(config)?;
    describe(&world);
    // A few ticks first so the measurement is not dominated by warm-up.
    for _ in 0..10 {
        world.step();
    }
    let began = Instant::now();
    for _ in 0..ticks {
        world.step();
    }
    let elapsed = began.elapsed().as_secs_f64();
    let per_tick = elapsed / ticks as f64;
    println!();
    println!("{ticks} ticks in {elapsed:.3} s");
    println!("  {:.1} ticks/s", ticks as f64 / elapsed);
    println!("  {:.3} ms/tick", per_tick * 1e3);
    println!(
        "  {:.1} ns per voxel-tick",
        per_tick / world.grid.len() as f64 * 1e9
    );
    println!(
        "  simulated {:.1} s of world time per second of wall clock",
        world.config.dt / per_tick
    );
    Ok(())
}
