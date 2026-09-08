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
use hadean_chem::element::{self, N_ELEMENTS};
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
        /// found it in -- which is also the instant `choose_metabolism` reads
        /// it at, and on this chemistry that instant is a transient.
        ///
        /// Give several, comma separated, to probe the same pond at several
        /// depths: `--settle 150,600,2200`. A rate that falls between them was
        /// a stock being emptied at a steady speed, which is the one thing a
        /// single reading cannot tell from a supply however long its window.
        #[arg(long, value_delimiter = ',')]
        settle: Vec<f64>,
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
        /// Directory of settled snapshots, read from and written to.
        ///
        /// The settle is most of a probe's wall clock and it is the same
        /// lifeless world every time, so keeping it makes re-probing a config
        /// nearly free and a settle sweep affordable. Files are keyed by the
        /// config's digest and the tick they hold, so a cache directory serves
        /// several configs and cannot hand back the wrong pond.
        #[arg(long, value_name = "DIR")]
        settle_cache: Option<PathBuf>,
    },
    /// Measure what a population could live on, by running its metabolism in
    /// the water with no cells in it.
    ///
    /// `supply` asks how fast the pond replaces a compound removed from the
    /// world, and no organism ever asks that: a probe exports matter and a
    /// cell does not. A cell turns its substrate into products and leaves
    /// every atom in the pond, so what bounds a population is whether the
    /// light and the chemistry can drive those products back round. Same
    /// perturbation -- hold the substrate at zero -- reached by eating it
    /// rather than by taking it away, which is the difference between a bound
    /// on the probe and a bound on a population.
    Returns {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Seconds of lifeless world before probing. Comma-separate for a
        /// sweep, exactly as in `supply`.
        #[arg(long, value_delimiter = ',')]
        settle: Vec<f64>,
        /// Seconds to run each metabolism for.
        #[arg(long, default_value_t = 600.0)]
        probe: f64,
        /// Probe only this reaction, by id. `hadean chem --reactions` lists
        /// them.
        #[arg(long, value_name = "ID", conflicts_with = "metabolism")]
        reaction: Option<u32>,
        /// Probe only this metabolism, named by its substrates the way a
        /// config's `cells.metabolism` names one -- `"HO2M + HM"`.
        ///
        /// The same spelling as the field this command exists to decide, so an
        /// answer can be carried from one to the other without translating
        /// through a reaction id that changes with the seed.
        #[arg(long, value_name = "SUBSTRATES")]
        metabolism: Option<String>,
        /// How many metabolisms to list per depth.
        #[arg(short, long, default_value_t = 12)]
        limit: usize,
        /// Probes to run at once. Each is an independent world, so this
        /// changes the wall clock and nothing else.
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        /// Directory of settled snapshots, read from and written to. Shared
        /// with `supply`: the settled pond is the same world, so a depth
        /// settled by one command is free for the other.
        #[arg(long, value_name = "DIR")]
        settle_cache: Option<PathBuf>,
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
            settle_cache,
        } => supply(config, settle, probe, only, jobs, settle_cache),
        Command::Returns {
            config,
            settle,
            probe,
            reaction,
            metabolism,
            limit,
            jobs,
            settle_cache,
        } => returns(
            config,
            settle,
            probe,
            reaction,
            metabolism,
            limit,
            jobs,
            settle_cache,
        ),
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
                    let traits = world.cells.trait_summary(&world.config.cells);
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
        let traits = world.cells.trait_summary(&world.config.cells);
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
    let traits = world.cells.trait_summary(&world.config.cells);
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
    /// Atoms of each element the vents delivered during each window. Matter
    /// crosses this world's boundary in exactly two places -- `vent_matter`
    /// puts it in, `harvest` takes it out -- and both book into
    /// `Audit::elements_in`, so this is read back off the ledger rather than
    /// taken on trust from the config's stated flux.
    vents: [[f64; N_ELEMENTS]; SUPPLY_WINDOWS],
    /// Atoms per particle of the compound being probed.
    formula: [u16; N_ELEMENTS],
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

    fn window_seconds(&self) -> f64 {
        self.seconds / SUPPLY_WINDOWS as f64
    }

    /// The largest harvest the vents could have *paid for* in window `w`,
    /// particles per second.
    ///
    /// A probe exports matter: what it takes out of the pond never comes back.
    /// So a sustained harvest is bounded by the rate at which the world gains
    /// the atoms the compound is made of, and the only thing that brings atoms
    /// in is a vent -- a photon carries energy, not matter, and can rearrange
    /// an atom but cannot deliver one.
    ///
    /// The bound is deliberately generous. It credits this one compound with
    /// *every* atom the vents delivered, as though nothing else in the pond
    /// wanted any of them, and it ignores the standing stock entirely. A
    /// compound that still reads zero under that accounting is not being
    /// resupplied at all, and no length of window will change it.
    fn funded(&self, w: usize) -> f64 {
        let seconds = self.window_seconds();
        (0..N_ELEMENTS)
            .filter(|&e| self.formula[e] > 0)
            .map(|e| self.vents[w][e] / seconds / self.formula[e] as f64)
            .fold(f64::INFINITY, f64::min)
    }

    /// What fraction of the sustained harvest the world's inflow can pay for.
    ///
    /// One means every particle handed over was matched by the matter to build
    /// another one. Zero means the whole measured rate was the pond being
    /// emptied at a steady speed -- which is the case `holding` cannot see,
    /// because a steady speed holds its rate across a window by definition.
    fn provenance(&self) -> f64 {
        let sustained = self.sustained();
        if sustained > 0.0 {
            (self.funded(SUPPLY_WINDOWS - 1) / sustained).min(1.0)
        } else {
            0.0
        }
    }

    /// The element whose inflow binds the harvest: the one the pond runs short
    /// of first, and the specific thing that is missing when a resupply is
    /// really a stock.
    fn starved(&self) -> Option<u8> {
        let seconds = self.window_seconds();
        let per = |e: usize| self.vents[SUPPLY_WINDOWS - 1][e] / seconds / self.formula[e] as f64;
        (0..N_ELEMENTS)
            .filter(|&e| self.formula[e] > 0)
            .min_by(|&a, &b| per(a).partial_cmp(&per(b)).unwrap_or(std::cmp::Ordering::Equal))
            .map(|e| e as u8)
    }
}

/// One settled lifeless world, and every probe taken from it.
struct Checkpoint {
    /// Seconds of world time the pond had been left alone for.
    settle: f64,
    harvests: Vec<Harvest>,
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
///
/// It was then wrong a fourth time inside this command, by reading that rate
/// at one instant. `--settle` defaults to `seed_delay`, which is exactly when
/// the ancestors are handed their diet, so both instruments read the same
/// transient at the same moment and neither could see it was one. Probing at
/// several depths is the cheap half of the answer -- the element ledger is
/// the other half, and the two are independent.
fn supply(
    config: Option<PathBuf>,
    settle: Vec<f64>,
    probe: f64,
    only: Option<String>,
    jobs: usize,
    settle_cache: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut cfg = load_config(config)?;
    let dt = cfg.dt;
    if probe <= 0.0 {
        bail!("probe must be positive");
    }
    let depths = settle_depths_of(settle, cfg.cells.seed_delay as f64, dt)?;

    // A probe measures the world, not the population. Leave the cells in and
    // the reading is the pond's production *minus whatever they were eating*,
    // which is the exact confound this command exists to remove.
    let cells = cfg.cells.clone();
    cfg.cells.enabled = false;

    let mut world = World::new(cfg)?;
    describe(&world);
    println!("probe     hold one compound at zero for {probe:.0} s, in {SUPPLY_WINDOWS} windows");
    println!(
        "settles   {} -- the same pond, probed at each depth",
        depths
            .iter()
            .map(|(s, _)| format!("{s:.0} s"))
            .collect::<Vec<_>>()
            .join(", ")
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
    let window_ticks = ((probe / dt).round() as u64 / SUPPLY_WINDOWS as u64).max(1);
    let seconds = (window_ticks * SUPPLY_WINDOWS as u64) as f64 * dt;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()?;

    let settled = settle_depths(&mut world, &depths, settle_cache.as_ref())?;

    // The inflow is a cumulative ledger over a flux that does not vary, so the
    // deepest settle is its best-averaged reading. It is printed before the
    // tables because it decides most of them in advance -- and it is available
    // as soon as the settling is done, which is why the settling all happens
    // first.
    let deepest = depths.last().map(|&(s, _)| s).unwrap_or(0.0);
    println!();
    report_inflow(&world, deepest);

    // Each depth's table is printed as its probes land rather than at the end.
    // A three-depth sweep at a 600 s probe is an hour and a half of wall clock,
    // and a command that says nothing for an hour and a half is one you cannot
    // tell from a hung one.
    let mut checkpoints: Vec<Checkpoint> = Vec::new();
    for (&(settle, _), base) in depths.iter().zip(&settled) {
        let harvests = probe_all(
            base,
            &candidates,
            window_ticks,
            seconds,
            &pool,
            &Progress::new(candidates.len(), settle),
        )?;
        println!();
        println!(
            "resupply after {settle:.0} s of settling, particles per second, holding each \
compound at zero  [{:.0} s wall]",
            began.elapsed().as_secs_f64()
        );
        report_harvests(&world.chem, &harvests);
        checkpoints.push(Checkpoint { settle, harvests });
    }

    if checkpoints.len() > 1 {
        report_sweep(&world.chem, &checkpoints);
    }

    // A reaction's sustainable turnover is set by its *scarcest* substrate, so
    // the livings table needs a measured rate for all of them. With `--only`
    // the rest are zero, which would not read as "not measured" -- it would
    // read as "resupplied at nothing", and the one reaction whose substrates
    // all happen to be the probed compound would be crowned on a rate the run
    // never took. Say nothing rather than that.
    if only.is_some() {
        println!(
            "\nno livings table: `--only` measured one substrate, and a living is bounded by \
its scarcest.\nrun without it to rank what this pond can feed."
        );
        println!("\n{:.1} s wall", began.elapsed().as_secs_f64());
        return Ok(());
    }

    // Everything below is read off the deepest settle. A shallower one is a
    // younger pond, not a wrong one, but a population has to live in the pond
    // that is left after the transients have run out.
    let harvests = &checkpoints[checkpoints.len() - 1].harvests;
    let rate: Vec<f64> = {
        let mut v = vec![0.0; world.chem.n_compounds()];
        for h in harvests {
            v[h.compound as usize] = h.sustained();
        }
        v
    };
    // Kept beside the rate rather than folded into it. Bounding the livings
    // below by the vents' inflow would be wrong in the opposite direction to
    // the error this column exists to catch: a *cell* exports nothing, so a
    // living can be sustained by matter that never enters the pond at all, as
    // long as photochemistry keeps driving the products back. The honest thing
    // is to rank on what was measured and say how much of it was stock.
    let paid: Vec<f64> = {
        let mut v = vec![0.0; world.chem.n_compounds()];
        for h in harvests {
            v[h.compound as usize] = h.provenance();
        }
        v
    };

    // What that buys, in cells. A reaction can only turn over as fast as its
    // scarcest substrate is resupplied, and a cell keeps `capture_efficiency`
    // of what a turnover releases and spends `maintenance_power` staying
    // alive. The quotient is a carrying capacity, and it is the number this
    // project has been missing: everything before it was measured in joules
    // standing in the water, which is a larder and not an income.
    let settled = world.abundance();
    let mut livings: Vec<(f64, f64, f64, u32, f64)> = world
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
                // A living is no better funded than its worst-funded substrate,
                // the same way it is no faster than its scarcest.
                let fed = r
                    .reactants
                    .iter()
                    .map(|&(c, _)| paid[c as usize])
                    .fold(f64::INFINITY, f64::min);
                (power, power / cells.maintenance_power, reach, r.id, fed)
            })
        })
        .collect();
    livings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!(
        "the livings this pond can sustain after {deepest:.0} s, at capture_efficiency {:.2} \
and maintenance_power {:.2e} W",
        cells.capture_efficiency, cells.maintenance_power
    );
    // Said before the table rather than after it. Every figure below is built
    // on a harvest, and a harvest exports matter where a cell does not, so
    // this table bounds what could be *taken out of* the pond and not what
    // could live in it. `returns` is the one that answers the second question.
    println!(
        "  (these are built on a harvest, which exports matter. A population does not -- run `hadean returns` for that bound.)"
    );
    // Two different bounds, and a living needs to clear both. `cells` is what
    // the *pond* can feed: total sustainable watts over one cell's upkeep. The
    // last column is what one cell can *reach*: a passive membrane holds its
    // own volume's share of what the water holds, so a dilute substrate is
    // dilute inside the cell too, however much of the pond is producing it.
    // The first column being large tells you nothing if the last one is below
    // one -- which on this chemistry is exactly the case that cost eight runs.
    println!(
        "  {:>11}  {:>11}  {:>10}  {:>8}  reaction",
        "sustains", "population", "per newborn", "vent-fed"
    );
    if livings.is_empty() {
        println!("  none: no exergonic thermal reaction has all of its substrates resupplied");
    }
    for &(power, capacity, reach, id, fed) in livings.iter().take(10) {
        println!(
            "  {:>9.3e} W  {:>7.0} cells  {:>8.3}x  {:>7.0}%  {}",
            power,
            capacity,
            reach,
            fed * 100.0,
            equation(&world.chem, id)
        );
    }

    println!();
    let reachable = livings.iter().find(|l| l.2 >= 1.0);
    match (livings.first(), reachable) {
        (Some(&(power, capacity, reach, id, _)), _) if reach >= 1.0 => {
            println!(
                "the best living here is {} at {:.3e} W, which keeps {:.0} cells alive, \
and a newborn earns {:.2}x its upkeep on it",
                equation(&world.chem, id),
                power,
                capacity,
                reach
            );
        }
        (Some(&(power, capacity, reach, id, _)), best) => {
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
                Some(&(p, c, reach, id, _)) => println!(
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

    // The line that stops this table being read as more than it is. The
    // vent-fed column can be zero across the board and the pond still feed a
    // population, because the two things are answers to different questions.
    if let Some(&(_, _, _, id, fed)) = livings.first() {
        if fed < 0.5 {
            println!();
            println!(
                "read that against the vent-fed column: only {:.0}% of the rate measured \
for {} was paid for by matter entering the pond, and the rest came out of standing stock.",
                fed * 100.0,
                equation(&world.chem, id)
            );
            println!(
                "that is a bound on the *probe*, which exports matter, and not on a population, \
which does not -- a cell turns its substrate into products and leaves every atom"
            );
            println!(
                "in the pond. What decides a population is whether the photochemistry can drive \
those products back to the substrate, and that is"
            );
            println!(
                "what `hadean returns` measures. It shares this command's `--settle-cache`, so \
on a config already settled here it is nearly free."
            );
        }
    }
    println!("\n{:.1} s wall", began.elapsed().as_secs_f64());
    Ok(())
}

/// A line per finished probe, because the alternative is silence.
///
/// A three-depth sweep at a 600 s probe is hours of wall clock, and the first
/// version of these commands printed nothing at all until a whole depth was
/// done -- thirty-five minutes in which a run that had died and a run that was
/// working looked exactly alike. The counter also says *which* probe finished,
/// which is how you find out that one candidate is taking as long as the other
/// twenty-two together.
///
/// Printed from whichever worker finished, so the order is arrival order and
/// not candidate order. `println!` takes the stdout lock per call, so the
/// lines do not interleave.
struct Progress {
    done: std::sync::atomic::AtomicUsize,
    total: usize,
    depth: f64,
    began: Instant,
}

impl Progress {
    fn new(total: usize, depth: f64) -> Self {
        Self {
            done: std::sync::atomic::AtomicUsize::new(0),
            total,
            depth,
            began: Instant::now(),
        }
    }

    fn tick(&self, what: &str) {
        let n = self
            .done
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        println!(
            "          {n:>3}/{} at {:.0} s, {:.0} s wall  {what}",
            self.total,
            self.depth,
            self.began.elapsed().as_secs_f64()
        );
    }
}

/// Settle one lifeless world through each requested depth, snapshotting at
/// every one.
///
/// A deep settle passes through every shallower one, so a sweep costs the
/// deepest settle rather than the sum of them. The cache turns the *repeat* of
/// that into nothing, which is what makes probing the same config again, or at
/// another depth, something you do rather than something you budget for: the
/// settle was 84% of a probe's wall clock and it is the same lifeless world
/// every time.
///
/// The world is left standing at the deepest depth, because everything read
/// off the pond itself -- concentrations, the element ledger -- should be read
/// there.
fn settle_depths(
    world: &mut World,
    depths: &[(f64, u64)],
    cache: Option<&PathBuf>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let digest = world.config.digest();
    let mut out = Vec::with_capacity(depths.len());
    for &(settle, ticks) in depths {
        let path = cache.map(|dir| dir.join(format!("settle-{digest:016x}-{ticks}.snap")));
        let reached = Instant::now();
        // Two different failures, and they deserve opposite treatment.
        //
        // A file this build cannot *read* is a file from an older build --
        // adding a field to the config changes every digest and bumps the
        // snapshot format -- and re-settling from scratch produces exactly the
        // right answer, so it says what it is doing and does it. Killing a
        // two-hour probe over a stale cache would be a papercut, and one that
        // teaches people not to use the cache.
        //
        // A file that reads fine but belongs to a *different config* is the
        // dangerous one: `load` rebuilds the world from the snapshot's own
        // config, not from the one on the command line, so using it would
        // quietly probe a different pond and report it under this config's
        // name. That bails.
        let restored = match &path {
            Some(p) if p.exists() => match snapshot::read_file(p) {
                Ok(w) => Some(w),
                Err(e) => {
                    println!("          {} is unreadable ({e}); settling again", p.display());
                    None
                }
            },
            _ => None,
        };
        let cached = match restored {
            Some(restored) => {
                let p = path.as_ref().expect("a cache hit came from a path");
                if restored.config.digest() != digest {
                    bail!(
                        "{} was settled from a different config than the one given",
                        p.display()
                    );
                }
                if restored.tick() != ticks {
                    bail!(
                        "{} holds tick {}, not the {ticks} its name claims",
                        p.display(),
                        restored.tick()
                    );
                }
                *world = restored;
                true
            }
            None => {
                while world.tick() < ticks {
                    world.step();
                }
                false
            }
        };
        let base = snapshot::save(world)?;
        // Written whenever this depth was settled rather than read, which
        // includes replacing a file an older build left behind.
        if let (Some(p), false) = (&path, cached) {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, &base).with_context(|| format!("writing {}", p.display()))?;
        }
        // The wall clock is this depth's own -- the time to get here from the
        // one before it -- and not the cumulative figure, because what a cache
        // is worth is exactly the difference between those two.
        println!(
            "settled   {settle:>6.0} s, {ticks} ticks{}, {:.1} s wall, snapshot {} KiB",
            if cached { ", read from cache" } else { "" },
            reached.elapsed().as_secs_f64(),
            base.len() / 1024
        );
        out.push(base);
    }
    Ok(out)
}

/// Parse the `--settle` list into ascending depths, deduplicated by tick.
///
/// Deduplicated by *tick* rather than by the seconds asked for: the depths are
/// built by stepping one world through all of them, so two settles that round
/// to the same tick are the same pond and probing it twice would measure only
/// the probe.
fn settle_depths_of(mut settles: Vec<f64>, fallback: f64, dt: f64) -> anyhow::Result<Vec<(f64, u64)>> {
    if settles.is_empty() {
        settles.push(fallback);
    }
    if settles.iter().any(|&s| s < 0.0) {
        bail!("a settle time cannot be negative");
    }
    settles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut depths: Vec<(f64, u64)> = settles
        .iter()
        .map(|&s| (s, (s / dt).round() as u64))
        .collect();
    depths.dedup_by_key(|&mut (_, ticks)| ticks);
    Ok(depths)
}

/// Hold each candidate at zero in its own copy of one settled world.
///
/// Each probe is restored from the same bytes, so nothing about the result
/// depends on `--jobs`, and nothing a probe does to its world can reach
/// another's.
fn probe_all(
    base: &[u8],
    candidates: &[u16],
    window_ticks: u64,
    seconds: f64,
    pool: &rayon::ThreadPool,
    progress: &Progress,
) -> anyhow::Result<Vec<Harvest>> {
    let harvests: anyhow::Result<Vec<Harvest>> = pool.install(|| {
        candidates
            .par_iter()
            .map(|&compound| {
                let mut w = snapshot::load(base)?;
                let formula = w.chem.compound(compound).formula;
                let standing = w.harvest(compound);
                let mut windows = [0.0f64; SUPPLY_WINDOWS];
                let mut vents = [[0.0f64; N_ELEMENTS]; SUPPLY_WINDOWS];
                for (window, vented) in windows.iter_mut().zip(vents.iter_mut()) {
                    let before = w.audit.elements_in;
                    for _ in 0..window_ticks {
                        w.step();
                        *window += w.harvest(compound);
                    }
                    *vented = vent_inflow(
                        &w.audit.elements_in,
                        &before,
                        *window,
                        &formula,
                        window_ticks,
                    );
                }
                progress.tick(&w.chem.compound(compound).name);
                Ok(Harvest {
                    compound,
                    standing,
                    windows,
                    vents,
                    formula,
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
    Ok(harvests)
}

/// The probe table for one settled pond.
fn report_harvests(chem: &hadean_chem::Chemistry, harvests: &[Harvest]) {
    println!(
        "  {:<10} {:>12}  {:>11}  {:>11}  {:>11}  {:>7}",
        "compound", "standing", "opening", "sustained", "funded", "holding"
    );
    for h in harvests {
        println!(
            "  {:<10} {:>12.3e}  {:>9.3e}/s  {:>9.3e}/s  {:>9.3e}/s  {:>7.3}  {}",
            chem.compound(h.compound).name,
            h.standing,
            h.opening(),
            h.sustained(),
            h.funded(SUPPLY_WINDOWS - 1),
            h.holding(),
            verdict(h),
        );
    }
}

/// One reading per compound per depth, ordered by the deepest probe's ranking.
///
/// The deepest one orders the table because it is the pond a population would
/// have to live in; the shallower columns are there to show what the pond used
/// to look like, which is what the shallow reading mistook for what it is.
fn sweep_rows(checkpoints: &[Checkpoint], value: impl Fn(&Harvest) -> f64) -> Vec<(u16, Vec<f64>)> {
    let last = checkpoints
        .last()
        .expect("a sweep is built from at least one depth");
    last.harvests
        .iter()
        .map(|h| {
            let series = checkpoints
                .iter()
                .map(|c| {
                    c.harvests
                        .iter()
                        .find(|x| x.compound == h.compound)
                        .map(&value)
                        .unwrap_or(0.0)
                })
                .collect();
            (h.compound, series)
        })
        .collect()
}

/// "the one metabolism" when there is one of it, and the plural phrasing when
/// there is not. A table of one is a common case here -- `--only` and
/// `--reaction` both produce it -- and "every one of the 1 metabolisms" reads
/// like a bug in the instrument, which is not a thing to make a reader wonder
/// about while they are deciding whether to trust its numbers.
fn plural(n: usize, singular: &str, many: &str) -> String {
    if n == 1 {
        format!("the one {singular}")
    } else {
        many.replace("{}", &n.to_string())
    }
}

/// Which way a reading moved across the sweep, coarsely.
///
/// Three buckets, because the summary line under a sweep table has to agree
/// with the words in it: the first version counted only transients and then
/// announced that everything "held its rate" on a table whose one row said
/// "climbing 12x".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drift {
    /// Ten times or more *higher* at the shallow read. The reading this
    /// whole apparatus exists to catch.
    Transient,
    /// Ten times or more higher at the deep read: a pond still filling when
    /// the shallow probe ran, which is not a transient read backwards.
    Climbed,
    Steady,
}

fn drift_bucket(first: f64, last: f64) -> Drift {
    if last <= 0.0 {
        // Gone by the deep read, or never there at all. Either way the shallow
        // number is not something to plan around.
        return if first > 0.0 {
            Drift::Transient
        } else {
            Drift::Steady
        };
    }
    if first <= 0.0 {
        return Drift::Climbed;
    }
    let factor = first / last;
    if factor >= 10.0 {
        Drift::Transient
    } else if factor <= 0.1 {
        Drift::Climbed
    } else {
        Drift::Steady
    }
}

/// What moving between the shallowest and deepest settle says about a reading.
///
/// Returns the factor as it should be printed, and the word for it.
fn drift_word(first: f64, last: f64) -> (String, String) {
    if first <= 0.0 && last <= 0.0 {
        return ("-".into(), "dead end at every depth".into());
    }
    if last <= 0.0 {
        return ("gone".into(), "collapsed to nothing by the deep read".into());
    }
    if first <= 0.0 {
        return ("new".into(), "nothing at the shallow read; arrives with depth".into());
    }
    // Ten is the line between the two verdicts that matter, and it is drawn
    // where it is because the reading this exists to catch missed by 1.9e5.
    // The two bands either side of "steady" are not accusations; they say the
    // pond was still moving when the shallow probe read it, which is a thing
    // to know before quoting either number.
    let factor = first / last;
    if factor >= 10.0 {
        (
            format!("{factor:.1e}x"),
            format!("transient - the shallow read is {factor:.1e} times the deep one"),
        )
    } else if factor >= 2.0 {
        (
            format!("{factor:.2}x"),
            format!("falling - {factor:.1}x less by the deep read"),
        )
    } else if factor >= 0.5 {
        (format!("{factor:.2}x"), "steady across the sweep".into())
    } else if factor > 0.1 {
        (
            format!("{factor:.2}x"),
            format!("climbing - {:.1}x more at depth", 1.0 / factor),
        )
    } else {
        (
            format!("{factor:.1e}x"),
            format!("climbing - {:.1e} times more at depth", 1.0 / factor),
        )
    }
}

/// How the readings moved between depths.
///
/// `--settle` defaults to `seed_delay`, which is the instant
/// `choose_metabolism` reads the pond at, so both instruments read the same
/// transient at the same moment and neither could see that it was one. On
/// `renew.toml` the shallow read gives HO2M 8.817e9/s and the deep one
/// 4.543e4/s, with nothing alive in the pond at either depth: this world
/// retires its own oxygen unaided.
///
/// A single reading cannot tell a supply from a stock drained at a steady
/// speed, because holding its rate across a window is what a steady speed
/// *is*. Two readings taken far apart can, and they do it without the element
/// ledger -- which makes them an independent check on it rather than a
/// restatement.
fn report_sweep(chem: &hadean_chem::Chemistry, checkpoints: &[Checkpoint]) {
    let head: String = checkpoints
        .iter()
        .map(|c| format!("{:>13}", format!("{:.0} s", c.settle)))
        .collect();
    let first = checkpoints[0].settle;
    let deep = checkpoints[checkpoints.len() - 1].settle;

    println!();
    println!("settle sweep: sustained resupply at each depth, particles per second");
    println!("  {:<10}{head}  {:>8}", "compound", "factor");
    let rates = sweep_rows(checkpoints, |h| h.sustained());
    for (compound, series) in &rates {
        let cells: String = series.iter().map(|v| format!("{v:>11.3e}/s")).collect();
        let (factor, word) = drift_word(series[0], series[series.len() - 1]);
        println!(
            "  {:<10}{cells}  {factor:>8}  {word}",
            chem.compound(*compound).name
        );
    }

    println!();
    println!("              standing stock at each depth, particles");
    println!("  {:<10}{head}  {:>8}", "compound", "factor");
    for (compound, series) in sweep_rows(checkpoints, |h| h.standing) {
        let cells: String = series.iter().map(|v| format!("{v:>13.3e}")).collect();
        let (factor, _) = drift_word(series[0], series[series.len() - 1]);
        println!(
            "  {:<10}{cells}  {factor:>8}",
            chem.compound(compound).name
        );
    }

    let bucket = |s: &Vec<f64>| drift_bucket(s[0], s[s.len() - 1]);
    let transient = rates
        .iter()
        .filter(|(_, s)| bucket(s) == Drift::Transient)
        .count();
    let climbed = rates
        .iter()
        .filter(|(_, s)| bucket(s) == Drift::Climbed)
        .count();
    println!();
    if transient > 0 {
        println!(
            "{transient} of {} compounds read more than ten times higher at {first:.0} s than \
at {deep:.0} s, in a pond with nothing living in it.",
            rates.len()
        );
        println!(
            "a rate that falls while nothing is eating it was never a rate. Read the deep \
column; the shallow one is what the pond used to have."
        );
    } else if climbed > 0 {
        println!(
            "nothing fell across the sweep, but {climbed} of {} compounds came back more than \
ten times faster at {deep:.0} s than at {first:.0} s:",
            rates.len()
        );
        println!(
            "at {first:.0} s this pond was still filling, and a rate read there is a floor \
rather than a transient."
        );
    } else {
        println!(
            "{} held its rate from {first:.0} s to {deep:.0} s.",
            plural(rates.len(), "compound", "every one of the {} compounds")
        );
    }
}

/// What running one metabolism in the water, with no cells in it, took out of
/// the pond.
struct Living {
    reaction: u32,
    /// Turnovers the water paid for in each window, in order.
    windows: [f64; SUPPLY_WINDOWS],
    /// Chemical energy the *forward leg* took out of the water in each
    /// window, J. Measured from the amounts actually moved rather than from
    /// `turnovers * -dh`, so a clamp or an `f32` rounding is in it rather than
    /// assumed away.
    ///
    /// This is a gross figure and on its own it is not a living. See `net`.
    released: [f64; SUPPLY_WINDOWS],
    /// Chemical energy the *pond* gave up in each window, J: how much further
    /// its chemical total fell than the same pond's did with no metabolism
    /// running in it.
    ///
    /// The difference between this and `released` is the whole hazard of this
    /// probe. If the water puts the substrate back through the reverse of the
    /// reaction being driven, the forward leg's heat is what pays for it: the
    /// chemical energy goes down and comes straight back up, `released` counts
    /// every joule of it, and a cell sitting in that loop would be running a
    /// heat engine off an ambient bath. `net` is zero there, because nothing
    /// was actually consumed.
    net: [f64; SUPPLY_WINDOWS],
    /// Turnovers the standing stock paid for on the first sweep. Thrown away
    /// and not counted, for the same reason `Harvest::standing` is: it is the
    /// larder, and mistaking it for a living is the error both probes exist to
    /// stop.
    standing: f64,
    /// The control pond's own window-to-window spread in chemical energy, J.
    ///
    /// `net` is a difference against a control, and a control that is still
    /// settling moves on its own. Anything smaller than how much the control
    /// moved between its own windows is not a measurement of the metabolism;
    /// it is the two worlds drifting apart. Copied into every probe because it
    /// is a property of the depth, not of the reaction.
    floor: f64,
    seconds: f64,
}

impl Living {
    fn window_seconds(&self) -> f64 {
        self.seconds / SUPPLY_WINDOWS as f64
    }

    fn rate(&self, w: usize) -> f64 {
        self.windows[w] / self.window_seconds()
    }

    fn opening(&self) -> f64 {
        self.rate(0)
    }

    /// What the pond still paid for at the end, which is the only turnover a
    /// population gets to plan around.
    fn sustained(&self) -> f64 {
        self.rate(SUPPLY_WINDOWS - 1)
    }

    fn holding(&self) -> f64 {
        let opening = self.opening();
        if opening > 0.0 {
            self.sustained() / opening
        } else {
            0.0
        }
    }

    /// Watts a population could take off this, at the given capture
    /// efficiency: what the pond actually gave up in the last window, over
    /// that window, times the share a cell keeps.
    ///
    /// **Signed, and deliberately not clamped at zero.** A metabolism can
    /// leave the pond holding *more* chemical energy than the control does --
    /// consuming a compound can unblock a photochemical route that stores more
    /// than the reaction released -- and that is a real reading about this
    /// world, not a small positive one. Clamping it would print the most
    /// interesting rows in the table as zeroes.
    fn power(&self, capture: f64) -> f64 {
        self.net[SUPPLY_WINDOWS - 1] / self.window_seconds() * capture
    }

    /// The same on the forward leg alone, which is what the probe moved rather
    /// than what the pond paid for. Printed beside `power` because their ratio
    /// is the reading: gross far above net is a futile cycle.
    fn gross(&self, capture: f64) -> f64 {
        self.released[SUPPLY_WINDOWS - 1] / self.window_seconds() * capture
    }

    /// Is the net figure bigger than the control's own drift?
    ///
    /// A false here does not mean the living is worth nothing. It means this
    /// probe could not tell, and saying so is the difference between an
    /// instrument and a number generator.
    fn resolved(&self) -> bool {
        self.net[SUPPLY_WINDOWS - 1].abs() > self.floor
    }

    /// How many seconds of the sustained turnover the standing stock was
    /// worth, printed rather than the stock itself.
    ///
    /// This is the larder measured in the only unit that makes it comparable
    /// to an income. A pond holding a hundred thousand seconds of its own
    /// resupply will feed a boom that looks like a carrying capacity for as
    /// long as anyone watches, and then stop; every wrong diet this project
    /// has chosen was chosen off a number that this column would have made
    /// obvious. `None` when there is no income to divide by, which is the
    /// worst case rather than a missing reading.
    fn larder_seconds(&self) -> Option<f64> {
        let sustained = self.sustained();
        (sustained > 0.0).then(|| self.standing / sustained)
    }
}

/// Measure what a population could actually live on, by running its metabolism
/// in the water with no cells in it.
///
/// `hadean supply` asks how fast the pond replaces a compound *removed from
/// the world*, and no organism ever asks that. A probe exports matter; a cell
/// does not. A cell turns its substrate into products and leaves every atom in
/// the pond, so what bounds a population is not whether the vents can deliver
/// the atoms -- they are already here -- but whether the light and the
/// chemistry can drive the products back round to the substrate. That is the
/// return leg, and until this command existed no number in this project
/// bounded a population's food supply. `supply`'s `funded` column bounds the
/// probe, and says so in its own output.
///
/// Same machinery, different perturbation. `supply` holds a compound at zero
/// by taking it away; this holds it at zero by *eating* it, which is the thing
/// a population does, and puts the products where a population would put them.
/// It is an upper bound and deliberately so: the water is the only limit here,
/// where a real cell is also limited by its membrane and its enzymes. A living
/// that does not appear in this table is a living no cell of any design could
/// make in this pond.
#[allow(clippy::too_many_arguments)]
fn returns(
    config: Option<PathBuf>,
    settle: Vec<f64>,
    probe: f64,
    reaction: Option<u32>,
    metabolism: Option<String>,
    limit: usize,
    jobs: usize,
    settle_cache: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut cfg = load_config(config)?;
    let dt = cfg.dt;
    if probe <= 0.0 {
        bail!("probe must be positive");
    }
    let depths = settle_depths_of(settle, cfg.cells.seed_delay as f64, dt)?;

    // Same reason as `supply`: leave the cells in and the reading is the
    // pond's production minus whatever they were eating.
    let cells = cfg.cells.clone();
    cfg.cells.enabled = false;

    let mut world = World::new(cfg)?;
    describe(&world);
    println!("probe     run one metabolism in the water for {probe:.0} s, in {SUPPLY_WINDOWS} windows");
    println!(
        "settles   {} -- the same pond, probed at each depth",
        depths
            .iter()
            .map(|(s, _)| format!("{s:.0} s"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if cells.enabled {
        println!("          cells are switched off for the probe; the pond is measured empty");
    }

    let mut candidates: Vec<u32> = world
        .chem
        .reactions
        .iter()
        .filter(|r| r.drive == Drive::Thermal && r.dh < 0.0)
        .map(|r| r.id)
        .collect();
    // A named metabolism resolves through the same parser the config field
    // uses, so `--metabolism "HO2M + HM"` and `cells.metabolism = "HO2M + HM"`
    // cannot disagree about what they mean.
    let want = match (&metabolism, reaction) {
        (Some(spec), _) => Some(
            hadean_cell::named_metabolism(&world.chem, spec)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("resolving metabolism {spec:?}"))?,
        ),
        (None, r) => r,
    };
    if let Some(want) = want {
        // Range-checked before anything tries to name it. `Chemistry::reaction`
        // indexes, so `--reaction 9999` on an 88-reaction chemistry would
        // panic inside the error message being built to explain the mistake.
        if want as usize >= world.chem.n_reactions() {
            bail!(
                "there is no reaction {want} in this chemistry; it has {} \
(`hadean chem --reactions` lists them)",
                world.chem.n_reactions()
            );
        }
        candidates.retain(|&r| r == want);
        if candidates.is_empty() {
            bail!(
                "{} is not an exergonic thermal reaction in this chemistry, so nothing \
could live on it",
                equation(&world.chem, want)
            );
        }
    }
    println!(
        "          {} metabolisms to probe, at {:.0} s of world time each",
        candidates.len(),
        probe
    );

    let began = Instant::now();
    let window_ticks = ((probe / dt).round() as u64 / SUPPLY_WINDOWS as u64).max(1);
    let seconds = (window_ticks * SUPPLY_WINDOWS as u64) as f64 * dt;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()?;

    let settled = settle_depths(&mut world, &depths, settle_cache.as_ref())?;

    println!();
    println!(
        "there is no vent-fed column here and there should not be. A turnover moves no matter \
across the world's"
    );
    println!(
        "boundary -- every atom that leaves the substrate arrives in the products, in the same \
voxel -- so the element"
    );
    println!(
        "ledger has nothing to say about it. What bounds this reading is whether the light and \
the chemistry can drive"
    );
    println!("those products back, and that is the thing it measures.");

    let deepest = depths.last().map(|&(s, _)| s).unwrap_or(0.0);
    // The energy budget, which bounds this table the way the element ledger
    // bounds `supply`'s -- and for a sharper reason. This probe drives the
    // reaction forward and lets the water put the substrate back; if the water
    // does that through the *reverse of the same reaction*, the loop is a
    // futile cycle running on the heat the forward leg just deposited, and a
    // cell sitting in it would be extracting work from an ambient thermal
    // bath. The simulation's audit stays flat through that -- nothing is
    // created, the heat merely goes round -- so the audit cannot catch it and
    // this line has to.
    //
    // Sunlight is the pond's one unambiguous source of low-entropy energy.
    // Vent chemistry is the other, and its free energy is not a number this
    // ledger holds, so the comparison is stated rather than enforced.
    let sunlight = if deepest > 0.0 {
        world.audit.ledger.light_in / deepest
    } else {
        0.0
    };
    println!();
    println!(
        "budget    the sun puts {sunlight:.3e} W into this pond, averaged over the settle, and \
no cell can take more free"
    );
    println!(
        "          energy out of this world than enters it. The `sustains` column is measured \
against a control pond with"
    );
    println!(
        "          nothing driven in it, so a metabolism whose substrate comes back round the \
reverse of its own reaction --"
    );
    println!(
        "          a futile cycle on the forward leg's own heat, which the energy audit passes \
because nothing is created --"
    );
    println!(
        "          reads a large `gross` and a `sustains` of nothing. Vent chemistry is the \
other source of free energy"
    );
    println!(
        "          here and this ledger holds only the enthalpy its matter carried, not its \
free energy, so the sun is a"
    );
    println!("          check and not the whole budget.");
    let abundance = world.abundance();
    // Probed and printed one depth at a time, for the same reason `supply`
    // does it: a sweep at a long probe is hours, and silence for hours is
    // indistinguishable from a hang.
    let mut rounds: Vec<(f64, Vec<Living>)> = Vec::new();
    for (&(depth, _), base) in depths.iter().zip(&settled) {
        let livings = drive_all(
            base,
            &candidates,
            window_ticks,
            seconds,
            &pool,
            // Plus one for the control pond, which is a probe's worth of work
            // and should be visible as such.
            &Progress::new(candidates.len() + 1, depth),
        )?;
        println!();
        println!(
            "what the pond keeps paying for after {depth:.0} s of settling, at \
capture_efficiency {:.2} and maintenance_power {:.2e} W  [{:.0} s wall]",
            cells.capture_efficiency,
            cells.maintenance_power,
            began.elapsed().as_secs_f64()
        );
        // `opening` is not a column: `holding` is sustained over opening, so
        // the pair already carries it, and the width is better spent on the
        // two that decide the table. `larder s` is what the standing stock was
        // worth in seconds of the income that replaces it, which is the
        // reading that would have caught both of this project's wrong diets on
        // sight. `gross` is what the probe's forward leg moved and `sustains`
        // is what the pond was actually left short of, measured against a
        // control world with no metabolism running in it -- gross far above
        // net means the substrate came straight back round the reverse of the
        // same reaction on the forward leg's own heat, and a cell living in
        // that loop would be a heat engine running off an ambient bath.
        //
        // Rows are ordered by `sustains`, not by turnover, for that reason.
        println!(
            "  {:>11}  {:>7}  {:>9}  {:>11}  {:>11}  {:>11}  {:>10}  reaction",
            "sustained", "holding", "larder s", "gross", "sustains", "population", "per newborn"
        );
        let floor = livings
            .first()
            .map(|l| l.floor / l.window_seconds() * cells.capture_efficiency as f64)
            .unwrap_or(0.0);
        println!(
            "  (a net reading below the control pond's own drift of {floor:.3e} W reads \
`unresolved`; the probe could not tell)"
        );
        // A pond still relaxing drifts harder than anything living in it could
        // ever consume, and then no row can resolve however long the probe.
        // Saying "unresolved" twenty-three times without saying why would let
        // a reader conclude the pond is empty.
        if sunlight > 0.0 && floor > sunlight {
            println!(
                "  the control drifts harder than the {sunlight:.3e} W of sunlight entering \
this pond, so nothing here can resolve:"
            );
            println!(
                "  this depth is still settling. Probe deeper -- the drift fell by fifty times \
between 20 s and 60 s on renew.toml."
            );
        }
        if livings.is_empty() {
            println!("  none: no exergonic thermal reaction turns over at all in this water");
        }
        for l in livings.iter().take(limit) {
            let power = l.power(cells.capture_efficiency as f64);
            let reach = hadean_cell::subsistence(
                &cells,
                &world.chem,
                l.reaction,
                &abundance,
                &world.grid,
            );
            println!(
                "  {:>9.3e}/s  {:>7.3}  {:>9}  {:>9.3e} W  {:>11}  {:>11}  {:>9.3}x  {}",
                l.sustained(),
                l.holding(),
                match l.larder_seconds() {
                    Some(s) => format!("{s:.2e}"),
                    None => "-".to_string(),
                },
                l.gross(cells.capture_efficiency as f64),
                // A net figure inside the control's own drift is not a small
                // living, it is no reading at all, and printing it as a number
                // would invite exactly the arithmetic this project keeps
                // getting caught by.
                if l.resolved() {
                    format!("{power:.3e} W")
                } else {
                    "unresolved".to_string()
                },
                // A negative net is a pond left richer than the control, which
                // is a finding and not a population. Only a positive one
                // divides into a head count.
                if l.resolved() && power > 0.0 {
                    format!("{:.0} cells", power / cells.maintenance_power)
                } else {
                    "-".to_string()
                },
                reach,
                equation(&world.chem, l.reaction)
            );
        }
        rounds.push((depth, livings));
    }

    // `per newborn` is read at the deepest settle for every table above,
    // because `subsistence` reads the pond's concentrations and the world is
    // left standing at the deepest depth. Say so rather than let a reader
    // assume each row was measured where its turnovers were.
    if rounds.len() > 1 {
        println!();
        println!(
            "note: `per newborn` is the same in every table -- it reads the water's \
concentrations, and those are the deepest settle's."
        );
        report_return_sweep(&world.chem, &rounds);
    }

    println!();
    // The best living is the one the pond actually pays for, not the one it
    // shuffles fastest. Ranking on turnover here would crown a futile cycle,
    // which is the whole thing `net` was added to prevent.
    let best = rounds.last().and_then(|(_, l)| {
        l.iter()
            .filter(|x| {
                x.sustained() > 0.0 && x.resolved() && x.power(cells.capture_efficiency as f64) > 0.0
            })
            .max_by(|a, b| {
                a.power(cells.capture_efficiency as f64)
                    .partial_cmp(&b.power(cells.capture_efficiency as f64))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    // "Could not tell" and "is nothing" are different findings and this
    // project has been caught before by writing one and reading the other.
    let any_resolved = rounds
        .last()
        .map(|(_, l)| l.iter().any(|x| x.resolved()))
        .unwrap_or(false);
    match best {
        None if !any_resolved => {
            println!(
                "after {deepest:.0} s not one metabolism's net energy came out above the \
control pond's own drift, so this run does not say"
            );
            println!(
                "what the pond can feed -- it says the probe could not tell. A longer \
`--probe` narrows the control's drift and is the thing to try."
            );
        }
        None => println!(
            "after {deepest:.0} s no metabolism leaves this pond short of any net chemical \
energy. What turns over here turns over in a circle."
        ),
        Some(l) => {
            let power = l.power(cells.capture_efficiency as f64);
            let reach =
                hadean_cell::subsistence(&cells, &world.chem, l.reaction, &abundance, &world.grid);
            println!(
                "the best living the pond will keep paying for is {} at {:.3e} W, which is \
{:.0} cells, and a newborn earns {:.3}x its upkeep on it",
                equation(&world.chem, l.reaction),
                power,
                power / cells.maintenance_power,
                reach
            );
            if reach < 1.0 {
                println!(
                    "the pond produces it and no cell of this design can reach it. That is a \
cell-layer problem, not a chemistry one."
                );
            }
            if l.holding() < 0.5 {
                println!(
                    "and read the holding column before believing it: this one delivered {:.0}% \
of its opening rate by the last window, so it was still running down.",
                    l.holding() * 100.0
                );
            }
            let gross = l.gross(cells.capture_efficiency as f64);
            if gross > 10.0 * power.max(f64::MIN_POSITIVE) {
                println!(
                    "note the two power columns: the forward leg moved {gross:.3e} W and the \
pond gave up {power:.3e} W of it, so {:.0}% of what",
                    (1.0 - power / gross) * 100.0
                );
                println!(
                    "the probe turned over came straight back round the reverse of the same \
reaction, on the heat the forward leg deposited."
                );
            }
            if power < 0.0 {
                println!(
                    "and note the sign: running this metabolism leaves the pond holding {:.3e} W \
more chemical energy than one with nothing",
                    -power
                );
                println!(
                    "running in it. Consuming the substrate is unblocking something that stores \
more than the reaction releases -- worth chasing, and not a living."
                );
            }
            if sunlight > 0.0 && power > sunlight {
                println!(
                    "DISCARD IT. {power:.3e} W is {:.1e} times the {sunlight:.3e} W of sunlight \
entering this pond, and the vents",
                    power / sunlight
                );
                println!(
                    "do not close that gap. Net or not, no cell can take more free energy out of \
this world than enters it."
                );
            }
        }
    }
    println!("\n{:.1} s wall", began.elapsed().as_secs_f64());
    Ok(())
}

/// One probe's raw readings, before the control pond is subtracted from them.
struct Driven {
    reaction: u32,
    windows: [f64; SUPPLY_WINDOWS],
    released: [f64; SUPPLY_WINDOWS],
    /// How far the pond's chemical energy fell in each window. Becomes
    /// `Living::net` once the control's own fall is taken off it.
    fell: [f64; SUPPLY_WINDOWS],
    standing: f64,
}

/// Run each metabolism in its own copy of one settled world.
///
/// A control world is run first, from the same bytes and for the same windows,
/// with nothing driven in it. Its chemical energy over each window is the
/// baseline every probe's is measured against, and that subtraction is what
/// turns a gross forward-leg figure into a net one -- see `Living::net`.
///
/// The subtraction is not exact and cannot be. A probe heats its own water,
/// heat changes the rate constants, and the control's chemistry therefore
/// evolves down a slightly different path from the probe's. What it does do is
/// separate a pond that gives up chemical energy from one whose energy is
/// going round in a circle, and those differ by orders of magnitude rather
/// than by the size of that error.
fn drive_all(
    base: &[u8],
    reactions: &[u32],
    window_ticks: u64,
    seconds: f64,
    pool: &rayon::ThreadPool,
    progress: &Progress,
) -> anyhow::Result<Vec<Living>> {
    let chemical_over_windows = |w: &mut World| -> [f64; SUPPLY_WINDOWS] {
        let mut out = [0.0f64; SUPPLY_WINDOWS];
        let mut before = w.audit_now().energy.chemical;
        for fall in out.iter_mut() {
            for _ in 0..window_ticks {
                w.step();
            }
            let after = w.audit_now().energy.chemical;
            *fall = before - after;
            before = after;
        }
        out
    };

    // The control is one more world of the same length as a probe, so running
    // it before the probes would put a probe's worth of wall clock on one
    // thread with the other nineteen idle. It goes in the pool beside them and
    // the subtraction happens afterwards.
    let (control, livings) = pool.install(|| {
        rayon::join(
            || -> anyhow::Result<[f64; SUPPLY_WINDOWS]> {
                let mut w = snapshot::load(base)?;
                let out = chemical_over_windows(&mut w);
                progress.tick("the control pond");
                Ok(out)
            },
            || -> anyhow::Result<Vec<Driven>> {
                reactions
                    .par_iter()
                    .map(|&reaction| {
                        let mut w = snapshot::load(base)?;
                        // The first sweep clears whatever was standing in
                        // the water, the way `harvest` does before a probe's
                        // first window. What it pays for is the larder and is
                        // not counted.
                        let standing = w.turn_over(reaction).turnovers;
                        let mut windows = [0.0f64; SUPPLY_WINDOWS];
                        let mut released = [0.0f64; SUPPLY_WINDOWS];
                        let mut fell = [0.0f64; SUPPLY_WINDOWS];
                        let mut before = w.audit_now().energy.chemical;
                        for ((window, energy), fall) in windows
                            .iter_mut()
                            .zip(released.iter_mut())
                            .zip(fell.iter_mut())
                        {
                            for _ in 0..window_ticks {
                                w.step();
                                let out = w.turn_over(reaction);
                                *window += out.turnovers;
                                *energy += out.released;
                            }
                            let after = w.audit_now().energy.chemical;
                            *fall = before - after;
                            before = after;
                        }
                        progress.tick(&equation(&w.chem, reaction));
                        Ok(Driven {
                            reaction,
                            windows,
                            released,
                            fell,
                            standing,
                        })
                    })
                    .collect()
            },
        )
    });
    let control = control?;
    // How much the control moved between its own windows. A net figure below
    // this is two worlds drifting apart, not a metabolism.
    let floor = control.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - control.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut livings: Vec<Living> = livings?
        .into_iter()
        .map(|d| {
            let mut net = d.fell;
            for (fall, baseline) in net.iter_mut().zip(control.iter()) {
                *fall -= baseline;
            }
            Living {
                reaction: d.reaction,
                windows: d.windows,
                released: d.released,
                net,
                standing: d.standing,
                floor,
                seconds,
            }
        })
        .collect();
    // Ordered by what the pond was left short of, with anything the control's
    // own drift swallowed pushed below everything it did not. Ordering by
    // turnover would put a futile cycle on top, which is the thing the net
    // column exists to demote.
    livings.sort_by(|a, b| {
        let key = |l: &Living| {
            if l.resolved() {
                (1, l.net[SUPPLY_WINDOWS - 1])
            } else {
                (0, l.sustained())
            }
        };
        let (ra, va) = key(a);
        let (rb, vb) = key(b);
        rb.cmp(&ra)
            .then_with(|| vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(livings)
}

/// How each metabolism's turnover moved between settle depths.
///
/// The same reading, and the same reason, as the sweep in `supply`: a pond
/// still on its way somewhere gives a rate that is not the rate it will give,
/// and one probe at one depth cannot tell those apart.
fn report_return_sweep(chem: &hadean_chem::Chemistry, rounds: &[(f64, Vec<Living>)]) {
    let head: String = rounds
        .iter()
        .map(|(depth, _)| format!("{:>13}", format!("{depth:.0} s")))
        .collect();
    let first = rounds[0].0;
    let deep = rounds[rounds.len() - 1].0;
    let last = &rounds[rounds.len() - 1].1;

    println!();
    println!("settle sweep: sustained turnover at each depth, per second");
    // The equation goes last. An equation is not a fixed width and putting it
    // first would shove every number in the row sideways by however long it
    // happened to be.
    println!("{head}  {:>8}  {:<40}  reaction", "factor", "");
    let mut transient = 0;
    let mut climbed = 0;
    for l in last {
        let series: Vec<f64> = rounds
            .iter()
            .map(|(_, round)| {
                round
                    .iter()
                    .find(|x| x.reaction == l.reaction)
                    .map(|x| x.sustained())
                    .unwrap_or(0.0)
            })
            .collect();
        let cells: String = series.iter().map(|v| format!("{v:>11.3e}/s")).collect();
        let (factor, word) = drift_word(series[0], series[series.len() - 1]);
        match drift_bucket(series[0], series[series.len() - 1]) {
            Drift::Transient => transient += 1,
            Drift::Climbed => climbed += 1,
            Drift::Steady => {}
        }
        println!(
            "{cells}  {factor:>8}  {:<40}  {}",
            word,
            equation(chem, l.reaction)
        );
    }
    println!();
    if transient > 0 {
        println!(
            "{transient} of {} metabolisms turned over more than ten times faster at {first:.0} s \
than at {deep:.0} s, with nothing living in the pond.",
            last.len()
        );
        println!(
            "read the deep column. The shallow one is a pond still on its way somewhere, and it \
is the column both of this project's wrong diets were chosen off."
        );
    } else if climbed > 0 {
        println!(
            "nothing fell across the sweep, but {climbed} of {} metabolisms turned over more \
than ten times faster at {deep:.0} s than at {first:.0} s:",
            last.len()
        );
        println!(
            "at {first:.0} s this pond had not finished making their substrates, so a turnover \
read there is a floor rather than a transient."
        );
    } else {
        println!(
            "{} held its turnover from {first:.0} s to {deep:.0} s.",
            plural(last.len(), "metabolism", "every one of the {} metabolisms")
        );
    }
}

/// What the vents put into the pond during one window, element by element.
///
/// `Audit::elements_in` nets both boundary crossings -- `vent_matter` adds and
/// `harvest` subtracts -- so the vents' share is what the ledger gained plus
/// what the probe took back out.
///
/// The subtraction is done at the magnitude of every atom of that element that
/// has ever crossed the boundary, which once the standing stock has been
/// harvested is the size of the pond's whole inventory of it. Each tick's
/// booking rounds at an ulp of *that*, and a window carries up to `ticks` of
/// them. So a reading below that floor is not a small inflow, it is the
/// arithmetic's own noise, and it is reported as the zero it is: measured
/// against a pond holding 1.15e15 atoms of oxygen, the residue came out at
/// -1.6e-1 atoms a second, and "no oxygen enters this pond" and "oxygen enters
/// this pond at minus a sixth of an atom a second" are the same fact with only
/// one of them readable as one.
fn vent_inflow(
    after: &[f64; N_ELEMENTS],
    before: &[f64; N_ELEMENTS],
    harvested: f64,
    formula: &[u16; N_ELEMENTS],
    ticks: u64,
) -> [f64; N_ELEMENTS] {
    let mut out = [0.0; N_ELEMENTS];
    for e in 0..N_ELEMENTS {
        let raw = after[e] - before[e] + harvested * formula[e] as f64;
        let floor = ticks as f64 * f64::EPSILON * after[e].abs().max(before[e].abs());
        out[e] = if raw > floor { raw } else { 0.0 };
    }
    out
}

/// What the vents actually deliver, element by element.
///
/// Every atom in this pond either started here or came out of a vent, and a
/// settled lifeless world has had nothing else happen to it, so the audit's
/// element ledger over the settle *is* the world's matter budget. It is
/// printed before the table because it decides most of the table in advance:
/// a harvest of a compound built from an element with no inflow is a stock
/// being emptied, however flat its curve looks over a window.
fn report_inflow(world: &World, settle: f64) {
    if settle <= 0.0 {
        return;
    }
    let symbol = |e: usize| element::element(e as u8).symbol;
    let fed: Vec<String> = (0..N_ELEMENTS)
        .filter(|&e| world.audit.elements_in[e] > 0.0)
        .map(|e| format!("{} {:.3e}", symbol(e), world.audit.elements_in[e] / settle))
        .collect();
    let dry: Vec<&str> = (0..N_ELEMENTS)
        .filter(|&e| world.audit.elements_in[e] <= 0.0)
        .map(symbol)
        .collect();
    if fed.is_empty() {
        println!("inflow    the vents deliver no matter at all: every harvest below is stock");
        return;
    }
    println!(
        "inflow    the vents deliver {} -- atoms per second, off the audit's element ledger",
        fed.join(", ")
    );
    if !dry.is_empty() {
        println!(
            "          nothing brings {} into this pond, so a harvest of anything containing \
one of them is stock",
            dry.join(", ")
        );
    }
}

/// A word for what a probe found, so the table can be read without doing
/// arithmetic in your head.
fn verdict(h: &Harvest) -> String {
    if h.sustained() <= 0.0 {
        return "dead end - nothing comes back".to_string();
    }
    // Provenance is asked before the shape of the curve, because it outranks
    // it. A stock drained at a steady speed holds its rate perfectly across a
    // window -- holding its rate is what a steady speed *is* -- so `holding`
    // near one is exactly as consistent with a larder as with a supply, and on
    // this chemistry it endorsed one at 1.231. The element ledger is not a
    // matter of how long the window was.
    let starved = h.starved().map(|e| element::element(e).symbol);
    if h.funded(SUPPLY_WINDOWS - 1) <= 0.0 {
        return match starved {
            Some(e) => format!("stock - no {e} enters this pond"),
            None => "stock - nothing enters this pond".to_string(),
        };
    }
    let paid = h.provenance();
    if paid < 0.5 {
        return match starved {
            Some(e) => format!("part stock - {e} inflow pays {:.0}% of it", paid * 100.0),
            None => format!("part stock - inflow pays {:.0}% of it", paid * 100.0),
        };
    }
    if h.holding() >= 0.5 {
        "renewed".to_string()
    } else if h.holding() >= 0.05 {
        "draining".to_string()
    } else if h.standing / h.sustained() > 1.0e5 {
        "larder - a big stock on a trickle".to_string()
    } else {
        "draining".to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    use hadean_chem::element::{H, M, O};

    /// A probe result with the numbers set by hand, so the arithmetic can be
    /// checked without settling a world.
    fn probe(
        formula: [u16; N_ELEMENTS],
        per_window: f64,
        vent: [f64; N_ELEMENTS],
    ) -> Harvest {
        Harvest {
            compound: 0,
            standing: 0.0,
            windows: [per_window; SUPPLY_WINDOWS],
            vents: [vent; SUPPLY_WINDOWS],
            formula,
            // One second per window, so a window's particles are its rate.
            seconds: SUPPLY_WINDOWS as f64,
        }
    }

    /// HO2M, in a pond whose vents inject HM and H2 -- which is `renew.toml`.
    fn ho2m(vent: [f64; N_ELEMENTS]) -> Harvest {
        let mut formula = [0u16; N_ELEMENTS];
        formula[H as usize] = 1;
        formula[O as usize] = 2;
        formula[M as usize] = 1;
        probe(formula, 100.0, vent)
    }

    fn vents(h: f64, o: f64, m: f64) -> [f64; N_ELEMENTS] {
        let mut v = [0.0; N_ELEMENTS];
        v[H as usize] = h;
        v[O as usize] = o;
        v[M as usize] = m;
        v
    }

    #[test]
    fn an_element_with_no_inflow_funds_nothing() {
        let h = ho2m(vents(1.0e3, 0.0, 1.0e3));
        assert_eq!(h.sustained(), 100.0);
        assert_eq!(h.funded(SUPPLY_WINDOWS - 1), 0.0);
        assert_eq!(h.provenance(), 0.0);
        assert_eq!(h.starved(), Some(O));
    }

    /// The bug this column was built for. A stock drained at a constant speed
    /// holds its rate across every window, so `holding` reads a perfect 1.0 --
    /// which is what endorsed HO2M at 8.512e9/s on a pond with no oxygen
    /// coming into it. Provenance is the reading that does not care how long
    /// the window was.
    #[test]
    fn a_steady_drain_holds_its_rate_and_is_still_a_stock() {
        let h = ho2m(vents(1.0e3, 0.0, 1.0e3));
        assert_eq!(h.holding(), 1.0);
        assert!(verdict(&h).starts_with("stock - no O"));
    }

    /// The bound is generous on purpose: every atom the vents delivered is
    /// credited to this one compound, as though nothing else wanted any.
    #[test]
    fn funded_credits_the_whole_inflow_to_one_compound() {
        // 100 particles/s of HO2M needs 200 O atoms/s. Half that inflow funds
        // half the harvest, and no more, whatever the other elements do.
        let h = ho2m(vents(1.0e9, 100.0, 1.0e9));
        assert_eq!(h.funded(SUPPLY_WINDOWS - 1), 50.0);
        assert_eq!(h.provenance(), 0.5);
        assert_eq!(h.starved(), Some(O));
    }

    #[test]
    fn a_vent_fed_compound_is_fully_funded() {
        let mut formula = [0u16; N_ELEMENTS];
        formula[H as usize] = 1;
        formula[M as usize] = 1;
        let h = probe(formula, 100.0, vents(100.0, 0.0, 100.0));
        assert_eq!(h.funded(SUPPLY_WINDOWS - 1), 100.0);
        assert_eq!(h.provenance(), 1.0);
        assert_eq!(verdict(&h), "renewed");
    }

    /// The vents put in 1.2e10 H and 1.2e10 M a second and no oxygen at all;
    /// the probe took 100 particles of HO2M back out. The ledger has to hand
    /// back exactly that, with the oxygen at zero.
    #[test]
    fn vent_inflow_reads_the_ledger_back() {
        let mut formula = [0u16; N_ELEMENTS];
        formula[H as usize] = 1;
        formula[O as usize] = 2;
        formula[M as usize] = 1;
        let before = [0.0; N_ELEMENTS];
        let mut after = [0.0; N_ELEMENTS];
        after[H as usize] = 1.2e10 - 100.0;
        after[O as usize] = -200.0;
        after[M as usize] = 1.2e10 - 100.0;
        let v = vent_inflow(&after, &before, 100.0, &formula, 1200);
        assert_eq!(v[H as usize], 1.2e10);
        assert_eq!(v[O as usize], 0.0);
        assert_eq!(v[M as usize], 1.2e10);
    }

    /// The defect the first live run exposed. Differencing the ledger across a
    /// window happens at the magnitude of the pond's whole inventory, so an
    /// element with no inflow lands a fraction of an atom either side of zero
    /// -- and a negative "largest harvest the vents could pay for" is not a
    /// small number, it is a wrong one. Below the floor the answer is zero.
    #[test]
    fn ledger_rounding_is_not_reported_as_inflow() {
        let mut formula = [0u16; N_ELEMENTS];
        formula[H as usize] = 2;
        formula[O as usize] = 1;
        // A pond holding 1.15e15 oxygen atoms, all of it already harvested, so
        // the window's delta is differenced against that magnitude.
        let harvested = 5.0e5;
        let mut before = [0.0; N_ELEMENTS];
        before[O as usize] = -1.152e15;
        before[H as usize] = -2.304e15;
        let mut after = before;
        // What the books would say if the arithmetic were exact, off by the
        // ulp the run actually produced.
        after[O as usize] -= harvested;
        after[H as usize] -= 2.0 * harvested;
        after[O as usize] -= 0.1595;
        let v = vent_inflow(&after, &before, harvested, &formula, 1200);
        assert_eq!(v[O as usize], 0.0, "rounding must not read as inflow");
        assert_eq!(v[H as usize], 0.0);
    }

    /// And the floor must not swallow a real one: three vents at 4e9/s is the
    /// figure `renew.toml` states, and it has to survive being measured.
    #[test]
    fn the_floor_does_not_swallow_a_real_inflow() {
        let mut formula = [0u16; N_ELEMENTS];
        formula[H as usize] = 1;
        formula[M as usize] = 1;
        let mut before = [0.0; N_ELEMENTS];
        before[H as usize] = 1.0e15;
        before[M as usize] = 1.0e15;
        let mut after = before;
        after[H as usize] += 1.2e11;
        after[M as usize] += 1.2e11;
        let v = vent_inflow(&after, &before, 0.0, &formula, 1200);
        assert_eq!(v[H as usize], 1.2e11);
        assert_eq!(v[M as usize], 1.2e11);
    }

    fn checkpoint(settle: f64, per_window: f64, standing: f64) -> Checkpoint {
        let mut formula = [0u16; N_ELEMENTS];
        formula[H as usize] = 1;
        formula[O as usize] = 2;
        formula[M as usize] = 1;
        let mut h = probe(formula, per_window, vents(1.0e3, 0.0, 1.0e3));
        h.standing = standing;
        Checkpoint {
            settle,
            harvests: vec![h],
        }
    }

    /// The reading the settle sweep exists for. `renew.toml` reports HO2M at
    /// 8.817e9/s after 150 s of settling and 4.543e4/s after 2200 s, in a pond
    /// with nothing alive in it at either depth. `holding` is 1.0 at both --
    /// each window is a steady drain -- so only the comparison between depths
    /// says what happened.
    #[test]
    fn a_rate_that_falls_between_depths_is_a_transient() {
        let (factor, word) = drift_word(8.817e9, 4.543e4);
        assert_eq!(factor, "1.9e5x");
        assert!(word.starts_with("transient"), "{word}");
    }

    /// And a vent-fed compound has to survive the same comparison: HM reads
    /// 1.205e10/s at every depth because three vents at 4e9/s do not tire.
    #[test]
    fn a_vent_fed_rate_holds_across_the_sweep() {
        let (factor, word) = drift_word(1.205e10, 1.204e10);
        assert_eq!(factor, "1.00x");
        assert_eq!(word, "steady across the sweep");
    }

    /// O2 goes from a trickle to nothing between the two depths, and "gone" is
    /// a different fact from "fell by a lot".
    #[test]
    fn a_rate_that_reaches_zero_is_named_as_gone() {
        let (factor, word) = drift_word(3.285e3, 0.0);
        assert_eq!(factor, "gone");
        assert!(word.starts_with("collapsed"), "{word}");
        // Dividing by it would give an infinity, not a fact.
        assert_eq!(drift_word(0.0, 0.0).0, "-");
    }

    /// A compound the pond is still building when the shallow probe runs is
    /// not a transient read backwards, and must not be reported as one.
    #[test]
    fn a_rate_that_grows_with_depth_is_not_a_transient() {
        let (_, word) = drift_word(1.0e3, 1.0e6);
        assert!(word.starts_with("climbing"), "{word}");
    }

    /// "Steady" has to mean steady. A pond four times slower by the deep read
    /// is not a transient -- ten is the line, and it is drawn where the
    /// reading this exists to catch was 1.9e5 past it -- but it was still
    /// moving when the shallow probe read it, and that is worth a word.
    #[test]
    fn the_band_between_the_verdicts_is_not_called_steady() {
        assert!(drift_word(4.0, 1.0).1.starts_with("falling"), "4x down");
        assert!(drift_word(1.0, 4.0).1.starts_with("climbing"), "4x up");
        assert_eq!(drift_word(1.2, 1.0).1, "steady across the sweep");
        assert_eq!(drift_word(1.0, 1.2).1, "steady across the sweep");
    }

    /// The summary line under a sweep table has to agree with the words in
    /// the table. The first version counted only transients and then announced
    /// that everything "held its rate" on a table whose one row said "climbing
    /// 12x" -- which is the sort of thing that makes a reader stop trusting
    /// the numbers above it, and rightly.
    #[test]
    fn a_climb_is_not_counted_as_holding_its_rate() {
        assert_eq!(drift_bucket(8.817e9, 4.543e4), Drift::Transient);
        assert_eq!(drift_bucket(1.880e3, 2.295e4), Drift::Climbed);
        assert_eq!(drift_bucket(1.205e10, 1.204e10), Drift::Steady);
        // Gone by the deep read is a transient, not a steady zero: the shallow
        // number was real and is not something to plan around.
        assert_eq!(drift_bucket(3.285e3, 0.0), Drift::Transient);
        // Nothing at either depth is nothing, and counting it as a transient
        // would inflate the headline on a table full of dead ends.
        assert_eq!(drift_bucket(0.0, 0.0), Drift::Steady);
        // And arriving with depth is a climb.
        assert_eq!(drift_bucket(0.0, 5.25), Drift::Climbed);
    }

    /// The buckets and the words must not disagree either: anything the bucket
    /// calls a transient has to read as one in the row.
    #[test]
    fn the_bucket_and_the_word_agree() {
        for (first, last) in [(8.817e9, 4.543e4), (3.285e3, 0.0)] {
            let (_, word) = drift_word(first, last);
            assert!(
                word.starts_with("transient") || word.starts_with("collapsed"),
                "{first:e} -> {last:e} bucketed as a transient but reads {word:?}"
            );
        }
        for (first, last) in [(1.880e3, 2.295e4), (0.0, 5.25)] {
            let (_, word) = drift_word(first, last);
            assert!(
                word.starts_with("climbing") || word.starts_with("nothing at the shallow"),
                "{first:e} -> {last:e} bucketed as a climb but reads {word:?}"
            );
        }
    }

    /// The sweep is ordered by the *deepest* probe, because that is the pond a
    /// population would have to live in, and every depth contributes one
    /// reading per compound in the order the depths were probed.
    #[test]
    fn a_sweep_reads_one_value_per_depth_in_order() {
        let checkpoints = vec![
            checkpoint(150.0, 100.0, 1.854e10),
            checkpoint(2200.0, 1.0, 1.037e4),
        ];
        let rows = sweep_rows(&checkpoints, |h| h.sustained());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, vec![100.0, 1.0]);
        let standing = sweep_rows(&checkpoints, |h| h.standing);
        assert_eq!(standing[0].1, vec![1.854e10, 1.037e4]);
    }

    /// A return-leg probe with the numbers set by hand.
    fn living(per_window: f64, released: f64, net: f64, standing: f64, floor: f64) -> Living {
        Living {
            reaction: 0,
            windows: [per_window; SUPPLY_WINDOWS],
            released: [released; SUPPLY_WINDOWS],
            net: [net; SUPPLY_WINDOWS],
            standing,
            floor,
            // One second per window, so a window's quantity is its rate.
            seconds: SUPPLY_WINDOWS as f64,
        }
    }

    /// The futile cycle, which is the reading `returns` exists to demote. The
    /// forward leg moves a great deal of energy and the pond is left short of
    /// none of it, because the reverse leg put it all back on the heat the
    /// forward leg deposited.
    #[test]
    fn a_futile_cycle_has_gross_power_and_no_net_power() {
        let l = living(1.0e9, 1.0e-9, 0.0, 0.0, 1.0e-12);
        assert_eq!(l.sustained(), 1.0e9);
        assert_eq!(l.gross(1.0), 1.0e-9);
        assert_eq!(l.power(1.0), 0.0);
        assert!(!l.resolved(), "zero net must not read as resolved");
    }

    /// A pond left *richer* than the control is a real reading about this
    /// world -- consuming a substrate can unblock a photochemical route that
    /// stores more than the reaction released -- and clamping it at zero would
    /// print the most interesting row in the table as nothing.
    #[test]
    fn a_metabolism_that_leaves_the_pond_richer_is_not_reported_as_zero() {
        let l = living(1.0e9, 1.0e-9, -4.0e-12, 0.0, 1.0e-12);
        assert!(l.resolved(), "a large negative net is a reading, not noise");
        assert!(l.power(1.0) < 0.0, "the sign has to survive to the report");
        assert_eq!(l.power(1.0), -4.0e-12);
    }

    /// And a net figure inside the control pond's own drift is not a small
    /// living. It is no reading at all, and the two must not be printed the
    /// same way.
    #[test]
    fn a_net_inside_the_controls_drift_is_unresolved() {
        assert!(!living(1.0e9, 1.0e-9, 5.0e-13, 0.0, 1.0e-12).resolved());
        assert!(living(1.0e9, 1.0e-9, 2.0e-12, 0.0, 1.0e-12).resolved());
    }

    /// The larder in the unit that makes it comparable to an income. A pond
    /// holding a hundred thousand seconds of its own resupply feeds a boom
    /// that looks like a carrying capacity for as long as anyone watches.
    #[test]
    fn the_larder_is_reported_in_seconds_of_the_income() {
        let l = living(100.0, 1.0e-9, 1.0e-10, 1.0e7, 0.0);
        assert_eq!(l.sustained(), 100.0);
        assert_eq!(l.larder_seconds(), Some(1.0e5));
        // No income to divide by is the worst case, not a missing reading.
        assert_eq!(living(0.0, 0.0, 0.0, 1.0e7, 0.0).larder_seconds(), None);
    }

    /// A capture efficiency below one scales both power columns and neither
    /// verdict.
    #[test]
    fn capture_efficiency_scales_both_powers() {
        let l = living(1.0e9, 1.0e-9, 4.0e-10, 0.0, 1.0e-12);
        assert!((l.gross(0.6) - 6.0e-10).abs() < 1.0e-24);
        assert!((l.power(0.6) - 2.4e-10).abs() < 1.0e-24);
    }

    /// `--only` and `--reaction` both make a table of one, and "every one of
    /// the 1 metabolisms" reads like a bug in the instrument.
    #[test]
    fn one_of_a_thing_is_not_described_in_the_plural() {
        assert_eq!(plural(1, "metabolism", "every one of the {} metabolisms"), "the one metabolism");
        assert_eq!(
            plural(23, "compound", "every one of the {} compounds"),
            "every one of the 23 compounds"
        );
    }

    /// Two settles that round to the same tick are the same pond, and probing
    /// it twice would measure only the probe.
    #[test]
    fn settle_depths_are_ascending_and_unique_by_tick() {
        let depths = settle_depths_of(vec![600.0, 150.0, 600.004, 2200.0], 0.0, 0.01).unwrap();
        assert_eq!(
            depths,
            vec![(150.0, 15_000), (600.0, 60_000), (2200.0, 220_000)]
        );
        // An empty list falls back to the config's seed delay.
        assert_eq!(settle_depths_of(vec![], 150.0, 0.01).unwrap(), vec![(150.0, 15_000)]);
        assert!(settle_depths_of(vec![-1.0], 150.0, 0.01).is_err());
    }

    /// Provenance is a fraction of what was measured, so a pond delivering
    /// more than the probe took does not report more than all of it.
    #[test]
    fn provenance_does_not_exceed_one() {
        let mut formula = [0u16; N_ELEMENTS];
        formula[H as usize] = 1;
        formula[M as usize] = 1;
        let h = probe(formula, 100.0, vents(1.0e6, 0.0, 1.0e6));
        assert_eq!(h.provenance(), 1.0);
    }
}
