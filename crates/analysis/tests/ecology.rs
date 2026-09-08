//! The Phase 2 gate, and the tests that keep it honest.
//!
//! The gate itself -- a population that grows into its food supply, overshoots
//! it, crashes and recovers -- takes a world of tens of thousands of voxels
//! and a run of hundreds of thousands of ticks. That belongs in
//! `hadean ecology`, not in `cargo test`, so it lives here behind `#[ignore]`
//! and is named in the README as the command to run.
//!
//! What is worth running every build is everything around it: that a real run
//! produces a series the analysis can read, and -- more importantly -- that
//! the analysis says *no* when it should. A gate that cannot fail is not a
//! gate, and the specific failure worth rehearsing is the one this project
//! actually hit: a population that stops growing because it reached a number
//! in a config file, and looks from a distance exactly like one that reached
//! its carrying capacity.

use hadean_analysis::population::{population_curve, Thresholds};
use hadean_analysis::Series;
use hadean_chem::element::{C, N_ELEMENTS, O};
use hadean_sim::config::{GridConfig, WorldConfig};
use hadean_sim::World;

/// A pond small enough to run in a test, with everything else left alone.
fn small(cells: hadean_cell::CellConfig) -> WorldConfig {
    WorldConfig {
        seed: 5,
        grid: GridConfig {
            nx: 10,
            ny: 10,
            nz: 6,
            dx: 25.0e-6,
        },
        cells,
        ..Default::default()
    }
}

/// Fill every voxel with what the ancestors eat.
///
/// A generated chemistry makes its photochemical products slowly, which is
/// exactly why the pond runs before life does. A test that wants to watch
/// cells rather than sunlight lays the table itself.
fn lay_the_table(world: &mut World) {
    let reaction = world.cells.metabolic_reaction.expect("metabolism");
    let reactants: Vec<u16> = world
        .chem
        .reaction(reaction)
        .reactants
        .iter()
        .map(|&(c, _)| c)
        .collect();
    for c in reactants {
        world.amounts.plane_mut(c as usize).fill(2.0e9);
    }
    // The soup changed, so the audit's idea of the world it is watching has
    // to change with it.
    world.rebaseline_audit();
}

fn run_primed(config: WorldConfig, ticks: u64, prime: bool) -> (World, Series) {
    let mut world = World::new(config).expect("world");
    if prime {
        lay_the_table(&mut world);
    }
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
    (world, series)
}

fn run(config: WorldConfig, ticks: u64) -> (World, Series) {
    run_primed(config, ticks, false)
}

#[test]
fn a_run_produces_a_series_the_curve_analysis_can_read() {
    let cells = hadean_cell::CellConfig {
        initial_count: 4,
        seed_delay: 2.0,
        ..Default::default()
    };
    let cap = cells.population_cap as u64;
    // Long enough for the pond's photochemistry to put something on the
    // table, which is what the food column is reading.
    let (world, series) = run(small(cells), 12_000);

    assert!(series.len() > 10, "not enough samples to say anything");
    let curve = population_curve(&series, cap, Thresholds::default());
    assert_eq!(curve.samples, series.len());
    assert_eq!(curve.start, 4, "the boom is measured from the founders");
    assert!(!curve.cap_limited);
    assert!(world.limiting_substrate().is_finite());
    assert!(
        series.rows.last().expect("a last row").substrate > 0.0,
        "the pond never held any of what its cells eat"
    );
}

#[test]
fn a_metabolism_is_only_chosen_when_the_pond_has_laid_a_table() {
    // The check behind choosing a metabolism from measured abundance rather
    // than from the reaction graph. Seed 5 used to pick a reaction eating a
    // compound whose only source was a photoreaction with no substrates; seed
    // 4 picked one that is reachable in principle and never accumulates. Both
    // cohorts sat in the pond with a food supply of exactly zero.
    //
    // Not every generated world offers a living. Seed 4 populates eleven of
    // its thirty-two compounds and none of its downhill reactions runs on
    // what is there; the right answer for that pond is no ancestor at all,
    // not an ancestor that starves. So this asks for the contract -- chosen
    // implies present -- and that the seeds the project actually ships with
    // do offer one.
    let mut barren = Vec::new();
    for seed in 1..8u64 {
        let mut config = small(hadean_cell::CellConfig {
            // The pond runs lifeless, so what it holds is what it makes.
            enabled: false,
            ..Default::default()
        });
        config.seed = seed;
        let delay = config.cells.seed_delay as f64;

        let mut world = World::new(config).expect("world");
        world.run((delay / world.config.dt) as u64);

        let abundance: Vec<f64> = (0..world.chem.n_compounds())
            .map(|c| world.amounts.total_of(c))
            .collect();
        let Some(chosen) =
            hadean_cell::choose_metabolism(&world.chem, &abundance, None, world.grid.len())
        else {
            barren.push(seed);
            continue;
        };

        for &(c, _) in &world.chem.reaction(chosen).reactants {
            assert!(
                abundance[c as usize] > 0.0,
                "seed {seed} chose to eat {}, and after {delay} s its pond has made none of it",
                world.chem.compound(c).name
            );
        }
    }
    assert!(
        !barren.contains(&1),
        "the default seed's pond offers no living"
    );
    assert!(
        barren.len() < 4,
        "most generated worlds are barren, which is a chemistry problem: {barren:?}"
    );
}

#[test]
fn a_population_pinned_to_its_cap_is_reported_as_such() {
    // The failure mode this whole module exists to catch. Give the ancestors
    // a cap of a handful and free food, and they will sit on it -- a flat top
    // that is a fact about the config file and not about the pond.
    let cells = hadean_cell::CellConfig {
        initial_count: 2,
        population_cap: 8,
        seed_delay: 0.0,
        maintenance_power: 0.0,
        division_reserve: 1.0e-16,
        ..Default::default()
    };
    let cap = cells.population_cap as u64;
    let (world, series) = run_primed(small(cells), 3_000, true);

    assert_eq!(world.cells.alive(), 8, "the cap was not reached");
    let curve = population_curve(&series, cap, Thresholds::default());
    assert!(curve.cap_limited);
    assert!(
        !curve.passes(),
        "a cap-limited run must never pass the gate"
    );
    assert!(
        curve.verdict().starts_with("CAP-LIMITED"),
        "{}",
        curve.verdict()
    );
}

#[test]
fn the_checked_in_pond_preset_keeps_its_day() {
    // There was briefly a second preset that nailed the sun to the sky, on the
    // theory that a fixed sun made the population's crash unambiguously about
    // resources. It made the pond uninhabitable instead: with the sun fixed,
    // the compound the ancestor eats builds to a peak at about 250 s and then
    // decays to nothing -- 1.27e12 particles down to 40 by 1500 s, with a
    // single non-dividing cell in the world to prove nothing was eating it.
    // The day/night cycle is not a confound in this world, it is what makes
    // the food supply renewable.
    let preset = WorldConfig::from_toml(include_str!("../../../configs/pond.toml"))
        .expect("pond preset parses");
    preset.validate().expect("pond preset is a runnable world");
    assert!(
        preset.light.day_length > 0.0,
        "the pond's sun has to rise and set; a fixed sun drains its photochemistry"
    );
    assert!(preset.cells.enabled);
    World::new(preset).expect("pond preset builds");
}

#[test]
fn the_gate_preset_is_the_pond_in_everything_but_size_and_lifecycle() {
    // `gate.toml` exists so the population gate is runnable in twenty minutes
    // rather than an afternoon. That is only a legitimate reason to have a
    // second preset if the physics is identical -- the last second preset
    // changed the sun and quietly made the world uninhabitable.
    let pond = WorldConfig::from_toml(include_str!("../../../configs/pond.toml"))
        .expect("pond preset parses");
    let gate = WorldConfig::from_toml(include_str!("../../../configs/gate.toml"))
        .expect("gate preset parses");
    gate.validate().expect("gate preset is a runnable world");

    assert_eq!(gate.seed, pond.seed);
    assert_eq!(gate.dt, pond.dt);
    assert_eq!(gate.chemistry, pond.chemistry);
    assert_eq!(gate.light, pond.light, "the sun must be the same sun");
    assert_eq!(gate.heat, pond.heat);
    assert_eq!(gate.flow, pond.flow);
    assert_eq!(gate.vents, pond.vents);
    assert_eq!(gate.initial, pond.initial);
    assert!(
        gate.grid.nx < pond.grid.nx,
        "the gate pond should be smaller"
    );
    assert_eq!(gate.grid.dx, pond.grid.dx, "same voxel, fewer of them");

    World::new(gate).expect("gate preset builds");
}

#[test]
fn the_evolve_preset_is_the_gate_pond_with_a_genome() {
    // Same argument as the gate/pond parity above, and the same reason to
    // enforce it: `evolve.toml` exists to isolate what a genome changes, and
    // it can only do that if nothing else changed with it. A preset that
    // quietly moved the sun or the vents alongside would make every
    // comparison between the two worthless.
    let gate = WorldConfig::from_toml(include_str!("../../../configs/gate.toml"))
        .expect("gate preset parses");
    let evolve = WorldConfig::from_toml(include_str!("../../../configs/evolve.toml"))
        .expect("evolve preset parses");
    evolve.validate().expect("evolve preset is a runnable world");

    assert_eq!(evolve.seed, gate.seed);
    assert_eq!(evolve.dt, gate.dt);
    assert_eq!(evolve.grid, gate.grid, "same pond, not a smaller one");
    assert_eq!(evolve.chemistry, gate.chemistry);
    assert_eq!(evolve.light, gate.light, "the sun must be the same sun");
    assert_eq!(evolve.heat, gate.heat);
    assert_eq!(evolve.flow, gate.flow);
    assert_eq!(evolve.vents, gate.vents);
    assert_eq!(evolve.initial, gate.initial);
    assert_eq!(evolve.schedule, gate.schedule);

    assert!(evolve.cells.genome, "the evolve preset has no genome");
    assert!(!gate.cells.genome, "the control preset has a genome");
    // The lifecycle numbers are the gate's, carried over deliberately, so the
    // only difference between the two worlds is the cell's heritable state.
    assert_eq!(
        hadean_cell::CellConfig {
            genome: gate.cells.genome,
            mutation: gate.cells.mutation,
            enzyme_sigma: gate.cells.enzyme_sigma,
            transport_sigma: gate.cells.transport_sigma,
            structural_benefit: gate.cells.structural_benefit,
            ..evolve.cells.clone()
        },
        gate.cells,
        "the two presets differ somewhere other than the genome"
    );

    World::new(evolve).expect("evolve preset builds");
}

#[test]
fn the_renew_preset_is_the_gate_pond_on_a_living_it_can_resupply() {
    // Same argument as the two parity tests above. `renew.toml` exists to
    // isolate what changing the *diet* does, and it can only do that if the
    // pond around it did not change too.
    let gate = WorldConfig::from_toml(include_str!("../../../configs/gate.toml"))
        .expect("gate preset parses");
    let renew = WorldConfig::from_toml(include_str!("../../../configs/renew.toml"))
        .expect("renew preset parses");
    renew.validate().expect("renew preset is a runnable world");

    assert_eq!(renew.seed, gate.seed);
    assert_eq!(renew.dt, gate.dt);
    assert_eq!(renew.grid, gate.grid, "same pond, not a smaller one");
    assert_eq!(renew.chemistry, gate.chemistry);
    assert_eq!(renew.light, gate.light, "the sun must be the same sun");
    assert_eq!(renew.heat, gate.heat);
    assert_eq!(renew.flow, gate.flow);
    assert_eq!(renew.vents, gate.vents);
    assert_eq!(renew.initial, gate.initial);
    assert_eq!(renew.schedule, gate.schedule);

    // Unlike `evolve.toml`, this preset's lifecycle numbers are deliberately
    // *not* the gate's -- carrying them over was tried and the cohort starved,
    // because they were measured against a substrate standing six hundred
    // times higher. So the parity enforced here is the pond, and the four
    // numbers that may differ are named one by one rather than waved through:
    // a fifth drifting in unremarked is exactly what this test is for.
    assert!(gate.cells.metabolism.is_none(), "the control names a diet");
    let named = renew.cells.metabolism.clone().expect("renew names no diet");
    assert_eq!(
        hadean_cell::CellConfig {
            metabolism: gate.cells.metabolism.clone(),
            membrane_scale: gate.cells.membrane_scale,
            metabolic_rate: gate.cells.metabolic_rate,
            maintenance_power: gate.cells.maintenance_power,
            division_reserve: gate.cells.division_reserve,
            ..renew.cells.clone()
        },
        gate.cells,
        "the two presets differ somewhere other than the diet and its energetics"
    );

    // The diet has to name a real living in this chemistry. Whether a newborn
    // can *pay* for it is not a property of the config file -- it depends on
    // what the pond has made by t = seed_delay, and HO2M is a compound the
    // chemistry builds rather than one the soup is seeded with, so at tick 0
    // there is none of it and the question has no answer yet. `ecology` prints
    // that check at the moment it becomes answerable; the arithmetic behind it
    // is tested in `hadean_cell`.
    let world = World::new(renew).expect("renew preset builds");
    let reaction = hadean_cell::named_metabolism(&world.chem, &named).expect("a named reaction");
    assert!(
        world.chem.reaction(reaction).dh < 0.0,
        "the named diet is uphill"
    );
}

#[test]
fn the_vent5_preset_is_the_renew_pond_with_oxygen_coming_into_it() {
    // `vent5.toml` exists to isolate what putting O2 in the water does, and it
    // can only do that if the pond around it did not change too. Same argument
    // as the three parity tests above, with one wrinkle worth stating: the
    // change is *inside* `[vents]`, so this is the one preset whose parity
    // cannot be asserted by comparing that whole block.
    let renew = WorldConfig::from_toml(include_str!("../../../configs/renew.toml"))
        .expect("renew preset parses");
    let vent5 = WorldConfig::from_toml(include_str!("../../../configs/vent5.toml"))
        .expect("vent5 preset parses");
    vent5.validate().expect("vent5 preset is a runnable world");

    assert_eq!(vent5.seed, renew.seed);
    assert_eq!(vent5.dt, renew.dt);
    assert_eq!(vent5.grid, renew.grid, "same pond, not a smaller one");
    assert_eq!(vent5.chemistry, renew.chemistry);
    assert_eq!(vent5.light, renew.light, "the sun must be the same sun");
    assert_eq!(vent5.heat, renew.heat, "the same vents in the same places");
    assert_eq!(vent5.flow, renew.flow);
    assert_eq!(vent5.initial, renew.initial);
    assert_eq!(vent5.schedule, renew.schedule);
    assert_eq!(
        vent5.cells, renew.cells,
        "the lifecycle numbers are renew's, inherited on trust and to be re-measured"
    );

    // One field differs, and it is the one the file is named for.
    assert_eq!(
        hadean_sim::config::VentConfig {
            fuel_species: renew.vents.fuel_species,
            ..vent5.vents
        },
        renew.vents,
        "vent5 differs from renew somewhere other than fuel_species"
    );
    assert!(
        vent5.vents.fuel_species > renew.vents.fuel_species,
        "vent5 must inject more species than renew, not fewer"
    );

    // The point of the file, asserted rather than assumed: at this
    // `fuel_species` the vents actually deliver oxygen, and at renew's they do
    // not. `rank_vent_fuel` is a function of the seed, so this is a claim
    // about seed 1's chemistry and it should fail loudly if that generator
    // changes rather than quietly measuring a different pond.
    let world = World::new(vent5.clone()).expect("vent5 preset builds");
    let injected = |n: usize| -> Vec<[u16; N_ELEMENTS]> {
        world.chem.vent_fuel[..n.min(world.chem.vent_fuel.len())]
            .iter()
            .map(|&c| world.chem.compound(c).formula)
            .collect()
    };
    let carries = |fuels: &[[u16; N_ELEMENTS]], e: u8| {
        fuels.iter().any(|f| f[e as usize] > 0)
    };
    assert!(
        carries(&injected(vent5.vents.fuel_species), O),
        "vent5 brings no oxygen into the pond, which is the only reason it exists"
    );
    assert!(
        !carries(&injected(renew.vents.fuel_species), O),
        "renew already brought oxygen in, so vent5 changes nothing"
    );
    // And the fact that bounds every carbon living on this seed, whatever the
    // vents are set to.
    assert!(
        !carries(&injected(world.chem.vent_fuel.len()), C),
        "carbon enters this pond at some fuel_species after all -- \
         the handoff says it never does, and one of the two is now wrong"
    );
}

/// The Phase 2 gate. Minutes, not seconds -- run it with
/// `cargo test --release -p hadean-analysis -- --ignored --nocapture`, or
/// more usually as `hadean ecology --config configs/gate.toml`.
#[test]
#[ignore = "population gate; takes about twenty minutes"]
fn the_gate_preset_booms_busts_and_recovers() {
    let config = WorldConfig::from_toml(include_str!("../../../configs/gate.toml"))
        .expect("gate preset parses");
    let cap = config.cells.population_cap as u64;
    let (_, series) = run(config, 250_000);
    let curve = population_curve(&series, cap, Thresholds::default());
    assert!(curve.passes(), "{}", curve.verdict());
    assert!(
        curve.substrate_drawdown > 2.0,
        "the boom did not draw its own food down: {:.2}x",
        curve.substrate_drawdown
    );
}
