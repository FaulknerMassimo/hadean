//! Cross-cutting gates that exercise cells together with fields, chemistry,
//! snapshots, hashing, and the global audit.

use hadean_sim::config::{GridConfig, WorldConfig};
use hadean_sim::{snapshot, World};

fn small() -> WorldConfig {
    WorldConfig {
        seed: 17,
        grid: GridConfig {
            nx: 8,
            ny: 8,
            nz: 6,
            dx: 25.0e-6,
        },
        cells: hadean_cell::CellConfig {
            initial_count: 6,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn cell_worlds_are_bit_identical() {
    let mut a = World::new(small()).expect("world");
    let mut b = World::new(small()).expect("world");
    a.run(250);
    b.run(250);
    assert_eq!(a.state_digest(), b.state_digest());
}

#[test]
fn cell_worlds_resume_bit_identically() {
    let mut uninterrupted = World::new(small()).expect("world");
    uninterrupted.run(120);
    let bytes = snapshot::save(&uninterrupted).expect("snapshot");
    let mut resumed = snapshot::load(&bytes).expect("restore");
    uninterrupted.run(130);
    resumed.run(130);
    assert_eq!(uninterrupted.state_digest(), resumed.state_digest());
}

#[test]
fn cell_activity_stays_inside_the_energy_and_mass_audit() {
    let mut world = World::new(small()).expect("world");
    let report = world.run(2_000).expect("audit");
    assert!(report.relative.abs() < 1.0e-8, "energy drift: {report:?}");
    assert!(report.mass_drift.abs() < 1.0e-8, "mass drift: {report:?}");
}

#[test]
fn a_fed_protocell_divides_in_the_full_world() {
    let mut config = small();
    config.cells.initial_count = 1;
    config.cells.division_reserve = 1.0e-15;
    config.cells.maintenance_power = 0.0;
    // Hand-feed the ancestor at tick zero rather than waiting for the pond's
    // photochemistry to lay a table.
    config.cells.seed_delay = 0.0;
    let mut world = World::new(config).expect("world");
    let reaction = world.cells.metabolic_reaction.expect("metabolism");
    for &(compound, coefficient) in &world.chem.reaction(reaction).reactants {
        world.cells.cells[0].contents[compound as usize] = 1.0e8 * coefficient as f64;
    }
    for _ in 0..25 {
        world.step();
    }
    assert!(world.cells.births > 1, "the fed ancestor did not divide");
    assert!(world.cells.alive() > 1);
}

#[test]
fn ancestors_arrive_on_schedule_and_the_lifeless_world_still_audits() {
    // A generated chemistry has no photochemical products at tick zero, so
    // ancestors introduced then starve in an empty larder. They wait instead,
    // and the pond in the meantime is an ordinary audited world.
    let mut config = small();
    config.cells.seed_delay = 3.0;
    let seed_tick = (config.cells.seed_delay as f64 / config.dt) as u64;

    let mut world = World::new(config).expect("world");
    assert_eq!(world.cells.alive(), 0, "life started before its cue");
    assert!(
        world.cells.metabolic_reaction.is_none(),
        "what the ancestors eat is settled when they arrive, from the pond \
         the lifeless world has actually made"
    );

    let report = world.run(seed_tick - 1).expect("audit");
    assert_eq!(world.cells.alive(), 0);
    assert!(report.relative.abs() < 1.0e-8, "lifeless drift: {report:?}");

    let report = world.run(2).expect("audit");
    assert_eq!(world.cells.alive(), 6, "the founding cohort did not arrive");
    assert!(
        world.cells.metabolic_reaction.is_some(),
        "the cohort arrived without a metabolism"
    );
    // Ancestors arrive empty-handed, so their arrival is invisible to the
    // ledger.
    assert!(report.relative.abs() < 1.0e-8, "seeding drift: {report:?}");
}

#[test]
fn a_genome_population_feeds_itself_and_grows() {
    // The end-to-end claim, in a full world: cells built out of bytes take up
    // compounds through transporters they decode, catalyse a reaction an
    // enzyme they decode recognises, bank the energy, and divide on it. Every
    // step of that goes through the genome; none of it is reachable by a
    // pre-genome cell's hardcoded path.
    let mut config = small();
    config.cells.genome = true;
    config.cells.initial_count = 4;
    config.cells.division_reserve = 1.0e-15;
    config.cells.maintenance_power = 0.0;
    config.cells.seed_delay = 0.0;
    let mut world = World::new(config).expect("world");

    let reaction = world.cells.metabolic_reaction.expect("metabolism");
    for &(compound, coefficient) in &world.chem.reaction(reaction).reactants {
        for cell in &mut world.cells.cells {
            cell.contents[compound as usize] = 1.0e8 * coefficient as f64;
        }
    }
    assert!(
        world.cells.cells.iter().all(|c| c.genome.is_some()),
        "the founding cohort arrived without genomes"
    );

    // Long enough for the ancestor's protein to actually be there. A gene at
    // stability 0.02 is at a twentieth of its level after a second, so a test
    // that looked immediately after seeding would see cells that had already
    // divided on a metabolism it would report as absent.
    for _ in 0..2_000 {
        world.step();
    }
    assert!(world.cells.births > 4, "no genome cell divided");

    // The ancestral reaction has to be what the bulk of the population is
    // running -- that is the claim, that a decoded enzyme feeds a cell.
    //
    // It is deliberately not the *only* row. This assertion was an equality
    // once and it failed the moment point substitution became a graded step
    // instead of a uniform redraw: the same run came back catalysing five
    // reactions, one of them on 59876 cells and four of them on a handful
    // each. That is lineages drifting onto neighbouring reactions, which is
    // the entire reason the genome exists, and a test that forbids it is
    // testing for the absence of the feature.
    let diet = world.cells.diet();
    let ancestral = diet
        .iter()
        .find(|&&(id, _)| id == reaction)
        .map(|&(_, n)| n)
        .unwrap_or(0);
    let total: usize = diet.iter().map(|&(_, n)| n).sum();
    assert!(
        ancestral * 2 > total,
        "the ancestral reaction is not the population's main living: {diet:?}"
    );
}

#[test]
fn a_genome_world_is_bit_identical_and_resumes() {
    // Determinism is the property everything else here rests on, and the
    // genome adds three new places to lose it: a variable number of mutation
    // draws per division, an `Arc` shared between siblings, and a decode that
    // is rebuilt rather than stored. All three are checked by these two
    // equalities.
    let mut config = small();
    config.cells.genome = true;
    config.cells.seed_delay = 0.0;

    let mut a = World::new(config.clone()).expect("world");
    let mut b = World::new(config.clone()).expect("world");
    a.run(250);
    b.run(250);
    assert_eq!(a.state_digest(), b.state_digest(), "two runs diverged");

    let bytes = snapshot::save(&a).expect("snapshot");
    let mut resumed = snapshot::load(&bytes).expect("restore");
    a.run(250);
    resumed.run(250);
    assert_eq!(
        a.state_digest(),
        resumed.state_digest(),
        "a resumed genome world diverged"
    );
}

#[test]
fn genome_cells_stay_inside_the_energy_and_mass_audit() {
    // The genome does not create matter: proteins are an energetic burden and
    // not a material one, and every joule a cell spends still lands in the
    // heat field. If either were false it would show up here first.
    let mut config = small();
    config.cells.genome = true;
    config.cells.seed_delay = 0.0;
    let mut world = World::new(config).expect("world");
    let report = world.run(2_000).expect("audit");
    assert!(report.relative.abs() < 1.0e-8, "energy drift: {report:?}");
    assert!(report.mass_drift.abs() < 1.0e-8, "mass drift: {report:?}");
}

#[test]
fn mutation_is_what_makes_a_population_stop_being_one_lineage() {
    // The control this whole layer needs. A genome with every mutation rate at
    // zero is a clone line for ever: same bytes, same allocation, one lineage
    // however long it runs. Turn the rates on and the same world splits.
    let lineages = |mutation| {
        let mut config = small();
        config.cells.genome = true;
        config.cells.seed_delay = 0.0;
        config.cells.division_reserve = 1.0e-15;
        config.cells.maintenance_power = 0.0;
        config.cells.mutation = mutation;
        let mut world = World::new(config).expect("world");
        let reaction = world.cells.metabolic_reaction.expect("metabolism");
        for &(compound, coefficient) in &world.chem.reaction(reaction).reactants {
            for cell in &mut world.cells.cells {
                cell.contents[compound as usize] = 1.0e9 * coefficient as f64;
            }
        }
        world.run(400).expect("audit");
        (
            world.cells.trait_summary().distinct_genomes,
            world.cells.alive(),
        )
    };

    let (clones, clone_population) = lineages(hadean_cell::MutationRates::none());
    let (varied, varied_population) = lineages(hadean_cell::MutationRates::default());
    assert!(clone_population > 6 && varied_population > 6, "nothing bred");
    assert_eq!(clones, 1, "a clone line grew {clones} genomes");
    assert!(
        varied > 1,
        "mutation produced {varied} lineages out of {varied_population} cells"
    );
}

#[test]
fn checked_in_pond_preset_matches_the_default() {
    let preset = WorldConfig::from_toml(include_str!("../../../configs/pond.toml"))
        .expect("pond preset parses");
    assert_eq!(preset, WorldConfig::default());
}
