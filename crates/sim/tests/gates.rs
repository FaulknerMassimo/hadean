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
fn checked_in_pond_preset_matches_the_default() {
    let preset = WorldConfig::from_toml(include_str!("../../../configs/pond.toml"))
        .expect("pond preset parses");
    assert_eq!(preset, WorldConfig::default());
}
