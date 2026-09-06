//! Is the audit drifting, or is it wandering?
//!
//! `cargo run -p hadean-sim --release --example leak`
//!
//! This is the tool that found the residual bug, and it is worth keeping for
//! the next one. A threshold cannot tell a leak from float noise -- both look
//! like a small number -- but their *shapes* differ, and the shape is only
//! visible over many blocks and several seeds. Run a closed world (no light,
//! no vents, no surface exchange, so the only correct drift is zero) and watch
//! the last two columns:
//!
//! * `drift / t` flat and `drift / sqrt(t)` climbing: the drift is linear in
//!   ticks. That is a leak, and it will not go away by itself.
//! * `drift / t` falling and `drift / sqrt(t)` flat: a random walk. That is
//!   rounding, and it is as good as float arithmetic gets.
//!
//! When this was written the first pattern showed at 8e-8 relative, small
//! enough to pass any sane threshold, and it was real.
use hadean_chem::chemistry::N_BANDS;
use hadean_fields::flow::FlowConfig;
use hadean_fields::heat::HeatConfig;
use hadean_fields::light::LightConfig;
use hadean_sim::config::{GridConfig, VentConfig};
use hadean_sim::{World, WorldConfig};

fn closed(seed: u64) -> WorldConfig {
    WorldConfig {
        seed,
        grid: GridConfig {
            nx: 24,
            ny: 24,
            nz: 12,
            dx: 25.0e-6,
        },
        light: LightConfig {
            irradiance: [0.0; N_BANDS],
            day_length: 0.0,
            ..Default::default()
        },
        heat: HeatConfig {
            surface_transfer: 0.0,
            vents: 0,
            vent_power: 0.0,
            ..Default::default()
        },
        vents: VentConfig {
            fuel_rate: 0.0,
            fuel_species: 0,
        },
        flow: FlowConfig {
            speed: 0.0,
            rolls: 2,
        },
        ..Default::default()
    }
}

fn main() {
    let ticks_per_block = 500u64;
    let blocks = 16;
    let seeds = 3u64;

    // Root-mean-square drift across seeds, so a random walk shows its shape
    // instead of one sample's coin flips.
    let mut worlds: Vec<World> = (0..seeds)
        .map(|s| World::new(closed(s + 1)).unwrap())
        .collect();
    println!(
        "{:>7} {:>14} {:>14} {:>10} {:>10}",
        "tick", "rms mass drift", "rms energy", "/t", "/sqrt(t)"
    );
    for block in 1..=blocks {
        for w in worlds.iter_mut() {
            w.run(ticks_per_block);
        }
        let t = (block * ticks_per_block) as f64;
        let mut mass = 0.0;
        let mut energy = 0.0;
        for w in worlds.iter_mut() {
            let r = w.audit_now();
            mass += r.mass_drift * r.mass_drift;
            energy += r.relative * r.relative;
        }
        let mass = (mass / seeds as f64).sqrt();
        let energy = (energy / seeds as f64).sqrt();
        println!(
            "{:>7.0} {mass:>14.4e} {energy:>14.4e} {:>10.3e} {:>10.3e}",
            t,
            mass / t,
            mass / t.sqrt()
        );
    }
}
