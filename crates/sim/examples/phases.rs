//! Per-phase timing. `cargo run -p hadean-sim --release --example phases`
use std::time::Instant;

use hadean_fields::heat;
use hadean_fields::transport::{advect, diffuse};
use hadean_sim::{World, WorldConfig};
use rayon::prelude::*;

fn main() {
    let cfg = WorldConfig::default();
    let mut w = World::new(cfg).unwrap();
    for _ in 0..5 {
        w.step();
    }
    let n = 20;
    let dt = w.config.dt;
    let n_voxels = w.grid.len();
    let n_compounds = w.chem.n_compounds();
    println!("threads: {}", rayon::current_num_threads());

    let mut t_light = 0.0;
    let mut t_react = 0.0;
    let mut t_transpose = 0.0;
    let mut t_diff = 0.0;
    let mut t_adv = 0.0;
    let mut t_heat = 0.0;

    let mut transpose = Vec::new();
    let mut residual_transpose = Vec::new();
    let mut outcomes = Vec::new();
    let mut scratch = Vec::new();

    for _ in 0..n {
        let t = Instant::now();
        w.solver
            .propagate(&w.grid, &w.chem, &w.amounts, w.elapsed(), dt, &mut w.light);
        t_light += t.elapsed().as_secs_f64();

        let t = Instant::now();
        w.amounts.tile(0, n_voxels, &mut transpose);
        w.residual.tile(0, n_voxels, &mut residual_transpose);
        t_transpose += t.elapsed().as_secs_f64();

        let t = Instant::now();
        {
            let net = &w.network;
            let light = &w.light;
            let hf = &w.heat;
            transpose
                .par_chunks_mut(n_compounds)
                .zip(residual_transpose.par_chunks_mut(n_compounds))
                .enumerate()
                .map(|(v, (a, r))| {
                    let absorbed = light.absorbed_bands(v);
                    net.step_voxel(a, r, hf.temperature(v), dt, &absorbed, &[])
                })
                .collect_into_vec(&mut outcomes);
        }
        t_react += t.elapsed().as_secs_f64();

        let t = Instant::now();
        w.amounts.untile(0, n_voxels, &transpose);
        w.residual.untile(0, n_voxels, &residual_transpose);
        t_transpose += t.elapsed().as_secs_f64();

        let t = Instant::now();
        {
            let grid = &w.grid;
            let compounds = &w.chem.compounds;
            let residual = &mut w.residual.data;
            w.amounts
                .data
                .par_chunks_mut(n_voxels)
                .zip(residual.par_chunks_mut(n_voxels))
                .enumerate()
                .for_each_init(Vec::new, |s, (c, (plane, r))| {
                    diffuse(grid, plane, r, compounds[c].diffusion, dt, s)
                });
        }
        t_diff += t.elapsed().as_secs_f64();

        let t = Instant::now();
        {
            let grid = &w.grid;
            let flow = &w.flow;
            let residual = &mut w.residual.data;
            w.amounts
                .data
                .par_chunks_mut(n_voxels)
                .zip(residual.par_chunks_mut(n_voxels))
                .for_each_init(Vec::new, |s, (plane, r)| {
                    advect(grid, plane, r, flow, dt, s)
                });
        }
        t_adv += t.elapsed().as_secs_f64();

        let t = Instant::now();
        heat::step(&w.grid, &mut w.heat, &w.config.heat, dt, &mut scratch);
        t_heat += t.elapsed().as_secs_f64();
    }

    let ms = |x: f64| x / n as f64 * 1e3;
    println!("light      {:>8.2} ms", ms(t_light));
    println!("transpose  {:>8.2} ms", ms(t_transpose));
    println!("react      {:>8.2} ms", ms(t_react));
    println!("diffuse    {:>8.2} ms", ms(t_diff));
    println!("advect     {:>8.2} ms", ms(t_adv));
    println!("heat       {:>8.2} ms", ms(t_heat));
    println!(
        "total      {:>8.2} ms",
        ms(t_light + t_transpose + t_react + t_diff + t_adv + t_heat)
    );
}
