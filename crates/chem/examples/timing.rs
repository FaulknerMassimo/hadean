//! Generation timing and network shape.
//! `cargo run -p hadean-chem --release --example timing`
use hadean_chem::{generate::generate, ChemParams};

fn main() {
    println!(
        "{:>4}  {:>9}  {:>9}  {:>9}  {:>8}",
        "n", "mean ms", "worst ms", "reactions", "min"
    );
    for n in [32usize, 40, 48, 64] {
        let params = ChemParams {
            n_compounds: n,
            ..Default::default()
        };
        let (mut ms, mut worst, mut rx, mut lo) = (0.0, 0.0f64, 0usize, usize::MAX);
        for seed in 0..8u64 {
            let t = std::time::Instant::now();
            let c = generate(seed, params);
            let el = t.elapsed().as_secs_f64() * 1e3;
            ms += el;
            worst = worst.max(el);
            c.verify().unwrap();
            rx += c.n_reactions();
            lo = lo.min(c.n_reactions());
        }
        println!(
            "{n:>4}  {:>9.0}  {worst:>9.0}  {:>9}  {lo:>8}",
            ms / 8.0,
            rx / 8
        );
    }
}
