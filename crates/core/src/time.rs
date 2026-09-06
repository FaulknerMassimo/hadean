//! Fixed timestep and multi-rate scheduling.
//!
//! `dt` is never coupled to frame time. The viewer runs at its own rate and
//! interpolates; the simulation advances in fixed increments so a run is
//! reproducible from `(seed, config, tick_count)` alone.
//!
//! Processes have different natural timescales, and running all of them every
//! tick is the single largest avoidable cost in the simulation. [`Schedule`]
//! encodes the intervals, and [`Schedule::stagger`] spreads per-entity work
//! across the interval so load stays flat instead of spiking every Nth tick.

use serde::{Deserialize, Serialize};

/// The simulation clock.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Clock {
    pub tick: u64,
    /// Simulated seconds per tick.
    pub dt: f64,
}

impl Clock {
    pub fn new(dt: f64) -> Self {
        assert!(dt > 0.0, "dt must be positive");
        Self { tick: 0, dt }
    }

    /// Simulated seconds elapsed since the start of the run.
    #[inline]
    pub fn elapsed(&self) -> f64 {
        self.tick as f64 * self.dt
    }

    #[inline]
    pub fn advance(&mut self) {
        self.tick += 1;
    }
}

/// How often each phase runs, in ticks.
///
/// Defaults follow the timescale table in the design: physics and chemistry
/// every tick, transcription an order of magnitude slower, development slower
/// again.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Schedule {
    pub physics: u64,
    pub chemistry: u64,
    pub transport: u64,
    pub neural: u64,
    pub expression: u64,
    pub growth: u64,
    pub development: u64,
    pub phylogeny: u64,
    pub audit: u64,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            physics: 1,
            chemistry: 1,
            transport: 1,
            neural: 1,
            expression: 10,
            growth: 20,
            development: 50,
            phylogeny: 500,
            audit: 100,
        }
    }
}

impl Schedule {
    /// Does a global (non-per-entity) phase with interval `every` run this tick?
    #[inline]
    pub fn due(tick: u64, every: u64) -> bool {
        every <= 1 || tick % every == 0
    }

    /// Does entity `id`'s slot of a phase with interval `every` run this tick?
    ///
    /// Entities are spread across the interval by id, so roughly `1/every` of
    /// the population is processed each tick rather than all of them at once.
    /// The entity still gets exactly one update per `every` ticks.
    #[inline]
    pub fn due_for(tick: u64, id: u64, every: u64) -> bool {
        every <= 1 || id % every == tick % every
    }

    /// Effective timestep for a phase running every `every` ticks.
    #[inline]
    pub fn phase_dt(dt: f64, every: u64) -> f64 {
        dt * every.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_tracks_ticks() {
        let mut c = Clock::new(0.01);
        for _ in 0..100 {
            c.advance();
        }
        assert_eq!(c.tick, 100);
        assert!((c.elapsed() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn every_entity_runs_once_per_interval() {
        let every = 10;
        for id in 0..50u64 {
            let hits = (0..every)
                .filter(|t| Schedule::due_for(*t, id, every))
                .count();
            assert_eq!(hits, 1, "entity {id} ran {hits} times in one interval");
        }
    }

    #[test]
    fn stagger_spreads_load_evenly() {
        let every = 10;
        let n = 1000u64;
        for t in 0..every {
            let due = (0..n).filter(|id| Schedule::due_for(t, *id, every)).count();
            assert_eq!(due, (n / every) as usize);
        }
    }

    #[test]
    fn interval_one_always_runs() {
        for t in 0..10 {
            assert!(Schedule::due(t, 1));
            assert!(Schedule::due(t, 0));
            assert!(Schedule::due_for(t, 12345, 1));
        }
    }
}
