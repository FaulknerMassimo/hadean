//! Reading a population curve.
//!
//! The Phase 2 gate in the design is not "cells exist". It is a *shape*: a
//! population that grows into its food supply, overshoots it, crashes, and
//! then comes back on what the crash left behind. That shape has to be found
//! in the data rather than asserted, because the whole point is that nobody
//! wrote the logistic curve -- it has to fall out of membranes, kinetics and
//! starvation.
//!
//! So this module does the boring, falsifiable version of eyeballing a graph.
//! It walks the population column and picks out four landmarks:
//!
//! ```text
//!            peak
//!             /\            recovery
//!            /  \            /\/
//!   start __/    \          /
//!                 \        /
//!                  \______/
//!                    trough
//! ```
//!
//! and then asks whether each leg is big enough to be a phase rather than
//! jitter. Two failure modes get their own verdicts, because both are easy to
//! mistake for success:
//!
//! * **Cap-limited.** A population pinned at its safety cap has not found its
//!   carrying capacity, it has found a `usize` the author typed. The cap is
//!   there to stop a runaway allocating the machine to death; a run that
//!   touches it proves nothing about resources.
//! * **Extinct.** A curve that booms and then goes to zero has a boom and a
//!   bust and no ecosystem.

use crate::series::Series;

/// A landmark on the population curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mark {
    pub tick: u64,
    pub population: u64,
}

/// How much of a change counts as a phase rather than noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// The peak must be at least this many times the starting population.
    pub boom: f64,
    /// The trough must fall to at most this fraction of the peak.
    pub bust: f64,
    /// The recovery must reach at least this many times the trough.
    pub recovery: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        // A quadrupling, a halving, and a doubling. Deliberately modest: the
        // gate is about the shape being real, not about it being dramatic.
        Self {
            boom: 4.0,
            bust: 0.5,
            recovery: 2.0,
        }
    }
}

/// What a population did over a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationCurve {
    pub start: u64,
    /// The highest population reached, and when.
    pub peak: Mark,
    /// The lowest population *after* the peak.
    pub trough: Mark,
    /// The highest population after the trough.
    pub recovery: Mark,
    pub finish: u64,
    /// The configured safety cap, and whether the run ever reached it.
    pub cap: u64,
    pub cap_limited: bool,
    /// Food in the pond at the start over food at the peak. Above one means
    /// the boom really did eat its own supply down rather than peaking for
    /// some unrelated reason.
    pub substrate_drawdown: f64,
    /// Food at the trough over food at the peak. Above one means the crash
    /// let the larder refill -- which is what the recovery then spends.
    pub substrate_rebound: f64,
    /// Cells born after the peak.
    ///
    /// The recovery leg cannot happen without this, and for a long time it was
    /// exactly zero: every run of the gate ended with `births` equal to the
    /// peak population, because a population of clones all reach break-even
    /// together and none of them ever has the surplus to divide again. That is
    /// a freeze rather than a carrying capacity, and it looks identical to one
    /// in the population column -- so the number gets reported whether the run
    /// passes or not.
    pub births_after_peak: u64,
    pub samples: usize,
    pub thresholds: Thresholds,
}

impl PopulationCurve {
    pub fn boomed(&self) -> bool {
        self.start > 0 && self.peak.population as f64 >= self.thresholds.boom * self.start as f64
    }

    pub fn busted(&self) -> bool {
        self.peak.population > 0
            && self.trough.tick > self.peak.tick
            && (self.trough.population as f64) <= self.thresholds.bust * self.peak.population as f64
    }

    pub fn recovered(&self) -> bool {
        self.trough.population > 0
            && self.recovery.tick > self.trough.tick
            && self.recovery.population as f64
                >= self.thresholds.recovery * self.trough.population as f64
    }

    pub fn extinct(&self) -> bool {
        self.finish == 0
    }

    /// Did the run demonstrate the Phase 2 gate?
    ///
    /// All three legs, without ever leaning on the safety cap, and with
    /// something still alive at the end.
    pub fn passes(&self) -> bool {
        self.boomed() && self.busted() && self.recovered() && !self.cap_limited && !self.extinct()
    }

    /// One line for a run report.
    pub fn verdict(&self) -> String {
        if self.samples < 4 {
            return "too few samples".into();
        }
        if self.cap_limited {
            return format!(
                "CAP-LIMITED - the population sat on its safety cap of {}, so nothing here is \
                 about resources",
                self.cap
            );
        }
        if self.extinct() {
            return "EXTINCT - the population did not survive the run".into();
        }
        if !self.boomed() {
            return format!(
                "no boom - peaked at {} from {}, short of {:.0}x",
                self.peak.population, self.start, self.thresholds.boom
            );
        }
        if !self.busted() {
            return format!(
                "no bust - held at {} after peaking at {}",
                self.trough.population, self.peak.population
            );
        }
        if !self.recovered() {
            return format!(
                "no recovery - still at {} since the crash to {}, on {} births since the peak",
                self.recovery.population, self.trough.population, self.births_after_peak
            );
        }
        format!(
            "boom/bust/recovery - {} -> {} -> {} -> {}",
            self.start, self.peak.population, self.trough.population, self.recovery.population
        )
    }
}

/// `numerator / denominator`, or 1.0 when there is nothing to compare --
/// a world with no metabolism reports no food, and that is not a drawdown.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 && numerator.is_finite() && denominator.is_finite() {
        numerator / denominator
    } else {
        1.0
    }
}

/// Find the boom, the bust, and the recovery in a run's population column.
pub fn population_curve(series: &Series, cap: u64, thresholds: Thresholds) -> PopulationCurve {
    let rows = &series.rows;
    let empty = PopulationCurve {
        start: 0,
        peak: Mark::default(),
        trough: Mark::default(),
        recovery: Mark::default(),
        finish: 0,
        cap,
        cap_limited: false,
        substrate_drawdown: 1.0,
        substrate_rebound: 1.0,
        births_after_peak: 0,
        samples: rows.len(),
        thresholds,
    };
    if rows.is_empty() {
        return empty;
    }

    let mark = |i: usize| Mark {
        tick: rows[i].tick,
        population: rows[i].population,
    };
    // Ties go to the earliest sample, so a plateau is dated from where it
    // began. That matters for the peak: the bust has to come after it, and a
    // late-dated peak can hide a crash inside its own plateau.
    let extremum = |from: usize, to: usize, want_max: bool| -> usize {
        let mut best = from;
        for i in from..to {
            let better = if want_max {
                rows[i].population > rows[best].population
            } else {
                rows[i].population < rows[best].population
            };
            if better {
                best = i;
            }
        }
        best
    };

    // Ancestors arrive after the pond has had time to make them something to
    // eat, so the run opens with samples of an empty world. Those are not a
    // population of zero that then boomed -- the boom is measured from the
    // founding cohort.
    let founding = rows.iter().position(|r| r.population > 0).unwrap_or(0);

    let build = |peak: usize| -> PopulationCurve {
        let trough = extremum(peak, rows.len(), false);
        let recovery = extremum(trough, rows.len(), true);
        PopulationCurve {
            start: rows[founding].population,
            peak: mark(peak),
            trough: mark(trough),
            recovery: mark(recovery),
            finish: rows[rows.len() - 1].population,
            cap,
            cap_limited: cap > 0 && rows.iter().any(|r| r.population >= cap),
            substrate_drawdown: ratio(rows[founding].substrate, rows[peak].substrate),
            substrate_rebound: ratio(rows[trough].substrate, rows[peak].substrate),
            births_after_peak: rows[rows.len() - 1]
                .births
                .saturating_sub(rows[peak].births),
            samples: rows.len(),
            ..empty
        }
    };

    // The question is whether the run ever showed the shape, so every sample
    // gets a turn as the peak. Taking the highest sample instead reads a
    // world that cycles -- and one lit by a sun that rises and sets is such a
    // world -- from wherever its very best day happened to fall. If that is
    // the last cycle, there is no room left in the run for the crash and the
    // recovery that in fact happened twice already.
    let highest = build(extremum(founding, rows.len(), true));
    if highest.passes() {
        return highest;
    }
    // Otherwise take the biggest cycle that did complete, and fall back to the
    // highest peak for reporting when none did -- its verdict names whichever
    // leg was missing.
    (founding..rows.len())
        .map(build)
        .filter(PopulationCurve::passes)
        .max_by_key(|c| c.peak.population)
        .unwrap_or(highest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Row;

    fn curve_of(populations: &[u64], cap: u64) -> PopulationCurve {
        let mut series = Series::new();
        for (i, &population) in populations.iter().enumerate() {
            series.push(Row {
                tick: i as u64 * 100,
                elapsed: i as f64,
                population,
                // Food falls as the population rises, which is the point.
                substrate: 1.0e12 / (population.max(1) as f64),
                ..Default::default()
            });
        }
        population_curve(&series, cap, Thresholds::default())
    }

    #[test]
    fn a_boom_bust_recovery_is_recognised() {
        let c = curve_of(&[10, 40, 200, 900, 300, 60, 120, 340, 300], 10_000);
        assert_eq!(c.peak.population, 900);
        assert_eq!(c.peak.tick, 300);
        assert_eq!(c.trough.population, 60);
        assert_eq!(c.recovery.population, 340);
        assert!(c.boomed() && c.busted() && c.recovered());
        assert!(c.passes(), "{}", c.verdict());
        assert!(c.substrate_drawdown > 1.0, "the boom did not eat anything");
        assert!(
            c.substrate_rebound > 1.0,
            "the crash did not refill the larder"
        );
    }

    #[test]
    fn the_boom_is_measured_from_the_founding_cohort() {
        // The pond runs lifeless while its photochemistry lays a table, so
        // the first samples are of an empty world. Reading those as the
        // starting population makes every boom infinite and every curve fail
        // for the wrong reason.
        let c = curve_of(&[0, 0, 0, 24, 90, 400, 900, 300, 60, 120, 340, 300], 10_000);
        assert_eq!(c.start, 24);
        assert_eq!(c.peak.population, 900);
        assert!(c.passes(), "{}", c.verdict());
    }

    #[test]
    fn a_run_that_leans_on_its_cap_proves_nothing() {
        let c = curve_of(&[10, 40, 200, 900, 300, 60, 120, 340, 300], 900);
        assert!(c.cap_limited);
        assert!(!c.passes());
        assert!(c.verdict().starts_with("CAP-LIMITED"));
    }

    #[test]
    fn plain_logistic_growth_is_not_a_bust() {
        let c = curve_of(&[10, 40, 200, 600, 880, 900, 905, 903, 904], 10_000);
        assert!(c.boomed());
        assert!(!c.busted(), "a plateau is not a crash");
        assert!(!c.passes());
    }

    #[test]
    fn a_cycling_population_is_read_from_a_cycle_that_completed() {
        // Three day/night cycles, each bigger than the last. The global peak
        // is in the third, which the run ends inside -- so judging from it
        // alone finds a boom, no bust and no recovery, and calls a textbook
        // oscillation a failure.
        let c = curve_of(
            &[20, 300, 900, 200, 50, 400, 1200, 260, 70, 600, 1600, 1400],
            10_000,
        );
        assert!(c.passes(), "{}", c.verdict());
        assert!(
            c.peak.population < 1600,
            "the judged peak {} was the unfinished one",
            c.peak.population
        );
        assert!(c.trough.tick > c.peak.tick);
        assert!(c.recovery.tick > c.trough.tick);
    }

    #[test]
    fn extinction_is_not_a_recovery() {
        let c = curve_of(&[10, 40, 200, 900, 300, 60, 0, 0, 0], 10_000);
        assert!(c.boomed() && c.busted());
        assert!(c.extinct());
        assert!(!c.passes());
    }

    #[test]
    fn a_peak_is_dated_from_where_its_plateau_began() {
        // The crash has to be found after the peak, so a flat top must not
        // let the peak slide to the right of it.
        let c = curve_of(&[10, 400, 400, 400, 50, 200], 10_000);
        assert_eq!(c.peak.tick, 100);
        assert!(c.busted() && c.recovered());
    }

    #[test]
    fn an_empty_series_says_so_instead_of_panicking() {
        let c = curve_of(&[], 10_000);
        assert_eq!(c.samples, 0);
        assert!(!c.passes());
        assert_eq!(c.verdict(), "too few samples");
    }
}
