//! Drift trend analysis.
//!
//! A small constant drift in the energy audit is float arithmetic. A drift
//! that *grows* is a leak, and a leak is the failure the design warns about
//! most sharply: organisms will find it and evolve into perpetual motion
//! machines. The distinction is a trend, not a threshold, so it is worth
//! measuring properly rather than eyeballing the last number.
//!
//! The test is an ordinary least-squares fit of drift against tick, and it
//! asks two questions of the fit, because either alone gives a wrong answer.
//!
//! **Is the slope real?** A drift that jitters around zero has a slope
//! indistinguishable from its own scatter. A leak has a slope that dominates.
//!
//! **Would it matter?** This one was learned the hard way. Once the residuals
//! were fixed, the remaining drift stopped being white noise and became a
//! smooth, bounded wander -- f32 accumulation tracking the state of a world
//! that is itself evolving smoothly. A smooth curve fits a line with almost no
//! scatter, so the first test alone fires on it every time, and the sign of
//! the "leak" flips depending on how long you watch. What separates that from
//! a real leak is size: extrapolate the fitted slope to the million ticks the
//! design's Phase 1 gate names, and ask whether the implied drift would breach
//! the audit tolerance. The leak this module was built to catch projects to
//! about 1e-5 relative over that horizon. The wander that replaced it projects
//! to 1e-10, five orders of magnitude clear.

use crate::series::Series;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trend {
    /// Joules of drift accumulated per tick.
    pub slope: f64,
    /// Drift at tick zero, from the fit.
    pub intercept: f64,
    /// Root-mean-square residual around the fit, J.
    pub scatter: f64,
    /// Largest absolute drift seen, as a fraction of total energy.
    pub worst_relative: f64,
    /// Mean magnitude of the world's total energy over the samples: the scale
    /// a projected drift is judged against. One when the series carries no
    /// energy, so a slope is then read as plain joules.
    pub scale: f64,
    pub samples: usize,
}

/// Ticks a fitted slope is projected over when asking whether it matters.
/// The design's Phase 1 gate is an audit that stays flat across a million
/// ticks, so that is the horizon a slope has to survive.
pub const PROJECTION_TICKS: u64 = 1_000_000;

/// Relative drift at the projection horizon that counts as a leak. Matches
/// the default tolerance `hadean verify` holds the audit itself to.
pub const MATERIAL_DRIFT: f64 = 1.0e-6;

impl Trend {
    /// Is the fitted slope bigger than the scatter it sits in?
    ///
    /// True when the fitted change across the whole observed span is several
    /// times the scatter -- necessary for a leak, but not sufficient, because
    /// any smooth curve passes it.
    pub fn is_significant(&self, span_ticks: u64) -> bool {
        if self.samples < 4 {
            return false;
        }
        let fitted_change = (self.slope * span_ticks as f64).abs();
        fitted_change > 4.0 * self.scatter.max(f64::MIN_POSITIVE)
    }

    /// Relative drift this slope implies after [`PROJECTION_TICKS`].
    pub fn projected_relative(&self) -> f64 {
        let projected = (self.slope * PROJECTION_TICKS as f64).abs();
        if self.scale > 0.0 {
            projected / self.scale
        } else {
            projected
        }
    }

    /// Is the drift growing rather than jittering?
    ///
    /// Both tests: the slope has to stand out from the scatter *and* have to
    /// matter at the horizon the design cares about.
    pub fn is_growing(&self, span_ticks: u64) -> bool {
        self.is_significant(span_ticks) && self.projected_relative() > MATERIAL_DRIFT
    }

    /// One-line verdict for a run report.
    pub fn verdict(&self, span_ticks: u64) -> String {
        if self.samples < 4 {
            "too few samples".into()
        } else if self.is_growing(span_ticks) {
            format!(
                "GROWING - energy is leaking ({:.1e} relative per {} ticks)",
                self.projected_relative(),
                PROJECTION_TICKS
            )
        } else if self.is_significant(span_ticks) {
            format!(
                "flat - a smooth {:.1e} relative per {} ticks, below the {:.0e} gate",
                self.projected_relative(),
                PROJECTION_TICKS,
                MATERIAL_DRIFT
            )
        } else {
            "flat - within float noise".into()
        }
    }
}

/// Fit drift against tick.
pub fn analyse(series: &Series) -> Trend {
    let n = series.rows.len();
    if n < 2 {
        return Trend {
            slope: 0.0,
            intercept: series.rows.first().map_or(0.0, |r| r.drift),
            scatter: 0.0,
            worst_relative: series.rows.first().map_or(0.0, |r| r.relative_drift.abs()),
            scale: series
                .rows
                .first()
                .map_or(1.0, |r| r.total.abs())
                .max(1.0e-300),
            samples: n,
        };
    }

    let xs: Vec<f64> = series.rows.iter().map(|r| r.tick as f64).collect();
    let ys: Vec<f64> = series.rows.iter().map(|r| r.drift).collect();
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;

    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        sxy += dx * (ys[i] - mean_y);
        sxx += dx * dx;
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let intercept = mean_y - slope * mean_x;

    let residual: f64 = (0..n)
        .map(|i| {
            let e = ys[i] - (intercept + slope * xs[i]);
            e * e
        })
        .sum::<f64>()
        / n as f64;

    Trend {
        slope,
        intercept,
        scatter: residual.sqrt(),
        worst_relative: series
            .rows
            .iter()
            .map(|r| r.relative_drift.abs())
            .fold(0.0, f64::max),
        scale: {
            let mean = series.rows.iter().map(|r| r.total.abs()).sum::<f64>() / n as f64;
            if mean > 0.0 {
                mean
            } else {
                1.0
            }
        },
        samples: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Row;

    fn row(tick: u64, drift: f64) -> Row {
        Row {
            tick,
            elapsed: tick as f64 * 0.01,
            chemical: -1.0,
            total: -1.0,
            expected: -1.0 - drift,
            drift,
            relative_drift: drift,
            ..Default::default()
        }
    }

    fn series_from(drifts: &[f64]) -> Series {
        let mut s = Series::new();
        for (i, &d) in drifts.iter().enumerate() {
            s.push(row(i as u64 * 100, d));
        }
        s
    }

    #[test]
    fn a_steady_leak_is_detected() {
        let s = series_from(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let t = analyse(&s);
        assert!(t.slope > 0.0);
        assert!(t.scatter < 1e-9, "a perfect line should have no scatter");
        assert!(t.is_growing(700));
        assert!(t.verdict(700).starts_with("GROWING"));
    }

    #[test]
    fn float_noise_is_not_a_leak() {
        // Alternating sign, no trend: exactly what rounding looks like.
        let s = series_from(&[1e-9, -1e-9, 1.2e-9, -0.8e-9, 0.9e-9, -1.1e-9, 1e-9, -1e-9]);
        let t = analyse(&s);
        assert!(
            !t.is_growing(700),
            "slope {:e} scatter {:e}",
            t.slope,
            t.scatter
        );
        assert_eq!(t.verdict(700), "flat - within float noise");
    }

    #[test]
    fn a_negative_leak_is_also_caught() {
        let s = series_from(&[0.0, -1.0, -2.0, -3.0, -4.0, -5.0]);
        let t = analyse(&s);
        assert!(t.slope < 0.0);
        assert!(t.is_growing(500));
    }

    #[test]
    fn too_few_samples_gives_no_verdict() {
        let s = series_from(&[0.0, 5.0]);
        let t = analyse(&s);
        assert!(!t.is_growing(100));
        assert_eq!(t.verdict(100), "too few samples");
        assert_eq!(analyse(&Series::new()).samples, 0);
    }

    /// A series at the pond's real energy scale, so projections mean what
    /// they mean in a run: 1.5 mJ of world, sampled every hundred ticks.
    fn at_pond_scale(slope_per_tick: f64, wobble: f64) -> Series {
        let mut s = Series::new();
        for i in 0..40u64 {
            let tick = i * 100;
            let drift = slope_per_tick * tick as f64 + wobble * ((i % 3) as f64 - 1.0);
            s.push(Row {
                tick,
                drift,
                total: -1.54e-3,
                relative_drift: drift / 1.54e-3,
                ..Default::default()
            });
        }
        s
    }

    #[test]
    fn the_leak_this_module_was_built_for_is_still_caught() {
        // Before the residuals were introduced the audit bled about 1.2e-10 J
        // over 8000 ticks. That is 1.5e-14 J/tick, which projects to 1e-5 of
        // the pond's energy across a million ticks -- ten times the gate.
        let t = analyse(&at_pond_scale(1.5e-14, 0.0));
        assert!(t.is_significant(4000));
        assert!(
            t.projected_relative() > MATERIAL_DRIFT,
            "projected {:e}",
            t.projected_relative()
        );
        assert!(t.is_growing(4000));
        assert!(t.verdict(4000).starts_with("GROWING"));
    }

    #[test]
    fn a_smooth_but_immaterial_wander_is_not_called_a_leak() {
        // What the audit actually does now: a clean line with almost no
        // scatter, at 1e-19 J/tick. It fits a trend beautifully and projects
        // to 1e-10 relative, which is five orders below the gate. Judging it
        // on significance alone fails every long run for nothing.
        let t = analyse(&at_pond_scale(1.0e-19, 0.0));
        assert!(t.is_significant(4000), "the fit is clean, and that is fine");
        assert!(!t.is_growing(4000));
        assert!(t.verdict(4000).starts_with("flat"), "{}", t.verdict(4000));
    }

    #[test]
    fn worst_relative_is_the_largest_magnitude() {
        let s = series_from(&[1e-9, -5e-9, 2e-9]);
        assert_eq!(analyse(&s).worst_relative, 5e-9);
    }
}
