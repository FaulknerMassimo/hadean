//! Text charts.
//!
//! There is no viewer yet, and a population curve is the one thing in this
//! project that is genuinely hard to read as a column of numbers -- the whole
//! Phase 2 gate is a claim about a *shape*. A terminal plot is not a
//! substitute for volumetric rendering, but it is the difference between
//! "population fell from 8100 to 1200 over forty samples" and seeing the
//! overshoot.
//!
//! Two rules keep these honest. Each chart carries its own axis labels, so a
//! curve is never read against an invisible scale. And two quantities on the
//! same axes would need a shared scale to mean anything, so stacked charts
//! sharing an x axis are drawn instead of one overlay with two normalisations.

use std::fmt::Write as _;

/// Empty, then the eight vertical block glyphs, so a column resolves to an
/// eighth of a row rather than to a whole one.
const BARS: [char; 9] = [
    ' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
    '\u{2588}',
];

/// Draw `values` as a filled bar chart `width` columns by `height` rows.
///
/// Values are bucketed by mean when there are more of them than columns, and
/// the y axis always includes zero: a population that halves should look
/// halved, which it does not on an axis that starts at the minimum.
pub fn chart(label: &str, values: &[f64], width: usize, height: usize) -> String {
    let width = width.max(4);
    let height = height.max(2);
    if values.is_empty() {
        return format!("{label}: no data\n");
    }

    let columns = bucket(values, width);
    let top = columns
        .iter()
        .copied()
        .fold(0.0f64, |a, b| if b.is_finite() { a.max(b) } else { a });
    if top <= 0.0 {
        return format!("{label}: flat at {:.3e}\n", values[0]);
    }

    // Each row is one eighth-resolved band of the range.
    let mut out = String::new();
    let gutter = 10;
    for row in 0..height {
        let ceiling = top * (height - row) as f64 / height as f64;
        let floor = top * (height - row - 1) as f64 / height as f64;
        let _ = write!(out, "{:>gutter$.2e} |", ceiling);
        for &v in &columns {
            let eighths = ((v - floor) / (ceiling - floor) * 8.0).round();
            let eighths = if eighths.is_finite() { eighths } else { 0.0 };
            out.push(BARS[eighths.clamp(0.0, 8.0) as usize]);
        }
        out.push('\n');
    }
    let _ = write!(out, "{:>gutter$} +", "0");
    for _ in 0..columns.len() {
        out.push('-');
    }
    let _ = writeln!(out, "  {label}");
    out
}

/// Average `values` down to at most `width` columns.
fn bucket(values: &[f64], width: usize) -> Vec<f64> {
    if values.len() <= width {
        return values.to_vec();
    }
    (0..width)
        .map(|i| {
            let from = i * values.len() / width;
            let to = ((i + 1) * values.len() / width).max(from + 1);
            let slice = &values[from..to.min(values.len())];
            slice.iter().sum::<f64>() / slice.len() as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chart_has_the_shape_it_was_given() {
        let rising: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let text = chart("rising", &rising, 20, 6);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 7, "six rows and an axis");
        // The top row is filled on the right and empty on the left.
        let top = lines[0];
        let bars = top.split('|').nth(1).expect("bars");
        assert!(bars.starts_with(' '), "top row should start empty: {top:?}");
        assert!(
            bars.ends_with('\u{2588}'),
            "top row should end full: {top:?}"
        );
        assert!(text.contains("rising"));
    }

    #[test]
    fn buckets_average_rather_than_drop_samples() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let columns = bucket(&values, 10);
        assert_eq!(columns.len(), 10);
        assert!((columns[0] - 4.5).abs() < 1e-9, "{:?}", columns[0]);
        assert!((columns[9] - 94.5).abs() < 1e-9, "{:?}", columns[9]);
    }

    #[test]
    fn degenerate_input_does_not_panic() {
        assert!(chart("empty", &[], 20, 6).contains("no data"));
        assert!(chart("flat", &[0.0, 0.0, 0.0], 20, 6).contains("flat"));
        // A single sample, and a width narrower than the minimum.
        assert!(!chart("one", &[3.0], 1, 1).is_empty());
    }
}
