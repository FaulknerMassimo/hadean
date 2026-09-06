//! Analysis: time series, drift trends, and run reporting.
//!
//! Everything here reads the simulation; nothing writes to it. It runs off the
//! hot path so a long run can log continuously without slowing down.

pub mod chart;
pub mod drift;
pub mod population;
pub mod series;

pub use chart::chart;
pub use drift::{analyse, Trend};
pub use population::{population_curve, PopulationCurve, Thresholds};
pub use series::{Row, Series};
