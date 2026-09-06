//! Time series recording.
//!
//! Rows go to CSV, which is the format that is still readable in five years
//! and opens in anything. The columns are deliberately flat: one row per
//! reading, no nesting, no schema to remember.

use std::io::Write;
use std::path::Path;

use hadean_sim::audit::AuditReport;
use hadean_sim::World;

/// One row of the run log.
///
/// `Default` is all zeros, which is not a world any run produces -- it exists
/// so tests can name the two or three columns they care about and leave the
/// rest alone as the schema grows.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Row {
    pub tick: u64,
    pub elapsed: f64,
    pub chemical: f64,
    pub thermal: f64,
    pub total: f64,
    pub expected: f64,
    pub drift: f64,
    pub relative_drift: f64,
    pub mass_drift: f64,
    pub light_in: f64,
    pub radiated_out: f64,
    pub vent_heat_in: f64,
    pub mean_temperature: f64,
    pub surface_temperature: f64,
    pub floor_temperature: f64,
    pub surface_light: f64,
    pub population: u64,
    pub decomposing: u64,
    pub births: u64,
    pub deaths: u64,
    pub cellular_reserve: f64,
    /// Particles of the scarcest compound the population's metabolism eats.
    /// The food supply, recorded next to the mouths that depend on it.
    pub substrate: f64,
}

impl Row {
    pub const HEADER: &'static str = "tick,elapsed_s,chemical_j,thermal_j,total_j,expected_j,\
drift_j,relative_drift,mass_drift,light_in_j,radiated_out_j,vent_heat_in_j,\
mean_temperature_k,surface_temperature_k,floor_temperature_k,surface_light_w_m2,\
population,decomposing,births,deaths,cellular_reserve_j,substrate_particles";

    /// Take a reading from a world and its latest audit.
    pub fn sample(world: &World, report: &AuditReport) -> Self {
        let temperature = world.temperature_profile();
        let light = world.light_profile();
        Self {
            tick: report.tick,
            elapsed: world.elapsed(),
            chemical: report.energy.chemical,
            thermal: report.energy.thermal,
            total: report.energy.total(),
            expected: report.expected,
            drift: report.drift,
            relative_drift: report.relative,
            mass_drift: report.mass_drift,
            light_in: world.audit.ledger.light_in,
            radiated_out: world.audit.ledger.radiated_out,
            vent_heat_in: world.audit.ledger.vent_heat_in,
            mean_temperature: world.heat.mean_temperature(),
            surface_temperature: temperature.first().copied().unwrap_or(0.0),
            floor_temperature: temperature.last().copied().unwrap_or(0.0),
            surface_light: light.first().copied().unwrap_or(0.0),
            population: world.cells.alive() as u64,
            decomposing: world.cells.decomposing() as u64,
            births: world.cells.births,
            deaths: world.cells.deaths,
            cellular_reserve: world.cells.total_reserve(),
            substrate: world.limiting_substrate(),
        }
    }

    pub fn to_csv(&self) -> String {
        format!(
            "{},{:.6},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.6e},{:.6e},{:.9e},{:.9e},{:.9e},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.9e},{:.9e}",
            self.tick,
            self.elapsed,
            self.chemical,
            self.thermal,
            self.total,
            self.expected,
            self.drift,
            self.relative_drift,
            self.mass_drift,
            self.light_in,
            self.radiated_out,
            self.vent_heat_in,
            self.mean_temperature,
            self.surface_temperature,
            self.floor_temperature,
            self.surface_light,
            self.population,
            self.decomposing,
            self.births,
            self.deaths,
            self.cellular_reserve,
            self.substrate,
        )
    }
}

/// Collects rows and writes them out.
#[derive(Debug, Clone, Default)]
pub struct Series {
    pub rows: Vec<Row>,
}

impl Series {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, row: Row) {
        self.rows.push(row);
    }

    pub fn record(&mut self, world: &World, report: &AuditReport) {
        self.push(Row::sample(world, report));
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn write_csv(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "{}", Row::HEADER)?;
        for row in &self.rows {
            writeln!(file, "{}", row.to_csv())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hadean_sim::config::{GridConfig, WorldConfig};

    fn world() -> World {
        World::new(WorldConfig {
            seed: 3,
            grid: GridConfig {
                nx: 6,
                ny: 6,
                nz: 5,
                dx: 25.0e-6,
            },
            ..Default::default()
        })
        .expect("builds")
    }

    #[test]
    fn rows_have_as_many_fields_as_the_header() {
        let mut w = world();
        let report = w.run(10).expect("audited");
        let row = Row::sample(&w, &report);
        let header_fields = Row::HEADER.split(',').count();
        let row_fields = row.to_csv().split(',').count();
        assert_eq!(header_fields, row_fields);
    }

    #[test]
    fn a_series_writes_a_readable_file() {
        let mut w = world();
        let mut series = Series::new();
        for _ in 0..3 {
            let report = w.run(5).expect("audited");
            series.record(&w, &report);
        }
        let path = std::env::temp_dir().join("hadean-series-test.csv");
        series.write_csv(&path).expect("writes");
        let text = std::fs::read_to_string(&path).expect("reads");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "header plus three rows");
        assert_eq!(lines[0], Row::HEADER);
        let _ = std::fs::remove_file(&path);
    }
}
