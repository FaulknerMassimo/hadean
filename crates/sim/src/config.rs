//! World configuration.
//!
//! A run is reproducible from `(config, tick_count)` alone: the config carries
//! the seed, and every subsystem derives its state from it. Nothing in the
//! simulation reads a wall clock, an environment variable, or a file that is
//! not named here.

use hadean_cell::CellConfig;
use hadean_chem::ChemParams;
use hadean_core::hash::{HashState, StateHasher};
use hadean_core::{Grid, Schedule};
use hadean_fields::flow::FlowConfig;
use hadean_fields::heat::HeatConfig;
use hadean_fields::light::LightConfig;
use serde::{Deserialize, Serialize};

/// Everything needed to build and run a world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorldConfig {
    pub name: String,
    /// Master seed. Every other random draw descends from this.
    pub seed: u64,
    pub grid: GridConfig,
    /// Simulated seconds per tick.
    pub dt: f64,
    pub schedule: Schedule,
    pub chemistry: ChemParams,
    pub light: LightConfig,
    pub heat: HeatConfig,
    pub flow: FlowConfig,
    pub vents: VentConfig,
    pub initial: InitialConfig,
    pub cells: CellConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GridConfig {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    /// Voxel edge length, metres.
    pub dx: f32,
}

impl Default for GridConfig {
    fn default() -> Self {
        // A development-sized pond. The design's target is 200x200x60; that is
        // a GPU-era grid, and this is the CPU reference implementation.
        Self {
            nx: 48,
            ny: 48,
            nz: 20,
            dx: 25.0e-6,
        }
    }
}

impl From<GridConfig> for Grid {
    fn from(c: GridConfig) -> Self {
        Grid::new(c.nx, c.ny, c.nz, c.dx)
    }
}

/// Thermal vents: the world's second energy tap.
///
/// Vents inject heat and *reduced compounds* at the floor. Together with
/// photochemistry at the surface this gives the world two independent ways to
/// make a living, which is what produces divergence rather than a monoculture.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VentConfig {
    /// Particles of vent fuel injected per second, per vent.
    pub fuel_rate: f32,
    /// How many of the chemistry's ranked fuel compounds to inject.
    pub fuel_species: usize,
}

impl Default for VentConfig {
    fn default() -> Self {
        Self {
            fuel_rate: 4.0e9,
            fuel_species: 2,
        }
    }
}

/// What the world starts with.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InitialConfig {
    /// Particles of water per voxel. The solvent everything is dissolved in.
    pub water: f32,
    /// Particles per voxel of every other primordial compound.
    pub primordial: f32,
    /// Whether the starting soup is uniform or noisy. Noise gives the world
    /// spatial structure from tick zero.
    pub noise: f32,
}

impl Default for InitialConfig {
    fn default() -> Self {
        Self {
            water: 2.0e11,
            primordial: 2.0e9,
            noise: 0.25,
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            name: "pond".into(),
            seed: 1,
            grid: GridConfig::default(),
            dt: 0.01,
            schedule: Schedule::default(),
            chemistry: ChemParams::default(),
            light: LightConfig::default(),
            heat: HeatConfig::default(),
            flow: FlowConfig::default(),
            vents: VentConfig::default(),
            initial: InitialConfig::default(),
            cells: CellConfig::default(),
        }
    }
}

impl WorldConfig {
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("config is serialisable")
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Ok(Self::from_toml(&text)?)
    }

    pub fn grid(&self) -> Grid {
        self.grid.into()
    }

    /// Digest of the configuration. Snapshots record it so a reload cannot
    /// silently resume into a differently configured world.
    pub fn digest(&self) -> u64 {
        let mut h = StateHasher::new();
        self.hash_state(&mut h);
        h.finish()
    }

    /// Reject configurations that cannot produce a valid run.
    pub fn validate(&self) -> Result<(), String> {
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(format!("dt must be positive, got {}", self.dt));
        }
        if self.grid.nx == 0 || self.grid.ny == 0 || self.grid.nz == 0 {
            return Err("grid dimensions must all be non-zero".into());
        }
        if self.grid.dx <= 0.0 {
            return Err("dx must be positive".into());
        }
        if self.chemistry.n_compounds < 9 {
            return Err("chemistry needs room for at least the primordials".into());
        }
        if self.initial.water < 0.0 || self.initial.primordial < 0.0 {
            return Err("initial amounts cannot be negative".into());
        }
        if !(0.0..=1.0).contains(&self.initial.noise) {
            return Err("initial noise must be a fraction in 0..1".into());
        }
        self.cells.validate()?;
        Ok(())
    }
}

impl HashState for WorldConfig {
    fn hash_state(&self, h: &mut StateHasher) {
        // Hash the serialised form: it covers every field without this
        // function having to be updated whenever one is added.
        h.str(&self.to_toml());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_toml() {
        let c = WorldConfig::default();
        let text = c.to_toml();
        let back = WorldConfig::from_toml(&text).expect("parses");
        assert_eq!(c, back);
    }

    #[test]
    fn a_partial_config_fills_in_defaults() {
        let text = r#"
            name = "shallow"
            seed = 99
            [grid]
            nz = 8
        "#;
        let c = WorldConfig::from_toml(text).expect("parses");
        assert_eq!(c.name, "shallow");
        assert_eq!(c.seed, 99);
        assert_eq!(c.grid.nz, 8);
        assert_eq!(c.grid.nx, GridConfig::default().nx);
        assert_eq!(c.dt, WorldConfig::default().dt);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        // A silent typo in a config is a lost afternoon.
        let text = r#"
            name = "typo"
            sedd = 4
        "#;
        assert!(WorldConfig::from_toml(text).is_err());
    }

    #[test]
    fn the_digest_tracks_every_field() {
        let a = WorldConfig::default();
        let mut b = a.clone();
        b.seed += 1;
        let mut c = a.clone();
        c.grid.nz += 1;
        let mut d = a.clone();
        d.light.day_length += 1.0;
        assert_ne!(a.digest(), b.digest());
        assert_ne!(a.digest(), c.digest());
        assert_ne!(a.digest(), d.digest());
        assert_eq!(a.digest(), a.clone().digest());
    }

    #[test]
    fn validation_catches_bad_worlds() {
        let ok = WorldConfig::default();
        assert!(ok.validate().is_ok());
        for bad in [
            WorldConfig {
                dt: 0.0,
                ..ok.clone()
            },
            WorldConfig {
                grid: GridConfig { nx: 0, ..ok.grid },
                ..ok.clone()
            },
            WorldConfig {
                initial: InitialConfig {
                    noise: 2.0,
                    ..ok.initial
                },
                ..ok.clone()
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }
}
