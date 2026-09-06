//! The flow field.
//!
//! Phase 1 uses a static convection roll rather than a fluid solver, which the
//! design calls sufficient through Phase 5. What matters is not that the flow
//! is dynamic but that it is **exactly divergence-free on the discrete grid**:
//! advection is written in conservative form, so any spurious divergence would
//! show up directly as material appearing or vanishing, and the mass audit
//! would blame the chemistry.
//!
//! The trick is to derive face velocities from a stream function sampled at
//! voxel corners. Then each voxel's discrete divergence is a sum of corner
//! values that cancel in pairs, identically, for any stream function at all.
//!
//! A note on the physics: at 25 um and these speeds the Reynolds number is
//! around 1e-3, deep in the Stokes regime, where inertia is irrelevant and
//! swimming is genuinely strange. The design's advice is to accept a
//! prescribed flow plus simple drag rather than make flagellar propulsion a
//! research project, and that is what this is.

use hadean_core::{Axis, Grid};
use serde::{Deserialize, Serialize};

use crate::transport::FaceVelocity;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlowConfig {
    /// Peak speed, m/s.
    pub speed: f32,
    /// Number of counter-rotating cells across the x axis.
    pub rolls: u32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            speed: 2.0e-5,
            rolls: 2,
        }
    }
}

/// Build a convection roll: a divergence-free circulation in the x-z plane,
/// uniform along y.
pub fn convection_roll(grid: &Grid, cfg: &FlowConfig) -> FaceVelocity {
    let mut vel = FaceVelocity::zero(grid);
    if cfg.speed == 0.0 {
        return vel;
    }
    let (nx, ny, nz) = (grid.nx as usize, grid.ny as usize, grid.nz as usize);
    let (lx, _, lz) = grid.extent();
    let rolls = cfg.rolls.max(1) as f32;

    // Amplitude chosen so the peak velocity is roughly `speed`.
    let amplitude = cfg.speed * lz / std::f32::consts::PI;

    // Stream function at voxel corners.
    //
    // It is pinned to exactly zero on the world boundary rather than left to
    // `sin(2*pi)`, which in f32 is about -1.7e-7 and would put a slow leak
    // through the wall. Forcing the corners cannot disturb the divergence:
    // every voxel's divergence is a sum of corner values that cancel in pairs
    // whatever those values are.
    let psi = |i: usize, k: usize| -> f32 {
        if i == 0 || i == nx || k == 0 || k == nz {
            return 0.0;
        }
        let x = i as f32 * grid.dx;
        let z = k as f32 * grid.dx;
        amplitude
            * (rolls * std::f32::consts::PI * x / lx).sin()
            * (std::f32::consts::PI * z / lz).sin()
    };

    // u = d(psi)/dz on faces normal to x; w = -d(psi)/dx on faces normal to z.
    for k in 0..nz {
        for i in 0..=nx {
            let u = (psi(i, k + 1) - psi(i, k)) / grid.dx;
            for j in 0..ny {
                vel.set(Axis::X, i, j, k, u);
            }
        }
    }
    for k in 0..=nz {
        for i in 0..nx {
            let w = -(psi(i + 1, k) - psi(i, k)) / grid.dx;
            for j in 0..ny {
                vel.set(Axis::Z, i, j, k, w);
            }
        }
    }
    vel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_roll_is_divergence_free() {
        let g = Grid::new(24, 6, 12, 25.0e-6);
        let cfg = FlowConfig {
            speed: 5.0e-5,
            rolls: 2,
        };
        let vel = convection_roll(&g, &cfg);
        // Compare against the velocity scale: divergence has units of 1/s.
        let scale = vel.max_speed() / g.dx;
        assert!(
            vel.max_divergence() < scale * 1e-5,
            "divergence {:e} against scale {scale:e}",
            vel.max_divergence()
        );
    }

    #[test]
    fn the_world_boundary_is_closed() {
        let g = Grid::new(16, 4, 8, 25.0e-6);
        let vel = convection_roll(&g, &FlowConfig::default());
        for k in 0..g.nz as usize {
            for j in 0..g.ny as usize {
                assert_eq!(vel.get(Axis::X, 0, j, k), 0.0);
                assert_eq!(vel.get(Axis::X, g.nx as usize, j, k), 0.0);
            }
        }
        for i in 0..g.nx as usize {
            for j in 0..g.ny as usize {
                assert_eq!(vel.get(Axis::Z, i, j, 0), 0.0);
                assert_eq!(vel.get(Axis::Z, i, j, g.nz as usize), 0.0);
            }
        }
    }

    #[test]
    fn speed_tracks_the_configured_scale() {
        let g = Grid::new(32, 4, 16, 25.0e-6);
        let vel = convection_roll(
            &g,
            &FlowConfig {
                speed: 1.0e-4,
                rolls: 1,
            },
        );
        let peak = vel.max_speed();
        assert!((0.5e-4..2.0e-4).contains(&peak), "peak speed {peak:e}");
    }

    #[test]
    fn zero_speed_gives_a_still_world() {
        let g = Grid::new(8, 4, 4, 25.0e-6);
        let vel = convection_roll(
            &g,
            &FlowConfig {
                speed: 0.0,
                rolls: 2,
            },
        );
        assert_eq!(vel.max_speed(), 0.0);
    }
}
