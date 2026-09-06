//! **L0 -- Substrate.** Units, deterministic randomness, the voxel grid, the
//! fixed-timestep clock, and state hashing.
//!
//! Nothing in this crate knows what a compound, a cell, or a genome is. Every
//! layer above depends on it; it depends on nothing but `serde`.
//!
//! The invariant this crate exists to protect is **determinism**: the same
//! `(seed, config, tick_count)` must produce a bit-identical world on every
//! run, on every thread count. That requires all three of:
//!
//! 1. Counter-based RNG ([`rng::Counter`]) so draws do not depend on order.
//! 2. Reductions in a fixed order, or in `f64`/fixed-point, because parallel
//!    float summation is not associative.
//! 3. No wall-clock reads, no uninitialised memory, and no iteration over
//!    containers whose order depends on pointer or address values.

pub mod grid;
pub mod hash;
pub mod rng;
pub mod time;
pub mod units;

pub use grid::{Axis, Grid, VoxelId};
pub use hash::{HashState, StateHasher};
pub use rng::{Counter, Purpose, SeqRng};
pub use time::{Clock, Schedule};
