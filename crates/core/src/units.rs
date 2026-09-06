//! The unit system. Everything in the simulation is expressed in these units,
//! and this is the only file allowed to define physical constants.
//!
//! | Quantity     | Unit                       | Type alias  |
//! |--------------|----------------------------|-------------|
//! | Length       | metre (m)                  | `Metres`    |
//! | Time         | second (s)                 | `Seconds`   |
//! | Energy       | joule (J)                  | `Joules`    |
//! | Amount       | particle count (not moles) | `Particles` |
//! | Temperature  | kelvin (K)                 | `Kelvin`    |
//! | Mass         | kilogram (kg)              | `Kilograms` |
//! | Velocity     | m/s                        | `Velocity`  |
//!
//! Amounts are raw particle counts stored as `f32`. At the pond scale a voxel
//! is 25 um on a side (1.56e-14 m^3), so 1e9 particles in a voxel is about
//! 100 uM -- a realistic metabolite concentration. Counts in the 1e7..1e11
//! range are therefore the normal operating band, which `f32` represents with
//! ~7 significant digits. Global reductions (the energy audit) are always
//! accumulated in `f64`; see `hadean_sim::audit`.

/// Length in metres.
pub type Metres = f32;
/// Duration in seconds.
pub type Seconds = f64;
/// Energy in joules.
pub type Joules = f64;
/// Amount as a raw particle count.
pub type Particles = f32;
/// Temperature in kelvin.
pub type Kelvin = f32;
/// Mass in kilograms.
pub type Kilograms = f32;
/// Speed in metres per second.
pub type Velocity = f32;

/// Boltzmann constant, J/K.
pub const KB: f64 = 1.380_649e-23;
/// Avogadro constant, 1/mol.
pub const AVOGADRO: f64 = 6.022_140_76e23;
/// Unified atomic mass unit, kg.
pub const AMU: f64 = 1.660_539_066_60e-27;

/// Reference amount used to non-dimensionalise mass-action kinetics.
/// One `N_REF` in a 25 um voxel is approximately 100 uM.
pub const N_REF: f32 = 1.0e9;

/// Standard ambient temperature, K.
pub const T_AMBIENT: Kelvin = 293.15;

/// Volumetric heat capacity of the medium (water), J/(m^3 K).
pub const HEAT_CAPACITY_VOL: f64 = 4.186e6;

/// Thermal diffusivity of the medium (water), m^2/s.
pub const THERMAL_DIFFUSIVITY: f32 = 1.43e-7;

/// Dynamic viscosity of the medium (water at 20 C), Pa s.
pub const VISCOSITY: f32 = 1.002e-3;

/// Convert kJ/mol to joules per particle.
#[inline]
pub fn kj_per_mol(x: f64) -> Joules {
    x * 1000.0 / AVOGADRO
}

/// Convert joules per particle to kJ/mol (for human-readable output).
#[inline]
pub fn to_kj_per_mol(x: Joules) -> f64 {
    x * AVOGADRO / 1000.0
}

/// Convert an electronvolt to joules.
#[inline]
pub fn ev(x: f64) -> Joules {
    x * 1.602_176_634e-19
}

/// Thermal energy scale kB*T at temperature `t`, in joules.
#[inline]
pub fn kt(t: Kelvin) -> f64 {
    KB * t as f64
}

/// Arrhenius factor `exp(-Ea / kB T)`, clamped to avoid denormal underflow.
///
/// At ambient temperature kB*T is 4.05e-21 J while a typical activation
/// energy is 1e-19 J, so this is ~1e-11: uncatalysed reactions run at
/// effectively zero rate. Enzymes matter because they lower `ea`.
#[inline]
pub fn arrhenius(ea: f64, t: Kelvin) -> f64 {
    if ea <= 0.0 {
        return 1.0;
    }
    let x = -ea / kt(t);
    if x < -700.0 {
        0.0
    } else {
        x.exp()
    }
}

/// Convert an amount (particles) in a voxel to a number concentration (1/m^3).
#[inline]
pub fn number_density(amount: Particles, voxel_volume: f32) -> f32 {
    amount / voxel_volume
}

/// Convert an amount in a voxel to molar concentration (mol/L), for display.
pub fn molar(amount: Particles, voxel_volume: f32) -> f64 {
    let per_m3 = amount as f64 / voxel_volume as f64;
    per_m3 / AVOGADRO / 1000.0
}
