//! Wave channels: one interface over every way information travels.
//!
//! The design's instinct was that light and sound are "the same thing at
//! different frequencies". They are not -- sound is a mechanical pressure wave
//! in a medium, light is electromagnetic, and they do not sit on one spectrum.
//! But the *useful* half of that instinct is right: what a sensor protein
//! needs from a channel is always the same three things -- how much is
//! arriving, in which band, and from which direction. So that is what gets
//! unified here, while the propagation physics stays honest and different per
//! channel.
//!
//! | Channel      | Propagation             | Speed     | Sense         |
//! |--------------|-------------------------|-----------|---------------|
//! | [`Chemical`] | diffusion (L1)          | very slow | smell, taste  |
//! | [`Light`]    | Beer-Lambert columns    | instant   | sight         |
//! | [`Mechanical`] | event list, 1/r       | fast      | hearing, touch|
//!
//! Sensor and effector proteins will be written against [`WaveChannel`], not
//! against any implementation, so a genome that evolves "emit into channel X
//! at band B" does not care which physics is underneath.
//!
//! On acoustics specifically: a grid-based FDTD solver is not viable here. At
//! `dx` = 25 um a realistic speed of sound needs `dt` around 1e-8 s, a million
//! substeps per chemistry tick. But at pond scale the propagation delay is
//! nanoseconds -- physically irrelevant. What matters at the receiver is
//! amplitude, frequency and direction, so [`Mechanical`] models emitters as
//! events and integrates them directly. That is cheap and, at this scale,
//! more accurate than a grid would be.

use hadean_chem::chemistry::N_BANDS;
use hadean_core::Grid;

use crate::light::LightField;
use crate::scalar::ChemField;

/// What a receptor reads: how much is arriving, and which way it is falling off.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sample {
    /// Summed intensity over the requested bands, in the channel's own units.
    pub intensity: f32,
    /// Spatial gradient of that intensity, per metre. Points up-gradient, so a
    /// chemotactic cell swims along it.
    pub gradient: [f32; 3],
}

impl Sample {
    /// Magnitude of the gradient.
    pub fn slope(&self) -> f32 {
        let [x, y, z] = self.gradient;
        (x * x + y * y + z * z).sqrt()
    }
}

/// A medium that carries banded information from emitters to receivers.
pub trait WaveChannel {
    /// Advance the channel's own state. Channels whose propagation is already
    /// paid for elsewhere -- diffusion for chemicals, the radiative solver for
    /// light -- do nothing here.
    fn propagate(&mut self, dt: f64);

    /// Inject signal at a world position.
    fn emit(&mut self, position: [f32; 3], band: u8, amplitude: f32);

    /// Read the channel at a world position, over an inclusive band range.
    fn sample(&self, position: [f32; 3], bands: (u8, u8)) -> Sample;

    /// How many bands this channel has.
    fn band_count(&self) -> usize;
}

/// Central-difference gradient of a per-voxel quantity, per metre.
fn gradient_at(grid: &Grid, voxel: usize, value: impl Fn(usize) -> f32) -> [f32; 3] {
    use hadean_core::Axis;
    let mut g = [0.0f32; 3];
    for (k, axis) in Axis::ALL.into_iter().enumerate() {
        let lo = grid.neighbour(voxel, axis, -1);
        let hi = grid.neighbour(voxel, axis, 1);
        g[k] = match (lo, hi) {
            (Some(a), Some(b)) => (value(b) - value(a)) / (2.0 * grid.dx),
            // One-sided at a wall, so a cell against the boundary still reads
            // a usable gradient rather than zero.
            (None, Some(b)) => (value(b) - value(voxel)) / grid.dx,
            (Some(a), None) => (value(voxel) - value(a)) / grid.dx,
            (None, None) => 0.0,
        };
    }
    g
}

/// Chemical signalling: diffusible compounds. Bands are compound ids.
///
/// Propagation is already paid for by the chemistry layer's diffusion pass,
/// which is why this is the cheapest channel in the simulation -- and, at
/// micrometre scale with a Peclet number well below one, the one that actually
/// dominates biology.
pub struct Chemical<'a> {
    pub grid: &'a Grid,
    pub field: &'a mut ChemField,
}

impl WaveChannel for Chemical<'_> {
    fn propagate(&mut self, _dt: f64) {}

    fn emit(&mut self, position: [f32; 3], band: u8, amplitude: f32) {
        let voxel = self.grid.voxel_at(position);
        if (band as usize) < self.field.n_compounds {
            self.field.add(band as usize, voxel, amplitude);
        }
    }

    fn sample(&self, position: [f32; 3], bands: (u8, u8)) -> Sample {
        let voxel = self.grid.voxel_at(position);
        let (lo, hi) = (
            bands.0 as usize,
            (bands.1 as usize).min(self.field.n_compounds - 1),
        );
        let total = |v: usize| -> f32 { (lo..=hi).map(|c| self.field.get(c, v)).sum() };
        Sample {
            intensity: total(voxel),
            gradient: gradient_at(self.grid, voxel, total),
        }
    }

    fn band_count(&self) -> usize {
        self.field.n_compounds
    }
}

/// Sight. Bands are spectral bands; intensity is irradiance in W/m^2.
///
/// Long-range transport is the radiative solver's job. What [`emit`] adds are
/// local sources -- bioluminescence, fluorescence -- which get a bounded-radius
/// `1/r^2` scatter rather than full transport. At pond scale that is plenty.
///
/// [`emit`]: WaveChannel::emit
pub struct Light<'a> {
    pub grid: &'a Grid,
    pub field: &'a mut LightField,
    /// Radius of an emitter's influence, in voxels.
    pub emit_radius: u32,
}

impl<'a> Light<'a> {
    pub fn new(grid: &'a Grid, field: &'a mut LightField) -> Self {
        Self {
            grid,
            field,
            emit_radius: 4,
        }
    }
}

impl WaveChannel for Light<'_> {
    fn propagate(&mut self, _dt: f64) {}

    fn emit(&mut self, position: [f32; 3], band: u8, amplitude: f32) {
        let band = band as usize;
        if band >= N_BANDS || amplitude <= 0.0 {
            return;
        }
        let source = self.grid.voxel_at(position);
        let (sx, sy, sz) = self.grid.coords(source);
        let r = self.emit_radius as i64;
        let base = band * self.field.n_voxels;

        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y, z) = (sx as i64 + dx, sy as i64 + dy, sz as i64 + dz);
                    if x < 0 || y < 0 || z < 0 {
                        continue;
                    }
                    let (x, y, z) = (x as u32, y as u32, z as u32);
                    if x >= self.grid.nx || y >= self.grid.ny || z >= self.grid.nz {
                        continue;
                    }
                    let d2 = (dx * dx + dy * dy + dz * dz) as f32;
                    // Half a voxel softening keeps the source itself finite.
                    let falloff = 1.0 / (d2 + 0.25);
                    let voxel = self.grid.idx(x, y, z);
                    self.field.intensity[base + voxel] += amplitude * falloff;
                }
            }
        }
    }

    fn sample(&self, position: [f32; 3], bands: (u8, u8)) -> Sample {
        let voxel = self.grid.voxel_at(position);
        let (lo, hi) = (bands.0 as usize, (bands.1 as usize).min(N_BANDS - 1));
        let total = |v: usize| -> f32 { (lo..=hi).map(|b| self.field.intensity_at(b, v)).sum() };
        Sample {
            intensity: total(voxel),
            gradient: gradient_at(self.grid, voxel, total),
        }
    }

    fn band_count(&self) -> usize {
        N_BANDS
    }
}

/// One mechanical disturbance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pulse {
    pub position: [f32; 3],
    pub band: u8,
    pub amplitude: f32,
    /// Seconds of life remaining.
    pub remaining: f64,
}

/// Hearing and touch, as an event list rather than a grid.
///
/// Emitters are events; a receiver integrates their contributions with `1/r`
/// attenuation and exponential absorption. Cost is proportional to nearby
/// emitters, not to the volume of the world.
#[derive(Debug, Clone, Default)]
pub struct Mechanical {
    pub pulses: Vec<Pulse>,
    /// How long a pulse stays audible, seconds.
    pub lifetime: f64,
    /// Absorption coefficient of the medium, 1/m.
    pub absorption: f32,
    /// Bands the channel distinguishes.
    pub bands: usize,
}

impl Mechanical {
    pub fn new(bands: usize) -> Self {
        Self {
            pulses: Vec::new(),
            lifetime: 0.05,
            absorption: 60.0,
            bands,
        }
    }
}

impl WaveChannel for Mechanical {
    fn propagate(&mut self, dt: f64) {
        for p in &mut self.pulses {
            p.remaining -= dt;
        }
        self.pulses.retain(|p| p.remaining > 0.0);
    }

    fn emit(&mut self, position: [f32; 3], band: u8, amplitude: f32) {
        if amplitude <= 0.0 || band as usize >= self.bands {
            return;
        }
        self.pulses.push(Pulse {
            position,
            band,
            amplitude,
            remaining: self.lifetime,
        });
    }

    fn sample(&self, position: [f32; 3], bands: (u8, u8)) -> Sample {
        let mut intensity = 0.0f32;
        let mut gradient = [0.0f32; 3];
        for p in &self.pulses {
            if p.band < bands.0 || p.band > bands.1 {
                continue;
            }
            let d = [
                position[0] - p.position[0],
                position[1] - p.position[1],
                position[2] - p.position[2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r = r2.sqrt().max(1.0e-9);
            let value = p.amplitude * (-self.absorption * r).exp() / r;
            intensity += value;
            // The gradient of `A exp(-a r) / r` points back toward the source.
            let slope = -value * (self.absorption + 1.0 / r) / r;
            for k in 0..3 {
                gradient[k] += slope * d[k];
            }
        }
        Sample {
            intensity,
            gradient,
        }
    }

    fn band_count(&self) -> usize {
        self.bands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hadean_chem::{generate, ChemParams};

    fn grid() -> Grid {
        Grid::new(12, 12, 8, 25.0e-6)
    }

    #[test]
    fn a_chemical_gradient_points_at_its_source() {
        let g = grid();
        let chem = generate(21, ChemParams::default());
        let mut field = ChemField::new(&g, chem.n_compounds());
        let source = g.idx(8, 6, 4);
        field.set(0, source, 1.0e9);
        // A shallow ramp toward the source so the gradient is well defined.
        field.set(0, g.idx(7, 6, 4), 5.0e8);
        field.set(0, g.idx(6, 6, 4), 2.0e8);

        let channel = Chemical {
            grid: &g,
            field: &mut field,
        };
        let probe = g.centre(g.idx(7, 6, 4));
        let s = channel.sample(probe, (0, 0));
        assert!(s.intensity > 0.0);
        assert!(s.gradient[0] > 0.0, "gradient should point toward +x");
        assert!(s.slope() > 0.0);
    }

    #[test]
    fn chemical_emission_lands_in_the_right_voxel() {
        let g = grid();
        let chem = generate(21, ChemParams::default());
        let mut field = ChemField::new(&g, chem.n_compounds());
        let target = g.idx(3, 4, 5);
        {
            let mut channel = Chemical {
                grid: &g,
                field: &mut field,
            };
            channel.emit(g.centre(target), 2, 1.0e8);
        }
        assert_eq!(field.get(2, target), 1.0e8);
        assert_eq!(field.total_of(2), 1.0e8);
    }

    #[test]
    fn light_emission_falls_off_with_distance() {
        let g = grid();
        let mut field = LightField::new(&g);
        let source = g.idx(6, 6, 4);
        {
            let mut channel = Light::new(&g, &mut field);
            channel.emit(g.centre(source), 3, 1.0);
        }
        let at = |v: usize| field.intensity_at(3, v);
        assert!(at(source) > at(g.idx(7, 6, 4)));
        assert!(at(g.idx(7, 6, 4)) > at(g.idx(9, 6, 4)));
        // Other bands untouched.
        assert_eq!(field.intensity_at(4, source), 0.0);
    }

    #[test]
    fn light_sampling_reads_the_band_range() {
        let g = grid();
        let mut field = LightField::new(&g);
        let v = g.idx(5, 5, 3);
        let (i1, i6) = (field.index(1, v), field.index(6, v));
        field.intensity[i1] = 2.0;
        field.intensity[i6] = 5.0;
        let channel = Light::new(&g, &mut field);
        assert_eq!(channel.sample(g.centre(v), (0, 7)).intensity, 7.0);
        assert_eq!(channel.sample(g.centre(v), (1, 1)).intensity, 2.0);
        assert_eq!(channel.sample(g.centre(v), (2, 5)).intensity, 0.0);
        assert_eq!(channel.band_count(), N_BANDS);
    }

    #[test]
    fn mechanical_pulses_fade_and_expire() {
        let mut m = Mechanical::new(4);
        m.lifetime = 0.05;
        m.emit([0.0, 0.0, 0.0], 1, 1.0);
        assert_eq!(m.pulses.len(), 1);
        let near = m.sample([1.0e-5, 0.0, 0.0], (0, 3)).intensity;
        let far = m.sample([1.0e-3, 0.0, 0.0], (0, 3)).intensity;
        assert!(near > far, "no attenuation with distance");

        m.propagate(0.03);
        assert_eq!(m.pulses.len(), 1);
        m.propagate(0.03);
        assert!(m.pulses.is_empty(), "pulse outlived its lifetime");
        assert_eq!(m.sample([0.0, 0.0, 0.0], (0, 3)).intensity, 0.0);
    }

    #[test]
    fn mechanical_gradient_points_back_at_the_source() {
        let mut m = Mechanical::new(2);
        m.emit([0.0, 0.0, 0.0], 0, 1.0);
        // Standing at +x, the gradient must point back toward the origin.
        let s = m.sample([2.0e-4, 0.0, 0.0], (0, 1));
        assert!(s.intensity > 0.0);
        assert!(s.gradient[0] < 0.0, "gradient should point back toward -x");
    }

    #[test]
    fn mechanical_respects_band_filtering() {
        let mut m = Mechanical::new(4);
        m.emit([0.0, 0.0, 0.0], 3, 1.0);
        assert!(m.sample([1.0e-5, 0.0, 0.0], (3, 3)).intensity > 0.0);
        assert_eq!(m.sample([1.0e-5, 0.0, 0.0], (0, 2)).intensity, 0.0);
        // Out-of-range bands are refused rather than silently stored.
        m.emit([0.0, 0.0, 0.0], 9, 1.0);
        assert_eq!(m.pulses.len(), 1);
    }

    #[test]
    fn boundary_voxels_still_report_a_gradient() {
        let g = grid();
        let chem = generate(21, ChemParams::default());
        let mut field = ChemField::new(&g, chem.n_compounds());
        field.set(0, g.idx(1, 0, 0), 1.0e9);
        let channel = Chemical {
            grid: &g,
            field: &mut field,
        };
        let s = channel.sample(g.centre(g.idx(0, 0, 0)), (0, 0));
        assert!(
            s.gradient[0] > 0.0,
            "one-sided gradient at a wall should still work"
        );
    }
}
