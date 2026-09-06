//! Deterministic randomness.
//!
//! Two facilities, and the distinction matters:
//!
//! * [`Counter`] -- a *counter-based* generator. It is a pure function of
//!   `(world_seed, tick, entity, purpose, stream)` with no mutable state, so
//!   the value an entity draws does not depend on thread scheduling, on how
//!   many other entities drew before it, or on iteration order. Everything
//!   inside the simulation loop must use this.
//!
//! * [`SeqRng`] -- an ordinary stateful stream, legal *only* in single-threaded
//!   set-up code that runs in a fixed order (world generation, chemistry
//!   generation). Never call it from a per-tick code path.

/// SplitMix64 finaliser. Strong avalanche, one multiply-xorshift chain.
#[inline]
pub const fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Tag identifying *what* a random draw is for.
///
/// Two draws made in the same tick by the same entity must use different
/// purposes, or they return the same bits. Add variants freely; the numeric
/// values are part of the replay contract, so never renumber an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum Purpose {
    Brownian = 1,
    ReactionNoise = 2,
    Mutation = 3,
    Division = 4,
    Death = 5,
    Partition = 6,
    Expression = 7,
    Neural = 8,
    Placement = 9,
    Disturbance = 10,
}

/// A stateless, counter-based generator.
///
/// Cheap enough to construct per draw; it holds only the world seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counter {
    seed: u64,
}

impl Counter {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    #[inline]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Raw 64-bit draw for `(tick, entity, purpose, stream)`.
    ///
    /// `stream` distinguishes repeated draws for the same purpose (e.g. the
    /// three axes of a Brownian kick).
    #[inline]
    pub fn bits(&self, tick: u64, entity: u64, purpose: Purpose, stream: u64) -> u64 {
        let mut h = self.seed;
        h = mix64(h ^ tick.wrapping_mul(0xD6E8_FEB8_6659_FD93));
        h = mix64(h ^ entity.wrapping_mul(0xA076_1D64_78BD_642F));
        h = mix64(h ^ (purpose as u64).wrapping_mul(0x8EBC_6AF0_9C88_C6E3));
        mix64(h ^ stream.wrapping_mul(0x5851_F42D_4C95_7F2D))
    }

    /// Uniform in `[0, 1)`, 24-bit mantissa.
    #[inline]
    pub fn unit(&self, tick: u64, entity: u64, purpose: Purpose, stream: u64) -> f32 {
        u64_to_unit(self.bits(tick, entity, purpose, stream))
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&self, tick: u64, entity: u64, p: Purpose, s: u64, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit(tick, entity, p, s)
    }

    /// Standard normal, via the Box-Muller transform.
    #[inline]
    pub fn normal(&self, tick: u64, entity: u64, purpose: Purpose, stream: u64) -> f32 {
        let u1 = self.unit(tick, entity, purpose, stream * 2).max(1.0e-7);
        let u2 = self.unit(tick, entity, purpose, stream * 2 + 1);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// True with probability `p`.
    #[inline]
    pub fn chance(&self, tick: u64, entity: u64, purpose: Purpose, stream: u64, p: f32) -> bool {
        self.unit(tick, entity, purpose, stream) < p
    }
}

/// Map the high bits of a `u64` into `[0, 1)`.
#[inline]
pub fn u64_to_unit(x: u64) -> f32 {
    // 24 bits is exactly the f32 mantissa; using more would round to 1.0.
    ((x >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// A stateful SplitMix64 stream. Set-up code only -- see the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqRng {
    state: u64,
}

impl SeqRng {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn unit(&mut self) -> f32 {
        u64_to_unit(self.next_u64())
    }

    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// Log-uniform in `[lo, hi)`. Useful for rate constants, which span decades.
    #[inline]
    pub fn log_range(&mut self, lo: f32, hi: f32) -> f32 {
        (lo.ln() + (hi.ln() - lo.ln()) * self.unit()).exp()
    }

    /// Uniform integer in `[0, n)`. Returns 0 for `n == 0`.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        // Multiply-shift; the modulo bias is ~2^-64 and irrelevant here.
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    #[inline]
    pub fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }

    /// Fisher-Yates, in a fixed order so the result is reproducible.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }

    /// Pick an element uniformly, or `None` if empty.
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> Option<&'a T> {
        if xs.is_empty() {
            None
        } else {
            Some(&xs[self.below(xs.len())])
        }
    }

    /// Fork an independent stream, so a subsystem can draw without perturbing
    /// the caller's sequence.
    #[inline]
    pub fn fork(&mut self, label: u64) -> SeqRng {
        SeqRng::new(mix64(self.next_u64() ^ mix64(label)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_is_stateless_and_order_independent() {
        let c = Counter::new(42);
        let a = c.unit(100, 7, Purpose::Brownian, 0);
        // Interleave unrelated draws; the original must be unchanged.
        for i in 0..1000 {
            let _ = c.unit(i, i * 3, Purpose::Mutation, i);
        }
        assert_eq!(a, c.unit(100, 7, Purpose::Brownian, 0));
    }

    #[test]
    fn counter_decorrelates_neighbouring_inputs() {
        let c = Counter::new(1);
        let a = c.bits(1, 1, Purpose::Brownian, 0);
        let b = c.bits(1, 2, Purpose::Brownian, 0);
        let d = c.bits(2, 1, Purpose::Brownian, 0);
        let e = c.bits(1, 1, Purpose::Brownian, 1);
        assert_ne!(a, b);
        assert_ne!(a, d);
        assert_ne!(a, e);
    }

    #[test]
    fn unit_is_in_range() {
        let c = Counter::new(9);
        for i in 0..10_000u64 {
            let u = c.unit(i, i ^ 0x5555, Purpose::Death, 0);
            assert!((0.0..1.0).contains(&u), "u = {u}");
        }
    }

    #[test]
    fn normal_has_roughly_unit_variance() {
        let c = Counter::new(3);
        let n = 20_000u64;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for i in 0..n {
            let x = c.normal(i, 0, Purpose::Brownian, 0) as f64;
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.1, "var = {var}");
    }

    #[test]
    fn seq_rng_replays() {
        let a: Vec<u64> = (0..64)
            .scan(SeqRng::new(7), |r, _| Some(r.next_u64()))
            .collect();
        let b: Vec<u64> = (0..64)
            .scan(SeqRng::new(7), |r, _| Some(r.next_u64()))
            .collect();
        assert_eq!(a, b);
    }
}
