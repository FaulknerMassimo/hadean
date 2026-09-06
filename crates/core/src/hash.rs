//! Deterministic state hashing.
//!
//! Used by the determinism harness: run N ticks twice, hash the whole world,
//! assert the digests match. Hashing is bitwise over IEEE-754 representations,
//! so it catches a one-ULP divergence that an epsilon comparison would miss.
//!
//! NaN would break the contract (many bit patterns, all "equal"), so
//! [`StateHasher::f32`] canonicalises NaN and normalises -0.0 to +0.0.

/// FNV-1a, 64-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateHasher {
    h: u64,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl Default for StateHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StateHasher {
    #[inline]
    pub const fn new() -> Self {
        Self { h: FNV_OFFSET }
    }

    #[inline]
    pub fn byte(&mut self, b: u8) -> &mut Self {
        self.h ^= b as u64;
        self.h = self.h.wrapping_mul(FNV_PRIME);
        self
    }

    #[inline]
    pub fn bytes(&mut self, bs: &[u8]) -> &mut Self {
        for &b in bs {
            self.byte(b);
        }
        self
    }

    #[inline]
    pub fn u64(&mut self, x: u64) -> &mut Self {
        self.bytes(&x.to_le_bytes())
    }

    #[inline]
    pub fn u32(&mut self, x: u32) -> &mut Self {
        self.bytes(&x.to_le_bytes())
    }

    #[inline]
    pub fn usize(&mut self, x: usize) -> &mut Self {
        self.u64(x as u64)
    }

    #[inline]
    pub fn bool(&mut self, x: bool) -> &mut Self {
        self.byte(x as u8)
    }

    /// Hash an `f32` bitwise, with NaN canonicalised and -0.0 folded to +0.0.
    #[inline]
    pub fn f32(&mut self, x: f32) -> &mut Self {
        let bits = if x.is_nan() {
            0x7fc0_0000
        } else if x == 0.0 {
            0
        } else {
            x.to_bits()
        };
        self.u32(bits)
    }

    #[inline]
    pub fn f64(&mut self, x: f64) -> &mut Self {
        let bits = if x.is_nan() {
            0x7ff8_0000_0000_0000
        } else if x == 0.0 {
            0
        } else {
            x.to_bits()
        };
        self.u64(bits)
    }

    #[inline]
    pub fn f32_slice(&mut self, xs: &[f32]) -> &mut Self {
        self.usize(xs.len());
        for &x in xs {
            self.f32(x);
        }
        self
    }

    #[inline]
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.usize(s.len());
        self.bytes(s.as_bytes())
    }

    #[inline]
    pub fn finish(&self) -> u64 {
        self.h
    }

    /// Lower-case hex digest, for logs and CI output.
    pub fn hex(&self) -> String {
        format!("{:016x}", self.h)
    }
}

/// Anything that contributes to the reproducible world state.
pub trait HashState {
    fn hash_state(&self, h: &mut StateHasher);

    /// Convenience: digest of just this value.
    fn state_hash(&self) -> u64 {
        let mut h = StateHasher::new();
        self.hash_state(&mut h);
        h.finish()
    }
}

impl HashState for f32 {
    fn hash_state(&self, h: &mut StateHasher) {
        h.f32(*self);
    }
}

impl<T: HashState> HashState for [T] {
    fn hash_state(&self, h: &mut StateHasher) {
        h.usize(self.len());
        for x in self {
            x.hash_state(h);
        }
    }
}

impl<T: HashState> HashState for Vec<T> {
    fn hash_state(&self, h: &mut StateHasher) {
        self.as_slice().hash_state(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_one_ulp() {
        let a = 1.0f32;
        let b = f32::from_bits(a.to_bits() + 1);
        let mut ha = StateHasher::new();
        let mut hb = StateHasher::new();
        ha.f32(a);
        hb.f32(b);
        assert_ne!(ha.finish(), hb.finish());
    }

    #[test]
    fn negative_zero_matches_zero() {
        let mut ha = StateHasher::new();
        let mut hb = StateHasher::new();
        ha.f32(0.0);
        hb.f32(-0.0);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn nan_is_canonical() {
        let mut ha = StateHasher::new();
        let mut hb = StateHasher::new();
        ha.f32(f32::NAN);
        hb.f32(-f32::NAN);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn length_is_part_of_the_digest() {
        let mut ha = StateHasher::new();
        let mut hb = StateHasher::new();
        ha.f32_slice(&[1.0, 2.0]);
        hb.f32_slice(&[1.0, 2.0, 0.0]);
        assert_ne!(ha.finish(), hb.finish());
    }
}
