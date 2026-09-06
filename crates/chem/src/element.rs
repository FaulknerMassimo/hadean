//! The element table and bond energetics.
//!
//! Six elements, chosen to span the valences that make an interesting organic
//! chemistry possible. They are *like* their real counterparts but the
//! simulation never claims to be Earth chemistry -- what matters is that the
//! energetics are internally consistent.
//!
//! Bond energies are the spine of the whole simulation. A compound's formation
//! enthalpy is defined as the negative of its total bond energy (the
//! atomisation convention: free atoms sit at zero). Because that makes
//! enthalpy a *state function of the compound*, the enthalpy change around any
//! closed cycle of reactions is exactly zero by construction. Organisms
//! therefore cannot evolve a reaction cycle that nets free energy -- the
//! failure mode that ends simulations like this one.

use hadean_core::units::{kj_per_mol, Joules};
use serde::{Deserialize, Serialize};

/// Element identifier, an index into [`ELEMENTS`].
pub type ElementId = u8;

pub const H: ElementId = 0;
pub const C: ElementId = 1;
pub const O: ElementId = 2;
pub const N: ElementId = 3;
pub const S: ElementId = 4;
/// Monovalent metal ion, sodium-like.
pub const M: ElementId = 5;

pub const N_ELEMENTS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub symbol: &'static str,
    /// Number of bonds the atom must form. Molecules are only valid when every
    /// atom's total bond order equals its valence exactly.
    pub valence: u8,
    /// Atomic mass in unified atomic mass units.
    pub mass_amu: f32,
    /// Pauling-style electronegativity, used for bond polarity.
    pub electronegativity: f32,
}

pub const ELEMENTS: [Element; N_ELEMENTS] = [
    Element {
        symbol: "H",
        valence: 1,
        mass_amu: 1.008,
        electronegativity: 2.20,
    },
    Element {
        symbol: "C",
        valence: 4,
        mass_amu: 12.011,
        electronegativity: 2.55,
    },
    Element {
        symbol: "O",
        valence: 2,
        mass_amu: 15.999,
        electronegativity: 3.44,
    },
    Element {
        symbol: "N",
        valence: 3,
        mass_amu: 14.007,
        electronegativity: 3.04,
    },
    Element {
        symbol: "S",
        valence: 2,
        mass_amu: 32.060,
        electronegativity: 2.58,
    },
    Element {
        symbol: "M",
        valence: 1,
        mass_amu: 22.990,
        electronegativity: 0.93,
    },
];

#[inline]
pub fn element(e: ElementId) -> &'static Element {
    &ELEMENTS[e as usize]
}

#[inline]
pub fn valence(e: ElementId) -> u8 {
    ELEMENTS[e as usize].valence
}

/// Elements that can form more than one bond, and so can be a molecular
/// skeleton. Hydrogen and the metal ion are terminators.
pub const SKELETAL: [ElementId; 4] = [C, O, N, S];

/// Single-bond dissociation energies, kJ/mol. Symmetric.
#[rustfmt::skip]
const SINGLE: [[f32; N_ELEMENTS]; N_ELEMENTS] = [
    //  H      C      O      N      S      M
    [ 436.0, 413.0, 463.0, 391.0, 339.0, 180.0 ], // H
    [ 413.0, 348.0, 358.0, 305.0, 272.0, 150.0 ], // C
    [ 463.0, 358.0, 146.0, 201.0, 265.0, 400.0 ], // O
    [ 391.0, 305.0, 201.0, 163.0, 250.0, 220.0 ], // N
    [ 339.0, 272.0, 265.0, 250.0, 266.0, 300.0 ], // S
    [ 180.0, 150.0, 400.0, 220.0, 300.0,  75.0 ], // M
];

/// Explicit multiple-bond energies, kJ/mol. Anything absent falls back to a
/// multiple of the single-bond value.
#[rustfmt::skip]
const MULTIPLE: &[(ElementId, ElementId, u8, f32)] = &[
    (C, C, 2, 614.0), (C, O, 2, 799.0), (C, N, 2, 615.0), (C, S, 2, 573.0),
    (O, O, 2, 495.0), (N, N, 2, 418.0), (N, O, 2, 607.0), (S, O, 2, 522.0),
    (S, S, 2, 425.0),
    (C, C, 3, 839.0), (C, N, 3, 891.0), (N, N, 3, 941.0), (C, O, 3, 1072.0),
];

/// Bond dissociation energy for `a`-`b` at `order`, in joules per particle.
///
/// Always positive: breaking a bond costs energy, forming one releases it.
pub fn bond_energy(a: ElementId, b: ElementId, order: u8) -> Joules {
    kj_per_mol(bond_energy_kj(a, b, order))
}

/// Bond energy in kJ/mol. Defined recursively so that an order falling back to
/// the generic multiplier can never come out below the order beneath it, which
/// would make a stronger bond cheaper to break than a weaker one.
fn bond_energy_kj(a: ElementId, b: ElementId, order: u8) -> f64 {
    let single = SINGLE[a as usize][b as usize] as f64;
    match order {
        0 => 0.0,
        1 => single,
        _ => {
            let explicit = MULTIPLE
                .iter()
                .find(|(x, y, o, _)| *o == order && ((*x == a && *y == b) || (*x == b && *y == a)))
                .map(|(_, _, _, e)| *e as f64);
            match explicit {
                Some(e) => e,
                None => {
                    let generic = single * if order == 2 { 1.75 } else { 2.30 };
                    generic.max(bond_energy_kj(a, b, order - 1) * 1.25)
                }
            }
        }
    }
}

/// Ionic character of a bond, from the electronegativity difference.
/// Zero for a symmetric bond, approaching one for a strongly polar bond.
pub fn bond_polarity(a: ElementId, b: ElementId) -> f32 {
    let d = (element(a).electronegativity - element(b).electronegativity).abs();
    // Pauling's ionic-character estimate.
    1.0 - (-0.25 * d * d).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hadean_core::units::to_kj_per_mol;

    #[test]
    fn single_bond_table_is_symmetric() {
        for (a, row) in SINGLE.iter().enumerate() {
            for (b, &energy) in row.iter().enumerate() {
                assert_eq!(energy, SINGLE[b][a], "{a}-{b}");
            }
        }
    }

    #[test]
    fn bond_energies_are_positive_and_ordered() {
        for a in 0..N_ELEMENTS as u8 {
            for b in 0..N_ELEMENTS as u8 {
                let s = bond_energy(a, b, 1);
                assert!(s > 0.0);
                if valence(a) > 1 && valence(b) > 1 {
                    assert!(bond_energy(a, b, 2) > s, "double must beat single");
                    assert!(bond_energy(a, b, 3) > bond_energy(a, b, 2));
                }
            }
        }
    }

    #[test]
    fn energies_are_in_a_plausible_range() {
        // Every bond should land between 50 and 1200 kJ/mol.
        for a in 0..N_ELEMENTS as u8 {
            for b in 0..N_ELEMENTS as u8 {
                let kj = to_kj_per_mol(bond_energy(a, b, 1));
                assert!((50.0..=1200.0).contains(&kj), "{a}-{b} = {kj}");
            }
        }
    }

    #[test]
    fn lookup_is_order_insensitive() {
        assert_eq!(bond_energy(C, O, 2), bond_energy(O, C, 2));
        assert_eq!(bond_energy(N, O, 2), bond_energy(O, N, 2));
    }

    #[test]
    fn polarity_is_zero_for_identical_atoms() {
        assert_eq!(bond_polarity(C, C), 0.0);
        assert!(bond_polarity(M, O) > bond_polarity(C, H));
    }
}
