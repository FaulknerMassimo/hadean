//! Molecules as explicit atom graphs.
//!
//! Hydrogen is a real atom here, not an implicit count. That costs a little
//! memory and buys a lot: every reaction can be expressed as a graph rewrite
//! on the atoms themselves, so **element balance is structural rather than
//! checked**. A reaction cannot be unbalanced, because it is built by moving
//! atoms between graphs.
//!
//! A molecule is *valid* only when every atom's total bond order equals its
//! element's valence exactly, and the graph is connected. Both are enforced.

use crate::element::{bond_energy, bond_polarity, element, valence, ElementId, N_ELEMENTS};
use hadean_core::units::{Joules, AMU};
use serde::{Deserialize, Serialize};

/// Largest molecule the canonicaliser will handle comfortably.
pub const MAX_ATOMS: usize = 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bond {
    pub a: u8,
    pub b: u8,
    pub order: u8,
}

impl Bond {
    #[inline]
    pub fn new(a: u8, b: u8, order: u8) -> Self {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        Self { a, b, order }
    }

    #[inline]
    pub fn other(&self, i: u8) -> Option<u8> {
        if self.a == i {
            Some(self.b)
        } else if self.b == i {
            Some(self.a)
        } else {
            None
        }
    }
}

/// A connected, valence-saturated atom graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Molecule {
    pub atoms: Vec<ElementId>,
    pub bonds: Vec<Bond>,
}

impl Molecule {
    pub fn new(atoms: Vec<ElementId>, bonds: Vec<Bond>) -> Self {
        Self { atoms, bonds }
    }

    /// A lone atom of element `e`. Only valid for a valence-0 element, so this
    /// is used as a scratch value during assembly rather than as a compound.
    pub fn single(e: ElementId) -> Self {
        Self {
            atoms: vec![e],
            bonds: Vec::new(),
        }
    }

    #[inline]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Total bond order at atom `i`.
    pub fn degree(&self, i: u8) -> u8 {
        self.bonds
            .iter()
            .filter_map(|b| b.other(i).map(|_| b.order))
            .sum()
    }

    /// Neighbours of atom `i` as `(atom, bond order)`.
    pub fn neighbours(&self, i: u8) -> impl Iterator<Item = (u8, u8)> + '_ {
        self.bonds
            .iter()
            .filter_map(move |b| b.other(i).map(|o| (o, b.order)))
    }

    /// Per-element atom counts. This is the conserved quantity: every reaction
    /// must leave the summed formula of its reactants and products identical.
    pub fn formula(&self) -> [u16; N_ELEMENTS] {
        let mut f = [0u16; N_ELEMENTS];
        for &a in &self.atoms {
            f[a as usize] += 1;
        }
        f
    }

    /// Molecular mass in kilograms.
    pub fn mass(&self) -> f32 {
        let amu: f64 = self.atoms.iter().map(|&a| element(a).mass_amu as f64).sum();
        (amu * AMU) as f32
    }

    /// Every atom's valence is exactly satisfied.
    pub fn is_saturated(&self) -> bool {
        (0..self.atoms.len()).all(|i| self.degree(i as u8) == valence(self.atoms[i]))
    }

    /// The graph is connected (or is a single atom).
    pub fn is_connected(&self) -> bool {
        if self.atoms.len() <= 1 {
            return true;
        }
        let mut seen = vec![false; self.atoms.len()];
        let mut stack = vec![0u8];
        seen[0] = true;
        let mut count = 1;
        while let Some(i) = stack.pop() {
            for (j, _) in self.neighbours(i) {
                if !seen[j as usize] {
                    seen[j as usize] = true;
                    count += 1;
                    stack.push(j);
                }
            }
        }
        count == self.atoms.len()
    }

    pub fn is_valid(&self) -> bool {
        !self.atoms.is_empty()
            && self.atoms.len() <= MAX_ATOMS
            && self.is_saturated()
            && self.is_connected()
            && self.bonds.iter().all(|b| {
                b.a != b.b
                    && (b.a as usize) < self.atoms.len()
                    && (b.b as usize) < self.atoms.len()
                    && b.order >= 1
            })
    }

    /// Number of independent rings (cyclomatic number).
    pub fn ring_count(&self) -> usize {
        self.bonds.len() + 1 - self.atoms.len()
    }

    /// Count of bonds of order 2 or higher that are adjacent to another such
    /// bond -- a crude conjugation measure, used for absorption spectra.
    pub fn conjugation(&self) -> usize {
        let multiple: Vec<Bond> = self
            .bonds
            .iter()
            .copied()
            .filter(|b| b.order >= 2)
            .collect();
        multiple
            .iter()
            .enumerate()
            .filter(|(i, b)| {
                multiple
                    .iter()
                    .enumerate()
                    .any(|(j, c)| j != *i && (c.a == b.a || c.a == b.b || c.b == b.a || c.b == b.b))
            })
            .count()
    }

    /// Formation enthalpy, joules per particle, on the atomisation convention:
    /// free gaseous atoms are the zero, so a stable molecule is negative.
    ///
    /// This is a pure function of structure, which is exactly what makes the
    /// reaction network thermodynamically closed: `dH` is a difference of
    /// state functions, so it sums to zero around any cycle.
    pub fn formation_enthalpy(&self) -> Joules {
        let bonds: Joules = self
            .bonds
            .iter()
            .map(|b| bond_energy(self.atoms[b.a as usize], self.atoms[b.b as usize], b.order))
            .sum();
        // Structural corrections, both deterministic functions of the graph.
        let strain = self.strain_energy();
        let resonance = self.resonance_energy();
        -bonds + strain - resonance
    }

    /// Ring strain: small rings are destabilised.
    fn strain_energy(&self) -> Joules {
        if self.ring_count() == 0 {
            return 0.0;
        }
        // Approximate the smallest ring by the graph girth.
        let girth = self.girth().unwrap_or(6);
        let kj = match girth {
            3 => 115.0,
            4 => 110.0,
            5 => 26.0,
            6 => 0.0,
            _ => 10.0,
        };
        hadean_core::units::kj_per_mol(kj * self.ring_count() as f64)
    }

    /// Conjugation stabilises; this is the delocalisation bonus.
    fn resonance_energy(&self) -> Joules {
        hadean_core::units::kj_per_mol(18.0 * self.conjugation() as f64)
    }

    /// Length of the shortest cycle, or `None` for an acyclic graph.
    pub fn girth(&self) -> Option<usize> {
        let n = self.atoms.len();
        let mut best: Option<usize> = None;
        for root in 0..n {
            // BFS; a non-tree edge closes a cycle of known length.
            let mut dist = vec![usize::MAX; n];
            let mut parent = vec![usize::MAX; n];
            let mut queue = std::collections::VecDeque::new();
            dist[root] = 0;
            queue.push_back(root);
            while let Some(u) = queue.pop_front() {
                for (v, _) in self.neighbours(u as u8) {
                    let v = v as usize;
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        parent[v] = u;
                        queue.push_back(v);
                    } else if parent[u] != v {
                        let c = dist[u] + dist[v] + 1;
                        best = Some(best.map_or(c, |b: usize| b.min(c)));
                    }
                }
            }
        }
        best
    }

    /// Mean bond ionic character, in `[0, 1]`. Drives membrane permeability:
    /// polar molecules do not cross a lipid bilayer unaided.
    pub fn polarity(&self) -> f32 {
        if self.bonds.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .bonds
            .iter()
            .map(|b| bond_polarity(self.atoms[b.a as usize], self.atoms[b.b as usize]))
            .sum();
        sum / self.bonds.len() as f32
    }

    /// Net formal charge. The monovalent metal donates; an oxygen or sulfur
    /// bonded only to metals carries the matching negative charge.
    pub fn charge(&self) -> i8 {
        let mut q = 0i32;
        for (i, &e) in self.atoms.iter().enumerate() {
            if e == crate::element::M {
                q += 1;
            } else if matches!(e, crate::element::O | crate::element::S) {
                let all_metal = self
                    .neighbours(i as u8)
                    .all(|(j, _)| self.atoms[j as usize] == crate::element::M);
                if all_metal && self.neighbours(i as u8).count() > 0 {
                    q -= 1;
                }
            }
        }
        q.clamp(-4, 4) as i8
    }

    /// Hill-ish formula string, e.g. `C2H6O`.
    pub fn formula_string(&self) -> String {
        let f = self.formula();
        let order = [
            crate::element::C,
            crate::element::H,
            crate::element::O,
            crate::element::N,
            crate::element::S,
            crate::element::M,
        ];
        let mut s = String::new();
        for e in order {
            let n = f[e as usize];
            if n == 0 {
                continue;
            }
            s.push_str(element(e).symbol);
            if n > 1 {
                s.push_str(&n.to_string());
            }
        }
        s
    }
}

/// Sum two formulas.
#[inline]
pub fn formula_add(a: &mut [u16; N_ELEMENTS], b: &[u16; N_ELEMENTS]) {
    for i in 0..N_ELEMENTS {
        a[i] += b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{C, H, O};

    /// Water: O with two H.
    fn water() -> Molecule {
        Molecule::new(vec![O, H, H], vec![Bond::new(0, 1, 1), Bond::new(0, 2, 1)])
    }

    /// Carbon dioxide: O=C=O.
    fn co2() -> Molecule {
        Molecule::new(vec![C, O, O], vec![Bond::new(0, 1, 2), Bond::new(0, 2, 2)])
    }

    fn methane() -> Molecule {
        Molecule::new(
            vec![C, H, H, H, H],
            vec![
                Bond::new(0, 1, 1),
                Bond::new(0, 2, 1),
                Bond::new(0, 3, 1),
                Bond::new(0, 4, 1),
            ],
        )
    }

    #[test]
    fn hand_built_molecules_are_valid() {
        for m in [water(), co2(), methane()] {
            assert!(m.is_valid(), "{} invalid", m.formula_string());
        }
    }

    #[test]
    fn unsaturated_is_rejected() {
        // Carbon with only three hydrogens.
        let m = Molecule::new(
            vec![C, H, H, H],
            vec![Bond::new(0, 1, 1), Bond::new(0, 2, 1), Bond::new(0, 3, 1)],
        );
        assert!(!m.is_saturated());
        assert!(!m.is_valid());
    }

    #[test]
    fn disconnected_is_rejected() {
        let mut m = water();
        m.atoms.push(H);
        m.atoms.push(H);
        m.bonds.push(Bond::new(3, 4, 1));
        assert!(m.is_saturated());
        assert!(!m.is_connected());
        assert!(!m.is_valid());
    }

    #[test]
    fn formula_strings_read_correctly() {
        assert_eq!(water().formula_string(), "H2O");
        assert_eq!(co2().formula_string(), "CO2");
        assert_eq!(methane().formula_string(), "CH4");
    }

    #[test]
    fn formation_enthalpy_is_negative_and_sane() {
        // Water is roughly -2 * 463 kJ/mol on this convention.
        let h = hadean_core::units::to_kj_per_mol(water().formation_enthalpy());
        assert!((-960.0..-900.0).contains(&h), "H2O = {h} kJ/mol");
        assert!(co2().formation_enthalpy() < 0.0);
        assert!(methane().formation_enthalpy() < 0.0);
    }

    #[test]
    fn polarity_ranks_as_expected() {
        assert!(water().polarity() > methane().polarity());
    }

    #[test]
    fn girth_finds_the_smallest_ring() {
        // Cyclopropane-like C3H6 ring.
        let ring = Molecule::new(
            vec![C, C, C, H, H, H, H, H, H],
            vec![
                Bond::new(0, 1, 1),
                Bond::new(1, 2, 1),
                Bond::new(2, 0, 1),
                Bond::new(0, 3, 1),
                Bond::new(0, 4, 1),
                Bond::new(1, 5, 1),
                Bond::new(1, 6, 1),
                Bond::new(2, 7, 1),
                Bond::new(2, 8, 1),
            ],
        );
        assert!(ring.is_valid());
        assert_eq!(ring.ring_count(), 1);
        assert_eq!(ring.girth(), Some(3));
        assert!(water().girth().is_none());
    }

    #[test]
    fn mass_is_additive() {
        let m = methane().mass() as f64 / AMU;
        assert!((m - 16.043).abs() < 0.01, "CH4 = {m} amu");
    }
}
