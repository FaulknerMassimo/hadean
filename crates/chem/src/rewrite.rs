//! Graph rewrites on molecules.
//!
//! Reactions are not written down as stoichiometric equations and then checked
//! for balance. They are *built* by cutting and re-joining atom graphs, so
//! every reaction the generator emits conserves every element exactly, by
//! construction. There is no code path that can produce an unbalanced reaction.
//!
//! Two rewrite families cover the useful ground:
//!
//! * **Exchange** (`A + B -> C + D`) -- cut one single bond in each reactant
//!   and recombine the halves crosswise. With `B` = water this is hydrolysis;
//!   with `B` = H2 it is a hydrogen transfer; in general it is metathesis.
//! * **Addition** (`A + B -> C`) -- open one multiple bond in `A` and add the
//!   two halves of `B` across it. Run backwards it is elimination.
//!
//! Both are expressed through one primitive: an [`Open`] molecule, which is a
//! graph carrying a list of unsatisfied valence sites, and [`merge`], which
//! joins two of them with a new single bond.

use crate::molecule::{Bond, Molecule, MAX_ATOMS};

/// A molecule with unsatisfied valences. Intentionally not a valid
/// [`Molecule`] until every site is consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Open {
    pub mol: Molecule,
    /// Atom indices each missing exactly one unit of bond order.
    pub sites: Vec<u8>,
}

impl Open {
    /// The molecule, if every valence is now satisfied.
    pub fn close(self) -> Option<Molecule> {
        if self.sites.is_empty() && self.mol.is_valid() {
            Some(self.mol)
        } else {
            None
        }
    }
}

/// Cut a single bond that separates the molecule into two pieces.
///
/// Returns `None` for a ring bond (cutting it leaves one connected fragment
/// with two open sites, which we do not use) or a bond of order above one.
pub fn split_bond(mol: &Molecule, bond: usize) -> Option<(Open, Open)> {
    let b = *mol.bonds.get(bond)?;
    if b.order != 1 {
        return None;
    }
    let mut rest = mol.bonds.clone();
    rest.remove(bond);
    let side_a = reachable(mol.n_atoms(), &rest, b.a);
    if side_a[b.b as usize] {
        return None; // ring bond: still connected
    }
    let atoms_a: Vec<u8> = (0..mol.n_atoms() as u8)
        .filter(|&i| side_a[i as usize])
        .collect();
    let atoms_b: Vec<u8> = (0..mol.n_atoms() as u8)
        .filter(|&i| !side_a[i as usize])
        .collect();
    Some((
        subgraph(mol, &rest, &atoms_a, &[b.a]),
        subgraph(mol, &rest, &atoms_b, &[b.b]),
    ))
}

/// Reduce a bond of order >= 2 by one, leaving two open sites on the molecule.
pub fn open_multiple(mol: &Molecule, bond: usize) -> Option<Open> {
    let b = *mol.bonds.get(bond)?;
    if b.order < 2 {
        return None;
    }
    let mut bonds = mol.bonds.clone();
    bonds[bond] = Bond::new(b.a, b.b, b.order - 1);
    Some(Open {
        mol: Molecule::new(mol.atoms.clone(), bonds),
        sites: vec![b.a, b.b],
    })
}

/// Join two open molecules with a new single bond between the chosen sites.
///
/// `sa` and `sb` index into each side's `sites` list, not into its atoms.
pub fn merge(a: &Open, sa: usize, b: &Open, sb: usize) -> Option<Open> {
    let atom_a = *a.sites.get(sa)?;
    let atom_b = *b.sites.get(sb)?;
    let offset = a.mol.n_atoms() as u8;
    if a.mol.n_atoms() + b.mol.n_atoms() > MAX_ATOMS {
        return None;
    }

    let mut atoms = a.mol.atoms.clone();
    atoms.extend_from_slice(&b.mol.atoms);

    let mut bonds = a.mol.bonds.clone();
    for bond in &b.mol.bonds {
        bonds.push(Bond::new(bond.a + offset, bond.b + offset, bond.order));
    }
    bonds.push(Bond::new(atom_a, atom_b + offset, 1));

    let mut sites: Vec<u8> = a.sites.clone();
    sites.remove(sa);
    for (i, &s) in b.sites.iter().enumerate() {
        if i != sb {
            sites.push(s + offset);
        }
    }

    Some(Open {
        mol: Molecule::new(atoms, bonds),
        sites,
    })
}

/// `A + B -> C + D`: cut `bond_a` in `a` and `bond_b` in `b`, then recombine
/// the four halves.
///
/// Cutting `A1-A2` and `B1-B2` admits two recombinations, and which one is
/// chemically interesting depends on the molecules: `swap` selects
/// `{A1B1, A2B2}` rather than `{A1B2, A2B1}`. Either can come out as the
/// identity when two halves happen to be the same fragment (cutting C-H and
/// O-H and pairing H with H just rebuilds the reactants), so the caller must
/// reject reactions whose products match their reactants.
pub fn exchange(
    a: &Molecule,
    bond_a: usize,
    b: &Molecule,
    bond_b: usize,
    swap: bool,
) -> Option<(Molecule, Molecule)> {
    let (a1, a2) = split_bond(a, bond_a)?;
    let (b1, b2) = split_bond(b, bond_b)?;
    let (first, second) = if swap { (&b1, &b2) } else { (&b2, &b1) };
    let c = merge(&a1, 0, first, 0)?.close()?;
    let d = merge(&a2, 0, second, 0)?.close()?;
    Some((c, d))
}

/// `A + B -> C`: open a multiple bond in `a` and add the two halves of `b`
/// across it.
pub fn addition(a: &Molecule, bond_a: usize, b: &Molecule, bond_b: usize) -> Option<Molecule> {
    let opened = open_multiple(a, bond_a)?;
    let (b1, b2) = split_bond(b, bond_b)?;
    // Consume site 0, then whatever site remains.
    let half = merge(&opened, 0, &b1, 0)?;
    merge(&half, 0, &b2, 0)?.close()
}

/// Which atoms are reachable from `start` using `bonds`.
fn reachable(n: usize, bonds: &[Bond], start: u8) -> Vec<bool> {
    let mut seen = vec![false; n];
    let mut stack = vec![start];
    seen[start as usize] = true;
    while let Some(i) = stack.pop() {
        for bond in bonds {
            if let Some(j) = bond.other(i) {
                if !seen[j as usize] {
                    seen[j as usize] = true;
                    stack.push(j);
                }
            }
        }
    }
    seen
}

/// Extract the induced subgraph on `atoms`, renumbering to `0..atoms.len()`.
/// `open_atoms` are translated into the resulting site list.
fn subgraph(mol: &Molecule, bonds: &[Bond], atoms: &[u8], open_atoms: &[u8]) -> Open {
    let mut map = vec![u8::MAX; mol.n_atoms()];
    for (new, &old) in atoms.iter().enumerate() {
        map[old as usize] = new as u8;
    }
    let new_atoms: Vec<u8> = atoms.iter().map(|&i| mol.atoms[i as usize]).collect();
    let new_bonds: Vec<Bond> = bonds
        .iter()
        .filter(|b| map[b.a as usize] != u8::MAX && map[b.b as usize] != u8::MAX)
        .map(|b| Bond::new(map[b.a as usize], map[b.b as usize], b.order))
        .collect();
    Open {
        mol: Molecule::new(new_atoms, new_bonds),
        sites: open_atoms.iter().map(|&i| map[i as usize]).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{C, H, N_ELEMENTS, O};

    fn water() -> Molecule {
        Molecule::new(vec![O, H, H], vec![Bond::new(0, 1, 1), Bond::new(0, 2, 1)])
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

    fn ethene() -> Molecule {
        Molecule::new(
            vec![C, C, H, H, H, H],
            vec![
                Bond::new(0, 1, 2),
                Bond::new(0, 2, 1),
                Bond::new(0, 3, 1),
                Bond::new(1, 4, 1),
                Bond::new(1, 5, 1),
            ],
        )
    }

    fn total_formula(ms: &[&Molecule]) -> [u16; N_ELEMENTS] {
        let mut f = [0u16; N_ELEMENTS];
        for m in ms {
            crate::molecule::formula_add(&mut f, &m.formula());
        }
        f
    }

    #[test]
    fn split_and_rejoin_is_the_identity() {
        let m = methane();
        let (a, b) = split_bond(&m, 0).expect("cuttable");
        let rejoined = merge(&a, 0, &b, 0).unwrap().close().expect("valid");
        assert_eq!(
            crate::canon::canonical_key(&rejoined),
            crate::canon::canonical_key(&m)
        );
    }

    #[test]
    fn exchange_conserves_every_element() {
        let (a, b) = (methane(), water());
        for swap in [false, true] {
            let (c, d) = exchange(&a, 0, &b, 0, swap).expect("exchange");
            assert!(c.is_valid() && d.is_valid(), "products must be saturated");
            assert_eq!(total_formula(&[&a, &b]), total_formula(&[&c, &d]));
        }
    }

    #[test]
    fn exchange_pairings_give_methanol_or_the_identity() {
        // Cut C-H and O-H. Pairing CH3 with OH gives methanol and hydrogen;
        // pairing CH3 with the water's spare H just rebuilds the reactants.
        let sorted = |swap| {
            let (c, d) = exchange(&methane(), 0, &water(), 0, swap).unwrap();
            let mut n = [c.formula_string(), d.formula_string()];
            n.sort();
            n
        };
        assert_eq!(sorted(true), ["CH4O".to_string(), "H2".to_string()]);
        assert_eq!(sorted(false), ["CH4".to_string(), "H2O".to_string()]);
    }

    #[test]
    fn addition_conserves_every_element() {
        let (a, b) = (ethene(), water());
        let c = addition(&a, 0, &b, 0).expect("addition");
        assert!(c.is_valid());
        assert_eq!(total_formula(&[&a, &b]), total_formula(&[&c]));
        assert_eq!(c.formula_string(), "C2H6O");
    }

    #[test]
    fn ring_bonds_are_not_cuttable() {
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
        assert!(split_bond(&ring, 0).is_none(), "ring bond must not split");
        assert!(split_bond(&ring, 3).is_some(), "C-H bond must split");
    }

    #[test]
    fn multiple_bonds_are_not_split_only_opened() {
        assert!(split_bond(&ethene(), 0).is_none());
        let opened = open_multiple(&ethene(), 0).expect("openable");
        assert_eq!(opened.sites.len(), 2);
        assert!(open_multiple(&methane(), 0).is_none());
    }

    #[test]
    fn merge_respects_the_atom_limit() {
        let h2 = Molecule::new(vec![H, H], vec![Bond::new(0, 1, 1)]);
        let (_, spare) = split_bond(&h2, 0).unwrap();
        let full = Open {
            mol: Molecule::new(vec![C; MAX_ATOMS], vec![]),
            sites: vec![0],
        };
        assert!(
            merge(&full, 0, &spare, 0).is_none(),
            "must refuse to overflow"
        );
        let room = Open {
            mol: Molecule::new(vec![C; MAX_ATOMS - 1], vec![]),
            sites: vec![0],
        };
        assert!(merge(&room, 0, &spare, 0).is_some());
    }
}
