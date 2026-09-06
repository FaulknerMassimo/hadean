//! Canonical labelling of molecules.
//!
//! Two molecules are the same compound when their atom graphs are isomorphic,
//! not when their atom arrays happen to be written in the same order. Reaction
//! products come out of a rewrite in whatever order the rewrite produced them,
//! so we need a canonical form to intern them against the compound pool.
//!
//! The algorithm is standard individualisation-refinement:
//!
//! 1. Colour atoms by element, then refine by the multiset of neighbouring
//!    colours until the partition is stable (1-dimensional Weisfeiler-Leman).
//! 2. If a colour class still has more than one member, individualise each
//!    member in turn, re-refine, and recurse.
//! 3. Take the lexicographically smallest encoding over all leaves.
//!
//! A node budget bounds the search. Exceeding it is not a correctness problem
//! for the simulation -- the fallback ordering is still element-consistent, so
//! two aliased molecules always have the same formula and the reaction network
//! stays mass-balanced -- but it would let one compound be interned twice.
//! Molecules here are at most a dozen atoms, so it does not happen in practice.

use crate::molecule::Molecule;

/// The canonical encoding of a molecule. Ordering is lexicographic and
/// meaningful, so this doubles as a stable sort key for the compound table.
pub type MoleculeKey = Vec<u8>;

const NODE_BUDGET: u32 = 20_000;

/// Canonical key for `mol`. Isomorphic molecules produce equal keys.
pub fn canonical_key(mol: &Molecule) -> MoleculeKey {
    let order = canonical_order(mol);
    encode(mol, &order)
}

/// The canonical atom ordering: `order[new_index] = old_index`.
pub fn canonical_order(mol: &Molecule) -> Vec<u8> {
    let n = mol.n_atoms();
    if n == 1 {
        return vec![0];
    }
    let adj = adjacency(mol);
    let colours = refine(
        &adj,
        &mol.atoms.iter().map(|&e| e as u32).collect::<Vec<_>>(),
    );
    let mut budget = NODE_BUDGET;
    let mut best: Option<(Vec<u8>, Vec<u8>)> = None;
    search(mol, &adj, &colours, &mut budget, &mut best);
    match best {
        Some((order, _)) => order,
        None => fallback_order(&colours),
    }
}

/// Adjacency list: `adj[i] = [(neighbour, bond order)]`.
fn adjacency(mol: &Molecule) -> Vec<Vec<(u8, u8)>> {
    let mut adj = vec![Vec::new(); mol.n_atoms()];
    for b in &mol.bonds {
        adj[b.a as usize].push((b.b, b.order));
        adj[b.b as usize].push((b.a, b.order));
    }
    for a in &mut adj {
        a.sort_unstable();
    }
    adj
}

/// Refine a colouring until stable, by the multiset of neighbouring colours.
fn refine(adj: &[Vec<(u8, u8)>], initial: &[u32]) -> Vec<u32> {
    let n = adj.len();
    let mut colours = dense_rank(initial.iter().map(|&c| (c, Vec::new())).collect());
    loop {
        let sigs: Vec<(u32, Vec<(u8, u32)>)> = (0..n)
            .map(|i| {
                let mut nb: Vec<(u8, u32)> = adj[i]
                    .iter()
                    .map(|&(j, o)| (o, colours[j as usize]))
                    .collect();
                nb.sort_unstable();
                (colours[i], nb)
            })
            .collect();
        let next = dense_rank(sigs);
        if next == colours {
            return colours;
        }
        colours = next;
    }
}

/// Map signatures to dense ranks, ordered lexicographically so the result does
/// not depend on iteration order.
fn dense_rank(sigs: Vec<(u32, Vec<(u8, u32)>)>) -> Vec<u32> {
    let mut unique: Vec<&(u32, Vec<(u8, u32)>)> = sigs.iter().collect();
    unique.sort();
    unique.dedup();
    sigs.iter()
        .map(|s| unique.binary_search(&s).expect("signature present") as u32)
        .collect()
}

/// Depth-first individualisation. Records the best (smallest) encoding found.
fn search(
    mol: &Molecule,
    adj: &[Vec<(u8, u8)>],
    colours: &[u32],
    budget: &mut u32,
    best: &mut Option<(Vec<u8>, Vec<u8>)>,
) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;

    // The target class is the smallest non-singleton class, lowest colour first.
    let mut classes: Vec<(u32, Vec<u8>)> = Vec::new();
    for (i, &c) in colours.iter().enumerate() {
        match classes.iter_mut().find(|(k, _)| *k == c) {
            Some((_, v)) => v.push(i as u8),
            None => classes.push((c, vec![i as u8])),
        }
    }
    classes.sort_by_key(|(c, _)| *c);

    let target = classes
        .iter()
        .filter(|(_, v)| v.len() > 1)
        .min_by_key(|(c, v)| (v.len(), *c));

    match target {
        None => {
            // Discrete partition: the colouring is a total order on atoms.
            let order = fallback_order(colours);
            let code = encode(mol, &order);
            if best.as_ref().map_or(true, |(_, b)| code < *b) {
                *best = Some((order, code));
            }
        }
        Some((_, members)) => {
            for &v in members {
                let mut individualised: Vec<u32> = colours.iter().map(|c| c * 2).collect();
                individualised[v as usize] += 1;
                let refined = refine(adj, &individualised);
                search(mol, adj, &refined, budget, best);
                if *budget == 0 {
                    return;
                }
            }
        }
    }
}

/// Order atoms by `(colour, index)`. Exact when the colouring is discrete.
fn fallback_order(colours: &[u32]) -> Vec<u8> {
    let mut order: Vec<u8> = (0..colours.len() as u8).collect();
    order.sort_by_key(|&i| (colours[i as usize], i));
    order
}

/// Encode a molecule under an atom ordering.
///
/// Layout: `[n_atoms, elements in new order.., n_bonds, (a, b, order)..]`
/// with bonds sorted, so equal encodings mean identical labelled graphs.
fn encode(mol: &Molecule, order: &[u8]) -> Vec<u8> {
    let n = mol.n_atoms();
    // position[old] = new
    let mut position = vec![0u8; n];
    for (new, &old) in order.iter().enumerate() {
        position[old as usize] = new as u8;
    }
    let mut out = Vec::with_capacity(2 + n + mol.bonds.len() * 3);
    out.push(n as u8);
    for &old in order {
        out.push(mol.atoms[old as usize]);
    }
    let mut bonds: Vec<(u8, u8, u8)> = mol
        .bonds
        .iter()
        .map(|b| {
            let (x, y) = (position[b.a as usize], position[b.b as usize]);
            (x.min(y), x.max(y), b.order)
        })
        .collect();
    bonds.sort_unstable();
    out.push(bonds.len() as u8);
    for (a, b, o) in bonds {
        out.extend_from_slice(&[a, b, o]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{C, H, N, O};
    use crate::molecule::{Bond, Molecule};

    fn ethanol_a() -> Molecule {
        // C-C-O with hydrogens: atoms 0=C 1=C 2=O then H's.
        Molecule::new(
            vec![C, C, O, H, H, H, H, H, H],
            vec![
                Bond::new(0, 1, 1),
                Bond::new(1, 2, 1),
                Bond::new(2, 8, 1),
                Bond::new(0, 3, 1),
                Bond::new(0, 4, 1),
                Bond::new(0, 5, 1),
                Bond::new(1, 6, 1),
                Bond::new(1, 7, 1),
            ],
        )
    }

    fn ethanol_b() -> Molecule {
        // Same molecule, atoms written in a different order.
        Molecule::new(
            vec![H, O, H, C, H, C, H, H, H],
            vec![
                Bond::new(5, 3, 1),
                Bond::new(3, 1, 1),
                Bond::new(1, 0, 1),
                Bond::new(5, 2, 1),
                Bond::new(5, 4, 1),
                Bond::new(5, 6, 1),
                Bond::new(3, 7, 1),
                Bond::new(3, 8, 1),
            ],
        )
    }

    #[test]
    fn isomorphic_molecules_share_a_key() {
        let a = ethanol_a();
        let b = ethanol_b();
        assert!(a.is_valid() && b.is_valid());
        assert_eq!(a.formula_string(), b.formula_string());
        assert_eq!(canonical_key(&a), canonical_key(&b));
    }

    #[test]
    fn isomers_get_different_keys() {
        // Dimethyl ether C-O-C vs ethanol C-C-O: same formula C2H6O.
        let ether = Molecule::new(
            vec![C, O, C, H, H, H, H, H, H],
            vec![
                Bond::new(0, 1, 1),
                Bond::new(1, 2, 1),
                Bond::new(0, 3, 1),
                Bond::new(0, 4, 1),
                Bond::new(0, 5, 1),
                Bond::new(2, 6, 1),
                Bond::new(2, 7, 1),
                Bond::new(2, 8, 1),
            ],
        );
        assert!(ether.is_valid());
        assert_eq!(ether.formula_string(), ethanol_a().formula_string());
        assert_ne!(canonical_key(&ether), canonical_key(&ethanol_a()));
    }

    #[test]
    fn bond_order_is_part_of_identity() {
        let ethene = Molecule::new(
            vec![C, C, H, H, H, H],
            vec![
                Bond::new(0, 1, 2),
                Bond::new(0, 2, 1),
                Bond::new(0, 3, 1),
                Bond::new(1, 4, 1),
                Bond::new(1, 5, 1),
            ],
        );
        let ethyne = Molecule::new(
            vec![C, C, H, H],
            vec![Bond::new(0, 1, 3), Bond::new(0, 2, 1), Bond::new(1, 3, 1)],
        );
        assert!(ethene.is_valid() && ethyne.is_valid());
        assert_ne!(canonical_key(&ethene), canonical_key(&ethyne));
    }

    #[test]
    fn symmetric_rings_canonicalise() {
        // A six-membered all-carbon ring, built starting from two rotations.
        let ring = |shift: usize| {
            let idx = |i: usize| ((i + shift) % 6) as u8;
            let mut bonds = Vec::new();
            for i in 0..6 {
                bonds.push(Bond::new(
                    idx(i),
                    idx(i + 1),
                    if i % 2 == 0 { 2 } else { 1 },
                ));
                bonds.push(Bond::new(idx(i), 6 + i as u8, 1));
            }
            let mut atoms = vec![C; 6];
            atoms.extend(std::iter::repeat(H).take(6));
            Molecule::new(atoms, bonds)
        };
        let a = ring(0);
        let b = ring(3);
        assert!(a.is_valid(), "ring invalid");
        assert_eq!(canonical_key(&a), canonical_key(&b));
    }

    #[test]
    fn key_is_stable_across_calls() {
        let m = ethanol_a();
        assert_eq!(canonical_key(&m), canonical_key(&m));
    }

    #[test]
    fn different_elements_differ() {
        let a = Molecule::new(
            vec![N, H, H, H],
            vec![Bond::new(0, 1, 1), Bond::new(0, 2, 1), Bond::new(0, 3, 1)],
        );
        let b = Molecule::new(vec![O, H, H], vec![Bond::new(0, 1, 1), Bond::new(0, 2, 1)]);
        assert_ne!(canonical_key(&a), canonical_key(&b));
    }
}
