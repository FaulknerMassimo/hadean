//! Snapshots: save a world, load it back bit-identically.
//!
//! A run is reproducible from `(config, tick_count)`, so in principle a
//! snapshot is redundant. In practice, replaying forty hours of simulation to
//! look at something is not a workflow, so the state is written out whole and
//! compressed.
//!
//! The chemistry is *not* stored. It is a pure function of the seed and the
//! chemistry parameters, both of which live in the config, so it is
//! regenerated on load and its digest checked against the one recorded. That
//! keeps snapshots small -- the compound pool with all its molecule graphs is
//! far bigger than the fields for a development-sized grid -- and it turns a
//! silent version skew into a loud error.

use std::io::{Read, Write};
use std::path::Path;

use hadean_cell::{Cell, CellState};
use hadean_chem::element::N_ELEMENTS;
use hadean_core::hash::HashState;

use crate::audit::EnergyState;
use crate::config::WorldConfig;
use crate::world::World;

const MAGIC: &[u8; 8] = b"HADEANv1";
const FORMAT: u32 = 3;
/// zstd level 3 is the usual sweet spot: most of the ratio, little of the cost.
const COMPRESSION: i32 = 3;

/// Serialise a world into a compressed byte stream.
pub fn save(world: &World) -> anyhow::Result<Vec<u8>> {
    let mut raw = Vec::new();
    let config = world.config.to_toml();

    raw.extend_from_slice(MAGIC);
    put_u32(&mut raw, FORMAT);
    put_bytes(&mut raw, config.as_bytes());
    put_u64(&mut raw, world.config.digest());
    put_u64(&mut raw, world.chem.state_hash());
    put_u64(&mut raw, world.tick());

    put_u32(&mut raw, world.amounts.n_compounds as u32);
    put_u64(&mut raw, world.amounts.n_voxels as u64);
    put_f32s(&mut raw, &world.amounts.data);
    // The transport residuals are part of the conserved state, not scratch:
    // they hold material the fields own but cannot yet represent. Dropping
    // them across a save would reintroduce the very leak they exist to stop.
    put_f32s(&mut raw, &world.residual.data);

    put_f32(&mut raw, world.heat.reference);
    put_f32s(&mut raw, &world.heat.deviation.data);
    put_f32s(&mut raw, &world.heat.residual.data);

    put_u64(&mut raw, world.cells.next_id);
    put_u32(&mut raw, world.cells.metabolic_reaction.unwrap_or(u32::MAX));
    put_u64(&mut raw, world.cells.births);
    put_u64(&mut raw, world.cells.deaths);
    put_u64(&mut raw, world.cells.cells.len() as u64);
    for cell in &world.cells.cells {
        put_u64(&mut raw, cell.id);
        put_u64(&mut raw, cell.parent.unwrap_or(u64::MAX));
        for &x in &cell.pos {
            put_f32(&mut raw, x);
        }
        put_f32(&mut raw, cell.radius);
        put_f64s(&mut raw, &cell.contents);
        put_f64(&mut raw, cell.reserve);
        put_f32(&mut raw, cell.damage);
        put_f32(&mut raw, cell.age);
        raw.push(cell.state as u8);
        put_u32(&mut raw, cell.generation);
    }

    let a = &world.audit;
    put_f64(&mut raw, a.ledger.light_in);
    put_f64(&mut raw, a.ledger.vent_heat_in);
    put_f64(&mut raw, a.ledger.vent_chemical_in);
    put_f64(&mut raw, a.ledger.radiated_out);
    put_f64(&mut raw, a.baseline.chemical);
    put_f64(&mut raw, a.baseline.thermal);
    put_f64(&mut raw, a.baseline.cellular);
    for e in 0..N_ELEMENTS {
        put_f64(&mut raw, a.baseline_elements[e]);
    }
    for e in 0..N_ELEMENTS {
        put_f64(&mut raw, a.elements_in[e]);
    }

    Ok(zstd::encode_all(raw.as_slice(), COMPRESSION)?)
}

/// Rebuild a world from a compressed byte stream.
pub fn load(bytes: &[u8]) -> anyhow::Result<World> {
    let raw = zstd::decode_all(bytes)?;
    let mut cursor = Cursor { data: &raw, at: 0 };

    let magic = cursor.take(8)?;
    if magic != MAGIC {
        anyhow::bail!("not a hadean snapshot");
    }
    let format = cursor.u32()?;
    if format != FORMAT {
        anyhow::bail!("snapshot format {format}, this build reads {FORMAT}");
    }

    let config_text = std::str::from_utf8(cursor.bytes()?)?;
    let config = WorldConfig::from_toml(config_text)?;
    let config_digest = cursor.u64()?;
    if config.digest() != config_digest {
        anyhow::bail!("snapshot config does not match its own digest");
    }
    let chem_hash = cursor.u64()?;
    let tick = cursor.u64()?;

    // Rebuilding regenerates the chemistry and reseeds the soup; every part of
    // that is then overwritten from the snapshot.
    let mut world = World::new(config)?;
    if world.chem.state_hash() != chem_hash {
        anyhow::bail!(
            "chemistry regenerated differently from the snapshot; \
             the generator has changed since it was written"
        );
    }

    let n_compounds = cursor.u32()? as usize;
    let n_voxels = cursor.u64()? as usize;
    if n_compounds != world.amounts.n_compounds || n_voxels != world.amounts.n_voxels {
        anyhow::bail!(
            "snapshot has {n_compounds} compounds over {n_voxels} voxels, \
             world has {} over {}",
            world.amounts.n_compounds,
            world.amounts.n_voxels
        );
    }
    world.amounts.data = cursor.f32s()?;
    world.residual.data = cursor.f32s()?;

    world.heat.reference = cursor.f32()?;
    world.heat.deviation.data = cursor.f32s()?;
    world.heat.residual.data = cursor.f32s()?;

    world.cells.next_id = cursor.u64()?;
    let reaction = cursor.u32()?;
    world.cells.metabolic_reaction = (reaction != u32::MAX).then_some(reaction);
    world.cells.births = cursor.u64()?;
    world.cells.deaths = cursor.u64()?;
    let n_cells = cursor.u64()? as usize;
    world.cells.cells.clear();
    world.cells.cells.reserve(n_cells);
    for _ in 0..n_cells {
        let id = cursor.u64()?;
        let parent = cursor.u64()?;
        let pos = [cursor.f32()?, cursor.f32()?, cursor.f32()?];
        let radius = cursor.f32()?;
        let contents = cursor.f64s()?;
        if contents.len() != n_compounds {
            anyhow::bail!(
                "cell {id} has {} compounds, expected {n_compounds}",
                contents.len()
            );
        }
        let reserve = cursor.f64()?;
        let damage = cursor.f32()?;
        let age = cursor.f32()?;
        let state = match cursor.byte()? {
            0 => CellState::Alive,
            1 => CellState::Decomposing,
            2 => CellState::Dormant,
            value => anyhow::bail!("cell {id} has unknown state {value}"),
        };
        let generation = cursor.u32()?;
        world.cells.cells.push(Cell {
            id,
            parent: (parent != u64::MAX).then_some(parent),
            pos,
            radius,
            contents,
            reserve,
            damage,
            age,
            state,
            generation,
        });
    }

    world.audit.ledger.light_in = cursor.f64()?;
    world.audit.ledger.vent_heat_in = cursor.f64()?;
    world.audit.ledger.vent_chemical_in = cursor.f64()?;
    world.audit.ledger.radiated_out = cursor.f64()?;
    world.audit.baseline = EnergyState {
        chemical: cursor.f64()?,
        thermal: cursor.f64()?,
        cellular: cursor.f64()?,
    };
    for e in 0..N_ELEMENTS {
        world.audit.baseline_elements[e] = cursor.f64()?;
    }
    for e in 0..N_ELEMENTS {
        world.audit.elements_in[e] = cursor.f64()?;
    }

    world.clock.tick = tick;
    Ok(world)
}

pub fn write_file(world: &World, path: &Path) -> anyhow::Result<u64> {
    let bytes = save(world)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}

pub fn read_file(path: &Path) -> anyhow::Result<World> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    load(&bytes)
}

// --- little-endian primitives -------------------------------------------

fn put_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_f32(out: &mut Vec<u8>, x: f32) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_f64(out: &mut Vec<u8>, x: f64) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_f64s(out: &mut Vec<u8>, xs: &[f64]) {
    put_u64(out, xs.len() as u64);
    for &x in xs {
        put_f64(out, x);
    }
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn put_f32s(out: &mut Vec<u8>, xs: &[f32]) {
    put_u64(out, xs.len() as u64);
    for &x in xs {
        put_f32(out, x);
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        if self.at + n > self.data.len() {
            anyhow::bail!("snapshot ended early");
        }
        let s = &self.data[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }

    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn byte(&mut self) -> anyhow::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn f32(&mut self) -> anyhow::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn f64(&mut self) -> anyhow::Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn bytes(&mut self) -> anyhow::Result<&'a [u8]> {
        let n = self.u64()? as usize;
        self.take(n)
    }

    fn f32s(&mut self) -> anyhow::Result<Vec<f32>> {
        let n = self.u64()? as usize;
        let raw = self.take(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is four bytes")))
            .collect())
    }

    fn f64s(&mut self) -> anyhow::Result<Vec<f64>> {
        let n = self.u64()? as usize;
        let raw = self.take(
            n.checked_mul(8)
                .ok_or_else(|| anyhow::anyhow!("invalid vector length"))?,
        )?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("chunk is eight bytes")))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GridConfig;

    fn small() -> WorldConfig {
        WorldConfig {
            seed: 5,
            grid: GridConfig {
                nx: 8,
                ny: 8,
                nz: 6,
                dx: 25.0e-6,
            },
            ..Default::default()
        }
    }

    #[test]
    fn a_snapshot_round_trips_exactly() {
        let mut w = World::new(small()).expect("builds");
        w.run(40);
        let digest = w.state_digest();
        let bytes = save(&w).expect("saves");
        let back = load(&bytes).expect("loads");
        assert_eq!(back.tick(), w.tick());
        assert_eq!(back.state_digest(), digest);
        assert_eq!(back.amounts.data, w.amounts.data);
        assert_eq!(back.heat.deviation.data, w.heat.deviation.data);
        assert_eq!(back.audit.ledger, w.audit.ledger);
    }

    #[test]
    fn a_dormant_cell_survives_a_round_trip() {
        // `CellState` is written as `state as u8`, so a state the reader does
        // not know about is a hard error rather than a silent misread. That is
        // the right behaviour and it means every new state needs a decode arm;
        // this is the test that notices when one is missing.
        let mut w = World::new(small()).expect("builds");
        w.run(40);
        w.cells.cells.push(Cell {
            id: 9_000,
            parent: None,
            pos: [25.0e-6, 25.0e-6, 25.0e-6],
            radius: w.config.cells.birth_radius,
            contents: vec![0.0; w.chem.n_compounds()],
            reserve: 1.0e-12,
            damage: 0.25,
            age: 3.0,
            state: CellState::Dormant,
            generation: 2,
        });
        let digest = w.state_digest();

        let back = load(&save(&w).expect("saves")).expect("loads");
        assert_eq!(back.state_digest(), digest);
        assert_eq!(back.cells.dormant(), 1);
        assert_eq!(back.cells.cells.last().expect("the cell"), w.cells.cells.last().expect("the cell"));
    }

    #[test]
    fn a_reloaded_world_continues_identically() {
        // This is the property that makes snapshots worth having: stopping and
        // resuming must be indistinguishable from never having stopped.
        let mut a = World::new(small()).expect("builds");
        a.run(30);
        let bytes = save(&a).expect("saves");
        let mut b = load(&bytes).expect("loads");

        a.run(30);
        b.run(30);
        assert_eq!(a.tick(), b.tick());
        assert_eq!(a.state_digest(), b.state_digest());
    }

    #[test]
    fn compression_actually_helps() {
        let w = World::new(small()).expect("builds");
        let compressed = save(&w).expect("saves").len();
        let raw = w.amounts.data.len() * 4 + w.heat.deviation.data.len() * 4;
        assert!(compressed < raw, "{compressed} bytes vs {raw} raw");
    }

    #[test]
    fn a_truncated_snapshot_is_refused() {
        let w = World::new(small()).expect("builds");
        let bytes = save(&w).expect("saves");
        // Truncating the compressed stream must not panic.
        assert!(load(&bytes[..bytes.len() / 2]).is_err());
    }

    #[test]
    fn foreign_bytes_are_refused() {
        assert!(load(b"not a snapshot at all").is_err());
    }

    #[test]
    fn a_file_round_trips() {
        let dir = std::env::temp_dir().join("hadean-snapshot-test");
        let path = dir.join("world.snap");
        let mut w = World::new(small()).expect("builds");
        w.run(10);
        let size = write_file(&w, &path).expect("writes");
        assert!(size > 0);
        let back = read_file(&path).expect("reads");
        assert_eq!(back.state_digest(), w.state_digest());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
