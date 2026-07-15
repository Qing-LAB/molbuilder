"""Phase 2 (atom-annotations.md § 3): .molstruct.json v4 annotations --
round-trip + v3 back-read + dual-write."""
import json
import numpy as np

from molbuilder.structure import Structure, AtomChannel
from molbuilder.sidecars import molstruct


def _hash():
    return "a" * 32


def test_to_dict_v4_dual_writes_annotations_and_builtins():
    d = molstruct.to_dict(
        n_atoms_total=5, structure_hash=_hash(),
        regions={"L-electrode": [0, 1]}, frozen_atoms=[4],
        annotations={"charge": AtomChannel("value", {0: -1.0, 1: 0.5})})
    assert d["schema_version"] == 5   # current schema (kgrid dropped @ v5)
    assert d["regions"] == {"L-electrode": [0, 1]}      # dual-write: built-ins kept
    assert d["frozen_atoms"] == [4]
    assert d["annotations"]["charge"] == {
        "kind": "value", "data": {"0": -1.0, "1": 0.5}}  # int keys -> str in JSON


def test_save_load_apply_roundtrips_annotations(tmp_path):
    s = Structure(elements=["C"] * 5,
                  positions=np.arange(15, dtype=float).reshape(5, 3))
    s.set_channel("charge", AtomChannel("value", {0: -1.0, 2: 9.0}))
    s.set_channel("tail", AtomChannel("tag", [3, 4]))
    p = tmp_path / "x.molstruct.json"
    molstruct.save(p, molstruct.to_dict(
        n_atoms_total=5, structure_hash=_hash(),
        regions=s.regions, frozen_atoms=s.frozen_atoms,
        annotations=s.annotations))
    loaded = molstruct.load(p)
    assert loaded["schema_version"] == 5   # current schema (kgrid dropped @ v5)
    back = Structure(elements=["C"] * 5,
                     positions=np.arange(15, dtype=float).reshape(5, 3))
    molstruct.apply_to_structure(back, loaded)
    assert back.get_channel("charge").data == {0: -1.0, 2: 9.0}   # keys back to int
    assert back.get_channel("tail").data == [3, 4]


def test_v3_sidecar_back_reads_with_empty_annotations(tmp_path):
    """A pre-existing v3 file (no annotations) still loads; annotations empty,
    regions/frozen intact."""
    p = tmp_path / "old.molstruct.json"
    p.write_text(json.dumps({
        "schema_version": 3, "n_atoms_total": 3, "structure_hash": _hash(),
        "regions": {"bridge": [1]}, "frozen_atoms": [0],
        "created_by": "old", "created_at": "2026-01-01T00:00:00Z",
    }))
    loaded = molstruct.load(p)
    back = Structure(elements=["C", "C", "C"],
                     positions=np.zeros((3, 3)))
    molstruct.apply_to_structure(back, loaded)
    assert back.annotations == {}                        # no channels from v3
    assert back.regions == {"bridge": [1]} and back.frozen_atoms == [0]


def test_load_rejects_out_of_range_annotation(tmp_path):
    import pytest
    p = tmp_path / "bad.molstruct.json"
    p.write_text(json.dumps({
        "schema_version": 4, "n_atoms_total": 2, "structure_hash": _hash(),
        "annotations": {"x": {"kind": "tag", "data": [5]}},   # idx 5 >= 2
    }))
    # Annotation indices are validated against n_atoms_total AT LOAD
    # (mirrors regions/frozen), so a bad sidecar fails early + clearly.
    with pytest.raises(Exception, match="out of range"):
        molstruct.load(p)
