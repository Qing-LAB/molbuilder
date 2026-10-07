""".molstruct.json annotations (model/structure-annotations.md § 3)."""
import json
import numpy as np

from molbuilder.structure import Structure, AtomChannel, annotations_to_json
from molbuilder.sidecars import molstruct


def _hash():
    return "a" * 32


def test_to_dict_v4_dual_writes_annotations_and_builtins():
    d = molstruct.to_dict(
        {"regions": {"L-electrode": [0, 1], "frozen_atoms": [4]},
         "annotations": annotations_to_json(
             {"charge": AtomChannel("value", {0: -1.0, 1: 0.5})})},
        n_atoms_total=5, structure_hash=_hash())
    assert d["schema_version"] == molstruct.SCHEMA_VERSION
    # ONE label store: the reserved label is in `regions` with the others, and
    # there is no second key beside it.
    assert d["regions"] == {"L-electrode": [0, 1], "frozen_atoms": [4]}
    assert "frozen_atoms" not in d
    assert d["annotations"]["charge"] == {
        "kind": "value", "data": {"0": -1.0, "1": 0.5}}  # int keys -> str in JSON


def test_save_load_apply_roundtrips_annotations(tmp_path):
    s = Structure(elements=["C"] * 5,
                  positions=np.arange(15, dtype=float).reshape(5, 3))
    s.set_channel("charge", AtomChannel("value", {0: -1.0, 2: 9.0}))
    s.set_channel("tail", AtomChannel("tag", [3, 4]))
    p = tmp_path / "x.molstruct.json"
    molstruct.save(p, molstruct.to_dict(
        {"regions": s.regions,   # the whole label store, reserved ones in it
         "annotations": annotations_to_json(s.annotations)},
        n_atoms_total=5, structure_hash=_hash()))
    loaded = molstruct.load(p)
    assert loaded["schema_version"] == molstruct.SCHEMA_VERSION
    back = Structure(elements=["C"] * 5,
                     positions=np.arange(15, dtype=float).reshape(5, 3))
    molstruct.apply_to_structure(back, loaded)
    assert back.get_channel("charge").data == {0: -1.0, 2: 9.0}   # keys back to int
    assert back.get_channel("tail").data == [3, 4]


def test_the_annotations_key_is_optional(tmp_path):
    """A sidecar carrying no extensible channels loads with an empty set, its
    labels intact."""
    p = tmp_path / "old.molstruct.json"
    p.write_text(json.dumps({
        "schema_version": 7, "n_atoms_total": 3, "structure_hash": _hash(),
        "regions": {"bridge": [1], "frozen_atoms": [0]},
        "created_by": "old", "created_at": "2026-01-01T00:00:00Z",
    }))
    loaded = molstruct.load(p)
    back = Structure(elements=["C", "C", "C"],
                     positions=np.zeros((3, 3)))
    molstruct.apply_to_structure(back, loaded)
    assert back.annotations == {}                        # key absent -> empty
    # The reserved label sits in the one store with the others.
    assert back.regions == {"bridge": [1], "frozen_atoms": [0]}
    assert back.frozen_atoms == [0]


def test_load_rejects_out_of_range_annotation(tmp_path):
    import pytest
    p = tmp_path / "bad.molstruct.json"
    p.write_text(json.dumps({
        "schema_version": 7, "n_atoms_total": 2, "structure_hash": _hash(),
        "annotations": {"x": {"kind": "tag", "data": [5]}},   # idx 5 >= 2
    }))
    # Annotation indices are validated against n_atoms_total AT LOAD
    # (mirrors regions/frozen), so a bad sidecar fails early + clearly.
    with pytest.raises(Exception, match="out of range"):
        molstruct.load(p)
