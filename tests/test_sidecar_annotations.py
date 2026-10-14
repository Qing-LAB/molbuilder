""".molstruct.json annotations (model/structure-annotations.md § 3)."""
import json
import numpy as np

from molbuilder.structure import Structure, AtomChannel, annotations_to_json
from molbuilder.sidecars import molstruct
from molbuilder.workingcopy_structure import StructureCodec


def _hash():
    """A geometry hash's form -- the builder checks the form; comparing it
    with a file is the read door's (`structure-molstruct.md` § 3)."""
    return "a" * 64


def test_to_dict_v4_dual_writes_annotations_and_builtins():
    d = molstruct.to_dict(
        {"regions": {"L-electrode": [0, 1], "frozen_atoms": [4]},
         "annotations": annotations_to_json(
             {"charge": AtomChannel("value", {0: -1.0, 1: 0.5})})},
        n_atoms_total=5, geometry_hash=_hash())
    assert d["schema_version"] == molstruct.SCHEMA_VERSION
    # ONE label store: the reserved label is in `regions` with the others, and
    # there is no second key beside it.
    assert d["regions"] == {"L-electrode": [0, 1], "frozen_atoms": [4]}
    assert "frozen_atoms" not in d
    assert d["annotations"]["charge"] == {
        "kind": "value", "data": {"0": -1.0, "1": 0.5}}  # int keys -> str in JSON


def test_the_pair_roundtrips_annotations(tmp_path):
    s = Structure(elements=["C"] * 5,
                  positions=np.arange(15, dtype=float).reshape(5, 3))
    s.set_channel("charge", AtomChannel("value", {0: -1.0, 2: 9.0}))
    s.set_channel("tail", AtomChannel("tag", [3, 4]))
    p = StructureCodec().write(s, tmp_path / "x.xyz")
    back = StructureCodec().load(p)
    assert back.get_channel("charge").data == {0: -1.0, 2: 9.0}   # keys back to int
    assert back.get_channel("tail").data == [3, 4]


def test_the_annotations_key_is_optional(tmp_path):
    """A pair carrying no extensible channels loads with an empty set, its
    labels intact."""
    p = StructureCodec().write(
        Structure(elements=["C", "C", "C"],
                  positions=np.arange(9, dtype=float).reshape(3, 3),
                  regions={"bridge": [1], "frozen_atoms": [0]}),
        tmp_path / "plain.xyz")
    back = StructureCodec().load(p)
    assert back.annotations == {}                        # key absent -> empty
    # The reserved label sits in the one store with the others.
    assert back.regions == {"bridge": [1], "frozen_atoms": [0]}
    assert back.frozen_atoms == [0]


def test_load_rejects_out_of_range_annotation(tmp_path):
    import pytest
    p = StructureCodec().write(
        Structure(elements=["C", "C"],
                  positions=np.arange(6, dtype=float).reshape(2, 3)),
        tmp_path / "bad.xyz")
    side = molstruct.sidecar_path_for(p)
    side.write_text(json.dumps(dict(
        json.loads(side.read_text()),
        annotations={"x": {"kind": "tag", "data": [5]}})))   # idx 5 >= 2
    # Annotation indices are validated against n_atoms_total AT LOAD
    # (mirrors regions/frozen), so a bad sidecar fails early + clearly.
    with pytest.raises(molstruct.MolstructJsonError, match="out of range"):
        StructureCodec().load(p)
