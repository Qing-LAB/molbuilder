"""Round-trip invariant for the consolidated Structure authority
(docs/model/structure.md).

The one test that would have caught the recurring ``cell_origin -> 0`` bug at
the source: build a Structure with EVERY metadata field set to a NON-default
value (crucially a non-zero ``cell_origin``) and assert it survives each hop of
the ONE codec unchanged:

  * ``Structure.from_dict(s.to_dict())``      -- the pure Python round-trip unit
  * ``Structure.read(s.write(path))``         -- the paired .xyz + .json file unit
  * ``s.to_wire()``                           -- the server->client view carries
                                                 the server-resolved origin

Nobody outside ``Structure`` names a metadata field, so pinning it here pins it
everywhere the codec is used.
"""
import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.workingcopy_structure import StructureCodec


# --------------------------------------------------------------------------- #
#  Fixture -- every metadata field non-default                                #
# --------------------------------------------------------------------------- #

_META = {
    "cell":         [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    "cell_origin":  [1.5, 2.5, 3.5],          # <- NON-zero: the field that dropped
    "pbc":          [True, True, False],
    "axis_kind":    ["periodic", "periodic", "isolated"],
    "vacuum":       [0.0, 0.0, 12.0],
    "regions":      {"electrode": [0], "channel": [1]},
    "frozen_atoms": [0],
    "annotations":  {"charge": {"kind": "value",
                                "data": {"0": 0.1, "1": -0.1}}},
}

# The canonical field set the codec must preserve intact (annotations compared
# via metadata_to_dict so channel serialisation is included).
_METADATA_FIELDS = ("cell", "cell_origin", "pbc", "axis_kind",
                    "vacuum", "regions", "frozen_atoms")


def _fully_populated_structure() -> Structure:
    s = Structure(
        elements=["C", "O"],
        positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        title="junction-cell",
    )
    s.apply_metadata_dict(_META)
    return s


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.allclose(np.asarray(a, float), np.asarray(b, float))
    return a == b


def _assert_metadata_preserved(got: Structure, want: Structure) -> None:
    for f in _METADATA_FIELDS:
        assert _eq(getattr(got, f), getattr(want, f)), (
            f"metadata field {f!r} drifted: "
            f"{getattr(got, f)!r} != {getattr(want, f)!r}"
        )
    # The whole metadata block (incl. annotations) must be byte-identical.
    assert got.metadata_to_dict() == want.metadata_to_dict()
    # Coordinates + identity columns survive too.
    assert np.allclose(got.positions, want.positions)
    assert list(got.elements) == list(want.elements)
    assert got.title == want.title


# --------------------------------------------------------------------------- #
#  § 5.1  Pure Python codec round-trip                                         #
# --------------------------------------------------------------------------- #

def test_to_dict_from_dict_round_trip_preserves_all_metadata():
    s = _fully_populated_structure()
    r = Structure.from_dict(s.to_dict())
    _assert_metadata_preserved(r, s)


def test_to_dict_metadata_nested_under_metadata_key():
    """The canonical dict nests the metadata block under ``metadata`` (the ONE
    home) -- so the field set is named once, not sprayed at the top level."""
    d = _fully_populated_structure().to_dict()
    assert set(_METADATA_FIELDS).issubset(d["metadata"].keys())
    # Top level carries only coords + identity columns, never a metadata field.
    assert not (set(_METADATA_FIELDS) & set(d.keys()))


# --------------------------------------------------------------------------- #
#  § 5.2  Paired .xyz + .molstruct.json file round-trip (L2 StructureCodec --  #
#         the paired-file door; the pure codec is L1, the file door is L2      #
#         because pairing needs the L2 sidecar codec, structure-authority §3.3)#
# --------------------------------------------------------------------------- #

def test_read_write_pair_round_trip_preserves_all_metadata(tmp_path):
    s = _fully_populated_structure()
    xyz = tmp_path / "m.xyz"
    StructureCodec().write(s, xyz)
    assert (tmp_path / "m.molstruct.json").exists(), "sidecar half not written"
    r = StructureCodec().read(xyz)
    _assert_metadata_preserved(r, s)


def test_write_plain_molecule_writes_no_sidecar(tmp_path):
    """No metadata worth persisting => no ``.json`` half (``no .json == empty
    metadata``)."""
    s = Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
    xyz = tmp_path / "h2.xyz"
    StructureCodec().write(s, xyz)
    assert xyz.exists()
    assert not (tmp_path / "h2.molstruct.json").exists()
    # Round-trips to an equivalent (metadata-empty) structure.
    r = StructureCodec().read(xyz)
    assert r.metadata_to_dict() == s.metadata_to_dict()


def test_write_removes_stale_sidecar_when_metadata_cleared(tmp_path):
    """Writing a now-plain structure over a path that had a sidecar removes the
    stale ``.json`` so the pair can't disagree."""
    xyz = tmp_path / "m.xyz"
    codec = StructureCodec()
    codec.write(_fully_populated_structure(), xyz)
    assert (tmp_path / "m.molstruct.json").exists()
    # Re-write with a plain structure (same atom count, no metadata).
    plain = Structure(elements=["C", "O"],
                      positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]))
    codec.write(plain, xyz)
    assert not (tmp_path / "m.molstruct.json").exists()


# --------------------------------------------------------------------------- #
#  § 5.3  Wire view carries the server-resolved origin                         #
# --------------------------------------------------------------------------- #

def test_to_wire_carries_raw_and_resolved_cell_origin():
    s = _fully_populated_structure()
    per = s.to_wire()["periodicity"]
    # Raw stored corner survives (§ 3c).
    assert per["cell_origin"] == [1.5, 2.5, 3.5]
    # Explicit cell + cell_origin (junction) -> resolved origin IS the corner.
    assert per["resolved_cell_origin"] == [1.5, 2.5, 3.5]
    assert per["resolved_cell"] == _META["cell"]
    assert per["axis_kind"] == ["periodic", "periodic", "isolated"]
    assert per["vacuum"] == [0.0, 0.0, 12.0]


def test_to_wire_resolved_origin_none_for_world_origin_crystal():
    """Explicit cell, NO cell_origin (imported crystal, atoms already in
    [0,cell)) -> resolved origin is None (world origin, no shift)."""
    s = Structure(elements=["C"], positions=np.array([[0.5, 0.5, 0.5]]))
    s.apply_metadata_dict({
        "cell": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        "pbc":  [True, True, True],
    })
    per = s.to_wire()["periodicity"]
    assert per["cell_origin"] is None
    assert per["resolved_cell_origin"] is None


def test_from_dict_rejects_none():
    with pytest.raises(ValueError):
        Structure.from_dict(None)
