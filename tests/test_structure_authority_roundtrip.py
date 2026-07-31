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
    "cell_origin":  [0.5, 1.5, 2.5],          # <- NON-zero (and CONTAINING: the
                                           #    read gate heals a pair whose box
                                           #    does not wrap the atoms, so the
                                           #    verbatim round-trip invariant only
                                           #    holds for legal rows 2/4 - 6.1)
    "pbc":          [True, True, False],
    "axis_kind":    ["periodic", "periodic", "isolated"],
    "vacuum":       [0.0, 0.0, 2.0],
    # ONE label store: the reserved label is a member, not a field beside it.
    "regions":      {"electrode": [0], "channel": [1], "frozen_atoms": [0]},
    "annotations":  {"charge": {"kind": "value",
                                "data": {"0": 0.1, "1": -0.1}}},
}

# The canonical field set the codec must preserve intact (annotations compared
# via metadata_to_dict so channel serialisation is included).
# `regions` is the whole label store -- the reserved labels are in it, so there
# is no `frozen_atoms` field to preserve separately (molview.md § 6.6).
_METADATA_FIELDS = ("cell", "cell_origin", "pbc", "axis_kind",
                    "vacuum", "regions")


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
    assert per["cell_origin"] == [0.5, 1.5, 2.5]
    # Explicit cell + cell_origin (junction) -> resolved origin IS the corner.
    assert per["resolved_cell_origin"] == [0.5, 1.5, 2.5]
    assert per["resolved_cell"] == _META["cell"]
    assert per["axis_kind"] == ["periodic", "periodic", "isolated"]
    assert per["vacuum"] == [0.0, 0.0, 2.0]


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


def test_stored_pair_without_an_origin_resolves_the_corner_not_the_world(
        tmp_path):
    """The frame contract's read gate (structure-periodicity.md 6.1 row 3): a
    stored pair whose explicit cell does NOT wrap its atoms round-trips
    VERBATIM, and the wrapping corner comes back as the resolved VIEW -- the
    box never jumps to the world origin, and no computed value is written into
    the truth (2026-07-29 decision)."""
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    s = Structure(elements=["H", "H"],
                  positions=np.array([[10.0, 10.0, 10.0],
                                      [12.0, 10.0, 10.0]]),
                  vacuum=(2.5, 2.5, 2.5))
    s.cell = np.eye(3) * 7.0            # atoms far outside [0, cell)
    s.__post_init__()
    codec = StructureCodec()
    codec.write(s, tmp_path / "bad.xyz")
    back = codec.read(tmp_path / "bad.xyz")
    assert back.cell_origin is None                          # truth untouched
    assert np.allclose(back.resolve_cell_origin(), [7.5, 7.5, 7.5])
    assert back.cell_contains_atoms(back.resolve_cell_origin())


def test_derived_structure_round_trips_with_cell_still_null(tmp_path):
    """§ 6.1 clause 1: derived-ness SURVIVES the pair round-trip — a
    resolved view must never be persisted as truth."""
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    s = Structure(elements=["H", "H"],
                  positions=np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]),
                  vacuum=(3.0, 3.0, 3.0))
    codec = StructureCodec()
    codec.write(s, tmp_path / "d.xyz")
    back = codec.read(tmp_path / "d.xyz")
    assert back.cell is None and back.cell_origin is None
    assert back.vacuum == (3.0, 3.0, 3.0)


def test_to_wire_derived_keeps_cell_null_and_resolves_view():
    """Truth vs view never conflated on the wire for the DERIVED case."""
    import numpy as np
    from molbuilder.structure import Structure
    s = Structure(elements=["H", "H"],
                  positions=np.array([[10.0, 10.0, 10.0],
                                      [12.0, 10.0, 10.0]]),
                  vacuum=(2.5, 2.5, 2.5))
    per = s.to_wire()["periodicity"]
    assert per["cell"] is None
    assert np.allclose(np.diag(np.array(per["resolved_cell"])),
                       [7.0, 5.0, 5.0])
    assert np.allclose(per["resolved_cell_origin"], [7.5, 7.5, 7.5])


def test_save_endpoint_gates_a_corrupted_blob_without_inventing_an_origin(
        tmp_path, monkeypatch):
    """The SAVER half of the gate (§ 6.1 clause 2): a browser blob in the
    hemeC-corrupted state passes through the gate on its way to disk, and the
    written sidecar carries NO invented origin -- the corner is a view, so the
    file that comes back resolves the wrapping corner (2026-07-29)."""
    import json as _json
    import numpy as np
    import pytest as _pytest
    _pytest.importorskip("flask")
    from molbuilder.diagnostics import Capabilities, set_capabilities
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    monkeypatch.chdir(tmp_path)
    sdir = tmp_path / "projects" / "P" / "structure"
    sdir.mkdir(parents=True)
    s = Structure(elements=["H", "H"],
                  positions=np.array([[10.0, 10.0, 10.0],
                                      [12.0, 10.0, 10.0]]),
                  vacuum=(2.5, 2.5, 2.5))
    s.cell = np.eye(3) * 7.0          # corrupted: no origin, atoms outside
    s.__post_init__()
    # The save door takes the STRUCTURE, not a document the caller wrote
    # (molview.md § 11.7) -- `to_dict` is the shape every door speaks.
    set_capabilities(Capabilities(runtime_config={},
                                  conda_binary="/usr/bin/conda"))
    try:
        from molbuilder.web.app import create_app
        client = create_app(config={}).test_client()
        r = client.post("/api/structure/save", json={
            "path": str(sdir / "m.xyz"), "structure": s.to_dict()})
        assert r.status_code == 200, r.get_json()
        side = _json.loads((sdir / "m.molstruct.json").read_text())
        assert side.get("cell_origin") is None
        back = StructureCodec().read(sdir / "m.xyz")
        assert np.allclose(back.resolve_cell_origin(), [7.5, 7.5, 7.5])
    finally:
        set_capabilities(None)
