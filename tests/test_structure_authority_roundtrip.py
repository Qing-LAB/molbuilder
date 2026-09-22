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
    "cell_origin":  [0.5, 1.5, 2.5],          # <- NON-zero (and CONTAINING: a box
                                           #    that does not wrap the atoms is
                                           #    still round-tripped verbatim, but
                                           #    it comes back with a warning, and
                                           #    this fixture is about the pair, not
                                           #    about the warning - 6.1 rows 2/4)
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
# `pbc` was here until 2026-09-22 and it made this loop BLIND.  It is a
# method now, so `getattr(got, "pbc")` returned a bound method; two bound
# methods off two instances are never equal, so the assertion failed on
# every input -- and because it sat third, the loop stopped there and
# `axis_kind`, `vacuum` and `regions` were no longer checked at all.  The
# one test built to catch a field silently dropping could not have seen one.
_METADATA_FIELDS = ("cell", "cell_origin", "axis_kind",
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
    # The tree is THIS tmp one.  A chdir used to say that on its own,
    # because `projects_root` was cwd-anchored; it resolves from the
    # molbuilder root now (2026-08-22), so the door is told directly.
    from molbuilder.projects import PROJECTS_ROOT_ENV
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "projects"))
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


class TestAnEditOutdatesTheContractWithoutErasingIt:
    """`molview.md` § 8.4a: an edit VOIDS a recorded calculation, and the
    way it says so is a flag ON the record — `structure_modified` for a
    geometry or cell op, `labels_modified` for a name.
    `transport.compose.recorded_contract_of` reads the first back as
    *"the mesh cutoff and transverse k-mesh below were converged for a
    cell that is no longer there"*.

    Rebuilding a Structure without `info` deletes the thing that warning
    reads, so the flag has nothing to mark and a form-B citation quietly
    inherits catalogue defaults instead. Voiding is a MARK on the record;
    a record that is gone cannot carry one.
    """

    def _with_contract(self):
        import numpy as np
        from molbuilder.structure import Structure
        s = Structure(elements=["H", "H", "O"],
                      positions=np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]]))
        s.cell = np.diag([8., 8., 8.])
        s.info = {"calculation": {"engine": "siesta",
                                  "contract": {"siesta_mesh_cutoff_ry": 300}}}
        return s

    @pytest.mark.parametrize("name", [
        "copy", "delete_atoms", "add_atom", "translate", "translate_subset",
        "rotate", "orient", "append", "calibrate",
    ])
    def test_every_atom_edit_carries_the_recorded_contract(self, name):
        import numpy as np
        from molbuilder.structure import Structure
        from molbuilder import modify as M
        s = self._with_contract()
        other = Structure(elements=["C"], positions=np.array([[4., 4, 4]]))
        ops = {
            "copy":             lambda x: x.copy(),
            "delete_atoms":     lambda x: M.delete_atoms(x, [2]),
            "add_atom":         lambda x: M.add_atom(x, "C", 0, [2., 2., 2.]),
            "translate":        lambda x: M.translate(x, [1., 0, 0]),
            "translate_subset": lambda x: M.translate(x, [1., 0, 0], indices=[0]),
            "rotate":           lambda x: M.rotate_around_axis(x, "z", 90.0),
            "orient":           lambda x: M.orient_along_axis(x, [0, 1], "z"),
            "append":           lambda x: M.append_structure(x, other),
            "calibrate":        lambda x: M.calibrate_to_cell(x),
        }
        out = ops[name](s)
        if isinstance(out, tuple):
            out = out[0]
        out = getattr(out, "structure", out)
        assert out.info.get("calculation", {}).get("contract"), (
            f"{name} dropped info.calculation -- the flag that marks this "
            f"edit as outdating the contract now has nothing to mark")

    def test_the_contract_carries_its_OUTDATED_flag_through_an_edit(self):
        """The flag is the point: it must survive the very edit that set
        it, or the warning downstream can never fire."""
        from molbuilder.modify import delete_atoms
        s = self._with_contract()
        s.info["calculation"]["structure_modified"] = True
        out = delete_atoms(s, [2])
        out = getattr(out, "structure", out)
        assert out.info["calculation"]["structure_modified"] is True

    @pytest.mark.parametrize("op,mark", [
        ("delete_atoms",  "structure_modified"),
        ("add_atom",      "structure_modified"),
        ("translate",     "structure_modified"),
        ("rotate",        "structure_modified"),
        ("calibrate",     "structure_modified"),
    ])
    def test_a_python_edit_marks_the_contract_outdated(self, op, mark):
        """The backend marks what MolView marks.

        `molview/model.js` sets this flag on every `applyOp`; Python set
        it nowhere, so `molbuilder modify IN OUT --delete N` wrote an
        edited pair still carrying the relaxation's mesh cutoff and
        k-mesh with no staleness mark — and a transport citation of that
        pair sealed to settings converged for a structure that no longer
        exists, silently.  `compose.py` reads this flag to warn.
        """
        from molbuilder import modify as M
        s = self._with_contract()
        out = {
            "delete_atoms": lambda: M.delete_atoms(s, [2]),
            "add_atom":     lambda: M.add_atom(s, "H", 0, [1.0, 1.0, 1.0]),
            "translate":    lambda: M.translate(s, [1.0, 0.0, 0.0]),
            "rotate":       lambda: M.rotate_around_axis(s, "z", 30.0),
            "calibrate":    lambda: M.calibrate_to_cell(s),
        }[op]()
        assert out.info["calculation"][mark] is True
        assert out.info["calculation"]["contract"], "the record was erased, not marked"
        assert "structure_modified" not in s.info["calculation"], \
            "the edit marked its SOURCE too"

    @pytest.mark.parametrize("op, payload", [
        ("vacuum",      [4.0, 4.0, 4.0]),
        ("axis_kind",   ["periodic", "periodic", "isolated"]),
        ("cell",        [[9., 0, 0], [0, 9., 0], [0, 0, 9.]]),
        ("cell_origin", [1.0, 1.0, 1.0]),
        ("block",       {"cell": [[9., 0, 0], [0, 9., 0], [0, 0, 9.]],
                         "cell_origin": None,
                         "axis_kind": ["periodic", "periodic", "periodic"],
                         "vacuum": None}),
    ])
    def test_a_box_edit_marks_the_contract_outdated(self, op, payload):
        """Every op of the periodicity door, not just the geometry ones.

        Mesh cutoff is a grid density over the CELL and the transverse
        k-mesh samples the reciprocal cell, so changing the box is
        exactly what invalidates the inherited settings — as much as
        moving an atom is.  This is the half the browser used to decide
        for itself: `/api/structure/periodicity` returned only the
        `periodicity` block, so `commitPeriodicityOp` marked the store
        locally and Python marked nothing.  Now the door marks and the
        answer carries it, which means the decision has to be pinned
        HERE, in the language that makes it.
        """
        from molbuilder.periodicity_gate import apply_edit
        s = self._with_contract()
        out, _notices = apply_edit(s, op, payload)
        assert out.info["calculation"]["structure_modified"] is True
        assert out.info["calculation"]["contract"], \
            "the record was erased, not marked"
        assert out.info["calculation"].get("labels_modified") is None, \
            "a box edit claimed the labels moved"
        assert "structure_modified" not in s.info["calculation"], \
            "the edit marked its SOURCE too"

    def test_a_refused_box_edit_marks_nothing(self):
        """The mark rides the returned copy, and a refusal returns none.

        `apply_edit` marks immediately after `struct.copy()`, before the
        per-op branches — which is only safe because every refusal
        raises instead of answering. If one ever returned the copy on a
        rejected edit, the pair would carry a staleness flag for an edit
        that never happened.
        """
        from molbuilder.periodicity_gate import apply_edit
        s = self._with_contract()
        with pytest.raises(ValueError):
            apply_edit(s, "axis_kind", ["periodic", "sideways", "isolated"])
        assert "structure_modified" not in s.info["calculation"]

    def test_a_reorder_is_not_an_edit(self):
        """`categorical_sort` derives through the same door but changes
        no geometry, so marking there would warn about every composed
        junction."""
        import numpy as np
        from molbuilder.structure import Structure
        from molbuilder.transport.sort import categorical_sort
        s = Structure(
            elements=["Au", "S", "S", "Au"],
            positions=np.array([[0., 0, 0], [0, 0, 2.], [0, 0, 4.], [0, 0, 6.]]),
            cell=np.diag([8., 8., 12.]),
            regions={"L-electrode": [0], "bridge": [1, 2], "R-electrode": [3]},
            info={"calculation": {"engine": "siesta",
                                  "contract": {"siesta_mesh_cutoff_ry": 300}}})
        out = categorical_sort(s).structure
        assert "structure_modified" not in out.info["calculation"]
        assert out.info["calculation"]["contract"]

    def test_appending_to_a_derived_box_keeps_its_axes_and_its_vacuum(self):
        """A cell nobody STATED still has facts on it.

        `concat` takes the lattice from whichever input carries an
        explicit cell; when none does it took nothing at all, so a canvas
        in the derived-box regime lost its transport axis and the vacuum
        the person typed the moment anything was appended to it — and
        said nothing, because a dropped field raises no notice. § 2.2a
        lists this seam among the ones that carry everything.
        """
        import numpy as np
        from molbuilder.structure import Structure
        from molbuilder.modify import append_structure
        base = Structure(elements=["C"], positions=np.array([[0., 0, 0]]),
                         axis_kind=("isolated", "isolated", "transport"),
                         vacuum=(5.0, 5.0, 0.0))
        other = Structure(elements=["H"], positions=np.array([[3., 0, 0]]))
        out = append_structure(base, other)
        out = out[0] if isinstance(out, tuple) else out
        assert out.axis_kind == ("isolated", "isolated", "transport"), \
            "the transport axis was dropped"
        assert out.vacuum == (5.0, 5.0, 0.0), "the typed vacuum was dropped"

    def test_an_incoming_fragment_does_not_overwrite_the_canvas_facts(self):
        """The other half, and the one that bit: when the ADDITION is the
        one carrying a cell, its `axis_kind` and `vacuum` rode in with it.

        Appending a slab onto a molecule the user had given 8 Å of vacuum
        replaced that vacuum with the slab's deliberate zero and turned two
        isolated axes crystalline — silently, because afterwards nothing was
        outside the box for `cell.check` to notice. A cell is the one thing
        a fragment can supply that the canvas lacks; the rest are facts OF
        the canvas, exactly as `info` is.
        """
        import numpy as np
        from molbuilder.structure import Structure
        from molbuilder.modify import append_structure
        canvas = Structure(elements=["C", "O"],
                           positions=np.array([[0., 0, 0], [1.13, 0, 0]]),
                           vacuum=(8.0, 8.0, 8.0))
        slab = Structure(elements=["Au", "Au"],
                         positions=np.array([[0., 0, 5.], [1.44, 1.44, 5.]]),
                         cell=np.diag([2.88, 2.88, 20.]),
                         axis_kind=("periodic", "periodic", "isolated"),
                         vacuum=(0.0, 0.0, 0.0))
        out = append_structure(canvas, slab)
        out = out[0] if isinstance(out, tuple) else out
        assert out.vacuum == (8.0, 8.0, 8.0), \
            "the fragment's vacuum replaced the one the user typed"
        assert out.axis_kind == ("isolated", "isolated", "isolated"), \
            "the fragment's axis kinds replaced the canvas's"

    def test_append_takes_the_contract_from_the_structure_APPENDED_TO(self):
        """Not from whichever structure happens to carry the cell --
        `concat` picks the lattice that way and `info` rode along, so a
        base with no box lost its contract to the incoming one."""
        import numpy as np
        from molbuilder.structure import Structure
        from molbuilder.modify import append_structure
        base = self._with_contract()
        base.cell = None
        other = Structure(elements=["C"], positions=np.array([[4., 4, 4]]))
        other.cell = np.diag([9., 9., 9.])
        out = append_structure(base, other)
        out = getattr(out[0] if isinstance(out, tuple) else out, "structure",
                      out[0] if isinstance(out, tuple) else out)
        assert out.info.get("calculation", {}).get("contract")


class TestInfoIsANamespaceOfClusters:
    """PINS: `model/structure.md` § 2.2a — `info` is a namespace, one
    top-level key per subsystem, and nothing writes inside someone else's.

    WHY IT NEEDED A DOOR *(user, 2026-09-22)*.  The shape was already
    decided on the browser side — `molview.data.info` has offered
    `set(key, value)` / `remove(key)` since it shipped — and Python had no
    equivalent. So three callers assigned `struct.info` directly, which
    replaces the WHOLE store: any subsystem recording its own metadata
    would take the recorded calculation contract with it unless it thought
    to copy that across too.
    """

    @staticmethod
    def _recorded():
        s = Structure(elements=["H"], positions=np.array([[0.0, 0.0, 0.0]]))
        s.set_info("calculation", {"engine": "siesta",
                                   "contract": {"mesh_cutoff_ry": 400}})
        return s

    def test_one_cluster_does_not_disturb_another(self):
        """The whole point: this is what a bare `struct.info = {...}` got
        wrong."""
        s = self._recorded()
        s.set_info("provenance", {"built_by": "the slab wizard"})
        assert s.info["calculation"]["contract"] == {"mesh_cutoff_ry": 400}
        assert s.info["provenance"] == {"built_by": "the slab wizard"}

    def test_a_cluster_is_dropped_explicitly_and_alone(self):
        """§ 2.2a: a strip is explicit. `drop_info` says it for one cluster,
        `replace(info={})` for the whole store."""
        s = self._recorded()
        s.set_info("provenance", {"built_by": "x"})
        assert s.drop_info("provenance") is True
        assert sorted(s.info) == ["calculation"]
        assert s.drop_info("provenance") is False, "already gone is not an error"

    def test_a_cluster_must_survive_the_sidecar(self):
        """`info` is written to `.molstruct.json`, so a value that cannot be
        JSON is refused HERE — with the caller and the bad value both in
        hand — rather than at the writer, several steps later, on a
        structure that has since been edited."""
        s = self._recorded()
        with pytest.raises(ValueError, match="JSON"):
            s.set_info("bad", {"fn": object()})
        with pytest.raises(ValueError, match="non-empty"):
            s.set_info("", {})
        assert sorted(s.info) == ["calculation"], "a refusal changed nothing"

    def test_the_whole_store_door_checks_the_shape(self):
        """`apply_info_dict` is the in-place sibling of `replace(info=...)`
        and the one the three whole-store writers needed — a sidecar load, a
        caller-stated block, and the Results tab's run record each adopt an
        entire store rather than one cluster."""
        s = self._recorded()
        with pytest.raises(ValueError, match="cluster-name"):
            s.apply_info_dict(["not", "a", "dict"])
        s.apply_info_dict({"calculation": {"engine": "pyscf"}})
        assert s.info == {"calculation": {"engine": "pyscf"}}
        s.apply_info_dict(None)
        assert s.info == {}, "None clears it -- 'this pair records nothing'"

    # NOT TESTED HERE: that the store survives a pair round trip.
    # `test_molstruct_json.py::test_info_rides_the_pair_whole` already
    # asserts exactly that, through the same codec door, on a nested store.
    # A second copy would be a test per call site, which earns no place
    # (`process/testing.md`).
