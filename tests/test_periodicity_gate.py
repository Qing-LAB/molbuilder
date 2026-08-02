"""The frame-contract gate — structure-periodicity.md § 6.1 (heal table)
+ § 6.2 v3 (the unified door's regime model).

Python owns every periodicity-metadata change (the gate); the JS only
calls.  These tests pin each heal-table row, each op's v3 semantics, the
calibrate≡emit frame equivalence, and the unified endpoint envelope.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.periodicity_gate import (
    OPS, apply_edit, contains_atoms, expected_corner, validate_periodicity)


def _mol(off=(10.0, 10.0, 10.0), vacuum=(2.5, 2.5, 2.5)):
    """Two H atoms 2 Å apart along x, offset from the world origin."""
    o = np.asarray(off, dtype=float)
    return Structure(
        elements=["H", "H"],
        positions=np.array([o, o + [2.0, 0.0, 0.0]]),
        vacuum=tuple(vacuum),
    )


# ------------------------------------------------------------------ #
#  § 6.1 heal table (stored state)                                    #
# ------------------------------------------------------------------ #


class TestHealTable:
    """PINS: docs/model/structure-periodicity.md § 6.1 — the heal table, one
    row per stored state.

    INVARIANT: for every (cell, cell_origin, containment) combination the gate
    takes exactly the action § 6.1 tabulates — and since the 2026-07-29
    "derive the corner" decision that action never REWRITES the truth: a
    resolved corner is a view (clause 1), so the gate validates and reports.

    PREVENTS: the 2026-07 hemeC corruption class — a resolved cell materialised
    into ``cell`` with the origin dropped, which drew the box from (0,0,0) with
    the molecule outside it.
    """

    def test_derived_state_is_untouched(self):
        s = _mol()
        healed, notes = validate_periodicity(s)
        assert healed.cell is None and not notes    # row 1

    def test_explicit_no_origin_containing_is_legal(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        healed, notes = validate_periodicity(s)
        assert healed.cell_origin is None and not notes   # row 2

    def test_hemec_state_derives_the_corner_without_materialising_it(self):
        """Row 3 — the 2026-07 hemeC corruption: explicit cell, no origin,
        atoms far outside [0, cell).  The corner is DERIVED (a view), the
        truth is left alone, and the box wraps the structure (2026-07-29
        decision: no explicit origin means derive the corner)."""
        s = _mol()                                   # atoms near (10,10,10)
        s.cell = np.eye(3) * 7.0
        s.__post_init__()
        out, notes = validate_periodicity(s)
        assert out.cell_origin is None               # truth untouched
        assert np.allclose(out.resolve_cell_origin(), [7.5, 7.5, 7.5])
        assert contains_atoms(out, out.resolve_cell_origin())
        assert notes and notes[0]["level"] == "info"
        # Nothing was modified, so nothing claims the session is dirty.
        assert not any(n.get("kind") == "heal" for n in notes)

    def test_no_seam_materialises_a_resolved_corner(self):
        """The invariant behind the 2026-07-29 decision: for ONE state
        (explicit cell, no origin, atoms outside), the load/save gate and the
        reset-origin op must agree — both leave cell_origin None and both
        resolve the same wrapping corner.  Disagreement here was the bug."""
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.__post_init__()
        gated, _ = validate_periodicity(s)
        reset, _ = apply_edit(s, "cell_origin", None)
        assert gated.cell_origin is None and reset.cell_origin is None
        assert np.allclose(gated.resolve_cell_origin(),
                           reset.resolve_cell_origin())

    def test_user_owned_origin_is_never_healed(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        healed, notes = validate_periodicity(s)
        assert np.allclose(healed.cell_origin, [0.5, 0.5, 0.5])  # row 4
        assert not notes

    def test_live_origin_edit_is_accepted_with_warning(self):
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.cell_origin = np.array([100.0, 100.0, 100.0])  # nonsense corner
        s.__post_init__()
        healed, notes = validate_periodicity(s, live_edit=True)
        assert np.allclose(healed.cell_origin, [100.0, 100.0, 100.0])
        assert notes and notes[0]["level"] == "warn"     # row 5, live half

    def test_too_small_cell_is_a_hard_error(self):
        s = _mol()
        s.cell = np.eye(3) * 1.0                     # extent 2 Å can't fit
        s.__post_init__()
        with pytest.raises(ValueError, match="cannot contain"):
            validate_periodicity(s)

    def test_left_handed_cell_is_refused(self):
        s = _mol()
        s.cell = np.diag([7.0, 7.0, -7.0])
        s.__post_init__()
        with pytest.raises(ValueError, match="right-handed"):
            validate_periodicity(s)


# ------------------------------------------------------------------ #
#  § 6.2 v3 op semantics (the regime model)                           #
# ------------------------------------------------------------------ #


class TestApplyEditV3:
    """PINS: docs/model/structure-periodicity.md § 6.2 — the v3 regime model,
    one row per Cell-page op.

    INVARIANT: an edit to an UPSTREAM parameter never silently contradicts
    downstream state — it resets it, loudly.  Editing vacuum or axis kinds
    returns the box to the DERIVED regime; an explicit cell demotes vacuum to
    reference-only; an explicit origin overrides the derived corner
    ("origin first, then vacuum").  No op ever moves an atom: coordinate
    rewrites are a Modify op, not a periodicity edit.
    """

    def test_calibrate_is_not_a_periodicity_op(self):
        assert "calibrate" not in OPS
        with pytest.raises(ValueError, match="unknown periodicity op"):
            apply_edit(_mol(), "calibrate", None)

    def test_vacuum_edit_resets_manual_state_to_derived(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        out, notes = apply_edit(s, "vacuum", [3.0, 3.0, 3.0])
        assert out.cell is None and out.cell_origin is None
        assert out.vacuum == (3.0, 3.0, 3.0)
        assert any("DERIVED regime" in n["message"] for n in notes)

    def test_vacuum_edit_refused_on_a_periodic_axis(self):
        s = _mol()
        s.cell = np.diag([10.0, 10.0, 4.0])
        s.axis_kind = ("isolated", "isolated", "periodic")
        s.__post_init__()
        with pytest.raises(ValueError, match="periodic"):
            apply_edit(s, "vacuum", [3.0, 3.0, 3.0])

    def test_axis_edit_resets_to_derived_when_nonperiodic(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(
            s, "axis_kind", ["isolated", "isolated", "isolated"])
        assert out.cell is None and out.cell_origin is None
        assert any("DERIVED regime" in n["message"] for n in notes)
        # A transport axis with zero structure extent would be a
        # degenerate derived box -> refused (its own pin below).

    def test_axis_to_periodic_keeps_the_explicit_cell(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(
            s, "axis_kind", ["isolated", "isolated", "periodic"])
        assert out.cell is not None                      # respected, kept
        assert any("respected" in n["message"] for n in notes)

    def test_axis_to_periodic_without_a_cell_is_refused(self):
        with pytest.raises(ValueError, match="explicit commensurate cell"):
            apply_edit(_mol(), "axis_kind",
                       ["isolated", "isolated", "periodic"])

    def test_cell_edit_respects_existing_origin_first(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        out, _ = apply_edit(s, "cell", (np.eye(3) * 12.0).tolist())
        assert np.allclose(out.cell_origin, [0.5, 0.5, 0.5])

    def test_cell_edit_without_origin_respects_vacuum(self):
        s = _mol()                                   # atoms near (10,10,10)
        out, notes = apply_edit(s, "cell", (np.eye(3) * 8.0).tolist())
        # The corner honours the vacuum, but as a VIEW: the truth keeps no
        # explicit origin, so a later reset has nothing to undo and the two
        # seams cannot drift (2026-07-29).
        assert out.cell_origin is None
        assert np.allclose(out.resolve_cell_origin(), expected_corner(out))
        assert np.allclose(out.resolve_cell_origin(), [7.5, 7.5, 7.5])
        assert any("origin first, then vacuum" in n["message"]
                   for n in notes)

    def test_cell_null_returns_to_derived(self):
        s = _mol()
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, _ = apply_edit(s, "cell", None)
        assert out.cell is None and out.cell_origin is None

    def test_origin_edit_warns_vacuum_not_respected(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(s, "cell_origin", [0.2, 0.2, 0.2])
        assert np.allclose(out.cell_origin, [0.2, 0.2, 0.2])
        assert any("NOT respected" in n["message"] for n in notes)

    def test_reset_origin_clears_to_none(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        out, notes = apply_edit(s, "cell_origin", None)
        assert out.cell_origin is None
        assert any("freedom back" in n["message"] for n in notes)

    def test_no_op_ever_moves_atoms(self):
        s = _mol()
        out, _ = apply_edit(s, "vacuum", [1.0, 1.0, 1.0])
        assert np.allclose(out.positions, s.positions)
        withcell, _ = apply_edit(s, "cell", (np.eye(3) * 9.0).tolist())
        assert np.allclose(withcell.positions, s.positions)
        for op, payload in [("cell_origin", [7.0, 7.0, 7.0]),
                            ("cell_origin", None),
                            ("cell", None)]:
            out, _ = apply_edit(withcell, op, payload)
            assert np.allclose(out.positions, s.positions), op


# ------------------------------------------------------------------ #
#  Calibrate ≡ emit — the implicit-translation equivalence            #
# ------------------------------------------------------------------ #


def _coords_block(fdf: str) -> str:
    lines = fdf.splitlines()
    a = next(i for i, l in enumerate(lines)
             if "%block AtomicCoordinatesAndAtomicSpecies" in l)
    b = next(i for i, l in enumerate(lines[a:], start=a)
             if "%endblock" in l)
    return "\n".join(lines[a:b + 1])


def test_calibrated_then_emit_equals_emit():
    """§ 6.2: emission translates implicitly — a user who never clicks
    calibrate gets the identical SIESTA frame."""
    from molbuilder.modify import calibrate_to_cell
    from molbuilder.siesta.input import render_fdf
    from molbuilder.config.siesta import SiestaConfig
    s = _mol()
    cfg = SiestaConfig(verbose_comments=False)
    direct = _coords_block(render_fdf(s, cfg))
    baked = _coords_block(render_fdf(calibrate_to_cell(s), cfg))
    assert direct == baked


# ------------------------------------------------------------------ #
#  The unified endpoint                                               #
# ------------------------------------------------------------------ #


class TestPeriodicityDoor:
    """PINS: docs/model/structure-periodicity.md § 6.2 — the unified door
    ``POST /api/structure/periodicity``.

    INVARIANT: Python owns every metadata change and the client only calls.  The
    door takes THE ENVELOPE every other structure door takes (web-api.md § 1) and
    returns the cell block the gate accepted -- raw values and the § 3 resolved
    views together, in the shape ``/api/build/load`` sends -- which the client
    adopts verbatim; an unknown op is a 400, never a silent no-op.

    It took a ``{"data": {xyz, sidecar}}`` blob until 2026-07-31, which its one
    caller could not produce: MolView writes no coordinate document (molview.md
    § 11.7), so the one door the cell changes through answered 400 to every
    request ever made of it.
    """

    @pytest.fixture
    def client(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    def _envelope(self, struct):
        """What MolView hands over: the atoms as NUMBERS and the facts beside
        them -- ``structureForServer``'s output shape."""
        return struct.to_dict()

    def test_vacuum_op_round_trips_the_truth(self, client):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0                    # manual state...
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "vacuum",
            "payload": [3.0, 3.0, 3.0]})
        assert r.status_code == 200, r.get_json()
        j = r.get_json()
        assert j["ok"] is True
        per = j["periodicity"]
        assert per["cell"] is None                   # ...reset to derived
        assert per["vacuum"] == [3.0, 3.0, 3.0]
        # The RESOLVED views ride in the same block, so the client cannot be
        # handed a cell block whose "as it will actually be used" half is
        # missing (molview.md § 9.3).
        assert per["resolved_cell"] is not None
        assert "resolved_cell_origin" in per and "resolved_vacuum" in per
        assert any(n["level"] == "warn" for n in j["notices"])

    def _explicit_box_structure(self):
        """Three atoms in a row inside a box the USER typed: 4 A cube at the
        world origin, all three axes isolated.  Manual, so no resolver
        recomputes it -- what happens to it is only ever reported.
        """
        s = Structure(elements=["H", "H", "H"],
                      positions=np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]))
        s.cell = np.eye(3) * 4.0
        s.cell_origin = np.zeros(3)
        s.axis_kind = ("isolated",) * 3
        s.__post_init__()
        return s

    def test_moving_one_atom_out_of_an_explicit_box_is_reported(self, client):
        """A PARTIAL translate moves atoms and leaves the box where it is
        (``modify.py:422`` -- "``indices`` -> move ONLY those atoms, box
        untouched"), so it is the op that can strand a manual box.  The user
        typed this box, so nothing rewrites it; the validation at the single
        exit says what is now true of it.
        """
        s = self._explicit_box_structure()
        r = client.post("/api/modify/translate", json={
            "structure": self._envelope(s),
            "dx": 50.0, "dy": 0.0, "dz": 0.0, "indices": [0]})
        assert r.status_code == 200
        body = r.get_json()
        assert body["periodicity"]["cell"][0][0] == 4.0, "the typed box is kept"
        said = " ".join(n["message"] for n in (body.get("notices") or []))
        assert "does NOT contain" in said, f"no containment notice: {body.get('notices')}"

    def test_a_derived_box_regrows_around_an_atom_that_moved(self, client):
        """The box the USER did not type.  Nothing stores a cell for it: the
        resolver computes one from the atoms + vacuum every time the structure
        is serialised (``to_wire`` -> ``resolve_cell``), so moving an atom
        moves the box that reports back.  No healing, no write-back -- the
        derived cell was never a stored value to correct.
        """
        s = Structure(elements=["H", "H", "H"],
                      positions=np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]))
        s.axis_kind = ("isolated",) * 3
        s.vacuum = (5.0, 5.0, 5.0)
        s.__post_init__()
        assert s.cell is None, "this structure has no cell of its own"

        before = client.post("/api/modify/translate", json={
            "structure": self._envelope(s),
            "dx": 0.0, "dy": 0.0, "dz": 0.0}).get_json()
        after = client.post("/api/modify/translate", json={
            "structure": self._envelope(s),
            "dx": 20.0, "dy": 0.0, "dz": 0.0, "indices": [2]}).get_json()

        span_before = before["periodicity"]["resolved_cell"][0][0]
        span_after = after["periodicity"]["resolved_cell"][0][0]
        assert span_after > span_before + 15.0, (
            f"the derived box did not follow the atom: {span_before} -> {span_after}")
        assert after["periodicity"]["cell"] is None, "still nothing stored"
        said = " ".join(n["message"] for n in (after.get("notices") or []))
        assert "does NOT contain" not in said, f"a derived box cannot fail to contain: {said}"

    def test_translating_the_whole_molecule_keeps_the_box_with_it(self, client):
        """The same distance, every atom: ``affine`` carries ``cell_origin``
        with the atoms (``structure.py:1619``), so the structure sits in the
        box exactly as before and there is nothing to report.  This is the
        pair to the test above -- the warning must depend on the atoms moving
        RELATIVE to the box, not on their coordinates being large.
        """
        s = self._explicit_box_structure()
        r = client.post("/api/modify/translate", json={
            "structure": self._envelope(s), "dx": 50.0, "dy": 0.0, "dz": 0.0})
        assert r.status_code == 200
        body = r.get_json()
        assert body["periodicity"]["cell_origin"][0] == 50.0, "the box moved too"
        said = " ".join(n["message"] for n in (body.get("notices") or []))
        assert "does NOT contain" not in said, f"spurious notice: {body.get('notices')}"

    def test_a_fixed_box_is_not_still_reported_as_broken(self, client):
        """molview.md § 6.8: a CONDITION describes the state the answer carries.

        The door used to validate what ARRIVED, then apply the edit, then return
        both sets together — so correcting a box that did not contain the
        structure came back with the edit's receipt AND the pre-edit warning that
        it does not contain the structure. The user fixes the problem and is told
        it is still broken, which is how a warning surface gets ignored.
        """
        import numpy as np
        from molbuilder.structure import Structure

        # A box that does NOT contain the structure: origin far from the atoms.
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        s.cell = np.eye(3) * 5.0
        s.cell_origin = np.array([50.0, 50.0, 50.0])
        s.axis_kind = ("isolated",) * 3
        s.__post_init__()

        broken = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "cell_origin",
            "payload": [50.0, 50.0, 50.0]})
        assert broken.status_code == 200, broken.get_json()
        assert any("does NOT contain" in n["message"]
                   for n in broken.get_json()["notices"]), (
            "a box that does not contain the structure must be reported: "
            f"{broken.get_json()['notices']}")

        # Now FIX it — an origin that does contain the atoms.
        fixed = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "cell_origin",
            "payload": [-1.0, -2.0, -2.0]})
        assert fixed.status_code == 200, fixed.get_json()
        notices = fixed.get_json()["notices"]
        assert not any("does NOT contain" in n["message"] for n in notices), (
            "the box now contains the structure, so the answer must not carry "
            f"the pre-edit warning that it does not: {notices}")

    def test_a_condition_is_reported_once_not_twice(self, client):
        """§ 6.8: `apply_edit` emits RECEIPTS, `validate_periodicity` emits
        CONDITIONS. Setting a cell whose result still does not contain the
        structure used to produce the containment fact twice — once from each —
        in two different wordings.
        """
        import numpy as np
        from molbuilder.structure import Structure

        # The INCOMING structure already has a box that does not contain it --
        # so the incoming validation has something to say -- and the edit leaves
        # it not containing, so the edit path has the same thing to say.
        # The box runs 5..15 along x; an atom sits at x = 0.
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([5.0, 0.0, 0.0])
        s.axis_kind = ("isolated",) * 3
        s.__post_init__()

        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "cell",
            "payload": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]})
        assert r.status_code == 200, r.get_json()
        notices = r.get_json()["notices"]
        containment = [n for n in notices if "does NOT contain" in n["message"]]
        assert len(containment) <= 1, (
            f"the same condition was reported {len(containment)} times: {notices}")

    def test_the_door_takes_the_envelope_molview_can_actually_produce(self, client):
        """molview.md § 11.7: the browser hands over the structure and writes no
        coordinate document, so the door must accept the atoms as numbers."""
        s = _mol()
        env = self._envelope(s)
        assert "elements" in env and "positions" in env, (
            "the envelope is the atoms as numbers")
        assert "xyz" not in env, "the browser writes no coordinate document"
        r = client.post("/api/structure/periodicity", json={
            "structure": env, "op": "cell",
            "payload": [[9, 0, 0], [0, 9, 0], [0, 0, 9]]})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["periodicity"]["cell"] == [[9, 0, 0], [0, 9, 0],
                                                       [0, 0, 9]]

    def test_a_missing_envelope_is_a_400(self, client):
        r = client.post("/api/structure/periodicity",
                        json={"op": "vacuum", "payload": [1, 1, 1]})
        assert r.status_code == 400
        assert "structure" in r.get_json()["error"]

    def test_unknown_op_is_a_400(self, client):
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(_mol()), "op": "calibrate"})
        assert r.status_code == 400


# ------------------------------------------------------------------ #
#  The LOADER gate (§ 6.1 clause 1-2) — the live hemeC symptom        #
# ------------------------------------------------------------------ #


class TestLoaderGate:
    """PINS: docs/model/structure-periodicity.md § 6.1 clause 2 — ONE gate,
    on both seams of the .xyz/.molstruct.json pair.

    The gate must sit on BOTH pair seams.  Gating the in-memory seam alone left
    /api/build/load serving corrupted pairs unhealed — MolView drew the
    box from the world origin while the Cell page showed healed values
    (observed live on projects/hemeC-dithiol, 2026-07-29)."""

    def _write_corrupted_pair(self, dirpath):
        """The pair written BY HAND, not through ``write()`` — the point is to
        put a corrupted pair on disk, so it must not pass through the door whose
        healing is under test."""
        import json as _json
        from molbuilder.workingcopy_structure import StructureCodec
        s = _mol()                          # atoms near (10,10,10), vac 2.5
        s.cell = np.eye(3) * 7.0            # explicit cell, no origin,
        s.__post_init__()                   # atoms far outside [0, cell)
        made = StructureCodec().pair(s)
        (dirpath / "m.xyz").write_text(made.document, encoding="utf-8")
        (dirpath / "m.molstruct.json").write_text(
            _json.dumps(made.sidecar), encoding="utf-8")
        return dirpath / "m.xyz"

    def test_codec_read_heals_a_corrupted_pair(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        xyz = self._write_corrupted_pair(tmp_path)
        out = StructureCodec().read(xyz)
        # The pair round-trips VERBATIM (no origin invented in the sidecar);
        # the wrapping corner comes back as the resolved view.
        assert out.cell_origin is None
        assert np.allclose(out.resolve_cell_origin(), [7.5, 7.5, 7.5])
        assert contains_atoms(out, out.resolve_cell_origin())

    def test_load_endpoint_serves_the_healed_resolved_origin(
            self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from molbuilder.diagnostics import Capabilities, set_capabilities
        monkeypatch.chdir(tmp_path)
        sdir = tmp_path / "projects" / "P" / "structure"
        sdir.mkdir(parents=True)
        xyz = self._write_corrupted_pair(sdir)
        set_capabilities(Capabilities(runtime_config={},
                                      conda_binary="/usr/bin/conda"))
        try:
            from molbuilder.web.app import create_app
            client = create_app(config={}).test_client()
            r = client.post("/api/build/load", json={"path": str(xyz)})
            assert r.status_code == 200, r.get_json()
            per = r.get_json()["periodicity"]
            assert per["resolved_cell_origin"] is not None
            assert np.allclose(per["resolved_cell_origin"],
                               [7.5, 7.5, 7.5])
        finally:
            set_capabilities(None)


# ------------------------------------------------------------------ #
#  Per-axis-kind containment (review findings, 2026-07-29)            #
# ------------------------------------------------------------------ #


class TestPeriodicAxesAreNeverContained:
    """PINS: docs/model/structure-periodicity.md § 2 (which axis an image
    belongs to decides whether it is a defect) + § 6.1 (containment is required
    along NON-PERIODIC axes only).

    Along a periodic axis, atoms outside [0, cell) are periodic
    images — legal.  Requiring containment there made real crystal and
    junction files unopenable."""

    def test_periodic_crystal_with_outside_atoms_is_legal(self):
        s = _mol(off=(-3.0, 25.0, 1.0), vacuum=(0.0, 0.0, 0.0))
        s.cell = np.eye(3) * 10.0
        s.axis_kind = ("periodic", "periodic", "periodic")
        s.__post_init__()
        healed, notes = validate_periodicity(s)
        assert healed.cell_origin is None and not notes

    def test_junction_periodic_periodic_transport_is_legal(self):
        """The BDT/Au junction shape: periodic x/y, transport z; atoms
        wrapped to negative fractionals along x/y must not trip the
        gate; the transport axis still heals via the corner."""
        s = _mol(off=(-3.0, -2.0, 10.0), vacuum=(0.0, 0.0, 0.0))
        s.cell = np.diag([10.0, 10.0, 2.0])
        s.axis_kind = ("periodic", "periodic", "transport")
        s.__post_init__()
        out, notes = validate_periodicity(s)
        # x/y ignored (periodic); z extent 0 fits; the transport axis takes
        # its corner from the DERIVED view (bbox_min), not a stored value.
        assert out.cell_origin is None
        assert contains_atoms(out, out.resolve_cell_origin())

    def test_stored_manual_origin_is_warned_never_healed(self):
        """Row 5 stored half: a manual origin round-trips verbatim (the
        silent flip on reload is the review finding)."""
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.cell_origin = np.array([100.0, 100.0, 100.0])
        s.__post_init__()
        healed, notes = validate_periodicity(s)          # NOT live_edit
        assert np.allclose(healed.cell_origin, [100.0, 100.0, 100.0])
        assert notes and notes[0]["level"] == "warn"

    def test_unfittable_cell_edit_is_refused_not_stored(self):
        """A cell the structure cannot fit is refused at the edit — a
        stored-but-invalid cell locked every later door."""
        s = _mol()                                    # extent 2 Å
        with pytest.raises(ValueError, match="cannot contain"):
            apply_edit(s, "cell", (np.eye(3) * 1.0).tolist())

    def test_reset_to_derived_survives_a_zero_extent_isolated_axis(self):
        """Was a refusal ("axis would be degenerate") until the § 6.1
        minimum-thickness floor landed: a zero-extent ISOLATED axis is now
        rescued with 3 Å per side instead of blocking the reset.  A transport
        axis, where vacuum has no meaning, still refuses -- see
        TestMinimumThicknessFloor."""
        s = _mol(vacuum=(2.5, 2.5, 0.0))              # extent 0 on y,z
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(s, "vacuum", [3.0, 3.0, 0.0])
        assert out.cell is None                        # reset went through
        assert float(np.linalg.det(out.resolve_cell())) > 0.0
        assert out.effective_vacuum()[2] == 3.0
        assert any("minimum-thickness floor" in n["message"] for n in notes)


# ------------------------------------------------------------------ #
#  THE INVARIANT (user, 2026-07-29): the model's metadata reaches     #
#  the engine — a render body's periodicity governs the emitted frame #
# ------------------------------------------------------------------ #


class TestTabEmitContract:
    """structure-periodicity.md § 7: the tab sends the MODEL's
    periodicity truth in the render body (never a second source), and
    the emitted deck reflects it.  This wire was severed 2026-06-14 by
    the label-presence branch; these pins make it un-severable."""

    @pytest.fixture
    def client(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    _XYZ = ("2\nvacuum-pin\n"
            "H 10.0 10.0 10.0\n"
            "H 12.0 10.0 10.0\n")

    def _lattice_and_coords(self, fdf: str):
        lines = fdf.splitlines()
        a = next(i for i, l in enumerate(lines)
                 if "%block LatticeVectors" in l)
        lat = [[float(x) for x in lines[a + r + 1].split()[:3]]
               for r in range(3)]
        b = next(i for i, l in enumerate(lines)
                 if "%block AtomicCoordinatesAndAtomicSpecies" in l)
        coords = []
        for l in lines[b + 1:]:
            if "%endblock" in l:
                break
            coords.append([float(x) for x in l.split()[:3]])
        return np.array(lat), np.array(coords)

    def test_body_vacuum_governs_the_emitted_fdf(self, client):
        r = client.post("/api/build/fdf", json={
            "xyz": self._XYZ,
            "params": {"system_label": "pin", "verbose_comments": False},
            "frozen_atoms": [], "regions": {},          # labels present!
            "periodicity": {"cell": None, "cell_origin": None,
                            "axis_kind": ["isolated"] * 3,
                            "vacuum": [2.5, 2.5, 2.5]},
        })
        assert r.status_code == 200, r.get_json()
        lat, coords = self._lattice_and_coords(r.get_json()["fdf"])
        # bbox (2,0,0) + 2*2.5 vacuum -> 7 x 5 x 5 box...
        assert np.allclose(np.diag(lat), [7.0, 5.0, 5.0])
        # ...with the molecule centred: min corner at exactly vacuum.
        assert np.allclose(coords.min(axis=0), [2.5, 2.5, 2.5])

    def test_body_explicit_cell_and_origin_shift_the_frame(self, client):
        r = client.post("/api/build/fdf", json={
            "xyz": self._XYZ,
            "params": {"system_label": "pin", "verbose_comments": False},
            "frozen_atoms": [], "regions": {},
            "periodicity": {"cell": (np.eye(3) * 10.0).tolist(),
                            "cell_origin": [9.0, 9.0, 9.0],
                            "axis_kind": ["isolated"] * 3,
                            "vacuum": [0.0, 0.0, 0.0]},
        })
        assert r.status_code == 200, r.get_json()
        lat, coords = self._lattice_and_coords(r.get_json()["fdf"])
        assert np.allclose(np.diag(lat), [10.0, 10.0, 10.0])
        assert np.allclose(coords[0], [1.0, 1.0, 1.0])   # 10 - 9: shifted

    def test_preflight_sees_what_generate_sees(self, client):
        """A planar molecule with vacuum: preflight must NOT judge the
        vacuum-0 phantom (it used to error/advise on a degenerate box)."""
        r = client.post("/api/build/preflight", json={
            "xyz": self._XYZ, "engine": "siesta",
            "params": {"system_label": "pin"},
            "periodicity": {"cell": None, "cell_origin": None,
                            "axis_kind": ["isolated"] * 3,
                            "vacuum": [4.0, 4.0, 4.0]},
        })
        assert r.status_code == 200, r.get_json()
        texts = " ".join(i.get("message", "") for i in
                         (r.get_json().get("issues") or []))
        assert "Thin vacuum" not in texts or "4.0" not in texts

    def test_tabs_send_the_model_periodicity(self):
        """Source pin: every calculation tab's render body includes the
        model's periodicity block (the client half of the contract)."""
        base = Path(__file__).resolve().parents[1] / "molbuilder/web/static"
        for rel, needle in [
            ("structure-optimization/viewer.js", "periodicity: _modelPeriodicity()"),
            ("lib/spectra/core.js", "periodicity:"),
            ("lib/transport/core.js", "_genBody.periodicity"),
        ]:
            text = (base / rel).read_text(encoding="utf-8")
            assert needle in text, f"{rel} lost the tab-emit contract"


from pathlib import Path  # noqa: E402  (used by TestTabEmitContract)


class TestLoadHealIsVisible:
    """§ 6.1 clause 6 (approved 2026-07-29): a heal at the load door must
    never be silent — the response carries the notices, marked with a
    machine-readable kind so the client can dirty-mark the session."""

    def test_load_response_carries_the_heal_notice(self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from molbuilder.diagnostics import Capabilities, set_capabilities
        import json as _json
        from molbuilder.workingcopy_structure import StructureCodec
        monkeypatch.chdir(tmp_path)
        sdir = tmp_path / "projects" / "P" / "structure"
        sdir.mkdir(parents=True)
        s = _mol()
        s.cell = np.eye(3) * 7.0        # the hemeC-class corrupted pair
        s.__post_init__()
        made = StructureCodec().pair(s)
        (sdir / "m.xyz").write_text(made.document, encoding="utf-8")
        (sdir / "m.molstruct.json").write_text(
            _json.dumps(made.sidecar), encoding="utf-8")
        set_capabilities(Capabilities(runtime_config={},
                                      conda_binary="/usr/bin/conda"))
        try:
            from molbuilder.web.app import create_app
            client = create_app(config={}).test_client()
            r = client.post("/api/build/load",
                            json={"path": str(sdir / "m.xyz")})
            assert r.status_code == 200
            j = r.get_json()
            notices = j.get("notices") or []
            # The corner is DERIVED, so the load REPORTS it (info) and
            # changes nothing: no "heal" kind, hence no phantom dirty state.
            assert notices and notices[0]["level"] == "info"
            assert not any(n.get("kind") == "heal" for n in notices)
            # The served model: no invented truth, the corner as a view.
            per = j["periodicity"]
            assert per.get("cell_origin") is None
            assert np.allclose(per["resolved_cell_origin"], [7.5, 7.5, 7.5])
        finally:
            set_capabilities(None)


class TestDoorHygieneAndRemainingOps:
    """PINS: docs/model/structure-periodicity.md § 6.2 — the door's error
    contract and the remaining op paths end to end.

    INVARIANT: every contract violation reaches the caller as a clean 400 with
    a reason (never a 500, never a silent success): a malformed blob, a bad cell
    shape, and an edit the gate refuses.  Each op is exercised through the HTTP
    door, not just the Python function, so the endpoint and the gate cannot
    drift apart.
    """
    """Door error hygiene (approved batch item 2) + the door-op endpoint
    pins the coverage audit called for."""

    @pytest.fixture
    def client(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    def _envelope(self, struct):
        return struct.to_dict()

    def test_malformed_envelope_is_a_clean_400(self, client):
        for bad in (5, "x", [], {"positions": []}, {"elements": ["C"]},
                    {"elements": ["C"], "positions": [[0, 0, 0]],
                     "regions": {}}):     # metadata at the top level
            r = client.post("/api/structure/periodicity",
                            json={"structure": bad, "op": "vacuum",
                                  "payload": [1, 1, 1]})
            assert r.status_code == 400, (bad, r.get_json())
            assert r.get_json()["ok"] is False
            assert r.get_json().get("error"), bad

    def test_bad_cell_shape_is_a_clean_400(self, client):
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(_mol()), "op": "cell",
            "payload": [1, 2, 3]})
        assert r.status_code == 400
        assert "3×3 matrix" in r.get_json()["error"]

    def test_refused_edit_maps_to_a_clean_400(self, client):
        s = _mol()
        s.cell = np.diag([10.0, 10.0, 4.0])
        s.axis_kind = ("isolated", "isolated", "periodic")
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "vacuum",
            "payload": [3, 3, 3]})
        assert r.status_code == 400
        assert "periodic" in r.get_json()["error"]

    def test_cell_op_anchors_origin_through_the_door(self, client):
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(_mol()), "op": "cell",
            "payload": (np.eye(3) * 8.0).tolist()})
        assert r.status_code == 200, r.get_json()
        per = r.get_json()["periodicity"]
        # Truth carries no invented origin; the view carries the corner.
        assert per.get("cell_origin") is None
        assert np.allclose(per["resolved_cell_origin"], [7.5, 7.5, 7.5])

    def test_axis_kind_op_resets_to_derived_through_the_door(self, client):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "axis_kind",
            "payload": ["isolated"] * 3})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["periodicity"]["cell"] is None

    def test_origin_reset_null_payload_through_the_door(self, client):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "cell_origin",
            "payload": None})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["periodicity"]["cell_origin"] is None


class TestDocMatchesTheDoor:
    """PINS: the op set stays identical in all THREE places that state it —
    ``periodicity_gate.OPS``, the § 6.2 table in
    docs/model/structure-periodicity.md, and the door's own docstring.

    INVARIANT: documentation and code cannot disagree about what the door
    accepts.  Also pins that the notice envelope is documented where it is
    produced AND where it is consumed, because ``kind: "heal"`` is load-bearing
    (the load door marks the session dirty on it) rather than decorative.
    """
    """The op set is documented in three places; a guard so they cannot rot
    apart again (the door's docstring claimed FIVE ops including a
    ``calibrate`` that was deliberately removed -- found 2026-07-29)."""

    DOC = "docs/model/structure-periodicity.md"

    def test_every_op_has_a_row_in_the_doc_table(self):
        import pathlib
        text = pathlib.Path(self.DOC).read_text(encoding="utf-8")
        for op in OPS:
            assert f"| `{op}` |" in text, (
                f"op {op!r} has no row in {self.DOC} § 6.2 — the door and the "
                f"doc disagree")

    def test_the_doc_does_not_advertise_a_calibrate_op(self):
        import pathlib
        text = pathlib.Path(self.DOC).read_text(encoding="utf-8")
        assert "| `calibrate` |" not in text
        assert "There is no calibrate button." in text

    def test_the_door_docstring_names_the_real_op_set(self):
        from molbuilder.web.blueprints.build import api_periodicity
        doc = api_periodicity.__doc__ or ""
        for op in OPS:
            assert f"``{op}``" in doc, (
                f"the door's docstring does not name op {op!r}")
        assert "NO ``calibrate`` op" in doc


class TestMinimumThicknessFloor:
    """§ 6.1 (2026-07-29): an isolated axis whose derived length would fall
    below 3 Å gets at least 3 Å of vacuum per side, so a flat or linear
    molecule can never produce a zero-thickness box."""

    @staticmethod
    def _planar():
        """Water: exactly zero extent along z."""
        return Structure(
            elements=["O", "H", "H"],
            positions=np.array([[0.0, 0.0, 0.0],
                                [0.757, 0.586, 0.0],
                                [-0.757, 0.586, 0.0]]))

    @staticmethod
    def _linear():
        """A diatomic: zero extent along TWO axes."""
        return Structure(elements=["H", "H"],
                         positions=np.array([[0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.74]]))

    def test_a_planar_molecule_gets_a_three_dimensional_box(self):
        """Water has exactly zero extent along z; with no vacuum the box used to
        have zero thickness there (a zero determinant)."""
        s = self._planar()
        cell = s.resolve_cell()
        assert float(np.linalg.det(cell)) > 0.0, "box is still degenerate"
        assert np.diag(cell)[2] >= 3.0            # the flat axis
        assert s.vacuum == (0.0, 0.0, 0.0), "the STORED vacuum must not change"
        assert s.effective_vacuum()[2] == 3.0

    def test_a_linear_molecule_gets_a_three_dimensional_box(self):
        """A diatomic is the harder case: TWO axes have zero extent."""
        cell = self._linear().resolve_cell()
        assert float(np.linalg.det(cell)) > 0.0
        assert min(np.diag(cell)) >= 3.0

    def test_the_floored_box_stays_centred_on_the_structure(self):
        """The corner must use the same floored vacuum, or the box grows on one
        face only and the molecule sits off-centre."""
        s = self._planar()
        cell, origin = s.resolve_cell(), s.resolve_cell_origin()
        lo, hi = s.positions.min(axis=0), s.positions.max(axis=0)
        assert np.allclose((lo + hi) / 2.0, origin + np.diag(cell) / 2.0)

    def test_an_axis_with_enough_vacuum_is_untouched(self):
        """The floor is a floor, not an override -- a real vacuum wins."""
        s = self._planar()
        s.vacuum = (8.0, 8.0, 8.0)
        s.__post_init__()
        assert s.effective_vacuum() == (8.0, 8.0, 8.0)
        assert s.vacuum_floor_axes() == []

    def test_only_the_thin_axis_is_floored(self):
        """Per-axis, not per-structure: a thin z must not inflate x and y."""
        s = self._planar()
        s.vacuum = (4.0, 4.0, 0.0)
        s.__post_init__()
        assert s.vacuum_floor_axes() == [2]
        assert s.effective_vacuum() == (4.0, 4.0, 3.0)

    def test_a_periodic_or_transport_axis_is_never_floored(self):
        """Vacuum has no meaning there: the lattice / device length sets it."""
        s = self._planar()
        s.cell = np.diag([5.0, 5.0, 5.0])
        s.axis_kind = ("periodic", "transport", "isolated")
        s.__post_init__()
        eff = s.effective_vacuum()
        assert eff[0] == 0.0 and eff[1] == 0.0
        assert eff[2] == 3.0

    def test_the_floor_is_announced_not_silent(self):
        """The box ends up thicker than the vacuum on screen, so the gate must
        say so -- the stored value is deliberately left alone."""
        s = self._planar()
        out, notes = apply_edit(s, "vacuum", [4.0, 4.0, 0.0])
        floor = [n for n in notes if "minimum-thickness floor" in n["message"]]
        assert floor, [n["message"][:60] for n in notes]
        assert "axis 2" in floor[0]["message"]
        assert floor[0]["level"] == "info"
        assert out.vacuum == (4.0, 4.0, 0.0)

    def test_a_planar_structure_can_now_reset_to_derived(self):
        """It used to be refused ("axis 2 would be degenerate"): a planar
        molecule with an explicit cell could not go back to the derived box."""
        s = self._planar()
        s.cell = np.diag([9.0, 9.0, 9.0])
        s.__post_init__()
        out, _ = apply_edit(s, "vacuum", [0.0, 0.0, 0.0])
        assert out.cell is None
        assert float(np.linalg.det(out.resolve_cell())) > 0.0

    def test_a_zero_extent_transport_axis_still_refuses(self):
        """Vacuum cannot rescue a transport axis -- its length is the captured
        device length, so the refusal must stay."""
        s = self._planar()
        s.cell = np.diag([9.0, 9.0, 9.0])
        s.axis_kind = ("isolated", "isolated", "transport")
        s.__post_init__()
        with pytest.raises(ValueError, match="degenerate"):
            apply_edit(s, "vacuum", [2.0, 2.0, 2.0])

    def test_the_wire_carries_both_the_stored_and_the_effective_vacuum(self):
        """Clause 1 on the wire: `vacuum` is the truth the user typed,
        `resolved_vacuum` is the view the box was built from, and the Cell page
        needs both so a thicker-than-displayed box is never a surprise."""
        per = self._planar().to_wire()["periodicity"]
        assert per["vacuum"] == [0.0, 0.0, 0.0]
        assert per["resolved_vacuum"] == [3.0, 3.0, 3.0]


class TestSiestaNeverReceivesAZeroVolumeCell:
    """PINS: docs/model/structure-periodicity.md § 6.1 (the minimum-thickness
    floor) + the emitter's own last-line check.

    INVARIANT: no code path can hand SIESTA a zero-volume lattice.  SIESTA
    builds reciprocal vectors from the cell, so a zero determinant fails the run
    outright — we refuse first, at whichever layer sees it, with a message that
    matches the actual cause.
    """
    """SIESTA builds reciprocal vectors from the lattice, so a zero-volume cell
    fails outright.  FOUR independent layers stop one from ever being emitted;
    this pins each so a future change cannot quietly remove the last of them."""

    @staticmethod
    def _flat():
        return Structure(elements=["O", "H", "H"],
                         positions=np.array([[0.0, 0.0, 0.0],
                                             [0.757, 0.586, 0.0],
                                             [-0.757, 0.586, 0.0]]))

    def test_layer1_a_singular_explicit_cell_cannot_even_be_constructed(self):
        s = self._flat()
        s.cell = np.diag([8.0, 8.0, 0.0])
        with pytest.raises(ValueError, match="singular|degenerate"):
            s.__post_init__()

    def test_layer2_the_gate_refuses_a_zero_volume_cell_edit(self):
        with pytest.raises(ValueError, match="right-handed"):
            apply_edit(self._flat(), "cell",
                       [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 0.0]])

    def test_layer3_the_floor_makes_an_isolated_axis_never_zero(self):
        """The path that used to reach the emitter: a flat molecule with no
        vacuum.  The § 6.1 floor closes it."""
        cell = self._flat().resolve_cell()
        assert abs(float(np.linalg.det(cell))) > 1e-6
        assert min(np.diag(cell)) >= 3.0

    def test_layer4_the_emitter_refuses_the_one_remaining_case(self):
        """A zero-extent TRANSPORT axis: vacuum does not pad it, so the floor
        deliberately does not apply and the emitter is the last stop."""
        import warnings
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.siesta import render_fdf
        s = self._flat()
        s.axis_kind = ("isolated", "isolated", "transport")
        s.__post_init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="degenerate") as exc:
                render_fdf(s, SiestaConfig())
        msg = str(exc.value)
        # It must name the offending axis AND its kind: "set a vacuum" is wrong
        # advice for a transport axis, and that used to be what it said.
        assert "axis 2" in msg and "transport" in msg
        assert "device length" in msg

    def test_no_emitted_fdf_ever_carries_a_zero_lattice_row(self):
        """Belt across the shapes a user actually builds: flat, linear, and a
        single atom -- each must emit a lattice with three real rows."""
        import warnings
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.siesta import render_fdf
        shapes = {
            "planar":  self._flat(),
            "linear":  Structure(elements=["H", "H"],
                                 positions=np.array([[0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.74]])),
            "single":  Structure(elements=["He"],
                                 positions=np.array([[0.0, 0.0, 0.0]])),
        }
        for name, s in shapes.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fdf = render_fdf(s, SiestaConfig())
            rows = fdf.split("%block LatticeVectors")[1].split(
                "%endblock")[0].strip().splitlines()
            assert len(rows) == 3, name
            for r, row in enumerate(rows):
                length = max(abs(float(x)) for x in row.split())
                assert length > 1e-6, f"{name}: lattice row {r} is zero"

class TestEveryOpIsChecked:
    """The guarantee is not "this op can break the box" -- it is that the
    check RUNS, on every op, every time.

    Each op below is handed a structure that is ALREADY outside its typed
    box.  Whatever the op does to it, the single exit
    (``_shared.ok_structure_response``) validates what comes out, so every
    one of them must say so.  An op that stays silent has found a way past
    the exit.

    The route list is read from the app, not typed here, so an op added
    later fails this test until it is covered.
    """

    #: op -> the minimum body it accepts (the structure is added per-test).
    #: Read from each route's own validation, molbuilder/web/blueprints/modify.py.
    OPS = {
        "/api/modify/delete":      {"indices": [2]},
        "/api/modify/add_atom":    {"element": "H", "anchor_index": 0,
                                    "offset": [0.0, 0.0, 1.0]},
        "/api/modify/orient":      {"anchors": [0, 1]},
        "/api/modify/rotate":      {"axis": "z", "angle": 30.0},
        "/api/modify/translate":   {"dx": 1.0, "dy": 0.0, "dz": 0.0},
        "/api/modify/calibrate":   {},
        "/api/modify/electrode":   {"element": "Au", "plane": "111",
                                    "size": [1, 1, 2]},
        "/api/modify/symmetric_electrodes": {"element": "Au", "plane": "111",
                                             "size": [1, 1, 2], "gap": 8.0},
    }

    @pytest.fixture
    def client(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    #: GET, returns dropdown enums -- no structure goes in or out, so there is
    #: nothing to validate (modify.py:113).
    NOT_AN_OP = {"/api/modify/meta"}

    def _stranded(self):
        """Atoms at x = 50..52 in a 4 A box the user typed at the origin."""
        s = Structure(elements=["H", "H", "H"],
                      positions=np.array([[50.0, 0, 0], [51.0, 0, 0], [52.0, 0, 0]]))
        s.cell = np.eye(3) * 4.0
        s.cell_origin = np.zeros(3)
        s.axis_kind = ("isolated",) * 3
        s.__post_init__()
        return s

    def test_the_op_list_is_complete(self, client):
        live = {str(r.rule) for r in client.application.url_map.iter_rules()
                if str(r.rule).startswith("/api/modify/")}
        missing = live - set(self.OPS) - self.NOT_AN_OP
        assert not missing, (
            "these modify ops are not covered by the always-checked test:\n  "
            + "\n  ".join(sorted(missing))
            + "\n\nAdd it to OPS with the body it needs, or to NOT_AN_OP with"
              "\nthe reason it returns no structure."
        )

    @pytest.mark.parametrize("route", sorted(OPS))
    def test_every_op_runs_the_check(self, client, route, monkeypatch):
        """The check RAN.  Not "it printed something" -- a silent response can
        be perfectly correct (``/api/modify/electrode`` turns two axes
        periodic, and an atom cannot be outside an axis that wraps), so
        absence of a message proves nothing in either direction.  What must
        hold for every op is that the structure it returns went past the
        validator on its way out.
        """
        from molbuilder.web.blueprints import _shared
        real, seen = _shared.validate_periodicity, []

        def spy(struct):
            seen.append(struct.n_atoms)
            return real(struct)

        monkeypatch.setattr(_shared, "validate_periodicity", spy)
        body = dict(self.OPS[route])
        body["structure"] = self._stranded().to_dict()
        r = client.post(route, json=body)
        assert r.status_code == 200, f"{route}: {r.get_json()}"
        assert seen, (
            f"{route} returned a structure that never went past the "
            f"periodicity check.  Every op leaves through "
            f"_shared.ok_structure_response; this one found another way out.")

    def test_the_check_reaches_the_user_when_it_has_something_to_say(self, client):
        """The companion to the test above: running the check is worth nothing
        if its verdict is dropped between the validator and the wire.  A
        partial translate strands a typed box, so this op has something to
        say, and it has to arrive in ``notices``.
        """
        body = {"structure": self._stranded().to_dict(),
                "dx": 1.0, "dy": 0.0, "dz": 0.0}
        said = client.post("/api/modify/translate", json=body).get_json()
        messages = [n["message"] for n in (said.get("notices") or [])]
        assert any("does NOT contain" in m for m in messages), messages
