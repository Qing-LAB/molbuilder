"""The frame-contract gate — structure-periodicity.md § 6.1 (state table)
+ § 6.2 v3 (the unified door's regime model).

Python owns every periodicity-metadata change (the gate); the JS only
calls.  These tests pin each state-table row, each op's v3 semantics, the
calibrate≡emit frame equivalence, and the unified endpoint envelope.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.periodicity_gate import (
    OPS, apply_edit, validate_periodicity)


def _mol(off=(10.0, 10.0, 10.0), vacuum=(2.5, 2.5, 2.5)):
    """Two H atoms 2 Å apart along x, offset from the world origin."""
    o = np.asarray(off, dtype=float)
    return Structure(
        elements=["H", "H"],
        positions=np.array([o, o + [2.0, 0.0, 0.0]]),
        vacuum=tuple(vacuum),
    )


# ------------------------------------------------------------------ #
#  § 6.1 state table (stored state)                                   #
# ------------------------------------------------------------------ #



# --------------------------------------------------------------------- #
#  Keying on the ID, not the prose                                      #
#                                                                       #
#  The gate's own header says "callers surface notices; they never parse #
#  the message text", and until 2026-08-03 four tests in this file did   #
#  exactly that -- matching on "does NOT contain".  A reworded sentence  #
#  broke them; a DELETED check would not have, which is the wrong way    #
#  round for a regression test.  Notices now carry ``where``, the same   #
#  stable id ``Issue`` uses, so a test can name the finding it means.    #
# --------------------------------------------------------------------- #


def _problems(notices):
    """Findings that say something is WRONG -- warn or error.

    "Legal" used to be spelled ``not notices``, which also forbade INFO.  That
    broke the moment the gate started disclosing true-but-harmless facts, like
    a vacuum you typed being inert under a cell you typed
    (``cell.vacuum_ignored``).  A legal state may still have something worth
    saying about it; what it may not have is a complaint.
    """
    return [n for n in (notices or []) if n.get("level") in ("warn", "error")]


def _wheres(notices):
    """The stable ids in a notice list."""
    return [n.get("where") for n in (notices or [])]


def _said(notices, where):
    """Did the gate report this specific finding?"""
    return where in _wheres(notices)


class TestTheStateTable:
    """PINS: docs/model/structure-periodicity.md § 6.1 — the state table, one
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
        checked, notes = validate_periodicity(s)
        assert checked.cell is None and not notes    # row 1

    def test_explicit_no_origin_containing_is_legal(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        checked, notes = validate_periodicity(s)
        assert checked.cell_origin is None and not _problems(notes)   # row 2

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
        assert out.cell_contains_atoms(out.resolve_cell_origin())
        assert notes and notes[0]["level"] == "info"
        # A notice says how loud it is, what it says, and WHAT IT IS ABOUT --
        # the subject is what decides where it is shown. What it must never
        # carry is a key saying "and I changed something", because nothing here
        # changes anything: that is the whole of clause 1, and a phantom dirty
        # state is what a correction marker would produce.
        assert all(set(n) == {"level", "message", "where", "about"}
                   for n in notes), notes
        assert all(n["about"] == "cell" for n in notes), notes

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

    def test_user_owned_origin_is_never_rewritten(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        checked, notes = validate_periodicity(s)
        assert np.allclose(checked.cell_origin, [0.5, 0.5, 0.5])  # row 4
        assert not _problems(notes)

    def test_a_manual_origin_gets_the_same_answer_from_either_direction(self):
        """Row 5 is ONE row.  A nonsense corner the user typed on the Cell page
        and the same corner read back off disk are the same state, so they get
        the same answer: kept verbatim, warned about, never auto-fixed.

        This used to be two tests either side of a ``live_edit`` flag.  Both
        passed, which was the proof the flag selected nothing -- it was removed
        2026-08-02.  What the pair was really pinning is the equality below, so
        that is what this asserts.
        """
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.cell_origin = np.array([100.0, 100.0, 100.0])  # nonsense corner
        s.__post_init__()
        checked, notes = validate_periodicity(s)
        assert np.allclose(checked.cell_origin, [100.0, 100.0, 100.0])
        assert _problems(notes), notes

        # The live half: the same corner set through the Cell-page door.
        live = _mol()
        live.cell = np.eye(3) * 7.0
        live.__post_init__()
        typed, _ = apply_edit(live, "cell_origin", [100.0, 100.0, 100.0])
        assert np.allclose(typed.cell_origin, [100.0, 100.0, 100.0])
        assert validate_periodicity(typed)[1] == notes

    def test_too_small_cell_is_a_hard_error(self):
        s = _mol()
        s.cell = np.eye(3) * 1.0                     # extent 2 Å can't fit
        s.__post_init__()
        # Matched on the sentence ("cannot contain") until 2026-08-03.  What
        # this test means is that the state is REFUSED and the user is told
        # which axis -- not that a particular phrasing survives.
        with pytest.raises(ValueError) as exc:
            validate_periodicity(s)
        assert "a" in str(exc.value), (
            f"the refusal does not name the offending axis: {exc.value}")

    def test_left_handed_cell_is_refused(self):
        s = _mol()
        s.cell = np.diag([7.0, 7.0, -7.0])
        s.__post_init__()
        with pytest.raises(ValueError, match="left-handed"):
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
        assert np.allclose(out.resolve_cell_origin(), out.expected_cell_corner())
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
        """After setting an origin, the user is told their vacuum is inert.

        RECEIPTS vs CONDITIONS (molview.md § 6.8).  ``apply_edit`` returns what
        the edit DID; what is now TRUE of the result comes from the gate, which
        the door runs on the result -- so this test asks the same two questions
        in the same order the door does.  It matched a receipt sentence ("NOT
        respected") until 2026-08-03, when that fact became a condition
        (``cell.vacuum_ignored``) reported whenever it holds instead of only
        when nothing else was wrong (§ 3c).
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.vacuum = (2.5, 2.5, 2.5)          # set, and about to be ignored
        s.__post_init__()
        out, receipts = apply_edit(s, "cell_origin", [0.2, 0.2, 0.2])
        assert np.allclose(out.cell_origin, [0.2, 0.2, 0.2])
        assert _said(receipts, "cell.edit"), _wheres(receipts)
        _checked, conditions = validate_periodicity(out)
        assert _said(conditions, "cell.vacuum_ignored"), _wheres(conditions)

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
        # THE SHARPER OF THE TWO containment findings, and the right one here:
        # atom 0 went from 0 to 50 while the others stayed, so the structure is
        # now 49 Å across in a 4 Å box.  No corner can make that fit, and
        # `cell.unfittable` says exactly that -- where `cell.atoms_outside`
        # would suggest moving the origin, which cannot help.  Before the two
        # were split, this case reported the vaguer one.
        assert _said(body.get("notices"), "cell.unfittable"), (
            f"the stranded box was not reported: {_wheres(body.get('notices'))}")

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
        assert not _said(after.get("notices"), "cell.atoms_outside"), (
            f"a derived box cannot fail to contain: {_wheres(after.get('notices'))}")

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
        assert not _said(body.get("notices"), "cell.atoms_outside"), (
            f"spurious finding: {_wheres(body.get('notices'))}")

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
        assert _said(broken.get_json()["notices"], "cell.atoms_outside"), (
            "a box that does not contain the structure must be reported: "
            f"{_wheres(broken.get_json()['notices'])}")

        # Now FIX it — an origin that does contain the atoms.
        fixed = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "cell_origin",
            "payload": [-1.0, -2.0, -2.0]})
        assert fixed.status_code == 200, fixed.get_json()
        notices = fixed.get_json()["notices"]
        assert not _said(notices, "cell.atoms_outside"), (
            "the box now contains the structure, so the answer must not carry "
            f"the pre-edit warning that it does not: {_wheres(notices)}")

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
        containment = [n for n in notices
                       if n.get("where") == "cell.atoms_outside"]
        assert len(containment) <= 1, (
            f"the same condition was reported {len(containment)} times: "
            f"{_wheres(notices)}")

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

    Both seams must answer the same way.  While the in-memory seam derived the
    corner and the read seam did not, /api/build/load served a pair whose box
    MolView drew from the world origin while the Cell page showed the wrapping
    corner: one state, two answers (observed live on projects/hemeC-dithiol,
    2026-07-29)."""

    def _write_pair_with_no_stored_corner(self, dirpath):
        """A pair whose sidecar holds an explicit cell and NO origin, with the
        atoms outside it at the world origin — the hemeC state, and a legal one
        (row 3).  Written BY HAND rather than through ``write()`` so it reaches
        the read seam exactly as a file on disk would."""
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

    def test_the_read_seam_invents_no_corner(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        xyz = self._write_pair_with_no_stored_corner(tmp_path)
        out = StructureCodec().read(xyz)
        # The pair round-trips VERBATIM (no origin invented in the sidecar);
        # the wrapping corner comes back as the resolved view.
        assert out.cell_origin is None
        assert np.allclose(out.resolve_cell_origin(), [7.5, 7.5, 7.5])
        assert out.cell_contains_atoms(out.resolve_cell_origin())

    def test_the_load_door_serves_the_derived_corner(
            self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from molbuilder.diagnostics import Capabilities, set_capabilities
        monkeypatch.chdir(tmp_path)
        sdir = tmp_path / "projects" / "P" / "structure"
        sdir.mkdir(parents=True)
        xyz = self._write_pair_with_no_stored_corner(sdir)
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
        checked, notes = validate_periodicity(s)
        assert checked.cell_origin is None and not _problems(notes)

    def test_junction_periodic_periodic_transport_is_legal(self):
        """The BDT/Au junction shape: periodic x/y, transport z; atoms
        wrapped to negative fractionals along x/y must not trip the
        gate; the transport axis is still judged, via the derived corner."""
        s = _mol(off=(-3.0, -2.0, 10.0), vacuum=(0.0, 0.0, 0.0))
        s.cell = np.diag([10.0, 10.0, 2.0])
        s.axis_kind = ("periodic", "periodic", "transport")
        s.__post_init__()
        out, notes = validate_periodicity(s)
        # x/y ignored (periodic); z extent 0 fits; the transport axis takes
        # its corner from the DERIVED view (bbox_min), not a stored value.
        assert out.cell_origin is None
        assert out.cell_contains_atoms(out.resolve_cell_origin())

    def test_stored_manual_origin_is_warned_never_rewritten(self):
        """Row 5 stored half: a manual origin round-trips verbatim (the
        silent flip on reload is the review finding)."""
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.cell_origin = np.array([100.0, 100.0, 100.0])
        s.__post_init__()
        checked, notes = validate_periodicity(s)
        assert np.allclose(checked.cell_origin, [100.0, 100.0, 100.0])
        assert _problems(notes), notes

    def test_unfittable_cell_edit_is_refused_not_stored(self):
        """A cell the structure cannot fit is refused at the edit — a
        stored-but-invalid cell locked every later door.

        The assertion is REFUSED-AND-NOT-STORED, which is what the title says
        and what matters.  It matched the sentence ("cannot contain") until
        2026-08-03; the wording now comes from ``cell.check`` and names the
        axis, and pinning prose would have failed on an improvement while
        passing on a deletion."""
        s = _mol()                                    # extent 2 Å
        before = s.cell
        with pytest.raises(ValueError):
            apply_edit(s, "cell", (np.eye(3) * 1.0).tolist())
        assert s.cell is before, "the refused cell must not have been stored"

    def test_reset_to_derived_survives_a_zero_extent_isolated_axis(self):
        """Was a refusal ("axis would be degenerate"): a structure with a
        zero-extent ISOLATED axis could not go back to the derived box.

        CLEARING the vacuum is the way back -- the § 6.1 default then gives that
        axis 3 Å per side.  Asking for an explicit zero there is still refused,
        and must be: the value is honoured, so the box really would have no
        volume (see TestTheDefaultVacuumGap).  A transport axis, where vacuum
        has no meaning, refuses either way."""
        s = _mol(vacuum=(2.5, 2.5, 0.0))              # extent 0 on y,z
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(s, "vacuum", None)
        assert out.cell is None                        # reset went through
        assert float(np.linalg.det(out.resolve_cell())) > 0.0
        assert out.effective_vacuum()[2] == 3.0
        assert any("cleared" in n["message"] for n in notes)


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

    def test_tabs_send_the_model_cell_with_the_atoms(self):
        """Source pin: the cell a tab sends comes from the same read as the
        atoms.

        The tab-emit contract asks for the MODEL's cell, never a second source
        -- and the shape that guarantees it is one read, not a `periodicity`
        key.  `exportFile()` returns the atoms, the labels and the cell
        together, in the server's words, for the frame on screen (molview.md
        § 9.3a), so a tab cannot send a cell that is younger or older than the
        coordinates beside it.

        THIS PINNED THE OPPOSITE UNTIL 2026-08-03: it required a `periodicity:`
        key in each render body, which meant a SECOND read of a fact the
        structure already carried -- and the server preferred that later copy,
        so the structure's own cell was never the one judged.

        Transport is the one tab still exempt: its geometry comes from the file
        on disk, not from the viewer, so its cell has nothing to travel with.
        That deviation is recorded in the route itself and is not this test's
        subject.
        """
        import re as _re

        def _code_only(src: str) -> str:
            """The source with comments removed.

            A pin that greps the whole file also reads the PROSE, and the prose
            here explains the very bug being pinned -- so the comment "the cell
            went in as `metadata: {periodicity: …}`" tripped the check that no
            `periodicity` key is sent.  A test that cannot tell a warning from
            the thing it warns about will be silenced by deleting the warning.
            """
            src = _re.sub(r"/\*.*?\*/", "", src, flags=_re.S)
            return _re.sub(r"^\s*//.*$", "", src, flags=_re.M)

        base = Path(__file__).resolve().parents[1] / "molbuilder/web/static"
        for rel in ("structure-optimization/viewer.js", "lib/spectra/core.js"):
            text = _code_only((base / rel).read_text(encoding="utf-8"))
            assert "exportFile()" in text, (
                f"{rel} no longer asks the viewer for the structure it sends; a "
                f"hand-built body is how the cell ended up under a key the "
                f"receiver refuses")
            assert "periodicity:" not in text, (
                f"{rel} sends a `periodicity` key beside the structure -- a "
                f"second read of a fact `exportFile()` already carried")


from pathlib import Path  # noqa: E402  (used by TestTabEmitContract)


class TestReadingDoesNotJudge:
    """§ 8.2 (2026-08-03): a file whose sidecar holds an unusable box OPENS.

    The reader used to raise, and that put a user in a trap with no way out
    inside the app: the Cell page is the one place a box can be corrected, and
    it cannot be reached without the structure on screen.

    What must never happen is a CALCULATION built on an impossible box, and that
    is refused where it belongs -- by the validator, at every emitter. Both
    halves are asserted here together, because either alone is the wrong
    behaviour: opening without the refusal downstream would ship a bad deck;
    refusing at the read is the trap.
    """

    def _pair(self, tmp_path, cell):
        import json as _json
        from molbuilder.sidecars.molstruct import sha256_of_file
        (tmp_path / "wire.xyz").write_text("2\nx\nO 0 0 0\nH 1 0 0\n")
        (tmp_path / "wire.molstruct.json").write_text(_json.dumps({
            "schema_version": 7, "n_atoms_total": 2,
            "structure_hash": sha256_of_file(tmp_path / "wire.xyz"),
            "cell": cell, "regions": {"frozen_atoms": [0]},
        }))
        return tmp_path / "wire.xyz"

    def test_an_unusable_box_still_opens_with_its_labels(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        path = self._pair(tmp_path, [[7, 0, 0], [0, 7, 0], [0, 0, -7]])
        struct = StructureCodec().read(path)          # must not raise
        assert len(struct.elements) == 2
        assert dict(struct.regions) == {"frozen_atoms": [0]}, (
            "the labels were lost on the way in, so 'open it and fix it' costs "
            "the user the work they had already done"
        )
        assert struct.cell is not None, (
            "the bad box was silently dropped; the user cannot correct a value "
            "they were never shown"
        )

    def test_but_no_calculation_is_generated_from_it(self):
        """The other half, and the reason opening it is safe."""
        from molbuilder.config.pyscf import PySCFConfig
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.pyscf import render_script
        from molbuilder.siesta import render_fdf
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0, 0], [1, 0, 0]]),
                      cell=[[7, 0, 0], [0, 7, 0], [0, 0, -7]])
        for name, render, cfg in (("SIESTA", render_fdf, SiestaConfig()),
                                  ("PySCF", render_script, PySCFConfig())):
            with pytest.raises(Exception, match="cell|determinant|hand"):
                render(s, cfg)


class TestTheLoadAnswerIsNotSilent:
    """§ 6.1 clause 6 (approved 2026-07-29): what the gate finds at the load
    door must never be silent — the answer carries it.

    The clause was written when a load could REWRITE stored state, and asked for
    a machine-readable marker so the client could dirty-mark the session.
    Nothing rewrites anything now, so there is nothing to mark and no marker:
    the answer reports, and the pair on disk is still what the user saved."""

    def test_the_load_answer_carries_what_the_gate_found(self, tmp_path, monkeypatch):
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
            # The corner is DERIVED, so the load REPORTS it (info) and changes
            # nothing -- and the shape says so: two keys, no third one claiming
            # a correction, hence no phantom dirty state.
            assert notices and notices[0]["level"] == "info"
            assert all(set(n) == {"level", "message", "where", "about"}
                       for n in notices), notices
            # ...and `about` is the SUBJECT, which is what puts the sentence on
            # the Cell page rather than above the atom list (molview.md § 6.8).
            assert all(n["about"] == "cell" for n in notices), notices
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
    accepts.  The docstring once claimed FIVE ops, including a ``calibrate``
    that had been deliberately removed (found 2026-07-29) -- this is the guard
    so the three cannot rot apart again.
    """

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


class TestTheDefaultVacuumGap:
    """§ 6.1 (2026-08-03): vacuum has THREE states, and the third is what makes
    the rule sayable.

      * A vacuum is SET -> used verbatim, however small.  Never overridden.
      * NOTHING is set   -> every ISOLATED axis gets 3 A per side.

    THE DISTINCTION THAT MATTERS: 3 A is a default GAP, not a minimum box
    length.  3 A of empty space is 3 A whether the molecule is 2 A across or
    200, so a large molecule gets it too -- and a typed 1.0 A is kept, not
    raised.

    WHAT THIS REPLACED.  Until 2026-08-03 the rule was a floor on the BOX:
    ``extent + 2*vacuum < 3 -> vacuum = max(yours, 3)``.  It asked about the box
    rather than about what the user wanted, and got both ends wrong -- it raised
    a typed 1.0 to 3.0, OVERRIDING a stated value, and it left a large molecule
    with NO gap at all because its box already exceeded 3 A.  Both are the same
    confusion: a minimum box length is not a vacuum.
    """

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

    @staticmethod
    def _big():
        """A molecule 20 A across -- the case the old floor left with NO gap,
        because its box already exceeded 3 A."""
        return Structure(elements=["H", "H"],
                         positions=np.array([[0.0, 0.0, 0.0],
                                             [20.0, 20.0, 20.0]]))

    # -- nothing set: the default gap -------------------------------------- #

    def test_nothing_set_means_unset_not_zero(self):
        """The whole rule rests on this: `None` is a state the model can hold,
        distinct from a deliberate zero."""
        s = self._planar()
        assert s.vacuum is None, "an unstated vacuum must not become (0,0,0)"
        assert s.effective_vacuum() == (3.0, 3.0, 3.0)
        assert s.defaulted_vacuum_axes() == [0, 1, 2]

    def test_a_planar_molecule_gets_a_three_dimensional_box(self):
        """Water has exactly zero extent along z; with no vacuum and no default
        the box would have zero thickness there (a zero determinant)."""
        s = self._planar()
        cell = s.resolve_cell()
        assert float(np.linalg.det(cell)) > 0.0, "box is still degenerate"
        assert np.diag(cell)[2] == pytest.approx(6.0)   # 0 extent + 2 x 3
        assert s.vacuum is None, "the STORED vacuum must stay unset"

    def test_a_linear_molecule_gets_a_three_dimensional_box(self):
        """A diatomic is the harder case: TWO axes have zero extent."""
        cell = self._linear().resolve_cell()
        assert float(np.linalg.det(cell)) > 0.0
        assert min(np.diag(cell)) == pytest.approx(6.0)

    def test_a_large_molecule_gets_THE_SAME_gap(self):
        """THE CORRECTION OF 2026-08-03, pinned.

        3 A is the vacuum DISTANCE, not the size of the molecule.  The old floor
        asked "is the box under 3 A?" -- so a 20 A molecule, whose box was
        already 20 A, got a gap of ZERO.  A big molecule needs the empty space
        just as much as a small one; it needs MORE box, not less gap.
        """
        s = self._big()
        assert s.effective_vacuum() == (3.0, 3.0, 3.0), (
            "a large molecule was denied the default gap -- the old floor's "
            "bug, where 'the box is already big enough' was mistaken for "
            "'the molecule already has vacuum'")
        assert np.diag(s.resolve_cell()) == pytest.approx([26.0, 26.0, 26.0])

    # -- a value that IS set: used verbatim --------------------------------- #

    def test_a_typed_vacuum_is_used_however_small(self):
        """The old floor RAISED a typed 1.0 to 3.0.  You dictate what you want:
        a thin gap is warned about (cell.vacuum_thin), never overridden."""
        s = self._planar()
        s.vacuum = (1.0, 1.0, 1.0)
        s.__post_init__()
        assert s.effective_vacuum() == (1.0, 1.0, 1.0)
        assert s.defaulted_vacuum_axes() == [], "nothing was defaulted"
        assert np.diag(s.resolve_cell())[2] == pytest.approx(2.0)

    def test_setting_one_axis_sets_them_all(self):
        """Vacuum is stored as a whole triple, so a zero on one axis is a
        DELIBERATE zero -- it does not fall back to the default there.  Under
        the old floor this axis was silently topped up to 3."""
        s = self._planar()
        s.vacuum = (4.0, 4.0, 0.0)
        s.__post_init__()
        assert s.effective_vacuum() == (4.0, 4.0, 0.0)
        assert s.defaulted_vacuum_axes() == []

    def test_a_periodic_or_transport_axis_never_gets_a_default(self):
        """Vacuum has no meaning there: the lattice / device length sets it."""
        s = self._planar()
        s.cell = np.diag([5.0, 5.0, 5.0])
        s.axis_kind = ("periodic", "transport", "isolated")
        s.__post_init__()
        eff = s.effective_vacuum()
        assert eff[0] == 0.0 and eff[1] == 0.0
        assert eff[2] == 3.0
        assert s.defaulted_vacuum_axes() == [2]

    # -- the box built from it ---------------------------------------------- #

    def test_the_derived_box_stays_centred_on_the_structure(self):
        """The corner must use the same effective vacuum, or the box grows on
        one face only and the molecule sits off-centre."""
        s = self._planar()
        cell, origin = s.resolve_cell(), s.resolve_cell_origin()
        lo, hi = s.positions.min(axis=0), s.positions.max(axis=0)
        assert np.allclose((lo + hi) / 2.0, origin + np.diag(cell) / 2.0)

    # -- it is never silent -------------------------------------------------- #

    def test_the_default_is_announced_on_every_hand_over(self):
        """A number the user did not choose is sizing their box, so it must be
        said -- and said by the check EVERY hand-over runs, not only by the edit
        path.  Before 2026-08-03 you could load a structure and generate from it
        without ever being told (cell-plan.md 3f)."""
        _, notes = validate_periodicity(self._planar())
        # BY ITS ID, not by a phrase in it. `where` is the stable finding id
        # (validation contract); the sentence is wording and was rewritten
        # 2026-08-04 for readability, which is exactly the edit a prose match
        # turns into a false failure.
        said = [n for n in notes if n["where"] == "cell.vacuum_defaulted"]
        assert said, [n["message"][:70] for n in notes]
        assert said[0]["level"] == "info"
        assert said[0]["about"] == "cell"
        msg = said[0]["message"]
        assert "3 Å" in msg          # the gap it chose, in the message
        # It must state the physical consequence in the currency that matters:
        # vacuum is per side, so the gap between images is TWICE it.
        assert "6 Å" in msg, f"the image gap is not named: {msg}"

    def test_a_set_vacuum_is_not_announced(self):
        """Nothing was defaulted, so there is nothing to disclose."""
        s = self._planar()
        s.vacuum = (5.0, 5.0, 5.0)
        s.__post_init__()
        _, notes = validate_periodicity(s)
        assert not [n for n in notes
                    if "no vacuum was set" in n["message"].lower()]

    # -- clearing it back to unset ------------------------------------------ #

    def test_null_clears_the_vacuum(self):
        """molview.md 9.5 has always documented this payload as 'null clears'.
        Until vacuum became Optional there was nothing to clear TO, and the op
        answered 'must be 3 non-negative floats'."""
        s = self._planar()
        s.vacuum = (4.0, 4.0, 4.0)
        s.__post_init__()
        out, notes = apply_edit(s, "vacuum", None)
        assert out.vacuum is None
        assert out.effective_vacuum() == (3.0, 3.0, 3.0)
        assert [n for n in notes if "cleared" in n["message"]]

    def test_a_planar_structure_can_reset_to_derived(self):
        """It used to be refused ("axis 2 would be degenerate"): a planar
        molecule with an explicit cell could not go back to the derived box.
        Clearing the vacuum is the way back -- the default gives z a thickness.
        """
        s = self._planar()
        s.cell = np.diag([9.0, 9.0, 9.0])
        s.axis_kind = ("isolated", "isolated", "isolated")
        s.__post_init__()
        out, _ = apply_edit(s, "vacuum", None)
        assert out.cell is None
        assert float(np.linalg.det(out.resolve_cell())) > 0.0

    def test_typing_a_zero_gap_on_a_flat_axis_is_refused_at_the_edit(self):
        """The Cell page refuses the value you TYPE (8.2): its whole subject is
        that value, so this is immediate feedback, not a block on getting work
        done -- a good value entered straight after is accepted.

        A FILE that already holds this state still opens and is reported
        instead; see TestTheLoadAnswerIsNotSilent.  Same state, two verdicts,
        decided by whether you just typed it.
        """
        s = self._planar()
        with pytest.raises(ValueError, match="degenerate"):
            apply_edit(s, "vacuum", [0.0, 0.0, 0.0])

    def test_a_zero_extent_transport_axis_still_refuses(self):
        """Vacuum cannot rescue a transport axis -- its length is the captured
        device length, so the refusal must stay."""
        s = self._planar()
        s.cell = np.diag([9.0, 9.0, 9.0])
        s.axis_kind = ("isolated", "isolated", "transport")
        s.__post_init__()
        with pytest.raises(ValueError, match="degenerate"):
            apply_edit(s, "vacuum", [2.0, 2.0, 2.0])

    # -- the wire ------------------------------------------------------------ #

    def test_the_wire_carries_unset_and_the_resolved_view(self):
        """Clause 1 on the wire: `vacuum` is the truth the user typed -- or
        `null` -- and `resolved_vacuum` is the view the box was built from.  The
        Cell page needs both so a box built from a number nobody typed is never
        a surprise."""
        per = self._planar().to_wire()["periodicity"]
        assert per["vacuum"] is None, "unset must travel as null, not [0,0,0]"
        assert per["resolved_vacuum"] == [3.0, 3.0, 3.0]

    def test_a_stored_zero_means_a_deliberate_zero(self):
        """`None` and `[0,0,0]` are DIFFERENT and both are honoured.

        A stored all-zero briefly read as UNSET, so that sidecars written
        before vacuum gained its third state kept behaving as they had.  That
        cost the ability to express a deliberate zero at all -- and bought
        compatibility with files that are residue.  Removed 2026-08-03.
        """
        s = self._planar()
        s.apply_metadata_dict({"vacuum": [0.0, 0.0, 0.0]})
        assert s.vacuum == (0.0, 0.0, 0.0)
        assert s.effective_vacuum() == (0.0, 0.0, 0.0), (
            "a vacuum you set is used verbatim, and zero is a value")
        s.apply_metadata_dict({"vacuum": None})
        assert s.vacuum is None
        assert s.effective_vacuum() == (3.0, 3.0, 3.0)

class TestSiestaNeverReceivesAZeroVolumeCell:
    """PINS: docs/model/structure-periodicity.md § 6.1 (the default vacuum gap)
    + the emitter's own last-line check.

    INVARIANT: no code path can hand SIESTA a zero-volume lattice.  SIESTA
    builds reciprocal vectors from the cell, so a zero determinant fails the run
    outright -- we refuse first, at whichever layer sees it, with a message that
    matches the actual cause.

    FOUR independent layers stop one from ever being emitted; this pins each so
    a future change cannot quietly remove the last of them.
    """

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
        """Matched on "right-handed" until 2026-08-03, which was an accident of
        the old check order: ``det > 0`` fails for det == 0, so a FLAT cell was
        reported as a HANDEDNESS problem.  It is a volume problem, and now says
        so.  What this test means is that the edit is refused -- not which
        sentence explains it."""
        with pytest.raises(ValueError) as exc:
            apply_edit(self._flat(), "cell",
                       [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 0.0]])
        assert "handed" not in str(exc.value).lower(), (
            f"a flat cell is not a handedness problem: {exc.value}")

    def test_layer3_the_default_gap_makes_an_unset_isolated_axis_never_zero(self):
        """The path that used to reach the emitter: a flat molecule with no
        vacuum.  The § 6.1 default closes it -- for the UNSET case only.  A
        deliberate zero is still honoured (that is the rule), which is why
        layer 4 below has to stay."""
        cell = self._flat().resolve_cell()
        assert abs(float(np.linalg.det(cell))) > 1e-6
        assert min(np.diag(cell)) == pytest.approx(6.0)

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
        # SPIES ON THE CHECKER, not on the gate (2026-08-03).  The modifying
        # doors no longer call ``validate_periodicity`` -- that one RAISES, and
        # a modify door reports rather than refuses (§ 8.2) -- so they ask
        # ``cell.resolve_and_check`` directly.  Same invariant, one entry point
        # further in: whatever these routes return went past the checker.
        real, seen = _shared.resolve_and_check, []

        def spy(struct):
            seen.append(struct.n_atoms)
            return real(struct)

        monkeypatch.setattr(_shared, "resolve_and_check", spy)
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
        assert _said(said.get("notices"), "cell.atoms_outside"), (
            f"the verdict was dropped between validator and wire: "
            f"{_wheres(said.get('notices'))}")


class TestARefusedCellIsA400:
    """A cell the gate REFUSES is the user's to fix, so the door has to say so.

    ``validate_periodicity`` raises ``ValueError`` for a state that cannot be
    represented at all -- here a left-handed cell (det < 0).  SEVEN doors run the
    gate on the way IN; the six below called it outside any try (the seventh,
    the Cell-page door, has always handled it).  The refusal became an unhandled
    exception, Flask answered 500 with an HTML page, and the browser's
    ``r.json()`` reported it as a network failure -- so the one sentence that
    said what was wrong ("swap two lattice vectors") never reached anybody.

    The fix is not six try/excepts: ``_shared.checked_periodicity`` raises
    ``PeriodicityRefused`` and ONE handler in ``web/app.py`` answers it, the way
    the 413 handler beside it already works.  Each test asserts the gate's OWN
    words come back, because a 400 from some earlier check would otherwise pass
    this test while the bug survived.
    """

    #: det = -64.  Refused by ``_require_right_handed`` before anything else in
    #: the gate runs, so no test here depends on containment or atom count.
    LEFT_HANDED = [[-4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    XYZ = "1\nrefused-cell fixture\nH 0.000 0.000 0.000\n"

    @pytest.fixture
    def client(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    @pytest.fixture
    def client_with_root(self, monkeypatch, tmp_path):
        """A client whose picker allowlist contains ``tmp_path`` -- the transport
        door takes a PATH, not an envelope, so it needs a file it may read."""
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        from molbuilder import diagnostics
        caps = diagnostics.Capabilities(
            runtime_config={}, conda_binary=None, conda_envs=frozenset())
        monkeypatch.setattr(type(caps), "file_picker_roots",
                            lambda self: ((tmp_path.resolve(), "projects"),))
        diagnostics.set_capabilities(caps)
        return create_app(config={}).test_client(), tmp_path

    def _assert_refused(self, response, door):
        assert response.status_code == 400, (
            f"{door}: a refusable cell answered {response.status_code}, not 400"
            f" -- the gate's ValueError escaped the door")
        body = response.get_json()
        assert body is not None, f"{door}: answered with something that is not JSON"
        assert body.get("ok") is False, f"{door}: {body}"
        # PROSE, because a refusal raises and only its sentence reaches the
        # wire -- there is no `where` on an error body. Matched on the term the
        # message is built around; if that has to change, this changes with it.
        assert "left-handed" in (body.get("error") or ""), (
            f"{door}: answered 400, but not with the gate's reason: "
            f"{body.get('error')!r}")

    def test_the_fdf_door_refuses(self, client):
        self._assert_refused(client.post("/api/build/fdf", json={
            "xyz": self.XYZ, "params": {},
            "periodicity": {"cell": self.LEFT_HANDED}}), "/api/build/fdf")

    def test_the_pyscf_door_refuses(self, client):
        self._assert_refused(client.post("/api/build/pyscf", json={
            "xyz": self.XYZ, "params": {},
            "periodicity": {"cell": self.LEFT_HANDED}}), "/api/build/pyscf")

    def test_the_preflight_door_refuses(self, client):
        self._assert_refused(client.post("/api/build/preflight", json={
            "xyz": self.XYZ, "engine": "siesta", "params": {},
            "periodicity": {"cell": self.LEFT_HANDED}}), "/api/build/preflight")

    def test_the_spectra_door_refuses(self, client):
        """The spectra door takes the structure AS DATA, like every other one.

        It used to take `structure_text`, and this test posted that -- so after
        the route changed it was refused for the wrong reason ("no 'structure'
        provided") and still passed the status check while never reaching the
        periodicity gate at all.  The subject here is the gate, so the body has
        to be one the route accepts.
        """
        s = Structure(elements=["H"], positions=np.zeros((1, 3)))
        s.cell = self.LEFT_HANDED
        s.__post_init__()
        # IN the envelope.  This stated the bad cell in a top-level
        # `periodicity` block BESIDE the envelope, and passed because the emit
        # doors applied that block over the structure -- a second source for the
        # cell, the same shape the labels had (#41).  The doors check now and
        # apply nothing, so the cell has to be where the structure keeps it.
        self._assert_refused(client.post("/api/spectra/render", json={
            "structure": s.to_dict(), "params": {}}), "/api/spectra/render")

    def test_the_export_door_refuses(self, client):
        """The export door reads the cell off the ENVELOPE rather than a
        `periodicity` block, and called the gate directly -- so it is the one
        door the shared entry helper does not cover, and it needs the same
        wrapper."""
        s = Structure(elements=["H"], positions=np.zeros((1, 3)))
        s.cell = self.LEFT_HANDED
        s.__post_init__()
        self._assert_refused(client.post("/api/structure/export", json={
            "structure": s.to_dict(), "name": "refused"}),
            "/api/structure/export")

    def test_the_transport_door_refuses(self, client_with_root):
        """Transport takes the structure as DATA since 2026-08-03 -- it used to
        take a file path as its geometry, with the labels riding beside it."""
        client, tmp = client_with_root
        xyz = tmp / "refused.xyz"
        xyz.write_text(self.XYZ)
        s = Structure(elements=["H"], positions=np.zeros((1, 3)))
        s.cell = self.LEFT_HANDED
        s.__post_init__()
        self._assert_refused(client.post("/api/transport/render", json={
            "structure": s.to_dict(), "structure_path": str(xyz),
            "params": {}}),
            "/api/transport/render")
