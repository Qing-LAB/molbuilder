"""The frame-contract gate — structure-periodicity.md § 6.0 (where the box
sits: the rule, or an offset the structure states) + § 6.1 (what is stored)
+ § 6.2 v3 (the unified door's regime model).

Python owns every periodicity-metadata change (the gate); the JS only
calls.  These tests pin the three box states -- derived, typed, typed with an
assigned origin (plan § 5q.1) -- each op's semantics, and the unified
endpoint envelope.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder import cell as cellmod
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
    return [n for n in (notices or []) if n.get("severity") in ("warn", "error")]


def _wheres(notices):
    """The stable ids in a notice list."""
    return [n.get("where") for n in (notices or [])]


def _said(notices, where):
    """Did the gate report this specific finding?"""
    return where in _wheres(notices)


class TestTheStateTable:
    """PINS: docs/model/structure-periodicity.md § 6.0 + § 6.1 — the three box
    states: derived (nothing stored), a typed cell (the rule places the atoms),
    and a typed cell with an origin the person assigned (stored as the offset
    that puts the box there).

    INVARIANT: the gate never REWRITES the truth.  The rule's offset is a view
    (§ 6.1 clause 1), an assigned one is the person's, and the gate validates
    and reports.

    PREVENTS: the 2026-07 hemeC corruption class — a resolved cell materialised
    into ``cell`` with the origin dropped, which drew the box from (0,0,0) with
    the molecule outside it.
    """

    def test_derived_state_is_untouched(self):
        """Row 1: a plain molecule must come back with NOTHING stored and NOTHING said.

        Catches the gate materialising a cell onto a structure that never had one
        (the 2026-07 corruption class, from the other end), and a gate that
        complains about the most ordinary state in the app -- either turns "open a
        molecule" into a warning surface users learn to ignore.

        Contract: `model/structure-periodicity.md` § 6.1, state-table row 1
        (cell None, origin None -> validate, change nothing).
        """
        s = _mol()
        checked, notes = validate_periodicity(s)
        assert checked.cell is None and not notes    # row 1

    def test_a_typed_cell_with_no_assigned_origin_stores_nothing_and_wraps_the_atoms(self):
        """A typed cell and no origin -- the hemeC state, atoms far from the
        world origin: the gate stores nothing, says nothing is wrong, and the
        box the rule places wraps the atoms.

        Catches the gate INVENTING an origin -- writing the rule's offset into
        the truth, which § 6.1 clause 1 forbids and which made a saved pair
        differ from the pair loaded -- and a box drawn at (0,0,0) with the
        molecule outside it, the 2026-07 hemeC symptom.

        Contract: `model/structure-periodicity.md` § 6.0 (absent means the
        rule) + § 6.1 clause 1.  Replaces the row-2 / row-3 pair and the
        reset-origin seam test, whose derived corner and op are retired.
        """
        s = _mol()                                   # atoms near (10,10,10)
        s.cell = np.eye(3) * 7.0
        s.__post_init__()
        checked, notes = validate_periodicity(s)
        assert checked.engine_offset is None, "the rule's offset was stored"
        assert not _problems(notes), notes
        assert cellmod.resolve(checked).contains_atoms, (
            "the box is not drawn round the atoms")
        # A notice says how loud it is, what it says, and WHAT IT IS ABOUT --
        # the subject is what decides where it is shown. What it must never
        # carry is a key saying "and I changed something", because nothing
        # here changes anything: that is the whole of clause 1.
        assert all(set(n) == {"severity", "message", "where", "about"}
                   for n in notes), notes
        assert all(n["about"] == "cell" for n in notes), notes

    # `test_hemec_state_derives_the_corner_without_materialising_it`,
    # `test_no_seam_materialises_a_resolved_corner` and
    # `test_user_owned_origin_is_never_rewritten` RETIRED 2026-09-25 with the
    # derived corner and the `cell_origin` op (§ 6.0; plan § 5q.4).  The first
    # two are the test above; the third read back an attribute it had set on
    # the object itself, so it could not fail -- an assigned origin kept
    # verbatim is the test below.

    def test_a_manual_origin_gets_the_same_answer_from_either_direction(self):
        """ONE state, one answer.  An origin the person assigned that leaves the
        atoms outside, typed on the Cell page, and the same origin read back
        off disk: kept verbatim, warned about (the edit stands), never
        auto-fixed.  Assigning P stores -P (D1).

        This used to be two tests either side of a ``live_edit`` flag.  Both
        passed, which was the proof the flag selected nothing -- it was removed
        2026-08-02.  What the pair was really pinning is the equality below, so
        that is what this asserts.

        Contract: `model/structure-periodicity.md` § 6.0, *A stated offset*.
        """
        P = np.array([100.0, 100.0, 100.0])      # a corner nowhere near the atoms
        stored = _mol()
        stored.cell = np.eye(3) * 7.0
        stored.engine_offset = -P
        stored.__post_init__()
        checked, notes = validate_periodicity(stored)
        np.testing.assert_allclose(checked.engine_offset, -P)
        assert _said(notes, "cell.atoms_outside"), _wheres(notes)

        # The live half: the same corner assigned through the Cell-page door.
        live = _mol()
        live.cell = np.eye(3) * 7.0
        live.__post_init__()
        typed, _ = apply_edit(live, "box_corner", P.tolist())
        np.testing.assert_allclose(typed.engine_offset, -P)
        assert validate_periodicity(typed)[1] == notes

    def test_too_small_cell_is_a_hard_error(self):
        """SCIENCE. A cell shorter than the molecule's extent is REFUSED outright,
        naming the axis that is too short.

        Catches a box that cannot hold the structure reaching an emitter: SIESTA
        would fold the molecule onto its own periodic image and return numbers for
        a system nobody asked for. `_mol()` is 2 Å across and the cell is 1 Å, so
        no choice of origin can fit it -- the refusal is unconditional, not a
        warning.

        Contract: `model/structure-periodicity.md` § 6.0, check 1 (a span wider
        than the cell along a non-periodic axis is refused naming the axis) +
        § 6.1a.

        *(This carried a "KNOWN WEAK" note saying the axis half was asserted
        as `"a" in str(exc)` -- any English sentence satisfies that.  Fixed
        since: the assertions below require `"along a"` AND that b and c are
        not named.  The note outlived the weakness by long enough to
        contradict the code fifteen lines under it.)*
        """
        s = _mol()
        s.cell = np.eye(3) * 1.0                     # extent 2 Å can't fit
        s.__post_init__()
        # Matched on the sentence ("cannot contain") until 2026-08-03.  What
        # this test means is that the state is REFUSED and the user is told
        # which axis -- not that a particular phrasing survives.
        with pytest.raises(ValueError) as exc:
            validate_periodicity(s)
        # `"a" in str(exc.value)` stood here until 2026-09-09 and could not
        # fail: the letter is in "than", "cannot", every English sentence, so
        # a refusal naming the WRONG axis or no axis at all passed.  `along a`
        # is the emitter's own phrasing and is the smallest string that can
        # only come from naming this axis.
        msg = str(exc.value)
        assert "along a" in msg, (
            f"the refusal does not name the offending axis: {msg}")
        assert "along b" not in msg and "along c" not in msg, (
            f"the refusal names an axis that fits: {msg}")

    def test_left_handed_cell_is_refused(self):
        """SCIENCE. A lattice with det < 0 is refused before anything else is judged.

        Catches a left-handed frame reaching an engine. Handedness is not
        cosmetic: reciprocal vectors, k-point sampling and any cross-product
        convention flip sign with it, so the run completes and the numbers are
        wrong rather than absent. `diag(7, 7, -7)` is the minimal such cell.

        Contract: `model/structure-periodicity.md` § 6.1 (`_require_right_handed`);
        the door-level half of the same rule is `TestARefusedCellIsA400`.
        """
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
    returns the box to the DERIVED regime, and an assigned origin goes with the
    typed cell; an explicit cell demotes vacuum to reference-only; an origin is
    assigned on a typed cell only, and the edits that keep the cell keep it
    (§ 6.0, *A stated offset*).  No op ever moves an atom: coordinate
    rewrites are a Modify op, not a periodicity edit.
    """

    def test_calibrate_is_not_a_periodicity_op(self):
        """`calibrate` is not a periodicity op, and asking for it is an ERROR rather
        than a silent no-op.

        Catches the op's re-introduction, and -- the sharper half -- catches
        `apply_edit` growing a permissive fall-through: an unknown op that returns
        the structure unchanged with status 200 tells the client its edit landed
        when nothing happened.

        Contract: `model/structure-periodicity.md` § 6.2 (the closed op set, no
        calibrate; emission places the atoms, so there is nothing to bake --
        § 6.0, D3). Commemorates 2026-07-29, when the door's docstring still
        advertised it.
        """
        assert "calibrate" not in OPS
        with pytest.raises(ValueError, match="unknown periodicity op"):
            apply_edit(_mol(), "calibrate", None)

    def test_vacuum_edit_resets_manual_state_to_derived(self):
        """Editing vacuum -- an UPSTREAM parameter -- discards the explicit cell and
        origin instead of keeping a box the new vacuum contradicts.

        Catches the half-applied state: a user widens the vacuum, the stored cell
        ignores it, and the box on screen no longer matches the number in the
        field. The reset must also be announced ("DERIVED regime"), because
        silently dropping a cell the user typed is worse than refusing.

        Contract: `model/structure-periodicity.md` § 6.2, the v3 regime model.
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.engine_offset = np.array([-0.5] * 3)       # an origin assigned
        s.__post_init__()
        out, notes = apply_edit(s, "vacuum", [3.0, 3.0, 3.0])
        assert out.cell is None and out.engine_offset is None
        assert out.vacuum == (3.0, 3.0, 3.0)
        assert any("DERIVED regime" in n["message"] for n in notes)

    def test_vacuum_edit_refused_on_a_periodic_axis(self):
        """SCIENCE. Vacuum cannot be set on a `periodic` axis; the edit is refused,
        naming the axis kind.

        Catches padding being added to an axis that wraps. On a periodic axis the
        lattice constant IS the length -- inserting vacuum there breaks
        commensurability with the bulk crystal, and the run silently models a
        slab with a gap instead of the periodic solid the user asked for.

        Contract: `model/structure-periodicity.md` § 2 (what each axis kind means)
        + § 6.2 (which ops each kind accepts).
        """
        s = _mol()
        s.cell = np.diag([10.0, 10.0, 4.0])
        s.axis_kind = ("isolated", "isolated", "periodic")
        s.__post_init__()
        with pytest.raises(ValueError, match="periodic"):
            apply_edit(s, "vacuum", [3.0, 3.0, 3.0])

    def test_axis_edit_resets_to_derived_when_nonperiodic(self):
        """Turning every axis back to `isolated` drops the explicit cell that only
        made sense while an axis was periodic, and says so.

        Catches the stale-downstream case in the other direction: an axis kind
        changed while a commensurate cell stays stored, so the box still has the
        crystal's length on an axis that is now a molecule in vacuum.

        Contract: `model/structure-periodicity.md` § 6.2 (an upstream edit resets
        downstream state, loudly).
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.engine_offset = np.array([-0.5] * 3)       # an origin assigned
        s.__post_init__()
        out, notes = apply_edit(
            s, "axis_kind", ["isolated", "isolated", "isolated"])
        assert out.cell is None and out.engine_offset is None
        assert any("DERIVED regime" in n["message"] for n in notes)
        # A transport axis with zero structure extent would be a
        # degenerate derived box -> refused (its own pin below).

    def test_axis_to_periodic_keeps_the_explicit_cell(self):
        """The reset is CONDITIONAL: turning an axis periodic while an explicit cell
        is stored keeps that cell, and says it was respected.

        Catches an over-eager reset. This is the pair to the test above -- if
        `axis_kind` reset unconditionally, a user making an axis periodic would
        lose the lattice they had just typed and land in the state the very next
        test shows is refused.

        Contract: `model/structure-periodicity.md` § 6.2.
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(
            s, "axis_kind", ["isolated", "isolated", "periodic"])
        assert out.cell is not None                      # respected, kept
        assert any("respected" in n["message"] for n in notes)

    def test_axis_to_periodic_without_a_cell_is_refused(self):
        """SCIENCE. An axis cannot become `periodic` unless an explicit commensurate
        cell is stored.

        Catches a periodic axis whose length comes from a derived bounding box +
        vacuum -- a lattice constant nobody chose. Every periodic property (band
        structure, k-sampling, the image separation) is a function of that length,
        so a derived one produces a run that converges on the wrong crystal.

        Contract: `model/structure-periodicity.md` § 2 + § 6.2; the atomic form of
        the same pairing rule is `TestTheBlockOp`.
        """
        with pytest.raises(ValueError, match="explicit commensurate cell"):
            apply_edit(_mol(), "axis_kind",
                       ["isolated", "isolated", "periodic"])

    def test_cell_edit_keeps_an_assigned_origin(self):
        """Typing a new cell keeps an origin the person had already assigned.

        Catches a cell edit dropping a stated offset -- the user resizes the box
        and it jumps back to the rule's place, the frame moving under the
        coordinates without anyone asking.

        Contract: `model/structure-periodicity.md` § 6.0, *A stated offset*
        (edits keep it).
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.engine_offset = np.array([-0.5] * 3)
        s.__post_init__()
        out, _ = apply_edit(s, "cell", (np.eye(3) * 12.0).tolist())
        np.testing.assert_allclose(out.engine_offset, [-0.5] * 3)

    # `test_cell_edit_without_origin_respects_vacuum` RETIRED 2026-09-25: it
    # pinned "origin first, then vacuum" and the vacuum-derived corner
    # (§ 6.0 retires both: a vacuum places nothing under a typed cell).  That
    # nothing is stored for the rule's offset is the state-table test above.

    def test_cell_null_returns_to_derived(self):
        """`cell: null` is the way back to the derived regime, and it clears the
        origin with it.

        Catches an orphaned assigned origin: an offset left behind after the
        cell it placed is gone. That state is refused by the block op two
        classes down, so leaving it reachable here makes the field-at-a-time
        path able to build a structure the atomic path rejects.

        Contract: `model/structure-periodicity.md` § 6.0 (an origin is assigned
        on a typed cell only) + § 6.2.
        """
        s = _mol()
        s.cell = np.eye(3) * 10.0
        s.engine_offset = np.array([-1.0] * 3)
        s.__post_init__()
        out, _ = apply_edit(s, "cell", None)
        assert out.cell is None and out.engine_offset is None

    # `test_origin_edit_warns_vacuum_not_respected` RETIRED 2026-09-25: a
    # vacuum is inert because the cell is typed, not because of an origin, and
    # `test_cell.py` pins that finding (`cell.vacuum_ignored`) where it is made.

    def test_automatic_clears_a_stated_offset_on_a_box_sized_from_vacuum(self):
        """*Automatic* clears an engine's stated 0 on a box sized from the
        vacuum -- how a PySCF run saves its geometry -- and stores None.

        Catches the typed-cell check refusing the way back: an origin is
        ASSIGNED on a typed cell only, but clearing one needs no cell, and a
        structure refused here could never return to the rule.  (The typed-cell
        case is `test_origin_reset_null_payload_through_the_door`.)

        Contract: `model/structure-periodicity.md` § 6.0, *A stated offset*
        (*Automatic* clears it; an engine's output states 0).
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.engine_offset = np.zeros(3)
        s.__post_init__()
        assert s.cell is None
        out, _ = apply_edit(s, "box_corner", None)
        assert out.engine_offset is None

    def test_no_op_ever_moves_atoms(self):
        """SCIENCE. No periodicity op moves an atom -- across vacuum, cell, origin
        and their resets.

        Catches a metadata edit that "helpfully" wraps or recentres coordinates.
        That silently changes the physical system: bond lengths across a boundary,
        which atoms are neighbours, and the frozen-atom indices that name them.
        The atoms are placed where the engine gets them, at emission -- never
        on the stored structure.

        Contract: `model/structure-periodicity.md` § 6.2 (the ops change metadata
        only) + § 6.0 (placement happens at emission).
        """
        s = _mol()
        out, _ = apply_edit(s, "vacuum", [1.0, 1.0, 1.0])
        assert np.allclose(out.positions, s.positions)
        withcell, _ = apply_edit(s, "cell", (np.eye(3) * 9.0).tolist())
        assert np.allclose(withcell.positions, s.positions)
        for op, payload in [("box_corner", [7.0, 7.0, 7.0]),
                            ("box_corner", None),
                            ("cell", None),
                            ("block", {"cell": (np.eye(3) * 9.0).tolist(),
                                       "box_corner": [1.0, 2.0, 3.0],
                                       "axis_kind": ["isolated"] * 3,
                                       "vacuum": None})]:
            out, _ = apply_edit(withcell, op, payload)
            assert np.allclose(out.positions, s.positions), op


# `test_calibrated_then_emit_equals_emit` RETIRED 2026-09-25 with calibrate
# itself (`model/structure-periodicity.md` § 6.0, user: "we can retire the
# calibrate button").  What it guarded -- emission independent of a prior
# bake -- is the placement rule's idempotence, pinned by `tests/test_cell.py`
# and, through prep, by `tests/test_engine_offset_reaches_every_deck.py`.


# ------------------------------------------------------------------ #
#  The unified endpoint                                               #
# ------------------------------------------------------------------ #


class TestTheBlockOp:
    """`block` sets § 6.2's whole cell and checks ONCE.

    IT EXISTS BECAUSE THE FIELD-AT-A-TIME OPS CANNOT EXPRESS A CHANGE TO TWO
    OF THEM.  Two requests are not atomic -- the second can be refused after
    the first has landed -- and two of the transitions are unreachable in
    EITHER order, which is what these tests are mostly about:

      * an axis cannot become `periodic` until an explicit `cell` is stored;
      * that cell cannot be cleared while an axis IS `periodic`.

    So "become a periodic crystal" and "go back to a derived box" were both
    journeys through a state the gate refuses.  `block` builds the result and
    validates that, so the intermediate states never exist.
    """

    def test_becoming_a_periodic_crystal_takes_one_request(self):
        """Unreachable before: `axis_kind` refuses periodic with no cell, and
        `cell` alone leaves the axes isolated."""
        s = _mol()
        assert s.cell is None
        out, notes = apply_edit(s, "block", {
            "cell": [[10, 0, 0], [0, 10, 0], [0, 0, 20]],
            "box_corner": None,
            "axis_kind": ["periodic", "periodic", "transport"],
            "vacuum": [5.0, 5.0, 0.0],
        })
        assert out.axis_kind == ("periodic", "periodic", "transport")
        assert np.allclose(out.cell, np.diag([10.0, 10.0, 20.0]))
        assert any("explicit cell" in n["message"] for n in notes), notes

    def test_going_back_to_a_derived_box_takes_one_request(self):
        """The other direction, and the one the panel's "Use default" button
        could not offer: it had to disable itself on exactly the structures a
        user most wants it for."""
        s = _mol()
        s.cell = np.diag([10.0, 10.0, 20.0])
        s.axis_kind = ("periodic", "periodic", "transport")
        s.engine_offset = np.array([-1.0] * 3)       # an origin assigned
        s.__post_init__()
        out, notes = apply_edit(s, "block", {
            "cell": None, "box_corner": None,
            "axis_kind": ["isolated"] * 3, "vacuum": [4.0, 4.0, 4.0],
        })
        assert out.cell is None and out.engine_offset is None
        assert out.axis_kind == ("isolated",) * 3
        assert out.vacuum == (4.0, 4.0, 4.0)
        assert any("derived" in n["message"] for n in notes), notes

    def test_a_periodic_axis_with_no_cell_is_refused_on_the_whole_block(self):
        """The one pairing rule, stated on the RESULT rather than on the order
        the fields arrived in."""
        with pytest.raises(ValueError, match="periodic axis needs an explicit cell"):
            apply_edit(_mol(), "block", {
                "cell": None, "box_corner": None,
                "axis_kind": ["periodic", "isolated", "isolated"],
                "vacuum": None,
            })

    def test_an_origin_is_assigned_on_a_typed_cell_only(self):
        """An origin is assigned on a typed cell and nowhere else, whichever
        door asks -- and on a typed cell, the block's corner is stored as the
        offset that puts the box there (P -> -P).

        Catches the meaningless state slipping through the ATOMIC door: `block`
        is the one path that could build it in a single request, so it runs
        the same pairing check on the RESULT.  The positive half is the Cell
        page's own path: `modify/periodicity.js` sends the whole block.

        Contract: `model/structure-periodicity.md` § 6.0, *A stated offset*
        (D1) + § 6.2 (the block op validates the composed result).
        """
        with pytest.raises(ValueError, match="typed cell only"):
            apply_edit(_mol(), "block", {
                "cell": None, "box_corner": [0.0, 0.0, 0.0],
                "axis_kind": ["isolated"] * 3, "vacuum": None,
            })
        with pytest.raises(ValueError, match="typed cell only"):
            apply_edit(_mol(), "box_corner", [0.0, 0.0, 0.0])
        out, _ = apply_edit(_mol(), "block", {
            "cell": (np.eye(3) * 9.0).tolist(), "box_corner": [1.0, 2.0, 3.0],
            "axis_kind": ["isolated"] * 3, "vacuum": None})
        np.testing.assert_allclose(out.engine_offset, [-1.0, -2.0, -3.0])

    def test_a_partial_block_is_refused_rather_than_half_applied(self):
        """A key this op does not know is a caller sending a shape it invented,
        and quietly ignoring it is how half a cell lands."""
        with pytest.raises(ValueError, match="unknown key"):
            apply_edit(_mol(), "block", {
                "cell": None, "box_corner": None,
                "axis_kind": ["isolated"] * 3, "vacuum": None,
                "kgrid": [4, 4, 1],
            })
        with pytest.raises(ValueError, match="whole cell"):
            apply_edit(_mol(), "block", [[10, 0, 0], [0, 10, 0], [0, 0, 10]])

    def test_a_left_handed_cell_is_refused_by_the_one_checker(self):
        """`block` runs the same check the field-at-a-time ops run -- once, on
        the result -- so nothing is enforced less because it arrived together.
        """
        with pytest.raises(ValueError, match="left-handed"):
            apply_edit(_mol(), "block", {
                "cell": [[-4, 0, 0], [0, 4, 0], [0, 0, 4]],
                "box_corner": None,
                "axis_kind": ["isolated"] * 3, "vacuum": None,
            })

    def test_a_null_vacuum_stays_never_chosen(self):
        """Vacuum has three states (§ 6.1) and `block` must carry all three:
        the panel sends null for "nobody decided", which is what the 3 Å
        default exists FOR.  Sending a number instead would stamp a value the
        user never chose and lose the (default) mark."""
        s = _mol()
        s.vacuum = (7.0, 7.0, 7.0)
        s.__post_init__()
        out, _notes = apply_edit(s, "block", {
            "cell": None, "box_corner": None,
            "axis_kind": ["isolated"] * 3, "vacuum": None,
        })
        assert out.vacuum is None
        assert out.effective_vacuum() == (3.0, 3.0, 3.0)


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
        """The door's happy path: the op is applied, the manual state resets, and the
        answer carries BOTH the raw values and the § 3 resolved views.

        Catches the client being handed a cell block whose "as it will actually be
        used" half is missing -- MolView adopts this block verbatim, so a missing
        `resolved_cell` draws no box at all, and a missing warn level loses the
        only notice that says the user's typed cell was just discarded.

        Contract: `model/structure-periodicity.md` § 6.2 (the door's answer shape)
        + `web/molview.md` § 9.3 (both halves travel together).
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0                    # manual state...
        s.engine_offset = np.array([-1.0] * 3)       # ...with an origin assigned
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
        assert per["box_corner"] is not None and "resolved_vacuum" in per
        # The reset takes the assigned origin with the typed cell (§ 6.0).
        assert per["engine_offset"] is None
        assert any(n["severity"] == "warn" for n in j["notices"])

    def _explicit_box_structure(self):
        """Three atoms in a row inside a box the USER typed: a 4 A cube with its
        origin assigned at the world origin, all three axes isolated.  Manual,
        so no rule recomputes where it sits -- what happens to it is only ever
        reported.
        """
        s = Structure(elements=["H", "H", "H"],
                      positions=np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]))
        s.cell = np.eye(3) * 4.0
        s.engine_offset = np.zeros(3)
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

    def test_translating_the_whole_molecule_leaves_the_box_and_says_so(self,
                                                                       client):
        """Every atom moved 50 Å: a box whose origin the person ASSIGNED stays
        where they put it, and the atoms now outside it are named.

        Until 2026-09-25 a whole-structure move carried the box along and there
        was nothing to report; the user retired that -- *"leave the cell alone,
        moving atoms only moves atoms"* (`model/structure-periodicity.md`
        § 6.0).  Under *Automatic* the rule places the box around the atoms
        wherever they now are, so this holds for an assigned origin only.
        """
        s = self._explicit_box_structure()
        r = client.post("/api/modify/translate", json={
            "structure": self._envelope(s), "dx": 50.0, "dy": 0.0, "dz": 0.0})
        assert r.status_code == 200
        body = r.get_json()
        per = body["periodicity"]
        assert per["engine_offset"] == [0.0, 0.0, 0.0] and \
            per["box_corner"] == [0.0, 0.0, 0.0], "the box moved with the atoms"
        assert _said(body.get("notices"), "cell.atoms_outside"), (
            f"atoms left the box and nothing said so: "
            f"{_wheres(body.get('notices'))}")

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

        # A box that does NOT contain the structure: an origin assigned far
        # from the atoms.
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        s.cell = np.eye(3) * 5.0
        s.engine_offset = np.array([-50.0, -50.0, -50.0])
        s.axis_kind = ("isolated",) * 3
        s.__post_init__()

        broken = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "box_corner",
            "payload": [50.0, 50.0, 50.0]})
        assert broken.status_code == 200, broken.get_json()
        assert _said(broken.get_json()["notices"], "cell.atoms_outside"), (
            "a box that does not contain the structure must be reported: "
            f"{_wheres(broken.get_json()['notices'])}")

        # Now FIX it — an origin that does contain the atoms.
        fixed = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "box_corner",
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
        # it not containing (a `cell` edit keeps the assigned origin), so the
        # edit path has the same thing to say.
        # The box runs 5..15 along x; an atom sits at x = 0.
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
        s.cell = np.eye(3) * 10.0
        s.engine_offset = np.array([-5.0, 0.0, 0.0])
        s.axis_kind = ("isolated",) * 3
        s.__post_init__()

        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "cell",
            "payload": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]})
        assert r.status_code == 200, r.get_json()
        notices = r.get_json()["notices"]
        containment = [n for n in notices
                       if n.get("where") == "cell.atoms_outside"]
        assert len(containment) == 1, (
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
        """A request with no `structure` is a 400 that names the missing key.

        Catches the door reaching into `None` and answering 500 with an HTML page:
        the browser's `r.json()` then reports a network failure and the real
        reason never reaches anybody -- the same failure mode `TestARefusedCellIsA400`
        exists for, at the envelope layer instead of the cell layer.

        Contract: `web/web-api.md` § 1 (every structure door takes the envelope;
        contract violations are a clean 400).
        """
        r = client.post("/api/structure/periodicity",
                        json={"op": "vacuum", "payload": [1, 1, 1]})
        assert r.status_code == 400
        assert "structure" in r.get_json()["error"]

    def test_unknown_op_is_a_400(self, client):
        """An op the door does not implement is a 400, not a silent 200.

        Catches the dangerous no-op: a client asking for `calibrate` (retired) and
        being told OK while the structure came back untouched. The user believes
        the edit landed; nothing did.

        Contract: `model/structure-periodicity.md` § 6.2 (the op set is closed).
        """
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
    2026-07-29).  Where the box sits is now the rule's unless the structure
    states an offset (§ 6.0), so the seams have one thing to agree on: nothing
    is invented."""

    def _write_pair_stating_no_offset(self, dirpath, v9_corner=None):
        """A pair whose sidecar holds an explicit cell and NO origin, with the
        atoms outside it at the world origin — the hemeC state, and a legal one.
        Written BY HAND rather than through ``write()`` so it reaches the read
        seam exactly as a file on disk would.

        ``v9_corner`` writes it as a v9 pair carrying that retired
        ``cell_origin`` instead -- what the electrode builder stored flush
        against the atoms, the placement TranSIESTA refused."""
        import json as _json
        from molbuilder.workingcopy_structure import StructureCodec
        s = _mol()                          # atoms near (10,10,10), vac 2.5
        s.cell = np.eye(3) * 7.0            # explicit cell, no origin,
        s.__post_init__()                   # atoms far outside [0, cell)
        made = StructureCodec().pair(s)
        side = dict(made.sidecar)
        if v9_corner is not None:
            side["schema_version"] = 9
            side.pop("engine_offset", None)
            side["cell_origin"] = list(v9_corner)
        (dirpath / "m.xyz").write_text(made.document, encoding="utf-8")
        (dirpath / "m.molstruct.json").write_text(
            _json.dumps(side), encoding="utf-8")
        return dirpath / "m.xyz"

    @pytest.mark.parametrize("v9_corner", [None, [10.0, 10.0, 10.0]],
                             ids=["v10-automatic", "v9-with-a-retired-corner"])
    def test_the_read_seam_invents_no_corner(self, tmp_path, v9_corner):
        """The READ seam of the pair states nothing it did not read: no offset
        is invented, and the box the rule places wraps the atoms.

        Catches the hemeC divergence at its source: while the read seam invented
        an origin (or refused to derive one), the same file gave two answers
        depending on which door opened it, and re-saving it changed a file the
        user had not edited.  And a v9 pair's `cell_origin` is RETIRED, not
        migrated (D2): v9 cannot tell a corner somebody typed from the one the
        electrode builder stored flush against the atoms.

        Contract: `model/structure-periodicity.md` § 6.1 clause 2 (ONE gate, both
        seams) + § 6.0; `structure-molstruct.md`, *v10*. Commemorates
        projects/hemeC-dithiol, observed live 2026-07-29.
        """
        from molbuilder.workingcopy_structure import StructureCodec
        xyz = self._write_pair_stating_no_offset(tmp_path, v9_corner)
        out = StructureCodec().read(xyz)
        assert out.engine_offset is None, "an offset was invented on the read"
        assert cellmod.resolve(out).contains_atoms, (
            "the box is not drawn round the atoms")

    def test_the_load_door_serves_the_rules_box_corner(
            self, tmp_path, monkeypatch):
        """The same state, one layer out: `/api/build/load` serves the corner the rule
        places the box at, in `box_corner`, so the browser draws the box where the
        file means -- and states no offset, because the file states none.

        Catches the seam between the codec and the wire dropping the resolved half
        -- the exact live symptom of 2026-07-29, where MolView drew the box from
        the world origin while the Cell page showed the wrapping corner. The
        codec-level test above cannot see that; only the served payload can.

        Contract: `model/structure-periodicity.md` § 6.0 (the rule; the wire's
        `box_corner`) + § 8.1 (where the gate runs).
        """
        pytest.importorskip("flask")
        from molbuilder.diagnostics import Capabilities, set_capabilities
        monkeypatch.chdir(tmp_path)
        # The tree is THIS tmp one.  A chdir used to say that on its own,
        # because `projects_root` was cwd-anchored; it resolves from the
        # molbuilder root now (2026-08-22), so the door is told directly.
        from molbuilder.projects import PROJECTS_ROOT_ENV
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "projects"))
        sdir = tmp_path / "projects" / "P" / "structure"
        sdir.mkdir(parents=True)
        xyz = self._write_pair_stating_no_offset(sdir)
        set_capabilities(Capabilities(runtime_config={},
                                      conda_binary="/usr/bin/conda"))
        try:
            from molbuilder.web.app import create_app
            client = create_app(config={}).test_client()
            r = client.post("/api/build/load", json={"path": str(xyz)})
            assert r.status_code == 200, r.get_json()
            per = r.get_json()["periodicity"]
            assert per["engine_offset"] is None
            # Centred on the atoms (x 10..12, y and z at 10), stated as the
            # atoms' own middle rather than as a literal corner.
            np.testing.assert_allclose(
                np.asarray(per["box_corner"]) + 3.5, [11.0, 10.0, 10.0])
        finally:
            set_capabilities(None)


# ------------------------------------------------------------------ #
#  Per-axis-kind containment (review findings, 2026-07-29)            #
# ------------------------------------------------------------------ #


class TestPeriodicAxesAreNeverContained:
    """PINS: docs/model/structure-periodicity.md § 2 (which axis an image
    belongs to decides whether it is a defect) + § 6.0, check 1 (containment
    is required along NON-PERIODIC axes only).

    Along a periodic axis, atoms past a face are periodic images — legal, and
    WARNED (`cell.beyond_periodic_face`, user 2026-09-25: *"warning/error when
    atoms are outside boundary for all cases"*).  Requiring containment there
    made real crystal and junction files unopenable."""

    @pytest.mark.parametrize("kinds, refused", [
        (("periodic",) * 3, False),
        (("periodic", "periodic", "transport"), True)],
        ids=["crystal", "junction"])
    def test_a_span_wider_than_the_cell_is_legal_only_along_a_periodic_axis(
            self, kinds, refused):
        """SCIENCE. Two atoms 12 Å apart along c, in a 10 Å cell.  Along a
        PERIODIC c that is an image the engine wraps -- legal, and warned; along
        a TRANSPORT c it is a device longer than its box, refused naming c.

        Catches the containment check being applied to periodic axes -- it was,
        and it made real crystal and junction files unopenable: a fractional
        coordinate of -0.3 is the same atom as +0.7 where the box repeats -- and
        the opposite failure, an axis that does not repeat judged as if it did.
        Whether an atom outside is a defect is decided by the axis kind, nothing
        else.

        Contract: `model/structure-periodicity.md` § 2 (the axis-kind table) +
        § 6.0, check 1. Review finding, 2026-07-29 (the crystal and junction
        halves were two tests until 2026-09-25).
        """
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0, 0], [0.0, 0, 12.0]]),
                      cell=np.eye(3) * 10.0, axis_kind=kinds)
        if refused:
            with pytest.raises(ValueError, match="along c"):
                validate_periodicity(s)
        else:
            _, notes = validate_periodicity(s)
            assert [(n["where"], n["severity"]) for n in _problems(notes)] == [
                ("cell.beyond_periodic_face", "warn")], notes

    # `test_stored_manual_origin_is_warned_never_rewritten` RETIRED 2026-09-25:
    # the stored half of `test_a_manual_origin_gets_the_same_answer_from_either_
    # direction` above, which asserts it.

    def test_unfittable_cell_edit_is_refused_naming_the_axis(self):
        """A cell the structure cannot fit is refused at the edit, naming the
        axis it is too short along -- a stored-but-invalid cell locked every
        later door.

        CONTRACT: `model/structure-periodicity.md` § 6.0, check 1 (refused
        naming the axis).  The molecule is 2 Å along x in a 1 Å cube, so a is
        the axis; the sentence is the checker's (``cell.unfittable``)."""
        with pytest.raises(ValueError, match=r"along a\b"):
            apply_edit(_mol(), "cell", (np.eye(3) * 1.0).tolist())

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

    def test_preflight_sees_what_generate_sees(self, client):
        """A planar molecule with vacuum: preflight must NOT judge the vacuum-0
        phantom (it used to error on a degenerate box) -- it judges the body's
        own 4 Å, which SIESTA's advice calls thin (§ 6.1a: under 8 Å per side
        is advised against, not refused).

        The assertion this replaced -- "Thin vacuum" absent or "4.0" absent --
        could not fail: the message prints the gap as "4 Å"."""
        r = client.post("/api/build/preflight", json={
            "structure": _env_per(self._XYZ, {"cell": None,
                            "axis_kind": ["isolated"] * 3,
                            "vacuum": [4.0, 4.0, 4.0]}), "engine": "siesta",
            "params": {"system_label": "pin"},
        })
        assert r.status_code == 200, r.get_json()
        issues = r.get_json().get("issues") or []
        wheres = {i["where"] for i in issues}
        assert "cell.no_volume" not in wheres, wheres
        assert not any(i["severity"] == "error" for i in issues), issues
        assert "cell.vacuum_thin" in wheres, (
            "the body's vacuum never reached the judge", wheres)



from pathlib import Path  # noqa: E402  (used by TestTabEmitContract)

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import (from_xyz as _env,
                             from_xyz_with_periodicity as _env_per)



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
        """A pair whose sidecar holds a left-handed cell OPENS -- with its regions
        intact and the bad cell still present.

        Catches two distinct traps. Raising at the read leaves the user unable to
        reach the one page where a box can be corrected (the Cell page needs the
        structure on screen). Silently DROPPING the bad cell is worse: the user
        cannot correct a value they were never shown. Losing `regions` on the way
        in costs them the frozen-atom work they had already done.

        Contract: `model/structure-periodicity.md` § 8.2, "reading does not judge"
        (2026-08-03). The other half -- that no calculation is generated from it --
        is `test_but_no_calculation_is_generated_from_it`.
        """
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
        """A load of a pair whose assigned origin leaves the atoms outside the box
        reports it (`cell.atoms_outside`), with `about: cell` -- serves the origin
        as stored -- and carries NO key claiming anything was corrected.

        Catches two failures at once: a load that gates silently (the user is never
        told the box no longer wraps the atoms, cell-plan.md 3f), and a load that
        "fixes" the origin and carries a correction marker -- a phantom dirty state
        that makes the session look edited and re-saves a file the user did not
        touch. `about` is what puts the sentence on the Cell page rather than
        above the atom list.

        Contract: `model/structure-periodicity.md` § 6.1 clause 6 + § 8.2 ("the
        report has to arrive"); `web/molview.md` § 6.8 for the notice shape.
        """
        pytest.importorskip("flask")
        from molbuilder.diagnostics import Capabilities, set_capabilities
        import json as _json
        from molbuilder.workingcopy_structure import StructureCodec
        monkeypatch.chdir(tmp_path)
        # The tree is THIS tmp one.  A chdir used to say that on its own,
        # because `projects_root` was cwd-anchored; it resolves from the
        # molbuilder root now (2026-08-22), so the door is told directly.
        from molbuilder.projects import PROJECTS_ROOT_ENV
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "projects"))
        sdir = tmp_path / "projects" / "P" / "structure"
        sdir.mkdir(parents=True)
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.engine_offset = np.array([-50.0, -50.0, -50.0])   # the corner at 50 Å
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
            # The load REPORTS what is wrong and changes nothing -- and the
            # shape says so: no key claiming a correction, hence no phantom
            # dirty state.
            assert _said(notices, "cell.atoms_outside"), _wheres(notices)
            assert all(set(n) == {"severity", "message", "where", "about"}
                       for n in notices), notices
            # ...and `about` is the SUBJECT, which is what puts the sentence on
            # the Cell page rather than above the atom list (molview.md § 6.8).
            assert all(n["about"] == "cell" for n in notices), notices
            # The served model: the origin as stored through the v10 pair,
            # nothing corrected.
            per = j["periodicity"]
            assert per["engine_offset"] == [-50.0] * 3, per["engine_offset"]
            assert per["box_corner"] == [50.0] * 3, per["box_corner"]
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
        """Six malformed envelopes -- including metadata at the top level instead of
        inside the structure -- each answer 400 with `ok: False` and a reason.

        Catches the door crashing into a 500 on a shape it did not expect. Flask's
        500 is an HTML page, so the browser's `r.json()` raises and the failure
        reaches the user as "network error" with no reason at all. The
        top-level-`regions` case is the realistic one: a caller that flattened the
        envelope by hand.

        Contract: `web/web-api.md` § 1 (the envelope) + `model/structure-periodicity.md`
        § 6.2 (the door's error contract).
        """
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
        """A `cell` payload that is not 3x3 is a 400 whose message says what shape was
        wanted.

        Catches a malformed lattice reaching numpy, where `[1, 2, 3]` broadcasts or
        raises deep in the gate and surfaces as a 500 with a traceback -- rather
        than the one sentence that tells the caller what to send.

        Contract: `model/structure-periodicity.md` § 6.2 (the door's error contract).
        """
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(_mol()), "op": "cell",
            "payload": [1, 2, 3]})
        assert r.status_code == 400
        assert "3×3 matrix" in r.get_json()["error"]

    def test_refused_edit_maps_to_a_clean_400(self, client):
        """A REFUSED edit -- vacuum on a periodic axis -- reaches the client as a 400
        carrying the gate's own reason.

        Catches the gate's `ValueError` escaping the door as a 500. This is the
        Cell-page arm of the failure `TestARefusedCellIsA400` documents for the
        other six doors: the refusal is the user's to fix, so the reason has to
        survive the trip.

        Contract: `model/structure-periodicity.md` § 6.2 + § 8.2 (one way in, two
        verdicts).
        """
        s = _mol()
        s.cell = np.diag([10.0, 10.0, 4.0])
        s.axis_kind = ("isolated", "isolated", "periodic")
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "vacuum",
            "payload": [3, 3, 3]})
        assert r.status_code == 400
        assert "periodic" in r.get_json()["error"]

    # `test_cell_op_anchors_origin_through_the_door` RETIRED 2026-09-25: a thin
    # caller of `to_wire`, whose first assertion (`.get("cell_origin") is None`)
    # could not fail once the key was retired.  What the wire sends for the
    # rule's box is `test_the_load_door_serves_the_rules_box_corner` and
    # `test_the_wire_carries_unset_and_the_resolved_view`.

    def test_axis_kind_op_resets_to_derived_through_the_door(self, client):
        """The `axis_kind` op's reset survives the door: the answer carries
        `cell: null`.

        Catches the endpoint and the gate drifting apart -- a door that applies the
        edit but serialises the INCOMING cell would report the reset as having not
        happened, and the browser would keep drawing the discarded box.

        Contract: `model/structure-periodicity.md` § 6.2.
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "axis_kind",
            "payload": ["isolated"] * 3})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["periodicity"]["cell"] is None

    def test_origin_reset_null_payload_through_the_door(self, client):
        """A JSON `null` payload reaches the gate as *Automatic*, and a MISSING
        payload is a 400 rather than a silent clear.

        Catches the door treating `payload: null` as absent and skipping the op --
        the Automatic button then does nothing, silently, with a 200 -- and the
        converse, a dropped key read as a clear.  `box_corner` is an op whose
        payload is legitimately null, so it is one where "missing" and "null"
        must not be conflated.

        Contract: `model/structure-periodicity.md` § 6.0, *A stated offset*
        (*Automatic* clears it) + § 6.2; the door's docstring (a payload is
        required, and may be null).
        """
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.engine_offset = np.array([-0.5] * 3)
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "box_corner",
            "payload": None})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["periodicity"]["engine_offset"] is None
        missing = client.post("/api/structure/periodicity", json={
            "structure": self._envelope(s), "op": "box_corner"})
        assert missing.status_code == 400, missing.get_json()


# `TestDocMatchesTheDoor` RETIRED 2026-09-25 -- its two tests pinned the TEXT
# of a shipped document (§ 6.2's op table) and of the door's docstring
# (`process/testing.md` § 3a: a test asserts behaviour, not prose).  The op set
# lives in `periodicity_gate.OPS`; an unknown op is refused
# (`test_unknown_op_is_a_400`); the documents are kept true by review.


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

    # `test_the_derived_box_stays_centred_on_the_structure` RETIRED 2026-09-25:
    # it pinned the derived corner's use of the effective vacuum.  Every box
    # is centred by the one rule now (§ 6.0), asserted of the wire below and
    # of the rule itself in `test_cell.py`.

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
        assert said[0]["severity"] == "info"
        assert said[0]["about"] == "cell"
        msg = said[0]["message"]
        assert "3 Å" in msg          # the gap it chose, in the message
        # It must state the physical consequence in the currency that matters:
        # vacuum is per side, so the gap between images is TWICE it.
        assert "6 Å" in msg, f"the image gap is not named: {msg}"

    def test_a_set_vacuum_is_not_announced(self):
        """Nothing was defaulted, so there is nothing to disclose.

        BY ITS ID, for the reason the test above states in full -- and this
        one is why that reason is not theoretical.  It matched the prose
        ``"no vacuum was set"``; the note says *"No vacuum set"*, with no
        "was", so the filter matched nothing and the assert was vacuous from
        the day it was written.  A negative test that cannot see the thing it
        denies passes whatever the code does.
        """
        s = self._planar()
        s.vacuum = (5.0, 5.0, 5.0)
        s.__post_init__()
        _, notes = validate_periodicity(s)
        assert not [n for n in notes
                    if n["where"] == "cell.vacuum_defaulted"], (
            "a vacuum the user SET was announced as defaulted")

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
        a surprise.

        And the same for WHERE the box sits (plan § 5q.3, D4): `engine_offset`
        is what the structure states -- `null` under Automatic -- and
        `box_corner` is where the box is drawn, `-offset` of these coordinates:
        centred on the atoms by the rule, or at the origin a person assigned."""
        s = self._planar()
        per = s.to_wire()["periodicity"]
        assert per["vacuum"] is None, "unset must travel as null, not [0,0,0]"
        assert per["resolved_vacuum"] == [3.0, 3.0, 3.0]
        assert per["engine_offset"] is None
        lo, hi = s.positions.min(axis=0), s.positions.max(axis=0)
        np.testing.assert_allclose(
            np.asarray(per["box_corner"]) + np.diag(per["resolved_cell"]) / 2.0,
            (lo + hi) / 2.0, err_msg="the rule's box is not centred on the atoms")

        assigned = Structure(elements=["H"], positions=np.zeros((1, 3)),
                             cell=np.eye(3) * 10.0, axis_kind=("isolated",) * 3,
                             engine_offset=np.array([1.0, 2.0, 3.0]))
        tp = assigned.to_wire()["periodicity"]
        assert tp["engine_offset"] == [1.0, 2.0, 3.0]
        assert tp["box_corner"] == [-1.0, -2.0, -3.0]

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

    THREE independent layers stop one from ever being emitted; this pins each
    so a future change cannot quietly remove the last of them.

    THERE WERE FOUR until 2026-09-21.  The first was `__post_init__` refusing
    to construct one at all, and it was removed on purpose: § 8.2 says reading
    does not judge, and a model that cannot HOLD a bad box cannot show one on
    the Cell page either -- so a pair whose sidecar carried one could not be
    opened, and the only ways out were to hand-edit the JSON outside
    molbuilder or delete it and lose the labels with it.  What replaces it is
    not silence: the box loads and the load REPORTS it, which is layer 0
    below.  Refusing still happens, at the two doors that are actually about
    to act on the box.
    """

    def test_layer0_it_can_be_held_and_is_reported_rather_than_hidden(self):
        """The layer that replaced "cannot even be constructed".

        A degenerate lattice may exist in memory -- that is what lets the
        Cell page show you the box you have to fix.  What must never happen
        is it existing QUIETLY, so the same checker that refuses at the edit
        and emit doors answers here too, as a finding.
        """
        from molbuilder.cell import resolve_and_check
        s = self._flat()
        s.cell = np.diag([8.0, 8.0, 0.0])
        s.__post_init__()                      # must not raise
        assert s.cell is not None, "the box the user has to fix was dropped"
        _rc, issues = resolve_and_check(s)
        assert [i.where for i in issues] == ["cell.no_volume"]

    @staticmethod
    def _flat():
        return Structure(elements=["O", "H", "H"],
                         positions=np.array([[0.0, 0.0, 0.0],
                                             [0.757, 0.586, 0.0],
                                             [-0.757, 0.586, 0.0]]))

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
        # The slab op (archive/2026-09-01-modify-redesign-plan.md § 3).  It reads no selection --
        # dx, dy and start_z are absolute -- so unlike every op above it needs
        # no `indices` to have something to do.  It replaced `electrode`,
        # which centred on a picked group; that route went 2026-09-01 and its
        # row here with it.
        # 2 x 2 is the smallest slab the builder accepts -- a one-wide
        # surface cell puts anything on it in contact with its own image.
        # The size is incidental here; this fixture is about the gate.
        "/api/modify/slab":        {"element": "Au", "plane": "111",
                                    "m": 2, "n": 2, "layers": 2},
        # Adding one structure into another (user, 2026-09-07).  It takes a
        # SECOND envelope beside the one every op takes, and reads no selection
        # -- the incoming fragment is placed on the world origin.
        "/api/modify/append":      {"addition": {
            "elements": ["N"], "positions": [[0.0, 0.0, 0.0]], "metadata": {},
        }},
    }

    @pytest.fixture
    def client(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    #: Routes that return no structure, so there is nothing to validate.
    NOT_AN_OP = {
        # GET, returns dropdown enums -- no structure goes in or out.
        "/api/modify/meta",
        # Measures a bulk run's lattice constant and returns NUMBERS (a, the
        # spacings it implies, and notes).  It reads a file the user points
        # at and never touches the structure on the bench.
        "/api/modify/lattice-from-run",
        # GET, and a pure lookup: element + surface + lattice constant ->
        # the layer spacing and the bond length that follow from them.  No
        # structure is sent, none comes back, and nothing on the bench is
        # read -- it answers a question about a CRYSTAL, not about the
        # thing being edited.  (It exists because the Slab panel used to
        # compute those numbers in JavaScript and was missing d(110).)
        "/api/modify/spacings",
    }

    def _stranded(self):
        """Atoms at x = 50..52 in a 4 A box the user typed, its origin ASSIGNED
        at the world origin -- so it stays there and the atoms are outside."""
        s = Structure(elements=["H", "H", "H"],
                      positions=np.array([[50.0, 0, 0], [51.0, 0, 0], [52.0, 0, 0]]))
        s.cell = np.eye(3) * 4.0
        s.engine_offset = np.zeros(3)
        s.axis_kind = ("isolated",) * 3
        s.__post_init__()
        return s

    def test_the_op_list_is_complete(self, client):
        """ARTIFACT LINT. Every live `/api/modify/*` route is either covered by the
        always-checked sweep or explicitly excused with a reason.

        Catches the real drift: a modify op added later that never goes past the
        periodicity check on its way out, and no one notices because the sweep only
        tested the ops that existed when it was written. The route list is read
        from the running app, so a new op fails this test the day it lands. This is
        what replaces the manual sweep -- there is no per-op test to forget.

        Contract: `model/structure-periodicity.md` § 8.1 (the gate runs at every
        exit) + `process/testing.md` § 3b (an artifact lint over a whole class).
        """
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
        be perfectly correct (``/api/modify/slab`` turns two axes periodic,
        and an atom cannot be outside an axis that wraps), so absence of a
        message proves nothing in either direction.  What must
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
        if its verdict is dropped between the validator and the wire.  A move
        leaves a box whose origin was assigned where it is (D6), so these atoms
        stay stranded outside it; the op has something to say, and it has to
        arrive in ``notices``.
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



    def test_the_preflight_door_refuses(self, client):
        """The preflight door answers a left-handed cell with 400 and the GATE's own
        sentence.

        Catches the refusal escaping as an unhandled exception: Flask answers 500
        with HTML, the browser's `r.json()` throws, and the one line that says what
        to do ("swap two lattice vectors") never reaches the user. Asserting the
        gate's own word rather than just the status matters -- a 400 from some
        earlier check would otherwise pass while the bug survived.

        Contract: `model/structure-periodicity.md` § 6.1 (right-handedness) +
        § 8.1 (the seven doors that run the gate on the way in).
        """
        self._assert_refused(client.post("/api/build/preflight", json={
            # THE BOX IS PART OF THE STRUCTURE.  This stated it in a
            # top-level `periodicity` block beside the envelope -- the legacy
            # request shape, retired 2026-08-04 once nothing sent it.
            "structure": _env_per(self.XYZ, {"cell": self.LEFT_HANDED}), "engine": "siesta",
            "params": {}}), "/api/build/preflight")

    # The /api/spectra/render arm retired with the route (P3);
    # the hand-over door runs the same gate and is pinned in
    # test_task_setup_tab.py.

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

    # `test_the_transport_door_refuses` deleted 2026-09-17 with POST /api/transport/render.



class TestTheCellPageCanRepairWhatItRefuses:
    """A bad box is ADMITTED at the load door precisely so it can be fixed
    here (`blueprints/build.py`: *"a load that refused would leave a
    structure with a bad box unopenable, and so unfixable"*).

    The edit door then gated on the INCOMING state, so every op was
    refused with the very sentence that asks the user to perform it.
    """

    def _mirrored(self):
        import numpy as np
        from molbuilder.structure import Structure
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0., 0., 0.], [0., 0., 1.]]))
        s.cell = np.array([[10., 0, 0], [0, 0, 10.], [0, 10., 0]])
        return s

    def test_a_mirrored_cell_is_admitted_so_it_can_be_corrected(self):
        """`__post_init__` refuses only a near-zero determinant."""
        assert self._mirrored().cell is not None

    def test_setting_a_good_cell_over_a_mirrored_one_is_ACCEPTED(self):
        """The fix the refusal asks for -- 'swap any two of the three
        rows' -- was itself refused."""
        import numpy as np
        from molbuilder.periodicity_gate import apply_edit
        out, _receipts = apply_edit(self._mirrored(), "cell",
                                    [[10., 0, 0], [0, 10., 0], [0, 0, 10.]])
        assert float(np.linalg.det(np.asarray(out.cell, float))) > 0

    def test_clearing_a_mirrored_cell_is_ACCEPTED(self):
        from molbuilder.periodicity_gate import apply_edit
        out, _receipts = apply_edit(self._mirrored(), "cell", None)
        assert out.cell is None

    def test_an_edit_that_LEAVES_the_box_mirrored_is_still_refused(self):
        """The result is what is gated, so a repair that does not repair
        must still be refused -- otherwise this fix would open a hole."""
        import pytest as _pytest
        from molbuilder.periodicity_gate import apply_edit
        with _pytest.raises(ValueError, match="mirrored"):
            apply_edit(self._mirrored(), "cell",
                       [[10., 0, 0], [0, 0, 10.], [0, 20., 0]])
