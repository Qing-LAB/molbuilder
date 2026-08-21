"""The one process line for the cell — resolve once, check once.

Contract: docs/model/structure-periodicity.md § 6.1a (both matrices) and
docs/archive/2026-08-20-cell-plan.md § 6a.

WHAT THIS MODULE REPLACED, because it is what these tests are really guarding.
Before 2026-08-03 the same box was judged in several places at once:

  * "this box has no volume" was decided in FOUR places at TWO thresholds
    (structure.py 1e-8, the gate 1e-8, the gate's reset path 1e-6, the emitter
    1e-6) by THREE mechanisms (raise / notice / raise);
  * findings travelled as gate ``notices`` AND as validator ``Issue``s, two
    vocabularies for one subject, so the Cell page and the Generate panel could
    disagree;
  * notices carried no id, so tests matched on message prose -- which fails on a
    reworded sentence and PASSES on a deleted check.

So the tests below assert on ``where``, never on wording, and the central
property they pin is ONE FINDING PER CAUSE: a broken box must not arrive wearing
three different names.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder import cell as cellmod
from molbuilder.structure import Structure


WATER = np.array([[0.0, 0.0, 0.0],          # flat: zero extent along z
                  [0.757, 0.586, 0.0],
                  [-0.757, 0.586, 0.0]])
ISOLATED = ("isolated", "isolated", "isolated")


def _mol(**kw):
    return Structure(elements=["O", "H", "H"], positions=WATER, **kw)


def _wheres(struct):
    _rc, issues = cellmod.resolve_and_check(struct)
    return [i.where for i in issues]


def _by_id(struct, where):
    _rc, issues = cellmod.resolve_and_check(struct)
    hits = [i for i in issues if i.where == where]
    return hits[0] if hits else None


# --------------------------------------------------------------------- #
#  Matrix A -- what sets the box                                        #
# --------------------------------------------------------------------- #

class TestWhatSetsTheBox:
    """§ 6.1a matrix A, one test per row."""

    def test_an_explicit_cell_is_the_box_and_vacuum_is_ignored(self):
        rc = cellmod.resolve(_mol(cell=np.eye(3) * 20,
                                  vacuum=(5.0, 5.0, 5.0),
                                  axis_kind=ISOLATED))
        assert rc.regime == "manual"
        assert np.allclose(np.diag(rc.box), [20.0, 20.0, 20.0]), (
            "the typed cell must be used verbatim -- not 2 + 2×5")

    def test_a_set_vacuum_makes_the_derived_box(self):
        rc = cellmod.resolve(_mol(vacuum=(2.0, 2.0, 2.0)))
        assert rc.regime == "derived"
        # extent 1.514 × 0.586 × 0  +  2 × 2 on each axis
        assert np.allclose(np.diag(rc.box), [5.514, 4.586, 4.0], atol=1e-6)

    def test_an_unset_vacuum_uses_the_default_gap(self):
        rc = cellmod.resolve(_mol())
        assert rc.stated_vacuum is None, "nobody chose one"
        assert rc.vacuum == (3.0, 3.0, 3.0)
        assert rc.defaulted_axes == (0, 1, 2)

    def test_the_default_gap_does_not_depend_on_the_molecule_s_size(self):
        """3 Å is a vacuum DISTANCE, not a minimum box length (user, 2026-08-03).

        The rule this replaced asked "is the box under 3 Å?", so a large
        molecule -- whose box already exceeded 3 Å -- was given NO gap at all.
        """
        big = Structure(elements=["H", "H"],
                        positions=np.array([[0.0, 0, 0], [20.0, 20.0, 20.0]]))
        assert cellmod.resolve(big).vacuum == (3.0, 3.0, 3.0)
        assert np.allclose(np.diag(cellmod.resolve(big).box), [26.0] * 3)

    def test_vacuum_never_applies_to_a_transport_axis(self):
        rc = cellmod.resolve(_mol(axis_kind=("isolated", "isolated", "transport")))
        assert rc.vacuum[2] == 0.0, "the device length sets a transport axis"
        assert 2 not in rc.defaulted_axes

    def test_a_periodic_axis_with_no_lattice_cannot_resolve(self):
        """A bounding box is not a lattice and never becomes one (§ 3).

        It comes back as ``unresolvable`` rather than an exception: the state is
        a legitimate thing to have loaded and be about to fix, and one code path
        has to serve both "refuse to generate" and "open it so it can be
        corrected" (§ 8.2).
        """
        rc = cellmod.resolve(_mol(axis_kind=("periodic", "isolated", "isolated")))
        assert rc.box is None
        assert rc.unresolvable


# --------------------------------------------------------------------- #
#  Matrix B -- what is checked                                          #
# --------------------------------------------------------------------- #

class TestWhatIsChecked:

    def test_a_healthy_box_is_silent(self):
        assert _wheres(_mol(vacuum=(8.0, 8.0, 8.0))) == []

    def test_a_crystal_is_silent(self):
        """An imported cell whose atoms sit inside it, on periodic axes: the
        commonest state a crystal file has, and nothing to say about it."""
        assert _wheres(_mol(cell=np.eye(3) * 5.6)) == []

    def test_the_default_gap_is_disclosed(self):
        found = _by_id(_mol(), "cell.vacuum_defaulted")
        assert found is not None and found.severity == "info"
        assert "3 Å" in found.message
        # In the currency that matters: vacuum is per side, so the gap between
        # a molecule and its image is TWICE it.
        assert "6 Å" in found.message

    def test_an_ignored_vacuum_is_disclosed(self):
        """§ 3c: the sentence saying why a number you typed stopped mattering
        used to be emitted only ``if not conditions`` -- dropped exactly when
        the box ALSO had a problem, which is when you most needed it."""
        found = _by_id(_mol(cell=np.eye(3) * 30, vacuum=(5.0,) * 3,
                            axis_kind=ISOLATED),
                       "cell.vacuum_ignored")
        assert found is not None and found.severity == "info"

    def test_no_vacuum_set_under_a_typed_cell_says_nothing_about_vacuum(self):
        """Nothing was chosen, so there is no expectation to correct."""
        got = _wheres(_mol(cell=np.eye(3) * 30, axis_kind=ISOLATED))
        assert "cell.vacuum_ignored" not in got
        assert "cell.vacuum_defaulted" not in got, (
            "a defaulted gap is meaningless under a typed cell -- the box is "
            "the cell, and announcing a 3 Å default would describe a number "
            "that never reaches the calculation")

    def test_a_box_with_no_volume_is_an_error(self):
        found = _by_id(_mol(vacuum=(5.0, 5.0, 0.0)), "cell.no_volume")
        assert found is not None and found.severity == "error"

    def test_a_cell_shorter_than_the_structure_is_an_error(self):
        found = _by_id(_mol(cell=np.eye(3) * 1.0, axis_kind=ISOLATED),
                       "cell.unfittable")
        assert found is not None and found.severity == "error"

    def test_a_left_handed_cell_is_an_error(self):
        found = _by_id(_mol(cell=np.diag([20.0, 20.0, -20.0]),
                            axis_kind=ISOLATED),
                       "cell.left_handed")
        assert found is not None and found.severity == "error"

    def test_atoms_outside_a_user_owned_origin_is_a_warning(self):
        s = _mol(cell=np.eye(3) * 20, axis_kind=ISOLATED)
        s.cell_origin = np.array([50.0, 50.0, 50.0])
        s.__post_init__()
        found = _by_id(s, "cell.atoms_outside")
        assert found is not None and found.severity == "warn"
        # The clearances ride WITH the finding they are about.
        # It must SHOW the clearances, whatever the wording: the axis letters
        # and numbers with a sign, so a user can see which side sticks out.
        assert "a " in found.message and "/" in found.message

    def test_an_empty_structure_has_nothing_to_say(self):
        assert _wheres(Structure(elements=[], positions=np.zeros((0, 3)))) == []


# --------------------------------------------------------------------- #
#  THE property the whole module exists for                             #
# --------------------------------------------------------------------- #

class TestOneFindingPerCause:
    """A broken box must not arrive wearing three names.

    Each of these WAS a double- or triple-report before the checker was one
    thing, and each duplicate sends the user to fix something that is not the
    problem.
    """

    def test_a_zero_volume_box_is_not_also_left_handed(self):
        """det == 0 fails ``det > 0``, so a FLAT cell used to be reported as a
        HANDEDNESS problem -- and "swap two lattice vectors" is useless advice
        for a molecule with no thickness."""
        got = _wheres(_mol(vacuum=(5.0, 5.0, 0.0)))
        assert got == ["cell.no_volume"], got

    def test_a_zero_volume_box_is_not_also_uncontained(self):
        """Nothing fits in nothing, so containment fails by construction. The
        volume is the cause; containment is its shadow."""
        assert "cell.atoms_outside" not in _wheres(_mol(vacuum=(5.0, 5.0, 0.0)))

    def test_an_unfittable_cell_is_not_also_uncontained(self):
        """``unfittable`` is the sharper of the two and rules out the repair
        ``atoms_outside`` would suggest: no corner can help."""
        got = _wheres(_mol(cell=np.eye(3) * 1.0, axis_kind=ISOLATED))
        assert "cell.atoms_outside" not in got
        assert "cell.unfittable" in got

    def test_a_left_handed_cell_reports_nothing_else(self):
        """In a mirrored frame the fractional coordinates run backwards, so
        containment and clearances describe a box nobody has."""
        got = _wheres(_mol(cell=np.diag([20.0, 20.0, -20.0]), axis_kind=ISOLATED))
        assert got == ["cell.left_handed"], got

    def test_an_unresolvable_box_reports_nothing_else(self):
        got = _wheres(_mol(axis_kind=("periodic", "isolated", "isolated")))
        assert got == ["cell.unresolvable"], got


class TestOneThreshold:
    """Every site that asks "does this box have a volume?" asks it the same way."""

    def test_the_constant_is_shared_not_copied(self):
        import inspect
        from molbuilder import structure as structmod
        from molbuilder.siesta import input as siesta_input
        for mod in (structmod, siesta_input):
            src = inspect.getsource(mod)
            assert "ZERO_VOLUME_TOL" in src, (
                f"{mod.__name__} does not take the zero-volume threshold from "
                f"the one place that defines it; it used to carry its own "
                f"literal, and the two disagreed (1e-8 vs 1e-6)")

    def test_the_resolver_and_the_checker_agree_on_the_boundary(self):
        """``has_volume`` is the ONE answer; nothing recomputes a determinant."""
        rc = cellmod.resolve(_mol(vacuum=(5.0, 5.0, 0.0)))
        assert rc.has_volume is False
        assert rc.volume < cellmod.ZERO_VOLUME_TOL


class TestTwoVerdictsOneChecker:
    """§ 8.2: what the request is FOR decides what a bad box costs."""

    def test_generating_refuses(self):
        """``report()`` raises on error severity, which is the refusal every
        emit door already promises."""
        from molbuilder.issues import ValidationError
        from molbuilder.validation import report
        _rc, issues = cellmod.resolve_and_check(_mol(vacuum=(5.0, 5.0, 0.0)))
        with pytest.raises(ValidationError):
            report(issues)

    def test_loading_reports_the_same_finding_as_a_warning(self):
        """The severity is not softened -- the same Issue is answered to a
        different question, so a broken box still OPENS and can be fixed."""
        from molbuilder.periodicity_gate import notices_for_report
        _rc, issues = cellmod.resolve_and_check(_mol(vacuum=(5.0, 5.0, 0.0)))
        notices = notices_for_report(issues)
        assert [n["where"] for n in notices] == ["cell.no_volume"]
        assert notices[0]["level"] == "warn", (
            "a loading or modifying door reports; only a generating one refuses")

    def test_every_notice_carries_its_id(self):
        """The absence of this is why four tests once matched message prose --
        which fails on a rewording and passes on a deleted check."""
        from molbuilder.periodicity_gate import notices_for_report
        _rc, issues = cellmod.resolve_and_check(_mol())
        for n in notices_for_report(issues):
            assert set(n) == {"level", "message", "where", "about"}
            assert n["where"].startswith("cell.")
            assert n["about"] == "cell"


class TestNothingIsWrittenBack:
    """§ 6.1 clause 1: resolved values are VIEWS and never become truth."""

    @pytest.mark.parametrize("kw", [
        {},                                                   # unset vacuum
        {"vacuum": (5.0, 5.0, 0.0)},                          # a broken box
        {"cell": np.eye(3) * 30, "axis_kind": ISOLATED},      # manual regime
    ])
    def test_resolving_and_checking_leaves_the_structure_alone(self, kw):
        s = _mol(**kw)
        before = (s.cell.copy() if s.cell is not None else None,
                  s.cell_origin.copy() if s.cell_origin is not None else None,
                  s.vacuum, s.axis_kind, s.positions.copy())
        cellmod.resolve_and_check(s)
        assert (s.cell is None) == (before[0] is None)
        if s.cell is not None:
            assert np.array_equal(s.cell, before[0])
        assert (s.cell_origin is None) == (before[1] is None)
        assert s.vacuum == before[2]
        assert s.axis_kind == before[3]
        assert np.array_equal(s.positions, before[4]), "coordinates moved"
