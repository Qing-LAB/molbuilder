"""Which atoms a mode belongs to — ``spectra.results.motion_share_by_element``.

THE BUG THIS ANSWERS.  The mode panel used to describe a mode by the atom with
the largest displacement.  In the benzene-dithiol result that is a hydrogen in
32 of 36 modes, including the 1648 cm⁻¹ ring stretch, where the H's travel 18%
further than the carbons and carry 9% of the motion.  "Furthest" and "carries
the mode" are different questions, and only the second one is worth reading.

So these tests assert the physics, not the code: a heavy atom moving slightly
less than a light one still owns the mode, the answer does not depend on which
of the two stored eigenvector forms you hand in, and the shares of a mode add
up to all of it.
"""
import math

import numpy as np
import pytest

from molbuilder.chemistry import atomic_mass
from molbuilder.spectra.results import motion_share_by_element


# --------------------------------------------------------------------- #
#  The mass table itself                                                #
# --------------------------------------------------------------------- #

class TestAtomicMass:
    """``chemistry.atomic_mass`` is a NAME for ASE's table, not a copy.

    So there is nothing here asserting a mass.  A test that transcribed
    ASE's weights would be the second source of truth the function exists
    to avoid, one directory over -- and it could only restate the table,
    never check it.  Which table we read is a one-line import, verified by
    reading it.

    That the masses are RIGHT is observable where it matters, in
    ``TestMotionShare`` below: real eigenvectors through
    ``motion_share_by_element``, where a wrong table moves the answer.
    What is ours, and is tested here, is the label handling and the
    refusal.
    """

    def test_a_species_label_carries_to_its_element(self):
        """``Au1``/``Au2`` are one element with two labels -- a mass-weighted
        quantity over them needs gold's weight, not a KeyError.

        Contract: ``chemistry.resolve_element``.
        """
        assert atomic_mass("Au1") == atomic_mass("Au")
        assert atomic_mass("Au2") == atomic_mass("Au")

    def test_a_name_we_cannot_look_up_is_a_question_not_a_guess(self):
        """No case folding: `atomic_mass` reads through `resolve_element`, so
        a name that is not a symbol is refused rather than repaired.
        `model/chemistry.md` § 3 owns the rule.
        """
        for typo in ("au", "Xx"):
            with pytest.raises(KeyError) as exc:
                atomic_mass(typo)
            assert "unknown element symbol" in str(exc.value)


# --------------------------------------------------------------------- #
#  The share itself                                                     #
# --------------------------------------------------------------------- #

class TestMotionShare:

    def test_the_heavier_atom_owns_the_mode_even_moving_less(self):
        """THE WHOLE POINT, in the numbers that fooled the old readout.

        Mode 30 of the BDT result: |L| = 1.15 for each of four hydrogens,
        0.98 for each of four carbons.  Hydrogen moves furthest; carbon
        carries the mode.
        """
        elements = ["C", "C", "C", "C", "H", "H", "H", "H"]
        rows = ([[0.98, 0, 0]] * 4) + ([[1.15, 0, 0]] * 4)
        share = motion_share_by_element(elements, rows)

        assert share["C"] > share["H"]
        assert share["C"] == pytest.approx(0.90, abs=0.02)
        assert share["H"] == pytest.approx(0.10, abs=0.02)

    def test_every_element_appears_once_ordered_by_how_much_it_carries(self):
        """The two things a caller can rely on, neither of which is arithmetic.

        `sum(shares) == 1` USED to be asserted here and cannot fail --
        `results.py` returns `w / total` where `total` is that same sum, so
        the claim is division, not behaviour (measured 2026-09-09).  What can
        fail is the element set (an atom silently dropped, or two rows mapped
        to one atom) and the ORDER, which the docstring promises largest-first
        and the Spectra panel renders in the order it is given.
        """
        elements = ["C", "H", "S", "Au"]
        rows = [[0.4, 0.1, 0], [1.0, 0, 0.2], [0.3, 0.3, 0], [0.05, 0, 0.05]]
        share = motion_share_by_element(elements, rows)

        assert set(share) == {"C", "H", "S", "Au"}
        assert list(share) == sorted(share, key=share.get, reverse=True), (
            f"shares must come back largest-first, got {list(share)}")

    def test_either_stored_eigenvector_gives_the_same_answer(self):
        """The two forms differ by one scalar per mode (schema v2), and a
        scalar cancels in a ratio — so the panel may pass whichever pairing
        is on screen without the composition changing under the user."""
        elements = ["C", "C", "H", "H"]
        display = np.array([[0.9, 0.1, 0], [0.2, 0.8, 0],
                            [1.0, 0.3, 0], [0.4, 1.0, 0]])
        canonical = display * 0.1342          # what the emitter's scaling does
        assert (motion_share_by_element(elements, display)
                == pytest.approx(motion_share_by_element(elements, canonical)))

    def test_a_free_atom_basis_maps_rows_onto_the_right_atoms(self):
        """A mode over a partially frozen structure carries one row per FREE
        atom.  Reading row k as atom k would attribute the motion to whatever
        element happens to sit at that index — a silent, plausible wrong
        answer, which is the kind this test exists to stop."""
        elements = ["Au", "Au", "C", "H"]      # atoms 0,1 frozen
        rows = [[1.0, 0, 0], [1.0, 0, 0]]      # equal motion on C and H
        share = motion_share_by_element(elements, rows, atom_idxs=[2, 3])

        assert set(share) == {"C", "H"}, "a frozen gold atom cannot carry motion"
        assert share["C"] / share["H"] == pytest.approx(
            atomic_mass("C") / atomic_mass("H"), rel=1e-9)

    def test_a_mode_that_does_not_move_reports_nothing(self):
        """Rather than dividing by zero.  Not hypothetical: a fully frozen
        selection produces all-zero rows."""
        assert motion_share_by_element(["C", "H"], [[0, 0, 0], [0, 0, 0]]) == {}

    def test_a_mode_that_does_not_match_the_structure_is_refused(self):
        with pytest.raises(ValueError) as exc:
            motion_share_by_element(["C", "H"], [[1, 0, 0]], atom_idxs=[0, 1])
        assert "does not match the structure" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            motion_share_by_element(["C"], [[1, 0, 0]], atom_idxs=[7])
        assert "outside the structure" in str(exc.value)

    def test_rows_must_be_three_vectors(self):
        with pytest.raises(ValueError) as exc:
            motion_share_by_element(["C"], [[1.0, 0.0]])
        assert "(n_atoms, 3)" in str(exc.value)


# --------------------------------------------------------------------- #
#  Against the real result                                              #
# --------------------------------------------------------------------- #

class TestAgainstKnownSpectroscopy:
    """Constructed from textbook assignments rather than from our own output:
    a C–H stretch near 3000 cm⁻¹ is hydrogen motion, a ring stretch near
    1600 cm⁻¹ is carbon motion.  If the share ever stops agreeing with that,
    the number is wrong however cleanly it computes."""

    def test_a_ch_stretch_is_hydrogen_motion(self):
        # C nearly still, H swinging: the classic high-frequency stretch.
        share = motion_share_by_element(
            ["C", "H"], [[0.08, 0, 0], [1.0, 0, 0]])
        assert share["H"] > 0.9
