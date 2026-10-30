"""The .MD.nc reader — and the trap it exists to avoid.

`docs/model/parse.md` § 6 gives the shape; what needs pinning here is the
BOOKKEEPING, because every mistake in it is silent:

  * a .MD.nc row mixes two steps — ``xa`` is the geometry about to be tried,
    every scalar on the same row describes the geometry just evaluated.  Pair
    them naively and every energy in the trajectory is off by one move, with
    a plausible-looking plot and nothing raised;
  * the file has no row for the input geometry, and the .out repeats its
    final one, so aligning the two by index arithmetic pairs a frame with a
    row it is not;
  * ``xa`` is Bohr while ``volume`` is Ang**3 IN THE SAME FILE, so a blanket
    unit assumption is wrong even when it looks consistent.

What the reader makes of a real history -- the file read, its units, a
frame's geometry and energy paired, the output's frames kept -- is read off a
run made on the road with the real SIESTA, `tests/test_siesta_optimization_run_e2e.py`
(user, 2026-10-06: "when a test need siesta's output why is it
not part of a e2e test?"; `process/testing.md` § 6).  What stays here is the
alignment's own rule, on frames built in memory.
"""
from __future__ import annotations

import numpy as np

from molbuilder.frame import Frame
from molbuilder.parse.engines.siesta_mdnc import align_to_reference
from molbuilder.structure import Structure


def _frames(*bonds: float):
    """Synthetic 2-atom frames, one per bond length."""
    out = []
    for i, b in enumerate(bonds):
        out.append(Frame(
            structure=Structure(elements=["H", "H"],
                                positions=np.array([[0.0, 0.0, 0.0],
                                                    [b, 0.0, 0.0]])),
            step_index=i))
    return out


# ---- alignment --------------------------------------------------- #


def test_alignment_reports_a_miss_rather_than_guessing():
    ref = _frames(1.00, 5.55, 1.20)
    candidate = _frames(1.00, 1.20)
    assert align_to_reference(ref, candidate) == [0, None, 1]


def test_a_repeated_final_geometry_does_not_claim_one_row_twice():
    """A relaxation prints its final geometry again at the end.  Both .out
    frames match the same coordinates, and a naive matcher would map both
    onto the single .MD.nc row -- double-counting the last step."""
    ref = _frames(1.00, 1.20, 1.20)
    candidate = _frames(1.00, 1.20)
    assert align_to_reference(ref, candidate) == [0, 1, None]


def test_alignment_is_empty_when_there_is_nothing_to_align():
    assert align_to_reference([], _frames(1.0)) == []
    assert align_to_reference(_frames(1.0), []) == [None]

