"""A mid-write frame is LISTED but never PLOTTED — the whole chain.

`results.md` § 4: partial frames are kept out of the plots. That needs
three things to agree:

1. the adapter must EMIT a per-frame `in_progress` array, or the browser
   has nothing to filter on;
2. it must COLLAPSE that array to `[]` when nothing is partial, matching
   the `max_forces_constrained` convention — a browser reading a long
   array of `false` would filter every frame through a branch that can
   never fire;
3. the array must stay aligned 1:1 with the frames, or a partial flag
   lands on the wrong frame.

These call the real adapter and assert on what comes
back, so a rename changes nothing and a broken alignment fails.
"""
from __future__ import annotations

from pathlib import Path

from molbuilder.frame import Frame, Trajectory
from molbuilder.parse.engines._helpers import (
    trajectory_result_to_legacy_dict as to_legacy,
    wrap_trajectory,
)
from molbuilder.structure import Structure

import numpy as np


def _frame(step, *, partial):
    return Frame(
        structure=Structure(elements=["H"], positions=np.array([[0.0, 0.0, 0.0]]),
                            vacuum=(10.0, 10.0, 10.0)),
        step_index=step, energy=-1.0 * step, forces=None, max_force=None,
        max_force_constrained=None, lattice=None, scf_history=[],
        wall_clock_s=None, elapsed_s=None, in_progress=partial)


def _legacy(*partials):
    traj = Trajectory(source_format="molwatch",
                      frames=[_frame(i, partial=p) for i, p in enumerate(partials)],
                      lattice=None, run_state="running", scf_converged=None,
                      error_message=None, runtime_info={}, parse_warnings=[])
    return to_legacy(wrap_trajectory(traj, "molwatch", Path("/tmp/x.molwatch.log")))


def test_the_adapter_reports_which_frames_are_mid_write():
    """One partial frame among three, and the browser can tell which."""
    out = _legacy(False, True, False)
    assert out["in_progress"] == [False, True, False], (
        f"the adapter must say WHICH frame is mid-write; it emitted "
        f"{out.get('in_progress')!r}. Without it the viewer has nothing to "
        f"filter on and plots a half-written geometry.")
    assert len(out["frames"]) == 3, "all three frames are still LISTED"


def test_it_collapses_to_empty_when_nothing_is_partial():
    """`[]` means *nothing to filter*, matching `max_forces_constrained`.

    An array of three `false` would be honest too, and is rejected on
    purpose: every reader would then walk a branch that can never fire.
    """
    assert _legacy(False, False, False)["in_progress"] == []


def test_a_run_that_is_entirely_mid_write_still_says_so():
    """The collapse is *no partials*, not *all the same*."""
    assert _legacy(True, True)["in_progress"] == [True, True]
