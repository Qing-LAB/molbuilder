"""L1 tests for ``_attach_iter_walltime`` -- the mtime-delta per-iter
wall-time computation that powers the Results-tab "~Xs/iter (measured)"
annotation.

Why mtime is the right clock source for SIESTA (recap):

  * SIESTA emits the ``timer: IterSCF`` line EXACTLY ONCE per run
    (after iter 1 of step 1) and a full ``>>> timer`` block at end-of-
    run -- no per-iter timing mid-run.
  * The .out's filesystem mtime DOES advance on every flush, which
    matches "the engine just finished writing the latest SCF cycle".
  * Mtime survives browser reload and is engine-agnostic.

Algorithm exercised here:

  * Find the latest non-empty SCF step in ``scf_history``.
  * Look BACKWARDS in the sample buffer for the most recent entry
    with the same ``step_idx`` (cross-step deltas conflate inter-step
    bookkeeping with per-iter cost).
  * per-iter = (mtime_now - mtime_prev) / (iters_now - iters_prev).
    Naturally averages when the poll cadence missed > 1 iter.
  * Cap the buffer at 16 entries so it stays bounded.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from molbuilder.web.blueprints.watch import (
    _ITER_WALLTIME_BUFFER_CAP,
    _attach_iter_walltime,
)


def _data(scf_history: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Build a minimal new_data dict with the only field the helper
    actually inspects (``scf_history``)."""
    return {"scf_history": scf_history}


def _iters(n: int) -> List[Dict[str, Any]]:
    return [{"cycle": i + 1, "energy": -800000.0 - i} for i in range(n)]


def test_first_call_records_sample_but_emits_no_per_iter():
    """No prior sample -> nothing to compute.  The buffer gets the
    seed sample so subsequent calls have something to delta against."""
    samples: List[Dict[str, Any]] = []
    data = _data([_iters(3)])
    _attach_iter_walltime(data, mtime=1000.0, samples=samples)
    assert "wall_time_per_iter_s" not in data
    assert len(samples) == 1
    assert samples[0] == {"mtime": 1000.0, "step_idx": 0, "iters_in_step": 3}


def test_two_calls_same_step_one_iter_delta_emits_per_iter():
    """Canonical case: poll 1 sees 3 iters, poll 2 sees 4 iters,
    Delta-mtime = 16.5 s -> per-iter = 16.5 s."""
    samples: List[Dict[str, Any]] = []
    _attach_iter_walltime(_data([_iters(3)]), mtime=1000.0, samples=samples)
    data2 = _data([_iters(4)])
    _attach_iter_walltime(data2, mtime=1016.5, samples=samples)
    assert data2["wall_time_per_iter_s"] == pytest.approx(16.5)
    win = data2["wall_time_per_iter_window"]
    assert win["iters"] == 1
    assert win["seconds"] == pytest.approx(16.5)
    assert win["step_idx"] == 0


def test_two_calls_same_step_multi_iter_delta_averages():
    """Poll cadence missed an iter -- delta = 2 iters in 33 s ->
    per-iter = 16.5 s.  Division NATURALLY averages, no special case."""
    samples: List[Dict[str, Any]] = []
    _attach_iter_walltime(_data([_iters(3)]), mtime=1000.0, samples=samples)
    data2 = _data([_iters(5)])
    _attach_iter_walltime(data2, mtime=1033.0, samples=samples)
    assert data2["wall_time_per_iter_s"] == pytest.approx(16.5)
    assert data2["wall_time_per_iter_window"]["iters"] == 2


def test_step_boundary_does_not_emit_per_iter():
    """When step advances, the iter counter resets to 1.  Crossing
    that boundary would compute a bogus negative or absurdly small
    delta.  Helper MUST skip the calculation in that case."""
    samples: List[Dict[str, Any]] = []
    # Step 0 with 10 iters at t=1000
    _attach_iter_walltime(_data([_iters(10)]), mtime=1000.0, samples=samples)
    # Step 1 with 2 iters at t=1050
    data2 = _data([_iters(10), _iters(2)])
    _attach_iter_walltime(data2, mtime=1050.0, samples=samples)
    assert "wall_time_per_iter_s" not in data2, (
        "Cross-step delta would mix DM-extrapolation cost into per-iter; "
        "must skip until we have two same-step samples."
    )
    # But a THIRD call in the new step should now have a same-step
    # prior to delta against.
    data3 = _data([_iters(10), _iters(4)])
    _attach_iter_walltime(data3, mtime=1083.0, samples=samples)
    assert data3["wall_time_per_iter_s"] == pytest.approx(16.5)
    assert data3["wall_time_per_iter_window"]["step_idx"] == 1


def test_same_mtime_no_emit():
    """mtime equal == no new data == no measurement.  Don't divide
    by zero or treat a stale poll as evidence of a fast iter."""
    samples: List[Dict[str, Any]] = []
    _attach_iter_walltime(_data([_iters(3)]), mtime=1000.0, samples=samples)
    data2 = _data([_iters(3)])
    _attach_iter_walltime(data2, mtime=1000.0, samples=samples)
    assert "wall_time_per_iter_s" not in data2


def test_no_scf_history_silently_ignored():
    """A trajectory with no SCF data (e.g. pure xyz, geomeTRIC's
    _geom_optim.xyz) must not crash the helper."""
    samples: List[Dict[str, Any]] = []
    data = {"scf_history": []}
    _attach_iter_walltime(data, mtime=1000.0, samples=samples)
    assert "wall_time_per_iter_s" not in data
    # And the buffer is NOT polluted with a step_idx=-1 phantom
    # (the helper short-circuits before recording).
    assert samples == []


def test_buffer_capped_at_max():
    """Bounded memory: keep at most _ITER_WALLTIME_BUFFER_CAP samples.
    Capping at the OLDEST end (FIFO) preserves the freshest
    measurements, which is what STAGE 2 actually consumes."""
    samples: List[Dict[str, Any]] = []
    # Fill PAST the cap.
    for i in range(_ITER_WALLTIME_BUFFER_CAP + 5):
        _attach_iter_walltime(_data([_iters(i + 1)]),
                              mtime=1000.0 + 10.0 * i,
                              samples=samples)
    assert len(samples) == _ITER_WALLTIME_BUFFER_CAP
    # The OLDEST entries got dropped, not the newest.
    assert samples[-1]["iters_in_step"] == _ITER_WALLTIME_BUFFER_CAP + 5


def test_recent_match_preferred_over_older_match():
    """When the buffer has multiple same-step priors, use the MOST
    RECENT (smallest dt, freshest measurement)."""
    samples: List[Dict[str, Any]] = []
    _attach_iter_walltime(_data([_iters(1)]), mtime=1000.0, samples=samples)
    _attach_iter_walltime(_data([_iters(2)]), mtime=1100.0, samples=samples)
    _attach_iter_walltime(_data([_iters(3)]), mtime=1110.0, samples=samples)
    data4 = _data([_iters(4)])
    _attach_iter_walltime(data4, mtime=1120.0, samples=samples)
    # Most recent prior (iters=3 @ t=1110) -> dt=10, di=1 -> 10s
    # NOT iters=1 @ t=1000 -> dt=120, di=3 -> 40s
    assert data4["wall_time_per_iter_s"] == pytest.approx(10.0)
