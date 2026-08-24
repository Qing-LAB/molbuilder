"""Merging chained stages must not corrupt either clock.

``_merge_molwatch_trajectories`` concatenates several stage logs into
one payload.  The two time series behave differently under that
concatenation and the difference is the whole point of naming them
apart (docs/model/parse.md § 2a):

* ``wall_clock_s`` is an absolute epoch — every stage's readings are
  already on the same clock, so it concatenates untouched.
* ``elapsed_s`` counts from the start of *a run*, and each stage is a
  separate run whose timer restarts at zero.  Concatenating raw values
  gives a sawtooth — up, back to zero, up again — so each stage is
  offset by the running total of the stages before it (P-T3).

There was no test over this function before 2026-08-24; the sawtooth
shipped unnoticed because the browser derived its "total" as
``last - first``, which happens to look plausible on a sawtooth.
"""

from __future__ import annotations

import pytest

from molbuilder.web.blueprints.watch import _merge_molwatch_trajectories


# Stage 1 runs for 10 s.  Stage 2 starts 90 s later and runs for 30 s.
# Total COMPUTE is 40 s; the 90 s the job spent queued between them is
# not compute and must not be counted as such.
_E0 = 1761396000.0
_STAGE1 = (_E0, _E0 + 10.0)
_STAGE2 = (_E0 + 100.0, _E0 + 130.0)


def _write_stage(tmp_path, name: str, epochs) -> str:
    blocks = []
    for i, ep in enumerate(epochs):
        blocks.append(
            f"==== molwatch step {i} begin ====\n"
            f"step_index: {i}\n"
            f"wall_time: {ep:.3f}\n"
            "n_atoms: 1\n"
            "coordinates (Ang):\n"
            "   H  0.0  0.0  0.0\n"
            f"energy (eV): {-1.0 - i}\n"
            "forces (eV/Ang):\n"
            "   H  0.0  0.0  0.0\n"
            "max_force (eV/Ang): 0.0\n"
            "scf_history begin\n"
            "scf_history end\n"
            f"==== molwatch step {i} end ====\n"
        )
    body = ("# molwatch trajectory log v1\n"
            "# engine: pyscf\n"
            "# created: 2026-04-25T11:00:00\n\n" + "\n".join(blocks))
    p = tmp_path / name
    p.write_text(body)
    return str(p)


@pytest.fixture
def two_stages(tmp_path):
    return [_write_stage(tmp_path, "job-stage1.molwatch.log", _STAGE1),
            _write_stage(tmp_path, "job-stage2.molwatch.log", _STAGE2)]


def test_epoch_series_concatenates_untouched(two_stages):
    """Every stage's epochs are already on the same clock, so the
    merged series is the plain concatenation — no offsetting."""
    merged, stages = _merge_molwatch_trajectories(two_stages)
    assert len(stages) == 2
    assert merged["wall_clock_s"] == [
        pytest.approx(v) for v in (*_STAGE1, *_STAGE2)]


def test_elapsed_series_is_offset_not_concatenated(two_stages):
    """Stage 2's timer restarts at zero, so its values are lifted by
    stage 1's total before being appended."""
    merged, _ = _merge_molwatch_trajectories(two_stages)
    # stage 1 -> [0, 10]; stage 2 -> [0, 30] lifted by 10 -> [10, 40].
    assert merged["elapsed_s"] == [
        pytest.approx(v) for v in (0.0, 10.0, 10.0, 40.0)]


def test_merged_elapsed_never_goes_backwards(two_stages):
    """The property the sawtooth violated, stated directly: time spent
    computing only ever increases as you walk the merged frames."""
    merged, _ = _merge_molwatch_trajectories(two_stages)
    series = [v for v in merged["elapsed_s"] if v is not None]
    assert series == sorted(series), (
        f"elapsed_s went backwards across a stage boundary: {series}")


def test_total_is_compute_time_and_excludes_the_gap(two_stages):
    """40 s of compute across the two stages; the 90 s the job spent
    between them is not compute.  With no wall clock in the file there
    would be no way to know that gap at all — excluding it is the
    honest answer, and it is what the browser now shows as "total"."""
    merged, _ = _merge_molwatch_trajectories(two_stages)
    assert merged["elapsed_s"][-1] == pytest.approx(40.0)
    # Not the 130 s of elapsed wall-clock between first and last frame.
    span = merged["wall_clock_s"][-1] - merged["wall_clock_s"][0]
    assert span == pytest.approx(130.0)
    assert merged["elapsed_s"][-1] < span
