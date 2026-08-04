"""The trajectory viewer stops polling when the run stops -- including
when it stopped by CRASHING.

The bug this guards (#12, fixed 2026-08-04)
===========================================

``_settlePostLoad`` decides, after every poll, whether the viewer keeps
watching or settles.  It used to ask ``run_state === "errored"``.  Nothing
has ever emitted that word: the vocabulary is the parser's, fixed at
``parse/engines/_helpers.py:181`` as
``"ongoing" | "finished" | "error" | "unknown"`` and passed through the
watch endpoint unchanged.  So a crashed run matched neither the error
branch nor the finished branch, fell into the "still going" tail, and the
viewer re-polled ``/api/watch/data`` every 15 seconds until the user left
the tab.

It was invisible because the run-state BADGE, a few hundred lines down in
the same file, tested ``"error"`` and got it right.  The user saw
**Stopped** and a timer that never stopped.

Why these tests are not e2e: reaching the crashed-run branch through a
browser needs a run directory whose engine output ends in a real failure,
polled live.  The decision itself is a pure function of ``run_state``, so
it is tested here directly -- one run through Node per state, with a fake
``state`` and a recording ``transition``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _slice(src: str, start_marker: str, end_marker: str) -> str:
    ix = src.index(start_marker)
    return src[ix:src.index(end_marker, ix)].rstrip()


def _settle(run_state, *, machine="LOADING", finished_ticks=0):
    """Run ``_settlePostLoad`` for one ``run_state`` and report what it did.

    Returns ``{"transitions": [...], "finishedTicks": N}``.  ``run_state``
    of ``None`` stands for a static file, which carries no run state at all.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    src = MODULE.read_text()
    # Both halves are lifted from the real source, so a rename or a
    # re-spelling breaks this test instead of sliding past it.
    run_state_const = _slice(src, "const RUN_STATE = Object.freeze", "});") + "});"
    fn_source = _slice(src, "function _settlePostLoad", "function plottableFrames")

    harness = f"""
        {run_state_const}
        const _seen = [];
        function transition(name) {{ _seen.push(name); state.machine = name; }}
        const state = {{
            machine: {json.dumps(machine)},
            lifecycle: {{ finishedTicks: {finished_ticks} }},
            fileState: {{ data: {json.dumps(
                None if run_state is None else {"run_state": run_state})} }},
        }};
        {fn_source}
        _settlePostLoad();
        console.log(JSON.stringify({{
            transitions:   _seen,
            finishedTicks: state.lifecycle.finishedTicks,
        }}));
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", harness],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestACrashedRunSettles:

    def test_error_goes_straight_to_ERROR(self):
        """``error`` is terminal and needs no confirmation tick -- a crash
        does not un-crash on the next poll."""
        out = _settle("error")
        assert out["transitions"] == ["ERROR"]

    def test_the_word_the_backend_actually_sends_is_the_word_tested(self):
        """The regression itself: ``errored`` is not a run state.

        If someone re-introduces it -- or follows task #12's original text
        and writes ``failed``, which belongs to the STATUS envelope
        (``parse/dirs/job.py::_build_status``) and never reaches this file
        -- the crashed run silently goes back to polling forever.
        """
        for not_a_run_state in ("errored", "failed"):
            out = _settle(not_a_run_state)
            assert out["transitions"] == ["WATCHING"], (
                f"{not_a_run_state!r} must NOT be treated as terminal; the "
                "only terminal error state is 'error'")


class TestTheOtherStatesAreUnchanged:

    def test_finished_needs_two_consecutive_ticks(self):
        """A single ``finished`` tick may lie while the parser flushes, so
        the viewer stays in WATCHING until the second one."""
        first = _settle("finished", finished_ticks=0)
        assert first["transitions"] == ["WATCHING"]
        assert first["finishedTicks"] == 1

        second = _settle("finished", machine="WATCHING", finished_ticks=1)
        assert second["transitions"] == ["LOADED"]

    def test_ongoing_keeps_watching_and_breaks_the_finished_streak(self):
        out = _settle("ongoing", finished_ticks=1)
        assert out["transitions"] == ["WATCHING"]
        assert out["finishedTicks"] == 0, \
            "the two-tick buffer counts CONSECUTIVE finished ticks only"

    def test_a_static_file_carries_no_run_state_and_keeps_watching(self):
        """No ``run_state`` at all is the static-file case; the caller stops
        the timer by another route, so this must not claim terminal."""
        out = _settle(None)
        assert out["transitions"] == ["WATCHING"]
