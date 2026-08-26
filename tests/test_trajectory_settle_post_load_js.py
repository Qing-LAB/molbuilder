"""The trajectory viewer stops polling when the run stops -- including
when it stopped by CRASHING.

The bug this guards (#12, fixed 2026-08-04)
===========================================

``_settlePostLoad`` decides, after every poll, whether the viewer keeps
watching or settles.  It used to ask ``run_state === "errored"``.  Nothing
has ever emitted that word, so a crashed run matched neither the error
branch nor the finished branch, fell into the "still going" tail, and the
viewer re-polled ``/api/watch/data`` every 15 seconds until the user left
the tab.

It was invisible because the run-state BADGE, a few hundred lines down in
the same file, tested the real word and got it right.  The user saw
**Stopped** and a timer that never stopped.

The vocabulary changed under it (2026-08-25)
============================================

``model/parse.md`` § 2b replaced ``"ongoing" | "finished" | "error" |
"unknown"`` with ``running | ended | stopped | out_of_memory | unknown``,
because *how a run ended* is a fact about the process and *whether the
science converged* is a separate one -- conflating them made the Results
tab call six healthy benchmark trials failures.

That rename is the same shape of bug as #12, one level up: a consumer
holding a private copy of the vocabulary keeps compiling and silently
stops matching.  It struck twice in one day.  ``cli.py``'s ``watch tail``
kept its own ``("finished", "error")`` tuple and, when neither word could
occur any more, polled a finished run forever -- the test that should have
caught it had no time bound, so it hung the whole suite for eleven hours
instead of failing.  So ``TestTheRetiredWordsStayRetired`` below now drives
every retired spelling through this function: if one is ever re-introduced
as terminal, or a future rename leaves a copy behind, it fails here.

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


def _node_or_skip() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    return node


def _run_state_values() -> list[str]:
    """``RUN_STATE``'s values, read by EVALUATING the real constant.

    Lifted and executed rather than pattern-matched: a test that parses
    JavaScript with string surgery breaks on a reformat and, worse, can be
    fooled by one.  Node already knows how to read the file's own syntax.
    """
    node = _node_or_skip()
    src = MODULE.read_text()
    const = _slice(src, "const RUN_STATE = Object.freeze", "});") + "});"
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e",
         const + "\nconsole.log(JSON.stringify(Object.values(RUN_STATE)));"],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _settle(run_state, *, machine="LOADING", finished_ticks=0):
    """Run ``_settlePostLoad`` for one ``run_state`` and report what it did.

    Returns ``{"transitions": [...], "finishedTicks": N}``.  ``run_state``
    of ``None`` stands for a static file, which carries no run state at all.
    """
    node = _node_or_skip()

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

    @pytest.mark.parametrize("run_state", ["stopped", "out_of_memory"])
    def test_a_run_that_did_not_reach_its_end_goes_straight_to_ERROR(self, run_state):
        """Both ways of not finishing are terminal, and neither needs a
        confirmation tick -- a crash does not un-crash on the next poll.

        ``out_of_memory`` is a separate state rather than a flavour of
        ``stopped`` (§ 2b P-S1) because it is the most actionable cause,
        and it must reach the same branch: a run the kernel killed is no
        less over than one that aborted itself.
        """
        out = _settle(run_state)
        assert out["transitions"] == ["ERROR"]


class TestTheRetiredWordsStayRetired:

    #: Every spelling that has ever been wrong here, and why.
    RETIRED = {
        "errored":   "#12's invented word -- never emitted by anything",
        "failed":    "belongs to the STATUS envelope "
                     "(`parse/dirs/job.py::_build_status`), never reaches this file",
        "error":     "retired by § 2b; `stopped` and `out_of_memory` replaced it",
        "finished":  "retired by § 2b; `ended` replaced it",
        "ongoing":   "retired by § 2b; `running` replaced it",
    }

    @pytest.mark.parametrize("word", sorted(RETIRED))
    def test_a_retired_spelling_is_never_terminal(self, word):
        """None of these may settle the viewer.

        Two of them once did the opposite damage -- `cli.py`'s `watch tail`
        went on polling a finished run because it still compared against
        `finished`/`error` after the rename.  A word that is not in the
        vocabulary must produce the honest answer, "I do not know that this
        is over", and that answer is: keep watching.
        """
        out = _settle(word)
        assert out["transitions"] == ["WATCHING"], (
            f"{word!r} must NOT be treated as terminal -- "
            f"{self.RETIRED[word]}")

    def test_the_live_vocabulary_and_the_retired_list_do_not_overlap(self):
        """A word cannot be both retired and current.

        Without this, re-introducing a retired name into ``RUN_STATE``
        would leave the case above asserting that a LIVE state is not
        terminal -- a green test pinning the opposite of the contract.
        """
        live = set(_run_state_values())
        assert live == {"running", "ended", "stopped", "out_of_memory", "unknown"}, (
            "`model/parse.md` § 2b's vocabulary is closed; core.js disagrees")
        assert not (live & set(self.RETIRED)), (
            f"re-introduced as live: {sorted(live & set(self.RETIRED))}")


class TestTheOtherStatesAreUnchanged:

    def test_ended_needs_two_consecutive_ticks(self):
        """A single ``ended`` tick may lie while the parser flushes, so
        the viewer stays in WATCHING until the second one."""
        first = _settle("ended", finished_ticks=0)
        assert first["transitions"] == ["WATCHING"]
        assert first["finishedTicks"] == 1

        second = _settle("ended", machine="WATCHING", finished_ticks=1)
        assert second["transitions"] == ["LOADED"]

    def test_running_keeps_watching_and_breaks_the_ended_streak(self):
        out = _settle("running", finished_ticks=1)
        assert out["transitions"] == ["WATCHING"]
        assert out["finishedTicks"] == 0, \
            "the two-tick buffer counts CONSECUTIVE ended ticks only"

    def test_unknown_is_not_evidence_of_ending_and_keeps_watching(self):
        """§ 2b P-S1: ``unknown`` is "no evidence either way".  Reading it
        as terminal stops the viewer watching a run that is still alive --
        the mirror of the bug at the top of this file."""
        out = _settle("unknown", finished_ticks=1)
        assert out["transitions"] == ["WATCHING"]
        assert out["finishedTicks"] == 0

    def test_a_static_file_carries_no_run_state_and_keeps_watching(self):
        """No ``run_state`` at all is the static-file case; the caller stops
        the timer by another route, so this must not claim terminal."""
        out = _settle(None)
        assert out["transitions"] == ["WATCHING"]
