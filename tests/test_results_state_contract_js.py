"""What is LEFT of `results.md` § 4's source pins — two, and why.

This file held 28 tests that read `lib/trajectory/core.js` as text. Over
2026-09-04 twenty-six were replaced by tests that RUN the code
(`test_trajectory_transition_js.py`,
`test_in_progress_frames_stay_out_of_plots.py`), each one mutation-verified
against the defect it claims to catch, and each deletion made only after
its replacement was green.

Two remain, for stated reasons rather than by omission:

* **`test_helper_called_from_loadByPath`** — a wiring assertion. The
  behavioural version means driving `loadByPath`, which is 206 lines
  touching ~35 collaborators including the DOM, Blob and the MolView
  mount. The harness would be larger than the thing it tests and brittle
  with it. The consequence if it broke is also small: a load of an
  already-finished run would settle one poll later instead of at once.
  The same wiring on the POLL side — where the consequence is a finished
  run re-fetched every 15 s for ever — is covered behaviourally in
  `test_trajectory_transition_js.py::test_a_poll_that_finds_the_run_
  ended_settles_it`.

* **`test_applyNewData_routes_writes_through_transition`** — not a
  spelling pin but a NEGATIVE LINT (`process/testing.md` § 6): it
  quantifies over a function body for a family of forbidden writes
  (`state.mtime =`, `state.data =`, …) that would bypass § 4's single
  entry point. It fails when an offender appears, which is the shape
  § 6 sanctions.

**What the deleted twenty-six taught, measured rather than assumed.**
Breaking the two-tick buffer while leaving `>= 2` alive in a comment
passed all 28. Deleting the Refresh registration passed 155 tests. And
`test_merge_propagates_in_progress` was guarding lines that cannot
execute at all — see the note in
`test_in_progress_frames_stay_out_of_plots.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_STATIC = (Path(__file__).resolve().parent.parent
           / "molbuilder" / "web" / "static")
_LIB = _STATIC / "lib"


# --------------------------------------------------------------------- #
#  Trajectory: bucketed state shape                                     #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def core_body():
    return (_LIB / "trajectory" / "core.js").read_text()


def _braced(src: str, open_idx: int) -> str:
    """The object literal that starts at ``src[open_idx] == '{'``, brace
    to brace -- so a pin over a literal covers the whole literal however
    long it grows, rather than a fixed character count that a new field
    can push a bucket out of."""
    assert src[open_idx] == "{", "not an opening brace"
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx: i + 1]
    raise AssertionError("unbalanced braces from the state literal")


class TestBucketedStateShape:
    """``state`` carries five named buckets (fileState, viewState,
    uiPrefs, lifecycle, derived) + a ``machine`` field.  The contract
    § 3 data-buckets section requires the partition; later tests
    assume the buckets exist."""


class TestBackcompatAliases:
    """Existing render code reads/writes flat ``state.X`` (mtime,
    data, currentFrame, ...).  PR 2 keeps that surface working via
    Object.defineProperty getter/setter aliases that route to the
    bucketed canonical home.  Pin the alias wiring so a refactor that
    drops it doesn't silently break ~3000 lines of legacy reads."""


# --------------------------------------------------------------------- #
#  Trajectory: transition() orchestrator                                #
# --------------------------------------------------------------------- #


class TestTransitionOrchestrator:
    """``transition(target, payload)`` is the SINGLE entry-point for
    state-machine transitions.  Contract § 2 forbids direct mutation
    of fileState / lifecycle / derived outside this function."""


class TestSettlePostLoad:
    """The 2-consecutive-ticks WATCHING -> LOADED buffer
    (`results.md` § 4.1) lives in ``_settlePostLoad()``.  Called from both
    loadByPath and pollOnce; reads run_state, decides which
    transition to invoke."""


    def test_helper_called_from_loadByPath(self, core_body):
        """loadByPath's success path MUST call _settlePostLoad after
        applyNewData -- otherwise the freshly-loaded file's run_state
        is never inspected and we never transition out of LOADING."""
        m = re.search(
            r"async\s+function\s+loadByPath\s*\([^)]*\)\s*\{(.+?)\n\s{4}\}",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "_settlePostLoad" in body, (
            "loadByPath no longer calls _settlePostLoad after "
            "applyNewData.  state.machine stays 'LOADING' forever; "
            "the poll timer never starts.")


    # RETIRED 2026-09-03 — the third grep to fail on the day the code it
    # describes was corrected, for the same reason as its two siblings in
    # the spectra file.  Its regex was `APPLY ... (.+?) return;`,
    # NON-GREEDY, so the guard clause added to drop an answer meant for a
    # file the user has moved off became the first `return` and the capture
    # never reached the writes below it.  The assertion message read
    # "transition('APPLY') no longer writes state.fileState.mtime.  The
    # atomic-replacement semantics is broken" -- while the change it was
    # reporting on is the one that MADE the replacement atomic.
    #
    # `results.md` § 4 had said "replaced atomically" since the state
    # machine landed.  It was not: the noNewContent branch passed
    # {mtime, data} and left path standing "because the file identity
    # didn't change", an assumption nothing checked.  APPLY now requires
    # the path -- which every server reply already carries as `r.path` --
    # and drops a reply whose file is not the one on screen.
    def test_applyNewData_routes_writes_through_transition(self, core_body):
        """applyNewData's two write blocks (noNewContent + full
        rebuild) MUST both go through transition('APPLY').  Direct
        ``state.mtime = ...`` / ``state.data = ...`` writes are
        forbidden -- the alias bridge would route them to fileState
        but bypass the single-entry-point contract."""
        m = re.search(
            r"function\s+applyNewData\s*\(\s*r\s*\)\s*\{(.+?)\n\s{4}\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "applyNewData function not found"
        body = m.group(1)
        # No direct legacy-alias writes of fileState fields.
        forbidden = [
            r"\bstate\.mtime\s*=",
            r"\bstate\.data\s*=",
            r"\bstate\.format\s*=",
            r"\bstate\.label\s*=",
            r"\bstate\.path\s*=",
        ]
        for pat in forbidden:
            assert not re.search(pat, body), (
                f"applyNewData still has a direct fileState write "
                f"matching ``{pat}``.  PR 2.3 routed these through "
                f"transition('APPLY', ...); the regression brings "
                f"back the contract § 2 'sole-writer' violation.")
        # AT LEAST two transition('APPLY', ...) call sites in the
        # function body (one per write block).
        apply_calls = re.findall(
            r"transition\s*\(\s*[\"']APPLY[\"']", body
        )
        assert len(apply_calls) >= 2, (
            f"applyNewData has {len(apply_calls)} transition('APPLY') "
            f"call(s).  Expected >= 2 (noNewContent path + full-"
            f"rebuild path).  Either a write block was removed or a "
            f"direct write was reintroduced.")


class TestRefreshListenerWiredOnce:
    """PR 2.1 audit follow-up: the EVENT_REFRESH_REQUESTED listener
    is wired ONCE at mount via _wireRefreshListener(), not re-wired
    per-load by startPolling().  Pre-fix the listener piled up on
    every load and only tore down on dispose."""


# --------------------------------------------------------------------- #
#  Refresh = file-switch (contract § 5)                                 #
# --------------------------------------------------------------------- #


class TestRefreshIsFileSwitch:
    """The Refresh button MUST route through loadByPath() (which
    delegates to transition('LOADING')), NOT pollOnce() directly.
    Pre-PR-2 the inline pollOnce() left scfPollHistory + firstFit +
    fileState entirely untouched -- the "half-refresh" bug class."""


# --------------------------------------------------------------------- #
#  Invariant 1: file-identity guard at fetch resolution                 #
# --------------------------------------------------------------------- #


    # RETIRED 2026-09-04 with the counter they describe.
    #
    # `fetchSeq` was a sequence number: bumped on LOADING, snapshotted
    # before each fetch, re-checked after, to notice that a response had
    # arrived for a file the user had moved off.  It existed because the
    # filename and the data were written in two separate steps, so an
    # answer could land under the wrong name.
    #
    # Since 2026-09-03 `transition("APPLY", ...)` REQUIRES the path and
    # drops a payload whose file is not the one on screen, so the answer
    # carries its own identity.  The remaining guards -- the status banner,
    # the consecutive-error count, `stopWatch` -- now ask the same question
    # of the same fact: `path !== state.fileState.path`.
    #
    # The replacement is STRICTLY STRONGER, which is why this is a deletion
    # and not a trade.  `transition("IDLE")` never bumped the counter, so a
    # fetch in flight when the inspector was disposed passed the old guard
    # and fails the new one (`fileState.path` is null by then).  And
    # `signal.aborted`, already checked beside it, is the only thing that
    # can tell two loads of the SAME file apart -- which a counter could and
    # a path cannot, so both halves are kept.
    #
    # These seven pinned the MECHANISM by name: "LOADING increments
    # fetchSeq", "loadByPath captures mySeq", "the guard compares them".
    # None could survive the mechanism being replaced by a better one, and
    # none was checking the property -- that a late answer cannot be
    # painted under the wrong file -- which is pinned behaviourally by
    # test_spectra_from_a_real_run_e2e.py and
    # test_trajectory_from_a_real_run_e2e.py, both mutation-verified
    # against APPLY's path requirement.
# --------------------------------------------------------------------- #
#  Invariant 2: in-progress frame filter                                #
# --------------------------------------------------------------------- #


class TestInProgressFilter:
    """Server emits ``data.in_progress[i]`` per-frame bool.  JS
    ``plottableFrames(data)`` MUST filter those frames from plot
    trace y-arrays -- the parser's placeholder energy is a
    ``step_initial_etot`` fallback, not a real measurement."""


# --------------------------------------------------------------------- #
#  Wire-format: in_progress array landed                                #
# --------------------------------------------------------------------- #


class TestWireFormatInProgress:
    """The server-side trajectory_result_to_legacy_dict adapter MUST
    emit ``in_progress`` in the output dict.  Empty list when no
    frame is in-progress.  Module rehomed in parse-module H2.adapter
    (2026-06-20): legacy ``molbuilder/parsers/__init__.py`` →
    ``molbuilder/parse/engines/_helpers.py``."""

    _ADAPTER_PATH = (Path(__file__).resolve().parent.parent
                     / "molbuilder" / "parse" / "engines"
                     / "_helpers.py")


