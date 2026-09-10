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


def _unused_braced(src: str, open_idx: int) -> str:
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


# Seven classes with a docstring and no test method stood here, plus an
# unused `_ADAPTER_PATH`.  They claimed contracts -- bucketed state,
# back-compat aliases, the transition orchestrator, refresh wiring, the
# in-progress filter -- and asserted none of them.  Removed 2026-09-10;
# the live versions are in `test_trajectory_transition_js.py`.

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


