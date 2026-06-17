"""Pin the load-bearing invariants from
docs/protocols/results-state-contract.md for the trajectory inspector
(PR 2 of the migration).

What this file covers
=====================

The contract structures the inspector's state into five disjoint
buckets with a single ``transition()`` orchestrator and three
invariants (file-identity guard, in-progress frame filter,
render-with-snapshot).  PR 2 lands the trajectory side; PR 3 will
mirror it onto spectra.

Each test below pins a different load-bearing property.  A future
refactor that "simplifies" any of them re-introduces the bug class
the contract was written to prevent.

Out of scope for PR 2 (deferred to a future commit per the contract
§ 10 migration plan):
  * Snapshot-signature conversion of every render function.
  * Plotly call-ladder optimization (.restyle / .relayout / .extendTraces).
  * Spectra-side state-machine refactor (PR 3).
  * Parser cache LRU + freshness gate (PR 4).
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


class TestBucketedStateShape:
    """``state`` carries five named buckets (fileState, viewState,
    uiPrefs, lifecycle, derived) + a ``machine`` field.  The contract
    § 3 data-buckets section requires the partition; later tests
    assume the buckets exist."""

    def test_state_has_machine_field(self, core_body):
        assert re.search(
            r"const\s+state\s*=\s*\{[^}]*machine\s*:\s*[\"']IDLE[\"']",
            core_body, re.DOTALL,
        ), ("trajectory/core.js state object no longer declares the "
            "``machine`` field initialized to 'IDLE'.  The state "
            "machine has no resting state; the contract § 2 transitions "
            "table has no starting point.")

    @pytest.mark.parametrize("bucket", [
        "fileState", "viewState", "uiPrefs", "lifecycle", "derived",
    ])
    def test_state_carries_each_bucket(self, core_body, bucket):
        """Each of the five buckets must appear as a top-level key
        of the ``state`` literal.  The state object literal is
        ~80 lines so a non-greedy [^}]* won't span it; use a
        wider window anchored at ``const state = {`` and look
        within the first ~3000 chars."""
        m = re.search(r"const\s+state\s*=\s*\{", core_body)
        assert m is not None, "state literal not found"
        # Look within the next 4000 chars (the state object is ~80
        # lines of ~50 chars each = ~4000).
        window = core_body[m.end(): m.end() + 4000]
        assert re.search(
            r"^\s+" + bucket + r"\s*:\s*\{", window, re.MULTILINE,
        ), (f"trajectory/core.js state object no longer carries the "
            f"``{bucket}`` bucket.  Contract § 3 requires the five-"
            f"bucket partition; collapsing the buckets re-introduces "
            f"the field-drift bug class.")


class TestBackcompatAliases:
    """Existing render code reads/writes flat ``state.X`` (mtime,
    data, currentFrame, ...).  PR 2 keeps that surface working via
    Object.defineProperty getter/setter aliases that route to the
    bucketed canonical home.  Pin the alias wiring so a refactor that
    drops it doesn't silently break ~3000 lines of legacy reads."""

    def test_alias_helper_present(self, core_body):
        assert re.search(
            r"function\s+_wireBackcompatAliases",
            core_body,
        ), ("trajectory/core.js no longer defines the "
            "_wireBackcompatAliases IIFE that bridges flat "
            "``state.X`` reads to the bucketed canonical home.  "
            "Legacy render code breaks.")

    @pytest.mark.parametrize("flat,bucket", [
        ("mtime",          "fileState"),
        ("data",           "fileState"),
        ("path",           "fileState"),
        ("format",         "fileState"),
        ("label",          "fileState"),
        ("currentFrame",   "viewState"),
        ("firstFit",       "viewState"),
        ("pollTimer",      "lifecycle"),
        ("pollInFlight",   "lifecycle"),
        ("loadAbort",      "lifecycle"),
        ("pollAbort",      "lifecycle"),
        ("scfPollHistory", "derived"),
    ])
    def test_each_legacy_field_aliased(self, core_body, flat, bucket):
        """Each legacy flat field name MUST be aliased to a specific
        bucket.  Pin the (flat -> bucket) mapping per contract § 7
        trajectory mapping table."""
        assert re.search(
            r"alias\s*\(\s*[\"']" + flat + r"[\"']\s*,\s*[\"']"
            + bucket + r"[\"']\s*\)",
            core_body,
        ), (f"trajectory/core.js no longer aliases ``state.{flat}`` to "
            f"``state.{bucket}.{flat}``.  Field drift returns; "
            f"contract § 7 mapping is broken.")


# --------------------------------------------------------------------- #
#  Trajectory: transition() orchestrator                                #
# --------------------------------------------------------------------- #


class TestTransitionOrchestrator:
    """``transition(target, payload)`` is the SINGLE entry-point for
    state-machine transitions.  Contract § 2 forbids direct mutation
    of fileState / lifecycle / derived outside this function."""

    def test_transition_function_exists(self, core_body):
        assert re.search(
            r"function\s+transition\s*\(\s*target\s*,\s*payload\s*\)",
            core_body,
        ), ("trajectory/core.js no longer defines the transition() "
            "orchestrator.  Direct bucket mutation re-introduces the "
            "reset-matrix-drift bug class.")

    def test_transition_loading_clears_derived(self, core_body):
        """The LOADING branch MUST clear ``state.derived.scfPollHistory``.
        Pre-PR-2 the Refresh button left this buffer with ~32 stale
        samples carrying bogus per-iter time estimates."""
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']LOADING[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None, "LOADING branch missing"
        body = m.group(1)
        assert "scfPollHistory" in body and "length = 0" in body, (
            "trajectory/core.js transition('LOADING') no longer "
            "clears state.derived.scfPollHistory.  The Refresh "
            "button half-refresh bug returns -- per-iter time "
            "estimates carry stale samples for ~32 polls.")

    def test_transition_loading_bumps_fetchSeq(self, core_body):
        """fetchSeq MUST be incremented on every LOADING transition.
        Invariant 1 (file-identity guard) depends on it."""
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']LOADING[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert re.search(r"fetchSeq\s*\+\+", body), (
            "trajectory/core.js transition('LOADING') no longer "
            "increments lifecycle.fetchSeq.  Invariant 1 file-"
            "identity guard cannot distinguish in-flight responses "
            "of the new vs prior file.")

    def test_transition_loading_aborts_controllers(self, core_body):
        """Both loadAbort and pollAbort MUST be aborted in the
        LOADING branch -- prevents stale-response races (audit § 1
        late-response bug)."""
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']LOADING[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "loadAbort" in body and "abort()" in body, (
            "transition('LOADING') no longer aborts loadAbort.")
        assert "pollAbort" in body, (
            "transition('LOADING') no longer aborts pollAbort.")

    def test_transition_idle_clears_filestate(self, core_body):
        """IDLE branch MUST null out fileState fields -- prevents the
        ``dispose leaks state.data`` bug (audit § 1)."""
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']IDLE[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None, "IDLE branch missing"
        body = m.group(1)
        assert "fileState.data" in body and "= null" in body, (
            "trajectory/core.js transition('IDLE') no longer nulls "
            "fileState.data.  Stale frames leak across remounts.")
        assert "scfPollHistory" in body, (
            "trajectory/core.js transition('IDLE') no longer clears "
            "scfPollHistory.")


# --------------------------------------------------------------------- #
#  Refresh = file-switch (contract § 5)                                 #
# --------------------------------------------------------------------- #


class TestRefreshIsFileSwitch:
    """The Refresh button MUST route through loadByPath() (which
    delegates to transition('LOADING')), NOT pollOnce() directly.
    Pre-PR-2 the inline pollOnce() left scfPollHistory + firstFit +
    fileState entirely untouched -- the "half-refresh" bug class."""

    def test_refresh_handler_calls_loadByPath(self, core_body):
        """The EVENT_REFRESH_REQUESTED handler body MUST call
        loadByPath(...) -- not pollOnce()."""
        m = re.search(
            r"EVENT_REFRESH_REQUESTED[^{]*\{"
            r".+?_onRefresh\s*=\s*\(\)\s*=>\s*\{([^}]+)\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "Refresh handler not found"
        body = m.group(1)
        assert "loadByPath" in body, (
            "Refresh handler no longer calls loadByPath().  Pre-PR-2 "
            "it called pollOnce() directly which skipped the reset "
            "matrix; the half-refresh bug returns.")
        assert "pollOnce" not in body, (
            "Refresh handler is back to calling pollOnce() directly. "
            "scfPollHistory leaks stale samples; firstFit stays false; "
            "the half-refresh bug returns.")


# --------------------------------------------------------------------- #
#  Invariant 1: file-identity guard at fetch resolution                 #
# --------------------------------------------------------------------- #


class TestFileIdentityGuard:
    """Every async fetch resolution MUST compare its captured
    ``mySeq`` against the live ``state.lifecycle.fetchSeq``.  If a
    newer transition('LOADING') ran while the fetch was on the wire,
    the response is dropped."""

    def test_loadByPath_captures_fetchSeq(self, core_body):
        """loadByPath snapshots fetchSeq AFTER transition() runs --
        the bumped value is the one this fetch carries."""
        m = re.search(
            r"async\s+function\s+loadByPath\s*\([^)]*\)\s*\{(.+?)\n\s{4}\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "loadByPath function not found"
        body = m.group(1)
        # The seq capture follows transition('LOADING', ...).
        assert re.search(
            r"transition\s*\(\s*[\"']LOADING[\"'].*?mySeq\s*=\s*"
            r"state\.lifecycle\.fetchSeq",
            body, re.DOTALL,
        ), ("loadByPath no longer captures mySeq AFTER "
            "transition('LOADING') runs.  File-identity guard is "
            "broken; late responses can apply to wrong file.")

    def test_loadByPath_guards_fetch_resolution(self, core_body):
        """The .then() / await resolution MUST check fetchSeq before
        applying the response."""
        m = re.search(
            r"async\s+function\s+loadByPath\s*\([^)]*\)\s*\{(.+?)\n\s{4}\}",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert re.search(
            r"state\.lifecycle\.fetchSeq\s*!==\s*mySeq",
            body,
        ), ("loadByPath fetch-resolution path no longer checks "
            "fetchSeq.  A late response from a prior file can apply "
            "to the current file's view.")

    def test_pollOnce_guards_fetch_resolution(self, core_body):
        """pollOnce MUST also carry a fetchSeq guard -- a poll that
        spans a file-switch transition has the same race."""
        m = re.search(
            r"async\s+function\s+pollOnce\s*\(\)\s*\{(.+?)\n\s{4}\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "pollOnce function not found"
        body = m.group(1)
        assert "mySeq" in body and "fetchSeq" in body, (
            "pollOnce no longer carries a fetchSeq guard.  A poll "
            "in-flight at file-switch time can land on the new "
            "file's view with stale data.")


# --------------------------------------------------------------------- #
#  Invariant 2: in-progress frame filter                                #
# --------------------------------------------------------------------- #


class TestInProgressFilter:
    """Server emits ``data.in_progress[i]`` per-frame bool.  JS
    ``plottableFrames(data)`` MUST filter those frames from plot
    trace y-arrays -- the parser's placeholder energy is a
    ``step_initial_etot`` fallback, not a real measurement."""

    def test_plottableFrames_helper_exists(self, core_body):
        assert re.search(
            r"function\s+plottableFrames\s*\(\s*data\s*\)",
            core_body,
        ), ("trajectory/core.js no longer defines plottableFrames(). "
            "The odd-value-disappears-on-refresh bug class returns.")

    def test_plottableFrames_reads_in_progress(self, core_body):
        """The helper MUST read data.in_progress -- that's the
        canonical wire field added in this PR."""
        m = re.search(
            r"function\s+plottableFrames\s*\([^)]*\)\s*\{([^}]+?)\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "plottableFrames body shape changed"
        body = m.group(1)
        assert "in_progress" in body, (
            "plottableFrames no longer reads data.in_progress.  "
            "The filter is dead; partial frames re-appear in plots.")

    def test_energy_plot_uses_filter(self, core_body):
        """The energy plot trace MUST consume ``energies_plot`` (the
        filtered array), NOT ``state.data.energies`` directly."""
        # Find the energy-plot Plotly.react call body.
        m = re.search(
            r"Plotly\.react\s*\(\s*[\"']energy-plot[\"'].+?\}\s*\]",
            core_body, re.DOTALL,
        )
        assert m is not None, "energy-plot Plotly.react site not found"
        body = m.group(0)
        assert "energies_plot" in body, (
            "energy-plot trace no longer reads energies_plot (the "
            "in-progress-filtered array).  Partial-frame placeholder "
            "energies re-enter the plot.")
        assert "state.data.energies" not in body, (
            "energy-plot trace is back to reading state.data.energies "
            "directly -- the in-progress filter is bypassed.")


# --------------------------------------------------------------------- #
#  Wire-format: in_progress array landed                                #
# --------------------------------------------------------------------- #


class TestWireFormatInProgress:
    """The server-side trajectory_to_legacy_dict adapter (PR 2 also
    touched parsers/__init__.py) MUST emit ``in_progress`` in the
    output dict.  Empty list when no frame is in-progress."""

    def test_adapter_emits_in_progress(self):
        path = (Path(__file__).resolve().parent.parent
                / "molbuilder" / "parsers" / "__init__.py")
        body = path.read_text()
        assert re.search(
            r"\"in_progress\"\s*:\s*out_in_progress",
            body,
        ), ("parsers/__init__.py::trajectory_to_legacy_dict no "
            "longer emits ``in_progress`` -- the JS filter has no "
            "data to filter on.")

    def test_adapter_collapses_when_all_clean(self):
        """When no frame is in-progress, the array collapses to [] --
        matches the max_forces_constrained empty-list convention."""
        path = (Path(__file__).resolve().parent.parent
                / "molbuilder" / "parsers" / "__init__.py")
        body = path.read_text()
        assert re.search(
            r"if\s+not\s+any\s*\(\s*out_in_progress\s*\)\s*:\s*"
            r"\n\s+out_in_progress\s*=\s*\[\]",
            body,
        ), ("parsers/__init__.py no longer collapses all-clean "
            "in_progress to [].  Wire size bloats unnecessarily.")

    def test_merge_propagates_in_progress(self):
        """Multi-stage merge MUST keep the per-frame in_progress
        array aligned 1:1 with merged frames."""
        path = (Path(__file__).resolve().parent.parent
                / "molbuilder" / "web" / "blueprints" / "watch.py")
        body = path.read_text()
        assert re.search(
            r"\"in_progress\"\s*:\s*\[\]",
            body,
        ), ("watch.py multi-stage merge init no longer carries "
            "in_progress.  Per-frame partial flags get lost across "
            "stage boundaries.")
        assert re.search(
            r"merged\[\"in_progress\"\]\.extend",
            body,
        ), ("watch.py multi-stage merge no longer extends the "
            "in_progress array per stage.")
