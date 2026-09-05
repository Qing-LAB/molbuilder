"""`transition()` — the two side effects nothing else was checking.

`results.md` § 4 makes `transition()` the single entry point for state
writes, and two of its side effects had no behavioural test anywhere.
Both were found by mutation on 2026-09-04, while deciding whether the
28 source-greps in `test_results_state_contract_js.py` could be deleted:

* **`IDLE` clears `fileState.path`** — nothing caught its removal. 84
  tests passed with `dispose()` leaving the path set. It matters because
  that null is what makes a late poll answer from a DISPOSED inspector
  fail the path guard (`if (myPath !== state.fileState.path) return`).
  Leave it set and a dead viewer gets painted.
* **`LOADED` stops the poll timer** — caught only by a source-grep for
  the string `stopPolling()`, which dies with a rename.

The third behaviour in that group, § 4.1's two-tick settle, needed no
test written: `test_trajectory_settle_post_load_js.py` already drives
`_settlePostLoad` in node, and the mutation `finishedTicks >= 2` -> `>= 1`
fails it. That was checked before writing anything here.

**These RUN the real function.** `transition`'s source is lifted from the
shipped module and executed against a fake state, so what is asserted is
the OUTCOME — the path is null, the timer is off — and not the presence
of a string.

**One honest limitation, measured.** The harness stubs `transition`'s two
collaborators BY NAME (`startPolling` / `stopPolling`), so renaming one
breaks this file. That was checked: renaming both leaves the behaviour
identical and fails these tests. It is a real coupling and it is the
price of a unit harness that fakes collaborators; the difference from a
source-grep is the direction of the error. A grep passes quietly when the
behaviour breaks and fails when a name moves. This fails loudly when a
name moves — with a message saying so — and fails when the behaviour
breaks. Only one of those two is a test.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/lib/trajectory/core.js")


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)].rstrip()


def _transition(target, *, machine="WATCHING", path="/p/run.molwatch.log"):
    """Run the REAL `transition(target)` and report the state it left.

    Everything `transition` reaches outside itself is faked at the edge:
    the two pollers (which own a timer), and the abort controllers.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    src = MODULE.read_text()
    fn = _slice(src, "    function transition(target, payload) {",
                "\n    function ")

    harness = f"""
        let _timerRunning = true;
        function startPolling() {{ _timerRunning = true; }}
        function stopPolling()  {{ _timerRunning = false; }}
        const _aborted = [];
        const _ac = (name) => ({{ abort: () => _aborted.push(name) }});
        const state = {{
            machine: {json.dumps(machine)},
            fileState: {{ path: {json.dumps(path)}, mtime: 1, format: "siesta",
                         label: "l", data: {{}}, atomMetadata: {{}} }},
            viewState: {{}},
            uiPrefs: {{}},
            lifecycle: {{ loadAbort: _ac("load"), pollAbort: _ac("poll"),
                         pollInFlight: true, finishedTicks: 1,
                         pollTimer: 1 }},
            derived: {{ scfPollHistory: [1, 2] }},
        }};
        {fn}
        transition({json.dumps(target)});
        console.log(JSON.stringify({{
            machine:      state.machine,
            path:         state.fileState.path,
            timerRunning: _timerRunning,
            aborted:      _aborted,
            scfHistory:   state.derived.scfPollHistory.length,
            finishedTicks: state.lifecycle.finishedTicks,
        }}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        hint = ""
        if "is not defined" in proc.stderr:
            hint = ("\n\nTHIS HARNESS STUBS `transition`'s COLLABORATORS BY "
                    "NAME (`startPolling` / `stopPolling`).  Renaming one in "
                    "the module breaks the stub, not the behaviour -- add the "
                    "new name above.  Stated rather than hidden: a unit "
                    "harness that fakes collaborators is coupled to their "
                    "names, and the honest difference from the source-greps "
                    "these replace is that this fails LOUDLY here instead of "
                    "passing quietly on a broken behaviour.")
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}{hint}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_idle_clears_the_file_identity():
    """Disposing the viewer must forget WHICH FILE it was showing.

    A poll already on the wire when the inspector is disposed resolves
    afterwards and is dropped by comparing its file against
    `state.fileState.path`. If `IDLE` leaves that set, the comparison
    passes and a disposed viewer is painted.
    """
    out = _transition("IDLE")
    assert out["path"] is None, (
        "transition('IDLE') left fileState.path set. A late poll answer "
        "for the disposed file now passes the path guard and paints into "
        "a viewer that is gone.")


def test_idle_stops_the_timer_and_releases_the_requests():
    out = _transition("IDLE")
    assert out["timerRunning"] is False
    assert sorted(out["aborted"]) == ["load", "poll"], (
        f"IDLE must release both in-flight requests; released "
        f"{out['aborted']}")


def test_loaded_stops_polling():
    """A run that has settled is not polled again.

    Checked by a source-grep for the string `stopPolling()` until
    2026-09-04 — which a rename breaks and which says nothing about
    whether the timer actually stopped.
    """
    out = _transition("LOADED")
    assert out["timerRunning"] is False, (
        "transition('LOADED') left the poll timer running. A finished run "
        "is re-fetched every 15 s for as long as the tab is open.")
    assert out["machine"] == "LOADED"


def test_loading_releases_the_previous_files_requests_and_estimate():
    """A file switch must not carry the previous file's work forward.

    Two things ride on it. The in-flight requests are released, so a late
    answer for the old file cannot arrive at all; and the per-iteration
    wall-time estimate is emptied, because it is an average over the
    PREVIOUS run's polls and would otherwise be reported as this one's.
    """
    out = _transition("LOADING")
    assert sorted(out["aborted"]) == ["load", "poll"], (
        f"LOADING must release the previous file's requests; released "
        f"{out['aborted']}")
    assert out["scfHistory"] == 0, (
        "LOADING left the previous run's poll history in place. The "
        "per-iteration estimate is an average over those samples, so the "
        "new file's SCF line would report the old run's speed.")


def test_loading_resets_the_finished_counter():
    """`finishedTicks` is a count toward settling THIS file.

    Carried across a switch, a new file that reports `ended` once would
    settle immediately — skipping the two-tick buffer § 4.1 exists for,
    which is the whole defence against a parser caught mid-flush.
    """
    assert _transition("LOADING", machine="WATCHING")["finishedTicks"] == 0


# --------------------------------------------------------------------- #
#  plottableFrames — which frames the plots are allowed to use          #
# --------------------------------------------------------------------- #

def _plottable(frames, in_progress):
    """Run the REAL `plottableFrames(data)` and return the indices it keeps."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    fn = _slice(MODULE.read_text(), "    function plottableFrames(data) {",
                "\n    // Expose for tests")
    harness = (fn + "\nconsole.log(JSON.stringify(plottableFrames("
               + json.dumps({"frames": frames, "in_progress": in_progress})
               + ")));")
    proc = subprocess.run([shutil.which("node"), "--input-type=commonjs",
                           "-e", harness],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_mid_write_frame_is_kept_out_of_the_plots():
    """`results.md` § 4: partial frames are LISTED but not PLOTTED.

    The half-written geometry is still in the movie — you can scrub to it
    — but its energy is a number the engine had not finished computing,
    and plotting it puts a spike in the convergence curve that never
    happened.
    """
    assert _plottable(["a", "b", "c"], [False, True, False]) == [0, 2]


def test_an_empty_flag_array_means_every_frame_plots():
    """`[]` is the adapter's *nothing is partial* (see
    `test_in_progress_frames_stay_out_of_plots.py`), not *nothing plots*.

    Reading it the other way would blank every plot on every healthy run,
    which is the whole reason the collapse convention needs a reader that
    agrees with it.
    """
    assert _plottable(["a", "b"], []) == [0, 1]


def test_no_frames_is_not_an_error():
    assert _plottable([], []) == []


# --------------------------------------------------------------------- #
#  Refresh — the button must reach the loader                           #
# --------------------------------------------------------------------- #

def _refresh(*, path):
    """Wire the REAL `_wireRefreshListener`, fire the event, report what
    it called.

    Everything outside the function is faked at the edge: the constant
    bundle it reads the event name from, the listener registrar, the
    loader, and the observer it also installs.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    fn = _slice(MODULE.read_text(), "    function _wireRefreshListener() {",
                "\n    function ")
    harness = f"""
        const _loaded = [];
        const _handlers = {{}};
        const window = {{ molbuilder: {{ constants:
            {{ EVENT_REFRESH_REQUESTED: "molbuilder:results:refresh" }} }} }};
        const document = {{}};
        function _on(t, ev, h) {{ _handlers[ev] = h; }}
        function loadByPath(p) {{ _loaded.push(p); }}
        function resizePlots() {{}}
        class ResizeObserver {{ constructor(f) {{}} observe() {{}} }}
        const $ = () => null;
        // The same function also installs the plots' ResizeObserver, which
        // reaches the card's root element.  Faked so the wiring under test
        // runs; that observer has its own coverage.
        const rootEl = {{ querySelector: () => null }};
        const state = {{ fileState: {{ path: {json.dumps(path)} }} }};
        {fn}
        _wireRefreshListener();
        const h = _handlers["molbuilder:results:refresh"];
        if (h) h();
        console.log(JSON.stringify({{ wired: !!h, loaded: _loaded }}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_refresh_reloads_the_file_on_screen():
    """Pressing Refresh re-runs the load for the file being shown.

    `results.md` § 5: Refresh is a clean reload, not a nudge — it goes
    through `loadByPath`, so `transition('LOADING')` runs the whole reset
    matrix. Until 2026-09-04 the only guard on this was a grep for the
    string `_wireRefreshListener();`, so deleting the registration left
    the button inert with the suite green.
    """
    out = _refresh(path="/p/run.molwatch.log")
    assert out["wired"] is True, (
        "nothing subscribed to the refresh event: the Refresh button is "
        "wired to nothing and does nothing, silently")
    assert out["loaded"] == ["/p/run.molwatch.log"], (
        f"Refresh must reload the file on screen; it called loadByPath "
        f"with {out['loaded']!r}")


def test_refresh_before_anything_is_loaded_does_nothing():
    """No file, no reload — and no crash on a null path."""
    assert _refresh(path=None)["loaded"] == []
