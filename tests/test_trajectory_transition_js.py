"""`results.md` § 4 and § 4.2 on the TRAJECTORY side, run: the viewer's
own `transition()`, on no answer of the server's.

* `IDLE` clears the file's name, so an answer after dispose fails the
  guard; each target lands the machine and leaves the poll timer right;
* mid-write frames stay out of the plots; the legacy flat names are the
  buckets' values; starting the poll starts its timer.

What the viewer makes of the server's ANSWERS -- an answer written under its
own file's name or not at all, the poll loop taking how the run is doing,
the settle -- is fed what the routes answered of real runs:
`test_the_viewers_follow_a_real_run_e2e.py`.

**These RUN the real functions.** Their source is lifted from the shipped
module (`tests/_node_esm.lifted`) and executed against a fake state -- or,
where the state itself is the question, the module's real one -- so
what is asserted is the OUTCOME -- the path is null, the timer is off --
and not the presence of a string.

**One honest limitation, measured.** The harness stubs `transition`'s two
collaborators BY NAME (`startPolling` / `stopPolling`), so renaming one
breaks this file. That was checked: renaming both leaves the behaviour
identical and fails these tests. It is a real coupling and it is the
price of a unit harness that fakes collaborators; the difference from a
source-grep is the direction of the error. A grep passes quietly when the
behaviour breaks and fails when a name moves. This fails loudly when a
name moves — `lifted` names the marker it could not find — and fails
when the behaviour breaks. Only one of those two is a test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import lifted, run_node

MODULE = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/lib/trajectory/core.js")
LIFECYCLE = (Path(__file__).resolve().parents[1]
             / "molbuilder/web/static/lib/inspectors/lifecycle.js")


def _transition(target, *, machine="WATCHING", path="/p/run.molwatch.log",
                timer_running=True):
    """Run the REAL `transition(target)` and report the state it left.

    Everything `transition` reaches outside itself is faked at the edge:
    the two pollers (which own a timer), and the abort controllers.
    """
    fn = lifted(MODULE, "    function transition(target, payload) {",
                "\n    function ")

    harness = f"""
        let _timerRunning = {json.dumps(timer_running)};
        function startPolling() {{ _timerRunning = true; }}
        function stopPolling()  {{ _timerRunning = false; }}
        const _aborted = [];
        const _ac = (name) => ({{ abort: () => _aborted.push(name) }});
        const state = {{
            machine: {json.dumps(machine)},
            fileState: {{ path: {json.dumps(path)}, mtime: 1, format: "siesta",
                         label: "l", data: {{}}, structure: null,
                         run: {{ state: "running", live: true }} }},
            viewState: {{}},
            uiPrefs: {{}},
            lifecycle: {{ loadAbort: _ac("load"), pollAbort: _ac("poll"),
                         pollInFlight: true, pollTimer: 1 }},
            derived: {{}},
        }};
        {fn}
        transition({json.dumps(target)});
        console.log(JSON.stringify({{
            machine:      state.machine,
            path:         state.fileState.path,
            timerRunning: _timerRunning,
            aborted:      _aborted,
            run:          state.fileState.run,
        }}));
    """
    return run_node([], harness)


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


def test_loading_releases_the_previous_files_requests():
    """A file switch must not carry the previous file's work forward: the
    in-flight requests are released, so a late answer for the old file
    cannot arrive at all.
    """
    out = _transition("LOADING")
    assert sorted(out["aborted"]) == ["load", "poll"], (
        f"LOADING must release the previous file's requests; released "
        f"{out['aborted']}")


def test_loading_forgets_the_last_files_run():
    """How the run is doing belongs to THIS file (`results.md` § 4.1).

    Carried across a switch, the previous file's live run would keep the
    new file followed -- or its finished one would leave a live run
    unwatched -- until an answer for the new file landed.
    """
    assert _transition("LOADING", machine="WATCHING")["run"] is None


# --------------------------------------------------------------------- #
#  plottableFrames — which frames the plots are allowed to use          #
# --------------------------------------------------------------------- #

def _plottable(frames, in_progress):
    """Run the REAL `plottableFrames(data)` and return the indices it keeps."""
    fn = lifted(MODULE, "    function plottableFrames(data) {",
                "\n    // Expose for tests")
    return run_node([], fn + "\nconsole.log(JSON.stringify(plottableFrames("
                    + json.dumps({"frames": frames, "in_progress": in_progress})
                    + ")));")


def test_a_mid_write_frame_is_kept_out_of_the_plots():
    """`results.md` § 4: partial frames are LISTED but not PLOTTED.

    The half-written geometry is still in the movie — you can scrub to it
    — but its energy is a number the engine had not finished computing,
    and plotting it puts a spike in the convergence curve that never
    happened.
    """
    assert _plottable(["a", "b", "c"], [False, True, False]) == [0, 2]


def test_an_empty_flag_array_means_every_frame_plots():
    """`[]` is the adapter's *nothing is partial*
    (`trajectory_result_to_legacy_dict`'s own convention, which no test asks
    of a real run: the viewers' e2e asks only that no frame is flagged), not
    *nothing plots*.

    Reading it the other way would blank every plot on every healthy run,
    which is the whole reason the collapse convention needs a reader that
    agrees with it.
    """
    assert _plottable(["a", "b"], []) == [0, 1]


def test_no_frames_is_not_an_error():
    assert _plottable([], []) == []


# --------------------------------------------------------------------- #
#  A fresh viewer, and the legacy names that read through to the state #
# --------------------------------------------------------------------- #

def _aliased_state():
    """Build the REAL state object and run the REAL alias wiring.

    `results.md` § 4 names four buckets — the parsed file, your per-file
    view, your per-session preferences, and the poll timer. Older code
    reached for flat names (`state.path`), and `_wireBackcompatAliases`
    makes those read and write THROUGH the bucket rather than beside it.
    A flat name that stopped being an alias would become a second copy of
    the same fact, silently diverging.
    """
    state_lit = lifted(MODULE, "    const state = {",
                       "\n    (function _wireBackcompatAliases")
    wiring = lifted(MODULE, "    (function _wireBackcompatAliases",
                    "\n    // Transition orchestrator")
    harness = f"""
        const root = globalThis;
        {state_lit}
        {wiring}
        // Write through the FLAT name, read back through the BUCKET.
        state.path = "/p/x.out";
        console.log(JSON.stringify({{
            machine:      state.machine,
            throughFile:  state.fileState.path,
        }}));
    """
    return run_node([LIFECYCLE], harness)


def test_a_viewer_starts_idle():
    out = _aliased_state()
    assert out["machine"] == "IDLE", (
        "a freshly built viewer must be IDLE; anything else means a mount "
        "starts in a state whose transition side effects never ran")


def test_a_legacy_flat_name_writes_THROUGH_to_its_bucket():
    """`state.path = x` must land in `state.fileState.path`.

    If the alias were replaced by a plain property, both would exist and
    hold different values — and every guard that compares
    `state.fileState.path` would be reading a copy nobody updates.
    """
    out = _aliased_state()
    assert out["throughFile"] == "/p/x.out", (
        "writing the flat `state.path` did not reach `fileState.path`: the "
        "alias is gone and the two names are now separate storage")


@pytest.mark.parametrize("target,timer", [("LOADED", False),
                                          ("WATCHING", True),
                                          ("ERROR", False)])
def test_every_target_lands_the_machine_and_settles_the_timer(target, timer):
    """Each target reaches its own branch and leaves the poll timer right.

    WATCHING is the one that must leave the timer RUNNING — a run still going is
    the whole reason the viewer polls at all; LOADED and ERROR stop it, or a
    finished run is re-fetched every 15 s for as long as the tab is open.

    **The harness starts the timer in the OPPOSITE state**, or the
    assertion is free: with the timer already running, "WATCHING leaves
    it running" passes whether or not the branch starts it.
    """
    out = _transition(target, timer_running=not timer)
    assert out["machine"] == target
    assert out["timerRunning"] is timer, (
        f"transition('{target}') left the poll timer "
        f"{'running' if out['timerRunning'] else 'stopped'}; it must be "
        f"{'running' if timer else 'stopped'}")


def test_starting_the_poll_follows_the_run_every_15_s_once():
    """`startPolling` is what `transition('WATCHING')` calls to keep a live
    run followed (the harness above fakes it), so the shipped one runs here
    on the module's real state, its aliases wired, and Node's own timers --
    each call only watched: it polls the run (`pollOnce`) every 15 s
    (`results.md`: a trajectory is polled every 15 s), and a second start replaces the first rather than
    stacking a second poller on the same run.
    """
    state_lit = lifted(MODULE, "    const state = {",
                       "\n    (function _wireBackcompatAliases")
    wiring = lifted(MODULE, "    (function _wireBackcompatAliases",
                    "\n    // Transition orchestrator")
    every = lifted(MODULE, "    const POLL_MS = ", ";\n")
    fn = lifted(MODULE, "    function startPolling() {",
                "\n    function stopPolling")
    harness = f"""
        const root = globalThis;
        {state_lit}
        {wiring}
        {every};
        function pollOnce() {{}}
        const started = [], cleared = [];
        const realSet = setInterval, realClear = clearInterval;
        globalThis.setInterval = (f, ms) => {{
            const t = realSet(f, ms); started.push({{t, ms, f}}); return t; }};
        globalThis.clearInterval = (t) => {{ cleared.push(t); realClear(t); }};
        {fn}
        startPolling();
        const first = state.lifecycle.pollTimer;
        startPolling();
        const second = state.lifecycle.pollTimer;
        console.log(JSON.stringify({{
            every:    started.map((s) => s.ms),
            polls:    started.every((s) => s.f === pollOnce),
            replaced: cleared.length === 1 && cleared[0] === first,
            live:     second === started[1].t,
        }}));
        started.forEach((s) => realClear(s.t));
    """
    out = run_node([LIFECYCLE], harness)
    assert out["every"] == [15000, 15000], out
    assert out["polls"] and out["replaced"] and out["live"], out
