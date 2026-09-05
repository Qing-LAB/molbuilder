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


def _transition(target, *, machine="WATCHING", path="/p/run.molwatch.log",
                timer_running=True):
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
        let _timerRunning = {json.dumps(timer_running)};
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


# --------------------------------------------------------------------- #
#  The state's shape, and the legacy names that read through to it      #
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
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = MODULE.read_text()
    state_lit = _slice(src, "    const state = {", "\n    (function _wireBackcompatAliases")
    wiring = _slice(src, "    (function _wireBackcompatAliases", "\n    // Transition orchestrator")
    lifecycle = (Path(__file__).resolve().parents[1]
                 / "molbuilder/web/static/lib/inspectors/lifecycle.js").read_text()
    harness = f"""
        const root = globalThis;
        {lifecycle}
        {state_lit}
        {wiring}
        // Write through the FLAT name, read back through the BUCKET.
        state.path = "/p/x.out";
        state.scfPollHistory.push(7);
        console.log(JSON.stringify({{
            buckets:      Object.keys(state).filter(k =>
                              ["fileState","viewState","uiPrefs","lifecycle","derived"]
                              .includes(k)),
            machine:      state.machine,
            throughFile:  state.fileState.path,
            throughDeriv: state.derived.scfPollHistory,
        }}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_state_starts_idle_and_carries_its_buckets():
    out = _aliased_state()
    assert out["machine"] == "IDLE", (
        "a freshly built viewer must be IDLE; anything else means a mount "
        "starts in a state whose transition side effects never ran")
    for bucket in ("fileState", "viewState", "uiPrefs", "lifecycle"):
        assert bucket in out["buckets"], (
            f"§ 4's `{bucket}` bucket is missing from the state object")


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
    assert out["throughDeriv"] == [7], (
        "`state.scfPollHistory` no longer reads through to `derived`")


@pytest.mark.parametrize("target,timer", [("LOADED", False),
                                          ("WATCHING", True),
                                          ("ERROR", False)])
def test_every_target_lands_the_machine_and_settles_the_timer(target, timer):
    """Each target reaches its own branch and leaves the poll timer right.

    Checked by grepping for `state.machine = "<NAME>"` until 2026-09-04,
    which says a branch exists and nothing about what it does. WATCHING
    is the one that must leave the timer RUNNING — a run still going is
    the whole reason the viewer polls at all.

    **The harness starts the timer in the OPPOSITE state**, or the
    assertion is free: with the timer already running, "WATCHING leaves
    it running" passes whether or not the branch starts it. Measured —
    the first version of this test survived deleting `startPolling()`
    from the WATCHING branch, which is a live run that never updates.
    """
    out = _transition(target, timer_running=not timer)
    assert out["machine"] == target
    assert out["timerRunning"] is timer, (
        f"transition('{target}') left the poll timer "
        f"{'running' if out['timerRunning'] else 'stopped'}; it must be "
        f"{'running' if timer else 'stopped'}")


def test_error_stops_the_poll_but_does_NOT_release_its_controller():
    """What the ERROR branch actually does — which is not what I said.

    This was `test_error_releases_the_request_that_failed`, asserting
    *"a failed load must not leave its controller behind"*.  It does
    leave it behind: `core.js`'s ERROR branch clears `watchTimer` and
    `watchInFlight` and never calls `.abort()`, unlike IDLE, LOADING and
    LOADED, which all do.  The test's one assertion was
    `timerRunning is False` — an exact duplicate of the `("ERROR",
    False)` row of the parametrized test above — so it passed while its
    name and docstring taught the opposite of the code.

    Asserted here as OBSERVED behaviour, not as a rule: no document
    states an abort contract for ERROR, and inventing one in a test is
    how a test starts pinning a requirement nobody wrote.  If ERROR
    should release its controllers, that is a change to `core.js` and to
    `results.md`, and this test then changes with them.
    """
    out = _transition("ERROR", timer_running=True)
    assert out["machine"] == "ERROR"
    assert out["timerRunning"] is False, "the poll must stop on error"
    assert out["aborted"] == [], (
        "ERROR now aborts its in-flight requests — real behaviour changed, "
        f"and no document says which way is right: {out['aborted']}")

    # The three that DO release, so the contrast is measured rather than
    # asserted from memory.
    for target in ("IDLE", "LOADING"):
        assert _transition(target, timer_running=True)["aborted"], (
            f"transition('{target}') stopped aborting its controllers")


def test_starting_the_poll_wires_no_listener():
    """`startPolling` is timer-only.

    The Refresh listener is wired ONCE at mount. It used to be wired
    here, so every load stacked another handler and one Refresh fired N
    loads; only dispose tore them down. This runs the real function with
    a spying registrar and asserts it registered nothing — the previous
    guard was a regex over the function body, which a handler added
    through any other spelling would slip past.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    fn = _slice(MODULE.read_text(), "    function startPolling() {",
                "\n    function stopPolling")
    harness = f"""
        const _registered = [];
        function _on(t, ev) {{ _registered.push(ev); }}
        function pollOnce() {{}}
        const POLL_MS = 15000;
        const state = {{ pollTimer: null }};
        const document = {{ addEventListener: (ev) => _registered.push(ev) }};
        const window = {{ molbuilder: {{ constants:
            {{ EVENT_REFRESH_REQUESTED: "molbuilder:results:refresh" }} }} }};
        globalThis.setInterval = () => 1;
        globalThis.clearInterval = () => {{}};
        {fn}
        startPolling();
        console.log(JSON.stringify({{ registered: _registered,
                                      timer: state.pollTimer }}));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["registered"] == [], (
        f"startPolling registered {out['registered']!r}. Every load would "
        f"add another handler, so one Refresh fires N reloads.")
    assert out["timer"] is not None, "startPolling must actually start a timer"


# --------------------------------------------------------------------- #
#  The loaders must actually CALL the settle                            #
# --------------------------------------------------------------------- #

def _poll_once(*, run_state, path="/p/run.molwatch.log"):
    """Run the REAL `pollOnce` against a stubbed `/api/watch/data`.

    Everything it reaches is faked at the edge — `fetch`, the applier,
    the status line — except `_settlePostLoad`, which is lifted from the
    module and run for real. That is the point: the previous guard was a
    grep for the string `_settlePostLoad` inside `pollOnce`'s body, which
    says the call is written and nothing about whether it happens.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = MODULE.read_text()
    run_state_const = _slice(src, "const RUN_STATE = Object.freeze", "});") + "});"
    settle = _slice(src, "    function _settlePostLoad", "\n    function plottableFrames")
    poll = _slice(src, "    async function pollOnce()", "\n    function ")
    harness = f"""
        {run_state_const}
        const _seen = [];
        function transition(name) {{ _seen.push(name); state.machine = name; }}
        function applyNewData(r) {{ state.fileState.data = r.data; }}
        function setStatus() {{}}
        function dispose() {{}}
        const state = {{
            machine: "WATCHING",
            fileState: {{ path: {json.dumps(path)}, mtime: 1,
                         data: {{ run_state: {json.dumps(run_state)} }} }},
            lifecycle: {{ pollInFlight: false, pollAbort: null,
                         finishedTicks: 1 }},
            derived: {{}},
        }};
        globalThis.fetch = async () => ({{
            ok: true,
            json: async () => ({{ ok: true, changed: true, mtime: 2,
                                  data: {{ run_state: {json.dumps(run_state)},
                                          frames: [] }} }}),
        }});
        {settle}
        {poll}
        pollOnce().then(() => console.log(JSON.stringify({{
            transitions: _seen, machine: state.machine,
        }})));
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_poll_that_finds_the_run_ended_settles_it():
    """The poll loop must ASK `_settlePostLoad`, not just contain a call
    to it.

    With `finishedTicks` already at 1, an `ended` answer is the second
    consecutive one, so the run settles and polling stops. If `pollOnce`
    stopped calling the settle, a finished run would be re-fetched every
    15 s for as long as the tab is open — which is bug #12, and the grep
    that guarded it could not tell a written call from a reached one.
    """
    out = _poll_once(run_state="ended")
    assert "LOADED" in out["transitions"], (
        f"a second 'ended' poll must settle the run; the poll transitioned "
        f"{out['transitions']!r}")


def test_a_poll_on_a_running_run_keeps_watching():
    out = _poll_once(run_state="running")
    assert out["transitions"] == ["WATCHING"], out["transitions"]
