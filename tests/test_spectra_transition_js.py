"""`results.md` § 4 on the SPECTRA side, run: the spectra viewer's own
`transition()` -- a second implementation of the rules the trajectory
viewer's keeps (`test_trajectory_transition_js.py`), asked of its own
state.

* a file switch releases the last file's requests and forgets it, resets
  the view and keeps the preferences; after an answer the viewer follows
  the run the server named while it is live, and not once it is not, or
  for a file that belongs to no run (§ 4, § 4.1; `spectra.md` § 7);
* an answer is written under its own file's name or not at all, one that
  names no file is refused, after dispose a late answer writes nothing,
  and a poll's answer that arrives after the same file was opened again
  draws nothing (§ 4.2);
* the legacy flat names and the buckets are one value, and a knob's write
  is saved (`spectra.md` § 7).

**These RUN the shipped functions**, lifted from `lib/spectra/core.js`
(`_node_esm.lifted`) and run against fakes at the edge -- the timers, the
abort controllers, the loader -- so what is asserted is what the viewer
then holds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import lifted, run_node


_LIB = Path(__file__).resolve().parents[1] / "molbuilder/web/static/lib"
MODULE = _LIB / "spectra" / "core.js"
LIFECYCLE = _LIB / "inspectors" / "lifecycle.js"


# --------------------------------------------------------------------- #
#  What the viewer holds after each event                               #
# --------------------------------------------------------------------- #

_A = "/p/CO2/02_freq/run-0/CO2.spectra.json"
_B = "/p/CO2/02_freq/run-1/CO2.spectra.json"
_PREFS = {"modeFilter": "ir", "broadeningFWHM": 12}
_LIVE = {"state": "running", "detail": "running", "live": True}
_FINISHED = {"state": "finished", "detail": "job_completed", "live": False}
_FAILED = {"state": "failed", "detail": "stopped", "live": False}


def _viewer(script, *, machine, path=_A, results="A", polling=True,
            run=None, in_flight=True):
    """The spectra viewer's state -- ``machine``, showing ``path`` with
    ``results`` (a mark, or none), its poll timer running when ``polling``
    and a poll on the wire when ``in_flight`` -- driven by ``script`` through the REAL `transition`, `dispose`,
    `_settlePostLoad` and `watchTick`; what it then holds, and what it drew.

    Faked at the edge: the timers and the abort controllers (each says
    when it is used), the server (``answer``, which a script may replace),
    the listener registrar, the drawing calls, and what dispose lets go of
    besides the state."""
    held = {"from": results} if results else None
    harness = f"""
        const _aborted = [];
        const _ac = (name) => ({{ abort: () => _aborted.push(name) }});
        let _polling = {json.dumps(polling)};
        globalThis.setInterval = () => {{ _polling = true; return 7; }};
        globalThis.clearInterval = () => {{ _polling = false; }};
        const WATCH_INTERVAL_MS = 2000;
        const WATCH_MAX_ERRORS = 5;
        const _drawn = [];
        let answer = async () => ({{ body: {{ ok: true, results: {{}} }} }});
        // WHAT A REQUEST ASKS, carried with its answer as the real one
        // carries it -- read when the request leaves, before its await.
        let _ask = {{}};
        let _asks = 0;
        function _askOf() {{ return _ask; }}
        const fetchResults = async (p, signal) => {{
            const asked = JSON.stringify(_askOf());
            _asks++;
            return Object.assign({{ asked }}, await answer(p, signal));
        }};
        function setStatus() {{ _drawn.push("status"); }}
        function stopWatch() {{ _drawn.push("stopped"); }}
        function renderResults() {{ _drawn.push("results"); }}
        function updatePhaseIndicator() {{ _drawn.push("phases"); }}
        function _showRunStatus() {{ _drawn.push("run status"); }}
        const _listeners = {{ disposeAll() {{}} }};
        const document = {{}};
        let chart = null, chartReady = null;
        const els = {{}};
        const root = globalThis;
        const state = {{
            machine: {json.dumps(machine)},
            fileState: {{ path: {json.dumps(path)}, results: {json.dumps(held)},
                         run: {json.dumps(run)} }},
            viewState: {{ selectedMode: 3 }},
            uiPrefs: {json.dumps(_PREFS)},
            lifecycle: {{ loadAbort: _ac("load"), watchAbort: _ac("watch"),
                         watchInFlight: {json.dumps(in_flight)},
                         watchTimer: {json.dumps(7 if polling else None)},
                         watchErrors: 2 }},
            derived: {{}},
            vib: null,
        }};
        {lifted(MODULE, "    function transition(target, payload) {",
                "\n    function ")}
        {lifted(MODULE, "    function _abortViews() {", "\n    }")}
        }}
        {lifted(MODULE, "    function _settlePostLoad() {", "\n    }")}
        }}
        {lifted(MODULE, "    async function watchTick() {",
                "\n    function ")}
        {lifted(MODULE, "    function _askAgainIfChanged(asked) {", "\n    }")}
        }}
        const handle = {{
        {lifted(MODULE, "        dispose() {", "\n        },")}
        }}
        }};
        let threw = null;
        (async () => {{
        try {{ {script} }}
        catch (e) {{ threw = String((e && e.message) || e); }}
        console.log(JSON.stringify({{
            machine: state.machine,
            path: state.fileState.path,
            results: state.fileState.results && state.fileState.results.from,
            run: state.fileState.run,
            selectedMode: state.viewState.selectedMode,
            prefs: state.uiPrefs,
            polling: _polling,
            aborted: _aborted.sort(),
            drawn: _drawn,
            asks: _asks,
            threw,
        }}));
        }})();
    """
    return run_node([], harness)


@pytest.mark.parametrize("start,script,holds", [
    pytest.param(
        {"machine": "WATCHING", "run": _LIVE},
        f'transition("LOADING", {{ path: "{_B}" }});',
        {"machine": "LOADING", "path": _B, "results": None, "run": None,
         "selectedMode": None, "prefs": _PREFS, "polling": False,
         "aborted": ["load", "watch"]},
        id="a file switch releases the last file and keeps the preferences"),
    pytest.param(
        {"machine": "WATCHING", "run": _LIVE},
        'transition("LOADED");',
        {"machine": "LOADED", "polling": False},
        id="a run no longer live is not polled"),
    pytest.param(
        {"machine": "LOADED", "polling": False},
        'transition("WATCHING");',
        {"machine": "WATCHING", "polling": True},
        id="a live run is polled"),
    pytest.param(
        {"machine": "LOADED"},
        f'transition("APPLY", {{ path: "{_B}", results: {{ from: "answer" }} }});',
        {"path": _A, "results": "A"},
        id="an answer for another file writes nothing"),
    pytest.param(
        {"machine": "LOADING", "results": None},
        f'transition("APPLY", {{ path: "{_A}", results: {{ from: "answer" }} }});',
        {"path": _A, "results": "answer"},
        id="an answer for the file on screen is written"),
    pytest.param(
        {"machine": "LOADED"},
        'transition("APPLY", { results: { from: "answer" } });',
        {"results": "A", "threw": "APPLY without a path"},
        id="an answer that names no file is refused"),
    pytest.param(
        {"machine": "WATCHING", "run": _LIVE},
        f'handle.dispose(); transition("APPLY", {{ path: "{_A}", '
        f'results: {{ from: "late" }} }});',
        {"machine": "IDLE", "path": None, "results": None, "polling": False,
         "aborted": ["load", "watch"]},
        id="after dispose a late answer for the file writes nothing"),
    pytest.param(
        {"machine": "LOADING", "polling": False, "run": _LIVE},
        '_settlePostLoad();',
        {"machine": "WATCHING", "polling": True},
        id="a run the server says is live is followed"),
    pytest.param(
        {"machine": "LOADING", "run": _FINISHED},
        '_settlePostLoad();',
        {"machine": "LOADED", "polling": False},
        id="a run that finished is not followed"),
    pytest.param(
        {"machine": "LOADING", "run": _FAILED},
        '_settlePostLoad();',
        {"machine": "LOADED", "polling": False},
        id="a run that stopped is not followed"),
    pytest.param(
        {"machine": "LOADING", "run": None},
        '_settlePostLoad();',
        {"machine": "LOADED", "polling": False},
        id="a file that belongs to no run is not followed"),
    pytest.param(
        {"machine": "WATCHING", "run": _LIVE, "in_flight": False},
        'answer = async (p) => { transition("LOADING", { path: p });'
        '  return { body: { ok: true, results: { from: "tick" },'
        '                   run: ' + json.dumps(_LIVE) + ' } }; };'
        ' await watchTick();',
        {"machine": "LOADING", "path": _A, "results": None, "polling": False,
         "drawn": []},
        id="a poll answered after the same file was opened again draws "
           "nothing"),
    pytest.param(
        {"machine": "WATCHING", "run": _LIVE, "in_flight": False},
        'answer = async () => ({ body: { ok: true, results: { from: "tick" },'
        '                   run: ' + json.dumps(_FINISHED) + ' } });'
        ' await watchTick();',
        {"machine": "LOADED", "run": _FINISHED, "polling": False,
         "drawn": ["results", "phases", "run status"]},
        id="a poll that says the run finished is drawn, and ends the follow"),
    pytest.param(
        {"machine": "WATCHING", "run": _LIVE, "in_flight": False},
        # The person types a laser while the poll is out: its answer is to
        # the older ask.
        'answer = async () => { _ask = { raman: { laser_nm: 633 } };'
        '  answer = async () => new Promise(() => {});'
        '  return { body: { ok: true, results: { from: "tick" },'
        '                   run: ' + json.dumps(_LIVE) + ' } }; };'
        ' await watchTick();',
        {"drawn": ["results", "phases", "run status"], "asks": 2},
        id="a poll answering an older ask is drawn, then the file asked "
           "again"),
    pytest.param(
        {"machine": "WATCHING", "run": _LIVE, "in_flight": False},
        'answer = async () => ({ body: { ok: true, results: { from: "tick" },'
        '                   run: ' + json.dumps(_LIVE) + ' } });'
        ' await watchTick();',
        {"drawn": ["results", "phases", "run status"], "asks": 1},
        id="a poll answering the ask as it stands asks nothing more"),
])
def test_what_the_spectra_viewer_holds_after(start, script, holds):
    """THE SILENT FAILURES: a late answer for a file you have moved off,
    drawn under the one on screen; a disposed viewer painted; a file switch
    that keeps the last file's requests, view or run, or drops your
    preferences; a finished run polled every two seconds for as long as the
    tab is open, or a live one never followed.

    MUTATIONS THIS MUST FAIL AGAINST: APPLY's path guard gone; IDLE leaving
    the path; dispose not reaching IDLE; LOADING keeping a request or
    clearing the preferences; LOADED leaving the timer; WATCHING starting
    none; the settle following any run, live or not; a tick's answer drawn
    after its request was let go."""
    out = _viewer(script, **start)
    for key, want in holds.items():
        if key == "threw":
            assert out["threw"] and want in out["threw"], out
        else:
            assert out[key] == want, (key, out)
    if "threw" not in holds:
        assert out["threw"] is None, out


# --------------------------------------------------------------------- #
#  The legacy names and the buckets are one value                       #
# --------------------------------------------------------------------- #

#: flat name -> the bucket that owns the value (`results.md` § 4).
_FIELDS = [
    ("results",           "fileState"),
    ("selectedMode",      "viewState"),
    ("modeFilter",        "uiPrefs"),
    ("sortColumn",        "uiPrefs"),
    ("sortDir",           "uiPrefs"),
    ("broadeningFWHM",    "uiPrefs"),
    ("animAmplitude",     "uiPrefs"),
    ("animSpeed",         "uiPrefs"),
    ("animAmplitudeMode", "uiPrefs"),
    ("animTemperature",   "uiPrefs"),
    ("watchTimer",        "lifecycle"),
    ("watchInFlight",     "lifecycle"),
    ("watchAbort",        "lifecycle"),
    ("loadAbort",         "lifecycle"),
    ("watchErrors",       "lifecycle"),
]

_KNOBS     = [f for f, b in _FIELDS if b == "uiPrefs"]
_NOT_KNOBS = [f for f, b in _FIELDS if b != "uiPrefs"]


@pytest.fixture(scope="module")
def wired():
    """Run the shipped `_wireBackcompatAliases` and report what it did.

    Everything it reaches outside itself is real or faked at the edge:
    `inspectorLifecycle.alias` is the module the core actually calls,
    `state` is a bare object with the five buckets, and `_prefsSchedule` is
    a counter -- the save the uiPrefs setters kick is itself a behaviour
    (`spectra.md` § 7), so it is observed rather than stubbed away.
    """
    body = lifted(MODULE, "(function _wireBackcompatAliases()",
                  "})();") + "})();"
    probe = """
const state = { fileState: {}, viewState: {}, uiPrefs: {}, lifecycle: {} };
let scheduled = 0;
function _prefsSchedule() { scheduled += 1; }
const root = globalThis;
""" + body + """
const out = { roundTrip: {}, saves: {} };
for (const [flat, bucket] of %s) {
    const before = scheduled;
    state[flat] = "v:" + flat;                     // write the LEGACY name
    out.roundTrip[flat] = (state[bucket][flat] === "v:" + flat)  // bucket sees it
                       && (state[flat] === "v:" + flat);         // and reads back
    out.saves[flat] = scheduled - before;
}
console.log(JSON.stringify(out));
""" % json.dumps([[f, b] for f, b in _FIELDS])
    return run_node([LIFECYCLE], probe)


class TestBackcompatAliases:
    """The legacy flat names and the buckets are ONE value.

    ~3000 lines of render and event code in `spectra/core.js` read and write
    `state.modeFilter`, `state.results`, `state.watchTimer` and the rest.
    The canonical home is a bucket (`results.md` § 4), and an alias is what
    keeps the two from ever disagreeing.  Lose one and the flat write lands
    on a plain property while every reader of the bucket sees nothing —
    silently, because both spellings still exist.

    **This RUNS the real wiring**, the way `test_trajectory_transition_js.py`
    runs the real `transition()`: `_wireBackcompatAliases` is lifted from the
    shipped module and executed in node against the shared alias helper it
    actually calls, so what is asserted is the round trip — write the flat
    name, read the bucket — and not the spelling of the call that wired it.
    """


    @pytest.mark.parametrize("flat,bucket", _FIELDS)
    def test_the_legacy_name_and_the_bucket_are_one_value(
            self, wired, flat, bucket):
        assert wired["roundTrip"][flat], (
            f"writing ``state.{flat}`` did not land in "
            f"``state.{bucket}.{flat}``.  The alias is gone, so the flat "
            f"name and the bucket are now two values and the render code "
            f"reading one cannot see the other.")

    @pytest.mark.parametrize("flat", _KNOBS)
    def test_a_knob_write_schedules_the_save(self, wired, flat):
        """`spectra.md` § 7: the viewer's knobs survive a reload, and the
        alias setter is the one door every write in the body passes
        through.  A knob aliased WITHOUT the save persists nothing — the
        state is right and the reload is empty."""
        assert wired["saves"][flat] == 1, (
            f"writing ``state.{flat}`` scheduled "
            f"{wired['saves'][flat]} saves, not 1.  The knob is aliased "
            f"but not persisted: § 7's lane never hears about the write.")

    @pytest.mark.parametrize("flat", _NOT_KNOBS)
    def test_a_non_knob_write_schedules_nothing(self, wired, flat):
        """The lane is the UI's knobs and nothing else.  A timer handle or
        a parsed file scheduling a save would write per poll tick."""
        assert wired["saves"][flat] == 0, (
            f"writing ``state.{flat}`` — not a uiPrefs knob — scheduled a "
            f"preferences save.")
