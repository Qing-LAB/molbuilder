"""`results.md` § 4 on the SPECTRA side, run: the spectra viewer's own
`transition()` -- a second implementation of the rules the trajectory
viewer's keeps (`test_trajectory_transition_js.py`), asked of its own
state, on no answer of the server's.

* a file switch releases the last file's requests and forgets it, resets
  the view and keeps the preferences; a run no longer live is not polled,
  a live one is (§ 4; `spectra.md` § 7);
* the page's own write of an answer: under its own file's name or not at
  all, one that names no file refused, after dispose nothing (§ 4.2) --
  the answer an opaque token here, as no line of this asks inside it;
* the legacy flat names and the buckets are one value, and a knob's write
  is saved (`spectra.md` § 7).

What the viewer makes of the server's ANSWERS -- whether it follows the run
a load or a poll names, a poll answered after the file was opened again, a
poll answering an older ask -- is fed what `/api/spectra/load` answered of a
real run: `test_the_viewers_follow_a_real_run_e2e.py`.

**These RUN the shipped functions**, lifted from `lib/spectra/core.js`
(`_node_esm.lifted`) and run against fakes at the edge -- the timers, the
abort controllers -- so what is asserted is what the viewer then holds.
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
#: What the viewer holds of the run a server answer named -- a token here:
#: no function these rows run reads inside it.
_HELD_RUN = {"held": "the run a server answer named"}


def _viewer(script, *, machine, path=_A, results="A", polling=True,
            run=None):
    """The spectra viewer's state -- ``machine``, showing ``path`` with
    ``results`` (a mark, or none), its poll timer running when ``polling``
    and a poll on the wire -- driven by ``script`` through the REAL
    `transition` and `dispose`; what it then holds.

    Faked at the edge: the timers and the abort controllers (each says
    when it is used), the listener registrar, and what dispose lets go of
    besides the state."""
    held = {"from": results} if results else None
    harness = f"""
        const _aborted = [];
        const _ac = (name) => ({{ abort: () => _aborted.push(name) }});
        let _polling = {json.dumps(polling)};
        globalThis.setInterval = () => {{ _polling = true; return 7; }};
        globalThis.clearInterval = () => {{ _polling = false; }};
        const WATCH_INTERVAL_MS = 2000;
        // What the poll timer runs: its ticks are not what these rows ask.
        function watchTick() {{}}
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
                         watchInFlight: true,
                         watchTimer: {json.dumps(7 if polling else None)},
                         watchErrors: 2 }},
            derived: {{}},
            vib: null,
        }};
        {lifted(MODULE, "    function transition(target, payload) {",
                "\n    function ")}
        {lifted(MODULE, "    function _abortViews() {", "\n    }")}
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
            threw,
        }}));
        }})();
    """
    return run_node([], harness)


@pytest.mark.parametrize("start,script,holds", [
    pytest.param(
        {"machine": "WATCHING", "run": _HELD_RUN},
        f'transition("LOADING", {{ path: "{_B}" }});',
        {"machine": "LOADING", "path": _B, "results": None, "run": None,
         "selectedMode": None, "prefs": _PREFS, "polling": False,
         "aborted": ["load", "watch"]},
        id="a file switch releases the last file and keeps the preferences"),
    pytest.param(
        {"machine": "WATCHING", "run": _HELD_RUN},
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
        {"machine": "WATCHING", "run": _HELD_RUN},
        f'handle.dispose(); transition("APPLY", {{ path: "{_A}", '
        f'results: {{ from: "late" }} }});',
        {"machine": "IDLE", "path": None, "results": None, "polling": False,
         "aborted": ["load", "watch"]},
        id="after dispose a late answer for the file writes nothing"),
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
    none."""
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

#: Each legacy flat name that is not a knob -> the bucket that owns its value
#: (`results.md` § 4).  The knobs are every name the viewer's `uiPrefs`
#: declares, read from the shipped module.
_NOT_KNOBS = [
    ("results",           "fileState"),
    ("selectedMode",      "viewState"),
    ("watchTimer",        "lifecycle"),
    ("watchInFlight",     "lifecycle"),
    ("watchAbort",        "lifecycle"),
    ("loadAbort",         "lifecycle"),
    ("watchErrors",       "lifecycle"),
]


@pytest.fixture(scope="module")
def wired():
    """Run the shipped `_wireBackcompatAliases` on a state whose `uiPrefs`
    is the shipped declaration, and report, for each knob it declares and
    each other legacy name, what a write of the flat name did.

    Everything it reaches outside itself is real or faked at the edge:
    `inspectorLifecycle.alias` is the module the core actually calls, the
    other buckets are bare, and `_prefsSchedule` is a counter -- the save
    the uiPrefs setters kick is itself a behaviour (`spectra.md` § 7), so it
    is observed rather than stubbed away.
    """
    prefs = (lifted(MODULE, "        uiPrefs: {", "\n        },")
             + "\n        },")
    body = lifted(MODULE, "(function _wireBackcompatAliases()",
                  "})();") + "})();"
    probe = """
const state = { fileState: {}, viewState: {}, lifecycle: {},
""" + prefs + """
};
let scheduled = 0;
function _prefsSchedule() { scheduled += 1; }
const root = globalThis;
const knobs = Object.keys(state.uiPrefs);
""" + body + """
const out = { knobs, roundTrip: {}, saves: {} };
const names = knobs.map((k) => [k, "uiPrefs"]).concat(%s);
for (const [flat, bucket] of names) {
    const before = scheduled;
    state[flat] = "v:" + flat;                     // write the LEGACY name
    out.roundTrip[flat] = (state[bucket][flat] === "v:" + flat)  // bucket sees it
                       && (state[flat] === "v:" + flat);         // and reads back
    out.saves[flat] = scheduled - before;
}
console.log(JSON.stringify(out));
""" % json.dumps([[f, b] for f, b in _NOT_KNOBS])
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
    The knobs are the ones the shipped `uiPrefs` declares, so a knob added
    there without its alias is caught the day it appears.
    """

    def test_the_legacy_name_and_the_bucket_are_one_value(self, wired):
        assert wired["knobs"], "the shipped uiPrefs declares no knob"
        apart = sorted(f for f, one in wired["roundTrip"].items() if not one)
        assert apart == [], (
            f"writing ``state.<name>`` did not land in its bucket for "
            f"{apart}.  The alias is gone, so the flat name and the bucket "
            f"are now two values and the render code reading one cannot "
            f"see the other.")

    def test_a_write_schedules_the_save_exactly_when_it_is_a_knob(
            self, wired):
        """`spectra.md` § 7: the viewer's knobs survive a reload, and the
        alias setter is the one door every write in the body passes
        through.  A knob aliased WITHOUT the save persists nothing — the
        state is right and the reload is empty.  And the lane is the UI's
        knobs and nothing else: a timer handle or a parsed file scheduling
        a save would write per poll tick."""
        want = {**{k: 1 for k in wired["knobs"]},
                **{f: 0 for f, _ in _NOT_KNOBS}}
        assert wired["saves"] == want, (
            "a knob's write must schedule one preferences save, and "
            "nothing else's any")
