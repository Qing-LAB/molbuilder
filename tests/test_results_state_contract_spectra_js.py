"""**`results.md` § 4 on the SPECTRA side** — the same rules, a second
inspector, not a copy.

§ 4 is about "a mounted viewer", and there is more than one: the trajectory
inspector and this one hold the same four buckets (`fileState`, `viewState`,
`uiPrefs`, `lifecycle`), move through one `transition()`, and carry the same
two guards. Spectra adds `APPLY` — every write to `fileState` goes through it
— and an `IDLE` state trajectory has no use for.

**Why both files exist rather than one parametrized over two modules.** They
are two implementations that agree, and a test that ran the same assertions
against whichever module it was handed would pass while one of them drifted
into the other's shape. The rule is *both inspectors obey § 4*, and the
honest way to check it is twice.

**Read as SOURCE-PINNING**, with the limitation the sibling states: this
greps the module for structure rather than running it, so it catches a
refactor that removes a guard and not one that keeps the shape and breaks the
behaviour.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests._node_esm import run_node


_STATIC = (Path(__file__).resolve().parent.parent
           / "molbuilder" / "web" / "static")
_LIB = _STATIC / "lib"


@pytest.fixture(scope="module")
def core_body():
    return (_LIB / "spectra" / "core.js").read_text()


# --------------------------------------------------------------------- #
#  Spectra: bucketed state shape                                        #
# --------------------------------------------------------------------- #


class TestBucketedStateShape:
    """``state`` carries the same five buckets as trajectory + a
    ``machine`` field.  Form / calculation fields (schema,
    ...) stay at the top level of ``state`` outside the buckets;
    they're owned by a different (workspace) contract."""

    def test_state_has_machine_field(self, core_body):
        assert re.search(
            r"machine\s*:\s*[\"']IDLE[\"']",
            core_body,
        ), ("spectra/core.js state object no longer initializes "
            "machine to 'IDLE'.  The state machine has no resting "
            "state; the contract § 2 transitions table has no "
            "starting point.")

    @pytest.mark.parametrize("bucket", [
        "fileState", "viewState", "uiPrefs", "lifecycle", "derived",
    ])
    def test_state_carries_each_bucket(self, core_body, bucket):
        m = re.search(r"const\s+state\s*=\s*\{", core_body)
        assert m is not None, "state literal not found"
        window = core_body[m.end(): m.end() + 6000]
        assert re.search(
            r"^\s+" + bucket + r"\s*:\s*\{", window, re.MULTILINE,
        ), (f"spectra/core.js state object no longer carries the "
            f"``{bucket}`` bucket (`results.md` § 4).  Both inspectors "
            f"five-bucket partition.")


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
def wired(core_body):
    """Run the shipped `_wireBackcompatAliases` and report what it did.

    Everything it reaches outside itself is real or faked at the edge:
    `inspectorLifecycle.alias` is the module the core actually calls,
    `state` is a bare object with the five buckets, and `_prefsSchedule` is
    a counter -- the save the uiPrefs setters kick is itself a behaviour
    (`spectra.md` § 7), so it is observed rather than stubbed away.
    """
    i = core_body.index("(function _wireBackcompatAliases()")
    body = core_body[i: core_body.index("})();", i) + len("})();")]
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
    return run_node([_LIB / "inspectors" / "lifecycle.js"], probe)


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


# --------------------------------------------------------------------- #
#  Spectra: transition() orchestrator                                   #
# --------------------------------------------------------------------- #


class TestTransitionOrchestrator:
    """``transition(target, payload)`` is the SINGLE entry-point for
    state-machine transitions.  Mirrors trajectory's transition()."""

    def test_transition_function_exists(self, core_body):
        assert re.search(
            r"function\s+transition\s*\(\s*target\s*,\s*payload\s*\)",
            core_body,
        ), ("spectra/core.js no longer defines the transition() "
            "orchestrator.")

    @pytest.mark.parametrize("state_name", [
        "LOADING", "IDLE", "LOADED", "WATCHING", "ERROR", "APPLY",
    ])
    def test_each_branch_present(self, core_body, state_name):
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']" + state_name
            + r"[\"']\s*\)\s*\{",
            core_body,
        )
        assert m is not None, (
            f"spectra/core.js transition() has no '{state_name}' "
            f"branch.  Per contract § 2 all six targets MUST be "
            f"implemented.")

    def test_transition_loading_aborts_controllers(self, core_body):
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']LOADING[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "loadAbort" in body and "watchAbort" in body, (
            "transition('LOADING') doesn't abort both controllers.")
        assert "abort()" in body

    def test_transition_idle_clears_filestate(self, core_body):
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']IDLE[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "fileState.path" in body and "= null" in body
        assert "fileState.results" in body

    def test_transition_watching_starts_timer(self, core_body):
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']WATCHING[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert re.search(r"setInterval\s*\(\s*watchTick", body), (
            "transition('WATCHING') doesn't start the watchTick "
            "interval.  Contract § 3 matrix row 'fetch resolved, "
            "run ongoing' violated.")

    def test_transition_loaded_stops_timer(self, core_body):
        m = re.search(
            r"if\s*\(\s*target\s*===\s*[\"']LOADED[\"']\s*\)\s*\{"
            r"(.+?)return\s*;",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "clearInterval" in body, (
            "transition('LOADED') doesn't clear the watchTimer.  "
            "A finished run keeps polling forever.")

# --------------------------------------------------------------------- #
#  Refresh listener wired ONCE at mount                                 #
# --------------------------------------------------------------------- #


class TestRefreshListenerWiredOnce:
    """PR 3: the EVENT_REFRESH_REQUESTED listener is wired ONCE at
    mount via _wireRefreshListener.  Pre-PR-3 spectra didn't listen
    for the event at all -- file-picker Refresh fired into the
    void."""

    def test_wire_function_exists(self, core_body):
        assert re.search(
            r"function\s+_wireRefreshListener\s*\(\s*\)",
            core_body,
        ), ("spectra/core.js doesn't define _wireRefreshListener.  "
            "Refresh button is unwired -- contract § 5 violated.")

    def test_wire_function_called_at_mount(self, core_body):
        """Called BEFORE the mount return so it fires exactly once
        per mount."""
        assert "_wireRefreshListener();" in core_body, (
            "_wireRefreshListener is defined but never called.  "
            "Refresh button is unwired.")

    def test_refresh_handler_calls_loadByPath(self, core_body):
        """Refresh handler MUST call loadByPath (the same code path
        as Load-once / file-switch).  Per contract § 5 Refresh =
        file-switch with current path."""
        m = re.search(
            r"function\s+_wireRefreshListener\s*\(\s*\)\s*\{(.+?)\n\s{4}\}",
            core_body, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "loadByPath" in body, (
            "_wireRefreshListener doesn't call loadByPath.  Refresh "
            "would skip the LOADING reset matrix.")


# --------------------------------------------------------------------- #
#  loadByPath + dispose route through transition()                      #
# --------------------------------------------------------------------- #


class TestEntryPointsRouteThroughTransition:
    """The public entry points (loadByPath, stopWatch, dispose) MUST route
    state mutations through transition()."""


    def test_dispose_calls_transition_idle(self, core_body):
        # The dispose handler is in the return-object literal.
        m = re.search(
            r"dispose\s*\(\s*\)\s*\{(.+?)\n\s{8}\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "dispose method not found"
        body = m.group(1)
        assert re.search(
            r"transition\s*\(\s*[\"']IDLE[\"']", body,
        ), ("dispose doesn't call transition('IDLE').  fileState "
            "leaks across remounts; the audit § 1 'dispose leaks "
            "state.data' bug class is back for spectra.")
