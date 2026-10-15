"""The listener scope tears down exactly what it registered.

**Why this file exists.** `lib/inspectors/lifecycle.js` was extracted on
2026-08-23 because both inspector cores spelled the same scope out
byte-identically -- and the extraction moved the REGISTRATION and left each
core's old ``_cleanups`` array behind.  ``dispose()`` drained that array while
every listener sat in the new scope, so a mount/dispose cycle removed nothing:
19 listeners registered, 0 removed, on every mount of the spectra inspector.

The suite did not see it.  `tests/spectra/test_blueprint.py`'s dispose contract
pins that every registration goes THROUGH the helper -- which was still true;
what broke was the drain.  Only `test_inspector_registry_e2e.py` counts
add/remove pairs across a real mount, and that lane is six minutes and had not
run since before the extraction landed.

So this file asks the teardown question two ways, in the fast lane:

  * **behaviourally**, against the scope itself, in Node;
  * **structurally**, of each core -- the scope ``_on`` writes into must be the
    scope ``dispose()`` drains, and no core may keep a second registry beside
    it.  Two registries is how one of them becomes the one nobody drains.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._node_esm import run_node

STATIC = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static"
LIFECYCLE = STATIC / "lib" / "inspectors" / "lifecycle.js"

#: Every core that mounts through the shared scope.  A new inspector added
#: here inherits the whole file; that is the point of listing them.
CORES = ("lib/spectra/core.js", "lib/trajectory/core.js")


# --------------------------------------------------------------------- #
#  Behaviour: the scope itself                                          #
# --------------------------------------------------------------------- #

_TARGET_STUB = """
globalThis.window = globalThis;
// A minimal EventTarget that records what is attached and detached, so the
// assertions are about PAIRS rather than about calls we made ourselves.
globalThis.makeTarget = function () {
    const live = new Set();
    return {
        live,
        addEventListener(ev, h) { live.add(ev + ':' + h.name); },
        removeEventListener(ev, h) { live.delete(ev + ':' + h.name); },
    };
};
"""


def _scope_result(snippet: str):
    return run_node([LIFECYCLE], snippet, globals_js=_TARGET_STUB)


def test_disposeall_removes_every_listener_the_scope_registered():
    """The whole contract, in one sentence: after `disposeAll`, nothing the
    scope attached is still attached."""
    out = _scope_result("""
        const { listeners } = globalThis.molbuilder.inspectorLifecycle;
        const t = globalThis.makeTarget();
        const s = listeners();
        function a() {} function b() {} function c() {}
        s.on(t, 'click', a); s.on(t, 'input', b); s.on(t, 'change', c);
        const attached = t.live.size;
        s.disposeAll();
        console.log(JSON.stringify({attached, left: t.live.size}));
    """)
    assert out["attached"] == 3, "the scope did not attach what it was given"
    assert out["left"] == 0, (
        f"{out['left']} listener(s) survived disposeAll() -- a mount/dispose "
        f"cycle leaks them onto a dead DOM")


def test_a_deferred_teardown_runs_with_the_listeners():
    """`defer` exists so a core needs no second registry for the teardowns
    that are not listeners (a ResizeObserver to disconnect)."""
    out = _scope_result("""
        const { listeners } = globalThis.molbuilder.inspectorLifecycle;
        const t = globalThis.makeTarget();
        const s = listeners();
        let ran = 0;
        function h() {}
        s.on(t, 'click', h);
        s.defer(() => { ran += 1; });
        s.disposeAll();
        console.log(JSON.stringify({ran, left: t.live.size}));
    """)
    assert out["ran"] == 1 and out["left"] == 0


def test_one_broken_teardown_does_not_strand_the_rest():
    """A teardown that throws must not leave the ones behind it attached --
    otherwise one bad handler turns into a silent leak of everything else."""
    out = _scope_result("""
        const { listeners } = globalThis.molbuilder.inspectorLifecycle;
        const t = globalThis.makeTarget();
        const s = listeners();
        function a() {} function b() {}
        s.on(t, 'click', a);
        s.defer(() => { throw new Error('boom'); });
        s.on(t, 'input', b);
        s.disposeAll();
        console.log(JSON.stringify({left: t.live.size}));
    """)
    assert out["left"] == 0


def test_a_second_dispose_is_a_no_op():
    """The registry may call dispose twice; the scope must not throw."""
    out = _scope_result("""
        const { listeners } = globalThis.molbuilder.inspectorLifecycle;
        const t = globalThis.makeTarget();
        const s = listeners();
        function a() {}
        s.on(t, 'click', a);
        s.disposeAll();
        let threw = false;
        try { s.disposeAll(); } catch (_) { threw = true; }
        console.log(JSON.stringify({threw, left: t.live.size}));
    """)
    assert out["threw"] is False and out["left"] == 0


# --------------------------------------------------------------------- #
#  Structure: each core drains the scope it registers into              #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("core", CORES)
def test_no_core_registers_a_listener_outside_the_scope(core):
    """Not one bare `addEventListener` in either core.

    A direct `els.foo.addEventListener("click", h)` escapes the scope, so
    `dispose()` cannot reach it however correctly it drains -- and no
    teardown test can see it, because from the scope's side it never
    existed.  Reading the file settles it; running the card cannot.
    """
    js = (STATIC / core).read_text()
    adds = re.findall(r"\.addEventListener\(", js)
    assert len(adds) == 0, (
        f"{core} registers {len(adds)} listener(s) directly -- every one is "
        f"outside the lifecycle scope, so dispose() cannot remove it and no "
        f"teardown test can notice.  Route it through _on().")
    shared = (STATIC / "lib/inspectors/lifecycle.js").read_text()
    assert len(re.findall(r"\.addEventListener\(", shared)) == 1, (
        "the shared listener scope must hold exactly ONE registration site "
        "-- that single door is what makes the drain provably complete")
    assert len(re.findall(r"\b_on\(", js)) >= 5, (
        f"{core} has almost no _on() call sites left; if the wiring moved, "
        f"the absence check above has stopped meaning anything")


@pytest.mark.parametrize("core", CORES)
def test_no_core_keeps_a_second_teardown_registry(core):
    """One registry, or one of them is the one nobody drains.

    A local array of teardown closures beside the scope is exactly the shape
    the 2026-08-23 leak had.  Non-listener teardowns go through
    `_listeners.defer()`.
    """
    js = (STATIC / core).read_text()
    decls = re.findall(r"^\s*(?:const|let|var)\s+(_\w*cleanup\w*)\s*=\s*\[\]",
                       js, re.MULTILINE)
    assert decls == [], (
        f"{core} declares {decls} beside the shared listener scope -- two "
        f"registries, and dispose() can only drain one of them.  Use "
        f"_listeners.defer() for teardowns that are not listeners.")


# --------------------------------------------------------------------- #
#  The defect class, tree-wide                                          #
# --------------------------------------------------------------------- #

#: A container is DECLARED EMPTY here and then drained somewhere else.  Filling
#: it is the producer; ``pop``/``shift``/``delete`` are consumers, and counting
#: a consumer as proof of use is precisely what hid the 2026-08-23 leak --
#: `dispose()` draining an array nobody fills reads, to a grep, exactly like
#: an array in daily use.
_DECL_EMPTY = re.compile(
    r"(?:const|let|var)\s+(\w+)\s*=\s*(?:\[\s*\]|new (?:Map|Set)\(\s*\))\s*;")


