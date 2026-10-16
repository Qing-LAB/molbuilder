"""The listener scope tears down exactly what it registered
(`lib/inspectors/lifecycle.js`), asked of the scope itself in Node: what it
attached, `disposeAll` takes back -- with a deferred teardown, past a broken
one, and a second call doing nothing.

Whether every viewer registers through it and drains it is asked of the live
page: a viewer opened and closed again and again leaves what one open and
close left (`test_inspector_registry_e2e.py::TestInspectorListenerTeardown`).
"""
from __future__ import annotations

from pathlib import Path

from tests._node_esm import run_node

STATIC = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static"
LIFECYCLE = STATIC / "lib" / "inspectors" / "lifecycle.js"

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
