"""L2 Node test: ``applyHideFrozen`` overlay-payload CONTRACT.

History (E1 demote, 2026-06-14 round-3 follow-up)
=================================================

The original E1 test (``test_hide_frozen_toggle_e2e.py``) was L5 --
mounted the full trajectory inspector in Playwright, wrote a real
SIESTA .out fixture, captured the embed handle via a wrapper on
``window.molbuilder.viewer.embed``, then clicked the checkbox and
asserted ``handle.getOverlays()`` returned the right shape.

Per the round-3 R3-B test-pyramid audit + ``docs/protocols/test-
strategy.md`` § 4: the click-handler → overlay-payload contract is
a pure JS function chain that L2 can drive in Node without a
browser.  Only the "real script-tag order + DOM mount" half is
e2e-load-bearing.

This file is the L2 demote.  Extracts ``applyHideFrozen`` from
``lib/trajectory/core.js`` and drives it with synthetic state +
mocked ``_handle.setOverlays``, asserting the call shape directly.
Runs in <1s vs the L5's ~10s of Chromium startup.

What L5 still pins
==================

``tests/test_hide_frozen_toggle_e2e.py`` keeps the "the chain
actually mounts under real script-tag order + a real fetch returns
data" smoke -- but no longer the per-click overlay shape (which
this L2 owns).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.module


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _have_node():
    return shutil.which("node") is not None


pytestmark = pytest.mark.skipif(
    not _have_node(),
    reason="node not installed; L2 Node-driven test",
)


def _extract_fn_source(name: str) -> str:
    """Extract the source of a named function from core.js by
    brace-counting.  Same shape as test_hide_frozen_visibility_
    transition_js.py's helper but caches the file read.
    """
    src = _MODULE.read_text(encoding="utf-8")
    needle = f"    function {name}("
    start = src.find(needle)
    if start < 0:
        pytest.fail(
            f"Could not find ``    function {name}(`` in "
            f"{_MODULE.relative_to(_ROOT)}.  Was the function "
            f"renamed or moved?"
        )
    open_brace = src.find("{", start)
    depth = 0
    i = open_brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    pytest.fail(f"Unbalanced braces in function {name}")


def _run_apply_hide_frozen(
    frozen_indices,
    *,
    cb_checked: bool,
    cb_present: bool = True,
    handle_present: bool = True,
) -> dict:
    """Invoke ``applyHideFrozen()`` in a Node bootstrap with the
    given preconditions; return what the function did.

    Returns a dict with keys::

      * ``setOverlays_calls``: list of objects (one per setOverlays
        call), each carrying ``atoms`` -- a sanitized copy of the
        spec passed to setOverlays.  Used to assert call shape.
      * ``rebuildInspectAtomList_calls``: int.  Pre-G6 this was 0;
        post-G6 the function MUST call it (the atom list re-renders
        in lockstep with the overlay).
    """
    fn_src = _extract_fn_source("applyHideFrozen")
    handle_init = (
        "{ setOverlays: function (spec) { _calls.push(spec); } }"
        if handle_present else "null"
    )
    cb_init = (
        "{ checked: " + ("true" if cb_checked else "false") + " }"
        if cb_present else "null"
    )
    frozen_json = json.dumps(frozen_indices)

    bootstrap = f"""
        const _calls = [];
        let _rebuildCount = 0;
        const _handle = {handle_init};
        const _cb = {cb_init};
        function $(id) {{
            if (id === 'hide-frozen') return _cb;
            return null;
        }}
        function _frozenSet() {{
            const f = {frozen_json};
            if (!Array.isArray(f) || !f.length) return null;
            return new Set(f);
        }}
        function rebuildInspectAtomList() {{ _rebuildCount += 1; }}

        {fn_src}

        applyHideFrozen();

        // Serialise the call list as plain data so JSON.stringify
        // can round-trip it (Set -> Array).  We only care about
        // the atoms[] payload + each atom's indices + style.
        const sanitised = _calls.map(function (c) {{
            return {{
                atoms: (c && c.atoms ? c.atoms : []).map(function (a) {{
                    return {{
                        indices: a.indices ? Array.from(a.indices) : null,
                        style:   a.style   ? a.style              : null,
                    }};
                }}),
            }};
        }});
        console.log(JSON.stringify({{
            setOverlays_calls: sanitised,
            rebuildInspectAtomList_calls: _rebuildCount,
        }}));
    """
    proc = subprocess.run(
        ["node", "--input-type=commonjs", "-e", bootstrap],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- #
#  Happy-path: checkbox checked + frozen indices present                 #
# --------------------------------------------------------------------- #


def test_checkbox_checked_with_frozen_calls_setoverlays_with_hidden_true():
    """The canonical case: parser surfaced frozen_atoms, user
    ticked the checkbox.  setOverlays MUST be called with a single
    entry whose indices match the frozen set AND whose style
    carries ``hidden: true``.
    """
    out = _run_apply_hide_frozen([2, 5, 9], cb_checked=True)
    assert len(out["setOverlays_calls"]) == 1, (
        f"applyHideFrozen with checked + frozen MUST call "
        f"setOverlays exactly once; got "
        f"{len(out['setOverlays_calls'])}"
    )
    spec = out["setOverlays_calls"][0]
    assert len(spec["atoms"]) == 1, (
        f"setOverlays atoms[] must have one entry; got "
        f"{spec['atoms']!r}"
    )
    entry = spec["atoms"][0]
    assert sorted(entry["indices"]) == [2, 5, 9], (
        f"overlay indices must match the frozen set; got "
        f"{entry['indices']!r}"
    )
    assert entry["style"] == {"hidden": True}, (
        f"overlay style must be {{hidden: true}}; got "
        f"{entry['style']!r}"
    )


# --------------------------------------------------------------------- #
#  Bail paths: setOverlays receives an empty atoms[] (clear filter)      #
# --------------------------------------------------------------------- #


def test_no_frozen_indices_clears_overlay():
    """When the parser surfaced ZERO frozen indices (no constraints
    block, no sidecar), applyHideFrozen MUST clear any prior
    overlay by calling setOverlays({atoms: []}).  Without this the
    embed could keep a stale hidden-set from a prior trajectory."""
    out = _run_apply_hide_frozen([], cb_checked=True)
    assert len(out["setOverlays_calls"]) == 1
    spec = out["setOverlays_calls"][0]
    assert spec["atoms"] == [], (
        f"empty frozen set must clear overlay; got {spec['atoms']!r}"
    )


def test_checkbox_unchecked_clears_overlay():
    """User explicitly un-ticked the checkbox: clear the overlay
    even though the parser surfaced frozen indices.  Without this
    the embed would keep atoms hidden after the user said 'show
    them'."""
    out = _run_apply_hide_frozen([0, 1], cb_checked=False)
    assert len(out["setOverlays_calls"]) == 1
    spec = out["setOverlays_calls"][0]
    assert spec["atoms"] == [], (
        f"unchecked must clear overlay; got {spec['atoms']!r}"
    )


def test_missing_checkbox_clears_overlay():
    """Defensive: if the row got removed from the DOM (e.g. an
    inspector remount without the partial), applyHideFrozen must
    fall to the clear-overlay path rather than throw."""
    out = _run_apply_hide_frozen([0, 1], cb_checked=False, cb_present=False)
    assert len(out["setOverlays_calls"]) == 1
    spec = out["setOverlays_calls"][0]
    assert spec["atoms"] == []


def test_missing_handle_is_a_safe_noop():
    """Pre-mount: the embed handle isn't created yet.  Function
    MUST bail without touching setOverlays (would crash on null)."""
    out = _run_apply_hide_frozen([0, 1], cb_checked=True, handle_present=False)
    assert out["setOverlays_calls"] == [], (
        f"with no handle, applyHideFrozen must be a no-op; got "
        f"{out['setOverlays_calls']!r}"
    )


# --------------------------------------------------------------------- #
#  G6 lockstep: rebuildInspectAtomList is called every time              #
# --------------------------------------------------------------------- #


def test_rebuild_inspect_atom_list_fires_when_handle_present():
    """G6 contract: when the inspector IS mounted (handle truthy),
    the atom-list re-renders in lockstep with the overlay so
    frozen rows don't stay visible+clickable after the user ticks
    the checkbox.

    When the handle is NULL (pre-mount, between dispose and
    remount, etc.) the function bails before the rebuild --
    there's no atom-list DOM to rebuild yet either.  That branch
    is exercised by ``test_missing_handle_is_a_safe_noop``.

    Pre-G6 applyHideFrozen NEVER called rebuildInspectAtomList;
    this test catches a regression that drops the call from the
    handle-present path.
    """
    for frozen, checked in [
        ([0, 1], True),    # canonical hide
        ([],     True),    # clear (empty frozen)
        ([0, 1], False),   # clear (unchecked)
    ]:
        out = _run_apply_hide_frozen(
            frozen, cb_checked=checked, handle_present=True,
        )
        assert out["rebuildInspectAtomList_calls"] == 1, (
            f"applyHideFrozen with handle present MUST trigger one "
            f"atom-list rebuild (G6 contract).  Inputs: "
            f"frozen={frozen} checked={checked}; got "
            f"{out['rebuildInspectAtomList_calls']} rebuilds."
        )


def test_missing_handle_skips_rebuild_too():
    """The complement of the test above: the handle-null bail at
    the TOP of applyHideFrozen also skips the rebuild.  Pin so a
    future refactor doesn't move the rebuild call ABOVE the
    handle-null guard (which would crash because the atom list
    can't render without the embed being ready)."""
    out = _run_apply_hide_frozen([0, 1], cb_checked=True, handle_present=False)
    assert out["rebuildInspectAtomList_calls"] == 0, (
        f"applyHideFrozen with handle=null must bail BEFORE the "
        f"rebuild; got {out['rebuildInspectAtomList_calls']} "
        f"rebuilds."
    )


# --------------------------------------------------------------------- #
#  Source-text guard: the load-bearing pattern survives refactors        #
# --------------------------------------------------------------------- #


def test_apply_hide_frozen_invokes_set_overlays():
    """Source-text guard.  Refactors that rename setOverlays /
    replace it with a different mechanism (e.g. CSS class-toggle)
    fail this loudly; the wire to the embed MUST be via the
    documented method."""
    fn_src = _extract_fn_source("applyHideFrozen")
    assert "_handle.setOverlays" in fn_src, (
        "applyHideFrozen must wire its result through "
        "``_handle.setOverlays(...)``; that's the embed's "
        "documented hide-atoms primitive."
    )
    assert "rebuildInspectAtomList" in fn_src, (
        "applyHideFrozen must call rebuildInspectAtomList() so the "
        "Inspect atom-list re-renders in lockstep with the overlay "
        "(G6 contract)."
    )
