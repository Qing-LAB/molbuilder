"""Hide-frozen-atoms feature — source-text invariants.

The trajectory inspector's "Hide frozen atoms" checkbox has three
dependent computations.  Each must read the toggle, or the feature
silently ignores half its promise:

  1. ``_buildArrowsForFrame`` — must skip frozen indices when on
     so force arrows don't draw on hidden atoms.
  2. ``refreshForcesStatus`` — must filter the reported max |F|
     and arrow count by the same skip rule, or the status line
     keeps reporting the all-atom max after the user hides the
     frozen ones (the 2026-06-14 user-reported bug).
  3. ``applyHideFrozen`` — must dispatch the overlay payload that
     hides the frozen atoms in the 3Dmol viewer.

Why source-text, not a runtime test:
* The functions live inside an IIFE and read closure-private
  state + the DOM.  A runtime harness would need stubs for $(),
  state.data, _handle.setOverlays, _handle.setAnimation, plus
  full force-frame fixtures — substantial setup to catch a
  "function forgot to read the toggle" regression.
* The original bug was a function that read its OWN state
  correctly but ignored the toggle entirely.  Source-text grep
  catches this directly: assert each function's body contains
  the toggle id (``hide-frozen``) AND the frozen-set helper
  (``_frozenSet``) AND the skip pattern (``frozen.has(i)``).

Per docs/protocols/test-strategy.md § 5 (source-text invariants):
canonical use case.  The contract is "every code path that depends
on this state actually reads this state"; verifying it by grep
over the function bodies is the right level of abstraction.

Filed in response to task #387 (the bug fix) — these tests pin
the contract going forward so a future refactor that splits one
of these functions and forgets the toggle check fails CI loudly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/core.js"

# Token the toggle's id is rendered as in the partial template,
# matched verbatim by ``$("hide-frozen")`` calls inside the module.
_TOGGLE_ID = "hide-frozen"
# The closure-private helper that returns the Set<number> of frozen
# atom indices (or null when no sidecar provided frozen_atoms).
_FROZEN_SET_HELPER = "_frozenSet"


def _extract_fn_body(name: str) -> str:
    """Return the body source of the named top-level function inside
    the trajectory module's IIFE.

    The module is a single big IIFE; functions are indented one
    level (4 spaces).  This walks from ``function <name>`` to the
    matching closing brace at the same indentation.  Plain-text
    bracket-matching is brittle in general, but the module sticks
    to a consistent 4-space-indented top-level + a final ``}``
    that lines up with the ``function`` keyword — verified by the
    same trick used in ``tests/test_trajectory_status_js.py``.
    """
    src = MODULE.read_text()
    needle = f"function {name}("
    start = src.find(needle)
    if start < 0:
        pytest.fail(
            f"Could not find ``function {name}(`` in "
            f"{MODULE}.  Either the function was renamed or this "
            f"test's parser needs updating."
        )
    # Walk forward to find the opening brace of the function body
    # and then match braces until depth returns to 0.
    open_brace = src.find("{", start)
    if open_brace < 0:
        pytest.fail(f"No opening brace after function {name}")
    depth = 0
    i = open_brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace + 1:i]
        i += 1
    pytest.fail(f"Unbalanced braces in function {name}")


# A function "honours the hide-frozen toggle" when its body
# references the toggle id, calls the _frozenSet helper, and
# contains the canonical skip pattern that filters frozen indices.
# The skip pattern is the small idiom this codebase uses everywhere
# the toggle gates a loop: ``if (frozen && frozen.has(i)) continue``.
_SKIP_PATTERN = re.compile(
    r"frozen\s*&&\s*frozen\.has\(",
)


def _assert_honours_hide_frozen(fn_name: str, body: str) -> None:
    """Three checks: the body reads the toggle id, the body calls
    the frozen-set helper, and the body contains the
    skip-frozen-indices idiom.  Together these guarantee the
    function actually responds to the toggle instead of computing
    over every atom unconditionally."""
    assert _TOGGLE_ID in body, (
        f"{fn_name}() does not reference the {_TOGGLE_ID!r} "
        f"checkbox id.  When the toggle is on, this function "
        f"would compute over every atom — the exact regression "
        f"class this test exists to catch (see 2026-06-14 "
        f"refreshForcesStatus bug fix in commit 65e8246)."
    )
    assert _FROZEN_SET_HELPER in body, (
        f"{fn_name}() does not call {_FROZEN_SET_HELPER}().  "
        f"Without it, the function has no way to know which "
        f"indices are frozen even if it reads the toggle."
    )
    assert _SKIP_PATTERN.search(body), (
        f"{fn_name}() does not contain the canonical "
        f"``if (frozen && frozen.has(i)) continue`` skip "
        f"idiom.  Other patterns may work but the test pins "
        f"the shared idiom so refactors stay consistent."
    )


def test_buildArrowsForFrame_honours_hide_frozen():
    """The arrow-rendering path skips frozen atoms when the toggle
    is on so the viewer doesn't draw vectors on atoms the overlay
    has hidden."""
    body = _extract_fn_body("_buildArrowsForFrame")
    _assert_honours_hide_frozen("_buildArrowsForFrame", body)


def test_refreshForcesStatus_honours_hide_frozen():
    """The status-line max-|F| + arrow-count computation skips
    frozen atoms when the toggle is on so the reported values
    describe what's actually drawn (not the whole frame).

    Caught the 2026-06-14 regression where this function ignored
    the toggle entirely: viewer arrows hid correctly but the
    status line still said ``max |F| = <frozen atom's force>`` —
    the toggle looked like a no-op on the readout."""
    body = _extract_fn_body("refreshForcesStatus")
    _assert_honours_hide_frozen("refreshForcesStatus", body)


def test_applyHideFrozen_dispatches_overlay_payload():
    """The 3Dmol-overlay handler that actually hides the atoms.
    Must read the toggle, look up the frozen set, and call the
    embed's setOverlays.  The contract is documented in the
    function's own header comment + the embed's overlay schema."""
    body = _extract_fn_body("applyHideFrozen")
    assert _TOGGLE_ID in body, (
        f"applyHideFrozen() does not reference the {_TOGGLE_ID!r} "
        f"checkbox id.  Without that, toggling on does nothing."
    )
    assert _FROZEN_SET_HELPER in body, (
        f"applyHideFrozen() does not call {_FROZEN_SET_HELPER}().  "
        f"Without that, the overlay payload has no indices to hide."
    )
    assert "setOverlays" in body, (
        f"applyHideFrozen() does not call setOverlays().  Without "
        f"that, the 3Dmol viewer never receives the hide-atoms "
        f"payload and the visual effect is lost."
    )


def test_hide_frozen_change_listener_invokes_both_handlers():
    """The change event on the hide-frozen checkbox must fire BOTH
    applyHideFrozen (the viewer overlay) AND drawForces (the arrow
    rebuild that picks up the frozen filter via
    _buildArrowsForFrame).  If only one fires, the user sees the
    toggle take partial effect — e.g., arrows suppress but atoms
    stay visible, or atoms hide but arrows still draw on them."""
    src = MODULE.read_text()
    # Find the change listener wired to the hide-frozen checkbox.
    # The codebase uses the ``_on($("hide-frozen"), "change", ...)``
    # idiom; pick out the handler body.
    match = re.search(
        r'_on\(\$\("hide-frozen"\)\s*,\s*"change"\s*,\s*'
        r'function\s*\(\s*\)\s*\{([^}]*)\}',
        src,
    )
    assert match, (
        "Could not locate the hide-frozen change listener in "
        f"{MODULE}.  If the wiring was rewritten, update this "
        "test's parser; the contract that both handlers fire on "
        "toggle remains load-bearing."
    )
    handler = match.group(1)
    assert "applyHideFrozen" in handler, (
        "hide-frozen change listener does not call applyHideFrozen() "
        "— the 3Dmol viewer would never hide the frozen atoms."
    )
    assert "drawForces" in handler, (
        "hide-frozen change listener does not call drawForces() — "
        "the force arrows would not rebuild, so previously drawn "
        "arrows on frozen atoms would linger until the next frame "
        "change."
    )
