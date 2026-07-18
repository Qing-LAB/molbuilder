"""Hide-frozen-atoms feature — source-text invariants.

The trajectory inspector's "Exclude frozen atoms" checkbox gates the
force-arrow computation.  The one surviving dependent computation must
read the toggle, or the feature silently ignores its promise:

  1. ``_buildArrowsForFrame`` — must skip frozen indices when on
     so force arrows don't draw on frozen atoms.

History: two other computations used to depend on this toggle.
``refreshForcesStatus`` (a "Showing N arrows / max |F|" readout) was
removed when the force overlay became a single MolView-owned switch —
the arrow count/max is redundant with the on-page force plot.
``applyHideFrozen`` (a 3Dmol overlay payload that hid frozen atoms in
the viewer) was retired in the MolView migration (task #34); atom
hiding is MolView's job.  So the arrow-builder filter below, plus the
change listener that re-runs it, are the whole contract now.

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


def test_hide_frozen_change_listener_rebuilds_arrows():
    """The change event on the hide-frozen checkbox must rebuild the
    force arrows (drawForces), which re-derives the overlay for the
    current frame with the frozen atoms filtered out via
    _buildArrowsForFrame.  Post-task-#34 that is the toggle's ONLY
    effect — MolView owns atom hiding in the viewer, so there is no
    separate overlay handler to fire."""
    src = MODULE.read_text()
    # The wiring is ``_on($("hide-frozen"), "change", drawForces)`` — a
    # bare handler reference (the old inline function that also called
    # applyHideFrozen is gone).  Accept either the bare reference or an
    # inline function body that calls drawForces.
    bare = re.search(
        r'_on\(\$\("hide-frozen"\)\s*,\s*"change"\s*,\s*drawForces\s*\)',
        src,
    )
    inline = re.search(
        r'_on\(\$\("hide-frozen"\)\s*,\s*"change"\s*,\s*'
        r'function\s*\(\s*\)\s*\{([^}]*)\}',
        src,
    )
    assert bare or (inline and "drawForces" in inline.group(1)), (
        "hide-frozen change listener does not rebuild the force arrows "
        f"(drawForces) in {MODULE}.  Toggling 'Exclude frozen atoms' "
        "would leave the previously-drawn arrows on frozen atoms until "
        "the next frame change."
    )
    # And it must NOT reach for the retired viewer-overlay handler.
    assert not (inline and "applyHideFrozen" in inline.group(1)), (
        "hide-frozen change listener still calls applyHideFrozen() — "
        "that viewer-overlay handler was retired in the MolView "
        "migration (task #34); atom hiding is MolView's job now."
    )
