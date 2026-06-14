"""L2 Node test: hide-frozen-row visibility transition.

Pins the contract that ``refreshHideFrozenAvailability`` (in
``lib/trajectory/core.js``) drives the ``#hide-frozen-row``
checkbox's visibility based on ``state.data.runtime_info.
frozen_atoms``:

  * non-empty array  → row.hidden = false
  * empty array      → row.hidden = true
  * missing field    → row.hidden = true
  * row missing      → silent no-op (defensive)

The original 2026-06-14 user-reported bug had three layers
contributing — the CSS ``[hidden]`` override (separate test:
``test_css_hidden_attribute_audit.py``), the SIESTA parser not
populating ``frozen_atoms`` without a sidecar (separate test:
``test_siesta_fdf_constraints.py``), and the JS state-machine that
drives the visibility from ``frozen_atoms``.  This file covers the
third layer end-to-end so a regression in any of the four code
paths above surfaces immediately at CI time.

The trajectory module is an IIFE — the function under test is
closure-private.  We extract its source via regex (same pattern as
``test_trajectory_status_js.py::_classifyStopReason``) and run it
under Node with stubbed ``$`` + ``state`` + ``applyHideFrozen``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _extract_fn_source(name: str) -> str:
    """Extract the literal source of a top-level function inside
    the trajectory module's IIFE.  Walks brace-depth from
    ``function <name>`` to the matching ``}`` at depth 0.

    Same pattern the embed-handle invariant tests use; works
    because the module sticks to a consistent 4-space-indented
    top-level layout.
    """
    src = MODULE.read_text(encoding="utf-8")
    start = src.find(f"function {name}(")
    if start < 0:
        pytest.fail(
            f"Could not find ``function {name}(`` in "
            f"{MODULE.relative_to(ROOT)}.  Either the function "
            f"was renamed or this test's parser needs updating."
        )
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
                return src[start:i + 1]
        i += 1
    pytest.fail(f"Unbalanced braces in function {name}")


def _run_with_state(
    frozen_atoms: list[int] | None,
    *,
    row_exists: bool = True,
    cb_starts_checked: bool = False,
) -> dict[str, object]:
    """Build a Node program that stubs the closure dependencies
    (``$``, ``state``, ``applyHideFrozen``) the way the trajectory
    module's IIFE would have populated them, then evaluates the
    extracted ``_frozenSet`` + ``refreshHideFrozenAvailability``
    sources and reports the resulting ``row.hidden`` /
    ``cb.checked`` state plus a count of how many times
    ``applyHideFrozen`` was called.

    Returns the parsed JSON observation.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    fn_frozen_set = _extract_fn_source("_frozenSet")
    fn_refresh    = _extract_fn_source("refreshHideFrozenAvailability")

    # Build runtime_info.  ``None`` → no field; empty list → field
    # is present but empty (different code path through _frozenSet).
    runtime_info: dict[str, object] = {}
    if frozen_atoms is not None:
        runtime_info["frozen_atoms"] = frozen_atoms

    state_payload = {
        "data": {
            "runtime_info": runtime_info,
        },
    }

    bootstrap = f"""
        // ===== Closure dependencies the trajectory module would
        //       have populated; we stub them here. =====
        const state = {json.dumps(state_payload)};
        let _applyCalls = 0;
        function applyHideFrozen() {{ _applyCalls += 1; }}

        // Mock DOM: a single ``getElementById``-like lookup.  Only
        // the two ids the function touches need stubs.
        const dom = {{
            "hide-frozen-row": {json.dumps(
                {"hidden": False, "_id": "hide-frozen-row"}
                if row_exists else None
            )},
            "hide-frozen":     {json.dumps(
                {"checked": cb_starts_checked, "_id": "hide-frozen"}
            )},
        }};
        function $(id) {{ return dom[id] || null; }}

        // ===== Extracted closure-private functions =====
        {fn_frozen_set}

        {fn_refresh}

        // ===== Drive the test scenario =====
        refreshHideFrozenAvailability();
        console.log(JSON.stringify({{
            row_hidden:     dom["hide-frozen-row"]
                              ? dom["hide-frozen-row"].hidden : null,
            cb_checked:     dom["hide-frozen"].checked,
            apply_calls:    _applyCalls,
        }}));
    """

    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- #
#  Contract: non-empty frozen_atoms → row visible                        #
# --------------------------------------------------------------------- #


def test_non_empty_frozen_atoms_shows_the_row():
    """A trajectory whose runtime_info.frozen_atoms carries 1+
    indices is the case the toggle is FOR — the row reveals so
    the user can click the checkbox."""
    out = _run_with_state(frozen_atoms=[3, 5, 7])
    assert out["row_hidden"] is False
    # The function doesn't touch the checkbox or call
    # applyHideFrozen on this branch (the toggle just becomes
    # available; the user activates it themselves).
    assert out["cb_checked"] is False
    assert out["apply_calls"] == 0


def test_non_empty_with_single_atom_shows_the_row():
    """Boundary: even one frozen atom counts."""
    out = _run_with_state(frozen_atoms=[42])
    assert out["row_hidden"] is False


# --------------------------------------------------------------------- #
#  Contract: empty frozen_atoms → row hidden                             #
# --------------------------------------------------------------------- #


def test_empty_frozen_atoms_hides_the_row():
    """Empty array → toggle has nothing to gate; hide the row so
    the user doesn't click a non-functional control."""
    out = _run_with_state(frozen_atoms=[])
    assert out["row_hidden"] is True
    assert out["apply_calls"] == 0


def test_missing_frozen_atoms_field_hides_the_row():
    """The trajectory has no frozen_atoms field at all (no sidecar,
    no .fdf with Constraints).  Same outcome as empty: hide."""
    out = _run_with_state(frozen_atoms=None)
    assert out["row_hidden"] is True


# --------------------------------------------------------------------- #
#  Contract: row missing from DOM → silent no-op                         #
# --------------------------------------------------------------------- #


def test_row_missing_from_dom_silent_no_op():
    """If the partial DOM hasn't injected ``#hide-frozen-row`` yet
    (defensive guard), the function returns without throwing —
    it must not crash trajectory mount."""
    out = _run_with_state(frozen_atoms=[1, 2, 3], row_exists=False)
    # No row → row_hidden in our mock is null (the element didn't
    # exist).  The fact that the function returned at all (i.e.,
    # the Node script didn't fail) is the assertion.
    assert out["row_hidden"] is None


# --------------------------------------------------------------------- #
#  Contract: when row hides, a previously-checked checkbox resets        #
# --------------------------------------------------------------------- #


def test_hiding_the_row_clears_a_stale_checked_state():
    """Edge case: user had hide-frozen ON for a previous trajectory
    with frozen atoms.  They open a different trajectory with no
    frozen atoms.  Hide the row AND uncheck the checkbox so the
    overlay state matches reality (and apply the hide overlay's
    empty payload via applyHideFrozen so the viewer clears any
    lingering hidden-atoms overlay)."""
    out = _run_with_state(
        frozen_atoms=[], cb_starts_checked=True,
    )
    assert out["row_hidden"] is True
    assert out["cb_checked"] is False, (
        "checkbox must be unchecked when the row hides — "
        "otherwise a remount with new data would keep the "
        "stale checked state visible in DOM"
    )
    assert out["apply_calls"] == 1, (
        "applyHideFrozen() must fire on uncheck so the embed's "
        "overlay clears the previously-hidden atoms"
    )


def test_already_unchecked_does_not_trigger_apply():
    """Symmetric case to the cleanup test: when the row hides AND
    the checkbox was already unchecked, the function shouldn't
    spuriously fire applyHideFrozen — no overlay change is needed.
    """
    out = _run_with_state(
        frozen_atoms=[], cb_starts_checked=False,
    )
    assert out["row_hidden"] is True
    assert out["cb_checked"] is False
    assert out["apply_calls"] == 0
