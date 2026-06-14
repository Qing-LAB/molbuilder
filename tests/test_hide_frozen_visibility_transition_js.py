"""L2 Node test: hide-frozen-row visibility CONTRACT.

2026-06-14 contract change: the "Hide frozen atoms" row is ALWAYS
visible.  UI presence is not data.  The control's PRESENCE is a
stable affordance the user can build muscle memory around; only its
EFFECT depends on data (when there are no frozen atoms, clicking the
checkbox is a no-op).

Pre-fix this test pinned the OPPOSITE behaviour -- the row was
hidden when ``frozen_atoms`` was empty / missing.  That contract
caused the row to silently vanish every time the parser failed to
surface indices (which it did on every wrapper-launched .out
because of the filename-pairing bug) and broke users' muscle memory
on constraint-free runs.  See user feedback 2026-06-14: ``it is a
fucking UI`` -- meaning UI affordances should not appear and
disappear based on data shape.

What ``refreshHideFrozenAvailability`` does now:
  * Row visibility is template-owned: ``hidden`` attribute is gone.
    The JS NEVER touches ``row.hidden``.
  * When ``frozen_atoms`` is empty / missing AND the checkbox is
    checked, un-check it (so the user doesn't see a stale "filter
    on" state with nothing to filter) and call ``applyHideFrozen()``
    once to clear any leftover overlay.
  * When ``frozen_atoms`` is non-empty, the function is a no-op
    (the user's checkbox state is preserved).
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
TEMPLATE = (
    ROOT / "molbuilder/web/templates/_trajectory_inspector.html"
)


def _extract_fn_source(name: str) -> str:
    src = MODULE.read_text(encoding="utf-8")
    needle = f"    function {name}("
    start = src.find(needle)
    if start < 0:
        pytest.fail(
            f"Could not find ``    function {name}(`` in "
            f"{MODULE.relative_to(ROOT)}."
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


def _run_with_frozen(
    frozen,
    *,
    row_present: bool = True,
    cb_present: bool = True,
    cb_checked: bool = False,
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    fn_src = _extract_fn_source("refreshHideFrozenAvailability")
    bootstrap = f"""
        const row = {("{ hidden: false, _id: 'row' }"
                      if row_present else "null")};
        const cb  = {("{ checked: " + ("true" if cb_checked else "false")
                      + ", _id: 'cb' }" if cb_present else "null")};
        function $(id) {{
            if (id === 'hide-frozen-row') return row;
            if (id === 'hide-frozen')     return cb;
            return null;
        }}
        const _frozen = {json.dumps(frozen)};
        function _frozenSet() {{
            if (_frozen === null || _frozen === undefined) return null;
            return new Set(_frozen);
        }}
        let applyHideFrozenCalls = 0;
        function applyHideFrozen() {{ applyHideFrozenCalls += 1; }}

        {fn_src}

        refreshHideFrozenAvailability();

        console.log(JSON.stringify({{
            row_hidden_after: row ? row.hidden : null,
            cb_checked_after: cb ? cb.checked : null,
            apply_calls:       applyHideFrozenCalls,
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
#  Architectural contract: presence is data-independent                  #
# --------------------------------------------------------------------- #


class TestRowAlwaysVisibleContract:
    """The row's PRESENCE is unconditional.  Pins both the template
    + the JS so neither side can re-introduce the visibility gate."""

    def test_template_does_not_ship_hidden_on_row(self):
        html = TEMPLATE.read_text(encoding="utf-8")
        m = re.search(
            r'<label[^>]*id="hide-frozen-row"[^>]*>',
            html,
        )
        assert m is not None, (
            "could not find <label id=\"hide-frozen-row\"> in "
            "the template"
        )
        opening = m.group(0)
        assert " hidden" not in opening and "hidden=" not in opening, (
            f"#hide-frozen-row must NOT ship with the ``hidden`` "
            f"attribute -- the row is ALWAYS visible per the "
            f"2026-06-14 UI contract.  Opening tag: {opening!r}"
        )

    def test_refresh_never_writes_row_hidden(self):
        """The JS function must NEVER set ``row.hidden``.  Pre-fix
        it wrote ``row.hidden = true / false`` to gate visibility
        on data; new contract: presence is data-independent."""
        fn = _extract_fn_source("refreshHideFrozenAvailability")
        assert "row.hidden" not in fn, (
            f"refreshHideFrozenAvailability must NOT touch "
            f"``row.hidden``; the row's visibility is owned by "
            f"the template and is unconditional.  Function body:\n"
            f"{fn}"
        )


# --------------------------------------------------------------------- #
#  Runtime behaviour: stale checkbox cleanup                             #
# --------------------------------------------------------------------- #


class TestRefreshBehaviour:
    """When frozen_atoms goes empty AND the checkbox was checked,
    un-check it so the user doesn't see a stale ``filter on`` state
    with nothing to filter.  Everything else is a no-op."""

    def test_non_empty_frozen_atoms_is_a_noop(self):
        out = _run_with_frozen([5, 7, 9], cb_checked=True)
        assert out["row_hidden_after"] is False
        assert out["cb_checked_after"] is True
        assert out["apply_calls"] == 0

    def test_non_empty_with_unchecked_box_is_noop(self):
        out = _run_with_frozen([5, 7, 9], cb_checked=False)
        assert out["row_hidden_after"] is False
        assert out["cb_checked_after"] is False
        assert out["apply_calls"] == 0

    def test_empty_frozen_atoms_unchecks_stale_box(self):
        out = _run_with_frozen([], cb_checked=True)
        assert out["row_hidden_after"] is False
        assert out["cb_checked_after"] is False
        assert out["apply_calls"] == 1

    def test_empty_with_already_unchecked_box_is_noop(self):
        out = _run_with_frozen([], cb_checked=False)
        assert out["row_hidden_after"] is False
        assert out["cb_checked_after"] is False
        assert out["apply_calls"] == 0

    def test_missing_frozen_field_behaves_like_empty(self):
        out = _run_with_frozen(None, cb_checked=True)
        assert out["row_hidden_after"] is False
        assert out["cb_checked_after"] is False
        assert out["apply_calls"] == 1
