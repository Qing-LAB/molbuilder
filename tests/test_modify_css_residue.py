"""Guard: the Modify-tab stylesheet stays free of the dead post-migration
selectors, and uses the shared --font-mono token (not a re-declared family list).

Pins the 2026-07 residue cleanup so the dead blocks can't creep back.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "molbuilder/web/static/modify/style.css"


def test_dead_selectors_are_gone():
    src = STYLE.read_text(encoding="utf-8")
    for dead in (
        ".viewer-controls",              # View menu is .mol-viewer-menu-* now
        ".edit-panel .selection-info",   # per-atom info table retired
        ".edit-panel .readout ",         # only .readout-inline is emitted
        ".edit-panel .selection-block",  # the pre-tabs summary block is gone
        "future-op",                     # no such class in template/JS
        ".modify-selection",             # empty migration-window placeholder
    ):
        assert dead not in src, f"dead selector {dead!r} is back in modify/style.css"


def test_monospace_uses_the_shared_token():
    # §2.1: one shared font source -- no re-declared `ui-monospace, "SF Mono", ...`
    # family list; use var(--font-mono).
    src = STYLE.read_text(encoding="utf-8")
    assert 'ui-monospace, "SF Mono"' not in src, (
        "a raw monospace family list is re-declared -- use var(--font-mono)")
    assert "var(--font-mono)" in src

# The two checks that stood here -- one reading `flex-direction: row` out of
# the stylesheet, one asking whether "modify-check-row" appeared between a
# checkbox's id and the nearest preceding `<label` in the template -- are now
# ONE measurement in tests/test_molbuilder_e2e.py::
# test_a_checkbox_sits_beside_its_own_text, which opens the Slab op-tab and
# reads the painted geometry.  The second was green by coincidence: the
# checkbox it named has no wrapping <label>, so the rfind landed on an
# unrelated field's label and the class matched by adjacency.  Neither could
# see the cascade, which is the only place the bug ever lived.
