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


# --------------------------------------------------------------------- #
#  The junction panel's stacked-label rule must not catch checkboxes     #
#                                                                       #
#  Contract: docs/web/ui-contract.md § 1 (a page sheet arranges, it does #
#  not re-decide form-control layout) and science/junction-cell.md § 6   #
#  (the panel carries two switches).                                     #
#                                                                       #
#  The rule below was a blanket `label` descendant selector.  It exists  #
#  for m/n/layers -- label above a number input -- but it also caught    #
#  `<label><input type=checkbox> text</label>`, where                     #
#  `flex-direction: column` puts the box above its text and              #
#  `align-items: stretch` widens the <input> to the full row (measured:  #
#  384.5px), painting the tick centred in empty space.                   #
# --------------------------------------------------------------------- #

TEMPLATE = ROOT / "molbuilder/web/templates/modify.html"


def test_a_checkbox_row_is_laid_out_inline():
    src = STYLE.read_text(encoding="utf-8")
    start = src.index(".modify-edit-panel .modify-check-row {")
    block = src[start:src.index("}", start)]
    assert "flex-direction: row" in block, "the box sits BEFORE its text, not above it"
    assert "align-items: center" in block, "box and text share a baseline row"


def test_both_junction_checkboxes_use_the_row_class():
    """Every checkbox in the panel opts in -- one that forgets gets the
    stacked treatment back, silently."""
    html = TEMPLATE.read_text(encoding="utf-8")
    # `elc-orthogonal` and `elc-pad-gap` were the Junction panel's two, and
    # the panel is gone (redesign plan § 3.4).  The Slab panel's box is the
    # one left, and the rule it is held to is the same: a checkbox row wears
    # the row class so the label sits beside the box rather than under it.
    for cid in ("slab-orthogonal",):
        i = html.find(f'id="{cid}"')
        assert i != -1, f'id="{cid}" not found in modify.html'
        label_start = html.rfind("<label", 0, i)
        assert label_start != -1, f"{cid} is not inside a <label>"
        assert "modify-check-row" in html[label_start:i], (
            f"{cid} is not inside a .modify-check-row label -- it will get the "
            f"stacked treatment and paint its box above its own text"
        )
