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


def test_send_to_handoff_is_not_described_as_current():
    # The Send-to-Optimization / Send-to-Build handoff was removed; comments must
    # not describe it as a live "bottom footer" feature.  ("...were removed" is fine.)
    src = STYLE.read_text(encoding="utf-8")
    assert "Footer row at the bottom holds Save" not in src


def test_monospace_uses_the_shared_token():
    # §2.1: one shared font source -- no re-declared `ui-monospace, "SF Mono", ...`
    # family list; use var(--font-mono).
    src = STYLE.read_text(encoding="utf-8")
    assert 'ui-monospace, "SF Mono"' not in src, (
        "a raw monospace family list is re-declared -- use var(--font-mono)")
    assert "var(--font-mono)" in src
