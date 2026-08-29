"""The `info` store's JS surface (molview.md § 8.4a,
plans/structure-info-plan.md): source pins over the module files, the
same style the other wiring guards use — the doors exist and are
ungated, the pane exists, and the wire carries the store both ways.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOLVIEW = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"


def _stripped(path: Path) -> str:
    src = path.read_text()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"^\s*//.*$", "", src, flags=re.M)


def test_the_model_exposes_the_three_doors_ungated():
    """`data.info.set/remove/get` — and NOT through `gated(...)`:
    § 9.4's one question answers no for a store that DESCRIBES the
    structure, and gating it would break the read-only Results
    viewer's contract recording."""
    src = _stripped(MOLVIEW / "model.js")
    assert "info: {" in src
    for door in ("set(key, value)", "remove(key)", "get()"):
        assert door in src, f"info door {door!r} missing"
    # The doors live in a plain object literal, not wrapped in gated().
    info_block = src.split("info: {", 1)[1].split("\n        },", 1)[0]
    assert "gated(" not in info_block, (
        "the info doors must stay UNGATED -- info describes the "
        "structure, it is not the structure (molview.md § 8.4a)")


def test_the_panel_has_the_metadata_page():
    src = _stripped(MOLVIEW / "ui.js")
    assert '["info", "Metadata"]' in src, "the third tab is gone"
    assert "drawInfo" in src, "the page never repaints"
    assert "molviewer-info-list" in src


def test_the_wire_carries_info_both_ways():
    src = _stripped(MOLVIEW / "model-jobs.js")
    assert "payload.info" in src, (
        "structureFromServer drops the store the server sent")
    assert "out.info = structure.info" in src, (
        "structureForServer drops the store on the way out -- the "
        "exported pair would lose what the pane shows")


def test_the_pane_vocabulary_is_defined_in_the_module_sheet():
    css = (MOLVIEW / "molview.css").read_text()
    for cls in ("molviewer-info-empty", "molviewer-info-list",
                "molviewer-info-key", "molviewer-info-value"):
        assert "." + cls in css, f"{cls} unstyled"
