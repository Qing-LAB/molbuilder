"""The `info` store's JS surface (molview.md § 8.4a,
archive/2026-09-01-structure-info-plan.md): source pins over the module files, the
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


#  ``test_the_wire_carries_info_both_ways`` stood here until 2026-08-30.
#  It asserted the string ``payload.info`` appeared in the module -- and
#  that string WAS the bug: no route has ever sent a flat ``payload.info``
#  (the store arrives inside the canonical ``structure`` envelope, like
#  every other field of a Structure), so every load answered an empty
#  store at HTTP 200 while this pin stayed green.  A pin that asks whether
#  a name is mentioned cannot tell a working path from a dead one.
#
#  Both directions are now pinned by ``tests/test_structure_info_bridge.py``,
#  which walks the chain end to end -- a store stated to the load door
#  comes back on the structure, a saved pair brings its store back when
#  re-opened, and the reader names the envelope the value arrives in.


def test_the_pane_vocabulary_is_defined_in_the_module_sheet():
    css = (MOLVIEW / "molview.css").read_text()
    for cls in ("molviewer-info-empty", "molviewer-info-list",
                "molviewer-info-key", "molviewer-info-value"):
        assert "." + cls in css, f"{cls} unstyled"


def test_every_edit_landed_site_outdates_the_record():
    """The structure_modified flag (user, 2026-08-29): each of the
    three edit-landed sites (the history.edited() marks -- geometry op,
    cell op, label write) also marks a recorded contract outdated, so a
    later reader of the pair knows these atoms are no longer the
    structure the contract described."""
    src = _stripped(MOLVIEW / "model.js")
    assert src.count("history.edited();") == 3, (
        "the edit-landed sites moved; re-anchor this pin AND the "
        "markContractOutdated hooks together")
    assert src.count("markContractOutdated();") == 3, (
        "an edit-landed site no longer outdates the record -- a pair "
        "exported after that edit would carry a contract that silently "
        "no longer describes its atoms")
    assert "structure_modified" in src
