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
    """Every place an edit is MARKED also records it against the contract.

    The three `recordEdit(...)` sites are the gated points where an edit has
    landed -- a read-only viewer and a failed edit never reach them -- so a
    contract-outdating call must sit beside each, or a pair exported after
    that edit carries a contract that silently no longer describes it.

    **This is a lint and stays one**: it quantifies over the edit-landed
    sites as a CLASS, and a fourth appearing without its mark is exactly what
    no behaviour test can catch. What it must NOT do is measure the call's
    spelling -- it asserted `count("markContractOutdated();") == 3` until
    2026-09-07, which went false the moment the label door started naming
    which flag it sets, on a change that made the mechanism more correct.
    WHICH flag each door raises is behaviour, and is driven in
    tests/test_molview_model.py: a label write raises `labels_modified` and
    not the other, a cell edit the reverse.

    Counted the same way for the same reason: the sites now say whether the
    edit lays down a timeline point (`recordEdit(true)` / the operation row's
    `checkpoint`), so the ARGUMENT varies by door and only the call itself is
    the class.  Whether each door checkpoints is behaviour, driven in
    tests/test_molview_stores_history.py.
    """
    src = _stripped(MOLVIEW / "model.js")
    assert src.count("recordEdit(") == 4, (
        "the edit-landed sites moved (3 calls + 1 definition); re-anchor this "
        "lint AND the markContractOutdated hooks together")
    assert src.count("markContractOutdated(") == 4, (
        "an edit-landed site no longer outdates the record (3 calls + 1 "
        "definition) -- a pair exported after that edit would carry a "
        "contract that silently no longer describes its atoms")
