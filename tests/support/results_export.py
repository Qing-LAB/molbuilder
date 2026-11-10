"""Saving the structure on screen out of the Results tab, as a person does.

Export -> Data -> Save to project on the viewer the tab is showing.  When the
viewer holds more than one frame the dialog asks which, and the answer is the
LAST -- asked for past the end, because the dialog clamps to the frames the
viewer holds.  The folder is one below the projects root, opened down to by
the picker's rows -- the root itself is never an answer (`projects.md`
§ 4.1); the name is typed over the one the dialog offered, and that offer is
returned so a caller can hold it too.

It waits for the VIEWER, not the page: the run-state badge and the SCF line
are written before MolView has received its frames (`lib/trajectory/core.js`
installs them after a `/api/build/load` round trip), and an Export pressed in
between is answered "the structure cannot be exported".
"""
from __future__ import annotations

from support.sidebar import WAIT_MS


def wait_for_viewer(page, *, atoms: int, frames: int) -> None:
    """Until MolView holds ``atoms`` atoms and, for a trajectory, ``frames``
    frames -- the frame bar is drawn only for more than one."""
    page.wait_for_function(
        "(n) => { const m = (document.querySelector("
        "  '.molviewer-selection-count')?.textContent || '')"
        "  .match(/ of (\\d+) selected/); return !!m && Number(m[1]) === n; }",
        arg=atoms, timeout=WAIT_MS)
    if frames > 1:
        page.wait_for_function(
            "(n) => { const m = (document.querySelector("
            "  '.molviewer-frames-counter')?.textContent || '')"
            "  .match(/\\/\\s*(\\d+)/); return !!m && Number(m[1]) === n; }",
            arg=frames, timeout=WAIT_MS)


def save_to_project(page, name: str, *, frames: int, tree, folder: str) -> str:
    """Save the viewer's structure as ``<folder>/<name>.xyz`` (+ its sidecar)
    -- ``folder`` a tree address in the projects tree ``tree`` -- and return
    the name the dialog offered."""
    from pathlib import Path
    page.locator(".molviewer-menu > summary", has_text="Export").click()
    page.locator(".molviewer-export-section", has_text="Data").locator(
        "button", has_text="Save to project").click()
    if frames > 1:
        ask = page.locator(".molviewer-export-dialog")
        ask.wait_for(state="visible", timeout=WAIT_MS)
        for box in ask.locator("input[type=number]").all():
            box.fill("9999")
        ask.locator("[data-action='confirm']").click()
    dialog = page.locator("dialog.molbuilder-tree-picker-dialog")
    parts = Path(folder).parts
    for depth in range(1, len(parts)):
        dialog.locator(f'li.tp-node[data-path="{tree.joinpath(*parts[:depth])}"]'
                       " > .tp-row > .tp-twisty").click()
    where = dialog.locator(f'li.tp-node[data-path="{tree / folder}"] > .tp-row')
    where.wait_for(state="visible", timeout=WAIT_MS)
    where.click()
    page.locator("dialog [data-action='confirm']").first.click()
    box = page.locator("dialog input[data-role='name']")
    box.wait_for(state="visible", timeout=WAIT_MS)
    offered = box.input_value()
    box.fill(name)
    page.locator("dialog [data-action='confirm']").last.click()
    page.wait_for_function(
        "() => /^saved /.test(document.querySelector("
        "  '.molviewer-export-status')?.textContent || '')",
        timeout=WAIT_MS)
    return offered
