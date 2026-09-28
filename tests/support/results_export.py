"""Saving the structure on screen out of the Results tab, as a person does.

Export -> Data -> Save to project on the viewer the tab is showing.  When the
viewer holds more than one frame the dialog asks which, and the answer is the
LAST -- asked for past the end, because the dialog clamps to the frames the
viewer holds.  The folder is the picker's first row, the projects root; the
name is typed over the one the dialog offered, and that offer is returned so
a caller can hold it too.

It waits for the VIEWER, not the page: the run-state badge and the SCF line
are written before MolView has received its frames (`lib/trajectory/core.js`
installs them after a `/api/build/load` round trip), and an Export pressed in
between is answered "the structure cannot be exported".
"""
from __future__ import annotations

_WAIT_MS = 30_000


def wait_for_viewer(page, *, atoms: int, frames: int) -> None:
    """Until MolView holds ``atoms`` atoms and, for a trajectory, ``frames``
    frames -- the frame bar is drawn only for more than one."""
    page.wait_for_function(
        "(n) => { const m = (document.querySelector("
        "  '.molviewer-selection-count')?.textContent || '')"
        "  .match(/ of (\\d+) selected/); return !!m && Number(m[1]) === n; }",
        arg=atoms, timeout=_WAIT_MS)
    if frames > 1:
        page.wait_for_function(
            "(n) => { const m = (document.querySelector("
            "  '.molviewer-frames-counter')?.textContent || '')"
            "  .match(/\\/\\s*(\\d+)/); return !!m && Number(m[1]) === n; }",
            arg=frames, timeout=_WAIT_MS)


def save_to_project(page, name: str, *, frames: int) -> str:
    """Save the viewer's structure as ``<projects root>/<name>.xyz`` (+ its
    sidecar) and return the name the dialog offered."""
    page.locator(".molviewer-menu > summary", has_text="Export").click()
    page.locator(".molviewer-export-section", has_text="Data").locator(
        "button", has_text="Save to project").click()
    if frames > 1:
        ask = page.locator(".molviewer-export-dialog")
        ask.wait_for(state="visible", timeout=_WAIT_MS)
        for box in ask.locator("input[type=number]").all():
            box.fill("9999")
        ask.locator(".is-confirm").click()
    where = page.locator("dialog .tp-row:not(.tp-row--inert)").first
    where.wait_for(state="visible", timeout=_WAIT_MS)
    where.click()
    page.locator("dialog [data-action='confirm']").first.click()
    box = page.locator("dialog input[data-role='name']")
    box.wait_for(state="visible", timeout=_WAIT_MS)
    offered = box.input_value()
    box.fill(name)
    page.locator("dialog [data-action='confirm']").last.click()
    page.wait_for_function(
        "() => /^saved /.test(document.querySelector("
        "  '.molviewer-export-status')?.textContent || '')",
        timeout=_WAIT_MS)
    return offered
