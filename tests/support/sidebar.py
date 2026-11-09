"""THE PROJECTS SIDEBAR, DRIVEN AS A PERSON DRIVES IT (`web/projects.md` § 4).

A page test reaches a folder or a file the way a person does: the sidebar's
breadcrumb and rows, clicked.  What those clicks set -- the selection the tabs
read -- is never set in its place (`process/testing.md` § 5, *Drive the page as
a person does*; § 4: a page's input is what the server answers).

Used by every browser test that reaches a folder or a file -- and the home of
the two keystrokes every page's fields and editors take the same way
(:func:`committed`, :func:`replace_text`): the module of doors a browser test
drives a page through, where `support.pages` reads served pages with no
browser at all.
"""
from __future__ import annotations

import re
from pathlib import Path

#: The bound a browser test waits by for the page to answer an act -- a
#: page drawn, a file listed, a load landed -- named, never a number of the
#: test's own: every door here, in `support.task_setup_page` and in
#: `support.results_export` waits by it.  One bound, so a slow machine
#: costs every test alike.
WAIT_MS = 30_000


def row(page, path):
    """``path``'s row in the sidebar's list."""
    return page.locator(f'.ps-entry[data-path="{Path(path).resolve()}"]')


def walk_to(page, target, *, use: bool = False) -> None:
    """The sidebar on ``target`` -- a folder opened, or a file picked in its
    folder, double-clicked with ``use`` (*use it*) -- by its own clicks: up the
    breadcrumb to the nearest folder on ``target``'s way, then down its rows,
    one click a level.  ``target`` lies under the projects root the page
    lists."""
    from playwright.sync_api import expect
    target = Path(target).resolve()
    here = page.locator(".ps-crumb.is-current")
    here.wait_for(timeout=WAIT_MS)
    hops = [Path(t) for t in page.locator(".ps-crumb").evaluate_all(
        "cs => cs.map(c => c.title)")]
    at = max((h for h in hops if h == target or h in target.parents),
             key=lambda h: len(h.parts))
    if at != hops[-1]:
        page.locator(f'.ps-crumb[title="{at}"]').click()
        expect(here).to_have_attribute("title", str(at), timeout=WAIT_MS)
    for part in target.relative_to(at).parts:
        at = at / part
        page.locator(f'.ps-entry[data-path="{at}"]').click(timeout=WAIT_MS)
    if target.is_dir():
        expect(here).to_have_attribute("title", str(target), timeout=WAIT_MS)
    else:
        if use:
            row(page, target).dblclick(timeout=WAIT_MS)
        expect(row(page, target)).to_have_class(
            re.compile(r"\bis-selected\b"), timeout=WAIT_MS)


def picked(page, base: str, tab: str, target) -> None:
    """``tab`` opened, listening to the sidebar, ``target`` picked in it by
    its rows -- and its Load button awake, for the caller to press.

    LISTENING, not merely drawn: each tab subscribes to the sidebar inside
    ``whenReady("projects")``, right after it writes the load bar's readout
    in the same synchronous step -- so a written readout means a pick is
    heard.  A pick made before that left the Load button asleep (measured
    2026-09-29: a warm second page)."""
    page.goto(f"{base}/{tab}", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => (document.getElementById('load-source-readout') || {})"
        "      .textContent", timeout=WAIT_MS)
    walk_to(page, target)
    page.wait_for_function(
        "() => !document.getElementById('load-from-sidebar-btn').disabled",
        timeout=WAIT_MS)


def reload_from(page, folder) -> None:
    """The Results tab, open on ``page``, bound to ``folder`` as a person
    binds it: the sidebar walked to the folder, *Reload from current project
    dir* pressed -- the tab follows the sidebar on that press alone
    (`web/results.md` § 2.1)."""
    walk_to(page, folder)
    page.get_by_role("button", name="Reload from current project dir").click()


def results_on(page, base: str, folder) -> None:
    """The Results tab opened and bound to ``folder`` (:func:`reload_from`)."""
    page.goto(f"{base}/results")
    reload_from(page, folder)


def committed(page, field, value: str) -> None:
    """``value`` typed into the field ``field`` (a selector -- its first
    match -- or the field's locator) and the field left by Tab -- the
    keystroke that commits a field, which its `change` listener hears."""
    box = page.locator(field).first if isinstance(field, str) else field
    box.fill(value)
    box.press("Tab")


def replace_text(page, editor: str, text: str) -> None:
    """The text of the CodeMirror editor ``editor`` (a selector of its
    ``.CodeMirror`` node) replaced by ``text``, as a person replaces it: the
    editor clicked, its text selected from its start to its end -- Ctrl-A is
    off in the file preview's editor (`preview.js`: a multi-MB select-all
    froze the tab) -- and ``text`` put in its place."""
    page.click(editor)
    page.keyboard.press("Control+Home")
    page.keyboard.press("Control+Shift+End")
    page.keyboard.insert_text(text)


def history_shown(page) -> None:
    """The run history brought forward in the sidebar, as a person brings it
    -- its toggle in the filter bar -- and idempotent about it: the panel
    shares its slot with the file list, and the toggle remembers which one
    is showing, so a second press after a reload puts the list back."""
    page.wait_for_selector("#ps-checkpoint-toggle", timeout=WAIT_MS)
    showing = page.evaluate(
        "() => { const p = document.getElementById('ps-checkpoint');"
        "        return !!(p && !p.hidden && p.offsetParent !== null); }")
    if not showing:
        page.click("#ps-checkpoint-toggle")
    page.wait_for_selector("#ps-checkpoint", state="visible", timeout=WAIT_MS)
