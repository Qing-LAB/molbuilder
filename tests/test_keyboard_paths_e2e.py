"""**Every action a click performs has a keyboard path** (`web/ui-contract.md`
§ 4.1) -- the surfaces the UI review found reached by the mouse alone, each
worked here by the keys a person presses:

* the spectrum's modes table -- a column sorted, a mode shown;
* a saved state's actions in the run history -- Restore reached;
* the projects sidebar's rows -- a folder opened;
* a Documents folder -- opened and closed, and every folder a document
  sits in opened when the page opens on it;
* the Transport engine strip -- the app's one tab strip, its panel named.

A list of choices is one tab stop, the arrows moving, Enter doing what a
click does (`lib/one-stop-list.js`).

THE SILENT FAILURE, the same for each: the action a click performs is there
for a mouse and nowhere for a keyboard -- a header cell, a table row, a list
row or a folder row with a click handler alone -- and a person without a
mouse, or a screen reader, never reaches it.
"""
from __future__ import annotations

import pytest

from support.sidebar import WAIT_MS, history_shown, results_on
from support.task_setup_page import task_setup_on

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


def _focused(page):
    """The control holding focus, by its name -- ``None`` on the body."""
    return page.evaluate(
        "() => { const a = document.activeElement;"
        " return (!a || a === document.body) ? null"
        "   : (a.getAttribute('aria-label') || a.textContent.trim()); }")


def test_the_modes_table_is_worked_by_the_keys(page, real_water_vibration,
                                               monkeypatch):
    """`web/spectra.md` § 3: a column's header is a button that sorts by it,
    and the modes one list of choices -- one tab stop, the mode shown; the
    arrows move; Enter shows the mode -- the water vibration's run
    (`support.real_runs.water`), its spectrum opened in Results as a person
    opens it."""
    from conftest import point_the_one_door_at
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from support.live_server import serve
    folder = real_water_vibration.attempt("freq")
    spectrum = str(on_label_at(folder, ROLE_SPECTRA))
    point_the_one_door_at(real_water_vibration.tree.parent,
                          monkeypatch.setenv)
    with serve() as base:
        results_on(page, base, folder)
        page.wait_for_function(
            "(want) => [...document.querySelectorAll("
            "  '#results-file-picker-select option')]"
            "  .some(o => o.value === want)", arg=spectrum, timeout=WAIT_MS)
        page.select_option("#results-file-picker-select", value=spectrum)
        page.wait_for_selector("#modes-tbody tr", state="attached",
                               timeout=WAIT_MS)
        # A COLUMN SORTED: the frequency header's button, Enter.
        head = '#modes-thead-row th[data-col="frequency_cm1"]'
        page.locator(f"{head} button").focus()
        assert _focused(page) == "ω (cm⁻¹)", _focused(page)
        before = page.get_attribute(head, "aria-sort")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "(a) => document.querySelector(a.head).getAttribute('aria-sort')"
            "       !== a.before", arg={"head": head, "before": before},
            timeout=WAIT_MS)
        # ONE STOP, ON THE MODE SHOWN.
        stops = page.eval_on_selector_all(
            "#modes-tbody tr[data-mode]",
            "rs => rs.filter(r => r.tabIndex === 0)"
            "        .map(r => [r.dataset.mode, r.classList.contains('active')])")
        assert len(stops) == 1 and stops[0][1], stops
        # A MODE SHOWN: the stop, the arrow to the next mode, Enter -- that
        # row the one shown, the stop with it.
        page.locator("#modes-tbody tr[tabindex='0']").focus()
        page.keyboard.press("ArrowDown")
        moved = page.evaluate(
            "() => { const a = document.activeElement;"
            " return a.matches('#modes-tbody tr[data-mode]')"
            "        && !a.classList.contains('active') ? a.dataset.mode"
            "                                           : null; }")
        assert moved, "the arrow did not move to another mode"
        page.keyboard.press("Enter")
        page.wait_for_function(
            "(n) => { const r = document.getElementById('mode-row-' + n);"
            "  return r.classList.contains('active') && r.tabIndex === 0; }",
            arg=moved, timeout=WAIT_MS)


@pytest.fixture
def flask_server(isolated_projects_root):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def test_a_saved_states_restore_is_reached_by_the_keys(page, flask_server,
                                                       calc_dir):
    """`execution/checkpointing.md` § 5, *The panel*: the states are one list
    of choices; Enter on a state opens its row, and Restore is the next
    stop; a Restore runs under the page-wide cover (`ui-contract.md` § 10)
    and the keys come back to it.  The folder put under the history by its
    *Set up*, which saves the first state."""
    task_setup_on(page, flask_server, calc_dir)
    history_shown(page)
    page.wait_for_selector("#ps-checkpoint-init", state="visible",
                           timeout=WAIT_MS)
    page.click("#ps-checkpoint-init")
    page.wait_for_selector(".ps-checkpoint-list-item", timeout=WAIT_MS)
    row = page.locator(".ps-checkpoint-row-open[tabindex='0']")
    row.focus()
    assert row.get_attribute("aria-expanded") == "false"
    page.keyboard.press("Enter")
    assert row.get_attribute("aria-expanded") == "true"
    page.keyboard.press("Tab")
    assert _focused(page) == "Restore to here", _focused(page)
    # THE RESTORE, under the cover: every claim and release recorded, the
    # question answered, the cover down -- and the keys back on Restore.
    page.evaluate(
        "() => { const f = window.molbuilder.pageBusy; window.__fence = [];"
        "  const c = f.claim.bind(f), r = f.release.bind(f);"
        "  f.claim = (why, cx) => { window.__fence.push(why); return c(why, cx); };"
        "  f.release = () => { window.__fence.push('released'); return r(); }; }")
    page.keyboard.press("Enter")
    asked = page.locator("dialog.molbuilder-warning-modal[open]")
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("[data-action='discard']").click()
    page.wait_for_function(
        "() => window.__fence.includes('released')"
        "      && !window.molbuilder.pageBusy.isClaimed()", timeout=WAIT_MS)
    fence = page.evaluate("() => window.__fence")
    assert fence[0].startswith("Putting this folder back"), fence
    page.wait_for_function(
        "() => (document.activeElement || {}).textContent === 'Restore to here'",
        timeout=WAIT_MS)


def test_the_sidebars_rows_open_a_folder_by_the_keys(page, flask_server,
                                                     calc_dir):
    """`web/projects.md` § 4: the rows are one list of choices -- the arrows
    move, Enter on a folder opens it, as a click does, and the keys land on
    its first row."""
    task_setup_on(page, flask_server, calc_dir.parent)
    page.locator(".ps-entry[tabindex='0']").focus()
    folder = page.evaluate(
        "() => { const rows = [...document.querySelectorAll('.ps-entry')];"
        " const at = rows.indexOf(document.activeElement);"
        " return rows.slice(at).find(r => r.dataset.kind === 'directory')"
        "        ?.dataset.path || null; }")
    assert folder, "no folder below the stop"
    while page.evaluate("() => document.activeElement.dataset.path") \
            != folder:
        page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    page.wait_for_function(
        "(p) => [...document.querySelectorAll('.ps-entry')]"
        "        .every(r => r.dataset.path.startsWith(p + '/'))"
        "    && document.querySelectorAll('.ps-entry').length > 0",
        arg=folder, timeout=WAIT_MS)
    # ...THE KEYS ON THE FOLDER'S FIRST ROW, never on the page's body: the
    # list drawn again under them (`lib/one-stop-list.js` `redraw`).
    assert page.evaluate(
        "() => document.activeElement.matches('.ps-entry[tabindex=\"0\"]')"), \
        page.evaluate("() => document.activeElement.tagName")


def test_a_documents_folder_opens_and_closes_by_the_keys(page, flask_server):
    """`web/ui-contract.md` § 4.1: a folder's arrow is a button naming the
    folder and saying whether it is open; Enter opens and closes it."""
    page.goto(f"{flask_server}/documents")
    toggle = page.locator(".docs-tree-toggle").first
    toggle.wait_for(state="visible", timeout=WAIT_MS)
    folder = toggle.locator("xpath=ancestor::div[@class='docs-tree-dir'][1]")
    body = folder.locator(":scope > .docs-tree-body")
    toggle.focus()
    was = toggle.get_attribute("aria-expanded")
    assert was in ("true", "false"), was
    page.keyboard.press("Enter")
    now = toggle.get_attribute("aria-expanded")
    assert now == ("false" if was == "true" else "true"), now
    assert ("is-collapsed" in (body.get_attribute("class") or "")) \
        == (now == "false")


def test_a_document_opened_by_its_address_opens_every_folder_it_sits_in(
        page, flask_server):
    """`web/ui-contract.md` § 4.1: the page opened on a document three
    folders deep shows its link -- every folder above it open, the top one
    too, each button saying so."""
    # A DOCUMENT THREE FOLDERS DEEP, as the served tree has one.
    page.goto(f"{flask_server}/documents")
    page.wait_for_selector(".docs-tree-dir", state="attached",
                           timeout=WAIT_MS)
    href = page.evaluate(
        "() => [...document.querySelectorAll('.docs-tree-body .docs-link')]"
        "  .find(a => { let n = 0, d = a.closest('.docs-tree-dir');"
        "    while (d) { n++; d = d.parentElement"
        "      && d.parentElement.closest('.docs-tree-dir'); }"
        "    return n >= 3; })?.getAttribute('href')")
    assert href, "the served tree holds no document three folders deep"
    page.goto(f"{flask_server}/documents{href}")
    link = page.locator(f'.docs-link[href="{href}"]')
    link.wait_for(state="visible", timeout=WAIT_MS)
    closed = page.evaluate(
        "(el) => { const out = []; let d = el.closest('.docs-tree-dir');"
        " while (d) { const t = d.querySelector("
        "   ':scope > .docs-tree-header > .docs-tree-toggle');"
        "   if (t && t.getAttribute('aria-expanded') !== 'true')"
        "     out.push(t.getAttribute('aria-label'));"
        "   d = d.parentElement && d.parentElement.closest('.docs-tree-dir'); }"
        " return out; }", link.element_handle())
    assert closed == [], closed


def test_the_transport_engine_strip_is_the_apps_tab_strip(page,
                                                          flask_server):
    """`web/ui-contract.md` § 4.1: the engine strip is mounted by
    `lib/tab-strip.js` -- its panel named by its tab, which a screen reader
    reads as it enters the panel."""
    page.goto(f"{flask_server}/transport-calculation")
    page.wait_for_selector('#transport-engine-tabs [aria-selected="true"]',
                           timeout=WAIT_MS)
    tabs = page.eval_on_selector_all(
        "#transport-engine-tabs [role=tab]",
        "ts => ts.map(t => ({ on: t.getAttribute('aria-selected'),"
        "  stop: t.getAttribute('tabindex'), id: t.id,"
        "  panel: t.getAttribute('aria-controls') }))")
    chosen = [t for t in tabs if t["on"] == "true"]
    assert len(chosen) == 1, tabs
    named = page.get_attribute(f"#{chosen[0]['panel']}", "aria-labelledby")
    assert chosen[0]["id"] and named == chosen[0]["id"], (named, tabs)
