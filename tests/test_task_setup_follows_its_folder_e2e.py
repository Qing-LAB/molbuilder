"""**A folder changed by another writer reaches the tab showing it**
(`web/projects.md` § 2; `web/task-setup.md` § 2.1).

The file doors announce every folder they change, and whoever shows one
follows it: here a record of the calculation renamed from the sidebar's own
menu -- another writer as far as Task setup is concerned -- while Task setup
shows the folder.

* With nothing unsaved, the page reads its folder again: the record is no
  longer there, and the page says so.
* With an edit not yet saved, the page keeps it -- the field as typed, the
  bar saying it is unsaved -- and says the folder changed on disk.

THE SILENT FAILURES: through 2.68.4 a rename, move, copy, upload, delete or
the file viewer's Save announced nothing, and Task setup went on showing a
folder that no longer was; and had it re-read on every announcement, a
rename of any file in the folder would have dropped the person's unsaved
edits without a word.
"""
from __future__ import annotations

import pytest

from support.sidebar import WAIT_MS
from support.task_setup_page import task_setup_on

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture
def calc_dir(tmp_path, monkeypatch):
    """A described calculation (`support.road`), its one stage `coarse`."""
    from support.road import described, road_tree
    return described(road_tree(tmp_path, monkeypatch), name="probe",
                     stage_strategy="")


@pytest.fixture
def flask_server(isolated_projects_root):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _renamed_in_the_sidebar(page, name, to):
    """``name`` renamed as a person renames it: its row's menu, *Rename…*,
    the page's dialog answered."""
    page.click(f'button[aria-label="Actions for {name}"]')
    page.get_by_role("menuitem", name="Rename…").click()
    asked = page.locator("dialog.mb-dialog[open]")
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("input").fill(to)
    asked.locator("[data-action='confirm']").click()
    page.wait_for_selector(f'.ps-entry[data-path$="/{to}"]', timeout=WAIT_MS)


def test_nothing_unsaved_the_page_reads_its_folder_again(page, flask_server,
                                                         calc_dir):
    task_setup_on(page, flask_server, calc_dir)
    page.wait_for_selector('#ts-f-req[data-exists="yes"]', state="attached",
                           timeout=WAIT_MS)
    _renamed_in_the_sidebar(page, "job-request.json", "job-request.old")
    page.wait_for_selector('#ts-f-req[data-exists="no"]', state="attached",
                           timeout=WAIT_MS)


def test_an_unsaved_edit_is_kept_and_the_change_said(page, flask_server,
                                                     calc_dir):
    task_setup_on(page, flask_server, calc_dir)
    field = page.locator(
        '.ts-next-step:not([hidden]) .ts-runcard input[aria-label="value for '
        'omp_threads"]').first
    field.wait_for(state="visible", timeout=WAIT_MS)
    field.fill("3")
    field.press("Tab")
    page.wait_for_function("() => !window.molbuilder.pageBusy.isClaimed()",
                           timeout=WAIT_MS)
    assert page.is_visible("#ts-dirty"), "the edit was not held as unsaved"
    _renamed_in_the_sidebar(page, "job-request.json", "job-request.old")
    page.wait_for_function(
        "() => /changed on disk/.test("
        "  (document.getElementById('ts-save-said') || {}).textContent || '')",
        timeout=WAIT_MS)
    assert page.is_visible("#ts-dirty"), "the unsaved edit was dropped"
    assert page.locator(
        '.ts-next-step:not([hidden]) .ts-runcard input[aria-label="value for '
        'omp_threads"]').first.input_value() == "3"
    # ...and the page was not read again: it still names the record it read.
    assert page.get_attribute("#ts-f-req", "data-exists") == "yes"
