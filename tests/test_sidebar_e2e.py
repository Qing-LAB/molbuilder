"""The projects sidebar, in the browser.

* THE PLACE EACH TAB KEEPS (`web/projects.md` § 2): its folder and the file
  picked in it, come back to as left, whatever another tab picked meanwhile.
* ITS COSTLIEST QUESTION.  Deleting a whole project removes every file under
  it, and nothing brings it back.  `ui-contract.md` § 11: such a question is
  answered by TYPING the project's name (`projects/dialogs.js`
  ``confirmByTyping``), so a stray Enter or a click on Delete cannot answer
  it.  The dialog is the app's own (`lib/dialog.js`), never the browser's
  ``prompt()`` -- which is why this can be driven at all.
"""
from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


def test_each_tab_comes_back_to_its_own_folder_and_file(
        page, flask_server, isolated_projects_root):
    """Each tab keeps its own place -- the folder AND the file picked in it
    -- and a tab never visited starts at the most recent place anywhere
    (`web/projects.md` § 2; the user, 2026-08-19: *"switching back and forth
    should not be trouble"*).

    THE SILENT FAILURE (2026-08-20).  A returning tab came back to its own
    folder but not to its own file: the sidebar's list read the file from
    the slot every tab shares, so what the tab had picked was gone, and
    nothing said so.

    Driven through the tab bar's links and the sidebar's own rows: a file
    picked on the Structure-optimization tab, another in another folder on
    the Molbuilder tab, and back.

    MUTATIONS THIS MUST FAIL AGAINST: the file kept on re-listing, or the
    file restored, read from the shared slot; every page keyed alike."""
    from playwright.sync_api import expect
    proj = isolated_projects_root.resolve() / "P"
    here, there = proj / "structure", proj / "optimization" / "run"
    here.mkdir(parents=True)
    there.mkdir(parents=True)
    x, y = here / "x.xyz", there / "y.xyz"
    for f in (x, y):
        f.write_text("1\n\nH 0 0 0\n")

    def row(path):
        return page.locator(f'.ps-entry[data-path="{path}"]')

    def shows(folder, file):
        """The sidebar on ``folder``, ``file`` picked: its row marked and the
        selection line naming it."""
        expect(page.locator(".ps-crumb.is-current")).to_have_attribute(
            "title", str(folder))
        expect(row(file)).to_have_class(re.compile(r"\bis-selected\b"))
        expect(page.locator("#ps-actions .ps-selection")).to_have_attribute(
            "title", str(file))

    page.goto(f"{flask_server}/structure-optimization")
    row(proj).click()
    row(here).click()
    row(x).click()
    shows(here, x)

    page.locator('.app-tabs a[href="/molbuilder"]').click()
    shows(here, x)
    page.locator(f'.ps-crumb[title="{proj}"]').click()
    row(proj / "optimization").click()
    row(there).click()
    row(y).click()
    shows(there, y)

    page.locator('.app-tabs a[href="/structure-optimization"]').click()
    shows(here, x)


def test_a_project_is_deleted_only_once_its_name_is_typed(
        page, flask_server, tmp_path, monkeypatch):
    """Esc answers no even with the name typed, Enter on the empty field and
    the disabled Delete answer nothing, and the project stays; its name typed,
    Delete takes it, and the list says so.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "doomed"
    (proj / "structure").mkdir(parents=True)
    (proj / "structure" / "x.xyz").write_text("1\n\nH 0 0 0\n")
    native = []
    page.on("dialog", lambda d: (native.append(d.message), d.dismiss()))

    page.goto(f"{flask_server}/molbuilder", wait_until="domcontentloaded")
    row = page.locator(f'.ps-entry[data-path="{proj.resolve()}"]')
    row.wait_for(state="visible", timeout=20000)
    asked = page.locator("dialog.mb-dialog[open]")

    def ask():
        row.locator(".ps-entry-kebab").click()
        page.locator(".ps-entry-menu-item", has_text="Delete project").click()
        asked.wait_for(state="visible", timeout=5000)

    ask()
    asked.locator("input[data-role='typed']").fill("doomed")
    page.keyboard.press("Escape")
    asked.wait_for(state="detached", timeout=5000)
    page.wait_for_timeout(500)
    assert proj.is_dir(), "Esc deleted the project"

    ask()
    typed = asked.locator("input[data-role='typed']")
    delete = asked.locator("[data-action='confirm']")
    typed.press("Enter")
    assert asked.is_visible() and proj.is_dir(), (
        "Enter on the empty field answered the question")
    assert delete.is_disabled(), "Delete could be clicked before the name"
    typed.fill("doome")
    assert delete.is_disabled(), "Delete took a name that is not the project's"

    typed.fill("doomed")
    assert delete.is_enabled()
    delete.click()
    row.wait_for(state="detached", timeout=20000)
    assert not proj.exists(), "the dialog closed but the project is still there"
    assert native == [], f"a native dialog was opened: {native}"
