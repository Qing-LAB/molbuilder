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

from playwright.sync_api import expect  # noqa: E402

from support.sidebar import WAIT_MS, row, walk_to  # noqa: E402


@pytest.fixture
def flask_server(isolated_projects_root):
    """A server for this test, on the test's own projects tree
    (`support.live_server.serve`)."""
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


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

    MUTATIONS THIS MUST FAIL AGAINST: the file kept on re-listing read from
    the shared slot; every page keyed alike."""
    proj = isolated_projects_root.resolve() / "P"
    here, there = proj / "structure", proj / "optimization" / "run"
    here.mkdir(parents=True)
    there.mkdir(parents=True)
    x, y = here / "x.xyz", there / "y.xyz"
    for f in (x, y):
        f.write_text("1\n\nH 0 0 0\n")

    def shows(folder, file):
        """The sidebar on ``folder``, ``file`` picked: its row marked
        (`projects.md` § 4: *"The selected file's row is highlighted"*)."""
        expect(page.locator(".ps-crumb.is-current")).to_have_attribute(
            "title", str(folder), timeout=WAIT_MS)
        expect(row(page, file)).to_have_class(
            re.compile(r"\bis-selected\b"), timeout=WAIT_MS)

    page.goto(f"{flask_server}/structure-optimization")
    walk_to(page, x)
    shows(here, x)

    page.locator('.app-tabs a[href="/molbuilder"]').click()
    shows(here, x)
    walk_to(page, y)
    shows(there, y)

    page.locator('.app-tabs a[href="/structure-optimization"]').click()
    shows(here, x)


def test_a_project_is_deleted_only_once_its_name_is_typed(
        page, flask_server, isolated_projects_root):
    """Esc answers no even with the name typed, Enter on the empty field and
    the disabled Delete answer nothing, and the project stays; its name typed,
    Delete takes it, and the list says so.
    """
    proj = isolated_projects_root.resolve() / "doomed"
    (proj / "structure").mkdir(parents=True)
    (proj / "structure" / "x.xyz").write_text("1\n\nH 0 0 0\n")
    native = []
    page.on("dialog", lambda d: (native.append(d.message), d.dismiss()))

    page.goto(f"{flask_server}/molbuilder", wait_until="domcontentloaded")
    doomed = row(page, proj)
    doomed.wait_for(state="visible", timeout=WAIT_MS)
    asked = page.locator("dialog.mb-dialog[open]")

    def ask():
        doomed.locator(".ps-entry-kebab").click()
        page.locator(".ps-entry-menu-item", has_text="Delete project").click()
        asked.wait_for(state="visible", timeout=WAIT_MS)

    ask()
    asked.locator("input[data-role='typed']").fill("doomed")
    page.keyboard.press("Escape")
    asked.wait_for(state="detached", timeout=WAIT_MS)
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
    doomed.wait_for(state="detached", timeout=WAIT_MS)
    assert not proj.exists(), "the dialog closed but the project is still there"
    assert native == [], f"a native dialog was opened: {native}"


def test_the_create_bar_makes_a_project_a_folder_and_an_upload(
        page, flask_server, isolated_projects_root, tmp_path_factory):
    """The sidebar's three actions (`web/projects.md` § 4), each pressed and
    its dialog answered: what it made is on disk and in the list, and the
    sidebar is where the person now works.

    THE SILENT FAILURE: a button losing its id leaves the JS unable to wire
    it: the control renders and does nothing when clicked, with no error
    anywhere."""
    root = isolated_projects_root.resolve()
    proj = root / "made"
    notes = proj / "user" / "notes"
    src = tmp_path_factory.mktemp("on_disk") / "water.xyz"
    src.write_text("3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
    bar = page.locator(".ps-create-bar")
    asked = page.locator("dialog.mb-dialog[open]")

    def on(folder):
        expect(page.locator(".ps-crumb.is-current")).to_have_attribute(
            "title", str(folder), timeout=WAIT_MS)

    def named(button, name):
        bar.locator(button).click()
        asked.wait_for(state="visible", timeout=WAIT_MS)
        asked.locator("input[data-role='name']").fill(name)
        asked.locator("[data-action='confirm']").click()
        asked.wait_for(state="detached", timeout=WAIT_MS)

    page.goto(f"{flask_server}/molbuilder", wait_until="domcontentloaded")
    expect(bar.locator("#ps-create-project-btn")).to_contain_text(
        "New project", timeout=WAIT_MS)
    on(root)    # drawn after the bar's buttons are wired: `projects-sidebar.js`

    named("#ps-create-project-btn", "made")
    on(proj)
    assert proj.is_dir()
    walk_to(page, proj / "user")

    expect(bar.locator("#ps-create-folder-btn")).to_contain_text(
        "New folder", timeout=WAIT_MS)
    named("#ps-create-folder-btn", "notes")
    on(notes)
    assert notes.is_dir()

    expect(bar.locator("#ps-create-upload-btn")).to_contain_text(
        "Upload", timeout=WAIT_MS)
    bar.locator("#ps-create-upload-btn").click()
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("input[data-role='file']").set_input_files(str(src))
    asked.locator("[data-action='confirm']").click()
    row(page, notes / "water.xyz").wait_for(state="visible", timeout=WAIT_MS)
    assert (notes / "water.xyz").read_bytes() == src.read_bytes()

    # THE SAME NAME AGAIN (`projects.md` § 5, 2.69.9): the sidebar asks, as a
    # save asks, and a yes replaces the file -- through 2.69.8 the upload was
    # refused with words telling the person to "pass overwrite=true".
    again = tmp_path_factory.mktemp("again") / "water.xyz"
    again.write_text("3\nwater, moved\nO 0 0 1\nH 0.957 0 1\nH -0.239 0.927 1\n")
    bar.locator("#ps-create-upload-btn").click()
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("input[data-role='file']").set_input_files(str(again))
    asked.locator("[data-action='confirm']").click()
    replace = page.locator("dialog.mb-dialog[open]",
                           has_text="Replace water.xyz?")
    replace.wait_for(state="visible", timeout=WAIT_MS)
    replace.locator("[data-action='discard']").click()
    replace.wait_for(state="detached", timeout=WAIT_MS)
    for _ in range(100):           # the second post lands after the yes
        if (notes / "water.xyz").read_bytes() == again.read_bytes():
            break
        page.wait_for_timeout(100)
    assert (notes / "water.xyz").read_bytes() == again.read_bytes(), (
        "the yes did not replace the file")
