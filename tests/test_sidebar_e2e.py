"""The projects sidebar, in the browser.

* THE PLACE EACH TAB KEEPS (`web/projects.md` § 2): its folder and the file
  picked in it, come back to as left, whatever another tab picked meanwhile.
* A ROW'S MENU stays over its row: a scroll that moves the row closes it, one
  of the tab's own pane does not (`web/projects.md` § 4).
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
        page, flask_server, isolated_projects_root, tmp_path_factory,
        data_dir):
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
    # A STRUCTURE FILE IS WRITTEN AS A PAIR, through the codec
    # (`model/structure.md` § 2.4): its sidecar beside it, the pair reading
    # the atoms uploaded.
    from molbuilder.workingcopy_structure import StructureCodec
    assert (notes / "water.molstruct.json").is_file()
    assert StructureCodec().load(notes / "water.xyz").n_atoms == 3
    # ...AND SAID WHERE IT LANDED (`projects.md` § 5, `upload`): the path the
    # server wrote.  THE SILENT FAILURE (to 2.69.17): nothing said on success.
    said = page.locator("#app-notifications .app-notification",
                        has_text="Uploaded water.xyz")
    expect(said).to_contain_text("as made/user/notes/water.xyz.",
                                 timeout=WAIT_MS)

    # A PDB A PERSON BRINGS IN is imported as its .xyz (`model/structure.md`
    # § 2.4: a .pdb is read, never written), and the sidebar says so.
    pdb = data_dir / "multi_chain_well_formed.pdb"
    bar.locator("#ps-create-upload-btn").click()
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("input[data-role='file']").set_input_files(str(pdb))
    asked.locator("[data-action='confirm']").click()
    row(page, notes / "multi_chain_well_formed.xyz").wait_for(
        state="visible", timeout=WAIT_MS)
    expect(page.locator("#app-notifications .app-notification",
                        has_text=f"Uploaded {pdb.name}")).to_contain_text(
        "as made/user/notes/multi_chain_well_formed.xyz.", timeout=WAIT_MS)
    assert not (notes / pdb.name).exists()

    # THE SAME NAME AGAIN (`projects.md` § 3, 2.69.10): the files door asks
    # the one overwrite question, naming both halves a yes replaces, and a
    # yes replaces the pair.
    from support.road import until
    again = tmp_path_factory.mktemp("again") / "water.xyz"
    again.write_text("3\nwater, moved\nO 0 0 1\nH 0.957 0 1\nH -0.239 0.927 1\n")
    bar.locator("#ps-create-upload-btn").click()
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("input[data-role='file']").set_input_files(str(again))
    asked.locator("[data-action='confirm']").click()
    replace = page.locator("dialog.mb-dialog[open]",
                           has_text="Overwrite water.xyz?")
    replace.wait_for(state="visible", timeout=WAIT_MS)
    expect(replace).to_contain_text("water.molstruct.json")
    replace.locator("[data-action='discard']").click()
    replace.wait_for(state="detached", timeout=WAIT_MS)
    until(lambda: StructureCodec().load(notes / "water.xyz")
          .positions[0][2] == 1.0, within=WAIT_MS / 1000,
          what="the yes replaced the pair")


@pytest.mark.parametrize("scrolled, stays", [
    (".app-content", True), ("#ps-list", False)],
    ids=["the tab's own pane", "the sidebar's list"])
def test_a_rows_menu_closes_only_when_its_row_moves(page, flask_server,
                                                    calc_dir, scrolled,
                                                    stays):
    """The per-row ``⋯`` menu (`web/projects.md` § 4) stands over its row,
    fixed where the row was: a scroll that moves the row -- the sidebar's
    own list -- closes it, and one that does not -- the pane that holds the
    tab (`ui-contract.md` § 3.1) -- leaves it where it belongs, *Rename…*
    still answering.  THE SILENT FAILURE: the menu shut under the person's
    pointer by a scroll they did not make -- Task setup's pane moves after
    a run-card edit (one probe, 2026-11-12) -- so the click on *Rename…* met
    nothing (`test_task_setup_follows_its_folder_e2e`'s unsaved edit).  The
    viewport is short so that both panes can scroll once the folder is open
    (`process/testing.md` § 5)."""
    from support.task_setup_page import task_setup_on
    task_setup_on(page, flask_server, calc_dir)
    page.set_viewport_size(dict(page.viewport_size, height=600))
    pane = page.locator(scrolled)
    assert pane.evaluate(
        "e => e.scrollTop + e.clientHeight < e.scrollHeight"), (
        f"{scrolled} cannot scroll down, so this row asks nothing")
    # THE ROW'S BUTTON IN VIEW, so opening its menu scrolls nothing.
    kebab = page.locator('button[aria-label="Actions for job-request.json"]')
    shown, there = (page.locator("#ps-list").bounding_box(),
                    kebab.bounding_box())
    top, bottom = shown["y"], shown["y"] + shown["height"]
    assert top <= there["y"] and there["y"] + there["height"] <= bottom, (
        "the row is out of the list's view: opening its menu would scroll")
    kebab.click()
    rename = page.get_by_role("menuitem", name="Rename…")
    expect(rename).to_be_visible(timeout=WAIT_MS)
    # THE SCROLL ARRIVES BEFORE ANYTHING IS ASKED: its event, delivered to
    # the pane, has passed every listener the page holds for it.
    page.evaluate("""(s) => { window.__scrolled = false;
        document.querySelector(s).addEventListener(
            "scroll", () => { window.__scrolled = true; },
            { once: true }); }""", scrolled)
    box = pane.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, 200)
    page.wait_for_function("() => window.__scrolled", timeout=WAIT_MS)
    expect(rename).to_have_count(1 if stays else 0)
    if stays:
        rename.click()
        expect(page.locator("dialog.mb-dialog[open]")).to_be_visible(
            timeout=WAIT_MS)
