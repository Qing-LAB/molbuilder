"""The projects sidebar's costliest question, in the browser.

Deleting a whole project removes every file under it, and nothing brings it
back.  `ui-contract.md` § 11: such a question is answered by TYPING the
project's name (`projects/dialogs.js` ``confirmByTyping``), so a stray Enter
or a click on Delete cannot answer it.  The dialog is the app's own
(`lib/dialog.js`), never the browser's ``prompt()`` -- which is why this can
be driven at all.
"""
from __future__ import annotations

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
