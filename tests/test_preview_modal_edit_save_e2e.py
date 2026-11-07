"""End-to-end Playwright tests for the projects sidebar's
View/Preview modal — view → edit → save with mtime-based safe
overwrite (task #302, 2026-06-09), through /api/files/write and its
``expected_mtime`` conflict detection.

The editor is a single CodeMirror 5 instance.  The tests open a file as a
person does -- its sidebar row's *View* -- and type into the editor; they READ
it through the test handle exposed on ``elModal.__molbuilder_test_cm`` so they
don't depend on CM's internal DOM.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from support.sidebar import WAIT_MS, replace_text, row, walk_to  # noqa: E402


@pytest.fixture
def flask_server(isolated_projects_root):
    """A server for this test, on the test's own projects tree
    (`support.live_server.serve`)."""
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _view(page, target_path):
    """``target_path`` opened in the preview modal as a person opens it: its
    row's menu, *View* (`projects.md` § 4), the sidebar walked to its folder
    first -- and the CodeMirror instance mounted."""
    from pathlib import Path
    target = Path(target_path)
    walk_to(page, target)
    row(page, target).locator(".ps-entry-kebab").click()
    page.get_by_role("menuitem", name="View", exact=True).click()
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-modal').hidden",
        timeout=WAIT_MS,
    )
    # Wait for the CM test handle to attach — _ensureCmMounted
    # sets it after the vendored bundle finishes loading.  Without
    # this, downstream getValue() calls would race.
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm !== undefined",
        timeout=WAIT_MS,
    )


def _setup(page, base_url, target_path):
    """Open /molbuilder and view ``target_path`` (:func:`_view`).

    All E2E tests below read the editor through the CM test handle
    ``modalEl.__molbuilder_test_cm`` once this returns; that's the contract
    preview.js carries (set in ``_ensureCmMounted``).
    """
    page.goto(f"{base_url}/molbuilder", wait_until="domcontentloaded")
    _view(page, target_path)


def _close(page):
    """The modal closed by its footer's Close, when it is open."""
    if not page.evaluate(
            "() => document.getElementById('ps-preview-modal').hidden"):
        page.locator(".ps-preview-close-footer").click()
        page.wait_for_function(
            "() => document.getElementById('ps-preview-modal').hidden")


def _cm_value(page):
    """Read the editor's current text via the test handle."""
    return page.evaluate(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue()"
    )


def _cm_set_value(page, text):
    """Replace the editor's text as a person does
    (`support.sidebar.replace_text`) -- the edit that drives the Save-enable
    + dirty status."""
    replace_text(page, "#ps-preview-modal .CodeMirror", text)


def test_preview_modal_view_edit_save_round_trip(
        page, flask_server, isolated_projects_root):
    """The canonical edit-save round trip (`projects.md` § 4, the preview
    pop-up): open modal → click Edit → modify → click Save → verify file
    on disk + status shows "Saved at ..." → close → reopen → verify new
    content is what's rendered.

    THE SILENT FAILURE: an edit Save never wrote to disk, or a reopened
    file showing the text it had before the save."""
    proj = isolated_projects_root.resolve() / "myproj"
    proj.mkdir()
    target = proj / "notes.txt"
    original = "first line\nsecond line\n"
    target.write_text(original)

    _setup(page, flask_server, str(target))

    # View mode: editor populated with the file content, Edit
    # enabled (file under cap), Save disabled (no edits).  CM's
    # getValue() preserves trailing newline verbatim.
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length > 0",
        timeout=WAIT_MS,
    )
    assert _cm_value(page) == original
    assert page.locator("#ps-preview-edit-btn").is_enabled()
    assert page.locator("#ps-preview-save-btn").is_disabled()

    # Click Edit → editor goes editable, button states flip.
    page.locator("#ps-preview-edit-btn").click()
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-modal')"
        "          .__molbuilder_test_cm.getOption('readOnly')"
    )
    assert page.locator("#ps-preview-edit-btn").is_disabled()
    # Save still disabled — no dirty edits yet.
    assert page.locator("#ps-preview-save-btn").is_disabled()

    # Type the new content; the edit flips Save-enable + status.
    new_content = original + "third line — edited from the modal\n"
    _cm_set_value(page, new_content)
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-save-btn').disabled"
    )
    status = page.locator("#ps-preview-status").inner_text()
    assert "Unsaved" in status, status

    # Save → /api/files/write fires → file on disk updates →
    # status flips to "Saved at HH:MM:SS".
    page.locator("#ps-preview-save-btn").click()
    page.wait_for_function(
        "() => /Saved at/.test("
        "  document.getElementById('ps-preview-status').textContent)",
        timeout=WAIT_MS,
    )
    # File on disk reflects the edit.
    assert target.read_text() == new_content

    # Save button disabled again: nothing dirty.
    assert page.locator("#ps-preview-save-btn").is_disabled()

    # Close + re-open → the saved content is what the editor shows.
    _close(page)
    _view(page, target)
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue()"
        "         .includes('third line — edited from the modal')",
        timeout=WAIT_MS,
    )


def test_preview_modal_save_handles_mtime_conflict(
        page, flask_server, isolated_projects_root):
    """If the file changes on disk between Read and Save, the
    server returns 409 with a clear message; the modal surfaces
    that in the error slot instead of silently clobbering
    (`projects.md` § 4: *a concurrent change on disk is caught ...
    instead of being silently overwritten*)."""
    proj = isolated_projects_root.resolve() / "myproj"
    proj.mkdir()
    target = proj / "conflict.txt"
    target.write_text("original\n")

    _setup(page, flask_server, str(target))
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length > 0",
        timeout=WAIT_MS,
    )
    page.locator("#ps-preview-edit-btn").click()
    _cm_set_value(page, "edited in browser\n")
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-save-btn').disabled"
    )

    # An out-of-band write before the user clicks Save: another writer's
    # text, written until the file's mtime is not the one the modal read --
    # the contract is "a different mtime", and a filesystem whose clock
    # ticks coarser than one write is written again, never given a forged
    # stamp.
    import time as _time
    read_at = target.stat().st_mtime_ns
    while True:
        target.write_text("someone else's edit\n")
        if target.stat().st_mtime_ns != read_at:
            break
        _time.sleep(0.01)

    page.locator("#ps-preview-save-btn").click()
    page.wait_for_function(
        "() => /mtime|conflict|changed/i.test("
        "  document.getElementById('ps-preview-error').textContent)",
        timeout=WAIT_MS,
    )
    # Disk content reflects the OTHER writer, not the browser's
    # edit — the safe-overwrite contract held.
    assert target.read_text() == "someone else's edit\n"


def test_preview_modal_disables_edit_for_large_files(
        page, flask_server, isolated_projects_root):
    """`projects.md` § 4: files up to 1 MB are editable, larger ones open
    view-only, and very large ones are paged in as you scroll.  In
    `preview.js`: past `VIEW_ONLY_BYTES` (1 MB) Edit is disabled and the
    editor read-only, with a "use an external editor" hint; past the
    one-read ceiling `BULK_READ_MAX_BYTES` (16 MB) the file is paged in,
    `PAGE_BYTES` (256 KB) at a time through /api/files/read_range.  Pin
    both halves so a refactor that reads a very large file whole, or
    re-enables Edit on one, surfaces.

    The file is 33 MB of 'A' -- past both caps, so the paged path fires;
    its UTF-8 size is its byte size and its first page lands quickly.  The
    1 MB edit cap alone (a file between 1 and 16 MB) is not reached here."""
    proj = isolated_projects_root.resolve() / "myproj"
    proj.mkdir()
    big = proj / "large.txt"
    # 33 MB > the 16 MB one-read ceiling.  Write in a single shot;
    # trivially compressible at the filesystem layer but the modal sees
    # the full size on stat.
    big.write_text("A" * (33 * 1024 * 1024))

    _setup(page, flask_server, str(big))

    # Edit button must be DISABLED — file is past the cap.
    page.wait_for_function(
        "() => document.getElementById('ps-preview-edit-btn').disabled",
        timeout=WAIT_MS,
    )

    # Status line explains why + advertises the keybindings.
    status = page.locator("#ps-preview-status").inner_text()
    assert "Large file" in status or "edit cap" in status, status
    assert "external editor" in status, status
    assert "Ctrl-F" in status, (
        "status line should advertise the search keybinding so the "
        "feature is discoverable above the cap"
    )

    # Wait for the first chunk to land in the editor doc.  The
    # paginated path fires the first range read asynchronously
    # after setting the Edit-disabled state.
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length > 0",
        timeout=WAIT_MS,
    )
    body_len = page.evaluate(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length"
    )
    # And it's nowhere near the full 33 MB — the modal loaded a
    # range chunk, not the whole file.
    assert body_len < 1024 * 1024, (
        f"paginated load should render in chunks; got {body_len} bytes "
        f"in first paint — looks like the bulk-read path ran"
    )

    # Editor is read-only for above-cap files.
    is_read_only = page.evaluate(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getOption('readOnly')"
    )
    assert is_read_only is True, (
        "above-cap files must keep readOnly=true; CM has no save "
        "path for chunk-replace-at-offset"
    )


def test_preview_modal_close_prompts_when_dirty(
        page, flask_server, isolated_projects_root):
    """Closing the modal with unsaved edits asks first -- the one confirm
    (`ui-contract.md` § 11), never the browser's own: Cancel keeps the
    editor open with the edit, the costly answer closes it unsaved."""
    proj = isolated_projects_root.resolve() / "myproj"
    proj.mkdir()
    target = proj / "to_close.txt"
    target.write_text("hello\n")

    _setup(page, flask_server, str(target))
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length > 0",
        timeout=WAIT_MS,
    )
    page.locator("#ps-preview-edit-btn").click()
    _cm_set_value(page, "unsaved edit\n")
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-save-btn').disabled"
    )
    native = []
    page.on("dialog", lambda d: (native.append(d.message), d.dismiss()))
    asked = page.locator("dialog.molbuilder-warning-modal[open]")

    # Esc asks the same question -- the key that closes the preview must
    # not also answer the confirm it opens.
    page.keyboard.press("Escape")
    asked.wait_for(state="visible", timeout=WAIT_MS)
    page.wait_for_timeout(300)
    assert asked.is_visible(), "Esc answered the question it had just asked"
    asked.locator("[data-action='cancel']").click()
    asked.wait_for(state="detached", timeout=WAIT_MS)

    # A close: Cancel; the modal stays open.
    page.locator(".ps-preview-close-footer").click()
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("[data-action='cancel']").click()
    asked.wait_for(state="detached", timeout=WAIT_MS)
    assert not page.evaluate(
        "() => document.getElementById('ps-preview-modal').hidden"
    ), "modal closed despite the dirty-confirm declining"

    # Second close: the costly answer; the modal closes.
    page.locator(".ps-preview-close-footer").click()
    asked.wait_for(state="visible", timeout=WAIT_MS)
    asked.locator("[data-action='discard']").click()
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal').hidden"
    )
    assert native == [], f"a native dialog was opened: {native}"
    # The on-disk file was NOT modified (we declined to save).
    assert target.read_text() == "hello\n"


#: `static/vendor/README.md`, *CodeMirror language modes*: each suffix and
#: the mode it is highlighted with -- ``"null"``, plain text, for
#: molbuilder's own formats, on purpose.  The `.md` row comes first: markdown
#: needs xml loaded with it, and no row before it loads xml.
_HIGHLIGHTED = [
    ("README.md", "markdown"), ("notes.markdown", "markdown"),
    ("data.json", "javascript"), ("tool.js", "javascript"),
    ("script.py", "python"), ("H2.template.toml", "toml"),
    ("H2_01_coarse.run.sh", "shell"), ("job.bash", "shell"),
    ("H2.sbatch", "shell"), ("data.xml", "xml"), ("style.css", "css"),
    ("conf.yaml", "yaml"), ("conf.yml", "yaml"),
    ("H2.fdf", "null"), ("H2.xyz", "null"), ("H2.out", "null"),
    ("H2.log", "null"), ("H2.molwatch.log", "null"), ("H2.STRUCT_OUT", "null"),
]
#: JSON is the javascript mode in its JSON setting -- the mode a `.json` asks
#: for, and the one a plain `.js` does not.
_JSON = {"data.json": True, "tool.js": False}


def test_the_editor_searches_jumps_and_highlights_what_it_says(
        page, flask_server, isolated_projects_root):
    """The editor's files are vendored -- served by molbuilder itself, no
    CDN (`static/vendor/README.md`) -- and one the bundle lacks fails in
    silence: its script 404s, its command or mode never registers, and the
    editor shows plain text and ignores Ctrl-F.

    Opened as a person opens it, in the preview modal: Ctrl-F brings up the
    search dialog, styled, and a word finds it in the file, and Alt-G and a
    line number jump to that line; every suffix
    the README's table names opens in its mode, molbuilder's own formats in
    plain text -- a file large enough to be paged in included, opened right
    after one that was highlighted -- and the first markdown file opened
    highlights its HTML, through the xml mode it needs; no request for an
    editor file fails.

    MUTATIONS THIS MUST FAIL AGAINST: a mode file missing from the bundle;
    the search, its cursor, the jump addon or the dialog's stylesheet left
    out of the loader; markdown
    loaded without xml; JSON highlighted as plain JavaScript; a
    paged file left in the last file's mode."""
    proj = isolated_projects_root.resolve() / "myproj"
    proj.mkdir()
    # Each file's own name its first line; the markdown file an HTML tag
    # after it, which markdown hands to the xml mode.
    texts = {name: f"{name}\n" + ("<b>bold</b>\n" if name == "README.md"
                                  else "")
             for name, _ in _HIGHLIGHTED}
    for name, text in texts.items():
        (proj / name).write_text(text)
    failed = []
    page.on("response", lambda r: failed.append(f"{r.status} {r.url}")
            if "/static/vendor/codemirror/" in r.url and r.status >= 400
            else None)
    page.on("requestfailed", lambda r: failed.append(f"failed {r.url}")
            if "/static/vendor/codemirror/" in r.url else None)
    cm = "document.getElementById('ps-preview-modal').__molbuilder_test_cm"
    dialog = "#ps-preview-modal .CodeMirror-dialog"

    _setup(page, flask_server, str(proj / _HIGHLIGHTED[0][0]))
    page.locator("#ps-preview-modal .CodeMirror").click()
    page.keyboard.press("Control+f")
    page.wait_for_selector(dialog, timeout=WAIT_MS)
    # The dialog sits over the editor, as its stylesheet places it.
    assert page.evaluate(
        "(s) => getComputedStyle(document.querySelector(s)).position",
        dialog) == "absolute", "the search dialog drawn without its styles"
    page.keyboard.type("README")
    page.keyboard.press("Enter")
    page.wait_for_function(f"() => {cm}.getSelection() === 'README'",
                           timeout=WAIT_MS)
    # Enter both searches and closes the dialog.
    page.wait_for_selector(dialog, state="detached", timeout=WAIT_MS)
    page.keyboard.press("Alt+g")
    page.wait_for_selector(dialog, timeout=WAIT_MS)
    page.keyboard.type("2")
    page.keyboard.press("Enter")
    page.wait_for_function(f"() => {cm}.getCursor().line === 1",
                           timeout=WAIT_MS)

    shown = {}
    for name, _ in _HIGHLIGHTED:
        _close(page)
        _view(page, proj / name)
        page.wait_for_function(f"(t) => {cm}.getValue() === t",
                               arg=texts[name], timeout=WAIT_MS)
        if name == "README.md":
            xml_with_markdown = page.evaluate(
                f"() => String({cm}.getTokenTypeAt({{line: 1, ch: 1}}))"
                "  .includes('tag')")
        shown[name] = page.evaluate(f"() => {cm}.getMode().name")
        if name in _JSON:
            assert bool(page.evaluate(f"() => {cm}.getMode().jsonMode")) \
                is _JSON[name], name
    assert shown == dict(_HIGHLIGHTED), {
        n: m for n, m in shown.items() if m != dict(_HIGHLIGHTED)[n]}

    # A file past the one-read ceiling is paged in -- after a .py.
    big = proj / "big.out"
    big.write_text("big\n" + ("x" * 99 + "\n") * (17 * 1024 * 1024 // 100))
    # The sidebar walked up and back down, so the new file has its row.
    _close(page)
    walk_to(page, proj.parent)
    _view(page, proj / "script.py")
    page.wait_for_function(f"() => {cm}.getMode().name === 'python'",
                           timeout=WAIT_MS)
    _close(page)
    _view(page, big)
    page.wait_for_function(f"() => {cm}.getValue().startsWith('big')",
                           timeout=WAIT_MS)
    assert page.evaluate(f"() => {cm}.getMode().name") == "null", (
        "a paged-in file kept the last file's highlighting")
    assert xml_with_markdown, (
        "the first markdown file opened did not highlight its HTML -- "
        "markdown set up without the xml mode it needs")
    assert not failed, failed
