"""End-to-end Playwright tests for the projects sidebar's
View/Preview modal — view → edit → save with mtime-based safe
overwrite (task #302, 2026-06-09).

Pre-task-#302 the modal showed file contents read-only with a
disabled Save button.  The /api/files/write endpoint (with
``expected_mtime`` for conflict detection) had been shipped for
months; only the UI wiring was missing.  Pin the new flow
end-to-end here.

Task #310 (2026-06-09) replaced the <pre>+<textarea> pair with a
single CodeMirror 5 instance — virtual scroll caps DOM memory
for large files, search + jump-to-line addons handle find /
Go-to-line.  The tests drive CM via the test handle exposed on
``elModal.__molbuilder_test_cm`` so they don't depend on CM's
internal DOM.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture
def flask_server():
    """Function-scoped Flask server so each test can register its
    own tmp_path as a picker root."""
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


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


def _setup(page, base_url, target_path):
    """Open /molbuilder, prime the sidebar's current file pointer
    to ``target_path``, open the preview modal, and wait for the
    CodeMirror instance to mount.

    All E2E tests below assume the CM test handle is reachable
    via ``modalEl.__molbuilder_test_cm`` once this returns; that's
    the contract preview.js carries (set in ``_ensureCmMounted``).
    """
    page.goto(f"{base_url}/molbuilder", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.projects "
        "      && typeof window.molbuilder.projects.setShared "
        "             === 'function'",
        timeout=5000,
    )
    page.evaluate(
        "(p) => window.molbuilder.projects.setShared("
        "  p.substring(0, p.lastIndexOf('/')), p)",
        target_path,
    )
    # Open the preview modal programmatically — the per-entry View
    # button click depends on the sidebar list re-render landing
    # for the right file row, which adds flake for no value.
    # The exported showPreview() function is the same code path
    # the button would reach.
    page.evaluate(
        "async () => {"
        "  const mod = await import("
        "    '/static/lib/projects/preview.js');"
        "  await mod.showPreview();"
        "}"
    )
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-modal').hidden",
        timeout=5000,
    )
    # Wait for the CM test handle to attach — _ensureCmMounted
    # sets it after the vendored bundle finishes loading.  Without
    # this, downstream getValue() / setValue() calls would race.
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm !== undefined",
        timeout=10000,
    )


def _cm_value(page):
    """Read the editor's current text via the test handle."""
    return page.evaluate(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue()"
    )


def _cm_set_value(page, text):
    """Replace the editor's text via the test handle.  Equivalent
    to a user clearing the editor and typing — fires the change
    event that drives the Save-enable + dirty status."""
    page.evaluate(
        "(t) => document.getElementById('ps-preview-modal')"
        "          .__molbuilder_test_cm.setValue(t)",
        text,
    )


def _cm_line_count(page):
    """Number of lines currently in the editor doc."""
    return page.evaluate(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.lineCount()"
    )


def test_preview_modal_view_edit_save_round_trip(
        page, flask_server, tmp_path, monkeypatch):
    """The canonical edit-save round trip: open modal → click
    Edit → modify → click Save → verify file on disk + status
    shows "Saved at ..." → close → reopen → verify new content
    is what's rendered."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "myproj"
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
        timeout=5000,
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

    # Modify content via the CM handle (equivalent to typing); the
    # change event flips Save-enable + status.
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
        timeout=5000,
    )
    # File on disk reflects the edit.
    assert target.read_text() == new_content

    # Save button disabled again (nothing dirty), Edit still
    # disabled (we're still in edit mode).
    assert page.locator("#ps-preview-save-btn").is_disabled()

    # Close + re-open → the saved content is what shows up in
    # both the editor body AND the originalText path.
    page.locator(".ps-preview-close-footer").click()
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal').hidden"
    )
    page.evaluate(
        "async () => {"
        "  const mod = await import("
        "    '/static/lib/projects/preview.js');"
        "  await mod.showPreview();"
        "}"
    )
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-modal').hidden"
    )
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue()"
        "         .includes('third line — edited from the modal')",
        timeout=5000,
    )


def test_preview_modal_save_handles_mtime_conflict(
        page, flask_server, tmp_path, monkeypatch):
    """If the file changes on disk between Read and Save, the
    server returns 409 with a clear message; the modal surfaces
    that in the error slot instead of silently clobbering."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "myproj"
    proj.mkdir()
    target = proj / "conflict.txt"
    target.write_text("original\n")

    _setup(page, flask_server, str(target))
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length > 0",
        timeout=5000,
    )
    page.locator("#ps-preview-edit-btn").click()
    _cm_set_value(page, "edited in browser\n")
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-save-btn').disabled"
    )

    # Simulate an out-of-band write: the file gets a NEW mtime
    # before the user clicks Save.  Set the mtime explicitly via
    # os.utime so the test doesn't depend on filesystem mtime
    # resolution (the previous time.sleep(1.1) burned ~1s per run
    # for no contract value — the contract is "different mtime,"
    # not "mtime distinguishable to the second").
    import os as _os
    old_mtime = target.stat().st_mtime
    target.write_text("someone else's edit\n")
    _os.utime(target, (old_mtime + 2.0, old_mtime + 2.0))

    page.locator("#ps-preview-save-btn").click()
    page.wait_for_function(
        "() => /mtime|conflict|changed/i.test("
        "  document.getElementById('ps-preview-error').textContent)",
        timeout=5000,
    )
    # Disk content reflects the OTHER writer, not the browser's
    # edit — the safe-overwrite contract held.
    assert target.read_text() == "someone else's edit\n"


def test_preview_modal_disables_edit_for_large_files(
        page, flask_server, tmp_path, monkeypatch):
    """Files past the EDIT_MAX_BYTES cap (32 MB) load in
    paginated VIEW mode — bytes stream in 256 KB chunks via
    /api/files/read_range and Edit is disabled with a "use
    external editor" hint.  Pin both halves so a future refactor
    that drops the size check (or silently re-enables Edit on a
    too-big file) surfaces.

    The cap is in the JS — the test exercises a file built just
    over the cap so the paginated path fires.  The fixture writes
    33 MB of 'A' so the file's UTF-8 size matches the byte size
    exactly + the chunked load completes quickly."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "myproj"
    proj.mkdir()
    big = proj / "large.txt"
    # 33 MB > 32 MB cap.  Write in a single shot; trivially
    # compressible at the filesystem layer but the modal sees the
    # full size on stat.
    big.write_text("A" * (33 * 1024 * 1024))

    _setup(page, flask_server, str(big))

    # Edit button must be DISABLED — file is past the cap.
    page.wait_for_function(
        "() => document.getElementById('ps-preview-edit-btn').disabled",
        timeout=10000,
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
        timeout=5000,
    )
    body_len = page.evaluate(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length"
    )
    assert body_len > 0, "first chunk did not render"
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
        page, flask_server, tmp_path, monkeypatch):
    """Closing the modal with unsaved edits must prompt the user.
    Pin the dialog so a future refactor that drops the
    ``confirm()`` (or trades it for a silent close) surfaces."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "myproj"
    proj.mkdir()
    target = proj / "to_close.txt"
    target.write_text("hello\n")

    _setup(page, flask_server, str(target))
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal')"
        "         .__molbuilder_test_cm.getValue().length > 0",
        timeout=5000,
    )
    page.locator("#ps-preview-edit-btn").click()
    _cm_set_value(page, "unsaved edit\n")
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-save-btn').disabled"
    )

    # Dialog handler — Playwright only honours one listener per
    # page, so we flip a flag controlling accept/dismiss inside
    # the same handler.
    decisions = {"action": "dismiss"}
    page.on("dialog", lambda d: (
        d.accept() if decisions["action"] == "accept" else d.dismiss()
    ))

    # First close: dismiss the confirm; modal stays open.
    page.locator(".ps-preview-close-footer").click()
    page.wait_for_timeout(200)
    assert not page.evaluate(
        "() => document.getElementById('ps-preview-modal').hidden"
    ), "modal closed despite the dirty-confirm declining"

    # Second close: accept; modal closes.
    decisions["action"] = "accept"
    page.locator(".ps-preview-close-footer").click()
    page.wait_for_function(
        "() => document.getElementById('ps-preview-modal').hidden"
    )
    # The on-disk file was NOT modified (we declined to save).
    assert target.read_text() == "hello\n"
