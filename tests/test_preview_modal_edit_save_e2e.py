"""End-to-end Playwright tests for the projects sidebar's
View/Preview modal — view → edit → save with mtime-based safe
overwrite (task #302, 2026-06-09).

Pre-task-#302 the modal showed file contents read-only with a
disabled Save button.  The /api/files/write endpoint (with
``expected_mtime`` for conflict detection) had been shipped for
months; only the UI wiring was missing.  Pin the new flow
end-to-end here.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest


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
    to ``target_path``, and open the preview modal."""
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

    # View mode: <pre> visible, textarea hidden, Edit enabled,
    # Save disabled.  textContent preserves the file bytes
    # verbatim; inner_text() normalises a trailing newline so
    # use textContent here.
    body_text = page.evaluate(
        "() => document.getElementById('ps-preview-body').textContent"
    )
    assert body_text == original
    assert page.locator("#ps-preview-edit").is_hidden()
    assert page.locator("#ps-preview-edit-btn").is_enabled()
    assert page.locator("#ps-preview-save-btn").is_disabled()

    # Click Edit → swap, focus textarea, button states flip.
    page.locator("#ps-preview-edit-btn").click()
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-edit').hidden"
    )
    assert page.locator("#ps-preview-edit-btn").is_disabled()
    # Save still disabled — no dirty edits yet.
    assert page.locator("#ps-preview-save-btn").is_disabled()

    # Modify content; Save enables; status hints "Unsaved changes".
    new_content = original + "third line — edited from the modal\n"
    page.locator("#ps-preview-edit").fill(new_content)
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
    # both the pre body (View mode) AND the originalText path.
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
    # View body shows the saved content (modulo trailing newline,
    # which textContent strips).
    page.wait_for_function(
        "() => document.getElementById('ps-preview-body').textContent"
        "        .includes('third line — edited from the modal')"
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
    page.locator("#ps-preview-edit-btn").click()
    page.locator("#ps-preview-edit").fill("edited in browser\n")
    page.wait_for_function(
        "() => !document.getElementById('ps-preview-save-btn').disabled"
    )

    # Simulate an out-of-band write: the file gets a NEW mtime
    # before the user clicks Save.  Sleep briefly to ensure the
    # filesystem records a distinguishable mtime; then rewrite.
    time.sleep(1.1)
    target.write_text("someone else's edit\n")

    page.locator("#ps-preview-save-btn").click()
    page.wait_for_function(
        "() => /mtime|conflict|changed/i.test("
        "  document.getElementById('ps-preview-error').textContent)",
        timeout=5000,
    )
    # Disk content reflects the OTHER writer, not the browser's
    # edit — the safe-overwrite contract held.
    assert target.read_text() == "someone else's edit\n"


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
    page.locator("#ps-preview-edit-btn").click()
    page.locator("#ps-preview-edit").fill("unsaved edit\n")
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
