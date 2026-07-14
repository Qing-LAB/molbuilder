"""App-wide system-notification bar (docs/protocols/notifications.md).

Pins the contract: a STACK the user clears INDIVIDUALLY, dedup by id, and the
``molbuilder:persist-error`` -> error-row wiring (the workspace-contract §4.7
non-blocking/error-explicit persist failure finally has a consistent home).
"""
from __future__ import annotations

import threading

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture(scope="module")
def flask_server():
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


def _boot(page, base):
    page.goto(f"{base}/molbuilder", wait_until="domcontentloaded")
    # The bar's module loads with defer; wait for its API.
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.notify "
        "&& typeof window.molbuilder.notify.show === 'function'",
        timeout=8000)


def test_bar_present_and_hidden_until_first_notification(page, flask_server):
    _boot(page, flask_server)
    assert page.locator("#app-notifications").count() == 1
    # Hidden while empty (notifications.md §1).
    assert page.locator("#app-notifications").is_hidden()
    assert page.evaluate("() => window.molbuilder.notify.list()") == []


def test_persist_error_event_surfaces_one_error_row(page, flask_server):
    _boot(page, flask_server)
    page.evaluate(
        "() => window.dispatchEvent(new CustomEvent('molbuilder:persist-error',"
        "  { detail: { op: 'write-state', status: 500 } }))")
    page.wait_for_selector(".app-notification.app-notification--error")
    rows = page.locator(".app-notification")
    assert rows.count() == 1
    txt = rows.first.inner_text()
    assert "Couldn't save state to disk" in txt
    lst = page.evaluate("() => window.molbuilder.notify.list()")
    assert len(lst) == 1 and lst[0]["level"] == "error" and lst[0]["id"] == "persist-error"


def test_repeat_persist_error_dedups_into_one_row_with_counter(page, flask_server):
    _boot(page, flask_server)
    for _ in range(3):
        page.evaluate(
            "() => window.dispatchEvent(new CustomEvent('molbuilder:persist-error',"
            "  { detail: { op: 'write-state', status: 500 } }))")
    page.wait_for_selector(".app-notification--error")
    # Dedup by id: still ONE row, count 3.
    assert page.locator(".app-notification").count() == 1
    lst = page.evaluate("() => window.molbuilder.notify.list()")
    assert len(lst) == 1 and lst[0]["count"] == 3
    assert "×3" in page.locator(".app-notification").first.inner_text()


def test_stack_of_two_and_individual_clearance(page, flask_server):
    _boot(page, flask_server)
    # A persist error + a distinct info notification -> a STACK of two.
    page.evaluate(
        "() => window.dispatchEvent(new CustomEvent('molbuilder:persist-error',"
        "  { detail: { op: 'write-state', status: 500 } }))")
    page.evaluate(
        "() => window.molbuilder.notify.show({ id: 'hello', level: 'info',"
        "  message: 'a second, unrelated message' })")
    page.wait_for_function("() => window.molbuilder.notify.list().length === 2")
    assert page.locator(".app-notification").count() == 2

    # Dismiss ONLY the persist error via its × button; the info row remains.
    err = page.locator(".app-notification--error")
    err.locator(".app-notification__dismiss").click()
    page.wait_for_function("() => window.molbuilder.notify.list().length === 1")
    remaining = page.evaluate("() => window.molbuilder.notify.list()")
    assert remaining[0]["id"] == "hello"           # the info one survived
    assert page.locator(".app-notification--error").count() == 0

    # Clear the last one -> bar hides again.
    page.evaluate("() => window.molbuilder.notify.clearAll()")
    page.wait_for_function("() => window.molbuilder.notify.list().length === 0")
    assert page.locator("#app-notifications").is_hidden()
