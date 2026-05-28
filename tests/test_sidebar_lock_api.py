"""Playwright tests for the Projects sidebar lock API (2026-05-27).

The lock API lives in ``molbuilder/web/static/lib/projects/state.js``
and is used by every long-running save pipeline (Save .fdf, Save
.py, Save .spectra.py).  These tests pin the *behaviour contract*:

  1. ``lock(reason)`` flips ``isLocked()`` true + records the reason
  2. ``unlock()`` releases it (idempotent)
  3. ``onLockChange`` subscribers fire on every transition
  4. Re-entering lock() while already locked throws
  5. ``cancelLockedOperation()`` runs registered cancelers in order
  6. The sidebar DOM gets ``.is-locked`` + the lock banner appears
     (with the reason) when lock() is called

The DOM checks (6) catch a regression where the visual coupling
breaks even though the logical lock state is right -- the user
wouldn't see the lock and could click through to navigate.
"""
from __future__ import annotations

import threading

import pytest


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


def _open_index(page, base_url):
    """Open the Build page (any page with the sidebar works) +
    wait for window.molbuilder.projects to mount."""
    errors = []
    page.on("pageerror", lambda exc: errors.append(("pageerror", str(exc))))
    page.on("console", lambda msg: (
        errors.append(("console.error", msg.text))
        if msg.type == "error" else None
    ))
    page.goto(f"{base_url}/")
    page.wait_for_function(
        "() => window.molbuilder "
        "&& window.molbuilder.projects "
        "&& typeof window.molbuilder.projects.lock === 'function'"
    )
    # The DOM-visual subscription (list.js's _applyLockVisual) is wired
    # inside initList(), which runs AFTER projects-sidebar.js's async
    # bootstrap resolves /api/files/roots.  Wait for initList to have
    # completed by polling for the breadcrumb to have been rendered --
    # the breadcrumb is empty until openDir(start) runs at the tail
    # of init(), which is downstream of initList().
    page.wait_for_function(
        "() => document.querySelector('#ps-breadcrumb')"
        " && document.querySelector('#ps-breadcrumb').children.length > 0"
    )
    return errors


class TestLockApi:

    def test_lock_unlock_flips_isLocked(self, page, flask_server):
        _open_index(page, flask_server)
        result = page.evaluate("""() => {
            const p = window.molbuilder.projects;
            const before = p.isLocked();
            p.lock("Saving FDF…", []);
            const during  = p.isLocked();
            const reason  = p.getLockReason();
            p.unlock();
            const after   = p.isLocked();
            return { before, during, reason, after };
        }""")
        assert result["before"] is False
        assert result["during"] is True
        assert result["reason"] == "Saving FDF…"
        assert result["after"]  is False

    def test_unlock_is_idempotent(self, page, flask_server):
        _open_index(page, flask_server)
        # Two unlocks in a row must not throw + must leave isLocked false.
        result = page.evaluate("""() => {
            const p = window.molbuilder.projects;
            p.unlock();
            p.unlock();
            return p.isLocked();
        }""")
        assert result is False

    def test_reentrant_lock_throws(self, page, flask_server):
        _open_index(page, flask_server)
        # Acquiring while already locked must throw -- nested locks would
        # tangle the cancel-button semantics (whose cancelers do we run?).
        result = page.evaluate("""() => {
            const p = window.molbuilder.projects;
            p.lock("first op", []);
            let threw = false, msg = "";
            try { p.lock("second op", []); }
            catch (e) { threw = true; msg = e.message; }
            p.unlock();
            return { threw, msg };
        }""")
        assert result["threw"] is True
        assert "already locked" in result["msg"]

    def test_onLockChange_fires_on_transitions(self, page, flask_server):
        _open_index(page, flask_server)
        events = page.evaluate("""() => {
            const p = window.molbuilder.projects;
            const log = [];
            // Subscribe -- fires once immediately with the current state.
            const unsubscribe = p.onLockChange(
                (ev) => log.push({ locked: ev.locked, reason: ev.reason }));
            p.lock("op A", []);
            p.unlock();
            unsubscribe();
            // Should NOT fire after unsubscribe.
            p.lock("op B", []);
            p.unlock();
            return log;
        }""")
        # Expected sequence: initial (unlocked) -> lock A -> unlock A.
        # The post-unsubscribe lock/unlock pair must NOT appear.
        assert len(events) == 3
        assert events[0] == {"locked": False, "reason": ""}
        assert events[1] == {"locked": True,  "reason": "op A"}
        assert events[2] == {"locked": False, "reason": ""}

    def test_cancelLockedOperation_runs_cancelers(self, page, flask_server):
        _open_index(page, flask_server)
        out = page.evaluate("""() => {
            const p = window.molbuilder.projects;
            let calls = [];
            p.lock("op", [
                () => calls.push("first"),
                () => calls.push("second"),
            ]);
            p.cancelLockedOperation();
            // cancelers ran but the lock is still held -- the operation's
            // own try/finally is what releases it after the abort unwinds.
            const stillLocked = p.isLocked();
            p.unlock();
            return { calls, stillLocked };
        }""")
        assert out["calls"] == ["first", "second"]
        assert out["stillLocked"] is True

    def test_cancelLockedOperation_when_unlocked_is_noop(
            self, page, flask_server):
        _open_index(page, flask_server)
        # Calling cancel with no lock active must not throw + must not
        # invoke anything.  Guards against a stale Cancel-button click
        # arriving after the operation has already finished.
        threw = page.evaluate("""() => {
            const p = window.molbuilder.projects;
            try { p.cancelLockedOperation(); return false; }
            catch (_) { return true; }
        }""")
        assert threw is False

    def test_lock_paints_is_locked_class_and_banner(
            self, page, flask_server):
        _open_index(page, flask_server)
        # The visibility check uses getComputedStyle().display, NOT
        # just the .hidden attribute -- 2026-05-28 regression caught
        # by the user: ``.ps-lock-banner { display: flex }`` (author
        # CSS) tied on specificity with the user-agent ``[hidden] {
        # display: none }`` rule, so the author rule won the cascade
        # and the banner was visible at page load despite hidden=
        # being set.  Attribute-only ``!ban.hidden`` would have passed
        # the test even with the bug.
        result = page.evaluate("""() => {
            const p   = window.molbuilder.projects;
            const sb  = document.getElementById("projects-sidebar");
            const ban = document.getElementById("ps-lock-banner");
            const msg = document.getElementById("ps-lock-message");
            const visible = (el) => getComputedStyle(el).display !== 'none';
            // BEFORE any lock: banner must be rendered as hidden
            // (sanity check the page-load state).
            const at_load = {
                hidden_attr:    ban.hasAttribute('hidden'),
                rendered_hidden: !visible(ban),
                hasClass:       sb.classList.contains('is-locked'),
            };
            p.lock("Saving Spectra…", []);
            const after_lock = {
                hasClass:       sb.classList.contains("is-locked"),
                banner_visible: visible(ban),
                hidden_attr:    ban.hasAttribute('hidden'),
                msg_text:       msg ? msg.textContent : "",
            };
            p.unlock();
            const after_unlock = {
                hasClass:        sb.classList.contains("is-locked"),
                banner_visible:  visible(ban),
                hidden_attr:     ban.hasAttribute('hidden'),
            };
            return { at_load, after_lock, after_unlock };
        }""")
        # Page-load: banner is hidden BOTH as attribute AND as render.
        assert result["at_load"]["hidden_attr"]     is True
        assert result["at_load"]["rendered_hidden"] is True, (
            "banner has hidden= attribute but is RENDERED visible at "
            "page load -- author CSS display: rule outranks the UA "
            "stylesheet's [hidden] rule.  Add ``.ps-lock-banner[hidden] "
            "{ display: none; }`` to projects-sidebar.css."
        )
        assert result["at_load"]["hasClass"]        is False
        # After lock: visible + class set + reason text right.
        assert result["after_lock"]["hasClass"]       is True
        assert result["after_lock"]["banner_visible"] is True
        assert result["after_lock"]["hidden_attr"]    is False
        assert result["after_lock"]["msg_text"]       == "Saving Spectra…"
        # After unlock: hidden again (both attr + render).
        assert result["after_unlock"]["hasClass"]       is False
        assert result["after_unlock"]["banner_visible"] is False
        assert result["after_unlock"]["hidden_attr"]    is True

    def test_cancel_button_invokes_cancelers(self, page, flask_server):
        _open_index(page, flask_server)
        # Pin the DOM wiring: a real click on #ps-lock-cancel must run
        # the registered canceler.  This is the user-facing escape
        # hatch (Layer C of the recovery design).
        result = page.evaluate("""() => {
            const p = window.molbuilder.projects;
            let aborted = false;
            p.lock("Saving FDF…", [() => { aborted = true; }]);
            document.getElementById("ps-lock-cancel").click();
            const wasAborted = aborted;
            p.unlock();
            return wasAborted;
        }""")
        assert result is True
