"""Pin the pageshow / visibilitychange refresh contract for the
trajectory and spectra inspectors (audit task #194, 2026-06-02).

Background.  The 2026-06-02 /results stale-dropdown bug (#192) was
shaped: ``state stays cached across a tab re-entry; UI shows old
data until the user forces a manual refresh``.  After fixing the
file-picker for that specific manifestation, an audit found two
adjacent inspectors with the same shape:

  * ``lib/trajectory/core.js`` -- polls every 15 s for mtime drift,
    but setInterval timers are PAUSED while the page sits in
    bfcache.  After a back/forward restore, the next poll fires up
    to 15 s LATER.  A user who generated more frames in another tab
    can be staring at the old trajectory for that long.

  * ``lib/spectra/core.js`` -- mount-once, no auto-poll, no reload
    button.  A spectra calculation re-run in another tab is
    invisible until the user re-picks the file from the dropdown.

Fix: hook ``pageshow`` (covers bfcache restore + initial load) and
``visibilitychange`` -> visible (covers backgrounded-tab re-focus)
to force a fresh server round-trip on tab re-entry.

These tests pin the contract at the JS-event level (the same shape
as the /results file-picker tests in
``test_results_file_picker_e2e.py``):  dispatch the event manually
from inside the page + observe a fresh HTTP request fire to the
inspector's backing endpoint.

NOT covered here:

  * Full data-refresh round trip (would require realistic
    trajectory / spectra fixtures + a server that actually serves
    different content on the second call).  The HTTP-request-was-
    fired assertion is the right size for catching a regression
    that drops the event listener; the contents-actually-update
    side is owned by the underlying ``loadByPath`` / ``pollOnce``
    code paths which are exercised by the existing inspector
    tests.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


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


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Pin tmp_path as the only picker root."""
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None,
        conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)


_FIXTURE_SIESTA_OUT = (
    Path(__file__).resolve().parent
    / "watch" / "fixtures" / "siesta_frozen"
    / "hemeC-stage1-scf_not_conv-5fr.out"
)


@pytest.fixture
def project_with_trajectory(tmp_path, monkeypatch):
    """Project directory containing a real (small) SIESTA .out -- the
    trajectory inspector's primary file type.  Returns the absolute
    path string."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "myproj" / "spectra" / "test"
    proj.mkdir(parents=True)
    target = proj / "run.out"
    target.write_text(_FIXTURE_SIESTA_OUT.read_text())
    return str(target)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _open_results_and_select(page, base_url, file_path):
    """Open /results with the sidebar driven into ``file_path``'s
    parent dir + the file selected.  Mirrors the user flow for
    landing on the trajectory inspector via a sidebar click."""
    from pathlib import Path as P
    parent = str(P(file_path).parent)
    page.goto(f"{base_url}/structure")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.projects "
        "      && typeof window.molbuilder.projects.setShared "
        "             === 'function'"
    )
    page.evaluate(
        "(c) => window.molbuilder.projects.setShared(c.dir, c.file)",
        {"dir": parent, "file": file_path},
    )
    page.wait_for_timeout(200)
    page.goto(f"{base_url}/results")
    page.wait_for_selector("#inspector-host", timeout=5000)


# --------------------------------------------------------------------- #
#  Trajectory inspector: pageshow fires pollOnce                        #
# --------------------------------------------------------------------- #


class TestTrajectoryInspectorPageshowRefresh:
    """pageshow / visibilitychange events MUST trigger a fresh
    /api/watch/data round-trip when a trajectory is loaded.  Without
    this the user sees stale frames for up to POLL_MS after bfcache
    restore."""

    def test_pageshow_triggers_watch_data_request(
            self, page, flask_server, project_with_trajectory):
        _open_results_and_select(
            page, flask_server, project_with_trajectory)
        # Wait for the trajectory inspector to land + the initial
        # /api/watch/load to complete.  The frame strip's
        # .frame-counter element renders "<i+1> / <total>" — i.e.
        # the second number > 0 once frames have loaded.  (The old
        # #frame-tot was retired during #246-A1 when the frame
        # strip moved into the embed; see EMBED_OWNED_IDS in
        # test_trajectory_inspector_partial.py.)
        page.wait_for_selector("#viewer", timeout=5000)
        page.wait_for_function(
            """() => {
                const el = document.querySelector('.frame-counter');
                if (!el) return false;
                const ix = el.textContent.indexOf('/');
                if (ix < 0) return false;
                const total = parseInt(
                    el.textContent.slice(ix + 1).trim(), 10);
                return total > 0;
            }""",
            timeout=5000,
        )

        # Intercept subsequent /api/watch/data requests.
        watch_data_calls = []

        def _on_request(req):
            if "/api/watch/data" in req.url:
                watch_data_calls.append(req.url)

        page.on("request", _on_request)

        # Dispatch the pageshow event.  In a real browser this fires
        # on bfcache restore; we synthesise it for the test so we
        # don't depend on Playwright's bfcache behaviour (which is
        # off by default in headless Chromium).
        page.evaluate("""() => {
            window.dispatchEvent(new PageTransitionEvent("pageshow", {
                persisted: true,
            }));
        }""")
        # Give pollOnce a tick to fire its fetch.
        page.wait_for_function(
            "() => true", timeout=1500,
        )
        page.wait_for_timeout(500)

        assert len(watch_data_calls) >= 1, (
            "pageshow dispatch did not trigger /api/watch/data; "
            "trajectory inspector's pollOnce handler is missing or "
            "broken"
        )

    def test_visibilitychange_triggers_watch_data_request(
            self, page, flask_server, project_with_trajectory):
        _open_results_and_select(
            page, flask_server, project_with_trajectory)
        page.wait_for_selector("#viewer", timeout=5000)
        # See pageshow test for the .frame-counter rationale.
        page.wait_for_function(
            """() => {
                const el = document.querySelector('.frame-counter');
                if (!el) return false;
                const ix = el.textContent.indexOf('/');
                if (ix < 0) return false;
                const total = parseInt(
                    el.textContent.slice(ix + 1).trim(), 10);
                return total > 0;
            }""",
            timeout=5000,
        )

        watch_data_calls = []

        def _on_request(req):
            if "/api/watch/data" in req.url:
                watch_data_calls.append(req.url)

        page.on("request", _on_request)

        page.evaluate("""() => {
            document.dispatchEvent(new Event("visibilitychange"));
        }""")
        page.wait_for_timeout(500)

        assert len(watch_data_calls) >= 1, (
            "visibilitychange dispatch did not trigger /api/watch/data; "
            "trajectory inspector's pollOnce handler is missing or "
            "broken"
        )


