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

  * ``lib/spectra/core.js`` -- mount-once, no reload button, and
    (until 2026-09-28) no auto-poll: it follows a run still going now,
    but a FINISHED result re-run in another tab was invisible until the
    user re-picked the file from the dropdown.

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

import pytest

from _road import (conda_hook, env_available, h2_relaxed_for_vibration,
                   live_siesta)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


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


@pytest.fixture(scope="module")
def live_output(isolated_projects_root_module, tmp_path_factory):
    """A real run's output -- the trajectory inspector's primary file type
    -- made on the road with the real SIESTA: an H2 relaxation, its moves
    the frames (`process/testing.md` § 6; a frozen output of an older run,
    copied into a folder of its own as ``run.out``, stood here until
    2026-10-06)."""
    tree = isolated_projects_root_module
    with live_siesta(tree, tmp_path_factory):
        yield (h2_relaxed_for_vibration(tree) / "01_relax" / "run-0"
               / "H2_01_relax-run0.out")


@pytest.fixture
def project_with_trajectory(live_output, monkeypatch):
    """The run's output, its projects tree the only picker root: the
    absolute path string."""
    root = next(p for p in live_output.parents if p.name == "projects")
    _register_tmp_as_picker_root(root, monkeypatch)
    return str(live_output)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _open_results_and_select(page, base_url, file_path):
    """Open /results with the sidebar driven into ``file_path``'s
    parent dir + the file selected.  Mirrors the user flow for
    landing on the trajectory inspector via a sidebar click."""
    from pathlib import Path as P
    parent = str(P(file_path).parent)
    page.goto(f"{base_url}/molbuilder")
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
        # /api/watch/load to complete.  MolView's frame bar (task #34)
        # renders a .molviewer-frames-counter element showing "<i+1> / <total>" —
        # i.e. the second number > 0 once reloadFrames has populated the
        # frame series and MolView shows the bar (frameCount > 1).
        page.wait_for_selector("#viewer-host", timeout=5000)
        page.wait_for_function(
            """() => {
                const el = document.querySelector('.molviewer-frames-counter');
                if (!el) return false;
                const ix = el.textContent.indexOf('/');
                if (ix < 0) return false;
                const total = parseInt(
                    el.textContent.slice(ix + 1).trim(), 10);
                return total > 0;
            }""",
            timeout=5000,
        )

        # Dispatch the pageshow event and wait for the resulting
        # ``/api/watch/data`` fetch via ``expect_request``: the
        # matcher is armed BEFORE the dispatch (no race window) and
        # returns as soon as a matching request fires.  A fixed
        # ``wait_for_timeout`` was previously flaky under CI load
        # when the fetch landed just past the 500 ms wall.
        #
        # In a real browser pageshow fires on bfcache restore; we
        # synthesise it here so the test doesn't depend on
        # Playwright's bfcache behaviour (off by default in
        # headless Chromium).
        try:
            with page.expect_request(
                lambda req: "/api/watch/data" in req.url,
                timeout=10_000,
            ):
                page.evaluate("""() => {
                    window.dispatchEvent(new PageTransitionEvent("pageshow", {
                        persisted: true,
                    }));
                }""")
        except Exception as exc:                                  # pragma: no cover
            raise AssertionError(
                "pageshow dispatch did not trigger /api/watch/data "
                "within 10 s; trajectory inspector's pollOnce handler "
                "is missing or broken."
            ) from exc

    def test_visibilitychange_triggers_watch_data_request(
            self, page, flask_server, project_with_trajectory):
        _open_results_and_select(
            page, flask_server, project_with_trajectory)
        page.wait_for_selector("#viewer-host", timeout=5000)
        # See pageshow test for the .frame-counter rationale.
        page.wait_for_function(
            """() => {
                const el = document.querySelector('.molviewer-frames-counter');
                if (!el) return false;
                const ix = el.textContent.indexOf('/');
                if (ix < 0) return false;
                const total = parseInt(
                    el.textContent.slice(ix + 1).trim(), 10);
                return total > 0;
            }""",
            timeout=5000,
        )

        # See the pageshow variant for why ``expect_request`` beats
        # a fixed ``wait_for_timeout`` here.
        try:
            with page.expect_request(
                lambda req: "/api/watch/data" in req.url,
                timeout=10_000,
            ):
                page.evaluate("""() => {
                    document.dispatchEvent(new Event("visibilitychange"));
                }""")
        except Exception as exc:                                  # pragma: no cover
            raise AssertionError(
                "visibilitychange dispatch did not trigger /api/watch/data "
                "within 10 s; trajectory inspector's pollOnce handler "
                "is missing or broken."
            ) from exc


