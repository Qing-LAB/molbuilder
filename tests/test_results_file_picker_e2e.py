"""End-to-end Playwright tests for the /results tab-level file picker.

Regression tests for the 2026-06-02 "stale dropdown / no scan on
first click" bug.  Symptoms reported by the user:

    "the results tab does not automatically scan the current directory
     when first clicked.  if we are already inside a project directory,
     we will need to manually get out of the directory and get back in
     to have the results tab to refresh"

Root cause was TWO compounding issues in ``lib/results/file-picker.js``:

  1. The picker rescanned only when the sessionStorage directory
     CHANGED (``if (dir !== lastScannedDir)`` in _onSelectionChange).
     A re-visit to /results with the same dir hit the same-dir branch
     and reused the stale ``cachedResults`` -- which never reflected
     files generated in another tab since the previous scan.

  2. Even when a fresh scan DID fire, the browser HTTP cache served
     the previous ``/api/files/list`` response for the same URL.  Same
     URL + default cache policy = ``304 Not Modified`` -> the picker
     saw last visit's entries, not what was actually on disk.

The fix:

  * ``pageshow`` + ``visibilitychange`` event hooks call ``_forceRescan()``
    which resets ``lastScannedDir`` and re-fires ``_onSelectionChange``
    so a same-dir re-entry triggers the rescan branch.
  * ``fetch(..., { cache: "no-store" })`` on the directory listing so
    the rescan actually reaches the server.

These tests exercise both fixes end-to-end.
"""
from __future__ import annotations

import threading
import time

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


@pytest.fixture
def project_with_one_out(tmp_path, monkeypatch):
    """A project dir under tmp_path holding exactly one ``.out`` result
    file.  Returns ``(project_dir, dir_path_str)``."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "myproj" / "spectra" / "test"
    proj.mkdir(parents=True)
    (proj / "run1.out").write_text(">> End of run: 2026-01-01\n")
    return proj, str(proj)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _picker_options(page):
    return page.evaluate(
        "() => Array.from(document.querySelectorAll("
        "    '#results-file-picker-select option'))"
        ".map(o => o.textContent)"
    )


def _option_basenames(opts):
    """Drop the ``(N seconds ago)`` suffix; return the bare filenames
    so timing differences don't flake the assertion."""
    out = []
    for opt in opts:
        space_idx = opt.find(" (")
        out.append(opt[:space_idx] if space_idx > 0 else opt)
    return out


def _setup_modify_dir(page, base_url, dir_path):
    page.goto(f"{base_url}/modify")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.projects "
        "      && typeof window.molbuilder.projects.setShared "
        "             === 'function'"
    )
    page.evaluate(
        "(d) => window.molbuilder.projects.setShared(d, '')",
        dir_path,
    )
    page.wait_for_timeout(300)


# --------------------------------------------------------------------- #
#  Tests                                                                #
# --------------------------------------------------------------------- #


class TestStaleResultsRefresh:
    """The user's reported bug: after generating a new result file in
    another tab, returning to /results should show it without needing
    to manually navigate the sidebar out and back in."""

    def test_new_result_file_appears_on_back_to_results(
            self, page, flask_server, project_with_one_out):
        """The end-to-end scenario from the user's report:
            1. Set dir on /modify.
            2. Visit /results -- picker scans, shows run1.out only.
            3. Navigate back to /modify.
            4. A new result file (run2.out) appears on disk.
            5. Navigate back to /results.
            6. Picker MUST show both run1.out and run2.out without
               any sidebar interaction.
        """
        proj_dir, dir_str = project_with_one_out
        _setup_modify_dir(page, flask_server, dir_str)

        # First /results visit -- only run1.out exists.
        page.goto(f"{flask_server}/results")
        page.wait_for_selector(
            "#results-file-picker-bar:not([hidden])", timeout=5000,
        )
        opts1 = _option_basenames(_picker_options(page))
        assert opts1 == ["run1.out"]

        # Go back to /modify -- as if the user opened the generator tab.
        page.goto(f"{flask_server}/modify")
        page.wait_for_timeout(300)

        # A new result file appears on disk (simulating the user
        # generating output in /spectra or /modify).
        (proj_dir / "run2.out").write_text(
            ">> End of run: 2026-01-02\n"
        )
        # Filesystem mtime resolution: give it a beat so the new file
        # has a distinguishable mtime + the OS has flushed it.
        time.sleep(0.5)

        # Re-visit /results.  Pre-fix this returned to the cached scan
        # (only run1.out).  Post-fix both files appear.
        page.goto(f"{flask_server}/results")
        page.wait_for_selector(
            "#results-file-picker-bar:not([hidden])", timeout=5000,
        )
        # Wait for the rescan to land -- the force-rescan on pageshow
        # fires the second scan; allow a tick for fetch + DOM update.
        page.wait_for_function(
            "() => Array.from(document.querySelectorAll("
            "    '#results-file-picker-select option')).length >= 2",
            timeout=5000,
        )
        opts2 = _option_basenames(_picker_options(page))
        # Both files present.  Newest first per the picker's mtime sort.
        assert set(opts2) == {"run1.out", "run2.out"}


class TestPageshowForcesRescan:
    """The mechanism behind the user-facing fix: hitting ``pageshow``
    must trigger _forceRescan() even when sessionStorage's dir is
    unchanged.  Pin the property at the JS event level."""

    def test_pageshow_event_triggers_listing_fetch(
            self, page, flask_server, project_with_one_out):
        proj_dir, dir_str = project_with_one_out
        _setup_modify_dir(page, flask_server, dir_str)
        page.goto(f"{flask_server}/results")
        page.wait_for_selector(
            "#results-file-picker-bar:not([hidden])", timeout=5000,
        )

        # Add a new file + dispatch a pageshow event manually.  This
        # mimics what the browser does on bfcache restore.
        (proj_dir / "run2.out").write_text(
            ">> End of run: 2026-01-02\n"
        )
        time.sleep(0.5)
        page.evaluate("""() => {
            window.dispatchEvent(new PageTransitionEvent("pageshow", {
                persisted: true,
            }));
        }""")
        page.wait_for_function(
            "() => Array.from(document.querySelectorAll("
            "    '#results-file-picker-select option')).length >= 2",
            timeout=5000,
        )
        opts = _option_basenames(_picker_options(page))
        assert set(opts) == {"run1.out", "run2.out"}


class TestVisibilityChangeForcesRescan:
    """``visibilitychange`` -> visible (tab gains focus after being
    in the background) is the secondary refresh trigger.  Same fix,
    different event."""

    def test_visibility_visible_event_triggers_rescan(
            self, page, flask_server, project_with_one_out):
        proj_dir, dir_str = project_with_one_out
        _setup_modify_dir(page, flask_server, dir_str)
        page.goto(f"{flask_server}/results")
        page.wait_for_selector(
            "#results-file-picker-bar:not([hidden])", timeout=5000,
        )

        (proj_dir / "run2.out").write_text(
            ">> End of run: 2026-01-02\n"
        )
        time.sleep(0.5)
        # Simulate a visibilitychange -> visible event.  The picker
        # reads document.visibilityState which is "visible" by default
        # in a foregrounded Playwright page, so the handler will rescan.
        page.evaluate("""() => {
            document.dispatchEvent(new Event("visibilitychange"));
        }""")
        page.wait_for_function(
            "() => Array.from(document.querySelectorAll("
            "    '#results-file-picker-select option')).length >= 2",
            timeout=5000,
        )
        opts = _option_basenames(_picker_options(page))
        assert set(opts) == {"run1.out", "run2.out"}


class TestNoStaleScanOnRevisit:
    """Even the same-dir, same-files revisit case must NOT show stale
    relative-time text.  After a rescan the meta line + dropdown
    re-render from the fresh response, not from the cached one."""

    def test_revisit_replaces_dropdown_in_place(
            self, page, flask_server, project_with_one_out):
        proj_dir, dir_str = project_with_one_out
        _setup_modify_dir(page, flask_server, dir_str)
        page.goto(f"{flask_server}/results")
        page.wait_for_selector(
            "#results-file-picker-bar:not([hidden])", timeout=5000,
        )
        opts1 = _picker_options(page)
        assert len(opts1) == 1   # only run1.out

        # Re-visit /results without changing anything.
        page.goto(f"{flask_server}/modify")
        page.wait_for_timeout(300)
        page.goto(f"{flask_server}/results")
        page.wait_for_selector(
            "#results-file-picker-bar:not([hidden])", timeout=5000,
        )
        opts2 = _picker_options(page)
        # Still exactly one entry; bar didn't double-up or disappear.
        assert len(opts2) == 1
        assert "run1.out" in opts2[0]
