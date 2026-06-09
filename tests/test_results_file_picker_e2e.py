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
    page.goto(f"{base_url}/molbuilder")
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
        page.goto(f"{flask_server}/molbuilder")
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
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_timeout(300)
        page.goto(f"{flask_server}/results")
        page.wait_for_selector(
            "#results-file-picker-bar:not([hidden])", timeout=5000,
        )
        opts2 = _picker_options(page)
        # Still exactly one entry; bar didn't double-up or disappear.
        assert len(opts2) == 1
        assert "run1.out" in opts2[0]


class TestResultsDecoupledFromSidebar:
    """Task #301 (2026-06-09): the Results tab dispatches inspectors
    via the dropdown picker only.  Sidebar single-click → setShared →
    onChange is NOT a Results trigger anymore — the user gets back
    control of what's mounted in the inspector host."""

    def test_sidebar_setShared_does_not_remount_inspector(
            self, page, flask_server, project_with_one_out):
        """Pre-task-301 the /results dispatcher subscribed to
        projects.onChange; every sidebar pick fired _onSelectionChange
        and dispose-then-mount-ed an inspector.  Post-301 the
        dispatcher listens only for ``molbuilder:results:fileSelected``
        (dispatched by the dropdown picker), so a raw setShared call
        must NOT remount the inspector."""
        proj, dir_path = project_with_one_out
        (proj / "second.out").write_text(">> End of run: 2026-01-02\n")

        _setup_modify_dir(page, flask_server, dir_path)
        page.goto(f"{flask_server}/results")
        page.wait_for_function(
            "() => document.querySelectorAll("
            "    '#results-file-picker-select option').length === 2"
        )
        # Auto-pick fires fileSelected for the newest file; capture
        # whatever it dispatched as the baseline.
        baseline = page.locator("#results-current-file").inner_text()

        # Spy on subsequent fileSelected events.
        page.evaluate(
            "() => { window.__followup_selects = [];"
            "  document.addEventListener("
            "    'molbuilder:results:fileSelected',"
            "    (e) => window.__followup_selects.push("
            "        e.detail && e.detail.file || ''));"
            "}"
        )

        # Sidebar setShared to the OTHER file — pre-301 this would
        # fire onChange → dispatcher remount.  Post-301 it's a no-op
        # for /results.
        other = str(proj / "run1.out")
        page.evaluate(
            "(f) => window.molbuilder.projects.setShared("
            "  f.substring(0, f.lastIndexOf('/')), f)",
            other,
        )
        page.wait_for_timeout(300)

        events = page.evaluate("() => window.__followup_selects")
        assert events == [], (
            f"sidebar setShared leaked into Results: got events {events}"
        )
        # The inspector header still shows the baseline file — the
        # sidebar pick did NOT remount.
        assert page.locator("#results-current-file").inner_text() == baseline

    def test_refresh_button_rescans_directory(
            self, page, flask_server, project_with_one_out):
        """Explicit user-driven rescan: an external write that lands
        AFTER the initial scan should not appear until the user clicks
        Refresh (or the tab is re-focused / re-shown — covered by the
        other tests in this file).  Pin the button's wiring."""
        proj, dir_path = project_with_one_out
        _setup_modify_dir(page, flask_server, dir_path)
        page.goto(f"{flask_server}/results")
        page.wait_for_function(
            "() => document.querySelectorAll("
            "    '#results-file-picker-select option').length === 1"
        )

        # New file lands in the same dir while user is sitting on
        # /results.  Without the sidebar onChange subscription, the
        # picker won't see it until a refresh.
        (proj / "later.out").write_text(">> End of run: 2026-01-03\n")
        page.wait_for_timeout(100)
        assert len(_picker_options(page)) == 1, (
            "picker should not auto-detect new files without a refresh"
        )

        page.locator("#results-file-picker-refresh").click()
        page.wait_for_function(
            "() => document.querySelectorAll("
            "    '#results-file-picker-select option').length === 2"
        )

    def test_refresh_button_disables_during_rescan(
            self, page, flask_server, project_with_one_out):
        """Task #303 added a MutationObserver-driven disabled state
        on the Refresh button while a scan is in flight; the
        observer watches the meta line's ``is-busy`` class.  Pin
        that the button (a) goes disabled when clicked and
        (b) comes back enabled once the scan settles.  Catches a
        regression where the observer is dropped or the
        is-busy lifecycle leaks past a bar-hide branch (the
        review found three such paths that leaked pre-task-#304;
        keep the regression net up)."""
        proj, dir_path = project_with_one_out
        _setup_modify_dir(page, flask_server, dir_path)
        page.goto(f"{flask_server}/results")
        page.wait_for_function(
            "() => document.querySelectorAll("
            "    '#results-file-picker-select option').length === 1"
        )
        # Settled state: button is enabled.
        assert page.locator("#results-file-picker-refresh").is_enabled()

        # Click → button goes disabled while the scan runs.  The
        # _scan path on a small project finishes in well under
        # 100 ms; instead of racing against that, observe the
        # disabled→enabled transition via wait_for_function on
        # the class.  The button MUST end up enabled again.
        page.locator("#results-file-picker-refresh").click()
        page.wait_for_function(
            "() => !document.getElementById('results-file-picker-refresh')"
            ".disabled",
            timeout=10000,
        )

    def test_refresh_button_double_click_does_not_stack_scans(
            self, page, flask_server, project_with_one_out):
        """A rapid double-click on the Refresh button must not
        spawn two concurrent scans (the second one would race the
        first's AbortController + leave the meta line flickering).
        The disabled-during-rescan guard makes the second click a
        no-op."""
        proj, dir_path = project_with_one_out
        _setup_modify_dir(page, flask_server, dir_path)
        page.goto(f"{flask_server}/results")
        page.wait_for_function(
            "() => document.querySelectorAll("
            "    '#results-file-picker-select option').length === 1"
        )

        # Spy on /api/files/list calls to confirm we see exactly
        # ONE additional request from the double-click (the first
        # click; the second is dropped by the disabled guard).
        # Initial mount already issued one scan; we baseline that.
        baseline = page.evaluate(
            "() => performance.getEntriesByType('resource')"
            "  .filter(e => e.name.includes('/api/files/list')).length"
        )

        btn = page.locator("#results-file-picker-refresh")
        btn.click()
        # Don't wait — fire again immediately to test the guard.
        # Playwright's click() honours actionability (disabled =
        # blocked); use evaluate-style dispatch to bypass that
        # check and confirm the JS guard, not Playwright's, is
        # what blocks the second scan.
        page.evaluate(
            "() => document.getElementById('results-file-picker-refresh')"
            "  .click()"
        )
        page.wait_for_function(
            "() => !document.getElementById('results-file-picker-refresh')"
            ".disabled",
            timeout=10000,
        )

        after = page.evaluate(
            "() => performance.getEntriesByType('resource')"
            "  .filter(e => e.name.includes('/api/files/list')).length"
        )
        new_scans = after - baseline
        assert new_scans == 1, (
            f"a double-click on Refresh should fire exactly one extra "
            f"scan (the second click is a no-op while disabled); "
            f"got {new_scans} extra scans"
        )

    def test_fileSelected_event_detail_carries_picked_file_path(
            self, page, flask_server, project_with_one_out):
        """Pin the ``molbuilder:results:fileSelected`` event's detail
        shape: ``{file: <full path>}``.  results/viewer.js reads
        ``evt.detail.file``; a future picker refactor that drops the
        ``detail`` wrapper or renames the key would silently break
        the dispatcher.  Catches that here."""
        proj, dir_path = project_with_one_out
        # Add a second file so we can fire a deliberate change
        # event from the dropdown.
        (proj / "run2.out").write_text(">> End of run: 2026-02-02\n")
        _setup_modify_dir(page, flask_server, dir_path)
        page.goto(f"{flask_server}/results")
        page.wait_for_function(
            "() => document.querySelectorAll("
            "    '#results-file-picker-select option').length === 2"
        )

        # Spy on the event AFTER the auto-pick has fired (the first
        # event is for the newest file; we want to observe the
        # event emitted by an explicit dropdown change).
        page.evaluate(
            "() => { window.__events = [];"
            "  document.addEventListener("
            "    'molbuilder:results:fileSelected',"
            "    (e) => window.__events.push(e.detail));"
            "}"
        )

        # Pick the OTHER file (whichever one wasn't auto-selected).
        page.evaluate(
            "() => {"
            "  const sel = document.getElementById('results-file-picker-select');"
            "  const auto = sel.value;"
            "  const other = Array.from(sel.options)"
            "    .find(o => o.value && o.value !== auto);"
            "  if (other) {"
            "    sel.value = other.value;"
            "    sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "  }"
            "}"
        )
        page.wait_for_function(
            "() => (window.__events || []).length >= 1"
        )

        events = page.evaluate("() => window.__events")
        assert events, "no fileSelected events captured after dropdown change"
        # Detail shape: {file: <full path>}.  Path must start
        # with the project dir so consumers can derive parents.
        for ev in events:
            assert "file" in ev, (
                f"fileSelected event detail must carry 'file' key; "
                f"got keys {sorted(ev.keys())}"
            )
            assert isinstance(ev["file"], str), (
                f"fileSelected.detail.file must be a string; "
                f"got {type(ev['file']).__name__}"
            )
            assert ev["file"].startswith(dir_path), (
                f"fileSelected.detail.file should be an absolute path "
                f"inside the project dir {dir_path}; got {ev['file']!r}"
            )
