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

  * a fresh visit scans once, at mount; a restore from the back/forward
    cache (``pageshow`` with ``persisted``) and ``visibilitychange`` ->
    visible re-read the folder the panel is bound to (``_rescanBound``).
  * ``fetch(..., { cache: "no-store" })`` on the directory listing so
    the rescan actually reaches the server.

These tests exercise both fixes end-to-end.  Each reads the menu only after
the scan has landed (``_open_results``): the picker bar is always visible, so
its visibility says nothing about the scan.
"""
from __future__ import annotations


import pytest


pytestmark = pytest.mark.e2e

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
def project_with_one_xyz(tmp_path, monkeypatch):
    """A folder holding one structure and nothing else -- no calculation, so
    the server offers no pick and the structure is shown only when a person
    picks it in the menu.  The structure viewer is the one that does NOT
    reload its own file when the tab comes back, so a rescan's
    re-announcement is the only thing that can end the picker's "Parsing…"
    line for it."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "myproj" / "structure" / "w"
    proj.mkdir(parents=True)
    (proj / "w.xyz").write_text(
        "3\nwater\nO 0.0 0.0 0.119\nH 0.0 0.757 -0.477\nH 0.0 -0.757 -0.477\n")
    return proj, str(proj)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


#: Installed on the page before any of its own scripts, on every navigation:
#: counts the page's folder scans -- its calls to the picker's door -- and
#: the picker's announcements, which close each scan (`results.md` § 2.2);
#: and marks the task after `pageshow`, by which every `pageshow` handler has
#: run -- this listener is the first registered, so its timer is queued
#: before the page's own handlers run.
_WATCH_THE_PICKER = """(() => {
    window.__mbScans = 0;
    window.__mbAnnounced = 0;
    window.__mbShown = false;
    window.addEventListener("pageshow", () => {
        setTimeout(() => { window.__mbShown = true; }, 0);
    });
    const fetch0 = window.fetch;
    window.fetch = function (input) {
        const url = String((input && input.url) || input || "");
        if (url.indexOf("/api/results/dir") !== -1) window.__mbScans += 1;
        return fetch0.apply(this, arguments);
    };
    document.addEventListener("DOMContentLoaded", () => {
        const C = (window.molbuilder || {}).constants || {};
        document.addEventListener(C.EVENT_FILE_SELECTED,
                                  () => { window.__mbAnnounced += 1; });
    });
})();"""


def _open_results(page, base_url):
    """Open the tab and wait until the load is over -- its ``pageshow``
    handled -- and its scan has LANDED: the picker's announcement, which it
    makes once per scan.  What follows reads the menu that scan built.

    Not the picker bar's visibility: the bar is always visible (since
    2026-06-15), so waiting for it returned at once and the tests read the
    menu mid-scan.  Needs `_WATCH_THE_PICKER`, which `_setup_modify_dir`
    installs."""
    page.goto(f"{base_url}/results")
    page.wait_for_function(
        "() => window.__mbShown && window.__mbAnnounced >= 1", timeout=10000)


def _setup_modify_dir(page, base_url, dir_path):
    page.add_init_script(_WATCH_THE_PICKER)
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


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 12 tests here listed SIESTA outputs invented as text -- five classes whole
# (`TestStaleResultsRefresh`, `TestPageshowForcesRescan`, `TestOneScanPerVisit`,
# `TestResultsDecoupledFromSidebar`, `TestThePanelOwnsItsFolder`) and one test
# beside the test kept below (`process/testing.md` § 6).


class TestVisibilityChangeForcesRescan:
    """``visibilitychange`` -> visible (tab gains focus after being
    in the background) is the secondary refresh trigger.  Same fix,
    different event."""

    def test_a_rescan_of_the_file_on_screen_does_not_leave_parsing_up(
            self, page, flask_server, project_with_one_xyz):
        """A tab return rescans the folder and re-announces the file already
        on screen, and its viewer answers ready at once -- from inside that
        announcement.  The picker started its "Parsing…" line only AFTER
        announcing, so the answer met nothing to clear and the line sat
        busy for its whole timer (the Results-tab review, 2026-09-28).

        MUTATION THIS MUST FAIL AGAINST: `_startParseStatus` called after
        the dispatch in `_emitFileSelected`.
        """
        proj, dir_str = project_with_one_xyz
        _setup_modify_dir(page, flask_server, dir_str)
        _open_results(page, flask_server)
        xyz = str(proj / "w.xyz")
        page.wait_for_function(
            "(want) => [...document.querySelectorAll("
            "  '#results-file-picker-select option')].some(o => o.value === want)",
            arg=xyz, timeout=20000)
        # the person picks the structure; it mounts and says it is drawn
        page.select_option("#results-file-picker-select", value=xyz)
        page.wait_for_selector("#inspector-host .structure-status",
                               timeout=20000)
        page.wait_for_function("""() => {
            const m = document.getElementById("results-file-picker-meta");
            const s = document.querySelector("#inspector-host .structure-status");
            return m && !m.classList.contains("is-busy")
                && s && /Loaded/.test(s.textContent); }""", timeout=30000)
        page.evaluate("""() => {
            const m = document.getElementById("results-file-picker-meta");
            window.__seen = [];
            new MutationObserver(() => window.__seen.push(m.textContent))
                .observe(m, { childList: true, characterData: true,
                              subtree: true });
            document.dispatchEvent(new Event("visibilitychange")); }""")
        page.wait_for_timeout(3000)
        state = page.evaluate("""() => {
            const m = document.getElementById("results-file-picker-meta");
            return { seen: window.__seen, busy: m.classList.contains("is-busy"),
                     text: m.textContent }; }""")
        assert any("Scanning" in t for t in state["seen"]), state  # it rescanned
        assert not state["busy"], state


