"""The /results tab's file picker, in the browser, when the tab is shown
again.

A tab return -- ``visibilitychange`` to visible -- re-reads the folder the
panel is bound to (`web/results.md` § 2.1, *Come back to the browser tab*)
and re-announces the file on screen; the viewer showing it answers ready from
inside that announcement, so the picker's *Parsing…* line lifts at once, not
after its safety timer (§ 2.2; `web/presenters.md` § 3, step 4).

The menu is read only after the scan has landed (``_open_results``): the
picker bar is always visible, so its visibility says nothing about the scan.
"""
from __future__ import annotations

import re

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.fixture
def project_with_one_xyz(isolated_projects_root):
    """A folder holding one structure and nothing else -- no calculation, so
    the server offers no pick and the structure is shown only when a person
    picks it in the menu.  The structure viewer is the one that does NOT
    reload its own file when the tab comes back, so a rescan's
    re-announcement is the only thing that can end the picker's "Parsing…"
    line for it.  In a projects tree of the test's own, the one door
    pointed at it."""
    proj = isolated_projects_root / "myproj" / "structure" / "w"
    proj.mkdir(parents=True)
    (proj / "w.xyz").write_text(
        "3\nwater\nO 0.0 0.0 0.119\nH 0.0 0.757 -0.477\nH 0.0 -0.757 -0.477\n")
    return proj


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


#: Installed on the page before any of its own scripts, on every navigation:
#: counts the picker's announcements, which close each scan (`results.md`
#: § 2.2); and marks the task after `pageshow`, by which every `pageshow`
#: handler has run -- this listener is the first registered, so its timer is
#: queued before the page's own handlers run.
_WATCH_THE_PICKER = """(() => {
    window.__mbAnnounced = 0;
    window.__mbShown = false;
    window.addEventListener("pageshow", () => {
        setTimeout(() => { window.__mbShown = true; }, 0);
    });
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

    Not the picker bar's visibility: the bar is always visible, so waiting
    for it would return at once, mid-scan.  Needs `_WATCH_THE_PICKER`,
    which `_setup_modify_dir` installs."""
    page.goto(f"{base_url}/results")
    page.wait_for_function(
        "() => window.__mbShown && window.__mbAnnounced >= 1", timeout=10000)


def _setup_modify_dir(page, base_url, dir_path):
    """The sidebar walked to ``dir_path`` on another tab, by its rows -- the
    most recent place, where the Results tab opens (`web/projects.md` § 2)
    -- with `_WATCH_THE_PICKER` installed for every page after."""
    from support.sidebar import walk_to
    page.add_init_script(_WATCH_THE_PICKER)
    page.goto(f"{base_url}/molbuilder")
    walk_to(page, dir_path)


# --------------------------------------------------------------------- #
#  Tests                                                                #
# --------------------------------------------------------------------- #


class TestVisibilityChangeForcesRescan:
    """``visibilitychange`` -> visible (the tab shown again after being in
    the background) re-reads the folder the panel is bound to
    (`web/results.md` § 2.1, *Come back to the browser tab*)."""

    def test_a_rescan_of_the_file_on_screen_does_not_leave_parsing_up(
            self, page, flask_server, project_with_one_xyz):
        """A tab return rescans the folder and re-announces the file already
        on screen, and its viewer answers ready at once -- from inside that
        announcement -- which lifts the "Parsing…" line (`web/results.md`
        § 2.1-§ 2.2; `web/presenters.md` § 3, step 4).  The picker started
        that line only AFTER announcing, so the answer met nothing to clear
        and the line sat busy for its whole timer (the Results-tab review,
        2026-09-28).

        MUTATION THIS MUST FAIL AGAINST: `_startParseStatus` called after
        the dispatch in `_emitFileSelected`.
        """
        from playwright.sync_api import expect
        proj = project_with_one_xyz
        _setup_modify_dir(page, flask_server, proj)
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
        # THE TAB SHOWN AGAIN, as the browser says it: the event a return
        # fires, the page visible.  Dispatched here because headless
        # Chromium never changes a page's visibility -- measured 2026-11-07,
        # Playwright 1.63, headless shell and new headless alike: another
        # page brought to the front, the window minimized and restored, the
        # page frozen and resumed, and `visibilityState` stayed "visible"
        # with no event fired; nor does it restore from the back/forward
        # cache (`pageshow` never `persisted`).
        announced = page.evaluate("() => window.__mbAnnounced")
        page.evaluate(
            "() => document.dispatchEvent(new Event('visibilitychange'))")
        # IT RESCANNED: the announcement that closes a scan.
        page.wait_for_function("(n) => window.__mbAnnounced > n",
                               arg=announced, timeout=10000)
        # ...AND THE LINE LIFTED AT ONCE, well inside the 8 s safety timer
        # that ends a "Parsing…" no answer met (`file-picker.js`).
        expect(page.locator("#results-file-picker-meta")).not_to_have_class(
            re.compile(r"\bis-busy\b"), timeout=1000)
