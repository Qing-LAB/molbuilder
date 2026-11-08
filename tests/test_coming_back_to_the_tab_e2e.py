"""Coming back to the tab asks once: a page shown again from the browser's
cache, or a tab made visible again, sends one `GET /api/watch/data` at once,
live run or not (`web/trajectory.md` § 5, *Coming back to the tab asks
once*), and the Results tab's picker re-reads its folder once
(`web/results.md` § 2.1, *Come back to the browser tab*) -- coming back being
the page turning visible, the one event each listens to.

THE SILENT FAILURES: the trajectory viewer's 15 s poll is a timer, and timers
are paused while a page sits in the back/forward cache -- a restored page kept
the frames it held when the person left, and a frame written while they were
away showed only at the next tick (2026-06-02, #192); and a restore, which
fires `visibilitychange` (visible) and then `pageshow` (`persisted`), heard by
a listener on each, scanned the folder twice, the first scan thrown away.

The run is a real one, made once in this pass on the road
(`support/real_runs.py`), and the Results tab opened on it as a person opens
it: the file picked in the sidebar, the tab chosen in the tab bar.  The page
restored from the cache is the person's own way back: another tab chosen in
the tab bar, then Back -- in a browser that keeps a page left, which the one
the suite's ``page`` runs in does not (:func:`caching_browser`).  A tab made
visible again is the event the viewer listens for, dispatched in the page,
because no action makes it in a headless browser: a page put behind
another, in the headless shell and in the full Chromium's headless mode,
stays `visible` (`process/testing.md` § 5).
"""
from __future__ import annotations

import pytest

from _road import conda_hook, env_available

pytestmark = [
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
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.fixture(scope="module")
def live_output(real_co2_vibration):
    """A real run's output -- the trajectory inspector's primary file type
    -- made on the road with the real SIESTA, once in this pass
    (`support/real_runs.py`): a CO2 relaxation, the `relax` call of a
    vibration's run in phases, its moves the frames -- the output the
    Results tab opens in its folder (`runs.openable`)."""
    from molbuilder.runs import openable
    from molbuilder.siesta.vibration_phases import RELAX
    return openable(real_co2_vibration.call("freq", RELAX))[0]


@pytest.fixture(scope="module")
def caching_browser(launch_browser):
    """A headless Chromium that keeps a page left in its back/forward cache,
    as a person's browser does: the full Chromium (``channel="chromium"``),
    launched without the switch Playwright adds to turn the cache off
    (``--disable-back-forward-cache``).  Measured 2026-11-07, Chromium 153,
    a page left for `/documents` and gone back to: the headless shell the
    suite's ``page`` runs in arrives anew, the switch taken out or not
    (`persisted` false, the reason masked); this one comes back as it was
    left (`persisted` true, on `/results` and `/molbuilder`)."""
    browser = launch_browser(
        channel="chromium",
        ignore_default_args=["--disable-back-forward-cache"])
    yield browser
    browser.close()


@pytest.fixture
def project_with_trajectory(real_co2_vibration, live_output, monkeypatch):
    """The run's output, its projects tree the server's (the one door,
    `projects_root`, read as each request comes): the absolute path
    string."""
    from conftest import point_the_one_door_at
    point_the_one_door_at(real_co2_vibration.tree.parent, monkeypatch.setenv)
    return str(live_output)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _open_results_and_select(page, base_url, file_path):
    """The file picked in the sidebar on the Molbuilder tab, by its row,
    then the Results tab opened from the tab bar: a tab never visited
    starts at the most recent place anywhere (`web/projects.md` § 2), and
    the Results tab binds to it and opens its result (`web/results.md`
    § 2.1, *Open the tab*)."""
    from pathlib import Path as P
    from support.sidebar import walk_to
    page.goto(f"{base_url}/molbuilder")
    walk_to(page, P(file_path))
    page.locator('.app-tabs a[href="/results"]').click()
    page.wait_for_selector("#inspector-host", timeout=5000)


# --------------------------------------------------------------------- #
#  Coming back to the tab: one scan, one ask                            #
# --------------------------------------------------------------------- #


def _loaded(page, base_url, file_path):
    """The Results tab open on ``file_path`` (:func:`_open_results_and_select`)
    and its trajectory loaded: the frame bar counts its frames."""
    _open_results_and_select(page, base_url, file_path)
    page.wait_for_selector("#viewer-host", timeout=5000)
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.molviewer-frames-counter');
            if (!el) return false;
            const ix = el.textContent.indexOf('/');
            if (ix < 0) return false;
            return parseInt(el.textContent.slice(ix + 1).trim(), 10) > 0;
        }""",
        timeout=5000,
    )


def _asks_at_once(page, came_back, how):
    """``came_back()`` makes the page ask `/api/watch/data` within 10 s --
    armed before, so the ask cannot land before the wait."""
    try:
        with page.expect_request(
                lambda req: "/api/watch/data" in req.url, timeout=10_000):
            came_back()
    except Exception as exc:                                  # pragma: no cover
        raise AssertionError(
            f"coming back to the tab ({how}) did not ask /api/watch/data "
            f"within 10 s") from exc


def test_a_page_restored_from_the_cache_asks_the_server_once(
        caching_browser, flask_server, project_with_trajectory):
    """A page shown again from the browser's back/forward cache makes ONE
    scan of the bound folder (`/api/results/dir`, `web/results.md` § 2.1-
    § 2.2) and ONE `/api/watch/data`, at once, live run or not
    (`web/trajectory.md` § 5) -- this run finished, so no poll is left to ask
    in its place.  The person leaves for the Molbuilder tab by the tab bar
    and comes back by Back; the page that comes back is the one left, not a
    new load -- its navigation is still the first one, not a back/forward
    load (a new load asks as it opens its file, whatever the listener does).

    A RESTORE TURNS THE PAGE VISIBLE BEFORE IT SHOWS IT: the page hears
    `visibilitychange` (visible), then `pageshow` (`persisted`) -- measured
    2026-11-07, Chromium 153 -- so the picker, listening to each, scanned the
    folder twice.  The viewer listened to each too, but its own guard stopped
    the second ask (a poll already in flight), so its count of one holds
    either way: what it pins is that the viewer asks at all.  Every request
    the restore makes is recorded, a scan the page aborted too, and counted
    once both answers have landed.

    MUTATION THIS MUST FAIL AGAINST: the viewer's `visibilitychange`
    listener left out, or asking only while the run is live; the picker
    scanning on the restore's `pageshow` as well."""
    page = caching_browser.new_page()
    try:
        _loaded(page, flask_server, project_with_trajectory)
        page.locator('.app-tabs a[href="/molbuilder"]').click()
        page.wait_for_url("**/molbuilder")
        asked = []
        page.on("request", lambda req: asked.append(req.url))
        with page.expect_response(
                lambda res: "/api/results/dir" in res.url, timeout=10_000), \
             page.expect_response(
                lambda res: "/api/watch/data" in res.url, timeout=10_000):
            page.go_back(wait_until="commit")
        assert page.evaluate("() => performance.getEntriesByType("
                             "'navigation')[0].type") == "navigate", (
            "Back loaded the page anew: the browser did not keep it")
        scans = [u for u in asked if "/api/results/dir" in u]
        polls = [u for u in asked if "/api/watch/data" in u]
        assert (len(scans), len(polls)) == (1, 1), (
            f"coming back by Back made {len(scans)} scan(s) of the folder "
            f"and {len(polls)} ask(s) of /api/watch/data, not one of each: "
            f"{scans + polls}")
    finally:
        page.close()


def test_a_tab_made_visible_again_asks_the_server_at_once(
        page, flask_server, project_with_trajectory):
    """A tab made visible again asks `/api/watch/data` at once, live run or
    not (`web/trajectory.md` § 5).  The event is dispatched in the page: no
    action makes it in a headless browser (the module's header).

    MUTATION THIS MUST FAIL AGAINST: the viewer's `visibilitychange`
    listener left out, or asking only while the run is live."""
    _loaded(page, flask_server, project_with_trajectory)
    _asks_at_once(page, lambda: page.evaluate(
        "() => document.dispatchEvent(new Event('visibilitychange'))"),
        "a tab made visible again")
