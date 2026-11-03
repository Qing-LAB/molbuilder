"""Coming back to the tab asks once: a page shown again from the browser's
cache, or a tab made visible again, sends one `GET /api/watch/data` at once,
live run or not (`web/trajectory.md` § 5, *Coming back to the tab asks
once*).

THE SILENT FAILURE (2026-06-02, #192): the trajectory viewer's 15 s poll is a
timer, and timers are paused while a page sits in the back/forward cache --
a restored page kept the frames it held when the person left, and a frame
written while they were away showed only at the next tick.

The run is a real one, made once in this pass on the road
(`support/real_runs.py`); the tab return is the event the viewer listens
for, dispatched in the page (headless Chromium keeps no back/forward cache).
"""
from __future__ import annotations

import pytest

from _road import conda_hook, env_available

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
def live_output(real_co2_vibration):
    """A real run's output -- the trajectory inspector's primary file type
    -- made on the road with the real SIESTA, once in this pass
    (`support/real_runs.py`): a CO2 relaxation, the `relax` call of a
    vibration's run in phases, its moves the frames -- the output the
    Results tab opens in its folder (`runs.openable`)."""
    from molbuilder.runs import openable
    from molbuilder.siesta.vibration_phases import RELAX
    return openable(real_co2_vibration.call("freq", RELAX))[0]


@pytest.fixture
def project_with_trajectory(real_co2_vibration, live_output, monkeypatch):
    """The run's output, its projects tree the only picker root: the
    absolute path string."""
    _register_tmp_as_picker_root(real_co2_vibration.tree, monkeypatch)
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
#  The trajectory viewer: a tab return asks once                        #
# --------------------------------------------------------------------- #


#: The two ways a person comes back to the tab, as the page hears them: a
#: page restored from the back/forward cache, and a tab made visible again.
_TAB_RETURNS = {
    "pageshow": """() => window.dispatchEvent(
        new PageTransitionEvent("pageshow", { persisted: true }))""",
    "visibilitychange": """() => document.dispatchEvent(
        new Event("visibilitychange"))""",
}


@pytest.mark.parametrize("event", list(_TAB_RETURNS))
def test_coming_back_to_the_tab_asks_the_server_at_once(
        page, flask_server, project_with_trajectory, event):
    """A tab return asks `/api/watch/data` at once, live run or not
    (`web/trajectory.md` § 5) -- this run finished, so no poll is left to
    ask in its place.

    MUTATION THIS MUST FAIL AGAINST: the viewer's `pageshow` or
    `visibilitychange` listener left out, or asking only while the run is
    live."""
    _open_results_and_select(page, flask_server, project_with_trajectory)
    # The trajectory loaded: the frame bar counts its frames.
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
    # Armed before the event, so the ask cannot land before the wait.
    try:
        with page.expect_request(
                lambda req: "/api/watch/data" in req.url, timeout=10_000):
            page.evaluate(_TAB_RETURNS[event])
    except Exception as exc:                                  # pragma: no cover
        raise AssertionError(
            f"coming back to the tab ({event}) did not ask /api/watch/data "
            f"within 10 s") from exc
