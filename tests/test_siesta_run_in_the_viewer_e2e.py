"""What the trajectory viewer states about a finished SIESTA run, each fact
from its one source (`model/parse.md` § 2a P-T2 and P-T4, `web/results.md`
§ 3a): the SCF line's seconds per iteration are the SCF-timing log's, read
by the reader the run record reads them by; the badge's "ended" time is the
output's own ``>> End of run``, and its "total" the run's own length, the
engine's wall its record states (`web/trajectory.md` § 4).

The run is the layered H2 relaxation's first `coarse` run, made once in this
pass on the road (`support/real_runs.layered`, `process/testing.md` tier 3),
opened in the Results tab as a person opens it, in its folder.
"""
from __future__ import annotations

import re
from datetime import datetime

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


@pytest.fixture(scope="module")
def finished(real_h2_layered):
    """The attempt directory of a finished H2 relaxation, made on the road
    (`support/real_runs.RealRun.attempt`) -- finished, by the one status
    door."""
    from molbuilder.jobset.runstatus import run_answer
    attempt = real_h2_layered.attempt("coarse")
    said = run_answer(attempt)
    assert (said["state"], said["detail"]) == ("finished", "ended, exit 0"), said
    return attempt


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, flask_server, folder, tree, monkeypatch):
    """The Results tab bound to ``folder``, its output shown, in a fresh
    page -- bound as a person binds it, by the sidebar's rows and Reload
    (`support.sidebar.results_on`), in the projects ``tree`` the run was
    made in, the one door pointed at it."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.sidebar import results_on
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    results_on(page, flask_server, folder)
    page.wait_for_function(
        "() => /Finished/.test((document.getElementById('run-state-label')"
        " || {}).textContent || '')"
        " && /SCF cycle/.test((document.getElementById('scf-status')"
        " || {}).textContent || '')", timeout=30000)


def _the_runs_own_times(attempt):
    """What the run record states -- the SCF-timing log's rate and the
    output's own end -- through the door the Results tab asks."""
    from molbuilder.runs import folder_answer
    time = folder_answer(attempt)["record"]["computation"]["time"]
    return time, datetime.fromisoformat(time["run_end_local"])


def test_in_its_folder_the_viewer_states_the_runs_rate_and_end(
        finished, real_h2_layered, page, flask_server, monkeypatch):
    """The SCF line's rate is the number the run record states -- the
    SCF-timing log, through one reader -- and the badge dates the run by its
    output's end and says its total as the record's engine wall, the Run
    panel's.

    THE SILENT FAILURES: the viewer computing its own rate; the badge's
    total read off a SIESTA timer line that times one SCF iteration, so a
    24 s run read *total 0s* (2.56.1).
    """
    from molbuilder.report_fields import span_words
    time, ended = _the_runs_own_times(finished)
    _open(page, flask_server, finished, real_h2_layered.tree, monkeypatch)
    scf = page.locator("#scf-status").inner_text()
    rate = re.search(r"([0-9.]+) s/iter", scf)
    assert rate, scf
    assert float(rate.group(1)) == pytest.approx(time["s_per_iter"],
                                                 rel=5e-3), (scf, time)
    detail = page.locator("#run-state-detail").inner_text()
    assert "ended" in detail and ended.strftime(":%M:%S") in detail, detail
    assert f"total {span_words(time['engine_elapsed_s'])}" in detail, (
        detail, time)


def test_the_scf_plot_draws_both_tests_siesta_requires(
        finished, real_h2_layered, page, flask_server, monkeypatch):
    """A SIESTA SCF is converged when dDmax is under `DM.Tolerance` AND dHmax
    under `SCF.H.Tolerance` (`web/trajectory.md` § 3.0): the residuals plot
    draws both, each against the tolerance the run's own deck states, and
    the SCF line quotes both.

    MUTATION THIS MUST FAIL AGAINST: the plot drawing one residual standing
    for the rest -- dHmax alone, as it did until 2.9.5."""
    from molbuilder.parse.fdf import parse_fdf_params
    from molbuilder.runs import run_of
    deck = parse_fdf_params(run_of(finished).deck.read_text(),
                            source="the deck")
    assert deck.dm_tolerance and deck.h_tolerance_ev, deck
    _open(page, flask_server, finished, real_h2_layered.tree, monkeypatch)
    drawn = page.evaluate(
        "() => { const el = document.getElementById('scf-gnorm-plot');"
        " return { names: (el.data || []).map(t => t.name),"
        "          lines: ((el.layout || {}).shapes || []).map(s => s.y0) }; }")
    assert "dDmax" in drawn["names"] and "dHmax (eV)" in drawn["names"], drawn
    assert sorted(drawn["lines"]) == pytest.approx(
        sorted([deck.dm_tolerance, deck.h_tolerance_ev])), drawn
    scf = page.locator("#scf-status").inner_text()
    assert "dDmax=" in scf and "dHmax=" in scf, scf


def test_every_element_the_viewer_draws_into_is_on_its_panel(
        finished, real_h2_layered, page, flask_server, monkeypatch):
    """Every element the trajectory viewer looks up on its panel answers, one
    element, at the moment it is asked (`process/code-audit.md` § 3.4) --
    as the run opens, and as its three force controls are used
    (`web/trajectory.md` § 1).

    THE SILENT FAILURE: an element the viewer draws into missing from the
    panel the server sends -- a plot, the badge or a control's readout
    stays blank, and nothing says why; or there twice, the viewer writing
    into one copy while the person reads the other.

    Every lookup the viewer makes on the Results tab's viewer host is
    recorded with how many elements answered it.

    MUTATIONS THIS MUST FAIL AGAINST: an element taken out of the panel;
    one written into it twice."""
    page.add_init_script("""(() => {
        const one = Element.prototype.querySelector;
        const all = Element.prototype.querySelectorAll;
        window.__asked = [];
        Element.prototype.querySelector = function (sel) {
            if (this.id === "inspector-host" && /^#[\\w-]+$/.test(sel))
                window.__asked.push([sel, all.call(this, sel).length]);
            return one.call(this, sel);
        };
    })();""")
    _open(page, flask_server, finished, real_h2_layered.tree, monkeypatch)
    page.locator("#force-scale").fill("2.5")
    page.locator("#force-min").fill("0.05")
    page.locator("#hide-frozen").check()
    asked = page.evaluate("() => window.__asked")
    assert asked, "the viewer asked its panel for nothing"
    wrong = sorted({f"{sel} answered by {n}" for sel, n in asked if n != 1})
    assert not wrong, wrong

