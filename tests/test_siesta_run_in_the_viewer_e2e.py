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
    from conftest import point_the_one_door_at
    from support.sidebar import results_on
    point_the_one_door_at(tree.parent, monkeypatch.setenv)
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


def test_the_status_line_says_the_box_is_the_runs(
        finished, real_h2_layered, page, flask_server, monkeypatch):
    """`web/trajectory.md` § 4: on an ordinary load the panel's status line
    says where the unit cell came from when the run supplied it -- the box
    frame 0's envelope states, the structure sent once (`web-api.md` § 1).

    THE SILENT FAILURE: the line read a copy of the box beside the
    envelope, and with the copy gone said nothing of a box drawn round the
    molecule that the person never set."""
    _open(page, flask_server, finished, real_h2_layered.tree, monkeypatch)
    status = page.locator("#trajectory-status").inner_text()
    assert "Unit cell from the run" in status, status


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


def test_each_chart_exports_what_it_draws(finished, real_h2_layered, page,
                                          flask_server, monkeypatch,
                                          tmp_path):
    """`web/results.md` § 1.2, `model/datafiles.md` § 9.7: each of the
    viewer's four charts carries its own Export, and its file holds what the
    chart draws -- the run's own numbers: each step's energy and largest
    force the run door's (`read_steps`), to the 0.0001 eV SIESTA prints; the
    force tolerance the run's record's, and both SCF tolerances the deck's;
    the SCF energy of the step shown the run door's rows of that step.  Its
    picture is a PNG the size of the chart on screen; its tables lie beside
    their README.  The energy and the largest-force charts keep Plotly's
    toolbar, its buttons the view's alone, the Export above it.

    THE SILENT FAILURE: a file whose numbers are not the chart's -- another
    step's, a curve or a threshold lost -- or one naming no chart; a picture
    stretched to the box around the chart; an Export
    over the toolbar or the curve."""
    import io
    import zipfile

    import numpy as np
    from molbuilder.parse.fdf import parse_fdf_params
    from molbuilder.postprocess import read_chart_data, read_steps
    from molbuilder.runs import run_of
    from support.pages import chart_exported, export_clear_of_the_chart
    steps = read_steps(finished)
    deck = parse_fdf_params(run_of(finished).deck.read_text(),
                            source="the deck")
    _open(page, flask_server, finished, real_h2_layered.tree, monkeypatch)

    def drawn(path, group):
        """The file read back through its reader (`read_chart_data`): its
        root, and each curve of ``group`` -- its finite values, whether
        hidden."""
        data = read_chart_data(path)
        v = data["groups"][group]["variables"]
        return data, [(y[np.isfinite(y)], hidden == 1) for y, hidden in
                      zip(v[group]["data"], v["curve_hidden"]["data"])]

    def held(values):
        return values[np.isfinite(values)]
    for box in ("#energy-plot", "#force-plot"):
        export_clear_of_the_chart(page.locator(box))
    energy = chart_exported(page, page.locator("#energy-plot"), "NetCDF",
                            tmp_path)
    assert energy.name.endswith(".total-energy.nc"), energy.name
    data, [(curve, _)] = drawn(energy, "total_energy_ev")
    assert data["attrs"]["chart"] == "total-energy", data["attrs"]
    np.testing.assert_allclose(curve, held(steps.energy_ev), atol=1e-4)

    force = chart_exported(page, page.locator("#force-plot"), "NetCDF",
                           tmp_path)
    # SIESTA's Max line is its largest Cartesian component: the figure the
    # run door recomputes from the forces (`model/datafiles.md` § 9.7).
    data, curves = drawn(force, "stated_max_force_ev_per_ang")
    np.testing.assert_allclose(curves[0][0],
                               held(steps.max_force_ev_per_ang), atol=1e-4)
    root = data["groups"][""]["variables"]
    assert float(root["force_tolerance_ev_per_ang"]["data"]) == (
        pytest.approx(steps.relaxation["force_tolerance_ev_ang"]))

    residuals = chart_exported(page, page.locator("#scf-gnorm-plot"),
                               "NetCDF", tmp_path)
    data, _ = drawn(residuals, "scf_ddmax")
    root = data["groups"][""]["variables"]
    assert float(root["scf_ddmax_tolerance"]["data"]) == pytest.approx(
        deck.dm_tolerance)
    assert float(root["scf_dhmax_tolerance_ev"]["data"]) == pytest.approx(
        deck.h_tolerance_ev)
    assert drawn(residuals, "scf_dhmax_ev")[1]

    # THE STEP THE SCF CHARTS DRAW, named in the file: its rows the run's.
    scf = chart_exported(page, page.locator("#scf-energy-plot"), "NetCDF",
                         tmp_path)
    data, curves = drawn(scf, "scf_energy_ev")
    shown = [c for c, hidden in curves if not hidden]
    last = [k for k, rows in enumerate(steps.scf_rows) if rows][-1]
    assert "step_index" in data["attrs"]["chosen"], data["attrs"]["chosen"]
    np.testing.assert_allclose(np.concatenate(shown),
                               [r["energy"] for r in steps.scf_rows[last]])

    picture = chart_exported(page, page.locator("#energy-plot"), "PNG",
                             tmp_path)
    assert picture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", picture.name
    # THE CHART AS IT STANDS: the picture is the drawing's size on screen,
    # not its box's, whose top is the Export's strip (`web/results.md` § 1.2).
    from PIL import Image
    on_screen = page.locator("#energy-plot .svg-container").bounding_box()
    with Image.open(picture) as png:
        size = png.size
    # A canvas holds whole pixels: the drawing's fraction is dropped.
    assert all(0 <= drawn - got < 1 for got, drawn in
               zip(size, (on_screen["width"], on_screen["height"]))), (
        f"the picture is {size}, the chart on screen "
        f"{on_screen['width']} x {on_screen['height']}")
    vector = chart_exported(page, page.locator("#force-plot"), "SVG",
                            tmp_path)
    assert vector.name.endswith(".max-force.svg"), vector.name
    assert vector.read_text().lstrip().startswith("<svg"), vector.name
    tables = chart_exported(page, page.locator("#energy-plot"), "CSV",
                            tmp_path)
    names = zipfile.ZipFile(io.BytesIO(tables.read_bytes())).namelist()
    assert any(n.endswith("README.txt") for n in names), names
    assert any(n.endswith("total_energy_ev/curve_curve_point.csv")
               for n in names), names


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



def test_a_viewer_that_could_not_start_leaves_the_run_drawn(
        finished, real_h2_layered, page, flask_server, monkeypatch):
    """`web/trajectory.md` § 4, `web/results.md` § 3: the run's state, its
    SCF line and its plots need no 3-D viewer -- the Results tab in a window
    too narrow for MolView's card, the viewer refuses to mount and says so
    on its card, and the run is drawn beside it -- the tab bound in a wide
    window, the window narrowed and the page reloaded, as a person does.
    The silent failure (F12's review, J2-D2): the load threw on the viewer
    it did not have, the status line said "Cannot read properties of null",
    and nothing was drawn."""
    _open(page, flask_server, finished, real_h2_layered.tree, monkeypatch)
    page.set_viewport_size({"width": 340, "height": 900})
    page.reload()
    page.wait_for_function(
        "() => /Finished/.test((document.getElementById('run-state-label')"
        " || {}).textContent || '')"
        " && /SCF cycle/.test((document.getElementById('scf-status')"
        " || {}).textContent || '')", timeout=30000)
    assert page.locator(".molviewer-embed-error").count() == 1
    status = page.locator("#trajectory-status").inner_text()
    assert "Cannot read" not in status, status
    assert page.locator("#energy-plot").inner_html().strip(), "no energy plot"
