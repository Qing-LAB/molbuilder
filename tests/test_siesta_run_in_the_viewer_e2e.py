"""What the trajectory viewer states about a finished SIESTA run, each fact
from its one source (`model/parse.md` § 2a P-T2 and P-T4, `web/results.md`
§ 3a): the SCF line's seconds per iteration are the SCF-timing log's, read
by the reader the run record reads them by; the badge's "ended" time is the
output's own ``>> End of run``.

The run is the layered H2 relaxation's first `coarse` run, made once in this
pass on the road (`support/real_runs.py`, run A of plan § 5y), opened in the
Results tab as a person opens it, in its folder.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

import pytest

from _road import conda_hook, env_available

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


@pytest.fixture(scope="module")
def finished(real_h2_layered):
    """The attempt directory of a finished H2 relaxation, made on the road
    (`support/real_runs.RealRun.attempt`) -- finished, by the one status
    door."""
    from molbuilder.runs import run_answer
    attempt = real_h2_layered.attempt("coarse")
    said = run_answer(attempt)
    assert (said["state"], said["detail"]) == ("finished", "ended, exit 0"), said
    return attempt


@pytest.fixture(scope="module")
def tree(real_h2_layered):
    """The projects tree the run was made in."""
    return real_h2_layered.tree


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, flask_server, folder, tree, monkeypatch):
    """The Results tab bound to ``folder``, its output shown, in a fresh
    page -- the folder set the way a returning tab finds it, the run's
    projects ``tree`` the picker's root."""
    from molbuilder import diagnostics
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((tree.resolve(), "road"),))
    page.add_init_script(
        "try { sessionStorage.setItem('molbuilder.current_dir', "
        f"{json.dumps(str(folder))}); }} catch (_) {{}}")
    page.goto(f"{flask_server}/results")
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
        finished, tree, page, flask_server, monkeypatch):
    """The SCF line's rate is the number the run record states -- the
    SCF-timing log, through one reader -- and the badge dates the run by its
    output's end.

    MUTATION THIS MUST FAIL AGAINST: the viewer computing its own rate.
    """
    time, ended = _the_runs_own_times(finished)
    _open(page, flask_server, finished, tree, monkeypatch)
    scf = page.locator("#scf-status").inner_text()
    rate = re.search(r"([0-9.]+) s/iter", scf)
    assert rate, scf
    assert float(rate.group(1)) == pytest.approx(time["s_per_iter"],
                                                 rel=5e-3), (scf, time)
    detail = page.locator("#run-state-detail").inner_text()
    assert "ended" in detail and ended.strftime(":%M:%S") in detail, detail


def test_the_scf_plot_draws_both_tests_siesta_requires(
        finished, tree, page, flask_server, monkeypatch):
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
    _open(page, flask_server, finished, tree, monkeypatch)
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
        finished, tree, page, flask_server, monkeypatch):
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
    _open(page, flask_server, finished, tree, monkeypatch)
    page.locator("#force-scale").fill("2.5")
    page.locator("#force-min").fill("0.05")
    page.locator("#hide-frozen").check()
    asked = page.evaluate("() => window.__asked")
    assert asked, "the viewer asked its panel for nothing"
    wrong = sorted({f"{sel} answered by {n}" for sel, n in asked if n != 1})
    assert not wrong, wrong


def test_opening_the_run_again_and_again_leaves_what_once_left(
        finished, tree, page, flask_server, monkeypatch):
    """The trajectory viewer, opened on the run and put away five times,
    leaves on the page what once left -- the listeners on the window, the
    document or an element still on the page, and the live GL contexts
    (`web/molview.md` § 8, § 9.9a; `web/presenters.md` § 3, step 2): the
    inspector teardown rows' trajectory one, on output the engine wrote.

    THE SILENT FAILURE: a listener a mount adds and its dispose leaves,
    firing against a dead panel on every later mount; a drawing window
    left alive with its listeners for the page's life.

    Mounted through the registry on the Results tab, each mount waited out
    by the viewer's word that its load ended, which must carry no error;
    paired add to remove as the browser pairs them (`support/leftovers.SPY`).

    MUTATION THIS MUST FAIL AGAINST: a listener the trajectory viewer adds
    outside the viewers' shared scope."""
    from molbuilder import diagnostics
    from molbuilder.runs import folder_answer, openable
    from support.leftovers import SPY, grew
    output, _ = openable(finished)
    # THE SERVER'S ANSWER FOR THE FILE, as the picker passes it to the
    # viewer (`/api/results/dir`'s row, `file-picker.js` `_metaFor`).
    answer = folder_answer(os.path.dirname(output))
    row = next(f for f in answer["files"]
               if f["name"] == os.path.basename(output))
    meta = {**{k: row[k] for k in ("role", "label", "stage", "parser")},
            "engine": answer["engine"]}
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((tree.resolve(), "road"),))
    page.goto(f"{flask_server}/results", wait_until="networkidle")
    page.wait_for_function(
        "(c) => { const r = (window.molbuilder || {}).inspectors;"
        "  return !!r && (r.pick(c.file, c.meta) || {}).name === 'trajectory'; }",
        arg={"file": str(output), "meta": meta}, timeout=10000)
    left = page.evaluate("""async (c) => {
        const reg = window.molbuilder.inspectors;
        const spy = """ + SPY + """;
        const cycle = async () => {
            const host = document.createElement("div");
            host.style.width = "900px";
            document.body.appendChild(host);
            const ended = new Promise((resolve) => spy.quietly(document,
                window.molbuilder.constants.EVENT_INSPECTOR_READY,
                (e) => resolve((e && e.detail) || {}), { once: true }));
            const handle = reg.mount(host, c.file,
                                     reg.createDefaultContext(host), c.meta);
            const said = await Promise.race([ended,
                new Promise((r) => setTimeout(() => r(null), 15000))]);
            handle.dispose();
            document.body.removeChild(host);
            if (!said) return "a mount never ended";
            if (said.error) return "it could not load: " + said.error;
            return null;
        };
        try {
            const first = await cycle();
            if (first) return { error: first };
            const once = spy.state();
            for (let i = 1; i < c.times; i++) {
                const again = await cycle();
                if (again) return { error: again };
            }
            return { once, again: spy.state() };
        } finally {
            spy.restore();
        }
    }""", {"file": str(output), "meta": meta, "times": 5})
    assert "error" not in left, left
    assert not grew(left), (
        "opening the run five times left more on the page than opening it "
        "once:\n  " + "\n  ".join(grew(left)))
    assert left["again"]["contexts"] == left["once"]["contexts"], left
