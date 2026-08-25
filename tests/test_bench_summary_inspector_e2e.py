"""The bench-sweep inspector, end to end — a real sweep, in a real browser.

Contract: ``docs/web/bench-summary.md``.

Everything else about this feature is tested a layer down: the composition
in ``test_prep_bench_fold.py``, the route in ``test_api_bench_summary.py``,
the dispatch in ``test_inspector_registry_dispatch_js.py``.  What only a
browser can show is that the three meet — that picking a sweep's
``job-set.json`` actually draws the sweep, with the verdict on it and a
card for every trial including the ones with nothing to say (B3).

So this file builds a REAL sweep with the real ``prep`` machinery, finishes
one trial, and reads the page.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from molbuilder import describe as D                       # noqa: E402
from molbuilder import diagnostics                         # noqa: E402
from molbuilder.config.siesta import SiestaConfig          # noqa: E402
from molbuilder.jobset._cli import _bench_inputs           # noqa: E402
from molbuilder.jobset.model import Resources              # noqa: E402
from molbuilder.jobset.prep import prep_calculation        # noqa: E402
from molbuilder.scheduler import Environment, Topology     # noqa: E402
from molbuilder.siesta.stages import default_siesta_stages  # noqa: E402
from molbuilder.structure import Structure                 # noqa: E402
from molbuilder.task import Stage                          # noqa: E402


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, tmp_path_factory, monkeypatch):
    """cwd + HOME + projects-root isolation, the pattern the prep tests
    use: without it the run reads the developer's own config cascade."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    box = tmp_path_factory.mktemp("sandbox")
    monkeypatch.chdir(box)
    monkeypatch.setenv("HOME", str(box / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (box / "home").mkdir()
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """A prepped GPU sweep with ONE finished trial — returns
    ``(job_set_path, winner_label, n_trials)``."""
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    calc = tmp_path / "calc"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB", use_gpu=True,
                                         diag_algorithm="ELPA-1STAGE"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        calc)
    from conftest import write_pseudos
    write_pseudos(calc, ["H"])
    (calc / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    (calc / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100")).to_json() + "\n")

    sweep_spec, pins, translation = _bench_inputs(calc, None)
    prep_calculation(calc, "coarse",
                     allocation=Resources(mpi_np=8, cpus_per_task=8),
                     sweep=sweep_spec, pins=pins, translation=translation,
                     emit_sbatch=False)
    bench = calc / "01_coarse" / "bench"
    js = json.loads((bench / "job-set.json").read_text())
    winner = js["jobs"][0]["name"]

    # One trial finishes, with a timing its steady state can be read from.
    d = bench / f"bench-{winner}"
    (d / f"JOB-{winner}_01_coarse-run0.out").write_text("x\n>> End of run:\n")
    (d / f"JOB-{winner}_01_coarse-run0.scf-timing.log").write_text(
        "100.0 scf 1\n104.0 scf 2\n108.0 scf 3\n112.0 scf 4\n")

    # The server may read this tree, and nothing else.
    caps = diagnostics.Capabilities(runtime_config={}, conda_binary=None,
                                    conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)
    return bench / "job-set.json", winner, len(js["jobs"])


@pytest.fixture
def server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    srv = make_server("127.0.0.1", 0, create_app(config={}), threaded=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        t.join(timeout=5)


def _mount(page, base, path):
    """Open /results and mount whichever inspector claims ``path`` — the
    same route the controller takes when you pick a file."""
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{base}/results", wait_until="networkidle")
    # Wait for THIS inspector to have registered, not merely for the
    # registry to exist: every inspector is a separate deferred script, so
    # `pick` is callable a tick before bench-summary.js has run and would
    # hand the file to whoever is registered by then.
    page.wait_for_function("() => window.molbuilder"
                           " && window.molbuilder.inspectors"
                           " && window.molbuilder.inspectors.pick"
                           " && window.molbuilder.inspectors"
                           "        .benchSummaryInspector")
    name = page.evaluate(
        """(p) => {
            const insp = window.molbuilder.inspectors.pick(p);
            if (!insp) return null;
            const host = document.getElementById('inspector-host');
            host.innerHTML = '';
            window.__handle = insp.mount(host, p, { showError: (m) => {
                host.textContent = 'ERROR: ' + m; } });
            return insp.name;
        }""", str(path))
    return name, errors


def test_picking_a_sweep_draws_it(page, server, sweep):
    jpath, winner, n_trials = sweep
    name, errors = _mount(page, server, jpath)
    assert name == "bench-summary", "the sweep's plan was claimed by someone else"
    page.wait_for_selector(".bench-summary", timeout=10000)

    # B3: EVERY trial gets a card, including the ones that have not run.
    cards = page.locator(".bench-trial")
    assert cards.count() == n_trials

    # the verdict is on the page, and it names the trial that won
    verdict = page.locator(".bench-verdict")
    assert winner in verdict.inner_text()
    assert "is-decided" in (verdict.get_attribute("class") or "")

    # the winning card is marked, and carries its measurement
    won = page.locator(f".bench-trial.is-winner")
    assert won.count() == 1
    assert winner in won.inner_text()
    assert "4.00 s/iter" in won.inner_text()

    assert not errors, errors


def test_a_trial_with_no_timing_says_so_rather_than_vanishing(
        page, server, sweep):
    """B3 again, from the other side: silence would answer 'where did my
    third trial go?' with nothing at all."""
    jpath, winner, n_trials = sweep
    _mount(page, server, jpath)
    page.wait_for_selector(".bench-summary", timeout=10000)
    notes = page.locator(".bench-note")
    # every trial except the finished one has nothing to show, and says it
    assert notes.count() == n_trials - 1
    # WHAT it says is the run's own detail when the run has one -- here
    # "no result file yet" from runstatus -- and only falls back to the
    # generic sentence when it does not.  Preferring the specific word is
    # the point, so this asserts that the card speaks, not that it recites
    # a particular string.
    said = notes.first.inner_text().strip()
    assert said, "an untimed trial rendered an EMPTY note, which is silence"
    assert page.locator(".bench-trial").count() == n_trials


def test_dispose_stops_the_polling_and_clears_the_host(page, server, sweep):
    """B4 polls every 15 s; a viewer that keeps polling after it is put
    away is the leak the registry's dispose contract exists to prevent."""
    jpath, _winner, _n = sweep
    _mount(page, server, jpath)
    page.wait_for_selector(".bench-summary", timeout=10000)
    page.evaluate("() => window.__handle.dispose()")
    assert page.locator(".bench-summary").count() == 0
    assert page.evaluate(
        "() => document.getElementById('inspector-host').innerHTML") == ""


def _finish(bench, label, dt):
    """Give one trial a finished .out and a timing whose steady state is
    ``dt`` s/iter."""
    d = bench / f"bench-{label}"
    (d / f"JOB-{label}_01_coarse-run0.out").write_text("x\n>> End of run:\n")
    (d / f"JOB-{label}_01_coarse-run0.scf-timing.log").write_text(
        "".join(f"{100.0 + i * dt} scf {i + 1}\n" for i in range(4)))


def test_the_chart_plots_an_axis_THESE_trials_actually_differ_in(
        page, server, sweep):
    """The bug this pins was visible only on screen.

    ``varied`` says what the SWEEP varied — the right answer for the sweep
    and the wrong one for the chart.  Early on only a few trials have
    finished and they may share a value of the first varied coordinate;
    taking ``varied[0]`` regardless drew every finished trial at the same
    x — a vertical line that looks like data.  Here the finished trials
    differ in K and share C, so the chart must choose K.
    """
    jpath, _winner, _n = sweep
    bench = jpath.parent
    js = json.loads(jpath.read_text())
    names = [j["name"] for j in js["jobs"]]
    # G1K1C1 / G1K2C1 / G1K4C1 -- K varies, C is 1 for all three
    same_c = [n for n in names if n.endswith("C1")][:3]
    assert len(same_c) == 3, names
    for label, dt in zip(same_c, (4.0, 2.5, 1.8)):
        _finish(bench, label, dt)

    _mount(page, server, jpath)
    page.wait_for_selector(".bench-summary", timeout=10000)
    # Ask Plotly what it drew, rather than matching its internal DOM
    # class names -- it attaches `layout` and `data` to the div it owns.
    page.wait_for_function(
        "() => { const c = document.querySelector('.bench-chart');"
        "        return c && c.layout && c.layout.xaxis; }", timeout=10000)
    axis = page.evaluate(
        "() => document.querySelector('.bench-chart').layout.xaxis.title.text")
    n_pts = page.evaluate(
        "() => document.querySelector('.bench-chart').data[0].x.length")
    assert n_pts == 3, f"expected the three finished trials, drew {n_pts}"
    assert axis == "K", (
        f"the chart chose {axis!r}, which is constant across every trial it "
        f"is drawing -- that renders as a vertical line")
