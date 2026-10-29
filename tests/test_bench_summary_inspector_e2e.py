"""The bench-sweep inspector, end to end — a real sweep, in a real browser.

Contract: ``docs/web/bench-summary.md``.

Everything else about this feature is tested a layer down: the composition
in ``test_prep_bench_fold.py``, the route in ``test_api_bench_summary.py``,
the dispatch in ``tests/data/presenters.toml``'s rows.  What only a
browser can show is that the three meet: the page that mounts on a
sweep's ``job-set.json`` polls it, says when it last looked, keeps the
trajectory viewer's cadence and stops when it is disposed of.

So this file builds a REAL sweep with the real ``prep`` machinery and reads
the page.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from molbuilder.calculation import open as open_calculation

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from molbuilder import describe as D                       # noqa: E402
from molbuilder import diagnostics                         # noqa: E402
from molbuilder.config.siesta import SiestaConfig          # noqa: E402
from molbuilder.jobset.model import Resources              # noqa: E402
from molbuilder.jobset.prep import prep_bench              # noqa: E402
from molbuilder.scheduler import Environment, Topology     # noqa: E402
from molbuilder.jobset.ladder import engine_ladder  # noqa: E402
from molbuilder.structure import Structure                 # noqa: E402
from support.machines import WORKSTATION


@pytest.fixture(autouse=True)
def _sandbox(isolated_projects_root, tmp_path_factory, monkeypatch):
    """cwd + HOME + projects-root isolation, the pattern the prep tests
    use: without it the run reads the developer's own config cascade."""
    box = tmp_path_factory.mktemp("sandbox")
    monkeypatch.chdir(box)
    monkeypatch.setenv("HOME", str(box / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (box / "home").mkdir()


@pytest.fixture
def sweep(isolated_projects_root, monkeypatch):
    """A prepared GPU sweep, nothing run — returns
    ``(job_set_path, first_trial_label, n_trials)``."""
    tree = isolated_projects_root
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tree / "h2.xyz").write_text(struct.to_xyz())
    calc = tree / "calc"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB", use_gpu=True,
                                         diag_algorithm="ELPA-1STAGE"),
                            engine_ladder("siesta", "publishable"),
                            engine="siesta",
                            calculation="optimization",
                            shape="hierarchical", name="JOB",
                            source=str(tree / "h2.xyz")),
        calc)
    from conftest import write_pseudos
    write_pseudos(calc, ["H"])
    (calc / "environment.json").write_text(
        Environment(machine="this", scheduler=WORKSTATION,
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100"),
                    env_init={"activation": "conda activate",
                                       "preamble": "true"}).to_json() + "\n")

    prep_bench(open_calculation(calc), "coarse",
               allocation=Resources(mpi_np=8, cpus_per_task=8),
               emit_sbatch=False)
    bench = calc / "01_coarse" / "bench"
    js = json.loads((bench / "job-set.json").read_text())
    winner = js["jobs"][0]["name"]

    # The server may read this tree, and nothing else.
    caps = diagnostics.Capabilities(runtime_config={}, conda_binary=None,
                                    conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tree.resolve(), "projects"),))
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


def _open(page, base, folder, *, before=""):
    """The Results tab bound to the sweep's ``folder``, the way a returning
    tab finds it -- a benchmark's folder opens its ``job-set.json``
    (`runs._container_product`), which the picker announces with the
    server's answer for it and the tab's viewer mounts (`web/presenters.md`
    § 2) -- until the sweep's page is on screen.  ``before`` runs ahead of
    every page script.  Returns the page's errors, collected as they come."""
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.add_init_script(
        "try { sessionStorage.setItem('molbuilder.current_dir', "
        f"{json.dumps(str(folder))}); }} catch (_) {{}}" + before)
    page.goto(f"{base}/results")
    page.wait_for_selector(".bench-summary", timeout=15000)
    return errors


def _put_away(page):
    """What the tab does when the picker announces nothing to show: the
    viewer disposes of the page on screen (`results/viewer.js`)."""
    page.evaluate("""() => document.dispatchEvent(new CustomEvent(
        window.molbuilder.constants.EVENT_FILE_SELECTED,
        { detail: { file: "" } }))""")


def test_the_page_says_when_it_last_looked(page, server, sweep):
    """B4's other half.  "The page is live, AND SAYS WHEN IT LAST LOOKED" --
    a sweep is watched precisely while it runs, so a silently stale verdict
    is worse than none."""
    jpath, _winner, _n = sweep
    _open(page, server, jpath.parent)
    foot = page.locator(".bench-foot").inner_text().strip()
    assert "last looked" in foot, foot
    # and it carries a clock reading, not just the words
    import re as _re
    assert _re.search(r"\d{1,2}:\d{2}", foot), foot


def test_the_declared_cadence_is_the_trajectory_viewers(page, server, sweep):
    """B4 says "polled on the trajectory viewer's cadence (15 s)".  Nothing
    else on this page can see that number, so a viewer that drifted off it
    would pass every other test."""
    jpath, _winner, _n = sweep
    _open(page, server, jpath.parent)
    assert page.evaluate(
        "() => window.molbuilder.inspectors.benchSummaryInspector.pollMs"
    ) == 15000


def test_it_really_re_polls_and_stops_when_put_away(page, server, sweep):
    """B4's first half: the page IS live -- and put away, it stops and
    leaves nothing (the registry's dispose contract: a viewer that keeps
    polling after it is put away is the leak it exists to prevent).

    The declared cadence collapsed rather than waited through -- the
    presenter schedules through ``window.setTimeout``, so shortening the
    15 s schedules proves the loop re-arms without the test sitting
    through a real cadence.  A cycle is that short wait AND the server's
    answer, measured here at 150-400 ms, so "stopped" is watched for five
    of the longest cycles seen while it ran, never less than 2 s: a watch
    shorter than one cycle sees a running loop as stopped.
    """
    jpath, _winner, _n = sweep
    errors = _open(page, server, jpath.parent, before="""
        window.__starts = [];
        const realFetch = window.fetch;
        window.fetch = (...a) => {
            if (String(a[0]).includes('/api/bench/summary'))
                window.__starts.push(performance.now());
            return realFetch(...a);
        };
        const realTimeout = window.setTimeout;
        window.setTimeout = (fn, ms, ...r) =>
            realTimeout(fn, ms === 15000 ? 40 : ms, ...r);""")
    page.wait_for_function("() => window.__starts.length >= 3", timeout=10000)
    starts = page.evaluate("() => window.__starts")
    cycle_ms = max(b - a for a, b in zip(starts, starts[1:]))
    _put_away(page)
    assert page.locator(".bench-summary").count() == 0
    before = page.evaluate("() => window.__starts.length")
    page.wait_for_timeout(max(2000, 5 * cycle_ms))
    assert page.evaluate("() => window.__starts.length") == before, (
        "it kept polling after it was put away")
    assert page.locator(".bench-summary").count() == 0
    assert not errors, errors
