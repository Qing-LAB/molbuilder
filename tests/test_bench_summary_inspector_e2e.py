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

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from molbuilder import describe as D                       # noqa: E402
from molbuilder import diagnostics                         # noqa: E402
from molbuilder.config.siesta import SiestaConfig          # noqa: E402
from molbuilder.jobset.model import Resources              # noqa: E402
from molbuilder.jobset.prep import prep_bench              # noqa: E402
from molbuilder.scheduler import Environment, Topology     # noqa: E402
from molbuilder.siesta.stages import default_siesta_stages  # noqa: E402
from molbuilder.structure import Structure                 # noqa: E402


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
                            default_siesta_stages("publishable"),
                            engine="siesta",
                            calculation="optimization",
                            shape="hierarchical", name="JOB",
                            source=str(tree / "h2.xyz")),
        calc)
    from conftest import write_pseudos
    write_pseudos(calc, ["H"])
    (calc / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100"),
                    env_init={"activation": "conda activate",
                                       "preamble": "true"}).to_json() + "\n")

    prep_bench(calc, "coarse",
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


def test_the_page_says_when_it_last_looked(page, server, sweep):
    """B4's other half.  "The page is live, AND SAYS WHEN IT LAST LOOKED" --
    a sweep is watched precisely while it runs, so a silently stale verdict
    is worse than none."""
    jpath, _winner, _n = sweep
    _mount(page, server, jpath)
    page.wait_for_selector(".bench-summary", timeout=10000)
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
    _mount(page, server, jpath)
    page.wait_for_selector(".bench-summary", timeout=10000)
    assert page.evaluate(
        "() => window.molbuilder.inspectors.benchSummaryInspector.pollMs"
    ) == 15000


def test_it_really_re_polls(page, server, sweep):
    """B4's first half: the page IS live.

    Collapsing the long delay rather than waiting 15 s for it -- the
    presenter schedules through ``window.setTimeout``, so shortening only
    the long schedules proves the loop re-arms without the test sitting
    through a real cadence.
    """
    jpath, _winner, _n = sweep
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/results", wait_until="networkidle")
    page.wait_for_function("() => window.molbuilder"
                           " && window.molbuilder.inspectors"
                           "        .benchSummaryInspector")
    page.evaluate("""(p) => {
        window.__fetches = 0;
        const realFetch = window.fetch;
        window.fetch = (...a) => {
            if (String(a[0]).includes('/api/bench/summary')) window.__fetches++;
            return realFetch(...a);
        };
        const realTimeout = window.setTimeout;
        window.setTimeout = (fn, ms, ...r) =>
            realTimeout(fn, ms >= 10000 ? 40 : ms, ...r);
        const insp = window.molbuilder.inspectors.pick(p);
        const host = document.getElementById('inspector-host');
        host.innerHTML = '';
        window.__handle = insp.mount(host, p, { showError: () => {} });
        return 0;
    }""", str(jpath))
    page.wait_for_function("() => window.__fetches >= 3", timeout=10000)
    page.evaluate("() => window.__handle.dispose()")
    before = page.evaluate("() => window.__fetches")
    page.wait_for_timeout(400)
    assert page.evaluate("() => window.__fetches") == before, (
        "it kept polling after dispose")
    assert not errors, errors
