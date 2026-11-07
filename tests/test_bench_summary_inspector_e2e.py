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

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from molbuilder import diagnostics                         # noqa: E402
from molbuilder.jobset.model import JobSet                 # noqa: E402
from molbuilder.scheduler import Topology                  # noqa: E402
from support.road import (describe_calculation, jobset,    # noqa: E402
                          one_rank, read_request, write_request)


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """A prepared GPU sweep, nothing run — returns
    ``(job_set_path, first_trial_label, n_trials)``.

    THE GPU IS THE PERSON'S, ON THE RUN CARD (`execution/gpu.md` § 1.1:
    *"`use_gpu` on the run card"*; G2: nothing derives it from the
    machine): the calculation's card states it, and *"whether the trials
    are a GPU family is … the bench's points for it, else the calculation's
    card"* (`generator.md` § 4.3a) -- the template holds no run setting
    (`task-setup.md` § 6.1).  THE MACHINE IS THIS ONE'S RECORD, one a100,
    where `jobset probe --write` puts it (`conftest.write_machine_record`);
    prep copies it into the calculation (`configuration.md` M-3)."""
    from conftest import write_machine_record
    write_machine_record(
        topology=Topology(sockets=1, cores_per_socket=4, gpus_per_node=1,
                          gpu_type="a100"))
    calc = describe_calculation(tmp_path, monkeypatch, name="JOB")
    one_rank(calc, use_gpu=True, diag_algorithm="ELPA-1STAGE", gpu_count=1)
    req = read_request(calc)
    req["bench"] = {"mpi_np": [1, 2, 4], "omp_threads": [1, 2],
                    "gpu_count": [1]}
    write_request(calc, req)
    r = jobset("prep", "bench", "coarse", "--bundle", calc, "--target", "this",
               "--np", "8", "--cpus-per-task", "8", "--no-sbatch")
    assert r.exit_code == 0, r.output
    bench = calc / "01_coarse" / "bench"
    js = JobSet.of(bench).to_dict()
    winner = js["jobs"][0]["name"]

    # The server may read this tree, and nothing else.
    caps = diagnostics.Capabilities(runtime_config={}, conda_binary=None,
                                    conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: (((tmp_path / "projects").resolve(),
                                       "projects"),))
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


def test_it_really_re_polls_and_stops_when_put_away(page, server, sweep):
    """B4's first half: the page IS live, *"polled on the trajectory
    viewer's cadence (15 s)"* -- and put away, it stops and leaves nothing
    (the registry's dispose contract: a viewer that keeps polling after it
    is put away is the leak it exists to prevent).

    The declared cadence collapsed rather than waited through -- the
    presenter schedules through ``window.setTimeout``, so shortening the
    15 s schedules proves the loop re-arms without the test sitting
    through a real cadence.  THE CADENCE ITSELF is what armed the re-polls:
    each summary fetch made from inside a timer records that timer's delay,
    so a viewer that drifted off 15 s is seen re-polling at another.  A
    cycle is that short wait AND the server's answer, measured here at
    150-400 ms, so "stopped" is watched for five of the longest cycles seen
    while it ran, never less than 2 s: a watch shorter than one cycle sees
    a running loop as stopped.

    MUTATION THIS MUST FAIL AGAINST: `POLL_MS` off 15 000 (the re-polls
    armed at another delay).
    """
    jpath, _winner, _n = sweep
    errors = _open(page, server, jpath.parent, before="""
        window.__starts = [];
        window.__armedBy = [];
        let firing = null;
        const realFetch = window.fetch;
        window.fetch = (...a) => {
            if (String(a[0]).includes('/api/bench/summary')) {
                window.__starts.push(performance.now());
                window.__armedBy.push(firing);
            }
            return realFetch(...a);
        };
        const realTimeout = window.setTimeout;
        window.setTimeout = (fn, ms, ...r) => realTimeout(
            typeof fn !== 'function' ? fn : (...x) => {
                firing = ms;
                try { return fn(...x); } finally { firing = null; }
            },
            ms === 15000 ? 40 : ms, ...r);""")
    page.wait_for_function("() => window.__starts.length >= 3", timeout=10000)
    armed = page.evaluate("() => window.__armedBy")
    assert 15000 in armed, (
        "no re-poll was armed on the trajectory viewer's 15 s cadence "
        f"(bench-summary.md B4); the summary fetches were armed by {armed}")
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
