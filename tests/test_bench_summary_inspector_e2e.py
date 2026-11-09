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

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from molbuilder.scheduler import Topology                  # noqa: E402
from support.road import declare_bench, described, jobset, one_rank, road_tree


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """A prepared GPU sweep, nothing run: its folder, the `coarse` stage's
    benchmark container, whose ``job-set.json`` the Results tab opens.

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
    calc = described(road_tree(tmp_path, monkeypatch), name="JOB")
    one_rank(calc, use_gpu=True, diag_algorithm="ELPA-1STAGE", gpu_count=1)
    declare_bench(calc, {"mpi_np": [1, 2, 4], "omp_threads": [1, 2],
                         "gpu_count": [1]})
    r = jobset("prep", "bench", "coarse", "--bundle", calc, "--target", "this",
               "--np", "8", "--cpus-per-task", "8", "--no-sbatch")
    assert r.exit_code == 0, r.output
    return calc / "01_coarse" / "bench"


@pytest.fixture
def server(isolated_projects_root):
    """A server for this test, on the test's own projects tree -- the one
    the sweep is described in (`support.live_server.serve`)."""
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, base, folder, *, before=""):
    """The Results tab bound to the sweep's ``folder`` as a person binds it
    (`support.sidebar.results_on`) -- a benchmark's folder opens its
    ``job-set.json`` (`runs._container_product`), which the picker announces
    with the server's answer for it and the tab's viewer mounts
    (`web/presenters.md` § 2) -- until the sweep's page is on screen.
    ``before`` runs ahead of every page script.  Returns the page's errors,
    collected as they come."""
    from support.sidebar import results_on
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    if before:
        page.add_init_script(before)
    results_on(page, base, folder)
    page.wait_for_selector(".bench-summary", timeout=15000)
    return errors


def _put_away(page):
    """The tab bound to a folder with nothing to show -- the projects root
    (`support.sidebar.reload_from`) -- so the picker announces nothing and
    the viewer disposes of the page on screen (`results/viewer.js`)."""
    from molbuilder.projects import projects_root
    from support.sidebar import reload_from
    reload_from(page, projects_root())


def test_the_page_says_when_it_last_looked(page, server, sweep):
    """B4's other half.  "The page is live, AND SAYS WHEN IT LAST LOOKED" --
    a sweep is watched precisely while it runs, so a silently stale verdict
    is worse than none."""
    _open(page, server, sweep)
    foot = page.locator(".bench-foot").inner_text().strip()
    # The words AND the clock reading they name -- not the server's
    # `measured` time the foot also carries.
    import re as _re
    assert _re.search(r"last looked \d{1,2}:\d{2}", foot), foot


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

    AND A LOOK THAT FAILS IS NOT THE END (B4, 2.69.9): the second look finds
    no server -- a restart, as the browser meets it -- and the sweep stays on
    screen, the failure said in its foot in the reader's words, and the next
    look comes on time.  Through 2.69.8 the page replaced the sweep with the
    error and stopped looking.

    MUTATIONS THIS MUST FAIL AGAINST: `POLL_MS` off 15 000 (the re-polls
    armed at another delay); a failed look that stops the loop, or that is
    not said.
    """
    errors = _open(page, server, sweep, before="""
        window.__starts = [];
        window.__armedBy = [];
        let firing = null;
        const realFetch = window.fetch;
        window.fetch = (...a) => {
            if (String(a[0]).includes('/api/bench/summary')) {
                window.__starts.push(performance.now());
                window.__armedBy.push(firing);
                // THE SECOND LOOK FINDS NO SERVER.
                if (window.__starts.length === 2)
                    return Promise.reject(new TypeError('Failed to fetch'));
            }
            return realFetch(...a);
        };
        window.__failedSaid = [];
        new MutationObserver(() => {
            const f = document.querySelector('.bench-failed');
            if (f) window.__failedSaid.push(f.textContent);
        }).observe(document, {subtree: true, childList: true,
                              characterData: true});
        const realTimeout = window.setTimeout;
        window.setTimeout = (fn, ms, ...r) => realTimeout(
            typeof fn !== 'function' ? fn : (...x) => {
                firing = ms;
                try { return fn(...x); } finally { firing = null; }
            },
            ms === 15000 ? 40 : ms, ...r);""")
    page.wait_for_function("() => window.__starts.length >= 3", timeout=10000)
    said = page.evaluate("() => window.__failedSaid")
    assert any("failed" in t and "could not reach the server" in t
               for t in said), (
        f"the failed look was not said in the sweep's foot: {said}")
    assert page.locator(".bench-summary").count() == 1, (
        "the sweep left the screen after a look that failed")
    armed = page.evaluate("() => window.__armedBy")
    assert 15000 in armed, (
        "no re-poll was armed on the trajectory viewer's 15 s cadence "
        f"(bench-summary.md B4); the summary fetches were armed by {armed}")
    starts = page.evaluate("() => window.__starts")
    cycle_ms = max(b - a for a, b in zip(starts, starts[1:]))
    _put_away(page)
    page.wait_for_selector(".bench-summary", state="detached", timeout=15000)
    before = page.evaluate("() => window.__starts.length")
    page.wait_for_timeout(max(2000, 5 * cycle_ms))
    assert page.evaluate("() => window.__starts.length") == before, (
        "it kept polling after it was put away")
    assert page.locator(".bench-summary").count() == 0
    assert not errors, errors
