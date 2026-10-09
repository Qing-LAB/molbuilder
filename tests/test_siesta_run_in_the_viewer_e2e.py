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
import re
from datetime import datetime
from pathlib import Path

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
    """The attempt directory of a finished H2 relaxation, made on the road."""
    attempt = real_h2_layered.bundle / "01_coarse" / "run-0"
    assert ">> End of run" in (attempt / "H2_01_coarse-run0.out").read_text(
        errors="replace"), "the run did not reach its end"
    return attempt


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, flask_server, folder, monkeypatch):
    """The Results tab bound to ``folder``, its output shown, in a fresh
    page -- the folder set the way a returning tab finds it."""
    from molbuilder import diagnostics
    root = next(p for p in Path(folder).parents if p.name == "projects")
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((root.resolve(), "road"),))
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
        finished, page, flask_server, monkeypatch):
    """The SCF line's rate is the number the run record states -- the
    SCF-timing log, through one reader -- and the badge dates the run by its
    output's end.

    MUTATION THIS MUST FAIL AGAINST: the viewer computing its own rate.
    """
    time, ended = _the_runs_own_times(finished)
    _open(page, flask_server, finished, monkeypatch)
    scf = page.locator("#scf-status").inner_text()
    rate = re.search(r"([0-9.]+) s/iter", scf)
    assert rate, scf
    assert float(rate.group(1)) == pytest.approx(time["s_per_iter"],
                                                 rel=5e-3), (scf, time)
    detail = page.locator("#run-state-detail").inner_text()
    assert "ended" in detail and ended.strftime(":%M:%S") in detail, detail
