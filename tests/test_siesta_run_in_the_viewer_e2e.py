"""What the trajectory viewer states about a finished SIESTA run, each fact
from its one source (`model/parse.md` § 2a P-T2 and P-T4, `web/results.md`
§ 3a): the SCF line's seconds per iteration are the SCF-timing log's, read
by the reader the run record reads them by; the badge's "ended" time is the
output's own ``>> End of run``.

``jobset init`` -> ``prep run coarse`` -> ``launch run --mode direct`` on an
H2 relaxation, then opened in the Results tab as a person opens it, in its
folder.  Until 2026-09-27 the viewer estimated its own rate three ways
(SIESTA's first-iteration timer, the browser's poll times, the output's file
times), and dated an ended run by the output's file time, so a copy made
three days later read "ended" at the copy's time.  *(An output copied alone
into a folder of its own was opened here too until 2026-10-03; the badge
reads the RUN now, and an output whose run's records are not beside it is a
run that never concluded -- `web/results.md` § 4.1.)*
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, env_bin

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONDA_SH = conda_hook()

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (CONDA_SH.is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, list(args))


@pytest.fixture(scope="module")
def finished(isolated_projects_root_module, tmp_path_factory):
    """The attempt directory of a finished H2 relaxation, made on the road."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    tree = isolated_projects_root_module
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    shutil.copy(FIXTURES / "psml" / "H.psml",
                tree / "pseudopotential" / "H.psml")
    StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]]),
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3),
        tree / "P" / "structure" / "h2.xyz")

    mp = pytest.MonkeyPatch()
    try:
        # The suite's config rule, which its autouse fixture applies per test
        # and so not to a module's fixture (`conftest`).
        mp.delenv("MOLBUILDER_CONFIG_DIR", raising=False)
        mp.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
        # ...the box probed, its record saying how a shell enters conda here
        # -- the activation the generator reads (`configuration.md` § 4).
        from conftest import write_machine_record
        write_machine_record(env_init={
            "activation": "conda activate", "preamble": f"source {CONDA_SH}"})
        mp.chdir(tree.parent)
        bin_ = env_bin("molbuilder-siesta")
        assert (bin_ / "siesta").is_file(), bin_
        mp.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")

        r = _jobset("init", "--structure", "P/structure/h2.xyz",
                    "--bundle", "P/opt/R", "--engine", "siesta",
                    "--shape", "hierarchical", "--calculation", "optimization",
                    "--name", "H2", "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        bundle = tree / "P" / "opt" / "R"
        task = json.loads((bundle / "task.json").read_text())
        task["execution"] = {**task.get("execution", {}), "mpi_np": 1,
                             "omp_threads": 1}
        (bundle / "task.json").write_text(json.dumps(task, indent=2))
        r = _jobset("prep", "run", "coarse", "--bundle", str(bundle),
                    "--target", "this")
        assert r.exit_code == 0, r.output
        r = _jobset("launch", "run", "coarse", "--bundle", str(bundle),
                    "--mode", "direct", "--yes")
        assert r.exit_code == 0, r.output
        attempt = bundle / "01_coarse" / "run-0"
        out = attempt / "H2_01_coarse-run0.out"
        assert ">> End of run" in out.read_text(errors="replace"), (
            "the run did not reach its end")
        yield attempt
    finally:
        mp.undo()


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
    from molbuilder.parse import parse_dir
    time = parse_dir(attempt).record["computation"]["time"]
    return time, datetime.fromisoformat(time["run_end_local"])


def test_in_its_folder_the_viewer_states_the_runs_rate_and_end(
        finished, page, flask_server, monkeypatch):
    """The SCF line's rate is the number the run record states -- the
    SCF-timing log, through one reader -- and the badge dates the run by its
    output's end.

    MUTATION THIS MUST FAIL AGAINST: the viewer computing its own rate (the
    three estimates before 2026-09-27).
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
