"""A monitor says, in the run's session log, that it loaded -- or why it did
not (`execution/run-reports.md` § 2.6): ``monitor: starting ...`` before it
imports the monitor and ``monitor: mb_monitor.pyz started`` once it has, in
the session log's own line; a monitor that died loading says ``starting``, one
``ERROR`` line naming the exception, its traceback, and no ``started``.

The monitor opens its `.monitor.log` only once it has loaded, so the session
log is the one place a failed load can be seen at all.  Until 2026-09-27 the
bundle's entry caught the import error and exited without a word; before
that the wrapper sent the monitor's stderr to `/dev/null`.

And the wrapper's ending door, which reads the same bundle: it asks the
bundle ONCE whether it loads, so a bundle that cannot load prints its error
once and the wrapper says the ending cannot be read (`job-contracts.md`
§ 2.6) -- where every question printed the error again until 2026-09-28.

Driven through ``jobset init`` -> ``prep run`` -> ``launch run --mode
direct``: an H2 PySCF run once as prepped and once with one of the bundle's
own member files replaced after prep by one that fails at import, as a file
the job's python cannot load does; and an H2 SIESTA relaxation broken the same
way, because only the SIESTA wrapper asks how its run ended.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, env_bin

CONDA_SH = conda_hook()
FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.engine

_needs = {env: pytest.mark.skipif(
    not (CONDA_SH.is_file() and env_available(env)),
    reason=f"needs the {env} env + a detectable conda hook")
    for env in ("molbuilder-pySCF", "molbuilder-siesta")}

#: What the broken member says as it dies -- a string no healthy run writes.
_DYING_WORDS = "monitor-start-test: a shipped file this python cannot load"


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _launched(tmp_path, monkeypatch, *, engine="pyscf", broken_member=None):
    """An H2 run launched directly -- a flat PySCF optimisation, or a
    hierarchical SIESTA relaxation -- with ``broken_member`` of the shipped
    bundle made to fail at import, when named; the folder the run ran in and
    its session log text."""
    from conftest import write_machine_record
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.runwrap import MONITOR_BUNDLE
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    write_machine_record()
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]])),
        tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    siesta = engine == "siesta"
    if siesta:
        (tree / "pseudopotential").mkdir()
        shutil.copy(FIXTURES / "psml" / "H.psml",
                    tree / "pseudopotential" / "H.psml")
        monkeypatch.setenv("PATH", f"{env_bin('molbuilder-siesta')}"
                                   f"{os.pathsep}{os.environ['PATH']}")
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", "P/optimization/H2", "--engine", engine,
                "--shape", "hierarchical" if siesta else "flat",
                "--name", "H2",
                *(("--psml-lib", "pseudopotential") if siesta else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    # How a shell enters conda HERE is this machine's record's to say.
    from conftest import write_machine_record
    write_machine_record(env_init={
        "activation": "conda activate", "preamble": f"source {CONDA_SH}"})
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this", *(("--np", 1) if siesta else ()),
                "--cpus-per-task", 1)
    assert r.exit_code == 0, r.output
    ran_in = bundle / "01_coarse" / "run-0" if siesta else bundle

    if broken_member:
        shipped = ran_in / MONITOR_BUNDLE
        with zipfile.ZipFile(shipped) as z:
            members = {n: z.read(n) for n in z.namelist()}
        assert broken_member in members, sorted(members)
        members[broken_member] = (
            f"raise ImportError({_DYING_WORDS!r})\n".encode())
        with zipfile.ZipFile(shipped, "w") as z:
            for name, text in members.items():
                z.writestr(name, text)

    _jobset("launch", "run", "coarse", "--bundle", bundle,
            "--mode", "direct", "--yes")
    logs = sorted(ran_in.glob("*.runwrap-*.log"))
    assert logs, sorted(p.name for p in ran_in.iterdir())
    said = "\n".join(p.read_text(errors="replace") for p in logs)
    assert "monitor: pid=" in said, (
        "the wrapper did not start the monitor, so this run shows nothing "
        "about how the monitor's start is kept")
    return ran_in, said


@_needs["molbuilder-pySCF"]
def test_a_monitor_that_loads_says_so(tmp_path, monkeypatch):
    """``starting`` and then ``started``, and the monitor went on to open its
    own log.

    MUTATION THIS MUST FAIL AGAINST: the bundle's entry importing the
    monitor with nothing said (`runwrap._BUNDLE_MAIN` before 2026-09-27).
    """
    from molbuilder.runwrap import MONITOR_BUNDLE
    bundle, said = _launched(tmp_path, monkeypatch)
    starting = said.find(f"[INFO ] monitor: starting {MONITOR_BUNDLE} on python ")
    started = said.find(f"[INFO ] monitor: {MONITOR_BUNDLE} started")
    assert starting != -1 and started != -1, said[-3000:]
    assert starting < started, "started was said before starting"
    assert list(bundle.glob("*.monitor.log")), (
        "the monitor said it started and never opened its own log")


@_needs["molbuilder-pySCF"]
def test_a_monitor_that_dies_loading_says_why(tmp_path, monkeypatch):
    """``starting``, the ``ERROR`` line naming the exception, the traceback
    naming the member that failed, and no ``started`` -- a member of the
    bundle failing at import, the real entry catching it.

    MUTATIONS THIS MUST FAIL AGAINST: the entry catching the import error and
    exiting without a word (`runwrap._BUNDLE_MAIN` before 2026-09-27); the
    entry saying ``started`` before the import; the ``ERROR`` line or the
    traceback left out.
    """
    from molbuilder.runwrap import MONITOR_BUNDLE
    bundle, said = _launched(tmp_path, monkeypatch,
                             broken_member="report_fields.py")
    assert f"[INFO ] monitor: starting {MONITOR_BUNDLE} on python " in said, (
        said[-3000:])
    error = said.find(f"[ERROR] monitor: {MONITOR_BUNDLE} did not load -- "
                      f"ImportError: {_DYING_WORDS}")
    assert error != -1, (
        "the monitor died loading and its reason is not in the session "
        "log:\n" + said[-3000:])
    assert said.find(broken := "report_fields.py", error) != -1, (
        f"no traceback naming {broken} follows the error:\n"
        + said[error:error + 3000])
    assert f"monitor: {MONITOR_BUNDLE} started" not in said, (
        "a monitor that failed to load said it started")
    assert not list(bundle.glob("*.monitor.log")), (
        "a monitor that failed at import wrote its own log -- the premise "
        "of this test does not hold")


@_needs["molbuilder-siesta"]
def test_an_ending_the_bundle_cannot_read_is_said_once(tmp_path, monkeypatch):
    """A SIESTA relaxation, the bundle broken after prep: the wrapper asks
    the bundle once whether it loads, before its first question about how the
    run ended, so the error appears twice in all -- the monitor's own start,
    and that one ask -- and the wrapper says once that the ending cannot be
    read.  (A relaxation asks twice on its way out, with warm retries on.)

    MUTATION THIS MUST FAIL AGAINST: the wrapper asking the bundle at every
    question (`_mb_ending_able` checking only that the files are there,
    before 2026-09-28), which printed the error three times and never said
    the ending could not be read.
    """
    from molbuilder.runwrap import MONITOR_BUNDLE, _ENDING_UNREADABLE
    _ran_in, said = _launched(tmp_path, monkeypatch, engine="siesta",
                              broken_member="report_fields.py")
    assert "benchmark:" in said, "the engine never ran to its end"
    errors = said.count(f"[ERROR] monitor: {MONITOR_BUNDLE} did not load")
    assert errors == 2, (errors, said[-4000:])
    assert said.count(_ENDING_UNREADABLE) == 1, said[-4000:]
