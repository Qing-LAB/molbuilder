"""A monitor that dies while starting says why, in the run's session log
(`execution/run-reports.md` § 2.3).

The monitor opens its `.monitor.log` only once it has loaded, and the wrapper
started it with its stderr at `/dev/null` until 2026-09-27 -- so a shipped
file that reached into molbuilder, or a package the job's env lacked, left
no trace at all.  `runwrap` records the last time it happened: every
production run's monitor dying at import, with nothing to say so.

Driven through ``jobset init --shape flat`` -> ``prep run`` -> ``launch run
--mode direct`` on an H2 PySCF run, with the shipped bundle replaced after
prep by one that fails at import, as a broken shipped file does.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from _road import conda_hook, env_available

CONDA_SH = conda_hook()

pytestmark = pytest.mark.skipif(
    not (CONDA_SH.is_file() and env_available("molbuilder-pySCF")),
    reason="needs the molbuilder-pySCF env + a detectable conda hook")

#: What the broken bundle says as it dies -- a string no healthy run writes.
_DYING_WORDS = "monitor-start-test: a shipped file needs a package this env lacks"


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def test_a_monitor_that_dies_starting_says_why_in_the_session_log(
        tmp_path, monkeypatch):
    """The run goes on; the monitor never opens its log; its traceback is in
    the session log beside the run's own lines.

    MUTATION THIS MUST FAIL AGAINST: the wrapper starting the monitor with
    its stderr at ``/dev/null`` (`runwrap._monitor_block`, before 2026-09-27).
    """
    from conftest import write_machine_record
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.runwrap import MONITOR_BUNDLE

    write_machine_record()
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", "P/optimization/H2", "--engine", "pyscf",
                "--shape", "flat", "--name", "H2")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": f"source {CONDA_SH}"}}))
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this")
    assert r.exit_code == 0, r.output

    shipped = bundle / MONITOR_BUNDLE
    assert shipped.is_file(), sorted(p.name for p in bundle.iterdir())
    with zipfile.ZipFile(shipped, "w") as z:
        z.writestr("__main__.py", f"raise ImportError({_DYING_WORDS!r})\n")

    _jobset("launch", "run", "coarse", "--bundle", bundle,
            "--mode", "direct", "--yes")

    logs = sorted(bundle.glob("*.runwrap-*.log"))
    assert logs, sorted(p.name for p in bundle.iterdir())
    said = "\n".join(p.read_text(errors="replace") for p in logs)
    assert "monitor: pid=" in said, (
        "the wrapper did not start the monitor, so this run shows nothing "
        "about how a start-up failure is kept")
    assert _DYING_WORDS in said, (
        "the monitor died at start and its reason is not in the session "
        "log:\n" + said[-3000:])
    assert not list(bundle.glob("*.monitor.log")), (
        "a monitor that failed at import wrote its own log -- the premise "
        "of this test does not hold")
