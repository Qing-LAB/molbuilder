"""A monitor says, in the run's session log, that it loaded -- or why it did
not (`execution/run-reports.md` § 2.3): ``monitor: starting ...`` before it
imports anything and ``monitor: mb_monitor.pyz started`` once it has, so a
``starting`` with no ``started`` is a monitor that died loading, and its
error follows.

The monitor opens its `.monitor.log` only once it has loaded, so the session
log is the one place a failed load can be seen at all.  Until 2026-09-27 the
bundle's entry caught the import error and exited without a word; before
that the wrapper sent the monitor's stderr to `/dev/null`.

Driven through ``jobset init --shape flat`` -> ``prep run`` -> ``launch run
--mode direct`` on an H2 PySCF run: once as prepped, and once with one of the
bundle's own member files replaced after prep by one that fails at import, as
a file the job's python cannot load does.
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

#: What the broken member says as it dies -- a string no healthy run writes.
_DYING_WORDS = "monitor-start-test: a shipped file this python cannot load"


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _launched(tmp_path, monkeypatch, *, broken_member=None):
    """An H2 PySCF run launched directly -- with ``broken_member`` of the
    shipped bundle made to fail at import, when named -- and its bundle and
    session log text."""
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

    if broken_member:
        shipped = bundle / MONITOR_BUNDLE
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
    logs = sorted(bundle.glob("*.runwrap-*.log"))
    assert logs, sorted(p.name for p in bundle.iterdir())
    said = "\n".join(p.read_text(errors="replace") for p in logs)
    assert "monitor: pid=" in said, (
        "the wrapper did not start the monitor, so this run shows nothing "
        "about how the monitor's start is kept")
    return bundle, said


def test_a_monitor_that_loads_says_so(tmp_path, monkeypatch):
    """``starting`` and then ``started``, and the monitor went on to open its
    own log.

    MUTATION THIS MUST FAIL AGAINST: the bundle's entry importing the
    monitor with nothing said (`runwrap._BUNDLE_MAIN` before 2026-09-27).
    """
    from molbuilder.runwrap import MONITOR_BUNDLE
    bundle, said = _launched(tmp_path, monkeypatch)
    starting = said.find(f"monitor: starting {MONITOR_BUNDLE} on python ")
    started = said.find(f"monitor: {MONITOR_BUNDLE} started")
    assert starting != -1 and started != -1, said[-3000:]
    assert starting < started, "started was said before starting"
    assert list(bundle.glob("*.monitor.log")), (
        "the monitor said it started and never opened its own log")


def test_a_monitor_that_dies_loading_says_why(tmp_path, monkeypatch):
    """``starting``, the error with its reason, and no ``started`` -- a
    member of the bundle failing at import, the real entry catching it.

    MUTATION THIS MUST FAIL AGAINST: the entry catching the import error and
    exiting without a word (`runwrap._BUNDLE_MAIN` before 2026-09-27).
    """
    from molbuilder.runwrap import MONITOR_BUNDLE
    bundle, said = _launched(tmp_path, monkeypatch,
                             broken_member="report_fields.py")
    assert f"monitor: starting {MONITOR_BUNDLE} on python " in said, (
        said[-3000:])
    assert f"monitor: {MONITOR_BUNDLE} did not load" in said, said[-3000:]
    assert _DYING_WORDS in said, (
        "the monitor died loading and its reason is not in the session "
        "log:\n" + said[-3000:])
    assert f"monitor: {MONITOR_BUNDLE} started" not in said, (
        "a monitor that failed to load said it started")
    assert not list(bundle.glob("*.monitor.log")), (
        "a monitor that failed at import wrote its own log -- the premise "
        "of this test does not hold")
