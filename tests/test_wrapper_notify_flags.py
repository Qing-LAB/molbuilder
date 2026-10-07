"""The reporting policy reaches the monitor, and nothing else does.

`task.json` says WHEN this calculation should speak up; the monitor is what
speaks.  Between them sits the wrapper, which bakes the policy in as flags
on the monitor's launch line.

**Why the policy rides ``Resources``.**  It is not a scheduler ask and
becomes no ``sbatch`` directive — like ``continue_retries``, which the class
docstring keeps there deliberately: *"this is the road every field the deck
never carries already rides… the alternative was a second, hand-maintained
road from a job to its wrapper."*  That road has lost a field to a
hand-copied argument list twice (``max_memory_mb``, then the ranks/cores
pair), which is the whole argument for not opening a third one.

**And what must never ride it: the destination.**  The URL and its
credential are the user's own file on the machine that runs the job.  A
wrapper is written to disk, copied into composed copies and read by anyone
who can see the run directory; a token in one would be a token published.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from molbuilder import runwrap
from molbuilder.jobset.model import Resources
from molbuilder.diagnostics import Capabilities
from molbuilder.diagnostics import set_capabilities
from molbuilder.runfiles import RunNames


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    """This machine's record (its activation: the refuse-to-emit contract)
    + synthetic caps."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT: without naming the directory the
    # write lands in a file nothing opens, and the test passes having
    # configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    # The record follows the config root -- and carries the activation, the
    # probe's copy the generator reads (`configuration.md` § 4).
    from conftest import write_machine_record
    write_machine_record()
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta", "molbuilder-siesta-gpu"}),
    ))
    yield


def _monitor_line(tmp_path: Path, deck: str = ".fdf", **kw) -> str:
    """The run script's monitor launch, for a stage's deck -- written with
    what travels beside it, the monitor's bundle (`write_run_wrapper`)."""
    names = RunNames.of("job", "01_coarse", "hierarchical")
    f = tmp_path / names.name(deck)
    f.write_text("SystemLabel job\nNumberOfAtoms 8\n" if deck == ".fdf"
                 else 'JOB = "job"\n')
    wrapper = runwrap.write_run_wrapper(
        f, names=names, emit_sbatch=False,
        resources=Resources(mpi_np=4, cpus_per_task=1, **kw))
    lines = [ln for ln in wrapper.read_text().splitlines()
             if runwrap.MONITOR_BUNDLE in ln and "--label" in ln]
    assert len(lines) == 1, f"expected one monitor launch, got {len(lines)}"
    return lines[0]


# --------------------------------------------------------------------- #
#  which channels -- names, and the two ways of saying none              #
# --------------------------------------------------------------------- #

def test_the_names_survive_a_job_set_file(tmp_path):
    """**A tuple out, a tuple back.**

    `Resources.to_dict` is `asdict`, so a job-set file stores the names as a
    JSON array and `from_dict` hands them back as a LIST -- which never
    equals the tuple it was written from.

    It breaks QUIETLY, which is why this test exists rather than a comment:
    the names still reach the wrapper either way, and only equality lies --
    so the symptom would surface somewhere far from the cause.
    """
    import json as _json
    from molbuilder.jobset.model import Resources

    for channels in (("slack", "lab"), (), None):
        r = Resources(mpi_np=4, notify_channels=channels)
        back = Resources.from_dict(_json.loads(_json.dumps(r.to_dict())))
        assert back == r, f"{channels!r} did not survive the file"
        assert back.notify_channels == channels


def test_a_list_of_names_is_accepted_and_normalised():
    """Four roads reach `Resources` (the CLI, `run-config.toml`, `prep`'s
    fold, and a job-set file somebody edited). A caller handing a list is
    not wrong; the class holds its own invariant, exactly as it does for
    `time` and `mem`."""
    from molbuilder.jobset.model import Resources
    assert Resources(notify_channels=["a", "b"]).notify_channels == ("a", "b")


def test_the_monitor_reads_back_what_the_wrapper_emitted(tmp_path):
    """Two files, one command line.  A wrapper that renders a value the
    monitor parses differently fails backgrounded and silent.  No `notify`
    block is nothing sent (`()`); a block naming no channels travels as the
    every-channel marker and is every channel there (`None`)."""
    from molbuilder import monitor as M
    from molbuilder.config_dir import ALL_CHANNELS
    for channels, expected in ((None, ()), ((ALL_CHANNELS,), None),
                               (("slack", "lab"), ("slack", "lab")),
                               ((), ())):
        line = _monitor_line(tmp_path, notify_on_scf=True,
                             notify_channels=channels)
        m = re.search(r'--notify-channels "([^"]*)"', line)
        got = M._channels_from_flag(None if m is None else m.group(1))
        assert got == expected, (channels, line)


# --------------------------------------------------------------------- #
#  the flags the monitor actually has                                    #
# --------------------------------------------------------------------- #

def test_every_flag_emitted_is_one_the_monitor_accepts(tmp_path):
    """The wrapper and the shipped script are two files that must agree
    about a command line.

    A flag renamed on one side and not the other produces a monitor that
    dies at argument parsing -- backgrounded, with its output redirected to
    /dev/null, so the job runs on and the only symptom is a `util.csv` that
    never appears.  Nothing else in the suite would notice.

    The authority is the SHIPPED monitor, asked by running it -- not the
    installed module.  Its bundle (`runwrap.MONITOR_BUNDLE`) is what sits in
    the run directory and what the wrapper invokes: with the job's own
    python, from the working directory, with no molbuilder on the path.
    Testing the installed module would pass in an environment the job
    never has.
    """
    import subprocess
    import sys

    line = _monitor_line(tmp_path, notify_on_scf=True, notify_every_hours=3)
    proc = subprocess.run([sys.executable, runwrap.MONITOR_BUNDLE, "--help"],
                          capture_output=True, text=True, timeout=60,
                          cwd=str(tmp_path))
    assert proc.returncode == 0, f"could not ask the monitor: {proc.stderr}"
    accepted = set(re.findall(r"--[a-z][a-z0-9-]+", proc.stdout))
    assert "--watch-pid" in accepted, (
        f"--help did not parse as expected; got {sorted(accepted)[:8]}")

    emitted = {tok for tok in line.split() if tok.startswith("--")}
    missing = emitted - accepted
    assert not missing, f"the wrapper emits flags the monitor rejects: {missing}"


# --------------------------------------------------------------------- #
#  what the job holds, and whether it can be watched                     #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("deck,use_gpu,told", [
    # (a SIESTA GPU deck needs the GPU env to render: `test_gpu_loadbalance`)
    (".fdf", True, True),
    (".fdf", False, False),
    (".py", True, True),              # PySCF's GPU is the same answer
    (".py", False, False),
])
def test_the_monitor_is_told_whether_the_run_uses_a_gpu(tmp_path, deck,
                                                        use_gpu, told):
    """A GPU is sampled and judged only for a run that uses one, and the
    wrapper says so from the answer it launches with -- `use_gpu`, either
    engine (`run-reports.md` § 2.1a).  The PySCF wrapper said no, always,
    so a PySCF run on a GPU was judged on its CPU alone (2026-09-26).

    MUTATION THIS MUST FAIL AGAINST: never pass ``--gpu`` for a ``.py``."""
    line = _monitor_line(tmp_path, deck, use_gpu=use_gpu,
                         gres="gpu:1" if use_gpu else None)
    assert ("--gpu" in line.split()) is told, line
