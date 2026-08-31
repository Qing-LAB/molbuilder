"""The reporting policy reaches the monitor, and nothing else does.

`task.json` says WHEN this calculation should speak up; the monitor is what
speaks.  Between them sits the wrapper, which bakes the policy in as flags
on the `mb_monitor.py` line.

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

import json
import re
from pathlib import Path

import pytest

from molbuilder import runwrap
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.jobset.model import Resources


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    """Activation config (refuse-to-emit contract) + synthetic caps."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "source activate"}
    }))
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta"}),
    ))
    yield


def _monitor_line(tmp_path: Path, **kw) -> str:
    f = tmp_path / "job.fdf"
    f.write_text("SystemLabel job\nNumberOfAtoms 8\n")
    text = runwrap.render_run_wrapper(f, resources=Resources(mpi_np=4, **kw))
    lines = [ln for ln in text.splitlines()
             if "mb_monitor.py" in ln and "--out" in ln]
    assert len(lines) == 1, f"expected one monitor launch, got {len(lines)}"
    return lines[0]


def _notify_flags(line: str):
    return re.findall(r"--notify-[a-z-]+(?:\s+[\d.]+)?", line)


# --------------------------------------------------------------------- #
#  absent stays absent                                                   #
# --------------------------------------------------------------------- #

def test_a_description_that_asked_for_nothing_emits_no_flags(tmp_path):
    """The wrapper for a calculation with no `notify` block must look
    exactly as it did before this feature existed -- otherwise every
    prepped bundle changes, and "off" acquires a spelling."""
    assert _notify_flags(_monitor_line(tmp_path)) == []


@pytest.mark.parametrize("kw", [
    {"notify_on_scf": False},
    {"notify_every_hours": 0},
    {"notify_on_scf": False, "notify_every_hours": 0},
])
def test_explicitly_off_is_the_same_as_absent(tmp_path, kw):
    """A policy that reports on nothing renders no flags, so there is one
    spelling of off on the wrapper as there is one in the description."""
    assert _notify_flags(_monitor_line(tmp_path, **kw)) == []


# --------------------------------------------------------------------- #
#  what is asked for is what is emitted                                  #
# --------------------------------------------------------------------- #

def test_the_scf_trigger_reaches_the_monitor(tmp_path):
    line = _monitor_line(tmp_path, notify_on_scf=True)
    assert "--notify-on-scf" in line
    assert "--notify-every-hours" not in line


def test_the_period_reaches_the_monitor_in_hours(tmp_path):
    """HOURS on both sides.  A unit converted in transit is how "4h"
    reached `sbatch` as `-t 4h` and was refused (task.Allocation) -- the
    lesson being that a number crossing a boundary must not change
    meaning."""
    line = _monitor_line(tmp_path, notify_every_hours=6)
    assert "--notify-every-hours 6" in line


def test_a_fractional_period_survives(tmp_path):
    """Half-hourly is a reasonable ask and must not silently become 0 or
    30.  `%g` renders it without inventing trailing zeros."""
    line = _monitor_line(tmp_path, notify_every_hours=2.5)
    assert "--notify-every-hours 2.5" in line


def test_both_triggers_are_emitted_together(tmp_path):
    """They combine with OR -- checkboxes, not a choice -- so the wrapper
    must be able to carry both at once."""
    line = _monitor_line(tmp_path, notify_on_scf=True, notify_every_hours=1)
    assert "--notify-on-scf" in line
    assert "--notify-every-hours 1" in line


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

    The authority is the SHIPPED script, asked by running it -- not the
    installed module.  `mb_monitor.py` is what actually sits in the run
    directory and what the wrapper actually invokes: with the job's own
    python, from the working directory, with no molbuilder on the path.
    Testing the installed module would pass in an environment the job
    never has.
    """
    import subprocess
    import sys

    shipped = tmp_path / "mb_monitor.py"
    shipped.write_text(runwrap._monitor_source(), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(shipped), "--help"],
                          capture_output=True, text=True, timeout=60,
                          cwd=str(tmp_path))
    assert proc.returncode == 0, f"could not ask the monitor: {proc.stderr}"
    accepted = set(re.findall(r"--[a-z][a-z0-9-]+", proc.stdout))
    assert "--watch-pid" in accepted, (
        f"--help did not parse as expected; got {sorted(accepted)[:8]}")

    line = _monitor_line(tmp_path, notify_on_scf=True, notify_every_hours=3)
    emitted = {tok for tok in line.split() if tok.startswith("--")}
    missing = emitted - accepted
    assert not missing, f"the wrapper emits flags the monitor rejects: {missing}"


# --------------------------------------------------------------------- #
#  the destination must not be here                                      #
# --------------------------------------------------------------------- #

def test_no_destination_or_credential_is_written_into_the_wrapper(tmp_path):
    """A wrapper is a file on disk in the run directory: copied into
    composed copies, readable by anyone with the filesystem.  The URL and
    its token belong to the machine, in the user's own 0600 file, and must
    never be baked in here.
    """
    f = tmp_path / "job.fdf"
    f.write_text("SystemLabel job\nNumberOfAtoms 8\n")
    text = runwrap.render_run_wrapper(
        f, resources=Resources(mpi_np=4, notify_on_scf=True,
                               notify_every_hours=6))
    lowered = text.lower()
    for leak in ("hooks.slack.com", "discord.com/api/webhooks",
                 "authorization:", "bearer ", "--notify-url", "notify_token"):
        assert leak not in lowered, f"the wrapper carries {leak!r}"
