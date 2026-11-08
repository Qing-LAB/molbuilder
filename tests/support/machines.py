"""A machine record's scheduler of each kind (`configuration.md` § 5 M-2):
its kind, and each of its commands by the name `scheduler/kinds.toml` gives
it -- run from PATH, where the suite's refusing `sbatch` stands first
(`conftest.the_suite_cannot_submit_a_job`); a test that launches to a queue
names its own submit command (`support.road.a_machine_with_queues`).  A probe
records the full path it finds; a record naming the bare name is the
declared form, which a test uses so it never spells where a command is.  One
per kind, so a test names a SLURM machine, or one with no scheduler.

And the GPU tool the load monitor asks of this machine, `nvidia-smi`, first
on PATH (:func:`a_gpu_tool`) or nowhere on it, as on a CPU-only box
(:func:`no_gpu_tool`), its sampler then asked anew
(:func:`gpu_reading_asked_anew`) -- the sampler itself is never stood in
for.
"""
from __future__ import annotations

from molbuilder.scheduler import Scheduler
from molbuilder.scheduler.kinds import kind


def scheduler_of(name: str) -> Scheduler:
    """The scheduler of kind ``name``, each command by its name on PATH."""
    return Scheduler(kind=name, **dict(kind(name).commands))


SLURM = scheduler_of("slurm")
WORKSTATION = scheduler_of("workstation")


#: WHAT `nvidia-smi` DOES ON A HOST WHOSE DRIVER AND LIBRARY HAVE DIVERGED,
#: measured 2026-09-18 on the development host: the reason on STDOUT, an
#: empty stderr, exit 18.
DRIVER_MISMATCH = ("echo 'Failed to initialize NVML: "
                   "Driver/library version mismatch'\nexit 18")


def gpu_reading_asked_anew(monkeypatch) -> None:
    """The load monitor's GPU sampler (`web/blueprints/system_load.
    _gpu_snapshot`) with no reading kept: its next one asks the tool.  Its
    cache would otherwise answer, for its TTL, with what an earlier test's
    tool said."""
    from molbuilder.web.blueprints import system_load
    monkeypatch.setattr(system_load, "_smi_cache",
                        {"t": 0.0, "gpus": [], "error": None})


def a_gpu_tool(tmp_path, monkeypatch, body: str):
    """``nvidia-smi`` first on this test's PATH, doing ``body`` (a shell
    script's lines -- :data:`DRIVER_MISMATCH`, or what a test's case needs),
    and the sampler asked anew (:func:`gpu_reading_asked_anew`): the load
    monitor's real sampler runs it, as it runs the host's.  Returns the
    tool's path."""
    import os
    tool = tmp_path / "nvidia-smi"
    tool.write_text(f"#!/bin/sh\n{body}\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    gpu_reading_asked_anew(monkeypatch)
    return tool


def no_gpu_tool(monkeypatch) -> None:
    """No ``nvidia-smi`` on this test's PATH -- every folder of it that holds
    one taken off, as a CPU-only box has none -- and the sampler asked anew
    (:func:`gpu_reading_asked_anew`): the load monitor's real sampler looks
    for the tool and finds none, as it does there."""
    import os
    monkeypatch.setenv("PATH", os.pathsep.join(
        d for d in os.environ["PATH"].split(os.pathsep)
        if d and not os.path.exists(os.path.join(d, "nvidia-smi"))))
    gpu_reading_asked_anew(monkeypatch)
