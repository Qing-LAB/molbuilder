"""A machine record's scheduler of each kind (`configuration.md` § 5 M-2):
its kind, and each of its commands by the name `scheduler/kinds.toml` gives
it -- run from PATH, where the suite's refusing `sbatch` stands first
(`conftest.the_suite_cannot_submit_a_job`); a test that launches to a queue
names its own submit command (`support.road.a_machine_with_queues`).  A probe
records the full path it finds; a record naming the bare name is the
declared form, which a test uses so it never spells where a command is.  One
per kind, so a test names a SLURM machine, or one with no scheduler.
"""
from __future__ import annotations

from molbuilder.scheduler import Scheduler
from molbuilder.scheduler.kinds import kind


def scheduler_of(name: str) -> Scheduler:
    """The scheduler of kind ``name``, each command by its name on PATH."""
    return Scheduler(kind=name, **dict(kind(name).commands))


SLURM = scheduler_of("slurm")
WORKSTATION = scheduler_of("workstation")
