"""The header and the command line are two spellings of ONE placement.

The gate for phase 5 of `execution/scheduler.md` § 8, and the assertion
neither emitter could have made alone.

Both ASU Sol failures of 2026-08-23 are the same split seen from opposite
sides.  ``runwrap.render_sbatch`` wrote the ``#SBATCH`` header and
``jobset.submit._sbatch_resource_flags`` wrote the ``sbatch`` flags, and each
decided for itself what queue and what wall to name:

  * the header named ``htc/debug`` (15 minutes) while the command line asked
    for 38 minutes, and the scheduler refused the combination with
    ``QOSMaxWallDurationPerJobLimit``;
  * the header named a queue and stated **no** wall at all, so ``sbatch`` by
    hand inherited a partition default that the named QOS forbids -- a trap
    armed only for a human, because ``launch`` passes ``-t`` on the command
    line where flags win.

Neither writer could catch either one, because catching it means comparing
the two, and nothing held both.  `scheduler.emit.Directives` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from molbuilder.scheduler.emit import Directives
from molbuilder.scheduler.place import Placement


@dataclass
class _Res:
    """A stand-in for `jobset.model.Resources`.

    Duck-typed on purpose: `emit` sits below the layer that defines
    Resources and reads the attributes it needs without naming the type, so
    the test hands it the same shape rather than importing across the layer
    it is checking.
    """
    time: Optional[str] = None
    mpi_np: Optional[int] = None
    cpus_per_task: Optional[int] = None
    gres: Optional[str] = None
    mem: Optional[str] = None
    exclusive: bool = False


def _directives(**kw):
    base = dict(partition="htc", qos="public", walltime="0-00:38:00",
                ntasks=64, cpus_per_task=1, gres="gpu:a100:2", mem="390G")
    base.update(kw)
    return Directives(**base)


def _header_value(lines, flag):
    """The value of a ``#SBATCH <flag> <value>`` line, or None."""
    for line in lines:
        parts = line.replace("#SBATCH ", "").split()
        if parts and parts[0] == flag and len(parts) > 1:
            return parts[1]
        if parts and parts[0].startswith(flag + "="):
            return parts[0].split("=", 1)[1]
    return None


def _flag_value(flags, flag):
    for i, f in enumerate(flags):
        if f == flag and i + 1 < len(flags):
            return flags[i + 1]
        if f.startswith(flag + "="):
            return f.split("=", 1)[1]
    return None


class TestTheTwoSpellingsCannotDisagree:

    @pytest.mark.parametrize("flag", ["-p", "-q", "-t", "-n", "-c"])
    def test_the_same_value_reaches_both(self, flag):
        d = _directives()
        assert _header_value(d.header_lines(), flag) == \
               _flag_value(d.sbatch_flags(), flag), \
               f"{flag} differs between the header and the command line"

    def test_the_queue_is_one_answer(self):
        d = _directives()
        assert d.queue() == ("htc", "public")
        assert _header_value(d.header_lines(), "-p") == d.partition
        assert _flag_value(d.sbatch_flags(), "-p") == d.partition

    def test_naming_a_queue_without_a_wall_is_impossible_to_express_twice(self):
        """The second Sol failure: a header that names a queue and states no
        wall.  It can still happen -- a caller may hold no wall -- but it
        cannot now happen on ONE side only."""
        d = _directives(walltime=None)
        assert _header_value(d.header_lines(), "-t") is None
        assert _flag_value(d.sbatch_flags(), "-t") is None

    def test_a_placement_binds_both(self):
        p = Placement(domain=None, partition="general", qos="public")
        d = Directives.of(p, _Res(time="1-00:00:00", mpi_np=8))
        assert d.queue() == ("general", "public")
        assert "#SBATCH -p general" in d.header_lines()
        assert ["-p", "general"] == d.sbatch_flags()[:2]

    def test_no_menu_states_no_queue_on_either_side(self):
        """A machine with no domains promised nothing (R6), so neither
        spelling invents a queue."""
        d = Directives.of(None, _Res(time="0-00:10:00", mpi_np=4))
        assert _header_value(d.header_lines(), "-p") is None
        assert _flag_value(d.sbatch_flags(), "-p") is None


class TestTheMemoryRuleIsWrittenOnce:
    """``--exclusive`` and ``--mem`` are mutually exclusive: whole-node
    ownership already grants all the node's memory, and some sites reject the
    pair.  Two copies of a mutual exclusion is how one of them comes to allow
    it."""

    def test_exclusive_suppresses_mem_in_both(self):
        d = _directives(exclusive=True, mem="390G")
        assert not any("--mem=390G" in l for l in d.header_lines())
        assert not any("--mem=390G" in f for f in d.sbatch_flags())

    def test_the_header_says_all_of_it_out_loud(self):
        """The one place the spellings differ, and they differ in VERBOSITY,
        not meaning: a person reads the .sbatch file."""
        d = _directives(exclusive=True)
        assert "#SBATCH --exclusive" in d.header_lines()
        assert "#SBATCH --mem=0" in d.header_lines()
        assert d.sbatch_flags().count("--exclusive") == 1
        assert not any("--mem" in f for f in d.sbatch_flags())

    def test_without_exclusive_the_memory_is_stated_on_both(self):
        d = _directives(exclusive=False, mem="120G")
        assert "#SBATCH --mem=120G" in d.header_lines()
        assert "--mem=120G" in d.sbatch_flags()


class TestBothObjectsTravelWhole:
    """A8 — a door takes the object, so a caller has no subset to pass.

    `Directives.of` took six loose keyword arguments until 2026-08-23, and the
    architecture rule caught it: at two `field=obj.field` arguments the caller
    is re-assembling what the callee should have been handed, and the seventh
    field is the one that gets forgotten.
    """

    def test_the_resources_object_is_handed_over_intact(self):
        r = _Res(time="0-01:00:00", mpi_np=32, cpus_per_task=2,
                 gres="gpu:a100:1", mem="64G")
        d = Directives.of(Placement(domain=None, partition="p", qos="q"), r)
        assert d.walltime == r.time
        assert d.ntasks == r.mpi_np
        assert d.cpus_per_task == r.cpus_per_task
        assert d.gres == r.gres
        assert d.mem == r.mem

    def test_no_resources_at_all_is_a_queue_and_nothing_else(self):
        d = Directives.of(Placement(domain=None, partition="p", qos="q"))
        assert d.header_lines() == ["#SBATCH -p p", "#SBATCH -q q"]
        assert d.sbatch_flags() == ["-p", "p", "-q", "q"]
