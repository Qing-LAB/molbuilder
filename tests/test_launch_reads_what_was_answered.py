"""Launch must not re-ask a question its own artifacts already answered.

Two failures on Sol, 2026-08-24, from one bundle whose `task.json` said
everything:

    "allocation": {"domain": "htc", "time": "4h", "mem": "256G"}

`prep` baked all three into every job's `resources`.  Then:

* `launch` printed the queue table and refused for want of a `--domain`,
  though `resources.domain` said `htc` -- R9 says what was admitted when
  the work was built is re-admitted when it is sent, and it cannot be
  re-admitted unread.
* with `--domain htc` supplied by hand, it refused again: *"prep baked
  time='4h', which does not parse as a SLURM walltime"* -- the tool
  rejecting a value the tool had written.  `Allocation.time` was
  documented as what a person types and `Resources.time` as what SLURM
  takes, and nothing translated between them.

The rule: **the record holds ONE spelling, SLURM's, and
translation happens at the edges where humans are** (`task.py::Allocation`,
`scheduler.quantities.canonical_time`/`canonical_mem`).
"""
from __future__ import annotations

import json
import re

import pytest

from molbuilder.jobset.model import Job, Resources


SLURM_TIME = re.compile(r"^(?:\d+-)?\d+(?::\d{2}){0,2}$")
SLURM_MEM = re.compile(r"^\d+(?:\.\d+)?[KMGT]?$")


class TestTheRecordHoldsOneSpelling:

    def test_a_human_time_never_survives_into_the_record(self):
        """`Resources` enforces its own invariant, so no road into it --
        CLI flag, run-config.toml, prep's fold, a hand-edited job-set --
        can leave a human spelling in a field SLURM has to read."""
        r = Resources(time="4h", mem="80GB")
        assert r.time == "0-04:00:00"
        assert r.mem == "80G"

    def test_it_survives_a_json_round_trip(self):
        r = Resources.from_dict(json.loads(json.dumps(
            Resources(time="4h", mem="80GB").to_dict())))
        assert r.time == "0-04:00:00" and r.mem == "80G"


    def test_task_json_allocation_is_normalised_on_read(self):
        from molbuilder.task import _allocation_from_obj
        a = _allocation_from_obj(
            {"allocation": {"domain": "htc", "time": "4h", "mem": "256G"}})
        assert (a.domain, a.time, a.mem) == ("htc", "0-04:00:00", "256G")

    def test_an_unreadable_allocation_names_its_field(self):
        from molbuilder.task import _allocation_from_obj
        with pytest.raises(Exception) as e:
            _allocation_from_obj({"allocation": {"time": "banana"}})
        assert "allocation.time" in str(e.value)

    def test_what_reaches_sbatch_is_what_sbatch_takes(self):
        """The end of the chain, and the assertion the whole rule exists
        for: `-t 4h` was emitted and SLURM refused it."""
        from molbuilder.scheduler.emit import Directives
        flags = Directives.of(None, Resources(
            time="4h", mem="80GB", mpi_np=48, cpus_per_task=1)).sbatch_flags()
        for i, tok in enumerate(flags):
            if tok == "-t":
                assert SLURM_TIME.match(flags[i + 1]), flags
            if tok.startswith("--mem="):
                assert SLURM_MEM.match(tok.split("=", 1)[1]), flags
        assert "-t" in flags and any(f.startswith("--mem=") for f in flags)


class TestAStatedValueIsNeverLostQuietly:
    """The family both Sol failures belong to: a person states something and
    it does not arrive.  These are the other doors it could happen at."""

    def test_an_unknown_resources_key_is_refused_not_dropped(self):
        """`"memory"` and `"walltime"` are both plausible and neither is a
        field name.  Filtering them silently gave the job the scheduler's
        default and said nothing."""
        with pytest.raises(ValueError) as e:
            Resources.from_dict({"memory": "256G", "mpi_np": 48})
        assert "memory" in str(e.value) and "known keys" in str(e.value)

    def test_the_refusal_names_which_job(self):
        """A sweep holds six; a refusal you have to bisect a file to act on
        is most of the cost of the bug still being there."""
        with pytest.raises(ValueError) as e:
            Job.from_dict({"name": "G0K48C1", "script": "x.fdf",
                           "resources": {"walltime": "4h"}})
        assert "G0K48C1" in str(e.value)

    def test_a_sliver_of_memory_never_becomes_all_of_it(self):
        """SLURM reads `--mem=0` as ALL the node's memory, so rounding a
        small positive ask down to zero flips the smallest request into the
        largest -- silently, and in the direction that gets a job refused or
        a node monopolised."""
        from molbuilder.scheduler.quantities import slurm_mem
        assert slurm_mem(0.0001) == "1M"
        assert slurm_mem(0) == "0"          # an explicit 0 still means all

    def test_the_queue_table_does_not_under_report_a_ceiling(self):
        """It read `1h` for a 90-minute queue -- integer division, no
        remainder -- in the one table a person reads to decide what to ask
        for, so they would ask for less than it would have given them."""
        from molbuilder.scheduler.quantities import human_wall
        assert human_wall(5400) == "1h30m"
        assert human_wall(271800) == "75h30m"
        # and unchanged for every round value the listing already showed
        assert [human_wall(s) for s in (900, 14400, 86400, 604800)] == [
            "15m", "4h", "24h", "168h"]
