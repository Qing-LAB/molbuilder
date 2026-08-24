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

The rule that replaced it: **the record holds ONE spelling, SLURM's, and
translation happens at the edges where humans are** (`task.py::Allocation`,
`ask.canonical_time`/`canonical_mem`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from molbuilder.jobset.model import Job, JobSet, Resources


SLURM_TIME = re.compile(r"^(?:\d+-)?\d+(?::\d{2}){0,2}$")
SLURM_MEM = re.compile(r"^\d+(?:\.\d+)?[KMGT]?$")


def _runner():
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner(), jobset_group


def _sweep(domain=None, domains=None):
    """A two-point sweep.  ``domains`` gives the two jobs different ones."""
    a, b = (domains or (domain, domain))
    return JobSet(
        name="sw", engine="siesta", kind="sweep",
        jobs=[Job(name="G1K1C4", script="job.fdf",
                  resources=Resources(mpi_np=1, cpus_per_task=4, domain=a)),
              Job(name="G1K2C4", script="job.fdf",
                  resources=Resources(mpi_np=2, cpus_per_task=4, domain=b))])


def _write_domains(where, rows):
    from molbuilder.scheduler import (FILENAME, Domain, Environment,
                                      Topology, write_environment)
    return write_environment(
        Environment(scheduler="slurm",
                    topology=Topology(cores_per_socket=64),
                    domains=[Domain(name=n, partition=p, qos=q, max_time=t)
                             for n, p, q, t in rows]),
        Path(where) / FILENAME)


@pytest.fixture
def bundle(tmp_path, monkeypatch, isolated_projects_root):
    """An isolated bundle with a probed menu and no machine-wide answers.

    `isolated_projects_root` matters: `--bundle` must name a calculation
    INSIDE the projects tree, and without it the guard points at the
    developer's real one.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    b = isolated_projects_root / "proj" / "topic" / "calc"
    b.mkdir(parents=True)
    _write_domains(b, [("htc", "htc", "public", "0-04:00:00"),
                       ("general", "general", "public", "14-00:00:00")])
    return b


def _cfg(b, **execution):
    (b / ".molbuilder.json").write_text(json.dumps({
        "execution": {"mode": "submit", **execution},
        "scheduler": {"kind": "slurm",
                      "directives": {"partition": "htc", "qos": "public"}},
    }))


class TestTheDomainTheBundleCarries:

    def test_prep_baked_domain_is_used_without_a_flag(self, bundle):
        """The reported failure, directly: the person chose `htc` in the
        browser, and `launch` must not ask them again."""
        _cfg(bundle)                          # no execution.domain
        _sweep(domain="htc").write(bundle / "job-set.json")
        runner, grp = _runner()
        r = runner.invoke(grp, ["launch", "bench", "--bundle", str(bundle),
                                "--dry-run", "--yes"])
        assert r.exit_code == 0, r.output
        assert "-p htc" in r.output and "-q public" in r.output
        assert "no --domain" not in r.output

    def test_the_bundle_beats_the_machine_wide_default(self, bundle):
        """Most specific wins: `execution.domain` is said once about a
        MACHINE, the bundle's is said about THIS WORK."""
        _cfg(bundle, domain="general")
        _sweep(domain="htc").write(bundle / "job-set.json")
        runner, grp = _runner()
        r = runner.invoke(grp, ["launch", "bench", "--bundle", str(bundle),
                                "--dry-run", "--yes"])
        assert r.exit_code == 0, r.output
        assert "-p htc" in r.output, "the bundle's own domain must win"

    def test_an_explicit_flag_beats_the_bundle(self, bundle):
        """--domain is said about THIS launch, which is more specific
        still -- and is how a person overrides a bundle they are reusing."""
        _cfg(bundle)
        _sweep(domain="htc").write(bundle / "job-set.json")
        runner, grp = _runner()
        r = runner.invoke(grp, ["launch", "bench", "--bundle", str(bundle),
                                "--domain", "general", "--dry-run", "--yes"])
        assert r.exit_code == 0, r.output
        assert "-p general" in r.output

    def test_two_baked_domains_are_named_not_picked(self, bundle):
        """A cpu side and a gpu side may want different queues.  Two baked
        answers are not one answer, so it says which they are rather than
        choosing -- `--domain`/`--gpu-domain` already model the split."""
        _cfg(bundle)
        _sweep(domains=("htc", "general")).write(bundle / "job-set.json")
        runner, grp = _runner()
        r = runner.invoke(grp, ["launch", "bench", "--bundle", str(bundle),
                                "--dry-run", "--yes"])
        assert r.exit_code != 0
        assert "more than one domain" in r.output
        assert "htc" in r.output and "general" in r.output

    def test_nothing_answered_still_refuses(self, bundle):
        """The guard this must not weaken (S5): with no flag, nothing baked
        and no config, the queue is still NOT guessed."""
        _cfg(bundle)
        _sweep().write(bundle / "job-set.json")       # no baked domain
        runner, grp = _runner()
        r = runner.invoke(grp, ["launch", "bench", "--bundle", str(bundle),
                                "--dry-run", "--yes"])
        assert r.exit_code != 0
        assert "no --domain" in r.output


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

    def test_a_hand_edited_job_set_is_normalised_on_load(self):
        """The case that unblocks a bundle already on disk: a file written
        before this rule still carries `4h`, and reading it fixes it."""
        r = Resources.from_dict({"time": "4h", "mem": "256G", "mpi_np": 48})
        assert r.time == "0-04:00:00"

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
