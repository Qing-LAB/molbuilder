"""S3 — a measurement is not portable by default.

`execution/submission.md` S3.  `asu-sol.md` § 5.3 says of `node_type`:
*everything else bounds an allocation; `node_type` is what lets a benchmark
result say whether it may be carried from the domain it was measured on to the
domain a run will use.*

**Nothing read it.** It was the fourth field declared on the record,
serialised, and consulted by no code — after the memory ceiling, the GPU
redirect and the per-core memory default. It is the only one of the four that
guards **scientific validity** rather than a resource: seconds-per-cycle taken
on a 48-core GPU node describes that node, and a walltime derived from it for
a 128-core CPU node is not conservative or optimistic, it is meaningless.

Two halves land together, because a check needs both:

  * **the measurement says where it was taken** — a trial's launch record now
    names the domain it went to, instead of leaving it buried in the `sbatch`
    argv the same file records (parsing that back is the re-derivation A4
    exists to remove);
  * **the apply step compares** — and refuses, rather than warning. A warning
    about a number that is already wrong asks the person to do the comparison
    the framework was holding both halves of.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.jobset.materialize import RUN_LAUNCH_FILE, write_run_launch


class _Domain:
    def __init__(self, name, node_type):
        self.name, self.node_type = name, node_type


class _Placement:
    def __init__(self, domain, partition="htc", qos="public"):
        self.domain, self.partition, self.qos = domain, partition, qos


# --------------------------------------------------------------------- #
#  the measurement says where it was taken                              #
# --------------------------------------------------------------------- #

def test_a_launch_records_the_node_type_it_ran_on(tmp_path):
    from molbuilder.jobset.submit import _placed_on
    p = write_run_launch(
        tmp_path, mode="submit", command=["sbatch", "-p", "general", "x"],
        job_id="1", placed_on=_placed_on(_Placement(
            _Domain("gpu", "gpu-a100"), partition="general")))
    body = json.loads(p.read_text())
    assert body["placed_on"] == {"domain": "gpu", "partition": "general",
                                 "qos": "public", "node_type": "gpu-a100"}


def test_a_run_with_no_placement_says_so_by_ABSENCE(tmp_path):
    """A direct run has no placement.  Absent means *the question cannot be
    answered*, which a reader must not read as a match — the same
    absent-not-null rule `continued_from` already follows."""
    from molbuilder.jobset.submit import _placed_on
    p = write_run_launch(tmp_path, mode="direct", command=["bash", "x"],
                         placed_on=_placed_on(None))
    assert "placed_on" not in json.loads(p.read_text())


def test_the_placement_is_recorded_not_parsed_back_out_of_the_argv(tmp_path):
    """The point of the field.  The partition was always IN this file, inside
    the command — recovering it meant parsing a command line, and a reader
    that does that is re-deriving what the writer knew (A4)."""
    from molbuilder.jobset.submit import _placed_on
    p = write_run_launch(
        tmp_path, mode="submit", command=["sbatch", "-p", "htc", "job.sbatch"],
        placed_on=_placed_on(_Placement(_Domain("htc", "standard"))))
    body = json.loads(p.read_text())
    assert body["placed_on"]["node_type"] == "standard", (
        "the node type is not in the record, so a reader would have to guess "
        "it from the argv -- which carries the partition and never the type")


# --------------------------------------------------------------------- #
#  reading it back                                                      #
# --------------------------------------------------------------------- #

def _trial(root, name, node_type):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    write_run_launch(d, mode="submit", command=["sbatch", "x"],
                     placed_on=({"domain": "d", "partition": "p",
                                 "qos": "q", "node_type": node_type}
                                if node_type else None))


def test_one_node_type_across_the_trials_is_the_answer(tmp_path):
    from molbuilder.jobset._cli import _measured_on
    for n in ("bench-a", "bench-b"):
        _trial(tmp_path, n, "gpu-a100")
    assert _measured_on(tmp_path) == "gpu-a100"


def test_trials_that_disagree_answer_NOTHING(tmp_path):
    """A benchmark whose trials ran on different hardware has no single node
    type to carry anywhere.  Picking one would be inventing the answer."""
    from molbuilder.jobset._cli import _measured_on
    _trial(tmp_path, "bench-a", "gpu-a100")
    _trial(tmp_path, "bench-b", "standard")
    assert _measured_on(tmp_path) is None


def test_an_older_bundle_answers_nothing_rather_than_matching(tmp_path):
    """Records written before the field existed carry no placement.  That is
    *cannot tell*, and the check stays silent on it — but it must not become
    a match, or every old bundle would transfer unchecked."""
    from molbuilder.jobset._cli import _measured_on
    _trial(tmp_path, "bench-a", None)
    assert _measured_on(tmp_path) is None


# --------------------------------------------------------------------- #
#  and the refusal — the half that had no code at all                   #
# --------------------------------------------------------------------- #

def _machine(tmp_path, monkeypatch, target_node_type):
    """A bundle whose target machine's first queue has this node type."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "environment.json").write_text(json.dumps({
        "schema": "molbuilder/environment@2", "scheduler": "slurm",
        "site": {"name": "t", "scheduler": "slurm"},
        "domains": [{"name": "first", "partition": "htc", "qos": "public",
                     "max_time": "0-04:00:00",
                     "node_type": target_node_type}],
        "topology": {"cores": 128},
    }))
    return tmp_path


def _check(base, root):
    from molbuilder.jobset._cli import _refuse_if_measured_elsewhere
    _refuse_if_measured_elsewhere(base, root, None)


def test_a_verdict_measured_elsewhere_is_REFUSED(tmp_path, monkeypatch):
    """**The rule.**  Not a warning: a warning about a number that is already
    wrong hands the person the comparison the framework was holding both
    halves of."""
    import click
    base = _machine(tmp_path, monkeypatch, "standard")
    root = base / "bench"
    _trial(root, "bench-a", "gpu-a100")
    with pytest.raises(click.ClickException) as e:
        _check(base, root)
    msg = str(e.value)
    assert "gpu-a100" in msg and "standard" in msg, (
        "the refusal must name BOTH types -- R10: say what would fit")
    assert "re-run the benchmark" in msg and "flags" in msg, (
        "a refusal owes the two ways out")


def test_the_same_node_type_carries_without_complaint(tmp_path, monkeypatch):
    base = _machine(tmp_path, monkeypatch, "gpu-a100")
    root = base / "bench"
    _trial(root, "bench-a", "gpu-a100")
    _check(base, root)          # must not raise


@pytest.mark.parametrize("measured,target,why", [
    (None, "standard", "the trials do not say where they ran"),
    ("gpu-a100", None, "the target's row states no node type"),
    (None, None, "neither says anything"),
])
def test_it_stays_silent_when_it_cannot_tell(tmp_path, monkeypatch,
                                             measured, target, why):
    """**Silence on the honest unknowns**, and this is deliberate rather than
    lenient: *cannot tell* is not *matches*, and refusing on it would block
    every older bundle and every machine whose record is terse.  What makes
    those cases visible instead is the submission display, where an unknown
    provenance is shown as one (§ 7)."""
    base = _machine(tmp_path, monkeypatch, target)
    root = base / "bench"
    _trial(root, "bench-a", measured)
    _check(base, root)          # must not raise
