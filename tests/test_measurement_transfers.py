"""S3 — a measurement is not portable by default.

`execution/submission.md` S3, applied at the one moment nobody is looking:
`prep run` carrying a benchmark's verdict into a production allocation.

**This file's previous version is the cautionary tale the migration plan
tells** (`archive/2026-09-01-machine-identity-plan.md` § 0).  It built a fake domain carrying
`node_type="standard"` and asserted the plumbing moved it — both ends were
fixtures, so it proved the wire and never the current, and the check it
"covered" had never once fired in production: the probe never wrote that
field.  The scalar is retired (`scheduler.md` R11).

Now the measured side comes from the verdict's own record —
`bench-result.json` points carrying the monitor's `[MACHINE]` line — and
every machine in these fixtures enters through `parse_machine` on a line
the monitor really writes.  The target side is a probed-shape row whose
`node_types` list is what the probe really produces.  The comparison is
CORES only, by decision recorded in the plan's P4: it is the one fact both
sides state in one vocabulary, and the Sol hazard every document cites
(48-core GPU measurement carried to a 128-core CPU run) is a cores
mismatch.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.bench.result import BenchPoint, BenchResult, parse_machine


A100_LINE = ("[t] [MACHINE] node={host} cores=48 mem_gb=503.5 "
             "gpu=NVIDIA A100-SXM4-80GB\n")
STD_LINE = "[t] [MACHINE] node={host} cores=128 mem_gb=503.2 gpu=none\n"


def _verdict(root, *machine_lines):
    """A ``bench-result.json`` whose points ran on these machines — each
    machine entering through the real parser, never a hand-typed dict."""
    root.mkdir(parents=True, exist_ok=True)
    pts = [BenchPoint(label=f"t{i}", engine="cpu", state="completed",
                      machine=parse_machine(ln) if ln else {})
           for i, ln in enumerate(machine_lines)]
    (root / "bench-result.json").write_text(
        BenchResult(points=pts).to_json(), encoding="utf-8")


def _machine(tmp_path, monkeypatch, node_types):
    """A bundle whose target queue holds these machines — the probed shape:
    ``node_types`` rows, no scalar (the probe never wrote one)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir(exist_ok=True)
    row = {"name": "first", "partition": "htc", "qos": "public",
           "max_time": "0-04:00:00"}
    if node_types is not None:
        row["node_types"] = node_types
    (tmp_path / "environment.json").write_text(json.dumps({
        "schema": "molbuilder/environment@2", "scheduler": "slurm",
        "site": {"name": "t", "scheduler": "slurm"},
        "domains": [row],
        "topology": {"cores": 128},
    }))
    return tmp_path


def _check(base, root):
    from molbuilder.jobset._cli import _refuse_if_measured_elsewhere
    _refuse_if_measured_elsewhere(base, root, None)


# --------------------------------------------------------------------- #
#  reading the measured side — the verdict's own record                 #
# --------------------------------------------------------------------- #

def test_one_kind_across_the_trials_is_the_answer(tmp_path):
    from molbuilder.jobset._cli import _measured_on
    _verdict(tmp_path, A100_LINE.format(host="g042"),
             A100_LINE.format(host="g117"))
    kinds = _measured_on(tmp_path)
    assert len(kinds) == 1, (
        "two hosts of one kind are ONE machine (R11) -- comparing "
        "hostnames would refuse every sweep ever run")
    assert list(kinds.values()) == ["48c 500G A100"]


def test_a_record_before_the_MACHINE_line_answers_nothing(tmp_path):
    """*Cannot tell* — and the check stays silent on it, but it must not
    become a match, or every old record would transfer unchecked."""
    from molbuilder.jobset._cli import _measured_on
    _verdict(tmp_path, "", "")
    assert _measured_on(tmp_path) == {}


def test_no_record_at_all_answers_nothing(tmp_path):
    from molbuilder.jobset._cli import _measured_on
    assert _measured_on(tmp_path / "nowhere") == {}


# --------------------------------------------------------------------- #
#  the refusal                                                          #
# --------------------------------------------------------------------- #

def test_a_verdict_the_target_rules_out_is_REFUSED(tmp_path, monkeypatch):
    """**The rule.**  Measured on 48-core nodes; the target's menu holds
    only 128-core machines.  Not a warning: a warning about a number that
    is already wrong hands the person the comparison the framework was
    holding both halves of."""
    import click
    base = _machine(tmp_path, monkeypatch,
                    [{"cores": 128, "nodes": 107, "mem_gb": 503.5}])
    root = base / "bench"
    _verdict(root, A100_LINE.format(host="g042"))
    with pytest.raises(click.ClickException) as e:
        _check(base, root)
    msg = str(e.value)
    assert "48c 500G A100" in msg and "128" in msg, (
        "the refusal must name both sides -- R10: say what would fit")
    assert "re-run the benchmark" in msg and "flags" in msg, (
        "a refusal owes the ways out")


def test_trials_on_SEVERAL_kinds_are_refused_by_name(tmp_path, monkeypatch):
    """D3's fix: *disagree* is a stronger fact than *unknown* and stops
    being spelled like it.  A verdict that ranked measurements of two
    machines against each other has no single basis to carry."""
    import click
    base = _machine(tmp_path, monkeypatch,
                    [{"cores": 48, "nodes": 51, "mem_gb": 503.5,
                      "gpu": {"a100": 4}}])
    root = base / "bench"
    _verdict(root, A100_LINE.format(host="g042"), STD_LINE.format(host="c1"))
    with pytest.raises(click.ClickException) as e:
        _check(base, root)
    msg = str(e.value)
    assert "2 kinds of node" in msg
    assert "48c 500G A100" in msg and "128c 500G no gpu" in msg


def test_a_menu_holding_the_measured_cores_carries(tmp_path, monkeypatch):
    """`public`'s real shape: 128-core standard nodes AND 48-core A100
    nodes.  A 48-core measurement is not ruled out — the scheduler may
    land the run on that very kind — so the check stays quiet (R3:
    refuse only what the record positively rules out)."""
    base = _machine(tmp_path, monkeypatch,
                    [{"cores": 128, "nodes": 107, "mem_gb": 503.5},
                     {"cores": 48, "nodes": 52, "mem_gb": 503.5,
                      "gpu": {"a100": 4}}])
    root = base / "bench"
    _verdict(root, A100_LINE.format(host="g042"))
    _check(base, root)          # must not raise


@pytest.mark.parametrize("machine_lines,node_types,why", [
    (("",), [{"cores": 128, "nodes": 1, "mem_gb": 503.5}],
     "the record predates the [MACHINE] line"),
    ((A100_LINE.format(host="g042"),), None,
     "the target row lists no machines"),
    (("",), None, "neither side says anything"),
])
def test_it_stays_silent_when_it_cannot_tell(tmp_path, monkeypatch,
                                             machine_lines, node_types, why):
    """**Silence on the honest unknowns**, deliberate rather than lenient:
    *cannot tell* is not *matches*, and refusing on it would block every
    older record and every machine whose record is terse."""
    base = _machine(tmp_path, monkeypatch, node_types)
    root = base / "bench"
    _verdict(root, *machine_lines)
    _check(base, root)          # must not raise


# --------------------------------------------------------------------- #
#  the SENT half stays the queue's — and only the queue's               #
# --------------------------------------------------------------------- #

def test_run_json_records_where_it_was_SENT_and_no_machine(tmp_path):
    """R12's split, from the writing side: ``placed_on`` is domain,
    partition, qos — known at sbatch-accept.  A machine field here would
    be the queue's opinion of itself, which is exactly the retired scalar
    coming back through the launch record."""
    from molbuilder.jobset.materialize import write_run_launch
    from molbuilder.jobset.submit import _placed_on

    class _Domain:
        name = "gpu"

    class _Placement:
        domain, partition, qos = _Domain(), "general", "public"

    p = write_run_launch(tmp_path, mode="submit",
                         command=["sbatch", "-p", "general", "x"],
                         job_id="1", placed_on=_placed_on(_Placement()))
    body = json.loads(p.read_text())
    assert body["placed_on"] == {"domain": "gpu", "partition": "general",
                                 "qos": "public"}


def test_a_run_with_no_placement_says_so_by_ABSENCE(tmp_path):
    """A direct run has no placement.  Absent means *the question cannot
    be answered*, which a reader must not read as a match — the same
    absent-not-null rule ``continued_from`` follows."""
    from molbuilder.jobset.materialize import write_run_launch
    from molbuilder.jobset.submit import _placed_on
    p = write_run_launch(tmp_path, mode="direct", command=["bash", "x"],
                         placed_on=_placed_on(None))
    assert "placed_on" not in json.loads(p.read_text())
