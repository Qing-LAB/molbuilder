"""The summary reads and shows what each trial ran on — never judges it.

`generator.md` § 4.4b: record, present, and stop there.  `scheduler.md` R11:
the comparison is by KIND (cores, memory to the nearest 10 GB, device
models), never by hostname — SLURM spreads a sweep over whatever boxes are
free, so hostname comparison would flag every sweep ever run (trap T1).

Fixtures here write real monitor logs and parse them through the real
readers; nothing injects a machine dict the monitor could not have written.
That rule is `archive/2026-09-01-machine-identity-plan.md` § 7: the migration this belongs to
exists because a check was kept green for four days by fixtures supplying a
value production never wrote.
"""
from pathlib import Path

from molbuilder.bench.result import BenchPoint, machine_brief, machine_kind
from molbuilder.parse.instruments.monitor import monitor_metrics


def _machine(text):
    return monitor_metrics(text)["machine"]
from molbuilder.jobset.summarize import parse_point, summary_text
from molbuilder.bench.result import BenchResult


A100_LINE = ("[2026-08-27T14:02:11] [MACHINE] node={host} cores=48 "
             "mem_gb={mem} gpu=NVIDIA A100-SXM4-80GB\n")
STD_LINE = ("[2026-08-27T14:02:11] [MACHINE] node={host} cores=128 "
            "mem_gb=503.2 gpu=none\n")


def _trial(d: Path, basename: str, machine_line: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{basename}-run0.monitor.log").write_text(
        machine_line
        + "[2026-08-27T14:02:11] [MONITOR] start (interval=30s "
          "watch_pid=1) state=starting\n", encoding="utf-8")


# ------------------------------------------------------------------ parse

def test_parse_reads_the_line_and_a_legacy_log_reads_empty(tmp_path):
    m = _machine(A100_LINE.format(host="sol-g042", mem="503.5"))
    assert m == {"node": "sol-g042", "cores": "48", "mem_gb": "503.5",
                 "gpu": "NVIDIA A100-SXM4-80GB"}
    assert _machine("[ts] [MONITOR] start ...\n") == {}, (
        "a log from before the [MACHINE] line must read as absent, "
        "not raise or invent")


def test_parse_point_carries_the_machine(tmp_path):
    _trial(tmp_path, "j", A100_LINE.format(host="sol-g042", mem="503.5"))
    pt = parse_point("t", tmp_path, "j", "gpu", {})
    assert pt.machine.get("node") == "sol-g042"
    assert pt.machine.get("gpu") == "NVIDIA A100-SXM4-80GB"


# ------------------------------------------------------------- T1: the kind

def test_same_kind_on_two_hosts_is_one_machine():
    """The T1 guard.  Two boxes, same silicon, MemTotal jittered by BIOS
    reservations (the real figures from Sol's standard pool) — one kind.
    Compare hostnames or exact memory instead and every sweep warns."""
    a = _machine(A100_LINE.format(host="sol-g042", mem="503.4"))
    b = _machine(A100_LINE.format(host="sol-g117", mem="503.5"))
    assert machine_kind(a) == machine_kind(b)


def test_different_hardware_is_a_different_kind():
    a = _machine(A100_LINE.format(host="h1", mem="503.5"))
    b = _machine(STD_LINE.format(host="h2"))
    assert machine_kind(a) != machine_kind(b)


def test_absent_machine_has_no_kind():
    """*Cannot tell* is not a kind (R3): a pre-[MACHINE] record must not
    compare equal to anything, including another absent one's ''."""
    assert machine_kind({}) is None
    assert machine_brief({}) == ""


# ---------------------------------------------------------------- showing

def _result(points):
    return BenchResult(environment={}, system={}, points=points, choice={})


def _pt(label, log_line, tmp_path, engine="cpu"):
    d = tmp_path / label
    _trial(d, "j", log_line)
    return parse_point(label, d, "j", engine, {})


def test_two_kinds_are_stated_plainly_and_without_judgement(tmp_path):
    pts = [_pt("g1", A100_LINE.format(host="a", mem="503.5"), tmp_path,
               engine="gpu"),
           _pt("c1", STD_LINE.format(host="b"), tmp_path),
           _pt("c2", STD_LINE.format(host="c"), tmp_path)]
    text = summary_text(_result(pts), tmp_path / "r.json")
    assert "2 kinds of node" in text
    assert "48c 500G A100" in text and "128c 500G no gpu" in text
    for verdict_word in ("warning", "invalid", "not comparable", "!!"):
        assert not any(verdict_word in ln for ln in text.splitlines()
                       if "kinds of node" in ln), (
            "the machine statement judged the comparison — 4.4b says "
            "present the data, the reader is the analyzer")


def test_one_kind_on_many_hosts_says_nothing(tmp_path):
    """Six identical boxes are ONE machine — the statement must stay quiet
    or it becomes noise attached to every healthy sweep (T1)."""
    pts = [_pt(f"c{i}", STD_LINE.format(host=f"h{i}"), tmp_path)
           for i in range(3)]
    text = summary_text(_result(pts), tmp_path / "r.json")
    assert "kinds of node" not in text


def test_the_table_names_each_trials_machine(tmp_path):
    pts = [_pt("g1", A100_LINE.format(host="a", mem="503.5"), tmp_path,
               engine="gpu"),
           _pt("c1", STD_LINE.format(host="b"), tmp_path)]
    text = summary_text(_result(pts), tmp_path / "r.json")
    head = next(ln for ln in text.splitlines() if "machine" in ln)
    assert "machine" in head, "no machine column despite recorded machines"


def test_legacy_records_get_no_machine_column(tmp_path):
    """Absent is absent: a sweep recorded before the [MACHINE] line must
    render exactly as before — no column of '--'."""
    d = tmp_path / "old"
    d.mkdir()
    (d / "j-run0.monitor.log").write_text(
        "[ts] [MONITOR] start ...\n", encoding="utf-8")
    pt = parse_point("old", d, "j", "cpu", {})
    text = summary_text(_result([pt]), tmp_path / "r.json")
    assert "machine" not in text.splitlines()[1], (
        "a legacy sweep grew a machine column with nothing to put in it")


# ------------------------------------------------------------- round-trip

def test_machine_survives_the_result_json(tmp_path):
    pt = _pt("g1", A100_LINE.format(host="a", mem="503.5"), tmp_path)
    res = _result([pt])
    back = BenchResult.from_dict(res.to_dict())
    assert back.points[0].machine == pt.machine, (
        "the machine field fell out of the bench-result round-trip; the "
        "web page composes from this JSON and would show nothing")
