"""The summary reads and shows what each trial ran on — never judges it.

`generator.md` § 4.4b: record, present, and stop there.  `scheduler.md` R11:
the comparison is by KIND (cores, memory to the nearest 10 GB, device
models), never by hostname — SLURM spreads a sweep over whatever boxes are
free, so hostname comparison would flag every sweep ever run (trap T1).

The lines here are the monitor's own and are parsed by its one reader;
nothing injects a machine dict the monitor could not have written.
That rule is `archive/2026-09-01-machine-identity-plan.md` § 7: the migration this belongs to
exists because a check was kept green for four days by fixtures supplying a
value production never wrote.
"""
from molbuilder.bench.result import machine_brief, machine_kind
from molbuilder.parse.instruments.monitor import monitor_metrics


def _machine(text):
    return monitor_metrics(text)["machine"]


A100_LINE = ("[2026-08-27T14:02:11] [MACHINE] node={host} cores=48 "
             "mem_gb={mem} gpu=NVIDIA A100-SXM4-80GB\n")
STD_LINE = ("[2026-08-27T14:02:11] [MACHINE] node={host} cores=128 "
            "mem_gb=503.2 gpu=none\n")


# ------------------------------------------------------------------ parse

def test_parse_reads_the_line_and_a_legacy_log_reads_empty(tmp_path):
    m = _machine(A100_LINE.format(host="sol-g042", mem="503.5"))
    assert m == {"node": "sol-g042", "cores": "48", "mem_gb": "503.5",
                 "gpu": "NVIDIA A100-SXM4-80GB"}
    assert _machine("[ts] [MONITOR] start ...\n") == {}, (
        "a log from before the [MACHINE] line must read as absent, "
        "not raise or invent")


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
