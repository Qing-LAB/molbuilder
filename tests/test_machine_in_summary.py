"""The summary reads and shows what each trial ran on — never judges it.

`generator.md` § 4.4b: record, present, and stop there.  `scheduler.md` R11:
the comparison is by KIND (cores, memory to the nearest 10 GB, device
models), never by hostname — SLURM spreads a sweep over whatever boxes are
free, so hostname comparison would flag every sweep ever run (trap T1).

The tests that parsed monitor lines here typed those lines by hand and were
retired 2026-10-04 (`process/testing.md` § 6); a monitor line is read from a
log the monitor wrote.
"""
from molbuilder.bench.result import machine_brief, machine_kind


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 3 tests here read monitor lines typed by hand (`process/testing.md` § 6).


# ------------------------------------------------------------- T1: the kind


def test_absent_machine_has_no_kind():
    """*Cannot tell* is not a kind (R3): a pre-[MACHINE] record must not
    compare equal to anything, including another absent one's ''."""
    assert machine_kind({}) is None
    assert machine_brief({}) == ""
