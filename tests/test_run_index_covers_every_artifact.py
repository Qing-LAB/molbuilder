"""Every per-run artifact carries the run index — not two of five.

`project-layout.md` § 1.5a. In a **flat** calculation attempts are told apart
by the wrapper's filename index, and its own docstring is emphatic: *"any later
run AUTO-ADVANCES to max(N)+1 by default so re-running NEVER overwrites."*

That was true of the `.out` and the timing log, and **false of the other two**:

* `<basename>.monitor.log` was appended to — two runs interleaved, no marker;
* `<basename>.util.csv` was written with `write_text`, so a re-run **truncated
  it**.

`util.csv` is what a benchmark is measured from, so re-running a trial
destroyed the measurement it existed to repeat. Found 2026-08-27 by reading the
write mode rather than the design, and **not a sweep problem**: a flat ladder
stage re-run loses its `util.csv` today for the same reason.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _wrapper_source() -> str:
    return (ROOT / "molbuilder/runwrap.py").read_text()


def test_the_monitor_files_carry_the_run_index():
    """Both of them, in the line the wrapper actually emits."""
    src = _wrapper_source()
    assert '--log "{basename}-run${{_run_n}}.monitor.log"' in src
    assert '--util "{basename}-run${{_run_n}}.util.csv"' in src


def test_no_unindexed_monitor_artifact_is_emitted():
    """The stronger form: the old spelling must not survive anywhere in the
    emitted script, or one path writes the indexed name and another the
    bare one."""
    src = _wrapper_source()
    for bad in ('"{basename}.monitor.log"', '"{basename}.util.csv"'):
        assert bad not in src, f"the wrapper still emits {bad}"


def test_the_reader_looks_for_the_indexed_name():
    """Writer and reader change together or nothing is found. `summarize`
    used `_latest_run_file` for the `.out` and the timing log and an exact
    unindexed name for these two."""
    src = (ROOT / "molbuilder/jobset/summarize.py").read_text()
    reads = set(re.findall(r'_latest_run_file\(d, basename, "([^"]+)"\)', src))
    assert {"out", "scf-timing.log", "monitor.log", "util.csv"} <= reads, reads
    assert 'd / f"{basename}.util.csv"' not in src
    assert 'd / f"{basename}.monitor.log"' not in src


def test_the_cold_sweep_knows_both_spellings():
    """`identity.py` lists the files that belong to a run, and a cold
    restart moves them aside. A pattern that misses the new name leaves the
    file to be appended to or truncated by the next run — which is the very
    failure being fixed.

    Both spellings, because a directory can hold artifacts written before
    the change.
    """
    src = (ROOT / "molbuilder/identity.py").read_text()
    for pat in ('"{label}-run*.monitor.log"', '"{label}_*-run*.monitor.log"',
                '"{label}-run*.util.csv"', '"{label}_*-run*.util.csv"'):
        assert pat in src, f"the cold sweep does not know {pat}"
    # and the pre-index spellings are still listed
    assert '"{label}.util.csv"' in src


def test_util_csv_still_truncates_which_is_WHY_the_index_matters():
    """The monitor writes `util.csv` with `write_text` — deliberately, since
    a fresh run's samples must not continue a previous run's series.

    That is correct **given a fresh filename**, and destructive without one.
    This pins the pairing: if the write ever became an append, the index
    would stop being what protects the earlier measurement, and this test
    should be read again rather than deleted.
    """
    src = (ROOT / "molbuilder/monitor.py").read_text()
    assert "Path(util_path).write_text(" in src, (
        "util.csv no longer truncates -- re-read the reasoning in "
        "project-layout.md 1.5a before changing this test")
