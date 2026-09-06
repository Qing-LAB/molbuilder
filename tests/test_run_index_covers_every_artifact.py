"""Every per-run artifact carries the run index — asked of a real run directory.

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

CONVERTED 2026-09-06 — `plans/plan.md` § 5h, cluster 1.
------------------------------------------------------
Every assertion in this file used to read `runwrap.py`, `summarize.py`,
`identity.py` or `monitor.py` **as text** and check for a spelling::

    assert '--log "{basename}-run${{_run_n}}.monitor.log"' in src

That is behaviour-blind in both directions. Reformat the f-string — split it,
re-indent it, build the flag from a variable — and the test fails while the
wrapper is perfect. Change what `_run_n` RESOLVES to, so every attempt lands on
`-run0`, and the test passes while each run destroys the last. The string is
not the behaviour; it is one spelling the behaviour currently happens to have.

**The directory is built by `prep_jobset`, not by hand** *(user, 2026-09-06:
"you should construct run dir with actual backend if you are testing it")*.
The run index is a property of *a directory that already holds attempts* —
the wrapper scans for `-runN` and advances past the highest — so a
hand-assembled directory would prove the wrapper indexes files in a layout
nobody ever creates. Prep is what makes one, and prep is what ships the real
`mb_monitor.py` into it (`jobset/prep.py:181`), so the monitor here is the
real one writing real samples to the flags the real wrapper passed it.

**Only two things are stubbed, and neither is under test.** `siesta`, because
the run index has nothing to do with what the engine computes and
`test_conclusion_marker.py` already established the pattern — *"the whole
meaning lives in shell control flow no unit test of Python can see."* And
`conda`, because `script_generation.activation` is a CLOSED set (the config
refuses anything but `conda activate` / `source activate`), so the rendered
script always shells out to it and a bare `bash` has no `conda init` behind
it — stubbing it keeps the test dependent on itself rather than on whether
the developer's shell happens to be initialised.

What each replacement must fail against is recorded on the test.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from molbuilder.identity import is_ours
from molbuilder.jobset.model import Job, JobSet, Resources
from molbuilder.jobset.prep import prep_jobset
from molbuilder.jobset.summarize import _latest_run_file
from molbuilder.monitor import run_monitor

LABEL = "J_01_coarse"
STAGE_DIR = "01_01_coarse"

#: Long enough for the monitor -- started with a 1 s interval below -- to take
#: at least one sample before the wrapper's cleanup stops it.  A run that ends
#: instantly is not the case this file is about: the artifacts only collide
#: when a run lasts long enough to be measured.
_ENGINE = '#!/bin/bash\nsleep 2\necho "Job completed"\n'
_CONDA = "#!/bin/bash\nexit 0\n"


def _a_prepared_calculation(tmp_path: Path) -> Path:
    """A real one-stage ladder, prepped by the real `prep_jobset`.

    Returns the stage directory -- deck, wrapper and the shipped
    `mb_monitor.py`, exactly as a person would find it after `jobset prep`.
    """
    root = tmp_path / "calc"
    root.mkdir()
    # A bundle carries the script_generation block the wrapper needs; the
    # machine record is the conftest's autouse `write_machine_record`, since
    # 2026-09-02 a precondition rather than something prep arranges.
    (root / ".molbuilder.json").write_text(
        '{"script_generation": {"preamble": "", "activation": "conda activate"}}')
    (root / f"{LABEL}.fdf").write_text(
        "SystemName test\nSystemLabel J\nNumberOfAtoms 2\n"
        "DM.UseSaveDM .false.\nMD.UseSaveXV .false.\n")

    jobset = JobSet(name="J", engine="siesta", kind="ladder",
                    jobs=[Job(name="01_coarse", script=f"{LABEL}.fdf",
                              resources=Resources(mpi_np=1))])
    prep_jobset(jobset, root, env="molbuilder-siesta", emit_sbatch=False)

    stage = root / STAGE_DIR
    assert (stage / "mb_monitor.py").exists(), (
        "prep did not ship mb_monitor.py -- this test would then be measuring "
        "nothing, so it is a precondition rather than an assertion")
    binned = stage / "bin"
    binned.mkdir()
    for name, body in (("siesta", _ENGINE), ("conda", _CONDA)):
        stub = binned / name
        stub.write_text(body)
        stub.chmod(0o755)
    return stage


def _run_the_wrapper(stage: Path) -> None:
    env = dict(os.environ,
               PATH=f"{stage / 'bin'}:{os.environ['PATH']}",
               MB_MONITOR="1", MB_MONITOR_INTERVAL="1",
               MB_LAUNCHED_BY="manual")
    subprocess.run(["bash", f"{LABEL}.run.sh"], cwd=str(stage), env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=180, check=False)


# --------------------------------------------------------------- the wrapper


def test_a_rerun_writes_new_monitor_files_and_never_touches_the_first(tmp_path):
    """The claim in this file's title, run rather than read.

    Two attempts in one prepped directory.  Both monitor artifacts must land
    on the SECOND attempt's index, and the first attempt's must still hold
    their own bytes afterwards -- which is precisely what was false before
    2026-08-27, when `util.csv` had one name and `write_text` truncated it.

    MUTATION THIS MUST FAIL AGAINST: pin `_run_n=0` so the index never
    advances, or build the monitor's `--util` path from something other than
    the resolved index.  Both leave every f-string in `runwrap.py`
    byte-identical, so the retired spelling-pin passed through either one.
    """
    stage = _a_prepared_calculation(tmp_path)

    _run_the_wrapper(stage)
    first_log = stage / f"{LABEL}-run0.monitor.log"
    first_csv = stage / f"{LABEL}-run0.util.csv"
    assert first_log.exists() and first_csv.exists(), (
        "the first attempt wrote no indexed monitor artifacts: "
        f"{sorted(p.name for p in stage.iterdir())}")

    first_csv.write_text("t,cpu\n0,MEASURED\n")      # the earlier measurement
    keep = first_log.read_text()

    _run_the_wrapper(stage)
    assert (stage / f"{LABEL}-run1.monitor.log").exists(), (
        "the re-run did not advance the monitor log's index: "
        f"{sorted(p.name for p in stage.iterdir())}")
    assert (stage / f"{LABEL}-run1.util.csv").exists(), (
        "the re-run did not advance util.csv's index -- the benchmark "
        "measurement is what this destroys")

    assert first_csv.read_text() == "t,cpu\n0,MEASURED\n", (
        "the re-run overwrote the FIRST attempt's util.csv, which is the "
        "whole defect project-layout.md 1.5a exists to close")
    assert first_log.read_text() == keep, (
        "the re-run appended to (or truncated) the first attempt's monitor log")


def test_no_unindexed_monitor_artifact_reaches_the_directory(tmp_path):
    """The stronger form, asked of the DIRECTORY rather than of the source.

    One path writing the indexed name while another writes the bare one is
    invisible to a test that only proves the indexed spelling appears
    somewhere in the file.  Here the bare names simply must not turn up.
    """
    stage = _a_prepared_calculation(tmp_path)
    _run_the_wrapper(stage)

    for bare in (f"{LABEL}.monitor.log", f"{LABEL}.util.csv"):
        assert not (stage / bare).exists(), (
            f"an unindexed {bare} was written -- some path still emits the "
            "pre-2026-08-27 name")


# ---------------------------------------------------------------- the reader


def test_the_reader_takes_the_newest_attempt(tmp_path):
    """Writer and reader change together or nothing is found.

    `summarize` used an exact unindexed name for these two while using
    `_latest_run_file` for the `.out` and the timing log, so the moment the
    writer gained an index the reader stopped seeing them at all.

    MUTATION THIS MUST FAIL AGAINST: `max` -> `min` in `_latest_run_file`
    (it reads a stale attempt), or dropping the `-run*` glob (it reads
    nothing).  The retired test asserted the CALL SITES were spelled with
    `_latest_run_file` and could not have noticed either.
    """
    d = tmp_path / "calc"
    d.mkdir()
    for n, body in ((0, "OLD"), (3, "NEWEST"), (2, "MIDDLE")):
        (d / f"{LABEL}-run{n}.util.csv").write_text(body)
    (d / f"{LABEL}.util.csv").write_text("UNINDEXED")

    found = _latest_run_file(d, LABEL, "util.csv")
    assert found is not None, "the reader found no attempt at all"
    assert found.read_text() == "NEWEST", (
        f"the reader took {found.name}, not the highest run index")

    assert _latest_run_file(d, LABEL, "monitor.log") is None, (
        "a suffix with no attempt on disk must report nothing, not guess")


# ----------------------------------------------------------- the cold sweep


@pytest.mark.parametrize("name", [
    f"{LABEL}-run0.util.csv",
    f"{LABEL}-run17.monitor.log",
    f"{LABEL}.util.csv",               # written before the index existed
    f"{LABEL}.monitor.log",
])
def test_the_cold_sweep_claims_both_spellings(name):
    """`is_ours` decides what a cold restart moves aside.

    A name it does not claim is left in place to be appended to or truncated
    by the next run -- the very failure being fixed.  Both spellings, because
    a directory can hold artifacts written before the change.

    MUTATION THIS MUST FAIL AGAINST: drop either the `-run*` pattern or the
    pre-index one from `OUR_FILE_PATTERNS`.  The retired test asserted four
    pattern STRINGS appeared in `identity.py`; it could not tell whether
    `is_ours` consulted them, and `{label}` vs `{label}_*` had already
    silently become `*` once before (the note at `identity.py:212`).
    """
    assert is_ours(name, LABEL), f"the cold sweep does not claim {name}"


def test_the_wrappers_cold_sweep_names_the_indexed_artifacts(tmp_path):
    """The SECOND consumer of `OUR_FILE_PATTERNS`, and the only one that
    needs its `-run*` rows.

    FOUND BY MUTATION, 2026-09-06.  Deleting `"{label}-run*.util.csv"` from
    the list does not move `is_ours` at all -- it tries every pattern against
    TWO stems (`{label}` and `{label}-*`), so `J-run0.util.csv` still matches
    through the plain `.util.csv` row under the qualified stem.  The wrapper's
    cold-restart list substitutes ONE anchor and does not expand
    (`runwrap.py:490`), so there the row is load-bearing -- and the test above
    would have stayed green while `--cold` silently stopped protecting every
    indexed monitor artifact.

    Asserted on the RENDERED script, which is generated output: a real
    property of a real product, and what reading text is legitimately for.
    """
    stage = _a_prepared_calculation(tmp_path)
    script = (stage / f"{LABEL}.run.sh").read_text()

    for artifact in ("-run*.util.csv", "-run*.monitor.log"):
        assert artifact in script, (
            f"the wrapper's cold-restart list does not name {artifact}, so "
            "--cold would overwrite indexed monitor artifacts without saying "
            "so.  identity.OUR_FILE_PATTERNS is the one enumeration; check "
            "its -run* rows before changing this")


def test_the_cold_sweep_does_not_claim_the_engines_own_files():
    """The inversion only works while it stays narrow: claiming everything
    would move a SIESTA restart file aside as if we had written it."""
    for theirs in ("H.psml", "INPUT_TMP.0", "FORCE_STRESS"):
        assert not is_ours(theirs, LABEL), f"{theirs} is the engine's"


# --------------------------------------------------------------- the pairing


def test_util_csv_truncates_which_is_WHY_the_index_matters(tmp_path):
    """The monitor truncates `util.csv` -- deliberately, since a fresh run's
    samples must not continue a previous run's series.

    That is correct **given a fresh filename** and destructive without one, so
    the two facts are one decision.  This pins the pairing: if the write ever
    became an append, the index would stop being what protects the earlier
    measurement, and this test should be READ AGAIN rather than deleted.

    Now measured by running the monitor twice at one path instead of grepping
    `monitor.py` for `Path(util_path).write_text(` -- a spelling that survives
    being changed to `open(util_path, "a")` two lines further down.
    """
    d = tmp_path / "calc"
    d.mkdir()
    out, timing, log = d / "j.out", d / "j.timing", d / "j.monitor.log"
    out.write_text("Job completed\n")
    util = d / "j.util.csv"

    def _once():
        run_monitor(out, timing, log, util_path=util, max_ticks=1,
                    interval=0.0, sleep=lambda _s: None)

    _once()
    assert util.read_text(), "the monitor wrote no util.csv at all"

    # A SENTINEL, not a row count.  How many samples one call writes is not
    # deterministic -- the sampler is change-gated with a keepalive, so two
    # runs of the same length legitimately differ by a row, and an earlier
    # draft of this test compared counts and flaked 50% of the time.  What
    # IS deterministic is whether what was in the file survives.
    util.write_text("t,cpu\nAN-EARLIER-RUNS-MEASUREMENT\n")
    _once()

    assert "AN-EARLIER-RUNS-MEASUREMENT" not in util.read_text(), (
        "util.csv kept the earlier run's rows -- the write became an append, "
        "so the run index is no longer what protects the earlier measurement. "
        "Re-read project-layout.md 1.5a before changing this.")
