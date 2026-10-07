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

**The directory is built by `jobset prep`, not by hand** *(user, 2026-09-06:
"you should construct run dir with actual backend if you are testing it")*.
The run index is a property of *a directory that already holds attempts* —
the wrapper scans for `-runN` and advances past the highest — so a
hand-assembled directory would prove the wrapper indexes files in a layout
nobody ever creates. Prep is what makes one, and prep is what ships the real
`mb_monitor.pyz` into it, so the monitor here is the
real one writing real samples to the flags the real wrapper passed it.

**Only two things are stubbed, and neither is under test.** `siesta`, because
the run index has nothing to do with what the engine computes: its whole
meaning lives in the run script's shell control flow. And
`conda`, because the record's activation is a CLOSED set (anything but
`conda activate` / `source activate` is refused), so the rendered
script always shells out to it and a bare `bash` has no `conda init` behind
it — stubbing it keeps the test dependent on itself rather than on whether
the developer's shell happens to be initialised.

What each replacement must fail against is recorded on the test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.identity import is_ours

#: The stage's files' stem in a flat calculation -- where a run's attempts
#: are told apart by the index in their names (`project-layout.md` § 1.5a).
LABEL = "H2_01_coarse"

#: Long enough for the monitor -- started with a 1 s interval below -- to take
#: at least one sample before the wrapper's cleanup stops it.  A run that ends
#: instantly is not the case this file is about: the artifacts only collide
#: when a run lasts long enough to be measured.
_ENGINE = '#!/bin/bash\nsleep 2\necho "stand-in engine"\n'
_CONDA = "#!/bin/bash\nexit 0\n"


def _a_prepared_calculation(tmp_path: Path, monkeypatch,
                            engine: str = _ENGINE) -> Path:
    """A real flat calculation, described by `jobset init` and prepped by
    `jobset prep` -- the road a person takes.

    Returns the folder its stage runs in -- deck, wrapper and the shipped
    `mb_monitor.pyz`, exactly as a person would find it after `jobset prep`.
    """
    from support.road import describe_calculation, jobset
    bundle = describe_calculation(tmp_path, monkeypatch, shape="flat")
    r = jobset("prep", "run", "coarse", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    stage = bundle
    assert (stage / "mb_monitor.pyz").exists(), (
        "prep did not ship mb_monitor.pyz -- this test would then be "
        "measuring nothing, so it is a precondition rather than an assertion")
    binned = stage / "bin"
    binned.mkdir()
    for name, body in (("siesta", engine), ("conda", _CONDA)):
        stub = binned / name
        stub.write_text(body)
        stub.chmod(0o755)
    return stage


# ----------------------------------------------------------- the cold sweep


@pytest.mark.parametrize("name", [
    f"{LABEL}-run0.util.csv",
    f"{LABEL}-run17.monitor.log",
])
def test_the_cold_sweep_claims_the_monitors_numbered_files(name):
    """`is_ours` decides what a cold restart moves aside.

    A name it does not claim is left in place to be appended to or truncated
    by the next run -- the very failure being fixed.  The monitor's files
    carry the run index, always (`runfiles.WRITTEN`).

    MUTATION THIS MUST FAIL AGAINST: drop the `-run*` pattern from
    `OUR_FILE_PATTERNS`.
    """
    assert is_ours(name, LABEL), f"the cold sweep does not claim {name}"


def test_the_wrappers_cold_sweep_names_the_indexed_artifacts(tmp_path,
                                                             monkeypatch):
    """The SECOND consumer of `OUR_FILE_PATTERNS`, and the only one that
    needs its `-run*` rows.

    FOUND BY MUTATION, 2026-09-06.  Deleting `"{label}-run*.util.csv"` from
    the list does not move `is_ours` at all -- it tries every pattern against
    TWO stems (`{label}` and `{label}-*`), so `J-run0.util.csv` still matches
    through the plain `.util.csv` row under the qualified stem.  The wrapper's
    cold-restart list substitutes ONE anchor and does not expand, so there the row is load-bearing -- and the test above
    would have stayed green while `--cold` silently stopped protecting every
    indexed monitor artifact.

    Asserted on the RENDERED script, which is generated output: a real
    property of a real product, and what reading text is legitimately for.
    """
    stage = _a_prepared_calculation(tmp_path, monkeypatch)
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
