"""A trial's directory is composed in ONE place.

The rule — `<container>/bench-<point>` — was written twice: `job_dir_names`
composed it for a whole JobSet, and `prep.prep_calculation` composed it again
from the same two facts. The second carried a comment saying so and calling it
safe:

    "The directory is the same one `job_dir_names` will answer for this job,
     computed from the same two facts (token + trial-ness), so the deck is
     born where the launch will look for it."

They did agree. **A second computation kept in step by hand only ever agrees
until something moves** — and what moved was the attempt layer
(`project-layout.md` § 1.5a): one side learned about `run-<n>` and the other
did not, so the deck landed in the container while the shared package landed in
the attempt. Found 2026-08-27 by attempting that change and watching a deck go
missing.

`prep` cannot simply call `job_dir_names`: it is **building** the JobSet in the
loop that needs the directory, so there is nothing to ask yet. That is what
makes a shared *rule* the fix rather than a shared lookup.
"""
from __future__ import annotations


def test_both_composers_ask_the_same_function():
    from molbuilder.jobset.materialize import trial_dir
    from molbuilder.paths import Shape
    for shape_name in ("hierarchical", "flat"):
        sh = Shape.named(shape_name)
        got = trial_dir(sh, "01_coarse", "G1K4C6")
        assert got.endswith("/bench-G1K4C6"), got
        assert "bench" in got


# `test_the_two_agree_on_a_real_bundle` retired 2026-10-06 (MEMORY gate 5):
# a hand-built JobSet whose trial decks are not named as prep names them.
# The test below preps a REAL benchmark and asks launch where it looks.


def test_prep_writes_where_job_dir_names_will_look(tmp_path, monkeypatch):
    """Prep a REAL bench, then ask launch where it will look.

    THE PROPERTY THE COMMENT ASSERTED AND NOTHING CHECKED, one layer up from
    `test_the_two_agree_on_a_real_bundle`: that test builds a JobSet by hand
    and compares `job_dir_names` against `trial_dir` -- both sides of
    materialize, neither of them prep.  This runs `prep bench` through the
    one entry and looks at the directories that actually appeared.

    That is the 2026-08-27 failure: one side learned about the attempt layer
    (`project-layout.md` § 1.5a) and the other did not, so the deck landed
    in the container while the shared package landed in `run-0`, and the
    launch found nothing.  Two computations kept in step by hand agree until
    something moves.
    """

    import numpy as np

    from conftest import write_machine_record, write_pseudos
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.materialize import (job_dir_names, shape_of,
                                               trial_work_dir)
    from molbuilder.jobset.model import JobSet, Resources
    from molbuilder.jobset.prep import prep_stage
    from molbuilder.siesta.stages import default_siesta_stages
    from molbuilder.structure import Structure
    monkeypatch.chdir(tmp_path)
    write_machine_record()

    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    src = tmp_path / "h2.xyz"
    src.write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    stages = default_siesta_stages("publishable")
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"), stages,
                            engine="siesta",
                            calculation="optimization",
                            shape="hierarchical", name="JOB",
                            source=str(src)),
        dest)
    write_pseudos(dest, ["H"])

    stage = stages[0].name
    prep_stage(dest, "bench", stage, allocation=Resources(mpi_np=8),
               emit_sbatch=False)

    decks = sorted(dest.rglob("job-set.json"))
    assert len(decks) == 1, [str(d.relative_to(dest)) for d in decks]
    js = JobSet.load(decks[0])          # the deck reader that exists
    assert js.jobs, "the bench prepped no trials, so nothing below is tested"

    shape = shape_of(js, dest)
    where = job_dir_names(js, shape)
    for job in js.jobs:
        answered = dest / where[job.name]
        assert answered.is_dir(), (
            f"launch will look in {where[job.name]} for {job.name!r} and "
            f"prep created no such directory.  What prep DID create: "
            f"{sorted(str(p.relative_to(dest)) for p in dest.rglob('bench-*'))}")
        work = trial_work_dir(answered, shape)
        assert any(work.iterdir()), (
            f"{work.relative_to(dest)} is empty -- prep answered the right "
            f"directory and then wrote the trial's files somewhere else, "
            f"which is the attempt-layer split of 2026-08-27")
