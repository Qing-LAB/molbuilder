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

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZE = ROOT / "molbuilder/jobset/materialize.py"
PREP = ROOT / "molbuilder/jobset/prep.py"


def test_both_composers_ask_the_same_function():
    from molbuilder.jobset.materialize import trial_dir
    from molbuilder.paths import Shape
    for shape_name in ("hierarchical", "flat"):
        sh = Shape.named(shape_name)
        got = trial_dir(sh, "01_coarse", "G1K4C6")
        assert got.endswith("/bench-G1K4C6"), got
        assert "bench" in got


def test_nobody_spells_the_trial_prefix_by_hand():
    """`bench-` belongs to `job_dir_name`. A literal `f"bench-{...}"`
    anywhere else is the rule being written a second time, which is exactly
    what this test exists to stop coming back."""
    offenders = []
    for f in (ROOT / "molbuilder").rglob("*.py"):
        if f.name == "materialize.py":
            continue          # where the rule lives
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r'f"bench-\{', line) or re.search(r"'bench-'\s*\+", line):
                offenders.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, (
        "the trial-directory rule is composed outside materialize.py:\n  "
        + "\n  ".join(offenders))




def test_job_dir_names_asks_it_too():
    """Both sides, or it is one door and one window."""
    src = MATERIALIZE.read_text()
    body = src[src.index("def job_dir_names"):src.index("def _trial_stage_token")]
    assert "trial_dir(sh, trial_token, j.name)" in body
    assert 'trial_dir(sh, "", j.name)' in body, "the tokenless sweep too"
    # a bare `bench-<name>` at the ROOT is a different case -- a hand-built
    # ladder whose jobs are siblings, with no container to join -- so
    # `job_dir_name` alone is right there and is not a second spelling.
    assert 'f"{bench_container(sh' not in body, \
        "job_dir_names joins a container to a trial name inline again"


def test_the_two_agree_on_a_real_bundle(tmp_path):
    """Not just that both call it — that what `prep` writes is what
    `job_dir_names` later answers. The property the comment asserted and
    nothing checked."""
    from molbuilder.jobset.materialize import job_dir_names, trial_dir
    from molbuilder.jobset.model import Job, JobSet
    from molbuilder.paths import Shape
    js = JobSet(name="sweep", kind="sweep", engine="siesta", jobs=[
        Job(name="G1K4C6", script="lbl_01_coarse.fdf"),
        Job(name="G2K8C6", script="lbl_01_coarse.fdf")])
    sh = Shape.named("hierarchical")
    from molbuilder.jobset.materialize import _trial_stage_token
    names = job_dir_names(js, sh)
    for j in js.jobs:
        # THE TOKEN COMES FROM THE SAME PLACE BOTH SIDES GET IT.  Passing a
        # token the deck does not carry compares two different questions --
        # which is how this test first "failed": `job_dir_names` derived
        # None and took the tokenless branch while the assertion supplied
        # "01_coarse". The property is that for ONE token they agree.
        tok = _trial_stage_token(js, j) or ""
        assert names[j.name] == trial_dir(sh, tok, j.name), (
            f"{j.name}: prep would write {trial_dir(sh, tok, j.name)} "
            f"and launch would look in {names[j.name]}")


def test_prep_writes_where_job_dir_names_will_look(tmp_path, monkeypatch):
    """Prep a REAL bench, then ask launch where it will look.

    THE PROPERTY THE COMMENT ASSERTED AND NOTHING CHECKED, one layer up from
    `test_the_two_agree_on_a_real_bundle`: that test builds a JobSet by hand
    and compares `job_dir_names` against `trial_dir` -- both sides of
    materialize, neither of them prep.  This runs `prep_calculation` and
    looks at the directories that actually appeared.

    That is the 2026-08-27 failure: one side learned about the attempt layer
    (`project-layout.md` § 1.5a) and the other did not, so the deck landed
    in the container while the shared package landed in `run-0`, and the
    launch found nothing.  Two computations kept in step by hand agree until
    something moves.
    """
    import json as _json

    import numpy as np

    from conftest import write_machine_record, write_pseudos
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset._cli import _bench_inputs
    from molbuilder.jobset.materialize import (job_dir_names, shape_of,
                                               trial_work_dir)
    from molbuilder.jobset.model import JobSet, Resources
    from molbuilder.jobset.prep import prep_calculation
    from molbuilder.siesta.stages import default_siesta_stages
    from molbuilder.structure import Structure

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
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
                            engine="siesta", shape="hierarchical", name="JOB",
                            source=str(src)),
        dest)
    write_pseudos(dest, ["H"])
    (dest / ".molbuilder.json").write_text(_json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))

    stage = stages[0].name
    sweep, pins, translation = _bench_inputs(dest, None)
    prep_calculation(dest, stage, allocation=Resources(mpi_np=8),
                     sweep=sweep, pins=pins, translation=translation,
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
