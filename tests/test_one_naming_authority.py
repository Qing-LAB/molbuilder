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
    from molbuilder.jobset.shape import Shape
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


def test_prep_asks_rather_than_composes():
    """The specific regression: `prep_calculation` joining a container to a
    hand-built `bench-<point>`."""
    src = PREP.read_text()
    assert "trial_dir(_shape, token, _pt(element.point))" in src
    assert 'bench_container(_shape, token) \\' not in src, \
        "prep composes the trial path again"


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
    from molbuilder.jobset.shape import Shape
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
