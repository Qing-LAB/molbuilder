"""Who answers each GPU question — the line every GPU defect crossed.

`execution/gpu.md` names three answerers: the person, the machine, and the
probe.  Five GPU defects in one month were each a GPU fact with more than one
home, and **two of the four self-contradictions the contract found put the
person's choice on the machine's side** — the single distinction everything
else depends on.

That error had already been corrected once, in `engines/stages.md`, with a
dated 2026-08-07 note saying the earlier wording *"invited the other
reading"*.  It was restated in `engines/tuning.md` anyway and sat there for
sixteen days.  Prose cannot stop that; this can.

**What the catalogue already guards, and what it cannot.**  Marking a setting
`allocation = true` while it still carries a `value` is refused at parse time
-- *"a template declares the question and never asserts the answer"* -- so the
careless version of this mistake cannot reach the tree.  The COHERENT version
can: drop the value, add the flag, and the file parses cleanly with the GPU
flag on the machine's side.  Verified by mutation 2026-08-23, where only the
two tests below noticed.  That gap is what this file is for.
"""
from __future__ import annotations

import pytest

#: `gpu.md` G2.  The machine answers what the SCHEDULER grants; a person
#: answers everything else in the staging group.  Membership is the
#: catalogue's `allocation` flag, read through one door.
_MACHINE_ANSWERS = {"mpi_np", "gpu_count", "omp_threads", "max_memory_mb",
                    "threads"}


def _item(name):
    """One item, through the one read API (`engines/template.md` § 8.0) --
    never a comprehension over `.items`."""
    from molbuilder.template import catalogue, one
    return one(catalogue(), name)


def _allocation_names():
    """The machine-answered set, asked of the catalogue on the axis the item
    already declares.  `select(t, allocation=True)` IS the question; deriving
    it any other way would be the second reader this file exists to prevent."""
    from molbuilder.template import catalogue, select
    return {i.name for i in select(catalogue(), allocation=True)}


def test_the_machine_answered_set_is_exactly_what_the_contract_names():
    """**G2**, from the data rather than from prose.

    A document that calls one of these the person's -- or one of the person's
    a machine fact -- now disagrees with a test, not just with another
    document.
    """
    assert _allocation_names() == _MACHINE_ANSWERS, (
        f"the machine-answered set changed: catalogue says "
        f"{sorted(_allocation_names())}, `execution/gpu.md` G2 says "
        f"{sorted(_MACHINE_ANSWERS)}.  If a setting genuinely moved sides, "
        f"amend G2 and the § 1 vocabulary table with it")


def test_the_gpu_flag_is_the_persons_choice_not_the_machines():
    """The specific claim `tuning.md` got backwards until 2026-08-23."""
    assert "use_gpu" not in _allocation_names(), (
        "the GPU flag is marked as machine-answered.  It is an ordinary "
        "explicit option with a real default, chosen at the Job Prep UI "
        "(`web/task-setup.md` § 6.2, user ruling 2026-08-16) -- nothing "
        "derives it from the machine (gpu.md G2)")


def test_the_flag_has_a_real_default_which_is_what_chosen_means():
    """A machine-answered setting has no default a person could mean; a chosen
    one does, and `false` is it."""
    item = _item("use_gpu")
    assert item.default is False
    assert item.group == "staging"


@pytest.mark.parametrize("name", sorted(_MACHINE_ANSWERS))
def test_each_machine_answered_setting_really_carries_the_flag(name):
    """The other direction: the contract's list is not allowed to name
    something the catalogue does not mark."""
    item = _item(name)
    if item is None:
        pytest.skip(f"{name} is not in the catalogue for any shipped engine")
    assert item.allocation, (
        f"`gpu.md` G2 lists {name} as machine-answered and the catalogue "
        f"does not mark it -- so a surface would offer it to a person")


def test_the_solver_decides_no_resource_and_no_environment():
    """**G3.**  `diag_algorithm` is the eigensolver and nothing else -- the
    packaged SIESTA runs both ELPA stages on CPU, measured, so only
    `Diag.ELPA.GPU` re-routes anything.

    Checked by SOURCE reach rather than behaviour: the claim is that the
    solver's name never arrives at the code that picks an environment, a
    gres or a partition.  A behavioural test would only cover the paths it
    happened to walk.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "molbuilder"
    resource_side = ["runwrap.py", "resolve.py", "jobset/submit.py",
                     "scheduler/admit.py", "scheduler/place.py",
                     "scheduler/emit.py"]
    offenders = [f for f in resource_side
                 if (root / f).is_file()
                 and "diag_algorithm" in (root / f).read_text()]
    assert not offenders, (
        f"`diag_algorithm` reaches {offenders} -- the solver is deciding a "
        f"resource or an environment.  That premise was measured FALSE on "
        f"2026-08-13 (the packaged SIESTA runs ELPA on CPU through ELSI) and "
        f"the deck-text scan it justified was deleted rather than replaced "
        f"(gpu.md G3)")
