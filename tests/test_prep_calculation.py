"""`prep`, entire -- what a calculation's pseudopotentials beside it, a record
prep wrote and the migration of a calculation prep prepared can show.

Contract: ``docs/execution/project-layout.md`` § 2.3.1 (the five steps) ·
``docs/execution/job-contracts.md`` § 2.5a (the pseudopotentials) ·
`job-system.md` § 5.0 (the entry).

Every calculation here is `molbuilder describe`'s and every prep `jobset
prep`'s (`support.road`).  The rules a road row holds:

* the deck, its run script, the machine's copy, the plan -- every file the
  Task setup card names for a stage, `tests/data/the_catalogue.toml`;
* the stage named is the stage written, a template value no stage varies,
  the retry budget, the reporting policy, a stage the ladder does not hold,
  a structure changed since it was described -- `tests/data/prep_protocol.toml`;
* the deck agreeing with its launch -- every launch row that sends; the
  allocation reaching the header -- `tests/data/launch_values.toml`;
* a folder that is not a calculation -- `test_jobset.py`'s
  `test_a_folder_that_is_no_calculation_is_refused_saying_so`.
"""
from __future__ import annotations

import pytest

from conftest import write_pseudos
from support.road import (CO2, H2, STRUCTURE, describe_calculation,
                          group_words, jobset, molbuilder, one_rank, stage_file)


def _prep(calc, stage="coarse"):
    """`jobset prep task` of the group ``stage`` is in, on this machine."""
    r = jobset("prep", "task", *group_words(calc, stage), "--bundle", calc,
               "--target", "this")
    assert r.exit_code == 0, r.output


def _with_its_files_beside_it(tmp_path, monkeypatch, *, structure=H2,
                              library=None):
    """`molbuilder describe` into a folder that already holds its
    pseudopotentials, as a person leaves them there first (`job-system.md`
    § 5.1: covered by *that folder or `--psml-lib`*) -- naming ``library``, the
    one holding only what it holds, or none.  The tree is the road's; the
    description's folder, its run card stated."""
    describe_calculation(tmp_path, monkeypatch, structure=structure)
    folder = tmp_path / "projects" / "P" / "optimization" / "calc"
    folder.mkdir()
    write_pseudos(folder, sorted(set(structure["elements"])))
    r = molbuilder("describe", "--structure", STRUCTURE,
                   "--bundle", "P/optimization/calc", "--engine", "siesta",
                   "--calculation", "optimization", "--name", "calc",
                   "--stage-strategy", "publishable",
                   *(("--psml-lib", library) if library else ()))
    assert r.exit_code == 0, r.output
    one_rank(folder)
    return folder


# --------------------------------------------------------------------- #
#  The pseudopotentials checked are the ones the run will open           #
# --------------------------------------------------------------------- #

def test_pseudopotentials_beside_the_calculation_need_no_library(tmp_path,
                                                                 monkeypatch):
    """A calculation whose `.psml` files are in its own folder is not told
    `psml_lib` is unset.

    GOAL: prep used the folder's files -- the folder wins -- while the report
    beside the deck said "cfg.psml_lib is not set ... SIESTA will refuse to
    start": the settings gate was never told the folder (2026-09-25).
    CONTRACT: `job-contracts.md` § 2.5a (pseudopotentials already beside the
    calculation are used without this field); `pseudos.psml_sources` (the
    folder first, then the library -- one rule for prep and the gate).
    """
    calc = _with_its_files_beside_it(tmp_path, monkeypatch)
    _prep(calc)
    report = stage_file(calc, "coarse", ".validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


def test_the_folder_wins_over_a_library_that_lacks_a_species(tmp_path,
                                                             monkeypatch):
    """A library that lacks a species the folder has does not stop prep.

    GOAL: prep's provider takes the folder's files and never asks the library
    for them, while the gate read the library alone -- so a calculation with
    every file beside it was refused over a library it did not need
    (2026-09-25).  CONTRACT: `pseudos.psml_sources` -- the folder first, then
    the library for what the folder lacks.
    """
    lacking = tmp_path / "projects" / "oxygen-only"
    lacking.mkdir(parents=True)
    write_pseudos(lacking, ["O"])                  # the library: oxygen only
    calc = _with_its_files_beside_it(tmp_path, monkeypatch, structure=CO2,
                                     library="oxygen-only")
    _prep(calc)
    report = stage_file(calc, "coarse", ".validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


# --------------------------------------------------------------------- #
#  What prep records, and what migrate then finds                        #
# --------------------------------------------------------------------- #

def test_what_a_stage_was_prepared_with_leaves_run_settings_to_its_card(
        tmp_path, monkeypatch):
    """What prep records a stage was prepared with (`prepared-with.json`,
    read through its door) holds the template's settings as resolved for
    the stage; a run setting is its run card's (`template.run_settings`),
    recorded there alone (`job-system.md`, *Changed since prepared*).
    THE SILENT FAILURE: a run setting recorded at its default among the
    settings -- a newer molbuilder's default then reads as the person's
    change to what the stage computes, and a launched group is told to
    start from the beginning."""
    from molbuilder.jobset.prepared_with import read
    from molbuilder.template import run_settings
    calc = describe_calculation(tmp_path, monkeypatch)
    _prep(calc)
    (settings,) = [v["settings"] for k, v in read(calc).items()
                   if k == "coarse"]
    assert "mesh_cutoff" in settings, sorted(settings)
    assert not set(settings) & run_settings("siesta"), sorted(
        set(settings) & run_settings("siesta"))


def test_a_calculation_this_molbuilder_prepared_is_current_to_migrate(
        tmp_path, monkeypatch):
    """`jobset migrate` on a calculation this molbuilder described and
    prepared says it is current and rewrites nothing (`migrate.CURRENT`).
    The silent failure: a key a release retired still listed among those an
    older plan lacks, so every current plan is "given" it, rewritten and
    kept beside a copy, on every run."""
    from molbuilder.jobset.migrate import CURRENT
    calc = describe_calculation(tmp_path, monkeypatch)
    _prep(calc)
    before = sorted(p.relative_to(calc) for p in calc.rglob("*") if p.is_file())
    r = jobset("migrate", "--bundle", calc)
    assert r.exit_code == 0, r.output
    assert CURRENT in r.output, r.output
    after = sorted(p.relative_to(calc) for p in calc.rglob("*") if p.is_file())
    assert after == before, set(after) ^ set(before)


@pytest.mark.parametrize("now, changed", [
    (["frame 1 of 2 · 0 V", "frame 2 of 2 · 0 V"], False),
    (["frame 1 of 2 · 0 V", "frame 2 of 2 · 0 V", "frame 1 of 2 · 0.2 V"], True),
], ids=["only reworded", "a voltage added"])
def test_a_record_from_before_it_held_folders_still_compares_its_points(
        now, changed):
    """`job-system.md`, *Changed since prepared*: a stage prepared before 2.67.2
    holds its points in the words of then, as a real frame set's record does.
    THE SILENT FAILURE: a voltage added, no change said, and the launch running
    the old points; or every such frame set called changed for a rewording."""
    from molbuilder.jobset.prepared_with import POINTS, differences
    was = {"points": ["frame index 0 (of 2) · 0 V", "frame index 1 (of 2) · 0 V"]}
    found = differences(was, {"points": now, "point_folders": ["a"]})
    assert [d.kind for d in found] == ([POINTS] if changed else []), found


def test_a_template_that_does_not_read_is_refused_by_migrate(tmp_path,
                                                             monkeypatch):
    """`job-request.md` § 8: a template that does not read is refused by name
    and nothing is written.  THE SILENT FAILURE: a Python traceback, its file
    unchanged and no way on said, as a retired word left by hand in a template
    of today's shape ended in."""
    from molbuilder.calculation import open as open_calculation
    calc = describe_calculation(tmp_path, monkeypatch)
    template = open_calculation(calc).template.path
    text = template.read_text()
    item = "[item.spin_treatment]\n"
    assert item in text
    edited = text.replace(item, item + 'value = "polarized"\n')
    template.write_text(edited)
    r = jobset("migrate", "--bundle", calc)
    assert r.exit_code != 0, r.output
    assert "does not read, so nothing was" in r.output, r.output
    assert template.read_text() == edited
