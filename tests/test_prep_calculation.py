"""`prep`, entire — what only a described calculation that `molbuilder describe` did
not make can show.

Contract: ``docs/execution/project-layout.md`` § 2.3.1 (the five steps) ·
``docs/execution/job-contracts.md`` § 2.5a (the pseudopotentials) ·
`job-system.md` § 5.0 (the entry).

Every prep here goes through the one entry, `prep.prep_task` -- what
`jobset prep` and the Task setup tab call.  The rules a road row holds:

* the deck, its run script, the machine's copy, the plan -- every file the
  Task setup card names for a stage, `tests/data/the_catalogue.toml`;
* the stage named is the stage written, a template value no stage varies,
  the retry budget, the reporting policy, a stage the ladder does not hold
  -- `tests/data/prep_protocol.toml`;
* the deck agreeing with its launch -- every launch row that sends; the
  allocation reaching the header -- `tests/data/launch_values.toml`;
* a folder that is not a calculation -- `test_jobset.py`'s
  `test_cli_prep_is_described_only`.

What stays is API-level, and each says why.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.jobset.errors import PrepError
from molbuilder.jobset.model import Resources
from conftest import prep_the_group_of
from molbuilder.jobset.ladder import engine_ladder
from molbuilder.structure import Structure

from conftest import write_pseudos as _pseudos_for

#: BDT's first atoms -- three species, which the pseudopotential cases need:
#: a library that lacks one of them is the case.
_BDT = Structure(elements=["S", "C", "C", "H"],
                 positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                     [1.21, 0.0, 2.48], [2.15, 0.0, 1.94]]),
                 vacuum=(10.0, 10.0, 10.0))


pytestmark = pytest.mark.usefixtures("in_a_folder_of_its_own")


def _prep(calc, stage="coarse"):
    """`prep task <stage>`, through the one entry, the launch shape stated
    on the command line."""
    return prep_the_group_of(calc, stage,
                             allocation=Resources(mpi_np=8, cpus_per_task=1))


@pytest.fixture
def calc(tmp_path):
    """A described calculation whose pseudopotentials are in its own folder
    and whose description names no library -- which `molbuilder describe` does not
    make: it names one.

    How a shell enters an environment is read off the MACHINE RECORD --
    conftest's probed record carries it, so the calculation needs no config
    of its own.
    """
    src = tmp_path / "bdt.xyz"
    src.write_text(_BDT.to_xyz())
    dest = tmp_path / "calc"
    desc = D.build_description(
        _BDT, SiestaConfig(system_label="calc", mesh_cutoff=300.0),
        engine_ladder("siesta", "publishable"),
        engine="siesta",
        calculation="optimization",
        name="calc", source=str(src))
    D.write_description(desc, dest)
    _pseudos_for(dest, ["S", "C", "H"])
    return dest


# --------------------------------------------------------------------- #
#  The pseudopotentials checked are the ones the run will open           #
# --------------------------------------------------------------------- #

def test_pseudopotentials_beside_the_calculation_need_no_library(calc):
    """A calculation whose `.psml` files are in its own folder is not told
    `psml_lib` is unset.

    GOAL: prep used the folder's files -- the folder wins -- while the report
    beside the deck said "cfg.psml_lib is not set ... SIESTA will refuse to
    start": the settings gate was never told the folder (2026-09-25).
    CONTRACT: `job-contracts.md` § 2.5a (pseudopotentials already beside the
    calculation are used without this field); `pseudos.psml_sources` (the
    folder first, then the library -- one rule for prep and the gate).

    API-LEVEL: a description naming no library, its files put beside it --
    `molbuilder describe` names one, so the road does not make this calculation.
    """
    _prep(calc)
    report = (calc / "01_coarse" / "calc_01_coarse.validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


def test_the_folder_wins_over_a_library_that_lacks_a_species(
        isolated_projects_root):
    """A library that lacks a species the folder has does not stop prep.

    GOAL: prep's provider takes the folder's files and never asks the library
    for them, while the gate read the library alone -- so a calculation with
    every file beside it was refused over a library it did not need
    (2026-09-25).  CONTRACT: `pseudos.psml_sources` -- the folder first, then
    the library for what the folder lacks.

    API-LEVEL, for the reason above: the road's calculation is H2, one
    species, so no library can lack one the folder has.
    """
    tree = isolated_projects_root
    lib = tree / "pseudopotential"
    lib.mkdir()
    _pseudos_for(lib, ["S"])                       # the library: sulfur only
    src = tree / "bdt.xyz"
    src.write_text(_BDT.to_xyz())
    dest = tree / "P" / "optimization" / "calc"
    D.write_description(D.build_description(
        _BDT, SiestaConfig(system_label="calc", mesh_cutoff=300.0,
                           psml_lib="pseudopotential"),
        engine_ladder("siesta", "publishable"),
        engine="siesta",
        calculation="optimization",
        name="calc", source=str(src)),
        dest)
    _pseudos_for(dest, ["S", "C", "H"])            # the folder: all three
    _prep(dest)
    report = (dest / "01_coarse" / "calc_01_coarse.validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


# --------------------------------------------------------------------- #
#  Refusals                                                              #
# --------------------------------------------------------------------- #

def test_a_structure_that_changed_since_describing_is_refused(calc):
    """§ 6.3's witness earning its place: the description records a formula and
    an atom count, so building a *different* calculation under the same id is
    caught rather than discovered in the results.

    The mutated file is the CALCULATION'S OWN copy — `describe` copies the
    structure in, and that copy is what `prep` reads: the one `task.json`
    records.

    API-LEVEL: a refusal the road cannot reach -- nothing molbuilder does
    writes another structure over the calculation's copy; a hand does."""
    from molbuilder.task import of as task_of
    own = task_of(calc).structure.source
    (calc / own).write_text(
        Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                  vacuum=(10.0, 10.0, 10.0)).to_xyz())
    with pytest.raises(PrepError, match=r"structure has changed"):
        _prep(calc)


# --------------------------------------------------------------------- #
#  The pipeline log's resource line                                      #
# --------------------------------------------------------------------- #

def test_the_decision_log_prints_channel_names_readably():
    """The pipeline log's whole value is that a person can read it, which is what
    `_flat_resources`' own docstring says.

    Every field it renders was a scalar until `notify_channels` (2026-08-31),
    so the default `f"{v}"` put a Python repr into the file -- and a bare
    `()` for the answer that matters most, *send this calculation nowhere*.
    """
    from molbuilder.jobset.prep import _flat_resources

    assert "notify_channels=slack,lab" in _flat_resources(
        Resources(mpi_np=4, notify_channels=("slack", "lab")))
    assert "notify_channels=(none)" in _flat_resources(
        Resources(mpi_np=4, notify_channels=()))
    # absent stays absent: `None` means "not asked for" and is left out
    # rather than printed as a null.
    assert "notify_channels" not in _flat_resources(Resources(mpi_np=4))


def test_what_a_stage_was_prepared_with_leaves_run_settings_to_its_card(
        calc):
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
    _prep(calc)
    (settings,) = [v["settings"] for k, v in read(calc).items()
                   if k == "coarse"]
    assert "mesh_cutoff" in settings, sorted(settings)
    assert not set(settings) & run_settings("siesta"), sorted(
        set(settings) & run_settings("siesta"))


def test_a_calculation_this_molbuilder_prepared_is_current_to_migrate(calc):
    """`jobset migrate` on a calculation this molbuilder described and
    prepared says it is current and rewrites nothing (`migrate.CURRENT`).
    The silent failure: a key a release retired still listed among those an
    older plan lacks, so every current plan is "given" it, rewritten and
    kept beside a copy, on every run."""
    from molbuilder.jobset.migrate import migrate_state
    _prep(calc)
    before = sorted(p.relative_to(calc) for p in calc.rglob("*") if p.is_file())
    # NOTHING SAID IS NOTHING MIGRATED: the verb prints `migrate.CURRENT`
    # for an empty answer (`_cli.migrate_cmd`).
    said = migrate_state(calc)
    assert said == [], said
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


def test_a_template_that_does_not_read_is_refused_by_migrate(calc):
    """`job-request.md` § 8: a template that does not read is refused by name
    and nothing is written.  THE SILENT FAILURE: a Python traceback, its file
    unchanged and no way on said, as a retired word left by hand in a template
    of today's shape ended in."""
    from molbuilder.jobset.migrate import MigrateError, migrate_state
    template = next(calc.glob("*.template.toml"))
    text = template.read_text()
    item = "[item.spin_treatment]\n"
    assert item in text
    edited = text.replace(item, item + 'value = "polarized"\n')
    template.write_text(edited)
    with pytest.raises(MigrateError, match="does not read, so nothing was"):
        migrate_state(calc)
    assert template.read_text() == edited
