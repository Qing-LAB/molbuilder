"""`prep`, entire — what only a described calculation that `jobset init` did
not make can show.

Contract: ``docs/execution/project-layout.md`` § 2.3.1 (the five steps) ·
``docs/execution/job-contracts.md`` § 2.5a (the pseudopotentials) ·
`job-system.md` § 5.0 (the entry).

Every prep here goes through the one entry, `prep.prep_stage` -- what
`jobset prep` and the Task setup tab call.  **These tests drove
`prep_calculation` directly until 2026-10-06**, the five steps below the
entry, which then read and resolved the calculation on its own; that door
went (`job-system.md` § 5.0), and with it the tests whose rule a road row
holds:

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
from molbuilder.jobset.prep import prep_stage
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure

from conftest import write_pseudos as _pseudos_for

#: BDT's first atoms -- three species, which the pseudopotential cases need:
#: a library that lacks one of them is the case.
_BDT = Structure(elements=["S", "C", "C", "H"],
                 positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                     [1.21, 0.0, 2.48], [2.15, 0.0, 1.94]]),
                 vacuum=(10.0, 10.0, 10.0))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """The rest of the sandbox (B-9, 2026-08-13): the file ran with cwd =
    repo root, so the repo's molbuilder.json kept folding into every wrapper
    under test."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


def _prep(calc, stage="coarse"):
    """`prep run <stage>`, through the one entry, the launch shape stated
    on the command line."""
    return prep_stage(calc, "run", stage,
                      allocation=Resources(mpi_np=8, cpus_per_task=1))


@pytest.fixture
def calc(tmp_path):
    """A described calculation whose pseudopotentials are in its own folder
    and whose description names no library -- which `jobset init` does not
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
        default_siesta_stages("publishable"),
        engine="siesta", shape="hierarchical", name="calc", source=str(src))
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
    `jobset init` names one, so the road does not make this calculation.
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
        default_siesta_stages("publishable"),
        engine="siesta", shape="hierarchical", name="calc", source=str(src)),
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
    structure in since 2026-08-12 (M9's walk found nothing made "beside the
    calculation first" true), and that copy is what `prep` reads: the one
    `task.json` records, named for the label since 2026-10-04 (plan D20).

    API-LEVEL: a refusal the road cannot reach -- nothing molbuilder does
    writes another structure over the calculation's copy; a hand does."""
    from molbuilder.task import read_task
    own = read_task(calc / "task.json").structure.source
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
    """The ledger's whole value is that a person can read it, which is what
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
