"""Tests for the per-stage convergence-target extension of the
molwatch log format (#542 / C1.4).

Pins: prepping a SIESTA ladder seeds one ``<label>_<NN>_<name>.molwatch.log``
per enabled stage, each carrying that stage's own targets as
``# convergence.<key>:`` lines under the names the reader asks for.
"""
from __future__ import annotations

import textwrap

import pytest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """Prep here renders real decks and wrappers, which resolve config
    from the config root (H-8, 2026-08-13: unsandboxed, the emitted text
    under test varied with the developer's config)."""
    cwd = tmp_path_factory.mktemp("cwd")
    monkeypatch.chdir(cwd)
    # THE SANDBOX IS THE CONFIG ROOT -- and holds no config: how a shell
    # enters an environment is the machine record's (`configuration.md`
    # § 5 M-1), written below.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cwd))
    # AND THE RECORD GOES IN LAST, after every scope-moving call above:
    # it is written where `machine_scope_path()` resolves NOW, and prep
    # refuses without one (`running-a-job.md` § 3.1).
    from conftest import write_machine_record
    write_machine_record()


# A 3-D (non-linear) molecule so the derived vacuum cell isn't degenerate at
# the default vacuum=0 (structure-periodicity.md).  These prep tests exercise
# stage/log mechanics, not geometry, so methane is fine.
_XYZ = textwrap.dedent("""\
    5

    C  0.000  0.000  0.000
    H  0.629  0.629  0.629
    H -0.629 -0.629  0.629
    H -0.629  0.629 -0.629
    H  0.629 -0.629 -0.629
""")


@pytest.fixture
def xyz(tmp_path):
    p = tmp_path / "h2.xyz"
    p.write_text(_XYZ)
    return p


# --------------------------------------------------------------------- #
#  Prep: a SIESTA ladder seeds one molwatch log per stage                #
# --------------------------------------------------------------------- #


def _staged(xyz, tmp_path, strategy):
    """Describe a calculation and `prep` every enabled stage of *strategy*,
    each through the one entry and from the structure (`--cold`): what each
    stage's seeded log says is the question, not what it builds on.  In the
    hierarchy, where a stage can start from the structure (in the flat shape
    each builds on the run before it).
    """
    from molbuilder import describe as D
    from molbuilder.workingcopy_structure import StructureCodec
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.prep import prep_stage
    from molbuilder.siesta.stages import default_siesta_stages

    struct = StructureCodec().load(xyz)
    stages = default_siesta_stages(strategy)
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"), stages,
                            engine="siesta",
                            calculation="optimization",
                            shape="hierarchical", name="JOB",
                            source=str(xyz)),
        tmp_path)
    from conftest import write_pseudos
    write_pseudos(tmp_path, sorted(set(struct.elements)))
    for s in stages:
        prep_stage(tmp_path, "task", s.name, cold=True,
                   allocation=Resources(mpi_np=4, cpus_per_task=1))
    return tmp_path


def _log(where, name):
    """The one progress log of that name the preps seeded -- in its stage's
    attempt, where its run will write it."""
    found = list(where.rglob(name))
    assert len(found) == 1, found
    return found[0].read_text()


def test_per_stage_molwatch_log_carries_stage_target(xyz, tmp_path):
    """Each per-stage log carries the stage's own force tolerance UNDER THE
    NAME ITS READER ASKS FOR -- so the watch-tab horizontal threshold matches
    the stage that's currently running.

    The card asks for `max_force_tol_eV_per_A` (trajectory/core.js): a
    spelling this test invented is not a contract; the reader's vocabulary is.

    Default ladder (siesta/stages.py::default_siesta_stages) per stage:
      01_coarse: 0.05 (loose preopt)
      02_medium: 0.04 (publishable)
      03_tight:  0.01 (crystal-tight)
    """
    _staged(xyz, tmp_path, "vib-quality")
    text1 = _log(tmp_path, "JOB_01_coarse.molwatch.log")
    text2 = _log(tmp_path, "JOB_02_medium.molwatch.log")
    text3 = _log(tmp_path, "JOB_03_tight.molwatch.log")
    assert "# convergence.max_force_tol_eV_per_A: 0.05" in text1
    assert "# convergence.max_force_tol_eV_per_A: 0.04" in text2
    assert "# convergence.max_force_tol_eV_per_A: 0.01" in text3


def test_per_stage_molwatch_log_carries_its_geometry_step_cap(xyz, tmp_path):
    """Each per-stage log carries its own geometry-step cap, under the name
    its reader asks for (`max_geom_iter`), so the inspector can render the
    right "progress through the stage" indicator.  Defaults: coarse=600,
    medium=200, tight=100."""
    _staged(xyz, tmp_path, "vib-quality")
    text1 = _log(tmp_path, "JOB_01_coarse.molwatch.log")
    text2 = _log(tmp_path, "JOB_02_medium.molwatch.log")
    text3 = _log(tmp_path, "JOB_03_tight.molwatch.log")
    assert "# convergence.max_geom_iter: 600" in text1
    assert "# convergence.max_geom_iter: 200" in text2
    assert "# convergence.max_geom_iter: 100" in text3
