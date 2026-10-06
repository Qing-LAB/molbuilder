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
    """Prep here renders real decks and wrappers, which
    resolve config from cwd + HOME/XDG (H-8, 2026-08-13): unsandboxed,
    the file ran against the repo root's molbuilder.json, so the emitted
    text under test varied with the developer's config."""
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


# Retired 2026-10-05: four tests of the seed writer's own arguments.  Its
# ``stage_name`` and the ``# stage:`` line went (no reader read it); the
# convergence keys are the closed set the reader asks for
# (`trajectory_log.emitter._LEAF_KEYS`), so no key can carry whitespace; and
# the header and step 0 come from one writer, `header_and_preview`, whatever
# it is handed.


# --------------------------------------------------------------------- #
#  Prep: a SIESTA ladder seeds one molwatch log per stage                #
# --------------------------------------------------------------------- #


def _staged(xyz, tmp_path, strategy):
    """Describe a calculation and `prep` every enabled stage of *strategy*,
    each through the one entry and from the structure (`--cold`): what each
    stage's seeded log says is the question, not what it builds on.

    **Repointed 2026-08-11**, when `molbuilder fdf` was deleted, to one
    stage per `prep` call; **and 2026-10-06** to the entry, when the five
    steps stopped taking a call of their own -- in the hierarchy, where a
    stage can start from the structure (in the flat shape each builds on
    the run before it).
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
                            engine="siesta", shape="hierarchical", name="JOB",
                            source=str(xyz)),
        tmp_path)
    from conftest import write_pseudos
    write_pseudos(tmp_path, sorted(set(struct.elements)))
    for s in stages:
        if s.enabled:
            prep_stage(tmp_path, "run", s.name, cold=True,
                       allocation=Resources(mpi_np=4, cpus_per_task=1))
    return tmp_path


def _log(where, name):
    """The one progress log of that name the preps seeded -- in its stage's
    attempt, where its run will write it."""
    found = list(where.rglob(name))
    assert len(found) == 1, found
    return found[0].read_text()


# `test_multi_stage_cli_emits_per_stage_molwatch_logs` and
# `test_two_stage_strategy_emits_only_two_logs` retired 2026-10-06: the
# log each stage's prep seeds is there, named by the stage's token, as the
# card names it -- `tests/data/the_catalogue.toml` -- and a disabled stage
# is never prepped (`tests/data/prep_protocol.toml`).


def test_per_stage_molwatch_log_carries_stage_target(xyz, tmp_path):
    """Each per-stage log carries the stage's own force tolerance UNDER THE
    NAME ITS READER ASKS FOR -- so the watch-tab horizontal threshold matches
    the stage that's currently running.

    This asserted `max_force_ev_per_ang` until 2026-09-05, which is what the
    seeder wrote and what NOTHING read: the card asks for
    `max_force_tol_eV_per_A` (trajectory/core.js), the name the other two
    producers of this header already used. So the test passed, the docstring
    claimed the threshold was drawn, and the threshold was never drawn. A
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
    its reader asks for (`max_geom_iter`) rather than the `max_steps` this
    test used to pin -- see the sibling above for why that mattered. So the
    inspector can render the right "progress through the stage"
    indicator.  Defaults: stage1=600, stage2=200, stage3=100."""
    _staged(xyz, tmp_path, "vib-quality")
    text1 = _log(tmp_path, "JOB_01_coarse.molwatch.log")
    text2 = _log(tmp_path, "JOB_02_medium.molwatch.log")
    text3 = _log(tmp_path, "JOB_03_tight.molwatch.log")
    assert "# convergence.max_geom_iter: 600" in text1
    assert "# convergence.max_geom_iter: 200" in text2
    assert "# convergence.max_geom_iter: 100" in text3


def test_every_convergence_key_the_seeder_writes_is_one_the_card_reads():
    """THE BINDING, and the only thing that would have caught the 2026-09-05 bug.

    Two writers and one reader share the `# convergence.<key>:` header. The
    tests above pin the spelling on the WRITE side only, which is exactly how
    the seeder came to emit `max_force_ev_per_ang` / `max_steps` for three
    weeks: every one of them passed while the card they exist to feed drew
    nothing at all. A key nobody reads is a value the writer thinks it saved.

    So this asks the reader. `trajectory/core.js` names the keys it consumes
    as `ct.<key>`; anything the seeder emits outside that set is dead on
    arrival, whatever the header looks like.
    """
    import re
    from pathlib import Path

    core = (Path(__file__).resolve().parents[1]
            / "molbuilder/web/static/lib/trajectory/core.js").read_text(encoding="utf-8")
    reads = set(re.findall(r"\bct\.([A-Za-z_][A-Za-z0-9_]*)", core))
    assert len(reads) >= 5, (
        f"only {len(reads)} `ct.<key>` reads found in core.js -- the scan is "
        "blind, so the assertion below would pass vacuously")

    # What the SEEDER actually chooses, from the shipped mapping rather than
    # a list retyped here (retyping it is what the broken tests did).
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/prep.py").read_text(encoding="utf-8")
    block = src[src.index("    targets = {}\n    for key, attr in ("):]
    block = block[:block.index("))")]
    written = set(re.findall(r'\("([a-zA-Z_][a-zA-Z0-9_]*)",\s*"', block))
    assert len(written) >= 2, (
        f"found only {written} in the seeder's mapping -- repoint this test")

    orphans = sorted(written - reads)
    assert not orphans, (
        f"the stage seeder writes {orphans} into every per-stage "
        f".molwatch.log and `trajectory/core.js` reads none of them, so the "
        f"convergence card renders empty for staged runs. It reads: "
        f"{sorted(reads & {k for k in reads if 'max_' in k or 'tol' in k})}")
