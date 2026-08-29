"""``jobset init`` — the portable description, and what it must never write.

Contract: ``docs/execution/architecture.md`` § 4 (the route: *ask → check →
write*, **floor 2 only**) · ``docs/execution/job-system.md`` § 5.1 (the command)
· ``docs/execution/project-layout.md`` § 2.1 (what makes the folder portable) ·
``docs/engines/stages.md`` § 6.3 (the structure is a reference), § 6.5 (a
calculation with one parameter set), § 6.6 (the split preflight).

Written with the verb, 2026-08-11, from the contract rather than from the
implementation — the properties below are the ones the design would be wrong
without, not the ones the code happens to have.
"""
from __future__ import annotations

import json
import tomllib

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure
from molbuilder.task import Stage, read_task
from molbuilder.template import read_template


@pytest.fixture
def struct() -> Structure:
    return Structure(elements=["S", "C", "C", "H"],
                     positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                         [1.21, 0.0, 2.48], [2.15, 0.0, 1.94]]),
                     vacuum=(10.0, 10.0, 10.0))


@pytest.fixture
def cfg() -> SiestaConfig:
    return SiestaConfig(system_label="relax", mesh_cutoff=300.0)


#: § 6.5 (2026-08-16): a job always has at least one stage, so the helper's
#: default is ONE stage carrying no overrides -- the calculation that is just
#: the template.  It defaulted to ``()`` while a stage-less description was a
#: legal shape; that shape is gone, and an empty ladder is now refused.
_ONE_STAGE = (Stage(name="coarse", enabled=True, overrides={}),)


def _describe(struct, cfg, stages=_ONE_STAGE, *, shape="hierarchical",
              name="relax", source="structures/bdt.xyz"):
    return D.build_description(struct, cfg, stages, engine="siesta",
                               shape=shape, name=name, source=source)


# --------------------------------------------------------------------- #
#  Floor 2 only — what it must NOT write                                #
# --------------------------------------------------------------------- #

def test_describing_writes_no_deck(struct, cfg, tmp_path):
    """**The route's whole definition.**  A deck carries values that depend on
    how it will be launched — ``BlockSize`` above all — so rendering belongs to
    ``prep`` on the machine that will run it.  A ``describe`` that emitted a
    deck would put step 3 before step 1 (`project-layout.md` § 2.3.1)."""
    written = D.write_description(_describe(struct, cfg,
                                            default_siesta_stages("publishable")),
                                  tmp_path / "calc")
    assert not [p for p in written if p.suffix == ".fdf"]
    assert not list((tmp_path / "calc").glob("*.fdf"))


def test_it_writes_exactly_the_template_and_the_description(struct, cfg, tmp_path):
    written = D.write_description(_describe(struct, cfg), tmp_path / "calc")
    assert sorted(p.name for p in written) == ["relax.template.toml",
                                               "task.json"]


def test_the_description_names_no_machine(struct, cfg, tmp_path):
    """`project-layout.md` § 2.1: *"none of them mention a machine"*, which is
    what makes the folder mean the same thing wherever you copy it.

    ``continue_retries`` is deliberately **not** counted: it is a retry
    *policy*, not a machine fact, and floor 2 is the right home for it.  That
    distinction was got wrong once and is pinned here so it is not got wrong
    again.
    """
    D.write_description(_describe(struct, cfg), tmp_path / "calc")
    raw = tomllib.loads((tmp_path / "calc" / "relax.template.toml").read_text())
    # RESTATED AT @2 (§ 6.4).  The rule is "name no machine", and what
    # names a machine is an ANSWER, not a question.  These items ARE now
    # declared -- a surface must be able to ask for ranks, and the wrapper
    # writer must know to look -- but they carry no `value`, so the folder
    # still means the same thing wherever it is copied.
    machine = ("mpi_np", "omp_threads", "max_memory_mb")
    answered = [n for n in machine
                if n in raw["item"] and "value" in raw["item"][n]]
    assert answered == [], (
        f"floor 2 is naming a machine: {answered} carry a value. The item "
        f"declares the QUESTION; `prep` states the answer on the machine "
        f"that will run it (engines/template.md § 2, § 6.4).")
    # ...and the questions must be present, or no surface could ask them.
    for n in machine:
        assert n in raw["item"], f"{n} is not declared; a surface cannot ask for it"


# --------------------------------------------------------------------- #
#  What it does write                                                   #
# --------------------------------------------------------------------- #

def test_the_structure_is_a_reference_and_a_witness_never_a_copy(struct, cfg,
                                                                 tmp_path):
    """§ 6.3.  The witness is what lets a description opened against a
    structure that has since changed *say so*, instead of silently building a
    different calculation under the same id."""
    D.write_description(_describe(struct, cfg, source="structures/bdt.xyz"),
                        tmp_path / "calc")
    task = read_task(tmp_path / "calc" / "task.json")
    assert task.structure.source == "structures/bdt.xyz"
    assert task.structure.formula == struct.formula
    assert task.structure.atoms == struct.n_atoms
    # The source names a path that does not exist HERE, so there is
    # nothing to copy -- the reference still records.  (U20: this line
    # used to read as a NO-COPY pin, the pre-M9 contract, green only by
    # this fixture accident; the copy half is its own test below.)
    assert not list((tmp_path / "calc").glob("*.xyz"))


def test_an_existing_structure_file_travels_with_the_calculation(
        struct, cfg, tmp_path):
    """M9 / stages.md § 6.3 (amended U19): the data FILE is copied beside
    task.json, exactly like the pseudos -- what stays forbidden is
    coordinates embedded IN task.json.  This is the pin test_describe
    lacked while test_prep_calculation asserted the copy: the two files
    asserted OPPOSITE contracts, both green, because this fixture's
    source never existed on disk."""
    src = tmp_path / "bdt.xyz"
    src.write_text(struct.to_xyz())
    D.write_description(_describe(struct, cfg, source=str(src)),
                        tmp_path / "calc")
    # The travelling copy carries the ``.source`` mark and the written
    # task.json records THAT name (`job-contracts.md` 6.3, 2026-08-19):
    # identities are dot-free, so no engine output can take the copy's
    # name, and the folder is self-contained -- prep resolves the local
    # marked file, never this machine's original path.
    copied = tmp_path / "calc" / "bdt.source.xyz"
    assert copied.is_file()
    assert copied.read_text() == src.read_text()
    task = read_task(tmp_path / "calc" / "task.json")
    assert task.structure.source == "bdt.source.xyz"


def test_a_described_modification_travels_as_the_codec_pair(
        struct, cfg, tmp_path):
    """The ``--vacuum`` fix (2026-08-12): describe can MODIFY the structure
    it was handed, and those facts live in metadata a bare .xyz has nowhere
    to put -- so the raw copy silently dropped them and prep rendered the
    3 A-default cell over an explicit scientific choice.  With ``struct``
    passed, a structure carrying metadata travels as the codec pair, and
    prep's own loader reads the vacuum back."""
    src = tmp_path / "bdt.xyz"
    src.write_text(struct.to_xyz())     # bare xyz: no vacuum in these bytes
    D.write_description(_describe(struct, cfg, source=str(src)),
                        tmp_path / "calc", struct=struct)
    assert (tmp_path / "calc" / "bdt.source.molstruct.json").is_file(), \
        "the metadata sidecar did not travel"
    from molbuilder.jobset.prep import _structure_for
    task = read_task(tmp_path / "calc" / "task.json")
    reloaded = _structure_for(task, tmp_path / "calc")
    assert reloaded.vacuum == (10.0, 10.0, 10.0), (
        "prep reloads the structure without the vacuum the description "
        "was built with -- the deck would get the default cell")


def test_the_vacuum_flag_reaches_the_travelling_structure(tmp_path):
    """The CLI half of the same fix: ``describe --vacuum 8`` on a bare
    XYZ must put 8 A on the structure that travels, not only on the one
    in memory."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.workingcopy_structure import StructureCodec
    import os
    # `init` cites both the structure and the calculation FROM THE PROJECTS
    # ROOT (job-contracts.md 2.5b), so the test says where its tree is --
    # the same thing a user does when calculations live on scratch.
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    xyz = tree / "P" / "structure" / "h2.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    os.environ[PROJECTS_ROOT_ENV] = str(tree)
    try:
        res = CliRunner().invoke(jobset_group, [
            "init", "--structure", "P/structure/h2.xyz",
            "--bundle", "P/optimization/calc",
            "--shape", "hierarchical", "--vacuum", "8", "--name", "JOB"])
    finally:
        os.environ.pop(PROJECTS_ROOT_ENV, None)
    assert res.exit_code == 0, res.output
    reloaded = StructureCodec().load(
        tree / "P" / "optimization" / "calc" / "h2.source.xyz")
    assert reloaded.vacuum == (8.0, 8.0, 8.0)


def test_the_calculation_key_is_absent_is_a_state(struct, cfg, tmp_path):
    """U0 (warm-files § 4.2a, 2026-08-13): `calculation` keys the engine's
    warm-file vocabulary.  Absent IS "optimization" (the § 6.5 pattern),
    so an optimization description writes no key; a non-default type
    round-trips; a shape-violating value is refused naming the rule.
    Membership (does the engine have this section?) is NOT checked here —
    that is the rules file's question, answered where it is read."""
    D.write_description(_describe(struct, cfg), tmp_path / "calc")
    raw = json.loads((tmp_path / "calc" / "task.json").read_text())
    assert "calculation" not in raw
    assert read_task(tmp_path / "calc" / "task.json").calculation == \
        "optimization"
    # ("vibration" stands in for any non-default kind; "transport" no
    # longer can -- since 2026-08-28 it is the COMPOSITE with required
    # slots and no structure block, guarded in test_transport_task.py.)
    desc = D.build_description(struct, cfg, _ONE_STAGE, engine="siesta",
                               shape="hierarchical", name="relax",
                               source="structures/bdt.xyz",
                               calculation="vibration")
    D.write_description(desc, tmp_path / "calc2")
    raw = json.loads((tmp_path / "calc2" / "task.json").read_text())
    assert raw["calculation"] == "vibration"
    assert read_task(tmp_path / "calc2" / "task.json").calculation == \
        "vibration"
    with pytest.raises(Exception, match=r"calculation.*A-Za-z0-9_"):
        D.build_description(struct, cfg, _ONE_STAGE, engine="siesta",
                            shape="hierarchical", name="relax",
                            source="structures/bdt.xyz",
                            calculation="no-hyphens!")


def test_a_ladder_becomes_stages_and_varies(struct, cfg, tmp_path):
    D.write_description(_describe(struct, cfg,
                                  default_siesta_stages("publishable")),
                        tmp_path / "calc")
    task = read_task(tmp_path / "calc" / "task.json")
    assert [s.name for s in task.stages] == ["coarse", "medium", "tight"]
    assert "relax_type" in task.varies


def test_an_empty_ladder_is_refused(struct, cfg):
    """§ 6.5 (2026-08-16): a job always has at least one stage.

    This asserted the opposite until then — that an empty ladder wrote a
    description with neither key, a stage-less shape meaning "just the
    template". That shape produced artifacts with NO stage token, and adding a
    second stage later left the first run belonging to no token at all. One
    shape removes the transition; an empty ladder is now refused, and the
    refusal names the fix."""
    with pytest.raises(D.DescribeError, match="at least one stage"):
        _describe(struct, cfg, ())


def test_the_label_and_the_id_derive_from_the_name(struct, cfg):
    """§ 3 rule 1: it happens once.  The label is not stored and the id is,
    and ``Task.__post_init__`` proves the two against each other — so a
    description that got here has already been checked."""
    desc = _describe(struct, cfg, name="bdt-relax")
    assert desc.label == "bdt-relax"
    assert desc.task.run.id == f"bdt-relax_{struct.formula}"


# --------------------------------------------------------------------- #
#  Ask → check → write, and the order is the point                      #
# --------------------------------------------------------------------- #

def test_a_bad_stage_name_is_refused_before_anything_is_written(struct, cfg,
                                                                tmp_path):
    """Names are validated *here, on your laptop, not on the cluster*.  A
    stage name becomes a filename and a shell word, so the set is narrow."""
    from molbuilder.task import Stage
    with pytest.raises(ValueError):
        _describe(struct, cfg, (Stage(name="not a name"),))
    assert not (tmp_path / "calc").exists()


def test_an_override_the_schema_does_not_know_is_refused_by_name(struct, cfg):
    """§ 6.6's third check, and it is the existing preflight doing it rather
    than a second implementation."""
    from molbuilder.task import Stage
    with pytest.raises(D.DescribeError, match=r"mesh_cutof"):
        _describe(struct, cfg,
                  (Stage(name="tight", overrides={"mesh_cutof": 300.0}),))


def test_a_value_outside_its_bounds_is_refused(struct, cfg):
    from molbuilder.task import Stage
    with pytest.raises(D.DescribeError):
        _describe(struct, cfg,
                  (Stage(name="tight", overrides={"mesh_cutoff": 1e9}),))


def test_a_failure_while_writing_publishes_nothing(struct, cfg, tmp_path,
                                                   monkeypatch):
    """*"Describing a calculation writes every file or none."*

    Mutation-tested by construction: the failure is injected, so this proves
    the staging directory is really doing the work rather than the happy path
    merely never failing.
    """
    desc = _describe(struct, cfg)
    real = D.Description.files
    monkeypatch.setattr(D.Description, "files",
                        lambda self: {**real(self), "task.json": None})
    with pytest.raises(TypeError):
        D.write_description(desc, tmp_path / "calc")
    assert list((tmp_path / "calc").iterdir()) == []
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".calc.")]


# --------------------------------------------------------------------- #
#  The name — inherited from the retired test_cli_system_label.py        #
# --------------------------------------------------------------------- #
#  That file's nine tests all went through `molbuilder fdf`, which was
#  deleted 2026-08-11.  Two of its properties are about the NAME rather than
#  about that verb, so they move here rather than going with it.

def test_a_name_that_cannot_normalise_is_a_clean_refusal(struct, cfg):
    """A label becomes a filename and a shell word, so the character set is
    narrow.  A name outside it must be refused by name, not crash."""
    with pytest.raises((ValueError, D.DescribeError)):
        _describe(struct, cfg, name="Über")


def test_a_refused_name_writes_nothing(struct, cfg, tmp_path):
    """The refusal happens while nothing has been written — *ask → check →
    write*, and the check is before the write rather than beside it."""
    with pytest.raises((ValueError, D.DescribeError)):
        D.write_description(_describe(struct, cfg, name="Über"),
                            tmp_path / "calc")
    assert not (tmp_path / "calc").exists()


# --------------------------------------------------------------------- #
#  The two surfaces write the same bytes                                #
# --------------------------------------------------------------------- #

def test_the_producer_is_pure_so_two_surfaces_cannot_disagree(struct, cfg):
    """§ 4.1's Promotion A: one producer, two writers.  A browser that built
    its own description would be a second place for the bytes to differ, and
    § 6.4 is the rule that they must not."""
    a = _describe(struct, cfg, default_siesta_stages("publishable"))
    b = _describe(struct, cfg, default_siesta_stages("publishable"))
    assert a.files() == b.files()


def test_psml_lib_bare_name_resolves_via_the_tree_not_the_cwd(tmp_path):
    """2026-08-28: --psml-lib carried a click-level exists check that
    validated against the WORKING DIRECTORY, while the resolver's rule
    (job-contracts 2.5a) says a bare name means the projects tree the
    calculation lives in.  From anywhere but inside the tree the two
    validators refused each other's accepted spelling -- click rejecting
    the bare in-tree name, the resolver rejecting the cwd-relative one
    click demanded.  One fact, one door: the click check is gone and the
    resolver decides, so the bare spelling works from any directory."""
    import os

    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    (tree / "pseudopotential" / "H.psml").write_text("<psml/>\n")
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\n\nH 0 0 0\nH 0 0 0.74\n")
    os.environ[PROJECTS_ROOT_ENV] = str(tree)
    try:
        # CliRunner's cwd is the test runner's -- ./pseudopotential does
        # NOT exist here, which is exactly the papercut's shape.
        res = CliRunner().invoke(jobset_group, [
            "init", "--structure", "P/structure/h2.xyz",
            "--bundle", "P/optimization/calc",
            "--shape", "hierarchical", "--vacuum", "8", "--name", "JOB",
            "--psml-lib", "pseudopotential"])
    finally:
        os.environ.pop(PROJECTS_ROOT_ENV, None)
    assert res.exit_code == 0, res.output
    assert (tree / "P" / "optimization" / "calc" / "H.psml").exists(), (
        "the bare name must resolve against the tree and the pseudo "
        "must travel with the calculation")
