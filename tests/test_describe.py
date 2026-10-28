"""``molbuilder describe`` — the portable description, and what it must never write.

Contract: ``docs/execution/architecture.md`` § 4 (the route: *ask → check →
write*, **floor 2 only**) · ``docs/execution/job-system.md`` § 5.1 (the command)
· ``docs/execution/project-layout.md`` § 2.1 (what makes the folder portable) ·
``docs/engines/stages.md`` § 6.3 (the structure is a reference), § 6.5 (a
calculation with one parameter set), § 6.6 (the split preflight).

Written from the contract rather than from the
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
from molbuilder.task import Stage, of as task_of
from molbuilder.template import read_template
from support.road import molbuilder
from molbuilder.jobset._cli import describe_cmd


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
#: the template.
_ONE_STAGE = (Stage(name="coarse", overrides={}),)


def _describe(struct, cfg, stages=_ONE_STAGE, *, shape="hierarchical",
              name="relax", source="structures/bdt.xyz"):
    return D.build_description(struct, cfg, stages, engine="siesta",
                               calculation="optimization",
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


def test_it_writes_exactly_the_template_the_description_and_its_request(
        struct, cfg, tmp_path):
    """`execution/job-request.md` § 9: describing writes the description and
    its job request -- a calculation's folder holds both (§ 7)."""
    written = D.write_description(_describe(struct, cfg), tmp_path / "calc")
    assert sorted(p.name for p in written) == ["job-request.json",
                                               "relax.template.toml",
                                               "task.json"]


def test_the_description_names_no_machine(struct, cfg, tmp_path):
    """`project-layout.md` § 2.1: *"none of them mention a machine"*, which is
    what makes the folder mean the same thing wherever you copy it -- and
    `engines/template.md` § 2: **no run setting is declared in a template,
    valueless or not** -- the machine's answers and a person's alike are the
    job request's.  A surface asks the catalogue which questions a run has.

    MUTATION THIS MUST FAIL AGAINST: the writer emitting run settings
    valueless again (2.27.x's slot, which migrate's refusal by content and
    its move by version disagreed about)."""
    from molbuilder.template import catalogue, is_run_setting, one
    D.write_description(_describe(struct, cfg), tmp_path / "calc")
    raw = tomllib.loads((tmp_path / "calc" / "relax.template.toml").read_text())
    held = [n for n in raw["item"]
            if is_run_setting(one(catalogue(), n, engine="siesta"))]
    assert held == [], (
        f"the template holds run settings {held} -- they are the job "
        f"request's (engines/template.md § 2)")


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
    task = task_of(tmp_path / "calc")
    assert task.structure.source == "structures/bdt.xyz"
    assert task.structure.formula == struct.formula
    assert task.structure.atoms == struct.n_atoms
    # The source names a path that does not exist HERE, so there is
    # nothing to copy -- the reference still records.
    assert not list((tmp_path / "calc").glob("*.xyz"))


def test_the_structure_travels_as_the_calculations_own_pair(tmp_path,
                                                            monkeypatch):
    """`molbuilder describe` writes the structure it describes as the calculation's
    own pair, named for the LABEL -- ``<label>.source.xyz`` and its sidecar,
    the catalogue's name and the hand-over's (`job-contracts.md` § 6.3,
    `StructureCodec.source_files`) -- ``task.json`` records that name, and
    the ``--vacuum 8`` it was described with travels in the sidecar, so
    the loader prep reads the pair with reads it back (the fix of 2026-08-12).

    The structure FILE is ``h2.xyz`` and the label ``JOB``: init named the
    pair ``h2.source.*`` until 2026-10-04 (plan D20)."""
    from click.testing import CliRunner
    from molbuilder.projects import PROJECTS_ROOT_ENV
    # `init` cites both the structure and the calculation FROM THE PROJECTS
    # ROOT (job-contracts.md 2.5b), so the test says where its tree is --
    # the same thing a user does when calculations live on scratch.
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    xyz = tree / "P" / "structure" / "h2.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    res = CliRunner().invoke(describe_cmd, [
        "--structure", "P/structure/h2.xyz",
        "--bundle", "P/optimization/calc", "--engine", "siesta",
        "--calculation", "optimization",
        "--shape", "hierarchical", "--vacuum", "8", "--name", "JOB"])
    assert res.exit_code == 0, res.output
    calc = tree / "P" / "optimization" / "calc"
    assert sorted(p.name for p in calc.glob("*.source.*")) == [
        "JOB.source.molstruct.json", "JOB.source.xyz"]
    task = task_of(calc)
    assert task.structure.source == "JOB.source.xyz"
    from molbuilder.workingcopy_structure import StructureCodec
    assert StructureCodec().load(calc / task.structure.source).vacuum == (
        8.0, 8.0, 8.0)


def test_every_description_states_its_kind(struct, cfg, tmp_path):
    """U0 (warm-files § 4.2a, 2026-08-13): `calculation` keys the engine's
    warm-file vocabulary, and every description states it, an
    optimization's included (`job-contracts.md` § 6.1; W57 R10, 2026-10-06
    -- an optimization's was left out until then); a shape-violating value
    is refused naming the rule.  Membership (does the engine have this
    section?) is NOT checked here — that is the rules file's question,
    answered where it is read."""
    D.write_description(_describe(struct, cfg), tmp_path / "calc")
    raw = json.loads((tmp_path / "calc" / "task.json").read_text())
    assert raw["calculation"] == "optimization"
    assert task_of(tmp_path / "calc").calculation == \
        "optimization"
    # ("vibration" stands in for any non-default kind; "transport" is the
    # COMPOSITE with required slots and no structure block, guarded in
    # test_transport_task.py.)
    desc = D.build_description(struct, cfg, _ONE_STAGE, engine="siesta",
                               shape="hierarchical", name="relax",
                               source="structures/bdt.xyz",
                               calculation="vibration")
    D.write_description(desc, tmp_path / "calc2")
    raw = json.loads((tmp_path / "calc2" / "task.json").read_text())
    assert raw["calculation"] == "vibration"
    assert task_of(tmp_path / "calc2").calculation == \
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
    task = task_of(tmp_path / "calc")
    assert [s.name for s in task.stages] == ["coarse", "medium"]
    assert "relax_type" in task.varies


def test_an_empty_ladder_is_refused(struct, cfg):
    """§ 6.5 (2026-08-16): a job always has at least one stage.

    A stage-less shape produced artifacts with NO stage token, and adding a
    second stage later left the first run belonging to no token at all. One
    shape removes the transition; an empty ladder is refused, and the
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


def test_an_override_the_schema_does_not_know_is_refused_by_name(struct, cfg):
    """§ 6.6's third check, and it is the existing preflight doing it rather
    than a second implementation."""
    from molbuilder.task import Stage
    with pytest.raises(D.DescribeError, match=r"mesh_cutof"):
        _describe(struct, cfg,
                  (Stage(name="tight", overrides={"mesh_cutof": 300.0}),))


def test_a_value_past_its_hard_limit_is_refused(struct, cfg):
    """A stage's value past its item's hard limit (`above`,
    `engines/template.md` § 5.3) is refused where the description is
    written, with the reason; a value merely outside the recommended range
    describes.  API-LEVEL, because the road cannot reach it: `molbuilder describe`
    describes a non-transport calculation from its ladder presets alone, so
    no person's value reaches this door but through the library."""
    from molbuilder.task import Stage
    with pytest.raises(D.DescribeError, match=r"k-point count"):
        _describe(struct, cfg,
                  (Stage(name="tight", overrides={"kgrid": [0, 4, 4]}),))
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
#  The name                                                             #
# --------------------------------------------------------------------- #

def test_a_name_that_cannot_normalise_is_a_clean_refusal(struct, cfg):
    """A label becomes a filename and a shell word, so the character set is
    narrow.  A name outside it must be refused by name, not crash."""
    with pytest.raises((ValueError, D.DescribeError)):
        _describe(struct, cfg, name="Über")


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


def test_psml_lib_bare_name_resolves_via_the_tree_not_the_cwd(tmp_path,
                                                               monkeypatch):
    """2026-08-28: --psml-lib carried a click-level exists check that
    validated against the WORKING DIRECTORY, while the resolver's rule
    (job-contracts 2.5a) says a bare name means the projects tree the
    calculation lives in.  From anywhere but inside the tree the two
    validators refused each other's accepted spelling -- click rejecting
    the bare in-tree name, the resolver rejecting the cwd-relative one
    click demanded.  One fact, one door: the click check is gone and the
    resolver decides, so the bare spelling works from any directory."""
    from click.testing import CliRunner
    from molbuilder.projects import PROJECTS_ROOT_ENV

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    (tree / "pseudopotential" / "H.psml").write_text("<psml/>\n")
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\n\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    # CliRunner's cwd is the test runner's -- ./pseudopotential does NOT
    # exist here, which is exactly the papercut's shape.
    res = CliRunner().invoke(describe_cmd, [
        "--structure", "P/structure/h2.xyz",
        "--bundle", "P/optimization/calc", "--engine", "siesta",
        "--calculation", "optimization",
        "--shape", "hierarchical", "--vacuum", "8", "--name", "JOB",
        "--psml-lib", "pseudopotential"])
    assert res.exit_code == 0, res.output
    assert (tree / "P" / "optimization" / "calc" / "H.psml").exists(), (
        "the bare name must resolve against the tree and the pseudo "
        "must travel with the calculation")
    # ...and the template says whose the folder is (`template.md` § 6.6).
    from molbuilder.template import one
    from molbuilder.template_file import of as template_of
    calc = tree / "P" / "optimization" / "calc"
    written = read_template(template_of(
        calc, task_of(calc).label).text)
    assert one(written, "psml_lib").source == "person"


@pytest.mark.parametrize("engine, ladder", [("siesta", ["freq"]),
                                            ("pyscf", ["freq"])])
def test_init_writes_the_kinds_recommendations_and_reads_its_ladder_from_the_box(
        tmp_path, monkeypatch, engine, ladder):
    """`template.md` § 6.3a: `molbuilder describe` writes a kind's recommended value
    as the item's `value` AND `default`, for every item the catalogue
    recommends for that kind -- so the vibration template starts tight.
    `vibration.md` § 5.11 § 1: one stage, `freq`, on both engines -- on
    SIESTA one run in phases, the relaxation a phase of it, switched by the
    box.  No engine runs: `init` describes."""
    from click.testing import CliRunner
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.template import catalogue, recommended_for, select
    from molbuilder.template_file import of as template_of

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    res = CliRunner().invoke(describe_cmd, [
        "--structure", "P/structure/h2.xyz", "--bundle", "P/frequency/V",
        "--engine", engine, "--shape", "hierarchical",
        "--calculation", "vibration", "--name", "V"])
    assert res.exit_code == 0, res.output
    bundle = tree / "P" / "frequency" / "V"
    assert [s.name for s in task_of(bundle).stages] == ladder

    written = {it.name: it for it in select(read_template(
        template_of(bundle, task_of(bundle).label).text))}
    recommended = {it.name: recommended_for(it, "vibration")
                   for it in select(catalogue(), engine=engine)
                   if recommended_for(it, "vibration") is not None}
    assert recommended, "the catalogue recommends nothing for this kind"
    for name, rec in recommended.items():
        assert written[name].value == rec and written[name].default == rec, (
            name, written[name].value, written[name].default, rec)
        # ...as nobody's choice, beside the name the person gave
        # (`template.md` § 6.6 obligation 2, plan § 5w K7).
        assert written[name].source == "default", (name, written[name].source)
    ident = "job_name" if engine == "pyscf" else "system_label"
    assert written[ident].source == "person", written[ident].source


#: `engines/stages.md` § 6.7, the shape a kind fixes, on the command line:
#: each row the vibration's engine, the ``--shape`` typed (``None`` left
#: out), the shape the contract says the kind fixes (``None``: none), and
#: what `molbuilder describe` does.
_SHAPE_ROWS = [
    ("siesta", "flat", "hierarchical", "refused in the kind's sentence"),
    ("siesta", None, "hierarchical", "refused, naming the kind's shape"),
    ("pyscf", "flat", None, "taken"),
    ("pyscf", None, None, "refused, asking for --shape"),
]


@pytest.mark.parametrize("engine, shape, kind_shape, outcome", _SHAPE_ROWS,
                         ids=[f"{e}-{s}" for e, s, _k, _o in _SHAPE_ROWS])
def test_the_shape_a_kind_fixes_on_the_command_line(
        tmp_path, monkeypatch, engine, shape, kind_shape, outcome):
    """`engines/stages.md` § 6.7: a SIESTA vibration is hierarchical, the
    kind's (`task.fixed_by_kind`) -- its description's reader refuses
    another shape in the kind's one sentence, and nothing is written;
    `describe` still asks for ``--shape``, its refusal naming the kind's
    shape and why; a PySCF vibration takes either shape.

    THE SILENT FAILURE: a SIESTA vibration described flat, its relaxation,
    force constants and optical calls run in one directory, each writing
    over the last one's .DM and .XV -- the description's reader the one
    door that stops it; and a person on the command line left guessing
    which shape a kind takes.

    MUTATION THIS MUST FAIL AGAINST: the reader's shape check removed;
    `fixed_by_kind` answering transport alone; the refusal not naming the
    kind's shape."""
    from support.road import describe
    r, bundle = describe(tmp_path, monkeypatch, shape=shape, name="V",
                         calculation="vibration", engine=engine)
    said = " ".join(r.output.split())
    if outcome == "taken":
        assert r.exit_code == 0, r.output
        assert task_of(bundle).shape == shape
        return
    assert r.exit_code != 0, r.output
    assert task_of(bundle) is None, "a refused describe wrote a description"
    if kind_shape:
        # THE KIND'S SENTENCE: the shape, and why -- its calls each in a
        # folder of their own (`engines/stages.md` § 6.7).
        assert f"is {kind_shape}" in said and "folder of its own" in said, (
            r.output)
    if shape is None:
        assert "--shape is required" in said, r.output
        assert ((f"--shape {kind_shape}" in said) if kind_shape
                else "Describe it with --shape" not in said), r.output


def test_init_refuses_a_folder_that_already_holds_a_calculation(
        tmp_path, monkeypatch):
    """One calculation per folder (user, 2026-10-01): `init` on a described
    folder is refused, naming where the description is changed -- and the
    description stays as it was.  Through the verb (`support.road`).

    PREVENTS (the W52 review, P1a-12): `init` re-describing a calculation
    silently, under whatever had been prepared from it.

    MUTATION THIS MUST FAIL AGAINST: the refusal removed (the second `init`
    rewrites task.json)."""
    from support.road import STRUCTURE, describe_calculation
    bundle = describe_calculation(tmp_path, monkeypatch)
    before = (bundle / "task.json").read_bytes()
    r = molbuilder("describe", "--structure", STRUCTURE,
               "--bundle", "P/optimization/H2", "--engine", "siesta",
               "--calculation", "optimization",
               "--shape", "flat", "--name", "H2",
               "--psml-lib", "pseudopotential")
    assert r.exit_code != 0, r.output
    assert "already holds a calculation" in r.output, r.output
    assert (bundle / "task.json").read_bytes() == before

