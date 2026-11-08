"""``molbuilder describe`` — the portable description, and what it must never write.

Contract: ``docs/execution/architecture.md`` § 4 (the route: *ask → check →
write*, **2, asking 3 and 4 about the description itself**) · ``docs/execution/job-system.md`` § 5.1 (the command)
· ``docs/execution/project-layout.md`` § 2.1 (what makes the folder portable) ·
``docs/engines/stages.md`` § 6.3 (the structure is a reference), § 6.5 (a
calculation with one parameter set), § 6.6 (the split preflight).

Written from the contract rather than from the
implementation — the properties below are the ones the design would be wrong
without, not the ones the code happens to have.  Every case is the command as
a person types it (`support.road`), and what it left on disk.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from molbuilder.calculation import open as open_calculation
from molbuilder.task import Stage, of as task_of
from molbuilder.template import read_template
from support.road import (STRUCTURE, _all_written, describe_calculation,
                          molbuilder)


def _typed(**flags):
    """`molbuilder describe` of the road's tree's structure, into a folder of
    its own, as a person types it -- ``flags`` over what the line says."""
    words = {"structure": STRUCTURE, "bundle": "P/optimization/JOB",
             "engine": "siesta", "calculation": "optimization",
             "name": "JOB", "psml-lib": "pseudopotential", **flags}
    return molbuilder("describe", *(w for flag, value in words.items()
                                    for w in (f"--{flag}", value)))


# --------------------------------------------------------------------- #
#  What it writes, and what it must NOT                                 #
# --------------------------------------------------------------------- #

def test_describing_writes_the_description_and_its_request_and_no_deck(
        tmp_path, monkeypatch):
    """**The route's whole definition.**  A deck carries values that depend on
    how it will be launched — ``BlockSize`` above all — so rendering belongs to
    ``prep``, for the machine that will run it.  A ``describe`` that emitted a
    deck would put step 3 before step 1 (`script-preparation.md` § 3.0).

    What the folder holds is `job-system.md` § 5.1's list: the template, the
    description, its job request (`execution/job-request.md` § 9: a
    calculation's folder holds both, § 7) and the data files -- the structure
    as its own pair, the pseudopotentials."""
    bundle = describe_calculation(tmp_path, monkeypatch)
    assert sorted(p.name for p in bundle.iterdir()) == [
        "H.psml", "H2.source.molstruct.json", "H2.source.xyz",
        "H2.template.toml", "job-request.json", "task.json"]


def test_the_description_names_no_machine(tmp_path, monkeypatch):
    """`project-layout.md` § 2.1: *"none of them mention a machine"*, which is
    what makes the folder mean the same thing wherever you copy it -- and
    `engines/template.md` § 2: **no run setting is declared in a template,
    valueless or not** -- the machine's answers and a person's alike are the
    job request's.  A surface asks the catalogue which questions a run has.

    MUTATION THIS MUST FAIL AGAINST: the writer emitting run settings
    valueless again (2.27.x's slot, which migrate's refusal by content and
    its move by version disagreed about)."""
    from molbuilder.template import catalogue, is_run_setting, one
    bundle = describe_calculation(tmp_path, monkeypatch)
    raw = tomllib.loads(open_calculation(bundle).template_text)
    held = [n for n in raw["item"]
            if is_run_setting(one(catalogue(), n, engine="siesta"))]
    assert held == [], (
        f"the template holds run settings {held} -- they are the job "
        f"request's (engines/template.md § 2)")


def test_the_structure_travels_as_the_calculations_own_pair(tmp_path,
                                                            monkeypatch):
    """`molbuilder describe` writes the structure it describes as the calculation's
    own pair, named for the LABEL -- ``<label>.source.xyz`` and its sidecar,
    the catalogue's name and the hand-over's (`job-contracts.md` § 6.3,
    `StructureCodec.source_files`) -- ``task.json`` records that name and a
    witness, its formula and atom count, which lets a description opened
    against a structure that has since changed *say so* (`stages.md` § 6.3);
    and the ``--vacuum 8`` it was described with travels in the sidecar, so
    the loader prep reads the pair with reads it back (the fix of 2026-08-12).

    The structure FILE is ``structure.xyz`` and the label ``JOB``: init named
    the pair after the file until 2026-10-04 (plan D20)."""
    from molbuilder.workingcopy_structure import StructureCodec
    describe_calculation(tmp_path, monkeypatch)
    r = _typed(vacuum=8)
    assert r.exit_code == 0, r.output
    calc = tmp_path / "projects" / "P" / "optimization" / "JOB"
    assert sorted(p.name for p in calc.glob("*.source.*")) == [
        "JOB.source.molstruct.json", "JOB.source.xyz"]
    task = task_of(calc)
    assert task.structure.source == "JOB.source.xyz"
    assert (task.structure.formula, task.structure.atoms) == ("H2", 2)
    assert StructureCodec().load(calc / task.structure.source).vacuum == (
        8.0, 8.0, 8.0)


@pytest.mark.parametrize("calculation", ["optimization", "vibration"])
def test_every_description_states_its_kind(tmp_path, monkeypatch, calculation):
    """U0 (warm-files § 4.2a, 2026-08-13): `calculation` keys the engine's
    warm-file vocabulary, and every description states it, an
    optimization's included (`job-contracts.md` § 6.1; W57 R10, 2026-10-06
    -- an optimization's was left out until then)."""
    bundle = describe_calculation(tmp_path, monkeypatch,
                                  calculation=calculation)
    raw = json.loads((bundle / "task.json").read_text())
    assert raw["calculation"] == calculation
    assert task_of(bundle).calculation == calculation


def test_a_ladder_becomes_stages_and_varies(tmp_path, monkeypatch):
    """`--stage-strategy` names a shipped ladder: its tiers are the stages,
    and what they set is what the description varies (`stages.md` § 6.5)."""
    task = task_of(describe_calculation(tmp_path, monkeypatch))
    assert [s.name for s in task.stages] == ["coarse", "medium"]
    assert "relax_type" in task.varies


def test_the_label_and_the_id_derive_from_the_name(tmp_path, monkeypatch):
    """§ 3 rule 1: it happens once.  The label is not stored and the id is,
    and ``Task.__post_init__`` proves the two against each other — so a
    description that got here has already been checked."""
    task = task_of(describe_calculation(tmp_path, monkeypatch,
                                        name="bdt-relax"))
    assert task.label == "bdt-relax"
    assert task.run.id == "bdt-relax_H2"


# --------------------------------------------------------------------- #
#  Ask → check → write, and the order is the point                      #
# --------------------------------------------------------------------- #

def _a_ladder_of(monkeypatch, **overrides):
    """The ladder `--stage-strategy` hands describe, as if a shipped preset
    said ``overrides``: what the command takes is a shipped ladder's name, so
    a value reaches describe's own check no other way (`job-system.md`
    § 5.1)."""
    monkeypatch.setattr("molbuilder.jobset.ladder.engine_ladder",
                        lambda engine, strategy: [Stage(name="tight",
                                                        overrides=overrides)])


#: ONE ROW A RULE, each refused in its own words: `job-system.md` § 5.1.  The
#: flags typed over the road's line, and what the refusal says.
_REFUSED = [
    # A label becomes a filename and a shell word, so the character set is
    # narrow; it is refused by name, not a crash (`run-identity.md` § 3).
    ("a name that cannot be a label", {"name": "Über"}, "letter or a digit"),
    # The kind is the kind table's to know: nothing is guessed from a word
    # (`architecture.md` § 3.3).
    ("a kind nothing describes", {"calculation": "no-hyphens!"},
     "no calculation kind 'no-hyphens!'"),
    # A SIESTA run cannot start without a `.psml` for every species, so a
    # describe naming a library without one is refused by name.  Until 2.65.3
    # the command line wrote the calculation and said only `WARN: no H.psml`,
    # a calculation found unable to start at prep.
    ("a species with no pseudopotential", {"psml-lib": "P/structure"},
     "no .psml file for H"),
    # One calculation per folder (user, 2026-10-01), naming where the
    # description is changed.  PREVENTS (the W52 review, P1a-12): describing
    # again, silently, under whatever had been prepared from it.
    ("a folder that already holds a calculation",
     {"bundle": "P/optimization/H2", "name": "H3"},
     "already holds a calculation"),
    # A calculation is never inside another's folder, nor holds one (the
    # user, 2026-11-07: "calculations can continue from another, not within
    # another"; `project-layout.md` § 7, invariant 2a), each named by its
    # address from the projects root.  PREVENTS: a calculation written two
    # folders down inside H2, whose history then holds it and whose every
    # later act walks over it -- or one written around H2.
    ("a folder inside another calculation's folder",
     {"bundle": "P/optimization/H2/variants/JOB"},
     "P/optimization/H2/variants/JOB is inside the calculation "
     "P/optimization/H2"),
    ("a folder that holds a calculation below it",
     {"bundle": "P/optimization"},
     "P/optimization holds the calculation P/optimization/H2"),
]


@pytest.mark.parametrize("flags, words", [r[1:] for r in _REFUSED],
                         ids=[r[0] for r in _REFUSED])
def test_describe_refuses_saying_why_and_writes_nothing(tmp_path, monkeypatch,
                                                        flags, words):
    """*"Describing a calculation writes every file or none"*, and a refusal
    happens while nothing has been written: every file and folder of the tree
    is as it was."""
    tree = tmp_path / "projects"
    describe_calculation(tmp_path, monkeypatch)
    was = _all_written(tree), set(tree.rglob("*"))
    r = _typed(**flags)
    assert r.exit_code != 0, r.output
    assert words in r.output, r.output
    assert (_all_written(tree), set(tree.rglob("*"))) == was


def test_describes_own_check_refuses_a_stage_the_schema_does_not_know(
        tmp_path, monkeypatch):
    """§ 6.6's third check, and it is the existing preflight doing it rather
    than a second implementation: describe asks it of the ladder it was
    handed, before anything is written.  The preflight's own cases are
    `test_task_preflight.py`'s, its hard limits at Save
    `test_task_setup_tab.py`'s -- what is here is that describe asks it.

    THE SILENT FAILURE: a shipped preset misspelt, described and found at
    prep, on the cluster."""
    tree = tmp_path / "projects"
    describe_calculation(tmp_path, monkeypatch)
    _a_ladder_of(monkeypatch, mesh_cutof=300.0)
    r = _typed(**{"stage-strategy": "publishable"})
    assert r.exit_code != 0, r.output
    assert "mesh_cutof" in r.output, r.output
    assert not (tree / "P" / "optimization" / "JOB").exists()


def test_a_value_outside_its_recommended_range_describes_and_says_so(
        tmp_path, monkeypatch):
    """A value merely outside the recommended range describes, the
    description's own check saying it, which the command prints
    (`engines/template.md` § 5.3): advice, never a refusal."""
    describe_calculation(tmp_path, monkeypatch)
    _a_ladder_of(monkeypatch, mesh_cutoff=1e9)
    r = _typed(**{"stage-strategy": "publishable"})
    assert r.exit_code == 0, r.output
    assert "outside the recommended range" in r.output, r.output


def test_a_failure_while_writing_publishes_nothing(tmp_path, monkeypatch):
    """*"Describing a calculation writes every file or none."*

    Mutation-tested by construction: the failure is injected, so this proves
    the staging directory is really doing the work rather than the happy path
    merely never failing.
    """
    tree = tmp_path / "projects"
    describe_calculation(tmp_path, monkeypatch)

    def fails(*_a, **_k):
        raise OSError("the disk filled")
    # THE JOB REQUEST'S WRITER FAILS, after the template and the description
    # were staged: nothing of either may reach the target.
    monkeypatch.setattr("molbuilder.job_request.write", fails)
    r = _typed()
    assert r.exit_code != 0, r.output
    assert "the disk filled" in r.output, r.output
    optimization = tree / "P" / "optimization"
    assert list((optimization / "JOB").iterdir()) == []
    assert not [p for p in optimization.iterdir()
                if p.name.startswith(".JOB.")]


# --------------------------------------------------------------------- #
#  The pseudopotentials                                                 #
# --------------------------------------------------------------------- #

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
    from molbuilder.template import one
    bundle = describe_calculation(tmp_path, monkeypatch)
    # The road stands above the tree: ./pseudopotential does NOT exist here,
    # which is exactly the papercut's shape.
    assert not Path("pseudopotential").exists()
    assert (bundle / "H.psml").exists(), (
        "the bare name must resolve against the tree and the pseudo "
        "must travel with the calculation")
    # ...and the template says whose the folder is (`template.md` § 6.6).
    written = read_template(open_calculation(bundle).template_text)
    assert one(written, "psml_lib").source == "person"


# --------------------------------------------------------------------- #
#  A kind's own start                                                   #
# --------------------------------------------------------------------- #

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
    from molbuilder.template import catalogue, recommended_for, select
    bundle = describe_calculation(tmp_path, monkeypatch, engine=engine,
                                  calculation="vibration", name="V")
    assert [s.name for s in task_of(bundle).stages] == ladder

    written = {it.name: it for it in select(read_template(
        open_calculation(bundle).template_text))}
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
