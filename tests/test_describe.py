"""``jobset describe`` — the portable description, and what it must never write.

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
from molbuilder.task import read_task
from molbuilder.template import read_template, schema_fingerprint


@pytest.fixture
def struct() -> Structure:
    return Structure(elements=["S", "C", "C", "H"],
                     positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                         [1.21, 0.0, 2.48], [2.15, 0.0, 1.94]]),
                     vacuum=(10.0, 10.0, 10.0))


@pytest.fixture
def cfg() -> SiestaConfig:
    return SiestaConfig(system_label="relax", mesh_cutoff=300.0)


def _describe(struct, cfg, stages=(), *, shape="hierarchical",
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
    machine = ("mpi_np", "omp_threads", "max_memory_mb")
    carrying = [n for n in machine if "value" in raw["item"][n]]
    assert carrying == [], f"floor 2 is naming a machine: {carrying}"


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
    assert not list((tmp_path / "calc").glob("*.xyz"))


def test_a_ladder_becomes_stages_and_varies(struct, cfg, tmp_path):
    D.write_description(_describe(struct, cfg,
                                  default_siesta_stages("publishable")),
                        tmp_path / "calc")
    task = read_task(tmp_path / "calc" / "task.json")
    assert [s.name for s in task.stages] == ["coarse", "medium", "tight"]
    assert "relax_type" in task.varies


def test_no_ladder_omits_both_keys_entirely(struct, cfg, tmp_path):
    """§ 6.5: *a description with no stages IS a single-parameter-set
    calculation*, and an empty ``stages`` would be a second way to spell it."""
    D.write_description(_describe(struct, cfg, ()), tmp_path / "calc")
    raw = json.loads((tmp_path / "calc" / "task.json").read_text())
    assert "stages" not in raw and "varies" not in raw


def test_the_fingerprint_is_stamped_and_matches_the_template(struct, cfg,
                                                             tmp_path):
    """Unit 4a's rule: whatever writes the template computes it.  The
    description and the catalogue must agree, or the preflight's one
    non-refusal row is comparing a number with nothing."""
    D.write_description(_describe(struct, cfg), tmp_path / "calc")
    task = read_task(tmp_path / "calc" / "task.json")
    tpl = read_template((tmp_path / "calc" / "relax.template.toml").read_text())
    assert task.schema_fingerprint == tpl.fingerprint == \
        schema_fingerprint(SiestaConfig)


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
#  The two surfaces write the same bytes                                #
# --------------------------------------------------------------------- #

def test_the_producer_is_pure_so_two_surfaces_cannot_disagree(struct, cfg):
    """§ 4.1's Promotion A: one producer, two writers.  A browser that built
    its own description would be a second place for the bytes to differ, and
    § 6.4 is the rule that they must not."""
    a = _describe(struct, cfg, default_siesta_stages("publishable"))
    b = _describe(struct, cfg, default_siesta_stages("publishable"))
    assert a.files() == b.files()
