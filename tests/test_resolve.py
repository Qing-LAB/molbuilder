"""`prep` step 2 — the description and the machine become a `ParameterSet`.

Contract: ``docs/execution/generator.md`` § 5 (the object and its precedence),
§ 4 (capability ⊇ allocation ⊇ sweep; the two axis families; the pin channel) ·
``docs/execution/project-layout.md`` § 2.3.1 step 2, M4 ·
``docs/engines/stages.md`` § 4 (effective config).

Written from the contract with the module, 2026-08-11.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.jobset.model import Resources
from molbuilder.resolve import (ALLOCATION_FIELDS, MachineTranslation,
                                ParameterSet, ResolveError, point_token,
                                resolve)
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure
from molbuilder.task import Stage, StructureRef, Task, derive_run
from molbuilder.template import template_with_values


@pytest.fixture
def template() -> str:
    return template_with_values(SiestaConfig(system_label="relax", mesh_cutoff=300.0,
                                        basis_size="DZP"))


#: The ordinary description's one rung.  `engines/stages.md` § 6.5: a job
#: always has at least one stage, so there is no stage-less `_task()` to
#: build -- and every `resolve` below therefore NAMES its stage, which is
#: the interaction the CLI actually has.
STAGE = "coarse"


def _task(stages=None):
    ladder = (tuple(stages) if stages is not None
              else (Stage(name=STAGE, enabled=True, overrides={}),))
    return Task(
        engine="siesta", shape="hierarchical",
        run=derive_run("relax", "C3H2S",
                       stage_names=tuple(s.name for s in ladder)),
        structure=StructureRef(source="bdt.xyz", formula="C3H2S", atoms=6),
        varies=("mesh_cutoff",),
        stages=ladder)


ALLOC = Resources(mpi_np=64, cpus_per_task=4, time="0-04:00:00")


# --------------------------------------------------------------------- #
#  A run is the sweep of length one — the fifth axis as a value          #
# --------------------------------------------------------------------- #

def test_a_run_is_a_parameter_set_of_length_one(template):
    """**The idea the module exists for.**  If a run were anything other than
    a one-element list, every step below would need to know which it was in —
    which is the branch `architecture.md` § 0 forbids."""
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC, stage=STAGE)
    assert isinstance(ps, ParameterSet) and len(ps) == 1
    assert not ps.is_sweep and ps[0].point == {}


def test_a_sweep_of_n_gives_n_elements_through_the_same_call(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"mpi_np": [8, 16, 32], "block_size": [16, 32]}, stage=STAGE)
    assert len(ps) == 6
    assert ps.axes == ("mpi_np", "block_size")


# --------------------------------------------------------------------- #
#  Precedence — template ⊕ stage ⊕ sweep ⊕ pin, and it is total          #
# --------------------------------------------------------------------- #

def test_the_stage_overrides_the_template(template):
    stages = (Stage(name="tight", overrides={"mesh_cutoff": 500.0}),)
    ps = resolve(template, _task(stages), SiestaConfig, allocation=ALLOC,
                 stage="tight")
    assert ps[0].values.mesh_cutoff == 500.0
    assert ps[0].values.basis_size == "DZP"        # untouched by the stage


def test_a_sweep_point_overrides_the_stage(template):
    stages = (Stage(name="tight", overrides={"mesh_cutoff": 500.0}),)
    ps = resolve(template, _task(stages), SiestaConfig, allocation=ALLOC,
                 stage="tight", sweep={"mesh_cutoff": [700.0]})
    assert ps[0].values.mesh_cutoff == 700.0


def test_a_pin_wins_over_everything(template):
    stages = (Stage(name="tight", overrides={"mesh_cutoff": 500.0}),)
    ps = resolve(template, _task(stages), SiestaConfig, allocation=ALLOC,
                 stage="tight", sweep={"mesh_cutoff": [700.0]},
                 pins={"mesh_cutoff": 900.0})
    assert ps[0].values.mesh_cutoff == 900.0


def test_provenance_says_which_source_set_each_value(template):
    """M3: *"the numbers were wrong"* is unanswerable if you cannot tell where
    a value came from."""
    stages = (Stage(name="tight", overrides={"mesh_cutoff": 500.0}),)
    ps = resolve(template, _task(stages), SiestaConfig, allocation=ALLOC,
                 stage="tight", sweep={"pao_energy_shift": [0.02]},
                 pins={"block_size": 16})
    p = ps[0].provenance
    assert p["mesh_cutoff"] == "stage"
    assert p["pao_energy_shift"] == "sweep"
    assert p["block_size"] == "pin"
    assert p["basis_size"] == "template"


# --------------------------------------------------------------------- #
#  The two axis families land in different places (§ 4)                  #
# --------------------------------------------------------------------- #

def test_a_machine_axis_lands_on_the_resources_not_the_values(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"mpi_np": [8, 16]}, stage=STAGE)
    assert [e.resources.mpi_np for e in ps] == [8, 16]
    # ...and the allocation's other fields ride along untouched
    assert all(e.resources.cpus_per_task == 4 for e in ps)


def test_a_parameter_axis_lands_on_the_values_not_the_resources(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"block_size": [16, 32]}, stage=STAGE)
    assert [e.values.block_size for e in ps] == [16, 32]
    assert all(e.resources.mpi_np == 64 for e in ps)


def test_the_split_is_a_lookup_not_a_guess():
    """Every field of the allocation, and only those, routes to resources."""
    assert "mpi_np" in ALLOCATION_FIELDS and "max_memory_mb" in ALLOCATION_FIELDS
    assert "mesh_cutoff" not in ALLOCATION_FIELDS


# --------------------------------------------------------------------- #
#  capability ⊇ allocation ⊇ sweep (§ 4.1)                               #
# --------------------------------------------------------------------- #

def test_a_sweep_point_over_the_allocation_is_refused_not_clamped(template):
    """**The rule, and the reason it is a refusal.**  A clamped trial measures
    something you did not ask for, and reports it as though you had."""
    with pytest.raises(ResolveError, match=r"exceeds this prep's allocation"):
        resolve(template, _task(), SiestaConfig,
                allocation=Resources(mpi_np=16), sweep={"mpi_np": [8, 32]}, stage=STAGE)


def test_the_bound_is_the_allocation_not_the_machine(template):
    """A point inside the allocation is fine however large the machine is —
    and one outside it is refused however small.  The allocation is a
    *decision*, which is what asking for less buys you in queue priority."""
    ps = resolve(template, _task(), SiestaConfig,
                 allocation=Resources(mpi_np=128), sweep={"mpi_np": [128]}, stage=STAGE)
    assert ps[0].resources.mpi_np == 128


# --------------------------------------------------------------------- #
#  The allocation rides on the element                                   #
# --------------------------------------------------------------------- #

def test_every_element_carries_its_own_allocation(template):
    """§ 5.  The deck writer needs the rank count and the wrapper writer needs
    the whole of it; **both read one object**, so neither re-derives."""
    ps = resolve(template, _task(), SiestaConfig,
                 allocation=Resources(mpi_np=32, max_memory_mb=4096), stage=STAGE)
    assert ps[0].resources.mpi_np == 32
    assert ps[0].resources.max_memory_mb == 4096


def test_the_description_no_longer_carries_the_rank_count(template):
    """The leak this step closes: `mpi_np` was a template item, so a folder
    that must name no machine named one — and floor 3 read it straight back
    out.  It is an allocation now, and the template has no opinion."""
    import tomllib
    raw = tomllib.loads(template)
    # @2: the item is DECLARED (a surface must be able to ask) but carries
    # no value -- so the leak this step closed stays closed: floor 3 has
    # nothing to read straight back out.
    assert "value" not in raw["item"]["mpi_np"]
    ps = resolve(template, _task(), SiestaConfig, allocation=Resources(mpi_np=8), stage=STAGE)
    assert ps[0].resources.mpi_np == 8


# --------------------------------------------------------------------- #
#  Labels, stages, refusals                                              #
# --------------------------------------------------------------------- #

def test_a_trial_is_relabelled_so_it_cannot_read_the_real_run(template):
    """SIESTA finds warm files by ``SystemLabel``.  A trial carrying the real
    run's label could read — or overwrite — its ``.DM`` and ``.XV``."""
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"mpi_np": [8]}, stage=STAGE)
    assert ps[0].label != "relax" and ps[0].label.startswith("relax")


def test_a_run_keeps_the_calculations_own_label(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC, stage=STAGE)
    assert ps[0].label == "relax"


def test_point_token_is_the_authoritys_rendering():
    """`job-contracts.md` § 6.3: the coordinate is ONE qualifier — ``G1K4C6``,
    concatenated, because a separator inside it would read as more qualifiers.
    (This joined with ``-`` until the bench fold, C6, 2026-08-11.)"""
    assert point_token({"G": 1, "K": 4, "C": 6}) == "G1K4C6"
    assert point_token({"mpi_np": 8, "block_size": 16}) == \
        "mpi_np8block_size16"
    assert "." not in point_token({"mesh_cutoff": 300.0})
    assert "-" not in point_token({"mesh_cutoff": 300.0})


# --------------------------------------------------------------------- #
#  The explicit-points form, and the specialisation's translation seam   #
# --------------------------------------------------------------------- #

_GKC = MachineTranslation(
    axes=("G", "K", "C"),
    # The benchmark's coupling: K ranks per GPU on G GPUs, C cores per rank.
    to_resources=lambda p, env: {"mpi_np": p["G"] * p["K"],
                                 "cpus_per_task": p["C"]})


def test_explicit_points_are_taken_verbatim_in_order(template):
    """A dependent enumeration is not a cross product: the caller hands the
    points themselves, and the set preserves them as given."""
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep=[{"G": 1, "K": 4, "C": 4}, {"G": 1, "K": 8, "C": 2}],
                 translation=_GKC, stage=STAGE)
    assert len(ps) == 2 and ps.is_sweep
    assert [e.point for e in ps] == [{"G": 1, "K": 4, "C": 4},
                                     {"G": 1, "K": 8, "C": 2}]
    assert ps.axes == ("G", "K", "C")


def test_the_translation_reaches_the_elements_resources(template):
    """The trial's rank count comes from the translated coordinate — the field
    that makes `Job.resources` copyable rather than re-derived (§ 5)."""
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep=[{"G": 1, "K": 4, "C": 2}], translation=_GKC, stage=STAGE)
    assert ps[0].resources.mpi_np == 4
    assert ps[0].resources.cpus_per_task == 2
    assert ps[0].label == "relax-G1K4C2"


def test_the_environment_reaches_the_translation(template):
    """`generator.md` § 6.1: the Environment is one of floor 3's inputs, and
    the translation is where it is consumed — a machine fact like the GPU
    type is read there, not re-detected downstream."""
    seen = {}

    def _with_gpu_type(p, env):
        seen["env"] = env
        return {"gres": f"gpu:{env['gpu_type']}:{p['G']}"}

    tr = MachineTranslation(axes=("G",), to_resources=_with_gpu_type)
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep=[{"G": 2}], translation=tr,
                 environment={"gpu_type": "a100"}, stage=STAGE)
    assert seen["env"] == {"gpu_type": "a100"}
    assert ps[0].resources.gres == "gpu:a100:2"


def test_a_translated_ask_is_bounded_by_the_allocation_too(template):
    """§ 4.1 does not care how the ask was spelled: a derived rank count over
    the allocation is the same refusal as naming ``mpi_np`` directly."""
    with pytest.raises(ResolveError, match=r"exceeds this prep's allocation"):
        resolve(template, _task(), SiestaConfig,
                allocation=Resources(mpi_np=16),
                sweep=[{"G": 1, "K": 32, "C": 1}], translation=_GKC, stage=STAGE)


def test_an_axis_that_names_nothing_is_refused_without_a_translation(template):
    with pytest.raises(ResolveError, match=r"name\s+neither"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                sweep=[{"G": 1, "K": 4, "C": 6}], stage=STAGE)


def test_an_orphan_axis_is_refused_even_with_a_translation(template):
    """The translation declares its axes so an axis NOBODY owns cannot be
    silently swallowed — an axis consumed by nothing would name the trial
    directory while affecting nothing, the *present but not honoured* shape."""
    with pytest.raises(ResolveError, match=r"'mesh_cutof'"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                sweep=[{"G": 1, "K": 4, "C": 2, "mesh_cutof": 150}],
                translation=_GKC, stage=STAGE)


def test_a_partial_coordinate_is_refused_by_name(template):
    """A translation's axes travel together: a point carrying G without K and
    C would make the machine ask a guess."""
    with pytest.raises(ResolveError, match=r"'K'"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                sweep=[{"G": 1}], translation=_GKC, stage=STAGE)


def test_a_run_with_a_translation_supplied_is_still_a_run(template):
    """No sweep + a translation on hand (the bench entry always passes its
    coupling) must not call it on the empty point — a run has no coordinate."""
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 translation=_GKC, stage=STAGE)
    assert len(ps) == 1 and ps[0].point == {}
    assert ps[0].resources.mpi_np == ALLOC.mpi_np


def test_a_translation_may_only_answer_with_allocation_fields(template):
    with pytest.raises(ResolveError, match=r"nothing on Resources"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                stage=STAGE, sweep=[{"G": 1}],
                translation=MachineTranslation(
                    axes=("G",),
                    to_resources=lambda p, env: {"mesh_cutoff": 300.0}))


def test_a_value_that_leaves_the_label_charset_is_refused(template):
    """`job-contracts.md` § 6.3: the token becomes a SystemLabel and a
    directory name, and a ``-`` inside it would announce a qualifier that is
    not there.  ``G-2`` is refused, never smuggled."""
    with pytest.raises(ResolveError, match=r"charset"):
        point_token({"G": -2})
    with pytest.raises(ResolveError, match=r"charset"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                sweep=[{"mpi_np": -8}], stage=STAGE)


def test_a_ladder_needs_a_stage_named(template):
    with pytest.raises(ResolveError, match=r"a stage has to be named"):
        resolve(template, _task(default_siesta_stages("publishable")),
                SiestaConfig, allocation=ALLOC)


def test_naming_a_stage_that_is_not_there_is_refused_with_the_options(template):
    with pytest.raises(ResolveError, match=r"coarse"):
        resolve(template, _task(default_siesta_stages("publishable")),
                SiestaConfig, allocation=ALLOC, stage="nonesuch")


# ``test_a_description_with_no_ladder_refuses_a_stage`` was retired
# 2026-08-16.  It pinned a refusal for a description with NO ladder -- a
# shape `engines/stages.md` § 6.5 deleted, so the test asserted an
# interaction that can no longer happen.  Naming an unknown stage is still
# refused, and that is pinned just above by
# ``test_an_unknown_stage_is_refused_with_the_ladder_listed``.


def test_a_pin_naming_nothing_in_the_schema_is_refused(template):
    with pytest.raises(ResolveError, match=r"nothing in the"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                pins={"blocksize": 16}, stage=STAGE)


# --------------------------------------------------------------------- #
#  § 7 — machine facts cannot enter through floor 2's doors (A-9)        #
# --------------------------------------------------------------------- #

def test_a_pin_naming_a_machine_fact_is_refused_with_its_story(template):
    """A-9 (final review, 2026-08-13): the gates here checked
    ``dataclasses.fields`` names, and ``mpi_np`` IS a field — tagged
    ``allocation: True``, § 7's forbidden machine fact.  A pin naming it
    passed every gate and the deck rendered for a rank count the
    allocation never granted."""
    with pytest.raises(ResolveError, match=r"machine fact"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                pins={"mpi_np": 8}, stage=STAGE)


def test_a_stage_override_naming_a_machine_fact_is_refused(template):
    ladder = (Stage(name="a", overrides={"omp_threads": 4}),)
    t = Task(engine="siesta", shape="hierarchical",
             run=derive_run("relax", "C3H2S", stage_names=("a",)),
             structure=StructureRef(source="bdt.xyz", formula="C3H2S",
                                    atoms=6),
             varies=("omp_threads",), stages=ladder)
    with pytest.raises(ResolveError, match=r"machine fact"):
        resolve(template, t, SiestaConfig, allocation=ALLOC, stage="a")


def test_a_sweep_axis_spelling_a_machine_fact_names_the_road(template):
    """``omp_threads`` is a config field whose EXCHANGE name belongs to
    the allocation (``cpus_per_task``, job-contracts § 6.2) — until the
    fix it slipped through the params split and landed on the VALUES, so
    the deck varied while the allocation stood still."""
    with pytest.raises(ResolveError, match=r"machine fact"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                sweep={"omp_threads": [1, 2]}, stage=STAGE)
