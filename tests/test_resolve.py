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
from molbuilder.resolve import (ALLOCATION_FIELDS, ParameterSet, ResolveError,
                                point_token, resolve)
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure
from molbuilder.task import Stage, StructureRef, Task, derive_run
from molbuilder.template import render_template


@pytest.fixture
def template() -> str:
    return render_template(SiestaConfig(system_label="relax", mesh_cutoff=300.0,
                                        basis_size="DZP"))


def _task(stages=()):
    ladder = tuple(stages)
    return Task(
        engine="siesta", shape="hierarchical",
        run=derive_run("relax", "C3H2S",
                       stage_names=tuple(s.name for s in ladder)),
        structure=StructureRef(source="bdt.xyz", formula="C3H2S", atoms=6),
        varies=(("mesh_cutoff",) if ladder else None),
        stages=(ladder or None))


ALLOC = Resources(mpi_np=64, cpus_per_task=4, time="0-04:00:00")


# --------------------------------------------------------------------- #
#  A run is the sweep of length one — the fifth axis as a value          #
# --------------------------------------------------------------------- #

def test_a_run_is_a_parameter_set_of_length_one(template):
    """**The idea the module exists for.**  If a run were anything other than
    a one-element list, every step below would need to know which it was in —
    which is the branch `architecture.md` § 0 forbids."""
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC)
    assert isinstance(ps, ParameterSet) and len(ps) == 1
    assert not ps.is_sweep and ps[0].point == {}


def test_a_sweep_of_n_gives_n_elements_through_the_same_call(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"mpi_np": [8, 16, 32], "parallel_block_size": [16, 32]})
    assert len(ps) == 6
    assert ps.axes == ("mpi_np", "parallel_block_size")


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
                 pins={"parallel_block_size": 16})
    p = ps[0].provenance
    assert p["mesh_cutoff"] == "stage"
    assert p["pao_energy_shift"] == "sweep"
    assert p["parallel_block_size"] == "pin"
    assert p["basis_size"] == "template"


# --------------------------------------------------------------------- #
#  The two axis families land in different places (§ 4)                  #
# --------------------------------------------------------------------- #

def test_a_machine_axis_lands_on_the_resources_not_the_values(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"mpi_np": [8, 16]})
    assert [e.resources.mpi_np for e in ps] == [8, 16]
    # ...and the allocation's other fields ride along untouched
    assert all(e.resources.cpus_per_task == 4 for e in ps)


def test_a_parameter_axis_lands_on_the_values_not_the_resources(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"parallel_block_size": [16, 32]})
    assert [e.values.parallel_block_size for e in ps] == [16, 32]
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
                allocation=Resources(mpi_np=16), sweep={"mpi_np": [8, 32]})


def test_the_bound_is_the_allocation_not_the_machine(template):
    """A point inside the allocation is fine however large the machine is —
    and one outside it is refused however small.  The allocation is a
    *decision*, which is what asking for less buys you in queue priority."""
    ps = resolve(template, _task(), SiestaConfig,
                 allocation=Resources(mpi_np=128), sweep={"mpi_np": [128]})
    assert ps[0].resources.mpi_np == 128


# --------------------------------------------------------------------- #
#  The allocation rides on the element                                   #
# --------------------------------------------------------------------- #

def test_every_element_carries_its_own_allocation(template):
    """§ 5.  The deck writer needs the rank count and the wrapper writer needs
    the whole of it; **both read one object**, so neither re-derives."""
    ps = resolve(template, _task(), SiestaConfig,
                 allocation=Resources(mpi_np=32, max_memory_mb=4096))
    assert ps[0].resources.mpi_np == 32
    assert ps[0].resources.max_memory_mb == 4096


def test_the_description_no_longer_carries_the_rank_count(template):
    """The leak this step closes: `mpi_np` was a template item, so a folder
    that must name no machine named one — and floor 3 read it straight back
    out.  It is an allocation now, and the template has no opinion."""
    import tomllib
    raw = tomllib.loads(template)
    assert "mpi_np" not in raw["item"]
    ps = resolve(template, _task(), SiestaConfig, allocation=Resources(mpi_np=8))
    assert ps[0].resources.mpi_np == 8


# --------------------------------------------------------------------- #
#  Labels, stages, refusals                                              #
# --------------------------------------------------------------------- #

def test_a_trial_is_relabelled_so_it_cannot_read_the_real_run(template):
    """SIESTA finds warm files by ``SystemLabel``.  A trial carrying the real
    run's label could read — or overwrite — its ``.DM`` and ``.XV``."""
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                 sweep={"mpi_np": [8]})
    assert ps[0].label != "relax" and ps[0].label.startswith("relax")


def test_a_run_keeps_the_calculations_own_label(template):
    ps = resolve(template, _task(), SiestaConfig, allocation=ALLOC)
    assert ps[0].label == "relax"


def test_point_token_is_deterministic_and_filename_safe():
    assert point_token({"mpi_np": 8, "parallel_block_size": 16}) == \
        "mpi_np8-parallel_block_size16"
    assert "." not in point_token({"mesh_cutoff": 300.0})


def test_a_ladder_needs_a_stage_named(template):
    with pytest.raises(ResolveError, match=r"a stage has to be named"):
        resolve(template, _task(default_siesta_stages("publishable")),
                SiestaConfig, allocation=ALLOC)


def test_naming_a_stage_that_is_not_there_is_refused_with_the_options(template):
    with pytest.raises(ResolveError, match=r"coarse"):
        resolve(template, _task(default_siesta_stages("publishable")),
                SiestaConfig, allocation=ALLOC, stage="nonesuch")


def test_a_description_with_no_ladder_refuses_a_stage(template):
    with pytest.raises(ResolveError, match=r"no ladder"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC, stage="tight")


def test_a_pin_naming_nothing_in_the_schema_is_refused(template):
    with pytest.raises(ResolveError, match=r"nothing in the"):
        resolve(template, _task(), SiestaConfig, allocation=ALLOC,
                pins={"blocksize": 16})
