"""P4 unit 5 — a resource-shaped override reaches all THREE destinations.

``engines/stages.md`` § 5 splits a promoted field by *where it lands*, and the
middle two rows are the ones a stage ladder makes visible:

  * **a deck line that is also a resource decision** — ``diag_algorithm``
    lands in the deck AND in the wrapper's env routing (§ 5.1: any
    ``Diag.Algorithm elpa*`` routes to the ELPA-linked build, CPU-ELPA
    included);
  * **a field the deck never carries** — ``mpi_np`` lands on the wrapper, and
    § 5.2 adds the twist this file exists for: a deck LINE can still be
    derived from it.  ``BlockSize`` is that line, and BENCH-MARKS is where the
    deck says so.

So one description with two stages that differ in solver and rank count must
produce: two decks whose lines differ, two wrappers activating two conda
environments, and two BENCH-MARKS blocks each declaring its own stage's
derivation.  Milestone M4 states exactly this.

Nothing here stubs the seam it is testing: the decks are rendered by the
shipped renderer, the wrappers by the shipped ``prep_jobset`` reading those
real decks off disk.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.jobset.prep import prep_jobset
from molbuilder.siesta.input import _auto_block_size, _block_size_bounds
from molbuilder.siesta import render_siesta_stage_fdfs
from molbuilder.siesta.stages import stages_to_jobset
from molbuilder.structure import Structure
from molbuilder.task import Stage


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def slab():
    """64 atoms — big enough that the rank constraint
    ``BlockSize <= floor(n_atoms / mpi_np)`` gives a DIFFERENT answer at 4
    ranks (cap 16) than at 16 (cap 4).  A 2-atom fixture would land on 1 for
    both and the per-stage derivation would be invisible."""
    pos = [[2.5 * i, 2.5 * j, 2.5 * k]
           for i in range(4) for j in range(4) for k in range(4)]
    return Structure(elements=["H"] * 64,
                     positions=np.array(pos, dtype=float),
                     vacuum=(12.0, 12.0, 12.0))


@pytest.fixture
def template():
    return SiestaConfig(system_label="JOB", mpi_np=4,
                        diag_algorithm="ScaLAPACK", enable_gpu=False)


@pytest.fixture
def ladder():
    """The M4 sentence as a ladder: ScaLAPACK at 4 ranks, then ELPA at 16."""
    return [
        Stage(name="coarse", enabled=True,
              overrides={"mesh_cutoff": 150.0, "mpi_np": 4,
                         "diag_algorithm": "ScaLAPACK"}),
        Stage(name="tight", enabled=True,
              overrides={"mesh_cutoff": 300.0, "mpi_np": 16,
                         "diag_algorithm": "ELPA-2STAGE"}),
    ]


def _bench_block(fdf_text: str) -> str:
    m = re.search(r"(?s)# === molbuilder bench-marks BEGIN ===(.*?)"
                  r"# === molbuilder bench-marks END ===", fdf_text)
    assert m, "the deck carries no BENCH-MARKS block"
    return m.group(1)


def _keyword(fdf_text: str, key: str):
    m = re.search(rf"(?im)^\s*{re.escape(key)}\s+(\S+)", fdf_text)
    return m.group(1) if m else None


# --------------------------------------------------------------------- #
#  Destination 1 — the deck line                                        #
# --------------------------------------------------------------------- #


def test_solver_override_reaches_each_deck(slab, template, ladder):
    fdfs = render_siesta_stage_fdfs(slab, template, ladder)
    # ScaLAPACK is SIESTA's built-in Divide-and-Conquer and is emitted by
    # OMITTING the keyword (engines/siesta.md § 13), so the coarse deck
    # asserts absence -- the honest form of "this stage did not opt in".
    assert _keyword(fdfs["JOB_01_coarse.fdf"], "Diag.Algorithm") is None
    assert _keyword(fdfs["JOB_02_tight.fdf"], "Diag.Algorithm") == "ELPA-2STAGE"


def test_rank_override_reaches_each_decks_blocksize(slab, template, ladder):
    """The derived line, per stage.  4 ranks over 64 atoms caps BlockSize at
    16; 16 ranks caps it at 4.  A template-wide derivation would give both
    decks the same number."""
    fdfs = render_siesta_stage_fdfs(slab, template, ladder)
    assert _keyword(fdfs["JOB_01_coarse.fdf"], "BlockSize") == "16"
    assert _keyword(fdfs["JOB_02_tight.fdf"], "BlockSize") == "4"


# --------------------------------------------------------------------- #
#  Destination 2 — the wrapper's environment                            #
# --------------------------------------------------------------------- #


def test_two_solvers_give_two_conda_environments(slab, template, ladder,
                                                 tmp_path):
    """M4: *"renders two decks whose solver differs AND two wrappers
    activating different conda environments"*.

    The routing is not passed in — ``write_run_wrapper`` reads it back out of
    each deck (§ 5.1), which is why this test writes the real files and preps
    the real JobSet rather than asserting on a config."""
    fdfs = render_siesta_stage_fdfs(slab, template, ladder)
    for name, text in fdfs.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "H.psml").write_text("stub", encoding="utf-8")

    jobset = stages_to_jobset(template, ladder, shared=["H.psml"])
    prep_jobset(jobset, tmp_path, emit_sbatch=False)

    def env_of(wrapper: str) -> str:
        text = (tmp_path / wrapper).read_text(encoding="utf-8")
        envs = re.findall(r"(?m)^\s*(?:source|conda)\s+activate\s+(\S+)", text)
        assert envs, f"{wrapper} activates no environment"
        return envs[0]

    coarse_env = env_of("JOB_01_coarse.run.sh")
    tight_env = env_of("JOB_02_tight.run.sh")
    assert coarse_env != tight_env
    # ELPA is linked only into the GPU build, CPU-ELPA included (§ 5.1).
    assert "gpu" not in coarse_env
    assert "gpu" in tight_env


def test_rank_override_reaches_each_stages_resources(template, ladder):
    """``mpi_np`` is the third row of § 5's table: a field the deck never
    carries, which rides to the wrapper on ``Job.resources``."""
    jobset = stages_to_jobset(template, ladder, shared=[])
    by_name = {j.name: j for j in jobset.jobs}
    assert by_name["coarse"].resources.mpi_np == 4
    assert by_name["tight"].resources.mpi_np == 16


# --------------------------------------------------------------------- #
#  Destination 3 — the BENCH-MARKS block                                #
# --------------------------------------------------------------------- #


def test_bench_marks_declares_each_stages_own_derivation(slab, template,
                                                         ladder):
    fdfs = render_siesta_stage_fdfs(slab, template, ladder)
    coarse = _bench_block(fdfs["JOB_01_coarse.fdf"])
    tight = _bench_block(fdfs["JOB_02_tight.fdf"])
    assert "default=16" in coarse and "range=[1,16]" in coarse
    assert "default=4" in tight and "range=[1,4]" in tight


def test_bench_marks_records_the_launch_quantity(slab, template, ladder):
    """§ 5.2: the block exists so a later change of launch can RE-DERIVE the
    coupled lines.  ``_auto_block_size`` takes ``mpi_np``, so a block that
    does not record it cannot be re-derived from — the declaration would be a
    number with no stated origin."""
    fdfs = render_siesta_stage_fdfs(slab, template, ladder)
    assert re.search(r"(?m)^#\s+mpi_np\s+4\b",
                     _bench_block(fdfs["JOB_01_coarse.fdf"]))
    assert re.search(r"(?m)^#\s+mpi_np\s+16\b",
                     _bench_block(fdfs["JOB_02_tight.fdf"]))


def test_bench_marks_records_auto_when_ranks_are_unset(slab, ladder):
    """An unset rank count is a real state, not zero: the picker takes its
    size-only branch.  The block says ``auto`` rather than inventing a
    number, matching PROVENANCE's spelling for the same field."""
    cfg = SiestaConfig(system_label="JOB", mpi_np=None)
    stage = Stage(name="only", enabled=True, overrides={})
    fdfs = render_siesta_stage_fdfs(slab, cfg, [stage])
    assert re.search(r"(?m)^#\s+mpi_np\s+auto\b",
                     _bench_block(fdfs["JOB_01_only.fdf"]))


# --------------------------------------------------------------------- #
#  The invariant the derived range buys                                 #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("n_atoms,mpi_np,gpu", [
    (64, 4, False), (64, 16, False), (20, 32, False), (200, 16, False),
    (2000, 64, False), (10000, 32, False), (81, None, False),
    (212, 4, True), (16, 8, True), (1000, 2, True),
])
def test_declared_range_always_contains_the_declared_default(n_atoms, mpi_np,
                                                             gpu):
    """The defect this replaced: the range was the constant ``(16, 256)``
    while the default was derived, so a block routinely declared its own
    emitted value out of bounds — and advised a bench tool UPWARD, past the
    rank constraint, which is the direction that empties ranks."""
    lo, hi = _block_size_bounds(n_atoms, mpi_np, gpu_mode=gpu)
    default = _auto_block_size(n_atoms, mpi_np, gpu_mode=gpu)
    assert lo <= default <= hi


def test_user_set_blocksize_is_inside_its_own_declared_range(slab):
    """``parallel_block_size`` is honoured verbatim by ``render_fdf`` and may
    exceed what the ranks would have chosen.  The window widens to hold it:
    the user's number is a decision, not a value to advertise as illegal."""
    cfg = SiestaConfig(system_label="JOB", mpi_np=16,
                       parallel_block_size=128)
    stage = Stage(name="only", enabled=True, overrides={})
    fdfs = render_siesta_stage_fdfs(slab, cfg, [stage])
    text = fdfs["JOB_01_only.fdf"]
    assert _keyword(text, "BlockSize") == "128"
    block = _bench_block(text)
    m = re.search(r"field BlockSize\s+anchor=BlockSize\s+type=pow2\s+"
                  r"range=\[(\d+),(\d+)\]\s+default=(\d+)", block)
    assert m, block
    lo, hi, default = (int(g) for g in m.groups())
    assert lo <= default <= hi
    assert default == 128
