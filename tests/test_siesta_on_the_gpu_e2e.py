"""A SIESTA run on the GPU, on the road, with the engine -- `execution/gpu.md`
§ 1.1 and G6 (a run that uses a GPU is taken to SIESTA's GPU build, and does
not fall back) and `execution/running-a-job.md` § 3.3 (how its ranks share
the GPU).

The silent failures: a run asked on the GPU that runs the packaged CPU build
-- it finishes either way, and only the build the run's SIESTA reports tells
which; and a run script that says MPS is on where it starts none, or off
where it does.  Read through the run's record (the run door's) and the run
script's own dry run.

Run where an NVIDIA GPU is visible, with the source-built
`molbuilder-siesta-gpu` env and the packaged `molbuilder-siesta` the road's
runs need; skipped elsewhere.
"""
from __future__ import annotations

import shutil

import pytest

from _road import conda_hook, env_available, gpu_visible


pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta-gpu")
             and gpu_visible()),
        reason="needs an NVIDIA GPU, the molbuilder-siesta-gpu env and a "
               "detectable conda hook"),
]


def _said(run, step, *words):
    """What ``step`` answered holds each of ``words``."""
    said = run.said[step]
    assert said.exit_code == 0, said.output
    for w in words:
        assert w in said.output, f"{step}: {w!r} not in:\n{said.output}"


def test_a_run_asked_on_the_gpu_runs_the_gpu_build(real_h2_gpu):
    """`gpu.md` § 1.1, G6: ``use_gpu`` on the run card takes SIESTA to its
    GPU build, with no fallback -- the run finished, and the ELPA GPU key
    the run's SIESTA reports, a constant of its build, is the NVIDIA one
    (the packaged CPU build's is ``no-gpu``), read from the run's record.
    With the deck's ``Diag.ELPA.GPU .true.``, which the description's table
    pins, that is the GPU solve SIESTA sets up.

    MUTATION ONLY THIS CATCHES: the run script's activation of the GPU env
    not taking, so the packaged CPU SIESTA -- first on PATH on the road --
    solves instead.  The deck's keyword and the env's name are the
    description's, pinned by `tests/data/gpu_contract.toml`."""
    from molbuilder.runs import folder_answer, run_answer
    attempt = real_h2_gpu.attempt("coarse")
    said = run_answer(attempt)
    assert (said["state"], said["detail"]) == ("finished",
                                               "job_completed"), said
    solver = folder_answer(attempt)["record"]["computation"]["solver"]
    assert solver["elpa_gpu"] == "nvidia-gpu", solver


@pytest.mark.skipif(shutil.which("nvidia-cuda-mps-control") is None,
                    reason="MPS is decided on only where its binary is on "
                           "PATH, and it is not here")
def test_its_dry_run_says_how_its_ranks_share_the_gpu(real_h2_gpu):
    """`running-a-job.md` § 3.3: MPS starts only where the ranks outnumber
    the GPUs, and the run script's dry run says so -- given one GPU, as a
    job asking for one is: one rank shares nothing, MPS off; two (``-np 2``)
    share it, MPS on, both on GPU 0.

    MUTATION THIS MUST FAIL AGAINST: the report read from MPS being allowed
    rather than from the decision (it said on for one rank on one GPU,
    measured 2026-10-17)."""
    _said(real_h2_gpu, "script dry run",
          "Ranks/GPU    : 1 (MPS off)", "rank 0 -> GPU 0")
    _said(real_h2_gpu, "script dry run -np 2",
          "Ranks/GPU    : 2 (MPS on)", "rank 0 -> GPU 0", "rank 1 -> GPU 0")
