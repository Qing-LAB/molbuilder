"""Tests for the ``molbuilder envs advise siesta-gpu`` advisor.

The advisor is host-aware (probes lscpu / nvidia-smi / nvidia-cuda-
mps-control), so every test injects a synthetic ``HostProbe`` instead
of running the real probes -- that keeps CI green on hosts with no
GPU and pins the deterministic behaviour of the recommendation +
formatter.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from molbuilder.envs import advise as _advise
from molbuilder.envs._cli import envs_group


def _probe(*, phys_cores=20, sockets=1, cps=20,
           gpu_name="NVIDIA RTX 4090",
           gpu_vram_mb=24576, gpu_compute_cap="89",
           gpu_numa=0, mps_available=True) -> _advise.HostProbe:
    return _advise.HostProbe(
        phys_cores=phys_cores, sockets=sockets,
        cores_per_socket=cps,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_compute_cap=gpu_compute_cap,
        gpu_numa=gpu_numa,
        mps_available=mps_available,
    )


# --------------------------------------------------------------------- #
#  recommend(): shape + numbers                                          #
# --------------------------------------------------------------------- #


def test_recommend_returns_three_named_presets_on_typical_workstation():
    """20-core / 1-GPU / MPS-installed: the canonical workstation.

    Policy (2026-06-16): OMP = (phys_cores - 1) // mpi_np for every
    preset -- fills the box, leaves 1 core for the ELPA-GPU host
    driver thread.  Across the three presets the user is choosing
    mpi_np (which sets the diag-parallelism + VRAM/rank trade-off);
    OMP follows.
    """
    presets = _advise.recommend(_probe())  # 20 phys cores
    assert [p.name for p in presets] == ["default", "memory", "fallback"]
    default, memory, fallback = presets
    # default: np=4 → OMP=(20-1)//4=4
    assert default.mpi_np == 4
    assert default.omp == 4
    assert default.mps is True
    # memory: np=2 → OMP=(20-1)//2=9
    assert memory.mpi_np == 2
    assert memory.omp == 9
    assert memory.mps is True
    # fallback: np=1 → OMP=(20-1)//1=19
    assert fallback.mpi_np == 1
    assert fallback.omp == 19
    assert fallback.mps is False


def test_recommend_default_omp_fills_the_box():
    """Policy-shape pin: OMP = (phys - 1) // np.  On 32-core box,
    default (np=4) should give OMP=7, NOT the old fixed-2 value."""
    presets = _advise.recommend(_probe(phys_cores=32, sockets=1, cps=32))
    assert presets[0].mpi_np == 4
    assert presets[0].omp == 7  # (32-1)//4 = 7


def test_recommend_caps_np_by_atom_count():
    """Tiny 3-atom system should cap mpi_np at 3 regardless of cores."""
    presets = _advise.recommend(_probe(), n_atoms=3)
    assert all(p.mpi_np <= 3 for p in presets)
    # The "default" preset's target is 4 but cap pulls it to 3.
    assert presets[0].mpi_np == 3


def test_recommend_caps_np_by_phys_cores_on_small_box():
    """4-core laptop: even the "default" preset can't ask for 4
    ranks if there aren't 4 cores' worth of independent budget."""
    presets = _advise.recommend(_probe(phys_cores=4, sockets=1, cps=4))
    for p in presets:
        assert p.mpi_np <= 4
        assert p.mpi_np * p.omp <= 4
    # OMP must still be at least 1 (never zero, never negative).
    assert all(p.omp >= 1 for p in presets)


def test_recommend_flags_mps_unavailable_in_default_notes():
    """When MPS isn't installed, the user must know -- a 4-rank run
    without MPS serialises through the CUDA driver context, which
    is worse than 1 rank.  Notes carry the warning."""
    presets = _advise.recommend(_probe(mps_available=False))
    assert "MPS" in presets[0].notes  # surfaced on the default preset
    # Fallback's MPS=False is the deliberate state, not a warning.
    assert presets[-1].mps is False
    assert presets[-1].mpi_np == 1


def test_recommend_estimates_vram_when_orbitals_given():
    """VRAM estimate is 16 bytes × N_orb² / np + ~500 MB overhead."""
    presets = _advise.recommend(_probe(), n_orbitals=600)
    for p in presets:
        assert p.est_vram_per_rank_mb is not None
        assert p.est_vram_per_rank_mb > 0


def test_recommend_no_orbitals_means_no_vram_estimate():
    presets = _advise.recommend(_probe())
    for p in presets:
        assert p.est_vram_per_rank_mb is None


# --------------------------------------------------------------------- #
#  format_report: shape of user-facing output                            #
# --------------------------------------------------------------------- #


def test_format_report_contains_host_summary_and_table_header():
    probe = _probe()
    presets = _advise.recommend(probe)
    text = _advise.format_report(probe, presets)
    # Host snapshot section
    assert "host:" in text
    assert "gpu:" in text
    assert "mps:" in text
    # Table columns
    assert "preset" in text and "mpi_np" in text and "omp" in text
    assert "default" in text and "memory" in text and "fallback" in text
    # Recommended-preset export block (the actionable part)
    assert "Recommended preset for this host" in text
    assert "MOLBUILDER_MPI_NP=" in text
    assert "MOLBUILDER_OMP_NUM_THREADS=" in text
    assert "MOLBUILDER_USE_MPS=" in text


def test_format_report_picks_default_when_mps_available():
    probe = _probe(mps_available=True)
    presets = _advise.recommend(probe)
    text = _advise.format_report(probe, presets)
    assert "MOLBUILDER_MPI_NP=4" in text
    assert "MOLBUILDER_USE_MPS=1" in text


def test_format_report_picks_fallback_when_mps_unavailable():
    """Without MPS, the only honest recommendation is 1-rank fallback;
    suggesting MPI_NP=4 would point the user at a config that runs
    SLOWER than single-rank because of CUDA-context serialisation."""
    probe = _probe(mps_available=False)
    presets = _advise.recommend(probe)
    text = _advise.format_report(probe, presets)
    assert "MOLBUILDER_MPI_NP=1" in text
    assert "MOLBUILDER_USE_MPS=0" in text


def test_format_report_problem_line_only_when_inputs_given():
    probe = _probe()
    presets = _advise.recommend(probe)
    assert "problem:" not in _advise.format_report(probe, presets)
    with_inputs = _advise.format_report(
        probe, presets, n_atoms=42, n_orbitals=420,
    )
    assert "problem:" in with_inputs
    assert "42 atoms" in with_inputs


# --------------------------------------------------------------------- #
#  CLI: click integration                                                #
# --------------------------------------------------------------------- #


@pytest.fixture
def stub_probe(monkeypatch):
    """Replace the live probe with a deterministic fixture so the CLI
    test is host-independent."""
    monkeypatch.setattr(
        _advise, "probe_host",
        lambda: _probe(),
    )


def test_cli_advise_siesta_gpu_prints_recommendation(stub_probe):
    runner = CliRunner()
    res = runner.invoke(envs_group, ["advise", "siesta-gpu"])
    assert res.exit_code == 0, res.output
    assert "Recommended preset" in res.output
    assert "MOLBUILDER_MPI_NP=" in res.output


def test_cli_advise_accepts_full_recipe_name(stub_probe):
    runner = CliRunner()
    res = runner.invoke(envs_group, ["advise", "molbuilder-siesta-gpu"])
    assert res.exit_code == 0, res.output


def test_cli_advise_unknown_recipe_emits_notice(stub_probe):
    runner = CliRunner()
    res = runner.invoke(envs_group, ["advise", "molbuilder-pySCF"])
    # Exit 0 (no error -- just informational) but with a clear notice
    # the user will see on stderr.
    assert res.exit_code == 0
    assert "no advisor" in res.output.lower()


def test_cli_advise_passes_n_atoms_to_recommendation(stub_probe):
    runner = CliRunner()
    res = runner.invoke(envs_group,
                        ["advise", "siesta-gpu", "--n-atoms", "5"])
    assert res.exit_code == 0
    assert "5 atoms" in res.output
    # n_atoms=5 caps the default preset's mpi_np to 4 (already capped
    # by atom-cap rule).  Just confirm one cap path actually fires.
    assert "MOLBUILDER_MPI_NP=4" in res.output
