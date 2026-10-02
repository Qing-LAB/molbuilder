"""Tests for the SLURM ``.sbatch`` submission-layer emitter
(``runwrap.render_sbatch`` + the ``render_wrappers`` / ``write_run_wrapper``
wiring).  ``write_sbatch`` was a second writer with no production caller and
went on 2026-08-18 (roadmap P6); one test below asserts its absence.

Authoritative design: docs/execution/job-system.md  two-layer model (header delegates to the unchanged .run.sh)
  § 5  block-by-block header
  § 6  value-source matrix
  running-a-job.md § 3.3  GPU gating + ranks-per-GPU + enforce-binding
  § 6   refuse-to-emit / skip-when-there-is-no-queue
  process/testing.md  the L1/L2 split
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import pytest

from molbuilder import diagnostics, runwrap
from molbuilder.diagnostics import Capabilities
from molbuilder.runwrap import WrapperError, render_sbatch
from molbuilder.jobset.model import Resources


def _stated(**over):
    """Every value a header carries, as the job states them -- the queue it
    named (bound on the record), its ranks, cores per rank, wall and memory
    (`architecture.md` § 5.2).  A `scheduler` config block supplied the
    queue and defaults until 2026-10-02, and mail and export lines with
    them."""
    out = dict(partition="public", qos="public", ntasks=8, cpus_per_task=1,
               time="0-04:00:00", mem="8G")
    out.update(over)
    return out


@pytest.fixture(autouse=True)
def _caps():
    """Synthetic Capabilities so no real ``conda env list`` runs."""
    diagnostics.set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta", "molbuilder-siesta-gpu"}),
    ))
    yield


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project dir carrying the asu-sol record -- a scheduler, the
    `public` queue, and how a shell enters an environment there -- with an
    isolated HOME so the server-wide lookup chain doesn't leak a real
    ~/.config file."""
    from molbuilder.scheduler import (FILENAME, Domain, Environment,
                                      Topology, write_environment)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    write_environment(Environment(
        scheduler="slurm", topology=Topology(sockets=2, cores_per_socket=64),
        domains=[Domain(name="public", partition="public", qos="public",
                        max_time="7-00:00:00")],
        env_init={"preamble": "module load mamba/latest",
                           "activation": "source activate"}),
        tmp_path / FILENAME)
    return tmp_path


#: A CPU run on the record's queue, every value stated.
_ON_PUBLIC = dict(cpus_per_task=1, domain="public", time="0-04:00:00",
                  mem="8G")


# --------------------------------------------------------------------- #
#  render_sbatch -- pure header rendering                               #
# --------------------------------------------------------------------- #


def test_cpu_header_shape(tmp_path):
    fdf = tmp_path / "cpu-np64.fdf"
    fdf.write_text("NumberOfAtoms 444\n")
    txt = render_sbatch(fdf, **_stated(ntasks=64))
    assert "#SBATCH -J cpu-np64" in txt
    assert "#SBATCH -N 1" in txt
    assert "#SBATCH -n 64" in txt
    assert "#SBATCH -c 1" in txt
    assert "#SBATCH -p public" in txt          # NOT general (job-system.md § 6, routing domains)
    assert "#SBATCH -q public" in txt
    assert "#SBATCH -t 0-04:00:00" in txt
    assert "#SBATCH --mem=8G" in txt
    assert "#SBATCH -o slurm.%j.out" in txt
    # No site settings: the mail and export lines went with the
    # `scheduler` block (2026-10-02).
    assert "--export" not in txt and "--mail" not in txt
    # CPU job: no GPU lines, no exclusive.
    assert "--gres" not in txt
    assert "--exclusive" not in txt
    # Delegation body, unchanged launcher (§ 3).
    assert 'bash cpu-np64.run.sh "$@"' in txt


def test_mem_emitted_when_set(tmp_path):
    fdf = tmp_path / "m.fdf"
    fdf.write_text("x\n")
    txt = render_sbatch(fdf, **_stated(mem="120G"))
    assert "#SBATCH --mem=120G" in txt


def test_ntasks_must_be_positive(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("x\n")
    with pytest.raises(WrapperError, match="ntasks"):
        render_sbatch(fdf, **_stated(ntasks=0))


@pytest.mark.parametrize("value", ["partition", "qos", "time", "mem"])
def test_a_value_stated_nowhere_refuses(tmp_path, value):
    """Every value the header carries is required -- the emitter's own guard
    for a caller that reaches it directly.  API-LEVEL because the road
    cannot reach it: prep refuses an unstated launch value first
    (`tests/data/launch_values.toml`).  Nothing fills one in -- not a
    config default, not the scheduler's own (`architecture.md` § 5.2)."""
    fdf = tmp_path / "x.fdf"
    fdf.write_text("x\n")
    with pytest.raises(WrapperError, match=value):
        render_sbatch(fdf, **_stated(**{value: None}))


# How a GPU ask is spelled -- a count, never a card -- is asserted through
# the road, `test_launch_door.py`
# (`test_a_gpu_ask_is_a_count_and_reaches_sbatch_in_slurms_spelling`).


# --------------------------------------------------------------------- #
#  bash -n validity                                                     #
# --------------------------------------------------------------------- #


def test_rendered_sbatch_is_valid_bash(tmp_path):
    """The header parses as shell before anything writes it.

    Asked ``write_sbatch`` until 2026-08-18 -- a second writer with no
    production caller, deleted with P6.  The gate itself did not move: it runs
    inside ``render_wrappers``, which is where the text is produced, and the
    mode the file lands with is asserted where the writing happens
    (``test_wrapper_emits_sbatch_when_scheduler_configured`` below)."""
    fdf = tmp_path / "gpu-2a100.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    text = render_sbatch(fdf, **_stated(ntasks=2, cpus_per_task=12),
                         gpu=True, gpu_count=2, exclusive=False)
    runwrap._validate_rendered_wrapper(text, fdf)   # raises if bash rejects it


# --------------------------------------------------------------------- #
#  write_run_wrapper wiring (§ 15 B)                                    #
# --------------------------------------------------------------------- #


def test_wrapper_emits_sbatch_when_scheduler_configured(project):
    fdf = project / "cpu-np64.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .false.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=64,
                                                       **_ON_PUBLIC))
    sbatch = project / "cpu-np64.sbatch"
    assert sbatch.is_file()
    txt = sbatch.read_text()
    assert "#SBATCH -n 64" in txt
    assert "--gres" not in txt          # CPU .fdf -> no GPU lines


def test_emit_sbatch_false_suppresses(project):
    fdf = project / "x.fdf"
    fdf.write_text("NumberOfAtoms 10\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4, **_ON_PUBLIC),
                              emit_sbatch=False)
    assert (project / "x.run.sh").is_file()
    assert not (project / "x.sbatch").exists()


def _min_cpu_fdf(tmp_path):
    """Minimal SIESTA .fdf carrying the fields the mem estimator reads."""
    fdf = tmp_path / "job.fdf"
    fdf.write_text(
        "SystemName test\n"
        "NumberOfAtoms 100\n"
        "NumberOfSpecies 1\n"
        "PAO.BasisSize DZP\n"
        "MeshCutoff 300 Ry\n"
        "LatticeConstant 1.0 Ang\n"
        "%block LatticeVectors\n"
        "20.0 0.0 0.0\n0.0 20.0 0.0\n0.0 0.0 20.0\n"
        "%endblock LatticeVectors\n"
        "%block kgrid_Monkhorst_Pack\n"
        "2 0 0 0.0\n0 2 0 0.0\n0 0 1 0.0\n"
        "%endblock kgrid_Monkhorst_Pack\n"
        "%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        + "".join(f"{i*0.1} 0.0 0.0 1\n" for i in range(100))
        + "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    return fdf


def test_explicit_mem_skips_estimate(tmp_path):
    fdf = _min_cpu_fdf(tmp_path)
    txt = render_sbatch(fdf, **_stated(ntasks=64, mem="120G"))
    assert "#SBATCH --mem=120G" in txt
    assert "auto-estimated" not in txt

