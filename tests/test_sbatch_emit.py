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

import json
import math
import re
import subprocess
from pathlib import Path

import pytest

from molbuilder import diagnostics, runwrap
from molbuilder.diagnostics import Capabilities
from molbuilder.runwrap import WrapperError, render_sbatch
from molbuilder.jobset.model import Resources


_SCHED = {
    "kind": "slurm",
    "directives": {
        "partition": "public", "qos": "public",
        "mail_type": "ALL", "mail_user": "%u@asu.edu", "export": "NONE",
    },
    "defaults": {"time": "0-04:00:00", "cpus_per_task": None, "mem": None},
}


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
    """A project dir carrying the asu-sol script_generation + scheduler
    config (project scope), with an isolated HOME so the server-wide
    lookup chain doesn't leak a real ~/.config file."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / ".molbuilder.json").write_text(json.dumps({
        "script_generation": {
            "preamble": "module load mamba/latest",
            "activation": "source activate",
        },
        "scheduler": _SCHED,
    }))
    return tmp_path


# --------------------------------------------------------------------- #
#  render_sbatch -- pure header rendering                               #
# --------------------------------------------------------------------- #


def test_cpu_header_shape(tmp_path):
    fdf = tmp_path / "cpu-np64.fdf"
    fdf.write_text("NumberOfAtoms 444\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=64)
    assert "#SBATCH -J cpu-np64" in txt
    assert "#SBATCH -N 1" in txt
    assert "#SBATCH -n 64" in txt
    assert "#SBATCH -p public" in txt          # NOT general (job-system.md § 6, routing domains)
    assert "#SBATCH -q public" in txt
    assert "#SBATCH -t 0-04:00:00" in txt
    assert "#SBATCH -o slurm.%j.out" in txt
    assert "#SBATCH --export=NONE" in txt
    # CPU job: no GPU lines, no exclusive.
    assert "--gres" not in txt
    assert "--exclusive" not in txt
    # Delegation body, unchanged launcher (§ 3).
    assert 'bash cpu-np64.run.sh "$@"' in txt


def test_mail_user_percent_pattern_is_flagged_not_dropped(tmp_path):
    # %u doesn't expand in --mail-user (only -o/-e/-i).  We KEEP the
    # user's value (don't twist their config) but flag it explicitly.
    fdf = tmp_path / "j.fdf"; fdf.write_text("NumberOfAtoms 1\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=8)        # _SCHED has %u
    assert '#SBATCH --mail-user="%u@asu.edu"' in txt   # kept verbatim
    assert "do NOT expand in --mail-user" in txt        # but flagged


def test_mail_user_real_address_not_flagged(tmp_path):
    sched = json.loads(json.dumps(_SCHED))
    sched["directives"]["mail_user"] = "me@asu.edu"
    fdf = tmp_path / "j.fdf"; fdf.write_text("NumberOfAtoms 1\n")
    txt = render_sbatch(fdf, sched, ntasks=8)
    assert '#SBATCH --mail-user="me@asu.edu"' in txt
    assert "do NOT expand" not in txt


def test_mem_emitted_when_set(tmp_path):
    fdf = tmp_path / "m.fdf"
    fdf.write_text("x\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=8, mem="120G")
    assert "#SBATCH --mem=120G" in txt


def test_explicit_mem_overrides_default(tmp_path):
    sched = dict(_SCHED)
    sched["defaults"] = dict(_SCHED["defaults"], mem="64G")
    fdf = tmp_path / "g.fdf"; fdf.write_text("Diag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, sched, ntasks=8, gpu=True, gpu_count=1,
                        mem="120G", exclusive=False)
    assert "#SBATCH --mem=120G" in txt and "64G" not in txt


def test_cpus_omitted_when_unset(tmp_path):
    fdf = tmp_path / "c.fdf"
    fdf.write_text("x\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=20)
    assert "#SBATCH -c " not in txt


def test_ntasks_must_be_positive(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("x\n")
    with pytest.raises(WrapperError, match="ntasks"):
        render_sbatch(fdf, _SCHED, ntasks=0)


def test_missing_partition_refuses(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("x\n")
    bad = {"kind": "slurm", "directives": {"qos": "public"}}
    with pytest.raises(WrapperError, match="partition"):
        render_sbatch(fdf, bad, ntasks=4)


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
    text = render_sbatch(fdf, _SCHED, ntasks=2, cpus_per_task=12,
                         gpu=True, gpu_count=2, exclusive=False)
    runwrap._validate_rendered_wrapper(text, fdf)   # raises if bash rejects it


# --------------------------------------------------------------------- #
#  write_run_wrapper wiring (§ 15 B)                                    #
# --------------------------------------------------------------------- #


def test_wrapper_emits_sbatch_when_scheduler_configured(project):
    fdf = project / "cpu-np64.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .false.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=64))
    sbatch = project / "cpu-np64.sbatch"
    assert sbatch.is_file()
    txt = sbatch.read_text()
    assert "#SBATCH -n 64" in txt
    assert "--gres" not in txt          # CPU .fdf -> no GPU lines


def test_no_scheduler_no_sbatch(tmp_path, monkeypatch):
    """`job-system.md` § 6, gate 2: with no `(partition, qos)` pair
    resolvable, there is no queue to address and only the `.run.sh` is
    written.  (Gate 1 -- a record saying `workstation` -- is the other
    reason, and is not what this exercises: here there is no record at all,
    which by itself keeps emitting.)

    Isolation is HOME **and cwd**: the machine scope reads cwd-first
    (running-a-job.md § 5.2), so without the chdir this test's verdict
    depended on the developer's own molbuilder.json at the repo root."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "home").mkdir()
    (tmp_path / ".molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "source activate"},
    }))
    fdf = tmp_path / "local.fdf"
    fdf.write_text("NumberOfAtoms 10\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4))
    assert (tmp_path / "local.run.sh").is_file()
    assert not (tmp_path / "local.sbatch").exists()


def test_emit_sbatch_false_suppresses(project):
    fdf = project / "x.fdf"
    fdf.write_text("NumberOfAtoms 10\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4), emit_sbatch=False)
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
    sched = dict(_SCHED); sched["defaults"] = dict(_SCHED["defaults"], mem=None)
    fdf = _min_cpu_fdf(tmp_path)
    txt = render_sbatch(fdf, sched, ntasks=64, mem="120G")
    assert "#SBATCH --mem=120G" in txt
    assert "auto-estimated" not in txt

