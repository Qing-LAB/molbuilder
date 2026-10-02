"""**A9 — two artifacts of one object agree**
(`docs/execution/architecture.md` § 7).

``.run.sh`` and ``.sbatch`` are two renderings of ONE ``Resources``.  Every
other wrapper test calls the renderer with explicit keyword arguments, so the
two files have only ever been compared against a test's own intent — never
against **each other**.  That absence is what let a ``.sbatch`` asking for
``-c 8`` ship beside a ``.run.sh`` whose OMP default was ``1`` (2026-08-17).

**Why a signature rule could not have caught it.**  A8
(`test_architecture_rules`) makes the mistake unwritable at the call site, and
both call sites were *internally* consistent — only the pair of files they
produced disagreed.  A rule about the shape of the source cannot see a wrong
number in a rendered file, so these two rules cover each other rather than
overlap.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from molbuilder import diagnostics
from molbuilder.diagnostics import Capabilities
from molbuilder.jobset.model import Resources
from molbuilder.runwrap import write_run_wrapper


_DECK = ("SystemLabel JOB\nNumberOfAtoms 8\nNumberOfSpecies 1\n"
         "%block ChemicalSpeciesLabel\n 1 1 H\n%endblock ChemicalSpeciesLabel\n")


@pytest.fixture(autouse=True)
def _caps():
    """Synthetic capabilities so no real ``conda env list`` runs."""
    diagnostics.set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta", "molbuilder-siesta-gpu"})))
    yield


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A bundle prepped for a machine with a scheduler -- its record lists the
    `public` queue and says how a shell enters an environment there -- and
    cwd + HOME isolated.

    Isolated for the reason the 2026-08-12 memory records: without it these
    read the DEVELOPER's config and pass off a setup the test never wrote.
    """
    from molbuilder.scheduler import (FILENAME, Domain, Environment,
                                      Topology, write_environment)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    write_environment(Environment(
        scheduler="slurm", topology=Topology(sockets=2, cores_per_socket=64),
        domains=[Domain(name="public", partition="public", qos="public",
                        max_time="7-00:00:00")],
        env_init={"preamble": "module load mamba/latest",
                           "activation": "source activate"}),
        tmp_path / FILENAME)
    return tmp_path


def _render(project: Path, resources: Resources, deck: str = _DECK):
    """Write one deck's pair from one allocation, and return both texts."""
    fdf = project / "JOB.fdf"
    fdf.write_text(deck)
    write_run_wrapper(fdf, resources=resources)
    return ((project / "JOB.run.sh").read_text(),
            (project / "JOB.sbatch").read_text())


def _baked(run_sh: str, name: str) -> list[str]:
    """The wrapper's baked values -- one assignment each since 2026-10-02,
    when the launch-time GPU/CPU branch they were chosen between went."""
    return re.findall(rf"{name}=(\S+)", run_sh)


def _directive(sbatch: str, flag: str):
    m = re.search(rf"^#SBATCH {re.escape(flag)}[= ](\S+)", sbatch, re.M)
    return m.group(1) if m else None


def test_the_pair_agrees_about_ranks_and_cores(project):
    """The defect, stated as a property: ask for 16 x 8 and BOTH files say so.

    The launcher's baked default is what runs when there is no scheduler — a
    workstation target is supported — so ``SLURM_CPUS_PER_TASK`` rescuing the
    scheduled path is not the same as the pair agreeing.
    """
    res = Resources(mpi_np=16, cpus_per_task=8, domain="public",
                    time="0-04:00:00", mem="8G")
    run_sh, sbatch = _render(project, res)

    assert _directive(sbatch, "-n") == "16", sbatch
    assert _directive(sbatch, "-c") == "8", (
        "the sbatch header lost the core count:\n" + sbatch)

    assert _baked(run_sh, "_mpi_np_default") == ["16"], (
        "the launcher's baked rank count disagrees with `-n 16`")
    assert _baked(run_sh, "_omp_threads_default") == ["8"], (
        "the launcher's baked OMP default disagrees with `-c 8`.  Off a "
        "scheduler this default IS the thread count, so a benchmark sweeping "
        "cores-per-rank measures one point N times.")


def test_the_door_refuses_a_loose_allocation(project):
    """A8's rule, enforced by the signature itself.

    The door takes the object; there is no subset to pass.  Pinned because the
    convenience of "just one more keyword" is exactly how eleven of them
    accumulated, and how two call sites came to disagree.
    """
    fdf = project / "JOB.fdf"
    fdf.write_text(_DECK)
    with pytest.raises(TypeError):
        write_run_wrapper(fdf, mpi_np=16, cpus_per_task=8)
