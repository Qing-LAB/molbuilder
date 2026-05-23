"""Shell-wrapper emission (``molbuilder.runwrap``).

Each test binds a synthetic Capabilities via ``set_capabilities``;
the autouse fixture in ``tests/conftest.py`` resets it afterwards.
``write_*`` tests use ``tmp_path`` for real on-disk I/O so we can
confirm chmod and the resulting filename.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from molbuilder.diagnostics import (Capabilities, EXTENSION_TO_CATEGORY,
                                      set_capabilities)
from molbuilder.runwrap import (WrapperError, render_run_wrapper,
                                  write_run_wrapper)


def _bind(envs_overrides=None):
    """Bind a synthetic snapshot for the current test."""
    cfg = {"envs": envs_overrides} if envs_overrides else {}
    set_capabilities(Capabilities(
        runtime_config = cfg,
        conda_binary   = "/usr/bin/conda",
    ))


# --------------------------------------------------------------------- #
#  Extension routing                                                    #
# --------------------------------------------------------------------- #


def test_extension_table_covers_two_engines():
    assert EXTENSION_TO_CATEGORY[".fdf"] == "siesta"
    assert EXTENSION_TO_CATEGORY[".py"]  == "pyscf"


def test_render_unknown_extension_raises():
    _bind()
    with pytest.raises(WrapperError, match="unsupported script extension"):
        render_run_wrapper(Path("/tmp/job.txt"))


# --------------------------------------------------------------------- #
#  SIESTA (.fdf) wrapper text                                           #
# --------------------------------------------------------------------- #


def test_render_siesta_single_process():
    _bind()
    text = render_run_wrapper(Path("/somewhere/my-job.fdf"))
    assert "siesta my-job.fdf > my-job.out" in text
    assert "mpirun" not in text
    assert "conda run -n molbuilder-siesta" in text
    assert text.startswith("#!/usr/bin/env bash\n")


def test_render_siesta_with_mpi_ranks():
    _bind()
    text = render_run_wrapper(Path("/somewhere/my-job.fdf"), mpi_np=4)
    assert "mpirun -np 4 siesta my-job.fdf > my-job.out" in text
    assert "molbuilder-siesta" in text


def test_render_siesta_mpi_np_below_two_ignored():
    """np=1 is single-process; don't pretend MPI."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=1)
    assert "mpirun" not in text


def test_render_siesta_redirects_stdout_per_job_layout_v1():
    """Stdout -> ``<basename>.out`` matches docs/protocols/job-layout.md."""
    _bind()
    text = render_run_wrapper(Path("/x/system-label.fdf"))
    assert "> system-label.out" in text


# --------------------------------------------------------------------- #
#  SIESTA-MPI thread pinning                                            #
#                                                                       #
#  User report (2026-05-19): N-rank ``mpirun -np N siesta`` was         #
#  spawning N×CPU_COUNT threads because each MPI rank inherits the      #
#  default BLAS / OpenMP thread count (= number of cores).  Manual      #
#  workaround was ``export OMP_NUM_THREADS=1`` etc. before mpirun.      #
#  The wrapper now does this automatically for SIESTA-MPI runs.         #
# --------------------------------------------------------------------- #


def test_render_siesta_mpi_pins_blas_to_one_and_sets_omp():
    """SIESTA + mpi_np >= 2: BLAS pinned to 1 per rank, OMP set
    (auto-divided across ranks by default) -- the cross-cutting
    anti-oversubscription recipe shared with PySCF / spectra.
    Rewritten 2026-05-22 from the original OMP=1 contract (which
    crippled hybrid runs) to OMP=physical_cores // mpi_np."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=4)
    # BLAS is always 1 per rank.
    assert "export MKL_NUM_THREADS=1" in text
    assert "export OPENBLAS_NUM_THREADS=1" in text
    # OMP is set to SOMETHING (auto-resolved or user-set), not absent.
    assert "export OMP_NUM_THREADS=" in text
    # Exports must precede the exec line so the subshell `conda run`
    # inherits them.
    exec_ix = text.find("exec conda run")
    omp_ix  = text.find("export OMP_NUM_THREADS=")
    assert 0 <= omp_ix < exec_ix, (
        "OMP_NUM_THREADS export must come BEFORE the exec line"
    )


def test_render_siesta_omp_threads_kwarg_wins():
    """User-set omp_threads overrides the physical_cores // mpi_np
    auto-detect.  The wrapper emits exactly the value the user asked
    for so cluster schedulers (which allocate cores explicitly) win."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=4, omp_threads=2)
    assert "export OMP_NUM_THREADS=2" in text


def test_render_siesta_single_process_still_pins_blas():
    """Single-process SIESTA: BLAS still pinned to 1 per process so
    BLAS doesn't spawn its own pool on top of OMP threads.  OMP gets
    physical cores by default (the user wants threading -- BLAS=1 +
    OMP=physical is the canonical recipe, not BLAS-only)."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"))     # no mpi_np
    assert "export MKL_NUM_THREADS=1" in text
    assert "export OPENBLAS_NUM_THREADS=1" in text
    # OMP set to physical cores (numerical value, not absent).
    assert "export OMP_NUM_THREADS=" in text


def test_render_siesta_mpi_np_one_pins_blas_too():
    """np=1 is single-process semantically; same recipe as no-mpi."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=1)
    assert "export OPENBLAS_NUM_THREADS=1" in text
    assert "export OMP_NUM_THREADS=" in text


def test_render_siesta_max_memory_emits_ulimit():
    """max_memory_mb kwarg becomes a ``ulimit -v`` soft cap so a
    runaway SIESTA process can't OOM the host."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=4,
                              max_memory_mb=8192)
    assert "ulimit -v" in text
    # 8192 MB = 8388608 KB.
    assert "8388608" in text


def test_render_pyscf_does_not_pin_threads():
    """PySCF parallelism is BLAS/OpenMP threading; never pin those
    to 1 for a PySCF wrapper."""
    _bind()
    text = render_run_wrapper(Path("/x/y.py"))
    for needle in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS"):
        assert needle not in text, (
            f"PySCF wrapper unexpectedly pins {needle!r} -- this "
            f"serialises PySCF's main parallelism path"
        )


# --------------------------------------------------------------------- #
#  PySCF (.py) wrapper text                                             #
# --------------------------------------------------------------------- #


def test_render_pyscf():
    _bind()
    text = render_run_wrapper(Path("/somewhere/my-job.py"))
    assert "python my-job.py" in text
    assert "conda run -n molbuilder-pySCF" in text
    # PySCF scripts handle their own logging; no stdout redirect.
    assert "> my-job" not in text


def test_render_pyscf_ignores_mpi_np():
    """np is meaningless for python; emit the same wrapper either way."""
    _bind()
    a = render_run_wrapper(Path("/x/y.py"))
    _bind()
    b = render_run_wrapper(Path("/x/y.py"), mpi_np=8)
    assert a == b


def test_render_multidot_basename_preserved():
    """``job.spectra.py`` should keep its multi-dotted stem in the
    wrapper text (so users see ``python job.spectra.py``)."""
    _bind()
    text = render_run_wrapper(Path("/x/job.spectra.py"))
    assert "python job.spectra.py" in text


# --------------------------------------------------------------------- #
#  Env override + missing routing                                       #
# --------------------------------------------------------------------- #


def test_render_explicit_env_override():
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), env="my-custom-siesta")
    assert "conda run -n my-custom-siesta" in text


def test_render_picks_up_config_env_override():
    """Per-machine envs overrides flow through Capabilities -> wrapper."""
    _bind({"siesta": "siesta-ng-v54"})
    text = render_run_wrapper(Path("/x/y.fdf"))
    assert "conda run -n siesta-ng-v54" in text


# --------------------------------------------------------------------- #
#  write_run_wrapper -- on-disk side effects                            #
# --------------------------------------------------------------------- #


def test_write_creates_sibling_dot_run_sh(tmp_path):
    _bind()
    script = tmp_path / "my-job.fdf"
    script.write_text("# fake fdf\n")
    wrapper = write_run_wrapper(script)
    assert wrapper == tmp_path / "my-job.run.sh"
    assert wrapper.is_file()
    assert "siesta my-job.fdf" in wrapper.read_text()


def test_write_sets_executable_bit(tmp_path):
    _bind()
    script = tmp_path / "x.py"
    script.write_text("print('ok')\n")
    wrapper = write_run_wrapper(script)
    mode = wrapper.stat().st_mode
    assert mode & stat.S_IXUSR
    assert mode & stat.S_IXGRP
    assert mode & stat.S_IXOTH


def test_write_preserves_multidot_basename(tmp_path):
    """``job.spectra.py`` -> ``job.spectra.run.sh`` (NOT ``job.run.sh``)."""
    _bind()
    script = tmp_path / "job.spectra.py"
    script.write_text("# fake\n")
    wrapper = write_run_wrapper(script)
    assert wrapper.name == "job.spectra.run.sh"


def test_write_overwrites_existing(tmp_path):
    _bind()
    script = tmp_path / "x.fdf"
    script.write_text("# fake\n")
    wrapper = write_run_wrapper(script)
    first = wrapper.read_text()
    write_run_wrapper(script, mpi_np=8)
    second = wrapper.read_text()
    assert first != second
    assert "-np 8" in second


def test_write_missing_script_raises(tmp_path):
    _bind()
    with pytest.raises(WrapperError, match="not found"):
        write_run_wrapper(tmp_path / "does-not-exist.fdf")
