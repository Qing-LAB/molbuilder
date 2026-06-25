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


def test_render_siesta_always_uses_mpirun():
    """SIESTA is fundamentally MPI-launched.  2026-05-24: changed
    from "bare siesta when mpi_np < 2" to "always mpirun, default
    np=physical_cores".

    2026-05-28: the probe's MPI launcher line now uses the runtime
    shell variable ``$_mpi_np`` (settable via ``-np N`` flag or
    ``MB_NP=N`` env) instead of a Python-baked rank count.  The
    generation-time value becomes ``_mpi_np_default``.  Exec is
    replaced by a captured run + propor diagnostic (see
    ``test_render_siesta_emits_propor_diagnostic``)."""
    _bind()
    text = render_run_wrapper(Path("/somewhere/my-job.fdf"))
    # The MPI branch sets ``_launch_cmd="$_numa_wrap_gpu mpirun
    # -np $_mpi_np $_mpirun_bind siesta"`` -- the ``$_numa_wrap_gpu``
    # slot is empty for CPU mode (defensive default) and populated
    # by ``_gpu_runtime_defaults_block`` for GPU mode.
    assert ('_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np '
            '$_mpirun_bind siesta"' in text)
    # The launcher call uses the probe-resolved cmd + the fdf;
    # `exec` was replaced by a captured invocation so the diagnostic
    # block can run after a crash.
    # 2026-05-30: stdout file is dynamic ``$_out_file``, resolved at
    # run-time by the run-index block (-run0 by default; -runN under
    # ``--continue``).
    assert '$_launch_cmd my-job.fdf > $_out_file' in text
    assert '_out_file="my-job-run${_run_n}.out"' in text
    assert "exec " not in text or "exec {" not in text, (
        "no top-level exec — the wrapper traps SIESTA's exit code"
    )
    # Post-2026-06-24: ``conda activate`` uses the ``_target_env`` var
    # rather than a hardcoded env name (so the same activation line
    # works in both autodetect=true and autodetect=false branches).
    # Pin both the var assignment AND the var-form activate call to
    # equate to the pre-2026-06-24 ``conda activate molbuilder-siesta``.
    assert '_target_env="molbuilder-siesta"' in text
    assert 'conda activate "$_target_env"' in text
    assert text.startswith("#!/usr/bin/env bash\n")


def test_render_siesta_with_mpi_ranks():
    """mpi_np from the form becomes the DEFAULT for -np; the launcher
    line uses the runtime $_mpi_np shell variable so user can override."""
    _bind()
    text = render_run_wrapper(Path("/somewhere/my-job.fdf"), mpi_np=4)
    # Generation-time default baked into a shell variable.
    assert "_mpi_np_default=4" in text
    # Probe block's MPI branch uses the runtime variable.
    assert '_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"' in text
    assert "molbuilder-siesta" in text


def test_render_siesta_mpi_np_one_still_uses_mpirun():
    """np=1 still goes through mpirun -- a SIESTA-MPI build needs
    the MPI runtime even for a single rank.  The default propagates."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=1)
    assert "_mpi_np_default=1" in text
    assert '_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"' in text


def test_render_siesta_emits_np_arg_parser():
    """2026-05-28: wrapper accepts ``-np N`` / ``MB_NP=N`` runtime
    override.  Pin the parser shape so a regression silently
    re-bakes the rank count."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=15)
    # Default fallback chain: arg -> env -> generation-time value.
    assert '_mpi_np="${MB_NP:-$_mpi_np_default}"' in text
    # Arg parser handles -np / --np / -h / unknown.
    assert "-np|--np)" in text
    assert "-h|--help)" in text
    # Integer validation.
    assert "must be a positive integer" in text


def test_render_siesta_emits_propor_diagnostic():
    """2026-05-28: on SIESTA exit-non-zero with propor: ERROR in the
    .out, the wrapper prints a focused retry hint that names the
    actual basename + specific safe -np values.  Pin the key text
    so a regression doesn't silently drop the diagnostic."""
    _bind()
    text = render_run_wrapper(Path("/x/hemeC.fdf"), mpi_np=15)
    # Captured run, not exec.
    assert "set +e" in text
    assert "_siesta_exit=$?" in text
    # Propor detection.  2026-05-30: stdout filename is dynamic
    # (``$_out_file`` so --continue can write -runN.out); the grep
    # reads from that variable, not the baked basename.
    assert 'grep -aq "propor: ERROR" "$_out_file"' in text
    # Retry hint names the actual basename (not $0).
    assert "bash hemeC.run.sh -np 8" in text
    # The empirical table is preserved.
    assert "works: 2, 4, 6, 8, 12, 14" in text
    assert "fails: 9, 10, 11, 13, 15, 16" in text
    # Re-exit with SIESTA's code.
    assert 'exit "$_siesta_exit"' in text


def test_render_siesta_emits_build_probe_block():
    """2026-05-24 evening: the wrapper now probes ``siesta --version``
    at run time and selects the launcher based on the binary's
    self-reported ``Parallelisations:`` line (MPI / OMP / serial).
    Pins the key shell idioms so a regression doesn't silently
    de-probe the wrapper back to a static launcher."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=4)
    # Probe runs siesta --version once.
    assert 'siesta --version 2>/dev/null' in text
    # Parses Version + Parallelisations.
    assert "/^Version/" in text
    assert "/^Parallelisations/" in text
    # All four branches present (MPI / OMP-only / unknown / serial).
    assert '_has_mpi=1' in text
    assert '_has_omp=1' in text
    assert 'serial build' in text
    assert 'MPI fallback' in text
    # Banner shows the probed values.
    assert 'SIESTA version' in text
    assert 'Build paral.' in text
    assert 'Launch mode' in text
    # 2026-05-26/-27: probe uses word-boundary matching so ``NoMPI``
    # / ``pre-MPI`` (hypothetical negative-disabled labels) don't
    # falsely set _has_mpi=1.  The normalisation step ALSO collapses
    # any POSIX whitespace (tab, vertical-tab, mixed spaces) to
    # single spaces so the probe focuses on CONTENT not formatting
    # -- a future SIESTA build that emits ``Parallelisations:\tMPI``
    # or ``MPI,\tOpenMP`` still parses correctly.  Pin the
    # normalisation pipeline + the spaced-anchor case patterns so a
    # regression to the loose ``*MPI*`` substring (or to ``tr ",;"``
    # alone, which would miss tabs) fails this test.
    assert "_par_norm=" in text, "probe must normalise separators"
    assert '*" MPI "*' in text, (
        "probe must use spaced-anchor ``*\" MPI \"*`` to reject NoMPI"
    )
    assert '*MPI*) _has_mpi=1' not in text, (
        "regression: loose ``*MPI*`` substring match falsely catches "
        "NoMPI / pre-MPI labels"
    )
    assert 'tr "[:space:]"' in text, (
        "probe must normalise ANY whitespace (tabs etc) via "
        "``tr \"[:space:]\"`` -- enumerating only ``,;`` would miss "
        "tab-separated future SIESTA output."
    )


def test_render_siesta_redirects_stdout_per_job_layout_v1():
    """Stdout -> ``<basename>-runN.out`` matches docs/protocols/
    job-layout.md (post-2026-05-30: ``-runN`` series for
    ``--continue`` support).  First run is -run0."""
    _bind()
    text = render_run_wrapper(Path("/x/system-label.fdf"))
    assert "> $_out_file" in text
    # The resolver bakes the basename into the per-script template.
    assert '_out_file="system-label-run${_run_n}.out"' in text


def test_render_siesta_auto_mpi_clamps_to_n_atoms(monkeypatch):
    """Auto-mpi path: when n_atoms is known + physical_cores >
    n_atoms, resolved_mpi must clamp to n_atoms.  Otherwise SIESTA
    aborts at propor IMAX=0 (small molecule + many-core host).
    Caught by the 2026-05-28 holistic-math audit."""
    _bind()
    # Pretend the host has 64 physical cores.
    monkeypatch.setattr("molbuilder.runtime_info.physical_core_count",
                        lambda: 64)
    # 30-atom molecule, no user-set mpi_np -> auto.
    text = render_run_wrapper(Path("/x/small-mol.fdf"), n_atoms=30)
    # The generation-time default must be 30, NOT 64.  The launcher
    # itself uses ``$_mpi_np`` so user can also override at run time
    # via ``-np N`` / ``MB_NP=N``.
    assert "_mpi_np_default=30" in text, (
        "auto-mpi must clamp default to n_atoms=30 (host has 64 cores, "
        "but 30 atoms = max usable rank count)"
    )
    assert "_mpi_np_default=64" not in text
    assert '_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"' in text
    # The clamp note must be visible in the wrapper.
    assert "auto-mpi clamped from 64" in text
    assert "to 30 (n_atoms)" in text


def test_render_siesta_auto_mpi_no_clamp_when_atoms_geq_cores(monkeypatch):
    """Auto-mpi path: when n_atoms >= physical_cores, no clamp --
    use all cores (the original auto behaviour)."""
    _bind()
    monkeypatch.setattr("molbuilder.runtime_info.physical_core_count",
                        lambda: 8)
    text = render_run_wrapper(Path("/x/big-mol.fdf"), n_atoms=200)
    assert "_mpi_np_default=8" in text
    assert '_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"' in text
    assert "clamped" not in text


def test_render_siesta_user_mpi_over_atoms_emits_warning(monkeypatch):
    """User-set mpi_np > n_atoms is HONOURED verbatim (sovereign
    override) but tagged with a runtime WARNING in the wrapper output
    so the user sees what's about to crash + how to fix it."""
    _bind()
    text = render_run_wrapper(
        Path("/x/tiny.fdf"), mpi_np=20, n_atoms=10
    )
    # Honoured verbatim (we do NOT silently override user input).
    assert "_mpi_np_default=20" in text
    assert '_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"' in text
    # But the warning is unmistakable.
    assert "WARNING: user-set mpi_np=20 > n_atoms=10" in text
    assert "propor IMAX=0" in text
    assert "Lower mpi_np to <= 10" in text


def test_write_run_wrapper_parses_n_atoms_from_fdf(tmp_path,
                                                    monkeypatch):
    """End-to-end: write_run_wrapper parses NumberOfAtoms from the
    .fdf so the auto-mpi clamp works WITHOUT the caller passing
    n_atoms explicitly.  This is the live path -- the web flow
    + CLI both invoke write_run_wrapper, never render_run_wrapper
    directly with n_atoms."""
    _bind()
    monkeypatch.setattr("molbuilder.runtime_info.physical_core_count",
                        lambda: 32)
    fdf_text = (
        "SystemName        tiny\n"
        "SystemLabel       tiny\n"
        "NumberOfAtoms     12\n"     # <-- the line being parsed
        "NumberOfSpecies   1\n"
    )
    fdf_path = tmp_path / "tiny.fdf"
    fdf_path.write_text(fdf_text)
    wrapper_path = write_run_wrapper(fdf_path)
    text = wrapper_path.read_text()
    assert "_mpi_np_default=12" in text, (
        "expected wrapper to auto-clamp default to n_atoms=12 parsed "
        "from .fdf (not 32 = physical cores)"
    )
    assert '_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"' in text
    assert "auto-mpi clamped from 32" in text


def test_write_run_wrapper_unparseable_fdf_falls_back(tmp_path,
                                                       monkeypatch):
    """If the .fdf doesn't have a parseable NumberOfAtoms line
    (truncated / corrupted / pre-emission stub) the wrapper falls
    back to the un-clamped auto-mpi (physical_cores) -- better to
    render SOMETHING than to refuse."""
    _bind()
    monkeypatch.setattr("molbuilder.runtime_info.physical_core_count",
                        lambda: 4)
    fdf_path = tmp_path / "broken.fdf"
    fdf_path.write_text("# .fdf with no NumberOfAtoms line\n")
    wrapper_path = write_run_wrapper(fdf_path)
    text = wrapper_path.read_text()
    # Falls back to physical_cores (4) without clamping.
    assert "_mpi_np_default=4" in text
    assert '_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"' in text
    assert "clamped" not in text


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
    # Exports must precede the launch line so the activated env
    # inherits them.  2026-05-28: the wrapper no longer uses ``exec`` --
    # it runs the launcher inside ``set +e`` so the post-run propor
    # diagnostic can inspect the .out.  Pin against the launch call
    # ``$_launch_cmd <fdf> > <out>`` instead.
    # 2026-05-30: launch line uses $_out_file (dynamic per-run name).
    launch_ix = text.find("$_launch_cmd y.fdf > $_out_file")
    omp_ix    = text.find("export OMP_NUM_THREADS=")
    assert 0 <= omp_ix < launch_ix, (
        "OMP_NUM_THREADS export must come BEFORE the launch line"
    )


def test_render_siesta_omp_threads_kwarg_wins():
    """User-set omp_threads overrides the physical_cores // mpi_np
    auto-detect.  The wrapper emits exactly the value the user asked
    for so cluster schedulers (which allocate cores explicitly) win."""
    _bind()
    text = render_run_wrapper(Path("/x/y.fdf"), mpi_np=4, omp_threads=2)
    # As of 2026-06-15, the wrapper exports a SHELL VAR (so it can
    # honor ``-omp N`` and ``OMP_NUM_THREADS`` env at run time) and the
    # numeric value lives in ``_omp_threads_default=N``.  Pin both
    # signatures so a regression in either spot is caught.
    assert "_omp_threads_default=2" in text
    assert "export OMP_NUM_THREADS=$_omp_threads" in text


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


def test_render_conda_activation_six_paths_and_logging():
    """The wrapper handles six conda-env detection scenarios in order
    of reliability so it runs as a fully standalone job script under
    SLURM/PBS without any pre-activation by the caller:
      1. CONDA_DEFAULT_ENV == target -> skip activation (idempotent).
      2. baked-in generator-time conda base (fast path on same machine).
      3. ``conda info --base`` (conda already on PATH).
      4. ``mamba info --base`` (mamba-only HPCs, e.g. ASU sc002).
      5. ``module load <name>`` then re-probe (module-gated HPCs).
      6. Common-location filesystem probe ($HOME/miniforge3 etc.).
    Falling through all six is an explicit error with actionable advice.

    Also: every wrapper invocation writes a vigorous per-run log
    (``<basename>.runwrap-<timestamp>.log``) with the path taken,
    activation state, and key env vars -- so a remote-scheduler
    failure can be diagnosed without re-submitting.

    Pins the 2026-06-24 upgrade from the older 3-path block to the
    6-path scheduler-aware block + per-run log file.  The full
    6-path block is opt-in via ``script_generation.autodetect_conda``
    (docs/config.md § 3.6) -- this test exercises the autodetect=True
    shape; the autodetect=False default shape is pinned separately."""
    _bind()
    text = render_run_wrapper(Path("/x/job.fdf"), mpi_np=4,
                                autodetect_override=True)
    # Per-run log file + tee.
    assert "_runwrap_log=" in text
    assert ".runwrap-$(date" in text
    assert 'exec > >(tee -a "$_runwrap_log")' in text
    # Pre-activation hook.
    assert "MOLBUILDER_PREACTIVATE_CMDS" in text
    # Each of the six paths must appear by its inline tag.
    for tag in ("path 1:", "path 2:", "path 3:", "path 4:",
                "path 5:", "path 6:"):
        assert tag in text, f"missing `{tag}` in rendered activation block"
    # Engine-specific knobs that must NOT regress.
    assert "conda activate" in text
    assert "molbuilder-siesta" in text
    assert "etc/profile.d/conda.sh" in text
    # Error message + actionable advice.
    assert "could not locate conda/mamba" in text
    assert "MOLBUILDER_PREACTIVATE_CMDS" in text
    assert "exit 1" in text
    # Activation block runs BEFORE the launch line (otherwise the env
    # isn't ready when SIESTA launches).  Post-2026-05-28 the launch
    # is no longer ``exec`` (so the propor diagnostic can run after);
    # check against the ``$_launch_cmd ... > ... .out`` line.
    activate_ix = text.find('conda activate "$_target_env"')
    # 2026-05-30: launch line uses $_out_file (dynamic per-run name).
    launch_ix   = text.find("$_launch_cmd job.fdf > $_out_file")
    assert 0 <= activate_ix < launch_ix, (
        "conda activation must precede the launch line; otherwise "
        "the subshell SIESTA runs in wouldn't have the env."
    )


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
    assert '_target_env="molbuilder-pySCF"' in text
    assert 'conda activate "$_target_env"' in text
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
    assert '_target_env="my-custom-siesta"' in text
    assert 'conda activate "$_target_env"' in text


def test_render_picks_up_config_env_override():
    """Per-machine envs overrides flow through Capabilities -> wrapper."""
    _bind({"siesta": "siesta-ng-v54"})
    text = render_run_wrapper(Path("/x/y.fdf"))
    assert '_target_env="siesta-ng-v54"' in text
    assert 'conda activate "$_target_env"' in text


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
    # The probe-driven launcher resolves the actual ``siesta`` command
    # at run time; we just check the .fdf appears in the exec line.
    assert "my-job.fdf" in wrapper.read_text()


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


# ---------------------------------------------------------------------
# --continue / --force / -run<N> series  (2026-05-30)
#
# These tests pin the render-level shape (text appears in the wrapper)
# AND the run-time behaviour (actually source the wrapper via bash
# with the relevant flags + check what $_out_file resolves to).  The
# bash-level checks short-circuit BEFORE the conda activation block --
# they ``exit 0`` right after the run-index resolver so we don't have
# to mock conda inside a unit test.
# ---------------------------------------------------------------------


import shutil
import subprocess


def _emit_truncated_wrapper(tmp_path, basename, suffix=".fdf"):
    """Render a wrapper + chop off everything after the run-index
    resolver block + strip the conda activation step, then append
    ``exit 0``.  The truncated wrapper short-circuits after
    $_out_file is set, so we can inspect that value without needing
    conda or the SIESTA / PySCF binary.

    Conda stripping is needed because the live PySCF env's
    ``activate.d/cuda-nvcc_activate.sh`` references unbound vars
    (host-specific issue, not a molbuilder bug) and ``set -u`` in
    the wrapper would abort before the resolver ran."""
    _bind()
    script = tmp_path / f"{basename}{suffix}"
    script.write_text("# fake\n")
    wrapper_path = write_run_wrapper(script)
    text = wrapper_path.read_text()
    # Strip the per-run logging + conda activation block in-place
    # (preserving line numbers downstream).  Range: from the
    # ``# --- Per-run log file`` comment that opens the logging
    # preamble through the trailing ``which python:`` log line that
    # closes the post-activation state dump.  Post 2026-06-24's
    # rewrite, the activation block emits 6 detection paths + per-run
    # log file + tee'd output, all of which we strip together so the
    # downstream tests can exercise just the run-index resolver.
    log_start = text.find("# --- Per-run log file")
    assert log_start >= 0, (
        "per-run-log + activation block not found "
        "(post-2026-06-24 marker missing)"
    )
    # The block ends with the post-activate ``which python:`` log line;
    # cut at the start of the next blank line after it.
    end_marker = '_log INFO "which python:'
    end_ix = text.find(end_marker, log_start)
    assert end_ix >= 0, "post-activate ``which python:`` log line not found"
    # Cut at the newline AFTER the ``which python:`` line.
    end_eol = text.find("\n", end_ix)
    assert end_eol >= 0
    # Plus the blank line that separates blocks.
    cut_end = end_eol + 1
    if text[cut_end:cut_end + 1] == "\n":
        cut_end += 1
    text = (
        text[:log_start]
        + "# Conda activation + logging stripped for the truncated test wrapper.\n"
        + ": # no-op\n\n"
        + text[cut_end:]
    )
    # Cut at the line right AFTER the resolver's echo to keep that
    # line in the wrapper -- it prints "[molbuilder] run index: N ...".
    marker = '[molbuilder] run index:'
    ix = text.find(marker)
    assert ix >= 0, "resolver marker not found in wrapper"
    end_of_echo = text.find("\n", ix)
    wrapper_path.write_text(
        text[: end_of_echo + 1]
        + '\necho "_out_file=$_out_file"\necho "_run_n=$_run_n"\nexit 0\n'
    )
    return wrapper_path


def _run_wrapper(wrapper_path, *args):
    """Source the truncated wrapper with ``args`` and return
    (stdout, exit_code)."""
    bash = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run(
        [bash, str(wrapper_path), *args],
        cwd=str(wrapper_path.parent),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout, proc.stderr, proc.returncode


def test_continue_first_run_is_run0(tmp_path):
    """First run (no prior -runN.out) -> -run0.out regardless of
    --continue / --force flags."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    stdout, _stderr, code = _run_wrapper(w)
    assert code == 0
    assert "_out_file=myjob-run0.out" in stdout
    assert "_run_n=0" in stdout


def test_continue_advances_when_prior_run_exists(tmp_path):
    """With -run0.out already present, --continue produces -run1.out."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    # Pretend -run0 already exists from a previous invocation.
    (tmp_path / "myjob-run0.out").write_text("prior result")
    stdout, _stderr, code = _run_wrapper(w, "--continue")
    assert code == 0
    assert "_out_file=myjob-run1.out" in stdout
    assert "_run_n=1" in stdout


def test_continue_picks_max_plus_one(tmp_path):
    """When -run0 and -run2 both exist (e.g. -run1 was manually
    deleted), --continue uses max(N)+1 = 3.  Pin the max-not-count
    behaviour."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    (tmp_path / "myjob-run0.out").write_text("r0")
    (tmp_path / "myjob-run2.out").write_text("r2")
    stdout, _stderr, code = _run_wrapper(w, "--continue")
    assert code == 0
    assert "_out_file=myjob-run3.out" in stdout


def test_no_flag_with_prior_run_refuses(tmp_path):
    """No --continue + prior run exists + no --force -> refuse,
    naming both alternative flags in the error message."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    (tmp_path / "myjob-run0.out").write_text("prior")
    stdout, stderr, code = _run_wrapper(w)
    assert code == 1, (stdout, stderr)
    assert "previous output exists" in stderr
    assert "--continue" in stderr
    assert "--force" in stderr


def test_force_overwrites_run0(tmp_path):
    """--force restarts from -run0 even when -run0 exists.  Old
    file is NOT deleted by the wrapper; only the run-index sequence
    resets (and SIESTA's stdout will overwrite the existing -run0.out
    when it launches)."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    (tmp_path / "myjob-run0.out").write_text("prior")
    stdout, _stderr, code = _run_wrapper(w, "--force")
    assert code == 0
    assert "_out_file=myjob-run0.out" in stdout
    # The prior file is still on disk (we didn't delete it).
    assert (tmp_path / "myjob-run0.out").exists()


def test_continue_short_form_works(tmp_path):
    """``-c`` is the short form of ``--continue``."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    (tmp_path / "myjob-run0.out").write_text("prior")
    stdout, _stderr, code = _run_wrapper(w, "-c")
    assert code == 0
    assert "_out_file=myjob-run1.out" in stdout


def test_continue_without_prior_warns_and_starts_run0(tmp_path):
    """--continue with no prior -runN -> warn + start at -run0
    (defensive fallback so a user who passes --continue on a fresh
    dir gets a useful run instead of an error)."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    stdout, stderr, code = _run_wrapper(w, "--continue")
    assert code == 0
    assert "no prior -runN.out found" in stderr
    assert "_out_file=myjob-run0.out" in stdout


def test_continue_combines_with_np_for_siesta(tmp_path):
    """``--continue -np 8`` works in either order; -np reaches the
    SIESTA-specific parser AFTER --continue is stripped."""
    w = _emit_truncated_wrapper(tmp_path, "myjob")
    # Just exercise that the wrapper doesn't crash on the combined
    # arg pattern; the truncated wrapper doesn't actually use $_mpi_np.
    stdout, _stderr, code = _run_wrapper(w, "--continue", "-np", "8")
    assert code == 0
    assert "_out_file=myjob-run0.out" in stdout


def test_help_flag_lists_continue_and_force(tmp_path):
    """``-h`` lists the new --continue / --force flags."""
    _bind()
    script = tmp_path / "myjob.fdf"
    script.write_text("# fake\n")
    # autodetect_override=True so the wrapper sources conda.sh before
    # ``conda activate`` -- otherwise the subprocess shell only has
    # conda as a binary on PATH and ``conda activate`` errors with
    # "Run 'conda init' before 'conda activate'".
    wrapper_path = write_run_wrapper(script, autodetect_override=True)
    bash = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run(
        [bash, str(wrapper_path), "-h"],
        capture_output=True, text=True, timeout=10,
    )
    # -h prints to stdout per the cat<<USAGE redirect.
    assert "--continue" in proc.stdout
    assert "--force" in proc.stdout


# ---------------------------------------------------------------------
# PySCF wrapper: same --continue / --force semantics + its own arg block
# ---------------------------------------------------------------------


def test_pyscf_wrapper_redirects_via_out_file(tmp_path):
    """PySCF wrapper redirects stdout to ``$_out_file`` — which
    now resolves to ``<basename>-runN.pyscf.log`` instead of
    ``.out`` so the Results-tab dispatcher can tell PySCF output
    apart from SIESTA's (Phase C, 2026-06-07)."""
    _bind()
    script = tmp_path / "myjob.py"
    script.write_text("# fake\n")
    wrapper_path = write_run_wrapper(script)
    text = wrapper_path.read_text()
    assert "python myjob.py > $_out_file 2>&1" in text
    assert '_out_file="myjob-run${_run_n}.pyscf.log"' in text


def test_pyscf_wrapper_emits_continue_args_block(tmp_path):
    """PySCF wrapper exposes --continue / --force / -h identically to
    SIESTA -- this is the shared cross-engine contract."""
    _bind()
    script = tmp_path / "myjob.py"
    script.write_text("# fake\n")
    wrapper_path = write_run_wrapper(script)
    text = wrapper_path.read_text()
    # 2026-06-14: ``--cold``/``--from-scratch`` joined the cross-
    # engine arg set; the case labels are aligned in the bash
    # source so the text now has a different whitespace pattern
    # but the same per-flag substring.
    assert "--continue|-c)" in text and "_continue=1" in text
    assert "--force|-f)" in text and "_force=1" in text
    assert "--cold|--from-scratch)" in text and "_cold=1" in text
    # The help is wired up.
    assert "--continue, -c" in text
    assert "--force, -f" in text
    assert "--cold," in text and "--from-scratch" in text


def test_pyscf_wrapper_continue_advances_run_index(tmp_path):
    """End-to-end bash check on PySCF wrapper (same resolver code as
    SIESTA, but PySCF outputs land in ``.pyscf.log`` instead of
    ``.out``)."""
    w = _emit_truncated_wrapper(tmp_path, "myjob", suffix=".py")
    (tmp_path / "myjob-run0.pyscf.log").write_text("prior")
    stdout, _stderr, code = _run_wrapper(w, "--continue")
    assert code == 0
    assert "_out_file=myjob-run1.pyscf.log" in stdout


def test_pyscf_wrapper_refuses_overwrite_without_force(tmp_path):
    """No --continue, no --force, prior run exists -> refuse."""
    w = _emit_truncated_wrapper(tmp_path, "myjob", suffix=".py")
    (tmp_path / "myjob-run0.pyscf.log").write_text("prior")
    _stdout, stderr, code = _run_wrapper(w)
    assert code == 1
    assert "previous output exists" in stderr


def test_pyscf_wrapper_does_not_collide_with_siesta_out(tmp_path):
    """A directory that has BOTH a SIESTA-style ``-run0.out`` AND
    no PySCF ``-run0.pyscf.log`` must still start the PySCF wrapper
    at -run0 — the resolver only scans for the engine-specific
    suffix.  Pins the no-collision guarantee Phase C added (the
    whole point of the rename)."""
    w = _emit_truncated_wrapper(tmp_path, "myjob", suffix=".py")
    # A stale SIESTA-style .out file from a hypothetical earlier
    # SIESTA run in the same dir — must NOT confuse the PySCF
    # resolver.
    (tmp_path / "myjob-run0.out").write_text("SIESTA-style prior")
    stdout, _stderr, code = _run_wrapper(w)
    assert code == 0, "fresh PySCF run should start regardless of stale .out"
    assert "_out_file=myjob-run0.pyscf.log" in stdout


def test_pyscf_wrapper_banner_mentions_pyscf_log_not_out(tmp_path):
    """BOMB-6 fix (2026-06-07): the wrapper's "Run directly:" banner
    in the header docstring says ``-run0.pyscf.log`` for PySCF
    scripts and ``-run0.out`` for SIESTA scripts.  Pre-fix the
    banner always said ``.out`` regardless of engine, so users
    of the PySCF wrapper went hunting for ``myjob-run0.out``
    files that didn't exist (the actual emission is .pyscf.log
    since Phase C).  The launch command emit inside the wrapper
    body already used $_out_file correctly — only the human-
    readable banner was stale."""
    _bind()
    script = tmp_path / "myjob.py"
    script.write_text("# fake\n")
    wrapper_path = write_run_wrapper(script)
    text = wrapper_path.read_text()
    assert "first run -> -run0.pyscf.log" in text, (
        "PySCF wrapper banner must show the .pyscf.log suffix"
    )
    assert "first run -> -run0.out" not in text, (
        "PySCF wrapper banner must NOT show the legacy .out suffix"
    )


def test_siesta_wrapper_banner_still_mentions_out(tmp_path):
    """Flip side of the BOMB-6 fix: SIESTA wrappers keep saying
    ``.out`` because their output extension is still ``.out``."""
    _bind()
    script = tmp_path / "myjob.fdf"
    script.write_text("SystemLabel myjob\n")
    wrapper_path = write_run_wrapper(script)
    text = wrapper_path.read_text()
    assert "first run -> -run0.out" in text
    assert "first run -> -run0.pyscf.log" not in text


# --- bash -n syntax-check gates ---------------------------------------- #


def test_pyscf_wrapper_passes_bash_n(tmp_path):
    """Regression for the 2026-06-20 PDT incident: the PySCF wrapper's
    warm-label-extraction line emitted a broken ``awk -F'["\\'"]'``
    which left bash with an unterminated DQ.  The user only found out
    when they tried to run the generated script.

    There was an analogous ``bash -n`` test for the SIESTA branch
    (``test_siesta_enable_gpu.py:test_gpu_wrapper_keeps_siesta_
    template_intact``) but NONE for the PySCF branch — the gap that
    let the bug ship.  This test closes that gap.

    Belt-and-braces: :func:`write_run_wrapper` also self-checks the
    rendered text via ``bash -n`` internally now (raises
    :exc:`WrapperError` on failure), so if the generator regresses
    this assertion will fire BEFORE the file is written to disk —
    but the post-write check here is still useful as a contract
    pin on the public surface.
    """
    _bind()
    import subprocess
    script = tmp_path / "pyscf_relax.py"
    # Use a non-trivial JOB literal with both double quotes (canonical)
    # so the awk -F regex actually has to handle them.
    script.write_text('JOB = "pdt-mol"\nimport pyscf\n')
    wrapper_path = write_run_wrapper(script)
    cp = subprocess.run(
        ["bash", "-n", str(wrapper_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert cp.returncode == 0, (
        f"PySCF wrapper failed bash -n syntax check.  This is the "
        f"2026-06-20 PDT-incident class of bug — a generator template "
        f"emitting invalid shell.  bash stderr:\n{cp.stderr}"
    )


def test_pyscf_wrapper_passes_bash_n_single_quoted_job(tmp_path):
    """Same as above but the JOB literal uses single quotes
    (``JOB = 'pdt-mol'``).  The awk -F character class must split on
    BOTH ``"`` and ``'`` — pinning the SQ variant catches a half-fix
    that only handled DQ."""
    _bind()
    import subprocess
    script = tmp_path / "pyscf_relax.py"
    script.write_text("JOB = 'pdt-mol'\nimport pyscf\n")
    wrapper_path = write_run_wrapper(script)
    cp = subprocess.run(
        ["bash", "-n", str(wrapper_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert cp.returncode == 0, (
        f"PySCF wrapper (SQ JOB) failed bash -n syntax check.  "
        f"bash stderr:\n{cp.stderr}"
    )


def test_siesta_wrapper_passes_bash_n(tmp_path):
    """Symmetric coverage for SIESTA — duplicates the assertion that
    :mod:`tests.test_siesta_enable_gpu` makes but in the file readers
    expect (``test_runwrap.py``).  Cheap insurance against a future
    refactor that breaks SIESTA template rendering."""
    _bind()
    import subprocess
    script = tmp_path / "myjob.fdf"
    script.write_text("SystemLabel myjob\nNumberOfAtoms 0\n")
    wrapper_path = write_run_wrapper(script)
    cp = subprocess.run(
        ["bash", "-n", str(wrapper_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert cp.returncode == 0, (
        f"SIESTA wrapper failed bash -n syntax check.  bash stderr:\n"
        f"{cp.stderr}"
    )


# --------------------------------------------------------------------- #
#  Task #539: PySCF --cold glob covers the full warm-restart inventory  #
#                                                                       #
#  Per docs/protocols/script-execution.md "Required tests" table, every #
#  engine must have a test that pins ``--cold`` aside-glob coverage     #
#  against the warm-restart inventory.  SIESTA's equivalent shipped     #
#  2026-06-14 (task #438).  This file pairs with the generator-side    #
#  geometry warm-restart hook in tests/test_pyscf.py.                  #
# --------------------------------------------------------------------- #


# Authoritative inventory: docs/protocols/script-execution.md § PySCF
# warm-restart file table.  Update BOTH the doc + this tuple in the
# same commit when a new warm-restart hook lands -- the parity is the
# whole point of pinning it here.
_PYSCF_WARM_RESTART_INVENTORY = (
    ".chk",                # SCF DM init guess
    "_optimized.xyz",      # geometry warm-restart hook (#539 generator)
    "_geom_optim.xyz",     # geomeTRIC trajectory (append-mode hazard)
    "_geom_optim.tmp",     # geomeTRIC checkpoint
    "_geom.tmp",           # geomeTRIC checkpoint
)


def test_pyscf_cold_aside_block_covers_full_warm_restart_inventory():
    """The PySCF ``--cold`` glob must move EVERY file in the
    warm-restart inventory aside (per script-execution.md "Required
    tests" item 2).  A future warm-restart hook added in the
    generator without a matching glob entry here would silently
    state-leak: the user runs ``--cold`` expecting a fresh start but
    geomeTRIC's append-mode (or the script's _atom_block override)
    picks up the prior run's file.

    Pinned per-suffix in BOTH the JOB-keyed (``$_warm_label``) form
    and the wrapper-basename-keyed fallback so a JOB-vs-basename
    mismatch can't slip a file past the move-aside step.
    """
    from molbuilder.runwrap import _cold_restart_aside_block
    basename = "myjob"
    block = _cold_restart_aside_block(basename, engine="pyscf")
    for suffix in _PYSCF_WARM_RESTART_INVENTORY:
        # JOB-keyed form: ``"${_warm_label}.chk"`` etc.  Braces are
        # load-bearing for suffixes starting with ``_`` (bash would
        # otherwise absorb the suffix into the variable name and
        # trip ``set -u``).  Quoted so a JOB string with whitespace
        # doesn't word-split.
        assert f'"${{_warm_label}}{suffix}"' in block, (
            f"PySCF --cold glob is missing JOB-keyed entry "
            f"${{_warm_label}}{suffix}; design.md "
            f'"Generator-side warm-restart contract" requires every '
            f"warm-restart hook ship its --cold glob entry in the "
            f"same commit (with braced ${{...}} expansion to handle "
            f"underscore-prefixed suffixes safely)")
        # Wrapper-basename fallback: ``myjob.chk`` etc.  Catches the
        # case where the script's JOB string equals the wrapper
        # basename and the SystemLabel/JOB extract returned the
        # default.
        assert f"{basename}{suffix}" in block, (
            f"PySCF --cold glob is missing basename-keyed entry "
            f"{basename}{suffix}; covers the JOB=='basename' case")


def test_pyscf_cold_aside_block_does_not_glob_unrelated_files():
    """Defense in depth: the PySCF cold block must not accidentally
    glob files that aren't in the warm-restart inventory.  Catches
    a copy-paste regression where the SIESTA extensions (.DM, .XV,
    .CG, .HSX, .WFSX, ...) get pasted into the PySCF branch and the
    runwrap would happily delete unrelated SIESTA artifacts in a
    mixed-engine project directory."""
    from molbuilder.runwrap import _cold_restart_aside_block
    block = _cold_restart_aside_block("myjob", engine="pyscf")
    # SIESTA-only extensions must NOT appear in the PySCF block.
    for ext in ("DM", "XV", "CG", "LWF", "HSX", "WFSX", "TSHS"):
        assert f"myjob.{ext}" not in block, (
            f"PySCF cold-restart block contains SIESTA-only "
            f"extension .{ext}; cross-engine leak risk")


def test_pyscf_wrapper_with_full_inventory_passes_bash_n(tmp_path):
    """End-to-end syntax check: the rendered PySCF wrapper, with
    the expanded inventory glob, must still pass ``bash -n``.  A
    quoting / interpolation bug in the multi-suffix expansion would
    fail bash parse + we'd never want to ship that wrapper.
    Catches the pre-2026-06-20 PySCF SQ-escape malformed pattern
    class (where awk's char class shipped unterminated DQ)."""
    _bind()
    import subprocess
    script = tmp_path / "myjob.py"
    script.write_text('JOB = "myjob"\nimport pyscf\n')
    wrapper_path = write_run_wrapper(script)
    cp = subprocess.run(
        ["bash", "-n", str(wrapper_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert cp.returncode == 0, (
        f"PySCF wrapper with full warm-restart inventory failed "
        f"bash -n syntax check.  bash stderr:\n{cp.stderr}"
    )


def test_pyscf_cold_actually_moves_optimized_xyz_aside(tmp_path):
    """End-to-end behavior: extract the cold-restart bash block from
    the runwrap emitter, plant ALL five warm-restart files (each
    with a distinct sentinel), run the block with ``_cold=1`` under
    bash, and assert every file got moved into the dated aside dir.

    Runs the actual bash block, not a model of it -- catches a bug
    where the glob entries are present but the move-aside loop has
    a typo that causes the actual ``mv`` to skip a file (the
    "branch present but never fires" class from design.md Required
    tests table).

    Uses bash subprocess directly (not the truncated wrapper helper
    -- that cuts at the run-index resolver, BEFORE the cold block
    runs in the rendered wrapper, so it can't exercise this path).
    """
    from molbuilder.runwrap import _cold_restart_aside_block
    job = "myjob"
    block = _cold_restart_aside_block(job, engine="pyscf")

    # The block reads JOB= from the .py script to populate
    # _warm_label.  Plant a minimal script alongside the warm-
    # restart files.
    (tmp_path / f"{job}.py").write_text(f'JOB = "{job}"\n')

    # Plant the warm-restart inventory.
    for suffix in _PYSCF_WARM_RESTART_INVENTORY:
        (tmp_path / f"{job}{suffix}").write_text(f"sentinel-{suffix}")

    # Wrap the block with the minimal harness it expects: ``_cold=1``
    # to trigger the move + the same ``set -euo pipefail`` shape the
    # real wrapper uses.  The block uses ``shopt -s nullglob``
    # internally so non-matching globs don't trip ``set -e``.
    harness = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "_cold=1\n"
        f"{block}\n"
    )
    script_path = tmp_path / "cold_block.sh"
    script_path.write_text(harness)
    script_path.chmod(0o755)

    bash = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run(
        [bash, str(script_path)],
        cwd=str(tmp_path),
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, (
        f"cold-restart block exited non-zero;\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    # No warm-restart files remain in cwd.
    for suffix in _PYSCF_WARM_RESTART_INVENTORY:
        assert not (tmp_path / f"{job}{suffix}").exists(), (
            f"--cold did not move {job}{suffix} aside; design.md "
            f"warm-restart contract violated")

    # Aside dir exists (dated), contains every planted file.
    aside_dirs = list(tmp_path.glob(f"{job}-restart-aside-*"))
    assert len(aside_dirs) == 1, (
        f"expected exactly one aside dir; got {aside_dirs}")
    moved = sorted(p.name for p in aside_dirs[0].iterdir())
    expected = sorted(f"{job}{s}" for s in _PYSCF_WARM_RESTART_INVENTORY)
    assert moved == expected, (
        f"aside dir contents diverge from planted inventory; "
        f"got {moved}, expected {expected}")
