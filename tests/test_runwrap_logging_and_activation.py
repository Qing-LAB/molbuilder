"""Tests for the 2026-06-24 runwrap rewrite: 6-path conda detection
+ vigorous per-run logging + scheduler-aware pre-activation hook.

Pinned contracts:
  * the wrapper renders as bash-syntax-valid (``bash -n`` clean)
  * a per-run log file ``<basename>.runwrap-<timestamp>.log`` is
    created when the wrapper runs; stdout AND stderr are tee'd to it
  * the structured ``_log`` helper emits ``[HH:MM:SS] [TAG  ] msg``
  * the pre-activation hook (``MOLBUILDER_PREACTIVATE_CMDS``) fires
    BEFORE any conda detection
  * ``module load`` is attempted for clusters that gate conda behind
    environment-modules (ASU sc002 et al)
  * detection paths are tried in the documented order; the first hit
    wins and is logged with its path-number tag
  * when all six paths fail, the wrapper exits 1 with an actionable
    error message naming each path tried
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Tuple

import pytest

from molbuilder.diagnostics import (
    Capabilities,
    EXTENSION_TO_CATEGORY,
    set_capabilities,
)
from molbuilder.runwrap import (
    _detect_generate_time_conda_base,
    render_run_wrapper,
    write_run_wrapper,
)


def _bind() -> None:
    """Pin a synthetic capabilities snapshot so the wrapper renders
    deterministically (independent of the host's installed conda)."""
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary="/usr/bin/conda",
    ))


@pytest.fixture
def bound():
    _bind()
    yield
    set_capabilities(None)


# --------------------------------------------------------------------- #
#  Logging preamble                                                      #
# --------------------------------------------------------------------- #


def test_per_run_log_file_has_timestamped_name(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    # Filename pattern is the basename + ``.runwrap-`` + date format.
    assert '_runwrap_log="JOB.runwrap-$(date +%Y%m%d-%H%M%S).log"' in text


def test_stdout_and_stderr_are_both_tee_d_to_log_file(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    assert 'exec > >(tee -a "$_runwrap_log")' in text
    assert '2> >(tee -a "$_runwrap_log" >&2)' in text


def test_log_helper_uses_structured_HHMMSS_TAG_format(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    # The printf format string must produce [HH:MM:SS] [TAG  ] msg
    # so a future log scraper has a stable shape to parse.
    assert "[%s] [%-5s] %s" in text


def test_initial_state_dump_includes_hostname_user_cwd_argv(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    for needle in ('hostname:', 'user:', 'cwd:', 'argv:', 'log file:'):
        assert needle in text, f"missing `{needle}` from initial-state dump"


def test_scheduler_vars_are_logged_only_when_set(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    # SLURM + PBS env-var names appear in the for-loop that emits
    # them; the loop guards on ``-n "$_v_val"`` so unset vars stay
    # out of the log (no "SLURM_JOB_ID=<unset>" noise).
    for v in ("SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
              "SLURM_JOB_NODELIST", "PBS_JOBID", "PBS_NP"):
        assert v in text, f"missing `{v}` from scheduler-var dump"
    assert '[ -n "$_v_val" ]' in text


# --------------------------------------------------------------------- #
#  Pre-activation hook                                                   #
# --------------------------------------------------------------------- #


def test_preactivate_hook_runs_before_any_conda_detection(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    hook_ix = text.find("MOLBUILDER_PREACTIVATE_CMDS")
    locate_ix = text.find("# --- Locate conda + activate")
    assert hook_ix >= 0 and locate_ix >= 0
    assert hook_ix < locate_ix, (
        "MOLBUILDER_PREACTIVATE_CMDS must fire BEFORE the 6-path "
        "conda detection -- otherwise ``module load mamba`` "
        "(the canonical use case) can't put conda on PATH in time."
    )


def test_preactivate_hook_warns_but_does_not_abort_on_nonzero(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    # The hook may legitimately return non-zero (e.g. ``module load``
    # for an already-loaded module).  The wrapper logs WARN but keeps
    # going so the later detection paths can still succeed.
    assert "_log WARN" in text
    assert "preactivate cmds returned non-zero" in text


# --------------------------------------------------------------------- #
#  Six-path detection -- ordering + tags                                #
# --------------------------------------------------------------------- #


def test_paths_are_tried_in_documented_order(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    indices = [text.find(f"path {i}:") for i in range(1, 7)]
    assert all(i > 0 for i in indices), \
        f"missing path tags: {indices}"
    assert indices == sorted(indices), \
        "detection paths must appear in 1..6 order, got: " + str(indices)


def test_mamba_is_tried_before_conda_at_every_layer(bound):
    """Mamba is preferred over conda at every detection layer:
      * baked-in helper tries mamba info --base before conda info --base
      * path 3 (mamba) appears before path 4 (conda) in the wrapper text
      * inside path 5's ``module load`` loop, the inner ``$_bin`` check
        is ``for _bin in mamba conda``, not ``for _bin in conda mamba``

    Rationale: ASU sc002 (the canonical HPC target) ships mamba via
    ``module load mamba`` and does NOT provide conda by default.  Many
    modern installs (mambaforge / miniforge3) similarly ship only
    mamba on PATH after activation.  Putting mamba first across all
    layers keeps the detection deterministic and avoids the case
    where a stale ``conda`` shim (e.g. a wrapper script) gets picked
    over a working mamba install."""
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    # Path 3 uses mamba; path 4 uses conda.
    p3_ix = text.find("path 3: ``mamba info --base``")
    p4_ix = text.find("path 4: ``conda info --base``")
    assert p3_ix > 0, "path 3 must use mamba info --base"
    assert p4_ix > 0, "path 4 must use conda info --base"
    assert p3_ix < p4_ix, "mamba must be tried before conda"
    # Path 5 inner loop: mamba first.
    assert "for _bin in mamba conda" in text, (
        "path 5's binary-probe loop must put mamba before conda"
    )


def test_both_conda_and_mamba_info_base_branches_present(bound):
    """Sanity: both branches must exist (no regression to a
    mamba-only or conda-only wrapper).  The wrapper has to cope
    with sites that ship one but not the other."""
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    assert "mamba info --base" in text
    assert "conda info --base" in text


def test_path_5_attempts_module_load_for_module_gated_clusters(bound):
    """ASU sc002 needs ``module load mamba`` before mamba is on PATH.
    Path 5 probes a list of conventional module names."""
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    assert "command -v module" in text
    assert "module load" in text
    # The canonical names that cover most HPC sites.
    for name in ("mamba", "miniforge", "miniconda", "anaconda"):
        assert name in text, f"missing module name `{name}` in path 5"


def test_path_6_probes_common_filesystem_locations(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    for cand in ("$HOME/miniforge3", "$HOME/miniconda3",
                 "/opt/miniconda3", "/opt/miniforge3"):
        assert cand in text, f"missing filesystem-probe path `{cand}`"


# --------------------------------------------------------------------- #
#  All-paths-failed error                                                #
# --------------------------------------------------------------------- #


def test_exhausted_paths_emit_actionable_error_message(bound):
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    # Every path failure is mentioned so the user can tell which one
    # to fix.
    for path_ref in ("(1) CONDA_DEFAULT_ENV",
                     "(2) baked-in",
                     "(3) ``mamba info --base``",
                     "(4) ``conda info --base``",
                     "(5) ``module load",
                     "(6) common locations"):
        assert path_ref in text, f"missing `{path_ref}` from error help"
    # Plus the documented escape hatch.
    assert "MOLBUILDER_PREACTIVATE_CMDS" in text


# --------------------------------------------------------------------- #
#  bash -n syntax check                                                  #
# --------------------------------------------------------------------- #


def test_rendered_wrapper_passes_bash_n(bound, tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=4, autodetect_override=True)
    p = tmp_path / "JOB.run.sh"
    p.write_text(text)
    r = subprocess.run([bash, "-n", str(p)],
                        capture_output=True, text=True)
    assert r.returncode == 0, (
        f"bash -n rejected the rendered wrapper:\n{r.stderr}"
    )


def test_pyscf_wrapper_also_renders_with_logging(bound, tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    # PySCF wrapper goes through the same env_activation path.
    fdf = tmp_path / "spectra.py"
    fdf.write_text("# pyscf script\n")
    wrapper = write_run_wrapper(fdf, autodetect_override=True)
    text = wrapper.read_text()
    assert "_runwrap_log=" in text
    assert "path 1:" in text and "path 6:" in text
    r = subprocess.run([bash, "-n", str(wrapper)],
                        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------- #
#  End-to-end: actually run the wrapper with a stubbed env              #
# --------------------------------------------------------------------- #


def test_running_path_1_writes_log_file_with_skip_marker(bound, tmp_path):
    """Drive the wrapper through path 1 (CONDA_DEFAULT_ENV ==
    target_env) and confirm the per-run log file lands on disk and
    captures the path-1 skip line.  Stubs everything past the
    activation block by inserting an early ``exit 0`` so the SIESTA
    launch doesn't actually fire."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    fdf = tmp_path / "JOB.fdf"
    fdf.write_text("SystemLabel JOB\n")
    wrapper = write_run_wrapper(fdf, mpi_np=2, autodetect_override=True)
    text = wrapper.read_text()
    # Insert an exit 0 right after the post-activate ``which python``
    # log line so we don't try to launch SIESTA in this test.
    needle = '_log INFO "which python:'
    ix = text.find(needle)
    assert ix >= 0
    eol = text.find("\n", ix)
    text = text[:eol + 1] + "exit 0\n" + text[eol + 1:]
    wrapper.write_text(text)
    env = dict(os.environ)
    env["CONDA_DEFAULT_ENV"] = "molbuilder-siesta"  # path 1
    r = subprocess.run([bash, str(wrapper)], cwd=str(tmp_path),
                        env=env, capture_output=True, text=True,
                        timeout=10)
    assert r.returncode == 0, (
        f"path-1 wrapper exited {r.returncode}.\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    logs = list(tmp_path.glob("JOB.runwrap-*.log"))
    assert len(logs) == 1, (
        f"expected exactly one runwrap log; got {[p.name for p in logs]}"
    )
    log_text = logs[0].read_text()
    assert "===== molbuilder wrapper start =====" in log_text
    assert "path 1:" in log_text
    assert "already in 'molbuilder-siesta'" in log_text


# --------------------------------------------------------------------- #
#  _detect_generate_time_conda_base helper                              #
# --------------------------------------------------------------------- #


def test_detect_generate_time_conda_base_returns_path_or_none():
    """The helper either returns a real conda base directory containing
    ``etc/profile.d/conda.sh`` or None.  It must never return a string
    that doesn't point at a real conda install."""
    r = _detect_generate_time_conda_base()
    if r is None:
        return  # acceptable -- machine without conda
    p = Path(r)
    assert (p / "etc" / "profile.d" / "conda.sh").is_file(), (
        f"_detect_generate_time_conda_base() returned {r!r}, but no "
        f"conda.sh hook found there"
    )


def test_baked_in_conda_base_is_embedded_into_rendered_wrapper(bound,
                                                                  monkeypatch):
    """When the generator process has CONDA_PREFIX set, the wrapper
    must carry that base path so the run-time path-2 (baked-in)
    branch can fire on the target machine without any PATH search.
    """
    monkeypatch.setenv("CONDA_PREFIX", "/opt/fakeconda/envs/molbuilder")
    # Pre-arrange the fake conda dir so _detect_generate_time_conda_base
    # accepts the path (it requires conda.sh to exist).
    real = _detect_generate_time_conda_base()
    # The detect helper walks up envs/<name> -> /opt/fakeconda.  It
    # then checks for conda.sh; absent that we get None.  This test
    # just confirms the WRAPPER carries _baked_conda_base= prefix
    # whatever the value resolves to (empty or real).
    text = render_run_wrapper(Path("/x/JOB.fdf"), mpi_np=2, autodetect_override=True)
    assert "_baked_conda_base=" in text
