"""Tests for the wrapper-independence rewrite (docs/execution/running-a-job.md § 5 v2).

The wrapper:
  * does not change cwd
  * has no runtime config-file reads
  * has no env-var-driven behaviour switching (no MOLBUILDER_PREACTIVATE_CMDS)
  * has no runtime detection (no 6-path block, no autodetect)
  * bakes ``preamble`` verbatim from the target machine's record
  * bakes a literal ``<activation_form> <env_name>`` line
  * refuses to emit when the record states no activation (the record is
    their one home since 2026-10-02 -- `configuration.md` § 5 M-1)

These tests pin those properties on the rendered shell text.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from molbuilder.diagnostics import (Capabilities, EXTENSION_TO_CATEGORY,
                                     set_capabilities)
from molbuilder.runwrap import (WrapperError, render_run_wrapper,
                                  write_run_wrapper)
from molbuilder.jobset.model import Resources
from molbuilder.scheduler import Environment, Topology


def _record(**script_generation):
    """A machine record stating how a shell enters an environment there --
    the activation and preamble's one home (`configuration.md` § 5 M-1)."""
    return Environment(scheduler="workstation", topology=Topology(),
                       script_generation=script_generation)


#: The canonical Sol setup, as `jobset probe --write --activation ...
#: --preamble ...` records it.
_SOL = _record(preamble="module load mamba\nexport FOO=bar",
               activation="source activate")

#: The launch shape every render here states (`architecture.md` § 5.2).
_SHAPE = Resources(mpi_np=4, cpus_per_task=1)


def _bind():
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary="/usr/bin/conda",
    ))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """This machine's record with the canonical Sol setup, at the config
    root, so a deck rendered under it reads it.  Also resets capabilities
    each test.
    """
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / "environment.json").write_text(_SOL.to_json())
    _bind()
    yield tmp_path
    set_capabilities(None)


# --------------------------------------------------------------------- #
#  Refuse-to-emit when activation isn't set                              #
# --------------------------------------------------------------------- #


def test_render_refuses_a_record_that_states_no_activation():
    """Per docs/execution/running-a-job.md § 5.2: a record with no
    ``activation`` -- a preamble alone does not count -> the generator
    refuses to emit a wrapper that can't activate its env.

    API-LEVEL because the road cannot reach it: prep refuses such a target
    first (`jobset.prep._require_activation`,
    `tests/data/launch_values.toml`).  This is the emitter's own guard, for
    a caller that reaches it directly."""
    _bind()
    try:
        with pytest.raises(WrapperError, match="activation has no default"):
            render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                               machine_record=_record(
                                   preamble="module load mamba"))
    finally:
        set_capabilities(None)


# --------------------------------------------------------------------- #
#  Wrapper does NOT cd                                                  #
# --------------------------------------------------------------------- #


def test_wrapper_does_not_cd(sandbox):
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    # No cd at the top of the wrapper.  Caller's cwd is the contract.
    assert "SLURM_SUBMIT_DIR" not in text or "cd " not in text.split(
        "SLURM_SUBMIT_DIR", 1)[0]
    assert 'cd "$(dirname "$0")"' not in text
    assert 'cd "$SLURM_SUBMIT_DIR"' not in text
    assert 'cd "$PBS_O_WORKDIR"' not in text


# --------------------------------------------------------------------- #
#  Preamble is baked verbatim with per-scope sentinels                  #
# --------------------------------------------------------------------- #


def test_preamble_baked_verbatim(sandbox):
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    # Each preamble line from the record appears literally.
    assert "module load mamba" in text
    assert "export FOO=bar" in text
    # And under a sentinel naming its source -- the target's record, the
    # one home (it read "SERVER PREAMBLE (from molbuilder.json)" until
    # 2026-10-02).
    assert "TARGET PREAMBLE (from the target machine's record)" in text


def test_empty_preamble_emits_placeholder():
    _bind()
    try:
        text = render_run_wrapper(
            Path("/x/JOB.fdf"), resources=_SHAPE,
            machine_record=_record(activation="source activate"))
        # The comment shows the empty case is explicit rather than silently
        # elided.  (It read "(none configured)" until 2026-10-02.)
        assert "(the target's record states no preamble)" in text
    finally:
        set_capabilities(None)


# --------------------------------------------------------------------- #
#  Activation is a single literal line                                  #
# --------------------------------------------------------------------- #


def test_activation_baked_as_single_line(sandbox):
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    # Literal one-liner.  No conda-activate-with-fallback, no
    # source-activate-with-fallback, no 6-path detection block.
    assert "source activate molbuilder-siesta" in text


def test_activation_form_from_the_record():
    """``activation: "conda activate"`` produces a ``conda activate``
    line, not ``source activate``."""
    _bind()
    try:
        text = render_run_wrapper(
            Path("/x/JOB.fdf"), resources=_SHAPE,
            machine_record=_record(activation="conda activate"))
        assert "conda activate molbuilder-siesta" in text
        assert "source activate molbuilder-siesta" not in text
    finally:
        set_capabilities(None)


# --------------------------------------------------------------------- #
#  Removed runtime behaviours -- pin that they're GONE                 #
# --------------------------------------------------------------------- #


def test_no_six_path_detection_block(sandbox):
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    for needle in ("path 1:", "path 2:", "path 3:", "path 4:",
                    "path 5:", "path 6:",
                    "mamba info --base", "conda info --base",
                    "module load mamba miniforge3",
                    "$HOME/miniforge3"):
        assert needle not in text, (
            f"runtime detection block leftover -- found {needle!r}"
        )


def test_no_molbuilder_preactivate_cmds_hook(sandbox):
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    # The runtime env-var hook is gone per docs/execution/running-a-job.md § 5
    assert "MOLBUILDER_PREACTIVATE_CMDS" not in text


def test_no_autodetect_field(sandbox):
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    assert "autodetect_conda" not in text


# --------------------------------------------------------------------- #
#  bash -n + end-to-end render                                          #
# --------------------------------------------------------------------- #


def test_rendered_wrapper_passes_bash_n(sandbox, tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    p = tmp_path / "JOB.run.sh"
    p.write_text(text)
    r = subprocess.run([bash, "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_write_run_wrapper_writes_chmod_x(sandbox, tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    fdf = tmp_path / "JOB.fdf"
    fdf.write_text("SystemLabel JOB\nNumberOfAtoms 2\n")
    p = write_run_wrapper(fdf, resources=Resources(mpi_np=2, cpus_per_task=1))
    assert p.stat().st_mode & stat.S_IXUSR
    r = subprocess.run([bash, "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------- #
#  Log file lands in cwd, not in the wrapper's own dir                  #
# --------------------------------------------------------------------- #


def test_log_filename_does_not_hardcode_directory(sandbox):
    """The session log lands in the INVOCATION's directory, never in the one
    the wrapper was generated in.

    This used to assert the literal relative filename.  It now resolves through
    ``$PWD``, which is the same directory -- ``$PWD`` is read before any ``cd``.

    That was originally required by ``attempt_dirs=True``, where the wrapper
    ``cd``'d into ``run-<n>/`` after the log was opened, so a RELATIVE
    ``$_runwrap_log`` resolved against the attempt, where the log is not.
    **P7 unit 1 retired that block and with it the only ``cd`` a wrapper ever
    did** (`job-contracts.md § 2.1`: the caller's working directory is the
    contract).  The ``$PWD`` form stays anyway, and not merely from inertia:
    it is what makes the wrapper independent of where it is invoked from,
    which is the property the launcher relies on when it runs a job inside an
    attempt directory that Python -- not bash -- created.

    What the original test was protecting -- no generation-time directory baked
    into the text -- is what is asserted here.
    """
    text = render_run_wrapper(Path("/x/JOB.fdf"), resources=_SHAPE,
                              machine_record=_SOL)
    assert 'JOB.runwrap-$(date +%Y%m%d-%H%M%S).log' in text
    assert "/x/" not in text, "the generation-time directory must not be baked in"
    assert '_runwrap_log="/' not in text, "not a literal absolute path"

    # Resolved at RUN time, against the invocation's own directory.
    assert '_runwrap_log="$PWD/JOB.runwrap-$(date +%Y%m%d-%H%M%S).log"' in text
