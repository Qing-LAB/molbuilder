"""The build probe cannot hang the job it was about to describe.

Before launching, the SIESTA wrapper asks the binary what it is::

    siesta --version   ->   Version         : 5.4.2
                            Parallelisations: MPI

and picks its launcher from the answer.  The call was written as
``$(siesta --version 2>/dev/null || true)`` — which handles a probe that
FAILS and cannot handle one that never RETURNS.

**SIESTA reads its deck from standard input.**  A build that does not
recognise ``--version`` therefore does not error: it waits for a deck.
Forever.  The wrapper stops there, before the engine, having logged
nothing since its banner — so on a cluster that is a queue slot spent, a
wall reached, and no output naming a cause.  Found live on qlabsrv
2026-08-25, where a root-owned 2023 ``/usr/local/bin/siesta`` does exactly
this; 28 blocked probes had accumulated from test runs alone.

Three things are needed, and each was found only after the previous one
failed to fix it:

1. a **clock** — closing stdin is not enough, measured: that binary blocks
   with stdin at ``/dev/null`` too;
2. a **file, not a pipe** — the clock is not enough either.  ``timeout``
   signals the process it started; the probe forks, and the child outlives
   the signal still holding the write end, so ``$( )`` waits on a pipe that
   never sees EOF.  The wrapper hangs AFTER the clock has already fired.
3. an **empty answer meaning "probe failed"**, which the launcher choice
   already treats as ``mpirun`` — the documented safe default.

This file is separate from ``test_runwrap_cold_restart.py`` because that
suite now stubs a WELL-BEHAVED ``siesta`` on PATH (it must, or it
interrogates whatever the host has installed).  A guard against hanging
cannot be tested by a stub that never hangs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.jobset.model import Resources
from molbuilder.runwrap import write_run_wrapper

pytestmark = pytest.mark.skipif(
    not shutil.which("bash") or not shutil.which("timeout"),
    reason="needs bash + coreutils timeout")

#: Generous next to the wrapper's own 5 s bound, and far under the time a
#: hanging probe would take (it never returns).  A failure here is a hang,
#: not a slow machine.
_ALLOW_S = 60


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    """A rendered SIESTA wrapper, cut off before the engine launch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preamble": "module load mamba",
                              "activation": "source activate"}}))
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset(["molbuilder-siesta"])))
    deck = tmp_path / "myjob.fdf"
    deck.write_text("SystemLabel myjob\nNumberOfAtoms 1\n"
                    "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
                    "%endblock AtomicCoordinatesAndAtomicSpecies\n")
    w = write_run_wrapper(deck, resources=Resources())
    text = w.read_text()
    # Drop the bootstrap (no conda in a bare shell) and stop before the
    # engine launch: the probe sits between them, which is the point.
    pre = text.find("# --- Baked preamble")
    start = text.rfind('if [ "$_mb_help" = "0" ]; then', 0, pre)
    close = text.find("\nfi\n", text.find("which python:", pre))
    text = text[:start] + "set -u\n" + text[close + 4:]
    cut = text.find("mpirun")
    # Echo the wrapper's OWN parsed value, so a test can assert what the
    # probe put in it rather than that the script merely survived.
    w.write_text(text[:cut] + '\necho "PROBE_VER=${_siesta_ver:-}"\nexit 0\n')
    return w


def _siesta(bin_dir: Path, body: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    s = bin_dir / "siesta"
    s.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    s.chmod(0o755)


def _run(wrapper: Path, bin_dir: Path):
    env = {**os.environ, "MB_LAUNCHED_BY": "manual",
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    t0 = time.time()
    proc = subprocess.run(["bash", str(wrapper)], cwd=wrapper.parent,
                          capture_output=True, text=True,
                          timeout=_ALLOW_S, env=env)
    return proc, time.time() - t0


def test_a_probe_that_never_answers_does_not_hang_the_wrapper(
        wrapper, tmp_path):
    """THE LIVE DEFECT.  A binary that waits for a deck instead of printing
    a version must cost the wrapper its bound and nothing more."""
    bin_dir = tmp_path / "bin"
    _siesta(bin_dir, 'if [ "${1:-}" = "--version" ]; then sleep 300; fi\n'
                     'echo "stub ran: $*"\n')
    proc, took = _run(wrapper, bin_dir)
    assert proc.returncode == 0, (
        f"wrapper exited {proc.returncode}\n{proc.stderr[-2000:]}")
    assert took < _ALLOW_S, f"the wrapper took {took:.1f}s"
    assert "PROBE_VER=" in (proc.stdout + proc.stderr)
    assert "PROBE_VER=5" not in (proc.stdout + proc.stderr), (
        "an unanswered probe must leave the version EMPTY -- that is what "
        "the launcher choice reads as 'probe failed' -> mpirun")


def test_a_probe_that_forks_does_not_hold_the_wrapper_open(
        wrapper, tmp_path):
    """The half a clock alone does not fix.  ``timeout`` signals the process
    it started; a CHILD that outlives it still holds the write end, and a
    ``$( )`` capture waits on a pipe that never closes.  Writing the probe's
    output to a FILE is what makes this survivable -- so this stub forks a
    child that outlives the signal and keeps the inherited stdout."""
    bin_dir = tmp_path / "bin"
    _siesta(bin_dir,
            'if [ "${1:-}" = "--version" ]; then\n'
            '    sleep 300 &\n'          # holds the inherited stdout
            '    trap "" TERM\n'
            '    sleep 300\n'
            'fi\n'
            'echo "stub ran: $*"\n')
    proc, took = _run(wrapper, bin_dir)
    assert proc.returncode == 0, (
        f"wrapper exited {proc.returncode}\n{proc.stderr[-2000:]}")
    assert took < _ALLOW_S, f"the wrapper took {took:.1f}s"


def test_a_working_probe_is_still_read(wrapper, tmp_path):
    """The bound must not cost the answer: a binary that reports normally
    is still parsed, or the launcher choice loses the fact it exists for."""
    bin_dir = tmp_path / "bin"
    _siesta(bin_dir,
            'if [ "${1:-}" = "--version" ]; then\n'
            '    echo "Version         : 5.4.2-stub"\n'
            '    echo "Parallelisations: MPI"\n'
            '    exit 0\n'
            'fi\n'
            'echo "stub ran: $*"\n')
    proc, _ = _run(wrapper, bin_dir)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "PROBE_VER=5.4.2-stub" in (proc.stdout + proc.stderr), (
        "the version the probe reported never reached the wrapper")
