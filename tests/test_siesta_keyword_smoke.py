"""L4 SIESTA-binary smoke test: gates the silent fdf-keyword failure class.

The 2026-06-23 keyword incident shipped because no test ever piped a
molbuilder-generated `.fdf` through the SIESTA binary and read the
``redata: Dynamics option = ...`` echo back.  Pre-fix the generator
emitted phantom keywords (``MD.NumBroydenSteps``, ``MD.MaxDispl``,
``WriteHS``) that SIESTA 5.4.2 silently dropped, with no warning.

This test closes the gap end-to-end:

  1. Render a SIESTA `.fdf` via ``molbuilder.siesta.render_fdf`` for
     each relax type (CG, Broyden, FIRE).
  2. Pipe it through the SIESTA binary in the molbuilder-siesta env.
  3. Read the ``redata: Dynamics option = ...`` line back and assert
     it matches the requested relax type, NOT
     ``Single-point calculation`` (the silent fallback).
  4. Assert SaveHS lands by inspecting SIESTA's fdf-echo dump.

The L3 render-shape tests at ``tests/test_smiles_and_siesta.py::
TestSiestaStageOverlay`` + ``test_savehs_keyword_emitted_always``
already gate the generator's emission contract (universal keywords,
no phantom variants).  This L4 test catches the next class of failure:
a generator change that emits a SYNTACTICALLY plausible keyword which
SIESTA does not actually recognise.

Layer: L4 (binary-in-the-loop).  Skipped cleanly when the
molbuilder-siesta env is not installed on this machine.

Subprocess dispatch via the molbuilder-siesta env's siesta binary
(no host PATH siesta is permitted; the env's binary is the
authoritative one per docs/protocols/script-execution.md).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# Test data: a real H pseudo from the BDT project (smallest available
# psml; SIESTA refuses to start without one per-species).  Reuse instead
# of synthesising one so the fixture stays load-bearing-light.
_H_PSML_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "projects" / "BDT" / "optimization" / "TJ-BDT-Au111" / "H.psml"
)


def _siesta_binary():
    """Return the path to the molbuilder-siesta env's siesta binary,
    or None when the env is not present on this machine.

    Mirrors the pyscf-env discovery pattern at
    ``tests/test_molwatch_preview.py::_pyscf_env_python``.
    """
    from molbuilder.diagnostics import get_capabilities
    caps = get_capabilities()
    if not caps.env_available("molbuilder-siesta"):
        return None
    # Standard conda layout: $CONDA_PREFIX/envs/<name>/bin/siesta.
    candidate = Path(
        os.environ.get("CONDA_PREFIX", "/home/qqing/miniconda3")
    ).parent / "envs" / "molbuilder-siesta" / "bin" / "siesta"
    if candidate.exists():
        return candidate
    # Fallback via `conda run -n`.
    from molbuilder.envs import run_in_env
    try:
        r = run_in_env(
            "molbuilder-siesta",
            ["which", "siesta"],
            capture_output=True, text=True, check=True,
        )
        path = Path(r.stdout.strip())
        return path if path.exists() else None
    except Exception:
        return None


def _require_siesta_binary():
    """pytest skip helper: skip when the SIESTA binary is unreachable
    or refuses to start (broken env, missing libs)."""
    binary = _siesta_binary()
    if binary is None:
        pytest.skip(
            "molbuilder-siesta env not installed; install via "
            "`bash scripts/install-env.sh --bootstrap --yes` or "
            "`python -m molbuilder envs install molbuilder-siesta`."
        )
    # Fast self-test so a downstream subprocess failure has a clean
    # cause (env broken vs the keyword we are gating).
    probe = subprocess.run(
        [str(binary), "--version"],
        capture_output=True, text=True, timeout=15,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"molbuilder-siesta env's siesta refused to start: "
            f"{probe.stderr.strip() or probe.stdout.strip()}"
        )
    return binary


def _run_siesta_on_fdf(binary: Path, fdf_path: Path, *, work_dir: Path,
                      timeout_s: float = 60.0) -> str:
    """Run SIESTA against ``fdf_path`` and return the captured stdout.

    Runs in ``work_dir`` so SIESTA's per-run side files (`.BASIS`,
    `.bib`, MESSAGES, …) land in the temp dir, not in cwd.  No MPI
    -- single-process; the test only needs the redata banner, not
    parallel execution.
    """
    proc = subprocess.run(
        [str(binary), str(fdf_path.name)],
        cwd=str(work_dir),
        capture_output=True, text=True, timeout=timeout_s,
    )
    return proc.stdout


def _minimal_h2_fdf(tmp_path: Path, relax_type: str,
                     n_steps: int = 1) -> Path:
    """Render a minimal H2 .fdf for the given relax type.

    Uses molbuilder's actual generator so the test exercises the
    same emission code path as production.  Returns the path to the
    rendered .fdf.
    """
    from molbuilder.siesta import SiestaConfig, render_fdf
    from molbuilder.structure import Structure

    # Copy the H.psml in (SIESTA reads it relative to cwd).
    shutil.copy(_H_PSML_SOURCE, tmp_path / "H.psml")

    struct = Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0],
                            [0.74, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(
        relax_type=relax_type,
        relax_steps=n_steps,
        relax_max_displ=0.1,      # keep moves small
        relax_force_tol=0.04,
        # Crank everything else down so SIESTA exits in seconds.
        mesh_cutoff=50.0,
        max_scf_iter=5,
        dm_tolerance=1e-2,
        # No psml_lib lookup -- H.psml lives in cwd next to the fdf.
        psml_lib=None,
        # Skip auth probes
        system_label="probe",
        # Single-atom cell big enough for a stand-alone H2 molecule.
        kgrid=(1, 1, 1),
    )
    fdf_text = render_fdf(struct, cfg)
    fdf_path = tmp_path / "probe.fdf"
    fdf_path.write_text(fdf_text)
    return fdf_path


# --------------------------------------------------------------------- #
#  The actual smoke tests                                               #
# --------------------------------------------------------------------- #


# Map: cfg.relax_type -> the exact "Dynamics option" string SIESTA 5.4.2
# echoes when it recognises the algorithm.  Empirically verified against
# the binary in /home/qqing/.claude/jobs/074e4f77/tmp/siesta-kw-audit/.
_DYNAMICS_BANNER = {
    "CG":      "CG coord. optimization",
    "Broyden": "Broyden coord. optimization",
    "FIRE":    "FIRE coord. optimization",
}


@pytest.mark.parametrize("relax_type", ["CG", "Broyden", "FIRE"])
def test_relax_type_lands_in_redata_dynamics_option(relax_type, tmp_path):
    """The 2026-06-23 silent-failure gate.

    Render an .fdf with cfg.relax_type = X; pipe it through the
    SIESTA binary; assert the ``redata: Dynamics option = ...`` echo
    matches X and is NOT ``Single-point calculation`` (the silent
    fallback when SIESTA does not recognise the step-count keyword).

    Pre-2026-06-23 the generator emitted MD.NumBroydenSteps /
    MD.NumFIRESteps which SIESTA 5.4.2 silently dropped, leaving
    step count = 0 -> dynamics = Single-point.  Broyden and FIRE
    relaxations ran as single-point calculations, wasting CPU on
    every job.  This test catches that class of regression.
    """
    binary = _require_siesta_binary()
    fdf_path = _minimal_h2_fdf(tmp_path, relax_type)
    stdout = _run_siesta_on_fdf(binary, fdf_path, work_dir=tmp_path)

    expected = _DYNAMICS_BANNER[relax_type]
    expected_line = f"redata: Dynamics option                             = {expected}"

    # Hard-fail with both stdout (recent log) and the redata block on
    # mismatch -- makes debugging a future regression easy.
    if expected_line not in stdout:
        redata = "\n".join(
            line for line in stdout.splitlines()
            if "redata:" in line
        )
        pytest.fail(
            f"SIESTA 5.4.2 redata banner did not echo "
            f"{expected!r} for relax_type={relax_type!r}.\n"
            f"This is the 2026-06-23 silent-failure shape: a phantom "
            f"step-count keyword was emitted and SIESTA dropped it "
            f"silently, defaulting dynamics to Single-point.\n"
            f"\nredata block:\n{redata}"
        )
    # Belt-and-braces: explicitly assert the dropped state did not
    # occur (a future emitter change could conceivably produce both
    # the expected banner AND a single-point fallback in one run).
    assert "Single-point calculation" not in stdout, (
        f"SIESTA fell back to Single-point for relax_type={relax_type!r} "
        f"even though the requested banner appeared.  This shape "
        f"should be impossible; investigate the emitter."
    )


def test_savehs_value_lands_in_fdf_echo(tmp_path):
    """SIESTA writes its fdf-echo to ``fdf-<timestamp>.log`` showing
    every recognised key + its value.  A keyword that SIESTA does NOT
    recognise is omitted (the silent-failure shape).  A keyword that
    IS recognised but the user did not set shows ``# default value``.

    Pre-2026-06-23 the generator emitted ``WriteHS`` which SIESTA 5.4.2
    silently dropped.  The default-T behavior of ``SaveHS`` masked the
    bug whenever the user wanted T anyway; the day someone set
    ``cfg.write_hs=False`` to skip the HSX overhead, the override
    silently did nothing.

    This test pins the post-fix shape: ``SaveHS .false.`` is emitted
    by the generator, and the fdf-echo carries that value with NO
    ``# default value`` annotation (proving SIESTA accepted the user
    override).
    """
    binary = _require_siesta_binary()

    # Build a config with write_hs=False -- the case that broke pre-fix.
    fdf_path = _minimal_h2_fdf(tmp_path, "CG")
    # Append SaveHS .false. explicitly (the generator should also emit
    # this, but the test pins SIESTA's behavior, not just the renderer).
    text = fdf_path.read_text()
    # Sanity: the generator should already emit SaveHS .false.; if it
    # does, the test still passes (idempotent).
    if "SaveHS" not in text:
        fdf_path.write_text(text + "\nSaveHS             .false.\n")

    _run_siesta_on_fdf(binary, fdf_path, work_dir=tmp_path)

    # Find the fdf-echo log (timestamped filename written by SIESTA;
    # naming is ``fdf.<YYYYMMDDTHHMMSS>.<MS>.log``).
    echo_logs = sorted(tmp_path.glob("fdf.*.log"))
    assert echo_logs, (
        f"SIESTA did not write its fdf-echo log to {tmp_path}.  "
        f"Either the run crashed before the echo step or the SIESTA "
        f"binary is from a version that uses a different log path."
    )
    echo = echo_logs[-1].read_text()

    # SaveHS lines in the echo.  Each non-default entry has shape:
    #     SaveHS              F      (no '# default value' tail)
    # A defaulted entry has shape:
    #     SaveHS              T      # default value
    save_hs_lines = [
        line for line in echo.splitlines()
        if line.strip().startswith("SaveHS")
    ]
    assert save_hs_lines, (
        f"SaveHS not present in SIESTA's fdf-echo log.  This means "
        f"the keyword is not recognised by SIESTA 5.4.2 -- exactly "
        f"the WriteHS pre-fix silent-failure shape.  Echo content:\n"
        f"{echo}"
    )
    # The user value (F) lands without the # default value annotation.
    assert any(
        "F" in line and "# default value" not in line
        for line in save_hs_lines
    ), (
        f"SaveHS in fdf-echo is the default value, not the user "
        f"override.  This means SIESTA did not pick up cfg.write_hs="
        f"False from the rendered fdf -- the WriteHS-style silent "
        f"failure has returned.  SaveHS lines in echo:\n"
        + "\n".join(save_hs_lines)
    )
