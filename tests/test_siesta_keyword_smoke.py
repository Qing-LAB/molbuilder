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
authoritative one per docs/execution/job-contracts.md).
"""

from __future__ import annotations

from _deck import assert_fdf

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
                      timeout_s: float = 60.0
                      ) -> "subprocess.CompletedProcess":
    """Run SIESTA against ``fdf_path`` and return the full
    ``CompletedProcess`` so the caller can inspect ``returncode``
    along with stdout.

    Runs in ``work_dir`` so SIESTA's per-run side files (`.BASIS`,
    `.bib`, MESSAGES, …) land in the temp dir, not in cwd.  No MPI
    -- single-process; the test only needs the redata banner, not
    parallel execution.

    Audit fix 2026-06-24: previously this helper returned just
    stdout, so a mid-init SIESTA crash (return code != 0) would
    surface to the caller as a clean string with no banner.  The
    caller's "redata banner X not in stdout" assertion would then
    fire a misleading "phantom keyword regressed" message when the
    actual cause was an environment-level crash.  Returning the
    full CompletedProcess lets the caller distinguish.
    """
    return subprocess.run(
        [str(binary), str(fdf_path.name)],
        cwd=str(work_dir),
        capture_output=True, text=True, timeout=timeout_s,
    )


def _minimal_h2_fdf(tmp_path: Path, relax_type: str,
                     n_steps: int = 1,
                     write_hs: bool = True) -> Path:
    """Render a minimal H2 .fdf for the given relax type.

    Uses molbuilder's actual generator so the test exercises the
    same emission code path as production.  Returns the path to the
    rendered .fdf.

    ``write_hs`` lets the caller force the generator's
    ``SaveHS .true./.false.`` emission so the SaveHS-recognition
    test can pin that the GENERATOR emits the override -- not
    that the test happens to have appended it.  Default True
    matches SiestaConfig's default.
    """
    from molbuilder.siesta import SiestaConfig, render_fdf
    from molbuilder.structure import Structure

    # Copy the H.psml in (SIESTA reads it relative to cwd).
    shutil.copy(_H_PSML_SOURCE, tmp_path / "H.psml")

    struct = Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0],
                            [0.74, 0.0, 0.0]]), vacuum=(12.0, 12.0, 12.0))
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
        # SaveHS override controlled by caller.
        write_hs=write_hs,
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
    proc = _run_siesta_on_fdf(binary, fdf_path, work_dir=tmp_path)
    stdout = proc.stdout

    # Pre-condition: SIESTA actually ran far enough to print the
    # redata block.  A non-zero return code BEFORE the redata point
    # would falsely fail the "phantom keyword regressed" assertion
    # with a misleading message -- distinguish the crash case
    # explicitly.  Heuristic: redata always prints its first line
    # ("redata: " appears in stdout) shortly after input parse +
    # before any heavy compute, so requiring it BEFORE the dynamics
    # assertion correctly classifies env crashes vs keyword
    # regressions.
    if "redata:" not in stdout:
        pytest.fail(
            f"SIESTA did not print the redata banner -- the run "
            f"crashed before reaching the input-echo step.  "
            f"return code = {proc.returncode}.  This is an "
            f"environment-level failure, NOT the 2026-06-23 "
            f"silent-failure class.  Inspect:\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- last stdout lines ---\n"
            + "\n".join(stdout.splitlines()[-15:])
        )

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

    # Build a config with write_hs=False (the case that broke pre-fix)
    # so the GENERATOR is forced to emit ``SaveHS .false.`` -- not the
    # test.  Audit fix 2026-06-24: previously this test appended
    # ``SaveHS .false.`` itself when missing from the rendered fdf,
    # which would make the test pass for the wrong reason if the
    # generator regressed to ``WriteHS`` (the test would notice
    # SaveHS missing, append it, and SIESTA would see the appended
    # line).  Now: if the generator regresses, ``SaveHS`` is absent
    # from the rendered fdf, the fdf-echo shows it as ``# default
    # value``, and the assertion at the bottom of this test fails
    # cleanly.
    fdf_path = _minimal_h2_fdf(tmp_path, "CG", write_hs=False)
    text = fdf_path.read_text()
    # The 2026-06-23 WriteHS->SaveHS regression: the generator must emit
    # ``SaveHS .false.`` for cfg.write_hs=False.  `assert_fdf` names the
    # keyword and the value it found, so no separate message is needed.
    assert_fdf(text, "SaveHS", ".false."), (
        f"Generator must emit ``SaveHS .false.`` for cfg.write_hs="
        f"False, but the rendered fdf does not contain it.  This is "
        f"the 2026-06-23 WriteHS->SaveHS regression returning.  "
        f"Snippet of rendered fdf:\n"
        + "\n".join(
            ln for ln in text.splitlines()
            if "Save" in ln or "Write" in ln
        )
    )

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
    # Match SIESTA's fdf-echo column shape exactly: ``SaveHS  F``
    # (the substring ``F in line`` would falsely match
    # ``SaveHS .false.`` lowercase string echoes too -- pin the
    # actual SIESTA T/F single-char column with a regex).
    import re
    saved_user_value = any(
        re.match(r"^\s*SaveHS\s+F\b", line)
        and "# default value" not in line
        for line in save_hs_lines
    )
    assert saved_user_value, (
        f"SaveHS in fdf-echo is the default value, not the user "
        f"override.  This means SIESTA did not pick up cfg.write_hs="
        f"False from the rendered fdf -- the WriteHS-style silent "
        f"failure has returned.  SaveHS lines in echo:\n"
        + "\n".join(save_hs_lines)
    )


# --------------------------------------------------------------------- #
#  The k-grid displacement — SIESTA reads the block's fourth column     #
#                                                                        #
#  Added 2026-08-14 with the parameter itself.  molbuilder wrote a       #
#  hard-coded 0.0 there for the life of the project, so the classic      #
#  Monkhorst-Pack shift was unreachable                                  #
#  (docs/audit-2026-08-14-template-execution-review.md § 53, § 54).      #
#  The same silent-failure shape as the 2026-06-23 keyword incident:     #
#  a number in the deck that nothing proves the engine acts on.          #
# --------------------------------------------------------------------- #

def _h2_kgrid_fdf(tmp_path: Path, label: str, displ) -> Path:
    """A minimal H2 deck at 4x4x4 with the given displacement."""
    from molbuilder.siesta import SiestaConfig, render_fdf
    from molbuilder.structure import Structure

    shutil.copy(_H_PSML_SOURCE, tmp_path / "H.psml")
    struct = Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        vacuum=(6.0, 6.0, 6.0))
    cfg = SiestaConfig(
        relax_type="none", mesh_cutoff=50.0, max_scf_iter=3,
        dm_tolerance=1e-2, psml_lib=None, system_label=label,
        kgrid=(4, 4, 4), kgrid_displacement=displ)
    path = tmp_path / f"{label}.fdf"
    path.write_text(render_fdf(struct, cfg))
    return path


def _kgrid_echo(stdout: str):
    """SIESTA's own read-back: the three ``siesta: k-grid:`` rows and the
    irreducible k-point count it derived from them."""
    import re
    rows = re.findall(
        r"^siesta: k-grid:\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+([\d.]+)\s*$",
        stdout, re.MULTILINE)
    m = re.search(r"Number of k-points\s*=\s*(\d+)", stdout)
    return rows, (int(m.group(1)) if m else None)


def test_the_kgrid_displacement_reaches_siesta_and_changes_the_sampling(
        tmp_path):
    """Not "SIESTA echoed our number" — *SIESTA sampled differently*.

    Two decks, identical but for the fourth column.  SIESTA prints the
    supercell and displacements it will use, and then the number of
    **irreducible** k-points it derived.  Measured on SIESTA 5.4.2,
    2026-08-14:

    ======================  ========  ==================
    displacement            echo      irreducible k-pts
    ======================  ========  ==================
    ``[0, 0, 0]``           ``0.000``  44
    ``[0.5, 0.5, 0.5]``     ``0.500``  32
    ======================  ========  ==================

    The count is the evidence that matters: an echo could be a
    pass-through, but a different irreducible set means the shift entered
    the symmetry reduction.  (The *effective cutoff* is 24.000 Ang for
    both — it is a property of the supercell, not of where the mesh sits,
    which is why the count and not the cutoff is what this asserts.)

    Both runs abort at ``SCF convergence failure`` a few steps later, by
    design: ``max_scf_iter=3`` keeps them to seconds, and the k-grid is
    read and echoed long before.  The assertions never touch the exit code.
    """
    binary = _require_siesta_binary()

    counts = {}
    for label, displ in (("gamma", (0.0, 0.0, 0.0)),
                         ("shift", (0.5, 0.5, 0.5))):
        work = tmp_path / label
        work.mkdir()
        fdf = _h2_kgrid_fdf(work, label, displ)
        proc = _run_siesta_on_fdf(binary, fdf, work_dir=work, timeout_s=300.0)
        rows, n_k = _kgrid_echo(proc.stdout)
        assert len(rows) == 3, (
            f"SIESTA printed no k-grid read-back for {label}; it may have "
            f"died before reading the block.  stdout tail:\n"
            + "\n".join(proc.stdout.splitlines()[-15:]))
        want = f"{displ[0]:.3f}"
        assert [r[3] for r in rows] == [want] * 3, (
            f"{label}: SIESTA used displacements {[r[3] for r in rows]}, "
            f"not {want}.  The block's fourth column is not reaching the "
            f"engine -- the shape of the 2026-06-23 silent-keyword failure.")
        assert n_k, f"{label}: no k-point count in SIESTA's output"
        counts[label] = n_k

    assert counts["gamma"] != counts["shift"], (
        f"Both displacements gave {counts['gamma']} irreducible k-points.  "
        f"SIESTA echoed the shift but sampled the same set, so the "
        f"parameter is decorative.")
