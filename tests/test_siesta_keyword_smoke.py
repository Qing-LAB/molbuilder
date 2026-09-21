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

# THE PSEUDOPOTENTIAL COMES FROM THE SUITE'S OWN HELPER, not from a real
# project (2026-09-06).  This borrowed
# `projects/BDT/optimization/TJ-BDT-Au111/H.psml` -- a specific real
# calculation's file, chosen because it was "the smallest available" -- which
# is precisely what `test_no_tests_read_the_projects_tree.py` forbids: the
# file's relevance was never confirmed, it changes meaning with no diff, and
# on a machine without that project the test would skip while reading green.
# It survived because that guard scanned one LINE at a time and this
# expression was split across two.
#
# `conftest.write_pseudos` is NOT the answer here, and that was measured
# rather than assumed: its PSML is real enough for prep's screening but
# SIESTA does not start on it -- the swap failed five of this file's tests
# with "SIESTA printed no k-grid read-back; it may have died".  So the file
# is checked in beside the tests, which is what
# `tests/watch/fixtures/siesta_frozen/` already does for what cannot be
# honestly constructed.  See tests/fixtures/psml/README.md.
_H_PSML_SOURCE = Path(__file__).resolve().parent / "fixtures" / "psml" / "H.psml"


def _siesta_binary():
    """Return the path to the molbuilder-siesta env's siesta binary,
    or None when the env is not present on this machine.

    **Asks the product where the env IS** rather than guessing its layout
    (2026-08-25).  This computed ``Path(CONDA_PREFIX).parent / "envs" /
    <name>``, which is right only when ``CONDA_PREFIX`` is the BASE
    install.  Run from inside an env -- which is how this suite runs --
    ``.parent`` is ALREADY ``envs/``, so it built ``envs/envs/<name>/...``
    and never matched.  It then fell back to ``<mgr> run -n <env> which
    <tool>``, and ``which`` searches the INHERITED PATH: on a machine with
    a system install that returns the HOST's binary.  So this file's own
    rule -- *"no host PATH siesta is permitted; the env's binary is the
    one"* -- was broken by its own fallback, silently, for as long as it
    has existed; on the dev workstation it was measuring a root-owned 2023 build.
    ``_env_prefix`` is the product's four-tier resolver and cannot return
    something outside the env.
    """
    from molbuilder.diagnostics import get_capabilities
    from molbuilder.envs.install import _env_prefix
    caps = get_capabilities()
    if not caps.env_available("molbuilder-siesta"):
        return None
    prefix = _env_prefix("molbuilder-siesta", caps.conda_binary)
    if not prefix:
        return None
    candidate = Path(prefix) / "bin" / "siesta"
    return candidate if candidate.exists() else None


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
# the binary in /home/u/.claude/jobs/074e4f77/tmp/siesta-kw-audit/.
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
#  (docs/archive/2026-08-14-template-execution-review.md § 53, § 54).      #
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


# ===================================================================== #
#  SIESTA'S OWN UNIT RULES, asked of the binary                         #
# ===================================================================== #
#
# `parse/fdf.py` has to decide what a keyword MEANS when the deck states
# no unit.  Those decisions were invented once -- a bare energy "is Ry",
# coordinates "are Ang" -- and both were wrong, the second badly enough
# to refuse a correct junction.  An invented rule is a rule nothing can
# check, so these ask the engine instead: whatever SIESTA does IS the
# requirement, and if a future SIESTA changes it these go red.


def _unit_probe_deck(tmp_path, *, extra_lines: str = "",
                     coord_format: str = None) -> Path:
    """A two-atom H deck, minimal enough that SIESTA reaches the reader."""
    shutil.copy(_H_PSML_SOURCE, tmp_path / "H.psml")
    fmt = f"AtomicCoordinatesFormat {coord_format}\n" if coord_format else ""
    path = tmp_path / "probe.fdf"
    path.write_text(
        "SystemLabel probe\nNumberOfAtoms 2\nNumberOfSpecies 1\n"
        "%block ChemicalSpeciesLabel\n 1 1 H\n%endblock ChemicalSpeciesLabel\n"
        + fmt + extra_lines +
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        " 0.0 0.0 0.0 1\n 0.0 0.0 1.4 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n")
    return path


@pytest.mark.parametrize("keyword,bare,with_unit", [
    ("MeshCutoff",           "MeshCutoff 250",          "MeshCutoff 250 Ry"),
    ("PAO.EnergyShift",      "PAO.EnergyShift 0.01",    "PAO.EnergyShift 0.01 Ry"),
    ("ElectronicTemperature", "ElectronicTemperature 300",
     "ElectronicTemperature 300 K"),
    ("LatticeConstant",      "LatticeConstant 10.0",    "LatticeConstant 10.0 Ang"),
])
def test_siesta_REFUSES_a_physical_value_with_no_unit(keyword, bare,
                                                      with_unit, tmp_path):
    """THE REQUIREMENT, from the engine: there is no default unit.

    `parse/fdf.py` therefore passes no `default=` for any of these, and
    a bare value is left unanswered rather than read as a guess.  A deck
    carrying one is a deck SIESTA would not have run.
    """
    binary = _require_siesta_binary()
    bad_dir = tmp_path / "bad"; bad_dir.mkdir(parents=True)
    out = _run_siesta_on_fdf(
        binary, _unit_probe_deck(bad_dir, extra_lines=bare + "\n"),
        work_dir=bad_dir)
    assert "no unit specified" in (out.stdout + out.stderr), (
        f"SIESTA accepted a bare {keyword}; if it has gained a default "
        f"unit, parse/fdf.py may adopt it -- but read it off THIS output, "
        f"not off a manual:\n{out.stdout[-800:]}")

    good_dir = tmp_path / "good"; good_dir.mkdir(parents=True)
    ok = _run_siesta_on_fdf(
        binary, _unit_probe_deck(good_dir, extra_lines=with_unit + "\n"),
        work_dir=good_dir)
    assert "no unit specified" not in (ok.stdout + ok.stderr), (
        f"the control failed: {keyword} WITH a unit was also refused")

    # AND OUR READER FOLLOWS IT.  Establishing the engine's rule is only
    # half a gate; this is the half that fails when we drift from it.
    from molbuilder.parse.fdf import parse_fdf_params
    from molbuilder.units import UnknownUnit
    probe = f"{bare}\n"
    if keyword == "LatticeConstant":
        probe += ("%block LatticeVectors\n 1 0 0\n 0 1 0\n 0 0 1\n"
                  "%endblock LatticeVectors\n")
    try:
        got = parse_fdf_params(probe, source="probe.fdf")
    except UnknownUnit:
        return                      # refused, which is the engine's answer
    field = {"MeshCutoff": "mesh_cutoff_ry",
             "PAO.EnergyShift": "energy_shift_ry",
             "ElectronicTemperature": "electronic_temperature_k",
             "LatticeConstant": "cell_ang"}[keyword]
    assert getattr(got, field) is None, (
        f"SIESTA refuses a bare {keyword}, so parse/fdf.py must not "
        f"answer one -- it returned {getattr(got, field)!r}")


def test_siesta_defaults_omitted_coordinates_to_BOHR(tmp_path):
    """The one keyword here that DOES have a default, and it is not Ang.

    `AtomicCoordinatesFormat` is read with `fdf_string(key, default)`,
    so omitting it is legal and means something.  `parse/fdf.py` read it
    as Ang, which is 1.89x out -- and `coords_ang` is the frozen gate's
    baseline, so a correct junction cited as a foreign deck was refused
    for atoms that had not moved.
    """
    binary = _require_siesta_binary()
    (tmp_path / "d").mkdir(parents=True, exist_ok=True)
    deck = _unit_probe_deck(tmp_path / "d")
    out = _run_siesta_on_fdf(binary, deck, work_dir=deck.parent)
    text = out.stdout + out.stderr
    assert "Bohr" in text and "coor:" in text, (
        f"could not read the coordinate-format banner:\n{text[-800:]}")
    assert "Angstrom" not in text.split("coor:")[1][:200], (
        "SIESTA now defaults omitted coordinates to Angstrom; "
        "parse/fdf.py's default must follow THIS, not a manual")

    # AND OUR READER FOLLOWS IT: 1.4 with no keyword is 1.4 BOHR.
    from molbuilder.constants import BOHR_ANGSTROM
    from molbuilder.parse.fdf import parse_fdf_params
    got = parse_fdf_params(deck.read_text(), source="probe.fdf")
    assert got.coords_ang is not None
    assert got.coords_ang[1][2] == pytest.approx(1.4 * BOHR_ANGSTROM), (
        f"SIESTA read this deck in Bohr; parse/fdf.py read "
        f"{got.coords_ang[1][2]} A, which is Angstrom")
