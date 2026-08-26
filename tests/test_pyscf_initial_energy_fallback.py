"""Regression pin: PySCF early-window energy fallback.

2026-06-14 carryover from the SIESTA preamble-Etot fallback.  When
the user loads ``<JOB>_initial.xyz`` on a fresh PySCF run, the
parsed frame has no energy because the comment line is a plain
XYZ description.  But the sibling ``<JOB>-run<N>.pyscf.log``
(written by the geomeTRIC optimizer) may already carry the first
Step-0 energy.  Surfacing it lets the Results-tab energy plot
render one data point during the initialization window of a
fresh run.

These tests pin:
  1. The parser scans sibling logs and returns the Step-0 energy
     when present.
  2. Multi-run layouts pick the LATEST -run<N>.pyscf.log.
  3. ANSI color escapes geomeTRIC emits don't break the regex.
  4. Multi-frame trajectories (_geom_optim.xyz) are unaffected --
     fallback only triggers on 1-frame + no-energy.
  5. Missing log file = clean no-op (energy stays None).
"""
from __future__ import annotations

from textwrap import dedent

import pytest

from molbuilder.parse.engines.pyscf import PySCFParser


# Minimal _initial.xyz fixture: 2 atoms, no energy in comment.
_H2_INITIAL_XYZ = """2
fresh run -- initial geometry
H   0.000   0.000   0.000
H   0.740   0.000   0.000
"""


# Minimal .pyscf.log carrying a geomeTRIC ``Step 0 ... Energy = X``.
_PYSCF_LOG_STEP0 = dedent("""\
    > Job type: Energy minimization
    Step    0 : Gradient = 5.500e-02/1.076e-01 (rms/max) Energy = -1.234567890
    Step    1 : Gradient = 4.200e-02/8.500e-02 (rms/max) Energy = -1.345678901
""")


# Same content with ANSI color escapes (geomeTRIC injects them).
_PYSCF_LOG_STEP0_ANSI = (
    "Step    0 : Gradient = \x1b[0m5.500e-02\x1b[0m/\x1b[0m"
    "1.076e-01\x1b[0m (rms/max) Energy = -1.234567890\n"
    "Step    1 : Gradient = \x1b[0m4.200e-02\x1b[0m/\x1b[0m"
    "8.500e-02\x1b[0m (rms/max) Energy = -1.345678901\n"
)


_HARTREE_TO_EV = 27.211386245988


class TestInitialEnergyFallback:

    def test_initial_xyz_alone_yields_none_energy(self, tmp_path):
        """Baseline: without a sibling log, the fallback doesn't
        fire -- energy stays None.  Same trajectory shape pre-
        and post-fix when no log is present."""
        xyz = tmp_path / "h2_initial.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        traj = PySCFParser.parse(str(xyz))
        assert len(traj.frames) == 1
        assert traj.frames[0].energy is None

    def test_initial_xyz_with_log_surfaces_step0_energy(
            self, tmp_path):
        """The fallback path: sibling ``<stem>.pyscf.log`` exists
        with a Step 0 entry.  frame[0].energy must be the Step-0
        value converted Hartree -> eV."""
        xyz = tmp_path / "h2_initial.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        # Bare ``.pyscf.log`` (no run-index) -- covers the
        # single-run case.
        log = tmp_path / "h2.pyscf.log"
        log.write_text(_PYSCF_LOG_STEP0)
        traj = PySCFParser.parse(str(xyz))
        assert len(traj.frames) == 1
        expected_ev = -1.234567890 * _HARTREE_TO_EV
        assert traj.frames[0].energy == pytest.approx(expected_ev), (
            f"expected Step-0 energy {expected_ev:.4f} eV; got "
            f"{traj.frames[0].energy!r}"
        )

    def test_run_index_log_higher_wins(self, tmp_path):
        """When multiple ``-run<N>.pyscf.log`` files exist (the
        molbuilder wrapper produces a new one per re-run), the
        HIGHEST-N file wins -- it's the live one."""
        xyz = tmp_path / "h2_initial.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        # Older run -- different energy.
        (tmp_path / "h2-run0.pyscf.log").write_text(
            "Step    0 : Gradient = ... Energy = -0.999000\n"
        )
        # Newer run -- the live one.
        (tmp_path / "h2-run3.pyscf.log").write_text(
            "Step    0 : Gradient = ... Energy = -1.234567\n"
        )
        traj = PySCFParser.parse(str(xyz))
        expected_ev = -1.234567 * _HARTREE_TO_EV
        assert traj.frames[0].energy == pytest.approx(expected_ev), (
            "highest -run<N>.pyscf.log must win"
        )

    def test_ansi_color_escapes_stripped_before_match(self, tmp_path):
        """geomeTRIC's per-step output is ANSI-colored on TTY
        runs.  The fallback regex must NOT depend on the escapes
        being absent; we strip them at parse time."""
        xyz = tmp_path / "h2_initial.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        log = tmp_path / "h2.pyscf.log"
        log.write_text(_PYSCF_LOG_STEP0_ANSI)
        traj = PySCFParser.parse(str(xyz))
        expected_ev = -1.234567890 * _HARTREE_TO_EV
        assert traj.frames[0].energy == pytest.approx(expected_ev)

    def test_multiframe_trajectory_not_affected(self, tmp_path):
        """``_geom_optim.xyz`` is the multi-frame geomeTRIC
        trajectory.  When it has >=2 frames, the fallback MUST
        NOT trigger -- per-frame energies come from the comment
        lines, not from the sibling log."""
        xyz = tmp_path / "h2_geom_optim.xyz"
        xyz.write_text(dedent("""\
            2
            Iteration 0 Energy -1.500000
            H   0.000   0.000   0.000
            H   0.740   0.000   0.000
            2
            Iteration 1 Energy -1.600000
            H   0.000   0.000   0.000
            H   0.730   0.000   0.000
        """))
        # Plant a log that would surface a DIFFERENT energy if the
        # fallback misfired.
        (tmp_path / "h2.pyscf.log").write_text(
            "Step    0 : ... Energy = -9.99999\n"
        )
        traj = PySCFParser.parse(str(xyz))
        assert len(traj.frames) == 2
        # Each frame keeps its own comment-derived energy.
        assert traj.frames[0].energy == pytest.approx(
            -1.500000 * _HARTREE_TO_EV
        )
        assert traj.frames[1].energy == pytest.approx(
            -1.600000 * _HARTREE_TO_EV
        )

    def test_single_frame_with_energy_not_affected(self, tmp_path):
        """``_optimized.xyz`` carries the final-step coords with a
        comment line that includes the final energy.  When the
        comment matches the geomeTRIC pattern, the parser uses
        THAT value -- not the fallback."""
        xyz = tmp_path / "h2_optimized.xyz"
        xyz.write_text(dedent("""\
            2
            Iteration 14 Energy -1.999000
            H   0.000   0.000   0.000
            H   0.740   0.000   0.000
        """))
        # Plant a log with a different value to verify it's IGNORED.
        (tmp_path / "h2.pyscf.log").write_text(
            "Step    0 : ... Energy = -1.234567\n"
        )
        traj = PySCFParser.parse(str(xyz))
        assert len(traj.frames) == 1
        assert traj.frames[0].energy == pytest.approx(
            -1.999000 * _HARTREE_TO_EV
        ), (
            "single-frame trajectory with a comment-derived energy "
            "must NOT trigger the fallback"
        )

    def test_log_without_step_zero_returns_none(self, tmp_path):
        """A .log that has no Step 0 line (e.g., a corrupt early-
        write) leaves frame.energy as None.  Defensive: don't
        invent data when there isn't any."""
        xyz = tmp_path / "h2_initial.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        (tmp_path / "h2.pyscf.log").write_text(
            "Job type: Energy minimization\n"
            "Step    7 : ... Energy = -1.234567\n"   # no Step 0
        )
        traj = PySCFParser.parse(str(xyz))
        assert traj.frames[0].energy is None


# --- Sibling .molwatch.log header/footer enrichment ------------------- #


class TestMolwatchSiblingEnrichment:
    """Regression for the 2026-06-20 PDT incident: when the user opens
    the geomeTRIC ``_geom_optim.xyz`` trajectory directly, the PySCF
    parser previously returned a Trajectory with run_state="unknown"
    and no convergence_targets — the Results-tab badge showed
    "Ongoing" + the convergence-targets banner said "not found in
    source" even though the sibling ``.molwatch.log`` carried both.

    These tests pin:
      1. When sibling .molwatch.log is present, convergence_targets
         lift onto runtime_info verbatim (incl. the source stamp).
      2. ``# concluded: <iso>`` footer → run_state = "finished".
      3. ``# error: <msg>`` footer → run_state = "error" +
         error_message populated; error wins over concluded.
      4. No sibling log = clean no-op (run_state="unknown", no
         convergence_targets key).
    """

    @staticmethod
    def _mw_log(*, concluded: bool = False, error: str = None) -> str:
        body = (
            "# molwatch trajectory log v1\n"
            "# generator: molbuilder/pyscf_input\n"
            "# engine: pyscf\n"
            "# created: 2026-06-20T13:22:56\n"
            "# convergence.max_force_tol_eV_per_A: 0.023139\n"
            "# convergence.scf_energy_tol: 1e-09\n"
            "# convergence.max_scf_iter: 100\n"
            "# convergence.max_geom_iter: 200\n"
            "\n"
            "==== molwatch step 0 begin ====\n"
            "step_index: 0\n"
            "n_atoms: 2\n"
            "coordinates (Ang):\n"
            "   H  0.0  0.0  0.0\n"
            "energy (eV): -0.5\n"
            "forces (eV/Ang):\n"
            "max_force (eV/Ang): 0.0\n"
            "scf_history begin\n"
            "scf_history end\n"
            "==== molwatch step 0 end ====\n"
        )
        if error is not None:
            body += f"\n# error: {error}\n"
        if concluded:
            body += "\n# concluded: 2026-06-20T13:23:42\n"
        return body

    def test_sibling_molwatch_log_surfaces_convergence_targets(self, tmp_path):
        xyz = tmp_path / "h2_geom_optim.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        (tmp_path / "h2.molwatch.log").write_text(
            self._mw_log(concluded=True))
        traj = PySCFParser.parse(str(xyz))
        ct = traj.runtime_info.get("convergence_targets")
        assert ct is not None, (
            "PySCF parser must lift convergence targets from sibling "
            ".molwatch.log (2026-06-20 PDT incident)"
        )
        assert ct["source"] == "molwatch_header"
        assert ct["max_force_tol_eV_per_A"] == pytest.approx(0.023139)
        assert ct["scf_energy_tol"] == pytest.approx(1e-09)
        assert ct["max_scf_iter"] == 100
        assert ct["max_geom_iter"] == 200

    def test_sibling_molwatch_concluded_sets_finished_run_state(
            self, tmp_path):
        xyz = tmp_path / "h2_geom_optim.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        (tmp_path / "h2.molwatch.log").write_text(
            self._mw_log(concluded=True))
        traj = PySCFParser.parse(str(xyz))
        assert traj.run_state == "ended"
        assert traj.error_message is None

    def test_sibling_molwatch_error_sets_error_run_state(self, tmp_path):
        xyz = tmp_path / "h2_geom_optim.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        (tmp_path / "h2.molwatch.log").write_text(
            self._mw_log(error="diverged at SCF cycle 47", concluded=True))
        traj = PySCFParser.parse(str(xyz))
        # Error has priority over concluded (per molwatch parser
        # contract; the excepthook fires before atexit so both lines
        # appear and error wins).
        assert traj.run_state == "stopped"
        assert traj.error_message == "diverged at SCF cycle 47"

    def test_no_sibling_molwatch_log_is_clean_noop(self, tmp_path):
        xyz = tmp_path / "h2_geom_optim.xyz"
        xyz.write_text(_H2_INITIAL_XYZ)
        # No sibling .molwatch.log written.
        traj = PySCFParser.parse(str(xyz))
        assert traj.run_state == "unknown"
        assert traj.error_message is None
        assert "convergence_targets" not in traj.runtime_info
