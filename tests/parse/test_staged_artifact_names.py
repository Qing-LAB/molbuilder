"""A staged PySCF run's trajectory finds its run's other files by its own
name (`docs/model/parse.md` § 5.3).

geomeTRIC's trajectory is named on its run's stem --
``<label>_<token>_geom_optim.xyz``, the token right after the label
(`job-contracts.md` § 2.2a) -- and every other file of the run is named on
the same stem: its progress log, PySCF's own log, the wrapper's output.  So
the reader takes the stem off the trajectory's name and asks for each by it.
Three readers kept private stem-strippers until 2026-08-19 and lost a staged
run's metadata; one inverse that split the name into a label and a stage
with a stage pattern of its own, and took a stageless file's stage from the
newest progress log in the folder, stood until 2026-10-04 (plan W56 4d).

The run's files are named here by the writer's own composer
(`runfiles.compose`), never spelled by hand.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse.engines.pyscf import PySCFParser, _parse_pyscf_xyz
from molbuilder.runfiles import compose


def _staged_set(d: Path, job="w", token="01_coarse"):
    """The files one staged rung writes, minimal, named as the writer names
    them."""
    traj = d / compose(job, "_geom_optim.xyz", token)
    traj.write_text(
        "3\nIteration 0 Energy -76.10000000\n"
        "O 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"
        "3\nIteration 1 Energy -76.20000000\n"
        "O 0.0 0.0 0.0\nH 0.95 0.0 0.0\nH -0.24 0.92 0.0\n")
    (d / compose(job, ".molwatch.log", token)).write_text(
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        f"# convergence.{token}.max_force_tol_eV_per_A: 0.0231\n"
        f"# convergence.{token}.max_geom_iter: 200\n"
        "\n# concluded: 2026-08-19T12:00:00\n")
    (d / compose(job, ".log", token)).write_text(
        "cycle= 1 E= -76.0  delta_E= 1.0  |g|= 0.5  |ddm|= 0.1\n"
        "cycle= 2 E= -76.1  delta_E= -0.1  |g|= 0.05  |ddm|= 0.01\n"
        "converged SCF energy = -76.1\n")
    # The DECOY the old strippers used to read: geomeTRIC's own opt log,
    # which holds no SCF cycles.
    (d / compose(job, "_geom.log", token)).write_text(
        "Step    0 : Energy = -76.1\n")
    return traj


# `test_the_inverse_reads_the_writers_grammar` and
# `test_a_tokenless_artifact_resolves_its_token_from_the_molwatch_beside_it`
# retired 2026-10-04 (W56 4d): their subject, `_resolve_job_token`, is gone --
# and the names they pinned put the token inside the role
# (`w_geom_01_coarse_optim.xyz`), a spelling no writer of ours has produced
# since 2026-09-07.


def test_a_staged_trajectory_carries_its_run_metadata(tmp_path):
    """The whole enrichment chain on a staged run's names: nested
    digit-first targets, the conclusion, and SCF cycles from the pyscf
    stdout -- not the geomeTRIC decoy."""
    traj = _staged_set(tmp_path)
    out = _parse_pyscf_xyz(str(traj))
    assert out.run_state == "ended"
    ct = out.runtime_info["convergence_targets"]
    assert ct["01_coarse"]["max_force_tol_eV_per_A"] == 0.0231
    assert ct["01_coarse"]["max_geom_iter"] == 200
    assert out.frames[0].scf_history, (
        "SCF cycles must come from <stem>.log, which exists")


# --- An unstaged run's progress log, and what it says --------------------- #
# (from `tests/test_pyscf_initial_energy_fallback.py`, deleted 2026-10-04 with
# the fallback it was named for: a lone ``<JOB>_initial.xyz`` names no run, so
# it reads only itself -- plan W56 4d.  Its seven fallback tests went with it.)

_H2_XYZ = """2
fresh run -- initial geometry
H   0.000   0.000   0.000
H   0.740   0.000   0.000
"""


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
        xyz.write_text(_H2_XYZ)
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
        xyz.write_text(_H2_XYZ)
        (tmp_path / "h2.molwatch.log").write_text(
            self._mw_log(concluded=True))
        traj = PySCFParser.parse(str(xyz))
        assert traj.run_state == "ended"
        assert traj.error_message is None

    def test_sibling_molwatch_error_sets_error_run_state(self, tmp_path):
        xyz = tmp_path / "h2_geom_optim.xyz"
        xyz.write_text(_H2_XYZ)
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
        xyz.write_text(_H2_XYZ)
        # No sibling .molwatch.log written.
        traj = PySCFParser.parse(str(xyz))
        assert traj.run_state == "unknown"
        assert traj.error_message is None
        assert "convergence_targets" not in traj.runtime_info
