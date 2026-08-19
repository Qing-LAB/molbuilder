"""A staged PySCF run's siblings are found by the REAL name grammar.

The generator stems every token-carrying sibling on ``<job>[_<token>]``
(`pyscf/input.py`; token = `identity.stage_token`, digit-first
``01_coarse``).  Three readers in ``parse/engines/pyscf.py`` each kept a
private pre-stage stem-stripper, so for every staged run the
trajectory-view path silently lost its molwatch metadata (targets +
run_state), its SCF history (it read the geomeTRIC opt log instead of
the pyscf stdout), and its Step-0 energy fallback (2026-08-19).  One
inverse (``_resolve_job_token``) now answers for all of them, and these
tests hold it to the writer's grammar -- with the digit-first token,
never the imagined ``stageN`` spelling.
"""
from __future__ import annotations

from pathlib import Path

from molbuilder.parse.engines.pyscf import (
    _parse_pyscf_xyz, _resolve_job_token, _sibling_molwatch_log)


def _staged_set(d: Path, job="w", token="01_coarse"):
    """The artifact set one staged rung writes, minimal but real-named."""
    stem = f"{job}_{token}"
    (d / f"{job}_geom_{token}_optim.xyz").write_text(
        "3\nIteration 0 Energy -76.10000000\n"
        "O 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"
        "3\nIteration 1 Energy -76.20000000\n"
        "O 0.0 0.0 0.0\nH 0.95 0.0 0.0\nH -0.24 0.92 0.0\n")
    (d / f"{stem}.molwatch.log").write_text(
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        f"# convergence.{token}.max_force_tol_eV_per_A: 0.0231\n"
        f"# convergence.{token}.max_geom_iter: 200\n"
        "\n# concluded: 2026-08-19T12:00:00\n")
    (d / f"{stem}.log").write_text(
        "cycle= 1 E= -76.0  delta_E= 1.0  |g|= 0.5  |ddm|= 0.1\n"
        "cycle= 2 E= -76.1  delta_E= -0.1  |g|= 0.05  |ddm|= 0.01\n"
        "converged SCF energy = -76.1\n")
    # The DECOY the old strippers used to read: geomeTRIC's own opt log,
    # which holds no SCF cycles.
    (d / f"{job}_geom_{token}.log").write_text("Step    0 : Energy = -76.1\n")
    return d / f"{job}_geom_{token}_optim.xyz"


def test_the_inverse_reads_the_writers_grammar(tmp_path):
    """(job, token) from each artifact spelling the generator emits."""
    assert _resolve_job_token(str(tmp_path), "w_geom_01_coarse_optim.xyz") \
        == ("w", "01_coarse")
    assert _resolve_job_token(str(tmp_path), "w_geom_optim.xyz") == ("w", None)
    # A job name may legally CONTAIN ``_geom``: rightmost split wins.
    assert _resolve_job_token(str(tmp_path), "x_geom_y_geom_02_tight_optim.xyz") \
        == ("x_geom_y", "02_tight")


def test_a_tokenless_artifact_resolves_its_token_from_the_molwatch_beside_it(tmp_path):
    traj = _staged_set(tmp_path)
    assert _resolve_job_token(str(tmp_path), "w_optimized.xyz") == ("w", "01_coarse")
    assert _sibling_molwatch_log(str(traj)) == str(tmp_path / "w_01_coarse.molwatch.log")


def test_a_staged_trajectory_carries_its_run_metadata(tmp_path):
    """The whole enrichment chain on real staged names: nested
    digit-first targets, the conclusion, and SCF cycles from the pyscf
    stdout -- not the geomeTRIC decoy."""
    traj = _staged_set(tmp_path)
    out = _parse_pyscf_xyz(str(traj))
    assert out.run_state == "finished"
    ct = out.runtime_info["convergence_targets"]
    assert ct["01_coarse"]["max_force_tol_eV_per_A"] == 0.0231
    assert ct["01_coarse"]["max_geom_iter"] == 200
    assert out.frames[0].scf_history, (
        "SCF cycles must come from <job>_<token>.log, which exists")
