"""Round-trip tests for the convergence_targets nested-shape (task #534
commit 3b — pre-cutover wire-format support).

Contract:

* Flat header lines (legacy single-stage runs)::

    # convergence.max_force_tol_eV_per_A: 0.023
    # convergence.scf_energy_tol:         1e-9

  parser → ``runtime_info["convergence_targets"] = {key: val, ...}``

* Nested header lines (staged runs) — the first segment is the stage's
  artifact TOKEN (`identity.stage_token`), which is DIGIT-FIRST
  (``01_coarse``, `job-contracts.md` § 6.3).  Until 2026-08-19 this file
  round-tripped an identifier-shaped ``stageN`` spelling no real deck
  writes, so the reader's letter-first key regex passed every test here
  while parsing every real staged header to an empty dict::

    # convergence.01_coarse.max_force_tol_eV_per_A: 0.103
    # convergence.01_coarse.scf_energy_tol:         1e-7
    # convergence.02_tight.max_force_tol_eV_per_A: 0.023
    # convergence.02_tight.scf_energy_tol:         1e-9

  parser → ``runtime_info["convergence_targets"] = {stage_name:
                                                    {key: val}}``

The two never mix on a single run (emitter picks one shape based
on whether its input dict has nested-dict values).
"""
from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------- #
#  Emitter — dispatch on shape                                          #
# --------------------------------------------------------------------- #


def test_emitter_writes_flat_lines_for_flat_input(tmp_path):
    """Legacy single-stage input emits bare ``# convergence.<leaf>:``
    lines without a stage prefix."""
    pytest.importorskip("pyscf")
    from molbuilder.trajectory_log.emitter import MolwatchEmitter
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    out = tmp_path / "flat.molwatch.log"
    MolwatchEmitter(
        str(out), "test", mol,
        runtime_info=None,
        convergence_targets={
            "max_force_tol_eV_per_A": 0.023,
            "scf_energy_tol":         1.0e-9,
            "max_geom_iter":          200,
        },
    )
    body = out.read_text()
    assert "# convergence.max_force_tol_eV_per_A: 0.023" in body
    assert "# convergence.scf_energy_tol: 1e-09"        in body
    assert "# convergence.max_geom_iter: 200"           in body
    # No nested-shape lines.
    assert "convergence.stage" not in body


def test_emitter_writes_nested_lines_for_nested_input(tmp_path):
    """Staged input emits ``# convergence.<stage>.<leaf>:`` lines
    per stage."""
    pytest.importorskip("pyscf")
    from molbuilder.trajectory_log.emitter import MolwatchEmitter
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    out = tmp_path / "nested.molwatch.log"
    MolwatchEmitter(
        str(out), "test", mol,
        runtime_info=None,
        convergence_targets={
            "01_coarse": {
                "max_force_tol_eV_per_A": 0.103,
                "scf_energy_tol":         1.0e-7,
                "max_geom_iter":          50,
            },
            "02_tight": {
                "max_force_tol_eV_per_A": 0.023,
                "scf_energy_tol":         1.0e-9,
                "max_geom_iter":          200,
            },
        },
    )
    body = out.read_text()
    assert "# convergence.01_coarse.max_force_tol_eV_per_A: 0.103" in body
    assert "# convergence.01_coarse.scf_energy_tol: 1e-07"         in body
    assert "# convergence.02_tight.max_force_tol_eV_per_A: 0.023" in body
    assert "# convergence.02_tight.scf_energy_tol: 1e-09"         in body
    # No flat-shape lines.
    assert "# convergence.max_force" not in body


# --------------------------------------------------------------------- #
#  Parser — round-trip both shapes                                      #
# --------------------------------------------------------------------- #


_SAMPLE_FLAT = """\
# molwatch trajectory log v1
# engine: pyscf
# convergence.max_force_tol_eV_per_A: 0.023
# convergence.scf_energy_tol: 1e-09
# convergence.max_geom_iter: 200

==== molwatch step 0 begin ====
step_index: 0
n_atoms: 1
coordinates (Ang):
   H      0.00000000      0.00000000      0.00000000
energy (eV): -1.0
max_force (eV/Ang): None
scf_history begin
scf_history end
==== molwatch step 0 end ====
"""

_SAMPLE_NESTED = """\
# molwatch trajectory log v1
# engine: pyscf
# convergence.01_coarse.max_force_tol_eV_per_A: 0.103
# convergence.01_coarse.scf_energy_tol: 1e-07
# convergence.01_coarse.max_geom_iter: 50
# convergence.02_tight.max_force_tol_eV_per_A: 0.023
# convergence.02_tight.scf_energy_tol: 1e-09
# convergence.02_tight.max_geom_iter: 200

==== molwatch step 0 begin ====
step_index: 0
n_atoms: 1
coordinates (Ang):
   H      0.00000000      0.00000000      0.00000000
energy (eV): -1.0
max_force (eV/Ang): None
scf_history begin
scf_history end
==== molwatch step 0 end ====
"""


def _parse(text: str, tmp_path):
    from molbuilder.parse.engines.molwatch import MolwatchLogParser
    p = tmp_path / "fixture.molwatch.log"
    p.write_text(text)
    return MolwatchLogParser.parse(str(p))


def test_parser_reads_flat_header_as_flat_dict(tmp_path):
    """Legacy flat header round-trips: leaf keys at top level."""
    traj = _parse(_SAMPLE_FLAT, tmp_path)
    ct = traj.runtime_info["convergence_targets"]
    assert ct["max_force_tol_eV_per_A"] == pytest.approx(0.023)
    assert ct["scf_energy_tol"]         == pytest.approx(1.0e-9)
    assert ct["max_geom_iter"]          == 200
    assert ct["source"] == "molwatch_header"


def test_parser_reads_nested_header_as_nested_dict(tmp_path):
    """Nested header round-trips: stage_name → leaf_key dict."""
    traj = _parse(_SAMPLE_NESTED, tmp_path)
    ct = traj.runtime_info["convergence_targets"]
    assert isinstance(ct["01_coarse"], dict)
    assert ct["01_coarse"]["max_force_tol_eV_per_A"] == pytest.approx(0.103)
    assert ct["01_coarse"]["scf_energy_tol"]         == pytest.approx(1.0e-7)
    assert ct["01_coarse"]["max_geom_iter"]          == 50
    assert ct["02_tight"]["max_force_tol_eV_per_A"] == pytest.approx(0.023)
    assert ct["02_tight"]["scf_energy_tol"]         == pytest.approx(1.0e-9)
    assert ct["source"] == "molwatch_header"


def test_parser_detects_nested_via_top_level_dict_values(tmp_path):
    """Detection rule (mirrored by the JS reader): any top-level
    value that's a dict tags the payload as nested.  ``source`` is
    a string, doesn't count."""
    traj = _parse(_SAMPLE_NESTED, tmp_path)
    ct = traj.runtime_info["convergence_targets"]
    nested = any(isinstance(v, dict) for k, v in ct.items()
                 if k != "source")
    assert nested is True

    traj_flat = _parse(_SAMPLE_FLAT, tmp_path)
    ct_flat = traj_flat.runtime_info["convergence_targets"]
    nested_flat = any(isinstance(v, dict) for k, v in ct_flat.items()
                      if k != "source")
    assert nested_flat is False
