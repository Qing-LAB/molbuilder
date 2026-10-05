"""Round-trip tests for the convergence_targets nested-shape (task #534
commit 3b — pre-cutover wire-format support).

Contract:

* Flat header lines (legacy single-stage runs)::

    # convergence.max_force_tol_eV_per_A: 0.023
    # convergence.energy_step_tol_eV:         1e-9

  parser → ``runtime_info["convergence_targets"] = {key: val, ...}``

* Nested header lines (staged runs) — the first segment is the stage's
  artifact TOKEN (`identity.stage_token`), which is DIGIT-FIRST
  (``01_coarse``, `job-contracts.md` § 6.3).  Until 2026-08-19 this file
  round-tripped an identifier-shaped ``stageN`` spelling no real deck
  writes, so the reader's letter-first key regex passed every test here
  while parsing every real staged header to an empty dict::

    # convergence.01_coarse.max_force_tol_eV_per_A: 0.103
    # convergence.01_coarse.energy_step_tol_eV:         1e-7
    # convergence.02_tight.max_force_tol_eV_per_A: 0.023
    # convergence.02_tight.energy_step_tol_eV:         1e-9

  parser → ``runtime_info["convergence_targets"] = {stage_name:
                                                    {key: val}}``

The two never mix on a single run (emitter picks one shape based
on whether its input dict has nested-dict values).
"""
from __future__ import annotations


import pytest


# --------------------------------------------------------------------- #
#  Emitter — dispatch on shape                                          #
# --------------------------------------------------------------------- #


def test_emitter_writes_flat_lines_for_flat_input(tmp_path):
    """Legacy single-stage input emits bare ``# convergence.<leaf>:``
    lines without a stage prefix."""
    # `pyscf`, not `pyscf.gto`, let a HALF-INSTALLED pyscf through:
    # `pyscf-properties` ships pyscf/prop and pyscf/pbc with no top-level
    # module, so `import pyscf` succeeds on a namespace package and this
    # guard did not guard.  Found 2026-09-12 on a host env carrying that
    # package.  Ask for the submodule the test actually imports.
    pytest.importorskip("pyscf.gto")
    from molbuilder.trajectory_log.emitter import MolwatchEmitter
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    out = tmp_path / "flat.molwatch.log"
    MolwatchEmitter(
        str(out), "test", mol,
        runtime_info=None,
        convergence_targets={
            "max_force_tol_eV_per_A": 0.023,
            "energy_step_tol_eV":         1.0e-9,
            "max_geom_iter":          200,
        },
    )
    body = out.read_text()
    assert "# convergence.max_force_tol_eV_per_A: 0.023" in body
    assert "# convergence.energy_step_tol_eV: 1e-09"        in body
    assert "# convergence.max_geom_iter: 200"           in body
    # No nested-shape lines.
    assert "convergence.stage" not in body


def test_emitter_writes_nested_lines_for_nested_input(tmp_path):
    """Staged input emits ``# convergence.<stage>.<leaf>:`` lines
    per stage."""
    # `pyscf`, not `pyscf.gto`, let a HALF-INSTALLED pyscf through:
    # `pyscf-properties` ships pyscf/prop and pyscf/pbc with no top-level
    # module, so `import pyscf` succeeds on a namespace package and this
    # guard did not guard.  Found 2026-09-12 on a host env carrying that
    # package.  Ask for the submodule the test actually imports.
    pytest.importorskip("pyscf.gto")
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
                "energy_step_tol_eV":         1.0e-7,
                "max_geom_iter":          50,
            },
            "02_tight": {
                "max_force_tol_eV_per_A": 0.023,
                "energy_step_tol_eV":         1.0e-9,
                "max_geom_iter":          200,
            },
        },
    )
    body = out.read_text()
    assert "# convergence.01_coarse.max_force_tol_eV_per_A: 0.103" in body
    assert "# convergence.01_coarse.energy_step_tol_eV: 1e-07"         in body
    assert "# convergence.02_tight.max_force_tol_eV_per_A: 0.023" in body
    assert "# convergence.02_tight.energy_step_tol_eV: 1e-09"         in body
    # No flat-shape lines.
    assert "# convergence.max_force" not in body


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 3 tests here parsed a progress log typed by hand; the two
# above write theirs with our own writer (`process/testing.md` § 6).
