"""The Convergence targets card's words (`web/trajectory.md` § 3): a target
reads as a person reads it.  What a log's header lines say is
`tests/data/prep_protocol.toml`'s (`progress_log_targets`).
"""
from __future__ import annotations


import pytest


@pytest.mark.parametrize("key, value, says", [
    ("energy_step_tol_eV", 2.7211386245987997e-05,
     ["energy step", "2.72e-05 eV"]),
    ("max_force_tol_eV_per_A", 0.05, ["max |F|", "0.0500 eV/Å"]),
    ("max_scf_iter", 1000, ["max SCF iterations", "1000"]),
    ("max_geom_iter", 2345, ["geom steps cap", "2345"]),
], ids=["a tolerance", "a force", "a round count", "a long count"])
def test_a_target_reads_as_a_person_reads_it(key, value, says):
    """`keys.targets_said`, the Convergence targets card: a tolerance converted
    to eV reads to three figures, a count whole.  The silent failure: *energy
    step 2.7211386245987997e-05 eV*, its binary tail on the page, and
    *max SCF iterations 1e+03*, a count read as a rounded float."""
    from molbuilder.trajectory_log import keys
    assert key in keys.LEAF_KEYS
    assert keys.targets_said({key: value}) == [says]
