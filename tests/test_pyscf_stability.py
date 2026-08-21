"""The open-shell stability check: before the geometry work, and acting.

An open-shell SCF can settle on a broken-symmetry SADDLE point -- the
energy stops moving, ``mf.converged`` is True, and the wavefunction is
not the variational minimum.  Until 2026-08-13 the emitted script called
``mf.stability()`` AFTER the entire optimization and only printed what it
found, so a run that landed on a saddle at the first geometry optimized
every later step and computed its frequencies on the wrong electronic
state -- then said so at the end.  The emitter's own comment named the
remedy and declined to apply it.

These assert on the EMITTED TEXT (cheap, always run).  The end-to-end
behaviour was verified by running a generated O2-triplet script under
molbuilder-pySCF:

    initial SCF: -147.63411571 Hartree (converged=True)
    round 1: instability repaired, -> -147.63562789 (dE=-1.512e-03)
    round 2: no further improvement (dE=+5.3e-11); treating as stable
    stability: CHECKED, reached a stable solution after 1 restart(s)
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import render_script
from molbuilder.structure import Structure

_BASE = dict(basis="sto-3g", optimize=False,
             write_molwatch_log=False, write_trajectory=False,
             save_optimized_xyz=False, save_initial_xyz=False,
             chkfile=False, dispersion=None)


def _o2():
    return Structure(elements=["O", "O"],
                     positions=np.array([[0., 0., 0.], [0., 0., 1.21]]))


def _script(method="UHF", spin=2, **kw):
    cfg = PySCFConfig(job_name="j", method=method, spin=spin,
                      **{**_BASE, **kw})
    return render_script(_o2(), cfg)


def test_stability_runs_before_the_geometry_optimization():
    """THE fix.  Optimizing on a saddle produces a wrong geometry and
    wrong frequencies; finding out afterwards helps nobody."""
    t = _script(optimize=True)
    assert "_internal = mf.stability()[0]" in t
    assert t.index("_internal = mf.stability()[0]") < t.index("optimize("), (
        "the stability check must precede the optimizer")


def test_an_instability_is_repaired_not_merely_reported():
    """It rebuilds the density matrix from the suggested orbitals and
    re-converges -- the step the old comment described and skipped."""
    t = _script()
    assert "mf.make_rdm1(_internal, mf.mo_occ)" in t
    assert "mf.kernel(mf.make_rdm1(" in t


def test_the_energy_is_the_criterion_not_the_orbitals():
    """Comparing orbital COEFFICIENTS looks obvious and is wrong: a
    degenerate shell (O2's pi pair) is rotated freely within its
    degenerate space, so stability() returns numerically different
    orbitals for an identical state, for ever.  Measured: rounds 2 and 3
    on O2 gave dE = +3.6e-10 and +1.6e-09 -- no improvement -- yet a
    coefficient test called all three unstable and ended in a false
    warning."""
    t = _script()
    assert "_MB_STABILITY_ETOL" in t
    assert "if e < _e_prev - _MB_STABILITY_ETOL:" in t


def test_persistent_instability_warns_and_continues():
    """User decision 2026-08-13: a hint does not get to end the run."""
    t = _script()
    assert "_MB_STABILITY_MAX = 3" in t
    assert "still internally unstable after" in t
    assert "advice, not a veto" in t
    assert "raise" not in t.split("stability")[1][:2000]


@pytest.mark.parametrize("needle", [
    "stability: NOT CHECKED",
    "stable on the first SCF",
    "reached a stable solution after",
    "WARNING -- still internally unstable",
])
def test_the_log_distinguishes_every_outcome(needle):
    """'Checked and stable' must never read the same as 'never checked'
    -- the same rule the bench readback applies to hardware."""
    assert needle in _script()


def test_a_method_that_cannot_be_checked_says_so():
    """Law A: a check that could not run reports it.  Silence would
    read as a clean bill of health."""
    t = _script()
    assert "except (NotImplementedError, AttributeError)" in t
    assert "NOT CHECKED" in t


def test_closed_shell_gets_no_stability_block_but_still_runs_scf():
    """Closed-shell stability is a singlet->triplet question the user
    rarely asked -- but the SCF itself must not go missing with it."""
    t = _script(method="RHF", spin=0)
    # Anchor on the EXECUTED form, not the bare name: a troubleshooting
    # comment elsewhere in the script mentions `mf.stability()` as a
    # thing the user could try, and a substring test reads that as a
    # call.  (Same error as the `.pyscf.log` wrapper test, fixed today.)
    assert "_internal = mf.stability()[0]" not in t
    assert "=== Stage: SCF + stability ===" not in t
    assert "mf.kernel()" in t


def test_open_shell_single_point_does_not_run_two_scfs():
    """The stability block already converged it; a second kernel would
    repeat the work and discard the stabilised orbitals."""
    t = _script(method="UHF", spin=2, optimize=False)
    assert t.count("e = mf.kernel()") == 1
