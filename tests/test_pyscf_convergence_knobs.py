"""SCF convergence: the gradient criterion, and second-order SCF.

Two parameters that decided real numbers while being invisible
(plan P2b / P2c).

``conv_tol_grad`` was never set, so PySCF derived it -- verified against
the installed 2.13.0 ``scf.hf.kernel``::

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)

The shipped ``conv_tol = 1e-9`` therefore converged the ORBITAL GRADIENT
only to ~3.2e-5, and the gradient is what the forces come from.  Per-stage
tightening of ``scf_conv_tol`` moved the criterion that matters as a
square root.

``mf.newton()`` appeared nowhere: the escalation toolkit stopped at
``level_shift`` / ``damp`` / ``diis_space``, with no rung for an SCF that
oscillates indefinitely.

Law A: a parameter is an explicit field (one declaration -> UI + template
+ doc) or it is reported in ``_RUNTIME_INFO``.  Never neither -- so these
tests check BOTH the emission and the readback.

End-to-end behaviour was verified by RUNNING generated scripts under
molbuilder-pySCF, not by reading the emitter:

    [molbuilder] SCF convergence: energy 1.0e-09 Hartree, orbital
                 gradient 1.0e-06 (explicit); solver SecondOrderDFUKS.
    [molbuilder] SCF convergence: energy 1.0e-09 Hartree, orbital
                 gradient 3.2e-05 (derived: sqrt(conv_tol)); solver DFUKS.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import render_script
from molbuilder.structure import Structure

_BASE = dict(basis="sto-3g", optimize=False, compute_frequencies=False,
             write_molwatch_log=False, write_trajectory=False,
             save_optimized_xyz=False, save_initial_xyz=False,
             chkfile=False, dispersion=None)


def _o2():
    return Structure(elements=["O", "O"],
                     positions=np.array([[0., 0., 0.], [0., 0., 1.21]]))


def _script(method="UKS", spin=2, **kw):
    cfg = PySCFConfig(job_name="j", method=method, spin=spin,
                      **{**_BASE, **kw})
    return render_script(_o2(), cfg)


# ------------------------------------------------------------------ #
#  P2c -- conv_tol_grad                                              #
# ------------------------------------------------------------------ #

def test_the_field_exists_with_ui_metadata():
    """Law A's first half: an explicit field carries the metadata that
    makes ONE declaration reach the UI, the template and the doc.  A
    field with no label/help is invisible where the user sets it."""
    f = PySCFConfig.__dataclass_fields__["scf_conv_tol_grad"]
    assert f.metadata["section"] == "SCF"
    assert f.metadata["engine_key"] == "mf.conv_tol_grad"
    assert f.metadata["label"]
    assert f.metadata["help"]
    # Tightens stage-to-stage, exactly like the energy tolerance it
    # qualifies -- not a one-off profile choice.
    assert f.metadata["workflow_group"] == "stage"


def test_explicit_gradient_tolerance_is_emitted():
    t = _script(scf_conv_tol_grad=1e-6)
    assert "mf.conv_tol_grad = 1e-06" in t


def test_default_leaves_pyscfs_derivation_alone():
    """Default 0.0 must NOT emit an assignment.

    Picking a number here would silently re-tune every existing run's
    SCF; the point of the field is to make the value visible, not to
    change it behind the user's back.
    """
    t = _script()
    assert "mf.conv_tol_grad =" not in t


def test_the_derived_value_is_still_stated():
    """Law A's second half: not emitting is not the same as hiding.

    A number the run obeys but the user cannot see is the same problem
    as an undocumented one, so the unset case explains the derivation
    and quotes the resulting figure.
    """
    t = _script()
    assert "sqrt(conv_tol)" in t
    # 1e-9 -> 3.16e-05, PySCF's own rule applied by the emitter.
    assert f"{math.sqrt(1e-9):.2e}" in t


def test_the_effective_value_is_reported_from_the_live_object():
    """Reported off ``mf``, not restated from the config.

    A readback that re-prints what we asked for cannot notice that
    something downstream changed it; reading the object it will
    actually converge with can.
    """
    t = _script()
    assert "_RUNTIME_INFO['scf_conv_tol_grad'] = float(mf.conv_tol_grad" in t
    assert "_RUNTIME_INFO['scf_conv_tol'] = float(mf.conv_tol)" in t


@pytest.mark.parametrize(
    "kw, expected",
    [({}, "derived"), ({"scf_conv_tol_grad": 1e-6}, "explicit")],
    ids=["derived", "explicit"],
)
def test_chosen_and_derived_are_distinguishable(kw, expected):
    """"We chose 3.2e-5" and "PySCF derived 3.2e-5" are different
    facts about a run, and only one of them is a decision."""
    t = _script(**kw)
    assert "'scf_conv_tol_grad_source'" in t
    assert expected in t


# ------------------------------------------------------------------ #
#  P2b -- SOSCF                                                      #
# ------------------------------------------------------------------ #

def test_soscf_field_exists_with_ui_metadata():
    f = PySCFConfig.__dataclass_fields__["scf_soscf"]
    assert f.metadata["section"] == "SCF"
    assert f.metadata["engine_key"] == "mf.newton()"
    assert f.metadata["label"]
    assert f.metadata["help"]
    # An SCF-algorithm choice made with the system, like level_shift.
    assert f.metadata["workflow_group"] == "profile"


def test_soscf_off_by_default():
    assert PySCFConfig.__dataclass_fields__["scf_soscf"].default is False
    assert "mf.newton()" not in _script()


def test_soscf_emits_newton():
    assert "mf = mf.newton()" in _script(scf_soscf=True)


def test_soscf_wraps_after_the_gpu_promotion():
    """Order is load-bearing in both directions.

    ``.to_gpu()`` wants the fully-assembled plain SCF object, and
    gpu4pyscf's own classes carry ``.newton()`` (checked against
    gpu4pyscf 1.7.0) -- so wrapping last works on CPU and GPU alike,
    while wrapping first would hand ``to_gpu`` a class it need not
    know.  A reordering edit would leave both strings present and only
    this test would notice.
    """
    t = _script(scf_soscf=True)
    assert t.index("_mb_to_gpu_if_enabled(mf)") < t.index("mf = mf.newton()")


def test_soscf_still_reaches_the_stability_check():
    """P2a's open-shell check must survive the wrap.

    ``mf.newton()`` returns a NEW object; if the stability block were
    emitted against the pre-wrap ``mf`` the check would silently test
    something other than what runs.  (``SecondOrderUKS`` does implement
    ``stability()`` -- probed on PySCF 2.13.0.)
    """
    t = _script(scf_soscf=True)
    assert t.index("mf = mf.newton()") < t.index("_internal = mf.stability()[0]")


def test_soscf_documents_what_changes_underneath():
    """max_cycle silently changes meaning and two knobs go inert.

    Someone reading the script needs that where the line is, not in a
    doc they have not opened -- the same reason the emitted SIESTA
    blocks explain themselves.
    """
    t = _script(scf_soscf=True)
    assert "MACRO iterations" in t
    assert "ah_level_shift" in t


def test_the_solver_is_reported_by_class_not_by_flag():
    """Reading ``type(mf).__name__`` names every decoration that ended
    up on the object -- ``SecondOrderDFUKS`` says second-order AND
    density-fitted.  A boolean we set ourselves could not."""
    t = _script(scf_soscf=True)
    assert "_RUNTIME_INFO['scf_solver_class'] = type(mf).__name__" in t
