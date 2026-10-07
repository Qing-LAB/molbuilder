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
+ doc) or it is reported in ``_RUNTIME_INFO``.  Never neither.
"""
from __future__ import annotations


from molbuilder.config.pyscf import PySCFConfig


# ------------------------------------------------------------------ #
#  P2c -- conv_tol_grad                                              #
# ------------------------------------------------------------------ #

def test_the_field_exists_with_ui_metadata():
    """Law A's first half: an explicit field carries the metadata that
    makes ONE declaration reach the UI, the template and the doc.  A
    field with no label/help is invisible where the user sets it."""
    f = PySCFConfig.__dataclass_fields__["scf_conv_tol_grad"]
    assert f.metadata["engine_key"] == "mf.conv_tol_grad"
    assert f.metadata["label"]
    # The help lives in the CATALOGUE -- one home, asked
    # through `template.help_for`, which is what every surface reads.
    from molbuilder.template import help_for
    assert help_for("scf_conv_tol_grad")
    # Tightens stage-to-stage, exactly like the energy tolerance it
    # qualifies -- not a one-off profile choice.
    assert f.metadata["workflow_group"] == "stage"


# ------------------------------------------------------------------ #
#  P2b -- SOSCF                                                      #
# ------------------------------------------------------------------ #

def test_soscf_field_exists_with_ui_metadata():
    f = PySCFConfig.__dataclass_fields__["scf_soscf"]
    assert f.metadata["engine_key"] == "mf.newton()"
    assert f.metadata["label"]
    # The help lives in the CATALOGUE -- one home, asked
    # through `template.help_for`, which is what every surface reads.
    from molbuilder.template import help_for
    assert help_for("scf_soscf")
    # An SCF-algorithm choice made with the system, like level_shift.
    assert f.metadata["workflow_group"] == "profile"


# --------------------------------------------------------------------- #
#  max_memory: one allocation item for both engines, and UNSET means    #
#  no cap (`template.md` § 2, G1: an allocation item states no value    #
#  at floor 2 -- a machine fact has no place in a portable description) #
# --------------------------------------------------------------------- #


def test_memory_is_one_item_across_both_engines():
    """§ 6.3's merge, and the reason PySCF's declaration had to change.

    Both engines answer *"how much memory may this run use"* the same way --
    unset is no cap, and nothing fills one (`template.md` § 2).  Under § 5.6's
    mechanism they are one item by being spelled the same, so their
    declarations must agree; they disagreed on six attributes until this fix.
    """
    from molbuilder.template import declarations_for
    from molbuilder.config.siesta import SiestaConfig
    s = {d.name: d for d in declarations_for(SiestaConfig)}["max_memory_mb"]
    p = {d.name: d for d in declarations_for(PySCFConfig)}["max_memory_mb"]
    for attr in ("kind", "type", "default", "category",
                 "allocation", "optional", "unit"):
        assert getattr(s, attr) == getattr(p, attr), (
            f"max_memory_mb.{attr}: siesta={getattr(s, attr)!r} "
            f"pyscf={getattr(p, attr)!r} -- the two halves of ONE item "
            f"disagree (template.md § 6.3)")
