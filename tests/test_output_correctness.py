"""Output-correctness invariants for SIESTA + PySCF generators.

These tests pin the *generated artefact* against what each engine
needs to produce a scientifically correct calculation.  They live
separately from test_pyscf_spec.py because they cover both engines
and they're the layer that should have caught the C1-C4 bugs.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.pyscf import PySCFConfig, render_script
from molbuilder.siesta import SiestaConfig, render_fdf
from molbuilder.structure import Structure


@pytest.fixture
def small_struct():
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0, 0, 0], [0.957, 0, 0], [-0.24, 0.927, 0]]),
        title="water", vacuum=(12.0, 12.0, 12.0))


# --------------------------------------------------------------------- #
#  C1 — _initial.xyz captured BEFORE optimization mutates mol           #
# --------------------------------------------------------------------- #


def test_c1_initial_xyz_captured_before_the_optimization(small_struct):
    """Spec: capture the user's actual input geometry NOW, before the
    optimization has a chance to modify it.
    Otherwise _initial.xyz would save the post-stage-N geometry because
    ``mol_eq`` shadows ``mol`` (and reset() rebinds mf.mol).
    """
    text = render_script(small_struct, PySCFConfig())

    # 2026-05-27: _initial.xyz path routes through _mb_outfile() so it
    # lands next to the script regardless of cwd at run time.
    save_pos = text.find(
        '_save_structure(mol, _mb_outfile(JOB + "_initial.xyz")')
    assert save_pos != -1, "no _initial.xyz save call found"

    # The optimization comes after the SCF setup.
    opt_pos = text.find("mol_eq = _mb_run_optimization(")
    assert opt_pos != -1, "no optimization call found"

    # The save MUST come before it.
    assert save_pos < opt_pos, (
        "_initial.xyz is saved AFTER the optimization starts; mol "
        "may have been rebound to mol_eq via mf.reset() by then so "
        "the file would contain the post-relax geometry, not the "
        "user's input."
    )


def test_c1_initial_xyz_save_helper_defined_early(small_struct):
    """Corollary: the pair writer must be defined before _initial.xyz is
    called, which means before the gto.M(mol = ...) line."""
    text = render_script(small_struct, PySCFConfig())
    helper_pos = text.find("def _save_structure(")
    mol_pos    = text.find("mol = gto.M(")
    assert helper_pos != -1 and mol_pos != -1
    assert helper_pos < mol_pos, (
        "_save_structure is defined AFTER mol is built; it can't be "
        "called "
        "to snapshot the input geometry."
    )


# --------------------------------------------------------------------- #
#  C2 — SIESTA spin polarisation                                        #
# --------------------------------------------------------------------- #


def test_c2_the_spin_keyword_is_the_current_one(small_struct):
    """The electronic state is spelled in SIESTA 5's one ``Spin <option>``
    keyword, and a pinned count as the two-line ``Spin.Fix`` + ``Spin.Total``
    pair (gap #1) -- never the retired ``SpinPolarized`` flag, never the
    single ``SpinTotal <v>`` token SIESTA silently ignored (gap #2).  The
    two spellings converge on one variable in 5.4.2's ``spin_subs.F90``, and
    the manual deprecates the old booleans (verified 2026-08-15)."""
    import re
    closed = render_fdf(small_struct, SiestaConfig(verbose_comments=False))
    assert re.search(r"^Spin\s+non-polarized\s*$", closed, re.M), closed
    assert not re.search(r"^\s*Spin\.Fix", closed, re.M)
    fdf = render_fdf(small_struct,
                     SiestaConfig(spin_treatment="unrestricted",
                                  unpaired_electrons=2,
                                  verbose_comments=False))
    assert re.search(r"^Spin\s+polarized\s*$", fdf, re.M), fdf
    assert re.search(r"^\s*Spin\.Fix\s+\.true\.\s*$", fdf, re.M)
    assert re.search(r"^\s*Spin\.Total\s+2\.0\s*$", fdf, re.M)
    for text in (closed, fdf):
        assert not re.search(r"^SpinPolarized\b", text, re.M)
        assert "SpinTotal " not in text


# --------------------------------------------------------------------- #
#  C3 — stages loop: assert_convergence is per-stage                    #
# --------------------------------------------------------------------- #


def test_c3_the_deck_renders_its_own_non_convergence_policy(small_struct):
    """proceed / continue / halt, for THIS rung.

    A deck is one rung (`stages.md` § 1.1a), so the policy it renders is its own
    and the branch is chosen at render time rather than dispatched at run time
    over a table of rungs.

    **What retired with the loop, and where it went.** The old version also
    asserted an ``is_final`` override -- the in-script loop forced the last rung
    to `halt` whatever the user declared, so that no knob could silently ship a
    non-converged answer. A deck no longer knows whether it is last, so the
    script cannot enforce that. The guarantee is not abandoned; it is homeless,
    and it is recorded as such in `archive/2026-09-01-roadmap.md` § 6 rather than quietly dropped.
    """
    for policy, expected in (("halt", "mol_eq = _mb_run_optimization(_hard_fail=True)"),
                             ("proceed", "mol_eq = _mb_run_optimization(_hard_fail=False)"),
                             ("continue", "for _attempt in range(_budget):")):
        text = render_script(small_struct,
                             PySCFConfig(on_nonconvergence=policy))
        assert expected in text, f"policy {policy!r} did not render {expected!r}"

    text = render_script(small_struct, PySCFConfig())
    helper = text.split("def _mb_run_optimization(")[1].split("\n\n")[0]
    assert "assert_convergence    = _hard_fail" in helper, (
        "the single call site must thread assert_convergence through the "
        f"helper's parameter rather than hardcoding it.  helper:\n{helper}")
    assert "STAGES = [" not in text, "the in-script ladder is retired"

