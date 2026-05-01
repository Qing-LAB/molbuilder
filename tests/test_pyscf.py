"""Tests for molbuilder.pyscf_input -- runnable PySCF script generator.

We don't actually invoke PySCF (heavyweight install, ~30 s startup).
Instead we verify the generator's output is well-formed: correct atom
block, correct charge, correct sections present/absent based on flags,
valid Python syntax (compile()).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import molbuilder
from molbuilder.pyscf import (
    PySCFConfig,
    render_script,
    convert,
)
from molbuilder.pyscf.input import _SOLVENTS
from molbuilder.structure import Structure


@pytest.fixture
def h2o():
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [0.957, 0.0, 0.0],
            [-0.240, 0.927, 0.0],
        ]),
        title="water",
    )


# --------------------------------------------------------------------- #
#  Sanity / sections                                                    #
# --------------------------------------------------------------------- #


def test_default_render_compiles(h2o):
    text = render_script(h2o)
    compile(text, "<rendered>", "exec")
    for needle in (
        "import os",
        "from pyscf import gto, scf, dft",
        "from pyscf.geomopt.geometric_solver import optimize",
        "mol = gto.M(",
        "mf = dft.RKS(mol)",
        'mf.xc = "B3LYP"',
        "mf = mf.density_fit()",
        'mf.disp = "d3bj"',
        "mol_eq = optimize(",
        "_save_xyz(",
    ):
        assert needle in text, f"missing {needle!r}"


def test_atom_block_format(h2o):
    text = render_script(h2o, PySCFConfig(verbose_comments=False))
    assert re.search(r"^\s*O\s+0\.00000000\s+0\.00000000\s+0\.00000000",
                     text, re.M)
    assert re.search(r"^\s*H\s+0\.95700000\s+0\.00000000\s+0\.00000000",
                     text, re.M)


# --------------------------------------------------------------------- #
#  Charge handling                                                      #
# --------------------------------------------------------------------- #


def test_charge_explicit_overrides_auto(h2o):
    text = render_script(h2o, PySCFConfig(charge=-1))
    assert "charge     = -1," in text


def test_charge_auto_detect_from_phosphates(deprotonated_diester):
    text = render_script(deprotonated_diester,
                         PySCFConfig(verbose_comments=False))
    assert "charge     = -1," in text


# --------------------------------------------------------------------- #
#  Section toggles                                                      #
# --------------------------------------------------------------------- #


def test_no_optimize_drops_geom_block(h2o):
    text = render_script(h2o, PySCFConfig(optimize=False, verbose_comments=False))
    assert "mol_eq = optimize(" not in text
    assert "e = mf.kernel()" in text
    assert "_optimized.xyz" not in text


def test_preopt_block_emitted_when_enabled(h2o):
    text = render_script(h2o, PySCFConfig(preopt=True))
    assert "Pre-optimization" in text or "pre-optimization" in text
    assert "mol_pre" in text
    assert "mf1 = dft.RKS(mol_pre)" in text


def test_preopt_does_not_rebuild_mol_via_gto_M(h2o):
    """Regression: pre-opt must NOT regenerate the production mol via
    `gto.M(...)` because that opens <JOB>.log in 'w' mode and wipes the
    pre-opt log entries.  We reuse mol_pre instead."""
    text = render_script(h2o, PySCFConfig(preopt=True))
    # The post-preopt rebuild block should NOT contain a fresh gto.M
    # CALL (as opposed to a comment mentioning it) between pre-opt's
    # optimize() and the production mf setup.
    after_preopt = text.split("Pre-opt done")[1]
    before_main_mf = after_preopt.split("mf = ")[0]
    code_lines = [ln for ln in before_main_mf.splitlines()
                  if not ln.lstrip().startswith("#")]
    code_only = "\n".join(code_lines)
    assert "gto.M(" not in code_only, (
        "post-preopt rebuild uses gto.M(...) which truncates <JOB>.log; "
        "should reuse mol_pre instead"
    )
    # And the mol = mol_pre line should be there.
    assert "mol = mol_pre" in code_only


def test_preopt_writes_its_own_trajectory_when_enabled(h2o):
    """When write_trajectory + preopt + geometric, pre-opt's optimize()
    must also pass prefix=JOB+'_preopt' so molwatch can watch the pre-
    opt stage's streaming trajectory file."""
    text = render_script(h2o,
                         PySCFConfig(preopt=True, write_trajectory=True))
    assert 'prefix            = JOB + "_preopt"' in text
    # Production stage still uses _geom prefix.
    assert 'prefix                = JOB + "_geom"' in text


def test_preopt_basis_change_triggers_rebuild(h2o):
    """If the production basis differs from the pre-opt basis, mol must
    have its basis swapped and rebuilt; otherwise no rebuild needed."""
    # Same basis -> no rebuild
    same = render_script(h2o, PySCFConfig(preopt=True,
                                          basis="def2-SVP",
                                          preopt_basis="def2-SVP"))
    after_same = same.split("mol = mol_pre")[1].split("mf = ")[0]
    assert "mol.build" not in after_same

    # Different basis -> rebuild
    diff = render_script(h2o, PySCFConfig(preopt=True,
                                          basis="def2-TZVP",
                                          preopt_basis="def2-SVP"))
    after_diff = diff.split("mol = mol_pre")[1].split("mf = ")[0]
    assert 'mol.basis = "def2-TZVP"' in after_diff
    assert "mol.build(dump_input=False)" in after_diff


def test_dispersion_can_be_disabled(h2o):
    text = render_script(h2o, PySCFConfig(dispersion=None))
    assert "mf.disp" not in text


def test_solvent_emits_pcm_block(h2o):
    text = render_script(h2o, PySCFConfig(solvent="water"))
    assert "from pyscf.solvent import pcm" in text
    assert "mf = pcm.PCM(mf)" in text
    eps = _SOLVENTS["water"]
    assert f"mf.with_solvent.eps = {eps}" in text


def test_uks_for_radicals(h2o):
    text = render_script(h2o, PySCFConfig(method="UKS", spin=1, charge=1))
    assert "mf = dft.UKS(mol)" in text
    assert "spin       = 1," in text
    assert "charge     = 1," in text


def test_threads_emit_env_pin(h2o):
    text = render_script(h2o, PySCFConfig(threads=8))
    assert 'os.environ.setdefault("OMP_NUM_THREADS", "8")' in text
    assert 'os.environ.setdefault("MKL_NUM_THREADS", "8")' in text


def test_no_density_fit(h2o):
    """When density_fit is off, no `mf.density_fit()` call is emitted.

    (Verbose-mode troubleshooting block still mentions the option by
    name, so check for the call site specifically.)
    """
    text = render_script(h2o, PySCFConfig(density_fit=False))
    assert "mf = mf.density_fit(" not in text
    assert "mf = mf.density_fit()" not in text


def test_verbose_comments_off_strips_hints(h2o):
    text_v = render_script(h2o, PySCFConfig(verbose_comments=True))
    text_q = render_script(h2o, PySCFConfig(verbose_comments=False))
    assert "TROUBLESHOOTING" in text_v
    assert "TROUBLESHOOTING" not in text_q
    assert len(text_q) < len(text_v)


# --------------------------------------------------------------------- #
#  Validation                                                           #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("kwargs, name", [
    ({"method":    "MP2"},            "method"),
    ({"solvent":   "liquid_helium"},  "solvent"),
    ({"optimizer": "bfgs"},           "optimizer"),
])
def test_invalid_inputs_raise(h2o, kwargs, name):
    with pytest.raises(ValueError):
        render_script(h2o, PySCFConfig(**kwargs))


# --------------------------------------------------------------------- #
#  convert() -- file in, .py out                                        #
# --------------------------------------------------------------------- #


def test_convert_xyz_to_py(h2o, tmp_path):
    xyz_p = tmp_path / "h2o.xyz"
    py_p  = tmp_path / "h2o_relax.py"
    h2o.to_xyz(str(xyz_p))
    summary = convert(str(xyz_p), str(py_p),
                      PySCFConfig(verbose_comments=False))
    assert summary["n_atoms"] == 3
    assert summary["py"] == str(py_p)
    text = py_p.read_text()
    compile(text, str(py_p), "exec")
    assert re.search(r"^\s*O\s+0\.00000000\s+0\.00000000\s+0\.00000000",
                     text, re.M)


def test_convert_pdb_to_py(tmp_path):
    """End-to-end: peptide built via molbuilder -> PDB -> .py."""
    pytest.importorskip("PeptideBuilder")
    s = molbuilder.build_peptide("AC", add_hydrogens=False)
    pdb_p = tmp_path / "ac.pdb"
    py_p  = tmp_path / "ac.py"
    s.to_pdb(str(pdb_p))
    convert(str(pdb_p), str(py_p), PySCFConfig(job_name="ac_test"))
    text = py_p.read_text()
    compile(text, str(py_p), "exec")
    assert 'JOB = "ac_test"' in text


def test_loaded_structure_to_pyscf_script(h2o, tmp_path):
    """Mirror the FDF flow: build -> load -> render PySCF."""
    xyz_p = tmp_path / "h2o.xyz"
    h2o.to_xyz(str(xyz_p))
    s2 = molbuilder.load(str(xyz_p))
    text = render_script(s2, PySCFConfig(job_name="reloaded"))
    assert 'JOB = "reloaded"' in text
    compile(text, "<reloaded>", "exec")
