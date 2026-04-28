"""Tests for molbuilder.pyscf_input -- runnable PySCF script generator.

We don't actually invoke PySCF here (would slow tests by ~30 s and
require a heavyweight install).  Instead we verify the generator's
*output is well-formed*: correct atom block, correct charge, correct
sections present/absent based on flags, valid Python syntax (compile()).
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import molbuilder
from molbuilder.pyscf_input import (
    PySCFConfig,
    render_script,
    convert,
    _SOLVENTS,
)
from molbuilder.structure import Structure


def _h2o() -> Structure:
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [0.957, 0.0, 0.0],
            [-0.240, 0.927, 0.0],
        ]),
        title="water",
    )


def test_default_render_compiles() -> None:
    """The default-config script must be syntactically valid Python."""
    s = _h2o()
    text = render_script(s)
    compile(text, "<rendered>", "exec")
    # Must contain the structural sections
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


def test_atom_block_format() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(verbose_comments=False))
    # Atoms appear inside a triple-quoted block; each line "El  x  y  z".
    # The element is left-padded to 2 chars with 2-space gap, and each
    # coord gets a 14-wide %.8f field.  Don't pin exact spacing, just
    # verify the values made it through.
    import re
    assert re.search(r"^\s*O\s+0\.00000000\s+0\.00000000\s+0\.00000000",
                     text, re.M)
    assert re.search(r"^\s*H\s+0\.95700000\s+0\.00000000\s+0\.00000000",
                     text, re.M)


def test_charge_explicit_overrides_auto() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(charge=-1))
    assert "charge     = -1," in text


def test_charge_auto_detect_from_phosphates() -> None:
    """Synthetic phosphate diester missing both HOP atoms -> charge = -1."""
    elements = ["C", "O", "P", "O", "O", "O", "C"]
    positions = np.array([
        [-2.5, 0.0, 0.0], [-1.4, 0.0, 0.0], [0.0, 0.0, 0.0],
        [ 0.0, 1.5, 0.0], [ 0.0, -0.8, 1.3],
        [ 1.4, 0.0, 0.0], [ 2.5, 0.0, 0.0],
    ])
    s = Structure(elements=elements, positions=positions,
                  atom_names=["C5'","O5'","P","OP1","OP2","O3'","C3'"])
    text = render_script(s, PySCFConfig(verbose_comments=False))
    assert "charge     = -1," in text


def test_no_optimize_drops_geom_block() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(optimize=False, verbose_comments=False))
    assert "mol_eq = optimize(" not in text
    assert "e = mf.kernel()" in text
    assert "_optimized.xyz" not in text


def test_preopt_block_emitted_when_enabled() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(preopt=True))
    assert "Pre-optimization" in text or "pre-optimization" in text
    assert "mol_pre" in text
    assert "mf1 = dft.RKS(mol_pre)" in text
    assert "Production-stage" in text or "Main run" in text \
           or "production functional" in text


def test_dispersion_can_be_disabled() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(dispersion=None))
    assert "mf.disp" not in text


def test_solvent_emits_pcm_block() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(solvent="water"))
    assert "from pyscf.solvent import pcm" in text
    assert "mf = pcm.PCM(mf)" in text
    eps = _SOLVENTS["water"]
    assert f"mf.with_solvent.eps = {eps}" in text


def test_uks_for_radicals() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(method="UKS", spin=1, charge=1))
    assert "mf = dft.UKS(mol)" in text
    assert "spin       = 1," in text
    assert "charge     = 1," in text


def test_threads_emit_env_pin() -> None:
    s = _h2o()
    text = render_script(s, PySCFConfig(threads=8))
    assert 'os.environ.setdefault("OMP_NUM_THREADS", "8")' in text
    assert 'os.environ.setdefault("MKL_NUM_THREADS", "8")' in text


def test_no_density_fit() -> None:
    """When density_fit is off, no `mf.density_fit()` call is emitted.

    (The verbose-mode troubleshooting block still mentions the option
    by name, so check for the call site specifically.)
    """
    s = _h2o()
    text = render_script(s, PySCFConfig(density_fit=False))
    assert "mf = mf.density_fit(" not in text
    assert "mf = mf.density_fit()" not in text


def test_invalid_method_raises() -> None:
    s = _h2o()
    try:
        render_script(s, PySCFConfig(method="MP2"))
    except ValueError:
        return
    assert False, "expected ValueError for unsupported method"


def test_invalid_solvent_raises() -> None:
    s = _h2o()
    try:
        render_script(s, PySCFConfig(solvent="liquid_helium"))
    except ValueError:
        return
    assert False, "expected ValueError for unknown solvent"


def test_invalid_optimizer_raises() -> None:
    s = _h2o()
    try:
        render_script(s, PySCFConfig(optimizer="bfgs"))
    except ValueError:
        return
    assert False, "expected ValueError for unknown optimizer"


def test_convert_xyz_to_py(tmp_path_str: str) -> None:
    import re
    s = _h2o()
    xyz_p = os.path.join(tmp_path_str, "h2o.xyz")
    py_p  = os.path.join(tmp_path_str, "h2o_relax.py")
    s.to_xyz(xyz_p)
    summary = convert(xyz_p, py_p, PySCFConfig(verbose_comments=False))
    assert summary["n_atoms"] == 3
    assert summary["py"] == py_p
    assert os.path.isfile(py_p)
    text = open(py_p).read()
    compile(text, py_p, "exec")
    assert re.search(r"^\s*O\s+0\.00000000\s+0\.00000000\s+0\.00000000",
                     text, re.M)


def test_convert_pdb_to_py(tmp_path_str: str) -> None:
    """End-to-end load+render: peptide built via molbuilder -> PDB -> .py."""
    try:
        s = molbuilder.build_peptide("AC", add_hydrogens=False)
    except ImportError:
        print("  (skip: PeptideBuilder not installed)")
        return
    pdb_p = os.path.join(tmp_path_str, "ac.pdb")
    py_p  = os.path.join(tmp_path_str, "ac.py")
    s.to_pdb(pdb_p)
    convert(pdb_p, py_p, PySCFConfig(job_name="ac_test"))
    text = open(py_p).read()
    compile(text, py_p, "exec")
    assert 'JOB = "ac_test"' in text


def test_loaded_structure_to_pyscf_script(tmp_path_str: str) -> None:
    """Mirror the FDF flow: build -> load -> render PySCF."""
    s = _h2o()
    xyz_p = os.path.join(tmp_path_str, "h2o.xyz")
    s.to_xyz(xyz_p)
    s2 = molbuilder.load(xyz_p)
    text = render_script(s2, PySCFConfig(job_name="reloaded"))
    assert 'JOB = "reloaded"' in text
    compile(text, "<reloaded>", "exec")


def test_verbose_comments_off_strips_hints() -> None:
    s = _h2o()
    text_v = render_script(s, PySCFConfig(verbose_comments=True))
    text_q = render_script(s, PySCFConfig(verbose_comments=False))
    # Verbose mode contains tuning hints; quiet mode does not
    assert "TROUBLESHOOTING" in text_v
    assert "TROUBLESHOOTING" not in text_q
    assert len(text_q) < len(text_v)


def main() -> None:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(globals()):
            if not name.startswith("test_"):
                continue
            fn = globals()[name]
            try:
                if "tmp_path_str" in fn.__code__.co_varnames:
                    fn(tmp)
                else:
                    fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                print(f"  FAIL {name}: {e}")
                failures.append(name)
            except Exception as e:
                print(f"  ERR  {name}: {type(e).__name__}: {e}")
                failures.append(name)
    if failures:
        sys.exit(f"FAILED: {failures}")
    print("OK -- pyscf_input module renders valid scripts for every "
          "config variant tested.")


if __name__ == "__main__":
    main()
