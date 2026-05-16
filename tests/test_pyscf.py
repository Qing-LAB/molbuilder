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


def test_hf_render_omits_dft_import(h2o):
    """Pure-HF runs (method=RHF/UHF, no preopt) don't touch the dft
    module; the `from pyscf import ... dft` should drop out so the
    script reads `from pyscf import gto, scf` instead.  Pre-opt always
    uses DFT (cheap warm-up), so even an HF production run keeps the
    dft import when preopt=True.

    Tier 2 #13 from the deep code review."""
    text = render_script(h2o, PySCFConfig(method="RHF", preopt=False))
    compile(text, "<rendered>", "exec")
    assert "from pyscf import gto, scf" in text
    assert "from pyscf import gto, scf, dft" not in text
    # Sanity: HF script doesn't accidentally use dft.* anywhere.
    assert "dft." not in text or all(
        ln.lstrip().startswith("#") for ln in text.splitlines() if "dft." in ln
    ), "HF script should not reference dft.* in live code"

    # With preopt=True the pre-opt block forces DFT regardless of the
    # production method (preopt is always a cheap functional warm-up).
    text2 = render_script(h2o, PySCFConfig(method="RHF", preopt=True))
    assert "from pyscf import gto, scf, dft" in text2


def test_atom_block_format(h2o):
    text = render_script(h2o, PySCFConfig(verbose_comments=False))
    assert re.search(r"^\s*O\s+0\.00000000\s+0\.00000000\s+0\.00000000",
                     text, re.M)
    assert re.search(r"^\s*H\s+0\.95700000\s+0\.00000000\s+0\.00000000",
                     text, re.M)


def test_geometric_optparams_accepts_pyscf_optimize_kwargs():
    """Pin the PySCF/geomeTRIC API contract that the generator depends on.

    The generated script calls::

        optimize(mf,
                 convergence_energy = 1e-6,
                 convergence_grms   = 3e-4,
                 convergence_gmax   = 4.5e-4,
                 maxsteps           = N,
                 callback           = ...)

    PySCF's ``geometric_solver.optimize(method, **kwargs)`` forwards
    ``**kwargs`` into geomeTRIC's ``OptParams(**kwargs)`` constructor,
    which accepts the lowercase ``convergence_*`` keys and stores them
    on the instance as ``Convergence_*`` (capital C).  TIER 2 #8 of
    the 2026-05-05 review claimed the kwargs raise ``TypeError`` --
    they don't, but the only way to be sure (and stay sure across
    PySCF / geomeTRIC version bumps) is to introspect the actual API.

    Probe ``OptParams`` directly with our exact key names + sentinel
    values, then read the canonical attributes back.  No subprocess,
    no dependency on PySCF -- if either side renames or rejects the
    keys, this fails at unit-test time with a clean message rather
    than letting a generated script crash at user runtime.
    """
    pytest.importorskip("geometric")
    from geometric.optimize import OptParams

    # Sentinel values picked to be unmistakable in error messages.
    p = OptParams(convergence_energy=1.234e-6,
                  convergence_grms  =5.678e-4,
                  convergence_gmax  =9.012e-4)
    assert p.Convergence_energy == 1.234e-6, (
        "geomeTRIC OptParams stopped honouring `convergence_energy`; "
        "molbuilder's generated script will silently use the default "
        "or crash at runtime"
    )
    assert p.Convergence_grms == 5.678e-4, (
        "geomeTRIC OptParams stopped honouring `convergence_grms`"
    )
    assert p.Convergence_gmax == 9.012e-4, (
        "geomeTRIC OptParams stopped honouring `convergence_gmax`"
    )


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


def test_molwatch_log_instantiated_before_preopt(h2o):
    """Critical UX guarantee: with preopt=True the user shouldn't have
    to wait for preopt to finish before .molwatch.log appears.  Preopt
    can take hours on a real molecule; the Watch tab needs SOMETHING
    to load from second one.

    Pin the source ordering: `_molwatch = MolwatchEmitter(...)` must
    appear BEFORE `mol_pre = optimize(mf1, ...)`.  Both callbacks
    (mf1.callback for SCF, optimize(callback=...) for opt steps) must
    also wire into the preopt stage so steps stream from the start."""
    text = render_script(h2o, PySCFConfig(preopt=True))
    inst_at      = text.find('_molwatch = MolwatchEmitter(JOB')
    preopt_at    = text.find("mol_pre = optimize(")
    prod_at      = text.find("mol_eq = optimize(")
    mf1_callback = text.find("mf1.callback = _molwatch.scf_cycle_hook")
    mf_callback  = text.find("mf.callback = _molwatch.scf_cycle_hook")
    preopt_step  = text.find("callback          = _molwatch.opt_step_hook")
    prod_step    = text.find("callback              = _molwatch.opt_step_hook")
    # Every position must be present (>= 0) and in the right order.
    for name, off in [
        ("_molwatch instantiation", inst_at),
        ("mol_pre = optimize(",     preopt_at),
        ("mol_eq = optimize(",      prod_at),
        ("mf1.callback wiring",     mf1_callback),
        ("mf.callback wiring",      mf_callback),
        ("preopt opt_step callback", preopt_step),
        ("production opt_step callback", prod_step),
    ]:
        assert off >= 0, f"missing in script: {name}"
    # Order:
    #   inst (creates .molwatch.log immediately)
    #     < mf1.callback wiring (preopt SCF hook)
    #       < preopt optimize( ... callback=opt_step_hook ... )
    #         < mf.callback wiring (production SCF hook, post-rebind)
    #           < production optimize( ... callback=opt_step_hook ... )
    # `preopt_step` and `prod_step` lie INSIDE their respective
    # optimize() argument lists, so they fall between the opening
    # `optimize(` of one stage and the opening `optimize(` of the next.
    assert inst_at < mf1_callback < preopt_at < preopt_step < mf_callback < prod_at < prod_step, (
        "molwatch wiring is out of order; expected "
        "inst < mf1_cb < preopt_at < preopt_step < mf_cb < prod_at < prod_step.  "
        f"Got: inst={inst_at}, mf1_cb={mf1_callback}, preopt_at={preopt_at}, "
        f"preopt_step={preopt_step}, mf_cb={mf_callback}, prod_at={prod_at}, "
        f"prod_step={prod_step}"
    )


def test_molwatch_log_instantiation_skipped_when_optimizer_is_berny(h2o):
    """The molwatch log emitter requires the geomeTRIC `callback=` API.
    Berny doesn't expose an equivalent hook, so we skip emission when
    optimizer != 'geometric' rather than emit an unwired class."""
    text = render_script(h2o,
                         PySCFConfig(optimize=True, optimizer="berny",
                                     write_molwatch_log=True))
    assert "MolwatchEmitter" not in text
    assert ".molwatch.log" not in text or text.count(".molwatch.log") <= 1


def test_stability_analysis_skipped_for_closed_shell(h2o):
    """Closed-shell scripts (RKS / RHF) shouldn't carry a
    `mf.stability()` call -- closed-shell stability is rarely the
    user's concern and the call adds noise to a tutorial script
    that's already dense.  Open-shell coverage is in
    test_science_gaps.test_gap_4_pyscf_uks_emits_stability_analysis."""
    text = render_script(h2o, PySCFConfig(method="RKS"))
    code_lines = [ln for ln in text.splitlines()
                  if not ln.lstrip().startswith("#")]
    assert not any("mf.stability(" in ln for ln in code_lines), (
        "RKS script should not emit a non-commented mf.stability() call"
    )


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


def test_dispersion_none_string_does_not_crash_pyscf(h2o):
    # PySCF's check_disp raises NotImplementedError if mf.disp is the
    # literal string "none" (only None / 0 / a real version like "d3bj"
    # are accepted).  Make sure the string sentinel is treated like None.
    text = render_script(h2o, PySCFConfig(dispersion="none"))
    assert "mf.disp" not in text


def test_preopt_dispersion_none_string_does_not_crash_pyscf(h2o):
    text = render_script(h2o,
                        PySCFConfig(preopt=True, preopt_dispersion="none"))
    assert "mf1.disp" not in text


def test_solvent_emits_pcm_block(h2o):
    text = render_script(h2o, PySCFConfig(solvent="water"))
    # The pcm import remains because importing it patches the .PCM()
    # method onto the SCF base class.
    assert "from pyscf.solvent import pcm" in text
    # PySCF 2.x SCF-method form (P1).  The older ``pcm.PCM(mf)`` form
    # returns a bare solvent object that doesn't expose .with_solvent
    # and would crash the next two lines at runtime.
    assert "mf = mf.PCM()" in text
    assert "pcm.PCM(mf)" not in text
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


# --------------------------------------------------------------------- #
#  PCM solvent uses the SCF-method form (P1)                            #
# --------------------------------------------------------------------- #


def test_pcm_uses_mf_method_form(h2o):
    """Generated script must wrap PCM via ``mf = mf.PCM()`` (PySCF 2.x
    SCF-method form), not the lower-level ``pcm.PCM(mf)`` constructor
    -- the latter returns a bare solvent object with no
    ``.with_solvent`` attribute and the next two lines used to crash."""
    text = render_script(h2o, PySCFConfig(solvent="water"))
    compile(text, "<solvent>", "exec")
    assert "mf = mf.PCM()" in text
    assert "pcm.PCM(mf)" not in text
    # And the with_solvent settings still land on the wrapped mf.
    assert "mf.with_solvent.method" in text
    assert "mf.with_solvent.eps"    in text


# --------------------------------------------------------------------- #
#  Pre-opt mf1 inherits hard-SCF settings from cfg (P2)                 #
# --------------------------------------------------------------------- #


def test_preopt_inherits_init_guess(h2o):
    """When cfg.preopt=True, mf1 must get the production
    init_guess so a stiff SCF that needed e.g. huckel to converge
    actually has it during the warm-up too."""
    text = render_script(
        h2o,
        PySCFConfig(preopt=True, scf_init_guess="huckel"),
    )
    assert 'mf1.init_guess = "huckel"' in text


def test_preopt_inherits_level_shift_and_diis(h2o):
    """level_shift, diis_space, and damp must be mirrored onto mf1."""
    text = render_script(
        h2o,
        PySCFConfig(preopt=True,
                    level_shift=0.2,
                    diis_space=16,
                    damp=0.3),
    )
    assert "mf1.level_shift = 0.2" in text
    assert "mf1.diis_space = 16"   in text
    assert "mf1.damp = 0.3"        in text


def test_preopt_omits_default_diis_and_damp(h2o):
    """At default cfg.diis_space=8 and cfg.damp=0.0 we keep the script
    clean (no redundant ``mf1.diis_space = 8`` line)."""
    text = render_script(h2o, PySCFConfig(preopt=True))
    assert "mf1.diis_space" not in text
    assert "mf1.damp"       not in text


# --------------------------------------------------------------------- #
#  Production stage uses mf.reset(mol_eq) not mf.mol = (P3)             #
# --------------------------------------------------------------------- #


def test_post_opt_uses_mf_reset_not_attribute_assignment(h2o):
    """After geomopt completes, the script re-evaluates at mol_eq.
    PySCF 2.x's canonical form is ``mf.reset(mol_eq)`` which drops
    cached integrals; ``mf.mol = mol_eq`` leaves them stale and
    kernel() may use integrals built at the previous geometry."""
    text = render_script(h2o, PySCFConfig(optimize=True))
    assert "mf.reset(mol_eq)" in text
    assert "mf.mol = mol_eq"  not in text


def test_post_opt_warm_starts_from_converged_dm(h2o):
    """R1: the post-opt re-eval must pass dm0=dm_prev to mf.kernel(),
    where dm_prev is the converged DM at the previous geometry (or
    None if the geomopt left mf in a partial state).  Without the
    warm-start, kernel() restarts from MINAO (the default init_guess)
    and burns 10-30 SCF cycles re-converging from scratch rather than
    warm-starting from the line-search density."""
    text = render_script(h2o, PySCFConfig(optimize=True))
    # The DM snapshot is guarded so a failed/partial optimize doesn't
    # crash on mo_occ=None inside make_rdm1().
    assert "mf.make_rdm1()" in text
    assert "mf.mo_coeff is not None" in text
    assert "mf.mo_occ is not None"   in text
    # The kernel() call passes the (possibly None) DM as the warm start.
    assert "mf.kernel(dm0=dm_prev)"   in text
    # And the bare mf.kernel() form must NOT appear in the post-opt path
    # (it can still appear in the single-point path `e = mf.kernel()`).
    rest = text.split("mf.reset(mol_eq)", 1)[-1]
    # Take only the next ~5 lines after reset() to scope the assertion.
    rest = "\n".join(rest.split("\n")[:8])
    assert "mf.kernel()" not in rest


# --------------------------------------------------------------------- #
#  Bridge: choices accept any case via the bridge (R2)                  #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("flag,value", [
    ("--method",          "uks"),       # default RKS / choices include UKS
    ("--method",          "Uks"),
    ("--scf-init-guess",  "HUCKEL"),
    ("--optimizer",       "BERNY"),
    ("--dispersion",      "D3BJ"),      # R4
    ("--dispersion",      "NONE"),      # R4 + cmd_pyscf coercion
])
def test_pyscf_choice_accepts_mixed_case(flag, value, monkeypatch, tmp_path):
    """R2: ``case_sensitive=False`` on the bridge's click.Choice lets
    users type the choice in any case.  Without this, the renderer's
    own ``.upper()`` is dead code at the CLI layer."""
    from molbuilder import cli as _cli
    captured = {}

    def fake_convert(input_path, py_path, config):
        captured["cfg"] = config
        return {"py": py_path, "n_atoms": 0, "charge": 0, "label": "x"}
    monkeypatch.setattr("molbuilder.pyscf.convert", fake_convert)

    in_xyz = tmp_path / "h2.xyz"
    in_xyz.write_text("2\nh2\nH 0 0 0\nH 0.74 0 0\n")
    out_py = tmp_path / "h2.py"
    rc = _cli.main(["pyscf", str(in_xyz), str(out_py), flag, value])
    assert rc == 0
    # The captured config must hold a value (either the original-case
    # match from the choices list or the post-coercion None for
    # --dispersion NONE).
    assert "cfg" in captured


# --------------------------------------------------------------------- #
#  Dispersion choices reject typos at parse time (R4)                   #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("flag,bad_val", [
    ("--dispersion",        "d3-bj"),
    ("--dispersion",        "Grimme-D4"),
    ("--preopt-dispersion", "d4bj"),
])
def test_pyscf_dispersion_typo_rejected_at_parse_time(
        flag, bad_val, tmp_path):
    """R4: dispersion / preopt_dispersion now carry choices metadata
    so a typo fails at CLI parse time instead of reaching PySCF."""
    from molbuilder import cli as _cli
    in_xyz = tmp_path / "h2.xyz"
    in_xyz.write_text("2\nh2\nH 0 0 0\nH 0.74 0 0\n")
    out_py = tmp_path / "h2.py"
    with pytest.raises(SystemExit) as exc:
        _cli.main(["pyscf", str(in_xyz), str(out_py), flag, bad_val])
    assert exc.value.code == 2


# --------------------------------------------------------------------- #
#  ECP "none" sentinel works from the Python API too (P4)               #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("ecp_value", ["", "none", "None", "NONE", "  none  "])
def test_python_api_ecp_none_sentinel_disables_ecp(h2o, ecp_value):
    """Python-API users who pass ``PySCFConfig(ecp=...)`` with the
    case-insensitive none sentinel (or empty string) get the same
    behaviour as ``--ecp none`` from the CLI: no ``ecp=`` in
    gto.M(...).  Without the normalisation the script reaches
    ``gto.M(ecp="none")`` which raises at runtime."""
    text = render_script(h2o, PySCFConfig(ecp=ecp_value))
    compile(text, "<ecp>", "exec")
    # The literal strings ecp="none" / ecp="None" must NEVER appear in
    # the generated gto.M(...) call.
    assert 'ecp     = "none"' not in text
    assert 'ecp     = "None"' not in text
    assert 'ecp = "none"'     not in text
    assert 'ecp = "None"'     not in text


def test_python_api_ecp_explicit_lanl2dz_passes_through(h2o):
    """An explicit ECP name still propagates verbatim."""
    text = render_script(h2o, PySCFConfig(ecp="lanl2dz", basis="cc-pVDZ"))
    assert ('ecp     = "lanl2dz"' in text
            or 'ecp = "lanl2dz"' in text)



# ---- Staged-relaxation suffix (job-layout v1) ---------------------------- #


def test_pyscf_molwatch_emitter_uses_stage_suffix(h2o):
    """When cfg.stage is 1/2/3 the inlined ``MolwatchEmitter(...)``
    call writes to ``<JOB>-stage<N>.molwatch.log`` so stages don't
    overwrite each other in a shared directory."""
    text = render_script(h2o, PySCFConfig(stage=2, job_name="my-job"))
    # Quote style is repr()'s choice (single or double); the contract
    # is the JOB + "<suffix>" expression with the right suffix.
    assert ("MolwatchEmitter(JOB + '-stage2.molwatch.log'" in text
            or 'MolwatchEmitter(JOB + "-stage2.molwatch.log"' in text)


def test_pyscf_molwatch_emitter_unsuffixed_when_stage_is_none(h2o):
    text = render_script(h2o, PySCFConfig(stage=None, job_name="my-job"))
    assert ("MolwatchEmitter(JOB + '.molwatch.log'" in text
            or 'MolwatchEmitter(JOB + ".molwatch.log"' in text)
    emitter_line = [ln for ln in text.splitlines()
                    if "MolwatchEmitter(JOB" in ln][0]
    assert "stage" not in emitter_line


# ---- Post-relax frequencies + RRHO thermochemistry ---------------------- #


def test_frequencies_default_off(h2o):
    """compute_frequencies defaults to False so the existing script
    shape is unchanged; no Hessian / thermo / thermo.txt appears."""
    text = render_script(h2o, PySCFConfig())
    assert "mf.Hessian()" not in text
    assert "harmonic_analysis" not in text
    assert "thermo.txt" not in text
    assert "RRHO" not in text


def test_frequencies_block_emitted_when_enabled(h2o):
    """compute_frequencies=True: the Hessian + thermo block appears
    AFTER the post-opt SCF (so mf is converged at mol_eq) and BEFORE
    the optimized-geometry save (so the script keeps writing
    <job>_optimized.xyz even if Hessian fails).  Block is wrapped in
    try/except so a failure doesn't lose the converged energy."""
    text = render_script(h2o, PySCFConfig(compute_frequencies=True))
    assert "mf.Hessian().kernel()" in text
    assert "harmonic_analysis" in text
    assert "_mb_thermo.thermo(mf" in text
    assert ".thermo.txt" in text
    # try/except guard so a failed Hessian doesn't lose the run.
    assert "Frequency analysis FAILED" in text
    # Block sits between the post-opt SCF (Final energy print) and
    # the saved optimized-xyz writeout.  Match the actual _save_xyz
    # call (not the early "Files written" header which also mentions
    # _optimized.xyz).
    final_energy_pos    = text.index('Final energy:')
    hessian_pos         = text.index("mf.Hessian().kernel()")
    save_optimized_pos  = text.index('_save_xyz(mol_eq')
    assert final_energy_pos < hessian_pos < save_optimized_pos


def test_frequencies_writes_temperature_and_pressure(h2o):
    """T/P propagate verbatim to the thermo.thermo() call.  Pressure
    is converted from atm to Pascal (PySCF's thermo() wants Pa)."""
    text = render_script(h2o, PySCFConfig(compute_frequencies=True,
                                          temperature_K=400.0,
                                          pressure_atm=2.0))
    # The thermo.thermo() call should carry T=400.0 and
    # P=2.0*101325.0 = 202650.0 Pa.
    assert "thermo(mf, _mb_freq[\"freq_au\"], 400.0, 202650.0)" in text
    # Header comment in the .thermo.txt file echoes the user-facing units.
    assert "T = 400.0 K, P = 2.0 atm" in text


def test_frequencies_imag_warn_only_for_optimize(h2o):
    """The imaginary-mode WARN only fires when the script optimized
    the geometry (otherwise imag modes at a non-stationary point are
    expected and not a problem)."""
    text_opt = render_script(h2o, PySCFConfig(compute_frequencies=True,
                                              optimize=True))
    text_sp  = render_script(h2o, PySCFConfig(compute_frequencies=True,
                                              optimize=False))
    assert "imaginary mode(s) at the relaxed geometry" in text_opt
    assert "imaginary mode(s) at the relaxed geometry" not in text_sp


def test_frequencies_block_python_parses(h2o):
    """Sanity: the emitted script with compute_frequencies=True still
    compiles as valid Python (catches any quoting / f-string bugs in
    the embedded thermo block)."""
    text = render_script(h2o, PySCFConfig(compute_frequencies=True))
    compile(text, "<freq>", "exec")


def test_frequencies_list_in_header_files_section(h2o):
    """When compute_frequencies=True the script's header "Files
    written" section advertises <job>.thermo.txt so users know
    what to look for."""
    text = render_script(h2o, PySCFConfig(compute_frequencies=True))
    assert ".thermo.txt" in text
    assert "harmonic frequencies" in text  # in the header description
