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

from molbuilder.pyscf import (
    PySCFConfig,
    render_script,
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


def test_the_run_with_block_names_the_wrapper_and_the_real_filename(h2o):
    """The script is run BY THE WRAPPER, and its header has to say so.

    It said ``python <job>.py`` -- the design from before a run had a
    directory, when PySCF decks ran themselves (user, 2026-09-07).  Two things
    were wrong with it by then.  It named a file that does not exist on a
    staged run, because the deck is ``<job>_<stage>.py`` and only the stage
    token was missing; and following it skips everything the wrapper exists to
    do -- the environment activation, the tee, the SIGTERM trap and the monitor
    that writes the live log (`running-a-job.md` § 2).

    THE PLAIN INVOCATION STAYS (user, 2026-09-07): driving PySCF from the
    command line is a legitimate thing to do, and the header still writes it
    out.  What changed is which one is offered first, and that the direct one
    now says whose job the environment and the redirection are.
    """
    text = render_script(h2o, PySCFConfig(job_name="my-job"),
                         stage_token="02_medium")
    # The stage by its NAME, as `launch` takes it (`job-system.md` § 5.3):
    # the token printed here until K12 is a legal name of another stage.
    assert "molbuilder jobset launch run medium\n" in text, (
        "the header does not offer the managed path, by the stage's name")
    assert "direct|submit" not in text, (
        "the header offers `--mode direct|submit`, which bash runs as a pipe")
    assert "bash my-job_02_medium.run.sh" in text, (
        "the header does not name the wrapper that runs this deck")
    # The direct line is still here, and names the file that EXISTS.
    assert "python my-job_02_medium.py" in text, (
        "the plain invocation was dropped; it is a supported way to run this")
    assert "python my-job.py" not in text, (
        "the header names a file that does not exist on a staged run -- the "
        "stage token is missing from the deck's own name")
    i_managed = text.index("molbuilder jobset launch")
    i_direct = text.index("python my-job_02_medium.py")
    assert i_managed < i_direct, (
        "the direct invocation is offered before the managed one")


def test_default_render_compiles(h2o):
    text = render_script(h2o)
    compile(text, "<rendered>", "exec")
    for needle in (
        "import os",
        "from pyscf import gto, scf, dft",
        "from pyscf.geomopt import geometric_solver",
        "mol = gto.M(",
        "mf = dft.RKS(mol)",
        'mf.xc = "B3LYP"',
        "mf = mf.density_fit()",
        'mf.disp = "d3bj"',
        # The one relaxation function (relax_policy.py), spliced, and
        # the call that applies this rung's policy to geomeTRIC's answer.
        "def relax(mf, policy, retries, **geometric_kw):",
        "mol_eq, _GEOM_CONVERGED = relax(",
        "_save_structure(",
    ):
        assert needle in text, f"missing {needle!r}"


def test_atom_block_format(h2o):
    """One row per atom: the element, then three coordinates to 8 decimals.

    The values are the PLACED ones -- the design coordinates plus the engine
    offset (`model/structure-periodicity.md` § 6.0); coordinates + offset is
    the invariant.  This asserted the design coordinates themselves until
    2026-09-25, when every engine began receiving the atoms centred."""
    from molbuilder import cell as cellmod
    text = render_script(h2o, PySCFConfig(verbose_comments=False))
    placed = h2o.positions + cellmod.engine_offset(h2o)
    for el, (x, y, z) in zip(h2o.elements, placed):
        assert re.search(rf"^\s*{el}\s+{x:.8f}\s+{y:.8f}\s+{z:.8f}\s*$",
                         text, re.M), (el, x, y, z)


# --------------------------------------------------------------------- #
#  Section toggles                                                      #
# --------------------------------------------------------------------- #


def test_no_optimize_drops_geom_block(h2o):
    text = render_script(h2o, PySCFConfig(optimize=False, verbose_comments=False))
    assert "= relax(" not in text
    assert "e = mf.kernel()" in text
    # The _save_structure call that WRITES <JOB>_optimized.xyz must not
    # appear -- there's no optimized geometry to save.  The
    # geometry-warm-restart hook at gto.M() time (task #539) still
    # references the file (auto-resume from a prior optimize=True
    # run with the same JOB), so don't assert ``_optimized.xyz``
    # is absent globally; assert only the WRITE site is gone.
    assert "_save_structure(mol_eq" not in text
    assert 'JOB + "_optimized.xyz"), "Final optimized geometry"' not in text


def test_one_optimize_call_site_carrying_this_rung_s_targets(h2o):
    """A deck is one rung, so there is one relaxation call in it -- the one
    spliced ``relax(mf, ...)`` (`relax_policy.py`, `engines/pyscf.md` § 3).

    `stages.md` § 1.1a retired the in-script ladder: the six convergence targets
    belong to THIS deck and arrive as named constants the single call reads.
    Pinning the kwarg list here means a regression that loses one of them
    surfaces immediately, which is what the old staged-loop version of this test
    was for -- the guarantee outlived the loop.
    """
    text = render_script(h2o, PySCFConfig())
    assert text.count("mol_eq, _GEOM_CONVERGED = relax(") == 1, (
        f"expected exactly one relax() call; got "
        f"{text.count('mol_eq, _GEOM_CONVERGED = relax(')}")
    assert "optimize(mf" not in text, (
        "optimize() drops geomeTRIC's convergence flag; the deck relaxes "
        "through relax(), which asks for it")
    for kwarg in ("convergence_energy", "convergence_grms", "convergence_gmax",
                  "convergence_drms", "convergence_dmax", "maxsteps"):
        assert f"{kwarg} " in text, f"missing geomeTRIC kwarg {kwarg!r}"
    assert "for STAGE in STAGES:" not in text, (
        "the in-script ladder is retired: a PySCF ladder is N decks and N "
        "jobs, so that a person can look between the rungs")


def test_molwatch_log_instantiated_before_the_optimization(h2o):
    """Critical UX guarantee: ``.molwatch.log`` exists from the moment
    the script starts running, BEFORE any stage's optimize() can take
    hours on a real molecule.  Pin the source ordering:
    ``_molwatch = MolwatchEmitter(...)`` must appear before the optimization,
    ``mf.callback`` wiring before optimize() runs, and the opt-step callback
    INSIDE the optimize() kwargs.
    """
    text = render_script(h2o, PySCFConfig())
    inst_at      = text.find('_molwatch = MolwatchEmitter(_mb_outfile(JOB')
    mf_callback  = text.find("mf.callback = _molwatch.scf_cycle_hook")
    helper_def   = text.find("def relax(mf, policy, retries, **geometric_kw):")
    opt_at       = text.find("mol_eq, _GEOM_CONVERGED = relax(")
    step_cb      = text.find("callback              = _molwatch.opt_step_hook")
    for name, off in [
        ("_molwatch instantiation", inst_at),
        ("mf.callback wiring",      mf_callback),
        ("the relax() definition",  helper_def),
        ("the relaxation call",     opt_at),
        ("opt_step callback",       step_cb),
    ]:
        assert off >= 0, f"missing in script: {name}"
    # inst < mf_callback (sets the SCF-cycle hook on the prod mf)
    #     < helper_def (the spliced relax(), defined before its call)
    #         < opt_at (the one call)
    #             < step_cb (opt_step_hook among the call's kwargs)
    assert inst_at < mf_callback < helper_def < opt_at < step_cb, (
        "molwatch wiring out of order; expected inst < mf_callback < "
        "relax def < relax call < step_cb.  "
        f"Got: inst={inst_at}, mf_cb={mf_callback}, "
        f"helper={helper_def}, call={opt_at}, step={step_cb}"
    )


def test_stability_analysis_skipped_for_closed_shell(h2o):
    """Closed-shell scripts (RKS / RHF) shouldn't carry a
    `mf.stability()` call -- closed-shell stability is rarely the
    user's concern and the call adds noise to a tutorial script
    that's already dense.  Open-shell coverage is in
    `test_pyscf_stability.py`."""
    text = render_script(h2o, PySCFConfig(spin_treatment="restricted"))
    code_lines = [ln for ln in text.splitlines()
                  if not ln.lstrip().startswith("#")]
    assert not any("_mb_stability" in ln for ln in code_lines), (
        "RKS script should not emit the stability check"
    )


def test_dispersion_can_be_disabled(h2o):
    # "none" is the item's value for no correction -- its one spelling
    # (`config/pyscf.py`'s note on the field).  PySCF's check_disp raises
    # NotImplementedError on the literal string, so the deck writes no
    # assignment at all.  The ASSIGNMENT is the guarantee: the
    # effective-parameters record reads `mf.disp` back to show the engine has
    # none, and that read is the record working, not the setting leaking.
    text = render_script(h2o, PySCFConfig(dispersion="none"))
    assert "mf.disp = " not in text


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


def test_threads_emit_env_pin(h2o):
    """The shared molbuilder.runtime_info emitter pins BLAS to 1
    per worker and sets OMP_NUM_THREADS via ``setdefault`` to the
    user's requested count (refactored 2026-05-22 from the old
    inline format that emitted the threads value literally in the
    env-export line)."""
    text = render_script(h2o, PySCFConfig(threads=8))
    # cfg.threads=8 -> _MB_REQUESTED_THREADS = 8 as a literal.
    assert "_MB_REQUESTED_THREADS = 8" in text
    # OMP pinned via setdefault (user-set value).
    assert "os.environ.setdefault('OMP_NUM_THREADS',      str(_MB_REQUESTED_THREADS))" in text
    # BLAS always pinned to 1 (canonical anti-oversubscription).
    assert "os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')" in text
    assert "os.environ.setdefault('MKL_NUM_THREADS',      '1')" in text
    # Post-import num_threads(N) call to size the in-process pool.
    assert "_pyscf_lib.num_threads(_MB_REQUESTED_THREADS)" in text


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
])
def test_invalid_inputs_raise(h2o, kwargs, name):
    with pytest.raises(ValueError):
        render_script(h2o, PySCFConfig(**kwargs))


# --------------------------------------------------------------------- #
#  convert() -- file in, .py out                                        #
# --------------------------------------------------------------------- #


def test_loaded_structure_to_pyscf_script(h2o, tmp_path):
    """Mirror the FDF flow: build -> load -> render PySCF."""
    xyz_p = tmp_path / "h2o.xyz"
    from molbuilder.workingcopy_structure import StructureCodec
    StructureCodec().write(h2o, xyz_p)
    s2 = StructureCodec().load(str(xyz_p))
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


def test_chkfile_continuation_shim_emitted(h2o):
    """A rung that CONTINUES starts its SCF from the density the rung before
    it converged: the script detects a non-empty ``.chk`` and flips
    ``mf.init_guess`` to ``"chkfile"`` instead of MINAO / atom / huckel.

    Rendered with ``restart="continue"`` since 2026-08-18.  It used to be
    unconditional, which meant ``clean`` could only be had by turning off the
    WRITE -- and that threw away the checkpoint the next rung wanted
    (`run-identity.md` § 4 rule 2).  The clean side is asserted in
    ``test_pyscf_rung_artifacts.py``."""
    text = render_script(h2o, PySCFConfig(restart="continue"))
    # The chkfile assignment is still there.
    assert 'mf.chkfile = _mb_outfile(JOB + ".chk")' in text
    # The auto-detect shim is appended right after.
    assert "import os as _os" in text
    assert '_chk_path = _mb_outfile(JOB + ".chk")' in text
    assert ("_os.path.exists(_chk_path) and "
            "_os.path.getsize(_chk_path) > 0") in text
    assert 'mf.init_guess = "chkfile"' in text
    # Order: chkfile assignment must come BEFORE the shim (so
    # mf.chkfile has the resolved path at the time we test for it).
    chk_assign_ix = text.find('mf.chkfile = _mb_outfile(JOB + ".chk")')
    shim_ix      = text.find("_chk_path = _mb_outfile(JOB")
    assert chk_assign_ix < shim_ix


def test_chkfile_disabled_skips_continuation_shim(h2o):
    """When cfg.chkfile=False the shim is NOT emitted (nothing to
    load from)."""
    text = render_script(h2o, PySCFConfig(chkfile=False))
    assert 'mf.chkfile = _mb_outfile(JOB + ".chk")' not in text
    assert "_chk_path = _mb_outfile(JOB" not in text


# --------------------------------------------------------------------- #
#  Task #539: geometry warm-restart hook at gto.M() time                #
# --------------------------------------------------------------------- #


def test_geometry_warm_restart_block_emitted(h2o):
    """Task #539 / decision-log 2026-06-22+: every PySCF script
    must auto-resume from ``<JOB>_optimized.xyz`` when present
    (analog to SIESTA's automatic ``.XV`` read).  This is the
    geometry side of the warm-restart contract documented in
    docs/execution/job-contracts.md § "Generator-side warm-
    restart contract" item 2.

    Without this hook, a rung that says ``continue`` would re-start from the
    script's literal coordinates -- discarding the optimization the rung
    before it paid for.

    Rendered with ``restart="continue"`` since 2026-08-18: the read is gated
    on that field and on nothing else (`run-identity.md` § 4 rule 2).
    """
    text = render_script(h2o, PySCFConfig(restart="continue"))
    # The literal coordinates land in ``_atom_block``, not directly
    # as ``atom='...'`` -- so the warm-restart block can override
    # the variable before gto.M() consumes it.
    assert "_atom_block = '''" in text
    assert "atom       = _atom_block," in text
    # The auto-detect shim: file existence + non-empty guard, XYZ
    # parse, _atom_block override, continuation print.
    assert '_opt_path = _mb_outfile(JOB + "_optimized.xyz")' in text
    assert ("_os.path.exists(_opt_path) and "
            "_os.path.getsize(_opt_path) > 0") in text
    assert 'continuation: loaded geometry from' in text
    # Fall-through guard: a parse failure prints a warning and the
    # literal _atom_block is used.  Without this, a malformed XYZ
    # would silently feed garbage to gto.M().
    assert "except (OSError, ValueError, IndexError)" in text
    assert "could not parse" in text


def test_geometry_warm_restart_block_precedes_gto_M(h2o):
    """The warm-restart override MUST run before gto.M() reads
    ``_atom_block`` -- otherwise the literal is consumed and the
    override is dead code.  Pins the lexical order in the rendered
    script."""
    text = render_script(h2o, PySCFConfig(restart="continue"))
    opt_block_ix = text.index('_opt_path = _mb_outfile(JOB + "_optimized.xyz")')
    gto_call_ix  = text.index("mol = gto.M(")
    assert opt_block_ix < gto_call_ix, (
        "warm-restart override must precede gto.M() so the literal "
        "_atom_block has been overridden by the time PySCF builds mol")


def test_geometry_warm_restart_compiles(h2o):
    """The generated script must compile to bytecode (no syntax
    errors) -- the warm-restart block uses try/except/with/for which
    are easy to mis-emit at the join.  Pins script-render correctness
    end-to-end so a regression that breaks the template surfaces
    immediately (rather than at PySCF launch time)."""
    text = render_script(h2o, PySCFConfig())
    compile(text, "<rendered-script>", "exec")


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 2 tests here warm-started from a PySCF `_optimized.xyz` written by
# hand (`process/testing.md` § 6).


def test_geometry_warm_restart_falls_through_when_xyz_absent(h2o, tmp_path):
    """Symmetric end-to-end: when no ``<JOB>_optimized.xyz`` exists,
    the warm-restart block falls through cleanly and ``_atom_block``
    keeps the literal value the generator wrote into the script.
    This is the cold-start path; a regression that always-triggers
    the override (e.g., bad ``if`` predicate) would silently break
    cold runs."""
    text = render_script(h2o, PySCFConfig())
    block_start = text.index("_atom_block = '''")
    block_end   = text.index("mol = gto.M(")
    slice_text  = text[block_start:block_end]

    ns = {
        "_os": __import__("os"),
        "JOB": "no_such_job",
        "_mb_outfile": lambda name: str(tmp_path / name),
    }
    exec(slice_text, ns)
    # Literal H2O survives -- no override fired.
    assert " O " in ns["_atom_block"]
    assert " H " in ns["_atom_block"]
    assert "He" not in ns["_atom_block"]


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


# RETIRED 2026-08-18 -- `test_staged_opt_warm_starts_inside_stage_loop`.
#
# It gated the warm-start pair (`mf.reset(mol_eq); mf.kernel(dm0=dm_prev)`)
# INSIDE the in-script `for STAGE in STAGES:` body, and said so: its whole value
# over `test_post_opt_warm_starts_from_converged_dm` was that it sliced the
# script by the loop rather than by the first `mf.reset()`.  `stages.md` § 1.1a
# retired that loop -- a PySCF ladder is N decks and N jobs -- so the thing this
# test scoped to no longer exists.
#
# The SURVIVING guarantee is that a deck re-converges SCF at its relaxed
# geometry from the converged density rather than from MINAO, and
# `test_post_opt_warm_starts_from_converged_dm` asserts exactly that, on the
# deck this design actually produces.  Between rungs the warm start is now
# `.chk` + `<JOB>_optimized.xyz` on disk, which is the warm-file vocabulary's
# job and is tested there.


# --------------------------------------------------------------------- #
#  Bridge: choices accept any case via the bridge (R2)                  #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  Dispersion choices reject typos at parse time (R4)                   #
# --------------------------------------------------------------------- #


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
    # No ecp KWARG line at all -- matched as a LINE (B-4, 2026-08-13):
    # the fixed-width literals this replaces guessed at the emitter's
    # column alignment (1 and 5 spaces where the emission is 8), so all
    # four asserted the absence of strings the script never contains in
    # any state, and the verbose COMMENT that mentions `ecp = ...` kept
    # the sibling positive test green with the kwarg block deleted.
    import re as _re
    assert not _re.search(r'^\s*ecp\s+=', text, _re.M), (
        "no ecp kwarg may be emitted for the none sentinel")
    # Four `assert 'ecp     = "none"' not in text` lines stood here and were
    # strictly subsumed by the line match above -- it catches any spacing,
    # they caught one guess at it.  They were also the SAME fixed-width
    # literals the comment above says were the original defect, left in place
    # when the regex replaced them.  Removed 2026-09-09.


def test_python_api_ecp_reaches_the_gto_kwarg(h2o):
    """A declared ECP propagates as the gto.M KWARG, matched as a line.

    The fixed-width literal this replaces (5-space / 1-space) never
    matched the 8-space emission and passed via the verbose COMMENT
    (`# \\`ecp = "..."\\``): deleting the kwarg block stayed green (B-4,
    2026-08-13).  Kept, with the selector the field now requires.
    """
    import re as _re
    text = render_script(h2o, PySCFConfig(
        ecp="lanl2dz", ecp_atoms=["O"], basis="cc-pVDZ"))
    assert _re.search(r"^\s*ecp\s+= \{'O': 'lanl2dz'\},$", text, _re.M), (
        "the ecp kwarg line is missing from the emitted gto.M(...) call")


def test_python_api_ecp_name_without_atoms_emits_nothing(h2o):
    """The complement, and the point of the rewrite: a name alone
    selects no atoms, so no ECP is emitted.  Empty means empty."""
    import re as _re
    text = render_script(h2o, PySCFConfig(ecp="lanl2dz", basis="cc-pVDZ"))
    assert not _re.search(r"^\s*ecp\s+=", text, _re.M), (
        "an ECP name with no atoms selected must emit no kwarg")


# ---- Staged-relaxation suffix (job-layout v1) ---------------------------- #


def test_pyscf_molwatch_emitter_uses_stage_suffix(h2o):
    """Given a rung's token the inlined ``MolwatchEmitter(...)`` writes to
    ``<JOB>_<token>.molwatch.log``, so two rungs sharing a directory cannot
    overwrite each other's log.

    The token is a render ARGUMENT, not a config field (roadmap C7 closed
    2026-08-18): `prep` holds the StageRef, so `prep` says the word."""
    text = render_script(h2o, PySCFConfig(job_name="my-job"),
                         stage_token="02_medium")
    # Quote style is repr()'s choice (single or double); the contract
    # is the JOB + "<suffix>" expression with the right suffix.
    # 2026-05-27: path arg now wraps in _mb_outfile() so the log
    # resolves against the script directory regardless of cwd.
    assert ("MolwatchEmitter(_mb_outfile(JOB + '_02_medium.molwatch.log')" in text
            or 'MolwatchEmitter(_mb_outfile(JOB + "_02_medium.molwatch.log")' in text)


def test_pyscf_molwatch_emitter_unsuffixed_when_stage_is_none(h2o):
    text = render_script(h2o, PySCFConfig(job_name="my-job"), stage_token=None)
    # 2026-05-27: path arg now wraps in _mb_outfile().
    assert ("MolwatchEmitter(_mb_outfile(JOB + '.molwatch.log')" in text
            or 'MolwatchEmitter(_mb_outfile(JOB + ".molwatch.log")' in text)
    emitter_line = [ln for ln in text.splitlines()
                    if "MolwatchEmitter(_mb_outfile(JOB" in ln][0]
    assert "stage" not in emitter_line


# ---- Post-relax frequencies + RRHO thermochemistry ---------------------- #


# The six frequencies-block tests retired with the in-deck
# compute_frequencies path (spectra-migration plan D2/P3,
# 2026-08-21): the vibration calculation kind is the one
# Hessian door, and its thermochemistry is pinned by
# tests/test_vibration_e2e.py against a live run.
