"""The vibration deck RUNS the spectra science gate -- and it refuses.

Between P1 and P3 the deck's ``validate(struct, view)`` call silently
skipped the spectra science (grid / amplitude / parity / method /
open-shell): the engine dispatch inside ``validate()`` is keyed on
``type(cfg)`` and the deck's config view is an ADAPTER over
PySCFConfig, so no registered validator matched -- a green E2E hid a
gate that never ran.  Found 2026-08-21 while retiring the engine
registry (spectra-migration plan P3).  The science moved whole to
``validation/spectra.py`` and the deck composes it BY NAME; these pins
hold the composition down at its two observable ends.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import spec_for
from molbuilder.script_emit import render_deck
from molbuilder.structure import Structure


def _water() -> Structure:
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.119],
                            [0.0, 0.757, -0.477],
                            [0.0, -0.757, -0.477]]))


def _render(cfg: PySCFConfig) -> str:
    s = _water()
    return render_deck(spec_for(s, cfg, calculation="vibration"),
                       s, cfg, verbose=False)


def test_parity_error_refuses_the_deck():
    """spin=1 on water's 10 electrons is impossible; the gate's
    parity check must refuse at RENDER, not at PySCF runtime."""
    with pytest.raises(Exception) as exc:
        _render(PySCFConfig(spin=1, method="UKS"))
    msg = str(exc.value).lower()
    assert "spin" in msg and "electron" in msg


def test_amplitude_advisory_reaches_the_person():
    """A 0.5 A displacement is outside the accepted window; the deck
    still renders (a warn is advice, not a refusal) and the warning
    goes where a person sees it."""
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        text = _render(PySCFConfig(displacement_amplitude_ang=0.5))
    finally:
        sys.stderr = old
    assert "Hessian" in text, "the deck did not render"
    assert "0.02-0.20" in err.getvalue(), (
        "the amplitude advisory never surfaced -- the science gate "
        "is not composed into the deck's render")


def test_registry_path_serves_a_real_spectra_config():
    """The registered validator runs the SAME body -- one gate, two
    doors.  No production code constructs a SpectraConfig any more;
    this pins the registry ENTRY itself, which stays until the class's
    deferred retirement lands with transport's round (it shares the
    four-engine registry)."""
    from molbuilder.config.spectra import SpectraConfig
    from molbuilder.validation import validate
    issues = validate(_water(), SpectraConfig(es_mode_selection="top_n"))
    assert any("Raman-weak" in i.message for i in issues)


def test_the_gpu_advisory_path_renders_and_speaks():
    """use_gpu=True walks the gate's GPU advisory -- the exact path
    where the P3 move left a latent NameError (`cls.` in a plain
    function, guarded by this very flag; found and fixed 2026-08-21).
    The deck must render AND the advisory must surface on a host
    without the GPU stack."""
    import io
    import sys
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        text = _render(PySCFConfig(use_gpu=True))
    finally:
        sys.stderr = old
    assert "Hessian" in text
    assert "gpu4pyscf" in err.getvalue() or "compute capability" in err.getvalue()


def test_an_ecp_deck_compiles_and_carries_one_ecp_kwarg():
    """The gold-dimer ECP deck: exactly ONE `ecp        =` kwarg in the
    gto.M call and the whole deck compiles.  A duplicated
    resolution+emission pair shipped 2026-08-21 made every ECP deck a
    SyntaxError (`keyword argument repeated`) while the text-diff
    honesty gate stayed green -- this pin is that failure's shape."""
    s = Structure(elements=["Au", "Au"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]]))
    cfg = PySCFConfig(ecp="lanl2dz", ecp_atoms=["Au"])
    text = render_deck(spec_for(s, cfg, calculation="vibration"),
                       s, cfg, verbose=False)
    assert text.count("ecp        =") == 1, "the ECP kwarg must appear once"
    assert "'Au': 'lanl2dz'" in text
    compile(text, "<ecp-deck>", "exec")


# --------------------------------------------------------------------- #
#  The U2 correctness pins (2026-08-21)                                  #
# --------------------------------------------------------------------- #

def test_an_hf_raman_deck_never_mentions_the_dft_name():
    """E-M4.7's shape, tightened at the U6 close: on an HF deck the
    import block emits no ``dft``, so ANY reference to that name is a
    NameError waiting in dead text -- and the original bug fired it
    with ``force_cpu=True`` AFTER the full Hessian was paid for.  The
    method is a render-time fact, so an HF deck now carries no DFT arm
    at all; a DFT deck still evaluates ``dft`` only on its force_cpu
    pick."""
    import ast
    hf = _render(PySCFConfig(method="RHF", compute_raman=True))
    tree = ast.parse(hf)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_build_mf_at")
    loads = [n for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "dft"]
    assert not loads, (
        f"`dft` appears in an HF deck's _build_mf_at at line(s) "
        f"{[n.lineno for n in loads]} -- a NameError waiting in dead text")
    dft_deck = _render(PySCFConfig(method="RKS", compute_raman=True))
    assert "_dft_mod = dft if force_cpu else _dft" in dft_deck, (
        "the DFT deck lost its force_cpu module pick -- retarget")


def test_the_vibration_deck_runs_the_engines_own_deck_gate():
    """E-M4.6r: `check_rules` was None, so a non-compiling vibration deck
    (the shipped ECP double-kwarg's whole class) was never parse-checked
    at prep.  The spec now points at the engine's ONE gate -- and that
    gate's identity probe must accept the vibration deck's own JOB
    spelling (an aligned repr, not the optimization deck's quoted
    form)."""
    from molbuilder.pyscf import layout as _layout
    cfg = PySCFConfig(job_name="vibjob")
    s = _water()
    spec = spec_for(s, cfg, calculation="vibration")
    assert spec.check_rules is _layout.check_rules, (
        "the vibration deck does not run the engine's one deck gate")
    text = render_deck(spec, s, cfg, verbose=False)
    assert _layout.check_rules(text, cfg=cfg) == [], (
        "a healthy vibration deck fails its own gate")
    broken = [i for i in _layout.check_rules("def broken(:", cfg=cfg)
              if i.severity == "error"]
    assert any("parse" in i.message for i in broken)


def test_the_vibration_charge_runs_the_one_resolver(deprotonated_diester):
    """E-M3.1 / V-3c: `net_charge or 0` dropped the phosphate
    auto-detection the optimization deck runs -- a nucleic-acid
    vibration with charge unset was silently a DIFFERENT calculation
    than its optimization sibling.  The lift boundary now asks
    chemistry.resolve_net_charge: this diester's heuristic charge is
    -1, and the deck must carry it."""
    text = render_deck(
        spec_for(deprotonated_diester,
                 PySCFConfig(net_charge=None), calculation="vibration"),
        deprotonated_diester, PySCFConfig(net_charge=None), verbose=False)
    assert "charge     = -1," in text


def test_soscf_reaches_the_relax_site():
    """M1.3: the § 7a role table promises the `newton()` wrap at the
    vibration RELAXATION site; without it a `scf_soscf=true` run
    relaxes under DIIS while its equilibrium SCF runs Newton."""
    on = _render(PySCFConfig(scf_soscf=True))
    off = _render(PySCFConfig(scf_soscf=False))
    assert "_mf_relax = _mf_relax.newton()" in on
    assert "_mf_relax.newton()" not in off


def test_an_hf_vibration_deck_declines_dft_knobs():
    """E-M6.3: the spec's `line=` classified anything but literal "HF"
    as DFT -- RHF/UHF included.  A DFT-only knob asked through an RHF
    vibration spec must decline (None), exactly as the optimization
    deck's line does."""
    from molbuilder.script_emit import parameter
    cfg = PySCFConfig(method="RHF", functional="b3lyp")
    spec = spec_for(_water(), cfg, calculation="vibration")
    p = parameter("functional", "pyscf", config=cfg)
    assert spec.line(p) is None, (
        "an RHF vibration spec emits mf.xc -- RHF is being classified "
        "as DFT")


# --------------------------------------------------------------------- #
#  Frozen means frozen (the user's ruling, 2026-08-21) + the dedup       #
# --------------------------------------------------------------------- #

def _frozen_water():
    from molbuilder.structure import FROZEN_LABEL
    s = _water()
    s.regions[FROZEN_LABEL] = [0]
    return s


def test_frozen_atoms_stay_frozen_through_the_relaxation():
    """The ruling: frozen means frozen through EVERY phase.  The
    pre-Hessian relaxation takes the frozen set as geomeTRIC's $freeze
    constraints file -- the optimization deck's own mechanism -- so the
    fixed atoms never move before the (partial) Hessian is built over
    the free ones.  Before this the relaxation silently moved them
    (E-V2e: no constraints reached geomeTRIC)."""
    s = _frozen_water()
    cfg = PySCFConfig()
    text = render_deck(spec_for(s, cfg, calculation="vibration"),
                       s, cfg, verbose=False)
    assert "_FROZEN_CONSTRAINTS_PATH" in text
    assert "$freeze" in text
    assert "xyz 1" in text, "geomeTRIC indices are 1-based; atom 0 -> 1"
    assert "constraints=str(_FROZEN_CONSTRAINTS_PATH)" in text, (
        "the constraints file is written but never handed to geomeTRIC")
    # An unfrozen deck carries none of it.
    free = _render(PySCFConfig())
    assert "_FROZEN_CONSTRAINTS_PATH" not in free


def test_the_frozen_regime_is_said_out_loud():
    """"It just has to be explicit" -- the preflight names the frozen
    set and what it means (an info, nothing is wrong), and the deck's
    Methods paragraph states that the frequencies are those of the free
    atoms in the static field of the fixed ones."""
    from molbuilder.validation import validate
    s = _frozen_water()
    cfg = PySCFConfig()
    infos = [i for i in validate(s, cfg, calculation="vibration")
             if i.severity == "info" and "frozen" in i.message]
    assert len(infos) == 1, "the frozen regime is not announced"
    assert "relaxation holds them fixed" in infos[0].message
    assert "Hessian excludes them" in infos[0].message
    text = render_deck(spec_for(s, cfg, calculation="vibration"),
                       s, cfg, verbose=False)
    assert "static field of the fixed" in text, (
        "the Methods paragraph does not state the frozen regime")
    assert "partial Hessian" in text


def test_the_reserved_frozen_label_is_never_warned_unconsumed():
    """E-M7.1: since schema 7 the frozen set lives INSIDE regions, and
    this engine consumes it -- the relaxation constrains it and the
    Hessian mask reads it.  Pattern B must not warn "NOT consumed"
    about it, while a genuinely inert label still warns."""
    from molbuilder.validation import validate
    s = _frozen_water()
    issues = validate(s, PySCFConfig(), calculation="vibration")
    false_alarms = [i for i in issues
                    if "does NOT consume" in i.message
                    and "frozen_atoms" in i.message]
    assert false_alarms == [], "Pattern B warns about the frozen label"
    s2 = _water()
    s2.regions["L-electrode"] = [1]
    inert = [i for i in validate(s2, PySCFConfig(),
                                 calculation="vibration")
             if "does NOT consume" in i.message]
    assert len(inert) == 1 and "L-electrode" in inert[0].message


def test_one_fact_one_finding_on_a_vibration_deck():
    """The dedup ruling: on a vibration deck the kind owns the parity /
    grid verdicts and the engine copy DEFERS -- each fact earns exactly
    one finding.  Before, spin=1 on water produced the engine's parity
    finding AND the kind's, one of them reasoned from optimization
    fields the vibration deck ignores."""
    from molbuilder.validation import validate
    parity = [i for i in validate(_water(), PySCFConfig(spin=1,
                                                        method="UKS"),
                                  calculation="vibration")
              if "parity" in i.message]
    assert len(parity) == 1, (
        f"{len(parity)} parity findings for one fact: "
        f"{[i.where for i in parity]}")
    # ...and the optimization route still gets the engine's own.
    parity_opt = [i for i in validate(_water(), PySCFConfig(spin=1,
                                                            method="UKS"))
                  if "parity" in i.message]
    assert len(parity_opt) == 1
    grid = [i for i in validate(_water(),
                                PySCFConfig(functional="b3lyp",
                                            grid_level=1),
                                calculation="vibration")
            if i.where == "config.grid_level" and "hybrid" in i.message.lower()]
    assert len(grid) == 1, f"{len(grid)} grid findings for one fact"


def test_the_ir_only_phase_does_each_units_work_once():
    """The U5 efficiency pair, pinned structurally: the IR-only phase
    emits exactly ONE per-mode projection loop (it was nested inside a
    second per-mode loop -- N² idempotent work), and the displaced
    builds are not followed by a second kernel() (`_build_mf_at`
    already converges before returning -- the extra call re-ran every
    SCF for identical numbers)."""
    text = _render(PySCFConfig(compute_raman=False, compute_ir=True))
    ir = text.split("Phase 3-IR", 1)[1]
    assert ir.count("for _n in range(len(modes_payload)):") == 1
    assert "_mfp.kernel()" not in ir and "_mfm.kernel()" not in ir


def test_each_deck_carries_one_gpu_mechanism():
    """M1.4: the engine had two GPU-consumption mechanisms and the
    vibration deck's text carried BOTH -- the promotion helper emitted
    dead beside the class selection that actually does the work.  Each
    deck now carries exactly its own: the vibration deck selects
    gpu4pyscf classes (right for a deck rebuilding mol per geometry),
    the optimization deck promotes the assembled mf."""
    from molbuilder.pyscf.input import render_script
    vib = _render(PySCFConfig())
    assert "_mb_to_gpu_if_enabled" not in vib
    assert "_gpu_scf" in vib, "the class-selection mechanism left too"
    opt = render_script(_water(), PySCFConfig())
    assert "def _mb_to_gpu_if_enabled" in opt
    assert "mf = _mb_to_gpu_if_enabled(mf)" in opt


def test_the_dft_trio_has_one_spelling_per_deck():
    """M1.2: the functional / grid / dispersion trio was spelled twice
    -- layout's for the optimization deck, hand-constants inside the
    vibration deck's constructions.  The vibration deck now defines
    `_mb_configure_dft` (generated from the SAME DFT_SECTION + line)
    and every construction site calls it; the hand spellings are gone."""
    text = _render(PySCFConfig(functional="b3lyp", dispersion="d3bj",
                               grid_level=4))
    assert "def _mb_configure_dft(mf):" in text
    assert text.count("_mb_configure_dft(") >= 3   # def + 2 call sites
    assert 'mf.xc = "b3lyp"' in text               # layout's spelling
    assert "mf.xc = FUNCTIONAL" not in text        # the hand spelling
    assert "_mf2.grids.level = GRID_LEVEL" not in text
    # An HF deck carries NO dresser at all -- both call sites branch
    # on the method, so an emitted pass-through would be dead text
    # (tightened at the U6 close).
    hf = _render(PySCFConfig(method="RHF"))
    assert "_mb_configure_dft" not in hf
