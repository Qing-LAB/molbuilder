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
from molbuilder.runfiles import RunNames


def _names(cfg):
    """The names prep gives this stage's deck (`runfiles.RunNames`),
    under the config's own label."""
    label = getattr(cfg, "system_label", None) or cfg.job_name
    return RunNames.of(label, '01_freq', "hierarchical")


def _water() -> Structure:
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.119],
                            [0.0, 0.757, -0.477],
                            [0.0, -0.757, -0.477]]))


def _methyl() -> Structure:
    """Nine electrons: odd, so a closed shell stated on it cannot hold."""
    return Structure(
        elements=["C", "H", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [1.08, 0.0, 0.0],
                            [-0.54, 0.935, 0.0], [-0.54, -0.935, 0.0]]))


def _render(cfg: PySCFConfig, struct: Structure = None) -> str:
    s = struct if struct is not None else _water()
    return render_deck(spec_for(s, cfg, calculation="vibration",
                                names=_names(cfg)),
                       s, cfg, verbose=False)


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


def test_the_kind_door_runs_the_same_gate_body():
    """One gate, and the door that reaches it is the CALCULATION KIND.

    A vibration's science is the kind's:
    `_validate_vibration_kind` runs `spectra_render_checks` against the
    deck's config view, which is the object the emitters get.  Same body,
    reached the way production reaches it: `validate(..., calculation=
    "vibration")` with the config a person describes, the door `prep` and
    the Task-setup preflight both go through.
    """
    from molbuilder.validation import validate
    cfg = PySCFConfig(es_mode_selection="explicit", es_explicit_indices="")
    issues = validate(_water(), cfg, calculation="vibration")
    assert any(i.where == "config.es_explicit_indices" for i in issues), [
        (i.where, i.message) for i in issues]


def test_an_ecp_deck_compiles_and_carries_one_ecp_kwarg():
    """The gold-dimer ECP deck: exactly ONE `ecp        =` kwarg in the
    gto.M call and the whole deck compiles.  A duplicated
    resolution+emission pair shipped 2026-08-21 made every ECP deck a
    SyntaxError (`keyword argument repeated`) while the text-diff
    honesty gate stayed green -- this pin is that failure's shape."""
    s = Structure(elements=["Au", "Au"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]]))
    cfg = PySCFConfig(ecp="lanl2dz", ecp_atoms=["Au"])
    text = render_deck(spec_for(s, cfg, calculation="vibration",
                                names=_names(cfg)),
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
    hf = _render(PySCFConfig(method="HF", compute_raman=True))
    tree = ast.parse(hf)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_build_mf_at")
    loads = [n for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "dft"]
    assert not loads, (
        f"`dft` appears in an HF deck's _build_mf_at at line(s) "
        f"{[n.lineno for n in loads]} -- a NameError waiting in dead text")
    dft_deck = _render(PySCFConfig(method="DFT", compute_raman=True))
    assert "_dft_mod = dft if force_cpu else _dft" in dft_deck, (
        "the DFT deck lost its force_cpu module pick -- retarget")


def test_soscf_reaches_the_relax_site():
    """M1.3: the § 7a role table promises the `newton()` wrap at the
    vibration RELAXATION site; without it a `scf_soscf=true` run
    relaxes under DIIS while its equilibrium SCF runs Newton."""
    on = _render(PySCFConfig(scf_soscf=True))
    off = _render(PySCFConfig(scf_soscf=False))
    assert "_mf_relax = _mf_relax.newton()" in on
    assert "_mf_relax.newton()" not in off


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
    text = render_deck(spec_for(s, cfg, calculation="vibration",
                                names=_names(cfg)),
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
    assert "holds them fixed" in infos[0].message
    assert "taken over the free atoms only" in infos[0].message
    text = render_deck(spec_for(s, cfg, calculation="vibration",
                                names=_names(cfg)),
                       s, cfg, verbose=False)
    assert "static field of the fixed" in text, (
        "the Methods paragraph does not state the frozen regime")
    assert "partial Hessian" in text


def test_one_fact_one_finding_on_a_vibration_deck():
    """The dedup ruling: each fact earns exactly one finding.  The parity
    of the electronic state is the state's one family, asked once by
    `validate` for every kind -- and the grid verdict is the kind's, the
    engine copy DEFERRING on a vibration deck."""
    from molbuilder.validation import validate
    # A parity mismatch each kind can reach: a PySCF vibration offers
    # `restricted` alone (`engines/template.md` § 6.3a), so its mismatch is
    # a closed shell on an odd count; an optimization's is 2S = 1 on water.
    for kind, struct, cfg in (
            ("vibration", _methyl(),
             PySCFConfig(spin_treatment="restricted", unpaired_electrons=0)),
            ("optimization", _water(),
             PySCFConfig(spin_treatment="unrestricted",
                         unpaired_electrons=1))):
        parity = [i for i in validate(struct, cfg, calculation=kind)
                  if "the electron count and the spin disagree"
                  in i.message.lower()]
        assert len(parity) == 1, (
            f"{kind}: {len(parity)} parity findings for one fact: "
            f"{[i.where for i in parity]}")
    grid = [i for i in validate(_water(),
                                PySCFConfig(functional="b3lyp",
                                            grid_level=1),
                                calculation="vibration")
            if i.where == "config.grid_level" and "hybrid" in i.message.lower()]
    assert len(grid) == 1, f"{len(grid)} grid findings for one fact"


def test_the_level_of_theory_has_one_spelling_per_deck():
    """M1.2: the functional / grid / dispersion trio was spelled twice
    -- layout's for the optimization deck, hand-constants inside the
    vibration deck's constructions.  The vibration deck now defines
    `_mb_configure_theory` (generated from the SAME THEORY_SECTION + line)
    and every construction site calls it; the hand spellings are gone.

    On a Hartree-Fock deck too: HF has no functional and no grid, and
    takes the dispersion correction like any method (engines/pyscf.md
    § 7a), so its dresser holds the dispersion line alone and both of its
    construction sites call it.

    MUTATION THIS MUST FAIL AGAINST: the dresser emitted for DFT alone."""
    text = _render(PySCFConfig(functional="b3lyp", dispersion="d3bj",
                               grid_level=4))
    assert "def _mb_configure_theory(mf):" in text
    assert text.count("_mb_configure_theory(") >= 3   # def + 2 call sites
    assert 'mf.xc = "b3lyp"' in text                  # layout's spelling
    assert "mf.xc = FUNCTIONAL" not in text           # the hand spelling
    assert "_mf2.grids.level = GRID_LEVEL" not in text
    hf = _render(PySCFConfig(method="HF", dispersion="d3bj"))
    fn = hf[hf.index("def _mb_configure_theory(mf):"):]
    fn = fn[:fn.index("return mf")]
    assert 'mf.disp = "d3bj"' in fn, fn
    assert "mf.xc" not in fn and "mf.grids" not in fn, fn
    assert hf.count("_mb_configure_theory(") >= 3, "a call site skips it"


def test_the_kind_gate_refuses_an_engine_it_has_no_science_for():
    """A check is never skipped in silence (science/validation.md F4).
    The vibration kind's science is written for PySCFConfig; a
    SiestaConfig described as a vibration must not come back with an
    EMPTY verdict, which every surface reads as 'checked, nothing
    found'.  It is refused by name -- and the same config through the
    same door is an ordinary optimization gate."""
    import dataclasses

    from molbuilder.validation import validate

    @dataclasses.dataclass
    class OtherEngineConfig:
        basis: str = "x"

    with pytest.raises(TypeError, match="vibration kind has no science"):
        validate(_water(), OtherEngineConfig(), calculation="vibration")
    assert isinstance(validate(_water(), OtherEngineConfig()), list)
