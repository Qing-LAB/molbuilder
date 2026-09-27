"""Tests for the engine-parameter adapter Protocol + registry (L3
of the scientific-validation machinery; see
``docs/science/validation.md`` § 4).

# Retired 2026-09-10: `@dataclass(frozen=True)` is the enforcement, and
# CPython refuses a non-frozen subclass of a frozen one on its own.
# A test that mutates an instance to watch Python raise tests Python.

Pins the contract:

* Each registered adapter returns a frozen dataclass whose field
  names match the engine's web-form / Config field names.
* Adapters are PURE translators — they consume a
  ``ChemistryAnalysis`` and translate; they MUST NOT re-do
  chemistry detection or parity work.  Pinned by handing each one
  conclusions that contradict the composition they came with.
* The registry has a working on-ramp: adding an adapter via
  ``@register_adapter`` makes it appear in ``registered_adapters()``
  without endpoint code changes.
* Cross-engine consistency: for any structure, all adapters'
  outputs carry the same ``treatment``-equivalent decision
  (spelled differently per engine, but the conclusion is identical).

The "new-engine on-ramp" test uses the test-only
``_clear_adapters_for_test`` helper to verify registration works
on an empty registry without polluting the production adapters
for sibling tests.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass

import numpy as np
import pytest

from molbuilder.chemistry import (
    ChemistryAnalysis,
    analyze_structure,
    register_adapter,
    registered_adapters,
    _clear_adapters_for_test,
)
from molbuilder.structure import Structure
from molbuilder.siesta.auto_defaults import SiestaAdapter, SiestaSuggestedParams
from molbuilder.pyscf.auto_defaults  import PyscfAdapter,  PyscfSuggestedParams


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


def _mk(elements):
    n = len(elements)
    return Structure(
        elements      = list(elements),
        positions     = np.zeros((n, 3)),
        atom_names    = [f"A{i}" for i in range(n)],
        residue_ids   = [1] * n,
        residue_names = ["MET"] * n,
        chain_ids     = ["A"] * n,
    )


@pytest.fixture(autouse=True)
def _restore_adapter_registry():
    """Each test that mutates the registry (via _clear_adapters_for_test
    or register_adapter) gets isolated.  The class-attribute leak
    pattern that caused test_capabilities_returns_only_projects_root
    to flap (see commit ba96288) taught us this fixture is mandatory
    when tests touch class-level / module-level singletons.
    """
    saved = registered_adapters()
    yield
    _clear_adapters_for_test()
    for name, cls in saved.items():
        register_adapter(name)(cls)


# --------------------------------------------------------------------- #
#  Registry — built-in adapters                                         #
# --------------------------------------------------------------------- #


def test_siesta_and_pyscf_adapters_registered_at_import():
    """Importing the adapter modules registers them.  The web
    blueprint's __init__ does this at app startup; the test
    suite triggers it via the import at top of file."""
    reg = registered_adapters()
    assert "siesta" in reg
    assert "pyscf"  in reg
    assert reg["siesta"] is SiestaAdapter
    assert reg["pyscf"]  is PyscfAdapter


def test_registered_adapters_returns_defensive_copy():
    """A consumer that mutates the returned dict must not poison
    the registry for the rest of the process."""
    reg = registered_adapters()
    reg.clear()
    # The actual registry is untouched.
    assert registered_adapters() == {"siesta": SiestaAdapter,
                                      "pyscf":  PyscfAdapter}


# --------------------------------------------------------------------- #
#  SIESTA adapter                                                       #
# --------------------------------------------------------------------- #


def test_siesta_adapter_open_shell_metal_path():
    """Fe → spin_treatment="polarized" + spin_total=2.0 + rationale carries
    'Fe' in the engine-agnostic explanation."""
    a = analyze_structure(_mk(["Fe"]))
    p = SiestaAdapter.to_params(a)
    assert p.net_charge     == 0
    assert p.spin_treatment == "polarized"
    assert p.spin_total     == 2.0
    assert "Fe" in p.rationale


def test_siesta_adapter_closed_shell_path():
    """Pure organic, even electrons → spin_treatment="non-polarized",
    spin_total=0."""
    a = analyze_structure(_mk(["C", "H", "H", "H", "H"]))
    p = SiestaAdapter.to_params(a)
    assert p.spin_treatment == "non-polarized"
    assert p.spin_total     == 0.0


# --------------------------------------------------------------------- #
#  PySCF adapter                                                        #
# --------------------------------------------------------------------- #


def test_pyscf_adapter_open_shell_metal_path():
    """Fe → spin=2 + method='UKS' + rationale."""
    a = analyze_structure(_mk(["Fe"]))
    p = PyscfAdapter.to_params(a)
    assert p.net_charge == 0
    assert p.spin   == 2
    assert p.method == "UKS"
    assert "Fe" in p.rationale


def test_pyscf_adapter_closed_shell_path():
    """Pure organic, even electrons → spin=0, method='RKS'."""
    a = analyze_structure(_mk(["C", "H", "H", "H", "H"]))
    p = PyscfAdapter.to_params(a)
    assert p.spin   == 0
    assert p.method == "RKS"


# --------------------------------------------------------------------- #
#  Cross-engine consistency invariant                                   #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("elements", [
    ["C", "H", "H", "H", "H"],       # CH4 — closed shell
    ["Fe"],                            # Fe — Fe(II) intermediate
    ["Cu"],                            # Cu(II) doublet
    ["Mn"],                            # Mn(II) high-spin
])
def test_all_adapters_agree_on_treatment(elements):
    """For any structure, every registered adapter's output carries
    the same treatment-equivalent decision.  Spelled differently
    per engine (SIESTA spin_treatment="polarized", PySCF method='UKS'),
    but the conclusion is identical.

    This is the structural realisation of the cross-engine
    consistency rule (science/chemistry-correctness.md § 2) extended to the auto-detect
    surface.  An adapter that returned treatment='open' translated
    to RKS (a closed-shell method) would fail this test.
    """
    struct = _mk(elements)
    analysis = analyze_structure(struct)
    reg = registered_adapters()
    si = reg["siesta"].to_params(analysis)
    py = reg["pyscf"].to_params(analysis)

    # Open-shell agreement: SIESTA spin_treatment iff PySCF method UKS
    # A MODE, not a flag, since 2026-08-15: anything but "non-polarized"
    # means the calculation carries separate spin channels.
    si_open = si.spin_treatment != "non-polarized"
    py_open = py.method == "UKS"
    assert si_open == py_open, (
        f"SIESTA and PySCF disagree on open/closed for elements "
        f"{elements}: SIESTA spin_treatment={si_open}, "
        f"PySCF method={py.method!r}"
    )

    # Same spin (modulo type — SIESTA float μB, PySCF int 2S)
    assert float(py.spin) == si.spin_total, (
        f"SIESTA spin_total={si.spin_total} but PySCF spin={py.spin} "
        f"— translations must agree on the underlying 2S value"
    )

    # Same charge (universal)
    # ONE NAME since 2026-08-19: this line read `si.net_charge == py.charge`
    # and existed to bridge two spellings of one question.  The catalogue
    # merged them (`template.md` § 6.3), so the bridge is now an identity.
    assert si.net_charge == py.net_charge


# --------------------------------------------------------------------- #
#  asdict round-trip (the HTTP boundary)                                #
# --------------------------------------------------------------------- #


def test_asdict_round_trip_for_each_adapter():
    """Endpoint serialises adapter outputs via dataclasses.asdict —
    pin that round-trip works and preserves field names."""
    a = analyze_structure(_mk(["Fe"]))
    si_d = asdict(SiestaAdapter.to_params(a))
    py_d = asdict(PyscfAdapter.to_params(a))
    assert si_d == {
        "net_charge":     0,
        "spin_treatment": "polarized",
        "spin_total":     2.0,
        "rationale":      si_d["rationale"],   # value-agnostic
    }
    assert py_d["net_charge"] == 0
    assert py_d["spin"]   == 2
    assert py_d["method"] == "UKS"


# --------------------------------------------------------------------- #
#  New-engine on-ramp                                                   #
# --------------------------------------------------------------------- #


def test_registration_works_with_synthetic_adapter():
    """Pin the new-engine on-ramp: drop a fake adapter, register
    it, verify it appears in registered_adapters() AND can be
    iterated by a consumer that expects ``to_params`` to return a
    dataclass.  Guards against a regression where the endpoint
    hardcodes the engine list."""
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class SyntheticSuggestedParams:
        synth_charge: int
        synth_treatment: str
        rationale: str

    @register_adapter("synthetic")
    class SyntheticAdapter:
        name = "synthetic"

        @classmethod
        def to_params(cls, analysis):
            return SyntheticSuggestedParams(
                synth_charge    = analysis.suggested_charge,
                synth_treatment = analysis.suggested_treatment,
                rationale       = analysis.rationale,
            )

    reg = registered_adapters()
    assert "synthetic" in reg
    assert reg["synthetic"] is SyntheticAdapter

    # The endpoint iterates the registry — make sure that loop works
    # for a freshly-registered adapter.
    a = analyze_structure(_mk(["Fe"]))
    suggested = {
        name: asdict(cls.to_params(a))
        for name, cls in reg.items()
    }
    assert "synthetic" in suggested
    assert suggested["synthetic"]["synth_treatment"] == "open"


# --------------------------------------------------------------------- #
#  Adapter purity invariant                                             #
# --------------------------------------------------------------------- #


def test_an_adapter_translates_the_analysis_and_never_re_derives_it():
    """**An adapter is a PURE translator** (`science/validation.md` § 3): it
    reads the analysis's conclusions and spells them for its engine.  One
    that looked at the COMPOSITION again -- the metals, the electron count --
    would be a second chemistry answer, and two answers about one structure
    are how the auto-detect and the validator come to disagree.

    So each registered adapter is handed an analysis whose conclusions
    CONTRADICT its composition -- methane's elements, told open shell, spin
    2, charge +1 -- and a second analysis with the same conclusions over a
    very different composition: copper, one metal, an odd electron count.
    A translator returns the same thing for both.  One that re-derives
    returns methane's answer for the first.

    Walks the registry, so an engine's adapter added later is held to it
    without editing this test.  (This read each adapter module's imports
    until 2026-09-26; the adapter's answer is what a reader acts on.)

    API-level: ``/api/structure/analyze`` always hands an adapter the
    analysis OF the structure it was sent, which never contradicts its own
    composition -- so only a hand-made analysis can tell a translator from
    an adapter that re-derives.

    MUTATION THIS MUST FAIL AGAINST: an adapter reading
    ``analysis.metals`` / ``n_electrons_neutral`` for the shell or the spin
    instead of ``suggested_treatment`` / ``suggested_spin``.
    """
    import dataclasses

    told = dict(suggested_charge=1, suggested_spin=2,
                suggested_treatment="open",
                rationale="told by the analysis, not found by the adapter")
    methane = analyze_structure(_mk(["C", "H", "H", "H", "H"]))
    copper = analyze_structure(_mk(["Cu"]))
    assert (methane.metals, methane.n_electrons_neutral % 2) == ([], 0)
    assert (copper.metals, copper.n_electrons_neutral % 2) == (["Cu"], 1)
    contradicted = dataclasses.replace(methane, **told)
    same_verdict = dataclasses.replace(copper, **told)

    reg = registered_adapters()
    assert reg, "registry is empty -- at least siesta + pyscf must be registered"
    for engine, adapter in reg.items():
        got = asdict(adapter.to_params(contradicted))
        assert got == asdict(adapter.to_params(same_verdict)), (
            f"{engine}: the same conclusions over a different composition "
            f"translated differently -- the adapter is reading the "
            f"composition, not the analysis:\n  {got}")
        assert got != asdict(adapter.to_params(methane)), (
            f"{engine}: told open shell, spin 2 and charge +1, it answered "
            f"what methane's composition says -- the analysis was ignored")
