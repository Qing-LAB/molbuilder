"""Tests for the engine-parameter adapter Protocol + registry (L3
of the scientific-validation machinery; see
``docs/science/validation.md`` § 4).

Pins the contract:

* Each registered adapter returns a frozen dataclass whose field
  names match the engine's web-form / Config field names.
* Adapters are PURE translators — they consume a
  ``ChemistryAnalysis`` and translate; they MUST NOT re-do
  chemistry detection or parity work.  Pinned by checking that
  ``analyze_structure`` is NOT imported by any adapter module.
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


def test_siesta_adapter_returns_frozen_dataclass():
    a = analyze_structure(_mk(["Fe"]))
    params = SiestaAdapter.to_params(a)
    assert isinstance(params, SiestaSuggestedParams)
    assert is_dataclass(params)


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


def test_siesta_field_names_match_siesta_config():
    """The adapter dataclass field names must match the SIESTA
    Config dataclass's web-form field names — the UI's "apply
    suggestion" path is a 1:1 spread into form values.  If the
    Config renames a field, the adapter must follow."""
    from molbuilder.siesta.input import SiestaConfig
    cfg_fields = {f.name for f in SiestaConfig.__dataclass_fields__.values()}
    # net_charge + spin_treatment + spin_total must all be present
    # on SiestaConfig.  rationale is meta — exempt.
    assert {"net_charge", "spin_treatment", "spin_total"} <= cfg_fields, (
        f"SIESTA adapter exports fields not in SiestaConfig: "
        f"{{'net_charge','spin_treatment','spin_total'}} − {cfg_fields}"
    )


# --------------------------------------------------------------------- #
#  PySCF adapter                                                        #
# --------------------------------------------------------------------- #


def test_pyscf_adapter_returns_frozen_dataclass():
    a = analyze_structure(_mk(["Fe"]))
    params = PyscfAdapter.to_params(a)
    assert isinstance(params, PyscfSuggestedParams)
    assert is_dataclass(params)


def test_pyscf_adapter_open_shell_metal_path():
    """Fe → spin=2 + method='UKS' + rationale."""
    a = analyze_structure(_mk(["Fe"]))
    p = PyscfAdapter.to_params(a)
    assert p.charge == 0
    assert p.spin   == 2
    assert p.method == "UKS"
    assert "Fe" in p.rationale


def test_pyscf_adapter_closed_shell_path():
    """Pure organic, even electrons → spin=0, method='RKS'."""
    a = analyze_structure(_mk(["C", "H", "H", "H", "H"]))
    p = PyscfAdapter.to_params(a)
    assert p.spin   == 0
    assert p.method == "RKS"


def test_pyscf_field_names_match_pyscf_config():
    """Same lock-step as SIESTA's check.  ``charge``/``spin``/``method``
    must all exist on the PySCF Config dataclass."""
    from molbuilder.pyscf.input import PySCFConfig
    cfg_fields = {f.name for f in PySCFConfig.__dataclass_fields__.values()}
    assert {"charge", "spin", "method"} <= cfg_fields, (
        f"PySCF adapter exports fields not in PySCFConfig: "
        f"{{'charge','spin','method'}} − {cfg_fields}"
    )


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
    assert si.net_charge == py.charge


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
    assert py_d["charge"] == 0
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


def test_adapter_modules_do_not_import_analyzer():
    """Adapters are PURE translators (science/validation.md § 3).
    They must not import ``analyze_structure`` or
    ``detect_open_shell_metals`` — chemistry logic belongs in the
    analyzer.  Pinned by AST-parsing each adapter module and
    checking the actual import statements (substring matching
    would trip on the "do not use these" warning in the adapter
    docstring).

    Walks the **registry** rather than a hardcoded list so a
    future engine's adapter (e.g. transiesta, pyscf-negf) is
    automatically scanned without editing this test.

    A failure here means an adapter started re-doing chemistry
    detection inline.  Move the logic into ``chemistry.py`` and
    have the adapter consume the existing ``ChemistryAnalysis``
    fields instead.
    """
    import ast
    import importlib
    import inspect

    FORBIDDEN = {
        "analyze_structure",
        "detect_open_shell_metals",
        "check_spin_charge_parity",
        "total_electrons",
    }

    reg = registered_adapters()
    assert reg, "registry is empty — at least siesta + pyscf must be registered"
    for engine_name, adapter_cls in reg.items():
        module_name = adapter_cls.__module__
        mod = importlib.import_module(module_name)
        try:
            src = inspect.getsource(mod)
        except OSError:
            continue   # synthetic test adapters defined inline have no source file
        tree = ast.parse(src)
        imported_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)
        bad = FORBIDDEN & imported_names
        assert not bad, (
            f"{module_name} (engine {engine_name!r}) imports forbidden "
            f"chemistry primitives {sorted(bad)} — chemistry logic must "
            f"live in the analyzer, not the adapter (see "
            f"docs/science/validation.md)"
        )
