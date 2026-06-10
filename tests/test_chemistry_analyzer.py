"""Tests for the engine-agnostic chemistry analyzer (L2 of the
scientific-validation machinery; see
``docs/protocols/scientific-validation.md`` § 3).

Pins the contract:

* ``ChemistryAnalysis`` is a frozen dataclass (immutable conclusion).
* ``analyze_structure(struct)`` is a pure function (same input →
  same output; no I/O; no engine awareness).
* Open-shell metal detection picks the per-element default 2S;
  parity is enforced once; rationale + warnings explain the choice.
* The analyzer field shapes are the contract the adapters depend
  on; changes here propagate to per-engine ``to_params`` outputs.

Single-source-of-truth claim: this same analyzer is consumed by
``validation._check_open_shell_metal`` (Phase 1d) and by
``/api/structure/analyze`` (Phase 1c).  Cross-engine consistency
test lives in ``tests/test_chemistry_adapters.py``.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, FrozenInstanceError

import numpy as np
import pytest

from molbuilder.chemistry import (
    ChemistryAnalysis,
    MetalHint,
    SpinChoice,
    analyze_structure,
)
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


def _mk(elements, residue_name="MET"):
    """Minimal Structure whose atoms sit at the origin — pure
    composition fixture for chemistry-analysis tests.  Geometry
    doesn't matter to the analyzer."""
    n = len(elements)
    return Structure(
        elements      = list(elements),
        positions     = np.zeros((n, 3)),
        atom_names    = [f"A{i}" for i in range(n)],
        residue_ids   = [1] * n,
        residue_names = [residue_name] * n,
        chain_ids     = ["A"] * n,
    )


# --------------------------------------------------------------------- #
#  Dataclass shape                                                      #
# --------------------------------------------------------------------- #


def test_chemistry_analysis_is_frozen_dataclass():
    """The dataclass is frozen — conclusions are immutable once
    computed, so a consumer cannot mutate (and thereby disagree
    with) another consumer reading the same instance."""
    a = analyze_structure(_mk(["C", "H", "H", "H", "H"]))
    assert is_dataclass(a)
    with pytest.raises(FrozenInstanceError):
        a.suggested_spin = 99   # type: ignore[misc]


def test_metal_hint_and_spin_choice_are_frozen_dataclasses():
    """The nested hint dataclasses are also frozen.  Same rationale."""
    a = analyze_structure(_mk(["Fe"]))
    assert is_dataclass(a.metal_hints[0])
    assert is_dataclass(a.metal_hints[0].common_spins[0])
    with pytest.raises(FrozenInstanceError):
        a.metal_hints[0].element = "X"   # type: ignore[misc]


def test_asdict_round_trip_carries_full_payload():
    """``dataclasses.asdict`` is the single serialisation point for
    the HTTP wire boundary.  Pin every documented field is reachable
    via ``asdict``."""
    a = analyze_structure(_mk(["Fe"]))
    d = asdict(a)
    assert set(d.keys()) >= {
        "n_atoms", "elements", "n_electrons_neutral",
        "metals", "metal_hints",
        "suggested_charge", "suggested_spin", "suggested_treatment",
        "rationale", "warnings",
    }
    # Nested metal_hints serialise to dicts too.
    assert d["metal_hints"][0]["element"] == "Fe"
    assert "common_spins" in d["metal_hints"][0]


# --------------------------------------------------------------------- #
#  Engine-agnostic conclusions                                          #
# --------------------------------------------------------------------- #


def test_pure_organic_even_electrons_closed_singlet():
    """Pure organic with even electron count → closed-shell singlet.
    No metals, no advisories, no warnings."""
    # CH4: Z = 6 + 4 = 10 (even)
    a = analyze_structure(_mk(["C", "H", "H", "H", "H"]))
    assert a.metals == []
    assert a.suggested_spin == 0
    assert a.suggested_treatment == "closed"
    assert a.warnings == []


def test_pure_organic_odd_electrons_doublet():
    """Pure organic with odd electron count (a radical, e.g. CH3·)
    → spin = 1 open, no metal warnings."""
    # CH3: Z = 6 + 3 = 9 (odd)
    a = analyze_structure(_mk(["C", "H", "H", "H"]))
    assert a.metals == []
    assert a.suggested_spin == 1
    assert a.suggested_treatment == "open"
    assert a.warnings == []


def test_single_fe_gets_fe2_intermediate_default_spin():
    """Fe alone → spin=2 (Fe(II) intermediate, the molbuilder
    hemeC use case).  Pin the per-element default + the rationale
    points at the metal explicitly."""
    a = analyze_structure(_mk(["Fe"]))
    assert a.metals == ["Fe"]
    assert a.suggested_spin == 2
    assert a.suggested_treatment == "open"
    assert "Fe" in a.rationale
    assert a.warnings == []   # Fe Z=26 even, default 2 even → no parity bump


def test_cu2_gets_doublet_spin1():
    """Cu(II) is d⁹ — exactly one unpaired electron.  Pin the
    per-element default 2S=1 for Cu (the textbook chemistry)."""
    # Cu Z=29 (odd), default spin=1 (odd) → parity matches → no bump
    a = analyze_structure(_mk(["Cu"]))
    assert a.metals == ["Cu"]
    assert a.suggested_spin == 1
    assert a.suggested_treatment == "open"


def test_mn_high_spin_default():
    """Mn(II) is overwhelmingly high-spin S=5/2 (2S=5) in
    biological contexts.  Pin the per-element default."""
    # Mn Z=25 (odd), default 2S=5 (odd) → matches → no bump
    a = analyze_structure(_mk(["Mn"]))
    assert a.metals == ["Mn"]
    assert a.suggested_spin == 5


# --------------------------------------------------------------------- #
#  Parity rule (single source of truth)                                 #
# --------------------------------------------------------------------- #


def test_parity_bumps_spin_and_records_warning():
    """When the per-element default spin's parity contradicts
    the electron-count parity, the analyzer bumps the spin AND
    records the adjustment in ``warnings``.  Pin so an adapter
    that re-does parity work would surface (the work belongs
    here, once)."""
    # Fe (Z=26 even, default 2S=2 even → matches) + H (Z=1 → total odd)
    # → spin must be odd → bump from 2 to 1
    a = analyze_structure(_mk(["Fe", "H"]))
    assert a.metals == ["Fe"]
    assert a.suggested_spin == 1
    assert len(a.warnings) == 1
    assert "parity" in a.warnings[0].lower()
    assert "27" in a.warnings[0]   # sum(Z) reported in the warning


def test_parity_zero_default_bumps_up_not_down():
    """A zero-spin default that contradicts odd electron count
    bumps UP (to 1), not down (which would underflow to -1).

    Pins the ``spin + 1 if spin == 0 else spin - 1`` branch of the
    parity rule.  Without this test the +1 (up-bump) path is dead
    code as far as the suite is concerned.
    """
    # Ni Z=28 (even), default 2S=0 (even, matches) + H (Z=1) → Z=29 odd
    # → bump from 0 to 1 (UP)
    a = analyze_structure(_mk(["Ni", "H"]))
    assert a.suggested_spin == 1
    assert a.warnings, "Parity bump should record a warning"
    assert "parity" in a.warnings[0].lower()


def test_unknown_metal_falls_through_to_default_spin_2():
    """A metal not in ``_ANALYZER_DEFAULT_SPIN`` falls through to
    spin=2 (the heavier-d-block default).  Pin so a future
    refactor that changes the fallthrough value (or removes the
    ``.get(metal, 2)`` default) surfaces.

    Mo is open-shell (in OPEN_SHELL_METALS) but not in the
    first-row-d table; it's the classic fallthrough case.
    Mo Z=42 (even); default 2 (even) → matches → no parity bump.
    """
    a = analyze_structure(_mk(["Mo"]))
    assert a.metals == ["Mo"]
    assert a.suggested_spin == 2
    assert a.warnings == []   # no parity bump


# --------------------------------------------------------------------- #
#  Metal hint completeness                                              #
# --------------------------------------------------------------------- #


def test_metal_hints_list_one_per_metal_in_order():
    """One hint per detected metal, in first-appearance order
    (matches ``detect_open_shell_metals`` contract)."""
    a = analyze_structure(_mk(["Fe", "C", "Cu", "H"]))
    assert [h.element for h in a.metal_hints] == ["Fe", "Cu"]


def test_metal_hint_carries_ranked_spin_choices():
    """Each MetalHint carries a non-empty common_spins list with
    SpinChoice entries that have ``spin`` (int) + ``label`` (str)."""
    a = analyze_structure(_mk(["Fe"]))
    hint = a.metal_hints[0]
    assert len(hint.common_spins) >= 1
    for c in hint.common_spins:
        assert isinstance(c.spin, int)
        assert c.label and isinstance(c.label, str)


# --------------------------------------------------------------------- #
#  Purity                                                               #
# --------------------------------------------------------------------- #


def test_analyze_structure_is_deterministic():
    """Same input → same output.  Pin so a future caching or
    randomised heuristic introduction surfaces."""
    s = _mk(["Fe", "C", "N", "N", "N", "N"])
    a1 = analyze_structure(s)
    a2 = analyze_structure(s)
    # asdict round-trip because frozen dataclasses compare by value
    # but list fields compare element-wise (which is also fine here).
    assert asdict(a1) == asdict(a2)


def test_n_atoms_matches_structure():
    """The analyzer reads n_atoms from the Structure dataclass — no
    parallel state."""
    a = analyze_structure(_mk(["C", "H", "H"]))
    assert a.n_atoms == 3


def test_elements_unique_sorted():
    """``elements`` is the sorted unique set, not the per-atom list."""
    a = analyze_structure(_mk(["H", "Fe", "C", "H", "C", "N"]))
    assert a.elements == ["C", "Fe", "H", "N"]


def test_unknown_element_raises_keyerror():
    """An unknown element symbol propagates as KeyError from
    ``total_electrons`` — the endpoint catches it for a clean 400."""
    with pytest.raises(KeyError):
        analyze_structure(_mk(["Xx"]))
