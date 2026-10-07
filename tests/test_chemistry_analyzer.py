"""The chemistry FACTS of a structure -- what ``analyze_structure`` answers.

PINS: ``docs/science/validation.md`` § 2 -- the analyzer reports what a
structure IS (its elements, its electron count, the metals that bear on its
spin and their common spin states), purely and deterministically.  It
decides nothing: which charge and spin a calculation carries is the
electronic-state class's, pinned through ``jobset prep`` in
``tests/test_electronic_state.py``.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from molbuilder.chemistry import analyze_structure
from molbuilder.structure import Structure


def _mk(elements, residue_name="MET"):
    """Minimal Structure whose atoms sit at the origin — a composition
    fixture: geometry does not matter to the facts."""
    n = len(elements)
    return Structure(
        elements      = list(elements),
        positions     = np.zeros((n, 3)),
        atom_names    = [f"A{i}" for i in range(n)],
        residue_ids   = [1] * n,
        residue_names = [residue_name] * n,
        chain_ids     = ["A"] * n,
    )


def test_metal_hints_list_one_per_metal_in_order():
    """One hint per metal that bears on the spin: the open-d metals, then
    the noble ones, each sorted."""
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


def test_elements_unique_sorted():
    """``elements`` is the sorted unique set, not the per-atom list."""
    a = analyze_structure(_mk(["H", "Fe", "C", "H", "C", "N"]))
    assert a.elements == ["C", "Fe", "H", "N"]


# --------------------------------------------------------------------- #
#  The metal classes (`science/validation.md` § 2.1): which set an       #
#  element is in decides its row of the detection table                 #
#  (`electronic_state.recommend`, `chemistry-correctness.md` § 2a.1b) -- #
#  a noble s1 metal in a cluster or a lead is a closed shell, an open-d #
#  metal is not.  (The rows themselves are pinned through prep in       #
#  tests/test_electronic_state.py.)                                     #
# --------------------------------------------------------------------- #


def test_open_d_transition_metal_subsets_have_no_overlap():
    """Defensive check: the three categorized sets are pairwise
    disjoint.  An element in two categories would be decided by
    whichever row of the detection table asked first."""
    from molbuilder.chemistry import (
        OPEN_D_TRANSITION_METALS,
        NOBLE_METALS_S1,
        CLOSED_D10_METALS,
    )
    assert not (OPEN_D_TRANSITION_METALS & NOBLE_METALS_S1)
    assert not (OPEN_D_TRANSITION_METALS & CLOSED_D10_METALS)
    assert not (NOBLE_METALS_S1 & CLOSED_D10_METALS)


def test_pd_pt_excluded_from_open_d_transition_set():
    """Pin the specific correction: Pd + Pt are NOT in
    OPEN_D_TRANSITION_METALS.  Pd is 4d¹⁰ 5s⁰ atomic ground state;
    Pt is 5d⁹ 6s¹ but metallic Pt is conventionally closed-shell in
    surface DFT.  Both belong in CLOSED_D10_METALS."""
    from molbuilder.chemistry import (
        OPEN_D_TRANSITION_METALS,
        CLOSED_D10_METALS,
    )
    assert "Pd" not in OPEN_D_TRANSITION_METALS
    assert "Pt" not in OPEN_D_TRANSITION_METALS
    assert "Pd" in CLOSED_D10_METALS
    assert "Pt" in CLOSED_D10_METALS


def test_unknown_element_raises_keyerror():
    """An unknown element symbol propagates as KeyError from
    ``resolve_element`` — the analyze route turns it into a clean 400."""
    with pytest.raises(KeyError):
        analyze_structure(_mk(["Xx"]))


def test_spin_upper_bound_and_electron_sanity():
    """Beyond parity: 2S cannot exceed the electron count
    (n_beta = (n_e - spin)/2 >= 0), and an over-ionised system
    (charge > sum Z) has no electrons."""
    import numpy as np
    from molbuilder.chemistry import check_spin_charge_parity
    from molbuilder.structure import Structure
    h2 = Structure(elements=["H", "H"],
                   positions=np.array([[0., 0, 0], [0, 0, 0.74]]))
    # 2 electrons: spin (2S) up to 2 is allowed; above that is impossible.
    assert check_spin_charge_parity(h2, 0, 2) is None          # 2 unpaired OK
    assert check_spin_charge_parity(h2, 0, 10) is not None     # > n_elec: error
    assert "exceeds" in check_spin_charge_parity(h2, 0, 10)
    # Over-ionised past the nuclei: negative electron count.
    assert check_spin_charge_parity(h2, 3, 0) is not None
