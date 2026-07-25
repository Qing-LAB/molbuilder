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
``validation.check_open_shell_metal`` (Phase 1d) and by
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


# --------------------------------------------------------------------- #
#  Noble-metal context awareness (2026-06-13)                           #
#                                                                       #
#  Cu / Ag / Au are open-shell as ATOMS (nd¹⁰ (n+1)s¹) but closed-shell #
#  singlet in any extended metallic context (cluster ≥ 4 atoms, surface,#
#  junction).  Stoner criterion fails for noble metals; the s-band      #
#  delocalises.  Pre-2026-06-13 the analyzer treated Au-BDT-Au as       #
#  open-shell and silently suggested spin=2, which is wrong for         #
#  every published Au transport calculation.                            #
#                                                                       #
#  Refs: Taylor/Brandbyge/Stokbro PRB 63 (2001) 245407 — the original   #
#  TranSIESTA Au-BDT-Au benchmark, spin-restricted DFT;                 #
#  Marder Ch. 17 — Stoner criterion derivation; Cu/Ag/Au explicitly     #
#  non-magnetic in bulk.                                                #
# --------------------------------------------------------------------- #


def test_au_cluster_4_atoms_closed_shell_singlet():
    """Au_4 (or larger): metallic bonding, even electron count.
    Closed-shell singlet — the standard treatment for Au junctions
    + Au surfaces in published transport / catalysis work."""
    a = analyze_structure(_mk(["Au"] * 4))
    assert a.suggested_spin == 0
    assert a.suggested_treatment == "closed"
    # Rationale must cite the metallic-bonding argument (not just
    # parity) so the user knows WHY we override the per-atom default.
    assert "metallic" in a.rationale.lower()
    assert "noble" in a.rationale.lower() or "Au" in a.rationale
    assert a.warnings == []


def test_au_bdt_au_junction_closed_shell_singlet():
    """The user-reported Au-BDT-Au case: 4 Au atoms + BDT ligand.
    Even total electron count, ≥ 4 Au → closed-shell singlet, NOT
    the pre-fix open-shell spin=2."""
    # 4 Au + benzene-1,4-dithiol (C6H4S2): Au_Z=79*4=316,
    # C_Z=6*6=36, H_Z=1*4=4, S_Z=16*2=32; total=388 (even).
    elements = ["Au"] * 4 + ["S"] * 2 + ["C"] * 6 + ["H"] * 4
    a = analyze_structure(_mk(elements))
    assert a.suggested_spin == 0
    assert a.suggested_treatment == "closed"


def test_single_au_atom_keeps_open_shell_doublet():
    """A SINGLE Au atom with odd electron count: respect the atomic
    open-shell ground state (5d¹⁰ 6s¹, S=½).  The metallic-bonding
    argument needs at least 4 atoms; below that, atomic physics wins."""
    a = analyze_structure(_mk(["Au"]))
    assert a.suggested_spin == 1
    assert a.suggested_treatment == "open"


def test_au_dimer_falls_through_to_parity():
    """Au_2 (sub-threshold cluster, even electron count): doesn't
    qualify for the closed-shell metallic-bonding argument (n < 4)
    but parity gives spin=0 anyway.  Rationale should flag that we're
    in the small-cluster regime, not the cluster-context argument."""
    a = analyze_structure(_mk(["Au"] * 2))
    assert a.suggested_spin == 0
    # Note in rationale that this is the small-cluster ambiguous case.
    assert "small" in a.rationale.lower() or "cluster" in a.rationale.lower()


# --------------------------------------------------------------------- #
#  J4 round-3 follow-up: noble-metal cluster threshold direction        #
#                                                                        #
#  The audit flagged that the existing Au-cluster tests only pin the    #
#  "closed-shell" side of the threshold — Au_4 / Cu_4 → closed.  A      #
#  regression where the rule ALWAYS returns closed-shell would          #
#  silently pass.  Pin the open-shell pole below (single atom + odd-    #
#  electron clusters above the threshold) so the threshold is checked  #
#  from BOTH directions.                                                #
# --------------------------------------------------------------------- #


def test_single_au_atom_open_shell():
    """A single Au atom: Z=79 (odd) → spin=1, treatment=open.
    Atomic ground state is 5d¹⁰ 6s¹, a single unpaired s electron.
    Pre-J4 the existing tests didn't pin treatment on the open-shell
    pole; a regression that hard-coded ``treatment="closed"`` would
    pass everywhere except this test."""
    a = analyze_structure(_mk(["Au"]))
    assert a.suggested_spin == 1, (
        f"Single Au: expected spin=1 (odd electron parity); "
        f"got spin={a.suggested_spin}"
    )
    assert a.suggested_treatment == "open", (
        f"Single Au: expected treatment='open' (s¹ ground state); "
        f"got treatment={a.suggested_treatment!r}"
    )


def test_au_trimer_below_threshold_odd_parity_open_shell():
    """Au_3: below the cluster threshold (3 < 4) AND odd-electron
    parity (3 × 79 = 237).  Should fall through to parity → spin=1,
    treatment=open.  The noble-metal cluster rule MUST NOT fire
    here -- if it did, the user would get a closed-shell singlet
    for an odd-electron system (catastrophic for SCF convergence).
    """
    a = analyze_structure(_mk(["Au"] * 3))
    assert a.suggested_spin == 1, (
        f"Au_3: expected spin=1 (odd parity); got spin={a.suggested_spin}.  "
        f"Cluster threshold should NOT fire below n=4."
    )
    assert a.suggested_treatment == "open", (
        f"Au_3: expected treatment='open'; got "
        f"treatment={a.suggested_treatment!r}"
    )


def test_au_5_atoms_above_threshold_but_odd_parity_open_shell():
    """Au_5: above the noble-metal cluster threshold (5 >= 4) but
    odd-electron parity (5 × 79 = 395).  The cluster rule should
    DEFER to parity for odd-electron systems -- the s-band-
    delocalisation argument only justifies closed-shell when the
    parity also allows it.  Pin so a future refactor that
    unconditionally fires closed-shell above the threshold (a real
    regression direction) doesn't silently corrupt every odd-
    parity Au-junction calculation."""
    a = analyze_structure(_mk(["Au"] * 5))
    assert a.suggested_spin == 1, (
        f"Au_5: odd parity must override the cluster rule; got "
        f"spin={a.suggested_spin}"
    )
    # The cluster rule logically applies (n>=4) but the analyzer
    # should NOT close-shell an odd-electron system.  The treatment
    # could legitimately be either "open" (the parity says so) or
    # the analyzer could fall through to "open" via the same path.
    # Either way, treatment != "closed".
    assert a.suggested_treatment != "closed", (
        f"Au_5: odd-electron above-threshold must NOT be "
        f"treatment='closed'; got {a.suggested_treatment!r}"
    )


def test_cu_cluster_4_atoms_closed_shell():
    """Cu obeys the same Stoner-fails / s-band-delocalises argument
    as Au.  Cu_4 → closed-shell singlet (4 × Cu_Z=29 = 116, even)."""
    a = analyze_structure(_mk(["Cu"] * 4))
    assert a.suggested_spin == 0
    assert a.suggested_treatment == "closed"


def test_pd_molecule_is_closed_shell():
    """Pd ground state is 4d¹⁰ 5s⁰ — a closed-shell atom (NIST
    spectra database).  The prior flat OPEN_SHELL_METALS incorrectly
    flagged Pd as open-shell; this test pins the correction."""
    a = analyze_structure(_mk(["Pd"] * 2))
    # Pd_Z=46, 2 × 46 = 92 (even), no open-d transition metal
    # present, no noble metal → falls through to parity → closed
    # singlet.  Pd ends up NOT in a.metals (it's CLOSED_D10).
    assert a.suggested_spin == 0
    assert a.suggested_treatment == "closed"
    assert "Pd" not in a.metals


def test_au_with_fe_coadsorbate_keeps_open_shell():
    """When an open-d transition metal IS present (Fe-Au alloy, or
    Au junction with an Fe co-adsorbate), the open-d metal's
    open-shell requirement overrides the noble-metal cluster logic.
    The system needs spin-polarised DFT for the Fe regardless of
    how many Au atoms surround it."""
    a = analyze_structure(_mk(["Au"] * 4 + ["Fe"]))
    assert a.suggested_treatment == "open"
    # Rationale notes that an open-d metal was detected.
    assert "Fe" in a.rationale


def test_open_d_transition_metal_subsets_have_no_overlap():
    """Defensive check: the three categorized sets are pairwise
    disjoint.  An element in two categories would produce
    contradictory spin suggestions depending on which membership
    test ran first."""
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
    ``total_electrons`` — the endpoint catches it for a clean 400."""
    with pytest.raises(KeyError):
        analyze_structure(_mk(["Xx"]))


def test_spin_upper_bound_and_electron_sanity():
    """SCIENTIFIC-AUDIT FIX: check_spin_charge_parity validated PARITY only;
    2S cannot exceed the electron count (n_beta = (n_e - spin)/2 >= 0), and an
    over-ionised system (charge > sum Z) has no electrons.  Both are exact and
    were previously unflagged (e.g. spin=10 on H2 passed)."""
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
