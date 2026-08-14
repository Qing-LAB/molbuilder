"""Cross-engine chemistry helpers.

L1 module shared by SIESTA + PySCF + Spectra emitters and the
validation pass.  Grew from the original "two phosphate functions"
into a small library of inferences that work on a bare ``Structure``
(no explicit bond orders required) using heavy-atom adjacency.

Public surface (grouped by purpose):

  Charge inference (DNA / RNA / phosphates):
    formal_charge_from_phosphates(struct) -> int
        Sum of -1 per deprotonated non-bridging phosphate oxygen.
    protonate_phosphate_oxygens(struct) -> (struct, n_added)
        Add H to deprotonated O so the molecule becomes neutral.
    expected_pH7_peptide_charge(struct) -> Optional[int]
        Rough physiological-pH net charge for a peptide.

  Electron / spin parity (open-shell guards):
    total_electrons(struct, charge) -> int
    check_spin_charge_parity(struct, charge, spin) -> Optional[str]

  Open-shell transition metals:
    detect_open_shell_metals(struct) -> List[str]
    explain_metal_spin(element, spin) -> Optional[str]
    suggest_spin_total(metals) -> (preferred, alternatives)
        Used by the SIESTA preflight to recommend a Spin.Total when
        the user enables spin polarisation but leaves the target
        spin unset (catches the propor: ERROR: IMAX = 0 abort).

  ECP selection (PySCF):
    resolve_pyscf_ecp(struct, ecp, basis) -> ECP-or-None
        Cross-engine helper used by both Build PySCF and Spectra so
        the heuristic stays in one place.

  Hydrogen placement:
    add_hydrogens(struct, ...) -> struct
        Geometry-only H placement (no force-field).

The chemistry rule encoded in the phosphate helpers is the standard
interpretation: one P=O double bond, the other oxygens single-bonded
to H, R-O-, or O- depending on protonation.  "Non-bridging" is
inferred purely from heavy-atom adjacency (only P as a heavy
neighbour).  Other charged groups (carboxylates, protonated amines)
are NOT counted.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from typing import (Any, Dict, List, Literal, Optional, Protocol, Sequence,
                    Tuple, Type)

import numpy as np

from .structure import Structure


# Charged amino-acid side chains at physiological pH (7.4).  Used by
# `expected_pH7_peptide_charge()` to estimate the net charge of a
# peptide for the validator's "you built a neutral peptide but it
# would be charged at pH 7" hint.
#
# Histidine is intentionally skipped: pKa ~6 means it's roughly half-
# protonated at pH 7, and its protonation state is sequence- and
# environment-dependent (usually neutral N-tau-H tautomer in
# proteins).  Cys and Tyr have side-chain pKa > 8, so they're
# neutral at pH 7 -- not counted.
_CHARGED_RESIDUES_PH7 = {
    "ASP": -1, "GLU": -1,                       # acidic
    "LYS":  1, "ARG":  1,                       # basic
}

# Standard (and modified) amino-acid 3-letter codes that come out of
# molbuilder's peptide builder.  Used to detect "is this a peptide"
# for the protonation hint.
_AMINO_ACID_RESIDUE_NAMES = frozenset({
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    # Modified residues we support
    "SEP","TPO","PTR","MLY","M3L","ALY",
})


# Open-shell transition metals + lanthanides.  Their ground-state
# configurations have unpaired electrons in d (or f) shells in every
# common oxidation state, so running them through a closed-shell
# singlet SCF (spin=0, method=RKS) typically converges to a fictitious
# state with garbage forces.  The user MUST either explicitly set
# spin to a sensible value AND switch to UKS / ROKS, or accept that
# the result is nonsensical.  We surface a preflight WARN for any
# structure containing one of these.
#
# Source: ground-state electron configurations from NIST atomic
# spectra database.
#
# 2026-06-13 split (replaces the prior flat ``OPEN_SHELL_METALS``).
# The unified set treated Au-BDT-Au junctions as open-shell and
# silently produced wrong spin suggestions.  Three physical categories:
#
#   1. OPEN_D_TRANSITION_METALS — incomplete d-shell in the atomic
#      ground state AND extended phases.  Stoner criterion satisfied
#      for the 3d ferromagnets; itinerant moments for the 4d / 5d
#      analogues.  Open-shell DFT is the default expectation.
#
#   2. NOBLE_METALS_S1 — Cu, Ag, Au.  Atomic ground state is
#      nd¹⁰ (n+1)s¹ — single unpaired s electron — but in any
#      extended metallic context (cluster ≥ 4 atoms, surface, bulk,
#      junction) the s-band delocalizes and the system is closed-
#      shell singlet for even total electron count.  Stoner criterion
#      fails for noble metals: I·N(E_F) < 1, no spontaneous magnetism
#      in bulk.  Standard treatment for Au transport junctions is
#      spin-restricted DFT.
#      Refs:
#        * Taylor, Brandbyge, Stokbro, PRB 63 (2001) 245407 — the
#          original TranSIESTA Au-BDT-Au paper, spin-restricted.
#        * Ke, Baranger, Yang, JCP 122 (2005) 074704 — Au-BDT-Au NEGF.
#        * Verzijl, Thijssen, JPCC 116 (2012) 24811 — DFT+Σ Au-
#          alkanedithiol-Au benchmark, spin-restricted.
#        * Marder, "Condensed Matter Physics" Ch. 17 — Stoner
#          criterion derivation; Cu/Ag/Au listed as non-magnetic.
#      When open-shell IS the right call for noble-metal systems:
#      sub-4-atom clusters (shell-closing incomplete; Au_2 / Au_4),
#      single Au atom on insulator (Au/CeO2, Au/MgO catalysis lit.),
#      Au with magnetic 3d co-adsorbate (Au-Co, Au-Fe alloys), or
#      explicit Kondo / spin-orbit studies.  Users in those regimes
#      override via the form.
#
#   3. CLOSED_D10_METALS — Zn, Cd, Hg (always nd¹⁰ (n+1)s² in
#      common oxidation states) PLUS Pd (4d¹⁰ 5s⁰ atomic ground state
#      per NIST — the prior flat set incorrectly classified Pd as
#      open-shell) AND Pt (5d⁹ 6s¹ atom but 5d¹⁰-like in metallic
#      bonding; same logic as the noble metals but conventionally
#      treated as closed-shell in catalysis surface DFT).
OPEN_D_TRANSITION_METALS = frozenset({
    # First-row 3d (incomplete d-shell)
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    # Second-row 4d (Pd excluded — closed-shell atomic ground state)
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    # Third-row 5d (Pt + Au excluded — handled below)
    "Hf", "Ta", "W",  "Re", "Os", "Ir",
    # Lanthanides (4f incomplete)
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Common actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu",
})

NOBLE_METALS_S1 = frozenset({"Cu", "Ag", "Au"})

CLOSED_D10_METALS = frozenset({
    "Zn", "Cd", "Hg",          # ns² nd¹⁰ — always closed-shell
    "Pd",                       # 4d¹⁰ 5s⁰ atomic ground state (NIST)
    "Pt",                       # 5d⁹ 6s¹ atom; metallic Pt is
                                # conventionally closed-shell in surface
                                # DFT.  Catalysis lit. treats Pt(111)
                                # as RKS unless studying magnetism.
})

# Backward-compat alias.  Old callers that imported the flat set get
# the union; new code reaches for the categorized sets above.  Keep
# this alias for the deprecation window — and document the new
# distinction so callers can migrate.
OPEN_SHELL_METALS = OPEN_D_TRANSITION_METALS | NOBLE_METALS_S1


def total_electrons(struct: Structure, charge: int = 0) -> int:
    """Sum of atomic numbers minus charge.  Used by parity checks
    (closed-shell spin=0 requires an even count; an odd total means
    spin must be at least 1).  Raises KeyError on an unknown element
    symbol (catches typos before PySCF does the same).
    """
    from ase.data import atomic_numbers as _Z
    total = 0
    for el in struct.elements:
        try:
            total += int(_Z[el.capitalize()])
        except KeyError:
            raise KeyError(
                f"unknown element symbol {el!r} -- check the structure "
                f"file (typos, lowercase letters, missing-element-column "
                f"fallback failures)"
            )
    return total - int(charge)


def atomic_mass(element: str) -> float:
    """Standard atomic weight of ``element``, in amu.

    The same lookup :func:`total_electrons` does for atomic number, for
    the other number every engine needs: ASE ships the IUPAC standard
    atomic weights, so this is a name for a table rather than a copy of
    one.  Typing 118 masses into this file would be a second source of
    truth that no test would ever notice going stale.

    WHO NEEDS IT.  Any mass-weighted quantity: which atoms carry a
    vibrational mode (``spectra.results.motion_share_by_element``),
    reduced masses, centre of mass.  Mass is a chemistry fact and lives
    beside the other chemistry facts, not in the module that happens to
    want it first.

    NOT the isotope-resolved mass -- the standard weight is the natural
    isotopic average, which is what a Hessian is built with unless a run
    deliberately substitutes (deuteration), and PySCF's own default
    table is the same convention.  Raises KeyError on an unknown symbol,
    with the same message shape as :func:`total_electrons`.
    """
    from ase.data import atomic_numbers as _Z, atomic_masses as _M
    try:
        return float(_M[_Z[str(element).capitalize()]])
    except KeyError:
        raise KeyError(
            f"unknown element symbol {element!r} -- check the structure "
            f"file (typos, lowercase letters, missing-element-column "
            f"fallback failures)"
        )


def check_spin_charge_parity(struct: Structure, charge: int, spin: int
                              ) -> Optional[str]:
    """Return a human-readable error string when the (charge, spin)
    pair is impossible for ``struct``, else None.

    Rule: PySCF's / SIESTA's spin counts UNPAIRED electrons (= 2S =
    n_alpha - n_beta), so its parity must match the total electron
    count's parity (Σ Z - charge).  Closed-shell singlet (spin=0)
    requires an even electron count; doublet (spin=1) requires odd.
    PySCF raises ``RuntimeError("Mol.nelectron N is odd, but spin =
    0")`` at runtime on a mismatch; catching it at preflight gives
    a clearer message before the user spends minutes on a doomed
    SCF.

    Engine-independent: callers from BOTH _validate_pyscf and
    _validate_siesta share this helper (electron count doesn't
    care which code runs the SCF).
    """
    # Reject non-integer / negative spin up front.  The existing
    # _validate_pyscf has a separate "spin < 0" check; we add the
    # type check here so the parity-vs-spin arithmetic never silently
    # operates on a float (which would give a useless suggested fix
    # like "change spin to 2.5 / 0.5").
    if not isinstance(spin, int) or isinstance(spin, bool):
        return (f"spin={spin!r} must be a non-negative int "
                f"(2S = number of unpaired electrons)")
    if spin < 0:
        return (f"spin={spin} is negative; spin counts unpaired "
                f"electrons (2S), must be 0 or positive")
    n_elec = total_electrons(struct, charge)
    # Electron-count sanity: over-ionised past the nucleus is impossible.
    if n_elec < 0:
        return (
            f"charge={charge} removes more electrons than exist "
            f"(sum(Z) - charge = {n_elec} < 0) -- the system has no "
            f"electrons left.  Reduce the positive charge."
        )
    # Spin upper bound (EXACT): 2S = n_alpha - n_beta with n_alpha + n_beta =
    # n_elec, so n_beta = (n_elec - spin)/2 >= 0 requires spin <= n_elec.  You
    # cannot have more unpaired electrons than electrons.  (Parity alone let
    # e.g. spin=10 on H2 pass.)
    if spin > n_elec:
        return (
            f"spin={spin} exceeds the electron count (sum(Z) - charge "
            f"= {n_elec}): 2S = number of unpaired electrons cannot be "
            f"larger than the total number of electrons.  Lower spin to "
            f"at most {n_elec}."
        )
    if (n_elec % 2) != (spin % 2):
        return (
            f"Electron-count parity mismatch: sum(Z) - charge "
            f"= {n_elec}, which is {'even' if n_elec % 2 == 0 else 'odd'}; "
            f"spin={spin} requires a{'n even' if spin % 2 == 0 else 'n odd'} "
            f"electron count.  Either adjust charge by ±1 or change "
            f"spin to {spin + 1} / {max(0, spin - 1)} to restore parity."
        )
    return None


# Common spin-state -> (oxidation state, name) mapping for first-row
# transition metals.  Keyed by (element, spin = 2S).  Used by the
# preflight to explain to the user what their (charge, spin) input
# IMPLIES about the metal centre.
#
# Source: standard ligand-field theory; ground-state d-electron counts
# from CRC handbook.  Entries are intentionally restricted to the
# COMMON oxidation states a user is likely to encounter in
# biological / coordination chemistry; rare ones (e.g. Fe(0), Fe(VI))
# are omitted to avoid noisy guesses.
_METAL_SPIN_HINTS: dict = {
    # Fe: d⁶ for Fe(II), d⁵ for Fe(III)
    ("Fe", 0): "Fe(II), low-spin (S=0, 0 unpaired) -- e.g. CO- or CN⁻-bound heme",
    ("Fe", 2): "Fe(II), intermediate-spin (S=1, 2 unpaired) -- rare",
    ("Fe", 4): "Fe(II), high-spin (S=2, 4 unpaired) -- e.g. deoxy-heme, bis-thiolate",
    ("Fe", 1): "Fe(III), low-spin (S=1/2, 1 unpaired) -- e.g. bis-imidazole heme",
    ("Fe", 3): "Fe(III), intermediate-spin (S=3/2, 3 unpaired) -- e.g. quantum-admixed S=3/2 5-coord Fe(III) porphyrins (oxoferryl is Fe(IV), not this)",
    ("Fe", 5): "Fe(III), high-spin (S=5/2, 5 unpaired) -- e.g. met-myoglobin",
    # Mn: d⁵ for Mn(II), d⁴ for Mn(III)
    # (d⁵ is ODD -> minimum one unpaired electron; Mn(II) has NO 2S=0 state.
    #  Low-spin Mn(II) is S=1/2 = 2S=1, not 0.)
    ("Mn", 1): "Mn(II), low-spin (S=1/2, 1 unpaired) -- rare for Mn²⁺",
    ("Mn", 5): "Mn(II), high-spin (S=5/2, 5 unpaired) -- common for free Mn²⁺",
    ("Mn", 4): "Mn(III), high-spin (S=2, 4 unpaired)",
    # Co: d⁷ for Co(II), d⁶ for Co(III)
    ("Co", 1): "Co(II), low-spin (S=1/2, 1 unpaired)",
    ("Co", 3): "Co(II), high-spin (S=3/2, 3 unpaired)",
    ("Co", 0): "Co(III), low-spin (S=0, 0 unpaired) -- e.g. cobalamin/B12",
    # Cu: d⁹ for Cu(II), d¹⁰ for Cu(I)
    ("Cu", 1): "Cu(II) (S=1/2, 1 unpaired)",
    ("Cu", 0): "Cu(I), d¹⁰ closed-shell",
    # Ni: d⁸ for Ni(II)
    ("Ni", 0): "Ni(II), low-spin square-planar (S=0)",
    ("Ni", 2): "Ni(II), high-spin tetrahedral / octahedral (S=1, 2 unpaired)",
}


def explain_metal_spin(element: str, spin: int) -> Optional[str]:
    """Return a one-line description of what (element, spin) implies
    for a transition-metal centre (oxidation state + spin-state name).
    None if no hint is registered for this combination.
    """
    return _METAL_SPIN_HINTS.get((element.capitalize(), int(spin)))


# Per-element "starting value" recommendation for Spin.Total.  Used by
# the SIESTA preflight when spin_polarized=True + spin_total=None +
# the structure contains an open-shell metal.  Without a starting
# value SIESTA's initial-DM constructor (propor) can't find a
# zero-net-spin split for d/f shells and aborts with
# ``propor: ERROR: IMAX = 0`` before the SCF loop ever runs.
#
# Each entry is (preferred_starting_value, ranked alternatives).  The
# preferred value is the "most likely correct" guess for a typical
# biological / coordination-chem context (heme-like for Fe, etc.);
# the alternatives are ALL the registered (element, spin) hints sorted
# from low-spin to high-spin so the user can sweep them if the first
# guess doesn't converge.  Numbers are 2S (= Spin.Total in μB units),
# matching SIESTA's convention.
_SPIN_TOTAL_DEFAULTS: dict = {
    # ----- First-row d-block (the bio + organometallic mainstays) -----
    # Sc(III) is d⁰ closed-shell; Sc(II) is d¹ -- pick the open-shell
    # case as default since the check fires only on OPEN-shell metals.
    "Sc": 1.0,
    # Ti(III) is d¹ S=1/2; Ti(II) is d² S=1.  Pick HS-leaning default.
    "Ti": 2.0,
    # V(III) is d² S=1 octahedral; V(II) d³ S=3/2; V(IV) d¹ S=1/2.
    # Mid-row defaults to the most spin-active common state.
    "V":  3.0,
    # Cr(II) is d⁴ HS S=2; Cr(III) is d³ S=3/2.  Default to HS Cr(II)
    # (most common bio context: Cr-acetate, organometallic precursors).
    "Cr": 4.0,
    # Mn(II) is overwhelmingly high-spin S=5/2 in biological contexts.
    "Mn": 5.0,
    # Fe: heme-like deoxy-bis-thiolate is the molbuilder hemeC use
    # case -- high-spin Fe(II) S=2 is the most common starting point.
    "Fe": 4.0,
    # Co(II) octahedral is often high-spin S=3/2; low-spin variants
    # need explicit override.
    "Co": 3.0,
    # Ni(II) square-planar is closed-shell; octahedral is S=1.  No
    # safe default -- pick the higher-spin starting guess so SCF
    # has somewhere non-trivial to land.
    "Ni": 2.0,
    # Cu(II) is d⁹ -- one unpaired electron, period.
    "Cu": 1.0,
    # ----- Second-row d-block (heavier, often via ECP) -----
    # Mo(III) d³ S=3/2; Mo(IV) d² S=1.  Often HS in bio contexts
    # (Mo-nitrogenase active site).
    "Mo": 3.0,
    # Ru(II) low-spin d⁶ S=0; Ru(III) low-spin d⁵ S=1/2.  Pick Ru(III)
    # default since open-shell Ru is the case the check fires for.
    "Ru": 1.0,
    "Rh": 1.0,    # Rh(II) d⁷ S=1/2
    # ----- Third-row d-block -----
    "W":  2.0,    # W(IV) d² S=1
    "Re": 2.0,    # Re(III) d⁴ low-spin S=1 (5d ⇒ strong field ⇒ low-spin;
                  # 2S must be EVEN for even-electron d⁴ — 3.0 was parity-impossible)
    "Os": 1.0,    # Os(III) d⁵ low-spin S=1/2
    "Ir": 1.0,    # Ir(IV) d⁵ low-spin S=1/2
    "Pt": 1.0,    # Pt(III) d⁷ S=1/2 (Pt(II) / Pt(IV) are closed-shell)
    # ----- f-block (lanthanides + actinides) -----
    # 4f shells are usually well-localised; Hund's-rule HS is the
    # safe starting guess.  Numbers below are the free-ion ground-
    # state 2S values (NOT 2J; SIESTA's Spin.Total is 2S).
    "Ce": 1.0,    # 4f¹       2S=1
    "Pr": 2.0,    # 4f²       2S=2
    "Nd": 3.0,    # 4f³       2S=3
    "Pm": 4.0,    # 4f⁴       2S=4
    "Sm": 5.0,    # 4f⁵       2S=5
    "Eu": 6.0,    # 4f⁶       2S=6  (Eu(II) is 4f⁷ -> 2S=7; pick the
                  # less-extreme starter since Eu(III) more common)
    "Gd": 7.0,    # 4f⁷ S=7/2 -- archetypal "max unpaired" lanthanide
    "Tb": 6.0,    # 4f⁸       2S=6
    "Dy": 5.0,    # 4f⁹       2S=5
    "Ho": 4.0,    # 4f¹⁰      2S=4
    "Er": 3.0,    # 4f¹¹      2S=3
    "Tm": 2.0,    # 4f¹²      2S=2
    "Yb": 1.0,    # 4f¹³      2S=1
    # Actinides: defer to free-ion 2S for the +3 oxidation state.
    "U":  3.0,    # U(III) 5f³  -- common organoactinide oxidation state
    "Np": 4.0,    # Np(III) 5f⁴
    "Pu": 5.0,    # Pu(III) 5f⁵
}


def suggest_spin_total(metals: "Iterable[str]") -> "tuple[float, list[tuple[float, str]]]":
    """Recommend a starting ``Spin.Total`` (2S, in μB) for a structure
    containing the named open-shell metals + a ranked alternatives list.

    Pick rule when multiple metals are present: take the LARGEST per-
    element default (most-unpaired starting guess).  Reasoning: SIESTA's
    propor() failure mode is "can't split a d/f shell into zero net
    spin", so the safe starting bet is non-zero spin on the most-
    spin-active atom -- the optimiser can ramp DOWN from there if a
    lower-spin state is the true ground state.  Ramping UP from zero
    spin is what triggered the abort in the first place.

    Args:
      metals: result of detect_open_shell_metals(struct).

    Returns:
      (preferred_value, alternatives) where
        preferred_value: float, what to set Spin.Total to as a START.
        alternatives:    list of (value, "description") tuples drawn
                         from the per-element hints, in order from
                         low-spin to high-spin (so the user can sweep).
        If no metals are recognised, returns (1.0, []) -- a safe
        non-zero placeholder; the user will need to think about it.
    """
    metals_seen = [m.capitalize() for m in metals]
    if not metals_seen:
        return 1.0, []
    # Preferred starting value: max per-element default across the
    # metals present.  ``1.0`` is the fallback when a metal isn't
    # in our table (better than zero -- propor needs non-zero).
    preferred = max(
        (_SPIN_TOTAL_DEFAULTS.get(m, 1.0) for m in metals_seen),
        default=1.0,
    )
    # Alternatives list: every (element, spin) hint we have registered
    # for the metals present.  Sorted by spin value so the user reads
    # low-spin -> high-spin (chemists think in that order).
    alternatives: "list[tuple[float, str]]" = []
    for (el, spin_2s), desc in _METAL_SPIN_HINTS.items():
        if el in metals_seen:
            alternatives.append((float(spin_2s), f"{el}: {desc}"))
    alternatives.sort(key=lambda t: (t[0], t[1]))
    return float(preferred), alternatives


def resolve_pyscf_ecp(struct: Structure,
                      ecp: str,
                      ecp_atoms: "Sequence[str]") -> "Optional[Dict[str, str]]":
    """Which elements get which ECP -- the ONE place the rule lives.

    Called from BOTH Build (``pyscf/input.py::_resolve_ecp``) and Spectra
    (``spectra/pyscf_script.py::_emit_build_mol``) so the two generators
    cannot drift.

    Inputs:
      * ``ecp``        -- the ECP name, e.g. ``"lanl2dz"``.  Empty = no ECP.
      * ``ecp_atoms``  -- element patterns naming which atoms get it:
                          ``[]`` none · ``["*"]`` every element present ·
                          ``["Au"]`` that element · ``["A*"]`` every symbol
                          starting with A · ``["Au", "Pt"]`` several.

    Returns ``{element: ecp}`` for the elements actually present in
    ``struct`` that match, or ``None`` when nothing does -- which is the
    signal to omit the ``gto.M(ecp=...)`` kwarg entirely.

    **Empty means empty, and nothing is chosen for the user.**  Until
    2026-08-13 this function had three branches: ``""``/``"none"`` meant
    off, a str or dict passed through, and ``None`` meant *auto* -- add
    ``lanl2dz`` when any element had Z > 36 and the basis was not def2.
    Both halves of that heuristic were deleted on the user's ruling:
    *"there is no point to limit matching to heavy -- who defines heavy?
    there is no clear reasoning or standard ... explicit is better than
    implicit."*  ``basis`` left the signature with the def2 special case;
    a def2 basis brings its own ECP, and declaring another on top of it
    is now a visible choice rather than something silently suppressed.

    ``validation`` still HINTS when a structure looks like it wants an ECP
    and none is declared.  A hint is confirmed by a person; it is not this
    function quietly acting.
    """
    name = (ecp or "").strip()
    patterns = [p.strip() for p in (ecp_atoms or []) if p and p.strip()]
    if not name or not patterns:
        return None

    # Element symbols are canonically capitalised ("AU" / "au" -> "Au"), and
    # so are the patterns, so ``["au"]`` and ``["Au"]`` select the same atom
    # without either spelling being a second format to remember.
    present: List[str] = []
    for el in struct.elements:
        sym = str(el).capitalize()
        if sym not in present:
            present.append(sym)
    pats = [p.capitalize() for p in patterns]

    matched = {sym: name for sym in present
               if any(fnmatch.fnmatchcase(sym, p) for p in pats)}
    return matched or None


def detect_open_shell_metals(struct: Structure) -> List[str]:
    """Return the unique open-shell-metal element symbols present in
    ``struct``, in their first-appearance order.

    Capitalisation-insensitive: a PDB-loaded "FE" matches "Fe".  Use
    this in preflight to warn when a closed-shell singlet (spin=0,
    RKS/RHF) is requested for a molecule containing transition
    metals -- a common silent cause of unphysical forces / energies
    (see hemeC-dithiol 2026-05-22 incident).
    """
    seen = []
    seen_set: set = set()
    for el in struct.elements:
        key = el.capitalize()
        if key in OPEN_SHELL_METALS and key not in seen_set:
            seen.append(key)
            seen_set.add(key)
    return seen


# The full d-block (+ f-block) metal set for basis-adequacy checks: d-orbital
# coverage matters for CLOSED-shell metals (Zn/Cd/Hg d10, Pd/Pt) too, not only
# open-shell ones -- the concern is orbital coverage, orthogonal to spin state.
_ALL_TRANSITION_METALS = (OPEN_D_TRANSITION_METALS
                          | CLOSED_D10_METALS | NOBLE_METALS_S1)


def detect_transition_metals(struct: Structure) -> List[str]:
    """Every transition / f-block metal present (open- AND closed-shell), in
    first-appearance order.  For basis-adequacy: Zn/Cd/Hg/Pd/Pt need proper
    d/polarisation coverage even though ``detect_open_shell_metals`` skips
    them."""
    seen: List[str] = []
    seen_set: set = set()
    for el in struct.elements:
        key = el.capitalize()
        if key in _ALL_TRANSITION_METALS and key not in seen_set:
            seen.append(key)
            seen_set.add(key)
    return seen


# --------------------------------------------------------------------- #
#  L2 — engine-agnostic chemistry analyzer                              #
#                                                                       #
#  See docs/science/validation.md for the full         #
#  contract.  The analyzer wraps the L1 primitives above into a typed   #
#  ChemistryAnalysis the validators + the /api/structure/analyze        #
#  endpoint both consume — single source of truth for the chemistry-    #
#  driven (charge, spin, treatment) triplet plus open-shell-metal       #
#  hints.                                                               #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpinChoice:
    """One ranked spin candidate for an open-shell transition metal.

    ``spin`` is 2S (= number of unpaired electrons), matching PySCF's
    convention.  SIESTA's ``SpinTotal`` in μB equals ``float(spin)``.
    """
    spin:  int
    label: str   # e.g. "Fe(II), intermediate (4-coord porphyrin)"


@dataclass(frozen=True)
class MetalHint:
    """Per-element common-spin hints for an open-shell transition metal.

    ``common_spins`` are ordered low-spin → high-spin so a UI can show
    a sweep from "most-paired" to "most-unpaired" without sorting.
    """
    element:      str
    common_spins: List[SpinChoice]


# Per-element default 2S for open-shell transition metals — the
# analyzer's suggested defaults for ``/api/structure/analyze``'s
# Auto-detect button.
#
# Distinct (intentionally) from ``_SPIN_TOTAL_DEFAULTS`` above.  The
# two tables have different design goals and may carry different
# values for the same element:
#
#   * ``_SPIN_TOTAL_DEFAULTS`` is the SIESTA propor STARTING-VALUE
#     table.  When the user enables spin_polarized without setting
#     spin_total, propor needs a non-zero guess to split d/f shells.
#     Picks HIGH-SPIN-leaning values so the optimizer can ramp DOWN
#     to a lower-spin state if that's the true ground state —
#     ramping UP from zero is what triggered the "propor: ERROR:
#     IMAX = 0" abort.  Fe→4 (HS), Co→3 (HS), Ni→2 (HS) etc.
#
#   * ``_ANALYZER_DEFAULT_SPIN`` (this table) is the chemistry-
#     conservative default for the Auto-detect UI.  Picks the most
#     COMMON-OXIDATION-STATE spin — what a coordination chemist
#     would expect for "Fe in a porphyrin" or "Cu(II) in solution".
#     Fe→2 (intermediate, the hemeC use case), Co→1 (LS as a safe
#     pick), Ni→0 (square-planar LS) etc.  The user can always
#     override via the form; the suggestion is meant to MATCH what
#     they probably want, not what SIESTA can converge from.
#
# When the two tables AGREE (Mn=5, Cu=1), great — single chemistry
# fact, two purposes.  When they DISAGREE (Fe 4 vs 2, Co 3 vs 1),
# both are correct for their own purpose.  Don't unify; document.
#
# Conservative choices — favour the most common coordination
# chemistry; the user MUST verify against experimental data.  See
# docs/science/validation.md
_ANALYZER_DEFAULT_SPIN: Dict[str, int] = {
    "Fe": 2,    # Fe(II), intermediate-spin (S=1, 4-coord porphyrin —
                # the molbuilder hemeC use case).  HS Fe(II) is 4;
                # user can override.
    "Mn": 5,    # Mn(II), high-spin S=5/2 — overwhelming default in
                # bio/aqueous; LS Mn(II) is exceedingly rare.
    "Co": 3,    # Co(II), HIGH-spin S=3/2 — the COMMON octahedral /
                # aqueous / weak-field ligand case.  LS Co(II) (S=1/2)
                # requires strong-field ligands (CN⁻, phen, bipy);
                # user overrides to spin=1 for those.  Picking LS as
                # default biases toward textbook ideal at the cost of
                # the more common bio/coordination-chem reality.
    "Ni": 0,    # Ni(II), square-planar LS d⁸ S=0 — the common case
                # in metalloproteins + coordination chem.  Octahedral
                # HS (S=1) is less common; user overrides if needed.
    "Cu": 1,    # Cu(II), d⁹ S=1/2 — one unpaired electron, period.
                # No realistic alternative.
    "Cr": 3,    # Cr(III), d³ S=3/2 — the dominant oxidation state.
    "V":  3,    # V(II), d³ S=3/2 — common low-V oxidation state.
    "Ti": 2,    # Ti(II), d² S=1.
    "Sc": 1,    # Sc(II), d¹ S=1/2 — rare; Sc(III) (d⁰) is closed-shell
                # and wouldn't trigger this path.
    # Second-row + third-row + f-block fall through to a safe 2.
    # See ``_metal_hint`` for the full set of spin candidates a
    # user can pick via the Auto-detect panel's per-metal hints.
}


@dataclass(frozen=True)
class ChemistryAnalysis:
    """Engine-agnostic chemistry conclusions about a Structure.

    Single source of truth for every science-aware surface in the
    system: UI auto-detect (``/api/structure/analyze``), pre-emission
    validation (``validation.check_open_shell_metal``), future
    Transport-tab Auto-detect, CLI ``molbuilder analyze``.  Two
    surfaces consuming this dataclass cannot disagree about the
    chemistry by construction.

    All fields engine-agnostic.  Engine-specific translation lives
    in per-engine adapter classes (see protocols/scientific-validation.md
    § 4); adapters consume an instance of this class and emit a
    typed ``<Engine>SuggestedParams`` dataclass.
    """
    # Composition
    n_atoms:              int
    elements:             List[str]      # unique, sorted
    n_electrons_neutral:  int            # sum(Z) for neutral system

    # Open-shell transition metals
    metals:               List[str]      # ["Fe"], or [] for organics
    metal_hints:          List[MetalHint]

    # Engine-agnostic suggested defaults
    suggested_charge:     int
    suggested_spin:       int            # 2S = n_unpaired
    suggested_treatment:  Literal["closed", "open"]

    # Human-readable
    rationale:            str
    warnings:             List[str]


def _metal_hint(element: str) -> MetalHint:
    """Build a MetalHint by walking spin = 0..6 and collecting every
    spin for which ``explain_metal_spin`` registers a label."""
    spins: List[SpinChoice] = []
    for s in range(0, 7):
        label = explain_metal_spin(element, s)
        if label:
            spins.append(SpinChoice(spin=s, label=label))
    return MetalHint(element=element, common_spins=spins)


def _count_element(struct: Structure, symbol: str) -> int:
    """Number of atoms of ``symbol`` (case-insensitive) in struct."""
    sym = symbol.capitalize()
    return sum(1 for el in struct.elements if el.capitalize() == sym)


# Noble-metal cluster size at which the metallic-bonding closed-shell
# argument kicks in.  Below this size the per-atom open-shell state
# can still survive — small Au_n clusters (n=2..4) have magic-number
# physics where shell-closing is incomplete.  Above this size the
# 6s band delocalizes and the system is closed-shell singlet for
# even total electron count.  4 atoms is the conservative cutoff:
# overwhelmingly what published Au transport / surface DFT does;
# specialists working on Au_2 / Au_3 will override via the form.
_NOBLE_METAL_CLUSTER_THRESHOLD = 4


def analyze_structure(struct: Structure) -> ChemistryAnalysis:
    """Run the chemistry analysis on ``struct``.  Pure function — no
    I/O, no engine dependence, no global state.

    Returns a ``ChemistryAnalysis`` whose ``suggested_*`` fields
    carry chemistry-driven defaults; the rationale + warnings
    explain the choice.  Adapters translate these conclusions into
    each engine's parameter shape (see
    ``protocols/scientific-validation.md`` § 4).

    Spin policy (2026-06-13 — noble-metal-aware):

      1. **Open-d transition metal present** (Fe, Co, Ni, Mn, Cr, Ru,
         Rh, ...) → open-shell.  Spin from ``_ANALYZER_DEFAULT_SPIN``
         (Fe→2, Cu→1, ...), parity-corrected.

      2. **Noble metal only** (Cu / Ag / Au present, NO open-d metal):
         the metallic-bonding argument decides.  ≥ 4 atoms of the
         metal AND even electron count → closed-shell singlet
         (standard Au transport treatment per Taylor/Brandbyge/Stokbro
         PRB 63 (2001) 245407 + the Stoner-criterion-fails argument
         in Marder Ch. 17).  Single noble-metal atom with odd electron
         count → respect atomic open-shell state.  Other cases fall
         through to parity.

      3. **No open-shell metals** → ``treatment="closed"``, spin set
         by electron-count parity (0 if even, 1 if odd).
    """
    n_e = total_electrons(struct, 0)
    elements_sorted = sorted({el.capitalize() for el in struct.elements})
    metal_set = set(elements_sorted)

    # Categorize present metals.  Iterate the SORTED element list, NOT the
    # frozensets: a frozenset yields hash-order, which CPython randomizes per
    # process (PYTHONHASHSEED), so `open_d[0]` (whose default spin + rationale
    # get reported) would vary run-to-run for a multi-metal structure (e.g.
    # Fe+Cr).  A validator must be deterministic; drive the pick off the
    # structure's own sorted elements.
    open_d  = [m for m in elements_sorted if m in OPEN_D_TRANSITION_METALS]
    nobles  = [m for m in elements_sorted if m in NOBLE_METALS_S1]

    # Build metal_hints for the UI Auto-detect panel.  Includes both
    # categories — users still want to see hints for noble metals
    # ("if this IS a small cluster, here's the open-shell spin
    # you'd use").
    metal_hints = [_metal_hint(m) for m in (open_d + nobles)]

    warnings: List[str] = []
    suggested_charge = 0   # always 0 for v1 — overridable by user

    if open_d:
        # Path 1: open-d metal forces open-shell consideration.
        spin = _ANALYZER_DEFAULT_SPIN.get(open_d[0], 2)
        if (n_e % 2) != (spin % 2):
            old = spin
            spin = spin + 1 if spin == 0 else spin - 1
            warnings.append(
                f"Adjusted suggested spin from {old} to {spin} to match "
                f"electron-count parity (sum(Z)={n_e}, charge={suggested_charge})."
            )
        treatment: Literal["closed", "open"] = "open"
        first_label = explain_metal_spin(open_d[0], spin) or "?"
        # The list reported to the user includes any noble metals
        # too, so the rationale doesn't omit them.
        listed = ", ".join(open_d + nobles)
        rationale = (
            f"Detected open-shell d-block metal {listed}.  "
            f"Suggesting spin={spin} ({first_label}) with open-shell "
            f"treatment.  Verify against your experimental data "
            f"(Mössbauer / UV-Vis / EPR) — the right spin depends on "
            f"axial coordination, not just element identity."
        )
        metals_for_dataclass = open_d + nobles
    elif nobles:
        # Path 2: noble-metal-only system — cluster context decides.
        # Total atoms of all noble metal species combined; usually
        # a single species but a hypothetical Au/Ag alloy would still
        # be metallic at any reasonable size.
        n_noble_atoms = sum(_count_element(struct, m) for m in nobles)
        even_electrons = (n_e % 2 == 0)
        cluster_qualifies = n_noble_atoms >= _NOBLE_METAL_CLUSTER_THRESHOLD
        if cluster_qualifies and even_electrons:
            # Closed-shell singlet — the dominant case in published
            # Au junction / surface work.
            spin = 0
            treatment = "closed"
            rationale = (
                f"Detected metallic {', '.join(nobles)} system "
                f"({n_noble_atoms} atoms, even electron count). "
                f"Noble-metal clusters / surfaces / junctions are "
                f"conventionally treated as closed-shell singlet "
                f"(spin-restricted DFT) — the s-band delocalizes "
                f"and the Stoner criterion fails for Cu / Ag / Au, "
                f"so no spontaneous magnetism develops in bulk.  "
                f"Refs: Taylor, Brandbyge, Stokbro, PRB 63 (2001) "
                f"245407 (Au-BDT-Au TranSIESTA benchmark); Marder, "
                f"Condensed Matter Physics Ch. 17.  Override (set "
                f"spin > 0, switch to UKS/ROKS) if you're modelling "
                f"a sub-{_NOBLE_METAL_CLUSTER_THRESHOLD}-atom cluster, "
                f"a single noble-metal adatom on an insulator, a noble "
                f"metal with magnetic 3d co-adsorbate, or explicit "
                f"Kondo / spin-orbit physics."
            )
        elif n_noble_atoms == 1 and not even_electrons:
            # Single isolated noble-metal atom: respect the atomic
            # open-shell ground state (5d¹⁰ 6s¹ for Au, S=½).
            spin = 1
            treatment = "open"
            rationale = (
                f"Detected single {nobles[0]} atom in an "
                f"odd-electron system.  Noble-metal atomic ground "
                f"state is nd¹⁰ (n+1)s¹ — open-shell doublet.  "
                f"Suggesting spin=1 with open-shell treatment.  "
                f"(Cluster-context override does NOT apply at n=1; "
                f"that argument needs n ≥ "
                f"{_NOBLE_METAL_CLUSTER_THRESHOLD} for the s-band "
                f"to form.)"
            )
        else:
            # Ambiguous (2–3 atom cluster, or odd-electron count
            # with a multi-atom cluster).  Fall through to parity
            # but include a note pointing at the closed-shell default
            # if the user is in a junction context.
            spin = 0 if even_electrons else 1
            treatment = "open" if spin > 0 else "closed"
            rationale = (
                f"Detected small {nobles[0]} cluster "
                f"({n_noble_atoms} atom{'s' if n_noble_atoms != 1 else ''}). "
                f"At this size the noble-metal cluster-context closed-"
                f"shell argument doesn't cleanly apply (needs n ≥ "
                f"{_NOBLE_METAL_CLUSTER_THRESHOLD}).  Suggesting "
                f"electron-count parity: spin={spin}, treatment={treatment}."
            )
        metals_for_dataclass = nobles
    else:
        # Path 3: no transition metals at all — pure organic, light
        # main-group, or closed-d¹⁰ (Zn/Cd/Hg/Pd/Pt) systems.
        # Closed-shell singlet for even electron count.
        spin = 0 if (n_e % 2 == 0) else 1
        treatment = "open" if spin > 0 else "closed"
        rationale = (
            f"No open-shell metals detected; suggesting closed-shell "
            f"{'singlet' if spin == 0 else 'doublet'} "
            f"(spin={spin}, treatment={treatment})."
        )
        metals_for_dataclass = []

    # Preserve the legacy ``metals`` field shape: a flat list of
    # transition-metal symbols present in the structure.  Callers
    # downstream (validators, the UI's per-metal hint panel) iterate
    # this list; semantics unchanged for open-d metals, and now
    # includes noble metals when they're physically relevant.

    return ChemistryAnalysis(
        n_atoms             = struct.n_atoms,
        elements            = elements_sorted,
        n_electrons_neutral = n_e,
        metals              = metals_for_dataclass,
        metal_hints         = metal_hints,
        suggested_charge    = suggested_charge,
        suggested_spin      = spin,
        suggested_treatment = treatment,
        rationale           = rationale,
        warnings            = warnings,
    )


# --------------------------------------------------------------------- #
#  L3 — engine parameter adapter Protocol + registry                    #
#                                                                       #
#  See docs/science/validation.md for the full         #
#  contract.  Each engine module under molbuilder/<engine>/ exports an  #
#  adapter class that translates a ChemistryAnalysis into a typed,      #
#  engine-specific frozen dataclass.  Adapters register themselves on   #
#  import via the @register_adapter decorator; the /api/structure/     #
#  analyze endpoint iterates registered_adapters() to build the         #
#  ``suggested.<engine>`` block — new engines need no endpoint change.  #
# --------------------------------------------------------------------- #


class EngineParameterAdapter(Protocol):
    """Translate engine-agnostic ChemistryAnalysis conclusions into a
    typed, engine-specific parameter dataclass.

    Adapters live per-engine under ``molbuilder/<engine>/auto_defaults.py``
    and register themselves at import time via ``@register_adapter``.

    Design rules (see scientific-validation.md § 4.4):

    * PURE translator.  An adapter MUST NOT re-do chemistry detection,
      parity checks, or any other analysis.  All chemistry logic lives
      in ``analyze_structure``; adapters only translate.
    * TYPED dataclass output.  Returns a frozen dataclass (e.g.
      ``SiestaSuggestedParams``), not a dict.  The HTTP boundary
      serialises via ``dataclasses.asdict``.
    * Field names match the engine's web-form / Config dataclass.
      The UI's "apply suggestion" path just spreads the dataclass
      into form values.
    """

    name: str   # registry key, e.g. "siesta", "pyscf"

    @classmethod
    def to_params(cls, analysis: "ChemistryAnalysis") -> Any:
        """Return an engine-specific frozen dataclass carrying the
        suggested defaults for this engine.  Always includes a
        ``rationale`` field; MAY include engine-specific notes.
        """
        ...


_ADAPTERS: Dict[str, Type[EngineParameterAdapter]] = {}


def register_adapter(name: str):
    """Decorator: register an adapter class under the given engine name.

    Usage::

        @register_adapter("siesta")
        class SiestaAdapter:
            name = "siesta"
            @classmethod
            def to_params(cls, analysis):
                return SiestaSuggestedParams(...)

    Imports of decorated classes have a side effect (registry
    mutation).  The canonical place to ensure adapters get imported
    at web-app startup is ``molbuilder/web/blueprints/__init__.py``;
    direct callers (CLI, tests) import the adapter module explicitly.
    """
    def deco(cls: Type[EngineParameterAdapter]) -> Type[EngineParameterAdapter]:
        _ADAPTERS[name] = cls
        return cls
    return deco


def registered_adapters() -> Dict[str, Type[EngineParameterAdapter]]:
    """Return a defensive copy of the current adapter registry.

    Callers (notably ``/api/structure/analyze``) iterate the returned
    dict.  Copying prevents a stray ``del`` or ``.clear()`` at a
    consumer from poisoning the registry for the rest of the process.
    """
    return dict(_ADAPTERS)


def _clear_adapters_for_test() -> None:
    """Test-only: empty the registry.  Used by tests that want a
    clean slate to verify the new-engine on-ramp; production code
    must never call this.
    """
    _ADAPTERS.clear()


def expected_pH7_peptide_charge(struct: Structure) -> Optional[int]:
    """Estimate net peptide charge at physiological pH (7.4).

    Returns
    -------
    int    Estimated charge from charged side chains.  Asp/Glu
           contribute -1 each; Lys/Arg contribute +1 each.  N- and
           C-termini cancel each other for a free peptide.  His /
           Cys / Tyr are intentionally skipped (His ambiguous; the
           others have pKa > 8).
    None   The structure doesn't look like a peptide (no recognised
           amino-acid residue names).  For nucleic acids use
           ``formal_charge_from_phosphates`` instead.

    Used by validators to surface the gap between
    ``cfg.charge = 0`` (default neutral build) and the physiological
    charge state the user often actually wants.  Never raises; never
    silently mutates the input.
    """
    if struct.residue_names is None or struct.residue_ids is None:
        return None
    # Collect one residue-name per residue id (atoms in the same
    # residue contribute to the count once, not per atom).
    seen: dict[int, str] = {}
    for rid, rname in zip(struct.residue_ids, struct.residue_names):
        if rid not in seen:
            seen[rid] = rname
    # Confirm we're looking at a peptide: at least one standard AA
    # residue name must appear.
    aa_present = sum(1 for n in seen.values()
                     if n in _AMINO_ACID_RESIDUE_NAMES)
    if aa_present == 0:
        return None
    # Sum the side-chain contributions.
    charge = 0
    for name in seen.values():
        charge += _CHARGED_RESIDUES_PH7.get(name, 0)
    return charge


# Bond cutoffs used for proximity-based adjacency.  Wide enough to
# catch slightly-distorted bonds, narrow enough not to misclassify
# 1,3 contacts as bonds.
_HX_CUT = 1.30   # X-H bond cutoff
_XX_CUT = 1.95   # heavy-heavy bond cutoff


def _adjacency(elements: List[str], positions: np.ndarray
               ) -> Tuple[List[List[int]], List[List[int]]]:
    """Return per-atom heavy-neighbour and H-neighbour lists."""
    n = len(elements)
    nb_heavy: List[List[int]] = [[] for _ in range(n)]
    nb_h:     List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        ei = elements[i]
        for j in range(i + 1, n):
            ej = elements[j]
            d = float(np.linalg.norm(positions[i] - positions[j]))
            cutoff = _HX_CUT if ("H" in (ei, ej)) else _XX_CUT
            if d > cutoff:
                continue
            if ej == "H": nb_h[i].append(j)
            else:         nb_heavy[i].append(j)
            if ei == "H": nb_h[j].append(i)
            else:         nb_heavy[j].append(i)
    return nb_heavy, nb_h


def formal_charge_from_phosphates(struct: Structure) -> int:
    """Estimate formal charge from phosphate protonation state.

    For each P atom:
        n_nb  = number of non-bridging O neighbours
        n_h   = number of those Os that already carry an H
        Of the n_nb non-bridging Os, one is the implicit P=O (no H);
        the remaining (n_nb - 1) should be -OH for a neutral phosphate.
        Missing Hs there each contribute -1 to the molecular charge:
            charge_contribution_per_P = -max(0, n_nb - 1 - n_h)

    Returns the sum across all P atoms.  Returns 0 for structures with
    no phosphates.
    """
    elements  = struct.elements
    positions = np.asarray(struct.positions, dtype=float)
    nb_heavy, nb_h = _adjacency(elements, positions)

    charge = 0
    for i, el in enumerate(elements):
        if el != "P":
            continue
        # Non-bridging O = O whose only heavy neighbour is this P
        non_bridging = [
            j for j in nb_heavy[i]
            if elements[j] == "O" and len(nb_heavy[j]) == 1
        ]
        if len(non_bridging) < 1:
            continue
        n_nb = len(non_bridging)
        n_h  = sum(1 for j in non_bridging if len(nb_h[j]) > 0)
        # Implicit P=O on the first one; (n_nb - 1) should be OH
        missing = max(0, (n_nb - 1) - n_h)
        charge -= missing
    return charge


def resolve_net_charge(struct: Structure,
                      explicit_charge: Optional[int]) -> int:
    """Resolve a molecule's net charge from an optional explicit override.

    The rule lives here so the SIESTA and PySCF generators (which
    name their dataclass fields differently -- ``cfg.net_charge`` vs
    ``cfg.charge``) don't each carry their own copy:

      1. Explicit override wins.  ``0`` is meaningful (forces neutral,
         disables auto-detection); only ``None`` triggers the
         auto-detect path.
      2. Otherwise, count the deprotonated phosphate non-bridging
         oxygens via :func:`formal_charge_from_phosphates`.

    The heuristic only sees phosphate groups; charged side chains
    (Asp / Glu / Lys / Arg / His) are NOT detected -- the user
    must override with a non-None explicit value for those.
    """
    if explicit_charge is not None:
        return int(explicit_charge)
    return formal_charge_from_phosphates(struct)


def protonate_phosphate_oxygens(struct: Structure) -> Tuple[Structure, int]:
    """Neutralise the molecule by adding Hs to deprotonated phosphate Os.

    Per P atom: leaves one non-bridging O alone (the implicit P=O) and
    adds an H to every other non-bridging O that doesn't already have
    one.

    H placement: 0.96 A from O at 109.47 deg from the P-O direction,
    rotated in the plane formed by P-O and the "outward" direction
    (away from the centroid of the rest of the molecule).  This puts
    the new -OH cleanly on the outside of the structure, matching what
    a force-field optimisation would settle on.

    Returns (new_structure, n_hs_added).  If no protonation is needed
    the original Structure is returned unchanged (n_hs_added = 0).
    """
    elements      = list(struct.elements)
    positions     = np.asarray(struct.positions, dtype=float).copy()
    atom_names    = list(struct.atom_names)
    residue_ids   = list(struct.residue_ids)
    residue_names = list(struct.residue_names)
    chain_ids     = list(struct.chain_ids)

    if not elements:
        return struct, 0

    nb_heavy, nb_h = _adjacency(elements, positions)

    BOND_OH = 0.96
    COS_SP3 = -1.0 / 3.0
    SIN_SP3 = math.sqrt(8.0) / 3.0    # = sin(109.47 deg)

    new_atoms: List[dict] = []

    for p_idx, el in enumerate(elements):
        if el != "P":
            continue
        non_bridging = [
            j for j in nb_heavy[p_idx]
            if elements[j] == "O" and len(nb_heavy[j]) == 1
        ]
        if len(non_bridging) < 2:
            continue   # nothing to do (need >= 1 P=O + >= 1 OH)
        # Choosing the implicit P=O:
        #   * An O that already carries an H must stay as -OH; protonating
        #     it would over-saturate the phosphate.
        #   * Among the bare Os we sort by atom name for deterministic
        #     output across runs.
        # So: take the bare-O list, sorted; the first is P=O, the rest
        # become -OH.  Os that already have H are never re-touched.
        bare = sorted([j for j in non_bridging if len(nb_h[j]) == 0],
                      key=lambda j: atom_names[j])
        if not bare:
            continue   # everything already protonated -- nothing to add
        # Reserve the first bare O as the implicit P=O and protonate the rest.
        targets = bare[1:]
        if not targets:
            continue
        for o_idx in targets:
            o_pos = positions[o_idx]
            p_pos = positions[p_idx]

            u_op = p_pos - o_pos
            n_op = float(np.linalg.norm(u_op))
            if n_op < 1e-9:
                continue
            u_op_norm = u_op / n_op

            # Outward direction = (this O) minus (centroid of P's other
            # heavy neighbours).  Project away from u_op to keep the
            # tilt-plane perpendicular to the P-O bond.
            other = [k for k in nb_heavy[p_idx] if k != o_idx]
            if other:
                centroid = positions[other].mean(axis=0)
                outward  = o_pos - centroid
                perp     = outward - np.dot(outward, u_op_norm) * u_op_norm
            else:
                perp = np.zeros(3)

            n_perp = float(np.linalg.norm(perp))
            if n_perp < 1e-6:
                # Pick any unit vector perpendicular to u_op_norm.
                seed = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(seed, u_op_norm)) > 0.9:
                    seed = np.array([0.0, 1.0, 0.0])
                perp = seed - np.dot(seed, u_op_norm) * u_op_norm
                n_perp = float(np.linalg.norm(perp))
            perp_norm = perp / n_perp

            h_pos = o_pos + BOND_OH * (COS_SP3 * u_op_norm + SIN_SP3 * perp_norm)

            # H name -- match common PDB conventions: HOP1, HOP2, ...
            h_name = ("H" + atom_names[o_idx])[:4]
            new_atoms.append({
                "element":  "H",
                "position": h_pos,
                "name":     h_name,
                "res_id":   residue_ids[o_idx],
                "res_name": residue_names[o_idx],
                "chain_id": chain_ids[o_idx],
            })

    if not new_atoms:
        return struct, 0

    new_elements      = elements + [a["element"] for a in new_atoms]
    new_positions     = np.vstack([positions,
                                   np.array([a["position"] for a in new_atoms])])
    new_atom_names    = atom_names    + [a["name"]     for a in new_atoms]
    new_residue_ids   = residue_ids   + [a["res_id"]   for a in new_atoms]
    new_residue_names = residue_names + [a["res_name"] for a in new_atoms]
    new_chain_ids     = chain_ids     + [a["chain_id"] for a in new_atoms]

    return (
        Structure(
            elements      = new_elements,
            positions     = new_positions,
            atom_names    = new_atom_names,
            residue_ids   = new_residue_ids,
            residue_names = new_residue_names,
            chain_ids     = new_chain_ids,
            title         = struct.title,
        ),
        len(new_atoms),
    )


# --------------------------------------------------------------------- #
#  Hydrogen addition: tool comparison + design rationale                #
#  ===================================================                  #
#                                                                       #
#  Single source of truth for adding explicit H atoms to a heavy-atom   #
#  skeleton.  Used by:                                                  #
#    - peptide.build_peptide        (PeptideBuilder emits heavy-only)   #
#    - nucleic.build_dna/build_rna  (X3DNA's `fiber` is heavy-only;     #
#                                    amber/rdkit produce H themselves   #
#                                    and skip this via the H/heavy>=0.3 #
#                                    gate in nucleic._maybe_add_hydrogens)
#                                                                       #
#  Why two engines, in this order                                       #
#  -------------------------------                                      #
#  Both OpenBabel and RDKit place H along correct sp3-tetrahedral /     #
#  sp2-planar / sp-linear vectors based on perceived bond orders.       #
#  They differ in how they handle ambiguous-valence sites (typically    #
#  exocyclic NH2 amines on nucleic acid bases and -NH3+ at peptide      #
#  N-termini):                                                          #
#                                                                       #
#  OpenBabel `OBMol.AddHydrogens()` (preferred):                        #
#    - Geometric H placement directly from each parent's hybridization  #
#      and existing neighbours.  No "give up" failure mode.             #
#    - On standard biomolecules (DA/DT/DG/DC, 20 amino acids) the       #
#      residue-template chemistry is mature and battle-tested.          #
#    - Doesn't reorder atoms, so PDB indices are preserved.             #
#    - Verified: X3DNA fiber heavy-skeleton -> AddHydrogens produces    #
#      the canonical 5 O-H + 37 C-H + 8 N-H breakdown for ATGC,         #
#      matching Amber-tleap and RDKit-via-SMILES exactly.               #
#                                                                       #
#  RDKit `Chem.AddHs(mol, addCoords=True)` (fallback):                  #
#    - Bond-order perception from PDB residue templates is correct      #
#      (this is well-tested).                                           #
#    - BUT for sites where the heavy-atom geometry doesn't constrain    #
#      H placement uniquely -- exocyclic -NH2 on bases (A.N6, G.N2,     #
#      C.N4), peptide N-terminal -NH3+ -- the addCoords=True flag       #
#      sometimes leaves H atoms AT THEIR PARENT'S COORDINATES (zero    #
#      distance "ghost H").  This is a known RDKit limitation when     #
#      placing H from a heavy-atom-only PDB.                            #
#    - For DNA bases this strips Watson-Crick H-bond donors (4 H short  #
#      out of 50 on an ATGC chain pre-OpenBabel -- the bug that         #
#      motivated the fallback ordering).                                #
#    - SMILES-construct path doesn't have this issue; only PDB-parse    #
#      then AddHs has it.  build_peptide and the rdkit nucleic backend  #
#      reach the SMILES path; the X3DNA path lands here.                #
#                                                                       #
#  Why not AmberTools `reduce`                                          #
#    - It's the gold standard for protein protonation (His tautomers,   #
#      Asn/Gln flips), but for DNA it's not better than OpenBabel and   #
#      requires shelling out + a temp-file round trip.  We already      #
#      have AmberTools as a transitive dep for the amber-tleap          #
#      backend; using `reduce` here wouldn't add a new dep but would    #
#      make this code path harder to reason about (subprocess vs        #
#      in-process).  Sticking with OpenBabel keeps the H-placement      #
#      logic uniform across peptide and nucleic flows.                  #
#                                                                       #
#  Why _drop_overlapping_hydrogens after each engine                    #
#    - Both engines (different reasons) can produce H at zero distance  #
#      from an anchor.  RDKit: addCoords ghost-H artifact above.        #
#      OpenBabel: rare, but multiple H written at the same coord for    #
#      tautomeric or ill-defined sites.  Keeping the post-pass means    #
#      the caller never sees the broken structure; downstream           #
#      validators don't have to special-case zero-distance pairs.       #
#    - Trade-off: for the RDKit path on nucleic acid bases, the drop    #
#      ALSO removes the legitimate-but-poorly-placed Watson-Crick H,    #
#      which is why OpenBabel is preferred-first.  Re-PLACING the       #
#      ghost H (rather than dropping) would be a smarter remediation    #
#      but is substantial new code and unnecessary as long as           #
#      OpenBabel is the primary engine.                                 #
# --------------------------------------------------------------------- #


def add_hydrogens(struct: Structure) -> Structure:
    """Add explicit H atoms to ``struct`` with correct sp3 / sp2 / sp
    geometry.

    Detection chain (first installed engine wins):
      1. OpenBabel ``OBMol.AddHydrogens()`` -- preferred.  Geometric H
         placement; doesn't fail on ambiguous-valence amine sites.
      2. RDKit ``Chem.AddHs(mol, addCoords=True)`` -- fallback.  Works
         well for SMILES-constructed molecules; for PDB-parsed inputs
         (heavy-atom only) it can leave exocyclic -NH2 H at parent
         coordinates.  See module-header comment for the full caveat.
      3. Neither: emit a RuntimeWarning and return the heavy-atom-only
         structure.  Callers should surface the warning since DFT will
         compute the wrong electron count.

    Both engines emit a final pass through ``_drop_overlapping_hydrogens``
    to strip any H that ended up sitting on another atom (the addCoords
    ghost-H artifact and rare OpenBabel duplicates).
    """
    # ---- try OpenBabel first (no ghost-coord failure mode) ----------
    try:
        from openbabel import openbabel as ob
    except ImportError:
        ob = None

    if ob is not None:
        return _protonate_openbabel(struct, ob)

    # ---- fall back to RDKit ------------------------------------------
    try:
        from rdkit import Chem
    except ImportError:
        Chem = None  # type: ignore

    if Chem is not None:
        return _protonate_rdkit(struct, Chem)

    import warnings
    warnings.warn(
        "Cannot add hydrogens: neither OpenBabel (`conda install -c "
        "conda-forge openbabel`) nor RDKit (`conda install -c conda-forge "
        "rdkit`) is installed.  Returning a HEAVY-ATOM-ONLY structure -- "
        "DFT will compute the wrong electron count.  Install OpenBabel "
        "for canonical biomolecule protonation; RDKit also works for "
        "SMILES-constructed inputs but has a known ambiguous-valence "
        "ghost-coord artifact for PDB-parsed nucleic-acid bases.",
        RuntimeWarning, stacklevel=3,
    )
    return struct


def _protonate_openbabel(struct: Structure, ob) -> Structure:
    """Geometric H placement via OBMol.AddHydrogens().

    OpenBabel's path: PDB -> OBMol (with bond perception from residue
    templates) -> AddHydrogens (geometric placement using sp3/sp2/sp
    vectors and existing neighbours) -> PDB.  Round-trip preserves
    atom order; placement is robust on ambiguous-valence amines that
    bite RDKit's PDB-then-AddHs path.
    """
    obconv = ob.OBConversion()
    obconv.SetInAndOutFormats("pdb", "pdb")
    mol = ob.OBMol()
    obconv.ReadString(mol, struct.to_pdb())
    mol.AddHydrogens()
    out = obconv.WriteString(mol)
    return _drop_overlapping_hydrogens(
        Structure.from_pdb(out, title=struct.title)
    )


def _protonate_rdkit(struct: Structure, Chem) -> Structure:
    """Fallback H placement via Chem.AddHs(mol, addCoords=True).

    Caveat (see module-header comment): for PDB-parsed inputs with
    ambiguous-valence sites (exocyclic -NH2 on bases, peptide
    N-terminal -NH3+), AddHs(addCoords=True) can leave H at parent
    coordinates.  ``_drop_overlapping_hydrogens`` removes those
    ghosts -- which is correct for peptides (the dropped H were
    extras, not load-bearing) but loses 4 Watson-Crick H on a typical
    DNA chain.  Use OpenBabel-first ordering to avoid landing here
    for nucleic-acid inputs.
    """
    mol = Chem.MolFromPDBBlock(struct.to_pdb(), removeHs=False, sanitize=False)
    if mol is None:
        # RDKit can choke on partial / unusual PDBs -- return as-is.
        import warnings
        warnings.warn(
            "RDKit failed to parse the heavy-atom PDB; returning "
            "heavy-atom-only structure.  Try installing OpenBabel.",
            RuntimeWarning, stacklevel=3,
        )
        return struct
    mol = Chem.AddHs(mol, addCoords=True)
    pdb_out = Chem.MolToPDBBlock(mol)
    return _drop_overlapping_hydrogens(
        Structure.from_pdb(pdb_out, title=struct.title)
    )


def _drop_overlapping_hydrogens(struct: Structure) -> Structure:
    """Remove H atoms that overlap (< 0.05 Å) with any other atom.

    Why 0.05 Å: a real X-H bond is always > 0.9 Å (the shortest
    physical X-H bond, H-F, is ~0.92 Å; C-H ~1.1 Å; N-H/O-H ~1.0 Å).
    A H within 0.05 Å of another atom is unambiguously a placement
    artifact -- typically a "ghost H" written at its parent atom's
    coordinates because the engine couldn't compute a real position.

    What this catches:
      * RDKit ``AddHs(addCoords=True)`` ghost H at ambiguous-valence
        sites (exocyclic -NH2 on nucleic-acid bases, peptide N-terminal
        -NH3+ extras).  The DEFINING failure mode of the RDKit fallback
        path; OpenBabel doesn't produce these.
      * Rare OpenBabel duplicates at tautomeric sites.

    What this does NOT do (and why):
      * Re-PLACE the ghost H at a sensible position.  That's the
        smarter remediation, but it requires hybridization perception
        (already in `_adjacency`) plus open-valence vector computation
        (new code).  Worth doing only if RDKit becomes the primary
        engine; with OpenBabel preferred, the drop is a safety net,
        not a load-bearing path.
      * Touch heavy atoms.  Only H-element atoms are candidates for
        removal; a heavy atom < 0.05 Å from another heavy atom is a
        broken structure that the validator should error on, not
        something we silently fix.

    Heavy atoms are never removed.

    H-H ghost pair handling: when two H atoms land at identical
    coordinates (rare, but possible for tautomer-ambiguous sites),
    a naive symmetric pass would mark BOTH as overlapping and drop
    them both -- removing real protons.  We track ``already_dropped``
    so once an H is flagged, it can't cause its peer to be flagged
    too.  Net effect on an H-H ghost pair: drop one, keep the other.
    """
    pos      = struct.positions
    elements = struct.elements
    n        = len(pos)
    keep     = np.ones(n, dtype=bool)
    for i in range(n):
        if elements[i] != "H" or not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if float(np.linalg.norm(pos[i] - pos[j])) < 0.05:
                keep[i] = False
                break
    if keep.all():
        return struct
    return Structure(
        elements      = [e for k, e in zip(keep, elements)             if k],
        positions     = pos[keep],
        atom_names    = ([a for k, a in zip(keep, struct.atom_names)    if k]
                         if struct.atom_names    is not None else None),
        residue_ids   = ([r for k, r in zip(keep, struct.residue_ids)   if k]
                         if struct.residue_ids   is not None else None),
        residue_names = ([n for k, n in zip(keep, struct.residue_names) if k]
                         if struct.residue_names is not None else None),
        chain_ids     = ([c for k, c in zip(keep, struct.chain_ids)     if k]
                         if struct.chain_ids     is not None else None),
        title         = struct.title,
    )


# --------------------------------------------------------------------- #
#  Steric-clash detection + relief                                      #
#                                                                        #
#  Building a duplex with a non-Watson-Crick base pair (a mismatch)     #
#  places the two bases at the standard WC frame, where they can        #
#  interpenetrate -- a topologically correct but physically unusable    #
#  starting structure (huge forces / SCF trouble if relaxed as-is).     #
#  `min_nonbonded_contact` DETECTS the overlap; `relieve_clashes`       #
#  optionally REMOVES it with a short force-field minimization.         #
# --------------------------------------------------------------------- #


def min_nonbonded_contact(struct: Structure, search_radius: float = 2.5):
    """Closest approach between atoms in DIFFERENT residues -- a steric-clash probe.

    Returns ``(distance, i, j)`` for the closest inter-residue atom pair within
    ``search_radius`` Angstrom, or ``(None, None, None)`` when the structure has no
    residue labels or no inter-residue pair inside the radius.

    Inter-residue only: within a residue, close contacts are covalent bonds (not
    clashes).  Between residues in DNA the only covalent link is the O3'-P
    backbone (~1.6 A), so any inter-residue pair well below that is an overlap --
    e.g. a mismatched base pair whose two bases interpenetrate.
    """
    if struct.residue_ids is None:
        return (None, None, None)
    P = np.asarray(struct.positions, dtype=float)
    if P.shape[0] < 2:
        return (None, None, None)
    chains = struct.chain_ids if struct.chain_ids is not None else [None] * len(P)
    keys = list(zip(chains, struct.residue_ids))
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    best = (None, None, None)
    for i, j in tree.query_pairs(r=search_radius):
        if keys[i] == keys[j]:
            continue
        d = float(np.linalg.norm(P[i] - P[j]))
        if best[0] is None or d < best[0]:
            best = (d, int(i), int(j))
    return best


def relieve_clashes(struct: Structure, steps: int = 1000) -> Structure:
    """Push apart steric overlaps with a short OpenBabel UFF minimization.

    Returns a new Structure (same atom order + count).  The GOAL is not an ideal
    geometry -- a subsequent DFT/SIESTA relaxation does that -- but to remove
    NEAR-COINCIDENT atoms that would otherwise make the first SCF step explode.

    Two robustness measures make this GENERAL (any sequence / mismatch), not tuned
    to one pair:

      1. **Bond cleanup.**  Our PDB carries no CONECT, so OpenBabel perceives bonds
         from geometry -- and perceives an OVERLAPPING pair as BONDED, which would
         pin it together and defeat the minimization.  We first DELETE every
         perceived inter-residue bond except the real O3'-P backbone link, so UFF
         sees the clashing atoms as non-bonded and repels them.  (This is also why
         the result is safe to hand downstream: no atoms left "wrongly bonded".)
      2. **Steepest-descent warm-up** then conjugate gradients: SD clears the hard
         short-range overlaps that CG alone can get stuck on.

    Raises RuntimeError if OpenBabel isn't installed / the force field won't set
    up -- the caller (build_dna) surfaces that rather than silently no-op'ing.
    """
    try:
        from openbabel import openbabel as ob
    except ImportError as exc:      # pragma: no cover - env-dependent
        raise RuntimeError(
            "relax_clashes needs OpenBabel (`conda install -c conda-forge "
            "openbabel`).") from exc

    conv = ob.OBConversion()
    conv.SetInAndOutFormats("pdb", "pdb")
    mol = ob.OBMol()
    conv.ReadString(mol, struct.to_pdb())

    def _res_key(atom):
        r = atom.GetResidue()
        return (r.GetChain(), r.GetNum()) if r else None

    def _atom_name(atom):
        r = atom.GetResidue()
        return r.GetAtomID(atom).strip() if r else ""

    # Drop perceived inter-residue bonds except the O3'-P backbone linkage: the
    # mis-perceived clash "bond" is what pins overlapping atoms together.
    doomed = []
    for bond in ob.OBMolBondIter(mol):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if _res_key(a1) != _res_key(a2):
            names = {_atom_name(a1), _atom_name(a2)}
            if names not in ({"O3'", "P"}, {"O3*", "P"}):
                doomed.append(bond)
    for bond in doomed:
        mol.DeleteBond(bond)

    ff = ob.OBForceField.FindForceField("UFF")
    if ff is None or not ff.Setup(mol):
        raise RuntimeError(
            "OpenBabel UFF force field could not be set up for clash relief.")
    ff.SteepestDescent(max(1, steps // 2))   # clear hard short-range overlaps
    ff.ConjugateGradients(steps)             # then settle
    ff.GetCoordinates(mol)
    return Structure.from_pdb(conv.WriteString(mol), title=struct.title)


# --------------------------------------------------------------------- #
#  Pauling electronegativity table + heuristic partial-charge estimate #
#                                                                        #
#  This is a heuristic, not a QM result.  Used by the validation pass  #
#  to estimate molecular dipoles for the "polar molecule in vacuum"    #
#  warning -- the goal is "is this molecule meaningfully polar?",     #
#  not a research-grade dipole.                                        #
# --------------------------------------------------------------------- #


# Pauling electronegativities for elements common in molbuilder targets.
# Source: standard chemistry references (e.g. Cotton & Wilkinson).
# Elements not in the table fall back to 2.20 (carbon-ish) -- the
# heuristic is forgiving about exact values; what matters is that
# polar bonds (>0.4 difference) tilt charge in the right direction.
_PAULING_EN = {
    "H":  2.20, "Li": 0.98, "Be": 1.57, "B":  2.04, "C":  2.55,
    "N":  3.04, "O":  3.44, "F":  3.98, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P":  2.19, "S":  2.58, "Cl": 3.16,
    "K":  0.82, "Ca": 1.00, "Br": 2.96, "I":  2.66,
}
_DEFAULT_EN = 2.20


def estimate_partial_charges(struct: Structure,
                             total_charge: float = 0.0,
                             *,
                             bond_cutoff: float = 1.95,
                             hx_cutoff:   float = 1.30) -> np.ndarray:
    """Heuristic per-atom partial charges from electronegativity gaps.

    For each bonded pair (heavy-heavy if d < bond_cutoff Å, X-H if
    d < hx_cutoff Å), Pauling's ionic-character formula gives the
    fractional charge transfer:

        ionic_fraction = 1 - exp(-0.25 * (Δχ)²)

    where Δχ is the Pauling EN difference.  The more electronegative
    atom receives a partial charge of -ionic_fraction; its partner
    receives +ionic_fraction.  Each per-bond shift is then capped at
    ±0.5 e to avoid outsize numbers on extreme pairs (e.g. F-Na).

    The result is shifted uniformly so that ``sum(q) == total_charge``,
    absorbing rounding error from any missed bonds at the edges.

    Cross-checks (Pauling formula; agreement within ~10% of reality):
        H2O    -> 1.8 D   (vs 1.85 D experimental)
        HF     -> 2.4 D   (vs 1.83 D)
        N2     -> 0.0 D   (vs 0)
        CO2    -> 0.0 D   (vs 0)
        CH3OH  -> 1.5 D   (vs 1.69 D)

    Not a substitute for QM partial charges.  Used by validation.py
    for the "polar molecule in vacuum" dipole warning, where the
    question is "is the dipole 0.5 D or 5 D?", not precise-to-decimal.
    """
    n         = struct.n_atoms
    elements  = list(struct.elements)
    positions = struct.positions
    q         = np.zeros(n, dtype=float)

    for i in range(n):
        ei   = elements[i]
        en_i = _PAULING_EN.get(ei, _DEFAULT_EN)
        for j in range(i + 1, n):
            ej = elements[j]
            d  = float(np.linalg.norm(positions[i] - positions[j]))
            if "H" in (ei, ej):
                if d > hx_cutoff:
                    continue
            elif d > bond_cutoff:
                continue
            en_j      = _PAULING_EN.get(ej, _DEFAULT_EN)
            delta_en  = en_i - en_j     # positive when i is more EN
            ionic     = 1.0 - math.exp(-0.25 * delta_en * delta_en)
            # Sign: more-EN atom gets negative.  ionic is always >= 0;
            # apply the sign of delta_en so i (more EN) ends up
            # negative when delta_en > 0.
            shift     = ionic if delta_en > 0 else -ionic
            shift     = max(-0.5, min(0.5, shift))
            q[i] -= shift          # more-EN atom: more negative
            q[j] += shift          # less-EN atom: more positive

    excess = q.sum() - total_charge
    if n > 0:
        q -= excess / n
    return q


def estimate_dipole_moment_debye(struct: Structure,
                                 total_charge: float = 0.0) -> float:
    """Magnitude of the heuristic molecular dipole, in Debye.

    Uses :func:`estimate_partial_charges` and computes
    ``|sum(q_i * r_i)|`` with positions in Å and charges in
    elementary units, converting via 1 e·Å = 4.80320 D.
    """
    if struct.n_atoms == 0:
        return 0.0
    q = estimate_partial_charges(struct, total_charge)
    p = (q[:, None] * struct.positions).sum(axis=0)   # e·Å
    return float(np.linalg.norm(p) * 4.80320)         # -> Debye
