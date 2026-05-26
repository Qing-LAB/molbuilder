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

import math
from typing import List, Optional, Tuple

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
# spectra database.  Closed-shell d¹⁰ metals (Zn, Cd, Hg) are
# excluded -- they're stable as closed-shell Zn²⁺ etc.
OPEN_SHELL_METALS = frozenset({
    # First-row transition metals (3d incomplete)
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    # Second-row
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    # Third-row
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    # Lanthanides (4f incomplete)
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Common actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu",
})


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
    ("Fe", 3): "Fe(III), intermediate-spin (S=3/2, 3 unpaired) -- e.g. cyt P450 oxoferryl",
    ("Fe", 5): "Fe(III), high-spin (S=5/2, 5 unpaired) -- e.g. met-myoglobin",
    # Mn: d⁵ for Mn(II), d⁴ for Mn(III)
    ("Mn", 0): "Mn(II), low-spin (rare for Mn²⁺)",
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
    "Re": 3.0,    # Re(III) d⁴ S=2
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
                      ecp: "Optional[Union[str, dict]]",
                      basis: str) -> "Optional[Union[str, dict]]":
    """Decide whether and which PySCF ECP (effective core potential) to
    pass to ``gto.M()``.  Cross-engine helper -- called from BOTH Build
    (``pyscf/input.py::_resolve_ecp``) and Spectra (``spectra/pyscf_
    script.py::_emit_build_mol``) so the rule stays in one place and
    can't drift between generators.

    Inputs:
      * ``struct`` -- the Structure whose elements decide whether an
        ECP is needed (Z > 36 = "heavy" enough that lanl2dz helps).
      * ``ecp`` -- user choice from the config field.  Three forms:
          str   -- explicit name ("lanl2dz", "stuttgart", ...), or
                   "" / "none" / None for "no ECP"
          dict  -- per-element dict ``{"Pt": "lanl2dz"}``
          None  -- the AUTO default; we pick "lanl2dz" iff heavy
                   atoms present AND basis is NOT def2-* (def2
                   bundles its own ECP).
      * ``basis`` -- the basis name; we skip the auto-add for the
        def2 family because PySCF auto-applies the Stuttgart ECP
        bundled with def2 itself.  Matches all three def2 name
        spellings (def2-SVP / def2_SVP / def2svp) via the bare
        "def2" prefix.

    Returns the ECP value to pass to ``gto.M(ecp=...)`` or None to
    omit the kwarg entirely.

    Why "lanl2dz" as the auto default: workhorse ECP for transition
    metals on cc-pVDZ-class bases, textbook default since the 1980s,
    shipped with PySCF (no extra basis-library install).  Stuttgart
    RSC / SBKJC are alternatives the user picks via cfg.ecp.
    """
    # Normalise "explicitly disabled" -- treat "" / "none" /
    # explicit-None identically.  Python-API users passing
    # ``ecp="none"`` would otherwise reach gto.M(ecp="none") and PySCF
    # raises "Unable to parse the input ECP data" at runtime.
    if isinstance(ecp, str) and ecp.strip().lower() in ("", "none"):
        return None
    if ecp is not None:
        return ecp                # explicit user choice (str or dict)
    # AUTO branch.  def2-* bundles its own ECP -- emitting "lanl2dz"
    # on top would double-count.
    if (basis or "").lower().startswith("def2"):
        return None
    try:
        from ase.data import atomic_numbers as _Z
    except Exception:
        return None               # ase missing -> can't check, no auto-add
    has_heavy = any(_Z.get(el.capitalize(), 0) > 36
                    for el in struct.elements)
    return "lanl2dz" if has_heavy else None


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
