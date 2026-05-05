"""Molecular-charge helpers.

Two functions:

  formal_charge_from_phosphates(struct) -> int
      Estimate the molecule's formal charge from the protonation state
      of its phosphate groups.  Each non-bridging phosphate oxygen
      that has no hydrogen and is not the implicit P=O is counted as
      contributing -1 to the molecular charge.  Other charged groups
      (carboxylates, protonated amines, etc.) are NOT counted in this
      release; we focus on the case the user actually hit -- DNA / RNA
      phosphate diester chains coming out of tleap.

  protonate_phosphate_oxygens(struct) -> (struct, n_added)
      Add hydrogens to deprotonated non-bridging phosphate oxygens so
      the molecule becomes formally neutral.  For each P with N
      non-bridging oxygens, leaves one as the implicit P=O (no H added)
      and ensures the other (N-1) carry an OH.  The H position is
      placed at the canonical sp3 angle (109.47 deg from the P-O axis,
      0.96 A bond length) on the side facing AWAY from the rest of the
      molecule, so the OH points outward.

The chemistry rule encoded here is the standard interpretation of a
phosphate group: one P=O double bond, the other oxygens single-bonded
to either H, R-O-, or charged O- depending on protonation.  We don't
require the user's structure to have explicit bond orders; we infer
"non-bridging" purely from heavy-atom adjacency (only P as a heavy
neighbour).
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
