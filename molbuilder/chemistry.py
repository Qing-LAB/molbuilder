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
from typing import List, Tuple

import numpy as np

from .structure import Structure


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
