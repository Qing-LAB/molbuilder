"""Generic, engine-agnostic geometry validators.

Split from the pre-2026-06-13 flat ``molbuilder/validation.py`` per
docs/science/validation.md  The function bodies
+ public signatures are identical to the pre-split versions; this
file is an organizational move, not a behaviour change.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..issues import Issue
from ..structure import Structure


# --------------------------------------------------------------------- #
#  Generic geometry checks                                              #
# --------------------------------------------------------------------- #


def validate_geometry(struct: Structure,
                      cell: Optional[np.ndarray] = None) -> List[Issue]:
    """Run only the geometry-side checks (no config / engine dispatch).

    Useful for surfaces that don't have a cfg yet -- e.g. the web
    Build page wants to flag a heavy-atom-only structure as soon as
    the user clicks Build, before they even pick SIESTA vs PySCF.

    Cell-dependent checks (volume / image distance / determinant) are
    skipped when ``cell is None``; the always-applicable checks
    (min atom distance, H/heavy ratio) still run.
    """
    issues: List[Issue] = []
    pos = struct.positions
    n   = len(pos)

    # Min atom-atom distance.  An O(N^2) pass is fine at the scale
    # this package targets (< 10k atoms); KD-tree only helps once the
    # constant overhead is amortised.
    if n >= 2:
        # Pairwise distance matrix without the diagonal.
        d  = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        min_d = float(d.min())
        if min_d < 0.3:
            issues.append(Issue(
                "error",
                f"closest atom pair is {min_d:.3f} Å apart -- atoms are "
                f"effectively on top of each other; SCF will diverge",
                "geometry.min_distance",
            ))
        elif min_d < 0.7:
            issues.append(Issue(
                "warn",
                f"closest atom pair is {min_d:.3f} Å apart -- this is "
                f"too short for any real bond; check for failed "
                f"protonation or backend output corruption",
                "geometry.min_distance",
            ))

    # H/heavy ratio.  Heavy-atom-only skeletons (typically X3DNA's `fiber`
    # output with `add_hydrogens=False`, or a hand-loaded heavy-atom PDB)
    # produce the wrong total electron count in DFT and are missing the
    # Watson-Crick / hydrogen-bond donors that hold the chemistry together.
    # Typical organic molecules sit at H/heavy ~ 0.6-1.5; nucleic acids
    # ~ 0.6.  A ratio below 0.3 is unambiguously a heavy-atom skeleton.
    #
    # **Severity: warn, not error.**  The user may legitimately want to
    # inspect or hand-process the heavy-atom skeleton (e.g., feed it to
    # an external protonator with different residue assumptions).  The
    # warning surfaces the issue prominently so they don't accidentally
    # ship a broken structure to a calculation.
    if n >= 1:
        n_h     = sum(1 for e in struct.elements if e == "H")
        n_heavy = sum(1 for e in struct.elements if e != "H")
        if n_heavy > 0 and (n_h / n_heavy) < 0.3:
            issues.append(Issue(
                "warn",
                f"H/heavy ratio is {n_h}/{n_heavy}={n_h/n_heavy:.2f} -- "
                f"structure looks like a heavy-atom skeleton (typical "
                f"organic molecules: H/heavy ~ 0.6-1.5).  DFT will "
                f"compute the wrong electron count without explicit H. "
                f"Did you mean to add hydrogens?",
                "geometry.h_ratio",
            ))

    issues += _check_polymer_orientation(struct)

    if cell is None:
        return issues

    # Cell determinant: <= 0 means left-handed or degenerate.
    det = float(np.linalg.det(cell))
    if det <= 0:
        issues.append(Issue(
            "error",
            f"cell determinant is {det:.3f} -- cell is degenerate or "
            f"left-handed (right-handed lattice vectors required)",
            "cell.determinant",
        ))
        # Don't run the volume / image checks if the cell is broken.
        return issues

    # Cell volume vs atom-bounding-volume: warn when the cell is so
    # tight that the molecule fills most of it (= guaranteed
    # image-image contact in PBC).
    if n >= 1:
        extent = pos.max(axis=0) - pos.min(axis=0)
        atom_box = float(np.prod(np.maximum(extent, 1.0)))   # min 1 Å on each side
        ratio = det / atom_box
        if ratio < 3:
            issues.append(Issue(
                "warn",
                f"cell volume / atom-bounding-volume = {ratio:.2f} -- cell "
                f"is suspiciously tight; expect image-image interactions",
                "cell.volume",
            ))

    # Atom-to-nearest-image (PBC minimum-image distance) along the VACUUM
    # directions only.  Stepping along a periodic axis would measure the
    # crystal's own neighbours (bulk gold: 2.88 A across the boundary, by
    # construction) and along a transport axis the device's intended tiling --
    # reporting the physics as a defect.  Only an ISOLATED axis has images that
    # are an artefact of the box, so only those are stepped
    # (structure-periodicity.md 2: the same reason containment is required on
    # non-periodic axes only).  A fully periodic cell has no vacuum direction
    # and the check is simply not applicable -- that is different from a check
    # that could not run, so it stays quiet rather than emitting an info.
    _kinds = struct.axis_kind or ("isolated", "isolated", "isolated")
    _vac_axes = [i for i, k in enumerate(_kinds) if k == "isolated"]
    if n >= 2 and _vac_axes:
        min_image = _min_image_distance(pos, cell, axes=_vac_axes)
        if min_image < 6.0:
            _dirs = ", ".join("abc"[i] for i in _vac_axes)
            issues.append(Issue(
                "warn",
                f"min atom-to-nearest-image distance is "
                f"{min_image:.2f} Å across the vacuum direction(s) {_dirs} -- "
                f"molecule images interact through the periodic boundary; "
                f"increase the cell or the structure's vacuum so this exceeds "
                f"~6 Å",
                "cell.image_distance",
            ))

    return issues


def _min_image_distance(positions: np.ndarray,
                        cell: np.ndarray,
                        inv:  np.ndarray = None,
                        *,
                        axes: Optional[Sequence[int]] = None) -> float:
    """Closest distance between any atom and any atom in a NEIGHBOURING
    cell (translation != (0, 0, 0)).

    The zero-translation case is excluded entirely: in-cell distances
    are real bonds, not images.  We only care about how close the
    molecule sits to its periodic copies.

    ``axes`` restricts which lattice directions may be stepped along.  This is
    the difference between a real finding and a false alarm: along an
    **isolated** (vacuum) axis a close image is an artefact of the box, but
    along a **periodic** axis the neighbouring cell holds the crystal's real
    neighbours -- bulk gold sits 2.88 Å across the boundary BY CONSTRUCTION --
    and along a **transport** axis the device is meant to tile seamlessly.
    Stepping along those would report the intended physics as a defect
    (a false positive on every crystal and every junction).  ``None`` steps
    along all three (the caller has decided they are all vacuum-like).

    For each atom i, distance to (atom j shifted by t) is computed over the
    permitted non-identity lattice translations and over all atoms j (including
    j == i, which is "this atom seeing its own copy").

    (``inv`` is vestigial -- the distances are computed in Cartesian space, so
    no inverse is needed.  Kept as an optional positional so existing callers
    keep working.)
    """
    n = positions.shape[0]
    if n == 0:
        return float("inf")
    steppable = tuple(range(3)) if axes is None else tuple(axes)
    if not steppable:
        return float("inf")          # nothing vacuum-like: no artificial images
    # Non-identity translations spanning the immediate neighbour shell, but
    # only along the permitted axes.
    ranges = [(-1, 0, 1) if i in steppable else (0,) for i in range(3)]
    shifts = [(a, b, c)
              for a in ranges[0] for b in ranges[1] for c in ranges[2]
              if (a, b, c) != (0, 0, 0)]
    translations = np.asarray(shifts, dtype=float) @ cell
    best = float("inf")
    for i in range(n):
        # Vector from atom i to every (atom j + every non-zero translation):
        deltas = (positions[None, :, :] + translations[:, None, :]
                  - positions[i, None, None, :])
        d = float(np.linalg.norm(deltas, axis=2).min())
        if d < best:
            best = d
    return best


# --------------------------------------------------------------------- #
#  Polymer orientation                                                  #
#                                                                       #
#  For nucleic acids, every backend builds residues 5' -> 3' (lowest    #
#  residue_id at the 5' end).  If a future backend (or a user-loaded    #
#  PDB from an external tool) lists residues 3' -> 5', the polymer is  #
#  chemically the same but the residue listing is reversed -- which     #
#  silently breaks any downstream code that infers orientation from    #
#  residue_ids[0] (CIF / PDB writers, web "Watch this run" handoff,    #
#  the X3DNA 5'-phosphate strip in `_threedna._strip_5prime_phosphate`).#
#                                                                       #
#  This check looks at the actual P-O3' bridges to find the structural  #
#  5' end (the residue with NO incoming bridge) and warns when it       #
#  doesn't match the lowest-numbered residue.                          #
# --------------------------------------------------------------------- #


def _check_polymer_orientation(struct: Structure) -> List[Issue]:
    if struct.residue_ids is None or struct.atom_names is None:
        return []
    # Locate P and O3' positions per residue.  If neither is present
    # this isn't a nucleic-acid polymer (or it's a heavy-atom-only
    # build with no backbone atoms named) -- silently skip.
    P_pos:  Dict[int, np.ndarray] = {}
    O3_pos: Dict[int, np.ndarray] = {}
    for i in range(struct.n_atoms):
        rid = struct.residue_ids[i]
        nm  = struct.atom_names[i]
        if nm == "P":
            P_pos[rid] = struct.positions[i]
        elif nm == "O3'":
            O3_pos[rid] = struct.positions[i]
    if not P_pos and not O3_pos:
        return []

    rids = sorted(set(struct.residue_ids))
    # has_predecessor[r] = True if residue r's P bonds to residue (r-1)'s O3'.
    has_predecessor = set()
    for r in rids:
        if r in P_pos and (r - 1) in O3_pos:
            d = float(np.linalg.norm(P_pos[r] - O3_pos[r - 1]))
            if d < 1.8:                                # bridged
                has_predecessor.add(r)

    five_prime_ends = [r for r in rids if r not in has_predecessor]
    if not five_prime_ends:
        # cyclic polymer -- nothing to orient against
        return []
    if len(five_prime_ends) > 1:
        # Multiple chains, branched, or multiple disconnected pieces.
        # If the structure has a single chain_id this is a polymer
        # integrity issue worth surfacing; multi-chain inputs (e.g.,
        # a duplex) get a pass.
        if struct.chain_ids is not None and len(set(struct.chain_ids)) <= 1:
            return [Issue(
                "warn",
                f"polymer has {len(five_prime_ends)} residues with no "
                f"preceding O3'-P bridge (residues {five_prime_ends}); "
                f"single-chain input expected exactly one 5' end.  "
                f"Possible disconnected backbone or unintended branching.",
                "polymer.orientation",
            )]
        return []

    # Exactly one 5' end -- it should be the lowest-numbered residue
    # (every backend builds 5' -> 3', so residue_ids[0] = 5' terminus).
    if five_prime_ends[0] != rids[0]:
        return [Issue(
            "warn",
            f"residue listing appears reversed: structural 5' end is "
            f"residue {five_prime_ends[0]} but residue_ids start at "
            f"{rids[0]}.  Backends should list 5' -> 3' (lowest residue_id "
            f"at the 5' end); a mismatch breaks downstream orientation-"
            f"sensitive code (terminal-phosphate stripping, FDF residue "
            f"numbering).  Likely a backend regression.",
            "polymer.orientation",
        )]
    return []
