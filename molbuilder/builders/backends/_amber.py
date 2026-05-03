"""AmberTools backend (uses ``tleap``).

Originally this drove ``amber``'s ``bdna()`` builder, but NAB was retired
in AmberTools 23+ and its standalone binary is no longer shipped.
``nabc`` (the C library that succeeded it) doesn't expose a
sequence-to-polymer command-line helper.

The closest remaining alternative inside AmberTools is ``tleap``'s
``sequence { ... }`` macro, which builds a chemically valid polymer
(proper backbone, AmberTools OL15 force-field topology, all hydrogens)
in **extended** conformation -- not B-form / A-form helix.  For DFT
that's fine: the bond lengths are right, the optimiser doesn't care
that the chain isn't pre-coiled.

If you actually need canonical B-form / A-form geometry, install 3DNA
and the ``fiber`` command becomes available; we don't have a backend
for that yet but it's a 30-line shellout.

This backend is registered as ``"amber"`` for backwards compatibility
with code/UI that already says ``backend="amber"``.

Install:
    conda install -c conda-forge ambertools

Detected via: ``tleap`` on PATH.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from ...structure import Structure
from ._common import (parse_pdb_to_structure, select_chain,
                      verify_backbone_connectivity)


def is_available() -> bool:
    return shutil.which("tleap") is not None


def build(kind: str, sequence: str, form: str, terminal: str,
          title: Optional[str] = None) -> Structure:
    if not is_available():
        from . import BackendUnavailable
        raise BackendUnavailable(
            "AmberTools `tleap` not found in PATH; install with "
            "`conda install -c conda-forge ambertools`"
        )
    if kind not in ("dna", "rna"):
        raise ValueError(f"AmberTools backend supports kind in 'dna'|'rna'; got {kind!r}")

    if form in ("B", "A", "Z"):
        import warnings
        warnings.warn(
            f"AmberTools backend (tleap) builds an EXTENDED polymer; "
            f"the requested {form}-form helical shape is not enforced. "
            f"Bond chemistry is correct (Amber OL15 force field). "
            f"For canonical helices install 3DNA and use its `fiber` "
            f"command externally.",
            RuntimeWarning, stacklevel=4,
        )

    seq = "".join(c for c in sequence.upper() if c.isalpha())
    if not seq:
        raise ValueError("Empty sequence")

    leaprc      = "leaprc.DNA.OL15" if kind == "dna" else "leaprc.RNA.OL3"
    res_codes   = _residue_codes_for_sequence(seq, kind, terminal)
    seq_block   = " ".join(res_codes)

    with tempfile.TemporaryDirectory(prefix="molbuilder_tleap_") as workdir:
        in_path  = os.path.join(workdir, "build.in")
        pdb_path = os.path.join(workdir, "out.pdb")

        with open(in_path, "w") as fh:
            fh.write(f"""\
source {leaprc}
m = sequence {{ {seq_block} }}
savepdb m {pdb_path}
quit
""")

        try:
            result = subprocess.run(
                ["tleap", "-f", in_path],
                capture_output=True, text=True, cwd=workdir,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "tleap did not finish within 120 s -- likely hung or "
                "stuck on a slow filesystem.  Try shorter sequence, or "
                "run tleap by hand to diagnose."
            ) from exc
        # tleap returns 0 even on some errors; check the actual output instead.
        if not os.path.isfile(pdb_path):
            raise RuntimeError(
                "tleap ran but produced no PDB.  Output was:\n"
                f"{result.stdout}\n--- stderr ---\n{result.stderr}"
            )

        with open(pdb_path) as fh:
            pdb_text = fh.read()

    struct = parse_pdb_to_structure(
        pdb_text,
        title=title or f"{kind} {seq} (tleap, extended chain)",
    )

    # tleap emits a single chain; this is a no-op when there's only one,
    # but keeps the API symmetric with the old amber path.
    chains = sorted(set(struct.chain_ids))
    if len(chains) > 1:
        struct = select_chain(struct, chains[0])

    err = verify_backbone_connectivity(struct, kind, max_O3_P=1.80)
    if err is not None:
        raise RuntimeError(
            f"tleap output failed connectivity check: {err}.  "
            f"This is unexpected from tleap and probably indicates a "
            f"force-field mismatch."
        )

    # tleap copies hydrogen positions verbatim from the residue
    # library, which assumes canonical intra-residue geometry.  At the
    # -CH2- methylene junctions in the polymer (notably C5' between O5'
    # and C4') the actual local geometry differs from the library
    # template, so the two methylene Hs end up at wrong angles.  We
    # recompute *only* those H positions using tetrahedral sp3 geometry
    # derived from the heavy-atom neighbours -- no atoms added or
    # removed, no other H positions touched, no bond-perception games
    # involving phosphates.
    struct = _fix_methylene_hydrogens(struct)
    return struct


# --------------------------------------------------------------------- #
# Hydrogen-position correction (post tleap)                             #
# --------------------------------------------------------------------- #


def _fix_methylene_hydrogens(struct):
    """Recompute the H positions on every sp3 -CH2- methylene group.

    tleap's ``sequence`` macro copies H positions verbatim from the
    residue library, which assumes a canonical intra-residue geometry.
    At inter-residue junctions, the actual heavy-atom geometry differs
    from the library template, so the two methylene Hs end up at wrong
    positions and the H-C-H angle is off.  In nucleic acids this hits
    every C5' between O5' and C4'; in proteins it hits side-chain CH2
    groups at residue junctions.

    What this helper does:
      * Find every C atom with exactly 2 heavy neighbours and 2 H
        neighbours -- that's the unambiguous "sp3 -CH2-" signature.
      * Compute the canonical tetrahedral H positions: the two Hs lie
        in the plane perpendicular to the heavy-atom-bisector, at half
        the H-C-H angle (109.47 deg / 2 = 54.74 deg) on each side.
      * Move the existing H atoms to those positions.

    No atoms are added or removed.  Heavy atoms are never touched.
    Phosphates and other groups are left alone, so we don't have to
    worry about RDKit's proximity-bond perception turning a phosphate
    into a phosphite.

    Returns a new Structure (the input is not mutated).  If the structure
    contains no methylene groups, returns it unchanged.
    """
    elements   = list(struct.elements)
    positions  = struct.positions.copy()
    n          = len(positions)

    # Build neighbour lists by simple covalent-distance bonding.  We
    # iterate per-atom for clarity; an O(N^2) pass is fine at the scale
    # this package targets (< 10k atoms).
    nb_heavy: List[List[int]] = [[] for _ in range(n)]
    nb_h:     List[List[int]] = [[] for _ in range(n)]
    HX_CUT  = 1.30   # X-H bond cutoff (longer than 1.09 to be safe)
    XX_CUT  = 1.95   # heavy-heavy cutoff (covers C-S, C-P, etc.)

    for i in range(n):
        ei = elements[i]
        for j in range(i + 1, n):
            ej = elements[j]
            d  = float(np.linalg.norm(positions[i] - positions[j]))
            if "H" in (ei, ej):
                if d > HX_CUT:
                    continue
            else:
                if d > XX_CUT:
                    continue
            # Bond accepted -- categorise neighbours.
            if ej == "H": nb_h[i].append(j)
            else:         nb_heavy[i].append(j)
            if ei == "H": nb_h[j].append(i)
            else:         nb_heavy[j].append(i)

    BOND_CH    = 1.09                     # C-H bond length, Ang
    HALF_HCH   = math.radians(109.471 / 2)  # tetrahedral half-angle

    fixed = 0
    for i in range(n):
        if elements[i] != "C": continue
        if len(nb_heavy[i]) != 2 or len(nb_h[i]) != 2: continue
        c     = positions[i]
        a, b  = nb_heavy[i]
        h1, h2 = nb_h[i]

        v1 = positions[a] - c;  v1 /= np.linalg.norm(v1)
        v2 = positions[b] - c;  v2 /= np.linalg.norm(v2)

        # Bisector pointing AWAY from the two heavy bonds (where the
        # methylene Hs go), and a perpendicular axis in the v1 x v2
        # plane to split the two Hs symmetrically.
        bis  = -(v1 + v2)
        nb   = np.linalg.norm(bis)
        if nb < 1e-9:           # heavy bonds are anti-parallel (rare)
            continue
        bis /= nb
        perp = np.cross(v1, v2)
        np_ = np.linalg.norm(perp)
        if np_ < 1e-9:           # heavy bonds are co-linear (degenerate)
            continue
        perp /= np_

        # The two H positions, exactly tetrahedral around C.
        d_along = math.cos(HALF_HCH)
        d_perp  = math.sin(HALF_HCH)
        positions[h1] = c + BOND_CH * (d_along * bis + d_perp * perp)
        positions[h2] = c + BOND_CH * (d_along * bis - d_perp * perp)
        fixed += 1

    if fixed == 0:
        return struct

    return Structure(
        elements      = list(struct.elements),
        positions     = positions,
        atom_names    = list(struct.atom_names),
        residue_ids   = list(struct.residue_ids),
        residue_names = list(struct.residue_names),
        chain_ids     = list(struct.chain_ids),
        title         = struct.title,
    )


# --------------------------------------------------------------------- #
# Residue-name mapping for tleap                                        #
# --------------------------------------------------------------------- #
#
# AmberTools uses these residue codes:
#
#   DNA:  DA5 / DT5 / DG5 / DC5     -- 5'-terminal (5'-OH)
#         DA  / DT  / DG  / DC      -- inner residues (with phosphate)
#         DA3 / DT3 / DG3 / DC3     -- 3'-terminal (3'-OH)
#
#   RNA:  A5  / U5  / G5  / C5      -- 5'-terminal
#         A   / U   / G   / C       -- inner
#         A3  / U3  / G3  / C3      -- 3'-terminal
#
# All 5'-terminal variants have a free 5'-hydroxyl, not a phosphate; if
# the user requested terminal="5P" or "PP" we'd need different residues
# (DA5MP and friends in some force fields).  Those aren't universally
# present so for v1 we just emit the standard 5'-OH terminals and the
# user gets terminal="OH" semantics regardless.

def _residue_codes_for_sequence(seq: str, kind: str, terminal: str) -> List[str]:
    if terminal not in ("OH",):
        import warnings
        warnings.warn(
            f"AmberTools backend currently emits only 5'-OH/3'-OH termini; "
            f"requested terminal={terminal!r} ignored.  For phosphorylated "
            f"termini, post-process the .pdb by hand or use a different "
            f"toolchain (e.g. 3DNA).",
            RuntimeWarning, stacklevel=5,
        )
    n = len(seq)
    out: List[str] = []
    for i, letter in enumerate(seq):
        prefix = "D" if kind == "dna" else ""
        if i == 0:
            suffix = "5"          # 5'-terminal
        elif i == n - 1:
            suffix = "3"          # 3'-terminal
        else:
            suffix = ""           # inner
        out.append(f"{prefix}{letter}{suffix}")
    return out
