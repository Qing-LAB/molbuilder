"""AmberTools backend (uses ``tleap``).

AmberTools 23+ does not ship NAB's standalone binary, so ``bdna()`` is
not reachable from a command line; ``nabc``, the C library that succeeded
it, exposes no sequence-to-polymer command-line helper either.

The closest remaining alternative inside AmberTools is ``tleap``'s
``sequence { ... }`` macro, which builds a chemically valid polymer
(proper backbone, AmberTools OL15 force-field topology, all hydrogens)
in **extended** conformation -- not B-form / A-form helix.  For DFT
that's fine: the bond lengths are right, the optimiser doesn't care
that the chain isn't pre-coiled.

Install:
    See ``docs/ops/installation.md`` § ``molbuilder-MDtools`` -- the one
    place ``tleap`` runs from: its own env, never the host's PATH
    (`installation.md` M6).

Detected via: the diagnostics :class:`~molbuilder.diagnostics.Capabilities`
snapshot, which reports ``tleap`` as available iff its routed env is on
this machine and holds it.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from ...diagnostics import get_capabilities
from ...envs import run_tool
from ...residues import AMBER_SUFFIX
from ...structure import Structure
from . import Built
from ._common import (first_chain,
                      verify_backbone_connectivity)

#: This backend builds a single strand only.
DUPLEX = False
#: How it is installed: its own env, AmberTools' recipe's (``ENV``, its
#: category; `ops/installation.md`) -- never a tleap on the host's PATH
#: (`installation.md` M6).
INSTALL = "it runs from its own env, AmberTools'"
ENV = "mdtools"


def is_available() -> bool:
    """``tleap`` can run: its routed env is on this machine and holds it.

    Asked of the diagnostics :class:`~molbuilder.diagnostics.Capabilities`
    snapshot (`tool_available`; never the host's PATH).  The
    snapshot is the server's, bound at start, or a command's, taken on its
    first use (`diagnostics.get_capabilities`), which lists the conda envs
    once for the process.
    """
    return get_capabilities().tool_available("tleap")


def build(kind: str, sequence: str, form: str, terminal: str,
          title: Optional[str] = None, double_strand: bool = False,
          strand2: Optional[str] = None) -> Structure:
    """A single strand of ``kind`` from ``sequence`` -- read by the grammar
    already -- as tleap's extended polymer: no form is shaped here, the check
    after the build says so (:func:`built`)."""
    leaprc      = "leaprc.DNA.OL15" if kind == "dna" else "leaprc.RNA.OL3"
    res_codes   = _residue_codes_for_sequence(sequence, kind)
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
            # run_tool runs the molbuilder-MDtools env's own tleap, by its
            # path, through the one door (`builds.dispatch_into_env`) --
            # never the host's PATH (`installation.md` M6).  Through the
            # door stdout and stderr come back as one stream.
            result = run_tool(
                "tleap", ["-f", in_path], cwd=workdir, timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "tleap did not finish within 120 s -- likely hung or "
                "stuck on a slow filesystem.  Try shorter sequence, or "
                "run the env's own tleap by hand to diagnose "
                "(\"${CONDA_PREFIX:?}/bin/tleap\", its env active)."
            ) from exc
        # tleap returns 0 even on some errors; check the actual output instead.
        if not os.path.isfile(pdb_path):
            # One transcript: through the env door stdout and stderr come
            # back as one stream, so a separate "stderr" heading would always
            # be empty and read as "tleap said nothing there".
            raise RuntimeError(
                "tleap ran but produced no PDB.  Output was:\n"
                f"{result.stdout}{result.stderr}"
            )

        with open(pdb_path) as fh:
            pdb_text = fh.read()

    struct = Structure.from_pdb(
        pdb_text,
        title=title or f"{kind} {sequence} (tleap, extended chain)",
    )

    # A backend returns ONE chain.  tleap emits a single chain, so this is
    # normally a no-op; it is kept because the guarantee is the backend's,
    # not tleap's, and a multi-chain emission must not leak downstream.
    struct = first_chain(struct)

    err = verify_backbone_connectivity(struct)
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

    from ...chemistry import is_atom

    fixed = 0
    for i in range(n):
        if not is_atom(elements[i], "C"): continue
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

    return struct.replace(positions=positions)


# --------------------------------------------------------------------- #
# Residue-name mapping for tleap                                        #
# --------------------------------------------------------------------- #
#
# AmberTools uses these residue codes:
#
#   DNA:  DA5 / DT5 / DG5 / DC5     -- 5'-terminal (5'-OH)
#         DA  / DT  / DG  / DC      -- inner residues (with phosphate)
#         DA3 / DT3 / DG3 / DC3     -- 3'-terminal (3'-OH)
#         DAN / DTN / DGN / DCN     -- a single nucleotide, free at BOTH ends
#
#   RNA:  A5  / U5  / G5  / C5      -- 5'-terminal
#         A   / U   / G   / C       -- inner
#         A3  / U3  / G3  / C3      -- 3'-terminal
#         AN  / UN  / GN  / CN      -- a single nucleotide
#
# A one-letter sequence is BOTH ends at once: its residue is the "N" one --
# the 5'-terminal residue alone would leave its 3' end open for a link that
# never comes, no hydrogen on its O3', an odd electron count (measured
# 2026-11-18: DNA "A" 131 electrons, RNA "A" 139).
#
def built(kind: str, form: str, terminal: str) -> Built:
    """What this backend builds of the ask (`backends.built`): an extended
    polymer, no helix, and 5'-OH / 3'-OH termini whatever is asked
    (:func:`_residue_codes_for_sequence`)."""
    return Built(
        form=None, terminal="OH",
        form_words="AmberTools (tleap) builds an extended polymer -- its bond "
                   "chemistry correct (Amber OL15 / OL3), not a helix",
        terminal_words="AmberTools writes only 5'-OH / 3'-OH termini")


def _residue_codes_for_sequence(seq: str, kind: str) -> List[str]:
    # ONLY 5'-OH / 3'-OH TERMINI are written, whatever is asked; the check
    # after the build says so (`built`).
    n = len(seq)
    prefix = "D" if kind == "dna" else ""
    out: List[str] = []
    for i, letter in enumerate(seq):
        where = ("single" if n == 1 else "5'" if i == 0
                 else "3'" if i == n - 1 else "inner")
        out.append(f"{prefix}{letter}{AMBER_SUFFIX[where]}")
    return out
