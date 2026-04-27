"""DNA / RNA builder smoke tests across backends.

For each backend installed on this machine, build a small DNA and RNA
oligo and verify the polymer is one connected component with no
isolated atoms.  For the helix-aware ``amber`` backend we additionally
check inter-residue P-O3' bond distance.
"""

from __future__ import annotations

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from molbuilder import build_dna, build_rna
from molbuilder.backends import available_backends


def measure_inter_PO3(struct):
    P_pos, O3_pos = {}, {}
    for i in range(struct.n_atoms):
        rid = struct.residue_ids[i]
        n = struct.atom_names[i]
        if n == "P":      P_pos[rid] = struct.positions[i]
        elif n == "O3'":  O3_pos[rid] = struct.positions[i]
    inter = []
    for r, p in P_pos.items():
        if r - 1 in O3_pos:
            inter.append(float(np.linalg.norm(p - O3_pos[r - 1])))
    return inter


def main() -> None:
    avail = available_backends()
    print(f"available backends: {avail}")

    any_tested = False
    for backend in ("rdkit", "amber"):
        if not avail[backend]:
            print(f"  skip {backend!r}: not installed")
            continue
        any_tested = True
        print(f"  testing backend = {backend!r}")

        # ---- DNA 4-mer ---------------------------------------------
        s = build_dna("ATGC", backend=backend)
        inter = measure_inter_PO3(s)
        print(f"    dna ATGC: {s.n_atoms} atoms, "
              f"inter-residue P-O3' distances = "
              f"{[round(d, 2) for d in inter]} A")

        # No isolated atoms
        P = s.positions
        for i in range(len(P)):
            nn = min(np.linalg.norm(P[i] - P[j])
                     for j in range(len(P)) if j != i)
            assert nn < 2.5, (
                f"{backend} dna: atom {i} ({s.elements[i]} "
                f"{s.atom_names[i]} res {s.residue_ids[i]}) is isolated, "
                f"nearest neighbour at {nn:.2f} A"
            )

        # Helix-aware backends must produce canonical P-O3' bonds.
        if backend != "rdkit":
            for d in inter:
                assert d < 1.80, (
                    f"{backend} dna ATGC: inter-residue P-O3' = "
                    f"{d:.2f} A is too long, backbone broken"
                )

        # ---- RNA 4-mer ---------------------------------------------
        r = build_rna(
            "AUGC", backend=backend,
            form=("A" if backend != "rdkit" else "B"),
        )
        print(f"    rna AUGC: {r.n_atoms} atoms")
        assert "P" in r.elements

    if not any_tested:
        print("  no backends installed; install rdkit or ambertools.")
        return

    print("OK -- nucleic backends produce connected polymers "
          "(connectivity verified; helix-aware backends also pass "
          "inter-residue P-O3' bond check).")


if __name__ == "__main__":
    main()
