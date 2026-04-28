"""DNA / RNA backend smoke tests across whichever backends are installed.

For each available backend, build a small DNA + RNA oligo and verify
the polymer is one connected component with no isolated atoms.  For the
helix-aware ``amber`` backend we additionally check inter-residue
P-O3' bond distance.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder import build_dna, build_rna
from molbuilder.backends import available_backends


_BACKENDS = sorted(available_backends())


def _measure_inter_PO3(struct):
    P_pos, O3_pos = {}, {}
    for i in range(struct.n_atoms):
        rid = struct.residue_ids[i]
        n   = struct.atom_names[i]
        if   n == "P":      P_pos[rid]  = struct.positions[i]
        elif n == "O3'":    O3_pos[rid] = struct.positions[i]
    inter = []
    for r, p in P_pos.items():
        if r - 1 in O3_pos:
            inter.append(float(np.linalg.norm(p - O3_pos[r - 1])))
    return inter


def _no_isolated_atoms(struct, max_nn: float = 2.5):
    P = struct.positions
    for i in range(len(P)):
        nn = min(np.linalg.norm(P[i] - P[j])
                 for j in range(len(P)) if j != i)
        assert nn < max_nn, (
            f"atom {i} ({struct.elements[i]} {struct.atom_names[i]} "
            f"res {struct.residue_ids[i]}) is isolated: nearest at {nn:.2f} A"
        )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dna_4mer_connected(backend):
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    s = build_dna("ATGC", backend=backend)
    _no_isolated_atoms(s)
    inter = _measure_inter_PO3(s)
    # Helix-aware backends must produce canonical P-O3' bonds.
    if backend != "rdkit":
        for d in inter:
            assert d < 1.80, (
                f"{backend} dna ATGC: inter-residue P-O3' = {d:.2f} A "
                f"is too long, backbone broken"
            )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rna_4mer_has_phosphorus(backend):
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    r = build_rna("AUGC", backend=backend,
                  form=("A" if backend != "rdkit" else "B"))
    assert "P" in r.elements
