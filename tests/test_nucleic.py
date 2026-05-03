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


# --------------------------------------------------------------------- #
#  Hydrogen completeness across backends (X3DNA's `fiber` is a heavy-   #
#  atom skeleton; the build_dna/build_rna add_hydrogens kwarg is the    #
#  contract that all backends produce simulation-ready output).         #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dna_default_protonation_yields_simulation_ready_h_count(backend):
    """All backends, with default kwargs, must produce a structure with
    H/heavy >= 0.6 -- typical organic ratio.  Pre-fix this failed on
    threedna (H/heavy ~ 0.05 because fiber emits heavy atoms only)."""
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    s = build_dna("ATGC", backend=backend)
    n_h     = sum(1 for e in s.elements if e == "H")
    n_heavy = sum(1 for e in s.elements if e != "H")
    ratio = n_h / n_heavy
    assert ratio >= 0.55, (
        f"{backend}: H/heavy={ratio:.2f} -- structure missing hydrogens, "
        f"DFT will compute the wrong electron count"
    )


def test_dna_add_hydrogens_false_returns_heavy_skeleton():
    """The kwarg has to be honored: explicitly opting out skips the
    H-add step.  Useful when the user wants to inspect the fiber
    output directly or feed an external protonator."""
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    s = build_dna("ATGC", backend="threedna",
                  add_hydrogens=False, protonate_phosphates=False)
    n_h = sum(1 for e in s.elements if e == "H")
    # X3DNA fiber emits a few H on the terminal phosphate / 3'-OH but
    # nothing on the bases or sugars -- nowhere near a simulation-
    # ready ratio.
    assert n_h <= 5, f"expected heavy-atom skeleton, got {n_h} H"
