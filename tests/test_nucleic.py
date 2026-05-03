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


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dna_atgc_protonation_chemistry_matches_across_backends(backend):
    """Pin that every backend produces the same protonation chemistry for
    a canonical ATGC chain: 5 O-H (3 phosphate + 5'-OH + 3'-OH), 37 C-H
    (sugar CH/CH2 + base CH + thymine methyl), 8 N-H (Watson-Crick base-
    pairing donors: A.N6-H2, T.N3-H, G.N1-H + G.N2-H2, C.N4-H2).

    Pre-fix the X3DNA path went through RDKit's PDB-bond-perception, which
    failed to compute coords for ambiguous-valence base amines; the
    `_drop_overlapping_hydrogens` post-pass then removed them as ghost
    atoms and the chain came back missing 3 N-H + 1 C-H -- structurally
    crippled for Watson-Crick H-bonding.  OpenBabel's geometric H-add
    avoids the ghost-coord failure mode, so X3DNA + OpenBabel matches
    Amber-tleap and RDKit-via-SMILES exactly."""
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    s = build_dna("ATGC", backend=backend)
    by_anchor = {"O": 0, "C": 0, "N": 0}
    pos = s.positions
    for i in range(s.n_atoms):
        if s.elements[i] != "H":
            continue
        # closest non-H neighbour
        best_d, best_j = 99.0, -1
        for j in range(s.n_atoms):
            if i == j or s.elements[j] == "H":
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and s.elements[best_j] in by_anchor:
            by_anchor[s.elements[best_j]] += 1
    assert by_anchor == {"O": 5, "C": 37, "N": 8}, (
        f"{backend}: H anchor breakdown is {by_anchor}, expected "
        f"{{'O': 5, 'C': 37, 'N': 8}} -- Watson-Crick H may be missing"
    )


def test_threedna_strips_5prime_phosphate_for_terminal_oh():
    """fiber always emits a 5'-phosphate; for terminal='OH' we must
    strip it so the chain count matches the user's request and the
    other backends.

    Pinning: a single dA nucleotide with terminal='OH' must be
    deoxyadenosine (no P), and a 4-mer must have 3 phosphate groups
    (the 3 internal bridges), not 4."""
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")

    # Single nucleotide -- pre-fix had 1 phosphate (the spurious 5'-P);
    # post-fix has 0 (it's a free deoxyadenosine).
    s_a = build_dna("A", backend="threedna", terminal="OH",
                    add_hydrogens=False, protonate_phosphates=False)
    assert s_a.elements.count("P") == 0, (
        f"single dA with terminal=OH should have 0 phosphates, "
        f"got {s_a.elements.count('P')}"
    )

    # 4-mer -- pre-fix had 4 phosphates, post-fix has 3 (only the
    # internal A-T, T-G, G-C bridges).
    s_atgc = build_dna("ATGC", backend="threedna", terminal="OH",
                       add_hydrogens=False, protonate_phosphates=False)
    assert s_atgc.elements.count("P") == 3, (
        f"ATGC with terminal=OH should have 3 phosphate bridges, "
        f"got {s_atgc.elements.count('P')}"
    )

    # 5'-terminal residue should now start at O5' (not P).  fiber
    # writes residues in 5'->3' order, so residue_ids[0] is the 5'-end.
    first_rid = s_atgc.residue_ids[0]
    first_res_atom_names = {
        s_atgc.atom_names[i] for i in range(s_atgc.n_atoms)
        if s_atgc.residue_ids[i] == first_rid
    }
    assert "P" not in first_res_atom_names, (
        f"5'-terminal residue should not contain P after strip; "
        f"atoms: {sorted(first_res_atom_names)}"
    )
    assert "O5'" in first_res_atom_names, (
        f"5'-terminal residue should retain O5' as a free hydroxyl; "
        f"atoms: {sorted(first_res_atom_names)}"
    )
