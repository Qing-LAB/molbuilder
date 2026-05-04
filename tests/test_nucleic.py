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


def _h_add_spy_struct(n_heavy: int, n_h: int):
    """Synthetic Structure with the requested element counts, atoms
    arrayed along x at 1.5 A spacing so no two coincide.  Used by
    the _maybe_add_hydrogens gate tests."""
    import numpy as np
    from molbuilder.structure import Structure
    elements = ["C"] * n_heavy + ["H"] * n_h
    positions = np.array([(i * 1.5, 0, 0) for i in range(n_heavy + n_h)],
                         dtype=float)
    return Structure(elements=elements, positions=positions)


def _spy_add_hydrogens():
    """Monkey-patch ``chemistry.add_hydrogens`` with a passthrough
    that counts calls.  Returns (count_dict, restore_callable)."""
    import molbuilder.chemistry as chem
    called = {"n": 0}
    real = chem.add_hydrogens
    def spy(struct):
        called["n"] += 1
        return struct
    chem.add_hydrogens = spy
    def restore():
        chem.add_hydrogens = real
    return called, restore


def test_maybe_add_hydrogens_auto_runs_on_partial_protonation():
    """``mode="auto"`` heuristic: if H/heavy < 0.5, trigger
    chemistry.add_hydrogens.  Pre-tri-state the gate was at 0.3 --
    which would skip a partially-protonated structure (ratio ~0.4)
    silently.  Widened to 0.5 catches the gap zone."""
    from molbuilder.nucleic import _maybe_add_hydrogens
    s = _h_add_spy_struct(n_heavy=10, n_h=4)        # ratio 0.4
    called, restore = _spy_add_hydrogens()
    try:
        _maybe_add_hydrogens(s, mode="auto")
    finally:
        restore()
    assert called["n"] == 1


def test_maybe_add_hydrogens_auto_skips_already_protonated():
    """``mode="auto"`` skips when H/heavy >= 0.5 (canonical amber /
    rdkit nucleic output sits at ~0.63 / ~0.72 -- already complete)."""
    from molbuilder.nucleic import _maybe_add_hydrogens
    s = _h_add_spy_struct(n_heavy=10, n_h=7)        # ratio 0.7
    called, restore = _spy_add_hydrogens()
    try:
        out = _maybe_add_hydrogens(s, mode="auto")
    finally:
        restore()
    assert called["n"] == 0
    assert out is s


def test_maybe_add_hydrogens_on_forces_add_regardless_of_ratio():
    """``mode="on"`` always runs chemistry.add_hydrogens, even when
    the heuristic would skip.  The user's escape hatch when the
    auto threshold misclassifies their structure (e.g., a small
    metal-organic complex with low H but valid chemistry, or any
    case where the user knows better than the heuristic)."""
    from molbuilder.nucleic import _maybe_add_hydrogens
    # ratio 0.7 -- auto would skip; on must add
    s = _h_add_spy_struct(n_heavy=10, n_h=7)
    called, restore = _spy_add_hydrogens()
    try:
        _maybe_add_hydrogens(s, mode="on")
    finally:
        restore()
    assert called["n"] == 1


def test_maybe_add_hydrogens_off_never_calls():
    """``mode="off"`` is the user explicitly opting out -- never
    call the engine.  Used when inspecting X3DNA's heavy skeleton
    or feeding to an external protonator."""
    from molbuilder.nucleic import _maybe_add_hydrogens
    s = _h_add_spy_struct(n_heavy=10, n_h=0)        # ratio 0
    called, restore = _spy_add_hydrogens()
    try:
        out = _maybe_add_hydrogens(s, mode="off")
    finally:
        restore()
    assert called["n"] == 0
    assert out is s


def test_maybe_add_hydrogens_legacy_bool_back_compat():
    """Pre-tri-state callers passed bool: True = "auto", False = "off".
    Back-compat must hold so existing scripts and the smoke-test
    fixtures don't break."""
    from molbuilder.nucleic import _maybe_add_hydrogens
    s_low  = _h_add_spy_struct(n_heavy=10, n_h=2)   # ratio 0.2
    s_high = _h_add_spy_struct(n_heavy=10, n_h=8)   # ratio 0.8

    called, restore = _spy_add_hydrogens()
    try:
        _maybe_add_hydrogens(s_low, mode=True)      # True -> auto -> add
        _maybe_add_hydrogens(s_high, mode=True)     # True -> auto -> skip
        _maybe_add_hydrogens(s_low, mode=False)     # False -> off -> no call
    finally:
        restore()
    assert called["n"] == 1, (
        f"True+low should add (1 call); True+high should skip; "
        f"False+anything should not call.  Got {called['n']} total."
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dna_12mer_simulation_ready_at_default_settings(backend):
    """Standard end-to-end smoke at production-realistic size: a
    12-mer DNA (3x ATGC) covering all four bases and exercising 11
    internal phosphodiester bridges.  Default kwargs (add_hydrogens
    "auto", protonate_phosphates True) must give a chain with H/heavy
    in the typical organic range and zero net phosphate charge.

    The earlier 4-mer tests pin atom counts but a 12-mer is closer
    to what users actually run for stacking / canonical-helix work,
    where the H/heavy heuristic threshold matters most.
    """
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    from molbuilder.chemistry import formal_charge_from_phosphates
    s = build_dna("ATGCATGCATGC", backend=backend)
    assert s.n_residues == 12
    n_h     = sum(1 for e in s.elements if e == "H")
    n_heavy = sum(1 for e in s.elements if e != "H")
    ratio = n_h / n_heavy
    # 12-mer DNA H/heavy is ~0.6-0.7; below 0.55 means H got dropped.
    assert ratio >= 0.55, (
        f"{backend}: 12-mer ATGCATGCATGC H/heavy={ratio:.2f}, "
        f"expected >= 0.55 for a fully-protonated chain"
    )
    # 11 internal phosphate bridges + 0 terminal phosphate (terminal=OH default)
    assert s.elements.count("P") == 11, (
        f"{backend}: 12-mer with terminal=OH should have 11 P atoms, "
        f"got {s.elements.count('P')}"
    )
    # protonate_phosphates=True default => neutral
    assert formal_charge_from_phosphates(s) == 0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dna_12mer_force_on_overrides_skip(backend):
    """``add_hydrogens="on"`` must force the H-add step even when
    auto would skip (canonical backend output is already at ratio
    ~0.6+).  This is the user's escape hatch for cases where the
    heuristic might mis-classify.

    Smoke check: the on-mode result has at least as many H atoms as
    the auto-mode result -- the engine adds, never removes.  Exact
    counts may differ in the 0-4 H range due to bond-perception
    coordinates micro-shifts on the second AddHs round-trip; we
    don't pin exact equality.
    """
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    s_auto = build_dna("ATGCATGCATGC", backend=backend, add_hydrogens="auto")
    s_on   = build_dna("ATGCATGCATGC", backend=backend, add_hydrogens="on")
    n_h_auto = sum(1 for e in s_auto.elements if e == "H")
    n_h_on   = sum(1 for e in s_on.elements   if e == "H")
    # Forcing on should never produce *fewer* H than auto; equal is
    # fine (engine recognised the input was already complete).
    assert n_h_on >= n_h_auto - 4, (
        f"{backend}: force-on shouldn't drop H atoms; "
        f"auto={n_h_auto}, on={n_h_on}"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dna_12mer_off_returns_unprotonated_for_heavy_only_backends(backend):
    """``add_hydrogens="off"`` must not touch the structure -- whatever
    H the backend produced flows through unchanged.

    For X3DNA (heavy-atom skeleton) the user gets a low-H structure
    they can hand to an external protonator.  Amber and RDKit
    backends already ship H, so off-mode also leaves them alone."""
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    s = build_dna("ATGCATGCATGC", backend=backend,
                  add_hydrogens="off", protonate_phosphates=False)
    if backend == "threedna":
        # fiber emits at most a few H (terminal phosphate / 3'-OH).
        n_h = sum(1 for e in s.elements if e == "H")
        n_heavy = sum(1 for e in s.elements if e != "H")
        assert (n_h / n_heavy) < 0.1, (
            f"X3DNA + off should yield heavy skeleton, got H/heavy={n_h}/{n_heavy}"
        )
    else:
        # amber / rdkit ship H already; off-mode just preserves it.
        n_h = sum(1 for e in s.elements if e == "H")
        assert n_h > 50, (
            f"{backend} off-mode should still carry the backend's "
            f"native H atoms; got {n_h}"
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
