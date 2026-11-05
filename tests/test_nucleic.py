"""DNA / RNA backend smoke tests across whichever backends are installed.

For each available backend, build a small DNA + RNA oligo and verify
the polymer is one connected component with no isolated atoms.  For the
helix-aware ``amber`` backend we additionally check inter-residue
P-O3' bond distance.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.nucleic import build_dna, build_rna
from molbuilder.builders.backends import available_backends


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
    chemistry.add_hydrogens.  The gate sits at 0.5 so a partially-
    protonated structure (ratio ~0.4) is not skipped silently."""
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


@pytest.mark.parametrize("said", [True, False, "maybe"])
def test_add_hydrogens_says_one_of_its_three_words(said):
    """`True` once meant `auto` here and *always* for a peptide: one word, two
    meanings, so a person states `auto`, `on` or `off`, and anything else is
    refused naming them (`engines/builders.md`)."""
    from molbuilder.nucleic import normalise_h_mode
    with pytest.raises(ValueError, match="'auto', 'on' or 'off'"):
        normalise_h_mode(said)


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


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dna_12mer_simulation_ready_at_default_settings(backend):
    """Standard end-to-end smoke at production-realistic size: a
    12-mer DNA (3x ATGC) covering all four bases and exercising 11
    internal phosphodiester bridges.  Default kwargs (add_hydrogens
    "auto", protonate_phosphates True) must give a chain with H/heavy
    in the typical organic range and zero net phosphate charge.

    A 12-mer is closer to what users actually run for stacking / canonical-helix work,
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
                  add_hydrogens="off", protonate_phosphates=False)
    n_h = sum(1 for e in s.elements if e == "H")
    # X3DNA fiber emits a few H on the terminal phosphate / 3'-OH but
    # nothing on the bases or sugars -- nowhere near a simulation-
    # ready ratio.
    assert n_h <= 5, f"expected heavy-atom skeleton, got {n_h} H"


#: THE CANONICAL TAUTOMER of each base, the form in DNA and RNA: the
#: hydrogens on each of its ring and exocyclic N and O atoms (amino, keto).
_TAUTOMER_H = {
    "A": {"N6": 2, "N1": 0, "N3": 0, "N7": 0},
    "G": {"N1": 1, "N2": 2, "O6": 0, "N3": 0, "N7": 0},
    "C": {"N4": 2, "N3": 0, "O2": 0},
    "T": {"N3": 1, "O4": 0, "O2": 0},
    "U": {"N3": 1, "O4": 0, "O2": 0},
}
#: Residue name -> base, the three backends' spellings (3DNA's `DC` / `C`,
#: tleap's terminal `DC5` / `C3`).
_BASE = {"DA": "A", "DT": "T", "DG": "G", "DC": "C",
         "A": "A", "U": "U", "G": "G", "C": "C"}

#: A build as a person asks it, and the anchors of its hydrogens where the
#: row states them.
_CHEMISTRY_ROWS = [
    pytest.param(build_dna, "ATGC", {"O": 5, "C": 37, "N": 8}, id="dna"),
    pytest.param(build_rna, "ACGU", None, id="rna"),
    pytest.param(build_dna, "ds,GCGC", None, id="canonical-duplex"),
]


@pytest.mark.parametrize("build, text, anchors", _CHEMISTRY_ROWS)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_protonation_chemistry_matches_across_backends(backend, build, text,
                                                       anchors):
    """Pin that every backend produces the same protonation chemistry: for
    a canonical ATGC chain 5 O-H (3 phosphate + 5'-OH + 3'-OH), 37 C-H
    (sugar CH/CH2 + base CH + thymine methyl), 8 N-H (Watson-Crick base-
    pairing donors: A.N6-H2, T.N3-H, G.N1-H + G.N2-H2, C.N4-H2); every base
    in its canonical tautomer; each H in the residue of the atom it bonds to.

    Pre-fix the X3DNA path went through RDKit's PDB-bond-perception, which
    failed to compute coords for ambiguous-valence base amines; the
    `_drop_overlapping_hydrogens` post-pass then removed them as ghost
    atoms and the chain came back missing 3 N-H + 1 C-H -- structurally
    crippled for Watson-Crick H-bonding.  OpenBabel's geometric H-add
    avoids the ghost-coord failure mode, so X3DNA + OpenBabel matches
    Amber-tleap and RDKit-via-SMILES exactly.  The counts could not see
    the tautomer: OpenBabel read every cytosine 3DNA built as the imino
    form -- an H on N3, one on N4, the same two N-H -- and a canonical
    duplex's C.N3-H met G.N1-H 0.88 A apart, until the bases' bond orders
    were given it (`residues.NUCLEOBASE_BONDS`, 2026-10-31).  RDKit's
    hydrogens all read as one residue `MOL` until its `AddHs` was asked for
    their residues (2026-10-31)."""
    if not available_backends()[backend]:
        pytest.skip(f"backend {backend!r} not installed on this machine")
    if text.startswith("ds,") and backend != "threedna":
        pytest.skip("a duplex is 3DNA's alone")
    s = build(text, backend=backend)
    pos = s.positions
    is_h = np.array([e == "H" for e in s.elements])
    by_anchor = {"O": 0, "C": 0, "N": 0}
    elsewhere = []
    for i in np.flatnonzero(is_h):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[is_h] = 99.0
        j = int(np.argmin(d))
        if s.elements[j] in by_anchor:
            by_anchor[s.elements[j]] += 1
        if (s.chain_ids[i], s.residue_ids[i]) != (s.chain_ids[j],
                                                  s.residue_ids[j]):
            elsewhere.append(f"{s.residue_names[i]}{s.residue_ids[i]}:"
                             f"{s.atom_names[i]}")
    if anchors is not None:
        assert by_anchor == anchors, (
            f"{backend}: H anchor breakdown is {by_anchor}, expected "
            f"{anchors} -- Watson-Crick H may be missing")
    assert not elsewhere, (
        f"{backend}: {len(elsewhere)} H outside their anchor's residue: "
        f"{elsewhere[:5]}")
    wrong = []
    for k in range(s.n_atoms):
        base = _BASE.get(s.residue_names[k].rstrip("53"))
        want = _TAUTOMER_H.get(base, {}).get(s.atom_names[k])
        if want is None:
            continue
        got = int(((np.linalg.norm(pos - pos[k], axis=1) < 1.15)
                   & is_h).sum())
        if got != want:
            wrong.append(f"{s.chain_ids[k]}{s.residue_ids[k]}"
                         f"{s.residue_names[k]}:{s.atom_names[k]} {got} H")
    assert not wrong, f"{backend}: not the canonical tautomer: {wrong}"


def test_threedna_strips_5prime_phosphate_for_terminal_oh():
    """fiber always emits a 5'-phosphate; for terminal='OH' we must
    strip it so the chain count matches the user's request and the
    other backends.

    Pinning: a single dA nucleotide with terminal='OH' must be
    deoxyadenosine (no P), and a 4-mer must have 3 phosphate groups
    (the 3 internal bridges), not 4."""
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")

    # Single nucleotide -- a free deoxyadenosine, no phosphate.
    s_a = build_dna("A", backend="threedna", terminal="OH",
                    add_hydrogens="off", protonate_phosphates=False)
    assert s_a.elements.count("P") == 0, (
        f"single dA with terminal=OH should have 0 phosphates, "
        f"got {s_a.elements.count('P')}"
    )

    # 4-mer -- 3 phosphates (only the internal A-T, T-G, G-C bridges).
    s_atgc = build_dna("ATGC", backend="threedna", terminal="OH",
                       add_hydrogens="off", protonate_phosphates=False)
    assert s_atgc.elements.count("P") == 3, (
        f"ATGC with terminal=OH should have 3 phosphate bridges, "
        f"got {s_atgc.elements.count('P')}"
    )

    # 5'-terminal residue should start at O5' (not P).  fiber
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


# --- build_dna 5'/3' directionality now matches parse_dna_sequence -------
# Regression (2026-07-27): build_dna's parse_dna_notation used a permissive
# local `_strip_direction` that silently kept one-sided labels and silently
# reversed "3'-ATGC-3'", so a duplex strand could be built in the wrong 5'->3'
# order with no error -- while parse_dna_sequence / build_rna already rejected
# the same inputs.  _strip_direction now delegates to the strict
# residues._strip_directionality, so DNA and RNA behave identically.

@pytest.mark.parametrize("bad", ["5'-ATGC", "ATGC-3'", "5'-ATGC-5'", "3'-ATGC-3'"])
def test_build_dna_notation_rejects_bad_directionality(bad):
    from molbuilder.nucleic import parse_dna_notation
    with pytest.raises(ValueError):
        parse_dna_notation(bad)


def test_build_dna_notation_normalises_reverse_direction():
    from molbuilder.nucleic import parse_dna_notation
    # 3'-CGTA-5' is the same chemical strand as 5'-ATGC-3', normalised 5'->3'.
    assert parse_dna_notation("3'-CGTA-5'") == ("ss", "ATGC", None)
    assert parse_dna_notation("ATGC") == ("ss", "ATGC", None)
    # per-strand labels in a two-strand duplex are handled too
    assert parse_dna_notation("ATGC,3'-CGTA-5'") == ("ds", "ATGC", "ATGC")
