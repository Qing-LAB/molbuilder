"""DNA double-strand (item 6): notation contract + X3DNA/B-form gate + duplex.

v1 scope (this ship): bare sequence -> single strand; ``ds,<seq>`` -> canonical
Watson-Crick DUPLEX (the complement is generated automatically by X3DNA fiber).
Arbitrary / mismatched duplexes are a planned follow-up (X3DNA ``rebuild``).
"""
from __future__ import annotations

import pytest

from molbuilder.nucleic import build_dna, _parse_dna_notation


# --------------------------------------------------------------------- #
#  Notation parser (pure -- no backend)                                 #
# --------------------------------------------------------------------- #

def test_bare_sequence_is_single_strand():
    assert _parse_dna_notation("ATGC") == ("ss", "ATGC", None)
    assert _parse_dna_notation("  atgc  ") == ("ss", "ATGC", None)


def test_ds_prefix_is_double_strand():
    assert _parse_dna_notation("ds,ATGC") == ("ds", "ATGC", None)
    assert _parse_dna_notation("DS:ATGC") == ("ds", "ATGC", None)


def test_direction_markers_normalise_to_5to3():
    assert _parse_dna_notation("5'-ATGC-3'") == ("ss", "ATGC", None)
    # 3'->5' input is reversed to internal 5'->3'.
    assert _parse_dna_notation("3'-ATGC-5'") == ("ss", "CGTA", None)
    assert _parse_dna_notation("ds,3'-ATGC-5'") == ("ds", "CGTA", None)


def test_two_explicit_strands_parse_to_a_duplex():
    # Two comma-separated strands -> an explicit duplex (ds), each 5'->3'.  The
    # ds prefix is optional (two strands already imply a duplex).
    assert _parse_dna_notation("ATGC,GCAT") == ("ds", "ATGC", "GCAT")
    assert _parse_dna_notation("ds,ATGC,GCAT") == ("ds", "ATGC", "GCAT")
    # Direction markers apply per strand.
    assert _parse_dna_notation("3'-ATGC-5',3'-GCAT-5'") == ("ds", "CGTA", "TACG")


def test_more_than_two_strands_rejected():
    with pytest.raises(ValueError, match="exactly two"):
        _parse_dna_notation("ATGC,GCAT,TTTT")


# --------------------------------------------------------------------- #
#  Gate (no X3DNA needed -- the gate fires before the backend runs)     #
# --------------------------------------------------------------------- #

def test_ds_rejected_on_non_x3dna_backend():
    with pytest.raises(ValueError, match="requires X3DNA"):
        build_dna("ds,ATGC", backend="rdkit")


def test_ds_rejected_on_unsupported_form():
    # B / A / Z are supported; anything else is rejected up front.
    from molbuilder.builders.backends import auto_backend_name
    if auto_backend_name() != "threedna":
        pytest.skip("X3DNA not installed; the backend gate fires first")
    with pytest.raises(ValueError, match="B / A / Z"):
        build_dna("ds,ATGC", backend="threedna", form="Q")


def test_two_strand_rejected_on_non_x3dna_backend():
    with pytest.raises(ValueError, match="requires X3DNA"):
        build_dna("ATGC,GCAT", backend="rdkit")


def test_two_strand_unequal_length_rejected():
    # Forcing threedna passes the backend gate; the equal-length check is a pure
    # validation before the backend runs, so it fires without X3DNA installed.
    with pytest.raises(ValueError, match="equal length"):
        build_dna("ATGC,GCA", backend="threedna")


def test_two_strand_non_B_form_rejected():
    with pytest.raises(ValueError, match="B-form only"):
        build_dna("ATGC,GCAT", backend="threedna", form="A")


# --------------------------------------------------------------------- #
#  Duplex build (needs X3DNA)                                           #
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def _x3dna():
    from molbuilder.builders.backends import auto_backend_name
    if auto_backend_name() != "threedna":
        pytest.skip("X3DNA (threedna) not installed")


def test_ds_build_is_a_two_chain_duplex(_x3dna):
    seq = "ATGCATGC"
    ss = build_dna(seq)
    ds = build_dna("ds," + seq)
    # A canonical duplex is the strand PLUS its complement -> ~2x the atoms.
    assert len(ds.elements) == pytest.approx(2 * len(ss.elements), abs=20)
    assert sorted(set(ds.chain_ids)) == ["A", "B"]


def test_ds_a_form_duplex(_x3dna):
    ss = build_dna("ATGCATGC", form="A")
    ds = build_dna("ds,ATGCATGC", form="A")
    assert len(ds.elements) == pytest.approx(2 * len(ss.elements), abs=20)
    assert sorted(set(ds.chain_ids)) == ["A", "B"]


def test_ds_z_form_duplex(_x3dna):
    # Z-DNA is inherently a duplex; requires alternating poly-d(GC).
    ss = build_dna("GCGCGC", form="Z")
    ds = build_dna("ds,GCGCGC", form="Z")
    assert len(ds.elements) == pytest.approx(2 * len(ss.elements), abs=20)
    assert sorted(set(ds.chain_ids)) == ["A", "B"]


def test_ds_both_strands_get_5prime_OH_by_default(_x3dna):
    # terminal="OH" (default) strips the spurious 5'-P on BOTH strands, so the
    # duplex carries no terminal phosphate (neutral-friendly for DFT).
    import numpy as np  # noqa: F401
    ds = build_dna("ds,ATGC")           # default terminal OH
    # Count 5'-terminal phosphorus: each strand's first residue should have had
    # its P stripped, so there is one fewer P than an internal-phosphate count.
    # Simpler invariant: a P sits on every internal linkage but NOT at either 5'
    # terminus -> the duplex has (n_res - n_chains) linkage phosphates.
    p_count = sum(1 for e in ds.elements if e == "P")
    n_res = len(set(zip(ds.chain_ids, ds.residue_ids)))
    n_chains = len(set(ds.chain_ids))
    assert p_count == n_res - n_chains, (
        f"expected {n_res - n_chains} backbone P (5'-P stripped on both "
        f"strands); got {p_count}")


# --------------------------------------------------------------------- #
#  Explicit / arbitrary two-strand duplex via X3DNA `rebuild`           #
# --------------------------------------------------------------------- #

def test_arbitrary_duplex_two_explicit_strands(_x3dna):
    # strand1,strand2 (both 5'->3', paired antiparallel) builds a 2-chain duplex.
    # GCATGCAT is the reverse-complement of ATGCATGC -> a canonical duplex, but
    # via the rebuild path (not fiber).
    ds = build_dna("ATGCATGC,GCATGCAT")
    assert sorted(set(ds.chain_ids)) == ["A", "B"]
    # 8 bp duplex -> chain A is 8 residues, chain B is 8 residues.
    per_chain = {c: len(set(r for cc, r in zip(ds.chain_ids, ds.residue_ids)
                            if cc == c)) for c in set(ds.chain_ids)}
    assert per_chain == {"A": 8, "B": 8}, per_chain


def test_arbitrary_duplex_allows_a_mismatch(_x3dna):
    # AAAA paired with TTTG -> one A-G (non-Watson-Crick) mismatch; rebuild must
    # still produce a valid two-chain duplex (a starting model to relax).
    ds = build_dna("AAAA,TTTG")
    assert sorted(set(ds.chain_ids)) == ["A", "B"]
    # 4 bp each strand.
    per_chain = {c: len(set(r for cc, r in zip(ds.chain_ids, ds.residue_ids)
                            if cc == c)) for c in set(ds.chain_ids)}
    assert per_chain == {"A": 4, "B": 4}, per_chain


def test_arbitrary_duplex_strips_5prime_P_on_both_strands(_x3dna):
    # Same 5'-OH invariant as the canonical path: (n_res - n_chains) backbone P.
    ds = build_dna("ATGC,GCAT")         # default terminal OH
    p_count = sum(1 for e in ds.elements if e == "P")
    n_res = len(set(zip(ds.chain_ids, ds.residue_ids)))
    n_chains = len(set(ds.chain_ids))
    assert p_count == n_res - n_chains, (
        f"expected {n_res - n_chains} backbone P (5'-P stripped on both "
        f"strands); got {p_count}")


# --------------------------------------------------------------------- #
#  Steric-clash handling (mismatched pairs interpenetrate at the frame) #
# --------------------------------------------------------------------- #

def test_canonical_explicit_duplex_has_no_clash_warning(_x3dna):
    # A canonical (complementary) explicit duplex is clash-free -> no warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # any clash warn -> failure
        build_dna("ATGC,GCAT")          # GCAT == revcomp(ATGC)


def test_mismatched_duplex_warns_by_default(_x3dna):
    # AAAA/TTTG -> an A.G purine-purine mismatch that interpenetrates; the default
    # (relax_clashes=False) must WARN (never silently emit the clashing structure).
    with pytest.warns(RuntimeWarning, match="CLASH"):
        build_dna("AAAA,TTTG")


def test_relax_clashes_clears_near_coincidence_generally(_x3dna):
    # GENERAL (not tuned to one pair): the worst single-pair clash (G.G) AND a
    # dense multi-mismatch run must both come back with NO near-coincident atoms
    # (>= 1.0 A) after relax_clashes -- a SIESTA-relaxable starting geometry.
    import warnings
    from molbuilder.chemistry import min_nonbonded_contact
    for notation in ("GCGCG,CGGGC",     # middle G.G mismatch (worst single pair)
                     "AAAA,AAAA"):       # every pair a purine-purine mismatch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            relaxed = build_dna(notation, relax_clashes=True)
        d, _, _ = min_nonbonded_contact(relaxed)
        assert d is not None and d >= 1.0, (
            f"{notation}: residual near-coincident contact {d} A after relief")
