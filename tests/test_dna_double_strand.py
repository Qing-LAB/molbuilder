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
    assert _parse_dna_notation("ATGC") == ("ss", "ATGC")
    assert _parse_dna_notation("  atgc  ") == ("ss", "ATGC")


def test_ds_prefix_is_double_strand():
    assert _parse_dna_notation("ds,ATGC") == ("ds", "ATGC")
    assert _parse_dna_notation("DS:ATGC") == ("ds", "ATGC")


def test_direction_markers_normalise_to_5to3():
    assert _parse_dna_notation("5'-ATGC-3'") == ("ss", "ATGC")
    # 3'->5' input is reversed to internal 5'->3'.
    assert _parse_dna_notation("3'-ATGC-5'") == ("ss", "CGTA")
    assert _parse_dna_notation("ds,3'-ATGC-5'") == ("ds", "CGTA")


def test_two_explicit_strands_rejected_with_pointer():
    with pytest.raises(ValueError, match="not yet supported|rebuild"):
        _parse_dna_notation("3'-ATGC-5',3'-GCAT-5'")


# --------------------------------------------------------------------- #
#  Gate (no X3DNA needed -- the gate fires before the backend runs)     #
# --------------------------------------------------------------------- #

def test_ds_rejected_on_non_x3dna_backend():
    with pytest.raises(ValueError, match="requires X3DNA"):
        build_dna("ds,ATGC", backend="rdkit")


def test_ds_rejected_on_non_B_form():
    # Force the X3DNA backend so we reach the form check (not the backend gate).
    from molbuilder.builders.backends import auto_backend_name
    if auto_backend_name() != "threedna":
        pytest.skip("X3DNA not installed; the backend gate fires first")
    with pytest.raises(ValueError, match="B-form only"):
        build_dna("ds,ATGC", backend="threedna", form="A")


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
