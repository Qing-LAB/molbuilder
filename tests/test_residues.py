"""Sequence-parser tests (no external deps)."""

from __future__ import annotations

import pytest

from molbuilder.residues import (
    parse_dna_sequence,
    parse_peptide_sequence,
    parse_rna_sequence,
)


# ---- peptides ----------------------------------------------------


@pytest.mark.parametrize("seq, expected", [
    ("ARNDC",                  ["ALA", "ARG", "ASN", "ASP", "CYS"]),
    ("arndc",                  ["ALA", "ARG", "ASN", "ASP", "CYS"]),
    ("A R N D C",              ["ALA", "ARG", "ASN", "ASP", "CYS"]),
    ("AR NDC",                 ["ALA", "ARG", "ASN", "ASP", "CYS"]),
    ("AR[SEP]C",               ["ALA", "ARG", "SEP", "CYS"]),
    ("[ALA][ARG][SEP][CYS]",   ["ALA", "ARG", "SEP", "CYS"]),
    ("A[PTR][SEP][TPO]C",      ["ALA", "PTR", "SEP", "TPO", "CYS"]),
])
def test_parse_peptide_sequence(seq, expected):
    assert parse_peptide_sequence(seq) == expected


@pytest.mark.parametrize("bad, why", [
    ("AXC",       "X isn't a valid AA letter"),
    ("ZZZ",       "Z isn't a valid AA letter"),
    ("Ala-Arg",   "dashes are no longer accepted"),
    ("AR(SEP)C",  "parens are no longer accepted (use brackets)"),
    ("AR[XXX]C",  "XXX isn't a known 3-letter code"),
    ("AR[SEPC",   "unclosed bracket"),
])
def test_parse_peptide_sequence_rejects(bad, why):
    with pytest.raises(ValueError):
        parse_peptide_sequence(bad)


# ---- DNA / RNA ----------------------------------------------------


@pytest.mark.parametrize("seq, expected", [
    ("ATGC",        ["DA", "DT", "DG", "DC"]),
    ("atgc",        ["DA", "DT", "DG", "DC"]),
    ("ATGC ATGC",   ["DA", "DT", "DG", "DC", "DA", "DT", "DG", "DC"]),
    ("[DA][DT]GC", ["DA", "DT", "DG", "DC"]),
])
def test_parse_dna_sequence(seq, expected):
    assert parse_dna_sequence(seq) == expected


def test_parse_rna_sequence():
    assert parse_rna_sequence("AUGC") == ["A", "U", "G", "C"]


# ---- DNA / RNA: optional 5'/3' directionality labels -----------------


@pytest.mark.parametrize("seq", [
    "5'-ATGC-3'",      # standard form
    "5'ATGC3'",        # no internal dashes
    "5' ATGC 3'",      # spaces instead of dashes
    "5'  -  ATGC  -  3'",   # mixed whitespace and dashes
])
def test_parse_dna_explicit_5to3_same_as_bare(seq):
    """Explicit 5'-...-3' labels are a no-op compared to bare letters --
    they just confirm the user's intent.  Covers a few common spacing /
    punctuation variants."""
    assert parse_dna_sequence(seq) == parse_dna_sequence("ATGC")


def test_parse_dna_3to5_reverses_to_match_bare_form():
    """3'-CGTA-5' is the same chemical polymer as 5'-ATGC-3'.  The
    parser reverses the residue list so the backend (which always
    builds 5' -> 3') produces a chain matching the user's stated
    direction."""
    assert parse_dna_sequence("3'-CGTA-5'") == parse_dna_sequence("ATGC")


def test_parse_rna_3to5_reverses_too():
    assert parse_rna_sequence("3'-CGUA-5'") == parse_rna_sequence("AUGC")


@pytest.mark.parametrize("bad, reason", [
    ("5'-ATGC",      "one-sided 5' label"),
    ("ATGC-3'",     "one-sided 3' label"),
    ("5'-ATGC-5'",  "two 5' ends -- a polymer has one of each"),
    ("3'-ATGC-3'",  "two 3' ends -- a polymer has one of each"),
])
def test_parse_dna_directionality_errors(bad, reason):
    """Self-contradictory or one-sided labels are unambiguously bugs;
    raise rather than silently picking a direction."""
    with pytest.raises(ValueError):
        parse_dna_sequence(bad)


def test_build_dna_3to5_label_produces_same_polymer_as_bare():
    """End-to-end check: build_dna('3'-CGTA-5'') and build_dna('ATGC')
    must produce the same atomic structure (modulo title)."""
    pytest.importorskip("rdkit")
    from molbuilder import build_dna
    s_bare = build_dna("ATGC", backend="rdkit", add_hydrogens=False,
                       protonate_phosphates=False)
    s_rev  = build_dna("3'-CGTA-5'", backend="rdkit", add_hydrogens=False,
                       protonate_phosphates=False)
    assert s_bare.elements == s_rev.elements, (
        "Reverse-direction label should produce the same residue order "
        "as bare; got divergent element lists"
    )
    assert s_bare.n_atoms == s_rev.n_atoms
