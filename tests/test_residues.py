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
