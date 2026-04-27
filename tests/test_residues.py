"""Sequence-parser tests (no external deps)."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molbuilder.residues import (
    parse_dna_sequence,
    parse_peptide_sequence,
    parse_rna_sequence,
)


def main() -> None:
    # ---- peptides ----------------------------------------------------
    # Standard 1-letter
    assert parse_peptide_sequence("ARNDC") == \
        ["ALA", "ARG", "ASN", "ASP", "CYS"]
    # Case-insensitive
    assert parse_peptide_sequence("arndc") == \
        ["ALA", "ARG", "ASN", "ASP", "CYS"]
    # Whitespace ignored
    assert parse_peptide_sequence("A R N D C") == \
        ["ALA", "ARG", "ASN", "ASP", "CYS"]
    assert parse_peptide_sequence("AR NDC") == \
        ["ALA", "ARG", "ASN", "ASP", "CYS"]
    # Bracketed modified residue
    assert parse_peptide_sequence("AR[SEP]C") == \
        ["ALA", "ARG", "SEP", "CYS"]
    # Bracketed standard residue (allowed; just verbose)
    assert parse_peptide_sequence("[ALA][ARG][SEP][CYS]") == \
        ["ALA", "ARG", "SEP", "CYS"]
    # Multiple modified residues mixed with 1-letter
    assert parse_peptide_sequence("A[PTR][SEP][TPO]C") == \
        ["ALA", "PTR", "SEP", "TPO", "CYS"]

    # ---- DNA ---------------------------------------------------------
    assert parse_dna_sequence("ATGC") == ["DA", "DT", "DG", "DC"]
    assert parse_dna_sequence("atgc") == ["DA", "DT", "DG", "DC"]
    assert parse_dna_sequence("ATGC ATGC") == \
        ["DA", "DT", "DG", "DC", "DA", "DT", "DG", "DC"]
    # Bracketed DA also works
    assert parse_dna_sequence("[DA][DT]GC") == ["DA", "DT", "DG", "DC"]

    # ---- RNA ---------------------------------------------------------
    assert parse_rna_sequence("AUGC") == ["A", "U", "G", "C"]

    # ---- Bad inputs --------------------------------------------------
    bad_cases = [
        ("AXC",       "X isn't a valid AA letter"),
        ("ZZZ",       "Z isn't a valid AA letter"),
        ("Ala-Arg",   "dashes are no longer accepted"),
        ("AR(SEP)C",  "parens are no longer accepted (use brackets)"),
        ("AR[XXX]C",  "XXX isn't a known 3-letter code"),
        ("AR[SEPC",   "unclosed bracket"),
    ]
    for bad, why in bad_cases:
        try:
            parse_peptide_sequence(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}: {why}")

    print("OK -- 1-letter parser + [XXX] modified-residue escapes "
          "works for peptide / DNA / RNA, rejects malformed input.")


if __name__ == "__main__":
    main()
