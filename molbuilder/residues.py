"""Residue name handling: 1-letter codes + bracketed escapes.

Grammar (deliberately tiny):

    sequence  = (oneletter | bracketed | whitespace)*
    oneletter = a single ASCII letter, case-insensitive
    bracketed = "[" 3-or-4-letter-code "]"
    whitespace is ignored everywhere

Examples:

    "ARNDC"           ->  Ala-Arg-Asn-Asp-Cys
    "AR[SEP]C"        ->  Ala-Arg-phosphoSer-Cys
    "A R N D C"       ->  Ala-Arg-Asn-Asp-Cys     (whitespace ignored)
    "atgc"            ->  DA-DT-DG-DC             (DNA, case-insensitive)
    "ATG[5MC]C"       ->  DA-DT-DG-5-methyl-C-DC  (when registered)

Inside brackets you can use any standard 3-letter PDB code OR any
modified-residue code registered in :data:`MODIFIED_RESIDUES`.  Outside
brackets, only single ASCII letters are allowed; dashes, plus-signs,
and friends are rejected so the parser stays unambiguous.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------- #
#  Standard amino acids                                                  #
# ---------------------------------------------------------------------- #

# 3-letter -> 1-letter
AA_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA_ONE_TO_THREE = {v: k for k, v in AA_THREE_TO_ONE.items()}


# ---------------------------------------------------------------------- #
#  Modified residues                                                     #
# ---------------------------------------------------------------------- #
# The "patch" describes how to derive the modified residue from a parent
# standard residue:
#   parent      -- the AA we ask PeptideBuilder to place first
#   add_atoms   -- list of (atom_name, element, attach_to, dx, dy, dz)
#                  added in the parent's local frame (Angstrom)
#   remove_atoms -- atom names to delete from the parent (e.g. "HG" of Ser
#                   when phosphorylating)
#
# Coordinates are *relative* offsets from the named anchor atom -- close
# enough to give SIESTA / PySCF a sane starting structure for relaxation.
MODIFIED_RESIDUES = {
    # ---- phosphorylated serine / threonine / tyrosine -----------------
    "SEP": {
        "parent": "SER",
        "long":   "phosphoserine",
        "remove_atoms": ["HG"],
        "add_atoms": [
            ("P",   "P", "OG",  1.59,  0.00,  0.00),
            ("O1P", "O", "OG",  2.10,  1.40,  0.00),
            ("O2P", "O", "OG",  2.10, -0.70,  1.21),
            ("O3P", "O", "OG",  2.10, -0.70, -1.21),
        ],
    },
    "TPO": {
        "parent": "THR",
        "long":   "phosphothreonine",
        "remove_atoms": ["HG1"],
        "add_atoms": [
            ("P",   "P", "OG1", 1.59,  0.00,  0.00),
            ("O1P", "O", "OG1", 2.10,  1.40,  0.00),
            ("O2P", "O", "OG1", 2.10, -0.70,  1.21),
            ("O3P", "O", "OG1", 2.10, -0.70, -1.21),
        ],
    },
    "PTR": {
        "parent": "TYR",
        "long":   "phosphotyrosine",
        "remove_atoms": ["HH"],
        "add_atoms": [
            ("P",   "P", "OH",  1.59,  0.00,  0.00),
            ("O1P", "O", "OH",  2.10,  1.40,  0.00),
            ("O2P", "O", "OH",  2.10, -0.70,  1.21),
            ("O3P", "O", "OH",  2.10, -0.70, -1.21),
        ],
    },
    # ---- methylated lysines ------------------------------------------
    "MLY": {
        "parent": "LYS",
        "long":   "N-methyl-lysine",
        "remove_atoms": ["HZ3"],
        "add_atoms": [
            ("CH3", "C", "NZ",  1.47,  0.0, 0.0),
            ("HM1", "H", "NZ",  1.99,  0.96, 0.0),
            ("HM2", "H", "NZ",  1.99, -0.48, 0.83),
            ("HM3", "H", "NZ",  1.99, -0.48,-0.83),
        ],
    },
    "M3L": {
        "parent": "LYS",
        "long":   "N,N,N-trimethyl-lysine",
        "remove_atoms": ["HZ1", "HZ2", "HZ3"],
        "add_atoms": [
            ("CH1", "C", "NZ",  1.47,  0.0,  0.0),
            ("CH2", "C", "NZ", -0.48,  1.40, 0.0),
            ("CH3", "C", "NZ", -0.48, -0.70, 1.21),
        ],
    },
    # ---- acetylated lysine -------------------------------------------
    "ALY": {
        "parent": "LYS",
        "long":   "N6-acetyl-lysine",
        "remove_atoms": ["HZ3"],
        "add_atoms": [
            ("CH",  "C", "NZ",  1.34,  0.0,  0.0),
            ("OAC", "O", "NZ",  1.93,  1.05, 0.0),
            ("CM",  "C", "NZ",  2.05, -1.20, 0.0),
            ("HAC", "H", "NZ",  3.13, -1.05, 0.0),
        ],
    },
}


def is_modified_residue(name: str) -> bool:
    return name.upper() in MODIFIED_RESIDUES


# ---------------------------------------------------------------------- #
#  DNA / RNA bases                                                       #
# ---------------------------------------------------------------------- #

DNA_ONE_TO_THREE = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}
DNA_THREE_TO_ONE = {v: k for k, v in DNA_ONE_TO_THREE.items()}

RNA_ONE_TO_THREE = {"A": "A",  "U": "U",  "G": "G",  "C": "C"}
RNA_THREE_TO_ONE = {v: k for k, v in RNA_ONE_TO_THREE.items()}


# ---------------------------------------------------------------------- #
#  Sequence parsing                                                      #
# ---------------------------------------------------------------------- #

def _parse(
    sequence: str,
    one_to_three: dict,
    three_table: set,
    modified: dict | None,
    kind: str,
) -> List[str]:
    """Walk the string left-to-right.

    Outside ``[...]``: each character must be an ASCII letter and a
    known 1-letter code (case-insensitive).  Whitespace is skipped.
    Inside ``[...]``: a 3- or 4-letter PDB / modified-residue code.
    Anything else raises a clear error.
    """
    out: List[str] = []
    pos = 0
    n = len(sequence)
    while pos < n:
        ch = sequence[pos]
        if ch.isspace():
            pos += 1
            continue
        if ch == "[":
            close = sequence.find("]", pos)
            if close == -1:
                raise ValueError(
                    f"Unclosed '[' at position {pos} in {sequence!r}"
                )
            tok = sequence[pos + 1:close].upper().strip()
            if tok in three_table or (modified and tok in modified):
                out.append(tok)
            else:
                raise ValueError(
                    f"Unknown {kind} 3-letter code in brackets: {tok!r}"
                )
            pos = close + 1
            continue
        if not ch.isalpha():
            raise ValueError(
                f"Unexpected character {ch!r} at position {pos} in "
                f"{sequence!r} (use a single 1-letter code, [3LET] for "
                f"modified/non-standard residues, or whitespace)"
            )
        u = ch.upper()
        if u not in one_to_three:
            raise ValueError(f"Unknown {kind} 1-letter code: {ch!r}")
        out.append(one_to_three[u])
        pos += 1
    return out


def parse_peptide_sequence(sequence: str) -> List[str]:
    """1-letter peptide sequence (with [SEP]/[TPO]/... for modified)."""
    return _parse(
        sequence, AA_ONE_TO_THREE, set(AA_THREE_TO_ONE),
        MODIFIED_RESIDUES, "amino-acid",
    )


# Optional 5'/3' directionality labels on nucleic-acid input.  Bare
# letters ("ATGC") follow biology convention: 5' on the left, 3' on
# the right.  Explicit labels ("5'-ATGC-3'") confirm intent without
# changing behaviour.  Reverse-direction input ("3'-CGTA-5'") is
# normalised by reversing the body so backends -- which all build
# 5'->3' internally -- produce a polymer matching the user's stated
# direction.  One-sided or self-contradictory labels raise.
import re as _re

_DIR_PREFIX_RE = _re.compile(r"^\s*([35])'\s*-?\s*")
_DIR_SUFFIX_RE = _re.compile(r"\s*-?\s*([35])'\s*$")


def _strip_directionality(seq: str) -> tuple:
    """Pull off optional 5'/3' end-labels from a nucleic-acid sequence.

    Returns ``(body, direction)`` where ``direction`` is one of:
      * ``None``     -- bare letters; assume 5'->3' (biology default).
      * ``"5to3"``   -- explicit 5'-...-3'.  Same effect as bare.
      * ``"3to5"``   -- explicit 3'-...-5'.  Caller should reverse the
                        parsed residue list before handing to a backend.
    Raises ValueError on:
      * one-sided labels ("5'-ATGC" or "ATGC-3'");
      * matching labels at both ends ("5'-ATGC-5'" / "3'-ATGC-3'").
    """
    pre = _DIR_PREFIX_RE.match(seq)
    suf = _DIR_SUFFIX_RE.search(seq)
    if pre is None and suf is None:
        return seq, None
    if pre is None or suf is None:
        raise ValueError(
            f"One-sided directionality label in {seq!r}: both ends must "
            f"be labelled or neither (write '5'-ATGC-3'', '3'-CGTA-5'', "
            f"or just 'ATGC' for the implicit 5'->3' default)"
        )
    pre_d, suf_d = pre.group(1), suf.group(1)
    if pre_d == suf_d:
        raise ValueError(
            f"Self-contradictory directionality in {seq!r}: both ends "
            f"labelled {pre_d}' (a polymer has one 5' end and one 3' end, "
            f"not two of the same)"
        )
    # Trim from end-of-prefix to start-of-suffix.  search() with $ in
    # the pattern guarantees suf.start() is past the end of the body.
    body = seq[pre.end():suf.start()].strip().strip("-").strip()
    return body, f"{pre_d}to{suf_d}"


def parse_dna_sequence(sequence: str) -> List[str]:
    """1-letter DNA sequence (A/T/G/C, case-insensitive).

    Optional 5'/3' end-labels are accepted: ``"5'-ATGC-3'"`` (explicit
    5'->3', same as bare ``"ATGC"``) or ``"3'-CGTA-5'"`` (reverse-
    direction; the residue list is reversed so the resulting polymer
    matches the user's stated direction).
    """
    body, direction = _strip_directionality(sequence)
    codes = _parse(
        body, DNA_ONE_TO_THREE, set(DNA_THREE_TO_ONE),
        None, "DNA base",
    )
    return list(reversed(codes)) if direction == "3to5" else codes


def parse_rna_sequence(sequence: str) -> List[str]:
    """1-letter RNA sequence (A/U/G/C, case-insensitive).

    Accepts the same optional 5'/3' end-labels as
    :func:`parse_dna_sequence`.
    """
    body, direction = _strip_directionality(sequence)
    codes = _parse(
        body, RNA_ONE_TO_THREE, set(RNA_THREE_TO_ONE),
        None, "RNA base",
    )
    return list(reversed(codes)) if direction == "3to5" else codes
