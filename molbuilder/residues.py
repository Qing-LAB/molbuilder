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

from .chemistry import BuildRefused


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
#: EACH MODIFIED RESIDUE AS HEAVY ATOMS ADDED TO A STANDARD PARENT, placed
#: in the parent's local frame (`engines/builders.md` § 2), each code the
#: PDB Chemical Component Dictionary's own (files.rcsb.org, read
#: 2026-11-18: MLZ N-methyl-, MLY N-dimethyl-, M3L N-trimethyl-, ALY
#: N6-acetyl-lysine; SEP, TPO, PTR the phosphorylated Ser, Thr, Tyr):
#:   parent    -- the amino acid PeptideBuilder places first
#:   add_atoms -- (atom_name, element, (a, b, c), bond, angle, dihedral):
#:                the atom bonded to ``c`` at ``bond`` Angstrom, at
#:                ``angle`` degrees b-c-new, at ``dihedral`` degrees
#:                a-b-c-new -- PeptideBuilder's own placement
#:                (``calculateCoordinates``), the way it places every side
#:                chain; ``a``/``b``/``c`` name atoms of the residue already
#:                there, the parent's or one added above.
#: A STARTING GEOMETRY the DFT relaxation refines: the bond lengths of the
#: common phosphate (P-O ester 1.61, terminal 1.51), amine (N-C 1.47-1.49)
#: and amide (N-C 1.34, C=O 1.23, C-C 1.51) groups; the angles of each
#: atom's hybridisation (sp3 near 109.5, sp2 near 120); the phosphate's
#: three oxygens staggered about the ester bond, and threonine's turned off
#: its methyl (CA-CB-OG1-P 120: at 180 a phosphate hydrogen sits 1.08 A
#: from a methyl one, measured 2026-11-18).  NO HYDROGEN IS LISTED:
#: PeptideBuilder writes heavy atoms only, and `chemistry.add_hydrogens`
#: places every hydrogen, the modified residue's with the rest.
_PHOSPHATE_OXYGENS = ("O1P", "O2P", "O3P")


def _phosphate(on: str, a: str, b: str, *, angle: float,
               dihedral: float) -> list:
    """A phosphate group on ``on`` -- its P bonded there at ``angle``
    degrees b-``on``-P and ``dihedral`` degrees a-b-``on``-P -- and the
    three oxygens on P, staggered about the ester bond."""
    return ([("P", "P", (a, b, on), 1.61, angle, dihedral)]
            + [(name, "O", (b, on, "P"), 1.51, 106.0, d)
               for name, d in zip(_PHOSPHATE_OXYGENS, (180.0, 60.0, -60.0))])


def _methyls(*names: str) -> list:
    """Methyl carbons on lysine's NZ, staggered about CE-NZ."""
    return [(name, "C", ("CD", "CE", "NZ"), 1.49, 109.5, d)
            for name, d in zip(names, (180.0, 60.0, -60.0))]


MODIFIED_RESIDUES = {
    # ---- phosphorylated serine / threonine / tyrosine -----------------
    "SEP": {"parent": "SER",
            "add_atoms": _phosphate("OG", "CA", "CB",
                                    angle=118.0, dihedral=180.0)},
    "TPO": {"parent": "THR",
            "add_atoms": _phosphate("OG1", "CA", "CB",
                                    angle=118.0, dihedral=120.0)},
    # The aryl phosphate's P out of the ring's plane.
    "PTR": {"parent": "TYR",
            "add_atoms": _phosphate("OH", "CE1", "CZ",
                                    angle=120.0, dihedral=90.0)},
    # ---- methylated lysines ------------------------------------------
    "MLZ": {"parent": "LYS", "add_atoms": _methyls("CM")},
    "MLY": {"parent": "LYS", "add_atoms": _methyls("CH1", "CH2")},
    "M3L": {"parent": "LYS", "add_atoms": _methyls("CM1", "CM2", "CM3")},
    # ---- acetylated lysine -------------------------------------------
    "ALY": {"parent": "LYS",
            "add_atoms": [("CH",  "C", ("CD", "CE", "NZ"), 1.34, 122.0, 180.0),
                          ("OH",  "O", ("CE", "NZ", "CH"), 1.23, 122.5, 0.0),
                          ("CH3", "C", ("CE", "NZ", "CH"), 1.51, 116.0, 180.0)]},
}


# ---------------------------------------------------------------------- #
#  DNA / RNA bases                                                       #
# ---------------------------------------------------------------------- #

DNA_ONE_TO_THREE = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}
DNA_THREE_TO_ONE = {v: k for k, v in DNA_ONE_TO_THREE.items()}

RNA_ONE_TO_THREE = {"A": "A",  "U": "U",  "G": "G",  "C": "C"}
RNA_THREE_TO_ONE = {v: k for k, v in RNA_ONE_TO_THREE.items()}

#: EACH BASE'S BONDS, by PDB atom name: the atoms of its rings and exocyclic
#: groups, and those of their bonds that are DOUBLE in the canonical (amino,
#: keto) tautomer -- the base as it is in DNA and RNA; every other bond
#: between two of the atoms is single.  A base's bond orders are its
#: chemistry, not its geometry: read off a heavy-atom skeleton they are a
#: guess -- 3DNA's cytosine has C4-N3 1.34 A and C4-N4 1.33 A, and bond
#: perception took the imino tautomer, a hydrogen on N3 and one on N4
#: (measured 2026-10-31, every cytosine 3DNA built).  The hydrogens of a
#: built structure are placed on these orders (`chemistry.add_hydrogens`).
#: Thymine's methyl carbon is ``C7``, or ``C5M`` in older files.
NUCLEOBASE_BONDS = {
    "A": (("N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"),
          (("N1", "C6"), ("C2", "N3"), ("C4", "C5"), ("N7", "C8"))),
    "G": (("N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"),
          (("C6", "O6"), ("C2", "N3"), ("C4", "C5"), ("N7", "C8"))),
    "C": (("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
          (("C2", "O2"), ("N3", "C4"), ("C5", "C6"))),
    "T": (("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7", "C5M"),
          (("C2", "O2"), ("C4", "O4"), ("C5", "C6"))),
    "U": (("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
          (("C2", "O2"), ("C4", "O4"), ("C5", "C6"))),
}


#: AmberTools' terminal suffix on a nucleotide's residue name, by where the
#: nucleotide sits: the 5' end, the 3' end, inner, or alone -- free at both
#: ends (`DA5`, `DC3`, `DG`, `DAN`).  The AmberTools backend writes them,
#: :func:`nucleobase` reads them.
AMBER_SUFFIX = {"5'": "5", "3'": "3", "inner": "", "single": "N"}


def _base_of(name: str):
    """The one letter of a nucleotide named ``name`` exactly: DNA's ``DC``,
    RNA's ``C``, or a DNA nucleotide named without its ``D``, as older PDB
    files name thymidine ``T``."""
    return (DNA_THREE_TO_ONE.get(name) or RNA_THREE_TO_ONE.get(name)
            or DNA_THREE_TO_ONE.get("D" + name))


def nucleobase(residue_name: str):
    """The base of a nucleotide residue named ``residue_name`` -- DNA's
    ``DC``, RNA's ``C``, an older file's ``T``, with an AmberTools terminal
    suffix (:data:`AMBER_SUFFIX`: ``DC5``, ``C3``, ``DAN``) -- as its one
    letter (a key of :data:`NUCLEOBASE_BONDS`), or ``None`` for any other
    residue."""
    name = residue_name.strip().upper()
    base = _base_of(name)
    if base is None and name[-1:] in AMBER_SUFFIX.values() and name[-1:]:
        base = _base_of(name[:-1])
    return base


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
    Anything else is refused (:class:`chemistry.BuildRefused`), naming it,
    and so is a sequence that holds no residue.
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
                raise BuildRefused(
                    f"Unclosed '[' at position {pos} in {sequence!r}"
                )
            tok = sequence[pos + 1:close].upper().strip()
            if tok in three_table or (modified and tok in modified):
                out.append(tok)
            else:
                raise BuildRefused(
                    f"Unknown {kind} 3-letter code in brackets: {tok!r}"
                )
            pos = close + 1
            continue
        if not ch.isalpha():
            raise BuildRefused(
                f"Unexpected character {ch!r} at position {pos} in "
                f"{sequence!r} (use a single 1-letter code, [3LET] for "
                f"modified/non-standard residues, or whitespace)"
            )
        u = ch.upper()
        if u not in one_to_three:
            raise BuildRefused(f"Unknown {kind} 1-letter code: {ch!r}")
        out.append(one_to_three[u])
        pos += 1
    if not out:
        raise BuildRefused(f"No {kind} residue in the sequence: nothing to "
                           f"build.")
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
    Refused (:class:`chemistry.BuildRefused`) on:
      * one-sided labels ("5'-ATGC" or "ATGC-3'");
      * matching labels at both ends ("5'-ATGC-5'" / "3'-ATGC-3'").
    """
    pre = _DIR_PREFIX_RE.match(seq)
    suf = _DIR_SUFFIX_RE.search(seq)
    if pre is None and suf is None:
        return seq, None
    if pre is None or suf is None:
        raise BuildRefused(
            f"One-sided directionality label in {seq!r}: both ends must "
            f"be labelled or neither (write '5'-ATGC-3'', '3'-CGTA-5'', "
            f"or just 'ATGC' for the implicit 5'->3' default)"
        )
    pre_d, suf_d = pre.group(1), suf.group(1)
    if pre_d == suf_d:
        raise BuildRefused(
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
