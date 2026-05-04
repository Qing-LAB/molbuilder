"""DNA / RNA polymer builders.

Public API:
    build_dna(sequence, *, backend='auto', form='B', terminal='OH')
    build_rna(sequence, *, backend='auto', form='A', terminal='OH')

All backends return the same :class:`molbuilder.structure.Structure`
type, so file writers, the FDF generator, and the web UI don't care
which backend produced the geometry.  See :mod:`molbuilder.backends`
for backend capabilities and install requirements.
"""

from __future__ import annotations

from typing import Optional

from .backends import dispatch
from .residues import parse_dna_sequence, parse_rna_sequence
from .structure import Structure


def build_dna(
    sequence: str,
    *,
    backend: str = "auto",
    form: str = "B",
    terminal: str = "OH",
    add_hydrogens: bool = True,
    protonate_phosphates: bool = True,
    title: Optional[str] = None,
) -> Structure:
    """Build a single-stranded DNA polymer from a 1-letter sequence.

    Parameters
    ----------
    sequence
        DNA sequence (``"ATGCATGC"``).  Whitespace is ignored.
    backend
        ``"auto"`` (default) -- pick the best installed backend
        (``threedna`` > ``amber`` > ``rdkit``).  Force one with
        ``"threedna"`` / ``"amber"`` / ``"rdkit"``.
    form
        Helical form: ``"B"`` (default), ``"A"``, or ``"Z"``.
        ``threedna`` honours all three; ``amber`` ignores form (always
        extended); ``rdkit`` always returns its own embedded conformer.
    terminal
        Terminal-phosphate state: ``"OH"`` (default; both ends -OH),
        ``"5P"``, ``"3P"``, or ``"PP"``.
    add_hydrogens
        If True (default), add explicit hydrogens via
        :func:`chemistry.add_hydrogens` (OpenBabel preferred, RDKit
        fallback).  Critical for X3DNA, whose ``fiber`` output is a
        heavy-atom skeleton -- DFT will compute the wrong electron
        count without H.  Amber and RDKit backends already produce
        H-complete structures, so this is a no-op for them.  Set
        False to keep the heavy-atom skeleton; you'll need to
        protonate it before any quantum-chemistry calculation.
    protonate_phosphates
        If True (default), add an H to each deprotonated non-bridging
        phosphate oxygen so the molecule is formally **neutral**.
        This is the easier starting point for DFT (no NetCharge to
        set, no oversized vacuum needed for charged-cell electrostatic
        compensation).  Set False to keep tleap's deprotonated state
        (more chemically realistic for solution-phase DNA, but the
        molecule then carries -(N-1) electrons and you must set
        ``NetCharge`` explicitly in the FDF -- ``render_fdf`` will
        emit it for you when it sees a non-zero charge).
    title
        Optional title written into XYZ comment / PDB TITLE.
    """
    # parse_dna_sequence returns 3-letter codes (DA, DT, DG, DC).
    # Backends want a 1-letter sequence, so strip the "D" prefix.
    codes = parse_dna_sequence(sequence)
    cleaned = "".join(c[1] for c in codes)
    struct = dispatch(
        "dna", cleaned,
        backend=backend, form=form, terminal=terminal, title=title,
    )
    struct = _maybe_add_hydrogens(struct, add_hydrogens)
    return _maybe_protonate(struct, protonate_phosphates)


def build_rna(
    sequence: str,
    *,
    backend: str = "auto",
    form: str = "A",
    terminal: str = "OH",
    add_hydrogens: bool = True,
    protonate_phosphates: bool = True,
    title: Optional[str] = None,
) -> Structure:
    """Build a single-stranded RNA polymer from a 1-letter sequence.

    See :func:`build_dna` for parameter descriptions.  Default
    ``form="A"`` because A-form is the canonical RNA helix.
    """
    codes = parse_rna_sequence(sequence)        # already 1-letter for RNA
    cleaned = "".join(codes)
    struct = dispatch(
        "rna", cleaned,
        backend=backend, form=form, terminal=terminal, title=title,
    )
    struct = _maybe_add_hydrogens(struct, add_hydrogens)
    return _maybe_protonate(struct, protonate_phosphates)


def _maybe_add_hydrogens(struct: Structure, add: bool) -> Structure:
    """Run chemistry.add_hydrogens if the user wants H, but skip the
    round-trip when the structure already has a sensible H count
    (Amber/RDKit backends produce H-complete output).

    Threshold rationale: organic molecules sit at H/heavy ~ 0.6-1.5;
    nucleic acids ~ 0.6; peptides ~ 0.7-0.9.  We gate at 0.5 so:

      * X3DNA fiber's near-zero-H output         (ratio ~0.05) -> add
      * a partially-protonated user input        (ratio ~0.4)  -> add
      * fully-built amber tleap output           (ratio ~0.63) -> skip
      * fully-built rdkit nucleic SMILES output  (ratio ~0.72) -> skip

    Pre-fix the gate was at 0.3, which silently skipped partial cases
    (a user-loaded structure missing N-H or amine H would slip through
    with no auto-fix).  The Layer-1 h_ratio validator uses < 0.3 -- so
    a structure between 0.3 and 0.5 would have been doubly missed.
    Widening to 0.5 closes that gap without false-positive-ing on any
    canonical backend output.
    """
    if not add:
        return struct
    n_h     = sum(1 for e in struct.elements if e == "H")
    n_heavy = sum(1 for e in struct.elements if e != "H")
    if n_heavy and (n_h / n_heavy) >= 0.5:
        return struct
    from .chemistry import add_hydrogens as _add_hydrogens
    return _add_hydrogens(struct)


def _maybe_protonate(struct: Structure, protonate: bool) -> Structure:
    if not protonate:
        return struct
    from .chemistry import protonate_phosphate_oxygens
    new_struct, _ = protonate_phosphate_oxygens(struct)
    return new_struct
