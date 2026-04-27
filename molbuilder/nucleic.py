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
        (``amber`` > ``rdkit``).  Force one with
        ``"amber"`` / ``"rdkit"``.
    form
        Helical form: ``"B"`` (default), ``"A"``, or ``"Z"``.
        Only the ``amber`` backend respects the form argument;
        ``rdkit`` always returns its own embedded conformer.
        ``"Z"`` requires 3DNA externally.
    terminal
        Terminal-phosphate state: ``"OH"`` (default; both ends -OH),
        ``"5P"``, ``"3P"``, or ``"PP"``.
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
    return _maybe_protonate(struct, protonate_phosphates)


def build_rna(
    sequence: str,
    *,
    backend: str = "auto",
    form: str = "A",
    terminal: str = "OH",
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
    return _maybe_protonate(struct, protonate_phosphates)


def _maybe_protonate(struct: Structure, protonate: bool) -> Structure:
    if not protonate:
        return struct
    from .chemistry import protonate_phosphate_oxygens
    new_struct, _ = protonate_phosphate_oxygens(struct)
    return new_struct
