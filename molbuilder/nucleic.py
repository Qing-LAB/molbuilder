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
    add_hydrogens: "bool | str" = "auto",
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
        Tri-state H-addition control:

          * ``"auto"`` (default) -- heuristic: skip if H/heavy >= 0.5,
            else add via :func:`chemistry.add_hydrogens` (OpenBabel
            preferred, RDKit fallback).  Works well for the canonical
            backends (X3DNA's heavy skeleton -> add; amber/rdkit
            H-complete output -> skip) but the 0.5 cutoff is a
            heuristic and should not be load-bearing for a user who
            knows what they want.
          * ``"on"`` -- always run ``chemistry.add_hydrogens``.  Use
            when you've loaded a partially-protonated structure that
            the heuristic might mis-classify, or when you want to
            guarantee H-completeness regardless of backend.
          * ``"off"`` -- never add.  Use when you want to inspect the
            heavy-atom skeleton or hand off to an external protonator.

        ``True`` (legacy) is normalised to ``"auto"``; ``False`` to
        ``"off"``.  No silent change of behaviour for callers using
        the previous bool API.
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
    add_hydrogens: "bool | str" = "auto",
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


def _normalise_h_mode(value: "bool | str") -> str:
    """Map the user's ``add_hydrogens`` argument to {auto, on, off}.

    Accepts the modern string API (``"auto"`` / ``"on"`` / ``"off"``)
    plus the legacy bool API (True -> "auto", False -> "off") for
    back-compat with callers written before the tri-state landed.
    Raises ValueError on anything else.
    """
    if value is True:
        return "auto"
    if value is False:
        return "off"
    if isinstance(value, str) and value.lower() in ("auto", "on", "off"):
        return value.lower()
    raise ValueError(
        f"add_hydrogens must be 'auto'/'on'/'off' (or bool); "
        f"got {value!r}"
    )


def _maybe_add_hydrogens(struct: Structure,
                         mode: "bool | str") -> Structure:
    """Apply chemistry.add_hydrogens based on the user's ``mode``.

    Tri-state semantics (see ``build_dna`` for the user-facing kwarg):

      * ``"off"``  -- return ``struct`` unchanged.
      * ``"on"``   -- always call :func:`chemistry.add_hydrogens`,
                      regardless of how many H the structure already
                      carries.
      * ``"auto"`` -- size-aware heuristic:
                        skip if H/heavy >= 0.5 (canonical backend
                        output: amber 0.63, rdkit 0.72 -- already
                        complete);
                        otherwise add (X3DNA's fiber-skeleton at
                        ratio ~0.05 lands here, as does a partially-
                        protonated user-loaded structure).

    Heuristic caveat: H/heavy depends on molecular size and chemistry
    -- small organics sit at ratio ~1-4, large polymers at ~0.6-0.8,
    metal complexes can be much lower without being broken.  The
    heuristic IS NOT a correctness check; it's a default for the
    common case where the user doesn't want to think about it.  When
    the structure lands in the gray zone (~0.3-0.6), prefer ``"on"``
    or ``"off"`` over ``"auto"`` so the decision is explicit.
    """
    norm = _normalise_h_mode(mode)
    if norm == "off":
        return struct
    if norm == "on":
        from .chemistry import add_hydrogens as _add_hydrogens
        return _add_hydrogens(struct)
    # auto
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
