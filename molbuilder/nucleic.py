"""DNA / RNA polymer builders: :func:`build_dna`, :func:`build_rna`.

What may be asked -- the forms, terminals and backends -- is the registry's
(:mod:`molbuilder.builders.backends`), and every backend returns the same
:class:`molbuilder.structure.Structure`, so file writers, the deck writers
and the web UI don't care which one built the geometry
(`engines/builders.md` § 3-§ 5).
"""

from __future__ import annotations

from typing import Optional

from .builders.backends import (check_ask, dispatch, duplex_backend,
                                resolved_backend)
from .chemistry import BuildRefused
from .residues import DNA_THREE_TO_ONE, parse_dna_sequence, parse_rna_sequence
from .structure import Structure


def _strip_direction(s: str) -> str:
    """Return ONE DNA strand as a bare internal-5'->3' letter string, read
    by THE ONE GRAMMAR whole (:func:`residues.parse_dna_sequence`) -- the
    same strict parser ``build_rna`` and the peptide builder reach, so a DNA
    strand refuses what they refuse (`engines/builders.md` § 2):

      * ``5'-ATGC-3'`` -> ``ATGC`` (bare == explicit);
      * ``3'-ATGC-5'`` -> reversed to ``CGTA`` (internal 5'->3');
      * a one-sided (``5'-ATGC``) or self-contradictory (``3'-ATGC-3'``)
        label is refused (``BuildRefused``): a duplex strand quietly built
        in the wrong 5'->3' order pairs the wrong bases with no error;
      * a dash, digit or bracket the grammar does not take is refused,
        naming its position -- never dropped to leave letters that build.
    """
    return "".join(DNA_THREE_TO_ONE[code]
                   for code in parse_dna_sequence(s.strip()))


def parse_dna_notation(text: str):
    """Parse the DNA-panel input notation (docs/web/tabs.md).

    Returns ``(mode, strand1, strand2)``:

    - bare sequence (``"ATGC"``)      -> ``("ss", "ATGC", None)``  single strand.
    - ``"ds,ATGC"`` / ``"ds:ATGC"``   -> ``("ds", "ATGC", None)``  canonical Watson-
      Crick duplex; the complementary strand is generated automatically (fiber).
    - two comma-separated strands     -> ``("ds", "ATGC", "GCAA")``  an EXPLICIT,
      possibly MISMATCHED duplex (X3DNA ``rebuild``).  Both strands are 5'->3' by
      default and paired ANTIPARALLEL.  The ``ds,`` prefix is optional here (two
      strands already imply a duplex): ``"ATGC,GCAA"`` == ``"ds,ATGC,GCAA"``.

    Direction markers are accepted per strand: ``5'-ATGC-3'`` (default) or
    ``3'-ATGC-5'`` (reversed to internal 5'->3').
    """
    s = (text or "").strip()
    mode = "ss"
    if s[:3].lower() in ("ds,", "ds:"):
        mode, s = "ds", s[3:].strip()
    if "," in s:
        # Two explicit strands => an arbitrary duplex (regardless of a ds prefix).
        parts = s.split(",")
        if len(parts) != 2:
            raise BuildRefused(
                "a two-strand duplex takes exactly two comma-separated strands "
                "(\"strand1,strand2\", each 5'->3' by default); got "
                f"{len(parts)} comma-separated segments.")
        return "ds", _strip_direction(parts[0]), _strip_direction(parts[1])
    return mode, _strip_direction(s), None


def build_dna(
    sequence: str,
    *,
    backend: str = "auto",
    form: Optional[str] = None,
    terminal: Optional[str] = None,
    add_hydrogens: str = "auto",
    protonate_phosphates: bool = True,
    relax_clashes: bool = False,
    title: Optional[str] = None,
) -> Structure:
    """Build a DNA polymer from a 1-letter sequence.

    A bare sequence builds a SINGLE strand; ``ds,<seq>`` builds a canonical
    Watson-Crick DUPLEX; two comma-separated strands (``"strand1,strand2"``) build
    an EXPLICIT, possibly mismatched duplex (on a backend that builds one; see
    ``parse_dna_notation``).

    ``relax_clashes`` (explicit-duplex only): a mismatched base pair placed at the
    standard frame can interpenetrate.  Set ``relax_clashes=True`` to relieve it
    with a short OpenBabel UFF minimization -- a cleaner starting geometry for a
    subsequent DFT relaxation.  A clash not relieved is said by the check
    after the build (`validation.builders.check_built`), naming the pair, and
    a near-coincident contact left after relief is said too -- through the
    build door every surface uses (`builders.build`); this function alone
    returns the structure.

    Parameters
    ----------
    sequence
        DNA sequence (``"ATGCATGC"``), or ``"ds,ATGCATGC"`` for a canonical
        double strand.  Direction markers (``5'-…-3'`` / ``3'-…-5'``) are
        accepted.  Whitespace is ignored.
    backend
        ``"auto"`` (default) -- the first installed of the registry's
        backends (``backends.BACKENDS``, in order) -- or one of them.
    form
        A helical form the kind is built in (``backends.KIND_FORMS``), or
        ``None`` for the kind's default; what a backend builds of it is its
        own (``backends.built``), said after the build.
    terminal
        A terminal state of ``backends.TERMINALS``, or ``None`` for the
        default, both ends -OH (``backends.DEFAULT_TERMINAL``).
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

    protonate_phosphates
        If True (default), add an H to each deprotonated non-bridging
        phosphate oxygen so the molecule is formally **neutral**.
        This is the easier starting point for DFT (no oversized vacuum
        needed for charged-cell electrostatic compensation).  Set False
        to keep tleap's deprotonated state (more chemically realistic
        for solution-phase DNA, but the molecule then carries -(N-1)
        electrons: a blank ``net_charge`` counts the deprotonated
        phosphates (`model/chemistry.md` § 1), and the deck states it).
    title
        Optional title written into XYZ comment / PDB TITLE.
    """
    # WHAT MAY BE ASKED, refused first, naming the words it takes
    # (`backends.check_ask`; `engines/builders.md` § 1) -- before any
    # backend runs, as are the hydrogen mode and the true-or-false asks.
    check_ask("dna", form, terminal, backend)
    add_hydrogens = normalise_h_mode(add_hydrogens)
    protonate_phosphates = _yes_or_no("protonate_phosphates",
                                      protonate_phosphates)
    relax_clashes = _yes_or_no("relax_clashes", relax_clashes)
    # Notation: bare -> single strand; "ds,<seq>" -> canonical WC duplex;
    # "strand1,strand2" -> explicit (possibly mismatched) duplex -- each
    # strand read by the one grammar, once (`_strip_direction`).
    mode, seq, seq2 = parse_dna_notation(sequence)
    double_strand = (mode == "ds")
    explicit_duplex = seq2 is not None
    if double_strand:
        # A DUPLEX ON A BACKEND THAT BUILDS ONE (`backends.duplex_backend`);
        # an explicit duplex's own rules are that backend's to say.
        backend = duplex_backend(resolved_backend(backend))
    struct = dispatch(
        "dna", seq,
        backend=backend, form=form, terminal=terminal, title=title,
        double_strand=double_strand, strand2=seq2,
    )
    struct = _maybe_add_hydrogens(struct, add_hydrogens)
    struct = _maybe_protonate(struct, protonate_phosphates)
    if explicit_duplex and relax_clashes:
        # A mismatched pair placed at the standard frame can interpenetrate --
        # relieved when asked and only where it clashes (`chemistry.
        # steric_clash`: a clash-free duplex is emitted as built), on the
        # FINAL structure (post H-add / protonation) so the relief acts on
        # the atoms DFT sees; an overlap left is the check after the build's
        # to say.
        from .chemistry import relieve_clashes, steric_clash
        if steric_clash(struct):
            struct = relieve_clashes(struct)
    return struct


def build_rna(
    sequence: str,
    *,
    backend: str = "auto",
    form: Optional[str] = None,
    terminal: Optional[str] = None,
    add_hydrogens: str = "auto",
    protonate_phosphates: bool = True,
    title: Optional[str] = None,
) -> Structure:
    """Build a single-stranded RNA polymer from a 1-letter sequence.

    See :func:`build_dna` for parameter descriptions; RNA is built in its
    A-form, the canonical RNA helix (``backends.KIND_FORMS``).
    """
    check_ask("rna", form, terminal, backend)
    add_hydrogens = normalise_h_mode(add_hydrogens)
    protonate_phosphates = _yes_or_no("protonate_phosphates",
                                      protonate_phosphates)
    struct = dispatch(
        "rna", "".join(parse_rna_sequence(sequence)),
        backend=backend, form=form, terminal=terminal, title=title,
    )
    struct = _maybe_add_hydrogens(struct, add_hydrogens)
    return _maybe_protonate(struct, protonate_phosphates)


def _yes_or_no(name: str, value) -> bool:
    """A true-or-false ask (``protonate_phosphates``, ``relax_clashes``) as
    it is; refused (:class:`chemistry.BuildRefused`) on anything else -- a
    ``"false"`` sent as text would read as true."""
    if isinstance(value, bool):
        return value
    raise BuildRefused(f"{name} takes true or false; got {value!r}")


def normalise_h_mode(value: str) -> str:
    """The user's ``add_hydrogens`` argument as one of {auto, on, off};
    refused (:class:`chemistry.BuildRefused`) on anything else."""
    if isinstance(value, str) and value.lower() in ("auto", "on", "off"):
        return value.lower()
    raise BuildRefused(f"add_hydrogens must be 'auto', 'on' or 'off'; "
                     f"got {value!r}")


def _maybe_add_hydrogens(struct: Structure,
                         mode: str) -> Structure:
    """Apply chemistry.add_hydrogens by ``mode``, the user's
    ``add_hydrogens`` as :func:`normalise_h_mode` read it before the build.

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
                        ratio ~0.05 lands here).

    Heuristic caveat: H/heavy depends on molecular size and chemistry
    -- small organics sit at ratio ~1-4, large polymers at ~0.6-0.8,
    metal complexes can be much lower without being broken.  The
    heuristic IS NOT a correctness check; it's a default for the
    common case where the user doesn't want to think about it.  When
    the structure lands in the gray zone (~0.3-0.6), prefer ``"on"``
    or ``"off"`` over ``"auto"`` so the decision is explicit.
    """
    if mode == "off":
        return struct
    if mode == "on":
        from .chemistry import add_hydrogens as _add_hydrogens
        return _add_hydrogens(struct)
    # auto -- a hydrogen asked the one way (`model/chemistry.md` § 3,
    # `chemistry.is_atom`), never by comparing a label.
    from .chemistry import add_hydrogens as _add_hydrogens
    from .chemistry import is_atom
    n_h     = sum(1 for e in struct.elements if is_atom(e, "H"))
    n_heavy = len(struct.elements) - n_h
    if n_heavy and (n_h / n_heavy) >= 0.5:
        return struct
    return _add_hydrogens(struct)


def _maybe_protonate(struct: Structure, protonate: bool) -> Structure:
    if not protonate:
        return struct
    from .chemistry import protonate_phosphate_oxygens
    new_struct, _ = protonate_phosphate_oxygens(struct)
    return new_struct
