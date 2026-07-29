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
# ``auto_backend_name`` resolves "auto" -> the chosen backend; used to gate
# double-strand, which requires X3DNA.
from .builders.backends import auto_backend_name
from .residues import parse_dna_sequence, parse_rna_sequence, _strip_directionality
from .structure import Structure


def _strip_direction(s: str) -> str:
    """Return ONE DNA strand as a bare internal-5'->3' letter string.

    Delegates to :func:`residues._strip_directionality` -- the SAME strict
    5'/3' parser ``build_rna`` reaches through ``parse_rna_sequence`` -- so
    DNA and RNA treat end-labels identically:

      * ``5'-ATGC-3'`` -> ``ATGC`` (bare == explicit);
      * ``3'-ATGC-5'`` -> reversed to ``CGTA`` (internal 5'->3');
      * a one-sided (``5'-ATGC``) or self-contradictory (``3'-ATGC-3'``)
        label RAISES ``ValueError``.

    Before 2026-07-27 this used a permissive local regex that required BOTH
    ends labelled and never checked they differed, so it silently kept a
    one-sided label and silently reversed ``3'-ATGC-3'`` -- a real trap: a
    duplex strand quietly built in the wrong 5'->3' order pairs the wrong
    bases with no error.  ``build_rna``/``parse_dna_sequence`` already
    rejected these; ``build_dna`` now matches (its earlier ``_strip_direction``
    pass had stripped the markers before ``parse_dna_sequence``'s strict
    check could see them).
    """
    body, direction = _strip_directionality(s.strip())
    core = "".join(c for c in body if c.isalpha()).upper()
    return core[::-1] if direction == "3to5" else core


def _parse_dna_notation(text: str):
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
            raise ValueError(
                "a two-strand duplex takes exactly two comma-separated strands "
                "(\"strand1,strand2\", each 5'->3' by default); got "
                f"{len(parts)} comma-separated segments.")
        s1, s2 = _strip_direction(parts[0]), _strip_direction(parts[1])
        if not s1 or not s2:
            raise ValueError(
                "both strands of a two-strand duplex must be non-empty.")
        return "ds", s1, s2
    return mode, _strip_direction(s), None


def build_dna(
    sequence: str,
    *,
    backend: str = "auto",
    form: str = "B",
    terminal: str = "OH",
    add_hydrogens: "bool | str" = "auto",
    protonate_phosphates: bool = True,
    relax_clashes: bool = False,
    title: Optional[str] = None,
) -> Structure:
    """Build a DNA polymer from a 1-letter sequence.

    A bare sequence builds a SINGLE strand; ``ds,<seq>`` builds a canonical
    Watson-Crick DUPLEX; two comma-separated strands (``"strand1,strand2"``) build
    an EXPLICIT, possibly mismatched duplex (X3DNA required; see
    ``_parse_dna_notation``).

    ``relax_clashes`` (explicit-duplex only): a mismatched base pair placed at the
    standard frame can interpenetrate.  By default such an overlap is DETECTED and
    a ``RuntimeWarning`` names the clashing pair (never silently emitted).  Set
    ``relax_clashes=True`` to instead relieve it with a short OpenBabel UFF
    minimization -- a cleaner starting geometry for a subsequent DFT relaxation.

    Parameters
    ----------
    sequence
        DNA sequence (``"ATGCATGC"``), or ``"ds,ATGCATGC"`` for a canonical
        double strand.  Direction markers (``5'-…-3'`` / ``3'-…-5'``) are
        accepted.  Whitespace is ignored.
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
    # Notation: bare -> single strand; "ds,<seq>" -> canonical WC duplex;
    # "strand1,strand2" -> explicit (possibly mismatched) duplex.
    mode, seq, seq2 = _parse_dna_notation(sequence)
    double_strand = (mode == "ds")
    explicit_duplex = seq2 is not None
    if double_strand:
        # GATE: any duplex needs X3DNA -- canonical via fiber, arbitrary via rebuild.
        resolved = backend if backend != "auto" else auto_backend_name()
        if resolved != "threedna":
            raise ValueError(
                "double-strand DNA requires X3DNA (the 'threedna' backend, which "
                "builds duplex geometry via fiber / rebuild).  "
                + ("It isn't installed / detected. " if resolved is None
                   else f"The selected backend is {resolved!r}. ")
                + "Install X3DNA from x3dna.org (non-commercial license).  "
                "Single strands work with any backend.")
        if explicit_duplex:
            # Arbitrary / mismatched duplex (rebuild): B-form only for now.
            if (form or "B").upper() != "B":
                raise ValueError(
                    f"two-strand (arbitrary / mismatched) duplexes are B-form only; "
                    f"got form={form!r}.  Use \"ds,<sequence>\" for an A- or Z-form "
                    f"canonical duplex (the complement is generated automatically).")
        elif (form or "B").upper() not in ("B", "A", "Z"):
            raise ValueError(
                f"double-strand DNA supports B / A / Z helix form; got "
                f"form={form!r}.  (Z requires an alternating poly-d(GC) sequence.)")

    # parse_dna_sequence returns 3-letter codes (DA, DT, DG, DC).
    # Backends want a 1-letter sequence, so strip the "D" prefix.
    codes = parse_dna_sequence(seq)
    cleaned = "".join(c[1] for c in codes)
    cleaned2 = None
    if explicit_duplex:
        codes2 = parse_dna_sequence(seq2)
        cleaned2 = "".join(c[1] for c in codes2)
        if len(cleaned) != len(cleaned2):
            raise ValueError(
                f"the two strands of a duplex must be equal length (a fully "
                f"base-paired duplex; overhangs / bulges aren't supported yet): "
                f"strand1={len(cleaned)} nt, strand2={len(cleaned2)} nt.")
    struct = dispatch(
        "dna", cleaned,
        backend=backend, form=form, terminal=terminal, title=title,
        double_strand=double_strand, strand2=cleaned2,
    )
    struct = _maybe_add_hydrogens(struct, add_hydrogens)
    struct = _maybe_protonate(struct, protonate_phosphates)
    if explicit_duplex:
        # A mismatched pair placed at the standard frame can interpenetrate --
        # detect it (warn) or relieve it (opt-in).  Runs on the FINAL structure
        # (post H-add / protonation) so the clash reflects the atoms DFT sees.
        struct = _handle_duplex_clashes(struct, relax_clashes)
    return struct


# Two thresholds (Angstrom), keyed to what actually harms a downstream DFT/SIESTA
# relaxation -- not to reaching an ideal geometry (SIESTA does that):
#   _CLASH_WARN_A  -- an inter-residue contact this close is a real steric overlap
#     (a mismatched base pair at the WC frame).  Below the tightest legitimate
#     canonical contact (~1.5 A adjacent-residue H-H) and a WC H-bond (~1.8 A), so
#     no false positives on clash-free duplexes.  Flag it so the user knows.
#   _CLASH_DANGER_A -- NEAR-COINCIDENT atoms (nuclei almost on top of each other):
#     THIS is what makes the first SCF step explode.  The opt-in relief only needs
#     to clear this line; a merely-tight contact above it, SIESTA relaxes fine
#     (user guidance 2026-07-14).
_CLASH_WARN_A   = 1.3
_CLASH_DANGER_A = 1.0


def _handle_duplex_clashes(struct: Structure, relax_clashes: bool) -> Structure:
    """Detect (warn) or relieve (opt-in) steric clashes in an explicit duplex.

    Never silently emits a clashing structure.  A mismatched base pair placed at
    the standard frame can interpenetrate; the danger for a subsequent relaxation
    is NEAR-COINCIDENT atoms (explosive first SCF step), not a merely imperfect
    geometry.  So:
      - ``relax_clashes=False`` (default): a ``RuntimeWarning`` names the clashing
        pair and, if it's near-coincident, flags the explosion risk.
      - ``relax_clashes=True``: a short force-field minimization pushes atoms apart
        to a SIESTA-relaxable distance (clears near-coincidence); a residual warning
        fires only if some pair is STILL near-coincident (< ``_CLASH_DANGER_A``).
    """
    import warnings
    from .chemistry import min_nonbonded_contact, relieve_clashes

    d, i, j = min_nonbonded_contact(struct)
    if d is None or d >= _CLASH_WARN_A:
        return struct                      # clash-free (canonical / spaced pairs)

    def _lab(k: int) -> str:
        ch  = struct.chain_ids[k]   if struct.chain_ids   is not None else "?"
        rid = struct.residue_ids[k] if struct.residue_ids is not None else "?"
        nm  = (struct.atom_names[k] if struct.atom_names  is not None
               else struct.elements[k])
        return f"{ch}{rid}:{nm}"

    if relax_clashes:
        struct = relieve_clashes(struct)
        d2, _, _ = min_nonbonded_contact(struct)
        if d2 is not None and d2 < _CLASH_DANGER_A:
            warnings.warn(
                f"Relieved clashes but a residual NEAR-COINCIDENT contact remains "
                f"({d2:.2f} A) -- an unusually dense run of mismatches; inspect or "
                f"minimize further before relaxing the geometry.",
                RuntimeWarning, stacklevel=3)
        return struct

    danger = ("  These atoms are NEAR-COINCIDENT -- relaxing as-is risks an "
              "explosive first SCF step. " if d < _CLASH_DANGER_A else "  ")
    warnings.warn(
        f"This duplex has a steric CLASH at {_lab(i)}<->{_lab(j)} ({d:.2f} A) -- a "
        f"non-Watson-Crick (mismatched) base pair at the standard frame." + danger
        + "Rebuild with relax_clashes=True (a short force-field minimization that "
        "removes the near-coincidence; SIESTA relaxes the rest) or minimize "
        "externally before a geometry optimization.",
        RuntimeWarning, stacklevel=3)
    return struct


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
