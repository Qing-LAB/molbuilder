"""The SIESTA family's output lines -- ONE table, every reader.

`model/parse.md` § 5d.5.  The parser reads these patterns and each line's one
reader; so do the cheap ending scan (`_run_ending`), the benchmark reader, the
TBtrans reader and the monitor, which imports this module from beside the job
(`runwrap.MONITOR_COMPANIONS`, `execution/run-reports.md` § 2.3); the
wrapper's SCF-timing tee, being shell, gets its pattern rendered from here.
Three hand-kept regexes for the one `scf:` row were how TranSIESTA's
`ts-scf:` loop went unseen by all three readers at once -- a device that ran
1000 NEGF iterations was timed as 7, and the monitor reported *"no SCF
progress"* for 7.6 hours (2026-09-25).

Every pattern is read off SIESTA 5.4.2's own writer, named beside it.  The
one older spelling kept, the 5.0 betas' ``Siesta Version``, is off that
build's own output (`tests/watch/fixtures/siesta_frozen/BDT_METAL-*`).
Stdlib only, and nothing of ours: it travels beside every job.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# ---- The SCF row, both phases ------------------------------------------------
#: SIESTA's periodic SCF and TranSIESTA's NEGF loop print one row per
#: iteration with one format (``Src/write_subs.F``: ``(a8,i4,3f16.6,3f10.6)``,
#: ``4f10.6`` under ``Spin.Fix``): the tag -- ``scf:`` or ``ts-scf:`` -- the
#: iteration, then Eharris, E_KS, FreeEng, dDmax, Ef (two under ``Spin.Fix``)
#: and dHmax, in ``write_subs.F``'s order.  Every pattern for the row is built
#: from these two strings, so the tee, the monitor, the parser and the timing
#: instrument cannot disagree about what a row is.
_NEGF_MARK = "ts-"
_ROW_TAG = "scf:"


def _either_case(word: str) -> str:
    """``scf:`` as ``[Ss][Cc][Ff]:`` -- case-blind in an ERE, which POSIX
    awk has no flag for, and the same set to Python.  The standing rule is
    that detection is immune to capitalisation (user, 2026-05-28)."""
    return "".join(f"[{c.upper()}{c.lower()}]" if c.isalpha() else c
                   for c in word)


#: The row's start as a POSIX ERE that is also a Python regex -- the wrapper's
#: awk takes them verbatim.
SCF_PREFIX_ERE = (rf"^[ \t]*({_either_case(_NEGF_MARK)})?"
                  rf"{_either_case(_ROW_TAG)}[ \t]*")
SCF_ROW_ERE = SCF_PREFIX_ERE + "[0-9]"
#: The row itself: group 1 the NEGF mark, 2 the iteration, 3 the columns.
#: :func:`scf_row` is how a reader asks it.
SCF_ROW = re.compile(SCF_PREFIX_ERE + r"([0-9]+)[ \t]+(.+)$")

#: The phases, as each cycle carries them.  The periodic SCF is SIESTA's own;
#: the NEGF loop is TranSIESTA's, which a device runs after the periodic
#: initialization (``transiesta: Initialization run using siesta``).
PHASE_PERIODIC = "periodic"
PHASE_NEGF = "negf"

#: The line opening each step, in SIESTA's own words and with its own number
#: (``Src/state_init.F``, ``write(6,'(t25,a,i6)')``): ``Begin <CG|Broyden|FIRE>
#: opt. move = <N>`` for a relaxation, ``Begin FC step = <N>`` for a
#: force-constant run (0 the undisplaced geometry), ``Begin MD step = <N>``.
#: Group 1 is what a step is, group 2 its number.  A single point prints
#: ``Single-point calculation`` instead, and so states no step -- a transport
#: rung is one.
STEP_BEGIN = re.compile(r"^\s*Begin\s+(.*?\S)\s*=\s*([0-9]+)\s*$",
                        re.IGNORECASE)

#: The names row SIESTA prints above a step's rows (``write_subs.F``:
#: ``iscf  Eharris(eV)  E_KS(eV) ...``), which the parser maps columns by.
SCF_HEADER = re.compile(r"^\s*iscf\s+\S", re.IGNORECASE)


class ScfRow(NamedTuple):
    """One SCF row: its phase, its iteration, and where its columns start --
    the origin of the fixed-width read (``3f16.6`` from there)."""
    phase: str
    iscf: int
    columns_at: int


def scf_row(line: str) -> Optional[ScfRow]:
    """The row ``line`` is, or ``None``."""
    m = SCF_ROW.match(line)
    if m is None:
        return None
    return ScfRow(PHASE_NEGF if m.group(1) else PHASE_PERIODIC,
                  int(m.group(2)), m.end(2))


# ---- The SCF row's values -----------------------------------------------------
# The row's ONE value reader, the parser's (`siesta_reader`).  It stood inside
# `siesta.py` until 2026-09-26, where the reading pass that runs beside a job
# with no numpy could not reach it.

# Defensive separator-inserter for SIESTA's fixed-width SCF columns.
# When two adjacent columns pack so tight that no whitespace separates
# them, the naive ``split()`` captures them as one token.  Two adjacency
# cases the regex catches:
#
#   * column NEXT to a fine column -- e.g. ``-1.929956131.029438`` --
#     both values fine, fields just touched (dHmax-and-Ef_dn case
#     fixed 2026-05-28).
#   * column NEXT to an overflowed column -- e.g. ``6.401317**********``
#     -- fine value glued to a Fortran field-width overflow (the
#     all-asterisks indicator).  Without splitting here, the joint
#     token fails ``float()`` AND ``fortran_float`` (which
#     only matches all-asterisks), nuking the row.  Added 2026-06-14
#     after the BDT-stage-2 divergent-SCF report.
#
# SIESTA's SCF data row uses Fortran format ``(3F16.6, 3F10.6)`` for
# closed-shell or ``(3F16.6, 4F10.6)`` for spin-polarized.  Every
# F-field ends with EXACTLY 6 decimal places: ``.NNNNNN`` is the
# canonical end-of-field signature.  Any non-whitespace character
# immediately following must be the first character of the next
# field -- a sign, a digit, an asterisk (Fortran overflow), or even
# a stray ``.`` from a value too long for its width.  Insert a
# separator there.
#
# Why the lookahead is ``\S`` and not the narrower ``[-+\d*]``:
# unconverged early SCF iters can chain together arbitrarily.  The
# original ``[\d*]`` missed the ``-`` case (BDT-Au stage3 iter 1
# regression, 2026-06-15: ``45.787763-15.068303410.273625`` -- Ef
# = -15.068 left no padding so the minus sign of the next field was
# the only separator).  Widening to ``[-+\d*]`` fixed that but is
# enumerative; ``\S`` is structurally exhaustive -- the ``.6``
# signature itself IS the field boundary, anything past it must
# be the next column.  No false positives are possible: Fortran
# F-format always ends a field at ``.NNNNNN``, never mid-value.
_SCF_TIGHT_PACK = re.compile(r"(\.\d{6})(?=\S)")

# Fortran fixed-width overflow indicator.  When a value can't fit its
# format field (e.g. ``f10.6`` for a magnitude > 999), Fortran prints
# the field as all asterisks (``**********``).  This is DISTINCT from
# the tight-pack case (where the values are fine but the columns
# touch): here the value itself was lost in the I/O layer; the SCF
# cycle is most likely diverging.  We tokenise these as NaN so the
# rest of the row's data is still recoverable -- the user CAN still
# see WHEN the divergence started (from the early healthy cycles) and
# WHICH column blew first; dropping the row entirely would erase that
# diagnostic.  Pattern: a contiguous run of 2+ asterisks (1 asterisk
# is too generic; Fortran always emits the full field width, which
# is always >= 2 for any column we care about).
_FORTRAN_OVERFLOW = re.compile(r"^\*{2,}$")


def fortran_float(tok: str) -> float:
    """``float(tok)`` but with one Fortran-only escape hatch.

    A token of all asterisks (``**********``) is what Fortran writes
    when a value can't fit its fixed-width format field.  In that
    case the magnitude is gone but the cycle around it is still
    interesting (often diagnostic of divergence).  Return NaN so the
    caller can keep parsing the row instead of dropping it.  Every
    other unparseable token still raises ``ValueError`` -- this is
    NOT a permissive ``try: float ... except: NaN`` blanket.
    """
    if _FORTRAN_OVERFLOW.match(tok):
        return float("nan")
    return float(tok)


# Fortran field widths in the SIESTA SCF data row format.  Three
# F16.6 energy columns followed by either 3 (closed-shell) or 4
# (spin-polarized) F10.6 columns.  Used by the column-position
# fallback when whitespace recovery fails.
_SCF_ENERGY_FIELD_WIDTH = 16
_SCF_RATIO_FIELD_WIDTH  = 10
_SCF_N_ENERGY_FIELDS    = 3
# Valid total counts after the iscf integer: 3 + 3 = 6 (closed-shell),
# 3 + 4 = 7 (spin-polarized).
_SCF_VALID_VALUE_COUNTS = (6, 7)


def scf_floats_by_columns(
    line: str, data_start: int,
) -> Optional[List[float]]:
    """Recover the SCF row's floats by Fortran column position.

    SIESTA writes the SCF row using format ``(3F16.6, 3F10.6)`` for
    closed-shell or ``(3F16.6, 4F10.6)`` for spin-polarized.  Each
    F-field has a fixed character width regardless of the value's
    magnitude, so we can slice at known boundaries even when:

      * Multiple values run together with no whitespace separator
        (the leading sign / digits of the next column consume all
        of its leading-space padding -- see ``_SCF_TIGHT_PACK``
        for the pure-text recovery, this is the structural fallback)
      * A column contains a Fortran overflow indicator
        (``**********``) that ``fortran_float`` decodes as NaN
      * The whitespace recovery would mis-bind tokens (e.g. a
        future SIESTA format variant we haven't characterised yet)

    ``data_start`` must be the position in ``line`` immediately AFTER
    the iscf integer -- i.e. the first character of the first F16.6
    field, INCLUDING its leading whitespace.  Callers obtain this
    from :func:`scf_row`'s ``columns_at``.

    Returns 6 floats (closed-shell) or 7 (spin-polarized); returns
    None when the column structure is missing or a slice can't be
    parsed as a Fortran float.  The 7th-column probe is OPTIONAL --
    a clean closed-shell row that has trailing whitespace beyond
    the 6 expected columns still returns the 6 floats.
    """
    pos = data_start
    floats: List[float] = []
    # Slot 1-3: F16.6 energy columns.
    for _ in range(_SCF_N_ENERGY_FIELDS):
        chunk = line[pos:pos + _SCF_ENERGY_FIELD_WIDTH].strip()
        if not chunk:
            return None
        try:
            floats.append(fortran_float(chunk))
        except ValueError:
            return None
        pos += _SCF_ENERGY_FIELD_WIDTH
    # Slot 4-6 (and optional 7): F10.6 ratio / threshold columns.
    # The minimum is 3 (closed-shell); a 7th if present is the spin
    # Ef_dn extra field.
    for slot_idx in range(4):
        if pos >= len(line):
            break
        chunk = line[pos:pos + _SCF_RATIO_FIELD_WIDTH].strip()
        if not chunk:
            # No more data; allowed only if we already have the
            # closed-shell complement.
            break
        try:
            floats.append(fortran_float(chunk))
        except ValueError:
            if len(floats) in _SCF_VALID_VALUE_COUNTS:
                # We already had a valid set; the trailing garbage
                # is noise (could be a ``** ...`` warning glued on).
                # Accept what we have.
                break
            return None
        pos += _SCF_RATIO_FIELD_WIDTH
    if len(floats) not in _SCF_VALID_VALUE_COUNTS:
        return None
    return floats


def scf_floats(
    rest: str,
    *,
    line: Optional[str] = None,
    data_start: Optional[int] = None,
) -> Optional[List[float]]:
    """Tokenize the post-``scf: <iscf>`` part of an SCF line into
    floats.  Returns the list, or None if both recovery layers fail.

    Two-layer recovery:

      1. **Whitespace recovery** (the fast path).  ``_SCF_TIGHT_PACK``
         inserts a separator after each ``.NNNNNN`` six-decimal field
         when the next column begins with a sign / digit / asterisk.
         Then ``split()`` + per-token ``fortran_float``.
         Handles the common cases:
           * Tight-packed columns where Ef or dHmax filled their
             whole F10.6 widths and bumped into the next column.
           * Fortran field overflow (``**********``) tokens decoded
             as NaN so the rest of the row survives.

      2. **Column-position fallback** (the safety net).  When the
         whitespace recovery yields the wrong number of floats
         (i.e. a glue pattern the regex didn't catch),
         ``scf_floats_by_columns`` slices the line at the
         known Fortran F-format boundaries (3 × F16.6 + 3..4 ×
         F10.6) and parses each field independently.  Requires the
         caller to pass ``line=`` + ``data_start=`` (the position
         immediately after the iscf integer).  This is the robust
         answer to any future format glitch we haven't characterised:
         column widths are stable across SIESTA versions back to v3.

    Returns ``None`` only when BOTH layers fail -- that's a real
    parser bug we want surfaced as a ParseWarning, not silently
    coerced into NaNs.
    """
    fixed = _SCF_TIGHT_PACK.sub(r"\1 ", rest)
    try:
        vals = [fortran_float(t) for t in fixed.split()]
    except ValueError:
        vals = None
    if vals is not None and len(vals) in _SCF_VALID_VALUE_COUNTS:
        return vals
    # Layer 2: column-position fallback.  Optional context lets
    # legacy callers (tests with synthetic ``rest`` strings) still
    # use the regex path alone; callers that have the full line
    # get the more robust slicing.
    if line is not None and data_start is not None:
        cols = scf_floats_by_columns(line, data_start)
        if cols is not None:
            return cols
    # Layer 1 may have produced an unusual count (e.g. 5 or 8) --
    # return None so the caller emits a ParseWarning rather than
    # passing through garbage.
    return None


# SCF column-header detection.  Real-world example lines:
#
#   v5 spin-polarized:
#       iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax     Ef_up Ef_dn(eV) dHmax(eV)
#
#   v5 closed-shell:
#       iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax     Ef(eV) dHmax(eV)
#
# We detect the line by its leading ``iscf`` token (case-insensitive;
# no other SIESTA output begins with that bare word) and parse the
# column names into canonical keys.  Subsequent ``scf:`` data rows
# are mapped by name, not by position -- a future SIESTA version
# that adds / reorders columns adapts automatically.
#
# Robustness policy (2026-05-28): all string matching here is
# case-insensitive AND tolerates the ``(unit)`` suffix being absent.
# So ``DHMAX``, ``dhmax``, ``dHmax(eV)``, and ``dhmax`` all map to
# the same canonical key.  This is the "names should be immune to
# capitalisation, small spelling differences" rule applied to the
# SCF column header.


def _normalise_column_token(tok: str) -> str:
    """Strip an optional ``(unit)`` suffix and lower-case for the
    column-key lookup.  Examples:

      ``dHmax(eV)`` -> ``dhmax``
      ``Ef_dn(eV)`` -> ``ef_dn``
      ``DDMAX``     -> ``ddmax``
    """
    return re.sub(r"\([^)]*\)$", "", tok).lower()


# Canonical-key map for SCF columns.  Lookup is via
# ``_normalise_column_token`` so a header token like ``DHmax(EV)``
# resolves to the same key as ``dHmax(eV)``.  ``None`` value =
# "valid bookkeeping column we don't extract".  Unknown tokens
# (a future SIESTA layout we haven't seen) stay as the raw
# normalised token so they STILL land in the per-cycle dict --
# downstream consumers can introspect.
#
# Canonical keys we PROMISE downstream consumers:
#   "cycle"   -- iscf
#   "energy"  -- E_KS, the energy we plot
#   "dDmax"   -- DM-mixing residual
#   "dHmax"   -- Hamiltonian-mixing residual
# Anything else is parser-driven from the header.
SCF_COLUMN_KEYS = {
    "iscf":     "cycle",
    "eharris":  None,
    "e_ks":     "energy",
    "freeeng":  None,
    "ddmax":    "dDmax",
    # KEPT, not discarded (2026-09-18).  These read `None` -- "valid
    # bookkeeping column we don't extract" -- since the parser was written.
    # For an ORDINARY run that was right: nothing plotted E_F.
    #
    # For a TRANSPORT lead it is the one number that matters.  An electrode
    # is a periodic BULK run and `engines/transport.md` says what it is for:
    # *"its E_F is the reference energy"*, the thing T(E) is measured
    # relative to and `G = G0 * T(E_F)` is evaluated at.  `electrode_kz`
    # defaults to 40 precisely because that is "the Fermi-level resolution",
    # and changing it invalidates both lead stages and everything downstream
    # "because the lead's Fermi level moved".  A lead that converged tells a
    # reader almost nothing; its E_F tells them what the junction is
    # referenced to -- and two leads that disagree is a defect nothing else
    # on the Results tab would show.
    #
    # They land in `Frame.scf_history` like every other column; the last
    # cycle's value is the converged one.
    "ef":       "ef",       # closed-shell Fermi level
    "ef_up":    "ef_up",    # collinear spin-polarized (up)
    "ef_dn":    "ef_dn",    # collinear spin-polarized (down)
    "ef_x":     None,    # non-collinear (hypothetical future)
    "ef_y":     None,
    "ef_z":     None,
    "dhmax":    "dHmax",
}


def scf_header(line: str) -> Optional[List[Optional[str]]]:
    """Tokenise a SIESTA SCF column header into a list of canonical
    keys.  ``None`` entries mark columns we ignore; ``str`` entries
    mark columns whose value will be stored in the per-cycle dict.

    Case-insensitive + ``(unit)``-suffix-tolerant per the policy
    above.  Returns ``None`` if the line doesn't look like an SCF
    header.
    """
    tokens = line.split()
    if not tokens or tokens[0].lower() != "iscf":
        return None
    return [SCF_COLUMN_KEYS.get(_normalise_column_token(t),
                                  _normalise_column_token(t))
            for t in tokens]


def cycle_from_header(
    iscf: int,
    vals: List[float],
    header: List[Optional[str]],
) -> Optional[Dict[str, Any]]:
    """Map a parsed SCF data row to a per-cycle dict using the
    column header.  Header[0] is the iscf column; the remaining
    header entries pair with ``vals`` by position.  Bookkeeping
    columns (header entry == None) are skipped.

    Returns ``None`` if the value count doesn't match the header.
    Otherwise returns a dict whose canonical keys (``cycle``,
    ``energy``, ``dDmax``, ``dHmax``) the downstream UI relies on,
    plus any extra columns SIESTA chose to emit.
    """
    expected = len(header) - 1   # minus the iscf column
    if len(vals) != expected:
        return None
    out: Dict[str, Any] = {"cycle": iscf}
    for key, val in zip(header[1:], vals):
        if key is not None:
            out[key] = val
    return out


def cycle_positional(
    iscf: int,
    vals: List[float],
) -> Optional[Dict[str, Any]]:
    """Fallback when no SCF column header was seen yet.  Dispatches
    on the value count -- 6 = closed-shell, 7 = collinear spin --
    using the historically-known SIESTA layouts:

      6 floats:  Eharris, E_KS, FreeEng, dDmax, Ef, dHmax
      7 floats:  Eharris, E_KS, FreeEng, dDmax, Ef_up, Ef_dn, dHmax

    This is a last resort -- prefer the header-driven path.  Returns
    ``None`` on unexpected count.
    """
    if len(vals) == 6:
        return {
            "cycle":  iscf,
            "energy": vals[1],
            "dDmax":  vals[3],
            "dHmax":  vals[5],
        }
    if len(vals) == 7:
        return {
            "cycle":  iscf,
            "energy": vals[1],
            "dDmax":  vals[3],
            "dHmax":  vals[6],
        }
    return None


# ---- The blocks a step is read from ------------------------------------------------
#: ``outcoor: <title> (Ang):`` opens a coordinates block -- ``Atomic
#: coordinates``, ``Relaxed atomic coordinates``, ``Final (unrelaxed) atomic
#: coordinates`` (``Src/outcoor.f``).
COORDS_BEGIN = "outcoor:"
#: ``outcell: Unit cell vectors (Ang):`` opens the cell block.
CELL_BEGIN = "outcell: Unit cell vectors"
#: ``siesta: Atomic forces (eV/Ang):`` opens the forces block.
FORCES_BEGIN = "siesta: Atomic forces"
#: A step's Kohn-Sham energy, ``siesta: E_KS(eV) = ...`` -- it sits mid-line.
E_KS_LINE = "siesta: E_KS(eV)"
#: The bare ``siesta: Etot = ...`` of the energy decomposition, never
#: ``Etot/N`` or ``Etot(eV)``, which are other rows.
ETOT_LINE = re.compile(r"^\s*siesta:\s+Etot\s*=\s*[-\d]", re.IGNORECASE)
#: SIESTA's IterSCF timer, CUMULATIVE since the start of the run: ``timer:
#: Routine,Calls,Time,% = IterSCF  <calls>  <time>  <percent>``.
ITER_SCF_TIMER = re.compile(
    r"^\s*timer:\s*Routine,Calls,Time,%\s*=\s*IterSCF\s+(.+)$",
    re.IGNORECASE)
#: Lines only a SIESTA-family output prints, in lower case, for the format
#: sniffer (`engines/siesta.py`): the banner, the system type, the blocks
#: above, and a relaxation's step openers.
SNIFF_MARKERS = (
    "welcome to siesta",
    "siesta: system type",
    FORCES_BEGIN.lower(),
    "outcoor: atomic coordinates",
    CELL_BEGIN.lower(),
    "begin cg opt", "begin md opt", "begin broyden opt", "begin fire opt",
)


# ---- The deck, as SIESTA echoes it -----------------------------------------------
#: SIESTA copies the deck it read into its output between two starred rules
#: (``Src/reinit_m.F90``: ``*** Dump of input data file ***`` and ``*** End of
#: input data file ***``) -- comments included.  NOTHING INSIDE IS THE RUN
#: SPEAKING: a marker matched there is the deck's own text.  molbuilder's decks
#: name the failures they guard against (``# 'propor: ERROR: IMAX = 0' on
#: parallel run: ...``), so until 2026-09-26 a relaxation that ran out of moves
#: read as stopped by ``propor``.  PySCF's reader anchors its end lines at
#: column 0 for the same reason: PySCF echoes the deck's source too.
INPUT_ECHO_BEGIN = re.compile(r"^\*+\s*Dump of input data file\s*\*+\s*$",
                              re.IGNORECASE)
INPUT_ECHO_END = re.compile(r"^\*+\s*End of input data file\s*\*+\s*$",
                            re.IGNORECASE)


def input_echo_edge(line: str) -> Optional[bool]:
    """``True`` at the line that opens SIESTA's echo of its input, ``False``
    at the line that closes it, ``None`` for any other line -- so a reader
    walking the output knows whether the run or the deck is speaking."""
    if INPUT_ECHO_BEGIN.match(line):
        return True
    if INPUT_ECHO_END.match(line):
        return False
    return None


# ---- How a run ends -----------------------------------------------------------
#: Markers that prove the run did NOT reach its own end, in the order a reader
#: should prefer them: ``(substring, run_state)``, matched case-insensitively
#: anywhere in the line -- SIESTA prefixes them with ``node 0:`` under MPI.
#: The reading pass (`siesta_reader`) builds its fatal rules from them -- the
#: one reader of how a SIESTA run ended (`model/parse.md` § 2b).
#: ``Src/propor.f``'s refusal of a distribution with an empty table
#: (``propor: ERROR: IMAX = 0``) -- a defective pseudopotential, a rank count
#: that leaves ranks empty, or no net spin on an open-shell metal.  Named, so
#: the wrapper's hint for it asks the reader rather than grepping.
PROPOR_MARKER = "propor: error"
FATAL_MARKERS: Tuple[Tuple[str, str], ...] = (
    # Out of memory is called out from the generic aborts because it is the
    # most common cause and the most actionable: "you ran out of memory" is
    # the one sentence that tells a user what to change.
    ("out of memory",                "out_of_memory"),
    ("oom-kill",                     "out_of_memory"),
    ("killed process",               "out_of_memory"),
    ("cannot allocate memory",       "out_of_memory"),
    ("insufficient virtual memory",  "out_of_memory"),
    ("siesta: error",                "stopped"),
    (PROPOR_MARKER,                  "stopped"),
    ("stopping program from node",   "stopped"),
    ("siesta died",                  "stopped"),
    ("abnormal_termination",         "stopped"),
)

#: The SCF met its criterion (``Src/scfconvergence_test.F``: ``SCF
#: Convergence by <criteria>``), and the two ways it did not: ``SCF_NOT_CONV:``,
#: the fatal form, and the softer line a relaxation can recover from.
#: Matched case-insensitively anywhere in the line.
SCF_CONVERGED_MARKER = "scf convergence by"
SCF_NOT_CONV_MARKER = "scf_not_conv"
SCF_NOT_CONVERGED_MARKER = "scf did not converge"

#: What each CAUSE means, in words a person reads -- a cause being the first
#: fatal line's marker, or :data:`SCF_NOT_CONV_MARKER` when SIESTA made the
#: SCF fatal (`_run_ending.RunEnding.cause`).  The viewer's "Reason:" line is
#: this table's, never a copy of the markers in the browser.
CAUSE_WORDS = {
    **{m: "out of memory" for m, st in FATAL_MARKERS if st == "out_of_memory"},
    "siesta: error": "an engine error",
    PROPOR_MARKER: "MPI rank distribution error (propor)",
    "stopping program from node": "a rank stopped the run",
    "siesta died": "the engine died",
    "abnormal_termination": "abnormal termination",
    SCF_NOT_CONV_MARKER: "SCF non-convergence (required to converge)",
}
#: WHEN ``SCF_NOT_CONV:`` IS THE CAUSE OF DEATH, SIESTA says so on the line:
#: under ``SCF.MustConverge`` it prints ``SCF_NOT_CONV: SCF did not converge
#: in maximum number of steps (required).`` and dies
#: (``Src/siesta_forces.F90`` ~608-619, ``die('ABNORMAL_TERMINATION')``);
#: without it the same line carries no ``(required)`` and the run goes on.
SCF_NOT_CONV_REQUIRED = "(required)"
#: SIESTA taking back a convergence it has just printed
#: (``Src/siesta_forces.F90``): TranSIESTA's charge still off
#: (``SCF cycle continued due to TranSiesta charge deviation``), or fewer
#: iterations than ``SCF.MinIterations``.  Until the next row, the phase has
#: not converged.
SCF_CONTINUED_MARKER = "scf cycle continued"

#: How a relaxation ended (``Src/siesta_analysis.F90``: ``outcoor(...,
#: 'Relaxed')`` when it converged, ``'Final (unrelaxed)'`` when it ran out of
#: moves), as ``outcoor``'s heading prints it -- ``outcoor: <phrase> atomic
#: coordinates (Ang):``.  A single point prints neither.  Matched
#: case-insensitively anywhere in the line.
RELAXED_MARKER = "outcoor: relaxed atomic coordinates"
UNRELAXED_MARKER = "outcoor: final (unrelaxed) atomic coordinates"


#: ``reinit: System Label: H2`` -- the SystemLabel the run read, which SIESTA
#: names its own files from (``<label>.MD.nc``, ``<label>.XV``): an output
#: of ours is named otherwise (``<label>_<token>-run<N>.out``), so its own
#: line is how it names its companions (`model/parse.md` § 5.3).  Measured:
#: line 36 of both recorded H2 outputs, among the setup lines.
SYSTEM_LABEL = re.compile(r"^reinit:\s*System Label:\s*(\S+)")


# ---- The launch lines, SIESTA and TBtrans alike -------------------------------
#: ``Src/runinfo_m.F90``, which TBtrans's ``tbt_init.F90`` calls too: the
#: ranks MPI gave the run, or serial mode -- one rank, whether the build has
#: MPI (``(only 1 MPI rank)``) or not.  Multiline, so a caller holding the
#: whole text can search it.
RUNNING_ON = re.compile(r"^\s*\*\s*Running on\s+(\d+)\s+nodes? in parallel",
                        re.IGNORECASE | re.MULTILINE)
RUNNING_SERIAL = re.compile(r"^\s*\*\s*Running in serial mode",
                            re.IGNORECASE | re.MULTILINE)
#: ``Src/initparallel.F``: the process grid and the orbital distribution's
#: block size -- what the deck's ``BlockSize`` asks for, adapted by SIESTA to
#: the rank count.
PROCESS_GRID = re.compile(
    r"^\s*\*\s*ProcessorY,\s*Blocksize:\s*(\d+)\s+(\d+)", re.IGNORECASE)
#: The time of day at both ends (``Src/timestamp.f90``:
#: ``>> <what>:  dd-MON-yyyy hh:mm:ss``, printed by SIESTA's ``siesta_init.F``
#: and ``siesta_end.F`` and by TBtrans's ``tbt_init.F90`` and ``tbt_end.F90``).
RUN_START = re.compile(r"^\s*>>\s*Start of run\b[:\s]*(.*?)\s*$",
                       re.IGNORECASE)
RUN_END = re.compile(r"^\s*>>\s*End of run\b[:\s]*(.*?)\s*$", re.IGNORECASE)


def local_time(stamp: str):
    """``24-SEP-2026   9:17:29`` as a naive ISO string, or ``None``.

    NAIVE on purpose: SIESTA prints the node's local time with no zone, and
    stamping one on would be a claim the file does not make.  The month is
    ``timestamp.f90``'s own English table, so no locale is involved.
    """
    from datetime import datetime
    try:
        return datetime.strptime(" ".join(stamp.split()),
                                 "%d-%b-%Y %H:%M:%S").isoformat()
    except ValueError:
        return None


def read_launch_line(line: str, facts: dict) -> Optional[str]:
    """Record one launch line into ``facts`` -- ``n_mpi_processes``,
    ``processor_y`` and ``blocksize``, ``run_start_local``,
    ``run_end_local`` -- and return the key, or ``None`` when the line is not
    one.  Each is stated once, so the first line wins."""
    m = PROCESS_GRID.match(line)
    if m:
        facts.setdefault("processor_y", int(m.group(1)))
        facts.setdefault("blocksize", int(m.group(2)))
        return "blocksize"
    m = RUNNING_ON.match(line)
    if m:
        facts.setdefault("n_mpi_processes", int(m.group(1)))
        return "n_mpi_processes"
    if RUNNING_SERIAL.match(line):
        facts.setdefault("n_mpi_processes", 1)
        return "n_mpi_processes"
    for key, pat in (("run_start_local", RUN_START),
                     ("run_end_local", RUN_END)):
        m = pat.match(line)
        if m:
            stamp = local_time(m.group(1))
            if stamp:
                facts.setdefault(key, stamp)
            return key
    return None


def mpi_ranks(text: str) -> Optional[int]:
    """The ranks a whole ``.out`` states, or ``None`` when it states none."""
    facts: dict = {}
    for line in text.splitlines():
        if read_launch_line(line, facts) == "n_mpi_processes":
            break
    return facts.get("n_mpi_processes")


# ---- The build header, SIESTA and TBtrans alike --------------------------------
#: Both programs open their ``.out`` with it (``Src/version-info-template.inc``),
#: so :func:`read_build_line` is the one reader of both.
EXECUTABLE = re.compile(r"^\s*Executable\s*:\s*(\S+)", re.IGNORECASE)
BUILD_VERSION = re.compile(r"^\s*(?:Siesta\s+)?Version\s*:\s*(\S+)",
                           re.IGNORECASE)
BUILD_ARCH = re.compile(r"^\s*Architecture\s*:\s*(.+?)\s*$", re.IGNORECASE)
BUILD_COMPILER = re.compile(r"^\s*Compiler version\s*:\s*(.+?)\s*$",
                            re.IGNORECASE)
#: ``MPI``, ``OpenMP``, both, or ``none``.
BUILD_PARALLEL = re.compile(r"^\s*Parallelisations?\s*:\s*(.+?)\s*$",
                            re.IGNORECASE)
#: One line per compiled-in component, and not always the bare name: 5.4.2
#: writes ``ELSI support. Solvers:`` and ``Native PEXSI support``, which a
#: bare-name pattern read as absent -- the packaged SIESTA's ELSI, the road
#: its ELPA takes, was recorded as missing until 2026-09-26.  ``ELPA`` and
#: ``FLOOK`` are older builds' spellings, kept.
BUILD_FEATURE = re.compile(
    r"^\s*(?:Native\s+)?(GEMM3M|NetCDF-4 MPI-IO|NetCDF-4|NetCDF|"
    r"METIS ordering|Lua|PEXSI|ELSI|ELPA|FLOOK|DFT-D3|Wannier90 wrapper)"
    r"\s+support\b", re.IGNORECASE)
#: Native ELPA prints no line of its own; built with GPU kernels it prints
#: `` --- ELPA GPU support: <kind>`` (measured: ``nvidia-gpu``).
BUILD_ELPA_GPU = re.compile(r"^\s*-+\s*ELPA GPU support:\s*(\S+)",
                            re.IGNORECASE)


def read_build_line(line: str, build: dict) -> Optional[str]:
    """Record one build-header line into ``build``; return the key it set,
    or ``None`` when the line is not one.  The header states each fact once,
    so the first line wins."""
    for key, pat in (("executable", EXECUTABLE), ("version", BUILD_VERSION),
                     ("architecture", BUILD_ARCH),
                     ("compiler", BUILD_COMPILER),
                     ("elpa_gpu", BUILD_ELPA_GPU)):
        m = pat.match(line)
        if m:
            build.setdefault(key, m.group(1).strip())
            return key
    m = BUILD_PARALLEL.match(line)
    if m:
        # "MPI, OpenMP" -> ["MPI", "OPENMP"]; "none" stays one token.
        build.setdefault("parallelisations",
                         [t.strip().upper()
                          for t in re.split(r"[,\s]+", m.group(1))
                          if t.strip()])
        return "parallelisations"
    m = BUILD_FEATURE.match(line)
    if m:
        # "NetCDF-4 MPI-IO" -> "netcdf4_mpiio"; "ELSI" -> "elsi".
        key = re.sub(r"\s+", "_", m.group(1).lower().replace("-", ""))
        build[key] = True
        return key
    return None


# ---- The solver ------------------------------------------------------------------
#: ``Src/diag_option.F90``'s ``print_diag``: once SIESTA has read its options it
#: prints what it RESOLVED, as ``diag: <label>  = <value>`` -- the algorithm,
#: the ELPA GPU string, the block size, the process grid.  It is the only
#: account 5.4.2 gives of its solver: its ``Src/`` writes no ``redata:`` line
#: for the algorithm and no GPU-detected banner, which is what the parser's
#: half of the reader this replaced matched (``_diag.py``, 2026-09-18 to
#: 2026-09-26) -- so no real run had a solver on record.  The diagonalizer's
#: block size here is its own (``Diag.BlockSize``); the orbital distribution's
#: is the launch line :data:`PROCESS_GRID`.
DIAG_LINE = re.compile(r"^\s*diag:\s*([A-Za-z][^=]*?)\s*=\s*(.+?)\s*$",
                       re.IGNORECASE)
#: The labels kept, under the names their readers use.
DIAG_FACTS = {"Algorithm": "algorithm",
              "ELPA GPU string key": "elpa_gpu",
              "Parallel block-size": "diag_blocksize",
              "Parallel distribution": "distribution",
              "Parallel over k": "parallel_over_k"}


def read_diag_line(line: str, diag: dict) -> Optional[str]:
    """Record one ``diag:`` line into ``diag``; return its key, or ``None``
    when the line is not one the table keeps.  The first line wins."""
    m = DIAG_LINE.match(line)
    key = DIAG_FACTS.get(m.group(1)) if m else None
    if key is None:
        return None
    value = m.group(2)
    if key == "diag_blocksize":
        try:
            value = int(value)
        except ValueError:
            return None
    elif key == "distribution":                    # "    4 x     5" -> "4 x 5"
        value = " x ".join(t.strip() for t in value.split("x"))
    diag.setdefault(key, value)
    return key


# ---- The pseudopotentials the run read ----------------------------------------
#: ``Src/basis_specs.f`` heads each species (``---- Processing specs for
#: species: <label>``); ``Src/ncps/src/m_ncps_reader.f`` names the file it read
#: on the line after ``Reading pseudopotential information in PSML from:``, and
#: then its ``PSML uuid``.  The file is the one the run opened, beside it.
PSML_SPECIES = re.compile(r"^-+\s*Processing specs for species:\s*(\S+)",
                          re.IGNORECASE)
PSML_FROM = re.compile(r"^\s*Reading pseudopotential information in PSML from:",
                       re.IGNORECASE)
PSML_UUID = re.compile(r"^\s*PSML uuid:\s*(\S+)", re.IGNORECASE)


def read_psml_lines(lines) -> List[dict]:
    """``[{species, file, uuid}]`` in the order SIESTA read them."""
    out: List[dict] = []
    cur, want_file = None, False
    for line in lines:
        m = PSML_SPECIES.match(line)
        if m:
            cur, want_file = {"species": m.group(1)}, False
            out.append(cur)
            continue
        if cur is None:
            continue
        if want_file:
            cur["file"], want_file = line.strip(), False
            continue
        if PSML_FROM.match(line):
            want_file = True
            continue
        m = PSML_UUID.match(line)
        if m:
            cur.setdefault("uuid", m.group(1))
    return [sp for sp in out if "file" in sp]


# ---- What the SCF must reach ----------------------------------------------------
#: ``Src/read_options.F90`` echoes each criterion as a pair -- ``redata: Require
#: <X> convergence for SCF = T|F`` and ``redata: <X> tolerance for SCF = <v>
#: [eV]`` -- and TranSIESTA states the NEGF loop's own in its start-up echo
#: (:data:`TS_CRITERIA`).  Keyed by the SCF-row column each one bounds, so a
#: plot draws a tolerance against the residual it is FOR: the H tolerance is
#: in eV and bounds dHmax, the DM one is dimensionless and bounds dDmax.
SCF_REQUIRE = re.compile(
    r"^\s*redata:\s*Require\s+(.+?)\s+convergence for SCF\s*=\s*([TF])",
    re.IGNORECASE)
SCF_TOLERANCE = re.compile(
    r"^\s*redata:\s*(.+?)\s+tolerance for SCF\s*=\s*(\S+)(?:\s+(eV))?",
    re.IGNORECASE)
#: ``redata:``'s names for a criterion -> the column it bounds.  Harris and EDM
#: bound nothing the row prints, and are recorded without a trace to draw.
SCF_CRITERION_OF = {"dm": "dDmax", "h": "dHmax", "hamiltonian": "dHmax",
                    "edm": "EDM", "harris": "Eharris",
                    "harris energy": "Eharris", "(free) energy": "FreeEng"}
#: TranSIESTA's echo labels for the NEGF loop's criteria (``m_ts_options.F90``),
#: and the one that says whether the charge must converge.
TS_CRITERIA = {"SCF DM tolerance": "dDmax",
               "SCF Hamiltonian tolerance": "dHmax",
               "SCF charge tolerance": "dq"}
TS_CHARGE_REQUIRED = "SCF converge charge"


def read_criterion_line(line: str, criteria: dict) -> Optional[str]:
    """Record one ``redata:`` criterion line into ``criteria`` --
    ``{column: {tolerance, unit, required}}`` -- and return the column, or
    ``None``."""
    m = SCF_REQUIRE.match(line)
    if m:
        col = SCF_CRITERION_OF.get(m.group(1).strip().lower())
        if col:
            criteria.setdefault(col, {})["required"] = m.group(2).upper() == "T"
        return col
    m = SCF_TOLERANCE.match(line)
    if m:
        col = SCF_CRITERION_OF.get(m.group(1).strip().lower())
        if col:
            try:
                criteria.setdefault(col, {})["tolerance"] = float(m.group(2))
            except ValueError:
                return None
            if m.group(3):
                criteria[col]["unit"] = m.group(3)
        return col
    return None


def negf_criteria(options: dict, periodic: dict) -> dict:
    """The NEGF loop's criteria, from TranSIESTA's echoed ``options``: its own
    tolerances, required as the periodic ones are -- TranSIESTA overrides the
    tolerance and keeps SIESTA's ``Require`` -- and the charge, required when
    the echo says it converges it."""
    out: dict = {}
    for label, col in TS_CRITERIA.items():
        text = str(options.get(label, "")).split()
        try:
            tol = float(text[0])
        except (IndexError, ValueError):
            continue
        entry = {"tolerance": tol}
        if len(text) > 1:
            entry["unit"] = text[1]
        if col == "dq":
            flag = str(options.get(TS_CHARGE_REQUIRED, "")).strip().upper()
            if flag in ("T", "F"):
                entry["required"] = flag == "T"
        elif "required" in periodic.get(col, {}):
            entry["required"] = periodic[col]["required"]
        out[col] = entry
    return out


# ---- The limits a run states, and its forces ----------------------------------
#: ``Src/read_options.F90``'s echo of the limits a run is held to, under the
#: cross-engine names its convergence targets carry (the molwatch header's
#: leaves, `trajectory_log.emitter`): the relaxation's force tolerance and
#: largest step, and the SCF's and the relaxation's iteration caps.  The
#: optimization cap is echoed whatever ``MD.TypeOfRun`` produced it.  These
#: stood as four private regexes in the parser until 2026-09-26.
TARGET_LINES: Tuple[Tuple[str, "re.Pattern", type], ...] = (
    ("max_force_tol_eV_per_A", re.compile(
        r"^\s*redata:\s+Force tolerance\s+=\s+([0-9.eE+-]+)\s+eV/Ang",
        re.IGNORECASE), float),
    ("max_scf_iter", re.compile(
        r"^\s*redata:\s+Max\. number of SCF Iter\s+=\s+(\d+)",
        re.IGNORECASE), int),
    ("max_displ_ang", re.compile(
        r"^\s*redata:\s+Max atomic displ per move\s+=\s+([0-9.eE+-]+)\s+Ang",
        re.IGNORECASE), float),
    ("max_geom_iter", re.compile(
        r"^\s*redata:\s+Maximum number of optimization moves\s+=\s+(\d+)",
        re.IGNORECASE), int),
)


def read_target_line(line: str) -> Optional[Tuple[str, Any]]:
    """``(target name, value)`` for one ``redata:`` limit line, or ``None``."""
    for key, pat, kind in TARGET_LINES:
        m = pat.match(line)
        if m:
            try:
                return key, kind(m.group(1))
            except ValueError:
                return None
    return None


#: The forces block's closing lines (``Src/write_subs.F``: ``'Max', fmax`` as
#: ``a6, f12.6``, and -- when an atom is constrained -- ``'Max', cfmax,
#: '    constrained'``, which is what SIESTA compares against
#: ``MD.MaxForceTol``).  Group 1 the value, group 2 the constrained mark.  A
#: reader asks it after a forces block: ``Max`` opens other lines too.
MAX_FORCE = re.compile(r"^\s*Max\s+(\S+)(\s+constrained)?\s*$",
                       re.IGNORECASE)


# ---- TranSIESTA's lines --------------------------------------------------------
#: The charge report TranSIESTA prints before each NEGF row
#: (``Src/ts_charge.F90``): a header -- ``D``, then per electrode ``E<i>`` and
#: ``C<i>``, then ``B`` when there is a buffer -- and a row of numbers, both
#: behind the ``ts-q:`` prefix.  The last one or two columns are no region's
#: charge: :data:`TS_Q_TOTALS`.
TS_Q_ROW = re.compile(r"^\s*ts-q:\s+(.+?)\s*$", re.IGNORECASE)
#: ``dQ`` -- the charge off its target -- and, spin-polarized, ``Qup-Qdn``.
TS_Q_TOTALS = ("dQ", "Qup-Qdn")

#: The Hartree-potential correction TranSIESTA applies each iteration
#: (``Src/m_ts_hartree.F90``: ``'ts-Vha: ',NEGF_Vha / eV,' eV'``).  In BOTH
#: phases: ``dhscf.F`` fixes the potential for the periodic initialization
#: too (*"We require that even the SIESTA potential is 'fixed'"*), so a
#: device's periodic rows carry it as well.
TS_VHA = re.compile(r"^\s*ts-Vha:\s*(\S+)\s*eV", re.IGNORECASE)

#: The start-up echo's frame and its lines (``Src/m_ts_options.F90``; the
#: contour part is ``Src/m_ts_contour_eq.f90``'s): a row of stars opens and
#: closes it; ``ts: <label> = <value>`` inside, sectioned by
#: ``>> Electrodes <<``, ``>> <electrode>`` and the contour banners.
TS_ECHO = re.compile(r"^\s*ts:\s", re.IGNORECASE)
TS_ECHO_FRAME = re.compile(r"^\s*ts:\s*\*{20,}\s*$", re.IGNORECASE)
TS_ECHO_LINE = re.compile(r"^\s*ts:\s+(.*?)\s+=\s+(.*?)\s*$", re.IGNORECASE)
TS_ECHO_SECTION = re.compile(r"^\s*ts:\s*>>\s*(.*?)\s*(?:<<)?\s*$",
                             re.IGNORECASE)
TS_ECHO_CONTOUR = re.compile(r"^\s*ts:\s*-{5,}\s*Contour\s*-{5,}\s*$",
                             re.IGNORECASE)
#: A contour banner lists one SEGMENT per chemical potential or contour part,
#: each with the same labels (``print_contour_eq_options``): a segment starts
#: at the line naming its chemical potential, or at ``Contour name``.
TS_ECHO_SEGMENT = re.compile(r"(chemical potential|^Contour name)$",
                             re.IGNORECASE)

#: The charge distribution TranSIESTA reports when it takes over from the
#: periodic density -- the baseline every NEGF iteration's charge is judged
#: against: ``transiesta: Charge distribution, target = <N>`` and then
#: ``<name> [<tag>] : <value>...`` rows -- one value, or with spin one per
#: channel, the ``[Q]`` row adding the total (``Src/ts_charge.F90``).
TS_CHARGE_START = re.compile(
    r"^\s*transiesta:\s*Charge distribution,\s*target\s*=\s*(\S+)",
    re.IGNORECASE)
TS_CHARGE_ROW = re.compile(r"^\s*(.+?)\s*\[(\w+)\]\s*:\s*(.+?)\s*$")

#: The electrode checks.  The principal-cell test (``Src/ts_electrode.F90``)
#: says one of three things: ``perfect!``, ``extending out with <n> elements,
#: all being zero.``, or ``extending out with <n> elements:`` and the
#: offending coupling.  The first two pass -- a coupling past the cell whose
#: elements are all zero couples nothing -- and ``ts_init.F90`` stops the run
#: on the third.  Then the surface Green's function's recursion.
TS_PRINCIPAL_CELL = re.compile(
    r"^\s*(\S+)\s+principal cell is\s+(.+?)\s*[.!:]*\s*$", re.IGNORECASE)
TS_GF_FOR = re.compile(
    r"^\s*Calculating surface Green functions for:\s*(\S+)", re.IGNORECASE)
TS_GF_MEAN_STD = re.compile(
    r"Lopez Sancho.*Mean/std iterations:\s*(\S+)\s*/\s*(\S+)", re.IGNORECASE)
TS_GF_MIN_MAX = re.compile(
    r"Lopez Sancho.*Min/Max iterations\s*:\s*(\d+)\s*/\s*(\d+)",
    re.IGNORECASE)

#: SIESTA's own Makov-Payne monopole term (``Src/write_subs.F``):
#: ``siesta: Emadel  = <eV>`` -- zero unless SIESTA applied it (a molecule
#: in a simple, face- or body-centred cubic cell, ``Src/madelung.f``).
EMADEL = re.compile(r"^\s*siesta:\s+Emadel\s*=\s*(\S+)", re.IGNORECASE)


def ts_charge_values(text: str) -> List[float]:
    """The numbers on a charge-distribution row: one, or one per spin
    channel (and the total, on the ``[Q]`` row); ``[]`` if any is not one."""
    out: List[float] = []
    for tok in text.split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def ts_q_line(line: str) -> Optional[Tuple[str, list]]:
    """One ``ts-q:`` line: ``("names", [D, E1, C1, ..., dQ])`` for the header,
    ``("values", [floats])`` for a row, ``None`` when the line is not one."""
    m = TS_Q_ROW.match(line)
    if not m:
        return None
    toks = m.group(1).split()
    try:
        return "values", [fortran_float(t) for t in toks]
    except ValueError:
        return "names", toks


def ts_q_row(names: Optional[List[str]],
             values: List[float]) -> Optional[Dict[str, Any]]:
    """A ``ts-q:`` row by the header's NAMES -- so a third electrode is a
    third pair of columns and not a reader change -- with the totals apart:
    ``{charges: {D, E1, C1, ...}, dq, qup_minus_qdn}``; ``None`` when the row
    does not pair with the header."""
    if not names or len(names) != len(values):
        return None
    row = dict(zip(names, values))
    dq_name, moment_name = TS_Q_TOTALS
    out: Dict[str, Any] = {}
    dq, moment = row.pop(dq_name, None), row.pop(moment_name, None)
    out["charges"] = row
    if dq is not None:
        out["dq"] = dq
    if moment is not None:
        out["qup_minus_qdn"] = moment
    return out


# ---- TBtrans (the transmission rung) -------------------------------------------
#: ``Util/TS/TBtrans/m_tbt_kpoint.F90`` states the transverse k-points and how
#: they were chosen; ``m_tbt_trik.F90`` the time each spin pass of the loop
#: took -- TBtrans runs its whole loop once per spin channel
#: (``m_tbtrans.F90``), printing ``tbt: Completed in`` and then that pass's
#: currents.
TBT_KPOINTS = re.compile(
    r"^\s*tbt:\s*Number of transport k-points\s*=\s*(\d+)", re.IGNORECASE)
TBT_KMETHOD = re.compile(r"^\s*tbt:\s*Method\s*=\s*(.+?)\.?\s*$",
                         re.IGNORECASE)
TBT_COMPLETED = re.compile(r"^\s*tbt:\s*Completed in\s+(\S+)\s*s",
                           re.IGNORECASE)
#: ``L -> R, V [V] / I [A]: 0.400000     V / 0.309835E-04 A`` and the ``P [W]``
#: line beside it (``m_tbt_save.F90``, one pair per electrode pair per pass):
#: the bias TBtrans itself applied -- the electrodes' chemical-potential
#: difference -- and what flowed.
TBT_CURRENT = re.compile(
    r"^\s*(\S+)\s*->\s*(\S+),\s*V \[V\] / I \[A\]:\s*(\S+)\s*V\s*/\s*(\S+)\s*A",
    re.IGNORECASE)
TBT_POWER = re.compile(
    r"^\s*(\S+)\s*->\s*(\S+),\s*V \[V\] / P \[W\]:\s*(\S+)\s*V\s*/\s*(\S+)\s*W",
    re.IGNORECASE)
#: The transmission files, per spin channel (``Util/TS/TBtrans/m_tbt_save.F90``
#: ``name_save``): ``<label>.TBT.AVTRANS_<E1>-<E2>`` unpolarized, and
#: ``<label>.TBT_UP.`` / ``<label>.TBT_DN.`` for the two channels of a
#: polarized run -- which a reader globbing ``.TBT.`` alone never finds.
#: Written beside the ``.TBT.nc`` by ``state_cdf2ascii`` when NetCDF-4 is
#: built in, as it is in the packaged SIESTA.  A deck setting
#: ``TBT.Directory`` would move them; molbuilder's decks set none.
TBT_CHANNELS = (("unpolarized", ".TBT."), ("up", ".TBT_UP."),
                ("down", ".TBT_DN."))

