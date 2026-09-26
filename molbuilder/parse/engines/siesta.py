"""SIESTA .out / .log FileParser.

Absorbed from the legacy ``molbuilder.parsers.siesta.SiestaParser``;
that package was deleted 2026-06-21 and this is the only SIESTA output
parser (provenance: `docs/archive/old_docs/protocols/parse-module.md` §
8).

For each completed CG/MD step the parser extracts:

  * coordinates      -- from ``outcoor: Atomic coordinates (Ang):`` blocks
  * total energy     -- from ``siesta: E_KS(eV) = ...``  (eV)
  * per-atom forces  -- from ``siesta: Atomic forces (eV/Ang):`` blocks
  * max force        -- from the ``Max <value>`` line that appears after
                        the per-atom force block (skipping the duplicate
                        line ending with ``constrained``)

Also captures the most recent unit-cell vectors from
``outcell: Unit cell vectors (Ang):`` blocks so the viewer can draw the
lattice.

A TranSIESTA device runs TWO SCF phases in one ``.out`` -- SIESTA's periodic
initialization (``scf:`` rows) and then the NEGF loop (``ts-scf:`` rows, each
preceded by its ``ts-q:`` charges and ``ts-Vha:`` correction).  Every cycle
carries its ``phase``, and the device's energy and convergence are its NEGF
phase's (`model/parse.md` § 5d.5-5d.6).  The line patterns are
``siesta_grammar``'s, the one table the wrapper and the monitor are rendered
from too.

Tolerant to in-progress + malformed files (Level 3 contract,
2026-05-28):
  * if the outcoor block is mid-write at EOF the partial frame is dropped
  * if a step has no energy / force yet, ``None`` is stored so per-step
    arrays stay index-aligned with frames
  * a malformed numeric line (SIESTA-side format glitch, partial flush,
    unexpected column count) becomes a :class:`ParseWarning` on
    ``Trajectory.parse_warnings`` instead of aborting the whole parse
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


from molbuilder.frame import Frame, ParseWarning, Trajectory
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import TrajectoryResult
from molbuilder.structure import Structure

from ._helpers import wrap_trajectory
from . import siesta_grammar as _G
from ._section_rules import (
    CONTINUE, END_BUBBLE, END_SECTION,
    SectionRule, any_of, compile_rules, contains_ci, matches_regex_ci,
    starts_with_ci,
)


# Runtime info detection (cross-cutting -- same display path as
# molwatch's runtime header):
#   * "* Running on  N nodes in parallel." / "* Running in serial mode"
#                                           -> n_mpi_processes
#   * Echoed .fdf comments "# runtime.<k>: <v>" -> all the user-set caps
# The launch lines are `siesta_grammar`'s (``read_launch_line``): TBtrans
# prints the same ones.  A "Running on host:" probe stood here until
# 2026-09-26; no SIESTA source prints that line.
# The runtime header is the SAME line format as the molwatch log's, so
# /spectra script writers and Build SIESTA writers emit IDENTICAL lines
# (cf. molbuilder.runtime_info, which owns the write side).  A private
# `_SIESTA_RUNTIME_RE` copy stood here until 2026-09-05; the grammar is
# read by `molwatch.parse_runtime_line`, which owns it.

# Convergence-target echo lines from SIESTA's ``redata:`` preamble.
# Captured into ``runtime_info["convergence_targets"]`` so the Results
# tab can render the threshold line + "current vs target" text without
# the user having to load the source .fdf next to the run.  Each entry
# matches ONE SIESTA echo line; values are floats / ints, no units in
# the captured string (units are documented per key on the receiving
# side).  Refs: SIESTA manual § 6 ("Output") -- ``redata:`` lines are a
# stable contract across SIESTA 4.x and 5.x.
_SIESTA_FORCE_TOL_RE = re.compile(
    r"^\s*redata:\s+Force tolerance\s+=\s+([0-9.eE+-]+)\s+eV/Ang", re.IGNORECASE)
_SIESTA_DM_TOL_RE = re.compile(
    r"^\s*redata:\s+DM tolerance for SCF\s+=\s+([0-9.eE+-]+)", re.IGNORECASE)
_SIESTA_MAX_SCF_RE = re.compile(
    r"^\s*redata:\s+Max\. number of SCF Iter\s+=\s+(\d+)", re.IGNORECASE)
_SIESTA_MAX_DISPL_RE = re.compile(
    r"^\s*redata:\s+Max atomic displ per move\s+=\s+([0-9.eE+-]+)\s+Ang", re.IGNORECASE)
# Geometry-optimisation step cap (MD.NumCGsteps / MD.Steps).  SIESTA
# echoes this as ``redata: Maximum number of optimization moves = N``
# regardless of which MD.TypeOfRun produced it.  Without this,
# runtime_info.convergence_targets carries max_scf_iter but not the
# OPTIMIZATION-level cap, so the trajectory inspector's convergence
# summary was missing the "geom steps cap" row even though the JS
# rendering code (lib/trajectory/core.js _renderConvergenceSummary)
# was ready to display it.  Added 2026-06-13 after the user reported
# the gap.
_SIESTA_MAX_OPT_RE = re.compile(
    r"^\s*redata:\s+Maximum number of optimization moves\s+=\s+(\d+)",
    re.IGNORECASE)

# SIESTA header probes -- the binary's self-report at the top of the .out,
# captured into ``runtime_info['siesta_build']`` so the Results tab can show
# what SIESTA actually ran with, not what the user requested.  The patterns
# and their one reader are `siesta_grammar`'s (``read_build_line``): TBtrans
# prints the same header, and two copies of it would drift.

# Diagonalizer echoes from the redata: block.  SIESTA echoes every
# diagonalization-affecting input the binary actually consumed -- this
# is the ground truth for which solver path the run took, NOT what the
# user wrote in the .fdf (those can drift when SIESTA's parser
# normalises or rejects).  Used to populate
# ``runtime_info['siesta_diag']``.
# The diag patterns live in `_diag.py`, which `bench/result.py` also reads.
# These were three byte-identical private copies until 2026-09-18.
from ._diag import GPU_DEVICE as _SIESTA_GPU_DEVICE_RE       # noqa: E402
from ._diag import REQUESTED as _DIAG_REQUESTED              # noqa: E402

_SIESTA_DIAG_ALGO_RE = _DIAG_REQUESTED["algorithm"]
_SIESTA_DIAG_ELPA_GPU_RE = _DIAG_REQUESTED["elpa_gpu"]

# Validation vocabularies for the loose-capture build/diag probes.  We
# still record whatever SIESTA printed, but a token OUTSIDE these sets
# is flagged as a ParseWarning rather than silently becoming "what the
# binary ran with" -- a future SIESTA print shape we don't understand
# must surface, not masquerade as ground truth (`model/parse.md`
# § 7 #9, no silent absorption; audit-2026-06-26 T1 BLOCKER 3).
#
# Parallelisation modes SIESTA prints on its ``Parallelisations:`` line.
# "none" is what a build with neither prints (Src/version-info-template.inc).
_SIESTA_PARALLELISATIONS = frozenset({"MPI", "OPENMP", "NONE"})
# Diag.Algorithm vocabulary -- the COMPLETE case-insensitive alias set
# SIESTA accepts (uppercased here), transcribed from the binary's own
# parser at Src/diag_option.F90 (read_diag).  SIESTA ``die()``s on any
# value outside this set, so a real run can only echo one of these; an
# out-of-set value means a future SIESTA added an alias we don't track.
_SIESTA_DIAG_ALGORITHMS = frozenset({
    "D&C", "DIVIDE-AND-CONQUER", "DANDC", "VD",
    "D&C-2", "D&C-2STAGE", "DIVIDE-AND-CONQUER-2STAGE",
    "DANDC-2STAGE", "DANDC-2", "VD_2STAGE",
    "ELPA-1", "ELPA-1STAGE",
    "ELPA", "ELPA-2STAGE", "ELPA-2",
    "MRRR", "RRR", "VR",
    "MRRR-2STAGE", "RRR-2STAGE", "MRRR-2", "RRR-2", "VR_2STAGE",
    "EXPERT", "VX",
    "EXPERT-2STAGE", "EXPERT-2", "VX_2STAGE",
    "NOEXPERT", "QR", "V",
    "NOEXPERT-2STAGE", "NOEXPERT-2", "QR-2STAGE", "QR-2", "V_2STAGE",
})

# The SCF iteration row of either phase -- SIESTA's periodic ``scf:`` and
# TranSIESTA's NEGF ``ts-scf:`` -- is the one grammar's (``_G.scf_row``).
# Its float columns are parsed by ``_parse_scf_floats`` below, which takes
# the ordinary form (6 values after iscf) AND the ``Spin.Fix`` form (7 -- Ef
# split into Ef_up + Ef_dn, ``Src/write_subs.F``).

# SIESTA timer lines emitted right after each SCF cycle.  Format:
#   timer: Routine,Calls,Time,% = IterSCF        1      40.820  49.49
# We capture CUMULATIVE Calls and Time (since the start of the run);
# the per-iteration time is computed downstream as the delta between
# successive cycles' cumulative values.  Format observed across
# SIESTA 4.1 / 5.0 / 5.4: ``Routine,Calls,Time,%`` header is fixed,
# then ``= <name>`` then four whitespace-separated fields.  We only
# care about the IterSCF row (other timer rows like ``Setup``,
# ``PostSCF`` etc. are emitted at different cadences and aren't
# load-bearing for the live SCF chart).
# Loose start matcher: just the prefix through ``IterSCF``.  Field
# parsing happens in the handler via ``_parse_fortran_float`` so a
# Fortran column overflow ("******") in Time or % degrades gracefully
# (NaN -> field omitted) rather than dropping the whole attribution.
_TIMER_ITERSCF_RE = re.compile(
    r"^\s*timer:\s*Routine,Calls,Time,%\s*=\s*IterSCF\s+(.+)$",
    re.IGNORECASE,
)

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
#     token fails ``float()`` AND ``_parse_fortran_float`` (which
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
_SCF_TIGHT_PACK_RE = re.compile(r"(\.\d{6})(?=\S)")

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
_FORTRAN_OVERFLOW_RE = re.compile(r"^\*{2,}$")


def _parse_fortran_float(tok: str) -> float:
    """``float(tok)`` but with one Fortran-only escape hatch.

    A token of all asterisks (``**********``) is what Fortran writes
    when a value can't fit its fixed-width format field.  In that
    case the magnitude is gone but the cycle around it is still
    interesting (often diagnostic of divergence).  Return NaN so the
    caller can keep parsing the row instead of dropping it.  Every
    other unparseable token still raises ``ValueError`` -- this is
    NOT a permissive ``try: float ... except: NaN`` blanket.
    """
    if _FORTRAN_OVERFLOW_RE.match(tok):
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


def _parse_scf_floats_by_columns(
    line: str, data_start: int,
) -> Optional[List[float]]:
    """Recover the SCF row's floats by Fortran column position.

    SIESTA writes the SCF row using format ``(3F16.6, 3F10.6)`` for
    closed-shell or ``(3F16.6, 4F10.6)`` for spin-polarized.  Each
    F-field has a fixed character width regardless of the value's
    magnitude, so we can slice at known boundaries even when:

      * Multiple values run together with no whitespace separator
        (the leading sign / digits of the next column consume all
        of its leading-space padding -- see ``_SCF_TIGHT_PACK_RE``
        for the pure-text recovery, this is the structural fallback)
      * A column contains a Fortran overflow indicator
        (``**********``) that ``_parse_fortran_float`` decodes as NaN
      * The whitespace recovery would mis-bind tokens (e.g. a
        future SIESTA format variant we haven't characterised yet)

    ``data_start`` must be the position in ``line`` immediately AFTER
    the iscf integer -- i.e. the first character of the first F16.6
    field, INCLUDING its leading whitespace.  Callers obtain this
    from the grammar's ``scf_row(line).columns_at``.

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
            floats.append(_parse_fortran_float(chunk))
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
            floats.append(_parse_fortran_float(chunk))
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


def _parse_scf_floats(
    rest: str,
    *,
    line: Optional[str] = None,
    data_start: Optional[int] = None,
) -> Optional[List[float]]:
    """Tokenize the post-``scf: <iscf>`` part of an SCF line into
    floats.  Returns the list, or None if both recovery layers fail.

    Two-layer recovery:

      1. **Whitespace recovery** (the fast path).  ``_SCF_TIGHT_PACK_RE``
         inserts a separator after each ``.NNNNNN`` six-decimal field
         when the next column begins with a sign / digit / asterisk.
         Then ``split()`` + per-token ``_parse_fortran_float``.
         Handles the common cases:
           * Tight-packed columns where Ef or dHmax filled their
             whole F10.6 widths and bumped into the next column.
           * Fortran field overflow (``**********``) tokens decoded
             as NaN so the rest of the row survives.

      2. **Column-position fallback** (the safety net).  When the
         whitespace recovery yields the wrong number of floats
         (i.e. a glue pattern the regex didn't catch),
         ``_parse_scf_floats_by_columns`` slices the line at the
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
    fixed = _SCF_TIGHT_PACK_RE.sub(r"\1 ", rest)
    try:
        vals = [_parse_fortran_float(t) for t in fixed.split()]
    except ValueError:
        vals = None
    if vals is not None and len(vals) in _SCF_VALID_VALUE_COUNTS:
        return vals
    # Layer 2: column-position fallback.  Optional context lets
    # legacy callers (tests with synthetic ``rest`` strings) still
    # use the regex path alone; callers that have the full line
    # get the more robust slicing.
    if line is not None and data_start is not None:
        cols = _parse_scf_floats_by_columns(line, data_start)
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
_SCF_COLUMN_KEYS = {
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


def _parse_scf_header(line: str) -> Optional[List[Optional[str]]]:
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
    return [_SCF_COLUMN_KEYS.get(_normalise_column_token(t),
                                  _normalise_column_token(t))
            for t in tokens]


def _build_cycle_dict_from_header(
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


def _build_cycle_dict_positional(
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


class SiestaParser:
    name  = "siesta"
    label = "SIESTA .out / .log"
    hint  = "the main SIESTA run output (run.out, siesta.log, etc.)"

    # `can_parse` is content-based, not banner-based.  SIESTA reshuffles
    # its header text across versions (v4.x had `Welcome to SIESTA`, v5
    # has `*  WELCOME TO SIESTA  *` plus a top-of-file `Executable:
    # siesta` line, future versions may reformat again), so we don't
    # rely on any specific banner string.  We accept the file if EITHER:
    #
    #   1. ANY one strong, content-bearing marker is present in the
    #      first 300 lines.  These are structural elements of SIESTA
    #      output -- block headers (`outcoor:`, `outcell:`), step
    #      banners (`Begin CG opt`), characteristic key lines
    #      (`siesta: System type`, `siesta: Atomic forces`).  They
    #      don't depend on banner text.
    #   2. We see at least 3 lines prefixed by `siesta:` or `redata:`
    #      in those 300 lines.  Real SIESTA output has dozens of such
    #      lines, so 3 is a near-certain match while still rejecting
    #      arbitrary log files that happen to contain the word
    #      "siesta:" once or twice.
    #
    # 300 lines is a generous scan window: a real SIESTA output has
    # plenty of structural markers within the first 100 lines on small
    # runs, and within ~700-800 on big v5 runs whose preamble grew.
    # Strong content markers (case-insensitive substring match; see
    # can_parse).  v4.x banner ("Welcome to SIESTA") and v5.x banner
    # ("WELCOME TO SIESTA") were enumerated separately pre-2026-05-29;
    # the case-insensitive lookup collapses them.  Listed lower-case
    # here because the matcher lower-cases its input.
    _STRONG_MARKERS = (
        "welcome to siesta",            # v4.x / v5.x banner (either case)
        "siesta: system type",
        "siesta: atomic forces",
        "outcoor: atomic coordinates",
        "outcell: unit cell vectors",
        "begin cg opt",
        "begin md opt",
        "begin broyden opt",
        "begin fire opt",
    )
    _PREFIX_MARKERS = ("siesta:", "redata:")
    _SCAN_LINES = 300
    _PREFIX_THRESHOLD = 3

    @classmethod
    def can_parse(cls, path: str) -> bool:
        # ``.molwatch.log`` files have an unambiguous first-line
        # header (``# molwatch trajectory log``) but can also carry
        # legitimate ``siesta:`` / ``redata:`` lines inside step
        # blocks (the engine echoes them).  Belongs strictly to
        # MolwatchLogFileParser; reject by extension to keep the
        # registry dispatch unambiguous.  Case-insensitive guard
        # against filesystem-case quirks (Windows / case-insensitive
        # mounts may surface ``.MOLWATCH.LOG``).
        if str(path).lower().endswith(".molwatch.log"):
            return False
        try:
            with open(path, "r", errors="replace") as fh:
                head_lines = [next(fh, "") for _ in range(cls._SCAN_LINES)]
        except OSError:
            return False
        # Lower-case the head ONCE; cheap (~30 KB of text) and lets
        # the marker / prefix checks below run as plain substring
        # ops with no per-line .lower() amortisation.  Consistent
        # with the rule-table case-insensitivity policy (#171).
        head_lower = "".join(head_lines).lower()
        head_lines_lower = [ln.lower() for ln in head_lines]
        # 0. A v5 build header NAMES the program (`siesta_grammar.EXECUTABLE`,
        #    the path it was invoked by): SIESTA's is ours, and TBtrans's --
        #    the same header, the transmission rung's -- is not.
        for ln in head_lines:
            m = _G.EXECUTABLE.match(ln)
            if m:
                prog = os.path.basename(m.group(1)).lower()
                if prog.startswith("siesta"):
                    return True
                if prog.startswith("tbtrans"):
                    return False
                break
        # 1. Any strong content marker wins immediately.
        if any(m in head_lower for m in cls._STRONG_MARKERS):
            return True
        # 2. Otherwise, count `siesta:` / `redata:` lines.
        prefix_hits = sum(
            1 for ln in head_lines_lower
            if any(ln.lstrip().startswith(p) for p in cls._PREFIX_MARKERS)
        )
        return prefix_hits >= cls._PREFIX_THRESHOLD

    @classmethod
    def parse(cls, path: str) -> Trajectory:
        from molbuilder.parse._log import ParseLogger
        with ParseLogger(path, parser_name="siesta") as _scan_log:
            return cls._parse_impl(path, _scan_log)

    @classmethod
    def _parse_impl(cls, path: str, _scan_log) -> Trajectory:
        frames: List[Frame] = []
        lattice: Optional[List[List[float]]] = None
        pending_lattice: Optional[List[List[float]]] = None
        # HOW THE RUN ENDED (`model/parse.md` § 2b, P-S1) -- a fact about
        # the process, never a grade for the science:
        #   "out_of_memory" -- an OOM marker matched.
        #   "stopped"       -- a fatal marker matched: it did not reach
        #                      its own end.
        #   "ended"         -- ">> End of run" emitted.
        #   "running"       -- no ending marker and no fault.  Content
        #                      cannot tell a slow step from a killed job;
        #                      `parse/dirs/job.py` settles that by age.
        # Convergence is NOT consulted here.  It rides out separately as
        # `scf_converged` (P-S2), because a capped benchmark that never
        # converges still ended perfectly normally.
        # SIESTA's clean-exit marker is "always written" only on
        # success; abort emits at least one of the fatal markers we
        # recognise below (per the 2026-05-29 user directive: detect
        # convergence / exit failures from the .out itself so the
        # /results badge can show "Error" without depending on the
        # wrapper's grep).
        run_state: str = "running"   # parse.md 2b, P-S1
        error_message: Optional[str] = None
        # Per-SCF-block convergence flag.  None = never saw an SCF
        # block; True = last block converged; False = last block hit
        # "SCF did NOT converge" / "SCF_NOT_CONV".  Used at EOF to
        # decide whether a torn run (no End-of-run marker) was a
        # silent abort vs an explicit non-convergence error.
        last_scf_converged: Optional[bool] = None
        # The ``SCF_NOT_CONV:`` line, held aside.  It is the most
        # INFORMATIVE cause-of-abort SIESTA prints, but it is not itself
        # proof of one (see `_on_scf_fatal_not_converged`), so it is kept
        # here and promoted to ``error_message`` only by whatever does
        # prove it -- a fatal marker, or the strict EOF check.
        scf_not_conv_line: Optional[str] = None
        # Runtime facts.  Empty dict when SIESTA didn't log any
        # (older / barebones builds).  Populated from two sources:
        # (a) SIESTA's own startup banner (`* Running on N nodes…`),
        # (b) echoed `# runtime.<k>:` comments from the .fdf -- those
        # come from molbuilder.runtime_info's canonical keys + the
        # SIESTA-specific omp_threads_requested + max_memory_mb.
        runtime_info: Dict[str, Any] = {}
        # Level-3 fail-soft accumulator: every non-fatal line-parsing
        # issue lands here as a ParseWarning and the parser continues.
        # The Results tab surfaces the list in a collapsible panel.
        parse_warnings: List[ParseWarning] = []

        def _warn(line_no: int, line: str, error: str,
                  category: str = "scf") -> None:
            parse_warnings.append(ParseWarning(
                line_no=line_no,
                snippet=line.rstrip()[:120],
                error=error,
                category=category,
            ))
            _scan_log.warn(error, line_no=line_no,
                           snippet=line.rstrip()[:120],
                           category=category)

        # SCF iteration history accumulator for the current step.  Each
        # entry is a per-cycle dict matching the schema in
        # docs/model/parse.md.  SIESTA's column set differs from
        # PySCF (dHmax / dDmax instead of |g| / |ddm|); the UI picks
        # the right residual to plot based on which keys are present.
        # Flushed onto Frame.scf_history at commit() time, then reset.
        current_scf: List[Dict[str, float]] = []
        prev_E_KS: Optional[float] = None
        # The most-recently-seen SCF column header, parsed into
        # canonical keys.  None until we encounter the first ``iscf
        # Eharris ...`` line; from then on, each ``scf:`` data row is
        # mapped by name through this list.  SIESTA emits the header
        # once per geometry step (sometimes once per file), so we keep
        # the latest -- subsequent data rows are interpreted against
        # the most recent header.
        scf_header: Optional[List[Optional[str]]] = None

        # THE TWO PHASES OF A TRANSIESTA DEVICE (`model/parse.md` § 5d.5).  The
        # phase of the last SCF row; what an iteration printed BEFORE its row
        # -- the ``ts-q:`` charges (NEGF only) and the ``ts-Vha:`` correction
        # (both phases), reported while the iteration is built and the row
        # printed after -- held until that row arrives; whether each phase
        # converged; and the start-up facts TranSIESTA states about itself.
        current_phase: Optional[str] = None
        pending_cycle: Dict[str, Any] = {}
        ts_q_names: Optional[List[str]] = None
        phase_converged: Dict[str, Optional[bool]] = {}
        ts_info: Dict[str, Any] = {}
        ts_echo_open = False
        ts_echo_where: List[Optional[str]] = ["options", None]
        ts_gf_electrode: Optional[str] = None
        ts_charge_take = False

        # Per-step buffers; flushed via _commit() when the next outcoor:
        # arrives or at EOF (only if the coords block is known to be
        # complete).
        step_frame: Optional[List[List[Any]]] = None
        step_energy: Optional[float] = None
        step_max_force: Optional[float] = None
        # 2026-06-12: SIESTA also emits a "Max <val> constrained" line
        # right after the unconstrained "Max <val>" when at least one
        # atom is constrained.  This second value is what SIESTA
        # compares against ``MD.MaxForceTol`` for relaxation
        # convergence — meaning when constraints exist the
        # unconstrained max is informational and the constrained max
        # is the real "did we converge?" signal.  Parse + propagate.
        step_max_force_constrained: Optional[float] = None
        step_forces: List[List[float]] = []
        # 2026-06-14: SIESTA emits ``siesta: Etot = <value>`` in its
        # ``Program's energy decomposition`` block right after the
        # initial DM is built but BEFORE the first SCF cycle of
        # each geometry step.  We capture it so the Results-tab
        # plot has a data point during the brief "DM built, SCF
        # not yet started" window of a fresh run -- otherwise the
        # plot stays blank until the first ``scf: 1`` line lands
        # (10-60 s for a 200-atom system).  Same idempotent fall-
        # back as the SCF-cycle path: only consulted when the
        # canonical ``E_KS(eV) =`` line hasn't been written yet
        # AND no SCF cycle exists either.
        step_initial_etot: Optional[float] = None

        def commit(in_progress: bool = False) -> None:
            """Commit the accumulated step state as a Frame.

            ``in_progress`` is set True by the EOF flush when the file
            ends mid-step (step has a structure echo + SCF cycles but
            no canonical step-end signal yet).  Without this flag the
            EOF-flushed frame would be indistinguishable from a real
            completed geom step in the Results-tab trajectory slider,
            so the user would see a "step 0" frame in the slider
            during the very first SCF cycle and the inspector would
            show a real-but-stale energy/force display while the
            actual run is still spinning up.

            See the EOF block below for the detection logic.
            """
            nonlocal step_frame, step_energy, step_max_force, step_forces
            nonlocal step_max_force_constrained
            nonlocal current_scf, prev_E_KS, step_initial_etot
            if not step_frame:
                step_frame = None
                step_energy = None
                step_max_force = None
                step_max_force_constrained = None
                step_forces = []
                # Don't reset current_scf or step_initial_etot here --
                # a torn frame at EOF has no Frame to attach to, but
                # otherwise both belong to a NOT-YET-committed frame
                # (SCF runs *before* outcoor in SIESTA's stream, and
                # the preamble Etot is written even earlier; commit()
                # bailing on an empty step_frame at the start of a run
                # is normal, and the accumulated SCF/Etot state must
                # survive into the next commit attempt).
                return
            elements  = [row[0] for row in step_frame]
            positions = np.array([row[1:4] for row in step_frame],
                                 dtype=float)
            struct = Structure(elements=elements, positions=positions)
            forces_arr = (np.asarray(step_forces, dtype=float)
                          if step_forces else None)
            # 2026-06-14: when SIESTA hasn't yet emitted
            # ``siesta: E_KS(eV) = ...`` (the first geom step is
            # still mid-SCF), fall back to the most recent FINITE
            # energy from the SCF cycle history.  Otherwise the
            # Results-tab energy plot is blank during the entire
            # first geometry step -- which for a heavy system (like
            # BDT-Au-junction) can be many tens of minutes.
            #
            # Robustness rules (each one was a real edge case
            # observed in production .out files, NOT speculative):
            #
            #   * ``step_energy is not None``: nothing to fall back
            #     to, frame.energy is canonical -- skip the fallback
            #     entirely.
            #   * ``current_scf`` empty / None: very first run,
            #     SIESTA hasn't even started cycling.  Leave None;
            #     plot draws nothing for this frame (correct).
            #   * Last cycle's energy is None / not numeric: defen-
            #     sive against a parser-side promotion failure.
            #   * Last cycle's energy is NaN / inf: ``_parse_fortran
            #     _float`` returns NaN on a SIESTA fixed-width over-
            #     flow.  Skip over those and keep walking backward
            #     until we find a finite value.  The user's last-
            #     known-good energy is more useful than NaN, and the
            #     overflow itself is visible elsewhere (the SCF cycle
            #     plot + the dHmax column).
            #   * All cycles infinite/NaN: leave None, plot blank.
            # Energy resolution order (each lower step is a fallback
            # for the one above):
            #
            #   1. canonical ``E_KS(eV) = ...`` line (step_energy).
            #   2. most recent FINITE SCF-cycle energy.
            #   3. preamble ``siesta: Etot = ...`` (post-initial-DM,
            #      pre-first-SCF).  Same numeric value SIESTA will
            #      use for scf:1 once it appears, so the plot is
            #      continuous when scf:1 lands a few seconds later.
            #
            # The three sources are listed from "most converged" to
            # "least converged"; we use the strongest available
            # signal.  None of them is invented -- each is a number
            # SIESTA itself wrote into the .out.
            frame_energy = step_energy
            if frame_energy is None and current_scf:
                for cycle in reversed(current_scf):
                    candidate = cycle.get("energy")
                    if (isinstance(candidate, (int, float))
                            and math.isfinite(candidate)):
                        frame_energy = float(candidate)
                        break
            # A DEVICE SPEAKS FOR ITS NEGF PHASE.  When the step ran
            # TranSIESTA's loop its energy is the last NEGF row's: SIESTA's own
            # closing ``E_KS(eV)`` line and the periodic initialization's
            # cycles are other quantities -- the two formulas already differ
            # by 66,041 eV at the first NEGF step -- and reporting the periodic one
            # is how a device 584 electrons short read as a sound -437,029 eV
            # (§ 5d).  No finite NEGF energy is ``None``: a divergence is not
            # covered by a number from the other phase.
            _negf = [c for c in current_scf
                     if c.get("phase") == _G.PHASE_NEGF]
            if _negf:
                frame_energy = next(
                    (float(c["energy"]) for c in reversed(_negf)
                     if isinstance(c.get("energy"), (int, float))
                     and math.isfinite(c["energy"])), None)
            # PR 4 (results-state-contract § 6): no preamble-Etot
            # fallback.  If the SCF ran without a finite energy, the
            # run is diverging; using ``step_initial_etot`` as a
            # frame energy would HIDE the divergence behind a
            # plausible-looking number.  ``frame_energy`` stays
            # ``None`` (-> JSON null -> Plotly gap in the line ->
            # user sees the divergence).  The preamble Etot is
            # preserved in ``runtime_info["initial_etot"]`` for
            # display, NOT as a frame energy.
            # SIESTA emits per-SCF-cycle ``timer: ... IterSCF N <cum_s>``
            # lines counting from the START OF THE RUN; we attach that
            # onto the last cycle dict as ``elapsed_s``.  For the CG
            # step's end-of-time we surface the LAST cycle's value --
            # that's when the SCF converged, i.e. when this CG step
            # finished.  Per-CG-step time is then
            # frames[i+1].elapsed_s - frames[i].elapsed_s.
            # None when no SCF cycles in this step (rare: single-shot
            # runs) or when SIESTA didn't emit the timer.
            #
            # ``wall_clock_s`` is left unset ON PURPOSE: a SIESTA .out
            # states the time of day only at its two ends, in local time
            # with no zone (``run_start_local`` / ``run_end_local``), and
            # no step carries one -- so for a frame the honest answer is
            # "this engine cannot say" (parse.md § 2a, P-T2).  Filling
            # it with the elapsed seconds is what made the browser
            # render a 6-minute run as "Dec 31, 5:06 PM".
            frame_elapsed_s: Optional[float] = None
            if current_scf:
                cum = current_scf[-1].get("elapsed_s")
                if isinstance(cum, (int, float)) and math.isfinite(cum):
                    frame_elapsed_s = float(cum)
            frames.append(Frame(
                structure   = struct,
                step_index  = len(frames),
                energy      = frame_energy,
                forces      = forces_arr,
                max_force   = step_max_force,
                max_force_constrained = step_max_force_constrained,
                scf_history = list(current_scf) if current_scf else None,
                elapsed_s   = frame_elapsed_s,
                in_progress = in_progress,
            ))
            # PR 4 (results-state-contract § 6): preserve the
            # preamble Etot into runtime_info BEFORE the reset.
            # Without this hop the value is lost — step_initial_etot
            # is per-step (gets cleared when the next step's preamble
            # runs) and the end-of-parse fallback at the bottom of
            # parse() only catches the IN-FLIGHT step.  Each commit
            # writes its step's value; the LAST committed step wins
            # for finished runs, the in-flight step wins for ongoing
            # runs.  Pinned by
            # tests/test_siesta_frame_energy_fallback.py.
            if (step_initial_etot is not None
                    and math.isfinite(step_initial_etot)):
                runtime_info["initial_etot"] = float(step_initial_etot)
            step_frame = None
            step_energy = None
            step_max_force = None
            step_max_force_constrained = None
            step_forces = []
            step_initial_etot = None
            current_scf = []
            prev_E_KS = None

        # ---- Section rules ---------------------------------------
        # Each rule is a (matcher + optional on_start + optional
        # consume) triple closed over the parser-local state above.
        # The driver below tries each rule's matcher on every
        # scan-state line in registration order, and dispatches
        # multi-line sections through ``consume``.  Case-insensitive
        # matching + per-rule alias lists deliver the
        # "small-spelling/capitalisation tolerance" the user asked
        # for (2026-05-28), without committing to fuzzy / Levenshtein
        # matching (which would invite false positives).
        #
        # ORDER MATTERS.  Place more specific matchers before more
        # general ones.  Concretely: ``outcell: Unit cell vectors``
        # must come BEFORE any future rule keyed on bare ``outcell:``;
        # the ``siesta: E_KS(eV)`` substring matcher must come BEFORE
        # the ``siesta: Atomic forces`` matcher (both substring, both
        # could in principle hit on a single line, though SIESTA
        # never emits them on the same line).
        #
        # Run-state marker.  Always fires in scan; single-line.  In
        # practice SIESTA either crashes (no End-of-run) or finishes
        # cleanly (no fatal marker); the precedence below is the
        # defensible default if both somehow appear, and it is stated
        # once, beside the code that applies it.
        def _on_end_of_run(line: str, line_no: int) -> None:
            nonlocal run_state
            # An abort already seen wins: SIESTA does not print both, but
            # if it somehow did, the abort is the load-bearing fact.
            if run_state not in ("stopped", "out_of_memory"):
                run_state = "ended"
            # The node's time of day at the end (naive: SIESTA states no
            # zone).  The start is `_on_run_start`'s.
            _G.read_launch_line(line, runtime_info)

        def _on_run_start(line: str, line_no: int) -> None:
            _G.read_launch_line(line, runtime_info)

        # Fatal error markers.  Each ``contains_ci`` substring is the
        # canonical SIESTA-emitted phrase that always indicates a
        # non-recoverable failure; the set lives in `siesta_grammar`
        # (`model/parse.md` § 2b) and both readers of it share that table.
        # Set 2026-05-29 per user directive.
        #
        # The wrapper is NOT a third reader of this set, and a comment here
        # claimed it was until 2026-08-26 ("the list mirrors the wrapper's
        # grep heuristic").  `runwrap.py` greps one phrase, `propor: ERROR`,
        # to decide whether a startup crash is worth RETRYING -- a different
        # question, asked while the job is alive.  How a run ended is asked
        # afterwards, and only here.
        def _fatal(state: str):
            """One handler per ending state, built from the shared table."""
            def _handler(line: str, line_no: int) -> None:
                nonlocal run_state, error_message
                # P-S1: it did not reach its own end.  An OOM outranks a
                # generic abort -- the aborts that follow are the cascade,
                # the memory is the cause.
                if run_state != "out_of_memory":
                    run_state = state
                # Keep the FIRST marker: later crashes cascade from the
                # original.  A held ``SCF_NOT_CONV:`` outranks even that --
                # it IS the original cause, where ABNORMAL_TERMINATION and
                # Stopping Program are the cascade the 2026-05-30 change
                # existed to skip past.
                if error_message is None:
                    error_message = (scf_not_conv_line or line.strip()[:200])
            return _handler

        # SCF-block convergence flags.  Two distinct SIESTA emit forms:
        #
        #   1. ``SCF_NOT_CONV: SCF did not converge ... (required).``
        #      -- the CONSTANT-prefixed form.  Only emitted when
        #      ``SCF.MustConverge=true`` (default) and SIESTA is about
        #      to abort.  Always followed in the same run by
        #      ABNORMAL_TERMINATION + Stopping Program from Node lines.
        #      Promoted to FATAL (2026-05-30) so the badge's
        #      error_message carries the informative root-cause line
        #      instead of the cascade "Stopping Program from Node: 0".
        #
        #   2. ``SCF did NOT converge`` (no SCF_NOT_CONV: prefix)
        #      -- informational form.  Can appear during a relax run
        #      that recovers in a later step.  Treated as a soft
        #      flag; only flips to error via the strict EOF check
        #      below (last-block-was-bad + no End-of-run).
        #
        # The success marker ``SCF Convergence by <criterion>`` is
        # unambiguous; just sets the flag.
        def _on_scf_converged(line: str, line_no: int) -> None:
            nonlocal last_scf_converged
            last_scf_converged = True
            # Both phases print this same line (``Src/scfconvergence_test.F``,
            # with ``+dQ`` in the NEGF phase's criteria), so it belongs to the
            # phase of the row before it.
            phase_converged[current_phase or _G.PHASE_PERIODIC] = True

        def _on_scf_continued(line: str, line_no: int) -> None:
            """SIESTA withdrew the convergence it had just printed: until
            the next row, this phase has not converged."""
            nonlocal last_scf_converged
            last_scf_converged = None
            phase_converged[current_phase or _G.PHASE_PERIODIC] = None

        def _on_scf_fatal_not_converged(line: str, line_no: int) -> None:
            """``SCF_NOT_CONV:`` -- the root cause when the run dies, and
            NOT a death certificate on its own.

            This set ``run_state = "error"`` outright until 2026-08-25, on
            the assumption written above it: that the line is emitted only
            with ``SCF.MustConverge=true``, and is *"always followed in the
            same run by ABNORMAL_TERMINATION + Stopping Program"*.

            **A benchmark deck breaks that assumption by design.**  It sets
            ``MaxSCFIterations 3`` and ``SCF.MustConverge .false.`` -- three
            steps, convergence explicitly not required, because what is
            being measured is SECONDS PER ITERATION.  SIESTA prints the
            same ``SCF_NOT_CONV:`` line, then carries straight on ("Using
            DM_out to compute the final energy and forces"), reaches
            ``>> End of run``, and writes ``0_NORMAL_EXIT``.  No abort
            marker anywhere.  Every trial of a healthy sweep was therefore
            reported FAILED, and the Results tab counted 0 of 6 done while
            showing six measured timings.

            So the line is HELD, not acted on.  Whatever actually proves
            the run died -- a fatal marker, or the strict EOF check below
            -- promotes it, and the informative message the 2026-05-30
            change wanted is still the one that surfaces.  A run that
            reaches End-of-run keeps ``finished``, which is what it is.
            """
            nonlocal scf_not_conv_line, last_scf_converged
            if scf_not_conv_line is None:
                scf_not_conv_line = line.strip()[:200]
            last_scf_converged = False
            phase_converged[current_phase or _G.PHASE_PERIODIC] = False

        def _on_scf_not_converged(line: str, line_no: int) -> None:
            """Soft handler for the informational form.  Flags only;
            strict EOF check decides whether the run errored."""
            nonlocal last_scf_converged
            last_scf_converged = False
            phase_converged[current_phase or _G.PHASE_PERIODIC] = False

        # Coords section: multi-line.  on_start flushes prev step
        # (commit()) + resets step_frame; consume parses one atom row
        # per line until a blank / malformed line ends the section.
        def _on_coords_start(line: str, line_no: int) -> None:
            nonlocal step_frame
            commit()
            step_frame = []

        def _consume_coords(line: str, line_no: int) -> str:
            stripped = line.strip()
            if not stripped:
                # Blank-line terminator is canonical; drop it.
                return END_SECTION
            parts = stripped.split()
            if len(parts) < 6:
                # Too few tokens: not an atom row.  The line might
                # itself be the start of the next section (e.g.
                # ``outcell: Unit cell vectors (Ang):`` -> 4 tokens),
                # so re-feed through scan rules.
                return END_BUBBLE
            try:
                x = _parse_fortran_float(parts[0])
                y = _parse_fortran_float(parts[1])
                z = _parse_fortran_float(parts[2])
            except ValueError:
                # Same: the line that ends a torn outcoor block may
                # be ``>> End of run`` -- re-feed so that rule fires.
                # ``_parse_fortran_float`` returns NaN for the
                # all-asterisks Fortran overflow case, so this branch
                # only fires on a genuine section-end / format glitch,
                # not on an overflowed numeric column.
                return END_BUBBLE
            step_frame.append([parts[-1], x, y, z])
            return CONTINUE

        # Cell section: multi-line, exactly 3 vector rows.
        def _on_cell_start(line: str, line_no: int) -> None:
            nonlocal pending_lattice
            pending_lattice = []

        def _consume_cell(line: str, line_no: int) -> str:
            nonlocal lattice, pending_lattice
            parts = line.strip().split()
            if len(parts) < 3:
                # The line that ends the cell block (too few tokens)
                # may itself be the start of the next section --
                # re-feed through scan rules.  Matches pre-refactor
                # fall-through semantics; needed if a future SIESTA
                # format drops the blank line between outcell and
                # the next section header.
                return END_BUBBLE
            try:
                row = [_parse_fortran_float(parts[0]),
                       _parse_fortran_float(parts[1]),
                       _parse_fortran_float(parts[2])]
            except ValueError:
                # Same: a non-vector line ending the cell block may
                # be the next section header.  Fortran-overflow
                # columns parse as NaN via _parse_fortran_float, so
                # this branch fires only on a real format mismatch.
                return END_BUBBLE
            pending_lattice.append(row)
            if len(pending_lattice) >= 3:
                lattice = pending_lattice
                pending_lattice = None
                return END_SECTION
            return CONTINUE

        # Forces section: multi-line, one ``<idx> fx fy fz`` row per
        # atom, ends on a non-conforming row (typically the "Max" line).
        def _on_forces_start(line: str, line_no: int) -> None:
            nonlocal step_forces
            step_forces = []

        def _consume_forces(line: str, line_no: int) -> str:
            parts = line.strip().split()
            if len(parts) < 4:
                # END_BUBBLE: the line that ends the forces section is
                # often the "Max <value>" line OR the next-section
                # header; we want the driver to re-feed it through
                # scan-state rules so max-force / outcoor matchers see it.
                return END_BUBBLE
            try:
                int(parts[0])  # atom index
                fx = _parse_fortran_float(parts[1])
                fy = _parse_fortran_float(parts[2])
                fz = _parse_fortran_float(parts[3])
            except ValueError:
                # _parse_fortran_float NaN-ifies a ``**********``
                # overflow column (likely on a divergent SCF where
                # forces blow up), so this branch is "atom-index
                # column isn't an int" or "next-section header
                # mis-shapes the line" -- the genuine end-of-block
                # cases that should re-feed through scan rules.
                return END_BUBBLE
            step_forces.append([fx, fy, fz])
            return CONTINUE

        # E_KS energy line: single-line.  Format:
        #   ``siesta: E_KS(eV) =       -1234.567``
        def _on_e_ks(line: str, line_no: int) -> None:
            nonlocal step_energy
            try:
                step_energy = _parse_fortran_float(
                    line.split("=", 1)[1].split()[0])
            except (ValueError, IndexError) as exc:
                _warn(line_no, line,
                      f"E_KS line: malformed value: {exc}",
                      category="energy")

        # Initial-DM energy decomposition: SIESTA emits a block
        #
        #     siesta: Program's energy decomposition (eV):
        #     siesta: Ebs     =   ...
        #     ...
        #     siesta: Etot    =   <value>
        #     ...
        #
        # right after the initial DM is built but BEFORE the first
        # SCF cycle of each geometry step.  ``Etot`` is the most
        # useful number to surface as a fallback because it's
        # exactly the value scf:1 will print a few seconds later
        # (same numerical column).  Capturing it lets the Results-
        # tab energy plot show a data point during the brief
        # initialization window of a fresh run.  See the energy-
        # resolution-order comment in commit().
        def _on_initial_etot(line: str, line_no: int) -> None:
            nonlocal step_initial_etot
            try:
                val = _parse_fortran_float(
                    line.split("=", 1)[1].split()[0])
                if math.isfinite(val):
                    step_initial_etot = val
            except (ValueError, IndexError):
                # Malformed Etot in the decomposition block is rare
                # and not worth a parser warning -- the SCF cycle
                # fallback will pick up after first scf:.
                pass

        # SCF column header: single-line.  Records the canonical
        # column layout for the upcoming ``scf:`` data rows.
        def _on_scf_header(line: str, line_no: int) -> None:
            nonlocal scf_header
            parsed = _parse_scf_header(line)
            if parsed is not None:
                scf_header = parsed

        # SCF data row: single-line, fires once per iteration of the
        # SCF cycle.  Pre-2026-05-28 had this in a 70-line inline
        # block; it's now collapsed into one ``on_start`` hook.
        def _on_scf_data(line: str, line_no: int) -> None:
            nonlocal current_scf, prev_E_KS, current_phase
            nonlocal last_scf_converged
            row = _G.scf_row(line)
            if row is None:
                return
            phase, iscf = row.phase, row.iscf
            # ``columns_at`` is the position immediately AFTER the iscf
            # integer in the original line -- i.e. the first character
            # of the first F16.6 field, with its leading whitespace
            # still attached.  The column-position fallback in
            # ``_parse_scf_floats`` needs this so it can slice at
            # the Fortran format boundaries (3 x F16.6 + 3..4 x F10.6).
            vals = _parse_scf_floats(line[row.columns_at:].lstrip(),
                                     line=line, data_start=row.columns_at)
            if vals is None:
                _warn(line_no, line,
                      "SCF line: could not tokenize as floats")
                return

            if scf_header is not None:
                cycle_dict = _build_cycle_dict_from_header(
                    iscf, vals, scf_header)
                if cycle_dict is None:
                    _warn(line_no, line,
                          f"SCF row has {len(vals)} values "
                          f"but header has {len(scf_header)-1} "
                          f"columns ({scf_header})")
                    return
            else:
                cycle_dict = _build_cycle_dict_positional(iscf, vals)
                if cycle_dict is None:
                    _warn(line_no, line,
                          f"SCF line has {len(vals)} floats "
                          f"after iscf; expected 6 (closed-"
                          f"shell) or 7 (spin-polarized), and "
                          f"no column header was seen")
                    return

            e_ks = cycle_dict.get("energy")
            if e_ks is None:
                _warn(line_no, line,
                      "SCF row missing 'energy' (E_KS) -- "
                      "downstream plot can't render this cycle")
                return

            # A NEW PHASE IS NOT A RESTART.  The periodic initialization's
            # cycles stay beside the NEGF loop's, and the new phase starts
            # with its convergence unanswered -- the initialization's "SCF
            # cycle converged" does not speak for the device.
            if phase != current_phase:
                if current_phase is not None:
                    last_scf_converged = None
                current_phase = phase
                prev_E_KS = None
            # iscf==1 starts a new SCF run of THIS phase.  See parse()
            # docstring for the failed-SCF-restart edge case.
            if iscf == 1:
                if current_scf and current_scf[-1].get("phase") == phase:
                    current_scf = []
                prev_E_KS = None
            delta_E = ((e_ks - prev_E_KS)
                       if prev_E_KS is not None else 0.0)
            cycle_dict["delta_E"] = delta_E
            cycle_dict["phase"] = phase
            # What TranSIESTA reported while building this iteration.
            cycle_dict.update(pending_cycle)
            pending_cycle.clear()
            current_scf.append(cycle_dict)
            prev_E_KS = e_ks

        def _on_iter_scf_timer(line: str, line_no: int) -> None:
            """Attach SIESTA's cumulative IterSCF wall-time to the most-
            recent SCF cycle.

            SIESTA emits the timer line RIGHT AFTER each SCF cycle's
            data line.  Format::

                timer: Routine,Calls,Time,% = IterSCF      1      40.820  49.49

            ``Calls`` (=1 here) is the cumulative iteration count and
            ``Time`` (=40.820) is the cumulative wall-time in seconds
            since the run started.  We attach BOTH to the cycle dict
            so the inspector can compute per-iteration deltas (Time_N
            - Time_N-1) AND show the absolute progress.  The JS chart
            does the delta arithmetic so a stale + fresh cycle render
            consistently.

            Defensive in two ways:

            * If the timer line arrives WITHOUT a preceding scf-data
              cycle (e.g. truncated file, weird ordering), drop it
              silently -- attaching wall-time to a non-existent cycle
              would be more misleading than just omitting the field.

            * Fortran column overflow ("******") in any of Calls /
              Time / % does NOT discard the whole attribution -- each
              field is parsed independently via ``_parse_fortran_float``
              (NaN on overflow / non-numeric), and the cycle dict only
              gets the keys that DID parse cleanly.  So a long run that
              overflows Time but not Calls still gets ``cumulative_calls``
              attached; the JS just falls back to the cycle index when
              ``elapsed_s`` is missing.
            """
            nonlocal current_scf
            m = _TIMER_ITERSCF_RE.match(line)
            if m is None:
                return
            if not current_scf:
                return
            # Tokenise the rest of the line: Calls Time % (3 fields).
            # Anything beyond the third is ignored -- some SIESTA builds
            # append extra columns we don't consume.
            tokens = m.group(1).split()
            if not tokens:
                return
            last_cycle = current_scf[-1]
            # Calls (integer): tolerate "*****" by skipping.
            calls_tok = tokens[0]
            if "*" not in calls_tok:
                try:
                    last_cycle["cumulative_calls"] = int(calls_tok)
                except (ValueError, TypeError):
                    pass
            # Time (float seconds): ``_parse_fortran_float`` returns NaN
            # for "*****" so the math.isfinite guard suppresses the
            # attachment cleanly without raising.
            if len(tokens) >= 2:
                cum_time_s = _parse_fortran_float(tokens[1])
                if math.isfinite(cum_time_s):
                    last_cycle["elapsed_s"] = cum_time_s

        # Max-force lines: single-line, both variants.  Gated by a
        # closure-captured check on ``step_forces`` -- only valid
        # after a Forces section closed.  Without the gate a stray
        # "Max <num>" line in the preamble would mis-attribute to
        # the first frame.
        #
        # SIESTA emits TWO forms here:
        #   * ``Max    0.789``                  → 2 tokens, all atoms
        #   * ``Max    0.456    constrained``   → 3 tokens, excludes
        #                                          constrained atoms
        # The constrained form only appears when at least one atom
        # is constrained (frozen).  Both forms route to separate
        # frame fields so the plot can show both traces — the
        # constrained one is what SIESTA actually compares against
        # ``MD.MaxForceTol`` for convergence.
        def _max_force_match(line: str) -> bool:
            if not step_forces:
                return False
            parts = line.strip().split()
            return (len(parts) == 2 and parts[0].lower() == "max")

        def _max_force_constrained_match(line: str) -> bool:
            if not step_forces:
                return False
            parts = line.strip().split()
            return (len(parts) == 3
                    and parts[0].lower() == "max"
                    and parts[2].lower() == "constrained")

        def _on_max_force(line: str, line_no: int) -> None:
            nonlocal step_max_force
            try:
                step_max_force = _parse_fortran_float(
                    line.strip().split()[1])
            except (ValueError, IndexError) as exc:
                _warn(line_no, line,
                      f"Max-force line: malformed value: {exc}",
                      category="forces")

        def _on_max_force_constrained(line: str, line_no: int) -> None:
            nonlocal step_max_force_constrained
            try:
                step_max_force_constrained = _parse_fortran_float(
                    line.strip().split()[1])
            except (ValueError, IndexError) as exc:
                _warn(line_no, line,
                      "Max-force-constrained line: malformed value: "
                      f"{exc}", category="forces")

        # ---- TranSIESTA (`model/parse.md` § 5d.5) ------------------------
        def _on_ts_q(line: str, line_no: int) -> None:
            """``ts-q:`` -- a header naming the regions (``D``, ``E1``,
            ``C1``, ..., ``B``) and the totals (``dQ``, ``Qup-Qdn``), then
            their values, held for the NEGF row that follows.  By NAME, so a
            third electrode is a third pair of columns and not a parser
            change."""
            nonlocal ts_q_names
            m = _G.TS_Q_ROW.match(line)
            if not m:
                return
            toks = m.group(1).split()
            try:
                vals = [_parse_fortran_float(t) for t in toks]
            except ValueError:
                ts_q_names = toks
                return
            if not ts_q_names or len(ts_q_names) != len(vals):
                _warn(line_no, line, "ts-q row with no matching header",
                      category="negf")
                return
            row = dict(zip(ts_q_names, vals))
            dq_name, moment_name = _G.TS_Q_TOTALS
            dq, moment = row.pop(dq_name, None), row.pop(moment_name, None)
            pending_cycle["charges"] = row
            if dq is not None:
                pending_cycle["dq"] = dq
            if moment is not None:
                pending_cycle["qup_minus_qdn"] = moment

        def _on_ts_vha(line: str, line_no: int) -> None:
            m = _G.TS_VHA.match(line)
            if not m:
                return
            try:
                pending_cycle["vha_ev"] = _parse_fortran_float(m.group(1))
            except ValueError:
                _warn(line_no, line, "ts-Vha: malformed value",
                      category="negf")

        def _on_ts_echo(line: str, line_no: int) -> None:
            """The start-up echo -- TranSIESTA's own account of the settings
            it runs with, which is the only place some of them exist: the
            continued-fraction contour's pole count is computed inside
            ``m_ts_chem_pot.F90`` and appears in no input and no fdf log.
            Only lines inside the star frame are the echo; ``ts:`` lines
            later in the run are the energy decomposition."""
            nonlocal ts_echo_open
            if _G.TS_ECHO_FRAME.match(line):
                ts_echo_open = not ts_echo_open
                return
            if not ts_echo_open:
                return
            if _G.TS_ECHO_CONTOUR.match(line):
                ts_echo_where[:] = ["contours", None]
                return
            m = _G.TS_ECHO_SECTION.match(line)
            if m:
                name = m.group(1).strip()
                if name.lower() == "electrodes":
                    ts_echo_where[:] = ["electrodes", None]
                elif ts_echo_where[0] in ("electrodes", "contours"):
                    ts_echo_where[1] = name
                return
            m = _G.TS_ECHO_LINE.match(line)
            where, name = ts_echo_where
            home = _ts_echo_home(where, name,
                                 m.group(1).strip() if m else None)
            if m:
                label, value = m.group(1).strip(), m.group(2).strip()
                # A label a segment states twice -- ``Option for contour
                # method`` -- keeps every value.
                if label in home:
                    prev = home[label]
                    home[label] = (prev if isinstance(prev, list)
                                   else [prev]) + [value]
                else:
                    home[label] = value
                return
            note = line.split(":", 1)[1].strip().strip("*").strip()
            if note:
                home.setdefault("notes", []).append(note)

        def _ts_echo_home(where, name, label):
            """Where an echo line belongs: the options, an electrode, or a
            contour SEGMENT -- one per chemical potential or contour part,
            each with the same labels, a new one starting at the line that
            names it (``siesta_grammar.TS_ECHO_SEGMENT``)."""
            if where == "options" or (where == "electrodes" and name is None):
                return ts_info.setdefault("options", {})
            if where == "electrodes":
                return ts_info.setdefault("electrodes", {}).setdefault(
                    name, {})
            segments = ts_info.setdefault("contours", {}).setdefault(
                name or "", [])
            if not segments or (label and _G.TS_ECHO_SEGMENT.search(label)):
                segments.append({})
            return segments[-1]

        def _on_ts_charge_start(line: str, line_no: int) -> None:
            """The charge distribution at the switch from the periodic
            density -- the baseline each NEGF iteration's charge is judged
            against.  The FIRST report only."""
            nonlocal ts_charge_take
            m = _G.TS_CHARGE_START.match(line)
            ts_charge_take = bool(m) and "charge_at_switch" not in ts_info
            if ts_charge_take:
                try:
                    target = _parse_fortran_float(m.group(1))
                except ValueError:
                    target = None
                ts_info["charge_at_switch"] = {"target": target}

        def _consume_ts_charge(line: str, line_no: int) -> str:
            m = _G.TS_CHARGE_ROW.match(line)
            if not m:
                return END_BUBBLE if line.strip() else END_SECTION
            if ts_charge_take:
                vals = _G.ts_charge_values(m.group(3))
                q = ts_info["charge_at_switch"]
                if len(vals) == 1:
                    q[m.group(2)] = vals[0]
                elif len(vals) in (2, 3):
                    # Spin-polarized (``Src/ts_charge.F90``): up and down,
                    # and on the ``[Q]`` row the total after them.
                    q[m.group(2)] = vals[2] if len(vals) == 3 else sum(vals)
                    q.setdefault("by_spin", {})[m.group(2)] = vals[:2]
                else:
                    _warn(line_no, line, "charge distribution: malformed row",
                          category="negf")
            return CONTINUE

        def _on_ts_principal_cell(line: str, line_no: int) -> None:
            m = _G.TS_PRINCIPAL_CELL.match(line)
            if m:
                ts_info.setdefault("electrodes", {}).setdefault(
                    m.group(1), {})["principal_cell"] = m.group(2)

        def _on_ts_gf_for(line: str, line_no: int) -> None:
            nonlocal ts_gf_electrode
            m = _G.TS_GF_FOR.match(line)
            if m:
                ts_gf_electrode = m.group(1)

        def _on_ts_gf_stats(line: str, line_no: int) -> None:
            if ts_gf_electrode is None:
                return
            el = ts_info.setdefault("electrodes", {}).setdefault(
                ts_gf_electrode, {})
            m = _G.TS_GF_MEAN_STD.search(line)
            if m:
                try:
                    el["gf_iterations_mean"] = float(m.group(1))
                    el["gf_iterations_std"] = float(m.group(2))
                except ValueError:
                    pass
                return
            m = _G.TS_GF_MIN_MAX.search(line)
            if m:
                el["gf_iterations_min"] = int(m.group(1))
                el["gf_iterations_max"] = int(m.group(2))

        def _on_emadel(line: str, line_no: int) -> None:
            m = _G.EMADEL.match(line)
            if m:
                try:
                    runtime_info["emadel_ev"] = _parse_fortran_float(
                        m.group(1))
                except ValueError:
                    pass

        rules: List[SectionRule] = [
            # FATAL MARKERS, FROM THE ONE TABLE (`siesta_grammar.FATAL_MARKERS`,
            # `model/parse.md` § 2b).  They fire FIRST so they win over any
            # section that might otherwise eat the line, and they are
            # substring matches because SIESTA prefixes them with "node 0: "
            # under MPI.
            #
            # Built from the shared table rather than written out here, so
            # the cheap `scan_ending()` scanner and this full parse cannot
            # disagree about what a fatal marker IS.  Two lists is how
            # `jobset/summarize.py` came to hold a private `_DONE_MARKERS`
            # that contradicted this file.
            *[
                SectionRule(
                    name=f"fatal_{_marker.replace(' ', '_').replace(':', '')}",
                    aliases=[_marker],
                    start=contains_ci(_marker),
                    on_start=_fatal(_state),
                )
                for _marker, _state in _G.FATAL_MARKERS
            ],
            # SCF_NOT_CONV: FATAL form.  Must be BEFORE the soft
            # "scf did not converge" rule -- a line containing both
            # ("SCF_NOT_CONV: SCF did not converge ...") must dispatch
            # to the fatal handler, not the soft flag.  The fatal
            # handler captures the informative root-cause line as
            # error_message; subsequent ABNORMAL_TERMINATION + Stop
            # cascades do NOT overwrite it.
            SectionRule(
                name="fatal_scf_not_conv",
                aliases=["SCF_NOT_CONV: ..."],
                start=contains_ci(_G.SCF_NOT_CONV_MARKER),
                on_start=_on_scf_fatal_not_converged,
            ),
            # SCF convergence success.  "by <criterion>" suffix
            # guards against matching a diagnostic line that mentions
            # both "SCF Convergence" and "did NOT converge".
            SectionRule(
                name="scf_converged",
                aliases=["SCF Convergence by ..."],
                start=contains_ci(_G.SCF_CONVERGED_MARKER),
                on_start=_on_scf_converged,
            ),
            # SIESTA taking the convergence back -- TranSIESTA's charge
            # still off, or too few iterations (``Src/siesta_forces.F90``).
            SectionRule(
                name="scf_continued",
                aliases=["SCF cycle continued ..."],
                start=contains_ci(_G.SCF_CONTINUED_MARKER),
                on_start=_on_scf_continued,
            ),
            # SCF non-convergence: SOFT informational form (no
            # SCF_NOT_CONV: prefix).  Can appear during a relax that
            # recovers in a later step; strict EOF check decides
            # error vs ongoing.
            SectionRule(
                name="scf_not_converged",
                aliases=["SCF did NOT converge"],
                start=contains_ci(_G.SCF_NOT_CONVERGED_MARKER),
                on_start=_on_scf_not_converged,
            ),
            # Specific (multi-token) section matchers next.
            SectionRule(
                name="cell",
                aliases=["outcell: Unit cell vectors"],
                start=starts_with_ci("outcell: Unit cell vectors"),
                on_start=_on_cell_start,
                consume=_consume_cell,
            ),
            SectionRule(
                name="end_of_run",
                aliases=[">> End of run"],
                start=matches_regex_ci(_G.RUN_END.pattern),
                on_start=_on_end_of_run,
            ),
            SectionRule(
                name="coords",
                aliases=["outcoor:"],
                start=starts_with_ci("outcoor:"),
                on_start=_on_coords_start,
                consume=_consume_coords,
            ),
            SectionRule(
                name="e_ks",
                aliases=["siesta: E_KS(eV)"],
                # Substring (not prefix): the marker sits mid-line.
                start=contains_ci("siesta: e_ks(ev)"),
                on_start=_on_e_ks,
            ),
            SectionRule(
                name="initial_etot",
                aliases=["siesta: Etot ="],
                # Anchored regex: ``siesta: Etot`` (whitespace) ``=``
                # exact match.  ``contains_ci("siesta: etot")``
                # would also match ``siesta: Etot/N = ...``,
                # ``siesta: Etot(eV) = ...``, and any future SIESTA
                # decomposition variant that starts with the same
                # word -- the wrong row would overwrite the
                # initial-Etot fallback.  Anchor explicitly to
                # the bare ``Etot`` token followed by ``=``.
                start=matches_regex_ci(
                    r"^\s*siesta:\s+Etot\s*=\s*[-\d]"
                ),
                on_start=_on_initial_etot,
            ),
            SectionRule(
                name="forces",
                aliases=["siesta: Atomic forces"],
                start=contains_ci("siesta: atomic forces"),
                on_start=_on_forces_start,
                consume=_consume_forces,
            ),
            SectionRule(
                name="scf_header",
                aliases=["iscf <columns>"],
                # The header always starts with the bare token ``iscf``
                # (case-insensitive).  Same shape as _SCF_HEADER_RE; we
                # use ``matches_regex_ci`` here so the rule participates
                # in the combined-regex pre-filter compiled by
                # :class:`CompiledRules`.
                start=matches_regex_ci(_G.SCF_HEADER.pattern),
                on_start=_on_scf_header,
            ),
            SectionRule(
                name="scf_data",
                aliases=["scf: <iscf> ..."],
                # The row's start, the pattern the tee and the monitor
                # match too; ``_on_scf_data`` asks ``_G.scf_row`` for the
                # rest.  Combined-regex eligible.
                start=matches_regex_ci(_G.SCF_ROW_ERE),
                on_start=_on_scf_data,
            ),
            SectionRule(
                name="iter_scf_timer",
                aliases=["timer: ... IterSCF"],
                # Matches the cumulative ``IterSCF`` timer line SIESTA
                # emits right after each completed SCF cycle so the
                # Results-tab inspector can show per-iteration wall
                # time live -- the canonical "is this run progressing
                # at a reasonable pace?" signal.  See _on_iter_scf_timer
                # for the cumulative-to-delta computation: we attach
                # ``elapsed_s`` to the SCF cycle dict that was just
                # appended; the JS chart computes per-iter deltas from
                # the cumulative series.  The key name is the contract:
                # SIESTA's timer counts from the start of the run, so
                # this is elapsed, never an epoch (parse.md § 2a).
                start=matches_regex_ci(
                    r"^\s*timer:\s*Routine,Calls,Time,%\s*=\s*IterSCF\s"),
                on_start=_on_iter_scf_timer,
            ),
            # TranSIESTA's lines (`model/parse.md` § 5d.5).  None of them can
            # collide with a rule above: each has its own prefix.
            SectionRule(
                name="ts_q",
                aliases=["ts-q: ..."],
                start=matches_regex_ci(_G.TS_Q_ROW.pattern),
                on_start=_on_ts_q,
            ),
            SectionRule(
                name="ts_vha",
                aliases=["ts-Vha: ... eV"],
                start=matches_regex_ci(_G.TS_VHA.pattern),
                on_start=_on_ts_vha,
            ),
            SectionRule(
                name="ts_echo",
                aliases=["ts: ..."],
                start=matches_regex_ci(_G.TS_ECHO.pattern),
                on_start=_on_ts_echo,
            ),
            SectionRule(
                name="ts_charge_distribution",
                aliases=["transiesta: Charge distribution, target = ..."],
                start=matches_regex_ci(_G.TS_CHARGE_START.pattern),
                on_start=_on_ts_charge_start,
                consume=_consume_ts_charge,
            ),
            SectionRule(
                name="ts_principal_cell",
                aliases=["<electrode> principal cell is ..."],
                start=matches_regex_ci(_G.TS_PRINCIPAL_CELL.pattern),
                on_start=_on_ts_principal_cell,
            ),
            SectionRule(
                name="ts_gf_for",
                aliases=["Calculating surface Green functions for: ..."],
                start=matches_regex_ci(_G.TS_GF_FOR.pattern),
                on_start=_on_ts_gf_for,
            ),
            SectionRule(
                name="ts_gf_stats",
                aliases=["Lopez Sancho ... iterations"],
                start=any_of(matches_regex_ci(_G.TS_GF_MEAN_STD.pattern),
                             matches_regex_ci(_G.TS_GF_MIN_MAX.pattern)),
                on_start=_on_ts_gf_stats,
            ),
            SectionRule(
                name="emadel",
                aliases=["siesta: Emadel = ..."],
                start=matches_regex_ci(_G.EMADEL.pattern),
                on_start=_on_emadel,
            ),
            SectionRule(
                name="run_start",
                aliases=[">> Start of run"],
                start=matches_regex_ci(_G.RUN_START.pattern),
                on_start=_on_run_start,
            ),
            SectionRule(
                name="max_force",
                aliases=["Max <value>"],
                start=_max_force_match,
                on_start=_on_max_force,
            ),
            # 2026-06-12: ``Max <value> constrained`` — SIESTA emits
            # this RIGHT AFTER the unconstrained max-force line when
            # at least one atom is constrained.  Excludes the
            # constrained atoms from the max — the value SIESTA
            # actually compares against MD.MaxForceTol.  Registered
            # AFTER ``max_force`` so the 2-token form's matcher gets
            # the first crack (gating on ``len(parts) == 2`` keeps
            # them mutually exclusive at the matcher level, but the
            # registration order is the explicit policy.
            SectionRule(
                name="max_force_constrained",
                aliases=["Max <value> constrained"],
                start=_max_force_constrained_match,
                on_start=_on_max_force_constrained,
            ),
        ]

        # ---- State-machine driver --------------------------------
        # state: either "scan" (try all rules) or the name of an
        # active multi-line rule (only that rule's ``consume`` runs).
        active: Optional[SectionRule] = None

        # Compile rules ONCE into a dispatch table:
        #   * combined-regex pre-filter over every ``_PatternMatcher``
        #     rule -- one DFA scan per scan-state line tests them all;
        #   * per-rule pre-compiled regex (preserves registration-
        #     order tie-break; the combined regex's leftmost-position
        #     match would otherwise silently change semantics);
        #   * predicate-only rules (e.g. ``_max_force_match`` closing
        #     over parser state) are iterated individually.
        compiled = compile_rules(rules)

        def _set_conv_target(key: str, value: Any) -> None:
            """Lazily create ``runtime_info['convergence_targets']`` and
            stamp ``source`` once.  Called by the ``redata:`` echo probes
            below — the Results tab consumes the populated subdict to
            draw threshold lines + the "current vs target" readout."""
            ct = runtime_info.setdefault("convergence_targets", {})
            ct[key] = value
            ct.setdefault("source", "siesta_input_echo")

        def _scan_runtime_info(line: str, line_no: int) -> bool:
            """Orthogonal runtime-info regex probes.  Not section
            boundaries -- just free-form key/value lines that may
            appear anywhere in scan state.  Returns True if the line
            was consumed (caller should skip rule dispatch).

            Loose-capture probes (parallelisations, diag algorithm)
            validate the captured token against a known vocabulary and
            emit a ParseWarning on a miss -- the value is still
            recorded, but an unrecognised token surfaces instead of
            silently becoming ground truth (no silent absorption)."""
            if (_G.RUNNING_ON.match(line)
                    or _G.RUNNING_SERIAL.match(line)):
                _G.read_launch_line(line, runtime_info)
                return True
            # ONE reader of the runtime-header grammar, in the module that
            # owns it.  A character-identical copy of the coercion stood
            # here until 2026-09-05; the format is deliberately shared with
            # the molwatch log, and a shared format read by two copies is
            # how the convergence header drifted on 2026-08-19.
            from .molwatch import parse_runtime_line
            if parse_runtime_line(line, runtime_info):
                return True
            # Convergence-target probes (SIESTA's ``redata:`` echo
            # block at run start).  Each line is matched at MOST
            # once per run; idempotent re-matches are safe.
            m = _SIESTA_FORCE_TOL_RE.match(line)
            if m:
                try:
                    _set_conv_target("max_force_tol_eV_per_A", float(m.group(1)))
                except ValueError:
                    pass
                return True
            m = _SIESTA_DM_TOL_RE.match(line)
            if m:
                try:
                    _set_conv_target("dm_tolerance", float(m.group(1)))
                except ValueError:
                    pass
                return True
            m = _SIESTA_MAX_SCF_RE.match(line)
            if m:
                try:
                    _set_conv_target("max_scf_iter", int(m.group(1)))
                except ValueError:
                    pass
                return True
            m = _SIESTA_MAX_DISPL_RE.match(line)
            if m:
                try:
                    _set_conv_target("max_displ_ang", float(m.group(1)))
                except ValueError:
                    pass
                return True
            m = _SIESTA_MAX_OPT_RE.match(line)
            if m:
                try:
                    _set_conv_target("max_geom_iter", int(m.group(1)))
                except ValueError:
                    pass
                return True
            # ---- SIESTA build-header probes ----------------------- #
            # Populate runtime_info["siesta_build"] -- what the binary
            # self-reports about its compiled-in capabilities.  The
            # Results tab uses this to show "what actually ran" vs the
            # user's requested params.
            build: Dict[str, Any] = {}
            key = _G.read_build_line(line, build)
            if key:
                have = runtime_info.setdefault("siesta_build", {})
                for k, v in build.items():
                    have.setdefault(k, v)
                unknown = [t for t in build.get("parallelisations", ())
                           if t not in _SIESTA_PARALLELISATIONS]
                if unknown:
                    _warn(line_no, line,
                          f"unrecognised SIESTA parallelisation "
                          f"token(s) {unknown}; recorded as-is but not "
                          f"in the known set "
                          f"{sorted(_SIESTA_PARALLELISATIONS)}",
                          category="runtime_info")
                return True
            # ---- SIESTA diagonalizer probes ----------------------- #
            # Populate runtime_info["siesta_diag"] -- which solver path
            # the run actually took (ground truth from redata: echo,
            # which is what SIESTA's own input parser consumed; the user's
            # .fdf may not match if SIESTA normalised/rejected a value).
            m = _SIESTA_DIAG_ALGO_RE.match(line)
            if m:
                algo = m.group(1).strip().upper()
                runtime_info.setdefault(
                    "siesta_diag", {})["algorithm"] = algo
                if algo not in _SIESTA_DIAG_ALGORITHMS:
                    _warn(line_no, line,
                          f"unrecognised SIESTA Diag.Algorithm "
                          f"{algo!r}; recorded as-is but not in the "
                          f"known vocabulary (Src/diag_option.F90) -- "
                          f"likely a newer SIESTA than this parser "
                          f"tracks",
                          category="runtime_info")
                return True
            m = _SIESTA_DIAG_ELPA_GPU_RE.match(line)
            if m:
                raw = m.group(1).strip().lower().strip(".")
                runtime_info.setdefault("siesta_diag", {})["elpa_gpu"] = (
                    raw in ("t", "true"))
                return True
            m = _SIESTA_GPU_DEVICE_RE.match(line)
            if m:
                diag = runtime_info.setdefault("siesta_diag", {})
                diag["gpu_device"] = m.group(1).strip()
                if m.group(2):
                    diag["gpu_compute_capability"] = m.group(2).strip()
                return True
            return False

        with open(path, "r", errors="replace") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.rstrip("\n")

                # Active multi-line section?  Run its ``consume``
                # first.  CONTINUE / END_SECTION skip to next line;
                # END_BUBBLE leaves the section AND re-feeds this
                # line through scan-state rules below.
                if active is not None:
                    sentinel = active.consume(line, line_no)
                    if sentinel == CONTINUE:
                        continue
                    if sentinel == END_SECTION:
                        active = None
                        continue
                    if sentinel == END_BUBBLE:
                        active = None
                        # fall through to scan-state dispatch below
                    else:
                        _warn(line_no, line,
                              f"section {active.name!r} returned "
                              f"unknown sentinel {sentinel!r}; "
                              f"ending section")
                        active = None
                        continue

                # Scan state.  Runtime-info probes are orthogonal to
                # the section state machine (free-form key/value
                # lines that can appear anywhere); try them first
                # because they're frequent in a SIESTA preamble.
                if _scan_runtime_info(line, line_no):
                    continue

                # Section dispatch: first rule whose matcher fires
                # wins.  Order in the ``rules`` list is significant
                # (see comment block above the list).  ``find_match``
                # uses the combined-regex pre-filter then per-rule
                # iteration in registration order; predicate-only
                # rules are invoked individually with § 6 error
                # isolation.
                rule = compiled.find_match(line)
                if rule is not None:
                    if rule.on_start is not None:
                        rule.on_start(line, line_no)
                    if rule.consume is not None:
                        active = rule

        # End-of-file: drop torn frames, then flush.  The SIESTA stream
        # is "SCF -> outcoor -> SCF -> outcoor -> ...", so a torn
        # outcoor at EOF means the current_scf belongs to a step we
        # can't materialize -- drop it with the frame.
        if active is not None and active.name == "coords":
            step_frame = None

        # EOF in-progress detection: when the file ends with a real
        # structure echo AND SCF cycles but NO canonical step-end
        # signal (no ``siesta: E_KS(eV) = ...`` for this step, no
        # forces emitted yet), the step is mid-flight.  SIESTA 5.4.2
        # writes the input-coordinate echo as an ``outcoor:`` block
        # BEFORE the first SCF, so a freshly-started run has
        # step_frame populated AND current_scf populated AND
        # step_energy=None -- which is the exact in-progress
        # signature.  Without this flag, the EOF-flushed frame
        # would surface in the trajectory slider as a "real" geom
        # step 0 during the first SCF cycle, leaking SCF-in-progress
        # state into the completed-frame UI.
        eof_in_progress = (
            run_state == "running"
            and bool(step_frame)
            and bool(current_scf)
            and step_energy is None
        )
        commit(in_progress=eof_in_progress)

        # In-progress SCF visibility: when the run is still active
        # (no End-of-run / fatal marker) and we have buffered SCF
        # cycles but no outcoor block to attach them to yet, emit a
        # synthetic Frame so the Results-tab inspector can render the
        # SCF convergence chart in real time.  Without this the
        # inspector appears stalled for 5-30 min during the first
        # SCF cycle of a heavy run (a 200-atom Au-thiol-Au junction
        # was the motivating case) -- and if the SCF is silently
        # diverging the user only finds out an hour later instead of
        # immediately.
        #
        # The synthetic frame is FLAGGED as in_progress so consumers
        # know to hide trajectory animation controls and show only
        # the SCF chart + a "calculation in progress" banner.
        if run_state == "running" and current_scf:
            # Geometry placeholder: use the most recent COMMITTED
            # frame's structure when available (correct for "SCF for
            # next geom step", since SIESTA holds geometry constant
            # within an SCF cycle).  When no committed frame exists
            # yet (first-SCF case), use a 1-atom "X" placeholder --
            # the inspector hides the viewer based on in_progress=True
            # so the placeholder geometry is never actually rendered.
            if frames:
                placeholder_struct = frames[-1].structure
            else:
                placeholder_struct = Structure(
                    elements=["X"],
                    positions=np.zeros((1, 3), dtype=float),
                )
            # Energy: most-recent FINITE SCF-cycle energy (E_KS first,
            # then Eharris, then FreeEng -- same hierarchy commit()
            # uses), falling back to the preamble Etot if no cycle has
            # a finite energy yet.  Same NaN-skipping logic as the
            # commit() fallback path.
            # SCF cycle dicts use the canonical key ``energy`` (mapped
            # from E_KS at parse time -- see _SCF_FIELD_MAP); walk back
            # to the most recent FINITE value.
            in_prog_energy: Optional[float] = None
            # A device mid-NEGF speaks for its NEGF phase, as commit() rules.
            _live = ([c for c in current_scf
                      if c.get("phase") == _G.PHASE_NEGF] or current_scf)
            for cycle in reversed(_live):
                v = cycle.get("energy")
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    in_prog_energy = fv
                    break
            # PR 4 (results-state-contract § 6): no
            # ``step_initial_etot`` fallback.  Pre-PR-4 the
            # preamble-Etot leaked into the in-progress frame's
            # ``energy`` field and the JS energy plot showed a
            # placeholder point that disappeared on full refresh
            # (the "odd value" bug class users reported 2026-06-17).
            # When the SCF hasn't reported a finite cycle yet,
            # ``in_prog_energy`` stays ``None`` -> JSON null -> the
            # JS plottableFrames filter omits the point.  The
            # preamble Etot is preserved separately in
            # ``runtime_info["initial_etot"]`` for display.
            # Same time surfacing as for committed frames: last SCF
            # cycle's ``elapsed_s`` = how far into the run SIESTA was
            # when it last ticked the IterSCF timer.  ``wall_clock_s``
            # stays None here for the same reason as above.
            ip_elapsed_s: Optional[float] = None
            if current_scf:
                cum = current_scf[-1].get("elapsed_s")
                if isinstance(cum, (int, float)) and math.isfinite(cum):
                    ip_elapsed_s = float(cum)
            frames.append(Frame(
                structure   = placeholder_struct,
                step_index  = len(frames),
                energy      = in_prog_energy,
                scf_history = list(current_scf),
                elapsed_s   = ip_elapsed_s,
                in_progress = True,
            ))

        # NO CONVERGENCE CLAUSE HERE -- `model/parse.md` § 2b, P-S2.
        # This read `if run_state == "ongoing" and last_scf_converged is
        # False: run_state = "error"`, which let the SCIENCE decide how the
        # PROCESS ended.  Not converging is normal and often deliberate,
        # and a run that stops without its end marker is `stopped` because
        # it stopped -- which the `running` default plus the DirParser's
        # file-age check already establish, without asking the physics.
        #
        # The held `SCF_NOT_CONV:` line is still the best cause-of-death
        # sentence when something else proves death, so it fills an empty
        # `error_message` in that case only.
        if run_state in ("stopped", "out_of_memory") and error_message is None:
            error_message = scf_not_conv_line

        # Surface frozen-atom indices to the consumer.  The
        # trajectory inspector uses this for the "Hide frozen atoms"
        # overlay + filters force arrows to free atoms only.
        # SIESTA's ``.out`` reports only the AGGREGATE "Max …
        # constrained" value, not which atoms are constrained; we
        # have to recover the indices from another source.
        #
        # ONE PRECEDENCE, ONE HOME.  The .out echo -> sidecar -> .fdf order
        # (2026-06-14 contract fix; this comment used to enumerate it, and
        # was already stale at two paths where the code tried three) is the
        # function's, in the module that owns all three sources.  `xv2xyz
        # --from-run` needs the same order for a `.XV`, and a second copy of
        # a precedence is how two answers start disagreeing.
        #
        # Every source returns empty on any failure — frozen-atom data is
        # optional UI metadata and must not break trajectory loading.
        from ._sidecar import read_frozen_atoms_for_siesta
        frozen_set = read_frozen_atoms_for_siesta(path)
        if frozen_set:
            runtime_info["frozen_atoms"] = sorted(frozen_set)

        # PR 4 (results-state-contract § 6): preserve the preamble
        # ``Etot`` as ``runtime_info["initial_etot"]`` for display.
        # Pre-PR-4 this value leaked into per-frame ``energy`` fields
        # via the now-deleted fallback paths.  The contract: it stays
        # informational (where the SCF started from), not a frame
        # energy.  ``step_initial_etot`` holds the most-recent value
        # seen during parse; for finished runs that's the LAST step's
        # initial Etot, for ongoing runs the IN-FLIGHT step's.
        if (step_initial_etot is not None
                and math.isfinite(step_initial_etot)):
            runtime_info["initial_etot"] = float(step_initial_etot)

        # THE PHASES, SUMMED (`model/parse.md` § 5d.6): how many cycles each
        # ran and whether it converged -- a device's verdict is its NEGF
        # phase's, and `scf_converged` above already answers for the LAST
        # phase that ran.
        _counts: Dict[str, int] = {}
        for _fr in frames:
            for _c in (_fr.scf_history or []):
                _ph = _c.get("phase") or _G.PHASE_PERIODIC
                _counts[_ph] = _counts.get(_ph, 0) + 1
        if _counts:
            runtime_info["scf_phases"] = {
                _ph: {"cycles": _n, "converged": phase_converged.get(_ph)}
                for _ph, _n in _counts.items()}
        # Only a run that RAN TranSIESTA gets its facts: every SIESTA run
        # prints the small ``ts:`` block of HS-save flags inside the same star
        # frame, and a relaxation's record would otherwise carry a
        # "transiesta" section about nothing (found by the golden audit).
        if ts_info and (_counts.get(_G.PHASE_NEGF)
                        or "electrodes" in ts_info
                        or "charge_at_switch" in ts_info):
            runtime_info["transiesta"] = ts_info

        _scan_log.info(
            f"parsed {len(frames)} frames, run_state={run_state}, "
            f"{len(parse_warnings)} warnings")
        if error_message:
            _scan_log.error(error_message)
        return Trajectory(
            source_format  = cls.name,
            frames         = frames,
            lattice        = lattice,
            run_state      = run_state,
            # P-S2: reported, never a verdict.  `None` when no SCF block
            # was seen at all -- a correct final answer, not a hole.
            scf_converged  = last_scf_converged,
            error_message  = error_message,
            runtime_info   = runtime_info,
            parse_warnings = parse_warnings,
        )


class SiestaOutFileParser(FileParser):
    """Parse a SIESTA stdout-redirected output file (``.out`` /
    ``.log``).  Returns a :class:`TrajectoryResult` with one Frame
    per geometry step + per-step SCF history."""

    name   = "siesta"
    label  = SiestaParser.label
    hint   = SiestaParser.hint

    @classmethod
    def footgun_hint_for(cls, filename: str):
        lower = filename.lower()
        if not lower.endswith(".fdf"):
            return None
        stem = filename[:-len(".fdf")]
        return (
            f"{filename} is the SIESTA INPUT file, not its output. "
            f"Point molbuilder at the .out file SIESTA wrote "
            f"(typically {stem}.out / siesta.out / <label>.out), "
            f"or at the unified {stem}.molwatch.log if the run "
            f"was generated through molbuilder."
        )
    output = TrajectoryResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return SiestaParser.can_parse(str(path))

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        traj = SiestaParser.parse(str(path))
        # THE .out STAYS THE TRAJECTORY; the netCDF only sharpens it.
        #
        # SIESTA writes <label>.MD.nc beside the .out whenever
        # WriteMDhistory is on and the binary was built with -DCDF (both
        # true for the packaged env).  It holds the same geometries and
        # energies WITHOUT having gone through a Fortran text formatter --
        # full double precision instead of the .out's four decimals, and
        # none of the fixed-width column collisions this module carries a
        # regex and a structural slicer to survive.
        #
        # Everything the netCDF does NOT have -- run state, errors, forces,
        # per-SCF-cycle history -- still comes from the text above, because
        # SIESTA writes no structured equivalent for any of it.  So this is
        # an upgrade of two fields on frames that already exist, never a
        # second source of truth: `upgrade_frames` returns the same frames
        # in the same order when there is no sibling, when it is
        # unreadable, or when nothing matches.
        from .siesta_mdnc import upgrade_frames
        traj.frames, _mdnc_info = upgrade_frames(traj.frames, Path(path))
        if _mdnc_info:
            traj.runtime_info = dict(traj.runtime_info or {})
            traj.runtime_info.update(_mdnc_info)
        return wrap_trajectory(traj, cls.name, path)
