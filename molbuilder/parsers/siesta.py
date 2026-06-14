"""SIESTA .out / .log parser.

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
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..frame import Frame, ParseWarning, Trajectory
from ..structure import Structure
from ._rules import (
    CONTINUE, END_BUBBLE, END_SECTION,
    SectionRule, compile_rules, contains_ci, matches_regex_ci,
    starts_with_ci,
)
from .base import TrajectoryParser


# Runtime info detection (cross-cutting -- same display path as
# molwatch's runtime header):
#   * "* Running on  N nodes in parallel."  -> n_mpi_processes
#   * "Running on host: <name>"             -> hostname  (some SIESTA builds)
#   * Echoed .fdf comments "# runtime.<k>: <v>" -> all the user-set caps
_SIESTA_NODES_RE   = re.compile(
    r"^\s*\*\s*Running on\s+(\d+)\s+nodes? in parallel", re.IGNORECASE)
_SIESTA_HOST_RE    = re.compile(
    r"^\s*Running on host:\s*(\S+)", re.IGNORECASE)
# Matches the same shape as the molwatch runtime header (so /spectra
# script writers + Build SIESTA writers can use IDENTICAL line
# format; cf. molbuilder.runtime_info docstring).
_SIESTA_RUNTIME_RE = re.compile(
    r"^#\s*runtime\.([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$")

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

# Lightweight prefix match for any SIESTA SCF iteration line.  We
# capture iscf + the rest of the line as a single string; the actual
# float columns are parsed separately by ``_parse_scf_floats`` below.
# This split lets us handle BOTH the closed-shell form (7 columns
# total) AND the spin-polarized form (8 columns -- Ef split into
# Ef_up + Ef_dn) without writing a brittle multi-regex dispatch.
_SCF_PREFIX_RE = re.compile(r"^\s*scf:\s*(\d+)\s+(.+)$", re.IGNORECASE)

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
# We detect SIESTA's f10.6-style 6-decimal field signature
# (``.NNNNNN`` followed immediately by either a digit OR an asterisk;
# both can only be the start of the next column) and insert a space.
# This is a FORMATTING quirk on the SIESTA side; we recover the
# columns without altering any values.
_SCF_TIGHT_PACK_RE = re.compile(r"(\.\d{6})(?=[\d*])")

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


def _parse_scf_floats(rest: str) -> Optional[List[float]]:
    """Tokenize the post-``scf: <iscf>`` part of an SCF line into
    floats.  Returns the list, or None if any token can't be parsed.

    Handles two SIESTA-side formatting quirks:

      * **Tight-packed columns** -- no whitespace between adjacent
        f10.6 fields when both values fill their widths.  Resolved
        by ``_SCF_TIGHT_PACK_RE`` which inserts a separator at the
        6-decimal signature.
      * **Fortran field overflow** -- a value too large for its field
        prints as ``**********``.  The cycle is most likely diverging;
        the asterisk token is parsed as NaN via
        ``_parse_fortran_float`` so the row's other columns (early
        ones with healthy magnitudes) survive and the divergence
        diagnostic stays visible in the Results tab.

    Returns ``None`` only on a genuinely unparseable token (neither
    a real float nor a Fortran-overflow indicator) -- those are real
    parser bugs we want surfaced as a ParseWarning, not silently
    coerced.
    """
    fixed = _SCF_TIGHT_PACK_RE.sub(r"\1 ", rest)
    try:
        return [_parse_fortran_float(t) for t in fixed.split()]
    except ValueError:
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
_SCF_HEADER_RE = re.compile(r"^\s*iscf\s+\S", re.IGNORECASE)


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
    "ef":       None,    # closed-shell Fermi level
    "ef_up":    None,    # collinear spin-polarized (up)
    "ef_dn":    None,    # collinear spin-polarized (down)
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


class SiestaParser(TrajectoryParser):
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
        "executable      : siesta",     # v5.x line 1
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
        frames: List[Frame] = []
        lattice: Optional[List[List[float]]] = None
        pending_lattice: Optional[List[List[float]]] = None
        # Run-state detection.  Three outcomes, in priority order:
        #   "error"    -- a fatal marker matched, OR the run ended
        #                 without ">> End of run" AND the last SCF
        #                 block did not converge.
        #   "finished" -- ">> End of run" emitted AND no fatal error.
        #   "ongoing"  -- no clean-exit marker, no detected fault.
        # SIESTA's clean-exit marker is "always written" only on
        # success; abort emits at least one of the fatal markers we
        # recognise below (per the 2026-05-29 user directive: detect
        # convergence / exit failures from the .out itself so the
        # /results badge can show "Error" without depending on the
        # wrapper's grep).
        run_state: str = "ongoing"
        error_message: Optional[str] = None
        # Per-SCF-block convergence flag.  None = never saw an SCF
        # block; True = last block converged; False = last block hit
        # "SCF did NOT converge" / "SCF_NOT_CONV".  Used at EOF to
        # decide whether a torn run (no End-of-run marker) was a
        # silent abort vs an explicit non-convergence error.
        last_scf_converged: Optional[bool] = None
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

        # SCF iteration history accumulator for the current step.  Each
        # entry is a per-cycle dict matching the schema in
        # docs/types/parsers.md.  SIESTA's column set differs from
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

        def commit() -> None:
            nonlocal step_frame, step_energy, step_max_force, step_forces
            nonlocal step_max_force_constrained
            nonlocal current_scf, prev_E_KS, step_initial_etot
            if not step_frame:
                step_frame = None
                step_energy = None
                step_max_force = None
                step_max_force_constrained = None
                step_forces = []
                step_initial_etot = None
                # Don't reset current_scf here -- a torn frame at EOF
                # has no Frame to attach to, but otherwise current_scf
                # may legitimately belong to a NOT-YET-committed frame
                # (SCF runs *before* outcoor in SIESTA's stream, so
                # commit() bailing on an empty step_frame at the start
                # of a run is normal).
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
            if (frame_energy is None
                    and step_initial_etot is not None
                    and math.isfinite(step_initial_etot)):
                frame_energy = step_initial_etot
            frames.append(Frame(
                structure   = struct,
                step_index  = len(frames),
                energy      = frame_energy,
                forces      = forces_arr,
                max_force   = step_max_force,
                max_force_constrained = step_max_force_constrained,
                scf_history = list(current_scf) if current_scf else None,
            ))
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
        # Run-state marker.  Always fires in scan; single-line.
        # "Finished" takes precedence over "ongoing" but NOT over
        # "error" -- if a fatal marker already set run_state="error",
        # a later End-of-run does not paper over it.  In practice
        # SIESTA either crashes (no End-of-run) or finishes cleanly
        # (no fatal marker), but the priority rule is the defensible
        # default if both somehow appear.
        def _on_end_of_run(line: str, line_no: int) -> None:
            nonlocal run_state
            if run_state != "error":
                run_state = "finished"

        # Fatal error markers.  Each ``contains_ci`` substring is
        # the canonical SIESTA-emitted phrase that always indicates
        # a non-recoverable failure.  The list mirrors the wrapper's
        # grep heuristic (runwrap.py) so .out -> badge tracking is
        # consistent across the live wrapper + the post-mortem
        # parser path.  Set 2026-05-29 per user directive.
        def _on_fatal_error(line: str, line_no: int) -> None:
            nonlocal run_state, error_message
            run_state = "error"
            # Preserve the FIRST fatal marker -- subsequent crashes
            # usually cascade from the original cause.
            if error_message is None:
                error_message = line.strip()[:200]

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

        def _on_scf_fatal_not_converged(line: str, line_no: int) -> None:
            """Handler for the FATAL SCF_NOT_CONV: form.  Sets error +
            captures the informative root-cause line as
            error_message (first-wins; cascade markers like
            ABNORMAL_TERMINATION and Stopping Program that follow
            do NOT overwrite it)."""
            nonlocal run_state, error_message, last_scf_converged
            run_state = "error"
            if error_message is None:
                error_message = line.strip()[:200]
            last_scf_converged = False

        def _on_scf_not_converged(line: str, line_no: int) -> None:
            """Soft handler for the informational form.  Flags only;
            strict EOF check decides whether the run errored."""
            nonlocal last_scf_converged
            last_scf_converged = False

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
            nonlocal current_scf, prev_E_KS
            m_scf_prefix = _SCF_PREFIX_RE.match(line)
            if not m_scf_prefix:
                return
            iscf = int(m_scf_prefix.group(1))
            rest = m_scf_prefix.group(2)
            vals = _parse_scf_floats(rest)
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

            # iscf==1 starts a new SCF run.  See parse() docstring for
            # the failed-SCF-restart edge case.
            if iscf == 1:
                if current_scf:
                    current_scf = []
                prev_E_KS = None
            delta_E = ((e_ks - prev_E_KS)
                       if prev_E_KS is not None else 0.0)
            cycle_dict["delta_E"] = delta_E
            current_scf.append(cycle_dict)
            prev_E_KS = e_ks

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

        rules: List[SectionRule] = [
            # Fatal-error markers fire FIRST so they win over any
            # section that might otherwise eat the line.  Substring
            # match (the markers can appear mid-line, e.g. preceded
            # by "node 0: " in MPI mode).
            SectionRule(
                name="fatal_siesta_error",
                aliases=["siesta: ERROR"],
                start=contains_ci("siesta: error"),
                on_start=_on_fatal_error,
            ),
            SectionRule(
                name="fatal_propor_error",
                aliases=["propor: ERROR"],
                start=contains_ci("propor: error"),
                on_start=_on_fatal_error,
            ),
            SectionRule(
                name="fatal_stopping_program",
                aliases=["Stopping Program from Node"],
                start=contains_ci("stopping program from node"),
                on_start=_on_fatal_error,
            ),
            SectionRule(
                name="fatal_siesta_died",
                aliases=["siesta died"],
                start=contains_ci("siesta died"),
                on_start=_on_fatal_error,
            ),
            # ABNORMAL_TERMINATION: SIESTA emits this on every
            # cause-of-abort (SCF_NOT_CONV with MustConverge, propor
            # crash, lapack failure, ...).  Always followed by per-
            # rank Stopping Program lines.  Confirmed in real-world
            # output 2026-05-30 (projects/hemeC-dithiol stage3 run).
            SectionRule(
                name="fatal_abnormal_termination",
                aliases=["ABNORMAL_TERMINATION"],
                start=contains_ci("abnormal_termination"),
                on_start=_on_fatal_error,
            ),
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
                start=contains_ci("scf_not_conv"),
                on_start=_on_scf_fatal_not_converged,
            ),
            # SCF convergence success.  "by <criterion>" suffix
            # guards against matching a diagnostic line that mentions
            # both "SCF Convergence" and "did NOT converge".
            SectionRule(
                name="scf_converged",
                aliases=["SCF Convergence by ..."],
                start=contains_ci("scf convergence by"),
                on_start=_on_scf_converged,
            ),
            # SCF non-convergence: SOFT informational form (no
            # SCF_NOT_CONV: prefix).  Can appear during a relax that
            # recovers in a later step; strict EOF check decides
            # error vs ongoing.
            SectionRule(
                name="scf_not_converged",
                aliases=["SCF did NOT converge"],
                start=contains_ci("scf did not converge"),
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
                start=starts_with_ci(">> End of run"),
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
                # Substring: the marker appears in the
                # ``Program's energy decomposition`` block
                # after the initial DM is built but before
                # scf:1.  See _on_initial_etot docstring.
                start=contains_ci("siesta: etot"),
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
                start=matches_regex_ci(r"^\s*iscf\s+\S"),
                on_start=_on_scf_header,
            ),
            SectionRule(
                name="scf_data",
                aliases=["scf: <iscf> ..."],
                # Same shape as _SCF_PREFIX_RE without the trailing
                # capture (the on_start callback does its own
                # ``_SCF_PREFIX_RE.match(line)`` to extract groups).
                # Combined-regex eligible.
                start=matches_regex_ci(r"^\s*scf:\s*\d+\s+"),
                on_start=_on_scf_data,
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

        def _scan_runtime_info(line: str) -> bool:
            """Orthogonal runtime-info regex probes.  Not section
            boundaries -- just free-form key/value lines that may
            appear anywhere in scan state.  Returns True if the line
            was consumed (caller should skip rule dispatch)."""
            m = _SIESTA_NODES_RE.match(line)
            if m:
                try:
                    runtime_info["n_mpi_processes"] = int(m.group(1))
                except ValueError:
                    pass
                return True
            m = _SIESTA_HOST_RE.match(line)
            if m:
                runtime_info["hostname"] = m.group(1).strip()
                return True
            m = _SIESTA_RUNTIME_RE.match(line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if val == "None":
                    runtime_info[key] = None
                elif val in ("True", "False"):
                    runtime_info[key] = (val == "True")
                else:
                    try:
                        runtime_info[key] = int(val)
                    except ValueError:
                        runtime_info[key] = val
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
                if _scan_runtime_info(line):
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
        commit()

        # Post-process: if we never saw "End of run" AND the last SCF
        # block failed to converge, this is a non-convergence error
        # even without an explicit "siesta: ERROR" line (some SIESTA
        # builds simply truncate on SCF_NOT_CONV without an error
        # marker).  Per 2026-05-29 user directive: strict policy --
        # final SCF must converge OR run must reach End-of-run, else
        # it's an error.  A fatal-marker run keeps its existing
        # error_message; the SCF case fills in a sensible default.
        if run_state == "ongoing" and last_scf_converged is False:
            run_state = "error"
            error_message = (
                "SCF did not converge in the final step "
                "(run truncated without '>> End of run' marker)"
            )

        # Surface frozen-atom indices to the consumer.  The
        # trajectory inspector uses this for the "Hide frozen atoms"
        # overlay + filters force arrows to free atoms only.
        # SIESTA's ``.out`` reports only the AGGREGATE "Max …
        # constrained" value, not which atoms are constrained; we
        # have to recover the indices from another source.  Two
        # paths are tried in order:
        #
        #   1. ``.molstruct.json`` sidecar — the canonical source
        #      when present (writes shared with the modify-tab's
        #      sidecar-aware save path).
        #   2. Sibling ``.fdf`` input's ``%block Geometry.
        #      Constraints`` — fallback for runs where no sidecar
        #      was written but the input's constraint block carries
        #      the indices we need.  2026-06-14 addition: the chart
        #      was showing the constrained trace correctly while
        #      the "Hide frozen atoms" toggle stayed hidden because
        #      step 1 was empty for these runs (see
        #      docs/protocols/code-audit.md § 3.3, cross-source-of-
        #      truth gap).
        #
        # Both helpers return empty on any failure — frozen-atom
        # data is optional UI metadata.
        from ._sidecar import (
            read_frozen_atoms,
            read_frozen_atoms_from_siesta_fdf,
            read_frozen_atoms_from_siesta_out,
        )
        # 2026-06-14 contract fix: read constraints from the .out's
        # OWN ``siesta: Constraints applied in the following order:``
        # echo as the primary source.  No filename heuristic needed
        # because the data lives in the same file we're parsing.
        # Fall back to sidecar -> paired .fdf only when the .out
        # echo is absent.  Order matters: the .out is authoritative
        # (SIESTA ran with whatever it parsed), the sidecar is
        # user-edited, the .fdf is what was on disk at some point
        # before the run.
        frozen_set = read_frozen_atoms_from_siesta_out(path)
        if not frozen_set:
            frozen_set = read_frozen_atoms(path)
        if not frozen_set:
            frozen_set = read_frozen_atoms_from_siesta_fdf(path)
        if frozen_set:
            runtime_info["frozen_atoms"] = sorted(frozen_set)

        return Trajectory(
            source_format  = cls.name,
            frames         = frames,
            lattice        = lattice,
            run_state      = run_state,
            error_message  = error_message,
            runtime_info   = runtime_info,
            parse_warnings = parse_warnings,
        )
