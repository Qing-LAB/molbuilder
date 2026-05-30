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

import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..frame import Frame, ParseWarning, Trajectory
from ..structure import Structure
from ._rules import (
    CONTINUE, END_BUBBLE, END_SECTION,
    SectionRule, contains_ci, starts_with_ci,
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

# Lightweight prefix match for any SIESTA SCF iteration line.  We
# capture iscf + the rest of the line as a single string; the actual
# float columns are parsed separately by ``_parse_scf_floats`` below.
# This split lets us handle BOTH the closed-shell form (7 columns
# total) AND the spin-polarized form (8 columns -- Ef split into
# Ef_up + Ef_dn) without writing a brittle multi-regex dispatch.
_SCF_PREFIX_RE = re.compile(r"^\s*scf:\s*(\d+)\s+(.+)$", re.IGNORECASE)

# Defensive separator-inserter for SIESTA's fixed-width SCF columns.
# When two adjacent columns pack so tight that no whitespace separates
# them (the dHmax-and-Ef_dn case in spin-polarized output where both
# values fill their fields), the naive ``split()`` captures them as
# one token like ``-1.929956131.029438``.  We detect SIESTA's f10.6-
# style 6-decimal field signature (``.NNNNNN`` immediately followed by
# a digit, which can only be the start of the next column) and insert
# a separator.  This is a FORMATTING quirk, NOT value overflow; the
# values themselves are fine, the issue is the field width.  Re-using
# the helper from the early 2026-05-28 patch with a more honest name.
_SCF_TIGHT_PACK_RE = re.compile(r"(\.\d{6})(?=\d)")


def _parse_scf_floats(rest: str) -> Optional[List[float]]:
    """Tokenize the post-``scf: <iscf>`` part of an SCF line into
    floats.  Returns the list, or None if any token can't be parsed.

    Handles SIESTA's tight-packed fixed-width fields (no whitespace
    between adjacent columns when both values fill their widths)
    by inserting a separator at the f10.6 6-decimal signature.
    """
    fixed = _SCF_TIGHT_PACK_RE.sub(r"\1 ", rest)
    try:
        return [float(t) for t in fixed.split()]
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
        step_forces: List[List[float]] = []

        def commit() -> None:
            nonlocal step_frame, step_energy, step_max_force, step_forces
            nonlocal current_scf, prev_E_KS
            if not step_frame:
                step_frame = None
                step_energy = None
                step_max_force = None
                step_forces = []
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
            frames.append(Frame(
                structure   = struct,
                step_index  = len(frames),
                energy      = step_energy,
                forces      = forces_arr,
                max_force   = step_max_force,
                scf_history = list(current_scf) if current_scf else None,
            ))
            step_frame = None
            step_energy = None
            step_max_force = None
            step_forces = []
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

        # SCF-block convergence flags.  SIESTA emits a header line
        # like ``SCF Convergence by DM+H criterion`` on success, and
        # either ``SCF did NOT converge`` (some versions) or
        # ``SCF_NOT_CONV`` (constant from internal sources) on
        # failure.  We track the LAST-seen status; at EOF, if the
        # run didn't reach End-of-run AND the last SCF block didn't
        # converge, we flip run_state to error.
        def _on_scf_converged(line: str, line_no: int) -> None:
            nonlocal last_scf_converged
            last_scf_converged = True

        def _on_scf_not_converged(line: str, line_no: int) -> None:
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
                x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
            except ValueError:
                # Same: the line that ends a torn outcoor block may
                # be ``>> End of run`` -- re-feed so that rule fires.
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
                row = [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError:
                # Same: a non-vector line ending the cell block may
                # be the next section header.
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
                fx = float(parts[1]); fy = float(parts[2]); fz = float(parts[3])
            except ValueError:
                return END_BUBBLE
            step_forces.append([fx, fy, fz])
            return CONTINUE

        # E_KS energy line: single-line.  Format:
        #   ``siesta: E_KS(eV) =       -1234.567``
        def _on_e_ks(line: str, line_no: int) -> None:
            nonlocal step_energy
            try:
                step_energy = float(line.split("=", 1)[1].split()[0])
            except (ValueError, IndexError) as exc:
                _warn(line_no, line,
                      f"E_KS line: malformed value: {exc}",
                      category="energy")

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

        # Max-force line: single-line.  Gated by a closure-captured
        # check on ``step_forces`` -- only valid after a Forces
        # section closed.  Without the gate a stray "Max <num>" line
        # in the preamble would mis-attribute to the first frame.
        def _max_force_match(line: str) -> bool:
            if not step_forces:
                return False
            parts = line.strip().split()
            # parts[0] case-insensitive: a hypothetical SIESTA build
            # that emits "MAX" / "max" still hits the rule.  The
            # gating on len(parts) == 2 keeps the constrained
            # duplicate (3 tokens) from misfiring.
            return (len(parts) == 2 and parts[0].lower() == "max")

        def _on_max_force(line: str, line_no: int) -> None:
            nonlocal step_max_force
            try:
                step_max_force = float(line.strip().split()[1])
            except (ValueError, IndexError) as exc:
                _warn(line_no, line,
                      f"Max-force line: malformed value: {exc}",
                      category="forces")

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
            # SCF convergence flags -- track only, don't terminate.
            # SUCCESS marker: SIESTA always writes "SCF Convergence
            # by <criterion>" -- the "by" keyword guards against
            # accidentally matching a diagnostic line that mentions
            # both "SCF Convergence" and "did NOT converge".  Loose
            # enough to handle every criterion variant SIESTA emits
            # (DM+H, DM alone, H alone, ...).
            SectionRule(
                name="scf_converged",
                aliases=["SCF Convergence by ..."],
                start=contains_ci("scf convergence by"),
                on_start=_on_scf_converged,
            ),
            # NON-convergence: two known SIESTA phrasings.
            SectionRule(
                name="scf_not_converged",
                aliases=["SCF did NOT converge", "SCF_NOT_CONV"],
                start=contains_ci("scf did not converge"),
                on_start=_on_scf_not_converged,
            ),
            SectionRule(
                name="scf_not_converged_const",
                aliases=["SCF_NOT_CONV"],
                start=contains_ci("scf_not_conv"),
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
                # (case-insensitive).  _SCF_HEADER_RE already encodes
                # this; we replicate the matcher in rule shape.
                start=lambda line: bool(_SCF_HEADER_RE.match(line)),
                on_start=_on_scf_header,
            ),
            SectionRule(
                name="scf_data",
                aliases=["scf: <iscf> ..."],
                start=lambda line: bool(_SCF_PREFIX_RE.match(line)),
                on_start=_on_scf_data,
            ),
            SectionRule(
                name="max_force",
                aliases=["Max <value>"],
                start=_max_force_match,
                on_start=_on_max_force,
            ),
        ]

        # ---- State-machine driver --------------------------------
        # state: either "scan" (try all rules) or the name of an
        # active multi-line rule (only that rule's ``consume`` runs).
        active: Optional[SectionRule] = None

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
                # (see comment block above the list).
                for rule in rules:
                    if rule.start(line):
                        if rule.on_start is not None:
                            rule.on_start(line, line_no)
                        if rule.consume is not None:
                            active = rule
                        break

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

        return Trajectory(
            source_format  = cls.name,
            frames         = frames,
            lattice        = lattice,
            run_state      = run_state,
            error_message  = error_message,
            runtime_info   = runtime_info,
            parse_warnings = parse_warnings,
        )
