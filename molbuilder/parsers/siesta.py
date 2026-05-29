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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..frame import Frame, ParseWarning, Trajectory
from ..structure import Structure
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
_SCF_PREFIX_RE = re.compile(r"^\s*scf:\s*(\d+)\s+(.+)$")

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
    _STRONG_MARKERS = (
        "Executable      : siesta",      # v5.x line 1
        "WELCOME TO SIESTA",             # v5.x banner (uppercase)
        "Welcome to SIESTA",             # v4.x banner (mixed case)
        "siesta: System type",
        "siesta: Atomic forces",
        "outcoor: Atomic coordinates",
        "outcell: Unit cell vectors",
        "Begin CG opt",
        "Begin MD opt",
        "Begin Broyden opt",
        "Begin FIRE opt",
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
        head = "".join(head_lines)
        # 1. Any strong content marker wins immediately.
        if any(m in head for m in cls._STRONG_MARKERS):
            return True
        # 2. Otherwise, count `siesta:` / `redata:` lines.
        prefix_hits = sum(
            1 for ln in head_lines
            if any(ln.lstrip().startswith(p) for p in cls._PREFIX_MARKERS)
        )
        return prefix_hits >= cls._PREFIX_THRESHOLD

    @classmethod
    def parse(cls, path: str) -> Trajectory:
        frames: List[Frame] = []
        lattice: Optional[List[List[float]]] = None
        pending_lattice: Optional[List[List[float]]] = None
        # Run-state detection: SIESTA writes ">> End of run:  <date>"
        # at end of a successful run; any other shutdown leaves no
        # such marker.  No equivalent for "error" state on SIESTA's
        # native output -- common abort patterns vary across versions
        # -- so we only differentiate finished vs ongoing here.
        run_state: str = "ongoing"
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

        state = "scan"  # "scan", "in_coords", "in_cell", "in_forces"

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

        with open(path, "r", errors="replace") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.rstrip("\n")
                stripped = line.strip()

                # SIESTA's clean-exit marker: ">> End of run: <date>".
                # Always written at the very end of a successful run;
                # absent when SIESTA aborts.  Detect anywhere outside
                # a coords block (it's a free-form line, not nested).
                if stripped.startswith(">> End of run"):
                    run_state = "finished"
                    continue

                # Runtime info: cheap regex probes outside the coords
                # blocks.  Three matchers, in order of frequency.
                # (Free-form lines; safe to test on every scan-state
                # line.)
                if state == "scan":
                    m = _SIESTA_NODES_RE.match(line)
                    if m:
                        try:
                            runtime_info["n_mpi_processes"] = int(m.group(1))
                        except ValueError:
                            pass
                        continue
                    m = _SIESTA_HOST_RE.match(line)
                    if m:
                        runtime_info["hostname"] = m.group(1).strip()
                        continue
                    m = _SIESTA_RUNTIME_RE.match(line)
                    if m:
                        key, val = m.group(1), m.group(2).strip()
                        # Coerce numeric / bool / None like the
                        # molwatch parser does so the inspector can
                        # rely on int / bool types where appropriate.
                        if val == "None":
                            runtime_info[key] = None
                        elif val in ("True", "False"):
                            runtime_info[key] = (val == "True")
                        else:
                            try:
                                runtime_info[key] = int(val)
                            except ValueError:
                                runtime_info[key] = val
                        continue

                if state == "in_coords":
                    if not stripped:
                        state = "scan"
                        continue
                    parts = stripped.split()
                    if len(parts) < 6:
                        state = "scan"
                        continue
                    try:
                        x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
                    except ValueError:
                        state = "scan"
                        continue
                    step_frame.append([parts[-1], x, y, z])
                    continue

                if state == "in_forces":
                    parts = stripped.split()
                    if len(parts) >= 4:
                        try:
                            int(parts[0])  # atom index
                            fx = float(parts[1]); fy = float(parts[2]); fz = float(parts[3])
                        except ValueError:
                            state = "scan"
                        else:
                            step_forces.append([fx, fy, fz])
                            continue
                    else:
                        state = "scan"

                if state == "in_cell":
                    parts = stripped.split()
                    if len(parts) >= 3:
                        try:
                            row = [float(parts[0]), float(parts[1]), float(parts[2])]
                        except ValueError:
                            state = "scan"
                        else:
                            pending_lattice.append(row)
                            if len(pending_lattice) >= 3:
                                lattice = pending_lattice
                                pending_lattice = None
                                state = "scan"
                            continue
                    else:
                        state = "scan"

                # ---- scan mode ----
                # SCF column header: SIESTA emits a line like
                #   ``iscf  Eharris(eV)  E_KS(eV)  ...  dHmax(eV)``
                # before the SCF block.  Parse it into canonical keys
                # so subsequent data rows map by NAME, not by position
                # (the layout differs between closed-shell and spin-
                # polarized runs, and may change again in future
                # SIESTA versions).  See _parse_scf_header docstring.
                if _SCF_HEADER_RE.match(line):
                    parsed_header = _parse_scf_header(line)
                    if parsed_header is not None:
                        scf_header = parsed_header
                        continue

                # SCF iteration line: collected into scf_history.  An
                # iscf = 1 starts a new SCF run (= new CG/MD step's
                # electronic problem).  Energy column is E_KS (already
                # eV), so no unit conversion needed -- contrast with
                # PySCF where Hartree -> eV happens at parse time.
                #
                # 2026-05-28: rewritten to be header-driven.  When we
                # have a parsed column header (always for v5; older
                # SIESTA may omit it), each value is mapped to a
                # canonical key by NAME -- no position assumptions.
                # When no header has been seen, we fall back to the
                # historically-known closed-shell (6 floats) /
                # spin-polarized (7 floats) layouts.  Either way
                # ``dHmax`` ends up in the per-cycle dict correctly.
                m_scf_prefix = _SCF_PREFIX_RE.match(line)
                if m_scf_prefix:
                    iscf = int(m_scf_prefix.group(1))
                    rest = m_scf_prefix.group(2)
                    vals = _parse_scf_floats(rest)
                    if vals is None:
                        _warn(line_no, line,
                              "SCF line: could not tokenize as floats")
                        continue

                    cycle_dict: Optional[Dict[str, Any]]
                    if scf_header is not None:
                        cycle_dict = _build_cycle_dict_from_header(
                            iscf, vals, scf_header)
                        if cycle_dict is None:
                            _warn(line_no, line,
                                  f"SCF row has {len(vals)} values "
                                  f"but header has {len(scf_header)-1} "
                                  f"columns ({scf_header})")
                            continue
                    else:
                        cycle_dict = _build_cycle_dict_positional(iscf, vals)
                        if cycle_dict is None:
                            _warn(line_no, line,
                                  f"SCF line has {len(vals)} floats "
                                  f"after iscf; expected 6 (closed-"
                                  f"shell) or 7 (spin-polarized), and "
                                  f"no column header was seen")
                            continue

                    e_ks = cycle_dict.get("energy")
                    if e_ks is None:
                        _warn(line_no, line,
                              "SCF row missing 'energy' (E_KS) -- "
                              "downstream plot can't render this cycle")
                        continue

                    # iscf==1 starts a new SCF run.  Normally the
                    # previous step's SCF was already attached to its
                    # Frame at commit() time (which happens between
                    # SCF runs, when the outcoor: block arrives), so
                    # current_scf is empty by the time we see iscf==1.
                    #
                    # SP-D: a failed-SCF restart can produce two SCF
                    # runs WITHOUT an intervening outcoor: line (e.g.
                    # SIESTA aborts the first SCF and the user's
                    # restart script kicks off another).  In that case
                    # current_scf still holds the previous (failed) run
                    # and would silently merge into this step's frame.
                    # Drop it: the failed run is informational at best,
                    # and we shouldn't attribute its cycles to the
                    # next geometry step.
                    if iscf == 1:
                        if current_scf:
                            current_scf = []
                        prev_E_KS = None
                    delta_E = ((e_ks - prev_E_KS)
                               if prev_E_KS is not None else 0.0)
                    cycle_dict["delta_E"] = delta_E
                    current_scf.append(cycle_dict)
                    prev_E_KS = e_ks
                    continue

                if stripped.startswith("outcoor:"):
                    commit()
                    step_frame = []
                    state = "in_coords"
                    continue

                if stripped.startswith("outcell: Unit cell vectors"):
                    pending_lattice = []
                    state = "in_cell"
                    continue

                if "siesta: E_KS(eV)" in line:
                    try:
                        step_energy = float(line.split("=", 1)[1].split()[0])
                    except (ValueError, IndexError) as exc:
                        _warn(line_no, line,
                              f"E_KS line: malformed value: {exc}",
                              category="energy")
                    continue

                if "siesta: Atomic forces" in line:
                    step_forces = []
                    state = "in_forces"
                    continue

                # Max force: the unconstrained value sits on a line of
                # the form "   Max    4.669483", emitted right after
                # the per-atom force block.  The duplicated line
                # ending with "constrained" has 3 tokens, so we filter
                # on token count.  Additionally we gate on a non-empty
                # `step_forces`, so a stray "Max <num>" line earlier
                # in the file (e.g. in a header / comment) can't be
                # mis-attributed to whatever step we're currently on.
                parts = stripped.split()
                if (parts and parts[0] == "Max" and len(parts) == 2
                        and step_forces):
                    try:
                        step_max_force = float(parts[1])
                    except ValueError as exc:
                        _warn(line_no, line,
                              f"Max-force line: malformed value: {exc}",
                              category="forces")
                    continue

        # End-of-file: drop torn frames, then flush.  The SIESTA stream
        # is "SCF -> outcoor -> SCF -> outcoor -> ...", so a torn
        # outcoor at EOF means the current_scf belongs to a step we
        # can't materialize -- drop it with the frame.
        if state == "in_coords":
            step_frame = None
        commit()

        return Trajectory(
            source_format  = cls.name,
            frames         = frames,
            lattice        = lattice,
            run_state      = run_state,
            runtime_info   = runtime_info,
            parse_warnings = parse_warnings,
        )
