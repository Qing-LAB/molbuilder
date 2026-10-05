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

import os
import re
from pathlib import Path
from typing import List, Set

import numpy as np


from molbuilder.frame import Frame, ParseWarning, Trajectory
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import TrajectoryResult
from molbuilder.structure import Structure

from ._helpers import wrap_trajectory
from . import siesta_grammar as _G
from .siesta_reader import read_output


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
# read by `molwatch_grammar.parse_runtime_line`, which owns it.

# Convergence-target echo lines from SIESTA's ``redata:`` preamble are the
# grammar's (``_G.TARGET_LINES``, read by ``_G.read_target_line``), captured
# into ``runtime_info["convergence_targets"]`` so the Results tab can render
# the threshold line + "current vs target" text without the user having to
# load the source .fdf next to the run.

# SIESTA header probes -- the binary's self-report at the top of the .out,
# captured into ``runtime_info['siesta_build']`` so the Results tab can show
# what SIESTA actually ran with, not what the user requested.  The patterns
# and their one reader are `siesta_grammar`'s (``read_build_line``): TBtrans
# prints the same header, and two copies of it would drift.

# The solver SIESTA ran: its `diag:` lines -- the ground truth for which
# solver path the run took, NOT what the user wrote in the .fdf (those can
# drift when SIESTA's parser normalises or rejects).  Used to populate
# ``runtime_info['siesta_diag']``.
# The solver lines are `siesta_grammar`'s (``read_diag_line``), which
# `bench/result.py` reads too.  Until 2026-09-26 this read the ``redata:``
# algorithm line and a GPU banner that SIESTA 5.4.2 never prints, so no real
# run had a solver on record.

# ---- the atoms the run held: the .out's own echo -------------------------
#
# THE .OUT STATES THEM, so the parser reads them as its own content
# (`model/parse.md` § 5.3) -- what the engine applied, in the file being
# parsed, so no name can point it at another run.  The sidecar beside the
# output and the deck in its folder were tried after it until 2026-10-04,
# from `_sidecar.py`, where this reader lived.

_SIESTA_CONSTRAINTS_HEADER_RE = re.compile(
    r"siesta:\s+Constraints\s+applied\s+in\s+the\s+following\s+order:",
    re.IGNORECASE,
)
_SIESTA_CONSTRAINT_LINE_RE = re.compile(
    r"^\s*siesta:\s+Constraint\s*\(\d+\)\s*:\s*pos\s*$",
    re.IGNORECASE,
)
_SIESTA_CONSTRAINT_RANGES_RE = re.compile(
    r"^\s*\[\s*(.+?)\s*\]\s*$"
)
_RANGE_PIECE_RE = re.compile(r"(\d+)\s*--\s*(\d+)")


def read_frozen_atoms_from_siesta_out(out_path: str) -> Set[int]:
    """Return 0-based frozen-atom indices from the .out's own echo of
    them -- ``siesta: Constraint (N): pos`` and the ranges below it, under a
    ``siesta: Constraints applied in the following order:`` header or, as
    SIESTA 5.4.2 prints it, under none (measured on both recorded H2 runs;
    the header is the earlier v5 output this reader was written against).

    AUTHORITATIVE source of truth for SIESTA constraints — the data
    lives in the same file the Results-tab UI reads, so there's no
    filename-pairing heuristic between the .out and a sibling .fdf.

    Streams the file line-by-line and stops at the first non-
    constraints line after the section, so for the typical case
    (constraints near the top of the .out) we only touch the first
    few hundred KB regardless of total file size.
    """
    one_based: Set[int] = set()
    state = "before_header"
    expecting_data = False
    just_blanked = False
    try:
        fh = open(out_path, encoding="utf-8", errors="replace")
    except OSError:
        return set()
    try:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if state == "before_header":
                if _SIESTA_CONSTRAINTS_HEADER_RE.search(line):
                    state = "in_section"
                elif _SIESTA_CONSTRAINT_LINE_RE.match(line):
                    # 5.4.2: the constraint lines with no header above
                    # them -- the section starts at the first one.
                    state = "in_section"
                    expecting_data = True
                continue

            if expecting_data:
                m_data = _SIESTA_CONSTRAINT_RANGES_RE.match(line)
                if m_data is None:
                    break
                body = m_data.group(1)
                for part in body.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    m_range = _RANGE_PIECE_RE.match(part)
                    if m_range is not None:
                        start = int(m_range.group(1))
                        end = int(m_range.group(2))
                        if end >= start:
                            for n in range(start, end + 1):
                                one_based.add(n)
                    elif part.isdigit():
                        one_based.add(int(part))
                expecting_data = False
                just_blanked = False
                continue

            if _SIESTA_CONSTRAINT_LINE_RE.match(line):
                expecting_data = True
                just_blanked = False
                continue

            if not line.strip():
                if just_blanked:
                    break
                just_blanked = True
                continue

            break
    finally:
        fh.close()

    # SIESTA echoes constraints 1-based; translate back to the 0-based
    # Structure identity through the engine index API (never a bare n - 1,
    # which would be wrong for a 0-based engine).
    from ...engine_atom_index import from_engine_index
    return {from_engine_index(n, "siesta") for n in one_based}


# The reading pass -- every rule, the runtime-info probes, the end-of-output
# judgement -- is `siesta_reader`'s, and this parser builds Frames from what it
# reads (`model/parse.md` § 5d.5).  The validation vocabularies, the IterSCF
# timer line and the rule table moved there with it on 2026-09-26.

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
    _STRONG_MARKERS = _G.SNIFF_MARKERS
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
        # A PYTHON FILE IS A SCRIPT, never an engine's output -- the person's
        # PySCF deck, or one of the framework modules shipped beside every job
        # (`runwrap.MONITOR_COMPANIONS`), whose sources quote SIESTA's own
        # lines: `siesta_reader.py` names `Begin Broyden opt. move` in its
        # docstring and was claimed as a SIESTA output (2026-09-26).
        if str(path).lower().endswith(".py"):
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
        """The output through the one reading pass (`siesta_reader`), and a
        Frame per step it read -- the arrays are this module's alone.

        Level-3 fail-soft: every line the reader could not read becomes a
        :class:`ParseWarning` here, logged, and the parse carries on."""
        parse_warnings: List[ParseWarning] = []

        def _warn(line_no: int, line: str, error: str,
                  category: str = "scf") -> None:
            parse_warnings.append(ParseWarning(
                line_no=line_no, snippet=line.rstrip()[:120],
                error=error, category=category))
            _scan_log.warn(error, line_no=line_no,
                           snippet=line.rstrip()[:120], category=category)

        # THE ONE READ (`siesta_reader.read_output`), and the version it read
        # -- an output that states how it ended has the ending no other file
        # changes, so the ending reader keeps this reading's rather than
        # reading the file again (`_run_ending.keep_reading`).
        from . import _run_ending
        before = _run_ending.version_of(path)
        read = read_output(path, warn=_warn).finish()
        _run_ending.keep_reading(path, before, read)
        runtime_info = read["runtime_info"]

        frames: List[Frame] = []
        for step in read["steps"]:
            coords = step["coords"]
            struct = Structure(
                elements=[row[0] for row in coords],
                positions=np.array([row[1:4] for row in coords], dtype=float))
            frames.append(Frame(
                structure   = struct,
                step_index  = len(frames),
                energy      = step["energy"],
                forces      = (np.asarray(step["forces"], dtype=float)
                               if step["forces"] else None),
                max_force   = step["max_force"],
                max_force_constrained = step["max_force_constrained"],
                scf_history = step["scf_history"],
                elapsed_s   = step["elapsed_s"],
                in_progress = step["in_progress"],
            ))
        # In-progress SCF visibility: a run still going with SCF cycles and no
        # coordinates to attach them to yet gets a synthetic Frame, FLAGGED
        # in_progress, so the Results tab draws the SCF chart at once.  Its
        # geometry is the last committed frame's (SIESTA holds geometry
        # constant within an SCF cycle), or a 1-atom placeholder the
        # inspector never renders.
        live = read["live_scf"]
        if live is not None:
            frames.append(Frame(
                structure   = (frames[-1].structure if frames else Structure(
                    elements=["X"], positions=np.zeros((1, 3), dtype=float))),
                step_index  = len(frames),
                energy      = live["energy"],
                scf_history = live["scf_history"],
                elapsed_s   = live["elapsed_s"],
                in_progress = True,
            ))

        # THE ATOMS THE RUN HELD, as its own .out echoes them (above).
        frozen_set = read_frozen_atoms_from_siesta_out(path)
        if frozen_set:
            runtime_info["frozen_atoms"] = sorted(frozen_set)

        _scan_log.info(
            f"parsed {len(frames)} frames, run_state={read['run_state']}, "
            f"{len(parse_warnings)} warnings")
        if read["error_message"]:
            _scan_log.error(read["error_message"])
        return Trajectory(
            source_format  = cls.name,
            frames         = frames,
            lattice        = read["lattice"],
            run_state      = read["run_state"],
            # P-S2: reported, never a verdict.  `None` when no SCF block was
            # seen at all -- a correct final answer, not a hole.
            scf_converged  = read["scf_converged"],
            error_message  = read["error_message"],
            cause          = read["cause"],
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
