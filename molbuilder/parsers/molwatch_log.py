"""Parser for the unified ``<job>.molwatch.log`` file emitted by
molbuilder-generated PySCF scripts.

The format is intentionally marker-driven (``==== molwatch step N
begin ====`` / ``==== molwatch step N end ====`` brackets, key:value
lines inside each block, ``scf_history begin`` / ``scf_history end``
sub-block) so a parser can locate every field by string match -- no
positional fragility, no dependence on column widths.

Example block::

    ==== molwatch step 0 begin ====
    step_index: 0
    n_atoms: 3
    coordinates (Ang):
       O   0.00000000   0.00000000   0.00000000
       H   0.95700000   0.00000000   0.00000000
       H  -0.23900000   0.92700000   0.00000000
    energy (eV): -76.12345600
    forces (eV/Ang):
       O  -0.00100000  -0.00200000   0.00000000
       H   0.00050000   0.00100000   0.00000000
       H   0.00050000   0.00100000   0.00000000
    max_force (eV/Ang): 0.00240000
    scf_history begin
    #  cycle      energy(eV)         delta_E(eV)        gnorm(eV/Ang)        ddm
           1     -76.00000000        0.00000000        5.00000000e-02    1.00000000e-01
           2     -76.10000000       -0.10000000        5.00000000e-03    1.00000000e-02
    scf_history end
    ==== molwatch step 0 end ====

Robustness:

* The ``# molwatch trajectory log`` header line in the first 5 lines
  is the format-detection marker for ``can_parse``.
* A torn final block (``begin`` without matching ``end``) is dropped
  silently -- so molwatch can tail a still-running job and won't
  show a half-written final step.
* Missing residual values may appear as the literal string ``None``;
  the parser converts those to JSON ``null``.

Engine identification: the ``# engine: <name>`` header line
determines what value goes into the parsed dict's ``source_format``
field.  The molwatch UI uses this for cosmetic things (axis labels)
but residual-axis selection is data-driven from the per-cycle keys.

Dispatch architecture (refactored 2026-06-01):
==============================================

Built on the shared rule primitives in :mod:`molbuilder.parsers._rules`
(introduced 2026-05-29 for SIESTA, generalised 2026-05-31 with the
``_PatternMatcher`` + ``CompiledRules`` abstraction).  The driver
maintains TWO compiled rule tables:

  * ``OUT_BLOCK_RULES`` -- header / footer markers + block_begin.
    Active when no ``==== molwatch step N begin ====`` line has been
    seen yet (or after a matching ``end`` marker closed the previous
    block).
  * ``IN_BLOCK_RULES`` -- block_end + section markers (coordinates /
    forces / scf_history begin / energy / max_force / wall_time) +
    block_begin (for torn-block recovery: a fresh ``begin`` mid-block
    abandons the partial frame and starts a new one).

The driver loop switches between the two tables on each iteration
based on the ``in_block`` flag, which the block_begin / block_end
``on_start`` callbacks mutate via ``nonlocal``.  Multi-line section
consume callbacks (``_consume_coords`` / ``_consume_forces`` /
``_consume_scf``) are unchanged from the pre-refactor state machine
semantics; they just sit on ``SectionRule.consume`` now instead of
inline ``if sub == "in_coords": ...`` blocks.

Why this matters: the pre-refactor scan loop did 4 cross-block
``re.match`` probes + 2 boundary ``re.search`` probes + 6 ``startswith``
checks per line in the worst case -- 12 individual checks per line on
a multi-MB log.  The combined-regex pre-filter collapses these into
ONE DFA scan per line; per-rule verification runs only on a hit.
Same correctness contract as SIESTA (preserved via the same
trace-equivalence approach used in commit e1a517d).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..frame import Frame, Trajectory
from ..structure import Structure
from ._rules import (
    CONTINUE, END_BUBBLE, END_SECTION,
    SectionRule, compile_rules, matches_regex_ci, starts_with_ci,
)
from .base import TrajectoryParser


# Compiled regexes kept around for the on_start callbacks to extract
# captured groups (the rule matchers only decide whether to dispatch;
# they don't return match objects).  Patterns mirror the matcher
# fragments used in the rule list below.
_BEGIN_RE     = re.compile(r"====\s*molwatch\s+step\s+(\d+)\s+begin\s*====")
_END_RE       = re.compile(r"====\s*molwatch\s+step\s+(\d+)\s+end\s*====")
_HEADER_RE    = re.compile(r"^#\s*molwatch\s+trajectory\s+log", re.IGNORECASE)
_ENGINE_RE    = re.compile(r"^#\s*engine:\s*(\S+)", re.IGNORECASE)
# Runtime-info header lines look like ``# runtime.<key>: <value>``
# (one per canonical key from molbuilder.runtime_info).  Captures both
# halves so the parser can drop them into Trajectory.runtime_info as
# a plain dict[str, str].  Numeric values are stringified by the
# emitter; the inspector parses ints/bools out of the strings as
# needed.
_RUNTIME_RE   = re.compile(r"^#\s*runtime\.([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$")
# Convergence-target header lines look like
# ``# convergence.<key>: <value>``.  Same shape as ``# runtime.*``
# above so a single regex family covers both — keeps the molwatch
# file format readable.  Captures into
# ``runtime_info["convergence_targets"]`` (a nested subdict, NOT a
# flat key) so the Results-tab inspector can render the threshold
# line + "current vs target" readout without having to guess at
# semantics from prefix matching.
_CONVERGENCE_RE = re.compile(
    r"^#\s*convergence\.([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$")
# Run-state markers, written by the inlined PySCF emitter via
# atexit/excepthook hooks.  Both lines may appear at the FOOTER of
# a log (atexit fires after the last step block).  When neither is
# present the run is treated as ongoing.
_CONCLUDED_RE = re.compile(r"^#\s*concluded:\s*(.+)$", re.IGNORECASE)
_ERROR_RE     = re.compile(r"^#\s*error:\s*(.+)$",     re.IGNORECASE)


def _maybe_float(token: str) -> Optional[float]:
    """Convert a token to float; return None for the literal 'None'."""
    if token == "None" or token == "null":
        return None
    try:
        return float(token)
    except ValueError:
        return None


class MolwatchLogParser(TrajectoryParser):
    name  = "molwatch"
    label = "molwatch unified log (.molwatch.log)"
    hint  = (
        "the unified per-step log emitted by molbuilder-generated PySCF "
        "scripts (e.g. <job>.molwatch.log)"
    )

    @classmethod
    def can_parse(cls, path: str) -> bool:
        try:
            with open(path, "r", errors="replace") as fh:
                head = [next(fh, "") for _ in range(5)]
        except OSError:
            return False
        return any(_HEADER_RE.match(line) for line in head)

    @classmethod
    def parse(cls, path: str) -> Trajectory:
        engine = "molwatch"
        frames: List[Frame] = []
        # Run-state markers default to "ongoing" -- only flip when the
        # writer emitted explicit `# concluded:` / `# error:` lines.
        run_state: str = "ongoing"
        error_message: Optional[str] = None
        # Runtime facts the emitter wrote in the file header
        # (``# runtime.<key>: <value>`` lines).  Empty dict when the
        # writer didn't emit them -- older log files just don't get
        # the CPU/GPU/Host rows in the /results inspector.
        runtime_info: Dict[str, Any] = {}

        # In-block accumulators; commit only on a matching `end` marker.
        in_block = False
        block_idx: Optional[int] = None
        block_frame: List[List[Any]] = []
        block_energy: Optional[float] = None
        block_forces: List[List[float]] = []
        block_max_force: Optional[float] = None
        block_scf: List[Dict[str, Any]] = []
        block_wall_time: Optional[float] = None

        def _reset_block() -> None:
            nonlocal in_block, block_idx, block_frame, block_energy
            nonlocal block_forces, block_max_force, block_scf, block_wall_time
            in_block        = False
            block_idx       = None
            block_frame     = []
            block_energy    = None
            block_forces    = []
            block_max_force = None
            block_scf       = []
            block_wall_time = None

        # ---- header / footer on_start callbacks --------------------------

        def _on_error(line: str, line_no: int) -> None:
            nonlocal run_state, error_message
            m = _ERROR_RE.match(line)
            if m:
                # Error has priority over concluded; the writer
                # emits "# error:" then "# concluded:" so a clean
                # parse of both should land on "error".
                error_message = m.group(1).strip()
                run_state = "error"

        def _on_concluded(line: str, line_no: int) -> None:
            nonlocal run_state
            if run_state != "error":
                run_state = "finished"

        def _on_engine(line: str, line_no: int) -> None:
            nonlocal engine
            m = _ENGINE_RE.match(line)
            if m:
                engine = m.group(1)

        def _on_runtime(line: str, line_no: int) -> None:
            m = _RUNTIME_RE.match(line)
            if not m:
                return
            key, val = m.group(1), m.group(2).strip()
            # Coerce trivially-typed values (int / bool / None) so the
            # inspector can compare without special-casing the bag.
            # Non-matching values stay strings.
            if val == "None":
                runtime_info[key] = None
            elif val in ("True", "False"):
                runtime_info[key] = (val == "True")
            else:
                try:
                    runtime_info[key] = int(val)
                except ValueError:
                    runtime_info[key] = val

        def _on_convergence(line: str, line_no: int) -> None:
            """``# convergence.<key>: <value>`` header lines populate
            ``runtime_info["convergence_targets"]``.  Stamps
            ``source = "molwatch_header"`` on first hit so the UI knows
            where the values came from."""
            m = _CONVERGENCE_RE.match(line)
            if not m:
                return
            key, val = m.group(1), m.group(2).strip()
            ct = runtime_info.setdefault("convergence_targets", {})
            ct.setdefault("source", "molwatch_header")
            # Coerce int / float / bool / None into Python types.
            if val == "None" or val == "null":
                ct[key] = None
                return
            if val in ("True", "False"):
                ct[key] = (val == "True")
                return
            try:
                ct[key] = int(val)
                return
            except ValueError:
                pass
            try:
                ct[key] = float(val)
                return
            except ValueError:
                pass
            ct[key] = val

        # ---- block boundary on_start callbacks ---------------------------

        def _on_block_begin(line: str, line_no: int) -> None:
            nonlocal in_block, block_idx
            m = _BEGIN_RE.search(line)
            # Any half-built previous block is silently abandoned --
            # _reset_block clears every accumulator AND sets in_block
            # False; we flip back to True after for the new block.
            _reset_block()
            in_block = True
            if m:
                try:
                    block_idx = int(m.group(1))
                except ValueError:
                    block_idx = None

        def _on_block_end(line: str, line_no: int) -> None:
            nonlocal in_block
            if not in_block:
                return
            if block_frame:
                elements  = [row[0] for row in block_frame]
                positions = np.array([row[1:4] for row in block_frame],
                                     dtype=float)
                struct = Structure(elements=elements, positions=positions)
                forces_arr = (np.asarray(block_forces, dtype=float)
                              if block_forces else None)
                idx = (block_idx if block_idx is not None
                       else len(frames))
                frames.append(Frame(
                    structure   = struct,
                    step_index  = idx,
                    energy      = block_energy,
                    forces      = forces_arr,
                    max_force   = block_max_force,
                    # Always a list (possibly empty) -- the .molwatch.log
                    # format always carries an scf_history block per step,
                    # even if it's empty (e.g. for an initial-state
                    # preview block).  None is reserved for parsers that
                    # genuinely have no SCF data source.
                    scf_history = list(block_scf),
                    wall_time   = block_wall_time,
                ))
            _reset_block()

        # ---- scalar key:value on_start callbacks -------------------------

        def _on_energy(line: str, line_no: int) -> None:
            nonlocal block_energy
            block_energy = _maybe_float(line.strip().split(":", 1)[1].strip())

        def _on_max_force(line: str, line_no: int) -> None:
            nonlocal block_max_force
            block_max_force = _maybe_float(
                line.strip().split(":", 1)[1].strip())

        def _on_wall_time(line: str, line_no: int) -> None:
            # Unix epoch seconds emitted by both the SIESTA-side
            # write_initial_preview helper and the inlined PySCF
            # MolwatchEmitter.  Optional -- older logs (and a log torn
            # before the wall_time line) parse fine with
            # block_wall_time = None, just no elapsed-time display.
            nonlocal block_wall_time
            block_wall_time = _maybe_float(
                line.strip().split(":", 1)[1].strip())

        # ---- multi-line section consume callbacks ------------------------

        def _consume_coords(line: str, line_no: int) -> str:
            stripped = line.strip()
            # Blank line or any line with ":" (next section header)
            # ends the coords block; bubble so the same line gets
            # re-fed through scan-state rules.  Matches pre-refactor
            # ``if not stripped or ":" in stripped:`` semantics.
            if not stripped or ":" in stripped:
                return END_BUBBLE
            parts = stripped.split()
            if len(parts) < 4:
                return END_BUBBLE
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                return END_BUBBLE
            block_frame.append([parts[0], x, y, z])
            return CONTINUE

        def _consume_forces(line: str, line_no: int) -> str:
            stripped = line.strip()
            if not stripped or ":" in stripped:
                return END_BUBBLE
            parts = stripped.split()
            if len(parts) < 4:
                return END_BUBBLE
            try:
                fx = float(parts[1])
                fy = float(parts[2])
                fz = float(parts[3])
            except ValueError:
                return END_BUBBLE
            block_forces.append([fx, fy, fz])
            return CONTINUE

        def _consume_scf(line: str, line_no: int) -> str:
            stripped = line.strip()
            # Explicit terminator.  Consumed (END_SECTION, not BUBBLE)
            # because the marker itself isn't a section header.
            if stripped.startswith("scf_history end"):
                return END_SECTION
            # Comments + blank lines: skip without leaving the section.
            if not stripped or stripped.startswith("#"):
                return CONTINUE
            parts = stripped.split()
            if len(parts) < 5:
                # Malformed row; skip but stay in scf.  Pre-refactor
                # behaviour: the for-loop continued; in_scf was not
                # exited until the explicit "scf_history end" marker.
                return CONTINUE
            try:
                cycle = int(parts[0])
                energy = float(parts[1])
                delta_E = float(parts[2])
            except ValueError:
                return CONTINUE
            gnorm = _maybe_float(parts[3])
            ddm   = _maybe_float(parts[4])
            block_scf.append({
                "cycle":   cycle,
                "energy":  energy,
                "delta_E": delta_E,
                "gnorm":   gnorm,
                "ddm":     ddm,
            })
            return CONTINUE

        # ---- rule tables -------------------------------------------------
        # Block-boundary rules: fire in BOTH out- and in-block states.
        # block_begin recovers from torn blocks (a fresh begin abandons
        # the partial frame); block_end commits the current frame.

        block_begin_rule = SectionRule(
            name="block_begin",
            aliases=["==== molwatch step N begin ===="],
            start=matches_regex_ci(
                r"====\s*molwatch\s+step\s+\d+\s+begin\s*===="),
            on_start=_on_block_begin,
        )
        block_end_rule = SectionRule(
            name="block_end",
            aliases=["==== molwatch step N end ===="],
            start=matches_regex_ci(
                r"====\s*molwatch\s+step\s+\d+\s+end\s*===="),
            on_start=_on_block_end,
        )

        # Outside-block: header/footer markers + block_begin.  Order
        # below mirrors the pre-refactor priority -- error > concluded
        # > engine > runtime > block_begin -- though the markers are
        # mutually exclusive in practice, so reordering is safe.
        out_block_rules: List[SectionRule] = [
            SectionRule(
                name="fatal_error",
                aliases=["# error: ..."],
                start=matches_regex_ci(r"^#\s*error:\s*."),
                on_start=_on_error,
            ),
            SectionRule(
                name="concluded",
                aliases=["# concluded: ..."],
                start=matches_regex_ci(r"^#\s*concluded:\s*."),
                on_start=_on_concluded,
            ),
            SectionRule(
                name="engine",
                aliases=["# engine: ..."],
                start=matches_regex_ci(r"^#\s*engine:\s*\S"),
                on_start=_on_engine,
            ),
            SectionRule(
                name="runtime",
                aliases=["# runtime.<key>: ..."],
                start=matches_regex_ci(
                    r"^#\s*runtime\.[a-zA-Z_][a-zA-Z0-9_]*:"),
                on_start=_on_runtime,
            ),
            SectionRule(
                name="convergence",
                aliases=["# convergence.<key>: ..."],
                start=matches_regex_ci(
                    r"^#\s*convergence\.[a-zA-Z_][a-zA-Z0-9_]*:"),
                on_start=_on_convergence,
            ),
            block_begin_rule,
        ]

        # Inside-block: block_end + section markers.  block_begin is
        # also in this list so a torn-block recovery (begin mid-block)
        # works regardless of which state we're in.
        in_block_rules: List[SectionRule] = [
            block_begin_rule,
            block_end_rule,
            SectionRule(
                name="coords",
                aliases=["coordinates (Ang):"],
                start=starts_with_ci("coordinates"),
                consume=_consume_coords,
            ),
            SectionRule(
                name="forces",
                aliases=["forces (eV/Ang):"],
                start=starts_with_ci("forces"),
                consume=_consume_forces,
            ),
            SectionRule(
                name="scf_history",
                aliases=["scf_history begin"],
                start=starts_with_ci("scf_history begin"),
                consume=_consume_scf,
            ),
            SectionRule(
                name="energy",
                aliases=["energy (eV):"],
                start=starts_with_ci("energy (eV):"),
                on_start=_on_energy,
            ),
            SectionRule(
                name="max_force",
                aliases=["max_force (eV/Ang):"],
                start=starts_with_ci("max_force (eV/Ang):"),
                on_start=_on_max_force,
            ),
            SectionRule(
                name="wall_time",
                aliases=["wall_time:"],
                start=starts_with_ci("wall_time:"),
                on_start=_on_wall_time,
            ),
        ]

        out_compiled = compile_rules(out_block_rules)
        in_compiled  = compile_rules(in_block_rules)

        # ---- state-machine driver ---------------------------------------
        active: Optional[SectionRule] = None

        with open(path, "r", errors="replace") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.rstrip("\n")

                # Active multi-line section?  Run its consume first.
                # CONTINUE / END_SECTION skip to next line; END_BUBBLE
                # leaves the section AND re-feeds this line through
                # scan-state rules below.
                if active is not None:
                    sentinel = active.consume(line, line_no)
                    if sentinel == CONTINUE:
                        continue
                    if sentinel == END_SECTION:
                        active = None
                        continue
                    if sentinel == END_BUBBLE:
                        active = None
                        # fall through to scan-state dispatch
                    else:
                        # Unknown sentinel; defensively drop the
                        # section (defensive copy of SIESTA driver).
                        active = None
                        continue

                # Pick the rule table for the current block state.
                # The block_begin / block_end on_start callbacks flip
                # ``in_block`` so the NEXT iteration uses the matching
                # table.
                compiled = in_compiled if in_block else out_compiled
                rule = compiled.find_match(line)
                if rule is not None:
                    if rule.on_start is not None:
                        rule.on_start(line, line_no)
                    if rule.consume is not None:
                        active = rule

        # Torn final block at EOF: drop it (in_block True, no `end` seen).

        # Surface the .molstruct.json sidecar's frozen_atoms list (same
        # contract as SIESTA + PySCF parsers).  Used by the trajectory
        # inspector's "Hide frozen atoms" overlay.
        from ._sidecar import read_frozen_atoms
        frozen = sorted(read_frozen_atoms(path))
        if frozen:
            runtime_info["frozen_atoms"] = frozen

        return Trajectory(
            source_format = engine,
            frames        = frames,
            lattice       = None,
            run_state     = run_state,
            error_message = error_message,
            runtime_info  = runtime_info,
        )
