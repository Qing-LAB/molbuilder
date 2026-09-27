"""A molwatch log, read -- the reading pass of the one molwatch parser.

**This is the parser**, minus the arrays: the header, step-block and footer
rules the molwatch parser matches (`molwatch_grammar`'s lines) and the state
it keeps, over plain Python records.  `molwatch.py` builds its Frames from what
this reads; the monitor reads a running PySCF job's progress log with it, fed
line by line as the log grows (`execution/run-reports.md` § 2.3).  It was the
inside of `molwatch._parse_molwatch_log_impl` until 2026-09-26.

**Stdlib only, and it travels beside every job** (`runwrap.MONITOR_COMPANIONS`).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:                                        # inside molbuilder
    from . import molwatch_grammar as _MG
    from ._section_rules import (
        CONTINUE, END_BUBBLE, END_SECTION,
        SectionRule, compile_rules, matches_regex_ci, starts_with_ci,
    )
except ImportError:                         # beside a job, as the monitor's
    import molwatch_grammar as _MG
    from _section_rules import (
        CONTINUE, END_BUBBLE, END_SECTION,
        SectionRule, compile_rules, matches_regex_ci, starts_with_ci,
    )


class MolwatchReader:
    """The one molwatch reading pass: :meth:`feed` it lines, ask :meth:`now`
    while the run grows, and :meth:`finish` at the end of the log.

    ``stage`` is the rung's stage token -- a staged header's convergence
    targets are keyed by it (``# convergence.01_coarse.<leaf>``).
    """

    def __init__(self, *, stage: Optional[str] = None):
        self.stage = stage
        #: Committed step blocks, oldest first: ``{index, kind, coords,
        #: energy, forces, max_force, scf_history, wall_clock_s}``.  Only a
        #: block with coordinates is committed -- the rule the parser always
        #: had -- and a torn final block (``begin`` with no ``end``) is not.
        self.blocks: List[Dict[str, Any]] = []
        self.engine = "molwatch"
        self.run_state: str = "running"      # parse.md 2b, P-S1
        self.error_message: Optional[str] = None
        self.runtime_info: Dict[str, Any] = {}
        self._in_block = False
        self._block: Dict[str, Any] = {}
        self._active: Optional[SectionRule] = None
        self._line_no = 0
        self._reset_block()
        out_rules, in_rules = self._rule_tables()
        self._out_rules = compile_rules(out_rules)
        self._in_rules = compile_rules(in_rules)

    def _reset_block(self) -> None:
        self._in_block = False
        self._block = {"index": None, "kind": None, "coords": [],
                       "energy": None, "forces": [], "max_force": None,
                       "scf_history": [], "wall_clock_s": None}

    # ---- the driver -------------------------------------------------------

    def feed(self, line: str, line_no: Optional[int] = None) -> None:
        """Read one line of the log (without its newline)."""
        self._line_no = line_no if line_no is not None else self._line_no + 1
        line_no = self._line_no
        if self._active is not None:
            sentinel = self._active.consume(line, line_no)
            if sentinel == CONTINUE:
                return
            if sentinel == END_SECTION:
                self._active = None
                return
            if sentinel == END_BUBBLE:
                self._active = None          # fall through to scan dispatch
            else:
                self._active = None
                return
        rules = self._in_rules if self._in_block else self._out_rules
        rule = rules.find_match(line)
        if rule is not None:
            if rule.on_start is not None:
                rule.on_start(line, line_no)
            if rule.consume is not None:
                self._active = rule

    def feed_text(self, text: str) -> "MolwatchReader":
        for line in text.splitlines():
            self.feed(line)
        return self

    # ---- header / footer ----------------------------------------------------

    def _on_error(self, line: str, line_no: int) -> None:
        m = _MG.ERROR.match(line)
        if m:
            self.error_message = m.group(1).strip()
            self.run_state = "stopped"

    def _on_concluded(self, line: str, line_no: int) -> None:
        if self.run_state != "stopped":
            self.run_state = "ended"

    def _on_engine(self, line: str, line_no: int) -> None:
        m = _MG.ENGINE.match(line)
        if m:
            self.engine = m.group(1)

    def _on_runtime(self, line: str, line_no: int) -> None:
        _MG.parse_runtime_line(line, self.runtime_info)

    def _on_convergence(self, line: str, line_no: int) -> None:
        """``# convergence.<key>: <value>`` into
        ``runtime_info["convergence_targets"]``, flat or staged (#534), the
        ``source`` stamped once -- through the format's one reader."""
        ct = self.runtime_info.setdefault("convergence_targets", {})
        if not _MG.parse_convergence_line(line, ct) and len(ct) == 0:
            # No match and nothing accumulated: absence reads as absence.
            self.runtime_info.pop("convergence_targets", None)

    # ---- the step block ---------------------------------------------------------

    def _on_block_begin(self, line: str, line_no: int) -> None:
        m = _MG.BLOCK_BEGIN.search(line)
        self._reset_block()
        self._in_block = True
        if m:
            try:
                self._block["index"] = int(m.group(1))
            except ValueError:
                pass

    def _on_block_end(self, line: str, line_no: int) -> None:
        if not self._in_block:
            return
        if self._block["coords"]:
            block = dict(self._block)
            if block["index"] is None:
                block["index"] = len(self.blocks)
            self.blocks.append(block)
        self._reset_block()

    def _on_kind(self, line: str, line_no: int) -> None:
        self._block["kind"] = line.strip().partition(":")[2].strip() or None

    def _on_energy(self, line: str, line_no: int) -> None:
        self._block["energy"] = _MG.field_value(line)

    def _on_max_force(self, line: str, line_no: int) -> None:
        self._block["max_force"] = _MG.field_value(line)

    def _on_wall_time(self, line: str, line_no: int) -> None:
        self._block["wall_clock_s"] = _MG.field_value(line)

    def _consume_coords(self, line: str, line_no: int) -> str:
        stripped = line.strip()
        if not stripped or ":" in stripped:
            return END_BUBBLE
        parts = stripped.split()
        if len(parts) < 4:
            return END_BUBBLE
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            return END_BUBBLE
        self._block["coords"].append([parts[0], x, y, z])
        return CONTINUE

    def _consume_forces(self, line: str, line_no: int) -> str:
        stripped = line.strip()
        if not stripped or ":" in stripped:
            return END_BUBBLE
        parts = stripped.split()
        if len(parts) < 4:
            return END_BUBBLE
        try:
            fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            return END_BUBBLE
        self._block["forces"].append([fx, fy, fz])
        return CONTINUE

    def _consume_scf(self, line: str, line_no: int) -> str:
        stripped = line.strip()
        if stripped.startswith(_MG.SCF_HISTORY_END):
            return END_SECTION
        row = _MG.scf_history_row(stripped)
        if row is not None:
            self._block["scf_history"].append(row)
        return CONTINUE

    # ---- the rule tables ----------------------------------------------------------

    def _rule_tables(self):
        """Outside a step block, and inside one -- the driver flips between
        them on the block's ``begin`` / ``end`` lines."""
        begin = SectionRule(name="block_begin",
                            aliases=["==== molwatch step N begin ===="],
                            start=matches_regex_ci(_MG.BLOCK_BEGIN.pattern),
                            on_start=self._on_block_begin)
        end = SectionRule(name="block_end",
                          aliases=["==== molwatch step N end ===="],
                          start=matches_regex_ci(_MG.BLOCK_END.pattern),
                          on_start=self._on_block_end)
        out_rules = [
            SectionRule(name="fatal_error", aliases=["# error: ..."],
                        start=matches_regex_ci(_MG.ERROR.pattern),
                        on_start=self._on_error),
            SectionRule(name="concluded", aliases=["# concluded: ..."],
                        start=matches_regex_ci(_MG.CONCLUDED.pattern),
                        on_start=self._on_concluded),
            SectionRule(name="engine", aliases=["# engine: ..."],
                        start=matches_regex_ci(_MG.ENGINE.pattern),
                        on_start=self._on_engine),
            SectionRule(name="runtime", aliases=["# runtime.<key>: ..."],
                        start=matches_regex_ci(_MG.RUNTIME.pattern),
                        on_start=self._on_runtime),
            SectionRule(name="convergence",
                        aliases=["# convergence.<key>: ...",
                                 "# convergence.<stage>.<key>: ..."],
                        start=matches_regex_ci(_MG.CONVERGENCE.pattern),
                        on_start=self._on_convergence),
            begin,
        ]
        in_rules = [
            begin,
            end,
            SectionRule(name="coords", aliases=["coordinates (Ang):"],
                        start=starts_with_ci("coordinates"),
                        consume=self._consume_coords),
            SectionRule(name="forces", aliases=["forces (eV/Ang):"],
                        start=starts_with_ci("forces"),
                        consume=self._consume_forces),
            SectionRule(name="scf_history", aliases=["scf_history begin"],
                        start=starts_with_ci(_MG.SCF_HISTORY_BEGIN),
                        consume=self._consume_scf),
            SectionRule(name="kind", aliases=["kind: ..."],
                        start=starts_with_ci(_MG.KIND),
                        on_start=self._on_kind),
            SectionRule(name="energy", aliases=["energy (eV):"],
                        start=starts_with_ci(_MG.ENERGY),
                        on_start=self._on_energy),
            SectionRule(name="max_force", aliases=["max_force (eV/Ang):"],
                        start=starts_with_ci(_MG.MAX_FORCE),
                        on_start=self._on_max_force),
            SectionRule(name="wall_time", aliases=["wall_time:"],
                        start=starts_with_ci(_MG.WALL_TIME),
                        on_start=self._on_wall_time),
        ]
        return out_rules, in_rules

    # ---- where the run is now, and the end of the log -------------------------------

    def targets(self) -> Dict[str, Any]:
        """The convergence targets that apply to this rung: its stage's,
        from a staged header, or the flat header's."""
        ct = self.runtime_info.get("convergence_targets") or {}
        mine = ct.get(self.stage) if self.stage else None
        applies = (mine if isinstance(mine, dict) else
                   {k: v for k, v in ct.items() if not isinstance(v, dict)})
        return {k: v for k, v in applies.items() if k != "source"}

    def now(self) -> Dict[str, Any]:
        """Where the run is NOW, as read so far -- the last step block the
        run wrote, which is written when its step ENDS.  ``{step, energy,
        max_force, cycle, residuals, scf_cycles, last_cycle, steps_done,
        scf_rows, targets}``, each present only when the log states it:
        the step's index, energy and largest force; its SCF's last cycle
        and that cycle's residuals -- ``{name: (value, None, unit)}``, the
        log stating no tolerance in their units (its ``scf_energy_tol`` is
        PySCF's, in Hartree); how many steps are done and how many SCF
        cycles they ran.

        A PREVIEW IS NOT PROGRESS: the step-0 block of kind
        ``initial_preview`` (`molwatch_grammar.PREVIEW_KIND`) shows the input
        before the run writes anything -- and a spectrum deck never writes
        more -- so it states no step.
        """
        out: Dict[str, Any] = {}
        done = [b for b in self.blocks if b.get("kind") != _MG.PREVIEW_KIND]
        block = done[-1] if done else None
        if block is not None:
            out["step"] = block["index"]
            if block["energy"] is not None:
                out["energy"] = block["energy"]
            if block["max_force"] is not None:
                out["max_force"] = block["max_force"]
            if block["scf_history"]:
                last = block["scf_history"][-1]
                out["scf_cycles"] = len(block["scf_history"])
                out["last_cycle"] = last
                if last.get("cycle") is not None:
                    out["cycle"] = last["cycle"]
                residuals = {
                    name: (last[key], None, unit)
                    for name, key, unit in (("dE", "delta_E", "eV"),
                                            ("|g|", "gnorm", "eV/Ang"),
                                            ("ddm", "ddm", ""))
                    if isinstance(last.get(key), (int, float))
                    and math.isfinite(last[key])}
                if residuals:
                    out["residuals"] = residuals
            out["steps_done"] = len(done)
            rows = sum(len(b["scf_history"]) for b in done)
            if rows:
                out["scf_rows"] = rows
        targets = self.targets()
        if targets:
            out["targets"] = targets
        return out

    def finish(self) -> Dict[str, Any]:
        """The end of the log -- a torn final block is dropped -- and the
        reading: ``{blocks, engine, run_state, error_message,
        runtime_info}``."""
        return {"blocks": self.blocks, "engine": self.engine,
                "run_state": self.run_state,
                "error_message": self.error_message,
                "runtime_info": self.runtime_info}
