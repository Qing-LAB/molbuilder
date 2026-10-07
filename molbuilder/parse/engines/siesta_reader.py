"""A SIESTA ``.out``, read -- the reading pass of the one SIESTA parser.

`model/parse.md` § 5d.5.  **This is the parser**, minus the arrays: every rule
the SIESTA parser matches, the state it keeps, and its judgement of how the
run ended, over plain Python records.  `siesta.py` builds its Frames from what
this reads; the monitor reads a running job with it, fed line by line as the
output grows (`execution/run-reports.md` § 2.3) -- one reader, so the Results
tab and the report a person gets at 3am cannot say different things about one
file.

**Stdlib only, and it travels beside every job** (`runwrap.MONITOR_COMPANIONS`):
its imports -- the SIESTA family's table, the rule engine, the runtime-header
reader -- are imported from the package here and from their copies there.

**What a step is, the output says.**  SIESTA opens every step with its own
words for it and its own number (``Begin Broyden opt. move = 3``, ``Begin FC
step = 3``) and a single point with ``Single-point calculation``
(`siesta_grammar.STEP_BEGIN`) -- so a relaxation, a force-constant run and a
transport rung are each read as what they are, with nothing told.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

try:                                        # inside molbuilder
    from . import siesta_grammar as _G
    from ._section_rules import (
        CONTINUE, END_BUBBLE, END_SECTION,
        SectionRule, any_of, compile_rules, contains_ci, matches_regex_ci,
        starts_with_ci,
    )
    from .molwatch_grammar import parse_runtime_line
except ImportError:                         # beside a job, as the monitor's
    import siesta_grammar as _G
    from _section_rules import (
        CONTINUE, END_BUBBLE, END_SECTION,
        SectionRule, any_of, compile_rules, contains_ci, matches_regex_ci,
        starts_with_ci,
    )
    from molwatch_grammar import parse_runtime_line


# Validation vocabularies for the loose-capture build/diag probes.  We still
# record whatever SIESTA printed, but a token OUTSIDE these sets is flagged as
# a warning rather than silently becoming "what the binary ran with" -- a
# future SIESTA print shape we don't understand must surface, not masquerade
# as ground truth (`model/parse.md` § 7 #9, no silent absorption).
#
# Parallelisation modes SIESTA prints on its ``Parallelisations:`` line.
# "none" is what a build with neither prints (Src/version-info-template.inc).
_SIESTA_PARALLELISATIONS = frozenset({"MPI", "OPENMP", "NONE"})
# Diag.Algorithm vocabulary -- the COMPLETE case-insensitive alias set SIESTA
# accepts (uppercased here), transcribed from the binary's own parser at
# Src/diag_option.F90 (read_diag).  SIESTA ``die()``s on any value outside this
# set, so a real run can only echo one of these; an out-of-set value means a
# future SIESTA added an alias we don't track.
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



#: The table's out-of-memory markers -- a cause that outranks the others.
_OOM = frozenset(m for m, st in _G.FATAL_MARKERS if st == "out_of_memory")


class SiestaReader:
    """The one SIESTA reading pass: :meth:`feed` it lines, ask :meth:`now`
    while the run grows, and :meth:`finish` at the end of the output.

    ``warn(line_no, line, error, category)`` hears every non-fatal line the
    reader could not read (Level-3 fail-soft); without one they are kept on
    :attr:`warnings`.
    """

    def __init__(self, *,
                 warn: Optional[Callable[[int, str, str, str], None]] = None):
        self._warn_to = warn
        self.warnings: List[Dict[str, Any]] = []
        #: Committed steps, oldest first: ``{coords, forces, energy,
        #: max_force, max_force_constrained, scf_history, elapsed_s,
        #: in_progress}`` -- coordinates as ``[element, x, y, z]`` rows.
        self.steps: List[Dict[str, Any]] = []
        self.lattice: Optional[List[List[float]]] = None
        self._pending_lattice: Optional[List[List[float]]] = None
        # HOW THE RUN ENDED (`model/parse.md` § 2b, P-S1) -- a fact about the
        # process, never a grade for the science:
        #   "out_of_memory" -- an OOM marker matched.
        #   "stopped"       -- a fatal marker matched: it did not reach its
        #                      own end.
        #   "ended"         -- ">> End of run" emitted.
        #   "running"       -- no ending marker and no fault: not finished.
        #                      Nothing tells a slow step from a killed job
        #                      (`model/parse.md` § 2b).
        # Convergence is NOT consulted here.  It rides out separately as
        # `scf_converged` (P-S2), because a capped benchmark that never
        # converges still ended perfectly normally.
        self.run_state: str = "running"
        self.error_message: Optional[str] = None
        # Per-SCF-block convergence flag.  None = never saw an SCF block;
        # True = last block converged; False = last block hit "SCF did NOT
        # converge" / "SCF_NOT_CONV".
        self.scf_converged: Optional[bool] = None
        # The ``SCF_NOT_CONV:`` line, held aside.  It is the most INFORMATIVE
        # cause-of-abort SIESTA prints, but it is not itself proof of one (see
        # `_on_scf_fatal_not_converged`), so it is kept here and promoted to
        # ``error_message`` only by whatever does prove it.
        self._scf_not_conv_line: Optional[str] = None
        # Runtime facts, from SIESTA's own startup lines and the echoed
        # `# runtime.<k>:` comments of the .fdf.
        self.runtime_info: Dict[str, Any] = {}
        # SCF iteration history of the current step, and the latest column
        # header the rows are mapped by.
        self._current_scf: List[Dict[str, Any]] = []
        self._prev_E_KS: Optional[float] = None
        self._scf_header: Optional[List[Optional[str]]] = None
        # THE TWO PHASES OF A TRANSIESTA DEVICE (`model/parse.md` § 5d.5).  The
        # phase of the last SCF row; what an iteration printed BEFORE its row
        # -- the ``ts-q:`` charges (NEGF only) and the ``ts-Vha:`` correction
        # (both phases) -- held until that row arrives; whether each phase
        # converged; and the start-up facts TranSIESTA states about itself.
        self.phase: Optional[str] = None
        self._pending_cycle: Dict[str, Any] = {}
        self._ts_q_names: Optional[List[str]] = None
        self.phase_converged: Dict[str, Optional[bool]] = {}
        #: WHAT STOPPED IT: the first fatal line's marker -- the lines after
        #: it are ``die``'s cascade, and an out-of-memory marker outranks the
        #: rest wherever it falls -- or `siesta_grammar.SCF_NOT_CONV_MARKER`
        #: when SIESTA stated the SCF's failure fatal (``(required)``).
        self.cause: Optional[str] = None
        #: A relaxation's last coordinates block says whether it converged:
        #: ``Relaxed`` (True) or ``Final (unrelaxed)`` (False); ``None`` for
        #: a run that relaxes nothing, or has not got there.
        self.relaxed: Optional[bool] = None
        self._scf_criteria: Dict[str, Dict[str, Any]] = {}
        self.ts_info: Dict[str, Any] = {}
        self._ts_echo_open = False
        self._ts_echo_where: List[Optional[str]] = ["options", None]
        self._ts_gf_electrode: Optional[str] = None
        self._ts_charge_take = False
        # The step being read; committed when the next ``outcoor:`` block
        # starts, or at the end of the output.
        self._step_frame: Optional[List[List[Any]]] = None
        self._step_energy: Optional[float] = None
        self._step_max_force: Optional[float] = None
        # SIESTA also emits "Max <val> constrained" when at least one atom is
        # constrained -- the value it compares against ``MD.MaxForceTol``.
        self._step_max_force_constrained: Optional[float] = None
        self._step_forces: List[List[float]] = []
        # The ``siesta: Etot`` of the energy decomposition SIESTA prints after
        # the initial DM and before the first SCF cycle of each step: where
        # the SCF started, kept for display (``runtime_info["initial_etot"]``),
        # never a frame energy.
        self._step_initial_etot: Optional[float] = None
        #: The step SIESTA said it began last: ``(what a step is, its
        #: number)``, in its own words (`siesta_grammar.STEP_BEGIN`).
        self.step_begun: Optional[Tuple[str, int]] = None
        self._active: Optional[SectionRule] = None
        self._in_input_echo = False
        self._line_no = 0
        self._rules = compile_rules(self._rule_table())

    # ---- warnings ---------------------------------------------------------

    def _warn(self, line_no: int, line: str, error: str,
              category: str = "scf") -> None:
        if self._warn_to is not None:
            self._warn_to(line_no, line, error, category)
        else:
            self.warnings.append({"line_no": line_no,
                                  "snippet": line.rstrip()[:120],
                                  "error": error, "category": category})

    # ---- the driver --------------------------------------------------------

    def feed(self, line: str, line_no: Optional[int] = None) -> None:
        """Read one line of the output (without its newline)."""
        self._line_no = line_no if line_no is not None else self._line_no + 1
        line_no = self._line_no
        # Active multi-line section?  Run its ``consume`` first.  CONTINUE /
        # END_SECTION skip to the next line; END_BUBBLE leaves the section
        # AND re-feeds this line through the scan-state rules below.
        if self._active is not None:
            sentinel = self._active.consume(line, line_no)
            if sentinel == CONTINUE:
                return
            if sentinel == END_SECTION:
                self._active = None
                return
            if sentinel == END_BUBBLE:
                self._active = None
            else:
                self._warn(line_no, line,
                           f"section {self._active.name!r} returned "
                           f"unknown sentinel {sentinel!r}; ending section")
                self._active = None
                return
        # Runtime-info probes are orthogonal to the section state machine
        # (free-form key/value lines that can appear anywhere); try them
        # first because they're frequent in a SIESTA preamble.
        if self._scan_runtime_info(line, line_no):
            return
        # THE DECK'S ECHO IS NOT THE RUN SPEAKING
        # (`siesta_grammar.INPUT_ECHO_BEGIN`): SIESTA copies the deck into its
        # output, comments included, and molbuilder's decks name the failures
        # they guard against.  The runtime-info probes above still read the echo: its `# runtime.`
        # comments are the deck's on purpose.
        edge = _G.input_echo_edge(line)
        if edge is not None:
            self._in_input_echo = edge
            return
        if self._in_input_echo:
            return
        # Section dispatch: first rule whose matcher fires wins; order in the
        # rule table is significant (see `_rule_table`).
        rule = self._rules.find_match(line)
        if rule is not None:
            if rule.on_start is not None:
                rule.on_start(line, line_no)
            if rule.consume is not None:
                self._active = rule

    def feed_text(self, text: str) -> "SiestaReader":
        for line in text.splitlines():
            self.feed(line)
        return self

    def new_channel(self) -> "SiestaReader":
        """What follows is the run's OTHER channel -- SIESTA's stderr, which
        its wrapper keeps apart -- read by the same rules after the output.
        The deck's echo is the output's alone, and a section open at the
        output's end does not continue into another file."""
        self._in_input_echo = False
        self._active = None
        return self

    # ---- a step ------------------------------------------------------------

    def _commit(self, in_progress: bool = False) -> None:
        """Commit the accumulated step as a record -- or keep the SCF state
        for the step still to come, when no coordinates have been read yet
        (SCF runs *before* outcoor in SIESTA's stream, and the preamble Etot
        is written even earlier).

        ``in_progress`` is set by :meth:`finish` when the output ends
        mid-step (a structure echo and SCF cycles, but no step-end yet), so
        the flushed step is not mistaken for a completed one.

        THE STEP'S ENERGY, strongest signal first -- each a number SIESTA
        wrote:
          1. the canonical ``siesta: E_KS(eV) = ...`` line;
          2. the most recent FINITE SCF-cycle energy (a Fortran overflow
             reads NaN and is walked past);
          3. A DEVICE SPEAKS FOR ITS NEGF PHASE: when the step ran
             TranSIESTA's loop its energy is the last finite NEGF row's --
             SIESTA's closing ``E_KS(eV)`` and the periodic initialization's
             cycles are other quantities, and reporting the periodic one is
             how a device 584 electrons short read as a sound -437,029 eV
             (§ 5d).  No finite NEGF energy is ``None``.
        No preamble-Etot fallback: a run
        without a finite energy is diverging, and a plausible number would
        hide it.
        """
        if not self._step_frame:
            self._step_frame = None
            self._step_energy = None
            self._step_max_force = None
            self._step_max_force_constrained = None
            self._step_forces = []
            return
        energy = self._step_energy
        if energy is None and self._current_scf:
            for cycle in reversed(self._current_scf):
                candidate = cycle.get("energy")
                if (isinstance(candidate, (int, float))
                        and math.isfinite(candidate)):
                    energy = float(candidate)
                    break
        negf = [c for c in self._current_scf
                if c.get("phase") == _G.PHASE_NEGF]
        if negf:
            energy = next((float(c["energy"]) for c in reversed(negf)
                           if isinstance(c.get("energy"), (int, float))
                           and math.isfinite(c["energy"])), None)
        # SIESTA's IterSCF timer counts from the START OF THE RUN; the step's
        # end-of-time is its LAST cycle's.  ``wall_clock_s`` is never set: a
        # SIESTA .out states the time of day only at its two ends (parse.md
        # § 2a, P-T2).
        elapsed_s: Optional[float] = None
        if self._current_scf:
            cum = self._current_scf[-1].get("elapsed_s")
            if isinstance(cum, (int, float)) and math.isfinite(cum):
                elapsed_s = float(cum)
        self.steps.append({
            "coords": self._step_frame,
            "forces": self._step_forces or None,
            "energy": energy,
            "max_force": self._step_max_force,
            "max_force_constrained": self._step_max_force_constrained,
            "scf_history": list(self._current_scf) or None,
            "elapsed_s": elapsed_s,
            "in_progress": in_progress,
        })
        # The preamble Etot is kept for display: each commit writes its
        # step's, so the LAST committed step wins for a finished run and the
        # in-flight step for an ongoing one.
        if (self._step_initial_etot is not None
                and math.isfinite(self._step_initial_etot)):
            self.runtime_info["initial_etot"] = float(self._step_initial_etot)
        self._step_frame = None
        self._step_energy = None
        self._step_max_force = None
        self._step_max_force_constrained = None
        self._step_forces = []
        self._step_initial_etot = None
        self._current_scf = []
        self._prev_E_KS = None

    # ---- how the run ends ---------------------------------------------------

    def _on_end_of_run(self, line: str, line_no: int) -> None:
        # An abort already seen wins: SIESTA does not print both, but if it
        # somehow did, the abort is the load-bearing fact.
        if self.run_state not in ("stopped", "out_of_memory"):
            self.run_state = "ended"
        # The node's time of day at the end (naive: SIESTA states no zone).
        _G.read_launch_line(line, self.runtime_info)

    def _on_run_start(self, line: str, line_no: int) -> None:
        _G.read_launch_line(line, self.runtime_info)

    def _on_step_begin(self, line: str, line_no: int) -> None:
        m = _G.STEP_BEGIN.match(line)
        if m:
            self.step_begun = (m.group(1), int(m.group(2)))

    def _fatal(self, marker: str, state: str):
        """One handler per fatal marker, built from the shared table
        (`siesta_grammar.FATAL_MARKERS`, `model/parse.md` § 2b)."""
        oom = state == "out_of_memory"

        def _handler(line: str, line_no: int) -> None:
            # THE FIRST FATAL LINE IS THE CAUSE; what follows it is `die`'s
            # cascade, and a memory marker, once seen, is the cause.
            if self.cause is None or (oom and self.cause not in _OOM):
                self.cause = marker
            # P-S1: it did not reach its own end.  An OOM outranks a generic
            # abort -- the aborts that follow are the cascade, the memory is
            # the cause.
            if self.run_state != "out_of_memory":
                self.run_state = state
            # Keep the FIRST marker: later crashes cascade from the original.
            # A held ``SCF_NOT_CONV:`` outranks even that -- it IS the
            # original cause.
            if self.error_message is None:
                self.error_message = (self._scf_not_conv_line
                                      or line.strip()[:200])
        return _handler

    # SCF-block convergence flags.  Two distinct SIESTA emit forms:
    #
    #   1. ``SCF_NOT_CONV: SCF did not converge ... (required).`` -- the
    #      CONSTANT-prefixed form, emitted when SIESTA is about to abort
    #      under ``SCF.MustConverge`` -- or, in a benchmark deck with
    #      ``SCF.MustConverge .false.``, printed and carried straight past.
    #   2. ``SCF did NOT converge`` -- informational; a relaxation can recover
    #      from it in a later step.
    #
    # The success marker ``SCF Convergence by <criterion>`` is unambiguous.
    def _on_scf_converged(self, line: str, line_no: int) -> None:
        self.scf_converged = True
        # Both phases print this same line (``Src/scfconvergence_test.F``,
        # with ``+dQ`` in the NEGF phase's criteria), so it belongs to the
        # phase of the row before it.
        self.phase_converged[self.phase or _G.PHASE_PERIODIC] = True

    def _on_scf_continued(self, line: str, line_no: int) -> None:
        """SIESTA withdrew the convergence it had just printed: until the
        next row, this phase has not converged."""
        self.scf_converged = None
        self.phase_converged[self.phase or _G.PHASE_PERIODIC] = None

    def _on_scf_fatal_not_converged(self, line: str, line_no: int) -> None:
        """``SCF_NOT_CONV:`` -- the root cause when the run dies, and NOT a
        death certificate on its own: a benchmark deck with
        ``SCF.MustConverge .false.`` prints it and reaches ``>> End of run``.
        So the line is HELD, and whatever actually proves the run died -- a
        fatal marker -- promotes it."""
        if self._scf_not_conv_line is None:
            self._scf_not_conv_line = line.strip()[:200]
        # SIESTA STATING IT FATAL is the cause of the death that follows; the
        # tolerated form is not, and a later crash keeps its own cause.
        if _G.SCF_NOT_CONV_REQUIRED in line.lower() and self.cause is None:
            self.cause = _G.SCF_NOT_CONV_MARKER
        self.scf_converged = False
        self.phase_converged[self.phase or _G.PHASE_PERIODIC] = False

    def _on_scf_not_converged(self, line: str, line_no: int) -> None:
        """Soft handler for the informational form.  Flags only."""
        self.scf_converged = False
        self.phase_converged[self.phase or _G.PHASE_PERIODIC] = False

    # ---- coordinates, cell, forces, energies ---------------------------------

    def _on_coords_start(self, line: str, line_no: int) -> None:
        low = line.lower()
        if _G.RELAXED_MARKER in low:
            self.relaxed = True
        elif _G.UNRELAXED_MARKER in low:
            self.relaxed = False
        self._commit()
        self._step_frame = []

    def _consume_coords(self, line: str, line_no: int) -> str:
        stripped = line.strip()
        if not stripped:
            return END_SECTION              # blank-line terminator
        parts = stripped.split()
        if len(parts) < 6:
            # Not an atom row: it may itself start the next section
            # (``outcell: Unit cell vectors (Ang):``), so re-feed it.
            return END_BUBBLE
        try:
            x = _G.fortran_float(parts[0])
            y = _G.fortran_float(parts[1])
            z = _G.fortran_float(parts[2])
        except ValueError:
            # The line that ends a torn outcoor block may be ``>> End of
            # run`` -- re-feed so that rule fires.  A Fortran overflow
            # reads NaN, so this branch is a genuine section end.
            return END_BUBBLE
        self._step_frame.append([parts[-1], x, y, z])
        return CONTINUE

    def _on_cell_start(self, line: str, line_no: int) -> None:
        self._pending_lattice = []

    def _consume_cell(self, line: str, line_no: int) -> str:
        parts = line.strip().split()
        if len(parts) < 3:
            return END_BUBBLE
        try:
            row = [_G.fortran_float(parts[0]),
                   _G.fortran_float(parts[1]),
                   _G.fortran_float(parts[2])]
        except ValueError:
            return END_BUBBLE
        self._pending_lattice.append(row)
        if len(self._pending_lattice) >= 3:
            self.lattice = self._pending_lattice
            self._pending_lattice = None
            return END_SECTION
        return CONTINUE

    def _on_forces_start(self, line: str, line_no: int) -> None:
        self._step_forces = []

    def _consume_forces(self, line: str, line_no: int) -> str:
        parts = line.strip().split()
        if len(parts) < 4:
            # The line that ends the forces block is often the "Max <value>"
            # line or the next section's header: re-feed it.
            return END_BUBBLE
        try:
            int(parts[0])                     # atom index
            fx = _G.fortran_float(parts[1])
            fy = _G.fortran_float(parts[2])
            fz = _G.fortran_float(parts[3])
        except ValueError:
            return END_BUBBLE
        self._step_forces.append([fx, fy, fz])
        return CONTINUE

    def _on_e_ks(self, line: str, line_no: int) -> None:
        """``siesta: E_KS(eV) =       -1234.567``"""
        try:
            self._step_energy = _G.fortran_float(
                line.split("=", 1)[1].split()[0])
        except (ValueError, IndexError) as exc:
            self._warn(line_no, line, f"E_KS line: malformed value: {exc}",
                       category="energy")

    def _on_initial_etot(self, line: str, line_no: int) -> None:
        """``siesta: Etot = ...`` of the decomposition printed after the
        initial DM and before the first SCF cycle of each step."""
        try:
            val = _G.fortran_float(line.split("=", 1)[1].split()[0])
            if math.isfinite(val):
                self._step_initial_etot = val
        except (ValueError, IndexError):
            pass

    def _on_scf_header(self, line: str, line_no: int) -> None:
        parsed = _G.scf_header(line)
        if parsed is not None:
            self._scf_header = parsed

    def _on_scf_data(self, line: str, line_no: int) -> None:
        row = _G.scf_row(line)
        if row is None:
            return
        phase, iscf = row.phase, row.iscf
        vals = _G.scf_floats(line[row.columns_at:].lstrip(),
                             line=line, data_start=row.columns_at)
        if vals is None:
            self._warn(line_no, line, "SCF line: could not tokenize as floats")
            return
        if self._scf_header is not None:
            cycle = _G.cycle_from_header(iscf, vals, self._scf_header)
            if cycle is None:
                self._warn(line_no, line,
                           f"SCF row has {len(vals)} values but header has "
                           f"{len(self._scf_header) - 1} columns "
                           f"({self._scf_header})")
                return
        else:
            cycle = _G.cycle_positional(iscf, vals)
            if cycle is None:
                self._warn(line_no, line,
                           f"SCF line has {len(vals)} floats after iscf; "
                           f"expected 6 (closed-shell) or 7 "
                           f"(spin-polarized), and no column header was seen")
                return
        e_ks = cycle.get("energy")
        if e_ks is None:
            self._warn(line_no, line,
                       "SCF row missing 'energy' (E_KS) -- downstream plot "
                       "can't render this cycle")
            return
        # A NEW PHASE IS NOT A RESTART.  The periodic initialization's cycles
        # stay beside the NEGF loop's, and the new phase starts with its
        # convergence unanswered.
        if phase != self.phase:
            if self.phase is not None:
                self.scf_converged = None
            self.phase = phase
            self.phase_converged.setdefault(phase, None)
            self._prev_E_KS = None
        # iscf==1 starts a new SCF run of THIS phase.
        if iscf == 1:
            if (self._current_scf
                    and self._current_scf[-1].get("phase") == phase):
                self._current_scf = []
            self._prev_E_KS = None
        cycle["delta_E"] = (e_ks - self._prev_E_KS
                            if self._prev_E_KS is not None else 0.0)
        cycle["phase"] = phase
        # What TranSIESTA reported while building this iteration.
        cycle.update(self._pending_cycle)
        self._pending_cycle.clear()
        self._current_scf.append(cycle)
        self._prev_E_KS = e_ks

    def _on_iter_scf_timer(self, line: str, line_no: int) -> None:
        """SIESTA's cumulative IterSCF timer onto the most recent cycle:
        ``Calls`` as ``cumulative_calls``, ``Time`` as ``elapsed_s``.  A timer
        with no cycle before it is dropped; an overflowed field is left off
        rather than costing the others."""
        m = _G.ITER_SCF_TIMER.match(line)
        if m is None or not self._current_scf:
            return
        tokens = m.group(1).split()
        if not tokens:
            return
        last = self._current_scf[-1]
        if "*" not in tokens[0]:
            try:
                last["cumulative_calls"] = int(tokens[0])
            except (ValueError, TypeError):
                pass
        if len(tokens) >= 2:
            cum = _G.fortran_float(tokens[1])
            if math.isfinite(cum):
                last["elapsed_s"] = cum

    # Max-force lines (``Src/write_subs.F``: ``Max`` and, with constrained
    # atoms, ``Max ... constrained`` -- what SIESTA compares against
    # ``MD.MaxForceTol``).  The line is the grammar's; the forces block before
    # it is this reader's context -- a stray "Max <num>" in the preamble must
    # not attribute to the first step.
    def _max_force_match(self, line: str) -> bool:
        m = _G.MAX_FORCE.match(line) if self._step_forces else None
        return bool(m) and not m.group(2)

    def _max_force_constrained_match(self, line: str) -> bool:
        m = _G.MAX_FORCE.match(line) if self._step_forces else None
        return bool(m) and bool(m.group(2))

    def _on_max_force(self, line: str, line_no: int) -> None:
        try:
            self._step_max_force = _G.fortran_float(
                _G.MAX_FORCE.match(line).group(1))
        except (ValueError, AttributeError) as exc:
            self._warn(line_no, line, f"Max-force line: malformed value: {exc}",
                       category="forces")

    def _on_max_force_constrained(self, line: str, line_no: int) -> None:
        try:
            self._step_max_force_constrained = _G.fortran_float(
                _G.MAX_FORCE.match(line).group(1))
        except (ValueError, AttributeError) as exc:
            self._warn(line_no, line,
                       f"Max-force-constrained line: malformed value: {exc}",
                       category="forces")

    # ---- TranSIESTA (`model/parse.md` § 5d.5) ---------------------------------

    def _on_ts_q(self, line: str, line_no: int) -> None:
        """``ts-q:`` -- a header naming the regions, then their values, held
        for the NEGF row that follows.  By NAME, so a third electrode is a
        third pair of columns and not a reader change."""
        q = _G.ts_q_line(line)
        if q is None:
            return
        if q[0] == "names":
            self._ts_q_names = q[1]
            return
        row = _G.ts_q_row(self._ts_q_names, q[1])
        if row is None:
            self._warn(line_no, line, "ts-q row with no matching header",
                       category="negf")
            return
        self._pending_cycle.update(row)

    def _on_ts_vha(self, line: str, line_no: int) -> None:
        m = _G.TS_VHA.match(line)
        if not m:
            return
        try:
            self._pending_cycle["vha_ev"] = _G.fortran_float(m.group(1))
        except ValueError:
            self._warn(line_no, line, "ts-Vha: malformed value",
                       category="negf")

    def _on_ts_echo(self, line: str, line_no: int) -> None:
        """The start-up echo -- TranSIESTA's own account of the settings it
        runs with, the only place some of them exist (the continued-fraction
        contour's pole count).  Only lines inside the star frame are the
        echo; ``ts:`` lines later in the run are the energy decomposition."""
        if _G.TS_ECHO_FRAME.match(line):
            self._ts_echo_open = not self._ts_echo_open
            return
        if not self._ts_echo_open:
            return
        if _G.TS_ECHO_CONTOUR.match(line):
            self._ts_echo_where[:] = ["contours", None]
            return
        m = _G.TS_ECHO_SECTION.match(line)
        if m:
            name = m.group(1).strip()
            if name.lower() == "electrodes":
                self._ts_echo_where[:] = ["electrodes", None]
            elif self._ts_echo_where[0] in ("electrodes", "contours"):
                self._ts_echo_where[1] = name
            return
        m = _G.TS_ECHO_LINE.match(line)
        where, name = self._ts_echo_where
        home = self._ts_echo_home(where, name,
                                  m.group(1).strip() if m else None)
        if m:
            label, value = m.group(1).strip(), m.group(2).strip()
            # A label a segment states twice keeps every value.
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

    def _ts_echo_home(self, where, name, label):
        """Where an echo line belongs: the options, an electrode, or a contour
        SEGMENT -- one per chemical potential or contour part, a new one
        starting at the line that names it (``TS_ECHO_SEGMENT``)."""
        if where == "options" or (where == "electrodes" and name is None):
            return self.ts_info.setdefault("options", {})
        if where == "electrodes":
            return self.ts_info.setdefault("electrodes", {}).setdefault(
                name, {})
        segments = self.ts_info.setdefault("contours", {}).setdefault(
            name or "", [])
        if not segments or (label and _G.TS_ECHO_SEGMENT.search(label)):
            segments.append({})
        return segments[-1]

    def _on_ts_charge_start(self, line: str, line_no: int) -> None:
        """The charge distribution at the switch from the periodic density
        -- the baseline each NEGF iteration's charge is judged against.  The
        FIRST report only."""
        m = _G.TS_CHARGE_START.match(line)
        self._ts_charge_take = (bool(m)
                                and "charge_at_switch" not in self.ts_info)
        if self._ts_charge_take:
            try:
                target = _G.fortran_float(m.group(1))
            except ValueError:
                target = None
            self.ts_info["charge_at_switch"] = {"target": target}

    def _consume_ts_charge(self, line: str, line_no: int) -> str:
        m = _G.TS_CHARGE_ROW.match(line)
        if not m:
            return END_BUBBLE if line.strip() else END_SECTION
        if self._ts_charge_take:
            vals = _G.ts_charge_values(m.group(3))
            q = self.ts_info["charge_at_switch"]
            if len(vals) == 1:
                q[m.group(2)] = vals[0]
            elif len(vals) in (2, 3):
                # Spin-polarized (``Src/ts_charge.F90``): up and down, and on
                # the ``[Q]`` row the total after them.
                q[m.group(2)] = vals[2] if len(vals) == 3 else sum(vals)
                q.setdefault("by_spin", {})[m.group(2)] = vals[:2]
            else:
                self._warn(line_no, line, "charge distribution: malformed row",
                           category="negf")
        return CONTINUE

    def _on_ts_principal_cell(self, line: str, line_no: int) -> None:
        m = _G.TS_PRINCIPAL_CELL.match(line)
        if m:
            self.ts_info.setdefault("electrodes", {}).setdefault(
                m.group(1), {})["principal_cell"] = m.group(2)

    def _on_ts_gf_for(self, line: str, line_no: int) -> None:
        m = _G.TS_GF_FOR.match(line)
        if m:
            self._ts_gf_electrode = m.group(1)

    def _on_ts_gf_stats(self, line: str, line_no: int) -> None:
        if self._ts_gf_electrode is None:
            return
        el = self.ts_info.setdefault("electrodes", {}).setdefault(
            self._ts_gf_electrode, {})
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

    def _on_emadel(self, line: str, line_no: int) -> None:
        m = _G.EMADEL.match(line)
        if m:
            try:
                self.runtime_info["emadel_ev"] = _G.fortran_float(m.group(1))
            except ValueError:
                pass

    # ---- the rule table -------------------------------------------------------

    def _rule_table(self) -> List[SectionRule]:
        """Every line rule, in the order they are tried.  ORDER MATTERS:
        more specific matchers before more general ones; the fatal markers
        first so they win over any section that might eat the line.
        Case-insensitive matching + alias lists give the small-spelling /
        capitalisation tolerance asked for."""
        return [
            # FATAL MARKERS, FROM THE ONE TABLE (`siesta_grammar.FATAL_MARKERS`,
            # `model/parse.md` § 2b), substring matches because SIESTA
            # prefixes them with "node 0: " under MPI.
            *[
                SectionRule(
                    name=f"fatal_{marker.replace(' ', '_').replace(':', '')}",
                    aliases=[marker],
                    start=contains_ci(marker),
                    on_start=self._fatal(marker, state),
                )
                for marker, state in _G.FATAL_MARKERS
            ],
            # SCF_NOT_CONV: the FATAL form, BEFORE the soft "scf did not
            # converge" rule -- a line holding both must reach this one.
            SectionRule(name="fatal_scf_not_conv",
                        aliases=["SCF_NOT_CONV: ..."],
                        start=contains_ci(_G.SCF_NOT_CONV_MARKER),
                        on_start=self._on_scf_fatal_not_converged),
            SectionRule(name="scf_converged",
                        aliases=["SCF Convergence by ..."],
                        start=contains_ci(_G.SCF_CONVERGED_MARKER),
                        on_start=self._on_scf_converged),
            # SIESTA taking the convergence back -- TranSIESTA's charge still
            # off, or too few iterations (``Src/siesta_forces.F90``).
            SectionRule(name="scf_continued",
                        aliases=["SCF cycle continued ..."],
                        start=contains_ci(_G.SCF_CONTINUED_MARKER),
                        on_start=self._on_scf_continued),
            SectionRule(name="scf_not_converged",
                        aliases=["SCF did NOT converge"],
                        start=contains_ci(_G.SCF_NOT_CONVERGED_MARKER),
                        on_start=self._on_scf_not_converged),
            SectionRule(name="cell", aliases=[_G.CELL_BEGIN],
                        start=starts_with_ci(_G.CELL_BEGIN),
                        on_start=self._on_cell_start,
                        consume=self._consume_cell),
            SectionRule(name="end_of_run", aliases=[">> End of run"],
                        start=matches_regex_ci(_G.RUN_END.pattern),
                        on_start=self._on_end_of_run),
            SectionRule(name="coords", aliases=[_G.COORDS_BEGIN],
                        start=starts_with_ci(_G.COORDS_BEGIN),
                        on_start=self._on_coords_start,
                        consume=self._consume_coords),
            SectionRule(name="e_ks", aliases=[_G.E_KS_LINE],
                        # Substring (not prefix): the marker sits mid-line.
                        start=contains_ci(_G.E_KS_LINE),
                        on_start=self._on_e_ks),
            SectionRule(name="initial_etot", aliases=["siesta: Etot ="],
                        start=matches_regex_ci(_G.ETOT_LINE.pattern),
                        on_start=self._on_initial_etot),
            SectionRule(name="forces", aliases=[_G.FORCES_BEGIN],
                        start=contains_ci(_G.FORCES_BEGIN),
                        on_start=self._on_forces_start,
                        consume=self._consume_forces),
            SectionRule(name="scf_header", aliases=["iscf <columns>"],
                        start=matches_regex_ci(_G.SCF_HEADER.pattern),
                        on_start=self._on_scf_header),
            SectionRule(name="scf_data", aliases=["scf: <iscf> ..."],
                        start=matches_regex_ci(_G.SCF_ROW_ERE),
                        on_start=self._on_scf_data),
            SectionRule(name="iter_scf_timer",
                        aliases=["timer: ... IterSCF"],
                        start=matches_regex_ci(_G.ITER_SCF_TIMER.pattern),
                        on_start=self._on_iter_scf_timer),
            # TranSIESTA's lines -- each has its own prefix, so none can
            # collide with a rule above.
            SectionRule(name="ts_q", aliases=["ts-q: ..."],
                        start=matches_regex_ci(_G.TS_Q_ROW.pattern),
                        on_start=self._on_ts_q),
            SectionRule(name="ts_vha", aliases=["ts-Vha: ... eV"],
                        start=matches_regex_ci(_G.TS_VHA.pattern),
                        on_start=self._on_ts_vha),
            SectionRule(name="ts_echo", aliases=["ts: ..."],
                        start=matches_regex_ci(_G.TS_ECHO.pattern),
                        on_start=self._on_ts_echo),
            SectionRule(name="ts_charge_distribution",
                        aliases=["transiesta: Charge distribution, target = ..."],
                        start=matches_regex_ci(_G.TS_CHARGE_START.pattern),
                        on_start=self._on_ts_charge_start,
                        consume=self._consume_ts_charge),
            SectionRule(name="ts_principal_cell",
                        aliases=["<electrode> principal cell is ..."],
                        start=matches_regex_ci(_G.TS_PRINCIPAL_CELL.pattern),
                        on_start=self._on_ts_principal_cell),
            SectionRule(name="ts_gf_for",
                        aliases=["Calculating surface Green functions for: ..."],
                        start=matches_regex_ci(_G.TS_GF_FOR.pattern),
                        on_start=self._on_ts_gf_for),
            SectionRule(name="ts_gf_stats",
                        aliases=["Lopez Sancho ... iterations"],
                        start=any_of(matches_regex_ci(_G.TS_GF_MEAN_STD.pattern),
                                     matches_regex_ci(_G.TS_GF_MIN_MAX.pattern)),
                        on_start=self._on_ts_gf_stats),
            SectionRule(name="emadel", aliases=["siesta: Emadel = ..."],
                        start=matches_regex_ci(_G.EMADEL.pattern),
                        on_start=self._on_emadel),
            SectionRule(name="run_start", aliases=[">> Start of run"],
                        start=matches_regex_ci(_G.RUN_START.pattern),
                        on_start=self._on_run_start),
            SectionRule(name="step_begin",
                        aliases=["Begin <what a step is> = <N>"],
                        start=matches_regex_ci(_G.STEP_BEGIN.pattern),
                        on_start=self._on_step_begin),
            SectionRule(name="max_force", aliases=["Max <value>"],
                        start=self._max_force_match,
                        on_start=self._on_max_force),
            # Registered AFTER ``max_force``: the unconstrained form's
            # matcher gets the first crack, by explicit policy.
            SectionRule(name="max_force_constrained",
                        aliases=["Max <value> constrained"],
                        start=self._max_force_constrained_match,
                        on_start=self._on_max_force_constrained),
        ]

    # ---- the runtime facts -------------------------------------------------------

    def _set_conv_target(self, key: str, value: Any) -> None:
        """``runtime_info['convergence_targets']``, created on first use and
        stamped with its ``source`` once."""
        ct = self.runtime_info.setdefault("convergence_targets", {})
        ct[key] = value
        ct.setdefault("source", "siesta_input_echo")

    def _scan_runtime_info(self, line: str, line_no: int) -> bool:
        """Free-form key/value lines that may appear anywhere.  Returns True
        when the line was consumed (rule dispatch is skipped)."""
        if _G.RUNNING_ON.match(line) or _G.RUNNING_SERIAL.match(line):
            _G.read_launch_line(line, self.runtime_info)
            return True
        # ONE reader of the runtime-header grammar, in the module that owns
        # it: the format is deliberately shared with the molwatch log.
        if parse_runtime_line(line, self.runtime_info):
            return True
        # The run's limits, from SIESTA's ``redata:`` echo.
        target = _G.read_target_line(line)
        if target is not None:
            self._set_conv_target(*target)
            return True
        # THE SCF CRITERIA, from the one reader of the redata pairs: each
        # tolerance keyed by the row column it bounds, with whether SIESTA
        # requires it (`web/trajectory.md` § 3).
        if _G.read_criterion_line(line, self._scf_criteria):
            return True
        # What the binary says about itself (``runtime_info["siesta_build"]``).
        build: Dict[str, Any] = {}
        if _G.read_build_line(line, build):
            have = self.runtime_info.setdefault("siesta_build", {})
            for k, v in build.items():
                have.setdefault(k, v)
            unknown = [t for t in build.get("parallelisations", ())
                       if t not in _SIESTA_PARALLELISATIONS]
            if unknown:
                self._warn(line_no, line,
                           f"unrecognised SIESTA parallelisation token(s) "
                           f"{unknown}; recorded as-is but not in the known "
                           f"set {sorted(_SIESTA_PARALLELISATIONS)}",
                           category="runtime_info")
            return True
        # The solver SIESTA resolved (``Src/diag_option.F90`` print_diag).
        diag: Dict[str, Any] = {}
        if _G.read_diag_line(line, diag):
            have = self.runtime_info.setdefault("siesta_diag", {})
            for k, v in diag.items():
                have.setdefault(k, v)
            algo = str(diag.get("algorithm", "")).upper()
            if algo and algo not in _SIESTA_DIAG_ALGORITHMS:
                self._warn(line_no, line,
                           f"unrecognised SIESTA Diag.Algorithm {algo!r}; "
                           f"recorded as-is but not in the known vocabulary "
                           f"(Src/diag_option.F90) -- likely a newer SIESTA "
                           f"than this parser tracks",
                           category="runtime_info")
            return True
        return False

    # ---- what the run must reach ---------------------------------------------------

    def criteria(self, *, negf: Optional[bool] = None) -> Dict[str, Any]:
        """WHAT EACH PHASE'S SCF MUST REACH, per residual column
        (`web/trajectory.md` § 3): the periodic SCF's from the redata pairs,
        the NEGF loop's from TranSIESTA's own echo.  ``negf`` says whether the
        NEGF phase counts -- by default, when the reader has seen its rows."""
        if not self._scf_criteria:
            return {}
        out = {_G.PHASE_PERIODIC: self._scf_criteria}
        if negf is None:
            negf = self.phase == _G.PHASE_NEGF or any(
                c.get("phase") == _G.PHASE_NEGF
                for s in self.steps for c in (s.get("scf_history") or []))
        if negf:
            out[_G.PHASE_NEGF] = _G.negf_criteria(
                self.ts_info.get("options", {}), self._scf_criteria)
        return out

    # ---- where the run is now, and how it ended -------------------------------------

    def now(self) -> Dict[str, Any]:
        """Where the run is NOW, as read so far -- for a caller watching the
        output grow.  Commits nothing and judges nothing.

        ``{phase, cycle, energy, dDmax, dHmax, dq, residuals, step,
        step_kind, steps_done, max_force, max_force_constrained, criteria,
        targets}``, each present only when the output states it: the
        current SCF row's values (a NEGF row's charge among them), and as
        ``residuals`` each beside the criterion its phase states --
        ``{name: (value, tolerance or None, unit)}``; the step SIESTA last
        began, its number and its words for what a step is, and so how many
        steps are done; the latest largest force, and what each must reach.
        """
        out: Dict[str, Any] = {}
        cycle = (self._current_scf[-1] if self._current_scf else
                 next((s["scf_history"][-1] for s in reversed(self.steps)
                       if s.get("scf_history")), None))
        if cycle is not None:
            out["phase"] = cycle.get("phase")
            out["cycle"] = cycle.get("cycle")
            for key in ("energy", "dDmax", "dHmax", "dq"):
                v = cycle.get(key)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    out[key] = v
        if self.step_begun is not None:
            out["step_kind"], out["step"] = self.step_begun
            # SIESTA begins step N once steps 0..N-1 are done.
            out["steps_done"] = self.step_begun[1]
        force = (self._step_max_force_constrained, self._step_max_force)
        if force == (None, None) and self.steps:
            force = (self.steps[-1]["max_force_constrained"],
                     self.steps[-1]["max_force"])
        if force[0] is not None:
            out["max_force"], out["max_force_constrained"] = force[0], True
        elif force[1] is not None:
            out["max_force"] = force[1]
        crit = self.criteria()
        if crit:
            out["criteria"] = crit
        mine = crit.get(out.get("phase") or "", {})
        residuals = {name: (out[key], mine.get(key, {}).get("tolerance"),
                            mine.get(key, {}).get("unit", ""))
                     for name, key in (("dDmax", "dDmax"), ("dHmax", "dHmax"),
                                       ("dQ", "dq"))
                     if key in out}
        if residuals:
            out["residuals"] = residuals
        targets = {k: v for k, v in
                   self.runtime_info.get("convergence_targets", {}).items()
                   if k != "source"}
        if targets:
            out["targets"] = targets
        return out

    def finish(self) -> Dict[str, Any]:
        """The end of the output: the torn step dropped, the step in flight
        committed and flagged, the run's facts summed.  Returns the reading
        -- ``{steps, live_scf, lattice, run_state, scf_converged, phases,
        relaxed, cause, error_message, runtime_info, warnings}`` -- and the
        reader is done.  The ending among them is THE answer to how a SIESTA
        run ended; `_run_ending.ending_of` asks for it alone.

        ``live_scf`` is the SCF of a run still going that has no coordinates
        to attach it to yet -- the Results tab draws it at once rather than
        appearing stalled for the first SCF of a heavy run (a 200-atom
        junction was the motivating case).
        """
        # A torn outcoor at EOF: the current SCF belongs to a step that
        # cannot be materialized -- drop it with the coordinates.
        if self._active is not None and self._active.name == "coords":
            self._step_frame = None
        # The step in flight: coordinates and SCF cycles but no E_KS yet.
        # SIESTA 5.4.2 writes the input-coordinate echo as an ``outcoor:``
        # block BEFORE the first SCF, so this is the exact in-progress
        # signature.
        in_progress = (self.run_state == "running"
                       and bool(self._step_frame)
                       and bool(self._current_scf)
                       and self._step_energy is None)
        self._commit(in_progress=in_progress)
        live_scf: Optional[Dict[str, Any]] = None
        if self.run_state == "running" and self._current_scf:
            # A device mid-NEGF speaks for its NEGF phase, as `_commit` rules.
            live = ([c for c in self._current_scf
                     if c.get("phase") == _G.PHASE_NEGF] or self._current_scf)
            energy: Optional[float] = None
            for cycle in reversed(live):
                try:
                    fv = float(cycle.get("energy"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    energy = fv
                    break
            elapsed_s: Optional[float] = None
            cum = self._current_scf[-1].get("elapsed_s")
            if isinstance(cum, (int, float)) and math.isfinite(cum):
                elapsed_s = float(cum)
            live_scf = {"energy": energy,
                        "scf_history": list(self._current_scf),
                        "elapsed_s": elapsed_s}
        # NO CONVERGENCE CLAUSE HERE -- `model/parse.md` § 2b, P-S2.  The held
        # `SCF_NOT_CONV:` line is the best cause-of-death sentence when
        # something else proves death, so it fills an empty message only then.
        if (self.run_state in ("stopped", "out_of_memory")
                and self.error_message is None):
            self.error_message = self._scf_not_conv_line
        if (self._step_initial_etot is not None
                and math.isfinite(self._step_initial_etot)):
            self.runtime_info["initial_etot"] = float(self._step_initial_etot)
        # THE PHASES, SUMMED (`model/parse.md` § 5d.6): how many cycles each
        # ran and whether it converged.
        counts: Dict[str, int] = {}
        histories = [s.get("scf_history") or [] for s in self.steps]
        if live_scf is not None:
            histories.append(live_scf["scf_history"])
        for hist in histories:
            for c in hist:
                ph = c.get("phase") or _G.PHASE_PERIODIC
                counts[ph] = counts.get(ph, 0) + 1
        if counts:
            self.runtime_info["scf_phases"] = {
                ph: {"cycles": n, "converged": self.phase_converged.get(ph)}
                for ph, n in counts.items()}
        # Only a run that RAN TranSIESTA gets its facts: every SIESTA run
        # prints the small ``ts:`` block of HS-save flags inside the same star
        # frame, and a relaxation's record would otherwise carry a
        # "transiesta" section about nothing.
        if self.ts_info and (counts.get(_G.PHASE_NEGF)
                             or "electrodes" in self.ts_info
                             or "charge_at_switch" in self.ts_info):
            self.runtime_info["transiesta"] = self.ts_info
        crit = self.criteria(negf=bool(counts.get(_G.PHASE_NEGF)))
        if crit:
            self.runtime_info["scf_criteria"] = crit
        return {"steps": self.steps, "live_scf": live_scf,
                "lattice": self.lattice, "run_state": self.run_state,
                "scf_converged": self.scf_converged,
                "phases": dict(self.phase_converged),
                "relaxed": self.relaxed, "cause": self.cause,
                "error_message": self.error_message,
                "runtime_info": self.runtime_info,
                "warnings": self.warnings}


def read_output(path, *, warn: Optional[Callable[[int, str, str, str], None]]
                = None) -> SiestaReader:
    """THE ONE WAY A SIESTA OUTPUT IS READ: line by line, as the file holds
    it, each line fed with its number -- a reader fed and not yet finished,
    so the caller can add the run's other channel (`new_channel`) before
    :meth:`SiestaReader.finish`.  The registered parser and the ending reader
    (`_run_ending`) both read through it, so the two readings of one version
    of a file are the same reading -- which is what lets the ending reuse a
    parse's (`_run_ending.keep_reading`)."""
    reader = SiestaReader(warn=warn)
    with open(path, "r", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            reader.feed(raw.rstrip("\n"), line_no)
    return reader
