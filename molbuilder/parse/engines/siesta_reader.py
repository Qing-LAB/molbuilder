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

import copy
try:
    from .transiesta import scf_words as _scf_words, setup_words as _setup_words
except ImportError:                         # beside the job
    from transiesta import scf_words as _scf_words, setup_words as _setup_words
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
    from ...engine_atom_index import convert_from_engine_numbering
except ImportError:                         # beside a job, as the monitor's
    import siesta_grammar as _G
    from _section_rules import (
        CONTINUE, END_BUBBLE, END_SECTION,
        SectionRule, any_of, compile_rules, contains_ci, matches_regex_ci,
        starts_with_ci,
    )
    from molwatch_grammar import parse_runtime_line
    from engine_atom_index import convert_from_engine_numbering


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



#: HOW THE ECHO'S ATOM RANGES COUNT -- TranSIESTA's ``Position in
#: geometry 1 -- 9`` is SIESTA's numbers, from 1, of the deck's own atom
#: list: each the engine index + 1 (`model/overview.md` § 2.2b, F5).
TS_ECHO_ATOM_NUMBERS = ("SIESTA's numbers, from 1, of the deck's own atom "
                        "list (engine index + 1): the electrodes' atom "
                        "ranges as TranSIESTA prints them")

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
        #: max_force, max_force_constrained, charges, scf_history,
        #: elapsed_s, in_progress}`` -- coordinates as ``[element, x, y,
        #: z]`` rows; ``charges`` each partition's every-atom charges the
        #: step printed (`siesta_grammar.PARTIAL_CHARGES_BEGIN`), ``None``
        #: where it printed none.
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
        # TBTRANS'S ACCOUNT OF ITS LOOP (`model/parse.md` § 5d.5): its size,
        # the spin passes it says it will make, and each pass begun -- its
        # channel, TBtrans's estimate while it runs, its time once it
        # completed, its currents.  Empty for an output that is not TBtrans's.
        self.tbtrans: Dict[str, Any] = {}
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
        #: THE ATOMS THE RUN HELD, 1-based as the output echoes them
        #: (`siesta_grammar.CONSTRAINT_LINE`); empty for a run that held
        #: none.  Translated to the structure's indices by the parser.
        self.held: set = set()
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
        # SIESTA's two closing lines of a step's forces, printed both, always:
        # ``Max <val>`` over every atom, then ``Max <val> constrained`` over
        # the free atoms, those the run does not hold -- the one it compares against
        # ``MD.MaxForceTol`` (`siesta_grammar.MAX_FORCE`).
        self._step_max_force: Optional[float] = None
        self._step_max_force_constrained: Optional[float] = None
        self._step_forces: List[List[float]] = []
        # The step's stress, ``Stress tensor Voigt[...] (kbar)``, as printed.
        self._step_stress: Optional[List[float]] = None
        # The step's partial charges, by partition, every atom in the
        # deck's order -- and the block being read now.
        self._step_charges: Dict[str, List[float]] = {}
        self._charges_partition: Optional[str] = None
        # TRANSIESTA COMPUTING NO FORCES (`siesta_grammar.TS_FORCES_WRONG`):
        # the sentence every surface says in their place; then no step keeps
        # a force or a stress.
        self.forces_not_computed: Optional[str] = None
        # The ``siesta: Etot`` of the energy decomposition SIESTA prints after
        # the initial DM and before the first SCF cycle of each step: where
        # the SCF started, kept for display (``runtime_info["initial_etot"]``),
        # never a frame energy.
        self._step_initial_etot: Optional[float] = None
        #: The step SIESTA said it began last: ``(what a step is, its
        #: number)``, in its own words (`siesta_grammar.STEP_BEGIN`).
        self.step_begun: Optional[Tuple[str, int]] = None
        #: THE PART OF THE RUN UNDERWAY (`execution/run-reports.md` § 2.8):
        #: a row of `siesta_grammar.PARTS` once its opening line is read,
        #: until the SCF's next row speaks or the run ends; ``None`` while
        #: the SCF speaks, before anything does, and once the run has ended.
        self.part: Optional[str] = None
        self._active: Optional[SectionRule] = None
        self._in_input_echo = False
        self._line_no = 0
        #: THE LAST LINE THAT SAID SOMETHING, and the one before this line:
        #: ``die``'s reason is the line before its marker
        #: (`siesta_grammar.DIE_MARKER`).
        self._last_said: Optional[str] = None
        self._said_before: Optional[str] = None
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
        self._said_before = self._last_said
        if line.strip():
            self._last_said = line
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
        self._last_said = self._said_before = None
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
            self._step_stress = None
            self._step_charges = {}
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
        # A RUN THAT COMPUTES NO FORCES keeps none -- nor the stress, built
        # from the same energy-density matrix TranSIESTA set to zero.
        computed = self.forces_not_computed is None
        self.steps.append({
            "coords": self._step_frame,
            "forces": (self._step_forces or None) if computed else None,
            "energy": energy,
            "max_force": self._step_max_force if computed else None,
            "max_force_constrained": (self._step_max_force_constrained
                                      if computed else None),
            "charges": dict(self._step_charges) or None,
            "stress_voigt_kbar": self._step_stress if computed else None,
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
        self._step_stress = None
        self._step_charges = {}
        self._step_initial_etot = None
        self._current_scf = []
        self._prev_E_KS = None

    # ---- how the run ends ---------------------------------------------------

    def _on_end_of_run(self, line: str, line_no: int) -> None:
        # An abort already seen wins: SIESTA does not print both, but if it
        # somehow did, the abort is the load-bearing fact.
        if self.run_state not in ("stopped", "out_of_memory"):
            self.run_state = "ended"
        # NO PART IS UNDERWAY once the run has ended (§ 2.8).
        self.part = None
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
            # original cause -- except at the wall, where SIESTA prints the
            # SCF warning on its way out and the wall is the cause.
            # A `die` ends on its marker, its REASON the line before
            # (`siesta_grammar.DIE_MARKER`) -- said with what it may mean
            # (`siesta_grammar.stop_line`).
            if marker == _G.OUT_OF_TIME_MARKER:
                self.cause = marker
                self.error_message = _G.stop_line(line)
            elif self.error_message is None:
                said = (self._said_before if marker == _G.DIE_MARKER
                        and self._said_before is not None else line)
                self.error_message = (self._scf_not_conv_line
                                      or _G.stop_line(said))
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

    def _on_system_label(self, line: str, line_no: int) -> None:
        m = _G.SYSTEM_LABEL.match(line.strip())
        if m:
            self.runtime_info["system_label"] = m.group(1)

    def _consume_constraint(self, line: str, line_no: int) -> str:
        # The ranges line follows the constraint line, blank lines between
        # allowed; anything else ends the block and is read as itself.
        if not line.strip():
            return CONTINUE
        atoms = _G.constraint_atoms(line)
        if not atoms:
            return END_BUBBLE
        self.held.update(atoms)
        return END_SECTION

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
        # THE SECOND BLOCK'S ROWS carry ``siesta:`` before the atom's index:
        # the same forces (`siesta_grammar.FORCE_ROW_PREFIX`).
        parts = _G.FORCE_ROW_PREFIX.sub("", line).strip().split()
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

    def _on_charges_start(self, line: str, line_no: int) -> None:
        """``<Partition> Atomic Populations:`` -- a step's partial charges
        begin (`siesta_grammar.PARTIAL_CHARGES_BEGIN`)."""
        self._charges_partition = (
            _G.PARTIAL_CHARGES_BEGIN.match(line).group(1).lower())
        self._step_charges[self._charges_partition] = []

    def _consume_charges(self, line: str, line_no: int) -> str:
        """One row of the block: the atom's number, then its charge -- the
        first number after it (`partial_charges.F90`'s ``charge [q]``
        column); the header row is read past, and the dashed rule and the
        total end the block."""
        parts = line.split()
        if parts and parts[0].lower() == "atom":
            return CONTINUE
        if parts and parts[0].startswith("---"):
            return CONTINUE
        if parts and parts[0].lower() == "total":
            self._charges_partition = None
            return END_SECTION
        try:
            int(parts[0])
            q = _G.fortran_float(parts[1])
        except (ValueError, IndexError):
            self._charges_partition = None
            return END_BUBBLE
        self._step_charges[self._charges_partition].append(q)
        return CONTINUE

    def _on_stress(self, line: str, line_no: int) -> None:
        """``Stress tensor Voigt[x,y,z,yz,xz,xy] (kbar):`` and six numbers."""
        try:
            values = [_G.fortran_float(v) for v in
                      _G.STRESS_VOIGT.match(line).group(1).split()]
        except (ValueError, AttributeError) as exc:
            self._warn(line_no, line, f"stress line: malformed value: {exc}",
                       category="forces")
            return
        if len(values) == 6:
            self._step_stress = values

    def _on_ts_forces_wrong(self, line: str, line_no: int) -> None:
        """TranSIESTA says it computes no forces (`siesta_grammar.
        TS_FORCES_WRONG`)."""
        self.forces_not_computed = _G.FORCES_NOT_COMPUTED
        self.runtime_info["forces_not_computed"] = _G.FORCES_NOT_COMPUTED

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
        self.part = None
        # A NEW PHASE IS NOT A RESTART.  The periodic initialization's cycles
        # stay beside the NEGF loop's, and the new phase starts with its
        # convergence unanswered.
        if phase != self.phase:
            if self.phase is not None:
                self.scf_converged = None
            self.phase = phase
            self.phase_converged.setdefault(phase, None)
            self._prev_E_KS = None
        # iscf==1 starts a new SCF run of THIS phase -- and its convergence
        # is unanswered until it says (P-S2: the flag is the LAST block's).
        # Found 2026-10-18 on two runs the scheduler cut mid-block: the
        # block before had converged, and the flag kept its True.
        if iscf == 1:
            if (self._current_scf
                    and self._current_scf[-1].get("phase") == phase):
                self._current_scf = []
            # Whatever emptied the block list -- the step boundary does, before
            # this row -- the new block's verdict is its own, not yet said.
            self.scf_converged = None
            self.phase_converged[phase] = None
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
            # HOW ITS ATOM RANGES COUNT, said beside them: an engine's fact
            # named as the engine's (`model/overview.md` § 2.2a, F5).
            self.ts_info.setdefault("atom_numbers", TS_ECHO_ATOM_NUMBERS)
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
            self.ts_info.setdefault("electrodes", {}).setdefault(
                m.group(1), {})["gf_started"] = True

    def _on_ts_gf_file(self, line: str, line_no: int) -> None:
        """The lead's file: its estimated size, and its being written --
        the set-up's place (`transiesta.setup_words`)."""
        if self._ts_gf_electrode is None:
            return
        el = self.ts_info.setdefault("electrodes", {}).setdefault(
            self._ts_gf_electrode, {})
        m = _G.TS_GF_SIZE.match(line)
        if m:
            el["gf_file_size"] = f"{m.group(1)} {m.group(2)}"
            return
        if _G.TS_GF_DONE.match(line):
            el["gf_done"] = True

    def _on_ts_gf_reused(self, line: str, line_no: int) -> None:
        """A lead's file found and read instead of computed (``Using
        GF-file``): the lead named by the file's tail after ``TSGF``."""
        m = _G.TS_GF_REUSED.match(line)
        if not m or "TSGF" not in m.group(1):
            return
        lead = m.group(1).rsplit("TSGF", 1)[-1]
        self.ts_info.setdefault("electrodes", {}).setdefault(
            lead, {})["gf_reused"] = True

    def _part_begins(self, name: str):
        """The handler of a `siesta_grammar.PARTS` row's opening line: the
        part it names is underway."""
        def _begin(line: str, line_no: int) -> None:
            self.part = name
        return _begin

    def _where_words(self) -> str:
        """WHERE THE RUN IS, in words -- the one precedence (`model/parse.md`
        § 2b, ``doing``): the row in flight's phase and iteration
        (`transiesta.scf_words`), else TranSIESTA's set-up as it stands
        (`transiesta.setup_words`); ``""`` with neither -- unless a part of
        `siesta_grammar.PARTS` began after the SCF's last row: then that
        part, in the table's words, and said so where it prints nothing
        until it ends (`run-reports.md` § 2.8).  Read by `now()` for the
        monitor and by `finish()` for the ending, so the two say the same
        words."""
        part = _G.output_part(self.part)
        if part is not None:
            return part.words + (f" -- {part.note}" if part.note else "")
        # THE ROW IN FLIGHT, by `now()`'s own rule: the current SCF's last
        # row, else the last committed step's -- so a step boundary keeps
        # its words.
        last = (self._current_scf[-1] if self._current_scf else
                next((s["scf_history"][-1] for s in reversed(self.steps)
                      if s.get("scf_history")), None))
        if last is not None:
            words = _scf_words(last.get("phase"), last.get("cycle"))
            if words:
                return words
        # A TBTRANS OUTPUT states no SCF and computes no file: its leads'
        # cell check would read as set-up for its whole loop.  Its loop is
        # said by `tbtrans.loop_words` where a surface asks for it.
        if self.tbtrans:
            return ""
        return _setup_words(self.ts_info)

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

    # ---- TBtrans's loop -------------------------------------------------------

    def _tbt_pass(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """The pass underway -- begun now when none is: at its data file, or
        at its first estimate in a build that writes none.  Its channel is
        the one its file names, else the next the set-up names."""
        passes = self.tbtrans.setdefault("passes", [])
        if passes and "loop_elapsed_s" not in passes[-1]:
            return passes[-1]
        planned = self.tbtrans.get("channels") or ()
        if channel is None and len(passes) < len(planned):
            channel = planned[len(passes)]
        passes.append({"channel": channel} if channel else {})
        return passes[-1]

    def _on_tbt_setup(self, line: str, line_no: int) -> None:
        """The loop's size and the passes TBtrans says it will make."""
        for key, pat, cast in (("k_points", _G.TBT_KPOINTS, int),
                               ("k_method", _G.TBT_KMETHOD, str)):
            m = pat.match(line)
            if m:
                self.tbtrans.setdefault(key, cast(m.group(1)))
                return
        channels = _G.tbt_channels(line)
        if channels:
            self.tbtrans.setdefault("channels", list(channels))

    def _consume_tbt_contour(self, line: str, line_no: int) -> str:
        """``ts: >> TBtrans contour <<``'s lines: each contour's points,
        summed -- the section ends at its first line that is not ``ts:``."""
        if not _G.TS_ECHO.match(line):
            return END_BUBBLE
        m = _G.TBT_CONTOUR_POINTS.match(line)
        if m:
            self.tbtrans["energy_points"] = (
                self.tbtrans.get("energy_points", 0) + int(m.group(1)))
        return CONTINUE

    def _on_tbt_data_file(self, line: str, line_no: int) -> None:
        m = _G.TBT_DATA_FILE.match(line)
        channel = _G.tbt_channel_of(m.group(1)) if m else None
        p = self._tbt_pass(channel)
        if channel:
            p["channel"] = channel

    def _on_tbt_progress(self, line: str, line_no: int) -> None:
        """TBtrans's estimate for the pass underway: its first, ``Initial
        ETA`` -- nothing done yet -- and each ``Calculated N %, ETA in S s``
        after it.  A time that is not a number states none: TBtrans divides
        by the share done, so at ``TBT.Progress 0`` its line at 0 % says
        ``Infinity`` or ``NaN`` (`m_tbt_trik.F90`)."""
        m = _G.TBT_PROGRESS.match(line)
        if m:
            done, eta = m.group(1), m.group(2)
        else:
            m = _G.TBT_INITIAL_ETA.match(line)
            if m is None:
                return
            done, eta = "0", m.group(1)
        try:
            done_pct, eta_s = _G.fortran_float(done), _G.fortran_float(eta)
        except ValueError:
            return
        if not math.isfinite(done_pct):
            return
        p = self._tbt_pass()
        p["done_pct"] = done_pct
        if math.isfinite(eta_s):
            p["eta_s"] = eta_s
        else:
            p.pop("eta_s", None)

    def _on_tbt_completed(self, line: str, line_no: int) -> None:
        """A pass's end, with the time its loop took: its estimate ends
        with it."""
        m = _G.TBT_COMPLETED.match(line)
        try:
            took = _G.fortran_float(m.group(1)) if m else None
        except ValueError:
            took = None
        if took is None or not math.isfinite(took):
            return
        p = self._tbt_pass()
        p["loop_elapsed_s"] = took
        p.pop("done_pct", None)
        p.pop("eta_s", None)

    def _on_tbt_current(self, line: str, line_no: int) -> None:
        """A pass's currents, printed after its ``Completed in``: per
        electrode pair, the bias TBtrans applied with the current and the
        power it reports."""
        for pat, field in ((_G.TBT_CURRENT, "current_a"),
                           (_G.TBT_POWER, "power_w")):
            m = pat.match(line)
            if m is None:
                continue
            try:
                bias, value = (_G.fortran_float(m.group(3)),
                               _G.fortran_float(m.group(4)))
            except ValueError:
                return
            passes = self.tbtrans.get("passes") or [self._tbt_pass()]
            rows = passes[-1].setdefault("currents", [])
            row = next((r for r in rows if (r["from"], r["to"], r["voltage_v"])
                        == (m.group(1), m.group(2), bias)), None)
            if row is None:
                row = {"from": m.group(1), "to": m.group(2),
                       "voltage_v": bias}
                rows.append(row)
            row[field] = value
            return

    def _on_orbital_count(self, line: str, line_no: int) -> None:
        """The run's atom and orbital counts, as the output states them."""
        m = _G.ORBITAL_COUNT.match(line)
        if m:
            self.runtime_info["n_atoms_stated"] = int(m.group(1))
            self.runtime_info["n_orbitals"] = int(m.group(2))

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
            # A STEP'S PARTIAL CHARGES, each partition's block (`model/
            # parse.md` § 4a; `engines/vibration.md` § 5.10 ②).
            SectionRule(name="partial_charges",
                        aliases=["Hirshfeld|Voronoi Atomic Populations:"],
                        start=matches_regex_ci(
                            _G.PARTIAL_CHARGES_BEGIN.pattern),
                        on_start=self._on_charges_start,
                        consume=self._consume_charges),
            # THE SETUP LINES: the SystemLabel the run read, and each held
            # block -- the constraint line, then its ranges line (the one
            # reader of both, `model/parse.md` § 4a).
            SectionRule(name="system_label",
                        aliases=["reinit: System Label:"],
                        start=matches_regex_ci(_G.SYSTEM_LABEL.pattern),
                        on_start=self._on_system_label),
            SectionRule(name="constraint",
                        aliases=["siesta: Constraint (N): pos"],
                        start=matches_regex_ci(_G.CONSTRAINT_LINE.pattern),
                        consume=self._consume_constraint),
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
            # TRANSIESTA COMPUTING NO FORCES, printed after the echo's frame
            # closes -- which the general ``ts:`` echo rule below would take
            # and set aside.
            SectionRule(name="ts_forces_wrong",
                        aliases=["ts: *** ALL FORCES AFTER TRANSIESTA HAS "
                                 "RUN ARE WRONG ***"],
                        start=matches_regex_ci(_G.TS_FORCES_WRONG.pattern),
                        on_start=self._on_ts_forces_wrong),
            # TBtrans's energies: its contour section, which the general
            # ``ts:`` echo rule below would take and set aside.
            SectionRule(name="tbt_contour",
                        aliases=["ts: >> TBtrans contour <<"],
                        start=matches_regex_ci(_G.TBT_CONTOUR_BEGIN.pattern),
                        consume=self._consume_tbt_contour),
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
            SectionRule(name="ts_gf_file",
                        aliases=["Estimated file size: ...",
                                 "Done creating '...'."],
                        start=any_of(matches_regex_ci(_G.TS_GF_SIZE.pattern),
                                     matches_regex_ci(_G.TS_GF_DONE.pattern)),
                        on_start=self._on_ts_gf_file),
            SectionRule(name="ts_gf_reused",
                        aliases=["Using GF-file '...'"],
                        start=matches_regex_ci(_G.TS_GF_REUSED.pattern),
                        on_start=self._on_ts_gf_reused),
            SectionRule(name="orbital_count",
                        aliases=["initatomlists: Number of atoms, orbitals, "
                                 "and projectors: ..."],
                        start=matches_regex_ci(_G.ORBITAL_COUNT.pattern),
                        on_start=self._on_orbital_count),
            SectionRule(name="emadel", aliases=["siesta: Emadel = ..."],
                        start=matches_regex_ci(_G.EMADEL.pattern),
                        on_start=self._on_emadel),
            # TBtrans's loop (`model/parse.md` § 5d.5).
            SectionRule(name="tbt_setup",
                        aliases=["tbt: Number of transport k-points = N",
                                 "tbt: Method = ...",
                                 "tbt: Calculate spin ...",
                                 "tbt: Non-polarized Hamiltonian"],
                        start=any_of(
                            matches_regex_ci(_G.TBT_KPOINTS.pattern),
                            matches_regex_ci(_G.TBT_KMETHOD.pattern),
                            matches_regex_ci(_G.TBT_SPIN.pattern)),
                        on_start=self._on_tbt_setup),
            SectionRule(name="tbt_data_file",
                        aliases=["tbt: Initializing data file: ..."],
                        start=matches_regex_ci(_G.TBT_DATA_FILE.pattern),
                        on_start=self._on_tbt_data_file),
            SectionRule(name="tbt_progress",
                        aliases=["tbt: Initial ETA in ...",
                                 "tbt: Calculated N %, ETA in ..."],
                        start=any_of(
                            matches_regex_ci(_G.TBT_INITIAL_ETA.pattern),
                            matches_regex_ci(_G.TBT_PROGRESS.pattern)),
                        on_start=self._on_tbt_progress),
            SectionRule(name="tbt_completed", aliases=["tbt: Completed in ..."],
                        start=matches_regex_ci(_G.TBT_COMPLETED.pattern),
                        on_start=self._on_tbt_completed),
            SectionRule(name="tbt_current",
                        aliases=["L -> R, V [V] / I [A]: ...",
                                 "L -> R, V [V] / P [W]: ..."],
                        start=any_of(
                            matches_regex_ci(_G.TBT_CURRENT.pattern),
                            matches_regex_ci(_G.TBT_POWER.pattern)),
                        on_start=self._on_tbt_current),
            # THE PARTS THAT BEGIN WITH A LINE OF THEIR OWN, from the one
            # table (`siesta_grammar.PARTS`, `run-reports.md` § 2.8).
            *[SectionRule(name=f"part_{p.name}", aliases=[p.begins],
                          start=contains_ci(p.begins),
                          on_start=self._part_begins(p.name))
              for p in _G.PARTS],
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
            SectionRule(name="stress_voigt",
                        aliases=["Stress tensor Voigt[x,y,z,yz,xz,xy] (kbar):"],
                        start=matches_regex_ci(_G.STRESS_VOIGT.pattern),
                        on_start=self._on_stress),
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
        step_kind, steps_done, step_moves, max_force, max_force_all,
        stress_voigt_kbar,
        forces_not_computed, criteria, targets, tbtrans, done_pct, eta_s,
        doing, part, scf_converged}``,
        each present only when the
        output states it: the
        current SCF row's values (a NEGF row's charge among them), and as
        ``residuals`` each beside the criterion its phase states --
        ``{name: (value, tolerance or None, unit)}``; the step SIESTA last
        began, its number and its words for what a step is, and so how many
        steps are done, and whether that kind of step moves the geometry
        (`siesta_grammar.MOVES_GEOMETRY`); the latest largest force on the
        free atoms
        and over every atom, and the latest stress -- or, for a run that
        computes no forces, the sentence in their place -- and what each
        must reach;
        TBtrans's account of its loop, a copy, and the pass underway's share
        done and the time TBtrans says it has left; where the run is, in
        words, and a part of the output underway by name; the last SCF's
        verdict.
        """
        out: Dict[str, Any] = {}
        cycle = (self._current_scf[-1] if self._current_scf else
                 next((s["scf_history"][-1] for s in reversed(self.steps)
                       if s.get("scf_history")), None))
        if cycle is not None:
            out["phase"] = cycle.get("phase")
            out["cycle"] = cycle.get("cycle")
            for key in ("energy",) + tuple(
                    r["key"] for r in _G.SCF_RESIDUALS):
                v = cycle.get(key)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    out[key] = v
        if self.step_begun is not None:
            out["step_kind"], out["step"] = self.step_begun
            # SIESTA begins step N once steps 0..N-1 are done.
            out["steps_done"] = self.step_begun[1]
            out["step_moves"] = bool(
                _G.MOVES_GEOMETRY.search(self.step_begun[0]))
        out.update(self._forces_now())
        crit = self.criteria()
        if crit:
            out["criteria"] = crit
        mine = crit.get(out.get("phase") or "", {})
        residuals = {r["name"]: (out[r["key"]],
                                 mine.get(r["criterion"], {}).get("tolerance"),
                                 r["unit"])
                     for r in _G.SCF_RESIDUALS if r["key"] in out}
        if residuals:
            out["residuals"] = residuals
        targets = {k: v for k, v in
                   self.runtime_info.get("convergence_targets", {}).items()
                   if k != "source"}
        if targets:
            out["targets"] = targets
        if self.tbtrans:
            out["tbtrans"] = copy.deepcopy(self.tbtrans)
            # THE PASS UNDERWAY'S estimate -- a completed pass states none.
            passes = self.tbtrans.get("passes") or []
            if passes:
                out.update({k: passes[-1][k] for k in ("done_pct", "eta_s")
                            if k in passes[-1]})
        if self.ts_info:
            # TRANSIESTA'S SET-UP, as it stands (`transiesta.setup_words`).
            out["transiesta"] = copy.deepcopy(self.ts_info)
        # WHERE THE RUN IS, in words -- said only when the output states
        # it (`_where_words`); what the monitor's summary line says.
        where = self._where_words()
        if where:
            out["doing"] = where
        # THE PART UNDERWAY, by name -- its sentence is ``doing`` -- and the
        # last SCF's verdict: what the monitor says an ended SCF's facts by
        # (`run-reports.md` § 2.8).
        if _G.output_part(self.part) is not None:
            out["part"] = self.part
        if self.scf_converged is not None:
            out["scf_converged"] = self.scf_converged
        return out

    def _forces_now(self) -> Dict[str, Any]:
        """The latest forces and stress the output states -- the step in
        flight's, else the last step that stated them: ``max_force``, the
        free atoms' largest component, ``max_force_all`` every atom's
        beside it, ``stress_voigt_kbar``; or, for a run that computes none,
        ``forces_not_computed``, the sentence in their place
        (`model/parse.md` § 4a).  :meth:`now` and :meth:`finish` both
        answer with it."""
        if self.forces_not_computed is not None:
            return {"forces_not_computed": self.forces_not_computed}
        # ONE STEP'S, all three: a step in flight that has printed its forces
        # and not yet its stress states no stress.
        free, every, stress = (self._step_max_force_constrained,
                                 self._step_max_force, self._step_stress)
        if (free, every, stress) == (None, None, None):
            stated = next((s for s in reversed(self.steps)
                           if s["max_force"] is not None
                           or s["max_force_constrained"] is not None
                           or s["stress_voigt_kbar"]), None)
            if stated is not None:
                free = stated["max_force_constrained"]
                every = stated["max_force"]
                stress = stated["stress_voigt_kbar"]
        out: Dict[str, Any] = {}
        if free is not None:
            out["max_force"] = free
            if every is not None:
                out["max_force_all"] = every
        elif every is not None:
            out["max_force"] = every
        if stress is not None:
            out["stress_voigt_kbar"] = list(stress)
        return out

    def finish(self) -> Dict[str, Any]:
        """The end of the output: the torn step dropped, the step in flight
        committed and flagged, the run's facts summed.  Returns the reading
        -- ``{steps, live_scf, lattice, run_state, scf_converged, phases,
        relaxed, cause, error_message, runtime_info, held, warnings}`` --
        and the reader is done.  ``held`` is the 1-based atoms the output
        says the run held; ``runtime_info["system_label"]`` the label it
        read.  The ending among them is THE answer to how a SIESTA
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
        # WHERE IT IS, read before the step in flight is committed (which
        # empties the rows): the ending's `doing` (`model/parse.md` § 2b).
        doing = self._where_words() or None
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
        # WHAT EACH SCF ROW MEASURES, the engine's one declaration.
        self.runtime_info["scf_residuals"] = [dict(r) for r in
                                              _G.SCF_RESIDUALS]
        # THE ATOMS THE RUN HELD, as its own .out echoes them
        # (`siesta_grammar.CONSTRAINT_LINE`), 1-based as SIESTA prints --
        # stated here once for every reader of the pass (the registry's
        # parser, a job's relaxation record), translated through the
        # engine index API, never a bare n - 1.
        if self.held:
            self.runtime_info["frozen_atoms"] = sorted(
                convert_from_engine_numbering(n, "siesta") for n in self.held)
        return {"steps": self.steps, "live_scf": live_scf, "doing": doing,
                "lattice": self.lattice, "run_state": self.run_state,
                "scf_converged": self.scf_converged,
                "phases": dict(self.phase_converged),
                "relaxed": self.relaxed, "cause": self.cause,
                "error_message": self.error_message,
                "runtime_info": self.runtime_info,
                "held": sorted(self.held),
                "warnings": self.warnings, **self._forces_now()}


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
