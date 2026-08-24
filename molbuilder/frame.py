"""Frame and Trajectory -- parser output types.

`Structure` (in molbuilder/structure.py) carries one geometric
configuration: elements, positions, and PDB metadata.  It's the
*build-side* type.

`Frame` is the *parse-side* type.  It wraps a Structure and adds the
per-step physics that comes out of a calculation: total energy, atomic
forces, the geom-opt / MD step index, and the per-cycle SCF history
for that step.

`Trajectory` is a thin wrapper holding `(source_format, frames,
lattice)` -- the format-level metadata that doesn't fit on any single
frame.  Parsers' `parse(path)` returns a Trajectory.

The molwatch unified-log parser surfaces a
`source_format` from the FILE's `# engine:` header -- it can differ
from the parser class's `cls.name` -- so the parser interface needs
*something* that carries that string alongside the frames.  The
minimal Trajectory below resolves that need without committing to a
richer trajectory type now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .structure import Structure


@dataclass
class ParseWarning:
    """One non-fatal parser issue.

    Parsers accumulate these in ``Trajectory.parse_warnings`` instead
    of raising mid-parse, so a single malformed line doesn't blank the
    whole results view.  The Results tab surfaces them in a
    collapsible "parsing notes" panel; the user can spot a known-
    benign warning (a tight-packed spin-polarized SCF line) or report
    a genuinely new one.

    Fields:

      line_no   1-based line number in the source file.
      snippet   The offending line, truncated to ~120 chars.
      error     Short, actionable description -- "could not parse
                floats: invalid literal …" or "SCF line has 9
                columns, expected 7 or 8".  Render verbatim.
      category  Free-text classifier ("scf", "outcoor", "forces",
                "runtime", ...).  The UI can group / colour-code by
                category; default "scf" is the historically most
                common site.
    """
    line_no:  int
    snippet:  str
    error:    str
    category: str = "scf"


@dataclass
class Frame:
    """One geom-opt / MD step's geometry + physics.

    Field-by-field:
      structure    -- the geometry of this step (elements + positions
                      + PDB metadata).  Built by the parser from the
                      raw atom rows in the trajectory.
      step_index   -- 0-based step number.  Step 0 is conventionally
                      the initial-state preview emitted by molbuilder
                      before the engine runs.
      energy       -- total electronic energy in eV, or None if the
                      engine hasn't reported one yet for this step.
      forces       -- per-atom force array, shape (N, 3), eV/A;
                      None when the parser couldn't extract forces
                      (e.g. geomeTRIC trajectories carry no forces).
      max_force    -- max per-atom force magnitude in eV/A across
                      ALL atoms (including any frozen ones), or
                      None.  Spec convention: max_i |F_i| (NOT
                      max(|F_component|)).
      max_force_constrained
                   -- max per-atom force magnitude in eV/A EXCLUDING
                      atoms that are constrained / frozen.  None
                      when the run has no constraints (SIESTA only
                      emits the line when at least one atom is
                      constrained) OR the parser couldn't extract
                      it.  This is the value SIESTA compares
                      against ``MD.MaxForceTol`` for relaxation
                      convergence — when constraints exist, this
                      is the meaningful "did we converge?" signal,
                      not ``max_force`` (which can stay high on a
                      frozen atom forever).
      lattice      -- (3, 3) Ang lattice vectors *for this frame*, or
                      None.  Today every parser sets None here and
                      puts the (constant) cell on Trajectory.lattice
                      instead; per-frame lattice is reserved for
                      variable-cell MD that no current parser
                      produces.
      scf_history  -- list of per-cycle dicts with at least the keys
                      `cycle`, `energy`, `delta_E`.  Engine-specific
                      residual keys: PySCF / molwatch_log use
                      `gnorm` / `ddm`; SIESTA uses `dHmax` / `dDmax`.
                      Consumers must not assume a fixed key set.
                      None when the parser couldn't find SCF data
                      (e.g. PySCF .log absent).
      wall_clock_s -- Absolute Unix epoch seconds when the engine
                      wrote this step.  Answers "at what time?", and
                      is the ONLY field a consumer may render as a
                      date (parse.md § 2a, P-T1).  Only a parser that
                      read a real clock reading out of the file may
                      fill it: the molwatch emitter stamps its own
                      ``time.time()``, so .molwatch.log has one.  A
                      SIESTA .out carries no time-of-day anywhere, so
                      it stays None -- that is an ANSWER ("this engine
                      cannot say"), not missing data, and it is what
                      lets the watch UI fall back to the file's mtime
                      deliberately instead of formatting a duration as
                      a date.  Never derived from ``elapsed_s``: the
                      file does not contain the missing addend.
      elapsed_s    -- Seconds since the run began.  Answers "how far
                      in?", and is the ONLY field a consumer may
                      render as a duration.  SIESTA fills it from its
                      ``timer: ... IterSCF`` lines.  Parsers whose
                      engine reports epochs leave it None and let
                      ``to_legacy_payload`` derive it from the epoch
                      series -- one derivation, one home (P-T3).
                      Both stay None for formats that surface no time
                      at all (geomeTRIC's _geom_optim.xyz).  Together
                      these drive the watch UI's "Started 2h 15m ago,
                      last step 30s ago" -- the latency-of-progress
                      signal a researcher actually wants when staring
                      at a long run.
      in_progress  -- True when this Frame represents a calculation
                      mid-flight rather than a completed geometry step
                      with a real outcoor block.  Set by parsers when
                      they emit a synthetic frame for the very first
                      SCF cycle (so the user can watch the residual
                      drop in real time instead of waiting for the
                      first outcoor block to land 5-30 min later).
                      Consumers should hide trajectory animation
                      controls and show a "SCF in progress" banner
                      when True; the SCF convergence chart is the
                      only meaningful display.  Always False after a
                      real outcoor block has been committed.
    """
    structure:    Structure
    step_index:   int
    energy:       Optional[float]                 = None
    forces:       Optional[np.ndarray]            = None
    max_force:    Optional[float]                 = None
    max_force_constrained: Optional[float]        = None
    lattice:      Optional[np.ndarray]            = None
    scf_history:  Optional[List[Dict[str, float]]] = None
    wall_clock_s: Optional[float]                 = None
    elapsed_s:    Optional[float]                 = None
    in_progress:  bool                            = False

    def __post_init__(self) -> None:
        # Be tolerant about input -- parsers may pass plain lists for
        # forces / lattice; coerce to np.ndarray here so downstream
        # code can rely on .shape and .tolist().
        if self.forces is not None and not isinstance(self.forces, np.ndarray):
            self.forces = np.asarray(self.forces, dtype=float).reshape(-1, 3)
        if self.lattice is not None and not isinstance(self.lattice, np.ndarray):
            self.lattice = np.asarray(self.lattice, dtype=float).reshape(3, 3)


@dataclass
class Trajectory:
    """A list of Frames plus format-level metadata.

      source_format -- which engine / format this run came from
                       ("siesta", "pyscf", "molwatch", ...).  For
                       engine-native parsers this is the parser
                       class's `cls.name`; the molwatch unified-log
                       parser overrides it with the `# engine:`
                       header value so a SIESTA run logged via
                       .molwatch.log retains source_format="siesta".
      frames        -- the frames, in step order.
      lattice       -- (3, 3) Ang shared lattice, or None for vacuum.
                       Today this is where every parser stores the
                       cell; per-frame lattice (Frame.lattice) is
                       reserved for variable-cell trajectories that
                       no current parser produces.
      run_state     -- "finished" | "ongoing" | "error".  Authoritative
                       when the writer emitted explicit end-of-run
                       markers (`# concluded:` / `# error:` in
                       .molwatch.log; `>> End of run` in SIESTA's .out).
                       Defaults to "ongoing" when no marker found --
                       better to under-claim than to misclassify a
                       slow run as stalled.  Long iteration times
                       (some DFT steps take hours) make any stall
                       heuristic unreliable, so we go marker-only.
      error_message -- one-line error description when run_state ==
                       "error", else None.
    """
    source_format: str
    frames:        List[Frame]
    lattice:       Optional[np.ndarray] = None
    run_state:     str                  = "ongoing"
    error_message: Optional[str]        = None
    # CPU/GPU/host facts the generator captured at run start.  Parsers
    # populate this from on-disk metadata (``# runtime.<key>:`` lines
    # in molwatch logs; future SIESTA / other parsers may grow their
    # own header readers).  Empty dict when the writer didn't emit
    # the block -- older log files render with "—" rows.  Canonical
    # keys: see :mod:`molbuilder.runtime_info`.
    runtime_info:  dict                 = field(default_factory=dict)
    # Non-fatal parse issues encountered while reading the file.  The
    # parser appends to this list whenever a line fails to match its
    # expected shape; the whole parse continues.  Empty list = no
    # issues.  See :class:`ParseWarning` for shape.
    parse_warnings: List[ParseWarning]  = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.lattice is not None and not isinstance(self.lattice, np.ndarray):
            self.lattice = np.asarray(self.lattice, dtype=float).reshape(3, 3)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]
