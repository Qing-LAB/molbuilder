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

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .structure import Structure


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
      max_force    -- max per-atom force magnitude in eV/A, or None.
                      Spec convention: max_i |F_i| (NOT
                      max(|F_component|)).
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
      wall_time    -- Unix epoch seconds when the engine wrote this
                      step.  None for parsers / formats that don't
                      surface a per-step timestamp (geomeTRIC's
                      _geom_optim.xyz, SIESTA's run.out without the
                      molwatch.log sibling).  Used by the watch UI to
                      show "Started 2h 15m ago, last step 30s ago" --
                      the latency-of-progress signal a researcher
                      actually wants when staring at a long run.
    """
    structure:    Structure
    step_index:   int
    energy:       Optional[float]                 = None
    forces:       Optional[np.ndarray]            = None
    max_force:    Optional[float]                 = None
    lattice:      Optional[np.ndarray]            = None
    scf_history:  Optional[List[Dict[str, float]]] = None
    wall_time:    Optional[float]                 = None

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

    def __post_init__(self) -> None:
        if self.lattice is not None and not isinstance(self.lattice, np.ndarray):
            self.lattice = np.asarray(self.lattice, dtype=float).reshape(3, 3)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]
