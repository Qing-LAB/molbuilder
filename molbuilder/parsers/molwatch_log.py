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
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..frame import Frame, Trajectory
from ..structure import Structure
from .base import TrajectoryParser


_BEGIN_RE  = re.compile(r"====\s*molwatch\s+step\s+(\d+)\s+begin\s*====")
_END_RE    = re.compile(r"====\s*molwatch\s+step\s+(\d+)\s+end\s*====")
_HEADER_RE = re.compile(r"^#\s*molwatch\s+trajectory\s+log", re.IGNORECASE)
_ENGINE_RE = re.compile(r"^#\s*engine:\s*(\S+)", re.IGNORECASE)
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

        # In-block accumulators; commit only on a matching `end` marker.
        in_block = False
        block_idx: Optional[int] = None
        block_frame: List[List[Any]] = []
        block_energy: Optional[float] = None
        block_forces: List[List[float]] = []
        block_max_force: Optional[float] = None
        block_scf: List[Dict[str, Any]] = []
        block_wall_time: Optional[float] = None
        # Sub-states inside a block:
        sub = "scan"   # "scan" | "in_coords" | "in_forces" | "in_scf"

        def _reset_block() -> None:
            nonlocal in_block, block_idx, block_frame, block_energy
            nonlocal block_forces, block_max_force, block_scf, sub
            nonlocal block_wall_time
            in_block        = False
            block_idx       = None
            block_frame     = []
            block_energy    = None
            block_forces    = []
            block_max_force = None
            block_scf       = []
            block_wall_time = None
            sub             = "scan"

        with open(path, "r", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                stripped = line.strip()

                # ---- header / footer lines (only meaningful outside a block)
                if not in_block:
                    m_err = _ERROR_RE.match(line)
                    if m_err:
                        # Error has priority over concluded; the writer
                        # emits "# error:" then "# concluded:" so a clean
                        # parse of both should land on "error".
                        error_message = m_err.group(1).strip()
                        run_state = "error"
                        continue
                    m_done = _CONCLUDED_RE.match(line)
                    if m_done:
                        if run_state != "error":
                            run_state = "finished"
                        continue
                    m_eng = _ENGINE_RE.match(line)
                    if m_eng:
                        engine = m_eng.group(1)
                        continue

                # ---- block start ----
                m_begin = _BEGIN_RE.search(line)
                if m_begin:
                    # Any half-built previous block is silently abandoned.
                    _reset_block()
                    in_block  = True
                    block_idx = int(m_begin.group(1))
                    sub       = "scan"
                    continue

                # ---- block end -- commit ----
                m_end = _END_RE.search(line)
                if m_end and in_block:
                    if block_frame:
                        elements  = [row[0] for row in block_frame]
                        positions = np.array([row[1:4] for row in block_frame],
                                             dtype=float)
                        struct = Structure(elements=elements,
                                           positions=positions)
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
                            # Always a list (possibly empty) -- the
                            # .molwatch.log format always carries an
                            # scf_history block per step, even if it's
                            # empty (e.g. for an initial-state preview
                            # block).  None is reserved for parsers
                            # that genuinely have no SCF data source.
                            scf_history = list(block_scf),
                            wall_time   = block_wall_time,
                        ))
                    _reset_block()
                    continue

                if not in_block:
                    continue

                # ---- inside a block ----
                if sub == "in_coords":
                    if not stripped or ":" in stripped:
                        sub = "scan"
                        # fall through so this same line gets re-examined
                    else:
                        parts = stripped.split()
                        if len(parts) >= 4:
                            try:
                                x = float(parts[1])
                                y = float(parts[2])
                                z = float(parts[3])
                            except ValueError:
                                sub = "scan"
                            else:
                                block_frame.append([parts[0], x, y, z])
                                continue
                        else:
                            sub = "scan"

                if sub == "in_forces":
                    if not stripped or ":" in stripped:
                        sub = "scan"
                    else:
                        parts = stripped.split()
                        if len(parts) >= 4:
                            try:
                                fx = float(parts[1])
                                fy = float(parts[2])
                                fz = float(parts[3])
                            except ValueError:
                                sub = "scan"
                            else:
                                block_forces.append([fx, fy, fz])
                                continue
                        else:
                            sub = "scan"

                if sub == "in_scf":
                    # End sub-state on the explicit marker; otherwise parse
                    # the row.  Skip comment lines and blank lines.
                    if stripped.startswith("scf_history end"):
                        sub = "scan"
                        continue
                    if not stripped or stripped.startswith("#"):
                        continue
                    parts = stripped.split()
                    if len(parts) >= 5:
                        try:
                            cycle = int(parts[0])
                            energy = float(parts[1])
                            delta_E = float(parts[2])
                        except ValueError:
                            continue
                        gnorm = _maybe_float(parts[3])
                        ddm   = _maybe_float(parts[4])
                        block_scf.append({
                            "cycle":   cycle,
                            "energy":  energy,
                            "delta_E": delta_E,
                            "gnorm":   gnorm,
                            "ddm":     ddm,
                        })
                    continue

                # ---- scan: pick up section headers + scalar key:value lines
                if stripped.startswith("coordinates"):
                    sub = "in_coords"
                    continue
                if stripped.startswith("forces"):
                    sub = "in_forces"
                    continue
                if stripped.startswith("scf_history begin"):
                    sub = "in_scf"
                    continue
                if stripped.startswith("energy (eV):"):
                    block_energy = _maybe_float(stripped.split(":", 1)[1].strip())
                    continue
                if stripped.startswith("max_force (eV/Ang):"):
                    block_max_force = _maybe_float(
                        stripped.split(":", 1)[1].strip()
                    )
                    continue
                if stripped.startswith("wall_time:"):
                    # Unix epoch seconds emitted by both the SIESTA-side
                    # write_initial_preview helper and the inlined PySCF
                    # _MolwatchEmitter.  Optional -- older logs (and a
                    # log torn before the wall_time line) parse fine
                    # with block_wall_time = None, just no elapsed-time
                    # display in the UI.
                    block_wall_time = _maybe_float(
                        stripped.split(":", 1)[1].strip()
                    )
                    continue
                # step_index: / n_atoms: are informational; we don't need to
                # parse them (frame index is taken from the begin marker).

        # Torn final block at EOF: drop it (in_block True, no `end` seen).

        return Trajectory(
            source_format = engine,
            frames        = frames,
            lattice       = None,
            run_state     = run_state,
            error_message = error_message,
        )
