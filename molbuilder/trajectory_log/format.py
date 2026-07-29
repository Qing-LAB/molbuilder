"""Tiny helper for writing the initial-state preview block of a
``<job>.molwatch.log``.

The block format matches the spec in ``docs/engines/pyscf.md``
(and the corresponding entry in molwatch's ``docs/model/parse.md``).
This module exists so non-runtime molbuilder code paths -- the SIESTA
FDF generator, future engine integrations -- can drop a one-block
preview file alongside their main output, giving molwatch something
to render the moment a user loads it (no waiting for the engine to
produce its first frame).

The PySCF-side emitter is generated inline into the script (see
``molbuilder/pyscf/input.py``: ``_emit_molwatch_emitter``) and writes
to the same format with the same structural keys.  Keeping the two
emitters in sync is a spec contract, not an import dependency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..structure import Structure


def molwatch_log_basename(system_label: str,
                          stage: Optional[int] = None) -> str:
    """Resolve the canonical ``<basename>[.stage<N>].molwatch.log``
    filename for a given system label + optional stage marker.

    Per ``docs/execution/job-contracts.md``: when the user is running a staged
    coarse->medium->tight relaxation, each stage gets a uniquely-named
    log file in the same directory (the basename stays identical so
    SIESTA's restart files transfer; only the molwatch-log filename
    grows the suffix).  ``stage=None`` returns the unsuffixed name.

    Returns just the filename, not a full path; callers join with the
    target directory.
    """
    if stage is None:
        return f"{system_label}.molwatch.log"
    return f"{system_label}-stage{int(stage)}.molwatch.log"


def write_initial_preview(
    struct: "Structure",
    path: str | Path,
    *,
    job: str,
    engine: str,
    generator: str = "molbuilder",
    stage_name: Optional[str] = None,
    convergence_targets: Optional[Mapping[str, float]] = None,
) -> None:
    """Write a ``<path>.molwatch.log`` containing exactly one preview
    block: step 0 with the structure's coordinates, no energy, no
    forces, no SCF history.

    The file is intended to be loaded into molwatch the moment the
    user wants to see what they're about to run -- before the engine
    has produced any of its native output.

    Parameters
    ----------
    struct : Structure
        The geometry to preview.  Read for elements + positions; not
        mutated.
    path : str | Path
        Output filename.  Existing files are overwritten.
    job : str
        Job name; goes into the ``# job:`` header line.
    engine : str
        Engine name (``"siesta"``, ``"pyscf"``, ...) for the
        ``# engine:`` header.  Determines what the molwatch UI uses
        for its cosmetic engine label.
    generator : str
        Tool that wrote the file; goes into the ``# generator:``
        header.  Default ``"molbuilder"``.
    stage_name : str, optional
        Stage label for multi-stage SIESTA / PySCF runs (e.g.
        ``"stage1"``, ``"stage2"``).  Emitted as ``# stage: <name>``
        when set; absent header line means "single-stage run".
    convergence_targets : Mapping[str, float], optional
        Per-target thresholds for the stage's convergence (e.g.
        ``{"max_force_ev_per_ang": 0.05, "max_steps": 600}``).  Each
        entry is emitted as ``# convergence.<key>: <value>``.  When
        absent, no convergence-target header lines are emitted (the
        molwatch inspector renders only the engine's defaults).
        Used by the watch-tab live plot to draw the right horizontal
        threshold line per stage on the max-force-vs-step trace.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    elements = list(struct.elements)
    positions = struct.positions  # (N, 3) array-like in Angstrom
    n = len(elements)
    now_epoch = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S",
                              time.localtime(now_epoch))
    lines: list[str] = [
        "# molwatch trajectory log v1",
        f"# generator: {generator}",
        f"# engine: {engine}",
        f"# job: {job}",
        "# units: energy=eV, force=eV/Ang, coords=Ang",
        f"# created: {timestamp}",
    ]
    # Optional stage label (multi-stage runs).  Absent header line
    # means "single-stage run"; the inspector reads this to title
    # the watch-tab panel correctly.
    if stage_name:
        lines.append(f"# stage: {stage_name}")
    # Optional convergence-target dict.  One header line per key
    # under the ``convergence.`` namespace so the inspector can
    # consume them generically without an engine switch.
    if convergence_targets:
        for k, v in convergence_targets.items():
            # Reject keys with whitespace / newlines that would break
            # the line-oriented parser the watch tab uses.  ``str(k)``
            # tolerates non-str types (rare).
            key = str(k).strip()
            if not key or any(c.isspace() for c in key):
                raise ValueError(
                    f"convergence_targets key {k!r} cannot contain "
                    f"whitespace; got {key!r}"
                )
            lines.append(f"# convergence.{key}: {v}")
    lines += [
        "",
        "==== molwatch step 0 begin ====",
        "step_index: 0",
        "kind: initial_preview",
        # Unix epoch seconds.  Used by the watch UI to render
        # "Started 2h 15m ago, last frame 30s ago" without timezone
        # gymnastics.  Same units as `time.time()` everywhere.
        f"wall_time: {now_epoch:.3f}",
        f"n_atoms: {n}",
        "coordinates (Ang):",
    ]
    for i, el in enumerate(elements):
        x, y, z = positions[i]
        lines.append(f"   {el:<2s}  {float(x):14.8f}  {float(y):14.8f}  {float(z):14.8f}")
    lines += [
        "energy (eV): None",
        "forces (eV/Ang):",
        "max_force (eV/Ang): None",
        "scf_history begin",
        "scf_history end",
        "==== molwatch step 0 end ====",
        "",
    ]
    p.write_text("\n".join(lines))


__all__ = ["write_initial_preview"]
