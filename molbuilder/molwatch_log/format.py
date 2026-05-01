"""Tiny helper for writing the initial-state preview block of a
``<job>.molwatch.log``.

The block format matches the spec in ``docs/spec/pyscf-script.md``
(and the corresponding entry in molwatch's ``docs/spec/parsers.md``).
This module exists so non-runtime molbuilder code paths -- the SIESTA
FDF generator, future engine integrations -- can drop a one-block
preview file alongside their main output, giving molwatch something
to render the moment a user loads it (no waiting for the engine to
produce its first frame).

The PySCF-side emitter is generated inline into the script (see
``molbuilder/pyscf_input.py``: ``_emit_molwatch_emitter``) and writes
to the same format with the same structural keys.  Keeping the two
emitters in sync is a spec contract, not an import dependency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..structure import Structure


def write_initial_preview(
    struct: "Structure",
    path: str | Path,
    *,
    job: str,
    engine: str,
    generator: str = "molbuilder",
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
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    elements = list(struct.elements)
    positions = struct.positions  # (N, 3) array-like in Angstrom
    n = len(elements)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines: list[str] = [
        "# molwatch trajectory log v1",
        f"# generator: {generator}",
        f"# engine: {engine}",
        f"# job: {job}",
        "# units: energy=eV, force=eV/Ang, coords=Ang",
        f"# created: {timestamp}",
        "",
        "==== molwatch step 0 begin ====",
        "step_index: 0",
        "kind: initial_preview",
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
