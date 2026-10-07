"""Tiny helper for writing the initial-state preview block of a
``<job>.molwatch.log``.

The block format matches the spec in ``docs/engines/pyscf.md``
(and the corresponding entry in molwatch's ``docs/model/parse.md``).
This module exists so prep (`jobset/prep.py::_seed_trajectory_log`) can
drop a one-block preview file beside the deck, giving molwatch something
to render the moment a user loads it (no waiting for the engine to
produce its first frame).

The text is the one writer's (``emitter.header_and_preview``), which the
PySCF script's own writer (``emitter.MolwatchEmitter``, imported from
``mb_pyscf.pyz``) starts its log with too -- one writer of the header and
the step-0 block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from ..structure import Structure


def write_initial_preview(
    struct: "Structure",
    path: str | Path,
    *,
    job: str,
    engine: str,
    generator: str = "molbuilder",
    convergence_targets: Optional[Mapping[str, float]] = None,
    frame=None,
    frozen_atoms=(),
    plan=None,
) -> None:
    """Write a ``<path>.molwatch.log`` holding its header and exactly one
    block: step 0, the structure's coordinates, no energy, no forces, no SCF
    history -- so the Results tab has a molecule to show the moment the run
    is prepped, before the engine has written anything of its own.

    The text is :func:`~molbuilder.trajectory_log.emitter.header_and_preview`'s,
    the one writer of a log's start.  ``engine`` and ``generator`` fill the
    ``# engine:`` and ``# generator:`` lines; ``convergence_targets`` (the
    reader's own keys, e.g. ``{"max_force_tol_eV_per_A": 0.05}``) its
    threshold lines; ``frozen_atoms`` (0-based) the atoms the run holds.

    THE ENGINE'S FRAME, like every step the run writes after it
    (`model/structure-periodicity.md` § 6.0).  ``frame`` is the DECK's, when a deck is beside this log: prep may hand
    the deck a box of its own (the vibration `freq` stage's), and step 0 is
    then what that deck writes, not a placement worked out again here.

    ``plan`` (`jobset.planned.Plan`) receives the file instead of the disk.
    """
    from ..cell import to_engine
    from .emitter import header_and_preview
    positions = (frame if frame is not None
                 else to_engine(struct)).positions  # (N, 3), Angstrom
    text = header_and_preview(
        job=job, engine=engine, generator=generator,
        elements=list(struct.elements), coords_ang=positions,
        frozen_atoms=frozen_atoms or (),
        convergence_targets=convergence_targets)
    if plan is not None:
        plan.text(path, text)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


__all__ = ["write_initial_preview"]
