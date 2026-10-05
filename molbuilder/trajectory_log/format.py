"""Tiny helper for writing the initial-state preview block of a
``<job>.molwatch.log``.

The block format matches the spec in ``docs/engines/pyscf.md``
(and the corresponding entry in molwatch's ``docs/model/parse.md``).
This module exists so non-runtime molbuilder code paths -- the SIESTA
FDF generator, future engine integrations -- can drop a one-block
preview file alongside their main output, giving molwatch something
to render the moment a user loads it (no waiting for the engine to
produce its first frame).

The text is the one writer's (``emitter.header_and_preview``), which the
PySCF script's own writer (``emitter.MolwatchEmitter``, imported from
``mb_pyscf.pyz``) starts its log with too -- one writer of the header and
the step-0 block, since 2026-10-05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, TYPE_CHECKING

from ..runfiles import compose as _rf

if TYPE_CHECKING:
    from ..structure import Structure


def molwatch_log_basename(system_label: str,
                          stage: Optional[str] = None) -> str:
    """The canonical ``<label>[_<NN>_<name>].molwatch.log`` for this run.

    ``stage`` is a stage's **artifact token** (``01_coarse``), the same one its
    deck carries -- :func:`molbuilder.identity.stage_token` builds it.  So a
    stage's log takes the basename of the deck that produced it, and the two
    can be matched by name.  ``stage=None`` returns the unsuffixed name for a
    single-run workflow.

    **A run's log is named for its deck, and that is one rule rather than two**
    (``engines/stages.md`` § 7).  Until 2026-08-10 this wrote
    ``<label>-stage<N>.molwatch.log`` while the deck beside it was
    ``<label>_<name>.fdf``: two spellings of one idea, and in a ladder whose
    stages the user had named, a stage's deck and its own log could not be
    matched at all.  The hyphen was wrong twice over -- ``job-contracts.md``
    § 6.3 reserves ``-`` for *a counter follows*.

    The basename stays identical across stages either way, so SIESTA's
    restart files still transfer untouched; only molbuilder's own names carry
    the stage.

    Returns just the filename, not a full path; callers join with the
    target directory.

    IT COMPOSES THROUGH THE GRAMMAR (`job-contracts.md` § 2.2a) rather than
    formatting the pieces itself.  This function was ALREADY the one place the
    molwatch name was built -- the pattern the rest of the run files did not
    have -- so what it gains is the shared refusal: a label or a token that
    could not be read back out of the name is now rejected here too, in the
    same words, instead of being formatted into a filename nothing can parse.
    """
    return _rf(system_label, ".molwatch.log", stage)


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
    (`model/structure-periodicity.md` § 6.0): step 0 was the design
    coordinates while the deck beside it was placed, so the log jumped by
    the whole offset between step 0 and step 1 (45 Å, measured 2026-09-22).
    ``frame`` is the DECK's, when a deck is beside this log: prep may hand
    the deck a box of its own (the vibration `freq` stage's), and step 0 is
    then what that deck writes, not a placement worked out again here.

    ``plan`` (`jobset.planned.Plan`) receives the file instead of the disk.

    (A ``# stage:`` line was written here until 2026-10-05, from a
    ``stage_name`` argument; no reader read it -- the stage is the run's
    files' own.)
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
