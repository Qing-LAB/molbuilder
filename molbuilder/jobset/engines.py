"""The engine seam -- what an engine supplies for `prep` to run its steps
over it (`execution/script-preparation.md` § 4), stated as data, and its one
door (:func:`engine_seam`), which resolves the part each engine's record
names (``seam``): each engine builds its own in its own package
(`siesta.seam`, `pyscf.seam`).  Floor 3 (`execution/architecture.md`
§ 2.1).

Nothing here prepares anything -- it describes an engine, and the conductor
asks it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .errors import PrepError


@dataclass(frozen=True)
class EngineSeam:
    """What an engine supplies for `prep` to run the steps over it —
    `script-preparation.md` § 4's seam, stated as data.

    **That document indexes the questions by the STEP that asks them**, which is
    the ordering to read them in: a bag of callables cannot answer *"what does
    this engine still owe?"*, and against the steps a gap is a blank row.

    **Of § 4's questions**, one is answered by shared code
    (`validation.validate`), and the ``spec.`` rows arrive together through
    ``spec_for`` -- the layout, the syntax, the record's values, the check
    rules and the finish are all the engine describing its deck, so they
    ride on one ``DeckSpec`` rather than on a seam member each.  A member
    that answers ``None`` is answering *nothing*, which is a real answer
    and a recorded one (§ 4, W5): PySCF gives it to ``provide_data``,
    ``shared_package`` and ``sibling_artifacts``, and to ``bench_marks`` on
    the spec.

    Everything engine-specific that the loop below needs lives HERE, so the
    loop itself never asks which engine it is in.
    """
    #: ``(structure, config, names=) -> DeckSpec`` — the engine
    #: DESCRIBES its deck; the framework renders, writes and checks it
    #: (`script-preparation.md` § 4.3).  The stage's names are a RENDER
    #: ARGUMENT (step 7, C7; `runfiles.RunNames`): the emitter never learns
    #: the word, the names it is handed carry it -- and every name the deck
    #: writes or prints.
    #:
    #: Given a form, the framework can also re-derive what the deck was
    #: supposed to contain, so nothing has to be carried alongside the text to
    #: make the check possible.
    spec_for: Callable
    #: The deck's type suffix (``.fdf``).
    suffix: str
    #: ``(config, label) -> config`` — the identity WRITTEN, for a trial's
    #: relabelling.  Filename relabelling alone is not the protection
    #: (`generator.md` § 4.2): the deck's own ``SystemLabel`` line is what
    #: keys the warm files.
    relabel: Callable
    #: ``(label, config, calculation, base_dir) -> warm-file declaration``
    #: for the Job -- the engine reads its § 4.2a rules file for the TYPE
    #: (U2), and ``base_dir`` lets a calculation's own fine-tuned copy win
    #: (U6a).
    warm_for: Callable
    #: ``config -> traits``: the facts its warm files are conditioned on --
    #: SIESTA's optimizer, which `model.warm_carry` compares (§ 4, 3.13).
    traits_for: Callable
    #: ``(struct, config, deck_path) -> None`` — the sibling files this
    #: engine's deck TEXT promises (E6: a charged SIESTA deck instructs
    #: running a script; the promise must be kept on every route that
    #: renders the deck).  ``None`` for an engine whose decks promise
    #: nothing.
    sibling_artifacts: Optional[Callable] = None
    #: ``(base_dir) -> [filename]`` — which files in the calculation are the
    #: SHARED PACKAGE every job copies.  The engine that put them there is
    #: the one that can name them.  ``None`` for an engine that puts nothing in — the package is then empty,
    #: which is the honest answer rather than an accident of a glob.
    shared_package: Optional[Callable] = None
    #: ``(struct, config, base_dir) -> None`` — the DATA FILES this engine's
    #: deck cannot run without, put into the calculation.
    #:
    #: Distinct from
    #: ``sibling_artifacts``, which is about what a deck's own text promises;
    #: this is about what the ENGINE will open.  ``None`` for an engine that
    #: needs none (PySCF's basis sets ship inside PySCF).
    #:
    #: It belongs to `prep` because `prep` is already what decides the
    #: shared package; which files it uses and where it looks for the
    #: missing ones -- the calculation's own folder first, then the library
    #: inside the projects tree -- are `job-contracts.md` § 2.5a.
    provide_data: Optional[Callable] = None
    #: ``(struct, config, *, calculation, dest_dir) -> (findings, refusal)``
    #: — the findings a describe adds before it writes this engine's
    #: calculation into ``dest_dir`` (`web/handover-procedure.md` § 2.2), and
    #: the sentence it is refused in while one is an error, else ``None``.
    #: SIESTA's: its pseudopotentials, the files already in the folder
    #: counted as at `prep`.  ``None`` for an engine whose describe adds
    #: none.  ASKED BEFORE `prep`, by the hand-over that writes the
    #: calculation (its ``step``, `script-preparation.md` § 4) -- every other
    #: member is a preparation step's.
    describe_findings: Optional[Callable] = field(
        default=None, metadata={"step": "describe"})


def relabel(engine: str) -> Callable:
    """``(config, label) -> config`` for ``engine``: the label written into
    the setting its record in the engine table names as the one that names
    a calculation's files (``identity_field`` -- ``system_label``,
    ``job_name``)."""
    from ..engine_table import relabelled
    return lambda cfg, label: relabelled(engine, cfg, label)


def engine_seam(engine: str) -> EngineSeam:
    """What ``engine`` supplies for `prep` to run its steps over it -- the
    seam its record in the engine table names (``seam``,
    `execution/architecture.md` § 3.3); an engine the table does not hold
    refused, saying an engine supplies its catalogue rows and an answer at
    each preparation step."""
    from ..engine_table import EngineTableError, engine as _engine
    from ..kind_table import part
    try:
        rec = _engine(str(engine))
    except EngineTableError as exc:
        raise PrepError(
            f"no deck writer for engine {engine!r}. An engine supplies its "
            f"catalogue rows and an answer at each preparation step"
            f"; {exc}") from exc
    return part(rec.seam, record=f"engine {rec.name}", key="seam")()


__all__ = ["EngineSeam", "engine_seam", "relabel"]
