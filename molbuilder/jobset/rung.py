"""The rung -- what a kind adds to the one table of steps
(`execution/script-preparation.md` § 3.0) -- and the structure step every
rung's builder starts from.

**What every builder shares, on floor 3, so no builder imports the
conductor** (`execution/architecture.md` § 3.3): the :class:`Rung` a
builder answers, the structure a calculation is *of* (:func:`structure_for`,
:func:`structure_path`, a citing kind's cited pair :func:`cited_pair_path`)
and the builder of a kind with no steps of its own (:func:`described_rung`).
A pair with steps of its own names its builder in the kind table (its
``rung`` part), in its kind's own package -- `transport/rung.py`,
`spectra/siesta_rung.py` -- and the conductor (`jobset.prep`) resolves it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from .errors import PrepError


def structure_path(task, base: Path) -> Optional[Path]:
    """Where the structure this calculation is *of* is -- beside the
    calculation first, at the recorded path second (`stages.md` § 6.3); the
    suffix-corrected candidate is the codec's own naming rule (a .pdb source
    travels as <stem>.xyz).  ``None`` when it is in neither place, and for
    a calculation of no structure of its own -- a transport calculation,
    which cites its junction."""
    from ..workingcopy_structure import StructureCodec
    if task.structure is None:
        return None
    src = Path(task.structure.source)
    for candidate in (base / src.name,
                      (base / src.name).with_suffix(
                          StructureCodec.GEOMETRY_SUFFIX),
                      src):
        if candidate.is_file():
            return candidate
    return None


def citation_of(task) -> Optional[str]:
    """What a citing kind's description cites -- its slot, the kind table's
    ``description.cites`` (a transport calculation's junction,
    `engines/transport.md` § 3.1) -- or ``None`` for a kind that cites
    nothing.  The one reading of the cited slot."""
    from ..kind_table import kind
    cites = kind(task.calculation).description.cites
    return (task.slots or {}).get(cites) if cites else None


def cited_pair_path(task, base: Path) -> Optional[Path]:
    """A citing kind's cited structure pair (:func:`citation_of`), fenced to
    the projects tree -- ``None`` for a cited run's folder, for a kind that
    cites nothing, and for a folder outside any projects tree, whose
    citation names nothing (its prep's own refusal says so)."""
    citation = citation_of(task)
    if not citation:
        return None
    from ..projects import OutsideRoot, contain, find_projects_root
    found = find_projects_root(base)
    if found is None:
        return None
    root = Path(found)
    try:
        path = contain(root / citation, root)
    except OutsideRoot:
        return None
    return path if path.is_file() else None


def structure_for(task, base: Path, pset):
    """The structure this calculation is *of*, from the reference in the
    description (`stages.md` § 6.3 — a reference plus a witness, never a copy).

    Looked for beside the calculation first and at the recorded path second, so
    a folder carried to a cluster with its structure alongside resolves without
    the original tree existing there.

    **The witness is checked and the mismatch is loud**: a description opened
    against a structure that has since changed would otherwise build a
    different calculation under the same id — § 1's second failure mode.
    """
    from ..runfiles import TASK_FILE
    from ..structure_pair import GeometryChangedError
    from ..workingcopy_structure import StructureCodec
    codec = StructureCodec()
    # The codec, not the bare loader: describe writes the structure as the
    # pair (--vacuum lives NOWHERE a bare .xyz can put it), and the codec
    # applies the .molstruct.json and compares its pin with the file
    # (`structure-molstruct.md` § 3).
    candidate = structure_path(task, base)
    if candidate is not None:
        try:
            struct = codec.load(candidate)
        except GeometryChangedError as exc:
            # THE PIN (`structure-molstruct.md` § 3): the atoms or the
            # cell moved after the pair was written -- a deck rendered
            # now would put the old labels on them.
            raise PrepError(f"the structure has changed since this "
                            f"calculation was described: {exc} -- describe it "
                            f"again to build on the edit") from None
    else:
        raise PrepError(
            f"the structure this calculation describes is not here: "
            f"{task.structure.source!r}. `{TASK_FILE}` records a REFERENCE to it "
            f"(engines/stages.md § 6.3), so `prep` needs the file to be either "
            f"beside the calculation or still at that path.")
    if task.structure.fingerprint and codec.fingerprint_on_disk(
            candidate) != task.structure.fingerprint:
        # THE WHOLE PAIR IS THE ONE IT WAS DESCRIBED WITH (`engines/stages.md`
        # § 6.3): every stage of a calculation is of one structure, and an
        # edit after its description -- a label, a held atom, a coordinate,
        # the box -- is refused, never worked around.
        raise PrepError(
            f"the structure has changed since this calculation was "
            f"described: {candidate.name} is not the pair its description "
            f"records ({TASK_FILE}'s structure fingerprint) -- an atom, a "
            f"label, a held atom or the box was edited after it.  Every "
            f"stage of a calculation is of the one structure it was "
            f"described with: describe it again to build on the edit.")
    if struct.formula != task.structure.formula or \
            struct.n_atoms != task.structure.atoms:
        raise PrepError(
            f"the structure has changed since this calculation was described: "
            f"the description witnesses {task.structure.formula} with "
            f"{task.structure.atoms} atoms, and {candidate} now holds "
            f"{struct.formula} with {struct.n_atoms}.\n"
            f"  Describing again is the honest fix -- rendering this deck would "
            f"build a different calculation under the same id.")
    # THE STRUCTURE ENTERS THIS CALCULATION (`model/structure-periodicity.md`
    # § 6.0): an engine's stated origin is not carried in, so the one rule
    # places it -- unless the kind keeps the engine's frame: a vibration
    # whose input is stated relaxed, as THIS stage resolves it (``pset``,
    # the stage's own values over the template's -- the door every other
    # reader of `already_relaxed` asks).
    from ..cell import entering
    relaxed = bool(getattr(pset[0].render_config(), "already_relaxed", False))
    return entering(struct, keep_engine_frame=relaxed)


# --------------------------------------------------------------------- #
#  The rung -- what a kind adds to the one table of steps               #
#  (`script-preparation.md` § 3.0)                                       #
# --------------------------------------------------------------------- #

def _at_no_point(cfg, point):
    return cfg


@dataclass
class Rung:
    """What one rung's decks are written from -- the structure step's
    answer, read by every step after it (`script-preparation.md` § 3.0).

    A pair with steps of its own builds one -- its builder the part the
    kind table names (``rung``); every other pair is a calculation of its
    described structure, one deck per element
    (:func:`described_rung`).  The conductor (:func:`prep_calculation`)
    reads these and never asks which kind it is in."""
    #: The structure the decks describe.
    struct: object
    #: The program each deck is -- `spec_for`'s ``calculation``: the
    #: calculation's own (a SIESTA vibration's phases are its parts,
    #: `spec_extra`'s ``part``).
    render_kind: str
    #: WHAT THE KIND SAYS OF THE RUN IT PREPARES, beside its request -- a
    #: SIESTA vibration's count and sizes (`engines/vibration.md` § 5.11
    #: § 9); prep's notes carry them to both doors.  Nothing judges the
    #: wall: it stays the person's.
    said: Tuple[str, ...] = ()
    #: ``cfg -> cfg``: the config an element renders from -- a lead's own
    #: label, the junction's electronic state folded in.
    configure: Callable = lambda cfg: cfg
    #: ``(element, cfg, point) -> {keyword: value}``: what else `spec_for`
    #: is told -- a vibration's block and its phase's part, the cell; the
    #: junction's electronic state and the deck's point.
    spec_extra: Callable = lambda element, cfg, point: {}
    #: ``((folder, point), ...)``: the folders each holding a deck written at
    #: that point -- a frame and a voltage (`transport.stages.Point`,
    #: `engines/transport.md` § 2a.9-11) -- each with its own wrapper; its
    #: runs are the stage's, ``run-<n>/<point>``.  Empty: one deck, in the
    #: element's folder.
    points: tuple = ()
    #: ``(cfg, point) -> cfg``: a deck's config at its point -- the rung
    #: answers the bias, the one value the catalogue does not hold
    #: (`engines/template.md` § 6.4).
    at_point: Callable = _at_no_point
    #: ``point -> struct``: the structure a point's deck describes -- its
    #: frame of the cited set; ``None``, every deck describes ``struct``.
    struct_at: Optional[Callable] = None
    #: ``(plan) -> None``: the data files, when the kind's come from
    #: somewhere other than the engine's library (transport: the citation);
    #: ``None`` is the engine's own step (`EngineSeam.provide_data`).
    provide_data: Optional[Callable] = None
    #: ``job -> job``: what the rung's job carries forward and runs, where
    #: the kind answers it per rung -- a transport rung's restart files, the
    #: transmission's program.
    job_facts: Callable = lambda job: job
    #: ``element -> ((item, who), ...)``: who set each value the rung's decks
    #: are rendered from -- `resolve`'s rows, and the rung's own words for a
    #: value its steps set after them (a transport rung's label;
    #: `job-contracts.md` § 3.2).
    sources: Callable = lambda element: element.deck_sources
    #: ``point -> role``: the one role a point of a run of points plays,
    #: where its stage plays several -- a SIESTA vibration's phase; ``None``,
    #: the stage's roles.  Its deck states the rows that role reads and its
    #: run script's record lists them (`template.reads`).
    role_at: Callable = lambda point: None
    #: ``point -> {resumes}``: how a point's run script is rendered beyond
    #: the job's own facts and what every point of a run of points has
    #: (`runwrap.PointOfRun`) -- a vibration phase's `resumes`, its
    #: warm-files section's (`engines/vibration.md` § 5.11 § 6); ``{}``
    #: renders the job's.
    wrapper_at: Callable = lambda point: {}
    #: ``point -> bool``: whether a point's deck relaxes -- its progress
    #: log's targets (`_seed_trajectory_log`); by default the render kind's
    #: record's (``relaxes``, the kind table).
    relaxes_at: Optional[Callable] = None
    #: ``(point, spec, folder, plan) -> None``: what the rung writes BESIDE
    #: a deck before the deck is checked -- a vibration phase's
    #: `coordinates.fdf`, which the deck reads as its coordinate block
    #: (`engines/vibration.md` § 5.11 § 4); ``None`` for a kind whose deck
    #: holds everything.
    beside: Optional[Callable] = None


def described_rung(calc, pset, *, seam, sweep, log, plan,
                   reuse=None) -> Rung:
    """A calculation of its described structure, one deck per element --
    every kind with no steps of its own.  ``reuse`` -- what the kind's own
    reuse decided for the prep, handed to its own builder -- is no concern
    of a kind with no steps of its own."""
    return Rung(struct=structure_for(calc.task, calc.base, pset),
                render_kind=calc.task.calculation)


__all__ = ["Rung", "citation_of", "described_rung", "structure_for",
           "structure_path", "cited_pair_path"]
