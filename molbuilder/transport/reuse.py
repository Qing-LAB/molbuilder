"""A TRANSPORT CALCULATION THAT REUSES A FINISHED ONE, described
(`engines/transport.md` § 2a.16, *Describing it*): the finished calculation
opened through the calculation door, and the new one written from its
records as they read now -- its description, template and job request, its
composed junction, read by the composition's one reader and written by its
one writer, and its pseudopotentials -- through the one writer every
describe takes (`describe.write_description`).

Asked by `molbuilder describe --reuses`; the Transport tab's describe door
writes through :func:`write_reusing` too.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .compose import ComposeError


class ReuseError(ValueError):
    """The folder ``--reuses`` names is not a transport calculation to
    reuse, or the new folder may not hold one that reuses it.  Nothing was
    written."""


def reused(raw, *, tree_root):
    """THE FINISHED CALCULATION a new one reuses -- ``raw`` its folder,
    from the projects tree's root or absolute inside it -- opened through
    the calculation door (`calculation.open`), and its folder tree-relative,
    as the new description's ``reuses`` records it.

    Refused (:class:`ReuseError`): a folder outside the tree; one that is
    not a described calculation, in the door's words; one whose description
    does not read, in its reader's; another kind of calculation."""
    from .. import calculation
    from ..projects import OutsideRoot, contain
    root = Path(tree_root).expanduser().resolve()
    path = Path(raw).expanduser()
    try:
        folder = contain(path if path.is_absolute() else root / path, root)
    except OutsideRoot as exc:
        raise ReuseError(f"{raw}: {exc}") from exc
    try:
        calc = calculation.open(folder)
    except ValueError as exc:
        raise ReuseError(f"{raw}: {exc}") from exc
    if calc.task.calculation != "transport":
        raise ReuseError(
            f"{raw}: it is not a transport calculation (its "
            f"description says {calc.task.calculation!r}) -- a calculation "
            f"reuses a finished transport calculation "
            f"(engines/transport.md 2a.16).")
    return calc, str(calc.base.relative_to(root))


def write_reusing(desc, dest, finished, *, tree_root
                  ) -> Tuple[List[Path], List[str]]:
    """``desc`` -- a transport description whose ``reuses`` names
    ``finished`` -- written into ``dest`` with the finished calculation's
    composed junction, as its record holds it, and its pseudopotentials,
    every file or none.  Answers what was written, and what the person is
    told: why no junction travels, when the finished calculation's record
    does not read -- prep then composes it from its citation, as for any
    calculation.

    Refused before anything is written: ``dest`` inside the finished
    calculation, or inside its junction's citation, as every describe
    refuses it (`compose.refuse_inside_citation`); a cited pair edited since
    the finished calculation was composed from it, by name
    (`compose.load_compose_record`)."""
    import dataclasses

    from ..chemistry import species_order
    from ..describe import write_description
    from ..pseudos import PSEUDO_DIRNAME
    from .compose import load_compose_record, refuse_inside_citation
    try:
        for cited, what in (
                (finished.base, "the finished calculation"),
                (Path(tree_root) / finished.task.slots["junction"],
                 "the cited junction")):
            refuse_inside_citation(dest, cited, what=what)
    except ComposeError as exc:
        raise ReuseError(str(exc)) from exc
    why: List[str] = []
    try:
        composed = load_compose_record(
            finished.base, citation=finished.task.slots["junction"],
            tree_root=tree_root, why=why,
            swap_electrodes=finished.task.swap_electrodes)
    except ComposeError as exc:
        raise ReuseError(str(exc)) from exc
    notes: List[str] = []
    if composed is None:
        notes.append(f"{finished.base.name} carries no composed junction -- "
                     f"{'; '.join(why)}; prep composes it from its citation")
    else:
        desc = dataclasses.replace(desc, pseudo_species=tuple(
            species_order(composed.junction.elements)))
    written = write_description(desc, dest,
                                psml_lib=finished.base / PSEUDO_DIRNAME,
                                junction=composed)
    return written, notes


__all__ = ["ReuseError", "reused", "write_reusing"]
