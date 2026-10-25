"""A TRANSPORT CALCULATION THAT REUSES A FINISHED ONE, described
(`engines/transport.md` § 2a.16, *Describing it*): the finished calculation
opened through the calculation door, and the new one written from its
records as they read now -- its description, template and job request, its
composed junction, read by the composition's one reader and written by its
one writer, and its pseudopotentials -- through the one writer every
describe takes (`describe.write_description`); or, in place of its junction,
a frame set whose frame 0 is that junction (*A frame set of the same
junction*, :func:`frame_set_refusal`), composed here.

Asked by `molbuilder describe --reuses`; the Transport tab's describe door
writes through :func:`write_reusing` too, and its pick of a frame set asks
:func:`frame_set_of`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def frame_set_refusal(junction, frame_set, *, name: str,
                      citation: str) -> Optional[str]:
    """WHETHER ``frame_set``'S FRAME 0 IS ``junction`` (`engines/transport.md`
    § 2a.16, *A frame set of the same junction*) -- ``None`` when it is,
    else the refusal, in words: ``junction`` the finished calculation's
    composed junction, ``name`` its folder's name; ``frame_set`` the one
    ``citation`` composes.  Each in the user order, its frame 0, its labels
    the transport partition alone (`sort.PARTITION_LABELS`): the same atoms
    at the same places, the same cell and axes, the same partition, by the
    one rule two junctions are compared by (`compose.junction_difference`).
    The held set and every other label are not compared: no transport deck
    states them."""
    from .compose import junction_difference
    from .sort import PARTITION_LABELS

    def frame_0(struct):
        s = struct.restored().frame_at(0)
        return s.replace(regions={k: list(v)
                                  for k, v in (s.regions or {}).items()
                                  if k in PARTITION_LABELS})
    differs = junction_difference(frame_0(junction), frame_0(frame_set))
    if differs is None:
        return None
    return (f"{citation}: its frame 0 is not the junction {name} ran -- "
            f"{differs}.  A calculation that extends {name} uses its "
            f"junction, or a frame set whose first frame is that junction: "
            f"the same atoms in the same order at the same places, the same "
            f"cell and axes, the same L-electrode, R-electrode, bridge and "
            f"buffer (engines/transport.md 2a.16).  A frame set of another "
            f"junction is a new calculation of its own, not an extension of "
            f"{name}.")


def frame_set_of(finished, citation: str, *, tree_root, composed=None):
    """THE FRAME SET A REUSE CITES IN THE FINISHED JUNCTION'S PLACE
    (`engines/transport.md` § 2a.16, *A frame set of the same junction*):
    ``citation`` composed with the finished calculation's rename -- or
    ``composed``, already composed so -- and checked against the junction
    the finished calculation ran, as its record holds it
    (:func:`frame_set_refusal`).  Refused, :class:`ReuseError`: a record
    that does not read, in its reader's words; a citation that does not
    compose, in the composer's; a frame 0 that is another junction, by
    name.  THE ONE CHECK the writer and the Transport tab's pick ask."""
    from .compose import compose_junction, load_compose_record
    from .sort import SortError
    why: List[str] = []
    try:
        ran = load_compose_record(
            finished.base, citation=finished.task.slots["junction"],
            tree_root=tree_root, why=why,
            swap_electrodes=finished.task.swap_electrodes)
    except ComposeError as exc:
        raise ReuseError(str(exc)) from exc
    if ran is None:
        raise ReuseError(
            f"{citation} is compared with the junction {finished.base.name} "
            f"ran, as its record holds it -- {'; '.join(why)} "
            f"(engines/transport.md 2a.16)")
    if composed is None:
        try:
            composed = compose_junction(
                citation, tree_root=tree_root,
                swap_electrodes=finished.task.swap_electrodes)
        except (ComposeError, SortError) as exc:
            raise ReuseError(str(exc)) from exc
    refused = frame_set_refusal(ran.junction, composed.junction,
                                name=finished.base.name, citation=citation)
    if refused:
        raise ReuseError(refused)
    return composed


def citing(task, junction: str):
    """``task`` -- a description reusing a finished calculation, made from
    its own -- citing ``junction`` in its junction's place
    (`engines/transport.md` § 2a.16, *A frame set of the same junction*):
    its run id is its label and its citation, derived again
    (`run-identity.md` § 2); ``task`` itself for the junction it cites.
    Whether ``junction`` is a frame set of it is the writer's question
    (:func:`write_reusing`)."""
    import dataclasses

    from ..task import derive_run
    if junction == task.slots.get("junction"):
        return task
    run = task.run
    return dataclasses.replace(
        task, slots={**task.slots, "junction": junction},
        run=derive_run(run.name, junction, created=run.created,
                       stage_names=tuple(st.name for st in task.stages)))


def write_reusing(desc, dest, finished, *, tree_root
                  ) -> Tuple[List[Path], list]:
    """``desc`` -- a transport description whose ``reuses`` names
    ``finished`` -- written into ``dest`` with the finished calculation's
    composed junction, as its record holds it, and its pseudopotentials,
    every file or none.  Answers what was written, and what the person is
    told, each at its weight (`issues.Issue`): why no junction travels,
    when the finished calculation's record does not read -- prep then
    composes it from its citation, as for any calculation -- a warning; the
    frame set written in its place, information.

    A ``desc`` citing another junction than the finished calculation's
    cites a frame set of it (`engines/transport.md` § 2a.16, *A frame set
    of the same junction*): composed and checked by the one check
    (:func:`frame_set_of`), every frame, and written in its place.

    Refused before anything is written: ``dest`` inside the finished
    calculation, or inside a junction's citation, as every describe
    refuses it (`compose.refuse_inside_citation`); a cited pair edited since
    the finished calculation was composed from it, by name
    (`compose.load_compose_record`); a frame set whose frame 0 is not the
    finished junction, or compared with a finished junction whose record
    does not read, by name."""
    import dataclasses

    from ..chemistry import species_order
    from ..describe import write_description
    from ..issues import Issue
    from ..pseudos import PSEUDO_DIRNAME
    from .compose import load_compose_record, refuse_inside_citation
    citation = desc.task.slots["junction"]
    ran = finished.task.slots["junction"]
    try:
        for cited, what in (
                (finished.base, "the finished calculation"),
                *((Path(tree_root) / c, "the cited junction")
                  for c in dict.fromkeys((ran, citation)))):
            refuse_inside_citation(dest, cited, what=what)
    except ComposeError as exc:
        raise ReuseError(str(exc)) from exc
    why: List[str] = []
    notes: list = []
    if citation != ran:
        # A FRAME SET OF THE SAME JUNCTION, cited in its place (§ 2a.16).
        composed = frame_set_of(finished, citation, tree_root=tree_root)
        notes.append(Issue(
            "info", f"the junction is {citation}, its "
                    f"{composed.junction.n_frames} frame(s); its first frame "
                    f"is the first frame {finished.base.name} ran"))
    else:
        try:
            composed = load_compose_record(
                finished.base, citation=ran, tree_root=tree_root, why=why,
                swap_electrodes=finished.task.swap_electrodes)
        except ComposeError as exc:
            raise ReuseError(str(exc)) from exc
    if composed is None:
        notes.append(Issue(
            "warn", f"{finished.base.name} carries no record of its junction "
                    f"that reads -- {'; '.join(why)}; prep composes the "
                    f"junction again from its citation, and nothing "
                    f"{finished.base.name} computed is then reused: every "
                    f"stage runs here"))
    else:
        desc = dataclasses.replace(desc, pseudo_species=tuple(
            species_order(composed.junction.elements)))
    written = write_description(desc, dest,
                                psml_lib=finished.base / PSEUDO_DIRNAME,
                                junction=composed)
    return written, notes


def fixed_rungs() -> List[str]:
    """THE RUNGS AN EXTENSION KEEPS (`engines/transport.md` § 2a.16,
    *Extending it*): those that run once for every voltage -- the complement
    of `stages.per_point_rungs`, the seed and the leads -- whose values, with
    the shared ones, the finished calculation fixes.  The rule found in the
    ladder."""
    from .stages import TRANSPORT_STAGES, per_point_rungs
    return [s for s in TRANSPORT_STAGES if s not in per_point_rungs()]


@dataclass(frozen=True)
class Refused:
    """A change an extension does not take, in its words, and where it lands
    -- ``where`` the control it names (`task.bias`), or ``""``."""
    message: str
    where: str = ""


@dataclass(frozen=True)
class Extension:
    """AN EXTENSION OF A FINISHED TRANSPORT CALCULATION (`engines/transport.md`
    § 2a.16, *Extending it*): what it keeps -- the shared values, the rungs
    that run once (:func:`fixed_rungs`) with the finished calculation's own
    values, its voltages -- what it offers, and what each change offered
    re-runs.  THE ONE DOOR every surface draws the limits from and both
    describing doors check them by (the Transport tab's Describe, Task
    setup's Save); the command line limits nothing and says what it reuses.
    """
    reuses: str          # the finished calculation's folder, tree-relative
    finished: object     # its description (`task.Task`)

    @classmethod
    def of(cls, reuses: str, *, tree_root) -> "Extension":
        """The extension of the finished calculation ``reuses`` names
        (:func:`reused`'s refusals its own)."""
        calc, rel = reused(reuses, tree_root=tree_root)
        return cls(reuses=rel, finished=calc.task)

    @property
    def name(self) -> str:
        """The finished calculation, by its folder's name."""
        return Path(self.reuses).name

    @property
    def fixed(self) -> List[str]:
        return fixed_rungs()

    @property
    def said(self) -> str:
        """What every surface says of a value it keeps, and the way out."""
        return (f"fixed by {self.name}: an extension copies in the runs "
                f"every voltage stands on, computed with these values -- to "
                f"change them, start a new calculation from this junction "
                f"on the Transport tab")

    @property
    def says(self) -> Dict[str, str]:
        """WHAT EACH CONTROL SAYS, whole sentences every surface draws as
        they are (`engines/transport.md` § 2a.16, *Extending it*, its
        table): the bias list, the treatment -- by the one the finished
        calculation ran -- the window, the shared values, and each rung:
        one it keeps, :attr:`said`; one it offers, what a change there
        re-runs; and ``fixed``, the short form a kept row is marked with."""
        volts = ", ".join(f"{float(v)} V" for v in self.finished.bias)
        treatment = {
            "self-consistent": (
                f"The low-bias approximation instead runs the device and the "
                f"transmission once, at 0 V -- {self.name}'s 0 V results on "
                f"its first frame, copied in -- every voltage's current "
                f"taken from that T(E)."),
            "low-bias-approximation": (
                "Self-consistent instead runs the device and the "
                "transmission at every voltage, each its own "
                "self-consistent point, the 0 V ones taken over."),
        }.get(str(self.finished.treatment), (
            f"{self.name} computed one voltage, so the treatment decides "
            f"what a voltage added runs: self-consistent, its device point "
            f"and its transmission point; the low-bias approximation, "
            f"nothing while the window reaches it."))
        out = {
            "bias": (f"{self.name} computed {volts}.  A voltage you keep is "
                     f"copied from it; one you add runs its device point "
                     f"(self-consistent) and its transmission point -- "
                     f"under the low-bias approximation, nothing while the "
                     f"window reaches it; one you remove is left behind."),
            "treatment": treatment,
            "window": ("A wider or finer window re-runs the transmission at "
                       "the new energies only -- the finished ones are "
                       "shared exactly."),
            "shared": f"The values {self.name} ran with -- {self.said}.",
            "fixed": f"fixed by {self.name}",
            "device": ("A change here re-runs every device point, and the "
                       "transmission on it."),
            "transmission": ("A change here re-runs the whole "
                             "transmission, every point."),
        }
        for rung in self.fixed:
            out[rung] = f"{self.said[0].upper()}{self.said[1:]}."
        return out

    def own(self, rung: str) -> Dict[str, object]:
        """The finished calculation's own values of ``rung``."""
        return _own(self.finished, rung)

    def fixes(self, *, surface: str, rung: Optional[str]) -> bool:
        """Whether a field of ``surface`` -- the shared panel, or ``rung``'s
        tab -- is one it keeps."""
        return surface == "shared" or rung in self.fixed

    def locked(self, field: Dict[str, object], *,
               rung: Optional[str]) -> Dict[str, object]:
        """A KEPT FIELD, DRAWN AT THE VALUE THE FINISHED CALCULATION RAN
        WITH (the form's ``locked``): ``rung``'s own value, else the
        template's stated one, else the default it runs; whose; ``chosen``
        -- whether a person chose it (the rung's own value, or one the
        template says is the person's), which a new calculation from the
        junction starts from; and ``blank`` -- nobody gave it a value (a
        default, or a cited value the person emptied), which that new
        calculation keeps blank, not cited anew."""
        own = self.own(rung) if rung else {}
        name = field["name"]
        if name in own:
            value, chosen = own[name], True
        elif "value" in field:
            value, chosen = field["value"], field.get("source") == "person"
        else:
            value, chosen = field.get("default"), False
        return {"value": value, "why": f"the value {self.name} ran with",
                "of": self.name, "chosen": chosen,
                "blank": name not in own and "value" not in field}

    def refusal(self, task, *, against=None,
                shared: Optional[Dict[str, object]] = None
                ) -> Optional[Refused]:
        """What of the description ``task`` it does not take, or ``None``,
        each change measured against ``against`` -- the description it is
        made from: the finished calculation's at the Transport tab's
        Describe, the folder's own at Task setup's Save -- so what the
        command line wrote, which limits nothing, is never refused for
        what it holds.  Refused: its rename or its label changed; a kept
        rung's values changed; a shared value chosen (``shared``, the
        panel's).  The voltages are the person's to add or remove (§ 2a.16,
        *A narrower question*).  The junction -- the finished one, or a
        frame set of it -- is the writer's to check (:func:`frame_set_of`,
        § 2a.16), and Task setup's Save of another is another calculation,
        refused by its id."""
        ref = against if against is not None else self.finished
        for what, now, was in (
                ("the electrodes' rename (swap_electrodes)",
                 task.swap_electrodes, ref.swap_electrodes),
                ("the label", task.label, ref.label)):
            if now != was:
                return Refused(f"{what} is {self.said}")
        for rung in self.fixed:
            if _own(task, rung) != _own(ref, rung):
                return Refused(f"{rung}'s values are {self.said}")
        if shared:
            return Refused(f"the shared values are {self.said}")
        return None

    def to_wire(self) -> Dict[str, object]:
        """Its wire form -- what the describe door and the folder answer
        serve: the rungs it keeps, and what each control says
        (:attr:`says`)."""
        return {"reuses": self.reuses, "name": self.name,
                "fixed": self.fixed, "says": self.says}


def _own(task, rung: str) -> Dict[str, object]:
    """``task``'s own values of ``rung``."""
    return next((dict(st.overrides) for st in task.stages
                 if st.name == rung), {})


__all__ = ["Extension", "Refused", "ReuseError", "citing", "fixed_rungs",
           "frame_set_of", "frame_set_refusal", "reused", "write_reusing"]
