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
    def offered(self) -> str:
        """What it offers, in the words the extension's panel opens with."""
        from .stages import per_point_rungs, TRANSPORT_STAGES
        rungs = [s for s in TRANSPORT_STAGES if s in per_point_rungs()]
        return (f"What is computed per voltage is yours to change: the "
                f"voltages, added to {self.name}'s own, the treatment, the "
                f"transmission window and how many times finer, and the "
                f"values of {' and '.join(rungs)}.  The shared values and "
                f"the values of {', '.join(self.fixed)} are {self.said}.")

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
                "The low-bias approximation instead runs the device and the "
                "transmission once, at 0 V -- every voltage's current taken "
                "from that T(E)."),
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
            "bias": (f"The voltages {self.name} computed -- {volts} -- are "
                     f"kept; add voltages to the list.  A voltage added "
                     f"runs its device point (self-consistent) and its "
                     f"transmission point; under the low-bias "
                     f"approximation, nothing while the window reaches "
                     f"it."),
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
        template's stated one, else the default it runs; whose; and
        ``chosen`` -- whether a person chose it (the rung's own value, or
        one the template says is the person's), which a new calculation
        from the junction starts from, and nothing else."""
        own = self.own(rung) if rung else {}
        name = field["name"]
        if name in own:
            value, chosen = own[name], True
        elif "value" in field:
            value, chosen = field["value"], field.get("source") == "person"
        else:
            value, chosen = field.get("default"), False
        return {"value": value, "why": f"the value {self.name} ran with",
                "of": self.name, "chosen": chosen}

    def refusal(self, task, *, against=None,
                shared: Optional[Dict[str, object]] = None
                ) -> Optional[Refused]:
        """What of the description ``task`` it does not take, or ``None``,
        each change measured against ``against`` -- the description it is
        made from: the finished calculation's at the Transport tab's
        Describe, the folder's own at Task setup's Save -- so what the
        command line wrote, which limits nothing, is never refused for
        what it holds.  Refused: the junction, its rename or its label
        changed; a voltage both computed by the finished calculation and
        held by ``against`` dropped; a kept rung's values changed; a shared
        value chosen (``shared``, the panel's)."""
        ref = against if against is not None else self.finished
        for what, now, was in (
                ("the junction", task.slots.get("junction"),
                 ref.slots.get("junction")),
                ("the electrodes' rename (swap_electrodes)",
                 task.swap_electrodes, ref.swap_electrodes),
                ("the label", task.label, ref.label)):
            if now != was:
                return Refused(f"{what} is {self.said}")
        kept = sorted((set(float(v) for v in self.finished.bias)
                       & set(float(v) for v in ref.bias))
                      - set(float(v) for v in task.bias))
        if kept:
            one = len(kept) == 1
            return Refused(
                f"an extension adds voltages and keeps {self.name}'s: "
                f"{', '.join(f'{v:g}' for v in kept)} V "
                f"{'is' if one else 'are'} not in the list -- a list "
                f"without {'it' if one else 'them'} is another calculation, "
                f"started from this junction", where="bias")
        for rung in self.fixed:
            if _own(task, rung) != _own(ref, rung):
                return Refused(f"{rung}'s values are {self.said}")
        if shared:
            return Refused(f"the shared values are {self.said}")
        return None

    def to_wire(self) -> Dict[str, object]:
        """Its wire form -- what the describe door and the folder answer
        serve: the rungs it keeps, what it offers, and what each control
        says (:attr:`says`)."""
        return {"reuses": self.reuses, "name": self.name,
                "fixed": self.fixed, "offered": self.offered,
                "says": self.says}


def _own(task, rung: str) -> Dict[str, object]:
    """``task``'s own values of ``rung``."""
    return next((dict(st.overrides) for st in task.stages
                 if st.name == rung), {})


__all__ = ["Extension", "Refused", "ReuseError", "fixed_rungs", "reused",
           "write_reusing"]
