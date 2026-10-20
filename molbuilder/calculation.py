"""A described calculation, read through one door (`execution/architecture.md`
§ 3.2, rule A16; `project-layout.md` § 2.1).

A DESCRIBED CALCULATION is a template plus ``task.json`` in one folder, and
its plan (``job-set.json``) once a prep wrote one.  Every verb that works on
a calculation opens it HERE, once, and passes the object down -- prep, launch,
status, summarize, the readiness doors, the Task setup tab's routes.  Each
record is read by its owner's door, which takes the folder (`task.of`,
`template.of`, `JobSet.of`); nothing here forms a path.  A plan with no
description beside it is no calculation of molbuilder's: the verbs refuse
it, and `launch`'s refusal says a hand-built sweep is launched and read by
hand.

A folder that is not a described calculation is refused as one thing
(:class:`NotACalculation`): from one of a calculation's stage or attempt
folders, naming the root the verbs work on; a description that does not
read is refused in its reader's own words.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import task as _task
from . import template as _template
from .jobset.model import JobSet
from .task import Task
from .template import TemplateFile


class NotACalculation(ValueError):
    """``base`` is not a described calculation: the reason, and where the
    calculation is when ``base`` is one of its folders."""


@dataclass(frozen=True)
class Calculation:
    """The calculation as its verb read it, once."""
    base: Path
    task: Task
    #: Always there for a calculation :func:`open` read; ``None`` only in
    #: `jobset migrate`'s own composition from raw records, of a folder
    #: whose template an older molbuilder never wrote.
    template: Optional[TemplateFile]
    #: The plan, when a prep has written one -- as the given ``plan``
    #: leaves it (`jobset.planned`).
    jobset: Optional[JobSet]

    @property
    def label(self) -> str:
        return self.task.label

    @property
    def template_text(self) -> Optional[str]:
        return self.template.text if self.template is not None else None

    @property
    def shape(self):
        """The layout, the description's (`Shape.named(task.shape)`):
        asked, never inferred from what is on disk."""
        from .paths import Shape
        return Shape.named(self.task.shape)

    def with_plan(self, jobset: Optional[JobSet]) -> "Calculation":
        """The same calculation with ``jobset`` as its plan -- what a prep
        holds while it plans: the plan with its own rows merged in."""
        return dataclasses.replace(self, jobset=jobset)

    def as_planned(self, plan) -> "Calculation":
        """The calculation as a prep's ``plan`` (`jobset.planned.Plan`)
        leaves it: its plan read through the overlay (`JobSet.of(base,
        plan=)`, the one read prep makes of the set it is writing)."""
        return self.with_plan(JobSet.of(self.base, plan=plan))


def open(base) -> Calculation:
    """THE DOOR: the described calculation in ``base``, read once.

    Refused (:class:`NotACalculation`) when ``base`` holds no description
    with its template -- saying which of the two is missing; from inside a
    calculation, naming its root (`calcdirs.root_of`), since the verbs work
    on the calculation and `init` there would describe a new one inside an
    attempt (W52).  A description or a template that does not read is
    refused in its reader's own words (`ValueError`): those name the file
    and, for a record written before a rule, `jobset migrate`.
    """
    calc = _open(base)
    if calc is None:
        raise NotACalculation(refusal(base))
    return calc


def _open(base) -> Optional[Calculation]:
    """The calculation, or ``None`` when ``base`` holds no description or
    no template -- one look at each record."""
    base = Path(base).resolve()
    task = _task.of(base)
    if task is None:
        return None
    template = _template.of(base, task.label)
    if template is None:
        return None
    return Calculation(base=base, task=task, template=template,
                       jobset=JobSet.of(base))


def refusal(base) -> str:
    """Why ``base`` is not a described calculation, in one sentence -- the
    one spelling every boundary shows: inside a calculation, its root;
    else which of the two records is missing."""
    from . import calcdirs
    base = Path(base).resolve()
    root = calcdirs.root_of(base)
    if root is not None and Path(root).resolve() != base:
        return (f"{base} is a folder of the calculation at {root}; the "
                f"verbs work on the calculation -- name that folder (the "
                f"command line's `--bundle`).")
    if _task.described(base):
        return (f"{base} holds {_task.FILENAME} but no template beside it "
                f"-- a described calculation is the two together "
                f"(project-layout.md § 2.1); run `molbuilder jobset init` "
                f"in a folder of its own.")
    return (f"{base} is not a described calculation -- no "
            f"{_task.FILENAME} + template pair (project-layout.md § 2.1); "
            f"run `molbuilder jobset init` first.")
