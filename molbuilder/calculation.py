"""A described calculation, read through one door (`execution/architecture.md`
§ 3.2, rule A16; `project-layout.md` § 2.1).

A DESCRIBED CALCULATION is a template plus ``task.json`` in one folder, with
its job request beside them (``job-request.json``, `execution/job-request.md`),
and its plan (``job-set.json``) once a prep wrote one.  Every verb that works
on a calculation opens it HERE, once, and passes the object down -- prep,
launch, status, summarize, the readiness doors, the Task setup tab's routes.
Each record is read by its owner's door, which takes the folder (`task.of`,
`job_request.of`, `template_file.of`, `JobSet.of`); nothing here forms a
path.  A plan with no
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

from . import job_request as _job_request
from . import task as _task
from . import template_file as _template_file
from .job_request import JobRequest
from .jobset.model import JobSet
from .task import Task
from .template_file import TemplateFile


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
    #: HOW AND WHERE IT RUNS (`execution/job-request.md`), read against the
    #: description -- always there for a calculation :func:`open` read;
    #: ``None`` only in `jobset migrate`'s own composition from raw records,
    #: before it has written one.
    request: Optional[JobRequest] = None

    @property
    def label(self) -> str:
        return self.task.label

    @property
    def template_text(self) -> Optional[str]:
        return self.template.text if self.template is not None else None

    def job(self, stage: str):
        """The job ``stage`` is prepared as -- its row in the plan
        (`JobSet.job`, matched as every reader matches a stage's name), or
        ``None``, as it is with no plan.  PREPARED is that row
        (`job-system.md` § 5.0): the one door every reader asks."""
        return self.jobset.job(stage) if self.jobset is not None else None

    def with_plan(self, jobset: Optional[JobSet]) -> "Calculation":
        """The same calculation with ``jobset`` as its plan -- what a prep
        holds while it plans: the plan with its own rows merged in."""
        return dataclasses.replace(self, jobset=jobset)

    def as_planned(self, plan) -> "Calculation":
        """The calculation as a prep's ``plan`` (`jobset.planned.Plan`)
        leaves it: its plan read through the overlay (`JobSet.of(base,
        plan=)`, the one read prep makes of the set it is writing)."""
        return self.with_plan(JobSet.of(self.base, plan=plan))


def register_readers() -> None:
    """THE READERS OF A CALCULATION'S OWN RECORDS, registered into the parse
    registry -- a transport calculation's description (`task_reader`, floor
    2) and a benchmark's plan (`jobset.sweep_reader`, floor 3), each on its
    record's floor and read through its owner's door, which the registry,
    on floor 1, imports none of (`execution/architecture.md` § 2.1) -- and
    each engine's reading of the same calculation, into its door.  Called
    by each surface as it starts -- the web app (`web.app.create_app`) and
    the command line (`cli.main`) -- so the Results tab's folder door finds
    both.  Idempotent, as the registry is."""
    from .jobset.sweep_reader import JobSetSweepFileParser
    from .parse import register
    from .task_reader import TransportTaskFileParser
    register(TransportTaskFileParser)
    register(JobSetSweepFileParser)
    # ...AND EACH ENGINE'S READING OF THE SAME CALCULATION, into its one
    # door on floor 1 (`deck_record.same_calculation`,
    # `execution/job-request.md` § 6), which imports none.
    from .deck_record import register_reading
    from .pyscf.script_input import script_differences
    from .siesta.engine_input import engine_input_differences
    register_reading("siesta", engine_input_differences)
    register_reading("pyscf", script_differences)


def open(base) -> Calculation:
    """THE DOOR: the described calculation in ``base``, read once.

    Refused (:class:`NotACalculation`) when ``base`` holds no description
    with its template and its job request -- saying which is missing; from
    inside a calculation, naming its root (`calcdirs.root_of`), since the
    verbs work on the calculation and `molbuilder describe` there would
    describe a new one inside an attempt (W52).  A record that does not read
    is refused in its reader's own words (`ValueError`): those name the file
    and, for a record written before a rule, `jobset migrate`.
    """
    calc = _open(base)
    if calc is None:
        raise NotACalculation(refusal(base))
    return calc


def at(path) -> Calculation:
    """THE CALCULATION A PATH NAMES -- its folder, or any file or folder
    below it, as a page names the file it shows: opened at its root
    (`calcdirs.where`) through :func:`open`.  A path that is not there is
    refused (``FileNotFoundError``); one in no calculation, or whose
    records do not read, as :func:`open` refuses it."""
    from . import calcdirs
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"nothing at {path}")
    folder = p if p.is_dir() else p.parent
    root = calcdirs.where(folder).root
    return open(root if root is not None else folder)


def _open(base) -> Optional[Calculation]:
    """The calculation, or ``None`` when ``base`` holds no description, no
    template or no job request -- one look at each record."""
    base = Path(base).resolve()
    task = _task.of(base)
    if task is None:
        return None
    template = _template_file.of(base, task.label)
    if template is None:
        return None
    request = _job_request.of(base, task)
    if request is None:
        return None
    return Calculation(base=base, task=task, template=template,
                       jobset=JobSet.of(base), request=request)


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
    if calcdirs.described(base):
        if not _job_request.path_of(base).is_file():
            return (f"{base} holds {_task.FILENAME} but no "
                    f"{_job_request.FILENAME} beside it -- a calculation is "
                    f"described by both (execution/job-request.md § 7); "
                    f"Task setup's Save writes them together.")
        return (f"{base} holds {_task.FILENAME} but no template beside it "
                f"-- a described calculation is the two together "
                f"(project-layout.md § 2.1); run `molbuilder describe` "
                f"in a folder of its own.")
    return (f"{base} is not a described calculation -- no "
            f"{_task.FILENAME} + template pair (project-layout.md § 2.1); "
            f"run `molbuilder describe` first.")
