"""``job-request.json`` -- how and where a calculation runs, and its one reader.

Contract: ``docs/execution/job-request.md``.  A calculation is two records
with two editors: the description (``task.json`` and the template) says what
is computed; this one says how and where it runs -- the queue, the wall and
the memory asked (``allocation``), the run cards (``execution``, the
calculation's and each stage's), which stages share one job (each stage's
``group``), what a benchmark measures (``bench``) and when the calculation
reports (``notify``).  Nothing in one is read as the other (A17).

READ WITH THE DESCRIPTION BESIDE IT.  Its ``stages`` hold every stage the
description holds and no other; a rung's card is laid over the calculation's
for the items that rung reads, which the description's engine and kind
answer.  So the reader takes the description its caller holds (A16) and
checks the record against it; it reads nothing else of it.

LAYER.  Floor 2, beside the description: at load it imports only stdlib-only
L1 modules (``persist``, ``identity``, ``runfiles``, ``config_dir``,
``report_fields``, ``scheduler.quantities``) and the standard library.  What
a name MEANS -- an `execution` catalogue item, a value its item takes -- is
the preflight's (``validation/task.py``), which reads the vocabulary; the
shape is checked here.
"""
from __future__ import annotations

import contextvars
import difflib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NoReturn, Optional, Tuple

from . import report_fields as _report_fields
from .config_dir import is_channel_name
from .identity import stage_key
from .persist import check_schema, read_json, write_json
from .scheduler.quantities import canonical_mem, canonical_time

if TYPE_CHECKING:
    from .task import Task


SCHEMA = "molbuilder/job-request@1"
#: Its name is the catalogue's (`runfiles.WRITTEN`, `job-contracts.md` § 2.2).
from .runfiles import JOB_REQUEST_FILE as FILENAME  # noqa: E402


_TOP_KEYS = ("schema", "allocation", "execution", "stages", "bench", "notify")
_STAGE_KEYS = ("group", "execution")
#: The SCHEDULER asks of an `allocation` block (`job-request.md` § 3.1): the
#: three SLURM spells as text, and the one it switches -- whether a GPU ask
#: carries its binding (`execution/gpu.md` G9).  The launch SHAPE -- ranks,
#: threads, GPUs -- is not here: it is the run card's (`execution`, § 3.2)
#: and a benchmark's axes (`bench`).
_ALLOCATION_KEYS = ("domain", "time", "mem")
_ALLOCATION_SWITCHES = ("gpu_binding",)
_NOTIFY_KEYS = ("on_scf_converged", "every_hours", "channels", "report")

#: The two SCHEDULER asks a run owns, admitted to `execution` by name
#: (`job-request.md` § 3.2a).  They are not catalogue items and never will
#: be: the catalogue describes what an ENGINE computes with, and no engine
#: has an opinion about a wall clock or a queue.
#:
#: `mem` is deliberately absent.  The line is *does this differ between the
#: bench lane and the run lane* -- a wall does (a trial's steps are cut) and a
#: queue does (short work and long work belong in different ones); the memory
#: a job holds does not, so a second home for it would be a second place to
#: look and no new answer.
LANE_ASKS = ("time", "domain")

#: THE LANE ASKS AS A PAGE OFFERS THEM -- each with its words, its help and
#: an example -- served to the Task setup page (`/api/task-setup/sweepable`,
#: `lane_asks`), which keeps no table of its own (`web/task-setup.md`
#: § 6.2b).  One home for the two rows' words and the two names above.
LANE_ASK_ROWS = (
    {"name": "time", "label": "wall clock for this run",
     "help": ("D-HH:MM:SS -- what the scheduler is asked for; in a group "
              "of stages each row is one stage's own, and the job asks for "
              "their sum (the bench keeps the calculation's short one)"),
     "placeholder": "2-00:00:00"},
    {"name": "domain", "label": "queue for this run",
     "help": "the bench can stay on a short-turnaround queue",
     "placeholder": "public"},
)
assert tuple(r["name"] for r in LANE_ASK_ROWS) == LANE_ASKS


# --------------------------------------------------------------------- #
#  The shapes                                                           #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Allocation:
    """**What this calculation asks the scheduler for** -- the queue, the
    wall, the memory, and whether a GPU ask carries its binding
    (`job-request.md` § 3.1).

    **Every field is optional, and absent means UNSTATED** -- never a
    default wearing a number's clothes (`submission.md` S1).  A run sent to
    a queue states its queue, wall and memory, here or as a flag, or prep
    refuses it (`architecture.md` § 5.2: nothing fills one in).

    **ONE SPELLING, AND IT IS SLURM'S** -- ``"0-04:00:00"``, ``"128G"``
    (user, 2026-08-24: *"your record should set unified time format while
    it is the UI that can do some translation for human readability /
    input"*).  The two vocabularies DISAGREE -- ``04:30`` is four minutes
    thirty to SLURM and four and a half hours to a person -- so a record
    that held whichever spelling arrived could not be read correctly by
    anybody.  Translation belongs at the edges where humans are -- the
    Task-setup tab's input box, the CLI's ``--time`` / ``--mem``, a file
    typed by hand -- and every one goes through `quantities.canonical_time`
    / `quantities.canonical_mem`; what the reader RETURNS is the record's.
    """
    domain: str = ""
    time: str = ""
    mem: str = ""
    #: ``False`` turns off ``--gres-flags=enforce-binding`` for this
    #: calculation's GPU asks -- its benchmark and its runs alike, so a
    #: benchmark measures the layout the run will use (`execution/gpu.md`
    #: G9).  ``None`` is unstated, and the rule is that the binding is sent.
    gpu_binding: Optional[bool] = None

    def __bool__(self) -> bool:
        return bool(self.domain or self.time or self.mem
                    or self.gpu_binding is not None)


@dataclass(frozen=True)
class Notify:
    """**WHEN this calculation should say something** -- and deliberately
    nothing about *where* (`job-request.md` § 3.5, `run-reports.md` § 1).

    The destination and its credential live on the machine that runs the
    job -- in the config directory, mode 0600 -- and never here: this file
    travels, and a policy is safe to carry where a token is not.

    **A block, or none**: ``JobRequest.notify`` is ``None`` with no block --
    no notification at all, not even the start and the end (user,
    2026-09-26) -- and a ``Notify`` is a block, which reports the start and
    the end at least, its four values written whole.  ``channels`` and
    ``report`` ``None`` are EVERY channel and EVERY field, written ``["*"]``.
    """
    #: Fire when a step's SCF reaches its criterion -- a relaxation's move, a
    #: force-constant run's displacement (`run-reports.md` § 2.2).
    on_scf_converged: bool = False
    #: Fire every N hours.  ``None`` is never, written ``null``; a period is
    #: a positive number (W57 R10: "every 0 hours" reads as anything but
    #: never).
    every_hours: Optional[float] = None
    #: WHICH channels, **by name** -- and a name is all that travels.
    #: ``None`` is every channel the running machine has, written ``["*"]``;
    #: an EMPTY tuple is none, written ``[]``.
    channels: Optional[Tuple[str, ...]] = None
    #: WHAT each report carries, beyond the name.  ``None`` is every field
    #: the monitor could determine; ``()`` the summary line alone.
    report: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class StageRequest:
    """One stage's entry: the job it runs in (`job-request.md` § 3.3) and
    its own run card, laid over the calculation's FIELD BY FIELD (§ 3.2)."""
    group: int
    execution: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobRequest:
    """How and where one calculation runs.

    ``stages`` holds every stage of the description beside it, by its name,
    in the ladder's order -- the order the description gives, so the groups
    read in the ladder's order.  ``engine`` and ``calculation`` are the
    description's, held so a rung's card is laid over the calculation's for
    the items that rung reads (:meth:`run_condition`); they are never
    written here."""
    stages: Mapping[str, StageRequest]
    allocation: Allocation = field(default_factory=Allocation)
    #: THE CALCULATION'S RUN CARD (§ 3.2) -- one value per parameter, never
    #: a list; every rung runs at it unless its own card says otherwise.
    execution: Mapping[str, Any] = field(default_factory=dict)
    #: WHAT TO MEASURE (§ 3.4): field name -> the points to try -- what the
    #: person ASKED, never what a machine found.  Empty: no benchmark planned.
    bench: Mapping[str, Tuple[Any, ...]] = field(default_factory=dict)
    #: WHEN THE CALCULATION SPEAKS (§ 3.5); ``None`` with no block.
    notify: Optional[Notify] = None
    engine: str = field(default="", compare=False)
    calculation: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """The groups' SHAPE holds for the object too (§ 3.3): the first
        stage is group 1, each later one repeats the number above it or
        adds one, so the groups run in the ladder's order.  Whether a stage
        may share a group is the ladder's, asked by the preflight
        (`group.check_groups`)."""
        above = 0
        for name, entry in self.stages.items():
            g = entry.group
            if not isinstance(g, int) or isinstance(g, bool) or g < 1:
                raise ValueError(
                    f"{FILENAME}: stage {name!r}: group {g!r} must be a whole "
                    f"number from 1 -- the job this stage runs in "
                    f"(execution/job-request.md 3.3)")
            if g not in (above, above + 1):
                raise ValueError(
                    f"{FILENAME}: stage {name!r} is in group {g}"
                    + (f", after group {above}" if above else "")
                    + " -- the first stage is group 1, and each later one "
                      "repeats the number above it or adds one, so the "
                      "groups run in the ladder's order "
                      "(execution/job-request.md 3.3)")
            above = g

    # ----- the groups (§ 3.3) ------------------------------------------ #

    def entry(self, stage: str) -> Optional[StageRequest]:
        """``stage``'s entry -- a name in any case one name -- or ``None``
        for a stage the description does not hold."""
        key = stage_key(stage)
        return next((e for n, e in self.stages.items()
                     if stage_key(n) == key), None)

    def group_of(self, stage: str) -> Optional[int]:
        """The group ``stage`` is in, or ``None`` for a stage not held --
        the one lookup."""
        e = self.entry(stage)
        return e.group if e is not None else None

    def members(self, number: int) -> List[str]:
        """The stages group ``number`` holds, in the ladder's order."""
        return [n for n, e in self.stages.items() if e.group == number]

    @property
    def numbers(self) -> List[int]:
        """Every group, in order."""
        return sorted({e.group for e in self.stages.values()})

    def with_groups(self, groups: Mapping[str, int]) -> "JobRequest":
        """This request with every stage's group as ``groups`` says -- the
        numbers a writer gives a ladder (`group.default_groups`)."""
        return replace(self, stages={
            n: replace(e, group=int(groups[n])) for n, e in self.stages.items()})

    # ----- the run card (§ 3.2) ---------------------------------------- #

    def run_condition(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """What a rung's run uses -- the calculation's card with the rung's
        own over it, FIELD BY FIELD (§ 3.2): a stage naming only `mpi_np`
        keeps the calculation's solver and its thread count.

        A CALCULATION-WIDE VALUE IS A RUNG'S WHERE THE RUNG READS IT -- the
        item's ``stages`` (`template.reads`), as an override is: a transport
        ladder's transmission takes none of the calculation's GPU values,
        its `tbtrans` having no GPU path.  With no ``stage``, the
        calculation's card alone -- what a benchmark's trials take (§ 3.4).
        """
        out = dict(self.execution or {})
        if stage is None:
            return out
        out = {k: v for k, v in out.items() if self._rung_reads(k, stage)}
        e = self.entry(stage)
        if e is not None:
            out.update(dict(e.execution or {}))
        return out

    def _rung_reads(self, name: str, stage: str) -> bool:
        """Whether the rung ``stage`` reads run-card ``name`` -- a catalogue
        item it reads (`template.reads`), or a name no item carries, the
        run's own scheduler asks (:data:`LANE_ASKS`)."""
        from .template import catalogue, one, reads
        try:
            item = one(catalogue(), name, engine=str(self.engine))
        except KeyError:
            return True                  # not an item: the lane's own asks
        # AN ITEM NOT FOR THIS ENGINE is the preflight's refusal.
        return item is None or reads(item, str(self.engine),
                                     self.calculation, stage)

    # ----- persistence --------------------------------------------------- #

    def to_dict(self) -> dict:
        """This request as the JSON object § 3 specifies -- every optional
        block omitted when it says nothing, so *unstated* has one spelling
        on disk."""
        out: Dict[str, Any] = {"schema": SCHEMA}
        if self.allocation:
            a = self.allocation
            out["allocation"] = {k: v for k, v in (("domain", a.domain),
                                                   ("time", a.time),
                                                   ("mem", a.mem)) if v}
            # A SWITCH IS WRITTEN WHEN IT IS SAID, false included.
            if a.gpu_binding is not None:
                out["allocation"]["gpu_binding"] = a.gpu_binding
        if self.execution:
            out["execution"] = dict(self.execution)
        out["stages"] = {
            n: {"group": e.group,
                **({"execution": dict(e.execution)} if e.execution else {})}
            for n, e in self.stages.items()}
        if self.bench:
            out["bench"] = {k: list(v) for k, v in self.bench.items()}
        if self.notify is not None:
            n = self.notify
            out["notify"] = {
                "on_scf_converged": bool(n.on_scf_converged),
                "every_hours": n.every_hours,
                "channels": ["*"] if n.channels is None else list(n.channels),
                "report": ["*"] if n.report is None else list(n.report),
            }
        return out


# --------------------------------------------------------------------- #
#  Refusals                                                             #
# --------------------------------------------------------------------- #

#: What a refusal calls the thing it is refusing -- the file, or the body
#: a page sent.
_SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "job_request_refusal_source", default=FILENAME)


def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(
        f"{_SOURCE.get()}{': ' + where if where else ''}: {msg}")


def _as_object(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _refuse(f"expected an object, got {type(value).__name__}", where=where)
    return value


def _check_keys(obj: Mapping[str, Any], allowed: Tuple[str, ...], *,
                where: str, note: str = "") -> None:
    """An unknown key is refused, not ignored -- an ignored key is a request
    quietly different from the one written -- with the near-miss named, or
    where the key belongs (``note``)."""
    for key in obj:
        if key in allowed:
            continue
        near = difflib.get_close_matches(key, allowed, n=1, cutoff=0.7)
        hint = f" -- did you mean {near[0]!r}?" if near else \
               f" (known keys: {', '.join(allowed)}){note}"
        _refuse(f"unknown key {key!r}{hint}", where=where)


#: A key of the description written here, said where it belongs.
_DESCRIPTION_NOTE = (". What is computed -- the stages' overrides, what "
                     "varies, the structure -- is the description's, "
                     "task.json (execution/job-request.md 1)")


# --------------------------------------------------------------------- #
#  Reading                                                              #
# --------------------------------------------------------------------- #

def from_dict(obj: Mapping[str, Any], description: "Task") -> JobRequest:
    """Parse a job request against ``description``, refusing rather than
    guessing: the schema first, then every key by name, then its stages
    against the description's -- each stage held, by its name, and no
    other (`job-request.md` § 3)."""
    if not isinstance(obj, Mapping):
        _refuse(f"expected a JSON object, got {type(obj).__name__}")
    check_schema(str(obj.get("schema") or ""), SCHEMA, label=FILENAME)
    _check_keys(obj, _TOP_KEYS, where="", note=_DESCRIPTION_NOTE)
    engine, calc = str(description.engine), str(description.calculation)
    return JobRequest(
        stages=_stages_from_obj(obj, description),
        allocation=_allocation_from_obj(obj),
        execution=_execution_from_obj(obj),
        bench=bench_from_obj(obj),
        notify=_notify_from_obj(obj, engine=engine, calculation=calc),
        engine=engine, calculation=calc)


def _stages_from_obj(obj: Mapping[str, Any], description: "Task"
                     ) -> Dict[str, StageRequest]:
    """``stages`` -> every stage of the description, in its order, by the
    description's own spelling of its name; a key that is no stage of it,
    two keys that are one name, or a stage with no entry, each refused
    naming both files -- one Save writes both (§ 7)."""
    from .task import FILENAME as DESCRIPTION
    raw = _as_object(obj.get("stages", {}), where="stages")
    if "stages" not in obj:
        _refuse(f"no 'stages' -- every stage of {DESCRIPTION} has its entry "
                f"here, its group (execution/job-request.md 3)")
    names = [st.name for st in (description.stages or ())]
    by_key = {stage_key(n): n for n in names}
    seen: Dict[str, str] = {}
    entries: Dict[str, Mapping[str, Any]] = {}
    for key, body in raw.items():
        k = stage_key(str(key))
        if k not in by_key:
            _refuse(f"stage {key!r} is no stage of {DESCRIPTION} (its "
                    f"stages: {', '.join(names) or 'none'}) -- the two files "
                    f"hold one ladder (execution/job-request.md 3)",
                    where="stages")
        if k in seen:
            _refuse(f"two entries, {seen[k]!r} and {key!r}, are one stage -- "
                    f"names are compared case aside", where="stages")
        seen[k] = str(key)
        entries[by_key[k]] = body
    missing = [n for n in names if n not in entries]
    if missing:
        _refuse(f"no entry for {', '.join(repr(n) for n in missing)} -- every "
                f"stage of {DESCRIPTION} states the job it runs in here "
                f"(execution/job-request.md 3.3)", where="stages")
    out: Dict[str, StageRequest] = {}
    for name in names:
        where = f"stages {name!r}"
        body = _as_object(entries[name], where=where)
        _check_keys(body, _STAGE_KEYS, where=where,
                    note=". A stage's overrides are the description's "
                         "(task.json)")
        if "group" not in body:
            _refuse("no 'group' -- every stage states the job it runs in "
                    "(execution/job-request.md 3.3)", where=where)
        out[name] = StageRequest(group=body["group"],
                                 execution=_execution_from_obj(body,
                                                               where=where))
    return out


def bench_from_obj(obj: Mapping[str, Any]) -> Dict[str, Tuple[Any, ...]]:
    """``bench`` -- field name -> the points to try (§ 3.4).

    Absent is the common state: no benchmark planned.  Present and empty is
    refused, because an empty plan and no plan are one state said two ways.
    SHAPE ONLY: whether a name is a field the engine declares sweepable is a
    membership question, the preflight's (`validation/task.py`)."""
    raw = obj.get("bench")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or not raw:
        _refuse("'bench' is present but not a non-empty object. Omit the key "
                "entirely when no benchmark is planned -- absent and empty "
                "would be two spellings of one state "
                "(execution/job-request.md 3.4)")
    out: Dict[str, Tuple[Any, ...]] = {}
    for name, points in raw.items():
        if not isinstance(points, (list, tuple)) or not points:
            _refuse(f"bench[{name!r}] must be a non-empty list of points to "
                    f"try, got {points!r}")
        if any(isinstance(p, (list, tuple, dict)) for p in points):
            _refuse(f"bench[{name!r}] takes scalar points; got a nested value")
        out[str(name)] = tuple(points)
    return out


def _allocation_from_obj(obj: Mapping[str, Any]) -> Allocation:
    """``allocation`` -> :class:`Allocation`; absent is an empty one.

    ``time`` and ``mem`` have a spelling this record defines (SLURM's), so a
    value that cannot be read AT ALL is refused now, by name, and one that
    can is stored in the record's spelling.  ``domain`` is NOT checked
    against this machine's queues: a request written for one cluster is
    opened on another; the target's record answers it at prep."""
    raw = obj.get("allocation")
    if raw is None:
        return Allocation()
    if not isinstance(raw, Mapping) or not raw:
        _refuse("'allocation' is present but not a non-empty object. Omit "
                "the key entirely when nothing is asked for -- absent and "
                "empty would be two spellings of one state")
    # A LAUNCH SHAPE LANDING HERE IS THE ORDINARY MISTAKE: the refusal names
    # the block it belongs in.
    _check_keys(raw, _ALLOCATION_KEYS + _ALLOCATION_SWITCHES,
                where="allocation",
                note=". Ranks, threads and GPUs are not asked here -- they "
                     "are the run card's (`execution`: \"mpi_np\": 8), and "
                     "a benchmark's axes (`bench`) "
                     "(execution/job-request.md 3.2)")
    for k in _ALLOCATION_KEYS:
        v = raw.get(k)
        if v is not None and not isinstance(v, str):
            _refuse(f"allocation.{k} must be a string -- write it the way "
                    f"you would type it (\"4h\", \"128G\", "
                    f"\"7-00:00:00\"); got {type(v).__name__}")
    for k in _ALLOCATION_SWITCHES:
        v = raw.get(k)
        if v is not None and not isinstance(v, bool):
            _refuse(f"allocation.{k} must be true or false; got "
                    f"{type(v).__name__}")

    def _canon(fn, key):
        v = raw.get(key)
        if not v:
            return ""
        try:
            return fn(v) or ""
        except ValueError as exc:
            _refuse(f"allocation.{key}: {exc}")
    return Allocation(domain=str(raw.get("domain") or ""),
                      time=_canon(canonical_time, "time"),
                      mem=_canon(canonical_mem, "mem"),
                      gpu_binding=raw.get("gpu_binding"))


def _notify_from_obj(obj: Mapping[str, Any], *, engine: str,
                     calculation: str) -> Optional[Notify]:
    """``notify`` -> :class:`Notify`, or ``None`` with no block (§ 3.5).

    Both triggers are refused by TYPE rather than coerced: ``"true"`` is not
    a boolean and ``"6"`` is not a number.  A period is a positive, finite
    number of hours, or ``null`` for never."""
    raw = obj.get("notify")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw:
        _refuse("'notify' is present but not a non-empty object. Omit the "
                "key when nothing should be reported -- absent and empty "
                "would be two spellings of one state")
    _check_keys(raw, _NOTIFY_KEYS, where="notify")
    # A BLOCK STATES ALL FOUR: "every channel" and "every field" are
    # written `["*"]`, never a key left out.
    missing = [k for k in _NOTIFY_KEYS if k not in raw]
    if missing:
        _refuse(f"notify states no {', '.join(repr(k) for k in missing)} -- "
                f"a notify block states all four: on_scf_converged, "
                f"every_hours (null for never), channels ([\"*\"] for every "
                f"channel), report ([\"*\"] for every field)")

    scf = raw["on_scf_converged"]
    if not isinstance(scf, bool):
        _refuse(f"notify.on_scf_converged must be true or false, not "
                f"{type(scf).__name__} -- write a JSON boolean, not a string")

    hours = raw["every_hours"]
    if hours is not None:
        if isinstance(hours, bool) or not isinstance(hours, (int, float)):
            _refuse(f"notify.every_hours must be a number of HOURS, not "
                    f"{type(hours).__name__} -- write 6, not \"6\" or "
                    f"\"6h\"")
        hours = float(hours)
        if hours != hours or hours in (float("inf"), float("-inf")):
            _refuse("notify.every_hours must be a finite number of hours")
        if hours <= 0:
            _refuse(f"notify.every_hours must be a positive number of hours "
                    f"(got {hours:g}) -- null is never")

    # EVERY, NONE, OR THESE -- three states, each written (`run-reports.md`
    # § 3.0): `["*"]` is every channel the running machine has (`None`
    # here), `[]` is none, a list is those.
    chans = _every_or_these(
        raw["channels"], "channels", "channel NAMES",
        lambda item: (None if is_channel_name(item) else
                      f"{item!r} is not a channel name -- letters, digits, "
                      f"'-' and '_'. A name is all that travels; the address "
                      f"and key stay on the machine"))
    # THE ONE DECLARATION answers, for THIS calculation: a field its runs
    # can never state is refused by name (§ 3.5).
    report = _every_or_these(
        raw["report"], "report", "report FIELD NAMES",
        lambda item: _report_fields.refusal(str(item), engine, calculation))
    return Notify(on_scf_converged=scf, every_hours=hours, channels=chans,
                  report=report)


def _every_or_these(got, key: str, what: str, refusal
                    ) -> Optional[Tuple[str, ...]]:
    """A notify list as written -- ``["*"]`` every (``None``), ``[]`` none,
    names those -- each name asked of ``refusal`` (a sentence, or ``None``
    for a name it takes)."""
    if not isinstance(got, list):
        _refuse(f"notify.{key} must be a list of {what}, not "
                f"{type(got).__name__} -- [\"*\"] for every one, [] for none")
    if "*" in got:
        if got != ["*"]:
            _refuse(f"notify.{key}: \"*\" is every one, and stands alone -- "
                    f"not beside names ({got!r})")
        return None
    names: List[str] = []
    for item in got:
        why = refusal(item)
        if why is not None:
            _refuse(f"notify.{key}: {why}")
        if item not in names:
            names.append(item)
    return tuple(names)


def _execution_from_obj(obj: Mapping[str, Any], *,
                        where: str = "") -> Dict[str, Any]:
    """``execution`` -> ``{name: one value}`` (§ 3.2).

    The arity is what tells it from `bench`: several values per axis is a
    plan to measure, one value per parameter is what to use.  So a LIST here
    is refused by name, the block it belongs in said out loud.  MEMBERSHIP
    is the preflight's, as for `bench`; the shape is checked here."""
    raw = obj.get("execution")
    if raw is None:
        return {}
    at = f"{where} execution" if where else "execution"
    out: Dict[str, Any] = {}
    for key, val in _as_object(raw, where=at).items():
        if isinstance(val, (list, tuple)):
            _refuse(f"{key!r} takes ONE value here, not a list -- a list of "
                    f"points to try is `bench`'s shape, and the two are "
                    f"separate blocks on purpose "
                    f"(execution/job-request.md 3.2)", where=at)
        if isinstance(val, Mapping):
            _refuse(f"{key!r} takes a scalar, got an object", where=at)
        out[key] = val
    return out


def path_of(base) -> Path:
    """``<base>/job-request.json`` -- for its writer, and for `jobset
    migrate`, which writes it from an older description (`architecture.md`
    A16)."""
    return Path(base) / FILENAME


def of(base, description: "Task") -> Optional[JobRequest]:
    """THE JOB REQUEST'S DOOR (`architecture.md` § 3.2, A16): the folder's
    ``job-request.json`` read and checked against ``description`` -- the
    one its caller holds -- or ``None`` when the folder holds none."""
    path = path_of(base)
    if not path.is_file():
        return None
    token = _SOURCE.set(str(path))
    try:
        return from_dict(read_json(path), description)
    finally:
        _SOURCE.reset(token)


def from_text(text: str, description: "Task",
              said_as: str = FILENAME) -> JobRequest:
    """A job request that is not yet a file -- the Task setup page's body
    (`task-setup.md` § 8) -- read by the one reader against ``description``;
    ``said_as`` names it in a refusal."""
    token = _SOURCE.set(said_as)
    try:
        try:
            obj = json.loads(text)
        except ValueError as exc:
            _refuse(f"not JSON -- {exc}")
        return from_dict(obj, description)
    finally:
        _SOURCE.reset(token)


def fresh(description: "Task", groups: Mapping[str, int]) -> JobRequest:
    """A new calculation's request: every stage of ``description`` in the
    group ``groups`` gives it, and nothing else stated -- what `molbuilder
    describe` writes when no flag states more (`job-request.md` § 9)."""
    return JobRequest(
        stages={st.name: StageRequest(group=int(groups[st.name]))
                for st in (description.stages or ())},
        engine=str(description.engine),
        calculation=str(description.calculation))


def write(base, request: JobRequest):
    """Write the folder's ``job-request.json`` atomically
    (``persist.write_json``), whole."""
    return write_json(path_of(base), request.to_dict())


__all__ = ["SCHEMA", "FILENAME", "LANE_ASKS", "LANE_ASK_ROWS",
           "Allocation", "Notify", "StageRequest", "JobRequest",
           "bench_from_obj", "from_dict", "from_text", "fresh", "of",
           "path_of", "write"]
