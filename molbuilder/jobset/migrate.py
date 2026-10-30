"""Rewrite a calculation an older molbuilder wrote -- the one road for a file
a reader refuses naming this command.  Every record is read once
(:class:`_Records`), every step decided before any file is written, and a
refusal of any step writes nothing.  **Each old file rewritten is kept
beside its new one, once, as** ``<name>.pre-<rule>`` -- the earliest rule it
was written before (:func:`_kept_as`): a plan's code (``m6``, ``w57``), a
day, or a release.  A file renamed keeps every byte, so nothing is kept
beside it.  Each change is said, with its rule.

**The job request** (2.27.0, :func:`_split`): a description written before
holds how it runs inside `task.json` (`molbuilder/task@1`) -- its
`allocation`, run cards, groups, `bench` and `notify` -- and its run
settings' values in its template.  They move into `job-request.json`, a file
of its own (`execution/job-request.md` § 8): the template's run-setting
values onto the calculation's run card, where it states none, but a value
the template records *not chosen* that is the catalogue's default now; and
`restart` and `density_fit`, which say what is computed, off the cards -- the
calculation's into the template, a stage's into its overrides.  **And run
settings leave the template** (2.28.0, `engines/template.md` § 2): every
run-setting item a template holds, valueless or not, is taken out, by what the
file holds and never by its version -- a value onto the calculation's run card
of the job request beside it (:func:`_onto_the_card`).

**The records** (:func:`_migrate_records`).  `task.json`: its `calculation`
(W57), a notify block's four values and *never* as ``null`` (W57); a stage's
`enabled` (2026-10-03) -- a ``true`` removed, a stage switched off refused by
name, since whether it runs is the person's; `centre_junction` removed
(2.6.0, the junction enters its calculation centred); a transport
description's voltages, ``[0.0]`` where it stated none (2.16.3); the
low-bias switch from the word it replaced (2026-10-08); every stage's
`group` in a description written before 2.27.0 (2.7.0,
:func:`_migrate_groups`) -- a prepared stage its job's, the rest the number
above where the ladder allows it; a stage its pair's record retired (the kind
table's `retired`: a SIESTA vibration's `relax`, 2.42.0) folded into the
others, its own role's overrides carried, and its job-request row dropped,
each tagged with that release.  Each `job-set.json` --
the root's and each sweep's: a job's `point`, `finish`, `resumes`,
`placement` and `members` (W57; `members` written as `group` before 2.7.0),
its `phases` (2.42.0) and its `scratch_dir` (2.22.0), each
with the meaning its absence had; the program it runs, onto the job
(`Job.program`, 2.27.0), and a placement's source ``description`` said
``allocation``.

**The machine's record** (:func:`_migrate_machine_record`): a calculation's
copy names its scheduler as a block since 2.10.1 (`configuration.md` § 5
M-2) -- the word it held becomes its kind, each command its kind's own name,
run from PATH as before.

**The run files**: a transport lead's files on the calculation's label
rather than its own stem (:func:`_migrate_lead_names`) -- every file before
2.17.0, its prep's log before 2.23.2 -- renamed, and the plan's deck names
with them.  A stage whose run script was written for the old names is said
with its way on.  A run of points' index as 2.42-2.54 wrote it -- a
vibration's ``.phases.json``, a transport rung's ``.points.json`` of its
earlier schema -- rewritten into the one index (:func:`_migrate_run_indexes`,
2.55.0).

**The template** (:func:`_decide_template`): the electronic state's items
(M6, ``science/chemistry-correctness.md`` § 2a) -- PySCF's ``spin`` and the
R/U inside its ``method``, SIESTA's ``spin_treatment`` spellings and
``spin_total`` -- each old value a STATED value, never a blank the structure
would now decide, a fractional ``Spin.Total`` and a stage override of the
state refused by name; a 0 that meant *not set* before 2.1.0 on the two
NEGF broadenings, written blank -- a rung's override taken out too
(`engines/transport.md` § 2a.13); an item renamed kept under its new name
(`template.RENAMED_ITEMS`), and an item a release retired taken out where
its value runs as before without it (`template.RETIRED_ITEMS`,
`Retired.runs`), any other value refused by name -- in the template, a
stage's own values and `varies` alike; a SIESTA vibration's Raman selector
before 2.45.0 stated as *Compute IR*, *Compute Raman* and *Raman for*
(:func:`_map_raman`), an explicit list of modes and a stage's own value of
it refused by name.  **What the rewrite carries is the
old file's, whole**: every value, every blank and each item's source as the
file records it (`engines/template.md` § 5); what it cannot carry is said --
a value of an item no longer of this kind, the comments a person wrote
inside the file.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional,
                    Tuple)

if TYPE_CHECKING:                      # annotations only
    from ..calculation import Calculation
    from ..template_file import TemplateFile


#: What a calculation holding nothing an older molbuilder wrote IS -- said
#: when a migration finds nothing to rewrite (:func:`migrate_state`).
CURRENT = ("nothing to migrate: it holds nothing an older molbuilder wrote "
           "-- its records are whole, its record of its machine is this "
           "molbuilder's, its leads' run files are on their own stem, and "
           "its template carries no item an older molbuilder meant "
           "otherwise, renamed or a release retired, and its runs of points "
           "keep this molbuilder's index")


class MigrateError(ValueError):
    """The calculation cannot be migrated as it stands -- the message says
    why and what to do."""


#: PySCF's old ``method`` was the SCF class: (method, spin treatment).
_PYSCF_CLASS = {
    "RKS": ("DFT", "restricted"), "UKS": ("DFT", "unrestricted"),
    "RHF": ("HF", "restricted"), "UHF": ("HF", "unrestricted"),
}

#: SIESTA's old ``spin_treatment`` spellings -> the engine-neutral words.
#: FROZEN, the pre-M6 template vocabulary: it happens to be SIESTA's own
#: keyword spelling (`siesta/layout.SPIN_SPELLING`'s values), but it is
#: what old FILES say, and it must not follow that table if SIESTA's
#: spelling ever changes.
_SIESTA_TREATMENT = {
    "non-polarized": "restricted", "polarized": "unrestricted",
    "non-colinear": "non-collinear", "spin-orbit": "spin-orbit",
}

def _renamed_conflict(old: str, old_value: Any, new: str, new_value: Any,
                      where: str = "the template") -> MigrateError:
    return MigrateError(
        f"{where} says {old} = {old_value!r} and {new} = {new_value!r}: one "
        f"item, renamed (`template.RENAMED_ITEMS`), at two values.  Decide "
        f"which the calculation carries, delete the other, and run the "
        f"migration again; nothing was written.")


def _conflict(old: str, old_value: Any, new: str, new_value: Any) -> MigrateError:
    return MigrateError(
        f"the template says {old} = {old_value!r} (written before the "
        f"electronic state's items) and {new} = {new_value!r}, and they do "
        f"not describe the same run.  Decide which the calculation carries, "
        f"delete the other from the template, and run the migration again.")


def _map_state(engine: str, vals: Dict[str, Any]
               ) -> Tuple[Dict[str, Any], List[str]]:
    """The old values -> the four items, and a line per change.  A current
    value the file already states is kept; an old one that disagrees with it
    is refused by name, never silently overwritten."""
    out: Dict[str, Any] = {}
    said: List[str] = []
    if engine == "pyscf":
        treatment = vals.get("spin_treatment")
        # An old file with no method item ran PySCF's default, RKS -- unless
        # it already states a treatment, which is then the file's word and
        # is kept (a conflict names only a value the file contains).
        old = vals.get("method", "RKS" if treatment is None else None)
        if old in _PYSCF_CLASS:
            method, implied = _PYSCF_CLASS[old]
            if treatment is not None and treatment != implied:
                raise _conflict("method", old, "spin_treatment", treatment)
            out["method"], out["spin_treatment"] = method, implied
            said.append(f"method = {old!r} -> method = {method!r}, "
                        f"spin_treatment = {implied!r}")
        if "spin" in vals:
            count = int(vals["spin"])
            new = vals.get("unpaired_electrons")
            if new is not None and new != count:
                raise _conflict("spin", vals["spin"], "unpaired_electrons", new)
            out["unpaired_electrons"] = count
            said.append(f"spin = {vals['spin']!r} -> unpaired_electrons = "
                        f"{count}")
        elif "unpaired_electrons" not in vals:
            # No spin item: the run took PySCF's default, 2S = 0.
            out["unpaired_electrons"] = 0
            said.append("spin unset -> unpaired_electrons = 0 (PySCF's "
                        "default, what ran)")
    else:
        old = vals.get("spin_treatment", "non-polarized")
        if old in _SIESTA_TREATMENT:
            treatment = _SIESTA_TREATMENT[old]
            out["spin_treatment"] = treatment
            said.append(f"spin_treatment = {old!r} -> {treatment!r}")
        else:
            treatment = old
        total = vals.get("spin_total")
        new = vals.get("unpaired_electrons")
        if total is None:
            if new is None and treatment == "unrestricted":
                # A polarized run with no total spin let the moment float.
                out["unpaired_electrons"] = "free"
                said.append("spin_total unset under polarized -> "
                            "unpaired_electrons = 'free' (the moment floated)")
        elif treatment == "restricted":
            # SIESTA read the number and ignored it: nothing to carry.
            said.append(f"spin_total = {total!r} under non-polarized was "
                        f"ignored by SIESTA -> dropped")
        elif float(total) != int(float(total)):
            raise MigrateError(
                f"spin_total = {total!r} is a fixed moment, not a whole "
                f"number of unpaired electrons, and unpaired_electrons is a "
                f"count (science/chemistry-correctness.md § 2a.1).  State the "
                f"count this calculation should hold, or 'free', by editing "
                f"the template, then run the migration again.")
        else:
            count = int(float(total))
            if new is not None and new != count:
                raise _conflict("spin_total", total, "unpaired_electrons", new)
            out["unpaired_electrons"] = count
            said.append(f"spin_total = {total!r} -> "
                        f"unpaired_electrons = {count}")
    return out, said


def _map_raman(vals: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """A SIESTA vibration described before 2.45.0, its Raman and infrared
    as the settings both engines share say them (`engines/vibration.md`
    § 5.11 § 9): its ``es_mode_selection`` decided the Raman -- ``skip``,
    or ``explicit`` with an empty list, none; ``all`` every mode within the
    window -- and its charges were always written.  An ``explicit`` list of
    modes is refused by name: mode numbers have no range to become."""
    from ..spectra.methods import frequency_range_words
    from ..spectra.selection import parse_index_list
    sel = vals.get("es_mode_selection", "skip")
    listed = str(vals.get("es_explicit_indices") or "")
    out: Dict[str, Any] = {"compute_ir": True}
    said = ["compute_ir = True (the charges were always written)"]
    if sel == "explicit" and parse_index_list(listed, first=1):
        raise MigrateError(
            f"es_mode_selection = 'explicit' ({listed!r}) picked this SIESTA "
            f"vibration's Raman modes by number, and Raman for takes every "
            f"mode or a frequency range -- mode numbers have no range to "
            f"become, so nothing was written.  Edit the template: "
            f"es_mode_selection = 'all', with freq_min_cm1 / freq_max_cm1 "
            f"for a range, or 'skip' for no Raman; then run the migration "
            f"again (engines/vibration.md 3.1).")
    if sel != "all":
        out["compute_raman"] = False
        said.append(f"es_mode_selection = {sel!r}"
                    + (" with no mode listed" if sel == "explicit" else "")
                    + " -> compute_raman = False (no optical call ran)")
        return out, said
    lo, hi = vals.get("freq_min_cm1"), vals.get("freq_max_cm1")
    out.update(compute_raman=True,
               raman_modes=("range" if lo is not None or hi is not None
                            else "all"),
               raman_freq_min_cm1=lo, raman_freq_max_cm1=hi)
    where = frequency_range_words(lo, hi)
    said.append("es_mode_selection = 'all' -> compute_raman = True, Raman "
                "for " + (f"the modes {where}" if where else "every mode"))
    return out, said


#: WHAT A KEY AN OLDER FILE LEFT OUT MEANT, with the rule it was written
#: before -- written in its place, so the rewritten file says what the old
#: one said by omitting it, and the old file kept as that rule's
#: (:func:`_kept_as`).
_TASK_ABSENT = {"calculation": ("optimization", "w57")}
#: A notify block's four, each left out meaning what it did (W57).
_NOTIFY_ABSENT = {"on_scf_converged": False, "every_hours": None,
                  "channels": ["*"], "report": ["*"]}
_JOB_ABSENT = {"point": ({}, "w57"), "finish": (None, "w57"),
               "resumes": (True, "w57"), "placement": (None, "w57"),
               "members": (None, "2026-10-07"),
               # A PREP BEFORE 2.42.0 knew no run in phases.  (A row's
               # `asked`, written 2.7.0-2.40.2, and its `written_at`,
               # written 2.7.0-2.41.0 -- each read by nothing since -- are
               # left as they lie: `Job.from_dict` reads past them.)
               "phases": (None, "2.42.0"),
               # A PREP BEFORE 2.22.0 kept no scratch folder: its device
               # deck names the leads' files loose in the run, and a launch
               # again of it empties nothing (`engines/transport.md`
               # § 2a.11); a prep again gives the stage the folder.
               "scratch_dir": (None, "2.22.0")}
#: A job's key an older file wrote under another name -- renamed, its value
#: kept: the members of the job it was prepared in (2.7.0).
_JOB_RENAMED = {"group": ("members", "2.7.0")}

#: THE RELEASE THE JOB REQUEST BECAME A RECORD OF ITS OWN
#: (`execution/job-request.md` § 8): the rule a description holding it and
#: a plan whose resources name the program were written before.
_SPLIT = "2.27.0"
#: THE RELEASE RUN SETTINGS LEFT THE TEMPLATE (`engines/template.md` § 2):
#: the rule a template holding one, valueless or not, was written before.
_RUN_OUT = "2.28.0"
#: The description's schema before the split.
_TASK_AT_1 = "molbuilder/task@1"
#: The keys of the job request a description held before it.
_JOB_KEYS = ("allocation", "execution", "bench", "notify")
#: What a run card held before that says what is computed: moved off the
#: cards, the calculation's into the template, a stage's into its overrides
#: (`execution/job-request.md` § 2).
_COMPUTED_ON_CARDS = ("restart", "density_fit")

#: The rules named by their plan's code, at the day each took effect; a
#: rule named by its day says it itself, and every numbered release came
#: after them all (2.0.0, 2026-10-11).
_RULE_DAYS = {"m6": "2026-09-28", "w57": "2026-10-06"}


def _kept_as(path: Path, rules) -> Path:
    """Where an old file is kept beside its rewrite, once -- the catalogue's
    name for it (`runfiles.KEPT_FILE`, ``<name>.pre-<rule>``): the EARLIEST
    of the ``rules`` it was written before, each a plan's code
    (:data:`_RULE_DAYS`), a day or a release (the template's retired and
    renamed items carry either); two rules of one day are told apart by
    name, so one file is kept under one name on every run."""
    import re
    from ..template import version_key

    def when(rule: str):
        day = _RULE_DAYS.get(rule, rule)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            return (0, day, (), rule)
        return (1, "", version_key(rule), rule)
    from ..runfiles import KEPT_FILE, fixed_name
    return path.with_name(fixed_name(KEPT_FILE, kept=path.name,
                                     rule=min(rules, key=when)))


class _Writes(dict):
    """EVERY FILE THIS MIGRATION REWRITES, once: ``{path: (new text, the
    rules it was written before)}``.  Two steps rewriting one file -- the
    records step and the lead names a plan, the records step and the
    template a description -- give one write: the later step's text is
    built on the body the earlier decided, and the rules gather, so the old
    file is kept once, under the earliest (:func:`_kept_as`)."""

    def add(self, path: Path, text: str, rules) -> None:
        _old, had = self.get(path, (None, frozenset()))
        self[path] = (text, had | frozenset(rules))


def _refused_rewrite(path: Path, exc: Exception) -> "MigrateError":
    """A rewritten record its reader would still refuse -- said once."""
    return MigrateError(f"{path} rewritten would still be refused, so "
                        f"nothing was written: {exc}")


#: Not yet asked (:attr:`_Records._template_decision`).
_UNDECIDED = object()


def _read_raw(path: Path) -> dict:
    """A record's JSON as it lies, for a rewrite -- one that does not read
    as JSON at all is no older molbuilder's, and is the migration's refusal,
    never a traceback."""
    from ..persist import read_json
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        raise MigrateError(f"{path} does not read ({exc}); nothing was "
                           f"written.") from None


@dataclass
class _Records:
    """EVERY RECORD OF THE CALCULATION, READ ONCE, RAW -- migrate's own
    exception to A16 (`architecture.md`): the readers refuse a file an
    older molbuilder wrote, so migrate reads the bytes through the owners'
    paths (`task.path_of`, `JobSet.path_of`) and the template's door, and
    decides every rewrite on these bodies; every step reads them here.  The
    machine's record is read once by its own step, which rewrites it
    (:func:`_migrate_machine_record`); a run's launch record is only read,
    through its owner (`materialize.newest_launched`) -- no step rewrites
    one.  The write keeps each old file, byte for byte, beside its new one."""
    base: Path
    #: `task.json` as it lies, decided in place by the records step.
    task: dict
    #: Each plan's body by its folder -- the root's and each sweep's bench
    #: container's (`materialize.sweep_set_dirs`) -- decided in place.
    jobsets: Dict[Path, dict]
    #: The template (`template_file.of`), read by the first step that knows the
    #: calculation's label -- derived, never stored (`Task.label`) -- and
    #: held for every later one; ``None`` when there is none.
    template: Optional["TemplateFile"] = None
    _template_read: bool = False
    #: The template's migration, decided once (:func:`_migrate_template`):
    #: ``_UNDECIDED`` until asked, then the decision or ``None``.
    _template_decision: Any = _UNDECIDED
    #: What the calculation's run card held that says what is computed
    #: (:data:`_COMPUTED_ON_CARDS`), taken off it for the template -- set by
    #: the records step before the template's decision is asked.
    from_card: Dict[str, Any] = None

    #: The job request as the records step decided it -- ``None`` until
    #: then (:func:`_migrate_records`).
    request: Any = None
    #: The stages the records step folded into another (the kind table's
    #: `retired`), each with the release that retired it: what the job
    #: request's rows drop with them.
    folded_stages: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def read(cls, base: Path) -> "_Records":
        from .. import calcdirs
        from .. import task as _task
        from ..calculation import refusal
        from .materialize import sweep_set_dirs
        from .model import JobSet
        if not calcdirs.described(base):
            raise MigrateError(refusal(base))
        obj = _read_raw(_task.path_of(base))
        # ANOTHER LAYOUT, refused before any rule is asked: a layout is
        # never rewritten (`engines/stages.md` § 6.7).
        try:
            _task.refuse_another_layout(obj)
        except ValueError as exc:
            raise MigrateError(str(exc)) from None
        jobsets: Dict[Path, dict] = {}
        for folder in [base, *sweep_set_dirs(base)]:
            path = JobSet.path_of(folder)
            if path.is_file():
                jobsets[folder] = _read_raw(path)
        return cls(base=base, task=obj, jobsets=jobsets, from_card={})

    def calculation(self, task) -> "Calculation":
        """The calculation as this migration decided it -- its description,
        its template and its root plan, the decided bodies -- for the steps
        that compose a way back (`commands.way_back`).  The one place a
        `Calculation` is composed outside its door: migrate alone reads its
        records raw (A16)."""
        from ..calculation import Calculation
        from .model import JobSet
        body = self.jobsets.get(self.base)
        try:
            js = JobSet.from_dict(body) if body is not None else None
        except ValueError as exc:
            raise MigrateError(
                f"{JobSet.path_of(self.base)} does not read as this "
                f"molbuilder writes it, and no rule here rewrites it; "
                f"nothing was written: {exc}") from None
        return Calculation(base=self.base, task=task,
                           template=self.template_for(task.label),
                           jobset=js, request=self.request)

    def template_for(self, label: str) -> Optional["TemplateFile"]:
        """The calculation's template, read once (`template_file.of`); a folder
        with two, or one under another name, is refused by name."""
        if not self._template_read:
            from ..template_file import of as template_of
            try:
                self.template = template_of(self.base, label, raw=True)
            except ValueError as exc:
                raise MigrateError(str(exc)) from exc
            self._template_read = True
        return self.template


def _migrate_groups(rec: "_Records", obj: dict) -> List[str]:
    """Every stage's `group` (2026-10-11, plan § 5aa M1), written into the
    raw description ``obj`` -- a line saying the numbers, or ``[]`` when
    every stage states one.  A stage prepared takes the group its job
    records (`Job.members`; prepared alone, a group of its own); one not
    prepared repeats the number above only where the stage above is not
    prepared either, no stage of that group is one it builds on
    (`ladder.upstream_of`) and their run cards share one allocation
    (`group.card_shares`) -- else one more.  A prepared group whose
    members are not one after another in the ladder cannot be stated by a
    number, and is refused by name: which jobs to keep is the person's.
    A description written since the split (`task@2`) holds no group -- its
    groups are its job request's (`execution/job-request.md` § 3.3)."""
    stages = obj.get("stages")
    if (str(obj.get("schema") or "") != _TASK_AT_1
            or not isinstance(stages, list) or all(
                not isinstance(st, dict) or "group" in st for st in stages)):
        return []
    from ..identity import stage_key
    from ..task import Task
    from .commands import rollback
    from .group import may_join
    base = rec.base
    # THE LADDER'S LINKS do not depend on the numbers: read on the
    # description (:func:`_description_view`), each stage's run card the
    # job request's, every stage a group of its own...
    task = Task.from_dict(_description_view(obj))
    cards = _cards_view(obj, task)
    # ...AS THIS MIGRATION LEAVES IT (:func:`_migrate_template`, every
    # decision before any write): the template it writes, and the stages'
    # own values with an older zero or a retired item taken out -- the links
    # are read on the ladder that resolves, and an older item is refused.
    decision = _migrate_template(rec, task)
    template = rec.template_for(task.label)
    text = (decision.text if decision is not None
            else template.text if template is not None else None)
    task = _edited(task, decision)
    # THE JOB SET AS AN OLDER PREP WROTE IT -- a description with no group
    # is one 2.7.0 never prepared, and before it a job's members were
    # written under `group` (`_JOB_RENAMED`).
    jobs = {stage_key(j["name"]): [stage_key(n) for n in (j.get("group")
                                                            or [j["name"]])]
            for j in ((rec.jobsets.get(base) or {}).get("jobs") or [])}
    names = [st.name for st in task.stages]
    groups: List[int] = []
    members: List[str] = []
    for i, name in enumerate(names):
        job = jobs.get(stage_key(name))
        above = names[i - 1] if i else None
        if job is not None and job[0] != stage_key(name):
            # a later member of a prepared job: the stage above is its
            # member before it, or the job cannot be stated by a number
            if above is None or stage_key(above) not in job:
                raise MigrateError(
                    f"`{name}` was prepared in one job with "
                    f"{', '.join(f'`{n}`' for n in job if n != stage_key(name))}"
                    f", which the ladder does not hold one after another -- "
                    f"a group is stages one after another (engines/stages.md "
                    f"§ 2), so no number states that job.  Which to keep "
                    f"is yours, the migration run again after: remove the "
                    f"stage between from the description, or "
                    + rollback("that prep", base=base))
            groups.append(groups[-1])
            members.append(name)
            continue
        joins = (job is None and above is not None
                 and jobs.get(stage_key(above)) is None
                 and may_join(task, cards, name, members, text))
        groups.append(groups[-1] if joins else (groups[-1] + 1 if groups
                                                else 1))
        members = members + [name] if joins else [name]
    for st, g in zip(stages, groups):
        st["group"] = g
    return [f"stages' groups {' '.join(map(str, groups))} (2026-10-11: "
            f"each stage states the job it runs in -- a prepared one its "
            f"job's, the rest the number above where the ladder allows it "
            f"and the run cards share one allocation)"]


def _description_view(obj: dict) -> dict:
    """``obj`` -- a raw description, before the split or after -- as the
    description alone: what is computed, the job request's keys left out
    (`execution/job-request.md` § 1), at the schema the reader reads -- and
    what a stage's run card said of what is computed (`restart`,
    `density_fit`) among its overrides, as the split leaves it, so every
    step reads the ladder the migration writes: the groups' links and the
    template's decision alike (§ 8)."""
    from ..task import SCHEMA
    out = {k: v for k, v in obj.items() if k not in _JOB_KEYS}
    out["schema"] = SCHEMA
    varies = list(obj.get("varies") or [])
    stages = []
    for st in (obj.get("stages") or []):
        if not isinstance(st, dict):
            stages.append(st)
            continue
        own = st.get("execution") or {}
        moved = {k: own[k] for k in _COMPUTED_ON_CARDS if k in own}
        view = {k: v for k, v in st.items()
                if k not in ("group", "execution")}
        if moved:
            view["overrides"] = {**(view.get("overrides") or {}), **moved}
            varies = list(dict.fromkeys([*varies, *moved]))
        stages.append(view)
    out["stages"] = stages
    if "varies" in obj or varies:
        out["varies"] = varies
    return out


def _cards_view(obj: dict, task):
    """The run cards ``obj`` holds -- the calculation's and each stage's --
    as a job request, each stage a group of its own: what the groups' rule
    reads of what a job shares (`group.card_shares`)."""
    from ..job_request import JobRequest, StageRequest
    by = {st.get("name"): st for st in (obj.get("stages") or [])
          if isinstance(st, dict)}
    return JobRequest(
        stages={st.name: StageRequest(
            group=i + 1, execution=dict((by.get(st.name) or {})
                                        .get("execution") or {}))
                for i, st in enumerate(task.stages)},
        execution=dict(obj.get("execution") or {}),
        engine=str(task.engine), calculation=str(task.calculation))


def _split(rec: "_Records", obj: dict, task) -> Tuple[dict, dict, List[str]]:
    """THE JOB REQUEST, A RECORD OF ITS OWN (`execution/job-request.md`
    § 8): ``(the description, the job request, a line per change)`` from
    ``obj``, a description written before.  Each stage's group and run card
    and the calculation's `allocation`, card, `bench` and `notify` move into
    the job request; a stage card's `restart` and `density_fit` into its
    overrides, `varies` gaining them (the calculation's went to the
    template, :func:`_decide_template`); the template's run-setting values
    onto the calculation's card, where it states none -- but one the
    template records *not chosen* that is the catalogue's default now.  A
    `bench` row naming what is computed is refused by name."""
    from ..job_request import SCHEMA
    said: List[str] = []
    desc = _description_view(obj)
    bench = dict(obj.get("bench") or {})
    measured = sorted(set(bench) & set(_COMPUTED_ON_CARDS))
    if measured:
        raise MigrateError(
            f"task.json's bench measures {', '.join(map(repr, measured))}, "
            f"which say{'s' if len(measured) == 1 else ''} what is computed, "
            f"never how fast -- a benchmark measures run settings alone "
            f"(execution/job-request.md § 2).  Remove "
            f"{'it' if len(measured) == 1 else 'them'} from the bench, then "
            f"run the migration again; nothing was written.")
    card = {k: v for k, v in (obj.get("execution") or {}).items()
            if k not in _COMPUTED_ON_CARDS}
    stages: Dict[str, dict] = {}
    for st, raw in zip(desc["stages"], obj.get("stages") or []):
        own = dict(raw.get("execution") or {})
        moved = {k: own.pop(k) for k in _COMPUTED_ON_CARDS if k in own}
        if moved:
            st["overrides"] = {**(st.get("overrides") or {}), **moved}
            desc["varies"] = list(dict.fromkeys(
                [*(desc.get("varies") or []), *moved]))
            said += [f"stage {st['name']!r} {k} = {v!r}: off its run card, "
                     f"into its overrides -- it says what is computed"
                     for k, v in moved.items()]
        stages[st["name"]] = {"group": raw.get("group"),
                              **({"execution": own} if own else {})}
    said += [f"{k} = {v!r}: off the calculation's run card, into the "
             f"template -- it says what is computed"
             for k, v in sorted(rec.from_card.items())]
    said += _onto_the_card(rec, task, card)
    request = {"schema": SCHEMA,
               **({"allocation": obj["allocation"]}
                  if obj.get("allocation") else {}),
               **({"execution": card} if card else {}),
               "stages": stages,
               **({"bench": bench} if bench else {}),
               **({"notify": obj["notify"]} if obj.get("notify") else {})}
    said.insert(0, "how it runs -- its allocation, run cards, groups, bench "
                   "and reports -- moved into job-request.json, a record of "
                   "its own")
    return desc, request, said


def _onto_the_card(rec: "_Records", task, card: dict) -> List[str]:
    """THE TEMPLATE'S RUN-SETTING VALUES, onto the calculation's run card
    ``card`` (changed in place), where it states none -- but a value the
    template records *not chosen* that is the catalogue's default now
    (`template.default_of`) -- a line each (`execution/job-request.md`
    § 8).  A run setting by the catalogue's word now
    (`template.run_settings`), never by the older template's own category:
    `restart` and `density_fit` were run settings there, and stay in the
    template, what is computed.  What the file holds decides, never its
    version."""
    from .. import template as _T
    template = rec.template_for(task.label)
    if template is None:
        return []
    engine, kind = str(task.engine), str(task.calculation)
    run_settings = _T.run_settings(engine)
    said: List[str] = []
    for it in _T.read_template(template.text).items:
        if it.name not in run_settings or it.value is None or it.name in card:
            continue
        if (it.source == "default"
                and it.value == _T.default_of(
                    _T.one(_T.catalogue(), it.name, engine=engine), kind)):
            said.append(f"{it.name} = {_T.shown(it.value)}: the catalogue's "
                        f"default, not chosen -- taken out of the template; "
                        f"a run takes it there still")
            continue
        card[it.name] = it.value
        said.append(f"{it.name} = {_T.shown(it.value)}: out of the template, "
                    f"onto the calculation's run card in job-request.json")
    return said


def _migrate_records(rec: "_Records"):
    """`task.json` and every `job-set.json` of the calculation, each key an
    older file left out given the meaning its absence had, and a key a
    release retired removed -- on the bodies read once (:class:`_Records`),
    since the readers refuse such a file, and each rewritten file checked by
    its reader.  Nothing is written here: returns what changed (a line
    each), the files to write (:class:`_Writes`) and the description as
    rewritten; the bodies in ``rec`` are the decided ones."""
    import json

    from ..persist import json_text
    from ..task import FILENAME as TASK_FILENAME
    from ..task import Task, never_as_zero, path_of
    from .model import JobSet
    base = rec.base
    said: List[str] = []
    writes = _Writes()

    task_path = path_of(base)
    obj = rec.task
    lines: List[Tuple[str, str]] = []          # (what changed, its rule)
    # WRITTEN BEFORE THE SPLIT (`task@1`): the calculation's card's
    # `restart` and `density_fit` are taken off it FIRST, for the
    # template's decision every later step asks.
    before_split = str(obj.get("schema") or "") == _TASK_AT_1
    if before_split:
        card = obj.get("execution")
        if isinstance(card, dict):
            rec.from_card = {k: card.pop(k) for k in _COMPUTED_ON_CARDS
                             if k in card}
    for key, (meant, rule) in _TASK_ABSENT.items():
        if key not in obj:
            obj[key] = meant
            lines.append((f"{key} = {meant!r} (it was left out)", rule))
    # `enabled` ON A STAGE, retired 2026-10-03 (`engines/stages.md` § 2): a
    # stage is in the description or it is not.  A `true` is what the key
    # meant by its absence since, and is taken out; a stage switched OFF is
    # one the old file did not run, and whether it now runs or leaves the
    # description is the person's -- refused by name.
    for st in obj.get("stages") or ():
        if not (isinstance(st, dict) and "enabled" in st):
            continue
        name = st.get("name", "?")
        if st["enabled"] is not True:
            raise MigrateError(
                f"stage {name!r} is switched off (\"enabled\": "
                f"{json.dumps(st['enabled'])}) -- a stage has no on/off switch since "
                f"2026-10-03: it is in the description or it is not "
                f"(engines/stages.md § 2), and which this one is, is yours.  "
                f"Remove the stage from {TASK_FILENAME} to leave it out, or "
                f"its \"enabled\" to run it, then run the migration again; "
                f"nothing was written.")
        st.pop("enabled")
        lines.append((f"stages[{name}].enabled removed (2026-10-03: a "
                      f"stage has no on/off switch)", "2026-10-03"))
    # A STAGE OLDER VERSIONS WROTE THAT THE PAIR NO LONGER HOLDS beside its
    # others (the kind table's `retired`, `execution/architecture.md`
    # § 3.3) -- a SIESTA vibration's `relax`, since 2.42.0 one run in
    # phases (`engines/vibration.md` § 5.11 § 9): folded into the others,
    # its overrides carried; a value the stages state otherwise is refused
    # by name.
    from ..kind_table import pair
    _part = pair(str(obj.get("calculation", "")),
                 str((obj.get("engine") or {}).get("name", "")))
    retired = _part.retired if _part is not None else None
    if retired is not None:
        from ..identity import stage_key
        stages = [st for st in (obj.get("stages") or [])
                  if isinstance(st, dict)]
        old = [st for st in stages
               if stage_key(str(st.get("name", "")))
               == stage_key(retired.stage)]
        if old and len(stages) > 1:
            relax = old[0]
            rest = [st for st in stages if st is not relax]
            # THE RETIRED ROLE'S OWN ITEMS go onto the others (§ 5.11 § 9):
            # those whose `stages` name that role and no other of the kind
            # -- an item every role reads, alone or as a set (`scf`), is
            # the others' to state.  Any other value the retired stage set
            # for its own run is a conflict, named -- one run in phases
            # computes every phase at one value.
            from ..kind_table import kind
            from ..template import catalogue, select
            others = set(kind(_part.kind).role_names) - {retired.role}
            role_items = {
                it.name for it in select(catalogue(), engine=_part.engine,
                                         stages=(retired.role,))
                if it.stages and not set(it.stages) & others}
            overrides = dict(relax.get("overrides") or {})
            carried = {k: v for k, v in overrides.items()
                       if k in role_items}
            foreign = [k for k in overrides if k not in role_items]
            for st in rest:
                own = dict(st.get("overrides") or {})
                clash = sorted(
                    [k for k in carried if k in own and own[k] != carried[k]]
                    + [k for k in foreign if own.get(k) != overrides[k]])
                if clash:
                    raise MigrateError(
                        f"stage {relax.get('name')!r} overrides "
                        f"{', '.join(clash)}, which stage {st.get('name')!r} "
                        f"does not state the same way -- one run in phases "
                        f"carries one value for each, every phase computed "
                        f"at it (engines/vibration.md 5.11 9): state "
                        f"{'it' if len(clash) == 1 else 'them'} on "
                        f"{st.get('name')!r} or remove "
                        f"{'it' if len(clash) == 1 else 'them'}, in "
                        f"{TASK_FILENAME}, then run the migration again; "
                        f"nothing was written.")
                if carried:
                    st["overrides"] = {**own, **carried}
            obj["stages"] = rest
            rec.folded_stages[str(relax.get("name"))] = retired.since
            lines.append((
                f"stage {relax.get('name')!r} folded into "
                f"{', '.join(repr(st.get('name')) for st in rest)} "
                f"({retired.since}: {retired.why}"
                + (f"; its overrides {', '.join(sorted(carried))} carried"
                   if carried else "") + ")", retired.since))
    if obj.pop("centre_junction", None) is not None:
        # RETIRED IN 2.6.0: the junction enters its calculation centred
        # (`model/structure-periodicity.md` § 6.0).
        lines.append(("centre_junction removed (2.6.0: the junction enters "
                      "its calculation centred)", "2.6.0"))
    bias = obj.get("bias")
    if obj.get("calculation") == "transport" and (
            bias is None
            or (isinstance(bias, dict) and not bias.get("voltages_v"))):
        # THE SINGLE BIAS, STATED (`engines/transport.md` § 2a.10, 2.16.3):
        # `molbuilder describe` with no `--bias` wrote no voltages before, and the
        # reader refuses a transport description stating none.
        obj["bias"] = {**(bias or {}), "voltages_v": [0.0]}
        lines.append(("bias.voltages_v = [0.0] (2.16.3: a transport "
                      "description states its voltages -- the single bias; "
                      "it stated none)", "2.16.3"))
    bias = obj.get("bias")
    if (isinstance(bias, dict) and len(bias.get("voltages_v") or []) > 1
            and "low_bias_approximation" not in bias):
        # THE SWITCH, from the word it replaced (`engines/transport.md`
        # § 2a.10, 2026-10-08): `low-bias` was the approximation;
        # `re-converged`, and a list written before either was stated, ran
        # the device at every voltage.
        old = bias.pop("treatment", None)
        bias["low_bias_approximation"] = old == "low-bias"
        lines.append((f"bias.low_bias_approximation = "
                      f"{bias['low_bias_approximation']} (from "
                      f"{'treatment ' + repr(old) if old else 'a list run at every voltage'})",
                      "2026-10-08"))
    notify = obj.get("notify")
    if isinstance(notify, dict) and notify:
        for key, meant in _NOTIFY_ABSENT.items():
            if key not in notify:
                notify[key] = meant
                lines.append((f"notify.{key} = {meant!r} (it was left out)",
                              "w57"))
        if never_as_zero(notify.get("every_hours")):
            notify["every_hours"] = None
            lines.append(("notify.every_hours = None (0 was never)", "w57"))
    from ..resolve import ResolveError
    try:
        grouped = _migrate_groups(rec, obj)
    except MigrateError:
        raise
    except (ValueError, ResolveError) as exc:
        # THE LADDER'S LINKS ARE READ ON THE DESCRIPTION AND ITS TEMPLATE: one
        # either refuses as it stands is said in its reader's words.
        raise MigrateError(f"{task_path.name}: its stages' groups cannot be "
                           f"decided, so nothing was written: {exc}") from None
    lines += [(ln, "2.7.0") for ln in grouped]
    request_body = None
    if before_split:
        try:
            view = Task.from_dict(_description_view(obj))
        except ValueError as exc:
            raise _refused_rewrite(task_path, exc) from None
        desc, request_body, split_said = _split(rec, obj, view)
        lines += [(ln, _SPLIT) for ln in split_said]
    else:
        desc = obj
    try:
        task = Task.from_dict(desc)
    except ValueError as exc:
        raise _refused_rewrite(task_path, exc) from None
    if lines:
        writes.add(task_path, json_text(task.to_dict()),
                   {rule for _ln, rule in lines})
        said += [f"{TASK_FILENAME}: {ln}" for ln, _rule in lines]
    if request_body is not None:
        from .. import job_request as _jr
        jr_path = _jr.path_of(base)
        if jr_path.exists():
            raise MigrateError(
                f"{jr_path.name} is already there beside a description "
                f"written before it -- which of the two says how the "
                f"calculation runs is yours to decide: remove one, then run "
                f"the migration again; nothing was written.")
        try:
            rec.request = _jr.from_dict(request_body, task)
        except ValueError as exc:
            raise _refused_rewrite(jr_path, exc) from None
        writes.add(jr_path, json_text(rec.request.to_dict()), frozenset())
        said.append(f"{jr_path.name}: written -- how the calculation runs, "
                    f"a record of its own ({_SPLIT})")
    else:
        # THE JOB REQUEST BESIDE IT, read raw (A16, migrate's exception) --
        # what every later step composes the calculation with -- and the
        # template's run-setting values onto its card, by what the template
        # holds (`execution/job-request.md` § 8).
        from .. import job_request as _jr
        jr_path = _jr.path_of(base)
        body = _read_raw(jr_path)
        card = dict(body.get("execution") or {})
        moved = _onto_the_card(rec, task, card)
        # EACH CHANGE UNDER ITS OWN RULE: the run settings' move is 2.28.0's,
        # a retired stage's row its release's.
        rules = ({_RUN_OUT} if card != (body.get("execution") or {})
                 else set())
        changed = bool(rules)
        body["execution"] = card
        # A STAGE FOLDED INTO ANOTHER (the kind table's `retired`) takes its
        # row with it; the stage it was folded into keeps its own.
        folded = {n: since for n, since in rec.folded_stages.items()
                  if n in (body.get("stages") or {})}
        if folded:
            body["stages"] = {k: v for k, v in body["stages"].items()
                              if k not in folded}
            # THE NUMBERS AGAIN FROM 1, each stage keeping who it shares a
            # job with -- the page's rule for a stage removed
            # (`web/task-setup.md` § 5.3; `job-request.md` § 3.3: the first
            # stage is group 1).
            kept = [st.name for st in task.stages
                    if isinstance(body["stages"].get(st.name), dict)]
            numbers: list = []
            for n in kept:
                if body["stages"][n].get("group") not in numbers:
                    numbers.append(body["stages"][n].get("group"))
            for n in kept:
                body["stages"][n]["group"] = (
                    numbers.index(body["stages"][n].get("group")) + 1)
            changed = True
            rules |= set(folded.values())
        try:
            rec.request = _jr.from_dict(body, task)
        except ValueError as exc:
            raise _refused_rewrite(jr_path, exc) from None
        if changed:
            writes.add(jr_path, json_text(rec.request.to_dict()), rules)
        said += [f"{ln} ({_RUN_OUT})" for ln in moved]
        said += [f"stage {n!r}'s row dropped with the stage ({since})"
                 for n, since in folded.items()]

    for folder, d in rec.jobsets.items():
        js_path = JobSet.path_of(folder)
        filled, missing, renamed = 0, set(), set()
        split = 0
        for job in d.get("jobs") or []:
            # THE PROGRAM IS THE JOB'S (`Job.program`, 2.27.0), and a
            # placement's source that was the description, the allocation's.
            res = job.get("resources")
            if isinstance(res, dict) and "program" in res:
                job["program"] = res.pop("program")
                split += 1
            elif "program" not in job:
                job["program"] = None
                split += 1
            src = (job.get("placement") or {}).get("from")
            if isinstance(src, dict) and "description" in src.values():
                job["placement"]["from"] = {
                    k: ("allocation" if v == "description" else v)
                    for k, v in src.items()}
                split += 1
            old = {o: n for o, (n, _rule) in _JOB_RENAMED.items()
                   if o in job and n not in job}
            for o, n in old.items():
                job[n] = job.pop(o)
            gaps = {k: v for k, (v, _rule) in _JOB_ABSENT.items()
                    if k not in job}
            if gaps or old:
                job.update(gaps)
                filled += 1
                missing |= set(gaps)
                renamed |= set(old)
        if not filled and not split:
            continue
        try:
            body = JobSet.from_dict(d).to_dict()
        except (ValueError, KeyError, TypeError) as exc:
            raise _refused_rewrite(js_path, exc) from None
        writes.add(js_path, json_text(body),
                   {_JOB_ABSENT[k][1] for k in missing}
                   | {_JOB_RENAMED[o][1] for o in renamed}
                   | ({_SPLIT} if split else set()))
        said.append(f"{js_path.relative_to(base)}: "
                    f"{max(filled, 1)} job(s) -- "
                    + "; ".join(
                        ([f"given {', '.join(k for k in _JOB_ABSENT if k in missing)}"
                          f" (left out, as their absence meant)"]
                         if missing else [])
                        + [f"`{o}` renamed `{_JOB_RENAMED[o][0]}`"
                           for o in sorted(renamed)]
                        + ([f"the program onto the job, a placement's source "
                            f"`description` said `allocation` ({_SPLIT})"]
                           if split else [])))
    return said, writes, task


def _migrate_machine_record(base: Path) -> Tuple[List[str], "_Writes"]:
    """The calculation's copy of its machine's record, when an older
    molbuilder wrote it (`record.older_record`): its scheduler word as the
    block -- the kind, and each command by the name the kind gives it, run
    from PATH as before (`scheduler/kinds.toml`).  Returns what changed and
    the file to write; nothing for a copy that reads, or none."""
    from ..persist import json_text
    from ..scheduler.kinds import KindError, kind
    from ..scheduler.record import (SCHEMA, Environment, calculation_record,
                                    older_record)
    writes = _Writes()
    path = calculation_record(base)
    if not path.is_file():
        return [], writes
    raw = _read_raw(path)
    if not older_record(raw):
        return [], writes
    word = raw.get("scheduler")
    try:
        k = kind(str(word))
    except KindError as exc:
        raise MigrateError(f"{path.name}: its scheduler {word!r} -- {exc}; "
                           f"nothing was written") from None
    raw["scheduler"] = {"kind": k.name, **dict(k.commands)}
    raw["schema"] = SCHEMA
    try:
        env = Environment.from_dict(raw)
    except (ValueError, TypeError, KeyError) as exc:
        raise _refused_rewrite(path, exc) from None
    writes.add(path, json_text(env.to_dict()), {"2.10.1"})
    return ([f"{path.name}: scheduler {word!r} -> {env.scheduler.said()} "
             f"(2.10.1: the kind and where its commands are; each run "
             f"from PATH, as before)"], writes)


def _migrate_lead_names(calc) -> Tuple[List[str], List[Tuple[Path, Path]],
                                       "_Writes"]:
    """A transport lead's files named on the calculation's label; a lead's
    are on its own stem (`job-contracts.md` § 2.1 rule 2;
    `kind_table.rung_label`), the stem its engine files were always
    on.  Before 2.17.0 every file of a lead was on the label, and before
    2.23.2 prep's log of a lead's prep in the calculation's root was
    (`script-preparation.md` § 4.5: the log's stem is the deck's).  Each
    such file -- in the calculation's root, the lead's stage folder and its
    attempts -- is renamed through the run-file door (`runfiles.parse`, its
    label the lead's), and the root's `job-set.json` naming the old deck is
    rewritten -- ``calc``'s plan, the set as the records step decided it
    (:meth:`_Records.calculation`).  A lead whose own folders held such
    files was prepared for the old names: its way on is said.  Decided,
    never done here: ``(said, moves, writes)``; nothing for a calculation
    whose leads' files are on their stems, or that has none."""
    from ..paths import attempt_dir, attempts_in, bench_container
    from ..persist import json_text
    from ..runfiles import find
    from ..runrecord import LaunchRecordError
    from .commands import way_back
    from .materialize import newest_launched
    from .naming import ladder_homes
    from .model import FILENAME as JOBSET_FILENAME, JobSet
    import dataclasses
    said: List[str] = []
    moves: List[Tuple[Path, Path]] = []
    writes = _Writes()
    renamed_decks: Dict[str, str] = {}
    base, task = calc.base, calc.task
    js = calc.jobset
    for home in ladder_homes(base, task):
        if home.label == task.label:
            continue
        own = ([home.dir] + [attempt_dir(home.dir, n)
                             for n in attempts_in(home.dir)]
               if home.dir is not None and home.dir.is_dir() else [])
        # ITS BENCHMARK'S LOG lies in its bench container, on the stem the
        # lead's prep named it (`paths.bench_container`).
        bench = base / bench_container(home.token)
        logs_at = [base] + ([bench] if bench.is_dir() else [])
        stage_moves: List[Tuple[Path, Path]] = []
        for folder in [*logs_at, *own]:
            for path, rf in find(folder, task.label, stage=home.token):
                new = folder / dataclasses.replace(rf, label=home.label).name
                stage_moves.append((path, new))
                if path.suffix == ".fdf" and folder == home.dir:
                    renamed_decks[path.name] = new.name
        if not stage_moves:
            continue
        moves += stage_moves
        said += [f"{old.relative_to(base)} -> {new.name} (a lead's files "
                 f"are on its own stem, job-contracts.md 2.1 rule 2)"
                 for old, new in stage_moves]
        if not any(old.parent in own for old, _new in stage_moves):
            continue
        # PREPARED FOR THE OLD NAMES: its run script and header name them.
        # LAUNCHED, as its runs' launch records say (`materialize.
        # newest_launched`, the one rule).
        try:
            launched = newest_launched(home.dir) is not None
        except LaunchRecordError as exc:
            raise MigrateError(f"{exc}; nothing was written") from None
        said.append(f"{home.name}: its run script and header were written "
                    f"for the old names -- "
                    + ("to launch it again, " if launched
                       else "to launch it, ")
                    + way_back(home.name, calc=calc, launched=launched))
    if renamed_decks and js is not None:
        jobs = [dataclasses.replace(j, script=renamed_decks[j.script])
                if j.script in renamed_decks else j for j in js.jobs]
        if jobs != list(js.jobs):
            writes.add(JobSet.path_of(base),
                       json_text(dataclasses.replace(js, jobs=jobs).to_dict()),
                       {"2.17.0"})
            said.append(f"{JOBSET_FILENAME}: each lead's deck named on its "
                        f"own stem")
    return said, moves, writes


def _migrate_run_indexes(calc) -> Tuple[List[str], List[Tuple[Path, Path]],
                                        "_Writes"]:
    """A RUN OF POINTS' INDEX as molbuilder 2.42-2.54 wrote it, rewritten
    into the one index every run of points keeps (`runpoints`,
    `execution/runs-of-points.md` § 2.3, § 3 op. 18): a SIESTA vibration
    run's ``.phases.json`` (``vibration-phases@1``) -- its file kept under
    its old name's ``.pre-2.55.0`` -- and a transport run's ``.points.json``
    of ``transport-points@1``, kept as every rewritten file is.  Each point
    keeps what it was and how it ended.  A vibration's call is its kind's
    declaration's (`vibration_phases.point_entry`): what it needs and must
    hold, its round and its words -- the same answers launch writes today;
    a transport point keeps its 2.54 entry's words, needs and must-hold,
    its round its point's (`stages.Point.round`).  A point taken over keeps no
    stamps: it did not run in that run.  Decided, never done here:
    ``(said, moves, writes)``; nothing for a calculation with no such
    index."""
    from ..persist import json_text, read_json
    from ..paths import attempt_dir, attempts_in
    from ..runpoints import (OLDER_SCHEMAS, add_point, empty_index,
                             index_path, older_index)
    from .naming import stage_home
    from .points import points_of, record_of
    said: List[str] = []
    moves: List[Tuple[Path, Path]] = []
    writes = _Writes()
    rule = "2.55.0"
    base, task = Path(calc.base), calc.task
    declared, record = points_of(task), record_of(task)
    if declared is None or calc.jobset is None:
        return said, moves, writes
    for job in calc.jobset.jobs:
        home = stage_home(base, task, job.name)
        if not home.dir.is_dir():
            continue
        names = home.names
        for n in attempts_in(home.dir):
            run = attempt_dir(home.dir, n)
            old_path, new_path = older_index(run, names), index_path(run, names)
            if old_path.is_file():
                src = old_path
            elif new_path.is_file():
                src = new_path
            else:
                continue
            try:
                old = read_json(src)
            except (OSError, ValueError) as exc:
                raise MigrateError(f"{src.relative_to(base)} does not read, "
                                   f"so nothing was written: {exc}") from None
            schema = old.get("schema") if isinstance(old, dict) else None
            if schema not in OLDER_SCHEMAS:
                continue
            index = _index_of_2_54(old, schema, calc=calc, stage=job.name,
                                   declared=declared, record=record,
                                   add_point=add_point,
                                   empty_index=empty_index)
            writes.add(new_path, json_text(index), [rule])
            kept = _kept_as(old_path, [rule]) if src == old_path else None
            if kept is not None:
                moves.append((old_path, kept))
            said.append(f"{src.relative_to(base)}: the run's index as "
                        f"molbuilder 2.54 wrote it, rewritten as "
                        f"{new_path.name} -- one index for every run of "
                        f"points (execution/runs-of-points.md 2.3)"
                        + (f"; {old_path.name} kept as {kept.name}"
                           if kept is not None else ""))
    return said, moves, writes


def _index_of_2_54(old: dict, schema: str, *, calc, stage: str, declared,
                   record, add_point, empty_index) -> dict:
    """A 2.54 index's points as the one index states them (`runpoints`):
    a vibration's calls by its phases table, a transport rung's points as
    its prepared points are -- what each is, what it needs and must hold
    -- and how each ended, kept."""
    from ..runpoints import OLDER_SCHEMAS
    keep = ("started", "ended", "exit", "verdict", "not_run")
    if schema == OLDER_SCHEMAS[0]:
        from ..siesta.vibration_phases import Call, phase, point_entry
        facts = {"phases": list(old.get("phases") or [])}
        if old.get("selection") is not None:
            facts["selection"] = old["selection"]
        index = empty_index(stage=stage, record=record, facts=facts)
        present = {c["phase"] for c in old.get("calls") or []}
        own = ("phase", "name", "mode", "side", "frequency_cm1",
               "dq_amu12_ang", "taken_from", "record_error") + keep
        # IN THE LISTING ORDER (`runs-of-points.md` § 1.1): the phases in
        # their table's order, each phase's calls as planned -- 2.54 wrote
        # the calls a launch took over first.
        from ..siesta.vibration_phases import PHASES
        order = [p.name for p in PHASES]
        calls = sorted(old.get("calls") or [],
                       key=lambda c: (order.index(c["phase"]),
                                      str(c.get("name") or "")))
        for c in calls:
            call = Call(phase=c["phase"], name=c.get("name"),
                        mode=c.get("mode"), side=c.get("side"),
                        frequency_cm1=c.get("frequency_cm1"),
                        dq_amu12_ang=c.get("dq_amu12_ang"))
            rule = phase(call.phase)
            e = point_entry(call, calc.task.label)
            row = add_point(
                index, rel=e["rel"], words=e["words"], phase=e["phase"],
                round=e["round"], must_hold=e["must_hold"],
                verdict_counts=e["verdict_counts"],
                needs=[Call(n, n).rel for n in rule.needs if n in present],
                planned_by=(Call(rule.planned_by, rule.planned_by).rel
                            if rule.planned else None),
                taken_from=c.get("taken_from"),
                facts={**e["facts"], **{k: v for k, v in c.items()
                                        if k not in own}})
            _kept_stamps(row, c, keep)
            if c.get("record_error") and not c.get("taken_from"):
                what, _sep, error = str(c["record_error"]).partition(": ")
                row["step_error"] = {"what": what, "error": error or None}
        index["finish"] = old.get("finish")
        return index
    # transport-points@1
    from ..transport.stages import Point
    prepared = {pt.rel: pt for pt in declared.rung_points(calc, stage)}
    by_rel = {p["rel"]: p for p in old.get("points") or []}
    order = [rel for rel in prepared if rel in by_rel] + [
        rel for rel in by_rel if rel not in prepared]
    index = empty_index(stage=stage, record=record,
                        facts={k: old.get(k) for k in ("computes", "pair",
                                                       "leads", "spin",
                                                       "weights")})
    for rel in order:
        p = by_rel[rel]
        pt = prepared.get(rel)
        facts = (pt.facts() if pt is not None else
                 {"frame": p.get("frame"), "frame_words": p.get("frame_words"),
                  "bias_v": p.get("bias_v")})
        row = add_point(
            index, rel=rel, words=p.get("words") or rel, phase=stage,
            round=(pt or Point(frame=p.get("frame"),
                               volts=p.get("bias_v"))).round,
            needs=[p["needs"]] if p.get("needs") else [],
            must_hold=[(m, w) for m, w in p.get("must_hold") or []],
            verdict_counts=bool(p.get("verdict_counts", True)),
            taken_from=p.get("taken_from"),
            facts={**facts, "window": p.get("window")})
        _kept_stamps(row, p, keep)
    return index


def _kept_stamps(row: dict, old: dict, keep) -> None:
    """How a point ended, as its 2.54 entry said -- none for a point taken
    over, which did not run in that run."""
    if old.get("taken_from"):
        return
    for k in keep:
        if old.get(k) is not None:
            row[k] = old[k]


def migrate_state(base) -> List[str]:
    """Rewrite the calculation in ``base`` -- every step in the module's
    docstring, each decided before any is written -- and return what
    changed, line by line.  Each old file rewritten
    is kept beside its new one, once (:func:`_kept_as`); a renamed one is
    its new one.

    No line when there is nothing to migrate -- the calculation is current
    (:data:`CURRENT`); raises :class:`MigrateError` when it cannot be
    migrated as it stands.
    """
    base = Path(base)
    # EVERY RECORD READ ONCE, then EVERY DECISION BEFORE ANY WRITE: a
    # refusal of any step leaves the calculation as it was, never half
    # rewritten (`architecture.md` A16, migrate's rule).
    rec = _Records.read(base)
    said, writes, task = _migrate_records(rec)
    m_said, m_writes = _migrate_machine_record(base)
    said += m_said
    for path, (text, rules) in m_writes.items():
        writes.add(path, text, rules)
    calc = rec.calculation(task)
    lead_said, moves, lead_writes = _migrate_lead_names(calc)
    for path, (text, rules) in lead_writes.items():
        writes.add(path, text, rules)
    ix_said, ix_moves, ix_writes = _migrate_run_indexes(calc)
    moves += ix_moves
    for path, (text, rules) in ix_writes.items():
        writes.add(path, text, rules)
    lead_said += ix_said
    try:
        decision = _migrate_template(rec, task)
    except MigrateError:
        raise
    except ValueError as exc:
        # A READER'S REFUSAL met while deciding the template's rewrite is
        # the migration's, in the reader's words.
        raise MigrateError(f"the template's rewrite cannot be decided, so "
                           f"nothing was written: {exc}") from None
    t_said: List[str] = []
    if decision is not None:
        t_said = list(decision.said)
        writes.add(decision.path, decision.text, decision.rules)
        if decision.task_rules:
            # THE DESCRIPTION AS EVERY STEP LEFT IT: the stages' old zeros,
            # renamed and retired items taken out of the records step's.
            from ..persist import json_text
            from ..task import path_of
            writes.add(path_of(base),
                       json_text(_edited(task, decision).to_dict()),
                       decision.task_rules)
    if not writes and not moves:
        # NOTHING TO MIGRATE is an answer, not a refusal.
        return []
    taken = [new for _old, new in moves if new.exists()]
    if taken:
        raise MigrateError(
            f"{', '.join(str(p.relative_to(base)) for p in taken)} already "
            f"exist -- a file is "
            f"never renamed over another; nothing was written.  Move "
            f"{'it' if len(taken) == 1 else 'them'} aside and run the "
            f"migration again.")
    # A KEPT FILE IS NEVER WRITTEN OVER: it is the only copy of what the
    # file was before an earlier migration.
    # A NEW FILE -- the job request a split writes -- keeps nothing.
    kept = {path: _kept_as(path, rules)
            for path, (_text, rules) in writes.items()
            if path.exists()}
    held = [k for k in kept.values() if k.exists()]
    if held:
        raise MigrateError(
            f"{', '.join(str(p) for p in held)} already "
            f"{'holds' if len(held) == 1 else 'hold'} an earlier version -- "
            f"a kept file is never written over; nothing was written.  Move "
            f"{'it' if len(held) == 1 else 'them'} aside and run the "
            f"migration again.")
    # THE PACKAGE'S ONE WRITER for every file a migration rewrites
    # (`persist.write_bytes`, `execution/architecture.md` § 2.1).
    from ..persist import write_bytes
    for path, (text, _rules) in writes.items():
        if path in kept:
            write_bytes(kept[path], path.read_bytes())
        write_bytes(path, text.encode("utf-8"))
    for old, new in moves:
        old.rename(new)
    said += [f"{path.relative_to(base)} kept as {kept[path].name}"
             for path in writes if path in kept]
    said += lead_said + t_said
    from . import ledger
    ledger.record(base, "migrate", "a calculation an older molbuilder "
                  "wrote, rewritten -- each change with its reason",
                  changes=said)
    return said


class _TemplateDecision(NamedTuple):
    """The template's migration, decided (:func:`_decide_template`)."""
    #: What changed, a line each.
    said: Tuple[str, ...]
    #: The template's file, and its new text.
    path: Path
    text: str
    #: The rules the old template was written before (:func:`_kept_as`).
    rules: frozenset
    #: The description's `varies` and each stage's own values as this
    #: step leaves them (:func:`_edited`), and the rules they were written
    #: before -- empty when it changed neither.
    varies: Tuple[str, ...]
    overrides: Dict[str, Dict[str, Any]]
    task_rules: frozenset


def _edited(task, decision: Optional["_TemplateDecision"]):
    """``task`` with the template step's edits to the description -- its
    `varies` and its stages' own values as that step decided them, a key
    a release renamed or retired and a zero that meant *not set* taken out
    -- or as it is when the step changed neither.  The edits are the
    stages' own, whatever their groups, so the groups step and the write
    apply one decision."""
    import dataclasses
    if decision is None or not decision.task_rules:
        return task
    return dataclasses.replace(task, varies=decision.varies, stages=tuple(
        dataclasses.replace(st, overrides=decision.overrides.get(
            st.name, st.overrides)) for st in task.stages))


def _migrate_template(rec: "_Records", task) -> Optional["_TemplateDecision"]:
    """The template's migration, decided ONCE per run (:class:`_Records`)
    -- ``None`` when the template, or the stages' values, carry nothing to
    rewrite -- the groups step and the write ask the same decision, each
    applying its edits to the description it holds (:func:`_edited`)."""
    if rec._template_decision is _UNDECIDED:
        try:
            rec._template_decision = _decide_template(rec, task)
        except _NothingInTheTemplate:
            rec._template_decision = None
    return rec._template_decision


class _NothingInTheTemplate(MigrateError):
    """The template carries no item written before the electronic state's
    and no item a release retired, and the calculation no 0 written before
    2.1.0 that meant *not set*."""


def _decide_template(rec: "_Records", task):
    """The template's charge and spin items, rewritten into the electronic
    state's -- this module's first migration (2026-09-28) -- its zeros, and
    its stages', that meant *not set* (2.1.0), an item renamed, kept under
    its new name, and the items a release retired, taken out of the template
    and of the description (`template.RETIRED_ITEMS`).  ``task`` is the
    description the asking step holds; what this step does to it is its
    stages' own values and `varies`, whatever their groups
    (:func:`_edited`).  Nothing is written here: returns the decision
    (:class:`_TemplateDecision`).  The items the electronic state replaced
    are the template's table (`template.MIGRATED_ITEMS`): ``method``,
    ``spin_treatment`` and ``net_charge`` are current items -- an OLD VALUE
    in one is rewritten, a current value is the file's own statement and is
    kept."""
    import dataclasses

    from .. import template as _T
    from .engines import engine_seam

    engine = task.engine
    kind = task.calculation
    base = rec.base
    template = rec.template_for(task.label)
    if template is None:
        raise _NothingInTheTemplate(f"{base} holds no template.")
    tmpl, text = template.path, template.text
    try:
        parsed = _T.read_template(text)   # read against its OWN declarations
        mine = (_T.select(parsed, engine=engine) if parsed.engines
                else parsed.items)
    except ValueError as exc:
        raise MigrateError(f"{tmpl.name} does not read, so nothing was "
                           f"written: {exc}") from exc
    vals = {it.name: it.value for it in mine if it.is_set}
    # A RENAMED ITEM KEEPS WHAT IT HELD, under its new name and said as such
    # -- its value, or its blank (`template.RENAMED_ITEMS`, the table the
    # reader refuses the old name by).  Filtering on today's schema alone
    # would drop it and write the default in its place.
    renamed_to = {old: new for old, (new, _why, _at)
                  in _T.RENAMED_ITEMS.items()}
    renamed: List[str] = []
    renamed_blank: List[str] = []
    renamed_at: List[str] = []
    for it in mine:
        new_name = renamed_to.get(it.name)
        if new_name is None:
            continue
        renamed_at.append(_T.RENAMED_ITEMS[it.name][2])
        if not it.is_set:
            renamed_blank.append(new_name)
            renamed.append(f"{it.name} (blank) -> {new_name} (renamed)")
            continue
        if new_name in vals and vals[new_name] != vals[it.name]:
            raise _renamed_conflict(it.name, vals[it.name], new_name,
                                    vals[new_name])
        vals[new_name] = vals.pop(it.name)
        renamed.append(f"{it.name} = {vals[new_name]!r} -> {new_name} "
                       f"(renamed)")
    # AND IN task.json: a stage's own value, and `varies`, under the new name
    # -- a stage holding both names at two values is refused, as the
    # template is.
    stage_renamed = [(st.name, k, v) for st in (task.stages or ())
                     for k, v in (st.overrides or {}).items()
                     if k in renamed_to]
    varies_renamed = [k for k in (task.varies or ()) if k in renamed_to]
    for st in (task.stages or ()):
        for k, v in (st.overrides or {}).items():
            new_name = renamed_to.get(k)
            if new_name is not None and new_name in st.overrides \
                    and st.overrides[new_name] != v:
                raise _renamed_conflict(k, v, new_name, st.overrides[new_name],
                                        f"stage {st.name!r} in task.json")
    if stage_renamed or varies_renamed:
        task = dataclasses.replace(
            task, varies=tuple(dict.fromkeys(
                renamed_to.get(k, k) for k in (task.varies or ()))),
            stages=tuple(dataclasses.replace(st, overrides={
                renamed_to.get(k, k): v
                for k, v in (st.overrides or {}).items()})
                for st in task.stages))
        renamed_at += [_T.RENAMED_ITEMS[k][2] for _s, k, _v in stage_renamed]
        renamed_at += [_T.RENAMED_ITEMS[k][2] for k in varies_renamed]
        renamed += ([f"stage {st!r} {k} = {v!r} -> {renamed_to[k]} in "
                     f"task.json (renamed)" for st, k, v in stage_renamed]
                    + [f"{k} -> {renamed_to[k]} in task.json's varies "
                       f"(renamed)" for k in varies_renamed])
    # AN ITEM A RELEASE RETIRED is taken out -- of the template, and of a
    # stage's own values and `varies` -- when its value runs as before
    # without it, and refused by name otherwise: what the calculation was is
    # kept, never rounded (`template.Retired.runs`).
    # ...ON THIS ENGINE: an item one engine keeps is another's to retire
    # (`template.Retired.engines`).
    def _gone(name: str) -> bool:
        r = _T.RETIRED_ITEMS.get(name)
        return r is not None and r.retired_on(engine)

    retired = [it for it in mine if _gone(it.name)]
    stage_retired = [(st.name, k, v) for st in (task.stages or ())
                     for k, v in (st.overrides or {}).items() if _gone(k)]
    varies_retired = [k for k in (task.varies or ()) if _gone(k)]
    # A STAGE'S OWN RAMAN, said by SIESTA's retired selector, is not taken
    # out: it decided the stage's optical phase, and the template's mapping
    # (:func:`_map_raman`) speaks for the template alone.  A `varies` entry
    # alone decides nothing, and is taken out as any retired item's.
    raman_said = [f"stage {st!r} overrides {k}" for st, k, _v in stage_retired
                  if k in _T.SIESTA_RAMAN_SELECTOR and engine == "siesta"]
    if raman_said:
        raise MigrateError(
            f"{'; '.join(raman_said)}, which picked that stage's Raman modes "
            f"on SIESTA -- since 2.45.0 Compute Raman and Raman for say it.  "
            f"Nothing was written: state them on the stage instead (or take "
            f"the override out), then run the migration again "
            f"(engines/vibration.md 3.1).")
    stuck = ([(f"the template says {it.name} = {it.value!r}", it.name)
              for it in retired
              if it.is_set
              and not _T.RETIRED_ITEMS[it.name].runs_without(it.value)]
             + [(f"stage {st!r} overrides {k} = {v!r}", k)
                for st, k, v in stage_retired
                if not _T.RETIRED_ITEMS[k].runs_without(v)])
    if stuck:
        said_of, name = stuck[0]
        raise MigrateError(
            f"{said_of}, and {name} was retired at a value that no longer "
            f"runs ({_T.RETIRED_ITEMS[name].why}).  Taking it out would "
            f"change the calculation, so nothing was written: delete it to "
            f"accept what runs now, then run the migration again.")
    if stage_retired or varies_retired:
        task = dataclasses.replace(
            task, varies=tuple(k for k in (task.varies or ())
                               if not _gone(k)),
            stages=tuple(dataclasses.replace(st, overrides={
                k: v for k, v in (st.overrides or {}).items()
                if not _gone(k)}) for st in task.stages))
    retired_said = (
        [(f"{it.name} = {it.value!r}" if it.is_set else f"{it.name} (blank)")
         + f" removed ({_T.RETIRED_ITEMS[it.name].why})" for it in retired]
        + [f"stage {st!r} {k} = {v!r} removed from task.json "
           f"({_T.RETIRED_ITEMS[k].why})" for st, k, v in stage_retired]
        + [f"{k} removed from task.json's varies "
           f"({_T.RETIRED_ITEMS[k].why})" for k in varies_retired])
    retired_at = [_T.RETIRED_ITEMS[n].at for n in _T.RETIRED_ITEMS
                  if any(it.name == n for it in retired)
                  or n in varies_retired
                  or any(k == n for _s, k, _v in stage_retired)]
    old_names = set(_T.MIGRATED_ITEMS)
    old_items = old_names & set(vals)
    # AN OLD VALUE is one whose spelling the electronic state changed -- a
    # value it keeps (`spin-orbit`) is today's, and the file's own word.
    old_values = {k for k in ("method", "spin_treatment")
                  if k in vals and (vals[k] in _PYSCF_CLASS if k == "method"
                                    else _SIESTA_TREATMENT.get(vals[k],
                                                               vals[k])
                                    != vals[k])}
    # A 0 WRITTEN BEFORE 2.1.0 on the two NEGF broadenings meant *not set*
    # (`template.ZERO_WAS_NOT_SET`): written blank, what it meant.  On a
    # stage it is the override taken out and the item written blank: each
    # of the two is read by one rung alone (its `stages`), so *not set on
    # that rung* is *not set*.
    zeros = _T.zeros_written_before(parsed)
    # A STAGE'S, where that stage reads the item: the override is taken out
    # and the item written blank.
    stage_zeros = [(name, k) for name, k
                   in _T.stage_zeros_written_before(parsed, task.stages or ())
                   if _T.reads(_T.one(_T.catalogue(), k, engine=engine),
                               engine, kind, name)]
    if stage_zeros:
        task = dataclasses.replace(task, stages=tuple(
            dataclasses.replace(st, overrides={
                k: v for k, v in (st.overrides or {}).items()
                if (st.name, k) not in stage_zeros})
            for st in task.stages))
    zero_said = ([f"{name} = 0 -> not set" for name in zeros]
                 + [f"stage {st!r} {name} = 0 -> not set (the override "
                    f"taken out of task.json; the template's {name} is "
                    f"written blank, since only that rung reads it)"
                    for st, name in stage_zeros])
    zeros = list(dict.fromkeys(zeros + [name for _st, name in stage_zeros]))
    # AN ITEM NO RELEASE HELD is refused by name, as the reader refuses it
    # (`template.config_from_template`): a misspelling -- carried by no
    # item, it would be dropped unsaid.
    config_cls = engine_seam(engine).config_cls
    known = _T.template_fields(config_cls)
    # A RUN SETTING, valueless or not, is taken out: the template holds
    # none (`engines/template.md` § 2) -- what it holds decides, never its
    # version -- its value moved onto the calculation's run card by the
    # records step (:func:`_onto_the_card`); and the calculation's card's
    # `restart` and `density_fit`, which say what is computed, come in.
    ours = _T.run_settings(engine)
    run_items = sorted(it.name for it in mine if it.name in ours)
    vals = {k: v for k, v in vals.items() if k not in run_items}
    unknown = sorted(k for k in vals if k not in known
                     and k not in old_names and not _gone(k))
    if unknown:
        raise MigrateError(
            f"{tmpl.name} names {', '.join(map(repr, unknown))}, which no "
            f"item holds -- a misspelling; prep refuses it too.  Correct "
            f"it, then run the migration again; nothing was written.")
    from_card = dict(rec.from_card or {})
    if not (old_items or old_values or zeros or renamed or retired
            or stage_retired or varies_retired or run_items or from_card):
        raise _NothingInTheTemplate(
            f"{tmpl.name} carries no item written before the electronic "
            f"state's, no 0 written before {_T.ZERO_MEANING_CHANGED_AT} "
            f"that meant 'not set', no item renamed and no item a release "
            f"retired.")
    state_said = bool(old_items or old_values)

    staged = [(st.name, k) for st in (task.stages or ())
              for k in (st.overrides or {}) if state_said
              and (k in old_names or k in _T.STATE_ITEMS)]
    if staged:
        name, item = staged[0]
        raise MigrateError(
            f"stage {name!r} overrides {item!r}, and the electronic state "
            f"belongs to the calculation, never to a stage -- every stage's "
            f"warm files are a density for one state "
            f"(science/chemistry-correctness.md § 2a, ES1).  Decide which "
            f"value the whole calculation carries, put it in the template, "
            f"remove the override from task.json, and run the migration "
            f"again.")

    state, said = (_map_state(engine, vals) if state_said else ({}, []))
    said = renamed + said + [
        f"{line} (before {_T.ZERO_MEANING_CHANGED_AT} a 0 meant the "
        f"program's own rule; since, it means no broadening)"
        for line in zero_said] + retired_said
    kept = {k: v for k, v in vals.items()
            if k in known and k not in old_names and k not in zeros}
    kept.update(state)
    kept.update(from_card)
    # A SIESTA VIBRATION'S RAMAN, AS ITS RETIRED SELECTOR SAID IT (2.45.0,
    # `engines/vibration.md` § 5.11 § 9): stated in the settings both
    # engines share, so the deck computes what it computed.
    if any(it.name == "es_mode_selection" for it in retired):
        switches, switches_said = _map_raman(vals)
        kept.update(switches)
        said += switches_said
    said += ([f"{', '.join(run_items)} taken out -- a run setting is the job "
              f"request's, never the template's ({_RUN_OUT})"]
             if run_items else [])
    try:
        cfg = config_cls(**kept)
    except (TypeError, ValueError) as exc:
        raise MigrateError(f"{tmpl.name} cannot be rewritten: {exc}") from exc
    # Unset stays unset: an item the old file left valueless is written
    # valueless again -- under its new name when it was renamed -- so a row
    # nobody answered is not given a default.
    valueless = [it.name for it in mine
                 if (not it.is_set or it.name in zeros) and it.name in known
                 and it.name not in _T.STATE_ITEMS] + renamed_blank
    # WHERE EACH ITEM'S VALUE CAME FROM, as the old file says it (§ 6.6
    # obligation 2; `engines/template.md` § 5, the `source` row): its source,
    # a blank item's too -- a cited run's `psml_lib` is blank because the run
    # brings its pseudopotentials, and *cited* says so -- and none where the
    # file records none, *not recorded*, never *not chosen*: what a file
    # written before sources were recorded says of every item, and of a value
    # mapped into the electronic state.  An item the old file never had is
    # nobody's choice.
    sources = {renamed_to.get(it.name, it.name): it.source for it in mine
               if renamed_to.get(it.name, it.name) in known}
    sources.update({k: sources.get(k) for k in kept})
    # THE CARD'S SCIENCE was the person's statement on the calculation's
    # run card.
    sources.update({k: "person" for k in from_card})
    new_text = _T.template_with_values(cfg, engine=engine, calculation=kind,
                                       title=_T.title_of(text),
                                       valueless=valueless,
                                       sources=sources)
    # The new file must be one prep accepts -- checked before anything is
    # written, so a failure leaves the calculation as it was, and is said
    # as the migration's refusal rather than a traceback.
    try:
        _T.config_from_template(new_text, config_cls)
        written = {it.name for it in _T.read_template(new_text).items}
    except ValueError as exc:
        raise MigrateError(
            f"{tmpl.name} rewritten would still be refused, so nothing was "
            f"written: {exc}") from exc
    # WHAT THE NEW FILE DOES NOT CARRY IS SAID: a value of an item no
    # longer one of this kind's, and the comments a person wrote inside the
    # file -- molbuilder's writer puts none there -- which the old file
    # keeps.
    said += [f"{k} = {v!r} not carried: no longer an item of a {kind} "
             f"calculation" for k, v in kept.items() if k not in written]
    inner = _inner_comments(text)
    if inner:
        said.append(f"{inner} comment line(s) written inside "
                    f"{tmpl.name} are not carried -- the old file keeps "
                    f"them")
    # THE RULES EACH FILE WAS WRITTEN BEFORE -- the old one kept as the
    # earliest (:func:`_kept_as`).
    rules = frozenset((["m6"] if state_said else [])
                      + ([_T.ZERO_MEANING_CHANGED_AT] if zeros else [])
                      + ([_SPLIT] if from_card else [])
                      + ([_RUN_OUT] if run_items else [])
                      + renamed_at + retired_at)
    task_rules = frozenset(
        ([_T.ZERO_MEANING_CHANGED_AT] if stage_zeros else [])
        + [_T.RETIRED_ITEMS[k].at for _s, k, _v in stage_retired]
        + [_T.RETIRED_ITEMS[k].at for k in varies_retired]
        + [_T.RENAMED_ITEMS[k][2] for _s, k, _v in stage_renamed]
        + [_T.RENAMED_ITEMS[k][2] for k in varies_renamed])
    return _TemplateDecision(
        said=tuple(said), path=tmpl, text=new_text, rules=rules,
        varies=tuple(task.varies or ()),
        overrides={st.name: dict(st.overrides or {})
                   for st in (task.stages or ())},
        task_rules=task_rules)


def _inner_comments(text: str) -> int:
    """How many comment lines a template holds below its header -- a
    person's own: molbuilder's writer puts none there."""
    lines = text.splitlines()
    head = next((i for i, ln in enumerate(lines) if not ln.startswith("#")),
                len(lines))
    return sum(1 for ln in lines[head:] if ln.lstrip().startswith("#"))
