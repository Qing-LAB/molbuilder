"""Rewrite a calculation an older molbuilder wrote -- the one road for a file
a reader refuses naming this command.  Every record is read once
(:class:`_Records`), every step decided before any file is written, and a
refusal of any step writes nothing.  **Each old file rewritten is kept
beside its new one, once, as** ``<name>.pre-<rule>`` -- the earliest rule it
was written before (:func:`_kept_as`): a plan's code (``m6``, ``w57``), a
day, or a release.  Each change is said, with its rule.

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
run from PATH as before -- and how a shell there enters its environment,
written ``script_generation`` before 2026-10-02, under its name now,
``env_init``.

**A run's files, named once** (2.60.0, :func:`_migrate_runs`): a run is
one start, its files named once on its stage's stem
(`execution/project-layout.md` § 1.6.1); a run an older molbuilder named
by the try -- ``<stem>-run0<role>``, its session log stamped with the
clock -- has each file renamed, every byte kept and its old name said.  A
run it cannot name so is left as it is, said with its reason, and not read
-- the five reasons of `job-request.md` § 8, decided in
:func:`_migrate_runs`.

**A retired run setting.**  A run setting a
release retired leaves `job-set.json`'s jobs and the job request's run cards
as it leaves the template (`template.RETIRED_ITEMS`), by the template's own
rule -- its engine's, at a value that runs as before; and a job set that
carried one was prepared by a molbuilder whose run scripts took what this
one's launch no longer hands them, so each group is named to prepare again
before its next launch, a benchmark its prep anew (:func:`_prepare_again`).

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
from functools import partial
from typing import (TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional,
                    Tuple)

if TYPE_CHECKING:                      # annotations only
    from ..template_file import TemplateFile
    from .model import JobSet


#: What a calculation holding nothing an older molbuilder wrote IS -- said
#: when a migration finds nothing to rewrite (:func:`migrate_state`).
CURRENT = ("nothing to migrate: it holds nothing an older molbuilder wrote "
           "-- its records are whole, its record of its machine is this "
           "molbuilder's, its runs' files are named as this molbuilder names "
           "them, and its template carries no item an older molbuilder "
           "meant otherwise, renamed or a release retired")


class MigrateError(ValueError):
    """The calculation cannot be migrated as it stands -- the message says
    why and what to do."""


class MigrationStopped(MigrateError):
    """A migration stopped partway, at a write or a rename that failed --
    written in the calculation's ledger by the migration itself, with what
    it had done (`jobset.planned.CutShort`), so its caller writes nothing
    more."""


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
                      where: str = "the template",
                      renamed: str = "`template.RENAMED_ITEMS`"
                      ) -> MigrateError:
    """One item under its old name and its new, at two values -- ``where``
    the file, ``renamed`` when or by what rule it was renamed."""
    return MigrateError(
        f"{where} says {old} = {old_value!r} and {new} = {new_value!r}: one "
        f"item, renamed ({renamed}), at two values.  Decide which the "
        f"calculation carries, delete the other, and run the migration "
        f"again; nothing was written.")


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
                f"count.  State the "
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
    from ..selection import parse_index_list
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
            f"again.")
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
               # A PREP BEFORE 2.42.0 knew no run in phases.
               "phases": (None, "2.42.0"),
               # A PREP BEFORE 2.22.0 kept no scratch folder: its device
               # deck names the leads' files loose in the run, and a launch
               # again of it empties nothing (`engines/transport.md`
               # § 2a.11); a prep again gives the stage the folder.
               "scratch_dir": (None, "2.22.0")}
#: THE RELEASE WHOSE READER REFUSES A JOB'S KEY NOTHING READS
#: (`jobset.model.JOB_RETIRED_KEYS`): a job set holding one was written
#: before it, by the release that wrote the key or by a migration that left
#: the key in place.
_READ_PAST_UNTIL = "2.65.7"
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
    """EVERY FILE THIS MIGRATION REWRITES, once: ``{path: (its write, the
    rules it was written before)}`` -- its write its owner's writer, taking
    the plan it writes into (`jobset.planned.Plan`; the template, which has
    no writer of its own, its text into the plan: A16 leaves raw writes to
    migrate).  Two steps rewriting one file -- the records step and the
    template a description -- give one write: the later step's is built on
    the body the earlier decided, and the rules gather, so the old file is
    kept once, under the earliest (:func:`_kept_as`)."""

    def add(self, path: Path, write, rules) -> None:
        _old, had = self.get(path, (None, frozenset()))
        self[path] = (write, had | frozenset(rules))


def _refused_rewrite(path: Path, exc: Exception) -> "MigrateError":
    """A rewritten record its reader would still refuse -- said once."""
    return MigrateError(f"{path} rewritten would still be refused, so "
                        f"nothing was written: {exc}")


#: Not yet asked (:attr:`_Records._template_decision`).
_UNDECIDED = object()


def _read_raw(path: Path) -> dict:
    """A record's JSON as it lies, for a rewrite -- one that does not read
    as one JSON object is no older molbuilder's, and is the migration's
    refusal, never a traceback.  No schema is asked: an older major is what
    a migration reads."""
    from ..persist import read_object
    try:
        return read_object(path)
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
    (:func:`_migrate_machine_record`); no step reads or rewrites a run's
    records -- a run's files an older grammar named are renamed by their own
    step, by name alone (:func:`_migrate_runs`).  The write keeps each old
    file, byte for byte, beside its new one."""
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
    #: then (:func:`_migrate_records`); the job-set step names each group's
    #: prep by it (:func:`_prepare_again`).
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


def _template_read(template: "TemplateFile"):
    """The template's items, read against its own declarations -- or refused by
    name, nothing written (`job-request.md` § 8): every step that reads it asks
    here, so a template that does not read never ends in a Python traceback."""
    from .. import template as _T
    try:
        return _T.read_template(template.text)
    except ValueError as exc:
        raise MigrateError(f"{template.path.name} does not read, so nothing "
                           f"was written: {exc}") from exc


def _migrate_groups(rec: "_Records", obj: dict) -> List[str]:
    """Every stage's `group` (2026-10-11, plan § 5aa M1), written into the
    raw description ``obj`` -- a line saying the numbers, or ``[]`` when
    every stage states one.  A stage prepared takes the group its job
    records (`Job.members`; prepared alone, a group of its own); one not
    prepared repeats the number above only where the stage above is not
    prepared either, no stage of that group is one it builds on
    (`ladder.upstream_of`) and their run cards state what one job's
    members share (`group.card_shares`) -- else one more.  A prepared
    group whose members are not one after another in the ladder cannot be
    stated by a number, and is refused by name: which jobs to keep is the
    person's.
    A description written since the split (`task@2`) holds no group -- its
    groups are its job request's (`execution/job-request.md` § 3.3)."""
    stages = obj.get("stages")
    if not isinstance(stages, list) or all(
            not isinstance(st, dict) or "group" in st for st in stages):
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
                    f"a group is stages one after another, so no number states that job.  Which to keep "
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
            f"and the run cards agree on what one job's members share)"]


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


def _retired_stuck(said_of: str, name: str) -> "MigrateError":
    """The refusal of an item a release retired, held at a value that no
    longer runs: taking it out would change the calculation, and what it
    was is kept, never rounded (`template.Retired.runs`)."""
    from .. import template as _T
    return MigrateError(
        f"{said_of}, and {name} was retired at a value that no longer "
        f"runs ({_T.RETIRED_ITEMS[name].why}).  Taking it out would "
        f"change the calculation, so nothing was written: delete it to "
        f"accept what runs now, then run the migration again.")


def _retired_off(values: dict, engine: str, said_of: str) -> List[str]:
    """EACH RUN SETTING A RELEASE RETIRED, out of ``values`` (changed in
    place) -- a run card, or a job's resources -- as the template's retired
    items leave the template: one ``engine``'s schema retired, its value
    running as before without it; any other value refused by name
    (:func:`_retired_stuck`).  The names taken out, in order."""
    from .. import template as _T
    gone = [k for k in values if _T.retired(k, engine) is not None]
    for k in gone:
        if not _T.RETIRED_ITEMS[k].runs_without(values[k]):
            raise _retired_stuck(f"{said_of} says {k} = {values[k]!r}", k)
    for k in gone:
        values.pop(k)
    return gone


def _cards_off(body: dict, engine: str) -> List[Tuple[Optional[str], str]]:
    """Each retired run setting off a job request's run cards -- the
    calculation's and each stage's, ``body`` changed in place
    (:func:`_retired_off`): ``(stage or None, name)`` each."""
    off = [(None, k) for k in _retired_off(body.get("execution") or {},
                                           engine,
                                           "the calculation's run card")]
    for n, row in (body.get("stages") or {}).items():
        own = row.get("execution") if isinstance(row, dict) else None
        if isinstance(own, dict):
            off += [(n, k) for k in _retired_off(own, engine,
                                                 f"stage {n!r}'s run card")]
    return off


def _cards_said(jr_name: str, off) -> List[str]:
    """What :func:`_cards_off` took out, a line each."""
    from .. import template as _T
    return [f"{jr_name}: {k} removed from "
            + ("the calculation's run card" if n is None
               else f"stage {n!r}'s run card")
            + f" ({_T.RETIRED_ITEMS[k].why})" for n, k in off]


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
    from ..runfiles import JOB_REQUEST_FILE, TASK_FILE
    said: List[str] = []
    desc = _description_view(obj)
    bench = dict(obj.get("bench") or {})
    measured = sorted(set(bench) & set(_COMPUTED_ON_CARDS))
    if measured:
        raise MigrateError(
            f"{TASK_FILE}'s bench measures {', '.join(map(repr, measured))}, "
            f"which say{'s' if len(measured) == 1 else ''} what is computed, "
            f"never how fast -- a benchmark measures run settings alone"
            f".  Remove "
            f"{'it' if len(measured) == 1 else 'them'} from the bench, then "
            f"run the migration again; nothing was written.")
    # WHAT A STAGE'S CARD SAID OF WHAT IS COMPUTED is already among its overrides
    # in ``desc`` (`_description_view`); the calculation's came off its card
    # before the split (`_migrate_description`).
    card = dict(obj.get("execution") or {})
    stages: Dict[str, dict] = {}
    for st, raw in zip(desc["stages"], obj.get("stages") or []):
        own = dict(raw.get("execution") or {})
        moved = {k: own.pop(k) for k in _COMPUTED_ON_CARDS if k in own}
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
    said.insert(0, f"how it runs -- its allocation, run cards, groups, bench "
                   f"and reports -- moved into {JOB_REQUEST_FILE}, a record of "
                   f"its own")
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
    from ..runfiles import JOB_REQUEST_FILE
    template = rec.template_for(task.label)
    if template is None:
        return []
    engine, kind = str(task.engine), str(task.calculation)
    run_settings = _T.run_settings(engine)
    said: List[str] = []
    for it in _template_read(template).items:
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
                    f"onto the calculation's run card in {JOB_REQUEST_FILE}")
    return said


def _migrate_records(rec: "_Records"):
    """`task.json`, `job-request.json` and every `job-set.json` of the
    calculation, each key an older file left out given the meaning its absence
    had, and a key a release retired removed -- on the bodies read once
    (:class:`_Records`), since the readers refuse such a file, and each
    rewritten file checked by its reader.  One function per record, so no
    record's names reach another's.  Nothing is written here: returns what
    changed (a line each), the files to write (:class:`_Writes`) and the
    description as rewritten; the bodies in ``rec`` are the decided ones."""
    said: List[str] = []
    writes = _Writes()
    task, request_body = _migrate_description(rec, said, writes)
    _migrate_request(rec, task, request_body, said, writes)
    _migrate_job_sets(rec, said, writes)
    return said, writes, task


def _migrate_description(rec: "_Records", said: List[str], writes: "_Writes"):
    """`task.json`: returns the description as rewritten and, for one written
    before the split, the job request's body taken out of it (``None`` for any
    other)."""
    import json

    from ..task import FILENAME as TASK_FILENAME
    from ..task import Task, never_as_zero, path_of, write_task
    base = rec.base
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
                f"2026-10-03: it is in the description or it is not"
                f", and which this one is, is yours.  "
                f"Remove the stage from {TASK_FILENAME} to leave it out, or "
                f"its \"enabled\" to run it, then run the migration again; "
                f"nothing was written.")
        st.pop("enabled")
        lines.append((f"stages[{name}].enabled removed (2026-10-03: a "
                      f"stage has no on/off switch)", "2026-10-03"))
    _fold_retired_stage(rec, obj, lines)
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
        grouped = _migrate_groups(rec, obj) if before_split else []
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
        writes.add(task_path, partial(write_task, base, task),
                   {rule for _ln, rule in lines})
        said += [f"{TASK_FILENAME}: {ln}" for ln, _rule in lines]
    return task, request_body


def _fold_retired_stage(rec: "_Records", obj: dict, lines: list) -> None:
    """A stage older versions wrote that the pair no longer holds, folded into
    the others (below); ``lines`` takes what changed and its rule."""
    from ..task import FILENAME as TASK_FILENAME
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
            # for its own run is a conflict, named -- one run of points
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
                        f"does not state the same way -- one run of points "
                        f"carries one value for each, every phase computed "
                        f"at it: state "
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


def _migrate_request(rec: "_Records", task, request_body, said: List[str],
                     writes: "_Writes") -> None:
    """`job-request.json`: written from the body the split took out of the
    description, or the one beside it read raw and rewritten; ``rec.request``
    holds it for every later step."""
    from .. import job_request as _jr
    from .. import template as _T
    base = rec.base
    jr_path = _jr.path_of(base)
    if request_body is not None:
        if jr_path.exists():
            raise MigrateError(
                f"{jr_path.name} is already there beside a description "
                f"written before it -- which of the two says how the "
                f"calculation runs is yours to decide: remove one, then run "
                f"the migration again; nothing was written.")
        off_cards = _cards_off(request_body, str(task.engine))
        try:
            rec.request = _jr.from_dict(request_body, task)
        except ValueError as exc:
            raise _refused_rewrite(jr_path, exc) from None
        writes.add(jr_path, partial(_jr.write, base, rec.request),
                   frozenset())
        said.append(f"{jr_path.name}: written -- how the calculation runs, "
                    f"a record of its own ({_SPLIT})")
        said += _cards_said(jr_path.name, off_cards)
    else:
        # THE JOB REQUEST BESIDE IT, read raw (A16, migrate's exception) --
        # what every later step composes the calculation with -- and the
        # template's run-setting values onto its card, by what the template
        # holds (`execution/job-request.md` § 8).
        body = _read_raw(jr_path)
        card = dict(body.get("execution") or {})
        moved = _onto_the_card(rec, task, card)
        # EACH CHANGE UNDER ITS OWN RULE: the run settings' move is 2.28.0's,
        # a retired stage's row its release's, a retired run setting its.
        rules = ({_RUN_OUT} if card != (body.get("execution") or {})
                 else set())
        body["execution"] = card
        # A RUN SETTING A RELEASE RETIRED, off every card -- the
        # calculation's and each stage's -- as the template's retired items
        # come out of the template (`template.RETIRED_ITEMS`).
        off_cards = _cards_off(body, str(task.engine))
        rules |= {_T.RETIRED_ITEMS[k].at for _n, k in off_cards}
        changed = bool(rules)
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
            writes.add(jr_path, partial(_jr.write, base, rec.request), rules)
        said += [f"{ln} ({_RUN_OUT})" for ln in moved]
        said += [f"stage {n!r}'s row dropped with the stage ({since})"
                 for n, since in folded.items()]
        said += _cards_said(jr_path.name, off_cards)


def _migrate_job_sets(rec: "_Records", said: List[str],
                      writes: "_Writes") -> None:
    """Every `job-set.json` of the calculation -- the root's and each
    benchmark's."""
    from .. import template as _T
    from ..persist import json_text
    from .model import JOB_RETIRED_KEYS, JobSet
    base = rec.base
    again: List[Tuple[bool, str]] = []
    for folder, d in rec.jobsets.items():
        js_path = JobSet.path_of(folder)
        missing, renamed, dropped = set(), set(), set()
        split = 0
        retired_res: set = set()
        touched = 0
        for job in d.get("jobs") or []:
            before = json_text(job)
            # THE PROGRAM IS THE JOB'S (`Job.program`, 2.27.0), and a
            # placement's source that was the description, the allocation's.
            res = job.get("resources")
            # A RUN SETTING A RELEASE RETIRED, out of the job's allocation
            # (`template.RETIRED_ITEMS`; `Resources.from_dict` refuses it).
            if isinstance(res, dict):
                retired_res |= set(_retired_off(
                    res, str(d.get("engine")),
                    f"job {job.get('name')!r}'s resources"))
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
            # A KEY NOTHING READS, out of the job (`Job.from_dict` refuses
            # it): `asked` and `written_at`.  Written before the release
            # whose reader refuses them (:data:`_READ_PAST_UNTIL`) -- an
            # earlier migration left them in place.
            for k in [k for k in JOB_RETIRED_KEYS if k in job]:
                job.pop(k)
                dropped.add(k)
            gaps = {k: v for k, (v, _rule) in _JOB_ABSENT.items()
                    if k not in job}
            if gaps or old:
                job.update(gaps)
                missing |= set(gaps)
                renamed |= set(old)
            touched += json_text(job) != before
        if not touched:
            continue
        try:
            rewritten = JobSet.from_dict(d)
        except (ValueError, KeyError, TypeError) as exc:
            raise _refused_rewrite(js_path, exc) from None
        writes.add(js_path, partial(rewritten.write, folder),
                   {_JOB_ABSENT[k][1] for k in missing}
                   | {_JOB_RENAMED[o][1] for o in renamed}
                   | ({_READ_PAST_UNTIL} if dropped else set())
                   | ({_SPLIT} if split else set())
                   | {_T.RETIRED_ITEMS[k].at for k in retired_res})
        said.append(f"{js_path.relative_to(base)}: "
                    f"{touched} job(s) -- "
                    + "; ".join(
                        ([f"given {', '.join(k for k in _JOB_ABSENT if k in missing)}"
                          f" (left out, as their absence meant)"]
                         if missing else [])
                        + [f"`{o}` renamed `{_JOB_RENAMED[o][0]}`"
                           for o in sorted(renamed)]
                        + [f"`{k}` removed, which nothing reads since "
                           f"{JOB_RETIRED_KEYS[k]} ({_READ_PAST_UNTIL})"
                           for k in sorted(dropped)]
                        + ([f"the program onto the job, a placement's source "
                            f"`description` said `allocation` ({_SPLIT})"]
                           if split else [])
                        + [f"{k} removed from each job's resources "
                           f"({_T.RETIRED_ITEMS[k].why})"
                           for k in sorted(retired_res)]))
        # PREPARED BY AN OLDER MOLBUILDER, whose run scripts took what this
        # one's launch no longer hands them: each group is prepared again
        # before its next launch -- a launched group's runs kept, what it
        # computes unchanged (`execution/job-request.md` § 8); a
        # benchmark's trials by their benchmark's prep anew.
        if retired_res:
            way = _prepare_again(folder, rewritten, rec)
            if way:
                again.append((folder == base, way))
    # A BENCHMARK'S WAY BACK FIRST: the state it restores is older than
    # the root's prep the last line asks for.
    said += ([w for root, w in again if not root]
             + [w for root, w in again if root])


def _prepare_again(folder: Path, js: "JobSet", rec: "_Records"
                   ) -> Optional[str]:
    """The way on for a job set an older molbuilder prepared (a run setting
    it carried retired since), ``js`` as this migration rewrites it: its
    groups prepared again; a benchmark's prep anew where a trial is not yet
    launched -- a launched one its launch passes over (`submit._plan_trial`,
    the same test), so nothing for a benchmark measured whole.  The one
    composer of each (`commands`)."""
    from ..runrecord import LaunchRecordError
    from .commands import block, prep_group_lines, rollback
    base = rec.base
    from .materialize import latest_attempt, newest_launched
    from .model import JobSet
    from .naming import job_dir_names
    where = JobSet.path_of(folder).relative_to(base)
    head = (f"{where}: prepared by an older molbuilder, whose run scripts "
            f"take what this one's launch no longer hands them -- ")
    if folder != base:
        dirs = job_dir_names(js)
        try:
            waiting = any(
                (run := latest_attempt(base / dirs[j.name])) is not None
                and newest_launched(base / dirs[j.name]) != run
                for j in js.jobs)
        except LaunchRecordError as exc:
            raise MigrateError(f"{exc}; nothing was written") from None
        if not waiting:
            return None
        return (head + "a trial not yet launched would be refused at its "
                "start: " + rollback("the benchmark's prep", base=base)
                + " -- running the migration again after the restore")
    groups = sorted({g for g in (rec.request.group_of(j.name)
                                 for j in js.jobs)
                     if g is not None})
    return (head + "prepare each group again before its next launch (a "
            "launched group keeps its runs):\n"
            + block(prep_group_lines(groups, base=base)))


def _migrate_machine_record(rec: "_Records", said: List[str],
                            writes: "_Writes") -> None:
    """The calculation's copy of its machine's record, when an older
    molbuilder wrote it (`record.older_record`): its scheduler word as the
    block -- the kind, and each command by the name the kind gives it, run
    from PATH as before (`scheduler/kinds.toml`); and how a shell there
    enters its environment, written ``script_generation`` before
    2026-10-02, under its name now, ``env_init`` -- one block at two values
    refused by name.  Nothing for a copy that reads, or none."""
    from ..scheduler.kinds import KindError, kind
    from ..scheduler.record import (SCHEMA, Environment, calculation_record,
                                    older_record, write_environment)
    path = calculation_record(rec.base)
    if not path.is_file():
        return
    raw = _read_raw(path)
    if not older_record(raw):
        return
    word = raw.get("scheduler")
    try:
        k = kind(str(word))
    except KindError as exc:
        raise MigrateError(f"{path.name}: its scheduler {word!r} -- {exc}; "
                           f"nothing was written") from None
    raw["scheduler"] = {"kind": k.name, **dict(k.commands)}
    raw["schema"] = SCHEMA
    rules, renamed = {"2.10.1"}, []
    if "script_generation" in raw:
        old, new = raw.pop("script_generation"), raw.get("env_init")
        if new not in (None, {}) and new != old:
            raise _renamed_conflict("script_generation", old, "env_init",
                                    new, where=path.name,
                                    renamed="2026-10-02")
        raw["env_init"] = old
        rules.add("2026-10-02")
        renamed.append(f"{path.name}: script_generation -> env_init "
                       f"(2026-10-02: how a shell there enters its "
                       f"environment, under its name now; its values "
                       f"kept)")
    try:
        env = Environment.from_dict(raw)
    except (ValueError, TypeError, KeyError) as exc:
        raise _refused_rewrite(path, exc) from None
    writes.add(path, partial(write_environment, env, path), rules)
    said += [f"{path.name}: scheduler {word!r} -> {env.scheduler.said()} "
             f"(2.10.1: the kind and where its commands are; each run "
             f"from PATH, as before)"] + renamed


@dataclass(frozen=True)
class _Rename:
    """One run's file renamed: ``old`` to ``new`` in its folder, every byte
    kept."""
    old: Path
    new: Path


def _runs_with_folders(base: Path) -> "Dict[Path, list]":
    """Every run of the calculation in ``base`` -- a stage's or a trial's
    attempt, a run of points with its root and each point's folder below
    it -- by its run's folder (`calcdirs.Where.run_folder`), each folder
    with its stage token, as its record places it (`calcdirs.where`);
    hidden folders are not walked."""
    import os
    from .. import calcdirs
    runs: Dict[Path, list] = {}
    for d, dirs, _files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        w = calcdirs.where(d)
        if w.stage and (w.what == calcdirs.ATTEMPT
                        or (w.what == calcdirs.POINT
                            and w.role == calcdirs.RUN)):
            runs.setdefault(w.run_folder or Path(d), []).append(
                (Path(d), w.stage))
    return runs


def _migrate_runs(base: Path) -> Tuple[List[str], List["_Rename"]]:
    """THE RUNS AN OLDER MOLBUILDER NAMED BY THE TRY (`runfiles.older_named`,
    before `runfiles.ONE_START_SINCE`), each file named once: ``<stem>-run0
    <role>`` as ``<stem><role>`` -- SIESTA's history as SIESTA names it, on
    the label alone, the name that run script pointed at the numbered file
    -- the start's session log ``<stem>.runwrap-<stamp>.log`` as
    ``<stem>.runwrap.log``, and a file an earlier migration kept beside its
    rewrite under the same rule: every byte kept, the old name said.
    Decided once per run, its root and its points' folders together: a run
    any folder of which started more than once (a ``-run1``, a second
    session log), whose job this machine holds and is going or cannot ask,
    any folder of which is not writable here, a new name a file already
    holds or two of its files would take, or a role this molbuilder does
    not name -- a kept copy's excepted -- is LEFT AS IT IS, whole, and said: it is not read
    (`job-request.md` § 8, its five reasons).  A run no job of this
    machine's holds is read as the copy it is.  Returns what is said and
    the renames, decided before any is made."""
    import os
    from .. import runfiles as _rf
    from ..joblist import going_of
    said: List[str] = []
    moves: List[_Rename] = []
    # THE FILES THIS MOLBUILDER WRITES -- the catalogue's own rows: an
    # older run's file is renamed onto one of them, an engine's file keeps
    # the name its engine gave it.
    known = set(_rf.roles(engines=False))
    for run, folders in _runs_with_folders(base).items():
        found = [(folder, stage, _rf.older_named(folder, stage))
                 for folder, stage in folders]
        found = [f for f in found if f[2]]
        if not found:
            continue
        rel = run.relative_to(base)
        stuck: List[str] = []
        for folder, _stage, older in found:
            # A KEPT FILE COUNTS: a copy an earlier migration kept of a
            # second try's file is a second start all the same.
            tries = sorted({o.tried for o in older if o.tried is not None})
            logs = [o for o in older if o.tried is None and o.kept is None]
            if tries not in ([], [0]) or len(logs) > 1:
                stuck.append(f"{folder.relative_to(base)} started more than "
                             f"once (tries {', '.join(map(str, tries)) or '0'}"
                             f", {len(logs)} session logs), and a run is one "
                             f"start since {_rf.ONE_START_SINCE}")
            elif not os.access(folder, os.W_OK):
                stuck.append(f"{folder.relative_to(base)} cannot be written "
                             f"here")
        going, _job, how = going_of(base, run)
        if going is not False:
            stuck.append(f"its job {'is still going' if going else how}; "
                         f"run the migration again once it has ended")
        run_moves: List[_Rename] = []
        lines: List[str] = []
        taken: set = set()
        for folder, stage, older in found if not stuck else ():
            names = []
            for o in older:
                # ONE CHECK: a file named on the stage is one of the
                # catalogue's own rows; SIESTA's history keeps the name SIESTA
                # gives it.  A KEPT COPY IS AN EARLIER MIGRATION'S OWN, of a
                # role this molbuilder may no longer write, and is named anew
                # every byte kept.
                if (o.kept is None and o.staged
                        and _rf.canonical_role(o.role)[0] not in known):
                    stuck.append(f"{o.path.name} names no file this "
                                 f"molbuilder writes")
                    continue
                new = o.path.with_name(o.anew(stage))
                # SIESTA'S NAME FOR ITS HISTORY, still pointing at the
                # numbered file where its run script stopped before its
                # exit: the rename takes its place.
                pointer = (new.is_symlink()
                           and new.resolve() == o.path.resolve())
                if (new.exists() or new.is_symlink()) and not pointer:
                    stuck.append(f"{new.name} is there beside "
                                 f"{o.path.name}")
                    continue
                if new in taken:
                    stuck.append(f"two of its files would be named "
                                 f"{new.name}")
                    continue
                taken.add(new)
                run_moves.append(_Rename(o.path, new))
                names.append(f"{o.path.name} -> {new.name}")
            lines.append(f"  {folder.relative_to(base)}: {', '.join(names)}")
        if stuck:
            said.append(f"{rel}: left as it is, and not read -- "
                        + "; ".join(stuck))
            continue
        moves += run_moves
        said += [f"{rel}: its files named anew ({_rf.ONE_START_SINCE}: a run "
                 f"is one start, its files named once -- on its stage's "
                 f"stem, SIESTA's history under SIESTA's own name; every "
                 f"byte kept):"] + lines
    return said, moves


def migrate_state(base) -> List[str]:
    """Rewrite the calculation in ``base`` -- every step in the module's
    docstring, each decided before any is written -- and return what
    changed, line by line.  Each old file rewritten
    is kept beside its new one, once (:func:`_kept_as`).

    No line when there is nothing to migrate -- the calculation is current
    (:data:`CURRENT`); raises :class:`MigrateError` when it cannot be
    migrated as it stands.
    """
    # RESOLVED ONCE: the runs' folders are placed as their records place
    # them (`calcdirs.where`), and every path said is said from here.
    base = Path(base).resolve()
    # EVERY RECORD READ ONCE, then EVERY DECISION BEFORE ANY WRITE: a
    # refusal of any step leaves the calculation as it was, never half
    # rewritten (`architecture.md` A16, migrate's rule).
    rec = _Records.read(base)
    said, writes, task = _migrate_records(rec)
    _migrate_machine_record(rec, said, writes)
    try:
        decision = _migrate_template(rec, task)
    except MigrateError:
        raise
    except ValueError as exc:
        # A READER'S REFUSAL met while deciding the template's rewrite is
        # the migration's, in the reader's words.
        raise MigrateError(f"the template's rewrite cannot be decided, so "
                           f"nothing was written: {exc}") from None
    r_said, moves = _migrate_runs(base)
    t_said: List[str] = []
    if decision is not None:
        t_said = list(decision.said)
        writes.add(decision.path,
                   lambda *, plan, d=decision: plan.text(d.path, d.text),
                   decision.rules)
        if decision.task_rules:
            # THE DESCRIPTION AS EVERY STEP LEFT IT: the stages' old zeros,
            # renamed and retired items taken out of the records step's.
            from ..task import path_of, write_task
            writes.add(path_of(base),
                       partial(write_task, base, _edited(task, decision)),
                       decision.task_rules)
    if not writes and not moves:
        # NOTHING TO MIGRATE is an answer, not a refusal -- a run left as it
        # is said all the same, and written down.
        if r_said:
            from . import ledger
            ledger.record(base, "migrate", "runs an older molbuilder named, "
                          "left as they are", changes=r_said)
        return r_said
    # A KEPT FILE IS NEVER WRITTEN OVER: it is the only copy of what the
    # file was before an earlier migration.
    # A NEW FILE -- the job request a split writes -- keeps nothing.
    kept = {path: _kept_as(path, rules)
            for path, (_write, rules) in writes.items()
            if path.exists()}
    held = [k for k in kept.values() if k.exists()]
    if held:
        raise MigrateError(
            f"{', '.join(str(p) for p in held)} already "
            f"{'holds' if len(held) == 1 else 'hold'} an earlier version -- "
            f"a kept file is never written over; nothing was written.  Move "
            f"{'it' if len(held) == 1 else 'them'} aside and run the "
            f"migration again.")
    # ONE PLAN, CARRIED OUT IN ITS ORDER (`jobset.planned.Plan`): each old
    # file kept, then its rewrite by its owner's writer; then each run's
    # file renamed, after every record is written -- a rename rewrites no
    # byte, so nothing is kept beside it: its old name is said, and the
    # ledger keeps it.
    from .planned import CutShort, Plan
    plan = Plan()
    for path, (write, _rules) in writes.items():
        if path in kept:
            plan.bytes(kept[path], path.read_bytes())
        write(plan=plan)
    for m in moves:
        plan.move(m.old, m.new)
    try:
        plan.carry_out()
    except CutShort as exc:
        # HOW FAR IT GOT, in the ledger: what was carried out before the
        # write that failed, and that write.
        from . import ledger
        ledger.record(base, "migrate", "a calculation an older "
                      "molbuilder wrote, rewritten -- stopped at a write "
                      "that failed", changes=said + exc.done,
                      failed=f"{exc.failed}: {exc.cause}")
        raise MigrationStopped(
            f"{exc}; each one carried out before it is written in the "
            f"ledger") from None
    said += [f"{path.relative_to(base)} kept as {kept[path].name}"
             for path in writes if path in kept]
    said += t_said + r_said
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
    from ..engine_table import config_class
    from ..runfiles import TASK_FILE

    engine = task.engine
    kind = task.calculation
    base = rec.base
    template = rec.template_for(task.label)
    if template is None:
        raise _NothingInTheTemplate(f"{base} holds no template.")
    tmpl, text = template.path, template.text
    parsed = _template_read(template)     # read against its OWN declarations
    try:
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
                                        f"stage {st.name!r} in {TASK_FILE}")
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
                     f"{TASK_FILE} (renamed)" for st, k, v in stage_renamed]
                    + [f"{k} -> {renamed_to[k]} in {TASK_FILE}'s varies "
                       f"(renamed)" for k in varies_renamed])
    # AN ITEM A RELEASE RETIRED is taken out -- of the template, and of a
    # stage's own values and `varies` -- when its value runs as before
    # without it, and refused by name otherwise: what the calculation was is
    # kept, never rounded (`template.Retired.runs`).
    # ...ON THIS ENGINE: an item one engine keeps is another's to retire
    # (`template.Retired.engines`).
    def _gone(name: str) -> bool:
        return _T.retired(name, engine) is not None

    retired = [it for it in mine if _gone(it.name)]
    stage_retired = [(st.name, k, v) for st in (task.stages or ())
                     for k, v in (st.overrides or {}).items() if _gone(k)]
    varies_retired = [k for k in (task.varies or ()) if _gone(k)]
    # A STAGE'S OVERRIDE A RETIRED ROW SAYS IS THE PERSON'S is not taken
    # out, but refused naming the row's way out (`Retired.on_a_stage`) --
    # SIESTA's Raman selector: it decided the stage's optical phase, and the
    # template's mapping (:func:`_map_raman`) speaks for the template alone.
    # A `varies` entry alone decides nothing, and is taken out as any
    # retired item's.
    refused = [(st, k) for st, k, _v in stage_retired
               if _T.RETIRED_ITEMS[k].on_a_stage]
    if refused:
        raise MigrateError(
            "; ".join(f"stage {st!r} overrides {k}: "
                      f"{_T.RETIRED_ITEMS[k].on_a_stage}"
                      for st, k in refused)
            + ".  Nothing was written: do that, then run the migration "
            "again.")
    stuck = ([(f"the template says {it.name} = {it.value!r}", it.name)
              for it in retired
              if it.is_set
              and not _T.RETIRED_ITEMS[it.name].runs_without(it.value)]
             + [(f"stage {st!r} overrides {k} = {v!r}", k)
                for st, k, v in stage_retired
                if not _T.RETIRED_ITEMS[k].runs_without(v)])
    if stuck:
        raise _retired_stuck(*stuck[0])
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
        + [f"stage {st!r} {k} = {v!r} removed from {TASK_FILE} "
           f"({_T.RETIRED_ITEMS[k].why})" for st, k, v in stage_retired]
        + [f"{k} removed from {TASK_FILE}'s varies "
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
                    f"taken out of {TASK_FILE}; the template's {name} is "
                    f"written blank, since only that stage reads it)"
                    for st, name in stage_zeros])
    zeros = list(dict.fromkeys(zeros + [name for _st, name in stage_zeros]))
    # AN ITEM NO RELEASE HELD is refused by name, as the reader refuses it
    # (`template.config_from_template`): a misspelling -- carried by no
    # item, it would be dropped unsaid.
    config_cls = config_class(engine)
    known = _T.template_fields(config_cls)
    # A RUN SETTING, valueless or not, is taken out: the template holds
    # none (`engines/template.md` § 2) -- what it holds decides, never its
    # version -- its value moved onto the calculation's run card by the
    # records step (:func:`_onto_the_card`); and the calculation's card's
    # `restart` and `density_fit`, which say what is computed, come in.
    ours = _T.run_settings(engine)
    run_items = sorted(it.name for it in mine if it.name in ours)
    vals = {k: v for k, v in vals.items() if k not in run_items}
    said = _T.why_not_items([k for k in vals if k not in old_names
                             and not _gone(k)], known,
                            among=_T.template_settings(config_cls))
    if said:
        raise MigrateError(
            f"{tmpl.name}: {said}.  Prep refuses it too: correct it, then "
            f"run the migration again -- nothing was written.")
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
            f"warm files are a density for one state"
            f".  Decide which "
            f"value the whole calculation carries, put it in the template, "
            f"remove the override from {TASK_FILE}, and run the migration "
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
