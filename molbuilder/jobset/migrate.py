"""Rewrite a calculation an older molbuilder wrote -- the one road for a file
a reader refuses naming this command.  Every record is read once
(:class:`_Records`), every step decided before any file is written, and a
refusal of any step writes nothing.  **Each old file rewritten is kept
beside its new one, once, as** ``<name>.pre-<rule>`` -- the earliest rule it
was written before (:func:`_kept_as`): a plan's code (``m6``, ``w57``), a
day, or a release.  A file renamed keeps every byte, so nothing is kept
beside it.  Each change is said, with its rule.

**The records** (:func:`_migrate_records`).  `task.json`: its `calculation`
(W57), a notify block's four values and *never* as ``null`` (W57); a stage's
`enabled` (2026-10-03) -- a ``true`` removed, a stage switched off refused by
name, since whether it runs is the person's; `centre_junction` removed
(2.6.0, the junction enters its calculation centred); a transport
description's voltages, ``[0.0]`` where it stated none (2.16.3); the
low-bias switch from the word it replaced (2026-10-08); every stage's
`group` (2.7.0, :func:`_migrate_groups`) -- a prepared stage its job's, the
rest the number above where the ladder allows it.  Each `job-set.json` --
the root's and each sweep's: a job's `point`, `finish`, `resumes`,
`placement` and `members` (W57; `members` written as `group` before 2.7.0),
its `written_at` and `asked` (2.7.0) and its `scratch_dir` (2.22.0), each
with the meaning its absence had.

**The machine's record** (:func:`_migrate_machine_record`): a calculation's
copy names its scheduler as a block since 2.10.1 (`configuration.md` § 5
M-2) -- the word it held becomes its kind, each command its kind's own name,
run from PATH as before.

**The run files**: a flat calculation's files that carry no run's number
(W57, :func:`_migrate_run_numbers`), renamed for their stage's newest run;
a transport lead's files on the calculation's label rather than its own
stem (:func:`_migrate_lead_names`) -- every file before 2.17.0, its prep's
log before 2.23.2 -- renamed, and the plan's deck names with them; a flat
folder's last run (:func:`_migrate_last_run`), recorded for runs launched
before 2.7.0.  A stage whose run script was written for the old names is
said with its way on.

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
stage's own values and `varies` alike.  **What the rewrite carries is the
old file's, whole**: every value, every blank and each item's source as the
file records it (`engines/template.md` § 5); what it cannot carry is said --
a value of an item no longer of this kind, the comments a person wrote
inside the file.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional,
                    Tuple)

if TYPE_CHECKING:                      # annotations only
    from ..calculation import Calculation
    from ..template_file import TemplateFile


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
               # A PREP BEFORE 2.7.0 recorded no `relax` run, and nothing it
               # was told: launch's check plans it as told nothing beside
               # the description, and one told flags is refused there,
               # naming its prep again.
               "written_at": (None, "2.7.0"),
               "asked": ({"allocation": {}, "env": None,
                          "emit_sbatch": True}, "2.7.0"),
               # A PREP BEFORE 2.22.0 kept no scratch folder: its device
               # deck names the leads' files loose in the run, and a launch
               # again of it empties nothing (`engines/transport.md`
               # § 2a.11); a prep again gives the stage the folder.
               "scratch_dir": (None, "2.22.0")}
#: A job's key an older file wrote under another name -- renamed, its value
#: kept: the members of the job it was prepared in (2.7.0).
_JOB_RENAMED = {"group": ("members", "2.7.0")}

#: The rules named by their plan's code, at the day each took effect; a
#: rule named by its day says it itself, and every numbered release came
#: after them all (2.0.0, 2026-10-11).
_RULE_DAYS = {"m6": "2026-09-28", "w57": "2026-10-06"}


def _kept_as(rules) -> str:
    """``.pre-<rule>`` -- the name an old file is kept under beside its
    rewrite, once: the EARLIEST of the ``rules`` it was written before, each
    a plan's code (:data:`_RULE_DAYS`), a day or a release (the template's
    retired and renamed items carry either); two rules of one day are told
    apart by name, so one file is kept under one name on every run."""
    import re
    from ..template import version_key

    def when(rule: str):
        day = _RULE_DAYS.get(rule, rule)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            return (0, day, (), rule)
        return (1, "", version_key(rule), rule)
    return ".pre-" + min(rules, key=when)


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
    machine's record and the run records are read once each by the step
    that rewrites them, and the write keeps each old file, byte for byte,
    beside its new one."""
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
        jobsets: Dict[Path, dict] = {}
        for folder in [base, *sweep_set_dirs(base)]:
            path = JobSet.path_of(folder)
            if path.is_file():
                jobsets[folder] = _read_raw(path)
        return cls(base=base, task=obj, jobsets=jobsets)

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
                           jobset=js)

    def template_for(self, label: str) -> Optional["TemplateFile"]:
        """The calculation's template, read once (`template_file.of`); a folder
        with two, or one under another name, is refused by name."""
        if not self._template_read:
            from ..template_file import of as template_of
            try:
                self.template = template_of(self.base, label)
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
    (`group.upstream_of`) and their run cards share one allocation
    (`group.card_shares`) -- else one more.  A prepared group whose
    members are not one after another in the ladder cannot be stated by a
    number, and is refused by name: which jobs to keep is the person's."""
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
    # description with each stage a group of its own...
    task = Task.from_dict({**obj, "stages": [
        {**st, "group": i + 1} for i, st in enumerate(stages)]})
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
                 and may_join(task, name, members, text))
        groups.append(groups[-1] if joins else (groups[-1] + 1 if groups
                                                else 1))
        members = members + [name] if joins else [name]
    for st, g in zip(stages, groups):
        st["group"] = g
    return [f"stages' groups {' '.join(map(str, groups))} (2026-10-11: "
            f"each stage states the job it runs in -- a prepared one its "
            f"job's, the rest the number above where the ladder allows it "
            f"and the run cards share one allocation)"]


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
        # `jobset init` with no `--bias` wrote no voltages before, and the
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
    try:
        task = Task.from_dict(obj)
    except ValueError as exc:
        raise _refused_rewrite(task_path, exc) from None
    if lines:
        writes.add(task_path, json_text(task.to_dict()),
                   {rule for _ln, rule in lines})
        said += [f"{TASK_FILENAME}: {ln}" for ln, _rule in lines]

    for folder, d in rec.jobsets.items():
        js_path = JobSet.path_of(folder)
        filled, missing, renamed = 0, set(), set()
        for job in d.get("jobs") or []:
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
        if not filled:
            continue
        try:
            body = JobSet.from_dict(d).to_dict()
        except (ValueError, KeyError, TypeError) as exc:
            raise _refused_rewrite(js_path, exc) from None
        writes.add(js_path, json_text(body),
                   {_JOB_ABSENT[k][1] for k in missing}
                   | {_JOB_RENAMED[o][1] for o in renamed})
        said.append(f"{js_path.relative_to(base)}: {filled} job(s) -- "
                    + "; ".join(
                        ([f"given {', '.join(k for k in _JOB_ABSENT if k in missing)}"
                          f" (left out, as their absence meant)"]
                         if missing else [])
                        + [f"`{o}` renamed `{_JOB_RENAMED[o][0]}`"
                           for o in sorted(renamed)]))
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


def _migrate_run_numbers(calc) -> Tuple[List[str], List[Tuple[Path, Path]]]:
    """A flat calculation's run files written before each carried its run's
    number (plan W57 decisions 2 and 6): every file of a stage whose role
    the catalogue numbers where a stage's runs share a folder, still
    unnumbered (`runrecord.unnumbered_files`), given the number of its
    stage's newest run -- 0 for a stage never launched.  A rename keeps
    every byte, so nothing is kept beside it.  Decided, never done here:
    returns what it says (a line each) and the moves.

    **What it cannot fix is said too**: such a stage was prepared before
    launch gave each run its number, so its run script takes no ``--run`` --
    to launch it, its group is prepared again; launched, from the state
    saved before its group's first launch (`project-layout.md` § 1.6.1)."""
    base, task = calc.base, calc.task
    if task.shape != "flat":
        return [], []
    from ..runfiles import FIRST_ATTEMPT, RunNames, latest_run, parse
    from ..runrecord import unnumbered_files
    from .commands import way_back
    from .naming import ladder_homes
    said: List[str] = []
    moves: List[Tuple[Path, Path]] = []
    for home in ladder_homes(base, task):
        # AN OLDER RELEASE'S FILES, which were on the calculation's label
        # for every stage -- read as they were written, not as `home.names`
        # names them now.
        names = RunNames.of(task.label, home.token, task.shape)
        old = unnumbered_files(base, names)
        if not old:
            continue
        newest = latest_run(base, task.label, stage=home.token)
        n = FIRST_ATTEMPT if newest is None else newest
        stage_moves = [(f, base / names.name(parse(f.name, task.label).role,
                                              n))
                       for f in old]
        moves += stage_moves
        said += [f"{f.name} -> {new.name} (run {n}, the stage's newest)"
                 for f, new in stage_moves]
        # LAUNCHED, whether its launch record is numbered yet or not: one
        # written before the numbering is among the files moved here.
        launched = (newest is not None
                    or any(parse(f.name, task.label).role == ".run.json"
                           for f in old))
        said.append(f"{home.name}: its run script was written before launch "
                    f"gave each run its number (--run N) -- "
                    + ("to launch it again, " if launched
                       else "to launch it, ")
                    + way_back(home.name, calc=calc, launched=launched))
    return said, moves


def _migrate_lead_names(calc) -> Tuple[List[str], List[Tuple[Path, Path]],
                                       "_Writes"]:
    """A transport lead's files named on the calculation's label; a lead's
    are on its own stem (`job-contracts.md` § 2.1 rule 2;
    `identity.rung_label`), the stem its engine files were always
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
    from ..runfiles import RunNames, find
    from ..runrecord import newest_run
    from .commands import way_back
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
        bench = base / bench_container(calc.shape, home.token)
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
        # LAUNCHED, read off the old names as they were written -- the run
        # door's own `newest_run`, asked with the names the files carry now.
        old_names = RunNames.of(task.label, home.token, task.shape)
        launched = any(newest_run(f, old_names) is not None for f in own)
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


#: When the run script began recording a flat folder's last run -- the
#: release that brought it, 2.7.0, shipped 2026-10-13 16:41 UTC
#: (`project-layout.md` § 5, the `<label>.last-run` row: `jobset migrate`
#: writes it for runs launched before), in a launch record's own spelling.
_LAST_RUN_RECORDED_FROM = "2026-10-13T16:41:03Z"


def _migrate_last_run(base: Path, task, moves
                      ) -> Tuple[List[str], Optional[Tuple[Path, str]]]:
    """A flat calculation whose runs were launched before its folder recorded
    its last run (2.7.0): ``<label>.last-run`` written for the run its
    readers took as the last until then -- the newest launch record from
    before that release, by when it was written, then its stage's place in
    the ladder, then its run's number -- read where each record will lie
    once ``moves`` (the run numbering) are made.  A run launched since
    records its own as its engine starts, so one still queued, or stopped
    before it started, is never named.  Decided, never done here: returns
    what it says and the write; nothing for a folder that holds the record,
    or launched no run before it."""
    if task.shape != "flat":
        return [], None
    from ..identity import parse_token
    from ..persist import read_json
    from ..runfiles import find_by_role, parse, run_name
    from ..runrecord import last_run_path
    target = last_run_path(base, task.label)
    if target.is_file():
        return [], None
    renamed = dict(moves)
    best, key = None, None
    for f in find_by_role(base, ".run.json"):
        rec = parse(renamed.get(f, f).name, task.label)
        if rec is None or rec.stage is None or rec.run is None:
            continue
        try:
            body = read_json(f)
        except (OSError, ValueError) as exc:
            raise MigrateError(
                f"{f} does not read ({exc}) -- which run last wrote this "
                f"folder cannot be told; nothing was written.") from None
        when = str((body or {}).get("launched_at") or "")
        if when >= _LAST_RUN_RECORDED_FROM:
            continue
        token = parse_token(rec.stage)
        k = (when, token[0] if token else -1, rec.run)
        if key is None or k > key:
            best, key = run_name(task.label, rec.stage, rec.run), k
    if best is None:
        return [], None
    return ([f"{target.name}: {best} -- the last run here, by the rule that "
             f"held before 2.7.0 (its newest launch record)"],
            (target, best + "\n"))


def migrate_state(base) -> List[str]:
    """Rewrite the calculation in ``base`` -- every step in the module's
    docstring, each decided before any is written -- and return what
    changed, line by line.  Each old file rewritten is kept beside its new
    one, once (:func:`_kept_as`); a renamed one is its new one.

    Raises :class:`MigrateError` when there is nothing to migrate or it cannot
    be migrated as it stands.
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
    n_said, moves = _migrate_run_numbers(calc)
    lead_said, lead_moves, lead_writes = _migrate_lead_names(calc)
    moves += lead_moves
    for path, (text, rules) in lead_writes.items():
        writes.add(path, text, rules)
    l_said, l_write = _migrate_last_run(base, task, moves)
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
    if not writes and not moves and l_write is None:
        raise MigrateError(
            f"{base} holds nothing an older molbuilder wrote -- its "
            f"records are whole, its record of its machine is this "
            f"molbuilder's, its run files carry their run's number and its "
            f"leads' their own stem, and its template carries no item an "
            f"older molbuilder meant otherwise, renamed or a release "
            f"retired.")
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
    kept = {path: path.with_name(path.name + _kept_as(rules))
            for path, (_text, rules) in writes.items()}
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
        write_bytes(kept[path], path.read_bytes())
        write_bytes(path, text.encode("utf-8"))
    for old, new in moves:
        old.rename(new)
    if l_write is not None:
        write_bytes(l_write[0], l_write[1].encode("utf-8"))
    said += [f"{path.relative_to(base)} kept as {kept[path].name}"
             for path in writes]
    said += n_said + lead_said + l_said + t_said
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
    retired = [it for it in mine if it.name in _T.RETIRED_ITEMS]
    stage_retired = [(st.name, k, v) for st in (task.stages or ())
                     for k, v in (st.overrides or {}).items()
                     if k in _T.RETIRED_ITEMS]
    varies_retired = [k for k in (task.varies or ())
                      if k in _T.RETIRED_ITEMS]
    stuck = ([(f"the template says {it.name} = {it.value!r}", it.name)
              for it in retired
              if it.is_set and it.value not in _T.RETIRED_ITEMS[it.name].runs]
             + [(f"stage {st!r} overrides {k} = {v!r}", k)
                for st, k, v in stage_retired
                if v not in _T.RETIRED_ITEMS[k].runs])
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
                               if k not in _T.RETIRED_ITEMS),
            stages=tuple(dataclasses.replace(st, overrides={
                k: v for k, v in (st.overrides or {}).items()
                if k not in _T.RETIRED_ITEMS}) for st in task.stages))
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
    # (`template.config_from_template`): a misspelling, or a value written
    # on a machine fact -- carried by no item, it would be dropped unsaid.
    config_cls = engine_seam(engine).config_cls
    known = _T.template_fields(config_cls)
    unknown = sorted(k for k in vals if k not in known
                     and k not in old_names and k not in _T.RETIRED_ITEMS)
    if unknown:
        raise MigrateError(
            f"{tmpl.name} names {', '.join(map(repr, unknown))}, which no "
            f"item holds -- a misspelling, or a value written on a machine "
            f"fact (engines/template.md § 7); prep refuses it too.  Correct "
            f"it, then run the migration again; nothing was written.")
    if not (old_items or old_values or zeros or renamed or retired
            or stage_retired or varies_retired):
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
    new_text = _T.template_with_values(cfg, engine=engine, calculation=kind,
                                       title=_leading_comment(text),
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


def _leading_comment(text: str) -> str:
    """The file's header comment, without its ``#`` -- what ``init`` wrote as
    the title, carried over unchanged."""
    lines = []
    for ln in text.splitlines():
        if not ln.startswith("#"):
            break
        lines.append(ln[2:] if ln.startswith("# ") else ln[1:])
    return "\n".join(lines)
