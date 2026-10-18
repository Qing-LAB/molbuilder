"""Rewrite a calculation an older molbuilder wrote: its records written whole
(2026-10-06), its stages' groups (2026-10-11), its machine record's scheduler
(2.10.1), a flat calculation's run files
numbered (2026-10-06), its template's electronic-state items (2026-09-28),
its template's zeros that meant *not set* (2.1.0), and the template items a
release retired (2.10.0).

**The run files** (`_migrate_run_numbers`): in the flat shape each run's
files -- every row the catalogue numbers where a stage's runs share a
folder (`runfiles.shared_numbered_roles`) -- carry the run's
number since 2026-10-06 (plan W57 decisions 2 and 6), and the launch
record's door refuses a stage whose still do not, naming this command.  Each
is renamed for its stage's newest run, and said.

**The records** (`_migrate_records`): `task.json` states its `calculation`
and a notify block its four values, and every job in a `job-set.json` its
`point`, `finish`, `resumes` and `placement` since 2026-10-06, its `group`
since 2026-10-07, and its `written_at` and `asked` since 2026-10-11 -- the
readers refuse a file that does not, naming this command (plan W57, decision
7).  Each key an older file left out is written with the meaning its absence
had -- `optimization`, every channel and every field (`["*"]`), off, an empty
point, no finish, resumes, no placement, prepared alone, no `relax` run on
record, nothing said beside the description -- and each is printed; the old
file
is kept beside the new as ``<name>.pre-w57``.  A key a release retired is
removed and said: `task.json`'s `centre_junction` (2.6.0, the junction enters
its calculation centred), the old file kept as ``<name>.pre-2.6.0`` when that
is the earliest change.

**The machine's record** (`_migrate_machine_record`): a calculation's copy
of its machine's record names its scheduler as a block since 2.10.1 -- its
kind and where its commands are (`configuration.md` § 5 M-2) -- and the
reader refuses an older copy, naming this command.  The word it held becomes
its kind, each command its kind's own name (`scheduler/kinds.toml`), run from
PATH as the release before ran it; the old file is kept as
``<name>.pre-2.10.1``.

**The groups** (`_migrate_groups`): every stage states the job it runs in
since 2026-10-11 (`engines/stages.md` § 2) -- a prepared stage its job's, the
rest the number above where the ladder allows it -- the old file kept as
``<name>.pre-2.7.0`` when that is the only change.

**The template**, as follows.

``docs/plans/plan.md`` § 5s.2, decision 7: *existing templates are migrated,
not tolerated* -- this is the one-time command that rewrites the old items
(PySCF's ``spin`` and the R/U inside its ``method``, SIESTA's
``spin_treatment`` spellings and ``spin_total``) into the four the class reads
(``science/chemistry-correctness.md`` § 2a).  Nothing reads the old names
afterwards; ``template.config_from_template`` refuses a template that still
carries one, naming this command.

**What the run was is kept.**  Every old value becomes a STATED value -- an
old template's ``non-polarized`` becomes ``restricted``, not a blank the
structure would now decide -- because the calculation the file described must
not change under a rename.  The output says so, item by item, and a person
who wants the structure to decide blanks the item afterwards.

A value the new items cannot express is refused by name rather than
rounded: a fractional ``Spin.Total`` is a fixed moment, not a count of
unpaired electrons.  A stage override of any of the four is refused too --
the state belongs to the calculation, never to a stage (ES1) -- naming the
stage, because which value the whole calculation should carry is the
person's to say.

The old template is kept beside the new one as ``<name>.pre-m6``.

**The zeros that meant not set** (2.1.0, `engines/transport.md` § 2a.13):
before 2.1.0 a 0 for ``negf_neq_eta_ev`` or ``tbt_contours_eta_ev`` wrote no
line and left the broadening to the program's own rule; since, a 0 is written
as zero broadening and *not set* is the item left blank.  A calculation
whose template was written before 2.1.0 (its ``written_by`` older, or absent)
holding such a 0 -- in the template, refused by
``template.config_from_template``, or as a rung's override in `task.json`,
refused by ``resolve`` -- names this command, which writes the item blank --
what it meant: a rung's override is taken out too, since each item is read
by one rung alone -- says so, and keeps each old file as
``<name>.pre-2.1.0`` (``.pre-m6`` / ``.pre-w57`` when an earlier step
rewrites it too).

**The items a release retired** (`template.RETIRED_ITEMS`, the reader's
refusal naming this command), and **an item renamed**
(`template.RENAMED_ITEMS`): a retired item is taken out -- of the template,
of a stage's own values and of `varies` -- when its value runs as before
without it (`Retired.runs`: `copy_psml`, which nothing read; PySCF's
`optimizer = "geometric"`), each said with its value and why it went; any
other value is refused by name, since taking it out would change the
calculation.  A renamed item keeps its value, or its blank, under its new
name.  Each old file is kept as ``<name>.pre-<the release or day that
retired or renamed it>`` when no earlier step rewrites it.

**What the rewrite carries is the old file's, whole**: every value, every
blank, and each item's source as the file records it -- none where it
records none, *not recorded* (`engines/template.md` § 5) -- and what it
cannot carry is said: a value of an item no longer of this kind, the
comments a person wrote inside the file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

#: The items the electronic state RETIRED -- the only names a template may
#: no longer carry.  ``method``, ``spin_treatment`` and ``net_charge`` are
#: current items: an OLD VALUE in one is rewritten, a current value is the
#: file's own statement and is kept.
_OLD_NAMES = ("spin", "spin_total")


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


#: What a key an older file left out MEANT -- written in its place, so the
#: rewritten file says what the old one said by omitting it.
_TASK_ABSENT = {"calculation": "optimization"}
_NOTIFY_ABSENT = {"on_scf_converged": False, "every_hours": None,
                  "channels": ["*"], "report": ["*"]}
_JOB_ABSENT = {"point": {}, "finish": None, "resumes": True,
               "placement": None, "members": None, "written_at": None,
               # A PREP BEFORE 2026-10-11 recorded nothing it was told:
               # launch's check plans it as told nothing beside the
               # description, and one told flags is refused there, naming
               # its prep again.
               "asked": {"allocation": {}, "env": None,
                         "emit_sbatch": True}}
#: A job's key an older file wrote under another name -- renamed, its value
#: kept: the members of the job it was prepared in (2.7.0).
_JOB_RENAMED = {"group": "members"}


def _never_as_zero(hours) -> bool:
    """Whether a period is the older spelling of never, ``0``."""
    return (isinstance(hours, (int, float)) and not isinstance(hours, bool)
            and hours == 0)


def _migrate_groups(base: Path, obj: dict) -> List[str]:
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
    from ..persist import read_json
    from ..task import Task
    from ..template import find_template
    from .commands import rollback
    from .group import may_join
    from .model import FILENAME as JOBSET_FILENAME
    # THE LADDER'S LINKS do not depend on the numbers: read on the
    # description with each stage a group of its own.
    task = Task.from_dict({**obj, "stages": [
        {**st, "group": i + 1} for i, st in enumerate(stages)]})
    try:
        tpl = find_template(base, task.label)
    except ValueError as exc:
        raise MigrateError(str(exc)) from exc
    text = tpl.read_text(encoding="utf-8") if tpl is not None else None
    if tpl is not None:
        # THE TEMPLATE AS THIS MIGRATION WRITES IT (:func:`_migrate_template`)
        # -- every decision before any write -- since the links are read on
        # the ladder it resolves, and an older template's items are refused.
        try:
            text = _migrate_template(base, task)[1][0][1]
        except _NothingInTheTemplate:
            pass
    # THE JOB SET AS AN OLDER PREP WROTE IT -- a description with no group
    # is one 2.7.0 never prepared, and before it a job's members were
    # written under `group` (`_JOB_RENAMED`).
    js = base / JOBSET_FILENAME
    jobs = {stage_key(j["name"]): [stage_key(n) for n in (j.get("group")
                                                            or [j["name"]])]
            for j in ((read_json(js).get("jobs") or [])
                      if js.is_file() else [])}
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
                 and may_join(base, task, name, members, text))
        groups.append(groups[-1] if joins else (groups[-1] + 1 if groups
                                                else 1))
        members = members + [name] if joins else [name]
    for st, g in zip(stages, groups):
        st["group"] = g
    return [f"stages' groups {' '.join(map(str, groups))} (2026-10-11: "
            f"each stage states the job it runs in -- a prepared one its "
            f"job's, the rest the number above where the ladder allows it "
            f"and the run cards share one allocation)"]


def _migrate_records(base: Path):
    """`task.json` and every `job-set.json` of the calculation, each key an
    older file left out given the meaning its absence had, and a key a
    release retired removed (`centre_junction`, 2.6.0) -- read raw, since
    the readers refuse such a file, and each rewritten file checked by its
    reader.  Nothing is written here: returns what changed (a line each), the
    files to write (``(path, text, kept-as suffix)``), and the description as
    rewritten."""
    from ..persist import json_text, read_json
    from ..task import FILENAME as TASK_FILENAME
    from ..task import Task
    from .materialize import sweep_set_paths
    from .model import FILENAME as JOBSET_FILENAME, JobSet
    said: List[str] = []
    writes: List[Tuple[Path, str, str]] = []

    task_path = base / TASK_FILENAME
    obj = read_json(task_path)
    lines = []
    for key, meant in _TASK_ABSENT.items():
        if key not in obj:
            obj[key] = meant
            lines.append(f"{key} = {meant!r} (it was left out)")
    retired = obj.pop("centre_junction", None) is not None
    if retired:
        # RETIRED IN 2.6.0: the junction enters its calculation centred
        # (`model/structure-periodicity.md` § 6.0).
        lines.append("centre_junction removed (2.6.0: the junction enters "
                     "its calculation centred)")
    bias = obj.get("bias")
    if isinstance(bias, dict) and len(bias.get("voltages_v") or []) > 1:
        # THE SWITCH, from the word it replaced (`engines/transport.md`
        # § 2a.10): `low-bias` was the approximation; `re-converged`, and a
        # list written before either was stated, ran the device at every
        # voltage.
        if "low_bias_approximation" not in bias:
            old = bias.pop("treatment", None)
            bias["low_bias_approximation"] = old == "low-bias"
            lines.append(f"bias.low_bias_approximation = "
                         f"{bias['low_bias_approximation']} (from "
                         f"{'treatment ' + repr(old) if old else 'a list run at every voltage'})")
    notify = obj.get("notify")
    if isinstance(notify, dict) and notify:
        for key, meant in _NOTIFY_ABSENT.items():
            if key not in notify:
                notify[key] = meant
                lines.append(f"notify.{key} = {meant!r} (it was left out)")
        if _never_as_zero(notify.get("every_hours")):
            notify["every_hours"] = None
            lines.append("notify.every_hours = None (0 was never)")
    records = len(lines) - retired
    from ..resolve import ResolveError
    try:
        grouped = _migrate_groups(base, obj)
    except MigrateError:
        raise
    except (ValueError, ResolveError) as exc:
        # THE LADDER'S LINKS ARE READ ON THE DESCRIPTION AND ITS TEMPLATE: one
        # either refuses as it stands is said in its reader's words.
        raise MigrateError(f"{task_path.name}: its stages' groups cannot be "
                           f"decided, so nothing was written: {exc}") from None
    lines += grouped
    try:
        task = Task.from_dict(obj)
    except ValueError as exc:
        raise MigrateError(f"{task_path.name} rewritten would still be "
                           f"refused, so nothing was written: {exc}")
    if lines:
        # KEPT AS WHAT IT WAS WRITTEN BEFORE: the earliest rule it was
        # written before -- the records' (W57), the release that retired a
        # key (2.6.0), the groups (2.7.0).
        writes.append((task_path, json_text(task.to_dict()),
                       ".pre-w57" if records else
                       ".pre-2.6.0" if retired else ".pre-2.7.0"))
        said += [f"{TASK_FILENAME}: {ln}" for ln in lines]

    for js_path in [base / JOBSET_FILENAME, *sweep_set_paths(base)]:
        if not js_path.is_file():
            continue
        d = read_json(js_path)
        filled, missing, renamed = 0, set(), set()
        for job in d.get("jobs") or []:
            old = {o: n for o, n in _JOB_RENAMED.items()
                   if o in job and n not in job}
            for o, n in old.items():
                job[n] = job.pop(o)
            gaps = {k: v for k, v in _JOB_ABSENT.items() if k not in job}
            if gaps or old:
                job.update(gaps)
                filled += 1
                missing |= set(gaps)
                renamed |= set(old)
        if filled:
            try:
                body = JobSet.from_dict(d).to_dict()
            except ValueError as exc:
                raise MigrateError(f"{js_path} rewritten would still be "
                                   f"refused, so nothing was written: {exc}")
            # KEPT AS WHAT IT WAS WRITTEN BEFORE: the records' rule (W57,
            # the members 2026-10-07), else the groups' release (2.7.0).
            writes.append((js_path, json_text(body),
                           ".pre-2.7.0" if missing <= {"written_at", "asked"}
                           else ".pre-w57"))
            said.append(f"{js_path.relative_to(base)}: {filled} job(s) -- "
                        + "; ".join(
                            ([f"given {', '.join(k for k in _JOB_ABSENT if k in missing)}"
                              f" (left out, as their absence meant)"]
                             if missing else [])
                            + [f"`{o}` renamed `{_JOB_RENAMED[o]}`"
                               for o in sorted(renamed)]))
    return said, writes, task


def _migrate_machine_record(base: Path
                            ) -> Tuple[List[str], List[Tuple[Path, str, str]]]:
    """The calculation's copy of its machine's record, when an older
    molbuilder wrote it (`record.older_record`): its scheduler word as the
    block -- the kind, and each command by the name the kind gives it, run
    from PATH as before (`scheduler/kinds.toml`).  Returns what changed and
    the file to write; nothing for a copy that reads, or none."""
    from ..persist import json_text, read_json
    from ..scheduler.kinds import KindError, kind
    from ..scheduler.record import (SCHEMA, Environment, calculation_record,
                                    older_record)
    path = calculation_record(base)
    if not path.is_file():
        return [], []
    raw = read_json(path)
    if not older_record(raw):
        return [], []
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
        raise MigrateError(f"{path.name} rewritten would still be refused, so "
                           f"nothing was written: {exc}") from None
    return ([f"{path.name}: scheduler {word!r} -> {env.scheduler.said()} "
             f"(2.10.1: the kind and where its commands are; each run "
             f"from PATH, as before)"],
            [(path, json_text(env.to_dict()), ".pre-2.10.1")])


def _migrate_run_numbers(base: Path, task) -> Tuple[List[str],
                                                     List[Tuple[Path, Path]]]:
    """A flat calculation's run files written before each carried its run's
    number (plan W57 decisions 2 and 6): every file of a stage whose role
    the catalogue numbers where a stage's runs share a folder
    (`runfiles.shared_numbered_roles`), still unnumbered, given the number
    of its stage's newest run -- 0 for a stage never launched.  A rename
    keeps every byte, so nothing is kept beside it.  Decided, never done
    here: returns what it says (a line each) and the moves.

    **What it cannot fix is said too**: such a stage was prepared before
    launch gave each run its number, so its run script takes no ``--run`` --
    to launch it, its group is prepared again; launched, from the state
    saved before its group's first launch (`project-layout.md` § 1.6.1)."""
    if task.shape != "flat":
        return [], []
    from ..runfiles import (FIRST_ATTEMPT, RunNames, latest_run,
                            shared_numbered_roles)
    from .commands import way_back
    from .materialize import ladder_homes
    said: List[str] = []
    moves: List[Tuple[Path, Path]] = []
    for home in ladder_homes(base, task):
        names = RunNames.of(task.label, home.token, task.shape)
        newest = latest_run(base, task.label, stage=home.token)
        n = FIRST_ATTEMPT if newest is None else newest
        stage_moves = [(base / names.unnumbered(role),
                        base / names.name(role, n))
                       for role in shared_numbered_roles()
                       if (base / names.unnumbered(role)).is_file()]
        if not stage_moves:
            continue
        moves += stage_moves
        said += [f"{old.name} -> {new.name} (run {n}, the stage's newest)"
                 for old, new in stage_moves]
        # LAUNCHED, whether its launch record is numbered yet or not: one
        # written before the numbering is the unnumbered `run.json` moved
        # here.
        launched = (newest is not None
                    or (base / names.unnumbered(".run.json")).is_file())
        said.append(f"{home.name}: its run script was written before launch "
                    f"gave each run its number (--run N) -- "
                    + ("to launch it again, " if launched
                       else "to launch it, ")
                    + way_back(home.name, base=base, task=task,
                               launched=launched))
    return said, moves


def _migrate_last_run(base: Path, task, moves
                      ) -> Tuple[List[str], Optional[Tuple[Path, str]]]:
    """A flat calculation whose runs were launched before its folder recorded
    its last run (2.7.0): ``<label>.last-run`` written for the run its
    readers took as the last until then -- the newest launch record, by when
    it was written, then its stage's place in the ladder, then its run's
    number -- read where each record will lie once ``moves`` (the run
    numbering) are made.  Decided, never done here: returns what it says and
    the write; nothing for a folder that holds the record, or launched no
    run."""
    if task.shape != "flat":
        return [], None
    from ..identity import parse_token
    from ..persist import read_json
    from ..runfiles import compose, find_by_role, parse, run_name
    target = base / compose(task.label, ".last-run")
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
        token = parse_token(rec.stage)
        k = (str((body or {}).get("launched_at") or ""),
             token[0] if token else -1, rec.run)
        if key is None or k > key:
            best, key = run_name(task.label, rec.stage, rec.run), k
    if best is None:
        return [], None
    return ([f"{target.name}: {best} -- the last run here, by the rule that "
             f"held before 2.7.0 (its newest launch record)"],
            (target, best + "\n"))


def migrate_state(base) -> List[str]:
    """Rewrite the calculation in ``base``; return what changed, line by line
    -- its records (:func:`_migrate_records`: the readers refuse a file
    written before they were whole), its copy of its machine's record
    (:func:`_migrate_machine_record`), a flat calculation's run files numbered
    (:func:`_migrate_run_numbers`) and its last run recorded
    (:func:`_migrate_last_run`), its template's electronic-state
    items, its zeros, and its stages', that meant *not set* before 2.1.0,
    an item renamed and the items a release retired
    (:func:`_migrate_template`), all decided before any is written.  Each old file rewritten is
    kept beside its new one; a renamed one is its new one.

    Raises :class:`MigrateError` when there is nothing to migrate or it cannot
    be migrated as it stands.
    """
    base = Path(base)
    # EVERY DECISION BEFORE ANY WRITE: a refusal of any step leaves the
    # calculation as it was, never half rewritten.
    said, writes, task = _migrate_records(base)
    m_said, m_writes = _migrate_machine_record(base)
    said, writes = said + m_said, writes + m_writes
    n_said, moves = _migrate_run_numbers(base, task)
    l_said, l_write = _migrate_last_run(base, task, moves)
    try:
        t_said, t_writes, t_task, t_keep = _migrate_template(base, task)
    except _NothingInTheTemplate:
        if not said and not moves and l_write is None:
            raise MigrateError(
                f"{base} holds nothing an older molbuilder wrote -- its "
                f"records are whole, its record of its machine is this "
                f"molbuilder's, its run files carry their run's "
                f"number, and its template carries no item an older "
                f"molbuilder meant otherwise, renamed or a release "
                f"retired.") from None
        t_said, t_writes, t_task, t_keep = [], [], task, None
    if t_task is not task:
        # ONE WRITE OF task.json: the stages' old zeros and retired items
        # taken out, on top of whatever the records step gave it; the old
        # file kept once, as the earliest rule it was written before.
        from ..persist import json_text
        from ..task import FILENAME as TASK_FILENAME
        task_path, text = base / TASK_FILENAME, json_text(t_task.to_dict())
        if any(p == task_path for p, _x, _k in writes):
            writes = [(p, text if p == task_path else x, k)
                      for p, x, k in writes]
        else:
            t_writes.append((task_path, text, t_keep))
    taken = [new for _old, new in moves if new.exists()]
    if taken:
        raise MigrateError(
            f"{', '.join(p.name for p in taken)} already exist -- a run file "
            f"cannot be numbered over another; nothing was written.")
    # A KEPT FILE IS NEVER WRITTEN OVER: it is the only copy of what the
    # file was before an earlier migration.
    held = [path.with_name(path.name + keep)
            for path, _text, keep in writes + t_writes
            if path.with_name(path.name + keep).exists()]
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
    for path, text, keep in writes + t_writes:
        kept = path.with_name(path.name + keep)
        write_bytes(kept, path.read_bytes())
        write_bytes(path, text.encode("utf-8"))
    for old, new in moves:
        old.rename(new)
    if l_write is not None:
        write_bytes(l_write[0], l_write[1].encode("utf-8"))
    said += n_said + l_said + t_said
    from . import ledger
    ledger.record(base, "migrate", "a calculation an older molbuilder "
                  "wrote, rewritten -- each change with its reason",
                  changes=said)
    return said


class _NothingInTheTemplate(MigrateError):
    """The template carries no item written before the electronic state's
    and no item a release retired, and the calculation no 0 written before
    2.1.0 that meant *not set*."""


def _migrate_template(base: Path, task):
    """The template's charge and spin items, rewritten into the electronic
    state's -- this module's first migration (2026-09-28) -- its zeros, and
    its stages', that meant *not set* (2.1.0), an item renamed, kept under
    its new name, and the items a release retired, taken out of the template
    and of the description (`template.RETIRED_ITEMS`).  ``task`` is the
    description as the records step left it.  Nothing is written here:
    returns what changed, the template's write (``(path, text, kept-as
    suffix)``), the description -- its stages' old zeros and retired items
    taken out -- and the suffix ``task.json`` is kept as when they were."""
    import dataclasses

    from .. import template as _T
    from ..config.pyscf import PySCFConfig
    from ..config.siesta import SiestaConfig

    engine = task.engine
    kind = task.calculation
    try:
        tmpl = _T.find_template(base, task.label)
    except ValueError as exc:
        raise MigrateError(str(exc)) from exc
    if tmpl is None:
        raise _NothingInTheTemplate(f"{base} holds no template.")
    text = tmpl.read_text(encoding="utf-8")
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
    old_items = set(_OLD_NAMES) & set(vals)
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
    stage_zeros = ([(st.name, k) for st in (task.stages or ())
                    for k, v in (st.overrides or {}).items()
                    if k in _T.ZERO_WAS_NOT_SET and v == 0
                    and _T.reads(_T.one(_T.catalogue(), k, engine=engine),
                                 engine, kind, st.name)]
                   if _T.written_before_zero_changed(parsed) else [])
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
    known = _T.template_fields(PySCFConfig if engine == "pyscf"
                               else SiestaConfig)
    unknown = sorted(k for k in vals if k not in known
                     and k not in _OLD_NAMES and k not in _T.RETIRED_ITEMS)
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
              and (k in _OLD_NAMES or k in _T.STATE_ITEMS)]
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
    config_cls = PySCFConfig if engine == "pyscf" else SiestaConfig
    kept = {k: v for k, v in vals.items()
            if k in known and k not in _OLD_NAMES and k not in zeros}
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
    keep = (".pre-m6" if state_said
            else f".pre-{_T.ZERO_MEANING_CHANGED_AT}" if zeros
            else f".pre-{renamed_at[-1]}" if renamed and not retired_at
            else f".pre-{retired_at[-1]}")
    task_keep = (f".pre-{_T.ZERO_MEANING_CHANGED_AT}" if stage_zeros
                 else f".pre-{retired_at[-1]}" if retired_at
                 else f".pre-{renamed_at[-1]}"
                 if stage_renamed or varies_renamed else None)
    return (said + [f"the old template is kept as {tmpl.name}{keep}"],
            [(tmpl, new_text, keep)], task, task_keep)


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
