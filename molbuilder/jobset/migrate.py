"""Rewrite a calculation an older molbuilder wrote: its records written whole
(2026-10-06), a flat calculation's run files numbered (2026-10-06), and its
template's electronic-state items (2026-09-28).

**The run files** (`_migrate_run_numbers`): in the flat shape each run's
files -- every row the catalogue numbers where a stage's runs share a
folder (`runfiles.shared_numbered_roles`) -- carry the run's
number since 2026-10-06 (plan W57 decisions 2 and 6), and the launch
record's door refuses a stage whose still do not, naming this command.  Each
is renamed for its stage's newest run, and said.

**The records** (`_migrate_records`): `task.json` states its `calculation`
and a notify block its four values, and every job in a `job-set.json` its
`point`, `finish`, `resumes` and `placement` since 2026-10-06, and its
`group` since 2026-10-07 -- the readers refuse a file that does not, naming
this command (plan W57, decision 7).  Each key an older file left out is
written with the meaning its absence had -- `optimization`, every channel and
every field (`["*"]`), off, an empty point, no finish, resumes, no placement,
prepped alone -- and each is printed; the old file
is kept beside the new as ``<name>.pre-w57``.

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
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


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
               "placement": None, "group": None}


def _never_as_zero(hours) -> bool:
    """Whether a period is the older spelling of never, ``0``."""
    return (isinstance(hours, (int, float)) and not isinstance(hours, bool)
            and hours == 0)


def _migrate_records(base: Path):
    """`task.json` and every `job-set.json` of the calculation, each key an
    older file left out given the meaning its absence had -- read raw, since
    the readers refuse such a file, and each rewritten file checked by its
    reader.  Nothing is written here: returns what changed (a line each), the
    files to write (``(path, text)``), and the description as rewritten."""
    from ..persist import json_text, read_json
    from ..task import FILENAME as TASK_FILENAME
    from ..task import Task
    from .materialize import sweep_set_paths
    from .model import FILENAME as JOBSET_FILENAME, JobSet
    said: List[str] = []
    writes: List[Tuple[Path, str]] = []

    task_path = base / TASK_FILENAME
    obj = read_json(task_path)
    lines = []
    for key, meant in _TASK_ABSENT.items():
        if key not in obj:
            obj[key] = meant
            lines.append(f"{key} = {meant!r} (it was left out)")
    notify = obj.get("notify")
    if isinstance(notify, dict) and notify:
        for key, meant in _NOTIFY_ABSENT.items():
            if key not in notify:
                notify[key] = meant
                lines.append(f"notify.{key} = {meant!r} (it was left out)")
        if _never_as_zero(notify.get("every_hours")):
            notify["every_hours"] = None
            lines.append("notify.every_hours = None (0 was never)")
    try:
        task = Task.from_dict(obj)
    except ValueError as exc:
        raise MigrateError(f"{task_path.name} rewritten would still be "
                           f"refused, so nothing was written: {exc}")
    if lines:
        writes.append((task_path, json_text(task.to_dict())))
        said += [f"{TASK_FILENAME}: {ln}" for ln in lines]

    for js_path in [base / JOBSET_FILENAME, *sweep_set_paths(base)]:
        if not js_path.is_file():
            continue
        d = read_json(js_path)
        filled = 0
        for job in d.get("jobs") or []:
            gaps = {k: v for k, v in _JOB_ABSENT.items() if k not in job}
            if gaps:
                job.update(gaps)
                filled += 1
        if filled:
            try:
                body = JobSet.from_dict(d).to_dict()
            except ValueError as exc:
                raise MigrateError(f"{js_path} rewritten would still be "
                                   f"refused, so nothing was written: {exc}")
            writes.append((js_path, json_text(body)))
            said.append(f"{js_path.relative_to(base)}: {filled} job(s) given "
                        f"every key -- {', '.join(_JOB_ABSENT)} "
                        f"(those left out, as their absence meant)")
    return said, writes, task


def _migrate_run_numbers(base: Path, task) -> Tuple[List[str],
                                                     List[Tuple[Path, Path]]]:
    """A flat calculation's run files written before each carried its run's
    number (plan W57 decisions 2 and 6): every file of a stage whose role
    the catalogue numbers where a stage's runs share a folder
    (`runfiles.shared_numbered_roles`), still unnumbered, given the number
    of its stage's newest run -- 0 for a stage never launched.  A rename
    keeps every byte, so nothing is kept beside it.  Decided, never done
    here: returns what it says (a line each) and the moves.

    **What it cannot fix is said too**: such a stage was prepped before
    launch gave each run its number, so its run script takes no ``--run`` --
    to launch it again, it is prepped anew, from the state saved before its
    prep (`project-layout.md` § 1.6.1)."""
    if task.shape != "flat":
        return [], []
    from ..runfiles import (FIRST_ATTEMPT, RunNames, latest_run,
                            shared_numbered_roles)
    from .commands import rollback
    from .materialize import ladder_homes
    said: List[str] = []
    moves: List[Tuple[Path, Path]] = []
    for home in ladder_homes(base, task):
        names = RunNames.of(task.label, home.token, task.shape)
        newest = latest_run(base, task.label, stage=home.token)
        n = FIRST_ATTEMPT if newest is None else newest
        stage_moves = [(base / (names.stem + role),
                        base / names.name(role, n))
                       for role in shared_numbered_roles()
                       if (base / (names.stem + role)).is_file()]
        if not stage_moves:
            continue
        moves += stage_moves
        said += [f"{old.name} -> {new.name} (run {n}, the stage's newest)"
                 for old, new in stage_moves]
        said.append(f"{home.name}: its run script was written before launch "
                    f"gave each run its number (--run N) -- to launch it "
                    f"again, prep it anew: " + rollback("its prep",
                                                        base=base))
    return said, moves


def migrate_state(base) -> List[str]:
    """Rewrite the calculation in ``base``; return what changed, line by line
    -- its records (:func:`_migrate_records`: the readers refuse a file
    written before they were whole), a flat calculation's run files numbered
    (:func:`_migrate_run_numbers`) and its template's electronic-state
    items, all decided before any is written.  Each old file rewritten is
    kept beside its new one; a renamed one is its new one.

    Raises :class:`MigrateError` when there is nothing to migrate or it cannot
    be migrated as it stands.
    """
    base = Path(base)
    # EVERY DECISION BEFORE ANY WRITE: a refusal of any step leaves the
    # calculation as it was, never half rewritten.
    said, writes, task = _migrate_records(base)
    n_said, moves = _migrate_run_numbers(base, task)
    try:
        t_said, t_writes = _migrate_template(base, task)
    except _NothingInTheTemplate:
        if not said and not moves:
            raise MigrateError(
                f"{base} holds nothing an older molbuilder wrote -- its "
                f"records are whole, its run files carry their run's "
                f"number, and its template carries no item written before "
                f"the electronic state's.") from None
        t_said, t_writes = [], []
    taken = [new for _old, new in moves if new.exists()]
    if taken:
        raise MigrateError(
            f"{', '.join(p.name for p in taken)} already exist -- a run file "
            f"cannot be numbered over another; nothing was written.")
    for path, text, keep in ([(p, x, ".pre-w57") for p, x in writes]
                             + t_writes):
        kept = path.with_name(path.name + keep)
        kept.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(text, encoding="utf-8")
    for old, new in moves:
        old.rename(new)
    said += n_said + t_said
    from . import ledger
    ledger.record(base, "migrate", "a calculation an older molbuilder "
                  "wrote, rewritten (plan W57 decisions 2, 6 and 7; "
                  "science/chemistry-correctness.md § 2a)", changes=said)
    return said


class _NothingInTheTemplate(MigrateError):
    """The template carries no item written before the electronic state's."""


def _migrate_template(base: Path, task):
    """The template's charge and spin items, rewritten into the electronic
    state's -- this module's first migration (2026-09-28).  ``task`` is the
    description as the records step left it.  Nothing is written here:
    returns what changed and the write (``(path, text, kept-as suffix)``)."""
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
        raise MigrateError(f"{base} holds no template to migrate.")
    text = tmpl.read_text(encoding="utf-8")
    parsed = _T.read_template(text)       # read against its OWN declarations
    mine = (_T.select(parsed, engine=engine) if parsed.engines
            else parsed.items)
    vals = {it.name: it.value for it in mine if it.is_set}
    # A RENAMED ITEM KEEPS ITS VALUE, under its new name and said as such --
    # the table the template reader refuses the old name by
    # (`template.RENAMED_ITEMS`).  Filtering on today's schema alone would
    # drop it and write the default in its place: what the run was is kept.
    renamed: List[str] = []
    for old_name, (new_name, _why) in _T.RENAMED_ITEMS.items():
        if old_name in vals and new_name not in vals:
            vals[new_name] = vals.pop(old_name)
            renamed.append(f"{old_name} = {vals[new_name]!r} -> {new_name} "
                           f"(renamed)")
    old_items = set(_OLD_NAMES) & set(vals)
    old_values = {k for k in ("method", "spin_treatment")
                  if k in vals and vals[k] in (_PYSCF_CLASS if k == "method"
                                               else _SIESTA_TREATMENT)}
    if not old_items and not old_values:
        raise _NothingInTheTemplate(
            f"{tmpl.name} carries no item written before the electronic "
            f"state's.")

    staged = [(st.name, k) for st in (task.stages or ())
              for k in (st.overrides or {}) if k in _OLD_NAMES
              or k in _T.STATE_ITEMS]
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

    state, said = _map_state(engine, vals)
    said = renamed + said
    config_cls = PySCFConfig if engine == "pyscf" else SiestaConfig
    known = _T.template_fields(config_cls)
    kept = {k: v for k, v in vals.items()
            if k in known and k not in _OLD_NAMES}
    kept.update(state)
    try:
        cfg = config_cls(**kept)
    except (TypeError, ValueError) as exc:
        raise MigrateError(f"{tmpl.name} cannot be rewritten: {exc}") from exc
    # Unset stays unset: an item the old file left valueless is written
    # valueless again, so a row nobody answered is not given a default.
    valueless = [it.name for it in mine
                 if not it.is_set and it.name in known
                 and it.name not in _T.STATE_ITEMS]
    # WHERE EACH VALUE CAME FROM, as the old file says it (§ 6.6 obligation
    # 2): a file written before sources were recorded says nothing, so every
    # value carried out of it -- renamed or mapped into the electronic state
    # included -- is *not recorded*, never *not chosen*.  An item
    # the old file never had is nobody's choice.
    recorded = {it.name: it.source for it in mine if it.source}
    new_text = _T.template_with_values(cfg, engine=engine, calculation=kind,
                                       title=_leading_comment(text),
                                       valueless=valueless,
                                       sources={k: recorded.get(k)
                                                for k in kept})
    # The new file must be one prep accepts -- checked before anything is
    # written, so a failure leaves the calculation as it was, and is said
    # as the migration's refusal rather than a traceback.
    try:
        _T.config_from_template(new_text, config_cls)
    except ValueError as exc:
        raise MigrateError(
            f"{tmpl.name} rewritten would still be refused, so nothing was "
            f"written: {exc}") from exc
    return (said + [f"the old template is kept as {tmpl.name}.pre-m6"],
            [(tmpl, new_text, ".pre-m6")])


def _leading_comment(text: str) -> str:
    """The file's header comment, without its ``#`` -- what ``init`` wrote as
    the title, carried over unchanged."""
    lines = []
    for ln in text.splitlines():
        if not ln.startswith("#"):
            break
        lines.append(ln[1:].lstrip() if ln.startswith("# ") else ln[1:])
    return "\n".join(lines)
