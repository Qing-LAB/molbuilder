"""Rewrite a calculation written before the electronic state's items.

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
    """The template cannot be migrated as it stands -- the message says why
    and what to do."""


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
#: file's own statement and is kept.  (All five stood here until the M6
#: review, so a half-migrated file lost what it already said: a stated
#: ``HF`` became DFT, a new count was overwritten.)
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


def migrate_state(base) -> List[str]:
    """Rewrite the calculation in ``base``; return what changed, line by line.

    Raises :class:`MigrateError` when there is nothing to migrate or it cannot
    be migrated as it stands.
    """
    from .. import template as _T
    from ..config.pyscf import PySCFConfig
    from ..config.siesta import SiestaConfig
    from ..task import FILENAME as TASK_FILENAME
    from ..task import read_task

    base = Path(base)
    task = read_task(base / TASK_FILENAME)
    engine = task.engine
    kind = task.calculation or "optimization"
    tmpl = _T.find_template(base)
    if tmpl is None:
        raise MigrateError(f"{base} holds no template to migrate.")
    text = tmpl.read_text(encoding="utf-8")
    parsed = _T.read_template(text)       # read against its OWN declarations
    mine = (_T.select(parsed, engine=engine) if parsed.engines
            else parsed.items)
    vals = {it.name: it.value for it in mine if it.is_set}
    # A RENAMED ITEM KEEPS ITS VALUE, under its new name and said as such --
    # the table the template reader refuses the old name by
    # (`template.RENAMED_ITEMS`).  Filtering on today's schema alone dropped
    # it and wrote the default in its place, which is the opposite of this
    # module's promise: what the run was is kept.
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
        raise MigrateError(
            f"{tmpl.name} carries no item written before the electronic "
            f"state's -- there is nothing to migrate.")

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
    # Where each value came from carries over (§ 6.6 obligation 2).
    new_text = _T.template_with_values(cfg, engine=engine, calculation=kind,
                                       title=_leading_comment(text),
                                       valueless=valueless,
                                       sources={it.name: it.source
                                                for it in mine if it.source})
    # The new file must be one prep accepts -- checked before anything is
    # written, so a failure leaves the calculation as it was, and is said
    # as the migration's refusal rather than a traceback.
    try:
        _T.config_from_template(new_text, config_cls)
    except ValueError as exc:
        raise MigrateError(
            f"{tmpl.name} rewritten would still be refused, so nothing was "
            f"written: {exc}") from exc
    keep = tmpl.with_name(tmpl.name + ".pre-m6")
    keep.write_text(text, encoding="utf-8")
    tmpl.write_text(new_text, encoding="utf-8")
    from . import ledger
    ledger.record(base, "migrate",
                  "the template's charge and spin items rewritten into the "
                  "electronic state's (science/chemistry-correctness.md § 2a)",
                  changes=said, kept_as=keep.name)
    return said + [f"the old template is kept as {keep.name}"]


def _leading_comment(text: str) -> str:
    """The file's header comment, without its ``#`` -- what ``init`` wrote as
    the title, carried over unchanged."""
    lines = []
    for ln in text.splitlines():
        if not ln.startswith("#"):
            break
        lines.append(ln[1:].lstrip() if ln.startswith("# ") else ln[1:])
    return "\n".join(lines)
