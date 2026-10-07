"""Generic dataclass-field-metadata-driven validation pass.

Reads ``range`` and ``validate`` off dataclass field metadata and
produces Issues.  This is what makes design.md Principle #1 load-
bearing: field metadata IS the source of truth; CLI / web /
validators all read from the same place.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import List

from ..issues import Issue


def _keyword_suffix(meta) -> str:
    """`` (MeshCutoff)`` — the engine's own spelling, beside the human name.

    **Both, always** (user, 2026-08-15: *"always include the keyword relevant
    to it in addition to the meaning of that, such that it is easy to detect,
    and meaning is still clear"*).  A warning has two jobs and one word cannot
    do them: *"Real-space grid cutoff"* says what is wrong and cannot be found
    in the input file; *"MeshCutoff"* can be searched for and says nothing.

    The BARE keyword, not the full ``engine_key``: a warning is scanned, and
    ``MD.MaxDispl (CG / Broyden / FIRE)`` is a phrase.  The
    full spelling belongs on the form's badge, where there is room for it.

    **Nothing is added for a setting that is not an engine keyword** — the
    molbuilder-side ones (``psml_lib``, ``copy_psml``, ``mpi_np``) whose
    ``engine_key`` is a parenthesised note.  There is no word to offer, and
    inventing one would make a search fail rather than merely not help.
    """
    from ..template import _bare_anchor
    kw = _bare_anchor(str(meta.get("engine_key", "") or ""))
    return f" ({kw})" if kw else ""


def outside_range(value, rng) -> list:
    """The components of ``value`` outside ``rng``, inclusive -- ``[(None,
    v)]`` for a number, ``[(i, v), ...]`` for a triple, whose range bounds
    each component (`engines/template.md` § 5).  ONE answer for the two
    places a range is warned: the settings gate's metadata pass and the
    description's own check (`validation/task.py`).  A component that is not
    a number -- a bool included -- is the type check's business, and is
    skipped here."""
    lo, hi = rng
    parts = (list(enumerate(value)) if isinstance(value, (list, tuple))
             else [(None, value)])
    return [(i, v) for i, v in parts
            if not isinstance(v, bool) and isinstance(v, (int, float))
            and (v < lo or v > hi)]


def _validate_config_metadata(cfg, refused=frozenset(),
                              foreign=frozenset()) -> List[Issue]:
    """The field metadata's own findings.  Two sets of fields have no range
    warning: ``refused``, the values the one per-value door refused
    (``template.why_not``) -- a value refused draws that refusal alone
    (`engines/template.md` § 5.3) -- and ``foreign``, the catalogue items
    this calculation does not carry: the kind narrows the catalogue (§ 6.3),
    so their values are not this calculation's to warn about
    (:func:`not_carried`)."""
    issues: List[Issue] = []
    if not is_dataclass(cfg):
        return issues
    for f in fields(cfg):
        meta = f.metadata or {}
        value = getattr(cfg, f.name)
        # range = (lo, hi) inclusive
        rng = meta.get("range")
        if (rng is not None and value is not None
                and f.name not in refused and f.name not in foreign):
            lo, hi = rng
            # A TRIPLE'S RANGE BOUNDS EACH COMPONENT (engines/template.md
            # § 5).  That is the only reading that means anything for
            # `kgrid` -- (1, 64) bounds each axis count, never their
            # product -- and it is what the form now puts on each of the
            # three inputs.
            label = meta.get("label", f.name)
            unit = f" {meta['unit']}" if meta.get("unit") else ""
            if isinstance(value, (tuple, list)):
                # EVERY TRIPLE, per component, through the one helper the
                # description's own check asks too (:func:`outside_range`).
                for i, v in outside_range(value, rng):
                    issues.append(Issue(
                        "warn",
                        f"{label}{_keyword_suffix(meta)}[{i}] = "
                        f"{v}{unit} is outside the recommended "
                        f"range [{lo}, {hi}]{unit}",
                        f"config.{f.name}",
                    ))
            else:
                try:
                    if value < lo or value > hi:
                        issues.append(Issue(
                            "warn",
                            f"{label}{_keyword_suffix(meta)} = {value}{unit} "
                            f"is outside the recommended range "
                            f"[{lo}, {hi}]{unit}",
                            f"config.{f.name}",
                        ))
                except TypeError:
                    # A SCALAR that will not compare -- a string where a
                    # number belongs, say.
                    issues.append(Issue(
                        "error",
                        (f"Field metadata for ``{f.name}`` declares "
                         f"``range = {rng}`` but the value "
                         f"{value!r} is not comparable with it.  "
                         f"This is a programmer bug: either the range or "
                         f"the field's type is wrong."),
                        f"config.{f.name}",
                    ))
        # AN ENUM'S VALUE IS ONE OF ITS MEMBERS, with the member's type
        # (`engines/template.md` § 5) -- the rule the template reader and the
        # form's coercion already hold, here for a config built any other
        # way.
        choices = meta.get("choices")
        if choices and value is not None:
            from ..template import is_member
            if not is_member(value, choices):
                issues.append(Issue(
                    "error",
                    f"{meta.get('label', f.name)}{_keyword_suffix(meta)} = "
                    f"{value!r} is not one of "
                    f"{', '.join(map(repr, choices))}",
                    f"config.{f.name}",
                ))
        # Optional callable: meta["validate"] -> Issue or None
        validator = meta.get("validate")
        if validator is not None:
            try:
                result = validator(value, cfg)
            except Exception as exc:
                # A validator-callable that raises (regex .match() on a
                # non-string, attribute-access on None, etc.) is surfaced
                # as an error-Issue, so the metadata bug is visible at
                # preflight time instead of the validator silently
                # disappearing.
                issues.append(Issue(
                    "error",
                    (f"Field metadata for ``{f.name}`` has a "
                     f"``validate`` callable that raised "
                     f"{type(exc).__name__}: {exc}.  This is a "
                     f"programmer bug: the callable should "
                     f"return Issue / list[Issue] / None instead "
                     f"of raising."),
                    f"config.{f.name}",
                ))
                continue
            if isinstance(result, Issue):
                issues.append(result)
            elif isinstance(result, list):
                issues.extend(i for i in result if isinstance(i, Issue))
    return issues


def not_carried(cfg, calculation: str) -> frozenset:
    """The catalogue items of ``cfg``'s engine that a ``calculation`` does
    not carry -- a lead's ``electrode_kz`` on an optimization -- whose field
    a config class holds all the same.  Empty for a config the catalogue
    does not describe."""
    from ..template import engine_name
    return not_carried_by(engine_name(type(cfg)), calculation)


def not_carried_by(engine: str, calculation: str) -> frozenset:
    """:func:`not_carried` by the engine's NAME -- what the description's
    own check has, before any config exists.  The kind narrows the
    catalogue (`engines/template.md` § 6.3), so neither door judges a value
    of an item the calculation does not carry."""
    from ..template import catalogue, select
    cat = catalogue()
    if engine not in cat.engines:
        return frozenset()
    carried = {it.name for it in select(cat, engine=engine,
                                        calculation=calculation)}
    return frozenset(it.name for it in select(cat, engine=engine)
                     if it.name not in carried)


def _check_fixed_on_every_rung(cfg, calculation: str) -> List[Issue]:
    """A config holding another value for what every rung fixes alike
    (`engines/template.md` § 6.4) -- SIESTA's per-step forces and
    coordinates -- is refused, naming the item and why.

    `resolve` lays these on every config it answers, so a config holding
    anything else was rendered without it -- a library call or a test --
    and its deck would print the note that says the value is fixed above
    a value nobody fixed.  The items answered rung by rung (a transport
    rung's solver, the leads' ``TS.HS.Save``, the bias point) have no one
    answer this check could hold a config to; `resolve` is their door.
    """
    from ..template import (catalogue, engine_name, fixed_on_every_rung,
                            why_role)
    engine = engine_name(type(cfg))
    if engine not in catalogue().engines:
        return []
    issues: List[Issue] = []
    for name, answer in fixed_on_every_rung(engine, calculation).items():
        have = getattr(cfg, name, answer)
        if have != answer:
            issues.append(Issue(
                "error",
                f"``{name}`` is {have!r}, and the rung fixes it at "
                f"{answer!r}: {why_role(name)}.  A deck is rendered from "
                f"the config `resolve` answered, which lays it on "
                f"(engines/template.md § 6.4).",
                f"config.{name}"))
    return issues


def _check_values(cfg, calculation: str) -> List[Issue]:
    """A value that cannot stand for its item on this kind is refused, with
    the ONE clause the per-value door gives (`engines/template.md` § 5.3,
    ``template.why_not``): a component the kind fixes, a choice it does not
    offer -- the electronic state's items excepted, whose RESOLVED values
    that family holds to the same sets (`check_electronic_state`) -- or a
    value at or below its hard limit.

    `resolve` refuses first, naming where the value came from; this holds a
    render that skipped `resolve` -- a library call or a test.  One pass, so
    one value draws one refusal.
    """
    from ..template import catalogue, engine_name, select, why_not
    engine = engine_name(type(cfg))
    cat = catalogue()
    if engine not in cat.engines:
        return []
    issues: List[Issue] = []
    for it in select(cat, engine=engine, calculation=calculation):
        have = getattr(cfg, it.name, None)
        clause = why_not(it, have, engine=engine, kind=calculation)
        if clause:
            shown = list(have) if isinstance(have, tuple) else have
            issues.append(Issue("error", f"{it.name} = {shown!r}{clause}.",
                                f"config.{it.name}"))
    return issues
