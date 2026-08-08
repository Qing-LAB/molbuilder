"""The template — an engine config as a readable, machine-readable file.

**Module:** L2 (model). Imports ``config.*`` and ``script_emit``; imported by
``siesta/*`` producers and, later, by ``jobset/prep``. Nothing here touches the
filesystem or a scheduler.

**Contract:** [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
§ 3.7 (the item-block format) · [`engines/stages.md`](?doc=engines/stages.md)
§ 4 (the template is the science backbone; effective config = its values ⊕ a
stage's ``overrides``).

The template is *"everything a script owns, with values"*: every field the
engine's config declares, each wrapped in

    # === molbuilder item <field> BEGIN ===
    #   field <name> anchor=… type=… range=[a,b] unit=… default=… group=…
    #   <what we know about this item, in prose>
    <the payload, exactly as it lands in the deck>
    # === molbuilder item <field> END ===

so that **one artifact serves four readers**: a person learns the calculation
and the reasoning, the UI renders it, ``prep`` extracts the deck, and the
validator gets a real config out of it.

**What this module does today, and what it deliberately does not.** It derives
the *declarations* from a config class and computes the *schema fingerprint*.
Emitting a whole template and reading one back are the rest of P2 unit 4a, and
two questions in § 3.7 have to be answered first — both recorded in the plan,
both found by measuring the shipped schema rather than by reading the contract
again:

  * ``engine_key`` is **not** an anchor for every field. Four carry prose or an
    alternation (``relax_steps`` → *"MD.NumCGsteps (universal…) |
    MD.FinalTimeStep (Verlet / Nose)"``; ``spin_total`` → *"Spin.Fix +
    Spin.Total"*). § 3.7 assumes one field ↔ one anchor.
  * Ten exposed fields are **conditionally emitted** — at their defaults the
    deck has no line at all — so § 3.7's *"the payload is exactly what lands in
    the deck"* and *"every item has a place in the file"* pull opposite ways.

Both are stated so the next commit answers them on purpose.
"""
from __future__ import annotations

import dataclasses
import hashlib
import typing
from typing import Any, List, Optional, Tuple

from .script_emit import BenchField


# --------------------------------------------------------------------- #
#  A config field's declaration                                         #
# --------------------------------------------------------------------- #

#: How a Python annotation becomes a declaration ``type`` (§ 3.3's vocabulary,
#: extended by § 3.7).  A field with ``choices`` is an ``enum`` whatever its
#: Python type, because that is what a reader needs in order to validate it.
_ANNOTATION_TYPES = {
    bool:  "bool",     # checked BEFORE int -- bool IS an int in Python, and
    int:   "int",      # getting that order wrong types every checkbox as int
    float: "float",
    str:   "str",
}


def _unwrap_optional(ann) -> Tuple[Any, bool]:
    """``Optional[X]`` → ``(X, True)``; anything else → ``(ann, False)``."""
    if typing.get_origin(ann) is typing.Union:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


def _decl_type(ann, choices) -> Optional[str]:
    """The declaration ``type`` for one annotation, or None if unnameable."""
    if choices:
        return "enum"
    if ann in _ANNOTATION_TYPES:
        return _ANNOTATION_TYPES[ann]
    # Tuple[int, int, int] -- the k-grid.  Named rather than flattened to
    # three fields: it is one decision ("how finely is reciprocal space
    # sampled"), and a stage overriding it overrides all three together.
    if typing.get_origin(ann) is tuple:
        args = typing.get_args(ann)
        if args and all(a is int for a in args):
            return "int3"
    return None


def declaration_for(f: "dataclasses.Field", annotation) -> Optional[BenchField]:
    """The § 3.7 declaration for one config field, or ``None`` if it has no
    place in a template.

    ``None`` means **not exposed** — a field with no ``section`` is internal
    (`web/form-schema.md § 1a`), so no surface renders it and a template that
    listed it would be offering the user something no tab can show.

    Raises ``ValueError`` for an exposed field whose type has no name in the
    grammar: that is a gap in the vocabulary, and the loud version of it is
    the only one that gets fixed.
    """
    if not f.metadata.get("section"):
        return None

    ann, optional = _unwrap_optional(annotation)

    # A ``List[<dataclass>]`` is a STAGE LADDER, and a ladder is not a template
    # item -- it is the user's decision about what varies, and it lives in
    # ``task.json`` (`engines/stages.md § 1.1`).  Excluded for what it is,
    # with a reason, rather than left to fall through to the type error below,
    # which would report a vocabulary gap where there is none.
    #
    # ``PySCFConfig.stages`` is the only one left: SIESTA's was deleted in P2
    # unit 2, and PySCF keeps its own because that ladder runs inside a single
    # process.  When PySCF's path is reworked this branch goes with it.
    _args = typing.get_args(ann)
    if (typing.get_origin(ann) in (list, tuple)
            and _args and dataclasses.is_dataclass(_args[0])):
        return None

    choices = f.metadata.get("choices")
    type_ = _decl_type(ann, choices)
    if type_ is None:
        raise ValueError(
            f"field {f.name!r}: no declaration type for annotation {ann!r}. "
            f"Add one to DECL_TYPES (job-contracts.md 3.3) rather than "
            f"leaving the field out of the template -- § 3.7's premise is "
            f"that every allowed item has a place in the file.")

    rng = f.metadata.get("range")
    return BenchField(
        name=f.name,
        anchor=f.metadata.get("engine_key", "") or f.name,
        type_=type_,
        range_=(tuple(rng) if rng else None),
        unit=f.metadata.get("unit"),
        group=f.metadata.get("workflow_group"),
        choices=(tuple(choices) if choices else None),
        optional=optional,
    )


def declarations_for(config_cls) -> List[BenchField]:
    """Every exposed field of *config_cls*, in declaration order.

    Declaration order, not alphabetical: the config's field order is the
    form's order and the deck's order, and a template a person reads should
    not be a third arrangement of the same things.
    """
    hints = typing.get_type_hints(config_cls)
    out: List[BenchField] = []
    for f in dataclasses.fields(config_cls):
        decl = declaration_for(f, hints[f.name])
        if decl is not None:
            out.append(decl)
    return out


# --------------------------------------------------------------------- #
#  The schema fingerprint                                               #
# --------------------------------------------------------------------- #

#: Bumped when the fingerprint's *recipe* changes, so an old fingerprint is
#: recognisably old rather than merely different.
FINGERPRINT_VERSION = "1"


def schema_fingerprint(config_cls) -> str:
    """A short, stable digest of the shape a description was written against.

    ``task.json`` carries one (`engines/stages.md § 6.6`), and the preflight's
    only **non-refusal** row is *"the schema fingerprint matches"* — a
    description written when ``mesh_cutoff`` was bounded [50, 2000] and read
    after it became [50, 800] names a field that still exists and a value that
    is no longer legal, which is worth saying out loud rather than discovering
    at the engine.

    **This is the writer that row has been missing since it was written.**
    Unit 4a's rule: it is computed by whatever writes the template, because
    that is the moment the schema is in hand.

    What goes in is what a *description* can depend on: the field's name, its
    declaration type, its bounds, its enum members, and whether unset is a
    state it has. Deliberately NOT included:

      * **the default** — a template records the value in use, so changing a
        default cannot invalidate a description that already carries values;
      * **help text, labels, units, `workflow_group`** — presentation. A
        reworded tooltip must not make every stored description suspect, and
        a fingerprint that cried wolf would be turned off.

    So it changes when a field is added, removed, retyped, re-bounded, or has
    its choices changed — and not otherwise.
    """
    parts: List[str] = [f"v{FINGERPRINT_VERSION}"]
    for d in sorted(declarations_for(config_cls), key=lambda d: d.name):
        rng = f"{d.range_[0]},{d.range_[1]}" if d.range_ else ""
        choices = "|".join(d.choices) if d.choices else ""
        parts.append(f"{d.name}:{d.type_}:{rng}:{choices}:{int(d.optional)}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def fingerprint_matches(config_cls, recorded: str) -> bool:
    """Whether *recorded* is this schema's fingerprint.

    An **empty** recorded fingerprint matches anything: a description written
    before this existed, or by hand, is not wrong — it simply makes no claim,
    and § 6.6 lists this as the one row that does not refuse.
    """
    if not recorded:
        return True
    return recorded == schema_fingerprint(config_cls)


__all__ = ["declaration_for", "declarations_for", "schema_fingerprint",
           "fingerprint_matches", "FINGERPRINT_VERSION"]
