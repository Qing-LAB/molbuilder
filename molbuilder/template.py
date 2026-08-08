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

**The read is from ``value=``, never from the payload** (§ 3.7 property 2,
decided 2026-08-07). A payload can be absent — ten of SIESTA's exposed fields
write no deck line at their defaults — several lines (``spin_total`` writes
``Spin.Fix`` *and* ``Spin.Total``), or a ``%block``. None of that changes how
the value is read, because the value is on the declaration line.

**And the deck is re-rendered, not spliced.** ``prep`` rebuilds a config, applies
the stage's ``overrides``, and goes through the same emitter every other deck
goes through. § 3.7 property 1's guarantee — *a value cannot change shape between
what a person read and what the engine got* — survives as a **checked** property:
``tests/test_template_roundtrip.py`` asserts the rendered line is byte-identical
to the template's payload for every item no stage overrode. Substituting in place
could not work at all, because a stage overriding ``relax_type`` moves the step
budget's site from ``MD.NumCGsteps`` to ``MD.FinalTimeStep`` — the anchor itself
is chosen by another field's value.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
import typing
from typing import Any, Dict, List, Optional, Tuple

from .script_emit import BLOCK_ITEM, BenchField, emit_item_block


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





# --------------------------------------------------------------------- #
#  Writing a template                                                   #
# --------------------------------------------------------------------- #

#: A ``%block`` payload runs to its ``%endblock``.
_BLOCK_OPEN = "%block"


def _anchor_token(engine_key: str) -> Optional[str]:
    """The literal token a deck line starts with, or ``None``.

    ``None`` for the three shapes ``engine_key`` actually carries besides a
    plain keyword, measured against ``SiestaConfig`` on 2026-08-07:

      * a **parenthesised note** — the field reaches no deck line at all
        (``mpi_np``, ``psml_lib``, ``restart``, ``continue_retries``, …);
      * an **alternation**, ``A | B`` — which site is used depends on another
        field's value (``relax_steps``);
      * a **conjunction**, ``A + B`` — one field writes two lines
        (``spin_total``).

    Only the payload is affected: the value rides on the declaration, so a
    ``None`` here costs a block its illustrative body and nothing else. This
    is exactly why § 3.7 stopped making the anchor load-bearing.
    """
    ek = (engine_key or "").strip()
    if not ek or ek.startswith("(") or "|" in ek or "+" in ek:
        return None
    return ek


def _payload_for(anchor: Optional[str], deck_lines: List[str]) -> str:
    """The deck lines this anchor owns, verbatim, or ``""``."""
    if not anchor:
        return ""
    if anchor.startswith(_BLOCK_OPEN):
        name = anchor.split(None, 1)[-1]
        out, inside = [], False
        for line in deck_lines:
            low = line.strip().lower()
            if low.startswith(f"{_BLOCK_OPEN} {name}".lower()):
                inside = True
            if inside:
                out.append(line)
                if low.startswith("%endblock"):
                    break
        return "\n".join(out)
    pat = re.compile(rf"^\s*{re.escape(anchor)}\b")
    return "\n".join(l for l in deck_lines if pat.match(l))


def render_template(deck_text: str, config, *, config_cls=None) -> str:
    """The template for *config*, given the deck it renders to.

    ``deck_text`` is what the ordinary emitter produced for this config — the
    payloads are lifted **out of it** rather than re-implemented, so
    ``payload == what lands in the deck`` is true by construction for every
    item with a usable anchor, and the round-trip guard covers the rest.
    """
    cls = config_cls or type(config)
    deck_lines = deck_text.splitlines()
    defaults = {f.name: (f.default if f.default is not dataclasses.MISSING
                         else None)
                for f in dataclasses.fields(cls)}
    out: List[str] = []
    for d in declarations_for(cls):
        out.append(emit_item_block(
            d,
            value=getattr(config, d.name),
            default=defaults.get(d.name),
            help_text=_help_of(cls, d.name),
            payload=_payload_for(_anchor_token(d.anchor), deck_lines),
        ))
    return "\n\n".join(out) + "\n"


def _help_of(cls, name: str) -> str:
    for f in dataclasses.fields(cls):
        if f.name == name:
            return str(f.metadata.get("help", "") or "")
    return ""


# --------------------------------------------------------------------- #
#  Reading one back                                                     #
# --------------------------------------------------------------------- #

_DECL_RE = re.compile(r"^#\s*field\s+(\S+)\s+(.*)$")


def _coerce(raw: str, type_: str):
    if type_ == "bool":
        return raw.lower() == "true"
    if type_ == "int":
        return int(raw)
    if type_ == "float":
        return float(raw)
    if type_ == "int3":
        return tuple(int(x) for x in raw.split(","))
    return raw                      # str, enum


def read_template(text: str) -> Dict[str, Any]:
    """``{field name: value}`` from a template's item blocks.

    A **scan**, not a parse: the markers name the field and the declaration
    carries the value, so nothing here understands ``.fdf`` syntax — which is
    § 3.7 property 2's whole point, and just as well, since nothing in
    molbuilder can read an ``.fdf`` back into a config.

    A block whose declaration has no ``value=`` yields ``None``: that is an
    optional field left unset, which is a real state and not the default.
    """
    from .script_emit import MARKER_RE

    values: Dict[str, Any] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = MARKER_RE.match(line)
        if m:
            name = m.group(1)
            if not name.startswith(f"{BLOCK_ITEM} "):
                current = None
                continue
            current = (name.split(None, 1)[1] if m.group(2) == "BEGIN"
                       else None)
            continue
        if current is None:
            continue
        d = _DECL_RE.match(line.strip())
        if not d or d.group(1) != current:
            continue
        kv = dict(tok.split("=", 1) for tok in d.group(2).split()
                  if "=" in tok)
        raw = kv.get("value")
        values[current] = (None if raw is None
                           else _coerce(raw, kv.get("type", "str")))
    return values


def config_from_template(text: str, config_cls):
    """An ordinary instance of *config_cls*, rebuilt from a template.

    What ``prep`` holds before it applies a stage's ``overrides``
    (`engines/stages.md § 4`). A field the template does not carry keeps the
    class default — a template written against an older schema is missing
    fields, not wrong about them, and the fingerprint is what says so.
    """
    known = {f.name for f in dataclasses.fields(config_cls)}
    vals = {k: v for k, v in read_template(text).items() if k in known}
    return config_cls(**vals)


__all__ = ["declaration_for", "declarations_for", "schema_fingerprint",
           "fingerprint_matches", "FINGERPRINT_VERSION",
           "render_template", "read_template", "config_from_template"]
