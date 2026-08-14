"""The template — a calculation's parameter catalogue, as one TOML file.

**Module:** L2 (model). Imports ``persist`` and the standard library; imported by
``siesta/*`` producers, by ``validation/task``, and by
``resolve/``. Nothing here touches the filesystem or a scheduler.

**Contract:** [`engines/template.md`](?doc=engines/template.md) — § 3 the required
keys, § 4 the TOML format, § 5 the anatomy of an item, § 6 the ``kind``
vocabulary and ``read_by`` · [`execution/generator.md`](?doc=execution/generator.md)
§ 3.1 — why the UI is a reader and which keys serve it.

WHAT IT IS.  Every parameter the engine's schema declares, each carrying its
value, what it is validated as, who owns it, and what we know about it.  **One
artifact, four readers** (`template.md` § 5): a person learns the calculation, a
**surface renders it**, ``prep`` rebuilds a config and renders decks, and
validation gets a real object out of it.

THE VALUE IS ON THE ITEM, ONCE.  A template stores each value exactly once,
which is the whole argument for TOML over the engine's own format (§ 4.1): a
file that parses cleanly and describes a *different* calculation than it appears
to is the failure mode worth designing out, and storing a value twice is how you
get one.

**An absent ``value`` means explicitly unset** (§ 3) — a real state, distinct
from the default.  TOML has no null, so absence is the only encoding, and it is
unambiguous.

WHAT CHANGED ON 2026-08-11, and why the old shape could not stay.  This module
emitted the retired item-block format — ``# === molbuilder item <field> BEGIN
===`` wrapping a copy of the deck's own lines — and it built that by taking a
**rendered deck** and lifting payloads out of it with a regex.  Two things were
wrong with it and both are structural:

  * **the direction was inverted.**  The contract is schema → template → deck.
    Deriving the template *from* a deck made the deck the source and the
    catalogue a projection of it, so ``prep`` could not render from the template
    without already having what it was about to render.
  * **the payload was a second copy of every value**, which is exactly the
    self-disagreement § 4.1 rejects.

TOML has no payload key, so ``render_template`` needs no deck and the whole
lifting apparatus — ``_anchor_token``, ``_payload_for``, ``_DECL_RE``,
``_coerce`` — is gone rather than ported.

THE WRITER CHECKS ITSELF.  ``tomllib`` reads TOML and does not write it, so the
emitter here is hand-rolled — and § 4.1 requires that whatever emits a template
**read its own output back and compare it to what it meant to write**.
:func:`render_template` builds the payload as a plain object, serialises it,
parses the result, and refuses if the two differ.  That turns *"we emitted TOML
correctly"* from an assumption into a property checked on every call.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
import tomllib
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Tuple

from .persist import check_schema


SCHEMA = "molbuilder/template@2"

#: The suffix of a template's filename; the stem is the ``SystemLabel``.
#: ``<label>.template.toml`` (`job-contracts.md` § 6.3).  It was
#: ``<label>.fdf.template`` until 2026-08-11 — an engine-flavoured name for a
#: file that is not an ``.fdf`` and never was.
SUFFIX = ".template.toml"

#: § 6 — closed. An unknown ``kind`` is refused, never skipped, because a reader
#: that quietly dropped an item it did not understand would emit a deck missing
#: a parameter and say nothing.
KINDS = ("engine", "deck", "wrapper", "produce", "monitor")

#: The closed CATEGORY vocabulary (`engines/template.md` § 6.2), in READING
#: ORDER -- a surface presents panels top to bottom in this order, because it
#: is the order a person decides in and a methods section is written in.
#:
#: A category has NO effect on the generated script: the deck writer filters on
#: ``kind``.  It is a presentation-and-discovery key, which is why an item may
#: carry SEVERAL -- the first is the panel it appears on, the rest make it
#: findable where a user would look.
#:
#: ``execution`` is the exception and is a CLAIM rather than a hint: a benchmark
#: takes it as the sweepable set -- knobs that change speed and not the answer.
#: Tagging something ``execution`` that changes the answer means a sweep
#: silently measures a different calculation at each point.
CATEGORIES = ("system", "method", "accuracy", "convergence",
              "procedure", "execution")

#: § 5 — the validation vocabulary. TOML types the *storage*; this types what a
#: reader must check, which a parser cannot know: that ``pow2`` is a power of
#: two, that ``enum`` is drawn from ``choices``, that ``text`` is verbatim engine
#: text to be copied rather than interpreted.
TYPES = ("int", "float", "str", "bool", "enum", "pow2",
         "int3", "strlist", "intlist", "text")


def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(f"template{': ' + where if where else ''}: {msg}")


# --------------------------------------------------------------------- #
#  One item                                                             #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Item:
    """One parameter, with everything all four readers need from it.

    The four required keys (§ 3) are the ones without defaults here:
    :attr:`name`, :attr:`kind`, :attr:`type`, :attr:`help`. Everything else is
    present when it applies and absent when it does not — *"absent is not a
    failure; it is the honest statement that the parameter has no default, no
    bounds, no unit, or no other reader."*

    ``value`` is ``None`` for **explicitly unset**, which is the state § 3
    distinguishes from the default. TOML cannot express null, so a written
    template simply omits the key.
    """
    name: str
    kind: str
    type: str
    help: str

    value:   Any = None
    default: Any = None

    # --- reaching the deck (§ 6) ---
    anchor:  str = ""                       # required when kind == "engine"
    expands: Tuple[str, ...] = ()           # required when kind == "deck"
    read_by: Tuple[str, ...] = ()           # § 6.1 — who ELSE derives from it

    # --- bounds and presentation ---
    choices: Optional[Tuple[str, ...]] = None   # required when type == "enum"
    range:   Optional[Tuple[float, float]] = None
    unit:    Optional[str] = None
    group:   Optional[str] = None           # workflow_group: profile/stage/budget

    # --- what a SURFACE needs (generator.md § 3.1a) -------------------- #
    # Added 2026-08-11 with the decision that the UI is built FROM the
    # template rather than merely generated from the same schema.  Without
    # these three a template cannot name its own fields or group them, and
    # ``optional`` says unset is a real state while nothing says how to show
    # it.  They cost nothing -- they are already in the field metadata -- and
    # adding them later would mean re-emitting every template written before.
    label:      str = ""                    # "MPI ranks (np)" -- the human name
    null_label: str = ""                    # "(auto)" -- what UNSET is called

    # --- § 6.2 / § 6.3 / § 6.4 (schema @2) ---------------------------- #
    #: Which questions this item answers.  FIRST is the panel; the rest make
    #: it findable (§ 6.2).  Replaced ``section``, which held a free-text
    #: fieldset name PER ENGINE ("SCF", "Compute & budget"), so two engines
    #: expressing one idea disagreed on the label and no surface could group
    #: across them.
    category: Tuple[str, ...] = ()
    #: Which engines this item applies to.  EMPTY MEANS ALL (§ 6.3).
    engines:  Tuple[str, ...] = ()
    #: Who computes this item's value when it is unset -- a NAME from a closed
    #: registry, never code (§ 6.4).  A template carrying executable content
    #: would be something you must trust rather than something you can read.
    resolver: str = ""

    #: Whether *unset* is a state this item has at all. Not written to the
    #: file — it is recoverable from the schema and is carried here because
    #: :func:`schema_fingerprint` has always included it.
    optional: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            _refuse(f"kind {self.kind!r} is not one of {', '.join(KINDS)}",
                    where=self.name)
        if self.type not in TYPES:
            _refuse(f"type {self.type!r} is not one of {', '.join(TYPES)}",
                    where=self.name)
        # § 3's conditionally-required keys. Each is required *by the item's own
        # declaration*, so the check belongs on the object and not only on the
        # parse -- a producer building Items by hand gets the same refusal.
        if self.kind == "engine" and not self.anchor:
            _refuse("kind='engine' needs an 'anchor' -- an engine item that "
                    "names no keyword cannot reach the deck", where=self.name)
        if self.kind == "deck" and not self.expands:
            _refuse("kind='deck' needs 'expands' -- it is how a reader learns "
                    "which keywords this item produces", where=self.name)
        if self.type == "enum" and not self.choices:
            _refuse("type='enum' needs 'choices' -- an enum with no members "
                    "cannot be validated or rendered as a control",
                    where=self.name)

    @property
    def is_set(self) -> bool:
        """Whether this item carries a value (§ 3's *absent means unset*)."""
        return self.value is not None


@dataclass(frozen=True)
class Template:
    """A parsed template: the three top-level keys, and the items in order."""
    engine: str
    fingerprint: str
    items: Tuple[Item, ...] = ()

    def __iter__(self):
        return iter(self.items)

    def get(self, name: str) -> Optional[Item]:
        for it in self.items:
            if it.name == name:
                return it
        return None

    def values(self) -> Dict[str, Any]:
        """``{name: value}`` for every item that carries one.

        Items that are **unset** are omitted rather than mapped to ``None``:
        see :func:`config_from_template` for why that is the correct reading
        and not a loss.
        """
        return {it.name: it.value for it in self.items if it.is_set}


# --------------------------------------------------------------------- #
#  From the schema — the edge that makes this data-driven               #
# --------------------------------------------------------------------- #

#: How a Python annotation becomes a declaration ``type``.  ``bool`` is checked
#: BEFORE ``int`` -- a bool IS an int in Python, and getting that order wrong
#: types every checkbox as a number.
_ANNOTATION_TYPES = {bool: "bool", int: "int", float: "float", str: "str"}

#: Which ``kind`` a field takes when its metadata does not say.  Almost every
#: exposed field is one of the engine's own keywords; the exceptions declare
#: themselves with an ``item_kind`` in their metadata.
_DEFAULT_KIND = "engine"


def _unwrap_optional(ann) -> Tuple[Any, bool]:
    """``Optional[X]`` → ``(X, True)``; anything else → ``(ann, False)``."""
    if typing.get_origin(ann) is typing.Union:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


_BARE_ANCHOR = re.compile(r"^\s*(%block\s+[A-Za-z][\w.-]*|[A-Za-z][\w.-]*)")


def _bare_anchor(engine_key: str) -> str:
    """The keyword an ``engine_key`` leads with, or ``""`` if it leads with none.

    § 5: an anchor is *"a bare keyword, never a sentence"*, but ``engine_key``
    is older than that rule and carries three other shapes, measured against
    ``SiestaConfig``: a **note** (``(molbuilder: ...)``), an **alternation**
    (``A | B``), and a **conjunction** (``A + B``). The last two are ``deck``
    items whose ``expands`` lists every keyword they may produce, declared in
    metadata; the first is not an engine item at all and is refused above.

    What is left for this function is the common case and one tidy-up: a bare
    keyword, possibly followed by a prose note in parentheses, of which only
    the keyword is the anchor.
    """
    m = _BARE_ANCHOR.match(engine_key or "")
    return m.group(1) if m else ""


def _decl_type(ann, choices) -> Optional[str]:
    """The declaration ``type`` for one annotation, or None if unnameable."""
    if choices:
        return "enum"
    if ann in _ANNOTATION_TYPES:
        return _ANNOTATION_TYPES[ann]
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    # Tuple[int, int, int] -- the k-grid.  Named rather than flattened to three
    # fields: it is ONE decision ("how finely is reciprocal space sampled"), and
    # a stage overriding it overrides all three together.
    if origin is tuple and args and all(a is int for a in args):
        return "int3"
    if origin is list and args:
        if args[0] is str:
            return "strlist"
        if args[0] is int:
            return "intlist"
    return None


def declaration_for(f: "dataclasses.Field", annotation) -> Optional[Item]:
    """The § 3 item for one config field, or ``None`` if it has no place here.

    ``None`` means **excluded by § 7's named rows** (a machine fact, or the
    ladder) — never by a missing ``section``: since U16 membership is total
    and ``section`` answers only *where on the form* (empty = no tab shows
    it, and the item still travels).  *(R8: this paragraph taught the
    retired section gate ten lines above the U16 comment that deleted it.)*

    Raises ``ValueError`` for an exposed field whose type has no name in the
    grammar: that is a gap in the vocabulary, and the loud version of it is the
    only one that gets fixed.
    """
    # NO section gate (U16, 2026-08-12).  ``section`` answers *where on
    # the form* -- a surface hint, legitimately absent for a field no tab
    # shows -- while membership is § 7's TOTAL rule: every parameter the
    # schema declares is an item, excluded only by the three rows below.
    # The gate that stood here was a fourth, unlisted exclusion, and it
    # silently kept species_order (identity-sensitive, run-identity
    # § 6a), write_forces, write_coor_step, write_molwatch_log and
    # copy_psml out of every template.

    # § 7 lists three things that are NOT items, and the first is "a machine
    # fact -- ranks, GPUs, queue, partition, wall time".  A field that declares
    # itself one is excluded HERE rather than by having its ``section`` taken
    # away, because a `section` answers *"may a surface show this"* and this
    # answers *"is it part of the calculation's description"* -- two questions
    # that happen to have had one switch.
    if f.metadata.get("allocation"):
        return None

    ann, optional = _unwrap_optional(annotation)

    # A ``List[<dataclass>]`` is a STAGE LADDER, and a ladder is not a template
    # item -- it is the user's decision about what varies, and it lives in
    # ``task.json`` (`engines/stages.md` § 1.1).  Excluded for what it IS, with
    # a reason, rather than left to fall through to the type error below, which
    # would report a vocabulary gap where there is none.
    _args = typing.get_args(ann)
    if (typing.get_origin(ann) in (list, tuple)
            and _args and dataclasses.is_dataclass(_args[0])):
        return None

    choices = f.metadata.get("choices")
    type_ = _decl_type(ann, choices)
    if type_ is None:
        raise ValueError(
            f"field {f.name!r}: no declaration type for annotation {ann!r}. "
            f"Add one to the grammar (engines/template.md § 5) rather than "
            f"leaving the field out of the template -- § 7's premise is that "
            f"every parameter the schema declares is an item.")

    rng = f.metadata.get("range")
    # § 6.2 -- REQUIRED, and validated here rather than at write time so the
    # refusal names the field.  A category cannot change the deck, but an item
    # without one has no panel to appear on, which is G2 ("enough on its own
    # for a surface") failing quietly.
    category = tuple(f.metadata.get("category", ()) or ())
    if not category:
        raise ValueError(
            f"field {f.name!r}: no `category`. Every item declares which "
            f"question about the calculation it answers "
            f"(engines/template.md § 6.2); the vocabulary is {CATEGORIES}.")
    for _c in category:
        if _c not in CATEGORIES:
            raise ValueError(
                f"field {f.name!r}: unknown category {_c!r}. The vocabulary "
                f"is closed: {CATEGORIES}.")
    engines = tuple(f.metadata.get("engines", ()) or ())
    resolver = str(f.metadata.get("resolver", "") or "")
    kind = f.metadata.get("item_kind") or _DEFAULT_KIND
    anchor = _bare_anchor(f.metadata.get("engine_key", "") or f.name)
    expands = tuple(f.metadata.get("expands", ()) or ())

    # § 7: "a parameter that cannot be given a ``kind`` is a gap in this
    # vocabulary, and the loud version of that is the only one that gets
    # fixed."  An ``engine_key`` that is a molbuilder NOTE rather than a
    # keyword -- ``(molbuilder: ...)`` -- names nothing the deck can carry, so
    # a field left at the default kind is not classified, it is unclassified.
    if kind == "engine" and not anchor:
        raise ValueError(
            f"field {f.name!r}: kind defaults to 'engine' but its engine_key "
            f"names no keyword ({f.metadata.get('engine_key','')!r}). Give it "
            f"an explicit metadata['item_kind'] -- one of "
            f"{', '.join(KINDS)} (engines/template.md § 6). Leaving it out "
            f"would put a note where the deck expects a keyword.")
    if kind == "deck" and not expands and anchor:
        expands = (anchor,)

    return Item(
        name=f.name,
        kind=kind,
        type=type_,
        help=str(f.metadata.get("help", "") or ""),
        default=(f.default if f.default is not dataclasses.MISSING else None),
        anchor=(anchor if kind == "engine" else ""),
        expands=expands,
        read_by=tuple(f.metadata.get("read_by", ()) or ()),
        choices=(tuple(choices) if choices else None),
        range=(tuple(rng) if rng else None),
        unit=f.metadata.get("unit"),
        group=f.metadata.get("workflow_group"),
        label=str(f.metadata.get("label", "") or ""),
        null_label=str(f.metadata.get("null_label", "") or ""),
        optional=optional,
        category=category,
        engines=engines,
        resolver=resolver,
    )


def declarations_for(config_cls) -> List[Item]:
    """Every exposed field of *config_cls*, in declaration order.

    Declaration order, not alphabetical: the config's field order is the form's
    order and the deck's order, and a template a person reads should not be a
    third arrangement of the same things.
    """
    hints = typing.get_type_hints(config_cls)
    out: List[Item] = []
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

    ``task.json`` carries one (`engines/stages.md` § 6.6), and the preflight's
    only **non-refusal** row is *"the schema fingerprint matches"* — a
    description written when ``mesh_cutoff`` was bounded [50, 2000] and read
    after it became [50, 800] names a field that still exists and a value that
    is no longer legal, which is worth saying out loud rather than discovering
    at the engine.

    What goes in is what a *description* can depend on: the field's name, its
    declaration type, its bounds, its enum members, and whether unset is a state
    it has. Deliberately NOT included:

      * **the default** — a template records the value in use, so changing a
        default cannot invalidate a description that already carries values;
      * **help text, labels, units, ``group``, ``category``** — presentation. A
        reworded tooltip must not make every stored description suspect, and a
        fingerprint that cried wolf would be turned off.

    So it changes when a field is added, removed, retyped, re-bounded, or has
    its choices changed — and not otherwise.

    **The recipe is unchanged from the item-block era on purpose.** Every
    fingerprint already stored in a ``task.json`` was computed by it, and
    altering the recipe would invalidate all of them at once for no gain.
    """
    parts: List[str] = [f"v{FINGERPRINT_VERSION}"]
    for d in sorted(declarations_for(config_cls), key=lambda d: d.name):
        rng = f"{d.range[0]},{d.range[1]}" if d.range else ""
        choices = "|".join(d.choices) if d.choices else ""
        parts.append(f"{d.name}:{d.type}:{rng}:{choices}:{int(d.optional)}")
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
#  Writing                                                              #
# --------------------------------------------------------------------- #

def _toml_basic(s: str) -> str:
    """A TOML basic string — one line, escapes processed."""
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return f'"{out}"'


def _toml_multiline(s: str) -> str:
    """A TOML multi-line basic string, for prose that carries newlines.

    Multi-line **basic** (``\"\"\"``) rather than literal (``'''``) because a
    literal string has no escape at all: a body containing ``'''`` could not be
    written, and help text is prose we do not control the punctuation of.
    """
    body = s.replace("\\", "\\\\").replace('"""', '\\"""')
    # A body ending in a quote would merge with the closing delimiter.
    if body.endswith('"'):
        body = body[:-1] + '\\"'
    # TOML trims a newline immediately after the opening delimiter, so the
    # value is exactly ``body`` -- which the round-trip check below proves.
    return f'"""\n{body}"""'


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):                 # before int -- a bool IS an int
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        # repr keeps the decimal point, so a float round-trips as a float and
        # not as an int -- 300.0 must not come back as 300.
        return repr(v)
    if isinstance(v, str):
        return (_toml_multiline(v) if "\n" in v else _toml_basic(v))
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"template: cannot write {type(v).__name__} to TOML")


#: The order keys appear inside an item.  Fixed so two templates of the same
#: calculation diff cleanly, and so the file reads the way § 4.2's example does:
#: what it is, then what it is worth, then what bounds it, then the prose.
_ITEM_KEY_ORDER = ("kind", "category", "engines", "anchor", "expands", "type",
                   "choices", "value", "default", "resolver", "unit", "range",
                   "group", "label", "null_label", "read_by", "help")


def _item_payload(it: Item) -> Dict[str, Any]:
    """The mapping one item becomes — omitting everything absent (§ 3)."""
    out: Dict[str, Any] = {"kind": it.kind, "type": it.type}
    if it.anchor:
        out["anchor"] = it.anchor
    if it.expands:
        out["expands"] = list(it.expands)
    if it.choices:
        out["choices"] = list(it.choices)
    # An absent ``value`` is the encoding of *explicitly unset* (§ 3).
    if it.value is not None:
        out["value"] = list(it.value) if isinstance(it.value, tuple) else it.value
    if it.default is not None:
        out["default"] = (list(it.default) if isinstance(it.default, tuple)
                          else it.default)
    if it.unit:
        out["unit"] = it.unit
    if it.range:
        out["range"] = list(it.range)
    if it.group:
        out["group"] = it.group
    if it.category:
        out["category"] = list(it.category)
    if it.engines:
        out["engines"] = list(it.engines)
    if it.resolver:
        out["resolver"] = it.resolver
    if it.label:
        out["label"] = it.label
    if it.null_label:
        out["null_label"] = it.null_label
    if it.read_by:
        out["read_by"] = list(it.read_by)
    out["help"] = it.help
    return out


def render_template(config, *, config_cls=None, engine: str = "",
                    title: str = "") -> str:
    """The template for *config*, as TOML.

    **It takes no deck.** The catalogue comes from the schema and the values
    from *config*; a deck is what ``prep`` renders *from* this, later and on the
    machine that will run it. Until 2026-08-11 this function took the rendered
    deck and lifted payloads out of it, which inverted the contract's direction
    and stored every value twice.

    *engine* names whose schema these items belong to (§ 3); it defaults to the
    config class's own ``ENGINE`` attribute or its lower-cased class-name stem.

    Raises ``ValueError`` if the emitted text does not parse back to what it
    meant to write — see the module docstring; § 4.1 asks for exactly this.
    """
    cls = config_cls or type(config)
    eng = engine or _engine_name(cls)
    items = [
        dataclasses.replace(d, value=getattr(config, d.name, None))
        for d in declarations_for(cls)
    ]
    # Unit 4a's rule: the fingerprint is computed by whatever writes the
    # template, because that is the moment the schema is in hand.  Nothing
    # wrote one until 2026-08-11, so `validation/task.py`'s check either never
    # fired or always complained.
    payload = {
        "schema": SCHEMA,
        "engines": [eng],
        "fingerprint": schema_fingerprint(cls),
        "item": {it.name: _item_payload(it) for it in items},
    }

    lines: List[str] = []
    if title:
        for ln in title.splitlines():
            lines.append(f"# {ln}".rstrip())
    lines.append(f"schema      = {_toml_value(SCHEMA)}")
    lines.append(f"engines     = {_toml_value([eng])}")
    lines.append(f"fingerprint = {_toml_value(payload['fingerprint'])}")
    for it in items:
        body = _item_payload(it)
        lines.append("")
        lines.append(f"[item.{it.name}]")
        for key in _ITEM_KEY_ORDER:
            if key in body:
                lines.append(f"{key} = {_toml_value(body[key])}")
    text = "\n".join(lines) + "\n"

    # § 4.1 -- read our own output back and compare it with what we meant.
    # tomllib does not write TOML, so this is the only thing standing between a
    # quoting bug and a template that parses cleanly and says something else.
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:      # pragma: no cover - defensive
        raise ValueError(
            f"template: emitted TOML does not parse ({exc}). This is a bug in "
            f"the writer, not in the config.") from exc
    if parsed != payload:
        diff = _first_difference(payload, parsed)
        raise ValueError(
            f"template: emitted TOML does not read back as written -- {diff}. "
            f"This is a bug in the writer (engines/template.md § 4.1 asks for "
            f"exactly this check).")
    return text


def _engine_name(cls) -> str:
    """Whose schema these items are — from the class, never guessed by a caller."""
    named = getattr(cls, "ENGINE", "")
    if named:
        return str(named)
    stem = cls.__name__
    for suffix in ("Config", "Configuration"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.lower()


def _first_difference(want: Any, got: Any, path: str = "") -> str:
    """Where two payloads first disagree, named so the bug is findable."""
    if isinstance(want, Mapping) and isinstance(got, Mapping):
        for k in want:
            if k not in got:
                return f"{path}.{k} is missing after the round trip".lstrip(".")
            sub = _first_difference(want[k], got[k], f"{path}.{k}")
            if sub:
                return sub
        for k in got:
            if k not in want:
                return f"{path}.{k} appeared from nowhere".lstrip(".")
        return ""
    if want != got:
        return (f"{path or 'the document'}: wrote {want!r}, read back {got!r}")
    return ""


# --------------------------------------------------------------------- #
#  Reading                                                              #
# --------------------------------------------------------------------- #

_REQUIRED_ITEM_KEYS = ("kind", "category", "type", "help")

_KNOWN_ITEM_KEYS = frozenset(_ITEM_KEY_ORDER)


def read_template(text: str) -> Template:
    """Parse a template, refusing rather than guessing.

    The order is § 3's: the schema string first, so a file from a future major
    fails saying so instead of failing on whichever key moved.

    Every refusal names the item it is about, because a template is a file a
    person is invited to edit (§ 4.1: *"hand-editable — yes"*), and *"missing
    required key 'kind'"* with no item name sends them to read the whole file.
    """
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        _refuse(f"not valid TOML -- {exc}")

    check_schema(str(raw.get("schema") or ""), SCHEMA, label="template")

    # § 3 at @2: `engines` is a LIST -- a template describes a CALCULATION and
    # a calculation may run on more than one engine (§ 6.3).  A @1 file names
    # one engine as a string; read it as a single-element list rather than
    # refusing, since that is exactly what it meant.
    engines = raw.get("engines")
    if isinstance(engines, str):            # tolerated @1 spelling
        engines = [engines]
    if engines is None and isinstance(raw.get("engine"), str):
        engines = [raw["engine"]]
    if not engines or not isinstance(engines, list) or \
            not all(isinstance(e, str) and e for e in engines):
        _refuse("missing required key 'engines' -- without it a reader cannot "
                "know which config classes these items belong to, and would "
                "have to infer it from their names (engines/template.md § 3)")
    engine = engines[0]
    if "fingerprint" not in raw:
        _refuse("missing required key 'fingerprint'. An EMPTY string is legal "
                "and means 'makes no claim'; the key itself is required so "
                "that silence is deliberate (engines/template.md § 3)")
    fingerprint = str(raw.get("fingerprint") or "")

    table = raw.get("item") or {}
    if not isinstance(table, Mapping):
        _refuse(f"'item' must be a table of items, got "
                f"{type(table).__name__}")

    items = tuple(_item_from(name, body) for name, body in table.items())
    # § 5: the ``type`` "types what a reader must check, which a parser
    # cannot know".  Until the U-program follow-up (2026-08-12) NO reader
    # checked: a hand-edit could put the string "three hundred" into a
    # float item and it flowed through config_from_template into the
    # engine config unrefused -- surfacing later, in rendering, with a
    # message about anything but the edit that caused it.
    # (value checking moved INTO _item_from at R4 -- raw, pre-shape; a
    # loop here saw values _shape had already coerced.)
    return Template(engine=str(engine), fingerprint=fingerprint, items=items)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


#: § 5's checkable meaning of each ``type``, for a SET value.
_TYPE_CHECKS = {
    "int":     _is_int,
    "float":   lambda v: _is_int(v) or isinstance(v, float),
    "str":     lambda v: isinstance(v, str),
    "text":    lambda v: isinstance(v, str),
    "bool":    lambda v: isinstance(v, bool),
    "enum":    lambda v: isinstance(v, str),
    "pow2":    lambda v: _is_int(v) and v > 0 and (v & (v - 1)) == 0,
    "int3":    lambda v: (isinstance(v, (list, tuple)) and len(v) == 3
                          and all(_is_int(x) for x in v)),
    "strlist": lambda v: (isinstance(v, list)
                          and all(isinstance(x, str) for x in v)),
    "intlist": lambda v: (isinstance(v, list) and all(_is_int(x) for x in v)),
}


def _check_raw_value(name: str, key: str, raw, type_: str,
                     choices) -> None:
    """Refuse a SET value (or default) that is not what its own
    declaration says it is -- on the RAW parsed TOML, before ``_shape``
    touches it.  ``range`` stays advisory (§ 3.3 calls it *"advisory
    bounds"* -- a value outside it is a choice, not a type error); the
    TYPE and an enum's membership are not advice."""
    if raw is None:
        return
    check = _TYPE_CHECKS.get(type_)
    if check is not None and not check(raw):
        raise ValueError(
            f"template: item {name!r} declares type {type_!r} but its "
            f"{key} is {raw!r}.  This file is hand-editable and the "
            f"declaration is the contract for the edit "
            f"(engines/template.md § 5).")
    if type_ == "enum" and choices and raw not in choices:
        raise ValueError(
            f"template: item {name!r} is an enum of "
            f"{', '.join(map(repr, choices))} but its {key} is {raw!r}.")


def _shape(v: Any, type_: str) -> Any:
    """Give a parsed value the Python shape its declared type implies.

    TOML has one sequence, and the config classes do not: ``kgrid`` is a
    ``Tuple[int, int, int]`` while ``species_order`` is a ``List[str]``, and
    both come back from ``tomllib`` as a plain list. Without this an item
    round-trips to a value that is *equal in content and different in type*,
    which is the quietest kind of loss — and it is what the writer's own
    round-trip check cannot see, because the check compares the TOML payload
    with itself rather than the config with the config.

    **This is not the deleted ``_coerce``.** That one parsed *strings* back
    into values because the item-block format stored everything as text. TOML
    types its own scalars; the only thing left to decide is list versus tuple,
    and the declared type is what decides it.
    """
    if v is None:
        return None
    if type_ == "int3":
        return tuple(int(x) for x in v)
    if type_ == "intlist":
        return [int(x) for x in v]
    if type_ == "strlist":
        return [str(x) for x in v]
    return v


def _item_from(name: str, body: Any) -> Item:
    if not isinstance(body, Mapping):
        _refuse(f"expected a table, got {type(body).__name__}", where=name)
    unknown = sorted(set(body) - _KNOWN_ITEM_KEYS)
    if unknown:
        _refuse(f"unknown key(s) {', '.join(repr(k) for k in unknown)} "
                f"(known: {', '.join(_ITEM_KEY_ORDER)})", where=name)
    for key in _REQUIRED_ITEM_KEYS:
        if key not in body:
            _refuse(f"missing required key {key!r}", where=name)

    rng = body.get("range")
    type_ = str(body["type"])
    # § 5's type check runs on the RAW TOML value, BEFORE _shape gives it
    # a Python shape (R4, 2026-08-12: the check ran post-construction, so
    # _shape mangled first -- a scalar on a strlist exploded "Au" into
    # ['A','u'] and PASSED, and a scalar on int3 died as a raw TypeError
    # naming no item).
    choices = body.get("choices")
    for key in ("value", "default"):
        _check_raw_value(name, key, body.get(key), type_,
                         tuple(choices) if choices else None)
    return Item(                       # Item.__post_init__ enforces § 3's rest
        name=name,
        kind=str(body["kind"]),
        type=type_,
        help=str(body["help"]),
        value=_shape(body.get("value"), type_),
        default=_shape(body.get("default"), type_),
        anchor=str(body.get("anchor", "") or ""),
        expands=tuple(body.get("expands", ()) or ()),
        read_by=tuple(body.get("read_by", ()) or ()),
        choices=(tuple(body["choices"]) if body.get("choices") else None),
        range=(tuple(rng) if rng else None),
        unit=body.get("unit"),
        group=body.get("group"),
        label=str(body.get("label", "") or ""),
        category=tuple(body.get("category", ()) or ()),
        engines=tuple(body.get("engines", ()) or ()),
        resolver=str(body.get("resolver", "") or ""),
        null_label=str(body.get("null_label", "") or ""),
    )


def template_fields(config_cls) -> set:
    """The field names a TEMPLATE may carry — the schema minus the machine
    facts.

    THE membership rule, spelled once (A-9, 2026-08-13).  A field tagged
    ``allocation: True`` is § 7's forbidden machine fact: it arrives as
    the ALLOCATION at `prep`, on the machine that will run it, and never
    as a template item, a stage override, a pin, or a parameter sweep
    axis.  :func:`declaration_for` already excluded such fields from the
    WRITE side; every read-side gate used ``dataclasses.fields`` names
    instead, so a hand-edited ``mpi_np`` item / override / pin passed and
    the deck rendered for a rank count the allocation never granted.
    """
    return {f.name for f in dataclasses.fields(config_cls)
            if not f.metadata.get("allocation")}


def config_from_template(text: str, config_cls):
    """An ordinary instance of *config_cls*, rebuilt from a template.

    What ``prep`` holds before it applies a stage's ``overrides``
    (`engines/stages.md` § 4).

    **An unset item is omitted rather than passed as ``None``**, so the class
    default applies. That is the correct reading rather than a loss: every
    field for which *unset* is a real state is annotated ``Optional[...]`` and
    defaults to ``None`` already, so omitting and passing ``None`` agree —
    while for a field that is not optional, passing ``None`` would replace a
    real default with a value its own type forbids.

    A field the template does not carry keeps the class default too: a template
    written against an older schema is **missing** fields, not wrong about them,
    and the fingerprint is what says so.
    """
    known = template_fields(config_cls)
    vals = read_template(text).values()
    machine = sorted(k for k in vals
                     if k not in known
                     and any(f.name == k
                             for f in dataclasses.fields(config_cls)))
    if machine:
        # The WRITE side never emits these (declaration_for returns None
        # for an allocation-tagged field), so one in a template is a hand
        # edit -- refused with the § 7 story rather than the typo story.
        raise ValueError(
            f"template names machine fact(s) "
            f"{', '.join(map(repr, machine))}, which floor 2 must never "
            f"carry (engines/template.md § 7): they arrive as the "
            f"ALLOCATION at `prep`, on the machine that runs the job.  "
            f"Remove them from the template and state them at prep.")
    unknown = sorted(k for k in vals if k not in known)
    if unknown:
        # Refused, never dropped (U16): this is a file people edit by
        # hand, and an item the schema does not know is a typo or a
        # renamed field -- either way, silently ignoring it renders a
        # deck missing what the person believes they set.
        raise ValueError(
            f"template names item(s) the {config_cls.__name__} schema does "
            f"not declare: {', '.join(map(repr, unknown))}.  A template "
            f"item is a schema field (engines/template.md § 7); check the "
            f"spelling against the schema's own names.")
    return config_cls(**vals)


__all__ = ["SCHEMA", "SUFFIX", "KINDS", "TYPES", "FINGERPRINT_VERSION",
           "Item", "Template",
           "declaration_for", "declarations_for",
           "schema_fingerprint", "fingerprint_matches",
           "render_template", "read_template", "config_from_template",
           "template_fields"]
