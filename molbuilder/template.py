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

TOML has no payload key, so nothing needs a deck and the whole lifting
apparatus — ``_anchor_token``, ``_payload_for``, ``_DECL_RE``, ``_coerce`` —
is gone rather than ported.

THE WRITER CHECKS ITSELF.  ``tomllib`` reads TOML and does not write it, so the
emitter here is hand-rolled — and § 4.1 requires that whatever emits a template
**read its own output back and compare it to what it meant to write**.
:func:`_emit` builds the payload as a plain object, serialises it, parses the
result, and refuses if the two differ.  That turns *"we emitted TOML correctly"*
from an assumption into a property checked on every call.

**THE CATALOGUE IS THE MASTER** (§ 2.1, § 4.3).  ``molbuilder/data/
catalogue.template.toml`` is authored as TOML and holds every parameter both
engines declare; :func:`template_with_values` narrows it to one engine and fills
in what a config holds, which is what a surface produces once a person has
answered.  ``render_template(config)`` — which reflected a Python class into a
file, making the class the master and the file its printout — was **deleted
2026-08-14**.  A config class is a translator on the way OUT: template → config
→ deck.
"""
from __future__ import annotations

import dataclasses
import re
import tomllib
import types
import typing
from pathlib import Path as _Path
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

#: The closed RESOLVER vocabulary (§ 6.4).  An item that cannot be answered by
#: a constant NAMES who computes its value, and ``prep`` calls it -- so no
#: layer carries a list of which fields are special, which is G3 again.
#:
#: A NAME, never code.  A template is data: hand-editable, and it travels
#: between machines.  Executable content in it would end both properties and
#: make a description something you must TRUST rather than something you can
#: READ.  An unknown name is an error a reader reports (§ 3).
#:
#: ``allocation`` resolvers answer from what the scheduler GRANTED and their
#: items are always valueless; ``block_size`` proposes a value a person may
#: also set or measure.
RESOLVERS = ("rank_count", "omp_threads", "node_memory", "block_size")

#: The resolvers that answer from what the scheduler GRANTED.  An item naming
#: one may never carry a value: that is § 2's rule that the template declares
#: the QUESTION and never asserts the ANSWER, and it is checked on READ because
#: a template is a file a person is invited to edit.  ``block_size`` is
#: deliberately absent -- prep PROPOSES it, but a person or a benchmark may
#: also set it, and honouring that was a 2026-08-11 user decision.
ALLOCATION_RESOLVERS = ("rank_count", "omp_threads", "node_memory")

#: § 5 — the validation vocabulary. TOML types the *storage*; this types what a
#: reader must check, which a parser cannot know: that ``pow2`` is a power of
#: two, that ``enum`` is drawn from ``choices``, that ``text`` is verbatim engine
#: text to be copied rather than interpreted.
TYPES = ("int", "float", "str", "bool", "enum", "pow2",
         "int3", "float3", "strlist", "intlist", "text")

#: The closed GROUP vocabulary -- *when, and on which surface, do I set this?*
#: It is the OUTER card of a form, load-bearing since 2026-06-13 (the stage
#: selector once silently rewrote budget and system fields), and it is a
#: different axis from :data:`CATEGORIES`, which asks what QUESTION about the
#: calculation an item answers.
#:
#: ``output`` was added 2026-08-15.  The three cards answered *what am I
#: computing*, *how tight*, and *how much compute* -- and there were always
#: FOUR questions: eleven items answer *what do I get back*, of which four sat
#: mis-filed under ``profile`` and seven carried no group at all and rendered
#: loose beneath the cards.
#:
#: ``staging`` names a parameter set by the STAGING surface rather than by the
#: tab that collects the physics (user, 2026-08-15). The item is a real
#: parameter and the template carries it; the parameter form simply is not
#: where it is answered.
#:
#: **Unlike `category` this is not required on every item** -- it is
#: presentation, and a template read headlessly by ``prep`` never asks. It is
#: required on every item of the CATALOGUE, which is a different claim and is
#: guarded by ``tests/test_catalogue_agreement.py``: an item there with no
#: group is a parameter no surface can place.
#: ``setup`` is FIRST because its members are the answers a calculation
#: cannot be built without: what the run is called, and where its
#: pseudopotentials come from (user, 2026-08-15). They lived in ``profile``,
#: whose own subtitle promised "identity (label, pseudopotentials, charge)" --
#: but a card is ordered by `category`, so the label sorted under *procedure*
#: near the bottom and the pseudopotential directory under *method* in the
#: middle. The two things a user must decide first were the two hardest to
#: find. A card is the only fix: re-ordering within one card would have to
#: break the shared legend order that every other card depends on.
GROUPS = ("setup", "profile", "stage", "budget", "output", "staging")


def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(f"template{': ' + where if where else ''}: {msg}")


# --------------------------------------------------------------------- #
#  One item                                                             #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Item:
    """One parameter, with everything all four readers need from it.

    **The four § 3 requires are `kind`, `category`, `type` and `help`** —
    matching :data:`_REQUIRED_ITEM_KEYS`.  ``name`` is not among them because it
    is not a key at all: it is the table header, ``[item.<name>]``.  Everything
    else is
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

    #: **How the engine SPELLS this**, in full — and a different fact from
    #: :attr:`anchor`, which is the bare leading keyword the deck writer
    #: matches on (§ 5: *"an anchor is a bare keyword, never a sentence"*).
    #:
    #: The two are not interchangeable and collapsing them lost information
    #: twice over on 2026-08-14, when the catalogue kept only the anchor:
    #:
    #:   * ``gto.M(basis=...)``, ``(charge=...)``, ``(spin=...)`` and
    #:     ``(symmetry=...)`` all reduce to ``gto.M`` — four controls showing
    #:     one useless badge;
    #:   * ``mf = mf.density_fit()`` / ``.PCM()`` / ``.add_dispersion(...)``
    #:     all reduce to ``mf``;
    #:   * and a **molbuilder note** — ``(molbuilder: .run.sh ``mpirun -np N``
    #:     only; not in .fdf)`` — leads with no keyword at all, so eleven
    #:     items lost their badge entirely. That note is the ONLY way a
    #:     reader learns the setting never reaches the deck, which is why
    #:     `web/form-schema.md` § 1a requires it always be present.
    #:
    #: A surface shows this. The deck writer never reads it — it takes
    #: ``anchor`` and ``expands``, which is why truncating the anchor was
    #: right and keeping only the anchor was not.
    engine_key: str = ""

    #: **Where the engine's own documentation defines this keyword** — one
    #: string, naming the version so a citation that goes stale is visible
    #: rather than silently wrong: ``SIESTA 5.4.2 §6.9.2 'Mixing options'``.
    #:
    #: A REVIEWER'S field, and deliberately the only one that lives in the
    #: catalogue alone.  Nothing renders it: no surface shows it, no deck
    #: carries it, and no config class mirrors it (user, 2026-08-15: *"this
    #: does not have to show up in the UI or generated config, but serve as a
    #: reference point when review of template or revision of it could
    #: benefit from"*).  Mirroring it into the dataclass the way ``help`` and
    #: ``label`` are mirrored would give a SECOND home to the one fact whose
    #: entire job is to let someone check the first — so ``declaration_for``
    #: never sets it, and ``MIRRORED`` in `tests/test_catalogue_agreement.py`
    #: leaves it out on purpose.
    #:
    #: The citations were DERIVED, not recalled: the 5.4.2 manual sources
    #: were parsed for every ``\begin{fdfentry}{...}`` and matched to our
    #: anchors through fdf's own ``labeleq`` normalisation, which is how
    #: ``SaveHS`` finds the entry the manual spells ``Save.HS``.
    manual: str = ""

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

    #: Whether this item's value belongs to the ALLOCATION rather than to the
    #: calculation (§ 2, G1).  **Not written to the file, and recoverable from
    #: it**: the three allocation items are exactly those whose `resolver` is in
    #: :data:`ALLOCATION_RESOLVERS` (§ 6.4), which is how
    #: :func:`template_with_values` knows to emit them VALUELESS however the
    #: config is filled.  Set from the schema on the WRITE side, so an item read
    #: back from a file carries ``False`` -- ask the resolver, not this flag.
    allocation: bool = False

    #: Whether *unset* is a state this item has at all — and since 2026-08-14
    #: it IS written to the file.  A surface must offer *(auto)* / *(no cap)*,
    #: and it cannot be inferred from ``null_label``: 17 items are optional and
    #: only 12 carry one, so five would silently lose the option.
    optional: bool = False

    #: ``basic`` or ``advanced`` — a judgement about the PARAMETER, so a
    #: surface can dim the advanced ones rather than asking a first-time reader
    #: to weigh every knob at once.  Empty means unclassified.
    tier: str = ""

    #: A regex the value must match.  Two items have one (``system_label``,
    #: ``job_name``); nothing else in § 5's vocabulary can say *"letters,
    #: digits, hyphens, underscores; no dots"*.
    pattern: str = ""

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
        # § 3's fourth REQUIRED key, enforced here for the same reason as the
        # conditional three above: a producer building Items by hand must get
        # the same refusal the parser gives.  It was missing until 2026-08-14
        # (audit § 1.4) -- so the contract, the parser and `Item` disagreed
        # about how many keys are required, and the object was the lenient one.
        if not self.category:
            _refuse("no 'category' -- § 3 requires one on every item, and "
                    "without it a surface has no panel to put this on "
                    f"(the vocabulary is {', '.join(CATEGORIES)})",
                    where=self.name)
        for _c in self.category:
            if _c not in CATEGORIES:
                _refuse(f"category {_c!r} is not one of "
                        f"{', '.join(CATEGORIES)}", where=self.name)
        # § 6.1: `read_by` names LAYERS, so it draws from the same closed
        # vocabulary `kind` does.  Unchecked until 2026-08-14 (audit § 1.3a) --
        # any string passed, on write and on read, while `engines` beside it is
        # refused when unknown.
        for _r in self.read_by:
            if _r not in KINDS:
                _refuse(f"read_by names {_r!r}, which is not a layer -- the "
                        f"vocabulary is {', '.join(KINDS)} (§ 6.1)",
                        where=self.name)
        # `group` is OPTIONAL (presentation), but when present it is closed --
        # a typo would put the item on no card at all, which renders it loose
        # below the form and sends every finding about it to the residual
        # panel.  Silent, and indistinguishable from "deliberately untagged".
        if self.group is not None and self.group not in GROUPS:
            _refuse(f"group {self.group!r} is not one of "
                    f"{', '.join(GROUPS)}", where=self.name)
        # Same argument, one axis over: `resolver` names WHO ANSWERS an item
        # that carries no value, and three of the four names make the item
        # allocation-backed -- which is what the § 2 / G1 check on READ keys
        # off.  It was unchecked until 2026-08-15, so a hand-edited
        # ``resolver = "rank_kount"`` silently turned that protection OFF and
        # let a template assert a machine fact.  `read_by` beside it was given
        # this check on 2026-08-14 and this one was not swept with it.
        if self.resolver and self.resolver not in RESOLVERS:
            _refuse(f"resolver {self.resolver!r} is not one of "
                    f"{', '.join(RESOLVERS)} -- a resolver is a NAME "
                    f"molbuilder ships, never code in the file (§ 6.4). An "
                    f"unknown one would silently disable the check that keeps "
                    f"a machine fact out of a portable template (§ 2)",
                    where=self.name)

    @property
    def is_set(self) -> bool:
        """Whether this item carries a value (§ 3's *absent means unset*)."""
        return self.value is not None


@dataclass(frozen=True)
class Template:
    """A parsed template: the three top-level keys, and the items in order."""
    engine: str                     # engines[0] -- the @1 spelling, kept for callers
    #: Every engine this calculation can run on (§ 6.3).  ``engine`` above is
    #: its first element; a @1 file yields a one-element list.
    engines: Tuple[str, ...] = ()
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


#: The two spellings of a union.  ``Optional[X]`` and ``X | None`` mean the same
#: thing and report DIFFERENT origins -- ``typing.Union`` and ``types.UnionType``
#: -- so a check for one silently misses the other.  Nothing in the configs uses
#: the newer spelling today, which is exactly why this is worth naming: the first
#: field written that way would be declared NOT optional and then fail to get a
#: type at all, and the error would point at the annotation rather than at the
#: check that could not read it (audit § 1.1).
_UNION_ORIGINS = (typing.Union, types.UnionType)


def _unwrap_optional(ann) -> Tuple[Any, bool]:
    """``Optional[X]`` / ``X | None`` → ``(X, True)``; anything else → ``(ann, False)``."""
    if typing.get_origin(ann) in _UNION_ORIGINS:
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
    # a stage overriding it overrides all three together.  The same argument
    # makes ``Tuple[float, float, float]`` one item: the k-grid DISPLACEMENT is
    # one decision about where the mesh sits, per axis (audit § 54).
    # The length is checked because the type NAME asserts it -- a two-tuple
    # named ``int3`` would pass here and be refused later by a message naming
    # the value rather than the field.
    if origin is tuple and len(args) == 3:
        if all(a is int for a in args):
            return "int3"
        if all(a is float for a in args):
            return "float3"
    if origin is list and args:
        if args[0] is str:
            return "strlist"
        if args[0] is int:
            return "intlist"
    # NO ``strmap`` (``str | dict``).  It existed for exactly one field --
    # PySCF's ``ecp``, where "lanl2dz" meant one ECP for every heavy atom and
    # {"Au": "lanl2dz"} named them per element.  It modelled the SHAPE PySCF
    # happens to accept rather than the question a person answers, and the
    # question turned out to be two: which ECP, and which atoms.  ``ecp`` is
    # now a plain ``str`` beside an ``ecp_atoms`` ``strlist``, so both halves
    # are ordinary types, both render as controls a surface can validate, and
    # the union case has no member left.  Retired with it 2026-08-13.
    return None


def declaration_for(f: "dataclasses.Field", annotation) -> Optional[Item]:
    """The § 3 item for one config field, or ``None`` if it has no place here.

    ``None`` means **excluded by § 7's named rows** — a machine fact, or the
    ladder. Nothing else excludes an item: membership is total (D5).

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
    # § 6.4 (schema @2): a machine fact's VALUE is still forbidden, but the
    # ITEM is declared -- valueless, with the resolver that will answer it.
    # A surface must know the question exists in order to ask it, and the
    # wrapper writer must know to look.  Until @2 this returned None, so the
    # question was invisible: a UI reading the execution panel had no way to
    # ask for ranks, threads or memory at all.
    #
    # The protection is unchanged and lives where it always did: the value
    # never reaches a config, because ``template_fields`` strips these names
    # on the rebuild path.  Declaring the question is portable; answering it
    # on the wrong machine is not.
    _alloc = bool(f.metadata.get("allocation"))

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
    # A field may DECLARE its type when the annotation cannot carry the
    # constraint.  ``parallel_block_size`` is ``Optional[int]`` and the thing a
    # reader must check is *is it a power of two* -- which is § 5's whole reason
    # for ``type`` existing: "what a parser cannot know".  Without this the
    # vocabulary had a member (`pow2`) that nothing could produce, and a
    # user-supplied 96 reached the deck unchecked (audit § 50.1).
    declared = f.metadata.get("decl_type")
    if declared is not None:
        if declared not in TYPES:
            raise ValueError(
                f"field {f.name!r}: metadata['decl_type'] = {declared!r} is not "
                f"in the type vocabulary {TYPES} (engines/template.md § 5).")
        return_type = declared
    else:
        return_type = None
    type_ = return_type or _decl_type(ann, choices)
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
    if resolver and resolver not in RESOLVERS:
        raise ValueError(
            f"field {f.name!r}: unknown resolver {resolver!r}. The vocabulary "
            f"is closed: {RESOLVERS}. A resolver is a NAME molbuilder ships, "
            f"never code in the file (engines/template.md § 6.4).")
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
        allocation=_alloc,
        kind=kind,
        type=type_,
        help=str(f.metadata.get("help", "") or ""),
        default=(f.default if f.default is not dataclasses.MISSING else None),
        anchor=(anchor if kind == "engine" else ""),
        # The FULL spelling, whatever the kind -- a wrapper or produce item
        # names no deck keyword and its note is exactly what says so.
        engine_key=str(f.metadata.get("engine_key", "") or ""),
        expands=expands,
        read_by=tuple(f.metadata.get("read_by", ()) or ()),
        choices=(tuple(choices) if choices else None),
        range=(tuple(rng) if rng else None),
        unit=f.metadata.get("unit"),
        group=f.metadata.get("workflow_group"),
        label=str(f.metadata.get("label", "") or ""),
        null_label=str(f.metadata.get("null_label", "") or ""),
        optional=optional,
        tier=str(f.metadata.get('tier', '') or ''),
        pattern=str(f.metadata.get('pattern', '') or ''),
        category=category,
        engines=engines,
        resolver=resolver,
    )


def declarations_for(config_cls) -> List[Item]:
    """Every exposed field of *config_cls*, in declaration order.

    ⚠ **This is the direction § 2.1 retired, and it has NO production caller.**
    A parameter is defined in the catalogue (§ 4.3); a config class is a
    translator. This function survives for exactly two reasons, and both end:

    * it is how ``data/catalogue.template.toml`` was extracted from the classes
      on 2026-08-14 — a one-off that will not be repeated;
    * the live Build form still reads ``label`` / ``help`` / ``range`` /
      ``unit`` / ``choices`` off the dataclass fields
      (``web/blueprints/_shared.py``), so that metadata cannot be deleted until
      the form is rebuilt from the catalogue — **deferred**
      (`template-unification-plan.md` § 4, § 2.1a(b)).

    **Until then 307 facts live in two homes**, which is D3 broken, and
    ``tests/test_catalogue_agreement.py`` is the only thing keeping them in
    step. When the form moves, the metadata goes, and this function goes with
    it.

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
_ITEM_KEY_ORDER = ("kind", "category", "engines", "anchor", "engine_key",
                   "manual", "expands", "type",
                   "choices", "value", "default", "optional", "resolver",
                   "unit", "range", "tier", "pattern",
                   "group", "label", "null_label", "read_by", "help")


def _item_payload(it: Item) -> Dict[str, Any]:
    """The mapping one item becomes — omitting everything absent (§ 3)."""
    out: Dict[str, Any] = {"kind": it.kind, "type": it.type}
    if it.anchor:
        out["anchor"] = it.anchor
    if it.engine_key:
        out["engine_key"] = it.engine_key
    if it.manual:
        out["manual"] = it.manual
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
    if it.optional:
        out["optional"] = True
    if it.tier:
        out["tier"] = it.tier
    if it.pattern:
        out["pattern"] = it.pattern
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


#: The MASTER file — every parameter both engines declare (§ 4.3).  Shipped
#: with the package; authored as TOML, never generated.
CATALOGUE = _Path(__file__).resolve().parent / "data" / "catalogue.template.toml"


def load_catalogue() -> str:
    """The catalogue's text (§ 4.3).

    One master file, authored rather than generated: *"where is this parameter
    defined?"* has one answer for every parameter in every engine.
    """
    return CATALOGUE.read_text(encoding="utf-8")


def template_with_values(config, *, engine: str = "", catalogue: str = "",
                         title: str = "") -> str:
    """**This calculation's** template: the catalogue, narrowed to one engine,
    carrying the values *config* holds (§ 4.3).

    This is what a surface produces once a person has answered — the questions
    were already asked by the catalogue, so nothing here invents an item. It
    replaced ``render_template(config)``, which reflected a Python class into a
    file and thereby made the class the master (§ 2.1).

    **Allocation items stay valueless whatever the config holds** (§ 2, G1, and
    § 6.4): their resolver names who answers them, and the answer belongs to the
    machine that granted it. The config may legitimately carry a rank count — it
    was resolved somewhere — but writing it here would make the description
    assert a machine fact and stop being portable.

    Raises ``ValueError`` if the emitted text does not parse back to what it
    meant to write — § 4.1 asks for exactly this.
    """
    parsed = read_template(catalogue or load_catalogue())
    eng = engine or _engine_name(type(config))
    items = [
        dataclasses.replace(
            it,
            # A per-calculation template serves ONE engine, so no item narrows
            # anything and none carries `engines` (§ 6.3's writer rule).
            engines=(),
            value=(None if it.resolver in ALLOCATION_RESOLVERS
                   else getattr(config, it.name, it.value)),
        )
        for it in select(parsed, engine=eng)
    ]
    return _emit(items, engines=(eng,), title=title)


def _emit(items, *, engines, title: str = "") -> str:
    """Serialise items as a template file, and read the result back (§ 4.1).

    ``tomllib`` reads TOML and does not write it, so this is the only thing
    standing between a quoting bug and a file that parses cleanly and says
    something else.
    """
    payload = {
        "schema": SCHEMA,
        "engines": list(engines),
        "item": {it.name: _item_payload(it) for it in items},
    }

    lines: List[str] = []
    if title:
        for ln in title.splitlines():
            lines.append(f"# {ln}".rstrip())
    lines.append(f"schema      = {_toml_value(SCHEMA)}")
    lines.append(f"engines     = {_toml_value(list(engines))}")
    for it in items:
        body = _item_payload(it)
        lines.append("")
        lines.append(f"[item.{it.name}]")
        for key in _ITEM_KEY_ORDER:
            if key in body:
                lines.append(f"{key} = {_toml_value(body[key])}")
    text = "\n".join(lines) + "\n"

    try:
        got = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:      # pragma: no cover - defensive
        raise ValueError(
            f"template: emitted TOML does not parse ({exc}). This is a bug in "
            f"the writer, not in the data.") from exc
    if got != payload:
        raise ValueError(
            f"template: emitted TOML does not read back as written -- "
            f"{_first_difference(payload, got)}. This is a bug in the writer "
            f"(engines/template.md § 4.1 asks for exactly this check).")
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
    engines_t = tuple(engines)
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
    # § 2 / G1: an allocation-backed item may be DECLARED but never
    # ANSWERED here.  Checked on read, not only on write, because a template
    # is a file a person is invited to edit (§ 4.1) -- and a hand-edited
    # ``mpi_np`` value is precisely the failure this rule was written
    # against: a deck once rendered for a rank count the allocation never
    # granted.  The item itself is legitimate at @2; its VALUE is not.
    for _it in items:
        if _it.resolver in ALLOCATION_RESOLVERS and _it.value is not None:
            _refuse(
                f"carries value {_it.value!r}, but its resolver "
                f"{_it.resolver!r} answers from the ALLOCATION -- what the "
                f"scheduler granted on the machine that runs it. A template "
                f"declares the question and never asserts the answer, or it "
                f"stops being portable (engines/template.md § 2). Remove the "
                f"value; `prep` fills it", where=_it.name)

    # § 6.3: an item's `engines` narrows the FILE's list -- it cannot widen
    # it.  An item naming an engine this calculation does not run on is a
    # contradiction between the two, and accepting it would let a surface
    # filter for that engine and be handed items from a calculation that
    # never offered it.  Which half is wrong is not ours to guess, so the
    # reader reports the disagreement and names both sides.
    for _it in items:
        _stray = [e for e in _it.engines if e not in engines_t]
        if _stray:
            _refuse(
                f"declares engines {_stray} that the template does not list "
                f"(engines = {list(engines_t)}). An item's `engines` narrows "
                f"the file's list; it cannot widen it (engines/template.md "
                f"§ 6.3)", where=_it.name)
    return Template(engine=str(engine), engines=engines_t, items=items)


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
    # ``pow2`` CHECKS only that it is an int.  The power-of-two constraint is
    # not a refusal, it is a COERCION -- ``_shape`` snaps the value (user,
    # 2026-08-14: *"instead of accepting and refusing, it should correctly
    # coerce the value to the allowed number"*).  Refusing would send a person
    # who typed 96 away to work out which numbers are legal; snapping tells
    # them, by doing it.
    "pow2":    _is_int,
    "int3":    lambda v: (isinstance(v, (list, tuple)) and len(v) == 3
                          and all(_is_int(x) for x in v)),
    # ``float3`` accepts an int per component: TOML's ``0`` and ``0.0`` are
    # different types to a parser and the same displacement to SIESTA, and a
    # hand-edited file will carry both.  ``_shape`` makes them one.
    "float3":  lambda v: (isinstance(v, (list, tuple)) and len(v) == 3
                          and all(_is_int(x) or isinstance(x, float)
                                  for x in v)),
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
    if type_ == "pow2":
        # Snap DOWN to the nearest power of two, matching what the AUTO path
        # already does -- ``_auto_block_size`` takes the LARGEST power of two
        # that still leaves >= 2 blocks per rank, so rounding down is the
        # direction that cannot make a block bigger than the caller asked for.
        # 0 passes through: it is the THIRD STATE (*omit the keyword entirely*,
        # `template.md` § 12), a sentinel rather than a value.
        n = int(v)
        if n <= 0:
            return 0 if n == 0 else n
        return 1 << (n.bit_length() - 1)
    if type_ == "int3":
        return tuple(int(x) for x in v)
    if type_ == "float3":
        return tuple(float(x) for x in v)
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
        engine_key=str(body.get("engine_key", "") or ""),
        manual=str(body.get("manual", "") or ""),
        expands=tuple(body.get("expands", ()) or ()),
        optional=bool(body.get("optional", False)),
        tier=str(body.get("tier", "") or ""),
        pattern=str(body.get("pattern", "") or ""),
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


# --------------------------------------------------------------------- #
#  § 8.0 -- THE ONE READ API                                            #
# --------------------------------------------------------------------- #

def _as_tuple(v) -> Tuple[str, ...]:
    if v is None:
        return ()
    return (v,) if isinstance(v, str) else tuple(v)


def _check_engine(t: "Template", engine) -> None:
    """Refuse an engine this template does not serve (§ 3).

    *"This calculation does not run on that engine"* and *"no items
    matched"* are different answers, and an empty list cannot tell them
    apart -- a caller would read a refusal as an absence and carry on.
    """
    if engine and t.engines and engine not in t.engines:
        raise ValueError(
            f"this template describes a calculation for {list(t.engines)}, "
            f"not {engine!r}. An empty result would read as 'no items "
            f"matched' when the truth is 'this calculation does not run on "
            f"that engine' (engines/template.md § 3).")


def select(t: "Template", *, category=None, engine=None,
           kind=None, read_by=None) -> List[Item]:
    """The items matching every filter given, **in category order**.

    One function, one file, every reader (`engines/template.md` § 8.0).
    Each argument filters on an axis the item already declares; **omitting
    one means do not filter on it**, so ``select(t)`` is every item -- which
    is exactly what ``prep`` step 2 wants.

        select(t, category="accuracy", engine="siesta")   # one panel
        select(t, kind=("engine", "deck"), engine=e)      # the deck writer
        select(t, category="execution", engine=e)         # a benchmark

    **A filter, not a query language.** The axes are closed vocabularies
    declared on the item, so this is a dict comparison -- no expression to
    parse, no index to build, and a reader in another language does the same
    with ``tomllib.load`` and a comprehension. § 8's *"a reader never asks a
    second source"* holds: this is a convenience over data the caller already
    has in hand, never a service it must call.

    Category order, not declaration order, because the categories ARE the
    reading order (§ 6.2) and a surface renders panels top to bottom. Within
    a category the config's own field order survives, which is § 8's rule
    that a template a person reads is not a third arrangement of the same
    things.
    """
    _check_engine(t, engine)
    want_cat = _as_tuple(category)
    want_kind = _as_tuple(kind)
    # ``read_by`` filters the same way its siblings do -- any-of, scalar or
    # sequence.  It took a SCALAR ONLY until 2026-08-14, so a caller asking for
    # two layers got an empty list rather than an error: the tuple was compared
    # against the item's members and never matched (audit § 1.3).
    want_read_by = _as_tuple(read_by)
    out: List[Item] = []
    for it in t.items:
        # `engines` empty means ALL engines (§ 6.3) -- absence is not a
        # narrower claim than presence, it is a wider one.
        if engine and it.engines and engine not in it.engines:
            continue
        if want_cat and not (set(want_cat) & set(it.category)):
            continue
        if want_kind and it.kind not in want_kind:
            continue
        if want_read_by and not (set(want_read_by) & set(it.read_by)):
            continue
        out.append(it)

    def _rank(it: Item) -> int:
        # The item's PRIMARY category is its panel (§ 6.2); an item with no
        # category sorts last rather than crashing, because a reader's job
        # is to report a malformed file, not to be the thing that breaks.
        if not it.category:
            return len(CATEGORIES)
        try:
            return CATEGORIES.index(it.category[0])
        except ValueError:
            return len(CATEGORIES)

    return sorted(out, key=_rank)      # stable: ties keep declaration order


def one(t: "Template", name: str, *, engine: str = None) -> Optional[Item]:
    """The item called *name*, or ``None`` when it does not apply here.

    **``None`` and an exception mean different things, and that is the
    point.** ``None`` is *"this calculation has no such parameter for this
    engine"* -- ``one(t, "mesh_cutoff", engine="pyscf")`` on a template whose
    item declares ``engines = ["siesta"]``. It raises when the name is not in
    the file at all, because that is a caller asking for something this
    format never had: *"does not apply"* and *"should be here and is not"*
    must never read the same. Law A, applied to a lookup.
    """
    _check_engine(t, engine)
    for it in t.items:
        if it.name != name:
            continue
        if engine and it.engines and engine not in it.engines:
            return None                      # declared, but not for this engine
        return it
    raise KeyError(
        f"no item {name!r} in this template (engines={list(t.engines)}). If the "
        f"parameter exists for another engine it would still be listed, so "
        f"this is a name this template never carried -- not an item that "
        f"does not apply here (engines/template.md § 8.0).")


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
    and the per-field checks are what name the ones that moved.
    """
    known = template_fields(config_cls)
    # § 6.3: ONE file carries every engine the calculation can run on, so the
    # items are filtered to this engine's BEFORE they are read as its fields.
    # Without this, a catalogue serving SIESTA and PySCF is refused for both --
    # each sees the other's items as names its schema does not declare, which
    # is the very thing the `engines` axis exists to prevent.  `select` refuses
    # an engine the file does not serve, so *"does not run on that engine"*
    # stays distinct from *"no items matched"*.
    parsed = read_template(text)
    eng = _engine_name(config_cls)
    mine = select(parsed, engine=eng) if parsed.engines else parsed.items
    vals = {it.name: it.value for it in mine if it.is_set}
    machine = sorted(k for k in vals
                     if k not in known
                     and any(f.name == k
                             for f in dataclasses.fields(config_cls)))
    if machine:
        # The write side emits these VALUELESS (since @2 ``declaration_for``
        # returns an item, not None -- § 6.4), so a VALUE on one is a hand
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


#: The module's public surface.  It omitted `select` and `one` -- which § 8.0
#: calls THE ONE READ API -- and the three closed vocabularies until
#: 2026-08-14, so a surface wanting to order panels by the closed six had to
#: hard-code them or reach past this line (audit § 1.5).
__all__ = ["SCHEMA", "SUFFIX", "KINDS", "TYPES", "CATEGORIES", "RESOLVERS", "ALLOCATION_RESOLVERS",
           "Item", "Template",
           "declaration_for", "declarations_for",
           "select", "one",
           "CATALOGUE", "load_catalogue", "template_with_values",
           "read_template", "config_from_template",
           "template_fields"]
