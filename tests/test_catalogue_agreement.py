"""The two homes agree — the mechanism that lets the duplication exist.

**The catalogue is the master** (`template.md` § 2.1, § 4.3). A parameter is
defined there, and a config class carries a name, a type and its validators.

**But the live Build form still reads the facts off the dataclass fields** —
:data:`MIRRORED`, which is the one place that set is named (`template.md` § 5.1
used to list it too, and listed five of the six). Until the UI is rebuilt from
the catalogue (deferred — tracked in `archive/2026-09-01-roadmap.md` workstream 3; the debt is
measured at `template.md` § 2.1a), those facts live
in **two** homes.

**The size of the debt is stated in `template.md` § 2.1a and asserted by
`tests/test_doc_claims.py`**, not typed here: it was 307 on 2026-08-14 and 452
on 2026-08-17, and a number in a docstring that nothing measures is how the
first figure survived three days past its truth. The growth is the argument —
the debt compounds with every parameter added, which is what makes deleting it
worth scheduling rather than admiring.

That breaks D3 (*each value is stored once*), and D3's own reasoning says what
happens next: *"then a hand edit of one is silently ignored — the file
disagreeing with itself."*

**So this file is the interim contract.** It cannot make the duplication go
away; it makes it impossible for the two to drift apart without a red test
naming the item and the key. When the form moves onto the catalogue, the
metadata is deleted and this file goes with it.
"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder import template as T
from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig

ENGINES = [("siesta", SiestaConfig), ("pyscf", PySCFConfig)]

#: The facts the catalogue owns that a dataclass field also spells today.
#: ``category`` is compared as a set — the catalogue writes a list and the
#: metadata a tuple, and the ORDER is meaningful (first = the panel), so it is
#: compared as a sequence rather than a set.
MIRRORED = ("help", "range", "unit", "choices", "label", "engine_key")

#: Facts the catalogue spells one way and a dataclass field spells another.
#: ``group`` is the CARD, and the two homes are read by different consumers:
#: the FORM takes it from the catalogue, while finding-placement takes it from
#: the class (``_shared.resolve_workflow_group``). A disagreement puts a
#: control on one card and its warnings on another — which is exactly the
#: state twenty-three fields were in on 2026-08-15, when the panels were
#: filled in on the catalogue side only.
RENAMED = {"group": "workflow_group"}


def _catalogue():
    return {i.name: i for i in T.read_template(T.load_catalogue()).items}


@pytest.mark.parametrize("engine,cls", ENGINES, ids=lambda x: getattr(x, "__name__", x))
def test_every_mirrored_fact_agrees(engine, cls):
    """The guard.  A drift here means a surface and a deck disagree about the
    same parameter — one showing bounds the other does not honour."""
    cat = _catalogue()
    bad = []
    for f in dataclasses.fields(cls):
        item = cat.get(f.name)
        if item is None:
            continue                      # not a catalogue item; § 7 exclusions
        for key in MIRRORED + tuple(RENAMED):
            meta = f.metadata.get(RENAMED.get(key, key))
            if meta is None:
                continue                  # the field says nothing; nothing to disagree with
            mine = getattr(item, key)
            if key in ("range", "choices") and mine is not None:
                meta, mine = tuple(meta), tuple(mine)
            if key == "help":
                # Compared with whitespace NORMALISED.  The catalogue's prose
                # carries meaningful line structure (`template.md` § 4.0a:
                # text whose breaks are load-bearing, never markup); the class's copy is a mirror wrapped to fit Python
                # source, so the two legitimately differ in line breaks.  What
                # this guard is for is a different FACT -- a stale sentence,
                # other bounds, another engine's wording -- and normalising
                # keeps it pointed at that instead of at re-wrapping.
                meta, mine = " ".join(meta.split()), " ".join(mine.split())
            if meta != mine:
                bad.append(f"{f.name}.{key}: class={meta!r} catalogue={mine!r}")
    assert not bad, (
        f"{cls.__name__} and the catalogue disagree about "
        f"{len(bad)} fact(s):\n  " + "\n  ".join(bad) +
        "\n\nThe CATALOGUE is the master (template.md § 2.1).  Fix the class, "
        "or -- if the catalogue is the one that is wrong -- fix the catalogue "
        "and say so.  They are two homes for one fact until the form is "
        "rebuilt from the catalogue, and this test is the only thing keeping "
        "them in step.")


@pytest.mark.parametrize("engine,cls", ENGINES, ids=lambda x: getattr(x, "__name__", x))
def test_the_category_agrees_in_ORDER_not_only_in_membership(engine, cls):
    """`category` is a LIST and the first entry is the panel the item appears
    on (§ 6.2).  Two homes agreeing on the set but not the order would put a
    parameter on a different panel depending on which one a surface read."""
    cat = _catalogue()
    bad = []
    for f in dataclasses.fields(cls):
        item = cat.get(f.name)
        if item is None or not f.metadata.get("category"):
            continue
        if tuple(f.metadata["category"]) != tuple(item.category):
            bad.append(f"{f.name}: class={tuple(f.metadata['category'])} "
                       f"catalogue={tuple(item.category)}")
    assert not bad, "category order disagrees:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("engine,cls", ENGINES, ids=lambda x: getattr(x, "__name__", x))
def test_every_field_the_translator_needs_is_in_the_catalogue(engine, cls):
    """The direction that actually matters: **template → config**.

    A config field with no catalogue item is a parameter the translator can
    never be given a value for — the calculation cannot express it.  The
    exclusions are § 7's own: a machine fact carries no value but IS an item,
    and a stage ladder is not a parameter at all.
    """
    cat = _catalogue()
    hints = __import__("typing").get_type_hints(cls)
    missing = []
    for f in dataclasses.fields(cls):
        ann = hints[f.name]
        args = __import__("typing").get_args(ann)
        is_ladder = (__import__("typing").get_origin(ann) in (list, tuple)
                     and args and dataclasses.is_dataclass(args[0]))
        if is_ladder:
            continue
        if f.name not in cat:
            missing.append(f.name)
    assert not missing, (
        f"{cls.__name__} has field(s) with no catalogue item: {missing}.  "
        f"Membership is TOTAL (template.md § 7) -- a parameter the catalogue "
        f"does not carry is one no surface can offer and no calculation can "
        f"record.")


def test_the_catalogue_carries_no_item_no_engine_can_hold():
    """The reverse: an item nothing can translate is a dead entry.

    It would show on a panel, take a value, and be dropped on the way to the
    engine -- the quietest way to lose a setting a person believes they set.
    """
    fields = {e: {f.name for f in dataclasses.fields(c)} for e, c in ENGINES}
    orphans = []
    for item in T.read_template(T.load_catalogue()).items:
        engs = item.engines or tuple(fields)
        if not any(item.name in fields[e] for e in engs if e in fields):
            orphans.append(f"{item.name} (engines={list(engs)})")
    assert not orphans, (
        "catalogue items no config class can carry:\n  " + "\n  ".join(orphans))


def test_every_catalogue_item_declares_a_panel():
    """A parameter the catalogue carries is a parameter a surface must be able
    to PLACE.

    ``group`` is optional on a template item — it is presentation, and ``prep``
    reading a template headlessly never asks. It is **not** optional here: the
    catalogue is what a form is built from (`web/form-schema.md` § 1), so an
    item with no group renders loose beneath the cards and every finding about
    it falls to the residual panel instead of sitting beside the field.

    That is not hypothetical. Fifteen items were in exactly that state until
    2026-08-15 — inherited when the catalogue was extracted from the config
    classes, where the old form's opt-in ``section`` tag meant a field nothing
    rendered also carried no group. The new form renders every item, so the
    hole became visible all at once.
    """
    missing = sorted(i.name for i in T.read_template(T.load_catalogue()).items
                     if not i.group)
    assert not missing, (
        f"catalogue item(s) with no panel: {missing}.\n"
        f"Give each a `group` from {T.GROUPS}. An item with none renders "
        f"below the cards and its warnings land in the residual panel.")


def test_the_group_vocabulary_is_closed_and_the_catalogue_stays_inside_it():
    """The typo guard. A misspelt group is indistinguishable from an absent
    one on the page — the field renders loose either way — so the refusal has
    to come from the reader, not from a person noticing."""
    used = {i.group for i in T.read_template(T.load_catalogue()).items}
    assert used <= set(T.GROUPS), f"unknown group(s): {used - set(T.GROUPS)}"


@pytest.mark.parametrize("engine,cls", ENGINES, ids=lambda x: getattr(x, "__name__", x))
def test_the_declared_TYPE_agrees_with_the_annotation(engine, cls):
    """The fact the mirrored-key guard could not see.

    ``type`` is not metadata — it is derived from the annotation — so it is not
    in :data:`MIRRORED`.  But the ⊕ operator asks the catalogue *"is this a
    float?"* and the emitters act on the annotation, so a disagreement means
    one of them coerces a value the other does not.

    That is not hypothetical: until 2026-08-14 the operator string-matched the
    annotation, and ``Optional[float]`` is not ``"float"``, so ``spin_total``
    and ``md_target_temperature`` were silently never widened (audit § 25.3).
    """
    import typing
    from molbuilder import template as _T
    cat = _catalogue()
    hints = typing.get_type_hints(cls)
    bad = []
    for f in dataclasses.fields(cls):
        item = cat.get(f.name)
        if item is None:
            continue
        inner, _opt = _T._unwrap_optional(hints[f.name])
        expected = {float: "float", int: "int", str: "str", bool: "bool"}.get(inner)
        if expected is None:
            continue                      # tuples, lists, enums — typed by rule
        if item.choices:
            continue                      # an enum is typed by its choices
        # A declared type may REFINE the annotation: `pow2` is an int that must
        # be a power of two, and `text` is a str copied verbatim.  § 5's whole
        # purpose is to carry what the annotation cannot, so a refinement is
        # agreement, not drift.
        REFINES = {"int": {"pow2"}, "str": {"text"}}
        if item.type in REFINES.get(expected, ()):
            continue
        if item.type != expected:
            bad.append(f"{f.name}: annotation={inner.__name__} "
                       f"catalogue={item.type}")
    assert not bad, (
        "the catalogue and the annotation disagree about a value's TYPE:\n  "
        + "\n  ".join(bad) +
        "\nOne of them coerces where the other does not.")
