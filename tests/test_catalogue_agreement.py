"""The two homes agree — the mechanism that lets the duplication exist.

**The catalogue is the master** (`template.md` § 2.1, § 4.3). A parameter is
defined there, and a config class carries a name, a type and its validators.

**But the live Build form still reads the facts off the dataclass fields** —
:data:`MIRRORED`, which is the one place that set is named. Until the UI is
rebuilt from the catalogue (the debt is measured at `template.md` § 2.1a),
those facts live in **two** homes.

**The size of the debt is stated in `template.md` § 2.1a and asserted by
`tests/test_doc_claims.py`**, not typed here: a number in a docstring that
nothing measures goes stale. The debt compounds with every parameter added, which is what makes deleting it
worth scheduling rather than admiring.

That breaks D3 (*each value is stored once*), and D3's own reasoning says what
happens next: *"then a hand edit of one is silently ignored — the file
disagreeing with itself."*

**So this file is the interim contract**, and drift in the mirrored facts
goes red naming the item and the key -- :func:`test_every_mirrored_fact_agrees`.

Beside it: category ORDER, declared TYPE, orphan items, and panel presence.

**A BLANKET agreement check would still be wrong**, and this one is
deliberately not that. `restart` declares `kind = deck` in the shared catalogue
row and `produce` in the PySCF dataclass, correctly, because it expands three
keywords on SIESTA and none on PySCF; of 109 shared items that is the only
`kind` disagreement, and no document states the exception. `kind` is not in
:data:`MIRRORED`. The facts that ARE there have no legitimate exception --
measured 2026-09-17, zero disagreements across both engines -- so the check
needs to know about nothing.

When the form moves onto the catalogue, the metadata is deleted and this file
goes with it.
"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder import template as T
from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig

ENGINES = [("siesta", SiestaConfig), ("pyscf", PySCFConfig)]

#: The facts the catalogue owns that a dataclass field also spells today.
#: ``category`` is compared in its own test, as a sequence: the ORDER is
#: meaningful (first = the panel).
#: ``help`` is not here: every surface asks ``template.help_for``, so it is
#: not a fact with two homes.
MIRRORED = ("range", "unit", "choices", "label", "engine_key")

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


def _same(a, b) -> bool:
    """Compare two spellings of one fact.  A list and a tuple of the same
    values are the same fact: TOML gives the catalogue a list where the field
    metadata is written as a tuple, and that difference is the file format's,
    not the parameter's."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return tuple(a) == tuple(b)
    return a == b


def _declared(v) -> bool:
    """Is this side carrying a value at all?  Absent is not a disagreement."""
    return v not in (None, "", (), [])


@pytest.mark.parametrize("engine,cls", ENGINES, ids=lambda x: getattr(x, "__name__", x))
def test_every_mirrored_fact_agrees(engine, cls):
    """THE GUARANTEE THREE SECTIONS OF THE CONTRACT PROMISE.

    `template.md` § 6.0 says editing one home and not the other is *"caught,
    loudly, by name -- which is the only reason the duplication is
    survivable"*, and § 5.1 says this file *"keeps the two in step"*.

    **Why this exists** (the admission question, and the answer has to be a
    failure nothing else catches): a surface reads these facts off
    whichever home it came through.  `range` disagreeing means a form accepts
    a value the other door then refuses; `choices` disagreeing means a valid
    setting is rejected depending on the tab; `engine_key` disagreeing means
    the deck gets a different keyword.  And `workflow_group` is measured, not
    hypothetical -- § 2.1a records twenty-three fields whose control sat on
    one card while their findings landed on another, because the panels were
    filled in on the catalogue side only.  No other test compares these keys.

    **Why SCOPED and not blanket.**  This file's header argues a blanket
    agreement check would be wrong, and it is right: `restart` declares
    `kind = deck` in the shared catalogue row and `produce` on the PySCF
    dataclass, correctly, because it expands three keywords on SIESTA and
    none on PySCF -- and no document states that exception.  `kind` is not in
    :data:`MIRRORED`.  This compares the facts the contract names, where
    there is no legitimate exception and, measured today, no disagreement.

    **One-sided is NOT a failure.**  Five PySCF items carry a catalogue label
    and no class copy, which is the debt being PAID -- one home, the direction
    `help` went.  A
    test that demanded both sides carry a copy would push the count up and
    call it progress.
    """
    cat = _catalogue()
    bad = []
    for f in dataclasses.fields(cls):
        item = cat.get(f.name)
        if item is None:
            continue
        for ckey in MIRRORED + tuple(RENAMED):
            fkey = RENAMED.get(ckey, ckey)
            cv, fv = getattr(item, ckey, None), f.metadata.get(fkey)
            if not (_declared(cv) and _declared(fv)):
                continue
            if not _same(cv, fv):
                bad.append(f"{f.name}.{ckey}: catalogue={cv!r} class={fv!r}"
                           + (f"  (class spells it {fkey!r})"
                              if fkey != ckey else ""))
    assert not bad, (
        f"the two homes disagree for {engine}, by item and key:\n  "
        + "\n  ".join(bad)
        + "\n\nThe CATALOGUE is the master (`template.md` § 2.1): fix the "
          "dataclass metadata unless the catalogue row is the wrong one.")




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


@pytest.mark.parametrize("engine,cls", ENGINES, ids=lambda x: getattr(x, "__name__", x))
def test_the_DEFAULT_agrees(engine, cls):
    """The default has two homes as well, read by two roads: the catalogue's
    is what a form offers, and the class's is what a new transport template
    is written with (`transport_template_text` builds it from a config).  Two
    homes that disagree give a person a different starting value by the road
    that described the calculation.  Measured 2026-09-29, zero disagreements
    across both engines -- the day M5 step 2 changed six defaults in both
    places (plan W48 records the facts still unguarded)."""
    cat = _catalogue()
    bad = []
    for f in dataclasses.fields(cls):
        item = cat.get(f.name)
        if item is None or item.default is None:
            continue
        if f.default is not dataclasses.MISSING:
            mine = f.default
        elif f.default_factory is not dataclasses.MISSING:
            mine = f.default_factory()
        else:
            continue
        if not _same(mine, item.default):
            bad.append(f"{f.name}: class={mine!r} catalogue={item.default!r}")
    assert not bad, "the default disagrees:\n  " + "\n  ".join(bad)
