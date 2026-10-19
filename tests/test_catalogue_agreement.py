"""The one fact two homes still state agrees — a field's default.

**The catalogue is the master** (`template.md` § 2.1, § 4.3). A parameter is
defined there, and a config class carries a name, a type, a default and its
validators -- nothing else since 2.15.4 (§ 2.1a, the debt paid 2026-10-18):
every other fact is read through `template.items_of`.

Beside the default: category ORDER, declared TYPE, orphan items, and panel presence.

"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder import template as T
from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig

ENGINES = [("siesta", SiestaConfig), ("pyscf", PySCFConfig)]



def _catalogue():
    return {i.name: i for i in T.read_template(T.load_catalogue()).items}




def _same(a, b) -> bool:
    """Compare two spellings of one fact.  A list and a tuple of the same
    values are the same fact: TOML gives the catalogue a list where the field
    metadata is written as a tuple, and that difference is the file format's,
    not the parameter's."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return tuple(a) == tuple(b)
    return a == b


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
    """``type`` is derived from the field's annotation, not declared on
    it.  The ⊕ operator asks the catalogue *"is this a
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
