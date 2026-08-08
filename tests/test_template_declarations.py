"""The template's declarations, and the fingerprint of the schema they describe.

Contract: ``docs/execution/job-contracts.md`` § 3.7 (the item block's ``field``
declaration line — *"the grammar § 3.3 already defines … extended with
``group=`` and ``choices=``. Not a parallel notation"*) and § 3.3 (that
grammar) · ``docs/engines/stages.md`` § 6.6 (the preflight row *"the schema
fingerprint matches"*, the only one that does not refuse).

P2 unit 4a. These are the two halves that do **not** depend on the two open
questions § 3.7 leaves — which anchor a multi-anchor field declares, and what a
conditionally-emitted item's payload is when the deck would have no line. A
declaration and a fingerprint are the same under every answer to those, so they
are built now and the emitter waits.
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import field as dc_field

import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.script_emit import DECL_TYPES, MARKER_RE
from molbuilder.template import (
    declarations_for,
    fingerprint_matches,
    schema_fingerprint,
)


ENGINES = [SiestaConfig, PySCFConfig]


def _decls(cls):
    return {d.name: d for d in declarations_for(cls)}


# --------------------------------------------------------------------- #
#  § 3.7 property 4 — every allowed item has a place in the file        #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_every_exposed_field_gets_a_declaration(cls):
    """The template's premise, asserted rather than hoped for: it is the
    engine's whole surface instantiated, not the subset somebody typed.

    A field the form shows and the template omits would be a setting a user
    can change and the calculation cannot record."""
    exposed = {f.name for f in dataclasses.fields(cls)
               if f.metadata.get("section")}
    declared = set(_decls(cls))
    # The one legitimate subtraction: a stage ladder is not a template item.
    ladders = {f.name for f in dataclasses.fields(cls)
               if _is_ladder(cls, f)}
    assert declared == exposed - ladders, sorted(exposed - ladders - declared)


def _is_ladder(cls, f) -> bool:
    ann = typing.get_type_hints(cls)[f.name]
    args = typing.get_args(ann)
    return (typing.get_origin(ann) in (list, tuple)
            and bool(args) and dataclasses.is_dataclass(args[0]))


@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_an_internal_field_gets_no_declaration(cls):
    """A field with no ``section`` is internal (`web/form-schema.md § 1a`) —
    no surface renders it, so a template listing it would offer the user
    something no tab can show."""
    internal = {f.name for f in dataclasses.fields(cls)
                if not f.metadata.get("section")}
    assert internal, "this test is vacuous if the config has no internal fields"
    assert not (internal & set(_decls(cls)))


def test_a_stage_ladder_is_not_a_template_item():
    """`stages.md § 1.1`: a ladder is the user's decision about what varies
    and lives in ``task.json``.  PySCF is the only config that still has one,
    and it is excluded for WHAT IT IS rather than by falling through to the
    unnameable-type error, which would report a vocabulary gap where there is
    none."""
    assert "stages" in {f.name for f in dataclasses.fields(PySCFConfig)}
    assert "stages" not in _decls(PySCFConfig)


# --------------------------------------------------------------------- #
#  The declaration grammar                                              #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_every_declaration_has_a_named_type(cls):
    for d in declarations_for(cls):
        assert d.type_ in DECL_TYPES, f"{d.name}: {d.type_}"


@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_an_enum_declaration_carries_its_members(cls):
    """§ 3.7 adds ``choices=`` precisely so a surface can build the dropdown
    and a reader can validate what was typed.  An enum without them declares
    a constraint nobody can check."""
    for d in declarations_for(cls):
        if d.type_ == "enum":
            assert d.choices, d.name


@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_optional_is_set_for_exactly_the_optional_fields(cls):
    """*Unset* is a real state and distinct from every value the field could
    hold — for these fields the engine gets no line at all, which is not the
    same as getting the default."""
    hints = typing.get_type_hints(cls)
    for d in declarations_for(cls):
        ann = hints[d.name]
        is_opt = (typing.get_origin(ann) is typing.Union
                  and type(None) in typing.get_args(ann))
        assert d.optional is is_opt, d.name


def test_a_bool_is_typed_bool_and_not_int():
    """``bool`` is a subclass of ``int`` in Python, so a dict lookup in the
    wrong order types every checkbox as an integer — and a surface would draw
    seven number boxes where the form draws seven checkboxes."""
    d = _decls(SiestaConfig)
    assert d["spin_polarized"].type_ == "bool"
    assert d["relax_steps"].type_ == "int"


def test_the_kgrid_is_one_declaration_not_three():
    """It is one decision — how finely reciprocal space is sampled — and a
    stage overriding it overrides all three components together."""
    assert _decls(SiestaConfig)["kgrid"].type_ == "int3"


def test_range_unit_and_group_come_from_the_field_metadata():
    """§ 3.7's table: the declaration is what makes the template enough on
    its own, so a surface holding the file needs nothing else to bound the
    control, label it, and decide whether its *vary per stage* box starts
    ticked."""
    d = _decls(SiestaConfig)["mesh_cutoff"]
    assert d.range_ == (100.0, 1000.0)
    assert d.unit == "Ry"
    assert d.group == "stage"


def test_an_unnameable_type_is_refused_by_name():
    """A gap in the type vocabulary is loud, because the quiet version is a
    field silently missing from the template — and § 3.7's premise is that
    every allowed item has a place in it."""
    odd = dataclasses.make_dataclass("Odd", [
        ("weird", complex, dc_field(default=0j,
                                    metadata={"section": "S",
                                              "workflow_group": "budget"}))])
    with pytest.raises(ValueError, match="weird"):
        declarations_for(odd)


def test_declarations_keep_the_configs_own_order():
    """The config's field order is the form's order and the deck's order; a
    template a person reads should not be a third arrangement of them."""
    names = [d.name for d in declarations_for(SiestaConfig)]
    expected = [f.name for f in dataclasses.fields(SiestaConfig)
                if f.metadata.get("section")]
    assert names == expected


# --------------------------------------------------------------------- #
#  § 6.6 — the schema fingerprint                                       #
# --------------------------------------------------------------------- #

def test_the_fingerprint_is_stable_and_short():
    a, b = schema_fingerprint(SiestaConfig), schema_fingerprint(SiestaConfig)
    assert a == b and len(a) == 16


def test_two_engines_do_not_share_a_fingerprint():
    assert schema_fingerprint(SiestaConfig) != schema_fingerprint(PySCFConfig)


def _variant(*, meta=None, ann=int, default=3):
    """A tiny config whose schema can be perturbed one axis at a time.

    Built with ``make_dataclass`` rather than a class body: this module has
    ``from __future__ import annotations``, so a class-body annotation is
    stored as a *string* and a computed one ("whatever ``ann`` holds") cannot
    be resolved back to a type at all."""
    md = {"section": "S", "workflow_group": "stage",
          "range": (1, 10), "help": "a", "label": "A", "unit": "Ry"}
    md.update(meta or {})
    return dataclasses.make_dataclass(
        "C", [("x", ann, dc_field(default=default, metadata=md))])


@pytest.mark.parametrize("what,kw", [
    ("a re-bound",   {"meta": {"range": (1, 20)}}),
    ("a retype",     {"ann": float}),
    ("new choices",  {"meta": {"choices": ("a", "b")}}),
])
def test_the_fingerprint_changes_when_the_shape_changes(what, kw):
    """A description written against the old shape names a field that still
    exists and a value that may no longer be legal — worth saying out loud
    rather than discovering at the engine."""
    assert schema_fingerprint(_variant()) != schema_fingerprint(_variant(**kw))


@pytest.mark.parametrize("what,kw", [
    ("a new default", {"default": 7}),
    ("reworded help", {"meta": {"help": "something else entirely"}}),
    ("a new label",   {"meta": {"label": "Renamed"}}),
])
def test_the_fingerprint_ignores_presentation_and_defaults(what, kw):
    """**The half that matters more.**  A template records the value in use,
    so changing a default cannot invalidate a description that already
    carries values; and a reworded tooltip must not make every stored
    description suspect.  A fingerprint that cried wolf would be turned off,
    and then the row it guards would be worth nothing."""
    assert schema_fingerprint(_variant()) == schema_fingerprint(_variant(**kw))


def test_adding_a_field_changes_the_fingerprint():
    before = schema_fingerprint(_variant())
    md = {"section": "S", "workflow_group": "stage"}
    two = dataclasses.make_dataclass("C2", [
        ("x", int, dc_field(default=3, metadata={**md, "range": (1, 10)})),
        ("y", int, dc_field(default=1, metadata=md)),
    ])
    assert schema_fingerprint(two) != before


def test_an_absent_fingerprint_matches_anything():
    """§ 6.6's one non-refusal row.  A description written by hand, or before
    this existed, is not wrong — it makes no claim."""
    assert fingerprint_matches(SiestaConfig, "")
    assert fingerprint_matches(SiestaConfig, schema_fingerprint(SiestaConfig))
    assert not fingerprint_matches(SiestaConfig, "0000000000000000")


# --------------------------------------------------------------------- #
#  The marker carries the field's name                                  #
# --------------------------------------------------------------------- #

def test_the_marker_accepts_an_item_block_naming_its_field():
    """§ 3.7 property 2, and *"this is the whole reason the design works"*:
    the marker carries the field's name, so ``prep`` rebuilds a config by
    scanning rather than by parsing an ``.fdf`` — which nothing in molbuilder
    can do."""
    m = MARKER_RE.match("# === molbuilder item mesh_cutoff BEGIN ===")
    assert m and m.group(1) == "item mesh_cutoff" and m.group(2) == "BEGIN"
    m = MARKER_RE.match("# === molbuilder item kgrid END ===")
    assert m and m.group(1) == "item kgrid" and m.group(2) == "END"


def test_the_reserved_block_markers_still_match_unchanged():
    for name in ("header", "provenance", "bench-marks", "user-custom"):
        m = MARKER_RE.match(f"# === molbuilder {name} BEGIN ===")
        assert m and m.group(1) == name


@pytest.mark.parametrize("line", [
    "MeshCutoff 300.0 Ry",
    "# Just a comment",
    "# === something else BEGIN ===",
    "# === molbuilder ===",
    "# === molbuilder item one two three BEGIN ===",
])
def test_the_marker_still_rejects_what_it_rejected(line):
    """Widening the name to admit ``item <field>`` must not admit anything
    else — a template's payload is verbatim, so a payload line that matched
    the marker would silently truncate a block."""
    assert MARKER_RE.match(line) is None
