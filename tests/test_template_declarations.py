"""The schema behind a template: what every parameter must declare, and the
fingerprint of the shape they were written against.

**Every test here is derived from a live contract**, and the ones that were not
were removed on 2026-08-11 rather than updated (see the retirement note at the
foot of this file). The rule that decided each: *a guard asserts what the
contract says. It must never assert what the contract says should NOT be true* —
a test pinning a retired format makes replacing it harder and reads to the next
person as policy.

Contracts:

* ``docs/engines/template.md`` — what a template **is**. § 3 (the required keys,
  and *a missing* ``value`` *means explicitly unset*), § 5 (the `type`
  vocabulary and where every key comes from), § 7 (membership is **total**, and
  the three things that are not items), § 10 (complete, lossless, and the
  fingerprint).
* ``docs/execution/job-contracts.md`` § 3.1 (the reserved blocks of a generated
  script, which the shared marker finds) · § 3.3 (BENCH-MARKS, whose
  declarations come from the **same** field metadata — so the two cannot drift).
* ``docs/engines/stages.md`` § 6.6 — the preflight row *"the schema fingerprint
  matches"*, the only row there that reports rather than refuses.

**What this file does NOT guard, deliberately.** The template's *file format* is
one TOML file (``template.md`` § 4), and nothing here asserts its serialization:
``molbuilder/template.py`` still emits the retired ``.fdf`` shape, and the
replacement is the plan's **P12 unit 6b**. Tests for the TOML writer are
designed from ``template.md`` when that lands — not patched out of these.

P2 unit 4a.
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


# SIESTA only (U16, 2026-08-12): membership went TOTAL (§ 7 -- the
# section gate was a fourth, unlisted exclusion), and under the total rule
# declarations_for(PySCFConfig) refuses LOUDLY on that schema's known
# vocabulary gaps -- which is § 7 working, pinned by name in
# test_pyscfs_vocabulary_gaps_refuse_loudly below.  PySCF rejoins this
# list when its template lands (it has no producer today; describe is
# SIESTA-only) and the gaps are modelled.
ENGINES = [SiestaConfig]


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
    # § 7 (U16): membership is TOTAL -- every schema field minus the
    # named exclusions (machine facts, the ladder), never a section
    # subset.  ``section`` answers only *where on the form*.
    members = {f.name for f in dataclasses.fields(cls)
               if not f.metadata.get("allocation")}
    declared = set(_decls(cls))
    ladders = {f.name for f in dataclasses.fields(cls)
               if _is_ladder(cls, f)}
    assert declared == members - ladders, sorted(
        (members - ladders) ^ declared)


def _is_ladder(cls, f) -> bool:
    ann = typing.get_type_hints(cls)[f.name]
    args = typing.get_args(ann)
    return (typing.get_origin(ann) in (list, tuple)
            and bool(args) and dataclasses.is_dataclass(args[0]))


def test_a_stage_ladder_is_not_a_template_item():
    """`stages.md § 1.1`: a ladder is the user's decision about what varies
    and lives in ``task.json``.  PySCF is the only config that still has one,
    and it is excluded for WHAT IT IS rather than by falling through to the
    unnameable-type error, which would report a vocabulary gap where there is
    none."""
    # (via declaration_for directly: PySCF's whole-schema walk refuses on
    # its § 7 vocabulary gaps -- pinned by name below -- but the ladder
    # exclusion is checkable per-field.)
    from molbuilder.template import declaration_for
    hints = typing.get_type_hints(PySCFConfig)
    f = next(f for f in dataclasses.fields(PySCFConfig)
             if f.name == "stages")
    assert declaration_for(f, hints["stages"]) is None


# --------------------------------------------------------------------- #
#  The declaration grammar                                              #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_every_declaration_has_a_named_type(cls):
    """Every item's type is in the TEMPLATE grammar; the subset that
    script_emit renders into BENCH-MARKS (engine-kind, anchored) must
    also be in ITS vocabulary -- the two share field metadata so they
    cannot drift, but they are not the same set: a produce-kind item
    (species_order's strlist) never reaches a BENCH-MARKS line."""
    from molbuilder.template import TYPES
    for d in declarations_for(cls):
        assert d.type in TYPES, f"{d.name}: {d.type}"
        if d.kind == "engine" and d.anchor:
            assert d.type in DECL_TYPES, f"{d.name}: {d.type}"


@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_an_enum_declaration_carries_its_members(cls):
    """§ 3.7 adds ``choices=`` precisely so a surface can build the dropdown
    and a reader can validate what was typed.  An enum without them declares
    a constraint nobody can check."""
    for d in declarations_for(cls):
        if d.type == "enum":
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
    assert d["spin_polarized"].type == "bool"
    assert d["relax_steps"].type == "int"


def test_the_kgrid_is_one_declaration_not_three():
    """It is one decision — how finely reciprocal space is sampled — and a
    stage overriding it overrides all three components together."""
    assert _decls(SiestaConfig)["kgrid"].type == "int3"


def test_range_unit_and_group_come_from_the_field_metadata():
    """§ 3.7's table: the declaration is what makes the template enough on
    its own, so a surface holding the file needs nothing else to bound the
    control, label it, and decide whether its *vary per stage* box starts
    ticked."""
    d = _decls(SiestaConfig)["mesh_cutoff"]
    assert d.range == (100.0, 1000.0)
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


def test_pyscfs_vocabulary_gaps_refuse_loudly():
    """§ 7: *"a parameter that cannot be given a kind is a gap in this
    vocabulary, and the loud version of that is the only one that gets
    fixed."*  PySCF has no template producer yet; what the total rule
    owes it TODAY is that the gaps are named, not skipped.  This list
    shrinking is progress; it growing is a new unclassified field."""
    import typing as _t
    hints = _t.get_type_hints(PySCFConfig)
    gaps = []
    for f in dataclasses.fields(PySCFConfig):
        try:
            from molbuilder.template import declaration_for
            declaration_for(f, hints[f.name])
        except ValueError:
            gaps.append(f.name)
    assert gaps == ["ecp", "save_optimized_xyz", "save_initial_xyz",
                    "write_trajectory", "write_molwatch_log", "stage"]


def test_declarations_keep_the_configs_own_order():
    """The config's field order is the form's order and the deck's order; a
    template a person reads should not be a third arrangement of them."""
    names = [d.name for d in declarations_for(SiestaConfig)]
    expected = [f.name for f in dataclasses.fields(SiestaConfig)
                if not f.metadata.get("allocation")]
    assert names == expected


# --------------------------------------------------------------------- #
#  § 6.6 — the schema fingerprint                                       #
# --------------------------------------------------------------------- #

def test_the_fingerprint_is_stable_and_short():
    a, b = schema_fingerprint(SiestaConfig), schema_fingerprint(SiestaConfig)
    assert a == b and len(a) == 16


@pytest.mark.skip(reason="PySCF's fingerprint cannot compute until its "
                  "§ 7 vocabulary gaps close (see test_pyscfs_vocabulary_"
                  "gaps_refuse_loudly); re-enable with its template")
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
#  The reserved script blocks — job-contracts.md § 3.1                  #
# --------------------------------------------------------------------- #

def test_the_reserved_block_markers_match_the_documented_set():
    """`job-contracts.md` § 3.1 names the reserved blocks of a **generated
    script**, and every one of them must be findable by the shared marker.

    These are live: a deck still carries PROVENANCE, BENCH-MARKS,
    ATOM-METADATA and USER-CUSTOM, and HEADER is reserved-but-unemitted.
    """
    for name in ("header", "provenance", "bench-marks",
                 "atom-metadata", "user-custom"):
        m = MARKER_RE.match(f"# === molbuilder {name} BEGIN ===")
        assert m and m.group(1) == name, name


@pytest.mark.parametrize("line", [
    "MeshCutoff 300.0 Ry",
    "# Just a comment",
    "# === something else BEGIN ===",
    "# === molbuilder ===",
])
def test_the_marker_rejects_what_is_not_a_block_marker(line):
    """A block's payload is copied verbatim, so a payload line that matched
    the marker would silently truncate the block."""
    assert MARKER_RE.match(line) is None


# --------------------------------------------------------------------- #
#  RETIRED 2026-08-11 — the item-block template format                  #
# --------------------------------------------------------------------- #
#
#  Three tests stood here and were deleted rather than updated:
#
#    test_the_marker_accepts_an_item_block_naming_its_field
#    test_the_marker_still_rejects_what_it_rejected   (the `item one two
#        three` case only — the rest survives above)
#    test_an_internal_field_gets_no_declaration
#
#  They asserted that the marker admits `# === molbuilder item <field> ===`
#  and that a field without a `section` gets no declaration.  BOTH ARE NOW
#  THINGS THE CONTRACT SAYS MUST **NOT** BE TRUE:
#
#    * `engines/template.md` D2/D3 retires the `.fdf`-with-item-blocks
#      template outright (archive/2026-08-11-template-item-blocks.md).  A
#      template is ONE TOML FILE; there is no item block to mark, and the
#      value is stored once so it cannot disagree with a payload line.
#    * membership is TOTAL and reads `kind`, not `section` (§ 7, and the
#      plan's P12 unit 6b).  `species_order`, `write_forces`,
#      `write_coor_step` and `write_molwatch_log` are items precisely
#      BECAUSE `section` went back to answering only *where on the form*.
#      The retired test asserted the opposite and would fail the correct
#      implementation.
#
#  Kept as a comment rather than deleted silently: a guard removed with no
#  record reads later as a guard nobody wrote.  `molbuilder/template.py`
#  still implements the old format, and P12 unit 6b replaces it — until then
#  that code is simply unguarded here, which is the honest state.  A test
#  that pins a format the contract rejects makes the replacement harder and
#  states policy that is not policy.
