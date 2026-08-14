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
    # @2 (§ 6.4): an allocation field IS a member -- the item is declared,
    # valueless, so a surface can ask for ranks and the wrapper writer knows
    # to look.  § 7's machine-fact row excludes the VALUE, not the item.
    members = {f.name for f in dataclasses.fields(cls)}
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
    """Every item's type is in the TEMPLATE grammar (`template.md` § 5)."""
    from molbuilder.template import TYPES
    for d in declarations_for(cls):
        assert d.type in TYPES, f"{d.name}: {d.type}"


def test_a_bench_marks_field_declares_the_same_type_as_its_template_item():
    """The *one source* rule, checked where it can actually break.

    `job-contracts.md` § 3.3: *"BENCH-MARKS and the template are emitted from
    ONE source, and that is a rule rather than a convenience … two
    hand-maintained copies of ``default=`` would drift, and the drift would be
    silent."*  ``SIESTA_BENCH_FIELDS`` **is** hand-maintained, so the rule is
    an intention there rather than a mechanism -- this test is the mechanism.
    Matched by ANCHOR, which is what a BENCH-MARKS line and a template item
    have in common.

    **Replaces a rule whose premise was false** (2026-08-14).  It read *every*
    engine-kind anchored item must have a type in ``DECL_TYPES``, on the
    reasoning that such an item could reach a BENCH-MARKS line.  It cannot:
    the block declares the five hand-listed fields below and nothing else --
    ``kgrid`` has been engine-kind and anchored all along and appears in no
    block.  The false premise had a cost: it made ``DECL_TYPES`` the gate on
    the TEMPLATE's vocabulary, so ``float3`` (audit § 54) could not be added to
    one without widening the other for a type no benchmark will ever turn.
    """
    from molbuilder.script_emit import SIESTA_BENCH_FIELDS
    # A keyword reaches the deck two ways: an ``engine`` item's ``anchor``, or
    # a ``deck`` item's ``expands`` -- ``MD.NumCGsteps`` is the second, one of
    # the two keywords ``relax_steps`` becomes depending on ``relax_type``.
    # Both are the keyword a BENCH-MARKS anchor greps for.
    by_keyword = {}
    for d in declarations_for(SiestaConfig):
        for kw in ((d.anchor,) if d.anchor else ()) + tuple(d.expands or ()):
            by_keyword.setdefault(kw, d)
    for bf in SIESTA_BENCH_FIELDS:
        item = by_keyword.get(bf.anchor)
        assert item is not None, (
            f"BENCH-MARKS declares {bf.anchor!r}, which no config field "
            f"anchors -- so a tool may override a line the template cannot "
            f"describe (job-contracts.md § 3.3).")
        assert bf.type_ == item.type, (
            f"{bf.anchor}: BENCH-MARKS says type={bf.type_!r}, the template "
            f"item {item.name!r} says {item.type!r}.  These are emitted from "
            f"ONE source by rule; a disagreement means a tool validates an "
            f"override against a type the deck no longer honours.")
        assert bf.type_ in DECL_TYPES, f"{bf.anchor}: {bf.type_}"


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


def test_the_kgrid_displacement_is_its_own_item_and_a_float3():
    """Two items, not one, and not a general ``matrix``.

    The mesh (how finely) and the origin (where it sits) are separate
    scientific decisions and a stage may vary one without the other, so they
    are two items.  Neither is a matrix: `kgridinit.F` accepts a full
    non-diagonal ``kscell``, but that serves supercells commensurate with a
    sub-lattice and nothing in molbuilder builds one -- recorded as *not
    offered* rather than half-offered
    (`docs/audit-2026-08-14-template-execution-review.md` § 53.5).

    ``float3`` is the type that leaves: the components are floats, they are
    independent, and no other member of § 5's vocabulary can carry them.
    """
    from molbuilder.template import TYPES
    assert "float3" in TYPES
    d = _decls(SiestaConfig)["kgrid_displacement"]
    assert d.type == "float3"
    assert d.default == (0.0, 0.0, 0.0)
    assert _decls(SiestaConfig)["kgrid"].type == "int3"   # still separate


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
                                    metadata={"category": ("method",),
                                              "workflow_group": "budget"}))])
    with pytest.raises(ValueError, match="weird"):
        declarations_for(odd)


def test_pyscf_has_no_vocabulary_gaps_left():
    """§ 7's total rule, now SATISFIED for PySCF (T5, 2026-08-13).

    This test used to assert the gaps were NAMED rather than skipped,
    and listed six: ecp, save_optimized_xyz, save_initial_xyz,
    write_trajectory, write_molwatch_log, stage.  Its own docstring said
    *"this list shrinking is progress"*.  It shrank to zero, so the test
    now guards the state that replaced it -- every PySCF field either
    renders or is a machine fact § 7 deliberately excludes, and nothing
    falls through unnamed.

    A field added later without a category, an item_kind, or a type the
    grammar knows fails HERE rather than the first time somebody tries
    to describe a PySCF calculation.
    """
    import typing as _t
    from molbuilder.template import declaration_for
    hints = _t.get_type_hints(PySCFConfig)
    gaps = []
    for f in dataclasses.fields(PySCFConfig):
        try:
            declaration_for(f, hints[f.name])
        except ValueError as exc:
            gaps.append(f"{f.name}: {exc}")
    assert gaps == [], (
        "PySCF fields cannot be placed in the template vocabulary:\n  "
        + "\n  ".join(gaps))


def test_pyscf_renders_a_template_at_all():
    """The thing T5 existed to make true.  Three of four engines could
    not produce a template; a plan that called the template the single
    source of truth had no source for PySCF at all."""
    from molbuilder.template import template_with_values, read_template
    t = read_template(template_with_values(PySCFConfig(), engine="pyscf"))
    assert len(t.items) > 30
    assert all(i.category for i in t.items)


def test_declarations_keep_the_configs_own_order():
    """The config's field order is the form's order and the deck's order; a
    template a person reads should not be a third arrangement of them."""
    names = [d.name for d in declarations_for(SiestaConfig)]
    expected = [f.name for f in dataclasses.fields(SiestaConfig)]
    assert names == expected


# --------------------------------------------------------------------- #
#  § 6.6 — the schema fingerprint                                       #
# --------------------------------------------------------------------- #

def test_the_fingerprint_is_stable_and_short():
    a, b = schema_fingerprint(SiestaConfig), schema_fingerprint(SiestaConfig)
    assert a == b and len(a) == 16


def test_two_engines_do_not_share_a_fingerprint():
    # Un-skipped 2026-08-13 (T5): it was waiting on PySCF's § 7 vocabulary
    # gaps closing, and they have.  A skip that outlives its reason is a
    # test nobody is running and everybody assumes passes.
    assert schema_fingerprint(SiestaConfig) != schema_fingerprint(PySCFConfig)


def _variant(*, meta=None, ann=int, default=3):
    """A tiny config whose schema can be perturbed one axis at a time.

    Built with ``make_dataclass`` rather than a class body: this module has
    ``from __future__ import annotations``, so a class-body annotation is
    stored as a *string* and a computed one ("whatever ``ann`` holds") cannot
    be resolved back to a type at all."""
    md = {"category": ("method",), "workflow_group": "stage",
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
    md = {"category": ("method",), "workflow_group": "stage"}
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
#  read_by — every deck keyword the wrapper reads is declared            #
# --------------------------------------------------------------------- #

def _wrapper_deck_scanners():
    """Every ``runwrap._fdf_requests_*`` — the wrapper's reads of the deck."""
    from molbuilder import runwrap
    return sorted(
        (n, getattr(runwrap, n)) for n in dir(runwrap)
        if n.startswith("_fdf_requests_")
    )


def _trigger_values(f):
    """Deck values worth trying for a field: every choice, or bool truth."""
    choices = list(f.metadata.get("choices") or ())
    return choices or [".true."]


def test_every_deck_keyword_the_wrapper_reads_is_declared_read_by(tmp_path):
    """`template.md` § 6.1: ``read_by`` says **who else derives something from
    the value**, so that the wrapper is *told* which items it depends on
    instead of knowing the keywords itself.

    A declaration nobody can be missing from is not a mechanism. This asserts
    the direction that actually catches drift: **for every place the wrapper
    reads the deck, some item declares that read.** Adding a scanner without
    the declaration fails here.

    The scanners are their own oracle — the test never restates a keyword. It
    writes a one-line deck from a field's ``engine_key`` and asks the scanner
    whether it sees it, which is exactly the question the wrapper asks.

    Found the gap it now guards (2026-08-13, T8): the wrapper scanned TWO
    keywords — ``Diag.Algorithm`` for the env route and ``Diag.ELPA.GPU`` for
    the GPU runtime — and only ``diag_algorithm`` declared ``read_by``. An
    implementation trusting the declarations would have dropped every GPU
    runtime fact (gres, MPS, the NUMA pin) in silence.
    """
    declared = [f for f in dataclasses.fields(SiestaConfig)
                if "wrapper" in (f.metadata.get("read_by") or ())]
    assert declared, "no item declares read_by=wrapper — § 6.1 is hollow"

    for name, scanner in _wrapper_deck_scanners():
        seen_by = []
        for f in declared:
            key = f.metadata.get("engine_key")
            if not key:
                continue
            for value in _trigger_values(f):
                deck = tmp_path / f"{name}_{f.name}.fdf"
                deck.write_text(f"{key} {value}\n")
                if scanner(deck):
                    seen_by.append(f.name)
                    break
        assert seen_by, (
            f"runwrap.{name} reads a deck keyword that NO item declares "
            f"read_by=('wrapper',). Declared today: "
            f"{[f.name for f in declared]}. Either the field's metadata is "
            f"missing the declaration, or the wrapper grew a read the "
            f"template does not know about (template.md § 6.1)."
        )


def test_the_read_by_guard_fails_when_a_declaration_goes_missing(tmp_path):
    """The guard above is only worth having if removing a declaration breaks
    it. Drop ``enable_gpu``'s and the GPU-runtime scanner is left unclaimed."""
    from molbuilder import runwrap
    declared = [f for f in dataclasses.fields(SiestaConfig)
                if "wrapper" in (f.metadata.get("read_by") or ())
                and f.name != "enable_gpu"]
    deck = tmp_path / "probe.fdf"
    orphaned = []
    for name, scanner in _wrapper_deck_scanners():
        hits = []
        for f in declared:
            key = f.metadata.get("engine_key")
            if not key:
                continue
            for value in _trigger_values(f):
                deck.write_text(f"{key} {value}\n")
                if scanner(deck):
                    hits.append(f.name)
                    break
        if not hits:
            orphaned.append(name)
    assert orphaned == ["_fdf_requests_gpu"], (
        "without enable_gpu's declaration exactly one scanner should be "
        f"unclaimed; got {orphaned}. If this changed, the guard above may "
        "have stopped testing anything.")
    assert runwrap._fdf_requests_gpu is not None      # the scanner is live


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


def test_both_spellings_of_optional_are_understood():
    """``Optional[int]`` and ``int | None`` are the same annotation.

    They report DIFFERENT origins -- ``typing.Union`` and ``types.UnionType`` --
    so a check for one silently misses the other.  Nothing in the configs uses
    the newer spelling today, which is why this is a trap rather than a live
    bug: the first field written that way would be declared NOT optional and
    then fail to get a type at all, with the error pointing at the annotation
    instead of at the check that could not read it (audit § 1.1).
    """
    import typing as _t
    from molbuilder.template import _unwrap_optional
    assert _unwrap_optional(_t.Optional[int]) == (int, True)
    assert _unwrap_optional(int | None) == (int, True)
    assert _unwrap_optional(_t.Optional[str]) == (str, True)
    assert _unwrap_optional(str | None) == (str, True)
    # A union that is not an Optional stays un-unwrapped, both ways round.
    assert _unwrap_optional(int | str) == (int | str, False)
    assert _unwrap_optional(int) == (int, False)


#: Fixtures for § 1.1, at MODULE scope on purpose: this file uses
#: ``from __future__ import annotations``, so annotations are strings and
#: ``get_type_hints`` resolves them against the module globals.  A dataclass
#: defined inside a test function cannot be resolved at all.
_PEP604_META = {"category": ("execution",), "engine_key": "Thing",
                "help": "a thing", "null_label": "(auto)"}


@dataclasses.dataclass
class _OldSpelling:
    thing: typing.Optional[int] = dc_field(default=None,
                                           metadata=_PEP604_META)


@dataclasses.dataclass
class _NewSpelling:
    thing: int | None = dc_field(default=None, metadata=_PEP604_META)


def test_a_field_annotated_the_new_way_declares_the_same_item():
    """End to end, because § 1.1's cost is a DECLARATION that differs.

    The same field written both ways must produce the same item -- same type,
    same ``optional`` -- or a config modernised one field at a time would
    silently change its own template.
    """
    from molbuilder.template import declaration_for
    old = declaration_for(dataclasses.fields(_OldSpelling)[0],
                          typing.get_type_hints(_OldSpelling)["thing"])
    new = declaration_for(dataclasses.fields(_NewSpelling)[0],
                          typing.get_type_hints(_NewSpelling)["thing"])
    assert old.type == new.type == "int"
    assert old.optional is new.optional is True


def test_the_module_exports_its_own_read_api_and_vocabularies():
    """§ 1.5: `__all__` omitted the module's headline API.

    `template.md` § 8.0 calls ``select`` and ``one`` **the one read API**, and
    neither was exported; nor were the three closed vocabularies, so a surface
    wanting to order panels by the closed six had to hard-code them or reach
    past the declared public surface.  Both are the same failure -- a module
    whose declared surface is narrower than the contract it implements.

    **Scoped to this module deliberately.** The audit filed this against the
    package convention on the evidence of *2 of 2 modules read*.  Measured
    across the package on 2026-08-14: **97 modules declare ``__all__`` and 80
    do not**, and `docs/process/package-layout.md` states no rule about either.
    A package-wide gate needs that rule written first; asserting one here from
    a sample of two would invent policy.
    """
    from molbuilder import template as _t
    for name in ("select", "one", "CATEGORIES", "RESOLVERS",
                 "ALLOCATION_RESOLVERS"):
        assert name in _t.__all__, (
            f"{name} is part of what template.md § 8.0 documents but is not "
            f"in __all__ -- the module's declared surface is narrower than "
            f"its contract")
    # And nothing is exported that does not exist.
    for name in _t.__all__:
        assert hasattr(_t, name), f"__all__ names {name}, which is not defined"

    # NOTHING here asserts "every public name is exported".  A module's
    # namespace also holds what it imported -- ``Any``, ``Dict``, ``Optional``
    # -- and no rule anywhere says a module must re-export or hide those.
    # Writing one from this module alone would be inventing policy, which is
    # the thing § 1.5 is complaining about in the other direction.
