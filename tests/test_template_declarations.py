"""The schema behind a template: what every parameter must declare, and the
shape they declare.

**Every test here is derived from a live contract.** The rule that decides
each: *a guard asserts what the contract says. It must never assert what the contract says should NOT be true* —
a test pinning a retired format makes replacing it harder and reads to the next
person as policy.

Contracts:

* ``docs/engines/template.md`` — what a template **is**. § 3 (the required keys,
  and *a missing* ``value`` *means explicitly unset*), § 5 (the `type`
  vocabulary and where every key comes from), § 7 (membership is **total**, and
  the three things that are not items), § 10 (complete, lossless).
* ``docs/execution/job-contracts.md`` § 3.1 (the reserved blocks of a generated
  script, which the shared marker finds) · § 3.3 (BENCH-MARKS, whose
  declarations come from the **same** field metadata — so the two cannot drift).
* ``docs/engines/stages.md`` § 6.6 — the preflight rows.
"""
from __future__ import annotations

import typing

import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.deck_record import MARKER_RE
from molbuilder.script_emit import benchmark_declarable_types
from molbuilder.template import items_of


ENGINES = [SiestaConfig]


def _cat(engine="siesta"):
    """Items as the CATALOGUE declares them — the master (§ 4.3)."""
    from molbuilder import template as _T
    return {i.name: i for i in _T.select(
        _T.read_template(_T.load_catalogue()), engine=engine)}


# --------------------------------------------------------------------- #
#  The declaration grammar                                              #
# --------------------------------------------------------------------- #


def test_a_bench_marks_field_declares_the_same_type_as_its_template_item():
    """The *one source* rule, checked where it can actually break.

    `job-contracts.md` § 3.3: *"BENCH-MARKS and the template are emitted from
    ONE source, and that is a rule rather than a convenience … two
    hand-maintained copies of ``default=`` would drift, and the drift would be
    silent."*  ``SIESTA_BENCH_FIELDS`` **is** hand-maintained, so the rule is
    an intention there rather than a mechanism -- this test is the mechanism.
    Matched by ANCHOR, which is what a BENCH-MARKS line and a template item
    have in common.
    """
    from molbuilder.script_emit import SIESTA_BENCH_FIELDS
    # A keyword reaches the deck two ways: an ``engine`` item's ``anchor``, or
    # a ``deck`` item's ``expands`` -- ``MD.Steps`` is the second, one of
    # the two keywords ``relax_steps`` becomes depending on ``relax_type``.
    # Both are the keyword a BENCH-MARKS anchor greps for.
    by_keyword = {}
    for d in _cat().values():
        for kw in ((d.anchor,) if d.anchor else ()) + tuple(d.expands or ()):
            by_keyword.setdefault(kw, d)
    for bf in SIESTA_BENCH_FIELDS:
        item = by_keyword.get(bf.anchor)
        assert item is not None, (
            f"BENCH-MARKS declares {bf.anchor!r}, which no config field "
            f"anchors -- so a tool may override a line the template cannot "
            f"describe (job-contracts.md § 3.3).")
        # THE BENCH MAY BE NARROWER THAN THE DECK, NEVER WIDER.
        #
        # ``BlockSize`` is a plain ``int`` on the template and ``pow2`` on
        # BENCH-MARKS, and the asymmetry is the point: the DECK honours any positive integer (SIESTA's own manual gives no
        # power-of-two rule for ``BlockSize``), while the BENCHMARK sweeps
        # powers of two because that is a sensible sweep, not a validity
        # constraint (`engines/tuning.md` § 2.11).
        #
        # Narrower is safe -- the bench simply never proposes a value the
        # deck would refuse.  WIDER is the dangerous direction, and it is
        # what this assertion is really for: a bench that could hand back
        # something the deck rejects.
        _NARROWINGS = {("pow2", "int")}
        assert (bf.type_ == item.type
                or (bf.type_, item.type) in _NARROWINGS), (
            f"{bf.anchor}: BENCH-MARKS says type={bf.type_!r}, the template "
            f"item {item.name!r} says {item.type!r}.  A bench type may be a "
            f"NARROWING of the template's (it sweeps a subset); anything "
            f"else means a tool could propose an override the deck refuses "
            f"(job-contracts.md § 3.3).")
        assert bf.type_ in benchmark_declarable_types(), \
            f"{bf.anchor}: {bf.type_}"


@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_optional_is_set_for_exactly_the_optional_fields(cls):
    """*Unset* is a real state and distinct from every value the field could
    hold — for these fields the engine gets no line at all, which is not the
    same as getting the default."""
    hints = typing.get_type_hints(cls)
    for d in items_of(cls).values():
        ann = hints[d.name]
        is_opt = (typing.get_origin(ann) is typing.Union
                  and type(None) in typing.get_args(ann))
        assert d.optional is is_opt, d.name


def test_the_kgrid_is_one_declaration_not_three():
    """It is one decision — how finely reciprocal space is sampled — and a
    stage overriding it overrides all three components together."""
    assert _cat()["kgrid"].type == "int3"


def test_the_kgrid_displacement_is_its_own_item_and_a_float3():
    """Two items, not one, and not a general ``matrix``.

    The mesh (how finely) and the origin (where it sits) are separate
    scientific decisions and a stage may vary one without the other, so they
    are two items.  Neither is a matrix: `kgridinit.F` accepts a full
    non-diagonal ``kscell``, but that serves supercells commensurate with a
    sub-lattice and nothing in molbuilder builds one -- recorded as *not
    offered* rather than half-offered
    (`docs/archive/2026-08-14-template-execution-review.md` § 53.5).

    ``float3`` is the type that leaves: the components are floats, they are
    independent, and no other member of § 5's vocabulary can carry them.
    """
    from molbuilder.template import TYPES
    assert "float3" in TYPES
    d = _cat()["kgrid_displacement"]
    assert d.type == "float3"
    assert d.default == (0.0, 0.0, 0.0)
    assert _cat()["kgrid"].type == "int3"   # still separate


def test_range_unit_and_group_come_from_the_CATALOGUE():
    """§ 4.3: a surface holding the file needs nothing else to bound the
    control, label it, and decide whether its *vary per stage* box starts
    ticked — and it reads all three from the catalogue.
    """
    d = _cat()["mesh_cutoff"]
    assert d.range == (100.0, 1000.0)
    assert d.unit == "Ry"
    assert d.group == "stage"


@pytest.mark.parametrize("table", [
    'offered = { vibraton = ["CG"] }',
    'offered = { siesta = { vibraton = ["CG"] } }',
    "offered = { siesta = {} }",
    'recommended = { vibraton = "CG" }',
    'offered = { vibration = ["CG"] }\n'
    'not_offered = [{ on = "vibraton", values = ["FIRE"], why = "w" }]',
])
def test_a_misspelled_kind_is_refused_by_name(table):
    """`engines/template.md` § 6.3a: a per-kind table keyed by a name that
    is no calculation kind is a table nothing asks -- for `offered`, every
    refusal the kind was meant to carry silently gone (the K2 review); for a
    `not_offered` row, its reason never given.
    API-level because the road cannot reach it: the catalogue is authored,
    not described."""
    from molbuilder import template as T
    text = ('schema = "molbuilder/template@2"\nengines = ["siesta"]\n'
            '[item.relax_type]\nkind = "engine"\ncategory = ["procedure"]\n'
            'engines = ["siesta"]\nanchor = "MD.TypeOfRun"\ntype = "enum"\n'
            'choices = ["CG", "FIRE"]\nhelp = "x"\n' + table + "\n")
    with pytest.raises(ValueError, match="calculation kind"):
        T.read_template(text)


def test_pyscf_renders_a_template_at_all():
    """The template is the single source of truth, so PySCF must produce
    one."""
    from molbuilder.template import template_with_values, read_template
    t = read_template(template_with_values(PySCFConfig(), engine="pyscf"))
    assert len(t.items) > 30
    assert all(i.category for i in t.items)


# --------------------------------------------------------------------- #
#  The reserved script blocks — job-contracts.md § 3.1                  #
# --------------------------------------------------------------------- #


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


# --------------------------------------------------------------------- #
#  The help-authoring convention `deck_note` depends on                  #
# --------------------------------------------------------------------- #


def test_the_calculation_kind_filters_the_generated_template():
    """`template.md` § 6.3's sibling rule (spectra-migration P0, 2026-08-20):
    `calculations` narrows an item to its kinds exactly as `engines` narrows
    it to its engines — absent means all.  An OPTIMIZATION template carries
    no vibration item; a VIBRATION template carries them PLUS the shared
    ones; and the generated file carries the key on no item (the writer
    strips it, the same rule as `engines`).  The twelve vibration rows
    leaked into every optimization template the day they were added — this
    is the pin that keeps the door shut."""
    from molbuilder import template as T
    from molbuilder.config.pyscf import PySCFConfig

    cfg = PySCFConfig(job_name="X")
    opt = T.template_with_values(cfg, engine="pyscf")
    vib = T.template_with_values(cfg, engine="pyscf",
                                 calculation="vibration")
    for name in ("already_relaxed", "compute_raman", "es_mode_selection"):
        assert f"[item.{name}]" not in opt, (
            f"{name} leaked into an optimization template")
        assert f"[item.{name}]" in vib
    assert "[item.basis]" in opt and "[item.basis]" in vib, (
        "shared items must ride both kinds")
    assert "calculations = " not in vib, (
        "the generated file must not carry the key -- selection already "
        "happened (the engines-stripping writer rule)")


def test_a_recommended_value_on_a_tier_field_is_the_engines_tight_tier():
    """`template.md` § 6.3a: a kind's `recommended` value on a field the
    engine's tier table also carries is that table's tight tier -- one
    number with two homes, compared here so they cannot drift.  The
    vibration kind's recommendations are the ones the rule was written
    for (`vibration.md` § 3.1)."""
    from molbuilder.config.pyscf import PYSCF_STAGE_PRESETS
    from molbuilder.config.siesta import SIESTA_STAGE_PRESETS
    from molbuilder.template import catalogue, select

    tight = {"siesta": SIESTA_STAGE_PRESETS[max(SIESTA_STAGE_PRESETS)],
             "pyscf": PYSCF_STAGE_PRESETS[max(PYSCF_STAGE_PRESETS)]}
    compared = []
    for engine, table in tight.items():
        for it in select(catalogue(), engine=engine):
            for kind, value in it.recommended:
                if it.name in table:
                    assert value == table[it.name], (
                        f"{engine} {it.name}: recommended {value!r} for "
                        f"{kind}, the tight tier says {table[it.name]!r}")
                    compared.append((engine, it.name))
    # the rule has subjects: every relaxation setting § 3.1 exposes on the
    # vibration form carries the tight tier as its recommendation
    assert {n for _e, n in compared} >= {
        "relax_type", "relax_steps", "relax_force_tol", "relax_max_displ",
        "geom_gmax", "geom_grms", "geom_dmax", "geom_drms", "geom_etol",
        "geom_max_steps"}


@pytest.mark.parametrize("start, what", [
    ("default = 0.0", "its default 0.0"),
    ("default = 0.04\nrecommended = { vibration = 0.0 }",
     "its recommended value for vibration 0.0"),
])
def test_the_catalogue_starts_no_calculation_past_a_limit(start, what):
    """`engines/template.md` § 5.3 and § 6.3a: an item's default and each
    kind's recommended value obey its own hard limit -- a kind is never
    started on a value every door would refuse.  A person's VALUE is not
    held here: in a calculation's template it is judged by the one
    per-value door, with its one message.  API-level because the road
    cannot reach it: the catalogue is authored, not described.

    MUTATION THIS MUST FAIL AGAINST: the recommended values left unchecked
    (the second case loads)."""
    from molbuilder import template as T
    text = ('schema = "molbuilder/template@2"\nengines = ["siesta"]\n'
            '[item.fc_displacement]\nkind = "engine"\ncategory = ["accuracy"]\n'
            'engines = ["siesta"]\nanchor = "FC.Displacement"\n'
            'type = "float"\nhelp = "x"\n'
            'above = { value = 0, why = "SIESTA divides by it" }\n'
            + start + "\n")
    with pytest.raises(ValueError, match="breaks its own limit") as refused:
        T.read_template(text)
    assert what in str(refused.value), str(refused.value)
    loads = T.read_template(text.replace(start, "default = 0.04\nvalue = 0.0"))
    assert T.one(loads, "fc_displacement").value == 0.0, (
        "a VALUE past the limit is the per-value door's, not the parser's")
