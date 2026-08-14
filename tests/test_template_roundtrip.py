"""The template as a real artifact: written, read back, and guarded.

Contract: ``docs/engines/template.md`` § 3 (the required keys), § 4 (the TOML
format and the writer's self-check), § 5 (the anatomy of an item), § 10
(complete and lossless) · ``docs/execution/generator.md`` § 3.1 (the UI is a
reader, and which keys serve it).

**Rewritten 2026-08-11 with the format.** Six tests here pinned the retired
item-block layout — ``test_every_payload_is_byte_identical_to_the_deck`` and its
neighbours asserted that each block embedded a copy of the deck's own lines.
That property was real for a format that stored the value twice; **TOML has no
payload key, so the property has nothing to be about** and the tests were
retired rather than ported. What replaced them is the writer's own round-trip
check (§ 4.1) plus the refusals § 3 asks for.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.siesta.input import effective_config
from molbuilder.structure import Structure
from molbuilder.task import Stage
from molbuilder import template as T


@pytest.fixture
def h2() -> Structure:
    return Structure(elements=["H", "H"],
                     positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                     vacuum=(12.0, 12.0, 12.0))


@pytest.fixture
def cfg() -> SiestaConfig:
    """Deliberately NOT all-defaults: a round-trip over default values passes
    even when every value is being dropped and re-defaulted."""
    return SiestaConfig(system_label="JOB", mesh_cutoff=275.0,
                        basis_size="TZP", kgrid=(3, 3, 1),
                        relax_type="Broyden", relax_steps=123,
                        spin_polarized=True, net_charge=-2,
                        restart="continue", continue_retries=4,
                        mpi_np=8, verbose_comments=False)


# --------------------------------------------------------------------- #
#  The round trip — § 10                                                #
# --------------------------------------------------------------------- #

def test_a_config_survives_the_round_trip(cfg):
    """**The unit's reason for existing.**  ``prep`` holds a template and must
    produce the ordinary config a stage's ``overrides`` land on (`stages.md`
    § 4).  If the trip is lossy, every stage resolves against something the
    user did not write."""
    back = T.config_from_template(T.render_template(cfg), SiestaConfig)
    for item in T.declarations_for(SiestaConfig):
        assert getattr(back, item.name) == getattr(cfg, item.name), item.name


def test_the_round_trip_preserves_types_not_just_content(cfg):
    """A value equal in content and different in type is the quietest loss.

    ``kgrid`` is the live case: it is a ``Tuple[int, int, int]`` and TOML has
    only one sequence, so it comes back a list unless the declared type is what
    decides the shape.  This caught a real bug on the day it was written.
    """
    back = T.config_from_template(T.render_template(cfg), SiestaConfig)
    assert back.spin_polarized is True
    assert isinstance(back.relax_steps, int)
    assert isinstance(back.mesh_cutoff, float)
    assert isinstance(back.kgrid, tuple) and back.kgrid == (3, 3, 1)


def test_an_unset_optional_comes_back_unset_not_defaulted():
    """*Unset* is a real state: SIESTA gets no line at all, which is not the
    same as getting the default.  A round trip that filled it in would turn
    "let the engine decide" into "I chose this"."""
    cfg = SiestaConfig(system_label="JOB", mpi_np=None, net_charge=None)
    back = T.config_from_template(T.render_template(cfg), SiestaConfig)
    assert back.mpi_np is None
    assert back.net_charge is None


def test_unset_is_encoded_as_an_absent_key_not_an_empty_one():
    """§ 3: *"a missing ``value`` means explicitly unset"*.  TOML has no null,
    so absence is the encoding — and writing ``value = ""`` instead would make
    *unset* indistinguishable from an empty string."""
    text = T.render_template(SiestaConfig(system_label="JOB", net_charge=None))
    item = T.read_template(text).get("net_charge")
    assert item.value is None and not item.is_set
    assert "\nvalue" not in text.split("[item.net_charge]")[1].split("[item.")[0]


def test_a_template_missing_an_item_keeps_the_class_default():
    """A template written against an older schema is **missing** items, not
    wrong about them.  The fingerprint is what says the shape moved; the
    reader's job is to not invent values."""
    text = T.render_template(SiestaConfig(system_label="JOB"))
    head, _, rest = text.partition("[item.mesh_cutoff]")
    trimmed = head + rest[rest.index("[item."):]
    back = T.config_from_template(trimmed, SiestaConfig)
    assert back.mesh_cutoff == SiestaConfig().mesh_cutoff


def test_a_stage_resolves_against_the_template_read_back(cfg):
    """What ``prep`` actually does: rebuild the config, then land the stage's
    ``overrides`` on it (`stages.md` § 4)."""
    base = T.config_from_template(T.render_template(cfg), SiestaConfig)
    eff = effective_config(base, Stage(name="tight",
                                       overrides={"mesh_cutoff": 500.0}))
    assert eff.mesh_cutoff == 500.0
    assert eff.basis_size == "TZP"          # untouched by the stage


# --------------------------------------------------------------------- #
#  The writer checks itself — § 4.1                                     #
# --------------------------------------------------------------------- #

def test_the_writer_refuses_output_that_does_not_read_back(monkeypatch, cfg):
    """§ 4.1 asks the emitter to *"read its own output back and compare it to
    what it meant to write"*.  A check nobody has broken is an assumption, so
    break it: corrupt one value on the way out and the writer must refuse and
    name the item."""
    real = T._toml_value
    monkeypatch.setattr(T, "_toml_value",
                        lambda v: '"WRONG"' if v == 275.0 else real(v))
    with pytest.raises(ValueError, match=r"does not read back as written"):
        T.render_template(cfg)


def test_the_emitted_file_is_valid_toml_and_carries_the_three_top_keys(cfg):
    """§ 3: ``schema``, ``engine`` and ``fingerprint``, all required."""
    import tomllib
    raw = tomllib.loads(T.render_template(cfg))
    assert raw["schema"] == T.SCHEMA
    assert raw["engines"] == ["siesta"]      # @2: a LIST -- a calculation may run on several
    assert raw["fingerprint"] == T.schema_fingerprint(SiestaConfig)


def test_the_writer_computes_the_fingerprint(cfg):
    """Unit 4a's rule: whatever writes the template computes it, because that
    is the moment the schema is in hand.  Nothing wrote one until 2026-08-11,
    so ``validation/task``'s check either never fired or always complained."""
    assert T.read_template(T.render_template(cfg)).fingerprint != ""


# --------------------------------------------------------------------- #
#  Refusals — § 3                                                       #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("missing", ["kind", "type", "help"])
def test_an_item_missing_a_required_key_is_refused_by_name(missing):
    text = (f'schema = "{T.SCHEMA}"\nengine = "siesta"\nfingerprint = ""\n\n'
            '[item.mesh_cutoff]\nkind = "engine"\ncategory = ["accuracy"]\n'
            'anchor = "MeshCutoff"\n'
            'type = "float"\nhelp = "x"\n')
    text = "\n".join(l for l in text.splitlines()
                     if not l.startswith(f"{missing} ")) + "\n"
    with pytest.raises(ValueError, match=rf"mesh_cutoff.*{missing}"):
        T.read_template(text)


def test_an_unknown_kind_is_refused_not_skipped():
    """§ 6: the vocabulary is closed.  *"A reader that quietly ignored an item
    it did not understand would produce a deck missing a parameter, and say
    nothing."*"""
    text = (f'schema = "{T.SCHEMA}"\nengine = "siesta"\nfingerprint = ""\n\n'
            '[item.x]\nkind = "wishful"\ncategory = ["method"]\n'
            'type = "int"\nhelp = "x"\n')
    with pytest.raises(ValueError, match=r"kind 'wishful' is not one of"):
        T.read_template(text)


def test_an_unknown_item_key_is_refused_not_ignored():
    """An ignored key is a calculation quietly different from the one asked
    for — the same rule ``task.json`` holds itself to."""
    text = (f'schema = "{T.SCHEMA}"\nengine = "siesta"\nfingerprint = ""\n\n'
            '[item.x]\nkind = "wrapper"\ntype = "int"\nhelp = "x"\n'
            'trailing_semicolon = true\n')
    with pytest.raises(ValueError, match=r"unknown key"):
        T.read_template(text)


def test_a_future_major_refuses_rather_than_guessing():
    text = ('schema = "molbuilder/template@9"\nengine = "siesta"\n'
            'fingerprint = ""\n')
    with pytest.raises(Exception):
        T.read_template(text)


# --------------------------------------------------------------------- #
#  The UI is a reader — generator.md § 3.1a                             #
# --------------------------------------------------------------------- #

def test_every_item_carries_what_a_surface_needs_to_render_it():
    """**A template missing these produces a UI that cannot name its own
    fields or group them.**  They were dropped until 2026-08-11 — ``label``
    held the *field* name, ``section`` was read only to decide exposure and
    then discarded, and ``null_label`` was gone entirely."""
    for item in T.declarations_for(SiestaConfig):
        # A label and a fieldset serve the FORM.  An item with no section
        # sits on no tab (§ 7/U16: membership is total; section answers
        # only *where on the form*), so demanding them there would invent
        # UI for a field no surface renders.  What every item DOES owe
        # every reader is its help text.
        if item.category:
            assert item.label, f"{item.name} has no human-readable label"
        assert item.help, f"{item.name} has no help text"


def test_an_optional_item_says_what_unset_is_called():
    """``optional`` says *unset* is a real state; ``null_label`` is what says
    how to show it.  Without it a tri-select has no third label."""
    text = T.render_template(SiestaConfig(system_label="JOB"))
    assert T.read_template(text).get("parallel_block_size").null_label


def test_the_three_surface_keys_survive_the_round_trip():
    text = T.render_template(SiestaConfig(system_label="JOB"))
    item = T.read_template(text).get("mesh_cutoff")
    assert item.label and item.category and item.unit == "Ry"


# --------------------------------------------------------------------- #
#  Every parameter is an item — § 7                                     #
# --------------------------------------------------------------------- #

def test_every_exposed_field_becomes_an_item_and_declares_its_kind():
    """§ 7's membership rule is total: *"every parameter the engine's schema
    declares is an item, and each one's ``kind`` says who consumes it."*  A
    field this vocabulary cannot place is a gap to fix, not an item to drop."""
    items = T.declarations_for(SiestaConfig)
    # The membership rule, stated exactly (U16 made it literal): every
    # parameter the schema declares is an item, excluded only by § 7's
    # named rows -- "a machine fact" (``allocation``, a fact a field
    # declares about itself) and the ladder.  ``section`` is NOT in this
    # expression at all: it answers *where on the form*, and gating
    # membership on it was the fourth, unlisted exclusion that silently
    # kept species_order (identity-sensitive) out of every template.
    members = [f.name for f in __import__("dataclasses").fields(SiestaConfig)
               if not f.metadata.get("allocation")]
    assert sorted(i.name for i in items) == sorted(members)
    assert all(i.kind in T.KINDS for i in items)


def test_an_engine_item_names_a_bare_keyword():
    """§ 5: an anchor is *"a bare keyword, never a sentence"* — so a note, an
    alternation or a conjunction must have been given an explicit kind."""
    for item in T.declarations_for(SiestaConfig):
        if item.kind == "engine":
            assert item.anchor and "(" not in item.anchor, item.name
            assert "|" not in item.anchor and "+" not in item.anchor, item.name


def test_a_deck_item_says_which_keywords_it_produces():
    """§ 3: ``expands`` is required for ``kind='deck'`` — it is how a reader
    learns which keywords the item becomes."""
    for item in T.declarations_for(SiestaConfig):
        if item.kind == "deck":
            assert item.expands, item.name


# ---- U16 (2026-08-12): total membership, exercised end to end -------- #

def test_the_five_ungated_fields_round_trip_through_a_template():
    """species_order (identity-sensitive, run-identity § 6a) and the four
    toggles the section gate silently dropped: set them, write the
    template, read it back -- the value survives, which is what 'the
    template describes the deck' means."""
    cfg = SiestaConfig(system_label="JOB",
                       species_order=["C", "H", "S", "Au"],
                       write_forces=False, copy_psml=False)
    back = T.config_from_template(T.render_template(cfg), SiestaConfig)
    assert list(back.species_order) == ["C", "H", "S", "Au"]
    assert back.write_forces is False
    assert back.copy_psml is False


def test_an_unknown_item_name_is_refused_not_dropped():
    """E4: this is a file people edit by hand; a typo'd item silently
    ignored renders a deck missing what the person believes they set.
    Until U16 config_from_template filtered on `k in known` -- the drop."""
    text = T.render_template(SiestaConfig(system_label="JOB"))
    assert "[item.mesh_cutoff]" in text
    broken = text.replace("[item.mesh_cutoff]", "[item.mesh_cutof]")
    with pytest.raises(ValueError, match="mesh_cutof"):
        T.config_from_template(broken, SiestaConfig)


def test_a_hand_edited_value_of_the_wrong_type_is_refused_by_name():
    """E5 (U-program follow-up): a template is hand-editable, and § 5 says
    the ``type`` is what a reader MUST check.  Until 2026-08-12 no reader
    did -- the string "three hundred" flowed into a float field and out
    the other side of config_from_template as a str, to surface later in
    rendering with a message about anything but the edit."""
    import re
    text = T.render_template(SiestaConfig(system_label="JOB",
                                          mesh_cutoff=300.0))
    i = text.index("[item.mesh_cutoff]")
    j = text.index("[item.", i + 10)
    block = text[i:j]
    mutated = text.replace(block, re.sub(
        r"^value = .*$", 'value = "three hundred"', block, flags=re.M))
    assert mutated != text
    with pytest.raises(ValueError, match="mesh_cutoff.*float"):
        T.read_template(mutated)
    # an enum outside its own choices is the same class of edit
    text2 = T.render_template(SiestaConfig(system_label="JOB",
                                           basis_size="TZP"))
    i = text2.index("[item.basis_size]")
    j = text2.index("[item.", i + 10)
    block = text2[i:j]
    mutated2 = text2.replace(block, re.sub(
        r"^value = .*$", 'value = "ENORMOUS"', block, flags=re.M))
    with pytest.raises(ValueError, match="basis_size"):
        T.read_template(mutated2)


def test_the_type_check_runs_before_shape_can_mangle(tmp_path):
    """R4: the E5 check ran post-construction, AFTER _shape -- so a
    scalar on a strlist exploded "Au" into ['A','u'] and PASSED the
    check (each element a str), and a scalar on int3 died as a raw
    TypeError naming no item.  The check now reads the RAW TOML value."""
    import re
    cfg = SiestaConfig(system_label="JOB", species_order=["C", "H"],
                       kgrid=(3, 3, 1))
    text = T.render_template(cfg)
    i = text.index("[item.species_order]")
    j = text.index("[item.", i + 10)
    block = text[i:j]
    mutated = text.replace(block, re.sub(
        r"^value = .*$", 'value = "Au"', block, flags=re.M))
    assert mutated != text
    with pytest.raises(ValueError, match="species_order.*strlist"):
        T.read_template(mutated)
    i = text.index("[item.kgrid]")
    j = text.index("[item.", i + 10)
    block = text[i:j]
    mutated = text.replace(block, re.sub(
        r"^value = .*$", "value = 3", block, flags=re.M))
    assert mutated != text
    with pytest.raises(ValueError, match="kgrid.*int3"):
        T.read_template(mutated)


def test_a_hand_added_machine_fact_item_is_refused_with_the_story():
    """A-9 (final review, 2026-08-13): ``declaration_for`` never WRITES an
    allocation-tagged field into a template, so one in a template is a
    hand edit — refused with § 7's machine-fact story (state it at prep),
    never the typo story."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.template import config_from_template, render_template
    text = render_template(SiestaConfig(system_label="JOB"))
    text += ('\n[item.mpi_np]\n'
             'kind = "wrapper"\n'
             'type = "int"\n'
             'value = 8\n'
             'default = 8\n'
             'group = "budget"\n'
             'category = ["execution"]\n'
             'label = "MPI ranks (np)"\n'
             'help = "hand-added"\n')
    with pytest.raises(ValueError, match=r"machine fact"):
        config_from_template(text, SiestaConfig)


# ------------------------------------------------------------------ #
#  § 6.2 -- the category guards                                       #
#                                                                     #
#  Both of these were added after a mutation sweep found the refusals  #
#  they defend were UNTESTED: opening the closed vocabulary, and       #
#  dropping `category` from the required keys, each left the suite     #
#  green.  A refusal nothing exercises is a refusal that will be       #
#  deleted by the next person who finds it inconvenient.               #
# ------------------------------------------------------------------ #

def test_an_unknown_category_is_refused_not_accepted():
    """The vocabulary is closed (§ 6.2).

    An accepted-but-unknown category is worse than a rejected one: a
    surface would silently drop the item -- it belongs to no panel -- so
    a parameter vanishes from the form while remaining in the file.
    """
    import dataclasses
    from dataclasses import field as dc_field
    from molbuilder.template import declarations_for
    bad = dataclasses.make_dataclass("Bad", [
        ("x", int, dc_field(default=1, metadata={
            "category": ("thermodynamics",),      # not one of the six
            "help": "x", "workflow_group": "stage"}))])
    with pytest.raises(ValueError, match=r"unknown category 'thermodynamics'"):
        declarations_for(bad)


def test_a_field_with_no_category_is_refused_by_name():
    """§ 3: `category` is required, and the refusal NAMES the field.

    'missing required key' with no field name sends a reader to search
    the whole schema -- the same reason § 3 requires item-named refusals.
    """
    import dataclasses
    from dataclasses import field as dc_field
    from molbuilder.template import declarations_for
    nocat = dataclasses.make_dataclass("NoCat", [
        ("mesh_cutoff", int, dc_field(default=1, metadata={
            "help": "x", "workflow_group": "stage"}))])
    with pytest.raises(ValueError, match=r"mesh_cutoff.*no `category`"):
        declarations_for(nocat)


def test_a_template_file_missing_category_is_refused_on_READ():
    """The write side validating is not enough: a template is a file a
    person is invited to edit (§ 4.1), so a hand-deleted `category` must
    be caught when the file is read, not only when it is generated."""
    text = (f'schema = "{T.SCHEMA}"\nengines = ["siesta"]\nfingerprint = ""\n\n'
            '[item.x]\nkind = "engine"\nanchor = "X"\n'
            'type = "int"\nhelp = "x"\n')
    with pytest.raises(ValueError, match=r"missing required key 'category'"):
        T.read_template(text)


# ------------------------------------------------------------------ #
#  § 8.0 -- the one read API                                          #
# ------------------------------------------------------------------ #

def _siesta_template():
    return T.read_template(T.render_template(SiestaConfig(), engine="siesta"))


def test_select_with_no_filter_is_every_item():
    """`prep` step 2 filters nothing -- it wants them all (§ 8)."""
    t = _siesta_template()
    assert len(T.select(t)) == len(t.items)


def test_select_returns_items_in_CATEGORY_order():
    """The categories ARE the reading order (§ 6.2); a surface renders
    panels top to bottom, so the API hands them over that way rather than
    making every caller re-sort."""
    t = _siesta_template()
    order = [T.CATEGORIES.index(i.category[0]) for i in T.select(t)]
    assert order == sorted(order)


def test_select_finds_an_item_by_its_SECOND_category():
    """The whole point of a category being a list: an item panels under
    its first and stays findable under the rest.  `pao_energy_shift`
    panels under `method` and must be found under `accuracy`, where a
    user hunting precision knobs will look."""
    t = _siesta_template()
    found = [i.name for i in T.select(t, category="accuracy")]
    assert "pao_energy_shift" in found
    assert T.one(t, "pao_energy_shift").category[0] == "method"


def test_select_filters_compose():
    t = _siesta_template()
    both = T.select(t, category="execution", kind="wrapper")
    assert both and all("execution" in i.category and i.kind == "wrapper"
                        for i in both)


def test_an_engine_this_template_does_not_serve_is_REFUSED():
    """*"This calculation does not run on that engine"* and *"no items
    matched"* are different answers, and an empty list cannot tell them
    apart -- a caller would read the refusal as an absence and carry on."""
    t = _siesta_template()
    with pytest.raises(ValueError, match=r"does not run on that engine|not 'pyscf'"):
        T.select(t, engine="pyscf")


def test_one_raises_for_a_name_the_template_never_had():
    """`None` means *does not apply here*; a missing NAME is a different
    thing and must not borrow that answer (Law A, applied to a lookup)."""
    t = _siesta_template()
    with pytest.raises(KeyError, match="no item 'not_a_parameter'"):
        T.one(t, "not_a_parameter")
