"""The template as a real artifact: written, read back, and guarded.

Contract: ``docs/engines/template.md`` § 3 (the required keys), § 4 (the TOML
format and the writer's self-check), § 5 (the anatomy of an item), § 10
(complete and lossless) · ``docs/execution/generator.md`` § 3.1 (the UI is a
reader, and which keys serve it).

What guards the file is the writer's own round-trip check (§ 4.1) plus the
refusals § 3 asks for.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.resolve import effective_config
from molbuilder.structure import Structure
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
                        spin_treatment="unrestricted", net_charge=-2,
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
    text = T.template_with_values(cfg)
    back = T.config_from_template(text, SiestaConfig)
    for item in T.items_of(SiestaConfig).values():
        if "execution" in (item.category or ()):
            # DELIBERATELY lossy (`execution/job-request.md` § 2): a run
            # setting's value -- the machine's answers among them (§ 2, G1)
            # -- is never written, so it cannot come back.  The item
            # travels; the value is the job request's run card, and `prep`
            # supplies it.  A round trip that preserved it would put one
            # value in two records.
            held = {it.name: it for it in T.read_template(text).items}
            assert item.name not in held or held[item.name].value is None, (
                f"{item.name}'s value survived the trip; the template "
                f"states a run setting")
            continue
        assert getattr(back, item.name) == getattr(cfg, item.name), item.name


def test_the_round_trip_preserves_types_not_just_content(cfg):
    """A value equal in content and different in type is the quietest loss.

    ``kgrid`` is the live case: it is a ``Tuple[int, int, int]`` and TOML has
    only one sequence, so it comes back a list unless the declared type is what
    decides the shape.  This caught a real bug on the day it was written.
    """
    back = T.config_from_template(T.template_with_values(cfg), SiestaConfig)
    assert back.write_hs is True                  # a real bool stays a bool
    assert back.spin_treatment == "unrestricted"  # an enum stays its own string
    assert isinstance(back.relax_steps, int)
    assert isinstance(back.mesh_cutoff, float)
    assert isinstance(back.kgrid, tuple) and back.kgrid == (3, 3, 1)


def test_an_unset_optional_comes_back_unset_not_defaulted():
    """*Unset* is a real state -- the engine's own default, or for the charge
    and spin the instruction "work it out" -- which is not the same as a
    value somebody chose.  A round trip that filled it in would turn "let it
    be decided" into "I chose this"."""
    cfg = SiestaConfig(system_label="JOB", mpi_np=None, net_charge=None)
    back = T.config_from_template(T.template_with_values(cfg), SiestaConfig)
    assert back.mpi_np is None
    assert back.net_charge is None


def test_unset_is_encoded_as_an_absent_key_not_an_empty_one():
    """§ 3: *"a missing ``value`` means explicitly unset"*.  TOML has no null,
    so absence is the encoding — and writing ``value = ""`` instead would make
    *unset* indistinguishable from an empty string."""
    text = T.template_with_values(SiestaConfig(system_label="JOB", net_charge=None))
    item = T.one(T.read_template(text), "net_charge")
    assert item.value is None and not item.is_set
    assert "\nvalue" not in text.split("[item.net_charge]")[1].split("[item.")[0]


def test_a_template_missing_an_item_keeps_the_class_default():
    """A template written against an older schema is **missing** items, not
    wrong about them.  The per-field checks name the ones that moved; the
    reader's job is to not invent values."""
    text = T.template_with_values(SiestaConfig(system_label="JOB"))
    head, _, rest = text.partition("[item.mesh_cutoff]")
    trimmed = head + rest[rest.index("[item."):]
    back = T.config_from_template(trimmed, SiestaConfig)
    assert back.mesh_cutoff == SiestaConfig().mesh_cutoff


def test_a_stage_resolves_against_the_template_read_back(cfg):
    """What ``prep`` actually does: rebuild the config, then land the stage's
    ``overrides`` on it (`stages.md` § 4)."""
    base = T.config_from_template(T.template_with_values(cfg), SiestaConfig)
    eff = effective_config(base, {"mesh_cutoff": 500.0})
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
        T.template_with_values(cfg)


def test_the_emitted_file_is_valid_toml_and_carries_the_three_top_keys(cfg):
    """§ 3: ``schema`` and ``engines`` — the two required top-level keys."""
    import tomllib
    raw = tomllib.loads(T.template_with_values(cfg))
    assert raw["schema"] == T.SCHEMA
    assert raw["engines"] == ["siesta"]      # @2: a LIST -- a calculation may run on several
    assert "fingerprint" not in raw     # § 10


# --------------------------------------------------------------------- #
#  Refusals — § 3                                                       #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("missing", ["kind", "type", "help"])
def test_an_item_missing_a_required_key_is_refused_by_name(missing):
    text = (f'schema = "{T.SCHEMA}"\nengines = ["siesta"]\n\n'
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
    text = (f'schema = "{T.SCHEMA}"\nengine = "siesta"\n\n'
            '[item.x]\nkind = "wishful"\ncategory = ["method"]\n'
            'type = "int"\nhelp = "x"\n')
    with pytest.raises(ValueError, match=r"kind 'wishful' is not one of"):
        T.read_template(text)


def test_an_unknown_item_key_is_refused_not_ignored():
    """An ignored key is a calculation quietly different from the one asked
    for — the same rule ``task.json`` holds itself to."""
    text = (f'schema = "{T.SCHEMA}"\nengine = "siesta"\n\n'
            '[item.x]\nkind = "wrapper"\ntype = "int"\nhelp = "x"\n'
            'trailing_semicolon = true\n')
    with pytest.raises(ValueError, match=r"unknown key"):
        T.read_template(text)


def test_a_future_major_refuses_rather_than_guessing():
    text = ('schema = "molbuilder/template@9"\nengine = "siesta"\n'
            )
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
    for item in T.items_of(SiestaConfig).values():
        # A label serves the FORM, which renders an item under its first
        # category; what every item owes every reader is its help text.
        if item.category:
            assert item.label, f"{item.name} has no human-readable label"
        assert item.help, f"{item.name} has no help text"


def test_an_optional_item_says_what_unset_is_called():
    """``optional`` says *unset* is a real state; ``null_label`` is what says
    how to show it.  Without it a tri-select has no third label."""
    assert T.one(T.catalogue(), "block_size", engine="siesta").null_label


def test_the_three_surface_keys_survive_the_round_trip():
    text = T.template_with_values(SiestaConfig(system_label="JOB"))
    item = T.one(T.read_template(text), "mesh_cutoff")
    assert item.label and item.category and item.unit == "Ry"


# --------------------------------------------------------------------- #
#  Every parameter is an item — § 7                                     #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", [SiestaConfig, PySCFConfig],
                         ids=lambda c: c.__name__)
def test_every_exposed_field_becomes_an_item_and_declares_its_kind(cls):
    """§ 7's membership rule is total: *"every parameter the engine's schema
    declares is an item, and each one's ``kind`` says who consumes it."*  A
    field the catalogue does not carry is a gap to fix, not an item to drop
    -- and since the class carries no copy of any fact (§ 2.1), the
    catalogue is the only place a field can be placed."""
    items = list(T.items_of(cls).values())
    # The membership rule, stated exactly: every parameter the schema
    # declares is an item.  No form-placement tag is in this expression.
    # An allocation field IS a member (§ 6.4): the item is declared,
    # valueless, so a surface can ask for ranks and the wrapper writer knows
    # to look -- the machine-fact row of § 7 excludes the VALUE, not the
    # item.
    members = [f.name for f in __import__("dataclasses").fields(cls)]
    assert sorted(i.name for i in items) == sorted(members)
    assert all(i.kind in T.KINDS for i in items)


# ---- total membership, exercised end to end -------------------------- #

def test_the_five_ungated_fields_round_trip_through_a_template():
    """species_order (identity-sensitive, run-identity § 6a) and a toggle:
    set them, write the template, read it back -- the value survives, which
    is what 'the template describes the deck' means."""
    cfg = SiestaConfig(system_label="JOB",
                       species_order=["C", "H", "S", "Au"],
                       write_hs=False)
    back = T.config_from_template(T.template_with_values(cfg), SiestaConfig)
    assert list(back.species_order) == ["C", "H", "S", "Au"]
    assert back.write_hs is False


def test_a_per_name_table_survives_the_writer():
    """The writer writes every key the reader reads: a kind's own
    recommendation (`recommended`, § 6.3a) and a rung's own answer
    (`role_values`, § 6.4) are tables, and `_toml_value` wrote none until
    2026-09-29 -- a catalogue item handed to it crashed.  A calculation's
    template carries neither (`template_with_values` strips both), so this
    is the writer's totality, asked of the catalogue's own rows."""
    cat = T.catalogue()
    for name in ("solution_method", "relax_force_tol"):
        item = T.one(cat, name, engine="siesta")
        back = T.one(T.read_template(T._emit([item], engines=("siesta",))),
                     name)
        assert (back.role_values, back.recommended) == (
            item.role_values, item.recommended), name


def test_an_unknown_item_name_is_refused_not_dropped():
    """E4: this is a file people edit by hand; a typo'd item silently
    ignored renders a deck missing what the person believes they set.
    Until U16 config_from_template filtered on `k in known` -- the drop."""
    text = T.template_with_values(SiestaConfig(system_label="JOB"))
    assert "[item.mesh_cutoff]" in text
    broken = text.replace("[item.mesh_cutoff]", "[item.mesh_cutof]")
    with pytest.raises(ValueError, match="mesh_cutof"):
        T.config_from_template(broken, SiestaConfig)


def test_a_hand_edited_value_of_the_wrong_type_is_refused_by_name():
    """E5 (U-program follow-up): a template is hand-editable, and § 5 says
    the ``type`` is what a reader MUST check.  Until 2026-08-12 no reader
    did -- the string "three hundred" flowed into a float field and out
    the other side of config_from_template as a str, to surface later in
    rendering with a message about anything but the edit.  The refusal
    names the item and what it declares, in the setting door's one wording
    (`engines/template.md` § 5.4)."""
    import re
    text = T.template_with_values(SiestaConfig(system_label="JOB",
                                          mesh_cutoff=300.0))
    i = text.index("[item.mesh_cutoff]")
    j = text.index("[item.", i + 10)
    block = text[i:j]
    mutated = text.replace(block, re.sub(
        r"^value = .*$", 'value = "three hundred"', block, flags=re.M))
    assert mutated != text
    with pytest.raises(ValueError, match="mesh_cutoff.*not a number"):
        T.read_template(mutated)
    # an enum outside its own choices is the same class of edit
    text2 = T.template_with_values(SiestaConfig(system_label="JOB",
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
    text = T.template_with_values(cfg)
    i = text.index("[item.species_order]")
    j = text.index("[item.", i + 10)
    block = text[i:j]
    mutated = text.replace(block, re.sub(
        r"^value = .*$", 'value = "Au"', block, flags=re.M))
    assert mutated != text
    with pytest.raises(ValueError,
                       match="species_order.*not a list of text values"):
        T.read_template(mutated)
    i = text.index("[item.kgrid]")
    j = text.index("[item.", i + 10)
    block = text[i:j]
    mutated = text.replace(block, re.sub(
        r"^value = .*$", "value = 3", block, flags=re.M))
    assert mutated != text
    with pytest.raises(ValueError, match="kgrid.*not three whole numbers"):
        T.read_template(mutated)


# ------------------------------------------------------------------ #
#  § 6.2 -- the category guards                                       #
#                                                                     #
#  A refusal nothing exercises is a refusal that will be deleted by    #
#  the next person who finds it inconvenient.                          #
# ------------------------------------------------------------------ #

def test_a_template_file_missing_category_is_refused_on_READ():
    """The write side validating is not enough: a template is a file a
    person is invited to edit (§ 4.1), so a hand-deleted `category` must
    be caught when the file is read, not only when it is generated."""
    text = (f'schema = "{T.SCHEMA}"\nengines = ["siesta"]\n\n'
            '[item.x]\nkind = "engine"\nanchor = "X"\n'
            'type = "int"\nhelp = "x"\n')
    with pytest.raises(ValueError, match=r"missing required key 'category'"):
        T.read_template(text)


# ------------------------------------------------------------------ #
#  § 8.0 -- the one read API                                          #
# ------------------------------------------------------------------ #

def _siesta_template():
    return T.read_template(T.template_with_values(SiestaConfig(), engine="siesta"))


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
    both = T.select(T.catalogue(), engine="siesta", category="execution",
                    kind="wrapper")
    assert both and all("execution" in i.category and i.kind == "wrapper"
                        for i in both)


def test_read_by_filters_on_SEVERAL_layers_like_its_siblings():
    """Every filter on `select` is any-of, and `read_by` was the exception.

    `category` and `kind` have always accepted a scalar or a sequence, because
    a reader legitimately asks for more than one at a time -- the deck writer
    wants `engine` and `deck` in one question.  `read_by` accepted a scalar
    only, and the failure was SILENT: asking for two layers compared the whole
    tuple against the item's members, matched nothing, and returned an empty
    list that reads exactly like *no item is read by either* (audit § 1.3).
    """
    t = T.catalogue()
    one = [i.name for i in T.select(t, read_by="wrapper")]
    assert one, "no item declares read_by=wrapper -- the fixture is wrong"
    many = [i.name for i in T.select(t, read_by=("wrapper", "monitor"))]
    assert many == one, (
        "asking for two layers must return at least what asking for one "
        "returns; an empty list here is the scalar-only bug")
    # NO assertion here about a layer nothing declares.  ``engine`` REFUSES an
    # unknown value (the test below says why: an empty list cannot tell
    # *"nothing matched"* from *"you asked for something that does not exist"*),
    # and ``read_by`` is validated against nothing at all -- so pinning its
    # empty answer would freeze an unvalidated axis as if it were designed.
    # Recorded as its own finding against § 6.1 instead.


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


def test_an_item_claiming_an_unlisted_engine_is_refused():
    """§ 6.3: an item's `engines` NARROWS the file's list; it cannot widen
    it.

    Found by reading the emitted file rather than the diff: a template
    whose header said ``engines = ["siesta"]`` happily carried an item
    declaring ``["siesta", "pyscf"]``.  Accepting that lets a surface
    filter for pyscf and be handed items from a calculation that never
    offered it — and which half is wrong is not the reader's to guess, so
    it reports the disagreement and names both sides.
    """
    text = (f'schema = "{T.SCHEMA}"\nengines = ["siesta"]\n\n'
            '[item.x]\nkind = "engine"\ncategory = ["execution"]\n'
            'engines = ["siesta", "pyscf"]\nanchor = "X"\n'
            'type = "int"\nhelp = "x"\n')
    with pytest.raises(ValueError, match=r"x: declares engines \['pyscf'\]"):
        T.read_template(text)


# ------------------------------------------------------------------ #
#  T4 -- the catalogue's machine-answered items are FLAGGED          #
# ------------------------------------------------------------------ #

def test_the_machine_answered_items_are_flagged_and_nothing_else_is():
    """**One flag, and it is the item's own**: the scheduler answers
    exactly these, and the flag says so on the item.
    """
    t = T.catalogue()
    answered = {i.name for i in T.select(t, engine="siesta", allocation=True)}
    # ``gpu_count`` (user, 2026-08-21: "explicit is what we need") -- the
    # device count is the scheduler's grant exactly like the rank count, and
    # as a bench axis it is declared, not derived.
    assert answered == {"mpi_np", "omp_threads", "max_memory_mb",
                        "gpu_count"}, answered

    # `block_size`: the scheduler does NOT grant it -- a benchmark measures it and `prep`
    # realigns it against the GPU target -- so its item may legitimately carry
    # a value.  It needs no name of its own to say that; being unflagged and
    # `optional` with no value already says it.
    bs = T.one(t, "block_size", engine="siesta")
    assert bs is not None and not bs.allocation and bs.optional
