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
    text = T.render_template(SiestaConfig(system_label="JOB", mpi_np=None))
    item = T.read_template(text).get("mpi_np")
    assert item.value is None and not item.is_set
    assert "\nvalue" not in text.split("[item.mpi_np]")[1].split("[item.")[0]


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
    assert raw["engine"] == "siesta"
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
            '[item.mesh_cutoff]\nkind = "engine"\nanchor = "MeshCutoff"\n'
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
            '[item.x]\nkind = "wishful"\ntype = "int"\nhelp = "x"\n')
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
        assert item.label, f"{item.name} has no human-readable label"
        assert item.section, f"{item.name} has no fieldset"


def test_an_optional_item_says_what_unset_is_called():
    """``optional`` says *unset* is a real state; ``null_label`` is what says
    how to show it.  Without it a tri-select has no third label."""
    text = T.render_template(SiestaConfig(system_label="JOB"))
    assert T.read_template(text).get("mpi_np").null_label


def test_the_three_surface_keys_survive_the_round_trip():
    text = T.render_template(SiestaConfig(system_label="JOB"))
    item = T.read_template(text).get("mesh_cutoff")
    assert item.label and item.section and item.unit == "Ry"


# --------------------------------------------------------------------- #
#  Every parameter is an item — § 7                                     #
# --------------------------------------------------------------------- #

def test_every_exposed_field_becomes_an_item_and_declares_its_kind():
    """§ 7's membership rule is total: *"every parameter the engine's schema
    declares is an item, and each one's ``kind`` says who consumes it."*  A
    field this vocabulary cannot place is a gap to fix, not an item to drop."""
    items = T.declarations_for(SiestaConfig)
    exposed = [f.name for f in __import__("dataclasses").fields(SiestaConfig)
               if f.metadata.get("section")]
    assert sorted(i.name for i in items) == sorted(exposed)
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
