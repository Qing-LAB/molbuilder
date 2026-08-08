"""The template as a real artifact: written, read back, and guarded.

Contract: ``docs/execution/job-contracts.md`` § 3.7 · ``docs/engines/stages.md``
§ 4 (the template is the science backbone a stage's ``overrides`` land on).

P2 unit 4a's second half, unblocked by the 2026-08-07 decision that ``prep``
**re-renders** rather than splicing. That decision moved § 3.7 property 1 from
a structural guarantee to a checked one, so the check is the point of this
file: :func:`test_every_payload_is_byte_identical_to_the_deck` is what property
1 now means.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.siesta.input import effective_config, render_fdf
from molbuilder.structure import Structure
from molbuilder.task import Stage
from molbuilder.template import (
    config_from_template,
    declarations_for,
    read_template,
    render_template,
)


@pytest.fixture
def h2() -> Structure:
    return Structure(elements=["H", "H"],
                     positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                     vacuum=(12.0, 12.0, 12.0))


def _template(cfg: SiestaConfig, h2: Structure) -> str:
    return render_template(render_fdf(h2, cfg), cfg)


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
#  The round trip                                                       #
# --------------------------------------------------------------------- #

def test_a_config_survives_the_round_trip(cfg, h2):
    """**The unit's reason for existing.**  ``prep`` holds a template and must
    produce the ordinary config a stage's ``overrides`` land on (§ 4).  If the
    trip is lossy, every stage resolves against something the user did not
    write."""
    back = config_from_template(_template(cfg, h2), SiestaConfig)
    for f in declarations_for(SiestaConfig):
        assert getattr(back, f.name) == getattr(cfg, f.name), f.name


def test_the_round_trip_preserves_types_not_just_text(cfg, h2):
    """A value that comes back as ``"true"`` or ``"3"`` compares equal in the
    wrong places and renders a deck that is subtly not the one asked for."""
    back = config_from_template(_template(cfg, h2), SiestaConfig)
    assert back.spin_polarized is True
    assert isinstance(back.relax_steps, int)
    assert isinstance(back.mesh_cutoff, float)
    assert tuple(back.kgrid) == (3, 3, 1)


def test_an_unset_optional_comes_back_unset_not_defaulted(h2):
    """*Unset* is a real state: SIESTA gets no line at all, which is not the
    same as getting the default.  A round trip that filled it in would turn
    "let the engine decide" into "I chose this"."""
    cfg = SiestaConfig(system_label="JOB", mpi_np=None, net_charge=None)
    back = config_from_template(_template(cfg, h2), SiestaConfig)
    assert back.mpi_np is None
    assert back.net_charge is None


def test_a_template_missing_a_field_keeps_the_class_default(h2):
    """A template written against an older schema is **missing** fields, not
    wrong about them.  The fingerprint is what says the shape moved; the
    reader's job is to not invent values."""
    text = _template(SiestaConfig(system_label="JOB"), h2)
    kept = [b for b in text.split("\n\n")
            if "item mesh_cutoff BEGIN" not in b]
    back = config_from_template("\n\n".join(kept), SiestaConfig)
    assert back.mesh_cutoff == SiestaConfig().mesh_cutoff


# --------------------------------------------------------------------- #
#  § 3.7 property 1 — now a checked property                            #
# --------------------------------------------------------------------- #

def test_every_payload_is_byte_identical_to_the_deck(cfg, h2):
    """**This test IS property 1.**

    The rule used to be *"producing that deck is a scan and a copy, never a
    re-render"* — structurally true, and unimplementable: a stage overriding
    ``relax_type`` moves the step budget's site from ``MD.NumCGsteps`` to
    ``MD.FinalTimeStep``, so there is no fixed site to substitute at.  The
    decision of 2026-08-07 was to re-render and *check*, which keeps the
    guarantee that matters — a value cannot change shape between what a
    person read and what the engine got — and this is the check.
    """
    deck = render_fdf(h2, cfg)
    text = render_template(deck, cfg)
    deck_lines = set(deck.splitlines())

    checked = 0
    for block in text.split("\n\n"):
        payload = [l for l in block.splitlines()
                   if not l.lstrip().startswith("#")]
        for line in payload:
            assert line in deck_lines, (
                f"template payload line is not in the deck:\n  {line!r}")
            checked += 1
    assert checked > 10, ("too few payload lines to be a real check -- the "
                          f"anchors stopped resolving (got {checked})")


def test_a_field_with_no_single_site_gets_an_empty_payload(cfg, h2):
    """Three shapes of ``engine_key`` name no single deck line, and each is a
    real field of the shipped schema: a **parenthesised note** (``mpi_np`` —
    never in the deck), an **alternation** (``relax_steps`` — the site depends
    on ``relax_type``), a **conjunction** (``spin_total`` — one field, two
    lines).  The block exists and its payload is empty; the value rides on the
    declaration, which is exactly why the anchor stopped being load-bearing."""
    # "# === molbuilder item <name> BEGIN ===" -> token 4 is the field.
    blocks = {b.split()[4]: b for b in _template(cfg, h2).split("\n\n")}
    for name in ("mpi_np", "relax_steps", "spin_total"):
        payload = [l for l in blocks[name].splitlines()
                   if not l.lstrip().startswith("#")]
        assert payload == [], (name, payload)
    # ...and the value still made the trip.
    back = config_from_template(_template(cfg, h2), SiestaConfig)
    assert back.mpi_np == 8 and back.relax_steps == 123


# --------------------------------------------------------------------- #
#  The template is what a stage lands on                                #
# --------------------------------------------------------------------- #

def test_a_stage_resolves_against_the_template_read_back(cfg, h2):
    """The join this unit exists to make: template → config → ⊕ overrides →
    deck.  Asserted end to end rather than at either side of the seam."""
    back = config_from_template(_template(cfg, h2), SiestaConfig)
    tight = effective_config(back, Stage(name="tight",
                                         overrides={"mesh_cutoff": 400.0}))
    assert tight.mesh_cutoff == 400.0
    assert tight.basis_size == "TZP"          # untouched, from the template
    assert tight.relax_steps == 123           # untouched, and it had no payload


def test_every_declared_item_has_a_block(cfg, h2):
    """§ 3.7 property 4: the template is the engine's whole surface,
    instantiated — not the subset that happened to render a line."""
    text = _template(cfg, h2)
    for d in declarations_for(SiestaConfig):
        assert f"molbuilder item {d.name} BEGIN" in text, d.name
        assert f"molbuilder item {d.name} END" in text, d.name


def test_the_block_carries_what_we_know_about_the_item(cfg, h2):
    """§ 3.7 property 3 — the block holds the field's own ``help``, so the
    documentation and the form are the same source and cannot drift."""
    text = _template(cfg, h2)
    helps = {f.name: f.metadata.get("help", "")
             for f in dataclasses.fields(SiestaConfig)}
    first = helps["mesh_cutoff"].strip().splitlines()[0].strip()
    assert first and first in text


def test_read_template_ignores_the_reserved_blocks(cfg, h2):
    """A template sits in a file beside HEADER / PROVENANCE / USER-CUSTOM, and
    the scan must not mistake one of those for an item."""
    text = ("# === molbuilder header BEGIN ===\n"
            "#   field not_a_field  anchor=X  type=int  value=99\n"
            "# === molbuilder header END ===\n\n" + _template(cfg, h2))
    assert "not_a_field" not in read_template(text)
