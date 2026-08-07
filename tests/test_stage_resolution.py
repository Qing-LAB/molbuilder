"""Resolution — what a staged render actually uses when two homes disagree.

Contract: ``docs/engines/stages.md`` § 4 — *effective config = ``base`` ⊕ that
stage's ``overrides``*, one object validated **and** rendered (R1), validated as
a resolved whole and never as a diff (R2).

P2 unit 1 of ``docs/execution/staged-runs-implementation-plan.md``: **pin
today's behaviour before changing anything.**  Four relaxation values live in
two places right now — on ``SiestaConfig`` and again on ``SiestaStageSpec`` —
and ``render_siesta_stage_fdfs`` resolves the collision with a
``dataclasses.replace``.  Whoever is relying on a staged ladder has already
built on whichever one wins, so it is written down here before the mechanism
under it is replaced.

**These assertions are meant to survive P2, not to be retired by it.**  The
*mechanism* changes — four hard-coded fields become an arbitrary ``overrides``
map — but the *rule* they pin is the one § 4 keeps: the stage's value beats the
shared one, and a field the stage says nothing about keeps the shared value.
If a P2 commit has to weaken one of these, that is a design change and belongs
in the contract first.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig, SiestaStageSpec
from molbuilder.siesta.input import render_siesta_stage_fdfs
from molbuilder.structure import Structure


@pytest.fixture
def h2() -> Structure:
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        vacuum=(12.0, 12.0, 12.0),
    )


def _two_stages(**shared) -> SiestaConfig:
    """A config whose SHARED values disagree with both stages' on purpose."""
    return SiestaConfig(
        system_label="JOB",
        stages=[
            SiestaStageSpec(name="coarse", enabled=True, relax_type="CG",
                            relax_steps=100, relax_force_tol=0.05,
                            relax_max_displ=0.30),
            SiestaStageSpec(name="tight", enabled=True, relax_type="Broyden",
                            relax_steps=900, relax_force_tol=0.01,
                            relax_max_displ=0.10),
        ],
        **shared,
    )


def _deck(h2: Structure, cfg: SiestaConfig, stage: str) -> str:
    return render_siesta_stage_fdfs(h2, cfg)[f"{cfg.system_label}_{stage}.fdf"]


def _value(deck: str, key: str) -> str:
    for line in deck.splitlines():
        parts = line.split()
        if parts and parts[0] == key:
            return parts[1]
    raise AssertionError(f"{key!r} not in the deck:\n{deck}")


# --------------------------------------------------------------------- #
#  The stage wins over the shared config                                #
# --------------------------------------------------------------------- #

def _same(emitted: str, expected) -> bool:
    """Compare a deck value to what the stage asked for.

    Numerically for numbers: the emitter formats 0.30 as ``0.3``, which is
    the same number and not this test's business.  Pinning the text would
    make a formatting change look like a resolution bug."""
    if isinstance(expected, str):
        return emitted == expected
    return float(emitted) == pytest.approx(expected)


@pytest.mark.parametrize("engine_key,coarse,tight", [
    ("MD.TypeOfRun",   "CG",   "Broyden"),
    ("MD.NumCGsteps",  100,    900),
    ("MD.MaxForceTol", 0.05,   0.01),
    ("MD.MaxCGDispl",  0.30,   0.10),
])
def test_the_stages_value_beats_the_shared_one(h2, engine_key, coarse, tight):
    """The four duplicated fields: each stage's deck carries ITS value, not
    the shared config's, even though the shared config sets all four to
    something different from both."""
    cfg = _two_stages(relax_type="FIRE", relax_steps=1,
                      relax_force_tol=0.99, relax_max_displ=0.99)
    assert _same(_value(_deck(h2, cfg, "coarse"), engine_key), coarse)
    assert _same(_value(_deck(h2, cfg, "tight"), engine_key), tight)


def test_two_stages_render_different_decks_from_one_config(h2):
    """The point of a ladder, asserted directly rather than field by field."""
    cfg = _two_stages()
    assert _deck(h2, cfg, "coarse") != _deck(h2, cfg, "tight")


# --------------------------------------------------------------------- #
#  A field the stage says nothing about keeps the shared value          #
# --------------------------------------------------------------------- #

def test_a_field_no_stage_carries_keeps_the_shared_value(h2):
    """``mesh_cutoff`` is not on the stage type, so both decks take it from
    the shared config.

    This is the *correct* half of today's behaviour and § 4 keeps it: a stage
    overrides what it names and inherits everything else.  What is missing is
    the other half — there is currently **no way** for a stage to name it,
    which is what M2 exists to fix and what
    ``test_stage_vocabulary.py`` counts.  So this test stays after P2; the
    one that changes is the *reach*, not the inheritance.
    """
    cfg = _two_stages(mesh_cutoff=275)
    for stage in ("coarse", "tight"):
        assert _value(_deck(h2, cfg, stage), "MeshCutoff") == "275"


def test_the_shared_config_is_not_mutated_by_rendering(h2):
    """R1's precondition: rendering a ladder resolves into a NEW object each
    time.  If the shared config were mutated in place, stage 2 would render
    against stage 1's values and the ladder would silently depend on order."""
    cfg = _two_stages(relax_type="FIRE", relax_steps=1,
                      relax_force_tol=0.99, relax_max_displ=0.99)
    before = dataclasses.asdict(cfg)
    render_siesta_stage_fdfs(h2, cfg)
    assert dataclasses.asdict(cfg) == before


# --------------------------------------------------------------------- #
#  What the ladder cannot express today -- the gate M2 opens            #
# --------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=(
    "M2 -- a stage can only vary the four values hard-coded into "
    "render_siesta_stage_fdfs's dataclasses.replace.  `overrides` is what "
    "lets a stage name any field of the shared schema; until it lands, "
    "mesh_cutoff cannot differ between stages by any route."))
def test_a_stage_can_vary_a_parameter_the_stage_type_never_carried(h2):
    """**The gate the whole design named.**

    `staged-runs-architecture.md § 8` puts this between its steps 2 and 3:
    the backend must be able to render a stage that overrides a parameter the
    stage type never carried, *before any of it is drawn*.  Draw first and the
    UI gets designed around what the model happens to allow.

    Written now, red, so M2 has something to turn green rather than a
    description of something to turn green.
    """
    cfg = _two_stages(mesh_cutoff=150)
    cfg.stages[1].overrides = {"mesh_cutoff": 300}      # type: ignore[attr-defined]
    assert _value(_deck(h2, cfg, "coarse"), "MeshCutoff") == "150"
    assert _value(_deck(h2, cfg, "tight"), "MeshCutoff") == "300"
