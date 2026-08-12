"""Resolution — what a staged render actually uses when two homes disagree.

Contract: ``docs/engines/stages.md`` § 4 — *effective config = the template's
values ⊕ that stage's ``overrides``*, one object validated **and** rendered
(R1), validated as a resolved whole and never as a diff (R2).

P2 unit 1 of ``docs/execution/staged-runs-implementation-plan.md`` pinned
today's behaviour **before** changing anything: four relaxation values lived
in two places — on ``SiestaConfig`` and again on ``SiestaStageSpec`` — and
the old renderer resolved the collision with a hard-coded
``dataclasses.replace``.  Whoever relied on a staged ladder had built on
whichever one won, so it was written down before the mechanism was replaced.

**Unit 2 replaced that mechanism, and these assertions survived it** — which
is what unit 1's note said they were for.  The *rule* is the one § 4 keeps:
the stage's value beats the shared one, and a field the stage says nothing
about keeps the shared value.  What changed is the *reach*: a stage names any
field of the shared schema through ``overrides``, instead of the privileged
four a deleted dataclass happened to carry.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.task import Stage


@pytest.fixture
def h2() -> Structure:
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        vacuum=(12.0, 12.0, 12.0),
    )


#: The two-stage ladder, as the DESCRIPTION carries it.  It is not a field of
#: the config below and never travels with it (engines/stages.md § 1.1).
def _ladder() -> list:
    return [
        Stage(name="coarse", overrides={
            "relax_type": "CG", "relax_steps": 100,
            "relax_force_tol": 0.05, "relax_max_displ": 0.30}),
        Stage(name="tight", overrides={
            "relax_type": "Broyden", "relax_steps": 900,
            "relax_force_tol": 0.01, "relax_max_displ": 0.10}),
    ]


def _tpl(**shared) -> SiestaConfig:
    """A template whose SHARED values disagree with both stages on purpose."""
    return SiestaConfig(system_label="JOB", **shared)


def _deck(h2: Structure, cfg: SiestaConfig, stage: str,
          stages=None) -> str:
    """The deck for the stage NAMED *stage*, found by its name.

    Looked up rather than spelled: since decision 27 a deck is
    ``<label>_<NN>_<name>.fdf``, and hard-coding the ordinal here would make
    every one of these tests depend on where the stage sits in ``_ladder()``
    -- which is the coupling R5 is about.  The caller says "the tight one";
    this finds it."""
    decks = _live_ladder_decks(
        h2, cfg, stages if stages is not None else _ladder())
    hit = [k for k in decks if k.endswith(f"_{stage}.fdf")]
    assert len(hit) == 1, f"no single deck for stage {stage!r} in {sorted(decks)}"
    return decks[hit[0]]


def _value(deck: str, key: str) -> str:
    for line in deck.splitlines():
        parts = line.split()
        if parts and parts[0] == key:
            return parts[1]
    raise AssertionError(f"{key!r} not in the deck:\n{deck}")


# --------------------------------------------------------------------- #
#  The stage wins over the template                                     #
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
    """The four values that used to live in two places: each stage's deck
    carries ITS value, not the template's, even though the template sets all
    four to something different from both."""
    cfg = _tpl(relax_type="FIRE", relax_steps=1,
                    relax_force_tol=0.99, relax_max_displ=0.99)
    assert _same(_value(_deck(h2, cfg, "coarse"), engine_key), coarse)
    assert _same(_value(_deck(h2, cfg, "tight"), engine_key), tight)


def test_two_stages_render_different_decks_from_one_tpl(h2):
    """The point of a ladder, asserted directly rather than field by field."""
    cfg = _tpl()
    assert _deck(h2, cfg, "coarse") != _deck(h2, cfg, "tight")


# --------------------------------------------------------------------- #
#  A field the stage says nothing about keeps the template's value      #
# --------------------------------------------------------------------- #

def test_a_field_no_stage_carries_keeps_the_shared_value(h2):
    """``mesh_cutoff`` is in neither stage's ``overrides``, so both decks
    take it from the template.

    § 4 keeps this: a stage overrides what it names and inherits everything
    else.  It is § 6.2's subset rule seen from the render side — the quiet
    cell in the table."""
    cfg = _tpl(mesh_cutoff=275)
    for stage in ("coarse", "tight"):
        assert _value(_deck(h2, cfg, stage), "MeshCutoff") == "275"


def test_the_template_is_not_mutated_by_rendering(h2):
    """R1's precondition: rendering a ladder resolves into a NEW object each
    time.  If the template were mutated in place, stage 2 would render
    against stage 1's values and the ladder would silently depend on order."""
    cfg = _tpl(relax_type="FIRE", relax_steps=1,
                    relax_force_tol=0.99, relax_max_displ=0.99)
    before = dataclasses.asdict(cfg)
    _live_ladder_decks(h2, cfg, _ladder())
    assert dataclasses.asdict(cfg) == before


# --------------------------------------------------------------------- #
#  The gate M2 opened -- no longer xfail                                #
# --------------------------------------------------------------------- #

def test_a_stage_can_vary_a_parameter_the_stage_type_never_carried(h2):
    """**The gate the whole design named**, and it is green.

    `engines/stages.md § 1.2` puts this between its steps 2 and 3:
    the backend must be able to render a stage that overrides a parameter
    the stage type never carried, *before any of it is drawn*.  Draw first
    and the UI gets designed around what the model happens to allow.

    It was written red on 2026-08-07 so P2 would have something to turn
    green rather than a description of something to turn green.  ``mesh_cutoff``
    was unreachable from a stage by ANY route; it is now an ordinary field of
    the shared schema like every other."""
    cfg = _tpl(mesh_cutoff=150)
    ladder = [
        Stage(name="coarse", overrides={"mesh_cutoff": 150}),
        Stage(name="tight", overrides={"mesh_cutoff": 300}),
    ]
    # Compared as NUMBERS.  ``mesh_cutoff`` is declared float, and an override
    # written ``150`` (JSON has one number) is widened to 150.0 on resolve, so
    # the deck reads the same however the description spelled it -- found by
    # the M2 seam walk, 2026-08-07.  Pinning the text here would pin the
    # inconsistency that fix removed.
    assert _same(_value(_deck(h2, cfg, "coarse", ladder), "MeshCutoff"), 150)
    assert _same(_value(_deck(h2, cfg, "tight", ladder), "MeshCutoff"), 300)


def test_a_stage_may_vary_a_field_from_any_section(h2):
    """Not a privileged four and not a privileged section either: the gate
    is about REACH, so assert it across the form's axes -- basis, grid,
    SCF -- not only the one parameter the plan named."""
    cfg = _tpl(basis_size="DZP", dm_tolerance=1e-4)
    ladder = [
        Stage(name="coarse", overrides={"basis_size": "SZ",
                                        "dm_tolerance": 1e-3,
                                        "kgrid": (1, 1, 1)}),
        Stage(name="tight", overrides={"basis_size": "TZP",
                                       "dm_tolerance": 1e-5,
                                       "kgrid": (4, 4, 1)}),
    ]
    coarse = _deck(h2, cfg, "coarse", ladder)
    tight = _deck(h2, cfg, "tight", ladder)
    assert _value(coarse, "PAO.BasisSize") == "SZ"
    assert _value(tight, "PAO.BasisSize") == "TZP"
    assert float(_value(coarse, "DM.Tolerance")) == pytest.approx(1e-3)
    assert float(_value(tight, "DM.Tolerance")) == pytest.approx(1e-5)


# --------------------------------------------------------------------- #
#  § 4 R1 -- the object validated IS the object rendered                #
# --------------------------------------------------------------------- #

def test_the_validator_sees_each_stages_resolved_config(h2, monkeypatch):
    """**R1, asserted rather than assumed.**  M2's wording is *assert
    identity, not equality*: what was checked and what was written cannot
    come apart.

    It holds by construction -- ``render_fdf`` calls the validator on the
    very object it then renders, and the live per-stage render hands it
    ``effective_config``'s result -- but "by construction" is exactly the
    kind of claim that stops being true during a refactor and takes a while
    to notice.  So: spy on the validator, render a two-stage ladder whose
    stages disagree, and assert the validator saw each stage's OWN resolved
    values rather than the template's.

    Before ``overrides`` existed this could only have been asked about the
    privileged four; the point of the gate is that it is now askable about
    any field, so it is asked about ``mesh_cutoff``."""
    import molbuilder.validation as mv

    # ``render_fdf`` imports ``validate`` from the package inside the
    # function body, so the package attribute is the seam to spy on.
    seen = []
    real = mv.validate

    def _spy(struct, cfg, **kw):
        seen.append(cfg)
        return real(struct, cfg, **kw)

    monkeypatch.setattr(mv, "validate", _spy)

    cfg = _tpl(mesh_cutoff=150)
    _live_ladder_decks(h2, cfg, [
        Stage(name="coarse", overrides={"mesh_cutoff": 150}),
        Stage(name="tight", overrides={"mesh_cutoff": 300})])

    assert [c.mesh_cutoff for c in seen] == [150, 300], (
        "the validator saw the template's value, not each stage's -- so a "
        "stage could be checked as one thing and written as another")
    # and none of them is the template object itself
    assert all(c is not cfg for c in seen)


# --------------------------------------------------------------------- #
#  What a ladder must refuse                                            #
# --------------------------------------------------------------------- #

# ``test_a_ladder_with_nothing_enabled_is_refused`` RETIRED 2026-08-12
# (u5): the render-time copy of that refusal died with its renderer.  The
# LIVE refusal is the description's -- a ladder is validated where it is
# written and read (describe / task.py), before anything renders.


def test_two_stages_with_one_name_are_refused_where_the_ladder_is_read():
    """Every per-stage artifact is keyed ``<label>_<name>.fdf``, so a name
    collision does not fail -- it silently overwrites a stage.  The refusal
    lives where the ladder is READ (task.py; repointed 2026-08-12, u5 --
    the render-time copy died with its renderer), and it is
    case-insensitive because the names become filenames."""
    from molbuilder.task import stages_from_dicts
    with pytest.raises(ValueError, match="two stages named"):
        stages_from_dicts([{"name": "tight"}, {"name": "Tight"}])


def test_a_stage_name_that_is_not_a_filename_is_refused():
    """The rule holds for the OBJECT, not only for a description parsed
    from disk: a stage name becomes a filename and an unquoted bash word."""
    with pytest.raises(ValueError, match=r"\[A-Za-z0-9_\]"):
        Stage(name="rm -rf /")


# --------------------------------------------------------------------- #
#  Unit 5 — effective_config, the one place a stage is resolved         #
# --------------------------------------------------------------------- #
#
# engines/stages.md § 4:
#
#   effective config = the template's values ⊕ that stage's `overrides`
#
#   R1  one object is validated AND rendered — what was checked and what
#       was written cannot come apart.
#   R2  a stage is validated as a resolved WHOLE, never as a diff.
#
# § 6.2's subset rule decides the fallback: a stage may omit a varied key,
# and omitting it means "use the template's value".

from molbuilder.siesta.input import effective_config                # noqa: E402


def _template() -> SiestaConfig:
    """The backbone: every field, with values.  What the generating tab
    wrote and what a stage overlays."""
    return SiestaConfig(system_label="JOB", mesh_cutoff=150.0,
                        relax_type="CG", relax_force_tol=0.05,
                        restart="clean")


def test_an_override_wins_over_the_template():
    cfg = effective_config(_template(),
                           Stage(name="tight",
                                 overrides={"mesh_cutoff": 300.0}))
    assert cfg.mesh_cutoff == 300.0


def test_a_field_the_stage_does_not_name_keeps_the_templates_value():
    """§ 6.2's subset rule: absent means 'use the template's value'."""
    cfg = effective_config(_template(),
                           Stage(name="tight",
                                 overrides={"mesh_cutoff": 300.0}))
    assert cfg.relax_type == "CG"
    assert cfg.relax_force_tol == 0.05


def test_it_returns_an_ordinary_config_of_the_engines_own_type():
    """§ 4: 'an ordinary instance of the engine's config dataclass — a
    SiestaConfig, not a new type'.  Which is what lets the SHIPPED
    validator and the SHIPPED emitter take it unchanged."""
    cfg = effective_config(_template(), Stage(name="tight"))
    assert type(cfg) is SiestaConfig


def test_the_template_is_not_mutated():
    """R1's precondition.  Resolving stage 2 must not disturb what stage 3
    resolves against, or the ladder depends on the order it was resolved
    in — a bug that would only appear with three stages."""
    tpl = _template()
    before = dataclasses.asdict(tpl)
    effective_config(tpl, Stage(name="tight", overrides={"mesh_cutoff": 300.0}))
    assert dataclasses.asdict(tpl) == before


def test_two_stages_resolve_independently():
    tpl = _template()
    coarse = effective_config(tpl, Stage(name="coarse",
                                         overrides={"mesh_cutoff": 150.0}))
    tight = effective_config(tpl, Stage(name="tight",
                                        overrides={"mesh_cutoff": 300.0}))
    assert (coarse.mesh_cutoff, tight.mesh_cutoff) == (150.0, 300.0)


def test_a_stage_may_override_ANY_field_not_a_privileged_four():
    """**The gate.**  `mesh_cutoff` was unreachable before; so were
    `basis_size` and `kgrid`.  Nothing about them is special now — they are
    fields of the shared schema like any other."""
    cfg = effective_config(_template(), Stage(name="tight", overrides={
        "mesh_cutoff": 400.0, "basis_size": "TZP", "kgrid": (2, 2, 2),
        "relax_type": "Broyden", "restart": "continue"}))
    assert cfg.mesh_cutoff == 400.0
    assert cfg.basis_size == "TZP"
    assert tuple(cfg.kgrid) == (2, 2, 2)
    assert cfg.relax_type == "Broyden"
    assert cfg.restart == "continue"


def test_an_unknown_field_is_refused_by_name():
    """P1's codec cannot check this — it has no schema.  P2 can, and § 6.6
    says the refusal names the field."""
    with pytest.raises(ValueError) as e:
        effective_config(_template(), Stage(name="tight",
                                            overrides={"mesh_cutof": 300.0}))
    assert "mesh_cutof" in str(e.value)
    assert "tight" in str(e.value)


def test_a_stage_field_name_in_overrides_is_refused():
    """§ 2: a stage has three fields and an override may not redefine one.
    P1 refuses this structurally; here it would also not be a schema field,
    so the message must still be the useful one."""
    with pytest.raises(ValueError) as e:
        effective_config(_template(), Stage(name="tight",
                                            overrides={"enabled": False}))
    assert "enabled" in str(e.value)


def test_an_error_in_any_stage_blocks_the_whole_produce(h2, monkeypatch):
    """§ 4's last line, which had no test until the M2 review looked for one.

    *"An `error` in any stage blocks the whole produce, not just its own
    deck"* — it falls out of § 7.2: the folder appears whole or not at all, so
    there is no such thing as writing the stages that passed.

    The error is INJECTED rather than provoked with a real bad config.  Two
    attempts to provoke one during the review produced no error-severity
    finding at all and so proved nothing either way; what is under test is the
    mechanism — one stage failing aborts the call — not which configs happen
    to be invalid today.
    """
    import molbuilder.validation as mv
    from molbuilder.issues import Issue, ValidationError

    real = mv.validate
    monkeypatch.setattr(mv, "validate", lambda st, cfg, **kw: (
        [Issue("error", "injected", "config.mesh_cutoff")]
        if cfg.mesh_cutoff == 999.0 else real(st, cfg, **kw)))

    with pytest.raises(ValidationError):
        _live_ladder_decks(h2, _tpl(), [
            Stage(name="ok"),
            Stage(name="bad", overrides={"mesh_cutoff": 999.0})])


def _live_ladder_decks(struct, template, stages):
    """The decks the LIVE route renders (repointed 2026-08-12, u5): each
    ENABLED stage through the one seam, exactly as `prep` builds them."""
    from molbuilder.identity import stage_token
    from molbuilder.siesta.input import effective_config, render_fdf
    out = {}
    for i, st in enumerate(stages, start=1):
        if not getattr(st, "enabled", True):
            continue
        eff = effective_config(template, st)
        tok = stage_token(i, st.name)
        out[f"{template.system_label}_{tok}.fdf"] = render_fdf(struct, eff, stage_token=tok)
    return out
