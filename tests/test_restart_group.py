"""P3 unit 4 — one `restart` field, expanded into the engine's group.

Contract: ``docs/execution/run-identity.md`` § 4 — rule 1 (*an engine declares
its group*), rule 2 (*the user says one thing; the generator sets the group*),
rule 3 (*it is per-stage, because it is an ordinary field*) — and
``docs/execution/job-contracts.md`` § 4.2 (which files those parameters
govern).

**The failure this prevents is silent in two directions** (§ 4): *honoured
with nothing to load*, where the deck says resume and the engine cold-starts;
and *present but not honoured*, where the files are right there and the stage
starts from scratch. The second was live until 2026-08-08 — the renderer read
three booleans that defaulted to True and never read ``restart`` at all, so
``--restart clean`` emitted the whole group and a stage told to start clean
continued.
"""
from __future__ import annotations

import pytest

from molbuilder.config.pyscf import PYSCF_RESTART_GROUP, PySCFConfig
from molbuilder.config.siesta import SIESTA_RESTART_GROUP, SiestaConfig
from molbuilder.identity import RestartGroup


# --------------------------------------------------------------------- #
#  Rule 1 — an engine declares its group                                #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("group", [SIESTA_RESTART_GROUP, PYSCF_RESTART_GROUP],
                         ids=["siesta", "pyscf"])
def test_every_shipped_engine_declares_a_group(group):
    """§ 4 rule 1: *"A new engine that cannot fill this in is a new engine
    whose restart behaviour nobody has thought about yet."*

    Both halves are required. An identity literal with no bound parameters
    describes a name and not a resume; bound parameters with no literal
    describes flags with nothing to key on."""
    assert isinstance(group, RestartGroup)
    assert group.literal
    assert group.mechanism


def test_the_two_engines_mean_the_same_idea_by_different_mechanisms():
    """§ 4's table, read across. SIESTA declares keys; PySCF generates
    control flow. Collapsing that difference is how a design ends up
    describing only the filename."""
    assert SIESTA_RESTART_GROUP.keys
    assert not PYSCF_RESTART_GROUP.keys
    assert SIESTA_RESTART_GROUP.literal != PYSCF_RESTART_GROUP.literal


# --------------------------------------------------------------------- #
#  Rule 2 — the user says ONE thing                                     #
# --------------------------------------------------------------------- #

def test_the_members_cannot_be_set_individually_any_more():
    """§ 4 rule 2: *"no description can carry its members individually and
    disagree with itself"*.

    The three booleans are gone from the schema, which also removes the three
    generated ``--use-save-*`` CLI flags and the three web form controls,
    since both surfaces are built from the dataclass."""
    names = {f.name for f in SiestaConfig.__dataclass_fields__.values()}
    assert not (names & {"use_save_dm", "use_save_cg", "use_save_xv"})
    assert "restart" in names


def test_a_missing_restart_field_reads_as_clean():
    """The safe reading of silence. The dangerous direction is resuming when
    nobody asked -- that discards nothing, but it silently changes what the
    run computed from."""
    class Bare:
        pass
    from molbuilder.identity import continues
    assert continues(Bare()) is False


# --------------------------------------------------------------------- #
#  Rule 3 — it is per-stage, because it is an ordinary field            #
# --------------------------------------------------------------------- #

def test_a_stage_sets_restart_like_any_other_field():
    """§ 4 rule 3. *"A first stage is normally clean and everything after it
    continue. Nothing special is needed to say so."*"""
    from molbuilder.resolve import effective_config
    tpl = SiestaConfig(system_label="job", restart="clean")
    later = effective_config(tpl, {"restart": "continue"})
    assert later.restart == "continue"
    assert tpl.restart == "clean", "the template must not be mutated"


def test_prep_carries_state_only_into_a_stage_that_will_read_it():
    """The other face of *present but not honoured*: state placed beside a
    run that was told not to look at it.

    The carry used to key on the template's ``use_save_dm``, so a stage
    saying 'clean' still had the previous stage's ``.DM`` carried in."""
    # The declaration is built the LIVE way -- the resolved stage through
    # the one seam, exactly what `prep`'s `_job_for` hands each Job.
    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import warm_for
    tpl = SiestaConfig(system_label="job", relax_type="CG")

    def warm(overrides):
        eff = effective_config(tpl, overrides)
        return warm_for("job", eff)

    # THE PROPERTY IS THE GATE, not the membership.  `clean` carries
    # nothing; `continue` carries whatever the rules file declares --
    # DERIVED here rather than listed, because a literal list is a fourth
    # copy of `siesta/warm-files.toml` and goes stale the moment the
    # vocabulary grows.
    from molbuilder.warmfiles import warm_list
    declared = [f"job{r.suffix}"
                for r in warm_list("siesta", "optimization").rules if r.carry]
    assert warm({"restart": "clean"}) == []   # told to start clean
    assert [w.name for w in warm({"restart": "continue"})] == declared
    # ...and the gate is the whole point: the two answers differ.
    assert declared, "the rules file declares no carry rows at all"


# --------------------------------------------------------------------- #
#  Rule 4 — what `continue` implies is a short fixed set, declared by    #
#  the engine (that prep knows none of it: `restart_files.toml`)         #
# --------------------------------------------------------------------- #

def test_only_the_optimizer_history_is_conditional():
    """§ 2.3.4's three rows: `.XV` *"always -- this is the point of
    continuing"*, `.DM` when the description says, `.CG` *"only if both stages
    use the same algorithm"*.

    Only the third needs a second stage, so only the third carries a
    condition — and a condition on the geometry would make a continuation
    silently lose the very thing it exists to move forward.
    """
    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import warm_for
    eff = effective_config(
        SiestaConfig(system_label="job", relax_type="CG"),
        {"restart": "continue"})
    warm = {w.name: w.requires_same for w in warm_for("job", eff)}
    # THE PROPERTY: exactly one row is conditional, and it is the optimiser
    # history.  Asserted as a property rather than as a full dict, because
    # the membership can grow while this rule does not change.
    conditional = {n: c for n, c in warm.items() if c is not None}
    assert conditional == {"job.CG": "optimizer"}, (
        f"expected only the optimiser history to be conditional, got "
        f"{conditional}")
    # And the geometry is unconditional -- a condition here would make a
    # continuation silently lose the very thing it exists to move forward.
    assert warm["job.XV"] is None
    assert warm["job.DM"] is None


# --------------------------------------------------------------------- #
#  PySCF, the other engine                                             #
# --------------------------------------------------------------------- #

def test_pyscf_also_says_it_with_one_field():
    """§ 4 rule 2 for the other engine.  A PySCF ladder is N decks and N jobs
    (`stages.md` § 1.1a consequence 3), so there is a real gap between rungs for the field to
    answer about, and rule 3's first-clean/rest-continue default has
    something to fill in."""
    names = {f.name for f in PySCFConfig.__dataclass_fields__.values()}
    assert "restart" in names
    fld = PySCFConfig.__dataclass_fields__["restart"]
    from molbuilder.template import items_of
    # The SAME two answers SIESTA gives.  A third value on one engine would
    # make `restart` mean different things in two descriptions.
    assert tuple(items_of(PySCFConfig)["restart"].choices) == ("clean", "continue")
    # And the same DEFAULT.  `continue` (user, 2026-08-18): a run started
    # in a folder that already holds a result was started after somebody read
    # that result, so it continues from it.  `clean` is that person overriding,
    # and it overwrites (`run-identity.md` § 4 rule 3).
    assert fld.default == "continue"
