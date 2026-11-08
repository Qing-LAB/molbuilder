"""One `restart` field, read one way, and what it hands a stage.

Contract: ``docs/execution/run-identity.md`` § 4 rules 2 and 3 and
``docs/execution/job-contracts.md`` § 4.2.

**The failure this prevents is silent in two directions** (§ 4): *honoured
with nothing to load*, where the deck says resume and the engine cold-starts;
and *present but not honoured*, where the files are right there and the stage
starts from scratch. The second was live until 2026-08-08 — the renderer read
three booleans that defaulted to True and never read ``restart`` at all, so
``--restart clean`` emitted the whole group and a stage told to start clean
continued.
"""
from __future__ import annotations

from molbuilder.config.siesta import SiestaConfig


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
        return warm_for("job", eff, "optimization")

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
    warm = {w.name: w.requires_same for w in warm_for("job", eff, "optimization")}
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
