"""The ONE live-route ladder renderer for tests (U20, 2026-08-12).

Three files each carried their own ``_live_ladder_decks`` after the u5
repoint -- the m3 and stages-emit copies identical, the stage-resolution
one a drifted shorthand: the "two lists" defect in test form.  What
`prep` renders is one question; the tests ask it through one function.

**And it is one question for BOTH engines.**  Until this took the engine as
an argument it imported ``render_fdf`` and spelled ``.fdf`` -- so a PySCF
ladder test would have needed a second copy, which is the same defect one
engine further along.  The loop is identical because the SEAM is what
differs: ``EngineSeam`` already names the deck's FORM and its suffix per
engine (`script-preparation.md` § 4), so this asks the seam rather than
knowing either answer itself.
"""
from __future__ import annotations


def _live_ladder_decks(struct, template, stages, engine: str = "siesta"):
    """The decks the LIVE route renders: each ENABLED stage through the one
    seam, one deck per element, exactly as `prep` builds them (repointed
    2026-08-12, u5, from the deleted ``render_siesta_stage_fdfs``).  The
    ordinal comes from the stage's place in the FULL ladder.

    Returns ``{deck filename: deck text}``.  The filename is the engine's --
    ``<label>_<NN>_<name><suffix>`` -- because the token is a render ARGUMENT
    and the NAME is where it lands (`stages.md` § 1.1).
    """
    from molbuilder.identity import stage_token
    from molbuilder.jobset.prep import _engine_seam
    from molbuilder.resolve import effective_config
    seam = _engine_seam(engine)
    label = seam.label_of(template)
    out = {}
    for i, st in enumerate(stages, start=1):
        if not getattr(st, "enabled", True):
            continue
        eff = effective_config(template, getattr(st, "overrides", None),
                               where=f"stage {st.name!r}")
        tok = stage_token(i, st.name)
        # THE SEAM CARRIES A FORM, not finished text (2026-08-18,
        # `script-preparation.md` § 4.3): the engine describes its deck and the
        # framework renders it.  Rendering without writing is what these tests
        # want, so they call the renderer; a caller that WRITES calls
        # ``prepare_deck``, which is validate -> render -> write -> check.
        from molbuilder import script_emit as _sc
        out[f"{label}_{tok}{seam.suffix}"] = _sc.render_deck(
            seam.spec_for(struct, eff, stage_token=tok), struct, eff,
            verbose=eff.verbose_comments)
    return out
