"""The ONE live-route ladder renderer for tests (U20, 2026-08-12).

Three files each carried their own ``_live_ladder_decks`` after the u5
repoint -- the m3 and stages-emit copies identical, the stage-resolution
one a drifted shorthand: the "two lists" defect in test form.  What
`prep` renders is one question; the tests ask it through one function.
"""
from __future__ import annotations


def _live_ladder_decks(struct, template, stages):
    """The decks the LIVE route renders: each ENABLED stage through the one
    seam, one deck per element, exactly as `prep` builds them (repointed
    2026-08-12, u5, from the deleted ``render_siesta_stage_fdfs``).  The
    ordinal comes from the stage's place in the FULL ladder."""
    from molbuilder.identity import stage_token
    from molbuilder.siesta.input import effective_config, render_fdf
    label = template.system_label
    out = {}
    for i, st in enumerate(stages, start=1):
        if not getattr(st, "enabled", True):
            continue
        eff = effective_config(template, st)
        tok = stage_token(i, st.name)
        out[f"{label}_{tok}.fdf"] = render_fdf(struct, eff, stage_token=tok)
    return out

