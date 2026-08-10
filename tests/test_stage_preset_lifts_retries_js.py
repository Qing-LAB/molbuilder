"""``continue_retries`` reaches the wrapper as an ORDINARY field.

**This file used to assert the opposite mechanism, and the correction is the
point.** `collectFdfParams` in ``structure-optimization/viewer.js`` carried a
block that looked up one row of a stages table, read its
``on_nonconvergence``, and copied the retry budget to the top level. This file
ran that lookup expression in Node and pinned its behaviour.

Two things were wrong with the thing it guarded, and neither was the lookup:

1. **``params.stages`` never existed on the SIESTA path.** ``SiestaConfig``
   has had no ``stages`` field since P2 deleted ``SiestaStageSpec``, and the
   collector returns *"one entry per dataclass field"* — so the array was
   ``undefined``, ``selStage`` fell back to ``{}``, and the gate could not
   fire whatever the index was. The earlier fix to the *indexing* (a token
   minus one is ``NaN``) was necessary and not sufficient.
2. **The lift was never needed.** ``continue_retries`` is an ordinary
   ``SiestaConfig`` field in the *Compute & budget* section, so ``collectForm``
   already returns it. `engines/stages.md § 3` says so outright: *"it is an
   ordinary shared field; what made it look special is only where it lands."*

So the block was gated on a lookup that always failed, for a value that did
not need lifting — and deleting it (P7 unit 2, with ``on_nonconvergence``) is
what makes the retry budget actually arrive.

**Why the old test passed anyway**, which is the lesson worth keeping: it
*supplied* a stages array as a stub input. It proved the expression worked
given a ladder, and never asked whether the SIESTA path had one. That is
`feedback_test_depth`'s *don't stub the seam that matters*, committed by me
earlier the same day.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from molbuilder.config.siesta import SiestaConfig

REPO = Path(__file__).resolve().parents[1]
VIEWER = (REPO / "molbuilder" / "web" / "static" / "structure-optimization"
          / "viewer.js")


def test_continue_retries_is_an_ordinary_collected_field():
    """The whole mechanism, in one assertion: the form collector returns one
    entry per dataclass field, and this is one — in a rendered section, with
    no flag holding it back."""
    fields = {f.name: f for f in dataclasses.fields(SiestaConfig)}
    assert "continue_retries" in fields
    md = fields["continue_retries"].metadata or {}
    assert md.get("section"), "a field with no section is not rendered at all"
    assert not md.get("skip_cli") and not md.get("hidden")


def test_the_siesta_form_has_no_stages_field_to_look_a_policy_up_in():
    """The premise the deleted block rested on, pinned so it cannot come back
    quietly. If a SIESTA stage table is ever reintroduced, this fails and
    whoever does it has to say what reads it."""
    assert "stages" not in {f.name for f in dataclasses.fields(SiestaConfig)}


def test_the_viewer_no_longer_lifts_a_policy_out_of_a_stages_table():
    """Source-level, because the failure being prevented is a *reintroduced*
    lookup rather than a wrong one: any read of ``params.stages`` in the
    SIESTA collector is a read of ``undefined``."""
    src = VIEWER.read_text(encoding="utf-8")
    body = src[src.index("function collectFdfParams()"):
               src.index("function collectPyscfParams()")]
    # Comments explain the removal and legitimately name it; code must not.
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("//"))
    assert not re.search(r"params\.stages", code)
    assert "on_nonconvergence" not in code
