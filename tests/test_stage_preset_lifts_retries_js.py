"""``continue_retries`` reaches the wrapper as an ORDINARY field.

It is an ordinary ``SiestaConfig`` field, so the form collector returns it
(`engines/stages.md § 3`: *"it is an ordinary shared field; what made it look
special is only where it lands."*).
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
    entry per dataclass field, and this is one -- on the form because the
    catalogue carries it (`web/form-schema.md` § 1a)."""
    fields = {f.name: f for f in dataclasses.fields(SiestaConfig)}
    assert "continue_retries" in fields
    from molbuilder import template as _T
    cat = _T.read_template(_T.load_catalogue())
    assert _T.one(cat, "continue_retries", engine="siesta") is not None, (
        "continue_retries is not in the catalogue, so no surface can offer it")


def test_the_siesta_form_has_no_stages_field_to_look_a_policy_up_in():
    """If a SIESTA stage table is ever reintroduced, this fails and whoever
    does it has to say what reads it."""
    assert "stages" not in {f.name for f in dataclasses.fields(SiestaConfig)}


def test_the_viewer_no_longer_lifts_a_policy_out_of_a_stages_table():
    """Source-level, because the failure being prevented is a *reintroduced*
    lookup rather than a wrong one: any read of ``params.stages`` in the
    SIESTA collector is a read of ``undefined``."""
    src = VIEWER.read_text(encoding="utf-8")
    # ONE collector (`collectParams(engine)`), and it must not read a stage
    # table.
    start = src.index("function collectParams(engine)")
    nxt = src.find("\n    function ", start)
    body = src[start:nxt if nxt != -1 else len(src)]
    # Comments may name it; code must not.
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("//"))
    assert not re.search(r"params\.stages", code)
    assert "on_nonconvergence" not in code
