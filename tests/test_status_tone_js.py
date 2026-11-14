"""The shell's tone on a note or an answer with a layout of its own --
`lib/status.js` `tone` (`web/ui-contract.md` § 5: a severity looks the same
on every page, its colour styled once in `page-shell.css`).

THE SILENT FAILURE: Task setup kept a severity vocabulary of its own
(`data-state` ``bad`` / ``warn`` / ``ok``), coloured by its own sheet -- so a
warning it had no rule for (*this folder changed on disk; what this page
holds is not saved*) read in the grey of a remark, and a success had no
colour at all.  A tone is the shell's: ``tone <kind>`` beside the element's
own classes, ``null`` the resting look.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _node_esm import run_node

STATUS = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/lib/status.js")


@pytest.mark.parametrize("before, kind, after", [
    ("ts-prep-say ts-keeps-lines", "warn", "ts-prep-say ts-keeps-lines tone warn"),
    ("ts-ask-note tone warn", "error", "ts-ask-note tone error"),
    ("ts-ask-note tone error", None, "ts-ask-note"),
    ("hint", "ok", "hint tone ok"),
    # A kind the shell does not declare is no tone at all -- never a class
    # nobody styles.
    ("hint tone ok", "bad", "hint"),
], ids=["a-warning", "a-tone-replaced", "back-to-resting", "a-success",
        "not-a-kind"])
def test_a_tone_is_the_shells_beside_the_elements_own_classes(
        before, kind, after):
    out = run_node([STATUS], f"""
        const el = {{ className: {before!r} }};
        molbuilder.status.tone(el, {"null" if kind is None else repr(kind)});
        console.log(JSON.stringify(el.className));
    """)
    assert out == after, out
