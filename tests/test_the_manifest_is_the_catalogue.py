"""The contract's manifest IS the catalogue -- `project-layout.md` § 5.

`docs/execution/project-layout.md` § 5 lists every file molbuilder writes into
a calculation folder, and its tables are `runfiles.WRITTEN`'s rows, rendered by
`tools/manifest.py` (user, 2026-10-04: "Generate from catalogue"): the one
source of what each file is, which the Task setup card and the Results tab's
file card read too.  So the document is checked against the rendering, and a
row is edited in the catalogue, never in the document.  The rendering refuses
a door that names a function the code does not have.
"""
from __future__ import annotations


def test_the_contracts_manifest_is_the_catalogue_rendered():
    from tools.manifest import DOC, render_into
    text = DOC.read_text(encoding="utf-8")
    assert render_into(text) == text, (
        "project-layout.md § 5 differs from runfiles.WRITTEN -- run "
        "`python -m tools.manifest`, and edit the catalogue, not the table")
