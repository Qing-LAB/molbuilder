"""L2 tests for the region-label definitions popover.

Pins:
  * The JS source contains all 4 canonical region-label
    definitions (L-electrode / R-electrode / bridge / interface)
    AND the synthetic *-electrode catch-all template.
  * The HTML template carries the ⓘ button + the popover panel.
  * The popover panel is hidden by default.
  * Each canonical definition has both a "what atoms belong here"
    field and a citation tag (so the popover stays useful as a
    fact-checker; missing fields would mean the user sees blanks).

Cheap L2 grep + AST checks; no browser required.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFS_JS = REPO_ROOT / "molbuilder" / "web" / "static" / "lib" / "region-label-definitions.js"
MODIFY_HTML = REPO_ROOT / "molbuilder" / "web" / "templates" / "modify.html"


def test_canonical_definitions_present():
    """All four canonical region-label keys appear in the JS source."""
    src = DEFS_JS.read_text()
    for key in ("L-electrode", "R-electrode", "bridge", "interface"):
        assert f'key: "{key}"' in src, (
            f"region-label definition for {key!r} missing from "
            f"region-label-definitions.js"
        )


def test_each_definition_has_atoms_and_citation():
    """Each definition must carry an ``atoms`` and ``citation`` field
    (the popover would render blanks otherwise — the user's fact-
    checker promise breaks).
    """
    src = DEFS_JS.read_text()
    # Match each ``{ key: ..., short: ..., atoms: ..., practice: ...,
    # citation: ... }`` block.  Lenient on whitespace + escaping; the
    # important invariant is "every entry has both fields."
    objs = re.findall(
        r'\{\s*key:\s*"([^"]+)"[\s\S]*?citation:\s*"([^"]+)"\s*\}',
        src,
    )
    keys = {k for k, _cite in objs}
    for canonical in ("L-electrode", "R-electrode", "bridge", "interface"):
        assert canonical in keys, (
            f"definition for {canonical!r} does not have the full "
            f"(key, short, atoms, practice, citation) shape; "
            f"region-defs-body would render blanks"
        )


def test_electrode_convention_helper_is_consistent_with_python():
    """``isElectrodeLabel`` in JS must mirror
    ``molbuilder.config.transport.is_electrode_label``.
    """
    src = DEFS_JS.read_text()
    # JS function should accept three shapes per the convention:
    # "electrode", "*-electrode", "*_electrode" (case-insensitive).
    assert 'lo === "electrode"' in src
    assert 'lo.endsWith("-electrode")' in src
    assert 'lo.endsWith("_electrode")' in src
    # Python side, same shapes.
    from molbuilder.config.transport import is_electrode_label
    assert is_electrode_label("electrode")
    assert is_electrode_label("L-electrode")
    assert is_electrode_label("tip_electrode")
    assert is_electrode_label("Tip-Electrode")     # case-insensitive
    assert not is_electrode_label("bridge")
    assert not is_electrode_label("interface")


# RETIRED 2026-08-01: `test_popover_html_landed_in_template`.
#
# The ⓘ region-label popover's markup lived in `templates/_selection_panel.html`,
# which is gone -- and with it the feature.  The rebuilt MolView never
# reimplemented it: `lib/molview/ui.js` creates no `selection-target-info-btn`,
# no `-panel`, no `region-defs-*`.  The JS below still ships on six pages via
# `_molview_scripts.html` and now populates nothing.
#
# The tests kept here are the ones still true: the DEFINITIONS in the JS source,
# which are the reusable half and which the feature would be rebuilt from.

def test_modify_template_loads_the_js_module():
    """``modify.html`` loads BOTH the library + the init script.

    2026-06-19 JS-quality review lifted the init code out of the
    inline ``<script>`` block (CSP-blocked in production) into its
    own ``region-label-popover-init.js`` file; this test now
    asserts both script tags are present + the init module exists.
    """
    html = MODIFY_HTML.read_text()
    # The library tag rides the SHARED molview stack include (2026-07 template
    # dedup): modify.html pulls _molview_scripts.html, which carries
    # region-label-definitions.js -- assert the chain, not a hand-pasted tag.
    assert '{% include "_molview_scripts.html" %}' in html, (
        "modify.html no longer pulls the shared molview stack include; "
        "region-label-definitions.js would not load and the ⓘ button "
        "will be a no-op"
    )
    include_html = (MODIFY_HTML.parent / "_molview_scripts.html").read_text()
    assert "region-label-definitions.js" in include_html, (
        "_molview_scripts.html does not load lib/region-label-definitions.js; "
        "the ⓘ button will be a no-op on every molview page"
    )
    assert "region-label-popover-init.js" in html, (
        "modify.html does not load lib/region-label-popover-init.js; "
        "the popover never wires to the workspace (the init code was "
        "moved out of an inline <script> block per the CSP contract)"
    )
    # The init module must call regionLabelDefinitions.init() so the
    # contract — "library binds via the namespace" — is honoured.
    init_js = (REPO_ROOT / "molbuilder" / "web" / "static" / "lib"
               / "region-label-popover-init.js").read_text()
    assert "regionLabelDefinitions" in init_js, (
        "region-label-popover-init.js does not reference "
        "regionLabelDefinitions; the popover will never wire up"
    )
    assert "defs.init(" in init_js, (
        "region-label-popover-init.js does not call defs.init(); "
        "the popover library is loaded but never bound to a getter"
    )


