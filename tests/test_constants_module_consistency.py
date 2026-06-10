"""Pin the lock-step between the classic-script constants module
(``lib/constants.js``) and the ES-module named exports in
``lib/projects/state.js``.

Two surfaces for the same string literals: classic-script consumers
(modify/viewer.js, results/viewer.js, lib/results/file-picker.js,
lib/inspectors/structure.js, lib/workspace/dispatcher.js, etc.)
read ``window.molbuilder.constants.SS_FILE``; ES-module consumers
(lib/projects/state.js, lib/projects/list.js) directly use the
``SS_FILE`` / ``SS_DIR`` named exports.  Both hold the same
literal values.  If one surface drifts (e.g. someone renames
``molbuilder.current_file`` to ``molbuilder.activeFile`` in the
ES-module side only), the round-trip silently breaks; this test
guards the convention so the drift surfaces in code review.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_JS = ROOT / "molbuilder/web/static/lib/constants.js"
STATE_JS     = ROOT / "molbuilder/web/static/lib/projects/state.js"


def _extract_constant(js_text: str, name: str) -> str:
    """Find ``<name>: "<value>"`` (constants.js) or
    ``export const <name>  = "<value>"`` (state.js) and return
    the value."""
    object_form = re.search(
        rf'{re.escape(name)}\s*:\s*"([^"]+)"', js_text)
    if object_form:
        return object_form.group(1)
    export_form = re.search(
        rf'export\s+const\s+{re.escape(name)}\s*=\s*"([^"]+)"',
        js_text)
    if export_form:
        return export_form.group(1)
    return ""


def test_constants_js_and_state_js_agree_on_SS_FILE():
    """Both surfaces must define ``SS_FILE`` and agree on the value."""
    constants_value = _extract_constant(
        CONSTANTS_JS.read_text(), "SS_FILE")
    state_value     = _extract_constant(
        STATE_JS.read_text(), "SS_FILE")
    assert constants_value, (
        "lib/constants.js must define SS_FILE — the classic-script "
        "consumers read ``window.molbuilder.constants.SS_FILE``"
    )
    assert state_value, (
        "lib/projects/state.js must export SS_FILE — the ES-module "
        "consumers ``import { SS_FILE } from ./state.js``"
    )
    assert constants_value == state_value, (
        f"SS_FILE drifted between the two surfaces:\n"
        f"  lib/constants.js   = {constants_value!r}\n"
        f"  lib/projects/state = {state_value!r}\n"
        f"Both must point at the same sessionStorage key; otherwise "
        f"the classic-script and ES-module consumers read different "
        f"slots."
    )


def test_constants_js_and_state_js_agree_on_SS_DIR():
    """Same lock-step for the SS_DIR key."""
    constants_value = _extract_constant(
        CONSTANTS_JS.read_text(), "SS_DIR")
    state_value     = _extract_constant(
        STATE_JS.read_text(), "SS_DIR")
    assert constants_value
    assert state_value
    assert constants_value == state_value, (
        f"SS_DIR drifted: constants={constants_value!r}, "
        f"state={state_value!r}"
    )


def test_constants_js_declares_event_names():
    """The custom event names are constants-only — ES-module
    consumers don't dispatch / listen for them today, so there's
    no second surface to keep in sync.  But pin the values
    against accidental rename."""
    text = CONSTANTS_JS.read_text()
    assert _extract_constant(text, "EVENT_FILE_SELECTED") \
        == "molbuilder:results:fileSelected"
    assert _extract_constant(text, "EVENT_INSPECTOR_READY") \
        == "molbuilder:inspector:ready"


def test_constants_js_declares_workspace_key():
    """The unified workspace sessionStorage key — pin the v1
    suffix so a future schema migration to v2 lands here as a
    visible doc/test update rather than a silent literal change
    that breaks restoration of older snapshots."""
    text = CONSTANTS_JS.read_text()
    assert _extract_constant(text, "SS_WORKSPACE") \
        == "molbuilder.workspace.v1"
