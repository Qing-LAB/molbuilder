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
