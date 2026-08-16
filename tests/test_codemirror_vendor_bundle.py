"""Vendored CodeMirror bundle integrity — L2 source-text invariant.

The preview-modal editor (``lib/projects/preview.js``) injects six
vendored files at first use: the core, two CSS files, and three
addons (dialog, searchcursor, search, jump-to-line).  The wiring
in preview.js depends on:

  * ``window.CodeMirror.commands.find``  — Ctrl-F dialog
  * ``window.CodeMirror.commands.jumpToLine``  — Alt-G dialog

A vendor bundle update that ships only the core, or drops one of
the addons, breaks the modal's search + jump bindings.  A previous
generation of these checks lived as two end-to-end Playwright
tests that spun up Chromium just to call
``typeof window.CodeMirror.commands.find === 'function'`` after
loading the bundle — pure existence checks of vendored code.

Per docs/process/testing.md (source-text invariants):
verifying that a vendored file is present + carries the symbol
the host code depends on is the canonical L2 shape.  No browser,
no JS runtime — just file existence + grep over the minified
source.

This test pulls the asset list directly from preview.js so any
future asset added to the bundle is automatically required.  The
addon → command mapping is hand-maintained: adding a third addon
should add an entry to ``_ADDON_COMMANDS``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"
VENDOR = STATIC / "vendor/codemirror"
LOADER_JS = STATIC / "lib/codemirror-load.js"


_INJECT_RE = re.compile(
    r'injectS(?:cript|tylesheet)\(\s*CM_VENDOR_BASE\s*\+\s*"([^"]+)"'
)


# Each addon registers one or more commands on
# ``CodeMirror.commands.X`` at parse time.  The minified source
# carries these as the literal ``commands.X`` token; a future
# vendor update that strips the command would silently break the
# host wiring (Ctrl-F / Alt-G no-ops).
_ADDON_COMMANDS: dict[str, list[str]] = {
    "search.min.js": ["commands.find"],
    "jump-to-line.min.js": ["commands.jumpToLine"],
}


def _expected_assets() -> list[str]:
    """Parse the shared loader's ``loadCodeMirror`` for every vendor
    file it injects.  The asset list is the source of truth — adding a
    bundle file there automatically extends this test's coverage.

    The loader moved out of ``lib/projects/preview.js`` into
    ``lib/codemirror-load.js`` on 2026-08-16, when the Job Prep tab
    needed the same editor: two lazy-loaders would be two places for
    this list to drift, which is the drift this test exists to catch.
    """
    src = LOADER_JS.read_text()
    assets = _INJECT_RE.findall(src)
    if not assets:
        pytest.fail(
            "codemirror-load.js no longer matches the injectScript / "
            "injectStylesheet(CM_VENDOR_BASE + ...) pattern; "
            "update this test's parser."
        )
    return assets


def test_every_injected_asset_exists_and_is_nonempty():
    """the shared loader lists six vendor files.  Each
    one must be on disk under ``static/vendor/codemirror/`` AND
    non-empty (a 0-byte file would silently break the bundle)."""
    missing: list[str] = []
    empty: list[str] = []
    for asset in _expected_assets():
        path = VENDOR / asset
        if not path.exists():
            missing.append(asset)
            continue
        if path.stat().st_size == 0:
            empty.append(asset)
    assert not missing, (
        f"Vendor files referenced by codemirror-load.js are missing from "
        f"{VENDOR}: {missing}.  The CodeMirror bundle update either "
        f"dropped them or the path/filename changed."
    )
    assert not empty, (
        f"Vendor files are zero-bytes: {empty}.  A stub file silently "
        f"breaks the bundle at runtime; restore the real vendor file."
    )


@pytest.mark.parametrize(
    "addon,commands",
    [
        pytest.param(addon, cmds, id=addon)
        for addon, cmds in _ADDON_COMMANDS.items()
    ],
)
def test_addon_registers_expected_commands(addon, commands):
    """Each addon must register the command the preview-modal UI
    binds to (Ctrl-F → find, Alt-G → jumpToLine).  Catches a
    future vendor update that ships the addon file with a
    different export surface."""
    src = (VENDOR / addon).read_text()
    for cmd in commands:
        assert cmd in src, (
            f"Vendor addon {addon} does not contain the literal "
            f"{cmd!r} — preview.js / job-prep wiring that depends on this "
            f"command will silently no-op.  Either restore the "
            f"addon's command registration or update preview.js + "
            f"this test's _ADDON_COMMANDS mapping."
        )
