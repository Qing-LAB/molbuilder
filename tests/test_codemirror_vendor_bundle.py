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
    ``lib/codemirror-load.js`` on 2026-08-16, when the Task Setup tab
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
            f"{cmd!r} — preview.js / task-setup wiring that depends on this "
            f"command will silently no-op.  Either restore the "
            f"addon's command registration or update preview.js + "
            f"this test's _ADDON_COMMANDS mapping."
        )


# --------------------------------------------------------------------- #
#  Language modes (added 2026-08-16)                                     #
# --------------------------------------------------------------------- #

_MODE_RE = re.compile(r'file:\s*"([^"]+\.min\.js)"')


def _declared_modes() -> dict[str, str]:
    """Parse ``CM_MODES`` in the shared loader for name -> file."""
    src = LOADER_JS.read_text()
    block = src[src.index("export const CM_MODES"):src.index("const SUFFIX_MODE")]
    out = {}
    for line in block.splitlines():
        m = re.match(r'\s*([a-z]+):\s*\{\s*file:\s*"([^"]+)"', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_every_declared_mode_is_on_disk_and_registers_itself():
    """A mode the loader names but the bundle lacks fails SILENTLY — the file
    404s, the mode never registers, and CodeMirror falls back to plain text
    with no error anywhere.  That is the whole failure class this catches."""
    modes = _declared_modes()
    assert modes, "CM_MODES no longer parses; update this test's parser"
    for name, filename in modes.items():
        path = VENDOR / filename
        assert path.is_file(), f"{filename} (mode {name!r}) is not vendored"
        assert path.stat().st_size > 0, f"{filename} is empty"
        assert "defineMode(" in path.read_text(), (
            f"{filename} carries no defineMode() — it is not a CodeMirror mode")


def test_markdowns_xml_dependency_is_declared_and_vendored():
    """CM5's markdown mode head is ``require("../xml/xml")``.  Without xml the
    mode half-registers and inline HTML stops highlighting — silently."""
    assert 'require("../xml/xml")' in (VENDOR / "markdown.min.js").read_text(), (
        "markdown.min.js no longer requires xml; the declared dependency in "
        "CM_MODES may now be wrong")
    src = LOADER_JS.read_text()
    assert re.search(r'markdown:\s*\{[^}]*needs:\s*\[\s*"xml"\s*\]', src), (
        "markdown's xml dependency is not declared in CM_MODES")
    assert (VENDOR / "xml.min.js").is_file(), "xml mode is not vendored"


def test_the_vendor_inventory_lists_the_modes():
    """`static/vendor/README.md` is the notice + inventory; a vendored file
    that is not in it is a licence-tracking gap (the repo's own rule)."""
    readme = (STATIC / "vendor/README.md").read_text()
    for filename in _declared_modes().values():
        assert filename in readme, f"{filename} is vendored but not in the inventory"


def test_molbuilders_own_formats_are_deliberately_plain_text():
    """`.fdf` / `.xyz` / `.out` have no upstream mode.  If someone maps one to
    a mode that does not exist, the file 404s and the failure is silent."""
    src = LOADER_JS.read_text()
    block = src[src.index("const SUFFIX_MODE"):src.index("export function modeForPath")]
    for absent in (".fdf", ".xyz", ".out", ".STRUCT_OUT"):
        assert f'"{absent}"' not in block, (
            f"{absent} is mapped to a mode; CodeMirror has none for it")
