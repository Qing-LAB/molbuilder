"""Workspace-contract.md §8 compliance test.

Enforces that no consumer outside the workspace-dispatcher internals
references the legacy globals ``window.molbuilder.structureCanvas``
or ``window.molbuilder.selection.store``.

Per workspace-contract.md §8, the only legitimate readers of those
globals are:

  * lib/workspace/dispatcher.js — delegates to them as internal
    implementation
  * lib/structure/canvas-state.js — self-mounts the canvas global
  * lib/selection/store.js — self-mounts the selection store global

Every other consumer must go through ``window.molbuilder.workspace.*``
(``ws.*``).  A new file matching the pattern outside the allow-list
is a contract violation; fix the file (route through ws.*) rather
than expanding the allow-list.

This test is the canonical compliance signal for Phase 10 Fix 4 —
when it passes, the migration is complete and the legacy globals
are safe to make private (or delete) in a follow-up.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"

# Files that ARE allowed to mention the legacy globals — they are
# the implementation modules themselves OR docstrings tracking the
# migration history.
ALLOW_LIST = {
    # The dispatcher reads the legacy stores as its internal
    # implementation; the migration target is to delete or rename
    # these modules + inline their logic into the dispatcher, but
    # the dispatcher itself is permitted to know about them.
    "lib/workspace/dispatcher.js",
    "lib/structure/canvas-state.js",
    "lib/selection/store.js",
    # The runtime registry's module docstring lists every mounted
    # name including the deprecated ones.  The list is documentation,
    # not consumption.
    "lib/molbuilder-runtime.js",
}

# Matches direct reads — ``window.molbuilder.structureCanvas`` or
# ``window.molbuilder.selection.store`` or the ``root.``/``mb.`` variants.
# Comments and the docstring "Internal as of Phase 9" banners are
# stripped before pattern matching.
LEGACY_PATTERN = re.compile(
    r"""\b(window|root|mb|globalThis)\.molbuilder\.
        (structureCanvas|selection\.store)\b""",
    re.VERBOSE,
)


def _is_in_comment(line: str, match_start: int) -> bool:
    """Approximate: line had a ``//`` or ``/*`` before the match."""
    pre = line[:match_start]
    if "//" in pre:
        return True
    if "/*" in pre and "*/" not in pre:
        return True
    return False


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return [(line_no, line)] of legacy-global references that are
    NOT inside a comment."""
    hits = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        # Skip lines that are entirely comments.
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        m = LEGACY_PATTERN.search(line)
        if not m:
            continue
        if _is_in_comment(line, m.start()):
            continue
        hits.append((line_no, line.rstrip()))
    return hits


def test_no_legacy_store_consumers_outside_allow_list():
    """No consumer file outside the implementation allow-list may
    reference ``window.molbuilder.structureCanvas`` or
    ``window.molbuilder.selection.store``.

    See workspace-contract.md §8.  If this test fails, the failing
    file should be migrated to use ``window.molbuilder.workspace.*``
    (=ws.*) instead.  Adding the file to the allow-list is the wrong
    fix — the allow-list represents implementation modules, not
    "this is too hard to migrate today".
    """
    violations: list[str] = []
    for path in STATIC.rglob("*.js"):
        rel = path.relative_to(STATIC).as_posix()
        if rel in ALLOW_LIST:
            continue
        # Skip vendored / minified.
        if "vendor/" in rel or rel.endswith(".min.js"):
            continue
        hits = _scan_file(path)
        for line_no, line in hits:
            violations.append(f"{rel}:{line_no}: {line.strip()}")

    assert not violations, (
        "workspace-contract.md §8 violation: consumer files outside the "
        "ALLOW_LIST reference the legacy structureCanvas / selection.store "
        "globals.  Migrate each call site to ws.* per the contract.\n\n"
        + "\n".join(violations)
    )


def test_allow_list_files_exist():
    """Every file in ALLOW_LIST must actually exist — guards against
    typos in the allow-list outliving a rename."""
    missing = []
    for rel in ALLOW_LIST:
        if not (STATIC / rel).exists():
            missing.append(rel)
    assert not missing, (
        f"ALLOW_LIST contains paths that no longer exist: {missing}.  "
        "Remove them from the list."
    )
