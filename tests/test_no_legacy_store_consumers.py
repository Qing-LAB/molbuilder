"""Workspace-contract.md §8 compliance test.

Enforces that no consumer outside the workspace-dispatcher internals
references the legacy globals ``window.molbuilder.structureCanvas``
or ``window.molbuilder.selection.store``.

Per workspace-contract.md §8, the only legitimate readers of those
globals are:

  * lib/workspace/dispatcher.js — delegates to them as internal
    implementation
  * lib/molview/_canvas-state-impl.js — Phase 9 (2026-06-13)
    moved out of lib/structure/canvas-state.js; no longer mounts
    the structureCanvas global, but places the singleton on the
    private workspace._canvasState slot the dispatcher consumes
  * lib/molview/_selection-store-impl.js — Phase 9 (2026-06-13)
    moved out of lib/selection/store.js; no longer mounts the
    selection.store global, but exposes the _createStore factory
    the dispatcher consumes

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
    "lib/molview/data-model.js",
    "lib/molview/_canvas-state-impl.js",
    "lib/molview/_selection-store-impl.js",
    # The runtime registry's module docstring lists every mounted
    # name including the deprecated ones.  The list is documentation,
    # not consumption.
    "lib/molbuilder-runtime.js",
}

# Matches direct reads of the three deprecated globals listed in
# workspace-contract.md §8:
#   * ``window.molbuilder.structureCanvas``
#   * ``window.molbuilder.selection.store``
#   * ``window.molbuilder.modify.state`` (the modify-tab IIFE state)
# Also matches the ``root.``/``mb.``/``globalThis.`` variants.
# Comments and "Internal as of Phase 9" banner lines are stripped
# before pattern matching.
LEGACY_PATTERN = re.compile(
    r"""\b(window|root|mb|globalThis)\.molbuilder\.
        (structureCanvas|selection\.store|modify\.state)\b""",
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


# --------------------------------------------------------------------- #
#  L1 tests for the scanner internals — added 2026-06-14 per task #390  #
#                                                                       #
#  ``_is_in_comment`` is documented as "approximate" — these tests pin  #
#  the exact behaviour so a future tightening (or a regression that    #
#  silently broadens it) surfaces immediately.  Comment-detection is   #
#  what gates whether a legitimate code reference becomes a false       #
#  positive (or vice versa) in the main compliance check.               #
# --------------------------------------------------------------------- #


class TestIsInComment:
    """The approximate comment detector at line 71.

    The function returns True when a ``//`` or an unclosed ``/*``
    appears in the line BEFORE ``match_start``.  Cases the docstring
    promises to handle:

      * ``//`` anywhere before the match → True (line comment that
        started earlier)
      * ``/*`` before the match without a closing ``*/`` between
        them → True (multi-line block-comment open)
      * ``/*`` ... ``*/`` BOTH before the match → False (block
        comment closed before the match)
      * Neither marker before the match → False (real code)
    """

    def test_no_markers_means_not_in_comment(self):
        line = "    foo = window.molbuilder.structureCanvas;"
        # ``window.molbuilder.structureCanvas`` starts at index 10.
        assert _is_in_comment(line, 10) is False

    def test_line_comment_before_match(self):
        line = "    // foo = window.molbuilder.structureCanvas;"
        # The match would land at index ~16; ``//`` at index 4.
        assert _is_in_comment(line, 16) is True

    def test_line_comment_in_middle(self):
        """``// stuff`` mid-line still gates the rest of the line."""
        line = "    foo();  // window.molbuilder.structureCanvas"
        assert _is_in_comment(line, 16) is True

    def test_block_comment_unclosed_before_match(self):
        """``/*`` started, no ``*/`` before the match → in comment."""
        line = "    /* TODO: window.molbuilder.structureCanvas"
        assert _is_in_comment(line, 18) is True

    def test_block_comment_closed_before_match(self):
        """``/* ... */`` fully closed before the match → not in
        comment.  This is the case where the doc says "approximate"
        and a more thorough parser might still flag it, but our
        contract is to treat closed comments as having ended."""
        line = "    /* prefix */ window.molbuilder.structureCanvas"
        # Match starts at index 17 (right after ``*/ ``).
        assert _is_in_comment(line, 17) is False

    def test_match_at_index_zero(self):
        """Edge: match at index 0 → no prefix to scan → not in
        comment (defensive default)."""
        line = "window.molbuilder.structureCanvas = 1"
        assert _is_in_comment(line, 0) is False

    def test_two_block_comments_before_match(self):
        """``/* a */ /* b */`` — both closed → not in comment."""
        line = "  /* one */ /* two */ window.molbuilder.structureCanvas"
        # The function's approximation: it counts ANY ``/*`` and ANY
        # ``*/`` in the prefix.  Two of each → still balanced → not
        # in comment.  Pinning this so future tightening doesn't
        # silently change behavior on real lines like this.
        prefix = line[: line.find("window")]
        assert "/*" in prefix and "*/" in prefix
        assert _is_in_comment(line, line.find("window")) is False


class TestScanFile:
    """End-to-end check that the file-level scanner correctly
    distinguishes real consumer code from comments + strings.  The
    scanner is the load-bearing check for the workspace-contract.md
    § 8 compliance contract."""

    def _make_file(self, tmp_path, body: str) -> Path:
        path = tmp_path / "snippet.js"
        path.write_text(body, encoding="utf-8")
        return path

    def test_real_consumer_is_flagged(self, tmp_path):
        path = self._make_file(tmp_path, """\
function read() {
    return window.molbuilder.structureCanvas.getText();
}
""")
        hits = _scan_file(path)
        assert len(hits) == 1
        assert "structureCanvas" in hits[0][1]

    def test_full_line_comment_is_skipped(self, tmp_path):
        path = self._make_file(tmp_path, """\
function read() {
    // window.molbuilder.structureCanvas is gone post-Phase-9;
    return window.molbuilder.workspace.getStructure();
}
""")
        hits = _scan_file(path)
        assert hits == [], (
            "full-line ``// …`` comment must be skipped; "
            f"got {hits!r}"
        )

    def test_jsdoc_star_line_is_skipped(self, tmp_path):
        """Lines whose first non-whitespace token is ``*`` are
        treated as JSDoc continuation lines — common in headers."""
        path = self._make_file(tmp_path, """\
/**
 * window.molbuilder.structureCanvas — retired 2026-06-13.
 */
function noop() {}
""")
        hits = _scan_file(path)
        assert hits == []

    def test_trailing_inline_comment_is_skipped(self, tmp_path):
        path = self._make_file(tmp_path, """\
function noop() {
    return null;  // window.molbuilder.structureCanvas is gone
}
""")
        hits = _scan_file(path)
        assert hits == []

    def test_multiple_consumers_all_flagged(self, tmp_path):
        path = self._make_file(tmp_path, """\
function a() { return window.molbuilder.structureCanvas; }
function b() { return root.molbuilder.selection.store; }
""")
        hits = _scan_file(path)
        assert len(hits) == 2
        assert any("structureCanvas" in h[1] for h in hits)
        assert any("selection.store" in h[1] for h in hits)


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


# --------------------------------------------------------------------- #
#  POST-CARVE guard (2026-07): the workspace is the PERSISTENCE layer;   #
#  the in-memory DATA model is window.molbuilder.molview.data.  Reading  #
#  a DATA method off window.molbuilder.workspace.* is a contract breach  #
#  (workspace-contract.md §4).  This is the automated enforcement of     #
#  "the workspace has zero data methods."                                #
# --------------------------------------------------------------------- #

# Methods that moved OFF the workspace onto molview.data.  (Persistence methods --
# persist / workspaceId / readPersistedSnapshot / mountRestoreTarget / STORAGE_KEY
# -- are NOT here; they legitimately stay on the workspace.)
_WORKSPACE_DATA_METHODS = (
    "getStructure|getAtoms|getSelection|getCoordinates|getElements|getUnitCell|"
    "getUnitCellInfo|getLattice|getKgrid|getKgridInfo|getAxisKind|getAxisKindInfo|"
    "getVacuum|getVacuumInfo|getFrozen|getRegions|getAtomsByLabel|atomFor3Dmol|"
    "toAddAtoms|getSource|getSourceFile|getLastSavedTo|getState|isDirty|isEmpty|"
    "installStructure|applyOp|applyPayload|setUnitCell|setLattice|setKgrid|"
    "setAxisKind|setVacuum|setLabel|commitPeriodicity|loadFromFile|loadFromText|"
    "generate|discard|undo|save|setFrame|addFrame|addFrames|reloadFrames|getFrame|"
    "getForces|currentForces|currentFrame|frameCount|getScratchBlob|draftIdentity|"
    "suspendPersist|resumePersist|markDirty|markSaved|selection|view"
)
WORKSPACE_DATA_PATTERN = re.compile(
    r"\b(window|root|mb|globalThis)\.molbuilder\.workspace\."
    r"(?:" + _WORKSPACE_DATA_METHODS + r")\b"
)

# Tab consumers that once read data off the workspace persistence layer -- all
# migrated to molbuilder.molview.data (task #42 complete: selection-panel.js,
# selection-bootstrap.js, and modify/viewer.js were the last three, now rewired).
# The set is empty, so the guard below is now absolute: ANY read of
# molbuilder.workspace.<data-method> fails.  The second assertion keeps this set
# from silently regrowing -- a re-added entry that no longer matches is flagged.
WORKSPACE_DATA_MIGRATING: set[str] = set()


def _scan_file_for(path: Path, pattern) -> list[tuple[int, str]]:
    hits = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        m = pattern.search(line)
        if not m or _is_in_comment(line, m.start()):
            continue
        hits.append((line_no, line.rstrip()))
    return hits


def test_data_is_not_read_off_the_workspace_persistence_layer():
    """POST-CARVE drift-guard (workspace-contract.md §4): the workspace is the
    persistence layer; read data from molbuilder.molview.data, NOT
    molbuilder.workspace.*.  Only the KNOWN tab consumers being migrated (task #42)
    may still do so; a NEW such read fails here.  The migrating set is also kept
    EXACT, so it can only shrink."""
    actual: dict[str, list] = {}
    for path in STATIC.rglob("*.js"):
        rel = path.relative_to(STATIC).as_posix()
        hits = _scan_file_for(path, WORKSPACE_DATA_PATTERN)
        if hits:
            actual[rel] = hits
    offenders = set(actual) - WORKSPACE_DATA_MIGRATING
    assert not offenders, (
        "workspace-contract.md §4 violation: read data from molbuilder.molview.data, "
        "not molbuilder.workspace.* (the workspace is persistence-only):\n\n"
        + "\n".join(f"{f}:{ln}: {l.strip()}"
                    for f in sorted(offenders) for ln, l in actual[f]))
    finished = WORKSPACE_DATA_MIGRATING - set(actual)
    assert not finished, (
        "WORKSPACE_DATA_MIGRATING is stale -- task #42 finished these; remove them "
        f"from the set: {sorted(finished)}")
