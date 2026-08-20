"""Persistence wiring pins — plan part 2A (2026-08-19).

Source-level pins for the three tabs whose truth-lane wiring was fixed.
The MECHANICS execute elsewhere: the restore-re-enters-HOLDING behavior
runs for real in test_molview_model.py
(test_a_restored_session_still_receives_frames), and the workspace door's
signature runs in test_workspace_tag_isolation.py.  What THESE pin is the
per-tab wiring that only reading the tab's source can see -- who calls the
restore, under which tag a note is written -- because each was individually
wrong on 2026-08-19: transport wrote drafts it never read back
(write-only persistence), structure-opt's note sat on the exact storage key
a history point 0 uses, and the demo's stand-in workspace still spoke the
pre-2026-08-02 four-argument surface, so its timeline was silently dead.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "molbuilder" / "web" / "static"

TRANSPORT = STATIC / "lib" / "transport" / "core.js"
STRUCTOPT = STATIC / "structure-optimization" / "viewer.js"
DEMO = STATIC / "lib" / "molview" / "demo.js"


def _stripped(path: Path) -> str:
    src = path.read_text()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"^\s*//.*$", "", src, flags=re.M)


def test_transport_reads_back_what_it_writes():
    """The tab restores at INIT — not in the lazy commit-mount, which can
    never fire on a reload — and its file note is written at commit under
    the tab's own tag (workspace.md § 4, the modify:panel pattern)."""
    src = _stripped(TRANSPORT)
    init_body = src.split("function _init()", 1)[1]
    assert "_restoreSession()" in init_body, (
        "transport never restores: drafts are written on every label edit "
        "and read back by nothing (write-only persistence)")
    # slice to the next TOP-LEVEL declaration (4-space indent) -- inline
    # anonymous functions are part of the body under test
    restore = src.split("function _restoreSession()", 1)[1].split(
        "\n    function ", 1)[0]
    assert ".load(0)" in restore, "the restore must adopt via load(0)"
    commit = src.split("async function _commit(", 1)[1].split(
        "\n    function ", 1)[0]
    assert "_writePanelNote(f)" in commit, (
        "the committed file is the tab's own fact and must be noted at the "
        "moment the tab commits it")
    assert 'WORKSPACE_TAG + ":panel"' in src, (
        "the note needs its OWN tag — the bare tag's identity is the "
        "viewer's")


def test_structure_opt_note_is_off_the_historys_key():
    """`{workspaceId(tag), state_index: 0}` is exactly where MolView's
    history writes point 0; the tab's note lives under its own :panel tag
    so the two can never share a file."""
    src = _stripped(STRUCTOPT)
    assert 'WORKSPACE_TAG + ":panel"' in src
    saved = src.split("const _SAVED", 1)[1].split(";", 1)[0]
    assert "PANEL_TAG" in saved, (
        "the note's identity still derives from the bare viewer tag — one "
        "mode flag away from two writers on one state file")
    remember = src.split("function _rememberStructure()", 1)[1].split(
        "function ", 1)[0]
    assert "ws.persist(PANEL_TAG" in remember


def test_the_demo_stand_in_speaks_the_real_persist_signature():
    """persist(tag, bytes, identity) — three arguments, the dispatcher's
    own surface.  The four-argument shape stored nothing (identity landed
    in the wrong slot) and the demo timeline was silently dead."""
    src = _stripped(DEMO)
    assert "persist(tag, bytes, identity)" in src
    assert "snapshotBlob" not in src
    assert "readPersistedSnapshot" not in src, (
        "no such member exists on the real surface — a stand-in shaped "
        "like a wish proves the wish")
