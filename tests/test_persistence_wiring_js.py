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

MOUNT = STATIC / "lib" / "molview" / "mount.js"
TRANSPORT = STATIC / "lib" / "transport" / "core.js"
STRUCTOPT = STATIC / "structure-optimization" / "viewer.js"
DEMO = STATIC / "lib" / "molview" / "demo.js"


def _stripped(path: Path) -> str:
    src = path.read_text()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"^\s*//.*$", "", src, flags=re.M)


def test_transport_reads_back_what_it_writes():
    """The tab restores at INIT, and its note is the CITATION (v2, P7b
    review 2026-08-29: the citation is the tab's one fact -- the viewer
    persists the structure itself), written at the moment of citing
    under the tab's own tag (workspace.md § 4, the modify:panel
    pattern)."""
    src = _stripped(TRANSPORT)
    init_body = src.split("function _init()", 1)[1]
    assert "_restoreSession()" in init_body, (
        "transport never restores: notes are written and read back by "
        "nothing (write-only persistence)")
    # slice to the next TOP-LEVEL declaration (4-space indent) -- inline
    # anonymous functions are part of the body under test
    restore = src.split("function _restoreSession()", 1)[1].split(
        "\n    function ", 1)[0]
    # No draft adoption: the viewer is READ-ONLY (molview.md § 9.4 --
    # load(0) is a documented no-op there), so the restore re-opens the
    # CITED file instead (§ 12.3, the inspector pattern; 2026-08-29).
    assert ".load(0)" not in restore, (
        "the restore adopts a draft again -- on the read-only mount "
        "that is a silent no-op (the inspector's 2026-08-03 bug)")
    assert "note.junction" in restore, (
        "the restore must re-adopt the cited junction")
    assert "_adoptCitation" in restore and "_describeAttempt" in restore, (
        "the restore must take the SAME path a pick takes -- re-describe "
        "the citation and adopt it (the server re-composes the structure "
        "fresh; molview.md § 12.3's reload, through the 4.1b seam)")
    adopt = src.split("function _adoptCitation(", 1)[1].split(
        "\n    function ", 1)[0]
    assert "_writePanelNote()" in adopt, (
        "the citation is the tab's own fact and must be noted at the "
        "moment it is adopted")
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


def test_every_mounted_viewer_gets_the_view_context():
    """mount.js attaches the lane (molview.md § 11.2b) for every viewer —
    one implementation, zero per-tab code.  The lane's behavior executes in
    test_molview_ui_context.py; this pins that mounting is where it is
    wired, with the truth-lane fact derived from the same mode flag the
    history gate reads."""
    src = _stripped(MOUNT)
    assert "attachUiContext({" in src
    assert 'hasTruthLane: opts.mode !== "readonly"' in src


