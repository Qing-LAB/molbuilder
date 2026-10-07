"""L2 source-text invariant: the optimization tab's record of its structure
is ``_sidebarLastFile`` (the committed load), NOT
``_proj.getCurrentFile()`` (the live sidebar pick).

Why this matters (2026-06-14 BDT incident): user loaded an .xyz with
a paired ``.molstruct.json`` sidecar carrying 25 frozen atoms, then
navigated the sidebar to a sibling directory to set ``dest_dir``,
then clicked Generate.  Pre-fix code did
``_proj.getCurrentFile()`` -> the LAST-CLICKED-IN-SIDEBAR file (the
dest-dir file, no sidecar there) -> server's
``apply_sidecar_if_possible`` looked for a sidecar next to the wrong
file -> found none -> emitted stage-2 .fdf with NO
``%block Geometry.Constraints`` -> SIESTA relaxed every atom
(including the Au surface that was meant to be frozen) -> the
Results-tab "constrained" force trace was identical to the
unconstrained trace (no fixed-atom plot).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
VIEWER_JS = REPO / "molbuilder/web/static/structure-optimization/viewer.js"


@pytest.fixture(scope="module")
def viewer_src() -> str:
    return VIEWER_JS.read_text(encoding="utf-8")


def test_the_committed_file_is_what_the_tab_records(viewer_src):
    """``_sidebarLastFile`` is the tab's record of what was COMMITTED.

    The INCIDENT the module docstring describes is about which file the tab
    believes it is holding, so the guard is stated on that,
    endpoint-independently.
    """
    assert "_sidebarLastFile" in viewer_src, (
        "viewer.js no longer tracks _sidebarLastFile.  Something has to hold "
        "'the file the user committed', distinct from 'the file highlighted "
        "in the sidebar right now' -- conflating them is the 2026-06-14 BDT "
        "frozen-atoms incident in the module docstring.")


def test_the_live_sidebar_pick_never_identifies_the_structure(viewer_src):
    """``getCurrentFile()`` is the LIVE pick and must not identify the
    structure being worked on.

    Exactly one use survives and it is the opposite case: seeding an EMPTY
    canvas at mount, where "what is highlighted" is the only thing there is to
    go on.  Any use inside a request body, or to decide which file's sidecar
    to read, is the incident again.
    """
    uses = [ln.strip() for ln in viewer_src.splitlines() if "getCurrentFile" in ln]
    assert len(uses) <= 2, (
        f"{len(uses)} uses of getCurrentFile() in viewer.js; expected only the "
        f"mount-time empty-canvas seed:\n  " + "\n  ".join(uses))
    joined = " ".join(uses)
    assert "_initialFile" in joined, (
        "the surviving getCurrentFile() call is no longer the mount-time seed "
        "(`_initialFile`).  If it moved into a request path, that is the "
        "2026-06-14 BDT frozen-atoms incident returning -- see the module "
        "docstring.")
