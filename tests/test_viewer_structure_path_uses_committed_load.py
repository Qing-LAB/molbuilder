"""L2 source-text invariant: the optimization-tab Generate handlers
must read ``_sidebarLastFile`` (the committed load) when populating
``structure_path``, NOT ``_proj.getCurrentFile()`` (the live sidebar
pick).

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

The fix is one-line: read ``_sidebarLastFile`` instead, which stays
pinned to the file the user committed via Load / dblclick.  This
test pins both call sites against the bug by source-text
inspection — cheap (no browser) and catches the regression at
the file-source level the next time someone refactors the click
handler.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
VIEWER_JS = REPO / "molbuilder/web/static/viewer.js"


@pytest.fixture(scope="module")
def viewer_src() -> str:
    return VIEWER_JS.read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
#  SIESTA: /api/build/fdf handler                                        #
# --------------------------------------------------------------------- #


def test_siesta_generate_uses_sidebar_last_file(viewer_src):
    """The block that constructs the /api/build/fdf POST body must
    assign ``_structPath`` from ``_sidebarLastFile``, not from
    ``_proj.getCurrentFile()``."""
    # Locate the SIESTA POST.
    post_ix = viewer_src.find('fetch("/api/build/fdf"')
    assert post_ix > 0, "SIESTA POST not found in viewer.js"

    # Search backwards in a reasonable window for the structPath
    # assignment.  100 lines is comfortably more than the prelude
    # but tight enough that a refactor moving the var out of scope
    # would fail this anchor.
    window = viewer_src[max(0, post_ix - 4000):post_ix]
    assigns = re.findall(
        r"const\s+_structPath\s*=\s*([^;]+);",
        window,
    )
    assert assigns, (
        "Could not find ``const _structPath = ...;`` before the "
        "/api/build/fdf POST.  Did you rename the variable?  "
        "Update this test along with the rename."
    )
    rhs = assigns[-1]
    assert "_sidebarLastFile" in rhs, (
        f"_structPath must be derived from _sidebarLastFile (the "
        f"committed Load file).  Got: ``{rhs.strip()}``.  See module "
        f"docstring for the BLOCKER this guards against."
    )
    assert "getCurrentFile" not in rhs, (
        f"_structPath must NOT read ``getCurrentFile()`` (the live "
        f"sidebar pick).  Got: ``{rhs.strip()}``."
    )


# --------------------------------------------------------------------- #
#  PySCF: /api/build/pyscf handler                                       #
# --------------------------------------------------------------------- #


def test_pyscf_generate_uses_sidebar_last_file(viewer_src):
    """Mirror of the SIESTA assertion for the PySCF generate handler.
    Same bug class -- the PySCF POST also needs the committed-load
    path so the sidecar applies."""
    post_ix = viewer_src.find('fetch("/api/build/pyscf"')
    assert post_ix > 0, "PySCF POST not found in viewer.js"

    window = viewer_src[max(0, post_ix - 4000):post_ix]
    assigns = re.findall(
        r"const\s+_structPathPy\s*=\s*([^;]+);",
        window,
    )
    assert assigns, (
        "Could not find ``const _structPathPy = ...;`` before the "
        "/api/build/pyscf POST.  Did you rename the variable?"
    )
    rhs = assigns[-1]
    assert "_sidebarLastFile" in rhs, (
        f"_structPathPy must be derived from _sidebarLastFile.  "
        f"Got: ``{rhs.strip()}``."
    )
    assert "getCurrentFile" not in rhs, (
        f"_structPathPy must NOT read ``getCurrentFile()``.  "
        f"Got: ``{rhs.strip()}``."
    )


# --------------------------------------------------------------------- #
#  Sanity: the variable being read is actually declared in scope         #
# --------------------------------------------------------------------- #


def test_sidebar_last_file_declared_at_module_scope(viewer_src):
    """``_sidebarLastFile`` must be declared inside the single
    top-level IIFE the file wraps.  If a future refactor moves it
    into an inner function (or deletes it), the generate handlers
    would silently get ``undefined`` -> falsy ``|| ""`` fallback
    -> empty structure_path -> sidecar never applied (the exact
    pre-fix symptom).  The test verifies the declaration is
    present at module-ish scope and not inside a nested function."""
    assert re.search(
        r"^    let\s+_sidebarLastFile\s*=", viewer_src, re.MULTILINE
    ) is not None, (
        "``let _sidebarLastFile = …`` declaration not found at "
        "single-IIFE scope (4-space indent).  The optimization-tab "
        "generate handlers depend on this variable being in scope."
    )
