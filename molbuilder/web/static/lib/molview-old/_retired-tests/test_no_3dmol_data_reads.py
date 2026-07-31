"""Guard: tests assert DATA via molview.data, NOT the concealed 3Dmol render target.

Why this file exists: the MolView concealment hides 3Dmol from *production* code
(no consumer imports it; everything goes through ``molview.data``).  But a
black-box Playwright test can still reach the raw viewer through the deliberate
``getViewer()`` / ``_viewer3dmol()`` escape hatch -- there is NO code boundary
that stops it.  THIS test is that boundary.

The trap: the render engine repaints 3Dmol on a DEFERRED double-``requestAnimationFrame``
(engine.js §8/§9).  A test that reads coordinates / elements off the drawn 3Dmol
atoms right after an op races the paint and sees the PRE-op state -- exactly the
rotate/element flake class.  The single source of truth is ``molview.data``
(``getElements()`` / ``getCoordinates()``), which every op updates synchronously.

The rule this enforces: every read of the 3Dmol render model
(``selectedAtoms`` / ``_viewer3dmol``) OUTSIDE the embed's own suite must be a
render-FACT read (serial / clickable / drawn-count / camera) justified inline
with a ``# 3dmol-ok: <why>`` marker within the preceding 6 lines.  A DATA read
(coords / elements) has no justification -- repoint it to ``molview.data``.
"""
import re
from pathlib import Path

import pytest

_TESTS = Path(__file__).parent

# The embed module's OWN suite IS the 3Dmol boundary -- it drives + reads the raw
# viewer to verify the seal itself, so it is exempt wholesale.
_EMBED_SUITE = {
    "test_mol_viewer_embed_e2e.py",
    "test_mol_viewer_embed_handle_surface_js.py",
}

# The render-model DATA primitives.  ``getView`` (camera) / ``shapes`` (overlay
# geometry) are render-STATE, a different (render-testing) category, and are not
# policed here -- only the drawn-atom / raw-viewer reads that leak coords/elements.
_FORBIDDEN = re.compile(r"\b(?:selectedAtoms|_viewer3dmol)\s*\(")
_MARKER = "3dmol-ok"
_WINDOW = 10   # lines of look-back for the justification marker (covers a
               # multi-line evaluate / JS-string block above the actual read)


def _policed_files():
    return sorted(
        p for p in _TESTS.glob("test_*.py")
        if p.name not in _EMBED_SUITE and p.name != Path(__file__).name
    )


@pytest.mark.parametrize("path", _policed_files(), ids=lambda p: p.name)
def test_no_unjustified_3dmol_render_reads(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    offenders = []
    for i, line in enumerate(lines):
        if not _FORBIDDEN.search(line):
            continue
        window = lines[max(0, i - _WINDOW): i + 1]
        if any(_MARKER in w for w in window):
            continue   # justified render-FACT read
        offenders.append((i + 1, line.strip()))
    assert not offenders, (
        f"{path.name}: {len(offenders)} unjustified 3Dmol render read(s). "
        f"Assert DATA via window.molbuilder.molview.data (getElements / "
        f"getCoordinates) -- the 3Dmol viewer is a deferred render target and "
        f"reading it races the repaint. A legitimate render-FACT read "
        f"(serial/clickable/drawn-count) must carry a '# 3dmol-ok: <why>' marker "
        f"within {_WINDOW} lines. Offenders: "
        + "; ".join(f"L{n}: {t}" for n, t in offenders)
    )
