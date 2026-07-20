"""Halo-overlay palette contract — L2 source-text invariant (doc ⇄ code).

Pins the named-region color table documented in
``docs/protocols/molview-module.md`` § 13.3 against the actual source,
in BOTH directions.  The palette authority now lives in the render
ENGINE (``lib/molview/engine/process.js``), which owns all halo /
overlay derivation (region tints / frozen markers / selection halo)
per molview-render-streamline.md § 2.4/§ 7.3 — the selection
viewer-adapter no longer paints, so the ``REGION_COLORS`` table moved
with it.

  * the named-region color table (``REGION_COLORS``) in process.js
    must match the § 13.3 table exactly — a color edit on either side
    that isn't mirrored fails here.

Why this is a contract and not a unit test: the color IS the
transport/junction vocabulary (L-electrode green, etc.), so a drift on
either side is a real regression, pointing at § 13.3 instead.  (The
region / frozen / selection LAYERING is exercised end-to-end in
tests/test_engine_process_js.py.)

Per docs/protocols/test-strategy.md § 5 (source-text invariants).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
PROCESS = ROOT / "molbuilder/web/static/lib/molview/engine/process.js"
DOC = ROOT / "docs/protocols/molview-module.md"


# --------------------------------------------------------------------- #
#  Color table: code ⇄ doc                                              #
# --------------------------------------------------------------------- #

_HEX = r"#[0-9a-fA-F]{6}"


def _region_colors_from_code() -> dict[str, str]:
    src = PROCESS.read_text()
    m = re.search(r"(?:const|var)\s+REGION_COLORS\s*=\s*\{(.*?)\};", src, re.DOTALL)
    assert m, "REGION_COLORS object literal not found in process.js"
    pairs = re.findall(rf'"([^"]+)"\s*:\s*"({_HEX})"', m.group(1))
    return {name: color.lower() for name, color in pairs}


def _region_colors_from_doc() -> dict[str, str]:
    text = DOC.read_text()
    # § 13.3 table rows:  | `L-electrode` | `#7fc97f` (green) |
    rows = re.findall(rf"\|\s*`([^`]+)`\s*\|\s*`({_HEX})`", text)
    return {name: color.lower() for name, color in rows}


def test_named_region_colors_code_matches_doc():
    """The REGION_COLORS map in the code must equal the § 13.3 table.

    A drift in EITHER direction fails: change a color in the code and
    the doc table no longer matches; edit the doc and the code no
    longer matches.  The color IS the transport/junction vocabulary
    (L-electrode green, etc.), so it is a real contract, not styling."""
    code = _region_colors_from_code()
    doc = _region_colors_from_doc()
    # The doc table may live among other tables; require that every
    # code entry appears in the doc with the same color, and that the
    # four named regions are all present (no silent drop).
    assert code, "no REGION_COLORS parsed from process.js"
    for name, color in code.items():
        assert doc.get(name) == color, (
            f"REGION_COLORS[{name!r}] = {color!r} in process.js but "
            f"molview-module.md § 13.3 says {doc.get(name)!r}.  Update the "
            f"doc table and the code together — the named-region palette is "
            f"a contract."
        )
    # Guard the known transport vocabulary is intact (catches an
    # accidental deletion that would keep code==doc but shrink both).
    for required in ("L-electrode", "R-electrode", "bridge", "interface"):
        assert required in code, (
            f"named region {required!r} dropped from REGION_COLORS; § 13.3 "
            f"pins the transport/junction palette."
        )
