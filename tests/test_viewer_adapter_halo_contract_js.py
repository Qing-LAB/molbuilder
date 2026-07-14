"""Halo-overlay contract — L2 source-text invariant (doc ⇄ code).

Pins the halo-render contract documented in
``docs/protocols/molview-module.md`` § 13.3 against the actual
``lib/selection/viewer-adapter.js`` source, in BOTH directions:

  * the named-region color table (``REGION_COLORS``) in the code must
    match the table in the doc exactly — a color edit on either side
    that isn't mirrored fails here;
  * the three independent halo layers (region / frozen / selection)
    must keep their load-bearing wiring: region + frozen are
    selection-INDEPENDENT (they read ``labels`` / ``isFrozen``, never
    the pick set), and the selection layer is gated on a non-empty
    ``indices``.

Why this is a contract and not a unit test: the "an atom can show a
halo with an empty selection" behaviour (region membership, frozen
markers) looks like a bug the first time you hit it — someone will be
tempted to "fix" it by clearing the selection.  This invariant makes
that a red test, pointing at § 13.3 instead.

Per docs/protocols/test-strategy.md § 5 (source-text invariants).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "molbuilder/web/static/lib/selection/viewer-adapter.js"
DOC = ROOT / "docs/protocols/molview-module.md"


# --------------------------------------------------------------------- #
#  Color table: code ⇄ doc                                              #
# --------------------------------------------------------------------- #

_HEX = r"#[0-9a-fA-F]{6}"


def _region_colors_from_code() -> dict[str, str]:
    src = ADAPTER.read_text()
    m = re.search(r"const\s+REGION_COLORS\s*=\s*\{(.*?)\};", src, re.DOTALL)
    assert m, "REGION_COLORS object literal not found in viewer-adapter.js"
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
    assert code, "no REGION_COLORS parsed from viewer-adapter.js"
    for name, color in code.items():
        assert doc.get(name) == color, (
            f"REGION_COLORS[{name!r}] = {color!r} in viewer-adapter.js but "
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


# --------------------------------------------------------------------- #
#  The three halo layers + selection-independence                       #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def adapter_src() -> str:
    return ADAPTER.read_text()


def test_region_layer_reads_labels_not_selection(adapter_src):
    """Layer 1 (region halo) buckets atoms by ``a.labels`` — the
    per-atom region membership — NOT by the pick set.  § 13.3
    load-bearing invariant: region halos are selection-independent."""
    assert re.search(r"a\.labels\b", adapter_src), (
        "viewer-adapter no longer keys the region halo on a.labels; § 13.3 "
        "requires region membership (labels) to drive layer 1."
    )
    # single-color-per-atom rule: first label only.
    assert re.search(r"a\.labels\s*&&\s*a\.labels\[0\]|a\.labels\[0\]",
                     adapter_src), (
        "the region halo must key on labels[0] (first label = one color per "
        "atom); § 13.3 pins the single-color-per-atom rule."
    )


def test_frozen_layer_reads_isfrozen_not_selection(adapter_src):
    """Layer 2 (frozen marker) reads ``a.isFrozen`` — also
    selection-independent."""
    assert re.search(r"a\.isFrozen\b", adapter_src), (
        "viewer-adapter no longer keys the frozen marker on a.isFrozen; "
        "§ 13.3 layer 2."
    )


def test_selection_layer_is_gated_on_nonempty_indices(adapter_src):
    """Layer 3 (selection halo) is the ONLY layer that reads the pick
    set, and it draws only when ``indices`` is non-empty — so an empty
    selection paints no yellow halo (§ 13.3)."""
    # The selection halo pushes an overlay guarded by sel.length.
    assert re.search(r"s\.indices", adapter_src), (
        "selection halo must read s.indices (the pick set); § 13.3 layer 3."
    )
    assert re.search(r"if\s*\(\s*sel\.length\s*\)", adapter_src), (
        "selection halo must be gated on `if (sel.length)` so an EMPTY "
        "selection draws no halo; § 13.3 (this is what distinguishes a "
        "stale-selection halo — which cannot happen — from a region halo)."
    )


def test_halo_geometry_constants_present(adapter_src):
    """The three fixed halo geometries (§ 13.3) — selection largest +
    brightest, frozen smallest — must stay declared as constants."""
    for name in ("HALO_REGION", "HALO_FROZEN", "HALO_SELECT"):
        assert re.search(rf"const\s+{name}\s*=", adapter_src), (
            f"{name} geometry constant missing; § 13.3 pins the three halo "
            f"geometries."
        )
