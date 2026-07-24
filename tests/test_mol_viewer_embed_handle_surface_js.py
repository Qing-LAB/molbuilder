"""Embed handle-surface + View-menu + overlay contract — L2 source-text invariants.

These are the STATIC (no-browser, no-Node) pins for the concealed 3Dmol
drawing layer that MolView draws through
(``molbuilder/web/static/lib/mol-viewer-embed.js``).  They read the module
source and assert the shape of three contracts that are fixed at the
source-code level:

  * PIN 2 — the handle's method set.  ``create()`` ends in a literal
    ``return { name: value, … }`` block that lists every exported method
    in source order.  This test extracts those keys and asserts the exact
    current surface, and that the RETIRED force/index doors
    (``showForces`` / ``showIndices``) are gone.

  * PIN 3 — the View menu (the embed's knob bar, ``_buildViewMenu``):
    "Reset view" is the FIRST item, then ONE untitled
    ``.mol-viewer-menu-toggles`` group holding axes / labels / overlay /
    unit-cell toggles with no per-toggle headings (molview-module.md
    §14.5 / View-menu contract).

  * PIN 4 (static half) — overlays are CONSUMER-HANDED.  The embed draws
    the arrows it is given via ``setArrows`` and gates their visibility on
    ``state.current.overlayOn``; it never reads force data and never
    synthesises force arrows (molview-module.md §14.5.1).

Per docs/protocols/test-strategy.md §5 (source-text invariants): the
handle's shape and the menu's structure are code-level contracts, so
grepping the source is the right level of abstraction.  The runtime
(Node) half of the normalisation contract lives in
``test_mol_viewer_embed_js.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module   # L2 — source-text invariant

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/mol-viewer-embed.js"


# --------------------------------------------------------------------- #
#  PIN 2 — the handle method set                                        #
# --------------------------------------------------------------------- #

# The handle's current documented method set — every function on the
# object ``create()`` returns.  Adding or removing a method requires
# updating BOTH the embed's return block and this list (intentional
# friction).  Drawn verbatim from mol-viewer-embed.js's handle return
# block + molview-module.md §14.5 (setArrows/setLabels are the overlay
# doors; setCell/setAxes/setStyle are the chrome doors).
EXPECTED_METHODS = sorted([
    # Data setters
    "setStructure", "appendFrames",
    # Style + overlays + projection
    "setStyle", "setAxes", "setCell", "setLabels", "setAtomLabels",
    "setArrows", "setOverlay", "setPick", "setBackground",
    "setOverlays", "setAtomStyle", "setProjection",
    # Camera
    "getCamera", "setCamera",
    # Knob bar + injected view toggles (molview adds isolate via this)
    "setKnobs", "addViewToggle",
    # Busy<->ready surface (the ONE loading/render scrim)
    "setBusy",
    # Animation control
    "setAnimation", "playAnimation", "pauseAnimation",
    "isAnimationPlaying", "setAnimationFrame", "getAnimationFrame",
    "appendFrameArrows",
    "getAnimationKind", "getFrameCount", "getFrameCoords",
    # Read accessors
    "getAtomCount", "getElements", "getAtomCoords",
    "getPickedIndices", "setPickedIndices",
    "getStructureText",
    # Declarative-state getters (round-trip with setX)
    "getStyle", "getAxes", "getCell", "getLabels",
    "getOverlays", "getPick", "getKnobs", "getArrows",
    "getOverlay", "getAnimation", "getBackground", "getLattice",
    # Ordered batch runner
    "applyState",
    # Output / export
    "screenshot", "exportData",
    "captureFrames", "exportAnimation",
    # Lifecycle
    "refit", "setPivot", "render", "dispose",
    # Batched render (molview-render-streamline.md §1/§5: coalesce a multi-door
    # update into ONE 3Dmol paint at the outermost close).
    "beginBatch", "endBatch",
    # Escape hatch
    "_viewer3dmol",
])

# ``_test`` is an OBJECT (the per-instance test handle), not a function.
EXPECTED_NON_FUNCTIONS = sorted(["_test"])

# Doors that USED to exist on the viewer and are now RETIRED: the embed
# no longer synthesises force arrows or index labels itself — the
# consumer hands overlays in via setArrows/setLabels (§14.5.1).  These
# must never reappear on the handle.
RETIRED_DOORS = ["showForces", "showIndices"]


def _extract_handle_keys() -> tuple[list[str], list[str]]:
    """Parse the embed module's handle return block → (function-keys,
    non-function-keys).

    The block lives at the end of ``create(target, opts)`` as a literal
    ``return { name: value, … }``.  A value that is a bare identifier
    (``setName``) is a function; a non-trivial expression
    (``_buildTestHandle(state)``) is a non-function export.
    """
    src = MODULE.read_text()
    start_match = re.search(
        r"^\s+return\s+\{\s*\n\s+setStructure:\s+setStructure,",
        src, re.MULTILINE,
    )
    if start_match is None:
        pytest.fail(
            "Could not locate the embed module's handle return block. "
            "If the format changed, update this test's parser.")
    start_idx = start_match.end()
    end_match = re.search(r"\n\s+\};", src[start_idx:])
    if end_match is None:
        pytest.fail(
            "Could not locate the handle return block terminator.")
    block = src[start_match.start():start_idx + end_match.start()]
    fns: list[str] = []
    nons: list[str] = []
    for m in re.finditer(
            r"^\s+([a-zA-Z_][a-zA-Z_0-9]*):\s+([^,\n]+),\s*$",
            block, re.MULTILINE):
        name, rhs = m.group(1), m.group(2).strip()
        if re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", rhs):
            fns.append(name)
        else:
            nons.append(name)
    return sorted(fns), sorted(nons)


class TestHandleSurface:
    """PIN 2 — the handle exposes exactly the current documented set."""

    def test_handle_exposes_every_documented_method(self):
        fns, _ = _extract_handle_keys()
        missing = [m for m in EXPECTED_METHODS if m not in fns]
        assert not missing, (
            f"Handle is missing documented methods: {missing}\n"
            f"Present functions: {fns}\n"
            f"Doc-vs-code drift; implement the missing methods or update "
            f"molview-module.md §14.5 + EXPECTED_METHODS.")

    def test_handle_exports_no_undocumented_methods(self):
        fns, _ = _extract_handle_keys()
        extras = [m for m in fns if m not in EXPECTED_METHODS]
        assert not extras, (
            f"Handle exports undocumented functions: {extras}.\n"
            f"Remove them or document them in EXPECTED_METHODS.")

    def test_handle_non_function_exports_match_documented_set(self):
        _, nons = _extract_handle_keys()
        assert nons == EXPECTED_NON_FUNCTIONS, (
            f"Non-function exports drifted.\n"
            f"  Expected: {EXPECTED_NON_FUNCTIONS}\n"
            f"  Got:      {nons}")

    def test_handle_carries_the_overlay_and_chrome_doors(self):
        """The load-bearing overlay + chrome doors MolView draws
        through must be present (§14.5): setArrows/setLabels (overlays),
        setCell/setAxes/setStyle/setStructure (chrome + data)."""
        fns, _ = _extract_handle_keys()
        required = ["setStructure", "setArrows", "setLabels",
                    "setCell", "setAxes", "setStyle"]
        missing = [m for m in required if m not in fns]
        assert not missing, (
            f"Handle is missing load-bearing doors: {missing}")

    def test_retired_force_and_index_doors_are_gone(self):
        """PIN 2 (negative) — the embed no longer generates overlays,
        so the retired ``showForces`` / ``showIndices`` doors must not
        appear on the handle NOR anywhere in the module."""
        fns, nons = _extract_handle_keys()
        keys = set(fns) | set(nons)
        on_handle = [d for d in RETIRED_DOORS if d in keys]
        assert not on_handle, (
            f"Retired doors reappeared on the handle: {on_handle}. "
            f"Overlays are consumer-handed via setArrows/setLabels "
            f"(§14.5.1); the viewer does not generate them.")
        src = MODULE.read_text()
        in_source = [d for d in RETIRED_DOORS if d in src]
        assert not in_source, (
            f"Retired door identifiers still present in the module: "
            f"{in_source}. They should be fully removed (no back-compat "
            f"shim — pre-1.0 clean break).")


# --------------------------------------------------------------------- #
#  PIN 3 — the View menu (_buildViewMenu)                               #
# --------------------------------------------------------------------- #

def _view_menu_source() -> str:
    """The body of ``_buildViewMenu`` — bounded between its own
    declaration and the next top-level function (``_buildExportMenu``)."""
    src = MODULE.read_text()
    start = re.search(r"^\s+function _buildViewMenu\(", src, re.MULTILINE)
    end = re.search(r"^\s+function _buildExportMenu\(", src, re.MULTILINE)
    assert start and end and end.start() > start.start(), (
        "Could not bound _buildViewMenu; update this parser.")
    return src[start.start():end.start()]


def _view_toggles_source() -> str:
    """The body of the ``VIEW_TOGGLES`` registry array — the single
    source that stamps out the left-RAIL toggle buttons."""
    src = MODULE.read_text()
    start = re.search(r"const VIEW_TOGGLES = \[", src)
    assert start, "Could not find the VIEW_TOGGLES registry."
    end = src.find("];", start.start())
    assert end != -1
    return src[start.start():end]


class TestViewToggleRegistry:
    """PIN 3 — the view toggles live on the left RAIL, built from the ONE
    VIEW_TOGGLES registry (NOT in the View dropdown menu anymore).  The
    menu holds only the richer style/background/projection controls, so
    there is one control per toggle (no rail-vs-menu duplication)."""

    def test_reset_axes_labels_overlay_cell_are_registry_entries(self):
        """reset / axes / labels / overlay / cell are entries in the ONE
        VIEW_TOGGLES registry, each with its data-key ``action``."""
        reg = _view_toggles_source()
        for action in ["reset", "axes", "labels", "overlay", "cell"]:
            assert f'action: "{action}"' in reg, (
                f"{action} must be a VIEW_TOGGLES registry entry "
                f'(action: "{action}").')

    def test_stateful_toggles_carry_their_menu_label(self):
        """The stateful toggles carry their human label (used for the
        button title + injected-menu text)."""
        reg = _view_toggles_source()
        for label in ["Show axes", "Show labels",
                      "Show overlay", "Show unit cell"]:
            assert label in reg, (
                f"VIEW_TOGGLES is missing the {label!r} label.")

    def test_cell_toggle_is_knob_gated(self):
        """The unit-cell toggle is gated by knobs.cell (via its
        ``knob: \"cell\"`` field), preserved from the menu design."""
        reg = _view_toggles_source()
        assert re.search(
            r'action:\s*"cell".*?knob:\s*"cell"', reg, re.DOTALL), (
            'The cell toggle must carry knob: "cell" so it is gated '
            "on knobs.cell.")

    def test_view_menu_has_no_toggle_duplication(self):
        """The View dropdown must NOT re-render the toggles or a reset
        action — those live on the rail now (no duplication)."""
        body = _view_menu_source()
        assert "mol-viewer-menu-toggles" not in body, (
            "The View menu still builds a toggle group; toggles moved "
            "to the left rail (remove the menu-toggles group).")
        assert '_addToggle(' not in body, (
            "The View menu still calls _addToggle; toggles are rail-only "
            "now.")
        assert 'data-action", "reset"' not in body, (
            "The View menu still has a Reset action; reset is a rail "
            "button now.")
        assert '_menuSection("style"' in body, (
            "The View menu must still hold the Style section.")


# --------------------------------------------------------------------- #
#  PIN 4 (static) — overlays are consumer-handed; never fabricated       #
# --------------------------------------------------------------------- #

def _fn_source(name: str, next_marker: str) -> str:
    src = MODULE.read_text()
    start = re.search(rf"^\s+function {re.escape(name)}\(", src, re.MULTILINE)
    end = re.search(rf"^\s+function {re.escape(next_marker)}\(",
                    src, re.MULTILINE)
    assert start and end and end.start() > start.start(), (
        f"Could not bound {name}; update this parser.")
    return src[start.start():end.start()]


class TestOverlaysConsumerHanded:
    """PIN 4 (static half) — the embed draws what it is HANDED; the
    force-arrow overlay is gated on the overlay toggle and never
    synthesised from force data (§14.5.1)."""

    def test_arrow_redraw_gates_on_overlayOn(self):
        body = _fn_source("_redrawArrows", "_redrawAllOverlays")
        assert "state.current.overlayOn" in body, (
            "Arrow redraw must gate on state.current.overlayOn.")
        # The gate short-circuits (returns) when the overlay is off OR
        # there are no arrows to draw.
        assert re.search(
            r"if\s*\(\s*!state\.current\.overlayOn\b", body), (
            "Arrow redraw must early-return when overlayOn is false.")
        assert "state.current.arrows" in body, (
            "Arrows drawn must come from state.current.arrows (what the "
            "consumer handed via setArrows).")

    def test_arrows_are_drawn_from_the_handed_specs_only(self):
        """``_drawArrows(viewer, arrows)`` iterates the arrow specs it
        is given (start/end/color/radius) and reads NO force / atom-
        coordinate data — it draws exactly what it's handed."""
        body = _fn_source("_drawArrows", "_redrawArrows")
        assert re.search(r"function _drawArrows\(viewer,\s*arrows\)", body), (
            "_drawArrows must take the arrow specs as an argument.")
        for token in ("force", "currentForces", "getForces"):
            assert token.lower() not in body.lower(), (
                f"_drawArrows must not touch force data (found {token!r}); "
                f"the consumer owns force→arrow generation (§14.5.1).")

    def test_embed_never_reads_force_data(self):
        """PIN 4 — MolView is a viewer: the embed synthesises no force
        arrows, so it references no force accessor anywhere."""
        src = MODULE.read_text()
        for token in ("currentForces", "getForces"):
            assert token not in src, (
                f"Embed references {token!r}; force data is consumer-"
                f"owned and must never be pulled by the viewer (§14.5.1).")

    def test_setArrows_stores_specs_then_redraws(self):
        """``setArrows`` records the handed specs on
        state.current.arrows and triggers a redraw — it forwards, it
        does not build."""
        body = _fn_source("setArrows", "setPick")
        assert "state.current.arrows = next" in body, (
            "setArrows must store the handed specs on "
            "state.current.arrows.")
        assert "_redrawArrows(state)" in body, (
            "setArrows must trigger _redrawArrows.")


class TestArrowHandOffDrivesOverlayVisibility:
    """REGRESSION PIN (2026-07, the order-dependent force-toggle bug):
    handed arrows drive overlay visibility at EVERY arrow hand-off door,
    not just the full movie load.  The render engine expresses the
    store's showForces switch through the payload (it bakes no arrows
    while the flag is off), and in a molview mount nothing else ever
    writes ``overlayOn`` (the store-backed toggle replaced setOverlay).
    When only ``_setAnimationImpl`` derived it, toggling "show forces"
    after an isolate regen re-baked real arrows through the PARTIAL
    setAnimation path, ``overlayOn`` stayed stale-false, and
    ``_redrawArrows`` (gated on it) drew nothing until the next full
    reload -- the "toggle isolate twice to see forces" bug.
    """

    def test_full_setAnimation_derives_overlayOn_from_payload(self):
        body = _fn_source("_setAnimationImpl", "_playImpl")
        assert re.search(
            r"state\.current\.overlayOn\s*=\s*_arrowsPerFrameHasAny\(", body), (
            "_setAnimationImpl must derive overlayOn from the handed "
            "arrowsPerFrame payload.")

    def test_partial_setAnimation_derives_overlayOn_from_payload(self):
        body = _fn_source("setAnimation", "playAnimation")
        assert re.search(
            r"state\.current\.overlayOn\s*=\s*\n?\s*_arrowsPerFrameHasAny\(", body), (
            "The PARTIAL setAnimation arrowsPerFrame path must derive "
            "overlayOn from the payload (the stale-gate order bug).")

    def test_appendFrameArrows_derives_overlayOn_from_payload(self):
        body = _fn_source("appendFrameArrows", "getCamera")
        assert re.search(
            r"state\.current\.overlayOn\s*=\s*_arrowsPerFrameHasAny\(", body), (
            "appendFrameArrows must derive overlayOn over the whole "
            "accumulated arrowsPerFrame set.")

    def test_setArrows_derives_overlayOn_from_payload(self):
        body = _fn_source("setArrows", "setPick")
        assert re.search(
            r"state\.current\.overlayOn\s*=\s*next\.length\s*>\s*0", body), (
            "setArrows (the static-frame door) must derive overlayOn "
            "from the handed specs.")
