"""Runtime (Node) unit tests for ``lib/mol-viewer-embed.js`` — the
concealed 3Dmol drawing layer MolView draws through.

These exercise the module's pure, exported helpers under Node with no
3Dmol and no browser (the option-normalisation surface published on
``window.molbuilder.viewer._normalise*``).  They PIN the two runtime
halves of the embed contract:

  * PIN 1 — ``_normaliseOpts({})`` returns the exact current
    ``state.current`` key set (the "ground truth" the idempotence diff
    is computed against).

  * PIN 4 (runtime half) — the consumer-handed force-arrow overlay:
    ``arrows`` pass through verbatim and ``overlayOn`` defaults OFF, so
    the specs the consumer hands via ``setArrows`` are STORED and drawn
    only when the overlay toggle is on (molview-module.md §14.5.1).  The
    embed never fabricates them.

The STATIC contracts (handle method set, View-menu structure, the
overlay-draw gate + never-reads-forces) live in
``test_mol_viewer_embed_handle_surface_js.py`` as source-text
invariants — ``_buildViewMenu`` / ``_redrawArrows`` are module-internal
and not reachable from this runtime harness.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from _node_esm import run_node


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/mol-viewer-embed.js"


def _run_node(snippet: str) -> object:
    """ES-module harness (tests/_node_esm): dynamic-import the embed -- now a native ES module
    that imports mol-viewer.js and publishes ``window.molbuilder.viewer`` (the §3.2 shim).  The
    snippet reads the internal helpers (``_normaliseOpts`` etc.) through that global.  The pure
    helpers we test need no 3Dmol at module-load time."""
    return run_node([MODULE], snippet)


# --------------------------------------------------------------------- #
#  PIN 1 — the internal state shape (_normaliseOpts key set)            #
# --------------------------------------------------------------------- #

# The exact keys ``_normaliseOpts({})`` produces — the ``state.current``
# shape.  Read verbatim from the module's ``_normaliseOpts`` return
# object.  Adding/removing a render-state slot requires updating this
# set (intentional friction).
EXPECTED_STATE_KEYS = sorted([
    "xyz", "pdb", "style", "axes", "cell", "labels",
    "arrows",       # consumer-handed force-arrow specs (§14.5.1)
    "overlayOn",    # the "Show overlay" toggle state — arrows on/off
    "pick", "lattice",
    "cellBox",      # unit-cell BOX channel: resolved cell + anchor origin (§3a, Issue #2)
    "overlays", "knobs", "projection",
])


class TestNormaliseOptsKeySet:

    def test_empty_opts_yields_exact_current_state_shape(self):
        """PIN 1 — ``_normaliseOpts({})`` returns exactly the current
        ``state.current`` key set."""
        out = _run_node("""
            const r = window.molbuilder.viewer._normaliseOpts({});
            console.log(JSON.stringify(Object.keys(r).sort()));
        """)
        assert out == EXPECTED_STATE_KEYS

    def test_no_retired_force_or_index_slots_in_state(self):
        """The retired self-generated overlay knobs must not reappear as
        state slots (overlays are consumer-handed — §14.5.1)."""
        out = _run_node("""
            const r = window.molbuilder.viewer._normaliseOpts({});
            console.log(JSON.stringify(Object.keys(r)));
        """)
        for retired in ("showForces", "showIndices", "forces"):
            assert retired not in out, (
                f"Retired state slot {retired!r} present; the viewer no "
                f"longer synthesises overlays.")


# --------------------------------------------------------------------- #
#  PIN 4 (runtime) — consumer-handed arrows + overlay gate default      #
# --------------------------------------------------------------------- #

class TestConsumerHandedArrows:

    def test_overlayOn_defaults_off(self):
        """``overlayOn`` (the force-arrow overlay toggle) defaults OFF,
        so handed arrows are stored but not drawn until the consumer /
        user turns the overlay on."""
        out = _run_node("""
            const r = window.molbuilder.viewer._normaliseOpts({});
            console.log(JSON.stringify({overlayOn: r.overlayOn}));
        """)
        assert out == {"overlayOn": False}

    def test_overlayOn_true_only_for_literal_true(self):
        out = _run_node("""
            const V = window.molbuilder.viewer;
            console.log(JSON.stringify({
                t:      V._normaliseOpts({overlayOn: true}).overlayOn,
                truthy: V._normaliseOpts({overlayOn: 1}).overlayOn,
                f:      V._normaliseOpts({overlayOn: false}).overlayOn,
            }));
        """)
        assert out == {"t": True, "truthy": False, "f": False}

    def test_arrows_default_to_empty_list(self):
        out = _run_node("""
            const r = window.molbuilder.viewer._normaliseOpts({});
            console.log(JSON.stringify({
                arrows: r.arrows, isArray: Array.isArray(r.arrows)}));
        """)
        assert out == {"arrows": [], "isArray": True}

    def test_handed_arrow_specs_pass_through_verbatim(self):
        """The embed draws what it's HANDED: the consumer's ArrowSpec
        list is copied through unchanged — the viewer neither builds nor
        normalises the geometry (§14.5.1)."""
        out = _run_node("""
            const specs = [
                {start:[0,0,0], end:[1,0,0], color:"#f00", radius:0.05},
                {start:[1,1,1], end:[2,2,2], label:"F"},
            ];
            const r = window.molbuilder.viewer._normaliseOpts({arrows: specs});
            console.log(JSON.stringify(r.arrows));
        """)
        assert out == [
            {"start": [0, 0, 0], "end": [1, 0, 0],
             "color": "#f00", "radius": 0.05},
            {"start": [1, 1, 1], "end": [2, 2, 2], "label": "F"},
        ]

    def test_arrows_slot_is_a_defensive_copy(self):
        """``_normaliseOpts`` slices the arrows array so a later mutation
        of the caller's list can't reach into render state."""
        out = _run_node("""
            const specs = [{start:[0,0,0], end:[1,0,0]}];
            const r = window.molbuilder.viewer._normaliseOpts({arrows: specs});
            specs.push({start:[9,9,9], end:[8,8,8]});
            console.log(JSON.stringify({len: r.arrows.length}));
        """)
        assert out == {"len": 1}

    def test_non_array_arrows_becomes_empty_list(self):
        out = _run_node("""
            const V = window.molbuilder.viewer;
            console.log(JSON.stringify({
                num:  V._normaliseOpts({arrows: 42}).arrows,
                str:  V._normaliseOpts({arrows: "x"}).arrows,
                nul:  V._normaliseOpts({arrows: null}).arrows,
            }));
        """)
        assert out == {"num": [], "str": [], "nul": []}


# --------------------------------------------------------------------- #
#  Supporting knob defaults that the View menu is built from            #
# --------------------------------------------------------------------- #

class TestKnobDefaults:
    """The View menu (`_buildViewMenu`) renders each toggle only when its
    knob is enabled; these defaults are what make Reset + axes/labels/
    overlay/cell appear by default."""

    def test_view_menu_knobs_default_on(self):
        out = _run_node("""
            const k = window.molbuilder.viewer._normaliseKnobs({});
            console.log(JSON.stringify({
                reset: k.reset, axes: k.axes, labels: k.labels,
                overlay: k.overlay, cell: k.cell, style: k.style,
                projection: k.projection,
            }));
        """)
        assert out == {
            "reset": True, "axes": True, "labels": True,
            "overlay": True, "cell": True, "style": True,
            "projection": True,
        }

    def test_cell_knob_can_be_suppressed(self):
        out = _run_node("""
            const k = window.molbuilder.viewer._normaliseKnobs({cell: false});
            console.log(JSON.stringify({cell: k.cell}));
        """)
        assert out == {"cell": False}


class TestProjectionDefault:

    def test_projection_defaults_to_perspective(self):
        out = _run_node("""
            const r = window.molbuilder.viewer._normaliseOpts({});
            console.log(JSON.stringify({projection: r.projection}));
        """)
        assert out == {"projection": "perspective"}

    def test_invalid_projection_falls_back(self):
        out = _run_node("""
            const r = window.molbuilder.viewer
                          ._normaliseOpts({projection: "fisheye"});
            console.log(JSON.stringify({projection: r.projection}));
        """)
        assert out == {"projection": "perspective"}
