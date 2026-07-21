/* MolView render engine -- process: the PURE per-frame processor.
 *
 * Contract: docs/protocols/molview-render-streamline.md §2, §7.3.
 * Module:   molbuilder.molview.engine.process   (lib/molview/engine/process.js)
 * Used by:  engine.js (§9) -- runs this once per frame to build the frame's finished render
 *           data, then hands the result to embedIo.
 *
 * PURE: no 3Dmol, no DOM, no store, no time (its one dependency is the equally-pure L1 index
 * helper `atomIndexModel`, IMPORTED from ../_atom-index.js, §16). `processFrame(frame, identity, flags)` is a
 * function of its inputs only -- so it is node-unit-testable in isolation. It performs §2:
 *
 *   §2.3 selection filter  -> which atoms are DRAWN (isolate on + selection -> selected only)
 *   §2.4 overlays          -> index labels · selection halos (region/frozen/selection) ·
 *                             force vectors -- all keyed to the DRAWN atoms, with the ORIGINAL
 *                             atom index preserved via `sourceIndex` (the drawn->original map).
 *
 * INPUTS
 *   frame    = { coords: Vec3[], forces: Vec3[]|null }   // ONE frame (all atoms).
 *   identity = { elements: string[], annotations: AtomAnno[] }   // shared across frames (§7.1).
 *   flags    = { selection:int[], isolate:bool, showIndex:bool, showForces:bool,
 *                forceScale?:number }                    // §7.2 (the per-atom-relevant subset).
 *   AtomAnno = { label?:string, frozen?:boolean }
 *
 * OUTPUT: ProcessedFrame (§7.3) -- see the return of processFrame().
 *
 * NOTE (supersedes task #44): force vectors are built HERE, in the streamline, from the raw
 * per-atom forces × `forceScale` -- NOT handed in pre-built by the consumer. That keeps the
 * force overlay data/flag driven and isolate-aware (only drawn atoms get arrows) instead of an
 * opaque arrow list built outside the one render place. The consumer supplies raw forces + the
 * scale flag; the streamline owns the geometry.
 *
 * A native ES module (private submodule of the MolView module, frontend-module-architecture.md
 * §4).  engine.js IMPORTS it directly; the browser-global publish at the bottom is a TEST SEAM
 * (tests/test_engine_process_js.py reads molview.engine.process), not a production consumer edge.
 */
"use strict";

import { atomIndexModel } from "../_atom-index.js";

// ---- Overlay style tokens (moved here from the selection viewer-adapter, which no longer
// paints; the streamline owns all overlay derivation). Named constants, not inline
// literals -- the halo geometry/colour vocabulary lives in exactly one place. --------- //

// Per-region tint palette -- mirrors the panel's region tag colours so viewer + panel agree.
var REGION_COLORS = {
    "L-electrode": "#7fc97f",
    "R-electrode": "#beaed4",
    "bridge":      "#fdc086",
    "interface":   "#ffff99",
};
// Stable fallback palette for a free-form region label (hash -> colour).
var FALLBACK_PALETTE = ["#a6d8a4", "#ffb482", "#a6c8ff", "#ffa6c5",
                        "#d4b8ff", "#ffd17c", "#7cdfdf", "#ff9b9b"];
// Halo geometry (Å) + opacity per highlight layer. radius reads as a halo without swamping
// neighbours; opacity < 1 lets the atom show through.
var HALO_REGION = { radius: 0.5,  opacity: 0.35 };
var HALO_FROZEN = { color: "#ff5050", radius: 0.25, opacity: 0.85 };
var HALO_SELECT = { color: "yellow", radius: 0.7,  opacity: 0.45 };
// Neutral default force scale (Å per force unit): identity, so raw forces draw at magnitude.
// The consumer overrides via flags.forceScale for a physically-meaningful length.
var DEFAULT_FORCE_SCALE = 1.0;

function _fallbackColor(label) {
    var h = 0;
    for (var i = 0; i < label.length; i++) h = ((h << 5) - h + label.charCodeAt(i)) | 0;
    return FALLBACK_PALETTE[Math.abs(h) % FALLBACK_PALETTE.length];
}
function _colorFor(label) {
    // hasOwnProperty.call: a label like "__proto__"/"constructor" must not pierce the
    // Object prototype (region labels are user-controlled, free-form).
    return Object.prototype.hasOwnProperty.call(REGION_COLORS, label)
        ? REGION_COLORS[label] : _fallbackColor(label);
}

// §2.3 selection filter: the DRAWN atom set, in original-atom order (ascending). Isolate ON
// with a non-empty selection keeps only the selected atoms; otherwise every atom is drawn.
// (Selection alone -- isolate off -- draws all atoms; it only highlights, §2.3.)
function _drawnAtoms(nAtoms, flags) {
    var sel = Array.isArray(flags.selection) ? flags.selection : [];
    var isolate = !!flags.isolate && sel.length > 0;
    var drawn = [];
    if (isolate) {
        var selSet = {};
        for (var k = 0; k < sel.length; k++) selSet[sel[k]] = true;
        for (var a = 0; a < nAtoms; a++) if (selSet[a]) drawn.push(a);
    } else {
        for (var b = 0; b < nAtoms; b++) drawn.push(b);
    }
    return drawn;
}

// §2.4 halos: region tints -> frozen markers -> selection halo, pushed in that order so the
// selection reads ON TOP (setOverlays draws entries in array order). Indices are in the
// DRAWN space (0..nDrawn-1); an atom's ORIGINAL identity is looked up via sourceIndex[m].
// Under isolate the drawn set IS the selection, but region/frozen still distinguish members.
function _buildHalos(drawn, annotations, selection) {
    var entries = [];
    // 1. region tints -- group drawn atoms by their (first) region label.
    var byRegion = {};        // label -> [drawn indices]
    var order = [];           // preserve first-seen label order for determinism
    for (var m = 0; m < drawn.length; m++) {
        var anno = annotations[drawn[m]] || {};
        var label = anno.label || null;
        if (!label) continue;
        if (!Object.prototype.hasOwnProperty.call(byRegion, label)) { byRegion[label] = []; order.push(label); }
        byRegion[label].push(m);
    }
    for (var r = 0; r < order.length; r++) {
        var lab = order[r];
        entries.push({ indices: byRegion[lab].slice(),
                       halo: { color: _colorFor(lab), radius: HALO_REGION.radius, opacity: HALO_REGION.opacity } });
    }
    // 2. frozen markers.
    var frozen = [];
    for (var f = 0; f < drawn.length; f++) if ((annotations[drawn[f]] || {}).frozen) frozen.push(f);
    if (frozen.length) entries.push({ indices: frozen,
        halo: { color: HALO_FROZEN.color, radius: HALO_FROZEN.radius, opacity: HALO_FROZEN.opacity } });
    // 3. selection halo (the live pick set) -- brightest + largest, on top.
    var selSet = {};
    (selection || []).forEach(function (i) { selSet[i] = true; });
    var selDrawn = [];
    for (var s = 0; s < drawn.length; s++) if (selSet[drawn[s]]) selDrawn.push(s);
    if (selDrawn.length) entries.push({ indices: selDrawn,
        halo: { color: HALO_SELECT.color, radius: HALO_SELECT.radius, opacity: HALO_SELECT.opacity } });

    return entries.length ? { atoms: entries } : null;
}

// THE per-frame processor (§2). Returns a ProcessedFrame (§7.3).
function processFrame(frame, identity, flags) {
    frame = frame || {};
    identity = identity || {};
    flags = flags || {};
    var coords = Array.isArray(frame.coords) ? frame.coords : [];
    var elements = Array.isArray(identity.elements) ? identity.elements : [];
    var annotations = Array.isArray(identity.annotations) ? identity.annotations : [];
    var nAtoms = coords.length;

    // §2.3 -- which atoms are drawn, + the drawn->original index map.
    var drawn = _drawnAtoms(nAtoms, flags);
    var positions = drawn.map(function (a) { var p = coords[a] || [0, 0, 0]; return [p[0], p[1], p[2]]; });
    var sourceIndex = drawn.slice();                       // sourceIndex[m] = original atom index
    var outElements = drawn.map(function (a) { return elements[a] || "X"; });

    // §2.4 -- index labels: explicit TEXT (the ORIGINAL index) at the drawn atom's position,
    // so an isolate-filtered / re-indexed model still shows the true atom index (the embed's
    // format:"index" would show the drawn index -- wrong under isolate). The displayed number
    // is 1-based (SIESTA/Fortran convention, data-vocabulary.md §3.1 / molview-module §16):
    // sourceIndex stays 0-based internal, but the label text goes through the L1 helper
    // `atomIndexModel.toDisplay` -- REUSED, never re-derived (a bare `a+1` would drift).
    // IMPORTED from the pure L1 index leaf (§16) -- always resolved, no load-order guard needed.
    var labels = null;
    if (flags.showIndex) {
        var toDisplay = atomIndexModel.toDisplay;
        labels = drawn.map(function (a, m) {
            return { position: positions[m], text: String(toDisplay(a)) };   // 1-based, SIESTA
        });
    }

    // §2.4 -- halos (region / frozen / selection), in the drawn-index space.
    var halos = _buildHalos(drawn, annotations, flags.selection);

    // §2.4 -- force vectors for THIS frame, built from the raw per-atom forces × scale, for
    // the drawn atoms only (isolate-aware). null when there are no forces or the overlay is off.
    var arrows = null;
    if (flags.showForces && Array.isArray(frame.forces)) {
        var scale = (typeof flags.forceScale === "number") ? flags.forceScale : DEFAULT_FORCE_SCALE;
        arrows = drawn.map(function (a, m) {
            var p = positions[m];
            var v = frame.forces[a] || [0, 0, 0];
            return { start: [p[0], p[1], p[2]],
                     end:   [p[0] + v[0] * scale, p[1] + v[1] * scale, p[2] + v[2] * scale] };
        });
    }

    return {
        positions:   positions,
        sourceIndex: sourceIndex,
        elements:    outElements,
        labels:      labels,
        halos:       halos,
        arrows:      arrows,
    };
}

export const process = { processFrame: processFrame };

// TEST SEAM: tests/test_engine_process_js.py reads globalThis.molbuilder.molview.engine.process.
// engine.js imports the export above; production reads no global.  Window-guarded.
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.engine = window.molbuilder.molview.engine || {};
    window.molbuilder.molview.engine.process = process;
}
