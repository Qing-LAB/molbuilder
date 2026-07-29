/* MolView render engine -- process: the PURE per-frame processor.
 *
 * Contract: docs/web/molview.md, §7.3.
 * Module:   molbuilder.molview.engine.process   (lib/molview/engine/process.js)
 * Used by:  engine.js (§9) -- runs this once per frame to build the frame's finished render
 *           data, then hands the result to embedIo.
 *
 * PURE: no 3Dmol, no DOM, no store, no time (its one dependency is the equally-pure L1 index
 * helper `atomIndexModel`, IMPORTED from ../_atom-index.js, §16). `processFrame(frame, identity, flags)` is a
 * function of its inputs only -- so it is node-unit-testable in isolation. It performs §2:
 *
 *   §2.3 selection filter  -> which atoms are DRAWN (isolate on + selection -> selected only)
 *   §2.4 overlays          -> index labels · selection highlight (WHICH atoms are selected) ·
 *                             force vectors -- all keyed to the DRAWN atoms, with the ORIGINAL
 *                             atom index preserved via `sourceIndex` (the drawn->original map).
 *
 * The output carries SEMANTIC CONTENT, never rendering style: `selection` is the list of drawn
 * indices to highlight; HOW a highlight looks (the glow geometry/colour) is the embed's business,
 * not baked into every frame. (§7.3 / §8.1.)
 *
 * INPUTS
 *   frame    = { coords: Vec3[], forces: Vec3[]|null }   // ONE frame (all atoms).
 *   identity = { elements: string[] }                    // shared across frames (§7.1).
 *   flags    = { selection:int[], isolate:bool, showIndex:bool, showForces:bool,
 *                forceScale?:number }                    // §7.2 (the per-atom-relevant subset).
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

// ---- Overlay tokens. Named constants, not inline literals. -------------------------- //

// Neutral default force scale (Å per force unit): identity, so raw forces draw at magnitude.
// The consumer overrides via flags.forceScale for a physically-meaningful length.
var DEFAULT_FORCE_SCALE = 1.0;
// Force-vector styling (§2.4): the largest drawn force is highlighted gold; the rest ramp from
// dim-red to orange-red by RELATIVE magnitude, and the arrow radius grows with it -- so the
// eye lands on the atom under the most force (the relaxation "hot spot"). A consumer suppresses
// a force (frozen atom, sub-threshold) by handing a ZERO vector: magnitudes at/under FORCE_EPS
// draw no arrow, so the consumer owns WHICH forces show; the engine owns HOW they look.
var FORCE_MAX_COLOR    = "#ffc400";           // gold: the single largest drawn force
var FORCE_RADIUS_MIN   = 0.05;                // Å: the thinnest (zero-relative-magnitude) arrow
var FORCE_RADIUS_SPAN  = 0.04;                // Å: added at the largest force
var FORCE_EPS          = 1e-9;                // |f| at/under this -> no arrow (a suppressed force)
function _forceRampColor(t) {                 // t in [0,1]: dim-red -> orange-red
    return "rgb(" + Math.floor(170 + 85 * t) + "," + Math.floor(40 + 60 * t) + ",32)";
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

// §2.4 selection highlight: the DRAWN indices of the selected atoms -- but ONLY when NOT
// isolating. Isolate makes the selection the entire drawn set, so an in-view highlight would add
// nothing (the selection IS all that's shown). Isolate off -> drawn = all atoms in original order,
// so a selected ORIGINAL index equals its drawn index. Returns the drawn indices (or null); the
// embed draws a translucent glow on them (the glow style is the embed's, not the engine's, §8.1).
function _selectionHighlight(drawn, flags) {
    if (flags.isolate) return null;                                   // no highlight under isolate
    var sel = Array.isArray(flags.selection) ? flags.selection : [];
    if (!sel.length) return null;
    var selSet = {};
    sel.forEach(function (i) { selSet[i] = true; });
    var out = [];
    for (var m = 0; m < drawn.length; m++) if (selSet[drawn[m]]) out.push(m);
    return out.length ? out : null;
}

// THE per-frame processor (§2). Returns a ProcessedFrame (§7.3).
function processFrame(frame, identity, flags) {
    frame = frame || {};
    identity = identity || {};
    flags = flags || {};
    var coords = Array.isArray(frame.coords) ? frame.coords : [];
    var elements = Array.isArray(identity.elements) ? identity.elements : [];
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

    // §2.4 -- selection highlight: the drawn indices to glow (isolate off), or null.
    var selection = _selectionHighlight(drawn, flags);

    // §2.4 -- force vectors for THIS frame, built from the raw per-atom forces × scale, for
    // the drawn atoms only (isolate-aware). null when there are no forces or the overlay is off.
    var arrows = null;
    if (flags.showForces && Array.isArray(frame.forces)) {
        var scale = (typeof flags.forceScale === "number") ? flags.forceScale : DEFAULT_FORCE_SCALE;
        // The KEPT (non-suppressed) set: a zero force -- the consumer's way to hide a frozen or
        // sub-threshold atom -- draws no arrow. maxMag over this set drives the gold highlight +
        // the color/radius ramp, so "biggest force" is relative to what is actually shown.
        var kept = [];
        var maxMag = 0, maxKi = -1;
        for (var ai = 0; ai < drawn.length; ai++) {
            var v = frame.forces[drawn[ai]] || [0, 0, 0];
            var mag = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (mag <= FORCE_EPS) continue;
            if (mag > maxMag) { maxMag = mag; maxKi = kept.length; }
            kept.push({ p: positions[ai], v: v, mag: mag });
        }
        arrows = kept.map(function (e, ki) {
            var t = maxMag > 0 ? e.mag / maxMag : 0;
            return {
                start:  [e.p[0], e.p[1], e.p[2]],
                end:    [e.p[0] + e.v[0] * scale, e.p[1] + e.v[1] * scale, e.p[2] + e.v[2] * scale],
                color:  ki === maxKi ? FORCE_MAX_COLOR : _forceRampColor(t),
                radius: FORCE_RADIUS_MIN + FORCE_RADIUS_SPAN * t,
            };
        });
        if (!arrows.length) arrows = null;   // nothing above the suppression floor -> no overlay
    }

    return {
        positions:   positions,
        sourceIndex: sourceIndex,
        elements:    outElements,
        labels:      labels,
        selection:   selection,   // drawn indices to glow (isolate off), or null
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
