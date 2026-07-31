/* MolView — MEASUREMENT: its own layer, not part of drawing.
 *
 * Contract: docs/web/molview.md § 11.6.  The readout in the 3D window is the result of a
 * user INTERACTING with the view, not part of producing a frame — so it is not an overlay
 * and the render pipeline does not make it.
 *
 * What it reads: its atoms come from the selection in PICK ORDER (which is why the vertex
 * of a three-atom angle is the atom picked second, not the middle one by number), and its
 * coordinates come from the MASTER COPY at the current frame (§ 6.3) — never from the
 * drawing.  That is what keeps it correct while a trajectory plays and under isolate,
 * where the drawn numbering no longer matches the real one (§ 6.5).
 *
 * When it repaints: on a selection change OR a frame change — it subscribes to both.
 *
 * Assembled 2026-07-30 from selection/measurements.js (the distance / angle maths) +
 * measurement-overlay.js (the readout that mounts them).  One layer, one file; the panel
 * imports the maths from here.  Bodies unchanged.
 */
"use strict";

import { atomIndexModel } from "./_atom.js";


/* ── The maths: position · distance · angle — was selection/measurements.js ──── */

function _vec3sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function _vec3norm(v) {
    return Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

function _vec3dot(a, b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

function distance(p, q) {
    return _vec3norm(_vec3sub(p, q));
}

/**
 * Bond angle in degrees, with ``b`` at the vertex.  Returns
 * the angle formed by the two bonds a-b and b-c (i.e. the
 * angle between vectors (a-b) and (c-b)).  Clamps the cosine
 * to [-1, 1] before acos to avoid NaN on near-collinear
 * triples where floating-point error pushes the dot product
 * slightly outside the valid range.
 */
function angleDeg(a, b, c) {
    const u = _vec3sub(a, b);
    const v = _vec3sub(c, b);
    const nu = _vec3norm(u);
    const nv = _vec3norm(v);
    if (nu === 0 || nv === 0) return NaN;
    let cos = _vec3dot(u, v) / (nu * nv);
    if (cos >  1) cos =  1;
    if (cos < -1) cos = -1;
    return Math.acos(cos) * 180 / Math.PI;
}

/**
 * Human label for an atom.  Prefers an explicit atom_name
 * (e.g. ``OE1``, ``Au-L1``) since residue-aware names tell
 * the chemist what they're looking at; falls back to
 * ``${element} #${index+1}`` when no atom_name is present.
 * Always 1-based to match the indices the user sees in the
 * atom list and the viewer overlays.
 */
// 1-based display goes through the ONE shared rule (data-vocabulary § 3.1),
// not an ad-hoc idx+1, so every surface agrees.  Falls back to idx+1 only
// when the model isn't loaded (e.g. a pure node measurement-math test).
function _toDisplay(idx) {
    return (atomIndexModel && typeof atomIndexModel.toDisplay === "function")
        ? atomIndexModel.toDisplay(idx) : idx + 1;
}

function labelOf(idx, atomsMeta) {
    const a = atomsMeta && atomsMeta[idx];
    if (a) {
        const name = (a.atom_name || a.atomName || "").toString().trim();
        if (name) return name + " #" + _toDisplay(idx);
        const el = (a.element || "").toString().trim();
        if (el) return el + " #" + _toDisplay(idx);
    }
    return "#" + _toDisplay(idx);
}

function _fmtCoord(v) {
    return Number(v).toFixed(3);
}

function _fmtXyz(p) {
    return "("
        + _fmtCoord(p[0]) + ", "
        + _fmtCoord(p[1]) + ", "
        + _fmtCoord(p[2])
        + ") Å";
}

/**
 * The single entry point.  Returns ``null`` when ``selection``
 * is empty or has 4+ atoms; otherwise the typed envelope
 * documented at the top of this file.  Caller is responsible
 * for hiding the readout when this returns null.
 *
 * Bails (returns null) if any selected index is out of range
 * for ``positions`` — caller should treat that as "no
 * measurement yet" (positions array hasn't caught up with a
 * recent atom-count-changing op).
 *
 * ``pickOrder`` (optional) is the user's click sequence.
 * When supplied with three atoms, the second click is taken
 * as the angle vertex (the chemist's convention: pick the
 * two ends + the middle atom in the middle of the sequence).
 * Falls back to the geometric heuristic (vertex = atom with
 * smallest sum-of-distances to the other two) when pickOrder
 * is missing or doesn't match the selection set.
 */
function compute(selection, atomsMeta, positions, pickOrder) {
    if (!Array.isArray(selection)) return null;
    const n = selection.length;
    if (n === 0 || n > 3) return null;
    if (!Array.isArray(positions)) return null;

    for (let i = 0; i < n; i++) {
        const idx = selection[i];
        if (typeof idx !== "number"
            || !positions[idx]
            || positions[idx].length < 3) {
            return null;
        }
    }

    const labels = selection.map((i) => labelOf(i, atomsMeta));

    if (n === 1) {
        const idx = selection[0];
        const p = positions[idx];
        return {
            kind:    "xyz",
            indices: [idx],
            labels:  [labels[0]],
            xyz:     [p[0], p[1], p[2]],
            display: labels[0] + " — " + _fmtXyz(p),
        };
    }

    if (n === 2) {
        // Use pick order when available so the readout reflects
        // the user's "from → to" intent — even though distance
        // is symmetric, the label A-B reads naturally as "the
        // bond I just drew."
        const order = _pickOrderMatches(pickOrder, selection)
            ? pickOrder : selection;
        const [iA, iB] = order;
        const d = distance(positions[iA], positions[iB]);
        const labA = labelOf(iA, atomsMeta);
        const labB = labelOf(iB, atomsMeta);
        return {
            kind:     "distance",
            indices:  [iA, iB],
            labels:   [labA, labB],
            valueAng: d,
            display:  "|" + labA + " – " + labB + "| = "
                      + d.toFixed(4) + " Å",
        };
    }

    // n === 3: angle.  Pick-order semantics win when the
    // user clicked the atoms in sequence (chemist's
    // convention: middle click = vertex), so the readout
    // reflects 1-2 + 2-3 bonds meeting at atom 2.  When
    // pickOrder is missing or stale (filter eval, session
    // restore, batch setSelection from a non-click source),
    // fall back to the geometric heuristic — vertex = atom
    // with smallest sum-of-distances to the other two, which
    // for a typical bonded triple still gives the centre
    // atom.
    let triplet;
    if (_pickOrderMatches(pickOrder, selection)) {
        triplet = pickOrder.slice();
    } else {
        triplet = _geometricVertexOrder(selection, positions);
    }
    const [iA, iV, iB] = triplet;
    const deg = angleDeg(
        positions[iA], positions[iV], positions[iB]);
    const labA = labelOf(iA, atomsMeta);
    const labV = labelOf(iV, atomsMeta);
    const labB = labelOf(iB, atomsMeta);
    return {
        kind:        "angle",
        indices:     [iA, iV, iB],
        labels:      [labA, labV, labB],
        vertexIndex: iV,
        valueDeg:    deg,
        display:     "∠" + labA + " – " + labV + " – " + labB
                     + " = " + deg.toFixed(1) + "°",
    };
}

// True iff ``pickOrder`` is the same SET as ``selection`` —
// i.e. the click trail is in sync with the sorted selection.
// A mismatch (length, missing atoms, repeats) usually means
// the source wasn't a click (filter eval, batch setSelection,
// corrupt pickOrder) and the angle vertex must be chosen
// geometrically.  Repeat detection is what saves the
// angle math from feeding two identical vectors to angleDeg
// and emitting "NaN°" — the helper is the last gate before
// pickOrder[1] becomes the vertex.
function _pickOrderMatches(pickOrder, selection) {
    if (!Array.isArray(pickOrder)) return false;
    if (pickOrder.length !== selection.length) return false;
    const set = new Set(selection);
    const seen = new Set();
    for (let i = 0; i < pickOrder.length; i++) {
        const idx = pickOrder[i];
        if (!set.has(idx) || seen.has(idx)) return false;
        seen.add(idx);
    }
    return true;
}

// Reorder a 3-atom selection so the geometric centre atom
// (smallest sum-of-distances to the other two) is in the
// middle.  Used when pickOrder is unavailable.
function _geometricVertexOrder(selection, positions) {
    const [i0, i1, i2] = selection;
    const p0 = positions[i0], p1 = positions[i1], p2 = positions[i2];
    const s0 = distance(p0, p1) + distance(p0, p2);
    const s1 = distance(p1, p0) + distance(p1, p2);
    const s2 = distance(p2, p0) + distance(p2, p1);
    let vertexLocal = 0;
    if (s1 < s0) vertexLocal = 1;
    if (s2 < (vertexLocal === 0 ? s0 : s1)) vertexLocal = 2;
    const others = [0, 1, 2].filter((k) => k !== vertexLocal);
    return [
        selection[others[0]],
        selection[vertexLocal],
        selection[others[1]],
    ];
}

// TEST SEAM: the node unit test (tests/test_selection_measurements_js.py) reads this via
// globalThis.molbuilder.molview.selection.measurements; production consumers (panel.js,
// measurement-overlay.js) IMPORT the `measurements` export below, so this publish is test-only.
export const measurements = {
    compute:   compute,
    distance:  distance,
    angleDeg:  angleDeg,
    labelOf:   labelOf,
};


// TEST SEAM publish (see the comment on `export const measurements` above): the node unit test
// reads globalThis.molbuilder.molview.selection.measurements.  Window-guarded (node has no window
// unless the harness stubs one).  Production consumers import the export; do not read this global.
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.selection = window.molbuilder.molview.selection || {};
    window.molbuilder.molview.selection.measurements = measurements;
}

/* ── The readout in the 3D window — was measurement-overlay.js ───────────────── */

export function mountMeasurementOverlay(viewerHost, opts) {
    opts = opts || {};
    const store = opts.store;
    const coordsProvider = opts.coordsProvider;
    if (!viewerHost || typeof viewerHost.appendChild !== "function") {
        throw new Error("mountMeasurementOverlay: a viewerHost element is required");
    }
    if (!store || typeof store.getState !== "function") {
        throw new Error("mountMeasurementOverlay: a store with getState() is required");
    }
    if (typeof coordsProvider !== "function") {
        throw new Error("mountMeasurementOverlay: a coordsProvider() is required");
    }
    const meas = measurements;   // imported pure geometry leaf (4a)

    const el = document.createElement("div");
    el.className = "molview-measurement-overlay selection-measurement-overlay";
    el.hidden = true;
    viewerHost.appendChild(el);

    function render() {
        const s = store.getState() || {};
        // Ordered picks: pickOrder (click order -> angle vertex) else indices. Both are
        // ORIGINAL 0-based atom indices, curated via the panel (§7.3 interaction contract).
        const picks = (Array.isArray(s.pickOrder) && s.pickOrder.length)
            ? s.pickOrder
            : (Array.isArray(s.indices) ? s.indices : []);
        if (!meas || picks.length < 1 || picks.length > 3) {
            el.hidden = true;
            el.textContent = "";
            return;
        }
        // coordsProvider() is the CURRENT frame's CLEAN, ORIGINAL-indexed coords (all atoms,
        // via molview.data.getFrameAllAtoms -- never the isolate-filtered draw). meas.compute indexes by
        // GLOBAL atom index, which is exactly this array's index -> NO re-keying, and it works
        // identically whether or not isolate is on (molview-render-streamline.md §7.3).
        const coords = coordsProvider() || [];
        const result = meas.compute(picks, s.atoms || [], coords, picks);
        if (result && result.display) {
            el.hidden = false;
            el.dataset.kind = result.kind;
            el.textContent = result.display;
        } else {
            el.hidden = true;
            el.textContent = "";
        }
    }

    let _unsub = null;
    if (typeof store.subscribe === "function") {
        _unsub = store.subscribe(function () { render(); });   // fires immediately
    } else {
        render();
    }

    return {
        render: render,
        dispose: function () {
            if (_unsub) { try { _unsub(); } catch (_) {} }
            if (el.parentNode) el.parentNode.removeChild(el);
        },
    };
}

// ── Transitional global (removed once every consumer imports this module) ──
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.mountMeasurementOverlay = mountMeasurementOverlay;
}
