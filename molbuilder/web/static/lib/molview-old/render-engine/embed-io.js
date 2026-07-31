/* MolView render engine -- embedIo: the ONE seal over the 3Dmol embed handle.
 *
 * Contract: docs/web/molview.md (and §3, §4, §8).
 * Module:   molbuilder.molview.renderEngine.embedIo   (lib/molview/render-engine/embed-io.js)
 * Used by:  engine.js -- the renderEngine orchestrator (§9), the ONLY caller. Nothing else in MolView
 *           may touch the 3Dmol handle: every setStructure / setAnimation / setArrows /
 *           setLabels / setOverlays / setBusy call funnels through HERE.
 *
 * This is a THIN, PURE-TRANSLATION layer: it turns the engine's plain data (§7.3
 * ProcessedFrame + scene) into the embed handle's door calls. It makes NO decisions about
 * WHICH primitive to run (that is the engine's §8 tiering) and holds NO state/flags. It owns
 * exactly one piece of 3Dmol knowledge the rest of the engine must not carry: the multi-frame
 * XYZ wire format 3Dmol parses.
 *
 * The fundamental primitives (each maps onto a §8 tier):
 *   loadFrames(movie)      -- STRUCTURAL REGEN: build the native multi-frame movie once (§3).
 *   swapFrame(i)           -- NATIVE SWAP: 3Dmol switches to a pre-parsed frame (no rebuild).
 *   appendFrames(tail)     -- APPEND: extend the movie with processed new frames (§6.2).
 *   applyOverlays(overlay) -- OVERLAY REFRESH: (re)apply labels/halos/arrows/cell/axis.
 *   setBusy(msg|null)      -- the §4 busy scrim.
 * Reads -- the engine's INTERNAL probes of the render seam, never surfaced upward:
 *   animationKind() -- does a trajectory movie exist? (the APPEND tier's precondition)
 *   frameCount()    -- how many frames the movie can actually show (the post-append check).
 * There is deliberately no currentFrame() read here: the shown frame is the ENGINE's owned
 * state (`_frame`, the §7.2 frame channel), and the embed follows it via swapFrame. A read-back
 * door would be a second answer to a question that already has an owner.
 *
 * INPUT SHAPES (all plain data -- no 3Dmol objects, no DOM):
 *   Movie = {
 *     frames:  ProcessedFrame[],     // §7.3; frames[0] establishes atom identity + count.
 *     cellBox: {lattice,origin}|null // the ONE cell geometry: a/b/c basis + anchor origin.
 *   }
 *   ProcessedFrame = { positions:Vec3[], sourceIndex:int[], elements:string[],
 *                      arrows:Arrow[]|null, labels:LabelOpts|null, selection:int[]|null }
 *   FrameTail = { frames: ProcessedFrame[] }               // the NEW frames only (§6.2).
 *   Overlay   = { labels?, selection?, arrows?, cellVisible?, axes? } // for the CURRENT frame; a field
 *                                                             // that is `undefined` is not touched.
 *
 * cellVisible is a PLAIN on/off boolean -- the cell VISIBILITY. The box GEOMETRY {lattice,origin}
 * is STRUCTURE data and rides the Movie.cellBox at load, NOT the overlay. The Arrow / LabelOpts
 * specs are exactly what the embed doors accept (setArrows / setLabels); `selection` is the list of
 * drawn atom indices to highlight (setSelectionHalo -- the embed owns the glow style). process.js
 * builds them; embedIo forwards them verbatim. embedIo never interprets their fields.
 *
 * A native ES module (private submodule of the MolView module, frontend-module-architecture.md
 * §4).  engine.js IMPORTS it directly; the browser-global publish at the bottom is a TEST SEAM
 * (tests/test_render_engine_embed_io_js.py reads molview.renderEngine.embedIo), not a production edge.
 */
"use strict";

// The 3Dmol XYZ wire format for one frame: "<n>\n<comment>\n<el x y z>...".
// The ONLY format knowledge in the engine; kept here so process.js/engine.js stay data-only.
function _buildXyz(elements, positions) {
    var n = positions.length;
    var lines = [String(n), "molview"];
    for (var i = 0; i < n; i++) {
        var p = positions[i];
        lines.push((elements[i] || "X") + " " + p[0] + " " + p[1] + " " + p[2]);
    }
    return lines.join("\n");
}

// Apply the CURRENT-frame overlays the engine hands us. Each field is forwarded to its
// embed door ONLY when present (an absent field leaves that door untouched -- so an
// overlay refresh that changes only labels doesn't clear the selection glow, and vice versa).
function _applyOverlays(handle, o) {
    o = o || {};
    if (o.labels    !== undefined && typeof handle.setAtomLabels === "function") handle.setAtomLabels(o.labels);
    if (o.selection !== undefined && typeof handle.setSelectionHalo === "function") handle.setSelectionHalo(o.selection);
    if (o.arrows    !== undefined && typeof handle.setArrows === "function") handle.setArrows(o.arrows);
    if (o.cellVisible !== undefined && typeof handle.setCell === "function") handle.setCell(!!o.cellVisible);
    if (o.axes    !== undefined && typeof handle.setAxes     === "function") handle.setAxes(o.axes);
}

function createEmbedIo(handle) {
    if (!handle) throw new Error("engine.embedIo.create: a viewer handle is required");

    // §1/§5: a MolView update prepares all data in one run, then 3Dmol renders ONCE. The embed
    // batches renders behind a depth counter; embedIo opens a batch around any multi-door op,
    // and the engine (§8) wraps a whole tier update, so the paint fires once at the outermost
    // close instead of once per setStructure/setOverlays/setCell/setAxes/setLabels door.
    function beginBatch() { if (typeof handle.beginBatch === "function") handle.beginBatch(); }
    function endBatch()   { if (typeof handle.endBatch   === "function") handle.endBatch(); }

    // STRUCTURAL REGEN (§3, §8): load all processed frames as ONE native movie -- COORDINATES
    // only. setStructure(frame0) establishes atom identity + count; a multi-frame set becomes
    // the native movie (addModelsAsFrames) with the per-frame arrows BAKED in (a native swap
    // shows frame i's arrows for free, §3). Labels/halos (and a single frame's arrows) are the
    // ENGINE's job -- it applies the shown frame's overlays right after this (the single render
    // place). loadFrames does NOT apply them; doing so would just double-draw frame 0.
    function loadFrames(movie) {
        movie = movie || {};
        var frames = Array.isArray(movie.frames) ? movie.frames : [];
        if (!frames.length) return;
        beginBatch();                                  // setStructure + setAnimation -> ONE render
        try {
            var f0 = frames[0];
            handle.setStructure({
                xyz:     _buildXyz(f0.elements, f0.positions),
                cellBox: movie.cellBox !== undefined ? movie.cellBox : null,
            });
            if (frames.length > 1 && typeof handle.setAnimation === "function") {
                handle.setAnimation({
                    kind:           "trajectory",
                    frames:         frames.map(function (f) { return f.positions; }),
                    arrowsPerFrame: frames.map(function (f) { return f.arrows || []; }),
                    frameStrip:     false,
                    paused:         true,
                });
            }
        } finally { endBatch(); }
    }

    // NATIVE SWAP (§3, §8): switch to a pre-parsed frame -- no processing, no rebuild.
    function swapFrame(i) {
        if (typeof handle.setAnimationFrame === "function") handle.setAnimationFrame(i);
    }

    // APPEND (§6.2, §8): extend the movie with the processed NEW frames only. Does not
    // move the shown frame. Arrows for the new frames append alongside.
    function appendFrames(tail) {
        tail = tail || {};
        var frames = Array.isArray(tail.frames) ? tail.frames : [];
        if (!frames.length) return;
        if (typeof handle.appendFrames === "function") {
            handle.appendFrames(frames.map(function (f) { return f.positions; }));
        }
        if (typeof handle.appendFrameArrows === "function") {
            handle.appendFrameArrows(frames.map(function (f) { return f.arrows || []; }));
        }
    }

    // OVERLAY REFRESH (§8): (re)apply overlays on the current frame without rebuilding
    // the movie. `overlay` is { labels?, halos?, arrows?, cellVisible?, axes? }.
    function applyOverlays(overlay) {
        beginBatch();                                  // labels + halos + cell + axis -> ONE render
        try { _applyOverlays(handle, overlay); } finally { endBatch(); }
    }

    // OVERLAY REFRESH of the BAKED per-frame arrows (§8): re-hand the whole arrowsPerFrame
    // set for the movie WITHOUT reparsing coordinates (setAnimation partial update). This is
    // how a force overlay/scale change stays an overlay refresh -- the arrows are baked per
    // frame (so a native swap shows them free), but re-baking them does not touch the coords.
    // Multi-frame only; a static structure's arrows go through applyOverlays.
    function setFrameArrows(arrowsPerFrame) {
        if (typeof handle.setAnimation === "function") {
            handle.setAnimation({ arrowsPerFrame: Array.isArray(arrowsPerFrame) ? arrowsPerFrame : [] });
        }
    }

    // The §4 busy scrim. `null` clears it.
    function setBusy(msg) {
        if (typeof handle.setBusy === "function") handle.setBusy(msg);
    }

    // Reads -- the engine's two probes of what the movie can actually do (see the header).
    function frameCount() {
        return (typeof handle.getFrameCount === "function") ? handle.getFrameCount() : 0;
    }
    function animationKind() {
        return (typeof handle.getAnimationKind === "function") ? handle.getAnimationKind() : null;
    }

    return {
        loadFrames:    loadFrames,
        // Cell GEOMETRY only ({lattice, origin} | null) -- the periodicity
        // tier (engine.setCell).  Visibility stays on the overlay pass.
        setCellGeometry: function (box) {
            if (typeof handle.setCellBox === "function") {
                handle.setCellBox(box || null);
            }
        },
        swapFrame:     swapFrame,
        appendFrames:  appendFrames,
        applyOverlays: applyOverlays,
        setFrameArrows: setFrameArrows,
        setBusy:       setBusy,
        beginBatch:    beginBatch,   // engine wraps a whole §8 tier update -> ONE render
        endBatch:      endBatch,
        frameCount:    frameCount,
        animationKind: animationKind,
    };
}

export const embedIo = { create: createEmbedIo };

// TEST SEAM: tests/test_render_engine_embed_io_js.py reads globalThis.molbuilder.molview.renderEngine.embedIo.
// engine.js imports the export above; production reads no global.  Window-guarded.
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.renderEngine = window.molbuilder.molview.renderEngine || {};
    window.molbuilder.molview.renderEngine.embedIo = embedIo;
}
