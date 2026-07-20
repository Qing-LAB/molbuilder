/* MolView render engine -- embedIo: the ONE seal over the 3Dmol embed handle.
 *
 * Contract: docs/protocols/molview-render-streamline.md §9.1 (and §3, §4, §8).
 * Module:   molbuilder.molview.engine.embedIo   (lib/molview/engine/embed-io.js)
 * Used by:  engine.js -- the orchestrator (§9), the ONLY caller. Nothing else in MolView
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
 * Reads (single-owner: the native movie is the coord owner, §7.1):
 *   frameCount() / currentFrame() / animationKind().
 *
 * INPUT SHAPES (all plain data -- no 3Dmol objects, no DOM):
 *   Movie = {
 *     frames:  ProcessedFrame[],     // §7.3; frames[0] establishes atom identity + count.
 *     cell:    lattice|null,         // explicit a/b/c lattice (the setStructure `lattice`).
 *     cellBox: {lattice,origin}|null // the resolved box that WRAPS the atoms.
 *   }
 *   ProcessedFrame = { positions:Vec3[], elements:string[],
 *                      arrows:Arrow[]|null, labels:LabelOpts|null, halos:HaloOverlay|null }
 *   FrameTail = { frames: ProcessedFrame[] }               // the NEW frames only (§6.2).
 *   Overlay   = { labels?, halos?, arrows?, cellBox?, axes? } // for the CURRENT frame; a field
 *                                                             // that is `undefined` is not touched.
 *
 * The Arrow / LabelOpts / HaloOverlay / cellBox specs are exactly what the embed doors accept
 * (setArrows / setLabels / setOverlays / setCell) -- process.js builds them; embedIo forwards
 * them verbatim. embedIo never interprets their fields.
 */
(function (root) {
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
    // overlay refresh that changes only labels doesn't clear halos, and vice versa).
    function _applyOverlays(handle, o) {
        o = o || {};
        if (o.labels  !== undefined && typeof handle.setLabels   === "function") handle.setLabels(o.labels);
        if (o.halos   !== undefined && typeof handle.setOverlays === "function") handle.setOverlays(o.halos);
        if (o.arrows  !== undefined && typeof handle.setArrows   === "function") handle.setArrows(o.arrows);
        if (o.cellBox !== undefined && typeof handle.setCell     === "function") handle.setCell(o.cellBox);
        if (o.axes    !== undefined && typeof handle.setAxes     === "function") handle.setAxes(o.axes);
    }

    // The CURRENT-frame overlay spec carried ON a ProcessedFrame (labels/halos/arrows).
    // Scene-level cell/axis are handed separately by the engine (they are not per-atom).
    function _overlayOf(pf) {
        return { labels: pf.labels, halos: pf.halos, arrows: pf.arrows };
    }

    function createEmbedIo(handle) {
        if (!handle) throw new Error("engine.embedIo.create: a viewer handle is required");

        // STRUCTURAL REGEN (§3, §8): load all processed frames as ONE native movie.
        //   1 frame  -> a plain static structure (no frame bar; frameCount === 1).
        //   N frames -> setStructure(frame0) establishes identity + count, then setAnimation
        //               builds the native multi-frame movie (addModelsAsFrames) with the
        //               per-frame arrows baked in. Overlays for the shown frame ride on top.
        // The engine ALWAYS hands ProcessedFrame[]; the static-vs-movie split is embedIo's
        // 3Dmol-format detail, not a second render path.
        function loadFrames(movie) {
            movie = movie || {};
            var frames = Array.isArray(movie.frames) ? movie.frames : [];
            if (!frames.length) return;
            var f0 = frames[0];
            handle.setStructure({
                xyz:     _buildXyz(f0.elements, f0.positions),
                lattice: movie.cell !== undefined ? movie.cell : null,
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
            // Frame-0 labels/halos (per-frame arrows are baked above via arrowsPerFrame).
            _applyOverlays(handle, _overlayOf(f0));
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
        // the movie. `overlay` is { labels?, halos?, arrows?, cellBox?, axes? }.
        function applyOverlays(overlay) {
            _applyOverlays(handle, overlay);
        }

        // The §4 busy scrim. `null` clears it.
        function setBusy(msg) {
            if (typeof handle.setBusy === "function") handle.setBusy(msg);
        }

        // Reads -- the native movie is the single coord owner (§7.1).
        function frameCount() {
            return (typeof handle.getFrameCount === "function") ? handle.getFrameCount() : 0;
        }
        function currentFrame() {
            return (typeof handle.getAnimationFrame === "function") ? handle.getAnimationFrame() : 0;
        }
        function animationKind() {
            return (typeof handle.getAnimationKind === "function") ? handle.getAnimationKind() : null;
        }

        return {
            loadFrames:    loadFrames,
            swapFrame:     swapFrame,
            appendFrames:  appendFrames,
            applyOverlays: applyOverlays,
            setBusy:       setBusy,
            frameCount:    frameCount,
            currentFrame:  currentFrame,
            animationKind: animationKind,
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.engine = root.molbuilder.molview.engine || {};
    root.molbuilder.molview.engine.embedIo = { create: createEmbedIo };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { create: createEmbedIo };
    }
})(typeof window !== "undefined" ? window : globalThis);
