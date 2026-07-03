/* Render-pipeline controller -- RUNS the § 6.3 compose against a live embed
 * (docs/protocols/atom-annotations.md § 6.3).
 *
 * The store is frame-INDEPENDENT; the FrameSet holds per-frame coords.  This
 * controller ties them to a viewer: on any relevant change it resolves the
 * current frame's coords, runs computeRender (selection/isolate filter -> k-grid
 * tile), rebuilds the structure the embed draws (element[sourceIndex[m]] at
 * positions[m]), and hands it to the embed via setStructure.  Decorations
 * (labels/arrows) are the renderer's, keyed off sourceIndex.
 *
 *   molbuilder.molview.createRenderController({handle, frameSet, store, elements, cell})
 *     handle   : embed handle exposing setStructure({xyz})
 *     frameSet : molview.createFrameSet(...) (layer 1)
 *     store    : selection store (getState() -> {indices, isolate, kgrid}, subscribe(fn))
 *     elements : per-GLOBAL-atom element symbols (identity, from the store side)
 *     cell     : lattice vectors for k-grid (may be null when k-grid unused)
 *   -> { render(), setFrame(t), dispose() }
 *
 * Re-renders on store change (selection/isolate/kgrid) + on setFrame (the
 * animation-acceleration boundary: a frame tick swaps coords only; a store or
 * cell/kgrid change rebuilds the displayed set).
 */
(function (root) {
    "use strict";

    function _buildXyz(elements, positions, sourceIndex) {
        const n = positions.length;
        const lines = [String(n), ""];
        for (let m = 0; m < n; m++) {
            const el = elements[sourceIndex[m]] || "X";
            const p = positions[m];
            lines.push(el
                + " " + Number(p[0]).toFixed(6)
                + " " + Number(p[1]).toFixed(6)
                + " " + Number(p[2]).toFixed(6));
        }
        return lines.join("\n");
    }

    function createRenderController(opts) {
        opts = opts || {};
        const handle   = opts.handle;
        const frameSet = opts.frameSet;
        const store    = opts.store;
        const elements = opts.elements || [];
        const cell     = opts.cell || null;
        if (!handle || typeof handle.setStructure !== "function") {
            throw new Error("render-controller: handle with setStructure() required");
        }
        if (!frameSet || typeof frameSet.coords !== "function") {
            throw new Error("render-controller: a FrameSet is required");
        }
        if (!store || typeof store.getState !== "function") {
            throw new Error("render-controller: a store with getState() is required");
        }
        const compute = root.molbuilder && root.molbuilder.molview
                      && root.molbuilder.molview.computeRender;
        if (typeof compute !== "function") {
            throw new Error("render-controller: molview.computeRender not loaded");
        }

        let disposed = false;

        function render() {
            if (disposed) return;
            const coords = frameSet.coords() || [];
            const out = compute(coords, store.getState(), cell);
            handle.setStructure({ xyz: _buildXyz(elements, out.positions, out.sourceIndex) });
        }

        // Subscribe for store changes.  The store fires immediately on
        // subscribe; skip that (we do an explicit initial render) so we don't
        // double-render, then re-render on every subsequent change.
        let _primed = false;
        let _unsub = null;
        if (typeof store.subscribe === "function") {
            _unsub = store.subscribe(function () { if (_primed) render(); });
        }
        _primed = true;
        render();   // initial

        return {
            render: render,
            setFrame: function (t) {
                if (typeof frameSet.setFrame === "function") frameSet.setFrame(t);
                render();
            },
            dispose: function () {
                disposed = true;
                if (_unsub) { try { _unsub(); } catch (_) {} }
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.createRenderController = createRenderController;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { createRenderController: createRenderController };
    }
})(typeof window !== "undefined" ? window : globalThis);
