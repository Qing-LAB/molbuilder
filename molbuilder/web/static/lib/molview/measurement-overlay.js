/* Measurement overlay -- a module decoration (molview-module.md §15).
 *
 * Paints the geometry readout (1 atom -> position, 2 -> distance, 3 -> angle) as
 * overlay text inside the viewer, derived from the SELECTION (decision A: no
 * separate pick).  Replaces the per-inspector chips/readouts; the math stays in
 * the shared lib/selection/measurements.js.
 *
 *   molview.mountMeasurementOverlay(viewerHost, { store, coordsProvider }) -> { render, dispose }
 *     viewerHost     : the viewer element; the overlay is absolutely positioned in it
 *     store          : selection store -- getState() -> { pickOrder, indices, atoms }, subscribe(fn)
 *     coordsProvider : () -> per-atom [x,y,z] for the CURRENT frame (the viewer handle / a trajectory frame, NOT the store)
 *   render()  : recompute + repaint (call on a frame tick)
 *   dispose() : detach the subscription + remove the overlay element
 *
 * Ordered by store.pickOrder (angle vertex = 2nd pick); geometry computed against
 * coordsProvider() so it's correct at whatever frame is on screen.  Hidden for
 * n atoms outside {1,2,3}.
 */
(function (root) {
    "use strict";

    function mountMeasurementOverlay(viewerHost, opts) {
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
        const meas = root.molbuilder && root.molbuilder.molview && root.molbuilder.molview.selection
                   && root.molbuilder.molview.selection.measurements;

        const el = document.createElement("div");
        el.className = "molview-measurement-overlay selection-measurement-overlay";
        el.hidden = true;
        viewerHost.appendChild(el);

        function render() {
            const s = store.getState() || {};
            // ISOLATE keeps the measurement working: the drawn atoms ARE the selected atoms,
            // the readout is derived from the selection (still curated via the panel), and we
            // simply re-key the coords below (§14.3, molview-module.md).
            // Ordered picks: pickOrder (click order -> angle vertex) else indices.
            const picks = (Array.isArray(s.pickOrder) && s.pickOrder.length)
                ? s.pickOrder
                : (Array.isArray(s.indices) ? s.indices : []);
            if (!meas || picks.length < 1 || picks.length > 3) {
                el.hidden = true;
                el.textContent = "";
                return;
            }
            // meas.compute indexes positions[] by GLOBAL atom index.  When isolate is on the
            // viewer draws ONLY the selected atoms, so coordsProvider() returns them in the
            // render controller's VISIBLE order (= s.indices filtered to valid, matching
            // computeRender's isolate filter in render-pipeline.js).  Re-key that filtered
            // list back to global index so each pick reads the right atom's coords.  (Kept
            // in lock-step with computeRender's isolate order.)
            let coords = coordsProvider() || [];
            const isolating = !!s.isolate
                && Array.isArray(s.indices) && s.indices.length > 0;
            if (isolating) {
                const natoms = (s.atoms || []).length;
                const visible = s.indices.filter(function (i) {
                    return i >= 0 && i < natoms;
                });
                const byGlobal = [];
                for (let m = 0; m < visible.length; m++) {
                    byGlobal[visible[m]] = coords[m];
                }
                coords = byGlobal;
            }
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

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mountMeasurementOverlay = mountMeasurementOverlay;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { mountMeasurementOverlay: mountMeasurementOverlay };
    }
})(typeof window !== "undefined" ? window : globalThis);
