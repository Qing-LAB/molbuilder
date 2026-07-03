/* Measurement overlay -- a module decoration (§ 6.3 layer 4 / § 6.4).
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
        const meas = root.molbuilder && root.molbuilder.selection
                   && root.molbuilder.selection.measurements;

        const el = document.createElement("div");
        el.className = "molview-measurement-overlay selection-measurement-overlay";
        el.hidden = true;
        viewerHost.appendChild(el);

        function render() {
            const s = store.getState() || {};
            // Option 1 (§ 6.3): while k-grid is on we show the tiled supercell,
            // not the unit cell -- per-unit-cell-index readout doesn't map, so
            // the measurement stands down until k-grid is off.
            if (s.kgrid && s.kgrid.enabled) {
                el.hidden = true;
                el.textContent = "";
                return;
            }
            // Ordered picks: pickOrder (click order -> angle vertex) else indices.
            const picks = (Array.isArray(s.pickOrder) && s.pickOrder.length)
                ? s.pickOrder
                : (Array.isArray(s.indices) ? s.indices : []);
            if (!meas || picks.length < 1 || picks.length > 3) {
                el.hidden = true;
                el.textContent = "";
                return;
            }
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

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mountMeasurementOverlay = mountMeasurementOverlay;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { mountMeasurementOverlay: mountMeasurementOverlay };
    }
})(typeof window !== "undefined" ? window : globalThis);
