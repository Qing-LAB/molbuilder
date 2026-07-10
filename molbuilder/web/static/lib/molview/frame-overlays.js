/* Overlay controller for MolView (molview-module.md §14.5.1).
 *
 * MolView is a VIEWER: it DRAWS overlays it is HANDED — it does NOT generate them.  The
 * CONSUMER computes WHAT to draw (e.g. force arrows from its own force data, with its OWN
 * scaling/normalization; or atom-index labels) and pushes it in via the handle's
 * `setArrows` / `setLabels`.  This controller only:
 *   - forwards the consumer's arrows/labels to the embed (`handle.setArrows` / `handle.setLabels`),
 *   - REMEMBERS the last set and re-applies it after an internal redraw — a per-frame
 *     `setStructure` clears the embed's overlays, so re-applying keeps the consumer's overlay
 *     visible across frame changes WITHOUT the viewer knowing what it means.
 *
 * It reads NO force data and synthesizes NO geometry.  An "arrow" here is an opaque spec the
 * embed understands (`{start,end,color,radius}`); a "label" spec is whatever `setLabels` takes.
 * MolView neither builds nor normalizes them — that is the consumer's logic and needs.
 *
 *   molview.mountOverlays(handle, store) -> { setArrows, setLabels, refresh, dispose }
 */
(function (root) {
    "use strict";

    function _noop() {}

    function mountOverlays(handle, store) {
        if (!handle) {
            return { setArrows: _noop, setLabels: _noop, refresh: _noop, dispose: _noop };
        }
        var _arrows = null;      // last arrow set the CONSUMER handed us (opaque specs)
        var _labels = null;      // last label spec the CONSUMER handed us
        var disposed = false;

        function _drawArrows() {
            if (typeof handle.setArrows === "function") handle.setArrows(_arrows || []);
        }
        function _drawLabels() {
            if (typeof handle.setLabels === "function") handle.setLabels(_labels || false);
        }

        // ── Consumer API: draw EXACTLY what you're handed (no interpretation) ──
        function setArrows(arrows) {
            _arrows = Array.isArray(arrows) ? arrows.slice() : [];
            if (!disposed) _drawArrows();
        }
        function setLabels(labels) {
            _labels = labels || false;
            if (!disposed) _drawLabels();
        }
        function refresh() { if (!disposed) { _drawArrows(); _drawLabels(); } }

        // A store change means the render may have re-drawn (a per-frame setStructure clears
        // embed overlays) — re-apply the consumer's last-set overlays so they survive.  We do
        // NOT recompute them; they are exactly what the consumer last handed us.
        var unsub = (store && typeof store.subscribe === "function")
            ? store.subscribe(function () { if (!disposed) { _drawArrows(); _drawLabels(); } })
            : _noop;

        return {
            setArrows: setArrows,
            setLabels: setLabels,
            refresh:   refresh,
            dispose: function () {
                disposed = true;
                try { unsub(); } catch (_) {}
                try { if (typeof handle.setArrows === "function") handle.setArrows([]); } catch (_) {}
                try { if (typeof handle.setLabels === "function") handle.setLabels(false); } catch (_) {}
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mountOverlays = mountOverlays;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { mountOverlays: mountOverlays };
    }
})(typeof window !== "undefined" ? window : this);
