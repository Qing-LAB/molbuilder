/* Overlay controller for MolView (molview-module.md §14.5.1).
 *
 * MolView is a VIEWER: it DRAWS overlays it is HANDED — it does NOT generate them.  The
 * CONSUMER computes WHAT to draw (e.g. force arrows from its own force data, with its OWN
 * scaling/normalization; or atom-index labels) and pushes it in via the handle's
 * `setArrows` / `setLabels`.  This controller only:
 *   - forwards the consumer's arrows/labels to the embed (`handle.setArrows` / `handle.setLabels`),
 *   - REMEMBERS the last set and re-applies it via `refresh()` — a redraw (`setStructure`) clears
 *     the embed's overlays, so the RENDER STREAMLINE calls refresh() right after it redraws
 *     (render.js afterRedraw), keeping the consumer's overlay visible across frame/structure
 *     changes.  This controller holds NO store subscription of its own: it must NOT re-apply on
 *     unrelated store changes (a mode switch / selection click don't redraw), and doing so used
 *     to clobber the "Show labels"/"Show overlay" VIEW toggles (which drive the same embed doors).
 *
 * It reads NO force data and synthesizes NO geometry.  An "arrow" here is an opaque spec the
 * embed understands (`{start,end,color,radius}`); a "label" spec is whatever `setLabels` takes.
 * MolView neither builds nor normalizes them — that is the consumer's logic and needs.
 *
 *   molview.mountOverlays(handle) -> { setArrows, setLabels, refresh, dispose }
 */
(function (root) {
    "use strict";

    function _noop() {}

    function mountOverlays(handle) {
        if (!handle) {
            return { setArrows: _noop, setLabels: _noop, refresh: _noop, dispose: _noop };
        }
        var _arrows = null;      // last arrow set the CONSUMER handed us (opaque specs)
        var _labels = null;      // last label spec the CONSUMER handed us
        var disposed = false;

        // Re-apply ONLY what the consumer actually handed us.  ``null`` = never set:
        // in that case do NOT touch the embed's setArrows/setLabels, because those
        // doors are ALSO driven by the view toggles ("Show overlay" / "Show labels",
        // VIEW_TOGGLES).  Forcing ``setLabels(false)`` here on any store change (e.g.
        // an Atom-list↔Filter mode switch) wiped the user's "Show labels" toggle --
        // two owners of one door.  The consumer sets a value (or an explicit clearing
        // value) when IT wants to drive the overlay; until then, hands off.
        function _drawArrows() {
            if (_arrows == null) return;
            if (typeof handle.setArrows === "function") handle.setArrows(_arrows);
        }
        function _drawLabels() {
            if (_labels == null) return;
            if (typeof handle.setLabels === "function") handle.setLabels(_labels);
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
        // Re-apply the consumer's last-set overlays.  Called by the render streamline right
        // after a redraw (render.js afterRedraw) -- NOT off a store subscription, so an
        // unrelated store change never re-applies (or clobbers a view toggle).  We do NOT
        // recompute; these are exactly what the consumer last handed us.
        function refresh() { if (!disposed) { _drawArrows(); _drawLabels(); } }

        return {
            setArrows: setArrows,
            setLabels: setLabels,
            refresh:   refresh,
            dispose: function () {
                disposed = true;
                // Clear the consumer's overlays on teardown -- but only if the consumer set
                // them (don't force-clear a view toggle's labels the consumer never owned).
                try { if (_arrows != null && typeof handle.setArrows === "function") handle.setArrows([]); } catch (_) {}
                try { if (_labels != null && typeof handle.setLabels === "function") handle.setLabels(false); } catch (_) {}
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
