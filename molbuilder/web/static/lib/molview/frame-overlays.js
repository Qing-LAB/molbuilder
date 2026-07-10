/* Frame-scoped overlays for MolView (molview-module.md §14.5.1).
 *
 * Two overlays that decorate the CURRENT frame, both via the embed's existing declarative API
 * (setLabels / setArrows) -- no reinvented 3Dmol:
 *   - atom-index LABELS: frame-independent data (the label "3" is always atom 3), positioned
 *     at the atoms -> they auto-follow the frame.  A showIndices toggle turns them on/off.
 *   - force ARROWS: per-frame VALUES (workspace.currentForces()), so they are rebuilt on every
 *     frame change: an arrow from each atom along its force.  A showForces toggle gates them.
 *
 * The host owns the toggle widgets; MolView owns the drawing.  This controller subscribes to
 * the store (a frame's coord swap notifies it) and redraws.
 *
 *   molview.mountFrameOverlays(handle, workspace, store, { getShowForces, getShowIndices, scale })
 *     -> { refresh, dispose }
 */
(function (root) {
    "use strict";

    var FORCE_COLOR = "#f0a020";

    function mountFrameOverlays(handle, workspace, store, opts) {
        opts = opts || {};
        if (!handle || !store || typeof store.subscribe !== "function") {
            return { refresh: function () {}, dispose: function () {} };
        }
        var getForces  = typeof opts.getShowForces === "function"
            ? opts.getShowForces : function () { return false; };
        var getIndices = typeof opts.getShowIndices === "function"
            ? opts.getShowIndices : function () { return false; };
        var scale = (typeof opts.scale === "number" && opts.scale > 0) ? opts.scale : 1.0;
        var disposed = false;

        // The CURRENT frame's coords are the store atoms (setFrame swapped them, workspace §1.5).
        function _coords() {
            var st = store.getState();
            var atoms = (st && st.atoms) || [];
            return atoms.map(function (a) { return [a.x, a.y, a.z]; });
        }

        // One arrow per atom that has a non-negligible force, from the atom along the vector.
        function _forceArrows(coords, forces) {
            var arrows = [];
            for (var i = 0; i < coords.length && i < forces.length; i++) {
                var f = forces[i];
                if (!Array.isArray(f)) continue;
                var mag = Math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
                if (mag < 1e-9) continue;               // no arrow for a ~zero force
                var p = coords[i];
                arrows.push({
                    start: [p[0], p[1], p[2]],
                    end:   [p[0] + f[0] * scale, p[1] + f[1] * scale, p[2] + f[2] * scale],
                    color: FORCE_COLOR,
                    radius: 0.05,
                });
            }
            return arrows;
        }

        function render() {
            if (disposed) return;
            // Index labels -- re-applied to match the toggle (idempotent; they follow the atoms).
            if (typeof handle.setLabels === "function") {
                handle.setLabels(getIndices()
                    ? { atoms: "all", format: "index" }
                    : false);
            }
            // Force arrows -- per-frame values, rebuilt each change.
            if (typeof handle.setArrows === "function") {
                var forces = (typeof workspace.currentForces === "function")
                    ? workspace.currentForces() : null;
                handle.setArrows(
                    (getForces() && Array.isArray(forces))
                        ? _forceArrows(_coords(), forces)
                        : []);
            }
        }

        var unsub = store.subscribe(render);
        render();   // initial

        return {
            refresh: render,
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
    root.molbuilder.molview.mountFrameOverlays = mountFrameOverlays;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { mountFrameOverlays: mountFrameOverlays };
    }
})(typeof window !== "undefined" ? window : this);
