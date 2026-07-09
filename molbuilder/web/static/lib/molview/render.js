/* molview render loop -- "molview owns the render" (molview-module.md §14).
 *
 * Given a viewer HANDLE + the workspace + the selection store, this draws the current
 * structure once and re-draws on every workspace change -- reading EVERYTHING through ws.*
 * (no local structure copy; that is the data-unification the design mandates, §14.0 "always
 * recompute from the CLEAN unit cell" = the data model).
 *
 * The render is ONE coordinate pipeline -> ONE 3dmol draw (§14.0):
 *   ws.getStructure()      -> the clean unit-cell atoms (the source of truth)
 *   handle.setStructure(..) -> base draw of that unit cell
 *   mountKgridRender(..)    -> k-grid tiling on top when the store's enable toggle is on (§14.2)
 * plus the measurement overlay (mountMeasurementOverlay, §15), coords read from the handle.
 *
 *   molview.mountRender(handle, workspace, store, { viewerHost }) -> { refresh, dispose }
 *
 * It composes the existing molview pieces (mountKgridRender, mountMeasurementOverlay) -- it
 * does not reimplement them.  mount.js assembles this with the viewer embed to make the
 * full component.
 */
(function (root) {
    "use strict";

    function _noop() {}

    function mountRender(handle, workspace, store, opts) {
        opts = opts || {};
        var mv = (root.molbuilder && root.molbuilder.molview) || {};
        if (!handle || !workspace || !store) {
            return { refresh: _noop, dispose: _noop };
        }

        // ---- Everything read through ws.* -- never a local copy (§14.0) ---------------- //

        // The clean unit cell, straight from the data model.
        function getUnit() {
            var s = (typeof workspace.getStructure === "function") ? workspace.getStructure() : null;
            if (!s || !Array.isArray(s.atoms)) return null;
            return {
                coords:   s.atoms.map(function (a) { return [a.x, a.y, a.z]; }),
                elements: s.atoms.map(function (a) { return a.element; }),
                xyz:      s.text,
            };
        }
        // The resolved lattice (explicit cell, or bbox+vacuum) via the accessor.
        function getCell() {
            if (typeof workspace.getUnitCellInfo === "function") {
                var info = workspace.getUnitCellInfo();
                if (info && info.value) return info.value;
            }
            return (typeof workspace.getUnitCell === "function") ? workspace.getUnitCell() : null;
        }
        // The k-grid DIMS = periodicity.kgrid (the ONE value, §14.0); NOT the store's dims.
        function getKgridDims() {
            return (typeof workspace.getKgrid === "function" && workspace.getKgrid()) || [1, 1, 1];
        }

        // ---- The pipeline -> one draw ------------------------------------------------- //

        // Base draw: the current unit cell.  k-grid tiles ON TOP of this when enabled.
        function drawBase() {
            var u = getUnit();
            if (u && typeof handle.setStructure === "function") {
                handle.setStructure({ xyz: u.xyz, lattice: getCell() || undefined });
            }
        }
        drawBase();

        // k-grid controller: tiles the CLEAN unit cell by periodicity.kgrid when the store's
        // enable toggle is on (§14.2).  We hand it getKgridDims so dims come from the data
        // model, not the store (no back-compat fallback).
        var kg = (typeof mv.mountKgridRender === "function")
            ? mv.mountKgridRender(handle, store, {
                  getUnit: getUnit, getCell: getCell, getKgridDims: getKgridDims,
              })
            : { refresh: _noop, dispose: _noop };

        // Measurement overlay: the ONE overlay (§15); coords come from the viewer handle,
        // never a held copy.  Skipped if the host didn't give a viewerHost to paint into.
        var meas = (typeof mv.mountMeasurementOverlay === "function" && opts.viewerHost)
            ? mv.mountMeasurementOverlay(opts.viewerHost, {
                  store: store,
                  coordsProvider: function () {
                      return (typeof handle.getAtomCoords === "function") ? handle.getAtomCoords() : [];
                  },
              })
            : { dispose: _noop };

        // Re-draw on every workspace change (a load, an edit): base then re-tile.
        function refresh() {
            drawBase();
            if (kg && typeof kg.refresh === "function") kg.refresh();
        }
        var wsUnsub = (typeof workspace.subscribe === "function") ? workspace.subscribe(refresh) : _noop;

        return {
            refresh: refresh,
            dispose: function () {
                try { wsUnsub(); } catch (_) {}
                try { kg.dispose(); } catch (_) {}
                try { meas.dispose(); } catch (_) {}
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mountRender = mountRender;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { mountRender: mountRender };
    }
})(typeof window !== "undefined" ? window : this);
