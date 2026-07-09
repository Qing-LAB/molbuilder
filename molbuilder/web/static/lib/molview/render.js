/* molview render loop -- "molview owns the render" (molview-module.md §14).
 *
 * Given a viewer HANDLE + the workspace + the selection store, this draws the current
 * structure once and re-draws on every workspace change -- reading EVERYTHING through ws.*
 * (no local structure copy; that is the data-unification the design mandates, §14.0 "always
 * recompute from the CLEAN unit cell" = the data model).
 *
 * The render is ONE coordinate pipeline -> ONE 3dmol draw (§14.0).  It is a READ-ONLY view
 * derivation: it GENERATES the coordinate list 3dmol draws from the stored atoms; it never
 * writes the data (isolate/k-grid change the render list, never the dataset).
 *   ws / store atoms       -> the clean unit-cell atoms (the source of truth)
 *   drawBase()             -> the plain unit cell (when neither isolate nor k-grid is on)
 *   mountKgridRender(..)   -> REPLACES it with the derived list when isolate (filtered to the
 *                             selected atoms) or k-grid (tiled) is on -- one setStructure, the
 *                             derived list instead of the base, NOT a second draw on top (§14.2)
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

        // Build an xyz body from the atoms THEMSELVES -- never a canvas text that a
        // store-only load (loadFromText, touchCanvas:false) leaves stale.
        function _buildXyz(elements, coords) {
            var lines = [String(coords.length), "molview"];
            for (var i = 0; i < coords.length; i++) {
                var p = coords[i];
                lines.push((elements[i] || "X") + " " + p[0] + " " + p[1] + " " + p[2]);
            }
            return lines.join("\n");
        }
        // The clean unit cell, read from the STORE -- the SAME atoms the panel lists, so the
        // viewer and the panel can never diverge (the data-consistency the design mandates).
        function getUnit() {
            var st = store.getState();
            var atoms = (st && Array.isArray(st.atoms)) ? st.atoms : [];
            if (!atoms.length) return null;
            var coords   = atoms.map(function (a) { return [a.x, a.y, a.z]; });
            var elements = atoms.map(function (a) { return a.element; });
            return { coords: coords, elements: elements, xyz: _buildXyz(elements, coords) };
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

        // Base draw: the plain current structure (all atoms, from the store).  When isolate
        // or k-grid is on, mountKgridRender REPLACES this with the derived list (§14.2).
        function drawBase() {
            var u = getUnit();
            if (u && typeof handle.setStructure === "function") {
                handle.setStructure({ xyz: u.xyz, lattice: getCell() || undefined });
            }
        }

        // k-grid controller: tiles the CLEAN unit cell by periodicity.kgrid when the store's
        // enable toggle is on (§14.2).  We hand it getKgridDims so dims come from the data
        // model, not the store (no back-compat fallback).
        var kg = (typeof mv.mountKgridRender === "function")
            ? mv.mountKgridRender(handle, store, {
                  getUnit: getUnit, getCell: getCell, getKgridDims: getKgridDims,
                  // Restore path (isolate + k-grid both off) -> the plain base draw.
                  drawBase: drawBase,
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

        // Re-draw the BASE only when the STRUCTURE (atoms or cell) changes -- NOT on every
        // selection click.  Selection/isolate overlays are the viewer-adapter's job; k-grid
        // is kg's.  A store-only load fires store.subscribe; a periodicity edit fires
        // ws.subscribe; a plain click fires store.subscribe but leaves the signature the same.
        var _prevSig = null;
        function _structureSig() {
            var st = store.getState();
            var atoms = (st && st.atoms) || [];
            var sig = String(atoms.length);
            for (var i = 0; i < atoms.length; i++) {
                var a = atoms[i];
                sig += "|" + a.element + "," + a.x + "," + a.y + "," + a.z;
            }
            var cell = getCell();
            return sig + "|cell:" + (cell ? JSON.stringify(cell) : "none");
        }
        function redrawIfStructureChanged() {
            var sig = _structureSig();
            if (sig === _prevSig) return;
            _prevSig = sig;
            drawBase();
            if (kg && typeof kg.refresh === "function") kg.refresh();
        }
        redrawIfStructureChanged();   // initial draw

        var storeUnsub = (typeof store.subscribe === "function")
            ? store.subscribe(redrawIfStructureChanged) : _noop;
        var wsUnsub = (typeof workspace.subscribe === "function")
            ? workspace.subscribe(redrawIfStructureChanged) : _noop;

        return {
            refresh: redrawIfStructureChanged,
            dispose: function () {
                try { storeUnsub(); } catch (_) {}
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
