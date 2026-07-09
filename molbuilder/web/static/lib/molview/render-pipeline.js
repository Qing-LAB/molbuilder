/* Render pipeline compose -- assembles the render steps (molview-module.md §14) into the
 * set of positions the viewer draws.
 *
 * The store is frame-INDEPENDENT; the caller resolves the current frame's coords
 * (layer 1 -- e.g. the viewer handle's getAtomCoords, or a trajectory's frame)
 * and passes the store view snapshot + the lattice cell.  This composes the
 * remaining layers, PURELY:
 *
 *   2 selection / isolate -> which GLOBAL atom indices are visible
 *   3 k-grid tiling        -> duplicate the visible atoms by the lattice
 *   (4 decorations: labels/arrows are applied by the renderer via sourceIndex)
 *
 *   molbuilder.molview.computeRender(coords, view, cell) -> { positions, sourceIndex }
 *     coords : per-atom [x,y,z] for the current frame (layer 1 output)
 *     view   : { indices:[...selection], isolate:bool, kgrid:{enabled,dims} }
 *     cell   : lattice vectors (only used when kgrid.enabled)
 *   positions   : the [x,y,z] to draw
 *   sourceIndex : positions[m] belongs to GLOBAL atom sourceIndex[m] -- the
 *                 renderer looks up element/style + decorations there (k-grid
 *                 images share their unit-cell atom's identity).
 */
(function (root) {
    "use strict";

    function computeRender(coords, view, cell) {
        coords = Array.isArray(coords) ? coords : [];
        view = view || {};
        const natoms = coords.length;

        // Layer 2: selection / isolate -> visible GLOBAL atom indices.  Isolate
        // ("show selected only") with a non-empty selection restricts the view
        // to those atoms; otherwise every atom is visible (selection merely
        // highlights, it does not filter).
        const sel = Array.isArray(view.indices) ? view.indices : [];
        let visible;
        if (view.isolate && sel.length > 0) {
            visible = sel.filter(function (i) { return i >= 0 && i < natoms; });
        } else {
            visible = [];
            for (let i = 0; i < natoms; i++) visible.push(i);
        }
        const visibleCoords = visible.map(function (i) {
            const p = coords[i];
            return [p[0], p[1], p[2]];
        });

        // Layer 3: k-grid tiling (display-only images).
        const kg = view.kgrid || { enabled: false, dims: [1, 1, 1] };
        const tile = root.molbuilder && root.molbuilder.molview
                   && root.molbuilder.molview.tileKgrid;
        if (kg.enabled && typeof tile === "function") {
            const tiled = tile(visibleCoords, cell, kg.dims);
            // tiled.sourceIndex is LOCAL (into visibleCoords) -> map back to the
            // GLOBAL atom index so element/label/arrow lookup stays correct.
            const sourceIndex = tiled.sourceIndex.map(function (k) { return visible[k]; });
            return { positions: tiled.positions, sourceIndex: sourceIndex };
        }
        return { positions: visibleCoords, sourceIndex: visible.slice() };
    }

    // Build an .xyz body from a computeRender result: positions[m] takes its
    // element from the unit-cell atom sourceIndex[m] (k-grid images share their
    // unit-cell atom's element).  The lattice rides separately on setStructure.
    function _buildXyz(elements, positions, sourceIndex) {
        const n = positions.length;
        const out = [String(n), "kgrid"];
        for (let m = 0; m < n; m++) {
            const el = (elements && elements[sourceIndex[m]]) || "X";
            const p = positions[m];
            out.push(el + " " + p[0] + " " + p[1] + " " + p[2]);
        }
        return out.join("\n");
    }

    // A cheap-but-exact fingerprint of the derived render list's INPUTS, so the
    // controller redraws only when the list it would produce actually changes --
    // NOT on every selection click (§14.2).  Inputs: the unit-cell geometry, the
    // isolate flag, the selection WHEN isolating (it changes which atoms survive),
    // the k-grid enable + dims, and the cell.
    function _renderSig(unit, effView, cell) {
        let s = "n:" + unit.coords.length;
        for (let i = 0; i < unit.coords.length; i++) {
            const p = unit.coords[i];
            s += "|" + (unit.elements[i] || "X") + "," + p[0] + "," + p[1] + "," + p[2];
        }
        s += "|iso:" + (effView.isolate ? "1" : "0");
        if (effView.isolate) s += "|sel:" + (effView.indices || []).join(",");
        s += "|kg:" + (effView.kgrid.enabled ? "1" : "0") + ":" + effView.kgrid.dims.join(",");
        s += "|cell:" + (cell ? JSON.stringify(cell) : "none");
        return s;
    }

    // The ONE structure-view render controller (molview-module.md §14.2).  The render
    // is a READ-ONLY view derivation: it takes the stored unit cell and GENERATES the
    // coordinate list for 3dmol -- it never writes the data.  ONE pipeline (computeRender)
    // -> ONE setStructure:
    //   * neither isolate nor k-grid  -> the host's plain base draw (drawBase()).
    //   * isolate on                  -> the list is FILTERED to the selected atoms
    //                                    (a real filter -- not an opacity/hidden trick;
    //                                    the hidden atoms are absent from the drawn list).
    //   * k-grid on                   -> the list is the tiled supercell.
    //   * both                        -> the selected atoms, tiled (§14.3).
    // While a derived list is shown the 3-D window is DISPLAY-ONLY for selection:
    // in-window picking + index-keyed decorations stand down (the adapter/measurement
    // gate on the same isolate||kgrid), so nothing needs a drawn->global index map.
    //
    //   handle : the viewer handle (setStructure)
    //   store  : the selection store -- isolate flag + k-grid ENABLE toggle + selection.
    //   opts.getUnit() -> { coords:[[x,y,z]...], elements:[...], xyz:"<unit xyz>" }
    //            the CURRENT unit-cell structure (host-owned; captured BEFORE any
    //            derivation so it never reads back a supercell from the handle).
    //   opts.getCell() -> 3x3 lattice | null  (the RESOLVED cell, for tiling).
    //   opts.getKgridDims() -> [nx,ny,nz]  the k-grid DIMS = periodicity.kgrid (the DFT
    //            value, unified -- structure-periodicity.md §3b), NOT a view count.
    //            Omit -> fall back to view.kgrid.dims.
    //   opts.drawBase() -> the host's plain base draw (all atoms, its own wireframe
    //            cell + refit).  Called to RESTORE when isolate + k-grid both go off,
    //            so the host keeps its base-draw nuances (e.g. explicit-cell-only box).
    // Returns { refresh, dispose }.  Call refresh() after a base render / structure /
    // periodicity change; call dispose() to unsubscribe.
    function mountKgridRender(handle, store, opts) {
        opts = opts || {};
        const getUnit = typeof opts.getUnit === "function"
            ? opts.getUnit : function () { return null; };
        const getCell = typeof opts.getCell === "function"
            ? opts.getCell : function () { return null; };
        const getKgridDims = typeof opts.getKgridDims === "function"
            ? opts.getKgridDims : null;
        const drawBase = typeof opts.drawBase === "function" ? opts.drawBase : null;
        let active = false;
        let prevSig = null;
        let disposed = false;

        function _render() {
            if (disposed) return;
            const unit = getUnit();
            if (!unit || !Array.isArray(unit.coords)) return;
            const view = store.getState() || {};
            const kg = view.kgrid || {};
            const dims = getKgridDims ? getKgridDims() : (kg.dims || [1, 1, 1]);
            const cell = getCell();
            const indices = Array.isArray(view.indices) ? view.indices : [];
            // isolate with an EMPTY selection derives nothing (there is no "selected
            // only" of zero atoms) -> fall through to the plain base draw.
            const isolate = !!view.isolate && indices.length > 0;
            const tiling  = !!kg.enabled && !!cell;
            const derived = isolate || tiling;

            if (derived) {
                const effView = {
                    indices: indices,
                    isolate: isolate,
                    kgrid:   { enabled: tiling, dims: dims },
                };
                const sig = _renderSig(unit, effView, cell);
                if (active && sig === prevSig) return;   // derived list unchanged -> no redraw
                prevSig = sig;
                const out = computeRender(unit.coords, effView, cell);
                handle.setStructure({
                    xyz: _buildXyz(unit.elements, out.positions, out.sourceIndex),
                    lattice: cell || undefined,
                });
                active = true;
            } else if (active) {
                // Derived view just turned off -> restore the host's plain base draw.
                active = false;
                prevSig = null;
                if (drawBase) drawBase();
                else handle.setStructure({ xyz: unit.xyz, lattice: cell || undefined });
            }
        }

        const unsub = store.subscribe(function () { _render(); });
        return {
            refresh: function () { _render(); },
            dispose: function () {
                disposed = true;
                try { unsub(); } catch (_) { /* already gone */ }
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.computeRender = computeRender;
    root.molbuilder.molview.mountKgridRender = mountKgridRender;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { computeRender: computeRender,
                           mountKgridRender: mountKgridRender };
    }
})(typeof window !== "undefined" ? window : globalThis);
