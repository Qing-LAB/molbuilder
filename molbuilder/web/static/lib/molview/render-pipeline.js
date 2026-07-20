/* Render pipeline compose -- assembles the render steps (molview-module.md §14) into the
 * set of positions the viewer draws.
 *
 * The store is frame-INDEPENDENT; the caller resolves the current frame's coords
 * (layer 1 -- e.g. the viewer handle's getAtomCoords, or a trajectory's frame)
 * and passes the store view snapshot.  This composes the remaining layer, PURELY:
 *
 *   2 selection / isolate -> which GLOBAL atom indices are visible
 *   (3 decorations: labels/arrows are applied by the renderer via sourceIndex)
 *
 *   molbuilder.molview.computeRender(coords, view) -> { positions, sourceIndex }
 *     coords : per-atom [x,y,z] for the current frame (layer 1 output)
 *     view   : { indices:[...selection], isolate:bool }
 *   positions   : the [x,y,z] to draw
 *   sourceIndex : positions[m] belongs to GLOBAL atom sourceIndex[m] -- the
 *                 renderer looks up element/style + decorations there.
 *
 * (k-grid tiling used to be layer 3 here; k-grid is a reciprocal-space SAMPLING
 * knob on SiestaConfig, not geometry, so it no longer lives in the viewer --
 * structure-periodicity.md.)
 */
(function (root) {
    "use strict";

    function computeRender(coords, view) {
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
        return { positions: visibleCoords, sourceIndex: visible.slice() };
    }

    // Build an .xyz body from a computeRender result: positions[m] takes its
    // element from the atom sourceIndex[m].  The lattice rides separately on
    // setStructure.
    function _buildXyz(elements, positions, sourceIndex) {
        const n = positions.length;
        const out = [String(n), "molview"];
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
    // and the cell (drawn as the wireframe box on the derived list).
    function _renderSig(unit, effView, cell) {
        let s = "n:" + unit.coords.length;
        for (let i = 0; i < unit.coords.length; i++) {
            const p = unit.coords[i];
            s += "|" + (unit.elements[i] || "X") + "," + p[0] + "," + p[1] + "," + p[2];
        }
        s += "|iso:" + (effView.isolate ? "1" : "0");
        if (effView.isolate) s += "|sel:" + (effView.indices || []).join(",");
        s += "|cell:" + (cell ? JSON.stringify(cell) : "none");
        return s;
    }

    // The ONE structure-view render controller (molview-module.md §14.2).  The render
    // is a READ-ONLY view derivation: it takes the stored unit cell and GENERATES the
    // coordinate list for 3dmol -- it never writes the data.  ONE pipeline (computeRender)
    // -> ONE setStructure:
    //   * isolate off  -> the host's plain base draw (drawBase()).
    //   * isolate on   -> the list is FILTERED to the selected atoms (a real filter --
    //                     not an opacity/hidden trick; the hidden atoms are absent from
    //                     the drawn list).
    // While a derived list is shown the 3-D window is DISPLAY-ONLY for selection:
    // in-window picking + index-keyed decorations stand down (the adapter/measurement
    // gate on isolate), so nothing needs a drawn->global index map.
    //
    //   handle : the viewer handle (setStructure)
    //   store  : the selection store -- isolate flag + selection.
    //   opts.getUnit() -> { coords:[[x,y,z]...], elements:[...], xyz:"<unit xyz>" }
    //            the CURRENT unit-cell structure (host-owned; captured BEFORE any
    //            derivation so it never reads back a derived list from the handle).
    //   opts.getCell() -> 3x3 lattice | null  (drawn as the wireframe box on the derived list).
    //   opts.drawBase() -> the host's plain base draw (all atoms, its own wireframe
    //            cell + refit).  Called to RESTORE when isolate goes off, so the host
    //            keeps its base-draw nuances (e.g. explicit-cell-only box).
    // Returns { refresh, dispose }.  Call refresh() after a base render / structure /
    // periodicity change; call dispose() to unsubscribe.
    function mountIsolateRender(handle, store, opts) {
        opts = opts || {};
        const getUnit = typeof opts.getUnit === "function"
            ? opts.getUnit : function () { return null; };
        const getCell = typeof opts.getCell === "function"
            ? opts.getCell : function () { return null; };
        const drawBase = typeof opts.drawBase === "function" ? opts.drawBase : null;
        let active = false;
        let prevSig = null;
        let disposed = false;

        function _render() {
            if (disposed) return;
            const unit = getUnit();
            if (!unit || !Array.isArray(unit.coords)) return;
            const view = store.getState() || {};
            const cell = getCell();
            const indices = Array.isArray(view.indices) ? view.indices : [];
            // isolate with an EMPTY selection derives nothing (there is no "selected
            // only" of zero atoms) -> fall through to the plain base draw.
            const isolate = !!view.isolate && indices.length > 0;

            if (isolate) {
                const effView = { indices: indices, isolate: true };
                const sig = _renderSig(unit, effView, cell);
                if (active && sig === prevSig) return;   // derived list unchanged -> no redraw
                prevSig = sig;
                const out = computeRender(unit.coords, effView);
                handle.setStructure({
                    xyz: _buildXyz(unit.elements, out.positions, out.sourceIndex),
                    lattice: cell || undefined,
                });
                active = true;
            } else if (active) {
                // Isolate just turned off -> restore the host's plain base draw.
                active = false;
                prevSig = null;
                if (drawBase) drawBase();
                else handle.setStructure({ xyz: unit.xyz, lattice: cell || undefined });
            }
        }

        // NO store subscription: the isolate view is re-rendered ONLY when the render
        // streamline calls refresh() (render.js onStoreChange, which reads the isolate flag
        // + selection and gates busy).  Subscribing here handcrafted a SECOND render outside
        // the streamline -- it raced render.js's base draw (leaving the FULL structure shown
        // after an edit while isolating) and bypassed the busy scrim (froze the tab on a big
        // isolate render).  The controller reads live state on each refresh(), so it needs no
        // subscription of its own.
        return {
            refresh: function () { _render(); },
            dispose: function () { disposed = true; },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.computeRender = computeRender;
    root.molbuilder.molview.mountIsolateRender = mountIsolateRender;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { computeRender: computeRender,
                           mountIsolateRender: mountIsolateRender };
    }
})(typeof window !== "undefined" ? window : globalThis);
