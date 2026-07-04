/* Render pipeline compose -- assembles the § 6.3 layers into the set of
 * positions the viewer draws (molview-module.md).
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

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.computeRender = computeRender;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { computeRender: computeRender };
    }
})(typeof window !== "undefined" ? window : globalThis);
