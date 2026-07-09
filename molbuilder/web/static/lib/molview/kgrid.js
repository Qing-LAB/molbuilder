/* k-grid tiling -- the k-grid STEP of the render pipeline
 * (molview-module.md §14).
 *
 * Duplicates the unit cell (the already-selected/visible atoms) in space by the
 * lattice vectors, nx*ny*nz times, so a designed periodic model can be visually
 * validated: vacuum spacing (zero vs non-zero), cell orientation, and how a
 * cell's boundary meets its images.  The images are DISPLAY-ONLY -- they never
 * enter the store or selection/measurement; `sourceIndex[m]` maps image m back
 * to the unit-cell atom it copies so the renderer looks up element/style there.
 * Pure -- no DOM/3Dmol.
 *
 *   molbuilder.molview.tileKgrid(coords, cell, kgrid) -> { positions, sourceIndex, nimages }
 *     coords : Array of [x, y, z]  (the unit-cell atoms)
 *     cell   : [a, b, c] lattice vectors, each [x, y, z]  (only used when a k-grid
 *              dimension > 1; ignored for [1,1,1])
 *     kgrid  : [nx, ny, nz]  (repetitions; each floored + clamped to >= 1)
 *   positions   : coords.length * nx*ny*nz entries; image (i,j,k) is offset by
 *                 i*a + j*b + k*c.  The original cell (0,0,0) comes first.
 *   sourceIndex : positions[m] is a copy of coords[sourceIndex[m]]
 *   nimages     : nx*ny*nz
 */
(function (root) {
    "use strict";

    function _normKgrid(kgrid) {
        const g = Array.isArray(kgrid) ? kgrid : [];
        const out = [1, 1, 1];
        for (let d = 0; d < 3; d++) {
            let n = Math.floor(Number(g[d]));
            if (!isFinite(n) || n < 1) n = 1;   // 0 / negative / NaN -> no tiling
            out[d] = n;
        }
        return out;
    }

    function _validCell(cell) {
        if (!Array.isArray(cell) || cell.length < 3) {
            throw new Error(
                "tileKgrid: a k-grid dimension > 1 needs a cell (3 lattice vectors)");
        }
        for (let d = 0; d < 3; d++) {
            const row = cell[d];
            if (!Array.isArray(row) || row.length < 3 ||
                !(typeof row[0] === "number" && isFinite(row[0])) ||
                !(typeof row[1] === "number" && isFinite(row[1])) ||
                !(typeof row[2] === "number" && isFinite(row[2]))) {
                throw new Error(
                    "tileKgrid: cell row " + d + " must be [x, y, z] finite numbers");
            }
        }
        return cell;
    }

    function tileKgrid(coords, cell, kgrid) {
        if (!Array.isArray(coords)) {
            throw new Error("tileKgrid: coords must be an array of [x,y,z]");
        }
        const k = _normKgrid(kgrid);
        const nimages = k[0] * k[1] * k[2];

        // Identity fast-path: no tiling -> the unit cell unchanged (no cell
        // required).  This is the static / no-k-grid case.
        if (nimages === 1) {
            return {
                positions:   coords.map(function (p) { return [p[0], p[1], p[2]]; }),
                sourceIndex: coords.map(function (_, i) { return i; }),
                nimages:     1,
            };
        }

        const c = _validCell(cell);
        const a = c[0], b = c[1], cc = c[2];
        const positions = [];
        const sourceIndex = [];
        // (0,0,0) first -> the original cell leads; then its images.
        for (let i = 0; i < k[0]; i++) {
            for (let j = 0; j < k[1]; j++) {
                for (let l = 0; l < k[2]; l++) {
                    const ox = i * a[0] + j * b[0] + l * cc[0];
                    const oy = i * a[1] + j * b[1] + l * cc[1];
                    const oz = i * a[2] + j * b[2] + l * cc[2];
                    for (let m = 0; m < coords.length; m++) {
                        const p = coords[m];
                        positions.push([p[0] + ox, p[1] + oy, p[2] + oz]);
                        sourceIndex.push(m);
                    }
                }
            }
        }
        return { positions: positions, sourceIndex: sourceIndex, nimages: nimages };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.tileKgrid = tileKgrid;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { tileKgrid: tileKgrid };
    }
})(typeof window !== "undefined" ? window : globalThis);
