/* VibrationView mode math -- PURE (no DOM/3Dmol), node-testable (vibrationview.md §2/§5).
 *
 * scatterDisplacements: a mode's eigenvector is indexed by FREE-atom ROW (the spectra
 * invariant, docs/tabs/spectra/spec.md §5.1) -- scatter it to GLOBAL atom order so the viewer
 * moves the right atoms and frozen atoms (absent from the free set) stay at zero.  This is the
 * one bit of science-shaped logic VibrationView owns; everything else is the animated view.
 *
 *   molbuilder.vibrationview.scatterDisplacements(displacements, freeAtomIdx, natoms) -> [dx,dy,dz][]
 *     displacements : [dx,dy,dz][] -- either GLOBAL-length (natoms rows, used as-is) or
 *                     FREE-length (one row per free atom, scattered via freeAtomIdx).
 *     freeAtomIdx   : number[] | null -- free-row k maps to global atom freeAtomIdx[k].
 *     natoms        : total atom count (global).
 *   -> a natoms-length array; frozen / un-mapped atoms are [0,0,0] (no motion).
 */
(function (root) {
    "use strict";

    function _copy3(p) {
        return [Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0];
    }

    function scatterDisplacements(displacements, freeAtomIdx, natoms) {
        var n = Math.max(0, Math.floor(Number(natoms) || 0));
        var out = [];
        for (var i = 0; i < n; i++) out.push([0, 0, 0]);
        if (!Array.isArray(displacements)) return out;
        // A free-atom MAP is authoritative when present: scatter free-row k to global atom
        // freeAtomIdx[k] (the rest -- frozen / un-mapped -- stay [0,0,0]).  This matches the
        // original spectra scatter and stays correct even if the free set is a non-identity
        // permutation (a length==natoms shortcut would mis-order it).
        if (Array.isArray(freeAtomIdx)) {
            for (var k = 0; k < freeAtomIdx.length && k < displacements.length; k++) {
                var gi = Math.floor(Number(freeAtomIdx[k]));
                if (gi >= 0 && gi < n && Array.isArray(displacements[k])) {
                    out[gi] = _copy3(displacements[k]);
                }
            }
            return out;
        }
        // No map -> the rows are already in GLOBAL order (one per atom); use them directly.
        if (displacements.length === n) {
            for (var g = 0; g < n; g++) {
                if (Array.isArray(displacements[g])) out[g] = _copy3(displacements[g]);
            }
        }
        return out;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.vibrationview = root.molbuilder.vibrationview || {};
    root.molbuilder.vibrationview.scatterDisplacements = scatterDisplacements;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { scatterDisplacements: scatterDisplacements };
    }
})(typeof window !== "undefined" ? window : globalThis);
