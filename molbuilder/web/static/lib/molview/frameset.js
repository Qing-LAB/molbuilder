/* FrameSet -- module-owned dynamic coordinates for the fused molview render
 * pipeline (docs/protocols/atom-annotations.md § 6.3).
 *
 * The selection store is frame-INDEPENDENT: atom identity, labels, selection,
 * and isolate are all sets of *indices*, valid for every frame.  The FrameSet
 * is the SEPARATE per-frame coordinate table.  A static structure is simply a
 * 1-frame FrameSet; a trajectory is N frames.  This module is render-pipeline
 * layer 1 (time-index); it has no DOM/3Dmol dependency and is pure.
 *
 *   molbuilder.molview.createFrameSet(frames) -> FrameSet
 *     frames : non-empty Array of frames; each frame is an Array of per-atom
 *              [x, y, z] (every frame shares the same atom count -- identity is
 *              fixed across time).
 *
 * FrameSet:
 *   nframes      : number of frames (>= 1)
 *   natoms       : atoms per frame
 *   isStatic     : nframes === 1
 *   currentFrame : 0-based current time index (starts at 0)
 *   coerce(t)    : the TIME-INDEX layer -- static => 0; else floor + clamp into
 *                  [0, nframes-1]; a non-finite t keeps the current frame.
 *   setFrame(t)  : coerce(t) then set currentFrame; returns the coerced index.
 *   coordsAt(t)  : the coerced frame t's per-atom coords (read-only reference).
 *   coords()     : coordsAt(currentFrame).
 */
(function (root) {
    "use strict";

    function createFrameSet(frames) {
        if (!Array.isArray(frames) || frames.length === 0) {
            throw new Error("createFrameSet: frames must be a non-empty array");
        }
        const nframes = frames.length;
        const natoms  = Array.isArray(frames[0]) ? frames[0].length : 0;
        for (let f = 0; f < nframes; f++) {
            if (!Array.isArray(frames[f]) || frames[f].length !== natoms) {
                throw new Error(
                    "createFrameSet: frame " + f + " has " +
                    (Array.isArray(frames[f]) ? frames[f].length : "non-array") +
                    " atoms, expected " + natoms +
                    " (atom identity is fixed across frames)");
            }
        }

        const fs = {
            nframes:      nframes,
            natoms:       natoms,
            isStatic:     nframes === 1,
            currentFrame: 0,
        };

        // Time-index layer (§ 6.3 layer 1).
        fs.coerce = function (t) {
            if (nframes <= 1) return 0;              // static: t is irrelevant
            let i = Math.floor(Number(t));
            if (!isFinite(i)) i = fs.currentFrame;   // non-finite t: hold frame
            if (i < 0) i = 0;
            if (i > nframes - 1) i = nframes - 1;
            return i;
        };

        fs.setFrame = function (t) {
            fs.currentFrame = fs.coerce(t);
            return fs.currentFrame;
        };

        fs.coordsAt = function (t) { return frames[fs.coerce(t)]; };
        fs.coords   = function ()  { return frames[fs.currentFrame]; };

        return fs;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.createFrameSet = createFrameSet;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { createFrameSet: createFrameSet };
    }
})(typeof window !== "undefined" ? window : globalThis);
