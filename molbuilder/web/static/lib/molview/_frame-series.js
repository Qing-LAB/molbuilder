/* Frame series -- the coordinate time axis (MolView's in-memory data model, molview-module.md
 * §14.5).  This is MolView's own data, NOT the workspace's: the workspace is persistence only
 * (workspace-contract.md), so the coordinate time series lives here in the molview module.
 *
 * PURE (no DOM, no stores): manages frames[] + optional forces[] + a currentFrame pointer,
 * enforcing the SAME-ATOMS invariant -- every frame has the same atom count.  A single static
 * structure is just the one-frame case.
 *
 * This layer validates INTERNAL consistency (all frames same count).  The structure-identity
 * check (element order matches the loaded atoms) lives one layer up, where the atoms are known.
 *
 *   molbuilder.molview._createFrameSeries() -> series
 *   series = { reset, reloadFrames, addFrame, addFrames, setFrame, getFrame, getForces,
 *              currentCoords, currentForces, currentFrame, frameCount, atomCount, clear }
 */
(function (root) {
    "use strict";

    function _copy3s(rows) {
        return rows.map(function (p) { return [Number(p[0]), Number(p[1]), Number(p[2])]; });
    }

    function _valid3s(rows) {
        return Array.isArray(rows) && rows.length > 0 && rows.every(function (p) {
            return Array.isArray(p) && p.length >= 3
                && isFinite(Number(p[0])) && isFinite(Number(p[1])) && isFinite(Number(p[2]));
        });
    }

    function createFrameSeries() {
        var _frames = [];   // Coords[] per frame  (frames[t] = [[x,y,z]...])
        var _forces = [];   // (Vec3[] | null) per frame, parallel to _frames
        var _cur = 0;

        function _atomCount() { return _frames.length ? _frames[0].length : 0; }

        // Validate one frame's coords (and optional forces) against the expected atom count.
        // expectN === null => this frame ESTABLISHES the count (the first frame).
        function _check(coords, forces, expectN) {
            if (!_valid3s(coords)) {
                throw new Error(
                    "frame: coords must be a non-empty array of [x,y,z] finite numbers");
            }
            if (expectN != null && coords.length !== expectN) {
                throw new Error(
                    "frame: atom count " + coords.length + " does not match the structure's "
                    + expectN + " atoms (same-atoms invariant, workspace-contract.md §1.5)");
            }
            if (forces != null) {
                if (!_valid3s(forces) || forces.length !== coords.length) {
                    throw new Error(
                        "frame: forces must be one [fx,fy,fz] per atom (" + coords.length + ")");
                }
            }
        }

        // Replace the whole series (a load / a HARD reload).  The atom count is set by frame 0.
        function reset(frames, opts) {
            opts = opts || {};
            if (!Array.isArray(frames) || frames.length < 1) {
                throw new Error("frame series: reset needs at least one frame");
            }
            var forcesList = Array.isArray(opts.forces) ? opts.forces : null;
            var n = frames[0].length;
            var nf = [], ff = [];
            for (var i = 0; i < frames.length; i++) {
                var fr = forcesList ? (forcesList[i] || null) : null;
                _check(frames[i], fr, n);
                nf.push(_copy3s(frames[i]));
                ff.push(fr ? _copy3s(fr) : null);
            }
            _frames = nf; _forces = ff; _cur = 0;
        }

        // Append one frame (a running job streaming a step).  The first frame sets the count.
        function addFrame(coords, opts) {
            opts = opts || {};
            var fr = (opts.forces != null) ? opts.forces : null;
            _check(coords, fr, _frames.length ? _atomCount() : null);
            _frames.push(_copy3s(coords));
            _forces.push(fr ? _copy3s(fr) : null);
        }

        function addFrames(list, opts) {
            opts = opts || {};
            var forcesList = Array.isArray(opts.forces) ? opts.forces : null;
            (Array.isArray(list) ? list : []).forEach(function (coords, i) {
                addFrame(coords, { forces: forcesList ? forcesList[i] : undefined });
            });
        }

        function setFrame(i) {
            var idx = Math.floor(Number(i));
            if (!(idx >= 0 && idx < _frames.length)) {
                throw new Error("frame series: setFrame(" + i + ") out of range [0.."
                    + (_frames.length - 1) + "]");
            }
            _cur = idx;
            return _cur;
        }

        return {
            reset:         reset,
            reloadFrames:  reset,          // a HARD reload IS a full replace
            addFrame:      addFrame,
            addFrames:     addFrames,
            setFrame:      setFrame,
            getFrame:      function (i) { return _frames[i] ? _copy3s(_frames[i]) : null; },
            getForces:     function (i) { return _forces[i] ? _copy3s(_forces[i]) : null; },
            currentCoords: function () { return _frames[_cur] ? _copy3s(_frames[_cur]) : null; },
            currentForces: function () { return _forces[_cur] ? _copy3s(_forces[_cur]) : null; },
            currentFrame:  function () { return _cur; },
            frameCount:    function () { return _frames.length; },
            atomCount:     _atomCount,
            clear:         function () { _frames = []; _forces = []; _cur = 0; },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview._createFrameSeries = createFrameSeries;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { createFrameSeries: createFrameSeries };
    }
})(typeof window !== "undefined" ? window : globalThis);
