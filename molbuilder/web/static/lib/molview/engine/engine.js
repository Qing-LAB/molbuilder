/* MolView render engine -- engine: the orchestrator (the single render place).
 *
 * Contract: docs/protocols/molview-render-streamline.md §5, §8, §9.
 * Module:   molbuilder.molview.engine.create   (lib/molview/engine/engine.js)
 * Used by:  mount.js (Phase 3) -- the ONE thing that turns data + flags into what 3Dmol shows.
 *
 * It holds the CLEAN StructureData (§7.1) + reads the ViewFlags (§7.2) from the store, runs
 * `process` per frame (§2), and drives `embedIo` (§9.1). Every UI interaction only writes
 * data/flags and asks this to render -- there is no other code path to the 3Dmol engine (§5).
 *
 * The §8 minimal-update tiers, chosen by WHAT changed (never by system size -- no magic number):
 *   - NATIVE SWAP     : currentFrame only (the frame channel) -> embedIo.swapFrame(i) + re-apply
 *                        the shown frame's index-keyed overlays. No busy.
 *   - OVERLAY REFRESH : overlay-only flags on the same drawn atom set (showIndex, selection halo
 *                        while NOT isolating, showCell, showAxis, a cell edit) -> re-apply the
 *                        current frame's overlays; the coordinate movie is NOT rebuilt. No busy.
 *   - APPEND          : streamed new frames (§6.2) -> validate + extend the movie. No busy.
 *   - STRUCTURAL REGEN: the drawn atom set changed (isolate; selection while isolating; force
 *                        overlay/scale -- the per-frame arrows are baked into the movie), or a
 *                        full new load -> re-process all frames + reload the movie. Raises BUSY.
 */
(function (root) {
    "use strict";

    function _noop() {}

    function create(handle, opts) {
        opts = opts || {};
        var eng = (root.molbuilder && root.molbuilder.molview && root.molbuilder.molview.engine) || {};
        var processFrame = (opts.process || eng.process || {}).processFrame;
        var embedIo = opts.embedIo || (eng.embedIo && eng.embedIo.create(handle));
        var store = opts.store || null;
        if (typeof processFrame !== "function") throw new Error("engine.create: process.processFrame missing");
        if (!embedIo) throw new Error("engine.create: embedIo missing (no handle?)");

        // ---- state ------------------------------------------------------------------------ //
        var _data = null;            // clean StructureData (§7.1) -- the source of truth we own.
        var _epoch = 0;              // bumped on every data change -> forces a structural regen.
        var _frame = 0;              // current frame (the frame channel, §7.2) -- NOT a view flag.
        var _flags = _blankFlags();
        var _prevStructSig = null;   // last structural signature (the §8 tier decision).
        var _prevArrowSig = null;    // last force-overlay signature (in-place arrow re-bake tier).
        var _storeUnsub = _noop;
        var _playTimer = null;
        var _frameListeners = [];    // the frame-bar channel (NOT the view store; §7.2).
        var _regenRaf = null;        // pending structural-regen paint yield (coalesces bursts).

        function _blankFlags() {
            return { selection: [], isolate: false, showIndex: false, showForces: false,
                     showCell: false, showAxis: false, forceScale: undefined };
        }
        // Snapshot the view flags from the store (§7.2). The store is the single source of the
        // low-frequency view state; currentFrame is separate (the frame channel).
        function _readFlags() {
            if (!store || typeof store.getState !== "function") return _flags;
            var s = store.getState() || {};
            return {
                selection:  Array.isArray(s.indices) ? s.indices.slice() : [],
                isolate:    !!s.isolate,
                showIndex:  !!s.showIndex,
                showForces: !!s.showForces,
                showCell:   !!s.showCell,
                showAxis:   !!s.showAxis,
                forceScale: (typeof s.forceScale === "number") ? s.forceScale : undefined,
            };
        }

        // ---- derive the per-frame render data (§2) ---------------------------------------- //
        function _identity() {
            return { elements: _data.elements, annotations: _data.annotations };
        }
        function _frameInput(f) {
            return { coords: _data.frames[f],
                     forces: _data.forcesPerFrame ? _data.forcesPerFrame[f] : null };
        }
        function _processAll() {
            return _data.frames.map(function (_, f) {
                return processFrame(_frameInput(f), _identity(), _flags);
            });
        }
        // The scene-level cell box (§7.3): drawn only when showCell is on; wraps the atoms.
        function _cellBox() {
            if (!_flags.showCell || !_data.cell) return null;
            return { lattice: _data.cell.lattice, origin: _data.cell.origin };
        }
        function _sceneCell() {
            return _data.cell ? _data.cell.lattice : null;   // explicit a/b/c basis (axes)
        }
        // Re-apply the CURRENT frame's index-keyed overlays (labels + halos) + the scene cell/
        // axis. Arrows are baked into the movie at load (structural), so a swap/overlay-refresh
        // does not re-hand them here.
        function _applyCurrentOverlays() {
            var pf = processFrame(_frameInput(_frame), _identity(), _flags);
            var overlay = {
                labels:  _flags.showIndex ? pf.labels : false,
                halos:   pf.halos,
                cellBox: _cellBox(),
                axes:    !!_flags.showAxis,
            };
            // Multi-frame: arrows are BAKED into the movie (a swap shows them free), so they are
            // NOT re-handed here -- that would double the layer. Single static frame: no movie,
            // so its arrows ARE applied here.
            if (frameCount() <= 1) overlay.arrows = pf.arrows;
            embedIo.applyOverlays(overlay);
        }
        // Arrow refresh (§8): re-bake every frame's arrows in place -- the coordinate movie is
        // untouched (no reparse). Multi-frame only; a static frame's arrows ride _applyCurrentOverlays.
        function _arrowRefresh() {
            var arrowsPerFrame = _data.frames.map(function (_, f) {
                var pf = processFrame(_frameInput(f), _identity(), _flags);
                return pf.arrows || [];
            });
            embedIo.setFrameArrows(arrowsPerFrame);
            _prevArrowSig = _arrowSig();
        }

        // ---- the §8 tiers ---------------------------------------------------------------- //
        // STRUCTURAL signature: what changes the DRAWN ATOM SET (=> reload the coordinate movie).
        // selection only counts while isolating. Force overlay/scale is NOT here -- the arrows
        // are baked per frame but can be re-baked IN PLACE (see _arrowRefresh), so a force change
        // is an overlay refresh, not a coord reload (§8).
        function _structSig() {
            var isoOn = _flags.isolate && _flags.selection.length > 0;
            return _epoch
                + "|iso:" + (isoOn ? "1" : "0")
                + "|sel:" + (isoOn ? _flags.selection.join(",") : "");
        }
        // Force-overlay signature: the baked per-frame arrows depend on this. A change re-bakes
        // the arrows in place (no coord reload) -- the §8 arrow flavour of an overlay refresh.
        function _arrowSig() {
            return "force:" + (_flags.showForces ? "1" : "0")
                 + "|scale:" + (_flags.showForces ? (_flags.forceScale === undefined ? "d" : _flags.forceScale) : "");
        }
        // Yield a paint so the busy scrim actually shows BEFORE the blocking movie rebuild
        // freezes the thread. loadFrames (setStructure + addModelsAsFrames) is synchronous and
        // blocks; without a yield, setBusy(on)->work->setBusy(off) run in one turn and the
        // browser never paints the scrim. A double rAF guarantees a paint between. On a burst of
        // changes the pending regen is cancelled so only the latest state draws. In node (no
        // requestAnimationFrame) it runs synchronously so the pure logic stays testable.
        function _yieldPaint(fn) {
            var raf = root.requestAnimationFrame, caf = root.cancelAnimationFrame;
            if (typeof raf !== "function") { fn(); return; }
            if (_regenRaf != null && typeof caf === "function") caf(_regenRaf);
            _regenRaf = raf(function () { _regenRaf = raf(function () { _regenRaf = null; fn(); }); });
        }
        function _structuralRegen() {
            embedIo.setBusy("Updating view…");
            _prevStructSig = _structSig();     // record the request's signatures synchronously.
            _prevArrowSig = _arrowSig();       // a reload re-bakes the current-flags arrows too.
            _yieldPaint(function () {
                var processed = _processAll();
                embedIo.loadFrames({ frames: processed, cell: _sceneCell(), cellBox: _cellBox() });
                // loadFrames resets to frame 0; restore the shown frame if the user was elsewhere.
                if (_frame > 0 && _frame < processed.length) embedIo.swapFrame(_frame);
                _applyCurrentOverlays();
                embedIo.setBusy(null);
            });
        }

        // THE render entry (§5): read the flags, pick the minimal tier by what changed (§8).
        function render() {
            if (!_data) return;                       // nothing loaded yet.
            _flags = _readFlags();
            var sig = _structSig();
            if (sig !== _prevStructSig) { _structuralRegen(); return; }  // drawn set changed -> reload.
            // If a structural regen is already scheduled (paint yield pending), the movie is NOT
            // rebuilt yet -- painting overlays now would key them to the OLD model. The pending
            // regen reads the latest _flags when it fires and applies everything then, so let it.
            if (_regenRaf != null) return;
            // Force overlay/scale changed -> re-bake the per-frame arrows IN PLACE (no coord
            // reload) for a loaded movie. A static frame's arrows ride the overlay refresh below.
            if (_arrowSig() !== _prevArrowSig && frameCount() > 1) { _arrowRefresh(); return; }
            _applyCurrentOverlays();          // overlay refresh (labels/halos/cell/axis; + arrows if static).
            _prevArrowSig = _arrowSig();      // static-frame arrows were handled here; stay in sync.
        }

        // ---- public API (§9) ------------------------------------------------------------- //
        // FULL LOAD (§6.1): replace everything, fix identity from frame 0, reset to frame 0.
        function setData(data) {
            data = data || {};
            var frames = Array.isArray(data.frames) ? data.frames : [];
            if (!frames.length) throw new Error("engine.setData: needs at least one frame");
            _data = {
                frames:         frames,
                elements:       Array.isArray(data.elements) ? data.elements : [],
                annotations:    Array.isArray(data.annotations) ? data.annotations : [],
                cell:           data.cell || null,
                forcesPerFrame: Array.isArray(data.forcesPerFrame) ? data.forcesPerFrame : null,
            };
            _epoch++;
            _frame = 0;
            _flags = _readFlags();
            _structuralRegen();
            _notifyFrame();
        }
        // STREAM APPEND (§6.2): validate same atom count (hard error, never coerce), extend the
        // movie with the processed new frames, DON'T move the shown frame.
        function appendFrames(coordsList, appendOpts) {
            appendOpts = appendOpts || {};
            if (!_data) throw new Error("engine.appendFrames: nothing loaded (no atom identity)");
            var list = Array.isArray(coordsList) ? coordsList : [];
            if (!list.length) return frameCount();
            var n = _data.frames[0].length;
            list.forEach(function (coords, i) {
                if (!Array.isArray(coords) || coords.length !== n) {
                    throw new Error("engine.appendFrames: frame " + i + " has "
                        + (Array.isArray(coords) ? coords.length : "?") + " atoms, expected " + n
                        + " (same-atoms invariant, §6.2)");
                }
            });
            var forces = Array.isArray(appendOpts.forces) ? appendOpts.forces : null;
            var startF = _data.frames.length;
            list.forEach(function (coords, i) {
                _data.frames.push(coords);       // clean source of truth grows first.
                if (_data.forcesPerFrame) _data.forcesPerFrame.push(forces ? forces[i] : null);
            });
            _epoch++;
            _prevStructSig = _structSig();   // epoch bumped, but appended (not reloaded) -> stay in sync.
            _prevArrowSig = _arrowSig();
            // If a structural regen is already scheduled (paint yield pending), the movie is about
            // to be rebuilt from _data.frames -- which now includes these. Appending to the stale
            // (not-yet-rebuilt) movie could even mismatch its atom count (e.g. an isolate regen is
            // pending). So skip the incremental append; the pending regen picks them up.
            if (_regenRaf == null) {
                var processedNew = list.map(function (_, i) {
                    return processFrame(_frameInput(startF + i), _identity(), _flags);
                });
                embedIo.appendFrames({ frames: processedNew });
            }
            _notifyFrame();
            return frameCount();
        }
        // NATIVE SWAP (§3): the frame channel. Set the shown frame + re-apply its overlays. No busy.
        function showFrame(i) {
            if (!_data) return;
            var idx = Math.floor(Number(i));
            if (!(idx >= 0 && idx < frameCount())) return;
            _frame = idx;
            // If a structural regen is pending (paint yield), the movie is not rebuilt yet --
            // don't swap/paint the stale model. Record the frame; the regen restores it (it does
            // swapFrame(_frame) + applies overlays when it fires).
            if (_regenRaf != null) { _notifyFrame(); return; }
            embedIo.swapFrame(idx);
            _applyCurrentOverlays();          // labels/halos follow the shown frame.
            _notifyFrame();
        }
        function play(playOpts) {
            playOpts = playOpts || {};
            if (frameCount() <= 1) return;
            var fps = (typeof playOpts.fps === "number" && playOpts.fps > 0) ? playOpts.fps : 10;
            pause();
            if (typeof root.setInterval !== "function") return;
            _playTimer = root.setInterval(function () {
                var n = frameCount();
                if (n <= 1) { pause(); return; }
                var next = _frame + 1;
                if (next >= n) next = 0;       // loop
                showFrame(next);
            }, 1000 / fps);
        }
        function pause() {
            if (_playTimer != null && typeof root.clearInterval === "function") root.clearInterval(_playTimer);
            _playTimer = null;
        }

        function frameCount() { return _data ? _data.frames.length : 0; }
        function currentFrame() { return _frame; }
        // The frame-bar subscribes HERE (not the view store) so playback never re-renders the panel.
        function onFrameChange(fn) {
            if (typeof fn !== "function") return _noop;
            _frameListeners.push(fn);
            return function () { var i = _frameListeners.indexOf(fn); if (i >= 0) _frameListeners.splice(i, 1); };
        }
        function _notifyFrame() {
            for (var i = 0; i < _frameListeners.length; i++) { try { _frameListeners[i](); } catch (_) {} }
        }

        function dispose() {
            pause();
            if (_regenRaf != null && typeof root.cancelAnimationFrame === "function") {
                root.cancelAnimationFrame(_regenRaf); _regenRaf = null;
            }
            try { embedIo.setBusy(null); } catch (_) {}
            try { _storeUnsub(); } catch (_) {}
            _frameListeners = [];
        }

        // The engine holds ONE subscription to the view store: a flag write -> render() (§5).
        if (store && typeof store.subscribe === "function") {
            _storeUnsub = store.subscribe(function () { render(); });
        }

        return {
            setData:       setData,
            appendFrames:  appendFrames,
            showFrame:     showFrame,
            play:          play,
            pause:         pause,
            render:        render,
            frameCount:    frameCount,
            currentFrame:  currentFrame,
            onFrameChange: onFrameChange,
            dispose:       dispose,
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.engine = root.molbuilder.molview.engine || {};
    root.molbuilder.molview.engine.create = create;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { create: create };
    }
})(typeof window !== "undefined" ? window : globalThis);
