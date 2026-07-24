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
 *
 * A native ES module (private submodule of the MolView module, frontend-module-architecture.md
 * §4) that ALSO publishes the transitional browser global
 * (``window.molbuilder.molview.engine.create``, §3) so still-classic consumers (mount.js /
 * data-model.js) keep reading it until they convert. process/embedIo are IMPORTED directly from
 * the sibling submodules (below); the globals those leaf modules publish are TEST SEAMS only.
 */
"use strict";

import { process as processMod } from "./process.js";
import { embedIo as embedIoMod } from "./embed-io.js";

function _noop() {}

function create(handle, opts) {
    opts = opts || {};
    // Deps are IMPORTED from the sibling submodules; `opts.process`/`opts.embedIo` stay as the
    // node-test injection seam (a test can pass a stub without the real 3Dmol embed).
    var processFrame = (opts.process || processMod).processFrame;
    var embedIo = opts.embedIo || embedIoMod.create(handle);
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
    var _frameListeners = [];    // the frame-bar channel (NOT the view store; §7.2).
    var _regenRaf = null;        // pending structural-regen paint yield (rAF id).
    var _regenTimer = null;      // fallback timer (a backgrounded tab suspends rAF).
    var _locked = false;         // an update is in flight -> the viewer is busy; API is refused.

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
    // The cell GEOMETRY {lattice, origin}: a property of the loaded STRUCTURE, read from the one
    // data model. It rides the load (_structuralRegen -> loadFrames) so the embed always holds
    // the real anchor corner. VISIBILITY is a SEPARATE plain boolean (_flags.showCell): the embed
    // draws the box iff showCell is on, using this geometry. Geometry is NEVER smuggled through
    // the visibility toggle (the old bug: geometry gated behind visibility -> box at [0,0,0] when
    // toggled on after a hidden load).
    function _cellGeom() {
        if (!_data || !_data.cell) return null;
        return { lattice: _data.cell.lattice, origin: _data.cell.origin };
    }
    // Re-apply the CURRENT frame's index-keyed overlays (labels + halos) + the scene cell/
    // axis. Arrows are baked into the movie at load (structural), so a swap/overlay-refresh
    // does not re-hand them here.
    function _applyCurrentOverlays() {
        var pf = processFrame(_frameInput(_frame), _identity(), _flags);
        var overlay = {
            labels:      _flags.showIndex ? pf.labels : false,
            halos:       pf.halos,
            cellVisible: !!_flags.showCell,   // plain on/off; the box GEOMETRY rode the load.
            axes:        !!_flags.showAxis,
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
    function _cancelPendingRegen() {
        if (_regenRaf != null && typeof globalThis.cancelAnimationFrame === "function") globalThis.cancelAnimationFrame(_regenRaf);
        if (_regenTimer != null && typeof globalThis.clearTimeout === "function") globalThis.clearTimeout(_regenTimer);
        _regenRaf = null; _regenTimer = null;
    }
    // Yield a paint so the busy scrim shows BEFORE the blocking movie rebuild freezes the
    // thread. loadFrames (setStructure + addModelsAsFrames) is synchronous; without a yield,
    // setBusy(on)->work->setBusy(off) run in one turn and the browser never paints the scrim.
    // A double rAF guarantees a paint between. In node (no requestAnimationFrame) it runs
    // synchronously. A backgrounded tab SUSPENDS rAF (it would never fire -> busy stuck, lock
    // stuck), so a setTimeout fallback runs the regen anyway; whichever fires first wins.
    function _yieldPaint(fn) {
        var raf = globalThis.requestAnimationFrame;
        if (typeof raf !== "function") { fn(); return; }   // node
        var ran = false;
        var go = function () { if (ran) return; ran = true; _cancelPendingRegen(); fn(); };
        _regenRaf = raf(function () { _regenRaf = raf(go); });
        if (typeof globalThis.setTimeout === "function") _regenTimer = globalThis.setTimeout(go, 200);
    }
    // A structural regen LOCKS the viewer for the whole update: the busy scrim blocks the
    // user and view-side API calls are refused until 3Dmol is ready. A NEW load supersedes a
    // pending one (latest data/flags win) -- setData is authoritative, not a click to refuse.
    function _structuralRegen() {
        _cancelPendingRegen();   // supersede any in-flight regen with this latest one.
        _locked = true;
        embedIo.setBusy("Updating view…");
        _prevStructSig = _structSig();
        _prevArrowSig = _arrowSig();
        _yieldPaint(function () {
            embedIo.beginBatch();   // §1/§5: process all frames + overlays, then 3Dmol paints ONCE
            try {
                var processed = _processAll();
                // Geometry (_cellGeom) is handed unconditionally so the anchor corner survives a
                // load with the cell hidden; VISIBILITY rides the _applyCurrentOverlays pass below.
                embedIo.loadFrames({ frames: processed, cellBox: _cellGeom() });
                // loadFrames resets to frame 0; restore the shown frame if the user was elsewhere.
                if (_frame > 0 && _frame < processed.length) embedIo.swapFrame(_frame);
                _applyCurrentOverlays();
            } finally {
                // ALWAYS release: end the batch (the single paint for the whole regen), then
                // clear the lock + scrim -- even if the rebuild threw (a malformed frame, a
                // 3Dmol error). Without this a throw mid-regen wedges the viewer locked + busy
                // forever -- every later render/showFrame/appendFrames refuses until reload
                // (§8 "the lock releases" once 3Dmol is ready).
                embedIo.endBatch();
                embedIo.setBusy(null);
                _locked = false;
            }
        });
    }

    // THE render entry (§5): read the flags, pick the minimal tier by what changed (§8).
    function render() {
        if (!_data || _locked) return;            // nothing loaded, or an update is in flight.
        _flags = _readFlags();
        var sig = _structSig();
        if (sig !== _prevStructSig) { _structuralRegen(); return; }  // drawn set changed -> reload.
        // Force overlay/scale changed -> re-bake the per-frame arrows IN PLACE (no coord
        // reload) for a loaded movie. A static frame's arrows ride the overlay refresh below.
        if (_arrowSig() !== _prevArrowSig && frameCount() > 1) { _arrowRefresh(); return; }
        _applyCurrentOverlays();          // overlay refresh (labels/halos/cell/axis; + arrows if static).
        _prevArrowSig = _arrowSig();      // static-frame arrows were handled here; stay in sync.
    }

    // ---- public API (§9) ------------------------------------------------------------- //
    // FULL LOAD (§6.1): replace everything, fix identity from frame 0, reset to frame 0.
    function setData(data) {
        // setData is authoritative data (a load) -- it SUPERSEDES a pending regen rather than
        // being refused, so a 2-step load (installMolecule then reloadFrames) both land.
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
        if (_locked) return frameCount();         // an update is in flight -> refuse until ready.
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
    // FORCE DATA update (§8): swap the per-frame forces and re-bake the arrow overlay IN PLACE --
    // the coordinate movie is untouched, so a force-filter change (threshold / hide-frozen) is a
    // cheap overlay refresh, not a reload flash. The consumer hands forcesPerFrame in ORIGINAL
    // atom order (one per-atom vector list per frame); a zero vector suppresses that atom's arrow
    // (process.js §2.4), and null clears the whole overlay. Multi-frame re-bakes every frame's
    // arrows; a static frame's arrows ride the overlay refresh.
    function setForces(forcesPerFrame) {
        if (!_data || _locked) return;
        _data.forcesPerFrame = Array.isArray(forcesPerFrame) ? forcesPerFrame : null;
        _flags = _readFlags();
        if (frameCount() > 1) _arrowRefresh();
        else _applyCurrentOverlays();
    }
    // NATIVE SWAP (§3): the frame channel. Set the shown frame + re-apply its overlays. No busy.
    function showFrame(i) {
        if (!_data || _locked) return;            // an update is in flight -> refuse until ready.
        var idx = Math.floor(Number(i));
        if (!(idx >= 0 && idx < frameCount())) return;
        _frame = idx;
        embedIo.beginBatch();             // swap + overlay re-apply -> ONE render (§1/§5)
        try {
            embedIo.swapFrame(idx);
            _applyCurrentOverlays();      // labels/halos follow the shown frame.
        } finally { embedIo.endBatch(); }
        _notifyFrame();
    }
    // Playback (the setInterval loop) lives ONE layer up, in mount.js (`_play`/`_stopPlay`),
    // which drives the frame-controls bar through `data.setFrame`.  The engine only exposes the
    // per-frame door (`showFrame`); it deliberately owns no timer, so there is a single playback
    // owner (§ single-loop) rather than a rival engine-side interval.

    function frameCount() { return _data ? _data.frames.length : 0; }
    function currentFrame() { return _frame; }
    // Read a frame's CLEAN, ORIGINAL-indexed coords (all atoms, never the isolate-filtered
    // draw). This is the accessor the interaction layer uses: measurement takes the panel
    // selection (original 0-based indices) and reads those atoms' coords here at the current
    // frame -- no drawn->original translation, no in-window picking (§7.3 interaction note).
    // Returns a defensive copy; null if out of range.
    function getFrame(i) {
        var idx = (i === undefined) ? _frame : Math.floor(Number(i));
        if (!_data || !(idx >= 0 && idx < _data.frames.length)) return null;
        return _data.frames[idx].map(function (p) { return [p[0], p[1], p[2]]; });
    }
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
        _locked = false;
        _cancelPendingRegen();
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
        setForces:     setForces,
        showFrame:     showFrame,
        render:        render,
        frameCount:    frameCount,
        currentFrame:  currentFrame,
        getFrame:      getFrame,
        onFrameChange: onFrameChange,
        dispose:       dispose,
    };
}

export { create };

// ── Transitional global (removed once every consumer imports this module) ──
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.engine = window.molbuilder.molview.engine || {};
    window.molbuilder.molview.engine.create = create;
}
