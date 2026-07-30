/* MolView renderEngine -- the orchestrator (the single render place).
 *
 * NAME: "renderEngine", never bare "engine". Across this project an *engine* is a
 * CALCULATION backend (SIESTA, PySCF, TranSIESTA, geometric -- `--engine siesta`,
 * `docs/engines/`, `validation/siesta.py`). This computes no physics: it turns
 * coordinate frames + view flags into what the drawing seal redraws.
 *
 * Contract: docs/web/molview.md, §8, §9.
 * Module:   molbuilder.molview.renderEngine.create   (lib/molview/render-engine/engine.js)
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
 *                        Only valid when a movie EXISTS; without one (a trajectory whose first
 *                        load carried a single frame) it promotes to a structural regen, and a
 *                        movie that fell behind the data heals the same way.
 *   - STRUCTURAL REGEN: the drawn atom set changed (isolate; selection while isolating; force
 *                        overlay/scale -- the per-frame arrows are baked into the movie), or a
 *                        full new load -> re-process all frames + reload the movie. Raises BUSY.
 *
 * A native ES module (private submodule of the MolView module, frontend-module-architecture.md
 * §4) that ALSO publishes the transitional browser global
 * (``window.molbuilder.molview.renderEngine.create``, §3) so still-classic consumers (mount.js /
 * data-model.js) keep reading it until they convert. process/embedIo are IMPORTED directly from
 * the sibling submodules (below); the globals those leaf modules publish are TEST SEAMS only.
 */
"use strict";

import { embedIo as embedIoMod } from "./embed-io.js";
import { atomIndexModel } from "../_atom.js";   // the per-frame maths below reads it

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
    var _notifyFrameChanged = _noop;  // ONE notifier, injected by the data model (see
                                 // setFrameNotifier). The engine owns the displayed index but
                                 // keeps no subscriber list of its own -- molview.data owns the
                                 // one list, so a frame change walks it once instead of being
                                 // relayed through two identical lists.
    var _regenRaf = null;        // pending structural-regen paint yield (rAF id).
    var _regenTimer = null;      // fallback timer (a backgrounded tab suspends rAF).
    var _locked = false;         // an update is in flight -> the viewer is busy; incoming ops queue.
    var _renderQueued = false;   // a render() arrived while locked -> replay it after unlock.
    var _pendingTx = [];         // consumer-push ops (setForces/showFrame/appendFrames) that
                                 // arrived while locked -> replayed in order after unlock.
                                 // VOIDED by setData (a full load replaces the atom set, so a
                                 // queued op references atoms that no longer exist) + dispose.

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
        // Only the per-frame processor consumes this, and it needs only elements (to colour drawn
        // atoms). Annotations are NOT read here -- region/frozen halos were removed (§8.1); the
        // selection panel reads annotations straight from molview.data. Selection highlighting
        // rides flags.selection, not identity.
        return { elements: _data.elements };
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
    // Re-apply the CURRENT frame's index-keyed overlays (labels + selection highlight) + the
    // scene cell/axis. Arrows are baked into the movie at load (structural), so a swap/overlay-
    // refresh does not re-hand them here.
    function _applyCurrentOverlays() {
        var pf = processFrame(_frameInput(_frame), _identity(), _flags);
        var overlay = {
            labels:      _flags.showIndex ? pf.labels : false,
            selection:   pf.selection,        // §8.1: WHICH atoms to glow (isolate off) -- embed owns HOW
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
    // Does the embed hold a real trajectory movie? The APPEND tier can only EXTEND one, and
    // loadFrames only builds one for >1 frame -- so a trajectory whose first load carried a
    // single frame has a plain static structure instead. PROBED, never remembered: the answer
    // has to come from the thing that renders, not from a flag we set ourselves (a flag would
    // just be a third place for the frame count to be wrong).
    function _movieExists() {
        return typeof embedIo.animationKind === "function"
            && embedIo.animationKind() === "trajectory";
    }
    // The movie's own frame count -- what the viewer can actually SHOW. Falls back to our own
    // count when a stub embedIo has no such read (then the check below is a no-op).
    function _movieFrameCount() {
        return (typeof embedIo.frameCount === "function") ? embedIo.frameCount() : frameCount();
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
    // Consumer-push ops arriving while an update is in flight are STORED as pending
    // transactions and replayed in ARRIVAL order after the unlock -- never silently refused
    // (the busy window is a real 16-200ms gap; a live-poll append or a knob change landing in
    // it must not be lost). Latest-wins per op for setForces/showFrame -- only the last force
    // set / seek matters; appendFrames chunks ACCUMULATE (each is a distinct tail). Validation
    // (frame range, same-atoms invariant) happens at REPLAY, by running the real op.
    function _queueTx(op, args) {
        if (op !== "appendFrames") {
            for (var i = _pendingTx.length - 1; i >= 0; i--) {
                if (_pendingTx[i].op === op) _pendingTx.splice(i, 1);
            }
        }
        _pendingTx.push({ op: op, args: args });
    }
    function _drainTx() {
        var tx = _pendingTx;
        _pendingTx = [];
        for (var i = 0; i < tx.length; i++) {
            try {
                if (tx[i].op === "setForces")         setForces(tx[i].args[0]);
                else if (tx[i].op === "showFrame")    showFrame(tx[i].args[0]);
                else if (tx[i].op === "setCell")      setCell(tx[i].args[0]);
                else if (tx[i].op === "appendFrames") appendFrames(tx[i].args[0], tx[i].args[1]);
            } catch (e) {
                // A replayed op has no synchronous caller left to throw to; report and keep
                // draining -- one bad transaction must not void the rest.
                try { globalThis.console.error("molview engine: queued " + tx[i].op + " replay failed:", e); } catch (_) {}
            }
        }
    }

    // A structural regen LOCKS the viewer for the whole update: the busy scrim blocks the
    // user and view-side API calls queue as transactions until 3Dmol is ready. A NEW load
    // supersedes a pending one (latest data/flags win) -- setData is authoritative.
    function _structuralRegen() {
        _cancelPendingRegen();   // supersede any in-flight regen with this latest one.
        _locked = true;
        embedIo.setBusy("Updating view…");
        _yieldPaint(function () {
            // LATEST FLAGS WIN (§8 supersede): re-read the store HERE, inside the yielded
            // callback -- not at schedule time.  The double-rAF yield opens a real window
            // (a whole 200ms in a backgrounded/headless tab) during which a store write
            // (isolate, selection, a view flag) arrives, render() refuses it (_locked),
            // and a schedule-time flag snapshot would then bake the STALE state into the
            // movie with nothing left to replay the dropped write -- the "isolate right
            // after load draws all atoms" race.  The signatures are captured from the
            // same fresh read so the next render() diffs against what was truly drawn.
            _flags = _readFlags();
            _prevStructSig = _structSig();
            _prevArrowSig = _arrowSig();
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
            // Replay everything that arrived while this regen was in flight: first the queued
            // consumer transactions (in arrival order -- their side effects on _data must land
            // faithfully), then a render that was refused (belt to the fresh-read braces above).
            // Outside the finally so a THROWN regen doesn't immediately re-enter; sig diffing
            // makes a no-op render replay cheap (one overlay refresh at worst).
            _drainTx();
            if (_renderQueued) { _renderQueued = false; render(); }
        });
    }

    // THE render entry (§5): read the flags, pick the minimal tier by what changed (§8).
    function render() {
        if (!_data) return;                       // nothing loaded.
        if (_locked) { _renderQueued = true; return; }   // in flight -> replay after unlock.
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
        // VOID the pending transactions: a full load replaces the atom set, so a queued
        // showFrame/setForces/appendFrames references atoms (or a movie) that no longer exist.
        _pendingTx = [];
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
    // PERIODICITY tier: the cell GEOMETRY changed without an atom change (a
    // Cell-page edit adopting the door's blob, or a heal arriving with a
    // load).  Update the engine's snapshot and hand the embed the new box —
    // atoms, movie, selection, and styles untouched (setStructure would
    // clear the animation; the wrong tool for a box move).  Rides the ONE
    // change channel: canvas change → data-model's single subscription →
    // this op (no consumer ever pushes geometry itself).
    function setCell(geom) {
        var next = (geom && geom.lattice)
            ? { lattice: geom.lattice, origin: geom.origin || null }
            : null;
        if (!_data) return;                       // nothing loaded yet
        if (_locked) { _queueTx("setCell", [next]); return; }
        _data.cell = next;
        embedIo.setCellGeometry(_cellGeom());
    }

    // STREAM APPEND (§6.2): validate same atom count (hard error, never coerce), extend the
    // movie with the processed new frames, DON'T move the shown frame.
    function appendFrames(coordsList, appendOpts) {
        appendOpts = appendOpts || {};
        if (!_data) throw new Error("engine.appendFrames: nothing loaded (no atom identity)");
        if (_locked) {
            // An update is in flight -> queue the tail as a transaction (replayed after the
            // unlock; chunks accumulate). A live-poll tick landing in the busy window must not
            // lose its frames. The returned count is the pre-replay count -- the caller sees
            // the appended frames via frameCount() once the queue drains.
            _queueTx("appendFrames", [coordsList, appendOpts]);
            return frameCount();
        }
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
            if (!_movieExists()) {
                // There is no movie to extend. The embed's appendFrames AND setAnimationFrame
                // are both documented no-ops without a trajectory animation, so an incremental
                // append here would grow _data.frames -- and frameCount, and the frame bar --
                // while the screen kept showing frame 0 forever. That is bug #35: a correct
                // frame count over a frozen structure, invisible because the only witness
                // anyone asked was the counter we had just incremented ourselves.
                // Promote to a STRUCTURAL REGEN, which rebuilds from _data.frames (now the
                // whole series) so setAnimation runs and a real movie exists.
                _structuralRegen();
            } else {
                var processedNew = list.map(function (_, i) {
                    return processFrame(_frameInput(startF + i), _identity(), _flags);
                });
                embedIo.appendFrames({ frames: processedNew });
                // The movie must now hold exactly what we hold. It is the thing that can SHOW a
                // frame, so a movie that fell behind means the frame bar is offering frames the
                // viewer cannot render. Rebuild rather than leave the two disagreeing -- the
                // divergence is the defect, not the append.
                if (_movieFrameCount() !== frameCount()) _structuralRegen();
            }
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
        if (!_data) return;
        if (_locked) { _queueTx("setForces", [forcesPerFrame]); return; }   // latest-wins replay.
        _data.forcesPerFrame = Array.isArray(forcesPerFrame) ? forcesPerFrame : null;
        _flags = _readFlags();
        if (frameCount() > 1) _arrowRefresh();
        else _applyCurrentOverlays();
    }
    // NATIVE SWAP (§3): the frame channel. Set the shown frame + re-apply its overlays. No busy.
    function showFrame(i) {
        if (!_data) return;
        if (_locked) { _queueTx("showFrame", [i]); return; }   // latest seek wins; range-checked
        var idx = Math.floor(Number(i));                       // at replay against the new movie.
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
    // THE frame-coordinate owner: every atom, in original 0-based order, before any selection
    // or isolate filtering -- the name says so because that is the whole reason this exists.
    // Measurement takes the panel selection (original indices) and reads those atoms here at the
    // current frame; export/save serialise the shown frame from here. No drawn->original
    // translation, no in-window picking (§7.3). Returns a defensive copy; null if out of range.
    //
    // Not readable from the 3Dmol movie: the movie is a RENDER of this data, and under isolate a
    // render of only the drawn atoms, so reading it back would give a filtered, renumbered array.
    function getFrameAllAtoms(i) {
        var idx = (i === undefined) ? _frame : Math.floor(Number(i));
        if (!_data || !(idx >= 0 && idx < _data.frames.length)) return null;
        return _data.frames[idx].map(function (p) { return [p[0], p[1], p[2]]; });
    }
    // The displayed index changed. Consumers do NOT subscribe here -- they subscribe to
    // molview.data.onFrameChange, which owns the single listener list; the data model injects
    // its notifier below at attach time. The channel stays separate from the view store because
    // playback moves the index ~10x/s and firing the store would re-render the selection panel
    // and steal focus from a filter input mid-play (streamline doc §7.2).
    function setFrameNotifier(fn) { _notifyFrameChanged = (typeof fn === "function") ? fn : _noop; }
    function _notifyFrame() {
        try { _notifyFrameChanged(); } catch (_) {}
    }

    function dispose() {
        _locked = false;
        _renderQueued = false;
        _pendingTx = [];
        _cancelPendingRegen();
        try { embedIo.setBusy(null); } catch (_) {}
        try { _storeUnsub(); } catch (_) {}
        _notifyFrameChanged = _noop;
    }

    // The engine holds ONE subscription to the view store: a flag write -> render() (§5).
    if (store && typeof store.subscribe === "function") {
        _storeUnsub = store.subscribe(function () { render(); });
    }

    return {
        setData:       setData,
        setCell:       setCell,
        appendFrames:  appendFrames,
        setForces:     setForces,
        showFrame:     showFrame,
        render:        render,
        frameCount:    frameCount,
        currentFrame:  currentFrame,
        getFrameAllAtoms:      getFrameAllAtoms,
        setFrameNotifier: setFrameNotifier,
        dispose:       dispose,
    };
}

export { create };

// ── Transitional global (removed once every consumer imports this module) ──
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.renderEngine = window.molbuilder.molview.renderEngine || {};
    window.molbuilder.molview.renderEngine.create = create;
}

/* ── The per-frame maths — the § 9.7 maths half, was process.js ──────────────── */

// ---- Overlay tokens. Named constants, not inline literals. -------------------------- //

// Neutral default force scale (Å per force unit): identity, so raw forces draw at magnitude.
// The consumer overrides via flags.forceScale for a physically-meaningful length.
var DEFAULT_FORCE_SCALE = 1.0;
// Force-vector styling (§2.4): the largest drawn force is highlighted gold; the rest ramp from
// dim-red to orange-red by RELATIVE magnitude, and the arrow radius grows with it -- so the
// eye lands on the atom under the most force (the relaxation "hot spot"). A consumer suppresses
// a force (frozen atom, sub-threshold) by handing a ZERO vector: magnitudes at/under FORCE_EPS
// draw no arrow, so the consumer owns WHICH forces show; the engine owns HOW they look.
var FORCE_MAX_COLOR    = "#ffc400";           // gold: the single largest drawn force
var FORCE_RADIUS_MIN   = 0.05;                // Å: the thinnest (zero-relative-magnitude) arrow
var FORCE_RADIUS_SPAN  = 0.04;                // Å: added at the largest force
var FORCE_EPS          = 1e-9;                // |f| at/under this -> no arrow (a suppressed force)
function _forceRampColor(t) {                 // t in [0,1]: dim-red -> orange-red
    return "rgb(" + Math.floor(170 + 85 * t) + "," + Math.floor(40 + 60 * t) + ",32)";
}

// §2.3 selection filter: the DRAWN atom set, in original-atom order (ascending). Isolate ON
// with a non-empty selection keeps only the selected atoms; otherwise every atom is drawn.
// (Selection alone -- isolate off -- draws all atoms; it only highlights, §2.3.)
function _drawnAtoms(nAtoms, flags) {
    var sel = Array.isArray(flags.selection) ? flags.selection : [];
    var isolate = !!flags.isolate && sel.length > 0;
    var drawn = [];
    if (isolate) {
        var selSet = {};
        for (var k = 0; k < sel.length; k++) selSet[sel[k]] = true;
        for (var a = 0; a < nAtoms; a++) if (selSet[a]) drawn.push(a);
    } else {
        for (var b = 0; b < nAtoms; b++) drawn.push(b);
    }
    return drawn;
}

// §2.4 selection highlight: the DRAWN indices of the selected atoms -- but ONLY when NOT
// isolating. Isolate makes the selection the entire drawn set, so an in-view highlight would add
// nothing (the selection IS all that's shown). Isolate off -> drawn = all atoms in original order,
// so a selected ORIGINAL index equals its drawn index. Returns the drawn indices (or null); the
// embed draws a translucent glow on them (the glow style is the embed's, not the engine's, §8.1).
function _selectionHighlight(drawn, flags) {
    if (flags.isolate) return null;                                   // no highlight under isolate
    var sel = Array.isArray(flags.selection) ? flags.selection : [];
    if (!sel.length) return null;
    var selSet = {};
    sel.forEach(function (i) { selSet[i] = true; });
    var out = [];
    for (var m = 0; m < drawn.length; m++) if (selSet[drawn[m]]) out.push(m);
    return out.length ? out : null;
}

// THE per-frame processor (§2). Returns a ProcessedFrame (§7.3).
function processFrame(frame, identity, flags) {
    frame = frame || {};
    identity = identity || {};
    flags = flags || {};
    var coords = Array.isArray(frame.coords) ? frame.coords : [];
    var elements = Array.isArray(identity.elements) ? identity.elements : [];
    var nAtoms = coords.length;

    // §2.3 -- which atoms are drawn, + the drawn->original index map.
    var drawn = _drawnAtoms(nAtoms, flags);
    var positions = drawn.map(function (a) { var p = coords[a] || [0, 0, 0]; return [p[0], p[1], p[2]]; });
    var sourceIndex = drawn.slice();                       // sourceIndex[m] = original atom index
    var outElements = drawn.map(function (a) { return elements[a] || "X"; });

    // §2.4 -- index labels: explicit TEXT (the ORIGINAL index) at the drawn atom's position,
    // so an isolate-filtered / re-indexed model still shows the true atom index (the embed's
    // format:"index" would show the drawn index -- wrong under isolate). The displayed number
    // is 1-based (SIESTA/Fortran convention, data-vocabulary.md §3.1 / molview-module §16):
    // sourceIndex stays 0-based internal, but the label text goes through the L1 helper
    // `atomIndexModel.toDisplay` -- REUSED, never re-derived (a bare `a+1` would drift).
    // IMPORTED from the pure L1 index leaf (§16) -- always resolved, no load-order guard needed.
    var labels = null;
    if (flags.showIndex) {
        var toDisplay = atomIndexModel.toDisplay;
        labels = drawn.map(function (a, m) {
            return { position: positions[m], text: String(toDisplay(a)) };   // 1-based, SIESTA
        });
    }

    // §2.4 -- selection highlight: the drawn indices to glow (isolate off), or null.
    var selection = _selectionHighlight(drawn, flags);

    // §2.4 -- force vectors for THIS frame, built from the raw per-atom forces × scale, for
    // the drawn atoms only (isolate-aware). null when there are no forces or the overlay is off.
    var arrows = null;
    if (flags.showForces && Array.isArray(frame.forces)) {
        var scale = (typeof flags.forceScale === "number") ? flags.forceScale : DEFAULT_FORCE_SCALE;
        // The KEPT (non-suppressed) set: a zero force -- the consumer's way to hide a frozen or
        // sub-threshold atom -- draws no arrow. maxMag over this set drives the gold highlight +
        // the color/radius ramp, so "biggest force" is relative to what is actually shown.
        var kept = [];
        var maxMag = 0, maxKi = -1;
        for (var ai = 0; ai < drawn.length; ai++) {
            var v = frame.forces[drawn[ai]] || [0, 0, 0];
            var mag = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (mag <= FORCE_EPS) continue;
            if (mag > maxMag) { maxMag = mag; maxKi = kept.length; }
            kept.push({ p: positions[ai], v: v, mag: mag });
        }
        arrows = kept.map(function (e, ki) {
            var t = maxMag > 0 ? e.mag / maxMag : 0;
            return {
                start:  [e.p[0], e.p[1], e.p[2]],
                end:    [e.p[0] + e.v[0] * scale, e.p[1] + e.v[1] * scale, e.p[2] + e.v[2] * scale],
                color:  ki === maxKi ? FORCE_MAX_COLOR : _forceRampColor(t),
                radius: FORCE_RADIUS_MIN + FORCE_RADIUS_SPAN * t,
            };
        });
        if (!arrows.length) arrows = null;   // nothing above the suppression floor -> no overlay
    }

    return {
        positions:   positions,
        sourceIndex: sourceIndex,
        elements:    outElements,
        labels:      labels,
        selection:   selection,   // drawn indices to glow (isolate off), or null
        arrows:      arrows,
    };
}

export const process = { processFrame: processFrame };

// TEST SEAM: tests/test_render_engine_process_js.py reads globalThis.molbuilder.molview.renderEngine.process.
// engine.js imports the export above; production reads no global.  Window-guarded.
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.renderEngine = window.molbuilder.molview.renderEngine || {};
    window.molbuilder.molview.renderEngine.process = process;
}

// The orchestrator above refers to the maths under its former import name.
const processMod = process;
