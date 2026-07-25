/* VibrationView -- the concealed normal-mode animation package (vibrationview.md).
 *
 * ONE job: animate a vibrational normal mode.  A SIBLING of MolView -- it never selects,
 * edits; MolView never animates.  The spectra inspector hands it the
 * equilibrium geometry + a mode's displacement vectors and drives playback through the
 * handle; the inspector keeps its OWN control widgets (wired to this API) + its spectrum
 * chart.  VibrationView renders the animated view only -- no control UI of its own (§3).
 *
 *   vibrationview.mount(host, { geometry?, freeAtomIdx?, frozenAtomIdx?, amplitude?, speedHz? }) -> handle
 *   handle = { showMode(mode), play(), pause(), isPlaying(),
 *              setAmplitude(a), setSpeed(hz), getMode(), dispose() }
 *
 *     geometry      : { elements:[...], positions:[[x,y,z]...] } -- the EQUILIBRIUM structure.
 *     mode          : { index, displacements:[[dx,dy,dz]...], geometry?, freeAtomIdx?,
 *                       frozenAtomIdx? } -- a mode of a structure.  geometry / free / frozen
 *                       may travel WITH the mode (a mode is defined against its structure); the
 *                       per-mode fields override the mount defaults.  displacements are free-row
 *                       or global (§2).
 *
 * The equilibrium baseline is (re)drawn only when the geometry (or frozen set) actually
 * changes, so browsing modes of ONE structure never rebuilds it.
 *
 * Phase 2 (task #51, SHIPPED 2026-07): VibrationView OWNS the animation -- the phase
 * clock, play/pause state, amplitude/speed, and the per-tick math
 * pos_i(t) = equilibrium_i + amplitude * cos(2*pi*speedHz*t) * displacement_i.  The embed
 * is a dumb drawing surface: each tick hands the computed coordinates through its generic
 * ``setAtomCoords`` door, and animation EXPORT (gif/webm) works through the external-
 * animator provider (``setAnimationProvider({frameCoords, restCoords, cycleSec})``) -- the
 * embed drives the capture clock, this module owns what every captured frame looks like.
 * The embed's ``kind:"vibration"`` animation was deleted with this move.
 *
 * A native ES module.  Internals are IMPORTED (mode-math -- package-private; the shared
 * stateless xyz writer); the embed borrow (root.molbuilder.viewer) stays a RUNTIME LOOKUP
 * (live module owned elsewhere).  The public door for classic consumers (spectra/core.js,
 * results) is ``window.molbuilder.vibrationview.mount`` -- the ONE unified API.
 */
"use strict";

import { scatterDisplacements } from "./mode-math.js";
import { toText } from "../xyz-io.js";

const root = (typeof window !== "undefined") ? window : globalThis;

    function _validGeom(g) {
        return !!(g && Array.isArray(g.elements) && Array.isArray(g.positions)
                  && g.positions.length > 0);
    }

    // The shared XYZ writer (lib/xyz-io.js) -- the ONE source of truth for the minimal
    // .xyz block; STATELESS, so imported directly (see header).
    function _buildXyz(elements, positions) {
        return toText(elements, positions);
    }

    function _geomSig(geom, frozen) {
        if (!geom) return "none";
        return JSON.stringify({ e: geom.elements, p: geom.positions, f: frozen });
    }

    // Uniform mount contract (matches molview.mount): a failure returns a no-op-disposer
    // handle with ``ok:false`` + a reason -- NEVER a sentinel null -- so a consumer branches
    // on ``handle.ok`` and can call ``handle.dispose()`` unconditionally.  Success handles
    // carry ``ok:true``.
    function _failMount(msg) {
        if (root.console) root.console.warn("[vibrationview.mount] " + msg);
        return { ok: false, error: msg, dispose: function () {} };
    }

    export function mount(host, opts) {
        opts = opts || {};
        var mb = root.molbuilder || {};
        var viewer = mb.viewer;   // shared-embed borrow: RUNTIME lookup (Phase 1, see header)
        if (!host || !viewer || typeof viewer.embed !== "function") {
            return _failMount("missing host or shared viewer embed (load order?)");
        }

        var geom          = _validGeom(opts.geometry) ? opts.geometry : null;
        var freeAtomIdx   = Array.isArray(opts.freeAtomIdx) ? opts.freeAtomIdx : null;
        var frozenAtomIdx = Array.isArray(opts.frozenAtomIdx) ? opts.frozenAtomIdx.map(Number) : [];
        var amplitude     = typeof opts.amplitude === "number" ? opts.amplitude : 0.15;
        var speedHz       = typeof opts.speedHz === "number" ? opts.speedHz : 1.0;

        var ready       = false;
        var pendingMode = null;
        var currentMode = null;
        var handle      = null;
        var _drawnSig   = null;   // signature of the currently-drawn geometry (+ frozen)

        // ---- The animation loop (OWNED here, task #51) ------------------------ //
        // disp is in GLOBAL atom order (scattered); frozen rows are zero, so they
        // never move.  The clock is phase-continuous across pause/play.
        var _disp     = null;    // [[dx,dy,dz]...] for the CURRENT mode
        var _rafId    = null;
        var _playing  = false;
        var _phase0   = 0;       // phase at the last play() (pause keeps it)
        var _t0       = 0;       // wall-clock ms of the last play()

        function _nowMs() {
            return (typeof performance !== "undefined" && performance.now)
                ? performance.now() : Date.now();
        }
        function _coordsAtPhase(phase) {
            var factor = amplitude * Math.cos(phase);
            var eq = geom.positions;
            var out = new Array(eq.length);
            for (var i = 0; i < eq.length; i++) {
                var d = (_disp && i < _disp.length) ? _disp[i] : [0, 0, 0];
                out[i] = [eq[i][0] + factor * d[0],
                          eq[i][1] + factor * d[1],
                          eq[i][2] + factor * d[2]];
            }
            return out;
        }
        function _tick() {
            if (!_playing || !handle || !geom || !_disp) return;
            var phase = _phase0 + 2 * Math.PI * speedHz * ((_nowMs() - _t0) / 1000);
            try { handle.setAtomCoords(_coordsAtPhase(phase)); } catch (_) {}
            _rafId = requestAnimationFrame(_tick);
        }
        function _play() {
            if (_playing || !_disp || !geom) return;
            _t0 = _nowMs();
            _playing = true;
            _rafId = requestAnimationFrame(_tick);
        }
        function _pause() {
            if (!_playing) return;
            _phase0 = _phase0 + 2 * Math.PI * speedHz * ((_nowMs() - _t0) / 1000);
            _playing = false;
            if (_rafId !== null) {
                try { cancelAnimationFrame(_rafId); } catch (_) {}
                _rafId = null;
            }
        }
        // Export provider: the embed drives the capture clock; WE own what each
        // captured frame looks like (and the post-capture rest restore).
        function _provider() {
            return {
                frameCoords: function (frameIdx, totalFrames, durationSec) {
                    var t = durationSec * (frameIdx / Math.max(1, totalFrames));
                    return _coordsAtPhase(2 * Math.PI * speedHz * t);
                },
                restCoords: function () {
                    return geom.positions.map(function (p) { return [p[0], p[1], p[2]]; });
                },
                cycleSec: 1 / Math.max(0.01, speedHz),
            };
        }

        function _greyFrozen() {
            if (!handle || typeof handle.setOverlays !== "function") return;
            // Frozen atoms are drawn greyed so the moving (free) atoms read clearly; they carry
            // a zero displacement (scatter, §2) so they never move.
            if (frozenAtomIdx.length) {
                handle.setOverlays({ atoms: [{ indices: frozenAtomIdx, style: { color: "#555" } }] });
            } else {
                handle.setOverlays(null);
            }
        }

        // Draw the equilibrium baseline -- but ONLY when the geometry/frozen actually changed
        // (browsing modes of one structure keeps the same baseline, so no rebuild/reframe).
        function _ensureBaseline() {
            if (!handle || !geom || typeof handle.setStructure !== "function") return;
            var sig = _geomSig(geom, frozenAtomIdx);
            if (sig === _drawnSig) return;
            _drawnSig = sig;
            handle.setStructure({ xyz: _buildXyz(geom.elements, geom.positions) });
            _greyFrozen();
            if (typeof handle.refit === "function") handle.refit();
        }

        // A mode may carry its own structure (a mode is defined against one) -- adopt those
        // before drawing / animating.
        function _adoptModeInputs(mode) {
            if (_validGeom(mode.geometry)) geom = mode.geometry;
            if (Array.isArray(mode.freeAtomIdx)) freeAtomIdx = mode.freeAtomIdx;
            if (Array.isArray(mode.frozenAtomIdx)) frozenAtomIdx = mode.frozenAtomIdx.map(Number);
        }

        function _applyMode(mode) {
            if (!handle || !geom || typeof handle.setAtomCoords !== "function") return;
            _disp = scatterDisplacements(
                mode.displacements, freeAtomIdx, geom.positions.length);
            // Fresh mode: restart the clock at the equilibrium-adjacent peak
            // (phase 0) and register the export provider for THIS mode.
            _pause();
            _phase0 = 0;
            if (typeof handle.setAnimationProvider === "function") {
                handle.setAnimationProvider(_provider());
            }
            currentMode = (mode.index != null) ? mode.index : null;
            _play();
        }

        handle = viewer.embed(host, {
            style:  { rep: "stick", radiusScale: 1.0 },
            pick:   { mode: "none" },              // display-only: no selection in a vibration view
            axes:   false,
            card:   { title: "Vibrational mode", showInfoLine: false, height: "100%" },
            export: { defaultName: "vibration" },
            onReady: function (h) {
                // Draw any mount-time (or pending-mode-adopted) baseline once the viewer is
                // ready, then flush a mode requested before ready.  (embed returns `handle`
                // synchronously; onReady fires on the next microtask, so `handle` is set.)
                host.__vibrationview_test_handle = h;   // test-only, mirrors molview
                ready = true;
                _ensureBaseline();
                if (pendingMode) { var pm = pendingMode; pendingMode = null; _applyMode(pm); }
            },
            onError: function (err) {
                try {
                    if (root.console) {
                        root.console.warn("[vibrationview]", err && err.code, err && err.message);
                    }
                } catch (_) {}
            },
        });
        if (!handle) return _failMount("viewer.embed returned no handle");

        return {
            ok: true,   // uniform mount contract: a live handle -> ok:true (see _failMount)
            showMode: function (mode) {
                if (!mode || !Array.isArray(mode.displacements)) return;
                _adoptModeInputs(mode);
                if (!geom) return;                     // no structure to animate against
                if (!ready) { pendingMode = mode; return; }   // deferred until the baseline is drawn
                _ensureBaseline();
                _applyMode(mode);
            },
            play:  _play,
            pause: _pause,
            isPlaying: function () { return _playing; },
            // Live knobs: the tick reads these every frame, so a slider drag
            // takes effect on the next rAF -- no re-registration needed.  The
            // speed change re-anchors the clock so the phase stays continuous
            // (no visual jump), matching the pause() bookkeeping.
            setAmplitude: function (a) {
                if (typeof a !== "number" || !isFinite(a)) return;
                amplitude = a;
            },
            setSpeed: function (hz) {
                if (typeof hz !== "number" || !isFinite(hz)) return;
                if (_playing) {
                    _phase0 = _phase0 + 2 * Math.PI * speedHz * ((_nowMs() - _t0) / 1000);
                    _t0 = _nowMs();
                }
                speedHz = hz;
            },
            getMode: function () { return currentMode; },
            dispose: function () {
                _pause();
                _disp = null;
                try {
                    if (handle.setAnimationProvider) handle.setAnimationProvider(null);
                } catch (_) {}
                try { if (handle.dispose) handle.dispose(); } catch (_) {}
            },
        };
    }

// The classic-script door -- the module's ONE public API for not-yet-converted consumers
// (spectra/core.js + the results tab read window.molbuilder.vibrationview.mount at call time).
root.molbuilder = root.molbuilder || {};
root.molbuilder.vibrationview = root.molbuilder.vibrationview || {};
root.molbuilder.vibrationview.mount = mount;
