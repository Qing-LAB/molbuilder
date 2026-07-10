/* VibrationView -- the concealed normal-mode animation package (vibrationview.md).
 *
 * ONE job: animate a vibrational normal mode.  A SIBLING of MolView -- it never selects,
 * edits, or tiles a k-grid; MolView never animates.  The spectra inspector hands it the
 * equilibrium geometry + a mode's displacement vectors and drives playback through the
 * handle; the inspector keeps its OWN control widgets (wired to this API) + its spectrum
 * chart.  VibrationView renders the animated view only -- no control UI of its own (§3).
 *
 *   vibrationview.mount(host, { geometry, freeAtomIdx, frozenAtomIdx, amplitude, speedHz }) -> handle
 *   handle = { showMode(mode), play(), pause(), isPlaying(),
 *              setAmplitude(a), setSpeed(hz), getMode(), dispose() }
 *     geometry      : { elements:[...], positions:[[x,y,z]...] } -- the EQUILIBRIUM structure.
 *     mode          : { index, displacements:[[dx,dy,dz]...] } -- free-row or global (§2).
 *
 * Phase 1 (vibrationview.md §4): wraps the shared viewer embed and drives its
 * setAnimation({kind:"vibration"}) loop under the hood -- reuse, no reinvented 3Dmol.  The
 * embed's loop computes pos_i(phi) = equilibrium_i + amplitude * cos(phi) * displacement_i;
 * amplitude / speedHz are live partial updates (no structure rebuild).
 */
(function (root) {
    "use strict";

    function _buildXyz(elements, positions) {
        var lines = [String(positions.length), "vibration"];
        for (var i = 0; i < positions.length; i++) {
            var p = positions[i];
            lines.push((elements[i] || "X") + " " + p[0] + " " + p[1] + " " + p[2]);
        }
        return lines.join("\n");
    }

    function mount(host, opts) {
        opts = opts || {};
        var mb = root.molbuilder || {};
        var viewer = mb.viewer;
        var vv = mb.vibrationview || {};
        var geom = opts.geometry;
        if (!host || !viewer || typeof viewer.embed !== "function") return null;
        if (!geom || !Array.isArray(geom.elements) || !Array.isArray(geom.positions)) return null;

        var natoms        = geom.positions.length;
        var freeAtomIdx   = Array.isArray(opts.freeAtomIdx) ? opts.freeAtomIdx : null;
        var frozenAtomIdx = Array.isArray(opts.frozenAtomIdx) ? opts.frozenAtomIdx.map(Number) : [];
        var amplitude     = typeof opts.amplitude === "number" ? opts.amplitude : 0.15;
        var speedHz       = typeof opts.speedHz === "number" ? opts.speedHz : 1.0;

        var ready       = false;
        var pendingMode = null;
        var currentMode = null;
        var handle      = null;

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

        function _drawEquilibrium() {
            if (!handle || typeof handle.setStructure !== "function") return;
            handle.setStructure({ xyz: _buildXyz(geom.elements, geom.positions) });
            _greyFrozen();
            if (typeof handle.refit === "function") handle.refit();
        }

        function _applyMode(mode) {
            if (!handle || typeof handle.setAnimation !== "function") return;
            var disp = (typeof vv.scatterDisplacements === "function")
                ? vv.scatterDisplacements(mode.displacements, freeAtomIdx, natoms)
                : mode.displacements;
            // Hand the mode to the embed's vibration loop; it snaps to the equilibrium baseline
            // and oscillates.  A later mode swap re-sets this with the new displacements.
            handle.setAnimation({
                kind:          "vibration",
                displacements: disp,
                amplitude:     amplitude,
                speedHz:       speedHz,
                paused:        false,
            });
            currentMode = (mode.index != null) ? mode.index : null;
        }

        handle = viewer.embed(host, {
            style:  { rep: "stick", radiusScale: 1.0 },
            pick:   { mode: "none" },              // display-only: no selection in a vibration view
            axes:   false,
            card:   { title: "Vibrational mode", showInfoLine: false, height: "100%" },
            export: { defaultName: "vibration" },
            onReady: function (h) {
                // Draw the equilibrium baseline once the viewer is ready, then flush any mode
                // requested before ready.  (embed returns `handle` synchronously; onReady fires
                // on the next microtask, so `handle` is assigned by now.)
                host.__vibrationview_test_handle = h;   // test-only, mirrors molview
                _drawEquilibrium();
                ready = true;
                if (pendingMode) { _applyMode(pendingMode); pendingMode = null; }
            },
            onError: function (err) {
                try {
                    if (root.console) {
                        root.console.warn("[vibrationview]", err && err.code, err && err.message);
                    }
                } catch (_) {}
            },
        });
        if (!handle) return null;

        return {
            showMode: function (mode) {
                if (!mode || !Array.isArray(mode.displacements)) return;
                if (!ready) { pendingMode = mode; return; }   // deferred until the baseline is drawn
                _applyMode(mode);
            },
            play:  function () { try { if (handle.playAnimation)  handle.playAnimation();  } catch (_) {} },
            pause: function () { try { if (handle.pauseAnimation) handle.pauseAnimation(); } catch (_) {} },
            isPlaying: function () {
                return (typeof handle.isAnimationPlaying === "function")
                    ? !!handle.isAnimationPlaying() : false;
            },
            setAmplitude: function (a) {
                if (typeof a !== "number" || !isFinite(a)) return;
                amplitude = a;
                if (currentMode != null) { try { handle.setAnimation({ amplitude: a }); } catch (_) {} }
            },
            setSpeed: function (hz) {
                if (typeof hz !== "number" || !isFinite(hz)) return;
                speedHz = hz;
                if (currentMode != null) { try { handle.setAnimation({ speedHz: hz }); } catch (_) {} }
            },
            getMode: function () { return currentMode; },
            dispose: function () {
                try { if (handle.pauseAnimation) handle.pauseAnimation(); } catch (_) {}
                try { if (handle.dispose) handle.dispose(); } catch (_) {}
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.vibrationview = root.molbuilder.vibrationview || {};
    root.molbuilder.vibrationview.mount = mount;
})(typeof window !== "undefined" ? window : this);
