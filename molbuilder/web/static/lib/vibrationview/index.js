/* VibrationView — the single ES-module entry, and the whole of what is importable.
 *
 * Contract: docs/web/vibrationview.md § 4, § 8, § 9.1, § 9.2.
 * Owns:     the one name a consumer may write — `mount` — and, behind it, level 1
 *           of § 7: the state, the frame loop, and every draw it causes.
 * Called by: a page's module script, once:
 *
 *     import { mount } from "/static/lib/vibrationview/index.js";
 *     const vib = await mount(hostEl, { amplitude: 0.15, fps: 30 });
 *
 * NEVER (§ 4):
 *   - export anything else. Every other file here is internal, and the
 *     underscore on its name says so; a consumer that imports one has broken the
 *     module, not found a shortcut.
 *   - read or write `window.molbuilder`, in either direction. Nothing this module
 *     needs comes from a global, and nothing it holds is published to one. Both
 *     halves of that rule were broken by the module this replaces, which is why
 *     it could not mount at all.
 *   - name the sealed layer. § 15: no consumer names its file and neither does
 *     the document.
 *
 * The clock lives HERE and not with the maths, on purpose (§ 7): timing is WHEN
 * to draw and the maths is WHAT to draw, and only the second is a pure function.
 * That split is what lets an eigenvector scatter be tested without faking a
 * browser first.
 */
"use strict";

import { scatter, positionsAtFrame, rate, regrid } from "./_maths.js";
import { create as createSurface } from "./_seal.js";
import { exportAnimation as runExport } from "./_export.js";

const root = (typeof window !== "undefined") ? window : globalThis;

/* Every default, in one place. The BOUNDS live with the rate arithmetic, which
 * is a different thing: what a value falls back to when nobody said, versus what
 * this module can actually deliver. Splitting the four across two files — which is
 * what an earlier draft did, because its clamp function also supplied defaults —
 * makes four facts into two places to look. */
const DEFAULTS = { amplitude: 0.15, fps: 30, cycleSec: 1.0, showLabel: true };

function num(v, fallback) {
    return (typeof v === "number" && isFinite(v)) ? v : fallback;
}

/* A frame counts as due once this much of its interval has passed.
 *
 * It is not 1.0, and the reason is the commonest configuration there is. A
 * display repaints on its own grid — 16.67 ms at 60 Hz — and the default 30 fps
 * asks for 33.33 ms, which is exactly two repaints. Comparing strictly makes that
 * a coin flip decided by timestamp jitter: land a hair under and the loop waits a
 * THIRD repaint, so the animation runs at 20 fps instead of 30, and alternates
 * between them as the jitter moves. A frame that is 90% due is due. */
const FRAME_DUE = 0.9;


function warn(msg) {
    try { if (root.console) root.console.warn("[vibrationview] " + msg); } catch (_) {}
}

/* Uniform mount contract (§ 8): a failure is a HANDLE carrying `ok:false` and a
 * working `dispose` — never a rejection and never null — so a caller branches on
 * `ok` and can tear down unconditionally. Teardown must never have to ask whether
 * setup worked. */
function failed(hostEl, message) {
    warn(message);
    try {
        if (hostEl && hostEl.classList) {
            hostEl.classList.add("vibview-window", "vibview-window--failed");
        }
        if (hostEl && "textContent" in hostEl) hostEl.textContent = message;
    } catch (_) {}
    return { ok: false, error: message, dispose: function () {} };
}

function validStructure(s) {
    return !!(s && Array.isArray(s.elements) && Array.isArray(s.positions)
              && s.positions.length > 0
              && s.elements.length === s.positions.length);
}


/* ── Making a viewer (§ 8) ────────────────────────────────────────────────────
 *
 * Asynchronous, and it ALWAYS RESOLVES. The handle it returns is live: the
 * surface is built and every door works from the first call. There is no
 * readiness flag and nothing deferred, because a viewer that is not ready yet is
 * a state a caller can get wrong — so it is not offered.
 */
export async function mount(hostEl, opts) {
    opts = opts || {};
    if (!hostEl) return failed(hostEl, "no element to mount into");

    let surface;
    try {
        surface = await createSurface(hostEl);
    } catch (e) {
        return failed(hostEl, (e && e.message) || "the drawing surface could not be built");
    }

    /* Everything one viewer holds (§ 6.1). One home each, and no second copy
     * anywhere — which is what makes "one place holds each fact" mean something
     * once you ask WHICH viewer's fact you meant. */
    let structure  = null;       // the equilibrium
    let disp       = null;       // the current mode, scattered to global order
    let amplitude  = typeof opts.amplitude === "number" ? opts.amplitude : DEFAULTS.amplitude;
    /* WHAT WAS ASKED FOR, and what that comes to.
     *
     * The request is kept because it is the user's, and the derived triple —
     * the rate the clock runs at, the frames in a cycle, and how long that cycle
     * actually lasts — is worked out from it on every change. Keeping only the
     * derived value would compound: ask for 5 fps and the cycle stretches to fit
     * the frame floor, and asking for 30 again would then stretch from the
     * stretched figure instead of returning to the second you asked for. */
    let reqFps      = num(opts.fps, DEFAULTS.fps);
    let reqCycleSec = num(opts.cycleSec, DEFAULTS.cycleSec);
    let r           = rate(reqFps, reqCycleSec);
    let frame       = 0;
    let playing    = false;
    let labelText  = null;
    let labelOn    = opts.showLabel !== false;
    let modeIndex  = null;       // carried for the export stamp; never interpreted
    let modeNorm   = null;       // ditto (§ 6.2)
    let rafId      = null;
    let lastDrawAt = 0;
    let disposed   = false;
    let exporting  = false;

    function draw() {
        if (disposed || !structure || !disp) return;
        surface.setAtomCoords(positionsAtFrame(
            structure.positions, disp, amplitude, frame, r.framesPerCycle));
    }

    /* The loop. A frame is advanced only when enough wall-clock has passed for
     * the chosen rate — so `fps` means frames per second, not "every repaint".
     *
     * When the browser cannot keep up we draw FEWER frames rather than jumping
     * ahead in the cycle: the phase comes from the frame number (§ 10.1), so a
     * slow machine shows a slightly slow vibration instead of a stuttering one.
     * For a vibration nobody is timing that is the better failure, and it is what
     * makes the on-screen sequence the same sequence an export encodes. */
    function tick(now) {
        if (disposed || !playing) return;
        rafId = root.requestAnimationFrame(tick);
        const t = (typeof now === "number") ? now : 0;
        if (t - lastDrawAt < (1000 / r.fps) * FRAME_DUE) return;
        lastDrawAt = t;
        frame += 1;
        draw();
    }

    function start() {
        if (disposed || playing || !disp || !structure) return;
        playing = true;
        lastDrawAt = 0;
        rafId = root.requestAnimationFrame(tick);
    }

    function stop() {
        if (!playing) return;
        playing = false;
        if (rafId !== null) {
            try { root.cancelAnimationFrame(rafId); } catch (_) {}
            rafId = null;
        }
    }

    /* A rate change must not move the molecule. The frame number is meaningless
     * without the count it is measured against, so when the count changes the
     * number is re-expressed against the new one — same phase, different
     * arithmetic. Without this, nudging a smoothness slider would visibly jump
     * the animation.
     *
     * The new phase is the NEAREST one the new rate can express, which is the
     * best that exists: a coarser grid has no point at the old phase. The error
     * is bounded by half a frame — π/N — and it does not accumulate, because each
     * change re-anchors from the phase rather than from a running offset. */
    function reframe(next) {
        if (next.framesPerCycle !== r.framesPerCycle) {
            frame = regrid(frame, r.framesPerCycle, next.framesPerCycle);
        }
        r = next;
    }

    function applyLabel() {
        surface.setLabel(labelOn ? labelText : null);
    }

    return {
        ok: true,

        /* The SLOW fact (§ 5.1): rare, and it costs a redraw and a refit.
         *
         * It also ENDS whatever mode was running. A mode belongs to the structure
         * it was computed against and means nothing against another one — and a
         * viewer animating structure B with structure A's eigenvector would look
         * entirely plausible while being nonsense. */
        setStructure(s) {
            if (disposed) return false;
            if (!validStructure(s)) {
                warn("setStructure needs { elements, positions } of equal length");
                return false;
            }
            stop();
            /* The caption goes with the mode. It names one — "Mode 7 · 1584.2
             * cm⁻¹" — so leaving it up over a structure that now has no mode is a
             * confident label on nothing, which is worse than no label. */
            disp = null; modeIndex = null; modeNorm = null; labelText = null;
            frame = 0;
            structure = {
                elements:  s.elements.slice(),
                positions: s.positions.map(function (p) { return [p[0], p[1], p[2]]; }),
            };
            surface.setStructure(structure.elements, structure.positions);
            surface.setHeldStill([]);
            applyLabel();
            surface.refit();
            return true;
        },

        /* The FAST fact (§ 5.1): frequent, and it costs neither a redraw nor a
         * refit — which is why browsing mode to mode of one result never disturbs
         * the camera.
         *
         * The play/pause state is NOT touched. Playing is its own fact with its
         * own home (§ 5.2); a new mode arriving is not a reason to overrule what
         * the user asked the pause button for. Paused, you see the new mode at
         * its peak, still. */
        showMode(mode) {
            if (disposed) return false;
            if (!structure) {
                warn("showMode before a structure: nothing to animate against");
                return false;
            }
            if (!mode || !Array.isArray(mode.displacements)) {
                warn("showMode needs { displacements: [[dx,dy,dz], ...] }");
                return false;
            }
            const basis = Array.isArray(mode.basis) ? mode.basis : null;
            let m;
            try {
                m = scatter(mode.displacements, basis, structure.positions.length);
            } catch (e) {
                // REFUSED, not padded (§ 6.3): this is a mode computed against a
                // different molecule, and zero-filling would animate the
                // structure partially, plausibly and wrongly. Nothing is drawn.
                warn((e && e.message) || "the mode does not fit this structure");
                return false;
            }
            disp      = m.displacements;
            modeIndex = (mode.index !== undefined && mode.index !== null) ? mode.index : null;
            modeNorm  = (mode.norm !== undefined && mode.norm !== null) ? String(mode.norm) : null;
            labelText = (mode.label !== undefined && mode.label !== null && mode.label !== "")
                ? String(mode.label) : null;
            frame = 0;
            surface.setHeldStill(m.heldStill);
            applyLabel();
            draw();
            return true;
        },

        play()      { start(); },
        pause()     { stop(); },
        isPlaying() { return playing; },

        /* The live knobs (§ 9.2): plain writes the running loop picks up on its
         * next frame. No rebuild, no re-registration — a slider drag never stops
         * the animation. */
        setAmplitude(a) {
            if (disposed || typeof a !== "number" || !isFinite(a)) return;
            amplitude = a;
            if (!playing) draw();          // paused: show the change now
        },

        /* Nonsense is IGNORED; a real number out of range is BROUGHT IN.
         *
         * The two are different answers on purpose. `setFps("fast")` is a caller
         * bug and resetting to the default would hide it. `setFps(0)` is a slider
         * at its end stop — a value with a clear intention that this module simply
         * cannot honour, so it is honoured as far as it goes. Leaving it raw is
         * what made the clock divide by zero and stop for good. */
        setFps(n) {
            if (disposed || typeof n !== "number" || !isFinite(n)) return;
            reqFps = n;
            reframe(rate(reqFps, reqCycleSec));
            if (!playing) draw();
        },

        setCycleSec(s) {
            if (disposed || typeof s !== "number" || !isFinite(s)) return;
            reqCycleSec = s;
            reframe(rate(reqFps, reqCycleSec));
            if (!playing) draw();
        },

        setLabelVisible(on) {
            if (disposed) return;
            labelOn = !!on;
            applyLabel();
        },

        /* § 12. The context below is built HERE, from this closure, and reaches
         * the exporter as an argument — which is why the handle needs no accessor
         * for the state, and why deleting the one it used to have cost nothing.
         *
         * ONE AT A TIME. Two exports would each resize the same surface and each
         * restore it to what IT believed was the original, and the loser leaves
         * the viewer wrong. */
        exportAnimation(opts) {
            if (disposed) return Promise.reject(new Error("vibrationview: disposed"));
            if (exporting) {
                return Promise.reject(new Error(
                    "vibrationview: an export is already running"));
            }
            exporting = true;
            return runExport(opts, {
                surface:  surface,
                coordsAt: function (n) {
                    return positionsAtFrame(structure.positions, disp, amplitude,
                                            n, r.framesPerCycle);
                },
                meta: {
                    ready:          !!(structure && disp),
                    framesPerCycle: r.framesPerCycle,
                    fps:            r.fps,
                    cycleSec:       r.cycleSec,
                    amplitude:      amplitude,
                    norm:           modeNorm,
                    index:          modeIndex,
                },
                playing: function () { return playing; },
                pause:   stop,
                play:    start,
                redraw:  draw,
            }).finally(function () { exporting = false; });
        },

        /* NO HATCH FOR THE EXPORT, and that is deliberate.
         *
         * An earlier draft hung a `_capture` object off this handle so § 12's
         * encoders could read the state and reach the drawing surface. It carried
         * a comment saying it was not really part of the handle, which was simply
         * untrue — it was a property of the object every host is given, and the
         * test that enumerates the doors had to filter it out to stay green: a
         * check laundering the thing it exists to catch.
         *
         * The export needs no hatch. It runs INSIDE this closure, which already
         * holds the structure, the mode, the amplitude, the rate and the surface;
         * § 12's door hands them over as arguments at call time and none of them
         * escapes. The sealed layer warns that every "just for tests" hatch it
         * ever had became a production read, and this one was on its way.
         */
        dispose() {
            if (disposed) return;
            stop();
            disposed = true;
            structure = null; disp = null;
            try { surface.dispose(); } catch (_) {}
            try { if (hostEl.classList) hostEl.classList.remove("vibview-window"); } catch (_) {}
        },
    };
}
