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
 *     needs comes from a global, and nothing it holds is published to one.
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
 * this module can actually deliver. Splitting the four across two files would
 * make four facts into two places to look. */
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
 * readiness flag and nothing waits for one, because a viewer that is not ready
 * yet is a state a caller can get wrong — so it is not offered.
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
    let amplitude  = num(opts.amplitude, DEFAULTS.amplitude);
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
    let labelOn    = (opts.showLabel === undefined) ? DEFAULTS.showLabel
                                                    : opts.showLabel !== false;
    let modeIndex  = null;       // carried for the export stamp; never interpreted
    let modeNorm   = null;       // ditto (§ 6.2)
    let rafId      = null;
    let lastDrawAt = 0;
    let disposed   = false;
    let exporting  = false;
    /* WHILE AN EXPORT RUNS THE VIEWER HOLDS WHAT IT WAS SET TO (§ 12): a door
     * that would change the picture -- a mode, an amplitude, the rate or the
     * cycle length, the caption, play or pause, a new structure -- takes what
     * it was handed at the call and waits here, in the order asked, to be
     * applied when the export ends, so every frame, its caption and the
     * stamped metadata are the export's.  `exportStop` is the export's own
     * way to stop: a new structure ends the mode it is of (§ 5.1), and a
     * dispose leaves it nothing to draw -- either cancels it. */
    let waiting    = [];
    let exportStop = null;
    let structureAhead = null;   // a structure waiting to be put in, if any

    // Runs now, or -- while an export runs -- when it ends.
    function afterExport(act) {
        if (exporting && !disposed) waiting.push(act);
        else act();
    }

    function waitsForExport(door) {
        return function () {
            const args = arguments;
            afterExport(function () { door.apply(null, args); });
        };
    }

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

    /* The mode that was running ends: its clock, its numbers, its caption and
     * its held-still marks go together -- a caption left naming a mode no
     * longer shown is a confident label on nothing. */
    function endMode() {
        stop();
        disp = null; modeIndex = null; modeNorm = null; labelText = null;
        frame = 0;
        surface.setHeldStill([]);
        applyLabel();
    }

    // A copy: what the caller does with its own afterwards changes nothing here.
    function copyStructure(s) {
        return {
            elements:  s.elements.slice(),
            positions: s.positions.map(function (p) { return [p[0], p[1], p[2]]; }),
        };
    }

    function putStructure(copy) {
        endMode();
        structure = copy;
        surface.setStructure(structure.elements, structure.positions);
        surface.refit();
    }

    /* What a mode asks to show, read whole at the call: whether it fits the
     * structure it would be shown against and, if it does, its rows scattered
     * onto that structure and what it is called.  Pure -- nothing is drawn. */
    function judge(mode, against) {
        if (!mode || !Array.isArray(mode.displacements)) {
            return { why: "showMode needs { displacements: [[dx,dy,dz], ...] }" };
        }
        let m;
        try {
            m = scatter(mode.displacements,
                        Array.isArray(mode.basis) ? mode.basis : null,
                        against.positions.length);
        } catch (e) {
            return { why: (e && e.message) || "the mode does not fit this structure" };
        }
        return {
            displacements: m.displacements,
            heldStill:     m.heldStill,
            index: (mode.index !== undefined && mode.index !== null) ? mode.index : null,
            norm:  (mode.norm !== undefined && mode.norm !== null) ? String(mode.norm) : null,
            label: (mode.label !== undefined && mode.label !== null && mode.label !== "")
                ? String(mode.label) : null,
        };
    }

    function applyMode(shown) {
        disp      = shown.displacements;
        modeIndex = shown.index;
        modeNorm  = shown.norm;
        labelText = shown.label;
        frame = 0;
        surface.setHeldStill(shown.heldStill);
        applyLabel();
        draw();
    }

    /* REFUSED, not padded (§ 6.3): nothing of the mode is drawn, and the
     * mode that was showing ends -- the structure stands at rest. */
    function refuse() {
        endMode();
        surface.setAtomCoords(structure.positions);
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
            const copy = copyStructure(s);
            if (exportStop) exportStop.abort();
            if (exporting) structureAhead = copy;
            afterExport(function () { putStructure(copy); });
            return true;
        },

        /* The FAST fact (§ 5.1): frequent, and it costs neither a redraw nor a
         * refit — which is why browsing mode to mode of one result never disturbs
         * the camera.
         *
         * An accepted mode does not touch the play/pause state. Playing is its
         * own fact with its own home (§ 5.2); a new mode arriving is not a
         * reason to overrule what the user asked the pause button for. Paused,
         * you see the new mode at its peak, still.
         *
         * The answer -- does it fit the structure it will be shown against --
         * and a refusal's warning are given at the call, even when the showing
         * waits for an export. */
        showMode(mode) {
            if (disposed) return false;
            const against = structureAhead || structure;
            if (!against) {
                warn("showMode before a structure: nothing to animate against");
                return false;
            }
            const judged = judge(mode, against);
            if (judged.why) warn(judged.why);
            afterExport(judged.why ? refuse : function () { applyMode(judged); });
            return !judged.why;
        },

        play:  waitsForExport(function () { start(); }),
        pause: waitsForExport(function () { stop(); }),
        isPlaying() { return playing; },

        /* The live knobs (§ 9.2): plain writes the running loop picks up on its
         * next frame -- or, while an export runs, once it ends (§ 12). No
         * rebuild, no re-registration — a slider drag never stops the
         * animation. */
        setAmplitude: waitsForExport(function (a, norm) {
            if (disposed || typeof a !== "number" || !isFinite(a)) return;
            amplitude = a;
            // THE LABEL TRAVELS WITH THE NUMBER (§ 6.1): a caller that changes
            // what the size stands for -- the thermal size taken at another
            // temperature -- restates `norm` with it, so the export names the
            // setting the picture was drawn at.  Omitted, the label stands.
            if (norm !== undefined && norm !== null) modeNorm = String(norm);
            if (!playing) draw();          // paused: show the change now
        }),

        /* Nonsense is IGNORED; a real number out of range is BROUGHT IN.
         *
         * The two are different answers on purpose. `setFps("fast")` is a caller
         * bug and resetting to the default would hide it. `setFps(0)` is a slider
         * at its end stop — a value with a clear intention that this module simply
         * cannot honour, so it is honoured as far as it goes. Left raw, the clock
         * would divide by zero and stop for good. */
        setFps: waitsForExport(function (n) {
            if (disposed || typeof n !== "number" || !isFinite(n)) return;
            reqFps = n;
            reframe(rate(reqFps, reqCycleSec));
            if (!playing) draw();
        }),

        setCycleSec: waitsForExport(function (s) {
            if (disposed || typeof s !== "number" || !isFinite(s)) return;
            reqCycleSec = s;
            reframe(rate(reqFps, reqCycleSec));
            if (!playing) draw();
        }),

        setLabelVisible: waitsForExport(function (on) {
            if (disposed) return;
            labelOn = !!on;
            applyLabel();
        }),

        /* § 12. The context below is built HERE, from this closure, and reaches
         * the exporter as an argument — which is why the handle needs no accessor
         * for the state.
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
            // The caller's signal still cancels it, through the export's own.
            // Every answer is the promise's, a signal that is not one included.
            const stopRun = new AbortController();
            const asked = opts || {};
            try {
                if (asked.signal && asked.signal.aborted) stopRun.abort();
                else if (asked.signal) asked.signal.addEventListener("abort",
                    function () { stopRun.abort(); }, { once: true });
            } catch (e) {
                return Promise.reject(e);
            }
            exporting = true;
            exportStop = stopRun;
            return runExport(Object.assign({}, asked, { signal: stopRun.signal }), {
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
            }).finally(function () {
                exporting = false;
                exportStop = null;
                structureAhead = null;
                const todo = waiting;
                waiting = [];
                todo.forEach(function (act) { act(); });
            });
        },

        dispose() {
            if (disposed) return;
            if (exportStop) exportStop.abort();
            stop();
            disposed = true;
            waiting = [];
            structure = null; disp = null;
            try { surface.dispose(); } catch (_) {}
            try { if (hostEl.classList) hostEl.classList.remove("vibview-window"); } catch (_) {}
        },
    };
}
