/* VibrationView — the maths. Level 2 of docs/web/vibrationview.md § 7.
 *
 * Module:    lib/vibrationview/ — INTERNAL. The leading underscore is the mark:
 *            nothing outside this directory may import this file. The module's
 *            one importable name is `index.js` (§ 4), and a guard test enforces
 *            it (tests/test_vibrationview_module_boundary.py).
 * Called by: index.js — once per animation frame;
 *            _export.js — once per frame it encodes.
 *            Both ask the same question and get the same answer, which is what
 *            makes "what is animated is what is exported" arithmetic rather than
 *            a promise (§ 5.3).
 *
 * PURE. No DOM, no drawing library, no clock, no state. Values in, values out — so a test
 * of an eigenvector scatter needs no browser and, in particular, no faked
 * `requestAnimationFrame`. That is the whole reason the clock lives one level up
 * (§ 7): timing is WHEN to draw, this is WHAT to draw, and only the second is a
 * function of its inputs.
 *
 * NEVER (§ 7 level 2): touch the DOM, keep state between calls, read a clock, or
 * name the drawing library — not in code, and not in a comment either. A guard
 * asserts it (tests/test_vibrationview_module_boundary.py), and it caught this
 * file's own header claiming innocence by naming the thing it disclaims.
 */
"use strict";


/* ── The rate band (§ 10.1) ───────────────────────────────────────────────────
 *
 * A rate outside this band is brought into it, not honoured and not refused: a
 * smoothness control that throws at a user for dragging a slider is worse than
 * one that stops where it stops.
 *
 * The floor is on FRAMES PER CYCLE, not on fps, because that is what decides
 * whether motion reads as smooth — 15 frames is a 24° phase step, which is where
 * a large-amplitude mode starts to look stepped rather than moving. The fps
 * bounds keep a viewer from drawing frames no display will show, or so few that
 * even a long cycle stutters.
 */
export const FPS_MIN = 5;
export const FPS_MAX = 120;
export const FRAMES_PER_CYCLE_MIN = 15;
export const FRAMES_PER_CYCLE_MAX = 1200;   // 120 fps × 10 s: bounds an export


function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

function isFiniteNumber(v) {
    return typeof v === "number" && isFinite(v);
}

function isVec3(v) {
    return Array.isArray(v) && v.length === 3
        && isFiniteNumber(Number(v[0]))
        && isFiniteNumber(Number(v[1]))
        && isFiniteNumber(Number(v[2]));
}

function refuse(msg) {
    throw new Error("vibrationview: " + msg);
}


/* ── Scattering a mode onto the structure (§ 6.3) ────────────────────────────
 *
 * A vibrational calculation runs over the atoms that were allowed to move, so a
 * mode carries one row per FREE atom and `basis` says which atom each row is.
 * This spreads those rows over every atom, leaving zeros where nothing moves.
 *
 * REFUSES rather than pads. A basis naming an atom the structure does not have,
 * or a row count that does not match, is a mode computed against a different
 * molecule; filling the gap with zeros yields a molecule that animates —
 * partially, plausibly, and wrongly. There is no safe partial answer here, so
 * there is no answer at all.
 *
 * It returns the HELD-STILL set as well, because that is the same reading of the
 * same list: which atoms the basis names is what decides both where a row goes
 * and which atoms have no row at all. Worked out twice, one rule would live in
 * two places and a change to it would have to find both.
 */
export function scatter(displacements, basis, atomCount) {
    const n = Math.floor(Number(atomCount));
    if (!isFinite(n) || n <= 0) {
        refuse("a mode needs a structure to be scattered onto (atom count "
             + String(atomCount) + ")");
    }
    if (!Array.isArray(displacements)) {
        refuse("a mode's displacements must be an array of [dx,dy,dz] rows");
    }
    for (let k = 0; k < displacements.length; k++) {
        if (!isVec3(displacements[k])) {
            refuse("displacement row " + k + " is not a finite [dx,dy,dz]");
        }
    }

    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = [0, 0, 0];

    // No basis: the rows are already one per atom, in order.
    if (!Array.isArray(basis)) {
        if (displacements.length !== n) {
            refuse("a mode with no basis needs one row per atom: got "
                 + displacements.length + " rows for " + n + " atoms");
        }
        for (let i = 0; i < n; i++) {
            out[i] = [Number(displacements[i][0]),
                      Number(displacements[i][1]),
                      Number(displacements[i][2])];
        }
        return { displacements: out, heldStill: [] };   // no basis -> all move
    }

    if (basis.length !== displacements.length) {
        refuse("a mode needs one row per moving atom: got "
             + displacements.length + " rows for " + basis.length
             + " atoms in the basis");
    }
    if (basis.length > n) {
        refuse("a mode's basis names " + basis.length
             + " moving atoms but the structure has " + n);
    }

    const seen = new Set();
    for (let k = 0; k < basis.length; k++) {
        const gi = Math.floor(Number(basis[k]));
        if (!isFinite(gi) || gi < 0 || gi >= n) {
            refuse("a mode's basis names atom " + String(basis[k])
                 + ", which the structure does not have (" + n + " atoms)");
        }
        if (seen.has(gi)) {
            refuse("a mode's basis names atom " + gi + " twice");
        }
        seen.add(gi);
        out[gi] = [Number(displacements[k][0]),
                   Number(displacements[k][1]),
                   Number(displacements[k][2])];
    }
    const still = [];
    for (let i = 0; i < n; i++) if (!seen.has(i)) still.push(i);
    return { displacements: out, heldStill: still };
}


/* ── Where the atoms are at one phase (§ 10) ─────────────────────────────────
 *
 *     position_i(φ) = equilibrium_i + amplitude · cos(φ) · displacement_i
 *
 * `displacements` is already scattered — one row per atom, global order, zeros
 * where nothing moves — so held-still atoms are not a special case in the loop.
 *
 * Deliberately unvalidated: the shapes were checked once, at `scatter`, and this
 * runs on every frame of every export. A check here would be the same refusal
 * paid for a few hundred times a second.
 */
export function positionsAtPhase(equilibrium, displacements, amplitude, phase) {
    const factor = amplitude * Math.cos(phase);
    const n = equilibrium.length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
        const e = equilibrium[i];
        const d = displacements[i];
        out[i] = [e[0] + factor * d[0],
                  e[1] + factor * d[1],
                  e[2] + factor * d[2]];
    }
    return out;
}


/* ── The rate this module will actually run at (§ 10.1) ──────────────────────
 *
 * ONE call, one answer. A rate drives two things — how many frames a cycle has,
 * and how often the next one is due — and they must agree, so they are worked out
 * together and returned together. Clamping them apart is how the clock ended up
 * dividing by an unclamped zero.
 *
 * The caller passes real numbers; supplying defaults for missing ones is the
 * caller's job, so those live in one place and it is not this one.
 *
 * A cycle is a WHOLE number of frames, always. `fps × cycleSec` need not come out
 * even — 25 fps over 0.3 s is 7.5 — so the count is rounded, and **the returned
 * `cycleSec` is what that rounding produced**, not what was asked for. That is
 * the honest number: it is what the animation does, and it is what an export
 * stamps into its metadata. A requested duration that survived into a caption
 * while the frames said otherwise would be a caption that lies.
 *
 * It is also why the seconds need no band of their own. The frame count is
 * bounded, so a wild request is already contained — a thousand-second cycle at 30
 * fps is 1200 frames and comes back as forty seconds — and bounding the seconds
 * as well would be a second fence around the same field.
 */
export function rate(fps, cycleSec) {
    const f = clamp(fps, FPS_MIN, FPS_MAX);
    const n = clamp(Math.round(f * cycleSec),
                    FRAMES_PER_CYCLE_MIN, FRAMES_PER_CYCLE_MAX);
    return { fps: f, framesPerCycle: n, cycleSec: n / f };
}


/* ── The phase of a frame (§ 10.1) ───────────────────────────────────────────
 *
 * The phase comes from the FRAME NUMBER, not from the wall clock. A frame is a
 * position in the cycle, so the sequence on screen and the sequence in an
 * exported file are the same sequence. When the browser cannot keep up the
 * animation slows a little rather than skipping ahead, which for a vibration
 * nobody is timing is the better failure — and pausing is just "keep the number".
 *
 * The wrap is what makes a cycle closed: frame `n` and frame `n + N` are the same
 * phase exactly, with no drift from accumulated floating-point addition.
 */
export function phaseOfFrame(frame, framesPerCycleN) {
    const N = Math.floor(Number(framesPerCycleN));
    if (!isFinite(N) || N <= 0) return 0;
    const f = Math.floor(Number(frame)) || 0;
    const wrapped = ((f % N) + N) % N;
    return 2 * Math.PI * wrapped / N;
}
