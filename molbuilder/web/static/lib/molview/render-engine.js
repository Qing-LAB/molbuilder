/* MolView — the renderEngine, and the drawing commands beneath it. Two levels,
 * one file, because § 9.7 splits them itself: "a MATHS HALF that works out what
 * to draw with no drawing library anywhere near it, and an I/O HALF that is the
 * only code allowed to issue drawing commands." That split is why the
 * interesting part can be exercised with no browser at all (§ 13.2).
 *
 * Contract: docs/web/molview.md § 9.7, § 9.8, § 10 whole.
 *
 * ── The maths half — § 7 level 5 ──────────────────────────────────────────────
 * Owns:     nothing. It is HANDED the master copy, the selection and the
 *           switches, works out what each frame looks like, and passes the
 *           result down. Chooses how much work a change costs (§ 10.5: frame
 *           swap / overlay refresh / append / rebuild) by WHAT CHANGED, never by
 *           atom count. Holds the rebuild window (§ 10.9), where nothing that
 *           arrives is dropped. Checks its own work (§ 10.10).
 * Called by: the model, and only the model.
 * Surface:  COMMANDS ONLY. "Here is the data", "here is the cell", "add these
 *           frames", "the forces changed", "show this frame", "draw", "throw it
 *           away". Every one an instruction; none a question, "because the
 *           renderEngine is told what to draw and is never consulted about what
 *           the data is."
 * NEVER (§ 7 level 5):
 *   - keep its own copy of the displayed frame;
 *   - answer a question about what the data is;
 *   - run a change notification of its own.
 *
 * ── The I/O half — § 7 level 6 ────────────────────────────────────────────────
 * Owns:     exactly one fact — the multi-frame format the library expects.
 * Called by: the maths half, and nothing else.
 * Surface:  small, decision-free operations. Load frames, swap to a frame,
 *           append frames, apply the overlays, set this frame's arrows, set the
 *           cell geometry, show or hide the "Updating view…" cover, batch a
 *           group of changes so the screen updates once — and produce a picture
 *           of what is currently drawn, since only the bottom can do that
 *           (§ 11.4). Plus the two self-check questions of § 9.8, asked only by
 *           the maths half, only about the drawing itself.
 * NEVER (§ 7 level 6):
 *   - decide how much work a change needs — that is the maths half's call;
 *   - hold state;
 *   - answer anything upward.
 *
 * Data goes down. Nothing comes back up (§ 7.1). The per-frame result carries
 * CONTENT and never styling (§ 6.5) — a `color` or a `radius` on per-frame data
 * is the specific defect this rule exists to catch.
 *
 * Step C wrote the per-frame maths of § 10.3 and § 6.5, below. Step F writes the
 * cost decision, the rebuild window, the self-checks, and the I/O half.
 */
"use strict";

import { toDisplay } from "./_atom.js";


/* ══ The per-frame calculation (§ 10.3, § 6.5) ═══════════════════════════════
 *
 * Values in, values out. No drawing library, no DOM, no store — which is what
 * makes the interesting part checkable with no browser at all (§ 9.7, § 13.2).
 *
 * The result is § 6.5's processed frame, and it is worked out FRESH ON EVERY
 * REDRAW AND NEVER STORED. A stored copy would be a third home for coordinates,
 * and § 6.3 allows exactly two.
 */

/**
 * One frame, after the switches have been applied.
 *
 * Two steps, and § 10.3 says the order matters: keep only what is shown, THEN
 * add the overlays keyed to whatever survived. Overlays computed first would be
 * keyed to atoms that are no longer drawn, and every number they carry would be
 * wrong under isolate.
 *
 * @param {object} input
 *   `elements`   the element per atom, shared by every frame (§ 6.2)
 *   `positions`  this frame's coordinates — `Vec3[]`
 *   `forces`     this frame's forces, or null
 *   `selection`  the selected atoms, as original numbers
 *   `switches`   `{isolate, showIndex, showForces, forceScale}` (§ 6.2)
 * @returns {object} § 6.5's processed frame
 */
export function processFrame(input) {
    const elements  = input.elements  || [];
    const positions = input.positions || [];
    const forces    = input.forces || null;
    const selection = input.selection || [];
    const sw        = input.switches || {};

    /* ── Step 1 — keep only what is shown ──────────────────────────────────
     *
     * Deliberately NOT called filtering: "filter" already means the panel's
     * Filter mode, which is a question asked of the SERVER about which atoms to
     * select (§ 9.5). One word for two mechanisms is how a reader comes to
     * believe the server is consulted on every redraw.
     *
     * Dropping atoms renumbers them, so this step records where each drawn atom
     * came from. Everything downstream depends on that map existing: it is what
     * lets a label still show #47 for an atom that is now third in the list.
     */
    const isolating = !!sw.isolate && selection.length > 0;
    let sourceIndex;
    if (isolating) {
        // Ascending original order, each atom once — the drawn list is a
        // cut-down structure, not a record of the order they were picked in.
        sourceIndex = Array.from(new Set(selection))
            .filter((i) => i >= 0 && i < positions.length)
            .sort((a, b) => a - b);
    } else {
        sourceIndex = positions.map((_, i) => i);
    }

    const drawnPositions = sourceIndex.map((i) => positions[i]);
    const drawnElements  = sourceIndex.map((i) => elements[i]);

    /* ── Step 2 — add the overlays, keyed to what survived ─────────────── */

    // Atom-number labels. The text is the atom's ORIGINAL number, recovered
    // through the map from step 1 and converted to 1-based by the one shared
    // translation (§ 11.5) — never its position in the cut-down list.
    let labels = null;
    if (sw.showIndex) {
        labels = sourceIndex.map((original, drawn) => ({
            position: drawnPositions[drawn],
            text:     String(toDisplay(original)),
        }));
    }

    // The highlight. Content, never styling (§ 6.5): this says WHICH atoms, and
    // what a highlight looks like is a constant owned by the sealed layer.
    //
    // Null under isolate is deliberate and is not the same as "nothing selected"
    // — the drawn set already IS the selection, so highlighting all of it would
    // say nothing.
    let highlight = null;
    if (!isolating && selection.length > 0) {
        const drawn = selection.filter((i) => i >= 0 && i < positions.length);
        highlight = drawn.length ? drawn : null;
    }

    // Force arrows for THIS frame, from THIS frame's forces. Getting that wrong
    // shows converged forces on an unconverged frame (§ 10.3).
    let arrows = null;
    if (sw.showForces && Array.isArray(forces)) {
        const scale = typeof sw.forceScale === "number" ? sw.forceScale : 1;
        arrows = [];
        for (let drawn = 0; drawn < sourceIndex.length; drawn++) {
            const original = sourceIndex[drawn];
            const f = forces[original];
            const p = drawnPositions[drawn];
            if (!f || !p) continue;
            arrows.push({
                start: [p[0], p[1], p[2]],
                end:   [p[0] + f[0] * scale,
                        p[1] + f[1] * scale,
                        p[2] + f[2] * scale],
            });
        }
    }

    return {
        positions:   drawnPositions,
        sourceIndex: sourceIndex,
        elements:    drawnElements,
        labels:      labels,
        selection:   highlight,
        arrows:      arrows,
    };
}

/**
 * Every frame, worked through the same steps.
 *
 * One frame is not a special case (§ 6.1): a single-frame structure is a
 * one-frame movie and takes this same path.
 */
export function processFrames(input) {
    const frames = input.frames || [];
    const forcesPerFrame = input.forcesPerFrame || null;
    return frames.map((positions, f) => processFrame({
        elements:  input.elements,
        positions: positions,
        forces:    forcesPerFrame ? forcesPerFrame[f] : null,
        selection: input.selection,
        switches:  input.switches,
    }));
}


/* ══ Scene-level data (§ 6.5, § 10.3) ════════════════════════════════════════
 *
 * The cell box and the axes are the same for every frame unless the cell itself
 * changes, so they are worked out ONCE and are not in the per-frame data.
 * Recomputing them per frame would be work that produces an identical answer
 * four hundred times.
 */

// At 1.5 Å the Cartesian triad is longer than a typical bond (~1.4 Å) and short
// enough to stay out of the way even in a dense junction.
const CARTESIAN_AXIS_LENGTH = 1.5;

// Red / green / blue maps to the first / second / third axis in either mode.
const AXIS_COLORS = ["#ff5555", "#55cc55", "#5588ff"];

// The label sits 15% past the arrow tip: clear of the arrowhead without flying
// off into space at long cell vectors.
const LABEL_PAST_TIP = 1.15;

function isLattice(cell) {
    if (!Array.isArray(cell) || cell.length !== 3) return false;
    return cell.every((row) =>
        Array.isArray(row) && row.length === 3
        && row.every((v) => typeof v === "number" && Number.isFinite(v)));
}

/**
 * The cell box and the axes, from the cell.
 *
 * GEOMETRY IS UNCONDITIONAL — this does not consult `showCell` or `showAxis`,
 * and that is a rule rather than an oversight (§ 10.3). The box's vectors and
 * its anchor corner are structure data, handed down whenever they change even
 * while the cell is hidden, so the anchor is always current. The visibility
 * switch carries only a boolean.
 *
 * If geometry were gated behind visibility it would only ever arrive while the
 * cell was already shown — so turning the cell ON AFTER A HIDDEN LOAD would draw
 * the box from the world origin instead of the structure's corner. Keeping the
 * two apart is also what makes a cell edit an overlay refresh rather than a
 * rebuild (§ 10.5): the atoms did not move.
 */
export function sceneFor(cell) {
    const lattice = (cell && isLattice(cell.lattice)) ? cell.lattice : null;
    const origin = (cell && Array.isArray(cell.origin) && cell.origin.length === 3)
        ? cell.origin : [0, 0, 0];

    // Cell mode follows the lattice vectors and labels them a/b/c; with no cell
    // the triad is Cartesian, at the world origin, labelled x/y/z.
    const vectors = lattice
        ? lattice.map((v) => [v[0], v[1], v[2]])
        : [[CARTESIAN_AXIS_LENGTH, 0, 0],
           [0, CARTESIAN_AXIS_LENGTH, 0],
           [0, 0, CARTESIAN_AXIS_LENGTH]];
    const names = lattice ? ["a", "b", "c"] : ["x", "y", "z"];
    const base = lattice ? origin : [0, 0, 0];

    const axes = vectors.map((v, i) => ({
        start:    [base[0], base[1], base[2]],
        end:      [base[0] + v[0], base[1] + v[1], base[2] + v[2]],
        color:    AXIS_COLORS[i],
        label:    names[i],
        labelEnd: [base[0] + v[0] * LABEL_PAST_TIP,
                   base[1] + v[1] * LABEL_PAST_TIP,
                   base[2] + v[2] * LABEL_PAST_TIP],
    }));

    return {
        cellBox: lattice ? { lattice: lattice, origin: origin } : null,
        axes:    axes,
    };
}
