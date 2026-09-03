/* MolView — the renderEngine, and the drawing commands beneath it. Two levels,
 * one file, because § 9.7 splits them itself: "a MATHS HALF that works out what
 * to draw with no drawing library anywhere near it, and an I/O HALF that is the
 * only code allowed to issue drawing commands." That split is why the
 * interesting part can be exercised with no browser at all (§ 13.2).
 *
 * Contract: docs/web/molview.md § 9.7, § 9.8, § 10 whole.
 *
 * ── The maths half — § 7 level 5 ──────────────────────────────────────────────
 * Owns:     nothing. It is HANDED the master copy, the selection, the ruler's
 *           track (§ 11.6) and the switches, works out what each frame looks
 *           like, and passes the
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
// The one answer to "which cell is this structure actually using" (§ 9.3). The
// drawing asks the same question the Cell page asks, so the two cannot describe
// different structures — which they did.
import { effectiveCell } from "./model-jobs.js";


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
 *   `measurement` the ruler's `{active, picks}` (§ 11.6) — its own track, not
 *                the selection, and it decides its own visibility: the marks
 *                belong to the ruler, so with the ruler off there are none
 *   `switches`   `{isolate, showIndex, showForces, forceScale}` (§ 6.2)
 * @returns {object} § 6.5's processed frame
 */
export function sourceIndexFor(selection, switches, atomCount) {
    /* WHICH ATOMS ARE DRAWN, and where each came from -- `sourceIndex[m]` is
     * the ORIGINAL number of drawn atom `m` (§ 6.5).
     *
     * Lifted out of `processFrame` so there is ONE definition of it.  It is
     * not per-frame -- it depends only on `isolate` and the selection, which
     * are the same for every frame -- and the 3-D window needs the same map
     * to turn a click back into an atom (§ 11.6).  Recomputing it at that
     * entry would be a second copy of the isolate rule, and the day the two
     * disagreed a click would measure the wrong atom while every frame still
     * drew correctly.
     */
    const picked = selection || [];
    const isolating = !!(switches || {}).isolate && picked.length > 0;
    if (!isolating) {
        return Array.from({ length: atomCount }, (_, i) => i);
    }
    // Ascending original order, each atom once -- the drawn list is a
    // cut-down structure, not a record of the order they were picked in.
    return Array.from(new Set(picked))
        .filter((i) => i >= 0 && i < atomCount)
        .sort((a, b) => a - b);
}


export function processFrame(input) {
    const elements  = input.elements  || [];
    const positions = input.positions || [];
    const forces    = input.forces || null;
    const selection = input.selection || [];
    const track     = input.measurement || {};
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
    const sourceIndex = sourceIndexFor(selection, sw, positions.length);

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

    /* The ruler's marks (§ 11.6).  Same shape as the highlight and a different
     * rule in one respect: they SURVIVE ISOLATE.  The highlight is null there
     * because the drawn set already IS the selection, so marking all of it says
     * nothing — but the measured atoms are a handful WITHIN that set, so which
     * three they are is exactly the thing still worth saying.
     *
     * Renumbered through step 1's map like everything else, so a mark lands on
     * the right atom whether or not the drawn list was cut down. */
    let measured = null;
    if (track.active && Array.isArray(track.picks) && track.picks.length) {
        const seat = new Map(sourceIndex.map((original, drawn) => [original, drawn]));
        const drawn = track.picks
            .map((i) => seat.get(i))
            .filter((d) => d !== undefined);
        measured = drawn.length ? drawn : null;
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
        measured:    measured,
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
        measurement: input.measurement,
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

/* THE TWO TRIADS ARE DIFFERENT COLOURS, because they are different things.
 *
 * The triad in the window is either the world's x/y/z or the cell's a/b/c
 * (§ 10.3), and which one you are looking at changes what every arrow means: the
 * Cartesian triad says which way is which in the room, the lattice triad says
 * which way the box repeats. Drawn in one palette — which they were — the two are
 * indistinguishable, so a skewed cell reads as a mis-drawn axis widget and a
 * structure whose cell failed to load looks exactly like one that never had a
 * cell. The label at each tip says which, but a label is read second and a
 * colour is seen first.
 *
 * x/y/z keeps red/green/blue: that is the near-universal convention for world
 * axes and nothing is gained by moving it. a/b/c takes a set that shares no hue
 * with it — amber, violet, teal — so the two cannot be confused at a glance, and
 * each stays legible against both the card's dark ground and a white background
 * chosen for export (§ 1.1).
 */
const CARTESIAN_AXIS_COLORS = ["#ff5555", "#55cc55", "#5588ff"];   // x / y / z
const LATTICE_AXIS_COLORS   = ["#ffb020", "#c56bff", "#20d0d0"];   // a / b / c

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
export function sceneFor(periodicity) {
    /* IN: the structure's periodicity block, under the server's names (§ 6.2).
     * OUT: a box in the DRAWING's vocabulary, which calls the vectors a lattice
     * and the corner an origin. That rename is this layer's job — the same
     * translation it does for `style` → `rep` (§ 9.8) — and it is the only place
     * in the module where the drawing's words are used. */
    /* THE CELL AS IT WILL ACTUALLY BE USED, not the raw field (§ 9.3). A
     * structure given no explicit cell still HAS one — the server works out the
     * box that wraps the atoms and sends it as `resolved_cell` — and that is the
     * box a calculation runs in, so it is the box to draw.
     *
     * Reading `cell` alone is why "Show unit cell" drew nothing on every plain
     * `.xyz`: the raw field is null there, so the box was null, and the axes
     * fell back to the Cartesian triad at the world origin — which is § 10.3's
     * named failure, reached from a different direction than the one it warns
     * about. */
    const used = effectiveCell(periodicity);
    const lattice = isLattice(used.cell) ? used.cell : null;
    const origin = (Array.isArray(used.cell_origin) && used.cell_origin.length === 3)
        ? used.cell_origin : [0, 0, 0];

    /* TWO TRIADS, AND THEY CO-EXIST.
     *
     * They answer different questions and a user needs both at once: the world
     * triad says WHICH WAY IS WHICH in the room — the frame every coordinate in
     * the file is written in — and the cell triad says WHICH WAY THE BOX
     * REPEATS, from the corner it is anchored at. On a skewed or rotated cell
     * those are not the same directions, and seeing the angle between them IS
     * the thing worth looking at.
     *
     * This used to return ONE of them — a/b/c if the structure had a cell, x/y/z
     * if it did not — so the world frame vanished the moment a cell appeared,
     * and nothing on screen said which of the two you were looking at except a
     * single letter at each tip. They are separate now, they carry separate
     * colours (above), and each rides its own switch: the world triad is what
     * "Show axes" means, and the cell triad belongs to the cell, so it comes and
     * goes with "Show unit cell" alongside the box it describes.
     */
    const axes = triad(
        [[CARTESIAN_AXIS_LENGTH, 0, 0],
         [0, CARTESIAN_AXIS_LENGTH, 0],
         [0, 0, CARTESIAN_AXIS_LENGTH]],
        [0, 0, 0], ["x", "y", "z"], CARTESIAN_AXIS_COLORS);

    const cellAxes = lattice
        ? triad(lattice, origin, ["a", "b", "c"], LATTICE_AXIS_COLORS)
        : null;

    return {
        cellBox:  lattice ? { lattice: lattice, origin: origin } : null,
        axes:     axes,
        cellAxes: cellAxes,
    };
}

/* Three arrows from one corner. The colours and the names arrive together so a
 * triad cannot end up labelled one thing and coloured another. */
function triad(vectors, base, names, colors) {
    return vectors.map((v, i) => ({
        start:    [base[0], base[1], base[2]],
        end:      [base[0] + v[0], base[1] + v[1], base[2] + v[2]],
        color:    colors[i],
        label:    names[i],
        labelEnd: [base[0] + v[0] * LABEL_PAST_TIP,
                   base[1] + v[1] * LABEL_PAST_TIP,
                   base[2] + v[2] * LABEL_PAST_TIP],
    }));
}


/* ══ The cost decision (§ 10.5) ══════════════════════════════════════════════
 *
 * "A render is NOT always a rebuild. Given what changed since the last one, the
 * pipeline does the least work that still produces the correct result — still
 * one place and one decision, not a second path."
 *
 * THE COST IS CHOSEN BY WHAT CHANGED, NEVER BY HOW BIG THE SYSTEM IS. There is
 * no atom-count threshold and no magic number anywhere in this decision, and a
 * change that adds one has changed the design.
 */
export const REBUILD = "rebuild";
export const APPEND  = "append";
export const SWAP    = "swap";
export const OVERLAY = "overlay";

/* The rebuild window (§ 10.9). */
const IDLE      = "idle";
const REBUILDING = "rebuilding";


/**
 * The renderEngine. Commands only (§ 9.7) — every entry is an instruction, none
 * is a question, "because the renderEngine is told what to draw and is never
 * consulted about what the data is".
 *
 * It is HANDED the master copy through a data source and holds nothing of its
 * own (§ 7 level 5). The one thing it keeps is what it last DREW, which is not
 * truth — it is the record of its own last instruction, and it exists so the
 * next one can be sized correctly.
 */
export function createRenderEngine(embed) {
    let source = null;
    let phase = IDLE;
    let held = [];
    let drawnKey = null;        // which atoms were drawn last time
    let costLog = [];           // what each change actually cost, for § 13.2

    /* THE SCENE, WORKED OUT ONCE (§ 6.5, § 10.3). "They are the same for every
     * frame unless the cell itself changes, so they are worked out once as
     * scene-level data and are NOT RECOMPUTED PER FRAME. Recomputing them per
     * frame would be work that produces an identical answer four hundred times."
     *
     * It was derived inside the per-frame overlay path, so a frame swap re-ran
     * it — four hundred identical derivations across a played trajectory, which
     * is exactly the rule's own example of what not to do.
     *
     * Held until the cell changes, and the only two things that can change it
     * say so: a new structure, and a cell edit. */
    let scene = null;
    const sceneNow = () => {
        if (!scene) {
            const s = structure();
            scene = sceneFor(s ? s.periodicity : null);
        }
        return scene;
    };
    const forgetScene = () => { scene = null; };

    const structure = () => (source && source.structure()) || null;
    const frames    = () => (source && source.frames()) || null;
    const forces    = () => (source && source.forces()) || null;
    const frameNow  = () => (source ? source.frame() : 0);
    const masterCount = () => {
        const f = frames();
        return Array.isArray(f) ? f.length : 0;
    };

    function switches() {
        return (source && source.switches) ? source.switches() : {};
    }
    function selection() {
        return (source && source.selection) ? source.selection() : [];
    }
    function measurement() {
        return (source && source.measurement) ? source.measurement() : {};
    }

    /* Which atoms are drawn, as a value that changes exactly when the SET does.
     *
     * Note what is not in it: how many atoms there are. § 10.5's four questions
     * do not include "how many atoms are there?", so neither does this. */
    function currentDrawnKey() {
        const sw = switches();
        return (sw.isolate && selection().length)
            ? "isolate:" + selection().slice().sort((a, b) => a - b).join(",")
            : "all";
    }

    function processAll() {
        return processFrames({
            elements:       structure() ? structure().elements : [],
            frames:         frames() || [],
            forcesPerFrame: forces(),
            selection:      selection(),
            measurement:    measurement(),
            switches:       switches(),
        });
    }

    /* ONE frame, for the two costs that only ever touch one (§ 10.5).
     *
     * A swap and an overlay refresh both need the overlays OF THE SHOWN FRAME
     * and nothing else. Working out all four hundred to use one is the redraw
     * § 10.4 says playing does not do — invisible, because the answer is right;
     * it is the cost that is wrong, and it grows with the trajectory. */
    function processOne(at) {
        const all = frames() || [];
        if (!all[at]) return null;
        const perFrame = forces();
        return processFrame({
            elements:  structure() ? structure().elements : [],
            positions: all[at],
            forces:    perFrame ? perFrame[at] : null,
            selection: selection(),
            measurement: measurement(),
            switches:  switches(),
        });
    }

    /* ── Handing a processed frame down (§ 9.8) ───────────────────────────
     *
     * Level 6's translation: finished data into the drawing's doors. No
     * decisions here — which door to use was decided above.
     */
    function applyOverlaysFor(processed) {
        embed.setOverlays({
            labels:    (processed.labels || []).map((l, drawn) => ({
                atom: drawn, text: l.text,
            })),
            highlight: processed.selection || [],
            measured:  processed.measured || [],
        });
        /* ONE ARROW SET, COMPOSED HERE. The axis triad rides the ordinary arrow
         * door carrying its own colours (§ 10.3), and the drawing has one such
         * door — so writing the forces and then the axes made the second erase
         * the first: with both switches on the force arrows vanished, and a
         * frame swap (which re-places the overlays and not the scene) erased
         * the axes instead. Whatever arrows exist are handed down together. */
        const sw = switches();
        const scene = sceneNow();
        // Each triad on its own switch: the world frame is what "Show axes"
        // means, and the cell's own directions belong to the cell, so they come
        // and go with the box they describe. Both can be up at once.
        const world = sw.showAxis ? scene.axes : [];
        const cell = (sw.showCell && scene.cellAxes) ? scene.cellAxes : [];
        embed.setArrows((processed.arrows || []).concat(world, cell));
    }

    function applyScene() {
        const sw = switches();
        const scene = sceneNow();
        // Geometry travels unconditionally; the switch carries only a boolean
        // (§ 10.3). So the box is handed down whenever it changes and the
        // SWITCH decides whether it is drawn — which is why turning the cell on
        // after a hidden load draws it at the structure's corner and not at the
        // world origin.
        embed.setCell(sw.showCell ? scene.cellBox : null);
    }

    /* ── The four costs ───────────────────────────────────────────────────── */

    /* @param fit  whether to re-fit the camera on the structure.
     *
     * § 9.6: "On load, AND ON RESET, the camera is fitted to the structure" —
     * those two moments, and no other. A rebuild is not one of them: isolate is
     * a rebuild (§ 10.5), so fitting on every rebuild threw away the angle the
     * user had set the moment they pressed the isolate switch. Nothing above the
     * drawing keeps the camera, so there is nothing to restore afterwards —
     * which is exactly why the fit has to be withheld rather than undone.
     *
     * A heal (§ 10.10) does not fit either: it repairs a drawing that came back
     * short, and the user did not ask for anything. */
    function doRebuild(fit) {
        const processed = processAll();
        const s = structure();
        if (!s || !processed.length) {
            /* NOTHING TO DRAW IS SOMETHING TO DRAW (molview.md § 6.7a).
             *
             * This returned without telling the drawing anything, so a model
             * emptied to NOTHING -- `clear()`, which sets the structure to
             * null -- left the previous molecule on screen while the panel's
             * atom list emptied beside it.  Reported 2026-09-02: "when i
             * click 'start empty' the atom list is clear, but 3dmol is not
             * updated ... it stays with the old model displayed".
             *
             * Deleting every atom took a different route and so looked
             * fixed: there the structure still EXISTS with zero elements, so
             * `loadFrames` was reached and cleared the viewer itself.  The
             * guard here is the same mistake one layer up -- "nothing to
             * draw" read as "nothing to do". */
            embed.beginBatch();
            /* THE TRIAD IS HANDED DOWN HERE, not assumed to be sitting in
             * the drawing already.
             *
             * `loadFrames([], [])` redraws the arrows the drawing is HOLDING
             * -- which is right when a structure was on screen a moment ago
             * and wrong on the path that matters most: a page reopened onto
             * an empty canvas has never handed any arrows down, so there was
             * nothing to redraw and the window came back a blank rectangle
             * with "Show axes" reading ON (browser walk, 2026-09-02).
             *
             * Sent BEFORE the clear, so the arrows are in hand when
             * `loadFrames` redraws them and sets the empty view's distance.
             * The overlays go out empty for the same reason the models do:
             * labels and highlights belong to atoms that are gone. */
            embed.setOverlays({ labels: [], highlight: [], measured: [] });
            embed.setArrows(switches().showAxis ? sceneNow().axes : []);
            embed.loadFrames([], []);      // clears models, redraws the triad
            applyScene();
            embed.endBatch();
            drawnKey = currentDrawnKey();
            costLog.push(REBUILD);
            return;
        }
        embed.beginBatch();
        embed.loadFrames(processed[0].elements,
                         processed.map((p) => p.positions));
        drawnKey = currentDrawnKey();
        const at = Math.min(frameNow(), processed.length - 1);
        embed.showFrame(at);
        applyOverlaysFor(processed[at]);
        applyScene();
        if (fit) embed.fitCamera();
        embed.endBatch();
        costLog.push(REBUILD);
        healIfShort();
    }

    function doAppend(from) {
        // § 10.10: "appending to a structure with no movie yet becomes a
        // rebuild." A movie is only built once, so a run caught at its very
        // first geometry has none — and appending to a movie that does not
        // exist quietly does nothing at all. This is the case the "is there a
        // movie?" question exists to catch.
        //
        // It FITS, because this is the first geometry of the run reaching the
        // drawing — the load § 9.6 means, arriving as an append.
        if (!embed.hasMovie()) { doRebuild(true); return; }

        // ONLY THE NEW FRAMES (§ 10.5: "process the new frames only, extend the
        // movie"). Working out all four hundred and keeping the tail made an
        // append cost what a rebuild costs, which is the distinction the whole
        // cost table exists to draw.
        const fresh = [];
        const total = masterCount();
        for (let f = Math.max(0, from); f < total; f++) {
            const one = processOne(f);
            if (one) fresh.push(one);
        }
        if (!fresh.length) return;
        embed.beginBatch();
        embed.appendFrames(fresh.map((p) => p.positions));
        embed.endBatch();
        costLog.push(APPEND);
        // The displayed frame does not move (§ 10.8 rule 5): a user watching
        // frame 12 keeps watching frame 12 while the run grows past it.
        healIfShort();
    }

    function doSwap() {
        const at = frameNow();
        embed.beginBatch();
        embed.showFrame(at);
        const processed = processOne(at);
        if (processed) applyOverlaysFor(processed);
        embed.endBatch();
        costLog.push(SWAP);
    }

    function doOverlay() {
        const at = frameNow();
        const processed = processOne(at);
        embed.beginBatch();
        if (processed) applyOverlaysFor(processed);
        applyScene();
        embed.endBatch();
        costLog.push(OVERLAY);
    }

    /* ── Checking its own work (§ 10.10) ──────────────────────────────────
     *
     * "A check is only worth making against something that could disagree —
     * asking the copy you just grew how big it is confirms nothing, because it
     * agrees with itself by construction. The only informative question is
     * whether the DRAWING ended up with as many frames as the STRUCTURE has."
     *
     * A mismatch is never shown to anybody; it triggers a rebuild.
     */
    // ONE retry, not a loop. A drawing that comes back short again is a drawing
    // that cannot hold what it was given, and rebuilding into it forever would
    // hang the page rather than heal it — a worse failure than the short drawing
    // it was trying to fix. The heal is a correction, not a guarantee.
    let healing = false;
    function healIfShort() {
        if (healing || !masterCount()) return;
        if (embed.drawnFrameCount() >= masterCount()) return;
        healing = true;
        try { doRebuild(false); } finally { healing = false; }
    }

    /* ── The rebuild window (§ 10.9) ──────────────────────────────────────
     *
     * A rebuild takes long enough to be visible, so it shows the cover and locks
     * the viewer. That leaves a window in which other things arrive anyway — a
     * user click, or a timer-driven poll delivering frames that no amount of
     * disabled buttons could stop. NOTHING THAT LANDS IN THAT WINDOW IS SILENTLY
     * DROPPED.
     */
    // Each rebuild carries a generation. A full load that arrives while one is
    // under way SUPERSEDES it (§ 10.9: "a full load is never itself refused: it
    // is the more authoritative statement about what the structure is") — so the
    // older one must not finish. Dropping what it HELD is not enough: its own
    // pass is still pending, and if it lands after the newer one it redraws the
    // structure that was just replaced. That is a movie of the previous load,
    // silently, with a frame bar offering frames it does not have.
    /* THE COVER IS AN OPERATION GUARD, NOT A MESSAGE.
     *
     * What it is for: while the core data is being replaced, the user must not
     * be able to pick an atom or move the camera. The rebuild regenerates
     * everything fed to the drawing library, so a click landing mid-way resolves
     * against atom numbers that are about to mean something else, and a camera
     * drag fights a scene being rebuilt underneath it. The "Updating view…"
     * message is the courtesy; the BLOCK is the job.
     *
     * How it blocks: the element spans the window at `pointer-events: auto`
     * above the canvas, so hit-testing stops at it. Measured on a real page --
     * at the canvas centre, `elementFromPoint` returns the canvas before, the
     * cover's own subtree while it is up, and the canvas again after.
     *
     * WHY THE WORK GETS ITS OWN TURN (PAINT_YIELD). The rebuild is synchronous
     * and freezes the page while it runs. Raising the cover and doing the work
     * in the SAME turn means the browser never gets between them: the cover is
     * never drawn, so a user is blocked with no explanation for it. Measured
     * before this: raised and lowered 8.8 ms apart with no task boundary in
     * between. `setTimeout` ends the turn, which is deliberately NOT
     * `requestAnimationFrame` -- rAF does not fire in a background tab, so a
     * rebuild there would wait for a frame that never comes and hang.
     *
     * WHY THE COVER OUTLIVES THE WORK (MINIMUM_ON_SCREEN). This is the part that
     * matters most and is easiest to miss. A frozen page does not discard the
     * clicks and drags made during the freeze -- it QUEUES them, and delivers
     * them the moment it is free. Lowering the cover as the work ends would let
     * that queue flush straight onto the canvas, against the new data: exactly
     * the operations the guard exists to stop, only later. Holding the cover a
     * beat longer means the queue drains into it instead. The message being
     * legible is a side benefit; catching the backlog is the reason.
     */
    const PAINT_YIELD = () => new Promise((r) => setTimeout(r, 0));
    // A clock that does not care what the wall says -- only elapsed time matters
    // here, and a user changing their timezone mid-load should not extend a cover.
    const now = () => (typeof performance !== "undefined" && performance.now)
        ? performance.now() : Date.now();
    const MINIMUM_ON_SCREEN = 200;   // ms

    let generation = 0;
    async function rebuildGuarded(fit) {
        const mine = ++generation;
        phase = REBUILDING;
        embed.setBusy("Updating view…");
        const shownAt = now();
        try {
            await PAINT_YIELD();              // the cover reaches the screen here
            if (mine !== generation) return;  // superseded: a newer load owns the drawing
            doRebuild(fit);
        } finally {
            if (mine !== generation) return;  // the newer one owns the cover too

            /* THE ENGINE IS IDLE THE MOMENT THE WORK IS DONE -- before the cover
             * comes down, and that ordering matters.
             *
             * These are two different jobs and I had them as one: `phase` holds
             * DATA arrivals (§ 10.9), the cover blocks POINTER input. Leaving
             * the phase REBUILDING until the cover dropped meant a switch
             * toggled in that window was DISCARDED -- § 10.9 says a switch is
             * not held, because "the rebuild reads the switches when it runs",
             * which is only true while a rebuild is actually running. Held over
             * a cosmetic wait it silently drops the user's input instead. */
            phase = IDLE;
            // Replayed IN ARRIVAL ORDER, then the viewer is idle again.
            const queued = held;
            held = [];
            for (const item of queued) item();

            /* The cover outlives the work on purpose: a frozen page delivers the
             * clicks and drags it queued the instant it is free, and they must
             * land on the cover rather than on a canvas whose atoms have just
             * been renumbered. Nothing is gated on this wait -- the engine is
             * already idle and accepting work. */
            const left = MINIMUM_ON_SCREEN - (now() - shownAt);
            if (left > 0) await new Promise((r) => setTimeout(r, left));
            if (mine !== generation) return;  // a newer load owns the cover now
            embed.setBusy(false);
        }
    }

    // A seek and a new set of forces keep only the LAST — only the frame you end
    // on matters, and only the last forces are the current answer. Appended
    // frames ACCUMULATE, because each poll tick's frames are a distinct piece of
    // the run and losing one would leave a hole in the middle of it.
    function hold(kind, fn) {
        if (kind === "seek" || kind === "forces") {
            held = held.filter((h) => h.kind !== kind);
        }
        fn.kind = kind;
        held.push(fn);
    }

    return {
        setDataSource(next) { source = next; },

        /* WHICH ATOM A CLICK IN THE 3-D WINDOW LANDED ON (§ 11.6).
         *
         * The window reports the index of the atom it DREW, and under isolate
         * that is not the atom's real number -- the drawn list is cut down to
         * the selection, so everything is renumbered.  This turns it back,
         * through the same map the labels are placed with (§ 6.5), so a click
         * measures the atom the person clicked rather than whichever atom
         * happens to hold that seat.
         *
         * The engine answers because the engine OWNS the map; asking it here
         * is what keeps the isolate rule in one place.  `null` for an index
         * that is not on screen, so the caller drops it rather than guessing.
         */
        drawnToOriginal(drawn) {
            if (typeof drawn !== "number" || !(drawn >= 0)) return null;
            const s = structure();
            const count = s && s.elements ? s.elements.length : 0;
            if (!count) return null;
            const map = sourceIndexFor(
                source ? source.selection() : [],
                source ? source.switches() : {},
                count);
            return drawn < map.length ? map[drawn] : null;
        },

        // "Here is the data." A whole new structure — the set of drawn atoms
        // changed by definition, so this is a rebuild (§ 10.5 question 1).
        //
        // A full load is NEVER itself refused: everything held is dropped,
        // because it refers to atoms or frames that no longer exist, and the
        // load supersedes the rebuild under way. It is the more authoritative
        // statement about what the structure is (§ 10.9).
        dataChanged() {
            held = [];
            // A new structure: the scene is worked out again (§ 10.3), and this
            // is one of § 9.6's two moments, so the camera is fitted.
            forgetScene();
            return rebuildGuarded(true);
        },

        // A switch changed. The set of drawn atoms changed only if isolate is
        // involved; everything else is an overlay refresh with the same atoms
        // drawn (§ 10.5).
        switchesChanged() {
            if (phase === REBUILDING) return;   // nothing is held: the rebuild
                                                // reads the switches when it RUNS
            // Isolate changed the drawn set — a rebuild, but NOT a load, so the
            // camera stays where the user put it (§ 9.6).
            if (currentDrawnKey() !== drawnKey) return rebuildGuarded(false);
            doOverlay();
        },

        // "Draw it this way." A drawing setting DERIVES NOTHING (§ 10.1): no
        // frame is re-processed, the movie is untouched, and the same frame is
        // simply painted differently. Never a rebuild, whatever the atom count.
        //
        // This is also level 6's translation (§ 9.8) — the store keeps the
        // vocabulary the UI speaks (`style`, `radius`), the seal keeps the one
        // the drawing library speaks (`rep`, `radiusScale`), and neither has to
        // learn the other's. It is the only place the two names meet.
        drawingChanged(settings) {
            const s = settings || {};
            embed.setStyle({
                rep:         s.style,
                radiusScale: s.radius,
                background:  s.background,
            });
            embed.setProjection(s.orthographic ? "orthographic" : "perspective");
        },

        // "Here is the cell." An overlay refresh, NOT a rebuild — the atoms did
        // not move; only the box and the axes changed (§ 10.5).
        cellChanged() {
            // The one thing besides a new structure that changes the scene, so
            // it is the one other place the scene is worked out again (§ 10.3).
            forgetScene();
            if (phase === REBUILDING) { hold("cell", () => doOverlay()); return; }
            doOverlay();
        },

        // "The forces changed." Re-derived for every frame and re-baked in
        // place, without touching the coordinates (§ 10.6).
        forcesChanged() {
            if (phase === REBUILDING) { hold("forces", () => doOverlay()); return; }
            doOverlay();
        },

        // "Add these frames." A streamed append EXTENDS the movie; it is not a
        // reload (§ 10.5).
        appendFrames(from) {
            if (phase === REBUILDING) { hold("append", () => doAppend(from)); return; }
            doAppend(from);
        },

        // "Show this frame." The library already holds it (§ 10.4), so playing
        // is a frame swap and not a redraw.
        showFrame() {
            if (phase === REBUILDING) { hold("seek", () => doSwap()); return; }
            doSwap();
        },

        // "Draw."
        render() { doOverlay(); },

        /* "Point the camera at it again." An instruction like the rest, and the
         * only one about the window rather than the data: § 9.6 says the camera
         * is fitted to the structure on load AND ON RESET, and § 9.9 keeps it
         * down in the seal where nothing above can read it. Nothing is derived
         * and no frame moves. */
        resetView() { embed.fitCamera(); },

        /* The pose pair (§ 9.6 / § 11.2b): passthroughs with no derivation --
         * the view context reads the pose at a gesture's end and points the
         * camera back on a matching restore.  Nothing here keeps a copy. */
        getCamera() { return embed.getCamera ? embed.getCamera() : null; },
        setCamera(pose) {
            return embed.setCamera ? embed.setCamera(pose) : false;
        },

        /* "Hand over the image" -- § 9.7's other bounded asking of the
         * window (§ 11.3: the drawing library already has the image, so it
         * is asked for it).  A Promise of a PNG Blob at the asked size. */
        capture(width, height) {
            return embed.capture
                ? embed.capture(width, height)
                : Promise.reject(new Error("capture: not available"));
        },

        // "Throw it away."
        dispose() {
            source = null;
            held = [];
            embed.dispose();
        },

        // What each change actually cost. Not a read of the DATA — it is the
        // record of this engine's own instructions, which is what § 13.2's
        // second level checks ("how much work a change takes"). Nothing in the
        // module reads it.
        __costs() { return costLog.slice(); },
        __resetCosts() { costLog = []; },
    };
}
