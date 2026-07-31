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
            switches:       switches(),
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
            markers:   [],
            halos:     [],
            highlight: processed.selection || [],
        });
        embed.setArrows(processed.arrows || []);
    }

    function applyScene() {
        const s = structure();
        const sw = switches();
        const scene = sceneFor(s ? s.cell : null);
        // Geometry travels unconditionally; the switch carries only a boolean
        // (§ 10.3). So the box is handed down whenever it changes and the
        // SWITCH decides whether it is drawn — which is why turning the cell on
        // after a hidden load draws it at the structure's corner and not at the
        // world origin.
        embed.setCell(sw.showCell ? scene.cellBox : null);
        if (sw.showAxis) {
            // The triad rides the ordinary arrow door, carrying its own colours.
            embed.setArrows((embed.__axes = scene.axes));
        }
    }

    /* ── The four costs ───────────────────────────────────────────────────── */

    function doRebuild() {
        const processed = processAll();
        const s = structure();
        if (!s || !processed.length) {
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
        embed.fitCamera();
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
        if (!embed.hasMovie()) { doRebuild(); return; }

        const processed = processAll();
        const fresh = processed.slice(from);
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
        const processed = processAll();
        if (processed[at]) applyOverlaysFor(processed[at]);
        embed.endBatch();
        costLog.push(SWAP);
    }

    function doOverlay() {
        const at = frameNow();
        const processed = processAll();
        embed.beginBatch();
        if (processed[at]) applyOverlaysFor(processed[at]);
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
        try { doRebuild(); } finally { healing = false; }
    }

    /* ── The rebuild window (§ 10.9) ──────────────────────────────────────
     *
     * A rebuild takes long enough to be visible, so it shows the cover and locks
     * the viewer. That leaves a window in which other things arrive anyway — a
     * user click, or a timer-driven poll delivering frames that no amount of
     * disabled buttons could stop. NOTHING THAT LANDS IN THAT WINDOW IS SILENTLY
     * DROPPED.
     */
    async function rebuildGuarded() {
        phase = REBUILDING;
        embed.setBusy("Updating view…");
        try {
            await Promise.resolve();          // the window other work arrives in
            doRebuild();
        } finally {
            embed.setBusy(false);
            phase = IDLE;
            // Replayed IN ARRIVAL ORDER, then the viewer is idle again.
            const queued = held;
            held = [];
            for (const item of queued) item();
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

        // "Here is the data." A whole new structure — the set of drawn atoms
        // changed by definition, so this is a rebuild (§ 10.5 question 1).
        //
        // A full load is NEVER itself refused: everything held is dropped,
        // because it refers to atoms or frames that no longer exist, and the
        // load supersedes the rebuild under way. It is the more authoritative
        // statement about what the structure is (§ 10.9).
        dataChanged() {
            held = [];
            return rebuildGuarded();
        },

        // A switch changed. The set of drawn atoms changed only if isolate is
        // involved; everything else is an overlay refresh with the same atoms
        // drawn (§ 10.5).
        switchesChanged() {
            if (phase === REBUILDING) return;   // nothing is held: the rebuild
                                                // reads the switches when it RUNS
            if (currentDrawnKey() !== drawnKey) return rebuildGuarded();
            doOverlay();
        },

        // "Here is the cell." An overlay refresh, NOT a rebuild — the atoms did
        // not move; only the box and the axes changed (§ 10.5).
        cellChanged() {
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
