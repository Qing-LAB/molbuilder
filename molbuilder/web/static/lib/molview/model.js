/* MolView — the model: the one place the structure lives. § 7 level 3, and the
 * central file of § 7.3.
 *
 * Contract: docs/web/molview.md § 9.3 (the data API), § 9.4 (read-only),
 *           § 6.3 (two copies), § 6.4 (the displayed frame and its range).
 * Owns:     THE MASTER COPY — what a save, an export, a measurement and a server
 *           request all read — plus the selection, the displayed frame and its
 *           range. This is where the rules are enforced and where read-only is
 *           applied, so nothing may go around it.
 * Called by: the handle, and every level inside the same viewer. One model per
 *           owner.
 *
 * NEVER (§ 7 level 3):
 *   - touch the drawing library;
 *   - exist as one shared instance behind several viewers.
 */
"use strict";

import {
    createLoad, createWriteOut, createEdits, createCellEdit, FROZEN_LABEL,
} from "./model-jobs.js";


/**
 * One model, for one owner.
 *
 * @param {object} opts  `{mode}` — "readonly" freezes the master copy (§ 9.4)
 */
export function createModel(opts) {
    opts = opts || {};
    const readOnly = opts.mode === "readonly";

    /* ── The master copy (§ 6.3) ───────────────────────────────────────────
     *
     * Every atom, every frame, in the ORIGINAL order. Kept clean — never
     * overwritten with a cut-down list — so every redraw starts from it rather
     * than from whatever is currently on screen. That is what lets the whole
     * structure come back the moment isolate is turned off.
     */
    let structure = null;             // {elements, annotations, cell} or null
    let frames = null;                // Vec3[][]
    let forcesPerFrame = null;

    /* ── The displayed frame and the range it lives in (§ 6.4) ─────────────
     *
     * ONE FACT, KEPT IN ONE PLACE. "A frame number without the range it is valid
     * in cannot be used for anything — you cannot draw a slider, clamp a seek,
     * or follow the end of a growing run without both." Splitting them is how a
     * slider comes to offer a frame that nothing can draw.
     */
    let frameIndex = 0;

    const structureListeners = [];
    const frameListeners = [];
    let renderer = null;

    /* ── § 6.4's ordering, in one place ────────────────────────────────────
     *
     * The reason this is a function and not four lines repeated at each write:
     * every path that changes the truth must do these in this order, and one
     * that does them in another order is how a subscriber comes to see a range
     * from the new structure beside a frame number from the old one.
     *
     *   1. the master copy is updated first, and COMPLETELY
     *   2. the range is recomputed FROM IT — not from the drawing, and not from
     *      what the caller said it was adding
     *   3. the frame number is checked against that range and moved if it no
     *      longer fits
     *   4. only then is anyone told, and what they see is a matching pair
     */
    function settle(change, options) {
        change();                                             // 1
        const count = Array.isArray(frames) ? frames.length : 0;   // 2
        const wanted = (options && options.resetFrame) ? 0 : frameIndex;
        const resolved = count                                      // 3
            ? Math.min(Math.max(0, Math.floor(wanted)), count - 1)
            : 0;
        const frameMoved = resolved !== frameIndex;
        frameIndex = resolved;

        if (renderer) renderer.dataChanged();
        announceStructure();                                        // 4
        if (frameMoved) announceFrame();
    }

    function announceStructure() {
        for (const fn of structureListeners.slice()) {
            try { fn(); } catch (_) {}
        }
    }

    function announceFrame() {
        for (const fn of frameListeners.slice()) {
            try { fn(frameIndex); } catch (_) {}
        }
    }

    /* ── The read-only gate (§ 9.4) ────────────────────────────────────────
     *
     * "A read-only viewer freezes the master copy. Nothing else changes."
     *
     * One question asked of every entry in § 9.3's table — does this change the
     * master copy? — and if yes it is a NO-OP that returns without effect AND
     * WITHOUT THROWING. There is no list of disabled controls; every previous
     * attempt to describe read-only became one, and it drifted.
     *
     * Wrapping each truth-changing door in this is what makes the guarantee hold
     * even for a door added later: forgetting the wrapper is visible, whereas
     * forgetting to add a name to a list is not.
     */
    function gated(fn, whenFrozen) {
        return function (...args) {
            if (readOnly) return whenFrozen;
            return fn.apply(null, args);
        };
    }

    // Deep enough that changing what you were given can never change the viewer
    // (§ 9.3). Structures are plain data, so this is the whole of it.
    const copy = (v) => (v == null ? v : JSON.parse(JSON.stringify(v)));

    function put(nextStructure, nextCoordinates) {
        structure = nextStructure;
        frames = nextCoordinates.frames;
        forcesPerFrame = nextCoordinates.forcesPerFrame || null;
    }

    /* ══ The helpers, each handed exactly what it may call (§ 7.3) ════════ */

    const installMolecule = createLoad({
        put: (s, c) => settle(() => put(s, c), { resetFrame: true }),
        announce: () => {},                 // settle already told everyone
        recordFirstState: () => { /* the history helper lands in step E */ },
    });

    const exportFile = createWriteOut({
        readStructure: () => structure,
        readFrame:     (i) => (frames ? frames[i] : null),
        currentFrame:  () => frameIndex,
    });

    const applyOp = createEdits({
        readStructure: () => structure,
        readFrame:     (i) => (frames ? frames[i] : null),
        currentFrame:  () => frameIndex,
        readSelection: () => [],            // the selection store lands in step E
        apply: (s, c) => settle(() => put(s, c), { resetFrame: true }),
    });

    const commitPeriodicityOp = createCellEdit({
        readStructure: () => structure,
        readFrame:     (i) => (frames ? frames[i] : null),
        currentFrame:  () => frameIndex,
        // A cell edit does not move an atom, so the frame and its range are
        // untouched — this is why § 10.5 makes it an overlay refresh.
        applyCell: (cell) => settle(() => { structure.cell = cell; }),
    });

    return {
        /* ══ Get the whole structure ═════════════════════════════════════
         *
         * With nothing loaded a read returns NOTHING rather than an empty
         * structure (§ 9.3): "there is nothing here" and "here is a structure
         * with no atoms" are different answers, and a caller has to be able to
         * tell them apart.
         *
         * One read holds everything a request needs, which is why there is no
         * separate door for the facts a request carries — and it is what makes
         * "the facts that leave together were read together" true.
         */
        getStructure() { return copy(structure); },

        // Narrower cuts of that one need. A cut may disappear; it must never
        // grow into a rival (§ 9.3).
        getElements()   { return structure ? structure.elements.slice() : null; },
        getCoordinates() {
            return frames ? copy({ frames, forcesPerFrame }) : null;
        },
        getAtoms() {
            if (!structure) return null;
            return structure.elements.map((element, i) => ({
                index:   i,
                element: element,
                labels:  (structure.annotations[i].labels || []).slice(),
                residue: structure.annotations[i].residue,
            }));
        },
        // "Which atoms are the electrodes" is a real question, and this is a CUT
        // OF THE LABELS — not a second place where groups of atoms are stored.
        getRegions() {
            if (!structure) return null;
            const out = {};
            structure.annotations.forEach((facts, i) => {
                for (const name of (facts.labels || [])) {
                    (out[name] = out[name] || []).push(i);
                }
            });
            return out;
        },
        // The atoms carrying the reserved frozen label. A cut of the same one
        // mechanism (§ 6.6), not a field of its own.
        getFrozen() {
            if (!structure) return null;
            const out = [];
            structure.annotations.forEach((facts, i) => {
                if ((facts.labels || []).indexOf(FROZEN_LABEL) >= 0) out.push(i);
            });
            return out;
        },

        /* ══ Get the cell ════════════════════════════════════════════════
         *
         * The main way in is the cell AS IT WILL ACTUALLY BE USED, so it always
         * has an answer. MolView fills in the SHAPE of what the structure left
         * unsaid; the rules for RESOLVING the values — how much vacuum an
         * isolated axis gets, what a missing axis kind means — belong to
         * model/structure-periodicity.md and are applied by the server. MolView
         * carries the block and interprets none of it (§ 6.2).
         */
        getUnitCellInfo() {
            const cell = structure ? structure.cell : null;
            return {
                lattice:  (cell && cell.lattice)  || null,
                origin:   (cell && cell.origin)   || null,
                axisKind: (cell && cell.axis_kind) || null,
                vacuum:   (cell && cell.vacuum)   || null,
            };
        },
        getUnitCell()       { return copy(structure && structure.cell && structure.cell.lattice) || null; },
        getUnitCellOrigin() { return copy(structure && structure.cell && structure.cell.origin) || null; },
        getAxisKind()       { return copy(structure && structure.cell && structure.cell.axis_kind) || null; },
        getVacuum()         { return copy(structure && structure.cell && structure.cell.vacuum) || null; },

        /* ══ Get one frame's coordinates ═════════════════════════════════
         *
         * Named for exactly what it promises: EVERY atom of frame i, in the
         * ORIGINAL numbering, before isolate cuts anything down. That is what
         * its callers want — measurement resolves panel numbers against it, and
         * export writes the frame from it.
         *
         * There is no rival: reading coordinates back out of the drawing would
         * give the isolated subset under its own renumbering, which is a
         * different thing and one MolView does not offer (§ 6.3).
         */
        getFrameAllAtoms(i) {
            if (!frames) return null;
            const frame = frames[i];
            return frame ? frame.map((p) => [p[0], p[1], p[2]]) : null;
        },

        /* ══ Know / move / follow the displayed frame (§ 6.4) ═════════════
         *
         * Three kinds of access and no fourth. Every UI reads and sets it
         * through exactly this API — the frame bar, a tab's scrubber, a keyboard
         * shortcut, playback, a restored session. There is no privileged writer
         * and no back channel.
         */
        currentFrame() { return frameIndex; },
        frameCount()   { return Array.isArray(frames) ? frames.length : 0; },

        // "A number outside the range is resolved against the range, never taken
        // on trust." Not an error: a slider at the end of a trajectory that just
        // got shorter is asking a reasonable question.
        //
        // NOT gated by read-only: scrubbing is looking at the picture, which is
        // what a read-only viewer is FOR (§ 9.4).
        setCurrentFrame(i) {
            const count = Array.isArray(frames) ? frames.length : 0;
            if (!count) return;
            const next = Math.min(Math.max(0, Math.floor(i)), count - 1);
            if (next === frameIndex) return;
            frameIndex = next;
            if (renderer) renderer.showFrame(next);
            // The write is the only way it moves, and it tells EVERY subscriber
            // regardless of what did the moving. Nothing anywhere needs its own
            // "did it change?" check.
            announceFrame();
        },
        onFrameChange(fn) {
            frameListeners.push(fn);
            return () => {
                const at = frameListeners.indexOf(fn);
                if (at >= 0) frameListeners.splice(at, 1);
            };
        },

        /* ══ Get the structure out as text ═══════════════════════════════
         *
         * Export is a READ (§ 9.4): getting bytes out of a viewer you cannot
         * edit is the point of a read-only viewer, so this is not gated.
         */
        exportFile: exportFile,

        /* ══ Hear that the structure changed ═════════════════════════════ */
        subscribe(fn) {
            structureListeners.push(fn);
            return () => {
                const at = structureListeners.indexOf(fn);
                if (at >= 0) structureListeners.splice(at, 1);
            };
        },

        /* ══ The doors that change the master copy (§ 9.4) ════════════════
         *
         * Each one wrapped in the gate, and each returns a value that says "no"
         * without throwing.
         */
        installMolecule:      gated(installMolecule, Promise.resolve(null)),
        applyOp:              gated(applyOp, Promise.resolve(null)),
        commitPeriodicityOp:  gated(commitPeriodicityOp, Promise.resolve(null)),

        // Load or extend the frames. The range is recomputed from the master
        // copy — never from what the caller said it was adding (§ 6.4 step 2).
        reloadFrames: gated(function (nextFrames, nextForces) {
            settle(() => {
                frames = nextFrames.map((f) => f.map((p) => [p[0], p[1], p[2]]));
                forcesPerFrame = nextForces || null;
            }, { resetFrame: true });
        }, undefined),

        addFrame: gated(function (frame, forces) {
            settle(() => {
                if (!frames) frames = [];
                frames.push(frame.map((p) => [p[0], p[1], p[2]]));
                if (forces || forcesPerFrame) {
                    // Back-fill so the forces of frame f stay at index f: a run
                    // caught at its first geometry carries no forces, and
                    // pushing the first ones that arrive onto an empty list
                    // would attach them to frame 0 for ever after.
                    if (!forcesPerFrame) forcesPerFrame = frames.map(() => null);
                    forcesPerFrame[frames.length - 1] = forces || null;
                }
            });
        }, undefined),

        addFrames: gated(function (moreFrames, moreForces) {
            settle(() => {
                if (!frames) frames = [];
                moreFrames.forEach((f, k) => {
                    frames.push(f.map((p) => [p[0], p[1], p[2]]));
                    if (moreForces || forcesPerFrame) {
                        if (!forcesPerFrame) forcesPerFrame = frames.map(() => null);
                        forcesPerFrame[frames.length - 1] =
                            (moreForces && moreForces[k]) || null;
                    }
                });
            });
        }, undefined),

        setForces: gated(function (perFrame) {
            settle(() => { forcesPerFrame = perFrame || null; });
        }, undefined),

        /* ══ Internal wiring ═════════════════════════════════════════════
         *
         * The renderEngine is CALLED ONLY BY THE MODEL (§ 7 level 5) and is
         * handed the master copy. It is attached here rather than reached,
         * because a renderer the model does not own is a renderer something else
         * can drive.
         */
        _attachRenderer(engine) {
            renderer = engine;
            if (engine) {
                engine.setDataSource({
                    structure: () => structure,
                    frames:    () => frames,
                    forces:    () => forcesPerFrame,
                    frame:     () => frameIndex,
                });
            }
        },
    };
}
