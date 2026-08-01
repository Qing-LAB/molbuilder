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
    structureForServer, groupByLabel, effectiveCell,
    resolveFilter as askServerToFilter,
} from "./model-jobs.js";
import { createSelectionStore, createViewStore } from "./stores.js";
import { createHistory } from "./history.js";


/* ══ What changed, in the renderEngine's own words (§ 9.7, § 10.5) ═══════════
 *
 * § 9.7's surface is commands — "here is the data", "add these frames", "the
 * forces changed", "here is the cell" — and § 10.5 says the ENGINE chooses what
 * each costs, from what changed, in one place. That only works if it is told
 * WHICH change this was.
 *
 * Saying "the data changed" for every write, which is what this did, collapses
 * four costs into one: a streamed append reloaded the whole movie instead of
 * extending it, a cell edit reloaded it, and tagging an atom — which moves
 * nothing and draws nothing — reloaded it too. The cost table was correct and
 * unreachable.
 *
 * `none` is not an omission. A label is a fact about an atom, not a thing on
 * screen (§ 6.6): the panel redraws because the structure changed, and the
 * drawing has nothing to do.
 */
const REDRAW = {
    data:   (engine)       => engine.dataChanged(),
    append: (engine, from) => engine.appendFrames(from),
    forces: (engine)       => engine.forcesChanged(),
    cell:   (engine)       => engine.cellChanged(),
    none:   ()             => {},
};


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

    /* ── WHAT THIS VIEWER IS, in one place (§ 5.2) ─────────────────────────
     *
     *   EMPTY    nothing has been put in yet.
     *   HOLDING  a structure is installed.
     *
     * § 9.4 freezes THE CORE DATA — and a viewer with nothing in it has none to
     * freeze. So the write that takes it out of EMPTY is allowed in any mode:
     * that is how a host says which structure this viewer shows. It is CHECKED
     * AND SET in one place, so coming back to the install door never finds EMPTY
     * a second time, and every install after the first meets the gate like any
     * other change.
     *
     * That keeps § 9.3's "the only way a structure gets in" true — there is no
     * second door — while making § 8's "a viewer mounts before it has a
     * structure" and § 12.3's read-only Results viewer both possible.
     *
     * IT REPLACES TWO ANSWERS TO ONE QUESTION. "Has this viewer got its core
     * data?" was asked as `seeded` at the install door and as "is the atom count
     * zero?" at the frame doors, and the two could disagree: a payload carrying
     * an empty atom list left `seeded` true while the count said nothing was
     * loaded, so one door refused a second install and the other refused to
     * append to it. § 5.2 is exactly this — two copies of a fact are two things
     * that must be kept in step.
     */
    const EMPTY = "empty";
    const HOLDING = "holding";
    let unit = EMPTY;

    /* ── The displayed frame and the range it lives in (§ 6.4) ─────────────
     *
     * ONE FACT, KEPT IN ONE PLACE. "A frame number without the range it is valid
     * in cannot be used for anything — you cannot draw a slider, clamp a seek,
     * or follow the end of a growing run without both." Splitting them is how a
     * slider comes to offer a frame that nothing can draw.
     */
    let frameIndex = 0;

    /* The name the structure came in under, kept for one caller: the default
     * filename of an export (§ 11.4). It is not structure data — no calculation
     * reads it, and it never travels to the server — which is why it sits here
     * beside the frame rather than inside § 6.2's Structure. */
    let sourceName = null;

    const structureListeners = [];
    const frameListeners = [];
    let renderer = null;

    /* ── The stores, assembled here and reached only through here (§ 9.3) ──
     *
     * § 7 level 4: "a change asked for through a store meets the same rules as
     * one asked for anywhere else". That is true because the model builds them
     * and hands each exactly what it may call — the label door goes back through
     * the gate below, and nothing else in the selection store touches the truth.
     */
    const selection = createSelectionStore({
        resolveFilter: (rule) => resolveFilter(structure, rule),
        writeLabel:    (name, atoms, verb) => writeLabel(name, atoms, verb),
    });
    const view = createViewStore();

    /* ── The switches and the selection reach the drawing (§ 10.5) ─────────
     *
     * Without this the store was complete, the renderEngine was correct, and
     * NOTHING ARRIVED: every frame was worked out from `{}` switches and an
     * empty selection, so atom-number labels, force arrows, the cell box, the
     * axes, the highlight and isolate were dead at once — six features, one
     * missing wire. It is the model that has to make this call, because the
     * renderEngine is called only by the model (§ 7 level 5) and a store may not
     * reach past it.
     *
     * WHAT IS COMPARED is exactly what the frame calculation reads (§ 6.5). The
     * snapshot also carries which editor is showing and the filter rows being
     * typed; those change no pixel, and re-deriving on every keystroke in a
     * filter box would be work with an identical answer. */
    let drawnFrom = null;
    selection.subscribe((state) => {
        const reads = JSON.stringify([
            state.selection, state.isolate, state.showIndex, state.showForces,
            state.showCell, state.showAxis, state.forceScale,
        ]);
        if (reads === drawnFrom) return;
        drawnFrom = reads;
        if (renderer) renderer.switchesChanged();
    });

    /* ── The history (§ 11.2) ──────────────────────────────────────────────
     *
     * Handed a way to record a state and a way to put one back, and nothing
     * else — it never looks inside either. What a state HOLDS is decided here,
     * by § 11.2's one rule: state is the truth, and what you are looking at is
     * not. So the structure, its frames and the selection go in; the camera, the
     * displayed frame, the switches and the drawing settings do not.
     */
    const history = createHistory({
        recordState: () => ({
            structure:   copy(structure),
            coordinates: copy({ frames, forcesPerFrame }),
            selection:   selection.get(),
        }),
        restoreState: (state) => {
            if (!state) return;
            settle(() => {
                structure = state.structure;
                frames = state.coordinates ? state.coordinates.frames : null;
                forcesPerFrame = state.coordinates ? state.coordinates.forcesPerFrame : null;
            }, { resetFrame: true });
            // It opens at the first frame, fitted, with the switches off and the
            // drawing back at its defaults — because none of that was ever part
            // of what you were working on (§ 11.2).
            selection.adopt(state.selection || []);
            view.reset();
        },
        store: opts.workspace || { read: async () => null, write: async () => {} },
        onBadge: () => announceStructure(),
    });

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
        const opts = options || {};
        change();                                             // 1
        const count = Array.isArray(frames) ? frames.length : 0;   // 2
        const wanted = opts.resetFrame ? 0 : frameIndex;
        const resolved = count                                      // 3
            ? Math.min(Math.max(0, Math.floor(wanted)), count - 1)
            : 0;
        const frameMoved = resolved !== frameIndex;
        frameIndex = resolved;

        if (renderer) (REDRAW[opts.redraw] || REDRAW.data)(renderer, opts.from);
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

    // The selection store's two doors back into the truth.
    function resolveFilter(current, rule) {
        return askServerToFilter(current, rule);
    }

    /**
     * Write a label onto a set of atoms (§ 9.5).
     *
     * GATED, because it is a change to the structure: the label becomes part of
     * what the atom is, goes into the sidecar and reaches the calculation. It is
     * reached from the selection door only because the atoms it applies to are
     * the selection, and that is convenience, not a drawing concern.
     *
     * Applying a label REPLACES that label's previous set of atoms.
     */
    const writeLabel = gated(function (name, atoms, verb) {
        if (!structure || !name) return false;
        const wanted = new Set(atoms);
        // A label changes what an atom IS, not what is drawn (§ 6.6): the panel
        // redraws because the structure changed, and the drawing has nothing to
        // do. This used to reload the whole movie.
        settle(() => {
            structure.annotations.forEach((facts, i) => {
                const had = (facts.labels || []).indexOf(name) >= 0;
                const picked = wanted.has(i);
                // One expression for all three verbs, so they cannot drift:
                //   replace — the label's set BECOMES the selection (§ 9.5)
                //   add     — union
                //   remove  — difference
                const keep = verb === "add"    ? (had || picked)
                           : verb === "remove" ? (had && !picked)
                           : picked;
                const labels = (facts.labels || []).filter((l) => l !== name);
                if (keep) labels.push(name);
                facts.labels = labels;
            });
        }, { redraw: "none" });
        history.edited();
        return true;
    }, false);

    /* ── The same-atoms rule, at the doors that could break it (§ 10.8) ────
     *
     * "A full load replaces everything. It establishes the atoms' identity —
     * count, elements, order — from frame 0." An APPEND adds to that identity
     * and may never change it, and § 10.8 states three rules for it:
     *
     *   1. SOMETHING MUST ALREADY BE LOADED. "Appending with nothing loaded is
     *      a hard error — there is no atom identity to append to."
     *   2. EACH NEW FRAME IS CHECKED AGAINST THAT IDENTITY before anything
     *      reaches the drawing. Same atom count. Elements are not re-sent,
     *      because a streamed frame carries coordinates only.
     *   3. A MISMATCH IS A HARD ERROR. "Never padded, never truncated, never
     *      guessed into fitting."
     *
     * None of it was here, and both halves failed silently: appending with
     * nothing loaded INVENTED an identity (`if (!frames) frames = []`), and a
     * frame of the wrong length was pushed straight into the master copy — so a
     * structure could hold two elements and a frame with one position. That
     * breaks the same-atoms rule of § 6.2 that everything downstream reads
     * against: the per-frame maths, measurement, and export all index the
     * coordinates by the element list.
     *
     * THEY THROW rather than returning false. A caller that appends a frame of
     * the wrong shape has a bug, and a frame dropped quietly leaves a hole in
     * the middle of a run that nothing downstream can see.
     */
    function atomCount() {
        return (structure && Array.isArray(structure.elements))
            ? structure.elements.length : 0;
    }

    // Against a count named by the caller, so the same check serves a frame
    // joining the loaded structure AND the frames arriving WITH a load, where
    // the identity being checked against is the one still being installed.
    function requireCount(coords, n, label) {
        if (!n) {
            throw new Error(label + ": no atoms to check against (§ 10.8)");
        }
        const got = Array.isArray(coords) ? coords.length : 0;
        if (got !== n) {
            throw new Error(label + ": a frame of " + got + " atoms cannot join "
                            + "a structure of " + n + " (§ 10.8)");
        }
    }

    // A frame joining the structure ALREADY held. "Something must already be
    // loaded" is the viewer's own state, asked where it lives (§ 10.8 rule 1).
    function requireMatch(coords, label) {
        if (unit === EMPTY) {
            throw new Error(label + ": nothing loaded — there is no atom "
                            + "identity to append to (§ 10.8)");
        }
        requireCount(coords, atomCount(), label);
    }

    // The same rule across a frame SET: every frame carries the same atoms.
    // Checked before any of them lands, so a bad frame halfway through a batch
    // does not leave the first half applied.
    function requireSameAtoms(list, label) {
        if (!Array.isArray(list) || !list.length) {
            throw new Error(label + ": needs at least one frame");
        }
        const n0 = Array.isArray(list[0]) ? list[0].length : 0;
        list.forEach((frame, i) => {
            const got = Array.isArray(frame) ? frame.length : 0;
            if (got !== n0) {
                throw new Error(label + ": frame " + i + " has " + got
                                + " atoms, expected " + n0 + " (§ 10.8)");
            }
        });
        return n0;
    }

    function put(nextStructure, nextCoordinates, name) {
        structure = nextStructure;
        frames = nextCoordinates.frames;
        forcesPerFrame = nextCoordinates.forcesPerFrame || null;
        // Only a LOAD names a structure. An edit replaces the atoms of the one
        // already open, so it keeps the name it came in under.
        if (name !== undefined) sourceName = name;
    }

    /* ══ The helpers, each handed exactly what it may call (§ 7.3) ════════ */

    const installMolecule = createLoad({
        put: (s, c, name) => {
            settle(() => put(s, c, name), { resetFrame: true });
            unit = HOLDING;
        },
        announce: () => {},                 // settle already told everyone
        // Point 0 — "the one point nobody asks for", the floor the sequence
        // stands on so a Retract from the first edit has somewhere to land. It
        // also clears anything held for the structure just replaced (§ 11.2).
        //
        // A READ-ONLY VIEWER ANCHORS NOTHING. § 9.4: it has no history, because
        // a history exists to get back to a state you left and nothing here can
        // leave one. Anchoring would also write point 0 to the workspace, which
        // is a persist a read-only viewer has no business doing.
        recordFirstState: () => (readOnly ? null : history.anchor()),
        /* The same-atoms rule (§ 10.8) applied to frames arriving WITH the load,
         * where the identity to check against is the one being installed. */
        checkFrames: (list, n) => {
            requireSameAtoms(list, "installMolecule");
            requireCount(list[0], n, "installMolecule");
        },
    });

    /* THE STRUCTURE AS DATA — ONE producer, read in one place and handed to
     * everything that sends or writes it (§ 9.3: "the facts that leave together
     * were read together"). The coordinates come from the DISPLAYED frame —
     * § 5.1's promise at the point it matters — and the metadata from the same
     * read, so the two can never be one edit apart.
     *
     * The SAME producer an edit uses. There were two, and two producers of one
     * fact is how an export came to write a server-request payload into a
     * `.molstruct.json`. */
    const readData = () => structureForServer(structure,
                                              frames ? frames[frameIndex] : null);

    const exportFile = createWriteOut({
        readData:   readData,
        readSource: () => sourceName,
    });

    const applyOp = createEdits({
        readStructure: () => structure,
        readFrame:     (i) => (frames ? frames[i] : null),
        currentFrame:  () => frameIndex,
        readSelection: () => selection.get(),
        apply: (s, c, countChanged) => {
            settle(() => put(s, c), { resetFrame: true });
            // The badge is raised HERE, inside the gate and after the change has
            // landed — which makes two of the contract's rules fall out rather
            // than needing cases of their own: a read-only viewer never reaches
            // this line, so its badge never appears (§ 9.4), and a failed edit
            // never reaches it either, so nothing is recorded (§ 11.1).
            history.edited();
            // An operation that grows or shrinks the structure clears the
            // selection: a kept one could point at an atom that is no longer the
            // one it meant. A count-preserving transform leaves it alone.
            if (countChanged) selection.clear();
        },
    });

    const commitPeriodicityOp = createCellEdit({
        readData: readData,
        // A cell edit does not move an atom, so the frame and its range are
        // untouched — this is why § 10.5 makes it an overlay refresh.
        applyCell: (block) => {
            settle(() => { structure.periodicity = block; }, { redraw: "cell" });
            history.edited();
        },
    });

    return {
        /* ══ Get the whole structure ═════════════════════════════════════
         *
         * THE MASTER COPY, WHOLE — every atom, every frame, in the original
         * order (§ 6.3). Every field § 6.2 lists is in here: the elements, the
         * per-atom facts, the cell block, the frames and their forces.
         *
         * ONE READ HOLDS EVERYTHING A REQUEST NEEDS, which is why there is no
         * separate door for the facts a request carries (§ 9.3), and it is what
         * makes "the facts that leave together were read together" a property of
         * this surface rather than a promise about how callers behave.
         *
         * This returned three of the five and left THE COORDINATES OUT, so the
         * one thing § 9.3 exists to prevent was what every caller had to do:
         * read the labels here and the positions somewhere else, and send a set
         * assembled from two moments. That is the failure § 9.3 tells the story
         * of — current labels with stale positions, and a server judging a
         * structure that was not the one on screen.
         *
         * With nothing loaded it returns NOTHING rather than an empty structure
         * (§ 9.3): "there is nothing here" and "here is a structure with no
         * atoms" are different answers, and a caller has to be able to tell them
         * apart.
         */
        getStructure() {
            if (!structure) return null;
            return copy({
                elements:       structure.elements,
                annotations:    structure.annotations,
                periodicity:    structure.periodicity,
                frames:         frames,
                forcesPerFrame: forcesPerFrame,
            });
        },

        // Narrower cuts of that one need. A cut returns exactly what the main
        // way in holds for that field, so the two cannot disagree (§ 13.3). A
        // cut may disappear; it must never grow into a rival (§ 9.3).
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
            return structure ? groupByLabel(structure.annotations) : null;
        },
        // The atoms carrying the reserved frozen label. A cut of the same one
        // mechanism (§ 6.6), not a field of its own.
        getFrozen() {
            if (!structure) return null;
            // A cut of the same grouping, not a second scan for the same fact.
            return groupByLabel(structure.annotations)[FROZEN_LABEL] || [];
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
            /* THE CELL AS IT WILL ACTUALLY BE USED — the server works it out and
             * sends it beside the raw values, and this reads it (§ 6.2: MolView
             * interprets none of it).
             *
             * THROUGH THE ONE FUNCTION THAT ANSWERS THAT QUESTION, because the
             * drawing asks it too (§ 5.2). This used to spell the fallback out
             * here while `sceneFor` spelled a different one out there, and the
             * two disagreed: this said a plain `.xyz` had a cell, the drawing
             * said it had none, and "Show unit cell" drew nothing while the Cell
             * page listed a lattice.
             *
             * COPIED, like every other read (§ 9.3) — the main way in was the one
             * way in that could be written through, which is the opposite of the
             * rule. */
            return copy(effectiveCell(structure && structure.periodicity));
        },
        // The RAW values — what the structure actually says, `null` where it says
        // nothing (§ 9.3: "the raw 3×3 or null").
        getUnitCell()       { return copy(structure && structure.periodicity && structure.periodicity.cell) || null; },
        getUnitCellOrigin() { return copy(structure && structure.periodicity && structure.periodicity.cell_origin) || null; },
        getAxisKind()       { return copy(structure && structure.periodicity && structure.periodicity.axis_kind) || null; },
        getVacuum()         { return copy(structure && structure.periodicity && structure.periodicity.vacuum) || null; },

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
        /* Leaving EMPTY is allowed in any mode: there is no core data to freeze
         * yet, so the first install is how a host says which structure this
         * viewer shows.
         *
         * REPLACING what is held is refused in a read-only viewer — unless the
         * caller says it means to, with `enforce`. Deciding which structure a
         * viewer shows is the HOST's business, and the host is the one that
         * asked for read-only; what read-only protects is the core data from
         * being EDITED (§ 9.4), and swapping the structure outright is not an
         * edit of it. Saying so explicitly is the whole of the flag's job: it
         * costs one word at the call site and leaves no state a viewer can get
         * stuck in.
         */
        installMolecule(input) {
            const enforced = !!(input && input.enforce);
            if (readOnly && unit !== EMPTY && !enforced) {
                return Promise.resolve(null);
            }
            return installMolecule(input);
        },
        applyOp:              gated(applyOp, Promise.resolve(null)),
        commitPeriodicityOp:  gated(commitPeriodicityOp, Promise.resolve(null)),

        /* ══ Load or extend the frames — NOT gated (§ 9.4) ════════════════
         *
         * The gate is on the doors that CHANGE THE CORE DATA — the structure and
         * the metadata that travels with it. These four are not those. They
         * deliver coordinates for the structure already installed: a running
         * job's own output arriving, poll after poll.
         *
         * THE SAME IN BOTH MODES. Not a read-only concession and not an editable
         * privilege — an editable viewer follows a running job exactly as a
         * read-only one does. What the mode decides is whether the structure can
         * be EDITED and whether a point can be recorded, and neither of those is
         * what happens here.
         *
         * None of them raises the unsaved badge either, and for the same reason:
         * the badge means "there is work here that is not on the sequence yet"
         * (§ 11.2), and a run's own output is not the user's work. A poll
         * arriving every few seconds would otherwise flicker it continuously.
         *
         * "What they cannot do is change the structure the calculation ran on"
         * — and frames from that calculation do not; they ARE it. § 10.8's
         * guards are what make that a fact rather than a reading: these doors
         * cannot alter the atom count, the elements, the labels or the cell.
         * They can only add positions for atoms whose identity was fixed at
         * load, so there is nothing here for the gate to protect.
         *
         * Gating them was reading § 9.3's "does this change the master copy?"
         * literally, and it cost the two things a read-only viewer is FOR: a
         * Results tab could not follow a running optimization (§ 12.2), and
         * § 12.3's read-only viewer could not "scrub to the last frame" because
         * the only frame it could ever hold was the one it was seeded with.
         *
         * The range is recomputed from the master copy — never from what the
         * caller said it was adding (§ 6.4 step 2).
         *
         * `reloadFrames` IS THE EXCEPTION AMONG THEM, and it sits here rather
         * than with the append doors for one reason: it REPLACES every
         * coordinate and can shrink the trajectory, where the others only
         * extend. After an append, frame 0 is still exactly what the run
         * produced; after a reload it need not be. That is a change to the
         * truth, so in a read-only viewer it is refused unless the caller says
         * it means it — the same word, for the same reason, as replacing the
         * structure outright.
         */
        reloadFrames: (function (nextFrames, options) {
            const nextForces = (options && options.forces) || null;
            if (readOnly && !(options && options.enforce)) return;
            // Checked BEFORE anything lands (§ 10.8): every frame carries the
            // same atoms, and those atoms are the loaded structure's.
            requireSameAtoms(nextFrames, "reloadFrames");
            requireMatch(nextFrames[0], "reloadFrames");
            settle(() => {
                frames = nextFrames.map((f) => f.map((p) => [p[0], p[1], p[2]]));
                forcesPerFrame = nextForces || null;
            }, { resetFrame: true });
        }),

        // `{forces}` — an OPTIONS object, which is the shape § 12.2's worked
        // example uses (`addFrames(newFrames, {forces})`). It took a bare array,
        // so the call the document shows handed an object where a list was
        // indexed and every force silently became null.
        addFrame: (function (frame, options) {
            const forces = (options && options.forces) || null;
            requireMatch(frame, "addFrame");
            // Where the new frames start is read BEFORE the change: it is what
            // the engine needs to extend the movie instead of reloading it, and
            // after the write it would be indistinguishable from the end.
            const from = Array.isArray(frames) ? frames.length : 0;
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
            }, { redraw: "append", from: from });
        }),

        addFrames: (function (moreFrames, options) {
            const moreForces = (options && options.forces) || null;
            // Every arriving frame is checked against the loaded identity
            // BEFORE any of them lands, so a bad frame halfway through a poll's
            // batch does not leave the first half applied (§ 10.8).
            requireSameAtoms(moreFrames, "addFrames");
            moreFrames.forEach((f) => requireMatch(f, "addFrames"));
            const from = Array.isArray(frames) ? frames.length : 0;
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
            }, { redraw: "append", from: from });
        }),

        setForces: (function (perFrame) {
            settle(() => { forcesPerFrame = perFrame || null; },
                   { redraw: "forces" });
        }),

        /* What this viewer IS, which is not the same as what it holds. § 9.4
         * says a read-only viewer "does not show the controls the gate would
         * swallow" — so MolView has to be able to ask which kind it is. This is
         * configuration, not data; the gate is still the thing that makes the
         * guarantee true, and hiding a control is courtesy on top of it. */
        get mode() { return readOnly ? "readonly" : "editable"; },

        /* ══ Reach the selection / the drawing settings (§ 9.3) ═══════════
         *
         * Doors rather than values: reaching one is how you ASK for a change,
         * and every change asked for through one meets the same rules as one
         * asked for here (§ 9.4). Nothing outside holds on to a door after it
         * has used it.
         */
        selection: selection,
        view:      view,

        /* ══ Save a point, and move through the sequence (§ 11.2) ═════════
         *
         * "A read-only viewer has no history." Saving does not itself change the
         * master copy — it records it — but a history exists to get back to a
         * state you left, and in a read-only viewer nothing can leave one. There
         * is nothing to record and nowhere to go back to, so these are no-ops
         * too, and the badge never appears.
         */
        save: gated((step) => history.save(step), Promise.resolve(false)),
        load: gated((step) => history.load(step), Promise.resolve(null)),
        undo: gated(() => history.undo(), Promise.resolve(null)),

        /* ══ Know where you are in the history (§ 9.3) ════════════════════
         *
         * A read, not a write — which is why it is its own row. Sitting it
         * beside the writes that move it made the "does this change the master
         * copy?" column unanswerable.
         */
        get state_index() { return history.state_index; },
        get uncommitted() { return readOnly ? false : history.uncommitted; },

        // A multi-step change: writes asked for inside are held and one lands at
        // the end, carrying the settled state (§ 11.2).
        beginChange() { history.beginChange(); },
        endChange()   { return history.endChange(); },

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
                /* Everything the frame calculation is a function of (§ 10.2):
                 * the data, AND what the user has set. Handing over only the
                 * first four is how the switches came to reach nothing — the
                 * renderEngine read `switches()` and `selection()` off this
                 * object, found neither, and drew from empty defaults without
                 * anything failing anywhere. */
                engine.setDataSource({
                    structure: () => structure,
                    frames:    () => frames,
                    forces:    () => forcesPerFrame,
                    frame:     () => frameIndex,
                    switches:  () => selection.switches(),
                    selection: () => selection.get(),
                });
            }
        },
    };
}
