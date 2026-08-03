/* MolView — the model's helpers: load a structure in, write it out, and the
 * geometry edits. Plus the one place the server's names become this module's.
 *
 * Contract: docs/web/molview.md § 7.3 (one central file, and helpers it hands
 *           work to), § 11.1 (geometry edits go to the server), § 9.3.
 * Owns:     three jobs, each doing only its own — load, write out, edits — and
 *           the inbound/outbound translation they share.
 * Called by: model.js, and nothing else. Not reachable from outside the model.
 *
 * THE RULE THAT MAKES THESE HELPERS (§ 7.3): "When the central file builds a
 * helper, it hands over exactly the functions that helper is allowed to call."
 *
 * NEVER (§ 7.3):
 *   - reach out on its own. A helper calls what it was handed and nothing else.
 *   - hold the cell. It is a field of the structure (§ 6.2), so it lives with
 *     the structure and is edited through the one cell door (§ 9.3).
 */
"use strict";


/* ══ The translation — one place, both directions ════════════════════════════
 *
 * § 11.1: "On the way in, the server's payload is normalised into the shapes of
 * § 6.2 — the server's names become this module's names, in one place, so
 * nothing downstream has to know both."
 *
 * § 9.3 says outbound is the same job facing the other way and belongs in the
 * same place. That is why there is no separate door for "the facts a request
 * carries": such a door would not be a second NEED, only a second SHAPE — the
 * same facts renamed and regrouped for the wire — and shaping a payload is a
 * translation.
 *
 * This is NOT where an engine's atom numbering is dealt with. The canonical
 * identity is the 0-based index into the structure, and turning that into a
 * SIESTA or PySCF atom number happens at each engine's input writer, server-side
 * (`molbuilder/engine_atom_index.py`). Nothing that crosses this boundary is
 * ever in an engine's numbering, so there is nothing here to convert.
 */

/* THE reserved label's ONE name, the same name the server stores it under.
 *
 * § 6.2 says an atom's facts are its labels and its residue — there is no frozen
 * field — and § 6.6 says a reserved label costs a NAME and one accessor and
 * nothing else. Both ends now hold that: the label arrives in `regions` with
 * every other label, and `getFrozen` is the one designated read of it.
 *
 * This file used to carry two translators here, and both are gone with the
 * server's second store (2026-07-31): an inbound alias that turned an
 * `is_frozen` flag into a label, and an outbound split that pulled the label
 * back out into a `frozen_atoms` field. The NAME did not have to change for
 * either to go: it is the name the server's `by label` rule always matched, and
 * now it is the name the server stores it under too.
 */
export const FROZEN_LABEL = "frozen_atoms";


/* THE NAMES OFFERED BEFORE ANYONE HAS USED THEM — MolView's own list.
 *
 * A predefined label is a SPELLING, not a meaning. It saves the user retyping
 * the four names nearly every device structure uses, because a retyped name is
 * where `L-electrode` and `L-Electrode` become two regions that look like one.
 * They are conveniences — commonest in SIESTA today, and the same idea will fit
 * whatever engine comes next. MolView assigns none of them and reads meaning
 * into none of them; each is a name sitting ready to be used.
 *
 * ONLY `frozen_atoms` IS DIFFERENT, and it is different because something
 * downstream acts on it: the atoms wearing it are held still by the
 * calculation. That is the one a user is owed a warning about before tagging
 * atoms with it by accident, and it is why it is `tone: "warn"` below while the
 * others are just told apart from each other.
 *
 * WHY THE LIST LIVES HERE AND NOT IN THE TABS (user decision, 2026-08-03).
 * It was a mount option, so each page would have had to hand in the same four
 * names — five copies of one list, drifting. Only the module's own demo page
 * ever did, which is why every label chip on every real tab came out the same
 * colour. These are MolView's conveniences, so MolView keeps them.
 *
 * `frozen_atoms` appears here as the CONSTANT, never as a second literal: the
 * reserved meaning costs one name and one accessor, and that name is spelled
 * once, above. */
export const PREDEFINED_LABELS = [
    { name: "L-electrode", tone: 1, description: "the left semi-infinite lead" },
    { name: "R-electrode", tone: 2, description: "the right semi-infinite lead" },
    { name: "bridge",      tone: 3,
      description: "the scattering region between the leads" },
    { name: "interface",   tone: 4,
      description: "contact atoms inside the bridge" },
    { name: FROZEN_LABEL,  tone: "warn",
      description: "these atoms are held still by the calculation" },
];

/* Just the names, in order — what every chooser offers. */
export const PREDEFINED_LABEL_NAMES = PREDEFINED_LABELS.map((e) => e.name);

/**
 * The server's structure payload, in the shapes of § 6.2.
 *
 * Returns `{structure, coordinates}` — or null, because with nothing loaded a
 * read must return NOTHING rather than an empty structure (§ 9.3): "there is
 * nothing here" and "here is a structure with no atoms" are different answers.
 */
export function structureFromServer(payload) {
    /* An answer with no atoms is a structure with no atoms, and that is the
     * caller's business. The module does not second-guess what was loaded — it
     * carries what it is given (§ 6.2). A viewer holding one is not stuck
     * either: an install can be enforced (§ 9.4). */
    if (!payload || !Array.isArray(payload.atoms)) return null;
    const atoms = payload.atoms;

    // Some per-atom facts arrive as parallel top-level arrays rather than on the
    // atom. Folding them in here is what lets one read of the structure hold
    // everything (§ 9.3) instead of callers keeping a second, parallel copy.
    const parallel = (name) => (Array.isArray(payload[name])
        && payload[name].length === atoms.length) ? payload[name] : null;
    const residueNames = parallel("residue_names");

    const elements = atoms.map((a) => a.element);
    const annotations = atoms.map((a, i) => {
        /* THE LABELS AN ATOM CARRIES ARE A SET (§ 6.2). Carrying the same name
         * twice is not a state the model may hold, so the duplicate is dropped
         * HERE — at the one place the server's payload becomes this module's
         * shape (§ 11.1) — rather than left for every reader to cope with.
         *
         * It matters because the count travels: `groupByLabel` turns a label an
         * atom carries twice into that atom's index listed twice, which goes
         * out in `regions`, into the sidecar, and into the generated input. A
         * `frozen_atoms: [0, 0]` reaching a constraints block is the same atom
         * held still twice. The writing side cannot produce this — it strips
         * every occurrence before adding one back — so the only way in is a
         * payload that already had it. */
        const labels = Array.isArray(a.regions)
            ? Array.from(new Set(a.regions)) : [];
        const residue = a.residue_name != null
            ? a.residue_name
            : (residueNames ? residueNames[i] : null);
        const facts = { labels: labels };
        if (residue != null && residue !== "") facts.residue = residue;
        return facts;
    });

    return {
        structure: {
            elements:    elements,
            annotations: annotations,
            /* THE PERIODICITY BLOCK, CARRIED VERBATIM (§ 6.2). Its field
             * names are the server's — `cell`, `cell_origin`, `axis_kind`,
             * `vacuum`, and the `resolved_*` answers beside them — and they stay
             * the server's the whole way through this module.
             *
             * It used to be renamed here to `{lattice, origin, …}`, and the
             * readers then asked a block that has never had a `lattice` key for
             * one. So the cell was null for every structure ever loaded: the box
             * could not be drawn, the axes always fell back to the Cartesian
             * triad, and an export carried no cell. Nothing failed, because a
             * missing key reads as "this structure is not periodic". */
            periodicity: payload.periodicity || null,
        },
        coordinates: {
            frames: [atoms.map((a) => [Number(a.x) || 0,
                                       Number(a.y) || 0,
                                       Number(a.z) || 0])],
            forcesPerFrame: null,
        },
        /* THE VERDICT TRAVELS WITH THE DATA. Every structure-returning route
         * leaves through the server's one exit, which validates what it is
         * about to send (structure-periodicity.md § 8.1), so the payload can
         * carry conditions about the structure beside it. Read HERE, where the
         * payload becomes this module's shape (§ 11.1), so no caller learns the
         * wire spelling and no caller can forget to look. */
        notices: Array.isArray(payload.notices) ? payload.notices : null,
    };
}

/* ══ The cell as it will actually be used ═══════════════════════════════════
 *
 * § 9.3 asks the block one question — "the cell as it will actually be used,
 * with the defaults filled in for whatever the structure left unsaid, so it
 * ALWAYS HAS AN ANSWER" — and the answer is the server's own: it sends the
 * `resolved_*` values beside the raw ones, and this reads them rather than
 * working anything out (§ 6.2: MolView interprets none of it).
 *
 * IT IS ONE FUNCTION BECAUSE THE QUESTION HAS ONE ANSWER (§ 5.2). Two readers
 * asked it separately and disagreed: the Cell page read the resolved values and
 * said a structure had a cell, while the drawing read the RAW ones and found
 * none — so "Show unit cell" drew nothing at all for every structure that had
 * not been given an explicit cell, which is every plain `.xyz`. The panel and
 * the window described different structures, and neither failed.
 */
export function effectiveCell(periodicity) {
    const per = periodicity || {};
    return {
        cell:        per.resolved_cell        || per.cell        || null,
        cell_origin: per.resolved_cell_origin || per.cell_origin || null,
        axis_kind:   per.axis_kind || null,
        vacuum:      per.resolved_vacuum      || per.vacuum      || null,
    };
}


/* ══ The labels, walked once ════════════════════════════════════════════════
 *
 * An atom carries a list of the names it is tagged with. Everything that needs
 * the flipped form — `{"L-electrode": [0, 1]}` — gets it from here, and the walk
 * exists once. It was written four times, and two of the copies differed from
 * the other two in a way that was correct but invisible: a reader could not tell
 * the deliberate variation from drift.
 */
export function groupByLabel(annotations) {
    const out = {};
    (annotations || []).forEach((facts, i) => {
        // An atom appears at most once under a label, whatever its own list
        // says. The inbound translation already drops a repeated name, and this
        // is the belt on the other side: every outbound `regions` and every
        // answer `getRegions` gives is read from this one walk, so a count that
        // could not be right has nowhere to enter.
        for (const name of new Set(facts.labels || [])) {
            (out[name] = out[name] || []).push(i);
        }
    });
    return out;
}

/**
 * The same facts, shaped for the wire — the outbound half.
 *
 * Takes ONE read of the structure and one frame of coordinates, so "the facts
 * that leave together were read together" (§ 9.3). That property is not
 * something a special door provides; it is what one read returning the whole
 * structure means. It matters because it went wrong once: a tab read the labels
 * and the cell fresh as it sent a request while the coordinates came from a copy
 * taken at page load, so the request carried current labels with stale positions
 * and the server judged a structure that was not the one on screen.
 */
export function structureForServer(structure, positions) {
    if (!structure || !positions) return null;

    /* THE COUNT INVARIANT, checked before anything leaves. The coordinates and
     * the per-atom facts are two lists that must index the same atoms; if they
     * disagree this REFUSES rather than sending a structure whose labels point
     * at atoms that are not there (§ 9.3). */
    const count = structure.elements.length;
    if (positions.length !== count || structure.annotations.length !== count) {
        return null;
    }

    /* COPIED, not referenced: this is handed to a caller (§ 9.3), and a body
     * holding the master copy's own arrays is a write into the structure
     * disguised as a read. */
    const clone = (v) => (v == null ? null : JSON.parse(JSON.stringify(v)));
    const per = structure.periodicity || {};
    return {
        elements:  structure.elements.slice(),
        positions: positions.map((p) => [p[0], p[1], p[2]]),
        /* METADATA IS NESTED, because that is where the envelope keeps it
         * (web-api.md § 1 — the envelope IS the structure's canonical dict, and
         * `Structure.from_dict` reads the block from here). These fields sat at
         * the TOP level until 2026-07-31, where nothing read them: every
         * geometry edit came back at HTTP 200 with its labels and its cell
         * silently gone. The receiver refuses a stray top-level key now, so the
         * same mistake cannot be quiet twice. */
        metadata: {
            regions:     groupByLabel(structure.annotations),
            cell:        clone(per.cell),
            cell_origin: clone(per.cell_origin),
            axis_kind:   clone(per.axis_kind),
            vacuum:      clone(per.vacuum),
        },
    };
}


/* ══ Load a structure in (§ 9.3) ═════════════════════════════════════════════ */

/**
 * `installMolecule` — the only way a structure gets in.
 *
 * "Everything upstream converges here — whatever built or fetched the text, it
 * arrives this way. One entrance means one place the rules are checked and one
 * place the history is anchored" (§ 9.3).
 *
 * Handed: where to put a loaded structure, how to announce a change, and a way
 * to record the first state (§ 7.3). It reaches nothing else.
 */
export function createLoad(handed) {
    return async function installMolecule(input) {
        const body = requestBodyFor(input);
        if (!body) {
            throw new TypeError(
                "installMolecule(input): pass { text, filename } or { path }");
        }
        const payload = await postJson("/api/build/load", body);
        const loaded = structureFromServer(payload);
        if (!loaded) throw said(payload,
            "The server answered, but not with a structure.");

        /* A TRAJECTORY ARRIVES WITH THE LOAD, not after it.
         *
         * The server answers with ONE geometry — it parses a file, and a file
         * has one. The frames of a run come from somewhere else: the tab's own
         * parsed run file (§ 6.3). So a caller opening a trajectory hands them
         * over HERE, and the whole structure — every frame — lands in one go.
         *
         * Doing it in a second call was three broken rules at once. § 9.3 says
         * this is "the only way a structure gets in" and it was not: the frames
         * came through another door. § 6.4 says the master copy is updated
         * "first, and COMPLETELY... No one ever observes a half-updated state",
         * and subscribers saw a one-frame structure that never existed. And
         * worst, § 11.2's point 0 was anchored on that one frame — so retracting
         * to the anchor threw the trajectory away.
         */
        if (Array.isArray(input.frames) && input.frames.length) {
            if (handed.checkFrames) {
                handed.checkFrames(input.frames, loaded.structure.elements.length);
            }
            loaded.coordinates = {
                frames: input.frames.map((f) => f.map((p) => [p[0], p[1], p[2]])),
                forcesPerFrame: Array.isArray(input.forces) ? input.forces : null,
            };
        }

        // Replace the whole model at once, then anchor a fresh history on it.
        /* THE ONLY THING KEPT ABOUT WHERE THIS CAME FROM is what an export
         * should be CALLED — "mine", no folder and no extension (§ 11.4).
         *
         * MolView tracks contents, not files. Which file is on screen is the
         * TAB's business: the tab did the file operation, the tab says "loaded
         * wire.xyz", and the tab remembers what it is showing. A viewer holding
         * a path would be a second answer to a question the tab already owns. */
        handed.put(loaded.structure, loaded.coordinates, stemOf(input),
                   /* what the read said about it (§ 6.8), from the ONE place
                    * the payload was read -- this line asked the wire for the
                    * field a second time, ten lines after `structureFromServer`
                    * had already normalised it. */
                   loaded.notices || []);
        handed.recordFirstState();
        handed.announce();
        return loaded.structure;
    };
}

/* The name a structure came in under, without its extension — what an export
 * defaults to (§ 11.4: `wire_frame50.xyz`, not `structure_frame50.xyz`). It is
 * the one thing about the SOURCE that a viewer keeps, and it is kept because
 * there is a caller: the file a user gets back should say what it came from.
 *
 * A pasted structure has no name and that is a real answer, not a missing one —
 * the export falls back to a generic stem rather than inventing provenance. */
function stemOf(input) {
    const named = (input && (input.filename || input.path)) || "";
    const base = String(named).split(/[\\/]/).pop();
    const dot = base.lastIndexOf(".");
    const stem = dot > 0 ? base.slice(0, dot) : base;
    return stem || null;
}

function requestBodyFor(input) {
    if (!input || typeof input !== "object") return null;
    // A project file is read BY THE SERVER, which owns file access and the
    // pairing with the sidecar — the browser sends a path and no text.
    if (typeof input.path === "string" && input.path) return { path: input.path };
    if (typeof input.text !== "string") return null;
    const body = { text: input.text, filename: input.filename };
    // An explicit format is for when the caller knows the text's format
    // regardless of the filename — canonical XYZ carried under a .pdb name would
    // otherwise be auto-detected as PDB and refused.
    if (input.format) body.format = input.format;
    if (typeof input.sidecar === "string") body.sidecar = input.sidecar;

    /* WHAT THE CALLER ALREADY KNEW ABOUT THESE ATOMS, when it did not come out
     * of a file. A `sidecar` is the CONTENT OF A `.molstruct.json` — an
     * untrusted document, so the server checks its envelope before applying it.
     * A run has no such document: the Results tab recovers its region labels
     * and frozen tags from the ATOM-METADATA block the Build tab wrote into the
     * input script. That is molbuilder's own emit, and the server has taken it
     * as `atom_metadata` all along, applying it through the one authority with
     * no envelope to satisfy. The browser simply never sent it, so a trajectory
     * opened with no labels — at HTTP 200, because nothing refused them; they
     * were dropped one layer above the request.
     *
     * IT TRAVELS AS BYTES AND IS NEVER OPENED HERE. It is a document the server
     * wrote, in a format the server owns, and it carries its own guard —
     * `n_atoms_total`, which is what stops a label set written for one
     * structure landing on another. Anything in the browser that parsed it and
     * put a key back would be writing that format, and re-stating the count is
     * how the guard stops being able to fire. */
    if (typeof input.atomMetadata === "string" && input.atomMetadata) {
        body.atom_metadata = input.atomMetadata;
    }

    /* THE CELL THE CALLER STATED — the same `{cell, cell_origin, axis_kind,
     * vacuum}` block every other structure door takes, so the server applies it
     * through the one seam that also CHECKS it (a refusable cell comes back as
     * a refusal, § 6.9, not a silently-accepted box).
     *
     * The caller for this is a trajectory: its lattice comes from the run's
     * output logs, which is a different source from the labels above and so a
     * different field. It went nowhere for as long as this builder ignored it,
     * which is why no trajectory has ever drawn its unit cell. */
    if (input.periodicity && typeof input.periodicity === "object") {
        body.periodicity = input.periodicity;
    }
    return body;
}


/* ══ Write the structure out (§ 9.3) ═════════════════════════════════════════ */

/**
 * `exportFile` — the exact inverse of installMolecule.
 *
 * IT RETURNS THE STRUCTURE AS DATA AND STOPS. It is a read: the atoms, their
 * positions at THE FRAME CURRENTLY DISPLAYED (§ 6.4, § 11.3 — scrub to frame 40
 * and frame 40 is what leaves), and the facts about them. It assembles no bytes,
 * because a coordinate document is a format the server owns and a second writer
 * in the browser is a second answer to "what does this structure look like on
 * disk" (§ 11.7). The two already differed.
 *
 * It stays SYNCHRONOUS. The round trip that turns this into bytes belongs to
 * whoever is putting them somewhere. Making this async would buy a new "the
 * server was unreachable" failure at the moment a user expects a file, and a gap
 * between sending and answering in which the structure could change underneath.
 *
 * It is not a disk write and not the session save.
 */
export function createWriteOut(handed) {
    /**
     * @param {object} [range]  `{from, to}` — inclusive, 0-based. Omitted, or
     *   with either end missing, it falls back to THE DISPLAYED FRAME, which is
     *   what makes § 5.1 true where a user acts: scrub to frame 40, export, and
     *   frame 40 is what leaves. Out-of-range ends are resolved against what
     *   exists rather than taken on trust (§ 6.4's rule for the frame, applied
     *   to both ends of a range), and a reversed range is read the way it was
     *   plainly meant rather than refused.
     *
     * @returns {object|null} `{name, structure}` for one frame; `{name,
     *   structure, frames}` when the range covers more. `frames` is ADDITIVE —
     *   a one-frame export is byte-for-byte the request it always was, and a
     *   consumer that does not know about ranges keeps working.
     */
    return function exportFile(range) {
        const count = handed.frameCount();
        if (!count) return null;

        const clamp = (v) => Math.min(Math.max(0, Math.floor(v)), count - 1);
        const asked = range || {};
        const first = clamp(asked.from != null ? asked.from : handed.currentFrame());
        const last  = clamp(asked.to   != null ? asked.to   : first);
        const lo = Math.min(first, last);
        const hi = Math.max(first, last);

        /* THE STRUCTURE IS THE RANGE'S FIRST FRAME, always. So the envelope is
         * the same shape at every door whatever the range, and the extra frames
         * ride BESIDE it — the same shape `installMolecule` takes on the way in
         * (§ 9.3), which is what makes one door serve both directions. */
        const structure = handed.readData(lo);
        if (!structure) return null;
        const out = {
            name:      handed.readSource ? handed.readSource() : null,
            structure: structure,
        };
        if (hi > lo) out.frames = handed.readFrames(lo, hi);
        return out;
    };
}


/* ══ The geometry edits (§ 11.1) ═════════════════════════════════════════════ */

/**
 * One small table declares each operation's shape, rather than each one being
 * hand-coded (§ 11.1). These columns drive one generic piece of code.
 *
 *   `emptySelection` — what an empty selection means for THIS operation, and the
 *                      three answers are genuinely different: act on all, refuse,
 *                      or fall back to centring on the origin.
 *   `needsExactly`   — checked BEFORE the request goes out, so `orient` with one
 *                      atom selected never reaches the network.
 *   `wholeStructure` — `calibrate` takes the whole-structure path even with a
 *                      partial selection, because it rigidly maps every atom into
 *                      the cell and clears the cell origin.
 */
export const OPERATIONS = {
    translate:             { emptySelection: "all",    needsExactly: null,
                             group: "indices" },
    rotate:                { emptySelection: "all",    needsExactly: null,
                             group: "indices" },
    orient:                { emptySelection: "refuse", needsExactly: 2,
                             group: "anchors" },
    add_atom:              { emptySelection: "refuse", needsExactly: 1,
                             group: "anchor_index", scalar: true },
    electrode:             { emptySelection: "origin", needsExactly: null,
                             group: "center_indices" },
    symmetric_electrodes:  { emptySelection: "origin", needsExactly: null,
                             group: "center_indices" },
    delete:                { emptySelection: "refuse", needsExactly: null,
                             group: "indices" },
    calibrate:             { emptySelection: "all",    needsExactly: null,
                             wholeStructure: true, group: null },
};

/**
 * `applyOp(name)` — post to the matching server route and apply the structure
 * that comes back, all at once.
 *
 * "If the edit does not come back, nothing happened" (§ 11.1). A request the
 * server refuses, or one that never arrives, leaves the structure exactly as it
 * was: nothing is half-applied, no history state is recorded, and the caller is
 * told. That is what lets a failed edit be a state the viewer can sit in without
 * being wrong, and it is why the model — not the caller — decides when a
 * structure has changed.
 *
 * Handed: read the atoms, and apply the structure the server sends back (§ 7.3).
 */
export function createEdits(handed) {
    /* ONE MUTATION IN FLIGHT. A second edit started while one is running is
     * REFUSED rather than interleaved: two responses applying over each other
     * produce a structure neither edit asked for, and the history records a
     * state the user never saw. The old registry had this rule; the rebuild
     * dropped it. */
    let running = false;

    return async function applyOp(name, params) {
        const spec = OPERATIONS[name];
        if (!spec) throw new Error("applyOp: unknown operation '" + name + "'");
        if (running) return null;

        const structure = handed.readStructure();
        if (!structure) return null;
        let selection = handed.readSelection();
        if (spec.wholeStructure) selection = [];

        // The count requirement is checked BEFORE the request goes out.
        if (spec.needsExactly != null && selection.length !== spec.needsExactly) {
            return null;
        }
        if (!selection.length && spec.emptySelection === "refuse") return null;

        const positions = handed.readFrame(handed.currentFrame());
        /* THE BODY IS FLAT. The route reads its own arguments off the body root
         * -- `dx`, `indices`, `anchors`, `element` -- so nesting them under
         * `params` sends them where nothing looks. That is what shipped: every
         * op's arguments were read by nobody, so `translate` answered 200 and
         * moved the structure by (0, 0, 0).
         *
         * `group` is § 11.1's "Where it lands" column: WHERE the resolved
         * selection goes in the body. The table did not have it once, and
         * knowing how many atoms an op needs without knowing where to put them
         * is why the rebuilt code could not build a body at all. It is OMITTED
         * when the selection is empty, so the server applies its own centring
         * rather than being handed an empty list. */
        const body = Object.assign({}, params || {}, {
            structure: structureForServer(structure, positions),
        });
        if (spec.group && selection.length) {
            body[spec.group] = spec.scalar ? selection[0] : selection.slice();
        }

        /* NO CATCH HERE. A refusal is the caller's to hear (§ 6.9): it owns the
         * button that was pressed and the place to say so. This used to swallow
         * it and answer `null`, which made "the server refused your edit" and
         * "there was nothing to do" the same answer — so the button went dead
         * with nothing on screen, which is the bug § 6.9 exists to end.
         *
         * The operation name IS the server route segment (§ 11.1): the delete
         * operation is `delete`, not `deleteAtoms`. */
        let payload;
        running = true;
        try {
            payload = await postJson("/api/modify/" + name, body);
        } finally {
            running = false;                    // even when the edit was refused
        }
        const applied = structureFromServer(payload);
        if (!applied) throw said(payload,
            "The server answered, but not with a structure.");
        /* The structure AND what the server found true of it, in one handoff.
         * The check runs on every op because it lives in the return path, not
         * in the ops -- so an op that says nothing has been checked and had
         * nothing to say, which is a different thing from not being checked. */
        handed.apply(applied.structure, applied.coordinates,
                     countChanged(structure, applied.structure),
                     applied.notices);
        return applied.structure;
    };
}

// An operation that grows or shrinks the structure clears the selection, because
// a kept selection could point at an atom that is no longer the one it meant.
// A count-preserving transform leaves it alone.
function countChanged(before, after) {
    return before.elements.length !== after.elements.length;
}


/* ══ Resolving the cell (§ 9.3) ══════════════════════════════════════════════ */

/**
 * `commitPeriodicityOp` — the ONE way the cell changes.
 *
 * § 6.2: the cell is one fact that travels together — the vectors, the anchor,
 * how each axis is treated, how much vacuum an isolated axis gets — which is why
 * there is one door and nothing writes a part of it on its own.
 *
 * MolView interprets none of it: it sends the block and stores what comes back.
 */
export function createCellEdit(handed) {
    return async function commitPeriodicityOp(op, payload) {
        /* THE SAME STRUCTURE THE EXPORT PRODUCES, under the key every other
         * door takes. A cell edit is the server deciding what the box becomes,
         * and it decides that FROM THE WHOLE STRUCTURE — the atoms it has to
         * wrap included. So it is handed the structure as data, exactly as a
         * save is, rather than a payload shaped for this one call.
         *
         * TWO WRONG SHAPES PRECEDED THIS ONE, and both failed the same silent
         * way — a 400 caught and turned into null — so the ONE door § 6.2 gives
         * the cell had never once succeeded. First `{op, params, structure}`,
         * which nothing read; then this same structure under `data`, which the
         * route rejected because it wanted a `{xyz, sidecar}` blob. The browser
         * cannot produce that blob: it writes no coordinate document (§ 11.7).
         * So the door takes the envelope, like every other one.
         */
        const structure = handed.readData();
        if (!structure) return null;
        /* NO CATCH HERE either, and this is the door it mattered most on. A cell
         * the gate cannot accept comes back 400 carrying the one sentence that
         * says what to do about it; swallowing that and answering `null` is why
         * the Update button appeared to do nothing (§ 6.9). */
        const answer = await postJson("/api/structure/periodicity", {
            structure: structure,
            op:        op,
            payload:   payload === undefined ? null : payload,
        });
        /* ADOPTED VERBATIM. The block comes back in the shape § 6.2 carries —
         * the server's own field names, with the `resolved_*` answers beside the
         * raw ones — which is the same shape `/api/build/load` sends, so there
         * is nothing to translate and nothing to pick out.
         *
         * It used to be rebuilt here from four named fields, which dropped the
         * resolved half: `getUnitCellInfo` reads `resolved_cell` first (§ 9.3's
         * "the cell as it will actually be used"), so after an edit the main way
         * in would quietly have answered with the raw value instead. */
        const block = answer && answer.periodicity;
        if (!answer || answer.ok === false || !block) {
            throw said(answer, "The server answered, but not with a cell.");
        }
        /* The answer's notices ride WITH the block it describes (§ 6.8). They
         * are passed on verbatim -- the server's own words, its own levels --
         * because rewording a warning here would put a second author on it. */
        handed.applyCell(block, Array.isArray(answer.notices) ? answer.notices : []);
        return block;
    };
}


/**
 * Ask the server which atoms match a filter rule (§ 9.5).
 *
 * "Filtering is a question asked of the server, not a scan done here. MolView
 * holds no matching logic" — the same boundary as § 2's: one place decides what
 * a structure means.
 */
export async function resolveFilter(structure, rule) {
    if (!structure) return null;
    // Evaluated against the atoms THIS VIEWER HOLDS, not a disk read — so a
    // filter sees unsaved labels and edits. Coordinates are not sent: no rule
    // matches on position (§ 9.5), and the one that looks like it does — by atom
    // index — matches on where an atom SITS, which is its place in this list.
    const atoms = structure.elements.map((element, i) => {
        const facts = structure.annotations[i] || {};
        return {
            element:     element,
            labels:      (facts.labels || []).slice(),
            residueName: facts.residue || null,
        };
    });
    /* THIS ONE STILL SWALLOWS, on purpose and not by oversight. § 6.9 governs
     * the doors that CHANGE the structure; a filter changes nothing, so nothing
     * was lost when it failed and there is no half-done state to explain. That
     * a failed filter selects nothing, silently, is a real gap of its own — but
     * widening § 6.9 to cover reads is a decision about that section, not
     * something to slip in beside this one. */
    try {
        const payload = await postJson("/api/selection/eval", { atoms, rule });
        return Array.isArray(payload && payload.selected_indices)
            ? payload.selected_indices : null;
    } catch (_) {
        return null;
    }
}


/* ══ The one way this module talks to the server ═════════════════════════════
 *
 * § 11.1 names FOUR routes, and this file calls exactly those four: load a
 * structure, perform one geometry edit, resolve a cell, resolve a filter. The
 * fourth used to be missing from that list while this module made the call —
 * "the kind of gap that lets a fifth appear unnoticed", as § 11.1 now puts it —
 * and the list was corrected rather than the call. The field-level JSON of these
 * payloads belongs to web-api.md.
 */
async function postJson(route, body) {
    let response;
    try {
        response = await fetch(route, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
        });
    } catch (_) {
        /* IT NEVER GOT THERE, so there is no sentence to quote — one of the two
         * cases § 6.9 lets MolView write a message of its own. The browser's
         * own wording for this is "Failed to fetch", which reads as a bug in the
         * page rather than as something the reader can act on. */
        throw new Error(
            "Could not reach the server. Check your connection and try again.");
    }
    if (!response.ok) {
        /* THE BODY IS WHERE THE REASON IS, and not reading it is how the reason
         * used to be lost. This threw `route + ": " + status` — a line nobody
         * could act on — while the server had answered 400 with the sentence
         * that says what to do: "swap two lattice vectors or negate one". The
         * envelope is `{ok: false, error}` at every door (web-api.md), so the
         * sentence is one read away and was never taken. */
        throw said(await bodyOf(response),
                   "The server refused the request and gave no reason.");
    }
    /* A 200 CARRYING SOMETHING THAT IS NOT JSON is rarer and lands in the same
     * place: a proxy or a cache answering in the server's stead. Letting the
     * parser's own rejection through would put "Unexpected token <" in front of
     * the reader, which is the kind of sentence § 6.9 exists to keep off the
     * screen. */
    try {
        return await response.json();
    } catch (_) {
        throw new Error("The server's answer could not be read.");
    }
}

/* The answer's own words if it has any, ours if it has none — § 6.9: "the words
 * are the server's, unchanged... MolView writes a message itself only when there
 * is none to quote". */
function said(payload, fallback) {
    const words = payload && typeof payload.error === "string"
        && payload.error.trim();
    return new Error(words || fallback);
}

/* The body of a refusal, or nothing. It is JSON in every case the server itself
 * writes, and not JSON when something upstream answered instead — a proxy, a
 * crash page. This must not throw: it runs on the failure path, and a parse
 * error raised here would replace the reason with a second, worse one. */
async function bodyOf(response) {
    try {
        return await response.json();
    } catch (_) {
        return null;
    }
}
