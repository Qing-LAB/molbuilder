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

/* THE ONE RESERVED-LABEL ALIAS, and the only place a reserved name is written.
 *
 * § 6.2 says an atom's facts are its labels and its residue — there is no frozen
 * field — and § 6.6 says MolView's end of a reserved label is "one mechanism, no
 * special case". The server has not caught up: it sends `regions` (the labels)
 * and a separate `is_frozen` flag, deliberately keeping them apart so the panel
 * does not render an atom's frozen state twice.
 *
 * So the flag becomes a label here, at the boundary, and downstream there is
 * exactly one mechanism. § 6.6 calls this "a translator at the point of use" and
 * names the whole arrangement — a separate field plus an alias between two
 * spellings — as what folding frozen onto the other four reserved labels will
 * remove. That fold belongs to model/structure-annotations.md, not here.
 *
 * The NAME matters and is not free to choose: this is the name MolView will
 * OFFER as a filter row, and the server's `by label` rule resolves the frozen
 * set under `frozen_atoms`. Offering a name the server cannot match would give a
 * row that always answers nothing.
 */
export const FROZEN_LABEL = "frozen_atoms";

/**
 * The server's structure payload, in the shapes of § 6.2.
 *
 * Returns `{structure, coordinates}` — or null, because with nothing loaded a
 * read must return NOTHING rather than an empty structure (§ 9.3): "there is
 * nothing here" and "here is a structure with no atoms" are different answers.
 */
export function structureFromServer(payload) {
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
        const labels = Array.isArray(a.regions) ? a.regions.slice() : [];
        if (a.is_frozen && labels.indexOf(FROZEN_LABEL) < 0) labels.push(FROZEN_LABEL);
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
            // The cell block is CARRIED, not interpreted (§ 6.2). Its field names
            // and the rules for resolving them belong to
            // model/structure-periodicity.md.
            cell: payload.periodicity || null,
        },
        coordinates: {
            frames: [atoms.map((a) => [Number(a.x) || 0,
                                       Number(a.y) || 0,
                                       Number(a.z) || 0])],
            forcesPerFrame: null,
        },
    };
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
    if (!structure) return null;
    const labels = {};
    const frozen = [];
    structure.annotations.forEach((facts, i) => {
        for (const name of (facts.labels || [])) {
            if (name === FROZEN_LABEL) { frozen.push(i); continue; }
            (labels[name] = labels[name] || []).push(i);
        }
    });
    return {
        elements:     structure.elements.slice(),
        positions:    positions.map((p) => [p[0], p[1], p[2]]),
        regions:      labels,
        frozen_atoms: frozen,
        periodicity:  structure.cell,
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
        if (!loaded) return null;
        // Replace the whole model at once, then anchor a fresh history on it.
        handed.put(loaded.structure, loaded.coordinates, stemOf(input));
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
    return body;
}


/* ══ Write the structure out (§ 9.3) ═════════════════════════════════════════ */

/**
 * `exportFile` — the exact inverse of installMolecule.
 *
 * Handed read-only access to the atoms, the cell and the displayed frame
 * (§ 7.3). Writes from THE FRAME CURRENTLY DISPLAYED (§ 6.4, § 11.3): scrub to
 * frame 40 and frame 40 is what the file holds, whatever isolate is doing.
 *
 * It is not a disk write and not the session save.
 */
export function createWriteOut(handed) {
    return function exportFile() {
        const structure = handed.readStructure();
        if (!structure) return null;
        const positions = handed.readFrame(handed.currentFrame());
        if (!positions) return null;
        const source = handed.readSource ? handed.readSource() : null;

        // "It REFUSES to produce anything when the geometry and the per-atom
        // labels disagree about how many atoms there are, returning nothing
        // rather than writing a corrupt structure" (§ 9.3).
        if (positions.length !== structure.elements.length
            || structure.annotations.length !== structure.elements.length) {
            return null;
        }

        const lines = [String(structure.elements.length), ""];
        for (let i = 0; i < structure.elements.length; i++) {
            const p = positions[i];
            lines.push(structure.elements[i] + " " + p[0] + " " + p[1] + " " + p[2]);
        }
        return {
            name:    source,
            text:    lines.join("\n") + "\n",
            sidecar: sidecarFor(structure, positions.length),
        };
    };
}

/**
 * The metadata that travels beside the geometry (§ 11.3) — the sidecar's
 * FIELDS, in the shape the rest of the application already speaks.
 *
 * WHAT THIS IS NOT: the request payload. This used to hand back
 * `structureForServer()` — elements, positions and a `periodicity` block — which
 * is what a server ROUTE wants and is not a sidecar at all. Written to disk it
 * pairs a good `.xyz` with a `.json` the codec cannot read, and the labels it
 * was carrying are lost at the next open. The two shapes look similar enough
 * that nothing complained.
 *
 * WHO FINISHES IT: the server. `model/structure-molstruct.md` § 1's envelope —
 * `schema_version`, `n_atoms_total` re-checked, and the `structure_hash` that
 * pins the pair — is stamped by `StructureCodec.write` when the bytes are
 * written, deliberately: a browser-authored envelope once shipped without
 * `schema_version` and the load door refused the pair on the next open. So this
 * carries the FACTS and no bookkeeping.
 *
 * The cell is spread into the sidecar's own field names — `cell`, `cell_origin`,
 * `axis_kind`, `vacuum` — which is a rename and not an interpretation: MolView
 * still reads none of it (§ 6.2).
 */
export function sidecarFor(structure, atomCount) {
    const labels = {};
    const frozen = [];
    structure.annotations.forEach((facts, i) => {
        for (const name of (facts.labels || [])) {
            if (name === FROZEN_LABEL) { frozen.push(i); continue; }
            (labels[name] = labels[name] || []).push(i);
        }
    });
    const cell = structure.cell || {};
    return {
        n_atoms_total: atomCount,
        regions:       labels,
        frozen_atoms:  frozen,
        cell:          cell.lattice || null,
        cell_origin:   cell.origin || null,
        axis_kind:     cell.axis_kind || null,
        vacuum:        cell.vacuum || null,
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
    translate:             { emptySelection: "all",    needsExactly: null },
    rotate:                { emptySelection: "all",    needsExactly: null },
    orient:                { emptySelection: "refuse", needsExactly: 2 },
    add_atom:              { emptySelection: "refuse", needsExactly: 1 },
    electrode:             { emptySelection: "origin", needsExactly: null },
    symmetric_electrodes:  { emptySelection: "origin", needsExactly: null },
    delete:                { emptySelection: "refuse", needsExactly: null },
    calibrate:             { emptySelection: "all",    needsExactly: null,
                             wholeStructure: true },
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
    return async function applyOp(name, params) {
        const spec = OPERATIONS[name];
        if (!spec) throw new Error("applyOp: unknown operation '" + name + "'");

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
        const body = {
            structure: structureForServer(structure, positions),
            selection: selection.slice(),
            params:    params || {},
        };

        // The operation name IS the server route segment (§ 11.1): the delete
        // operation is `delete`, not `deleteAtoms`.
        let payload;
        try {
            payload = await postJson("/api/modify/" + name, body);
        } catch (_) {
            return null;                        // nothing came back, nothing happened
        }
        const applied = structureFromServer(payload);
        if (!applied) return null;
        handed.apply(applied.structure, applied.coordinates,
                     countChanged(structure, applied.structure));
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
    return async function commitPeriodicityOp(op, params) {
        const structure = handed.readStructure();
        if (!structure) return null;
        const positions = handed.readFrame(handed.currentFrame());
        let payload;
        try {
            payload = await postJson("/api/structure/periodicity", {
                op:        op,
                params:    params || {},
                structure: structureForServer(structure, positions),
            });
        } catch (_) {
            return null;
        }
        if (!payload || !payload.periodicity) return null;
        handed.applyCell(payload.periodicity);
        return payload.periodicity;
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
 * § 11.1 names THREE routes: load a structure, perform one geometry edit,
 * resolve a cell. The filter of § 9.5 is a fourth — "a question asked of the
 * server" — which that sentence omits; the plan records it as an open item, and
 * the route test counts what is actually here rather than what the sentence
 * says. The field-level JSON of these payloads belongs to web-api.md.
 */
async function postJson(route, body) {
    const response = await fetch(route, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(body),
    });
    if (!response.ok) throw new Error(route + ": " + response.status);
    return response.json();
}
