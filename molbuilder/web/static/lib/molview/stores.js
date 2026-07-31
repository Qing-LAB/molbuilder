/* MolView — the two stores: what is picked out, and how it is drawn. § 7 level 4.
 *
 * Contract: docs/web/molview.md § 9.5 (`selection`) and § 9.6 (`view`).
 * Owns:     `selection` — what is picked, and the switches beside it.
 *           `view`      — style, radius, background, projection.
 * Called by: assembled by the model and reached ONLY through it (§ 9.3), so a
 *           change asked for through a store meets the same rules as one asked
 *           for anywhere else (§ 9.4).
 * Shape:    change-and-subscribe. "They exist so state has a home that knows
 *           nothing about drawing."
 *
 * NEVER (§ 7 level 4):
 *   - draw anything;
 *   - hold the displayed frame — that is not a switch (§ 6.4);
 *   - be kept by anything outside the viewer once it has been reached.
 *
 * AND NEVER (§ 9.6): hold the camera. Not here, not in the model, not in the
 * handle — it is held nowhere above the drawing.
 */
"use strict";

import { expressionToCode } from "./_atom.js";


/* ══ Which store does a thing belong in? (§ 9.6) ═════════════════════════════
 *
 * The test is one question: DOES WORKING OUT WHAT A FRAME CONTAINS REQUIRE
 * READING IT?
 *
 *   yes -> `selection`. It changes WHAT IS IN a frame — which atoms, and what is
 *          drawn beside them. If one changed and nothing was recomputed, the
 *          picture would be WRONG.
 *   no  -> `view`. It changes HOW THE SAME FRAME IS PAINTED. If one changed and
 *          nothing was recomputed, the picture would be CORRECT, painted
 *          differently.
 *
 * That line is checkable rather than a convention to remember: a switch the
 * frame calculation has to read belongs to `selection`; a setting the sealed
 * layer applies without that calculation ever seeing it belongs to `view`.
 *
 * The camera is in neither column, because it is in neither place.
 */

// Every one off by default, and the arrow scale at its default (§ 9.5).
const SWITCH_DEFAULTS = {
    isolate:     false,
    showIndex:   false,
    showForces:  false,
    showCell:    false,
    showAxis:    false,
    forceScale:  1,
};

/* `background: null` is not "no background" — it means THE 3D WINDOW'S OWN
 * GROUND, which the stylesheet declares and the drawing layer resolves. The
 * store carries what the USER chose, and until they choose one there is nothing
 * to carry; naming a colour here would put a second copy of the module's
 * surface above the one place that decides it (§ 5.2). */
const VIEW_DEFAULTS = {
    style:        "stick",
    radius:       1,
    background:   null,
    orthographic: false,
};


function subscribable() {
    const listeners = [];
    return {
        add(fn) {
            listeners.push(fn);
            return () => {
                const at = listeners.indexOf(fn);
                if (at >= 0) listeners.splice(at, 1);
            };
        },
        fire(...args) {
            for (const fn of listeners.slice()) {
                try { fn(...args); } catch (_) {}
            }
        },
    };
}


/* ══ `selection` — what is picked out, and what is drawn beside it (§ 9.5) ════
 *
 * "The panel, the highlight and the measurements are all READERS of it; none of
 * them keeps its own answer."
 *
 * @param handed  what the model allows this store to call (§ 7.3):
 *                `resolveFilter(rule)` — ask the server which atoms match
 *                `writeLabel(name, atoms)` — a change to the STRUCTURE, so it
 *                goes back through the model where the gate can see it (§ 9.4)
 */
export function createSelectionStore(handed) {
    handed = handed || {};

    // THE SELECTION IS THE TRUTH; click and filter are two EDITORS of it.
    // Switching between them does not touch what is selected.
    let selected = [];
    let pickOrder = [];          // the order atoms were picked, for measurement
    let mode = "click";          // which editor is showing — not what is selected
    let rows = [];               // the filter rows being built
    let combine = "and";
    let switches = Object.assign({}, SWITCH_DEFAULTS);

    const changed = subscribable();
    const set = (next, order) => {
        selected = next;
        pickOrder = order != null ? order : next.slice();
        /* ISOLATE TURNS ITSELF OFF WHEN THE SELECTION EMPTIES (§ 1.1) — "since
         * there would be nothing left to show". It is a SELECTION-STATE RULE,
         * so it lives here beside the fact it depends on rather than in the
         * control that happened to empty the selection; Clear, Remove, a filter
         * that matches nothing and a restored empty session all go through this
         * one line. It is settled BEFORE the snapshot fires, so no reader ever
         * sees isolate on with nothing selected. */
        if (switches.isolate && !selected.length) switches.isolate = false;
        changed.fire(snapshot());
    };

    /* ONE SETTLED STATE, HANDED OVER WHOLE (§ 8.4).
     *
     * The panel does not assemble what it draws from a dozen separate reads. It
     * is given one snapshot — what is selected and IN WHAT ORDER, which editor
     * is showing, the rows and how they combine, and every switch.
     *
     * This is the fix for a real failure, not a style preference: the pick order
     * was maintained correctly in the store for months and simply LEFT OUT of
     * the snapshot, so the panel read nothing, fell back to guessing an angle's
     * vertex from geometry, and § 11.6's rule was dead end to end while looking
     * implemented. A fact the store keeps but does not hand over does not exist.
     */
    function snapshot() {
        return {
            selection:  selected.slice(),
            pickOrder:  pickOrder.slice(),
            mode:       mode,
            filters:    rows.map((r) => Object.assign({}, r)),
            combinator: combine,
            isolate:    switches.isolate,
            showIndex:  switches.showIndex,
            showForces: switches.showForces,
            showCell:   switches.showCell,
            showAxis:   switches.showAxis,
            forceScale: switches.forceScale,
        };
    }

    return {
        /* ── Reading ─────────────────────────────────────────────────────── */
        getState()     { return snapshot(); },
        get()          { return selected.slice(); },
        switches()     { return Object.assign({}, switches); },

        // Handed one on subscribing, so the first paint needs no separate fetch
        // (§ 8.4) — and another after every change.
        subscribe(fn)  {
            const off = changed.add(fn);
            try { fn(snapshot()); } catch (_) {}
            return off;
        },

        /* ── The click operations (§ 9.5) ────────────────────────────────── */
        toggle(atom) {
            const at = selected.indexOf(atom);
            if (at >= 0) {
                set(selected.filter((i) => i !== atom),
                    pickOrder.filter((i) => i !== atom));
            } else {
                set(selected.concat([atom]), pickOrder.concat([atom]));
            }
        },
        /* THE BULK OPERATIONS CARRY NO PICK TRAIL, and say so by handing over an
         * empty one. `pickOrder` is the order atoms were CLICKED — a chemist's
         * "from → to", and the reason the vertex of an angle is the atom picked
         * second (§ 11.6). All, Invert, an applied filter and a restored session
         * are not clicks; there is no such order to report.
         *
         * Reporting the sorted selection instead, which is what these did, is
         * § 8.4's failure from the other side: the readout cannot tell a real
         * trail from a fabricated one, so it treats "the middle atom by number"
         * as the vertex the user chose. It looks implemented and is wrong. */
        add(atoms)    { set(Array.from(new Set(selected.concat(atoms))).sort((a, b) => a - b), []); },
        remove(atoms) {
            const drop = new Set(atoms);
            set(selected.filter((i) => !drop.has(i)),
                pickOrder.filter((i) => !drop.has(i)));
        },
        all(count)    { set(Array.from({ length: count }, (_, i) => i), []); },
        invert(count) {
            const has = new Set(selected);
            set(Array.from({ length: count }, (_, i) => i).filter((i) => !has.has(i)), []);
        },
        clear()       { set([]); },

        // A restored session's selection: intent the user expressed, so it is
        // part of what was saved (§ 11.2) and comes back with the structure.
        adopt(atoms)  { set(Array.isArray(atoms) ? atoms.slice() : [], []); },

        /* ── The switches (§ 9.5) ────────────────────────────────────────── */
        //
        // These live HERE — "not in the renderEngine and not in the panel" — so
        // a switch has one home and every reader of it agrees.
        setSwitch(name, value) {
            if (!(name in SWITCH_DEFAULTS)) return;
            if (switches[name] === value) return;
            switches[name] = value;
            changed.fire(snapshot());
        },
        // Isolate is the one switch with a control of its own ("Show selected
        // only"), so it has a name at this surface. It is the SAME switch — one
        // home, reached two ways, never two values.
        setIsolate(on) { this.setSwitch("isolate", !!on); },

        /* ── The filter, edited a row at a time (§ 8.4) ───────────────────── */
        //
        // A user adds a row, types in it, changes its kind, removes it, and
        // chooses how the rows combine — each its own small change, because that
        // is what the controls are. A surface that only took the whole set at
        // once would make the panel re-send rows it was in the middle of editing.
        setEditor(next) {
            if (next !== "click" && next !== "filter") return;
            mode = next;
            changed.fire(snapshot());   // the panel redraws; the selection does not move
        },
        addFilter(row) {
            rows.push(Object.assign({ kind: "by_element", value: "" }, row || {}));
            changed.fire(snapshot());
        },
        updateFilter(at, patch) {
            if (!rows[at]) return;
            Object.assign(rows[at], patch || {});
            changed.fire(snapshot());
        },
        removeFilter(at) {
            if (at < 0 || at >= rows.length) return;
            rows.splice(at, 1);
            changed.fire(snapshot());
        },
        setCombinator(next) {
            combine = next === "or" ? "or" : "and";
            changed.fire(snapshot());
        },

        /**
         * Apply the rows as one rule.
         *
         * "Filtering is a question asked of the server, not a scan done here.
         * MolView holds no matching logic" (§ 9.5) — the same boundary as § 2's:
         * one place decides what a structure means.
         */
        async applyFilter() {
            const rule = buildRule(rows, combine);
            if (!rule) return selected.slice();       // no rows means no filter at all
            const atoms = await handed.resolveFilter(rule);
            if (!Array.isArray(atoms)) return selected.slice();
            set(atoms.slice(), []);
            return atoms.slice();
        },

        /* ── Writing a label (§ 9.5, § 9.4) ──────────────────────────────── */
        //
        // "The one thing reached from here that is not like the others." It is a
        // change to the STRUCTURE, so it goes back through the model where the
        // gate can see it. A change the gate cannot see is a change the gate
        // does not stop. Applying a label REPLACES its previous set of atoms.
        // The verb is WHICH SET OPERATION this is — replace, add or remove. It
        // travels with the call because all three are the same truth change on
        // the same atoms, and splitting them into three doors would give the
        // gate three things to stand in front of instead of one (§ 9.4).
        writeLabel(name, verb) {
            return handed.writeLabel(name, selected.slice(), verb || "replace");
        },
    };
}

/**
 * The rows, as one rule the server can evaluate.
 *
 * THE RULE VOCABULARY IS THE SERVER'S, NOT THIS MODULE'S. § 9.5 describes four
 * kinds of row in prose and § 11.1 says the field-level JSON belongs to
 * web-api.md — so the names below are `molbuilder/selection.py`'s own
 * (`by_element`, `by_index_range`, `by_residue_name`, `by_region`, composed with
 * `and`/`or` over `operands`). Inventing a shape here would give a panel that
 * builds rules nothing can evaluate.
 *
 * A HALF-TYPED ROW CONSTRAINS NOTHING — it does not match nothing. An empty row
 * is dropped BEFORE the rule is built, so a row the user has not finished
 * filling in cannot silently empty the result under `and`. "You have not told me
 * anything to intersect with yet" is the correct reading of a blank row, and
 * treating it as "match nothing" would make the panel feel broken mid-typing.
 */
export function buildRule(rows, combine) {
    const operands = (rows || []).map(rowToRule).filter((r) => r !== null);
    if (!operands.length) return null;              // no rows means no filter
    if (operands.length === 1) return operands[0];  // one row is the rule itself
    return { op: combine === "or" ? "or" : "and", operands: operands };
}

function rowToRule(row) {
    if (!row || !row.kind || typeof row.value !== "string") return null;
    const raw = row.value.trim();
    if (raw === "") return null;                    // the half-typed row
    const list = () => raw.split(",").map((s) => s.trim()).filter(Boolean);

    switch (row.kind) {
        case "by_element": {
            const elements = list();
            return elements.length ? { op: "by_element", elements: elements } : null;
        }
        case "by_residue": {
            const names = list();
            return names.length ? { op: "by_residue_name", names: names } : null;
        }
        case "by_label":
            return { op: "by_region", name: raw };
        case "by_index":
            // THE ONE ROW THAT CROSSES THE NUMBERING BOUNDARY (§ 9.5). The user
            // types 1-based, matching what is on screen; the rule sent is
            // 0-based; and the shift happens exactly once, here, at the point
            // the row becomes a rule — through the one translation that owns it
            // (§ 11.5). No caller writes the shift itself.
            //
            // Every other row above compares names to names and never touches a
            // number, which is why this is the only case that calls it.
            return { op: "by_index_range", expression: expressionToCode(raw) };
        default:
            return null;
    }
}


/* ══ `view` — how the molecule is drawn (§ 9.6) ══════════════════════════════
 *
 * Four settings the user chose, written here by whatever control they touched,
 * wherever that control happens to sit (§ 11.4). "Nothing has to be read back to
 * know which style is active: the answer is whatever was last set."
 *
 * THE CAMERA IS NOT HERE, and that is the whole point of § 9.6. It is the one
 * thing a user changes without telling MolView — a drag rotates it directly in
 * the window — and MolView never records where it ended up, never reads it back
 * and never saves it. On load and on Reset it is FITTED TO THE STRUCTURE, the
 * only orientation guaranteed to show the molecule.
 *
 * What that costs is one sentence. What it buys is the removal of an entire
 * mechanism: nothing ever asks the sealed layer a question, there is no separate
 * trigger for saving a view-only change, and no persisted slot that has to be
 * patched independently of the structure it belongs to.
 */
export function createViewStore() {
    let settings = Object.assign({}, VIEW_DEFAULTS);
    const changed = subscribable();

    return {
        get()         { return Object.assign({}, settings); },
        subscribe(fn) { return changed.add(fn); },

        set(name, value) {
            if (!(name in VIEW_DEFAULTS)) return;
            if (settings[name] === value) return;
            settings[name] = value;
            changed.fire(this.get());
        },

        // Reopening a session puts the drawing back at its defaults, because
        // none of it was ever part of what you were working on (§ 11.2).
        reset() {
            settings = Object.assign({}, VIEW_DEFAULTS);
            changed.fire(this.get());
        },
    };
}
