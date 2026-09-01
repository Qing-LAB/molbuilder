/* MolView — the three stores: what is picked out, what is being measured, and
 * how it is drawn. § 7 level 4.
 *
 * Contract: docs/web/molview.md § 9.5 (`selection`), § 9.6 (`view`),
 *           § 11.6 (`measurement`).
 * Owns:     `selection`   — what is picked, and the switches beside it.
 *           `measurement` — the ruler's own track: at most three atoms, in the
 *                           order they were clicked, and whether it is on.
 *           `view`        — style, radius, background, projection.
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
    /* `pickOrder` WAS HERE, and it is gone (2026-08-31).  A click-order shadow
     * of `selected`, kept in lock-step on a store whose whole contract is that
     * order does NOT matter — this is a set, for managing groups and labels,
     * where "these forty atoms" has no first and no second (§ 9.5).
     *
     * It existed for the angle vertex, a MEASUREMENT, and when measuring got
     * its own track it was left with one reader: the Cell page's axis gesture,
     * which needs `second − first`.  That gesture now reads the ruler, whose
     * promise is exactly order and a count limit, so the shadow has no reader
     * at all (user, 2026-08-31: "having selection and this function
     * overlapping seems functionally wrong").
     *
     * Deleting it is what makes the two tracks actually independent — and the
     * payoff is elsewhere: with nothing ordered riding on the selection, a
     * window click under isolate can be let through for MEASURING while still
     * refused for SELECTING (§ 11.6). */
    let mode = "click";          // which editor is showing — not what is selected
    let rows = [];               // the filter rows being built
    let combine = "and";
    /* WHAT THE LAST APPLY FOUND, or null when no answer is current.
     * A filter that matches nothing sets the selection to empty, and an
     * empty selection is indistinguishable from never having filtered --
     * so the panel could not say "that rule matched no atoms", which is
     * the one thing the user needs to hear. Cleared whenever the rule
     * changes, because a stale answer to a question nobody asked is
     * worse than none. */
    let lastApply = null;
    let switches = Object.assign({}, SWITCH_DEFAULTS);

    const changed = subscribable();
    const set = (next) => {
        selected = next;
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
     * is given one snapshot — WHICH ATOMS are selected (a set: § 9.5, no
     * order), which editor is showing, the rows and how they combine, and every
     * switch.
     *
     * The lesson that shaped it is worth keeping even though its subject is
     * gone: a click-order shadow was maintained correctly in this store for
     * months and simply LEFT OUT of the snapshot, so the panel read nothing,
     * fell back to guessing an angle's vertex from geometry, and § 11.6's rule
     * was dead end to end while looking implemented. **A fact the store keeps
     * but does not hand over does not exist.** The shadow itself was retired
     * on 2026-08-31 — order lives in the measurement track, which promises it.
     */
    function snapshot() {
        return {
            selection:  selected.slice(),
            mode:       mode,
            filters:    rows.map((r) => Object.assign({}, r)),
            combinator: combine,
            filterOutcome: lastApply,
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
            if (at >= 0) set(selected.filter((i) => i !== atom));
            else         set(selected.concat([atom]));
        },
        add(atoms)    { set(Array.from(new Set(selected.concat(atoms))).sort((a, b) => a - b)); },
        remove(atoms) {
            const drop = new Set(atoms);
            set(selected.filter((i) => !drop.has(i)));
        },
        all(count)    { set(Array.from({ length: count }, (_, i) => i)); },
        invert(count) {
            const has = new Set(selected);
            set(Array.from({ length: count }, (_, i) => i).filter((i) => !has.has(i)));
        },
        clear()       { set([]); },

        // A restored session's selection: intent the user expressed, so it is
        // part of what was saved (§ 11.2) and comes back with the structure.
        adopt(atoms)  { set(Array.isArray(atoms) ? atoms.slice() : []); },

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
            lastApply = null;                 // the question changed
            changed.fire(snapshot());
        },
        /* A ROW'S VALUE BELONGS TO ITS KIND, so changing the kind clears it.
         *
         * "3-7" is an atom range. It is not an element, not a residue, and not
         * a label — so carrying it across a kind change leaves the row saying
         * something its new rule cannot mean. It was carried, and the by-label
         * chooser has to show whatever the row holds (a label deleted from the
         * structure must stay visible rather than silently becoming the first
         * option), so the leftover "3-7" appeared in the list of labels as
         * though somebody had defined one.
         *
         * Cleared HERE and not in the panel: the rule is about what a filter row
         * IS, so every caller gets it — a restored session and a test drive the
         * same store, and neither should have to remember. */
        updateFilter(at, patch) {
            if (!rows[at]) return;
            const changing = patch && patch.kind && patch.kind !== rows[at].kind;
            Object.assign(rows[at], patch || {});
            // Unless the same call also said what the new value is — one change,
            // not a change followed by a correction.
            if (changing && !(patch && "value" in patch)) rows[at].value = "";
            lastApply = null;                 // the question changed
            changed.fire(snapshot());
        },
        removeFilter(at) {
            if (at < 0 || at >= rows.length) return;
            rows.splice(at, 1);
            lastApply = null;
            changed.fire(snapshot());
        },
        setCombinator(next) {
            /* THREE, and the third is the complement of the second. "None of
             * these" is NOT(a OR b) -- every atom matching no row -- which is
             * what carves the rest of a structure out of a set you can describe.
             * `not` is the server's own operator (selection.py::Not), so this
             * adds no matching logic here: § 9.5's boundary holds. */
            combine = (next === "or" || next === "nor") ? next : "and";
            lastApply = null;                 // the question changed
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
            /* WHETHER ISOLATE IS ABOUT TO SWITCH ITSELF OFF, read BEFORE `set`
             * applies the rule above (line ~127): after it, isolate is already
             * false and "it was on and turned off" is indistinguishable from
             * "it was off all along". The panel needs the difference, because a
             * switch changing without the user touching it is worth a sentence.
             *
             * SET AFTER, because `set` fires its own snapshot: recording the
             * count second would send it out attached to the OLD selection. */
            lastApply = {
                matched: atoms.length,
                isolateTurnedOff: !!switches.isolate && atoms.length === 0,
            };
            set(atoms.slice());
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
        // `atoms` is optional and defaults to the selection, which is the case
        // § 9.5 describes — the label block acts on what you picked. Naming a
        // set explicitly is the same truth change on a different set, and it is
        // what the × on a single label chip needs: remove THIS label from THIS
        // atom, without disturbing what the user has selected. One door, one
        // gate; a second door for the one-atom case would be a truth change the
        // gate has to be taught about separately (§ 9.4).
        writeLabel(name, verb, atoms) {
            return handed.writeLabel(
                name,
                Array.isArray(atoms) ? atoms.slice() : selected.slice(),
                verb || "replace");
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
    /* "NONE OF THESE" IS THE COMPLEMENT OF "ANY OF THESE" -- not of "all of
     * these". With rows `Au` and `S`, what a user wants is every atom that is
     * neither, which is NOT(Au OR S). NOT(Au AND S) would be almost every atom,
     * since no atom is both, and would look broken.
     *
     * The complement wraps whatever the rows come to, so one row negates too:
     * NOT(Au) is a perfectly good "everything except the gold". */
    const negate = combine === "nor";
    const joined = (operands.length === 1)
        ? operands[0]                               // one row is the rule itself
        : { op: (combine === "or" || negate) ? "or" : "and", operands: operands };
    return negate ? { op: "not", rule: joined } : joined;
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


/* ══ `measurement` — the ruler's own track (§ 11.6) ══════════════════════════
 *
 * A THIRD STORE, AND THE POINT IS THAT IT IS A THIRD.
 *
 * Measuring is not selecting.  The selection is what an EDIT acts on -- delete,
 * translate, centre, a label -- and every op resolves its group through it.  A
 * measurement is a question about where atoms are, asked and answered without
 * changing anything.  Sharing one list would mean that picking a third atom to
 * read an angle silently changed what the next Delete would remove, and that
 * clearing a measurement changed it back.
 *
 * So the wall is a separate object, not a flag on `selection`: this store's
 * atoms never enter `selection`'s snapshot, and the panel, the halo, the count
 * and isolate -- all readers of that one settled object (§ 8.4) -- cannot see
 * them even by accident.
 *
 * It holds `active` as well as the picks, because a store owns its whole
 * feature: the rail button that turns measuring on reads its lit state from
 * here, the same way the other five read theirs from `selection`.  A flag in
 * one store governing a list in another is the split § 5.2 exists to stop.
 *
 * WHY A CAP OF THREE, and why the fourth pick drops the OLDEST (user,
 * 2026-08-30): three atoms is every measurement there is -- a position, a
 * distance, an angle -- and a fourth pick means "now measure from here",
 * which is measuring a chain.  Refusing the fourth would make the user clear
 * and re-pick two atoms they had already chosen.
 *
 * THE ORDER IS THE USER'S CLICK ORDER and it is the answer, not a detail: the
 * vertex of an angle is the atom picked SECOND (§ 11.6).  Because this track is
 * only ever built by clicks, that order always exists -- which is what retires
 * the geometric vertex guess the readout needed when its input was a selection
 * that could arrive from All, Invert or a filter with no trail at all.
 */

//: Every measurement there is fits in three atoms (§ 11.6).
const MEASUREMENT_MAX = 3;

export function createMeasurementStore() {
    let active = false;
    let picks = [];
    const changed = subscribable();

    function snapshot() {
        return { active: active, picks: picks.slice() };
    }
    const fire = () => changed.fire(snapshot());

    return {
        /* ── Reading ─────────────────────────────────────────────────────── */
        getState() { return snapshot(); },
        get()      { return picks.slice(); },

        // Handed one on subscribing, like `selection` — the first paint needs
        // no separate fetch (§ 8.4).
        subscribe(fn) {
            const off = changed.add(fn);
            try { fn(snapshot()); } catch (_) {}
            return off;
        },

        /* ── The toggle beside the other rail switches (§ 8.5) ───────────── */
        /* TURNING MEASURING OFF CLEARS THE PICKS (user, 2026-08-31).
         *
         * The toggle IS the measurement session, so leaving it ends the
         * session.  It kept them until then, on the reasoning that coming back
         * to a half-finished measurement costs nothing -- but the picks are
         * not free while they sit there:
         *
         *   - the Cell page's pick buttons read the COUNT, so they stayed
         *     enabled with the ruler off and a title saying "turn measuring
         *     on", and pressing one staged a row from picks no longer marked
         *     on the molecule;
         *   - nothing on screen shows them once the ruler is off, so the state
         *     that decides what those buttons do is invisible;
         *   - and it made the mode ambiguous: off-with-picks and off-without
         *     behaved differently and looked identical.
         *
         * Off means nothing is being measured.  One state, not two. */
        setActive(on) {
            const next = !!on;
            if (active === next) return;
            active = next;
            if (!next) picks = [];      // one `fire` below, not two
            fire();
        },

        /* ── The picks (§ 11.6) ──────────────────────────────────────────── */
        //
        // Toggle, so clicking a picked atom takes it back out — the same verb
        // the selection uses, because it is the same gesture.
        toggle(atom) {
            const at = picks.indexOf(atom);
            if (at >= 0) {
                picks.splice(at, 1);
            } else {
                picks.push(atom);
                // The fourth pick drops the OLDEST, so measuring along a chain
                // stays one click per step.
                while (picks.length > MEASUREMENT_MAX) picks.shift();
            }
            fire();
        },
        clear() {
            if (!picks.length) return;
            picks = [];
            fire();
        },

        /* NO `adopt`.  There was one, for restoring a session's picks from the
         * view-context lane -- and it was the bug: the lane's only guard was an
         * ATOM COUNT, which two different three-atom molecules pass, so the
         * readout came back quoting a bond length for atoms nobody had picked.
         * The picks do not persist at all now (§ 11.6), so there is nothing to
         * adopt them from, and leaving the door standing would be an invitation
         * to write that restore again. */
    };
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
