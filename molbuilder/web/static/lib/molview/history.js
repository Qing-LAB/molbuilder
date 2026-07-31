/* MolView — session history: the sequence of states, the position on it, and the
 * write machine.
 *
 * Contract: docs/web/molview.md § 11.2, § 7.3.
 * Owns:     the sequence and the position on it; point 0; the write machine
 *           (SETTLED / CHANGING / WRITING); the badge that says where you are.
 * Called by: model.js, and nothing else.
 *
 * NEVER (§ 11.2): know or care what is in it. "It is handed a way to make a
 * state and a way to put one back, and it never looks inside. So NOTHING ABOUT
 * SAVING CONSTRAINS WHAT MAY BE SAVED: a trajectory needs no new mechanism to
 * become restorable, and neither does anything added to the truth later. Only
 * the thing that writes the state has to include it."
 *
 * Which is why this file lists no exclusions. Nothing is left out by the saving
 * machinery. Things are left out because they are not the truth (§ 11.2's
 * table), and that is one rule rather than a list to maintain.
 *
 * WHY THIS IS ITS OWN FILE and not part of model-jobs.js: the claim above only
 * holds while it is not sitting beside the serialiser it is handed.
 *
 * MolView owns the whole mechanism AND the policy — what a save records, what to
 * prune, how far back a step goes, and the rule that nothing is recorded on its
 * own. The workspace owns only what sits underneath: where the bytes actually
 * go, reached through an accessor handed in at mount. That is the entire
 * division.
 */
"use strict";


/* The three states a write can be asked for in (§ 11.2). There is no fourth and
 * no other reason a write is delayed. */
export const SETTLED  = "settled";    // the structure is consistent
export const CHANGING = "changing";   // a multi-step change is under way
export const WRITING  = "writing";    // a write is on its way to storage


/**
 * @param handed  what the model allows this helper to call (§ 7.3):
 *                `recordState()`  — make a state out of the current structure
 *                `restoreState(s)`— put a state back
 *                `store`          — where the bytes go: {read(step), write(step, bytes)}
 *                `onBadge(flag)`  — the unsaved-changes badge changed
 *
 * Note what is NOT handed: anything that would let this look inside a state.
 */
export function createHistory(handed) {
    // 0 is the state the structure opened at.
    let position = 0;
    let highest = 0;               // the furthest point that still exists
    let uncommitted = false;       // is there work that is not on the sequence?
    let anchored = false;

    let state = SETTLED;
    let held = null;               // at most one, remembered during CHANGING
    let depth = 0;                 // nested brackets

    function setBadge(next) {
        if (uncommitted === next) return;
        uncommitted = next;
        if (handed.onBadge) handed.onBadge(uncommitted);
    }

    /* ── The write machine (§ 11.2) ─────────────────────────────────────────
     *
     * WHY CHANGING EXISTS AT ALL. Opening a structure arrives in two steps: the
     * new coordinates first, the labels for those atoms a moment later. Between
     * the two, the viewer is holding THE NEW POSITIONS WITH THE PREVIOUS FILE'S
     * LABELS — a structure that never existed.
     *
     * The example from the contract: a viewer shows `wire.xyz`, whose first
     * twenty atoms are labelled `L-electrode`. The user opens `slab.xyz` over
     * it; both have sixty atoms. A write landing in that gap saves the slab's
     * positions carrying the wire's labels — twenty slab atoms marked as an
     * electrode. THE ATOM COUNT MATCHES, SO NOTHING COMPLAINS, and the next
     * calculation generated from that file puts an electrode where the user
     * never put one.
     *
     * That is the whole reason writes are held rather than sent.
     */
    async function send(step, isSave) {
        state = WRITING;
        let landed = false;
        try {
            // THE STATE IS RECORDED HERE, when the write actually goes out —
            // never when it was asked for. § 11.2: "what lands is the settled
            // state". A write held through a bracket that captured its bytes at
            // request time would land the halfway structure the bracket exists
            // to keep out of storage.
            await handed.store.write(step, handed.recordState());
            landed = true;
        } catch (_) {
            landed = false;
        }
        // "it landed: the position moves" / "it failed: the position does not
        // move". Two writes in flight is how the position comes to describe a
        // state that was never written, which is why WRITING queues.
        if (landed && isSave) {
            position = step;
            highest = step;                  // a save drops every point above it
            setBadge(false);
        }
        state = SETTLED;
        // A write asked for while this one was on its way waits its turn.
        if (held) {
            const next = held;
            held = null;
            await send(next.step, next.isSave);
        }
        return landed;
    }

    // What is remembered is the INTENT — which step, and whether it is a save —
    // not the bytes. See send().
    async function request(step, isSave) {
        if (state === CHANGING || state === WRITING) {
            // At most one is remembered; if a SAVED STATE is among them, that is
            // the one sent, and a routine write arriving after it does not
            // replace it.
            if (!held || (isSave && !held.isSave)) held = { step, isSave };
            return false;
        }
        return send(step, isSave);
    }

    return {
        /* ── Where you are ──────────────────────────────────────────────── */
        get state_index() { return position; },
        // The badge is not bookkeeping: it shows in the corner of the 3D window,
        // so "there is work here that is not on the sequence yet" is visible
        // without opening a menu. Without it, an explicit-save history would
        // silently lose work a user assumed was being kept.
        get uncommitted() { return uncommitted; },
        get writeState()  { return state; },

        /* ── Point 0 ────────────────────────────────────────────────────── */
        //
        // "The one point nobody asks for", laid down when a structure is opened.
        // NOT a save — it is the floor the sequence stands on, so that a Retract
        // from the first edit has somewhere to land.
        //
        // Opening a new structure also CLEARS the machine: anything remembered,
        // and anything still on its way, belongs to the structure that was just
        // replaced. It is dropped rather than applied — applying it would put an
        // old state over a freshly opened structure. Same rule as § 10.9's, one
        // subsystem over: the more authoritative statement about what the
        // structure is beats whatever is in flight.
        anchor() {
            held = null;
            position = 0;
            highest = 0;
            anchored = true;
            setBadge(false);
            return request(0, true);
        },

        /* ── An edit happened ───────────────────────────────────────────── */
        //
        // "An edit — a delete, a rotate, a new electrode — changes the structure
        // and does NOT record a state; the user decides when the structure is
        // worth being able to come back to, and says so."
        //
        // Nothing is written on a timer and nothing is written because something
        // changed. Storage is touched by exactly three things: opening a
        // structure, an explicit save, and a load.
        edited() {
            if (anchored) setBadge(true);
        },

        /* ── save(step) (§ 11.2's table) ────────────────────────────────── */
        //
        //   save(1) — write a new point one step on, and DROP EVERY POINT ABOVE
        //             IT. This is what the user calls "Save state".
        //   save(0) — re-write the current point where it is, without moving.
        //
        // "Stepping forward lasts until you save. The moment a user commits to a
        // different path, the abandoned one stops existing."
        // A save asked for during CHANGING is HELD like any other write, and
        // § 11.2's table is what says so: "if a saved state is among them, that
        // is the one sent" only means anything if a save can be among them.
        //
        // "A save is never asked for during CHANGING" is a rule for CALLERS, not
        // a refusal by the machine — it moves your position on the sequence and
        // the write has to land together with the move, so anything wanting both
        // waits. The machine still holds it if asked, because dropping it would
        // lose the user's work exactly when they asked to keep it.
        save(step) {
            const target = position + (step === 0 ? 0 : 1);
            return request(target, true);
        },

        /* ── load(step) (§ 11.2's table) ────────────────────────────────── */
        //
        //   load(-1) — step back one point. `undo` is exactly this. "Retract".
        //   load(+1) — step forward again, into a point a Retract moved away from.
        //   load(0)  — NOT A MOVE: put back the point you were on. Zero is a
        //              different verb, and it is what a session restore needs.
        //
        // A RETRACT SPENDS UNSAVED WORK FIRST. From a saved point with edits
        // sitting on top of it, the first Retract discards the edits and leaves
        // you ON that point; only the next one steps to the point before it. The
        // first press undoes what you just did, not what you had already decided
        // to keep.
        async load(step) {
            if (!anchored) return null;
            let target;
            if (step === 0) {
                target = position;                 // restore where I was
            } else if (step < 0 && uncommitted) {
                target = position;                 // spend the unsaved work first
            } else {
                target = position + step;
            }
            if (target < 0 || target > highest) return null;

            const bytes = await handed.store.read(target);
            if (bytes == null) return null;
            handed.restoreState(bytes);
            position = target;
            setBadge(false);
            return target;
        },

        undo() { return this.load(-1); },

        /* ── The bracket (§ 11.2) ───────────────────────────────────────── */
        //
        // "A bracketed change writes once, at the end, and what lands is the
        // settled state." Nested brackets are counted, so a caller that wraps
        // another's work does not release it early.
        beginChange() {
            depth += 1;
            if (state === SETTLED) state = CHANGING;
        },
        async endChange() {
            if (depth > 0) depth -= 1;
            if (depth > 0) return false;
            if (state !== CHANGING) return false;
            state = SETTLED;
            if (!held) return false;
            const next = held;
            held = null;
            return send(next.step, next.isSave);
        },
    };
}
