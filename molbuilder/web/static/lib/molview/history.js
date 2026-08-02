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
 *                `workspace`      — the workspace's front door (workspace.md § 5):
 *                                   `persist` · `readState` · `pruneStatesAbove` ·
 *                                   `workspaceId`. Handed in at mount; MolView
 *                                   never learns where the bytes end up, only
 *                                   which calls to make.
 *                `tag`            — whose bytes these are (workspace.md § 4). The
 *                                   viewer's `owner`, so two viewers on one page
 *                                   never write over each other.
 *                `onBadge(flag)`  — the unsaved-changes badge changed
 *
 * Note what is NOT handed: anything that would let this look inside a state.
 *
 * IT CALLS THE WORKSPACE BY NAME, on purpose. The rebuild had it asking for a
 * two-call door of its own invention — `read(step)` / `write(step, bytes)` — which
 * nothing implemented, so nothing was ever saved and every stub in every test
 * satisfied it perfectly. A door shaped like the real dependency fails the moment
 * the two disagree; a door shaped like a wish does not.
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
            const bytes = handed.recordState();
            /* ONE CALL, BOTH COPIES (§ 11.3). The session copy goes to the browser
             * straight away so a reload finds it; the numbered point is sent to
             * the server without waiting, so a slow disk never stalls an edit.
             *
             * A ROUTINE WRITE SENDS NO POINT. `snapshotBlob` is null unless this is
             * a save, which is what keeps § 11.2's two rules from collapsing into
             * one: every change refreshes what would be lost, and only the user's
             * own Save state lays down somewhere to come back to. */
            /* A MILESTONE AND A DRAFT GO TO DIFFERENT FILES.
             *
             * `save` writes a numbered point you can come back to. An edit writes
             * the draft — what you would lose if the tab closed — and it must not
             * land on a numbered point, or every edit after a save would quietly
             * rewrite the thing Retract is supposed to return you to.
             *
             * They are two file names, not two stores: both are state files in
             * the same directory on the server. The draft is one file that keeps
             * being replaced, so there is exactly one of it.
             *
             * VERSION-STAMPED, because these files outlive the code that wrote
             * them. Someone upgrades molbuilder, opens a tab, and the reader is
             * new while the file on disk is old; without a stamp there is no way
             * to tell that apart from bytes it understands. */
            const id = handed.workspace.workspaceId(handed.tag);
            landed = handed.workspace.persist(
                handed.tag,
                { v: 1, state: bytes },
                isSave ? { workspace_id: id, state_index: step }
                       : { workspace_id: id + "-draft", state_index: 0 },
            ) !== false;
        } catch (_) {
            landed = false;
        }
        // "it landed: the position moves" / "it failed: the position does not
        // move". Two writes in flight is how the position comes to describe a
        // state that was never written, which is why WRITING queues.
        if (landed && isSave) {
            position = step;
            highest = step;                  // a save drops every point above it
            /* AND THE DROP REACHES THE DISK. `highest` alone only stops this
             * session from stepping forward into an abandoned point; the files
             * for those points stay on the server, so a later session could still
             * find them. The workspace's own ordering rule matters here: the
             * delete finishes before the next baseline is written, so the history
             * never briefly holds both (workspace.md § 9). */
            try {
                handed.workspace.pruneStatesAbove(
                    handed.workspace.workspaceId(handed.tag), step);
            } catch (_) { /* a failed prune leaves a stale tail, not a lost save */ }
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

    /* The write that is on its way, or null when none is.
     *
     * `send` drains anything held before it resolves, so this one promise covers
     * the whole run of writes — not just the first of them. It exists so a READ
     * can wait for a write, which the SETTLED / CHANGING / WRITING states cannot
     * do on their own: they order writes against other writes, and a read is
     * neither. */
    let inFlight = null;

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
        inFlight = send(step, isSave);
        try { return await inFlight; }
        finally { inFlight = null; }
    }

    /* Wait for a write that has not finished, if there is one.
     *
     * A failed write must not stop the read: the point of the wait is that the
     * position and the stored bytes have settled, and a write that failed has
     * settled too — it left the position where it was. */
    async function settled() {
        if (!inFlight) return;
        try { await inFlight; } catch (_) { /* it failed; it is still over */ }
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
        /* NO POINT is laid down on a timer or because something changed: the
         * sequence grows from opening a structure, an explicit save, and a load.
         *
         * THE SESSION COPY IS A DIFFERENT WRITE, and it follows every change
         * (§ 11.3). An edit changes what would be lost if the tab closed, so an
         * edit refreshes it — `request` with `isSave` false, which sends the
         * bytes to the browser and lays down no point.
         *
         * Persistence is about the accident, the timeline about the decision.
         * Making one behave like the other takes it away: save on every edit and
         * every keystroke becomes a milestone, so there is nothing left to come
         * back to; save on neither and a reload loses the afternoon.
         *
         * It goes through the same machine as any other write, so an edit landing
         * mid-bracket is held exactly like a save would be — the halfway
         * structure a bracket exists to keep out of storage is the same halfway
         * structure either way. */
        edited() {
            if (!anchored) return;
            setBadge(true);
            return request(position, false);
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
            /* WAIT FOR A WRITE THAT IS STILL ON ITS WAY, and do it BEFORE working
             * out which point to fetch.
             *
             * Both halves matter. A save moves `position` and lowers the badge
             * when it lands, and both are read below to decide the target — so a
             * Retract pressed while a save was in flight used to aim at the point
             * BEFORE the one just saved, and then fetch bytes the save had not
             * written yet. Two wrong answers from one missing wait.
             *
             * The write machine could not prevent it: SETTLED / CHANGING /
             * WRITING queue writes behind other writes, and a read is not a
             * write, so it walked straight past them. */
            await settled();
            let target;
            if (step === 0) {
                target = position;                 // restore where I was
            } else if (step < 0 && uncommitted) {
                target = position;                 // spend the unsaved work first
            } else {
                target = position + step;
            }
            if (target < 0 || target > highest) return null;

            /* THE POINT COMES BACK BY ITS NUMBER, from the id this tag's drafts
             * are kept under. `readState` answers null for a miss and for any
             * failure alike, deliberately (workspace.md § 5) — so "there is no
             * such point" and "the server could not be reached" arrive the same
             * way, and the only safe reading of null is "nothing to put back". */
            const saved = await handed.workspace.readState({
                workspace_id: handed.workspace.workspaceId(handed.tag),
                state_index:  target,
            });
            if (saved == null) return null;
            // Written by this module, so unwrapped by it. A file from a layout
            // this code does not know is not something to guess at.
            if (saved.v !== 1) return null;
            handed.restoreState(saved.state);
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
            // Released here, so this is where the wait a read performs has to be
            // able to see it — the same bookkeeping as `request`, for the same
            // reason (`settled`).
            inFlight = send(next.step, next.isSave);
            try { return await inFlight; }
            finally { inFlight = null; }
        },
    };
}
