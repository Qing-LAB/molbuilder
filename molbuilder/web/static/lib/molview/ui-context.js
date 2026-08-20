/* MolView — the view context: how you were looking, kept beside the truth.
 *
 * Contract: docs/web/molview.md § 11.2b (the lane), § 9.6 (the camera's
 *           place), § 9.7 (the bounded pose read).
 * Owns:     one workspace slot per viewer — tag `<owner>:ui`, one file at
 *           state_index 0 — holding view settings, switches, the camera
 *           pose, the displayed frame, and (for viewers with no truth lane)
 *           the selection.
 * Called by: mount.js, once per viewer, after the model and engine exist.
 *
 * NEVER (§ 11.2b — "looking is not changing"):
 *   - touch the timeline, the draft, or the badge.  This module holds no
 *     reference to the history and cannot reach it;
 *   - be read by a state, the draft, or an export;
 *   - apply the camera, the frame, or the selection onto a structure they
 *     did not belong to (`match` is the guard — a stale pose can leave the
 *     molecule off-screen, which is § 9.6's old refusal, kept as a guard).
 */
"use strict";

const VERSION = 1;
const WRITE_DELAY_MS = 400;

/**
 * Wire the lane onto a mounted viewer.  Returns a disposer.
 *
 * @param deps  {model, engine, workspace, owner, canvasEl, hasTruthLane}
 *              `hasTruthLane` is true for an editable viewer, whose draft
 *              already owns the selection (§ 11.2b: two lanes restoring one
 *              fact is § 5.2's drift).
 */
export function attachUiContext(deps) {
    const { model, engine, workspace, owner, canvasEl } = deps;
    const hasTruthLane = !!deps.hasTruthLane;
    if (!model || !engine || !workspace || !owner) return () => {};
    if (typeof workspace.workspaceId !== "function"
            || typeof workspace.persist !== "function"
            || typeof workspace.readState !== "function") return () => {};

    const TAG = owner + ":ui";
    const identity = () => ({
        workspace_id: workspace.workspaceId(TAG),
        state_index: 0,
    });

    /* Writes are DISARMED until the first restore attempt has finished:
     * the stores announce their defaults the moment they are subscribed to,
     * and a write fired then would clobber the very context the restore is
     * about to read. */
    let armed = false;
    /* And silenced while the restore APPLIES, so putting the context back
     * does not echo it straight into a rewrite. */
    let applying = false;
    let timer = null;
    let disposed = false;

    function matchOf() {
        const s = model.getStructure();
        if (!s || !Array.isArray(s.elements)) return null;
        return { n_atoms: s.elements.length, n_frames: model.frameCount() };
    }

    function payload() {
        const out = {
            v: VERSION,
            match: matchOf(),
            view: model.view.get(),
            switches: model.selection.switches(),
            camera: engine.getCamera(),
            frame: model.currentFrame(),
        };
        if (!hasTruthLane) out.selection = model.selection.get();
        return out;
    }

    function flush() {
        timer = null;
        if (disposed || !armed || applying) return;
        workspace.persist(TAG, payload(), identity());
    }

    function schedule() {
        if (disposed || !armed || applying) return;
        if (timer !== null) clearTimeout(timer);
        timer = setTimeout(flush, WRITE_DELAY_MS);
    }

    async function restore() {
        let saved = null;
        try {
            saved = await workspace.readState(identity());
        } catch (_) { /* unreachable storage reads as "nothing saved" */ }
        if (disposed) return;
        if (!saved || saved.v !== VERSION) { armed = true; return; }
        applying = true;
        try {
            // The standing preferences: how things are painted, and which
            // overlays are on.  These are the user's, whatever the molecule.
            const view = saved.view || {};
            for (const k of Object.keys(view)) model.view.set(k, view[k]);
            const sw = saved.switches || {};
            for (const k of Object.keys(sw)) {
                model.selection.setSwitch(k, sw[k]);
            }
            // The structure-bound half, only onto the structure it belonged
            // to (§ 11.2b's guard).
            const now = matchOf();
            const then_ = saved.match || null;
            const matches = !!(now && then_
                && now.n_atoms === then_.n_atoms
                && now.n_frames === then_.n_frames);
            if (matches) {
                if (Number.isInteger(saved.frame)) {
                    model.setCurrentFrame(saved.frame);
                }
                if (saved.camera) engine.setCamera(saved.camera);
                if (!hasTruthLane && Array.isArray(saved.selection)) {
                    model.selection.adopt(saved.selection);
                }
            }
        } finally {
            applying = false;
            armed = true;
        }
    }

    const offs = [];

    /* The restore runs ONCE, after the first structure the viewer holds —
     * installed or adopted, either mode.  The ordering § 11.2b states (the
     * context applies AFTER a draft restore's `view.reset()`, which runs
     * synchronously after its settle announces) is carried by `restore`'s
     * own first await: reading the slot suspends, so the caller's
     * synchronous tail always lands first.  Pinned by
     * test_the_context_applies_after_a_draft_restore_not_under_it — a
     * second deferral mechanism stood here until the mutation test showed
     * it guarded nothing. */
    let restoredOnce = false;
    offs.push(model.subscribe(() => {
        if (restoredOnce || !model.getStructure()) return;
        restoredOnce = true;
        restore().then(schedule);
    }));

    // The writers: any view setting, any switch or selection change, the
    // displayed frame, and the end of a camera gesture.  All debounced into
    // one write; the pose is read AT the write, never tracked.
    offs.push(model.view.subscribe(schedule));
    offs.push(model.selection.subscribe(schedule));
    offs.push(model.onFrameChange(schedule));
    if (canvasEl && typeof canvasEl.addEventListener === "function") {
        const onGesture = () => schedule();
        canvasEl.addEventListener("pointerup", onGesture);
        canvasEl.addEventListener("wheel", onGesture, { passive: true });
        offs.push(() => {
            canvasEl.removeEventListener("pointerup", onGesture);
            canvasEl.removeEventListener("wheel", onGesture);
        });
    }

    return function dispose() {
        disposed = true;
        if (timer !== null) { clearTimeout(timer); timer = null; }
        for (const off of offs.splice(0)) {
            try { off(); } catch (_) {}
        }
    };
}
