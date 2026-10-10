/* molbuilder Modify tab -- op controls + state-timeline, NO structure data of its own.
 *
 * The MolView module owns the
 * viewer, the render loop, the selection panel, AND the structure itself.
 * This file is handed that viewer and holds ONLY:
 *
 *   * the edit-op controls -- Atom (Delete, Add atom), Transform (Translate, Center,
 *     Rotate, Orient).  Each POSTs op-PARAMS via
 *     ``molview.data.applyOp(op, args)``; the MODULE builds the structure body from its own
 *     data and applies the response atomically.  This file
 *     never sends or holds structure geometry/metadata.
 *   * the state timeline -- "Save state" (``data.save(1)``) / "Retract" (``data.load(-1)``)
 *     and the reload-restore (``data.load(0)``).
 *   * op-tab enablement + anchor readouts, read LIVE off the unified API
 *     (``_elements`` / ``_coords`` -> getElements / getCoordinates).  There is NO local
 *     ``state.*`` structure mirror -- the ``state`` object below is just the in-flight lock.
 *
 * The viewer, click-select, halos, measurement, toggles, and persistence are all the
 * module's (mounted by modify/selection-bootstrap.js into the empty #molview-host).
 *
 * Spec: docs/web/tabs.md; docs/web/molview.md.
 */
import { formula as mvFormula, toDisplay as displayNumber }
    from "/static/lib/molview/index.js";

/**
 * Start the Modify tab's op controls against a viewer.
 *
 * @param viewer  the handle `mount` returned — this page's viewer. It is passed
 *                in by modify/selection-bootstrap.js, which mounts it. Nothing
 *                here looks a viewer up: the only way to one is the handle you
 *                were handed (molview.md § 5.6), and there is nowhere to look.
 */
/* THE ONE OP WRAPPER, LENT TO THE SIBLING PANELS.
 *
 * `postOp` below owns four things no panel should own twice: the in-flight
 * lock, the edit-status line, the selection refresh and the state-timeline
 * refresh.  The Slab panel needs all four.
 *
 * It is created inside `init` because it closes over this page's state, so the
 * door is published here once init has run.  One Modify page per document, so
 * one door.  (ui-contract.md § 1: a thing used by more than one caller has
 * exactly one owner.) */
let _postOp = null;

export function runOp(path, extraBody, label) {
    return _postOp
        ? _postOp(path, extraBody, label)
        : Promise.resolve(null);
}

export function init(viewer) {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // Transient UI state ONLY -- the in-flight op lock.  There is NO local structure
    // mirror: every geometry/metadata read goes LIVE through the unified molview.data API
    // (getStructure / getElements / getCoordinates), so the Modify tab holds zero copies of
    // the structure and can never drift from the single source.  The op-REQUEST body is
    // built inside the module (`applyOp`), not here.
    const state = {
        inFlight: false,       // true while an /api/modify/* fetch is open
    };

    // The reload-restore, so `init` can hand it to the owner (see the bottom of
    // this file).  Set once, at the end of the wiring block.
    let _restored = null;
    /* True while `#status` is showing the "Restored N-atom structure"
     * sentence.  A restore ANNOUNCES WHAT CAME BACK, so the first change to
     * the structure afterwards makes it history -- see `_retireRestoreNotice`. */
    let _restoreNoticeUp = false;

    // ----- Unified data model + persistence accessors ---------------- //
    //
    // Every structure read, every edit, every selection op and every
    // state-timeline call goes through the viewer this page mounted
    // (molview.md § 9.3).  Reload-restore is `load(0)`, which reads the state
    // this viewer's own tag was saved under.
    function _data() {
        return (viewer && viewer.ok) ? viewer.data : null;
    }

    // ----- Selection helpers ----------------------------------------- //
    //
    // The selection store is the canonical source of truth.  These
    // helpers read live so ops always see the current selection
    // without keeping a local mirror.  ``selectedIndices()`` returns
    // a sorted-ascending number[].

    // Selection reads/writes go through the unified model's selection
    // sub-namespace (``molview.data.selection``) -- never a direct
    // store reach.  Read-live so ops see the current selection without
    // a local mirror.
    function _selStore() {
        const d = _data();
        return (d && d.selection) ? d.selection : null;
    }
    /* WHICH ATOMS ARE SELECTED — AS A SET, AND NOT IN ANY ORDER.
     *
     * The store SORTS (`add()` does), and *All* / *Invert* / a filter build
     * one with no pick order at all.  Ordered gestures
     * read the ruler's track instead — `pickedInOrder()` below.
     *
     * `get()` is the selection door's read and hands back its
     * own copy (molview.md § 9.5). */
    function selectedIndices() {
        const s = _selStore();
        return s ? s.get() : [];
    }
    /* THE ORDERED TRACK, for the gestures whose answer is WHICH WAS FIRST
     * (molview.md § 11.6).  `model-jobs.js`'s `ordered` column already sends
     * these ops the picks; this is the same read, so the readout above a
     * button names the atoms the server will actually be given. */
    function pickedInOrder() {
        const d = _data();
        return (d && d.measurement) ? d.measurement.getState().picks : [];
    }

    // Structure reads through the unified API -- the SINGLE source (no state.* mirror).
    // Cheap accessors: getElements/getCoordinates map the store atoms without a full clone.
    function _elements() {
        const d = _data();
        return (d && typeof d.getElements === "function" && d.getElements()) || [];
    }
    function _nAtoms() { return _elements().length; }
    function _coords() {
        const d = _data();
        return (d && typeof d.getCoordinates === "function" && d.getCoordinates()) || [];
    }

    // The 3Dmol viewer, the render loop, the view chrome, the atom list and
    // click-to-select are the MODULE's (molview.mount, from
    // selection-bootstrap.js): this file has no viewer handle and no raw-3Dmol reach.

    // Edit-panel button enablement + per-op anchor readouts.
    //
    // Called from two places:
    //   1. The selection store's subscriber (re-runs on every store
    //      mutation) so buttons follow the live selection.
    //   2. ``postOp()`` start/end to flip enablement during in-flight
    //      requests (otherwise a double-click could submit twice).
    //
    // Selection is read live from the store.
    function refreshSelectionUI() {
        const sel    = selectedIndices();
        // The ordered ops read the ruler, so their readouts and their
        // enablement must read the same thing the server will be sent.
        const picks  = pickedInOrder();
        const locked = state.inFlight;
        const els    = _elements();   // LIVE elements from molview.data (no state.* mirror)

        // Delete: any selection + no op in flight.
        const deleteBtn = $("delete-apply");
        if (deleteBtn) deleteBtn.disabled = locked || sel.length === 0;

        // Add atom: one anchor, or none -- and none means the world origin
        // (molview.md § 11.1, `emptySelection: "origin"`).  Only an ambiguous
        // selection of two or more disables it, because that is the one case
        // with no answer: which of them would the offset be measured from?
        const addBtn = $("add-apply");
        const anchorReadout = $("add-anchor-readout");
        if (addBtn && anchorReadout) {
            if (sel.length === 1) {
                const a = sel[0];
                addBtn.disabled = locked;
                anchorReadout.textContent =
                    `Anchor: #${displayNumber(a)} ${els[a]}`;
            } else if (sel.length === 0) {
                // NAMED, not blank.  "(none)" read as "this cannot run yet";
                // the offset does have a reference here, and saying which one
                // is what makes the enabled button make sense.
                addBtn.disabled = locked;
                anchorReadout.textContent = "Anchor: origin (0, 0, 0)";
            } else {
                addBtn.disabled = true;
                anchorReadout.textContent =
                    "Anchor: pick one atom, or none for the origin";
            }
        }

        // Orient: exactly two anchors + no op in flight.
        const orientBtn = $("orient-apply");
        const orientReadout = $("orient-anchor-readout");
        if (orientBtn && orientReadout) {
            if (picks.length === 2) {
                const [a, b] = picks;
                orientBtn.disabled = locked;
                orientReadout.textContent =
                    `Anchors: #${displayNumber(a)} ${els[a]} → ` +
                    `#${displayNumber(b)} ${els[b]}`;
            } else {
                orientBtn.disabled = true;
                orientReadout.textContent =
                    picks.length === 0
                        ? "Anchors: pick two atoms with the ruler"
                        : picks.length === 1
                            ? "Anchors: pick one more atom"
                            : "Anchors: pick exactly two atoms";
            }
        }

        // Rotate / Center / Translate: no selection requirement;
        // just need a loaded structure.
        const rotateBtn = $("rotate-apply");
        if (rotateBtn) rotateBtn.disabled = locked || _nAtoms() === 0;
        const centerBtn = $("center-apply");
        if (centerBtn) centerBtn.disabled = locked || _nAtoms() === 0;
        const translateBtn = $("translate-apply");
        if (translateBtn) translateBtn.disabled = locked || _nAtoms() === 0;
    }

    /** The shared `.status` writer (lib/status.js).
     *
     *  `#status` is the SHARED status line -- page-shell.css owns `.status`
     *  -- which is why the severity here is bare `ok` / `error` and not this
     *  page's `modify-status--*`: they are different elements with different
     *  owners, and giving them one vocabulary would mean this page renaming a
     *  shared component. */
    function setStatus(msg, kind = null) {
        window.molbuilder.status.set("status", msg, kind);
    }

    // Update the section header's #title-readout from the LIVE structure (unified API).
    // The Hill formula() belongs to MolView and comes through its one door -- we
    // `import` it (as mvFormula, top of file) rather than re-implement it.  It is a
    // pure stateless helper, so it is a direct ES import (no global lookup, no load-order dance).
    function _refreshTitleReadout() {
        const el = $("title-readout");
        if (!el) return;
        const d = _data();
        const s = d ? d.getStructure() : null;
        const title = (s && s.title) || "";
        const f = mvFormula(_elements());
        el.textContent = title ? `${title} (${f})` : (s ? f : "");
    }

    // ----- State timeline: Save state / Retract ---------------------- //
    //
    // Undo is the model's
    // push-only state timeline: the user takes
    // an explicit checkpoint with "Save state" (``save(1)``), and
    // "Retract" (``load(-1)``) rolls the WHOLE model back to the
    // previous checkpoint.  ``state_index`` (0 = the loaded anchor)
    // and ``uncommitted`` (in-memory changes since the last checkpoint)
    // are LIVE reads off ``molview.data``.

    // Enable/disable the timeline controls off the model's LIVE reads.
    // Retract is meaningful when there is something to undo: an earlier
    // checkpoint (``state_index`` > 0) OR uncommitted edits since the last
    // checkpoint (``uncommitted``) -- the first Retract reverts uncommitted
    // edits to the current checkpoint (load(-1)), so it must be
    // enabled while dirty even at the index-0 anchor.  Save state whenever a
    // structure is loaded.  Named refreshUndoButton for continuity with the
    // render-hook callers.
    function refreshUndoButton() {
        const d = _data();
        const idx = (d && typeof d.state_index === "number") ? d.state_index : 0;
        const dirty = !!(d && d.uncommitted);
        const retractBtn = $("undo-op");
        if (retractBtn) retractBtn.disabled = state.inFlight || !(idx > 0 || dirty);
        const saveBtn = $("save-state");
        if (saveBtn) saveBtn.disabled = state.inFlight || _nAtoms() === 0;

        // Dirty/target indicator next to Retract: tell the user whether the
        // model has unsaved edits and WHICH state Retract will restore.
        //   * dirty  -> Retract discards the unsaved edits and returns to the
        //              LAST SAVED state (#idx);
        //   * clean & idx>0 -> Retract steps back a checkpoint (#idx -> #idx-1);
        //   * clean & idx==0 -> nothing to retract.
        const statusEl = $("timeline-status");
        if (statusEl) {
            /* AN EMPTY CANVAS STILL HAS A TIMELINE, and it stands at #0.
             *
             * `Clear` ANCHORS a fresh point 0 on the empty canvas
             * (molview.md § 6.7a), so after clearing there is a sequence and
             * the user is standing at its start (user, 2026-09-02: "time line
             * retract/save should be cleared to #0").
             *
             * So the empty case falls through to the branches below, which
             * read the model: #0, clean, nothing to retract -- which is
             * exactly what Retract does there, in both the cleared case and
             * on a page where nothing has ever been loaded. */
            if (dirty) {
                statusEl.textContent = `● unsaved — Retract → #${idx}`;
                statusEl.className = "modify-timeline-status is-dirty";
                statusEl.title =
                    `Unsaved edits since checkpoint #${idx}. `
                    + `Retract discards them and returns to #${idx}.`;
            } else if (idx > 0) {
                statusEl.textContent = `✓ saved #${idx} — Retract → #${idx - 1}`;
                statusEl.className = "modify-timeline-status is-clean";
                statusEl.title =
                    `At saved checkpoint #${idx}. Retract steps back to #${idx - 1}.`;
            } else {
                statusEl.textContent = `✓ saved #0`;
                statusEl.className = "modify-timeline-status is-clean";
                statusEl.title = "At the initial state (#0). Nothing to retract.";
            }
        }
    }

    // "Save state": commit an undoable checkpoint (``save(1)``).
    // The model persists the current snapshot and advances the index.
    async function saveState() {
        const d = _data();
        if (!d || typeof d.save !== "function") return;
        if (_nAtoms() === 0) {
            setEditStatus("Load a structure first.", "error");
            return;
        }
        try {
            await d.save(1);
            setEditStatus(
                `Saved state #${d.state_index}.`, "ok");
        } catch (e) {
            setEditStatus(
                `Save state failed: ${(e && e.message) || String(e)}`,
                "error");
        }
        refreshUndoButton();
    }

    // "Retract": roll back to the previous checkpoint (``load(-1)``).
    // Gate on ``uncommitted`` -- if there are in-memory changes since
    // the last Save state, warn (they will be discarded) before
    // popping.  Reuses the shared discard-unsaved warning modal.
    async function retractState() {
        const d = _data();
        if (!d || typeof d.load !== "function") return;
        if (d.uncommitted) {
            const modal = window.molbuilder && window.molbuilder.warningModal;
            if (modal && typeof modal.confirmDiscardUnsaved === "function") {
                const proceed = await modal.confirmDiscardUnsaved();
                if (!proceed) return;
            }
        }
        try {
            /* READ THE ANSWER.  `load` returns the index it moved to, or null
             * when it did not move (molview.md § 11.2).
             *
             * `at === null`, not `!at`: index 0 is a real state and a falsy
             * number, so a truthiness test would call the oldest retraction in
             * the sequence a failure. */
            const at = await d.load(-1);
            if (at === null || at === undefined) {
                /* ONE sentence, because the module cannot honestly say more.
                 * `load` returns null for a target out of range, for a point
                 * whose file is gone (a sequence is bounded at its last 30
                 * saves, workspace.md § 9.1), and for a server that did not
                 * answer -- `readState` conflates the last two on purpose
                 * (workspace.md § 5).  A distinction the code cannot make is
                 * not one to print. */
                setEditStatus(
                    "Nothing changed — that earlier state could not be "
                    + "brought back. A tab keeps its last 30 saves.",
                    "warn");
            } else {
                setEditStatus(`Retracted to state #${at}.`, "ok");
            }
        } catch (e) {
            setEditStatus(
                `Retract failed: ${(e && e.message) || String(e)}`,
                "error");
        }
        refreshUndoButton();
    }

    function setEditStatus(msg, kind = null) {
        const el = $("edit-status");
        if (!el) return;
        el.textContent = msg;
        el.className = "modify-status" + (kind ? ` modify-status--${kind}` : "");
    }

    /* NO ADVISORY REGION HERE. What the server says about a structure is shown
     * BY MOLVIEW, in its own panel (molview.md § 6.8): a cell notice under the
     * Cell rows it is about, everything else on one line above the tabs. This
     * tab drawing them too would put one fact in two places, which is exactly
     * what the plan's carve-out row for `commitPeriodicityOp` forbids.
     *
     * What reaches nobody is a
     * `validate_geometry` finding — a coincident-atom warning, say — because no
     * `/api/modify/*` route runs that validator at all. That is a hole in the
     * findings-delivery contract (task #37), not something a display fixes. */

    async function postOp(path, extraBody, label) {
        // Modifier-button wrapper around ``molview.data.applyOp``.  Owns the
        // UI-level concerns the module deliberately stays out of: the
        // in-flight lock (prevents a double-click double-fire), the
        // edit-status text, and the selection-UI refresh.  The module
        // (`applyOp`) owns the HTTP fetch + atomic state
        // replacement; this wrapper composes them with the button's
        // user-facing affordances.
        if (state.inFlight) return null;
        state.inFlight = true;
        refreshSelectionUI();
        setEditStatus(`${label}…`);
        const op = path.replace(/^\/api\/modify\//, "");
        let r = null;
        try {
            try {
                const d = _data();
                if (!d || typeof d.applyOp !== "function") {
                    setEditStatus(`${label} failed: data model unavailable.`,
                        "error");
                    return null;
                }
                r = await d.applyOp(op, extraBody);
            } catch (e) {
                setEditStatus(
                    `${label} failed: ${(e && e.message) || String(e)}`,
                    "error",
                );
                return null;
            }
            if (!r) {
                setEditStatus(`${label} failed.`, "error");
                return null;
            }
            /* THE COUNT IS READ OFF THE STRUCTURE THE DOOR HANDED BACK.
             * `applyOp` answers the structure itself (§ 6.9) — elements and
             * their metadata — not a report about it. */
            setEditStatus(
                `${label}: ${(r.elements || []).length} atoms.`, "ok");
            return r;
        } finally {
            state.inFlight = false;
            refreshSelectionUI();
            // The op cleared the in-flight lock + (usually) changed the model, so the
            // state-timeline buttons must re-evaluate: an op leaves the model `uncommitted`
            // and keeps Save state enabled; the in-flight disable is now lifted.
            refreshUndoButton();
        }
    }

    _postOp = postOp;   // publish the door for the sibling panels

    /* START EMPTY.  Not a server op -- there is nothing to send and nothing
     * to compute, so it does not go through `postOp`: it asks the module for
     * an empty viewer and the module decides what empty MEANS (molview.md
     * § 6.7a).  This tab never builds a structure of its own (§ 5.4), which
     * is also why "and it takes the cell" is not spelled out here. */
    async function applyClear() {
        const d = _data();
        if (!d || typeof d.clear !== "function") {
            setEditStatus("Clear failed: data model unavailable.", "error");
            return;
        }
        /* ASKED FOR, NEVER ASSUMED.  This does not only empty the canvas: it
         * re-anchors the timeline at #0 and persists the empty canvas as the
         * state it starts from, so every saved state before it stops being
         * reachable from here.  That is not something to discover afterwards
         * (user, 2026-09-02: "clear() need to be confirmed by the user
         * because of this clear of all saved state").
         *
         * Through the app's own modal, never `window.confirm`: a native
         * dialog blocks the page's event loop, and this app's other
         * destructive steps all go through this one door. */
        const modal = window.molbuilder && window.molbuilder.warningModal;
        if (modal && typeof modal.confirmDiscardUnsaved === "function") {
            const go = await modal.confirmDiscardUnsaved({
                title: "Clear structure?",
                body: "This removes the structure, its metadata and its "
                    + "cell, and restarts the timeline at #0 — the saved "
                    + "states before it will no longer be reachable here.",
                confirmLabel: "Clear structure",
            });
            if (!go) return;
        }
        if (d.clear() === false) {         // a read-only viewer says no
            setEditStatus("This viewer is read-only.", "error");
            return;
        }
        /* THE PAGE'S OWN NOTE GOES TOO.  Which file is on the canvas is the
         * PAGE's state, not the viewer's (molview.md § 6.7), and it is
         * persisted. */
        const page = window.molbuilder && window.molbuilder.structurePage;
        if (page && typeof page.markLoadedFrom === "function") {
            page.markLoadedFrom(null);
        }
        setEditStatus("Cleared — empty canvas, timeline back to #0.", "ok");
        refreshSelectionUI();
        refreshUndoButton();
    }

    async function applyDelete() {
        // The module resolves the acting group from the live selection and
        // rejects an empty one (delete's empty-policy = "reject"); the Delete
        // button is disabled at zero selection anyway.  We pass op-params only
        // -- NOT the group -- so the module owns resolution + enforcement.
        await postOp("/api/modify/delete", {}, "Deleted");
    }

    // ----- Transform subtab: rigid translate ops ------------------ //
    // Both ops route through the shared /api/modify/translate
    // endpoint; only the body changes (recenter:true vs explicit
    // dx/dy/dz).  After the structure shifts, the module's render
    // reacts to the molview.data change and re-fits the camera.
    async function applyCenter() {
        if (_nAtoms() === 0) {
            setEditStatus("Load a structure first.", "error");
            return;
        }
        await postOp(
            "/api/modify/translate",
            { recenter: true },
            "Centered at origin",
        );
    }

    async function applyTranslate() {
        if (_nAtoms() === 0) {
            setEditStatus("Load a structure first.", "error");
            return;
        }
        const dx = Number($("translate-dx").value) || 0;
        const dy = Number($("translate-dy").value) || 0;
        const dz = Number($("translate-dz").value) || 0;
        if (dx === 0 && dy === 0 && dz === 0) {
            setEditStatus(
                "Nothing to translate (Δx, Δy, Δz are all 0).",
                "error",
            );
            return;
        }
        await postOp(
            "/api/modify/translate",
            { dx, dy, dz },
            `Translated (${dx}, ${dy}, ${dz}) Å`,
        );
    }

    function readAddOffset() {
        return [
            Number($("add-dx").value),
            Number($("add-dy").value),
            Number($("add-dz").value),
        ];
    }

    function refreshAddDistance() {
        const [dx, dy, dz] = readAddOffset();
        $("add-dx-val").textContent = dx.toFixed(2);
        $("add-dy-val").textContent = dy.toFixed(2);
        $("add-dz-val").textContent = dz.toFixed(2);
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        $("add-distance").textContent = `${d.toFixed(2)} Å`;
    }

    async function applyAddAtom() {
        // The module resolves the single anchor from the SELECTION and
        // enforces arity 1 (`add_atom` needs exactly one, and a set of one has
        // no ambiguous first, so it is not an `ordered` row); the Add
        // button is disabled unless exactly one atom is picked.  We pass the
        // op-params only (element + placement offset) -- NOT the anchor index.
        const element = ($("add-element").value || "H").trim();
        if (!element) {
            setEditStatus("Element required.", "error");
            return;
        }
        const offset = readAddOffset();
        await postOp(
            "/api/modify/add_atom",
            { element, offset },
            `Added ${element}`,
        );
    }

    // ----- M4: orient + rotate ------------------------------------- //

    function getCheckedRadio(name) {
        const r = document.querySelector(
            `input[name="${name}"]:checked`,
        );
        return r ? r.value : null;
    }

    function refreshOrientAngleReadout() {
        const v = Number($("orient-angle").value);
        $("orient-angle-val").textContent = `${v}°`;
    }

    function refreshRotateAngleReadout() {
        const v = Number($("rotate-angle").value);
        $("rotate-angle-val").textContent = `${v}°`;
    }

    async function applyOrient() {
        // The module resolves the two anchors from THE RULER'S TRACK and
        // enforces arity 2 (`orient` declares `ordered: true` and
        // `needsExactly: 2` in the op table); the button is disabled unless
        // exactly two atoms are picked.  Anchor order is the CLICK order, so
        // first -> second sets the tilt direction in orient_along_axis.
        // We pass the op-params only -- NOT the anchors.
        const axis  = getCheckedRadio("orient-axis") || "z";
        const angle = Number($("orient-angle").value);
        const center = $("orient-center").value || "midpoint";
        await postOp(
            "/api/modify/orient",
            { axis, angle, center },
            angle === 0 ? `Oriented along ${axis}`
                        : `Oriented (${axis}, tilt ${angle}°)`,
        );
    }

    async function applyRotate() {
        const axis   = getCheckedRadio("rotate-axis") || "z";
        const angle  = Number($("rotate-angle").value);
        const center = ($("rotate-center") || {}).value || "centroid";
        if (angle === 0) {
            setEditStatus("Angle = 0; nothing to rotate.", "error");
            return;
        }
        await postOp(
            "/api/modify/rotate",
            { axis, angle, center },
            `Rotated ${angle}° around ${axis} (${center} pivot)`,
        );
    }

    // --------------------------------------------------------------- //
    //  Wire DOM events.                                                //
    // --------------------------------------------------------------- //
    // Run now rather than on DOMContentLoaded: the page mounted its viewer before
    // calling this, and the document is long since parsed by then. Waiting for an
    // event that has already fired is how a controller comes to sit there doing
    // nothing.
    (() => {
        // Subscribe to the selection store so the per-op buttons
        // re-evaluate enablement + anchor readouts on every
        // selection change.  Initial fire happens immediately,
        // giving us a clean disabled state before any structure
        // loads.
        const _store = _selStore();
        if (_store) {
            _store.subscribe(() => refreshSelectionUI());
        }
        /* ...and the ordered track, for the same reason: Orient is enabled
         * by the PICKS, so a click that changes them has to
         * reach these buttons. */
        const _d0 = _data();
        if (_d0 && _d0.measurement && _d0.measurement.subscribe) {
            _d0.measurement.subscribe(() => refreshSelectionUI());
        }
        // Composite model subscriber — the "Save to project" button's enable
        // rule depends on isDirty() + a saved target, and the Save-state /
        // Retract controls depend on the LIVE state_index / uncommitted timeline
        // reads; none of those fire through the selection store alone.
        // ``molview.data.subscribe`` fires on EVERY model change (canvas data +
        // selection + timeline), so both button groups stay in lockstep with
        // Save / Load /
        // modifier ops / checkpoints.  This is also the render-reaction
        // hook the module contract asks consumers to use once mutations
        // go through molview.data.
        const _d = _data();
        if (_d && typeof _d.subscribe === "function") {
            _d.subscribe(() => {
                _retireRestoreNotice();
                refreshSelectionUI();
                refreshUndoButton();
                _refreshTitleReadout();
            });
        }
        _refreshTitleReadout();   // initial paint (in case a structure is already loaded)
        // State timeline: "Retract" (undo) -> load(-1),
        // "Save state" -> save(1).  #undo-op is the Retract button.
        /* THE TRANSFORM TAB NEEDS ORDERED PICKS, so it does what the Cell
         * page does: turns the ruler on when reached, and says so (§ 11.6).
         * On the TAB CLICK, never at module load. */
        const transformTab = document.querySelector('[data-op-tab="transform"]');
        if (transformTab) {
            transformTab.addEventListener("click", () => {
                const d = _data();
                if (d && d.measurement && d.measurement.requestPicking()) {
                    const notify = (window.molbuilder || {}).notify;
                    if (notify && notify.show) {
                        notify.show({ id: "ruler-on-transform-tab",
                            level: "info", message: "Measuring is on: the "
                            + "Transform tab picks atoms with the ruler, in "
                            + "the order you click them. Turn it off when you "
                            + "are done here." });
                    }
                }
            });
        }

        const undoBtn = $("undo-op");
        if (undoBtn) undoBtn.addEventListener("click", retractState);
        const saveStateBtn = $("save-state");
        if (saveStateBtn) saveStateBtn.addEventListener("click", saveState);

        // Transform subtab: center-at-origin + translate-by-offset.
        const centerBtn = $("center-apply");
        if (centerBtn) centerBtn.addEventListener("click", applyCenter);
        const translateBtn = $("translate-apply");
        if (translateBtn) translateBtn.addEventListener("click", applyTranslate);

        // Delete + add-atom op buttons.
        const delBtn = $("delete-apply");
        if (delBtn) delBtn.addEventListener("click", applyDelete);
        // Whole-model, and so NOT an Atom-tab op: the button lives in the card
        // header beside Save (user, 2026-09-07).  Wired here anyway because
        // `applyClear` is this file's, and the wiring is by id -- where the
        // button sits in the template is the template's business.
        const clearBtn = $("clear-apply");
        if (clearBtn) clearBtn.addEventListener("click", applyClear);
        const addBtn = $("add-apply");
        if (addBtn) addBtn.addEventListener("click", applyAddAtom);
        // Live distance readout: every slider input refreshes the
        // |offset| display without hitting the server.
        ["add-dx", "add-dy", "add-dz"].forEach((id) => {
            const sl = $(id);
            if (sl) sl.addEventListener("input", refreshAddDistance);
        });
        refreshAddDistance();

        // Orient + rotate op buttons + live angle readouts.
        const orientBtn = $("orient-apply");
        if (orientBtn) orientBtn.addEventListener("click", applyOrient);
        const rotateBtn = $("rotate-apply");
        if (rotateBtn) rotateBtn.addEventListener("click", applyRotate);
        const orientAngle = $("orient-angle");
        if (orientAngle) {
            orientAngle.addEventListener("input", refreshOrientAngleReadout);
            refreshOrientAngleReadout();
        }
        const rotateAngle = $("rotate-angle");
        if (rotateAngle) {
            rotateAngle.addEventListener("input", refreshRotateAngleReadout);
            refreshRotateAngleReadout();
        }

        // (Non-blocking persist failures surface in the app-wide notification bar
        // -- lib/app-notifications.js listens for molbuilder:persist-error, so the
        // Modify tab wires nothing here; see docs/web/notifications.md.)

        // Sub-tabs: click an op-tab button to swap which panel is
        // visible.  Pure DOM toggle (no state in the IIFE; the
        // is-active class is the state).
        document.querySelectorAll(".modify-optab").forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.dataset.opTab;
                document.querySelectorAll(".modify-optab").forEach((b) => {
                    const on = (b.dataset.opTab === target);
                    b.classList.toggle("is-active", on);
                    // Keep aria-selected in sync so screen readers
                    // announce the active tab; pairs with the
                    // aria-controls / aria-labelledby links on the
                    // <button> and <div role="tabpanel"> elements.
                    b.setAttribute("aria-selected", on ? "true" : "false");
                });
                document.querySelectorAll(".modify-optab-panel").forEach((p) => {
                    p.classList.toggle(
                        "is-active",
                        p.dataset.opPanel === target,
                    );
                });
            });
        });

        // Init-structure tabs: same toggle pattern as the op-tabs above but for the
        // Init structure card's generator/loader bar.  Each tab
        // unhides ONE ``.modify-init-panel`` and hides the rest;
        // ``hidden`` is the canonical "panel not active" state
        // (matches the role="tabpanel" pattern).
        document.querySelectorAll(".modify-init-tab").forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.dataset.initTab;
                document.querySelectorAll(".modify-init-tab").forEach((b) => {
                    const on = (b.dataset.initTab === target);
                    b.classList.toggle("is-active", on);
                    b.setAttribute("aria-selected", on ? "true" : "false");
                });
                document.querySelectorAll(".modify-init-panel").forEach((p) => {
                    const on = (p.dataset.initPanel === target);
                    p.classList.toggle("is-active", on);
                    p.hidden = !on;
                });
            });
        });

        // Restore here (after every event handler is wired so the
        // restored UI behaves identically to a freshly-loaded one).
        /* KEPT SO THE OWNER CAN WAIT FOR IT. Whether this page arrived with work
         * already in it decides whether the sidebar's highlighted file may be
         * seeded onto the canvas, and that question has no answer until the
         * restore has finished. `init` hands the promise back. */
        _restored = restoreModifyState();
    })();

    // State persistence across tab navigation is owned by the workspace
    // module -- server-side state files, one per step (workspace.md § 2).  This
    // module's role is restore-only: `load(0)` adopts the stored sequence
    // and puts back the draft, so the WHOLE model (canvas + selection-store
    // atoms + render) rehydrates coherently.


    async function restoreModifyState() {
        // Reload-restore is the mount-restore primitive:
        // ``load(0)`` reloads the current committed state and applies it
        // to the WHOLE model WITHOUT
        // re-anchoring the timeline (unlike
        // ``installMolecule``, the NEW-molecule door).  The data model
        // reads the persisted snapshot itself, so this module does not
        // touch the persistence layer.  It restores
        // structure + selection + view + dirty + timeline position into molview.data; the
        // module's render + this file's molview.data subscription (refreshSelectionUI /
        // refreshUndoButton / _refreshTitleReadout) update the UI as a side effect.
        const d = _data();
        if (!d || typeof d.load !== "function") return;
        // Nothing to declare first: the viewer was mounted before this file was
        // started, and every workspace call names its tag (workspace.md § 4), so
        // there is no shared setting for two files to get out of order.
        /* A FAILED RESTORE IS SAID OUT LOUD AND STOPS HERE.
         *
         * The owner waits for this before wiring the sidebar, so letting it
         * reject would take the Load button and the file gate down with it —
         * a corrupt state file would cost the whole page. Catching it is not
         * hiding it: the sentence goes on the status line, and the page comes
         * up empty rather than not at all. */
        let at;
        try {
            at = await d.load(0);
        } catch (e) {
            setStatus(
                `Could not restore your last structure: `
                + `${(e && e.message) || String(e)}`,
                "error");
            return;
        }
        refreshUndoButton();
        _refreshTitleReadout();
        /* SAY NOTHING WHEN NOTHING CAME BACK. `load(0)` answers the point it put
         * back, or null when this tag has no saved point at all (§ 11.2). */
        if (at === null || at === undefined) return;
        // Read the restored structure LIVE from molview.data (the single source).
        const s = (d.getStructure && d.getStructure()) || null;
        const title = (s && s.title) ? s.title : "unnamed";
        setStatus(
            `Restored ${_nAtoms()}-atom structure (${title}).`,
            "ok");
        _restoreNoticeUp = true;
    }

    /** Clear the restore banner once the structure it describes is gone.
     *
     * A restore announces what came back.  Once you change anything, that
     * is history, so the first data change retires it -- and only the
     * first, because a later status (a save, an error) is not ours to clear.
     */
    function _retireRestoreNotice() {
        if (!_restoreNoticeUp) return;
        _restoreNoticeUp = false;
        setStatus("", null);
    }

    // ----- Test hook ------------------------------------------------- //
    // Exposes a small read-only surface for Playwright E2E tests.
    // Production has zero behavior change -- this just attaches a
    // few references to ``window`` that nothing else looks at.
    // ``getSelected`` reads live from the selection store.
    /* THERE IS NO WAY TO THE RAW 3Dmol VIEWER, and that is the contract, not a
     * gap: MolView conceals the embed (molview.md § 4).
     *
     * What tests need is what the page shows -- atom count, selection,
     * coordinates -- and those are read below through `molview.data`, the one
     * route (§ 9.3). A test that needs to reach past it is testing the embed,
     * which is 3Dmol's to test. */
    window.__molbuilder_modify_test = {
        getSelected: () => selectedIndices(),
        getNAtoms:   () => _nAtoms(),
    };

    window.molbuilder = window.molbuilder || {};

    /* HAND THE RESTORE BACK. Everything above is wired synchronously; the one
     * thing still in flight when `init` returns is `load(0)`. The owner awaits
     * this before deciding whether the sidebar's file may be seeded, because
     * "is there already work on this canvas?" is not answerable until it lands. */
    return _restored;
}
