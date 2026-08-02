/* molbuilder Modify tab -- op controls + state-timeline, NO structure data of its own.
 *
 * Post-migration (Track B + the state.* rip-out): the concealed MolView module owns the
 * viewer, the render loop, the selection panel, AND the structure itself.
 * This file is handed that viewer and holds ONLY:
 *
 *   * the edit-op controls -- Atom (Delete, Add atom), Transform (Translate, Center,
 *     Rotate, Orient), and the electrode/anchor (Junction) panel.  Each POSTs op-PARAMS via
 *     ``molview.data.applyOp(op, args)``; the MODULE builds the structure body from its own
 *     data (data-model._structureBody) and applies the response atomically.  This file
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
 * Spec: docs/web/tabs.md; docs/web/molview.md; molview-migration-plan.md.
 */
import { formula as mvFormula } from "/static/lib/molview/index.js";

// The atom number as the user sees it. MolView exports `mount` and `formula` and
// nothing else (molview.md § 4), so this is not imported; § 11.5's one home for
// the translation is inside the module, where its own panel and readout use it.
const displayNumber = (index) => index + 1;

/**
 * Start the Modify tab's op controls against a viewer.
 *
 * @param viewer  the handle `mount` returned — this page's viewer. It is passed
 *                in by modify/selection-bootstrap.js, which mounts it. Nothing
 *                here looks a viewer up: the only way to one is the handle you
 *                were handed (molview.md § 5.6), and there is nowhere to look.
 */
export function init(viewer) {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // Transient UI state ONLY -- the in-flight op lock.  There is NO local structure
    // mirror: every geometry/metadata read goes LIVE through the unified molview.data API
    // (getStructure / getElements / getCoordinates), so the Modify tab holds zero copies of
    // the structure and can never drift from the single source (the residue-across-ops /
    // electrode None>None class of bug).  The op-REQUEST body is built inside the module
    // (data-model.applyOp), not here.
    const state = {
        inFlight: false,       // true while an /api/modify/* fetch is open
    };

    // The reload-restore, so `init` can hand it to the owner (see the bottom of
    // this file).  Set once, at the end of the wiring block.
    let _restored = null;

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
    /* WHICH ATOMS ARE SELECTED, IN THE ORDER THEY WERE PICKED.
     *
     * `get()` is the selection door's read for exactly this and hands back its
     * own copy (molview.md § 9.5). This used to ask for `getState().indices` —
     * a key on no snapshot the store has ever produced, so the read was
     * `undefined.slice()`, a TypeError on the FIRST line of every refresh.
     *
     * Both subscriber paths swallow what a subscriber throws, so nothing was
     * printed and nothing looked wrong: the op buttons, every anchor readout,
     * Save state, Retract and the timeline indicator simply never updated
     * again after the page was built. */
    function selectedIndices() {
        const s = _selStore();
        return s ? s.get() : [];
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
    // The symmetric-electrode junction CENTRE: the centroid of the selected atom
    // group, or the ORIGIN when nothing is selected -- the exact rule the op uses
    // server-side (add_symmetric_electrodes).  Shown in the electrode readout so
    // the user can confirm where the slabs will land before applying.
    function _electrodeCenter(sel) {
        if (!sel || !sel.length) return { x: 0, y: 0, z: 0, source: "origin" };
        const c = _coords();
        let x = 0, y = 0, z = 0;
        for (const i of sel) {
            const p = c[i];
            if (p) { x += p[0]; y += p[1]; z += p[2]; }
        }
        const n = sel.length;
        return { x: x / n, y: y / n, z: z / n, source: "selection" };
    }
    // NOTE (Track B migration): the base render, the explicit-cell wireframe, and the
    // isolate render controller USED to live here (`_drawBase` / the cell accessors).
    // They are gone: the concealed MolView module owns the render loop (the render engine
    // in molview/render-engine/ draws from molview.data), so Modify no longer draws anything itself.

    // --------------------------------------------------------------- //
    //  The 3Dmol viewer is EMBEDDED BY THE MODULE.                     //
    //                                                                  //
    //  Track B migration: Modify no longer calls viewer.embed itself.  //
    //  selection-bootstrap.js mounts the concealed MolView module into //
    //  the empty #molview-host; molview.mount BUILDS the card, embeds  //
    //  the viewer, owns the render loop + measurement, wires          //
    //  the selection viewer-adapter (click + halos), and puts the      //
    //  toggles in the View menu.  Camera pivot-recentering that used   //
    //  to live here (focusMolecule / snapPivotToCenter, raw 3Dmol) is  //
    //  now the View-menu "Reset view" (handle.refit).  So there is NO  //
    //  viewer handle, no embed, and no raw-3Dmol reach in this file.   //
    //  --------------------------------------------------------------- //

    // Axes (and all view chrome) are the module's: the embed's knob bar owns the Axes
    // toggle; Modify holds no axis-drawing code (the old drawAxes / mol-axes adapter is gone).

    // --------------------------------------------------------------- //
    //  Atom list + click-to-select are owned by the selection panel    //
    //  (lib/selection-panel.js).  The legacy left-column atom-list +   //
    //  ``onAtomListRowClick`` / ``onViewerAtomClick`` handlers were    //
    //  retired 2026-05-20.  The viewer-adapter (auto-mounted by        //
    //  modify/selection-bootstrap.js) routes viewer clicks straight    //
    //  to ``ws.selection.toggle`` and draws the halo overlay; the panel //
    //  renders the per-atom list with checkboxes.                      //
    // --------------------------------------------------------------- //

    // Edit-panel button enablement + per-op anchor readouts.
    //
    // Called from two places:
    //   1. The selection store's subscriber (re-runs on every store
    //      mutation) so buttons follow the live selection.
    //   2. ``postOp()`` start/end to flip enablement during in-flight
    //      requests (otherwise a double-click could submit twice).
    //
    // Selection is read live from the store; the DOM atom-list +
    // selection-info table this used to populate now live in the
    // selection panel.
    function refreshSelectionUI() {
        const sel    = selectedIndices();
        const locked = state.inFlight;
        const els    = _elements();   // LIVE elements from molview.data (no state.* mirror)

        // Delete: any selection + no op in flight.
        const deleteBtn = $("delete-apply");
        if (deleteBtn) deleteBtn.disabled = locked || sel.length === 0;

        // Add atom: exactly one anchor + no op in flight.
        const addBtn = $("add-apply");
        const anchorReadout = $("add-anchor-readout");
        if (addBtn && anchorReadout) {
            if (sel.length === 1) {
                const a = sel[0];
                addBtn.disabled = locked;
                anchorReadout.textContent =
                    `Anchor: #${displayNumber(a)} ${els[a]}`;
            } else {
                addBtn.disabled = true;
                anchorReadout.textContent =
                    sel.length === 0
                        ? "Anchor: (none)"
                        : "Anchor: pick exactly one atom";
            }
        }

        // Orient: exactly two anchors + no op in flight.
        const orientBtn = $("orient-apply");
        const orientReadout = $("orient-anchor-readout");
        if (orientBtn && orientReadout) {
            if (sel.length === 2) {
                const [a, b] = sel;
                orientBtn.disabled = locked;
                orientReadout.textContent =
                    `Anchors: #${displayNumber(a)} ${els[a]} → ` +
                    `#${displayNumber(b)} ${els[b]}`;
            } else {
                orientBtn.disabled = true;
                orientReadout.textContent =
                    sel.length === 0
                        ? "Anchors: pick two atoms"
                        : sel.length === 1
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

        // Electrode: anchor count depends on mode.
        const elcBtn = $("elc-apply");
        const elcReadout = $("elc-anchor-readout");
        const mode = ($("elc-mode") || {}).value || "symmetric";
        if (elcBtn && elcReadout) {
            if (mode === "single") {
                // Same centring rule as pair mode: the slab centres on the
                // centroid of the selected atom group (any count), or the ORIGIN
                // when nothing is selected.  ``side`` picks which face it grows on.
                elcBtn.disabled = locked || _nAtoms() === 0;
                const center = _electrodeCenter(sel);
                const c = `(${center.x.toFixed(2)}, ${center.y.toFixed(2)}, `
                        + `${center.z.toFixed(2)}) Å`;
                const side = getCheckedRadio("elc-side") || "+z";
                elcReadout.textContent = (sel.length === 0)
                    ? `Single mode: slab centre = ${c} — the ORIGIN (nothing `
                      + `selected); grows on the ${side} face.`
                    : `Single mode: slab centre = ${c} — centroid of the `
                      + `${sel.length} selected atom${sel.length === 1 ? "" : "s"}`
                      + `; grows on the ${side} face.`;
            } else {
                // Pair mode: the junction CENTRE is the centroid of the
                // selected atom group -- any count works (1 atom -> that atom,
                // 2 -> midpoint, N -> centroid).  NOTHING selected -> the origin.
                // Enabled whenever a structure is loaded.  We SHOW the computed
                // centre (x, y, z) so the user can confirm where the slabs land.
                elcBtn.disabled = locked || _nAtoms() === 0;
                const center = _electrodeCenter(sel);   // {x,y,z, source}
                const c = `(${center.x.toFixed(2)}, ${center.y.toFixed(2)}, `
                        + `${center.z.toFixed(2)}) Å`;
                elcReadout.textContent = (sel.length === 0)
                    ? `Pair mode: junction centre = ${c} — the ORIGIN (nothing `
                      + `selected); slabs at z = ${center.z.toFixed(2)} ± gap/2.`
                    : `Pair mode: junction centre = ${c} — centroid of the `
                      + `${sel.length} selected atom${sel.length === 1 ? "" : "s"}.`;
            }
        }

        // (The "Send to Structure optimization" handoff was removed: cross-tab
        // transfer now goes ONLY through a saved project file -- Save to project
        // here, then "Load from project" in the target tab.  No in-memory "send
        // to", so no sessionStorage-vs-disk conflict class to guard.)
    }

    // --------------------------------------------------------------- //
    //  Load: POST /api/build/load with a multipart file upload.  Same  //
    //  endpoint the Build tab uses; M2 doesn't need its own route.     //
    // --------------------------------------------------------------- //
    function setStatus(msg, kind = null) {
        // Compound-class convention shared with Build + Watch:
        // ``className = "status ok"`` so the CSS rule ``.status.ok``
        // applies.  Earlier this tab emitted ``status-ok`` (hyphen
        // form) which would break against a future shared stylesheet;
        // aligned now -- see web/static/style.css +
        // web/static/lib/trajectory/trajectory-inspector.css.
        const el = $("status");
        el.textContent = msg;
        el.className = "status" + (kind ? " " + kind : "");
    }

    // (loadFile() removed 2026-05-18: was the multipart upload path
    // for the now-deleted #file-picker.  The sidebar-mediated
    // loader at window.molbuilder.loadStructureText below is the
    // only structure-loading path; it operates on already-fetched
    // text, not a File object.  Accepts XYZ and PDB content alike --
    // the server's /api/build/load sniffs the format.)

    // Update the section header's #title-readout from the LIVE structure (unified API).
    // The Hill formula() belongs to the molview module (static/lib/viewer/mol-format.js) -- we `import`
    // it (as mvFormula, top of file, from the molview door) rather than re-implement it.  It is a
    // pure stateless helper, so it is a direct ES import (no global lookup, no load-order dance).
    function _refreshTitleReadout() {
        const el = $("title-readout");
        if (!el) return;
        const d = _data();
        const s = (d && typeof d.getStructure === "function") ? d.getStructure() : null;
        const title = (s && s.title) || "";
        const f = mvFormula(_elements());
        el.textContent = title ? `${title} (${f})` : (s ? f : "");
    }

    // NOTE (state.* rip-out, task #42): the Modify tab used to keep a local ``state.*``
    // mirror of the structure (xyz / elements / positions / atom_names / …), populated by an
    // ``applyStructure`` modify-hook and re-read to build op requests.  That parallel copy is
    // GONE: molview.data is the single source, every read goes through the unified API
    // (getStructure / getElements / getCoordinates), and the op-REQUEST body is built INSIDE
    // the module (data-model.applyOp._structureBody).  So there is no currentStateBody and no
    // applyStructure here anymore -- op results flow store->UI via the molview.data
    // subscription (refreshSelectionUI / refreshUndoButton / _refreshTitleReadout).

    // ----- State timeline: Save state / Retract (§19.5) ----------- //
    //
    // The old in-memory undo stack is gone.  Undo is now the model's
    // push-only state timeline (data-model.js §19.5): the user takes
    // an explicit checkpoint with "Save state" (``save(1)``), and
    // "Retract" (``load(-1)``) rolls the WHOLE model back to the
    // previous checkpoint.  ``state_index`` (0 = the loaded anchor)
    // and ``uncommitted`` (in-memory changes since the last checkpoint)
    // are LIVE reads off ``molview.data``.

    // Enable/disable the timeline controls off the model's LIVE reads.
    // Retract is meaningful when there is something to undo: an earlier
    // checkpoint (``state_index`` > 0) OR uncommitted edits since the last
    // checkpoint (``uncommitted``) -- the first Retract reverts uncommitted
    // edits to the current checkpoint (data-model load(-1)), so it must be
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
            if (_nAtoms() === 0) {
                statusEl.textContent = "";
                statusEl.className = "timeline-status";
            } else if (dirty) {
                statusEl.textContent = `● unsaved — Retract → #${idx}`;
                statusEl.className = "timeline-status is-dirty";
                statusEl.title =
                    `Unsaved edits since checkpoint #${idx}. `
                    + `Retract discards them and returns to #${idx}.`;
            } else if (idx > 0) {
                statusEl.textContent = `✓ saved #${idx} — Retract → #${idx - 1}`;
                statusEl.className = "timeline-status is-clean";
                statusEl.title =
                    `At saved checkpoint #${idx}. Retract steps back to #${idx - 1}.`;
            } else {
                statusEl.textContent = `✓ saved #0`;
                statusEl.className = "timeline-status is-clean";
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
            await d.load(-1);
            setEditStatus(
                `Retracted to state #${d.state_index}.`, "ok");
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
        el.className = "modify-status" + (kind ? ` status-${kind}` : "");
    }

    /* WHAT THE SERVER SAID about the structure the edit produced.
     *
     * While EDITING these are shown and never block (design.md § 6). They are
     * read off the model with `getNotices()` (molview.md § 6.8), which answers
     * `{where, list:[{level, message}]}` for the structure now showing, or null
     * when the server had nothing to say. An empty list hides the region so no
     * row survives into the next op.
     *
     * They used to be read off applyOp's return as `r.issues`. That door
     * returns the STRUCTURE (§ 6.9) and never carried an `issues` key, so the
     * region has been fed `undefined` on every op — findings the server took the
     * trouble to compute reached nobody. */
    function renderEditAdvisories(notices) {
        const el = $("edit-advisories");
        if (!el) return;
        const list = (notices && Array.isArray(notices.list))
            ? notices.list.filter(Boolean) : [];
        el.textContent = "";                       // clear any prior op's rows
        if (!list.length) { el.hidden = true; return; }
        for (const iss of list) {
            const sev = (iss && iss.level) || "info";
            const li = document.createElement("li");
            li.className = "modify-advisory modify-advisory--" + sev;
            const tag = document.createElement("span");
            tag.className = "modify-advisory-tag";
            tag.textContent = sev.toUpperCase();
            const msg = document.createElement("span");
            msg.className = "modify-advisory-msg";
            msg.textContent = (iss && iss.message) || "";
            li.appendChild(tag);
            li.appendChild(msg);
            el.appendChild(li);
        }
        el.hidden = false;
    }

    async function postOp(path, extraBody, label) {
        // Modifier-button wrapper around ``molview.data.applyOp``.  Owns the
        // UI-level concerns the module deliberately stays out of: the
        // in-flight lock (prevents a double-click double-fire), the
        // edit-status text, and the selection-UI refresh.  The module
        // (data-model.applyOp) owns the HTTP fetch + atomic state
        // replacement; this wrapper composes them with the button's
        // user-facing affordances.
        if (state.inFlight) return null;
        state.inFlight = true;
        refreshSelectionUI();
        setEditStatus(`${label}…`);
        renderEditAdvisories(null);   // clear the previous op's advisories
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
             * their metadata — not a report about it. Asking it for `n_atoms`
             * put "Deleted: undefined atoms." on screen after every op. */
            setEditStatus(
                `${label}: ${(r.elements || []).length} atoms.`, "ok");
            renderEditAdvisories(
                (typeof _data().getNotices === "function")
                    ? _data().getNotices() : null);
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
    // reacts to the molview.data change and re-fits the camera --
    // there's no separate "re-fit camera" button anymore because
    // every coordinate-changing op already does the right thing.
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
        // The module resolves the single anchor from the selection and enforces
        // arity 1 (add_atom's empty-policy = "reject", arity = 1); the Add
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
        // The module resolves the two anchors from the selection and enforces
        // arity 2 (orient's empty-policy = "reject", arity = 2); the Orient
        // button is disabled unless exactly two atoms are picked.  Anchor order
        // follows the selection order (a0 -> a1 sets the tilt direction in
        // orient_along_axis).  We pass the op-params only -- NOT the anchors.
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

    // ----- M5: electrode panel ------------------------------------- //

    function readElcCommonBody() {
        // Bundle the shared electrode OP-PARAMS both single + symmetric modes need
        // (element / plane / size / orthogonal / offset / lattice_constant).  These are
        // op-args only; the module (data-model.applyOp) merges them with the structure body.
        const m         = Number($("elc-m").value);
        const n         = Number($("elc-n").value);
        const layers    = Number($("elc-layers").value);
        const element   = $("elc-element").value;
        const out = {
            element:    element,
            plane:      getCheckedRadio("elc-plane") || "111",
            size:       [m, n, layers],
            orthogonal: $("elc-orthogonal").checked,
            offset:     [
                Number($("elc-dx").value),
                Number($("elc-dy").value),
            ],
        };
        // Resolve the chosen lattice reference -> a numeric
        // `lattice_constant` payload field.  The API already accepts
        // `lattice_constant` (modify.py:448); the radio is purely a
        // client-side picker that looks up the right value from the
        // /api/modify/meta lattice_table.  Default ref "experimental"
        // matches pre-2026-06-18 behaviour (no field sent ->
        // backend falls back to its hardcoded experimental).
        const ref = getCheckedRadio("elc-lattice-ref") || "experimental";
        const lat = window.__elcLatticeTable
                    && window.__elcLatticeTable[element];
        if (ref !== "experimental" && lat) {
            const value = lat[`a_${ref}`];
            if (typeof value === "number") {
                out.lattice_constant = value;
            }
        }
        return out;
    }

    // Populate the element <select> and plane radios from the
    // server's /api/modify/meta endpoint so the UI never duplicates
    // ``SUPPORTED_FCC_ELEMENTS`` / ``SUPPORTED_FCC_PLANES`` from the
    // Python source.  Adding a new metal in molbuilder.modify reaches
    // the dropdown automatically; no template change.  Defaults: first
    // element is selected, plane "111" is checked when present (it's
    // the canonical close-packed surface for transport junctions).
    async function populateElectrodeMeta() {
        let meta = null;
        try {
            const r = await fetch("/api/modify/meta");
            meta = await r.json();
        } catch (_e) { /* fall through to a hard-coded fallback below */ }
        const elements = (meta && meta.fcc_elements)
            || ["Au", "Ag", "Cu", "Ni", "Pt", "Pd"];
        const planes   = (meta && meta.fcc_planes)
            || ["100", "110", "111"];
        // Stash the lattice table for readElcCommonBody.  Schema:
        // { Au: {a_experimental, a_pbe, a_pbe_siesta_psml, name, system}, ... }
        window.__elcLatticeTable = (meta && meta.lattice_table) || {};
        const elSel = $("elc-element");
        if (elSel) {
            elSel.innerHTML = "";
            for (const sym of elements) {
                const opt = document.createElement("option");
                opt.value = sym;
                opt.textContent = sym;
                if (sym === "Au") opt.selected = true;
                elSel.appendChild(opt);
            }
            // Re-render lattice-ref radios when the element changes
            // so the displayed values track the picker.
            elSel.addEventListener("change", renderLatticeRefRadios);
        }
        const planeBox = $("elc-plane-radios");
        if (planeBox) {
            planeBox.innerHTML = "";
            for (const p of planes) {
                const lbl = document.createElement("label");
                const inp = document.createElement("input");
                inp.type = "radio";
                inp.name = "elc-plane";
                inp.value = p;
                if (p === "111") inp.checked = true;
                lbl.appendChild(inp);
                lbl.appendChild(document.createTextNode(p));
                planeBox.appendChild(lbl);
            }
        }
        // Lattice-ref radios + ⓘ popover wiring.
        renderLatticeRefRadios();
        const infoBtn = $("elc-lattice-ref-info");
        const infoPanel = $("elc-lattice-ref-panel");
        if (infoBtn && infoPanel) {
            infoBtn.addEventListener("click", () => {
                const open = !infoPanel.hidden;
                infoPanel.hidden = open;
                infoBtn.setAttribute("aria-expanded", open ? "false" : "true");
            });
        }
    }

    function renderLatticeRefRadios() {
        const box = $("elc-lattice-ref-radios");
        if (!box) return;
        const element = ($("elc-element") && $("elc-element").value) || "Au";
        const lat = (window.__elcLatticeTable || {})[element] || {};
        const refs = [
            ["experimental",    "Experimental",       lat.a_experimental,    "Wyckoff 1963"],
            ["pbe",             "PBE (all-electron)", lat.a_pbe,             "Haas 2009"],
            ["pbe_siesta_psml", "Your bulk run",      lat.a_pbe_siesta_psml, "user-measured"],
        ];
        // Resolve what should be checked BEFORE the render loop so
        // the per-radio code stays simple.  Rule:
        //   1. If the user's current pick is still selectable
        //      (its numeric value is present), keep it.
        //   2. Otherwise (e.g. previous "siesta_psml" pick where the
        //      value is now null), fall back to "experimental".
        const currentPick = getCheckedRadio("elc-lattice-ref")
                            || "experimental";
        const isPickValid = refs.some(function (r) {
            return r[0] === currentPick && r[2] !== null && r[2] !== undefined;
        });
        const effectivePick = isPickValid ? currentPick : "experimental";
        box.innerHTML = "";
        for (const [value, label, num, src] of refs) {
            const lbl = document.createElement("label");
            const inp = document.createElement("input");
            inp.type = "radio";
            inp.name = "elc-lattice-ref";
            inp.value = value;
            const disabled = (num === null || num === undefined);
            if (disabled) {
                inp.disabled = true;
                // Accessibility: screen readers don't get any signal
                // from the visual opacity fade alone.
                inp.setAttribute("aria-disabled", "true");
                lbl.classList.add("elc-lattice-ref-disabled");
            }
            inp.checked = (value === effectivePick);
            lbl.appendChild(inp);
            const txt = (typeof num === "number")
                        ? ` ${label} (${num.toFixed(4)} Å — ${src})`
                        : ` ${label} (unset — ${src})`;
            lbl.appendChild(document.createTextNode(txt));
            box.appendChild(lbl);
        }
    }

    function refreshElcReadouts() {
        $("elc-gap-val").textContent = `${Number($("elc-gap").value).toFixed(1)} Å`;
        $("elc-dx-val").textContent  = Number($("elc-dx").value).toFixed(2);
        $("elc-dy-val").textContent  = Number($("elc-dy").value).toFixed(2);
        // The gap label tracks the mode: pair-mode gap is
        // electrode-to-electrode; single-mode gap is anchor-to-
        // closest-layer (i.e. ``contact_distance``).
        const mode = $("elc-mode").value;
        $("elc-gap-label").textContent =
            mode === "single" ? "contact" : "gap";
        // Show / hide the side picker by mode.
        const sideRow = $("elc-side-row");
        if (sideRow) sideRow.hidden = (mode !== "single");
    }

    // The gap slider is REUSED for both modes, but they need different physical
    // ranges.  Pair mode is the electrode-to-electrode GAP (wide, 4–30 Å).  Single
    // mode is the anchor-to-closest-layer CONTACT distance, which must reach down to
    // a real metal–adsorbate bond (Au–S ≈ 2.4 Å) — far below any pair gap; the old
    // shared 4 Å floor made a physical single-mode contact impossible.  Adjust the
    // range on mode switch + snap the value to the mode's default when the current
    // value falls outside the new range.  Kept out of refreshElcReadouts (which
    // fires on every slider input) so a drag isn't re-clamped mid-move.
    function applyElcGapRange(mode) {
        const gap = $("elc-gap");
        if (!gap) return;
        // Read the value BEFORE narrowing the range: setting a max below the
        // current value makes the browser clamp it, which would hide an
        // out-of-range value from the snap check below.
        const v = Number(gap.value);
        if (mode === "single") {
            gap.min = "1.5"; gap.max = "6.0"; gap.step = "0.05";
            if (!(v >= 1.5 && v <= 6.0)) gap.value = "2.4";   // canonical Au–S contact
        } else {
            gap.min = "4.0"; gap.max = "30.0"; gap.step = "0.1";
            if (!(v >= 4.0 && v <= 30.0)) gap.value = "12.0";
        }
    }

    async function applyElectrode() {
        const mode = $("elc-mode").value;
        const common = readElcCommonBody();
        const gap = Number($("elc-gap").value);
        // Electrode ops are ordinary modifier ops: the response flows through
        // molview.data.applyOp -> the model.  The module resolves the CENTRE
        // group (``center_indices``) from the current selection -- ONE unified
        // rule for both modes:
        //   single (electrode)            -> slab centred on the selection's
        //                                    centroid; grows on `side`.
        //   pair   (symmetric_electrodes) -> both slabs centred on that same
        //                                    centroid, separated by `gap`.
        // Any count is valid: 1 -> that atom, 2 -> midpoint, N -> centroid,
        // and NO selection -> the world origin (the field is omitted).
        // A rollback point is an explicit "Save state" checkpoint (§19.5) --
        // there is no per-op auto-push.  We pass op-params only; the module
        // fills in ``center_indices`` from the selection.
        let r = null;
        if (mode === "single") {
            const side = getCheckedRadio("elc-side") || "+z";
            r = await postOp(
                "/api/modify/electrode",
                Object.assign({}, common, {
                    side:             side,
                    contact_distance: gap,
                }),
                `Added ${common.element}(${common.plane}) ${side}`,
            );
        } else {
            r = await postOp(
                "/api/modify/symmetric_electrodes",
                Object.assign({}, common, { gap: gap }),
                `Added ${common.element}(${common.plane}) pair`,
            );
        }
        // Void of use: r reflects postOp success; the model already
        // applied it.  refreshUndoButton keeps the timeline controls
        // in step (uncommitted just flipped true).
        if (r) refreshUndoButton();
    }

    // (sendToBuild removed: the "Send to Structure optimization" handoff is gone.
    // Cross-tab/step transfer goes ONLY through a saved project file -- "Save to
    // project" here, then "Load from project" in the target tab.  This is the
    // data-transfer contract: no direct in-memory "send to" between tabs, so the
    // whole sessionStorage-vs-disk / in-memory-corruption class is eliminated.)

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
                refreshSelectionUI();
                refreshUndoButton();
                _refreshTitleReadout();
            });
        }
        _refreshTitleReadout();   // initial paint (in case a structure is already loaded)
        // The old "clear undo history on save/load/discard" subscriber
        // is gone: the state timeline now lives on the model (§19.5),
        // and ``installMolecule`` re-anchors it (prune + reset to index 0) while
        // ``save(1)`` prunes any abandoned tail -- the model owns
        // that lifecycle, so there is no in-viewer stack to clear.

        // (Legacy load-btn + file-picker dead-code block removed
        // 2026-05-18.  The browser-local file dialog was dropped
        // when the Projects sidebar took over.  The "Load from
        // current selection" button was further removed 2026-05-20
        // when the selection store began auto-loading on sidebar
        // change.  The sidebar -> selection-bootstrap.js ->
        // projects.parser.openMolecule -> installMolecule path is the ONLY
        // supported loader.  Test contract:
        // tests/test_web_files.py::TestNoLocalFileInputs pins that
        // #load-btn and #file-picker are NOT in the rendered
        // template.)
        // Style / labels / axes wiring removed by #203 -- the embed's
        // standard knob bar owns those controls now.  The bespoke
        // #rep / #show-indices / #show-axes HTML inputs are gone from
        // modify.html.
        // (The old viewer-bar "Clear selection" button was removed -- it duplicated the
        // selection panel's own Clear.  Clearing is the panel's job.)
        // State timeline (§19.5): "Retract" (undo) -> load(-1),
        // "Save state" -> save(1).  #undo-op keeps its id for
        // continuity; its label/title in modify.html now read
        // "Retract".
        const undoBtn = $("undo-op");
        if (undoBtn) undoBtn.addEventListener("click", retractState);
        const saveStateBtn = $("save-state");
        if (saveStateBtn) saveStateBtn.addEventListener("click", saveState);

        // Focus-molecule is now the module's View-menu "Reset view" (handle.refit);
        // Modify no longer owns a #focus-molecule button.

        // Rotation-pivot snap is now wired via opts.interaction.
        // onDragStart on the embed mount above (search for
        // "interaction:"); the embed owns the drag-detection
        // plumbing (threshold + modifier filtering + canvas-scoped
        // listeners) so we just register the policy here as data.

        // Transform subtab: center-at-origin + translate-by-offset.
        const centerBtn = $("center-apply");
        if (centerBtn) centerBtn.addEventListener("click", applyCenter);
        const translateBtn = $("translate-apply");
        if (translateBtn) translateBtn.addEventListener("click", applyTranslate);

        // M3: delete + add-atom op buttons.
        const delBtn = $("delete-apply");
        if (delBtn) delBtn.addEventListener("click", applyDelete);
        const addBtn = $("add-apply");
        if (addBtn) addBtn.addEventListener("click", applyAddAtom);
        // Live distance readout: every slider input refreshes the
        // |offset| display without hitting the server.
        ["add-dx", "add-dy", "add-dz"].forEach((id) => {
            const sl = $(id);
            if (sl) sl.addEventListener("input", refreshAddDistance);
        });
        refreshAddDistance();

        // M4: orient + rotate op buttons + live angle readouts.
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

        // M5: electrode panel.
        const elcBtn = $("elc-apply");
        if (elcBtn) elcBtn.addEventListener("click", applyElectrode);
        // Mode switch: re-evaluate selection requirement + show/hide
        // the side picker; live readouts update on slider drag.
        const modeSel = $("elc-mode");
        if (modeSel) {
            applyElcGapRange(modeSel.value);   // set the initial range for the default mode
            modeSel.addEventListener("change", () => {
                applyElcGapRange(modeSel.value);
                refreshElcReadouts();
                refreshSelectionUI();
            });
        }
        ["elc-gap", "elc-dx", "elc-dy"].forEach((id) => {
            const sl = $(id);
            if (sl) sl.addEventListener("input", refreshElcReadouts);
        });
        // Populate element + plane controls from /api/modify/meta;
        // refresh the readouts once the controls exist.  Async so we
        // don't block the rest of init -- the mode switch and selection
        // UI are wired off the existing readouts.
        populateElectrodeMeta().then(refreshElcReadouts);
        // (Non-blocking persist failures surface in the app-wide notification bar
        // -- lib/app-notifications.js listens for molbuilder:persist-error, so the
        // Modify tab wires nothing here; see docs/web/notifications.md.)

        // Sub-tabs: click an op-tab button to swap which panel is
        // visible.  Pure DOM toggle (no state in the IIFE; the
        // is-active class is the state).
        document.querySelectorAll(".optab").forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.dataset.opTab;
                document.querySelectorAll(".optab").forEach((b) => {
                    const on = (b.dataset.opTab === target);
                    b.classList.toggle("is-active", on);
                    // Keep aria-selected in sync so screen readers
                    // announce the active tab; pairs with the
                    // aria-controls / aria-labelledby links on the
                    // <button> and <div role="tabpanel"> elements.
                    b.setAttribute("aria-selected", on ? "true" : "false");
                });
                document.querySelectorAll(".optab-panel").forEach((p) => {
                    p.classList.toggle(
                        "is-active",
                        p.dataset.opPanel === target,
                    );
                });
            });
        });

        // Init-structure tabs (Sources reorganization 2026-06-08):
        // same toggle pattern as the op-tabs above but for the
        // Init structure card's generator/loader bar.  Each tab
        // unhides ONE ``.init-tab-panel`` and hides the rest;
        // ``hidden`` is the canonical "panel not active" state
        // (matches the role="tabpanel" pattern).
        document.querySelectorAll(".init-tab").forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.dataset.initTab;
                document.querySelectorAll(".init-tab").forEach((b) => {
                    const on = (b.dataset.initTab === target);
                    b.classList.toggle("is-active", on);
                    b.setAttribute("aria-selected", on ? "true" : "false");
                });
                document.querySelectorAll(".init-tab-panel").forEach((p) => {
                    const on = (p.dataset.initPanel === target);
                    p.classList.toggle("is-active", on);
                    p.hidden = !on;
                });
            });
        });

        // Phase 1: persist structure state across tab navigation.
        // Restore here (after every event handler is wired so the
        // restored UI behaves identically to a freshly-loaded one);
        // save on pagehide so the user's latest state survives a
        // click on /, /watch, or anywhere else.
        /* KEPT SO THE OWNER CAN WAIT FOR IT. Whether this page arrived with work
         * already in it decides whether the sidebar's highlighted file may be
         * seeded onto the canvas, and that question has no answer until the
         * restore has finished. `init` hands the promise back. */
        _restored = restoreModifyState();
    })();

    // State persistence across tab navigation is owned by the workspace
    // PERSISTENCE layer under sessionStorage["molbuilder.workspace.v1"]
    // (sole persistence key).  This module's role is restore-only: read
    // the persisted snapshot, then hand the structure to the unified
    // load door so the WHOLE model (canvas + selection-store atoms +
    // render) rehydrates coherently.


    async function restoreModifyState() {
        // Reload-restore is the §19.5 mount-restore primitive:
        // ``load(0)`` reloads the current committed state from the
        // session mirror and applies it to the WHOLE model WITHOUT
        // re-anchoring the timeline or a network round-trip (unlike
        // ``installMolecule``, the NEW-molecule door).  The data model
        // reads the persisted snapshot itself, so this module no
        // longer touches the persistence layer here.  It restores
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
         * back, or null when this tag has no saved point at all (§ 11.2). The
         * message used to be written either way, so a first visit was greeted
         * with "Restored 0-atom structure (unnamed)" — and that sentence then
         * sat there through the next file load, describing nothing. */
        if (at === null || at === undefined) return;
        // Read the restored structure LIVE from molview.data (the single source).
        const s = (d.getStructure && d.getStructure()) || null;
        const title = (s && s.title) ? s.title : "unnamed";
        setStatus(
            `Restored ${_nAtoms()}-atom structure (${title}).`,
            "ok");
    }

    // ----- Test hook ------------------------------------------------- //
    // Exposes a small read-only surface for Playwright E2E tests.
    // Production has zero behavior change -- this just attaches a
    // few references to ``window`` that nothing else looks at.
    // ``getSelected`` reads live from the selection store (the
    // viewer no longer owns selection state).
    window.__molbuilder_modify_test = {
        // Phase 5e B6: query the handle escape hatch lazily at call
        // time instead of caching the raw viewer at module scope.
        // Production code never has a tempting raw-viewer reference
        // to misuse, and tests still get the same object back via
        // the documented _viewer3dmol() escape (§ 2.4).
        // The viewer is the MODULE's now.  mount.js stashes the embed handle on the
        // built card's viewer host (viewerHost.__molview_test_handle); read it lazily so
        // e2e still gets the raw 3Dmol viewer via the documented escape hatch.
        getViewer:   () => {
            const vh = document.querySelector("#molview-host .viewer");
            const h  = vh && vh.__molview_test_handle;
            return h ? h._viewer3dmol() : null;
        },
        getSelected: () => selectedIndices(),
        getNAtoms:   () => _nAtoms(),
        // Transform-subtab tests probe coordinates after a translate / center op.  Read LIVE
        // from molview.data (the single source) -- there is no state.* mirror to expose.
        getState:    () => {
            const d = _data();
            const s = (d && d.getStructure && d.getStructure()) || null;
            const atoms = (s && Array.isArray(s.atoms)) ? s.atoms : [];
            return {
                n_atoms:   _nAtoms(),
                positions: _coords(),
                // Metadata read LIVE from molview.data's atoms (the single source).
                chain_ids: atoms.map((a) => (a.chainId != null ? a.chainId : null)),
                residue_ids: atoms.map((a) => (a.residueId != null ? a.residueId : null)),
            };
        },
    };

    // Public loader for the Projects sidebar's onLoad callback (and
    // any future tab-coordination code).  Reuses /api/build/load's
    // JSON path so we don't need a browser File object.
    window.molbuilder = window.molbuilder || {};
    // The viewer + its handle belong to the MODULE now: the module registers the embed handle
    // with molview.data at mount (data.attachViewHandle), so molview.data.view reads the
    // module-held handle (§20) -- there is no ``modify.handle`` global anymore.  And (state.*
    // rip-out) no ``modify.currentStateBody`` / ``modify.applyStructure`` either: the op-request
    // body is built INSIDE the module (data-model.applyOp._structureBody from molview.data) and
    // op results flow store -> UI via the molview.data subscription, not a consumer hand-off hook.
    // Load a structure text blob (XYZ or PDB) through the UNIFIED
    // open door (``molview.data.installMolecule({text, filename})``), which
    // sniffs the format from the filename + content and installs the
    // whole model atomically.  The function is named
    // ``loadStructureText`` because it genuinely accepts both formats.
    // NOTE (ESM finalization): the Sources-card generators NO LONGER read this global -- they
    // import { data } and inject their own ``(text,filename)=>data.installMolecule(...)`` adapter
    // (molview-esm-finalization.md §4.1).  This alias is RETAINED only as (a) the page-mount E2E
    // hook (test_molbuilder_e2e.py) and (b) the loader that emits the Modify #status feedback.
    // FOLLOW-UP: remove it once the E2E hook is repointed to data.installMolecule and the "Loaded
    // N-atom" status is rewired (it is the last window.molbuilder.* alias this file publishes).
    window.molbuilder.loadStructureText = async function (text, filename) {
        setStatus(`Loading ${filename}…`);
        const d = _data();
        if (!d || typeof d.installMolecule !== "function") {
            const msg = "Data model unavailable; cannot load structure.";
            setStatus(msg, "error");
            throw new Error(msg);
        }
        let r;
        try {
            r = await d.installMolecule({ text: text, filename: filename });
        } catch (e) {
            const msg = (e && e.message) ? e.message : String(e);
            setStatus(msg, "error");
            throw e;
        }
        const fmt = (r.source_format || "structure").toUpperCase();
        setStatus(
            `Loaded ${r.n_atoms}-atom ${fmt} from ${filename}.`,
            "ok",
        );
        return r;
    };
    // (No ``modify.handle`` runtime registration: the module owns the viewer + attaches
    // the selection adapter to it, so selection-bootstrap no longer waits on it.)

    // (The selection panel's measurement readout reads its coordinates straight
    // from molview.data.getCoordinates() -- a sibling within MolView -- so this
    // page no longer decorates a ``positionsProvider`` onto the molview.selection
    // namespace.  Concealment flows one way: consumers call INTO MolView, they
    // don't hang members ON it.)

    /* HAND THE RESTORE BACK. Everything above is wired synchronously; the one
     * thing still in flight when `init` returns is `load(0)`. The owner awaits
     * this before deciding whether the sidebar's file may be seeded, because
     * "is there already work on this canvas?" is not answerable until it lands. */
    return _restored;
}
