/* molbuilder Modify tab -- op controls + state-timeline, NO structure data of its own.
 *
 * Post-migration (Track B + the state.* rip-out): the concealed MolView module owns the
 * viewer, the render loop, the selection panel, AND the in-memory structure
 * (``window.molbuilder.molview.data``).  This file holds ONLY:
 *
 *   * the edit-op controls -- Delete, Add atom, Orient/Rotate (Pose), Center/Translate
 *     (Geom), and the electrode/anchor (Junction) panel.  Each POSTs op-PARAMS via
 *     ``molview.data.applyOp(op, args)``; the MODULE builds the structure body from its own
 *     data (data-model._structureBody) and applies the response atomically.  This file
 *     never sends or holds structure geometry/metadata.
 *   * the state timeline -- "Save state" (``data.save(1)``) / "Retract" (``data.load(-1)``)
 *     and the reload-restore (``data.load(0)``).
 *   * op-tab enablement + anchor readouts, read LIVE off the unified API
 *     (``_elements`` / ``_coords`` -> getElements / getCoordinates).  There is NO local
 *     ``state.*`` structure mirror -- the ``state`` object below is just the in-flight lock.
 *
 * The viewer, click-select, halos, k-grid, measurement, toggles, and persistence are all the
 * module's (mounted by modify/selection-bootstrap.js into the empty #molview-host).
 *
 * Spec: docs/tabs/molbuilder.md; docs/protocols/molview-module.md; molview-migration-plan.md.
 */
(function () {
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

    // ----- Unified data model + persistence accessors ---------------- //
    //
    // The in-memory DATA MODEL is MolView's unified door
    // (``window.molbuilder.molview.data``): every structure read /
    // mutation / selection op / state-timeline call goes through it
    // (data-model.js §19).  Persistence lives on
    // ``window.molbuilder.workspace`` but the tab no longer touches it
    // directly -- reload-restore is ``molview.data.load(0)`` (§19.5),
    // which reads the session mirror inside the data model.
    function _data() {
        return (window.molbuilder
                && window.molbuilder.molview
                && window.molbuilder.molview.data) || null;
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
    // a local mirror.  getState() returns the contract snapshot with
    // ``.indices``.
    function _selStore() {
        const d = _data();
        return (d && d.selection) ? d.selection : null;
    }
    function selectedIndices() {
        const s = _selStore();
        return s ? s.getState().indices.slice() : [];
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
    // NOTE (Track B migration): the base render, the explicit-cell wireframe, and the
    // k-grid tiling controller USED to live here (`_drawBase` / `_wireKgrid` / the cell
    // accessors).  They are gone: the concealed MolView module owns the render loop and
    // the k-grid controller (render.js composes mountKgridRender + mountMeasurementOverlay,
    // driven by molview.data), so Modify no longer draws anything itself.

    // --------------------------------------------------------------- //
    //  The 3Dmol viewer is EMBEDDED BY THE MODULE.                     //
    //                                                                  //
    //  Track B migration: Modify no longer calls viewer.embed itself.  //
    //  selection-bootstrap.js mounts the concealed MolView module into //
    //  the empty #molview-host; molview.mount BUILDS the card, embeds  //
    //  the viewer, owns the render loop + k-grid + measurement, wires  //
    //  the selection viewer-adapter (click + halos), and puts the      //
    //  toggles in the View menu.  Camera pivot-recentering that used   //
    //  to live here (focusMolecule / snapPivotToCenter, raw 3Dmol) is  //
    //  now the View-menu "Reset view" (handle.refit).  So there is NO  //
    //  viewer handle, no embed, and no raw-3Dmol reach in this file.   //
    //  --------------------------------------------------------------- //

    // ----- xyz axis triad ----------------------------------------- //
    //
    // The triad gives the user a fixed orientation reference while
    // rotating the camera.  Two modes (selected automatically based
    // on whether the loaded structure carries lattice vectors):
    //
    //   * Cartesian — fixed-length unit X/Y/Z arrows at the origin.
    //     Used when no cell is defined (the common case for /modify
    //     where structures load from XYZ before any cell is set).
    //   * Cell      — arrows along a/b/c lattice vectors, scaled to
    //     the cell vector lengths.  Used when a cell IS defined
    //     (PDB CRYST1 records, future sidecar metadata, etc.).
    //
    // The actual drawing logic lives in ``lib/mol-axes.js`` so the
    // contract is shared with /results' trajectory inspector and any
    // future tab that mounts a 3Dmol viewer.  This handler is a thin
    // adapter: read the checkbox state, gather the cell (if any),
    // delegate.

    // drawAxes() removed by #203 — the Axes toggle now lives in
    // the embed's standard knob bar, which calls setAxes directly.
    // Any code still calling drawAxes() should be replaced with a
    // direct _handle.setAxes(boolean) call.

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
                    `Anchor: #${a + 1} ${els[a]}`;
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
                    `Anchors: #${a + 1} ${els[a]} → ` +
                    `#${b + 1} ${els[b]}`;
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
                if (sel.length === 1) {
                    elcBtn.disabled = locked;
                    elcReadout.textContent =
                        `Anchor: #${sel[0] + 1} ${els[sel[0]]}.  ` +
                        `Side determines which face the slab grows on.`;
                } else {
                    elcBtn.disabled = true;
                    elcReadout.textContent =
                        "Single mode: pick exactly one anchor.";
                }
            } else {
                // Pair mode: 0 atoms = origin-centred (canonical),
                // 2 atoms = legacy anchor-midpoint.  1 atom is
                // ambiguous; rejected at apply time.
                elcBtn.disabled = locked || _nAtoms() === 0;
                if (sel.length === 0) {
                    elcReadout.textContent =
                        "Pair mode: slabs at z = ±gap/2 around the origin.  "
                        + "Centre + pose the molecule first (Geom + Pose).";
                } else if (sel.length === 2) {
                    const [a, b] = sel;
                    elcReadout.textContent =
                        `Legacy mode: slabs flank #${a + 1} ${els[a]} `
                        + `↔ #${b + 1} ${els[b]} `
                        + "(midpoint of the two anchors).";
                } else {
                    elcBtn.disabled = true;
                    elcReadout.textContent =
                        "Pair mode: select 0 atoms (origin-centred) or "
                        + "2 atoms (legacy anchor pair).";
                }
            }
        }

        const sendBtn = $("send-to-build");
        if (sendBtn) {
            // Save-first enable rule (task #294, 2026-06-08): the
            // Send button is only enabled when the workspace has a
            // clean structure backed by a project file — either
            // freshly loaded from the sidebar (source.kind=file)
            // OR explicitly saved (last_save_to set).  Both expose
            // an on-disk target via ``structureSave.targetPath()``.
            // Forces the workflow (Save → Send) and eliminates the
            // sessionStorage-vs-disk conflict class.
            // Read the dirty bit off the unified model (molview.data).
            const d = _data();
            const save = window.molbuilder
                    && window.molbuilder.structureSave;
            const targetPath = (save && typeof save.targetPath === "function")
                ? save.targetPath() : null;
            const savedAndClean = !!d
                && typeof d.isDirty === "function"
                && !d.isDirty()
                && !!targetPath;
            sendBtn.disabled = locked
                || _nAtoms() === 0
                || !savedAndClean;
            sendBtn.title = savedAndClean
                ? "Send the saved structure to /structure-optimization."
                : "Save the workspace to a project file first; the "
                + "optimization tab loads structures from the project "
                + "sidebar.";
        }
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
        // web/static/lib/trajectory-inspector.css.
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

    // formula() lives in static/lib/mol-format.js; loaded by the template above.  Used by
    // the #title-readout updater (wired on the molview.data subscription in DOMContentLoaded).
    const formula = (window.molbuilder && window.molbuilder.fmt
                     ? window.molbuilder.fmt.formula
                     : (els) => (els && els.length ? els.join("") : "—"));

    // Update the section header's #title-readout from the LIVE structure (unified API).
    function _refreshTitleReadout() {
        const el = $("title-readout");
        if (!el) return;
        const d = _data();
        const s = (d && typeof d.getStructure === "function") ? d.getStructure() : null;
        const title = (s && s.title) || "";
        const f = formula(_elements());
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

    // Enable/disable the timeline controls off the model's LIVE
    // ``state_index`` -- Retract is meaningful only above the anchor
    // (index > 0); Save state whenever a structure is loaded.  Named
    // refreshUndoButton for continuity with the render-hook callers.
    function refreshUndoButton() {
        const d = _data();
        const idx = (d && typeof d.state_index === "number") ? d.state_index : 0;
        const retractBtn = $("undo-op");
        if (retractBtn) retractBtn.disabled = state.inFlight || !(idx > 0);
        const saveBtn = $("save-state");
        if (saveBtn) saveBtn.disabled = state.inFlight || _nAtoms() === 0;
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
        el.className = "muted" + (kind ? ` status-${kind}` : "");
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
            setEditStatus(
                r.issues && r.issues.length
                    ? `${label}: ${r.n_atoms} atoms, ${r.issues.length} issue(s).`
                    : `${label}: ${r.n_atoms} atoms.`,
                "ok",
            );
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

    // ----- Geom subtab: rigid translate ops ----------------------- //
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

    // ----- M5: electrode panel + Send-to-Build handoff ------------- //

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

    async function applyElectrode() {
        const mode = $("elc-mode").value;
        const common = readElcCommonBody();
        const gap = Number($("elc-gap").value);
        // Electrode ops are ordinary modifier ops: the response flows through
        // molview.data.applyOp -> the model.  The module resolves the anchor
        // group from the selection and enforces arity; the Add-pair button is
        // disabled at wrong arity per mode:
        //   single (electrode)            -> exactly 1 anchor; slab on `side`.
        //   pair   (symmetric_electrodes) -> 0 anchors (origin-centred slabs)
        //          or 2 (legacy anchor-pair midpoint, ordered top/bottom by z
        //          INSIDE the module's mapGroup).
        // A rollback point is an explicit "Save state" checkpoint (§19.5) --
        // there is no per-op auto-push.  We pass op-params only, NOT the anchors.
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

    function sendToBuild() {
        // Save-first handoff (task #294, 2026-06-08): the Send
        // button only proceeds when the workspace structure is
        // already saved to a project file (canvas-state has a
        // ``last_save_to`` path AND ``isDirty()`` is false).  The
        // button's enable logic enforces this; ``sendToBuild`` is
        // a defensive re-check in case the dirty bit flipped
        // between subscriber-fire and click.
        //
        // The handoff itself is now a simple sidebar-selection
        // forwarding: we set
        // ``sessionStorage["molbuilder.current_file"]`` to the
        // saved path (the projects sidebar's persistence key) +
        // navigate to /structure-optimization, where the new
        // "Load from sidebar selection" button (task #295)
        // becomes the sole structure-data entry point.  No more
        // ``builder-structure`` payload — the persistence comes
        // from the project file on disk, eliminating the
        // sessionStorage-vs-disk conflict class.
        // Read the dirty bit through the unified model (molview.data).
        const d = _data();
        const save = window.molbuilder
                && window.molbuilder.structureSave;
        const targetPath = (save && typeof save.targetPath === "function")
            ? save.targetPath() : null;
        if (!d || !targetPath || d.isDirty()) {
            setEditStatus(
                "Save to project first — Send-to-Optimization "
              + "needs a project file as input.",
                "error",
            );
            return;
        }
        try {
            const C = (window.molbuilder || {}).constants || {};
            sessionStorage.setItem(
                C.SS_FILE || "molbuilder.current_file", targetPath);
            // Track the directory too so the sidebar opens at
            // the right folder; projects-sidebar's ``state.js``
            // also reads the same key (mirrored via
            // lib/constants.js).
            const slash = targetPath.lastIndexOf("/");
            if (slash >= 0) {
                // Root-level paths (``/foo.xyz``) collapse to ``/``;
                // anything deeper keeps its parent.  ``slash > 0``
                // (the previous guard) silently dropped the dir for
                // root-level files, leaving the sidebar pinned at
                // the previous folder.
                sessionStorage.setItem(
                    C.SS_DIR || "molbuilder.current_dir",
                    targetPath.slice(0, slash) || "/");
            }
        } catch (e) {
            setEditStatus(
                `Could not stage handoff: ${e && e.message}`, "error");
            return;
        }
        window.location.href = "/structure-optimization";
    }

    // --------------------------------------------------------------- //
    //  Wire DOM events.                                                //
    // --------------------------------------------------------------- //
    document.addEventListener("DOMContentLoaded", () => {
        // Subscribe to the selection store so the per-op buttons
        // re-evaluate enablement + anchor readouts on every
        // selection change.  Initial fire happens immediately,
        // giving us a clean disabled state before any structure
        // loads.
        const _store = _selStore();
        if (_store) {
            _store.subscribe(() => refreshSelectionUI());
        }
        // Composite model subscriber — the Send-to-Optimization button's
        // enable rule depends on isDirty() + a saved target, and the
        // Save-state / Retract controls depend on the LIVE state_index /
        // uncommitted timeline reads; none of those fire through the
        // selection store alone.  ``molview.data.subscribe`` fires on
        // EVERY model change (canvas data + selection + timeline), so
        // both button groups stay in lockstep with Save / Load /
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
        // and ``openMolecule`` re-anchors it (prune + reset to index 0) while
        // ``save(1)`` prunes any abandoned tail -- the model owns
        // that lifecycle, so there is no in-viewer stack to clear.

        // (Legacy load-btn + file-picker dead-code block removed
        // 2026-05-18.  The browser-local file dialog was dropped
        // when the Projects sidebar took over.  The "Load from
        // current selection" button was further removed 2026-05-20
        // when the selection store began auto-loading on sidebar
        // change.  The sidebar -> selection-bootstrap.js ->
        // store.setSourceFile -> loadStructureText path is the ONLY
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

        // Geom subtab: center-at-origin + translate-by-offset.
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

        // M5: electrode panel + Send-to-Build handoff.
        const elcBtn = $("elc-apply");
        if (elcBtn) elcBtn.addEventListener("click", applyElectrode);
        const sendBtn = $("send-to-build");
        if (sendBtn) sendBtn.addEventListener("click", sendToBuild);
        // Mode switch: re-evaluate selection requirement + show/hide
        // the side picker; live readouts update on slider drag.
        const modeSel = $("elc-mode");
        if (modeSel) {
            modeSel.addEventListener("change", () => {
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
        restoreModifyState();
    });

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
        // ``openMolecule``, the NEW-molecule door).  The data model
        // reads the persisted snapshot itself, so this module no
        // longer touches the persistence layer here.  It restores
        // structure + selection + view + dirty + timeline position into molview.data; the
        // module's render + this file's molview.data subscription (refreshSelectionUI /
        // refreshUndoButton / _refreshTitleReadout) update the UI as a side effect.
        const d = _data();
        if (!d || typeof d.load !== "function") return;
        // Declare THIS tab's persistence namespace before the restore reads the
        // session mirror.  The Modify tab is one consumer split across two files:
        // selection-bootstrap.js mounts molview with ``owner:"modify"`` (which calls
        // useNamespace), but THIS DOMContentLoaded restore runs first (viewer.js loads
        // earlier), so it must set the same namespace itself -- otherwise it reads the
        // un-namespaced base key and misses the ``::modify`` mirror the last visit wrote
        // (molview-module.md §18.4).  Idempotent with the mount's later call.
        const _ws = window.molbuilder && window.molbuilder.workspace;
        if (_ws && typeof _ws.useNamespace === "function") _ws.useNamespace("modify");
        await d.load(0);
        refreshUndoButton();
        _refreshTitleReadout();
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
        // Geom-subtab tests probe coordinates after a translate / center op.  Read LIVE
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
    // open door (``molview.data.openMolecule({text, filename})``), which
    // sniffs the format from the filename + content and installs the
    // whole model atomically.  The function is named
    // ``loadStructureText`` because it genuinely accepts both formats.
    // This global alias stays so existing consumers (selection-
    // bootstrap, the Sources-card generators' injected ``viewerLoader``,
    // the selection store's setLoader, the page-mount test hook) keep
    // working; they now flow through the unified load door rather than
    // the old persistence-layer text loader that was removed.
    window.molbuilder.loadStructureText = async function (text, filename) {
        setStatus(`Loading ${filename}…`);
        const d = _data();
        if (!d || typeof d.openMolecule !== "function") {
            const msg = "Data model unavailable; cannot load structure.";
            setStatus(msg, "error");
            throw new Error(msg);
        }
        let r;
        try {
            r = await d.openMolecule({ text: text, filename: filename });
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

    // Selection-driven measurement readout: the legacy panel path calls this provider to
    // compute xyz / distance / angle for the current selection.  Coordinates come LIVE from
    // molview.data (``_coords()`` -> getCoordinates); returning an empty array signals "positions not
    // available yet" and the panel hides the readout.  See
    // ``lib/selection/measurements.js`` for the shape and
    // ``lib/selection-panel.js`` ``renderMeasurement`` for the
    // consumer.
    window.molbuilder.selection = window.molbuilder.selection || {};
    window.molbuilder.selection.positionsProvider = function () {
        return _coords();
    };
})();
