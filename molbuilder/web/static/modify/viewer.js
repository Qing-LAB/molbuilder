/* molbuilder Modify tab.
 *
 * Three-pane UI (atom list ↔ 3Dmol viewer ↔ edit panel):
 *
 *   * the atom list on the left mirrors the structure's atoms;
 *     clicking a row highlights the atom in the viewer.
 *   * the viewer in the middle renders the structure via 3Dmol; clicking
 *     an atom highlights its row in the list.  An xyz axis triad
 *     (toggle: ``Show xyz axes``) sits at the world origin for
 *     orientation reference.
 *   * the edit panel on the right hosts the live edit operations:
 *     Delete, Add atom (with offset sliders + live |offset|
 *     readout), Orient along axis (anchor pair, tilt slider, center
 *     mode), Rotate around axis.  M5's electrode panel is the only
 *     remaining placeholder.
 *
 * Backend: file upload reuses ``POST /api/build/load``; edit ops
 * use ``POST /api/modify/{load,delete,add_atom,orient,rotate}``.
 * State is persisted across tab navigation via sessionStorage
 * (key: ``modify-state``) so a Modify -> Watch -> Modify round
 * trip preserves the loaded structure, the selection, and the
 * 3Dmol camera.
 *
 * Spec: docs/tabs/molbuilder.md (M2-M4 + Phase 1 cross-tab
 * persistence as of 2026-05-09).
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // --------------------------------------------------------------- //
    //  Module state.  Single canonical structure (xyz string + parsed  //
    //  atom metadata).  Selection state is NOT owned here -- it       //
    //  lives in the singleton selection store at                      //
    //  ``window.molbuilder.workspace.selection`` and is read on demand //
    //  through ``selectedIndices()`` below.                           //
    // --------------------------------------------------------------- //
    const state = {
        xyz: null,
        elements: [],          // ["C", "H", ...]
        positions: [],         // [[x, y, z], ...]   parsed from xyz
        atom_names: [],        // ["CA", "HB1", ...] (or [] if absent)
        residue_ids: [],       // [1, 1, 2, ...]   (or [] if absent)
        residue_names: [],     // ["MOL", ...]     (or [] if absent)
        chain_ids: [],         // ["A", ...]       (or [] if absent)
        title: "",
        n_atoms: 0,
        inFlight: false,       // true while an /api/modify/* fetch is open
        history: [],           // {response} snapshots; capped at HISTORY_MAX
    };

    // Cap on the undo history.  Each snapshot carries the full
    // structure response (xyz + metadata, ~50 KB for a 1k-atom
    // junction); 20 entries = ~1 MB worst case.  Older entries fall
    // off the bottom of the stack as new ops are pushed.
    const HISTORY_MAX = 20;

    // ----- Selection helpers ----------------------------------------- //
    //
    // The selection store is the canonical source of truth.  These
    // helpers read live so ops always see the current selection
    // without keeping a local mirror.  ``selectedIndices()`` returns
    // a sorted-ascending number[].

    // Phase 10 (workspace-contract.md §5): selection helpers read
    // through ws.selection, not the legacy selection.store global.
    // Same semantics — read-live so ops see current selection
    // without a local mirror — but the surface is unified.
    function _selStore() {
        return (window.molbuilder
                && window.molbuilder.workspace
                && window.molbuilder.workspace.selection)
               ? window.molbuilder.workspace.selection : null;
    }
    function selectedIndices() {
        const s = _selStore();
        return s ? s.getState().indices.slice() : [];
    }
    function clearStoreSelection() {
        const s = _selStore();
        if (s) s.clear();
    }

    // --------------------------------------------------------------- //
    //  Embedded MolViewer.                                             //
    //                                                                  //
    //  Migrated to the standard embeddable viewer (#198, 2026-06-02;   //
    //  contract: docs/protocols/embedded-viewer.md).  The handle's     //
    //  declarative API drives style / axes / index labels; the raw    //
    //  3Dmol viewer (handle._viewer3dmol()) is still used directly    //
    //  for two things the embed contract doesn't cover:               //
    //    1. The selection-store viewer-adapter (atom-pick + selection-//
    //       halo overlays).  The adapter calls 3Dmol primitives       //
    //       (setClickable / addSphere) on the underlying viewer.      //
    //    2. Camera-pivot recentering (focusMolecule / pivotSelection ///
    //       snapPivotToCenter) which needs ``viewer.zoomTo(selection)``//
    //       semantics not exposed via the embed.                       //
    //  --------------------------------------------------------------- //
    const _viewerHost = $("viewer");
    const _handle = window.molbuilder.viewer.embed(_viewerHost, {
        // No xyz at mount -- /modify loads structures asynchronously
        // via the sidebar's loadStructureText -> applyStructure path.
        // The viewer renders an empty canvas until the first
        // setStructure call.
        style: {
            rep:         "stick",
            radiusScale: 1.0,
        },
        axes:    true,
        // /modify uses the selection-store viewer-adapter for atom
        // pick; the embed's built-in pick is NOT used here (the
        // adapter does its own setClickable wiring + draws labelled
        // region halos which the embed's pick mode doesn't support
        // declaratively yet -- follow-up #203b).
        pick:    { mode: "none" },
        // Standard knob bar above the canvas after #203 migration.
        // The card header is /modify's own ``card.viewer-card``
        // (with #title-readout); the embed's knob bar slots between
        // that header and the 3D canvas via the standard chrome.
        // card.title: "" so the embed skips its own header — the
        // host's <header class="card-header">Viewer</header> in
        // modify.html already owns the title strip.  Without this
        // the user saw stacked "Viewer" + "Structure" h2s (A2,
        // 2026-06-04).
        card:    { title: "", showInfoLine: false, height: "100%" },
        export:  { defaultName: "modify" },
        // Rotation pivot snap policy.  3Dmol's pivot is the camera
        // lookAt; ctrl/shift-pan moves the lookAt off the molecule
        // and a follow-up rotation orbits eccentrically.  Snapping
        // the pivot back onto the molecule on the first drag-confirm
        // mousemove fixes that — the snap is invisible because the
        // user is already moving the camera at that moment.
        // Embed owns the threshold + modifier filtering + canvas
        // listener bookkeeping; we just pass the policy as data.
        interaction: {
            dragThresholdPx: 4,
            onDragStart(ev) {
                // Ctrl/shift/alt drags are pan/zoom gestures — leave
                // the pivot alone.  Plain left-drag is rotation;
                // snap.  snapPivotToCenter is defined below at the
                // top of the IIFE.
                if (ev.modifiers.ctrl
                    || ev.modifiers.shift
                    || ev.modifiers.alt) return;
                snapPivotToCenter();
            },
        },
        onError(err) {
            // Surface viewer errors via the same status pattern
            // /modify already uses (preflight / generator panels).
            try { console.warn("[modify.viewer]", err.code, err.message); }
            catch (_) {}
        },
    });
    // /modify viewer.js #235 follow-up: production code uses only
    // the handle.  Phase 5e B6 (#246) dropped the module-scope
    // ``const viewer = _handle._viewer3dmol()`` capture — the
    // Playwright test fixture (window.__molbuilder_modify_test.
    // getViewer) calls _handle._viewer3dmol() at request time
    // instead, so production code carries no production-tempting
    // reference to the raw 3Dmol viewer.

    // clearViewer() removed by #235 -- it was only callable from
    // applyStructure's `else` branch when state.xyz is falsy, and
    // every caller of applyStructure now passes xyz from a fresh
    // /api/build/load response.  Label / overlay teardown on a
    // structure swap is handled by handle.setStructure (label/axes
    // re-apply automatically) and by the selection adapter's
    // store-subscribe path (overlays clear on the next render).

    // ----- Camera anchoring -------------------------------------- //
    // 3Dmol's mouse-zoom and rotate-around-cursor handlers anchor on
    // the model's PIVOT, which ``viewer.zoomTo()`` sets to the
    // bounding-box centroid of the current selection.  When slabs
    // are added the bounding box grows ~10x and gets dominated by
    // the slabs; the pivot drifts off the molecule and zoom-into-
    // the-molecule starts feeling like "the molecule slides off
    // screen as I scroll".
    //
    // ``focusMolecule()`` re-anchors the pivot on the MOLECULE
    // (everything that's not residue ``ELC``) and zooms tight on it,
    // then pulls the camera back enough that the slabs remain in
    // frame as periphery.  Wired to the Focus-molecule toolbar
    // button -- the user clicks it whenever interaction feels off-
    // centre.  ``applyStructure`` does NOT call this automatically;
    // its default fit is a plain ``zoomTo`` showing the whole
    // structure so a fresh render always frames everything.
    // Indices of the "molecule" atoms (everything that isn't an
    // ELC electrode-slab residue).  Returns ``null`` when no slabs
    // are present — the camera ops then fall back to refit() /
    // setPivot() with no opts (= all atoms).  Shared between
    // focusMolecule + snapPivotToCenter so they always operate on
    // the same selection.
    function _moleculeIndices() {
        const rn = state.residue_names;
        if (!Array.isArray(rn) || rn.length === 0) return null;
        if (rn.indexOf("ELC") === -1) return null;
        const out = [];
        for (let i = 0; i < rn.length; i++) {
            if (rn[i] !== "ELC") out.push(i);
        }
        return out;
    }

    function focusMolecule() {
        if (!state.xyz || state.n_atoms === 0) return;
        const mol = _moleculeIndices();
        if (mol) {
            // Pivot + zoom-fit to the molecule (non-ELC atoms only).
            // Pull back so the slabs remain visible in the periphery;
            // without the pullback the slabs would be clipped or
            // behind the camera.
            _handle.refit({ indices: mol, pullback: 0.55 });
        } else {
            _handle.refit();
        }
    }

    // Snap the camera lookAt onto the structure centroid without
    // touching the zoom level.  ``handle.setPivot`` delegates to
    // 3Dmol's ``center()`` which translates the model so the
    // selection's centroid lands on the world origin (the rotation
    // pivot) — the camera distance stays where the user left it.
    // Used as a mousedown hook so every rotation drag pivots on
    // the structure regardless of any pan the user did before.
    function snapPivotToCenter() {
        if (!state.xyz || state.n_atoms === 0) return;
        _handle.setPivot({ indices: _moleculeIndices() || [] });
    }

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
                    `Anchor: #${a + 1} ${state.elements[a]}`;
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
                    `Anchors: #${a + 1} ${state.elements[a]} → ` +
                    `#${b + 1} ${state.elements[b]}`;
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
        if (rotateBtn) rotateBtn.disabled = locked || state.n_atoms === 0;
        const centerBtn = $("center-apply");
        if (centerBtn) centerBtn.disabled = locked || state.n_atoms === 0;
        const translateBtn = $("translate-apply");
        if (translateBtn) translateBtn.disabled = locked || state.n_atoms === 0;

        // Electrode: anchor count depends on mode.
        const elcBtn = $("elc-apply");
        const elcReadout = $("elc-anchor-readout");
        const mode = ($("elc-mode") || {}).value || "symmetric";
        if (elcBtn && elcReadout) {
            if (mode === "single") {
                if (sel.length === 1) {
                    elcBtn.disabled = locked;
                    elcReadout.textContent =
                        `Anchor: #${sel[0] + 1} ${state.elements[sel[0]]}.  ` +
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
                elcBtn.disabled = locked || state.n_atoms === 0;
                if (sel.length === 0) {
                    elcReadout.textContent =
                        "Pair mode: slabs at z = ±gap/2 around the origin.  "
                        + "Centre + pose the molecule first (Geom + Pose).";
                } else if (sel.length === 2) {
                    const [a, b] = sel;
                    elcReadout.textContent =
                        `Legacy mode: slabs flank #${a + 1} ${state.elements[a]} `
                        + `↔ #${b + 1} ${state.elements[b]} `
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
            // Phase 10 (workspace-contract.md §2): read dirty bit
            // off ws, not the legacy structureCanvas global.
            const cs = window.molbuilder
                    && window.molbuilder.workspace;
            const save = window.molbuilder
                    && window.molbuilder.structureSave;
            const targetPath = (save && typeof save.targetPath === "function")
                ? save.targetPath() : null;
            const savedAndClean = !!cs
                && typeof cs.isDirty === "function"
                && !cs.isDirty()
                && !!targetPath;
            sendBtn.disabled = locked
                || state.n_atoms === 0
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

    /**
     * Replace the modify-page IIFE state + 3Dmol embed model
     * from a server workspace payload.  Scope is intentionally
     * narrow: state.* fields, position-array parsing, embed
     * setStructure + refit, and UI refresh hooks.  No
     * cross-store work (canvas-state, selection store) lives
     * here — the workspace dispatcher's
     * ``_applyWorkspacePayload`` owns that and calls this
     * function as one of its steps (the modify hook).
     *
     * Callsites (all four currently):
     *   * dispatcher's ``_applyWorkspacePayload`` for every
     *     applyOp / applyPayload flow.
     *   * ``applyUndo`` — pops a history snapshot.
     *   * ``restoreModifyState`` — bfcache/navigation restore.
     */
    function applyStructure(r) {
        state.xyz           = r.xyz || "";
        state.elements      = Array.isArray(r.elements)      ? r.elements      : [];
        state.atom_names    = Array.isArray(r.atom_names)    ? r.atom_names    : [];
        state.residue_ids   = Array.isArray(r.residue_ids)   ? r.residue_ids   : [];
        state.residue_names = Array.isArray(r.residue_names) ? r.residue_names : [];
        state.chain_ids     = Array.isArray(r.chain_ids)     ? r.chain_ids     : [];
        state.title         = r.title || "";
        state.n_atoms       = Number(r.n_atoms || state.elements.length || 0);
        // Parse positions from the xyz string so the per-op anchor
        // readouts can show coordinates without an extra server
        // roundtrip.  Lines after the 2-line header are
        // ``<element>  <x>  <y>  <z>``; whitespace is forgiving.
        state.positions = [];
        if (state.xyz) {
            const lines = state.xyz.split("\n").slice(2);
            for (const line of lines) {
                const t = line.trim();
                if (!t) continue;
                const parts = t.split(/\s+/);
                if (parts.length < 4) continue;
                state.positions.push([
                    Number(parts[1]), Number(parts[2]), Number(parts[3]),
                ]);
                if (state.positions.length === state.n_atoms) break;
            }
        }
        // Defensive: a malformed XYZ payload (missing column, partial
        // line, etc.) could leave positions.length != n_atoms, which
        // would silently mis-attribute coordinates to atom indices in
        // the selection-info panel and the slab-placement math.  Log
        // loudly and clear so the next op fails fast rather than
        // emitting wrong physics.
        if (state.positions.length !== state.n_atoms) {
            console.warn(
                "XYZ parse anomaly: positions=" + state.positions.length
                + " but n_atoms=" + state.n_atoms
                + "; clearing positions array."
            );
            state.positions = [];
        }

        $("title-readout").textContent =
            state.title ? `${state.title} (${formula(state.elements)})`
                        : formula(state.elements);

        // Render via the embed.  setStructure replaces the model,
        // re-applies style + overlays (axes / labels / etc.), and
        // refits the camera.  Index labels + axes follow the
        // toggle-driven state below; setStructure leaves those
        // settings intact so they re-render against the new atoms.
        if (state.xyz) {
            _handle.setStructure({ xyz: state.xyz });
            // Style / labels / axes are owned by the standard knob
            // bar after #203; setStructure leaves the current settings
            // intact so they re-render against the new atoms.
            // Default fit shows the whole structure (atoms + slabs)
            // so the user always sees what's in the model after a
            // refresh.  The Focus-molecule toolbar button switches to
            // a molecule-anchored pivot for smooth zoom on the small
            // molecule when slabs are present.
            _handle.refit();
        }
        // No else-branch needed: applyStructure is only called with
        // an xyz from a successful /api/build/load response; the
        // empty-xyz path is unreachable in practice.  See #235.
        // Atom-level clicks + halo overlays are wired by the
        // viewer-adapter (lib/selection/viewer-adapter.js), which
        // re-arms ``setClickable`` on every render so a model swap
        // here doesn't drop the handler.

        // Cross-store sync (adoptAtoms + clearSelection + remap)
        // lives entirely in the workspace dispatcher's
        // ``_applyWorkspacePayload`` (Phase 5 single-sync-point
        // refactor; review cleanup 2026-06-08 finished the move).
        // ``applyStructure`` is the modify-tab IIFE's state + embed
        // updater only — it has NO knowledge of the selection store
        // or the canvas-state dirty bit.  Callers route through
        // ``ws.applyPayload(r, opts)`` or ``ws.applyOp(op, args)``;
        // those pipelines call this hook for the IIFE-state work
        // and own the cross-store work themselves.  Pre-cleanup,
        // applyStructure ALSO did adoptAtoms + clearSelection,
        // duplicating the dispatcher's logic and causing a double
        // ``clearSelection`` on the loadStructureText path.

        refreshSelectionUI();
        refreshUndoButton();
    }

    // formula() lives in static/lib/mol-format.js; loaded by the
    // template above.  Local alias keeps callers below readable.
    const formula = (window.molbuilder && window.molbuilder.fmt
                     ? window.molbuilder.fmt.formula
                     : (els) => (els && els.length ? els.join("") : "—"));

    // --------------------------------------------------------------- //
    //  Modify ops (M3).  Each op POSTs the current canonical state +  //
    //  op-specific args to /api/modify/<op>; on success the response   //
    //  IS the next canonical state, so we just feed it through         //
    //  applyStructure() and the UI updates atomically.                 //
    // --------------------------------------------------------------- //
    function currentStateBody() {
        // Bundle the canonical state for an /api/modify/* request.  We
        // ALWAYS send the full metadata bundle so the new structure
        // returned by the server preserves it (xyz alone would lose
        // atom_names / residue_ids -- per spec § 5).
        return {
            xyz:           state.xyz || "",
            atom_names:    state.atom_names,
            residue_ids:   state.residue_ids,
            residue_names: state.residue_names,
            chain_ids:     state.chain_ids,
            title:         state.title,
        };
    }

    // ----- Undo history ------------------------------------------- //
    // Snapshot of the canonical structure shape applyStructure() takes
    // -- same keys as an /api/modify/* response.  Scoped to electrode
    // (slab) ops ONLY: those are the ones the user wants to experiment
    // with and roll back.  Other ops (delete / add atom / orient /
    // rotate / translate / center) are committed immediately and
    // don't push history.  ``snapshotForHistory()`` is taken BEFORE
    // the network call; it's only pushed onto the stack AFTER a
    // successful response so failed ops don't burn an undo slot.
    function snapshotForHistory() {
        if (!state.xyz) return null;
        // Capture the selection store's atoms list + the user's
        // current selection so ``applyUndo`` can fully restore the
        // workspace — pre-fix the snapshot omitted both, so after
        // undo the selection store kept the post-op atoms while
        // the viewer showed the pre-op structure (deferred bug 3
        // from the multi-pass review; fix landed 2026-06-08).
        const _s = _selStore();
        return {
            xyz:           state.xyz,
            elements:      state.elements,
            atom_names:    state.atom_names,
            residue_ids:   state.residue_ids,
            residue_names: state.residue_names,
            chain_ids:     state.chain_ids,
            title:         state.title,
            n_atoms:       state.n_atoms,
            atoms:         _s ? _s.getState().atoms.slice() : [],
            selected:      _s ? _s.getState().selection.slice() : [],
        };
    }

    function refreshUndoButton() {
        const btn = $("undo-op");
        if (btn) btn.disabled = state.inFlight || state.history.length === 0;
    }

    function applyUndo() {
        if (!state.history.length) return;
        const prev = state.history.pop();
        // Route through the workspace dispatcher's canonical
        // ``applyPayload`` pipeline so the selection store's atoms
        // revert in lockstep with state.xyz + the embed model.
        // Pre-fix applyUndo called ``applyStructure(prev)`` directly,
        // which after the Phase 5 cleanup only updates the IIFE
        // state + embed — leaving the selection store with the
        // post-op atoms.  ``touchCanvas: true`` because undo of a
        // modifier op is itself a state-diverged-from-disk event.
        const ws = window.molbuilder && window.molbuilder.workspace;
        if (ws && typeof ws.applyPayload === "function") {
            ws.applyPayload(prev, { touchCanvas: true });
            // applyPayload does NOT restore selection from the
            // snapshot's saved indices (its remap step is for
            // server-emitted selection_remap, which an undo
            // doesn't have).  Set them explicitly.
            const _s = _selStore();
            if (_s && typeof _s.setSelection === "function"
                  && Array.isArray(prev.selected)) {
                _s.setSelection(prev.selected);
            }
        } else {
            applyStructure(prev);   // fallback (no dispatcher)
        }
        refreshUndoButton();
        setEditStatus(
            `Undid op (${prev.n_atoms} atoms restored).  ${state.history.length} step(s) left.`,
            "ok",
        );
    }

    function setEditStatus(msg, kind = null) {
        const el = $("edit-status");
        if (!el) return;
        el.textContent = msg;
        el.className = "muted" + (kind ? ` status-${kind}` : "");
    }

    async function postOp(path, extraBody, label) {
        // Modifier-button wrapper around ``ws.applyOp``.  Owns the
        // UI-level concerns the dispatcher deliberately stays out
        // of: the in-flight lock (prevents double-click double-
        // fire), the edit-status text, the AbortController for the
        // wedge-release path, and the selection-UI refresh.  The
        // dispatcher owns the HTTP fetch + cross-store state
        // replacement; this wrapper composes them with the button's
        // user-facing affordances.
        if (state.inFlight) return null;
        const ac = (typeof AbortController !== "undefined")
            ? new AbortController() : null;
        state.inFlight = true;
        state._inFlightAbort = ac;
        refreshSelectionUI();
        setEditStatus(`${label}…`);
        const op = path.replace(/^\/api\/modify\//, "");
        let r = null;
        try {
            try {
                r = await window.molbuilder.workspace.applyOp(op, extraBody);
            } catch (e) {
                if (e && e.name === "AbortError") {
                    setEditStatus(`${label} cancelled.`, "muted");
                    return null;
                }
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
            state._inFlightAbort = null;
            refreshSelectionUI();
        }
    }

    // Phase 6e seventh-review LANDMINE-3: expose the wedge-
    // release path.  Today only the bfcache-restore handler
    // calls it; a future Cancel button is a trivial UI
    // addition that wires here.
    state.abortInFlight = function () {
        const ac = state._inFlightAbort;
        if (ac) {
            try { ac.abort(); } catch (_) {}
        }
    };

    async function applyDelete() {
        const indices = selectedIndices();
        if (!indices.length) return;
        await postOp("/api/modify/delete", { indices }, "Deleted");
    }

    // ----- Geom subtab: rigid translate ops ----------------------- //
    // Both ops route through the shared /api/modify/translate
    // endpoint; only the body changes (recenter:true vs explicit
    // dx/dy/dz).  After the structure shifts, applyStructure() runs
    // viewer.zoomTo() so the camera re-fits the new bounding box --
    // there's no separate "re-fit camera" button anymore because
    // every coordinate-changing op already does the right thing.
    async function applyCenter() {
        if (!state.xyz) {
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
        if (!state.xyz) {
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
        const sel = selectedIndices();
        if (sel.length !== 1) {
            setEditStatus("Pick exactly one anchor atom first.", "error");
            return;
        }
        const anchor_index = sel[0];
        const element = ($("add-element").value || "H").trim();
        if (!element) {
            setEditStatus("Element required.", "error");
            return;
        }
        const offset = readAddOffset();
        await postOp(
            "/api/modify/add_atom",
            { element, anchor_index, offset },
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
        // The selection is the anchor pair.  Send sorted-ascending so
        // the renderer's "first anchor" semantic is reproducible from
        // the UI without exposing a "swap" affordance.  Tilted-pair
        // direction is determined by orient_along_axis (a0 -> a1).
        const sel = selectedIndices();
        if (sel.length !== 2) {
            setEditStatus("Pick exactly two anchor atoms first.", "error");
            return;
        }
        const axis  = getCheckedRadio("orient-axis") || "z";
        const angle = Number($("orient-angle").value);
        const center = $("orient-center").value || "midpoint";
        await postOp(
            "/api/modify/orient",
            { anchors: sel, axis, angle, center },
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
        // Bundle the shared fields both single + symmetric modes
        // need.  Returned as a plain object that postOp will merge
        // into currentStateBody().
        const m         = Number($("elc-m").value);
        const n         = Number($("elc-n").value);
        const layers    = Number($("elc-layers").value);
        return {
            element:    $("elc-element").value,
            plane:      getCheckedRadio("elc-plane") || "111",
            size:       [m, n, layers],
            orthogonal: $("elc-orthogonal").checked,
            offset:     [
                Number($("elc-dx").value),
                Number($("elc-dy").value),
            ],
        };
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
        const sel = selectedIndices();
        const mode = $("elc-mode").value;
        const common = readElcCommonBody();
        const gap = Number($("elc-gap").value);
        // Snapshot the pre-op state so a successful slab op can be
        // rolled back via Undo.  We only commit it to the history
        // stack after the response comes back ok -- a 400 / network
        // error must not consume an undo slot.
        const snap = snapshotForHistory();
        let r = null;
        if (mode === "single") {
            if (sel.length !== 1) {
                setEditStatus("Pick exactly one anchor for single mode.", "error");
                return;
            }
            const side = getCheckedRadio("elc-side") || "+z";
            r = await postOp(
                "/api/modify/electrode",
                Object.assign({}, common, {
                    anchor_index:     sel[0],
                    side:             side,
                    contact_distance: gap,
                }),
                `Added ${common.element}(${common.plane}) ${side}`,
            );
        } else {
            // Symmetric pair mode -- canonical placement is at the
            // world origin.  Slabs at z = ±gap/2, lateral xy on the
            // origin (plus the user's xy offset).  No anchor pick is
            // required; the user is expected to centre + pose the
            // molecule (Geom + Pose subtabs) before adding slabs.
            //
            // If the user has selected two atoms anyway, we forward
            // them as legacy anchor pair: the slabs centre on the
            // anchor-pair midpoint instead of the origin.  Useful for
            // un-centred structures.  Selecting one atom is
            // ambiguous -- treat it as user error so they explicitly
            // pick 0 or 2.
            if (sel.length === 1) {
                setEditStatus(
                    "Pair mode: select 0 atoms (origin-centred slabs)"
                    + " or 2 atoms (legacy anchor-pair midpoint).", "error");
                return;
            }
            const extra = { gap: gap };
            if (sel.length === 2) {
                const [i0, i1] = sel;
                const z0 = state.positions[i0][2];
                const z1 = state.positions[i1][2];
                const a_top = z0 >= z1 ? i0 : i1;
                const a_bot = z0 >= z1 ? i1 : i0;
                extra.anchors = [a_top, a_bot];
            }
            r = await postOp(
                "/api/modify/symmetric_electrodes",
                Object.assign({}, common, extra),
                `Added ${common.element}(${common.plane}) pair`,
            );
        }
        if (r && snap) {
            state.history.push(snap);
            if (state.history.length > HISTORY_MAX) state.history.shift();
            refreshUndoButton();
        }
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
        // Phase 10 — read dirty bit through ws.* (workspace-contract.md §2).
        const cs = window.molbuilder
                && window.molbuilder.workspace;
        const save = window.molbuilder
                && window.molbuilder.structureSave;
        const targetPath = (save && typeof save.targetPath === "function")
            ? save.targetPath() : null;
        if (!cs || !targetPath || cs.isDirty()) {
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
        // Canvas-state subscriber — the Send-to-Optimization button's
        // enable rule depends on canvas-state.isDirty() + getLastSavedTo(),
        // neither of which fires through the selection store.  Without
        // this subscription the Send button wouldn't re-enable after
        // a successful Save (or re-disable after a modifier op flipped
        // the dirty bit).  Task #294, 2026-06-08.
        // Phase 10 — subscribe to ws.* notifications (contract §2.1).
        // ws.subscribe fires on EVERY workspace-state change (canvas
        // + selection both), so the Send button stays in lockstep
        // with both Save and modifier ops.
        const _cs = window.molbuilder
                 && window.molbuilder.workspace;
        if (_cs && typeof _cs.subscribe === "function") {
            _cs.subscribe(() => refreshSelectionUI());
        }

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
        $("clear-selection").addEventListener("click", () => {
            // Delegate to the selection store; its subscriber will
            // re-run refreshSelectionUI automatically.
            clearStoreSelection();
        });
        const undoBtn = $("undo-op");
        if (undoBtn) undoBtn.addEventListener("click", applyUndo);

        const focusBtn = $("focus-molecule");
        if (focusBtn) focusBtn.addEventListener("click", focusMolecule);

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
        window.addEventListener("pagehide", saveModifyState);
    });

    // ----- State persistence across tab navigation -------------------- //
    // Build, Watch, and Modify are separate Flask routes (full page
    // reloads on every nav).  JS closure state -- the loaded
    // structure, the selection, the 3Dmol camera -- gets destroyed
    // when the user clicks /watch and again when they come back.
    // sessionStorage survives same-tab navigation so we use it as the
    // bridge.  This mirrors the pattern Build's viewer.js already
    // uses for form values (saveFormState / restoreFormState).
    //
    // Stored payload shape (key MODIFY_STATE_KEY):
    //   {
    //     v: 1,
    //     saved_at: <iso8601>,
    //     xyz, elements, atom_names, residue_ids, residue_names,
    //     chain_ids, title, n_atoms,
    //     selected: [...],   // sorted atom indices
    //     camera: viewer.getView(),
    //     show_axes:    bool,
    //     show_indices: bool,
    //     rep:          string,
    //   }
    //
    // We don't expire by saved_at -- sessionStorage already clears
    // on browser close, which is the right "fresh start" boundary.

    const MODIFY_STATE_KEY = "modify-state";
    const STATE_SCHEMA_VERSION = 1;

    function saveModifyState() {
        if (!state.xyz) return;     // nothing to save
        // Phase 8 follow-through (2026-06-08): when the workspace
        // dispatcher is mounted it owns the unified
        // ``molbuilder.workspace.v1`` mirror — which contains every
        // field this legacy mirror would store (xyz + metadata via
        // structure.atoms[], selected via selection.indices, camera
        // + chrome via view.getState).  Skipping the legacy write
        // here retires the triple-mirror overhead.
        if (window.molbuilder && window.molbuilder.workspace) return;
        // #235 follow-up: getCamera returns the documented opaque
        // CameraState {_viewer, _version, data} per § 3.13.
        // Round-trips cleanly through sessionStorage; restoreModifyState
        // hands it back via setCamera which no-ops on a future
        // _version bump (forward-compat).
        const camera = _handle.getCamera();
        const _s = _selStore();
        const sourceFile = _s ? (_s.getState().sourceFile || null) : null;
        // Persist atoms ONLY when canvas-state is dirty (the user did
        // modifier ops since the last load/save) OR when there's no
        // source file at all (generator output — in-memory IS the
        // source of truth, no disk to read from).  Clean + sourceFile
        // means disk == memory at save time, AND an external editor
        // could have replaced the file while the user was away — so
        // on restore the user wants the LATEST disk bytes, not the
        // stale in-memory snapshot.  Test pin:
        // TestModifySecondVisitExternalChange::test_external_xyz_
        // replacement_reloads_atom_list_on_revisit.
        let canvasDirty = false;
        try {
            // Phase 10 — dirty bit via ws.* (contract §2).
            const cs = window.molbuilder
                    && window.molbuilder.workspace;
            if (cs && typeof cs.isDirty === "function") {
                canvasDirty = !!cs.isDirty();
            }
        } catch (_) { /* default false */ }
        const shouldPersistAtoms = canvasDirty || !sourceFile;
        const payload = {
            v: STATE_SCHEMA_VERSION,
            saved_at: new Date().toISOString(),
            // Source-file path is what the selection store keys off;
            // saving it lets restoreModifyState rehydrate the store
            // (via adoptSession) so the panel + atom list re-sync to
            // the same structure the viewer is showing.  Without this
            // the post-restore store would have sourceFile=null and
            // atoms=[] while the 3D viewer renders the restored
            // structure -- a UI desync the user sees as "empty panel,
            // populated viewer".
            source_file:   sourceFile,
            xyz:           state.xyz,
            elements:      state.elements,
            atom_names:    state.atom_names,
            residue_ids:   state.residue_ids,
            residue_names: state.residue_names,
            chain_ids:     state.chain_ids,
            title:         state.title,
            n_atoms:       state.n_atoms,
            selected:      selectedIndices(),
            // BOMB-0 follow-up (2026-06-07): snapshot the selection
            // store's atoms list — the post-op shape with regions +
            // is_frozen + atom_name + residue_name + chain_id.  The
            // restore path adopts this directly instead of re-fetching
            // from disk via adoptSession's /api/selection/atoms call,
            // which would return the pre-op file contents (the disk
            // hasn't been re-saved after the modifier op, so the disk
            // atoms are stale relative to state.xyz).
            //
            // Conditional persistence (2026-06-07 audit): only carry
            // atoms forward when memory is more authoritative than
            // disk.  See ``shouldPersistAtoms`` above.  When omitted,
            // the restore path falls back to ``adoptSession``'s
            // disk-fetch branch and picks up any external file
            // change that happened while the user was away.
            atoms:         shouldPersistAtoms && _s
                              ? _s.getState().atoms
                              : undefined,
            camera:        camera,
            // Read the current chrome state via the embed's getters
            // (D3 — added 2026-06-04).  show_axes is "axes are
            // enabled" which on the handle = getAxes() returning a
            // truthy normalised object; show_indices is the same
            // shape for getLabels().  rep is the current
            // representation per getStyle().
            show_axes:     _handle.getAxes() !== null,
            show_indices:  _handle.getLabels() !== null,
            rep:           (_handle.getStyle() || {}).rep || "stick",
        };
        try {
            sessionStorage.setItem(
                MODIFY_STATE_KEY,
                JSON.stringify(payload),
            );
        } catch (e) {
            // QuotaExceededError on a >5 MB structure, or storage
            // disabled (private mode in some browsers).  Skip without
            // crashing -- the user simply loses persistence.
            console.warn("modify: could not save state:", e && e.message);
        }
    }

    /**
     * Translate the workspace dispatcher's canonical snapshot
     * (``ws.readPersistedSnapshot()``) into the legacy modify-state
     * shape ``restoreModifyState`` consumes.  Phase 8 follow-through:
     * lets the dispatcher own persistence while
     * ``restoreModifyState`` keeps its existing applyStructure +
     * adoptSession + applyState pipeline unchanged.  Returns null
     * if the snapshot doesn't carry enough data to restore.
     */
    function _modifyShapeFromDispatcherSnapshot(snap) {
        if (!snap || !snap.state) return null;
        const s = snap.state;
        const struct = s.structure;
        if (!struct || !struct.text) return null;
        const atoms = Array.isArray(struct.atoms) ? struct.atoms : [];
        const view  = s.view || {};
        const src   = s.source || {};
        const sel   = s.selection || {};
        // cd9655e dirty-gate applied at restore-time (the dispatcher
        // snapshot itself carries atoms unconditionally; this
        // translator decides whether the saved atoms are trustworthy
        // or whether to refetch from disk).  When canvas is clean
        // AND has a source file on disk, the disk is authoritative:
        // an external editor could have replaced the file while the
        // user was away.  Pass ``atoms === undefined`` so
        // adoptSession falls back to its disk-fetch branch.  When
        // dirty (modifier ops since last save) OR source-less
        // (generator output, never saved), in-memory IS truth —
        // install the saved atoms directly.
        const shouldUseSavedAtoms = !!s.dirty || !src.file;
        return {
            v:             STATE_SCHEMA_VERSION,
            source_file:   src.file || null,
            xyz:           struct.text,
            elements:      atoms.map(a => a.element),
            atom_names:    atoms.map(a => a.atom_name || ""),
            residue_ids:   atoms.map(
                a => a.residue_id != null ? a.residue_id : 0),
            residue_names: atoms.map(a => a.residue_name || ""),
            chain_ids:     atoms.map(a => a.chain_id || ""),
            title:         struct.title || "",
            n_atoms:       struct.n_atoms,
            selected:      Array.isArray(sel.indices) ? sel.indices : [],
            atoms:         shouldUseSavedAtoms ? atoms : undefined,
            camera:        view.camera || null,
            show_axes:     !!view.axes,
            show_indices:  !!view.labels,
            rep:           (view.style && view.style.rep) || "stick",
        };
    }

    function restoreModifyState() {
        let saved = null;
        // Phase 8 follow-through (2026-06-08): prefer the workspace
        // dispatcher's unified snapshot under
        // ``molbuilder.workspace.v1``.  Fall back to the legacy
        // modify-state key for users still mid-session when this
        // rolled out (single session boundary; no longer-term
        // back-compat needed once browser tabs close).
        const ws = window.molbuilder && window.molbuilder.workspace;
        if (ws && typeof ws.readPersistedSnapshot === "function") {
            const snap = ws.readPersistedSnapshot();
            saved = _modifyShapeFromDispatcherSnapshot(snap);
        }
        if (!saved) {
            try {
                saved = JSON.parse(sessionStorage.getItem(MODIFY_STATE_KEY) || "null");
            } catch (_e) {
                return;
            }
        }
        if (!saved || saved.v !== STATE_SCHEMA_VERSION) return;
        if (!saved.xyz) return;
        // Restore the chrome state via the embed batch runner
        // (D4 — applyState canonical-order restore; D3 getX/setX
        // round-trip pattern).  Only the fields the snapshot
        // actually carries are passed; undefined fields skip via
        // the applyState contract.  applyStructure() below handles
        // the structure swap separately because it does additional
        // /modify-side bookkeeping (atom list, info panel) that
        // applyState({structure:...}) wouldn't touch.
        _handle.applyState({
            style: typeof saved.rep === "string"
                       ? { rep: saved.rep } : undefined,
            axes:  typeof saved.show_axes === "boolean"
                       ? saved.show_axes : undefined,
            labels: typeof saved.show_indices === "boolean"
                       ? saved.show_indices : undefined,
        });
        // Feed the saved structure through the existing
        // applyStructure() path so the atom list, viewer, and info
        // panel all re-render via the same code as a fresh load.
        applyStructure({
            xyz:           saved.xyz,
            elements:      saved.elements,
            atom_names:    saved.atom_names,
            residue_ids:   saved.residue_ids,
            residue_names: saved.residue_names,
            chain_ids:     saved.chain_ids,
            title:         saved.title,
            n_atoms:       saved.n_atoms,
        });
        // Hydrate canvas-state so Save / Load / Generate know
        // there's something in the workspace.  Without this a
        // session-restored structure leaves canvas-state empty:
        // Save reports "No structure to save" and Load doesn't
        // fire the warning modal even after the user makes edits.
        //
        // Skip the setStructure call when canvas-state's bytes
        // already match the restored xyz (BOMB-2 invariant).
        // Post-Phase-8 (2026-06-08), both canvas-state and this
        // function read from the dispatcher's single
        // ``molbuilder.workspace.v1`` snapshot — bytes always
        // match on /molbuilder when the dispatcher is mounted,
        // so this branch is normally a no-op.  The check is
        // still load-bearing for:
        //   * test isolation contexts where canvas-state restored
        //     from its legacy structure_canvas key.
        //   * mid-Phase-8-rollout users with stale legacy keys.
        //   * future bfcache restore paths.
        // Calling setStructure unconditionally would reset
        // canvas-state's dirty bit — a user with unsaved
        // modifier-op edits would silently see them marked
        // clean, and the next Load wouldn't fire the warning
        // modal.  The matrix:
        //   * canvas empty                    → setStructure.
        //   * canvas text matches saved.xyz   → DO NOT touch.
        //   * canvas text differs             → setStructure
        //     (modify-state-shaped saved is authoritative on
        //     a divergence; this shouldn't happen with the
        //     unified mirror but the branch is defensive).
        try {
            // Phase 10 — install via ws.* (contract §3).  Routes
            // through the same canvas-state internally; consumer
            // surface is now unified.
            const cs = window.molbuilder
                    && window.molbuilder.workspace;
            if (cs && typeof cs.installStructure === "function"
                   && saved.xyz) {
                const existing = (typeof cs.getStructure === "function")
                    ? cs.getStructure() : null;
                const sameBytes = existing
                    && existing.text === saved.xyz;
                if (!sameBytes) {
                    cs.installStructure(
                        { source_format: "xyz", text: saved.xyz },
                        { kind: saved.source_file ? "file" : "load",
                          file: saved.source_file || null,
                          generator_input: null }
                    );
                }
            }
        } catch (_) { /* canvas-state optional — UX unaffected */ }
        // Rehydrate the store atomically.  We can't call
        // ``setSourceFile`` here because that path would re-issue
        // ``GET /api/files/read`` + a viewer model swap -- we just
        // applied the structure from sessionStorage, so re-loading
        // would discard the camera we're about to restore AND waste
        // a round-trip.  ``adoptSession`` takes the same {sourceFile,
        // selection} pair, skips the viewer load, and re-fetches the
        // atom list from the server so any sidecar updates done
        // elsewhere since the snapshot are reflected.
        const _s = _selStore();
        if (_s && typeof _s.adoptSession === "function") {
            const validSelection = Array.isArray(saved.selected)
                ? saved.selected.filter(
                    (i) => Number.isInteger(i)
                        && i >= 0
                        && i < state.n_atoms
                )
                : [];
            // BOMB-0 follow-up (2026-06-07): pass saved.atoms when
            // present so adoptSession installs the in-memory post-op
            // atoms list directly instead of going through the disk
            // fetch.  Disk reads are stale relative to state.xyz
            // until the user saves; without this, a Modify-tab
            // session that did a Delete + navigated away would come
            // back showing the PRE-delete atom list.  When atoms is
            // absent (cross-tab handoff, pre-fix sessions still in
            // storage), adoptSession falls back to the disk fetch.
            _s.adoptSession({
                sourceFile: saved.source_file || null,
                selection:  validSelection,
                atoms:      Array.isArray(saved.atoms)
                              ? saved.atoms : undefined,
            });
        }
        // Restore the camera AFTER applyStructure so it doesn't
        // fight the refit() that lives inside.  Camera goes
        // through applyState too for consistency, though a bare
        // setCamera would be equivalent for this one field.
        if (saved.camera) {
            _handle.applyState({ camera: saved.camera });
        }
        setStatus(
            `Restored ${state.n_atoms}-atom structure (${saved.title || "unnamed"}).`,
            "ok",
        );
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
        getViewer:   () => _handle._viewer3dmol(),
        getSelected: () => selectedIndices(),
        getNAtoms:   () => state.n_atoms,
        // getState is read-only (we expose the raw state object,
        // tests must not mutate it).  Used by Geom-subtab tests to
        // probe positions after a translate / center op without
        // round-tripping through xyz parsing.
        getState:    () => state,
    };

    // Public loader for the Projects sidebar's onLoad callback (and
    // any future tab-coordination code).  Reuses /api/build/load's
    // JSON path so we don't need a browser File object.
    window.molbuilder = window.molbuilder || {};
    // Expose the embed HANDLE (NOT a raw 3Dmol viewer) under the
    // per-tab namespace.  Every consumer drives the viewer via the
    // declarative handle API — `modify.handle.setOverlays(...)`,
    // `modify.handle.setPick(...)`, etc.  /modify is a single-mount
    // page (no within-page remount), so the handle reference is
    // stable for the page lifetime.  The legacy raw-viewer slot
    // ``modify.viewer`` was dropped in #239 (no Layer-1 self-
    // protection: a torn-down 3Dmol viewer crashes on access;
    // handle methods short-circuit on state.disposed).
    window.molbuilder.modify = window.molbuilder.modify || {};
    window.molbuilder.modify.handle = _handle;
    // Workspace-state Phase 5 (2026-06-07): expose
    // ``currentStateBody`` + ``applyStructure`` on the modify
    // namespace so the workspace dispatcher's _applyWorkspacePayload
    // pipeline can call them synchronously.  Pre-Phase-5 these
    // were IIFE-private and only reachable via the modifier-button
    // wiring; exposing them lets ``ws.applyOp`` own the
    // fetch+state-replacement pipeline end-to-end.
    window.molbuilder.modify.currentStateBody = currentStateBody;
    window.molbuilder.modify.applyStructure   = applyStructure;
    // Load a structure text blob (XYZ or PDB) via /api/build/load,
    // which sniffs the format from the filename + content.  The
    // function is named ``loadStructureText`` (renamed 2026-05-22
    // from the misleading legacy ``loadXyzText``) because it
    // genuinely accepts both formats; the field name lied about
    // its capability and caused real bugs (see design.md decision
    // log for the rename rationale).
    // Back-compat alias.  The actual fetch + applyPayload pipeline
    // moved into ``ws.loadFromText`` on 2026-06-08 so the
    // dispatcher owns every "parse text → install workspace"
    // entry point.  This alias stays so existing consumers
    // (selection-bootstrap, the Sources-card generators' injected
    // ``viewerLoader``, the page-mount test hook) keep working
    // without coordinating a rename.  The page-mount test still
    // waits on ``typeof window.molbuilder.loadStructureText
    // === 'function'`` — that contract holds.
    window.molbuilder.loadStructureText = async function (text, filename) {
        setStatus(`Loading ${filename}…`);
        let r;
        try {
            r = await window.molbuilder.workspace.loadFromText(
                text, filename);
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
    // Module-init contract: register the modify handle with the
    // runtime so ``selection-bootstrap`` can ``whenReady("modify.handle")``
    // (selection-bootstrap line ~359).  Also register
    // ``modify.applyUndo`` for the workspace dispatcher's
    // ``ws.undo()`` method (dispatcher.js).
    //
    // Review cleanup 2026-06-08: dropped two registrations that had
    // no consumers — ``modify.postOp`` (ws.applyOp is self-sufficient
    // since Phase 5; doesn't whenReady the postOp anymore) and
    // ``modify.loadStructureText`` (no whenReady consumer; the global
    // ``window.molbuilder.loadStructureText`` is still used directly
    // by selection-bootstrap + the selection store's setLoader).
    if (window.molbuilder.runtime
        && typeof window.molbuilder.runtime.register === "function") {
        window.molbuilder.runtime.register(
            "modify.handle", window.molbuilder.modify.handle);
        window.molbuilder.runtime.register("modify.applyUndo", applyUndo);
    }

    // Selection-driven measurement readout: the panel calls this
    // provider on every render to compute xyz / distance / angle
    // for the current selection.  ``state.positions`` is parsed
    // from the canonical XYZ payload above (cleared to [] on parse
    // anomaly); returning an empty array signals "positions not
    // available yet" and the panel hides the readout.  See
    // ``lib/selection/measurements.js`` for the shape and
    // ``lib/selection-panel.js`` ``renderMeasurement`` for the
    // consumer.
    window.molbuilder.selection = window.molbuilder.selection || {};
    window.molbuilder.selection.positionsProvider = function () {
        return state.positions || [];
    };
})();
