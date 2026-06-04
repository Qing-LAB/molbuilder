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
 * Spec: docs/tabs/modify.md (M2-M4 + Phase 1 cross-tab
 * persistence as of 2026-05-09).
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // --------------------------------------------------------------- //
    //  Module state.  Single canonical structure (xyz string + parsed  //
    //  atom metadata).  Selection state is NOT owned here -- it       //
    //  lives in the singleton selection store at                      //
    //  ``window.molbuilder.selection.store`` and is read on demand    //
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

    function _selStore() {
        return (window.molbuilder
                && window.molbuilder.selection
                && window.molbuilder.selection.store)
               ? window.molbuilder.selection.store : null;
    }
    function selectedIndices() {
        const s = _selStore();
        return s ? s.getState().selection.slice() : [];
    }
    function clearStoreSelection() {
        const s = _selStore();
        if (s) s.clearSelection();
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
        card:    { title: "Structure", showInfoLine: false,
                   height: "100%" },
        export:  { defaultName: "modify" },
        onError(err) {
            // Surface viewer errors via the same status pattern
            // /modify already uses (preflight / generator panels).
            try { console.warn("[modify.viewer]", err.code, err.message); }
            catch (_) {}
        },
    });
    // /modify viewer.js #235 follow-up: production code no longer
    // reaches for _viewer3dmol().  The selection-store viewer-
    // adapter (#229) uses handle.setOverlays + handle.setPick
    // exclusively; focusMolecule / snapPivotToCenter (#235) flow
    // through handle.refit({indices, pullback}) + handle.setPivot.
    // The capture below is ONLY for the Playwright test surface
    // (window.__molbuilder_modify_test.getViewer()) which probes
    // raw 3Dmol state (atom.serial, atom.clickable) for invariant
    // checks tests/test_modify_e2e.py:443+ cannot do via the
    // handle.  Production registry slot is window.molbuilder.
    // modify.handle (handle, NOT raw viewer) per #239.
    const viewer = _handle._viewer3dmol();

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
    //  to ``store.toggleAtom`` and draws the halo overlay; the panel   //
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
        if (sendBtn) sendBtn.disabled = locked || state.n_atoms === 0;
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
        return {
            xyz:           state.xyz,
            elements:      state.elements,
            atom_names:    state.atom_names,
            residue_ids:   state.residue_ids,
            residue_names: state.residue_names,
            chain_ids:     state.chain_ids,
            title:         state.title,
            n_atoms:       state.n_atoms,
        };
    }

    function refreshUndoButton() {
        const btn = $("undo-op");
        if (btn) btn.disabled = state.inFlight || state.history.length === 0;
    }

    function applyUndo() {
        if (!state.history.length) return;
        const prev = state.history.pop();
        applyStructure(prev);
        // applyStructure does NOT touch state.history.  The selection
        // store clears its own state when ``store.setSourceFile`` is
        // called on a new path -- after an undo we stay on the same
        // path so the store's selection persists; ops that need a
        // selection re-check via refreshSelectionUI on the next
        // store fire.
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
        // Drop the call entirely if a prior op is still in flight.
        // Without this guard, a user double-click on Apply would fire
        // two parallel fetches; the second is wasted work AND could
        // race with the first (e.g. add_atom twice with stale xyz
        // before the first response updates state).  Buttons are
        // also disabled while in-flight so the visible UI matches.
        if (state.inFlight) return null;
        state.inFlight = true;
        refreshSelectionUI();    // disable Delete/Add buttons during fetch
        setEditStatus(`${label}…`);
        const body = Object.assign(currentStateBody(), extraBody);
        let r = null;
        try {
            try {
                r = await fetch(path, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                }).then((x) => x.json());
            } catch (e) {
                setEditStatus(`Network error: ${e.message}`, "error");
                return null;
            }
            if (!r || !r.ok) {
                setEditStatus(r?.error || `${label} failed.`, "error");
                return null;
            }
            // Success: replace the state with the new structure and
            // clear the per-op selection (atom indices have shifted
            // after delete).  applyStructure() will rebuild the list
            // and call refreshSelectionUI which re-enables buttons.
            applyStructure(r);
            setEditStatus(
                r.issues && r.issues.length
                    ? `${label}: ${r.n_atoms} atoms, ${r.issues.length} issue(s).`
                    : `${label}: ${r.n_atoms} atoms.`,
                "ok",
            );
            return r;
        } finally {
            // Always release the lock so a transient error can't wedge
            // the UI permanently.  The selection-UI refresh in
            // applyStructure already ran on the success path; on the
            // error path we run it explicitly to flip buttons back.
            state.inFlight = false;
            refreshSelectionUI();
        }
    }

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
        // Persist the current structure under the same key Phase 1
        // uses for cross-tab navigation, then navigate to the Build
        // tab.  Build's restoreStructureState() picks it up
        // identically to a fresh build.  We bundle the response
        // shape applyStructureResult() expects so the Build tab
        // doesn't need a separate code path.
        if (!state.xyz || !state.n_atoms) {
            setEditStatus("Nothing to send -- load a structure first.", "error");
            return;
        }
        const payload = {
            v: 1,
            saved_at: new Date().toISOString(),
            response: {
                xyz:           state.xyz,
                pdb:           "",                  // Build computes if needed
                title:         state.title || "modify-handoff",
                n_atoms:       state.n_atoms,
                n_residues:    null,                // unknown here; harmless
                summary:       `${state.n_atoms} atoms (from Modify tab)`,
                elements:      state.elements,
                atom_names:    state.atom_names,
                residue_ids:   state.residue_ids,
                residue_names: state.residue_names,
                chain_ids:     state.chain_ids,
                source_format: "xyz",
            },
            camera: null,        // Build owns its own camera
        };
        try {
            sessionStorage.setItem(
                "builder-structure",
                JSON.stringify(payload),
            );
        } catch (e) {
            setEditStatus(
                `Could not stage handoff: ${e && e.message}`, "error");
            return;
        }
        // Save Modify's own state so a back-button trip preserves
        // the source structure too.
        try { saveModifyState(); } catch (_e) { /* ok if not yet wired */ }
        window.location.href = "/";
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

        // Re-anchor the rotation pivot on the FIRST drag-confirming
        // mousemove of every plain left-button gesture.  3Dmol's
        // pivot is the camera lookAt; ctrl/shift-pan moves the
        // lookAt, so a follow-up rotation would orbit the molecule
        // eccentrically.  Snapping back to the structure centroid
        // before rotation kicks in fixes that.
        //
        // We DON'T snap on raw mousedown -- that would visibly jump
        // the camera every time the user clicks an atom to select it
        // (mousedown without drag).  Instead we wait for a movement
        // > DRAG_THRESHOLD_PX from the press point; at that point
        // the gesture has committed to "drag", so a snap is invisible
        // (the user is already moving the camera).  Modifier keys
        // (ctrl / shift / alt) signal pan / zoom and must NOT snap.
        const DRAG_THRESHOLD_PX = 4;
        const viewerEl = $("viewer");
        if (viewerEl) {
            let pressX = null, pressY = null, pressNoMods = false;
            let snapped = false;
            viewerEl.addEventListener("mousedown", (e) => {
                if (e.button !== 0) {
                    pressNoMods = false;
                    return;
                }
                pressX = e.clientX;
                pressY = e.clientY;
                pressNoMods = !(e.ctrlKey || e.shiftKey || e.altKey);
                snapped = false;
            });
            viewerEl.addEventListener("mousemove", (e) => {
                if (!pressNoMods || snapped) return;
                if (pressX === null || pressY === null) return;
                const dx = e.clientX - pressX;
                const dy = e.clientY - pressY;
                if (dx * dx + dy * dy >= DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) {
                    snapPivotToCenter();
                    snapped = true;
                }
            });
            viewerEl.addEventListener("mouseup", () => {
                pressX = pressY = null;
                pressNoMods = false;
            });
            viewerEl.addEventListener("mouseleave", () => {
                pressX = pressY = null;
                pressNoMods = false;
            });
        }

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
        // #235 follow-up: getCamera returns the documented opaque
        // CameraState {_viewer, _version, data} per § 3.13.
        // Round-trips cleanly through sessionStorage; restoreModifyState
        // hands it back via setCamera which no-ops on a future
        // _version bump (forward-compat).
        const camera = _handle.getCamera();
        const _s = _selStore();
        const sourceFile = _s ? (_s.getState().sourceFile || null) : null;
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
            camera:        camera,
            // show_axes / show_indices / rep are no longer persisted
            // here -- the standard knob bar owns those controls and
            // they will round-trip via handle.getCamera/setCamera
            // once setKnobs persistence lands.  Retained as historical
            // payload fields so an older snapshot still deserialises
            // cleanly during the transition.
            show_axes:     true,
            show_indices:  false,
            rep:           "stick",
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

    function restoreModifyState() {
        let saved = null;
        try {
            saved = JSON.parse(sessionStorage.getItem(MODIFY_STATE_KEY) || "null");
        } catch (_e) {
            return;
        }
        if (!saved || saved.v !== STATE_SCHEMA_VERSION) return;
        if (!saved.xyz) return;
        // Style / labels / axes are owned by the knob bar after #203
        // and don't need restoring here -- the standard chrome shows
        // the current state directly.  Saved fields (rep / show_axes
        // / show_indices) are kept in the payload for older snapshots
        // but no longer drive any input element.
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
            _s.adoptSession({
                sourceFile: saved.source_file || null,
                selection:  validSelection,
            });
        }
        // Restore the camera last so it doesn't fight refit() inside
        // applyStructure.  #235 follow-up: setCamera handles the
        // CameraState blob format AND silently no-ops on a future
        // _version bump (forward-compat).  Legacy snapshots that
        // saved a raw 3Dmol view array (#229 pre-migration) are
        // ignored — setCamera rejects non-object inputs.
        if (saved.camera) {
            _handle.setCamera(saved.camera);
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
        getViewer:   () => viewer,
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
    // Load a structure text blob (XYZ or PDB) via /api/build/load,
    // which sniffs the format from the filename + content.  The
    // function is named ``loadStructureText`` (renamed 2026-05-22
    // from the misleading legacy ``loadXyzText``) because it
    // genuinely accepts both formats; the field name lied about
    // its capability and caused real bugs (see design.md decision
    // log for the rename rationale).
    window.molbuilder.loadStructureText = async function (text, filename) {
        setStatus(`Loading ${filename}…`);
        let r;
        try {
            const resp = await fetch("/api/build/load", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text: text, filename: filename}),
            });
            r = await resp.json();
        } catch (e) {
            // Throwing instead of silently returning lets callers
            // (e.g. selection.store._loadViewer) treat the load as a
            // failure rather than mistakenly continuing to populate
            // panels for a structure the viewer never rendered.
            setStatus("Network error: " + e.message, "error");
            throw new Error("Network error loading " + filename
                          + ": " + e.message);
        }
        if (!r.ok) {
            const msg = r.error || "Load failed.";
            setStatus(msg, "error");
            throw new Error(msg);
        }
        applyStructure(r);
        const fmt = (r.source_format || "structure").toUpperCase();
        setStatus(
            `Loaded ${r.n_atoms}-atom ${fmt} from ${filename}.`,
            "ok",
        );
    };
    // Module-init contract: register the modify handle + loader
    // with the runtime so consumers can ``whenReady("modify.handle")``
    // or ``whenReady("modify.loadStructureText")``.  See design.md.
    if (window.molbuilder.runtime
        && typeof window.molbuilder.runtime.register === "function") {
        window.molbuilder.runtime.register(
            "modify.handle", window.molbuilder.modify.handle);
        window.molbuilder.runtime.register(
            "modify.loadStructureText",
            window.molbuilder.loadStructureText);
    }
})();
