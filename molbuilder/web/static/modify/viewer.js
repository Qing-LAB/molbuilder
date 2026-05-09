/* molbuilder Modify tab — M2 (read-only inspection).
 *
 * Three-pane UI (atom list ↔ 3Dmol viewer ↔ edit panel placeholder):
 *
 *   * the atom list on the left mirrors the structure's atoms;
 *     clicking a row highlights the atom in the viewer.
 *   * the viewer in the middle renders the structure via 3Dmol; clicking
 *     an atom highlights its row in the list.
 *   * the edit panel on the right is a placeholder for M3-M5; M2 only
 *     shows the current selection.
 *
 * Backend dependency for M2: ``POST /api/build/load`` (already exists)
 * for file upload.  No /api/modify/* routes yet -- those land in M3
 * with the actual edit operations.
 *
 * Spec: docs/spec/modify-tab.md.
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // --------------------------------------------------------------- //
    //  Module state.  Single canonical structure (xyz string + parsed  //
    //  atom metadata) plus a per-atom selected/highlighted bookkeeping //
    //  set.  Multi-select is shift-click; M4 will use the selection as //
    //  an anchor pair for orient-along-z.                              //
    // --------------------------------------------------------------- //
    const state = {
        xyz: null,
        elements: [],          // ["C", "H", ...]
        positions: [],         // [[x, y, z], ...]   parsed from xyz
        atom_names: [],        // ["CA", "HB1", ...] (or [] if absent)
        residue_ids: [],       // [1, 1, 2, ...]   (or [] if absent)
        residue_names: [],     // ["MOL", ...]     (or [] if absent)
        title: "",
        n_atoms: 0,
        selected: new Set(),   // atom indices
        inFlight: false,       // true while an /api/modify/* fetch is open
    };

    const HIGHLIGHT_COLOR = "#fbbf24";       // amber, matches --warning
    const HIGHLIGHT_RADIUS = 0.55;           // ~vdW for a sulfur in Å

    // --------------------------------------------------------------- //
    //  3Dmol viewer.                                                   //
    // --------------------------------------------------------------- //
    const viewer = $3Dmol.createViewer("viewer", {
        backgroundColor: "white",
        defaultcolors:   $3Dmol.elementColors.Jmol,
    });

    function styleSpec() {
        const rep = $("rep").value;
        return window.molbuilder?.style?.spec
            ? window.molbuilder.style.spec({ rep, scale: 1.0 })
            : { stick: { radius: 0.13 } };
    }

    function applyStyle() {
        viewer.setStyle({}, styleSpec());
        // Re-paint highlight overlays since setStyle clears them.
        renderHighlights();
        renderIndexLabels();
        // Axes are independent shapes; setStyle doesn't disturb them
        // but we redraw whenever the structure changes.  Render once
        // at the end so partial updates aren't visible.
        drawAxes();
    }

    function renderHighlights() {
        // Re-apply a sphere overlay on each currently-selected atom.
        // Calling setStyle({serial: i}, {...}, true) ADDS the spec
        // alongside the base style instead of replacing it.
        state.selected.forEach((idx) => {
            viewer.setStyle(
                { serial: idx },
                {
                    sphere: {
                        scale: HIGHLIGHT_RADIUS,
                        color: HIGHLIGHT_COLOR,
                        opacity: 0.55,
                    },
                },
                /* add= */ true,
            );
        });
    }

    let _indexLabels = [];
    function clearIndexLabels() {
        _indexLabels.forEach((lbl) => viewer.removeLabel(lbl));
        _indexLabels = [];
    }
    function renderIndexLabels() {
        clearIndexLabels();
        if (!$("show-indices").checked) return;
        // 3Dmol's selectedAtoms returns the atom records (with x/y/z and
        // serial).  Label every atom with its 0-based index.
        const atoms = viewer.selectedAtoms({});
        atoms.forEach((a, i) => {
            const lbl = viewer.addLabel(String(i), {
                position: { x: a.x, y: a.y, z: a.z },
                backgroundColor: "rgba(0,0,0,0.55)",
                fontColor: "white",
                fontSize: 9,
                inFront: true,
                showBackground: true,
            });
            _indexLabels.push(lbl);
        });
    }

    function clearViewer() {
        viewer.removeAllModels();
        viewer.removeAllLabels();
        viewer.removeAllShapes();
        _indexLabels = [];
    }

    // ----- xyz axis triad ----------------------------------------- //
    // Draws a small RGB axis triad just outside the structure's
    // bounding box so the user has a fixed orientation reference.
    // Always anchored at the world origin; the arrows visually
    // co-move with the molecule when the user rotates the camera,
    // which is what makes them useful.

    let _axisShapes = [];
    let _axisLabels = [];

    function clearAxes() {
        _axisShapes.forEach((s) => viewer.removeShape(s));
        _axisLabels.forEach((l) => viewer.removeLabel(l));
        _axisShapes = [];
        _axisLabels = [];
    }

    function drawAxes() {
        clearAxes();
        const cb = $("show-axes");
        if (!cb || !cb.checked) {
            viewer.render();
            return;
        }
        // Length = 1.2 * the structure's outermost x/y/z extent so
        // the arrows reach past the molecule visually but don't get
        // gigantic for very long chains.  Floor at 1.5 Å for empty /
        // single-atom states.
        let extent = 1.5;
        if (state.positions.length) {
            for (const [x, y, z] of state.positions) {
                extent = Math.max(
                    extent, Math.abs(x), Math.abs(y), Math.abs(z),
                );
            }
        }
        const L = extent * 1.2;
        const triplet = [
            { dir: [L, 0, 0], color: "0xff5555", label: "x" },  // red
            { dir: [0, L, 0], color: "0x55cc55", label: "y" },  // green
            { dir: [0, 0, L], color: "0x5588ff", label: "z" },  // blue
        ];
        for (const { dir, color, label } of triplet) {
            const arrow = viewer.addArrow({
                start:  { x: 0, y: 0, z: 0 },
                end:    { x: dir[0], y: dir[1], z: dir[2] },
                radius: 0.05,
                radiusRatio: 2.5,
                mid:    0.92,
                color:  color,
            });
            _axisShapes.push(arrow);
            const lbl = viewer.addLabel(label, {
                position: {
                    x: dir[0] * 1.08, y: dir[1] * 1.08, z: dir[2] * 1.08,
                },
                fontColor: color,
                backgroundOpacity: 0.0,
                fontSize: 14,
                inFront: true,
            });
            _axisLabels.push(lbl);
        }
        viewer.render();
    }

    // --------------------------------------------------------------- //
    //  Atom list (left column).  Built once per structure load; row    //
    //  highlight syncs with state.selected on every change.            //
    // --------------------------------------------------------------- //
    function rebuildAtomList() {
        const tbody = $("atom-list-body");
        tbody.innerHTML = "";
        for (let i = 0; i < state.n_atoms; i++) {
            const tr = document.createElement("tr");
            tr.dataset.atomIndex = String(i);
            tr.innerHTML = `
                <td class="col-idx">${i}</td>
                <td class="col-el">${state.elements[i] || ""}</td>
                <td class="col-name">${state.atom_names[i] || ""}</td>
                <td class="col-res">${formatResidue(i)}</td>
            `;
            tr.addEventListener("click", (ev) => {
                onAtomListRowClick(i, ev.shiftKey);
            });
            tbody.appendChild(tr);
        }
        $("atom-count").textContent =
            state.n_atoms === 1 ? "1 atom" : `${state.n_atoms} atoms`;
    }

    function formatResidue(i) {
        const rn = state.residue_names[i] || "";
        const ri = state.residue_ids[i];
        if (ri === undefined || ri === null || ri === "") return rn || "—";
        return rn ? `${rn} ${ri}` : String(ri);
    }

    function refreshSelectionUI() {
        // 1. List: toggle .is-selected on every row.
        const rows = document.querySelectorAll("#atom-list-body tr");
        rows.forEach((tr) => {
            const idx = Number(tr.dataset.atomIndex);
            tr.classList.toggle("is-selected", state.selected.has(idx));
        });
        // 2. Selection readout in the right panel.  Two layers:
        //    - one-line summary (always visible)
        //    - per-atom table showing index, element, name, residue,
        //      x/y/z coordinates -- shown only when at least one atom
        //      is selected.  Coordinates come from state.positions
        //      which we parse from xyz at applyStructure() time.
        const sel = Array.from(state.selected).sort((a, b) => a - b);
        const out = $("selection-readout");
        if (!sel.length) {
            out.textContent = "No atoms selected.";
        } else {
            const parts = sel.map((i) => `#${i} ${state.elements[i]}`);
            out.textContent = parts.join(", ");
        }
        const infoTable = $("selection-info");
        const infoBody  = $("selection-info-body");
        if (infoTable && infoBody) {
            infoBody.innerHTML = "";
            if (sel.length) {
                infoTable.hidden = false;
                for (const i of sel) {
                    const p  = state.positions[i] || [0, 0, 0];
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td class="col-idx">${i}</td>
                        <td class="col-el">${state.elements[i] || ""}</td>
                        <td class="col-name">${state.atom_names[i] || ""}</td>
                        <td class="col-res">${formatResidue(i)}</td>
                        <td class="col-coord">${p[0].toFixed(3)}</td>
                        <td class="col-coord">${p[1].toFixed(3)}</td>
                        <td class="col-coord">${p[2].toFixed(3)}</td>
                    `;
                    infoBody.appendChild(tr);
                }
            } else {
                infoTable.hidden = true;
            }
        }
        // 3. Edit-panel button enablement.
        //    - Delete: any selection AND no op in flight.            (M3)
        //    - Add atom: exactly one anchor AND no op in flight.     (M3)
        //    - Orient: exactly two anchors AND no op in flight.      (M4)
        //    - Rotate: a structure is loaded AND no op in flight.    (M4)
        // The in-flight gate prevents a double-click on Apply from
        // firing two parallel fetches; postOp() flips the inFlight
        // bit + calls refreshSelectionUI when the op starts/ends.
        const locked = state.inFlight;
        const deleteBtn = $("delete-apply");
        if (deleteBtn) deleteBtn.disabled = locked || sel.length === 0;
        const addBtn = $("add-apply");
        const anchorReadout = $("add-anchor-readout");
        if (addBtn && anchorReadout) {
            if (sel.length === 1) {
                const a = sel[0];
                addBtn.disabled = locked;
                anchorReadout.textContent =
                    `Anchor: #${a} ${state.elements[a]}`;
            } else {
                addBtn.disabled = true;
                anchorReadout.textContent =
                    sel.length === 0
                        ? "Anchor: (none)"
                        : "Anchor: pick exactly one atom";
            }
        }
        // M4: Orient button + anchor-pair readout.
        const orientBtn = $("orient-apply");
        const orientReadout = $("orient-anchor-readout");
        if (orientBtn && orientReadout) {
            if (sel.length === 2) {
                const [a, b] = sel.slice().sort((x, y) => x - y);
                orientBtn.disabled = locked;
                orientReadout.textContent =
                    `Anchors: #${a} ${state.elements[a]} → ` +
                    `#${b} ${state.elements[b]}`;
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
        // M4: Rotate button -- enabled whenever a structure is
        // loaded.  Rotation doesn't need a selection.
        const rotateBtn = $("rotate-apply");
        if (rotateBtn) rotateBtn.disabled = locked || state.n_atoms === 0;
        // 4. Viewer: re-style so the highlight overlay reflects state.
        applyStyle();
    }

    // --------------------------------------------------------------- //
    //  Click handlers — keep list and viewer in sync.                  //
    //                                                                  //
    //  Plain click  -> single-select that atom.                        //
    //  Shift-click  -> toggle that atom's membership in the selection. //
    //                                                                  //
    //  M4 will use the selection (when len == 2) as the anchor pair    //
    //  for orient-along-z; the multi-select scaffolding is here so M4  //
    //  doesn't need a UI rewrite.                                      //
    // --------------------------------------------------------------- //
    function onAtomListRowClick(idx, shiftKey) {
        if (shiftKey) {
            if (state.selected.has(idx)) state.selected.delete(idx);
            else state.selected.add(idx);
        } else {
            state.selected.clear();
            state.selected.add(idx);
        }
        refreshSelectionUI();
    }

    function onViewerAtomClick(atom, _viewer, ev) {
        // 3Dmol passes the click event as the 4th arg in modern builds.
        // Older builds put the atom record's `clickEvt` field instead;
        // fall back to that when ev is missing.
        const shift = (ev && ev.shiftKey) ||
                      (atom && atom.clickEvt && atom.clickEvt.shiftKey) ||
                      false;
        const idx = atom.serial;
        onAtomListRowClick(idx, shift);
    }

    // --------------------------------------------------------------- //
    //  Load: POST /api/build/load with a multipart file upload.  Same  //
    //  endpoint the Build tab uses; M2 doesn't need its own route.     //
    // --------------------------------------------------------------- //
    function setStatus(msg, kind = null) {
        const el = $("status");
        el.textContent = msg;
        el.className = "status" + (kind ? ` status-${kind}` : "");
    }

    async function loadFile(file) {
        setStatus(`Loading ${file.name}…`);
        const fd = new FormData();
        fd.append("file", file);
        let r;
        try {
            r = await fetch("/api/build/load", { method: "POST", body: fd })
                .then((x) => x.json());
        } catch (e) {
            setStatus("Network error: " + e.message, "error");
            return;
        }
        if (!r.ok) {
            setStatus(r.error || "Load failed.", "error");
            return;
        }
        applyStructure(r);
        setStatus(
            `Loaded ${r.n_atoms}-atom ${r.source_format.toUpperCase()} from ${file.name}.`,
            "ok",
        );
    }

    function applyStructure(r) {
        state.xyz           = r.xyz || "";
        state.elements      = Array.isArray(r.elements)      ? r.elements      : [];
        state.atom_names    = Array.isArray(r.atom_names)    ? r.atom_names    : [];
        state.residue_ids   = Array.isArray(r.residue_ids)   ? r.residue_ids   : [];
        state.residue_names = Array.isArray(r.residue_names) ? r.residue_names : [];
        state.title         = r.title || "";
        state.n_atoms       = Number(r.n_atoms || state.elements.length || 0);
        state.selected      = new Set();
        // Parse positions from the xyz string so the selection-info
        // table can display per-atom (x, y, z) without an extra
        // server roundtrip.  Lines after the 2-line header are
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

        $("title-readout").textContent =
            state.title ? `${state.title} (${formula(state.elements)})`
                        : formula(state.elements);

        // Render in viewer.
        clearViewer();
        if (state.xyz) {
            viewer.addModel(state.xyz, "xyz");
            // Wire the per-atom click hook BEFORE the first render so
            // the click region is registered.  3Dmol's setClickable
            // takes (sel, clickable, callback).
            viewer.setClickable({}, true, onViewerAtomClick);
            applyStyle();
            viewer.zoomTo();
            viewer.render();
        }

        rebuildAtomList();
        refreshSelectionUI();
    }

    function formula(elements) {
        if (!elements || !elements.length) return "—";
        const counts = {};
        elements.forEach((e) => (counts[e] = (counts[e] || 0) + 1));
        const order = ["C", "H", "N", "O", "P", "S"];
        const parts = [];
        order.forEach((e) => {
            if (counts[e]) {
                parts.push(counts[e] > 1 ? `${e}${counts[e]}` : e);
                delete counts[e];
            }
        });
        Object.keys(counts).sort().forEach((e) => {
            parts.push(counts[e] > 1 ? `${e}${counts[e]}` : e);
        });
        return parts.join("");
    }

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
        const indices = Array.from(state.selected).sort((a, b) => a - b);
        if (!indices.length) return;
        await postOp("/api/modify/delete", { indices }, "Deleted");
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
        const sel = Array.from(state.selected);
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
        const sel = Array.from(state.selected).sort((a, b) => a - b);
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
        const axis  = getCheckedRadio("rotate-axis") || "z";
        const angle = Number($("rotate-angle").value);
        if (angle === 0) {
            setEditStatus("Angle = 0; nothing to rotate.", "error");
            return;
        }
        await postOp(
            "/api/modify/rotate",
            { axis, angle },
            `Rotated ${angle}° around ${axis}`,
        );
    }

    // --------------------------------------------------------------- //
    //  Wire DOM events.                                                //
    // --------------------------------------------------------------- //
    document.addEventListener("DOMContentLoaded", () => {
        $("load-btn").addEventListener("click", () => {
            const files = $("file-picker").files;
            if (!files.length) {
                setStatus("Pick a file first.", "error");
                return;
            }
            loadFile(files[0]);
        });
        // Auto-submit on file pick: most users pick once and forget.
        $("file-picker").addEventListener("change", () => {
            const files = $("file-picker").files;
            if (files.length) loadFile(files[0]);
        });
        $("rep").addEventListener("change", applyStyle);
        $("show-indices").addEventListener("change", applyStyle);
        const showAxes = $("show-axes");
        if (showAxes) showAxes.addEventListener("change", drawAxes);
        $("clear-selection").addEventListener("click", () => {
            state.selected.clear();
            refreshSelectionUI();
        });

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
    });

    // ----- Test hook ------------------------------------------------- //
    // Exposes the click-callback path so the Playwright E2E tests can
    // verify the viewer -> list direction WITHOUT clicking WebGL
    // canvas pixels at known atom positions (which would require
    // projecting atom coordinates through the camera matrix and is
    // brittle across viewport sizes).  Production has zero behavior
    // change -- this just attaches three references to ``window``
    // that nothing else looks at.
    window.__molbuilder_modify_test = {
        onAtomListRowClick: onAtomListRowClick,
        onViewerAtomClick:  onViewerAtomClick,
        getViewer:          () => viewer,
        getSelected:        () => Array.from(state.selected),
        getNAtoms:          () => state.n_atoms,
    };
})();
