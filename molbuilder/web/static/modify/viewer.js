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
 * Spec: docs/spec/modify-tab.md (M2-M4 + Phase 1 cross-tab
 * persistence as of 2026-05-09).
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    // --------------------------------------------------------------- //
    //  Module state.  Single canonical structure (xyz string + parsed  //
    //  atom metadata) plus a per-atom selected/highlighted bookkeeping //
    //  set.  Multi-select is shift-click; the orient op (M4) reads     //
    //  the selection as an anchor pair when its length is exactly 2.   //
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
        selected: new Set(),   // atom indices
        inFlight: false,       // true while an /api/modify/* fetch is open
    };

    // Selection halo: a thin translucent wireframe sphere drawn
    // ALONGSIDE the atom (as a separate shape, not via setStyle).
    // The atom keeps its element color and stick/ball-and-stick
    // appearance unchanged; the user sees a "ring" around it
    // marking the selection.  Radius is element-vdW + 0.35 Å so the
    // halo always sits just outside the atom's rendered surface
    // regardless of representation.
    const HIGHLIGHT_COLOR  = "#fbbf24";      // amber, matches --warning
    const HIGHLIGHT_PAD    = 0.35;           // Å beyond the atom's vdW
    const HIGHLIGHT_OPACITY = 0.55;
    // Approximate vdW radii (Å) for the elements the Modify tab is
    // likely to see.  Falls back to 1.5 for anything not in the
    // table -- close enough to put the halo outside the atom.
    const _VDW = {
        H: 1.20, He: 1.40, Li: 1.82, Be: 1.53, B: 1.92, C: 1.70,
        N: 1.55, O: 1.52, F: 1.47, Ne: 1.54, Na: 2.27, Mg: 1.73,
        Si: 2.10, P: 1.80, S: 1.80, Cl: 1.75, K: 2.75, Ca: 2.31,
        Ni: 1.63, Cu: 1.40, Zn: 1.39, Br: 1.85, I: 1.98,
        Pd: 1.63, Ag: 1.72, Pt: 1.75, Au: 1.66,
    };

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

    let _highlightShapes = [];

    function clearHighlights() {
        _highlightShapes.forEach((s) => viewer.removeShape(s));
        _highlightShapes = [];
    }

    function renderHighlights() {
        // Draw a thin amber halo around each selected atom.  Drawn
        // as a SEPARATE shape (addSphere with wireframe=true) so the
        // atom's element color + stick rendering are unchanged --
        // the halo just sits around it as a ring.
        clearHighlights();
        state.selected.forEach((idx) => {
            const p = state.positions[idx];
            if (!p) return;
            const el = state.elements[idx] || "C";
            const r = (_VDW[el] || 1.5) + HIGHLIGHT_PAD;
            const sphere = viewer.addSphere({
                center:    { x: p[0], y: p[1], z: p[2] },
                radius:    r,
                color:     HIGHLIGHT_COLOR,
                wireframe: true,
                opacity:   HIGHLIGHT_OPACITY,
            });
            _highlightShapes.push(sphere);
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
        // M5: Electrode + Send-to-Build buttons.  Electrode requires
        // the right number of anchors for its mode (1 for single,
        // 2 for symmetric); Send-to-Build needs a structure.
        const elcBtn = $("elc-apply");
        const elcReadout = $("elc-anchor-readout");
        const mode = ($("elc-mode") || {}).value || "symmetric";
        if (elcBtn && elcReadout) {
            const need = (mode === "single") ? 1 : 2;
            if (sel.length === need) {
                elcBtn.disabled = locked;
                if (need === 1) {
                    elcReadout.textContent =
                        `Anchor: #${sel[0]} ${state.elements[sel[0]]}.  ` +
                        `Side determines which face the slab grows on.`;
                } else {
                    const [a, b] = sel.slice().sort((x, y) => x - y);
                    elcReadout.textContent =
                        `Anchors: #${a} ${state.elements[a]} (-z) ↔ ` +
                        `#${b} ${state.elements[b]} (+z).`;
                }
            } else {
                elcBtn.disabled = true;
                elcReadout.textContent =
                    mode === "single"
                        ? "Single mode: pick exactly one anchor."
                        : "Pair mode: pick two atoms (the +z and -z anchors).";
            }
        }
        const sendBtn = $("send-to-build");
        if (sendBtn) sendBtn.disabled = locked || state.n_atoms === 0;
        // 4. Viewer: re-style so the highlight overlay reflects state.
        applyStyle();
    }

    // --------------------------------------------------------------- //
    //  Click handlers — keep list and viewer in sync.                  //
    //                                                                  //
    //  Plain click  -> single-select that atom.                        //
    //  Shift-click  -> toggle that atom's membership in the selection. //
    //                                                                  //
    //  The orient op (M4) reads the selection as the anchor pair       //
    //  (a0, a1) when its length is exactly 2; the add-atom op (M3)     //
    //  reads it as the anchor when the length is exactly 1.            //
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
        state.chain_ids     = Array.isArray(r.chain_ids)     ? r.chain_ids     : [];
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
        const sel = Array.from(state.selected).sort((a, b) => a - b);
        const mode = $("elc-mode").value;
        const common = readElcCommonBody();
        const gap = Number($("elc-gap").value);
        if (mode === "single") {
            if (sel.length !== 1) {
                setEditStatus("Pick exactly one anchor for single mode.", "error");
                return;
            }
            const side = getCheckedRadio("elc-side") || "+z";
            await postOp(
                "/api/modify/electrode",
                Object.assign({}, common, {
                    anchor_index:     sel[0],
                    side:             side,
                    contact_distance: gap,
                }),
                `Added ${common.element}(${common.plane}) ${side}`,
            );
        } else {
            // Symmetric mode.  The renderer wants
            // anchors=[a_top, a_bot] (+z anchor first).  Sort the
            // selection by z-coordinate so the user doesn't have to
            // care about click order.
            if (sel.length !== 2) {
                setEditStatus("Pair mode needs exactly two anchors.", "error");
                return;
            }
            const [i0, i1] = sel;
            const z0 = state.positions[i0][2];
            const z1 = state.positions[i1][2];
            const a_top = z0 >= z1 ? i0 : i1;
            const a_bot = z0 >= z1 ? i1 : i0;
            await postOp(
                "/api/modify/symmetric_electrodes",
                Object.assign({}, common, {
                    anchors: [a_top, a_bot],
                    gap:     gap,
                }),
                `Added ${common.element}(${common.plane}) pair`,
            );
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
        refreshElcReadouts();

        // Sub-tabs: click an op-tab button to swap which panel is
        // visible.  Pure DOM toggle (no state in the IIFE; the
        // is-active class is the state).
        document.querySelectorAll(".optab").forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.dataset.opTab;
                document.querySelectorAll(".optab").forEach((b) => {
                    b.classList.toggle(
                        "is-active",
                        b.dataset.opTab === target,
                    );
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
        let camera = null;
        try {
            camera = viewer.getView();
        } catch (_e) {
            // 3Dmol's getView is synchronous and shouldn't throw, but
            // be defensive in case the viewer was torn down early.
        }
        const payload = {
            v: STATE_SCHEMA_VERSION,
            saved_at: new Date().toISOString(),
            xyz:           state.xyz,
            elements:      state.elements,
            atom_names:    state.atom_names,
            residue_ids:   state.residue_ids,
            residue_names: state.residue_names,
            chain_ids:     state.chain_ids,
            title:         state.title,
            n_atoms:       state.n_atoms,
            selected:      Array.from(state.selected).sort((a, b) => a - b),
            camera:        camera,
            show_axes:     ($("show-axes")    || {}).checked || false,
            show_indices:  ($("show-indices") || {}).checked || false,
            rep:           ($("rep") || {}).value || "stick",
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
        // Restore the toolbar toggles BEFORE rendering so the first
        // applyStyle picks up the saved representation.
        if ($("rep") && saved.rep) $("rep").value = saved.rep;
        if ($("show-indices")) $("show-indices").checked = !!saved.show_indices;
        if ($("show-axes"))    $("show-axes").checked    = !!saved.show_axes;
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
        // Restore the selection (applyStructure cleared it).
        if (Array.isArray(saved.selected) && saved.selected.length) {
            saved.selected.forEach((i) => {
                if (Number.isInteger(i) && i >= 0 && i < state.n_atoms) {
                    state.selected.add(i);
                }
            });
            refreshSelectionUI();
        }
        // Restore the camera last so it doesn't fight zoomTo() inside
        // applyStructure.
        if (Array.isArray(saved.camera)) {
            try {
                viewer.setView(saved.camera);
                viewer.render();
            } catch (_e) {
                // Bad cached camera array -- fall back to default.
                viewer.zoomTo();
                viewer.render();
            }
        }
        setStatus(
            `Restored ${state.n_atoms}-atom structure (${saved.title || "unnamed"}).`,
            "ok",
        );
    }

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
