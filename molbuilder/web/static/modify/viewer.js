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
        atom_names: [],        // ["CA", "HB1", ...] (or [] if absent)
        residue_ids: [],       // [1, 1, 2, ...]   (or [] if absent)
        residue_names: [],     // ["MOL", ...]     (or [] if absent)
        title: "",
        n_atoms: 0,
        selected: new Set(),   // atom indices
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
        viewer.render();
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
        // 2. Selection readout in the right panel.
        const sel = Array.from(state.selected).sort((a, b) => a - b);
        const out = $("selection-readout");
        if (!sel.length) {
            out.textContent = "No atoms selected.";
        } else {
            const parts = sel.map((i) => `#${i} ${state.elements[i]}`);
            out.textContent = parts.join(", ");
        }
        // 3. Viewer: re-style so the highlight overlay reflects state.
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
        $("clear-selection").addEventListener("click", () => {
            state.selected.clear();
            refreshSelectionUI();
        });
    });
})();
