/* /spectra page bootstrap.
 *
 * Two responsibilities, both /spectra-specific (the generic
 * /results flow already lives in lib/spectra/core.js + the
 * spectra inspector adapter):
 *
 *   1. Mount the spectra-inspector core against ``document`` so
 *      the schema form + Generate / Methods / Issues / script-
 *      preview / Save handlers wire up.  This is the unchanged
 *      pre-task-#296 behaviour.
 *
 *   2. Wire the Inspect-structure card (task #296, 2026-06-09):
 *      mount a 3Dmol embed in ``#viewer-wrap`` and hook the
 *      ``#load-from-sidebar-btn`` so the user can pick a
 *      structure file in the Projects sidebar and click Load to
 *      see it in the viewer.  Loading also pushes the raw bytes
 *      into the spectra inspector via
 *      ``window.molbuilder.spectraInspector.setStructureText(text)``
 *      — the in-memory holder that backs ``getStructureText()``
 *      (task #309, 2026-06-09).  The pre-#309 path through a
 *      hidden ``<textarea id="structure-text">`` is gone.
 *
 * Mirrors the Optimization tab's pattern in static/viewer.js
 * (the Load button, the info readout, the sidebar onCommit
 * subscription).  Where the two pages diverge today:
 *   * Optimization carries a Build form + Generate buttons +
 *     SIESTA/PySCF schema; spectra carries the spectra schema
 *     and its own Generate/Methods/script-preview machinery in
 *     core.js.
 *   * Optimization's commit also POSTs /api/build/load to get
 *     the canonical workspace payload (atom list, residue ids,
 *     etc.) so the SIESTA validators have it.  Spectra doesn't
 *     consume that payload today; we only need the raw bytes
 *     for the form's structure_text field.  Skipping the build/
 *     load roundtrip keeps the load fast.
 */
(function () {
    "use strict";

    function _$(id) { return document.getElementById(id); }

    function setStatus(elId, msg, kind) {
        const el = _$(elId);
        if (!el) return;
        el.textContent = msg || "";
        el.className = "status"
            + (kind === "ok"    ? " ok"    : "")
            + (kind === "warn"  ? " warn"  : "")
            + (kind === "error" ? " error" : "");
    }

    function _bootstrapSpectraCore() {
        // Defensive: if core.js failed to load (e.g. a CDN-block in
        // some weird CSP edge case), fail loudly in the console
        // rather than silently rendering an empty <main>.
        const api = (window.molbuilder || {}).spectraInspector;
        if (!api || typeof api.mount !== "function") {
            console.error(
                "[spectra] spectraInspector.mount not found.  "
                + "Check that lib/spectra/core.js is loaded BEFORE "
                + "spectra/viewer.js (the script tag order in "
                + "spectra.html is load-bearing)."
            );
            return;
        }
        // Mount with ``document`` as the root so $() lookups find
        // /spectra's generate-side form ids (which live OUTSIDE any
        // partial; full-page mount).
        api.mount(document);
    }

    /**
     * Update the static info readout below the viewer header.
     * Title shows the basename of the loaded file; atom count +
     * formula are parsed from the XYZ payload.  Cheap; runs
     * synchronously after every successful Load.
     */
    function _updateInfo(filename, xyzText) {
        const title   = _$("info-title");
        const atomsEl = _$("info-atoms");
        const formula = _$("info-formula");
        if (!xyzText) {
            if (title)   title.textContent   = "no structure loaded";
            if (atomsEl) atomsEl.textContent = "—";
            if (formula) formula.textContent = "—";
            return;
        }
        const lines = xyzText.split(/\r?\n/);
        const nAtoms = parseInt((lines[0] || "").trim(), 10);
        const elements = [];
        for (let i = 2; i < lines.length && elements.length < nAtoms; i++) {
            const parts = lines[i].trim().split(/\s+/);
            if (parts.length >= 4) elements.push(parts[0]);
        }
        if (title)   title.textContent   = filename || "loaded";
        if (atomsEl) atomsEl.textContent = String(nAtoms || elements.length || "—");
        // Use the shared formula helper if available; else a
        // minimal element-count fallback.
        const fmt = (window.molbuilder && window.molbuilder.fmt
                     && typeof window.molbuilder.fmt.formula === "function")
            ? window.molbuilder.fmt.formula
            : (els) => {
                const counts = {};
                els.forEach((e) => { counts[e] = (counts[e] || 0) + 1; });
                return Object.keys(counts).sort()
                    .map((k) => k + (counts[k] > 1 ? counts[k] : ""))
                    .join("");
            };
        if (formula) formula.textContent = elements.length
            ? fmt(elements) : "—";
    }

    /**
     * Bootstrap the Inspect-structure card: mount the 3Dmol
     * embed, wire the Load button to ``/api/files/read`` + the
     * sidebar's current pick, and subscribe to ``onCommit`` so
     * dblclick on a sidebar .xyz/.pdb auto-loads.
     */
    function _bootstrapInspectCard() {
        const viewerSlot = _$("viewer");
        if (!viewerSlot) return;
        const embedApi = (window.molbuilder
                          && window.molbuilder.viewer
                          && window.molbuilder.viewer.embed);
        if (typeof embedApi !== "function") {
            setStatus("load-status",
                "Viewer unavailable: lib/mol-viewer-embed.js "
                + "missing from the template script tags.", "error");
            return;
        }
        const handle = embedApi(viewerSlot, {
            style: { rep: "ball-and-stick", radiusScale: 1.0 },
            card: {
                title:        "",
                showInfoLine: false,
                height:       "420px",
            },
            axes: true,
        });

        let _sidebarLastFile = "";
        let _loadSeq         = 0;

        // ---- Load button + sidebar onCommit subscription -------- //
        const _isLoadable = (name) => {
            const n = String(name || "").toLowerCase();
            return n.endsWith(".xyz") || n.endsWith(".pdb");
        };
        const _basename = (p) => {
            const ix = String(p || "").lastIndexOf("/");
            return ix >= 0 ? p.slice(ix + 1) : p;
        };

        let _candidatePath = "";
        function _refreshLoadButton() {
            const btn = _$("load-from-sidebar-btn");
            const readout = _$("load-source-readout");
            if (!btn) return;
            const loadable = _isLoadable(_candidatePath);
            btn.disabled = !loadable;
            if (readout) {
                const isLoaded = loadable
                    && _sidebarLastFile === _candidatePath;
                readout.textContent = isLoaded
                    ? `Loaded: ${_basename(_candidatePath)}`
                    : loadable
                        ? `Selected: ${_basename(_candidatePath)}`
                        : (_candidatePath
                            ? `Selected: ${_basename(_candidatePath)} (not loadable)`
                            : "Pick a .xyz / .pdb in the Projects sidebar.");
            }
        }

        async function _commitStructure(sel) {
            const f = (sel && sel.file) ? String(sel.file) : "";
            const ext = f.toLowerCase().split(".").pop();
            if (ext !== "xyz" && ext !== "pdb") {
                if (f) {
                    setStatus("load-status",
                        `${_basename(f)} is not a structure file `
                        + `(.xyz / .pdb only).`, "warn");
                }
                return;
            }
            if (f === _sidebarLastFile) return;
            const mySeq = ++_loadSeq;
            setStatus("load-status",
                `Loading ${_basename(f)}…`, null);
            let text;
            try {
                const r = await fetch(
                    "/api/files/read?path=" + encodeURIComponent(f)
                    + "&max_bytes=16777216"
                );
                const body = await r.json();
                if (!r.ok || !body.ok) {
                    setStatus("load-status",
                        body.error || `Read failed (HTTP ${r.status}).`,
                        "error");
                    return;
                }
                text = body.text;
            } catch (e) {
                setStatus("load-status",
                    "Network error: " + (e && e.message ? e.message : e),
                    "error");
                return;
            }
            if (mySeq !== _loadSeq) return;  // superseded by a newer load
            _sidebarLastFile = f;
            // Push the bytes into spectra/core.js's in-memory
            // holder (task #309, 2026-06-09 follow-up to #296).
            // Pre-#309 we wrote to a hidden ``<textarea id="
            // structure-text">``; that textarea is gone now and
            // the inspector exposes a setter directly on its
            // public API.
            const spec = (window.molbuilder || {}).spectraInspector;
            if (spec && typeof spec.setStructureText === "function") {
                spec.setStructureText(text);
            }
            // Drop into the viewer.
            try {
                handle.setStructure(ext === "pdb"
                    ? { pdb: text }
                    : { xyz: text });
                handle.refit();
            } catch (e) {
                // Viewer render error shouldn't block the load —
                // surface but keep the in-memory holder populated
                // (setStructureText already ran above) so Generate
                // still has the bytes it needs.
                console.warn("[spectra] viewer render failed:", e);
            }
            _updateInfo(_basename(f), text);
            setStatus("load-status",
                `Loaded ${_basename(f)}.`, "ok");
            _refreshLoadButton();
        }

        // Sidebar onChange / onCommit subscription + initial
        // candidate-path tracking.  Mirrors the Optimization
        // tab's pattern in static/viewer.js.
        const rt = (window.molbuilder || {}).runtime;
        const projP = (rt && typeof rt.whenReady === "function")
            ? rt.whenReady("projects")
            : Promise.resolve((window.molbuilder || {}).projects);
        projP.then((proj) => {
            if (!proj) return;
            // Initial mount-time auto-load (cross-tab handoff via
            // sessionStorage.molbuilder.current_file).
            const initialFile = (typeof proj.getCurrentFile === "function")
                ? proj.getCurrentFile() : "";
            if (initialFile) {
                _candidatePath = initialFile;
                _refreshLoadButton();
                _commitStructure({ file: initialFile });
            } else {
                _refreshLoadButton();
            }
            // Subscribe to sidebar changes for the candidate-path
            // readout (single-click → "Selected: foo.xyz" hint).
            if (typeof proj.onChange === "function") {
                proj.onChange((sel) => {
                    _candidatePath = (sel && sel.file) ? sel.file : "";
                    _refreshLoadButton();
                });
            }
            // Dblclick commits the file for loading (universal
            // interaction model — task #301 same channel).
            const subscribe = (typeof proj.onCommit === "function")
                ? proj.onCommit.bind(proj)
                : proj.onChange.bind(proj);
            subscribe(_commitStructure);
        });

        // Explicit Load button click → load whatever the sidebar
        // currently highlights.
        const loadBtn = _$("load-from-sidebar-btn");
        if (loadBtn) {
            loadBtn.addEventListener("click", () => {
                if (!_isLoadable(_candidatePath)) return;
                _commitStructure({ file: _candidatePath });
            });
        }
    }

    function bootstrapSpectraPage() {
        _bootstrapInspectCard();
        _bootstrapSpectraCore();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrapSpectraPage);
    } else {
        bootstrapSpectraPage();
    }
})();
