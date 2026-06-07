/* /modify atom-selection bootstrap -- page glue only.
 *
 * Responsibilities (and ONLY these):
 *
 *   1. Fetch the _selection_panel.html partial and insert it into
 *      the page's #selection-host.
 *   2. Mount the panel against the singleton store.
 *   3. Attach the viewer-adapter once modify/viewer.js has exposed
 *      its 3Dmol viewer.
 *   4. Forward the Projects sidebar's onChange (filtered to .xyz)
 *      into store.setSourceFile.
 *
 * That's it.  All HTTP wiring, rule translation, save dispatch and
 * label bookkeeping lives in the store.  See
 * docs/protocols/atom-selection.md for the full architecture.
 */
(function () {
    "use strict";

    const PARTIAL_URL = "/partials/selection-panel";
    const HOST_ID     = "selection-host";

    function _renderFailure(host, message) {
        // Surface the failure inline so the user doesn't stare at a
        // silently-blank column.  textContent keeps user-visible
        // strings (and the URL) out of the HTML parser.
        const banner = document.createElement("div");
        banner.className = "selection-bootstrap-error";
        banner.setAttribute("role", "alert");
        banner.textContent = "Selection panel failed to load: " + message;
        host.innerHTML = "";
        host.appendChild(banner);
    }

    async function bootstrap() {
        const host = document.getElementById(HOST_ID);
        if (!host) {
            console.warn("[selection-bootstrap] #" + HOST_ID
                       + " missing; skipping.");
            return;
        }

        // 1. Fetch + insert the partial.  Same-origin, autoescaped
        // Jinja render, no user data -> innerHTML is safe.
        let html;
        try {
            const r = await fetch(PARTIAL_URL);
            if (!r.ok) {
                console.error("[selection-bootstrap] partial fetch returned "
                            + r.status);
                _renderFailure(host, "HTTP " + r.status + " from " + PARTIAL_URL);
                return;
            }
            html = await r.text();
        } catch (e) {
            console.error("[selection-bootstrap] partial fetch threw", e);
            _renderFailure(host, (e && e.message) ? e.message : String(e));
            return;
        }
        host.innerHTML = html;

        // 2. Mount the panel.
        if (!window.molbuilder || !window.molbuilder.selectionPanel) {
            console.error("[selection-bootstrap] selectionPanel module missing");
            return;
        }
        window.molbuilder.selectionPanel.mount(host);

        // 2b. Inject the viewer-specific XYZ loader into the store so
        // the store doesn't reach into ``window.molbuilder`` to do
        // its own file-load (spec §5 rule 3: the store has no DOM
        // and no 3Dmol; the page wires those in).  modify/viewer.js
        // exposes ``window.molbuilder.loadStructureText`` once its
        // DOMContentLoaded runs; if missing, the store falls back
        // to atom-list-only mode -- still functional, just no
        // viewer.
        const _store0 = window.molbuilder.selection
                      && window.molbuilder.selection.store;
        if (_store0 && typeof _store0.setLoader === "function"
                    && window.molbuilder.loadStructureText) {
            _store0.setLoader(window.molbuilder.loadStructureText);
        }

        // 3. Sidebar selection sets a CANDIDATE — does NOT commit
        // the viewer load.  The user reviews the candidate path in
        // the loader-bar and clicks "Load picked file" to commit.
        //
        // Why candidate-only: clicking a file in the sidebar is a
        // browse action ("show me what's in this file"); committing
        // it as the workspace structure is a separate intent and
        // would discard any unsaved modifications.  See
        // docs/tabs/architecture.md § 5.2.
        //
        // Page-mount seeding: if the bare host was visited with an
        // existing sidebar selection in sessionStorage (cross-tab
        // handoff from /structure-optimization etc.), commit it on
        // mount — the user arrived with intent ("send my structure
        // to the Molbuilder tab"), not with a stray click.
        const projects = window.molbuilder && window.molbuilder.projects;
        const store    = window.molbuilder.selection.store;
        // Both .xyz and .pdb are loadable into /molbuilder -- the
        // server's selection blueprint dispatches by extension
        // (see web/blueprints/selection.py
        // ``_SUPPORTED_STRUCTURE_SUFFIXES``).  A pick of .log /
        // .fdf etc. is "view this file" not "swap the workspace
        // structure" -- those leave the candidate empty so the
        // Load button stays disabled.
        function _isLoadableStructure(name) {
            const lc = String(name || "").toLowerCase();
            return lc.endsWith(".xyz") || lc.endsWith(".pdb");
        }

        // ----- Candidate state ----------------------------------- //
        let _candidate = "";
        const _candidateListeners = [];
        function _notifyCandidate() {
            for (const fn of _candidateListeners.slice()) {
                try { fn(_candidate); } catch (_) {}
            }
        }
        function _setCandidate(path) {
            const next = _isLoadableStructure(path) ? String(path) : "";
            if (next === _candidate) return;
            _candidate = next;
            _notifyCandidate();
        }
        function _commitCandidate() {
            if (!_candidate) return null;
            const path = _candidate;
            store.setSourceFile(path);
            return path;
        }
        function _onCandidateChange(fn) {
            if (typeof fn !== "function") {
                throw new TypeError(
                    "onCandidateChange: fn must be a function");
            }
            _candidateListeners.push(fn);
            // Fire once immediately so subscribers see the current
            // state without waiting for the next sidebar pick.
            try { fn(_candidate); } catch (_) {}
            return function unsubscribe() {
                const ix = _candidateListeners.indexOf(fn);
                if (ix >= 0) _candidateListeners.splice(ix, 1);
            };
        }
        window.molbuilder.molbuilderTab = {
            getCandidate:     () => _candidate,
            commitCandidate:  _commitCandidate,
            onCandidateChange: _onCandidateChange,
        };

        if (projects) {
            const initial = projects.getCurrentFile() || "";
            if (_isLoadableStructure(initial)) {
                // Cross-tab handoff: commit immediately on mount.
                // Goes through the same _commitFile path as the Load
                // button so canvas-state stays in sync (the legacy
                // ``store.setSourceFile(initial)`` shortcut left
                // canvas-state empty for cross-tab-handed-off
                // structures, and Save then failed with
                // "No structure to save").
                _commitFile(initial);
                _setCandidate(initial);
            }
            projects.onChange((sel) => {
                const f = (sel && sel.file) ? sel.file : "";
                _setCandidate(f);  // empty string clears the candidate
            });
        }

        // ----- Wire the Load button + readout -------------------- //
        const _loadBtn  = document.getElementById("load-candidate-btn");
        const _readout  = document.getElementById("load-candidate-readout");
        function _basename(p) {
            const ix = p.lastIndexOf("/");
            return ix >= 0 ? p.slice(ix + 1) : p;
        }
        function _refreshLoadUI() {
            if (_loadBtn) _loadBtn.disabled = !_candidate;
            if (_readout) {
                _readout.textContent = _candidate
                    ? "Picked: " + _basename(_candidate)
                    : "";
            }
        }
        _onCandidateChange(_refreshLoadUI);

        // The single "commit a structure file into the workspace"
        // path.  Read the file ONCE, gate through structurePage so
        // a dirty canvas fires the warning modal, then drive the
        // viewer + selection-store atomically — without going
        // through ``store.setSourceFile`` (which would re-read the
        // file + re-invoke the viewer loader, racing the
        // already-rendered structure and wasting a roundtrip).
        // Used by the Load button AND the page-mount cross-tab
        // handoff so canvas-state stays populated either way.
        //
        // Falls back to a direct ``setSourceFile`` only when the
        // orchestrator + projects API aren't wired (early-boot
        // race in tests / non-Molbuilder embeds); the candidate-
        // only contract is unaffected.
        async function _commitFile(path) {
            if (!path) return;
            const sp = window.molbuilder
                    && window.molbuilder.structurePage;
            const projectsApi = window.molbuilder
                    && window.molbuilder.projects;
            const viewerLoader = window.molbuilder
                    && window.molbuilder.loadStructureText;
            if (!sp || !projectsApi
                || typeof projectsApi.readFile !== "function") {
                store.setSourceFile(path);
                return;
            }
            const r = await projectsApi.readFile(path);
            if (!r || !r.ok) {
                const s = document.getElementById("status");
                if (s) {
                    s.textContent = (r && r.error)
                        ? "Could not read file: " + r.error
                        : "Could not read file.";
                    s.className = "status error";
                }
                return;
            }
            const format = path.toLowerCase().endsWith(".pdb")
                ? "pdb" : "xyz";
            const gate = await sp.loadIntoCanvas(
                { source_format: format, text: r.text },
                { kind: "file", file: path }
            );
            if (!gate.ok) return;  // cancelled — leave viewer alone
            // Drive the viewer with the bytes we already have.
            if (typeof viewerLoader === "function") {
                try {
                    await viewerLoader(r.text, _basename(path));
                } catch (e) {
                    const s = document.getElementById("status");
                    if (s) {
                        s.textContent = "Viewer failed: "
                            + (e && e.message ? e.message : String(e));
                        s.className = "status error";
                    }
                    return;
                }
            }
            // Tell the selection store the file is now the source
            // WITHOUT re-loading the viewer (we just did).  This
            // fetches the atoms list for the panel.
            if (typeof store.adoptSession === "function") {
                store.adoptSession({ sourceFile: path, selection: [] });
            } else {
                store.setSourceFile(path);  // legacy fallback
            }
        }
        if (_loadBtn) {
            _loadBtn.addEventListener("click", () => _commitFile(_candidate));
        }

        // 4. Attach the viewer-adapter once modify/viewer.js has
        // registered its embed handle on the runtime registry.
        // #246 B2: the previous 10×100ms poll for
        // ``window.molbuilder.modify.handle`` is exactly the bug
        // class the runtime registry exists to retire (cf.
        // /build's runtime.whenReady("projects") pattern, see
        // lib/molbuilder-runtime.js docstring).  modify/viewer.js
        // calls ``runtime.register("modify.handle", _handle)`` at
        // boot; whenReady fires synchronously if already-registered
        // and queues otherwise.
        const adapterModule =
            (window.molbuilder.selection && window.molbuilder.selection.viewerAdapter)
                ? window.molbuilder.selection.viewerAdapter : null;
        if (!adapterModule) {
            console.warn("[selection-bootstrap] viewerAdapter module missing");
            return;
        }
        const runtime = window.molbuilder.runtime;
        if (!runtime || typeof runtime.whenReady !== "function") {
            console.warn(
                "[selection-bootstrap] molbuilder.runtime unavailable; "
              + "click integration disabled (lib/molbuilder-runtime.js "
              + "is the hard dep that should always load first)"
            );
            return;
        }
        runtime.whenReady("modify.handle").then((h) => {
            adapterModule.attach(h);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrap);
    } else {
        bootstrap();
    }
})();
