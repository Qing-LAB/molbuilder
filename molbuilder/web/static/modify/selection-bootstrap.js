/* /modify structure bootstrap -- page glue only.
 *
 * Post-migration (Track B), the concealed MolView module builds the entire
 * fused card (viewer + selection panel + View-menu toggles + measurement +
 * fold) and attaches the viewer-adapter itself.  This file does ONLY the
 * page-level DATA orchestration the module can't:
 *
 *   1. Mount the module into the empty #molview-host (molview.mount), handing
 *      it the workspace object.
 *   2. Inject the tab's XYZ/PDB loader into the module's selection namespace.
 *   3. Translate the Projects sidebar's pick/commit into a candidate + a
 *      gated _commitFile (warning modal on a dirty canvas), honouring the
 *      mount-restore ownership contract so it never clobbers a restore.
 *   4. Wire the "Load picked file" button + its readout.
 *
 * All HTTP wiring, rule translation, save dispatch and label bookkeeping live
 * in the module/store.  See docs/protocols/molview-module.md for the full
 * architecture.
 */
(function () {
    "use strict";

    // EMPTY host: molview.mount BUILDS the whole fused card into it (viewer + panel +
    // View-menu toggles + measurement + fold), exactly like the demo / Results.  Modify
    // no longer hand-builds a card, so the mount takes the module's build path.
    const HOST_ID     = "molview-host";

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

        // 1+2. Mount the WHOLE view-chrome through the embeddable molview component
        // (molview.mount, molview-module.md §18): the panel + the view-controls bar + the
        // fold handle, all bound to the workspace.  We hand it the workspace object (the
        // uniform ws.* data interface) -- no store/embed/loader wiring here.  Modify's
        // DATA orchestration (loader, sidebar candidate, Load button) stays below; molview
        // reacts to the workspace it was given.
        const _mv = window.molbuilder && window.molbuilder.molview;
        if (!_mv || typeof _mv.mount !== "function") {
            console.error("[selection-bootstrap] molview.mount missing");
            _renderFailure(host, "molview.mount module missing");
            return;
        }
        const _mounted = await _mv.mount(host, window.molbuilder.workspace, {
            mode: "modify",
            owner: "modify",   // namespaces this tab's workspace saving points (§18.4)
        });
        if (!_mounted) return;   // mount rendered its own banner / prerequisites absent

        // 2b. Inject the viewer-specific XYZ loader into the store so
        // the store doesn't reach into ``window.molbuilder`` to do
        // its own file-load (spec §5 rule 3: the store has no DOM
        // and no 3Dmol; the page wires those in).  modify/viewer.js
        // exposes ``window.molbuilder.loadStructureText`` once its
        // DOMContentLoaded runs; if missing, the store falls back
        // to atom-list-only mode -- still functional, just no
        // viewer.
        // Bind through the molview.data.selection namespace — the
        // unified DATA surface.  setLoader passes through to the
        // store's same method but keeps the legacy store private.
        const _data0 = window.molbuilder.molview
                    && window.molbuilder.molview.data;
        if (_data0 && _data0.selection
                 && typeof _data0.selection.setLoader === "function"
                 && window.molbuilder.loadStructureText) {
            _data0.selection.setLoader(window.molbuilder.loadStructureText);
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
        // The bootstrap drives molview.data.selection (set source
        // file, adopt session, subscribe to selection changes).  This
        // is the only surface; the legacy ``selection.store`` global
        // is a private implementation detail.
        const store    = window.molbuilder.molview.data.selection;
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
            onCandidateChange: _onCandidateChange,
            // commitFile(path): the canonical "load this file as
            // the workspace structure" entry point — identical to
            // what the Load button does on a dblclick.  Goes through
            // structurePage's gate (warning modal on dirty canvas) +
            // the ONE open door (openMolecule), which installs the
            // whole model in a single write (§19.3.1).  Exposed for
            // tests that need to drive the canonical sidebar→canvas
            // flow without depending on DOM clicks.
            // (The old `commitCandidate` -- a `store.setSourceFile`
            // reach-around that RELOADED the file -- was dead and is
            // removed; loading goes through commitFile only.)
            commitFile:       _commitFile,
        };

        if (projects) {
            const initial = projects.getCurrentFile() || "";
            // MOUNT-RESTORE OWNERSHIP (workspace-contract.md § "Mount-time
            // restore").  If the persisted snapshot will restore THIS same
            // file, the snapshot restore (viewer.js::restoreModifyState) is
            // the sole authority for hydrating it -- re-committing here would
            // race the restore and clobber the restored selection (the
            // two-writer mount race, fixed 2026-07-01).  Only a file the
            // snapshot does NOT own (a genuine cross-tab handoff of a
            // different/new structure) is committed on mount.
            const _ws = window.molbuilder && window.molbuilder.workspace;
            const _restoreTarget = (_ws
                && typeof _ws.mountRestoreTarget === "function")
                ? _ws.mountRestoreTarget() : null;
            if (_isLoadableStructure(initial) && initial !== _restoreTarget) {
                // Cross-tab handoff (no snapshot owns this file): commit
                // immediately on mount.  Goes through the same _commitFile
                // path as the Load button so canvas-state stays in sync.
                _commitFile(initial);
                _setCandidate(initial);
            } else if (_isLoadableStructure(initial)) {
                // Snapshot restore owns this file -- reflect the candidate for
                // the Load-button readout, but DO NOT re-commit (defer to
                // restoreModifyState, per the contract above).
                _setCandidate(initial);
            }
            projects.onChange((sel) => {
                const f = (sel && sel.file) ? sel.file : "";
                _setCandidate(f);  // empty string clears the candidate
            });
            // Universal commit subscription: a sidebar dblclick on a
            // file fires publishCommit, which lands here.  Same path
            // as the Load button — _commitFile gates through the
            // canvas-state warning modal if the canvas is dirty,
            // then renders + adopts.
            if (typeof projects.onCommit === "function") {
                projects.onCommit((sel) => {
                    const f = (sel && sel.file) ? sel.file : "";
                    if (_isLoadableStructure(f)) _commitFile(f);
                });
            }
        }

        // ----- Wire the Load button + readout -------------------- //
        const _loadBtn  = document.getElementById("load-candidate-btn");
        const _readout  = document.getElementById("load-candidate-readout");
        function _basename(p) {
            const ix = p.lastIndexOf("/");
            return ix >= 0 ? p.slice(ix + 1) : p;
        }
        // BOMB-7 fix (2026-06-07): readout distinguishes "Picked"
        // (candidate set but not yet committed) from "Loaded"
        // (candidate equals what's currently in the viewer's source
        // file).  Pre-fix the readout always said "Picked: X" even
        // after a successful Load — the user couldn't tell whether
        // clicking Load again would re-run a no-op or re-fire the
        // pipeline.  Button is disabled when there's no candidate
        // OR when the candidate already matches the loaded source.
        function _refreshLoadUI() {
            const loadedSrc = store.getState
                ? (store.getState().sourceFile || "") : "";
            const isLoaded = !!_candidate && _candidate === loadedSrc;
            if (_loadBtn) {
                _loadBtn.disabled = !_candidate || isLoaded;
            }
            if (_readout) {
                if (!_candidate) {
                    _readout.textContent = "";
                } else if (isLoaded) {
                    _readout.textContent = "Loaded: "
                        + _basename(_candidate);
                } else {
                    _readout.textContent = "Picked: "
                        + _basename(_candidate);
                }
            }
        }
        _onCandidateChange(_refreshLoadUI);
        // Re-render when the store's sourceFile changes (after a
        // successful Load / commit).  Without this subscription,
        // _refreshLoadUI only fires on candidate changes, so the
        // readout would still say "Picked" until the user re-clicks
        // the sidebar.
        if (store && typeof store.subscribe === "function") {
            store.subscribe(_refreshLoadUI);
        }

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
        // Dependency resolution uses ``runtime.whenReady`` instead
        // of a synchronous read so a future template change that
        // reorders the structure modules below this script doesn't
        // silently drop the warning-modal gate.  Pre-fix the
        // ordering invariant was "lib/structure/* loads before
        // ``DOMContentLoaded`` fires + selection-bootstrap runs"
        // — true today (the classic-script load order in
        // ``modify.html`` enforces it) but a single re-ordering
        // would silently degrade to the no-canvas-state fallback.
        // Falls back to a direct ``setSourceFile`` only when the
        // orchestrator is genuinely absent (non-Molbuilder embeds);
        // the candidate-only contract is unaffected.
        async function _resolveStructurePage() {
            // Fast path: already on window.
            const sp0 = window.molbuilder
                     && window.molbuilder.structurePage;
            if (sp0) return sp0;
            // Slow path: wait via the runtime registry.  If the
            // runtime isn't installed at all (tests / non-Molbuilder
            // embeds), return null and the caller drops to the
            // legacy setSourceFile path.
            const rt = window.molbuilder
                    && window.molbuilder.runtime;
            if (!rt || typeof rt.whenReady !== "function") return null;
            try {
                return await rt.whenReady("structure.page");
            } catch (_) {
                return null;
            }
        }
        async function _commitFile(path) {
            if (!path) return;
            const sp = await _resolveStructurePage();
            const projectsApi = window.molbuilder
                    && window.molbuilder.projects;
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
            // A6 (molview-migration-plan): load the workspace DATA through the
            // working-copy framework -- the per-atom rows (element + the sidecar's
            // regions/frozen, applied server-side by codec.load) + the full
            // periodicity (cell/axis_kind/vacuum).  Read the working-copy data through
            // the ONE data door (molview.data.readWorkingCopy), NOT a raw POST to
            // /api/workingcopy/open -- consumers don't reach around the unified
            // surface.  Returns null on failure -> a plain byte load.
            let opened = null;
            try {
                const _d = window.molbuilder.molview && window.molbuilder.molview.data;
                if (_d && typeof _d.readWorkingCopy === "function") {
                    opened = await _d.readWorkingCopy(path);
                }
            } catch (_) { opened = null; }
            const periodicity = (opened && opened.periodicity) || null;
            // F1: carry the annotation channels opaquely so a later Save re-emits
            // them (Modify doesn't edit annotations, but must not clobber them).
            const annotations = (opened && opened.annotations) || null;
            const sidecarAtoms = (opened && Array.isArray(opened.atoms))
                ? opened.atoms : null;
            // F4: suspend persistence across the text->atoms window so no draft is
            // written pairing the new geometry with the PREVIOUS file's labels; resume
            // at every exit (one resume per path).
            const _dataD = window.molbuilder.molview
                        && window.molbuilder.molview.data;
            const _resumeLoad = () => {
                if (_dataD && typeof _dataD.resumePersist === "function") _dataD.resumePersist();
            };
            if (_dataD && typeof _dataD.suspendPersist === "function") _dataD.suspendPersist();
            // THE LOAD CONTRACT (molview-module.md §19.3 + save-flow.md): a file load
            // is ONE call through the single open door.  Everything the molecule needs
            // -- geometry text, periodicity, annotations, the sidecar-enriched atoms
            // AND the source path -- rides IN so the WHOLE model (canvas + selection
            // store + render) is installed in ONE synchronous write and is fully
            // SETTLED before loadIntoCanvas resolves.  There is deliberately NO second
            // store write after this: the old trailing `store.adoptSession(...)` reset
            // the selection to [] a few hundred ms later (after `await
            // _anchorTimeline()`'s HTTP), which silently clobbered any atom the user
            // clicked in that gap -- the "selection never sticks right after Load"
            // race (fixed 2026-07).
            const gate = await sp.loadIntoCanvas(
                { source_format: format, text: r.text, periodicity: periodicity,
                  annotations: annotations, atoms: sidecarAtoms },
                { kind: "file", file: path }
            );
            if (!gate.ok) { _resumeLoad(); return; }  // cancelled — leave viewer alone
            // Fallback ONLY when readWorkingCopy failed (no sidecar atoms rode in):
            // the plain build/load atoms are installed, but the .molstruct.json's
            // regions/frozen overlay is missing.  refreshAtoms refetches the
            // sidecar-applied rows from /api/selection/atoms -- and, unlike a fresh
            // adoptSession, it does NOT reset the selection, so it cannot clobber a
            // just-made pick even though it lands after the "ready" signal.
            if (!sidecarAtoms && typeof store.refreshAtoms === "function") {
                await store.refreshAtoms();
            }
            _resumeLoad();   // F4: load done -> resume + write the consistent state
        }
        if (_loadBtn) {
            _loadBtn.addEventListener("click", () => _commitFile(_candidate));
        }

        // 4. Viewer-adapter (selection halos + frozen halos + click-to-select) is
        // attached BY THE MODULE: molview.mount's built-card path calls
        // selApi.viewerAdapter.attach(h, {store, mode}) on the viewer it embeds
        // (mount.js §built-card onReady).  Modify no longer embeds its own viewer or
        // attaches the adapter -- that was the pre-migration duplication.  Nothing to do
        // here.
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrap);
    } else {
        bootstrap();
    }
})();
