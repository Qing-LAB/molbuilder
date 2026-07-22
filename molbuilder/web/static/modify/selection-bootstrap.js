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
import { mount, data as mvData } from "/static/lib/molview/index.js";

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
        if (typeof mount !== "function" || !mvData) {
            console.error("[selection-bootstrap] molview import missing");
            _renderFailure(host, "molview module missing");
            return;
        }
        const _mounted = await mount(host, window.molbuilder.workspace, {
            mode: "modify",
            owner: "modify",   // namespaces this tab's workspace saving points (§18.4)
        });
        if (!_mounted || !_mounted.ok) return;   // mount contract: failure -> {ok:false}; it warned already

        // ---- COMPOSITION ROOT (molview-esm-finalization.md §4.1) --------------------------
        // This bootstrap is the ONE place that imports molview.data; it now WIRES the Modify
        // tab's structure orchestrators with it (+ the warning modal).  page/save/file.js are
        // pure dependency-injection: they never reach for window.molbuilder.molview.data
        // themselves.  Wired here, after MolView is mounted, so the model is live.
        (function _wireStructureModules() {
            const mb = window.molbuilder || {};
            const wm = mb.warningModal;
            if (mb.structurePage && wm && typeof mb.structurePage._bind === "function") {
                mb.structurePage._bind(mvData, wm);
            }
            if (mb.structureSave && typeof mb.structureSave.configure === "function") {
                mb.structureSave.configure({ workspace: mvData });
            }
            if (mb.structureFile && typeof mb.structureFile.configure === "function") {
                mb.structureFile.configure({ workspace: mvData });
            }
        })();

        // (The old store-loader injection is gone: the store no longer holds a
        // structure loader -- the render engine draws from molview.data, and file
        // loads flow through the unified ``molview.data.installMolecule`` door.)

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
        const store    = mvData.selection;
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
            // (The old `commitCandidate` -- a reach-around that
            // RELOADED the file instead of going through `_commitFile`
            // -> `openMolecule` -- was dead and is removed; loading
            // goes through commitFile only.)
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

        // "Commit a structure file into the workspace" -- a THIN wrapper over the ONE
        // format-aware load door, ``projects.parser.openMolecule`` (structure-load-save-
        // contract.md).  All the composition -- read the .xyz + .molstruct.json via the
        // projects file package, parse + apply the sidecar server-side, install through
        // the model primitive in ONE store write, tolerate a missing sidecar -- lives in
        // the parser door now, so this tab neither hand-wires the seam nor reaches around
        // a door into the store.  The tab's only jobs: inject the dirty-canvas WARNING (a
        // UI concern the DOM-free layer can't own) and surface an error banner.  Used by
        // the Load button + the sidebar dblclick (onCommit).
        async function _commitFile(path) {
            if (!path) return;
            const proj = window.molbuilder && window.molbuilder.projects;
            if (!proj || !proj.parser
                    || typeof proj.parser.openMolecule !== "function") {
                // No parser door on this page -- the page can't load a structure
                // without it (the old store-side Path-A load-fetch is gone).
                // Nothing to fall back to; warn and no-op.
                if (window.console && window.console.warn) {
                    window.console.warn(
                        "[selection-bootstrap] cannot commit file: projects "
                        + "parser door (openMolecule) not available on this page");
                }
                return;
            }
            const warn = window.molbuilder && window.molbuilder.warningModal;
            const confirmDiscard =
                (warn && typeof warn.confirmDiscardUnsaved === "function")
                    ? () => warn.confirmDiscardUnsaved() : null;
            const res = await proj.parser.openMolecule(path, { confirmDiscard });
            if (res && res.ok === false && !res.cancelled && res.error) {
                const s = document.getElementById("status");
                if (s) { s.textContent = res.error; s.className = "status error"; }
            }
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
