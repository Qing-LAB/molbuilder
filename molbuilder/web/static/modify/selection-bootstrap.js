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
 * in the module/store.  See docs/web/molview.md for the full
 * architecture.
 */
import { mount } from "/static/lib/molview/index.js";
import { init as startOpControls } from "./viewer.js";
import { init as startCellPanel }  from "./periodicity.js";

/* THIS FILE IS THE MOLBUILDER TAB'S OWNER. It mounts the one viewer this page
 * has and hands it to everything else on the page.
 *
 * The tab is split across several files, and each of them used to find the
 * viewer by name in `window.molbuilder.molview.data`. MolView publishes nothing
 * there (molview.md § 4) — a viewer belongs to whoever mounted it and the handle
 * is the only way to one (§ 5.6) — so every one of them was reading `undefined`.
 *
 * They are started HERE, after the mount, rather than each waking on its own
 * DOMContentLoaded. That is not tidiness: the load order put two of them before
 * this file, so they ran before any viewer existed and there was no arrangement
 * that could have fixed it while they were looking a viewer up rather than being
 * given one. */

(function () {
    "use strict";

    // EMPTY host: molview.mount BUILDS the whole fused card into it (viewer + panel +
    // View-menu toggles + measurement + fold), exactly like the demo / Results.  Modify
    // no longer hand-builds a card, so the mount takes the module's build path.
    const HOST_ID     = "molview-host";
    /* WHO THESE BYTES BELONG TO (workspace.md § 4) — the one string used BOTH as the
     * viewer's `owner` and as the tag on every workspace call, so the two cannot drift
     * into naming different slots. */
    const WORKSPACE_TAG = "modify";

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
        /* THE CHECK THAT KILLED THIS TAB. It tested for a viewer BEFORE
         * mounting one — a name looked up in a global that MolView has published
         * nothing to since it was rebuilt — so it failed every time and returned
         * without ever calling `mount`. The Molbuilder tab has had no viewer at
         * all since then. What is worth checking here is the import itself. */
        if (typeof mount !== "function") {
            console.error("[selection-bootstrap] molview import missing");
            _renderFailure(host, "molview module missing");
            return;
        }
        const _mounted = await mount(host, window.molbuilder.workspace, {
            mode: "modify",
            owner: WORKSPACE_TAG,
        });
        if (!_mounted || !_mounted.ok) return;   // mount contract: failure -> {ok:false}; it warned already

        /* THE REST OF THE PAGE, STARTED WITH THE VIEWER IT NEEDS. */
        // The op controls put this tag's last saved point back on the canvas as
        // the last thing they do, and that is still in flight when they return.
        // Held here and awaited below, where the answer is needed.
        const _restoring = startOpControls(_mounted);   // edit ops + state timeline
        startCellPanel(_mounted);      // the Cell op-tab's periodicity editors
        // The classic generator/save scripts loaded before this file and cannot
        // import; they are handed the viewer through their own bind door.
        const page = window.molbuilder && window.molbuilder.structurePage;
        if (page && typeof page.useViewer === "function") page.useViewer(_mounted);
        const save = window.molbuilder && window.molbuilder.structureSave;
        if (save && typeof save.useViewer === "function") save.useViewer(_mounted);

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
        // docs/web/tabs.md
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
        const store    = _mounted.data.selection;
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
        // Which file is actually ON the canvas, as opposed to picked in the
        // sidebar.  The page's own note — the viewer tracks contents, not files
        // (molview.md § 6.7).  Set in `_commitFile`; read by `_refreshLoadUI`.
        /* WHICH FILE IS ON THE CANVAS lives in `structurePage`, under the page's
         * own tag (workspace.md § 4's `modify:panel`), so it survives a reload
         * along with everything else the page did. This file only reads it. */
        const _loadedFrom = () =>
            (window.molbuilder.structurePage.getLoadedFrom() || "");
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
            // WHAT IS ALREADY HERE WINS.  Whatever this tab held when you left
            // it -- loaded from a file or generated (SMILES / RNA / peptide /
            // name) -- comes back on the canvas, and the sidebar's highlighted
            // file must NOT overwrite it.  Loading a file is an explicit action
            // (the Load button, or a double-click), never a side effect of a
            // file still being highlighted from last time.
            //
            // The sidebar seeds the canvas ONLY when the canvas came up empty:
            // that is the genuine "I came here to work on this file" case.  What
            // is asked is "are there atoms?", never "does the file on the canvas
            // match the one in the sidebar" -- a generated structure has atoms
            // and no file at all, so the file comparison read it as empty and
            // wiped it (the RNA-wipe bug).
            /* WE ASK THE CANVAS, NOT THE FILING CABINET.
             *
             * Wait for the restore, then look at what is actually on the canvas:
             * `getStructure()` answers null when nothing is loaded (molview.md
             * § 9.3), and atoms are what "there is work here" means.
             *
             * This used to read the saved bytes back out of the workspace and
             * inspect them — through `readPersistedSnapshot`, a door the
             * workspace no longer has. It was called behind a `typeof` guard, so
             * it quietly answered "nothing saved" on every visit, and the
             * sidebar's highlighted file was free to overwrite restored work. */
            await _restoring;
            /* AND THE PAGE'S OWN NOTE COMES BACK TOO. The viewer restores the
             * molecule; this restores what the PAGE did with it -- which file it
             * came from, where it was saved. Without it a restored session shows
             * a structure while the loader readout says nothing is loaded, and
             * the Load button re-enables against the very file the work came
             * from. */
            await page.restorePanelNote();
            const _onCanvas = _mounted.data.getStructure();
            const _hasRestore = !!(_onCanvas
                && (_onCanvas.elements || []).length);
            if (_isLoadableStructure(initial) && !_hasRestore) {
                // Empty canvas + a loadable sidebar pick: seed it on mount.
                // Goes through the same _commitFile path as the Load button
                // so canvas-state stays in sync.
                _commitFile(initial);
                _setCandidate(initial);
            } else if (_isLoadableStructure(initial)) {
                // A restorable snapshot exists (file-based OR generated) --
                // the restore owns the canvas.  Reflect the candidate for the
                // Load-button readout, but DO NOT commit (defer to
                // restoreModifyState; the user loads explicitly if they want
                // to swap to this file).
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
        /* WHICH FILE IS ON THE CANVAS — THE PAGE'S OWN NOTE.
         *
         * The readout tells "Picked" (chosen in the sidebar, not committed)
         * apart from "Loaded" (already on the canvas), and the Load button goes
         * dead for the second, so a user can tell a no-op click from a real one.
         *
         * It asked the selection snapshot for `sourceFile`. No snapshot has ever
         * had that key — and it correctly never will, because the viewer tracks
         * contents, not files (molview.md § 6.7). So the comparison was always
         * against `""`, the readout said "Picked" straight after a load, and the
         * button never went dead. The page performed the load, so the page is
         * what knows; this is the same note `lastSaveTo` is for the other
         * direction. It is declared up with `_candidate` — the mount seed calls
         * `_commitFile` before this point in the file. */
        function _refreshLoadUI() {
            const isLoaded = !!_candidate && _candidate === _loadedFrom();
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
        /* Re-paint when what is ON THE CANVAS changes, not only when the
         * sidebar pick does — an edit or a Retract leaves the file it came from
         * unchanged, but a load through any route (the button, a double-click,
         * the mount seed) has to reach the readout. The note is set in
         * `_commitFile`; this keeps the two in step. */
        if (typeof _mounted.data.subscribe === "function") {
            _mounted.data.subscribe(_refreshLoadUI);
        }
        // AND when the page's own note changes, which the viewer knows nothing
        // about: a generate replaces the structure AND clears the filename, and
        // only the second of those reaches the readout through this channel.
        if (page && typeof page.onPanelChange === "function") {
            page.onPanelChange(_refreshLoadUI);
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
            const res = await proj.parser.openMolecule(
                _mounted, path, { confirmDiscard });
            const s = document.getElementById("status");
            if (!s) return;
            if (res && res.ok === false && !res.cancelled && res.error) {
                s.textContent = res.error;
                s.className = "status error";
                return;
            }
            if (res && res.cancelled) return;    // the user kept what was there
            // This page performed the load, so this page is what knows which
            // file is on the canvas (§ 6.7).  Recorded under the page's own tag,
            // so a reopened tab still knows; both readouts below read it back.
            window.molbuilder.structurePage.markLoadedFrom(path);
            _refreshLoadUI();
            /* SAY WHAT LANDED. The line only ever spoke up when a load FAILED,
             * so after a successful one it still read "No structure loaded." —
             * the template's opening text — beside a drawn molecule. The count
             * comes from the viewer, which is the thing that now holds it. */
            const n = (_mounted.data.getElements() || []).length;
            s.textContent = `Loaded ${_basename(path)} — ${n} atoms.`;
            s.className = "status ok";
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
