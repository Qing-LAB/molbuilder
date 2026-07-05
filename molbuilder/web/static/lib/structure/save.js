/* Save panel — Molbuilder-tab workspace export.
 *
 * Writes the canvas-state contents back to disk.  The natural
 * target depends on how the canvas got loaded:
 *
 *   * kind="file"  — write back to ``source.file`` (the path the
 *                     structure was loaded from).
 *   * any other    — write back to ``last_save_to`` if a prior
 *                     Save has happened in this session; otherwise
 *                     the panel disables ("Save as" comes later).
 *
 * On a successful write the orchestrator's ``markSavedTo`` runs —
 * the dirty bit clears, ``last_save_to`` is recorded, and any
 * subsequent Load / Generate WILL NOT fire the warning modal
 * unless the user re-edits.
 *
 * Surface (``window.molbuilder.structureSave``):
 *
 *   save() -> Promise<{ok: bool, path?: string,
 *                      error?: string}>
 *     Perform a save against the currently-known target.  Refuses
 *     when there's no canvas content or no target path.
 *
 *   targetPath() -> string | null
 *     The path Save would write to right now (so the button readout
 *     and tooltip can show it before the user clicks).
 *
 *   configure(opts)
 *     Test seam — inject fake projects, structurePage, canvas.
 *
 *   wirePanel(opts?)
 *     Idempotent DOM wiring against #save-to-source-btn /
 *     #save-readout / #save-status + the canvas-state change
 *     subscription that keeps the readout + button state live.
 *
 * Design ref: docs/tabs/architecture.md § 5.6 (Save options).
 */
(function (root) {
    "use strict";

    var _projects      = null;
    var _structurePage = null;
    // Phase 10 (workspace-contract.md §8): the canvas reference is
    // now the unified ws.* surface, not the legacy structureCanvas
    // store.  Same getter set (getLastSavedTo / getSource / isEmpty
    // / getStructure / isDirty / subscribe) so the rest of this file
    // is unchanged in shape.  Old ``canvas`` opt is still accepted
    // by configure() for test contexts that pass a fake — the test
    // fake just needs the ws.* method names.
    var _workspace     = null;

    function configure(opts) {
        opts = opts || {};
        if (opts.projects)      _projects      = opts.projects;
        if (opts.structurePage) _structurePage = opts.structurePage;
        // Accept either ``workspace`` (canonical, Phase 10+) or
        // ``canvas`` (legacy alias kept for tests still passing the
        // old fake).  Both name the same object.
        if (opts.workspace)     _workspace     = opts.workspace;
        else if (opts.canvas)   _workspace     = opts.canvas;
    }

    /**
     * Resolve the production-singleton dependencies lazily.  The
     * projects sidebar mounts as a deferred ESM module, so it
     * isn't on ``window.molbuilder`` at the moment save.js's IIFE
     * runs — this re-resolves whichever slots are still null at
     * call time.  Test contexts that called configure() with
     * explicit fakes are unaffected (their values stay).
     */
    function _lazyResolve() {
        if (typeof root === "undefined" || !root.molbuilder) return;
        if (!_projects      && root.molbuilder.projects)
            _projects      = root.molbuilder.projects;
        if (!_structurePage && root.molbuilder.structurePage)
            _structurePage = root.molbuilder.structurePage;
        if (!_workspace     && root.molbuilder.workspace)
            _workspace     = root.molbuilder.workspace;
    }

    /**
     * Resolve the natural save target.  Order:
     *   1. ``last_save_to`` if previously saved this session.
     *   2. ``source.file`` if the canvas was loaded from a project file.
     *   3. null → Save disabled, user needs Save-as (future).
     */
    function targetPath() {
        _lazyResolve();
        if (!_workspace) return null;
        var lastSaved = _workspace.getLastSavedTo();
        if (lastSaved) return lastSaved;
        var src = _workspace.getSource();
        if (src && src.kind === "file" && src.file) return src.file;
        return null;
    }

    function _basename(p) {
        if (!p) return "";
        var ix = p.lastIndexOf("/");
        return ix >= 0 ? p.slice(ix + 1) : p;
    }

    function _dirname(p) {
        if (!p) return "";
        var ix = p.lastIndexOf("/");
        return ix > 0 ? p.slice(0, ix) : "";
    }

    /**
     * Gather regions + frozen_atoms from the workspace's current
     * atom list.  Used by ``_postWriteSuccess`` to propagate
     * workspace labels into the destination sidecar on Save-as
     * (different path from the source).
     *
     * Returns ``{regions: {label: [indices]}, frozen: [indices]}``.
     * Skips atoms without labels + non-frozen atoms; the result
     * is empty when the workspace has no label data.
     */
    function _gatherLabelsFromWorkspace() {
        if (!_workspace || typeof _workspace.getAtoms !== "function") {
            return { regions: {}, frozen: [], periodicity: null };
        }
        var atoms  = _workspace.getAtoms() || [];
        var regions = {};
        var frozen  = [];
        for (var i = 0; i < atoms.length; i++) {
            var a = atoms[i];
            if (!a) continue;
            var labels = a.labels || [];
            for (var j = 0; j < labels.length; j++) {
                var L = labels[j];
                if (!regions[L]) regions[L] = [];
                regions[L].push(i);
            }
            if (a.isFrozen) frozen.push(i);
        }
        // Periodicity rides with the whole-store save (workspace-contract.md
        // §4.0): read it from the store so Save writes the same cell/axis_kind/
        // vacuum/kgrid the user sees.
        var per = null;
        if (typeof _workspace.getStructure === "function") {
            var s = _workspace.getStructure();
            per = (s && s.periodicity) || null;
        }
        return { regions: regions, frozen: frozen, periodicity: per };
    }

    /**
     * Atomically REPLACE the destination sidecar with the whole store
     * (regions + frozen + periodicity).  Single HTTP call via
     * ``/api/selection/save-sidecar`` (REPLACE-all).
     *
     * workspace-contract.md §4.0: the store is authoritative -- a save writes
     * the whole store, wiping any stale sidecar at the destination.  Fires on
     * EVERY save (see _postWriteSuccess), not just save-as.
     */
    function _persistLabelsToDestination(path, labels, nAtoms) {
        if (!root.fetch) return Promise.resolve();
        return root.fetch("/api/selection/save-sidecar", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                structure_path: path,
                n_atoms:        nAtoms,
                regions:        labels.regions,
                frozen_atoms:   labels.frozen,
                periodicity:    labels.periodicity,
            }),
        }).then(function (r) { return r.json(); })
          .catch(function () { /* non-fatal */ });
    }

    /**
     * Common post-write success path.  Clears dirty + fires the
     * sidecar housekeeping.  Returns the user-facing ok envelope.
     *
     * Sidecar handling per save-flow.md §4:
     *   * Save back to source (final == source) — refresh-hash on
     *     the existing sidecar.
     *   * Save-as (final != source) — propagate workspace labels
     *     to a new sidecar at the destination, THEN refresh-hash
     *     so the hash field matches the just-written XYZ.
     */
    function _postWriteSuccess(path, opts) {
        opts = opts || {};
        var sourcePath = opts.sourcePath || null;
        var isSaveAs = !!sourcePath && path !== sourcePath;

        _structurePage.markSavedTo(path);

        // 2026-06-09: re-anchor the selection store's ``sourceFile`` to
        // the just-written path so subsequent panel label writes
        // (writeLabel) target the NEW location's sidecar.  Without
        // this, ``state.sourceFile`` keeps pointing at the original
        // load path (/A/water.xyz) even after Save-as to /B — and any
        // label added afterwards lands at /A's sidecar, surprising
        // the user who thinks they're editing /B.
        //
        // ``adoptSession`` with the workspace's CURRENT atoms +
        // selection preserves the in-memory state (no _fetchAtoms
        // disk re-read that would clobber unsaved modifier ops).
        //
        // IMPORTANT: adoptSession's internal ``_run`` calls
        // ``_newSignal`` which ABORTS any in-flight operation in the
        // selection store.  Capture labels BEFORE calling it so a
        // user who clicked Assign + Save in rapid succession doesn't
        // lose the just-assigned label to the abort — the labels
        // gathered from ws.getAtoms() before adoptSession reflect
        // whatever writeLabel had already applied in-place.
        var labels = (_workspace
                      && typeof _workspace.getAtoms === "function")
            ? _gatherLabelsFromWorkspace()
            : { regions: {}, frozen: [], periodicity: null };
        var nAtoms = (_workspace
                      && typeof _workspace.getAtoms === "function")
            ? (_workspace.getAtoms() || []).length : 0;

        if (_workspace
                && _workspace.selection
                && typeof _workspace.selection.adoptSession === "function") {
            var currentSel = (_workspace.getSelection
                              && _workspace.getSelection().indices) || [];
            var currentAtoms = (_workspace.getAtoms
                                && _workspace.getAtoms()) || [];
            try {
                _workspace.selection.adoptSession({
                    sourceFile: path,
                    selection:  currentSel,
                    atoms:      currentAtoms,
                });
            } catch (_) { /* non-fatal — selection re-anchor is housekeeping */ }
        }

        if (!root.fetch) return { ok: true, path: path };

        // §4.0: EVERY save writes the WHOLE store's sidecar (regions + frozen +
        // periodicity) from memory -- not just on save-as.  The store is the
        // truth, so we bulk-REPLACE the destination sidecar unconditionally
        // (this also WIPES any stale sidecar at the destination).  ``labels`` +
        // ``nAtoms`` were captured BEFORE the adoptSession abort (see above).
        // ``isSaveAs`` no longer gates the write; it is kept only for callers
        // that read the return provenance.
        void isSaveAs;
        var labelsPromise = _persistLabelsToDestination(path, labels, nAtoms);

        function _fireRefreshHash() {
            return root.fetch("/api/selection/refresh-hash", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ structure_path: path }),
            }).catch(function () { /* non-fatal */ });
        }

        if (labelsPromise) {
            // refresh-hash AFTER labels land so the sidecar's
            // structure_hash matches the just-written XYZ.
            labelsPromise.then(_fireRefreshHash);
        } else {
            _fireRefreshHash();
        }

        return { ok: true, path: path };
    }

    /**
     * Inner write loop: POST writeFile with ``overwrite=false``;
     * on 409 (file-exists), show the confirm-overwrite dialog and
     * retry with ``overwrite=true`` if the user confirms.
     *
     * Used by both ``save()`` (current target) and any future
     * Save-as path (user-typed name).  Returns the user-facing ok
     * envelope.
     */
    function _writeWithOverwriteGate(path, text, opts) {
        opts = opts || {};
        var alreadyConfirmed = !!opts.overwriteAlreadyConfirmed;
        var sourcePath = opts.sourcePath || null;
        var postOpts = { sourcePath: sourcePath };
        var dialog = root.molbuilder
                  && root.molbuilder.structureSaveDialog;
        return _projects.writeFile(path, text, {
            overwrite: alreadyConfirmed,
        }).then(function (r) {
            if (r && r.ok) return _postWriteSuccess(path, postOpts);
            // 409 (file exists, overwrite was false).  Re-prompt.
            if (r && (r.status === 409
                    || /already exists/i.test(r.error || ""))
                    && !alreadyConfirmed
                    && dialog
                    && typeof dialog.confirmOverwrite === "function") {
                return dialog.confirmOverwrite(_basename(path))
                    .then(function (proceed) {
                        if (!proceed) {
                            return { ok: false, cancelled: true,
                                     error: "Save cancelled." };
                        }
                        return _projects.writeFile(path, text, {
                            overwrite: true,
                        }).then(function (r2) {
                            if (r2 && r2.ok) return _postWriteSuccess(path, postOpts);
                            return {
                                ok:    false,
                                error: (r2 && r2.error) || "Save failed.",
                            };
                        });
                    });
            }
            return {
                ok:    false,
                error: (r && r.error) || "Save failed.",
            };
        }, function (err) {
            return {
                ok:    false,
                error: "Save failed: "
                     + (err && err.message ? err.message : String(err)),
            };
        });
    }

    function save() {
        _lazyResolve();
        if (!_workspace) {
            return Promise.reject(new Error(
                "save: workspace not configured"));
        }
        if (!_projects
            || typeof _projects.writeFile !== "function") {
            return Promise.reject(new Error(
                "save: projects.writeFile not configured"));
        }
        if (!_structurePage) {
            return Promise.reject(new Error(
                "save: structurePage not configured"));
        }
        if (_workspace.isEmpty()) {
            return Promise.resolve({
                ok: false, error: "No structure to save." });
        }
        var struct = _workspace.getStructure();
        var dialog = root.molbuilder
                  && root.molbuilder.structureSaveDialog;

        // 2026-06-09: the dialog only confirms the FILENAME.  The
        // directory is ALWAYS the sidebar's current project dir,
        // regardless of whether the workspace was loaded from a
        // file (in which case the original file's dir is ignored)
        // or generated (SMILES / DNA / RNA / peptide / name).  This
        // keeps the save semantics simple: "save the current
        // workspace into the project I'm looking at."
        var dir = (_projects
                   && typeof _projects.getCurrentDir === "function")
            ? (_projects.getCurrentDir() || "")
            : "";
        if (!dir) {
            return Promise.resolve({
                ok: false,
                error: "Pick a project directory in the sidebar "
                     + "before saving.",
            });
        }

        // save-flow.md §1: NO default filename.  The Modify tab exists to
        // MODIFY the structure, so a Save is a save-AS to a file the user names;
        // we never pre-fill the loaded file's name (that invites silently
        // overwriting the source).  Blank box -> the dialog keeps Save disabled
        // until the user types a name.  ``path`` (the loaded source) is still
        // threaded below, only for provenance (sidecar propagation).
        var path = targetPath();
        var initial = "";

        // Route the Save click through the confirm-name dialog so
        // the user can edit the filename + see what they're about
        // to overwrite.  Renaming OR saving into a different
        // sidebar dir triggers the overwrite-gate; pre-confirm only
        // when the chosen final path matches the workspace's
        // existing source (= clicking Save back to the file we
        // loaded from, in the same directory).
        if (!dialog || typeof dialog.chooseSaveName !== "function") {
            // No dialog mounted.  With no default filename (§1) there is no name
            // to save to without the dialog, so the save cannot proceed.
            return Promise.resolve({
                ok: false,
                error: "Save needs the name dialog to choose a filename.",
            });
        }
        return dialog.chooseSaveName(initial).then(function (chosen) {
            if (chosen === null || chosen === undefined) {
                return { ok: false, cancelled: true,
                         error: "Save cancelled." };
            }
            var finalPath = dir + "/" + chosen;
            // Pre-confirm overwrite ONLY when the chosen final path
            // exactly matches the workspace's existing source file
            // — clicking Save back to the file we loaded from (same
            // dir + same name) is unambiguous.  Any other path
            // (renamed, different sidebar dir, generator save-as)
            // routes through the overwrite-gate.  ``sourcePath`` is
            // threaded into _postWriteSuccess so the sidecar-
            // propagation logic (save-flow.md §4.3) knows whether
            // this is a Save-as or save-back-to-source.
            return _writeWithOverwriteGate(finalPath, struct.text, {
                // save-flow.md §1: overwrite is ALWAYS confirmed -- no
                // save-back-to-source skip.  Any existing name re-prompts.
                overwriteAlreadyConfirmed: false,
                sourcePath:                path,
            });
        });
    }

    var _wired = false;
    var _unsubCanvas = null;
    // 2026-06-12: while a Save click is in flight, the workspace's
    // subscriber callback fires DURING ``_postWriteSuccess`` (because
    // ``markSavedTo`` + ``adoptSession`` both call ws.notify()).  The
    // subscriber re-runs ``refreshState`` mid-save, which would re-
    // enable the button on a successful write — letting the user
    // click Save AGAIN before the first save's label propagation +
    // refresh-hash had finished and triggering a second dialog/POST
    // for the same workspace.  Track the in-flight state at module
    // scope and OR it into the disabled computation so the button
    // stays disabled until the click handler's .then() clears the
    // flag.  Cleared on both success and failure paths.
    var _saveInFlight = false;
    function wirePanel(opts) {
        opts = opts || {};
        var doc = opts.doc || root.document;
        if (!doc || _wired) return;
        _wired = true;

        var button  = doc.getElementById("save-to-source-btn");
        var readout = doc.getElementById("save-readout");
        var status  = doc.getElementById("save-status");
        if (!button) return;

        function _basename(p) {
            var i = p.lastIndexOf("/");
            return i >= 0 ? p.slice(i + 1) : p;
        }
        function refreshState() {
            _lazyResolve();
            var path = targetPath();
            var dirty = _workspace && _workspace.isDirty();
            // 2026-06-09: Save-as for generator-sourced workspaces.
            // When ``path`` is null (no backing file), allow Save
            // iff the workspace has content AND the sidebar has a
            // current directory to save into.  The Save() function
            // surfaces a clear error if either is missing at click
            // time; this just keeps the button visibly enabled when
            // a Save-as is actually possible.
            var hasContent = _workspace && !_workspace.isEmpty();
            var sidebarDir = (_projects
                              && typeof _projects.getCurrentDir
                                 === "function")
                ? (_projects.getCurrentDir() || "")
                : "";
            button.disabled = _saveInFlight
                              || !hasContent
                              || (!path && !sidebarDir);
            if (readout) {
                if (path) {
                    readout.textContent = (dirty ? "Unsaved — " : "")
                                        + "Target: " + _basename(path);
                } else if (hasContent && sidebarDir) {
                    readout.textContent =
                        "Save as… into " + _basename(sidebarDir) + "/";
                } else if (hasContent) {
                    readout.textContent =
                        "Pick a project directory in the sidebar to "
                        + "Save as…";
                } else {
                    readout.textContent = "";
                }
            }
        }
        function setStatus(msg, kind) {
            if (!status) return;
            status.textContent = msg || "";
            status.className = "muted"
                + (kind === "error" ? " is-error" : "");
        }

        // Initial paint + live updates from the workspace.
        // workspace-contract.md §2.1: ``ws.subscribe(fn)`` fires once
        // on subscribe AND on every notify() tick — same contract
        // the old ``canvas.onChange`` had.  No additional initial-
        // paint call is needed (subscribe fires synchronously).
        refreshState();
        if (_workspace && typeof _workspace.subscribe === "function") {
            // Replace any prior subscription on re-wire.
            if (typeof _unsubCanvas === "function") _unsubCanvas();
            _unsubCanvas = _workspace.subscribe(function () {
                refreshState();
            });
        }

        button.addEventListener("click", function () {
            _saveInFlight = true;
            button.disabled = true;
            setStatus("Saving…");
            save().then(function (r) {
                _saveInFlight = false;
                if (r.ok) {
                    setStatus("Saved " + _basename(r.path) + ".");
                } else {
                    setStatus(r.error || "Save failed.", "error");
                }
                refreshState();   // re-enable + update dirty hint
            }, function (err) {
                // Save's contract is to return an envelope, never
                // reject — but a misconfigured module (canvas /
                // projects / structurePage missing at call time)
                // throws synchronously inside the Promise chain.
                // Catch so the button reenables and the user
                // sees what's wrong instead of a hung "Saving…".
                _saveInFlight = false;
                setStatus(
                    "Save failed: " + (err && err.message
                                       ? err.message : String(err)),
                    "error");
                refreshState();
            });
        });
    }

    var api = {
        configure:  configure,
        save:       save,
        targetPath: targetPath,
        wirePanel:  wirePanel,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.structureSave = api;
        configure({
            projects:      root.molbuilder.projects,
            structurePage: root.molbuilder.structurePage,
            // workspace-contract.md §1: the dispatcher is the unified
            // surface; this module no longer reads structureCanvas
            // directly.  Lazy-resolve in ``_lazyResolve`` also picks
            // up window.molbuilder.workspace at call time so a late-
            // arriving dispatcher mount still works.
            workspace:     root.molbuilder.workspace,
        });
        if (root.document) {
            if (root.document.readyState === "loading") {
                root.document.addEventListener(
                    "DOMContentLoaded", function () { wirePanel(); });
            } else {
                wirePanel();
            }
        }
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.save", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
