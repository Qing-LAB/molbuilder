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
 * On a successful write the save door's ``molview.data.markSaved`` runs —
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
    var _model     = null;

    function configure(opts) {
        opts = opts || {};
        if (opts.projects)      _projects      = opts.projects;
        if (opts.structurePage) _structurePage = opts.structurePage;
        // Accept either ``workspace`` (canonical, Phase 10+) or
        // ``canvas`` (legacy alias kept for tests still passing the
        // old fake).  Both name the same object.
        if (opts.workspace)     _model     = opts.workspace;
        else if (opts.canvas)   _model     = opts.canvas;
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
        // _model = molview.data, LOOKED UP live (molview-module.md §D.0) -- renamed from
        // the misleading pre-carve ``_workspace`` (the WORKSPACE has no data API); reading is
        // decoupled from loading.  A test injects a stub via configure({workspace}).
        if (!_model && root.molbuilder.molview && root.molbuilder.molview.data)
            _model = root.molbuilder.molview.data;
    }

    /**
     * Resolve the natural save target.  Order:
     *   1. ``last_save_to`` if previously saved this session.
     *   2. ``source.file`` if the canvas was loaded from a project file.
     *   3. null → Save disabled, user needs Save-as (future).
     */
    function targetPath() {
        _lazyResolve();
        if (!_model) return null;
        var lastSaved = _model.getLastSavedTo();
        if (lastSaved) return lastSaved;
        var src = _model.getSource();
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

    // D3 (workspace-contract §1.4): the model serialises itself via
    // ``molview.data.exportFile()`` (built from the §1.2.1 accessors).  save.js no longer
    // hand-rolls the regions/frozen scan -- the old ``_gatherLabelsFromWorkspace`` +
    // ``_buildScratchBlob`` were exactly the duplicate-of-the-dispatcher access this
    // contract removes.

    /**
     * THE save -- a THIN wrapper over the ONE format-aware save door,
     * ``projects.parser.saveMolecule`` (structure-load-save-contract.md).  The door
     * serialises the SETTLED model (molview.data.exportFile), writes BOTH files --
     * ``.xyz`` + ``.molstruct.json`` -- through the projects file package
     * (projects.writeFile), and marks saved (canvas dirty=false + store source
     * re-anchor).  save.js no longer POSTs an endpoint or pokes the store itself.
     *
     * Overwrite is UI policy and stays HERE: the door returns ``needsOverwrite`` when
     * the target exists; we confirm via the dialog and retry with ``overwrite:true``.
     * A failure is SURFACED (returned), never swallowed.
     */
    function _saveDataset(path) {
        var proj  = root.molbuilder && root.molbuilder.projects;
        var saver = proj && proj.parser;
        if (!saver || typeof saver.saveMolecule !== "function") {
            return Promise.resolve({ ok: false, error: "save: parser door unavailable" });
        }
        var dialog = root.molbuilder && root.molbuilder.structureSaveDialog;
        return saver.saveMolecule(path, { overwrite: false }).then(function (r) {
            if (r.ok) return { ok: true, path: r.path };
            if (r.needsOverwrite && dialog
                    && typeof dialog.confirmOverwrite === "function") {
                return dialog.confirmOverwrite(_basename(path)).then(function (proceed) {
                    if (!proceed) {
                        return { ok: false, cancelled: true, error: "Save cancelled." };
                    }
                    return saver.saveMolecule(path, { overwrite: true })
                        .then(function (r2) {
                            return r2.ok
                                ? { ok: true, path: r2.path }
                                : { ok: false, error: r2.error || "Save failed." };
                        });
                });
            }
            return { ok: false, error: r.error || "Save failed." };
        });
    }

    function save() {
        _lazyResolve();
        if (!_model) {
            return Promise.reject(new Error(
                "save: workspace not configured"));
        }
        if (!_projects
            || typeof _projects.getCurrentDir !== "function") {
            return Promise.reject(new Error(
                "save: projects.getCurrentDir not configured"));
        }
        if (!_structurePage) {
            return Promise.reject(new Error(
                "save: structurePage not configured"));
        }
        if (_model.isEmpty()) {
            return Promise.resolve({
                ok: false, error: "No structure to save." });
        }
        var struct = _model.getStructure();
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
            // The user names the structure; the save produces the PAIR
            // ``<name>.xyz`` (coordinates) + ``<name>.molstruct.json`` (labels/
            // regions).  Both filenames come from the ONE base name, so we own
            // the extension: strip any suffix the user typed anyway (forgiving)
            // and append ``.xyz``.  Passing a bare ``<name>`` (no .xyz) would
            // write a coordinate file that can't be reloaded (StructureCodec.load
            // dispatches on the .xyz/.pdb suffix).
            var base = String(chosen).trim()
                .replace(/\.molstruct\.json$/i, "")
                .replace(/\.xyz$/i, "");
            if (!base) {
                return { ok: false, error: "Please enter a file name." };
            }
            var finalPath = dir + "/" + base + ".xyz";
            // ONE save door: _saveDataset -> projects.parser.saveMolecule, which
            // writes the whole dataset via the file-only save door (POST /api/structure/save;
            // the SERVER writes the .xyz + .molstruct.json pair).  Overwrite is ALWAYS confirmed inside _saveDataset
            // (needsOverwrite -> confirm -> retry with overwrite:true); no save-back skip.
            return _saveDataset(finalPath).then(function (res) {
                // On a successful write, refresh the sidebar listing so the new
                // <name>.xyz (+ its sidecar) appears without a manual reload --
                // the save door does auto-refresh the current dir, but
                // this is a belt-and-braces refresh for the save target.  Best-effort;
                // a refresh failure never fails the save.
                if (res && res.ok && _projects
                        && typeof _projects.refresh === "function") {
                    try { Promise.resolve(_projects.refresh()).catch(function () {}); }
                    catch (_) { /* refresh is cosmetic; never throw here */ }
                }
                return res;
            });
        });
    }

    var _wired = false;
    var _unsubCanvas = null;
    // 2026-06-12: while a Save click is in flight, the workspace's
    // subscriber callback fires DURING the save (the door's
    // ``molview.data.markSaved`` clears dirty + re-anchors the store -> notify()).  The
    // subscriber re-runs ``refreshState`` mid-save, which would re-
    // enable the button on a successful write — letting the user
    // click Save AGAIN before the first save's saveMolecule
    // (the file-only save door) had finished and triggering a second
    // dialog/write for the same workspace.  Track the in-flight state at module
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
            var dirty = _model && _model.isDirty();
            // 2026-06-09: Save-as for generator-sourced workspaces.
            // When ``path`` is null (no backing file), allow Save
            // iff the workspace has content AND the sidebar has a
            // current directory to save into.  The Save() function
            // surfaces a clear error if either is missing at click
            // time; this just keeps the button visibly enabled when
            // a Save-as is actually possible.
            var hasContent = _model && !_model.isEmpty();
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
        if (_model && typeof _model.subscribe === "function") {
            // Replace any prior subscription on re-wire.
            if (typeof _unsubCanvas === "function") _unsubCanvas();
            _unsubCanvas = _model.subscribe(function () {
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
            // workspace (molview.data) is LOOKED UP in _lazyResolve at call time; a test
            // may inject a stub here via configure({workspace}).
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
