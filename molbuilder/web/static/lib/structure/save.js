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
     * Common post-write success path.  Clears dirty + fires the
     * sidecar refresh-hash housekeeping.  Returns the user-facing
     * ok envelope.
     */
    function _postWriteSuccess(path) {
        _structurePage.markSavedTo(path);
        // 2026-06-09: refresh the sidecar's structure_hash against
        // the just-written XYZ bytes.  Fire-and-forget; a sidecar-
        // missing path is a no-op server-side.
        if (root.fetch) {
            root.fetch("/api/selection/refresh-hash", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ structure_path: path }),
            }).catch(function () { /* non-fatal */ });
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
        var dialog = root.molbuilder
                  && root.molbuilder.structureSaveDialog;
        return _projects.writeFile(path, text, {
            overwrite: alreadyConfirmed,
        }).then(function (r) {
            if (r && r.ok) return _postWriteSuccess(path);
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
                            if (r2 && r2.ok) return _postWriteSuccess(path);
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
        var path = targetPath();
        if (!path) {
            return Promise.resolve({
                ok: false,
                error: "This structure was built in the workspace "
                     + "and hasn't been written to a project file "
                     + "yet — Save as… will be available soon.",
            });
        }
        var struct = _workspace.getStructure();
        var dialog = root.molbuilder
                  && root.molbuilder.structureSaveDialog;

        // 2026-06-09: route the Save click through the confirm-name
        // dialog so the user can edit the filename + see what they're
        // about to overwrite.  If the user keeps the current name AND
        // the file at the resolved path already exists (the common
        // case for "save back to source"), ``_writeWithOverwriteGate``
        // pre-confirms because the user already saw the name they're
        // saving to.  Renaming triggers a fresh overwrite-check
        // against the new name.
        var initial = _basename(path);
        if (!dialog || typeof dialog.chooseSaveName !== "function") {
            // No dialog mounted (tests / legacy contexts).  Fall back
            // to the legacy "always overwrite source" behaviour so
            // tests + headless callers don't deadlock waiting for a
            // modal that will never appear.
            return _writeWithOverwriteGate(path, struct.text,
                { overwriteAlreadyConfirmed: true });
        }
        return dialog.chooseSaveName(initial).then(function (chosen) {
            if (chosen === null || chosen === undefined) {
                return { ok: false, cancelled: true,
                         error: "Save cancelled." };
            }
            var dir = _dirname(path);
            var finalPath = dir ? (dir + "/" + chosen) : chosen;
            // If the user kept the original name, this IS the
            // source file the workspace was loaded from — silently
            // overwrite without a second confirm (clicking Save on
            // the workspace's source is unambiguous).  If the name
            // changed, route through the overwrite-gate so a clash
            // with an unrelated existing file gets the warning.
            var nameUnchanged = chosen === initial;
            return _writeWithOverwriteGate(finalPath, struct.text, {
                overwriteAlreadyConfirmed: nameUnchanged,
            });
        });
    }

    var _wired = false;
    var _unsubCanvas = null;
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
            button.disabled = !_workspace || !path || _workspace.isEmpty();
            if (readout) {
                if (path) {
                    readout.textContent = (dirty ? "Unsaved — " : "")
                                        + "Target: " + _basename(path);
                } else if (_workspace && !_workspace.isEmpty()) {
                    readout.textContent = "No source file (Save as… later).";
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
            button.disabled = true;
            setStatus("Saving…");
            save().then(function (r) {
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
