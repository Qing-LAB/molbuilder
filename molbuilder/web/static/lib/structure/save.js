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
    var _canvas        = null;

    function configure(opts) {
        opts = opts || {};
        if (opts.projects)      _projects      = opts.projects;
        if (opts.structurePage) _structurePage = opts.structurePage;
        if (opts.canvas)        _canvas        = opts.canvas;
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
        if (!_canvas        && root.molbuilder.structureCanvas)
            _canvas        = root.molbuilder.structureCanvas;
    }

    /**
     * Resolve the natural save target.  Order:
     *   1. ``last_save_to`` if previously saved this session.
     *   2. ``source.file`` if the canvas was loaded from a project file.
     *   3. null → Save disabled, user needs Save-as (future).
     */
    function targetPath() {
        _lazyResolve();
        if (!_canvas) return null;
        var lastSaved = _canvas.getLastSavedTo();
        if (lastSaved) return lastSaved;
        var src = _canvas.getSource();
        if (src && src.kind === "file" && src.file) return src.file;
        return null;
    }

    function save() {
        _lazyResolve();
        if (!_canvas) {
            return Promise.reject(new Error(
                "save: canvas not configured"));
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
        if (_canvas.isEmpty()) {
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
        var struct = _canvas.getStructure();
        // overwrite:true — the user explicitly asked to Save to
        // this path; the default 409-on-existing would be wrong UX
        // (Save to source IS overwriting the source).
        return _projects.writeFile(
            path, struct.text, { overwrite: true }
        ).then(
            function (r) {
                if (!r || !r.ok) {
                    return {
                        ok:    false,
                        error: (r && r.error) || "Save failed.",
                    };
                }
                // Tell the orchestrator the save landed — clears
                // dirty + records last_save_to.
                _structurePage.markSavedTo(path);
                return { ok: true, path: path };
            },
            function (err) {
                return {
                    ok:    false,
                    error: "Save failed: "
                         + (err && err.message ? err.message
                                                : String(err)),
                };
            }
        );
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
            var dirty = _canvas && _canvas.isDirty();
            button.disabled = !_canvas || !path || _canvas.isEmpty();
            if (readout) {
                if (path) {
                    readout.textContent = (dirty ? "Unsaved — " : "")
                                        + "Target: " + _basename(path);
                } else if (_canvas && !_canvas.isEmpty()) {
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

        // Initial paint + live updates from the canvas.
        refreshState();
        if (_canvas && typeof _canvas.onChange === "function") {
            // Replace any prior subscription on re-wire.
            if (typeof _unsubCanvas === "function") _unsubCanvas();
            _unsubCanvas = _canvas.onChange(function () {
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
            canvas:        root.molbuilder.structureCanvas,
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
