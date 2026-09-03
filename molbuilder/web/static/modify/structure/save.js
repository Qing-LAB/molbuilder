/* Save panel — the Molbuilder tab's "Save to project" button.
 *
 * Contract: docs/web/tabs.md § 6 (the out-gate).
 *
 * THIS FILE DOES NOT SAVE.  It is a panel over one call:
 *
 *     projects.molviewFiles.save("project", stem, viewer.exportFile())
 *
 * -- the same door MolView's own Export -> Data row goes through
 * (`projects.md` § 5).  The door asks WHERE (a folder, then a name, via
 * `chooseSavePath`), posts `/api/structure/save`, runs the overwrite
 * confirmation, and refreshes the sidebar.  What is left here is the part
 * that is genuinely this page's:
 *
 *   * the readout beside the button -- "Target: <name>.xyz", and
 *     "Unsaved" once you edit again;
 *   * the button's enabled state;
 *   * recording where the bytes went (`structurePage.markSavedTo`) --
 *     the PAGE's note, because the viewer tracks contents and not files
 *     (`molview.md` § 6.7).
 *
 * WHAT WENT, AND WHY (2026-09-02).  This file was written before the front
 * end had modules.  It could not `import`, so it took its collaborators off
 * globals and grew its own copy of the save flow: its own name dialog, its
 * own overwrite confirm, its own sidebar refresh, and a destination forced
 * to the sidebar's current directory.  `molviewFiles` landed in 2026-08 on
 * the modern path with the destination as a QUESTION, and this tab has
 * mounted it into the viewer ever since -- so two implementations reached
 * the same route with the same body by two roads that were free to drift,
 * and had.  ~300 lines here and the whole of `save-dialog.js` were that
 * second road.
 *
 * The user-visible change: Save now asks which folder, instead of writing
 * into whichever one the sidebar happened to be showing -- and the button is
 * no longer greyed out for want of a sidebar selection, which was a refusal
 * the screen could not explain.
 *
 * Surface (``window.molbuilder.structureSave``):
 *
 *   save()        -> Promise<{ok, path?, error?, cancelled?}>
 *   targetPath()  -> string | null   (where the LAST save went)
 *   useViewer(v)  -- the page hands over the viewer it mounted
 *   configure(o)  -- test seam: {projects, structurePage, workspace}
 *   wirePanel(o?) -- idempotent DOM wiring
 */

(function (root) {
    "use strict";

    var _projects      = null;
    var _structurePage = null;
    var _handle        = null;      // the viewer this page mounted
    var _model         = null;      // its data model

    /* The page hands over the viewer it mounted.
     *
     * `viewer.ok` IS CHECKED: a mount that failed still returns a handle, and
     * its `data` is not a model.  Accepting it would put the panel in a state
     * where `getStructure` is not a function and every refresh throws.
     *
     * AND IT (RE)ATTACHES THE SUBSCRIPTION.  `wirePanel` runs from
     * DOMContentLoaded; the viewer arrives later, after an `await
     * MV.mount(...)` in `selection-bootstrap.js`.  Whichever lands first is a
     * race the page does not control -- and when the panel wired first, there
     * was no model to subscribe to, `_wired` stopped a second attempt, and the
     * readout never moved again: edit the structure and it went on saying
     * "Target: x.xyz" with no "Unsaved".  So the arrival of a viewer is
     * treated as the moment to (re)subscribe, not the wiring. */
    function useViewer(viewer) {
        _handle = (viewer && viewer.ok) ? viewer : null;
        _model  = _handle ? _handle.data : null;
        _attach();
    }

    function configure(opts) {
        opts = opts || {};
        if (opts.projects)      _projects      = opts.projects;
        if (opts.structurePage) _structurePage = opts.structurePage;
        // `workspace` / `canvas` both name a stand-in DATA model; the panel
        // reads a viewer, so wrap either into one.
        if (opts.workspace)     useViewer({ ok: true, data: opts.workspace });
        else if (opts.canvas)   useViewer({ ok: true, data: opts.canvas });
    }

    /* The projects sidebar mounts as a deferred ESM module, so it is not on
     * `window.molbuilder` when this IIFE runs.  Re-resolve whatever is still
     * null at call time; a test's explicit fakes are left alone. */
    function _lazyResolve() {
        if (typeof root === "undefined" || !root.molbuilder) return;
        if (!_projects && root.molbuilder.projects) {
            _projects = root.molbuilder.projects;
        }
        if (!_structurePage && root.molbuilder.structurePage) {
            _structurePage = root.molbuilder.structurePage;
        }
    }

    function _basename(p) {
        if (!p) return "";
        var i = String(p).lastIndexOf("/");
        return i >= 0 ? String(p).slice(i + 1) : String(p);
    }

    /* Where this page last saved, or null.
     *
     * THE PAGE'S OWN NOTE.  The viewer tracks contents, not files
     * (`molview.md` § 6.7) -- it never knew where anything came from or went
     * to.  `structurePage` keeps it because `structurePage` is what performed
     * the save.  It drives the readout; it is NOT a destination, and has not
     * been since the destination became a question. */
    function targetPath() {
        _lazyResolve();
        var snap = (_structurePage && _structurePage.getCanvasSnapshot)
            ? _structurePage.getCanvasSnapshot() : null;
        return (snap && snap.lastSaveTo) || null;
    }

    /* A name to suggest in the dialog: what this page last saved as, else the
     * file it loaded from, else nothing.  Only a SUGGESTION -- the door's
     * dialog is where the name is decided, and an empty box is a fine
     * starting point. */
    function _stem() {
        var last = targetPath();
        var from = (_structurePage
                    && typeof _structurePage.getLoadedFrom === "function")
            ? _structurePage.getLoadedFrom() : null;
        return _basename(last || from || "").replace(/\.xyz$/i, "");
    }

    function save() {
        _lazyResolve();
        if (!_model || typeof _model.exportFile !== "function") {
            return Promise.resolve(
                { ok: false, error: "Save: no viewer on this page yet." });
        }
        // Nothing loaded reads as nothing (`molview.md` § 9.3).
        if (_model.getStructure() === null) {
            return Promise.resolve(
                { ok: false, error: "No structure to save." });
        }
        var doors = _projects && _projects.molviewFiles;
        if (!doors || typeof doors.save !== "function") {
            return Promise.resolve(
                { ok: false, error: "Save: the project save door is not "
                                  + "available on this page." });
        }
        // THE MODEL SERIALISES ITSELF (`tabs.md` § 6 step 2): what is saved is
        // what is on screen, and this file scans nothing.  `exportFile()`
        // yields {name, structure, frames?} -- the door reads the last two.
        var payload = _model.exportFile();
        if (!payload) {
            return Promise.resolve(
                { ok: false,
                  error: "Save: the structure could not be written out." });
        }
        return Promise.resolve(doors.save("project", _stem(), payload))
            .then(function (res) {
                /* TELL THE PAGE WHERE THE BYTES WENT, from the SERVER's
                 * answer -- `auto_rename` may have written `<stem>-2.xyz`,
                 * and confirming the name we merely asked for would name a
                 * file that does not exist. */
                if (res && res.ok && _structurePage
                        && typeof _structurePage.markSavedTo === "function") {
                    _structurePage.markSavedTo(res.path);
                }
                return res;
            });
    }

    /* ---------- the panel ---------- */

    var _wired = false;
    var _unsub = null;
    var _inFlight = false;
    var _refresh = null;      // set by wirePanel; the panel's own repaint

    /* Paint once and follow the model.  Called from both ends of the race in
     * `useViewer`'s note -- whichever of (panel wired, viewer arrived) happens
     * second is the one that makes this do anything. */
    function _attach() {
        if (typeof _refresh !== "function") return;
        _refresh();
        if (typeof _unsub === "function") { _unsub(); _unsub = null; }
        if (_model && typeof _model.subscribe === "function") {
            // THE MODEL'S subscribe (`molview.md` § 8.4): a subscriber is
            // handed one state on subscribing, so the first paint needs no
            // separate fetch, and another after every change.
            _unsub = _model.subscribe(_refresh);
        }
    }

    function wirePanel(opts) {
        opts = opts || {};
        var doc = opts.doc || root.document;
        if (!doc || _wired) return;
        _wired = true;

        var button  = doc.getElementById("save-to-source-btn");
        var readout = doc.getElementById("save-readout");
        var status  = doc.getElementById("save-status");
        if (!button) return;

        function refreshState() {
            _lazyResolve();
            var hasContent = !!(_model
                                && _model.getStructure
                                && _model.getStructure() !== null);
            /* CONTENT IS THE ONLY PRECONDITION.  It also required a sidebar
             * directory, because that used to BE the destination -- so Save
             * sat greyed out for a reason nothing on screen stated.  The
             * destination is part of the question the dialog asks now. */
            button.disabled = _inFlight || !hasContent;

            if (!readout) return;
            var path  = targetPath();
            var dirty = !!(_model && _model.uncommitted);
            if (path) {
                readout.textContent = (dirty ? "Unsaved — " : "")
                                    + "Target: " + _basename(path);
            } else if (hasContent) {
                readout.textContent = "Not saved yet";
            } else {
                readout.textContent = "";
            }
        }

        /* The app's one severity surface (`lib/status.js`); its `error` IS
         * red, unlike the `.muted` + undefined-modifier spelling seven panels
         * shared before it. */
        function setStatus(msg, kind) {
            root.molbuilder.status.set(
                status, msg, kind === "error" ? "error" : null);
        }

        _refresh = refreshState;
        _attach();

        button.addEventListener("click", function () {
            _inFlight = true;
            button.disabled = true;
            setStatus("Saving…");
            save().then(function (r) {
                _inFlight = false;
                if (r && r.ok) {
                    setStatus("Saved " + _basename(r.path) + ".");
                } else if (r && r.cancelled) {
                    setStatus("");
                } else {
                    setStatus((r && r.error) || "Save failed.", "error");
                }
                refreshState();
            }, function (err) {
                /* `save` returns an envelope and does not reject -- but a
                 * collaborator missing at click time can throw inside the
                 * chain, and a hung "Saving…" tells the user nothing. */
                _inFlight = false;
                setStatus("Save failed: "
                          + ((err && err.message) || String(err)), "error");
                refreshState();
            });
        });
    }

    var api = {
        configure:  configure,
        useViewer:  useViewer,
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
            root.molbuilder.runtime.register("structure.save", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
