/* Save-name + overwrite-confirm dialogs for the Save panel.
 *
 * 2026-06-09: the Save flow now goes through two modal gates:
 *
 *   1. ``chooseSaveName(initialName)`` — asks the user to confirm
 *      or edit the destination filename.  Returns the final name
 *      (str) on accept, or null on cancel.
 *
 *   2. ``confirmOverwrite(filename)`` — shown when the chosen name
 *      already exists on disk and the user must confirm before
 *      clobbering.  Returns true on confirm, false on cancel.
 *
 * The Save panel composes these:
 *
 *     const name = await chooseSaveName(currentBaseName);
 *     if (!name) return;
 *     const r = await writeFile(target, text, {overwrite: false});
 *     if (!r.ok && r.status === 409) {
 *         const proceed = await confirmOverwrite(name);
 *         if (!proceed) return;
 *         await writeFile(target, text, {overwrite: true});
 *     }
 *
 * Concurrency: each dialog is single-instance.  Calling a function
 * while a prior call's modal is still open returns the SAME pending
 * promise so two near-simultaneous saves don't stack two modals.
 *
 * Surface mounted on ``window.molbuilder.structureSaveDialog``.
 */
(function (root) {
    "use strict";

    // ─── chooseSaveName ──────────────────────────────────────────── //

    var _activeName = null;

    /**
     * Show the save-name confirm dialog with the filename pre-filled.
     * Returns a Promise resolving to the final filename (str) on
     * accept, or null on cancel / ESC.
     *
     * @param {string} initialName - the basename to pre-fill (e.g.
     *   "water.xyz").  May be empty when the workspace is generator-
     *   sourced.
     * @param {object} [opts]
     * @param {Document} [opts.doc]
     */
    function chooseSaveName(initialName, opts) {
        opts = opts || {};
        var doc = opts.doc || root.document;
        if (!doc) {
            return Promise.reject(new Error(
                "save-dialog: no document"));
        }
        if (_activeName) return _activeName.promise;

        var dialog = _buildNameDialog(doc, initialName || "");
        doc.body.appendChild(dialog);

        var resolve;
        var promise = new Promise(function (res) { resolve = res; });
        _activeName = { dialog: dialog, resolve: resolve, promise: promise };

        function _settle(value) {
            if (!_activeName || _activeName.dialog !== dialog) return;
            _activeName = null;
            try { dialog.close(); } catch (_) {}
            try {
                if (dialog.parentNode) {
                    dialog.parentNode.removeChild(dialog);
                }
            } catch (_) {}
            resolve(value);
        }

        var input  = dialog.querySelector('[data-role="name-input"]');
        var errEl  = dialog.querySelector('[data-role="name-error"]');
        var saveBtn = dialog.querySelector('[data-action="save"]');

        // Validation: empty / whitespace-only → disabled.  Path-
        // separators (``/`` or ``\``) are rejected because the
        // Save panel resolves the chosen name within the current
        // project directory; a user typing ``foo/bar.xyz`` would
        // either escape the project root (caught server-side) or
        // create files in a sibling subdir without realising it.
        // Forces the user to pick a different directory via the
        // sidebar rather than smuggle a path through the input.
        var previewEl = dialog.querySelector('[data-role="name-preview"]');
        function _validate(raw) {
            var s = (raw || "").trim();
            // Forgiving: the save owns the extensions, so strip a ".xyz" or
            // ".molstruct.json" the user typed out of habit -- the returned value
            // is always the clean BASE name.
            s = s.replace(/\.molstruct\.json$/i, "").replace(/\.xyz$/i, "");
            if (!s) return { ok: false, reason: "" };
            if (s.indexOf("/") !== -1 || s.indexOf("\\") !== -1) {
                return {
                    ok: false,
                    reason: "Name can't contain '/' or '\\\\'."
                            + "  Pick a different directory via the"
                            + " sidebar instead.",
                };
            }
            if (s === "." || s === "..") {
                return { ok: false, reason: "Reserved name." };
            }
            return { ok: true, value: s };
        }
        function refreshSaveDisabled() {
            var v = _validate(input.value);
            saveBtn.disabled = !v.ok;
            if (errEl) {
                errEl.textContent = v.ok ? "" : v.reason;
                errEl.hidden = v.ok;
            }
            if (previewEl) {
                if (v.ok) {
                    previewEl.textContent =
                        "Writes  " + v.value + ".xyz  and  "
                        + v.value + ".molstruct.json";
                    previewEl.hidden = false;
                } else {
                    previewEl.hidden = true;
                }
            }
        }
        if (input) {
            input.addEventListener("input", refreshSaveDisabled);
            input.addEventListener("keydown", function (e) {
                if (e.key === "Enter") {
                    e.preventDefault();
                    var v = _validate(input.value);
                    if (v.ok) _settle(v.value);
                }
            });
        }
        refreshSaveDisabled();

        dialog.querySelector('[data-action="cancel"]')
            .addEventListener("click", function () { _settle(null); });
        saveBtn.addEventListener("click", function () {
            var v = _validate(input.value);
            if (v.ok) _settle(v.value);
        });
        dialog.addEventListener("cancel", function () { _settle(null); });
        dialog.addEventListener("close",  function () { _settle(null); });

        try {
            if (typeof dialog.showModal === "function") dialog.showModal();
            else if (typeof dialog.show === "function") dialog.show();
        } catch (_) {}

        // Focus the input + select-all so the user can either accept
        // the default by pressing Enter or start typing to replace it.
        try {
            if (input && typeof input.focus === "function") {
                input.focus();
                if (typeof input.select === "function") input.select();
            }
        } catch (_) {}

        return promise;
    }

    function _buildNameDialog(doc, initialName) {
        var dialog = doc.createElement("dialog");
        dialog.className = "molbuilder-save-name-modal";

        var title = doc.createElement("h2");
        title.textContent = "Save structure";
        dialog.appendChild(title);

        var label = doc.createElement("label");
        label.className = "molbuilder-save-name-label";

        var labelText = doc.createElement("span");
        labelText.textContent = "Filename";
        label.appendChild(labelText);

        var input = doc.createElement("input");
        input.type = "text";
        input.value = initialName || "";
        input.setAttribute("data-role", "name-input");
        input.setAttribute("autocomplete", "off");
        input.setAttribute("spellcheck", "false");
        input.setAttribute("placeholder", "e.g. my_molecule");
        label.appendChild(input);
        dialog.appendChild(label);

        var hint = doc.createElement("p");
        hint.className = "molbuilder-save-name-hint";
        hint.textContent =
            "Enter a name only — no file extension.  Saving writes TWO files "
            + "into the current project directory: the coordinates and a sidecar "
            + "with your labels / regions / cell.";
        dialog.appendChild(hint);

        // Live preview of the exact pair the save will write, so the user sees
        // that ONE name produces both files and never needs to type ".xyz".
        var preview = doc.createElement("p");
        preview.className = "molbuilder-save-name-preview";
        preview.setAttribute("data-role", "name-preview");
        preview.hidden = true;
        dialog.appendChild(preview);

        // Inline error slot for validation messages — hidden when
        // the input is valid, shown when the user types something
        // that would be rejected (e.g. a path separator).
        var err = doc.createElement("p");
        err.className = "molbuilder-save-name-error";
        err.setAttribute("data-role", "name-error");
        err.hidden = true;
        dialog.appendChild(err);

        var actions = doc.createElement("div");
        actions.className = "molbuilder-save-name-actions";

        var cancel = doc.createElement("button");
        cancel.type = "button";
        cancel.setAttribute("data-action", "cancel");
        cancel.textContent = "Cancel";
        actions.appendChild(cancel);

        var save = doc.createElement("button");
        save.type = "button";
        save.setAttribute("data-action", "save");
        save.textContent = "Save";
        actions.appendChild(save);

        dialog.appendChild(actions);
        return dialog;
    }

    // ─── confirmOverwrite ────────────────────────────────────────── //

    var _activeOverwrite = null;

    /**
     * Ask the user to confirm overwriting an existing file.  Returns
     * a Promise resolving to true on confirm, false on cancel / ESC.
     *
     * @param {string} filename - basename of the conflicting file
     * @param {object} [opts]
     */
    function confirmOverwrite(filename, opts) {
        opts = opts || {};
        var doc = opts.doc || root.document;
        if (!doc) {
            return Promise.reject(new Error("save-dialog: no document"));
        }
        if (_activeOverwrite) return _activeOverwrite.promise;

        var dialog = _buildOverwriteDialog(doc, filename || "");
        doc.body.appendChild(dialog);

        var resolve;
        var promise = new Promise(function (res) { resolve = res; });
        _activeOverwrite = { dialog: dialog, resolve: resolve, promise: promise };

        function _settle(value) {
            if (!_activeOverwrite || _activeOverwrite.dialog !== dialog) return;
            _activeOverwrite = null;
            try { dialog.close(); } catch (_) {}
            try {
                if (dialog.parentNode) {
                    dialog.parentNode.removeChild(dialog);
                }
            } catch (_) {}
            resolve(value);
        }

        dialog.querySelector('[data-action="cancel"]')
            .addEventListener("click", function () { _settle(false); });
        dialog.querySelector('[data-action="overwrite"]')
            .addEventListener("click", function () { _settle(true); });
        dialog.addEventListener("cancel", function () { _settle(false); });
        dialog.addEventListener("close",  function () { _settle(false); });

        try {
            if (typeof dialog.showModal === "function") dialog.showModal();
            else if (typeof dialog.show === "function") dialog.show();
        } catch (_) {}
        try {
            var cancelBtn = dialog.querySelector('[data-action="cancel"]');
            if (cancelBtn && typeof cancelBtn.focus === "function") {
                cancelBtn.focus();
            }
        } catch (_) {}
        return promise;
    }

    function _buildOverwriteDialog(doc, filename) {
        var dialog = doc.createElement("dialog");
        dialog.className = "molbuilder-save-overwrite-modal";

        var title = doc.createElement("h2");
        title.textContent = "File exists";
        dialog.appendChild(title);

        var body = doc.createElement("p");
        body.textContent =
            "A file named "
            + (filename ? "“" + filename + "” " : "")
            + "already exists in the project directory.  Overwrite it?";
        dialog.appendChild(body);

        var actions = doc.createElement("div");
        actions.className = "molbuilder-save-overwrite-actions";

        var cancel = doc.createElement("button");
        cancel.type = "button";
        cancel.setAttribute("data-action", "cancel");
        cancel.textContent = "Cancel";
        actions.appendChild(cancel);

        var overwrite = doc.createElement("button");
        overwrite.type = "button";
        overwrite.setAttribute("data-action", "overwrite");
        overwrite.textContent = "Overwrite";
        actions.appendChild(overwrite);

        dialog.appendChild(actions);
        return dialog;
    }

    // ─── Test seams ──────────────────────────────────────────────── //

    function isNameOpen() { return _activeName !== null; }
    function isOverwriteOpen() { return _activeOverwrite !== null; }
    function _reset() {
        [_activeName, _activeOverwrite].forEach(function (a) {
            if (!a) return;
            try { a.dialog.close(); } catch (_) {}
            try {
                if (a.dialog.parentNode) {
                    a.dialog.parentNode.removeChild(a.dialog);
                }
            } catch (_) {}
            a.resolve(null);
        });
        _activeName = null;
        _activeOverwrite = null;
    }

    var api = {
        chooseSaveName:    chooseSaveName,
        confirmOverwrite:  confirmOverwrite,
        isNameOpen:        isNameOpen,
        isOverwriteOpen:   isOverwriteOpen,
        _reset:            _reset,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.structureSaveDialog = api;
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.saveDialog", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
