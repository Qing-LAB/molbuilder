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

        var input = dialog.querySelector('[data-role="name-input"]');
        var saveBtn = dialog.querySelector('[data-action="save"]');
        function refreshSaveDisabled() {
            saveBtn.disabled = !(input.value || "").trim();
        }
        if (input) {
            input.addEventListener("input", refreshSaveDisabled);
            input.addEventListener("keydown", function (e) {
                if (e.key === "Enter") {
                    e.preventDefault();
                    if (!saveBtn.disabled) _settle((input.value || "").trim());
                }
            });
        }
        refreshSaveDisabled();

        dialog.querySelector('[data-action="cancel"]')
            .addEventListener("click", function () { _settle(null); });
        saveBtn.addEventListener("click", function () {
            _settle((input.value || "").trim());
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
        label.appendChild(input);
        dialog.appendChild(label);

        var hint = doc.createElement("p");
        hint.className = "molbuilder-save-name-hint";
        hint.textContent =
            "Saved to the same project directory.  Existing files "
            + "will prompt for overwrite confirmation.";
        dialog.appendChild(hint);

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
