/* Structure-tab "discard unsaved modifications" warning modal.
 *
 * The Structure tab fires this when one of the following actions
 * would overwrite the canvas AND the canvas is dirty:
 *
 *   - Clicking Load on the load-from-project panel.
 *   - Clicking Generate on any generator panel (SMILES, 3DNA, ...).
 *   - The browser fires beforeunload (tab close / refresh / navigate
 *     away).  The beforeunload handler is registered by the tab UI
 *     module — NOT this primitive — because beforeunload requires
 *     a sync return value, not an async confirmation modal.
 *
 * Surface:
 *
 *   const proceed = await warningModal.confirmDiscardUnsaved();
 *   if (proceed) { canvas.clear(); loadNew(); }
 *
 * Returns ``Promise<boolean>`` — true when the user clicked
 * "Discard and continue", false on Cancel (and on ESC, which the
 * native <dialog> element treats as a cancel).
 *
 * Modal content per docs/web/tabs.md:
 *
 *   Title:   "Unsaved modifications"
 *   Body:    "You have unsaved changes to the current canvas.
 *             Continuing will discard them."
 *   Buttons: Cancel (default focus) + Discard and continue.
 *
 * Concurrency: a single dialog instance is reused.  Calling
 * ``confirmDiscardUnsaved`` while a prior call's modal is still
 * open returns the SAME pending Promise — so two near-simultaneous
 * user actions don't stack two modals on top of each other.
 *
 * History: introduced when the Structure tab merged Build's
 * structure-generation paths so generated structures could be
 * edited in place rather than save → reload.  See
 * ``docs/web/tabs.md`` § 5.4 for the full design.
 */
(function (root) {
    "use strict";

    var TITLE   = "Unsaved modifications";
    var BODY    = "You have unsaved changes to the current canvas. "
                + "Continuing will discard them.";
    var CANCEL  = "Cancel";
    var DISCARD = "Discard and continue";

    // Single-instance state.  Reset to null when the dialog closes.
    var _active = null;   // { dialog, resolve, promise }

    /**
     * Show the discard-unsaved confirmation modal.
     *
     * @param {object} [opts]
     * @param {Document} [opts.doc] - the document to render into
     *   (defaults to the page's ``document``).  Tests inject a
     *   stub here; production callers rely on the default.
     *
     * @returns {Promise<boolean>} resolves true if the user clicks
     *   "Discard and continue", false on Cancel / ESC.
     */
    function confirmDiscardUnsaved(opts) {
        opts = opts || {};
        var doc = opts.doc || root.document;
        if (!doc) {
            return Promise.reject(new Error(
                "warningModal.confirmDiscardUnsaved: no document — "
              + "module must run in a browser-like environment"));
        }

        // Re-entrancy: return the in-flight Promise instead of
        // stacking a second modal.
        if (_active) return _active.promise;

        var dialog = _buildDialog(doc);
        doc.body.appendChild(dialog);

        var resolve;
        var promise = new Promise(function (res) { resolve = res; });
        _active = { dialog: dialog, resolve: resolve, promise: promise };

        function _settle(value) {
            if (!_active || _active.dialog !== dialog) return;
            _active = null;
            try { dialog.close(); } catch (_) {}
            // Remove from DOM so multiple invocations don't leak
            // detached <dialog>s and so a future open lands on a
            // freshly-built element.
            try {
                if (dialog.parentNode) dialog.parentNode.removeChild(dialog);
            } catch (_) {}
            resolve(value);
        }

        // Wire button clicks.
        dialog.querySelector('[data-action="cancel"]')
            .addEventListener("click", function () { _settle(false); });
        dialog.querySelector('[data-action="discard"]')
            .addEventListener("click", function () { _settle(true); });

        // <dialog>'s native ESC + form-close semantics resolve to
        // ``cancel`` — pin that to false.  The "close" event fires
        // after close() too, so guard against a second _settle
        // (the if-check inside _settle handles it).
        dialog.addEventListener("cancel", function () { _settle(false); });
        dialog.addEventListener("close",  function () { _settle(false); });

        // Open the dialog modally.  showModal is the right call;
        // show() leaves the page below interactable.  Some test
        // stubs only implement show, so fall back if needed.
        try {
            if (typeof dialog.showModal === "function") {
                dialog.showModal();
            } else if (typeof dialog.show === "function") {
                dialog.show();
            }
        } catch (_) {}

        // Default focus on Cancel — the safe action.  Per § 5.4
        // a destructive primary requires explicit travel.
        try {
            var cancelBtn = dialog.querySelector('[data-action="cancel"]');
            if (cancelBtn && typeof cancelBtn.focus === "function") {
                cancelBtn.focus();
            }
        } catch (_) {}

        return promise;
    }

    /**
     * Build the dialog DOM.  Kept in its own function so the call
     * site stays readable and tests can mock createElement at a
     * single seam.
     */
    function _buildDialog(doc) {
        var dialog = doc.createElement("dialog");
        dialog.className = "molbuilder-warning-modal";
        dialog.setAttribute("aria-labelledby", "molbuilder-warning-title");
        dialog.setAttribute("aria-describedby", "molbuilder-warning-body");

        var title = doc.createElement("h2");
        title.id = "molbuilder-warning-title";
        title.textContent = TITLE;
        dialog.appendChild(title);

        var body = doc.createElement("p");
        body.id = "molbuilder-warning-body";
        body.textContent = BODY;
        dialog.appendChild(body);

        var actions = doc.createElement("div");
        actions.className = "molbuilder-warning-modal-actions";

        var cancel = doc.createElement("button");
        cancel.type = "button";
        cancel.setAttribute("data-action", "cancel");
        cancel.textContent = CANCEL;
        actions.appendChild(cancel);

        var discard = doc.createElement("button");
        discard.type = "button";
        discard.setAttribute("data-action", "discard");
        discard.textContent = DISCARD;
        actions.appendChild(discard);

        dialog.appendChild(actions);
        return dialog;
    }

    /**
     * Test seam — checks whether a modal is currently visible.
     * Production code should not need this; tests use it to assert
     * single-instance semantics.
     */
    function isOpen() { return _active !== null; }

    /**
     * Test seam — force-close any open modal (resolves the pending
     * Promise to ``false``, like Cancel).  Production code should
     * never call this.
     */
    function _reset() {
        if (!_active) return;
        var pending = _active;
        _active = null;
        try { pending.dialog.close(); } catch (_) {}
        try {
            if (pending.dialog.parentNode) {
                pending.dialog.parentNode.removeChild(pending.dialog);
            }
        } catch (_) {}
        pending.resolve(false);
    }

    var api = {
        confirmDiscardUnsaved: confirmDiscardUnsaved,
        isOpen:                isOpen,
        _reset:                _reset,
        // String constants exposed so tests can pin the documented
        // copy without re-typing the strings.
        TITLE:   TITLE,
        BODY:    BODY,
        CANCEL:  CANCEL,
        DISCARD: DISCARD,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.warningModal = api;
        if (root.molbuilder.runtime
            && typeof root.molbuilder.runtime.register === "function") {
            root.molbuilder.runtime.register(
                "structure.warningModal", api);
        }
    }
})(typeof window !== "undefined" ? window : globalThis);
