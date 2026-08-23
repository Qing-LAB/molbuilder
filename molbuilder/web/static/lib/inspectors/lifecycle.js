/* lifecycle.js — the two helpers every inspector core needs.
 *
 * `lib/spectra/core.js` and `lib/trajectory/core.js` are two inspectors with
 * one lifecycle: mount, listen, dispose.  Both spelled these out
 * byte-identically, which is two places for a leak fix to miss.
 *
 * Exports (on window.molbuilder.inspectorLifecycle):
 *   listeners()          -> { on, defer, disposeAll }
 *   alias(state, k, b)   -> a legacy name that reads through to a bucket
 */
(function (root) {
    "use strict";

    /**
     * A listener scope that remembers how to undo itself.
     *
     * An inspector is mounted and disposed repeatedly as the user picks
     * files, so a listener that outlives its mount is a leak that fires
     * against a dead DOM.  Registering through here means the cleanup is
     * written at the same moment as the registration, which is the only way
     * the two stay in step.
     */
    function listeners() {
        var undo = [];
        return {
            on: function (target, event, handler, opts) {
                if (!target) return;
                target.addEventListener(event, handler, opts);
                undo.push(function () {
                    target.removeEventListener(event, handler, opts);
                });
            },
            /**
             * Register a teardown that is not a listener -- a
             * ResizeObserver to disconnect, an observer to stop.
             *
             * It exists so a core has exactly ONE registry.  When an
             * inspector kept a second array beside this scope, `dispose()`
             * drained that one and left every listener attached: a
             * mount/dispose cycle removed nothing (caught 2026-08-23 by
             * `test_inspector_registry_e2e.py`, which counts add/remove
             * pairs across a real mount).  Two registries is the same
             * defect as two readers -- one of them is the one that is used.
             */
            defer: function (undoFn) {
                if (typeof undoFn === "function") undo.push(undoFn);
            },
            disposeAll: function () {
                while (undo.length) {
                    try { undo.pop()(); } catch (_) { /* already gone */ }
                }
            },
        };
    }

    /**
     * Expose `state[key]` as a read/write view of `state[bucket][key]`.
     *
     * The inspectors keep their real state in buckets and carry older flat
     * names for the surfaces that still use them; an alias means the two can
     * never disagree, because there is only one value.
     */
    function alias(state, key, bucket) {
        Object.defineProperty(state, key, {
            get: function () { return state[bucket][key]; },
            set: function (v) { state[bucket][key] = v; },
            enumerable: true,
            configurable: true,
        });
    }

    var api = { listeners: listeners, alias: alias };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectorLifecycle = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
