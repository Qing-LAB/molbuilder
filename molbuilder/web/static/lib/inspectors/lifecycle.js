/* lifecycle.js — the two helpers every inspector core needs.
 *
 * `lib/spectra/core.js` and `lib/trajectory/core.js` are two inspectors with
 * one lifecycle: mount, listen, dispose, so a leak fix has one place.
 *
 * Exports (on window.molbuilder.inspectorLifecycle):
 *   listeners()          -> { on, defer, disposeAll }
 *   alias(state, k, b)   -> a legacy name that reads through to a bucket
 *   announceReady(name, detail) -> the load has ended: drawn, or refused
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
             * It exists so a core has exactly ONE registry: a second array
             * beside this scope would be drained by `dispose()` while every
             * listener stayed attached.
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

    /**
     * THE LOAD HAS ENDED -- the first render is on screen, or the load was
     * refused and its reason is on the inspector's status line.  One signal
     * for both, so the tab's loading cover and the picker's "Parsing…" line
     * go when the answer is on screen, whichever it is.  Deferred two frames
     * so the browser paints first -- AND a short timer beside them,
     * whichever comes first, ONCE: frames do not run in a background tab.
     */
    function announceReady(inspector, detail) {
        try {
            var fired = false;
            var dispatch = function () {
                if (fired) return;
                fired = true;
                root.document.dispatchEvent(new root.CustomEvent(
                    root.molbuilder.constants.EVENT_INSPECTOR_READY,
                    { detail: Object.assign({ inspector: inspector },
                                            detail || {}) }));
            };
            if (typeof root.requestAnimationFrame === "function") {
                root.requestAnimationFrame(function () {
                    root.requestAnimationFrame(dispatch);
                });
                root.setTimeout(dispatch, 250);
            } else {
                dispatch();
            }
        } catch (_) {
            // CustomEvent / rAF unavailable in some ancient runtimes; the
            // picker's timeout fallback covers it.
        }
    }

    var api = { listeners: listeners, alias: alias,
                announceReady: announceReady };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectorLifecycle = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
