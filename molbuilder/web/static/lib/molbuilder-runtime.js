/* molbuilder runtime — module registry + ready Promises.
 *
 * Solves the "modules race each other to attach to window.molbuilder"
 * problem: classic <script> tags execute in document order, but
 * <script type="module"> defers until AFTER all classic scripts + the
 * initial parse complete.  Mixing them in one page means a classic
 * script that does ``(window.molbuilder || {}).projects`` at IIFE
 * time may see ``undefined`` because the projects module (loaded as
 * type=module) hasn't initialised yet.  Polling fixes the symptom;
 * this registry fixes the structure.
 *
 * Contract (design.md "Module init contract"):
 *
 *   PRODUCER (a module that exposes a global namespace):
 *     window.molbuilder.runtime.register("<name>", api);
 *   CONSUMER (anyone that depends on it):
 *     window.molbuilder.runtime.whenReady("<name>").then((api) => ...);
 *
 * No more `if (window.molbuilder && window.molbuilder.foo)` guards.
 * No more polling.  No more script-tag-order superstitions.
 *
 * The registry is ADDITIVE: producers keep their existing
 * ``window.molbuilder.<foo> = api`` assignments so unmigrated
 * consumers continue to work.  Migrate at your own pace.
 *
 * THIS FILE MUST BE LOADED FIRST -- before any other molbuilder
 * script -- so register() / whenReady() are defined when other
 * modules start their IIFEs.  Convention: include it right after
 * the 3Dmol vendor script in every template (build, modify,
 * spectra, results).
 *
 * Naming: flat, dotted, lowercased.  Today's modules:
 *   "viewer"                    -- lib/mol-viewer.js
 *   "style"                     -- lib/mol-style.js
 *   "fmt"                       -- lib/mol-format.js
 *   "formSchema"                -- lib/form-schema.js
 *   "projects"                  -- lib/projects-sidebar.js
 *   "selection.store"           -- lib/selection/store.js
 *   "selection.panel"           -- lib/selection-panel.js
 *   "selection.viewerAdapter"   -- lib/selection/viewer-adapter.js
 *   "modify.viewer"             -- modify/viewer.js (per-tab)
 *   "modify.loadStructureText"  -- modify/viewer.js (per-tab; accepts XYZ + PDB)
 *   "inspectors"                -- lib/inspectors/registry.js
 *
 * Adding a new module-with-a-global: pick a name in this scheme,
 * call register() at the END of your IIFE, add the name to the
 * list above.
 */
(function (root) {
    "use strict";

    // Idempotent: if molbuilder-runtime.js gets loaded twice (e.g.
    // a template double-includes it), keep the first runtime
    // instance + ignore the second.  Re-creating the registry
    // would drop already-registered modules.
    root.molbuilder = root.molbuilder || {};
    if (root.molbuilder.runtime) return;

    const _registered = new Map();   // name -> api
    const _waiters    = new Map();   // name -> Array<(api) => void>

    function register(name, api) {
        if (typeof name !== "string" || !name) {
            throw new TypeError(
                "runtime.register(name, api): name must be a non-empty string"
            );
        }
        if (api === undefined || api === null) {
            throw new TypeError(
                "runtime.register(name, api): api must not be null/undefined"
            );
        }
        if (_registered.has(name)) {
            // Double-register is a programmer bug (two modules
            // claiming the same name) -- warn loudly but accept
            // the new value so the page doesn't break.
            if (root.console) root.console.warn(
                "[molbuilder.runtime] module '" + name
                + "' re-registered; replacing previous registration."
            );
        }
        _registered.set(name, api);
        // Drain any pending whenReady() Promises.  Resolvers run
        // synchronously (on the microtask queue when then'ed),
        // so subsequent register() calls cannot race against
        // pending resolutions.
        const ws = _waiters.get(name);
        if (ws) {
            ws.forEach((resolve) => {
                try { resolve(api); }
                catch (e) {
                    if (root.console) root.console.error(
                        "[molbuilder.runtime] whenReady resolver for '"
                        + name + "' threw:", e
                    );
                }
            });
            _waiters.delete(name);
        }
    }

    function whenReady(name) {
        if (typeof name !== "string" || !name) {
            return Promise.reject(new TypeError(
                "runtime.whenReady(name): name must be a non-empty string"
            ));
        }
        if (_registered.has(name)) {
            // Already registered -- resolve on the next microtask
            // so consumers always see consistent .then() timing
            // regardless of order.
            return Promise.resolve(_registered.get(name));
        }
        return new Promise((resolve) => {
            let queue = _waiters.get(name);
            if (!queue) {
                queue = [];
                _waiters.set(name, queue);
            }
            queue.push(resolve);
        });
    }

    function get(name) {
        // Synchronous peek; returns undefined if not yet registered.
        // Prefer whenReady() over get() in new code -- get() is for
        // tools that need a SYNCHRONOUS snapshot (debugging, tests).
        return _registered.get(name);
    }

    function listRegistered() {
        // Sorted snapshot of currently-registered module names.
        // Useful from devtools:
        //   molbuilder.runtime.listRegistered()
        return Array.from(_registered.keys()).sort();
    }

    function listPending() {
        // Names with whenReady() callers but no registration yet.
        // Diagnoses "consumer is waiting forever" bugs.
        return Array.from(_waiters.keys()).sort();
    }

    root.molbuilder.runtime = {
        register:        register,
        whenReady:       whenReady,
        get:             get,
        listRegistered:  listRegistered,
        listPending:     listPending,
    };
})(typeof window !== "undefined" ? window : globalThis);
