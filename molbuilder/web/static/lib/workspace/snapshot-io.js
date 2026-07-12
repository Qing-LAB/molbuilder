/* The ONE owner of the workspace sessionStorage snapshot IO.
 *
 * Both the dispatcher (WRITES the unified snapshot on every change) and the canvas-state
 * module (READS it to restore your unsaved work on reload) go through THIS module.
 *
 * Why it exists (data-model tightening): the canvas used to reach "up" to the dispatcher
 * (`ws.readPersistedSnapshot()`) for the reload restore, which (a) inverts the layering
 * and (b) only worked because the dispatcher happened to have mounted first -- if a canvas
 * read ran before the dispatcher mounted, the restore silently returned nothing and your
 * unsaved edits were dropped (a dishonest reload).  sessionStorage is ALWAYS available and
 * needs no module to be mounted, so reading it here makes the restore deterministic.
 *
 * Format: `{ v: 1, state: { structure, selection, view, ... } }` -- written by the
 * dispatcher's `_serialise()`, version-gated on read.  Loads BEFORE canvas + dispatcher.
 */
(function (root) {
    "use strict";
    root.molbuilder = root.molbuilder || {};

    // Active owner namespace (molview-module.md §18.4).  A single per-page value: each page
    // mounts exactly one active owner ("modify"; or one Results inspector at a time), so a
    // mutable holder is coherent.  ``null`` => the un-namespaced base key.  The dispatcher's
    // ``useNamespace(owner)`` sets this via ``setNamespace`` so the mirror key and the on-disk
    // ``workspace_id`` stay isolated per owner -- a Results session never overwrites Modify's,
    // and two inspectors on one page don't clobber each other's saved timeline.
    var _ns = null;

    function key() {
        var base = ((root.molbuilder.constants || {}).SS_WORKSPACE)
            || "molbuilder.workspace.v1";
        return _ns ? base + "::" + _ns : base;
    }

    root.molbuilder.workspaceSnapshot = {
        // Set the active owner namespace (or null to clear).  Called by the dispatcher's
        // useNamespace at mount, before any read/write for that owner.
        setNamespace: function (ns) { _ns = ns || null; },
        // Parsed, version-checked snapshot, or null (absent / corrupt / wrong version).
        read: function () {
            if (!root.sessionStorage) return null;
            var raw;
            try { raw = root.sessionStorage.getItem(key()); }
            catch (_) { return null; }
            if (!raw) return null;
            try {
                var parsed = JSON.parse(raw);
                return (parsed && parsed.v === 1) ? parsed : null;
            } catch (_) { return null; }
        },
        // Serialise the given envelope object to sessionStorage.  Returns false on a
        // quota/disabled-storage error (logged, non-fatal) -- same handling as before.
        write: function (obj) {
            if (!root.sessionStorage) return false;
            try {
                root.sessionStorage.setItem(key(), JSON.stringify(obj));
                return true;
            } catch (e) {
                if (root.console && root.console.warn) {
                    root.console.warn(
                        "workspace snapshot: could not persist:", e && e.message);
                }
                return false;
            }
        },
    };
})(typeof window !== "undefined" ? window : this);
