/* Workspace sessionStorage snapshot IO — the sole owner of the session mirror read/write.
 *
 * MODULE: workspace persistence (lib/workspace/) — the low-level sessionStorage half.
 *   Mounts window.molbuilder.workspaceSnapshot = { setNamespace, read, write }.  This is the
 *   ONLY place that touches sessionStorage for the workspace session mirror; everything else
 *   goes through here so there is one key + one format.  Sibling: dispatcher.js (the transport
 *   + public window.molbuilder.workspace surface).  Contract: workspace-contract.md §4.4.
 *
 * USED BY:
 *   - lib/workspace/dispatcher.js — WRITES the unified snapshot on every persist() (via write),
 *     READS it for readPersistedSnapshot()/workspaceId() (via read), and sets the owner
 *     namespace via setNamespace (from useNamespace).
 *   - lib/molview/_canvas-state-impl.js — READS it to restore unsaved work on reload.
 *
 * Why the canvas reads HERE and not up through the dispatcher: the canvas used to call
 * `ws.readPersistedSnapshot()`, which (a) inverts the layering and (b) only worked if the
 * dispatcher had mounted first -- a canvas read before the dispatcher mounted silently returned
 * nothing and dropped your unsaved edits (a dishonest reload).  sessionStorage is ALWAYS
 * available and needs no module mounted, so reading it here makes the restore deterministic.
 *
 * Format: `{ v: 1, state: { structure, selection, view, ... } }` -- written by the data model's
 * serialise, version-gated on read.  Loads BEFORE canvas + dispatcher.
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
