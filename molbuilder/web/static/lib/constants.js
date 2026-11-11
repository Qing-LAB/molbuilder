/* Shared string constants — single source of truth for the custom
 * event names that span multiple classic-script modules.
 *
 * Centralising them here means a typo is a compile-error
 * (ReferenceError on the missing constant), not a silent listener
 * mismatch.
 *
 * Loaded FIRST in every page template that consumes these
 * strings so ``window.molbuilder.constants`` is available when
 * later scripts run.  The sidebar's selection keys are not here: they
 * are `lib/projects/state.js`'s, read through its own door
 * (`projects.md` § 2).
 */
(function (root) {
    "use strict";

    root.molbuilder = root.molbuilder || {};
    // Single source of truth for the string constants.  Frozen
    // so a downstream module that accidentally writes to it
    // (e.g. ``molbuilder.constants.EVENT_FILE_SELECTED = "foo"``) throws
    // instead of silently changing every consumer.
    root.molbuilder.constants = Object.freeze({
        // ---- Custom event names ----------------------------- //
        EVENT_FILE_SELECTED:   "molbuilder:results:fileSelected",
        EVENT_INSPECTOR_READY: "molbuilder:inspector:ready",

        //: The panel's FOLDER changed, or the sidebar walked away from it.
        //: Display only -- it carries `{dir, diverged}` and moves nothing.
        //: It exists because divergence BEGINS after the last announcement:
        //: the panel says where it is when it mounts, and the sidebar can
        //: leave at any time after that, with no selection event to ride on.
        //: Without this the header would be honest only until you browsed.
        EVENT_SCOPE_CHANGED: "molbuilder:results:scopeChanged",
    });
})(typeof window !== "undefined" ? window : this);
