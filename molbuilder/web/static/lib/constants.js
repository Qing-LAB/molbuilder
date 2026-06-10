/* Shared string constants — single source of truth for the
 * sessionStorage keys + custom event names that span multiple
 * classic-script modules.
 *
 * Pre-task-#309 (2026-06-09) the same literals were duplicated
 * across 9+ files; a typo in any one of them silently broke a
 * round-trip without surfacing in tests until someone exercised
 * the exact path.  Centralising them here means a typo is a
 * compile-error (ReferenceError on the missing constant), not a
 * silent listener mismatch.
 *
 * Loaded FIRST in every page template that consumes these
 * strings so ``window.molbuilder.constants`` is available when
 * later scripts run.  ES-module consumers
 * (lib/projects/state.js, lib/projects/list.js) keep their own
 * named exports for direct ``import { SS_FILE } from
 * "./state.js"``; this module is the classic-script equivalent.
 * Both surfaces hold the same literal values — the convention
 * here is "if you change a string, change it in BOTH places";
 * a tiny e2e test
 * (tests/test_constants_module_consistency.py) pins the two
 * surfaces in lock-step.
 */
(function (root) {
    "use strict";

    root.molbuilder = root.molbuilder || {};
    // Single source of truth for the string constants.  Frozen
    // so a downstream module that accidentally writes to it
    // (e.g. ``molbuilder.constants.SS_FILE = "foo"``) throws
    // instead of silently changing every consumer.
    root.molbuilder.constants = Object.freeze({
        // ---- sessionStorage keys ---------------------------- //
        SS_FILE:      "molbuilder.current_file",
        SS_DIR:       "molbuilder.current_dir",
        SS_WORKSPACE: "molbuilder.workspace.v1",

        // ---- Custom event names ----------------------------- //
        EVENT_FILE_SELECTED:   "molbuilder:results:fileSelected",
        EVENT_INSPECTOR_READY: "molbuilder:inspector:ready",
    });
})(typeof window !== "undefined" ? window : this);
