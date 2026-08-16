/* codemirror-load.js — the ONE loader for the vendored CodeMirror 5 bundle.
 *
 * Extracted from ``lib/projects/preview.js`` on 2026-08-16, when the Job Prep
 * tab needed the same editor.  Two copies of a lazy-loader is two places for
 * the asset list to drift, and the vendor-integrity test pins that list — so
 * it has one home and both callers import it.
 *
 * WHAT IS VENDORED, and what follows from it (`static/vendor/README.md`):
 * CodeMirror **5.65.16**, MIT, served locally — the project ships browser
 * assets itself for offline use and a strict CSP, so there is no CDN path and
 * adding one is a reviewed decision, not a convenience.
 *
 * The bundle carries the core, the dialog/search/jump addons, and the
 * **markdown** mode.  It carries NO javascript/json mode, so a caller wanting
 * JSON highlighting would need a new vendored file + an inventory row + the
 * notice test — deliberately not done here.  ``mode: null`` is a real mode in
 * CodeMirror (plain text), and it still gives line numbers, editing, undo, and
 * the search addons.  A caller that asks for a mode the bundle lacks gets plain
 * text silently from CodeMirror itself, which is why callers pass ``null``
 * explicitly rather than a hopeful ``"application/json"``.
 *
 * Load order matters and is not an accident:
 *   1. CSS (parallel with the core — no execution dependency)
 *   2. the core, which must finish before any addon parses: addons register
 *      themselves on ``CodeMirror.commands`` / ``CodeMirror.defineOption`` at
 *      parse time, against a global that has to exist already
 *   3. dialog (search + jumpToLine both call its ``cm.openDialog``)
 *   4. searchcursor (search needs it)
 *   5. search + jumpToLine (both depend on the above; parallel with each other)
 */

export const CM_VENDOR_BASE = "/static/vendor/codemirror/";

/** Every file the bundle needs, in the order above.  The vendor-integrity
 *  test reads THIS list, so an asset added here is automatically required to
 *  exist on disk (`tests/test_codemirror_vendor_bundle.py`). */
export const CM_ASSETS = [
    "codemirror.min.css",
    "dialog.min.css",
    "codemirror.min.js",
    "dialog.min.js",
    "searchcursor.min.js",
    "search.min.js",
    "jump-to-line.min.js",
];

let _cmLoaderPromise = null;

export function injectStylesheet(href) {
    return new Promise((resolve, reject) => {
        const link = document.createElement("link");
        link.rel  = "stylesheet";
        link.href = href;
        link.onload  = () => resolve();
        link.onerror = () => reject(
            new Error("Could not load stylesheet: " + href));
        document.head.appendChild(link);
    });
}

export function injectScript(src) {
    return new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = src;
        s.async = false;  // preserve evaluation order vs. previous calls
        s.onload  = () => resolve();
        s.onerror = () => reject(
            new Error("Could not load script: " + src));
        document.head.appendChild(s);
    });
}

/**
 * Lazy-load the bundle.  The promise is cached, so several callers on one page
 * — the projects-sidebar preview modal and the Job Prep editor, say — share a
 * single fetch instead of racing to inject the same <script> twice.
 *
 * Returns ``window.CodeMirror``.
 */
export async function loadCodeMirror() {
    if (window.CodeMirror) return window.CodeMirror;
    if (_cmLoaderPromise)  return _cmLoaderPromise;
    _cmLoaderPromise = (async () => {
        await Promise.all([
            injectStylesheet(CM_VENDOR_BASE + "codemirror.min.css"),
            injectStylesheet(CM_VENDOR_BASE + "dialog.min.css"),
            injectScript(CM_VENDOR_BASE + "codemirror.min.js"),
        ]);
        await injectScript(CM_VENDOR_BASE + "dialog.min.js");
        await injectScript(CM_VENDOR_BASE + "searchcursor.min.js");
        await Promise.all([
            injectScript(CM_VENDOR_BASE + "search.min.js"),
            injectScript(CM_VENDOR_BASE + "jump-to-line.min.js"),
        ]);
        if (!window.CodeMirror) {
            throw new Error(
                "CodeMirror bundle loaded but the global is missing — "
                + "vendored bundle may be corrupt"
            );
        }
        return window.CodeMirror;
    })();
    return _cmLoaderPromise;
}
