/* /spectra page bootstrap.
 *
 * The spectra-inspector logic lives in
 * ``static/lib/spectra/core.js`` (shared with /results via the
 * Inspector Registry; see ``lib/inspectors/spectra.js``).  This
 * file is /spectra-specific: it picks WHEN to mount the inspector
 * on the /spectra page (on DOMContentLoaded, against ``document``
 * as the root so the generate-side form + (legacy) inspect-side
 * controls resolve against the full page).
 *
 * It is HTML-Script-included AFTER ``lib/spectra/core.js``, so
 * ``window.molbuilder.spectraInspector.mount`` is guaranteed to
 * exist when this file's bootstrap fires.
 *
 * The Spec / migration log: docs/protocols/results-tab.md § 4
 * (sub-stage 2.2 split this file from the inspector core).
 */
(function () {
    "use strict";

    function bootstrapSpectraPage() {
        // Defensive: if core.js failed to load (e.g. a CDN-block in
        // some weird CSP edge case), fail loudly in the console
        // rather than silently rendering an empty <main>.
        const api = (window.molbuilder || {}).spectraInspector;
        if (!api || typeof api.mount !== "function") {
            console.error(
                "[spectra] spectraInspector.mount not found.  "
                + "Check that lib/spectra/core.js is loaded BEFORE "
                + "spectra/viewer.js (the script tag order in "
                + "spectra.html is load-bearing)."
            );
            return;
        }
        // Mount with ``document`` as the root so $() lookups find
        // /spectra's generate-side form ids (which live OUTSIDE any
        // partial; full-page mount).
        api.mount(document);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrapSpectraPage);
    } else {
        bootstrapSpectraPage();
    }
})();
