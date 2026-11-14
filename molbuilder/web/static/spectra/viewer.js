/* The Spectrum calculation page's bootstrap (`web/spectra.md`).
 *
 * Three responsibilities, all this page's own (a finished result is shown on
 * the Results tab, through lib/spectra/core.js and the spectra inspector):
 *
 *   1. Mount the spectra core against ``document``, so the vibration form,
 *      its checks and the hand-over to Task setup wire up.
 *
 *   2. Attach the inspect card (`lib/structure-card.js`, the one the
 *      Structure-optimization tab attaches): the read-only viewer, the Load
 *      button, the structure kept as the tab's own draft and put back when
 *      the tab is shown again.  The hand-over reads the structure OFF THE
 *      VIEWER when it is sent -- no push, no in-memory holder.
 *
 *   3. Keep the forms in this browser tab (`formSchema.keepInTab`), as the
 *      Structure-optimization tab keeps its own (the user, 2026-11-14).
 */
import { attachStructureCard } from "/static/lib/structure-card.js";

(function () {
    "use strict";

    /* THE INSPECTOR THIS PAGE MOUNTED, kept so the page can hand it the viewer
     * it also mounted.  One page, two things it owns, and it introduces them --
     * rather than either of them looking the other up. */
    let _inspector = null;
    // THE CHEMISTRY CARD (`lib/chemistry.js`): the charge and spin each
    // engine form's vibration will carry, resolved by the one
    // electronic-state class for exactly what the form says, with each
    // value's reason (`science/chemistry-correctness.md` § 2a) -- about the
    // structure the tab would hand over, read by the inspector's one read
    // of the viewer.  It asks on every structure load and every edit to a
    // charge or spin field, and fills nothing in.
    const _chemistry = window.molbuilder.chemistry.attach({
        kind: "vibration",
        forms: () => (_inspector ? _inspector.stateForms() : {}),
        structure: () => (_inspector ? _inspector.structureForRequest()
                                     : null),
    });

    /* KEPT IN THIS BROWSER TAB: every part of the vibration's form, put back
     * once the inspector has drawn them. */
    const _forms = window.molbuilder.formSchema.keepInTab(
        "spectrum_calculation_forms",
        () => (_inspector ? _inspector.keptForms() : []));

    function bootstrapSpectraPage() {
        // Mount with ``document`` as the root so $() lookups find /spectra's
        // generate-side form ids (which live OUTSIDE any partial).
        _inspector = window.molbuilder.spectraInspector.mount(document, {
            // The forms exist now: put back what this tab kept, then the
            // chemistry card answers for them.
            onFormsReady: () => { _forms.restore(); _chemistry.refresh(); },
        });
        /* THE INSPECT CARD, under this tab's own tag (workspace.md § 4).  The
         * page mounted the viewer through it, so the page hands it on: the
         * Generate panel reads this same one.  A structure that lands --
         * loaded, or put back -- lets the structure pick the default engine
         * (a periodic structure is SIESTA's), reruns the live checks, and
         * asks the chemistry card again. */
        attachStructureCard({
            host: "spectra-molview-host",
            tag:  "spectra",
            onMounted: (handle) => _inspector.useViewer(handle),
            onLoaded: () => {
                _inspector.structureLoaded();
                _chemistry.refresh();
            },
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrapSpectraPage);
    } else {
        bootstrapSpectraPage();
    }
})();
