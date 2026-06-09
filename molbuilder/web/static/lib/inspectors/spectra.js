/* Spectra-results inspector -- registry-side adapter that wires
 * the shared partial-inspector factory to ``lib/spectra/core.js``.
 *
 * Match rule: ``*.spectra.json`` (PySCF spectrum results emitted
 * by molbuilder's spectra wrapper).
 *
 * Auto-load: the spectra core's mount() honors ``opts.file`` by
 * pre-filling the watch-path field + firing loadByPath(), so
 * picking a .spectra.json in the sidebar (or via the /results
 * dropdown) lands directly on the rendered modes + chart with no
 * extra click.
 *
 * Mount flow + error-card rendering live in
 * lib/inspectors/_partial_inspector_factory.js (DRY'd 2026-06-09,
 * task #308; pre-fix this wrapper carried a ~150-LoC scaffold
 * identical to trajectory.js's).
 */
(function (root) {
    "use strict";

    const factory = (root.molbuilder
                     && root.molbuilder.inspectors
                     && root.molbuilder.inspectors._partialInspectorFactory);
    if (!factory) {
        if (root.console) root.console.error(
            "[lib/inspectors/spectra.js] "
          + "_partial_inspector_factory.js did not load first; "
          + "spectra inspector NOT registered."
        );
        return;
    }

    const inspector = factory.makePartialInspector({
        name:           "spectra",
        displayName:    "Spectra results",
        coreApiKey:     "spectraInspector",
        coreScriptDir:  "spectra",
        partialUrl:     "/partials/spectra-inspector",
        match:          (file) => file.toLowerCase().endsWith(".spectra.json"),
        resultCategory: (_file) => "PySCF spectrum",
    });

    root.molbuilder.inspectors.spectraInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
