/* Trajectory inspector -- registry-side adapter that wires the
 * shared partial-inspector factory to ``lib/trajectory/core.js``.
 *
 * Match rule: ``*.molwatch.log`` (the canonical molbuilder format)
 * + ``.out`` (SIESTA's redirected stdout, content-sniffed by
 * SiestaParser).  ``.pyscf.log`` is NOT claimed here — it's plain
 * PySCF wrapper stdout, not a trajectory format; falls through to
 * the source inspector (text viewer) until a dedicated
 * ``pyscf-log`` inspector lands on the roadmap.  Plain ``.log``
 * stays unclaimed too — too generic.
 *
 * Mount flow + error-card rendering live in
 * lib/inspectors/_partial_inspector_factory.js (DRY'd 2026-06-09,
 * task #308; pre-fix this wrapper carried a ~150-LoC scaffold
 * identical to spectra.js's).
 */
(function (root) {
    "use strict";

    const factory = (root.molbuilder
                     && root.molbuilder.inspectors
                     && root.molbuilder.inspectors._partialInspectorFactory);
    if (!factory) {
        // Script-order regression: the factory must self-register
        // before this wrapper runs (see results.html).  Bail
        // loudly via console so a future contributor doesn't
        // spend an hour wondering why the trajectory inspector
        // never registers.
        if (root.console) root.console.error(
            "[lib/inspectors/trajectory.js] "
          + "_partial_inspector_factory.js did not load first; "
          + "trajectory inspector NOT registered."
        );
        return;
    }

    const inspector = factory.makePartialInspector({
        name:          "trajectory",
        displayName:   "Trajectory + SCF history",
        coreApiKey:    "trajectoryInspector",
        coreScriptDir: "trajectory",
        partialUrl:    "/partials/trajectory-inspector",
        match: (file) => {
            const lower = file.toLowerCase();
            return lower.endsWith(".molwatch.log")
                || lower.endsWith(".out");
        },
        // Two different engine outputs land here; the picker groups
        // them under distinct headers so the user scans visually.
        // ``.out`` → SIESTA (siesta wrapper redirects stdout into
        // ``<base>-runN.out``); ``.molwatch.log`` → the unified
        // molwatch format emitted by the PySCF geom-opt wrapper.
        resultCategory: (file) => {
            const lower = file.toLowerCase();
            if (lower.endsWith(".molwatch.log")) return "PySCF optimization";
            if (lower.endsWith(".out"))           return "SIESTA optimization";
            return "Trajectory";  // fallback (shouldn't fire given match())
        },
    });

    root.molbuilder.inspectors.trajectoryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
