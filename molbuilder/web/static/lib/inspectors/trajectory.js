/* Trajectory inspector -- registry-side adapter that wires the
 * shared partial-inspector factory to ``lib/trajectory/core.js``.
 *
 * Match rule:
 *   * ``*.molwatch.log`` — the canonical molbuilder format
 *     (single self-contained file with frames + SCF + forces)
 *   * ``*.out`` — SIESTA's redirected stdout, content-sniffed by
 *     SiestaParser
 *   * ``*_optim.xyz`` — PySCF / geomeTRIC's multi-frame
 *     trajectory XYZ (the geom-opt wrapper writes
 *     ``<job>_geom_optim.xyz``; older PySCF runs may use
 *     ``<job>_optim.xyz``).  PySCFParser handles both shapes via
 *     ``can_parse`` content-sniff in ``parsers/pyscf.py``.
 *
 * Registration order in results.html puts this BEFORE the
 * structure inspector (which matches all ``.xyz`` / ``.pdb``), so
 * the ``*_optim.xyz`` claim here wins over structure's generic
 * ``.xyz`` match.  Plain user-named single-frame ``.xyz`` files
 * still flow to the structure inspector — only the conventional
 * geomeTRIC / PySCF trajectory naming is intercepted.
 *
 * Intentional non-matches:
 *   * ``.pyscf.log`` — plain PySCF wrapper stdout, not a
 *     trajectory format; falls through to the source inspector
 *     (text viewer) until a dedicated ``pyscf-log`` inspector
 *     lands on the roadmap.
 *   * Plain ``.log`` — too generic.
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
                || lower.endsWith(".out")
                || lower.endsWith("_optim.xyz")
                || lower.endsWith("_geom_optim.xyz");
        },
        // Three different engine outputs land here; the picker groups
        // them under distinct headers so the user scans visually.
        //   ``.out``                 → SIESTA wrapper redirected stdout
        //   ``.molwatch.log``        → unified molwatch format (any engine)
        //   ``*_optim.xyz`` (incl. ``_geom_optim.xyz``)
        //                            → PySCF / geomeTRIC multi-frame XYZ
        resultCategory: (file) => {
            const lower = file.toLowerCase();
            if (lower.endsWith(".molwatch.log"))   return "PySCF optimization";
            if (lower.endsWith(".out"))             return "SIESTA optimization";
            if (lower.endsWith("_optim.xyz")
                || lower.endsWith("_geom_optim.xyz"))
                return "PySCF optimization";
            return "Trajectory";  // fallback (shouldn't fire given match())
        },

        /* A RUN IS ONE RESULT (results.md § 2.3).
         *
         * A `.molwatch.log` already carries the whole relaxation -- the
         * generator's own manifest calls it the "unified per-step log ...
         * single-file input for molwatch" (pyscf/input.py).  The files beside
         * it are that run's working parts, not peer results:
         *
         *   <base>_initial.xyz          the INPUT, echoed back
         *   <base>_optimized.xyz        final coords; also the seed the NEXT
         *                               run warm-starts from
         *   <base>_geom_<stage>_optim.xyz   geomeTRIC's per-stage stream,
         *                               the same steps the master log reports
         *
         * Listing them as peers turned one PySCF relaxation into five menu
         * entries (2026-08-04).
         *
         * NAMING: a staged run's master is `<base>-stage<N>.molwatch.log`
         * while its satellites stay plain `<base>_*` -- the `-stage<N>`
         * overlay suffix (job-contracts.md § 2.3) has to come off before the
         * prefixes can be compared.  Only `.molwatch.log` absorbs; a SIESTA
         * `.out` names its own stage files the same way and is left alone
         * until that is verified against a real staged SIESTA run.
         */
        absorbs: (master, other) => {
            const lower = master.toLowerCase();
            if (!lower.endsWith(".molwatch.log")) return false;
            const cut = (p) => {
                const ix = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
                return { dir: ix < 0 ? "" : p.slice(0, ix),
                         name: ix < 0 ? p : p.slice(ix + 1) };
            };
            const m = cut(master);
            const o = cut(other);
            if (m.dir !== o.dir) return false;          // same folder only
            // `<base>-stage3.molwatch.log` -> `<base>`
            const base = m.name
                .slice(0, -(".molwatch.log".length))
                .replace(/-stage\d+$/i, "");
            if (!base) return false;
            const n = o.name.toLowerCase();
            const b = base.toLowerCase();
            return n === b + "_initial.xyz"
                || n === b + "_optimized.xyz"
                || (n.startsWith(b + "_geom_") && n.endsWith("_optim.xyz"));
        },
    });

    root.molbuilder.inspectors.trajectoryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
