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
        /* THE ROLE, when the server gave one.  These four suffixes are
         * `runfiles.WRITTEN` rows spelled a second time here, which is the
         * duplication R-RO1 forbids; `meta.role` is the catalogue's own
         * answer, carried per file by `/api/results/dir`.  The suffix test
         * survives only as the fallback for a caller outside the Results
         * tab, which has no directory answer to pass. */
        match: (file, meta) => {
            if (meta && meta.role) {
                return meta.role === ".molwatch.log"
                    || meta.role === ".out"
                    || meta.role === "_geom_optim.xyz";
            }
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
        resultCategory: (file, meta) => {
            /* THE ENGINE COMES FROM THE SERVER NOW, and that lifts the
             * constraint this function was built around.  It used to read:
             * "The browser cannot know the engine: it is a fact about the
             * run DIRECTORY, and the picker has only a filename."  True
             * until 2026-09-18 -- `/api/results/dir` answers `engine` for
             * the directory (`plans/plan.md` N9), so the heading can name
             * it instead of guessing from the suffix.
             *
             * What the guess cost: `.out` was hardcoded "SIESTA" and
             * `_geom_optim.xyz` "PySCF".  Both are right for the two engines
             * that exist and wrong by construction for a third -- a VASP
             * directory's trajectory was labelled "PySCF optimization"
             * (demonstrated 2026-09-18).  And `.molwatch.log`, which every
             * engine writes, had to stay engine-LESS for exactly this
             * reason; now it need not.
             */
            const engine = (meta && meta.engine
                            && meta.engine !== "unknown") ? meta.engine : "";
            const role = (meta && meta.role) || "";
            const lower = file.toLowerCase();
            const what =
                (role === ".molwatch.log" || lower.endsWith(".molwatch.log"))
                    ? "optimization"
              : (role === ".out" || lower.endsWith(".out"))
                    ? "optimization"
              : (role === "_geom_optim.xyz" || lower.endsWith("_optim.xyz"))
                    ? "optimization"
              : "trajectory";
            if (!engine) {
                // No directory answer (a caller outside the Results tab).
                // Name what the file IS and leave the engine unclaimed --
                // which is what the 2026-09-04 fix established for
                // `.molwatch.log` and is now the rule for all of them.
                return what === "trajectory" ? "Trajectory" : "Optimization";
            }
            const label = engine === "siesta" ? "SIESTA"
                        : engine === "pyscf"  ? "PySCF"
                        : engine.toUpperCase();
            return label + " " + what;
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
         *   <base>_geom_optim.xyz       geomeTRIC's per-stage stream,
         *                               the same steps the master log reports
         *
         * Listing them as peers turned one PySCF relaxation into five menu
         * entries (2026-08-04).
         *
         * NAMING: a staged run's master is `<job>_<token>.molwatch.log`
         * where the token is the stage's artifact token -- digit-first,
         * `01_coarse` (job-contracts.md § 6.3) -- while `_initial` /
         * `_optimized` satellites stem on the bare `<job>` (they CARRY
         * between rungs) while the geomeTRIC stream is this rung's, so it
         * stems on the full `<job>_<token>` -- the token sits where every
         * token sits, right after the label, and `_geom_optim.xyz` is the
         * whole role (job-contracts.md § 2.2a).  So satellites are matched
         * against BOTH stems: the master's full stem and the
         * token-stripped job.  This comment said the token sat INSIDE the
         * role (`<job>_geom_<token>_optim.xyz`) until 2026-09-07, and the
         * test below was a prefix/suffix pair loose enough to pass either
         * way -- so the wrong belief cost nothing and was invisible.  All
         * three are exact names now.  (The `-stage<N>` strip that stood here
         * was the pre-rename spelling; after the token rename it
         * matched nothing, and every staged relaxation went back to
         * being five menu entries -- 2026-08-19.)  Only `.molwatch.log`
         * absorbs; a SIESTA `.out` names its own stage files the same
         * way and is left alone until that is verified against a real
         * staged SIESTA run.
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
            const stem = m.name.slice(0, -(".molwatch.log".length));
            if (!stem) return false;
            // `<job>_<token>` -> `<job>`; an unstaged stem passes through.
            const job = stem.replace(/_\d+_[A-Za-z0-9_]+$/, "");
            const n = o.name.toLowerCase();
            const stems = job && job !== stem
                ? [stem.toLowerCase(), job.toLowerCase()]
                : [stem.toLowerCase()];
            return stems.some((b) =>
                n === b + "_initial.xyz"
                || n === b + "_optimized.xyz"
                || n === b + "_geom_optim.xyz");
        },
    });

    root.molbuilder.inspectors.trajectoryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
