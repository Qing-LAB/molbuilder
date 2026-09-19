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

    /* THE RUN'S WORKING PARTS -- the roles a `.molwatch.log` master
     * subsumes (`results.md` § 2.3).  These are `runfiles` rows: two
     * declared in `WRITTEN` (`_initial.xyz`) and in `pyscf/warm-files.toml`
     * (`_optimized.xyz`, `_geom_optim.xyz`).
     *
     * KNOWN GAP, and it is not this change's to close: the same vocabulary
     * is spelled in `runfiles._CARRIED_ROLES` and in `pyscf/input.py`'s
     * `ROLE_*` constants as well, so the browser's copy is the fourth of
     * four rather than a lone offender.  `/api/results/dir` is the place a
     * single answer would arrive -- the route already sends each file's
     * role, and one more field would let this list be deleted.
     */
    const CARRIED_ROLES = [
        "_initial.xyz", "_optimized.xyz", "_geom_optim.xyz",
    ];

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
         *   <label>_initial.xyz          the INPUT, echoed back
         *   <label>_optimized.xyz        final coords; also the seed the NEXT
         *                                run warm-starts from
         *   <label>_<stage>_geom_optim.xyz
         *                                geomeTRIC's per-stage stream, the
         *                                same steps the master log reports
         *
         * Listing them as peers turned one PySCF relaxation into five menu
         * entries (2026-08-04).
         *
         * SAME RUN IS ONE EQUALITY, because the server reads each name back
         * with the label the deck states (`/api/results/dir` -> `label`,
         * `stage`, `role`; `runfiles.parse`).  The carried files stem on the
         * bare job label and this rung's stream on the same label with a
         * stage, so `label` is the thing they share and the test is `===`.
         *
         * WHAT THIS REPLACES, and why it had to go: this function used to
         * cut the master's stem itself --
         *
         *     const job = stem.replace(/_\d+_[A-Za-z0-9_]+$/, "");
         *
         * -- which is the label/stage boundary that `runfiles.parse`'s own
         * docstring says CANNOT be found from the string alone, because a
         * role may contain `_` and so may a label.  It matches leftmost, so
         * `au_2_bdt_02_fine` read as a label of `au`: the run's own
         * satellites were not absorbed (four menu entries for one run), and
         * a different job literally named `au` would have had its files
         * absorbed into this one and vanish from the picker.  Measured
         * 2026-09-19.
         *
         * The name test stays as the fallback for a caller with no directory
         * answer, which is the same shape `match` and `resultCategory` use.
         */
        absorbs: (master, other, mMeta, oMeta) => {
            const cut = (p) => {
                const ix = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
                return { dir: ix < 0 ? "" : p.slice(0, ix),
                         name: ix < 0 ? p : p.slice(ix + 1) };
            };
            const m = cut(master);
            const o = cut(other);
            if (m.dir !== o.dir) return false;          // same folder only

            // THE SERVER'S READING, when there is one.  Same label, and
            // then the STAGE decides which rung a part belongs to -- which
            // is the sentence above, said in the grammar's own words:
            // `_initial` / `_optimized` CARRY between rungs and so have no
            // stage, while the geomeTRIC stream is this rung's and carries
            // it.  A LADDER IS N RESULTS (`stages.md` § 1.1a), so 03_tight's
            // master must leave 01_coarse's stream to 01_coarse's master.
            if (mMeta && mMeta.label && oMeta && oMeta.label) {
                if (mMeta.role !== ".molwatch.log") return false;
                if (mMeta.label !== oMeta.label) return false;
                if (oMeta.stage && oMeta.stage !== mMeta.stage) return false;
                return CARRIED_ROLES.indexOf(oMeta.role) >= 0;
            }

            // NO DIRECTORY ANSWER: the names, and only the shapes that need
            // no boundary-finding -- an exact suffix on a shared prefix.
            if (!m.name.toLowerCase().endsWith(".molwatch.log")) return false;
            const stem = m.name.slice(0, -(".molwatch.log".length));
            if (!stem) return false;
            const n = o.name.toLowerCase();
            const b = stem.toLowerCase();
            return CARRIED_ROLES.some((r) => n === b + r);
        },
    });

    root.molbuilder.inspectors.trajectoryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
