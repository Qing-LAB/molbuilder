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
 *     ``can_parse`` content-sniff in ``parse/engines/pyscf.py``.
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
 * lib/inspectors/_partial_inspector_factory.js.
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
     * subsumes (`results.md` § 2.3).  These are `runfiles` roles: one
     * declared in `WRITTEN` (`_initial.xyz`) and two an engine writes
     * (`_optimized.xyz`, `_geom_optim.xyz`, `runfiles._ENGINE_ROLES`).
     *
     * KNOWN GAP: the same vocabulary
     * is spelled in `runfiles._ENGINE_ROLES` and in `pyscf/input.py`'s
     * `ROLE_*` constants as well, so the browser's copy is the fourth of
     * four rather than a lone offender.  `/api/results/dir` is the place a
     * single answer would arrive -- the route already sends each file's
     * role, and one more field would let this list be deleted.
     */
    /* THE SERVER'S ANSWER names what a file is: its ROLE in the catalogue
     * (`runs._file_row`, `runfiles.WRITTEN`), carried per file by
     * `/api/results/dir`.  A file with no answer is claimed by nothing --
     * never by a suffix read off its name here.  These three roles are the
     * run outputs this inspector presents. */
    const TRAJECTORY_ROLES = [".molwatch.log", ".out", "_geom_optim.xyz"];
    /* A run's structure parts -- the input echoed back, the geometry
     * reached, geomeTRIC's stream -- shown under the run's own log. */
    const PART_ROLES = ["_initial.xyz", "_optimized.xyz", "_geom_optim.xyz"];

    const inspector = factory.makePartialInspector({
        name:          "trajectory",
        displayName:   "Trajectory + SCF history",
        coreApiKey:    "trajectoryInspector",
        coreScriptDir: "trajectory",
        partialUrl:    "/partials/trajectory-inspector",
        match: (file, meta) =>
            !!(meta && TRAJECTORY_ROLES.indexOf(meta.role) >= 0),
        // The picker groups a run's files under the engine's own heading.
        resultCategory: (file, meta) => {
            /* THE ENGINE COMES FROM THE SERVER: `/api/results/dir` answers
             * `engine` for the directory (`plans/plan.md` N9), so the
             * heading names it instead of guessing from the suffix, a guess
             * wrong by construction for a third engine.
             */
            const engine = (meta && meta.engine
                            && meta.engine !== "unknown") ? meta.engine : "";
            // The folder answer names no engine: the heading leaves it
            // unclaimed.
            if (!engine) return "Optimization";
            const label = engine === "siesta" ? "SIESTA"
                        : engine === "pyscf"  ? "PySCF"
                        : engine.toUpperCase();
            return label + " optimization";
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
         * SAME RUN IS ONE EQUALITY, because the server reads each name back
         * with the label the deck states (`/api/results/dir` -> `label`,
         * `stage`, `role`; `runfiles.parse`).  The input echo and the
         * optimizer's geometry stem on the bare job label and this rung's stream on the same label with a
         * stage, so `label` is the thing they share and the test is `===`.
         *
         * The stem is never cut here: the label/stage boundary CANNOT be
         * found from the string alone (`runfiles.parse`), because a role may
         * contain `_` and so may a label.
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
            // `_initial` / `_optimized` are named on the label alone, with
            // no stage, while the geomeTRIC stream is this rung's and
            // carries its stage.  A LADDER IS N RESULTS (`stages.md` § 1.1a), so 03_tight's
            // master must leave 01_coarse's stream to 01_coarse's master.
            if (!(mMeta && mMeta.label && oMeta && oMeta.label)) return false;
            if (mMeta.role !== ".molwatch.log") return false;
            if (mMeta.label !== oMeta.label) return false;
            if (oMeta.stage && oMeta.stage !== mMeta.stage) return false;
            return PART_ROLES.indexOf(oMeta.role) >= 0;
        },
    });

    root.molbuilder.inspectors.trajectoryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
