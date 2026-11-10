/* Trajectory inspector -- registry-side adapter that wires the
 * shared partial-inspector factory to ``lib/trajectory/core.js``.
 *
 * Match rule -- a file the server's answer names a run's (its role) and
 * reads with one of three parsers (its parser), never a suffix:
 *   * ``<label>_<stage>.molwatch.log`` — the run's progress log,
 *     molbuilder's own format (frames + SCF + forces), parser ``molwatch``;
 *   * ``<label>_<stage>.out`` — SIESTA's stdout, parser ``siesta``;
 *   * ``<label>[_<stage>]_geom_optim.xyz`` — geomeTRIC's trajectory, parser
 *     ``pyscf``: the one XYZ that reader is for, by its name (`parse.md`
 *     § 4).  Every other XYZ -- a saved structure or a calculation's
 *     ``<label>.source.xyz``, pairs read by ``structure-pair``; SIESTA's own
 *     ``<label>.xyz``, read by ``xyz`` -- is left to the structure
 *     inspector.
 *
 * Registration order in results.html puts this BEFORE the
 * structure inspector (which matches every ``.xyz`` / ``.pdb``), so
 * geomeTRIC's trajectory opens here, not as frame 0 of a structure.
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

    /* THE SERVER'S ANSWER names what a file is: the PARSER that claims it
     * (`runs.file_row` asks the registry, `/api/results/dir` carries it per
     * file), as every inspector claims since 2.13.3 (`presenters.md` § 2).
     * A file with no answer is claimed by nothing -- never by a suffix or a
     * role spelled here (`parse.md` R-RO1).  The three parsers whose files
     * this inspector presents are the run's progress log, SIESTA's output
     * and geomeTRIC's trajectory, by the registry's own names -- each for a
     * file the catalogue NAMES (its role): the role says the file is a
     * run's of ours, the parser that it holds a run's steps.  geomeTRIC's
     * trajectory is the one XYZ its parser is for (`parse.md` § 4); every
     * other XYZ -- a pair `structure-pair`'s, any other `xyz`'s -- stays the
     * structure preview's
     * (`presenters.md` § 2). */
    const TRAJECTORY_PARSERS = ["molwatch", "siesta", "pyscf"];
    /* A run's structure parts -- the input echoed back and the geometry
     * reached (the PySCF geometry parser's files), and geomeTRIC's stream
     * (the trajectory parser's) -- shown under the run's own log. */
    const PART_PARSERS = ["pyscf-geom", "pyscf"];

    const inspector = factory.makePartialInspector({
        name:          "trajectory",
        displayName:   "Trajectory + SCF history",
        coreApiKey:    "trajectoryInspector",
        partialUrl:    "/partials/trajectory-inspector",
        match: (file, meta) =>
            !!(meta && meta.role
               && TRAJECTORY_PARSERS.indexOf(meta.parser) >= 0),
        // The picker groups a run's files under the engine's own heading.
        resultCategory: (file, meta) => {
            /* THE WORDS COME FROM THE SERVER: `/api/results/dir` answers
             * the engine's words and the calculation's noun from the two
             * tables (`presenters.md` § 2) -- *SIESTA optimization*,
             * *PySCF vibration* -- never guessed from the suffix; an answer
             * without both claims neither. */
            const words = meta && meta.engine_words;
            const noun = meta && meta.noun;
            return words && noun ? words + " " + noun : "Trajectory";
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
         * With no directory answer there is nothing to read the label from,
         * and nothing is absorbed.
         */
        absorbs: (master, other, mMeta, oMeta) => {
            // Same folder only -- THE ONE PATH DOOR (`lib/path-utils.js`),
            // read when called.
            const path = root.molbuilder.path;
            if (path.dirname(master) !== path.dirname(other)) return false;

            // THE SERVER'S READING, when there is one.  Same label, and
            // then the STAGE decides which rung a part belongs to -- which
            // is the sentence above, said in the grammar's own words:
            // `_initial` / `_optimized` are named on the label alone, with
            // no stage, while the geomeTRIC stream is this rung's and
            // carries its stage.  A LADDER IS N RESULTS (`stages.md` § 1.1a), so 03_tight's
            // master must leave 01_coarse's stream to 01_coarse's master.
            if (!(mMeta && mMeta.label && oMeta && oMeta.label)) return false;
            if (mMeta.parser !== "molwatch") return false;
            if (mMeta.label !== oMeta.label) return false;
            if (oMeta.stage && oMeta.stage !== mMeta.stage) return false;
            return PART_PARSERS.indexOf(oMeta.parser) >= 0;
        },
    });

    root.molbuilder.inspectors.trajectoryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
