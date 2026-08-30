/* Task hand-over sender -- THE one door from a parameter tab to Task setup.
 *
 * Extracted 2026-08-20 from structure-optimization/viewer.js's
 * `_sendToTaskSetup` when the spectra tab became the second sender
 * (spectra-migration-plan.md P2): the guards, the write order and the
 * notice handling are CONTRACT (`job-contracts.md` SS 2.1, SS 2.5;
 * `handover-procedure.md` SS 2), and two copies of a contract drift.
 *
 * Consumers:
 *   * structure-optimization/viewer.js  -- engine siesta|pyscf, optimization
 *   * lib/spectra/core.js               -- engine pyscf, calculation vibration
 *   * lib/transport/core.js             -- engine siesta, calculation
 *     transport (P7b): NO structure/params -- `junction` (the citation),
 *     `bias` (volts, 0.0 first) and `overrides` (transport-only knobs)
 *     ride instead, and the answer is ONE file (floor 2 = task.json
 *     alone; transport-design.md 4.1)
 *
 * Classic script on the `window.molbuilder` namespace (same loading
 * pattern as lib/form-schema.js) so both a classic-script consumer
 * (lib/spectra/core.js) and module consumers can reach it.
 *
 * send() takes everything it acts on as arguments -- it reads no DOM and
 * holds no state, so each tab keeps its own status line and structure
 * door:
 *
 *   send({
 *     projects,     // window.molbuilder.projects (the file layer)
 *     say,          // (kind, message) -> void; kinds: ok|muted|warn|error
 *     structure,    // the envelope from the tab's own MolView exportFile()
 *     engine,       // "siesta" | "pyscf"
 *     params,       // collectForm() output, keyed by catalogue names
 *     calculation,  // optional; omitted or "optimization" sends none
 *     junction,     // transport only: the citation string
 *     bias,         // transport only: [volts...], 0.0 first
 *     overrides,    // transport only: {TransportConfig field: value}
 *   })
 */
(function () {
    "use strict";

    async function send(opts) {
        const o = opts || {};
        const projects = o.projects;
        const say = (typeof o.say === "function") ? o.say : function () {};
        if (!projects || typeof projects.getCurrentDir !== "function") {
            say("error", "The projects sidebar is not available on this page.");
            return;
        }
        const dest = projects.getCurrentDir() || "";
        if (!dest) {
            say("warn", "Pick the folder to write into, in the "
                + "sidebar first — this button writes into the selected "
                + "folder and creates none of its own.");
            return;
        }
        const structure = o.structure;
        const isTransport = o.calculation === "transport";
        /* The transport composite carries NO structure -- its structure
         * IS the junction citation, copied in at prep
         * (transport-design.md 4.1).  Every other kind is OF a
         * structure and must bring one. */
        if (!structure && !isTransport) {
            say("warn", "Load a structure first — a description is "
                + "of something.");
            return;
        }
        if (isTransport && !o.junction) {
            say("warn", "Pick the relaxed junction's attempt first — "
                + "the composite derives everything from that citation.");
            return;
        }
        /* A CALCULATION LIVES UNDER A TOPIC (`job-contracts.md` § 2.5): the
         * tree is project / topic / calculation, and the three levels above
         * the calculation are organisational.  `safeSave` refuses only at the
         * projects ROOT, so without this a Send into a bare project or topic
         * folder would put a calculation where one may not live.
         *
         * Measured against the projects root rather than by counting slashes,
         * because the root is configurable — `relativeToProjects` is the
         * sidebar's own answer to "where am I". */
        const rel = (typeof projects.relativeToProjects === "function")
            ? String(projects.relativeToProjects(dest) || "") : "";
        const depth = rel.split("/").filter(Boolean).length;
        if (depth < 3) {
            say("warn",
                "That folder is too high in the tree — a calculation lives "
                + "under a topic (projects / project / topic / your folder). "
                + "Pick or make one further down in the sidebar.");
            return;
        }

        /* THE CALCULATION NEVER LIVES IN ITS CITATION (transport-design.md
         * 4.1b): describing into the cited directory would drop task.json --
         * and later every prep attempt -- inside the finished relaxation it
         * cites.  The sidebar selection lingers on the attempt the person
         * just browsed to cite, so this is the easy mistake to make.
         *
         * "Inside" means inside, not "equal to": a subfolder of the
         * citation buries the calculation just as thoroughly.  The check
         * used to compare only equality, which enforced something
         * narrower than the rule it quoted. */
        const relNorm = rel.replace(/\/+$/, "");
        const citeNorm = String(o.junction || "").replace(/\/+$/, "");
        const inside = citeNorm && relNorm
            && (relNorm === citeNorm || relNorm.startsWith(citeNorm + "/"));
        if (isTransport && inside) {
            say("warn",
                (relNorm === citeNorm
                    ? "That folder IS the cited junction"
                    : "That folder is INSIDE the cited junction")
                + " — the calculation never lives inside its citation.  "
                + "Make a folder for this transport job (e.g. under the "
                + "project's transport topic) and select it.");
            return;
        }

        /* ONE JOB PER FOLDER (`job-contracts.md` § 2.1 Rule 1).  Send writes
         * with `overwrite: true`, so without this it would silently replace
         * another calculation's template.  The check goes through the file
         * layer like every other read -- `missingOk` makes "no description
         * here" a 200 rather than a logged 404, which is the normal case.
         *
         * This guard EXISTED as a 409 in the hand-over endpoint and was lost
         * on 2026-08-16 when that endpoint stopped resolving the destination
         * to become render-only.  Restored on the side that chooses where to
         * write, which is where it belongs. */
        /* readFile never throws (projects/api.js: every failure is an
         * {ok:false, error} envelope), so with missingOk a genuine
         * absence is {ok:true, exists:false} and ok===false is ALWAYS a
         * real failure -- a 403/500/network drop.  The guard used to
         * fall through on exactly those and send anyway: a one-job-per-
         * folder check that fails OPEN overwrites the folder it could
         * not read.  It refuses now; the dead catch went with it. */
        const prior = await projects.readFile(dest + "/task.json",
                                              { missingOk: true });
        if (prior && prior.ok === false) {
            say("error", "Could not check the folder for an existing "
                + "description: " + (prior.error || "read failed")
                + " — nothing was written.");
            return;
        }
        if (prior && prior.exists !== false
            && typeof prior.text === "string" && prior.text.trim()) {
            say("error",
                "That folder already holds a task.json — it is a described "
                + "calculation. Sending here would overwrite it. Pick or "
                + "make another folder in the sidebar; one job per folder.");
            return;
        }

        say("muted", isTransport ? "Describing…" : "Rendering…");
        const req = {
            engine:    o.engine,
            name:      (dest.split("/").filter(Boolean).pop() || ""),
            // No `structure_path`.  It used to send the projects
            // sidebar's selected file, which is a SECOND fact read at a
            // second moment -- `molview.md` § 9.3a's rule is that the
            // facts which leave together were read together, and this
            // one was read from a cursor rather than from the
            // structure.  The server names the files it writes from the
            // structure that is right here in this body.
        };
        /* The kind rides only when it is not the default -- the same
         * economy the hand-over file itself keeps (`handover-procedure.md`
         * § 6: absent = optimization). */
        if (o.calculation && o.calculation !== "optimization") {
            req.calculation = o.calculation;
        }
        if (isTransport) {
            req.junction  = o.junction;
            req.bias      = o.bias || [0.0];
            req.overrides = o.overrides || {};
        } else {
            req.structure = structure;
            req.params    = o.params || {};
        }
        /* NO HAND-OVER FOR THE COMPOSITE (user ruling 2026-08-29):
         * nothing is awaiting -- stages, shape and identity are fixed
         * by design -- so the transport door answers with the FINISHED
         * task.json and this module only writes it where the user
         * chose.  Every other kind still hands to Task setup, which
         * owns the questions those kinds leave open. */
        const url = isTransport ? "/api/transport/describe"
                                : "/api/task-setup/handover";
        let out;
        try {
            const r = await fetch(url, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(req),
            });
            out = await r.json();
            if (!r.ok || !out || out.ok === false) {
                say("error",
                    (out && out.error) || ("render failed (" + r.status + ")"));
                return;
            }
        } catch (e) {
            say("error", "Could not reach the server: "
                + (e && e.message ? e.message : e));
            return;
        }

        /* The BYTES go through the content-blind file layer, never a fetch of
         * our own (`web/projects.md` § 1).  `safeSave` writes into the folder
         * the user selected, refuses at the projects root, re-lists the
         * sidebar afterwards, and returns the four-way shape below. */
        say("muted", "Writing…");
        /* The STRUCTURE goes first, and the order is deliberate: the hand-over
         * NAMES those files, so writing it before them would leave a folder
         * whose description points at nothing if the next write fails. */
        const parts = isTransport
            ? (out.files || []).map((f) => [f.name, f.text])
            : (out.structure_files || []).map((f) => [f.name, f.text])
                  .concat([[out.template_name, out.template_text],
                           [out.handover_name, out.handover_text]]);
        const toWrite = parts.filter(([n, x]) => n && typeof x === "string");
        for (const [name, text] of toWrite) {
            // safeSave(TEXT, FILENAME, opts) -- text first.
            const wrote = await projects.safeSave(text, name, { overwrite: true })
                .catch((e) => ({ ok: false, error: String(e && e.message || e) }));
            if (wrote && wrote.cancelled) {
                say("muted", "Cancelled — nothing written.");
                return;
            }
            if (!wrote || !wrote.ok) {
                say("error", "Could not write " + name + ": "
                    + ((wrote && wrote.error) || "no folder selected"));
                return;
            }
        }
        const written = toWrite.map(([n]) => n);

        /* WHAT THE CELL GATE SAID.  The same gate every structure door runs
         * answers with notices as well as refusals: a refusal is the 400 above,
         * but a notice is a box it accepted and wants read — an axis it could
         * not tell was periodic, a vacuum thinner than it expects.  These came
         * back on every send and were dropped on the floor, so a person whose
         * cell was questioned found out from the run.
         *
         * A notice does NOT hold the write back: the files are the person's own
         * parameters and refusing to save them helps nobody.  It holds back the
         * NAVIGATION, because a page that jumps away is a page whose warning
         * was never read (`tabs.md` — a tab does not decide for its user). */
        const notices = Array.isArray(out.notices) ? out.notices : [];
        if (notices.length) {
            /* The cell gate's notices carry `level` (its four-key contract,
             * periodicity_gate.py, since 2026-08-03) — reading `severity`
             * here meant the error arm could never fire. */
            const worst = notices.some((n) => n && n.level === "error")
                ? "error" : "warn";
            say(worst,
                "Wrote " + written.join(", ") + ". The cell was checked and "
                + (notices.length === 1 ? "one thing" : notices.length + " things")
                + " came back — read them, then open Task setup:\n"
                + notices.map((n) => "• "
                    + ((n && n.where) ? "[" + n.where + "] " : "")
                    + ((n && n.message) || String(n))).join("\n"));
            return;
        }

        if (isTransport) {
            /* The tab decided; the description is DONE.  No navigation
             * (user ruling 2026-08-29): the road is stated, and going
             * anywhere is the person's move.  Task setup still reads
             * the saved description like any other -- as the run
             * surface, not a hand-over target. */
            say("ok", "Described — wrote " + written.join(" and ")
                + " into " + (rel || "the selected folder")
                + ".  Next: prep run seed "
                + "(the CLI, or Task setup's prep buttons on this "
                + "folder), then launch stage by stage; summarize run "
                + "writes the I–V record.");
            return;
        }
        say("ok", "Wrote " + written.join(" and ")
            + " — opening Task setup…");
        // The sidebar's selection is shared through sessionStorage, so Task
        // setup opens on the same folder without this handing it anything.
        window.location.href = "/task-setup";
    }

    window.molbuilder = window.molbuilder || {};
    window.molbuilder.taskHandover = { send: send };
})();
