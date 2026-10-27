/* Task hand-over sender -- THE one door from a parameter tab to Task setup.
 *
 * The guards, the write order and the
 * notice handling are CONTRACT (`job-contracts.md` SS 2.1, SS 2.5;
 * `handover-procedure.md` SS 2), and two copies of a contract drift.
 *
 * Consumers:
 *   * structure-optimization/viewer.js  -- engine siesta|pyscf, optimization
 *   * lib/spectra/core.js               -- engine pyscf, calculation vibration
 *   * lib/transport/core.js             -- engine siesta, calculation
 *     transport: NO structure/params -- `junction` (the citation),
 *     `bias` (volts, 0.0 first), `low_bias_approximation`,
 *     `swap_electrodes`, `stages` (per-rung override bags) and `shared`
 *     ride instead, with the folder (`dest`) the door
 *     writes the description into -- a folder holding one is refused --
 *     and the answer names what it wrote (`written`; transport.md
 *     § 3.8.2a, § 3.8.5)
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
 *     showFindings, // (issues[]) -> void -- the tab's findings panel,
 *                   //   through lib/validation-findings.js (science/
 *                   //   validation.md § 4.1 R2, R2a: a notice IS a finding,
 *                   //   and one module draws them).  Handed every notice
 *                   //   and every finding a refusal carries, beside the
 *                   //   `standing` ones, after every send.
 *     standing,     // issues[] -- what the tab's live check shows now, still
 *                   //   true of the form it sends: a send adds its findings
 *                   //   to these and clears none (handover-procedure.md
 *                   //   § 2.1).  A tab with no live check sends none.
 *     structure,    // the envelope from the tab's own MolView exportFile()
 *     engine,       // "siesta" | "pyscf"
 *     params,       // collectForm() output, keyed by catalogue names
 *     calculation,  // REQUIRED: "optimization" | "vibration" | "transport"
 *     junction,     // transport only: the citation string
 *     bias,         // transport only: [volts...], 0.0 first
 *     low_bias_approximation, // transport only: true | false for several voltages
 *     swap_electrodes, // transport only: the electrode rename on the
 *                   //   calculation's own copy of the junction
 *     stages,       // transport only: {rung: {catalogue item: value}}, per-rung bags
 *     shared,       // transport only: {catalogue item: value}, the shared panel
 *   })
 */
(function () {
    "use strict";

    /* ONE HEAVY CLICK AT A TIME (`ui-contract.md` § 10): the fence covers
     * the window from the folder check to the last write, so a click on
     * another folder mid-send cannot split the files between two folders,
     * and its Cancel aborts the request in flight.  A transport Describe
     * offers none: its door writes the description in one act on the
     * server, which an aborted request does not stop. */
    async function send(opts) {
        const fence = window.molbuilder && window.molbuilder.pageBusy;
        if (!fence) return await sendUnder(opts, undefined);
        const ctl = new AbortController();
        const describe = !!(opts && opts.calculation === "transport");
        return await fence.run(
            describe ? "Describing\u2026" : "Sending to Task setup\u2026",
            describe ? [] : [() => ctl.abort()],
            () => sendUnder(opts, ctl.signal));
    }

    async function sendUnder(opts, signal) {
        const o = opts || {};
        const projects = o.projects;
        const say = (typeof o.say === "function") ? o.say : function () {};
        /* THE FINDINGS GO TO THE ONE RENDERER, never into the status line:
         * a second renderer is what R2 forbids and R2a names
         * (a notice is a finding).  A consumer that hands no renderer is
         * told so, rather than having its findings dropped in silence
         * (R3). */
        const showFindings = (typeof o.showFindings === "function")
            ? o.showFindings : null;
        /* BESIDE WHAT THE LIVE CHECK ALREADY SAYS: its findings are still
         * true of the form just sent, so a send adds its own and clears none.
         * One the live check
         * and the send both carry (the cell gate runs in each for a stated
         * box) is shown once. */
        const standing = Array.isArray(o.standing) ? o.standing : [];
        const beside = (found) => {
            const key = (i) => [i && i.severity, i && i.where,
                                i && i.message].join("\u0000");
            const seen = new Set(standing.map(key));
            return standing.concat((found || []).filter((i) => !seen.has(key(i))));
        };
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
         * (transport.md § 3.1).  Every other kind is OF a
         * structure and must bring one. */
        if (!structure && !isTransport) {
            say("warn", "Load a structure first — a description is "
                + "of something.");
            return;
        }
        if (isTransport && !o.junction) {
            say("warn", "Pick the relaxed junction first — a finished "
                + "relaxation run, or a structure pair saved from one: the "
                + "composite derives everything from that citation.");
            return;
        }
        /* THE FOLDER'S RULES ARE THE DOOR'S -- a calculation lives under a
         * topic, one calculation per folder (`job-contracts.md` § 2.5,
         * § 2.1 Rule 1): both hand-over doors refuse (`/api/task-setup/
         * handover`, `/api/transport/describe`), and their sentence is shown
         * below.  `rel` is the sidebar's own words for the folder. */
        const rel = (typeof projects.relativeToProjects === "function")
            ? String(projects.relativeToProjects(dest) || "") : "";

        say("muted", isTransport ? "Describing…" : "Rendering…");
        const req = {
            engine:    o.engine,
            name:      (dest.split("/").filter(Boolean).pop() || ""),
        };
        /* THE KIND RIDES EVERY HAND-OVER, stated by the tab that sends it
         * -- the server refuses one with none (2026-10-06). */
        if (!o.calculation) {
            throw new Error("task-handover.send: the tab states its "
                            + "calculation kind (optimization, vibration, "
                            + "transport)");
        }
        req.calculation = o.calculation;
        if (isTransport) {
            req.junction  = o.junction;
            req.bias      = o.bias || [0.0];
            // THE LOW-BIAS APPROXIMATION, as chosen -- none for a single
            // voltage (`engines/transport.md` § 2a.10).
            req.low_bias_approximation =
                typeof o.low_bias_approximation === "boolean"
                    ? o.low_bias_approximation : null;
            // THE ELECTRODE RENAME ON THE CALCULATION'S OWN COPY OF THE
            // JUNCTION (transport.md § 4), as the tab holds it.
            req.swap_electrodes = !!o.swap_electrodes;
            // THE CALCULATION IT REUSES, when a transport calculation's
            // folder was picked (`engines/transport.md` § 2a.16).
            if (o.reuses) req.reuses = o.reuses;
            req.stages    = o.stages || {};      // {rung: {item: value}}
            req.shared    = o.shared || {};
            // THE FOLDER THE DOOR WRITES INTO (transport.md § 3.8.2a).
            req.dest      = dest;
        } else {
            req.structure = structure;
            req.params    = o.params || {};
            // The folder the files go into: a SIESTA calculation's
            // pseudopotentials are checked against it, the files already
            // there counting (`handover-procedure.md` § 2.2).
            req.dest      = dest;
        }
        /* NO HAND-OVER FOR THE COMPOSITE (user ruling 2026-08-29):
         * nothing is awaiting -- stages, shape and identity are fixed
         * by the kind -- so the transport door WRITES the finished
         * description itself, into the folder named here (transport.md
         * § 3.8.2a).  Every other kind still hands to Task setup, which
         * owns the questions those kinds leave open. */
        const url = isTransport ? "/api/transport/describe"
                                : "/api/task-setup/handover";
        let out;
        try {
            const r = await fetch(url, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(req),
                signal: signal,
            });
            out = await r.json();
            if (!r.ok || !out || out.ok === false) {
                /* A refusal's findings, row by row, where the page shows its
                 * findings -- the door's own sentence in the status line. */
                // ...and a refusal that names none clears the last send's,
                // which are not this one's.
                if (showFindings) {
                    showFindings(beside(out && Array.isArray(out.findings)
                                        ? out.findings : []));
                }
                say("error",
                    (out && out.error) || ("render failed (" + r.status + ")"));
                return;
            }
        } catch (e) {
            if (projects.isCancelError(e)) {
                say("muted", "Cancelled — nothing written.");
                return;
            }
            say("error", "Could not reach the server: "
                + (e && e.message ? e.message : e));
            return;
        }

        /* WHAT A CALCULATION THAT REUSES A FINISHED ONE WILL COPY, AND WHAT
         * RUNS -- the transport describe's manifest, for the page to show
         * (`engines/transport.md` § 2a.16); null for any other. */
        if (isTransport && typeof o.showManifest === "function") {
            o.showManifest(out.manifest || null);
        }

        /* A HAND-OVER'S BYTES go through the content-blind file layer, never
         * a fetch of our own (`web/projects.md` § 1), into `dest` -- the
         * folder checked above, not whatever the sidebar holds when each
         * write lands.  `writeFile` re-lists the sidebar afterwards.  A
         * transport description was written by its door: there is nothing
         * to write, and the sidebar is re-listed to show it. */
        say("muted", "Writing…");
        /* The STRUCTURE goes first, and the order is deliberate: the hand-over
         * NAMES those files, so writing it before them would leave a folder
         * whose description points at nothing if the next write fails. */
        const parts = isTransport ? []
            : (out.structure_files || []).map((f) => [f.name, f.text])
                  .concat([[out.template_name, out.template_text],
                           [out.handover_name, out.handover_text]]);
        const toWrite = parts.filter(([n, x]) => n && typeof x === "string");
        if (isTransport && typeof projects.refresh === "function") {
            await projects.refresh();
        }
        const done = [];
        for (const [name, text] of toWrite) {
            // writeFile(PATH, TEXT, opts) -- path first.
            const wrote = await projects.writeFile(
                dest.replace(/\/$/, "") + "/" + name, text,
                { overwrite: true, signal: signal })
                .catch((e) => ({ ok: false, error: String(e && e.message || e) }));
            if (wrote && wrote.aborted) {
                // THE WRITE IN FLIGHT MAY HAVE LANDED -- an aborted request
                // does not stop one the server already took -- so the folder
                // is listed again and the line says so.
                if (typeof projects.refresh === "function") {
                    await projects.refresh();
                }
                say("muted", "Cancelled while writing " + name + " — "
                    + (done.length ? "after " + done.join(" and ") + "; " : "")
                    + "it may have been written: the sidebar shows the "
                    + "folder as it is.");
                return;
            }
            if (!wrote || !wrote.ok) {
                say("error", "Could not write " + name + ": "
                    + ((wrote && wrote.error) || "the write failed"));
                return;
            }
            done.push(name);
        }
        const written = isTransport ? (out.written || [])
                                    : toWrite.map(([n]) => n);

        /* WHAT THE CELL GATE SAID.  The same gate every structure door runs
         * answers with notices as well as refusals: a refusal is the 400 above,
         * but a notice is a box it accepted and wants read — an axis it could
         * not tell was periodic, a vacuum thinner than it expects.
         *
         * A notice does NOT hold the write back: the files are the person's own
         * parameters and refusing to save them helps nobody. */
        const notices = Array.isArray(out.notices) ? out.notices : [];
        if (showFindings) showFindings(beside(notices));
        // THE ROAD FROM A DESCRIBED TRANSPORT CALCULATION, pointed to whether
        // or not something came back -- a notice is read, not a stop.  The
        // road itself is said once, on the page's "After you describe" card
        // (`transport_calculation.html`); this line names it, never a copy.
        const transportNext = "Next: the steps under \u201cAfter you "
            + "describe\u201d, on this page.";
        const described = "Described — wrote " + written.join(" and ")
            + " into " + (rel || "the selected folder") + ".";
        if (notices.length) {
            // TWO TONES ON PURPOSE, and it is not R4's vocabulary.
            //
            // This is a page STATUS line about an action that has happened --
            // `status error|ok|warn|muted`, owned by `page-shell.css` -- not a
            // finding row.  There is no `.status.info`, and there should not
            // be: whatever came back, the line's job is "go and read it", so
            // the quiet tone would be the wrong one even if it existed.  The
            // findings themselves keep their own severities wherever they are
            // rendered.
            const worst = notices.some((n) => n && n.severity === "error")
                ? "error" : "warn";
            // WHO SPOKE is the door's: on a hand-over, what the server said
            // of the structure -- the cell gate, or a structure put back in
            // the user order (`model/overview.md` § 2.2a); the description's own
            // check on the transport describe, which has no Task setup to
            // open next (`workflow.md` § 9, gates ② and ③).
            // The findings themselves are the findings panel's, above.
            const count = (notices.length === 1 ? "one finding"
                           : notices.length + " findings");
            say(worst,
                (isTransport ? described + " The description was checked: "
                             : "Wrote " + written.join(", ")
                               + ". About the structure, ")
                + count + " came back"
                + (showFindings ? (notices.length === 1 ? " — read it"
                                                        : " — read them")
                                : ", and this page has no findings panel "
                                  + "to show them in")
                + (isTransport ? ".  " + transportNext
                               : ", then open Task setup."));
            return;
        }

        if (isTransport) {
            /* The tab decided; the description is DONE.  The road is
             * stated, and going anywhere is the person's move (below). */
            say("ok", described + "  " + transportNext);
            return;
        }
        /* NO NAVIGATION (user, 2026-09-19: *"we just skip
         * the fancy tab to tab jump connection to avoid implicit
         * coupling"*).  This tab's job ends when the files are on disk;
         * which tab you open next is yours.
         *
         * `web/tabs.md` § 1 already forbids an in-memory hand-off, for
         * costs that are all about hidden state.  A jump that depends on
         * which folder the other tab happens to remember is the same
         * hazard wearing a URL. */
        say("ok", "Wrote " + written.join(" and ")
            + " into " + (rel || "the selected folder")
            + (out.handover_here
               ? " — replacing the hand-over that was there" : "")
            + ".  Next: open Task setup and pick this folder — it asks for "
            + "the shape and the stages, which this hand-over leaves out on "
            + "purpose, and writes task.json when you save.");
    }

    window.molbuilder = window.molbuilder || {};
    window.molbuilder.taskHandover = { send: send };
})();
