/* Task hand-over sender -- THE one door from a parameter tab to Task setup.
 *
 * The guards, the write order and the
 * notice handling are CONTRACT (`job-contracts.md` SS 2.1, SS 2.5;
 * `handover-procedure.md` SS 2), and two copies of a contract drift.
 *
 * IT COMPARES NO KIND (`handover-procedure.md`, 2.65.0): the calling tab
 * hands it the door it writes through and the fields its kind's description
 * carries.  The hand-over door is this library's (`HANDOVER`): its answer's
 * files are written into the folder.  A writing door -- one the tab names
 * with `writes` -- writes the description itself into the folder (`dest`)
 * and names what it wrote (`written`; transport.md § 3.8.2a, § 3.8.5).
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
 *                   //   to these, a refusal replacing what they say of its
 *                   //   field (handover-procedure.md § 2.1).  A tab with no
 *                   //   live check sends none.
 *     engine,       // the engine the tab describes on
 *     calculation,  // REQUIRED: the kind, stated by the tab
 *     door,         // REQUIRED: {url, next, writes?, needs?} -- the route
 *                   //   it sends to; `next`, the words for the road after
 *                   //   it; `writes` for a door that writes the description
 *                   //   itself (nothing for the page to write, no Cancel:
 *                   //   one act on the server); `needs`, [[field, words],
 *                   //   ...], a field it must be given, said in its words
 *                   //   when it is not
 *     fields,       // what the kind's description carries, sent as given
 *     showManifest, // (lines|null) -> void -- a writing door's manifest
 *   })
 */
(function () {
    "use strict";

    /* ONE HEAVY CLICK AT A TIME (`ui-contract.md` § 10): the fence covers
     * the window from the folder check to the last write, so a click on
     * another folder mid-send cannot split the files between two folders,
     * and its Cancel aborts the request in flight.  A writing door's
     * Describe offers none: it writes the description in one act on the
     * server, which an aborted request does not stop. */
    async function send(opts) {
        const fence = window.molbuilder && window.molbuilder.pageBusy;
        if (!fence) return await sendUnder(opts, undefined);
        const ctl = new AbortController();
        const writes = !!(opts && opts.door && opts.door.writes);
        return await fence.run(
            writes ? "Describing\u2026" : "Sending to Task setup\u2026",
            writes ? [] : [() => ctl.abort()],
            () => sendUnder(opts, ctl.signal));
    }

    /* THE HAND-OVER DOOR: the field every kind handed over needs -- a
     * description is OF a structure -- and the road after it, each said
     * once, for the tabs that send through it. */
    const HANDOVER = {
        url: "/api/task-setup/handover",
        next: "Next: open Task setup and pick this folder — it shows the "
            + "stages proposed and writes task.json when you save.",
        needs: [["structure",
                 "Load a structure first — a description is of something."]],
    };

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
         * true of the form just sent, so a send adds its own.
         * One the live check
         * and the send both carry (the cell gate runs in each for a stated
         * box) is shown once, and a refusal replaces what the live check said
         * of the same field: the send knows the folder the live check could
         * not, so one fact is not said twice in two severities. */
        const standing = Array.isArray(o.standing) ? o.standing : [];
        const beside = (found) => {
            const key = (i) => [i && i.severity, i && i.where,
                                i && i.message].join("\u0000");
            const sent = found || [];
            const refused = new Set(sent.filter((i) => i && i.severity === "error"
                                                       && i.where)
                                        .map((i) => i.where));
            const kept = standing.filter((i) => !(i && refused.has(i.where)));
            const seen = new Set(kept.map(key));
            return kept.concat(sent.filter((i) => !seen.has(key(i))));
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
        const door = o.door;
        if (!door || !door.url || !door.next) {
            throw new Error("task-handover.send: the tab names the door it "
                            + "writes through, and the road after it");
        }
        const fields = o.fields || {};
        /* WHAT THE DOOR NEEDS GIVEN, in its own words -- the hand-over
         * door's structure, since a description is of one. */
        for (const [field, words] of (door.needs || [])) {
            if (!fields[field]) {
                say("warn", words);
                return;
            }
        }
        /* THE FOLDER'S RULES ARE THE DOOR'S -- a calculation lives under a
         * topic, one calculation per folder (`job-contracts.md` § 2.5,
         * § 2.1 Rule 1): every door refuses, and its sentence is shown
         * below.  `rel` is the sidebar's own words for the folder. */
        const rel = (typeof projects.relativeToProjects === "function")
            ? String(projects.relativeToProjects(dest) || "") : "";

        /* THE KIND RIDES EVERY HAND-OVER, stated by the tab that sends it
         * -- the server refuses one with none (2026-10-06). */
        if (!o.calculation) {
            throw new Error("task-handover.send: the tab states its "
                            + "calculation kind");
        }
        say("muted", door.writes ? "Describing…" : "Rendering…");
        const req = Object.assign({}, fields, {
            engine:    o.engine,
            name:      (dest.split("/").filter(Boolean).pop() || ""),
        });
        req.calculation = o.calculation;
        /* THE FOLDER: the one a writing door writes the description into
         * (transport.md § 3.8.2a); the one a hand-over's files go into, a
         * SIESTA calculation's pseudopotentials checked against it, the
         * files already there counting (`handover-procedure.md` § 2.2). */
        req.dest = dest;
        let out;
        try {
            const r = await fetch(door.url, {
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
         * RUNS -- a writing door's manifest, for the page to show
         * (`engines/transport.md` § 2a.16); null for a new one. */
        if (typeof o.showManifest === "function") {
            o.showManifest(out.manifest || null);
        }

        /* A HAND-OVER'S BYTES go through the content-blind file layer, never
         * a fetch of our own (`web/projects.md` § 1), into `dest` -- the
         * folder checked above, not whatever the sidebar holds when each
         * write lands.  `writeFile` announces the folder (`web/projects.md`
         * § 2).  A writing door wrote the description itself: there is
         * nothing to write, and the folder is announced here, the one
         * call. */
        say("muted", "Writing…");
        /* The STRUCTURE goes first, and the order is deliberate: the hand-over
         * NAMES those files, so writing it before them would leave a folder
         * whose description points at nothing if the next write fails. */
        const parts = door.writes ? []
            : (out.structure_files || []).map((f) => [f.name, f.text])
                  .concat([[out.template_name, out.template_text],
                           [out.handover_name, out.handover_text]]);
        const toWrite = parts.filter(([n, x]) => n && typeof x === "string");
        if (door.writes) projects.publishFolderChanged(dest);
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
                // is announced and the line says so.
                projects.publishFolderChanged(dest);
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
        const written = door.writes ? (out.written || [])
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
        // THE ROAD AFTER IT, pointed to whether or not something came back
        // -- a notice is read, not a stop: the door's own words for it
        // (`door.next`), never a copy here.
        const next = door.next;
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
            // check on a writing door, which has no Task setup to open next
            // (`workflow.md` § 9, gates ② and ③).
            // The findings themselves are the findings panel's, above.
            const count = (notices.length === 1 ? "one finding"
                           : notices.length + " findings");
            say(worst,
                (door.writes ? described + " The description was checked: "
                             : "Wrote " + written.join(", ")
                               + ". About the structure, ")
                + count + " came back"
                + (showFindings ? (notices.length === 1 ? " — read it"
                                                        : " — read them")
                                : ", and this page has no findings panel "
                                  + "to show them in")
                + ".  " + next);
            return;
        }

        if (door.writes) {
            /* The tab decided; the description is DONE.  The road is
             * stated, and going anywhere is the person's move (below). */
            say("ok", described + "  " + next);
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
            + ".  " + next);
    }

    window.molbuilder = window.molbuilder || {};
    window.molbuilder.taskHandover = { send: send, HANDOVER: HANDOVER };
})();
