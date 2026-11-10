/* Task hand-over sender -- THE one door from a parameter tab to Task setup.
 *
 * The guards and the notice handling are CONTRACT (`job-contracts.md`
 * SS 2.1, SS 2.5; `handover-procedure.md` SS 2), and two copies of a
 * contract drift.
 *
 * IT COMPARES NO KIND (`handover-procedure.md`, 2.65.0): the calling tab
 * hands it the door it writes through and the fields its kind's description
 * carries.  The hand-over door is this library's (`HANDOVER`).  EVERY DOOR
 * WRITES ON THE SERVER, every file or none, into the folder (`dest`), and
 * names what it wrote (`written`; `handover-procedure.md` § 2,
 * `transport.md` § 3.8.2a) -- a structure file reaches disk only through
 * the codec (`model/structure.md` § 2.4).  What differs between doors is
 * their words, which each door carries.
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
 *     door,         // REQUIRED: {url, next, doing, checked, needs?} -- the
 *                   //   route it sends to; `next`, the words for the road
 *                   //   after it; `doing`, what the page says while it
 *                   //   runs; `checked`, who spoke about the findings that
 *                   //   came back; `needs`, [[field, words], ...], a field
 *                   //   it must be given, said in its words when it is not
 *     fields,       // what the kind's description carries, sent as given
 *     frame,        // which frame of a structure with several is sent --
 *                   //   "frame 3 of 8, the one on screen" -- said with the
 *                   //   confirmation; null for one frame
 *     showManifest, // (lines|null) -> void -- a writing door's manifest
 *   })
 */
(function () {
    "use strict";

    /* ONE HEAVY CLICK AT A TIME (`ui-contract.md` § 10): the fence covers
     * the window from the folder check to the answer, so a click on another
     * folder mid-send cannot move where it writes.  It offers no Cancel:
     * the server writes in one act, which an aborted request does not stop. */
    async function send(opts) {
        const fence = window.molbuilder && window.molbuilder.pageBusy;
        if (!fence) return await sendUnder(opts);
        return await fence.run((opts && opts.door && opts.door.doing) || "",
                               [], () => sendUnder(opts));
    }

    /* THE HAND-OVER DOOR: the field every kind handed over needs -- a
     * description is OF a structure -- and the road after it, each said
     * once, for the tabs that send through it. */
    const HANDOVER = {
        url: "/api/task-setup/handover",
        doing: "Sending to Task setup\u2026",
        checked: "About the structure,",
        next: "Next: open Task setup and pick this folder — it shows the "
            + "stages proposed and writes task.json when you save.",
        needs: [["structure",
                 "Load a structure first — a description is of something."]],
    };

    async function sendUnder(opts) {
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
        if (!door || !door.url || !door.next || !door.doing
                || !door.checked) {
            throw new Error("task-handover.send: the tab names the door it "
                            + "writes through and the door's words -- what "
                            + "it is doing, who checked, the road after it");
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
        say("muted", door.doing);
        const req = Object.assign({}, fields, {
            engine:    o.engine,
            // The folder's own name, through THE ONE PATH DOOR
            // (`lib/path-utils.js`), which the page header loads.
            name:      window.molbuilder.path.basename(dest),
        });
        req.calculation = o.calculation;
        /* THE FOLDER every door writes into (`handover-procedure.md` § 2,
         * transport.md § 3.8.2a); a SIESTA calculation's pseudopotentials
         * checked against it, the files already there counting
         * (`handover-procedure.md` § 2.2). */
        req.dest = dest;
        // THE ONE READER (ui-contract.md § 12); a send is a click, after
        // the page's modules have run.
        const answer = await window.molbuilder.server.ask(door.url, {
            method: "POST", body: req });
        const out = answer.body;
        if (!answer.ok) {
            /* A refusal's findings, row by row, where the page shows its
             * findings -- the door's own sentence in the status line. */
            // ...and a refusal that names none clears the last send's,
            // which are not this one's.
            if (showFindings) {
                showFindings(beside(out && Array.isArray(out.findings)
                                    ? out.findings : []));
            }
            say("error", answer.error);
            return;
        }

        /* WHAT A CALCULATION THAT REUSES A FINISHED ONE WILL COPY, AND WHAT
         * RUNS -- a writing door's manifest, for the page to show
         * (`engines/transport.md` § 2a.16); null for a new one. */
        if (typeof o.showManifest === "function") {
            o.showManifest(out.manifest || null);
        }

        /* WRITTEN ON THE SERVER, every file or none (`handover-procedure.md`
         * § 2): the folder announced here, the one call (`projects.md` § 2). */
        projects.publishFolderChanged(dest);
        const written = Array.isArray(out.written) ? out.written : [];

        /* WHAT CAME BACK -- the cell gate's notices on a hand-over, the
         * description's own check on a writing door -- goes to the findings
         * panel (R2a: a notice is a finding), beside what the live check
         * still says.  A notice does NOT hold the write back: the files are
         * the person's own parameters and refusing to save them helps
         * nobody. */
        const notices = Array.isArray(out.notices) ? out.notices : [];
        if (showFindings) showFindings(beside(notices));
        const wrote = "Wrote " + written.join(", ")
            + " into " + (rel || "the selected folder")
            + (o.frame ? " \u2014 the structure at " + o.frame : "")
            + (out.handover_here
               ? " \u2014 replacing the hand-over that was there" : "")
            + ".";
        /* NO NAVIGATION (user, 2026-09-19: *"we just skip the fancy tab to
         * tab jump connection to avoid implicit coupling"*): this tab's job
         * ends when the files are on disk; the road after it is the door's
         * own words (`door.next`), and which tab you open next is yours. */
        if (notices.length) {
            // THE STATUS LINE says how many came back and to read them --
            // the findings keep their own severities where they are drawn.
            const worst = notices.some((n) => n && n.severity === "error")
                ? "error" : "warn";
            const count = (notices.length === 1 ? "one finding"
                           : notices.length + " findings");
            say(worst, wrote + "  " + door.checked + " " + count + " came back"
                + (showFindings ? (notices.length === 1 ? " — read it"
                                                        : " — read them")
                                : ", and this page has no findings panel "
                                  + "to show them in")
                + ".  " + door.next);
            return;
        }
        say("ok", wrote + "  " + door.next);
    }

    window.molbuilder = window.molbuilder || {};
    window.molbuilder.taskHandover = { send: send, HANDOVER: HANDOVER };
})();
