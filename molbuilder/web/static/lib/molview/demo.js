/* MolView — the in-repo demo page.
 *
 * Contract: docs/web/molview.md § 13.4 (what makes this testable at all), § 1.1
 *           (what it looks like in use), § 4 (the single import).
 * Owns:     a page that mounts a viewer over a multi-frame structure, using the
 *           module exactly as any other consumer does.
 * Called by: nobody. It is a consumer, not a layer.
 *
 * NEVER:
 *   - import anything but index.js. The demo is worth having precisely because
 *     it is held to § 4's single import like every other consumer; a demo with a
 *     private door proves nothing.
 *   - be what a lower-level change is judged by. § 13.2's third level is a
 *     FINISHED-MODULE check: a page-level test fails for anything on the page,
 *     so using one to judge a data-structure or an API change says nothing about
 *     the change and throws away work that was correct.
 *
 * THIS PAGE IS THE PROOF OF § 4. It loads the drawing library and this file, and
 * nothing else — no application runtime, no shared globals, no workspace module.
 * Everything MolView needs arrives as an argument to `mount`. If the module ever
 * grows a hidden dependency on something the app happens to have loaded, this
 * page is where it stops working.
 */
import { mount, formula } from "/static/lib/molview/index.js";

const SAMPLES = {
    water: "3\nwater\nO 0.000 0.000 0.000\nH 0.757 0.586 0.000\nH -0.757 0.586 0.000\n",
    benzene: "12\nbenzene\n"
        + "C 0.000 1.396 0\nC 1.209 0.698 0\nC 1.209 -0.698 0\nC 0.000 -1.396 0\n"
        + "C -1.209 -0.698 0\nC -1.209 0.698 0\nH 0.000 2.479 0\nH 2.147 1.240 0\n"
        + "H 2.147 -1.240 0\nH 0.000 -2.479 0\nH -2.147 -1.240 0\nH -2.147 1.240 0\n",
    gold: "4\nAu fcc\nAu 0 0 0\nAu 2.04 2.04 0\nAu 2.04 0 2.04\nAu 0 2.04 2.04\n",
};

/* A trajectory: the WHOLE MOLECULE slides along +x, its internal geometry
 * unchanged, so it is obvious at a glance whether the frames are moving on
 * screen — the failure a stand-in test cannot see (§ 13.2).
 *
 * It moves as a rigid body ON PURPOSE. The first version of this fixture slid
 * the oxygen away and left the hydrogens behind, which pulls the water apart:
 * by frame 2 the O–H distances are past bonding range, the drawing library
 * assigns NO BONDS, and the default `stick` representation — which draws bonds
 * and nothing else — renders an empty window. Every layer was correct, the
 * frame count was right, and both of the seal's self-checks reported healthy
 * (§ 10.10 asks HOW MANY frames, so a full drawing of unbondable atoms is
 * exactly what it cannot see). It read as a broken viewer for a long time.
 *
 * A demo fixture has to fail VISIBLY when the code is wrong and not otherwise;
 * one that dissociates cannot tell the two apart. */
const FRAMES = [0, 0.4, 0.8, 1.2, 1.6, 2.0].map((dx) => ([
    [dx, 0, 0], [0.757 + dx, 0.586, 0], [-0.757 + dx, 0.586, 0],
]));
const FORCES = FRAMES.map((_, f) => ([
    [2.0 - f * 0.4, 0, 0], [0.1, 0, 0], [0.1, 0, 0],
]));

/* A STAND-IN WORKSPACE — the calls of workspace.md § 5, answered from memory.
 *
 * IT IMPLEMENTS THE REAL DOOR, and that is the whole point of it. The first
 * version was a `Map` with `read(step)` / `write(step, bytes)`, which the module
 * asked for and no workspace has ever had: so the demo passed, every test passed,
 * and nothing on any tab could save. A stand-in shaped like a wish proves the
 * wish; a stand-in shaped like the dependency fails the moment the two disagree.
 *
 * It keeps one slot per tag, as the real one does — different tag, different
 * place — so a second viewer on this page would be isolated here too.
 */
function memoryWorkspace() {
    const sessions = new Map();   // tag  -> the session copy
    const points   = new Map();   // "id:step" -> a numbered point
    const ids      = new Map();   // tag  -> this tag's id
    return {
        workspaceId(tag) {
            if (!ids.has(tag)) ids.set(tag, "demo-" + tag);
            return ids.get(tag);
        },
        // THE REAL SIGNATURE (dispatcher.js): persist(tag, bytes, identity).
        // The four-arg shape this carried until 2026-08-19 was the
        // pre-2026-08-02 surface -- `identity` landed in the third slot, the
        // guard never fired, no point was ever stored, and the demo timeline
        // was silently dead.  A stand-in shaped like a wish proves the wish.
        persist(tag, bytes, identity) {
            if (bytes) sessions.set(tag, bytes);
            if (identity) {
                points.set(identity.workspace_id + ":" + identity.state_index,
                           bytes);
            }
            return true;
        },
        async readState(identity) {
            const key = identity.workspace_id + ":" + identity.state_index;
            return points.has(key) ? points.get(key) : null;
        },
        pruneStatesAbove(id, index) {
            for (const key of Array.from(points.keys())) {
                const [owner, step] = key.split(":");
                if (owner === id && Number(step) > index) points.delete(key);
            }
        },
        onPersistError() { return () => {}; },
    };
}

/* THE FILES DOOR (§ 6.7). MolView produces bytes and names a destination; what
 * happens to them is the HOST's business, and this page is the host. That
 * division is why MolView contains no file-handling code of its own — and why
 * this demo can offer a real download without the module knowing how. */
/* A HOST-SIDE SAVE. MolView hands over the structure, a destination and a STEM;
 * turning that into files is this side's job, and it does it by ASKING THE
 * SERVER -- the same generator a project save uses, so the two cannot differ.
 * The browser writes no coordinate document (molview.md § 11.7). */
function demoFiles(say) {
    return {
        async saveBinary(destination, filename, blob) {
            if (destination === "download") {
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url; a.download = filename;
                document.body.appendChild(a); a.click(); a.remove();
                setTimeout(() => URL.revokeObjectURL(url), 10000);
                say("downloaded " + filename);
                return { ok: true, files: [filename] };
            }
            say("would save " + filename + " to the project");
            return { ok: true, path: filename };
        },
        async save(destination, stem, payload) {
            let answer;
            try {
                const r = await fetch("/api/structure/export", {
                    method:  "POST",
                    headers: { "Content-Type": "application/json" },
                    /* THE STEM GOES WITH IT. MolView knows what this export is
                     * -- which structure, which frames; the server knows what
                     * the files are CALLED, because the extension follows the
                     * format and the format follows the frame count. */
                    body:    JSON.stringify({
                        structure: payload && payload.structure,
                        frames:    (payload && payload.frames) || undefined,
                        name:      stem,
                    }),
                });
                answer = await r.json();
                if (!answer || answer.ok !== true) {
                    throw new Error(answer && answer.error);
                }
            } catch (err) {
                say("export failed: " + (err && err.message ? err.message : err));
                return;
            }
            /* THE FILES COME BACK NAMED, and that is the whole of what this
             * page does with them. It appends no extension and serialises no
             * JSON: doing either would be a second answer to a question the
             * codec has already answered -- and when this page did append one,
             * a multi-frame download went out named ".xyz" with extended-XYZ
             * inside it. The .json rides along when there is metadata worth
             * keeping (§ 11.3); when there is none the server sends one file,
             * which is exactly when a save writes one. */
            const files = answer.files || [];
            const names = files.map((f) => f.name).join(" + ");
            if (destination === "download") {
                for (const file of files) {
                    const url = URL.createObjectURL(new Blob([file.text]));
                    const link = document.createElement("a");
                    link.href = url;
                    link.download = file.name;
                    link.click();
                    URL.revokeObjectURL(url);
                }
                say("downloaded " + names);
            } else {
                say("would save " + names + " to the project");
            }
        },
    };
}

async function start() {
    const host = document.getElementById("molview-demo-host");
    const status = document.getElementById("demo-status");
    const say = (text) => { if (status) status.textContent = text; };

    const viewer = await mount(host, memoryWorkspace(), {
        owner: "molview-demo",
        files: demoFiles(say),
    });

    // Mount always resolves (§ 8): on failure `ok` is false, `error` says why,
    // and `dispose` still works. A demo that assumed success would hide exactly
    // the case the rule exists for.
    if (!viewer.ok) {
        say("the viewer did not mount: " + viewer.error);
        return;
    }

    /* CATCHING IS THE HOST'S JOB (§ 6.9), and this page is a host — which is why
     * it does it here rather than leaving it to the module. A door that changes
     * the structure THROWS when the server refused it, carrying the server's own
     * sentence, and `null` means only that there was nothing to do. Two outcomes,
     * two lines, and the one the reader can act on is the message.
     *
     * A host that skips this gets an unhandled rejection and a page that goes
     * quiet — which is the exact failure § 6.9 was written to end, reappearing
     * one level up. */
    async function load(text, name) {
        say("loading " + name + "…");
        let structure;
        try {
            structure = await viewer.data.installMolecule({
                text: text, filename: name + ".xyz",
            });
        } catch (err) {
            say(err && err.message ? err.message : "could not load " + name);
            return;
        }
        if (!structure) { say("nothing to load from " + name); return; }
        report();
    }

    function report() {
        const elements = viewer.data.getElements();
        say([
            elements ? formula(elements) : "nothing loaded",
            elements ? elements.length + " atoms" : null,
            viewer.data.frameCount() > 1
                ? viewer.data.frameCount() + " frames" : null,
        ].filter(Boolean).join(" · "));
    }

    // Everything a tab would do, through the one route (§ 9.2).
    viewer.data.subscribe(report);

    const on = (id, run) => {
        const button = document.getElementById(id);
        if (button) button.addEventListener("click", run);
    };
    on("demo-water",   () => load(SAMPLES.water, "water"));
    on("demo-benzene", () => load(SAMPLES.benzene, "benzene"));
    on("demo-au-cell", () => load(SAMPLES.gold, "gold"));
    on("demo-trajectory", async () => {
        await load(SAMPLES.water, "water");
        // A trajectory extends what is already there: identity was fixed at
        // load, and a streamed frame carries coordinates only (§ 10.8).
        viewer.data.reloadFrames(FRAMES, { forces: FORCES });
        report();
    });

    /* WHAT THE SERVER SAYS, and where it lands (§ 6.8).
     *
     * Two real edits through the one route: give water a 5 Å box, then move the
     * box's corner 20 Å away so it no longer contains the molecule. The first
     * answers with a RECEIPT (what the edit did); the second with a CONDITION
     * (the box does not contain the structure, with the per-axis clearances),
     * and the Cell page shows what the LAST answer said -- a notice set belongs
     * to one exchange and the next one replaces it (§ 6.8).
     *
     * The second answer carries the condition ALONE, not both: the door adds
     * its "cell_origin set" receipt only when the result has nothing wrong with
     * it (periodicity_gate.py, the cell_origin branch), because the condition
     * already says the same thing in the words that matter.
     *
     * Nothing is faked here: no message is written by the demo, and the demo
     * does not reach into the viewer to place one. It performs an edit a user
     * could perform, and the notices arrive with the answer. */
    on("demo-bad-box", async () => {
        await load(SAMPLES.water, "water");
        try {
            await viewer.data.commitPeriodicityOp(
                "cell", [[5, 0, 0], [0, 5, 0], [0, 0, 5]]);
            await viewer.data.commitPeriodicityOp("cell_origin", [20, 20, 20]);
        } catch (err) {
            // The server refused one of the two. Its sentence says what to do
            // about it, so it is what goes on screen (§ 6.9).
            say(err && err.message ? err.message : "the cell edit did not happen");
            return;
        }
        say("the box was moved off the molecule — see the Cell page");
    });

    say("ready — pick a sample");
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
} else {
    start();
}
