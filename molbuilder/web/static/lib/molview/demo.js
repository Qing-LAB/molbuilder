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

/* THE WORKSPACE DOOR (§ 8): "anything that can store and return bytes satisfies
 * it". In this page it is a Map — which is the point: the module cannot tell the
 * difference, and that is what makes it embeddable anywhere. */
function memoryWorkspace() {
    const slots = new Map();
    return {
        async read(step) { return slots.has(step) ? slots.get(step) : null; },
        async write(step, bytes) { slots.set(step, bytes); },
    };
}

/* THE FILES DOOR (§ 6.7). MolView produces bytes and names a destination; what
 * happens to them is the HOST's business, and this page is the host. That
 * division is why MolView contains no file-handling code of its own — and why
 * this demo can offer a real download without the module knowing how. */
function demoFiles(say) {
    return {
        save(destination, filename, contents) {
            if (destination === "download") {
                const url = URL.createObjectURL(new Blob([contents]));
                const link = document.createElement("a");
                link.href = url;
                link.download = filename;
                link.click();
                URL.revokeObjectURL(url);
                say("downloaded " + filename + " (" + contents.length + " bytes)");
            } else {
                say("would save " + filename + " to the project ("
                    + contents.length + " bytes)");
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

    async function load(text, name) {
        say("loading " + name + "…");
        const structure = await viewer.data.installMolecule({
            text: text, filename: name + ".xyz",
        });
        if (!structure) { say("could not load " + name); return; }
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
        viewer.data.reloadFrames(FRAMES, FORCES);
        report();
    });

    say("ready — pick a sample");
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
} else {
    start();
}
