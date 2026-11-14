/* The inspect card of a tab that describes a calculation of a structure --
 * the Structure-optimization tab's and the Spectrum tab's
 * (`web/tabs.md` § 3, `web/spectra.md` § 5; the markup is
 * `templates/_structure_card.html`).
 *
 * MODULE: lib/structure-card (ES module).  `attachStructureCard(opts)`.
 *
 * WHAT IT OWNS: the read-only viewer, the Load button and its readout, the
 * sidebar's pick and its double-click, the atoms-and-formula readout, the
 * load line -- and WHAT THE TAB KEEPS: the structure on screen, as the tab's
 * own draft under its own tag (`workspace.md` § 2, § 4), put back when the
 * tab is shown again.  The page keeps its forms, its checks and its Send,
 * and hears of a mounted viewer and of a structure landing.
 *
 * ARRIVING: the tab's own draft first; only an empty canvas is seeded from
 * the sidebar's pick -- a pick left highlighted is a structure to show when
 * there is none, never one to replace what the tab holds (`web/tabs.md`
 * § 3).  LOADING: the Load button (always re-read from disk) or a
 * double-click (the same file twice keeps what is on screen); a file counts
 * as shown only once it has loaded, so a load that failed can be asked
 * again.  WHAT IS KEPT IS THE STRUCTURE, NOT THE PATH: a file changed on disk
 * while the tab was away does not quietly replace what it held.
 *
 * MolView is consumed through its one door (`web/molview.md` § 9.1); the
 * workspace and the projects door are classic globals, read at call time.
 *
 *   const card = attachStructureCard({
 *       host: "viewer-host",      // the viewer's host element's id
 *       tag:  "structure-opt",    // the viewer's owner and the draft's tag
 *       onMounted(handle) {},     // the viewer this card mounted
 *       onLoaded(name) {},        // a structure landed: loaded or put back
 *   });
 *   card.data()       // the viewer's data, or null
 *   card.structure()  // the structure as the server takes it, or null
 *   card.frame()      // "frame 3 of 9, the one on screen", or null
 */
import { mount as mvMount, formula as mvFormula }
    from "/static/lib/molview/index.js";
import { molviewDoors } from "/static/lib/projects/molview-doors.js";
import { isStructureFile } from "/static/lib/projects/state.js";
import { basename, dirname, join } from "/static/lib/path-utils.js";

export function attachStructureCard(opts) {
    const $ = (id) => document.getElementById(id);
    const tag = opts.tag;
    // THE TAB'S DRAFT, under its own tag (workspace.md § 4: one saver, one
    // tag) -- the viewer's owner is the same string, so the two cannot name
    // different slots.
    const DRAFT = tag + ":panel";
    const ws = () => (window.molbuilder && window.molbuilder.workspace) || null;
    const say = (msg, kind) =>
        window.molbuilder.status.set("load-status", msg, kind || null);

    let handle = null;      // the viewer this card mounted
    let mounting = null;    // its mount, while it runs
    let shown = "";         // the file the structure on screen came from
    let candidate = "";     // the sidebar's pick
    let seq = 0;            // the newest thing asked to go on the canvas
    let queue = Promise.resolve();

    /* WHAT GOES ON THE CANVAS GOES ON ONE AT A TIME, THE NEWEST ASKED
     * WINNING: the arrival's restore and every load wait their turn, and one
     * a newer ask overtook before its turn came is skipped -- so no load
     * lands in the middle of a restore, and no older, slower load replaces a
     * newer one.  A skipped turn answers null. */
    function onCanvas(job) {
        const mine = ++seq;
        const turn = queue.then(() => (mine === seq ? job() : null));
        queue = turn.catch(() => null);
        return turn;
    }

    const data = () => ((handle && handle.ok) ? handle.data : null);

    /* THE VIEWER, MOUNTED ONCE: a structure needs somewhere to go, and a
     * viewer mounts before it has one (molview.md § 8).  A mount that failed
     * is not kept, so the next load asks again. */
    function mounted() {
        if (handle) return Promise.resolve(handle);
        const w = ws();
        const host = $(opts.host);
        if (!w || !host) return Promise.resolve(null);
        if (!mounting) {
            mounting = mvMount(host, w, { mode: "readonly", owner: tag,
                                          ...molviewDoors })
                .then((h) => {
                    handle = (h && h.ok) ? h : null;
                    mounting = null;
                    if (handle && opts.onMounted) opts.onMounted(handle);
                    return handle;
                })
                .catch((e) => {
                    mounting = null;
                    say("Viewer failed to mount: "
                        + ((e && e.message) || "render failed"), "error");
                    return null;
                });
        }
        return mounting;
    }

    /* THE READOUT, read off the model the load installed. */
    function readInfo(title) {
        const elements = data() ? (data().getElements() || []) : [];
        const n = elements.length;
        $("info-title").textContent = n ? title : "no structure loaded";
        $("info-atoms").textContent = n ? String(n) : "—";
        $("info-formula").textContent = n ? mvFormula(elements) : "—";
    }

    /* A LOAD THAT FAILED says why, and the readout no longer names the
     * structure it was about to replace. */
    function failed(name, words) {
        say(words, "error");
        $("info-title").textContent = "load failed: " + name;
        $("info-atoms").textContent = "—";
        $("info-formula").textContent = "—";
    }

    function refreshLoadButton() {
        const btn = $("load-from-sidebar-btn");
        const readout = $("load-source-readout");
        const loadable = isStructureFile(candidate);
        if (btn) btn.disabled = !loadable;
        if (!readout) return;
        readout.textContent = loadable && shown === candidate
            ? `Loaded: ${basename(candidate)}`
            : loadable
                ? `Selected: ${basename(candidate)}`
                // SAY WHY, not just that: a single click names what would
                // be loadable.
                : candidate
                    ? `Selected: ${basename(candidate)} `
                      + "(not loadable — .xyz / .pdb only)"
                    : "Pick a .xyz / .pdb in the Projects sidebar.";
    }

    /* A STRUCTURE IS ON SCREEN, loaded or put back: the one way the card
     * says so, and the page hears it. */
    function landed(name, words) {
        readInfo(name);
        say(words, "ok");
        refreshLoadButton();
        if (opts.onLoaded) opts.onLoaded(name);
    }

    /* KEPT AT THE ONE MOMENT the tab acquires a structure -- not on a timer
     * or on the way out: `pagehide` does not fire reliably on a mobile tab
     * switch.  What the viewer hands out as data is what it can be handed
     * back (molview.md § 9.3). */
    function keep() {
        const w = ws();
        const out = data() ? data().exportFile() : null;
        if (!w || !out || !out.structure) return;
        w.persistDraft(DRAFT, { v: 1, structure: out.structure,
                                loadedFrom: shown });
    }

    /* Put the tab's own draft back, or answer false with nothing to put
     * back: bytes of a layout this build does not know are not guessed at.
     * Its turn is the canvas's first (`onCanvas`), into a viewer that holds
     * nothing yet. */
    async function restore() {
        const w = ws();
        const viewer = await mounted();
        if (!w || !viewer) return false;
        let saved;
        try { saved = await w.readDraft(DRAFT); }
        catch (_) { return false; }        // unreadable storage: nothing saved
        if (!saved || saved.v !== 1 || !saved.structure) return false;
        const put = await viewer.data.installMolecule(
            { structure: saved.structure });
        if (put === null) return false;    // the viewer took nothing
        shown = saved.loadedFrom || "";
        const name = shown ? basename(shown) : "restored structure";
        landed(name, `Restored ${name} — `
                     + `${saved.structure.elements.length} atoms.`);
        return true;
    }

    /* ONE LOAD DOOR (molview.md § 9.3, model/structure.md § 2.4): the file
     * and its sidecar read by the projects parser and installed into the
     * viewer.  ``again`` is the Load button's: the file re-read from disk
     * even when it is the one on screen. */
    function load(sel, again) {
        let f = (sel && sel.file) ? String(sel.file) : "";
        // A COMMIT CARRIES THE FILE'S FULL PATH; a bare name is completed from
        // its folder, or the server would resolve it against its own.
        if (f && !dirname(f) && sel && sel.dir) f = join(sel.dir, f);
        if (!isStructureFile(f)) {
            if (f) say(`${basename(f)} is not a structure file `
                       + "(this tab loads .xyz and .pdb files).", "warn");
            return null;
        }
        if (f === shown && !again) return null;
        return onCanvas(() => loadNow(f));
    }

    async function loadNow(f) {
        const name = basename(f);
        say(`Loading ${name}…`);
        const viewer = await mounted();
        if (!viewer) {
            failed(name, "Viewer unavailable: nothing to load the structure "
                         + "into.");
            return;
        }
        let res;
        try {
            res = await window.molbuilder.projects.parser.openMolecule(viewer, f);
        } catch (e) {
            // The load door's own sentence -- the one reader's words
            // (ui-contract.md § 12) where it asked the server.
            failed(name, (e && e.message) || String(e));
            return;
        }
        if (!res || res.ok === false) {
            failed(name, (res && res.error) || "Load failed.");
            return;
        }
        shown = f;
        keep();
        landed(name, `Loaded ${name} — `
                     + `${(data().getElements() || []).length} atoms.`);
    }

    window.molbuilder.runtime.whenReady("projects").then(async (proj) => {
        candidate = proj.getCurrentFile() || "";
        refreshLoadButton();
        proj.onChange((sel) => {
            candidate = (sel && sel.file) ? sel.file : "";
            refreshLoadButton();
        });
        proj.onCommit((sel) => load(sel, false));
        const btn = $("load-from-sidebar-btn");
        if (btn) {
            btn.addEventListener("click", () => {
                if (isStructureFile(candidate)) {
                    load({ dir: proj.getCurrentDir(), file: candidate }, true);
                }
            });
        }
        try {
            // SEEDED ONLY WHEN NOTHING CAME BACK AND NOTHING WAS ASKED
            // SINCE: a file double-clicked meanwhile is the person's own.
            const restored = onCanvas(restore);
            const asked = seq;
            if ((await restored) === false && seq === asked && candidate) {
                load({ dir: proj.getCurrentDir(), file: candidate }, false);
            }
        } catch (e) {
            /* NOBODY WAITS ON THIS, so a throw would land nowhere; failing
             * the first thing the page does in silence leaves an empty canvas
             * that reads as "there was nothing here". */
            say("Could not restore this tab's structure: "
                + ((e && e.message) || "unknown error")
                + ". Pick a file and press Load.", "error");
        }
    });

    return {
        data: data,
        /* THE STRUCTURE AS THE SERVER TAKES IT -- asked of the viewer's own
         * producer, one read (molview.md § 9.3). */
        structure() {
            const out = data() ? data().exportFile() : null;
            return (out && out.structure) ? out.structure : null;
        },
        /* WHICH FRAME IS SENT, of a structure with several: the one on
         * screen; null for one frame. */
        frame() {
            const d = data();
            const n = d && typeof d.frameCount === "function"
                ? d.frameCount() : 1;
            return n > 1 ? "frame " + (d.currentFrame() + 1) + " of " + n
                           + ", the one on screen" : null;
        },
    };
}
