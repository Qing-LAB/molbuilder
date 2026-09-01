/* MolView — assembling a viewer, and the handle that comes back. § 7 level 2.
 *
 * Contract: docs/web/molview.md § 8 (making and tearing down a viewer), § 8.1–8.3
 *           (what the card is made of, how it sizes, how it folds), § 9.2 (the
 *           handle).
 * Owns:     assembling the card, the panel, the controls and MolView's own menu;
 *           the playback timer; the handle itself.
 * Called by: a tab, through index.js — and a tab holds a handle and reaches its
 *           viewer only through it.
 *
 * A HANDLE *IS* A VIEWER: one owner, one structure, one of everything (§ 7).
 *
 * NEVER (§ 7 level 2, § 9.2):
 *   - hold structure data of its own;
 *   - answer a question the model already answers. "Adding a read to the handle
 *     that the model already answers is the specific move this rule forbids" —
 *     a mirrored read is a second surface over the same fact, and one of the two
 *     is the one somebody forgets to update;
 *   - expose the drawing library's object, or the DOM inside the card. (§ 9.2
 *     names the library there; this file does not, because § 5.3 says everything
 *     above the sealed layer can be read end to end without learning which
 *     library draws the molecule — a "never" that names it teaches it.);
 *   - accept a finished appearance. There is no "set the arrows", "set the
 *     atom-number labels", "show a busy state", "add a toggle": arrows, labels
 *     and the highlight are WORKED OUT FROM THE DATA by the renderEngine;
 *   - own playback's effect on truth. The timer lives here, but it moves the
 *     frame through the same write everyone else uses (§ 6.4).
 */
"use strict";

import { createModel } from "./model.js";
import { createRenderEngine } from "./render-engine.js";
import { create as createEmbed } from "./3dmol-embed.js";
import { attachUiContext } from "./ui-context.js";
import { mountControls, SPEED_MIN_MS, SPEED_MAX_MS, SPEED_DEFAULT_MS }
    from "./ui.js";


/* The card's own class names. They are the carried stylesheet's (§ 8.1–8.3), so
 * this file does not get to choose them — the design and the markup it styles
 * arrived together and are changed together. */
const CARD   = "molviewer-card";
const BODY   = "molviewer-card-body";
const VIEWER = "molviewer-window";
const PANEL  = "molviewer-panel";
const FOLD   = "molviewer-panel-fold-btn";



/**
 * Assemble a complete viewer in one call.
 *
 * MOUNT ALWAYS RESOLVES (§ 8). It never rejects and never returns nothing: it
 * always gives back an object with `ok` and a real `dispose`. On failure `ok` is
 * false and `error` says why, and `dispose` still works — so a caller's teardown
 * path needs no special case for the viewer that did not build.
 *
 * @param hostEl     an empty element. MolView builds everything inside it.
 * @param workspace  a DOOR, not a module: anything that can store and return
 *                   bytes satisfies it (§ 8). That is what lets a viewer mount
 *                   in a test page with a stand-in, and it is the whole of § 4's
 *                   "nothing it needs comes from a global".
 * @param opts       `{owner, mode, files}` — `owner` names the viewer and
 *                   therefore everything in it (§ 5.6); `mode: "readonly"`
 *                   freezes the master copy (§ 9.4); `files` is the door bytes
 *                   leave through (§ 6.7), handed in the same way the workspace
 *                   is, because MolView never reaches a file itself.
 *
 *                   There is no `reservedLabels` here any more. The names
 *                   MolView offers before anyone has used them are MolView's
 *                   own (model-jobs.js): they are conveniences, and asking five
 *                   pages to hand in the same four names meant five copies of
 *                   one list — of which only the demo page ever passed it, so
 *                   every label chip on every real page came out the same
 *                   colour.
 */
export async function mount(hostEl, workspace, opts) {
    opts = opts || {};
    const parts = [];                     // torn down in reverse order of assembly

    const fail = (message) => ({
        ok: false,
        error: message,
        dispose() { for (const p of parts.reverse()) { try { p(); } catch (_) {} } },
    });

    if (!hostEl) return fail("mount: no host element");
    // A viewer with no owner has no identity, which is why one is always given.
    if (!opts.owner) return fail("mount: no owner");

    let card;
    try {
        card = buildCard(hostEl, opts);
        parts.push(() => { try { card.root.remove(); } catch (_) {} });
    } catch (e) {
        return fail("mount: could not build the card — " + (e && e.message));
    }

    /* ── The sizing contract (§ 8.2) ───────────────────────────────────────
     *
     * The floor is the STACKED minimum, not the side-by-side sum: below the sum
     * the card does not break, it stacks, and both pieces stay usable. So only a
     * host narrower than the wider single piece is a broken embed — and treating
     * the sum as the floor would make MolView refuse hosts where it works fine.
     *
     * The number is read from the card's own custom property rather than written
     * here, so the stylesheet stays the one place the layout is decided.
     *
     * What is measured is the HOST, not the card. The card is `width: 100%` of
     * whatever it was given, so its own width is either the host's or — before
     * layout has run — nothing at all. The question being asked is "did the host
     * give us enough room", and only the host can answer it.
     */
    const floor = minimumWidth(card.root);
    const available = hostEl.clientWidth;
    if (floor && available && available < floor) {
        // "A viewer that cannot fit renders a blank card with the error written
        // in it, rather than a half-built viewer" (§ 8).
        writeSizingError(card, available, floor);
        return fail("mount: host is " + available + "px, MolView needs "
                    + floor + "px");
    }

    /* ── The layers, assembled bottom up ───────────────────────────────── */

    let embed;
    try {
        embed = createEmbed(card.canvas, {});
        parts.push(() => embed.dispose());
    } catch (e) {
        writeSizingError(card, 0, 0, "The 3D viewer could not start.");
        return fail("mount: the drawing could not start — " + (e && e.message));
    }

    /* `owner` goes down as well as onto the card. It is the viewer's identity
     * (§ 5.6) and it is the tag its saved work is kept under (workspace.md § 4),
     * and those are the same fact — so it is passed, not re-derived. Two viewers
     * on one page therefore save into two slots without either of them arranging
     * it. */
    const model = createModel({
        mode: opts.mode, workspace: workspace, owner: opts.owner,
    });
    const engine = createRenderEngine(embed);
    model._attachRenderer(engine);

    /* The drawing settings reach the drawing (§ 10.1). Until this line existed
     * the View menu changed a store NOBODY READ: the buttons took their pressed
     * state, the store fired, its only subscriber was the menu updating itself,
     * and the picture never changed. A door the module owns but never opens is
     * indistinguishable from a broken one.
     *
     * Seeded once with the store's current value so the drawing starts on the
     * store's defaults rather than the seal's own — two sets of defaults that
     * merely happen to agree is the second home § 7 forbids. */
    engine.drawingChanged(model.view.get());
    parts.push(model.view.subscribe((settings) => engine.drawingChanged(settings)));

    /* The view context (§ 11.2b): how the user was looking, restored on the
     * first structure and written back debounced -- never a state, never the
     * badge, never the draft.  A read-only viewer has no truth lane, so the
     * lane carries its selection too. */
    parts.push(attachUiContext({
        model, engine, workspace,
        owner: opts.owner,
        canvasEl: card.canvas,
        hasTruthLane: opts.mode !== "readonly",
    }));

    /* A click in the window arrives at the bottom and is the data's business,
     * not the drawing's (§ 9.5).  WHICH TRACK it lands in — the selection or
     * the measurement — is `model.pickAtom`'s to decide, and this file does not
     * get a second opinion about it (§ 11.6).
     *
     * THE INDEX IS TRANSLATED HERE, because this is the only entry where a
     * drawn index exists -- the atom rows already carry real ones (§ 11.6).
     * Under isolate the drawn numbering no longer matches the real one
     * (§ 6.5), and this used to drop the click entirely for that reason.  But
     * the answer to a numbering problem is the map, not a closed door: the
     * renderer owns one and `drawnToOriginal` asks it, so what reaches
     * `pickAtom` is an atom rather than a seat.
     *
     * WHAT STAYS SHUT under isolate is SELECTING by window click, and for its
     * own reason rather than a numbering one: isolate draws only the selected
     * atoms, so clicking one to toggle it would make it vanish under the
     * cursor.  The rows curate that instead.  Measuring has no such
     * circularity -- its picks change nothing -- and § 6.5's `measured` list
     * is already exempt from isolate on the way OUT, so this makes the two
     * directions agree (user, 2026-08-31). */
    embed.onPick((drawnIndex) => {
        const isolating = model.selection.getState().isolate;
        if (isolating && !model.measurement.getState().active) return;
        const original = model.drawnToOriginal(drawnIndex);
        if (typeof original !== "number") return;
        model.pickAtom(original);
    });

    /* ── Playback (§ 9.2) ──────────────────────────────────────────────────
     *
     * The timer lives here, and that is the whole of what the handle owns about
     * it. It moves the frame through `setCurrentFrame` — the same write the
     * slider, a keyboard shortcut and a restored session use — so every
     * subscriber hears it the same way and nothing needs its own "did it
     * change?" check (§ 6.4).
     */
    let timer = null;
    let loop = true;
    // Playback speed, in milliseconds per frame — the interval, straight to
    // setInterval. § 1.1 fixes the range and the default.
    let speedMs = SPEED_DEFAULT_MS;

    function stop() {
        if (timer !== null) { clearInterval(timer); timer = null; }
    }
    parts.push(stop);

    /* PLAY TAKES NO ARGUMENTS: speed is SET (below) and then played at. § 9.2 —
     * the handle exposes what a viewer does, never the knobs it does it with. */
    function play() {
        stop();
        const count = model.frameCount();
        if (count < 2) return;
        timer = setInterval(() => {
            const n = model.frameCount();
            const next = model.currentFrame() + 1;
            if (next >= n) {
                if (!loop) { stop(); return; }
                model.setCurrentFrame(0);
                return;
            }
            model.setCurrentFrame(next);
        }, speedMs);
    }

    /* ── The handle (§ 9.2) ────────────────────────────────────────────────
     *
     * Lifecycle, playback, and ONE route to the model, written `viewer.data`.
     * There is no other way to it, and no other viewer's model is reachable from
     * it. The handle CONTAINS the model; it does not mirror it.
     */
    /* ── WHAT LEAVES THROUGH THE HANDLE (§ 9.2, § 9.3) ─────────────────────
     *
     * The module's own pieces -- the panel, the view context, the render engine
     * -- are handed `model` itself, because they ARE the module.  A TAB IS NOT,
     * and it gets less: the three stores are replaced by surfaces carrying the
     * doors a consumer legitimately needs and nothing more.
     *
     * WHY (user, 2026-08-31: "why would you expose all internal when there is
     * no need for outside user to have them?").  `selection` alone has eighteen
     * doors; outside code uses three.  Among the fifteen it does not use is
     * `toggle` -- and a consumer calling `selection.toggle(i)` writes a pick
     * without going through `pickAtom`, so the rule that decides
     * measuring-vs-selecting never runs.  That is not hypothetical: it is
     * exactly the defect the measurement track had until this was written.
     *
     * The line is: READS ARE OPEN, and the only writes that leave are the ones
     * a consumer was found to need.  Everything else -- adding, removing,
     * inverting, filtering, labelling, adopting, picking -- is the panel's, and
     * the panel is inside.
     *
     * It is `Object.create(model)` so every OTHER door (`applyOp`,
     * `installMolecule`, the frame doors, the history, the getters) passes
     * through unchanged: this narrows three properties, it does not re-list the
     * model.
     */
    const data = Object.create(model);
    Object.defineProperties(data, {
        /* THE SELECTION, MINUS THE ONE DOOR THAT HAS A ROUTER.
         *
         * Everything here is an honest selection operation a consumer may ask
         * for.  `toggle` is not, and it is the only one left out: a pick is
         * routed by `pickAtom`, which decides whether a click means measuring
         * or selecting.  A consumer calling `toggle` writes straight past that
         * -- which is the defect the measurement track had, one store over.
         * Anything wanting to toggle an atom calls `pickAtom`. */
        selection: { enumerable: true, value: {
            get:           () => model.selection.get(),
            getState:      () => model.selection.getState(),
            subscribe:     (fn) => model.selection.subscribe(fn),
            add:           (a) => model.selection.add(a),
            remove:        (a) => model.selection.remove(a),
            clear:         () => model.selection.clear(),
            all:           (n) => model.selection.all(n),
            invert:        (n) => model.selection.invert(n),
            adopt:         (a) => model.selection.adopt(a),
            setSwitch:     (n, v) => model.selection.setSwitch(n, v),
            setIsolate:    (on) => model.selection.setIsolate(on),
            writeLabel:    (n, a, v) => model.selection.writeLabel(n, a, v),
            setEditor:     (m) => model.selection.setEditor(m),
            addFilter:     (r) => model.selection.addFilter(r),
            updateFilter:  (i, p) => model.selection.updateFilter(i, p),
            removeFilter:  (i) => model.selection.removeFilter(i),
            setCombinator: (c) => model.selection.setCombinator(c),
        } },
        view: { enumerable: true, value: {
            get:       () => model.view.get(),
            subscribe: (fn) => model.view.subscribe(fn),
            // `lib/transport/core.js` picks the style its page wants.
            set:       (name, value) => model.view.set(name, value),
        } },
        measurement: { enumerable: true, value: {
            getState:       () => model.measurement.getState(),
            subscribe:      (fn) => model.measurement.subscribe(fn),
            setActive:      (on) => model.measurement.setActive(on),
            clear:          () => model.measurement.clear(),
            positions:      () => model.measurementPositions(),
            requestPicking: () => model.requestPicking(),
        } },
    });

    const handle = {
        ok: true,
        error: null,

        // The one route. Everything a tab wants to read or change about this
        // viewer's data is here, behind the rules and the read-only gate.
        data: data,

        // Playback. Owning the timer is the handle's job; owning what the frame
        // number IS remains the model's.
        play,
        pause: stop,
        isPlaying() { return timer !== null; },
        setLoop(on) { loop = !!on; },
        getLoop()   { return loop; },

        /* HOW FAST IT PLAYS, in milliseconds per frame — set and read like
         * loop, because it is the same kind of thing: a setting of playback,
         * and playback is the handle's (§ 9.2).
         *
         * The clamp is HERE rather than in the control, so the range § 1.1
         * promises holds for every caller and not only for the one that happens
         * to have an `<input min max>` in front of it. Nonsense — a blank box,
         * text, NaN — leaves the speed alone rather than stopping the timer
         * dead. A running timer is restarted at the new speed, so "takes effect
         * now" is playback's rule and not a sequence each caller repeats. */
        setSpeed(ms) {
            const n = Math.round(Number(ms));
            if (!Number.isFinite(n)) return;
            speedMs = Math.min(SPEED_MAX_MS, Math.max(SPEED_MIN_MS, n));
            if (timer !== null) play();
        },
        getSpeed() { return speedMs; },

        // Hear that something changed, rather than polling for it.
        onChange(fn) { return model.subscribe(fn); },

        /* Re-fit the camera on the structure (§ 9.6, § 1.1's `⟲`).
         *
         * It sits HERE rather than on the model because it is not data: § 9.6
         * keeps the camera out of every layer above the drawing, so there is no
         * setting to write and nothing to read back — it is an action on the
         * window, like playback, which is the other thing this handle owns.
         * (§ 8.5 leaves Reset's owner open; this is the answer that adds no
         * second home for anything.) */
        resetView() { engine.resetView(); },

        /* A picture of the window as drawn (§ 11.3's Image; § 9.2: the
         * handle owns actions on the window).  `{width, height}` asks for a
         * size; omitted, the window's own. */
        capture(size) {
            const s = size || {};
            return engine.capture(s.width, s.height);
        },

        dispose() {
            for (const p of parts.slice().reverse()) {
                try { p(); } catch (_) {}
            }
        },
    };

    /* ── The controls MolView draws (§ 8.5) ────────────────────────────────
     *
     * Each is a caller of the model. The frame bar is handed BOTH — the model
     * for the frame and its range, the handle for playback — because those are
     * two different owners and it reads each from where it lives.
     */
    const controls = mountControls(card, model, handle, opts.files);
    parts.push(() => controls.dispose());

    return handle;
}


/* ══ The card (§ 8.1) ════════════════════════════════════════════════════════
 *
 * One card holds the 3D window and the panel side by side, with the fold handle
 * between them, the frame bar under the window, and the overlays and the menu
 * over its corners.
 *
 * The markup mirrors what the carried stylesheet expects. It is not free to
 * change: the design and the classes that express it arrived together.
 */
function buildCard(hostEl, opts) {
    const doc = hostEl.ownerDocument;
    const el = (tag, className) => {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        return node;
    };

    const root = el("section", CARD);
    root.setAttribute("data-owner", opts.owner);

    const body = el("div", BODY);

    // The 3D window: a square, sized by the stylesheet (§ 8.2). The chain below
    // it fills that square end to end, so the drawing is sized by the module's
    // own square and never by the library's standalone default.
    const viewer = el("div", VIEWER);
    const wrap = el("div", "molviewer-window-wrap");
    const inner = el("div", "molviewer-window-inner");
    /* THE STYLESHEET'S SELECTORS ARE THE MARKUP CONTRACT. This chain is not free
     * to rearrange: `.viewer > .molviewer-window-frame` is what carries `height: 100%`,
     * and the stage and canvas below it are `flex: 1 1 auto` — so they take
     * their height from this flex column and from nothing else. Leave the card
     * out and the stage collapses to its content, the canvas is 45px tall, and
     * the drawing renders into a sliver with no error anywhere.
     *
     * It is also where the 3D-scene custom properties live (the cell wireframe's
     * colour and thickness), which the sealed layer reads by computed style —
     * they are declared here rather than on the page root so they stay concealed
     * to the module. */
    const frame = el("div", "molviewer-window-frame");
    const stage = el("div", "molviewer-window-stage");
    const canvas = el("div", "molviewer-window-canvas");

    // The busy cover — "Updating view…" — which only a rebuild raises (§ 10.5).
    const busy = el("div", "molviewer-window-busy");
    busy.hidden = true;
    busy.setAttribute("aria-live", "polite");
    busy.appendChild(el("div", "molviewer-window-busy-msg"));
    canvas.appendChild(busy);

    /* THE RAIL OF SWITCHES IS THE STAGE'S LEFT COLUMN (§ 1.1): "six icon buttons
     * sit down the left edge, ALWAYS OUTSIDE THE CANVAS, NEVER ON TOP OF THE
     * MOLECULE". That is why it is a sibling of the window and not an overlay
     * over it, and why the square's arithmetic pays for its width up in
     * `--molviewer-size-rail-width` rather than taking it out of the drawing.
     *
     * Built here and FILLED by ui.js, the same division as the frame bar: the
     * scaffold owns where things sit, the controls own what they do. */
    const rail = el("div", "molviewer-rail");
    stage.appendChild(rail);
    stage.appendChild(canvas);
    frame.appendChild(stage);
    inner.appendChild(frame);
    wrap.appendChild(inner);
    viewer.appendChild(wrap);

    // The fold handle: a slim full-height rail between the two (§ 8.3). The
    // CHEVRON is its own element because only the glyph rotates — rotating the
    // button would swap its width and height in the stacked layout and lay a
    // rail across the window.
    const fold = el("button", FOLD);
    fold.type = "button";
    fold.setAttribute("aria-label", "Fold the panel");
    fold.setAttribute("aria-expanded", "true");
    const chevron = el("span", "molviewer-panel-fold-chevron");
    chevron.textContent = "›";
    fold.appendChild(chevron);
    fold.addEventListener("click", () => {
        const folded = root.classList.toggle("molviewer-is-folded");
        fold.setAttribute("aria-expanded", folded ? "false" : "true");
    });

    const panel = el("div", PANEL);

    body.appendChild(viewer);
    body.appendChild(fold);
    body.appendChild(panel);
    root.appendChild(body);

    /* The frame bar rides the KNOB BAR, beside View and Export — created here
     * because it is part of the scaffold, but PLACED by whoever builds that bar
     * (ui.js), which is the only piece that knows where the row is.
     *
     * It belongs in that row and not under the card: the stylesheet sizes it
     * `flex: 1 1 15rem` precisely so it shares the row with the two menus and
     * WRAPS to its own full-width line when they leave it less than that. Hung
     * below the card instead, it spans the window AND the panel, is governed by
     * neither, and reads as a page control rather than this viewer's.
     *
     * It is the one piece not decided at mount: a viewer mounts before it has a
     * structure, and the bar appears once one with more than one frame is
     * loaded into it (§ 8). */
    const frameBar = el("div", "molviewer-frames-bar");
    frameBar.hidden = true;

    hostEl.appendChild(root);
    return { root, body, viewer, rail, canvas, panel, frameBar, fold };
}

/* The floor a host must respect, read from the stylesheet rather than written
 * here (§ 8.2) — so the layout stays decided in one place. */
function minimumWidth(root) {
    const view = root.ownerDocument && root.ownerDocument.defaultView;
    if (!view || typeof view.getComputedStyle !== "function") return 0;
    try {
        const raw = view.getComputedStyle(root)
            .getPropertyValue("--molviewer-size-card-min-width").trim();
        return parseFloat(raw) || 0;
    } catch (_) { return 0; }
}

/* A viewer that cannot fit renders a blank card with the error written in it,
 * rather than a half-built viewer (§ 8). */
function writeSizingError(card, available, floor, message) {
    try {
        card.body.textContent = "";
        // The modifier drops the card's min-width, so the message fits a host
        // too narrow to hold the viewer. Without it the blank card overflows the
        // host it just refused to render in — which is the thing being avoided.
        card.root.classList.add("molviewer-card--unmountable");
        const note = card.root.ownerDocument.createElement("p");
        note.className = "molviewer-embed-error";
        note.textContent = message || (
            "This viewer needs at least " + floor + "px of width; it was given "
            + available + "px.");
        card.body.appendChild(note);
        card.frameBar.hidden = true;
    } catch (_) {}
}
