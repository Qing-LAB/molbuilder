/* MolView — every control MolView itself draws.
 *
 * Contract: docs/web/molview.md § 8.5 (the controls, and what each one reads),
 *           § 1.1, § 9.5, § 11.4, § 11.6, § 6.4.
 * Owns:     the frame bar; the View menu; the Export menu; the selection panel
 *           and click-to-select; the measurement readout; the corner badge.
 * Called by: mount.js, which assembles them. Each control is a CALLER OF THE
 *           MODEL — the same doors a tab would use, with the same rules and the
 *           same read-only gate (§ 9.4) in front of them.
 *
 * NEVER:
 *   - talk to the 3D window directly (§ 7.3). Click-to-select arrives as data
 *     from below; everything a control does goes back out through the model.
 *   - hold truth of its own. A control that remembers the displayed frame, the
 *     count, or what is selected has given that fact a second home (§ 5.2).
 *   - hand a finished appearance downward (§ 9.2). A control asks for a change
 *     in the DATA or a SWITCH; what that looks like is worked out below it.
 *   - reach past the model to a store it was not given (§ 7 level 4).
 */
"use strict";

import { toDisplay } from "./_atom.js";


/**
 * Draw and wire every control, and return one teardown for all of them.
 *
 * @param card    the elements mount.js built (§ 8.1)
 * @param model   the data API — what almost everything here reads and writes
 * @param handle  the viewer, for PLAYBACK only (§ 8.5, § 9.2)
 * @param files   the door bytes leave through (§ 6.7, § 8) — `save(destination,
 *                filename, contents)`. MolView never reaches a file itself.
 */
export function mountControls(card, model, handle, files) {
    const doc = card.root.ownerDocument;
    const off = [];

    const rail = mountRail(doc, card, model, handle);
    off.push(rail.dispose);

    const frameBar = mountFrameBar(doc, card, model, handle);
    off.push(frameBar.dispose);

    const menus = mountMenus(doc, card, model, files);
    off.push(menus.dispose);

    const badge = mountBadge(doc, card, model);
    off.push(badge.dispose);

    const readout = mountReadout(doc, card, model);
    off.push(readout.dispose);

    const panel = mountPanel(doc, card, model);
    off.push(panel.dispose);

    return {
        dispose() {
            for (const fn of off.reverse()) { try { fn(); } catch (_) {} }
        },
    };
}


/* ══ The rail of switches (§ 1.1) ════════════════════════════════════════════
 *
 * "Six icon buttons sit down the left edge, always outside the canvas, never on
 * top of the molecule." Those are the toolbar switches, and the last clause is
 * the design rather than a detail: they are what a user reaches for WHILE
 * looking at the molecule, so they may not cover it. The card pays for the
 * column in its own arithmetic (`--rail-w`) instead of taking it out of the
 * drawing.
 *
 * Five switches and one action, in one list because they are one surface. Each
 * reads its lit state FROM THE STORE and never from what it last did (§ 5.2) —
 * `aria-pressed` is what the stylesheet draws it from, so a switch flipped
 * anywhere else lights the right button here with nothing to keep in step.
 *
 * The glyphs, the order and the wording are § 1.1's own.
 */
const RAIL = [
    { glyph: "⟲", name: "Reset view",
      title: "Re-fit the camera on the structure." },
    { glyph: "✚", name: "Show axes",          flag: "showAxis",
      title: "Show / hide axes" },
    { glyph: "#", name: "Show atom labels",   flag: "showIndex",
      title: "Show / hide atom labels" },
    { glyph: "➤", name: "Show force vectors", flag: "showForces",
      title: "Show / hide force vectors" },
    { glyph: "▦", name: "Show unit cell",     flag: "showCell",
      title: "Show / hide unit cell" },
    { glyph: "◉", name: "Show selected only", flag: "isolate",
      title: "Hide unselected atoms so the current selection stands out." },
];

function mountRail(doc, card, model, handle) {
    const rail = card.rail;
    const lit = {};

    for (const spec of RAIL) {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "mol-viewer-quick";
        button.textContent = spec.glyph;
        button.title = spec.title;
        // The glyph is decoration; the name is what the button IS, and it is
        // the only thing a screen reader has to go on.
        button.setAttribute("aria-label", spec.name);

        if (spec.flag) {
            button.setAttribute("aria-pressed", "false");
            button.addEventListener("click", () => {
                // Read the switch back from its one home rather than tracking
                // it here — a button that remembered its own state would be the
                // second answer that goes stale the moment anything else set it
                // (isolate, for one, turns itself off when the selection empties).
                const on = model.selection.getState()[spec.flag];
                model.selection.setSwitch(spec.flag, !on);
            });
            lit[spec.flag] = button;
        } else {
            // Reset is an action on the WINDOW, not a switch: there is no state
            // to light, which is why it carries no pressed attribute (§ 9.6 —
            // the camera is not held anywhere above the drawing).
            button.addEventListener("click", () => handle.resetView());
        }
        rail.appendChild(button);
    }

    const off = model.selection.subscribe((state) => {
        for (const flag of Object.keys(lit)) {
            lit[flag].setAttribute("aria-pressed", state[flag] ? "true" : "false");
        }
    });

    return {
        dispose() { off(); try { rail.textContent = ""; } catch (_) {} },
    };
}


/* ══ The frame bar (§ 8.5, § 6.4) ════════════════════════════════════════════
 *
 * ONE CONTROL WITH TWO OWNERS, and it reads each from where the fact lives: the
 * frame and the count from the MODEL (§ 6.4), play, pause and loop from the
 * HANDLE (§ 9.2). A bar that took the frame from the handle would be reading a
 * mirror — and § 9.2 retires exactly those forwarded reads.
 *
 * It appears once there is more than one frame (§ 8), and it holds nothing: the
 * slider's position is read from the model on every change, never remembered.
 */
function mountFrameBar(doc, card, model, handle) {
    const bar = card.frameBar;
    const el = (tag, className) => {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        return node;
    };

    const prev = el("button", "mvf-step");
    prev.type = "button"; prev.textContent = "‹";
    prev.setAttribute("aria-label", "Previous frame");

    const playBtn = el("button", "mvf-play");
    playBtn.type = "button"; playBtn.textContent = "▶";
    playBtn.setAttribute("aria-label", "Play");

    const next = el("button", "mvf-step");
    next.type = "button"; next.textContent = "›";
    next.setAttribute("aria-label", "Next frame");

    const transport = el("div", "mvf-transport");
    transport.appendChild(prev);
    transport.appendChild(playBtn);
    transport.appendChild(next);

    const slider = el("input", "mvf-slider");
    slider.type = "range";
    slider.min = "0";
    slider.step = "1";
    slider.setAttribute("aria-label", "Frame");

    const counter = el("span", "mvf-counter");

    const loopWrap = el("label", "mvf-loop");
    const loopBox = doc.createElement("input");
    loopBox.type = "checkbox";
    loopBox.checked = handle.getLoop();
    loopWrap.appendChild(loopBox);
    loopWrap.appendChild(doc.createTextNode(" loop"));

    bar.appendChild(transport);
    bar.appendChild(slider);
    bar.appendChild(counter);
    bar.appendChild(loopWrap);

    // Every move goes through the model's one write, whatever pressed it — so
    // the slider, the arrows and playback are indistinguishable downstream.
    const step = (delta) => {
        if (handle.isPlaying()) { handle.pause(); reflect(); }
        model.setCurrentFrame(model.currentFrame() + delta);
    };

    prev.addEventListener("click", () => step(-1));
    next.addEventListener("click", () => step(+1));
    slider.addEventListener("input", (e) => {
        model.setCurrentFrame(Number(e.target.value) || 0);
    });
    playBtn.addEventListener("click", () => {
        if (handle.isPlaying()) handle.pause(); else handle.play();
        reflect();
    });
    loopBox.addEventListener("change", (e) => handle.setLoop(!!e.target.checked));

    /* Read everything back from where it lives. Nothing here is cached: the bar
     * is a view of the model, and a remembered count is how a slider comes to
     * offer a frame that nothing can draw (§ 6.4). */
    function reflect() {
        const count = model.frameCount();
        // The bar is the one piece not decided at mount: it appears once a
        // structure with more than one frame is loaded (§ 8).
        bar.hidden = count < 2;
        if (count < 2) return;
        const at = model.currentFrame();
        slider.max = String(count - 1);
        slider.value = String(at);
        // 1-based on screen, through the one translation (§ 11.5).
        counter.textContent = toDisplay(at) + " / " + count;
        const playing = handle.isPlaying();
        playBtn.textContent = playing ? "⏸" : "▶";
        playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
    }

    const offFrame = model.onFrameChange(reflect);
    const offData = model.subscribe(reflect);
    reflect();

    return {
        dispose() {
            offFrame(); offData();
            try { bar.textContent = ""; bar.hidden = true; } catch (_) {}
        },
    };
}


/* ══ The View and Export menus (§ 8.5, § 11.4) ═══════════════════════════════
 *
 * MolView's own menu surface, over the window's corner. `<details>`/`<summary>`
 * gives open, close and keyboard access for free; mutual exclusion and PLACING
 * THE POPOVER are what have to be wired.
 */

/* Where an open popover sits: how far under its trigger it hangs, and how close
 * to the window's edge it may come. BOTH LIVE IN THE STYLESHEET — `--mv-menu-gap`
 * and `--mv-menu-margin` — with every other distance the module uses, and this
 * asks for them rather than restating them. It is the same rule mount.js follows
 * for the sizing floor and the sealed layer follows for the scene constants: a
 * number a rule AND a script both need has one home, and it is the stylesheet.
 *
 * The pair below is the last resort for a page with no stylesheet at all (a node
 * test), not a second opinion about the design. */
const MENU_PLACEMENT = { gap: 4, margin: 8 };

function placementFor(el) {
    const view = el && el.ownerDocument && el.ownerDocument.defaultView;
    if (!view || typeof view.getComputedStyle !== "function") return MENU_PLACEMENT;
    const style = view.getComputedStyle(el);
    const read = (name, fallback) => {
        const px = parseFloat(style.getPropertyValue(name));
        return Number.isFinite(px) ? px : fallback;
    };
    return {
        gap:    read("--mv-menu-gap",    MENU_PLACEMENT.gap),
        margin: read("--mv-menu-margin", MENU_PLACEMENT.margin),
    };
}

function mountMenus(doc, card, model, files) {
    const bar = doc.createElement("div");
    bar.className = "mol-viewer-knobs";
    bar.setAttribute("role", "toolbar");
    bar.setAttribute("aria-label", "Viewer controls");

    const view = buildViewMenu(doc, model);
    const exportMenu = buildExportMenu(doc, model, files);
    bar.appendChild(view.root);
    bar.appendChild(exportMenu.root);
    // The playback bar is the third thing in this row (§ 8.5). The scaffold owns
    // the element; this is the only place that knows where the row is.
    if (card.frameBar) bar.appendChild(card.frameBar);

    /* ── Placing an open popover ───────────────────────────────────────────
     *
     * The stylesheet gives the menu body `position: fixed` and parks it at
     * -9999px until something measures where it goes. That is not decoration:
     * these menus sit OVER the 3D window, and the window clips its own contents
     * (`overflow: hidden`, so the drawing stays inside the card's rounded
     * corners) — an absolutely positioned popover is cut off at the canvas edge,
     * usually to nothing. Fixed escapes every clipping ancestor, and the price
     * of escaping them is that it has no anchor left: the coordinates have to be
     * measured against the trigger.
     *
     * Ship the stylesheet without this and the menu WORKS and shows nothing: it
     * opens, the trigger takes its open state, and the panel is on screen the
     * whole time at -9999px. */
    const menus = [view, exportMenu];

    function place(menu) {
        const win = doc.defaultView;
        if (!win || !menu.summary || !menu.body) return;
        // Read when the menu opens — the moment the placement is decided — and
        // reused while it stays open, so following a scroll costs no style
        // recalculation per frame.
        const step = menu.placement || (menu.placement = placementFor(card.root));
        const anchor = menu.summary.getBoundingClientRect();
        // Measured where it will be READ, not where it was parked: a popover at
        // -9999px reports the same size, but only if it has been laid out at a
        // sane place first do the clamps below have anything true to work with.
        menu.body.style.top  = (anchor.bottom + step.gap) + "px";
        menu.body.style.left = anchor.left + "px";
        const size = menu.body.getBoundingClientRect();
        const page = doc.documentElement;
        const vw = win.innerWidth  || (page && page.clientWidth)  || 0;
        const vh = win.innerHeight || (page && page.clientHeight) || 0;

        // It hangs from the trigger's left edge and is PULLED BACK inside the
        // window rather than allowed to leave it — a menu half off the screen is
        // a menu with items nobody can reach.
        let left = anchor.left;
        if (vw) left = Math.min(left, vw - step.margin - size.width);
        left = Math.max(left, step.margin);
        menu.body.style.left = left + "px";

        // No room below: open upwards instead, but only if there is room there.
        if (vh && anchor.bottom + step.gap + size.height > vh - step.margin) {
            const above = anchor.top - step.gap - size.height;
            if (above >= step.margin) menu.body.style.top = above + "px";
        }
    }

    // One open at a time, and the one that opens is placed.
    for (const menu of menus) {
        menu.root.addEventListener("toggle", () => {
            if (!menu.root.open) return;
            for (const other of menus) if (other !== menu) other.root.open = false;
            place(menu);
        });
    }

    /* An open popover is fixed to the VIEWPORT, so anything that moves its
     * trigger relative to the viewport — a scroll, a resize — moves the window
     * out from under it. It FOLLOWS rather than closing: shutting a menu the
     * user did not shut is its own kind of wrong. Capture, because the scroll
     * that matters is usually inside an ancestor, not on the window. */
    const win = doc.defaultView;
    const follow = () => {
        for (const menu of menus) if (menu.root.open) place(menu);
    };
    // A click anywhere outside closes it. Without this the trigger is the only
    // way out, which is a poor answer once the menu has drifted off-focus.
    const dismiss = (event) => {
        for (const menu of menus) {
            if (menu.root.open && !menu.root.contains(event.target)) {
                menu.root.open = false;
            }
        }
    };
    if (win) {
        win.addEventListener("scroll", follow, { passive: true, capture: true });
        win.addEventListener("resize", follow, { passive: true });
        doc.addEventListener("click", dismiss, true);
    }

    card.canvas.appendChild(bar);

    return {
        dispose() {
            if (win) {
                win.removeEventListener("scroll", follow, { capture: true });
                win.removeEventListener("resize", follow);
                doc.removeEventListener("click", dismiss, true);
            }
            view.dispose(); exportMenu.dispose();
            try { bar.remove(); } catch (_) {}
        },
    };
}

/* THE VIEW MENU WRITES TO TWO STORES, and § 9.6's question is what sorts them:
 * does working out what a frame contains require reading this?
 *
 *   atom numbers, the cell, the axes  -> YES, so they are SWITCHES
 *   style, background, projection     -> NO, so they go straight to the drawing
 *
 * The menu is one piece of UI. That does not make its contents one kind of
 * thing, and putting a drawing setting in the switch store would make every
 * style change re-derive four hundred frames.
 */
function buildViewMenu(doc, model) {
    const root = doc.createElement("details");
    root.className = "mol-viewer-menu";
    const summary = doc.createElement("summary");
    summary.textContent = "View";
    root.appendChild(summary);

    // The menu's contents live in a body, which is what the stylesheet lays out.
    const body = doc.createElement("div");
    body.className = "mol-viewer-menu-body";
    root.appendChild(body);

    const section = (heading) => {
        const wrap = doc.createElement("div");
        wrap.className = "mol-viewer-menu-section";
        const label = doc.createElement("div");
        label.className = "mol-viewer-menu-heading";
        label.textContent = heading;
        wrap.appendChild(label);
        body.appendChild(wrap);
        return wrap;
    };

    const offs = [];

    /* THE SWITCHES ARE NOT IN HERE. They are the rail (§ 1.1) — six icon buttons
     * down the left edge, one press each, visible without opening anything.
     * Repeating them in this menu would put one switch behind two controls: not
     * two homes for the FACT (both would write the one store), but two places a
     * user has to learn, and two things to keep looking the same. The rail is
     * the validated design; the menu keeps what the rail has no room for.
     *
     * What is left here is § 9.6's other column — the settings that change HOW
     * THE SAME FRAME IS PAINTED and that the frame calculation never reads.
     */
    const drawn = section("Draw as");
    const repRow = doc.createElement("div");
    repRow.className = "mol-viewer-rep-row";
    const REPS = [["stick", "Sticks"], ["ball-and-stick", "Ball & stick"],
                  ["sphere", "Spheres"], ["line", "Lines"]];
    const repButtons = {};
    for (const [value, label] of REPS) {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "mol-viewer-rep-btn";
        button.textContent = label;
        button.addEventListener("click", () => model.view.set("style", value));
        repRow.appendChild(button);
        repButtons[value] = button;
    }
    drawn.appendChild(repRow);

    /* THE RADIUS SLIDER (§ 1.1: "a radius slider from 0.2 to 2.5 that scales
     * stick thickness / sphere size / line width"). It sits with the reps
     * because it scales whichever one is showing, which is what the carried
     * stylesheet lays out — `.mol-viewer-radius-row` was styled all along and
     * had nothing to style.
     *
     * The label WRAPS the input rather than pairing with it through `for`/`id`.
     * § 5.6 puts two viewers on one page and an id is document-global, so the
     * second mount would duplicate it and both labels would point at the first
     * viewer's slider. */
    const radiusRow = doc.createElement("div");
    radiusRow.className = "mol-viewer-radius-row";
    const radiusLabel = doc.createElement("label");
    radiusLabel.textContent = "Radius";
    /* NO class on the input or the output. The stylesheet reaches them by
     * ELEMENT inside the row — `.mol-viewer-radius-row > input[type="range"]`
     * and `> output` — so a class here would be one the design never defines.
     * The old code set `mol-viewer-radius` and `mol-viewer-radius-out` and the
     * stylesheet used neither. */
    const radius = doc.createElement("input");
    radius.type = "range";
    radius.min = "0.2";
    radius.max = "2.5";
    radius.step = "0.05";
    radius.setAttribute("aria-label", "Atom radius scale");
    const radiusOut = doc.createElement("output");
    radius.addEventListener("input", (e) => {
        model.view.set("radius", Number(e.target.value));
    });
    radiusRow.appendChild(radiusLabel);
    radiusRow.appendChild(radius);
    radiusRow.appendChild(radiusOut);
    drawn.appendChild(radiusRow);

    /* THE BACKGROUND (§ 1.1: "a background colour with preset swatches plus a
     * picker. One preset is transparent — choose it before exporting a picture
     * to drop onto a slide.").
     *
     * The presets are the ones the old design shipped: the card's own dark, a
     * white for print, and transparent. `transparent` is a value like any other
     * here — what it MEANS to the drawing is the sealed layer's business (§ 9.8),
     * and this control neither knows nor needs to. */
    const BACKGROUNDS = [
        ["#1d2128",     "Dark"],
        ["#ffffff",     "White"],
        ["transparent", "Transparent"],
    ];
    const background = section("Background");
    const bgRow = doc.createElement("div");
    bgRow.className = "mol-viewer-bg-row";
    const swatches = {};
    for (const [value, name] of BACKGROUNDS) {
        const swatch = doc.createElement("button");
        swatch.type = "button";
        swatch.className = "mol-viewer-bg-swatch";
        swatch.setAttribute("aria-label", name + " background");
        swatch.title = name;
        if (value === "transparent") {
            // The stylesheet draws the checkerboard from this class; a colour
            // set inline would paint over it.
            swatch.classList.add("is-transparent");
        } else {
            swatch.style.background = value;
        }
        swatch.addEventListener("click", () => model.view.set("background", value));
        bgRow.appendChild(swatch);
        swatches[value] = swatch;
    }

    /* The picker. A styled label wraps the native colour input so the OS picker
     * still opens on click while the visible chip matches the preset row — the
     * stylesheet sizes the input to fill it, so the whole chip is the target. */
    const custom = doc.createElement("label");
    custom.className = "mol-viewer-bg-custom";
    custom.setAttribute("aria-label", "Custom background colour");
    custom.title = "Custom colour";
    const picker = doc.createElement("input");
    picker.type = "color";
    picker.addEventListener("input", (e) => {
        model.view.set("background", e.target.value);
    });
    custom.appendChild(picker);
    bgRow.appendChild(custom);
    background.appendChild(bgRow);

    const projection = doc.createElement("button");
    projection.type = "button";
    projection.className = "mol-viewer-toggle";
    projection.textContent = "Orthographic";
    projection.setAttribute("aria-pressed", "false");
    projection.addEventListener("click", () => {
        const on = projection.getAttribute("aria-pressed") === "true";
        model.view.set("orthographic", !on);
    });
    drawn.appendChild(projection);

    /* EVERY CONTROL READS ITS STATE BACK FROM `view` (§ 8.5), never from what it
     * last did — so a setting changed anywhere else lights the right control
     * here with nothing to keep in step. */
    offs.push(model.view.subscribe((settings) => {
        for (const [value] of REPS) {
            repButtons[value].classList.toggle("is-active", settings.style === value);
        }
        radius.value = String(settings.radius);
        radiusOut.textContent = Number(settings.radius).toFixed(2);
        for (const [value] of BACKGROUNDS) {
            swatches[value].classList.toggle("is-active",
                                             settings.background === value);
        }
        // `background: null` is not a colour — it is "the window's own ground",
        // which the drawing resolves (§ 9.6). No swatch is lit until the user
        // has chosen one, and the picker only shows a colour once it IS one.
        if (typeof settings.background === "string"
            && settings.background !== "transparent") {
            picker.value = settings.background;
        }
        projection.setAttribute("aria-pressed",
                                settings.orthographic ? "true" : "false");
    }));

    // The store hands nothing over on subscribing (§ 9.6 is a plain
    // change-and-subscribe), so the first paint is taken here.
    radius.value = String(model.view.get().radius);
    radiusOut.textContent = Number(model.view.get().radius).toFixed(2);

    // The trigger and the popover go back with the menu: whoever places it needs
    // both, and handing back what was just built beats searching the DOM for it.
    return {
        root, summary, body,
        dispose() { for (const fn of offs) { try { fn(); } catch (_) {} } },
    };
}

/* THE EXPORT MENU (§ 11.4). Every export enters at MolView, and what to export
 * and where it goes is decided HERE — above the model — because an export
 * carries a decision, and a decision made in the wrong place is exactly how the
 * sidecar came to be dropped.
 *
 * The data export is the only one that is the truth, and it is read from the
 * master copy at the frame the user chose (§ 11.3). A picture is a render.
 *
 * BYTES LEAVE THROUGH THE `files` DOOR, never through code written here (§ 6.7).
 * MolView builds no download link, makes no object URL, names no filesystem API
 * and calls no file endpoint — it produces the bytes and names the destination.
 * Save-to-project and download differ ONLY in that destination (§ 11.3), so they
 * are one call with an argument, not two paths to keep in step.
 */
function buildExportMenu(doc, model, files) {
    const root = doc.createElement("details");
    root.className = "mol-viewer-menu";
    const summary = doc.createElement("summary");
    summary.textContent = "Export";
    root.appendChild(summary);

    const body = doc.createElement("div");
    body.className = "mol-viewer-menu-body";
    root.appendChild(body);
    const section = doc.createElement("div");
    section.className = "mol-viewer-export-section";
    const label = doc.createElement("div");
    label.className = "mol-viewer-export-section-label";
    label.textContent = "Structure";
    section.appendChild(label);
    const row = doc.createElement("div");
    row.className = "mol-viewer-export-row";
    section.appendChild(row);
    body.appendChild(section);

    const item = (text, onClick) => {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "mol-viewer-export-btn";
        button.textContent = text;
        button.addEventListener("click", onClick);
        row.appendChild(button);
        return button;
    };

    /* WHAT LEAVES, AND WHERE IT GOES — and nothing about how it becomes bytes.
     * MolView hands over the STRUCTURE and names a destination; the door turns
     * it into the pair, through the server's one generator, so a project save
     * and a download cannot produce different bytes (§ 11.3, § 11.7).
     *
     * This used to assemble the bytes here: a hand-written `.xyz` and a
     * `JSON.stringify` of the sidecar. That is a second writer in the browser,
     * and both halves had already drifted from Python's -- the coordinate
     * document in its decimals, the sidecar in the version key that makes one
     * loadable at all. */
    function send(destination) {
        const file = model.exportFile();
        // It REFUSES rather than exporting a structure it cannot vouch for
        // (§ 9.3), and a refusal is not something to paper over with an empty
        // file.
        if (!file || !files || typeof files.save !== "function") return;
        files.save(destination, defaultStem(model, file.name), file.structure);
    }

    item("Save to project", () => send("project"));
    item("Download",        () => send("download"));

    return {
        root, summary, body,
        dispose() { try { root.remove(); } catch (_) {} },
    };
}

/* THE DEFAULT NAME (§ 11.4's `wire_frame50.xyz`), which is two facts joined:
 * WHAT it came from, and WHICH FRAME. The first is the structure's — the model
 * kept it from the load — and the second is this menu's, because the menu is
 * what knows the export is one frame out of many.
 *
 * A single-frame export out of a trajectory names its frame so the file says
 * which one it is without anyone having to remember; a static structure gets no
 * suffix, there being nothing to disambiguate. A structure that arrived with no
 * name — pasted text — falls back to a generic stem rather than borrowing one. */
function defaultStem(model, source) {
    const base = source || "structure";
    return model.frameCount() > 1
        ? base + "_frame" + toDisplay(model.currentFrame())
        : base;
}


/* ══ The unsaved-changes badge (§ 11.2) ══════════════════════════════════════
 *
 * "Not bookkeeping": it shows in the corner of the 3D window so "there is work
 * here that is not on the sequence yet" is visible without opening a menu.
 * Without it, an explicit-save history silently loses work a user assumed was
 * being kept.
 */
function mountBadge(doc, card, model) {
    const badge = doc.createElement("div");
    /* BOTTOM-right, not top: the top band of the window is the chrome row (View,
     * Export, the frame bar), and an overlay sitting on it covers controls. The
     * two corner overlays take the bottom — the measurement on the left, this on
     * the right — so the two bands never contend. § 11.2 asks for "the corner of
     * the 3D window" and does not say which. */
    badge.className = "molview-overlay molview-overlay--bottom-right molview-overlay--warn";
    badge.textContent = "Unsaved changes";
    badge.hidden = true;
    card.canvas.appendChild(badge);

    const off = model.subscribe(() => { badge.hidden = !model.uncommitted; });
    badge.hidden = !model.uncommitted;

    return {
        dispose() { off(); try { badge.remove(); } catch (_) {} },
    };
}


/* ══ The measurement readout (§ 11.6, § 8.5) ═════════════════════════════════
 *
 * Its own layer, not part of drawing. It takes atom numbers from the panel's
 * selection and coordinates from the MASTER COPY at the current frame — which is
 * exactly why it stays correct while a trajectory plays and under isolate, where
 * the drawn numbering no longer matches the real one.
 *
 * THE VERTEX OF AN ANGLE IS THE ATOM PICKED SECOND, not the middle one by
 * number. That is why the pick order is carried in the snapshot (§ 8.4) rather
 * than reconstructed from the sorted selection.
 */
function mountReadout(doc, card, model) {
    const readout = doc.createElement("div");
    readout.className = "molview-overlay molview-overlay--bottom-left molview-overlay--info";
    readout.hidden = true;
    card.canvas.appendChild(readout);

    function show() {
        const state = model.selection.getState();
        const frame = model.getFrameAllAtoms(model.currentFrame());
        const picked = orderedForMeasurement(state, frame);
        if (!frame || !picked.length || picked.length > 3) {
            readout.hidden = true;
            return;
        }
        const at = (i) => frame[i];
        let text = "";
        if (picked.length === 1) {
            const p = at(picked[0]);
            if (p) text = "#" + toDisplay(picked[0]) + "  "
                        + p.map((v) => v.toFixed(3)).join(", ");
        } else if (picked.length === 2) {
            const a = at(picked[0]), b = at(picked[1]);
            if (a && b) text = distance(a, b).toFixed(3) + " Å";
        } else {
            // picked[1] is the vertex: the atom clicked SECOND.
            const a = at(picked[0]), v = at(picked[1]), c = at(picked[2]);
            if (a && v && c) text = angle(a, v, c).toFixed(1) + "°";
        }
        readout.textContent = text;
        readout.hidden = text === "";
    }

    const offSel = model.selection.subscribe(show);
    const offFrame = model.onFrameChange(show);   // stays right while it plays
    const offData = model.subscribe(show);
    show();

    return {
        dispose() {
            offSel(); offFrame(); offData();
            try { readout.remove(); } catch (_) {}
        },
    };
}

/* WHICH ATOMS, IN WHICH ORDER (§ 11.6).
 *
 * The vertex of an angle is THE ATOM PICKED SECOND — a chemist's convention that
 * only the pick order can carry (§ 8.4). But a selection can arrive with no pick
 * order at all: All, Invert, an applied filter and a restored session are not
 * clicks, and the store now says so by handing over an empty trail instead of
 * inventing one out of the sorted selection.
 *
 * With no trail, the vertex comes from GEOMETRY — the atom closest to the other
 * two, which for a bonded triple is the middle one. That is a guess and it is
 * labelled as one here; what it replaces was also a guess, made silently, and
 * dressed up as the user's own choice.
 */
function orderedForMeasurement(state, frame) {
    const picked = state.selection;
    const trail = state.pickOrder;
    if (trail.length === picked.length) return trail;      // a real click trail
    if (picked.length !== 3 || !frame) return picked;      // no vertex to find
    return byGeometry(picked, frame);
}

function byGeometry(atoms, frame) {
    if (atoms.some((i) => !frame[i])) return atoms;
    const spread = (i) => atoms.reduce(
        (total, other) => total + (other === i ? 0 : distance(frame[i], frame[other])), 0);
    let vertex = atoms[0];
    for (const atom of atoms) if (spread(atom) < spread(vertex)) vertex = atom;
    const ends = atoms.filter((i) => i !== vertex);
    return [ends[0], vertex, ends[1]];
}

function distance(a, b) {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function angle(a, vertex, c) {
    const u = [a[0] - vertex[0], a[1] - vertex[1], a[2] - vertex[2]];
    const w = [c[0] - vertex[0], c[1] - vertex[1], c[2] - vertex[2]];
    const dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    const mag = Math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
              * Math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2);
    if (!mag) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
}


/* ══ The selection panel (§ 8.4, § 9.5) ══════════════════════════════════════
 *
 * A READER of the selection store. It is handed one settled snapshot and draws
 * from that — never from a dozen separate reads, and never from a copy of its
 * own (§ 8.4). What it writes goes back through the store, so a change made here
 * meets the same rules as one made anywhere else (§ 9.4).
 *
 * Two pages, switched by a tab bar: Selection and Cell. Switching pages never
 * resizes the card (§ 8.1) — the stylesheet gives the panel the viewer's extent,
 * so its height does not depend on what is inside it.
 *
 * The class names are the carried stylesheet's, so this function does not get to
 * choose them.
 */
function mountPanel(doc, card, model) {
    const el = (tag, className) => {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        return node;
    };

    const root = el("div", "selection-card");

    /* Every radio group in this panel is named after the OWNER (§ 5.6). The name
     * is what browsers group radios by, and it is global to the document — so
     * two viewers on one page with a fixed name are ONE group, and choosing a
     * page or an editor in the second silently un-chooses it in the first. */
    const owner = card.root.getAttribute("data-owner") || "molview";

    /* ── The two pages, and the tab bar that switches them (§ 8.1) ───────── */
    const header = el("div", "card-header");
    const tabs = el("div", "panel-page-switch selection-header-tabs");
    const pages = {};
    const tabInputs = {};
    for (const [key, label] of [["selection", "Selection"], ["cell", "Cell"]]) {
        /* A TAB IS A RADIO INSIDE A LABEL, not a button — the carried stylesheet
         * draws the chosen tab from `:has(input:checked)` (accent text, accent
         * underline) and its type from `.panel-page-option > span`. Built as a
         * bare `<button>` with the text on it, NEITHER rule can match: the
         * switch renders as two words in the browser's default button font with
         * no indication of which page you are on. The markup is as much the
         * stylesheet's contract as the class name is. */
        const tab = el("label", "panel-page-option");
        const radio = doc.createElement("input");
        radio.type = "radio";
        radio.name = "molview-page-" + owner;
        radio.addEventListener("change", () => showPage(key));
        const text = el("span");
        text.textContent = label;
        tab.appendChild(radio);
        tab.appendChild(text);
        tabs.appendChild(tab);
        tabInputs[key] = radio;

        const page = el("div", "panel-page");
        /* THE ROLE IS PART OF THE MARKUP CONTRACT, exactly like the class
         * (§ 8.1). The stylesheet singles the Cell page out by it — that page is
         * all read-only text, so it sizes to its content and then scrolls, while
         * the Selection page holds action buttons that must stay on screen and
         * so fills instead. Ship the page without it and it silently takes the
         * other page's behaviour.
         *
         * An ATTRIBUTE and not an id: § 5.6 puts two viewers on one page, and an
         * id is document-global, so the second mount would duplicate it. What
         * the stylesheet needs is what this element IS, never which one it is. */
        page.setAttribute("data-page", key);
        page.hidden = key !== "selection";
        pages[key] = page;
    }
    header.appendChild(tabs);
    root.appendChild(header);
    root.appendChild(pages.selection);
    root.appendChild(pages.cell);

    function showPage(which) {
        for (const key of Object.keys(pages)) {
            // The [hidden] attribute, not a class: the stylesheet's rules are
            // written `:not([hidden])` precisely so `display: flex` cannot
            // override it.
            pages[key].hidden = key !== which;
            // The tab is SET here rather than left to the click that opened the
            // page, so a page shown from anywhere else — the first one, or a
            // later caller — still lights the tab that goes with it.
            tabInputs[key].checked = key === which;
        }
    }
    showPage("selection");

    /* ── Which editor: click, or filter (§ 9.5) ──────────────────────────── */
    //
    // Two editors of ONE selection. Switching between them does not touch what
    // is selected — the panel redraws, the truth does not move.
    const mode = el("div", "selection-mode");
    const modeInputs = {};
    for (const [key, label] of [["click", "Click"], ["filter", "Filter"]]) {
        const option = el("label", "selection-mode-option");
        const radio = doc.createElement("input");
        radio.type = "radio";
        radio.name = "molview-mode-" + owner;
        radio.addEventListener("change", () => model.selection.setEditor(key));
        const text = el("span");
        text.textContent = label;
        option.appendChild(radio);
        option.appendChild(text);
        mode.appendChild(option);
        modeInputs[key] = radio;
    }
    pages.selection.appendChild(mode);

    /* ISOLATE IS NOT HERE. "Show selected only" is the rail's `◉` (§ 1.1) — one
     * switch, one control. The panel carried a checkbox for it as well, so the
     * same switch had two places to learn and two things to keep looking alike;
     * the old design removed the panel's copy for exactly that reason when the
     * switch moved to the rail, and this follows it. The switch itself is
     * unchanged: it hides atoms from the DRAWING, the master copy still has all
     * of them, and a read-only viewer isolates freely (§ 9.4).
     */

    /* ── The click page: the atom list ──────────────────────────────────── */
    const clickSection = el("div", "selection-click-section");
    /* `data-fill` is the role that makes a section TAKE THE LEFTOVER HEIGHT and
     * hold the page's single scroll region, so the atom list grows to the
     * panel's bottom and the buttons beneath it line up with the bottom of the
     * 3D window (§ 8.2's shared extent). Without it the section sizes to its
     * content and the page collapses upward. A page has exactly one. */
    clickSection.setAttribute("data-fill", "");
    const count = el("div", "selection-count");
    const listWrap = el("div", "selection-list-wrap");
    const list = el("table", "selection-atom-table");
    listWrap.appendChild(list);
    clickSection.appendChild(count);
    clickSection.appendChild(listWrap);
    pages.selection.appendChild(clickSection);

    /* ── The filter page: rows, edited one at a time (§ 8.4) ─────────────── */
    const filterSection = el("div", "selection-filter-section");
    filterSection.setAttribute("data-fill", "");
    const rows = el("div", "selection-filter-rows");
    filterSection.appendChild(rows);

    const actions = el("div", "selection-filter-actions");
    const addRow = el("button", "selection-add-filter-row");
    addRow.type = "button";
    addRow.textContent = "+ Add filter";
    addRow.addEventListener("click", () => model.selection.addFilter());
    actions.appendChild(addRow);
    filterSection.appendChild(actions);

    const footer = el("div", "selection-filter-footer");
    const combineRow = el("div", "selection-combinator-row");
    const combine = el("select", "selection-combinator-select");
    for (const [value, label] of [["and", "Match all"], ["or", "Match any"]]) {
        const option = doc.createElement("option");
        option.value = value; option.textContent = label;
        combine.appendChild(option);
    }
    combine.addEventListener("change", (e) => {
        model.selection.setCombinator(e.target.value);
    });
    combineRow.appendChild(combine);
    const apply = el("button", "selection-apply-filter-btn");
    apply.type = "button";
    apply.textContent = "Apply filter";
    // Filter mode composes a query the user EXPLICITLY applies, and applying it
    // REPLACES the selection (§ 9.5). Nothing happens while they type.
    apply.addEventListener("click", () => model.selection.applyFilter());
    footer.appendChild(combineRow);
    footer.appendChild(apply);
    filterSection.appendChild(footer);
    pages.selection.appendChild(filterSection);

    /* ── The click operations, and the label block ───────────────────────── */
    // A visual divider between the SELECT-side controls above and the ACT-side
    // controls below — the stylesheet's own separation of the two jobs.
    pages.selection.appendChild(el("div", "selection-divider"));

    // Every action button takes one of the five classes the stylesheet's shared
    // base is written over. A sixth name would get no base at all: no padding,
    // no font, no baseline — which is exactly how one button ends up a different
    // size from the two beside it.
    const actionsRow = el("div", "selection-actions-row");
    for (const [label, className, run] of [
        ["Clear",  "selection-clear-btn", () => model.selection.clear()],
        ["Invert", "selection-add-btn",   () => model.selection.invert(atomCount())],
        ["All",    "selection-add-btn",   () => model.selection.all(atomCount())],
    ]) {
        const button = el("button", className);
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", run);
        actionsRow.appendChild(button);
    }
    pages.selection.appendChild(actionsRow);

    // Tagging is an EDIT (§ 9.4): a label becomes part of what an atom is and
    // travels to the calculation, so it is frozen along with the rest. MolView
    // hides the controls the gate would swallow, because a button that silently
    // does nothing is a bad answer for a user — the gate is the contract, the
    // hiding is courtesy.
    //
    // The block stacks a TARGET row over a VERB row, so the buttons do not
    // compete for width with the input. The three verbs are one action with
    // three set operations, and THE COLOUR ENCODES WHICH: blue replaces, green
    // unions, red subtracts. That is the stylesheet's semantics, not a palette
    // choice — a verb wearing the wrong one would say the wrong thing.
    const assign = el("div", "selection-assign");

    const targetRow = el("div", "selection-target-row");
    const target = el("input", "selection-new-label");
    target.type = "text";
    target.placeholder = "Label name";
    targetRow.appendChild(target);
    assign.appendChild(targetRow);

    const verbRow = el("div", "selection-verb-row");
    const named = () => String(target.value || "").trim();
    for (const [label, className, verb] of [
        ["Assign",   "selection-assign-btn",        "replace"],
        ["+ Add",    "selection-add-target-btn",    "add"],
        ["− Remove", "selection-remove-target-btn", "remove"],
    ]) {
        const button = el("button", className);
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", () => {
            const name = named();
            if (!name) return;
            model.selection.writeLabel(name, verb);
        });
        verbRow.appendChild(button);
    }
    assign.appendChild(verbRow);
    pages.selection.appendChild(assign);

    /* ── The Cell page: read-only (§ 8.1) ────────────────────────────────── */
    const cellReadout = el("dl", "cell-readout");
    pages.cell.appendChild(cellReadout);

    function atomCount() {
        const atoms = model.getAtoms();
        return atoms ? atoms.length : 0;
    }

    /* ── Drawing, from ONE snapshot (§ 8.4) ──────────────────────────────── */
    function draw(state) {
        // The editor decides which body shows; the selection is untouched by it.
        clickSection.hidden = state.mode !== "click";
        filterSection.hidden = state.mode !== "filter";
        for (const key of Object.keys(modeInputs)) {
            modeInputs[key].checked = state.mode === key;
        }
        combine.value = state.combinator;

        drawList(state);
        drawRows(state);
        // A read-only viewer does not show the controls the gate would swallow.
        assign.hidden = model.mode === "readonly";
    }

    function drawList(state) {
        const atoms = model.getAtoms();
        count.textContent = atoms
            ? state.selection.length + " of " + atoms.length + " selected"
            : "";
        list.textContent = "";
        if (!atoms) return;
        const picked = new Set(state.selection);
        for (const atom of atoms) {
            const row = doc.createElement("tr");
            if (picked.has(atom.index)) row.className = "is-selected";
            const number = doc.createElement("td");
            // 1-based on screen, through the one translation (§ 11.5).
            number.textContent = String(toDisplay(atom.index));
            const element = doc.createElement("td");
            element.textContent = atom.element;
            const labels = doc.createElement("td");
            for (const name of atom.labels) {
                const tag = el("span", "selection-tag tag-region");
                tag.textContent = name;
                labels.appendChild(tag);
            }
            row.appendChild(number);
            row.appendChild(element);
            row.appendChild(labels);
            row.addEventListener("click", () => model.selection.toggle(atom.index));
            list.appendChild(row);
        }
    }

    function drawRows(state) {
        rows.textContent = "";
        if (!state.filters.length) {
            const empty = el("div", "selection-filter-empty");
            empty.textContent = "No filters yet.";
            rows.appendChild(empty);
            return;
        }
        state.filters.forEach((filter, at) => {
            const row = el("div", "selection-filter-row");

            const kind = el("select", "selection-filter-kind");
            // Which rows are worth offering is read from the structure, not
            // hard-coded (§ 9.5) — but WHICH RULES EXIST is the server's
            // vocabulary, and these are its names.
            for (const [value, label] of [["by_element", "By element"],
                                          ["by_index",   "By atom index"],
                                          ["by_residue", "By residue"],
                                          ["by_label",   "By label"]]) {
                const option = doc.createElement("option");
                option.value = value; option.textContent = label;
                kind.appendChild(option);
            }
            kind.value = filter.kind;
            kind.addEventListener("change", (e) => {
                model.selection.updateFilter(at, { kind: e.target.value });
            });

            const value = el("input", "selection-filter-text");
            value.type = "text";
            value.value = filter.value;
            value.addEventListener("input", (e) => {
                model.selection.updateFilter(at, { value: e.target.value });
            });

            const remove = el("button", "selection-filter-remove");
            remove.type = "button";
            remove.textContent = "×";
            remove.setAttribute("aria-label", "Remove this filter");
            remove.addEventListener("click", () => model.selection.removeFilter(at));

            row.appendChild(kind);
            row.appendChild(value);
            row.appendChild(remove);
            rows.appendChild(row);
        });
    }

    function drawCell() {
        cellReadout.textContent = "";
        // THE CELL AS IT WILL BE USED (§ 9.3's main way in), under the names the
        // whole system uses for it.
        const cell = model.getUnitCellInfo();
        const vector = (v) => (Array.isArray(v) ? v.map(Number).join(", ") : "—");
        for (const [label, value] of [
            ["Lattice", cell.cell ? "set" : "none"],
            ["Origin",  vector(cell.cell_origin)],
            ["Axes",    Array.isArray(cell.axis_kind) ? cell.axis_kind.join(" · ") : "—"],
            ["Vacuum",  vector(cell.vacuum)],
        ]) {
            const term = doc.createElement("dt");
            term.className = "selection-mini-label";
            term.textContent = label;
            const detail = doc.createElement("dd");
            detail.className = "cell-value";
            detail.textContent = value;
            cellReadout.appendChild(term);
            cellReadout.appendChild(detail);
        }
    }

    const offSelection = model.selection.subscribe(draw);
    const offData = model.subscribe(() => {
        draw(model.selection.getState());
        drawCell();
    });
    drawCell();

    card.panel.appendChild(root);

    return {
        dispose() {
            offSelection(); offData();
            try { root.remove(); } catch (_) {}
        },
    };
}
