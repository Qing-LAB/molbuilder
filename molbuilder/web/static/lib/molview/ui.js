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
import { FROZEN_LABEL, PREDEFINED_LABELS, PREDEFINED_LABEL_NAMES }
    from "./model-jobs.js";


/**
 * Draw and wire every control, and return one teardown for all of them.
 *
 * @param card    the elements mount.js built (§ 8.1)
 * @param model   the data API — what almost everything here reads and writes
 * @param handle  the viewer, for PLAYBACK only (§ 8.5, § 9.2)
 * @param files   the door bytes leave through (§ 6.7, § 8) — `save(destination,
 *                filename, contents)`. MolView never reaches a file itself.
 */
/* THE CLASS EACH PREDEFINED NAME WEARS. Written out rather than composed, so
 * the class an element gets is a string that appears in this file — which is
 * what lets "every class the module writes is one the stylesheet defines" be
 * checked at all. The stylesheet owns what each one looks like; the list in
 * model-jobs.js owns which name wears which. */
const TONE_CLASS = {
    1:      "molviewer-label-predefined--1",
    2:      "molviewer-label-predefined--2",
    3:      "molviewer-label-predefined--3",
    4:      "molviewer-label-predefined--4",
    // The one that means something: `frozen_atoms` changes the calculation.
    warn:   "molviewer-label-frozen",
};

/**
 * Draw and wire every control, and return one teardown for all of them.
 *
 * The names MolView offers before anyone has used them are its own
 * (`PREDEFINED_LABELS`, model-jobs.js). They are conveniences — spellings, so a
 * user does not retype `L-electrode` as `L-Electrode` — and MolView reads
 * meaning into none of them. The single exception is `frozen_atoms`, which the
 * calculation acts on, and it looks different for that reason.
 */
/* The playback-speed bounds the contract fixes (§ 1.1): milliseconds per frame,
 * 20 to 3000, defaulting to 150.
 *
 * EXPORTED, and the direction of that import is deliberate. These are § 1.1's
 * numbers for a CONTROL, so they belong in the file that builds § 1.1's
 * controls; mount.js imports them so the handle enforces exactly the range the
 * box offers. One definition, checked at both ends — rather than the box
 * offering 20–3000 while playback quietly allowed something else, which is the
 * shape the speed bug already took once. */
export const SPEED_MIN_MS = 20;
export const SPEED_MAX_MS = 3000;
export const SPEED_DEFAULT_MS = 150;


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
 * column in its own arithmetic (`--molviewer-size-rail-width`) instead of taking it out of the
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
        button.className = "molviewer-rail-button";
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

    const prev = el("button", "molviewer-frames-step");
    prev.type = "button"; prev.textContent = "‹";
    prev.setAttribute("aria-label", "Previous frame");

    const playBtn = el("button", "molviewer-frames-play");
    playBtn.type = "button"; playBtn.textContent = "▶";
    playBtn.setAttribute("aria-label", "Play");

    const next = el("button", "molviewer-frames-step");
    next.type = "button"; next.textContent = "›";
    next.setAttribute("aria-label", "Next frame");

    const transport = el("div", "molviewer-frames-transport");
    transport.appendChild(prev);
    transport.appendChild(playBtn);
    transport.appendChild(next);

    const slider = el("input", "molviewer-frames-slider");
    slider.type = "range";
    slider.min = "0";
    slider.step = "1";
    slider.setAttribute("aria-label", "Frame");

    const counter = el("span", "molviewer-frames-counter");

    const loopWrap = el("label", "molviewer-frames-loop");
    const loopBox = doc.createElement("input");
    loopBox.type = "checkbox";
    loopBox.checked = handle.getLoop();
    loopWrap.appendChild(loopBox);
    loopWrap.appendChild(doc.createTextNode(" loop"));

    /* HOW FAST THE MOVIE PLAYS (§ 1.1, § 8.5): milliseconds per frame, which is
     * the question a user has of a relaxation — "how long do I get to look at
     * each step?" — and also, exactly, what the timer is given. The box shows
     * the range (§ 1.1: 20–3000); the handle enforces it. */
    const speedWrap = el("label", "molviewer-frames-speed");
    speedWrap.title = "Playback speed (ms per frame)";
    // THE CLASS MATTERS: molview.css styles `.molviewer-frames-speed-input` (its width, and
    // the spinner-arrow removal). Built without it, the control appears and is
    // simply unstyled -- which is the same trap as the panel's controls, and I
    // walked into it once here before the browser check caught it.
    const speedBox = el("input", "molviewer-frames-speed-input");
    speedBox.type = "number";
    speedBox.min = String(SPEED_MIN_MS);
    speedBox.max = String(SPEED_MAX_MS);
    speedBox.step = "20";
    // Opens on what playback is set to, not on a number written here. They were
    // two copies and they disagreed: 150 in the box, 83 in the timer.
    speedBox.value = String(handle.getSpeed());
    speedBox.setAttribute("aria-label", "Playback speed, milliseconds per frame");
    speedWrap.appendChild(speedBox);
    speedWrap.appendChild(doc.createTextNode(" ms"));

    bar.appendChild(transport);
    bar.appendChild(slider);
    bar.appendChild(counter);
    bar.appendChild(loopWrap);
    bar.appendChild(speedWrap);

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
        if (handle.isPlaying()) handle.pause();
        else handle.play();
        reflect();
    });
    loopBox.addEventListener("change", (e) => handle.setLoop(!!e.target.checked));
    /* THE BOX DOES NOT HOLD THE SPEED — it sets it and then shows what was
     * taken. The handle owns it (§ 9.2, like loop), clamps it to § 1.1's range,
     * and restarts a running timer itself, so "changing the speed while it plays
     * takes effect now" is playback's rule rather than a sequence this control
     * has to remember to perform.
     *
     * Reading straight back is what makes the clamp honest: type 5000 and the
     * box settles to 3000, because that is what playback is actually doing. */
    speedBox.addEventListener("change", () => {
        handle.setSpeed(parseInt(speedBox.value, 10));
        speedBox.value = String(handle.getSpeed());
    });

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
        // Loop lives on the handle, so it is read back like everything else
        // here. It was taken once at build and never again — which held only
        // because this box is loop's sole writer. "Nothing here is cached"
        // above should be true of the whole bar, not most of it.
        loopBox.checked = handle.getLoop();
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
 * to the window's edge it may come. BOTH LIVE IN THE STYLESHEET — `--molviewer-menu-gap`
 * and `--molviewer-menu-margin` — with every other distance the module uses, and this
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
        gap:    read("--molviewer-menu-gap",    MENU_PLACEMENT.gap),
        margin: read("--molviewer-menu-margin", MENU_PLACEMENT.margin),
    };
}

function mountMenus(doc, card, model, files) {
    const bar = doc.createElement("div");
    bar.className = "molviewer-menu-bar";
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
    root.className = "molviewer-menu";
    const summary = doc.createElement("summary");
    summary.textContent = "View";
    root.appendChild(summary);

    // The menu's contents live in a body, which is what the stylesheet lays out.
    const body = doc.createElement("div");
    body.className = "molviewer-menu-body";
    root.appendChild(body);

    const section = (heading) => {
        const wrap = doc.createElement("div");
        wrap.className = "molviewer-menu-section";
        const label = doc.createElement("div");
        label.className = "molviewer-menu-heading";
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
    repRow.className = "molviewer-menu-style-row";
    const REPS = [["stick", "Sticks"], ["ball-and-stick", "Ball & stick"],
                  ["sphere", "Spheres"], ["line", "Lines"]];
    const repButtons = {};
    for (const [value, label] of REPS) {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "molviewer-menu-style-btn";
        button.textContent = label;
        button.addEventListener("click", () => model.view.set("style", value));
        repRow.appendChild(button);
        repButtons[value] = button;
    }
    drawn.appendChild(repRow);

    /* THE RADIUS SLIDER (§ 1.1: "a radius slider from 0.2 to 2.5 that scales
     * stick thickness / sphere size / line width"). It sits with the reps
     * because it scales whichever one is showing, which is what the carried
     * stylesheet lays out — `.molviewer-menu-radius-row` was styled all along and
     * had nothing to style.
     *
     * The label WRAPS the input rather than pairing with it through `for`/`id`.
     * § 5.6 puts two viewers on one page and an id is document-global, so the
     * second mount would duplicate it and both labels would point at the first
     * viewer's slider. */
    const radiusRow = doc.createElement("div");
    radiusRow.className = "molviewer-menu-radius-row";
    const radiusLabel = doc.createElement("label");
    radiusLabel.textContent = "Radius";
    /* NO class on the input or the output. The stylesheet reaches them by
     * ELEMENT inside the row — `.molviewer-menu-radius-row > input[type="range"]`
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
    bgRow.className = "molviewer-menu-background-row";
    const swatches = {};
    for (const [value, name] of BACKGROUNDS) {
        const swatch = doc.createElement("button");
        swatch.type = "button";
        swatch.className = "molviewer-menu-background-swatch";
        swatch.setAttribute("aria-label", name + " background");
        swatch.title = name;
        if (value === "transparent") {
            // The stylesheet draws the checkerboard from this class; a colour
            // set inline would paint over it.
            swatch.classList.add("molviewer-is-transparent");
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
    custom.className = "molviewer-menu-background-custom";
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
    projection.className = "molviewer-menu-projection";
    projection.textContent = "Orthographic";
    /* NO initial `aria-pressed` here, and the next value is NOT read back off
     * the attribute. Both were the same mistake: treating what was last PAINTED
     * as where the setting lives. `paint()` below sets the attribute from the
     * store, on the first pass like every other one.
     *
     * Reading it back was the dangerous half, because `view.set` DEDUPES
     * (stores.js: `if (settings[name] === value) return`). Let the attribute
     * drift from the store once and every click computes the same stale value,
     * sets what the store already holds, fires nothing, and repaints nothing —
     * the control is dead for good, silently. */
    projection.addEventListener("click", () => {
        model.view.set("orthographic", !model.view.get().orthographic);
    });
    drawn.appendChild(projection);

    /* EVERY CONTROL READS ITS STATE BACK FROM `view` (§ 8.5), never from what it
     * last did — so a setting changed anywhere else lights the right control
     * here with nothing to keep in step. */
    const paint = (settings) => {
        for (const [value] of REPS) {
            repButtons[value].classList.toggle("molviewer-is-active", settings.style === value);
        }
        radius.value = String(settings.radius);
        radiusOut.textContent = Number(settings.radius).toFixed(2);
        for (const [value] of BACKGROUNDS) {
            swatches[value].classList.toggle("molviewer-is-active",
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
    };
    offs.push(model.view.subscribe(paint));

    /* THE FIRST PAINT IS THE SAME PAINT. `view.subscribe` hands nothing over
     * (§ 9.6 is a plain change-and-subscribe), unlike `selection.subscribe`
     * which fires immediately — so a view subscriber has to take its own first
     * pass, and this one used to take it for the RADIUS ALONE. Style, background
     * and projection were left showing hand-written initial markup instead: a
     * second place the control's state was decided, which is the one thing
     * § 8.5 says a control must not have. Calling the painter is all it takes. */
    paint(model.view.get());

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
    root.className = "molviewer-menu";
    const summary = doc.createElement("summary");
    summary.textContent = "Export";
    root.appendChild(summary);

    const body = doc.createElement("div");
    body.className = "molviewer-menu-body";
    root.appendChild(body);
    const section = doc.createElement("div");
    section.className = "molviewer-export-section";
    const label = doc.createElement("div");
    label.className = "molviewer-export-section-label";
    label.textContent = "Structure";
    section.appendChild(label);
    const row = doc.createElement("div");
    row.className = "molviewer-export-row";
    section.appendChild(row);
    body.appendChild(section);

    const item = (text, onClick) => {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "molviewer-export-btn";
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
    badge.className = "molviewer-overlay molviewer-overlay--bottom-right molviewer-overlay--warn";
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
    readout.className = "molviewer-overlay molviewer-overlay--bottom-left molviewer-overlay--info";
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
        /* WHICH ATOM, not just which number (§ 1.1). A bare `#5` makes the
         * reader look away from the answer to find out what it is about, and
         * on a mixed structure the number alone does not say whether the 0.96 Å
         * is the bond they meant. The element comes from the master copy, read
         * here beside the coordinates so both describe the same moment. */
        const elements = model.getElements() || [];
        const name = (i) => (elements[i] || "?") + " #" + toDisplay(i);
        let text = "";
        if (picked.length === 1) {
            const p = at(picked[0]);
            if (p) text = name(picked[0]) + " — ("
                        + p.map((v) => v.toFixed(3)).join(", ") + ") Å";
        } else if (picked.length === 2) {
            const a = at(picked[0]), b = at(picked[1]);
            if (a && b) text = "|" + name(picked[0]) + " – " + name(picked[1])
                             + "| = " + distance(a, b).toFixed(3) + " Å";
        } else {
            // picked[1] is the vertex: the atom clicked SECOND. Writing the
            // three in that order is what says which one it is — the middle
            // position IS the claim, so the reader can check the answer against
            // the atoms it came from without being told the convention.
            const a = at(picked[0]), v = at(picked[1]), c = at(picked[2]);
            if (a && v && c) text = "∠" + name(picked[0]) + " – " + name(picked[1])
                                  + " – " + name(picked[2])
                                  + " = " + angle(a, v, c).toFixed(1) + "°";
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
    /* WHAT EACH PREDEFINED NAME WEARS, and what it says on hover. Read once
     * from MolView's own list (model-jobs.js).
     *
     * IT USED TO BE HANDED IN AT MOUNT, and that is why every chip on every real
     * page came out the same colour: the option existed, the tones existed, and
     * only the module's own demo page ever passed the list. Five pages would
     * have had to repeat the same four names — five copies of one list, drifting
     * apart. They are MolView's conveniences, so MolView keeps them. */
    const labelTone = new Map();
    const labelNote = new Map();
    PREDEFINED_LABELS.forEach((entry) => {
        const cls = TONE_CLASS[entry.tone];
        if (cls) labelTone.set(entry.name, cls);
        if (entry.description) labelNote.set(entry.name, entry.description);
    });

    const el = (tag, className) => {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        return node;
    };

    const root = el("div", "molviewer-selection-card");

    /* Every radio group in this panel is named after the OWNER (§ 5.6). The name
     * is what browsers group radios by, and it is global to the document — so
     * two viewers on one page with a fixed name are ONE group, and choosing a
     * page or an editor in the second silently un-chooses it in the first. */
    const owner = card.root.getAttribute("data-owner") || "molview";

    /* ── The two pages, and the tab bar that switches them (§ 8.1) ───────── */
    const header = el("div", "molviewer-card-header");
    const tabs = el("div", "molviewer-panel-tab-switch molviewer-selection-header-tabs");
    const pages = {};
    const tabInputs = {};
    for (const [key, label] of [["selection", "Selection"], ["cell", "Cell"]]) {
        /* A TAB IS A RADIO INSIDE A LABEL, not a button — the carried stylesheet
         * draws the chosen tab from `:has(input:checked)` (accent text, accent
         * underline) and its type from `.molviewer-panel-tab-option > span`. Built as a
         * bare `<button>` with the text on it, NEITHER rule can match: the
         * switch renders as two words in the browser's default button font with
         * no indication of which page you are on. The markup is as much the
         * stylesheet's contract as the class name is. */
        const tab = el("label", "molviewer-panel-tab-option");
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

        const page = el("div", "molviewer-panel-tab");
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
    /* WHAT THE SERVER SAID ABOUT THE STRUCTURE ITSELF (§ 6.8), above the tabs
     * so it is visible on either page: a notice about the whole structure --
     * from a load or from an edit -- has no row to sit under the way a cell
     * notice does. */
    const panelNotices = el("div", "molviewer-notices");
    panelNotices.hidden = true;
    root.appendChild(panelNotices);
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
        // Opening the page clears its mark: by then the message has been seen.
        markPageWithNotices();
    }
    showPage("selection");

    /* ── Which editor: the atom list, or the filter (§ 9.5) ──────────────── */
    //
    // Two editors of ONE selection. Switching between them does not touch what
    // is selected — the panel redraws, the truth does not move.
    //
    /* NAMED FOR WHAT IT IS, NOT FOR THE GESTURE THAT REACHES IT. This read
     * `Click` — an input device — above a list of atoms, and the name stopped
     * being true the moment the list learned shift-ranges and box-drag. The
     * store's key stays `click`: it is the editor's id, not its label, and
     * renaming it would reach into every saved session's `mode`. */
    const mode = el("div", "molviewer-selection-mode");
    const modeInputs = {};
    for (const [key, label] of [["click", "Atom list"], ["filter", "Filter"]]) {
        const option = el("label", "molviewer-selection-mode-option");
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
    const clickSection = el("div", "molviewer-selection-click-section");
    /* `data-fill` is the role that makes a section TAKE THE LEFTOVER HEIGHT and
     * hold the page's single scroll region, so the atom list grows to the
     * panel's bottom and the buttons beneath it line up with the bottom of the
     * 3D window (§ 8.2's shared extent). Without it the section sizes to its
     * content and the page collapses upward. A page has exactly one. */
    clickSection.setAttribute("data-fill", "");
    /* THE COUNT IS NOT PART OF EITHER EDITOR. It lives with the buttons below
     * (Clear / Invert / All), which are shared by both pages, because it is a
     * fact about THE SELECTION and the selection is one truth with two editors
     * (§ 9.5) — not a fact about the list you happen to be looking at.
     *
     * It sat inside this section, so it showed on the atom list and vanished on
     * Filter. The number was never wrong there: `draw` calls `drawList` on every
     * snapshot whichever editor caused it, so an accurate count was being kept
     * behind a hidden element. Only its placement made it unreadable. */
    const listWrap = el("div", "molviewer-selection-list-wrap");
    const list = el("table", "molviewer-atoms-table");
    listWrap.appendChild(list);
    clickSection.appendChild(listWrap);
    pages.selection.appendChild(clickSection);

    /* ── Picking a run, and picking a box (§ 9.5) ────────────────────────── */
    //
    /* WHERE A RUN COUNTS FROM: the last row the user clicked. That is a fact
     * about the pointer, not about the structure, so it lives here with the rest
     * of the interaction state and never enters the store — the store holds what
     * is selected and in what order it was picked, and an anchor is neither. It
     * survives a redraw because it lives in the panel's closure, not in a row. */
    let rangeAnchor = null;

    /* A RUN ADDS. Picking 1-10 and then 40-50 is the ordinary case, and a run
     * that replaced would make the second gesture undo the first; `Clear` is
     * how you start over. Both ends are inclusive, in either direction, and the
     * bulk door carries no pick trail — a run has no vertex to measure from. */
    /* THE RUN IS THE ROWS BETWEEN THEM, read off the list as drawn.
     *
     * It counted `for (i = lo; i <= hi; i++)` — arithmetic on two atom indices,
     * which INVENTS every index in between rather than reading atoms the model
     * handed over. That is only right while the list happens to show every atom
     * in index order: it says "the atoms between these two" when the user asked
     * for "the rows between these two", and those are the same sentence only by
     * coincidence of today's `getAtoms`. Show a subset, or order the rows by
     * element or residue, and shift-click starts selecting atoms that are not on
     * screen — silently, because a made-up index is still a valid one.
     *
     * `rowBoxes` is the list as the user sees it, in the order they see it, so
     * "between" means what it looks like it means and nothing is invented. */
    function selectRun(fromAtom, toAtom) {
        const from = rowBoxes.findIndex((e) => e.atom === fromAtom);
        const to   = rowBoxes.findIndex((e) => e.atom === toAtom);
        if (from < 0 || to < 0) return;   // a row that is no longer drawn
        const lo = Math.min(from, to);
        const hi = Math.max(from, to);
        model.selection.add(rowBoxes.slice(lo, hi + 1).map((e) => e.atom));
    }

    /* THE ROWS AS THEY ARE BUILT, so the drag box has something to hit-test
     * without asking the document to find them again. Rebuilt with the list
     * (drawList empties it), which is also what keeps a stale row out of a hit
     * after the structure changes under it. */
    const rowBoxes = [];

    /* IS THIS ONE OF THE ROW'S OWN CONTROLS? A checkbox or a label chip speaks
     * for itself and must not also start a drag. Written as a walk rather than
     * `closest` because the module should lean on the small, ordinary part of
     * the DOM: the panel is built and driven through a handful of calls, and
     * every one it adds is one more thing a host's environment has to provide. */
    function isControl(node) {
        for (let at = node; at && at !== listWrap; at = at.parentNode) {
            const tag = at.tagName;
            if (tag === "INPUT" || tag === "BUTTON" || tag === "SELECT") return true;
        }
        return false;
    }

    /* DRAG A BOX OVER THE ROWS and every row it touches is added.
     *
     * The box is drawn in the list's own coordinates so it scrolls with the
     * content, and the hit test is row-rectangle against box-rectangle — the
     * rows are the things being picked, so their boxes are what must intersect,
     * not the pointer's path.
     *
     * A drag that never leaves its starting row is a CLICK, and is left to the
     * click handler: dragging one pixel while picking a single atom must not
     * become a different gesture. */
    (function wireDragBox() {
        const DRAG_FLOOR_PX = 4;          // below this, it was a click
        let startX = 0, startY = 0, dragging = false, marquee = null;

        function rowsWithin(box) {
            const hit = [];
            for (const entry of rowBoxes) {
                const r = entry.row.getBoundingClientRect();
                if (r.bottom >= box.top && r.top <= box.bottom
                    && r.right >= box.left && r.left <= box.right) {
                    hit.push(entry.atom);
                }
            }
            return hit;
        }

        listWrap.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;                    // left button only
            if (isControl(e.target)) return;
            /* A BOX THAT OUTLIVED ITS DRAG IS SWEPT UP HERE. If the pointer
             * leaves the window mid-drag the mouseup lands somewhere else and
             * the box is left on screen, pointing at nothing. Clearing it as
             * the next drag starts costs one line and means a stray box can
             * never be more than one gesture old. */
            if (marquee) { marquee.remove(); marquee = null; }
            startX = e.clientX; startY = e.clientY; dragging = false;
            /* STOP THE BROWSER STARTING A TEXT SELECTION. Native drag-select
             * begins on mousedown, so preventing it on the first mousemove is
             * already too late — the words go blue under the box and the two
             * highlights fight. The controls are exempted above, so this takes
             * nothing away: the row is a thing you pick, not prose you read. */
            e.preventDefault();
        });

        listWrap.addEventListener("mousemove", (e) => {
            if (!e.buttons) return;
            if (!dragging) {
                if (Math.abs(e.clientX - startX) < DRAG_FLOOR_PX
                    && Math.abs(e.clientY - startY) < DRAG_FLOOR_PX) return;
                dragging = true;
                marquee = el("div", "molviewer-selection-marquee");
                listWrap.appendChild(marquee);
            }
            const wrap = listWrap.getBoundingClientRect();
            marquee.style.left   = (Math.min(startX, e.clientX) - wrap.left) + "px";
            marquee.style.top    = (Math.min(startY, e.clientY) - wrap.top
                                    + listWrap.scrollTop) + "px";
            marquee.style.width  = Math.abs(e.clientX - startX) + "px";
            marquee.style.height = Math.abs(e.clientY - startY) + "px";
        });

        doc.addEventListener("mouseup", (e) => {
            if (!dragging) return;
            dragging = false;
            if (marquee) { marquee.remove(); marquee = null; }
            const hit = rowsWithin({
                left:   Math.min(startX, e.clientX),
                right:  Math.max(startX, e.clientX),
                top:    Math.min(startY, e.clientY),
                bottom: Math.max(startY, e.clientY),
            });
            if (!hit.length) return;
            model.selection.add(hit);
            // The run's anchor follows the last thing the user touched.
            rangeAnchor = hit[hit.length - 1];
        });
    })();

    /* ── The filter page: rows, edited one at a time (§ 8.4) ─────────────── */
    const filterSection = el("div", "molviewer-filter-section");
    filterSection.setAttribute("data-fill", "");
    const rows = el("div", "molviewer-filter-rows");
    filterSection.appendChild(rows);

    const actions = el("div", "molviewer-filter-actions");
    const addRow = el("button", "molviewer-selection-add-filter-row");
    addRow.type = "button";
    addRow.textContent = "+ Add filter";
    addRow.addEventListener("click", () => model.selection.addFilter());
    actions.appendChild(addRow);
    filterSection.appendChild(actions);

    const footer = el("div", "molviewer-filter-footer");
    const combineRow = el("div", "molviewer-selection-combinator-row");
    const combine = el("select", "molviewer-selection-combinator-select");
    /* THE THIRD IS THE COMPLEMENT OF THE SECOND, and its label says the set it
     * gives rather than the operator it is: a chemist wants "everything that is
     * not the gold and not the sulfur", and reads that far faster than NOR. */
    for (const [value, label] of [["and", "Match all"],
                                  ["or", "Match any"],
                                  ["nor", "Match none"]]) {
        const option = doc.createElement("option");
        option.value = value; option.textContent = label;
        combine.appendChild(option);
    }
    combine.addEventListener("change", (e) => {
        model.selection.setCombinator(e.target.value);
    });
    combineRow.appendChild(combine);
    const apply = el("button", "molviewer-selection-apply-filter-btn");
    apply.type = "button";
    apply.textContent = "Apply filter";
    // Filter mode composes a query the user EXPLICITLY applies, and applying it
    // REPLACES the selection (§ 9.5). Nothing happens while they type.
    apply.addEventListener("click", () => model.selection.applyFilter());
    footer.appendChild(combineRow);
    footer.appendChild(apply);
    filterSection.appendChild(footer);
    /* WHEN A RULE MATCHES NOTHING, SAY SO. An empty result and never having
     * filtered leave the panel looking identical -- nothing selected -- so the
     * user is left to guess whether the rule was wrong or the button missed.
     *
     * It also explains the SECOND silence, which is the one that looks like a
     * bug: with nothing selected, "Show selected only" deliberately does
     * nothing (render-engine.js -- isolating requires a non-empty selection,
     * because the alternative is an empty window). The switch stays lit and the
     * structure stays whole, which is right, and unexplained is what made it
     * read as broken. */
    // The panel's existing notice styling, not a class of its own: this says the
    // same KIND of thing the cell notices say, and a second look for one job is
    // how two things that should match stop matching.
    const filterNote = el("div", "molviewer-notice molviewer-notice--warn");
    filterNote.hidden = true;
    filterSection.appendChild(filterNote);
    pages.selection.appendChild(filterSection);

    /* ── The click operations, and the label block ───────────────────────── */
    // A visual divider between the SELECT-side controls above and the ACT-side
    // controls below — the stylesheet's own separation of the two jobs.
    pages.selection.appendChild(el("div", "molviewer-selection-divider"));

    // Every action button takes one of the five classes the stylesheet's shared
    // base is written over. A sixth name would get no base at all: no padding,
    // no font, no baseline — which is exactly how one button ends up a different
    // size from the two beside it.
    const actionsRow = el("div", "molviewer-selection-actions-row");
    for (const [label, className, run] of [
        ["Clear",  "molviewer-selection-clear-btn", () => model.selection.clear()],
        ["Invert", "molviewer-selection-add-btn",   () => model.selection.invert(atomCount())],
        ["All",    "molviewer-selection-add-btn",   () => model.selection.all(atomCount())],
    ]) {
        const button = el("button", className);
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", run);
        actionsRow.appendChild(button);
    }
    /* Beside the buttons that change it, and after them so the eye lands on the
     * controls first and the number reads as their result. Shown on both pages
     * because the row is. */
    const count = el("div", "molviewer-selection-count");
    actionsRow.appendChild(count);
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
    const assign = el("div", "molviewer-selection-assign");

    /* THE TARGET: pick a label the structure already has, or type a new one.
     *
     * Typing a name that already exists is retyping something the structure can
     * simply be asked for — and a typo makes a SECOND label that looks like the
     * first, which is a whole extra region as far as anything downstream is
     * concerned. So the labels in play are offered, and the free-text box is
     * what you reach for only when none of them is what you meant.
     *
     * The list is read from the structure, never kept (§ 5.2): it is the names
     * `getRegions` groups by, which is the same one walk everything else reads
     * (§ 6.6). A label that stops being carried by any atom leaves the list on
     * its own, with nothing to keep in step.
     */
    const targetRow = el("div", "molviewer-selection-target-row");
    const chooser = el("select", "molviewer-selection-assign-select");
    chooser.setAttribute("aria-label", "Label to apply");
    const target = el("input", "molviewer-selection-new-label");
    target.type = "text";
    target.placeholder = "New label name";
    targetRow.appendChild(chooser);
    targetRow.appendChild(target);
    assign.appendChild(targetRow);

    /* WHAT IS CHOSEN — three states, and they are three TYPES rather than three
     * strings: `undefined` before the list has been drawn, `null` for
     * "+ new label…", and a string for a label that exists.
     *
     * "+ new label…" is not a label, so it does not get a label's value. It used
     * to: the option carried `"\0new"`, a name no label could have. Encoding
     * "this one is not a name" INTO the name is what needed a character no name
     * could contain, and the cost landed somewhere unrelated — a single NUL
     * makes ui.js binary to `grep`, which then reports no matches AND no error.
     * A search that silently finds nothing is worse than one that fails, and
     * this file is the module's whole UI layer.
     *
     * Held here rather than read back off the `<select>`, which is rebuilt
     * whenever the set of names changes: reading a choice off a control you are
     * about to rebuild is how a rebuild invents one (§ 8.5). */
    let chosen;
    const NO_NAME = "";     // the option's DOM value — a select carries strings
    // The free-text box is only in the way when an existing label is chosen.
    const showNewBox = () => { target.hidden = chosen !== null; };
    chooser.addEventListener("change", () => {
        chosen = chooser.value || null;
        showNewBox();
    });

    /* WHICH NAMES ARE ON OFFER, so the list is rebuilt only when it changes.
     *
     * This is drawn from the same snapshot the atom list is, which arrives on
     * every click — and rebuilding a `<select>` under a user who has it open
     * shuts it. Same rule as the filter rows below: rebuild when the SET
     * changes, never because something else did. */
    /* EVERY LABEL A CONTROL MAY OFFER: the predefined names, then whatever this
     * structure carries that is not already among them.
     *
     * ONE answer, because two controls ask the same question -- the Assign
     * chooser and the filter's `by label` row. Reading it in two places is how
     * they come to disagree about what exists.
     *
     * Predefined first and in their own order (the device reads left to right:
     * L-electrode, R-electrode, bridge, interface), then the structure's own,
     * sorted. § 9.5: what is worth offering is read from the structure -- the
     * predefined names are added to that reading, never instead of it. */
    const knownLabels = () => {
        const carried = Object.keys(model.getRegions() || {}).sort();
        return PREDEFINED_LABEL_NAMES.concat(
            carried.filter((name) => PREDEFINED_LABEL_NAMES.indexOf(name) < 0));
    };

    let renderedTargets = null;   // the last list of names drawn

    function drawTargets() {
        const names = knownLabels();
        /* Compare the LISTS. This used to flatten them into one string and
         * compare that, which needed a separator no name could contain -- the
         * second NUL in this file. Comparing what you already have needs no
         * separator, no encoding, and no character to reserve. */
        if (renderedTargets
            && renderedTargets.length === names.length
            && renderedTargets.every((name, at) => name === names[at])) return;
        renderedTargets = names;
        chooser.textContent = "";
        for (const name of names) {
            const option = doc.createElement("option");
            option.value = name;
            option.textContent = name;
            chooser.appendChild(option);
        }
        const fresh = doc.createElement("option");
        fresh.value = NO_NAME;
        fresh.textContent = "+ new label…";
        chooser.appendChild(fresh);

        /* KEEP WHAT WAS CHOSEN. If it is gone — the last atom carrying it just
         * had it taken off — fall back to the new-label box WITH THE NAME IN IT,
         * rather than silently landing on whichever label happens to sort first.
         * Quietly retargeting the next Assign is how a user labels the wrong
         * atoms and is never told. */
        if (chosen === undefined) {
            chosen = names.length ? names[0] : null;
        } else if (chosen !== null && names.indexOf(chosen) < 0) {
            if (!target.value) target.value = chosen;
            chosen = null;
        }
        chooser.value = chosen === null ? NO_NAME : chosen;
        showNewBox();
    }

    const verbRow = el("div", "molviewer-selection-verb-row");
    // What the three verbs act on: the chosen label, or the typed one.
    const named = () => (chosen === null
        ? String(target.value || "").trim()
        : String(chosen || ""));
    for (const [label, className, verb] of [
        ["Assign",   "molviewer-selection-assign-btn",        "replace"],
        ["+ Add",    "molviewer-selection-add-target-btn",    "add"],
        ["− Remove", "molviewer-selection-remove-target-btn", "remove"],
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
    /* WHICH REGIME YOU ARE IN, in one sentence, above the numbers.
     *
     * The four rows answer "is this box mine?" (the `(default)` tag) and never
     * answered "is my vacuum doing anything?" -- which is the question that
     * decides whether an edit will have any effect at all. A user with a typed
     * cell can change the vacuum all day and watch nothing happen.
     *
     * structure-periodicity.md § 6.1a, matrix A: an explicit cell IS the box,
     * and demotes vacuum to reference-only. */
    const cellRegime = el("p", "molviewer-cell-regime");
    pages.cell.appendChild(cellRegime);
    const cellReadout = el("dl", "molviewer-cell-readout");
    pages.cell.appendChild(cellReadout);
    // What the server said about this box, under the numbers it is about.
    const cellNotices = el("div", "molviewer-notices");
    cellNotices.hidden = true;
    pages.cell.appendChild(cellNotices);

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

        /* Drawn from the snapshot like everything else here: the store records
         * what the last apply found, and clears it the moment a row or the
         * combinator changes, so this can never show a stale answer to a
         * question the user has since edited. */
        const outcome = state.filterOutcome;
        const matchedNothing = !!outcome && outcome.matched === 0;
        filterNote.hidden = !matchedNothing;
        if (matchedNothing) {
            filterNote.textContent = state.isolate
                ? "No atoms matched this filter, so nothing is selected. "
                  + "“Show selected only” needs a selection, so the whole "
                  + "structure is still shown."
                : "No atoms matched this filter, so nothing is selected.";
        }

        drawList(state);
        drawRows(state);
        drawTargets();
        /* Read back from where it lives, like every other control here (§ 8.5).
         * The model clears the set on any change to the structure, so a line
         * that is gone is gone because the fact is, not because something here
         * remembered to remove it. */
        /* EACH NOTICE GOES WHERE ITS SUBJECT IS, and it says its own subject
         * (§ 6.8). A message about the box is drawn beside the box, on the page
         * a user would go to in order to change it; everything else is drawn
         * here, above the tabs, where it shows on either page.
         *
         * The split used to be made on where the BATCH came from -- a load, an
         * edit -- which is the wrong question: a warning that the cell is
         * unusable is about the cell whether it arrived with a file or with an
         * edit, and routing it by its origin put it above the atom list, three
         * clicks from the only control that could fix it.
         *
         * Written as "not cell" rather than a list of accepted subjects,
         * because a list has to be extended every time a notice gains a new
         * one, and the failure when somebody forgets is SILENT: the server
         * checks, the answer carries the verdict, and this line drops it. */
        drawNotices(panelNotices, notFor("cell"));
        markPageWithNotices();
        // A read-only viewer does not show the controls the gate would swallow.
        assign.hidden = model.mode === "readonly";
    }

    /* ONE ROW PER ATOM: a checkbox, the atom's facts, and the labels it carries
     * as chips you can take off (§ 6.2's three facts — element, labels, residue
     * — are exactly these columns).
     *
     * The checkbox is what makes the list a list of things you TICK. The row
     * stays clickable too, because a row is a bigger target than a box, and both
     * go through the same one write (§ 9.5) — so neither is a second answer.
     */
    function cell(className, text) {
        const node = el("td", className);
        node.textContent = text;
        return node;
    }

    /* A LABEL CHIP, with the × that takes it off THIS atom.
     *
     * Removing one label from one atom is a change to the structure like any
     * other, so it goes through the same door with the same gate (§ 9.4) — it
     * simply names the atom instead of letting the door default to the
     * selection. That is why it can be offered per row without the user first
     * having to select the atom, and why what they had selected is not disturbed.
     *
     * In a read-only viewer the × is not drawn at all: the gate would swallow
     * it, and a control that silently does nothing is a bad answer for a user
     * (§ 9.4). The chip itself stays — reading the labels is not an edit.
     */
    function labelTag(name, atomIndex) {
        /* A RESERVED LABEL READS DIFFERENTLY, and each one differently from the
         * others (§ 6.6). It is stored, filtered and applied exactly like any
         * other label — the chip is the only thing that changes — because
         * something downstream knows what it means and a user is owed that
         * before they tag atoms with one by accident. `frozen_atoms` and
         * `L-electrode` do very different things to a calculation, so one shared
         * "reserved" colour would say only that they are both special.
         *
         * The description rides on the chip, which is as far as the viewer goes:
         * it CARRIES what the list says and acts on none of it. */
        const tone = labelTone.get(name);
        const tag = el("span", tone
            ? "molviewer-selection-tag " + tone
            : "molviewer-selection-tag molviewer-label-region");
        const note = labelNote.get(name);
        if (note) {
            tag.title = name === FROZEN_LABEL
                ? name + " — " + note
                : name + " — a name MolView offers: " + note;
        }
        const text = el("span");
        text.textContent = name;
        tag.appendChild(text);
        if (model.mode === "readonly") return tag;
        const strip = el("button", "molviewer-selection-tag-remove");
        strip.type = "button";
        strip.textContent = "×";
        // 1-based on screen, through the one translation (§ 11.5).
        strip.title = "Remove " + name + " from atom #" + toDisplay(atomIndex);
        strip.setAttribute("aria-label", strip.title);
        strip.addEventListener("click", (e) => {
            // The row is clickable; taking a label off is not also a selection.
            e.stopPropagation();
            model.selection.writeLabel(name, "remove", [atomIndex]);
        });
        tag.appendChild(strip);
        return tag;
    }

    function drawList(state) {
        const atoms = model.getAtoms();
        count.textContent = atoms
            ? state.selection.length + " of " + atoms.length + " selected"
            : "";
        list.textContent = "";
        // Emptied WITH the list: a row that is gone must not still be hittable,
        // and a hit on a detached row would name an atom that may not exist.
        rowBoxes.length = 0;
        if (!atoms) return;
        const picked = new Set(state.selection);
        for (const atom of atoms) {
            const row = doc.createElement("tr");
            if (picked.has(atom.index)) row.className = "molviewer-is-selected";

            const checkCell = el("td", "molviewer-atoms-column-check");
            const check = doc.createElement("input");
            check.type = "checkbox";
            check.checked = picked.has(atom.index);
            check.setAttribute("aria-label",
                               "Select atom #" + toDisplay(atom.index));
            check.addEventListener("change", (e) => {
                // The row's own handler would toggle it straight back.
                e.stopPropagation();
                if (e.shiftKey && rangeAnchor !== null) {
                    selectRun(rangeAnchor, atom.index);
                    /* THE TICK IS NOT SET HERE. The browser flips a checkbox
                     * before `change` fires, so the box is momentarily wrong —
                     * and the fix is not to write the right answer over it, it
                     * is to let the redraw put every box where the selection
                     * says. A line here that says "ticked" is a second opinion
                     * about what is selected, and § 8.4 exists because the
                     * second opinion is the one that drifts. */
                    return;
                }
                rangeAnchor = atom.index;
                model.selection.toggle(atom.index);
            });
            checkCell.appendChild(check);
            row.appendChild(checkCell);

            // 1-based on screen, through the one translation (§ 11.5).
            row.appendChild(cell("molviewer-atoms-column-idx", String(toDisplay(atom.index))));
            row.appendChild(cell("molviewer-atoms-column-el", atom.element || ""));
            row.appendChild(cell("molviewer-atoms-column-res", atom.residue || ""));

            const labels = el("td", "molviewer-atoms-column-labels");
            for (const name of atom.labels) {
                labels.appendChild(labelTag(name, atom.index));
            }
            row.appendChild(labels);

            row.addEventListener("click", (e) => {
                // The controls inside the row speak for themselves.
                if (e.target === check) return;
                /* SHIFT SELECTS THE RUN from the last row clicked to this one
                 * (§ 9.5). With nothing to count from it is an ordinary click —
                 * an anchorless run has no meaning to guess at. */
                if (e.shiftKey && rangeAnchor !== null) {
                    selectRun(rangeAnchor, atom.index);
                    return;
                }
                rangeAnchor = atom.index;
                model.selection.toggle(atom.index);
            });
            rowBoxes.push({ atom: atom.index, row: row });   // for the drag box
            list.appendChild(row);
        }
    }

    /* HOW MANY ROWS ARE ON SCREEN, so a change to what is IN one does not
     * rebuild them all.
     *
     * Typing was the failure: every keystroke reached `updateFilter`, the store
     * handed back a snapshot, and this function emptied the container and built
     * fresh rows — destroying the very input the user was typing in. The caret
     * went with it, so each character had to be preceded by a click. § 8.4 has
     * the store take a filter A ROW AT A TIME exactly so this cannot happen, and
     * the panel undid that by redrawing the whole set anyway.
     *
     * The rows are rebuilt only when the SET of them changes — added, removed,
     * or the panel is drawing them for the first time. A change to a row's
     * contents needs no rebuild: the control the user typed into already holds
     * what they typed, and writing it back would move the caret to the end. */
    let renderedRows = null;

    function drawRows(state) {
        /* REBUILD ON THE SET *AND* ON EACH ROW'S KIND. The count alone was
         * enough while every row held the same free-text box; it is not now that
         * `by label` carries a chooser instead. Re-kinding a row leaves the count
         * unchanged, so a count-only guard would leave the old control in place
         * and the user typing into a box the rule no longer uses.
         *
         * A row's VALUE is still not a reason to rebuild -- that is the caret
         * bug above, and the control the user typed into already holds it. */
        const shape = state.filters.map((f) => f.kind);
        if (renderedRows
            && renderedRows.length === shape.length
            && renderedRows.every((kind, at) => kind === shape[at])) return;
        renderedRows = shape;
        rows.textContent = "";
        if (!state.filters.length) {
            const empty = el("div", "molviewer-filter-empty");
            empty.textContent = "No filters yet.";
            rows.appendChild(empty);
            return;
        }
        state.filters.forEach((filter, at) => {
            const row = el("div", "molviewer-filter-row");

            const kind = el("select", "molviewer-filter-kind");
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

            /* BY LABEL OFFERS THE DEFINED NAMES, and only those. A label that
             * does not exist matches nothing, so a free-text box here can only
             * ever produce an empty selection and a user wondering why -- and it
             * is the second place a name gets retyped into a near-duplicate.
             * The other three rules stay free text: an element symbol, an index
             * range and a residue name are typed, not chosen. */
            let value;
            if (filter.kind === "by_label") {
                value = el("select", "molviewer-filter-text");
                /* A ROW THAT HAS NOT BEEN FILLED IN SAYS SO. A fresh row -- and
                 * a row just switched to this kind -- holds no value, and a rule
                 * with no value is skipped, exactly like an empty text box on
                 * the other three kinds. Selecting the first label instead would
                 * make a choice on the user's behalf and show a filter that is
                 * not the one being applied. */
                if (!filter.value) {
                    const unset = doc.createElement("option");
                    unset.value = "";
                    unset.textContent = "Choose a label…";
                    value.appendChild(unset);
                }
                for (const name of knownLabels()) {
                    const option = doc.createElement("option");
                    option.value = name;
                    option.textContent = name;
                    value.appendChild(option);
                }
                /* A row carrying a name no longer defined keeps it visible
                 * rather than silently becoming the first option -- the rule the
                 * user wrote is still what the row says it is. */
                if (filter.value && knownLabels().indexOf(filter.value) < 0) {
                    const stale = doc.createElement("option");
                    stale.value = filter.value;
                    stale.textContent = filter.value;
                    value.appendChild(stale);
                }
                value.value = filter.value || "";
                value.addEventListener("change", (e) => {
                    model.selection.updateFilter(at, { value: e.target.value });
                });
            } else {
                value = el("input", "molviewer-filter-text");
                value.type = "text";
                value.value = filter.value;
                value.addEventListener("input", (e) => {
                    model.selection.updateFilter(at, { value: e.target.value });
                });
            }

            const remove = el("button", "molviewer-filter-remove");
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

    /* ── The Cell page (§ 8.1): the resolved periodicity, and what is DERIVED ──
     *
     * WHY EACH ROW SAYS WHETHER IT IS A DEFAULT.  The four values are shown as
     * they will actually be used (§ 9.3's main way in), and three of the four
     * can be values MolView never received: a structure with no explicit cell is
     * shown the box resolved from its vacuum, an unset origin is shown the
     * corner the box is drawn from, and neither came from the user. Showing a
     * lattice with no way to tell it apart from one the user set is how somebody
     * comes to believe they fixed a cell they never fixed -- and then wonders why
     * widening the vacuum moved it.
     *
     * THE DEFAULT RULE IS DIFFERENT FOR EACH ROW, which is why this cannot be
     * one test against null:
     *
     *   Lattice   default when there is no explicit `cell` -- the box shown is
     *             then bbox + 2*vacuum on each isolated axis, resolved server
     *             side (model/structure-periodicity.md § 3a).
     *   Origin    default when there is no explicit `cell_origin`.
     *   Axes      default when unset OR every axis is `isolated` -- a fresh
     *             molecule loads all-isolated, and that is still the default
     *             configuration rather than a choice somebody made.
     *   Vacuum    default when there is no explicit `vacuum`. The row then
     *             shows the RESOLVED gap (3 A per side on each isolated axis),
     *             which is the number the box was actually built from.
     *
     * VACUUM MOVED FROM A VALUE TEST TO A PROVENANCE TEST (2026-08-03), because
     * the model gained the state the old comment said it did not have. It used
     * to read "default when it is 0 on every axis, because vacuum ALWAYS has a
     * value; unset is not a state it has" -- true until `vacuum` became
     * Optional. Left alone, `allZero(null)` is false, so the commonest case of
     * all -- nobody set one -- would have lost its tag, which is exactly
     * backwards. `Axes` stays value-based: all-isolated really is a value
     * judgement, not an absence.
     */
    /* ONE WAY A NOTICE IS DRAWN, because the Cell page and the panel line show
     * the same thing in two places and two renderers would drift.
     *
     * The server's words and the server's level, verbatim: MolView reports what
     * it was told. Rewording a warning here would put a second author on it, and
     * the message carries numbers -- clearances, axes -- that only the server
     * computed. */
    /* THE TWO CUTS OF THE ONE LIST. A notice with no subject is nobody's in
     * particular, so it lands in the general place -- which is also what an
     * older server that has not learned to say `about` produces, and that
     * answer stays readable rather than disappearing. */
    /* Function declarations, not consts: `showPage` runs during assembly, below
     * the tab bar and far above this line, and a const would still be in its
     * dead zone when it did. */
    function onlyFor(subject) {
        return (model.getNotices() || []).filter((n) => n.about === subject);
    }
    function notFor(subject) {
        return (model.getNotices() || []).filter((n) => n.about !== subject);
    }

    /* A MESSAGE ON THE PAGE YOU ARE NOT LOOKING AT STILL HAS TO REACH YOU.
     *
     * Putting a cell notice beside the cell is right -- that is where it can be
     * acted on -- but a structure loaded from the Selection page would then
     * warn into a page nobody is on. So the tab says there is something there:
     * the mark is on the TAB, which is visible from either page, and the words
     * stay where they can be used. Cleared when the page is opened, because by
     * then it has been seen. */
    function markPageWithNotices() {
        const waiting = onlyFor("cell").length > 0;
        const tab = tabInputs.cell && tabInputs.cell.parentNode;
        if (!tab) return;
        const show = waiting && pages.cell.hidden;
        if (show) tab.setAttribute("data-has-notices", "1");
        else tab.removeAttribute("data-has-notices");
    }

    function drawNotices(into, list) {
        into.textContent = "";
        into.hidden = !(list && list.length);
        for (const notice of (list || [])) {
            const line = el("p", "molviewer-notice molviewer-notice--"
                                 + (notice.level === "warn" ? "warn" : "info"));
            line.textContent = notice.message;
            into.appendChild(line);
        }
    }

    function drawCell() {
        cellReadout.textContent = "";
        // THE CELL AS IT WILL BE USED (§ 9.3's main way in), under the names the
        // whole system uses for it -- beside the RAW values, which is how each
        // row knows whether what it shows was given or derived.
        const cell = model.getUnitCellInfo();
        const rawCell = model.getUnitCell();
        const rawOrigin = model.getUnitCellOrigin();
        const rawAxes = model.getAxisKind();
        const rawVacuum = model.getVacuum();

        const vector = (v) => (Array.isArray(v)
            ? v.map((n) => Number(n).toFixed(3)).join(", ") : "—");
        const isolatedEverywhere = (a) => Array.isArray(a)
            && a.length && a.every((k) => k === "isolated");

        /* THE REGIME, said before the numbers.  `rawCell` is the whole test:
         * a cell the structure itself states IS the box (matrix A), and
         * everything else follows from that one fact. */
        const manual = !!rawCell;
        cellRegime.textContent = manual
            ? "The box is the cell you typed — vacuum is not used."
            : "The box is worked out from the molecule, your vacuum and the "
              + "axis kinds.";

        /* WHAT EACH ROW DOES, not what it is (cell-plan.md § 7).  Everything a
         * user needs to understand the two regimes lived in a document they
         * are not reading; the page carried a `(default)` tooltip and nothing
         * else.  Kept as `title` so it is there on hover without spending four
         * lines of a narrow panel on prose. */
        const HINT = {
            Lattice: "The box the calculation runs in. Type one to fix it; "
                   + "leave it and it is worked out from the molecule, the "
                   + "vacuum and the axis kinds.",
            Origin:  "Which corner the box starts at. Only meaningful once "
                   + "you have typed a lattice.",
            Axes:    "periodic repeats forever · isolated is a molecule in a "
                   + "box · transport is a device length. Vacuum applies to "
                   + "isolated axes only.",
            Vacuum:  "The empty gap left on each side of the molecule when the "
                   + "box is worked out — so the gap between the molecule and "
                   + "its periodic image is twice this. Ignored when you have "
                   + "typed a lattice. ≥ 8 Å per side is the usual advice for "
                   + "an isolated molecule.",
        };

        for (const [label, value, isDefault, inert] of [
            // THE LATTICE IS SHOWN, not summarised. It read "set" / "none",
            // which tells a user their structure has a box and refuses to say
            // what it is -- on the page whose whole job is reporting it.
            ["Lattice", matrixText(cell.cell), !rawCell, false],
            ["Origin",  vector(cell.cell_origin), !rawOrigin, false],
            ["Axes",    Array.isArray(cell.axis_kind)
                            ? cell.axis_kind.join(" · ") : "—",
                        !rawAxes || isolatedEverywhere(rawAxes), false],
            // INERT IN THE MANUAL REGIME.  The number is still shown -- it is
            // what the structure says -- but marked as not reaching the
            // calculation, so nobody edits it expecting the box to move.
            ["Vacuum",  vector(cell.vacuum), !rawVacuum, manual],
        ]) {
            const term = doc.createElement("dt");
            term.className = "molviewer-selection-mini-label";
            term.textContent = label;
            if (HINT[label]) term.title = HINT[label];
            const detail = doc.createElement("dd");
            detail.className = "molviewer-cell-value"
                + (inert ? " molviewer-cell-value--inert" : "");
            detail.textContent = value;
            if (inert) {
                const note = el("span", "molviewer-cell-inert-tag");
                note.textContent = " — not used";
                detail.appendChild(note);
            }
            if (isDefault) {
                const tag = el("span", "molviewer-cell-default-tag");
                tag.textContent = " (default)";
                tag.title = "derived, not set on this structure";
                detail.appendChild(tag);
            }
            cellReadout.appendChild(term);
            cellReadout.appendChild(detail);
        }
        /* UNDER THE ROWS THEY ARE ABOUT (§ 6.8). A cell warning names an axis
         * and a clearance; those numbers are the four rows above it, so this is
         * where the user is already looking, having just typed one of them. */
        drawNotices(cellNotices, onlyFor("cell"));
    }

    /* The 3x3 as three rows of three, at a fixed precision so the columns line
     * up and a small change is visible between redraws. `—` when the structure
     * has no box at all, which is a real answer and not a missing one. */
    function matrixText(m) {
        if (!Array.isArray(m) || m.length !== 3) return "—";
        return m.map((row) => (Array.isArray(row) ? row : [])
            .map((n) => Number(n).toFixed(3)).join("  ")).join("\n");
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
