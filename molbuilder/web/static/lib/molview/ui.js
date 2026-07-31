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
 * gives open, close and keyboard access for free; only mutual exclusion is
 * wired, so opening one closes the other.
 */
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

    // One open at a time.
    const menus = [view.root, exportMenu.root];
    for (const menu of menus) {
        menu.addEventListener("toggle", () => {
            if (!menu.open) return;
            for (const other of menus) if (other !== menu) other.open = false;
        });
    }

    card.canvas.appendChild(bar);

    return {
        dispose() {
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

    /* ── The switches: they change WHAT IS IN a frame (§ 9.6) ──────────────
     *
     * A switch is a TOGGLE BUTTON carrying `aria-pressed`, not a checkbox. That
     * is the stylesheet's design and it is also the more honest markup: the
     * indicator is drawn from the pressed state, so there is one source for what
     * the control shows and what it means.
     */
    const shown = section("Show");
    const SWITCHES = [
        ["showIndex",  "Atom numbers"],
        ["showForces", "Force arrows"],
        ["showCell",   "Unit cell"],
        ["showAxis",   "Axes"],
    ];
    const toggles = {};
    for (const [name, label] of SWITCHES) {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "mol-viewer-toggle";
        button.textContent = label;
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("click", () => {
            const on = button.getAttribute("aria-pressed") === "true";
            model.selection.setSwitch(name, !on);
        });
        shown.appendChild(button);
        toggles[name] = button;
    }
    // The switches have one home, so the menu REFLECTS them rather than
    // remembering what it last set (§ 5.2).
    offs.push(model.selection.subscribe((state) => {
        for (const [name] of SWITCHES) {
            toggles[name].setAttribute("aria-pressed", state[name] ? "true" : "false");
        }
    }));

    /* ── The drawing settings: they change HOW THE SAME FRAME IS PAINTED ─── */
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

    offs.push(model.view.subscribe((settings) => {
        for (const [value] of REPS) {
            repButtons[value].classList.toggle("is-active", settings.style === value);
        }
        projection.setAttribute("aria-pressed",
                                settings.orthographic ? "true" : "false");
    }));

    return {
        root,
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

    // One place produces the bytes, so the geometry and its sidecar can never
    // come from different reads — which is how the sidecar came to be dropped.
    function bytes() {
        const file = model.exportFile();
        // It REFUSES rather than writing a corrupt structure (§ 9.3), and a
        // refusal is not something to paper over with an empty file.
        if (!file) return null;
        return {
            geometry: file.text,
            sidecar:  JSON.stringify(file.sidecar, null, 2),
        };
    }

    function send(destination) {
        const out = bytes();
        if (!out || !files || typeof files.save !== "function") return;
        const stem = defaultStem(model);
        // The .json goes WITH the .xyz, so labels and frozen atoms survive into
        // whatever is generated from it (§ 11.3).
        files.save(destination, stem + ".xyz", out.geometry);
        files.save(destination, stem + ".molstruct.json", out.sidecar);
    }

    item("Save to project", () => send("project"));
    item("Download",        () => send("download"));

    return { root, dispose() { try { root.remove(); } catch (_) {} } };
}

/* A single-frame export out of a trajectory names the frame it came from, so the
 * file says which frame it is without anyone having to remember (§ 11.4). A
 * static structure gets no suffix — there is nothing to disambiguate. */
function defaultStem(model) {
    const base = "structure";
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
    badge.className = "molview-overlay molview-overlay--top-right molview-overlay--warn";
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
        const picked = model.selection.getState().pickOrder;
        const frame = model.getFrameAllAtoms(model.currentFrame());
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

    /* ── The two pages, and the tab bar that switches them (§ 8.1) ───────── */
    const header = el("div", "card-header");
    const tabs = el("div", "panel-page-switch selection-header-tabs");
    const pages = {};
    const tabButtons = {};
    for (const [key, label] of [["selection", "Selection"], ["cell", "Cell"]]) {
        const tab = el("button", "panel-page-option");
        tab.type = "button";
        tab.textContent = label;
        tab.addEventListener("click", () => showPage(key));
        tabs.appendChild(tab);
        tabButtons[key] = tab;

        const page = el("div", "panel-page");
        /* THE ID IS PART OF THE MARKUP CONTRACT, exactly like the class (§ 8.1).
         * The stylesheet singles this page out by id — the Cell page is all
         * read-only text, so it sizes to its content and then scrolls, while
         * the Selection page holds action buttons that must stay on screen and
         * so fills instead. Ship the page without its id and it silently takes
         * the other page's behaviour. */
        page.id = "panel-page-" + key;
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
            tabButtons[key].setAttribute("aria-selected", String(key === which));
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
        radio.name = "molview-mode";
        radio.addEventListener("change", () => model.selection.setEditor(key));
        const text = el("span");
        text.textContent = label;
        option.appendChild(radio);
        option.appendChild(text);
        mode.appendChild(option);
        modeInputs[key] = radio;
    }
    pages.selection.appendChild(mode);

    /* ── Show selected only — isolate (§ 9.4: not an edit) ───────────────── */
    //
    // It hides atoms from the DRAWING; the master copy still has all of them,
    // which is why the whole structure comes back when it goes off. A read-only
    // viewer isolates freely.
    const isolateRow = el("label", "selection-mode-option");
    const isolateBox = doc.createElement("input");
    isolateBox.type = "checkbox";
    isolateBox.addEventListener("change", (e) => {
        model.selection.setIsolate(!!e.target.checked);
    });
    const isolateText = el("span");
    isolateText.textContent = "Show selected only";
    isolateRow.appendChild(isolateBox);
    isolateRow.appendChild(isolateText);
    pages.selection.appendChild(isolateRow);

    /* ── The click page: the atom list ──────────────────────────────────── */
    const clickSection = el("div", "selection-click-section");
    /* The id, again as contract: `#selection-click-section:not([hidden])` is the
     * rule that makes this section FILL the panel and hold the single scroll
     * region, so the atom list grows to the panel's bottom and the buttons
     * beneath it line up with the bottom of the 3D window (§ 8.2's shared
     * extent). Without the id the section sizes to its content and the whole
     * page collapses upward. */
    clickSection.id = "selection-click-section";
    const count = el("div", "selection-count");
    const listWrap = el("div", "selection-list-wrap");
    const list = el("table", "selection-atom-table");
    listWrap.appendChild(list);
    clickSection.appendChild(count);
    clickSection.appendChild(listWrap);
    pages.selection.appendChild(clickSection);

    /* ── The filter page: rows, edited one at a time (§ 8.4) ─────────────── */
    const filterSection = el("div", "selection-filter-section");
    filterSection.id = "selection-filter-section";
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
        isolateBox.checked = !!state.isolate;
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
        const cell = model.getUnitCellInfo();
        for (const [label, value] of [
            ["Lattice", cell.lattice ? "set" : "none"],
            ["Origin",  cell.origin ? cell.origin.join(", ") : "—"],
            ["Axes",    cell.axisKind || "—"],
            ["Vacuum",  cell.vacuum == null ? "—" : String(cell.vacuum)],
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
