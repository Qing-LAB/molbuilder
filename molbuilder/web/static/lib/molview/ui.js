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

    const prev = el("button", "molview-frame-btn");
    prev.type = "button"; prev.textContent = "‹";
    prev.setAttribute("aria-label", "Previous frame");

    const playBtn = el("button", "molview-frame-btn molview-frame-play");
    playBtn.type = "button"; playBtn.textContent = "▶";
    playBtn.setAttribute("aria-label", "Play");

    const next = el("button", "molview-frame-btn");
    next.type = "button"; next.textContent = "›";
    next.setAttribute("aria-label", "Next frame");

    const transport = el("div", "molview-frame-transport");
    transport.appendChild(prev);
    transport.appendChild(playBtn);
    transport.appendChild(next);

    const slider = el("input", "molview-frame-slider");
    slider.type = "range";
    slider.min = "0";
    slider.step = "1";
    slider.setAttribute("aria-label", "Frame");

    const counter = el("span", "molview-frame-counter");

    const loopWrap = el("label", "molview-frame-loop");
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
    root.className = "mol-viewer-knob mol-viewer-menu mol-viewer-menu-view";
    const summary = doc.createElement("summary");
    summary.textContent = "View";
    root.appendChild(summary);

    const section = (heading) => {
        const wrap = doc.createElement("div");
        wrap.className = "mol-viewer-menu-section";
        const label = doc.createElement("div");
        label.className = "mol-viewer-menu-heading";
        label.textContent = heading;
        wrap.appendChild(label);
        root.appendChild(wrap);
        return wrap;
    };

    const offs = [];

    // ── The switches: they change WHAT IS IN a frame (§ 9.6) ──────────────
    const shown = section("Show");
    const SWITCHES = [
        ["showIndex", "Atom numbers"],
        ["showForces", "Force arrows"],
        ["showCell",  "Unit cell"],
        ["showAxis",  "Axes"],
    ];
    const boxes = {};
    for (const [name, label] of SWITCHES) {
        const row = doc.createElement("label");
        row.className = "mol-viewer-menu-row";
        const box = doc.createElement("input");
        box.type = "checkbox";
        box.addEventListener("change", (e) => {
            model.selection.setSwitch(name, !!e.target.checked);
        });
        row.appendChild(box);
        row.appendChild(doc.createTextNode(" " + label));
        shown.appendChild(row);
        boxes[name] = box;
    }
    // The switches have one home, so the menu reflects them rather than
    // remembering what it last set (§ 5.2).
    offs.push(model.selection.subscribe((state) => {
        for (const [name] of SWITCHES) boxes[name].checked = !!state[name];
    }));

    // ── The drawing settings: they change HOW THE SAME FRAME IS PAINTED ───
    const drawn = section("Draw as");
    const style = doc.createElement("select");
    style.className = "mol-viewer-menu-select";
    for (const [value, label] of [["stick", "Sticks"],
                                  ["ball-and-stick", "Ball & stick"],
                                  ["sphere", "Spheres"],
                                  ["line", "Lines"]]) {
        const option = doc.createElement("option");
        option.value = value; option.textContent = label;
        style.appendChild(option);
    }
    style.addEventListener("change", (e) => model.view.set("style", e.target.value));
    drawn.appendChild(style);

    const projection = doc.createElement("label");
    projection.className = "mol-viewer-menu-row";
    const orthoBox = doc.createElement("input");
    orthoBox.type = "checkbox";
    orthoBox.addEventListener("change", (e) => {
        model.view.set("orthographic", !!e.target.checked);
    });
    projection.appendChild(orthoBox);
    projection.appendChild(doc.createTextNode(" Orthographic"));
    drawn.appendChild(projection);

    offs.push(model.view.subscribe((settings) => {
        style.value = settings.style;
        orthoBox.checked = !!settings.orthographic;
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
    root.className = "mol-viewer-knob mol-viewer-menu mol-viewer-menu-export";
    const summary = doc.createElement("summary");
    summary.textContent = "Export";
    root.appendChild(summary);

    const item = (label, onClick) => {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "mol-viewer-menu-item";
        button.textContent = label;
        button.addEventListener("click", onClick);
        root.appendChild(button);
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
    badge.className = "molview-overlay molview-overlay-badge molview-corner-tr";
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
    readout.className = "molview-overlay molview-overlay-readout molview-corner-bl";
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
