/* VibrationView — the sealed layer. Level 4 of docs/web/vibrationview.md § 7, § 9.3.
 *
 * Module:    lib/vibrationview/ — INTERNAL, and the most internal of all. The
 *            contract deliberately does not name this file (§ 15): no consumer
 *            names it and neither does the document. The underscore and
 *            tests/test_vibrationview_module_boundary.py are what keep that true.
 * Called by: index.js — every draw the handle causes;
 *            _export.js — the capture doors.
 *            Nothing else, ever.
 *
 * The ONLY file in this module that names 3Dmol. Nothing above it knows the
 * library exists, which is what makes "the graphics library is invisible" (§ 5.4)
 * a property of the code rather than a habit.
 *
 * It answers NO question about the structure, the frame, or the camera (§ 9.3).
 * What "held still" and the caption LOOK like is decided here and nowhere else:
 * the layer above says which atoms are held still and what the caption says,
 * never what colour or font either should be.
 *
 * ── Carried, not invented ────────────────────────────────────────────────────
 * Three things below are hard-won knowledge about a library that punishes
 * guessing. Two are carried from the retired embed rather than derived from the
 * document; the third was MEASURED against a real browser after the document
 * asserted the opposite. All three are marked in place. The first cost ten rounds
 * of "animation fixes" to find.
 */
"use strict";


/* ── Appearance, decided HERE ─────────────────────────────────────────────── */

// A vibration is read by watching atoms move relative to each other, so the
// representation shows every bond and hides nothing behind a big sphere.
const STYLE = { stick: { radius: 0.15 }, sphere: { scale: 0.25 } };

// Held-still atoms are drawn dead: no element colour, so the eye separates what
// the calculation let move from what it pinned. A COLOUR, decided here — the
// layer above sends indices and no appearance at all (§ 9.3).
const HELD_STILL_COLOR = "#555";

/* The caption is a DOM overlay, and _style.css decides what it looks like — so
 * there are no colours for it here. These are only what the COMPOSITE needs in
 * order to reproduce on a 2-D canvas what the browser drew with CSS, and each is
 * read off the live element (`getComputedStyle`) rather than declared twice. */
const CAPTION_CLASS = "vibview-caption";

// WebGL cannot take a colour from CSS — the clear colour is an argument to the
// library while the element behind it is painted by the stylesheet. Two paints of
// one surface, so one declared value, read off the host (see _style.css). The
// literal is the last resort for a page with no stylesheet, not a second palette.
const SCENE_BACKGROUND = { name: "--vibview-scene-background", fallback: "#0f1217" };

const root = (typeof window !== "undefined") ? window : globalThis;


/* The module links its own stylesheet (§ 13 says why), resolved from this file's
 * own URL so it never hardcodes where it is served from. */
const STYLESHEET_ID = "vibrationview-style";

function ensureStylesheet() {
    if (typeof document === "undefined") return;
    if (document.getElementById(STYLESHEET_ID)) return;
    try {
        const link = document.createElement("link");
        link.id   = STYLESHEET_ID;
        link.rel  = "stylesheet";
        link.href = new URL("./_style.css", import.meta.url).href;
        document.head.appendChild(link);
    } catch (_) { /* a page without a head is a test, not a browser */ }
}


function readCssVar(el, name, fallback) {
    if (!el || typeof getComputedStyle !== "function") return fallback;
    try {
        const v = getComputedStyle(el).getPropertyValue(name).trim();
        return v || fallback;
    } catch (_) { return fallback; }
}

function xyzText(elements, positions) {
    const lines = [String(positions.length), "vibrationview"];
    for (let i = 0; i < positions.length; i++) {
        const p = positions[i];
        lines.push((elements[i] || "C") + " " + p[0] + " " + p[1] + " " + p[2]);
    }
    return lines.join("\n");
}

/* ── One drawing surface ──────────────────────────────────────────────────── */

export function create(hostEl) {
    const $3Dmol = root.$3Dmol;
    if (!$3Dmol) {
        throw new Error("vibrationview: 3Dmol-min.js must be loaded first");
    }
    ensureStylesheet();
    // The class is the sheet's hook. Added here rather than asked of the host,
    // for the same reason the <link> is (§ 13): a page that has to remember it
    // can forget it, and the failure is silent.
    try { hostEl.classList.add("vibview"); } catch (_) {}

    const ground = readCssVar(hostEl, SCENE_BACKGROUND.name, SCENE_BACKGROUND.fallback);
    const viewer = $3Dmol.createViewer(hostEl, {
        backgroundColor: ground,
        defaultcolors:   $3Dmol.elementColors.Jmol,
    });

    // Everything this layer holds. Note what is NOT here: a frame number, a
    // phase, an amplitude, a mode. It draws what it is handed (§ 9.3).
    const state = {
        disposed:  false,
        heldStill: [],       // indices, so a coord change can re-apply the grey
        labelText: null,     // what the caption says; the layer above owns the words
        captionEl: null,     // the DOM overlay that shows them (§ 12.3)
        composite: null,     // the 2-D canvas an export reads (see compose())
        ground:    ground,
    };

    function canvasEl() {
        try { return hostEl.querySelector("canvas"); } catch (_) { return null; }
    }

    function mainModel() {
        try { return viewer.getModel() || null; } catch (_) { return null; }
    }

    /* CARRIED KNOWLEDGE (1 of 2) — THE ONE THAT MATTERS.
     *
     * 3Dmol caches its representation meshes (stick cylinders, sphere instances)
     * at `setStyle` time. Writing `atom.x/y/z` afterwards updates the data model
     * and NOT the scene: `render()` redraws the OLD mesh, so the molecule sits
     * perfectly still while the coordinates advance underneath it. Re-applying
     * the style spec is what regenerates the geometry at the new positions.
     *
     * This is why § 10's "a tick moves atoms, it does not rebuild the drawing"
     * is true of the MODEL and not of the meshes: the parse, the element
     * identities and the bond topology are established once, and the per-frame
     * cost is one style pass over the atoms — O(atoms), not O(1). At the few
     * hundred atoms this project targets that is the difference between smooth
     * and smooth; it is not free, and pretending otherwise is how the animation
     * stopped moving for ten rounds of fixes before anyone found it.
     */
    function restyle() {
        // Always followed by a paint, so it does its own: three call sites wrote
        // the pair, which is three chances to write only half of it — and half of
        // it is a style rebuilt and never shown.
        try { viewer.setStyle({}, STYLE); } catch (_) {}
        if (state.heldStill.length) {
            const grey = {
                stick:  { radius: STYLE.stick.radius, color: HELD_STILL_COLOR },
                sphere: { scale:  STYLE.sphere.scale,  color: HELD_STILL_COLOR },
            };
            try { viewer.setStyle({ index: state.heldStill, model: 0 }, grey); }
            catch (_) {}
        }
        paint();
    }

    function paint() {
        if (state.disposed) return;
        try { viewer.render(); } catch (_) {}
    }

    /* CARRIED KNOWLEDGE (3 of 3), and this one was measured rather than
     * remembered. The caption is a DOM overlay, NOT a mark inside the 3D scene.
     *
     * The design called for a label drawn into the canvas, so that it would ride
     * into an exported picture for free. Against a real browser that turns out to
     * be impossible with this library: a `useScreenCoordinates` label draws
     * NOTHING — `addLabel` returns an object, `render()` runs, and the pixels are
     * byte-identical with it and without it, on screen and in `pngURI` alike —
     * while a scene-positioned label does draw, and rides the camera, which is
     * not a caption.
     *
     * So the caption is a div over the canvas, exactly as MolView's corner badge
     * has been for months, and an export COMPOSITES it in on purpose (compose()).
     * The rule the design wanted survives — the caption is in the file — and only
     * the mechanism changed.
     */
    function drawLabel() {
        if (state.disposed) return;
        if (!state.captionEl) {
            // Nothing to say and nothing built yet: build nothing. A viewer that
            // never carries a caption should not carry an empty element for one.
            if (!state.labelText) return;
            if (typeof document === "undefined") return;
            const el = document.createElement("div");
            el.className = CAPTION_CLASS;
            el.hidden = true;
            try { hostEl.appendChild(el); } catch (_) { return; }
            state.captionEl = el;
        }
        const el = state.captionEl;
        el.textContent = state.labelText || "";
        el.hidden = !state.labelText;
    }

    /* ONE composited picture, and both ways out read it (§ 5.3).
     *
     * The drawing library paints a WebGL canvas; the caption is a div beside it.
     * A picture of "what is on screen" is therefore both, drawn onto one 2-D
     * canvas — and doing it in one place is what keeps a snapshot and a recorded
     * frame from being two different pictures.
     *
     * The caption is drawn from the computed style of the live element, so the
     * stylesheet stays the ONE place its appearance is declared (§ 13), and it is
     * scaled by how much bigger the capture is than the screen — an export
     * several thousand pixels wide would otherwise carry a caption sized for a
     * box a few hundred wide, which is the speck § 12.3 warns about. */
    function compose() {
        const src = canvasEl();
        if (!src) return null;
        if (!state.composite) {
            if (typeof document === "undefined") return null;
            state.composite = document.createElement("canvas");
        }
        const out = state.composite;
        /* Assigning `width` resets the drawing surface even when the value has not
         * changed — that is what the property does, not a quirk. During a
         * recording this runs once per frame, so assigning unconditionally would
         * buy a full canvas reallocation per frame in the path that is already the
         * expensive one. */
        if (out.width !== src.width)   out.width = src.width;
        if (out.height !== src.height) out.height = src.height;
        const g = out.getContext("2d");
        if (!g) return null;
        g.clearRect(0, 0, out.width, out.height);
        g.drawImage(src, 0, 0);

        const el = state.captionEl;
        if (el && !el.hidden && el.textContent) {
            let cs = null;
            try { cs = getComputedStyle(el); } catch (_) {}
            // How much bigger the captured buffer is than the element on screen.
            const shown = (hostEl.clientWidth || src.width) || 1;
            const k = src.width / shown;
            const fontPx = (parseFloat(cs && cs.fontSize) || 12) * k;
            const padX   = (parseFloat(cs && cs.paddingLeft) || 6) * k;
            const padY   = (parseFloat(cs && cs.paddingTop) || 3) * k;
            // WHERE it sits is the stylesheet's too, read like everything else.
            // Repeating the corner here as a literal would mean moving the
            // caption in CSS and having the exported one stay where it was.
            const x      = (parseFloat(cs && cs.left) || 8) * k;
            const y      = (parseFloat(cs && cs.top)  || 8) * k;

            g.font = fontPx + "px " + ((cs && cs.fontFamily) || "sans-serif");
            g.textBaseline = "top";
            const w = g.measureText(el.textContent).width;
            g.fillStyle = (cs && cs.backgroundColor) || "rgba(15,18,23,0.62)";
            g.fillRect(x, y, w + 2 * padX, fontPx * 1.35 + 2 * padY);
            g.fillStyle = (cs && cs.color) || "#e6e9ef";
            g.fillText(el.textContent, x + padX, y + padY);
        }
        return out;
    }

    return {
        /* Draw one structure. Establishes the model, the elements and the bond
         * topology — the things every later frame reuses. */
        /* The shapes were checked by the one caller this layer has (§ 9.3: called
         * by index.js and nothing else, ever). Checking again here would not make
         * anything safer — it would make one rule live in two places. */
        setStructure(elements, positions) {
            if (state.disposed) return false;
            try {
                viewer.removeAllModels();
                viewer.addModel(xyzText(elements, positions), "xyz");
            } catch (_) { return false; }
            restyle();
            return true;
        },

        /* Move the atoms already drawn. The style pass is not optional — see the
         * carried note above. */
        setAtomCoords(coords) {
            if (state.disposed || !Array.isArray(coords)) return;
            try {
                const model = mainModel();
                const atoms = model ? model.selectedAtoms({}) : [];
                const n = Math.min(atoms.length, coords.length);
                for (let i = 0; i < n; i++) {
                    const c = coords[i];
                    if (!c) continue;
                    atoms[i].x = c[0];
                    atoms[i].y = c[1];
                    atoms[i].z = c[2];
                }
            } catch (_) { return; }
            restyle();
        },

        /* WHICH atoms are held still. What that looks like is this layer's. */
        setHeldStill(indices) {
            if (state.disposed) return;
            state.heldStill = Array.isArray(indices)
                ? indices.map(Number).filter((i) => i >= 0) : [];
            restyle();
        },

        setLabel(text) {
            if (state.disposed) return;
            state.labelText = (text === null || text === undefined || text === "")
                ? null : String(text);
            drawLabel();
            paint();
        },

        refit() {
            if (state.disposed) return;
            try { viewer.zoomTo(); } catch (_) {}
            paint();
        },

        /* CARRIED KNOWLEDGE (2 of 2). Getting an export's size means resizing the
         * REAL canvas — the picture comes from the surface that is already there,
         * not a second one built for the occasion. `viewer.resize()` is the entry
         * that updates both the WebGL viewport and the canvas pixel buffer;
         * setting `canvas.width` alone leaves the viewport at the old size and the
         * render comes back letterboxed.
         *
         * Returns the undo rather than remembering what to undo, so nothing ever
         * has to ASK this layer how big it was or what colour it had — questions
         * § 9.3 refuses like all the others. */
        beginCapture(opts) {
            if (state.disposed) return function () {};
            opts = opts || {};
            const c = canvasEl();
            const origW = c ? c.width : 0;
            const origH = c ? c.height : 0;
            const origGround = state.ground;
            let resized = false;

            if (c && opts.width > 0 && opts.height > 0) {
                try {
                    c.width  = Math.round(opts.width);
                    c.height = Math.round(opts.height);
                    if (typeof viewer.resize === "function") viewer.resize();
                    resized = true;
                } catch (_) { resized = false; }
            }
            if (opts.background !== undefined && opts.background !== null) {
                try {
                    if (opts.background === "transparent") {
                        viewer.setBackgroundColor(0x000000, 0);
                    } else {
                        viewer.setBackgroundColor(opts.background, 1);
                    }
                    state.ground = opts.background;
                } catch (_) {}
            }
            paint();

            return function endCapture() {
                if (state.disposed) return;
                if (resized && c) {
                    try {
                        c.width = origW; c.height = origH;
                        if (typeof viewer.resize === "function") viewer.resize();
                    } catch (_) {}
                }
                if (state.ground !== origGround) {
                    try { viewer.setBackgroundColor(origGround, 1); } catch (_) {}
                    state.ground = origGround;
                }
                paint();
            };
        },

        /* A picture of what is on screen, right now — the molecule AND its
         * caption. No size argument: changing what is drawn is beginCapture's job
         * (§ 9.3). */
        snapshot() {
            return new Promise(function (resolve, reject) {
                if (state.disposed) { reject(new Error("snapshot: disposed")); return; }
                const out = compose();
                if (!out) { reject(new Error("snapshot: nothing drawn")); return; }
                try {
                    out.toBlob(function (b) {
                        b ? resolve(b) : reject(new Error("snapshot: toBlob returned null"));
                    }, "image/png");
                } catch (e) {
                    reject(new Error("snapshot: " + (e && e.message)));
                }
            });
        },

        /* The video stream comes off the COMPOSITE, not the drawing library's own
         * canvas — otherwise a recording would be the one export that silently
         * lost the caption. The caller repaints it per frame via `recompose`. */
        stream(fps) {
            const out = compose();
            if (!out || typeof out.captureStream !== "function") return null;
            try { return out.captureStream(fps); } catch (_) { return null; }
        },

        recompose() { compose(); },

        dispose() {
            if (state.disposed) return;
            state.disposed = true;
            try { if (state.captionEl) state.captionEl.remove(); } catch (_) {}
            state.captionEl = null;
            state.composite = null;
            try { viewer.removeAllModels(); } catch (_) {}
            try { viewer.clear(); } catch (_) {}
            try { hostEl.innerHTML = ""; } catch (_) {}
        },
    };
}
