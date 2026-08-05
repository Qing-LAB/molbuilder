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
 * Two things below are hard-won knowledge about a library that punishes guessing,
 * carried from the retired embed rather than derived from the document. Both are
 * marked in place. The first cost ten rounds of "animation fixes" to find.
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

// The caption. Sized as a FRACTION of the canvas, never in fixed pixels: an
// export is several thousand pixels wide where the screen is a few hundred, and
// a fixed size that reads well in one is a speck in the other (§ 12.3).
const LABEL = {
    heightFraction:    0.055,   // of canvas height
    minPx:             11,
    marginFraction:    0.03,
    fontColor:         "#e6e9ef",
    backgroundColor:   "rgba(0,0,0,0.45)",
    backgroundOpacity: 0.45,
};

// WebGL cannot take a colour from CSS — the clear colour is an argument to the
// library while the element behind it is painted by the stylesheet. Two paints of
// one surface, so one declared value, read off the host (see _style.css). The
// literal is the last resort for a page with no stylesheet, not a second palette.
const SCENE_BACKGROUND = { name: "--vibview-scene-background", fallback: "#0f1217" };

const root = (typeof window !== "undefined") ? window : globalThis;


/* ── The module's own stylesheet (§ 13) ───────────────────────────────────────
 *
 * The MODULE links it, not the page. A template that has to remember a <link> is
 * a template that can forget one, and the viewer then mounts unstyled with
 * nothing to catch it. Resolved from this file's own URL so the module never
 * hardcodes where it is served from.
 */
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

function dataUrlToBlob(dataUrl) {
    try {
        const m = /^data:([^;,]+);base64,(.*)$/.exec(dataUrl);
        if (!m) return null;
        const bin = root.atob(m[2]);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return new root.Blob([bytes], { type: m[1] || "image/png" });
    } catch (_) { return null; }
}


/* ── One drawing surface ──────────────────────────────────────────────────── */

export function create(hostEl) {
    const $3Dmol = root.$3Dmol;
    if (!$3Dmol) {
        throw new Error("vibrationview: 3Dmol-min.js must be loaded first");
    }
    ensureStylesheet();

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
        labelText: null,
        labelObj:  null,
        ground:    ground,
        capturing: false,
    };

    function canvasEl() {
        try { return hostEl.querySelector("canvas"); } catch (_) { return null; }
    }

    function mainModel() {
        try {
            const ms = viewer.getModel ? [viewer.getModel()] : [];
            return ms[0] || null;
        } catch (_) { return null; }
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
        try { viewer.setStyle({}, STYLE); } catch (_) {}
        if (state.heldStill.length) {
            const grey = {
                stick:  { radius: STYLE.stick.radius, color: HELD_STILL_COLOR },
                sphere: { scale:  STYLE.sphere.scale,  color: HELD_STILL_COLOR },
            };
            try { viewer.setStyle({ index: state.heldStill, model: 0 }, grey); }
            catch (_) {}
        }
    }

    function paint() {
        if (state.disposed) return;
        try { viewer.render(); } catch (_) {}
    }

    /* The caption is drawn INTO the scene, not laid over it — an export reads
     * canvas pixels, and an HTML overlay would be on screen and absent from every
     * exported frame (§ 12.3). `useScreenCoordinates` pins it to the corner
     * rather than to a point in the molecule, so it stays put while the camera
     * moves. */
    function drawLabel() {
        if (state.labelObj) {
            try { viewer.removeLabel(state.labelObj); } catch (_) {}
            state.labelObj = null;
        }
        if (state.disposed || !state.labelText) return;
        const c = canvasEl();
        const h = (c && c.height) || 340;
        const size   = Math.max(LABEL.minPx, Math.round(h * LABEL.heightFraction));
        const margin = Math.round(h * LABEL.marginFraction);
        try {
            state.labelObj = viewer.addLabel(String(state.labelText), {
                useScreenCoordinates: true,
                position:          { x: margin, y: margin, z: 0 },
                alignment:         "topLeft",
                fontSize:          size,
                fontColor:         LABEL.fontColor,
                backgroundColor:   LABEL.backgroundColor,
                backgroundOpacity: LABEL.backgroundOpacity,
                inFront:           true,
            });
        } catch (_) { state.labelObj = null; }
    }

    return {
        /* Draw one structure. Establishes the model, the elements and the bond
         * topology — the things every later frame reuses. */
        setStructure(elements, positions) {
            if (state.disposed) return false;
            if (!Array.isArray(elements) || !Array.isArray(positions)
                || positions.length === 0) return false;
            try {
                viewer.removeAllModels();
                viewer.addModel(xyzText(elements, positions), "xyz");
            } catch (_) { return false; }
            restyle();
            drawLabel();
            paint();
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
            paint();
        },

        /* WHICH atoms are held still. What that looks like is this layer's. */
        setHeldStill(indices) {
            if (state.disposed) return;
            state.heldStill = Array.isArray(indices)
                ? indices.map(Number).filter((i) => i >= 0) : [];
            restyle();
            paint();
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
            state.capturing = true;
            drawLabel();          // the caption re-sizes with the new canvas
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
                state.capturing = false;
                drawLabel();
                paint();
            };
        },

        /* A picture of what is drawn, right now. No size argument: changing what
         * is drawn is beginCapture's job (§ 9.3). */
        snapshot() {
            return new Promise(function (resolve, reject) {
                if (state.disposed) { reject(new Error("snapshot: disposed")); return; }
                try {
                    const blob = dataUrlToBlob(viewer.pngURI());
                    blob ? resolve(blob)
                         : reject(new Error("snapshot: pngURI conversion failed"));
                } catch (e) {
                    reject(new Error("snapshot: " + (e && e.message)));
                }
            });
        },

        stream(fps) {
            const c = canvasEl();
            if (!c || typeof c.captureStream !== "function") return null;
            try { return c.captureStream(fps); } catch (_) { return null; }
        },

        dispose() {
            if (state.disposed) return;
            state.disposed = true;
            try { if (state.labelObj) viewer.removeLabel(state.labelObj); } catch (_) {}
            try { viewer.removeAllModels(); } catch (_) {}
            try { viewer.clear(); } catch (_) {}
            try { hostEl.innerHTML = ""; } catch (_) {}
        },
    };
}
