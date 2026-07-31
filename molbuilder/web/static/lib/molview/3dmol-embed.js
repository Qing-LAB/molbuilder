/* MolView — the sealed layer: the only code in the module that names 3Dmol.
 *
 * Contract: docs/web/molview.md § 9.9, § 9.8, § 7 level 7, § 10.6, § 10.7.
 * Owns:     the DRAWING COPY — the movie, the camera, the styles, the picking,
 *           the highlight spheres. It draws the frame it is handed. (The camera
 *           is here because a window must have a point of view; § 9.6 is why
 *           nothing above it keeps one.)
 * Called by: level 6 — the drawing commands — and nothing else, ever.
 *
 * It answers EXACTLY TWO questions, and both are about itself (§ 9.9):
 *   hasMovie()         — is there a movie loaded at all?
 *   drawnFrameCount()  — how many frames does it have?
 * Both exist so the layer above can find out whether its own last instruction
 * landed (§ 10.10). Neither answer ever reaches a user.
 *
 * NEVER (§ 7 level 7):
 *   - keep its own frame number, or be a source of truth about coordinates;
 *   - offer any way to read coordinates out, to ask which frame is showing, or
 *     to ask where the camera is pointing. "Everything else, it refuses."
 *
 * The line is not a loophole: "did what I told you to do land?" is a check;
 * "what is the structure?" is a question about the truth, and the truth is not
 * here. Everything this layer holds is either DERIVED (the drawing copy, worked
 * out from the master copy every redraw) or GIVEN to it.
 *
 * ── Carried, not invented ─────────────────────────────────────────────────────
 * Step B of the rebuild. This is the one file NOT written from the document: the
 * bodies below are carried from the frozen tree because they are hard-won
 * knowledge about a library that punishes guessing. The three that cost the most
 * to learn are marked in place — the setStyle mesh cache, the native movie, and
 * the one-GLShape arrow batch.
 *
 * Carved OUT and up into the layers above: the card scaffold and info line
 * (mount.js); the knob bar, frame strip, animation interval, export menu,
 * snapshot and GIF encoder (ui.js, § 11.4). The option-normalising layer went
 * with them — this layer is handed finished data and normalises nothing.
 * DELETED outright: the `molbuilder.projects` reach and the `/api/files/*`
 * calls — a file route at the bottom of the stack (§ 6.7), and task #39.
 *
 * The axis triad went UP as well. Its geometry is worked out from the cell
 * (§ 10.3, "the cell box and the axes are worked out once"), which is maths, and
 * it arrives here as ordinary overlay arrows carrying their own colours.
 */
"use strict";


/* ── Appearance constants owned by THIS layer ─────────────────────────────────
 *
 * § 6.5: per-frame data says which arrows exist and where they point, and
 * nothing else. What they LOOK like is decided here, from the set handed in — a
 * `color` or a `radius` riding on per-frame data is the defect that rule exists
 * to catch.
 */

// The largest force in the set draws gold and the rest ramp dim-red -> orange-red
// by RELATIVE magnitude, the shaft thickening the same way, so the eye lands on
// the atom under the most force — the relaxation hot spot.
const FORCE_ARROW = {
    maxColor:   "#ffc400",   // gold: the single largest arrow in the set
    radiusMin:  0.05,        // Å, at zero relative magnitude
    radiusSpan: 0.04,        // Å, added at the largest
};

// The selection glow. A SHAPE, never a restyle (§ 10.7): restyling to show a
// selection rebuilds the whole model's geometry, so its cost grows with the
// structure; a sphere per picked atom does not.
const SELECTION_GLOW = { color: "#ffd54a", radius: 0.7, opacity: 0.5 };

const LABEL_STYLE = {
    fontSize:          12,
    fontColor:         "#222",
    backgroundColor:   "rgba(255,255,255,0.7)",
    backgroundOpacity: 0.7,
    inFront:           true,
};

const CELL_FALLBACK = { color: "#888", radius: 0.04 };

const root = (typeof window !== "undefined") ? window : globalThis;


/* ── The style spec ──────────────────────────────────────────────────────────
 *
 * § 9.6's four drawing settings turned into what the library wants. The only
 * place a representation name becomes a library concept.
 */
function styleSpec(view) {
    const rep   = view.rep || "stick";
    const scale = view.radiusScale || 1.0;
    const cs    = view.colorScheme;
    // Drop the colorscheme key when absent so the library falls back to its
    // viewer-level defaultcolors rather than being handed an empty one.
    const color = cs ? { colorscheme: cs } : {};
    switch (rep) {
        case "sphere":
            return { sphere: { scale: 1.0 * scale, ...color } };          // true CPK
        case "ball-and-stick":
            return {
                stick:  { radius: 0.15 * scale, ...color },
                sphere: { scale:  0.3  * scale, ...color },
            };
        case "line":
            return { line: { linewidth: 2 * scale, ...color } };
        case "cartoon":
            return { cartoon: { ...color } };
        default:
            return { stick: { radius: 0.15 * scale, ...color } };
    }
}


/* ── Creating one embed ──────────────────────────────────────────────────────
 *
 * Returns the surface of § 9.8 and nothing else. There is no accessor for the
 * 3Dmol object: a consumer that could reach it would make § 5.3 false, and every
 * "just for tests" hatch that ever existed here became a production read.
 */
export function create(hostEl, opts) {
    const $3Dmol = root.$3Dmol;
    if (!$3Dmol) {
        throw new Error("3dmol-embed: 3Dmol-min.js must be loaded first");
    }
    opts = opts || {};

    const viewer = $3Dmol.createViewer(hostEl, {
        backgroundColor: (opts.view && opts.view.background) || "white",
        defaultcolors:   $3Dmol.elementColors.Jmol,
    });

    // Everything this layer holds. Note what is NOT here: a frame number. The
    // library owns the playhead of its own movie; asking it back would be the
    // second home § 7 forbids.
    const state = {
        viewer:      viewer,
        hostEl:      hostEl,
        disposed:    false,
        view:        Object.assign({ rep: "stick", radiusScale: 1, background: "white" },
                                   opts.view || {}),
        // Shape handles we own and must remove before redrawing.
        cellShapes:      [],
        arrowShapes:     [],
        arrowLabels:     [],
        labelHandles:    [],
        markerShapes:    [],
        haloShapes:      [],
        highlightShapes: [],
        // What the overlays currently say, so a frame swap can re-place them
        // (§ 10.6: shapes move with the frames) without being re-sent.
        overlays:    { labels: [], markers: [], halos: [], highlight: [] },
        arrows:      [],
        pickHandler: null,
        pickWired:   false,
        renderDepth: 0,
        renderDirty: false,
    };

    /* ── The one render primitive ──────────────────────────────────────────
     *
     * Every draw site goes through here. While a batch is open it only marks
     * the scene dirty; the paint fires once at the outermost endBatch, not once
     * per door touched (§ 10.1, one render place).
     */
    function paint() {
        if (state.disposed || !state.viewer) return;
        if (state.renderDepth > 0) { state.renderDirty = true; return; }
        try { state.viewer.render(); } catch (_) {}
    }

    // The one structure model. The embed keeps exactly one — the glow is a
    // SHAPE, not a second model — so this names the intent rather than
    // scattering bare getModel(0) calls.
    function mainModel() {
        return (state.viewer && typeof state.viewer.getModel === "function")
            ? state.viewer.getModel(0) : null;
    }

    // The drawn atoms, for placing our own shapes on. Internal only: nothing
    // derived from this ever crosses back upward (§ 9.9).
    function drawnAtoms() {
        try {
            const m = mainModel();
            return m ? (m.selectedAtoms({}) || []) : [];
        } catch (_) { return []; }
    }

    function clear(list, kind) {
        const remove = kind === "label" ? "removeLabel" : "removeShape";
        for (const h of list) {
            try { state.viewer[remove](h); } catch (_) {}
        }
        return [];
    }

    /* ── Styles ────────────────────────────────────────────────────────────
     *
     * CARRIED KNOWLEDGE (1 of 3). 3Dmol caches its representation meshes —
     * stick cylinders, sphere instances, line segments — at setStyle time.
     * Mutating atom.x afterwards updates the data model but the VISIBLE
     * geometry stays put until something forces a rebuild. Re-applying the
     * current style spec is the cheapest regeneration: setStyle({}, spec)
     * rebuilds with the new positions in one pass.
     *
     * This is why § 10.5 puts a coordinate change and a switch flip in
     * different cost bands, and it is the bug class that shipped through ten
     * rounds of "animation fixes" before anyone noticed the frames were
     * advancing in the data and standing still on screen.
     */
    function applyStyle() {
        try { state.viewer.setStyle({ model: 0 }, styleSpec(state.view)); } catch (_) {}
        if (state.view.background) {
            try { state.viewer.setBackgroundColor(state.view.background); } catch (_) {}
        }
    }

    /* ── Overlay redraws ───────────────────────────────────────────────────
     *
     * Each takes positions from the DRAWN atoms, which is what makes § 10.6
     * true: after a frame swap the shapes sit on the atoms' new positions
     * without anything above re-sending them. The text of a label still comes
     * from above, because under isolate it must carry the ORIGINAL atom number
     * (§ 6.5) and this layer has no idea what that is.
     */
    function redrawLabels() {
        state.labelHandles = clear(state.labelHandles, "label");
        const atoms = drawnAtoms();
        for (const l of state.overlays.labels) {
            const a = atoms[l.atom];
            if (!a) continue;
            try {
                state.labelHandles.push(state.viewer.addLabel(String(l.text), {
                    position: { x: a.x, y: a.y, z: a.z },
                    ...LABEL_STYLE,
                }));
            } catch (_) {}
        }
    }

    function redrawMarkers() {
        state.markerShapes = clear(state.markerShapes);
        const atoms = drawnAtoms();
        for (const m of state.overlays.markers) {
            const a = atoms[m.atom];
            if (!a) continue;
            try {
                state.markerShapes.push(state.viewer.addSphere({
                    center:  { x: a.x, y: a.y, z: a.z },
                    radius:  m.radius,
                    color:   m.color,
                    opacity: m.opacity == null ? 1 : m.opacity,
                }));
            } catch (_) {}
        }
    }

    function redrawHalos() {
        state.haloShapes = clear(state.haloShapes);
        const atoms = drawnAtoms();
        for (const h of state.overlays.halos) {
            const a = atoms[h.atom];
            if (!a) continue;
            try {
                state.haloShapes.push(state.viewer.addSphere({
                    center:  { x: a.x, y: a.y, z: a.z },
                    radius:  h.radius,
                    color:   h.color,
                    opacity: h.opacity == null ? 0.5 : h.opacity,
                }));
            } catch (_) {}
        }
    }

    // § 10.7: a click adds or removes SHAPES and issues no model restyle, so its
    // cost does not grow with the structure.
    function redrawHighlight() {
        state.highlightShapes = clear(state.highlightShapes);
        const atoms = drawnAtoms();
        for (const i of state.overlays.highlight) {
            const a = atoms[i];
            if (!a) continue;
            try {
                state.highlightShapes.push(state.viewer.addSphere({
                    center:  { x: a.x, y: a.y, z: a.z },
                    radius:  SELECTION_GLOW.radius,
                    color:   SELECTION_GLOW.color,
                    opacity: SELECTION_GLOW.opacity,
                }));
            } catch (_) {}
        }
    }

    // |end - start| — the drawn length. The ramp uses the RATIO to the largest,
    // so whatever scale the caller applied cancels out.
    function arrowMagnitude(a) {
        const dx = a.end[0] - a.start[0];
        const dy = a.end[1] - a.start[1];
        const dz = a.end[2] - a.start[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /* CARRIED KNOWLEDGE (2 of 3). Every arrow is batched into ONE GLShape:
     * addArrow returns a shape, and every subsequent arrow appends to that same
     * shape via shape.addArrow, per-arrow colour preserved as vertex colour. A
     * whole overlay becomes one scene object and one geometry instead of N —
     * measured ~7x faster per frame (81 arrows: ~70ms -> ~10ms). A stand-in
     * whose addArrow return lacks .addArrow falls back to one shape per arrow.
     */
    function redrawArrows() {
        state.arrowShapes = clear(state.arrowShapes);
        state.arrowLabels = clear(state.arrowLabels, "label");
        const arrows = state.arrows;

        // Rank first: gold goes to the largest, everything else ramps against it.
        let maxMag = 0, maxAt = -1;
        for (let i = 0; i < arrows.length; i++) {
            const a = arrows[i];
            if (!a || !a.start || !a.end) continue;
            const m = arrowMagnitude(a);
            if (m > maxMag) { maxMag = m; maxAt = i; }
        }

        let batch = null;
        for (let i = 0; i < arrows.length; i++) {
            const a = arrows[i];
            if (!a || !a.start || !a.end) continue;
            const t = maxMag > 0 ? arrowMagnitude(a) / maxMag : 0;
            // An arrow that arrives WITH a colour keeps it: the axis triad hands
            // its own per-axis colours through this same door, and those are its
            // to choose. Force arrows arrive without one and are coloured here.
            const color = a.color
                || (i === maxAt ? FORCE_ARROW.maxColor : rampColor(t));
            const radius = typeof a.radius === "number"
                ? a.radius
                : FORCE_ARROW.radiusMin + FORCE_ARROW.radiusSpan * t;
            const spec = {
                start:       { x: a.start[0], y: a.start[1], z: a.start[2] },
                end:         { x: a.end[0],   y: a.end[1],   z: a.end[2] },
                radius:      radius,
                radiusRatio: 2.5,
                mid:         0.85,
                color:       color,
            };
            try {
                if (batch && typeof batch.addArrow === "function") {
                    batch.addArrow(spec);
                } else {
                    const shape = state.viewer.addArrow(spec);
                    state.arrowShapes.push(shape);
                    if (shape && typeof shape.addArrow === "function") batch = shape;
                }
            } catch (_) { continue; }
            if (a.label) {
                try {
                    state.arrowLabels.push(state.viewer.addLabel(String(a.label), {
                        position: {
                            x: a.end[0] * 1.05,
                            y: a.end[1] * 1.05,
                            z: a.end[2] * 1.05,
                        },
                        ...LABEL_STYLE,
                    }));
                } catch (_) {}
            }
        }
    }

    function rampColor(t) {   // t in [0,1]: dim-red -> orange-red
        return "rgb(" + Math.floor(170 + 85 * t) + "," + Math.floor(40 + 60 * t) + ",32)";
    }

    // Read a CSS custom property off the card, where these are declared and
    // concealed to the module — 3D-scene constants WebGL cannot style directly,
    // kept in one discoverable place instead of as magic numbers.
    function cssVar(name, fallback) {
        if (!state.hostEl || typeof getComputedStyle !== "function") return fallback;
        try {
            const v = getComputedStyle(state.hostEl).getPropertyValue(name).trim();
            return v || fallback;
        } catch (_) { return fallback; }
    }

    // Everything the overlays own, re-placed at whatever frame is now drawn.
    // Cheap — one frame's worth of shapes, not a movie rebuild. The cell is
    // lattice-only and static, so it is deliberately not in here.
    function replaceOverlays() {
        redrawLabels();
        redrawMarkers();
        redrawHalos();
        redrawHighlight();
    }

    /* ── Picking ───────────────────────────────────────────────────────────
     *
     * A click is the user's input ENTERING at the bottom, not this layer
     * answering a question — it reports which atom was hit and holds no notion
     * of what is selected. What a click means is § 9.5's business, above.
     */
    function wirePick() {
        if (state.pickWired) return;
        try {
            state.viewer.setClickable({ model: 0 }, true, function (atom) {
                if (state.disposed || !state.pickHandler) return;
                const idx = atom && (atom.index != null ? atom.index : atom.serial);
                if (typeof idx !== "number") return;
                try { state.pickHandler(idx); } catch (_) {}
            });
            state.pickWired = true;
        } catch (_) {}
    }

    /* CARRIED KNOWLEDGE (3 of 3). A trajectory is loaded ONCE as a native
     * multi-frame model: the library parses every frame and computes bonds a
     * single time, and setFrame(i) then swaps to a pre-parsed frame with NO
     * setStyle rebuild. Measured ~4ms/frame against ~50ms for the old
     * "overwrite one model and restyle" path — which is what § 10.4 means by
     * playing being a frame swap and not a redraw.
     *
     * It also makes the library the single owner of the frame COORDINATES, so
     * this layer never needs a second copy to do its job.
     */
    function multiFrameXyz(elements, frames) {
        const n = elements.length;
        const parts = [];
        for (let f = 0; f < frames.length; f++) {
            const fr = frames[f];
            parts.push(String(n));
            parts.push("");
            for (let i = 0; i < n; i++) {
                const c = fr[i] || [0, 0, 0];
                parts.push((elements[i] || "C") + " " + c[0] + " " + c[1] + " " + c[2]);
            }
        }
        return parts.join("\n");
    }

    return {
        /* ── Loading and playing the movie ─────────────────────────────── */

        // Replace whatever is drawn with a movie of these frames. One frame is
        // not a special case (§ 6.1) — it is a one-frame movie, so every path
        // below stays the same whether there is one frame or four hundred.
        loadFrames(elements, frames) {
            if (state.disposed) return false;
            if (!Array.isArray(elements) || !elements.length) return false;
            if (!Array.isArray(frames) || !frames.length) return false;
            const xyz = multiFrameXyz(elements, frames);
            try {
                state.viewer.removeAllModels();
                state.viewer.addModelsAsFrames(xyz, "xyz");
            } catch (_) {
                return false;
            }
            // Style once — it carries across every native frame. Then
            // re-establish clickability, which the model swap invalidated.
            applyStyle();
            state.pickWired = false;
            wirePick();
            replaceOverlays();
            redrawArrows();
            paint();
            return true;
        },

        // Push frames onto the movie without rebuilding it: clone the frame-0
        // atom template — which keeps element identity and bond topology — and
        // stamp the new coordinates onto the copy.
        appendFrames(frames) {
            if (state.disposed || !Array.isArray(frames) || !frames.length) return false;
            const model = mainModel();
            if (!model) return false;
            let template;
            try {
                template = model.selectedAtoms({});
                if (!template || !template.length) return false;
            } catch (_) { return false; }
            for (const coords of frames) {
                const atoms = template.map(function (at, i) {
                    const c = (coords && coords[i]) || [0, 0, 0];
                    const na = Object.assign({}, at);
                    na.x = c[0]; na.y = c[1]; na.z = c[2];
                    return na;
                });
                try {
                    if (typeof model.addFrame === "function") model.addFrame(atoms);
                    else model.frames.push(atoms);
                } catch (_) { return false; }
            }
            return true;
        },

        // Swap to a pre-parsed frame. The index is NOT kept: which frame is
        // showing is the model's fact (§ 7 level 7, § 6.4), and a copy here
        // would be the second home that drifts.
        showFrame(i) {
            if (state.disposed) return;
            try { state.viewer.setFrame(i); } catch (_) {}
            replaceOverlays();
            paint();
        },

        /* ── The things drawn beside the molecule ──────────────────────── */

        setStyle(view) {
            if (state.disposed) return;
            Object.assign(state.view, view || {});
            applyStyle();
            // A drawing setting derives nothing (§ 10.1): the style is
            // re-applied and the movie is left exactly as it was.
            paint();
        },

        setProjection(p) {
            if (state.disposed) return;
            try { state.viewer.setProjection(p); } catch (_) {}
            paint();
        },

        // The cell is lattice + the world-space CORNER the box is anchored at.
        // For a bbox-plus-vacuum cell the anchor is (atom_min - vacuum) so the
        // box WRAPS the atoms rather than starting at the world origin.
        setCell(cell) {
            if (state.disposed) return;
            state.cellShapes = clear(state.cellShapes);
            if (!cell || !cell.lattice) { paint(); return; }
            const [a, b, c] = cell.lattice;
            const o = (Array.isArray(cell.origin) && cell.origin.length === 3)
                ? cell.origin : [0, 0, 0];
            const corner = (i, j, k) => ({
                x: o[0] + i * a[0] + j * b[0] + k * c[0],
                y: o[1] + i * a[1] + j * b[1] + k * c[1],
                z: o[2] + i * a[2] + j * b[2] + k * c[2],
            });
            const edges = [
                [[0,0,0],[1,0,0]], [[0,1,0],[1,1,0]], [[0,0,1],[1,0,1]], [[0,1,1],[1,1,1]],
                [[0,0,0],[0,1,0]], [[1,0,0],[1,1,0]], [[0,0,1],[0,1,1]], [[1,0,1],[1,1,1]],
                [[0,0,0],[0,0,1]], [[1,0,0],[1,0,1]], [[0,1,0],[0,1,1]], [[1,1,0],[1,1,1]],
            ];
            // Read each draw so a theme override applies live; cell redraws are
            // infrequent, so the lookup costs nothing that matters.
            const color  = cssVar("--mol-cell-wireframe-color", CELL_FALLBACK.color);
            const radius = parseFloat(cssVar("--mol-cell-wireframe-radius",
                                             String(CELL_FALLBACK.radius)))
                           || CELL_FALLBACK.radius;
            for (const [u, v] of edges) {
                try {
                    state.cellShapes.push(state.viewer.addCylinder({
                        start:   corner(u[0], u[1], u[2]),
                        end:     corner(v[0], v[1], v[2]),
                        radius:  radius,
                        color:   color,
                        fromCap: 1, toCap: 1,
                    }));
                } catch (_) {}
            }
            paint();
        },

        // This frame's arrows. What they look like is decided here (§ 6.5).
        setArrows(arrows) {
            if (state.disposed) return;
            state.arrows = Array.isArray(arrows) ? arrows : [];
            redrawArrows();
            paint();
        },

        // One door for everything drawn beside the atoms. Each entry names the
        // atom it belongs to, never a position — so a frame swap re-places it
        // here (§ 10.6) instead of the layer above having to re-send it.
        setOverlays(overlays) {
            if (state.disposed) return;
            overlays = overlays || {};
            state.overlays = {
                labels:    overlays.labels    || [],
                markers:   overlays.markers   || [],
                halos:     overlays.halos     || [],
                highlight: overlays.highlight || [],
            };
            replaceOverlays();
            paint();
        },

        /* ── The window itself ─────────────────────────────────────────── */

        // The camera. Held here and nowhere above (§ 9.6), and there is
        // deliberately no way to ask where it is pointing — a reload fits it to
        // the structure rather than restoring an angle.
        fitCamera() {
            if (state.disposed) return;
            try { state.viewer.zoomTo(); } catch (_) {}
            paint();
        },

        // Show or hide the "Updating view…" cover. A string shows it with that
        // message; false hides it.
        setBusy(message) {
            if (state.disposed || !state.hostEl) return;
            const el = state.hostEl.querySelector(".molview-busy");
            if (!el) return;
            if (message) {
                const msg = el.querySelector(".molview-busy-msg");
                if (msg) msg.textContent = String(message);
                el.hidden = false;
            } else {
                el.hidden = true;
            }
        },

        // Batch a group of changes so the screen updates once. Nested opens are
        // counted; the paint fires at the outermost close.
        beginBatch() { state.renderDepth += 1; },
        endBatch() {
            if (state.renderDepth > 0) state.renderDepth -= 1;
            if (!state.renderDepth && state.renderDirty) {
                state.renderDirty = false;
                paint();
            }
        },

        // Produce a picture of what is currently drawn — only the bottom can do
        // that (§ 11.4). WHAT to export and where it goes is decided above.
        capture(width, height) {
            return new Promise((resolve, reject) => {
                if (state.disposed) { reject(new Error("capture: disposed")); return; }
                try {
                    const container = state.viewer.container;
                    const resize = (width && width !== container.clientWidth)
                                || (height && height !== container.clientHeight);
                    // A size override re-renders at that resolution; otherwise
                    // grab the live canvas directly, which is much faster.
                    if (resize) {
                        const blob = dataUrlToBlob(state.viewer.pngURI(width, height));
                        blob ? resolve(blob)
                             : reject(new Error("capture: pngURI conversion failed"));
                        return;
                    }
                    const canvas = container.querySelector("canvas");
                    if (!canvas || typeof canvas.toBlob !== "function") {
                        const blob = dataUrlToBlob(state.viewer.pngURI());
                        blob ? resolve(blob)
                             : reject(new Error("capture: no canvas, pngURI failed"));
                        return;
                    }
                    canvas.toBlob((b) => {
                        b ? resolve(b) : reject(new Error("capture: toBlob returned null"));
                    }, "image/png");
                } catch (e) {
                    reject(new Error("capture: " + (e && e.message)));
                }
            });
        },

        // Report a clicked atom upward. Input entering at the bottom — this
        // layer holds no notion of what is selected.
        onPick(handler) {
            state.pickHandler = (typeof handler === "function") ? handler : null;
            if (state.pickHandler) wirePick();
        },

        /* ── The only two questions (§ 9.9, § 10.10) ───────────────────── */

        // Is there a movie loaded at all? Asked so the layer above can find out
        // whether appending has anything to append TO — § 10.10's "appending to
        // a structure with no movie rebuilds instead of extending nothing".
        hasMovie() {
            return !!mainModel();
        },

        // How many frames does the DRAWING have? A self-check only: the count a
        // consumer reads comes from the master copy, never from here (§ 10.10,
        // "only the master copy's count is offered"). This exists so the layer
        // above can notice a short drawing and heal it.
        drawnFrameCount() {
            try {
                const m = mainModel();
                if (m && typeof m.getNumFrames === "function") return m.getNumFrames();
            } catch (_) {}
            return 0;
        },

        /* ── Teardown ──────────────────────────────────────────────────── */

        dispose() {
            if (state.disposed) return;
            state.disposed = true;
            state.pickHandler = null;
            try { state.viewer.removeAllModels(); } catch (_) {}
            try { state.viewer.removeAllShapes(); } catch (_) {}
            try { state.viewer.removeAllLabels(); } catch (_) {}
            state.viewer = null;
        },
    };
}


/* Convert a data: URL to a Blob. Synchronous via atob, which is enough for the
 * modest PNGs the library produces; canvas.toDataURL always emits base64, so
 * that is the only form handled. */
function dataUrlToBlob(dataUrl) {
    try {
        const m = /^data:([^;,]+);base64,(.*)$/.exec(dataUrl);
        if (!m) return null;
        const bin = root.atob(m[2]);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return new root.Blob([bytes], { type: m[1] || "application/octet-stream" });
    } catch (_) { return null; }
}
