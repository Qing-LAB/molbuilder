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

/* The measurement glow (molview.md § 11.6).  A SECOND glow, not a second
 * meaning for the first: an atom can be selected AND measured at once, and the
 * two must be tellable apart on sight -- so this one is cool where the
 * selection is warm, and a little wider, so a measured atom that is also
 * selected reads as a ring around the amber rather than replacing it.
 *
 * Cool blue is also what the ruler already is elsewhere: the chip's border
 * takes `--molviewer-color-accent`, the same hue. */
/* THE MARKS CARRY THE ORDER (user, 2026-08-31: "using arrows to show what is
 * the item that is selected and the direction of that selection ... then the
 * orientation, the order, and everything is already shown in the drawing").
 *
 * A glow said WHICH atoms and nothing else, so an ordered pick and an
 * unordered one looked identical -- which is why `orient` reading a sorted set
 * as though it were a click order went unnoticed for as long as it did.  An
 * arrow per step says which was first, and the picture stops needing the
 * caption.
 *
 * One pick has no direction to show, so it keeps a mark on the atom itself.
 * Two draw one arrow, three draw two -- first->second, second->third -- which
 * is also exactly the angle's vertex at the tail of the second arrow.
 *
 * Cool blue in both, the hue the ruler already is elsewhere: the chip's border
 * takes `--molviewer-color-accent`. */
const MEASURE_MARK  = { color: "#5ad2ff", radius: 0.55, opacity: 0.55 };
const MEASURE_ARROW = { color: "#5ad2ff", radius: 0.07 };

const LABEL_STYLE = {
    fontSize:          12,
    fontColor:         "#222",
    backgroundColor:   "rgba(255,255,255,0.7)",
    backgroundOpacity: 0.7,
    inFront:           true,
};

const CELL_FALLBACK = { color: "#888", radius: 0.04 };

/* THE 3D WINDOW'S GROUND — one colour, painted twice.
 *
 * WebGL cannot take a colour from CSS: the clear colour is an argument to the
 * library, and the element behind it is painted by the stylesheet. Those are two
 * paints of the SAME surface, so they are one declared value — the card's
 * `--molviewer-scene-background` — read here the way the cell wireframe's colour
 * already is. Written as two literals instead, they drift, and the drawing sits
 * as a bright rectangle inside a dark card (which is exactly what shipped).
 *
 * The literal below is the last resort for a page with no stylesheet at all — a
 * node test — not a second palette. */
const SCENE_BACKGROUND = { name: "--molviewer-scene-background", fallback: "#0f1217" };

/* The bondless-atom marker (user, 2026-08-29).  Sticks draw BONDS AND
 * NOTHING ELSE, so an atom the library perceives no bonds for simply
 * vanishes -- a dissociated frame or an oddly-spaced junction rendered
 * a blank window with every layer healthy (the demo fixture's own
 * recorded failure, demo.js).  Under the stick representation those
 * atoms draw as small amber spheres instead: visible, and visibly NOT
 * an ordinary element colour, so the screen says "these atoms are here
 * but nothing bonds them" rather than saying nothing.  Amber, not the
 * highlight blue or an element colour, because it is a diagnostic mark. */
const BONDLESS_MARKER = { scale: 0.25, color: "#e0a13d" };

const root = (typeof window !== "undefined") ? window : globalThis;


/* Read a CSS custom property off an element. The scene constants are declared on
 * the card and CONCEALED to the module (§ 5.4); this is how the drawing layer
 * gets at values WebGL cannot be styled with. */
function readCssVar(el, name, fallback) {
    if (!el || typeof getComputedStyle !== "function") return fallback;
    try {
        const v = getComputedStyle(el).getPropertyValue(name).trim();
        return v || fallback;
    } catch (_) { return fallback; }
}

function sceneBackground(el) {
    return readCssVar(el, SCENE_BACKGROUND.name, SCENE_BACKGROUND.fallback);
}


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
        // No colour of its own: the window's ground is the card's, so the very
        // first painted frame already matches the surface it sits in.
        backgroundColor: (opts.view && opts.view.background) || sceneBackground(hostEl),
        defaultcolors:   $3Dmol.elementColors.Jmol,
    });

    // Everything this layer holds. Note what is NOT here: a frame number. The
    // library owns the playhead of its own movie; asking it back would be the
    // second home § 7 forbids.
    const state = {
        viewer:      viewer,
        hostEl:      hostEl,
        disposed:    false,
        // `background: null` means THE CARD'S OWN GROUND, resolved at paint time
        // — not "no background". A concrete colour here would be a second home
        // for a value the stylesheet already declares.
        view:        Object.assign({ rep: "stick", radiusScale: 1, background: null },
                                   opts.view || {}),
        // Shape handles we own and must remove before redrawing.
        cellShapes:      [],
        arrowShapes:     [],
        arrowLabels:     [],
        labelHandles:    [],
        highlightShapes: [],
        measuredShapes:  [],
        // What the overlays currently say, so a frame swap can re-place them
        // (§ 10.6: shapes move with the frames) without being re-sent.
        overlays:    { labels: [], highlight: [], measured: [] },
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
        // Bondless atoms never vanish (BONDLESS_MARKER above).  Stick is
        // the one shipped rep that draws only bonds; `line` already marks
        // lone atoms with the library's native crosses, and the sphere /
        // ball-and-stick reps draw every atom by construction.
        if ((state.view.rep || "stick") === "stick") {
            try {
                state.viewer.setStyle({ model: 0, bonds: 0 },
                                      { sphere: BONDLESS_MARKER });
            } catch (_) {}
        }
        // Unset means the card's ground (SCENE_BACKGROUND), never "leave it as
        // it was": the library's own default is white, so a skipped call is how
        // the window came to stay white inside a dark card.
        const bg = state.view.background || sceneBackground(state.hostEl);
        try {
            /* "transparent" is THIS MODULE's word for the background § 1.1
             * offers before you export a picture. The library has no such
             * colour — it takes a colour AND an alpha — so the translation into
             * its vocabulary happens here, the one place allowed to know what it
             * wants (§ 9.8). Handed the bare word, as it was, the library
             * resolved it to black and the preset silently painted the window
             * dark instead of clear. */
            if (bg === "transparent") state.viewer.setBackgroundColor(0x000000, 0);
            else state.viewer.setBackgroundColor(bg);
        } catch (_) {}
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

    /* ONE GLOW PRIMITIVE, two callers (§ 10.7).
     *
     * A glow is a SHAPE, never a restyle: restyling to show a selection
     * rebuilds the whole model's geometry, so its cost grows with the
     * structure; a sphere per atom does not.
     *
     * It replaced two doors that did the same thing and were reached by
     * nobody -- `markers` and `halos`, identical but for a default opacity,
     * both hard-coded to `[]` by the only caller since the embed they came
     * from was retired.  They also took their colour and radius FROM THE
     * CALLER, which § 6.5 gives to this layer: "this says WHICH atoms, and
     * what a highlight looks like is a constant owned by the sealed layer."
     * So the replacement takes a list of atoms and a style THIS FILE owns.
     */
    function redrawGlow(bucket, indices, style) {
        state[bucket] = clear(state[bucket]);
        const atoms = drawnAtoms();
        for (const i of indices) {
            const a = atoms[i];
            if (!a) continue;
            try {
                state[bucket].push(state.viewer.addSphere({
                    center:  { x: a.x, y: a.y, z: a.z },
                    radius:  style.radius,
                    color:   style.color,
                    opacity: style.opacity,
                }));
            } catch (_) {}
        }
    }

    /* THE RULER'S MARKS (§ 11.6), in pick order.
     *
     * ITS OWN SHAPES, NOT `state.arrows`.  `redrawArrows` ranks every arrow it
     * holds by length to decide which one gets the gold that marks the largest
     * force (§ 1.1) -- so a measurement arrow in that bucket could be the
     * longest and take the gold off the force that earned it.  Two overlays,
     * two buckets, no interaction.
     *
     * A PICK THAT IS NOT ON SCREEN COSTS THE ARROWS, not their correctness.
     * Under isolate a picked atom may not be drawn; joining the two that ARE
     * drawn would assert a step the user never made.  So a missing pick falls
     * back to marking what IS visible, which says less rather than something
     * untrue.
     */
    function redrawMeasurement(indices) {
        state.measuredShapes = clear(state.measuredShapes);
        const atoms = drawnAtoms();
        const pts = [];
        let allDrawn = true;
        for (const i of indices) {
            const a = atoms[i];
            if (!a) { allDrawn = false; continue; }
            pts.push(a);
        }
        if (!pts.length) return;

        const mark = (a) => {
            try {
                state.measuredShapes.push(state.viewer.addSphere({
                    center:  { x: a.x, y: a.y, z: a.z },
                    radius:  MEASURE_MARK.radius,
                    color:   MEASURE_MARK.color,
                    opacity: MEASURE_MARK.opacity,
                }));
            } catch (_) {}
        };

        // One pick, or a broken chain: there is no direction to draw.
        if (pts.length === 1 || !allDrawn) {
            for (const a of pts) mark(a);
            return;
        }

        for (let k = 0; k + 1 < pts.length; k++) {
            const a = pts[k], b = pts[k + 1];
            try {
                state.measuredShapes.push(state.viewer.addArrow({
                    start:       { x: a.x, y: a.y, z: a.z },
                    end:         { x: b.x, y: b.y, z: b.z },
                    radius:      MEASURE_ARROW.radius,
                    radiusRatio: 2.5,
                    mid:         0.85,
                    color:       MEASURE_ARROW.color,
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

    /* CARRIED KNOWLEDGE (2 of 3), CORRECTED. Arrows are batched — a whole
     * overlay becomes a few scene objects instead of N, measured ~7x faster per
     * frame (81 arrows: ~70ms -> ~10ms) — but ONE SHAPE PER COLOUR, not one
     * shape for everything.
     *
     * The claim this replaces was that a GLShape preserves per-arrow colour as
     * vertex colour. It does not, and the library says so when asked: after
     * appending a green arrow to a shape created from a red one, `shape.color`
     * is still red and every vertex colour is 0,0,0. A GLShape carries a SINGLE
     * colour, and `addArrow` only adds geometry to it.
     *
     * So batching everything into one shape painted every arrow the colour of
     * whichever arrow happened to be first. Both things that use this door lost
     * their meaning: the two axis triads came out monochrome — which is the
     * whole of what tells the world frame from the cell's (§ 10.3) — and a force
     * set whose largest arrow was not first lost the gold that marks it (§ 1.1).
     * Neither could fail a node test: a stand-in records the call, and the
     * colour is only wrong once something renders it.
     *
     * A stand-in whose addArrow return lacks .addArrow falls back to one shape
     * per arrow, which is correct, just slower.
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

        // colour -> the shape every arrow of that colour is appended to.
        const batches = new Map();
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
                const batch = batches.get(color);
                if (batch && typeof batch.addArrow === "function") {
                    batch.addArrow(spec);
                } else {
                    const shape = state.viewer.addArrow(spec);
                    state.arrowShapes.push(shape);
                    if (shape && typeof shape.addArrow === "function") {
                        batches.set(color, shape);
                    }
                }
            } catch (_) { continue; }
            if (a.label) {
                /* WHERE THE CALLER SAYS, when it says. `labelEnd` is the point
                 * just past the tip, worked out from the arrow's own base — and
                 * it was computed and never read, so this scaled `end` by 1.05
                 * FROM THE WORLD ORIGIN instead. For a triad based at the origin
                 * the two agree, which is why it looked right; for the cell's
                 * triad, based at the box's corner, the label drifts off toward
                 * the origin by a fraction of the whole vector. */
                const at = (Array.isArray(a.labelEnd) && a.labelEnd.length === 3)
                    ? a.labelEnd
                    : [a.end[0] * 1.05, a.end[1] * 1.05, a.end[2] * 1.05];
                try {
                    state.arrowLabels.push(state.viewer.addLabel(String(a.label), {
                        position: { x: at[0], y: at[1], z: at[2] },
                        ...LABEL_STYLE,
                    }));
                } catch (_) {}
            }
        }
    }

    /* t in [0,1]: dim-red -> orange-red, IN BANDS.
     *
     * Quantised because the arrows are batched per colour above, and a
     * continuous ramp would give a distinct colour to nearly every arrow —
     * one shape each, which is the cost the batching exists to avoid. Bands put
     * a ceiling on the shape count that does not grow with the atom count.
     *
     * Nothing is lost by it: § 1.1's signal is that "converging forces visibly
     * shrink" — the LENGTH carries the magnitude, and the shade is a second cue
     * on a thin arrow, where the eye separates far fewer than this many steps
     * anyway. */
    const RAMP_BANDS = 8;
    function rampColor(t) {
        const band = Math.round(t * (RAMP_BANDS - 1)) / (RAMP_BANDS - 1);
        return "rgb(" + Math.floor(170 + 85 * band) + ","
                      + Math.floor(40 + 60 * band) + ",32)";
    }

    // Read a CSS custom property off the card, where these are declared and
    // concealed to the module — 3D-scene constants WebGL cannot style directly,
    // kept in one discoverable place instead of as magic numbers.
    function cssVar(name, fallback) {
        return readCssVar(state.hostEl, name, fallback);
    }

    // Everything the overlays own, re-placed at whatever frame is now drawn.
    // Cheap — one frame's worth of shapes, not a movie rebuild. The cell is
    // lattice-only and static, so it is deliberately not in here.
    function replaceOverlays() {
        redrawLabels();
        // The ruler's marks are drawn UNDER the selection glow, so an atom that
        // is both keeps the amber at its centre.
        redrawMeasurement(state.overlays.measured);
        redrawGlow("highlightShapes", state.overlays.highlight, SELECTION_GLOW);
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
            /* CARRIED KNOWLEDGE (4 of 4). setFrame is ASYNCHRONOUS — it returns
             * a promise, and the frame's geometry is not in place until that
             * settles. Painting straight after it paints the swap that has not
             * happened yet, and the window goes EMPTY: frame 0 keeps showing
             * whatever the load left, and every later frame draws nothing at
             * all, with no error and with the movie reporting its full length.
             *
             * The overlays have to wait too — they are placed from the drawn
             * atoms (§ 10.6), so re-placing them early puts them on the
             * previous frame's positions. */
            let swap;
            try { swap = state.viewer.setFrame(i); } catch (_) { return; }
            const settled = () => {
                if (state.disposed) return;
                /* CARRIED KNOWLEDGE (5 of 5).  setFrame swaps the model's
                 * active atom array for the FRAME'S OWN atom objects (the
                 * library's setFrame: `s.atoms = s.frames[e]`), and
                 * setClickable stamps clickable + callback on the objects
                 * active at wiring time -- so every frame but the wired one
                 * held zero clickable atoms and clicks fell straight
                 * through (found 2026-08-20: the window selected only on
                 * frame 0 while the atom list kept working).  Clickability
                 * is re-established after every settled swap, before the
                 * paint; it stamps flags on n atoms and rebuilds no
                 * geometry. */
                state.pickWired = false;
                wirePick();
                replaceOverlays();
                paint();
            };
            if (swap && typeof swap.then === "function") swap.then(settled, settled);
            else settled();
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
            const color  = cssVar("--molviewer-scene-cell-color", CELL_FALLBACK.color);
            const radius = parseFloat(cssVar("--molviewer-scene-cell-radius",
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
                highlight: overlays.highlight || [],
                measured:  overlays.measured  || [],
            };
            replaceOverlays();
            paint();
        },

        /* ── The window itself ─────────────────────────────────────────── */

        // The camera. Held here and nowhere above (§ 9.6): the pair below
        // REPORTS and POINTS it for the view context (§ 11.2b) -- the lane
        // stores what was read at one instant, and no layer above keeps a
        // copy.  A reload without a matching context still fits it to the
        // structure rather than restoring an angle.
        fitCamera() {
            if (state.disposed) return;
            try { state.viewer.zoomTo(); } catch (_) {}
            paint();
        },

        // "Report the pose" -- § 9.7's bounded asking of the WINDOW.  The
        // answer is the library's own opaque view array, carried verbatim;
        // null when there is nothing to report.
        getCamera() {
            if (state.disposed) return null;
            try {
                const v = state.viewer.getView();
                return Array.isArray(v) ? v.slice() : null;
            } catch (_) { return null; }
        },

        // "Point the camera here" -- the same array back, and nothing else
        // accepted: a pose from another library version that fails to apply
        // degrades to the fit the caller falls back to anyway.
        setCamera(pose) {
            if (state.disposed || !Array.isArray(pose)) return false;
            try { state.viewer.setView(pose); } catch (_) { return false; }
            paint();
            return true;
        },

        // Show or hide the "Updating view…" cover. A string shows it with that
        // message; false hides it.
        setBusy(message) {
            if (state.disposed || !state.hostEl) return;
            const el = state.hostEl.querySelector(".molviewer-window-busy");
            if (!el) return;
            if (message) {
                const msg = el.querySelector(".molviewer-window-busy-msg");
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
