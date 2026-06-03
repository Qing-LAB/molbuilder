/* mol-viewer-embed.js — the standard embeddable 3D viewer.
 *
 * Contract: docs/protocols/embedded-viewer.md.  The contract is
 * the SOLE source of truth; this file implements it.  When the
 * contract changes, this file changes in the same commit.
 *
 *   window.molbuilder.viewer.embed(host, opts) -> handle
 *
 * Composes the existing primitives (mol-style, mol-axes, mol-pick,
 * mol-format) into a card/panel + 3Dmol viewer that maintains its
 * own drawing.  The caller never reaches into 3Dmol directly --
 * all mutations go through the handle's methods.
 *
 * This file covers the STATIC-FEATURES contract (no animation):
 * structure load, style, axes, cell wireframe, labels, arrows,
 * atom-pick, card chrome, dispose.  Animation (vibration +
 * trajectory) is a separate stage so the static feature surface
 * lands first + gets exercised by /modify / Build / structure-
 * inspector migrations before the animation loop is added.
 */
(function (root) {
    "use strict";

    /* ------------------------------------------------------------ */
    /*  CSS class constants                                          */
    /* ------------------------------------------------------------ */

    const CARD_CLASS         = "card mol-viewer-card";
    const HEADER_CLASS       = "mol-viewer-card-header";
    const TITLE_CLASS        = "mol-viewer-card-title";
    const INFO_LINE_CLASS    = "mol-viewer-info-line";
    const CANVAS_CLASS       = "mol-viewer-canvas";
    const DEFAULT_HEIGHT     = "clamp(360px, 52vh, 500px)";
    const DEFAULT_BACKGROUND = "#ffffff";   // 3Dmol convention (web-api.md § 11.4)

    /* ------------------------------------------------------------ */
    /*  Pure-logic helpers (exported for unit testing)               */
    /* ------------------------------------------------------------ */

    /**
     * Normalise a ViewerOpts object into the internal state shape
     * used for idempotence comparison.  Each section becomes either
     * a normalised sub-object OR ``null`` (feature disabled).
     *
     * This function does NOT touch 3Dmol or the DOM; it's a pure
     * data transform.  Exported via window.molbuilder.viewer._normaliseOpts
     * for unit testing.
     */
    function _normaliseOpts(opts) {
        opts = opts || {};
        return {
            xyz:        typeof opts.xyz === "string" ? opts.xyz : null,
            pdb:        typeof opts.pdb === "string" ? opts.pdb : null,
            style:      _normaliseStyle(opts.style),
            axes:       _normaliseAxes(opts.axes),
            cell:       _normaliseCell(opts.cell),
            labels:     _normaliseLabels(opts.labels),
            arrows:     Array.isArray(opts.arrows) ? opts.arrows.slice() : [],
            pick:       _normalisePick(opts.pick),
            lattice:    _normaliseLattice(opts.lattice),
        };
    }

    function _normaliseStyle(s) {
        s = s || {};
        return {
            rep:          s.rep         || "stick",
            radiusScale:  typeof s.radiusScale === "number"
                            ? s.radiusScale : 1.0,
            colorScheme:  s.colorScheme || null,
            background:   s.background  || DEFAULT_BACKGROUND,
            showLabels:   s.showLabels  || false,
        };
    }

    function _normaliseAxes(a) {
        if (a === false || a === undefined || a === null) return null;
        if (a === true) return { mode: "auto" };
        return {
            mode:    a.mode    || "auto",
            length:  typeof a.length === "number" ? a.length : undefined,
            origin:  a.origin,
            labels:  a.labels,
            colors:  a.colors,
            radius:  typeof a.radius === "number" ? a.radius : undefined,
        };
    }

    function _normaliseCell(c) {
        if (c === false || c === undefined || c === null) return null;
        if (c === true) return { color: null, radius: 0.04 };
        return {
            color:  c.color || null,
            radius: typeof c.radius === "number" ? c.radius : 0.04,
        };
    }

    function _normaliseLabels(l) {
        if (l === false || l === undefined || l === null) return null;
        if (l === true) return { atoms: "indices", fontSize: 12 };
        return {
            atoms:       l.atoms      || "indices",
            fontSize:    l.fontSize   || 12,
            fontColor:   l.fontColor  || null,
            background:  l.background || null,
        };
    }

    function _normalisePick(p) {
        if (!p || p === false || p.mode === "none") return null;
        return {
            mode:        p.mode       || "single",
            haloColor:   p.haloColor  || "#ffd54a",
            haloRadius:  typeof p.haloRadius === "number"
                            ? p.haloRadius : 0.6,
            onPick:      typeof p.onPick === "function"
                            ? p.onPick : null,
        };
    }

    function _normaliseLattice(L) {
        if (!Array.isArray(L) || L.length !== 3) return null;
        for (let i = 0; i < 3; i++) {
            const row = L[i];
            if (!Array.isArray(row) || row.length !== 3) return null;
            for (let j = 0; j < 3; j++) {
                if (typeof row[j] !== "number"
                    || !Number.isFinite(row[j])) return null;
            }
        }
        return [
            [L[0][0], L[0][1], L[0][2]],
            [L[1][0], L[1][1], L[1][2]],
            [L[2][0], L[2][1], L[2][2]],
        ];
    }

    /**
     * Cheap structural equality for the small option sub-objects.
     * JSON.stringify is good enough at this UI scale; if perf
     * becomes an issue, swap for a typed deep-equal.
     */
    function _equalNormalised(a, b) {
        if (a === b) return true;
        if (a === null || b === null) return false;
        return JSON.stringify(a) === JSON.stringify(b);
    }

    /* ------------------------------------------------------------ */
    /*  Card scaffold construction                                   */
    /* ------------------------------------------------------------ */

    function _buildCardScaffold(opts) {
        const card = opts.card || {};

        // ``bare`` mode: embed inside an existing card without the
        // standard .card.mol-viewer-card chrome.  Use when the host
        // already has its own card wrapper (e.g. the structure /
        // trajectory / spectra inspectors, which carry per-tab
        // actions in their card headers).  The viewer still owns
        // the canvas + info-line via the handle methods; only the
        // outermost wrapper is suppressed.
        const bare = card.bare === true;
        const section = document.createElement(bare ? "div" : "section");
        section.className = (bare
                ? "mol-viewer-bare"
                : CARD_CLASS)
            + (card.className ? " " + card.className : "");
        section.setAttribute("data-mol-viewer", "1");

        // Header — only rendered if title or info-line is requested.
        // In bare mode the title is also suppressed (the host card's
        // header already shows it); the info-line still renders if
        // requested.
        const titleText = bare ? null : card.title;
        const showInfo  = card.showInfoLine !== false;
        let infoLineEl = null;
        if (titleText || showInfo) {
            const header = document.createElement("header");
            header.className = HEADER_CLASS;
            if (titleText) {
                const h2 = document.createElement("h2");
                h2.className = TITLE_CLASS;
                h2.textContent = titleText;
                header.appendChild(h2);
            }
            if (showInfo) {
                infoLineEl = document.createElement("span");
                infoLineEl.className = INFO_LINE_CLASS;
                infoLineEl.textContent = "";
                header.appendChild(infoLineEl);
            }
            section.appendChild(header);
        }

        const canvas = document.createElement("div");
        canvas.className = CANVAS_CLASS;
        canvas.style.height = card.height || DEFAULT_HEIGHT;
        section.appendChild(canvas);

        return { section, canvas, infoLineEl };
    }

    /* ------------------------------------------------------------ */
    /*  Structure load                                               */
    /* ------------------------------------------------------------ */

    function _loadStructure(viewer, current) {
        // Remove every existing model so a second setStructure call
        // doesn't stack the new atoms on top of the old.
        try { viewer.removeAllModels(); } catch (_) {}
        // PDB wins if both supplied (richer metadata).  Empty string
        // strictly disallowed: caller must pass a non-empty text.
        let text, format;
        if (current.pdb) { text = current.pdb; format = "pdb"; }
        else if (current.xyz) { text = current.xyz; format = "xyz"; }
        else return null;
        const model = viewer.addModel(text, format);
        return model;
    }

    function _atomCount(viewer) {
        try {
            const models = viewer.getModel();
            const atoms = models ? models.selectedAtoms({}) : [];
            return atoms ? atoms.length : 0;
        } catch (_) { return 0; }
    }

    function _elements(viewer) {
        try {
            const models = viewer.getModel();
            const atoms = models ? models.selectedAtoms({}) : [];
            return atoms ? atoms.map((a) => a.elem || a.element) : [];
        } catch (_) { return []; }
    }

    function _formula(viewer) {
        const fmt = (root.molbuilder || {}).fmt;
        if (!fmt || typeof fmt.formula !== "function") return "";
        return fmt.formula(_elements(viewer));
    }

    function _infoLineText(viewer) {
        const n = _atomCount(viewer);
        if (n === 0) return "";
        const f = _formula(viewer);
        return n + " atoms" + (f ? " · " + f : "");
    }

    /* ------------------------------------------------------------ */
    /*  Style application                                            */
    /* ------------------------------------------------------------ */

    function _applyStyle(viewer, style) {
        const styleApi = (root.molbuilder || {}).style;
        let spec;
        if (styleApi && typeof styleApi.spec === "function") {
            spec = styleApi.spec({
                rep:         style.rep,
                scale:       style.radiusScale,
                colorscheme: style.colorScheme,
            });
        } else {
            // Minimal fallback so the viewer renders something even
            // if lib/mol-style.js failed to load (defensive: the
            // template ALWAYS loads it).
            spec = { stick: {} };
        }
        try { viewer.setStyle({}, spec); }
        catch (_) {}
        try { viewer.setBackgroundColor(style.background); }
        catch (_) {}
    }

    /* ------------------------------------------------------------ */
    /*  Cell wireframe                                               */
    /* ------------------------------------------------------------ */

    function _drawCellWireframe(viewer, lattice, opts) {
        if (!lattice) return [];
        const [a, b, c] = lattice;
        const corner = (i, j, k) => ({
            x: i * a[0] + j * b[0] + k * c[0],
            y: i * a[1] + j * b[1] + k * c[1],
            z: i * a[2] + j * b[2] + k * c[2],
        });
        const edges = [
            [[0,0,0],[1,0,0]], [[0,1,0],[1,1,0]],
            [[0,0,1],[1,0,1]], [[0,1,1],[1,1,1]],
            [[0,0,0],[0,1,0]], [[1,0,0],[1,1,0]],
            [[0,0,1],[0,1,1]], [[1,0,1],[1,1,1]],
            [[0,0,0],[0,0,1]], [[1,0,0],[1,0,1]],
            [[0,1,0],[0,1,1]], [[1,1,0],[1,1,1]],
        ];
        const color  = opts.color  || "#888";
        const radius = typeof opts.radius === "number" ? opts.radius : 0.04;
        const shapes = [];
        for (const [u, v] of edges) {
            const s = viewer.addCylinder({
                start:  corner(u[0], u[1], u[2]),
                end:    corner(v[0], v[1], v[2]),
                radius: radius,
                color:  color,
                fromCap: 1, toCap: 1,
            });
            shapes.push(s);
        }
        return shapes;
    }

    /* ------------------------------------------------------------ */
    /*  Atom labels                                                  */
    /* ------------------------------------------------------------ */

    function _drawAtomLabels(viewer, opts) {
        if (!opts) return [];
        const handles = [];
        try {
            const model = viewer.getModel();
            const atoms = model ? model.selectedAtoms({}) : [];
            for (let i = 0; i < atoms.length; i++) {
                const a = atoms[i];
                let text;
                if (Array.isArray(opts.atoms)) {
                    if (opts.atoms.indexOf(i) < 0) continue;
                    text = String(i);
                } else if (opts.atoms === "names") {
                    text = a.atom || a.name || a.elem || String(i);
                } else {
                    text = String(i);
                }
                const lbl = viewer.addLabel(text, {
                    position:          { x: a.x, y: a.y, z: a.z },
                    fontSize:          opts.fontSize || 12,
                    fontColor:         opts.fontColor || "#222",
                    backgroundColor:   opts.background || "rgba(255,255,255,0.7)",
                    backgroundOpacity: opts.background ? 1.0 : 0.7,
                    inFront:           true,
                });
                handles.push(lbl);
            }
        } catch (_) {}
        return handles;
    }

    /* ------------------------------------------------------------ */
    /*  Arrow overlays                                               */
    /* ------------------------------------------------------------ */

    function _drawArrows(viewer, arrows) {
        const shapes = [];
        const labels = [];
        for (const a of arrows) {
            if (!a || !a.start || !a.end) continue;
            const color = a.color || "#888";
            const radius = typeof a.radius === "number" ? a.radius : 0.05;
            const arrow = viewer.addArrow({
                start: { x: a.start[0], y: a.start[1], z: a.start[2] },
                end:   { x: a.end[0],   y: a.end[1],   z: a.end[2] },
                radius:      radius,
                radiusRatio: 2.5,
                mid:         0.85,
                color:       color,
            });
            shapes.push(arrow);
            if (a.label) {
                const lbl = viewer.addLabel(a.label, {
                    position: {
                        x: a.end[0] * 1.05,
                        y: a.end[1] * 1.05,
                        z: a.end[2] * 1.05,
                    },
                    fontColor: color,
                    backgroundOpacity: 0.0,
                    fontSize: 11,
                    inFront: true,
                });
                labels.push(lbl);
            }
        }
        return { shapes, labels };
    }

    /* ------------------------------------------------------------ */
    /*  Atom-pick wiring                                             */
    /* ------------------------------------------------------------ */

    function _wirePick(viewer, state) {
        if (!state.current.pick) return;
        // Use 3Dmol's setClickable; per-atom click handler routes to
        // our pick handler.
        try {
            viewer.setClickable({}, true, function (atom, _viewer, _evt) {
                if (state.disposed) return;
                const idx = atom && (atom.index != null ? atom.index : atom.serial);
                if (typeof idx !== "number") return;
                _togglePick(state, idx);
            });
        } catch (_) {}
    }

    function _togglePick(state, idx) {
        const mode = state.current.pick.mode;
        let next;
        const pos = state.pickedIndices.indexOf(idx);
        if (pos >= 0) {
            // Already picked → deselect.
            next = state.pickedIndices.filter((i) => i !== idx);
        } else if (mode === "single") {
            next = [idx];
        } else if (mode === "pair") {
            next = state.pickedIndices.length < 2
                ? state.pickedIndices.concat([idx])
                : [state.pickedIndices[1], idx];
        } else {
            // multi
            next = state.pickedIndices.concat([idx]);
        }
        state.pickedIndices = next;
        _redrawPickHalos(state);
        if (state.current.pick.onPick) {
            try { state.current.pick.onPick(next.slice()); }
            catch (_) {}
        }
    }

    function _redrawPickHalos(state) {
        // Clear previous halos.
        for (const s of state.pickShapes) {
            try { state.viewer.removeShape(s); } catch (_) {}
        }
        state.pickShapes = [];
        if (!state.current.pick) {
            state.viewer.render();
            return;
        }
        const color = state.current.pick.haloColor;
        const radius = state.current.pick.haloRadius;
        try {
            const model = state.viewer.getModel();
            const atoms = model ? model.selectedAtoms({}) : [];
            for (const idx of state.pickedIndices) {
                if (idx < 0 || idx >= atoms.length) continue;
                const a = atoms[idx];
                const halo = state.viewer.addSphere({
                    center: { x: a.x, y: a.y, z: a.z },
                    radius: radius,
                    color:  color,
                    opacity: 0.35,
                });
                state.pickShapes.push(halo);
            }
        } catch (_) {}
        state.viewer.render();
    }

    /* ------------------------------------------------------------ */
    /*  Overlay re-application (called after structure load + on    */
    /*  setX methods that affect them)                              */
    /* ------------------------------------------------------------ */

    function _redrawAxes(state) {
        if (state.axesHandle) {
            state.axesHandle.clear();
            state.axesHandle = null;
        }
        if (!state.current.axes) return;
        const axesApi = (root.molbuilder || {}).axes;
        if (!axesApi || typeof axesApi.draw !== "function") return;
        const a = state.current.axes;
        const cell = (a.mode === "cartesian") ? null
                   : (a.mode === "cell")      ? state.current.lattice
                   : /* auto */                  state.current.lattice;
        state.axesHandle = axesApi.draw(state.viewer, {
            cell:   cell,
            length: a.length,
            origin: a.origin,
            labels: a.labels,
            colors: a.colors,
            radius: a.radius,
            render: false,   // we batch a single render at the end
        });
    }

    function _redrawCell(state) {
        for (const s of state.cellShapes) {
            try { state.viewer.removeShape(s); } catch (_) {}
        }
        state.cellShapes = [];
        if (!state.current.cell) return;
        state.cellShapes = _drawCellWireframe(
            state.viewer, state.current.lattice, state.current.cell);
    }

    function _redrawLabels(state) {
        for (const l of state.labelHandles) {
            try { state.viewer.removeLabel(l); } catch (_) {}
        }
        state.labelHandles = [];
        if (!state.current.labels) return;
        state.labelHandles = _drawAtomLabels(state.viewer, state.current.labels);
    }

    function _redrawArrows(state) {
        for (const s of state.arrowShapes) {
            try { state.viewer.removeShape(s); } catch (_) {}
        }
        for (const l of state.arrowLabels) {
            try { state.viewer.removeLabel(l); } catch (_) {}
        }
        state.arrowShapes = [];
        state.arrowLabels = [];
        if (!state.current.arrows.length) return;
        const out = _drawArrows(state.viewer, state.current.arrows);
        state.arrowShapes = out.shapes;
        state.arrowLabels = out.labels;
    }

    function _redrawAllOverlays(state) {
        _redrawAxes(state);
        _redrawCell(state);
        _redrawLabels(state);
        _redrawArrows(state);
        _redrawPickHalos(state);
    }

    /* ------------------------------------------------------------ */
    /*  Info-line refresh                                            */
    /* ------------------------------------------------------------ */

    function _refreshInfoLine(state) {
        if (!state.infoLineEl) return;
        state.infoLineEl.textContent = _infoLineText(state.viewer);
    }

    /* ------------------------------------------------------------ */
    /*  Animation — vibration (rAF cosine) + trajectory (interval)  */
    /* ------------------------------------------------------------ */

    function _normaliseAnimation(a) {
        if (!a || typeof a !== "object") return null;
        if (a.kind === "vibration") {
            if (!Array.isArray(a.displacements)) return null;
            return {
                kind:         "vibration",
                displacements: a.displacements,
                amplitude:    typeof a.amplitude === "number"
                                ? a.amplitude : 0.15,
                speedHz:      typeof a.speedHz === "number"
                                ? a.speedHz : 1.0,
                paused:       a.paused === true,
            };
        }
        if (a.kind === "trajectory") {
            if (!Array.isArray(a.frames) || a.frames.length === 0) return null;
            const nFrames = a.frames.length;
            const startFrame = (typeof a.startFrame === "number"
                                && a.startFrame >= 0
                                && a.startFrame < nFrames)
                ? Math.floor(a.startFrame) : 0;
            return {
                kind:         "trajectory",
                frames:       a.frames,
                startFrame:   startFrame,
                currentFrame: startFrame,
                fps:          typeof a.fps === "number" && a.fps > 0
                                ? a.fps : 10,
                paused:       a.paused !== false,  // default paused
                loop:         a.loop !== false,    // default loop
            };
        }
        return null;
    }

    function _captureBaselineCoords(viewer) {
        try {
            const model = viewer.getModel();
            const atoms = model ? model.selectedAtoms({}) : [];
            const out = [];
            for (const a of atoms) {
                out.push([a.x, a.y, a.z]);
            }
            return out;
        } catch (_) { return []; }
    }

    function _applyCoords(viewer, coords) {
        try {
            const model = viewer.getModel();
            const atoms = model ? model.selectedAtoms({}) : [];
            const n = Math.min(atoms.length, coords.length);
            for (let i = 0; i < n; i++) {
                const c = coords[i];
                if (!c) continue;
                atoms[i].x = c[0];
                atoms[i].y = c[1];
                atoms[i].z = c[2];
            }
        } catch (_) {}
    }

    function _postFramePositionRedraw(state) {
        // Position-aware overlays must recompute every frame so they
        // track the moving atoms.  Cell wireframe is lattice-only
        // (static); axes are origin-anchored (static); arrows are
        // caller-supplied (static unless caller updates via
        // setArrows during animation).
        _redrawLabels(state);
        _redrawPickHalos(state);
    }

    function _stopAnimationLoop(state) {
        const a = state._anim;
        if (a.rafId !== null) {
            try { cancelAnimationFrame(a.rafId); } catch (_) {}
            a.rafId = null;
        }
        if (a.intervalId !== null) {
            try { clearInterval(a.intervalId); } catch (_) {}
            a.intervalId = null;
        }
        a.playing = false;
        _refreshFrameStrip(state);
    }

    function _startVibrationLoop(state) {
        if (!state._anim.vibrationBaseline) {
            state._anim.vibrationBaseline = _captureBaselineCoords(state.viewer);
        }
        state._anim.startTimeMs = (typeof performance !== "undefined"
                                    && performance.now)
            ? performance.now() : Date.now();
        state._anim.playing = true;
        const tick = (tsMs) => {
            if (state.disposed || !state._anim.playing) return;
            const v = state.current.animation;
            if (!v || v.kind !== "vibration") return;
            const baseline = state._anim.vibrationBaseline;
            const disp     = v.displacements;
            const elapsedSec = (tsMs - state._anim.startTimeMs) / 1000;
            const phase = 2 * Math.PI * v.speedHz * elapsedSec;
            const factor = v.amplitude * Math.cos(phase);
            const out = [];
            for (let i = 0; i < baseline.length; i++) {
                const d = (i < disp.length) ? disp[i] : [0, 0, 0];
                out.push([
                    baseline[i][0] + factor * d[0],
                    baseline[i][1] + factor * d[1],
                    baseline[i][2] + factor * d[2],
                ]);
            }
            _applyCoords(state.viewer, out);
            _postFramePositionRedraw(state);
            state.viewer.render();
            state._anim.rafId = requestAnimationFrame(tick);
        };
        state._anim.rafId = requestAnimationFrame(tick);
        _refreshFrameStrip(state);
    }

    function _startTrajectoryLoop(state) {
        state._anim.playing = true;
        const t = state.current.animation;
        const periodMs = 1000 / Math.max(1, t.fps);
        state._anim.intervalId = setInterval(() => {
            if (state.disposed || !state._anim.playing) return;
            const a = state.current.animation;
            if (!a || a.kind !== "trajectory") return;
            let next = a.currentFrame + 1;
            if (next >= a.frames.length) {
                if (a.loop) next = 0;
                else {
                    _stopAnimationLoop(state);
                    return;
                }
            }
            _showTrajectoryFrame(state, next);
        }, periodMs);
        _refreshFrameStrip(state);
    }

    function _showTrajectoryFrame(state, idx) {
        const a = state.current.animation;
        if (!a || a.kind !== "trajectory") return;
        if (idx < 0 || idx >= a.frames.length) return;
        a.currentFrame = idx;
        _applyCoords(state.viewer, a.frames[idx]);
        _postFramePositionRedraw(state);
        state.viewer.render();
        _refreshFrameStrip(state);
    }

    /* ------------------------------------------------------------ */
    /*  Frame-strip card chrome (trajectory animation only)         */
    /* ------------------------------------------------------------ */

    function _buildFrameStrip(state) {
        if (state.frameStripEl) return;  // already built
        const opts = state.cardOpts || {};
        if (!opts.frameStrip) return;
        const a = state.current.animation;
        if (!a || a.kind !== "trajectory") return;

        const strip = document.createElement("div");
        strip.className = "mol-viewer-frame-strip";

        const prev = document.createElement("button");
        prev.type = "button";
        prev.className = "frame-prev";
        prev.textContent = "‹";
        prev.title = "Previous frame";
        prev.addEventListener("click", () => {
            _stopAnimationLoop(state);
            const cur = state.current.animation;
            if (!cur) return;
            const i = (cur.currentFrame - 1 + cur.frames.length) % cur.frames.length;
            _showTrajectoryFrame(state, i);
        });

        const playPause = document.createElement("button");
        playPause.type = "button";
        playPause.className = "frame-play-pause";
        playPause.textContent = "▶";
        playPause.title = "Play / pause";
        playPause.addEventListener("click", () => {
            if (state._anim.playing) _pauseImpl(state);
            else _playImpl(state);
        });

        const next = document.createElement("button");
        next.type = "button";
        next.className = "frame-next";
        next.textContent = "›";
        next.title = "Next frame";
        next.addEventListener("click", () => {
            _stopAnimationLoop(state);
            const cur = state.current.animation;
            if (!cur) return;
            const i = (cur.currentFrame + 1) % cur.frames.length;
            _showTrajectoryFrame(state, i);
        });

        const counter = document.createElement("span");
        counter.className = "frame-counter";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.className = "frame-slider";
        slider.min = "0";
        slider.max = String(a.frames.length - 1);
        slider.step = "1";
        slider.addEventListener("input", () => {
            _stopAnimationLoop(state);
            _showTrajectoryFrame(state, parseInt(slider.value, 10));
        });

        strip.appendChild(prev);
        strip.appendChild(playPause);
        strip.appendChild(next);
        strip.appendChild(counter);
        strip.appendChild(slider);

        // Insert before the canvas so the strip sits above it.
        state.cardEl.insertBefore(strip, state.canvasEl);
        state.frameStripEl = strip;
        state.frameStripParts = {
            prev: prev, playPause: playPause, next: next,
            counter: counter, slider: slider,
        };
        _refreshFrameStrip(state);
    }

    function _removeFrameStrip(state) {
        if (!state.frameStripEl) return;
        try { state.frameStripEl.remove(); } catch (_) {}
        state.frameStripEl = null;
        state.frameStripParts = null;
    }

    function _refreshFrameStrip(state) {
        if (!state.frameStripEl) return;
        const a = state.current.animation;
        if (!a || a.kind !== "trajectory") return;
        const p = state.frameStripParts;
        if (!p) return;
        p.counter.textContent = (a.currentFrame + 1) + " / " + a.frames.length;
        p.slider.max = String(a.frames.length - 1);
        p.slider.value = String(a.currentFrame);
        p.playPause.textContent = state._anim.playing ? "❚❚" : "▶";
    }

    /* ------------------------------------------------------------ */
    /*  Animation control implementations (shared by handle methods)*/
    /* ------------------------------------------------------------ */

    function _setAnimationImpl(state, next) {
        // Stop any in-flight loop + reset to baseline (vibration).
        _stopAnimationLoop(state);
        if (state._anim.vibrationBaseline) {
            // Snap back to baseline so the next setAnimation lands
            // on a known clean state.
            _applyCoords(state.viewer, state._anim.vibrationBaseline);
            _postFramePositionRedraw(state);
            state.viewer.render();
        }
        state._anim.vibrationBaseline = null;
        if (next && next.kind === "trajectory") {
            // Land on the requested startFrame so the user sees it.
            state.current.animation = next;
            _buildFrameStrip(state);
            _showTrajectoryFrame(state, next.startFrame);
            if (!next.paused) _playImpl(state);
        } else if (next && next.kind === "vibration") {
            state.current.animation = next;
            _removeFrameStrip(state);
            if (!next.paused) _playImpl(state);
        } else {
            state.current.animation = null;
            _removeFrameStrip(state);
        }
    }

    function _playImpl(state) {
        if (state.disposed) return;
        const a = state.current.animation;
        if (!a) return;
        if (state._anim.playing) return;
        if (a.kind === "vibration") _startVibrationLoop(state);
        else if (a.kind === "trajectory") _startTrajectoryLoop(state);
    }

    function _pauseImpl(state) {
        if (state.disposed) return;
        _stopAnimationLoop(state);
    }

    /* ------------------------------------------------------------ */
    /*  Public embed() entry point                                   */
    /* ------------------------------------------------------------ */

    function embed(host, opts) {
        opts = opts || {};
        if (!host || !host.appendChild) {
            throw new TypeError(
                "viewer.embed(host, opts): host must be a DOM element"
            );
        }
        // xyz/pdb are OPTIONAL at mount.  The caller can mount an
        // empty viewer (no structure loaded yet) and populate later
        // via handle.setStructure(...).  This matters for tabs that
        // build a viewer before the user has picked a file (/modify,
        // /build) -- the viewer renders an empty canvas until the
        // first setStructure call.
        const viewerApi = (root.molbuilder || {}).viewer;
        if (!viewerApi || typeof viewerApi.create !== "function") {
            throw new Error(
                "viewer.embed: lib/mol-viewer.js must be loaded first"
            );
        }

        // 1. Build the card scaffold + mount into host.
        const scaffold = _buildCardScaffold(opts);
        host.appendChild(scaffold.section);

        // 2. Create the 3Dmol viewer inside the canvas.
        const viewer = viewerApi.create(scaffold.canvas);

        // 3. Initial state.
        const current = _normaliseOpts(opts);
        current.animation = _normaliseAnimation(opts.animation);
        const state = {
            viewer:        viewer,
            hostEl:        host,
            cardEl:        scaffold.section,
            canvasEl:      scaffold.canvas,
            infoLineEl:    scaffold.infoLineEl,
            cardOpts:      opts.card || {},

            current:       current,

            axesHandle:    null,
            cellShapes:    [],
            labelHandles:  [],
            arrowShapes:   [],
            arrowLabels:   [],
            pickShapes:    [],
            pickedIndices: [],

            // Animation runtime state.  vibrationBaseline is captured
            // on first vibration play so we can restore on stop /
            // setAnimation(null) without re-loading the structure.
            _anim: {
                playing:            false,
                rafId:              null,
                intervalId:         null,
                startTimeMs:        null,
                vibrationBaseline:  null,
            },

            // Frame-strip DOM (built lazily when animation:trajectory
            // is set + card.frameStrip === true).
            frameStripEl:    null,
            frameStripParts: null,

            disposed:      false,
        };

        // 4. Load structure + apply initial overlays + render.
        _loadStructure(viewer, state.current);
        _applyStyle(viewer, state.current.style);
        _redrawAllOverlays(state);
        _wirePick(viewer, state);
        viewer.zoomTo();
        viewer.render();
        _refreshInfoLine(state);

        // 4a. Schedule a deferred resize() + render() so the 3Dmol
        // canvas picks up the host's final layout dimensions.  The
        // first render above measures clientWidth/clientHeight
        // synchronously -- which works IF the host was already
        // visible + sized -- but mounts inside a freshly-shown card
        // (e.g. /modify's #viewer with aspect-ratio + min-height
        // CSS) can see 0x0 before layout settles, leaving the WebGL
        // canvas blank.  A double-rAF gives the browser two paint
        // cycles to commit the layout, matching the existing
        // ``molbuilder:inspector:ready`` deferral pattern used on
        // /results.
        if (typeof requestAnimationFrame === "function") {
            requestAnimationFrame(() => requestAnimationFrame(() => {
                if (state.disposed) return;
                try { viewer.resize(); viewer.render(); }
                catch (_) {}
            }));
        }

        // 4b. If the caller supplied opts.animation, set it up now
        //     (after the structure is loaded so baseline coord
        //     capture sees the right atoms).  The setAnimation impl
        //     handles trajectory frame-strip mount + autoplay-unless-
        //     paused semantics.
        if (current.animation) {
            _setAnimationImpl(state, current.animation);
        }

        // 5. Fire onReady on the next microtask so the caller sees a
        //    fully-mounted handle (post-state-init + post-first-render).
        const handle = _buildHandle(state);
        if (typeof opts.onReady === "function") {
            Promise.resolve().then(() => {
                if (state.disposed) return;
                try { opts.onReady(handle); }
                catch (_) {}
            });
        }
        return handle;
    }

    /* ------------------------------------------------------------ */
    /*  Handle builder                                               */
    /* ------------------------------------------------------------ */

    function _buildHandle(state) {
        function setStructure(opts) {
            if (state.disposed) return;
            opts = opts || {};
            const next = Object.assign({}, state.current, {
                xyz:     typeof opts.xyz === "string" ? opts.xyz : state.current.xyz,
                pdb:     typeof opts.pdb === "string" ? opts.pdb : state.current.pdb,
                lattice: opts.lattice !== undefined
                            ? _normaliseLattice(opts.lattice)
                            : state.current.lattice,
            });
            // Clear conflicting source: if caller passes xyz, drop pdb
            // (and vice versa) so a re-call with a different format
            // doesn't end up with both.
            if (opts.xyz && !opts.pdb) next.pdb = null;
            if (opts.pdb && !opts.xyz) next.xyz = null;
            state.current = next;
            // Setting a fresh structure invalidates the animation
            // baseline (different atom count / different topology).
            // Stop the loop + clear the baseline; the caller can
            // re-call setAnimation with new displacements if they
            // want animation against the new structure.
            _stopAnimationLoop(state);
            state._anim.vibrationBaseline = null;
            state.current.animation = null;
            _removeFrameStrip(state);
            _loadStructure(state.viewer, state.current);
            _applyStyle(state.viewer, state.current.style);
            _redrawAllOverlays(state);
            _wirePick(state.viewer, state);
            state.pickedIndices = [];
            state.viewer.zoomTo();
            state.viewer.render();
            _refreshInfoLine(state);
        }

        function setStyle(s) {
            if (state.disposed) return;
            const next = _normaliseStyle(s);
            if (_equalNormalised(state.current.style, next)) return;
            state.current.style = next;
            _applyStyle(state.viewer, next);
            state.viewer.render();
        }

        function setAxes(a) {
            if (state.disposed) return;
            const next = _normaliseAxes(a);
            if (_equalNormalised(state.current.axes, next)) return;
            state.current.axes = next;
            _redrawAxes(state);
            state.viewer.render();
        }

        function setCell(c) {
            if (state.disposed) return;
            const next = _normaliseCell(c);
            if (_equalNormalised(state.current.cell, next)) return;
            state.current.cell = next;
            _redrawCell(state);
            state.viewer.render();
        }

        function setLabels(l) {
            if (state.disposed) return;
            const next = _normaliseLabels(l);
            if (_equalNormalised(state.current.labels, next)) return;
            state.current.labels = next;
            _redrawLabels(state);
            state.viewer.render();
        }

        function setArrows(arr) {
            if (state.disposed) return;
            const next = Array.isArray(arr) ? arr.slice() : [];
            // Idempotence: identity-stringify is fine at this scale.
            if (JSON.stringify(state.current.arrows) === JSON.stringify(next)) return;
            state.current.arrows = next;
            _redrawArrows(state);
            state.viewer.render();
        }

        function setPick(p) {
            if (state.disposed) return;
            const next = _normalisePick(p);
            if (_equalNormalised(state.current.pick, next)) return;
            state.current.pick = next;
            state.pickedIndices = [];
            _redrawPickHalos(state);
            _wirePick(state.viewer, state);
        }

        function getAtomCount() {
            if (state.disposed) return 0;
            return _atomCount(state.viewer);
        }
        function getElements() {
            if (state.disposed) return [];
            return _elements(state.viewer);
        }
        function getPickedIndices() {
            if (state.disposed) return [];
            return state.pickedIndices.slice();
        }

        function refit() {
            if (state.disposed) return;
            try { state.viewer.zoomTo(); state.viewer.render(); }
            catch (_) {}
        }
        function render() {
            if (state.disposed) return;
            try { state.viewer.render(); } catch (_) {}
        }

        function dispose() {
            if (state.disposed) return;
            state.disposed = true;
            // Stop the animation loop FIRST so a late rAF / interval
            // tick doesn't race against the teardown below.
            _stopAnimationLoop(state);
            try {
                if (state.axesHandle) state.axesHandle.clear();
                for (const s of state.cellShapes) state.viewer.removeShape(s);
                for (const l of state.labelHandles) state.viewer.removeLabel(l);
                for (const s of state.arrowShapes) state.viewer.removeShape(s);
                for (const l of state.arrowLabels) state.viewer.removeLabel(l);
                for (const s of state.pickShapes) state.viewer.removeShape(s);
                state.viewer.clear();
            } catch (_) {}
            try {
                if (state.cardEl && state.cardEl.parentNode) {
                    state.cardEl.parentNode.removeChild(state.cardEl);
                }
            } catch (_) {}
        }

        function _viewer3dmol() {
            // Escape hatch — see embedded-viewer.md § 2.2 notice.
            return state.viewer;
        }

        function setAnimation(animation) {
            if (state.disposed) return;
            const next = _normaliseAnimation(animation);
            _setAnimationImpl(state, next);
        }
        function playAnimation() {
            _playImpl(state);
        }
        function pauseAnimation() {
            _pauseImpl(state);
        }
        function isAnimationPlaying() {
            return !state.disposed && state._anim.playing;
        }
        function setAnimationFrame(idx) {
            if (state.disposed) return;
            const a = state.current.animation;
            if (!a || a.kind !== "trajectory") return;
            _stopAnimationLoop(state);
            _showTrajectoryFrame(state, idx);
        }
        function getAnimationFrame() {
            const a = state && state.current && state.current.animation;
            if (!a || a.kind !== "trajectory") return 0;
            return a.currentFrame;
        }

        return {
            setStructure:       setStructure,
            setStyle:           setStyle,
            setAxes:            setAxes,
            setCell:            setCell,
            setLabels:          setLabels,
            setArrows:          setArrows,
            setPick:            setPick,

            setAnimation:       setAnimation,
            playAnimation:      playAnimation,
            pauseAnimation:     pauseAnimation,
            isAnimationPlaying: isAnimationPlaying,
            setAnimationFrame:  setAnimationFrame,
            getAnimationFrame:  getAnimationFrame,

            getAtomCount:       getAtomCount,
            getElements:        getElements,
            getPickedIndices:   getPickedIndices,
            refit:              refit,
            render:             render,
            dispose:            dispose,
            _viewer3dmol:       _viewer3dmol,
        };
    }

    /* ------------------------------------------------------------ */
    /*  Public surface — extends window.molbuilder.viewer            */
    /* ------------------------------------------------------------ */

    root.molbuilder         = root.molbuilder         || {};
    root.molbuilder.viewer  = root.molbuilder.viewer  || {};
    root.molbuilder.viewer.embed              = embed;
    root.molbuilder.viewer._normaliseOpts     = _normaliseOpts;
    root.molbuilder.viewer._normaliseStyle    = _normaliseStyle;
    root.molbuilder.viewer._normaliseAxes     = _normaliseAxes;
    root.molbuilder.viewer._normaliseCell     = _normaliseCell;
    root.molbuilder.viewer._normaliseLabels   = _normaliseLabels;
    root.molbuilder.viewer._normalisePick     = _normalisePick;
    root.molbuilder.viewer._normaliseLattice  = _normaliseLattice;
    root.molbuilder.viewer._normaliseAnimation = _normaliseAnimation;
    root.molbuilder.viewer._equalNormalised   = _equalNormalised;
})(typeof window !== "undefined" ? window : this);
