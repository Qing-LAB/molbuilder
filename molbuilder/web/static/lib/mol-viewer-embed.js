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
        const section = document.createElement("section");
        section.className = CARD_CLASS + (card.className
            ? " " + card.className : "");
        section.setAttribute("data-mol-viewer", "1");

        // Header — only rendered if title or info-line is requested.
        const titleText = card.title;
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
    /*  Public embed() entry point                                   */
    /* ------------------------------------------------------------ */

    function embed(host, opts) {
        opts = opts || {};
        if (!host || !host.appendChild) {
            throw new TypeError(
                "viewer.embed(host, opts): host must be a DOM element"
            );
        }
        if (!opts.xyz && !opts.pdb) {
            throw new TypeError(
                "viewer.embed(host, opts): opts.xyz OR opts.pdb is required"
            );
        }
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
        const state = {
            viewer:        viewer,
            hostEl:        host,
            cardEl:        scaffold.section,
            canvasEl:      scaffold.canvas,
            infoLineEl:    scaffold.infoLineEl,

            current:       _normaliseOpts(opts),

            axesHandle:    null,
            cellShapes:    [],
            labelHandles:  [],
            arrowShapes:   [],
            arrowLabels:   [],
            pickShapes:    [],
            pickedIndices: [],

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

        // Animation stubs — stage 3 lands the real implementation.
        // Defined as no-ops so callers can safely write code against
        // the full handle shape today; calling them logs a debug
        // breadcrumb in dev consoles so a feature-flag pre-flight
        // can detect "animation requested but not yet implemented".
        function _animationNotImplemented() {
            if (root.console) {
                root.console.debug(
                    "[mol-viewer-embed] animation API is stubbed; " +
                    "stage 3 implementation lands the loop."
                );
            }
        }
        const setAnimation        = _animationNotImplemented;
        const playAnimation       = _animationNotImplemented;
        const pauseAnimation      = _animationNotImplemented;
        const isAnimationPlaying  = function () { return false; };
        const setAnimationFrame   = _animationNotImplemented;
        const getAnimationFrame   = function () { return 0; };

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
    root.molbuilder.viewer._equalNormalised   = _equalNormalised;
})(typeof window !== "undefined" ? window : this);
