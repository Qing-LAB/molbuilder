/* mol-viewer-embed.js — the standard embeddable 3D viewer.
 *
 * Contract: docs/protocols/embedded-viewer.md.  The contract is
 * the SOLE source of truth; this file implements it.  When the
 * contract changes, this file changes in the same commit.
 *
 *   window.molbuilder.viewer.embed(host, opts) -> handle
 *
 * Composes the existing primitives (mol-style, mol-axes,
 * mol-format) into a card/panel + 3Dmol viewer that maintains its
 * own drawing.  The pick halo geometry is internal — see
 * ``_redrawPickHalos``; the caller never reaches into 3Dmol directly --
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
    // Phase 6 chrome redesign: canvas backdrop defaults to the page's
    // card colour so the viewer reads as part of the dark theme
    // instead of a bright cut-out.  White stays a preset for
    // publication figures; transparent stays a preset for screenshots
    // composited over other media.
    const DEFAULT_BACKGROUND = "#1d2128";
    const DEFAULT_BG_PRESETS = ["#1d2128", "#ffffff", "transparent"];

    // Closed enums for sync setX input validation (§ 5.3).  Mirrors
    // the option lists in § 3.3 / 3.4 / 3.6 / 3.8 / 3.10 of the
    // embed contract; the corresponding setters dispatch
    // ``invalid_input`` when the supplied value falls outside the
    // closed set and continue with the documented default.
    // Phase 6: VALID_REPS trimmed to the four reps mol-style.js
    // actually implements (was [..., "cross", "cartoon"] — cartoon
    // is a macromolecular renderer that molbuilder doesn't ship;
    // cross was never wired through).
    const VALID_REPS = [
        "stick", "ball-and-stick", "sphere", "line",
    ];
    const VALID_AXES_MODES   = ["auto", "cartesian", "cell"];
    const VALID_PICK_MODES   = ["none", "single", "pair", "multi"];
    const VALID_LABEL_FORMS  = ["index", "name", "element"];

    /* ------------------------------------------------------------ */
    /*  Error model — contract § 3.14 + § 5                          */
    /* ------------------------------------------------------------ */

    // The closed set of error codes per § 5.2.  Used for `instanceof`
    // checks and direct equality in tests / consumers.
    const VIEWER_ERROR_CODES = Object.freeze([
        "missing_dependency",
        "no_structure", "static_structure",
        "no_project", "no_clipboard",
        "no_media_recorder", "no_gif_encoder",
        "io_error", "aborted", "disposed",
        "invalid_input", "unknown",
    ]);

    /**
     * Promote a ViewerError plain object to a real JS Error so
     * sync ``throw`` paths satisfy ``instanceof Error`` checks
     * and get a stack trace (review fix D4).  The {code, message,
     * cause} fields are also copied onto the Error so callers
     * can still switch on ``err.code``.
     */
    function _throwable(code, message, cause) {
        const v = _makeError(code, message, cause);
        const e = new Error(v.message);
        e.code  = v.code;
        e.cause = v.cause;
        e.name  = "ViewerError";
        return e;
    }

    /**
     * Construct a ViewerError per § 3.14.  Uses a plain object
     * (not a JS Error subclass) so the shape round-trips cleanly
     * through Promise.reject and structuredClone, and so consumers
     * can `if (err && err.code === "no_project")` without an
     * `instanceof` check.  For SYNCHRONOUS throw paths use
     * ``_throwable()`` instead so the thrown value is a real Error.
     */
    function _makeError(code, message, cause) {
        if (VIEWER_ERROR_CODES.indexOf(code) < 0) {
            // Defensive: a misspelled code in the embed itself is a
            // programming error.  Fall back to "unknown" + a marker
            // so tests catch it.
            return {
                code:    "unknown",
                message: "(viewer internal: bad error code '" + code
                       + "') " + (message || ""),
                cause:   cause,
            };
        }
        return {
            code:    code,
            message: message || code,
            cause:   cause,
        };
    }

    /**
     * Fire opts.onError per § 5.4 — rate-limited to one fire per
     * code per 500 ms per embed instance.  If onError itself throws,
     * the embed catches and logs to console; the original error
     * path continues uninterrupted.
     */
    function _dispatchError(state, err) {
        if (!err) return;
        try { console.warn("[viewer.embed]", err.code, err.message); }
        catch (_) { /* console may be absent in tests */ }

        const onError = state && state.userOpts && state.userOpts.onError;
        if (typeof onError !== "function") return;

        // Rate-limit: skip if this code fired within the last 500 ms.
        const now = Date.now();
        const lastFires = state._errorLastFires
                       || (state._errorLastFires = {});
        const prev = lastFires[err.code] || 0;
        if (now - prev < 500) return;
        lastFires[err.code] = now;

        try { onError(err); }
        catch (e) {
            try { console.error("[viewer.embed onError threw]", e); }
            catch (_) { /* console may be absent */ }
        }
    }

    function _dispatchInvalidInput(state, message) {
        // Sugar for "input failed a § 5.3 validation check".  Callers
        // pass a one-line message naming the method + the offending
        // field; the embed continues with the documented default and
        // surfaces this through opts.onError.
        _dispatchError(state, _makeError("invalid_input", message));
    }

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
        // Review fix D14: scope is render-only state (the inputs
        // _equalNormalised diffs against to decide if a setX call
        // is a no-op).  Non-render opts (animation, knobs, export,
        // onReady, onError, testInjection, preserveCamera, card)
        // are handled directly in embed() — they don't participate
        // in the idempotence diff path.
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
            overlays:   _normaliseOverlays(opts.overlays),
            knobs:      _normaliseKnobs(opts.knobs),
        };
    }

    function _normaliseStyle(s) {
        // Per § 3.3: ``showLabels`` was REMOVED — labels are controlled
        // exclusively via ``opts.labels`` / ``setLabels()`` (§ 3.6).
        // Two paths to the same state caused precedence ambiguity in
        // the v1 draft; the field is gone (review fix D3).
        s = s || {};
        const rep = (typeof s.rep === "string" && VALID_REPS.includes(s.rep))
                       ? s.rep : "stick";
        const radiusScale = (typeof s.radiusScale === "number"
                             && Number.isFinite(s.radiusScale))
                       ? s.radiusScale : 1.0;
        return {
            rep:          rep,
            radiusScale:  radiusScale,
            colorScheme:  s.colorScheme || null,
            background:   s.background  || DEFAULT_BACKGROUND,
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
        // Per § 3.6 (review fix D2):
        //   atoms  → "all" or number[] — WHICH atoms get a label
        //   format → "index" | "name" | "element" — WHAT each label says
        // The two are independent so a per-index selection can use any
        // text format.  Defaults: atoms="all", format="index".
        if (l === false || l === undefined || l === null) return null;
        if (l === true) {
            return { atoms: "all", format: "index", fontSize: 12,
                     fontColor: null, background: null };
        }
        let atoms  = l.atoms;
        let format = l.format;
        // Validate atoms shape: only "all" or number[] per § 3.6
        // (the v1 legacy sentinels "indices" / "names" were dropped
        // in #240 — no production consumer used them).
        if (atoms !== "all" && !Array.isArray(atoms)) atoms = "all";
        if (Array.isArray(atoms)) {
            atoms = atoms.filter((v) => Number.isInteger(v) && v >= 0);
        }
        // Validate format.
        if (format !== "index" && format !== "name"
            && format !== "element") format = "index";
        return {
            atoms:       atoms,
            format:      format,
            fontSize:    l.fontSize   || 12,
            fontColor:   l.fontColor  || null,
            background:  l.background || null,
        };
    }

    function _normalisePick(p) {
        // Per § 3.8 (review fix D5): the modern shape with halo
        // (object), style override, label format, plus a deprecated-
        // field precedence rule for the legacy haloColor / haloRadius
        // pair.
        //
        // If the caller supplies ``halo`` at all (even ``{}`` or
        // ``false``), the deprecated fields are ignored entirely.
        // Otherwise the legacy {haloColor, haloRadius} is synthesised
        // into the new halo object.
        if (!p || p === false || p.mode === "none") return null;
        const mode = p.mode || "single";

        // Halo normalisation.
        let halo;
        if (p.halo === false) {
            halo = null;   // explicit opt-out
        } else if (p.halo === undefined || p.halo === true) {
            // Phase 5e B7: treat ``halo: true`` as alias for "enabled
            // with defaults", matching the convention used for
            // ``opts.axes`` / ``opts.cell``.  Before this fix,
            // ``true`` fell into the trailing else and became null —
            // so trajectory's atom-pick halos rendered nothing since
            // #236.  Legacy fallback (haloColor / haloRadius) still
            // honored when ``halo`` is undefined.
            const legacyColor  = (p.halo === undefined
                                  && typeof p.haloColor  === "string")
                                  ? p.haloColor  : null;
            const legacyRadius = (p.halo === undefined
                                  && typeof p.haloRadius === "number")
                                  ? p.haloRadius : null;
            if (legacyColor !== null || legacyRadius !== null) {
                halo = {
                    color:   legacyColor  || "#ffd54a",
                    radius:  legacyRadius || 0.6,
                    opacity: 0.5,
                };
            } else {
                // Modern default per § 3.8: halo on by default.
                halo = { color: "#ffd54a", radius: 0.6, opacity: 0.5 };
            }
        } else if (p.halo && typeof p.halo === "object") {
            halo = {
                color:   typeof p.halo.color   === "string" ? p.halo.color   : "#ffd54a",
                radius:  typeof p.halo.radius  === "number" ? p.halo.radius  : 0.6,
                opacity: typeof p.halo.opacity === "number"
                            ? Math.max(0, Math.min(1, p.halo.opacity)) : 0.5,
            };
        } else {
            halo = null;
        }

        // Optional style override on picked atoms.
        let style = null;
        if (p.style && typeof p.style === "object") {
            const s = {};
            let any = false;
            if (typeof p.style.color === "string") {
                s.color = p.style.color; any = true;
            }
            if (typeof p.style.opacity === "number") {
                s.opacity = Math.max(0, Math.min(1, p.style.opacity));
                any = true;
            }
            if (typeof p.style.radiusScale === "number") {
                s.radiusScale = p.style.radiusScale; any = true;
            }
            if (any) style = s;
        }

        // Auto-label format.  Default: "index" per § 3.8.  False
        // explicitly disables; "name" / "element" are documented
        // alternatives.  Anything else falls back to "index".
        let label;
        if (p.label === false) {
            label = false;
        } else if (p.label === undefined) {
            label = "index";
        } else if (p.label === "index" || p.label === "name"
                   || p.label === "element") {
            label = p.label;
        } else {
            label = "index";
        }

        return {
            mode:       mode,
            halo:       halo,
            style:      style,
            label:      label,
            // Legacy compat: still expose the flat fields as derived
            // values for code that reads them (will drop once all
            // call sites move to the nested form).
            haloColor:  halo ? halo.color  : null,
            haloRadius: halo ? halo.radius : null,
            onPick:     typeof p.onPick === "function" ? p.onPick : null,
        };
    }

    /**
     * Normalise an OverlaySpec per § 3.12.  Each AtomOverlaySpec
     * entry is reshaped into a uniform internal form:
     *   { selectorKind: "indices"|"elements"|"residues",
     *     selectorValue: number[]|string[],
     *     style:  {...}|null, halo: {...}|null, marker: {...}|null }
     *
     * Selectors must specify EXACTLY ONE of indices / elements /
     * residues; entries with zero or multiple selectors are
     * dropped (the embed dispatches an invalid_input error per
     * § 5.3 at render time).
     *
     * Entries with no style/halo/marker are dropped (no-op).
     */
    function _normaliseOverlays(o) {
        if (!o || typeof o !== "object") return null;
        const inAtoms = Array.isArray(o.atoms) ? o.atoms : [];
        const atoms = [];
        for (const entry of inAtoms) {
            if (!entry || typeof entry !== "object") continue;

            // Resolve selector (exactly one of indices / elements / residues).
            const sel = _normaliseOverlaySelector(entry);
            if (sel === null) continue;

            // Resolve treatments (at least one of style / halo / marker).
            const style  = _normaliseOverlayStyle(entry.style);
            const halo   = _normaliseOverlayHalo(entry.halo);
            const marker = _normaliseOverlayMarker(entry.marker);
            if (!style && !halo && !marker) continue;

            atoms.push({
                selectorKind:  sel.kind,
                selectorValue: sel.value,
                style:  style,
                halo:   halo,
                marker: marker,
            });
        }
        if (atoms.length === 0) return null;
        return { atoms: atoms };
    }

    function _normaliseOverlaySelector(entry) {
        let count = 0;
        let result = null;
        if (Array.isArray(entry.indices)) {
            const vals = entry.indices.filter(
                (v) => Number.isInteger(v) && v >= 0);
            if (vals.length) { count++; result = { kind: "indices",  value: vals }; }
        }
        if (Array.isArray(entry.elements)) {
            const vals = entry.elements.filter(
                (v) => typeof v === "string" && v.length > 0);
            if (vals.length) { count++; result = { kind: "elements", value: vals }; }
        }
        if (Array.isArray(entry.residues)) {
            const vals = entry.residues.filter(
                (v) => Number.isInteger(v) && v >= 0);
            if (vals.length) { count++; result = { kind: "residues", value: vals }; }
        }
        // Exactly one selector required per § 3.12.
        if (count !== 1) return null;
        return result;
    }

    function _normaliseOverlayStyle(s) {
        if (!s || typeof s !== "object") return null;
        const out = {};
        let any = false;
        if (typeof s.rep === "string") {
            out.rep = s.rep; any = true;
        }
        if (typeof s.radiusScale === "number" && Number.isFinite(s.radiusScale)) {
            out.radiusScale = s.radiusScale; any = true;
        }
        if (typeof s.color === "string") {
            out.color = s.color; any = true;
        }
        if (typeof s.opacity === "number" && Number.isFinite(s.opacity)) {
            out.opacity = Math.max(0, Math.min(1, s.opacity)); any = true;
        }
        return any ? out : null;
    }

    function _normaliseOverlayHalo(h) {
        if (!h || typeof h !== "object") return null;
        return {
            color:   typeof h.color === "string"  ? h.color   : "#6ba6ff",
            radius:  typeof h.radius === "number" ? h.radius  : 0.6,
            opacity: typeof h.opacity === "number"
                       ? Math.max(0, Math.min(1, h.opacity)) : 0.5,
        };
    }

    function _normaliseOverlayMarker(m) {
        if (!m || typeof m !== "object") return null;
        const kind = m.kind;
        if (kind !== "lock" && kind !== "star" && kind !== "dot") return null;
        return {
            kind:  kind,
            color: typeof m.color === "string" ? m.color : "#222",
        };
    }

    /**
     * Normalise a KnobBarOpts object per § 3.10.  Returns the
     * canonical internal form used by the knob-bar builder.
     * Edge cases pinned per § 3.10 (Phase 6):
     *   - true / undefined → both menus visible with all submenu items.
     *   - false            → bar hidden entirely (null return).
     *   - object           → per-menu / per-item suppression; missing
     *                        keys default to true.
     */
    function _normaliseKnobs(k) {
        // Phase 6: two-menu shape — ``view`` collects style / labels /
        // background / axes / reset; ``export`` collects save-to-
        // project / download × {.xyz, .pdb, .png, .gif, .webm}.  The
        // top-level ``view`` / ``export`` keys (boolean) suppress the
        // matching menu entirely; per-item keys (style / labels /
        // background / axes / reset) suppress submenu rows.  Old
        // shape fields (``screenshot`` / ``labelsFormats`` /
        // ``position`` / ``compact``) are silently dropped — there
        // are no out-of-tree consumers.
        if (k === false) return null;
        const opts = (k && typeof k === "object" && k !== true) ? k : {};

        return {
            view:       opts.view       !== false,
            export:     opts.export     !== false,
            style:      opts.style      !== false,
            labels:     opts.labels     !== false,
            background: opts.background !== false,
            axes:       opts.axes       !== false,
            reset:      opts.reset      !== false,
            backgroundPresets: Array.isArray(opts.backgroundPresets)
                ? opts.backgroundPresets.slice()
                : DEFAULT_BG_PRESETS.slice(),
            backgroundAllowCustom: opts.backgroundAllowCustom !== false,
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

        // ``card.bare`` removed 2026-06-03 after the 5 consumer-site
        // migrations completed; silently ignored if still passed.
        const section = document.createElement("section");
        section.className = CARD_CLASS
                          + (card.className ? " " + card.className : "");
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

        // Standard knob bar per § 6.2 — built unless explicitly
        // disabled.  Sits between header and canvas; click handlers
        // wire to the handle methods in a second pass after
        // _buildHandle finishes.
        let knobsEl = null;
        const knobs = _normaliseKnobs(opts.knobs);
        if (knobs) {
            knobsEl = _buildKnobBarDOM(knobs);
            section.appendChild(knobsEl);
        }

        const canvas = document.createElement("div");
        canvas.className = CANVAS_CLASS;
        canvas.style.height = card.height || DEFAULT_HEIGHT;
        section.appendChild(canvas);

        // Phase 5d C2: don't return ``knobs`` — it's already in
        // state.current.knobs (the source of truth); returning it
        // here forced two parallel copies of the same data.
        return { section, canvas, infoLineEl, knobsEl };
    }

    /* ------------------------------------------------------------ */
    /*  Knob bar DOM (§ 6.2) — built before handle exists; wired    */
    /*  in a second pass after _buildHandle completes               */
    /* ------------------------------------------------------------ */

    function _buildKnobBarDOM(knobs) {
        // Phase 6: two-menu structure — View + Export — replaces the
        // 7-flat-knob bar.  Each menu opens an expandable panel
        // containing labelled sections.  HTML <details>/<summary>
        // provides native open/close + accessibility for free; mutual
        // exclusion is wired in _wireKnobBar.
        const bar = document.createElement("div");
        bar.className = "mol-viewer-knobs";
        bar.setAttribute("role", "toolbar");
        bar.setAttribute("aria-label", "Viewer controls");

        if (knobs.view) {
            bar.appendChild(_buildViewMenu(knobs));
        }
        if (knobs.export) {
            bar.appendChild(_buildExportMenu());
        }
        return bar;
    }

    function _buildViewMenu(knobs) {
        const det = document.createElement("details");
        det.className = "mol-viewer-knob mol-viewer-menu mol-viewer-menu-view";
        const sum = document.createElement("summary");
        sum.textContent = "View";
        det.appendChild(sum);
        const body = document.createElement("div");
        body.className = "mol-viewer-menu-body";

        if (knobs.style) {
            const sect = _menuSection("style", "Style");
            const row = document.createElement("div");
            row.className = "mol-viewer-rep-row";
            const reps = [
                ["stick",          "Stick"],
                ["ball-and-stick", "Ball & stick"],
                ["sphere",         "Sphere"],
                ["line",           "Line"],
            ];
            for (const [v, label] of reps) {
                const b = document.createElement("button");
                b.type = "button";
                b.className = "mol-viewer-rep-btn";
                b.setAttribute("data-rep", v);
                b.textContent = label;
                row.appendChild(b);
            }
            sect.appendChild(row);
            body.appendChild(sect);
        }

        if (knobs.labels) {
            const sect = _menuSection("labels", "Labels");
            const b = document.createElement("button");
            b.type = "button";
            b.className = "mol-viewer-toggle";
            b.setAttribute("data-action", "labels");
            b.setAttribute("aria-pressed", "false");
            b.textContent = "Show labels";
            sect.appendChild(b);
            body.appendChild(sect);
        }

        if (knobs.background) {
            const sect = _menuSection("background", "Background");
            const row = document.createElement("div");
            row.className = "mol-viewer-bg-row";
            for (const c of knobs.backgroundPresets) {
                const b = document.createElement("button");
                b.type = "button";
                b.className = "mol-viewer-bg-swatch";
                b.setAttribute("data-color", c);
                if (c === "transparent") {
                    b.classList.add("is-transparent");
                    b.setAttribute("aria-label", "Transparent");
                    b.title = "Transparent";
                } else {
                    b.style.background = c;
                    b.setAttribute("aria-label", "Background " + c);
                }
                row.appendChild(b);
            }
            if (knobs.backgroundAllowCustom) {
                // Styled label wraps the native <input type="color">
                // so the OS picker still opens on click, but the
                // visible chip is a swatch-sized button that matches
                // the preset row.  The input itself is sized to fill
                // the label so click-target is the whole chip.
                const lbl = document.createElement("label");
                lbl.className = "mol-viewer-bg-custom";
                lbl.setAttribute("aria-label", "Custom colour");
                lbl.title = "Custom colour";
                const inp = document.createElement("input");
                inp.type = "color";
                inp.setAttribute("data-knob", "background-custom");
                lbl.appendChild(inp);
                row.appendChild(lbl);
            }
            sect.appendChild(row);
            body.appendChild(sect);
        }

        if (knobs.axes) {
            const sect = _menuSection("axes", "Axes");
            const b = document.createElement("button");
            b.type = "button";
            b.className = "mol-viewer-toggle";
            b.setAttribute("data-action", "axes");
            b.setAttribute("aria-pressed", "false");
            b.textContent = "Show axes";
            sect.appendChild(b);
            body.appendChild(sect);
        }

        if (knobs.reset) {
            const sect = _menuSection("reset", null);
            const b = document.createElement("button");
            b.type = "button";
            b.className = "mol-viewer-action";
            b.setAttribute("data-action", "reset");
            b.textContent = "Reset view";
            sect.appendChild(b);
            body.appendChild(sect);
        }

        det.appendChild(body);
        return det;
    }

    function _buildExportMenu() {
        const det = document.createElement("details");
        det.className = "mol-viewer-knob mol-viewer-menu mol-viewer-menu-export";
        const sum = document.createElement("summary");
        sum.textContent = "Export";
        det.appendChild(sum);
        const body = document.createElement("div");
        body.className = "mol-viewer-menu-body";

        // Each target opens to the same 5 format buttons.  Animation
        // formats (gif / webm) default-hidden; toggled visible via
        // _syncKnobBarToAnimation when a trajectory or vibration
        // animation is mounted (covers § 4.3 onMount path).
        const FORMATS = [
            { kind: "structure", format: "xyz",  label: ".xyz" },
            { kind: "structure", format: "pdb",  label: ".pdb" },
            { kind: "image",     format: "png",  label: ".png" },
            { kind: "animation", format: "gif",  label: ".gif" },
            { kind: "animation", format: "webm", label: ".webm" },
        ];
        for (const target of ["project", "download"]) {
            const sect = _menuSection(
                "target-" + target,
                target === "project" ? "Save to project" : "Download");
            const row = document.createElement("div");
            row.className = "mol-viewer-export-row";
            for (const f of FORMATS) {
                const b = document.createElement("button");
                b.type = "button";
                b.className = "mol-viewer-export-btn";
                b.setAttribute("data-target", target);
                b.setAttribute("data-kind",   f.kind);
                b.setAttribute("data-format", f.format);
                if (f.kind === "animation") b.hidden = true;
                b.textContent = f.label;
                row.appendChild(b);
            }
            sect.appendChild(row);
            body.appendChild(sect);
        }

        det.appendChild(body);
        return det;
    }

    function _menuSection(key, heading) {
        const sect = document.createElement("section");
        sect.className = "mol-viewer-menu-section";
        sect.setAttribute("data-section", key);
        if (heading) {
            const h = document.createElement("h4");
            h.className = "mol-viewer-menu-heading";
            h.textContent = heading;
            sect.appendChild(h);
        }
        return sect;
    }

    /* ------------------------------------------------------------ */
    /*  Knob bar wiring — called after handle is built              */
    /* ------------------------------------------------------------ */

    function _syncKnobBarToStyle(state) {
        // Programmatic→UI sync per § 4.1 + § 6.2 (Phase 6 rewrite):
        // after setStyle(), the matching rep button in View → Style
        // gets ``is-active``.  No-op when the menu / section is
        // suppressed.
        if (!state.scaffold || !state.scaffold.knobsEl) return;
        const rep = state.current.style && state.current.style.rep;
        for (const btn of state.scaffold.knobsEl
                                .querySelectorAll(".mol-viewer-rep-btn")) {
            btn.classList.toggle("is-active",
                btn.getAttribute("data-rep") === rep);
        }
    }

    function _syncKnobBarToLabels(state) {
        // Phase 6: Labels is now a single On/Off toggle.  ``labels
        // === null`` → unpressed; any LabelOpts → pressed.  Format is
        // a mount-time config; user-facing UI no longer picks it.
        if (!state.scaffold || !state.scaffold.knobsEl) return;
        const btn = state.scaffold.knobsEl.querySelector(
            '.mol-viewer-toggle[data-action="labels"]');
        if (!btn) return;
        btn.setAttribute("aria-pressed",
            state.current.labels ? "true" : "false");
    }

    function _syncKnobBarToBackground(state) {
        // Programmatic→UI sync per § 4.1 + § 6.2: after
        // setBackground(), the matching preset swatch gets
        // ``is-active``.  Custom colours that don't match a preset
        // leave every swatch unmarked (the picker chip carries the
        // value).  Compare lower-cased so a host passing "#FFFFFF"
        // matches the lower-case "#ffffff" preset.
        if (!state.scaffold || !state.scaffold.knobsEl) return;
        const raw = state.current.style && state.current.style.background;
        const cur = typeof raw === "string" ? raw.toLowerCase() : raw;
        for (const btn of state.scaffold.knobsEl
                                .querySelectorAll(".mol-viewer-bg-swatch")) {
            const swatch = btn.getAttribute("data-color");
            const sw = typeof swatch === "string"
                ? swatch.toLowerCase() : swatch;
            btn.classList.toggle("is-active", sw === cur);
        }
    }

    function _syncKnobBarToAxes(state) {
        // Phase 6: Axes lives inside View → Axes as a toggle button.
        if (!state.scaffold || !state.scaffold.knobsEl) return;
        const btn = state.scaffold.knobsEl.querySelector(
            '.mol-viewer-toggle[data-action="axes"]');
        if (!btn) return;
        btn.setAttribute("aria-pressed",
            state.current.axes ? "true" : "false");
    }

    function _syncKnobBarToAnimation(state) {
        // Phase 6: gif / webm Export buttons are hidden by default
        // and revealed only when a trajectory or vibration animation
        // is currently mounted.  PNG, XYZ, PDB stay visible
        // unconditionally — they don't need an animation.
        if (!state.scaffold || !state.scaffold.knobsEl) return;
        const hasAnim = !!state.current.animation;
        const sel = ".mol-viewer-export-btn[data-kind=\"animation\"]";
        for (const btn of state.scaffold.knobsEl.querySelectorAll(sel)) {
            btn.hidden = !hasAnim;
        }
    }

    function _wireKnobBar(state, bar, knobs) {
        const handle = state.handle;

        // ----- View menu --------------------------------------------
        const viewDet = bar.querySelector(".mol-viewer-menu-view");

        // Style: button group of 4 reps; click sets that rep.
        for (const btn of bar.querySelectorAll(".mol-viewer-rep-btn")) {
            btn.addEventListener("click", () => {
                handle.setStyle({ rep: btn.getAttribute("data-rep") });
            });
        }

        // Labels: single On/Off toggle.  When turning on, restore
        // the host-configured format (state.lastLabelsConfig
        // remembers the last LabelOpts setLabels saw; falls back to
        // the documented {atoms:"all", format:"index"} default).
        const labelsBtn = bar.querySelector(
            '.mol-viewer-toggle[data-action="labels"]');
        if (labelsBtn) {
            labelsBtn.addEventListener("click", () => {
                if (state.current.labels) {
                    handle.setLabels(false);
                } else {
                    handle.setLabels(state.lastLabelsConfig
                                     || { atoms: "all", format: "index" });
                }
            });
        }

        // Background: preset swatches + styled custom-colour chip.
        for (const btn of bar.querySelectorAll(".mol-viewer-bg-swatch")) {
            btn.addEventListener("click", () => {
                handle.setBackground(btn.getAttribute("data-color"));
            });
        }
        const customInput = bar.querySelector(
            '[data-knob="background-custom"]');
        if (customInput) {
            customInput.addEventListener("input", () => {
                handle.setBackground(customInput.value);
            });
        }

        // Axes: toggle.
        const axesBtn = bar.querySelector(
            '.mol-viewer-toggle[data-action="axes"]');
        if (axesBtn) {
            axesBtn.addEventListener("click", () => {
                handle.setAxes(state.current.axes ? false : true);
            });
        }

        // Reset view.
        const resetBtn = bar.querySelector(
            '.mol-viewer-action[data-action="reset"]');
        if (resetBtn) {
            resetBtn.addEventListener("click", () => handle.refit());
        }

        // ----- Export menu ------------------------------------------
        const exportDet = bar.querySelector(".mol-viewer-menu-export");
        if (exportDet) {
            for (const btn of exportDet.querySelectorAll(
                                ".mol-viewer-export-btn")) {
                btn.addEventListener("click", () => {
                    const kind   = btn.getAttribute("data-kind");
                    const target = btn.getAttribute("data-target");
                    const format = btn.getAttribute("data-format");
                    let p;
                    if (kind === "structure") {
                        p = handle.exportData({
                            target: target, format: format });
                    } else if (kind === "image") {
                        p = handle.screenshot({ target: target });
                    } else if (kind === "animation") {
                        p = handle.exportAnimation({
                            format: format || "webm",
                            target: target || "download",
                        });
                    }
                    if (p) p.catch((err) => _dispatchError(state, err));
                    exportDet.open = false;
                });
            }
        }

        // Mutual exclusion: opening one top-level menu closes the
        // other (click/tap rule per § 6.2).
        const allDetails = bar.querySelectorAll(
            "details.mol-viewer-menu");
        for (const d of allDetails) {
            d.addEventListener("toggle", () => {
                if (d.open) {
                    for (const other of allDetails) {
                        if (other !== d) other.open = false;
                    }
                }
            });
        }

        // Keyboard shortcuts per § 6.2 (Phase 6 simplified):
        //   V toggles the View menu,
        //   X toggles the Export menu,
        //   R is Reset view (always),
        //   Esc closes any open menu.
        // Per-knob shortcuts (L / A / B / E) are gone; the deep menu
        // structure makes them less useful and the surface is now
        // discoverable by hover instead.
        if (!state.cardEl) return;
        state.cardEl.addEventListener("keydown", (e) => {
            if (state.disposed) return;
            const t = e.target;
            const tag = t && t.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA"
                || (t && t.isContentEditable)) return;

            const openMenu = bar.querySelector(
                "details.mol-viewer-menu[open]");
            const key = (e.key || "").toLowerCase();

            if (e.key === "Escape" && openMenu) {
                openMenu.open = false;
                e.preventDefault();
                return;
            }
            if (key === "r" && resetBtn) {
                resetBtn.click(); e.preventDefault();
            } else if (key === "v" && viewDet) {
                viewDet.open = !viewDet.open; e.preventDefault();
            } else if (key === "x" && exportDet) {
                exportDet.open = !exportDet.open; e.preventDefault();
            }
        });
        // Card must accept focus for keyboard handling to work.
        if (!state.cardEl.hasAttribute("tabindex")) {
            state.cardEl.setAttribute("tabindex", "0");
        }

        // Seed the initial active-affordance state so the menus
        // reflect mount-time opts (and any later setKnobs() rebuild
        // lands consistent with the live state).
        _syncKnobBarToStyle(state);
        _syncKnobBarToLabels(state);
        _syncKnobBarToBackground(state);
        _syncKnobBarToAxes(state);
        _syncKnobBarToAnimation(state);
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
            // mol-style.js's switch uses ``ballstick`` (one word) for
            // historical reasons.  The embed contract spells the rep
            // out hyphenated (``ball-and-stick``) per the picker
            // labels; translate at the boundary so "Ball & stick"
            // doesn't silently fall through to the stick default.
            const repForSpec = style.rep === "ball-and-stick"
                ? "ballstick" : style.rep;
            spec = styleApi.spec({
                rep:         repForSpec,
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
        // Per § 3.6 (review fix D2): opts.atoms picks WHICH atoms;
        // opts.format picks WHAT text each label says.
        if (!opts) return [];
        const handles = [];
        const fmt = opts.format || "index";
        try {
            const model = viewer.getModel();
            const atoms = model ? model.selectedAtoms({}) : [];
            // Build the "draw this atom?" predicate from opts.atoms.
            let shouldDraw;
            if (Array.isArray(opts.atoms)) {
                const set = new Set(opts.atoms);
                shouldDraw = (i) => set.has(i);
            } else {
                // "all" (default after normalisation) labels every atom.
                shouldDraw = () => true;
            }
            for (let i = 0; i < atoms.length; i++) {
                if (!shouldDraw(i)) continue;
                const a = atoms[i];
                let text;
                if (fmt === "name") {
                    text = a.atom || a.name || a.elem || String(i);
                } else if (fmt === "element") {
                    text = a.elem || a.element || "?";
                } else {
                    text = String(i);   // "index" (default)
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

    function _wireInteractionHooks(state, canvasEl, interaction) {
        // Canonical drag detection: fire onDragStart exactly once
        // per gesture after the pointer moves more than
        // dragThresholdPx (default 4) from the press point; fire
        // onDragEnd on mouseup or mouseleave for any gesture where
        // onDragStart fired.  Consumers receive the modifier state
        // so they can filter (e.g. plain-drag = rotate, ctrl-drag =
        // pan).  All listeners are scoped to the canvas host so
        // they don't leak; dispose() removes them via
        // _interactionTeardown.
        if (!interaction || typeof interaction !== "object"
            || !canvasEl) return;
        const onStart = typeof interaction.onDragStart === "function"
            ? interaction.onDragStart : null;
        const onEnd   = typeof interaction.onDragEnd === "function"
            ? interaction.onDragEnd : null;
        if (!onStart && !onEnd) return;
        const threshold = typeof interaction.dragThresholdPx === "number"
            && interaction.dragThresholdPx > 0
            ? interaction.dragThresholdPx : 4;
        let pressX = null, pressY = null, pressMods = null;
        let lastX  = null, lastY  = null;
        let dragging = false;
        function _modifiers(e) {
            return {
                ctrl:  !!e.ctrlKey, shift: !!e.shiftKey,
                alt:   !!e.altKey,  meta:  !!e.metaKey,
            };
        }
        function _onMouseDown(e) {
            if (e.button !== 0) { pressX = pressY = null; return; }
            pressX = e.clientX; pressY = e.clientY;
            pressMods = _modifiers(e);
            dragging = false;
        }
        function _onMouseMove(e) {
            if (pressX === null) return;
            // Track the latest cursor position so mouseleave can
            // report the where-the-pointer-just-was coords (Phase 5d
            // C8) instead of falling back to the press point.
            lastX = e.clientX;
            lastY = e.clientY;
            if (dragging) return;
            const dx = e.clientX - pressX;
            const dy = e.clientY - pressY;
            if (dx * dx + dy * dy < threshold * threshold) return;
            dragging = true;
            if (onStart) {
                try { onStart({
                    x: pressX, y: pressY,
                    modifiers: pressMods,
                }); }
                catch (err) {
                    try { console.error(
                        "[viewer.embed] onDragStart threw:", err); }
                    catch (_) {}
                }
            }
        }
        function _onMouseUp(e) {
            if (dragging && onEnd) {
                try { onEnd({
                    x: e.clientX, y: e.clientY,
                    modifiers: pressMods,
                }); }
                catch (err) {
                    try { console.error(
                        "[viewer.embed] onDragEnd threw:", err); }
                    catch (_) {}
                }
            }
            pressX = pressY = lastX = lastY = null;
            pressMods = null;
            dragging = false;
        }
        function _onMouseLeave() {
            // Phase 5d C8: report the last-known cursor coords
            // (where the pointer was when it left the canvas) on
            // mouseleave instead of falling back to the press
            // point.  Closer to the documented "end-point coords"
            // contract.  Falls back to press if no mousemove has
            // fired since mousedown (shouldn't happen given the
            // drag-detection threshold but kept defensive).
            if (dragging && onEnd) {
                try { onEnd({
                    x: lastX !== null ? lastX : pressX,
                    y: lastY !== null ? lastY : pressY,
                    modifiers: pressMods,
                }); }
                catch (err) {
                    try { console.error(
                        "[viewer.embed] onDragEnd threw:", err); }
                    catch (_) {}
                }
            }
            pressX = pressY = lastX = lastY = null;
            pressMods = null;
            dragging = false;
        }
        canvasEl.addEventListener("mousedown",  _onMouseDown);
        canvasEl.addEventListener("mousemove",  _onMouseMove);
        canvasEl.addEventListener("mouseup",    _onMouseUp);
        canvasEl.addEventListener("mouseleave", _onMouseLeave);
        state._interactionTeardown = function () {
            canvasEl.removeEventListener("mousedown",  _onMouseDown);
            canvasEl.removeEventListener("mousemove",  _onMouseMove);
            canvasEl.removeEventListener("mouseup",    _onMouseUp);
            canvasEl.removeEventListener("mouseleave", _onMouseLeave);
        };
    }

    function _wirePick(viewer, state) {
        if (!state.current.pick) return;
        // Review fix U7: avoid re-registering setClickable on every
        // setStructure call.  The captured handler already reads
        // state.current.pick dynamically, so the wiring only needs
        // to happen ONCE per atom-load cycle.  pickWired is reset
        // by _loadStructure (which swaps the model out from under
        // 3Dmol and invalidates the per-atom clickable flags).
        if (state.pickWired) return;
        try {
            viewer.setClickable({}, true, function (atom, _viewer, _evt) {
                if (state.disposed) return;
                if (!state.current.pick) return;
                const idx = atom && (atom.index != null ? atom.index : atom.serial);
                if (typeof idx !== "number") return;
                _togglePick(state, idx);
            });
            state.pickWired = true;
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
        // Clear previous halos + auto-labels (review fix D5).
        for (const s of state.pickShapes) {
            try { state.viewer.removeShape(s); } catch (_) {}
        }
        for (const l of state.pickLabels) {
            try { state.viewer.removeLabel(l); } catch (_) {}
        }
        state.pickShapes = [];
        state.pickLabels = [];
        const pick = state.current.pick;
        if (!pick) {
            state.viewer.render();
            return;
        }
        try {
            const model = state.viewer.getModel();
            const atoms = model ? model.selectedAtoms({}) : [];
            for (const idx of state.pickedIndices) {
                if (idx < 0 || idx >= atoms.length) continue;
                const a = atoms[idx];
                // Halo overlay.
                if (pick.halo) {
                    const halo = state.viewer.addSphere({
                        center:  { x: a.x, y: a.y, z: a.z },
                        radius:  pick.halo.radius,
                        color:   pick.halo.color,
                        opacity: pick.halo.opacity,
                    });
                    state.pickShapes.push(halo);
                }
                // Auto-label per § 3.8 (default "index").
                if (pick.label) {
                    let text;
                    if (pick.label === "name") {
                        text = a.atom || a.name || a.elem || String(idx);
                    } else if (pick.label === "element") {
                        text = a.elem || a.element || "?";
                    } else {
                        text = String(idx);
                    }
                    const lbl = state.viewer.addLabel(text, {
                        position:          { x: a.x, y: a.y, z: a.z },
                        fontSize:          11,
                        fontColor:         "#fff",
                        backgroundColor:   pick.halo
                                             ? pick.halo.color : "#ffd54a",
                        backgroundOpacity: 0.85,
                        inFront:           true,
                    });
                    state.pickLabels.push(lbl);
                }
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
        // Per § 3.12 layering: per-atom style overlays apply BEFORE
        // labels/arrows/markers/halos (which sit on top), and pick
        // halos draw above overlay halos.  Order:
        //   axes / cell -> per-atom-style overlays -> labels ->
        //   arrows -> overlay halos -> overlay markers -> pick halos.
        _redrawOverlayStyles(state);
        _redrawLabels(state);
        _redrawArrows(state);
        _redrawOverlayHalosAndMarkers(state);
        _redrawPickHalos(state);
    }

    /* ------------------------------------------------------------ */
    /*  Overlay rendering — § 3.12                                   */
    /* ------------------------------------------------------------ */

    /**
     * Build a 3Dmol stylespec selector from a normalised overlay
     * selector + the loaded atoms list.  Returns both the 3Dmol
     * selector (for setStyle calls) and the resolved 0-based
     * index list (for per-atom halo / marker positioning).
     */
    /**
     * Map an opts.indices array (0-based atom indices) to a
     * 3Dmol selector dict.  An empty / missing array becomes
     * ``{}`` (select all atoms — 3Dmol's "no filter" default).
     * Shared between handle.refit({indices}) and
     * handle.setPivot({indices}).
     */
    function _selectionFromIndices(indices) {
        if (!Array.isArray(indices) || indices.length === 0) return {};
        // 3Dmol's ``index`` selector takes 0-based indices; ``serial``
        // is 1-based.  We expose 0-based externally to match every
        // other handle method (overlays, animation frames).
        return { index: indices.slice() };
    }

    function _resolveOverlaySelector(sel, atoms) {
        const out = { spec: null, indices: [] };
        if (sel.selectorKind === "indices") {
            // Filter to atoms that actually exist.
            const valid = sel.selectorValue.filter(
                (i) => i >= 0 && i < atoms.length);
            out.spec    = { index: valid };
            out.indices = valid;
        } else if (sel.selectorKind === "elements") {
            const set = new Set(sel.selectorValue.map((s) => s.toUpperCase()));
            const resolved = [];
            for (let i = 0; i < atoms.length; i++) {
                const e = (atoms[i].elem || "").toUpperCase();
                if (set.has(e)) resolved.push(i);
            }
            out.spec    = { elem: Array.from(set) };
            out.indices = resolved;
        } else if (sel.selectorKind === "residues") {
            const set = new Set(sel.selectorValue);
            const resolved = [];
            for (let i = 0; i < atoms.length; i++) {
                if (set.has(atoms[i].resi)) resolved.push(i);
            }
            out.spec    = { resi: Array.from(set) };
            out.indices = resolved;
        }
        return out;
    }

    /**
     * Per-atom style overrides.  3Dmol's setStyle replaces the
     * stylespec for matching atoms (later calls win), so iterating
     * the array in order naturally implements § 3.12 layering
     * rule 2 ("later entry wins for overlapping atom sets").
     *
     * Style overlays do NOT use a removable handle — they're
     * "baked into" 3Dmol's style state and reset only when
     * _applyStyle is called next (which we do via the redraw
     * pipeline).
     */
    function _redrawOverlayStyles(state) {
        // First, restore base style for ALL atoms.  This wipes any
        // overlay-style overrides from a prior frame so the new
        // overlay set starts clean.  _applyStyle already did this
        // at structure-load + setStyle time, but we call it again
        // here for setOverlays / setAtomStyle paths.
        _applyStyle(state.viewer, state.current.style);

        if (!state.current.overlays) return;
        const styleApi = (root.molbuilder || {}).style;
        let atoms = [];
        try {
            const model = state.viewer.getModel();
            atoms = model ? model.selectedAtoms({}) : [];
        } catch (_) {}
        if (atoms.length === 0) return;

        for (const entry of state.current.overlays.atoms) {
            if (!entry.style) continue;
            const sel = _resolveOverlaySelector(entry, atoms);
            if (sel.indices.length === 0) continue;

            // Build a 3Dmol stylespec from the overlay's style block.
            // We start from the base stylespec so partial overrides
            // (e.g. just color) compose with the base rep correctly.
            const baseSpec = styleApi && typeof styleApi.spec === "function"
                ? styleApi.spec({
                    rep:         entry.style.rep || state.current.style.rep,
                    scale:       typeof entry.style.radiusScale === "number"
                                   ? entry.style.radiusScale
                                   : state.current.style.radiusScale,
                  })
                : { stick: {} };

            // Apply color + opacity to every rep key in the spec.
            for (const k of Object.keys(baseSpec)) {
                const sub = baseSpec[k];
                if (entry.style.color   !== undefined) sub.color   = entry.style.color;
                if (entry.style.opacity !== undefined) sub.opacity = entry.style.opacity;
            }

            try { state.viewer.setStyle(sel.spec, baseSpec); }
            catch (_) {}
        }
    }

    /**
     * Halos + markers draw as removable 3Dmol shapes / labels so
     * they can be cleared cleanly on next redraw or dispose.
     */
    function _redrawOverlayHalosAndMarkers(state) {
        // Clear previous overlay halo / marker shapes.
        for (const s of state.overlayHaloShapes) {
            try { state.viewer.removeShape(s); } catch (_) {}
        }
        for (const l of state.overlayMarkerLabels) {
            try { state.viewer.removeLabel(l); } catch (_) {}
        }
        state.overlayHaloShapes   = [];
        state.overlayMarkerLabels = [];
        if (!state.current.overlays) return;
        let atoms = [];
        try {
            const model = state.viewer.getModel();
            atoms = model ? model.selectedAtoms({}) : [];
        } catch (_) {}
        if (atoms.length === 0) return;

        for (const entry of state.current.overlays.atoms) {
            if (!entry.halo && !entry.marker) continue;
            const sel = _resolveOverlaySelector(entry, atoms);
            for (const idx of sel.indices) {
                const a = atoms[idx];
                if (entry.halo) {
                    try {
                        const halo = state.viewer.addSphere({
                            center:  { x: a.x, y: a.y, z: a.z },
                            radius:  entry.halo.radius,
                            color:   entry.halo.color,
                            opacity: entry.halo.opacity,
                        });
                        state.overlayHaloShapes.push(halo);
                    } catch (_) {}
                }
                if (entry.marker) {
                    try {
                        const glyph = _markerGlyph(entry.marker.kind);
                        const lbl = state.viewer.addLabel(glyph, {
                            position:          { x: a.x, y: a.y, z: a.z },
                            fontSize:          14,
                            fontColor:         entry.marker.color,
                            backgroundOpacity: 0,
                            inFront:           true,
                        });
                        state.overlayMarkerLabels.push(lbl);
                    } catch (_) {}
                }
            }
        }
    }

    /**
     * Convert a data: URL (e.g. canvas.toDataURL output) to a
     * Blob.  Synchronous via atob; sufficient for the modest PNGs
     * 3Dmol produces.  Returns null on parse failure.
     *
     * Review fix O8: simplified to the base64 branch only since
     * canvas.toDataURL always emits ``;base64,...`` (the only
     * caller of this helper).  The non-base64 + URL-decode path
     * was dead code in practice.
     */
    function _dataUrlToBlob(dataUrl) {
        try {
            const m = /^data:([^;,]+);base64,(.*)$/.exec(dataUrl);
            if (!m) return null;
            const mime = m[1] || "application/octet-stream";
            const bin = root.atob(m[2]);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) {
                bytes[i] = bin.charCodeAt(i);
            }
            return new root.Blob([bytes], { type: mime });
        } catch (_) {
            return null;
        }
    }

    /**
     * Phase 5b helper: drive the animation to a specific phase
     * point deterministically.  Used by captureFrames +
     * exportAnimation to step the animation frame-by-frame
     * without depending on the playback loop's wall-clock timing.
     *
     *   frameIdx     — 0-based output frame number
     *   totalFrames  — total output frames in the capture
     *   durationSec  — total capture duration in seconds
     *
     * For vibration: maps frameIdx to a cosine phase scaled by
     * speedHz so the capture covers ``durationSec`` worth of
     * oscillation.  For trajectory: maps frameIdx → input frame
     * index, wrapping if duration exceeds one loop.
     */
    function _driveAnimationFrame(state, frameIdx,
                                  totalFrames, durationSec) {
        const a = state.current.animation;
        if (!a) return;
        if (a.kind === "vibration") {
            if (!state._anim.vibrationBaseline) {
                state._anim.vibrationBaseline =
                    _captureBaselineCoords(state.viewer);
            }
            const baseline = state._anim.vibrationBaseline;
            const disp     = a.displacements;
            // elapsed seconds at this frame.
            const t = durationSec * (frameIdx / Math.max(1, totalFrames));
            const phase = 2 * Math.PI * a.speedHz * t;
            const factor = a.amplitude * Math.cos(phase);
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
        } else if (a.kind === "trajectory") {
            const n = a.frames.length;
            // Spread the capture across all input frames evenly so
            // duration scales the playback rate independently of fps.
            const inputIdx = Math.min(n - 1,
                Math.floor((frameIdx / Math.max(1, totalFrames)) * n));
            _showTrajectoryFrame(state, inputIdx);
        }
    }

    /**
     * Phase 5b helper: capture the embed's 3Dmol canvas as a PNG
     * Blob at the requested dimensions.  Resolves via
     * canvas.toBlob (async), which avoids the dataURL → Blob
     * conversion + base64 overhead that ``screenshot()`` pays for
     * single-frame capture.
     */
    function _canvasToPngBlob(state, width, height) {
        return new Promise((resolve, reject) => {
            try {
                // 3Dmol's pngURI generates a PNG at the requested
                // size by re-rendering at that resolution.  Use it
                // for size-override capture; otherwise grab the
                // live canvas directly via toBlob for speed.
                if ((width && width !== state.viewer.container.clientWidth)
                    || (height && height !== state.viewer.container.clientHeight)) {
                    const dataUrl = state.viewer.pngURI(width, height);
                    const blob = _dataUrlToBlob(dataUrl);
                    if (!blob) {
                        reject(_makeError("io_error",
                            "captureFrame: pngURI conversion failed"));
                        return;
                    }
                    resolve(blob);
                    return;
                }
                const canvas = state.viewer.container.querySelector("canvas");
                if (!canvas || typeof canvas.toBlob !== "function") {
                    // Fallback: dataURL path.
                    const dataUrl = state.viewer.pngURI();
                    const blob = _dataUrlToBlob(dataUrl);
                    if (!blob) {
                        reject(_makeError("io_error",
                            "captureFrame: no canvas + pngURI fallback failed"));
                        return;
                    }
                    resolve(blob);
                    return;
                }
                canvas.toBlob((b) => {
                    if (!b) {
                        reject(_makeError("io_error",
                            "captureFrame: canvas.toBlob returned null"));
                    } else {
                        resolve(b);
                    }
                }, "image/png");
            } catch (e) {
                reject(_makeError("io_error",
                    "captureFrame: " + (e && e.message), e));
            }
        });
    }

    /**
     * Phase 5b: lazy-load static/vendor/gif.min.js the first time
     * a GIF export is requested.  Cached on window so multiple
     * embeds share the load.  Resolves once ``window.GIF`` is
     * defined; rejects ``no_gif_encoder`` if the script tag fails.
     */
    function _loadGifEncoder() {
        if (root.GIF) return Promise.resolve(root.GIF);
        if (root._molbuilderGifLoading) {
            return root._molbuilderGifLoading;
        }
        // Phase 5d fix: store the WRAPPED promise (inner + catch
        // cleanup) in the cache, not the inner promise.  Otherwise a
        // second concurrent caller arriving at line "return cached"
        // gets the inner promise — its rejection bypasses the
        // catch-cleanup the first caller wired, and the cache is left
        // referencing a rejected promise.  Now both callers receive
        // the same wrapped promise and BOTH catch handlers run (the
        // cleanup is idempotent: setting null twice is fine).
        const inner = new Promise((resolve, reject) => {
            try {
                const s = root.document.createElement("script");
                s.src = "/static/vendor/gif.min.js";
                s.async = true;
                s.onload = () => {
                    if (root.GIF) resolve(root.GIF);
                    else reject(_makeError("no_gif_encoder",
                        "GIF encoder: script loaded but window.GIF "
                      + "is undefined"));
                };
                s.onerror = () => reject(_makeError("no_gif_encoder",
                    "GIF encoder: failed to load /static/vendor/"
                  + "gif.min.js (file may be absent)"));
                root.document.head.appendChild(s);
            } catch (e) {
                reject(_makeError("no_gif_encoder",
                    "GIF encoder: " + (e && e.message), e));
            }
        });
        const wrapped = inner.catch((err) => {
            root._molbuilderGifLoading = null;
            throw err;
        });
        root._molbuilderGifLoading = wrapped;
        return wrapped;
    }

    function _markerGlyph(kind) {
        if (kind === "lock") return "\u{1F512}";  // 🔒
        if (kind === "star") return "★";      // ★
        if (kind === "dot")  return "•";      // •
        return "?";
    }

    /**
     * Normalise a setAtomStyle selector argument into the same
     * shape as a normalised overlay entry (kind + value).  Returns
     * null on invalid input.
     */
    function _selectorToOverlayEntry(selector) {
        if (Array.isArray(selector)) {
            const vals = selector.filter(
                (v) => Number.isInteger(v) && v >= 0);
            return vals.length
                ? { selectorKind: "indices",  selectorValue: vals }
                : null;
        }
        if (selector && typeof selector === "object") {
            if (Array.isArray(selector.elements)) {
                const vals = selector.elements.filter(
                    (v) => typeof v === "string" && v.length > 0);
                return vals.length
                    ? { selectorKind: "elements", selectorValue: vals }
                    : null;
            }
            if (Array.isArray(selector.residues)) {
                const vals = selector.residues.filter(
                    (v) => Number.isInteger(v) && v >= 0);
                return vals.length
                    ? { selectorKind: "residues", selectorValue: vals }
                    : null;
            }
        }
        return null;
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
            // Optional per-frame arrow overlays per § 3.9.  If the
            // supplied array is shorter than frames.length, frames
            // beyond the tail render with no arrows (review fix O9
            // — was previously documented as "we cap" + dispatch
            // invalid_input, but neither cap nor dispatch actually
            // happens; the renderer just bounds-checks per-frame).
            const apf = Array.isArray(a.arrowsPerFrame)
                ? a.arrowsPerFrame : null;
            return {
                kind:           "trajectory",
                frames:         a.frames,
                arrowsPerFrame: apf,
                onFrame:        typeof a.onFrame === "function"
                                  ? a.onFrame : null,
                startFrame:     startFrame,
                // Phase 5h I-1: honor caller-supplied currentFrame so
                // applyState({animation: getAnimation()}) preserves the
                // playhead.  Without this clause the round-trip resets
                // to startFrame even mid-playback.  Out-of-range falls
                // back to startFrame.
                currentFrame:   (typeof a.currentFrame === "number"
                                 && a.currentFrame >= 0
                                 && a.currentFrame < nFrames)
                                  ? Math.floor(a.currentFrame)
                                  : startFrame,
                fps:            typeof a.fps === "number" && a.fps > 0
                                  ? a.fps : 10,
                paused:         a.paused !== false,  // default paused
                loop:           a.loop !== false,    // default loop
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
        // Single source of truth: keep config-state paused in sync with
        // runtime-state playing so getAnimation()/applyState round-trip
        // reflects current state, not mount-time intent.
        if (state.current.animation) state.current.animation.paused = true;
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
        if (state.current.animation) state.current.animation.paused = false;
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
        if (state.current.animation) state.current.animation.paused = false;
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
        // Per § 3.9: onFrame fires BEFORE each frame renders so the
        // host can mutate overlays / arrows reactively.  Setter
        // calls from inside onFrame are supported but add render
        // cost; prefer arrowsPerFrame for the common case.
        if (a.onFrame && state.handle) {
            try { a.onFrame(idx, state.handle); } catch (_) {}
        }
        _applyCoords(state.viewer, a.frames[idx]);
        // Per-frame arrows (arrowsPerFrame) overlay any
        // host-supplied arrows when they're available for this
        // frame.  Empty arrows[i] = "no arrows during frame i".
        if (a.arrowsPerFrame && idx < a.arrowsPerFrame.length) {
            const frameArrows = a.arrowsPerFrame[idx] || [];
            state.current.arrows = frameArrows.slice();
            _redrawArrows(state);
        }
        _postFramePositionRedraw(state);
        state.viewer.render();
        _refreshFrameStrip(state);
    }

    /* ------------------------------------------------------------ */
    /*  Frame-strip card chrome (trajectory animation only)         */
    /* ------------------------------------------------------------ */

    function _buildFrameStrip(state) {
        // Per § 6.3: frame strip auto-mounts whenever
        // animation.kind === "trajectory".
        if (state.frameStripEl) return;  // already built
        const a = state.current.animation;
        if (!a || a.kind !== "trajectory") return;

        const strip = document.createElement("div");
        strip.className = "mol-viewer-frame-strip";

        const prev = document.createElement("button");
        prev.type = "button";
        prev.className = "frame-prev";
        prev.setAttribute("data-action", "prev");
        prev.setAttribute("aria-label", "Previous frame");
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
        playPause.setAttribute("data-action", "play");
        playPause.setAttribute("aria-label", "Play / pause trajectory");
        playPause.textContent = "▶";
        playPause.title = "Play / pause";
        playPause.addEventListener("click", () => {
            if (state._anim.playing) _pauseImpl(state);
            else _playImpl(state);
        });

        const next = document.createElement("button");
        next.type = "button";
        next.className = "frame-next";
        next.setAttribute("data-action", "next");
        next.setAttribute("aria-label", "Next frame");
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
        slider.setAttribute("aria-label", "Trajectory frame");
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
        // Capture autoplay intent BEFORE _stopAnimationLoop, which now
        // syncs ``state.current.animation.paused = true`` (Phase 5g B-1
        // single-source-of-truth fix).  At mount, ``next`` aliases
        // ``state.current.animation`` (caller passes the same object),
        // so without this capture the autoplay check below would read
        // back ``true`` and skip _playImpl.
        const autoplay = !!(next && !next.paused);
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
            // Land on the desired frame.  ``currentFrame`` was already
            // resolved by _normaliseAnimation: a caller-supplied value
            // (round-trip) is honored, otherwise it falls back to
            // ``startFrame`` (fresh mount).  Passing startFrame here
            // unconditionally — as we did pre-Phase-5h — would clobber
            // the preserved playhead just before autoplay resumed.
            state.current.animation = next;
            _buildFrameStrip(state);
            _showTrajectoryFrame(state, next.currentFrame);
            if (autoplay) _playImpl(state);
        } else if (next && next.kind === "vibration") {
            state.current.animation = next;
            _removeFrameStrip(state);
            if (autoplay) _playImpl(state);
        } else {
            state.current.animation = null;
            _removeFrameStrip(state);
        }
        // Phase 6: Export menu's gif / webm buttons visibility
        // tracks animation presence.  Toggle on every _setAnimation
        // call so a mount-time animation, a partial-update that
        // changes kind, and a setAnimation(null) all stay
        // consistent.
        _syncKnobBarToAnimation(state);
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

        // Hard dependencies per § 2.5.1 — throw ViewerError with
        // code "missing_dependency" if any is absent.  These are
        // programming errors; throwing synchronously lets the page
        // loader fail fast.
        const viewerApi = (root.molbuilder || {}).viewer;
        if (typeof root.$3Dmol === "undefined") {
            throw _throwable(
                "missing_dependency",
                "viewer.embed: $3Dmol global must be loaded "
              + "(static/vendor/3dmol-min.js)"
            );
        }
        if (!viewerApi || typeof viewerApi.create !== "function") {
            throw _throwable(
                "missing_dependency",
                "viewer.embed: lib/mol-viewer.js must be loaded first"
            );
        }
        if (!root.molbuilder || !root.molbuilder.fmt) {
            throw _throwable(
                "missing_dependency",
                "viewer.embed: lib/mol-format.js must be loaded first "
              + "(expected window.molbuilder.fmt)"
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
            scaffold:      scaffold,   // retains knobsEl + knobs for late wiring
            cardOpts:      opts.card || {},

            // Retain caller's full opts so error dispatch + onReady
            // + onError can find their callbacks without per-callsite
            // plumbing.  Read-only after mount; do NOT mutate.
            userOpts:      opts,

            // Test injection per § 9.3.  When supplied, replaces
            // the global lookup (window.molbuilder.projects,
            // navigator.clipboard, MediaRecorder, GIF) for this
            // embed instance only.  Production passes nothing.
            testInjection: (opts.testInjection
                              && typeof opts.testInjection === "object")
                              ? opts.testInjection : {},

            // Rate-limit table for _dispatchError per § 5.4.
            // Allocated lazily on first error.
            _errorLastFires: null,

            current:       current,

            axesHandle:          null,
            cellShapes:          [],
            labelHandles:        [],
            arrowShapes:         [],
            arrowLabels:         [],
            pickShapes:          [],
            pickLabels:          [],
            pickedIndices:       [],
            pickWired:           false,

            // Per-atom overlay state (§ 3.12).  Style overrides are
            // baked into 3Dmol's setStyle and don't need handles;
            // halos and markers are removable shapes / labels.
            overlayHaloShapes:   [],
            overlayMarkerLabels: [],

            // Camera persistence per § 4.2.  hasFirstStructure flips
            // true after the first non-empty structure mounts; before
            // that, every setStructure calls zoomTo() so the first
            // sight of atoms is framed.  preserveCameraDefault is
            // the opt-level default (opts.preserveCamera; defaults
            // to true) — per-call overrides take precedence.
            hasFirstStructure:      !!(current.xyz || current.pdb),
            preserveCameraDefault:  opts.preserveCamera !== false,

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

            // Frame-strip DOM (built lazily whenever
            // animation.kind === "trajectory"; auto-mount per § 6.3).
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

        // 4a-2. ResizeObserver on the canvas host so the 3Dmol WebGL
        //       viewport tracks user-resizable containers (e.g. the
        //       Build tab's CSS resize handle on #viewer-wrap).
        //       3Dmol's canvas doesn't auto-track its parent box;
        //       this used to be wired per-tab via _viewer3dmol() +
        //       a bespoke ResizeObserver.  Now lives in the embed
        //       so every consumer benefits without per-tab code.
        if (typeof root.ResizeObserver === "function") {
            state._resizeObserver = new root.ResizeObserver(() => {
                if (state.disposed) return;
                try { viewer.resize(); viewer.render(); }
                catch (_) {}
            });
            try { state._resizeObserver.observe(scaffold.canvas); }
            catch (_) {}
        }

        // 4a-3. InteractionOpts mouse hooks per § 3.15.  Canonical
        //       drag-start / drag-end events let consumer tabs
        //       implement custom interaction policies (e.g. snap
        //       camera pivot to the molecule on first drag) without
        //       reaching into the raw canvas event stream.  All
        //       wiring lives in the embed so a UI redesign that
        //       changes the threshold / modifier filtering doesn't
        //       need a per-consumer edit.
        _wireInteractionHooks(state, scaffold.canvas, opts.interaction);

        // 4b. Build the handle + stash on state BEFORE applying the
        //     initial animation.  Trajectory autoplay (paused:false)
        //     fires onFrame(idx, handle) on its first tick; if the
        //     handle isn't on state yet, the callback receives
        //     undefined and the silent catch swallows the failure.
        //     Review fix P1 (#221).
        const handle = _buildHandle(state);
        state.handle = handle;

        // 4c. Wire the standard knob bar's click handlers + keyboard
        //     shortcuts now that the handle exists.  Phase 5 will
        //     extend the export / background actions; Phase 2 covers
        //     the static-rendering knobs (style / labels / axes / reset).
        //     state.scaffold is retained (not nulled here) because
        //     setKnobs needs scaffold.knobsEl to rebuild the bar in
        //     place; the knob config itself is read from
        //     state.current.knobs (the single source of truth).
        if (state.scaffold && state.scaffold.knobsEl) {
            _wireKnobBar(state, state.scaffold.knobsEl,
                                state.current.knobs);
        }

        // 4d. If the caller supplied opts.animation, set it up now
        //     (after the structure is loaded so baseline coord
        //     capture sees the right atoms; AND after state.handle
        //     is set so onFrame can fire correctly).  The setAnimation
        //     impl handles trajectory frame-strip mount + autoplay-
        //     unless-paused semantics.
        if (current.animation) {
            _setAnimationImpl(state, current.animation);
        }

        // 5. Fire onReady on the next microtask so the caller sees a
        //    fully-mounted handle (post-state-init + post-first-render).
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
            if (opts.xyz !== undefined && typeof opts.xyz !== "string") {
                _dispatchInvalidInput(state,
                    "setStructure: 'xyz' must be a string");
                return;
            }
            if (opts.pdb !== undefined && typeof opts.pdb !== "string") {
                _dispatchInvalidInput(state,
                    "setStructure: 'pdb' must be a string");
                return;
            }
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
            // Capture the pre-swap atom-element ordering so we can
            // honour § 3.8 / § 4.2.1 pick persistence: the pick state
            // (state.pickedIndices) survives setStructure IFF the new
            // structure has the same atom count AND element-by-element
            // ordering as the old one.  /modify's per-atom edit ops
            // (move / rotate / type-swap) preserve count + ordering, so
            // selection-survival across them is observable.
            const prevElements = _elements(state.viewer);
            const prevPicked   = state.pickedIndices.slice();
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
            // Review fix U7: new atom set → clickable flags reset on
            // 3Dmol's side; force _wirePick to re-register on the new
            // atoms.
            state.pickWired = false;
            _applyStyle(state.viewer, state.current.style);
            _redrawAllOverlays(state);
            _wirePick(state.viewer, state);
            // Declarative atom-indexed state persistence per § 3.8
            // / § 3.12 / § 4.2.1: pickedIndices + overlays (which
            // includes per-atom AtomStyleSpec) survive setStructure
            // IFF the new element sequence is identical to the
            // previous one (same count + same element-by-element
            // ordering).  Otherwise clear them — the atom-index
            // space changed, so an index that used to mean "the
            // carbon at position 5" now means something else.  Fire
            // onPick([]) on a clear so hosts mirroring picks into a
            // store see the transition.  The empty-prev case
            // (initial mount with no prior picks / overlays) skips
            // the onPick fire to avoid a spurious empty event at
            // boot.
            const nextElements = _elements(state.viewer);
            const sameAtoms =
                prevElements.length === nextElements.length
                && prevElements.length > 0
                && prevElements.every((e, i) => e === nextElements[i]);
            // Per § 4.2.1 + D1 escape hatch: ``preservePick`` /
            // ``preserveOverlays`` let hosts override the element-
            // comparison rule when they KNOW the swap is logically
            // same-structure (e.g. an atom-type edit where /modify
            // tracks the index mapping itself).  ``true`` forces
            // preservation; ``false`` forces clear; ``undefined``
            // (default) follows the element comparison above.
            const keepPick = typeof opts.preservePick === "boolean"
                ? opts.preservePick : sameAtoms;
            const keepOverlays = typeof opts.preserveOverlays === "boolean"
                ? opts.preserveOverlays : sameAtoms;
            if (keepPick) {
                state.pickedIndices = prevPicked;
            } else {
                state.pickedIndices = [];
                if (prevPicked.length > 0
                    && state.current.pick
                    && typeof state.current.pick.onPick === "function") {
                    try { state.current.pick.onPick([]); }
                    catch (_) {}
                }
            }
            if (!keepOverlays && state.current.overlays) {
                state.current.overlays = null;
                _redrawAllOverlays(state);
            }
            _redrawPickHalos(state);
            // Camera per § 4.2: the FIRST load (no prior structure)
            // always calls zoomTo() to frame the new structure;
            // subsequent loads preserve the user's camera unless
            // the caller explicitly opts out (per-call or opt-level
            // preserveCamera: false).
            const preserveCall = typeof opts.preserveCamera === "boolean"
                                   ? opts.preserveCamera
                                   : state.preserveCameraDefault;
            if (!state.hasFirstStructure || !preserveCall) {
                state.viewer.zoomTo();
            }
            state.hasFirstStructure = true;
            state.viewer.render();
            _refreshInfoLine(state);
        }

        function setStyle(s) {
            if (state.disposed) return;
            if (s && typeof s === "object") {
                if (s.rep !== undefined
                    && (typeof s.rep !== "string"
                        || !VALID_REPS.includes(s.rep))) {
                    _dispatchInvalidInput(state,
                        "setStyle: 'rep' must be one of " +
                        VALID_REPS.join(", ") +
                        "; got " + JSON.stringify(s.rep));
                }
                if (s.radiusScale !== undefined
                    && (typeof s.radiusScale !== "number"
                        || !Number.isFinite(s.radiusScale))) {
                    _dispatchInvalidInput(state,
                        "setStyle: 'radiusScale' must be a finite number");
                }
            }
            const next = _normaliseStyle(s);
            if (_equalNormalised(state.current.style, next)) return;
            state.current.style = next;
            _applyStyle(state.viewer, next);
            // Per § 3.12 layering: overlay style overrides must be
            // re-applied after the base style is reset (setStyle is
            // a replace, not an add).
            _redrawOverlayStyles(state);
            state.viewer.render();
            // Programmatic→UI sync per § 4.1.  setStyle covers both
            // the style picker (rep changed) and the background
            // swatch (setBackground routes through setStyle).
            _syncKnobBarToStyle(state);
            _syncKnobBarToBackground(state);
        }

        function setAxes(a) {
            if (state.disposed) return;
            if (a && typeof a === "object" && a !== true
                && a.mode !== undefined
                && !VALID_AXES_MODES.includes(a.mode)) {
                _dispatchInvalidInput(state,
                    "setAxes: 'mode' must be one of " +
                    VALID_AXES_MODES.join(", ") +
                    "; got " + JSON.stringify(a.mode));
            }
            // Phase 5d: explicit ``mode: "cell"`` requires a lattice
            // (§ 3.4 + § 5.3).  Without one, the renderer would
            // silently fall back to Cartesian — the user asked for
            // a cell-aligned triad and got x/y/z with no signal.
            // Halt so the caller knows; use ``mode: "auto"`` for the
            // graceful-fallback behavior.
            if (a && typeof a === "object" && a !== true
                && a.mode === "cell"
                && !state.current.lattice) {
                _dispatchInvalidInput(state,
                    "setAxes: mode 'cell' requires a lattice; call "
                  + "setStructure({xyz, lattice}) first OR use "
                  + "mode 'auto' for graceful fallback");
                return;
            }
            const next = _normaliseAxes(a);
            if (_equalNormalised(state.current.axes, next)) return;
            state.current.axes = next;
            _redrawAxes(state);
            state.viewer.render();
            _syncKnobBarToAxes(state);
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
            if (l && typeof l === "object" && l !== true) {
                if (l.atoms !== undefined && l.atoms !== "all"
                    && !Array.isArray(l.atoms)) {
                    _dispatchInvalidInput(state,
                        "setLabels: 'atoms' must be 'all' or a "
                      + "number[]");
                } else if (Array.isArray(l.atoms)) {
                    const bad = l.atoms.filter(
                        (v) => !Number.isInteger(v) || v < 0);
                    if (bad.length) {
                        _dispatchInvalidInput(state,
                            "setLabels: " + bad.length + " atom "
                          + "indices dropped (must be non-negative "
                          + "integers)");
                    }
                }
                if (l.format !== undefined
                    && !VALID_LABEL_FORMS.includes(l.format)) {
                    _dispatchInvalidInput(state,
                        "setLabels: 'format' must be one of " +
                        VALID_LABEL_FORMS.join(", ") +
                        "; got " + JSON.stringify(l.format));
                }
            }
            const next = _normaliseLabels(l);
            // Phase 6: remember the last non-null LabelOpts so the
            // View → Labels toggle can restore it when flipped back
            // on (otherwise toggle Off → On would always land on the
            // {atoms:"all", format:"index"} default, dropping the
            // host's chosen format).
            if (next) state.lastLabelsConfig = next;
            if (_equalNormalised(state.current.labels, next)) return;
            state.current.labels = next;
            _redrawLabels(state);
            state.viewer.render();
            _syncKnobBarToLabels(state);
        }

        function setArrows(arr) {
            if (state.disposed) return;
            if (arr !== undefined && arr !== null
                && !Array.isArray(arr)) {
                _dispatchInvalidInput(state,
                    "setArrows: argument must be an array of ArrowSpec "
                  + "or null");
                return;
            }
            if (Array.isArray(arr)) {
                let badCount = 0;
                for (const a of arr) {
                    if (!a || typeof a !== "object"
                        || !Array.isArray(a.start) || a.start.length !== 3
                        || !Array.isArray(a.end)   || a.end.length   !== 3) {
                        badCount++;
                    }
                }
                if (badCount) {
                    _dispatchInvalidInput(state,
                        "setArrows: " + badCount + " of " + arr.length
                      + " entries dropped (each must have start:[x,y,z] "
                      + "and end:[x,y,z])");
                }
            }
            const next = Array.isArray(arr) ? arr.slice() : [];
            // Idempotence: identity-stringify is fine at this scale.
            if (JSON.stringify(state.current.arrows) === JSON.stringify(next)) return;
            state.current.arrows = next;
            _redrawArrows(state);
            state.viewer.render();
        }

        function setPick(p) {
            if (state.disposed) return;
            if (p && typeof p === "object") {
                if (p.mode !== undefined
                    && !VALID_PICK_MODES.includes(p.mode)) {
                    _dispatchInvalidInput(state,
                        "setPick: 'mode' must be one of " +
                        VALID_PICK_MODES.join(", ") +
                        "; got " + JSON.stringify(p.mode));
                }
                if (p.label !== undefined && p.label !== false
                    && !VALID_LABEL_FORMS.includes(p.label)) {
                    _dispatchInvalidInput(state,
                        "setPick: 'label' must be false or one of " +
                        VALID_LABEL_FORMS.join(", ") +
                        "; got " + JSON.stringify(p.label));
                }
            }
            const next = _normalisePick(p);
            if (_equalNormalised(state.current.pick, next)) return;
            state.current.pick = next;
            state.pickedIndices = [];
            _redrawPickHalos(state);
            _wirePick(state.viewer, state);
        }

        function setBackground(color) {
            // Per § 3.2: a CSS color string applied to the canvas
            // backdrop ONLY.  Implemented as setStyle({background})
            // so the existing style pipeline owns the actual paint
            // and the idempotence diff works the same way.
            if (state.disposed) return;
            if (typeof color !== "string" || color.length === 0) {
                _dispatchInvalidInput(state,
                    "setBackground: color must be a non-empty CSS color string");
                return;
            }
            setStyle({
                rep:         state.current.style.rep,
                radiusScale: state.current.style.radiusScale,
                colorScheme: state.current.style.colorScheme,
                background:  color,
            });
        }

        function setKnobs(k) {
            // Per § 3.2: reconfigure visible knobs at runtime.
            // Rebuilds the knob bar DOM in place + re-wires it
            // against the same handle.  Idempotent against an
            // identical opts object via _equalNormalised.
            if (state.disposed) return;
            if (k && typeof k === "object" && k !== true) {
                if (k.backgroundPresets !== undefined
                    && !Array.isArray(k.backgroundPresets)) {
                    _dispatchInvalidInput(state,
                        "setKnobs: 'backgroundPresets' must be a string[] "
                      + "of CSS colors");
                }
            }
            const next = _normaliseKnobs(k);
            if (_equalNormalised(state.current.knobs, next)) return;
            state.current.knobs = next;
            if (!state.cardEl) return;
            // Remove old bar (if any).
            const old = state.cardEl.querySelector(":scope > .mol-viewer-knobs");
            if (old) old.remove();
            state.scaffold.knobsEl = null;
            if (next) {
                const bar = _buildKnobBarDOM(next);
                state.scaffold.knobsEl = bar;
                // Insert before the canvas (and frame strip if any)
                // to keep the §6.1 anatomy order.
                state.cardEl.insertBefore(bar, state.canvasEl);
                _wireKnobBar(state, bar, next);
            }
        }

        function applyState(spec) {
            // Per § 3.2 D4: ordered batch runner.  Each field is
            // optional; the embed dispatches the matching setX call
            // in a defined order so atom-keyed state (overlays,
            // labels-as-array, picks, animation, camera) lands AFTER
            // setStructure has reloaded the model.  Useful for host-
            // side persistence round-trips:
            //   const snap = {
            //     structure: { xyz, lattice },
            //     style: h.getStyle(), axes: h.getAxes(),
            //     overlays: h.getOverlays(),
            //     camera: h.getCamera(),
            //   };
            //   // ... later ...
            //   h.applyState(snap);
            // Not a "single render frame" guarantee — 3Dmol's render
            // is cheap and multiple intermediate renders are fine.
            // For atomic visual swaps a future ``defer: true`` flag
            // could defer rendering; not implemented today.
            if (state.disposed) return;
            if (!spec || typeof spec !== "object") {
                _dispatchInvalidInput(state,
                    "applyState: argument must be an object");
                return;
            }
            // Order matters: structure → base style → background →
            // world overlays → atom-keyed overlays → pick → animation
            // → camera → knobs.  Each step is independent of the
            // ones below it but depends on the ones above.
            if (spec.structure !== undefined && spec.structure !== null) {
                setStructure(spec.structure);
            }
            if (spec.style !== undefined) {
                setStyle(spec.style);
            }
            if (spec.background !== undefined) {
                setBackground(spec.background);
            }
            if (spec.axes !== undefined) {
                setAxes(spec.axes);
            }
            if (spec.cell !== undefined) {
                setCell(spec.cell);
            }
            if (spec.labels !== undefined) {
                setLabels(spec.labels);
            }
            if (spec.arrows !== undefined) {
                setArrows(spec.arrows);
            }
            if (spec.overlays !== undefined) {
                setOverlays(spec.overlays);
            }
            if (spec.pick !== undefined) {
                setPick(spec.pick);
            }
            if (spec.pickedIndices !== undefined) {
                setPickedIndices(spec.pickedIndices);
            }
            if (spec.animation !== undefined) {
                setAnimation(spec.animation);
            }
            if (spec.camera !== undefined && spec.camera !== null) {
                setCamera(spec.camera);
            }
            if (spec.knobs !== undefined) {
                setKnobs(spec.knobs);
            }
        }

        function setOverlays(o) {
            if (state.disposed) return;
            const next = _normaliseOverlays(o);
            // Review fix U9: dispatch invalid_input when entries
            // were silently dropped (bad selector / no treatment /
            // multiple selectors) so the host's onError sees it.
            if (o && Array.isArray(o.atoms)) {
                const inN  = o.atoms.length;
                const outN = next ? next.atoms.length : 0;
                if (outN < inN) {
                    _dispatchInvalidInput(state,
                        "setOverlays: " + (inN - outN) + " of " + inN
                      + " atom entries dropped (bad/missing/multiple "
                      + "selectors, or no style/halo/marker).");
                }
            }
            if (_equalNormalised(state.current.overlays, next)) return;
            state.current.overlays = next;
            // Style overrides are baked into setStyle, so a full
            // redraw is needed (base style → overlay styles → halos
            // → markers → pick halos all re-render in order).
            _redrawOverlayStyles(state);
            _redrawOverlayHalosAndMarkers(state);
            _redrawPickHalos(state);   // pick draws above overlay halos
            state.viewer.render();
        }

        function setAtomStyle(selector, style) {
            // Sugar for the common "give these atoms this style"
            // call.  Upserts a single overlays.atoms[] entry keyed
            // on the selector's normalised form per § 3.12.
            if (state.disposed) return;
            const entry = _selectorToOverlayEntry(selector);
            if (!entry) {
                _dispatchInvalidInput(state,
                    "setAtomStyle: selector must be number[] OR "
                  + "{elements: string[]} OR {residues: number[]}");
                return;
            }
            // style: null removes the entry; otherwise apply style.
            const current = state.current.overlays
                          ? state.current.overlays.atoms.slice() : [];
            const keyFor = (e) => e.selectorKind + ":"
                                + JSON.stringify(e.selectorValue);
            const want = keyFor(entry);
            const filtered = current.filter((e) => keyFor(e) !== want);
            if (style !== null && style !== undefined) {
                const normStyle = _normaliseOverlayStyle(style);
                if (!normStyle) {
                    _dispatchInvalidInput(state,
                        "setAtomStyle: style must include at least one of "
                      + "{rep, radiusScale, color, opacity}");
                    return;
                }
                filtered.push({
                    selectorKind:  entry.selectorKind,
                    selectorValue: entry.selectorValue,
                    style:  normStyle,
                    halo:   null,
                    marker: null,
                });
            }
            setOverlays({ atoms: filtered });
        }

        function appendFrames(frames) {
            // Trajectory live-poll path per § 3.2 + § 4.3:
            //   - vibration or no animation: silent no-op
            //   - trajectory: extend frames, preserve currentFrame
            //   - atom-count mismatch: invalid_input via onError
            if (state.disposed) return;
            const a = state.current.animation;
            if (!a || a.kind !== "trajectory") return;
            if (!Array.isArray(frames) || frames.length === 0) return;
            // Validate atom-count against the existing frames.
            const expectedN = (a.frames[0] && a.frames[0].length) || 0;
            for (const f of frames) {
                if (!Array.isArray(f) || f.length !== expectedN) {
                    _dispatchInvalidInput(state,
                        "appendFrames: atom count mismatch — existing "
                      + "trajectory has " + expectedN + " atoms per frame, "
                      + "appended frame has "
                      + (Array.isArray(f) ? f.length : "?"));
                    return;
                }
            }
            // Mutate in place; currentFrame is index-based so the
            // playhead naturally stays put.  The frame strip auto-
            // refreshes its counter / slider max via _refreshFrameStrip.
            for (const f of frames) a.frames.push(f);
            _refreshFrameStrip(state);
        }

        function getCamera() {
            // Capture position / look-at / zoom / rotation as an
            // opaque blob per § 3.13.  The discriminator + version
            // let setCamera() no-op on mismatch so a future
            // renderer swap doesn't crash consumers persisting old
            // states.
            if (state.disposed) {
                return { _viewer: "3dmol", _version: 1, data: null };
            }
            let data = null;
            try {
                if (typeof state.viewer.getView === "function") {
                    data = state.viewer.getView();
                }
            } catch (_) {}
            return { _viewer: "3dmol", _version: 1, data: data };
        }

        function setCamera(s) {
            if (state.disposed) return;
            if (!s || typeof s !== "object") {
                _dispatchInvalidInput(state,
                    "setCamera: argument must be a CameraState object");
                return;
            }
            // Version mismatch is silent (forward-compat) per § 3.13.
            if (s._viewer !== "3dmol" || s._version !== 1) return;
            if (s.data === null || s.data === undefined) return;
            try {
                if (typeof state.viewer.setView === "function") {
                    state.viewer.setView(s.data);
                    state.viewer.render();
                }
            } catch (_) {}
        }

        function getAtomCount() {
            if (state.disposed) return 0;
            return _atomCount(state.viewer);
        }
        function getElements() {
            if (state.disposed) return [];
            return _elements(state.viewer);
        }
        // Read accessors for declarative state.  Each returns a
        // defensive deep-clone of the current section so callers
        // can persist + restore via the matching setX without
        // mirroring their own bookkeeping.  Returns null when the
        // section is disabled (matches the setX null-clear shape);
        // setX(getX()) is the round-trip.
        function _clone(v) {
            // JSON round-trip is fine: the embed's normalised state
            // contains only plain objects / arrays / scalars (no
            // Date / Function / undefined keys / cycles).
            return v === null || v === undefined
                 ? v : JSON.parse(JSON.stringify(v));
        }
        function getStyle() {
            return state.disposed ? null : _clone(state.current.style);
        }
        function getAxes() {
            return state.disposed ? null : _clone(state.current.axes);
        }
        function getCell() {
            return state.disposed ? null : _clone(state.current.cell);
        }
        function getLabels() {
            return state.disposed ? null : _clone(state.current.labels);
        }
        function getOverlays() {
            return state.disposed ? null : _clone(state.current.overlays);
        }
        function getPick() {
            // PickOpts.onPick is a function which JSON-clone drops;
            // pick state is otherwise plain.  Surface the cloneable
            // fields + the live onPick reference so the getter can be
            // round-tripped via setPick.
            if (state.disposed) return null;
            const p = state.current.pick;
            if (!p) return null;
            const out = _clone(p);
            if (typeof p.onPick === "function") out.onPick = p.onPick;
            return out;
        }
        function getKnobs() {
            return state.disposed ? null : _clone(state.current.knobs);
        }
        function getArrows() {
            return state.disposed ? [] : _clone(state.current.arrows) || [];
        }
        function getAnimation() {
            // AnimationOpts.onFrame is a function; preserve like getPick.
            if (state.disposed) return null;
            const a = state.current.animation;
            if (!a) return null;
            const out = _clone(a);
            if (typeof a.onFrame === "function") out.onFrame = a.onFrame;
            return out;
        }
        function getBackground() {
            // Phase 5d: getBackground completes the applyState round-
            // trip for the background field (which applyState already
            // accepts, but had no documented producer).
            if (state.disposed) return null;
            return (state.current.style
                    && state.current.style.background) || null;
        }
        function getLattice() {
            // Phase 5d: getLattice closes the gap in the applyState
            // round-trip for cell-bearing structures.  Without this,
            // ``applyState({structure: {xyz: getStructureText()}})``
            // silently loses the cell + makes setAxes mode "cell"
            // unreachable post-round-trip.
            if (state.disposed) return null;
            return _clone(state.current.lattice);
        }

        function getPickedIndices() {
            if (state.disposed) return [];
            return state.pickedIndices.slice();
        }

        function setPickedIndices(indices) {
            // Push the pick state from an external source (host
            // atom list, panel, undo).  Re-renders halos / labels
            // according to the active pick.mode + pick.halo +
            // pick.label.  Does NOT fire onPick — that callback is
            // reserved for click-driven changes so hosts that
            // mirror picks into a store don't see a feedback loop.
            // Clamps to the mode's max (single: 1; pair: 2; multi:
            // unbounded).  Pass null or [] to clear.
            if (state.disposed) return;
            // Validate the argument type FIRST so a bad call fires
            // invalid_input even when no pick mode is configured
            // (per § 5.3 "halt" semantics — type errors are caller
            // bugs regardless of state).
            let next;
            if (indices === null || indices === undefined) {
                next = [];
            } else if (Array.isArray(indices)) {
                next = indices.filter(function (v) {
                    return Number.isInteger(v) && v >= 0;
                });
            } else {
                _dispatchInvalidInput(state,
                    "setPickedIndices: argument must be an array of "
                  + "non-negative integers or null");
                return;
            }
            if (!state.current.pick) return;
            const mode = state.current.pick.mode;
            if (mode === "single" && next.length > 1) {
                next = next.slice(-1);
            } else if (mode === "pair" && next.length > 2) {
                next = next.slice(-2);
            }
            state.pickedIndices = next;
            _redrawPickHalos(state);
        }

        function getStructureText(format) {
            // Returns the current structure as text in the
            // requested format per § 3.2.  Omit ``format`` →
            // returns whatever was supplied (pdb wins if both).
            // Phase 5a does NOT convert between formats; if the
            // caller asks for xyz but only pdb was supplied,
            // returns "".
            if (state.disposed) return "";
            const c = state.current;
            if (format === "xyz") return c.xyz || "";
            if (format === "pdb") return c.pdb || "";
            return c.pdb || c.xyz || "";
        }

        // -------------------------------------------------------- //
        //  Export helpers — § 3.11 + § 5                            //
        // -------------------------------------------------------- //

        function _filename(stem, ext) {
            const e = (state.userOpts.export
                       && state.userOpts.export.defaultName)
                       || stem
                       || "structure";
            // Strip illegal filesystem chars; keep ASCII-ish names.
            const safe = String(e).replace(/[^\w.\-]+/g, "_");
            return safe + "." + ext;
        }

        function _projectsApi() {
            return (state.testInjection && state.testInjection.projectsApi)
                || (root.molbuilder && root.molbuilder.projects)
                || null;
        }

        function _clipboardApi() {
            return (state.testInjection && state.testInjection.clipboardApi)
                || (root.navigator && root.navigator.clipboard)
                || null;
        }

        function _writeToProject(filename, data) {
            const proj = _projectsApi();
            if (!proj || typeof proj.writeFile !== "function") {
                return Promise.reject(_makeError(
                    "no_project",
                    "save-to-project: window.molbuilder.projects.writeFile "
                  + "not available"));
            }
            const currentDir = typeof proj.currentDir === "function"
                                 ? proj.currentDir() : proj.currentDir;
            if (!currentDir) {
                return Promise.reject(_makeError(
                    "no_project",
                    "save-to-project: no active project directory"));
            }
            const path = currentDir.replace(/\/$/, "") + "/" + filename;
            return Promise.resolve(proj.writeFile(path, data))
                .then((env) => {
                    if (env && env.ok === false) {
                        throw _makeError(
                            "io_error",
                            (env.error || "writeFile failed"), env);
                    }
                    // Review fix D12: contract says ``filename`` is
                    // the leaf name; expose the full path separately
                    // so consumers can still log / display it.
                    return {
                        filename: filename,
                        path:     path,
                        bytes:    typeof data === "string"
                                    ? data.length
                                    : (data && data.size) || 0,
                    };
                });
        }

        function _triggerDownload(filename, blob) {
            try {
                const url = root.URL.createObjectURL(blob);
                const a = root.document.createElement("a");
                a.href = url;
                a.download = filename;
                a.style.display = "none";
                root.document.body.appendChild(a);
                a.click();
                a.remove();
                // Revoke the object URL on the next microtask
                // (some browsers need a tick before the download
                // starts).
                Promise.resolve().then(() =>
                    root.URL.revokeObjectURL(url));
                return Promise.resolve({
                    filename: filename, bytes: blob.size,
                });
            } catch (e) {
                return Promise.reject(_makeError(
                    "io_error",
                    "download trigger failed: " + (e && e.message), e));
            }
        }

        function _aborted(signal) {
            if (!signal) return null;
            if (signal.aborted) {
                return _makeError("aborted",
                                  "operation aborted before start");
            }
            return null;
        }

        function _fireOnExport(info) {
            const cb = state.userOpts.export
                        && typeof state.userOpts.export.onExport === "function"
                        ? state.userOpts.export.onExport : null;
            if (cb) {
                try { cb(info); } catch (_) {}
            }
        }

        function exportData(opts) {
            opts = opts || {};
            if (state.disposed) {
                return Promise.reject(_makeError("disposed",
                    "exportData: viewer disposed"));
            }
            const aborted = _aborted(opts.signal);
            if (aborted) return Promise.reject(aborted);

            // Decide format.  Default: whatever the embed has.
            let format = opts.format;
            const haveXyz = !!state.current.xyz;
            const havePdb = !!state.current.pdb;
            if (!format) format = havePdb ? "pdb" : (haveXyz ? "xyz" : null);
            if (!format) {
                return Promise.reject(_makeError("no_structure",
                    "exportData: no structure loaded"));
            }
            const text = getStructureText(format);
            if (!text) {
                // Review fix D10: format mismatch is invalid_input
                // (the caller asked for a format the embed wasn't
                // given), not no_structure (which means "no structure
                // loaded at all").
                return Promise.reject(_makeError("invalid_input",
                    "exportData: requested format '" + format
                  + "' not available (supplied format was "
                  + (havePdb ? "pdb" : "xyz") + ")"));
            }
            const fname = _filename(null, format);

            let p;
            if (opts.target === "project") {
                p = _writeToProject(opts.filename || fname, text);
            } else if (opts.target === "download") {
                const blob = new root.Blob([text],
                                           { type: "text/plain" });
                p = _triggerDownload(opts.filename || fname, blob);
            } else if (opts.target === "clipboard") {
                const cb = _clipboardApi();
                if (!cb || typeof cb.writeText !== "function") {
                    p = Promise.reject(_makeError("no_clipboard",
                        "exportData: clipboard API unavailable "
                      + "(HTTPS or localhost required)"));
                } else {
                    p = cb.writeText(text)
                        .then(() => ({
                            filename: opts.filename || fname,
                            bytes: text.length,
                        }))
                        .catch((e) => {
                            throw _makeError("io_error",
                                "clipboard write failed: "
                              + (e && e.message), e);
                        });
                }
            } else {
                return Promise.reject(_makeError("invalid_input",
                    "exportData: target must be 'project', "
                  + "'download', or 'clipboard'"));
            }

            return p.then((r) => {
                _fireOnExport({
                    kind:     "structure",
                    target:   opts.target,
                    format:   format,
                    filename: r.filename,
                    bytes:    r.bytes,
                });
                return r;
            });
        }

        function _defaultFps(a) {
            if (a.kind === "trajectory" && a.fps > 0) return a.fps;
            return 30;
        }
        function _defaultDuration(a) {
            if (a.kind === "trajectory") {
                // Full one-loop = frames.length / fps.
                return a.frames.length / Math.max(1, _defaultFps(a));
            }
            // Vibration: one cosine cycle = 1 / speedHz.
            return 1 / Math.max(0.01, a.speedHz);
        }

        function captureFrames(opts) {
            // Per § 3.2 + 2026-06-03 decision log: drive the
            // animation deterministically and capture each frame as
            // a PNG blob.  Resolves with a Blob[] in render order.
            opts = opts || {};
            if (state.disposed) {
                return Promise.reject(_makeError("disposed",
                    "captureFrames: viewer disposed"));
            }
            const a = state.current.animation;
            if (!a) {
                return Promise.reject(_makeError("static_structure",
                    "captureFrames: opts.animation is null"));
            }
            if (_atomCount(state.viewer) === 0) {
                return Promise.reject(_makeError("no_structure",
                    "captureFrames: no structure loaded"));
            }
            const abortNow = _aborted(opts.signal);
            if (abortNow) return Promise.reject(abortNow);

            const fps = typeof opts.fps === "number"
                            && opts.fps > 0 ? opts.fps : _defaultFps(a);
            const duration = typeof opts.duration === "number"
                            && opts.duration > 0
                            ? opts.duration : _defaultDuration(a);
            const total = Math.max(1, Math.round(fps * duration));
            const width  = typeof opts.width  === "number" ? opts.width  : 0;
            const height = typeof opts.height === "number" ? opts.height : 0;

            // Pause the live playback loop so it doesn't fight the
            // deterministic frame drive.  Restored in the finally
            // block at the end.
            const wasPlaying = state._anim.playing;
            if (wasPlaying) _stopAnimationLoop(state);

            // Capture the original animation phase so we can
            // restore it post-capture (so a host's "Capture →
            // continue editing" flow doesn't lose its position).
            const savedFrame = a.kind === "trajectory"
                                 ? a.currentFrame : null;

            const blobs = [];
            let i = 0;

            function step() {
                if (state.disposed) {
                    return Promise.reject(_makeError("disposed",
                        "captureFrames: viewer disposed mid-capture"));
                }
                const ab = _aborted(opts.signal);
                if (ab) return Promise.reject(ab);
                if (i >= total) return Promise.resolve();
                _driveAnimationFrame(state, i, total, duration);
                state.viewer.render();
                return _canvasToPngBlob(state, width, height)
                    .then((blob) => {
                        blobs.push(blob);
                        if (typeof opts.onProgress === "function") {
                            try {
                                opts.onProgress((i + 1) / total,
                                    "frame " + (i + 1) + "/" + total);
                            } catch (_) {}
                        }
                        i++;
                        return step();
                    });
            }

            return step()
                .then(() => blobs)
                .finally(() => {
                    // Restore the pre-capture animation phase so the
                    // viewer doesn't jump on the user.  Phase 5d C3:
                    // also restore the vibration baseline coords —
                    // captureFrames left them at the last-driven
                    // cosine phase, so a one-shot capture from a
                    // paused vibration would freeze the model mid-
                    // displacement.
                    if (savedFrame !== null && state.current.animation
                        && state.current.animation.kind === "trajectory") {
                        try {
                            _showTrajectoryFrame(state, savedFrame);
                        } catch (_) {}
                    } else if (state.current.animation
                               && state.current.animation.kind === "vibration"
                               && state._anim.vibrationBaseline
                               && !state.disposed) {
                        try {
                            _applyCoords(state.viewer,
                                state._anim.vibrationBaseline);
                            _postFramePositionRedraw(state);
                            state.viewer.render();
                        } catch (_) {}
                    }
                    if (wasPlaying && !state.disposed) {
                        try { _playImpl(state); } catch (_) {}
                    }
                });
        }

        function _mediaRecorderApi() {
            // testInjection.mediaRecorder === null forces "absent"
            // (used by tests on browsers that ship MediaRecorder
            // natively).  undefined falls through to the global.
            if (state.testInjection
                && "mediaRecorder" in state.testInjection) {
                return state.testInjection.mediaRecorder;
            }
            return typeof root.MediaRecorder !== "undefined"
                ? root.MediaRecorder : null;
        }

        function _gifEncoderCtor() {
            // testInjection.gifEncoder === null forces "absent"
            // (used by tests on environments where the vendor file
            // IS available, so the lazy-load path can't be used to
            // surface no_gif_encoder).  undefined falls through to
            // the lazy-load (which checks window.GIF then attempts
            // /static/vendor/gif.min.js).  A function override
            // skips the lazy-load entirely.
            if (state.testInjection
                && "gifEncoder" in state.testInjection) {
                return Promise.resolve(state.testInjection.gifEncoder);
            }
            return _loadGifEncoder();
        }

        function _exportAnimationWebm(opts, a, fps, duration, total) {
            const MR = _mediaRecorderApi();
            if (!MR) {
                return Promise.reject(_makeError("no_media_recorder",
                    "exportAnimation: MediaRecorder unavailable"));
            }
            const canvas = state.viewer.container.querySelector("canvas");
            if (!canvas || typeof canvas.captureStream !== "function") {
                return Promise.reject(_makeError("no_media_recorder",
                    "exportAnimation: canvas.captureStream unavailable"));
            }
            let stream;
            try { stream = canvas.captureStream(fps); }
            catch (e) {
                return Promise.reject(_makeError("io_error",
                    "exportAnimation: canvas.captureStream threw: "
                  + (e && e.message), e));
            }
            const chunks = [];
            const mime = (typeof MR.isTypeSupported === "function"
                          && MR.isTypeSupported("video/webm;codecs=vp9"))
                ? "video/webm;codecs=vp9"
                : "video/webm";
            let recorder;
            try { recorder = new MR(stream, { mimeType: mime }); }
            catch (e) {
                return Promise.reject(_makeError("io_error",
                    "exportAnimation: MediaRecorder ctor failed: "
                  + (e && e.message), e));
            }
            recorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) chunks.push(e.data);
            };
            const stopped = new Promise((resolve) => {
                recorder.onstop = resolve;
            });
            recorder.start();

            // Drive frames at fps cadence.  setTimeout cadence
            // doesn't have to be exact — MediaRecorder samples the
            // captureStream at its own rate; we just need to keep
            // pushing new frames so the recorder has content.
            const frameMs = 1000 / fps;
            let i = 0;
            return new Promise((resolve, reject) => {
                // Phase 5d C4: on abort / disposed paths we still
                // await ``stopped`` before rejecting so the
                // recorder finishes flushing.  The chunks are
                // discarded by the rejected branch but the
                // recorder is no longer in a half-stopped state.
                function tick() {
                    if (state.disposed) {
                        try { recorder.stop(); } catch (_) {}
                        return stopped.then(() => reject(_makeError(
                            "disposed",
                            "exportAnimation: disposed mid-record")));
                    }
                    const ab = _aborted(opts.signal);
                    if (ab) {
                        try { recorder.stop(); } catch (_) {}
                        return stopped.then(() => reject(ab));
                    }
                    if (i >= total) {
                        try { recorder.stop(); } catch (_) {}
                        return stopped.then(resolve);
                    }
                    _driveAnimationFrame(state, i, total, duration);
                    state.viewer.render();
                    if (typeof opts.onProgress === "function") {
                        try {
                            opts.onProgress((i + 1) / total,
                                "encoding " + (i + 1) + "/" + total);
                        } catch (_) {}
                    }
                    i++;
                    setTimeout(tick, frameMs);
                }
                tick();
            }).then(() => new Blob(chunks, { type: "video/webm" }));
        }

        function _exportAnimationGif(opts, a, fps, duration, total) {
            return _gifEncoderCtor().then((GIF) => {
                if (!GIF) {
                    throw _makeError("no_gif_encoder",
                        "exportAnimation: GIF encoder unavailable "
                      + "(testInjection.gifEncoder forced absent)");
                }
                const canvas = state.viewer.container.querySelector("canvas");
                const w = canvas.width || canvas.clientWidth;
                const h = canvas.height || canvas.clientHeight;
                const gif = new GIF({
                    workers: 2, quality: 10,
                    width:   w, height: h,
                    workerScript: "/static/vendor/gif.worker.min.js",
                });
                return new Promise((resolve, reject) => {
                    let cancelled = false;
                    let abortListener = null;
                    const ab = _aborted(opts.signal);
                    if (ab) return reject(ab);
                    // Phase 5d C5: track the abort listener so it
                    // can be removed on completion / cancellation.
                    // Without this, a long-lived AbortController
                    // reused across many exports accumulates
                    // listeners and leaks closures.
                    function _detachAbort() {
                        if (opts.signal && abortListener) {
                            opts.signal.removeEventListener(
                                "abort", abortListener);
                            abortListener = null;
                        }
                    }
                    if (opts.signal) {
                        abortListener = () => {
                            cancelled = true;
                            try { gif.abort(); } catch (_) {}
                            _detachAbort();
                            reject(_makeError("aborted",
                                "exportAnimation: aborted"));
                        };
                        opts.signal.addEventListener(
                            "abort", abortListener);
                    }
                    let i = 0;
                    const delayMs = 1000 / fps;
                    function step() {
                        if (cancelled) return;
                        if (state.disposed) {
                            _detachAbort();
                            return reject(_makeError("disposed",
                                "exportAnimation: disposed mid-encode"));
                        }
                        if (i >= total) {
                            gif.on("finished", (blob) => {
                                _detachAbort();
                                resolve(blob);
                            });
                            gif.render();
                            return;
                        }
                        _driveAnimationFrame(state, i, total, duration);
                        state.viewer.render();
                        try {
                            gif.addFrame(canvas, {
                                copy: true, delay: delayMs,
                            });
                        } catch (e) {
                            _detachAbort();
                            return reject(_makeError("io_error",
                                "exportAnimation: gif.addFrame threw: "
                              + (e && e.message), e));
                        }
                        if (typeof opts.onProgress === "function") {
                            try {
                                opts.onProgress(
                                    0.5 * (i + 1) / total,
                                    "frame " + (i + 1) + "/" + total);
                            } catch (_) {}
                        }
                        i++;
                        // Yield to the browser so the canvas
                        // re-renders between addFrame calls.
                        setTimeout(step, 0);
                    }
                    step();
                });
            });
        }

        function exportAnimation(opts) {
            // Per § 3.2 + § 5.3: encode the animation to WebM (via
            // canvas.captureStream + MediaRecorder) or GIF (via
            // lazy-loaded gif.js).  Saves to project or triggers a
            // download per opts.target.
            opts = opts || {};
            if (state.disposed) {
                return Promise.reject(_makeError("disposed",
                    "exportAnimation: viewer disposed"));
            }
            const a = state.current.animation;
            if (!a) {
                return Promise.reject(_makeError("static_structure",
                    "exportAnimation: opts.animation is null"));
            }
            if (_atomCount(state.viewer) === 0) {
                return Promise.reject(_makeError("no_structure",
                    "exportAnimation: no structure loaded"));
            }
            if (opts.format !== "webm" && opts.format !== "gif") {
                return Promise.reject(_makeError("invalid_input",
                    "exportAnimation: format must be 'webm' or 'gif'"));
            }
            if (opts.target !== "project" && opts.target !== "download") {
                return Promise.reject(_makeError("invalid_input",
                    "exportAnimation: target must be 'project' or 'download'"));
            }
            const abortNow = _aborted(opts.signal);
            if (abortNow) return Promise.reject(abortNow);

            const fps = typeof opts.fps === "number"
                            && opts.fps > 0 ? opts.fps : _defaultFps(a);
            const duration = typeof opts.duration === "number"
                            && opts.duration > 0
                            ? opts.duration : _defaultDuration(a);
            const total = Math.max(1, Math.round(fps * duration));

            // Pause live playback so the encoder owns frame timing.
            const wasPlaying = state._anim.playing;
            if (wasPlaying) _stopAnimationLoop(state);
            const savedFrame = a.kind === "trajectory"
                                 ? a.currentFrame : null;

            const ext = opts.format === "webm" ? "webm" : "gif";
            const filename = opts.filename || _filename(null, ext);

            const encoded = opts.format === "webm"
                ? _exportAnimationWebm(opts, a, fps, duration, total)
                : _exportAnimationGif(opts, a, fps, duration, total);

            return encoded
                .then((blob) => {
                    if (opts.target === "project") {
                        return _writeToProject(filename, blob);
                    }
                    return _triggerDownload(filename, blob);
                })
                .then((r) => {
                    _fireOnExport({
                        kind:     "animation",
                        target:   opts.target,
                        format:   opts.format,
                        filename: r.filename,
                        bytes:    r.bytes,
                    });
                    return r;
                })
                .finally(() => {
                    // Phase 5d C3: restore vibration baseline as
                    // well as trajectory savedFrame (same rule as
                    // captureFrames above).
                    if (savedFrame !== null && state.current.animation
                        && state.current.animation.kind === "trajectory") {
                        try {
                            _showTrajectoryFrame(state, savedFrame);
                        } catch (_) {}
                    } else if (state.current.animation
                               && state.current.animation.kind === "vibration"
                               && state._anim.vibrationBaseline
                               && !state.disposed) {
                        try {
                            _applyCoords(state.viewer,
                                state._anim.vibrationBaseline);
                            _postFramePositionRedraw(state);
                            state.viewer.render();
                        } catch (_) {}
                    }
                    if (wasPlaying && !state.disposed) {
                        try { _playImpl(state); } catch (_) {}
                    }
                });
        }

        function screenshot(opts) {
            opts = opts || {};
            if (state.disposed) {
                return Promise.reject(_makeError("disposed",
                    "screenshot: viewer disposed"));
            }
            const aborted = _aborted(opts.signal);
            if (aborted) return Promise.reject(aborted);

            // Review fix P7: empty canvas isn't meaningful to capture.
            // Reject early with no_structure to match § 5.3.
            if (_atomCount(state.viewer) === 0) {
                return Promise.reject(_makeError("no_structure",
                    "screenshot: no structure loaded"));
            }

            // 3Dmol.pngURI(width, height) supports super-resolution
            // capture when width/height exceed the on-screen canvas
            // (see § 3.2 screenshot() docstring).
            let dataUrl;
            try {
                const v = state.viewer;
                if (opts.width || opts.height) {
                    dataUrl = v.pngURI(opts.width || undefined,
                                       opts.height || undefined);
                } else {
                    dataUrl = v.pngURI();
                }
            } catch (e) {
                return Promise.reject(_makeError("io_error",
                    "screenshot: pngURI failed: "
                  + (e && e.message), e));
            }
            if (!dataUrl) {
                return Promise.reject(_makeError("no_structure",
                    "screenshot: pngURI returned empty"));
            }
            // Convert data URL to a Blob synchronously.
            const blob = _dataUrlToBlob(dataUrl);
            if (!blob) {
                return Promise.reject(_makeError("io_error",
                    "screenshot: dataURL → blob conversion failed"));
            }
            const fname = _filename(null, "png");

            let chain;
            if (opts.target === "project") {
                chain = _writeToProject(opts.filename || fname, blob);
            } else if (opts.target === "download") {
                chain = _triggerDownload(opts.filename || fname, blob);
            } else {
                // No target: capture-only, resolve with the blob.
                chain = Promise.resolve({
                    filename: opts.filename || fname,
                    bytes:    blob.size,
                });
            }
            return chain.then((r) => {
                if (opts.target) {
                    _fireOnExport({
                        kind:     "image",
                        target:   opts.target,
                        format:   "png",
                        filename: r.filename,
                        bytes:    r.bytes,
                    });
                }
                return { dataUrl: dataUrl, blob: blob,
                         filename: r.filename, bytes: r.bytes };
            });
        }

        function refit(opts) {
            // Per § 3.2: re-fit the camera to the structure (or to
            // a subset of atoms when ``opts.indices`` is supplied).
            // Optional ``opts.pullback`` multiplies the zoom AFTER
            // the fit (e.g. 0.55 = pull back so 45% more of the
            // surroundings stay in frame).  Defaults: no opts =
            // ``zoomTo()`` on all atoms (the historical behavior).
            if (state.disposed) return;
            opts = opts || {};
            const sel = _selectionFromIndices(opts.indices);
            try {
                state.viewer.zoomTo(sel);
                if (typeof opts.pullback === "number"
                    && opts.pullback > 0 && opts.pullback !== 1) {
                    state.viewer.zoom(opts.pullback, 0);
                }
                state.viewer.render();
            } catch (_) {}
        }

        function setPivot(opts) {
            // Per § 3.2: re-anchor the rotation / zoom-into-cursor
            // pivot on a subset of atoms.  3Dmol's ``center()``
            // translates the model so the selection's centroid
            // lands on the world origin (where rotations pivot);
            // the camera distance stays exactly where the user
            // left it.  Used by /modify's snap-pivot-to-molecule
            // pattern when slabs are present + the bounding box
            // would otherwise dominate the auto-pivot.
            //
            // ``opts.indices: number[]`` selects the subset;
            // omitting opts (or passing ``{}``) centers on all
            // atoms — equivalent to 3Dmol's ``center({}, 0)``.
            if (state.disposed) return;
            opts = opts || {};
            const sel = _selectionFromIndices(opts.indices);
            try { state.viewer.center(sel, 0); } catch (_) {}
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
                for (const l of state.pickLabels) state.viewer.removeLabel(l);
                for (const s of state.overlayHaloShapes) state.viewer.removeShape(s);
                for (const l of state.overlayMarkerLabels) state.viewer.removeLabel(l);
                state.viewer.clear();
            } catch (_) {}
            // Stop tracking container size changes (4a-2 setup).
            if (state._resizeObserver) {
                try { state._resizeObserver.disconnect(); }
                catch (_) {}
                state._resizeObserver = null;
            }
            // Remove interaction hooks (4a-3 setup).
            if (typeof state._interactionTeardown === "function") {
                try { state._interactionTeardown(); }
                catch (_) {}
                state._interactionTeardown = null;
            }
            try {
                if (state.cardEl && state.cardEl.parentNode) {
                    state.cardEl.parentNode.removeChild(state.cardEl);
                }
            } catch (_) {}
            // Review fix U2: break the state ↔ handle reference
            // cycle so GC can collect both promptly.  state.handle's
            // closure captures state; state.handle references the
            // handle.  Clearing both ends drops the cycle.
            state.handle = null;
            state.scaffold = null;
        }

        function _viewer3dmol() {
            // Escape hatch — see embedded-viewer.md § 2.4 notice.
            return state.viewer;
        }

        function setAnimation(animation) {
            // Per § 3.2: ``null`` clears; full opts (with ``kind``)
            // replaces; PARTIAL opts (no ``kind``) merge into the
            // active animation and update individual fields without
            // restarting the loop.  Review fix D1/N1: partial-update
            // path was previously broken — _normaliseAnimation
            // returned null for partials, which stopped the loop
            // on every spectra amplitude/speed slider tick.
            if (state.disposed) return;
            if (animation === null || animation === undefined) {
                _setAnimationImpl(state, null);
                return;
            }
            const cur = state.current.animation;
            const hasKind = animation && typeof animation === "object"
                            && (animation.kind === "vibration"
                                || animation.kind === "trajectory");
            if (!hasKind && cur) {
                // Partial update: mutate live-readable fields in
                // place.  The vibration tick reads ``amplitude`` and
                // ``speedHz`` every frame, so a direct assignment
                // takes effect on the next rAF.  Trajectory's
                // ``fps`` requires re-arming the setInterval; we
                // handle that via _setAnimationImpl with the merged
                // payload.
                if (cur.kind === "vibration") {
                    if (typeof animation.amplitude === "number"
                        && Number.isFinite(animation.amplitude)) {
                        cur.amplitude = animation.amplitude;
                    }
                    if (typeof animation.speedHz === "number"
                        && animation.speedHz > 0) {
                        cur.speedHz = animation.speedHz;
                    }
                    if (typeof animation.paused === "boolean") {
                        if (animation.paused) _pauseImpl(state);
                        else _playImpl(state);
                    }
                    return;
                }
                if (cur.kind === "trajectory") {
                    // Phase 5k B-1 fast path: fields the renderer
                    // reads live per frame (arrowsPerFrame, onFrame,
                    // loop) mutate in place without restarting the
                    // setInterval — matches the vibration partial-
                    // update model (amplitude / speedHz / paused
                    // above).  This keeps the live-poll cadence
                    // jitter-free.  Fields that require a timer
                    // re-arm (fps) or a structural restart (frames)
                    // fall through to the full _setAnimationImpl
                    // path below.
                    const liveFields = ["arrowsPerFrame", "onFrame",
                                        "loop"];
                    const restartFields = ["fps", "frames",
                                           "currentFrame", "startFrame"];
                    let needRestart = false;
                    for (const f of restartFields) {
                        if (Object.prototype.hasOwnProperty
                                .call(animation, f)) {
                            needRestart = true;
                            break;
                        }
                    }
                    if (!needRestart) {
                        for (const f of liveFields) {
                            if (Object.prototype.hasOwnProperty
                                    .call(animation, f)) {
                                cur[f] = animation[f];
                            }
                        }
                        if (typeof animation.paused === "boolean") {
                            if (animation.paused) _pauseImpl(state);
                            else _playImpl(state);
                        }
                        _refreshFrameStrip(state);
                        return;
                    }
                    const merged = Object.assign({}, cur, animation);
                    // Re-normalise to revalidate field types.
                    merged.kind = "trajectory";
                    const next = _normaliseAnimation(merged);
                    if (next) {
                        // Phase 5h I-1: currentFrame preservation now
                        // lives in _normaliseAnimation (honors caller-
                        // supplied currentFrame), so the merge inherits
                        // cur.currentFrame via Object.assign + normalise
                        // without an explicit override here.
                        _setAnimationImpl(state, next);
                    }
                    return;
                }
            }
            // Full update (kind supplied) OR no current animation
            // to merge into — go through the normal replace path.
            // Validate first per § 5.3: explicit kind + bad shape +
            // atom-count mismatch all fire invalid_input.
            if (animation && typeof animation === "object") {
                if (animation.kind !== undefined
                    && animation.kind !== "vibration"
                    && animation.kind !== "trajectory") {
                    _dispatchInvalidInput(state,
                        "setAnimation: 'kind' must be 'vibration' or "
                      + "'trajectory'; got "
                      + JSON.stringify(animation.kind));
                    return;
                }
                if (animation.kind === "vibration") {
                    if (!Array.isArray(animation.displacements)) {
                        _dispatchInvalidInput(state,
                            "setAnimation: vibration requires "
                          + "'displacements' as a number[][][3] array");
                        return;
                    }
                    const nAtoms = _atomCount(state.viewer);
                    if (nAtoms > 0
                        && animation.displacements.length !== nAtoms) {
                        _dispatchInvalidInput(state,
                            "setAnimation: vibration atom-count "
                          + "mismatch — structure has " + nAtoms
                          + " atoms, displacements supplied for "
                          + animation.displacements.length);
                        return;
                    }
                }
                if (animation.kind === "trajectory") {
                    if (!Array.isArray(animation.frames)
                        || animation.frames.length === 0) {
                        _dispatchInvalidInput(state,
                            "setAnimation: trajectory requires "
                          + "non-empty 'frames' array");
                        return;
                    }
                    const nAtoms = _atomCount(state.viewer);
                    const f0 = animation.frames[0];
                    if (nAtoms > 0 && Array.isArray(f0)
                        && f0.length !== nAtoms) {
                        _dispatchInvalidInput(state,
                            "setAnimation: trajectory atom-count "
                          + "mismatch — structure has " + nAtoms
                          + " atoms, frame[0] supplied for "
                          + f0.length);
                        return;
                    }
                }
            }
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
            setBackground:      setBackground,
            setOverlays:        setOverlays,
            setAtomStyle:       setAtomStyle,

            setKnobs:           setKnobs,

            setAnimation:       setAnimation,
            appendFrames:       appendFrames,
            playAnimation:      playAnimation,
            pauseAnimation:     pauseAnimation,
            isAnimationPlaying: isAnimationPlaying,
            setAnimationFrame:  setAnimationFrame,
            getAnimationFrame:  getAnimationFrame,

            getAtomCount:       getAtomCount,
            getElements:        getElements,
            getPickedIndices:   getPickedIndices,
            setPickedIndices:   setPickedIndices,
            getStructureText:   getStructureText,
            getCamera:          getCamera,
            setCamera:          setCamera,

            // Read accessors for declarative state (D3 — round-trip
            // pattern for host-side persistence; setX(getX()) is
            // idempotent).
            getStyle:           getStyle,
            getAxes:            getAxes,
            getCell:            getCell,
            getLabels:          getLabels,
            getOverlays:        getOverlays,
            getPick:            getPick,
            getKnobs:           getKnobs,
            getArrows:          getArrows,
            getAnimation:       getAnimation,
            getBackground:      getBackground,
            getLattice:         getLattice,

            // Ordered batch runner (D4 — persistence round-trip).
            applyState:         applyState,

            exportData:         exportData,
            screenshot:         screenshot,
            captureFrames:      captureFrames,
            exportAnimation:    exportAnimation,

            refit:              refit,
            setPivot:           setPivot,
            render:             render,
            dispose:            dispose,
            _viewer3dmol:       _viewer3dmol,
            _test:              _buildTestHandle(state),
        };
    }

    /* ------------------------------------------------------------ */
    /*  Test affordance surface (§ 9.2)                              */
    /* ------------------------------------------------------------ */

    /**
     * Per-instance test handle.  Stable contract for Playwright +
     * unit tests; renames require updating § 9.2 + § 9.4 in the
     * doc.  Read-only / no-mutate methods that wrap internal
     * state inspection so tests don't need ``_viewer3dmol()``.
     */
    function _buildTestHandle(state) {
        return {
            getCanvasElement() {
                return state.canvasEl
                    ? state.canvasEl.querySelector("canvas") : null;
            },
            getOverlayShapeCount() {
                // Sum of every removable shape array (overlay halos,
                // pick halos, cell wireframe, arrow shafts).  Used
                // by tests asserting "setOverlays added N atoms-worth
                // of halos".
                return state.overlayHaloShapes.length
                     + state.pickShapes.length
                     + state.cellShapes.length
                     + state.arrowShapes.length;
            },
            getOverlayLabelCount() {
                return state.overlayMarkerLabels.length
                     + state.labelHandles.length
                     + state.arrowLabels.length;
            },
            getKnobBarElement() {
                return state.cardEl
                    ? state.cardEl.querySelector(".mol-viewer-knobs")
                    : null;
            },
            getFrameStripElement() {
                return state.frameStripEl || null;
            },
            hasAnimationLoop() {
                return !!(state._anim && (state._anim.rafId !== null
                                       || state._anim.intervalId !== null));
            },
            getCurrentBackground() {
                return state.current && state.current.style
                    ? state.current.style.background : null;
            },
            getCurrent() {
                // Snapshot of the embed's normalised render state.
                // Used by tests that need to assert on internal state
                // shape (e.g. "overlays cleared after element-mismatch
                // setStructure") without reaching into private state.
                // Returns the live ``state.current`` object — callers
                // MUST treat it as read-only.
                return state.current || null;
            },
            getDependencyStatus() {
                // Snapshot of soft / integration dep availability
                // per § 2.5.  Used by tests asserting "axes were
                // skipped because mol-axes.js is absent" without
                // injection (§ 9.3 spec).
                const mb = (root.molbuilder || {});
                return {
                    axes:          !!mb.axes,
                    style:         !!mb.style,
                    format:        !!mb.fmt,
                    projects:      !!mb.projects,
                    clipboard:     !!(root.navigator
                                      && root.navigator.clipboard),
                    mediaRecorder: typeof root.MediaRecorder !== "undefined",
                    gif:           !!root.GIF
                                      ? "loaded"
                                      : (root._molbuilderGifLoading
                                          ? "loading" : "absent"),
                };
            },
            triggerKnob(name, arg) {
                // Phase 6: 2-menu structure — name targets the
                // submenu item directly.  Supported names:
                //   "style"      → arg.rep (button under View → Style)
                //   "labels"     → toggle in View → Labels
                //   "background" → arg.color or arg.custom
                //   "axes"       → toggle in View → Axes
                //   "reset"      → button in View → Reset
                //   "export"     → arg.target + arg.kind + arg.format
                //                  (button in Export menu)
                //   "view"       → toggle the View menu open/close
                //   "export-menu"→ toggle the Export menu open/close
                const bar = this.getKnobBarElement();
                if (!bar) return;
                arg = arg || {};
                let target = null;
                if (name === "style" && arg.rep) {
                    target = bar.querySelector(
                        '.mol-viewer-rep-btn[data-rep="' + arg.rep + '"]');
                } else if (name === "labels") {
                    target = bar.querySelector(
                        '.mol-viewer-toggle[data-action="labels"]');
                } else if (name === "background" && arg.color) {
                    target = bar.querySelector(
                        '.mol-viewer-bg-swatch[data-color="' + arg.color + '"]');
                } else if (name === "axes") {
                    target = bar.querySelector(
                        '.mol-viewer-toggle[data-action="axes"]');
                } else if (name === "reset") {
                    target = bar.querySelector(
                        '.mol-viewer-action[data-action="reset"]');
                } else if (name === "export"
                           && arg.kind && arg.target && arg.format) {
                    target = bar.querySelector(
                        '.mol-viewer-export-btn[data-kind="' + arg.kind
                        + '"][data-target="' + arg.target
                        + '"][data-format="' + arg.format + '"]');
                } else if (name === "view") {
                    target = bar.querySelector(
                        ".mol-viewer-menu-view > summary");
                } else if (name === "export-menu") {
                    target = bar.querySelector(
                        ".mol-viewer-menu-export > summary");
                }
                if (target && typeof target.click === "function") {
                    target.click();
                }
            },
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
    root.molbuilder.viewer._normaliseOverlays = _normaliseOverlays;
    root.molbuilder.viewer._normaliseKnobs    = _normaliseKnobs;
    root.molbuilder.viewer._equalNormalised   = _equalNormalised;

    // Error model — § 3.14, § 5.  Exposed for consumers that want to
    // construct ViewerErrors at the host/test boundary (rare; mostly
    // the embed builds its own).
    root.molbuilder.viewer.ViewerErrorCodes   = VIEWER_ERROR_CODES;
    root.molbuilder.viewer._makeError         = _makeError;
    root.molbuilder.viewer._throwable         = _throwable;
    root.molbuilder.viewer._dispatchError     = _dispatchError;
})(typeof window !== "undefined" ? window : this);
