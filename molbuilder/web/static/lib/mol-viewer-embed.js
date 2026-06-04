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

    // Closed enums for sync setX input validation (§ 5.3).  Mirrors
    // the option lists in § 3.3 / 3.4 / 3.6 / 3.8 / 3.10 of the
    // embed contract; the corresponding setters dispatch
    // ``invalid_input`` when the supplied value falls outside the
    // closed set and continue with the documented default.
    const VALID_REPS = [
        "stick", "ball-and-stick", "sphere", "line", "cross",
        "cartoon",
    ];
    const VALID_AXES_MODES   = ["auto", "world"];
    const VALID_PICK_MODES   = ["none", "single", "pair", "multi"];
    const VALID_LABEL_FORMS  = ["index", "name", "element"];
    const VALID_KNOB_POSITIONS = ["top", "bottom"];

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
        // Normalise the legacy ``atoms: "indices" | "names"`` sentinel
        // to the modern split shape so downstream code only has to
        // handle one shape.  "indices" → all atoms, format index;
        // "names" → all atoms, format name.
        let atoms  = l.atoms;
        let format = l.format;
        if (atoms === "indices") { atoms = "all"; if (!format) format = "index"; }
        if (atoms === "names")   { atoms = "all"; if (!format) format = "name"; }
        // Validate atoms shape.
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
        } else if (p.halo === undefined) {
            // Legacy fallback OR default.
            const legacyColor  = typeof p.haloColor  === "string"  ? p.haloColor  : null;
            const legacyRadius = typeof p.haloRadius === "number" ? p.haloRadius : null;
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
     * Edge cases pinned per § 3.10:
     *   - true / undefined → all default knobs visible.
     *   - false            → bar hidden entirely (null return).
     *   - object           → per-knob true / false / "auto"; missing
     *                        keys default to true.
     *   - labelsFormats: [] → format selector empty (and bar warns
     *                        via dispatch caller).
     */
    function _normaliseKnobs(k) {
        if (k === false) return null;
        const opts = (k && typeof k === "object" && k !== true) ? k : {};

        // Per § 3.10 (review fix D6): handle the four ``labelsFormats``
        // edge cases.
        //   undefined            → all three formats offered
        //   ["index"] etc.       → single-format toggle
        //   []                   → invalid: hide the Labels knob AND
        //                          signal via the bag's
        //                          _invalidLabelsFormats flag so the
        //                          caller can dispatch onError
        //   duplicate entries    → de-duplicated, order preserved
        let lfClean;
        let labelsHidden = opts.labels === false;
        let invalidLabelsFormats = false;
        if (opts.labelsFormats === undefined) {
            lfClean = ["index", "name", "element"];
        } else if (Array.isArray(opts.labelsFormats)) {
            if (opts.labelsFormats.length === 0) {
                invalidLabelsFormats = true;
                labelsHidden = true;
                lfClean = [];
            } else {
                const filtered = opts.labelsFormats.filter(
                    (s) => s === "index" || s === "name"
                        || s === "element");
                const seen = new Set();
                lfClean = filtered.filter((s) => {
                    if (seen.has(s)) return false;
                    seen.add(s);
                    return true;
                });
                if (lfClean.length === 0) {
                    invalidLabelsFormats = true;
                    labelsHidden = true;
                }
            }
        } else {
            lfClean = ["index", "name", "element"];
        }

        return {
            style:      opts.style      !== false,
            labels:     !labelsHidden,
            axes:       opts.axes       !== false,
            reset:      opts.reset      !== false,
            screenshot: opts.screenshot !== false,
            background: opts.background !== false,
            export:     opts.export     !== false,
            position:   opts.position === "bottom" ? "bottom" : "top",
            compact:    opts.compact === true,
            labelsFormats: lfClean,
            backgroundPresets: Array.isArray(opts.backgroundPresets)
                ? opts.backgroundPresets.slice()
                : ["#ffffff", "#1c1c1c", "transparent"],
            backgroundAllowCustom: opts.backgroundAllowCustom !== false,
            // Internal flag so embed() can dispatch invalid_input
            // exactly once at mount.  Not part of the public KnobBarOpts.
            _invalidLabelsFormats: invalidLabelsFormats,
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

        // ``card.bare`` was a first-pass migration shim that let
        // host pages skip the standard chrome.  All five consumer
        // sites (Build, Modify, structure/trajectory/spectra
        // inspectors) finished migration on 2026-06-03; the option
        // is gone per § 2.4 deprecation removal trigger.  If a
        // legacy caller still passes ``card.bare: true`` we now
        // ignore it (the standard chrome shows regardless).
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

        return { section, canvas, infoLineEl, knobsEl, knobs };
    }

    /* ------------------------------------------------------------ */
    /*  Knob bar DOM (§ 6.2) — built before handle exists; wired    */
    /*  in a second pass after _buildHandle completes               */
    /* ------------------------------------------------------------ */

    function _buildKnobBarDOM(knobs) {
        const bar = document.createElement("div");
        bar.className = "mol-viewer-knobs"
                      + (knobs.compact ? " mol-viewer-knobs-compact" : "");
        bar.setAttribute("role", "toolbar");
        bar.setAttribute("aria-label", "Viewer controls");

        // Style picker (HTML <select>; default rep options come
        // from mol-style.js).
        if (knobs.style) {
            const sel = document.createElement("select");
            sel.className = "mol-viewer-knob mol-viewer-knob-style";
            sel.setAttribute("aria-label", "Representation style");
            const reps = [
                ["stick",          "Stick"],
                ["ball-and-stick", "Ball & stick"],
                ["sphere",         "Sphere"],
                ["line",           "Line"],
                ["cross",          "Cross"],
                ["cartoon",        "Cartoon"],
            ];
            for (const [v, label] of reps) {
                const opt = document.createElement("option");
                opt.value = v;
                opt.textContent = label;
                sel.appendChild(opt);
            }
            bar.appendChild(sel);
        }

        // Labels popover (4 buttons: Index / Name / Element / Off).
        if (knobs.labels) {
            const det = document.createElement("details");
            det.className = "mol-viewer-knob mol-viewer-knob-labels";
            const sum = document.createElement("summary");
            sum.textContent = "Labels";
            det.appendChild(sum);
            for (const fmt of knobs.labelsFormats) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.setAttribute("data-format", fmt);
                btn.textContent = fmt === "index"
                    ? "Index"
                    : fmt === "name"
                        ? "Name"
                        : "Element";
                det.appendChild(btn);
            }
            const off = document.createElement("button");
            off.type = "button";
            off.setAttribute("data-format", "off");
            off.textContent = "Off";
            det.appendChild(off);
            bar.appendChild(det);
        }

        // Axes toggle button.
        if (knobs.axes) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "mol-viewer-knob mol-viewer-knob-toggle";
            btn.setAttribute("data-knob", "axes");
            btn.setAttribute("aria-pressed", "false");
            btn.textContent = "Axes";
            bar.appendChild(btn);
        }

        // Reset view button.
        if (knobs.reset) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "mol-viewer-knob";
            btn.setAttribute("data-knob", "reset");
            btn.textContent = "Reset";
            bar.appendChild(btn);
        }

        // Screenshot button (Phase 5 wires the action).
        if (knobs.screenshot) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "mol-viewer-knob";
            btn.setAttribute("data-knob", "screenshot");
            btn.textContent = "PNG";
            btn.title = "Save current view as PNG";
            bar.appendChild(btn);
        }

        // Background popover (Phase 5 wires the action; DOM exists
        // now so test selectors are stable).
        if (knobs.background) {
            const det = document.createElement("details");
            det.className = "mol-viewer-knob mol-viewer-knob-background";
            const sum = document.createElement("summary");
            sum.textContent = "Background";
            det.appendChild(sum);
            for (const c of knobs.backgroundPresets) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.setAttribute("data-color", c);
                if (c !== "transparent") {
                    btn.style.background = c;
                    btn.setAttribute("aria-label", "Background " + c);
                } else {
                    btn.textContent = "·";
                    btn.title = "Transparent";
                }
                det.appendChild(btn);
            }
            if (knobs.backgroundAllowCustom) {
                const input = document.createElement("input");
                input.type = "color";
                input.setAttribute("data-knob", "background-custom");
                input.setAttribute("aria-label", "Custom background color");
                det.appendChild(input);
            }
            bar.appendChild(det);
        }

        // Export popover (DOM only; Phase 5 wires handlers).
        if (knobs.export) {
            const det = document.createElement("details");
            det.className = "mol-viewer-knob mol-viewer-knob-export";
            const sum = document.createElement("summary");
            sum.textContent = "Export";
            det.appendChild(sum);
            // Submenu fieldsets per § 6 DOM.  Hidden until structure
            // is loaded or animation is set — Phase 5 will toggle
            // their display on getStructureText / animation presence.
            for (const fieldset of _buildExportFieldsets()) {
                det.appendChild(fieldset);
            }
            bar.appendChild(det);
        }
        return bar;
    }

    function _buildExportFieldsets() {
        const sets = [];

        const struct = document.createElement("fieldset");
        struct.setAttribute("data-kind", "structure");
        const sLeg = document.createElement("legend");
        sLeg.textContent = "Structure";
        struct.appendChild(sLeg);
        for (const t of ["project", "download", "clipboard"]) {
            const b = document.createElement("button");
            b.type = "button";
            b.setAttribute("data-kind",   "structure");
            b.setAttribute("data-target", t);
            b.textContent = t === "project"
                ? "Save to project"
                : t === "download" ? "Download" : "Copy";
            struct.appendChild(b);
        }
        sets.push(struct);

        const img = document.createElement("fieldset");
        img.setAttribute("data-kind", "image");
        const iLeg = document.createElement("legend");
        iLeg.textContent = "Image";
        img.appendChild(iLeg);
        for (const t of ["project", "download"]) {
            const b = document.createElement("button");
            b.type = "button";
            b.setAttribute("data-kind",   "image");
            b.setAttribute("data-target", t);
            b.textContent = t === "project" ? "Save PNG to project" : "Download PNG";
            img.appendChild(b);
        }
        sets.push(img);

        const anim = document.createElement("fieldset");
        anim.setAttribute("data-kind", "animation");
        const aLeg = document.createElement("legend");
        aLeg.textContent = "Animation";
        anim.appendChild(aLeg);
        for (const f of ["webm", "gif"]) {
            for (const t of ["project", "download"]) {
                const b = document.createElement("button");
                b.type = "button";
                b.setAttribute("data-kind",   "animation");
                b.setAttribute("data-format", f);
                b.setAttribute("data-target", t);
                b.textContent = (t === "project" ? "Save " : "Download ")
                              + f.toUpperCase()
                              + (t === "project" ? " to project" : "");
                anim.appendChild(b);
            }
        }
        sets.push(anim);
        return sets;
    }

    /* ------------------------------------------------------------ */
    /*  Knob bar wiring — called after handle is built              */
    /* ------------------------------------------------------------ */

    function _wireKnobBar(state, bar, knobs) {
        const handle = state.handle;

        // Style picker.
        const styleSel = bar.querySelector(".mol-viewer-knob-style");
        if (styleSel) {
            styleSel.value = state.current.style.rep || "stick";
            styleSel.addEventListener("change", () => {
                handle.setStyle({ rep: styleSel.value });
            });
        }

        // Labels popover — each format button sets labels with that
        // format; "off" disables.  Per § 6.2 popover pattern: click
        // an action button → fires AND closes the popover.
        const labelsDet = bar.querySelector(".mol-viewer-knob-labels");
        if (labelsDet) {
            for (const btn of labelsDet.querySelectorAll("button[data-format]")) {
                btn.addEventListener("click", () => {
                    const fmt = btn.getAttribute("data-format");
                    if (fmt === "off") {
                        handle.setLabels(false);
                    } else {
                        handle.setLabels({ atoms: "all", format: fmt });
                    }
                    labelsDet.open = false;
                });
            }
        }

        // Axes toggle.
        const axesBtn = bar.querySelector('[data-knob="axes"]');
        if (axesBtn) {
            const initOn = !!state.current.axes;
            axesBtn.setAttribute("aria-pressed", initOn ? "true" : "false");
            axesBtn.addEventListener("click", () => {
                const nowOn = state.current.axes
                            ? false : true;
                handle.setAxes(nowOn);
                axesBtn.setAttribute("aria-pressed", nowOn ? "true" : "false");
            });
        }

        // Reset view.
        const resetBtn = bar.querySelector('[data-knob="reset"]');
        if (resetBtn) {
            resetBtn.addEventListener("click", () => handle.refit());
        }

        // Background popover — preset swatches change canvas color;
        // custom picker via <input type="color">.  Phase 2 wires
        // the action via handle.setStyle({background}); Phase 5
        // will integrate with save-to-project if requested.
        const bgDet = bar.querySelector(".mol-viewer-knob-background");
        if (bgDet) {
            for (const btn of bgDet.querySelectorAll("button[data-color]")) {
                btn.addEventListener("click", () => {
                    const c = btn.getAttribute("data-color");
                    handle.setStyle({
                        rep:         state.current.style.rep,
                        radiusScale: state.current.style.radiusScale,
                        background:  c,
                    });
                    bgDet.open = false;
                });
            }
            const customInput = bgDet.querySelector(
                '[data-knob="background-custom"]');
            if (customInput) {
                customInput.addEventListener("input", () => {
                    handle.setStyle({
                        rep:         state.current.style.rep,
                        radiusScale: state.current.style.radiusScale,
                        background:  customInput.value,
                    });
                });
            }
        }

        // Screenshot button → download a PNG immediately (Phase 5a).
        const ssBtn = bar.querySelector('[data-knob="screenshot"]');
        if (ssBtn) {
            ssBtn.addEventListener("click", () => {
                handle.screenshot({ target: "download" })
                      .catch((err) => _dispatchError(state, err));
            });
        }

        // Export popover — Structure / Image actions wire to
        // handle.exportData / handle.screenshot.  Animation
        // actions are stubbed until Phase 5b.
        const expDet = bar.querySelector(".mol-viewer-knob-export");
        if (expDet) {
            for (const btn of expDet.querySelectorAll("button[data-kind]")) {
                btn.addEventListener("click", () => {
                    const kind   = btn.getAttribute("data-kind");
                    const target = btn.getAttribute("data-target");
                    const format = btn.getAttribute("data-format");
                    let p;
                    if (kind === "structure") {
                        p = handle.exportData({ target: target });
                    } else if (kind === "image") {
                        p = handle.screenshot({ target: target });
                    } else if (kind === "animation") {
                        // Phase 5b: animation export.  Silent stub
                        // for now (no console spam from a click).
                        expDet.open = false;
                        return;
                    }
                    if (p) p.catch((err) => _dispatchError(state, err));
                    expDet.open = false;
                });
            }
        }

        // Popover mutual-exclusion: opening one closes the others
        // (click/tap rule per § 3.10).
        const allDetails = bar.querySelectorAll("details.mol-viewer-knob");
        for (const d of allDetails) {
            d.addEventListener("toggle", () => {
                if (d.open) {
                    for (const other of allDetails) {
                        if (other !== d) other.open = false;
                    }
                }
            });
        }

        // Keyboard shortcuts per § 3.10.  Listen on the card root so
        // typing in <input>/<textarea>/[contenteditable] inside the
        // card doesn't trigger.  When a popover is open, only its
        // own key / Esc / arrow nav fire.
        if (!state.cardEl) return;
        state.cardEl.addEventListener("keydown", (e) => {
            if (state.disposed) return;
            const t = e.target;
            const tag = t && t.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA"
                || (t && t.isContentEditable)) return;

            const openPopover = bar.querySelector("details.mol-viewer-knob[open]");
            const key = (e.key || "").toLowerCase();

            // Esc closes any popover.
            if (e.key === "Escape" && openPopover) {
                openPopover.open = false;
                e.preventDefault();
                return;
            }

            // While a popover is open, only the popover's own
            // opening key (re-press to close) fires; cross-knob
            // shortcuts are suppressed per § 3.10.
            if (openPopover) {
                const ownKey =
                    (openPopover.classList.contains("mol-viewer-knob-labels")    && key === "l")
                 || (openPopover.classList.contains("mol-viewer-knob-background") && key === "b")
                 || (openPopover.classList.contains("mol-viewer-knob-export")    && key === "e");
                if (ownKey) {
                    openPopover.open = false;
                    e.preventDefault();
                }
                return;
            }

            if (key === "r" && resetBtn) {
                resetBtn.click(); e.preventDefault();
            } else if (key === "l" && labelsDet) {
                labelsDet.open = true; e.preventDefault();
            } else if (key === "a" && axesBtn) {
                axesBtn.click(); e.preventDefault();
            } else if (key === "b" && bgDet) {
                bgDet.open = true; e.preventDefault();
            } else if (key === "e") {
                const expDet = bar.querySelector(".mol-viewer-knob-export");
                if (expDet) { expDet.open = true; e.preventDefault(); }
            }
        });
        // Card must accept focus for keyboard handling to work.
        if (!state.cardEl.hasAttribute("tabindex")) {
            // Review fix U3: tabindex="0" so the card is part of
            // the natural tab order; keyboard users can Tab onto
            // the viewer and trigger R / L / A / B / E shortcuts
            // without first clicking a knob.
            state.cardEl.setAttribute("tabindex", "0");
        }
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
                currentFrame:   startFrame,
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
        // Per § 6.3 (review fix D7): frame strip auto-mounts whenever
        // animation.kind === "trajectory".  The legacy
        // ``card.frameStrip`` opt was an undocumented gate that
        // would have required every trajectory consumer to opt in.
        if (state.frameStripEl) return;  // already built
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

        // 4b. Build the handle + stash on state BEFORE applying the
        //     initial animation.  Trajectory autoplay (paused:false)
        //     fires onFrame(idx, handle) on its first tick; if the
        //     handle isn't on state yet, the callback receives
        //     undefined and the silent catch swallows the failure.
        //     Review fix P1 (#221).
        const handle = _buildHandle(state);
        state.handle = handle;

        // Diagnostic dispatches per § 3.10 edge cases (review fix D6).
        // The knob bar is already built without the Labels knob when
        // labelsFormats: [] was passed; we just need to signal the
        // misconfiguration so the host's onError sees it.
        if (state.current.knobs
            && state.current.knobs._invalidLabelsFormats) {
            _dispatchError(state, _makeError(
                "invalid_input",
                "KnobBarOpts.labelsFormats: empty array hides the Labels "
              + "knob entirely; pass labels: false explicitly OR a "
              + "non-empty subset of ['index', 'name', 'element']."));
        }

        // 4c. Wire the standard knob bar's click handlers + keyboard
        //     shortcuts now that the handle exists.  Phase 5 will
        //     extend the export / background actions; Phase 2 covers
        //     the static-rendering knobs (style / labels / axes / reset).
        //     state.scaffold is retained (not nulled here) because
        //     setKnobs needs to find scaffold.knobsEl + scaffold.knobs
        //     to rebuild the bar in place (review fix O6 — could
        //     null after wiring + look up via the DOM, but the
        //     scaffold ref is cheap to retain).
        if (state.scaffold && state.scaffold.knobsEl) {
            _wireKnobBar(state, state.scaffold.knobsEl,
                                state.scaffold.knobs);
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
            state.pickedIndices = [];
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
            const next = _normaliseAxes(a);
            if (_equalNormalised(state.current.axes, next)) return;
            state.current.axes = next;
            _redrawAxes(state);
            state.viewer.render();
            // Sync the knob bar's aria-pressed state per § 4.1
            // invariant ("calling setLabels(true) programmatically
            // also updates the labels knob's pressed state").
            if (state.scaffold && state.scaffold.knobsEl) {
                const btn = state.scaffold.knobsEl.querySelector(
                    '[data-knob="axes"]');
                if (btn) {
                    btn.setAttribute("aria-pressed",
                                     next ? "true" : "false");
                }
            }
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
                    && l.atoms !== "indices" && l.atoms !== "names"
                    && !Array.isArray(l.atoms)) {
                    _dispatchInvalidInput(state,
                        "setLabels: 'atoms' must be 'all' or a "
                      + "number[]");
                } else if (Array.isArray(l.atoms)) {
                    const bad = l.atoms.filter(
                        (v) => !Number.isInteger(v) || v < 0);
                    if (bad.length) {
                        _dispatchInvalidInput(state,
                            "setLabels: " + bad.length + " atom index/"
                          + "indices out of range (must be non-negative "
                          + "integers); dropping them");
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
            if (_equalNormalised(state.current.labels, next)) return;
            state.current.labels = next;
            _redrawLabels(state);
            state.viewer.render();
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
                _dispatchError(state, _makeError(
                    "invalid_input",
                    "setBackground: color must be a non-empty CSS color string"));
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
                if (k.position !== undefined
                    && !VALID_KNOB_POSITIONS.includes(k.position)) {
                    _dispatchInvalidInput(state,
                        "setKnobs: 'position' must be one of " +
                        VALID_KNOB_POSITIONS.join(", ") +
                        "; got " + JSON.stringify(k.position));
                }
                if (k.labelsFormats !== undefined
                    && Array.isArray(k.labelsFormats)) {
                    const bad = k.labelsFormats.filter(
                        (v) => !VALID_LABEL_FORMS.includes(v));
                    if (bad.length) {
                        _dispatchInvalidInput(state,
                            "setKnobs: 'labelsFormats' contains "
                          + bad.length + " unknown format(s); valid: "
                          + VALID_LABEL_FORMS.join(", "));
                    }
                }
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
            state.scaffold.knobs   = next;
            if (next) {
                const bar = _buildKnobBarDOM(next);
                state.scaffold.knobsEl = bar;
                // Insert before the canvas (and frame strip if any)
                // to keep the §6.1 anatomy order.
                state.cardEl.insertBefore(bar, state.canvasEl);
                _wireKnobBar(state, bar, next);
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
                    _dispatchError(state, _makeError(
                        "invalid_input",
                        "setOverlays: " + (inN - outN) + " of " + inN
                      + " atom entries dropped (bad/missing/multiple "
                      + "selectors, or no style/halo/marker)."));
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
                _dispatchError(state, _makeError(
                    "invalid_input",
                    "setAtomStyle: selector must be number[] OR "
                  + "{elements: string[]} OR {residues: number[]}"));
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
                    _dispatchError(state, _makeError(
                        "invalid_input",
                        "setAtomStyle: style must include at least one of "
                      + "{rep, radiusScale, color, opacity}"));
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
                    _dispatchError(state, _makeError(
                        "invalid_input",
                        "appendFrames: atom count mismatch — existing "
                      + "trajectory has " + expectedN + " atoms per frame, "
                      + "appended frame has " + (Array.isArray(f) ? f.length : "?")
                    ));
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
                _dispatchError(state, _makeError(
                    "invalid_input",
                    "setCamera: argument must be a CameraState object"));
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

        function captureFrames(opts) {
            // Documented per § 3.2; landing in Phase 5b.  Stub
            // returns a clean reject so consumers get a typed
            // error rather than "method is undefined".
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
            return Promise.reject(_makeError("unknown",
                "captureFrames: not yet implemented (Phase 5b)"));
        }

        function exportAnimation(opts) {
            // Documented per § 3.2; landing in Phase 5b.  Stub
            // returns a clean reject so consumers get a typed
            // error rather than "method is undefined".
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
            if (opts.format !== "webm" && opts.format !== "gif") {
                return Promise.reject(_makeError("invalid_input",
                    "exportAnimation: format must be 'webm' or 'gif'"));
            }
            return Promise.reject(_makeError("unknown",
                "exportAnimation: not yet implemented (Phase 5b)"));
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
            // (§ 11.3 doc note).
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
            // Escape hatch — see embedded-viewer.md § 2.2 notice.
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
                    const merged = Object.assign({}, cur, animation);
                    // Re-normalise to revalidate field types.
                    merged.kind = "trajectory";
                    const next = _normaliseAnimation(merged);
                    if (next) {
                        // Preserve the live playback index so a
                        // partial update doesn't snap back to frame 0.
                        next.currentFrame = cur.currentFrame;
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
            getDependencyStatus() {
                // Snapshot of soft / integration dep availability
                // per § 2.5.  Used by tests asserting "axes were
                // skipped because mol-axes.js is absent" without
                // injection (§ 9.3 spec).
                const mb = (root.molbuilder || {});
                return {
                    axes:          !!mb.axes,
                    style:         !!mb.style,
                    pick:          !!mb.pick,
                    format:        !!mb.fmt,
                    projects:      !!mb.projects,
                    clipboard:     !!(root.navigator
                                      && root.navigator.clipboard),
                    mediaRecorder: typeof root.MediaRecorder !== "undefined",
                    gif:           !!root.GIF
                                      ? "loaded" : "absent",
                };
            },
            triggerKnob(name, arg) {
                // Phase 2 (knob bar) wires the actual click flows.
                // Phase 7 stub: locate the knob element + delegate
                // to a click() on the right sub-element.  Without
                // the knob bar built, this is a no-op so tests can
                // be written before Phase 2 lands.
                const bar = this.getKnobBarElement();
                if (!bar) return;
                arg = arg || {};
                let target = null;
                if (name === "labels"     && arg.format) {
                    target = bar.querySelector(
                        ".mol-viewer-knob-labels [data-format=\""
                          + arg.format + "\"]");
                } else if (name === "background" && arg.color) {
                    target = bar.querySelector(
                        ".mol-viewer-knob-background [data-color=\""
                          + arg.color + "\"]");
                } else if (name === "export"  && arg.kind && arg.target) {
                    let sel = ".mol-viewer-knob-export [data-kind=\""
                            + arg.kind + "\"][data-target=\""
                            + arg.target + "\"]";
                    if (arg.formatExport) {
                        sel += "[data-format=\"" + arg.formatExport + "\"]";
                    }
                    target = bar.querySelector(sel);
                } else if (name === "style"      && arg.rep) {
                    const sel = bar.querySelector(".mol-viewer-knob-style");
                    if (sel) {
                        sel.value = arg.rep;
                        sel.dispatchEvent(new root.Event("change",
                                                         { bubbles: true }));
                    }
                    return;
                } else {
                    // Plain button: axes / reset / screenshot, or
                    // popover toggle (labels / background / export
                    // without arg → click the summary).
                    target = bar.querySelector(
                        ".mol-viewer-knob[data-knob=\"" + name + "\"]"
                    ) || bar.querySelector(
                        ".mol-viewer-knob-" + name + " summary"
                    );
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
    root.molbuilder.viewer._equalNormalised   = _equalNormalised;

    // Error model — § 3.14, § 5.  Exposed for consumers that want to
    // construct ViewerErrors at the host/test boundary (rare; mostly
    // the embed builds its own).
    root.molbuilder.viewer.ViewerErrorCodes   = VIEWER_ERROR_CODES;
    root.molbuilder.viewer._makeError         = _makeError;
    root.molbuilder.viewer._throwable         = _throwable;
    root.molbuilder.viewer._dispatchError     = _dispatchError;
})(typeof window !== "undefined" ? window : this);
