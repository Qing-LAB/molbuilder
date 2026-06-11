/* Atom-selection viewer-adapter -- declarative consumer of the
 * selection store on top of the embedded MolViewer (#229 Part B
 * migration, 2026-06-03).
 *
 * Subscribes to the selection store and paints overlays via the
 * embed handle's declarative ``setOverlays`` API:
 *   * per-region color halos
 *   * frozen-atom red marker halos
 *   * selection halo on the current pick set
 *
 * Click handling: routes through the embed's PickOpts.onPick so
 * clicks reach the store via store.toggle.  The embed's own
 * halo/label rendering is DISABLED (we paint everything via
 * setOverlays so the store stays the single source of truth).
 *
 * Public surface:
 *
 *   attach(handle) -> { dispose }
 *
 * The argument is the embed HANDLE (from window.molbuilder.viewer.
 * embed()), not a raw 3Dmol viewer.  Pre-Part-B callers passing
 * a raw viewer now error at attach time so script-order bugs
 * surface immediately rather than silently degrading.
 *
 * Full spec: docs/protocols/atom-selection.md
 */
(function (root) {
    "use strict";

    // Default per-region palette.  Mirrors the green/red tag colors
    // in the panel's atom-list so the viewer "agrees" with the panel
    // on what each region looks like.
    const REGION_COLORS = {
        "L-electrode": "#7fc97f",
        "R-electrode": "#beaed4",
        "bridge":      "#fdc086",
        "interface":   "#ffff99",
    };

    // Halo geometry constants (Å).  ``radius`` picks a sphere big
    // enough to read as a halo, small enough not to swamp neighbours.
    // ``opacity`` < 1 lets the underlying atom show through.
    const HALO_REGION = { radius: 0.5, opacity: 0.35 };
    const HALO_FROZEN = { color: "#ff5050", radius: 0.25, opacity: 0.85 };
    const HALO_SELECT = { color: "yellow", radius: 0.7, opacity: 0.45 };

    // Stable-but-cheap label-to-color hash for unknown labels.
    function _fallbackColor(label) {
        let h = 0;
        for (let i = 0; i < label.length; i++) {
            h = ((h << 5) - h + label.charCodeAt(i)) | 0;
        }
        const palette = ["#a6d8a4", "#ffb482", "#a6c8ff", "#ffa6c5",
                         "#d4b8ff", "#ffd17c", "#7cdfdf", "#ff9b9b"];
        return palette[Math.abs(h) % palette.length];
    }

    function _colorFor(label) {
        // hasOwnProperty.call -- a label like ``"constructor"`` or
        // ``"__proto__"`` must not pierce through to the Object
        // prototype.  Labels are user-controlled (free-form via the
        // "+ new region label" input).
        if (Object.prototype.hasOwnProperty.call(REGION_COLORS, label)) {
            return REGION_COLORS[label];
        }
        return _fallbackColor(label);
    }

    function attach(handle) {
        if (!handle || typeof handle.setOverlays !== "function") {
            throw new Error(
                "viewer-adapter.attach: expected an embed handle "
              + "(window.molbuilder.viewer.embed return value), got "
              + (handle && typeof handle));
        }
        // Phase 10 (workspace-contract.md §5): bind against ws.selection,
        // not the legacy selection.store global.  ws.selection
        // exposes the full toggle/subscribe/getState surface used
        // by this adapter.
        const store = (root.molbuilder
                       && root.molbuilder.workspace
                       && root.molbuilder.workspace.selection);
        if (!store) {
            throw new Error(
                "viewer-adapter.attach: ws.selection missing; "
                + "load lib/workspace/dispatcher.js first"
            );
        }

        // ----- click wiring --------------------------------------- //
        //
        // Use the embed's PickOpts.onPick in "single" mode + halo/
        // label off (we paint everything via setOverlays).  Each
        // click toggles the just-clicked atom in the store; the
        // ``prevClicked`` shim tells us which atom to toggle when
        // single-mode pick reports an empty set (= same atom clicked
        // twice).
        //
        // Switching to filter mode hides the atom-list editor but
        // the 3D viewer stays visible -- a click there is still a
        // meaningful user action on the shared selection.
        let prevClicked = null;
        function onPick(curr) {
            if (!Array.isArray(curr)) return;
            if (curr.length === 0) {
                // Single-mode deselect: same atom clicked twice.
                if (prevClicked !== null) {
                    store.toggle(prevClicked);
                    prevClicked = null;
                }
            } else {
                const idx = curr[curr.length - 1];
                store.toggle(idx);
                prevClicked = idx;
            }
        }
        try {
            handle.setPick({
                mode:   "single",
                halo:   false,
                label:  false,
                onPick: onPick,
            });
        } catch (e) {
            if (root.console) root.console.warn(
                "[viewer-adapter] setPick failed", e
            );
        }

        // ----- render ---------------------------------------------- //
        //
        // Build an OverlaySpec from the store state.  Layering rules
        // from § 3.12 process entries in array order; later halos
        // sit on top so we push selection LAST (brightest yellow,
        // largest radius -> the "live" set reads above the frozen +
        // region halos at every overlap).

        function render(s) {
            const atoms = [];

            // 1. Region tints -- group by first label so each atom
            //    gets exactly one tint.  (An atom can carry multiple
            //    labels; the panel surfaces all of them in the label
            //    column.  The viewer picks the first for color so
            //    users still see the primary grouping at a glance.)
            //
            //    Use a Map (not ``{}``) so a label named ``__proto__``
            //    or ``constructor`` doesn't piggyback on
            //    Object.prototype: reading ``byRegion["__proto__"]``
            //    on a plain object returns Object.prototype (which
            //    has no ``push``), so the legacy form crashed the
            //    whole render and the try/catch swallowed it, leaving
            //    overlays silently disabled for the rest of the page
            //    lifetime.  Map keys are stored separately from any
            //    prototype.
            const byRegion = new Map();
            (s.atoms || []).forEach((a) => {
                const label = (a.labels && a.labels[0]) || null;
                if (!label) return;
                let bucket = byRegion.get(label);
                if (!bucket) {
                    bucket = [];
                    byRegion.set(label, bucket);
                }
                bucket.push(a.index);
            });
            byRegion.forEach((indices, label) => {
                atoms.push({
                    indices: indices,
                    halo: {
                        color:   _colorFor(label),
                        radius:  HALO_REGION.radius,
                        opacity: HALO_REGION.opacity,
                    },
                });
            });

            // 2. Frozen markers -- tiny red ball on each frozen atom.
            const frozenIdx = (s.atoms || [])
                .filter((a) => a.isFrozen)
                .map((a) => a.index);
            if (frozenIdx.length) {
                atoms.push({
                    indices: frozenIdx,
                    halo:    { color:   HALO_FROZEN.color,
                               radius:  HALO_FROZEN.radius,
                               opacity: HALO_FROZEN.opacity },
                });
            }

            // 3. Selection halo -- largest ball + brightest, so it
            //    reads as the "live" set above region tints.
            // Phase 10 (workspace-contract.md §5): subscriber state
            // shape is the contract object — selection lives on
            // ``indices``, not ``selection``.
            const sel = Array.isArray(s.indices) ? s.indices : [];
            if (sel.length) {
                atoms.push({
                    indices: sel.slice(),
                    halo:    { color:   HALO_SELECT.color,
                               radius:  HALO_SELECT.radius,
                               opacity: HALO_SELECT.opacity },
                });
            }

            try { handle.setOverlays({ atoms: atoms }); }
            catch (e) {
                if (root.console) root.console.warn(
                    "[viewer-adapter] setOverlays failed", e
                );
            }
        }

        // ----- subscribe ------------------------------------------- //
        //
        // The store's setSourceFile awaits loadStructureText BEFORE
        // firing subscribers, so the viewer's model is already
        // swapped by the time we paint overlays here.  No race, no
        // rerender dance.

        const unsubscribe = store.subscribe((s) => render(s));

        function dispose() {
            try { unsubscribe(); } catch (e) { /* ignore */ }
            // Detach the click handler so 3Dmol stops routing
            // clicks into the orphaned store reference.  Setting
            // mode: "none" disables pick entirely.
            try {
                handle.setPick({ mode: "none" });
            } catch (e) { /* ignore */ }
            // Drop our overlay set so the page's base style returns
            // to view (handle.setOverlays(null) clears all entries).
            try { handle.setOverlays(null); }
            catch (e) { /* ignore */ }
        }

        return { dispose: dispose };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.selection = root.molbuilder.selection || {};
    root.molbuilder.selection.viewerAdapter = {
        attach:         attach,
        _fallbackColor: _fallbackColor,   // exported for tests
        REGION_COLORS:  REGION_COLORS,
    };
    // Module-init contract: register with runtime (design.md).
    if (root.molbuilder.runtime
        && typeof root.molbuilder.runtime.register === "function") {
        root.molbuilder.runtime.register(
            "selection.viewerAdapter",
            root.molbuilder.selection.viewerAdapter);
    }
})(typeof window !== "undefined" ? window : globalThis);
