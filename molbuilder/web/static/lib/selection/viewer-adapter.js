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
 * Full spec: docs/protocols/molview-module.md
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

    function attach(handle, opts) {
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
        // Phase 5 (fused module): a caller MAY pass its own store instance
        // (``opts.store``) so a readonly/ephemeral inspector paints from an
        // ISOLATED selection; defaults to the workspace singleton (Modify
        // unchanged).
        const store = (opts && opts.store) || (root.molbuilder
                       && root.molbuilder.workspace
                       && root.molbuilder.molview && root.molbuilder.molview.data
                       && root.molbuilder.molview.data.selection);
        if (!store) {
            throw new Error(
                "viewer-adapter.attach: ws.selection missing; "
                + "load lib/workspace/dispatcher.js first"
            );
        }
        // Phase 5 (fused module, decision A / § 6.4): viewer clicks ALWAYS feed
        // the store selection -- including a ``readonly`` mount.  There is no
        // pick-vs-selection conflict to protect anymore: the measurement overlay
        // derives from the selection, so a click that toggles the store is what
        // drives the readout.  (S3 used to skip clicks in readonly to preserve a
        // separate triple-pick chip; that chip is retired.)

        // ----- click wiring (VIEW -> STORE, molview-module.md §13.2) --- //
        //
        // CONTRACT: a viewer click forwards to ``store.toggle(atom)`` -- the
        // STORE is the single source of truth for the selection (§13.2).  The
        // embed runs in "single" pick mode with halos/labels OFF (the adapter
        // paints every halo via setOverlays), so the embed's own pick buffer is
        // NOT a second selection: it only tracks the click to report WHICH atom
        // to toggle.  A multi-atom selection is still built by clicking -- each
        // click toggles one atom in the store, which accumulates them; the
        // measurement overlay (§15) then derives its 1/2/3-atom readout from
        // that store selection.  (Do NOT switch this to "multi" + store.set:
        // that moves the accumulator into the embed buffer -- a second source of
        // truth -- and breaks the inspector's single-mode pin.)
        //
        // The ``prevClicked`` shim tells us which atom to toggle when a
        // single-mode pick reports an EMPTY set (= the same atom clicked twice
        // to deselect it).  This is now the ONLY way onPick sees an empty set:
        // a programmatic structure swap (add/delete) clears the embed's pick
        // buffer SILENTLY -- mol-viewer-embed setStructure no longer fabricates
        // an onPick([]) -- so re-deriving the view can never masquerade as a
        // click and re-toggle the last-clicked atom back on after an edit.
        let prevClicked = null;
        function onPick(curr) {
            if (!Array.isArray(curr)) return;
            // §14.3 (molview-module.md): in-window picking is DISABLED while the viewer
            // shows a DERIVED list -- isolate (only the selected atoms are drawn) OR k-grid
            // (tiled copies): a click then has no unambiguous unit-cell atom, so drop it.
            // The selection panel (unit-cell atom list) still edits the selection.
            const st = store.getState() || {};
            const isolating = !!st.isolate
                && (Array.isArray(st.indices) ? st.indices.length : 0) > 0;
            if ((st.kgrid && st.kgrid.enabled) || isolating) { prevClicked = null; return; }
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
        // Build an OverlaySpec from the store state.  Overlay entries process in array
        // order; later halos sit on top, so selection is pushed LAST (brightest yellow,
        // largest radius -> the "live" set reads above the frozen + region halos).
        //
        // The adapter paints halos ONLY on the plain, full-list base draw, where the drawn
        // atom index equals the unit-cell index the halos are keyed on.  "Show selected
        // only" (isolate) is a REAL list filter in the render controller (mountKgridRender),
        // NOT an opacity/hidden trick here -- so when isolate (or k-grid) is on the adapter
        // stands its overlays down entirely (the derived list has a different index space).
        // isolate is STORE state (the single source of truth): the panel/view-controls drive
        // it via store.setIsolate; this adapter is a pure consumer with no isolate control
        // of its own.

        function render(s) {
            s = s || {};
            // DISPLAY-ONLY view (molview-module.md §14.3): while isolate OR k-grid is on,
            // the viewer shows a DERIVED atom list -- isolate FILTERS it to the selected
            // atoms (a real filter, in the render controller -- NOT a hidden/opacity trick),
            // k-grid TILES it.  Either way the drawn atom index no longer equals the
            // unit-cell index, so these overlay entries (keyed by unit-cell index) would
            // land on the wrong atoms.  Stand every overlay down; the panel is the
            // selection surface in these modes.  The render controller (mountKgridRender)
            // owns the derived draw; the adapter paints halos only on the plain, full-list
            // base draw, where drawn index == unit-cell index.
            const isolating = !!s.isolate
                && (Array.isArray(s.indices) ? s.indices.length : 0) > 0;
            const tiling = !!(s.kgrid && s.kgrid.enabled);
            if (isolating || tiling) {
                try { handle.setOverlays(null); } catch (_) { /* already clear */ }
                return;
            }

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
                const filtered = indices;
                if (!filtered.length) return;
                atoms.push({
                    indices: filtered,
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
            const frozenFiltered = frozenIdx;
            if (frozenFiltered.length) {
                atoms.push({
                    indices: frozenFiltered,
                    halo:    { color:   HALO_FROZEN.color,
                               radius:  HALO_FROZEN.radius,
                               opacity: HALO_FROZEN.opacity },
                });
            }

            // 3. Selection halo -- largest ball + brightest, so it
            //    reads as the "live" set above region tints.  In
            //    isolate mode the selection IS the visible set, so
            //    no filter is needed here.
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

        return {
            dispose:         dispose,
        };
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
