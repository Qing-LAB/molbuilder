/* Atom-selection viewer-adapter -- 3Dmol consumer of the selection store.
 *
 * Subscribes to the selection store and paints overlays on a 3Dmol
 * viewer:
 *   * per-region color tints
 *   * frozen-atom marker
 *   * halo for the current selection
 *
 * Forwards 3Dmol click events to ``store.toggleAtom`` so user picks
 * flow back through the same store the panel uses.
 *
 * Full spec: docs/protocols/atom-selection.md
 *
 * Public surface:
 *
 *   attach(viewer) -> { dispose }
 *
 * If the store is missing (script loaded out of order), throws at
 * attach time so the page boot fails loud rather than silently.
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

    // Sphere-shape specs used as overlays on top of the atom's primary
    // style.  ``radius`` is in Å.  ``opacity`` < 1 lets the underlying
    // atom show through; pick big enough to read as a halo, small
    // enough not to swamp neighbouring atoms.
    const HALO_STYLE = {
        color:   "yellow",
        radius:  0.7,
        opacity: 0.45,
    };

    const FROZEN_MARKER = {
        color:   "#ff5050",
        radius:  0.25,
        opacity: 0.85,
    };

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

    function attach(viewer) {
        if (!viewer) {
            throw new Error("viewer-adapter.attach: viewer required");
        }
        const store = (root.molbuilder
                       && root.molbuilder.selection
                       && root.molbuilder.selection.store);
        if (!store) {
            throw new Error(
                "viewer-adapter.attach: store missing; "
                + "load lib/selection/store.js first"
            );
        }

        // ----- click wiring --------------------------------------- //
        //
        // ``setClickable`` is called on every render rather than once
        // at attach time -- 3Dmol's click registration is per-model,
        // and ``loadStructureText`` swaps models on file change.  Re-arming
        // each tick guarantees the handler survives the swap.
        //
        // Clicks edit ``state.selection`` directly (the shared set
        // since 2026-05-20) regardless of UI mode.  Switching to
        // filter mode hides the atom-list editor but the 3D viewer
        // stays visible -- a click there is still a meaningful
        // user action on the shared selection.

        function armClick() {
            try {
                viewer.setClickable({}, true, (atom) => {
                    if (!atom || typeof atom.serial !== "number") return;
                    store.toggleAtom(atom.serial);
                });
            } catch (e) {
                if (root.console) root.console.warn(
                    "[viewer-adapter] setClickable failed", e
                );
            }
        }

        // ----- render ---------------------------------------------- //
        //
        // 3Dmol's ``addStyle({sphere: ...})`` REPLACES the existing
        // sphere style on the selected atoms (not additive at the
        // primitive level), which would hide the page's base atom
        // rendering and could not be undone on deselect.  Use
        // ``addSphere`` (a SHAPE, positioned at the atom's centre)
        // for our overlays instead -- shapes coexist with the
        // atom's primary style and can be removed by reference.
        //
        // Track every shape we add so subsequent renders can drop
        // them before adding fresh ones (otherwise overlays would
        // accumulate forever).

        const overlayShapes = [];

        function _clearOverlays() {
            overlayShapes.forEach((s) => {
                try { viewer.removeShape(s); } catch (e) { /* ignore */ }
            });
            overlayShapes.length = 0;
        }

        function render(s) {
            _clearOverlays();
            try {
                // Build a single serial -> atom map ONCE per render.
                // 3Dmol's selectedAtoms({serial:N}) linear-scans the
                // structure; calling it from a per-overlay lookup
                // turns each render into O(N * overlays), which is
                // O(N^2) when most atoms are tinted or selected.
                // Pulling the full atom list once and indexing it
                // gives us O(N) total for the lookups.
                const atomMap = new Map();
                try {
                    const all = viewer.selectedAtoms({}) || [];
                    for (let i = 0; i < all.length; i++) {
                        const a = all[i];
                        if (typeof a.serial === "number") {
                            atomMap.set(a.serial, a);
                        }
                    }
                } catch (_e) { /* viewer mid-swap; empty map */ }

                function paint(serial, spec) {
                    const a = atomMap.get(serial);
                    if (!a) return;
                    let shape;
                    try {
                        shape = viewer.addSphere(Object.assign(
                            { center: { x: a.x, y: a.y, z: a.z } }, spec
                        ));
                    } catch (e) {
                        if (root.console) root.console.warn(
                            "[viewer-adapter] addSphere failed", e
                        );
                        return;
                    }
                    // 3Dmol returns null on transient mid-swap state;
                    // pushing null would just produce a noisy
                    // removeShape(null) on the next render.
                    if (shape) overlayShapes.push(shape);
                }

                // 1. region tints -- group by first label so each
                //    atom gets exactly one tint.  (An atom can carry
                //    multiple labels; the panel surfaces all of them
                //    in the label column.  The viewer picks the
                //    first for color so users still see the primary
                //    grouping at a glance.)
                //
                // Use a Map (not ``{}``) so a label named ``__proto__``
                // or ``constructor`` doesn't piggyback on
                // Object.prototype: reading ``byRegion["__proto__"]``
                // on a plain object returns Object.prototype (which
                // has no ``push``), so the legacy form crashed the
                // whole render and the try/catch at line 137
                // swallowed it, leaving overlays silently disabled
                // for the rest of the page lifetime.  Map keys are
                // stored separately from any prototype.
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
                    const color = _colorFor(label);
                    indices.forEach((idx) => {
                        paint(idx, {
                            radius:  0.5,
                            color:   color,
                            opacity: 0.35,
                        });
                    });
                });

                // 2. fixed marker -- tiny red ball.
                (s.atoms || []).forEach((a) => {
                    if (!a.isFrozen) return;
                    paint(a.index, FROZEN_MARKER);
                });

                // 3. selection halo -- largest ball + brightest, so
                //    it reads as the "live" set above the region
                //    tints.
                (s.selection || []).forEach((idx) => {
                    paint(idx, HALO_STYLE);
                });

                viewer.render();
            } catch (e) {
                if (root.console) root.console.warn(
                    "[viewer-adapter] render failed", e
                );
            }
        }

        // ----- subscribe ------------------------------------------- //
        //
        // The store's setSourceFile awaits loadStructureText BEFORE firing
        // subscribers, so the viewer's model is already swapped by
        // the time we paint overlays here.  No race, no rerender
        // dance.

        const unsubscribe = store.subscribe((s) => {
            armClick();
            render(s);
        });

        function dispose() {
            try { unsubscribe(); } catch (e) { /* ignore */ }
            // Disarm the click handler so 3Dmol stops routing clicks
            // into the (now-orphaned) ``store.toggleAtom`` closure.
            // Without this, a re-attached adapter would see clicks
            // fire through BOTH closures -- and the original closure
            // keeps the store + the disposed adapter alive in the
            // 3Dmol model.
            try {
                viewer.setClickable({}, false, () => {});
            } catch (e) { /* ignore */ }
            // Drop our overlay shapes so the page's base style
            // returns to view (we don't touch ``setStyle`` so the
            // base style was never altered; clearing shapes is
            // sufficient).
            _clearOverlays();
            try { viewer.render(); } catch (e) { /* ignore */ }
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
