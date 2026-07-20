/* Atom-selection viewer-adapter -- declarative click-to-select
 * consumer of the selection store on top of the embedded MolViewer
 * (#229 Part B migration, 2026-06-03).
 *
 * Picking + the isolate view-toggle spec only.  The render engine
 * (engine/process.js) owns ALL halo/overlay derivation (region tints
 * / frozen markers / selection halo) per
 * docs/protocols/molview-render-streamline.md § 2.4/§ 7.3 -- the
 * adapter no longer paints.
 *
 * Click handling: routes through the embed's PickOpts.onPick so
 * clicks reach the store via store.toggle.  The store stays the
 * single source of truth for the selection.
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

    // isolate ("show selected only") is a view TOGGLE whose on/off value lives in
    // the selection store (the viewer needs to know WHICH atoms are selected to hide
    // the rest).  This factory returns the toggle SPEC the embed renders through its
    // ONE view-toggle path (handle.addViewToggle): a canvas quick button + a View-menu
    // entry, byte-identical to the built-in axes/labels/overlay/cell toggles.  The
    // spec's read/run/subscribe close over the store; the "auto-clear isolate when the
    // selection empties" rule lives in the store (a selection-state rule), not here.
    function isolateToggle(store) {
        return {
            action:    "isolate",
            glyph:     "◉",
            label:     "Show selected only",
            title:     "Hide unselected atoms so the current selection stands out.",
            read:      function () {
                const s = (store && store.getState && store.getState()) || {};
                return !!s.isolate;
            },
            run:       function () {
                const s = (store && store.getState && store.getState()) || {};
                if (store && typeof store.setIsolate === "function") {
                    store.setIsolate(!s.isolate);
                }
            },
            subscribe: function (cb) {
                return (store && typeof store.subscribe === "function")
                    ? store.subscribe(cb) : function () {};
            },
        };
    }

    function attach(handle, opts) {
        if (!handle || typeof handle.setOverlays !== "function") {
            throw new Error(
                "viewer-adapter.attach: expected an embed handle "
              + "(window.molbuilder.viewer.embed return value), got "
              + (handle && typeof handle));
        }
        // The selection store lives on molview.data (molview-module.md §12) --
        // resolve it DIRECTLY, not gated on the workspace persistence layer (a
        // page can mount a viewer without the dispatcher; gating on `workspace`
        // wrongly short-circuited to null there).  A caller MAY pass its own
        // store (``opts.store``) so a readonly/ephemeral inspector paints from an
        // ISOLATED selection; otherwise it's the molview.data singleton (Modify).
        const store = (opts && opts.store)
            || (root.molbuilder && root.molbuilder.molview
                && root.molbuilder.molview.data
                && root.molbuilder.molview.data.selection);
        if (!store) {
            throw new Error(
                "viewer-adapter.attach: no selection store; pass opts.store or "
                + "load lib/molview/data-model.js (molview.data.selection)"
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
        // embed runs in "single" pick mode with halos/labels OFF (the render
        // engine paints every halo), so the embed's own pick buffer is
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
            // shows a DERIVED list -- isolate (only the selected atoms are drawn): a click
            // then has no unambiguous unit-cell atom, so drop it.  The selection panel
            // (unit-cell atom list) still edits the selection.
            const st = store.getState() || {};
            const isolating = !!st.isolate
                && (Array.isArray(st.indices) ? st.indices.length : 0) > 0;
            if (isolating) { prevClicked = null; return; }
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

        function dispose() {
            // Detach the click handler so 3Dmol stops routing
            // clicks into the orphaned store reference.  Setting
            // mode: "none" disables pick entirely.
            try {
                handle.setPick({ mode: "none" });
            } catch (e) { /* ignore */ }
        }

        return {
            dispose:         dispose,
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.selection = root.molbuilder.molview.selection || {};
    root.molbuilder.molview.selection.viewerAdapter = {
        attach:         attach,
        isolateToggle:  isolateToggle,   // spec for the embed's addViewToggle
    };
    // Module-init contract: register with runtime (design.md).
    if (root.molbuilder.runtime
        && typeof root.molbuilder.runtime.register === "function") {
        root.molbuilder.runtime.register(
            "selection.viewerAdapter",
            root.molbuilder.molview.selection.viewerAdapter);
    }
})(typeof window !== "undefined" ? window : globalThis);
