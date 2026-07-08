/* Viewer view-controls -- the DISPLAY toggles that sit in the viewer's control bar
 * (beside "Focus molecule"): "Show selected only" (isolate) + "Show k-grid" (tiling).
 *
 * These are VIEW state, not selection state, and they are PER-VIEWER: each mount wires to
 * ITS store (Modify's workspace selection, or a Results card's isolated store), so two
 * cards never share a toggle.  The toggles used to live inside the shared selection panel;
 * they belong by the viewer they affect, in EVERY molview (Modify + Results).
 *
 *   mountViewControls(hostEl, store) -> { dispose }
 */
(function (root) {
    "use strict";

    function mountViewControls(hostEl, store) {
        if (!hostEl || !store || typeof store.subscribe !== "function") {
            return { dispose: function () {} };
        }
        hostEl.innerHTML =
            '<label class="viewer-toggle"'
          + ' title="Hide unselected atoms so the current selection stands out.">'
          + '<input type="checkbox" class="vc-isolate">'
          + '<span>Show selected only</span></label>'
          + '<label class="viewer-toggle"'
          + ' title="Tile the unit cell by the k-grid to check the periodic model.">'
          + '<input type="checkbox" class="vc-kgrid">'
          + '<span>Show k-grid</span></label>';
        var iso = hostEl.querySelector(".vc-isolate");
        var kg  = hostEl.querySelector(".vc-kgrid");

        iso.addEventListener("change", function (e) {
            store.setIsolate(!!e.target.checked);
        });
        kg.addEventListener("change", function (e) {
            store.setKgrid({ enabled: !!e.target.checked });
        });

        // Reflect the store; auto-clear isolate ONLY on the empty TRANSITION (a non-empty
        // selection just became empty while isolating) -- not on plain emptiness, so
        // "check the box, then select" still works (matches the old panel behaviour).
        var prevSelCount = 0;
        function reflect(s) {
            s = s || {};
            var n = (s.indices || []).length;
            if (n === 0 && prevSelCount > 0 && s.isolate) {
                store.setIsolate(false);   // re-notifies; the box syncs next tick
            } else if (document.activeElement !== iso) {
                iso.checked = !!s.isolate;
            }
            if (document.activeElement !== kg) {
                kg.checked = !!(s.kgrid && s.kgrid.enabled);
            }
            prevSelCount = n;
        }
        var unsub = store.subscribe(reflect);
        reflect(store.getState());   // initial

        return {
            dispose: function () { try { unsub(); } catch (_) { /* gone */ } },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mountViewControls = mountViewControls;
})(typeof window !== "undefined" ? window : this);
