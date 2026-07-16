/* Viewer view-controls -- the DISPLAY toggle that sits in the viewer's control bar
 * (beside "Focus molecule"): "Show selected only" (isolate).
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
        // Static template (no interpolation) -- one template literal so the XSS audit
        // (test_xss_audit.py) sees a safe single-literal RHS, not a concatenation.
        hostEl.innerHTML =
            `<label class="viewer-toggle" title="Hide unselected atoms so the current selection stands out."><input type="checkbox" class="vc-isolate"><span>Show selected only</span></label>`;
        var iso = hostEl.querySelector(".vc-isolate");

        iso.addEventListener("change", function (e) {
            store.setIsolate(!!e.target.checked);
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
