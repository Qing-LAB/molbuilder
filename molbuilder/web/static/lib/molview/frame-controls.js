/* Frame controls bar for MolView (molview-module.md §14.5) -- the trajectory playback UI.
 *
 * MolView renders this itself (like it renders the isolate / k-grid view-toggles), so a
 * consumer that hands MolView a trajectory gets the whole navigation UI for free.  A compact
 * bar:  [Forces] [Indices]   play/pause   ──slider──   i / N .
 *
 * It is SHOWN only when there is a trajectory (frameCount > 1); a single static structure has
 * no frame bar.  It subscribes to the store so the slider + counter track the current frame
 * (including during playback) and the bar appears/disappears as the frame count changes.
 *
 *   molview.mountFrameControls(hostEl, api, store) -> { refresh, dispose }
 *     api = { setFrame(i), frameCount(), currentFrame(), play(), pause(), isPlaying(),
 *             setShowForces(on), setShowIndices(on), hasForces() }
 */
(function (root) {
    "use strict";

    function mountFrameControls(hostEl, api, store) {
        if (!hostEl || !api) return { refresh: function () {}, dispose: function () {} };

        hostEl.innerHTML =
            '<label class="mvf-toggle" title="Show per-frame force arrows.">'
          + '<input type="checkbox" class="mvf-forces"><span>Forces</span></label>'
          + '<label class="mvf-toggle" title="Show atom-index labels.">'
          + '<input type="checkbox" class="mvf-indices"><span>Indices</span></label>'
          + '<button type="button" class="mvf-play" title="Play / pause the trajectory."'
          + ' aria-label="Play / pause">&#9654;</button>'
          + '<input type="range" class="mvf-slider" min="0" step="1" value="0"'
          + ' aria-label="Frame">'
          + '<span class="mvf-counter" aria-live="polite"></span>';

        var forcesCb  = hostEl.querySelector(".mvf-forces");
        var indicesCb = hostEl.querySelector(".mvf-indices");
        var playBtn   = hostEl.querySelector(".mvf-play");
        var slider    = hostEl.querySelector(".mvf-slider");
        var counter   = hostEl.querySelector(".mvf-counter");
        var doc       = root.document;

        function _syncPlay() {
            var on = api.isPlaying();
            playBtn.innerHTML = on ? "&#10073;&#10073;" : "&#9654;";   // ⏸ / ▶
            playBtn.setAttribute("aria-pressed", String(on));
        }

        forcesCb.addEventListener("change", function (e) { api.setShowForces(!!e.target.checked); });
        indicesCb.addEventListener("change", function (e) { api.setShowIndices(!!e.target.checked); });
        slider.addEventListener("input", function (e) { api.setFrame(Number(e.target.value) || 0); });
        playBtn.addEventListener("click", function () {
            if (api.isPlaying()) api.pause(); else api.play();
            _syncPlay();
        });

        function reflect() {
            var n = api.frameCount();
            // A single (or no) structure has no trajectory -> hide the whole bar.
            hostEl.hidden = !(n > 1);
            if (n <= 1) return;
            var cur = api.currentFrame();
            slider.max = String(n - 1);
            if (!doc || doc.activeElement !== slider) slider.value = String(cur);
            counter.textContent = (cur + 1) + " / " + n;
            if (typeof api.hasForces === "function") forcesCb.disabled = !api.hasForces();
            _syncPlay();
        }

        var unsub = (store && typeof store.subscribe === "function")
            ? store.subscribe(reflect) : function () {};
        reflect();

        return {
            refresh: reflect,
            dispose: function () {
                try { unsub(); } catch (_) {}
                try { hostEl.innerHTML = ""; } catch (_) {}
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mountFrameControls = mountFrameControls;
})(typeof window !== "undefined" ? window : this);
