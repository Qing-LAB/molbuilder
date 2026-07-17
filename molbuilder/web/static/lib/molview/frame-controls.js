/* Frame controls bar for MolView (molview-module.md §14.5) -- the trajectory PLAYBACK UI.
 *
 * MolView renders this itself (like the isolate view-toggle), so a consumer that
 * hands MolView a trajectory gets the navigation UI for free.  A compact bar:
 *   ‹ play/pause › · loop · speed(ms) · ──slider── · i / N .
 * Speed feeds api.play({fps}); loop feeds api.setLoop.
 *
 * Overlays (force arrows, atom-index labels) are NOT here: they are the CONSUMER's overlay
 * data, pushed via handle.setArrows / setLabels -- not a viewer toggle (molview-module §14.5.1).
 *
 * SHOWN only for a trajectory (frameCount > 1); a single static structure has no bar.  It
 * subscribes to the store so the slider + counter track the current frame (incl. during
 * playback) and the bar appears/disappears as the frame count changes.
 *
 *   molview.mountFrameControls(hostEl, api, store) -> { refresh, dispose }
 *     api = { setFrame(i), frameCount(), currentFrame(), play(), pause(), isPlaying() }
 */
(function (root) {
    "use strict";

    function mountFrameControls(hostEl, api, store) {
        if (!hostEl || !api) return { refresh: function () {}, dispose: function () {} };

        // Static template (no interpolation) -- one template literal so the XSS audit
        // (test_xss_audit.py) sees a safe single-literal RHS, not a concatenation.
        hostEl.innerHTML =
            `<span class="mvf-transport"><button type="button" class="mvf-step mvf-prev" title="Previous frame" aria-label="Previous frame">‹</button><button type="button" class="mvf-play" title="Play / pause." aria-label="Play / pause">▶</button><button type="button" class="mvf-step mvf-next" title="Next frame" aria-label="Next frame">›</button></span><label class="mvf-loop" title="Loop at the ends (play + step wrap around)." aria-label="Loop playback"><input type="checkbox" class="mvf-loop-cb"><span aria-hidden="true">⟳</span></label><label class="mvf-speed" title="Playback speed (ms per frame)."><input type="number" class="mvf-speed-input" min="20" max="3000" step="20" value="150" aria-label="Playback speed (ms per frame)"><span>ms</span></label><input type="range" class="mvf-slider" min="0" step="1" value="0" aria-label="Frame"><span class="mvf-counter" aria-live="polite"></span>`;

        var prevBtn = hostEl.querySelector(".mvf-prev");
        var nextBtn = hostEl.querySelector(".mvf-next");
        var playBtn = hostEl.querySelector(".mvf-play");
        var loopCb  = hostEl.querySelector(".mvf-loop-cb");
        var speedInput = hostEl.querySelector(".mvf-speed-input");
        var slider  = hostEl.querySelector(".mvf-slider");
        var counter = hostEl.querySelector(".mvf-counter");
        var doc     = root.document;

        function _loop() { return (typeof api.getLoop === "function") ? api.getLoop() : true; }
        // Playback speed as ms-per-frame -> fps for api.play({fps}); clamped so a stray
        // 0/blank can't divide-by-zero into a runaway loop.
        function _fps() {
            var ms = speedInput ? (parseInt(speedInput.value, 10) || 150) : 150;
            return 1000 / Math.max(20, ms);
        }
        function _syncPlay() {
            var on = api.isPlaying();
            playBtn.textContent = on ? "❙❙" : "▶";   // ❙❙ (pause) / ▶ (play)
            playBtn.setAttribute("aria-pressed", String(on));
        }
        // Single-step (‹ / ›) -- wraps when 'loop' is on, clamps at the ends when off.  Pauses
        // playback so the step sticks.
        function _step(delta) {
            var n = api.frameCount();
            if (n <= 1) return;
            if (api.isPlaying()) { api.pause(); _syncPlay(); }
            var next = api.currentFrame() + delta;
            next = _loop() ? ((next % n) + n) % n : Math.max(0, Math.min(n - 1, next));
            api.setFrame(next);
        }

        slider.addEventListener("input", function (e) { api.setFrame(Number(e.target.value) || 0); });
        prevBtn.addEventListener("click", function () { _step(-1); });
        nextBtn.addEventListener("click", function () { _step(1); });
        if (loopCb) {
            loopCb.checked = _loop();
            loopCb.addEventListener("change", function (e) {
                if (typeof api.setLoop === "function") api.setLoop(!!e.target.checked);
            });
        }
        playBtn.addEventListener("click", function () {
            if (api.isPlaying()) api.pause(); else api.play({ fps: _fps() });
            _syncPlay();
        });
        // Changing speed mid-playback restarts the timer at the new rate.
        if (speedInput) {
            speedInput.addEventListener("change", function () {
                if (api.isPlaying()) { api.pause(); api.play({ fps: _fps() }); _syncPlay(); }
            });
        }

        function reflect() {
            var n = api.frameCount();
            // A single (or no) structure has no trajectory -> hide the whole bar.
            hostEl.hidden = !(n > 1);
            if (n <= 1) return;
            var cur = api.currentFrame();
            slider.max = String(n - 1);
            if (!doc || doc.activeElement !== slider) slider.value = String(cur);
            counter.textContent = (cur + 1) + " / " + n;
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
