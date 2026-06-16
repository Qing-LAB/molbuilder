/* System-load monitor -- global 1 Hz strip at the bottom of every page.
 *
 * Polls /api/system/load (psutil + nvml under the hood, see
 * web/blueprints/system_load.py) and renders four sparkline canvases:
 * CPU %, RAM %, GPU util %, GPU mem %.  Samples held in 60-element
 * ring buffers (60 s of history at 1 Hz).
 *
 * Behaviour:
 *   * Mounts at DOMContentLoaded against #system-load-monitor (HTML
 *     lives in _app_header.html so every page gets the widget).
 *   * Pauses polling when document.hidden (browser-tab-backgrounded);
 *     resumes on visibilitychange.
 *   * Hides GPU + VRAM cells when the API returns gpus=[] (CPU-only
 *     host).  The widget collapses to 2 cells and the toggle is the
 *     same -- no separate code path.
 *   * Collapse toggle persists in sessionStorage so the user's choice
 *     survives intra-session navigation between tabs.
 *
 * No external dependencies; plain DOM + canvas.  Plotly would be
 * overkill for 120x32 sparklines redrawn at 1 Hz.
 */
(function() {
    "use strict";

    var ENDPOINT = "/api/system/load";
    var POLL_MS  = 1000;       // 1 Hz
    var BUFFER_N = 60;         // 60 samples = 60 s of history
    var SPARK_W  = 120;
    var SPARK_H  = 32;

    var STORAGE_KEY_COLLAPSED = "mb.system-load.collapsed";

    function $$(sel, root) { return (root || document).querySelector(sel); }

    /* Ring buffer of length N: push appends, drops the oldest when full.
     * Indexed in temporal order (oldest first) by ``items()``. */
    function ringBuffer(n) {
        var buf = [];
        return {
            push: function(v) {
                buf.push(v);
                if (buf.length > n) buf.shift();
            },
            items: function() { return buf; },
            length: function() { return buf.length; },
        };
    }

    /* Draw a single sparkline.  Values are 0-100; we draw a filled area
     * + a polyline, color-coded by current value (green < 50, amber <
     * 80, red >= 80).  Empty buffer = no draw (the "(starting)" text
     * label below the canvas handles the empty state). */
    function drawSparkline(canvas, values) {
        var ctx = canvas.getContext("2d");
        var w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        if (!values || values.length < 2) return;
        var cs = getComputedStyle(canvas);
        var lastV = values[values.length - 1];
        var stroke = (lastV >= 80) ? cs.getPropertyValue("--load-bad").trim()
                   : (lastV >= 50) ? cs.getPropertyValue("--load-warn").trim()
                                   : cs.getPropertyValue("--load-ok").trim();
        if (!stroke) stroke = "#3a9";   // safe fallback if tokens absent
        // X-axis: each sample takes (w / (BUFFER_N - 1)) px so the
        // sparkline fills the canvas at full history; partial history
        // draws at the right edge growing leftward as samples arrive.
        var dx = w / (BUFFER_N - 1);
        var n = values.length;
        var x0 = w - (n - 1) * dx;
        ctx.beginPath();
        for (var i = 0; i < n; i++) {
            var x = x0 + i * dx;
            // Y inverted; clamp to [0, 100].
            var v = Math.max(0, Math.min(100, values[i]));
            var y = h - (v / 100) * (h - 2) - 1;  // 1-px margin top/bottom
            if (i === 0) ctx.moveTo(x, y);
            else         ctx.lineTo(x, y);
        }
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        // Filled area under the line, ~20% opacity for context.
        ctx.lineTo(x0 + (n - 1) * dx, h);
        ctx.lineTo(x0, h);
        ctx.closePath();
        ctx.fillStyle = stroke;
        ctx.globalAlpha = 0.18;
        ctx.fill();
        ctx.globalAlpha = 1.0;
    }

    function mount() {
        var root = $$("#system-load-monitor");
        if (!root) return;  // page didn't include the widget HTML; bail

        var cellCpu  = $$("[data-metric='cpu']",  root);
        var cellRam  = $$("[data-metric='ram']",  root);
        var cellGpu  = $$("[data-metric='gpu']",  root);
        var cellVram = $$("[data-metric='vram']", root);
        var toggle   = $$(".system-load-toggle", root);

        var bufCpu  = ringBuffer(BUFFER_N);
        var bufRam  = ringBuffer(BUFFER_N);
        var bufGpu  = ringBuffer(BUFFER_N);
        var bufVram = ringBuffer(BUFFER_N);

        function updateCell(cell, buf, value, valueText) {
            if (!cell) return;
            var vEl = $$("[data-value]", cell);
            var cEl = $$("canvas.spark", cell);
            if (vEl) vEl.textContent = valueText;
            if (cEl) drawSparkline(cEl, buf.items());
        }

        var aborter = null;
        var lastFetchOk = true;

        function poll() {
            if (document.hidden) return;  // skip while tab is backgrounded
            if (aborter) {
                try { aborter.abort(); } catch (_) { /* ignore */ }
            }
            aborter = new AbortController();
            fetch(ENDPOINT, {
                credentials: "same-origin",
                signal:      aborter.signal,
                cache:       "no-store",
            })
                .then(function(r) { return r.ok ? r.json() : null; })
                .then(function(body) {
                    if (!body || body.ok !== true) {
                        lastFetchOk = false;
                        return;
                    }
                    lastFetchOk = true;
                    var d = body.data || {};
                    // CPU + RAM are always present (psutil never fails on
                    // a healthy box; if it did, the server returns ok=false
                    // and we hit the branch above).
                    var cpu = d.cpu_pct || 0;
                    var ram = d.ram_pct || 0;
                    bufCpu.push(cpu);
                    bufRam.push(ram);
                    updateCell(cellCpu, bufCpu, cpu,
                               cpu.toFixed(0) + "%");
                    updateCell(cellRam, bufRam, ram,
                               ram.toFixed(0) + "%  "
                               + (d.ram_used_gb || 0).toFixed(1) + "/"
                               + (d.ram_total_gb || 0).toFixed(1) + " GB");
                    // GPU: server returns [] when NVML init failed (CPU-
                    // only host or driver missing).  Hide both GPU cells
                    // entirely; widget collapses to 2 cells.
                    var gpus = d.gpus || [];
                    if (gpus.length === 0) {
                        cellGpu.hidden  = true;
                        cellVram.hidden = true;
                        return;
                    }
                    cellGpu.hidden  = false;
                    cellVram.hidden = false;
                    // Multi-GPU: report GPU 0 in the sparklines (the
                    // common case) and put the per-device breakdown in
                    // the cell title (hover tooltip).
                    var g0 = gpus[0];
                    var gUtil = (typeof g0.util_pct === "number") ? g0.util_pct : 0;
                    var gMem  = (typeof g0.mem_pct  === "number") ? g0.mem_pct  : 0;
                    bufGpu.push(gUtil);
                    bufVram.push(gMem);
                    updateCell(cellGpu, bufGpu, gUtil,
                               gUtil.toFixed(0) + "%");
                    updateCell(cellVram, bufVram, gMem,
                               gMem.toFixed(0) + "%  "
                               + ((g0.mem_used_mb || 0) / 1024).toFixed(1) + "/"
                               + ((g0.mem_total_mb || 0) / 1024).toFixed(1) + " GB");
                    // Tooltip: per-GPU name + util + mem.  Cheap derivation;
                    // text-only so no XSS surface.
                    var tip = gpus.map(function(g, i) {
                        return "GPU " + i + " " + (g.name || "?")
                            + ": util " + (g.util_pct || 0) + "%, mem "
                            + ((g.mem_used_mb || 0) / 1024).toFixed(1) + "/"
                            + ((g.mem_total_mb || 0) / 1024).toFixed(1) + " GB";
                    }).join("\n");
                    cellGpu.title  = tip;
                    cellVram.title = tip;
                })
                .catch(function(err) {
                    if (err && err.name === "AbortError") return;
                    lastFetchOk = false;
                });
        }

        var timer = null;
        function startTimer() {
            if (timer !== null) return;
            poll();  // immediate first sample so the user doesn't wait 1 s
            timer = setInterval(poll, POLL_MS);
        }
        function stopTimer() {
            if (timer === null) return;
            clearInterval(timer);
            timer = null;
        }
        document.addEventListener("visibilitychange", function() {
            if (document.hidden) stopTimer();
            else                 startTimer();
        });

        // Collapse toggle.  Persisted in sessionStorage so navigation
        // between tabs preserves the user's choice; cleared on
        // browser close (this is a transient UI preference, not a
        // user setting).
        function applyCollapsed(collapsed) {
            if (collapsed) {
                root.classList.add("is-collapsed");
                toggle.setAttribute("aria-pressed", "true");
                toggle.title = "Show server load";
            } else {
                root.classList.remove("is-collapsed");
                toggle.setAttribute("aria-pressed", "false");
                toggle.title = "Hide server load";
            }
        }
        if (toggle) {
            toggle.addEventListener("click", function() {
                var next = !root.classList.contains("is-collapsed");
                try { sessionStorage.setItem(STORAGE_KEY_COLLAPSED,
                                             next ? "1" : "0"); }
                catch (_) { /* private mode; ignore */ }
                applyCollapsed(next);
            });
            var saved = "0";
            try { saved = sessionStorage.getItem(STORAGE_KEY_COLLAPSED) || "0"; }
            catch (_) { /* ignore */ }
            applyCollapsed(saved === "1");
        }

        startTimer();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mount);
    } else {
        mount();
    }
})();
