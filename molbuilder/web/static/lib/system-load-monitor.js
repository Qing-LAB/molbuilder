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
    // 2026-06-21 user feedback: 60 s of history was too short to see
    // SCF-cycle rhythm or warm/cool spikes on multi-minute jobs.
    // 10 min @ 1 Hz = 600 samples, ~5 KB per metric.  Memory cost:
    // negligible.  Sparkline still readable at 0.5 px/sample on a
    // 320 px canvas (compresses ~25 s of detail into one pixel band;
    // peaks remain visible as 1-2 px spikes).
    var BUFFER_N = 600;        // 600 samples = 10 min of history
    var SPARK_W  = 320;        // wider canvas: panel has room now
    var SPARK_H  = 48;

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

        var cellCpu   = $$("[data-metric='cpu']",   root);
        var cellRam   = $$("[data-metric='ram']",   root);
        var cellGpu   = $$("[data-metric='gpu']",   root);
        var cellGpuBw = $$("[data-metric='gpubw']", root);
        var cellVram  = $$("[data-metric='vram']",  root);
        var toggle    = $$(".system-load-toggle", root);

        var bufCpu   = ringBuffer(BUFFER_N);
        var bufRam   = ringBuffer(BUFFER_N);
        var bufGpu   = ringBuffer(BUFFER_N);
        var bufGpuBw = ringBuffer(BUFFER_N);
        var bufVram  = ringBuffer(BUFFER_N);

        function updateCell(cell, buf, value, valueText) {
            if (!cell) return;
            var vEl = $$("[data-value]", cell);
            var cEl = $$("canvas.spark", cell);
            if (vEl) vEl.textContent = valueText;
            if (cEl) drawSparkline(cEl, buf.items());
        }

        /* 2026-06-21 user feedback: detail text used to live in browser
         * ``cell.title`` tooltips that re-positioned on every 1 Hz
         * redraw -- never stable, never readable.  Write it directly
         * into the cell's ``[data-detail]`` slot instead so it's
         * always on-screen.  The strings preserve their pre-formatted
         * ``\n``-separated layout; the CSS uses ``white-space:
         * pre-line`` to render them as multi-line. */
        function setDetail(cell, text) {
            if (!cell) return;
            var dEl = $$("[data-detail]", cell);
            if (dEl) dEl.textContent = text;
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
                    // Absolute core-equivalents: ``cpu_pct`` reports
                    // the percentage across ALL logical CPUs, so
                    // ``cpu_pct * cpu_count_logical / 100`` is the
                    // number of logical-thread-equivalents currently
                    // busy.  Divide by 2 if the system has SMT to
                    // approximate physical-core-equivalents -- or
                    // simply quote against ``cpu_count_physical``.
                    // We show "(~N/M cores)" where M is physical
                    // core count, computing N as
                    // ``cpu_pct * cpu_count_physical / 100`` (so
                    // 50% on a 20-phys / 40-logical box reads as
                    // "~10/20 cores" -- matches what a SIESTA job
                    // with mpi_np * omp = 20 actually consumes).
                    var nPhys = d.cpu_count_physical || 0;
                    var nLog  = d.cpu_count_logical  || 0;
                    var coresBusy = (nPhys > 0) ? cpu * nPhys / 100 : 0;
                    var cpuText = cpu.toFixed(0) + "%";
                    if (nPhys > 0) {
                        cpuText += "  ~" + coresBusy.toFixed(1) + "/"
                                 + nPhys + " cores";
                    }
                    updateCell(cellCpu, bufCpu, cpu, cpuText);
                    // CPU detail: spell out the logical vs physical
                    // distinction + the load-avg trio so the user can
                    // tell when the run queue is over-subscribed
                    // (load > cpu_count) even if cpu_pct is at 100.
                    var cpuDetail = nPhys + " physical cores"
                        + (nLog && nLog !== nPhys
                           ? " (" + nLog + " logical with SMT)" : "")
                        + "\n~" + coresBusy.toFixed(1)
                        + " physical-core-equivalents busy";
                    if (typeof d.loadavg_1m === "number") {
                        cpuDetail += "\nload avg: " + d.loadavg_1m.toFixed(2)
                                + " (1m), "    + d.loadavg_5m.toFixed(2)
                                + " (5m), "    + d.loadavg_15m.toFixed(2)
                                + " (15m)";
                        if (d.loadavg_1m > nPhys) {
                            cpuDetail += "\n[over-subscribed: load > physical cores]";
                        }
                    }
                    // Per-socket breakdown -- empty list on single-
                    // socket / lscpu-less hosts.  When present, this
                    // is the load-bearing diagnostic for the SIESTA-
                    // GPU NUMA-pin case: one socket fully busy +
                    // other socket idle means "GPU socket saturated,
                    // NUMA pin healthy".  Both half-busy means
                    // "ranks spread, paying UPI penalty".
                    var perSock = d.per_socket_pct;
                    if (Array.isArray(perSock) && perSock.length >= 2) {
                        cpuDetail += "\nper socket:";
                        var maxPct = 0, minPct = 100;
                        perSock.forEach(function(s) {
                            var p = (typeof s.pct === "number") ? s.pct : 0;
                            cpuDetail += "\n  socket " + s.socket
                                + ": "     + p.toFixed(1) + "%"
                                + " ("     + s.cpu_count + " logical CPUs)";
                            if (p > maxPct) maxPct = p;
                            if (p < minPct) minPct = p;
                        });
                        // Asymmetric load (one socket >70, another
                        // <20) is the NUMA-pin signature.
                        if (maxPct - minPct > 50 && maxPct > 70) {
                            cpuDetail += "\n[asymmetric: likely NUMA-pinned"
                                    + " to one socket]";
                        }
                    }
                    setDetail(cellCpu, cpuDetail);
                    updateCell(cellRam, bufRam, ram,
                               ram.toFixed(0) + "%  "
                               + (d.ram_used_gb || 0).toFixed(1) + "/"
                               + (d.ram_total_gb || 0).toFixed(1) + " GB");
                    setDetail(cellRam,
                        (d.ram_used_gb || 0).toFixed(2) + " GB used of "
                        + (d.ram_total_gb || 0).toFixed(2) + " GB total");
                    // GPU: server returns [] when NVML init failed (CPU-
                    // only host or driver missing).  Hide all three GPU
                    // cells entirely; widget collapses to 2 cells.
                    var gpus = d.gpus || [];
                    if (gpus.length === 0) {
                        cellGpu.hidden   = true;
                        cellGpuBw.hidden = true;
                        cellVram.hidden  = true;
                        return;
                    }
                    cellGpu.hidden   = false;
                    cellGpuBw.hidden = false;
                    cellVram.hidden  = false;
                    // Multi-GPU: report GPU 0 in the sparklines (the
                    // common case) and put the per-device breakdown in
                    // the cell title (hover tooltip).
                    var g0 = gpus[0];
                    var gUtil = (typeof g0.util_pct      === "number") ? g0.util_pct     : 0;
                    var gBw   = (typeof g0.util_mem_pct  === "number") ? g0.util_mem_pct : 0;
                    var gMem  = (typeof g0.mem_pct       === "number") ? g0.mem_pct      : 0;
                    bufGpu.push(gUtil);
                    bufGpuBw.push(gBw);
                    bufVram.push(gMem);
                    updateCell(cellGpu,   bufGpu,   gUtil, gUtil.toFixed(0) + "%");
                    updateCell(cellGpuBw, bufGpuBw, gBw,   gBw.toFixed(0)   + "%");
                    updateCell(cellVram, bufVram, gMem,
                               gMem.toFixed(0) + "%  "
                               + ((g0.mem_used_mb || 0) / 1024).toFixed(1) + "/"
                               + ((g0.mem_total_mb || 0) / 1024).toFixed(1) + " GB");
                    // Details: shared text-only summary across all 3
                    // GPU cells.  Surfaces power / temp / clocks
                    // which are NOT on the strip but matter for
                    // diagnosis: drops in power_w mid-run = thermal /
                    // power throttle; sm_clock_mhz drop at sustained
                    // 100% util = clock throttle (often the underlying
                    // cause).  All fields are best-effort -- show "—"
                    // when NVML didn't return them on this chip.
                    function _fmt(v, suffix) {
                        return (typeof v === "number") ? (v + suffix) : "—";
                    }
                    var gpuListDetail = gpus.map(function(g, i) {
                        var line = "GPU " + i + " " + (g.name || "?")
                            + "\n  SM compute   : " + _fmt(g.util_pct, "%")
                            + "\n  Memory BW    : " + _fmt(g.util_mem_pct, "%")
                            + "\n  VRAM         : "
                              + ((g.mem_used_mb || 0) / 1024).toFixed(1) + "/"
                              + ((g.mem_total_mb || 0) / 1024).toFixed(1) + " GB"
                              + "  (" + _fmt(g.mem_pct, "%") + ")"
                            + "\n  Power        : " + _fmt(g.power_w, " W")
                            + "\n  Temperature  : " + _fmt(g.temp_c, " °C")
                            + "\n  SM clock     : " + _fmt(g.sm_clock_mhz, " MHz")
                            + "\n  Memory clock : " + _fmt(g.mem_clock_mhz, " MHz");
                        return line;
                    }).join("\n");
                    // Per-cell detail: a short context line per metric
                    // followed by the shared per-device breakdown.
                    setDetail(cellGpu,   "SM busy %\n" + gpuListDetail);
                    setDetail(cellGpuBw,
                        "high here + low GPU = memory-bandwidth bound\n"
                        + "(more ranks won't help; smaller BlockSize might)\n\n"
                        + gpuListDetail);
                    setDetail(cellVram,  "VRAM occupancy\n" + gpuListDetail);
                })
                .catch(function(err) {
                    if (err && err.name === "AbortError") return;
                    lastFetchOk = false;
                });
        }

        // Polling-active iff (a) document is visible AND (b) user has
        // NOT collapsed the widget.  Both conditions are flipped by
        // independent event sources (visibilitychange + toggle click)
        // so we OR-stop / AND-start instead of letting either source
        // unconditionally call startTimer() (which would resume polling
        // even when the user has explicitly collapsed -- the original
        // bug this commit closes).
        var timer       = null;
        var userClosed  = false;   // user clicked the collapse toggle
        function startTimer() {
            if (timer !== null) return;
            if (userClosed)      return;     // collapsed -> never poll
            if (document.hidden) return;     // tab backgrounded -> wait
            poll();  // immediate first sample so the user doesn't wait 1 s
            timer = setInterval(poll, POLL_MS);
        }
        function stopTimer() {
            if (timer === null) return;
            clearInterval(timer);
            timer = null;
            // Cancel any in-flight fetch so we don't render a stale
            // sample on resume.  Idempotent if aborter is null.
            if (aborter) {
                try { aborter.abort(); } catch (_) { /* ignore */ }
                aborter = null;
            }
        }
        document.addEventListener("visibilitychange", function() {
            if (document.hidden) stopTimer();
            else                 startTimer();
        });

        // Collapse toggle.  Persisted in sessionStorage so navigation
        // between tabs preserves the user's choice; cleared on
        // browser close (this is a transient UI preference, not a
        // user setting).
        //
        // Collapsing now ALSO stops polling (and aborts any in-flight
        // request) -- with the widget hidden the snapshots are wasted
        // server work + wasted client bandwidth.  Expanding restarts
        // polling AND triggers an immediate first sample so the
        // sparklines re-populate without a 1 s wait.
        // PR 5 (2026-06-17 user-report-2): --monitor-height CSS var
        // retired -- the monitor moved INSIDE .results-main as a
        // normal flow element, so scroll containers no longer need
        // to reserve space.  See system-load-monitor.css for the
        // in-flow layout.

        function applyCollapsed(collapsed) {
            userClosed = !!collapsed;
            if (collapsed) {
                root.classList.add("is-collapsed");
                toggle.setAttribute("aria-pressed", "true");
                toggle.title = "Show server load";
                stopTimer();
            } else {
                root.classList.remove("is-collapsed");
                toggle.setAttribute("aria-pressed", "false");
                toggle.title = "Hide server load";
                startTimer();
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
            // results-state-contract.md § 9: COLLAPSED BY DEFAULT
            // on first visit.  Users opt in to the expanded strip;
            // it doesn't opt them in.  The expanded strip otherwise
            // overlays the bottom 48px of plots on every fresh
            // /results visit -- the bug that prompted the layout
            // contract.  ``saved === "0"`` is explicit-opt-in to
            // expanded; missing key OR explicit "1" -> collapsed.
            var saved = null;
            try { saved = sessionStorage.getItem(STORAGE_KEY_COLLAPSED); }
            catch (_) { /* ignore */ }
            var startCollapsed = (saved === null) ? true : (saved !== "0");
            applyCollapsed(startCollapsed);
        } else {
            // No toggle button mounted -- default to polling.
            startTimer();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mount);
    } else {
        mount();
    }
})();
