/* System-load monitor -- the 1 Hz "Server load" card on the Results tab.
 *
 * User-facing contract: docs/web/results.md § 6.
 *
 * Polls /api/system/load (psutil + nvml under the hood, see
 * web/blueprints/system_load.py) and renders five sparkline canvases:
 * CPU %, RAM %, GPU SM %, GPU BW %, VRAM %.  Samples held in 600-element
 * ring buffers (10 min of history at 1 Hz).
 *
 * Behaviour:
 *   * Mounts at DOMContentLoaded against #system-load-monitor.  The HTML
 *     is templates/_system_load_monitor.html, included by results.html
 *     ALONE -- mounting it app-wide charged every page a 1 Hz hardware
 *     probe nobody was reading.
 *   * Starts COLLAPSED on a first visit; the choice persists in
 *     sessionStorage so it survives navigation between tabs.
 *   * Pauses polling when document.hidden (browser-tab-backgrounded);
 *     resumes on visibilitychange.  Collapsing stops it outright.
 *   * Hides GPU + VRAM cells when the API returns gpus=[] (CPU-only
 *     host).  The widget collapses to 2 cells and the toggle is the
 *     same -- no separate code path.
 *   * Says why when the API also returns a ``gpu_error`` -- the cells
 *     are missing because the host's driver is unreachable, not because
 *     there is no GPU.  In TWO places, for two different readers: a
 *     notice under the strip, and a marker on the header pill for the
 *     (default) case where the card is closed.  See ``applyGpuStatus``.
 *
 * No external dependencies; plain DOM + canvas.  Plotly would be
 * overkill for 320x48 sparklines redrawn at 1 Hz.
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
    // Canvas pixel size is NOT set here: it comes from the width/height
    // attributes on each <canvas> in _system_load_monitor.html (320x48),
    // with CSS scaling the element.  This file only reads canvas.width /
    // canvas.height when drawing.

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
        var notice    = $$("[data-gpu-error]", root);
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

        /* Where a GPU fault gets said, and why it is said twice.
         *
         * The notice under the strip explains it to someone already
         * looking at the card.  The marker on the header pill exists
         * for someone who is NOT: the card is collapsed on first visit
         * (results-state-contract § 9), so the pill is the only part
         * that is always on screen.  Without a mark there, a broken
         * driver is invisible to anyone who never opens the card --
         * which is the failure this whole thing was added to end.
         *
         * ``gpuFault`` is kept because the pill's title is composed
         * from two independent sources (collapsed-or-not, faulted-or-
         * not) that change at different moments. */
        var gpuFault = "";

        function refreshToggleTitle() {
            if (!toggle) return;
            var base = root.classList.contains("is-collapsed")
                     ? "Show server load" : "Hide server load";
            toggle.title = gpuFault
                ? base + " — GPU stats unavailable (" + gpuFault + ")"
                : base;
        }

        function applyGpuStatus(d) {
            gpuFault = (typeof d.gpu_error === "string" && d.gpu_error)
                     ? d.gpu_error : "";
            if (notice) {
                if (gpuFault) {
                    notice.textContent =
                        "GPU stats unavailable — " + gpuFault
                        + "\nThe driver is checked once, when the server "
                        + "starts, so restart the server after fixing the "
                        + "host.";
                }
                notice.hidden = !gpuFault;
            }
            if (toggle) toggle.classList.toggle("has-gpu-fault", !!gpuFault);
            refreshToggleTitle();
        }

        /* ONE request at mount when the card starts collapsed.
         *
         * Collapsing means "stop polling" on purpose -- a folded-away
         * widget doing 1 Hz server work is waste.  But the card is
         * collapsed on FIRST visit, and taken literally that also
         * silences the one reading worth interrupting for: a host whose
         * GPU driver is unreachable, which will fail the calculation
         * the user is about to submit just as surely as it fails the
         * sparkline.  So the collapsed card asks exactly once, reads
         * only gpu_error, and marks the pill.  One request per page
         * load, not a stream -- the "no polling while collapsed" rule
         * is intact.
         *
         * Deliberately NOT routed through ``aborter``: that handle
         * belongs to the 1 Hz loop, and stopTimer() aborting this
         * one-shot would be exactly backwards. */
        function checkGpuOnce() {
            fetch(ENDPOINT, { credentials: "same-origin", cache: "no-store" })
                .then(function(r) { return r.ok ? r.json() : null; })
                .then(function(body) {
                    if (!body || body.ok !== true) return;
                    applyGpuStatus(body.data || {});
                })
                .catch(function() { /* opening the card asks again */ });
        }

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
                    // A failed reading leaves the cells showing the last
                    // good one.  That is deliberate: blanking them on a
                    // single dropped poll would flicker a card whose whole
                    // job is to show a trend.
                    if (!body || body.ok !== true) return;
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
                    // GPU: an empty ``gpus`` list has TWO causes and they
                    // are not the same news.  ``gpu_error`` is the one
                    // that tells them apart -- the server sets it only
                    // when NVML was installed and would not start.
                    //
                    //   * null  -> this host has no GPU support at all.
                    //             Drop the cells, say nothing.  A CPU-
                    //             only box is not a fault.
                    //   * a string -> the host is meant to have a GPU
                    //             and the driver is unreachable.  Still
                    //             no numbers to draw, so the cells stay
                    //             hidden, but the reason goes on screen.
                    //
                    // 2026-08-04: without this, the two look identical
                    // -- a tidy two-cell strip.  A driver/library
                    // mismatch sat on the development host for five
                    // weeks reading as "this machine has no GPU".
                    var gpus = d.gpus || [];
                    applyGpuStatus(d);
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
                .catch(function() {
                    // Swallowed on purpose, and nothing is recorded: an
                    // aborted fetch is the normal shutdown path (collapse
                    // /background), and a genuine network failure is
                    // handled the same way as a bad reading above -- keep
                    // the last good sample and try again in a second.
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
                stopTimer();
            } else {
                root.classList.remove("is-collapsed");
                toggle.setAttribute("aria-pressed", "false");
                startTimer();
            }
            // Composed, not assigned: the title carries the fault too,
            // and that half doesn't change when the card folds.
            refreshToggleTitle();
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
            // Expanded already polls, and its first sample carries the
            // GPU status.  Only the collapsed card needs asking.
            if (startCollapsed) checkGpuOnce();
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
