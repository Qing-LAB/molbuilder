/* Bench-sweep inspector -- a whole sweep on one page, while it runs.
 *
 * Spec: docs/web/bench-summary.md.
 *
 * It claims ``job-set.json``, which a sweep already writes into its own
 * bench/ directory -- so there is nothing new to detect and no new kind of
 * thing for the picker to list (§ 1).  Registration order matters: the
 * source inspector matches every ``.json``, so this one is listed FIRST in
 * results.html and tests/test_inspector_registration_order.py pins it.
 *
 * THIS FILE DRAWS; IT DOES NOT DECIDE (§ 2, B1).  Every number shown here
 * arrives composed from /api/bench/summary -- the trials, their measured
 * s/iter, the verdict, and which coordinate the sweep varied.  Nothing is
 * totalled, ranked or converted in the browser, because B2 is a real
 * scar: submission.md § 3 records a summary that said "170 minutes" for
 * five 38-minute jobs by working out its own total a second way, and a
 * page comparing six trials has six chances to repeat it.
 */
(function (root) {
    "use strict";

    /* B4: the trajectory viewer's cadence.  A sweep is watched precisely
     * while it runs, so a page showing a silently stale verdict would be
     * worse than one showing nothing. */
    const POLL_MS = 15000;

    const _basename = (root.molbuilder
                      && root.molbuilder.path
                      && root.molbuilder.path.basename)
                   || ((p) => String(p || "").split(/[\\/]/).pop());

    /* The state words runstatus hands out, and how each should read.  A
     * word missing from here still renders -- as itself, in the neutral
     * tone -- because inventing a severity for a state we do not know is
     * worse than showing the state. */
    const TONE = {
        finished:      "ok",
        running:       "busy",
        queued:        "busy",
        pending:       "busy",
        "not-started": "idle",
        stale:         "warn",
        failed:        "bad",
    };

    const el = (tag, cls, text) => {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined && text !== null) n.textContent = String(text);
        return n;
    };

    /** s/iter as the number the server measured, or an em-dash.  No
     *  rounding that changes the value: two decimals is presentation,
     *  and the raw number stays in the title attribute. */
    function _sPerIter(v) {
        return (typeof v === "number" && isFinite(v)) ? v.toFixed(2) : "—";
    }

    /** One decimal, or nothing at all.  `null`/absent means the monitor
     *  never measured it, which is not the same as zero. */
    function _num(v, unit, digits) {
        return (typeof v === "number" && isFinite(v))
            ? v.toFixed(digits === undefined ? 1 : digits) + unit : null;
    }

    /** "peak 101.2 GB · cpu 31% · gpu 23% · vram 14.0 GB · 275 s · host-bound"
     *
     *  Order is the order a person reads it in: what it needed, what it
     *  used, how long, and what held it back. */
    function _usageLine(t) {
        const m = t.metrics || {};
        const bits = [];
        const mem = _num(m.peak_rss_gb, " GB");
        if (mem) bits.push("peak " + mem);
        const cpu = _num(m.cpu_mean_pct, "%", 0);
        if (cpu) bits.push("cpu " + cpu);
        const sm = _num(m.gpu_sm_mean_pct, "%", 0);
        if (sm) bits.push("gpu " + sm);
        const vram = _num(m.gpu_vram_peak_gb, " GB");
        if (vram) bits.push("vram " + vram);
        const wall = _num(m.wall_s, " s", 0);
        if (wall) bits.push(wall);
        // WHAT HELD IT BACK -- the server's own word (BenchPoint.bound),
        // already shown in the knob line when present, repeated here only
        // when it is the answer to "why is this one slow".
        return bits.join(" \u00b7 ");
    }

    /** "np 4 · thr 1 · a100×1" -- the knobs, in the job-set's own words. */
    function _knobLine(t) {
        const bits = [];
        const k = t.knobs || {};
        if (k.mpi_np) bits.push(`np ${k.mpi_np}`);
        if (k.cpus_per_task) bits.push(`thr ${k.cpus_per_task}`);
        bits.push(k.gres ? String(k.gres) : "no gpu");
        if (t.bound) bits.push(`${t.bound}-bound`);
        return bits.join(" · ");
    }

    /* ---- the comparison charts ------------------------------------ *
     * § 3: every measured quantity against the coordinate the sweep
     * actually varied.  `varied` is the SERVER's answer, so the browser
     * never has to work out what the sweep was about.
     *
     * THREE PANELS, ONE X AXIS, because the quantities do not share a
     * scale: s/iter runs 60-135, memory 7-101 GB, utilisation 0-100 %.
     * Drawn on one axis the percentages would be a flat line along the
     * bottom.  Stacked and aligned, a person reads DOWN a coordinate --
     * "at G=4 it got faster, used less memory, and the GPU still sat at
     * 28%" -- which is the sentence a benchmark exists to produce.
     *
     * A series absent from `metrics` is NOT PLOTTED AS ZERO.  The monitor
     * could not measure it (no GPU on that shelf), and a zero would read
     * as "measured, and it was idle" -- the opposite of the truth.  A
     * panel whose every series is absent is dropped entirely.          */
    //: Group colours -- distinguishable on the dark card and to the most
    //: common colour-vision deficiencies (blue / orange / green / pink).
    const PALETTE = ["#5aa9e6", "#f4a259", "#69c37b", "#e07a9c", "#b48ead"];

    /** A design token's value, with the literal that mirrors it as the
     *  fallback -- the embed-safety pattern (`ui-contract.md` § 2), because
     *  Plotly is handed colours as strings and cannot read a `var()`. */
    function _token(name, fallback) {
        const v = getComputedStyle(document.documentElement)
            .getPropertyValue(name).trim();
        return v || fallback;
    }
    const PANELS = [
        { title: "s / iter", series: [
            { key: "s_per_iter", name: "s/iter", digits: 1,
              from: (t) => t.s_per_iter } ] },
        { title: "GB", series: [
            { key: "peak_rss_gb", name: "peak RAM", digits: 1,
              from: (t) => (t.metrics || {}).peak_rss_gb },
            { key: "gpu_vram_peak_gb", name: "peak VRAM", digits: 1,
              from: (t) => (t.metrics || {}).gpu_vram_peak_gb } ] },
        { title: "% busy", series: [
            { key: "cpu_mean_pct", name: "CPU", digits: 0,
              from: (t) => (t.metrics || {}).cpu_mean_pct },
            { key: "gpu_sm_mean_pct", name: "GPU", digits: 0,
              from: (t) => (t.metrics || {}).gpu_sm_mean_pct } ] },
    ];

    function drawChart(hostEl, data) {
        const varied = data.varied || [];
        if (!varied.length || !root.Plotly) return false;
        const timed = (data.trials || [])
            .filter((t) => typeof t.s_per_iter === "number");
        if (timed.length < 2) return false;

        /* Pick the axis that varies among the trials WE ARE ABOUT TO DRAW,
         * not merely across the sweep.  `varied` answers "what did this
         * sweep vary", which is the right question for the sweep and the
         * wrong one for the chart: early on, only a few trials have
         * finished, and they may all share a value of the first varied
         * coordinate.  Taking varied[0] regardless drew every finished
         * trial at the same x -- a vertical line that looks like data. */
        const spread = (k) => new Set(
            timed.map((t) => JSON.stringify((t.point || {})[k]))).size;
        const axis = varied.find((k) => spread(k) > 1);
        if (!axis) return false;      // nothing comparable yet: cards alone

        const at = (t) => (t.point ? t.point[axis] : undefined);
        const ordered = timed
            .filter((t) => at(t) !== undefined && at(t) !== null)
            .sort((a, b) => (at(a) > at(b) ? 1 : at(a) < at(b) ? -1 : 0));
        if (ordered.length < 2) return false;

        /* ONE LINE PER SERIES, NOT ONE LINE THROUGH EVERYTHING.
         *
         * A sweep usually varies more than one thing.  Au-BDT-Au varied G,
         * K, use_gpu and diag_algorithm: at each G there are TWO trials,
         * one per solver.  Plotted as a single line sorted by x, the two
         * land on the same x and the line draws a VERTICAL SEGMENT joining
         * them -- at G=0 it spanned 134 and 91 s/iter -- which reads as one
         * continuous curve and is not.  (The same class of defect as
         * taking `varied[0]` regardless: a shape that looks like data.)
         *
         * The grouping coordinates are the other varied ones that differ
         * WITHIN a single x.  A coordinate constant at each x is not a
         * second dimension, it is the SAME axis spelled differently -- K
         * and use_gpu here, which track G exactly -- and grouping on those
         * would split every line into singletons.
         */
        const groupKeys = varied.filter((k) => {
            if (k === axis) return false;
            const byX = new Map();
            for (const t of ordered) {
                const x = JSON.stringify(at(t));
                if (!byX.has(x)) byX.set(x, new Set());
                byX.get(x).add(JSON.stringify((t.point || {})[k]));
            }
            return [...byX.values()].some((vals) => vals.size > 1);
        });
        // With ONE grouping key the key name is the same on every line and
        // adds nothing -- "ELPA-1STAGE" reads; "diag_algorithm ELPA-1STAGE"
        // is the same word twice.  Name the key only when there are several
        // and the value alone would be ambiguous.
        const groupOf = (t) => groupKeys
            .map((k) => (groupKeys.length > 1 ? `${k} ` : "")
                        + `${(t.point || {})[k]}`)
            .join(" \u00b7 ");
        const groups = groupKeys.length
            ? [...new Set(ordered.map(groupOf))].sort()
            : [""];

        const live = PANELS
            .map((p) => ({
                title: p.title,
                series: p.series.filter(
                    (ser) => ordered.some(
                        (t) => typeof ser.from(t) === "number")),
            }))
            .filter((p) => p.series.length);
        if (!live.length) return false;

        const won = (data.choice || {}).label;
        const traces = [];
        // ONE key entry per group across the WHOLE figure.  Keying per
        // panel put the same two names up three times.
        const legended = new Set();
        const layout = {
            margin: { l: 64, r: 20, t: 34, b: 52 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            /* The card's own type, not Plotly's 12px default -- the panel
               labels were smaller than every other word on the page. */
            font: { color: getComputedStyle(document.body).color, size: 15 },
            showlegend: groups.length > 1
                        || live.some((p) => p.series.length > 1),
            legend: { orientation: "h", y: 1.18, x: 0,
                      font: { size: 15 } },
            /* HOVER THE BAR YOU MEANT.  "x unified" stacks every series
               at that x into one box -- four rows, each repeating the same
               trial name -- and the values are already printed on the bars,
               so the tip's remaining job is to name the ONE run under the
               cursor. */
            hovermode: "closest",
            /* THE TIP IS PART OF THE APP, NOT PLOTLY'S DEFAULT.  Unstyled
               it renders a pale box with the trace colour behind small
               type: light blue under white, on a dark card.  Read from the
               tokens at draw time so it follows the theme instead of
               freezing one palette (`ui-contract.md` § 2). */
            hoverlabel: {
                bgcolor: _token("--bg-page", "#14171c"),
                bordercolor: _token("--border-strong", "#3a3f48"),
                font: {
                    color: _token("--text-primary", "#e6e9ef"),
                    family: getComputedStyle(document.body).fontFamily,
                    size: 14,
                },
                align: "left",
            },
            barmode: "group",
            bargap: 0.35,
            bargroupgap: 0.08,
        };
        // Stack top-to-bottom.  Generous, because three panels crammed
        // into one card height is a sparkline with axis labels.
        const GAP = 0.16;
        const h = (1 - GAP * (live.length - 1)) / live.length;
        live.forEach((panel, i) => {
            const n = i + 1;                       // 1-based Plotly axis ids
            const ySuffix = n === 1 ? "" : String(n);
            const top = 1 - i * (h + GAP);
            layout["yaxis" + ySuffix] = {
                title: { text: panel.title, font: { size: 14 } },
                rangemode: "tozero",
                // Headroom so the value above the TALLEST bar is not
                // clipped by the panel above it.
                automargin: true,
                domain: [Math.max(0, top - h), top], automargin: true,
                /* A GRID IS A READING AID, NOT A DECORATION.  Plotly's
                   default draws solid mid-grey lines that on this dark card
                   read as strongly as the data.  A hairline at ~6% alpha is
                   enough to carry the eye across to the axis and disappears
                   the moment you stop looking for it. */
                gridcolor: "rgba(255,255,255,0.07)",
                zeroline: false,
                ticks: "outside", ticklen: 4,
                tickfont: { size: 13 },
            };
            panel.series.forEach((ser, si) => groups.forEach((g, gi) => {
                const pts = ordered.filter(
                    (t) => typeof ser.from(t) === "number"
                        && (!groupKeys.length || groupOf(t) === g));
                if (!pts.length) return;
                const key = g || ser.name;
                const firstOfItsKind = !legended.has(key);
                legended.add(key);
                traces.push({
                    x: pts.map((t) => String(at(t))),
                    y: pts.map(ser.from),
                    /* THE VALUE, ON THE BAR (2026-08-25, user).  A dozen
                       bars is few enough to label every one, and a printed
                       number beats an axis tick plus an estimate --
                       especially where two bars are close, which is the
                       comparison a sweep is FOR.  The trial name goes to
                       the hover instead: printed inside the bar it was
                       twelve rotated labels shrunk to fit, which is noise. */
                    text: pts.map((t) => {
                        const v = ser.from(t);
                        return v.toFixed(ser.digits === undefined
                                         ? 1 : ser.digits);
                    }),
                    textposition: "outside",
                    textfont: { size: 13 },
                    cliponaxis: false,
                    customdata: pts.map((t) => t.label),
                    // ONE LEGEND ENTRY PER GROUP, NOT PER TRACE.  Naming
                    // every trace put ten entries above a three-panel chart
                    // and buried the panels under their own key.  Colour
                    // carries the group, dash carries the series within a
                    // panel, and the y-axis title already says which
                    // quantity a panel is -- so the legend only has to
                    // answer "which line is which run".
                    name: key,
                    legendgroup: key,
                    showlegend: firstOfItsKind,
                    yaxis: "y" + ySuffix,
                    /* BARS, NOT LINES (2026-08-25, user).  These are six
                       SEPARATE RUNS at three GPU counts, not samples of a
                       continuous function: there is no G=1 trial, and a
                       line between G=0 and G=2 draws a value nobody
                       measured.  A bar claims only what was run.  Every
                       quantity here also has a meaningful zero -- seconds,
                       gigabytes, percent -- which is the other condition a
                       bar needs to be honest. */
                    type: "bar",
                    hovertemplate: "<b>%{customdata}</b><br>"
                                 + ser.name + ": %{y}<extra></extra>",
                    marker: {
                        color: PALETTE[gi % PALETTE.length],
                        /* The WINNER is outlined rather than recoloured:
                           colour already means "which run", and a second
                           meaning on the same channel is unreadable. */
                        line: {
                            color: pts.map(
                                (t) => (t.label === won ? "#f5d76e"
                                                        : "rgba(0,0,0,0)")),
                            width: pts.map((t) => (t.label === won ? 2 : 0)),
                        },
                        /* A second series in the same panel keeps the run's
                           colour and takes a hatch -- VRAM is not a
                           different run from RAM, it is a different thing
                           measured of it. */
                        pattern: si === 0 ? {} : { shape: "/", size: 6,
                                                   solidity: 0.35 },
                    },
                });
            }));
        });
        /* automargin, because the axis TITLES are the whole point: a bare
           "4" on the x-axis does not say it means K. */
        /* TICKS ONLY WHERE A TRIAL EXISTS.  Plotly's automatic ticks put
           labels at 0.5, 1.5, 2.5 -- and `G` is a COUNT OF GPUs, so those
           are values no trial could have.  A tick that cannot exist invites
           the eye to read the line between two points as measurement, and
           it is interpolation.  Vertical gridlines go for the same reason:
           there is nothing at those x to line up with. */
        const xs = [...new Set(ordered.map(at))]
            .sort((a, b) => (a > b ? 1 : a < b ? -1 : 0))
            .map(String);
        layout.xaxis = {
            title: { text: axis, font: { size: 15 } }, automargin: true,
            anchor: "y" + (live.length === 1 ? "" : String(live.length)),
            // CATEGORIES, in measured order.  A numeric axis would space
            // G=0,2,4 as though 1 and 3 were simply empty; they are not
            // empty, they were never run.
            type: "category", categoryorder: "array", categoryarray: xs,
            showgrid: false, zeroline: false,
            ticks: "outside", ticklen: 4,
            tickfont: { size: 14 },
        };
        root.Plotly.newPlot(hostEl, traces, layout,
                            { displayModeBar: false, responsive: true });
        return true;
    }

    /* ---- one trial's card ----------------------------------------- */
    function trialCard(t, wonLabel) {
        const card = el("div", "bench-trial");
        if (t.label === wonLabel) card.classList.add("is-winner");

        const head = el("div", "bench-trial-head");
        head.appendChild(el("span", "bench-trial-label", t.label));
        const tone = TONE[t.state] || "idle";
        head.appendChild(el("span", `bench-state is-${tone}`, t.state));
        const s = el("span", "bench-siter", `${_sPerIter(t.s_per_iter)} s/iter`);
        if (typeof t.s_per_iter === "number") s.title = String(t.s_per_iter);
        head.appendChild(s);
        card.appendChild(head);

        card.appendChild(el("div", "bench-knobs", _knobLine(t)));

        /* WHERE IT RAN -- what the other numbers on this card are a
         * measurement of (B5, scheduler.md R12).  The brief is the
         * SERVER's spelling (one door; a second one here would drift) and
         * the hostname rides along as provenance: which box, for tracing
         * a bad one -- never what the kinds census compares (R11).
         * Absent on records that predate the [MACHINE] line: absent is
         * absent, not "?". */
        if (t.machine_brief) card.appendChild(el(
            "div", "bench-machine",
            "on " + t.machine_brief
            + ((t.machine || {}).node ? ` (${t.machine.node})` : "")));

        /* WHAT IT ACTUALLY USED.  `summarize` already measures all of this
         * from the monitor's `<label>.util.csv` -- and until 2026-08-25
         * this card threw every field away, showing s/iter and nothing
         * else.  These are the numbers a person writes the RUN script
         * from: how much memory to ask for (peak, not the request), and
         * whether the accelerator paid for itself.  A 256 G ask that
         * peaked at 101 G with the GPU at 23% is a different script.
         *
         * B1 holds: every value is the server's, printed, not recomputed.
         * A field the monitor could not collect (no GPU on that shelf) is
         * absent from `metrics` and simply not drawn -- never a zero,
         * which would read as "measured, and it was idle". */
        const used = _usageLine(t);
        if (used) card.appendChild(el("div", "bench-usage", used));

        /* HOW MANY ITERATIONS THE HEADLINE RESTS ON.  Under the capped
         * 3-iteration trial `parse_scf_timing` drops the warm-up delta and
         * averages what is left -- which is ONE sample.  The verdict ranks
         * on it, and a 5% gap between two single measurements is not a
         * result.  Said out loud only when it is 1: at 3+ the average is
         * the ordinary case and the note would be noise. */
        const iters = (t.metrics || {}).iters_measured;
        if (iters === 1) card.appendChild(el(
            "div", "bench-note",
            "s/iter is a single iteration — no spread, so small gaps "
            + "between trials are not decisive"));

        /* WHAT IT ACTUALLY RAN WITH, read back out of the deck and the
         * wrapper log.  `mismatch` below shows only the DISAGREEMENTS;
         * this shows the settled truth -- the block size SIESTA chose,
         * the ELPA build that answered.  A sweep over solvers is largely
         * a question about these values, and they were being carried
         * across the wire and dropped. */
        /* Only what the knob line ABOVE did not already say.  `effective`
         * repeats the ranks and threads it was asked for, and printing
         * "np 48 · thr 1" and then "mpi_np 48 · omp_threads 1" one line
         * down is the same fact twice in two spellings -- which is how a
         * reader stops trusting either.  What is left is what only the RUN
         * can tell you: the block size SIESTA settled on, the ELPA build
         * that answered.  A key in `mismatch` is skipped too: that row
         * shows asked-vs-ran itself, in more detail. */
        const ALREADY_IN_KNOBS = ["mpi_np", "omp_threads", "cpus_per_task"];
        const eff = t.effective || {};
        const effBits = Object.keys(eff)
            .filter((k) => eff[k] !== null && eff[k] !== undefined
                        && !(k in (t.mismatch || {}))
                        && ALREADY_IN_KNOBS.indexOf(k) === -1)
            .map((k) => `${k} ${eff[k]}`);
        if (effBits.length) card.appendChild(
            el("div", "bench-effective", "ran with: " + effBits.join(" \u00b7 ")));

        /* B3: a trial with nothing to show SAYS so.  Hiding it would
         * answer "where did my third trial go?" with silence. */
        if (typeof t.s_per_iter !== "number") {
            card.appendChild(el(
                "div", "bench-note",
                t.detail || "no timing yet — this trial has not produced one"));
        }

        /* What it ASKED for beside what it RAN -- a silent eigensolver
         * fallback is exactly what a sweep exists to catch, and
         * BenchPoint.mismatch already worked out the disagreement. */
        const mm = t.mismatch || {};
        const keys = Object.keys(mm);
        if (keys.length) {
            const box = el("div", "bench-mismatch");
            box.appendChild(el("div", "bench-mismatch-head",
                               "ran something else than it was asked to:"));
            for (const k of keys) {
                box.appendChild(el(
                    "div", "bench-mismatch-row",
                    `${k}: asked ${JSON.stringify(mm[k].asked)}, `
                    + `ran ${JSON.stringify(mm[k].ran)}`));
            }
            card.appendChild(box);
        }
        return card;
    }

    function render(host, data) {
        host.innerHTML = "";
        const wrap = el("div", "bench-summary");

        const head = el("div", "bench-head");
        head.appendChild(el("h2", "bench-title", data.name || "sweep"));
        head.appendChild(el(
            "span", "bench-count",
            `${data.n_trials} trials · ${data.n_done} done`
            + (data.complete === false ? " · still running" : "")));
        wrap.appendChild(head);

        /* WHAT IT IS A BENCHMARK OF, AND WHERE IT RAN.  Both arrive in
         * every payload and neither was drawn until 2026-08-25 -- so the
         * page reported "62.6 s/iter" with no way to know it was 444
         * atoms on 2x64 slurm cores.  A number per iteration means
         * nothing without the system it iterated over: that is the first
         * question anyone asks of a benchmark, and the answer was already
         * in the response. */
        const ctx = [];
        const sys = data.system || {};
        if (sys.n_atoms) ctx.push(`${sys.n_atoms} atoms`);
        if (data.engine) ctx.push(String(data.engine));
        /* THE MACHINE UNDER THE TRIALS -- the composer's census
         * (`machines`: one entry per KIND, scheduler.md R11), which the
         * monitor recorded on the node at run time.  One kind: it joins
         * the context line.  Several: it gets its own line below, because
         * one core figure here would be picking a winner silently (B5).
         *
         * The `effective.node_phys_cores` path is the FALLBACK for sweeps
         * recorded before the [MACHINE] line existed -- it measured cores
         * only.  (It replaced `environment.topology` on 2026-08-25, which
         * described a node the sweep never necessarily saw.) */
        const machines = data.machines || [];
        if (machines.length === 1) {
            ctx.push(machines[0].brief);
        } else if (!machines.length) {
            const nodeCores = [...new Set((data.trials || [])
                .map((t) => (t.effective || {}).node_phys_cores)
                .filter((n) => typeof n === "number"))].sort((a, b) => a - b);
            if (nodeCores.length === 1) ctx.push(`${nodeCores[0]}-core node`);
            else if (nodeCores.length > 1) ctx.push(nodeCores.join("/") + "-core nodes");
        }
        const env = data.environment || {};
        if (env.scheduler) ctx.push(String(env.scheduler));
        if (ctx.length) wrap.appendChild(
            el("div", "bench-context", ctx.join(" \u00b7 ")));

        /* WHICH MACHINES, said plainly when there is more than one kind
         * (`generator.md` 4.4b).  A statement, not a warning: a mixed
         * CPU/GPU sweep spans machines by construction and may be exactly
         * the intended experiment -- the reader judges the comparison,
         * this line makes sure they know it is one.  B1 holds: the census
         * is the server's; nothing is grouped or counted here. */
        if (machines.length > 1) {
            wrap.appendChild(el(
                "div", "bench-machines",
                `trials ran on ${machines.length} kinds of node: `
                + machines.map((m) => `${m.brief} (${m.trials} trial`
                                      + (m.trials !== 1 ? "s" : "") + ")")
                          .join(" \u00b7 ")));
        }

        /* The verdict, whole -- including its absence, which is a real
         * answer: choose_winner returns {} when nothing was timed, OR
         * when every timed trial ran something other than its label. */
        const choice = data.choice || {};
        const verdict = el("div", "bench-verdict");
        if (choice.label) {
            verdict.classList.add("is-decided");
            /* The rationale is the server's sentence and already names the
             * winner ("G1K4C1 fastest (1.8 s/iter); vs ..."), so printing
             * the label in front of it says the name twice.  Show the
             * label alone only when there is no rationale to carry it. */
            const why = choice.rationale;
            if (why) verdict.appendChild(el("span", null, why));
            else verdict.appendChild(el("strong", null, choice.label));
        } else {
            verdict.appendChild(document.createTextNode(
                data.n_done
                    ? "No winner: nothing timed, or every timed trial ran "
                      + "something other than what it was asked for."
                    : "No verdict yet — no trial has finished."));
        }
        wrap.appendChild(verdict);

        const chart = el("div", "bench-chart");
        wrap.appendChild(chart);

        /* HOW TO READ IT.  The figures are exact and still not
         * self-explanatory: "GPU 28" is a share of WALL TIME, not of the
         * card's capacity, and a reader who takes it for the latter draws
         * the opposite conclusion about where the bottleneck is.  This is
         * a legend for the quantities, not a verdict about this sweep --
         * B1 holds, nothing here is computed from the trials. */
        const caption = el("div", "bench-caption");
        caption.appendChild(el(
            "div", null,
            "Solid bars are the host's (RAM, CPU); hatched are the "
            + "accelerator's (VRAM, GPU). The gold outline is the winner."));
        caption.appendChild(el(
            "div", null,
            "s/iter — lower is better; it is what the verdict ranks on. "
            + "GB — the PEAK reached, not what was requested. "
            + "% busy — the share of wall time the device had work to do, "
            + "so a low GPU beside a high CPU means the host could not feed "
            + "it, and adding cards will not help."));
        wrap.appendChild(caption);

        const list = el("div", "bench-trials");
        for (const t of (data.trials || [])) {
            list.appendChild(trialCard(t, choice.label));
        }
        wrap.appendChild(list);

        /* B4: the page says WHEN IT LAST LOOKED -- and, since 2026-08-25,
         * ALSO when the data it is showing was measured.
         *
         * They answer different staleness questions and only together
         * cover both.  The browser clock says the poll is still running;
         * it cannot tell you the ANSWER is old, because a response served
         * from a stale cache -- or by a server running six-day-old code --
         * ticks along just as freshly.  `generated_at` is the server's own
         * stamp on the composition, so a widening gap between the two is
         * visible rather than silent.  B4's whole point is that a stale
         * verdict must not look live. */
        const foot = el("div", "bench-foot");
        const measured = data.generated_at
            ? new Date(data.generated_at).toLocaleTimeString()
            : null;
        foot.appendChild(el("span", null,
            "last looked " + new Date().toLocaleTimeString()
            + (measured ? " \u00b7 measured " + measured : "")));
        wrap.appendChild(foot);

        host.appendChild(wrap);
        if (!drawChart(chart, data)) chart.remove();
    }

    const inspector = {
        name:        "bench-summary",
        displayName: "Bench sweep",
        isResult:    true,
        /* Declared so B4's cadence is checkable without a test sitting
         * through it.  A viewer that quietly stopped matching the
         * trajectory viewer's 15 s would still pass every other test
         * here, because nothing else can see the number. */
        pollMs:      POLL_MS,
        resultCategory: () => "Benchmark sweeps",
        /* An exact basename, which is as specific as a match gets -- and
         * why this registers ahead of the .json catch-all. */
        match:       (file) => _basename(file) === "job-set.json",

        mount(host, file, ctx) {
            const state = { disposed: false, timer: null };

            async function refresh() {
                if (state.disposed) return;
                let body;
                try {
                    const r = await fetch("/api/bench/summary?path="
                                          + encodeURIComponent(file));
                    body = await r.json();
                } catch (e) {
                    body = { ok: false, error: String(e) };
                }
                if (state.disposed) return;
                if (!body.ok) {
                    if (ctx && ctx.showError) ctx.showError(body.error);
                    else host.textContent = body.error || "could not read it";
                    return;                       // and stop polling a refusal
                }
                render(host, body);
                state.timer = root.setTimeout(refresh, POLL_MS);
            }
            refresh();

            return {
                dispose() {
                    state.disposed = true;
                    if (state.timer) root.clearTimeout(state.timer);
                    state.timer = null;
                    try {
                        const c = host.querySelector(".bench-chart");
                        if (c && root.Plotly) root.Plotly.purge(c);
                    } catch (_) { /* the chart may never have been drawn */ }
                    host.innerHTML = "";
                },
            };
        },
    };

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.benchSummaryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
