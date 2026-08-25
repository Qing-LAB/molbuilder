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

    /* ---- the comparison chart ------------------------------------- *
     * § 3: the verdict axis (s/iter) against the coordinate the sweep
     * actually varied.  A sweep that varied nothing comparable gets the
     * cards alone rather than a chart of one column -- and `varied` is
     * the SERVER's answer, so the browser never has to work out what the
     * sweep was about.                                                */
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

        const pts = timed
            .map((t) => ({ x: t.point ? t.point[axis] : undefined,
                           y: t.s_per_iter, label: t.label }))
            .filter((p) => p.x !== undefined && p.x !== null);
        if (pts.length < 2) return false;
        pts.sort((a, b) => (a.x > b.x ? 1 : a.x < b.x ? -1 : 0));

        const won = (data.choice || {}).label;
        root.Plotly.newPlot(hostEl, [{
            x: pts.map((p) => p.x),
            y: pts.map((p) => p.y),
            text: pts.map((p) => p.label),
            mode: "lines+markers",
            type: "scatter",
            hovertemplate: "%{text}<br>%{y:.2f} s/iter<extra></extra>",
            marker: {
                size: pts.map((p) => (p.label === won ? 13 : 8)),
                symbol: pts.map((p) => (p.label === won ? "star" : "circle")),
            },
        }], {
            /* automargin, because the axis TITLES are the whole point: a
               bare "4" on the x-axis does not say it means K.  A fixed
               bottom margin clipped them. */
            margin: { l: 56, r: 16, t: 8, b: 40 },
            xaxis: { title: { text: axis }, automargin: true },
            yaxis: { title: { text: "s / iter" }, rangemode: "tozero",
                     automargin: true },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: { color: getComputedStyle(document.body).color },
            showlegend: false,
        }, { displayModeBar: false, responsive: true });
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
            `${data.n_trials} trials · ${data.n_done} done`));
        wrap.appendChild(head);

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

        const list = el("div", "bench-trials");
        for (const t of (data.trials || [])) {
            list.appendChild(trialCard(t, choice.label));
        }
        wrap.appendChild(list);

        const foot = el("div", "bench-foot");
        foot.appendChild(el("span", null,
                            "last looked " + new Date().toLocaleTimeString()));
        wrap.appendChild(foot);

        host.appendChild(wrap);
        if (!drawChart(chart, data)) chart.remove();
    }

    const inspector = {
        name:        "bench-summary",
        displayName: "Bench sweep",
        isResult:    true,
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
