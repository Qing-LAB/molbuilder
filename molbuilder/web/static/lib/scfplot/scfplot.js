/* scfplot.js — a run's SCF, drawn: energy and residual per iteration; one
 * run, or several on one axis.
 *
 * THE ONE SCF PLOT (`web/trajectory.md` § 2–3; `web/results.md` § 2.5): the
 * trajectory viewer and the transport report both draw a run's SCF through
 * it, into the two elements each hands in — never by document id, so several
 * can share a page.
 *
 * Each run's ``rows`` are one step's SCF cycles as the readers give them
 * (``{cycle, energy, delta_E, gnorm | dHmax, phase?}``).  **Each phase is a
 * trace on its own iteration axis** — a TranSIESTA device's periodic start
 * and its NEGF loop are different calculations — and the last phase is shown
 * by default, the earlier ones a legend click away.  The residual is PySCF's
 * ``|g|`` or SIESTA's ``dHmax``, whichever the rows carry; the criterion of
 * the phase on screen is a dashed line where the run requires it -- one line,
 * every run drawn being one deck's (``criteria``:
 * ``{phase: {residual: {tolerance, unit, required}}}``).
 */
(function (root) {
    "use strict";

    /* The phases in the order the run went through them -- a row stating
     * none is the run's one phase. */
    function phasesOf(rows) {
        const order = [];
        const by = {};
        rows.forEach((c) => {
            const ph = c.phase || "";
            if (!(ph in by)) { by[ph] = []; order.push(ph); }
            by[ph].push(c);
        });
        return order.map((ph) => ({ phase: ph, rows: by[ph] }));
    }

    /* The residual the rows carry: ``[key, name]``, or ``null``. */
    function residualOf(rows) {
        const r = rows[0] || {};
        if (r.gnorm !== undefined) return ["gnorm", "|g|"];
        if (r.dHmax !== undefined) return ["dHmax", "dHmax"];
        return null;
    }

    function _name(ph) {
        return ph === "negf" ? "NEGF" : (ph ? ph : "SCF");
    }

    /* Draw ``runs`` -- ``[{name, rows, colour?, bold?}]``, one run's SCF
     * each, one or many -- into ``energyEl`` and ``residualEl``: a trace per
     * run and phase, coloured by phase for one run and by run for several
     * (a frame set's device runs, `web/results.md` § 2.5) -- a run's own
     * ``colour`` when the caller gives it, so a run keeps the colour it has
     * on the caller's other charts -- the ``bold`` run drawn heavier, each
     * run's last phase shown and its earlier ones a legend click away.  Answers ``{phases,
     * residual}`` -- what was drawn, for the caller's status line -- or
     * ``null`` when there is nothing to draw (the caller hides its
     * section). */
    function draw(energyEl, residualEl, runs, opts) {
        const o = opts || {};
        const Plotly = root.Plotly;
        const given = (Array.isArray(runs) ? runs : []).filter(
            (r) => r && Array.isArray(r.rows) && r.rows.length);
        if (!Plotly || !given.length) return null;
        const theme = root.molbuilder.plotTheme();
        const layoutOf = root.molbuilder.themedLayout;
        const colour = (i) => theme.palette[i % theme.palette.length];
        // SEVERAL RUNS ASKED FOR are named and coloured as runs, though only
        // one holds rows yet -- its plot says whose it is.
        const many = (Array.isArray(runs) ? runs : []).length > 1;
        /* EVERY TRACE, run by run and phase by phase. */
        const traces = [];
        given.forEach((run, r) => {
            const groups = phasesOf(run.rows);
            groups.forEach((g, i) => traces.push({
                run: r, group: g, last: i === groups.length - 1,
                name: (many ? run.name + (groups.length > 1
                    ? " · " + _name(g.phase) : "") : _name(g.phase)),
                colour: many && run.colour ? run.colour : colour(many ? r : i),
                width: run.bold ? 2.5 : 1.5,
            }));
        });
        const several = traces.length > 1;
        const shown = (tr) => (tr.last ? true : "legendonly");

        root.molbuilder.plotBox.draw(energyEl, traces.map((tr) => ({
            x: tr.group.rows.map((c) => c.cycle),
            y: tr.group.rows.map((c) => c.energy),
            mode: "lines+markers", marker: { size: 4 },
            line: { color: tr.colour, width: tr.width },
            name: tr.name, visible: shown(tr),
        })), layoutOf({
            title: { text: o.energyTitle || "SCF energy", font: { size: 12 } },
            margin: { l: 8, r: 12, t: 28, b: 30 },
            showlegend: several,
            legend: { orientation: "h", y: -0.25 },
            xaxis: { title: { text: "SCF cycle", standoff: 4 },
                     zeroline: false, automargin: true, nticks: 6 },
            /* NINE FIGURES: an SCF's last cycles differ in the fifth decimal
             * of a few-thousand-eV energy, and six showed every tick the
             * same. */
            yaxis: { title: { text: "E (eV)", standoff: 4 },
                     tickformat: ".9~r", zeroline: false,
                     automargin: true, nticks: 5 },
        }, theme), { displayModeBar: false });

        const phases = Array.from(new Set(traces.map((tr) => tr.group.phase)));
        const res = residualOf(given[0].rows);
        if (!res || !residualEl) {
            if (residualEl) residualEl.hidden = true;
            return { phases: phases, residual: null };
        }
        residualEl.hidden = false;
        const [key, name] = res;
        const crit = o.criteria || {};
        const critPhases = Object.keys(crit);
        const shapes = [];
        const notes = [];
        /* THE CRITERION OF THE PHASE ON SCREEN, once: every run drawn is one
         * deck's, so it is one line. */
        const onScreen = traces.find((tr) => tr.last && tr.run === 0);
        const ph = onScreen.group.phase
            || (critPhases.length === 1 ? critPhases[0] : null);
        const c = ph && crit[ph] ? crit[ph][name] : null;
        if (c && typeof c.tolerance === "number" && c.required !== false) {
            shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1,
                          yref: "y", y0: c.tolerance, y1: c.tolerance,
                          line: { color: theme.success, width: 1.5,
                                  dash: "dash" } });
            notes.push({ xref: "paper", x: 1, xanchor: "right", yref: "y",
                         y: c.tolerance, yanchor: "bottom",
                         text: "tol " + c.tolerance.toExponential(1),
                         font: { size: 9, color: theme.success },
                         showarrow: false });
        }
        root.molbuilder.plotBox.draw(residualEl, traces.map((tr) => ({
            x: tr.group.rows.map((c) => c.cycle),
            y: tr.group.rows.map((c) => c[key]),
            mode: "lines+markers", marker: { size: 4 },
            line: { color: tr.colour, width: tr.width },
            name: tr.name + " " + name, visible: shown(tr),
        })), layoutOf({
            title: { text: "SCF residual " + name, font: { size: 12 } },
            margin: { l: 8, r: 12, t: 28, b: 30 },
            showlegend: several,
            legend: { orientation: "h", y: -0.25 },
            xaxis: { title: { text: "SCF cycle", standoff: 4 },
                     zeroline: false, automargin: true, nticks: 6 },
            yaxis: { title: { text: name + " (eV)", standoff: 4 },
                     type: "log", zeroline: false, tickformat: ".0e",
                     automargin: true, nticks: 5 },
            shapes: shapes, annotations: notes,
        }, theme), { displayModeBar: false });
        return { phases: phases, residual: name };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.scfPlot = { draw: draw, phasesOf: phasesOf,
                                residualOf: residualOf };
})(typeof window !== "undefined" ? window : this);
