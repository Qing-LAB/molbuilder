/* scfplot.js — one run's SCF, drawn: energy and residual per iteration.
 *
 * THE ONE SCF PLOT (`web/trajectory.md` § 2–3; `web/results.md` § 2.5): the
 * trajectory viewer and the transport report both draw a run's SCF through
 * it, into the two elements each hands in — never by document id, so several
 * can share a page.
 *
 * ``rows`` are one step's SCF cycles as the readers give them
 * (``{cycle, energy, delta_E, gnorm | dHmax, phase?}``).  **Each phase is a
 * trace on its own iteration axis** — a TranSIESTA device's periodic start
 * and its NEGF loop are different calculations — and the last phase is shown
 * by default, the earlier ones a legend click away.  The residual is PySCF's
 * ``|g|`` or SIESTA's ``dHmax``, whichever the rows carry; each phase's own
 * criterion is a dashed line where the run requires it (``criteria``:
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

    /* Draw ``rows`` into ``energyEl`` and ``residualEl``.  Answers
     * ``{phases, residual}`` -- what was drawn, for the caller's status
     * line -- or ``null`` when there is nothing to draw (the caller hides
     * its section). */
    function draw(energyEl, residualEl, rows, opts) {
        const o = opts || {};
        const Plotly = root.Plotly;
        if (!Plotly || !Array.isArray(rows) || rows.length === 0) return null;
        const theme = root.molbuilder.plotTheme();
        const layoutOf = root.molbuilder.themedLayout;
        const groups = phasesOf(rows);
        const last = groups.length - 1;
        const colour = (i) => theme.palette[i % theme.palette.length];
        const shown = (i) => (i === last ? true : "legendonly");
        const several = groups.length > 1;

        Plotly.react(energyEl, groups.map((g, i) => ({
            x: g.rows.map((c) => c.cycle),
            y: g.rows.map((c) => c.energy),
            mode: "lines+markers", marker: { size: 4 },
            line: { color: colour(i), width: 1.5 },
            name: _name(g.phase), visible: shown(i),
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
        }, theme), { displayModeBar: false, responsive: true });

        const res = residualOf(rows);
        if (!res || !residualEl) {
            if (residualEl) residualEl.hidden = true;
            return { phases: groups.map((g) => g.phase), residual: null };
        }
        residualEl.hidden = false;
        const [key, name] = res;
        const crit = o.criteria || {};
        const critPhases = Object.keys(crit);
        const shapes = [];
        const notes = [];
        groups.forEach((g, i) => {
            /* THE PHASE'S OWN CRITERION; rows that state no phase are the
             * run's one phase. */
            const ph = g.phase || (critPhases.length === 1 ? critPhases[0]
                                                           : null);
            const c = ph && crit[ph] ? crit[ph][name] : null;
            if (!c || typeof c.tolerance !== "number" || c.required === false)
                return;
            if (i !== last) return;     // the line of the phase on screen
            shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1,
                          yref: "y", y0: c.tolerance, y1: c.tolerance,
                          line: { color: theme.success, width: 1.5,
                                  dash: "dash" } });
            notes.push({ xref: "paper", x: 1, xanchor: "right", yref: "y",
                         y: c.tolerance, yanchor: "bottom",
                         text: "tol " + c.tolerance.toExponential(1),
                         font: { size: 9, color: theme.success },
                         showarrow: false });
        });
        Plotly.react(residualEl, groups.map((g, i) => ({
            x: g.rows.map((c) => c.cycle),
            y: g.rows.map((c) => c[key]),
            mode: "lines+markers", marker: { size: 4 },
            line: { color: colour(i), width: 1.5 },
            name: _name(g.phase) + " " + name, visible: shown(i),
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
        }, theme), { displayModeBar: false, responsive: true });
        return { phases: groups.map((g) => g.phase), residual: name };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.scfPlot = { draw: draw, phasesOf: phasesOf,
                                residualOf: residualOf };
})(typeof window !== "undefined" ? window : this);
