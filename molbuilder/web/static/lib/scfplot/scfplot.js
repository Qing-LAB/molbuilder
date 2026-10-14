/* scfplot.js — a run's SCF, drawn: energy and residuals per iteration; one
 * run, or several on one axis.
 *
 * THE ONE SCF PLOT (`web/trajectory.md` § 2–3; `web/results.md` § 2.5): the
 * trajectory viewer and the transport report both draw a run's SCF through
 * it, into the two elements each hands in — never by document id, so several
 * can share a page.
 *
 * Each run's ``rows`` are one step's SCF cycles as the readers give them
 * (``{cycle, energy, delta_E, dDmax, dHmax, dq | gnorm, phase?}``).  **Each
 * phase is a trace on its own iteration axis** — a TranSIESTA device's
 * periodic start and its NEGF loop are different calculations — and the last
 * phase is shown by default, the earlier ones a legend click away.
 *
 * **Every residual the engine converges on is drawn** (`web/trajectory.md`
 * § 3.0) -- **as the run declares them** (``residuals``: the engine's one
 * declaration, `siesta_grammar.SCF_RESIDUALS` / `molwatch_grammar.
 * SCF_RESIDUALS`, served with every run as ``scf_residuals``): SIESTA's
 * dDmax and dHmax and a NEGF loop's dQ, PySCF's dE and |g| -- each in its
 * own unit, and each against its own criterion, a dashed line where the
 * phase on screen requires it (``criteria``:
 * ``{phase: {criterion: {tolerance, unit, required}}}``; every run drawn is
 * one deck's, so one line a residual).  This module keeps no list.
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

    /* The residuals to draw of what the run declares (``declared``: each
     * ``{name, key, unit, criterion}``): the ones a test bounds, that the
     * rows carry -- in the declaration's order. */
    function residualsOf(rows, declared) {
        return (Array.isArray(declared) ? declared : []).filter((x) =>
            x && x.criterion && rows.some(
                (c) => typeof c[x.key] === "number" && isFinite(c[x.key])));
    }
    /* A residual's name with its unit, as a legend and a line say it. */
    function _said(x) {
        return x.name + (x.unit ? " (" + x.unit + ")" : "");
    }

    function _name(ph) {
        return ph === "negf" ? "NEGF" : (ph ? ph : "SCF");
    }

    /* Draw ``runs`` -- ``[{name, rows, colour?, bold?}]``, one run's SCF
     * each, one or many -- into ``energyEl`` and ``residualEl``: a trace per
     * run and phase -- the energy coloured by phase for one run and by run
     * for several (a frame set's device runs, `web/results.md` § 2.5), each
     * residual a colour of its own for one run and a dash of its own for
     * several -- a run's own
     * ``colour`` when the caller gives it, so a run keeps the colour it has
     * on the caller's other charts -- the ``bold`` run drawn heavier, each
     * run's last phase shown and its earlier ones a legend click away.  Answers ``{phases,
     * residuals}`` -- what was drawn, for the caller's status line -- or
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

        // EACH PANEL'S LEGEND UNDER ITS AXIS TITLE, the theme's one rule.
        const energyLegend = root.molbuilder.legendBelow(several);
        root.molbuilder.plotBox.draw(energyEl, traces.map((tr) => ({
            x: tr.group.rows.map((c) => c.cycle),
            y: tr.group.rows.map((c) => c.energy),
            mode: "lines+markers", marker: { size: 4 },
            line: { color: tr.colour, width: tr.width },
            name: tr.name, visible: shown(tr),
        })), layoutOf({
            title: { text: o.energyTitle || "SCF energy", font: { size: 12 } },
            margin: { l: 8, r: 12, t: 28, b: energyLegend.bottom },
            showlegend: energyLegend.showlegend,
            legend: energyLegend.legend,
            xaxis: { title: { text: "SCF cycle", standoff: 4 },
                     zeroline: false, automargin: true, nticks: 6 },
            yaxis: Object.assign({ title: { text: "E (eV)", standoff: 4 },
                                   zeroline: false, automargin: true,
                                   nticks: 5 }, window.molbuilder.energyAxis),
        }, theme), { displayModeBar: false });

        const phases = Array.from(new Set(traces.map((tr) => tr.group.phase)));
        const all = given.reduce((rows, run) => rows.concat(run.rows), []);
        const residuals = residualsOf(all, o.residuals);
        if (!residuals.length || !residualEl) {
            if (residualEl) residualEl.hidden = true;
            return { phases: phases, residuals: [] };
        }
        residualEl.hidden = false;
        const crit = o.criteria || {};
        const critPhases = Object.keys(crit);
        /* ONE RUN: a colour a residual.  SEVERAL: the run's colour, and a
         * dash a residual -- so a run keeps its colour across the charts. */
        const dashes = ["solid", "dot", "dash"];
        const look = (tr, i) => (many
            ? { color: tr.colour, dash: dashes[i % dashes.length] }
            : { color: colour(i), dash: "solid" });
        const lines = [];
        traces.forEach((tr) => residuals.forEach((x, i) => {
            const rows = tr.group.rows.filter(
                (c) => typeof c[x.key] === "number" && isFinite(c[x.key]));
            if (!rows.length) return;
            lines.push({
                x: rows.map((c) => c.cycle),
                // A LOG AXIS takes a size: dE and dQ carry a sign.
                y: rows.map((c) => Math.abs(c[x.key])),
                mode: "lines+markers", marker: { size: 4 },
                line: Object.assign({ width: tr.width }, look(tr, i)),
                // THE PHASE OR RUN NAMED only where there are several.
                name: (several && tr.name ? tr.name + " · " : "") + _said(x),
                visible: shown(tr),
            });
        }));
        /* EACH RESIDUAL'S CRITERION, for the phase on screen, where the
         * run requires it -- one line a residual, in its colour. */
        const onScreen = traces.find((tr) => tr.last && tr.run === 0);
        const ph = onScreen.group.phase
            || (critPhases.length === 1 ? critPhases[0] : null);
        const shapes = [];
        const notes = [];
        residuals.forEach((x, i) => {
            const c = ph && crit[ph] ? crit[ph][x.criterion] : null;
            if (!c || typeof c.tolerance !== "number" || c.required === false)
                return;
            // THE THEME'S ONE THRESHOLD MARK, on this log axis.
            const mark = root.molbuilder.thresholdMark(
                c.tolerance, x.name + " tol " + c.tolerance.toExponential(1)
                             + (x.unit ? " " + x.unit : ""),
                many ? theme.success : colour(i), true);
            shapes.push(mark.shape);
            notes.push(mark.note);
        });
        const residualLegend = root.molbuilder.legendBelow(
            residuals.length > 1 || several);
        root.molbuilder.plotBox.draw(residualEl, lines, layoutOf({
            title: { text: "SCF residuals " + residuals.map(
                         (x) => x.name).join(" · "), font: { size: 12 } },
            margin: { l: 8, r: 12, t: 28, b: residualLegend.bottom },
            showlegend: residualLegend.showlegend,
            legend: residualLegend.legend,
            xaxis: { title: { text: "SCF cycle", standoff: 4 },
                     zeroline: false, automargin: true, nticks: 6 },
            // EACH IN ITS OWN UNIT, named in the legend: no one unit here.
            yaxis: { title: { text: "residual", standoff: 4 },
                     type: "log", zeroline: false, tickformat: ".0e",
                     automargin: true, nticks: 5 },
            shapes: shapes, annotations: notes,
        }, theme), { displayModeBar: false });
        return { phases: phases, residuals: residuals.map((x) => x.name) };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.scfPlot = { draw: draw, phasesOf: phasesOf,
                                residualsOf: residualsOf };
})(typeof window !== "undefined" ? window : this);
