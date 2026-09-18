/* transport.js — the deliverable of a conductance run.
 *
 * A finished junction writes `<label>.transport.json` (`transport/record.py`):
 * one entry per bias point, each carrying the transmission curve, the
 * conductance at the Fermi level, and the current.  Until this presenter
 * existed the Results picker matched NOTHING against that file, so
 * `pickResult` returned null and the picker dropped it — the deliverable of a
 * five-rung run was invisible on the tab that exists to show results, while
 * the rungs' own `.out` files listed as five unrelated "SIESTA optimization"
 * entries (`plans/plan.md` § 5p.3p.3, items 1–2).
 *
 * WHAT THIS IS AND IS NOT.  It is the table the record already carries — bias,
 * G(E_F), current — and the honest statement of what is not drawn.  It is NOT
 * the transmission chart: each point holds `energy_ev` and `transmission`, so
 * the DATA is here and only the plot is missing, and this says exactly that
 * rather than the broader claim the step originally called for.
 *
 * It also does NOT make a five-directory ladder read as one run.  `absorbs`
 * collapses siblings within ONE directory and cannot express it; that is
 * § 5c.1's open question and not a presenter's to answer.
 *
 * Reads through `ctx.readFile` — the one shared reader every presenter uses
 * (`web/presenters.md` § 2) — so it opens no route of its own.
 */
(function (root) {
    "use strict";

    function _fmt(x, digits, exp) {
        if (x === null || x === undefined) return "—";
        const n = Number(x);
        if (!isFinite(n)) return "—";
        return exp ? n.toExponential(digits) : n.toFixed(digits);
    }

    function _el(tag, cls, text) {
        const e = root.document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined) e.textContent = text;
        return e;
    }

    function render(host, rec) {
        host.innerHTML = "";
        const wrap = _el("div", "transport-record");

        const pts = Array.isArray(rec.points) ? rec.points : [];
        const pending = Array.isArray(rec.pending) ? rec.pending : [];

        wrap.appendChild(_el("h3", "transport-title",
            "Transport — " + (rec.label || "(unlabelled)")));
        wrap.appendChild(_el("p", "transport-sub",
            pts.length + " bias point" + (pts.length === 1 ? "" : "s")
            + (pending.length ? ", " + pending.length + " pending" : "")));

        const table = _el("table", "transport-iv");
        const head = _el("tr");
        ["Bias [V]", "G(E_F) [G0]", "Current [A]"].forEach((h) => {
            head.appendChild(_el("th", null, h));
        });
        table.appendChild(head);

        pts.forEach((p) => {
            const tr = _el("tr");
            tr.appendChild(_el("td", null, _fmt(p.bias_v, 3, false)));
            tr.appendChild(_el("td", null, _fmt(p.conductance_g0, 4, false)));
            tr.appendChild(_el("td", null, _fmt(p.current_a, 4, true)));
            table.appendChild(tr);
        });
        /* A pending point is a point that has not RUN, not a failure: the
         * record is written by a reader, so a transmission stage still in the
         * queue reads as pending rather than as a broken set. */
        pending.forEach((p) => {
            const tr = _el("tr", "transport-pending");
            tr.appendChild(_el("td", null, _fmt(p.bias_v, 3, false)));
            tr.appendChild(_el("td", null, "pending"));
            tr.appendChild(_el("td", null, p.why || ""));
            table.appendChild(tr);
        });
        wrap.appendChild(table);

        /* THE LADDER, in order.  A transport result is FIVE calculations and
         * the record now says so (`transport/record.py::_stage_facts`); this
         * renders that structure rather than only its last rung.  Sequential
         * dependence means an unfinished rung explains the ones after it, so
         * the honest statement per rung is enough -- no inference, no
         * progress bar. */
        const stages = Array.isArray(rec.stages) ? rec.stages : [];
        if (stages.length) {
            wrap.appendChild(_el("h4", "transport-stages-title", "Stages"));
            const list = _el("div", "transport-stages");
            stages.forEach((st) => {
                const card = _el("div", "transport-stage");
                card.appendChild(_el("div", "transport-stage-name",
                                     st.token ? st.token : st.stage));
                const bits = [];
                if (st.state === "ran") {
                    bits.push(st.scf_converged === true ? "converged"
                            : st.scf_converged === false ? "NOT converged"
                            : "ran");
                    if (st.run_state && st.run_state !== "ended")
                        bits.push(st.run_state);
                    /* The lead's Fermi level is the reference the whole
                     * junction is measured against, so it leads the line for
                     * an electrode rather than trailing the energy. */
                    if (st.fermi_ev !== undefined && st.fermi_ev !== null)
                        bits.unshift("E_F = " + _fmt(st.fermi_ev, 3, false)
                                     + " eV");
                    if (st.energy_ev !== undefined && st.energy_ev !== null)
                        bits.push("E = " + _fmt(st.energy_ev, 4, false)
                                  + " eV");
                } else if (st.state === "not_run") {
                    bits.push("not run yet");
                } else if (st.state === "no_output") {
                    bits.push("prepared, no output yet");
                } else if (st.state === "unreadable") {
                    bits.push("could not be read" + (st.why ? ": " + st.why : ""));
                } else if (st.state === "not_described") {
                    bits.push("not in this description");
                } else {
                    bits.push(String(st.state));
                }
                const body = _el("div", "transport-stage-facts",
                                 bits.join(" \u00b7 "));
                if (st.state !== "ran") body.classList.add("is-pending");
                card.appendChild(body);
                list.appendChild(card);
            });
            wrap.appendChild(list);

            /* TWO LEADS THAT DISAGREE is a defect nothing else on this tab
             * would show: both are periodic bulk runs of the same metal, so
             * their Fermi levels should match.  Said here because this is the
             * only place both numbers appear together. */
            const efs = stages
                .filter((st) => st.fermi_ev !== undefined && st.fermi_ev !== null)
                .map((st) => Number(st.fermi_ev));
            if (efs.length === 2 && isFinite(efs[0]) && isFinite(efs[1])) {
                const d = Math.abs(efs[0] - efs[1]);
                if (d > 0.05) {
                    wrap.appendChild(_el("p", "transport-warn",
                        "The two electrodes' Fermi levels differ by "
                        + _fmt(d, 3, false) + " eV. They are bulk runs of the "
                        + "same lead, so they should agree \u2014 T(E) is "
                        + "measured relative to this."));
                }
            }
        }

        /* Say what is NOT here, and say it narrowly.  The curve's numbers ARE
         * in this file; what is missing is the plot. */
        const withCurve = pts.filter(
            (p) => Array.isArray(p.transmission) && p.transmission.length).length;
        if (withCurve) {
            wrap.appendChild(_el("p", "transport-note",
                "The transmission curve T(E) is in this record for "
                + withCurve + " of " + pts.length + " point"
                + (pts.length === 1 ? "" : "s")
                + " — the chart for it is not built yet."));
        }
        if (rec.energies_relative_to_ef) {
            wrap.appendChild(_el("p", "transport-note",
                "Energies are relative to E_F."));
        }
        host.appendChild(wrap);
    }

    const inspector = {
        name:        "transport",
        displayName: "Transport result",
        isResult:    true,
        resultCategory: () => "Transport",
        /* The record's own spelling (`record.py::record_path`), and specific
         * enough to register ahead of the `.json` catch-all. */
        match: (file) => String(file).toLowerCase().endsWith(".transport.json"),

        mount(host, file, ctx) {
            let disposed = false;
            (async function () {
                let rec;
                try {
                    rec = JSON.parse(await ctx.readFile(file));
                } catch (e) {
                    if (disposed) return;
                    if (ctx && ctx.showError) ctx.showError(String(e));
                    else host.textContent = String(e);
                    return;
                }
                if (!disposed) render(host, rec);
            })();
            return {
                dispose() { disposed = true; host.innerHTML = ""; },
            };
        },
    };

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.transportInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
