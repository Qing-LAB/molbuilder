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
