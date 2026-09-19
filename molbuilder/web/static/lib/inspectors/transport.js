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

        /* THE CURVE -- the deliverable.  `engines/transport.md` 2a.12: the
         * surface presents T(E) for a single bias, or the family T(E, V) for
         * a scan, and NAMES ITS TREATMENT beside it. */
        const curves = pts.filter(
            (p) => Array.isArray(p.transmission) && p.transmission.length
                   && Array.isArray(p.energy_ev) && p.energy_ev.length);
        if (curves.length) {
            /* The label is the RECORD's, not this file's guess.  2a.10: the
             * mechanism is the same either way and what differs is how many
             * device SCFs were paid for -- so an I-V read off ONE zero-bias
             * slice is the linear-response approximation, and one from a
             * re-converged scan is not.  2a.12 puts that beside the curve
             * rather than in metadata, "because the two kinds of I-V are
             * different claims and look identical on a plot". */
            const single = rec.treatment === "single-bias";
            wrap.appendChild(_el("h4", "transport-curve-title",
                single ? "Transmission T(E) \u2014 single bias"
                       : "Transmission T(E, V) \u2014 finite bias, "
                         + curves.length + " slices"));
            wrap.appendChild(_el("p", "transport-note", single
                ? "One device SCF. An I\u2013V derived from this curve is the "
                  + "LINEAR-RESPONSE approximation: integrating a zero-bias "
                  + "slice cannot reproduce a resonance entering the bias "
                  + "window, nor that resonance moving under the field."
                : "The device SCF was re-converged at every voltage, so each "
                  + "slice is its own solution and the I\u2013V carries no "
                  + "approximation beyond the method."));
            const plot = _el("div", "transport-curve");
            wrap.appendChild(plot);
            _plotLater(plot, curves);
        }
        /* THE PROVENANCE CHAIN -- `engines/transport.md` § 2a.12, the third
         * thing that section requires of this surface and the one it is
         * bluntest about: *"A transmission curve without its chain cannot be
         * interpreted, reproduced, or compared with another."*
         *
         * Which relaxation the junction came from, what form it was cited in,
         * whether that relaxation had actually CONCLUDED, and the bytes it was
         * built from.  The hashes are not decoration: a result can always say
         * which files produced it, and two curves that disagree can be told
         * apart by whether they were built from the same junction at all. */
        const prov = (rec.provenance || {}).slot;
        if (prov) {
            wrap.appendChild(_el("h4", "transport-prov-title", "Provenance"));
            const dl = _el("div", "transport-prov");

            function row(k, v, cls) {
                const r = _el("div", "transport-prov-row");
                r.appendChild(_el("span", "transport-prov-key", k));
                r.appendChild(_el("span", cls || "transport-prov-val",
                                  String(v)));
                dl.appendChild(r);
            }

            if (prov.citation) row("cited run", prov.citation);
            if (prov.form) row("cited as", prov.form);

            /* EVIDENCE IS THE HONEST FIELD.  `compose` writes the concluding
             * record line when one exists; "no-record" when a relaxation's
             * .XV was taken as final without one; "given" for a cited
             * structure pair that never claimed to be relaxed.  The last two
             * are not failures -- they are what the citation actually
             * offered -- but a reader judging a curve needs to see which. */
            if (prov.evidence !== undefined && prov.evidence !== null) {
                const weak = prov.evidence === "no-record"
                          || prov.evidence === "given";
                row("relaxation evidence",
                    prov.evidence === "no-record"
                        ? "no concluding record \u2014 the .XV was taken as final"
                        : prov.evidence === "given"
                            ? "a cited structure pair, not a relaxation"
                            : prov.evidence,
                    weak ? "transport-prov-val is-weak" : "transport-prov-val");
            }

            const files = prov.files || {};
            const names = Object.keys(files);
            if (names.length) {
                row("built from", names.length + " file"
                    + (names.length === 1 ? "" : "s"));
                const fl = _el("ul", "transport-prov-files");
                names.sort().forEach((n) => {
                    const li = _el("li", "transport-prov-file");
                    li.appendChild(_el("span", "transport-prov-fname", n));
                    /* Short enough to compare by eye, long enough to be a
                     * hash: the full value is on the element for a copy. */
                    const h = String(files[n] || "");
                    const short = _el("span", "transport-prov-hash",
                                      h ? h.slice(0, 12) : "\u2014");
                    if (h) short.title = h;
                    li.appendChild(short);
                    fl.appendChild(li);
                });
                dl.appendChild(fl);
            }
            wrap.appendChild(dl);
        }

        if (rec.energies_relative_to_ef) {
            wrap.appendChild(_el("p", "transport-note",
                "Energies are relative to E_F."));
        }
        host.appendChild(wrap);
    }

    /* Colours from the tokens, exactly as `lib/trajectory/core.js` does it --
     * theme-responsive, with a literal fallback for a headless render. */
    function _themeColors() {
        const cs = root.getComputedStyle
            ? root.getComputedStyle(root.document.documentElement) : null;
        const get = (n, fb) =>
            ((cs && cs.getPropertyValue(n)) || "").trim() || fb;
        return { textMuted: get("--text-muted", "#6c7280") };
    }

    /* One Plotly call, the same shape every other plot in this tree uses
     * (`trajectory/core.js`, `spectra/core.js`, `spectrumchart/_seal.js`):
     * `Plotly.react(node, traces, layout, {displayModeBar:false,
     * responsive:true})`.  There is no wrapper to reach for -- eight call
     * sites, no abstraction over them -- so this follows the house pattern
     * rather than inventing a ninth shape. */
    function _plotLater(el, curves) {
        /* A tick later: the caller appends `el` to the host right after this
         * returns, and Plotly measures a node that must already be laid out. */
        root.setTimeout(function () {
            if (!root.Plotly || (el.isConnected === false)) return;
            const theme = _themeColors();
            const traces = curves.map((p) => ({
                x: p.energy_ev,
                y: p.transmission,
                mode: "lines",
                line: { width: 1.5 },
                name: curves.length === 1
                    ? "T(E)" : (Number(p.bias_v).toFixed(3) + " V"),
                connectgaps: false,
            }));
            root.Plotly.react(el, traces, {
                margin: { l: 8, r: 12, t: 12, b: 32 },
                showlegend: curves.length > 1,
                legend: { font: { size: 9 } },
                xaxis: {
                    /* Relative to E_F -- which is the LEAD's, and is why the
                     * electrode cards above carry it. */
                    title: { text: "E \u2212 E_F (eV)", standoff: 4 },
                    zeroline: true, zerolinecolor: theme.textMuted,
                    automargin: true, nticks: 7,
                },
                yaxis: {
                    /* LOG.  Transmission spans orders of magnitude -- a
                     * molecular junction runs 1e-6 to 1 -- and on a linear
                     * axis the whole curve reads as a flat line on zero.
                     * Plotly drops non-positive points on a log axis, which
                     * is the honest rendering of T = 0 rather than a floor
                     * invented to make the plot look continuous. */
                    title: { text: "T", standoff: 4 },
                    type: "log", automargin: true, nticks: 5,
                },
                font: { family: "system-ui, sans-serif", size: 10 },
            }, { displayModeBar: false, responsive: true });
        }, 0);
    }

    const inspector = {
        name:        "transport",
        displayName: "Transport result",
        isResult:    true,
        resultCategory: () => "Transport",
        /* The record's own spelling (`record.py::record_path`), and specific
         * enough to register ahead of the `.json` catch-all. */
        // THE ROLE, when the server gave one (see the spectra inspector).
        match: (file, meta) => (meta && meta.role)
            ? meta.role === ".transport.json"
            : String(file).toLowerCase().endsWith(".transport.json"),

        mount(host, file, ctx) {
            let disposed = false;
            (async function () {
                let rec;
                try {
                    // ctx.readFile answers an envelope {ok, text, error?},
                    // not a string (registry.js).
                    const body = await ctx.readFile(file);
                    if (!body || !body.ok) {
                        throw new Error(
                            "could not read " + file + ": " +
                            ((body && body.error) || "unknown"));
                    }
                    rec = JSON.parse(body.text || "");
                } catch (e) {
                    if (disposed) return;
                    if (ctx && ctx.showError) ctx.showError(String(e));
                    else host.textContent = String(e);
                    return;
                }
                if (!disposed) render(host, rec);
            })();
            return {
                dispose() {
                    disposed = true;
                    try {
                        const c = host.querySelector
                            && host.querySelector(".transport-curve");
                        if (c && root.Plotly) root.Plotly.purge(c);
                    } catch (_) { /* it may never have been drawn */ }
                    host.innerHTML = "";
                },
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
