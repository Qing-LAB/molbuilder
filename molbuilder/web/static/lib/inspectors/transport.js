/* transport.js — a transport calculation's report (`web/results.md` § 2.5).
 *
 * COMPOSED ON READ (`engines/transport.md` § 2a.12): the record arrives from
 * `/api/transport/record`, which composes it from the rungs as they are now
 * -- every rung's state the one status door's, every point's its run's --
 * never the copy `summarize run` wrote.  It decides nothing about the
 * physics: every number drawn is one the record carries.
 *
 * TWO SELECTIONS, ONE OWNER EACH (the spectra tab's pattern):
 *   * the FRAME -- MolView's model owns it (`data.setCurrentFrame` /
 *     `data.onFrameChange`, `web/molview.md` § 6.4); nothing here keeps a
 *     copy.  A calculation has one frame until the frame axis (plan § 5u.1
 *     step 11), and MolView hides its bar for one frame;
 *   * the BIAS POINT -- this report owns it (`state.bias`): a click on a
 *     T(E) curve, an I-V point or a table row sets it, and T(E), the DOS,
 *     the I-V and the device's convergence follow.
 *
 * Built from what exists: MolView mounted read-only (the structure
 * presenter's door), the one SCF plot (`lib/scfplot/scfplot.js`), the
 * charts' theme (`lib/plot-theme.js`), the page's cards and panel tabs, the
 * ladder's state chip.
 */
import { mount } from "/static/lib/molview/index.js";
import { molviewFiles } from "../projects/molview-doors.js";

/* WHO THESE BYTES BELONG TO (workspace.md § 4): the viewer's `owner`. */
const WORKSPACE_TAG = "results:transport";

(function (root) {
    "use strict";

    const _el = (tag, cls, text) => root.molbuilder.dom.el(tag, cls, text);

    function _fmt(x, digits, exp) {
        if (x === null || x === undefined) return "—";
        const n = Number(x);
        if (!isFinite(n)) return "—";
        return exp ? n.toExponential(digits) : n.toFixed(digits);
    }

    /* THE ONE STATE CHIP -- the ladder's and the bench summary's words and
     * tones (`inspectors.stateChip`). */
    function _chip(state) {
        const reg = (root.molbuilder || {}).inspectors || {};
        if (typeof reg.stateChip === "function") return reg.stateChip(state);
        return _el("span", null, String(state));
    }

    function _dirOf(file) {
        const i = String(file).lastIndexOf("/");
        return i < 0 ? "" : String(file).slice(0, i);
    }

    function _plot(node, traces, layout) {
        if (!root.Plotly) {
            node.textContent = "Plotly is not loaded on this page.";
            return;
        }
        root.Plotly.react(node, traces,
                          root.molbuilder.themedLayout(layout),
                          { displayModeBar: false, responsive: true });
    }

    /* PANEL TABS -- the page's `.panel-tabs` markup (`ui-contract.md`); a
     * panel's charts are resized when it is shown, since a chart drawn
     * hidden has no width. */
    function _tabs(names) {
        const bar = _el("div", "panel-tabs");
        bar.setAttribute("role", "tablist");
        const panels = {};
        const buttons = {};
        const box = _el("div", "transport-tabpanels");
        function show(name) {
            Object.keys(panels).forEach((n) => {
                const on = n === name;
                panels[n].hidden = !on;
                buttons[n].classList.toggle("is-active", on);
                buttons[n].setAttribute("aria-selected", on ? "true" : "false");
            });
            if (root.Plotly) {
                panels[name].querySelectorAll(".js-plotly-plot").forEach((p) => {
                    try { root.Plotly.Plots.resize(p); } catch (_) { /* not drawn */ }
                });
            }
        }
        names.forEach(([key, label]) => {
            const b = _el("button", "panel-tab", label);
            b.type = "button";
            b.setAttribute("role", "tab");
            b.addEventListener("click", () => show(key));
            bar.appendChild(b);
            buttons[key] = b;
            const p = _el("div", "panel-tabpanel");
            p.setAttribute("role", "tabpanel");
            box.appendChild(p);
            panels[key] = p;
        });
        if (names.length) show(names[0][0]);
        return { bar, box, panels, show };
    }

    function _card(title, cls) {
        const c = _el("section", "card transport-card " + (cls || ""));
        c.appendChild(_el("h4", "transport-card-title", title));
        return c;
    }

    /* ------------------------------------------------------------------ */
    /*  The report                                                         */
    /* ------------------------------------------------------------------ */

    function render(host, rec, file, state) {
        host.innerHTML = "";
        const pts = Array.isArray(rec.points) ? rec.points : [];
        const pending = Array.isArray(rec.pending) ? rec.pending : [];
        const failed = Array.isArray(rec.failed) ? rec.failed : [];
        const stages = Array.isArray(rec.stages) ? rec.stages : [];
        state.points = pts;
        if (state.bias === null && pts.length) state.bias = pts[0].bias_v;

        const wrap = _el("div", "transport-record");
        const head = _el("div", "card transport-head");
        head.appendChild(_el("h3", "transport-title",
            "Transport — " + (rec.label || "(unlabelled)")));
        head.appendChild(_el("p", "transport-sub",
            ({"single-bias": "single bias",
              "low-bias": "low-bias approximation",
              "re-converged": "re-converged at every voltage"}[rec.treatment]
             || rec.treatment)
            + " · " + pts.length + " point" + (pts.length === 1 ? "" : "s")
            + " with a transmission"
            + (pending.length ? " · " + pending.length + " pending" : "")
            + (failed.length ? " · " + failed.length + " failed" : "")
            + (rec.energies_relative_to_ef ? " · energies relative to E_F"
                                           : "")));
        wrap.appendChild(head);

        /* ROW 1: the device beside the curves. */
        const row = _el("div", "card-row transport-row");
        const device = _card("Device", "transport-device-card");
        const molHost = _el("div", "molviewer-host transport-molview");
        device.appendChild(molHost);
        device.appendChild(_el("p", "transport-note",
            "Select atoms here (Selection page: pick, by element, by label) "
            + "to add their PDOS in the DOS tab."));
        row.appendChild(device);

        const curves = _card("Results", "transport-curves-card");
        const tabs = _tabs([["te", "T(E)"], ["dos", "DOS"], ["iv", "I–V"]]);
        curves.appendChild(tabs.bar);
        curves.appendChild(tabs.box);
        row.appendChild(curves);
        wrap.appendChild(row);

        state.nodes = {
            te: _el("div", "transport-plot"),
            eig: _el("p", "transport-note"),
            dos: _el("div", "transport-plot"),
            iv: _el("div", "transport-plot"),
            table: null,
        };
        _fillTE(tabs.panels.te, rec, state);
        _fillDOS(tabs.panels.dos, rec, file, state);
        _fillIV(tabs.panels.iv, rec, pending, failed, state);

        /* ROW 2: every rung's convergence. */
        wrap.appendChild(_convergence(stages, state));
        /* ROW 3: the rungs, the provenance, the caveat. */
        wrap.appendChild(_rungs(rec, stages));
        host.appendChild(wrap);

        _mountDevice(molHost, state);
        _redraw(state);
    }

    function _fillTE(panel, rec, state) {
        /* THE CURVE AND ITS TREATMENT, named beside it (`engines/
         * transport.md` § 2a.10, § 2a.12): an I-V read off ONE zero-bias
         * slice is the linear-response approximation; a re-converged scan
         * is not. */
        const NOTE = {
            "single-bias": "One device SCF. An I–V derived from this "
                + "curve is the LINEAR-RESPONSE approximation: integrating a "
                + "zero-bias slice cannot reproduce a resonance entering the "
                + "bias window, nor that resonance moving under the field.",
            "low-bias": "LOW-BIAS (linear-response) approximation: the "
                + "device SCF converged once, at 0 V, and each point's current "
                + "is TBtrans's integral over that point's bias window on the "
                + "zero-bias Hamiltonian — a resonance entering the window "
                + "is not re-converged, nor moved by the field.",
            "re-converged": "The device SCF was re-converged at every "
                + "voltage, so each slice is its own solution and the "
                + "I–V carries no approximation beyond the method.",
        };
        panel.appendChild(_el("p", "transport-note",
            NOTE[rec.treatment] || ("treatment: " + rec.treatment)));
        panel.appendChild(state.nodes.te);
        panel.appendChild(state.nodes.eig);
    }

    function _fillDOS(panel, rec, file, state) {
        panel.appendChild(state.nodes.dos);
        const menu = rec.pdos_orbitals || {};
        const bar = _el("div", "transport-pdos-bar");
        const name = _el("input", "transport-pdos-name");
        name.type = "text";
        name.placeholder = "name (e.g. S, ring C)";
        const orb = _el("select", "transport-pdos-orb");
        (menu.types || ["all"]).forEach((t) => {
            const o = _el("option", null, t === "all" ? "all orbitals" : t);
            o.value = t;
            orb.appendChild(o);
        });
        const add = _el("button", "transport-pdos-add", "Add selection");
        add.type = "button";
        const msg = _el("span", "transport-pdos-msg");
        bar.appendChild(name);
        bar.appendChild(orb);
        bar.appendChild(add);
        bar.appendChild(msg);
        panel.appendChild(bar);
        if (menu.note) panel.appendChild(_el("p", "transport-note", menu.note));
        const list = _el("ul", "transport-pdos-list");
        panel.appendChild(list);
        state.nodes.pdosList = list;
        add.addEventListener("click", () => {
            const atoms = _selection(state);
            if (!atoms.length) {
                msg.textContent = "Select atoms in the device card first.";
                return;
            }
            msg.textContent = "";
            state.pdos.push({
                name: name.value.trim() || ("selection " + (state.pdos.length + 1)),
                atoms: atoms, orbitals: orb.value, curve: null, error: null,
            });
            name.value = "";
            _loadPdos(state);
        });
    }

    function _fillIV(panel, rec, pending, failed, state) {
        panel.appendChild(state.nodes.iv);
        const table = _el("table", "transport-iv");
        const head = _el("tr");
        /* THE CURRENT IS THE JUNCTION'S TOTAL, both spin channels, beside
         * the figure TBtrans printed -- one channel's (`engines/
         * transport.md` § 2a.4; `record.py` CURRENT_MEANS). */
        ["Bias [V]", "G(E_F) [G0]", "Current, total [A]",
         "TBtrans printed [A]"].forEach((h) => head.appendChild(_el("th", null, h)));
        table.appendChild(head);
        state.points.forEach((p) => {
            const tr = _el("tr", "transport-iv-row");
            tr.dataset.bias = String(p.bias_v);
            tr.appendChild(_el("td", null, _fmt(p.bias_v, 3, false)));
            tr.appendChild(_el("td", null, _fmt(p.conductance_g0, 4, false)));
            tr.appendChild(_el("td", null, _fmt(p.current_a, 4, true)));
            tr.appendChild(_el("td", null, _fmt(p.current_a_printed, 4, true)));
            tr.addEventListener("click", () => _select(state, p.bias_v));
            table.appendChild(tr);
        });
        /* A POINT WITHOUT ITS TRANSMISSION, in its run's words: pending is
         * still to come, never a failure; failed ended without it. */
        pending.concat(failed).forEach((p) => {
            const tr = _el("tr", "transport-pending");
            tr.appendChild(_el("td", null, _fmt(p.bias_v, 3, false)));
            const st = _el("td");
            st.appendChild(_chip(p.state));
            tr.appendChild(st);
            const why = _el("td", null, p.why || "");
            why.setAttribute("colspan", "2");
            tr.appendChild(why);
            table.appendChild(tr);
        });
        panel.appendChild(table);
        state.nodes.table = table;
        Object.keys(rec.current_means || {}).sort().forEach((spin) => {
            panel.appendChild(_el("p", "transport-note",
                                  spin + ": " + rec.current_means[spin]));
        });
    }

    /* EVERY RUNG'S SCF, one tab each, through the one SCF plot: a device's
     * periodic start and its NEGF loop are separate traces.  A scan's
     * rung shows the selected bias point's run. */
    function _convergence(stages, state) {
        const card = _card("Convergence", "transport-conv-card");
        const withScf = stages.filter((st) =>
            (Array.isArray(st.scf) && st.scf.length) || Array.isArray(st.by_point));
        if (!withScf.length) {
            card.appendChild(_el("p", "transport-note",
                "No rung has an SCF to draw yet."));
            return card;
        }
        const tabs = _tabs(withScf.map((st) => [st.stage,
                                                String(st.stage).replace("_", " ")]));
        card.appendChild(tabs.bar);
        card.appendChild(tabs.box);
        state.conv = {};
        withScf.forEach((st) => {
            const panel = tabs.panels[st.stage];
            const line = _el("div", "transport-conv-line");
            const pair = _el("div", "transport-conv-plots");
            const e = _el("div", "transport-plot transport-plot-half");
            const r = _el("div", "transport-plot transport-plot-half");
            pair.appendChild(e);
            pair.appendChild(r);
            panel.appendChild(line);
            panel.appendChild(pair);
            state.conv[st.stage] = { st, line, e, r };
        });
        return card;
    }

    function _drawConvergence(state) {
        Object.values(state.conv || {}).forEach(({ st, line, e, r }) => {
            /* A SCAN'S RUNG speaks for the selected bias point. */
            const src = Array.isArray(st.by_point)
                ? (st.by_point.find((p) => p.bias_v === state.bias)
                   || st.by_point[0])
                : st;
            line.replaceChildren();
            line.appendChild(_chip(src.state || st.state));
            const bits = [];
            if (src.bias_v !== undefined && src.bias_v !== null)
                bits.push(_fmt(src.bias_v, 3, false) + " V");
            if (src.detail) bits.push(src.detail);
            if (src.negf) {
                bits.push("NEGF: " + src.negf.cycles + " cycles");
                if (src.negf.ef !== undefined)
                    bits.push("E_F " + _fmt(src.negf.ef, 3, false) + " eV");
                if (src.negf.dq !== undefined)
                    bits.push("dQ " + _fmt(src.negf.dq, 2, true));
            }
            line.appendChild(_el("span", "transport-stage-detail",
                                 " " + bits.join(" · ")));
            const rows = Array.isArray(src.scf) ? src.scf : [];
            e.hidden = !rows.length;
            r.hidden = !rows.length;
            if (rows.length) {
                root.molbuilder.scfPlot.draw(e, r, rows,
                                             { criteria: src.scf_criteria || {} });
            }
        });
    }

    function _rungs(rec, stages) {
        const card = _card("Rungs and provenance", "transport-rungs-card");
        const table = _el("table", "transport-rungs");
        const head = _el("tr");
        ["Rung", "State", "E_F [eV]", "E [eV]", "SCF", "Attempt"].forEach(
            (h) => head.appendChild(_el("th", null, h)));
        table.appendChild(head);
        stages.forEach((st) => {
            const tr = _el("tr");
            tr.appendChild(_el("td", null, st.token || st.stage));
            const s = _el("td");
            if (st.state === "not_described") {
                s.textContent = "not in this description";
            } else {
                s.appendChild(_chip(st.state));
                if (st.detail) s.appendChild(_el("span", "transport-stage-detail",
                                                 " " + st.detail));
            }
            tr.appendChild(s);
            /* A DEVICE'S E_F IS ITS NEGF PHASE'S, never the periodic
             * start's (§ 2a.12); a lead's is its own. */
            const ef = st.negf && st.negf.ef !== undefined ? st.negf.ef : st.fermi_ev;
            tr.appendChild(_el("td", null, _fmt(ef, 3, false)));
            tr.appendChild(_el("td", null, _fmt(st.energy_ev, 4, false)));
            tr.appendChild(_el("td", null,
                st.scf_converged === true ? "converged"
                : st.scf_converged === false ? "NOT converged"
                : (st.unreadable ? "unreadable: " + st.unreadable : "—")));
            tr.appendChild(_el("td", null, st.attempt || "—"));
            table.appendChild(tr);
        });
        card.appendChild(table);

        /* TWO LEADS THAT DISAGREE is a defect nothing else shows: both are
         * bulk runs of the same metal. */
        const efs = stages.filter((st) => st.fermi_ev !== undefined
                                          && st.fermi_ev !== null)
                          .map((st) => Number(st.fermi_ev));
        if (efs.length === 2 && isFinite(efs[0]) && isFinite(efs[1])
                && Math.abs(efs[0] - efs[1]) > 0.05) {
            card.appendChild(_el("p", "transport-warn",
                "The two electrodes' Fermi levels differ by "
                + _fmt(Math.abs(efs[0] - efs[1]), 3, false) + " eV. They are "
                + "bulk runs of the same lead, so they should agree — "
                + "T(E) is measured relative to this."));
        }

        /* THE CHAIN -- what each rung's attempt was gathered from
         * (`.gathered-from`; § 2a.12: "A transmission curve without its
         * chain cannot be interpreted, reproduced, or compared"). */
        const chain = ((rec.provenance || {}).chain) || [];
        if (chain.length) {
            card.appendChild(_el("h5", "transport-prov-title", "What each rung took"));
            const ul = _el("ul", "transport-chain");
            chain.forEach((c) => {
                const li = _el("li", null, c.attempt + ": ");
                li.appendChild(_el("span", "transport-prov-val",
                    c.gathered.length
                        ? c.gathered.map((g) => g.file + " ← " + g.from).join(", ")
                        : "nothing gathered"));
                ul.appendChild(li);
            });
            card.appendChild(ul);
        }
        const slot = (rec.provenance || {}).slot;
        if (slot) card.appendChild(_slot(slot));
        if (rec.caveat) card.appendChild(_el("p", "transport-caveat", rec.caveat));
        return card;
    }

    /* THE CITATION: which relaxation the junction came from, in what form,
     * whether it had concluded, and the bytes it was built from. */
    function _slot(prov) {
        const dl = _el("div", "transport-prov");
        function row(k, v, cls) {
            const r = _el("div", "transport-prov-row");
            r.appendChild(_el("span", "transport-prov-key", k));
            r.appendChild(_el("span", cls || "transport-prov-val", String(v)));
            dl.appendChild(r);
        }
        if (prov.citation) row("cited run", prov.citation);
        if (prov.form) row("cited as", prov.form);
        /* EVIDENCE IS THE HONEST FIELD: "no-record" when a relaxation's .XV
         * was taken as final without a concluding record; "given" for a
         * cited structure pair that never claimed to be relaxed. */
        if (prov.evidence !== undefined && prov.evidence !== null) {
            const weak = prov.evidence === "no-record" || prov.evidence === "given";
            row("relaxation evidence",
                prov.evidence === "no-record"
                    ? "no concluding record — the .XV was taken as final"
                    : prov.evidence === "given"
                        ? "a cited structure pair, not a relaxation"
                        : prov.evidence,
                weak ? "transport-prov-val is-weak" : "transport-prov-val");
        }
        const files = prov.files || {};
        const names = Object.keys(files);
        if (names.length) {
            row("built from", names.length + " file" + (names.length === 1 ? "" : "s"));
            const fl = _el("ul", "transport-prov-files");
            names.sort().forEach((n) => {
                const li = _el("li", "transport-prov-file");
                li.appendChild(_el("span", "transport-prov-fname", n));
                const h = String(files[n] || "");
                const short = _el("span", "transport-prov-hash",
                                  h ? h.slice(0, 12) : "—");
                if (h) short.title = h;
                li.appendChild(short);
                fl.appendChild(li);
            });
            dl.appendChild(fl);
        }
        return dl;
    }

    /* ------------------------------------------------------------------ */
    /*  The selections                                                     */
    /* ------------------------------------------------------------------ */

    function _select(state, bias) {
        if (state.bias === bias) return;
        state.bias = bias;
        _loadPdos(state);
    }

    /* THE ATOMS SELECTED in the device card -- MolView's own selection,
     * read through its handle; nothing here keeps a copy. */
    function _selection(state) {
        const h = state.handle;
        const sel = h && h.data && h.data.selection
            && typeof h.data.selection.get === "function"
            ? h.data.selection.get() : null;
        if (!sel) return [];
        const arr = Array.isArray(sel) ? sel : Array.from(sel);
        return arr.map((a) => Number(a)).filter((a) => Number.isInteger(a))
                  .sort((a, b) => a - b);
    }

    function _point(state) {
        return state.points.find((p) => p.bias_v === state.bias) || null;
    }

    function _redraw(state) {
        const pts = state.points;
        const p = _point(state);
        const t = root.molbuilder.plotTheme();
        const col = (i) => t.palette[i % t.palette.length];

        /* T(E): one curve per bias, the selected one bold, its eigenchannels
         * beneath it. */
        const te = [];
        pts.forEach((q, i) => {
            if (!Array.isArray(q.transmission) || !q.transmission.length) return;
            const on = q.bias_v === state.bias;
            te.push({ x: q.energy_ev, y: q.transmission, mode: "lines",
                      name: _fmt(q.bias_v, 3, false) + " V",
                      line: { color: col(i), width: on ? 2.5 : 1 },
                      opacity: on ? 1 : 0.55, meta: q.bias_v });
        });
        const eig = (p && p.dos && Array.isArray(p.dos.eigenchannels))
            ? p.dos.eigenchannels : [];
        eig.forEach((ch, k) => te.push({
            x: p.dos.energy_ev, y: ch, mode: "lines",
            name: "channel " + (k + 1),
            line: { color: t.muted, width: 1, dash: "dot" } }));
        state.nodes.eig.textContent = p && !eig.length
            ? "No eigenchannels for this point" + (p.dos_why ? ": " + p.dos_why : ".")
            : "";
        if (te.length) {
            _plot(state.nodes.te, te, {
                margin: { l: 8, r: 12, t: 8, b: 34 }, height: 340,
                xaxis: { title: { text: "E − E_F (eV)", standoff: 4 },
                         automargin: true, zeroline: false },
                yaxis: { title: { text: "T", standoff: 4 }, type: "log",
                         automargin: true, exponentformat: "power" },
                legend: { orientation: "h", y: -0.2 },
            });
            _onClick(state.nodes.te, state);
        } else {
            state.nodes.te.textContent = "No transmission has run yet.";
        }

        /* DOS: the selected point's total, its parts by region, the leads',
         * and every selection added. */
        const dos = p && p.dos;
        if (!dos || !Array.isArray(dos.energy_ev)) {
            state.nodes.dos.textContent = p
                ? "No DOS for this point" + (p.dos_why ? ": " + p.dos_why : ".")
                : "No transmission has run yet.";
        } else {
            const tr = [];
            let i = 0;
            if (dos.total) tr.push({ x: dos.energy_ev, y: dos.total, mode: "lines",
                name: "device (total)", line: { color: t.ink, width: 2 } });
            Object.keys(dos.by_label || {}).forEach((lab) => tr.push({
                x: dos.energy_ev, y: dos.by_label[lab], mode: "lines",
                name: lab, line: { color: col(i++), width: 1.5 } }));
            Object.keys(dos.lead_spectral || {}).forEach((e) => tr.push({
                x: dos.energy_ev, y: dos.lead_spectral[e], mode: "lines",
                name: "spectral " + e, visible: "legendonly",
                line: { color: col(i++), width: 1, dash: "dash" } }));
            Object.keys(dos.lead_bulk || {}).forEach((e) => tr.push({
                x: dos.energy_ev, y: dos.lead_bulk[e], mode: "lines",
                name: "bulk " + e, visible: "legendonly",
                line: { color: col(i++), width: 1, dash: "dot" } }));
            state.pdos.forEach((s) => {
                if (s.curve) tr.push({ x: s.curve.energy_ev, y: s.curve.pdos,
                    mode: "lines", name: s.name + " (" + s.orbitals + ")",
                    line: { color: col(i++), width: 2 } });
            });
            _plot(state.nodes.dos, tr, {
                margin: { l: 8, r: 12, t: 8, b: 34 }, height: 340,
                xaxis: { title: { text: "E − E_F (eV)", standoff: 4 },
                         automargin: true, zeroline: false },
                yaxis: { title: { text: "DOS (states/eV)", standoff: 4 },
                         automargin: true },
                legend: { orientation: "h", y: -0.2 },
            });
        }
        _drawPdosList(state);

        /* I-V: the curve, its points clickable, the selected one marked. */
        const iv = pts.filter((q) => q.current_a !== null && q.current_a !== undefined);
        if (iv.length) {
            _plot(state.nodes.iv, [{
                x: iv.map((q) => q.bias_v), y: iv.map((q) => q.current_a),
                mode: "lines+markers", name: "I (total)",
                line: { color: t.accent, width: 1.5 },
                marker: { size: iv.map((q) => (q.bias_v === state.bias ? 12 : 6)) },
                meta: iv.map((q) => q.bias_v),
            }], {
                margin: { l: 8, r: 12, t: 8, b: 34 }, height: 300,
                xaxis: { title: { text: "Bias (V)", standoff: 4 }, automargin: true },
                yaxis: { title: { text: "Current (A)", standoff: 4 },
                         automargin: true, exponentformat: "e" },
                showlegend: false,
            });
            _onClick(state.nodes.iv, state);
        } else {
            state.nodes.iv.textContent = "No current yet.";
        }
        if (state.nodes.table) {
            state.nodes.table.querySelectorAll(".transport-iv-row").forEach((tr) => {
                tr.classList.toggle("is-selected", Number(tr.dataset.bias) === state.bias);
            });
        }
        _drawConvergence(state);
    }

    /* A CLICK ON A CURVE OR A POINT selects its bias -- each trace carries
     * its bias as `meta`.  Bound once per chart. */
    function _onClick(node, state) {
        if (node.dataset.bound || typeof node.on !== "function") return;
        node.dataset.bound = "1";
        node.on("plotly_click", (ev) => {
            const pt = ev && ev.points && ev.points[0];
            if (!pt) return;
            const m = pt.data.meta;
            const bias = Array.isArray(m) ? m[pt.pointIndex] : m;
            if (typeof bias === "number") _select(state, bias);
        });
    }

    function _drawPdosList(state) {
        const ul = state.nodes.pdosList;
        if (!ul) return;
        ul.replaceChildren();
        state.pdos.forEach((s, k) => {
            const li = _el("li", "transport-pdos-item",
                s.name + " — " + s.atoms.length + " atom"
                + (s.atoms.length === 1 ? "" : "s") + ", " + s.orbitals
                + (s.error ? " — " + s.error : ""));
            const rm = _el("button", "transport-pdos-remove", "remove");
            rm.type = "button";
            rm.addEventListener("click", () => {
                state.pdos.splice(k, 1);
                _redraw(state);
            });
            li.appendChild(rm);
            ul.appendChild(li);
        });
    }

    /* EACH SELECTION'S PDOS AT THE SELECTED POINT, asked of the server
     * (`/api/transport/pdos`) -- nothing is summed here. */
    async function _loadPdos(state) {
        const bias = state.bias;
        if (bias !== null && state.pdos.length) {
            await Promise.all(state.pdos.map(async (s) => {
                try {
                    const q = "/api/transport/pdos?path="
                        + encodeURIComponent(state.file)
                        + "&point=" + encodeURIComponent(bias)
                        + "&atoms=" + s.atoms.join(",")
                        + "&orbitals=" + encodeURIComponent(s.orbitals);
                    const body = await (await fetch(q)).json();
                    if (body && body.ok) {
                        s.curve = body;
                        s.error = body.outside_device && body.outside_device.length
                            ? body.outside_device.length + " atom(s) outside the "
                              + "device region are not counted" : null;
                    } else {
                        s.curve = null;
                        s.error = (body && body.error) || "no answer";
                    }
                } catch (e) {
                    s.curve = null;
                    s.error = String(e);
                }
            }));
        }
        if (state.disposed || bias !== state.bias) return;
        _redraw(state);
    }

    /* THE DEVICE, read-only -- the structure presenter's door: MolView
     * mounted with its own owner, the composed junction opened through the
     * one file door (its sidecar brings the labels and the cell). */
    async function _mountDevice(molHost, state) {
        const ws = root.molbuilder && root.molbuilder.workspace;
        const proj = root.molbuilder && root.molbuilder.projects;
        if (!ws || !proj || !proj.parser) {
            molHost.textContent = "The 3D viewer is not available on this page.";
            return;
        }
        let handle;
        try {
            handle = await mount(molHost, ws, { mode: "readonly",
                                                owner: WORKSPACE_TAG,
                                                files: molviewFiles });
        } catch (e) {
            molHost.textContent = "Viewer failed: " + e;
            return;
        }
        if (state.disposed) {
            try { handle && handle.dispose && handle.dispose(); } catch (_) { /* gone */ }
            return;
        }
        if (!handle || !handle.ok) {
            molHost.textContent = "Viewer failed: "
                + ((handle && handle.error) || "molview.mount failed.");
            return;
        }
        state.handle = handle;
        const res = await proj.parser.openMolecule(
            handle, (state.dir ? state.dir + "/" : "") + "junction.xyz");
        if (res && res.ok === false) {
            molHost.appendChild(_el("p", "transport-note",
                "The composed junction could not be opened: " + (res.error || "")));
        }
        /* THE FRAME'S ONE OWNER is the model; the report follows it.  One
         * frame until the frame axis -- the record carries no frame yet. */
        if (handle.data && typeof handle.data.onFrameChange === "function") {
            state.unsubFrame = handle.data.onFrameChange(() => _redraw(state));
        }
    }

    /* ------------------------------------------------------------------ */
    /*  The presenter                                                      */
    /* ------------------------------------------------------------------ */

    const inspector = {
        name:        "transport",
        displayName: "Transport result",
        isResult:    true,
        resultCategory: () => "Transport",
        // THE ROLE, when the server gave one (see the spectra inspector).
        match: (file, meta) => (meta && meta.role)
            ? meta.role === ".transport.json"
            : String(file).toLowerCase().endsWith(".transport.json"),

        mount(host, file, ctx) {
            const state = { bias: null, points: [], pdos: [], nodes: {},
                            conv: null, handle: null, unsubFrame: null,
                            disposed: false, file: file, dir: _dirOf(file) };
            (async function () {
                let rec;
                try {
                    const r = await fetch("/api/transport/record?path="
                                          + encodeURIComponent(file));
                    const body = await r.json();
                    if (!body || !body.ok) {
                        throw new Error("could not read " + file + ": "
                                        + ((body && body.error) || "unknown"));
                    }
                    rec = body.record;
                } catch (e) {
                    if (state.disposed) return;
                    if (ctx && ctx.showError) ctx.showError(String(e));
                    else host.textContent = String(e);
                    return;
                }
                if (state.disposed) return;
                render(host, rec, file, state);
                /* FIRST RENDER ON SCREEN: the picker drops its "Parsing..."
                 * line on this event (`lib/results/file-picker.js`). */
                root.document.dispatchEvent(new root.CustomEvent(
                    ((root.molbuilder || {}).constants || {})
                        .EVENT_INSPECTOR_READY || "molbuilder:inspector:ready",
                    { detail: { inspector: "transport" } }));
            })();
            return {
                dispose() {
                    state.disposed = true;
                    try { state.unsubFrame && state.unsubFrame(); } catch (_) { /* gone */ }
                    try { state.handle && state.handle.dispose(); } catch (_) { /* gone */ }
                    if (root.Plotly) {
                        host.querySelectorAll(".js-plotly-plot").forEach((p) => {
                            try { root.Plotly.purge(p); } catch (_) { /* never drawn */ }
                        });
                    }
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
