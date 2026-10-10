/* transport.js — a transport calculation's report (`web/results.md` § 2.5).
 *
 * COMPOSED ON READ (`engines/transport.md` § 2a.12): the record arrives from
 * `/api/transport/record`, which composes it from the rungs as they are now
 * -- every rung's state the one status door's, every point's its run's --
 * never the copy `summarize task` wrote.  It decides nothing about the
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

    /* THE ONE STATE CHIP (`lib/state-chip.js`). */
    function _chip(state) {
        return root.molbuilder.stateChip(state);
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
    let _tabsSeq = 0;
    function _tabs(names) {
        const bar = _el("div", "panel-tabs");
        bar.setAttribute("role", "tablist");
        const panels = {};
        const buttons = {};
        const box = _el("div", "transport-tabpanels");
        // The ARIA tabs pattern (ui-contract.md § 4.1): ids pair each tab
        // with its panel, one tab stop, the arrows move the selection.
        const prefix = "transport-tab-" + (++_tabsSeq) + "-";
        const keys = names.map((n) => n[0]);
        function show(name) {
            Object.keys(panels).forEach((n) => {
                const on = n === name;
                panels[n].hidden = !on;
                buttons[n].classList.toggle("is-active", on);
                buttons[n].setAttribute("aria-selected", on ? "true" : "false");
                buttons[n].tabIndex = on ? 0 : -1;
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
            b.id = prefix + key;
            b.setAttribute("role", "tab");
            b.setAttribute("aria-controls", prefix + key + "-panel");
            b.addEventListener("click", () => show(key));
            b.addEventListener("keydown", (ev) => {
                const i = keys.indexOf(key);
                const to = ev.key === "ArrowRight" ? keys[(i + 1) % keys.length]
                    : ev.key === "ArrowLeft" ? keys[(i - 1 + keys.length) % keys.length]
                    : ev.key === "Home" ? keys[0]
                    : ev.key === "End" ? keys[keys.length - 1] : null;
                if (!to) return;
                ev.preventDefault();
                show(to);
                buttons[to].focus();
            });
            bar.appendChild(b);
            buttons[key] = b;
            const p = _el("div", "panel-tabpanel");
            p.id = prefix + key + "-panel";
            p.setAttribute("role", "tabpanel");
            p.setAttribute("aria-labelledby", prefix + key);
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
        state.iv = rec.iv || {};
        if (state.bias === null && pts.length) state.bias = pts[0].bias_v;

        const wrap = _el("div", "transport-record");
        const head = _el("div", "card transport-head");
        head.appendChild(_el("h3", "transport-title",
            "Transport — " + (rec.label || "(unlabelled)")));
        head.appendChild(_el("p", "transport-sub",
            rec.treatment + " · " + pts.length + " point" + (pts.length === 1 ? "" : "s")
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

        _mountDevice(molHost, state, rec.junction_file);
        _redraw(state);
    }

    function _fillTE(panel, rec, state) {
        /* THE CURVE AND ITS TREATMENT, named beside it (`engines/
         * transport.md` § 2a.10, § 2a.12) -- the record's own words
         * (`record.TREATMENT_NOTE`), which `summarize` prints too. */
        panel.appendChild(_el("p", "transport-note", rec.treatment_note));
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
        name.setAttribute("aria-label", "name for the selection");
        const orb = _el("select", "transport-pdos-orb");
        orb.setAttribute("aria-label", "orbital type");
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
        /* THE TREATMENT, NAMED BESIDE THE I-V (`engines/transport.md`
         * § 2a.10, § 2a.12): the two kinds of I-V look identical on a plot,
         * so the record's own words stand on this tab too. */
        panel.appendChild(_el("p", "transport-note", rec.treatment_note));
        panel.appendChild(state.nodes.iv);
        const table = _el("table", "transport-iv");
        const head = _el("tr");
        const iv = rec.iv || {};
        const computed = iv.computed === "linear-response";
        /* THE CURRENT IS THE JUNCTION'S TOTAL, both spin channels, beside
         * the figure TBtrans printed -- one channel's (`engines/
         * transport.md` § 2a.4; `record.py` CURRENT_MEANS).  Under the
         * low-bias approximation the record computed it from T(E, 0) for
         * each listed voltage (`record.linear_response_iv`). */
        ["Bias [V]", "G(E_F) [G0]",
         computed ? "Current, computed from T(E, 0) [A]" : "Current, total [A]",
         // Under the approximation TBtrans ran at 0 V only: a dash at the
         // other voltages means not applicable, and the header says so.
         computed ? "TBtrans printed [A] (0 V only)" : "TBtrans printed [A]"
        ].forEach((h) => head.appendChild(_el("th", null, h)));
        table.appendChild(head);
        if (computed) {
            const zero = state.points.find((p) => Math.abs(p.bias_v) < 1e-9);
            (iv.voltages_v || []).forEach((v, k) => {
                const tr = _el("tr", "transport-iv-row");
                tr.dataset.bias = String(v);
                tr.appendChild(_el("td", null, _fmt(v, 3, false)));
                tr.appendChild(_el("td", null,
                    Math.abs(v) < 1e-9 && zero ? _fmt(zero.conductance_g0, 4, false) : "—"));
                const i = (iv.current_a || [])[k];
                const note = (iv.notes || {})[String(Number(v))];
                tr.appendChild(_el("td", null, i === null || i === undefined
                    ? (note || "—") : _fmt(i, 4, true)));
                tr.appendChild(_el("td", null,
                    _fmt(((iv.current_a_printed || [])[k]), 4, true)));
                table.appendChild(tr);
            });
        }
        state.points.forEach((p) => {
            if (computed) return;
            const tr = _el("tr", "transport-iv-row is-pick");
            tr.dataset.bias = String(p.bias_v);
            // THE PICK IS A BUTTON (ui-contract.md § 4.1): the bias cell
            // names the point and takes the keyboard; the row's click is
            // the mouse's wider target for the same pick.
            const pick = _el("button", "transport-iv-pick", _fmt(p.bias_v, 3, false));
            pick.type = "button";
            pick.setAttribute("aria-label", "show the " + _fmt(p.bias_v, 3, false) + " V point");
            const biasCell = _el("td", null);
            biasCell.appendChild(pick);
            tr.appendChild(biasCell);
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
     * periodic start and its NEGF loop are separate traces.  A sweep's
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
                                                String(st.stage)]));
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
            /* THE RUN THE TRANSMISSION GATHERED answers for the device
             * (§ 2a.12): said beside its attempt, with the contour its
             * density was integrated on. */
            const where = (st.attempt || "—")
                + (st.gathered_by ? " · gathered by the " + st.gathered_by : "")
                + (st.negf && st.negf.contour && st.negf.contour.poles
                    ? " · " + st.negf.contour.poles + " poles" : "")
                // THE POINT THE ROW'S NUMBERS COME FROM (results.md § 2.4).
                + (typeof st.facts_at_v === "number"
                    ? " · at " + st.facts_at_v + " V" : "");
            tr.appendChild(_el("td", null, where));
            table.appendChild(tr);
        });
        card.appendChild(table);
        /* EACH FERMI LEVEL IN ITS RUN'S OWN FRAME -- the record's one
         * sentence (`record.fermi_frames`), so 5.17 eV beside −1.92 eV is
         * never read as a mismatch (§ 2a.12). */
        if (rec.fermi_frames && rec.fermi_frames.note) {
            card.appendChild(_el("p", "transport-note", rec.fermi_frames.note));
        }

        /* TWO LEADS THAT DISAGREE is a defect nothing else shows -- the
         * record's comparison (`record.leads_agreement`). */
        const leads = rec.leads;
        if (leads && !leads.agree) {
            card.appendChild(_el("p", "transport-warn",
                "The two electrodes' Fermi levels differ by "
                + _fmt(leads.differ_ev, 3, false) + " eV (more than "
                + leads.tolerance_ev + " eV). They are bulk runs of the "
                + "same lead, so they should agree — T(E) is measured "
                + "relative to this."));
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
                /* A SWEPT RUN'S POINTS: what each started from (a point
                 * taken over names the run that computed it; a walked
                 * point the point before it) and what it alone took
                 * (§ 2a.11, § 2a.12). */
                if (Array.isArray(c.points) && c.points.length) {
                    const pts = _el("ul", "transport-chain-points");
                    c.points.forEach((p) => {
                        const bits = [];
                        if (p.started_from) bits.push("from " + p.started_from);
                        /* `took` is the status door's own sentence per
                         * file ("<file> <- <run>"). */
                        (p.took || []).forEach((g) => bits.push(String(g)));
                        pts.appendChild(_el("li", null,
                            Number(p.bias_v) + " V: "
                            + (bits.length ? bits.join(", ")
                               : "started from what the run gathered")));
                    });
                    li.appendChild(pts);
                }
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
        /* THE CITATION'S KIND (§ 3.1): a run's folder, or a structure
         * pair's .xyz -- and a pair's frames. */
        if (prov.citation) {
            row(prov.kind === "pair" ? "cited pair" : "cited run",
                prov.citation);
        }
        if (prov.kind === "pair" && prov.frames) row("frames", prov.frames);
        /* HOW THE CITED RUN ENDED AND WHAT IT CONVERGED (§ 3.1): its
         * record's line, and the status verb's reading of its geometry. */
        if (prov.evidence) row("concluded", prov.evidence);
        const relaxed = prov.relaxation || {};
        if (relaxed.converged) {
            row("converged", relaxed.converged
                + (/NO$/.test(relaxed.converged)
                    ? " — the geometry cited is the last one SIESTA wrote, "
                      + "not a converged minimum" : ""),
                /NO$/.test(relaxed.converged)
                    ? "transport-prov-val is-weak" : "transport-prov-val");
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
                margin: { l: 8, r: 12, t: 8, b: 34 },
                xaxis: { title: { text: "E − E_F (eV)", standoff: 4 },
                         automargin: true, zeroline: false },
                yaxis: { title: { text: "T", standoff: 4 }, type: "log",
                         automargin: true, exponentformat: "power" },
                legend: { orientation: "h", y: -0.2 },
            });
            _onClick(state.nodes.te, state);
        } else {
            state.nodes.te.textContent = state.points.length && state.bias !== null
                ? "No transmission at " + state.bias + " V yet."
                : "No transmission has run yet.";
        }

        /* DOS: the selected point's total, its parts by region, the leads',
         * and every selection added. */
        const dos = p && p.dos;
        if (!dos || !Array.isArray(dos.energy_ev)) {
            state.nodes.dos.textContent = p
                ? "No DOS for this point" + (p.dos_why ? ": " + p.dos_why : ".")
                : (state.points.length && state.bias !== null
                    ? "No transmission at " + state.bias + " V yet."
                    : "No transmission has run yet.");
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
                margin: { l: 8, r: 12, t: 8, b: 34 },
                xaxis: { title: { text: "E − E_F (eV)", standoff: 4 },
                         automargin: true, zeroline: false },
                yaxis: { title: { text: "DOS (states/eV)", standoff: 4 },
                         automargin: true },
                legend: { orientation: "h", y: -0.2 },
            });
        }
        _drawPdosList(state);

        /* I-V: the curve, its points clickable, the selected one marked.
         * Under the low-bias approximation the curve is the record's own
         * I(V) computed from T(E, 0) (`record.linear_response_iv`), one
         * value per listed voltage, the measured 0 V slice its only point
         * (`engines/transport.md` § 2a.12). */
        const computed = state.iv && state.iv.computed === "linear-response";
        const iv = computed
            ? (state.iv.voltages_v || []).map((v, k) => (
                  { bias_v: v, current_a: (state.iv.current_a || [])[k] }))
                  .filter((q) => q.current_a !== null && q.current_a !== undefined)
            : pts.filter((q) => q.current_a !== null && q.current_a !== undefined);
        if (iv.length) {
            _plot(state.nodes.iv, [{
                x: iv.map((q) => q.bias_v), y: iv.map((q) => q.current_a),
                mode: "lines+markers",
                name: computed ? "I computed from T(E, 0)" : "I (total)",
                line: { color: t.accent, width: 1.5,
                        dash: computed ? "dash" : "solid" },
                marker: { size: iv.map((q) => (q.bias_v === state.bias ? 12 : 6)) },
                /* A click selects the slice behind the point: under the
                 * low-bias approximation every point comes from the one
                 * 0 V slice. */
                meta: iv.map((q) => (computed ? 0 : q.bias_v)),
            }], {
                margin: { l: 8, r: 12, t: 8, b: 34 },
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
                const on = Number(tr.dataset.bias) === state.bias;
                tr.classList.toggle("is-selected", on);
                const pick = tr.querySelector(".transport-iv-pick");
                if (pick) pick.setAttribute("aria-pressed", on ? "true" : "false");
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
            rm.setAttribute("aria-label", "remove the selection " + s.name);
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
    async function _mountDevice(molHost, state, junctionFile) {
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
            handle, (state.dir ? state.dir + "/" : "") + junctionFile);
        if (res && res.ok === false) {
            molHost.appendChild(_el("p", "transport-note",
                "The composed junction could not be opened: " + (res.error || "")));
        }
        /* SIDE-ON: the transport axis is c = z, and looked at down z a
         * junction is a square with the molecule hidden inside it (the
         * 2026-10-08 road walk, plan § 5x.7 F3).  The window's one camera
         * action beyond reset (`mount` handle `lookAlong`). */
        if (typeof handle.lookAlong === "function") handle.lookAlong("x");
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
        /* THE HANDLE IS THE CALCULATION'S DESCRIPTION at its root -- the
         * server's registry says so (`parse.sidecars.task`, parser
         * `transport-task`): the report is composed on read from the
         * folder (`engines/transport.md` § 2a.12), so it is there before
         * any `summarize task`; `<label>.transport.json` is that command's
         * copy and opens as a file, not as this report. */
        match: (file, meta) => !!(meta && meta.parser === "transport-task"),

        mount(host, file, ctx) {
            const state = { bias: null, points: [], pdos: [], nodes: {},
                            conv: null, handle: null, unsubFrame: null,
                            disposed: false, file: file, dir: _dirOf(file) };
            (async function () {
                let rec;
                try {
                    const r = await fetch("/api/transport/record?path="
                                          + encodeURIComponent(file));
                    /* A SERVER FAULT ANSWERS A PAGE, NOT JSON: said as the
                     * status it is, never as a JSON syntax error. */
                    if (!r.ok && !/json/.test(r.headers.get("content-type") || "")) {
                        throw new Error("the record could not be composed: the "
                                        + "server answered " + r.status);
                    }
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
                /* THE BIAS POINT, SET FROM OUTSIDE: the ladder's row for a
                 * transmission run opens the report at its point
                 * (results.md § 2.4) -- before the record is drawn it is
                 * the starting selection, after it the same pick a row
                 * click makes. */
                selectBias(v) {
                    if (state.points.length) _select(state, v);
                    else state.bias = v;
                },
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
