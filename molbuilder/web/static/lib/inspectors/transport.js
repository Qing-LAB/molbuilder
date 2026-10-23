/* transport.js — a transport calculation's report (`web/results.md` § 2.5).
 *
 * COMPOSED ON READ (`engines/transport.md` § 2a.12): the record arrives from
 * `/api/transport/record`, which composes it from the rungs as they are now
 * -- every rung's state the one status door's, every point's its run's --
 * never the copy `summarize task` wrote.  It decides nothing about the
 * physics: every number drawn is one the record carries.
 *
 * TWO SELECTIONS, ONE OWNER EACH (the spectra tab's pattern), and a point is
 * the pair of them (`engines/transport.md` § 2a.9):
 *   * the FRAME -- MolView's model owns it (`data.setCurrentFrame` /
 *     `data.onFrameChange`, `web/molview.md` § 6.4) once the junction is in
 *     it; nothing here keeps a copy (`_frame` asks it) -- until then, or
 *     when the junction cannot be opened, the report holds the frame a pick
 *     asked for.  MolView hides its bar for one structure;
 *   * the BIAS POINT -- this report owns it (`state.bias`): a click on a
 *     curve, an I-V point or a table row sets it.
 * THE FRAME IS AN AXIS of every chart (`web/results.md` § 2.5): each card
 * draws one frame -- the frame bar's -- or all frames, a legend entry each
 * (`lib/frame-axis.js`); every point is found by one lookup (`_find`, over
 * the record's points or a rung's) and named by
 * `lib/results/point-words.js`, and nothing else.
 *
 * Built from what exists: MolView mounted read-only (the structure
 * presenter's door), the one SCF plot (`lib/scfplot/scfplot.js`), the
 * charts' theme (`lib/plot-theme.js`), the page's cards and panel tabs, the
 * ladder's state chip.
 */
import { mount, toDisplay } from "/static/lib/molview/index.js";
import { molviewFiles } from "../projects/molview-doors.js";
import { ALL, ONE, frameAxisControl } from "../frame-axis.js";
import { frameWords, pointWords } from "../results/point-words.js";

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

    /* TBTRANS'S LOOP, in the one sentence a point's run record states it in
     * (`progress.said`, `tbtrans.loop_words`, model/parse.md § 5d.5) -- and,
     * while its estimate stands, how long ago it was made, as every page
     * says an age (`lib/time-ago.js`).  "" where the record states none. */
    function _loopWords(pr) {
        if (!pr || !pr.said) return "";
        const at = pr.estimated_at ? Date.parse(pr.estimated_at) / 1000 : NaN;
        return pr.said + (isFinite(at)
            ? " · estimated " + root.molbuilder.timeAgo(at) : "");
    }

    /* THE RUNG'S ROW (results.md § 2.5): its running point, named as every
     * page names one; that point's loop; how many points are still to come
     * after it; and the last finished point's loop, the only time a point
     * still to come is judged by -- said as every page says a time. */
    function _rungLoopWords(pr) {
        const said = _loopWords(pr);
        if (!said) return "";
        const after = Number(pr.after) || 0;
        const m = pr.measured;
        const tail = [
            after ? after + " more point" + (after === 1 ? "" : "s")
                    + " after it" : "",
            m ? "the loop at " + pointWords(m.frame, m.of, m.bias_v)
                + " took " + root.molbuilder.timeSpan(m.loop_elapsed_s) : "",
        ].filter(Boolean).join("; ");
        return "at " + pointWords(pr.frame, pr.of, pr.bias_v) + ": " + said
            + (tail ? " — " + tail : "");
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
        root.molbuilder.plotBox.draw(node, traces,
                                     root.molbuilder.themedLayout(layout),
                                     { displayModeBar: false });
    }

    /* PANEL TABS -- the page's `.panel-tabs` markup, driven by the one tab
     * strip (`lib/tab-strip.js`, ui-contract.md § 4.1); a chart in a panel
     * shown follows its box by itself (`lib/plot-box.js`). */
    let _tabsSeq = 0;
    function _tabs(names) {
        const bar = _el("div", "panel-tabs");
        bar.setAttribute("role", "tablist");
        const panels = {};
        const box = _el("div", "transport-tabpanels");
        const prefix = "transport-tab-" + (++_tabsSeq) + "-";
        names.forEach(([key, label]) => {
            const b = _el("button", "panel-tab", label);
            b.type = "button";
            b.id = prefix + key;
            b.dataset.tab = key;
            b.setAttribute("role", "tab");
            b.setAttribute("aria-controls", prefix + key + "-panel");
            bar.appendChild(b);
            const p = _el("div", "panel-tabpanel");
            p.id = prefix + key + "-panel";
            box.appendChild(p);
            panels[key] = p;
        });
        // Built before it is in the page: the strip is told each tab's panel.
        root.molbuilder.tabStrip.mount(bar, {
            panelOf: (t) => panels[t.dataset.tab] || null });
        return { bar, box, panels };
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
        state.average = rec.average || {};
        // THE SET'S FRAME COUNT, as the record states it (its `average`
        // block names the composed junction's, `engines/transport.md`
        // § 2a.12); one structure is one.
        state.nFrames = Number(state.average.frames) || 1;
        // THE STARTING POINT: the first that exists -- a transmission's, or
        // one still to come or failed, whose device run may already speak.
        const first = pts[0] || pending[0] || failed[0];
        if (state.bias === null && first) state.bias = first.bias_v;

        const wrap = _el("div", "transport-record");
        const head = _el("div", "card transport-head");
        head.appendChild(_el("h3", "transport-title",
            "Transport — " + (rec.label || "(unlabelled)")));
        head.appendChild(_el("p", "transport-sub",
            rec.treatment
            + (state.nFrames > 1 ? " · " + state.nFrames + " frames" : "")
            + " · " + pts.length + " point" + (pts.length === 1 ? "" : "s")
            + " with a transmission"
            + (pending.length ? " · " + pending.length + " pending" : "")
            + (failed.length ? " · " + failed.length + " failed" : "")
            + (rec.energies_relative_to_ef ? " · energies relative to E_F"
                                           : "")));
        head.appendChild(_exportBar(file));
        wrap.appendChild(head);

        /* ROW 1: the device beside the curves. */
        const row = _el("div", "card-row transport-row");
        const device = _card("Device", "transport-device-card");
        const molHost = _el("div", "molviewer-host transport-molview");
        device.appendChild(molHost);
        /* A JUNCTION WHOSE ORDER IS NOT RECORDED -- its sidecar states none
         * (`model/overview.md` § 2.2a, I9) -- is drawn as TranSIESTA lists
         * it, and the record says so, with its way on, in its own words. */
        if (rec.order_unstated) {
            device.appendChild(_el("p", "transport-note", rec.order_unstated));
        }
        /* WHICH INDEX ITS ATOMS ARE NUMBERED BY (`model/overview.md`
         * § 2.2b; `web/results.md` § 2.5) -- the user's, the drawing put
         * back through the junction's order record, unless the record says
         * the order is not recorded (above), when it is TranSIESTA's. */
        if (!rec.order_unstated) {
            const line = _el("p", "transport-note",
                "Atoms are numbered as in your structure (user index + 1). "
                + "The engine's own files list them by engine index -- the "
                + "digest, and .TBT.nc as sisl reads it, counted from 0; ");
            line.appendChild(_el("code", null, "molbuilder atom-index"));
            line.appendChild(root.document.createTextNode(
                " shows both, and "));
            const how = _el("a", null, "the post-processing document");
            how.href = "/documents?doc=engines/transport-postprocessing.md";
            how.target = "_blank";
            how.rel = "noopener";
            line.appendChild(how);
            line.appendChild(root.document.createTextNode(
                " how to use them."));
            device.appendChild(line);
        }
        row.appendChild(device);

        const curves = _card("Results", "transport-curves-card");
        const tabs = _tabs([["te", "T(E)"], ["dos", "DOS"], ["iv", "I–V"]]);
        curves.appendChild(tabs.bar);
        curves.appendChild(tabs.box);
        row.appendChild(curves);
        wrap.appendChild(row);

        state.nodes = {
            te: _el("div", "transport-plot plot-box"),
            eig: _el("p", "transport-note"),
            avg: _el("p", "transport-note"),
            dos: _el("div", "transport-plot plot-box"),
            iv: _el("div", "transport-plot plot-box"),
            table: null,
        };
        _fillTE(tabs.panels.te, rec, state);
        _fillDOS(tabs.panels.dos, rec, file, state);
        _fillIV(tabs.panels.iv, rec, pending, failed, state);
        _frameControl(tabs.panels.te, state, "te");
        _frameControl(tabs.panels.dos, state, "dos");
        _frameControl(tabs.panels.iv, state, "iv");

        /* ROW 2: every rung's convergence. */
        wrap.appendChild(_convergence(stages, state));
        /* ROW 3: the rungs, the provenance, the caveat. */
        wrap.appendChild(_rungs(rec, stages, state));
        host.appendChild(wrap);

        _mountDevice(molHost, state, rec.junction_file, !!rec.order_unstated);
        _redraw(state);
    }

    function _fillTE(panel, rec, state) {
        /* THE CURVE AND ITS TREATMENT, named beside it (`engines/
         * transport.md` § 2a.10, § 2a.12) -- the record's own words
         * (`record.TREATMENT_NOTE`), which `summarize` prints too. */
        panel.appendChild(_el("p", "transport-note", rec.treatment_note));
        panel.appendChild(state.nodes.te);
        panel.appendChild(state.nodes.eig);
        /* THE MODE'S AVERAGE, said under the family in *all frames*: the
         * record's numbers, never worked out here (§ 2a.12). */
        panel.appendChild(state.nodes.avg);
    }

    /* THE FRAME AXIS ON A CARD (`web/results.md` § 2.5): one frame -- the
     * frame bar's -- or all frames, shown only for a frame set; switching
     * redraws the card. */
    function _frameControl(panel, state, key) {
        if (state.nFrames < 2) return;
        panel.insertBefore(frameAxisControl(root.document, state.frameMode[key],
            (m) => { state.frameMode[key] = m; _redraw(state); }),
            panel.firstChild);
    }

    /* THE EXPORT -- every result as it stands, as the data file or its
     * tables (`engines/transport.md` § 2a.12, *Export*): composed by the
     * server from what is already there, no data file read, and handed over
     * by the file layer's one download (`projects.download`); beside the
     * buttons the server's own sentence -- the points it holds without their
     * DOS by region and how to read them (its `X-Molbuilder-Said`) -- or a
     * refusal, and the post-processing document's link. */
    function _exportBar(file) {
        const bar = _el("div", "fdf-actions transport-export");
        bar.appendChild(_el("span", "hint",
                            "Export every result as it stands:"));
        const msg = _el("span", "hint");
        const buttons = [["CSV", "csv"], ["NetCDF", "netcdf"]].map(
            ([label, fmt]) => {
                const b = _el("button", null, label);
                b.type = "button";
                b.addEventListener("click", () => run(fmt));
                bar.appendChild(b);
                return b;
            });
        const run = async (fmt) => {
            buttons.forEach((b) => { b.disabled = true; });
            msg.textContent = "Composing the export.";
            try {
                const r = await fetch("/api/transport/export?path="
                    + encodeURIComponent(file) + "&format=" + fmt);
                if (!r.ok) {
                    let why = "HTTP " + r.status;
                    try { why = (await r.json()).error || why; }
                    catch (_) { /* the status is the answer */ }
                    msg.textContent = why;
                    return;
                }
                // THE NAME THE SERVER GAVE IT, as the header states it.
                const said = r.headers.get("Content-Disposition") || "";
                const utf = said.match(/filename\*=UTF-8''([^;]+)/i);
                const plain = said.match(/filename="?([^";]+)"?/i);
                const name = utf ? decodeURIComponent(utf[1])
                    : plain ? plain[1] : "export";
                root.molbuilder.projects.download(name, await r.blob());
                msg.textContent = r.headers.get("X-Molbuilder-Said") || "";
            } catch (e) {
                msg.textContent = String((e && e.message) || e);
            } finally {
                buttons.forEach((b) => { b.disabled = false; });
            }
        };
        bar.appendChild(msg);
        const how = _el("a", null, "How to read these in Python");
        how.href = "/documents?doc=engines/transport-postprocessing.md";
        how.target = "_blank";
        how.rel = "noopener";
        bar.appendChild(how);
        return bar;
    }

    /* EACH POINT'S DOS PER ORBITAL, READ ON THE PERSON'S WORD
     * (`engines/transport.md` § 2a.12, `model/parse.md` § 5e): what is kept
     * of each data file's reading, in the server's words, and its acts --
     * read what is not kept, read every one again, take a reading over --
     * each the server's to do (`/api/transport/digest`), one reading of a
     * file at a time wherever it is asked.  While one runs, how far it is,
     * asked every two seconds; read, the report's points read again. */
    function _digestCard(panel, file, state) {
        const box = _el("div", "transport-digest");
        const head = _el("p", "transport-note");
        const list = _el("ul", "transport-chain");
        const bar = _el("div", "fdf-actions");
        const read = _el("button", null, "Read the DOS per orbital");
        const again = _el("button", null, "Read again");
        const take = _el("button", null, "Take over");
        [read, again, take].forEach((b) => { b.type = "button"; });
        const msg = _el("span", "hint");
        bar.appendChild(read);
        bar.appendChild(again);
        bar.appendChild(take);
        bar.appendChild(msg);
        box.appendChild(head);
        box.appendChild(list);
        box.appendChild(bar);
        panel.appendChild(box);
        let busy = false;
        let reading = false;     // a reading under way at the last ask
        let follow = null;       // the next ask while another one reads
        const show = (points) => {
            const all = [];
            points.forEach((p) => p.readings.forEach(
                (r) => all.push({ ...r, words: p.words })));
            const todo = all.filter((r) => r.state !== "kept");
            // WHOSE CLAIM: alive on this host stands; gone on this host is
            // stale, the next reading's; on another host (alive null) stands
            // until the person takes it over (`model/parse.md` § 5e).
            const alive = (r) => (r.claim || {}).alive;
            head.textContent = !all.length
                ? "No finished transmission point has a data file yet."
                : todo.length
                ? "The DOS by region and a selection's PDOS read each "
                  + "point's data file once, then keep it beside the file."
                : "Every point's DOS per orbital is kept beside its data file.";
            list.replaceChildren(...all.map((r) => _el("li", null,
                r.words + ": " + r.file + " (" + r.size_words + ") — "
                + r.said)));
            read.hidden = !todo.some((r) => r.state !== "being read"
                                     || alive(r) === false);
            again.hidden = !all.length;
            take.hidden = busy || !todo.some((r) => r.state === "being read"
                                             && alive(r) === null);
            read.disabled = again.disabled = take.disabled = busy;
            return todo.some((r) => r.state === "being read"
                             && alive(r) !== false);
        };
        const ask = async () => {
            let b = null;
            try {
                b = await (await fetch("/api/transport/digest?path="
                                       + encodeURIComponent(file))).json();
            } catch (_) { b = null; }
            if (state.disposed) return;
            if (!b || !b.ok) {
                // A POLL THAT FAILED says nothing of a reading followed: asked
                // again in 2 s.
                if (!busy && reading) {
                    clearTimeout(follow);
                    follow = setTimeout(ask, 2000);
                }
                return;
            }
            const now = show(b.points);
            if (busy) return;               // this card's own reading asks
            clearTimeout(follow);
            // ANOTHER'S READING -- another tab's, `summarize task`'s, another
            // host's -- followed every 2 s; when none reads, the record again,
            // and one more look: a reading of several files is between two.
            if (now) follow = setTimeout(ask, 2000);
            else if (reading) {
                if (state.reload) state.reload();
                follow = setTimeout(ask, 2000);
            }
            reading = now;
        };
        const go = async (body) => {
            busy = true;
            clearTimeout(follow);
            msg.textContent = "Reading — on a cluster's file system a point "
                + "can take minutes; the rest of the report stays in use.";
            await ask();
            const timer = setInterval(() => {
                if (state.disposed) { clearInterval(timer); return; }
                ask();
            }, 2000);
            // THE ANSWER, handed to the one who asked: the outcomes, or why
            // there are none.
            let answer;
            try {
                const r = await fetch("/api/transport/digest", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: file, ...body }) });
                const b = await r.json();
                answer = b && b.ok
                    ? { ok: true, outcomes: b.outcomes || [] }
                    : { ok: false, error: String((b && b.error) || r.status) };
            } catch (e) {
                answer = { ok: false, error: String((e && e.message) || e) };
            } finally {
                clearInterval(timer);
                busy = false;
                reading = false;            // the record is fetched below
            }
            msg.textContent = answer.ok
                ? answer.outcomes.filter((o) => o.state !== "kept")
                    .map((o) => o.said).join("; ")
                : answer.error;
            if (state.disposed) return answer;
            await ask();
            if (state.reload) state.reload();
            return answer;
        };
        read.addEventListener("click", () => go({}));
        again.addEventListener("click", () => go({ fresh: true }));
        take.addEventListener("click", () => go({ take_over: true }));
        ask();
    }

    /* THE PDOS OF THE ATOMS YOU CHOOSE, in the order a person takes the
     * steps (`web/results.md` § 2.5; the user, 2026-10-23): 1 select atoms
     * in the Device view -- what is selected now said as it changes,
     * through MolView's own selection (`state.onSelection`) --, 2 which
     * orbitals, 3 name the curve and add it, disabled until atoms are
     * selected; a point whose DOS per orbital is not kept said at step 3. */
    function _fillDOS(panel, rec, file, state) {
        _digestCard(panel, file, state);
        panel.appendChild(state.nodes.dos);
        const menu = rec.pdos_orbitals || {};
        const box = _el("div", null);
        box.appendChild(_el("p", "transport-note",
                            "The PDOS of atoms you choose:"));
        const steps = _el("ol", "transport-pdos-steps");
        const step1 = _el("li", null,
            "Select atoms in the Device view (left): click them, or use its "
            + "Selection page to pick by element (every S) or by label.");
        const now = _el("p", "transport-note", "Selected now: none yet.");
        step1.appendChild(now);
        const step2 = _el("li", null, "Which orbitals: ");
        const orb = _el("select", "transport-pdos-orb");
        orb.setAttribute("aria-label", "orbital type");
        (menu.types || ["all"]).forEach((t) => {
            const o = _el("option", null, t === "all" ? "all orbitals" : t);
            o.value = t;
            orb.appendChild(o);
        });
        step2.appendChild(orb);
        if (menu.note) step2.appendChild(_el("p", "transport-note", menu.note));
        const step3 = _el("li", null, "Name the curve: ");
        const name = _el("input", "transport-pdos-name");
        name.type = "text";
        name.placeholder = "e.g. S, ring C";
        name.setAttribute("aria-label", "name for the curve");
        const add = _el("button", "transport-pdos-add", "Add to the plot");
        add.type = "button";
        add.disabled = true;
        const msg = _el("span", "transport-pdos-msg");
        step3.appendChild(name);
        step3.appendChild(add);
        step3.appendChild(msg);
        [step1, step2, step3].forEach((li) => steps.appendChild(li));
        box.appendChild(steps);
        box.appendChild(_el("p", "transport-note", "Curves on the plot:"));
        const list = _el("ul", "transport-pdos-list");
        box.appendChild(list);
        panel.appendChild(box);
        state.nodes.pdosList = list;
        /* WHAT IS SELECTED NOW, and whether the point shown has its DOS per
         * orbital kept -- each said where its step is. */
        state.onSelection = () => {
            const atoms = _selection(state);
            /* EACH ATOM AS THE VIEW NUMBERS IT, with its element
             * (`web/results.md` § 2.5): *2 atoms (#13 S, #14 S)*. */
            const named = (a) => "#" + toDisplay(a)
                + (state.elements && state.elements[a]
                   ? " " + state.elements[a] : "");
            now.textContent = atoms.length
                ? "Selected now: " + atoms.length + " atom"
                  + (atoms.length === 1 ? "" : "s") + " ("
                  + atoms.slice(0, 8).map(named).join(", ")
                  + (atoms.length > 8 ? ", …" : "") + ")."
                : "Selected now: none yet.";
            add.disabled = !atoms.length;
            const p = _pointAt(state, _frame(state), state.bias);
            const unread = p && (p.digest || []).some(
                (d) => d.state !== "kept");
            msg.textContent = unread
                ? "This point's DOS per orbital is not read yet -- use "
                  + "Read the DOS per orbital above first." : "";
        };
        state.onSelection();
        add.addEventListener("click", () => {
            const atoms = _selection(state);
            if (!atoms.length) return;
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
        const table = _el("table", "num-table transport-iv");
        const head = _el("tr");
        const iv = rec.iv || {};
        const computed = iv.computed === "linear-response";
        const n = state.nFrames;
        /* THE CURRENT IS THE JUNCTION'S TOTAL, both spin channels, beside
         * the figure TBtrans printed -- one channel's (`engines/
         * transport.md` § 2a.4; `record.py` CURRENT_MEANS).  Under the
         * low-bias approximation the record computed it from T(E, 0) for
         * each listed voltage (`record.linear_response_iv`). */
        [n > 1 ? "Point" : "Bias [V]", "G(E_F) [G0]",
         computed ? "Current, computed from T(E, 0) [A]" : "Current, total [A]",
         // Under the approximation TBtrans ran at 0 V only: a dash at the
         // other voltages means not applicable, and the header says so.
         computed ? "TBtrans printed [A] (0 V only)" : "TBtrans printed [A]"
        ].forEach((h) => head.appendChild(_el("th", null, h)));
        table.appendChild(head);
        /* A ROW IS A POINT -- a frame and a voltage -- named as the frame bar
         * counts, and a pick moves both (`web/results.md` § 2.5): ``at`` is
         * the point the row picks, the slice its numbers come from. */
        function pickRow(frame, bias, cells, at, cls) {
            const tr = _el("tr", "transport-iv-row" + (cls ? " " + cls : ""));
            tr.dataset.frame = String(at.frame);
            tr.dataset.bias = String(at.bias);
            const words = pointWords(frame, n, bias);
            const first = _el("td", null);
            // THE PICK IS A BUTTON (ui-contract.md § 4.1): the first cell
            // names the point and takes the keyboard; the row's click is the
            // mouse's wider target for the same pick.
            const pick = _el("button", "transport-iv-pick", words);
            pick.type = "button";
            // THE POINT IT PICKS, said as such -- under the low-bias
            // approximation a row's numbers come from its frame's 0 V slice.
            pick.setAttribute("aria-label", "show the point "
                              + pointWords(at.frame, n, at.bias));
            first.appendChild(pick);
            tr.addEventListener("click", () => _select(state, at));
            tr.appendChild(first);
            cells.forEach((c) => tr.appendChild(
                typeof c === "string" ? _el("td", null, c) : c));
            table.appendChild(tr);
        }
        if (computed) {
            const frames = iv.frame || [];
            (iv.voltages_v || []).forEach((v, k) => {
                const f = frames[k] == null ? 0 : frames[k];
                const zero = _pointAt(state, f, 0);
                const i = (iv.current_a || [])[k];
                const note = (iv.notes || {})[String(k)];
                pickRow(f, v, [
                    Math.abs(v) < 1e-9 && zero ? _fmt(zero.conductance_g0, 4, false) : "—",
                    i === null || i === undefined ? (note || "—") : _fmt(i, 4, true),
                    _fmt(((iv.current_a_printed || [])[k]), 4, true),
                // EVERY ROW OF A FRAME IS ITS ONE 0 V SLICE'S, so a row picks
                // that slice -- the pick the I-V chart's point makes.
                ], { frame: f, bias: 0 });
            });
            /* A FRAME'S OWN NOTE -- its 0 V transmission not run -- under the
             * table, in the record's words; a voltage's note stands in its
             * row. */
            Object.keys(iv.notes || {}).forEach((key) => {
                if (!isFinite(Number(key))) {
                    panel.appendChild(_el("p", "transport-note", iv.notes[key]));
                }
            });
        } else {
            state.points.forEach((p) => pickRow(_pf(p), p.bias_v, [
                _fmt(p.conductance_g0, 4, false), _fmt(p.current_a, 4, true),
                _fmt(p.current_a_printed, 4, true)],
                { frame: _pf(p), bias: p.bias_v }));
        }
        /* A POINT WITHOUT ITS TRANSMISSION, in its run's words: pending is
         * still to come, never a failure; failed ended without it.  A pick
         * like every row -- its device run may already speak. */
        pending.concat(failed).forEach((p) => {
            const st = _el("td");
            st.appendChild(_chip(p.state));
            const why = _el("td", null, [p.why, _loopWords(p.progress)]
                .filter(Boolean).join(" · "));
            why.setAttribute("colspan", "2");
            pickRow(_pf(p), p.bias_v, [st, why],
                    { frame: _pf(p), bias: p.bias_v }, "transport-pending");
        });
        panel.appendChild(table);
        state.nodes.table = table;
        Object.keys(rec.current_means || {}).sort().forEach((spin) => {
            panel.appendChild(_el("p", "transport-note",
                                  spin + ": " + rec.current_means[spin]));
        });
        /* WHAT A POINT IS A COPY OF, AND WHAT IT SHARES -- a calculation
         * that reuses a finished one (engines/transport.md § 2a.16): the
         * run that computed it, or the energies a finished point computed
         * and those its own run did, in the manifest's words. */
        state.points.forEach((p) => {
            const bits = [];
            if (p.copied_from) bits.push("copied from " + p.copied_from);
            if (p.shared) bits.push("shares " + p.shared.held + " of "
                + p.n_energies + " energies with " + p.shared.from
                + " — computes " + p.shared.computed + " here");
            if (bits.length) panel.appendChild(_el("p", "transport-note",
                pointWords(_pf(p), n, p.bias_v) + ": "
                + bits.join(" · ")));
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
            const e = _el("div", "transport-plot-half plot-box");
            const r = _el("div", "transport-plot-half plot-box");
            pair.appendChild(e);
            pair.appendChild(r);
            panel.appendChild(line);
            panel.appendChild(pair);
            state.conv[st.stage] = { st, line, e, r };
            /* A RUNG WITH A POINT PER FRAME -- the device -- carries the
             * frame axis; the seed and the leads have no frame. */
            if (Array.isArray(st.by_point)
                    && st.by_point.some((p) => _pf(p) > 0)) {
                _frameControl(panel, state, "conv");
            }
        });
        return card;
    }

    function _drawConvergence(state) {
        const n = state.nFrames;
        const frame = _frame(state);
        const t = root.molbuilder.plotTheme();
        Object.values(state.conv || {}).forEach(({ st, line, e, r }) => {
            /* A SCAN'S RUNG speaks for the selected point -- its frame and
             * its voltage; in *all frames*, every frame's run at the
             * voltage, on one axis, the frame bar's bold. */
            const pts = Array.isArray(st.by_point) ? st.by_point : null;
            const src = pts ? _find(pts, frame, state.bias) : st;
            if (!src) {
                line.replaceChildren(_el("span", "transport-stage-detail",
                    state.bias === null ? "this rung has not run yet"
                        : pointWords(frame, n, state.bias)
                          + " · this rung has not run at this point"));
                e.hidden = true;
                r.hidden = true;
                return;
            }
            const runs = pts && state.frameMode.conv === ALL
                ? pts.filter((p) => p.bias_v === state.bias).map((p) => (
                      { name: frameWords(_pf(p), n), rows: p.scf || [],
                        colour: t.palette[_pf(p) % t.palette.length],
                        bold: _pf(p) === frame }))
                : [{ name: "", rows: Array.isArray(src.scf) ? src.scf : [] }];
            line.replaceChildren();
            line.appendChild(_chip(src.state || st.state));
            const bits = [];
            if (src.bias_v !== undefined && src.bias_v !== null)
                bits.push(pointWords(_pf(src), n, src.bias_v));
            if (src.detail) bits.push(src.detail);
            // THE RUN IT IS A COPY OF -- copied in from the finished
            // calculation this one reuses (engines/transport.md § 2a.16).
            if (st.copied_from) bits.push("copied from " + st.copied_from);
            if (src.negf) {
                bits.push("NEGF: " + src.negf.cycles + " cycles");
                if (src.negf.ef !== undefined)
                    bits.push("E_F " + _fmt(src.negf.ef, 3, false) + " eV");
                if (src.negf.dq !== undefined)
                    bits.push("dQ " + _fmt(src.negf.dq, 2, true));
            }
            line.appendChild(_el("span", "transport-stage-detail",
                                 " " + bits.join(" · ")));
            const any = runs.some((x) => x.rows.length);
            e.hidden = !any;
            r.hidden = !any;
            if (any) {
                root.molbuilder.scfPlot.draw(e, r, runs, {
                    criteria: src.scf_criteria || {},
                    residuals: src.scf_residuals || [] });
            }
        });
    }

    function _rungs(rec, stages, state) {
        const n = state.nFrames;
        const card = _card("Rungs and provenance", "transport-rungs-card");
        const table = _el("table", "num-table transport-rungs");
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
                const said = [st.detail, _rungLoopWords(st.progress)]
                    .filter(Boolean).join(" · ");
                if (said) s.appendChild(_el("span", "transport-stage-detail",
                                            " " + said));
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
                // THE RUN IT IS A COPY OF (engines/transport.md § 2a.16).
                + (st.copied_from ? " · copied from " + st.copied_from : "")
                + (st.gathered_by ? " · gathered by the " + st.gathered_by : "")
                + (st.negf && st.negf.contour && st.negf.contour.poles
                    ? " · " + st.negf.contour.poles + " poles" : "")
                // THE POINT THE ROW'S NUMBERS COME FROM (results.md § 2.4).
                + (typeof st.facts_at_v === "number"
                    ? " · at " + pointWords(st.facts_at_frame,
                                            n, st.facts_at_v) : "");
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
                        if (p.copied_from) bits.push("copied from " + p.copied_from);
                        if (p.started_from) bits.push("from " + p.started_from);
                        if (p.start_unread) bits.push(String(p.start_unread));
                        /* `took` is the status door's own sentence per
                         * file ("<file> <- <run>"). */
                        (p.took || []).forEach((g) => bits.push(String(g)));
                        pts.appendChild(_el("li", null,
                            pointWords(p.frame, n, p.bias_v)
                            + ": "
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

    /* A POINT PICKED -- a curve, an I-V point, a table row, the ladder:
     * ``at`` is ``{frame, bias}``, either left out to keep it.  The voltage
     * is this report's; the frame goes to its owner -- the frame bar, whose
     * change redraws (`_mountDevice`), or, before the junction is in it,
     * the report, which redraws here. */
    function _select(state, at) {
        const bias = at && at.bias !== undefined ? at.bias : state.bias;
        const frame = at && at.frame !== undefined && at.frame !== null
            ? at.frame : _frame(state);
        if (bias === state.bias && frame === _frame(state)) return;
        state.bias = bias;
        if (frame !== _frame(state) && _setFrame(state, frame)) return;
        _loadPdos(state);
    }

    /* THE FRAME, asked of its one owner -- MolView's model, once the
     * junction is in it (`state.frames`) -- each time; until then, or when
     * the junction cannot be opened, the report holds it. */
    function _frame(state) {
        return state.frames ? state.frames.currentFrame()
                            : (state.pendingFrame || 0);
    }

    /* The frame told to its owner -- ``true``, its change redraws -- or,
     * with no owner yet, held here: ``false``, the caller redraws. */
    function _setFrame(state, frame) {
        if (state.frames) {
            state.frames.setCurrentFrame(frame);
            return true;
        }
        state.pendingFrame = frame;
        return false;
    }

    /* A record entry's frame: its index, or 0 for one structure's. */
    function _pf(p) {
        return p.frame === null || p.frame === undefined ? 0 : p.frame;
    }

    /* A point as a key -- which (frame, voltage) an answer belongs to. */
    function _pointKey(frame, bias) {
        return frame + "|" + bias;
    }

    /* THE ONE LOOKUP: the entry at (frame, voltage) of a list of points --
     * the record's, or a rung's `by_point` -- or null. */
    function _find(list, frame, bias) {
        return list.find((p) => _pf(p) === frame && p.bias_v === bias) || null;
    }

    function _pointAt(state, frame, bias) {
        return _find(state.points, frame, bias);
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

    function _redraw(state) {
        const pts = state.points;
        const n = state.nFrames;
        const frame = _frame(state);
        const p = _pointAt(state, frame, state.bias);
        const t = root.molbuilder.plotTheme();
        const col = (i) => t.palette[i % t.palette.length];
        const all = (key) => n > 1 && state.frameMode[key] === ALL;
        const here = pointWords(frame, n, state.bias);
        /* A SPIN-POLARISED CALCULATION draws each spin and their total, told
         * apart by colour and legend (`web/results.md` § 2.5): spin up the
         * first series colour, spin down the second, the total the ink --
         * so in it no voltage or region borrows a spin's colour.  Every
         * number is the record's, the sums too. */
        const spinRun = pts.some((q) => q.channels);
        const UP = t.palette[0], DOWN = t.palette[1];
        const nothingYet = () => (pts.length && state.bias !== null
            ? "No transmission at " + here + " yet."
            : "No transmission has run yet.");
        const axes = {
            margin: { l: 8, r: 12, t: 8, b: 34 },
            xaxis: { title: { text: "E − E_F (eV)", standoff: 4 },
                     automargin: true, zeroline: false },
            legend: { orientation: "h", y: -0.2 },
        };

        /* T(E): one frame -- a curve per voltage, the selected one bold, its
         * eigenchannels beneath; all frames -- a curve per frame at the
         * selected voltage, the frame bar's bold, and the mode's average
         * dashed beside them (`web/results.md` § 2.5). */
        const te = [];
        state.nodes.avg.textContent = "";
        state.nodes.eig.textContent = "";
        if (all("te")) {
            pts.filter((q) => q.bias_v === state.bias).forEach((q) => {
                if (!Array.isArray(q.transmission) || !q.transmission.length) return;
                const f = _pf(q);
                const on = f === frame;
                te.push({ x: q.energy_ev, y: q.transmission, mode: "lines",
                          name: frameWords(f, n),
                          line: { color: col(f), width: on ? 2.5 : 1 },
                          opacity: on ? 1 : 0.6, meta: { frame: f, bias: q.bias_v } });
            });
            const a = (state.average.at || []).find((e) => e.bias_v === state.bias);
            if (a && Array.isArray(a.transmission)) {
                te.push({ x: a.energy_ev, y: a.transmission, mode: "lines",
                          name: "mode average",
                          line: { color: t.ink, width: 2, dash: "dash" } });
            }
            /* THE RECORD'S NUMBERS for the average, or why there is none. */
            state.nodes.avg.textContent = a && a.conductance_g0 !== null
                    && a.conductance_g0 !== undefined
                ? "Mode average at " + pointWords(null, n, state.bias)
                  + ": ⟨G⟩ = " + _fmt(a.conductance_g0, 4, false) + " G0 — "
                  + frameWords(0, n) + "'s "
                  + _fmt(a.base_conductance_g0, 4, false) + " G0, "
                  + (a.conductance_change_percent !== null
                     && a.conductance_change_percent !== undefined
                      ? _fmt(a.conductance_change_percent, 2, false) + " %"
                      : "no change stated: " + (a.conductance_why || ""))
                  + "; the weights sum to " + _fmt(state.average.weight_sum, 6, false)
                  + ", checked to " + state.average.tolerance
                  + ((state.average.warnings || []).length
                     ? " (" + state.average.warnings.join(" ") + ")" : "")
                  + ".  It assumes: "
                  + (state.average.assumptions || []).join(" ")
                : ((a && (a.why || a.conductance_why)) || state.average.why || "");
        } else {
            pts.filter((q) => _pf(q) === frame).forEach((q, i) => {
                if (!Array.isArray(q.transmission) || !q.transmission.length) return;
                const on = q.bias_v === state.bias;
                const at = pointWords(null, n, q.bias_v);
                const meta = { frame: frame, bias: q.bias_v };
                if (q.channels && on) {
                    // THE SELECTED POINT: each spin, and their total.
                    te.push({ x: q.energy_ev, y: q.channels.up, mode: "lines",
                              name: "T↑ · " + at, meta: meta,
                              line: { color: UP, width: 1.75 } });
                    te.push({ x: q.energy_ev, y: q.channels.down, mode: "lines",
                              name: "T↓ · " + at, meta: meta,
                              line: { color: DOWN, width: 1.75 } });
                    te.push({ x: q.energy_ev, y: q.transmission_total,
                              mode: "lines", name: "T↑ + T↓ (total) · " + at,
                              meta: meta, line: { color: t.ink, width: 2.5 } });
                } else if (q.channels) {
                    // ANOTHER VOLTAGE: its total, quiet.
                    te.push({ x: q.energy_ev, y: q.transmission_total,
                              mode: "lines", name: "T↑ + T↓ · " + at,
                              meta: meta, opacity: 0.55,
                              line: { color: t.muted, width: 1 } });
                } else {
                    te.push({ x: q.energy_ev, y: q.transmission, mode: "lines",
                              name: at,
                              line: { color: col(i), width: on ? 2.5 : 1 },
                              opacity: on ? 1 : 0.55, meta: meta });
                }
            });
            // THE EIGENCHANNELS, one set per spin on a polarised point.
            const bySpin = p && p.dos && p.dos.by_spin;
            const sets = bySpin
                ? [["↑", UP, (bySpin.up || {}).eigenchannels],
                   ["↓", DOWN, (bySpin.down || {}).eigenchannels]]
                : [["", t.muted, p && p.dos && p.dos.eigenchannels]];
            let nEig = 0;
            sets.forEach(([mark, color, eig]) => (Array.isArray(eig) ? eig : [])
                .forEach((ch, k) => {
                    nEig += 1;
                    te.push({ x: p.dos.energy_ev, y: ch, mode: "lines",
                              name: "channel " + (k + 1) + (mark ? " " + mark : ""),
                              line: { color: color, width: 1, dash: "dot" } });
                }));
            state.nodes.eig.textContent = p && !nEig
                ? "No eigenchannels for this point" + (p.dos_why ? ": " + p.dos_why : ".")
                : "";
        }
        if (te.length) {
            /* THE AXIS SAYS WHAT T IS: a polarised point's curves name their
             * spin; a non-polarised run's one curve is each spin's, ↑ = ↓ --
             * the channel G = G₀·T(E_F) reads, half its 2T total; under
             * "all frames" a polarised run draws (T↑ + T↓)/2, the transmission
             * the mode's average is of. */
            const tTitle = !spinRun ? "T (each spin, ↑ = ↓)"
                : (all("te") ? "(T↑ + T↓)/2" : "T");
            _plot(state.nodes.te, te, { ...axes,
                yaxis: { title: { text: tTitle, standoff: 4 }, type: "log",
                         automargin: true, exponentformat: "power" } });
            _onClick(state.nodes.te, state);
        } else {
            root.molbuilder.plotBox.say(state.nodes.te, nothingYet());
        }

        /* DOS: one frame -- the selected point's total, its parts by region,
         * the leads', every selection added; all frames -- each frame's total
         * at the selected voltage. */
        if (all("dos")) {
            const tr = [];
            pts.filter((q) => q.bias_v === state.bias).forEach((q) => {
                const d = q.dos;
                if (!d || !Array.isArray(d.energy_ev) || !d.total) return;
                const f = _pf(q);
                tr.push({ x: d.energy_ev, y: d.total, mode: "lines",
                          name: frameWords(f, n) + " (total)",
                          line: { color: col(f), width: f === frame ? 2.5 : 1 } });
            });
            if (tr.length) {
                // TBtrans's DOS has no spin factor: a non-polarised run's is
                // each spin's, a polarised run's the two summed.
                _plot(state.nodes.dos, tr, { ...axes,
                    yaxis: { title: { text: spinRun ? "DOS (states/eV, ↑ + ↓)"
                                          : "DOS (states/eV, each spin)",
                                      standoff: 4 },
                             automargin: true } });
            } else {
                // THE TRANSMISSIONS THERE, THEIR DOS NOT: said in the
                // record's words; no transmission at all, said so.
                const at = pts.filter((q) => q.bias_v === state.bias);
                root.molbuilder.plotBox.say(state.nodes.dos, at.length
                    ? "No DOS at " + pointWords(null, n, state.bias)
                      + (at[0].dos_why ? ": " + at[0].dos_why : ".")
                    : nothingYet());
            }
        } else {
            const dos = p && p.dos;
            if (!dos || !Array.isArray(dos.energy_ev)) {
                root.molbuilder.plotBox.say(state.nodes.dos, p
                    ? "No DOS for this point" + (p.dos_why ? ": " + p.dos_why : ".")
                    : nothingYet());
            } else {
                const tr = [];
                // A POLARISED POINT keeps the first two colours for its spins,
                // and every other curve cycles through the rest.
                const bySpin = dos.by_spin;
                let i = 0;
                const other = (k) => (bySpin
                    ? t.palette[2 + (k % Math.max(1, t.palette.length - 2))]
                    : col(k));
                if (bySpin) {
                    [["↑", UP, bySpin.up], ["↓", DOWN, bySpin.down]].forEach(
                        ([mark, color, d]) => {
                            if (d && Array.isArray(d.total)) tr.push({
                                x: dos.energy_ev, y: d.total, mode: "lines",
                                name: "device " + mark,
                                line: { color: color, width: 1.5 } });
                        });
                }
                if (dos.total) tr.push({ x: dos.energy_ev, y: dos.total, mode: "lines",
                    name: bySpin ? "device total (↑ + ↓)" : "device (total)",
                    line: { color: t.ink, width: 2 } });
                Object.keys(dos.by_label || {}).forEach((lab) => tr.push({
                    x: dos.energy_ev, y: dos.by_label[lab], mode: "lines",
                    name: lab, line: { color: other(i++), width: 1.5 } }));
                Object.keys(dos.lead_spectral || {}).forEach((e) => tr.push({
                    x: dos.energy_ev, y: dos.lead_spectral[e], mode: "lines",
                    name: "spectral " + e, visible: "legendonly",
                    line: { color: other(i++), width: 1, dash: "dash" } }));
                Object.keys(dos.lead_bulk || {}).forEach((e) => tr.push({
                    x: dos.energy_ev, y: dos.lead_bulk[e], mode: "lines",
                    name: "bulk " + e, visible: "legendonly",
                    line: { color: other(i++), width: 1, dash: "dot" } }));
                state.pdos.forEach((s) => {
                    if (!s.curve || s.at !== _pointKey(frame, state.bias)) return;
                    tr.push({ x: s.curve.energy_ev, y: s.curve.pdos,
                        mode: "lines", name: s.name + " (" + s.orbitals + ")",
                        line: { color: other(i++), width: 2 } });
                });
                _plot(state.nodes.dos, tr, { ...axes,
                    yaxis: { title: { text: bySpin
                                          ? "DOS (states/eV, ↑ + ↓ unless marked)"
                                          : "DOS (states/eV, each spin)",
                                      standoff: 4 },
                             automargin: true } });
            }
        }
        _drawPdosList(state);

        /* I-V: the frame's curve, or one per frame, its points clickable, the
         * selected one marked.  Under the low-bias approximation each curve
         * is the record's own I(V) computed from that frame's T(E, 0)
         * (`record.linear_response_iv`), one value per listed voltage, the
         * measured 0 V slice its only point (`engines/transport.md`
         * § 2a.12). */
        const computed = state.iv && state.iv.computed === "linear-response";
        const frames = all("iv") ? Array.from({ length: n }, (_, f) => f) : [frame];
        const ivTraces = frames.map((f) => {
            const rows = computed
                ? (state.iv.voltages_v || []).map((v, k) => ({
                      frame: (state.iv.frame || [])[k],
                      bias_v: v, current_a: (state.iv.current_a || [])[k] }))
                      .filter((q) => (q.frame == null ? 0 : q.frame) === f)
                : pts.filter((q) => _pf(q) === f);
            const got = rows.filter((q) => q.current_a !== null && q.current_a !== undefined);
            return {
                x: got.map((q) => q.bias_v), y: got.map((q) => q.current_a),
                mode: "lines+markers",
                name: n > 1 ? frameWords(f, n)
                            : (computed ? "I computed from T(E, 0)" : "I (total)"),
                line: { color: col(f),
                        width: all("iv") && f === frame ? 2.5 : 1.5,
                        dash: computed ? "dash" : "solid" },
                marker: { size: got.map((q) => (f === frame
                    && q.bias_v === state.bias ? 12 : 6)) },
                /* A click selects the slice behind the point: under the
                 * low-bias approximation every point of a frame comes from
                 * its one 0 V slice. */
                meta: got.map((q) => ({ frame: f, bias: computed ? 0 : q.bias_v })),
            };
        }).filter((tr) => tr.x.length);
        /* EACH SPIN'S CURRENT beside the total, on a polarised calculation's
         * one frame (`web/results.md` § 2.5): the total then takes the ink. */
        const ivSpin = state.iv && Array.isArray(state.iv.current_up_a)
            && !all("iv") && ivTraces.length === 1;
        if (ivSpin) {
            const total = ivTraces[0];
            // UNDER THE APPROXIMATION every curve says it is computed.
            const how = computed ? ", computed from T(E, 0)" : "";
            total.name = "I total" + how;
            total.line = { ...total.line, color: t.ink, width: 2.5 };
            const rows = (state.iv.voltages_v || []).map((v, k) => ({
                frame: (state.iv.frame || [])[k], bias_v: v,
                up: state.iv.current_up_a[k],
                down: (state.iv.current_down_a || [])[k] }))
                .filter((q) => (q.frame == null ? 0 : q.frame) === frame);
            [["up", "I↑" + how, UP], ["down", "I↓" + how, DOWN]].forEach(([key, name, color]) => {
                const got = rows.filter((q) => q[key] !== null && q[key] !== undefined);
                if (!got.length) return;
                ivTraces.push({ x: got.map((q) => q.bias_v),
                    y: got.map((q) => q[key]), mode: "lines+markers", name: name,
                    line: { color: color, width: 1.5, dash: computed ? "dash" : "solid" },
                    marker: { size: 6 },
                    meta: got.map((q) => ({ frame: frame,
                                            bias: computed ? 0 : q.bias_v })) });
            });
        }
        if (ivTraces.length) {
            _plot(state.nodes.iv, ivTraces, {
                margin: { l: 8, r: 12, t: 8, b: 34 },
                xaxis: { title: { text: "Bias (V)", standoff: 4 }, automargin: true },
                yaxis: { title: { text: "Current (A)", standoff: 4 },
                         automargin: true, exponentformat: "e" },
                showlegend: ivTraces.length > 1,
                legend: { orientation: "h", y: -0.2 },
            });
            _onClick(state.nodes.iv, state);
        } else {
            root.molbuilder.plotBox.say(state.nodes.iv, "No current yet.");
        }
        if (state.nodes.table) {
            state.nodes.table.querySelectorAll(".transport-iv-row").forEach((tr) => {
                const on = Number(tr.dataset.frame) === frame
                    && Number(tr.dataset.bias) === state.bias;
                tr.classList.toggle("is-selected", on);
                const pick = tr.querySelector(".transport-iv-pick");
                if (pick) pick.setAttribute("aria-pressed", on ? "true" : "false");
            });
        }
        _drawConvergence(state);
    }

    /* A CLICK ON A CURVE OR A POINT selects its point -- each trace carries
     * ``{frame, bias}`` as `meta` (per marker, on the I-V).  Bound once per
     * chart. */
    function _onClick(node, state) {
        if (typeof node.on !== "function") return;
        // ONE HANDLER, set after every draw: a chart built afresh -- after a
        // sentence stood in its place (`plotBox.say`) -- holds none, and one
        // updated in place holds the last draw's, dropped here.
        node.removeAllListeners("plotly_click");
        node.on("plotly_click", (ev) => {
            const pt = ev && ev.points && ev.points[0];
            if (!pt) return;
            const m = pt.data.meta;
            const at = Array.isArray(m) ? m[pt.pointIndex] : m;
            if (at && typeof at.bias === "number") _select(state, at);
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
                + (s.error && s.at === _pointKey(_frame(state), state.bias)
                   ? " — " + s.error : ""));
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

    /* EACH SELECTION'S PDOS AT THE SELECTED POINT -- its frame and its
     * voltage -- asked of the server (`/api/transport/pdos`); nothing is
     * summed here. */
    async function _loadPdos(state) {
        const bias = state.bias;
        const frame = _frame(state);
        if (bias !== null && state.pdos.length) {
            /* EACH ANSWER KEPT WITH THE POINT IT WAS ASKED AT, and only the
             * newest ask's: an answer that lands after a later pick is
             * dropped, and a curve is drawn only at its own point. */
            const at = _pointKey(frame, bias);
            await Promise.all(state.pdos.map(async (s) => {
                s.asked = at;
                const keep = () => s.asked === at;
                try {
                    const q = "/api/transport/pdos?path="
                        + encodeURIComponent(state.file)
                        + "&point=" + encodeURIComponent(bias)
                        + (state.nFrames > 1 ? "&frame=" + frame : "")
                        + "&atoms=" + s.atoms.join(",")
                        + "&orbitals=" + encodeURIComponent(s.orbitals);
                    const body = await (await fetch(q)).json();
                    if (!keep()) return;
                    s.at = at;
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
                    if (!keep()) return;
                    s.at = at;
                    s.curve = null;
                    s.error = String(e);
                }
            }));
        }
        if (state.disposed || bias !== state.bias || frame !== _frame(state)) return;
        // THE STEPS' LINES follow the point shown (`_fillDOS`).
        if (state.onSelection) state.onSelection();
        _redraw(state);
    }

    /* THE DEVICE, read-only -- the structure presenter's door: MolView
     * mounted with its own owner, the composed junction opened through the
     * one file door (its sidecar brings the labels and the cell). */
    async function _mountDevice(molHost, state, junctionFile, unstated) {
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
        state.handle = handle;              // disposed with the report
        /* THE SELECTION FOLLOWED, for the PDOS steps' line (`_fillDOS`). */
        if (handle.data && handle.data.selection
                && typeof handle.data.selection.subscribe === "function") {
            state.unsubSelection = handle.data.selection.subscribe(() => {
                if (!state.disposed && state.onSelection) state.onSelection();
            });
        }
        /* IN THE USER ORDER (`model/overview.md` § 2.2a, I1): the composed
         * junction is the engine's record, in TranSIESTA's order; the report
         * draws it put back in the user order, and names a selection's atoms
         * so to the PDOS door, which crosses them into the junction's order
         * -- or, a junction whose order is not recorded (I9), drawn as a
         * viewer draws it, in TranSIESTA's order, marked, the record's line
         * above saying so (its PDOS refused by the record). */
        const res = await proj.parser.openMolecule(
            handle, (state.dir ? state.dir + "/" : "") + junctionFile,
            { restore: !unstated });
        if (state.disposed) return;
        if (res && res.ok === false) {
            molHost.appendChild(_el("p", "transport-note",
                "The composed junction could not be opened: " + (res.error || "")));
            _redraw(state);                 // the report keeps the frame
            return;
        }
        /* SIDE-ON: the transport axis is c = z, and looked at down z a
         * junction is a square with the molecule hidden inside it (the
         * 2026-10-08 road walk, plan § 5x.7 F3).  The window's one camera
         * action beyond reset (`mount` handle `lookAlong`). */
        if (typeof handle.lookAlong === "function") handle.lookAlong("x");
        /* EACH ATOM'S ELEMENT, once the junction is in -- what the PDOS
         * steps' line names beside each selected atom's number. */
        const drawn = handle.data.getStructure();
        state.elements = drawn && Array.isArray(drawn.elements)
            ? drawn.elements.slice() : null;
        if (state.onSelection) state.onSelection();
        /* THE FRAME'S ONE OWNER is the model, now the junction is in it:
         * the frame held until now told to it, and the report following it
         * -- every chart, and each selection's PDOS asked again at the new
         * frame. */
        state.frames = handle.data;
        if (state.pendingFrame !== null && state.pendingFrame !== undefined) {
            state.frames.setCurrentFrame(state.pendingFrame);
            state.pendingFrame = null;
        }
        /* ...EXCEPT WHILE THE MOVIE PLAYS: a frame it passes through is not
         * one being looked at, so the charts follow it -- drawn here, from
         * the record -- and each selection's PDOS is asked once, at the
         * frame playback stops on (`results.md` § 2.5). */
        state.unsubFrame = state.frames.onFrameChange(() => {
            if (handle.isPlaying()) _redraw(state);
            else _loadPdos(state);
        });
        state.unsubPlay = handle.onPlaybackChange((on) => {
            if (!on) _loadPdos(state);
        });
        _redraw(state);
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
         * server's registry says so (`task_reader`, parser
         * `transport-task`): the report is composed on read from the
         * folder (`engines/transport.md` § 2a.12), so it is there before
         * any `summarize task`; `<label>.transport.json` is that command's
         * copy and opens as a file, not as this report. */
        match: (file, meta) => !!(meta && meta.parser === "transport-task"),

        mount(host, file, ctx) {
            const state = { bias: null, points: [], pdos: [], nodes: {},
                            conv: null, handle: null, frames: null,
                            unsubFrame: null, unsubPlay: null, drawn: false,
                            disposed: false, file: file, dir: _dirOf(file),
                            nFrames: 1, average: {}, pendingFrame: null,
                            frameMode: { te: ONE, dos: ONE, iv: ONE, conv: ONE } };
            /* THE POINTS READ AGAIN -- once a data file's DOS per orbital
             * is read (`_digestCard`): their regions, and a selection's
             * PDOS asked again; the page around them stays as it is. */
            state.reload = async () => {
                try {
                    const b = await (await fetch("/api/transport/record?path="
                                                 + encodeURIComponent(file))).json();
                    if (state.disposed || !b || !b.ok) return;
                    state.points = Array.isArray(b.record.points) ? b.record.points : [];
                    _redraw(state);
                    _loadPdos(state);
                } catch (_) { /* the report as it was stays */ }
            };
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
                state.drawn = true;
                /* FIRST RENDER ON SCREEN: the picker drops its "Parsing..."
                 * line on this event (`lib/results/file-picker.js`). */
                root.document.dispatchEvent(new root.CustomEvent(
                    ((root.molbuilder || {}).constants || {})
                        .EVENT_INSPECTOR_READY || "molbuilder:inspector:ready",
                    { detail: { inspector: "transport" } }));
            })();
            return {
                /* THE POINT, SET FROM OUTSIDE -- ``{frame, bias}``: the
                 * ladder's row for a transmission run opens the report at
                 * its point (results.md § 2.4) -- before the record is drawn
                 * the starting selection, after it the same pick a row click
                 * makes. */
                selectPoint(at) {
                    if (state.drawn) { _select(state, at); return; }
                    if (at && at.bias !== undefined) state.bias = at.bias;
                    if (at && at.frame !== undefined && at.frame !== null)
                        state.pendingFrame = at.frame;
                },
                dispose() {
                    state.disposed = true;
                    try { state.unsubFrame && state.unsubFrame(); } catch (_) { /* gone */ }
                    try { state.unsubSelection && state.unsubSelection(); } catch (_) { /* gone */ }
                    try { state.unsubPlay && state.unsubPlay(); } catch (_) { /* gone */ }
                    try { state.handle && state.handle.dispose(); } catch (_) { /* gone */ }
                    host.querySelectorAll(".plot-box").forEach(
                        (p) => root.molbuilder.plotBox.release(p));
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
