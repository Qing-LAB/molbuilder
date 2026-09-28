/* fc-sweep.js — a SIESTA vibration's displacement sweep, summarized.
 *
 * `jobset summarize run` on a vibration with two or more force-constant stages
 * writes `<label>.fc-sweep.json` at the calculation root
 * (`spectra/displacement_sweep.py`, `engines/vibration.md` § 5.9): each
 * stage and what it varied, each mode's frequency at every stage (matched by
 * shape, not rank), how far the force constants moved, and where every
 * stage's own files are.  This draws that record: two tables and the paths.
 *
 * WHAT THIS IS AND IS NOT.  It is the comparison the record carries.  It is
 * NOT a stage's spectrum -- each stage's full result stays in its own attempt,
 * and the table names where, so a person opens it there with the spectrum
 * viewer.  Nothing here is recomputed; a stage still running reads as
 * pending, never as a failure.
 *
 * Reads through `ctx.readFile` -- the one shared reader every presenter uses
 * (`web/presenters.md` § 2) -- so it opens no route of its own.
 */
(function (root) {
    "use strict";

    function _el(tag, cls, text) {
        const e = root.document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined) e.textContent = text;
        return e;
    }

    function _num(x, digits) {
        if (x === null || x === undefined) return "—";
        const n = Number(x);
        return isFinite(n) ? n.toFixed(digits) : "—";
    }

    function _row(cells, cls) {
        const tr = _el("tr", cls);
        cells.forEach((c) => tr.appendChild(
            typeof c === "string" ? _el("td", null, c) : c));
        return tr;
    }

    function render(host, rec) {
        host.innerHTML = "";
        const wrap = _el("div", "fcsweep-record");
        const stages = Array.isArray(rec.stages) ? rec.stages : [];
        const modes = Array.isArray(rec.modes) ? rec.modes : [];
        const pending = Array.isArray(rec.pending) ? rec.pending : [];
        const names = stages.map((s) => s.name);
        const tol = rec.tolerance_cm1;

        wrap.appendChild(_el("h3", "fcsweep-title",
            "Displacement sweep — " + (rec.label || "(unlabelled)")));
        wrap.appendChild(_el("p", "fcsweep-sub",
            stages.length + " force-constant stage"
            + (stages.length === 1 ? "" : "s") + " with a result"
            + (pending.length ? ", " + pending.length + " pending" : "")
            + "; modes matched by shape to “"
            + (rec.reference_stage || "") + "”"
            + (tol === null || tol === undefined
                ? "; no tolerance was given, so nothing is flagged"
                : "; flagged when the spread exceeds " + tol + " cm⁻¹")));

        /* THE STAGES: what each varied, the displacement SIESTA used, its own
         * numerics, and where its files are -- the raw data stays there. */
        const st = _el("table", "fcsweep-table");
        st.appendChild(_row(["Stage", "δ (Å)", "Varies",
            "Asymmetry (eV/Å²)", "Stationary", "Modes",
            "Spectrum"].map((h) => _el("th", null, h))));
        stages.forEach((s) => {
            const varies = Object.entries(s.varies || {})
                .map(([k, v]) => k + " = " + v).join(", ");
            st.appendChild(_row([
                s.name, _num(s.fc_displacement_ang, 5),
                varies || "(the template’s values)",
                _num(s.fc_asymmetry_max_ev_ang2, 4),
                s.stationary === true ? "yes"
                    : s.stationary === false ? "NO" : "—",
                String(s.n_modes === undefined ? "—" : s.n_modes),
                s.spectrum || "—",
            ]));
        });
        pending.forEach((p) => {
            st.appendChild(_row([p.stage, "pending", p.why || "",
                "", "", "", ""], "fcsweep-pending"));
        });
        wrap.appendChild(st);

        /* THE MODES: one row each, one column per stage. */
        const mt = _el("table", "fcsweep-table");
        mt.appendChild(_row(["Mode"].concat(names.map((n) => n + " (cm⁻¹)"))
            .concat(["Spread", "Min. overlap"])
            .concat(tol === null || tol === undefined ? [] : ["Flag"])
            .map((h) => _el("th", null, h))));
        modes.forEach((m) => {
            const ovl = Object.values(m.overlap || {})
                .filter((o) => o !== null && o !== undefined);
            const cells = [String(m.index_1based)]
                .concat(names.map((n) => _num((m.frequency_cm1 || {})[n], 1)))
                .concat([_num(m.spread_cm1, 2),
                         ovl.length ? Math.min(...ovl).toFixed(4) : "—"]);
            if (!(tol === null || tol === undefined)) {
                cells.push(m.flagged === true ? "over"
                    : m.flagged === false ? "ok" : "—");
            }
            mt.appendChild(_row(cells, m.flagged === true ? "fcsweep-flagged"
                                                          : null));
        });
        wrap.appendChild(mt);

        (Array.isArray(rec.force_constants) ? rec.force_constants : [])
            .forEach((c) => {
                wrap.appendChild(_el("p", "fcsweep-note",
                    "Force constants, " + c.stage + " against " + c.against + ": "
                    + (c.max_abs_change_ev_ang2 === null
                        || c.max_abs_change_ev_ang2 === undefined
                        ? "not compared (" + (c.why || "") + ")"
                        : "largest change " + _num(c.max_abs_change_ev_ang2, 4)
                          + " eV/Å²"
                          + (c.relative_change === null
                             || c.relative_change === undefined ? ""
                             : " (" + (100 * c.relative_change).toPrecision(3)
                               + "% of the largest constant)"))));
            });
        wrap.appendChild(_el("p", "fcsweep-note",
            "Each stage’s full result — every mode, both eigenvector "
            + "forms, the thermochemistry — and its raw force constants stay "
            + "in its own attempt (the paths above); open a spectrum there to "
            + "see it."));
        host.appendChild(wrap);
    }

    const inspector = {
        name:        "fc-sweep",
        displayName: "Displacement sweep",
        isResult:    true,
        resultCategory: () => "Spectrum",
        // THE ROLE, when the server gave one (see the spectra inspector).
        match: (file, meta) => (meta && meta.role)
            ? meta.role === ".fc-sweep.json"
            : String(file).toLowerCase().endsWith(".fc-sweep.json"),

        mount(host, file, ctx) {
            let disposed = false;
            (async function () {
                let rec;
                try {
                    const body = await ctx.readFile(file);
                    if (!body || !body.ok) {
                        throw new Error("could not read " + file + ": "
                            + ((body && body.error) || "unknown"));
                    }
                    rec = JSON.parse(body.text || "");
                } catch (e) {
                    if (disposed) return;
                    if (ctx && ctx.showError) ctx.showError(String(e));
                    else host.textContent = String(e);
                    return;
                }
                if (disposed) return;
                render(host, rec);
                /* FIRST RENDER ON SCREEN: the picker drops its "Parsing…"
                 * line on this event (`lib/results/file-picker.js`) rather
                 * than waiting out its fallback timer.  A table draws
                 * synchronously, so there is nothing to wait for. */
                root.document.dispatchEvent(new root.CustomEvent(
                    ((root.molbuilder || {}).constants || {})
                        .EVENT_INSPECTOR_READY || "molbuilder:inspector:ready",
                    { detail: { inspector: "fc-sweep" } }));
            })();
            return {
                dispose() {
                    disposed = true;
                    host.innerHTML = "";
                },
            };
        },
    };

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.fcSweepInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
