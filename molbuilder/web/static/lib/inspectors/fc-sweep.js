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

    /* The one element builder, `lib/dom.js`, loaded before this file on
     * /results.  Looked up when called, so a page that has not loaded it
     * still registers this viewer and fails only where it draws. */
    const _el = (tag, cls, text) => root.molbuilder.dom.el(tag, cls, text);

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

    /* One stage's own overrides, each value in its unit (the record carries
     * the catalogue's), so `0.02` is never read as the Å beside it. */
    function _varies(s) {
        const units = s.varies_units || {};
        return Object.entries(s.varies || {})
            .map(([k, v]) => k + " = " + v + (units[k] ? " " + units[k] : ""))
            .join(", ") || "(the template’s values)";
    }

    function _base(path) {
        return String(path || "").split("/").pop();
    }

    /* A table in its own scroller: wider than the card on a narrow panel,
     * it scrolls inside the card rather than spilling past it -- and stays
     * a table, which `display: block` on the table itself would not for
     * every screen reader. */
    function _scroller(table) {
        const box = _el("div", "fcsweep-scroll");
        box.appendChild(table);
        return box;
    }

    function render(host, rec) {
        host.innerHTML = "";
        const wrap = _el("div", "fcsweep-record card");
        const stages = Array.isArray(rec.stages) ? rec.stages : [];
        const modes = Array.isArray(rec.modes) ? rec.modes : [];
        const pending = Array.isArray(rec.pending) ? rec.pending : [];
        const failed = Array.isArray(rec.failed) ? rec.failed : [];
        const names = stages.map((s) => s.name);
        const ref = rec.reference_stage || "";
        const tol = rec.tolerance_cm1;

        wrap.appendChild(_el("h3", "fcsweep-title",
            "Displacement sweep — " + (rec.label || "(unlabelled)")));
        wrap.appendChild(_el("p", "fcsweep-sub",
            stages.length + " force-constant stage"
            + (stages.length === 1 ? "" : "s") + " with a result"
            + (pending.length ? ", " + pending.length + " pending" : "")
            + (failed.length ? ", " + failed.length + " failed" : "")
            + "; modes matched by shape to “" + ref + "”"
            + (tol === null || tol === undefined
                ? "; no tolerance was given, so nothing is flagged"
                : "; flagged when the spread exceeds " + tol + " cm⁻¹")));

        /* THE STAGES: what each varied, the displacement SIESTA used, its own
         * numerics, and where its files are -- the raw data stays there. */
        const st = _el("table", "fcsweep-table");
        st.appendChild(_row(["Stage", "δ used (Å)", "Varies",
            "Stationary (largest free-atom force / criterion, eV/Å)",
            "Asymmetry (eV/Å²)", "Modes (motions removed)", "SIESTA",
            "Files"].map((h) => _el("th", null, h))));
        stages.forEach((s) => {
            const f = s.max_force_free_ev_ang, c = s.force_criterion_ev_ang;
            st.appendChild(_row([
                s.name, _num(s.fc_displacement_ang, 5), _varies(s),
                (s.stationary === true ? "yes"
                    : s.stationary === false ? "NO" : "—")
                    + " (" + _num(f, 4) + " / " + _num(c, 3) + ")",
                _num(s.fc_asymmetry_max_ev_ang2, 4),
                String(s.n_modes === undefined ? "—" : s.n_modes) + " ("
                    + String(s.removed_motions === undefined
                        || s.removed_motions === null ? "—"
                        : s.removed_motions) + ")",
                s.engine_version || "—",
                (s.attempt || "—") + ": " + _base(s.spectrum)
                    + (s.fc_file ? ", " + _base(s.fc_file)
                                 : ", no force-constant file"),
            ]));
        });
        wrap.appendChild(_scroller(st));

        /* THE MODES: one row each, one column per stage -- beside the
         * reference, the change, the shapes' overlap, and the mode it
         * matched when that is not the same rank. */
        const mt = _el("table", "fcsweep-table");
        mt.appendChild(_row(["Mode"].concat(names.map((n) => n + " (cm⁻¹)"))
            .concat(["Spread (cm⁻¹)"])
            .concat(tol === null || tol === undefined ? [] : ["Flag"])
            .map((h) => _el("th", null, h))));
        modes.forEach((m) => {
            const cells = [String(m.index_1based)].concat(names.map((n) => {
                const f = (m.frequency_cm1 || {})[n];
                if (f === null || f === undefined) return "—";
                if (n === ref) return _num(f, 1);
                const d = (m.change_from_reference_cm1 || {})[n];
                const o = (m.overlap || {})[n];
                const j = (m.matched_index_1based || {})[n];
                const bits = [];
                if (d !== null && d !== undefined) {
                    bits.push((d >= 0 ? "+" : "") + _num(d, 2));
                }
                if (o !== null && o !== undefined) bits.push("overlap " + _num(o, 4));
                if (j !== null && j !== undefined && j !== m.index_1based) {
                    bits.push("as mode " + j);
                }
                return _num(f, 1) + (bits.length ? " (" + bits.join("; ") + ")"
                                                 : "");
            })).concat([_num(m.spread_cm1, 2)]);
            if (!(tol === null || tol === undefined)) {
                cells.push(m.flagged === true ? "over"
                    : m.flagged === false ? "ok" : "—");
            }
            mt.appendChild(_row(cells, m.flagged === true ? "fcsweep-flagged"
                                                          : null));
        });
        wrap.appendChild(_scroller(mt));

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

        /* THE STAGES WITHOUT A RESULT, in their attempt's own words: still
         * to come is pending, never a failure; a run that failed says so. */
        if (pending.length || failed.length) {
            const wt = _el("table", "fcsweep-table");
            wt.appendChild(_row(["Stage", "", "State", "Why"]
                .map((h) => _el("th", null, h))));
            pending.forEach((p) => wt.appendChild(_row(
                [p.stage, "pending", p.state || "", p.detail || ""],
                "fcsweep-pending")));
            failed.forEach((p) => wt.appendChild(_row(
                [p.stage, "failed", p.state || "", p.detail || ""],
                "fcsweep-failed")));
            wrap.appendChild(_scroller(wt));
        }

        wrap.appendChild(_el("p", "fcsweep-note",
            "Two modes a few wavenumbers apart can mix between stages: a low "
            + "overlap there is the pair turning within its own plane, not a "
            + "changed motion.  Each stage’s full result — every mode, both "
            + "eigenvector forms, the thermochemistry — and its raw force "
            + "constants stay in its own attempt (the Files column); open a "
            + "spectrum there to see it."));
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
