/* Modify tab -- per-GROUP periodicity editors (structure-periodicity.md § 3b).
 *
 * MolView only DISPLAYS periodicity; the actual edits live in the Modify "Cell" op-tab,
 * one Update per group (vacuum / pbc / unit cell) so each can independently stay at its
 * default or be committed.  Each group reads the current value from the SAME molview
 * accessors the display uses (getVacuumInfo / getAxisKindInfo / getUnitCellInfo
 * -> { value, isDefault }) and commits through the MolView DATA API
 * (window.molbuilder.molview.data -- NOT the persistence-only workspace):
 *   - vacuum / pbc / unit cell -> data.commitPeriodicity (re-resolves the effective cell
 *     through the ONE server resolver, § 3a).
 * (k-grid is NOT here: it's a reciprocal-space sampling knob on SiestaConfig /
 * TransportConfig, not geometry -- structure-periodicity.md.)
 * No auto-commit: editing a field only stages; the group's Update button commits.
 */
(function () {
    "use strict";
    var root = window;
    var AXIS = ["isolated", "periodic", "transport"];

    function $(id) { return document.getElementById(id); }
    // DATA access (getStructure / get*Info / commitPeriodicity / subscribe) is the
    // in-memory model on molview.data; ``workspace`` is persistence-only now.
    function data() {
        return root.molbuilder && root.molbuilder.molview
            && root.molbuilder.molview.data;
    }
    function hasStructure() {
        var w = data();
        return !!(w && typeof w.getStructure === "function" && w.getStructure());
    }
    function round(n) { return Math.round(Number(n) * 1000) / 1000; }
    function setIdle(el, val) {
        if (el && document.activeElement !== el) el.value = val;
    }
    function tag(id, isDefault) {
        var el = $(id);
        if (el) el.textContent = isDefault ? "(default)" : "";
    }

    var cellInputs = [];   // nine <input>, row-major

    function buildCellGrid() {
        var grid = $("pv-cell-grid");
        if (!grid || grid.childNodes.length) return;
        cellInputs = [];
        for (var r = 0; r < 3; r++) {
            for (var c = 0; c < 3; c++) {
                var inp = document.createElement("input");
                inp.type = "number"; inp.step = "0.1";
                inp.className = "pv-num";
                inp.setAttribute("aria-label", "cell " + r + "," + c);
                grid.appendChild(inp);
                cellInputs.push(inp);
            }
        }
    }
    function fillAxisOptions() {
        ["pv-axis-a", "pv-axis-b", "pv-axis-c"].forEach(function (id) {
            var sel = $(id);
            if (!sel || sel.options.length) return;
            AXIS.forEach(function (k) {
                var o = document.createElement("option");
                o.value = k; o.textContent = k; sel.appendChild(o);
            });
        });
    }

    // Mirror the in-memory periodicity into the inputs (don't clobber a field the user is
    // editing) + tag each group "(default)" when it was never set.
    function refresh() {
        var panel = $("optab-panel-cell");
        if (!panel) return;
        var w = data();
        var has = hasStructure();
        var hint = $("pv-empty-hint");
        if (hint) hint.hidden = has;
        panel.querySelectorAll("fieldset").forEach(function (fs) { fs.disabled = !has; });
        if (!has || !w) return;

        // An EXPLICIT cell (isDefault === false) is the source of truth: vacuum is
        // inert (resolve_cell returns the cell verbatim) and "Use default" is invalid
        // for a periodic/transport axis (you can't derive a commensurate lattice from
        // a bbox -- clearing the cell would make the box DISAPPEAR).  Read both first
        // so the group guards below can react (structure-periodicity.md § 3c).
        var cellInfo = w.getUnitCellInfo ? w.getUnitCellInfo() : { isDefault: true };
        var axisInfo = w.getAxisKindInfo ? w.getAxisKindInfo() : { value: [] };
        var explicitCell = !cellInfo.isDefault;
        var hasPeriodicAxis = (axisInfo.value || []).some(function (k) {
            return k === "periodic" || k === "transport";
        });

        if (w.getVacuumInfo) {
            var v = w.getVacuumInfo(), vv = v.value || [0, 0, 0];
            setIdle($("pv-vac-a"), round(vv[0] || 0));
            setIdle($("pv-vac-b"), round(vv[1] || 0));
            setIdle($("pv-vac-c"), round(vv[2] || 0));
            // Vacuum only grows a DERIVED isolated axis; with an explicit cell it does
            // nothing, so mark the group "not applicable" instead of a silent no-op.
            tag("pv-vac-tag", explicitCell ? false : v.isDefault);
            var vacNa = $("pv-vac-na");
            if (vacNa) vacNa.hidden = !explicitCell;
            var vacBtn = $("pv-vac-update");
            if (vacBtn) vacBtn.disabled = explicitCell;
        }
        if (w.getAxisKindInfo) {
            var av = axisInfo.value || [];
            ["pv-axis-a", "pv-axis-b", "pv-axis-c"].forEach(function (id, i) {
                var sel = $(id);
                if (sel && document.activeElement !== sel) sel.value = av[i] || "isolated";
            });
            tag("pv-axis-tag", axisInfo.isDefault);
        }
        if (w.getUnitCellInfo && cellInputs.length === 9) {
            var m = cellInfo.value;
            for (var r = 0; r < 3; r++) {
                for (var col = 0; col < 3; col++) {
                    setIdle(cellInputs[r * 3 + col], m ? round(m[r][col]) : "");
                }
            }
            tag("pv-cell-tag", cellInfo.isDefault);
        }
        // "Use default" clears the explicit cell -> resolve_cell(); on a periodic /
        // transport axis that RAISES (no bbox-derived lattice), so disable it there --
        // offering it is what made the box vanish (§ 3c symptom c).
        var resetBtn = $("pv-cell-reset");
        if (resetBtn) {
            resetBtn.disabled = hasPeriodicAxis;
            resetBtn.title = hasPeriodicAxis
                ? "Not available: a periodic/transport axis needs an explicit cell "
                  + "(a bounding box can't define a commensurate lattice)."
                : "Clear the explicit cell and fall back to the bbox+vacuum box.";
        }
        var calBtn = $("pv-cell-calibrate");
        if (calBtn) calBtn.disabled = !explicitCell;
    }

    function num(id, dflt, isInt) {
        var raw = $(id) ? Number($(id).value) : NaN;
        if (!isFinite(raw)) return dflt;
        return isInt ? Math.max(1, Math.round(raw)) : raw;
    }
    function commit(patch) {
        var w = data();
        if (!w || typeof w.commitPeriodicity !== "function") return;
        Promise.resolve(w.commitPeriodicity(patch)).then(refresh);
    }

    function wire() {
        var vac = $("pv-vac-update");
        if (vac) vac.addEventListener("click", function () {
            commit({ vacuum: [num("pv-vac-a", 0), num("pv-vac-b", 0), num("pv-vac-c", 0)] });
        });
        var axis = $("pv-axis-update");
        if (axis) axis.addEventListener("click", function () {
            commit({ axis_kind: ["pv-axis-a", "pv-axis-b", "pv-axis-c"].map(function (id) {
                return $(id) ? $(id).value : "isolated";
            }) });
        });
        var cell = $("pv-cell-update");
        if (cell) cell.addEventListener("click", function () {
            var m = [];
            for (var r = 0; r < 3; r++) {
                var row = [];
                for (var col = 0; col < 3; col++) {
                    var raw = Number(cellInputs[r * 3 + col].value);
                    row.push(isFinite(raw) ? raw : 0);
                }
                m.push(row);
            }
            commit({ cell: m });   // explicit cell -- server keeps it, no re-resolve
        });
        var reset = $("pv-cell-reset");
        if (reset) reset.addEventListener("click", function () {
            commit({ cell: null });   // clear -> fall back to the resolved bbox+vacuum
        });
        // § 3c: calibrate is a GEOMETRY op (moves atoms into the cell), so it goes
        // through applyOp -- not commitPeriodicity (which only re-resolves the cell).
        var cal = $("pv-cell-calibrate");
        if (cal) cal.addEventListener("click", function () {
            var w = data();
            if (w && typeof w.applyOp === "function") {
                Promise.resolve(w.applyOp("calibrate", {})).then(refresh, function () {});
            }
        });
    }

    function init() {
        if (!$("optab-panel-cell")) return;
        buildCellGrid();
        fillAxisOptions();
        wire();
        refresh();
        // Refresh on ANY workspace change (load, modify op, or another periodicity edit):
        // ws.subscribe fires on the canvas onChange too (dispatcher wires cs.onChange).
        var w = data();
        if (w && typeof w.subscribe === "function") w.subscribe(refresh);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
