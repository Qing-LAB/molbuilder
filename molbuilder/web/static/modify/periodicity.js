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

// molview.data is MolView's live internal state -> LOOK IT UP at read time (molview-module.md
// §D.0), never import it. Returns whatever MolView currently has (null = nothing loaded).
function _mvdata() {
    return (window.molbuilder && window.molbuilder.molview
            && window.molbuilder.molview.data) || null;
}

(function () {
    "use strict";
    var root = window;
    var AXIS = ["isolated", "periodic", "transport"];

    function $(id) { return document.getElementById(id); }
    // DATA access (getStructure / get*Info / commitPeriodicity / subscribe) is the
    // in-memory model on molview.data; ``workspace`` is persistence-only now.
    function data() {
        return _mvdata();   // the in-memory molview.data model, imported from the door
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
            // § 6.2 v3: vacuum edits are ALLOWED under an explicit cell —
            // they reset the box to the derived regime (confirm-gated in
            // wire()).  The note warns; the button stays enabled.
            var vacNa = $("pv-vac-na");
            if (vacNa) vacNa.hidden = !explicitCell;
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
        // § 6.2 v3: no calibrate button — emission translates implicitly.

        // § 3c: the cell origin -- the low corner the box is drawn from.  Shows the
        // RESOLVED corner (getUnitCellOriginInfo().value); editing it sets an explicit
        // cell_origin.  cell_origin is ONLY meaningful with an explicit cell (the
        // dataclass drops it otherwise), so the group is enabled only there -- with a
        // bbox+vacuum cell the corner is auto and there is nothing to override.
        if (w.getUnitCellOriginInfo) {
            var oInfo = w.getUnitCellOriginInfo();
            var ov = oInfo.value || [0, 0, 0];
            setIdle($("pv-org-a"), round(ov[0] || 0));
            setIdle($("pv-org-b"), round(ov[1] || 0));
            setIdle($("pv-org-c"), round(ov[2] || 0));
            tag("pv-org-tag", oInfo.isDefault);
            var orgNa = $("pv-org-na");
            if (orgNa) orgNa.hidden = explicitCell;
            var orgBtn = $("pv-org-update");
            if (orgBtn) orgBtn.disabled = !explicitCell;
            var orgReset = $("pv-org-reset");
            if (orgReset) orgReset.disabled = !explicitCell;
        }
    }

    function num(id, dflt, isInt) {
        var raw = $(id) ? Number($(id).value) : NaN;
        if (!isFinite(raw)) return dflt;
        return isInt ? Math.max(1, Math.round(raw)) : raw;
    }
    // § 6.2 v3 — every edit goes through the ONE server door (Python owns
    // the change); this page renders what comes back and surfaces the
    // gate's notices through the app notification bar.
    function _surfaceNotices(notices) {
        var notify = (window.molbuilder || {}).notify;
        if (!notify || !notify.show) return;
        (notices || []).forEach(function (n, i) {
            notify.show({
                id:      "periodicity-" + i,
                level:   n.level === "warn" ? "warn" : "info",
                message: n.message,
            });
        });
    }
    function commitOp(op, payload) {
        var w = data();
        if (!w || typeof w.commitPeriodicityOp !== "function") return;
        return Promise.resolve(w.commitPeriodicityOp(op, payload))
            .then(function (res) {
                if (res && !res.ok && res.error) {
                    var notify = (window.molbuilder || {}).notify;
                    if (notify && notify.show) notify.show({
                        id: "periodicity-error", level: "error",
                        message: "Periodicity edit failed: " + res.error,
                    });
                } else if (res) {
                    _surfaceNotices(res.notices);
                }
                refresh();
            });
    }
    // Confirm-before for the reset-to-derived edits (vacuum / axis kinds
    // under an explicit cell): the box boundary is about to move — the
    // user must know BEFORE committing (§ 6.2 v3).
    function _confirmReset(body) {
        var w = data();
        var info = (w && w.getUnitCellInfo) ? w.getUnitCellInfo()
                                            : { isDefault: true };
        if (info.isDefault) return Promise.resolve(true);   // already derived
        var wm = (window.molbuilder || {}).warningModal;
        if (!wm || !wm.confirm) return Promise.resolve(true);
        return wm.confirm({
            title: "Reset the box to the derived regime?",
            body: body,
            confirmLabel: "Update and reset",
            cancelLabel:  "Cancel",
        });
    }

    function wire() {
        var vac = $("pv-vac-update");
        if (vac) vac.addEventListener("click", function () {
            var payload = [num("pv-vac-a", 0), num("pv-vac-b", 0),
                           num("pv-vac-c", 0)];
            _confirmReset(
                "Updating vacuum resets the explicit unit cell and origin: "
              + "the box is re-derived around the structure with the vacuum "
              + "placed symmetrically per direction — the cell boundary "
              + "will move."
            ).then(function (ok) { if (ok) commitOp("vacuum", payload); });
        });
        var axis = $("pv-axis-update");
        if (axis) axis.addEventListener("click", function () {
            var kinds = ["pv-axis-a", "pv-axis-b", "pv-axis-c"].map(
                function (id) { return $(id) ? $(id).value : "isolated"; });
            // Switching TO periodic keeps the explicit cell (no reset) —
            // the confirm applies only to the reset-to-derived path.
            var go = kinds.indexOf("periodic") !== -1
                ? Promise.resolve(true)
                : _confirmReset(
                    "Changing the periodicity resets the explicit unit "
                  + "cell and origin: the box is re-derived from the "
                  + "structure size and vacuum — the cell boundary will "
                  + "move.");
            go.then(function (ok) { if (ok) commitOp("axis_kind", kinds); });
        });
        // § 6.2 v3: a manual origin is respected verbatim; the server warns
        // that vacuum is no longer respected (only the cell parameters are).
        var org = $("pv-org-update");
        if (org) org.addEventListener("click", function () {
            commitOp("cell_origin", [num("pv-org-a", 0), num("pv-org-b", 0),
                                     num("pv-org-c", 0)]);
        });
        var orgReset = $("pv-org-reset");
        if (orgReset) orgReset.addEventListener("click", function () {
            commitOp("cell_origin", null);
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
            commitOp("cell", m);   // origin-first, then vacuum (§ 6.2 v3)
        });
        var reset = $("pv-cell-reset");
        if (reset) reset.addEventListener("click", function () {
            commitOp("cell", null);   // back to the derived regime
        });
        // § 6.2 v3: no calibrate handler — coordinate rewrites are not a
        // periodicity edit (emission translates implicitly; the explicit
        // rewrite lives with the Modify ops as /api/modify/calibrate).
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
