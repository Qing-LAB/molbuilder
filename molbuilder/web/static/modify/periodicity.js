/* Modify tab -- the Cell op-tab's per-GROUP periodicity editors.
 *
 * Contract: docs/model/structure-periodicity.md § 3b; docs/web/molview.md § 9.3.
 * Owns:     the form. One Update button per group (vacuum / periodicity / unit
 *           cell / origin) so each can independently stay at its default or be
 *           committed. Editing a field only stages; the button commits.
 * Called by: modify/selection-bootstrap.js, which mounts the viewer and hands it
 *           here. Nothing self-starts.
 *
 * WHERE THE VALUES COME FROM. Two reads, and the pair is the answer:
 *   getUnitCellInfo()  -- the cell AS IT WILL BE USED, defaults filled in, never null
 *   getUnitCell() / getUnitCellOrigin() / getAxisKind() / getVacuum()
 *                      -- what the structure ITSELF says, null where it says nothing
 * So "this value is a default" is "the raw read is null while the effective read
 * has one". The server works the cell out once and sends both halves; nothing here
 * decides which values are defaults, because a second opinion is a second resolver.
 *
 * This used to ask for getVacuumInfo / getAxisKindInfo / getUnitCellOriginInfo and
 * read `.isDefault` off them. None of those has ever existed. Every call was
 * written `w.getVacuumInfo ? … : fallback`, so the panel showed "(default)" on
 * every row for every structure instead of failing.
 *
 * (k-grid is NOT here: it is a reciprocal-space sampling knob on the config, not
 * geometry.)
 */
"use strict";

export function init(viewer) {
    var root = window;
    var AXIS = ["isolated", "periodic", "transport"];

    function $(id) { return document.getElementById(id); }
    // THE VIEWER THIS PAGE MOUNTED, handed in above. Not looked up: a viewer
    // belongs to whoever mounted it (molview.md § 5.6).
    function data() {
        return (viewer && viewer.ok) ? viewer.data : null;
    }
    function hasStructure() {
        var w = data();
        return !!(w && w.getStructure());
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
                inp.className = "modify-cell-num";
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

        /* TWO READS: what will be USED, and what the structure itself SAYS.
         *
         * The pair is what "(default)" means — the structure says nothing here, so
         * the value on screen was worked out for it. The server resolves the cell
         * once and sends both halves; this reads them and decides nothing. */
        var used = w.getUnitCellInfo();          // never null
        var rawCell   = w.getUnitCell();
        var rawOrigin = w.getUnitCellOrigin();
        var rawAxis   = w.getAxisKind();
        var rawVacuum = w.getVacuum();

        // An EXPLICIT cell is the source of truth: vacuum is inert (the box comes
        // back verbatim) and "Use default" is invalid for a periodic/transport axis
        // (you cannot derive a commensurate lattice from a bounding box -- clearing
        // it would make the box DISAPPEAR).  Read first so the groups below react.
        var explicitCell = rawCell !== null;
        var axes = used.axis_kind || [];
        var hasPeriodicAxis = axes.some(function (k) {
            return k === "periodic" || k === "transport";
        });

        var vac = used.vacuum || [0, 0, 0];
        setIdle($("pv-vac-a"), round(vac[0] || 0));
        setIdle($("pv-vac-b"), round(vac[1] || 0));
        setIdle($("pv-vac-c"), round(vac[2] || 0));
        // Vacuum ALWAYS has a value -- "unset" is not a state it has -- so what
        // marks it a default is it being zero on every axis. With an explicit cell
        // it grows nothing, so the group says so instead of silently doing nothing.
        tag("pv-vac-tag", explicitCell
            ? false
            : !rawVacuum || rawVacuum.every(function (x) { return !x; }));
        // Vacuum edits are ALLOWED under an explicit cell -- they reset the box to
        // the derived regime (confirm-gated in wire()).  The note warns; the button
        // stays enabled.
        var vacNa = $("pv-vac-na");
        if (vacNa) vacNa.hidden = !explicitCell;

        ["pv-axis-a", "pv-axis-b", "pv-axis-c"].forEach(function (id, i) {
            var sel = $(id);
            if (sel && document.activeElement !== sel) sel.value = axes[i] || "isolated";
        });
        // Unset, or every axis isolated: a fresh molecule loads all-isolated, and
        // that is still the default configuration rather than a choice made.
        tag("pv-axis-tag", !rawAxis
            || rawAxis.every(function (k) { return k === "isolated"; }));

        if (cellInputs.length === 9) {
            var m = used.cell;
            for (var r = 0; r < 3; r++) {
                for (var col = 0; col < 3; col++) {
                    setIdle(cellInputs[r * 3 + col], m ? round(m[r][col]) : "");
                }
            }
            tag("pv-cell-tag", !explicitCell);
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
        // corner the box is actually drawn from; editing it sets an explicit
        // cell_origin.  cell_origin is ONLY meaningful with an explicit cell (the
        // dataclass drops it otherwise), so the group is enabled only there -- with a
        // bbox+vacuum cell the corner is auto and there is nothing to override.
        {
            var ov = used.cell_origin || [0, 0, 0];
            setIdle($("pv-org-a"), round(ov[0] || 0));
            setIdle($("pv-org-b"), round(ov[1] || 0));
            setIdle($("pv-org-c"), round(ov[2] || 0));
            tag("pv-org-tag", rawOrigin === null);
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
    /* THREE OUTCOMES, and the middle one is why this exists (molview.md § 6.9):
     *
     *   the cell block  the edit happened
     *   a THROW         the server refused it, and the reason it threw IS the
     *                   server's own sentence -- "swap two lattice vectors or
     *                   negate one" -- which is what the user needs to read
     *   null            there was nothing to do; nothing to say either
     *
     * This used to expect a server envelope back — `{ok, error, notices}` — and
     * got the cell block, so the error branch never ran; and on a refusal it got
     * null and skipped both branches, so the Update button did nothing at all
     * while the server had answered with exactly the sentence that would have
     * explained it.
     *
     * The notices are NOT in the answer: they are delivered inside the door and
     * MolView draws them (§ 6.8). Pushing them to the notification bar as well
     * would put one fact in two places. */
    function commitOp(op, payload) {
        var w = data();
        if (!w) return Promise.resolve();
        return Promise.resolve()
            .then(function () { return w.commitPeriodicityOp(op, payload); })
            .then(function () { refresh(); })
            .catch(function (err) {
                var notify = (window.molbuilder || {}).notify;
                var said = (err && err.message) || "the cell edit did not happen";
                if (notify && notify.show) {
                    notify.show({ id: "periodicity-error", level: "error",
                                  message: said });
                }
                refresh();
            });
    }
    // Confirm-before for the reset-to-derived edits (vacuum / axis kinds
    // under an explicit cell): the box boundary is about to move — the
    // user must know BEFORE committing (§ 6.2 v3).
    function _confirmReset(body) {
        var w = data();
        // Already derived -> nothing to reset, so nothing to confirm. "Derived"
        // is the structure stating no cell of its own; asking `getUnitCellInfo`
        // would answer with the box that was worked out FOR it, which is never
        // absent and so never told us anything here.
        if (!w || w.getUnitCell() === null) return Promise.resolve(true);
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

    function start() {
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

    start();
}
