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
    /* ── Taking a value off the STRUCTURE instead of the keyboard (§ 7) ──
     *
     * Both gestures STAGE.  They write into the very inputs a user could have
     * typed, and the group's own Update button remains the only thing that
     * commits -- so what is about to be sent is on screen first, and there is
     * no second commit path for the gate to stand in front of.
     */

    //: The nine inputs as a matrix.  A blank box is 0, exactly as Update reads it.
    function stagedCell() {
        var m = [];
        for (var r = 0; r < 3; r++) {
            var row = [];
            for (var c = 0; c < 3; c++) {
                var raw = Number(cellInputs[r * 3 + c].value);
                row.push(isFinite(raw) ? raw : 0);
            }
            m.push(row);
        }
        return m;
    }
    function setStagedRow(r, vec) {
        for (var c = 0; c < 3; c++) cellInputs[r * 3 + c].value = round(vec[c]);
    }
    function norm(v) {
        return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    function chosenAxis() {
        var sel = $("pv-cell-axis");
        var at = sel ? "abc".indexOf(sel.value) : -1;
        return at < 0 ? 0 : at;
    }

    /* THE PICKS COME FROM THE RULER, NOT THE SELECTION (molview.md § 11.6).
     *
     * An axis needs a FIRST and a SECOND -- reverse the pair and it negates --
     * and an origin needs exactly one atom.  Order and a count limit are what
     * the pick track promises; `selection` is a SET, for managing groups and
     * labels, where "these forty atoms" has no first and no second.
     *
     * This used to read `selection`'s `pickOrder`, a click-order shadow kept
     * in lock-step on the very store whose contract says order does not
     * matter -- and that shadow existed for the ANGLE VERTEX, a measurement,
     * so once measuring got its own track this gesture was its only user.
     * Reading the track directly retires it (user, 2026-08-31: "having
     * selection and this function overlapping seems functionally wrong").
     *
     * It also makes the direction PROVABLE.  The two fields could not be made
     * to disagree through the UI, so a reader consulting the wrong one passed
     * the whole suite; the ruler's list is ordered by construction and a test
     * can drive it. */
    function pickedInOrder() {
        var w = data();
        if (!w || !w.measurement) return [];
        return w.measurement.getState().picks;
    }
    //: Where those atoms are AT THE FRAME ON SCREEN -- the same door the
    //  measurement readout reads, so a trajectory gives the axis of the frame
    //  the user is looking at rather than of frame zero.
    function positionsOf(indices) {
        var w = data();
        var frame = w ? w.getFrameAllAtoms(w.currentFrame()) : null;
        if (!frame) return null;
        var out = [];
        for (var i = 0; i < indices.length; i++) {
            var p = frame[indices[i]];
            if (!p) return null;
            out.push(p);
        }
        return out;
    }

    /* THE REFUSAL THIS GESTURE WILL ACTUALLY HIT, said before the request.
     *
     * The gate refuses a left-handed cell outright (det <= 0, HTTP 400).  Typing
     * nine numbers rarely produces one by accident; picking three atom pairs
     * will produce one about half the time, so this stops being rare the moment
     * the gesture ships.
     *
     * ADVISORY, AND THE GATE STILL DECIDES.  This predicts the refusal rather
     * than replacing it -- there is no second rule here, only the same sign
     * read early.  Silent near zero: that is the NO-VOLUME finding, which
     * `cell.py` reports instead and deliberately does not also call
     * "left-handed", because giving one cause two names is what it avoids by
     * checking volume first. */
    //: Å³.  Below this the box has no VOLUME, which `cell.py` reports as its own
    //  finding -- so the handedness note stays quiet rather than giving one
    //  cause two names.
    var DET_QUIET = 1e-6;
    //: Å.  A row with no direction has no length to scale: rescaling (0,0,0) is
    //  a division by zero dressed as an edit.  Separate from DET_QUIET above
    //  because a determinant and a length are not the same quantity, and one
    //  constant standing for both is a coincidence waiting to be edited.
    var LENGTH_QUIET = 1e-9;
    function refreshHandedness() {
        var note = $("pv-cell-hand");
        if (!note) return;
        var m = stagedCell();
        var det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if (det < -DET_QUIET) {
            note.textContent =
                // The server prints `det:.6g`; 3 dp would render a small
                // negative determinant as "-0", which reads as a typo.
                "These three vectors are left-handed (det = "
                + Number(det.toPrecision(6))
                + "), and the server will refuse them. Swap any two rows, or "
                + "pick one axis's two atoms in the other order.";
            note.classList.add("modify-op-hint--warn");
            note.hidden = false;
        } else {
            note.hidden = true;
        }
    }

    function fillCellAxisOptions() {
        var sel = $("pv-cell-axis");
        if (!sel || sel.options.length) return;
        ["a", "b", "c"].forEach(function (name) {
            var o = document.createElement("option");
            o.value = name; o.textContent = name;
            sel.appendChild(o);
        });
    }
    // The chooser's option text carries each axis's CURRENT LENGTH, so all
    // three are readable from one control while the box below edits the chosen
    // one -- rather than a second row of three read-only numbers saying what
    // the box already says.
    function labelCellAxes() {
        var sel = $("pv-cell-axis");
        if (!sel || sel.options.length !== 3) return;
        var m = stagedCell();
        for (var r = 0; r < 3; r++) {
            sel.options[r].textContent = "abc"[r] + " — " + round(norm(m[r])) + " Å";
        }
    }
    function syncLengthBox() {
        var box = $("pv-cell-len");
        if (!box || document.activeElement === box) return;
        box.value = round(norm(stagedCell()[chosenAxis()]));
    }
    /* Both "Use selection" buttons say what they need and why they cannot run,
     * because a disabled button with no reason is a dead end. */
    /* THE CELL PAGE TURNS THE RULER ON, AND SAYS SO (molview.md § 11.6).
     *
     * The gesture needs ordered picks and the ruler is where they come from,
     * so reaching for it against a ruler the person left off would fail
     * quietly -- and it could not be fixed from here either, because the pick
     * buttons stay disabled until two atoms are picked and no atom can be
     * picked with the ruler off.  So opening this page turns it on.
     *
     * IT SAYS SO, and that is not politeness.  A mode that switches itself on
     * silently is exactly what § 11.2b's lane exists to prevent; announcing it
     * is what makes the same act support rather than surprise (user,
     * 2026-08-31: "lock measurement toggle to be on with a message to user
     * saying so").  Leaving the page releases the lock and LEAVES THE RULER
     * ON -- turning it off again is the person's to do, not ours to undo
     * behind them. */
    function rulerIsOn() {
        var w = data();
        return !!(w && w.measurement && w.measurement.getState().active);
    }
    function lockRulerForPicking() {
        var w = data();
        if (!w || !w.measurement) return;
        if (!rulerIsOn()) {
            w.measurement.setActive(true);
            var notify = (window.molbuilder || {}).notify;
            if (notify && notify.show) {
                notify.show({ id: "cell-ruler-lock", level: "info", message:
                    "Measuring is on: the Cell page picks atoms with the "
                    + "ruler, in the order you click them. Turn it off when "
                    + "you are done here." });
            }
        }
    }

    function refreshPickButtons() {
        var n = pickedInOrder().length;
        var ruler = rulerIsOn();
        var axisBtn = $("pv-cell-from-selection");
        if (axisBtn) {
            axisBtn.disabled = n !== 2;
            axisBtn.title = !ruler
                ? "Turn measuring on, then pick two atoms."
                : n === 2
                ? "Set this axis to the vector from the first picked atom to "
                  + "the second. Pick them the other way round to flip it."
                : "Pick exactly two atoms with the ruler (" + n + " picked).";
        }
        var orgBtn = $("pv-org-from-selection");
        if (orgBtn) {
            orgBtn.disabled = n !== 1;
            orgBtn.title = !ruler
                ? "Turn measuring on, then pick one atom."
                : n === 1
                ? "Put the box's low corner on the picked atom."
                : "Pick exactly one atom with the ruler (" + n + " picked).";
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
        // WHAT MARKS IT A DEFAULT: the structure states no vacuum of its own, so
        // `getVacuum()` answers null while `getUnitCellInfo()` still has a number
        // -- the raw-vs-effective pair this file's header describes.
        //
        // The comment here read "vacuum ALWAYS has a value -- unset is not a state
        // it has", which was true until 2026-08-03, when `vacuum` became Optional
        // so that "I want no gap" and "I never chose one" could stop being the
        // same value.  The all-zero test below still fires for pre-2026-08-03
        // sidecars, which say [0,0,0] and are READ as unset (cell-plan.md § 5).
        //
        // With an explicit cell it grows nothing, so the group says so instead of
        // silently doing nothing.
        tag("pv-vac-tag", explicitCell
            ? false
            : !rawVacuum || rawVacuum.every(function (x) { return !x; }));
        // Vacuum edits are ALLOWED under an explicit cell -- they reset the box to
        // the derived regime (confirm-gated in wire()).  The note warns; the button
        // stays enabled.
        // INERT, AND SHOWN TO BE.  The note appears, and the three inputs dim,
        // so the row does not look like an editable number that will move the
        // box -- it will not; an explicit cell IS the box
        // (structure-periodicity.md § 6.1a, matrix A).  They stay ENABLED on
        // purpose: typing here is how you go back to the derived regime, which
        // is a real thing to want and is confirm-gated in wire().
        var vacNa = $("pv-vac-na");
        if (vacNa) vacNa.hidden = !explicitCell;
        ["pv-vac-a", "pv-vac-b", "pv-vac-c"].forEach(function (id) {
            var f = $(id);
            if (f) f.classList.toggle("pv-field--inert", explicitCell);
        });

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
            labelCellAxes();
            syncLengthBox();
            refreshHandedness();
        }
        refreshPickButtons();
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
    function _fmtCell(m) {
        /* The cell about to be discarded, in the user's own numbers.  A
         * confirm that does not name what it destroys asks you to trust it. */
        try {
            return m.map(function (row) {
                return row.map(function (n) { return Number(n).toFixed(3); })
                          .join(", ");
            }).join(" | ");
        } catch (_) { return "the cell you typed"; }
    }

    function _confirmReset(body) {
        var w = data();
        // Already derived -> nothing to reset, so nothing to confirm. "Derived"
        // is the structure stating no cell of its own; asking `getUnitCellInfo`
        // would answer with the box that was worked out FOR it, which is never
        // absent and so never told us anything here.
        if (!w || w.getUnitCell() === null) return Promise.resolve(true);
        var wm = (window.molbuilder || {}).warningModal;
        if (!wm || !wm.confirm) return Promise.resolve(true);
        /* NAMES THE CELL, and says the edit is final (2026-08-03).  A
         * periodicity op never enters MolView's history -- commitPeriodicityOp
         * calls applyCell directly -- so Ctrl-Z cannot bring the cell back,
         * and the dialog has to say so before you agree rather than after. */
        return wm.confirm({
            title: "Replace the cell you typed?",
            body: "This clears the cell you typed (" + _fmtCell(w.getUnitCell())
                  + ") and works a new box out from the molecule. "
                  + body + " It cannot be undone.",
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
        // Editing any of the nine restates the lengths and re-checks the sign,
        // so the note tracks what is staged rather than what was last committed.
        cellInputs.forEach(function (inp) {
            inp.addEventListener("input", function () {
                labelCellAxes(); syncLengthBox(); refreshHandedness();
            });
        });
        var axisPick = $("pv-cell-axis");
        if (axisPick) axisPick.addEventListener("change", syncLengthBox);

        var fromSel = $("pv-cell-from-selection");
        if (fromSel) fromSel.addEventListener("click", function () {
            var picked = pickedInOrder();
            if (picked.length !== 2) return;
            var pos = positionsOf(picked);
            if (!pos) return;
            setStagedRow(chosenAxis(), [pos[1][0] - pos[0][0],
                                        pos[1][1] - pos[0][1],
                                        pos[1][2] - pos[0][2]]);
            labelCellAxes(); syncLengthBox(); refreshHandedness();
        });

        var setLen = $("pv-cell-set-len");
        if (setLen) setLen.addEventListener("click", function () {
            var want = Number($("pv-cell-len") ? $("pv-cell-len").value : NaN);
            if (!isFinite(want) || want <= 0) return;
            var r = chosenAxis();
            var row = stagedCell()[r];
            var have = norm(row);
            // A row with no direction has no length to scale: rescaling (0,0,0)
            // is a division by zero dressed as an edit.  Give it a direction
            // first -- two atoms, or the keyboard.
            if (have < LENGTH_QUIET) return;
            var k = want / have;
            setStagedRow(r, [row[0] * k, row[1] * k, row[2] * k]);
            labelCellAxes(); refreshHandedness();
        });

        var orgFromSel = $("pv-org-from-selection");
        if (orgFromSel) orgFromSel.addEventListener("click", function () {
            var picked = pickedInOrder();
            if (picked.length !== 1) return;
            var pos = positionsOf(picked);
            if (!pos) return;
            /* Written DIRECTLY, not through `setIdle`.
             *
             * NOT A BUG FIX, and it is worth saying so: I changed this
             * believing `setIdle` would skip a focused box and leave one third
             * of the origin stale, then mutation-tested it — pressing the
             * button BLURS the input first, so `setIdle` writes all three and
             * the case cannot be reached by clicking.  The test that claimed
             * otherwise was deleted rather than kept green over nothing.
             *
             * The direct write stays for the reason that does hold: a
             * button-driven write should not consult `document.activeElement`
             * at all — the user asked for exactly this value — and
             * `setStagedRow` above already writes directly, so going the other
             * way would make the two gestures differ for no reason. */
            ["a", "b", "c"].forEach(function (ax, i) {
                var box = $("pv-org-" + ax);
                if (box) box.value = round(pos[0][i]);
            });
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
        fillCellAxisOptions();
        wire();
        refresh();
        // Refresh on ANY workspace change (load, modify op, or another periodicity edit):
        // ws.subscribe fires on the canvas onChange too (dispatcher wires cs.onChange).
        var w = data();
        if (w && typeof w.subscribe === "function") w.subscribe(refresh);
        /* And the RULER, which `subscribe` above does not carry: it announces
         * data changes, and picking two atoms changes no data.  The two
         * buttons say how many atoms are picked, so without this they said it
         * once and then went stale.
         *
         * This followed the SELECTION until 2026-08-31, which is where the
         * picks used to come from.  Same reason, one track over. */
        if (w && w.measurement && typeof w.measurement.subscribe === "function") {
            w.measurement.subscribe(refreshPickButtons);
        }

        /* Opening this page is what "reaching for the gesture" IS -- there is
         * no earlier moment, because every pick control here is disabled until
         * atoms are picked and none can be picked with the ruler off.  Wired
         * to the tab button rather than a panel-visibility watcher, because
         * the button press is the intent and the panel is only its effect. */
        var cellTab = $("optab-btn-cell");
        if (cellTab) cellTab.addEventListener("click", lockRulerForPicking);
        var panelNow = $("optab-panel-cell");
        if (panelNow && !panelNow.hidden) lockRulerForPicking();
    }

    start();
}
