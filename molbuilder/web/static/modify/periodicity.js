/* Modify tab -- the Cell op-tab's per-GROUP periodicity editors.
 *
 * Contract: docs/model/structure-periodicity.md § 7; docs/web/molview.md § 9.3.
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

    /* WHICH REGIME THE BOX IS IN -- the fact that decides what every other
     * control on this panel does, made into a control of its own (user,
     * 2026-09-07).  It was always there and never shown: you entered
     * "explicit" by typing a cell and left it by pressing a button called
     * "Update vacuum", with a modal afterwards to explain what that had done.
     *
     * On the wire it is not a field -- `cell: null` IS the derived regime
     * (structure-periodicity.md § 6.1) -- so this is a reading of the
     * structure, never a fifth thing to keep in step with it. */
    var REGIMES = [
        ["derived", "Derived from the atoms"],
        ["explicit", "An explicit cell I set"],
    ];
    var REGIME_NOTES = {
        derived: "The structure's extent plus the vacuum below, on each side. "
               + "It follows the atoms as you edit them.",
        explicit: "The cell you set below is the box, exactly. Vacuum becomes "
               + "reference-only, and a periodic axis needs this regime.",
    };

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
    /* One radio group, built rather than written into the template, so the
     * option list and the note that explains each option cannot drift apart. */
    function buildRegimeRadios() {
        var box = $("pv-regime-radios");
        if (!box || box.childNodes.length) return;
        REGIMES.forEach(function (pair) {
            var lbl = document.createElement("label");
            var inp = document.createElement("input");
            inp.type = "radio";
            inp.name = "pv-regime";
            inp.value = pair[0];
            inp.checked = (pair[0] === "derived");
            inp.addEventListener("change", renderRegime);
            lbl.appendChild(inp);
            lbl.appendChild(document.createTextNode(" " + pair[1]));
            box.appendChild(lbl);
        });
    }
    function regime() {
        var el = document.querySelector('input[name="pv-regime"]:checked');
        return el ? el.value : "derived";
    }
    function setRegime(value) {
        var el = document.querySelector(
            'input[name="pv-regime"][value="' + value + '"]');
        if (el) el.checked = true;
    }
    /* THE PANEL SHOWS THE FIELDS THE CHOSEN REGIME USES, and nothing else.
     * The alternative -- which this replaced -- was showing all of them and
     * dimming the inert ones with a note saying why, which is the same
     * information arranged so the reader has to assemble it. */
    function renderRegime() {
        var explicit = regime() === "explicit";
        var derived = $("pv-derived-only");
        if (derived) derived.hidden = explicit;
        document.querySelectorAll("#optab-panel-cell .pv-explicit-only")
            .forEach(function (fs) { fs.hidden = !explicit; });
        var note = $("pv-regime-note");
        if (note) note.textContent = REGIME_NOTES[regime()] || "";
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
    /* WHERE THE PICKED ATOMS ARE comes from the module (§ 11.6).
     *
     * This walked `getFrameAllAtoms(currentFrame())` itself, which was the
     * SAME walk the module's own readout does, staleness guard included -- one
     * question answered in two places, one of them outside the module that
     * owns it.  `measurement.positions()` is that answer. */

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
    /* The MODULE decides whether the ruler needed turning on; this page only
       says so, because only it knows which panel is asking and the notice is
       the app's own component (MolView reads no global, § 4). */
    function sayRulerIsOn(where) {
        var notify = (window.molbuilder || {}).notify;
        if (!notify || !notify.show) return;
        notify.show({ id: "ruler-on-" + where.toLowerCase().replace(/\s+/g, "-"),
            level: "info", message: "Measuring is on: the " + where
            + " picks atoms with the ruler, in the order you click them. "
            + "Turn it off when you are done here." });
    }

    function rulerIsOn() {
        var w = data();
        return !!(w && w.measurement && w.measurement.getState().active);
    }

    /* ONE RULE FOR BOTH GESTURES: read the picks in order and take what you
     * need from the front.  An axis is the first two, an origin is the first.
     *
     * There is no counting policy here and there should not be -- the track
     * belongs to MolView and this page just asks it what is picked, in order
     * (`molview.md` § 11.6).  Both buttons demanded an EXACT count until
     * 2026-08-31, which turned the ruler's own behaviour into bookkeeping for
     * the user: it holds up to three and drops the oldest at the fourth, so
     * after two picks the count runs 2, 3, 3, 3 and never returns to one.  The
     * origin button was then permanently dead for anyone who had measured
     * anything, with the remedy -- the ruler's Clear -- named nowhere. */
    function refreshPickButtons() {
        var n = pickedInOrder().length;
        var ruler = rulerIsOn();
        var axisBtn = $("pv-cell-from-selection");
        if (axisBtn) {
            axisBtn.disabled = n < 2;
            axisBtn.title = !ruler
                ? "Turn measuring on, then pick two atoms."
                : n < 2
                ? "Pick two atoms with the ruler."
                : "Set this axis to the vector from the first picked atom to "
                  + "the second. Pick them the other way round to flip it.";
        }
        var orgBtn = $("pv-org-from-selection");
        if (orgBtn) {
            orgBtn.disabled = n < 1;
            orgBtn.title = !ruler
                ? "Turn measuring on, then pick an atom."
                : n < 1
                ? "Pick an atom with the ruler."
                : "Put the box's low corner on the first atom you picked.";
        }
    }

    /* THE ATOMS' SPAN ALONG THE CHOSEN AXIS, and the gap a person adds to it.
     *
     * `c` is measured and set, never invented (`junction-cell.md` § 6).  The
     * builder stopped padding on 2026-08-31, so this is where a slab's box
     * gets its length along the transport axis -- and these two controls exist
     * to make the measuring cheap, not to make the decision.
     *
     * The SPAN is a fact about the atoms, so it is displayed rather than left
     * as arithmetic for a person to redo.  The GAP is theirs: for a boundary
     * meant to continue the crystal it is the layer spacing (pick two atoms in
     * adjacent layers, read the ruler's signed Δ); for a free surface it is
     * however much vacuum the run needs.  Nothing here judges which was meant
     * -- `classify_seam` reports on what was set, on every build. */
    function spanAlong(axisRow) {
        var w = data();
        var frame = w ? w.getFrameAllAtoms(w.currentFrame()) : null;
        if (!frame || !frame.length) return null;
        var row = stagedCell()[axisRow];
        var len = norm(row);
        /* Along the axis's own direction when it has one, so a tilted `c`
           measures the span that matters rather than the bounding box's.
           With no direction yet, z is the honest reading FOR `c` ONLY -- it is
           the axis this control exists for and the one a slab is built along.
           For `a` or `b` it is not honest at all: it reported the span
           measured along z under `a`'s label, and pressing Use then wrote
           `a = (0, 0, want)`, which lies on top of `c` and collapses the box
           to no volume.  No direction, no answer. */
        var u;
        if (len > LENGTH_QUIET) {
            u = [row[0] / len, row[1] / len, row[2] / len];
        } else if (axisRow === 2) {
            u = [0, 0, 1];
        } else {
            return null;
        }
        var lo = Infinity, hi = -Infinity;
        for (var i = 0; i < frame.length; i++) {
            var p = frame[i];
            var t = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
            if (t < lo) lo = t;
            if (t > hi) hi = t;
        }
        return (hi > lo || hi === lo) ? hi - lo : null;
    }

    /* WHAT GAP DID THE USER ASK FOR -- one reader, because two disagreed.
       An empty box is UNANSWERED (NaN), not zero: `Number("")` is 0, which
       enabled Use and printed "= 0" as though the user had chosen it. */
    function askedGap() {
        var box = $("pv-cell-gap");
        var raw = box ? box.value : "";
        return (raw === "" || raw == null) ? NaN : Number(raw);
    }

    function renderSpan() {
        var out = $("pv-cell-span");
        var btn = $("pv-cell-span-plus-gap");
        var note = $("pv-cell-span-note");
        if (!out) return;
        var span = spanAlong(chosenAxis());
        var gap = askedGap();
        out.textContent = span == null ? "\u2014" : round(span) + " \u00c5";
        var ready = span != null && isFinite(gap) && gap >= 0;
        if (btn) {
            btn.disabled = !ready;
            btn.title = span == null
                ? "Load a structure first."
                : !isFinite(gap)
                ? "Type the gap to add to the span."
                : "Set this axis to " + round(span) + " + " + round(gap)
                  + " = " + round(span + gap) + " \u00c5.";
        }
        if (note) {
            note.hidden = !ready;
            if (ready) {
                note.textContent =
                    "abc".charAt(chosenAxis()) + " = "
                    + round(span) + " + " + round(gap) + " = "
                    + round(span + gap) + " \u00c5.  For a boundary that "
                    + "continues the crystal the gap is one layer spacing -- "
                    + "measure it with the ruler.  The seam check reports on "
                    + "what you set.";
                note.classList.remove("modify-op-hint--warn");
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
        /* THE TWO ACTIONS ARE NOT IN A FIELDSET, so the sweep above does not
         * reach them.  The four "Update ..." buttons they replaced each sat
         * inside the group they committed and went dead with it; one commit
         * for the whole panel has nowhere to sit but outside, and would
         * otherwise stay live over an empty canvas -- offering to apply a box
         * to no atoms, which the gate refuses in a sentence nobody should have
         * had to read. */
        ["pv-apply", "pv-revert"].forEach(function (id) {
            var b = $(id);
            if (b) b.disabled = !has;
        });
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

        /* BLANK MEANS THE DEFAULT, and the default is shown as the
         * PLACEHOLDER -- the same idiom the origin uses below, for the same
         * reason.  Vacuum has THREE states (§ 6.1): a number, zero, and never
         * chosen.  Filling the boxes with the EFFECTIVE value collapsed the
         * third into the first, so pressing Apply after changing something
         * else stamped "3" into the structure as a value the user had chosen
         * -- the box does not move, but the "(default)" mark goes, and with it
         * the record that nobody had decided. */
        var vac = used.vacuum || [0, 0, 0];
        ["pv-vac-a", "pv-vac-b", "pv-vac-c"].forEach(function (id, i) {
            var f = $(id);
            if (!f) return;
            f.placeholder = String(round(vac[i] || 0));
            setIdle(f, rawVacuum ? round(rawVacuum[i] || 0) : "");
        });
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
        /* THE SWITCH IS SET FROM THE STRUCTURE, never the other way round.
         * `cell === null` IS the derived regime -- there is no stored flag to
         * read and none to keep in step (structure-periodicity.md § 6.1).
         *
         * Only while the panel is not being edited: `refresh` runs on every
         * store change, and moving the radio under someone who has just
         * chosen the other regime would undo the choice they are in the middle
         * of describing. */
        if (!document.activeElement
                || !document.activeElement.closest
                || !document.activeElement.closest("#optab-panel-cell")) {
            setRegime(explicitCell ? "explicit" : "derived");
        }
        renderRegime();

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
        renderSpan();
        /* "Use default" IS THE REGIME SWITCH NOW.  The button that stood here
         * cleared the explicit cell, which a periodic axis cannot survive -- a
         * bounding box is not a commensurate lattice -- so it had to disable
         * itself on exactly the structures where a user is most likely to
         * reach for it, with a tooltip as the only explanation.  Choosing
         * "derived" beside an axis that needs a lattice is refused by the one
         * gate, in a sentence, on Apply. */
        // § 6.2 v3: no calibrate button — emission translates implicitly.

        // § 3c: the cell origin -- the low corner the box is drawn from.  Shows the
        // corner the box is actually drawn from; editing it sets an explicit
        // cell_origin.  cell_origin is ONLY meaningful with an explicit cell (the
        // dataclass drops it otherwise), so the group is enabled only there -- with a
        // bbox+vacuum cell the corner is auto and there is nothing to override.
        {
            /* BLANK MEANS DERIVE, and the derived value is shown as the
             * PLACEHOLDER rather than as content.  Filling the boxes with it
             * -- which is what happened before -- made "no origin stated" and
             * "this exact origin stated" look identical, so pressing Apply
             * turned a derived corner into a fixed one nobody had asked for.
             * The corner the box is actually drawn from is still on screen;
             * it is just not pretending to be your input. */
            var ov = used.cell_origin || [0, 0, 0];
            ["pv-org-a", "pv-org-b", "pv-org-c"].forEach(function (id, i) {
                var f = $(id);
                if (!f) return;
                f.placeholder = String(round(ov[i] || 0));
                setIdle(f, rawOrigin === null ? "" : round(rawOrigin[i] || 0));
            });
            tag("pv-org-tag", rawOrigin === null);
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
    /* ── APPLY: the whole cell, once (§ 6.2) ─────────────────────────────
     *
     * "The cell is one fact that travels together -- the vectors, the anchor,
     * how each axis is treated, how much vacuum an isolated axis gets -- which
     * is why there is one door and nothing writes a part of it on its own."
     * The panel had FOUR commits onto that one fact, which is what made it
     * unreadable: which button you pressed decided which of your typed values
     * were sent and which were quietly dropped, and two of them reset the
     * others as a side effect the user learned about from a modal.
     *
     * THE REGIME DECIDES WHAT IS SENT, and it is the thing the user chose:
     *   derived  -- no cell, no origin; the vacuum is authoritative.
     *   explicit -- the nine numbers, and the corner if one was typed.
     * Vacuum travels either way because it is part of the block; under an
     * explicit cell the server keeps it as the reference-only value it is.
     *
     * THE TWO CONFIRM DIALOGS THAT STOOD HERE ARE GONE.  One asked before an
     * edit that would "reset the explicit cell", which was a side effect of
     * committing one field at a time and does not exist now -- choosing
     * "derived" IS asking for the derived box.  The other said the change
     * "cannot be undone", which stopped being true on 2026-09-07: a cell edit
     * records a timeline point like every other edit (§ 11.2), so Retract
     * brings the old box back.
     */
    function applyCell() {
        var explicit = regime() === "explicit";
        var kinds = ["pv-axis-a", "pv-axis-b", "pv-axis-c"].map(function (id) {
            return $(id) ? $(id).value : "isolated";
        });
        /* ALL THREE BLANK IS "no vacuum chosen", which the block sends as
         * null -- the state § 6.1 gives a default FOR.  A partly-typed one is
         * read as numbers with the untyped sides at 0, because a person who
         * typed one number did choose something. */
        var vacBoxes = ["pv-vac-a", "pv-vac-b", "pv-vac-c"].map(function (id) {
            var el = $(id);
            return el ? String(el.value).trim() : "";
        });
        var payload = {
            axis_kind: kinds,
            vacuum: vacBoxes.every(function (v) { return v === ""; })
                ? null
                : [num("pv-vac-a", 0), num("pv-vac-b", 0), num("pv-vac-c", 0)],
            cell: explicit ? stagedCell() : null,
            cell_origin: null,
        };
        if (explicit) {
            /* AN EMPTY BOX IS "DERIVE THE CORNER", not zero.  All three have
             * to be typed for the origin to be a statement -- a half-typed
             * corner is not a corner, and reading the blanks as 0 would move
             * the box to a place nobody chose. */
            var typed = ["pv-org-a", "pv-org-b", "pv-org-c"].map(function (id) {
                var el = $(id);
                return el && String(el.value).trim() !== "" ? Number(el.value) : null;
            });
            if (typed.every(function (v) { return v !== null && isFinite(v); })) {
                payload.cell_origin = typed;
            }
        }
        return commitOp("block", payload);
    }

    function wire() {
        /* ONE COMMIT AND ONE DISCARD, where four commits and two resets used
         * to be.  `applyCell` sends the whole block; Revert makes no request
         * at all -- it redraws every field from the structure, which throws
         * away what has been typed and not sent.  Undoing an APPLIED cell is
         * Retract's, and works here since a cell edit began recording a
         * timeline point (§ 11.2, 2026-09-07). */
        var apply = $("pv-apply");
        if (apply) apply.addEventListener("click", applyCell);
        var revert = $("pv-revert");
        if (revert) revert.addEventListener("click", function () {
            // Take the focus off the panel first: `refresh` deliberately
            // leaves the field being typed in alone, and Revert's whole job is
            // to overwrite exactly that one.
            if (document.activeElement && document.activeElement.blur) {
                document.activeElement.blur();
            }
            refresh();
        });
        // Blank the three origin boxes -- "derive the corner" said as a
        // gesture, since the way to say it is to type nothing and there is
        // otherwise no way to get BACK to nothing once something is typed.
        var orgDerive = $("pv-org-derive");
        if (orgDerive) orgDerive.addEventListener("click", function () {
            ["pv-org-a", "pv-org-b", "pv-org-c"].forEach(function (id) {
                var el = $(id);
                if (el) el.value = "";
            });
        });
        // Editing any of the nine restates the lengths and re-checks the sign,
        // so the note tracks what is staged rather than what was last committed.
        cellInputs.forEach(function (inp) {
            inp.addEventListener("input", function () {
                labelCellAxes(); syncLengthBox(); refreshHandedness();
                renderSpan();   /* the span is measured ALONG this row */
            });
        });
        var axisPick = $("pv-cell-axis");
        if (axisPick) axisPick.addEventListener("change", syncLengthBox);

        var fromSel = $("pv-cell-from-selection");
        if (fromSel) fromSel.addEventListener("click", function () {
            var pos = (function () {
                var all = data().measurement.positions();
                return (all && all.length >= 2) ? all.slice(0, 2) : null;
            })();
            if (!pos) return;
            setStagedRow(chosenAxis(), [pos[1][0] - pos[0][0],
                                        pos[1][1] - pos[0][1],
                                        pos[1][2] - pos[0][2]]);
            labelCellAxes(); syncLengthBox(); refreshHandedness();
            renderSpan();       /* the direction moved, so the span did too */
        });

        var gapBox = $("pv-cell-gap");
        if (gapBox) gapBox.addEventListener("input", renderSpan);
        var axisPickForSpan = $("pv-cell-axis");
        if (axisPickForSpan) axisPickForSpan.addEventListener("change", renderSpan);
        var spanPlus = $("pv-cell-span-plus-gap");
        if (spanPlus) spanPlus.addEventListener("click", function () {
            var span = spanAlong(chosenAxis());
            var gap = askedGap();
            if (span == null || !isFinite(gap) || gap < 0) return;
            var r = chosenAxis();
            var row = stagedCell()[r];
            var have = norm(row);
            var want = span + gap;
            if (have < LENGTH_QUIET) {
                /* No direction yet.  `c` gets z -- it is the axis this control
                   exists for and the one a slab is built along.  `a` and `b`
                   are refused, because writing (0, 0, want) into either puts
                   it on top of `c`: determinant zero, and the gate answers
                   NO VOLUME.  `spanAlong` already returns null for them, so
                   this is unreachable through the button; it stays because a
                   guard that depends on a caller checking first is not one. */
                if (r !== 2) return;
                setStagedRow(r, [0, 0, want]);
            } else {
                var k = want / have;
                setStagedRow(r, [row[0] * k, row[1] * k, row[2] * k]);
            }
            labelCellAxes(); syncLengthBox(); refreshHandedness(); renderSpan();
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
            var all = data().measurement.positions();
            if (!all || !all.length) return;
            var pos = [all[0]];          // the origin is the FIRST atom picked
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

        /* The nine inputs have NO commit of their own any more.  They are
         * read by `applyCell` through `stagedCell()` -- the same reader the
         * length box and the handedness note already use -- so what the note
         * describes and what Apply sends cannot differ. */
        // § 6.2 v3: no calibrate handler — coordinate rewrites are not a
        // periodicity edit (emission translates implicitly; the explicit
        // rewrite lives with the Modify ops as /api/modify/calibrate).
    }

    function start() {
        if (!$("optab-panel-cell")) return;
        buildCellGrid();
        buildRegimeRadios();
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
        if (cellTab) cellTab.addEventListener("click", function () {
            var w = data();
            if (w && w.measurement && w.measurement.requestPicking()) {
                sayRulerIsOn("Cell page");
            }
        });
        /* NOT ON STARTUP, and this is a correction rather than an omission.
         * It also ran here when the panel happened to be visible as the module
         * loaded -- which turned measuring on, and announced it, before the
         * person had touched anything.  A page that opens by telling you it
         * changed a mode you never asked for is exactly what announcing the
         * change was meant to prevent.  Caught by
         * `test_app_notifications_e2e`, whose whole subject is that the
         * notification bar stays empty until something happens.
         *
         * Opening the tab is the act; module load is not. */
    }

    start();
}
