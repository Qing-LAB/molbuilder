/* Modify tab -- the Slab op-tab (archive/2026-09-01-modify-redesign-plan.md § 3).
 *
 * Contract: docs/web/molview.md § 11.1 (the op table -- `slab` sends no
 *           selection), docs/web/web-api.md (`/api/modify/slab`,
 *           `/api/modify/lattice-from-run`).
 * Owns:     the Slab panel's controls and the one body it posts.
 * Called by: modify/selection-bootstrap.js, which mounts the viewer and hands
 *           it here.  Nothing self-starts.
 *
 * IT IS THE ONLY SLAB BUILDER.  It asks `/api/modify/meta` for its menu
 * rather than reading globals.
 *
 * IT READS NO SELECTION.  `dx`, `dy` and the starting z are measured from the
 * 3-D window's own origin, so the same numbers place the same slab whatever
 * is picked.  That is why `OPERATIONS.slab` carries `group: null`.
 */
"use strict";

import { runOp } from "./viewer.js";
import { clear as clearFindings, render as renderFindings }
    from "../lib/validation-findings.js";

const GROWS = [["+z", "+z (up)"], ["-z", "-z (down)"]];
const REGISTRY_NAMES = ["A", "B", "C"];

//: WHICH WAY THE REGISTRY CYCLE IS WALKED, read along the growth direction.
const SEQUENCES = ["ABC", "ACB"];

//: The walk written out from where it actually starts.  Starting on B the
//  forward walk is B -> C -> A -> B, and a label that said "A -> B -> C" there
//  would be describing a different slab from the one about to be built.
//  `period` names how many registries the surface has, so a 2-period surface
//  reads A -> B -> A rather than borrowing (111)'s third letter.
function walkLabel(sequence, start, period) {
    const step = (sequence === "ACB") ? -1 : 1;
    const seen = [];
    for (let i = 0; i <= period; i += 1) {
        seen.push(REGISTRY_NAMES[(((start + i * step) % period) + period) % period]);
    }
    return seen.join(" \u2192 ");
}

//: WHICH PUBLISHED LATTICE CONSTANT `a` STARTS FROM.  The keys are the
//  `lattice_table` columns /api/modify/meta answers with; "custom" is not a
//  column but the state the box enters once a person types in it or measures
//  a run into it.  Experimental is the default because it is what the server
//  already used for an empty box -- so filling the box changes what is SHOWN,
//  never what is built.
const LATTICE_REFS = [
    ["a_experimental", "Experimental"],
    ["a_pbe", "PBE"],
    ["custom", "Custom"],
];

export function init(viewer) {
    const $ = (id) => document.getElementById(id);
    if (!$("optab-panel-slab")) return;

    const data = () => (viewer && viewer.ok) ? viewer.data : null;
    let meta = { fcc_elements: [], fcc_planes: [], lattice_table: {},
                 stacking_period: {}, orthogonal_choices: {} };

    /* ── Small builders, so the five radio groups are one piece of code ── */

    function radios(hostId, name, entries, checked) {
        const box = $(hostId);
        if (!box) return;
        box.innerHTML = "";
        for (const [value, label] of entries) {
            const lbl = document.createElement("label");
            const inp = document.createElement("input");
            inp.type = "radio";
            inp.name = name;
            inp.value = value;
            inp.checked = (value === String(checked));
            lbl.appendChild(inp);
            lbl.appendChild(document.createTextNode(" " + label));
            box.appendChild(lbl);
        }
    }
    /* The same job as `radios` for a <select>: rebuild the options and keep
     * the current pick if it still exists.  The two menus that name a
     * REGISTRY -- the start and the walk -- are rewritten whenever the
     * surface changes, so keeping the pick is the whole point. */
    function options(id, entries, keep) {
        const sel = $(id);
        if (!sel) return;
        sel.innerHTML = "";
        for (const [value, label] of entries) {
            const o = document.createElement("option");
            o.value = value;
            o.textContent = label;
            if (value === String(keep)) o.selected = true;
            sel.appendChild(o);
        }
    }
    const picked = (name, fallback) => {
        const el = document.querySelector(`input[name="${name}"]:checked`);
        return el ? el.value : fallback;
    };
    const num = (id, dflt) => {
        const el = $(id);
        const v = el ? Number(el.value) : NaN;
        return Number.isFinite(v) ? v : dflt;
    };

    /* ── What the surface offers, from the server ────────────────────────
     *
     * The element list, the planes and the stacking periods are the Python
     * source's (`/api/modify/meta`), never a copy here -- adding a metal in
     * `molbuilder.modify` reaches this dropdown with no template change.
     */
    async function loadMeta() {
        try {
            const r = await fetch("/api/modify/meta");
            const j = await r.json();
            if (j && j.ok) meta = j;
        } catch (_) { /* a panel that cannot get its menu says so below */ }

        const elSel = $("slab-element");
        if (elSel) {
            elSel.innerHTML = "";
            for (const sym of meta.fcc_elements || []) {
                const o = document.createElement("option");
                o.value = sym; o.textContent = sym;
                if (sym === "Au") o.selected = true;
                elSel.appendChild(o);
            }
            // A DIFFERENT METAL HAS A DIFFERENT CONSTANT, so the box is
            // refilled from whichever reference is selected -- leaving Au's
            // 4.0782 sitting under "Ag / Experimental" would be the same
            // blank-box confusion in a worse form.
            elSel.addEventListener("change", fillFromLatticeRef);
        }
        radios("slab-plane-radios", "slab-plane",
               (meta.fcc_planes || []).map((p) => [p, p]), "111");
        radios("slab-grow-radios", "slab-grow", GROWS, "+z");
        radios("slab-lattice-ref-radios", "slab-lattice-ref",
               LATTICE_REFS, "a_experimental");
        for (const inp of document.querySelectorAll(
                'input[name="slab-plane"]')) {
            inp.addEventListener("change", onPlaneChanged);
        }
        for (const inp of document.querySelectorAll(
                'input[name="slab-lattice-ref"]')) {
            inp.addEventListener("change", onLatticeRefPicked);
        }
        onPlaneChanged();
        // AFTER the registry exists: the walk is written out from the start
        // registry, so it has nothing to say until that dropdown is built.
        renderSequence();
        fillFromLatticeRef();
    }

    /* ── The three notes, each tracking one control ──────────────────── */

    /* HOW MANY REGISTRIES THIS SURFACE HAS falls out of its stacking period
     * -- three on (111), two on the others -- so "A, B, or C *if available*"
     * needs no table of its own (§ 3.1).  An unknown plane offers one, which
     * says nothing rather than guessing. */
    function onPlaneChanged() {
        const period = surfacePeriod();
        const keep = Number(($("slab-registry") || {}).value || 0);
        options("slab-registry",
                Array.from({ length: period },
                           (_, i) => [String(i), REGISTRY_NAMES[i] || String(i)]),
                String(keep < period ? keep : 0));
        // The walk's LABELS name registries, so they are rewritten whenever
        // the surface changes how many there are.
        renderSequence();
        renderOrthogonalChoice();
        renderPeriodNote();
        // AND THE SPACING IS THE SURFACE'S: the note answers for THIS surface
        // and has to be re-asked when the surface changes.
        onLatticeInputsChanged();
    }

    const surfacePeriod = () =>
        (meta.stacking_period || {})[picked("slab-plane", "111")] || 1;

    /* THE WALK, WRITTEN OUT FROM THE START REGISTRY (user, 2026-09-07).
     *
     * Two options where the surface has three registries to walk through, and
     * NONE where it has two: modulo 2, forwards and backwards are the same
     * sequence, so both options would build the identical slab.  A control
     * that cannot change the answer is not a choice, so the row goes rather
     * than sitting there inert.
     *
     * Which surfaces those are is the server's `stacking_period` -- the same
     * fact the registry count and the period note already come from -- so the
     * crystallography stays in one place and this reads it. */
    function renderSequence() {
        const row = $("slab-sequence-row");
        const period = surfacePeriod();
        const start = Number(($("slab-registry") || {}).value || 0);
        if (row) row.hidden = period < 3;
        options("slab-sequence",
                SEQUENCES.map((v) => [v, walkLabel(v, start, period)]),
                ($("slab-sequence") || {}).value || SEQUENCES[0]);
    }

    /* THE CELL SHAPE IS NOT A FREE SWITCH (junction-cell.md § 2b): ASE builds
     * a non-orthogonal cell for fcc(111) only.  Where the surface allows one
     * shape, the box is set to it and disabled -- offering the other is
     * offering a slab that cannot be built.
     *
     * Which shapes exist is the server's fact (`orthogonal_choices`), never a
     * copy here.  A plane the server said nothing about leaves the box alone
     * rather than guessing at it. */
    function renderOrthogonalChoice() {
        const box = $("slab-orthogonal");
        const note = $("slab-orthogonal-note");
        if (!box) return;
        const plane = picked("slab-plane", "111");
        const choices = (meta.orthogonal_choices || {})[plane];
        if (!Array.isArray(choices) || choices.length !== 1) {
            box.disabled = false;
            if (note) note.hidden = true;
            return;
        }
        box.checked = !!choices[0];
        box.disabled = true;
        if (note) {
            note.textContent =
                `fcc(${plane}) is built with ${choices[0] ? "an orthogonal"
                : "a non-orthogonal"} cell only -- there is no choice to make `
                + `on this surface.`;
            note.classList.remove("modify-op-hint--warn");
            note.hidden = false;
        }
    }

    /* A seam only continues the crystal when the layer count is a whole
     * multiple of the stacking period (junction-cell.md § 3.1).  The period
     * is the server's; the arithmetic is one modulo and stays here. */
    function renderPeriodNote() {
        const note = $("slab-period-note");
        if (!note) return;
        const plane = picked("slab-plane", "111");
        const period = (meta.stacking_period || {})[plane];
        const layers = num("slab-layers", 0);
        if (!period || period < 2 || !layers) { note.hidden = true; return; }
        const rem = layers % period;
        note.textContent = rem === 0
            ? `${layers} layers is a whole number of ${period}-layer periods, `
              + `so a seam against another slab can continue the crystal.`
            : `${layers} layers is ${rem} past a whole ${period}-layer period `
              + `on fcc(${plane}) -- a seam here will not continue the `
              + `crystal. ${layers - rem} or ${layers + period - rem} would.`;
        note.classList.toggle("modify-op-hint--warn", rem !== 0);
        note.hidden = false;
    }

    /* THE CRYSTALLOGRAPHY IS THE SERVER'S, and this asks for it:
     * `/api/modify/spacings` derives the spacings from one rule and answers
     * for THE SELECTED plane.
     *
     * Stale answers cannot land: each request carries a ticket and only the
     * newest one is allowed to write, so typing quickly cannot leave an
     * earlier `a`'s numbers on screen. */
    let latticeTicket = 0;
    function onLatticeInputsChanged() {
        const note = $("slab-a-derived");
        if (!note) return;
        const a = num("slab-a", NaN);
        const element = ($("slab-element") || {}).value;
        const plane = picked("slab-plane", "111");
        if (!Number.isFinite(a) || a <= 0 || !element || !plane) {
            note.hidden = true;
            return;
        }
        const mine = ++latticeTicket;
        /* `reference` is the typed number, always -- the box holds whatever
         * the radio put there, so one value is sent and one is displayed.
         * The server requires it explicitly; there is no default to fall
         * back to. */
        const q = `element=${encodeURIComponent(element)}`
                + `&plane=${encodeURIComponent(plane)}`
                + `&reference=${encodeURIComponent(String(a))}`;
        fetch(`/api/modify/spacings?${q}`)
            .then((r) => r.json())
            .then((j) => {
                if (mine !== latticeTicket) return;      // superseded
                if (!j || j.ok !== true) { note.hidden = true; return; }
                note.textContent =
                    `d(${j.plane}) ${j.d_interlayer.toFixed(4)} Å`
                    + ` · nearest neighbour ${j.nearest_neighbour.toFixed(4)} Å`;
                note.hidden = false;
            })
            .catch(() => { if (mine === latticeTicket) note.hidden = true; });
    }

    /* ── The lattice reference (§ 3.3) ───────────────────────────────────
     *
     * THE BOX IS NEVER BLANK (user, 2026-09-07): the panel shows the number
     * it will build with.  Picking a reference fills the box; the box is still
     * typeable, and typing in it selects "Custom", so the two can never
     * disagree about what will be sent.
     */
    function fillFromLatticeRef() {
        const ref = picked("slab-lattice-ref", "a_experimental");
        if (ref === "custom") { onLatticeInputsChanged(); return; }
        const element = ($("slab-element") || {}).value;
        const value = ((meta.lattice_table || {})[element] || {})[ref];
        const box = $("slab-a");
        // A reference the table has no value for leaves the box alone rather
        // than clearing it: an element whose PBE column is missing should not
        // wipe a number the user can see is right.
        if (box && typeof value === "number" && value > 0) {
            box.value = value.toFixed(4);
            forgetMeasuredNotes();
        }
        onLatticeInputsChanged();
    }

    function onLatticeRefPicked() { fillFromLatticeRef(); }

    /* THE NOTES DESCRIBE THE VALUE IN THE BOX -- the file it was measured
     * from -- so a value that did not come off that file takes them away: a
     * published reference picked, or a number typed.  A failed measurement
     * leaves both the box and its notes as they were; a new one replaces
     * them (`renderFindings` clears before it draws). */
    function forgetMeasuredNotes() {
        clearFindings({ panel: $("slab-lattice-findings") });
    }

    /* TYPING MAKES IT CUSTOM.  Without this the radio would go on claiming
     * "Experimental" beside a number nobody published, which is exactly the
     * kind of quiet disagreement the row was added to end. */
    function markCustom() {
        const el = document.querySelector(
            'input[name="slab-lattice-ref"][value="custom"]');
        if (el) el.checked = true;
    }

    /* ── "From a bulk run…" -- § 3.3's door ──────────────────────────────
     *
     * The field stays typeable: this fills it, the derived line says what it
     * means, and the value can be overridden by hand.  The route measures the
     * ATOMS, not the cell, and returns notes rather than refusals -- the
     * setup is the user's to own -- so they are shown, not swallowed.
     */
    async function pickFromRun() {
        const notify = (window.molbuilder || {}).notify;
        const say = (level, message) => notify && notify.show
            && notify.show({ id: "slab-lattice-from-run", level, message });

        /* ASK FOR THE FILE, rather than reading whatever the sidebar happens
         * to have selected (user, 2026-08-31).  The sidebar's current
         * selection is implicit state -- it depends on what you last clicked,
         * possibly for an unrelated reason -- where a dialog asks the thing
         * the button is about.  `mode: "file"` is the tree picker's own
         * file-selection mode; the `pickable` filter narrows it to what a
         * lattice can actually be measured from. */
        const { pickPath } = await import("/static/lib/tree-picker.js");
        const path = await pickPath({
            title: "Measure the lattice from which run?",
            hint: "Pick a relaxed bulk result.  \u25b8 expands a folder.",
            mode: "file",
            confirmLabel: "Measure",
            pickable: (entry) => /\.(xyz|XV)$/i.test(entry.name || ""),
        });
        if (!path) return;                       // cancelled -- say nothing
        let j = null;
        try {
            const r = await fetch("/api/modify/lattice-from-run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    path: path,
                    element: ($("slab-element") || {}).value || undefined,
                }),
            });
            j = await r.json();
        } catch (e) {
            say("error", "could not reach the server: " + (e && e.message));
            return;
        }
        if (!j || j.ok !== true) {
            say("error", (j && j.error) || "the lattice could not be read");
            return;
        }
        const box = $("slab-a");
        if (box) box.value = j.a.toFixed(4);
        // A MEASURED VALUE IS A CUSTOM ONE -- it came off this user's own run,
        // not out of the published table, and the row has to say so.
        markCustom();
        onLatticeInputsChanged();
        // THE NOTES ARE FINDINGS, one row each at its own severity, drawn by
        // the one renderer under the box (`science/validation.md` § 4.1 R2a).
        renderFindings(j.notes || [], { panel: $("slab-lattice-findings") });
        say("info",
            `${j.element}${j.n_atoms} from ${j.source}: a = ${j.a.toFixed(4)} Å`);
    }

    /* ── Apply ───────────────────────────────────────────────────────────
     *
     * Through `applyOp`, like every other edit: the module builds the
     * structure body from its own data and applies the answer atomically
     * (molview.md § 11.1).  This passes only the op's own arguments.
     */
    async function apply() {
        if (!data()) return;   // no viewer, nothing to build on
        const body = {
            element: ($("slab-element") || {}).value || "Au",
            plane: picked("slab-plane", "111"),
            m: num("slab-m", 1),
            n: num("slab-n", 1),
            layers: num("slab-layers", 1),
            start_registry: Number(($("slab-registry") || {}).value || 0),
            sequence: ($("slab-sequence") || {}).value || "ABC",
            grow: picked("slab-grow", "+z"),
            start_z: num("slab-start-z", 0),
            orthogonal: !!($("slab-orthogonal") || {}).checked,
            dx: num("slab-dx", 0),
            dy: num("slab-dy", 0),
        };
        // A TYPED `a` WINS, and an empty box means "use the table's".  Sent
        // only when it is a real length, so the server keeps its own default
        // rather than being handed NaN.
        const a = num("slab-a", NaN);
        if (Number.isFinite(a) && a > 0) body.lattice_constant = a;
        /* THROUGH THE PAGE'S ONE OP WRAPPER (viewer.js `runOp`), which owns
           the in-flight lock and the edit-status line. */
        await runOp("/api/modify/slab", body, "Add slab");
    }

    const layersBox = $("slab-layers");
    if (layersBox) layersBox.addEventListener("input", renderPeriodNote);
    const aBox = $("slab-a");
    if (aBox) aBox.addEventListener("input", () => {
        markCustom();
        onLatticeInputsChanged();
        forgetMeasuredNotes();
    });
    const regBox = $("slab-registry");
    if (regBox) regBox.addEventListener("change", renderSequence);
    const pick = $("slab-pick-run");
    if (pick) pick.addEventListener("click", pickFromRun);
    const go = $("slab-apply");
    if (go) go.addEventListener("click", apply);

    loadMeta();
}
