/* Modify tab -- the Slab op-tab (archive/2026-09-01-modify-redesign-plan.md § 3).
 *
 * Contract: docs/web/molview.md § 11.1 (the op table -- `slab` sends no
 *           selection), docs/web/web-api.md (`/api/modify/slab`,
 *           `/api/modify/lattice-from-run`).
 * Owns:     the Slab panel's controls and the one body it posts.
 * Called by: modify/selection-bootstrap.js, which mounts the viewer and hands
 *           it here.  Nothing self-starts.
 *
 * IT IS THE ONLY SLAB BUILDER.  The Junction panel it was written beside is
 * gone (§ 3.4, done 2026-08-31) along with its half of viewer.js and the
 * symmetric-electrode route, and the per-side `/api/modify/electrode` route
 * went the day after (§ 3.4a) once nothing called it.  This file never shared
 * code with any of them -- it asks `/api/modify/meta` for its menu rather than
 * reading globals that panel stashed -- which is why the removals were
 * deletions and not untanglings.
 *
 * IT READS NO SELECTION.  `dx`, `dy` and the starting z are measured from the
 * 3-D window's own origin, so the same numbers place the same slab whatever
 * is picked.  That is why `OPERATIONS.slab` carries `group: null`.
 */
"use strict";

import { runOp } from "./viewer.js";

//: Grown downward, the registry either continues the crystal or mirrors it.
//  Growing up the two are identical, which is why the row hides (§ 3.2).
const STACKINGS = [
    ["continue", "continues the crystal"],
    ["mirror", "mirrors the slab"],
];
const GROWS = [["+z", "+z (up)"], ["-z", "-z (down)"]];
const REGISTRY_NAMES = ["A", "B", "C"];

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
            elSel.addEventListener("change", onLatticeInputsChanged);
        }
        radios("slab-plane-radios", "slab-plane",
               (meta.fcc_planes || []).map((p) => [p, p]), "111");
        radios("slab-grow-radios", "slab-grow", GROWS, "+z");
        radios("slab-stacking-radios", "slab-stacking", STACKINGS, "continue");
        for (const inp of document.querySelectorAll(
                'input[name="slab-plane"]')) {
            inp.addEventListener("change", onPlaneChanged);
        }
        for (const inp of document.querySelectorAll('input[name="slab-grow"]')) {
            inp.addEventListener("change", onGrowChanged);
        }
        onPlaneChanged();
        onGrowChanged();
        onLatticeInputsChanged();
    }

    /* ── The three notes, each tracking one control ──────────────────── */

    /* HOW MANY REGISTRIES THIS SURFACE HAS falls out of its stacking period
     * -- three on (111), two on the others -- so "A, B, or C *if available*"
     * needs no table of its own (§ 3.1).  An unknown plane offers one, which
     * says nothing rather than guessing. */
    function onPlaneChanged() {
        const plane = picked("slab-plane", "111");
        const period = (meta.stacking_period || {})[plane] || 1;
        const keep = Number(picked("slab-registry", "0"));
        radios("slab-registry-radios", "slab-registry",
               Array.from({ length: period },
                          (_, i) => [String(i), REGISTRY_NAMES[i] || String(i)]),
               String(keep < period ? keep : 0));
        renderOrthogonalChoice();
        renderPeriodNote();
    }

    /* THE CELL SHAPE IS NOT A FREE SWITCH (junction-cell.md § 2b): ASE builds
     * a non-orthogonal cell for fcc(111) only.  Where the surface allows one
     * shape, the box is set to it and disabled -- offering the other is
     * offering a slab that cannot be built, and the box starting unchecked is
     * exactly how a default (100) request came back a 400.
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

    /* Growing UP, "continues" and "mirrors" are the same slab, so the row is
     * hidden rather than offering a choice with no effect (§ 3.2). */
    function onGrowChanged() {
        const row = $("slab-stacking-row");
        if (row) row.hidden = picked("slab-grow", "+z") !== "-z";
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

    /* WHAT THE TYPED `a` MEANS, as the cross-check § 3.3 describes: the
     * derived spacings and how far the value sits from each literature
     * reference.  The one mistake anyone makes is picking a SECOND-shell
     * pair, which reads a factor 1.414 high and lands ~41% out -- where this
     * line says so at once. */
    function onLatticeInputsChanged() {
        const note = $("slab-a-derived");
        if (!note) return;
        const a = num("slab-a", NaN);
        const element = ($("slab-element") || {}).value;
        const row = (meta.lattice_table || {})[element] || {};
        if (!Number.isFinite(a) || a <= 0) { note.hidden = true; return; }
        const parts = [
            `d(111) ${(a / Math.sqrt(3)).toFixed(4)}`,
            `d(100) ${(a / 2).toFixed(4)}`,
            `nearest neighbour ${(a / Math.sqrt(2)).toFixed(4)} Å`,
        ];
        for (const [key, label] of [["a_experimental", "experimental"],
                                    ["a_pbe", "PBE"]]) {
            const ref = row[key];
            if (typeof ref === "number" && ref > 0) {
                const off = (a - ref) / ref * 100;
                parts.push(`${off >= 0 ? "+" : ""}${off.toFixed(1)}% from `
                           + `${label} (${ref.toFixed(4)})`);
            }
        }
        note.textContent = parts.join(" · ");
        note.hidden = false;
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
         * to have selected (user, 2026-08-31).
         *
         * This called `projects.shared().file` -- a method that has never
         * existed on that module.  Guarded by `&&`, so it was `undefined` on
         * every press and the button always took the refusal branch below:
         * the whole feature was unreachable, and no action a person could take
         * made it work.
         *
         * The picker is also the better question.  The sidebar's current
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
        onLatticeInputsChanged();
        const said = (j.notes || []).map((n) => n.message).join("  ·  ");
        say((j.notes || []).some((n) => n.level === "warn") ? "warn" : "info",
            `${j.element}${j.n_atoms} from ${j.source}: a = ${j.a.toFixed(4)} Å`
            + (said ? "  ·  " + said : ""));
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
            start_registry: Number(picked("slab-registry", "0")),
            start_z: num("slab-start-z", 0),
            grow: picked("slab-grow", "+z"),
            stacking: picked("slab-stacking", "continue"),
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
           the in-flight lock and the edit-status line.  This awaited
           `applyOp` directly and DISCARDED the answer -- and `applyOp`
           returns null both when another edit is in flight and when a
           precondition is refused, so a built slab and a refused one were
           indistinguishable, with no button disabled in between. */
        await runOp("/api/modify/slab", body, "Add slab");
    }

    const layersBox = $("slab-layers");
    if (layersBox) layersBox.addEventListener("input", renderPeriodNote);
    const aBox = $("slab-a");
    if (aBox) aBox.addEventListener("input", onLatticeInputsChanged);
    const pick = $("slab-pick-run");
    if (pick) pick.addEventListener("click", pickFromRun);
    const go = $("slab-apply");
    if (go) go.addEventListener("click", apply);

    loadMeta();
}
