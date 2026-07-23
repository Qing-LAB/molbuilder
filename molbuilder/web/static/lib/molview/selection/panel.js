/* Atom-selection panel -- DOM consumer of the selection store.
 *
 * Renders the _selection_panel.html partial against the current
 * store state and routes user input back through the store's
 * mutators.  No internal state: every render reads from the store,
 * every user action calls a mutator.
 *
 * Full spec: docs/protocols/molview-module.md
 *
 * Public surface:
 *
 *   mount(rootEl) -> { dispose }
 */
"use strict";

// 4a real-import graph: the pure leaves this panel depends on are IMPORTED (legible edges).
// The `molview.data` reads stay RUNTIME lookups on purpose -- they resolve the currently-active
// data instance (the active-instance seam; task #47 multi-embed keeps it dynamic).
import { atomIndexModel } from "../_atom-index.js";
import { measurements } from "./measurements.js";

const root = (typeof window !== "undefined") ? window : globalThis;

// Filter kinds exposed in the UI.
const FILTER_KINDS = [
    { kind: "by_element", label: "By element",
      placeholder: "comma-separated, e.g. Au,C" },
    { kind: "by_index",   label: "By atom index",
      placeholder: "e.g. 1-4, 6, 10-11" },
    { kind: "by_residue", label: "By residue",
      placeholder: "comma-separated, e.g. ALA,DA" },
    { kind: "by_label",   label: "By label",
      placeholder: "L-electrode" },
];

// Built-in label targets that always appear in the Assign
// dropdown (so the user can always grab the canonical
// transport-region names).  Plus any other labels currently
// on the structure get added dynamically by renderAssignTarget.
//
// Must stay in sync with the canonical region colors in
// ``lib/molview/engine/process.js`` (the REGION_COLORS
// map): any label the viewer paints a first-class color for
// should be assignable from this dropdown without typing
// ``__new__``.  ``interface`` was previously paintable but not
// assignable here; surfacing it removes that drift.
const BUILTIN_TARGETS = [
    { value: "L-electrode", label: "L-electrode" },
    { value: "R-electrode", label: "R-electrode" },
    { value: "bridge",      label: "bridge" },
    { value: "interface",   label: "interface" },
    { value: "frozen_atoms", label: "frozen atoms" },
];

// The "frozen atoms" set is stored as a Structure.frozen_atoms
// boolean column on the server, but rendered in the panel as a
// tag labelled FROZEN_TAG_LABEL.  These two strings have to stay
// paired: the tag's × button has to translate the display label
// back to the server-side target.
//
// The frozen tag's display label.  This equals the L1 channel model's
// FROZEN_CHANNEL by definition, but is kept as a local constant (not sourced
// from L1) so the panel stays loadable without forcing an L1 load-time
// dependency for one string; test_selection_panel_frozen_name_matches_l1
// asserts the two never drift.  FROZEN_TARGET is the SERVER-side
// synthetic-region name (store↔engine boundary), a separate concern.
const FROZEN_TAG_LABEL = "frozen";
const FROZEN_TARGET    = "frozen_atoms";

function mount(rootEl, opts) {
    rootEl = rootEl || document;
    // Phase 10 (workspace-contract.md §5): the panel drives the
    // unified ws.selection namespace, not the legacy
    // selection.store global.  Variable kept as ``store`` to
    // minimize churn — the surface is the same (toggle, set,
    // getState, subscribe, addFilter, ...) but every call now
    // routes through ws.selection.  Method names that drifted
    // from store→ws (toggleAtom→toggle, selectAll→all,
    // clearSelection→clear) are renamed at the call sites
    // below.
    // Phase 5 (fused module): a caller MAY pass its own store instance
    // (``opts.store``) so a readonly/ephemeral inspector mounts an
    // ISOLATED selection that never touches the workspace.  Defaults to
    // the workspace singleton, so Modify (which calls ``mount(host)``)
    // is unchanged.
    const store = (opts && opts.store) || (root.molbuilder
                   && root.molbuilder.molview && root.molbuilder.molview.data
                   && root.molbuilder.molview.data.selection);
    if (!store) {
        throw new Error(
            "selection-panel: molview.data.selection missing; "
          + "load lib/molview/data-model.js first"
        );
    }

    // Anchor index for shift-click range selection (2026-06-09).
    // Updated on every plain-click toggle; preserved across
    // renders so a shift-click after a re-render still sees the
    // prior anchor.
    let _anchorIndex = null;

    // Phase 5 (fused module): a ``readonly`` mount (the Results inspectors)
    // is view + filter/highlight ONLY -- no writes.  Hide the assign-label
    // section (the only mutation path in the panel); click-select + filter
    // + highlight all stay.
    if (opts && opts.mode === "readonly") {
        const assign = rootEl.querySelector(".selection-assign");
        if (assign) assign.hidden = true;
    }

    // (The isolate view TOGGLE was removed from the panel -- it lives in
    // the viewer's left toggle RAIL now (molview injects it via the embed's addViewToggle).)

    const $ = (id) => rootEl.querySelector("#" + id);
    const els = {
        modeClick:       $("selection-mode-click"),
        modeFilter:      $("selection-mode-filter"),
        clickSection:    $("selection-click-section"),
        filterSection:   $("selection-filter-section"),
        filterRows:      $("selection-filter-rows"),
        combinator:      $("selection-combinator"),
        addFilterBtn:    $("selection-add-filter"),
        applyFilterBtn:  $("selection-apply-filter"),
        clearBtn:        $("selection-clear-btn"),
        invertBtn:       $("selection-invert-btn"),
        count:           $("selection-count"),
        loading:         $("selection-loading"),
        atomList:        $("selection-atom-list"),
        listEmpty:       $("selection-list-empty"),
        selectAll:       $("selection-select-all"),
        assignTgt:       $("selection-assign-target"),
        assignNew:       $("selection-assign-new-label"),
        assignBtn:       $("selection-assign-btn"),
        addBtn:          $("selection-add-btn"),
        removeBtn:       $("selection-remove-btn"),
        errorEl:         $("selection-error"),
        // Cell page (structure-periodicity.md § 3b) -- read-only display (stage 2).
        pageRadioSel:    $("panel-page-radio-selection"),
        pageRadioCell:   $("panel-page-radio-cell"),
        pageSel:         $("panel-page-selection"),
        pageCell:        $("panel-page-cell"),
        cellAxis:        $("cell-axis-value"),
        cellVacuum:      $("cell-vacuum-value"),
        cellOrigin:      $("cell-origin-value"),
        cellMatrix:      $("cell-matrix-value"),
        cellTag:         $("cell-matrix-tag"),
        // Measurement readout lives OUTSIDE the panel partial —
        // it's a chip overlay on the 3D viewer canvas.  OPTIONAL:
        // when molview.mount builds the card it mounts its OWN
        // measurement overlay (mountMeasurementOverlay, class
        // ``.molview-measurement-overlay``), so no page-level
        // ``#selection-measurement-overlay`` id exists and the
        // panel's own renderMeasurement stands down (it null-guards).
        // A legacy page that still provides the id keeps working.
        measurement:     document.getElementById(
            "selection-measurement-overlay"),
    };
    // ``measurement`` is optional (the module owns it on a built card) — do NOT
    // error when it is absent, only for the genuinely-required partial ids.
    const missing = Object.keys(els).filter(
        (k) => k !== "measurement" && !els[k]);
    if (missing.length) {
        if (root.console) root.console.error(
            "[selection-panel] missing partial ids: "
            + missing.join(", ")
        );
    }

    // ----- listener bookkeeping -------------------------------- //
    //
    // ``on`` (static): one-time wiring at mount.  Removed on
    // dispose so a remount on the same host doesn't double up.
    // ``onTransient`` (per-render): listeners on DOM nodes that
    // get destroyed + re-created on every render (filter rows,
    // atom rows, label tags).  We deliberately do NOT track
    // these for cleanup: when the parent ``innerHTML = ""`` runs
    // on the next render, the detached nodes + their listeners
    // become unreachable and GC together.  Tracking them in
    // ``cleanups`` would defeat the GC by keeping references
    // alive, which was a slow memory growth in long sessions.
    const cleanups = [];
    function on(target, ev, fn, opts) {
        if (!target) return;
        target.addEventListener(ev, fn, opts);
        cleanups.push(() => target.removeEventListener(ev, fn, opts));
    }
    function onTransient(target, ev, fn, opts) {
        if (!target) return;
        target.addEventListener(ev, fn, opts);
    }

    // ----- helpers for label / assign wiring ------------------- //

    const resolveTarget = () => {
        if (!els.assignTgt) return null;
        const t = els.assignTgt.value === "__new__"
            ? (els.assignNew ? els.assignNew.value : "")
            : els.assignTgt.value;
        return (t || "").trim() || null;
    };

    const currentMembers = (target, atoms) => {
        const out = [];
        atoms.forEach((a) => {
            const isMember = target === FROZEN_TARGET
                ? a.isFrozen
                : (a.labels || []).indexOf(target) !== -1;
            if (isMember) out.push(a.index);
        });
        return out;
    };

    function onAssign() {
        const t = resolveTarget();
        if (!t) return;
        store.writeLabel(t, store.getState().indices);
    }
    function onAddToTarget() {
        const t = resolveTarget();
        if (!t) return;
        const s = store.getState();
        // Union of existing members + selection.  Sort handled
        // inside writeLabel via the server's canonical form.
        const merged = new Set(currentMembers(t, s.atoms));
        s.indices.forEach((i) => merged.add(i));
        store.writeLabel(t, Array.from(merged));
    }
    function onRemoveFromTarget() {
        const t = resolveTarget();
        if (!t) return;
        const s = store.getState();
        const sel = new Set(s.indices);
        store.writeLabel(t,
            currentMembers(t, s.atoms).filter((i) => !sel.has(i)));
    }
    function onRemoveSingleLabel(label, atomIndex) {
        const target = (label === FROZEN_TAG_LABEL) ? FROZEN_TARGET : label;
        const s = store.getState();
        const newMembers = currentMembers(target, s.atoms)
            .filter((i) => i !== atomIndex);
        store.writeLabel(target, newMembers);
    }

    // ----- rendering ------------------------------------------- //

    function renderStatus(s) {
        if (!els.count) return;
        // Gate on the ATOMS (the structure content the list also shows), NOT sourceFile:
        // a structure loaded from text has atoms but no file path, and the count must stay
        // in sync with the list rather than reporting "no structure" for a loaded molecule.
        var n = (s.atoms && s.atoms.length) || 0;
        if (n === 0) {
            els.count.textContent = "no structure";
        } else {
            els.count.textContent = s.indices.length + " / " + n + " atoms";
        }
        if (els.loading) els.loading.hidden = !s.loading;
        if (els.errorEl) {
            if (s.error) {
                els.errorEl.textContent = s.error;
                els.errorEl.hidden = false;
            } else {
                els.errorEl.textContent = "";
                els.errorEl.hidden = true;
            }
        }
    }

    // Render the selection-driven measurement (xyz / distance /
    // angle).  Positions come from the page's positions
    // provider (Modify viewer registers one against the parsed
    // XYZ; Results trajectory/structure inspectors register
    // their own).  Hidden when:
    //   * no positions provider registered (e.g. page mounted
    //     panel without atoms loaded);
    //   * selection has 0 or 4+ atoms;
    //   * any selected index sits outside the positions array
    //     (caller hasn't synced positions to a fresh atom count
    //     yet — keep the readout silent rather than guessing).
    // The shared math lives in lib/selection/measurements.js
    // and is unit-tested in tests/test_selection_measurements_js.py.
    function renderMeasurement(s) {
        if (!els.measurement) return;
        const meas = measurements;   // imported pure leaf (4a)
        // Coordinates come from molview.data (a sibling within MolView) --
        // the frame-aware current coords -- NOT from a global a consumer
        // decorated onto the namespace.  getCoordinates() reflects the shown
        // frame, so the readout matches the viewer.
        const dataApi = mv && mv.data;
        if (!meas || !dataApi
            || typeof dataApi.getCoordinates !== "function") {
            els.measurement.hidden = true;
            els.measurement.textContent = "";
            return;
        }
        const positions = dataApi.getCoordinates() || [];
        const result = meas.compute(
            s.indices, s.atoms, positions, s.pickOrder);
        if (!result) {
            els.measurement.hidden = true;
            els.measurement.textContent = "";
            return;
        }
        els.measurement.hidden = false;
        els.measurement.dataset.kind = result.kind;
        els.measurement.textContent = result.display;
    }

    function renderMode(s) {
        if (els.modeClick)     els.modeClick.checked     = (s.mode === "click");
        if (els.modeFilter)    els.modeFilter.checked    = (s.mode === "filter");
        // Mutually exclusive sections.  Click mode shows the
        // atom list; Filter mode shows the filter editor.  The
        // selection persists in both (state.selection is
        // shared); flipping modes is purely a UI swap.
        if (els.clickSection)  els.clickSection.hidden  = (s.mode !== "click");
        if (els.filterSection) els.filterSection.hidden = (s.mode !== "filter");
        if (els.combinator)    els.combinator.value     = s.combinator;
    }

    // Labels currently on at least one atom -- drives the
    // "By label" filter dropdown.  Includes ``FROZEN_TAG_LABEL``
    // when any atom carries ``isFrozen``, matching the visible
    // tag rendered in the atom rows.  The dropdown's option
    // value still maps to the canonical server-side target
    // (``frozen_atoms``) — see ``renderFilters`` for the
    // value/display split.
    // User-facing atom index is 1-based (data-vocabulary.md § 3.1); the
    // store's atom.index is 0-based.  Convert only for display, via the
    // shared L1 helper (loaded first by modify.html).
    function _toDisplayIndex(i) {
        return atomIndexModel.toDisplay(i);
    }

    function knownLabels(s) {
        // The "By label" dropdown offers every TAG + FLAG channel present
        // (regions -> tag, frozen -> flag, and any future tag/flag
        // channels), from the pure L1 channel model (atom-annotations.md
        // § 5), computed from the render SNAPSHOT so it stays consistent
        // with what's drawn.
        return root.molbuilder.atomChannelModel
            .channelKinds(s.atoms || [])
            .filter((c) => c.kind === "tag" || c.kind === "flag")
            .map((c) => c.name)
            .sort();
    }

    function renderFilters(s) {
        if (!els.filterRows) return;
        els.filterRows.innerHTML = "";
        if (s.filters.length === 0) {
            const p = document.createElement("p");
            p.className = "selection-filter-empty";
            p.textContent =
                "No filters yet.  Click + Add filter to start.";
            els.filterRows.appendChild(p);
            return;
        }
        const labels = knownLabels(s);
        s.filters.forEach((f, i) => {
            els.filterRows.appendChild(
                makeFilterRow(f, i, s.filters, labels));
        });
    }

    function makeFilterRow(filter, index, allFilters, labels) {
        const row = document.createElement("div");
        row.className = "selection-filter-row";
        row.dataset.index = String(index);

        const kindSelect = document.createElement("select");
        kindSelect.className = "selection-filter-kind";
        FILTER_KINDS.forEach((spec) => {
            const o = document.createElement("option");
            o.value = spec.kind;
            o.textContent = spec.label;
            if (filter.kind === spec.kind) o.selected = true;
            kindSelect.appendChild(o);
        });
        row.appendChild(kindSelect);

        let valueEl;
        if (filter.kind === "by_label") {
            valueEl = document.createElement("select");
            valueEl.className = "selection-filter-text";
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = labels.length
                ? "(pick a label)" : "(no labels yet)";
            valueEl.appendChild(placeholder);
            labels.forEach((label) => {
                const o = document.createElement("option");
                // 2026-06-12: ``frozen`` is a display alias for
                // the server-side ``frozen_atoms`` target
                // (the atom-row tag uses the short form; the
                // canonical sidecar key + the server's
                // ``ByRegion`` resolution use the long form).
                // Map the dropdown so the user picks the
                // familiar tag but the rule lands on the
                // synthetic ``frozen_atoms`` region wired in
                // ``selection.py::_load_structure``.
                if (label === FROZEN_TAG_LABEL) {
                    o.value = FROZEN_TARGET;
                    o.textContent = label;
                } else {
                    o.value = label;
                    o.textContent = label;
                }
                if (filter.value === o.value) o.selected = true;
                valueEl.appendChild(o);
            });
        } else {
            valueEl = document.createElement("input");
            valueEl.type = "text";
            valueEl.className = "selection-filter-text";
            const spec = FILTER_KINDS.find((s) => s.kind === filter.kind)
                       || FILTER_KINDS[0];
            valueEl.placeholder = spec.placeholder || "";
            valueEl.value = filter.value || "";
        }
        row.appendChild(valueEl);

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "selection-filter-remove";
        removeBtn.textContent = "×";
        removeBtn.title = "Remove this filter";
        row.appendChild(removeBtn);

        // Commit handlers route through the store's editing
        // helpers (addFilter / updateFilter / removeFilter) so
        // the panel doesn't manage filter-array boilerplate.

        onTransient(kindSelect, "change", () => {
            store.updateFilter(index, {
                kind:  kindSelect.value,
                value: "",
            });
        });
        // ``change`` covers both text inputs (fires on blur /
        // Enter) and <select> (fires on user pick).  We do NOT
        // register ``input`` here -- on text inputs it would
        // fire every keystroke, triggering setFilters ->
        // renderFilters wipe -> the user's input element is
        // destroyed mid-type and focus is lost.  Apply filter
        // is the explicit commit step, not a re-eval on every
        // change anyway.
        onTransient(valueEl, "change", () => {
            store.updateFilter(index, {
                kind:  kindSelect.value,
                value: valueEl.value,
            });
        });
        onTransient(removeBtn, "click", () => store.removeFilter(index));

        return row;
    }

    // Atom-list render: two paths based on structure size.
    //
    // (A) Small structures (n_atoms < VSCROLL_THRESHOLD): render
    //     all rows in the DOM; on subsequent notifies do a
    //     SELECTION-ONLY diff (toggle .is-selected + checkbox)
    //     unless the atoms array itself changed.  This is the
    //     fast path for the common case (a few hundred atoms).
    //
    // (B) Large structures (n_atoms >= VSCROLL_THRESHOLD): use
    //     a virtual scroller -- only render rows visible in the
    //     scroll viewport + a small buffer.  Spacer <tr>s above
    //     and below keep the scrollbar accurate.  This is what
    //     makes a 50k-atom protein render at all without jamming
    //     the browser.
    //
    // The cutoff is tunable; 2000 is a sweet spot between "small
    // enough that the full DOM is fine" and "large enough that
    // the user starts to notice jank on each toggle".
    const VSCROLL_THRESHOLD = 2000;
    const VSCROLL_ROW_HEIGHT = 22;       // px; matches CSS row height
    const VSCROLL_BUFFER = 10;           // rows above/below viewport

    // Force-mode switch.  Lets demos + automated tests exercise
    // either render path against the SAME structure -- useful to
    // verify virtual scroll works on a 1184-atom PDB without
    // needing a synthetic 5000+ atom fixture.  Resolution order
    // (first match wins):
    //
    //   1. ``window.molbuilder.molview.selection.panel.forceRenderMode``
    //      -- programmatic; what tests typically set.
    //   2. ``sessionStorage["molbuilder.panelMode"]``
    //      -- survives reloads in the same tab; convenient
    //      from devtools.
    //   3. URL query ``?panel-mode=virtual`` or ``?panel-mode=simple``
    //      -- one-shot, shareable link.
    //
    // Valid values: ``"virtual"``, ``"simple"``, or null/missing
    // (= auto-threshold based on atom count, the production
    // default).  Invalid values are ignored so a typo can't
    // break the renderer.
    const _PANEL_MODE_KEY = "molbuilder.panelMode";
    function _resolveForceMode() {
        try {
            const w = root.molbuilder && root.molbuilder.molview && root.molbuilder.molview.selection && root.molbuilder.molview.selection.panel
                      && root.molbuilder.molview.selection.panel.forceRenderMode;
            if (w === "virtual" || w === "simple") return w;
        } catch (_) { /* not yet set */ }
        try {
            const ss = sessionStorage.getItem(_PANEL_MODE_KEY);
            if (ss === "virtual" || ss === "simple") return ss;
        } catch (_) { /* sessionStorage denied (private mode) */ }
        try {
            const u = new URL(root.location.href);
            const q = u.searchParams.get("panel-mode");
            if (q === "virtual" || q === "simple") return q;
        } catch (_) { /* URL not parseable */ }
        return null;
    }

    // Cached snapshots of the LAST RENDERED state so the diff
    // path can detect "selection changed but atoms didn't" and
    // avoid rebuilding 5000 <tr>s on every click.  Stored as
    // _lastAtomsRef (object identity check) + _lastSelectionSet
    // for membership comparisons.
    let _lastAtomsRef = null;
    let _lastRenderMode = null;          // "empty" | "simple" | "virtual"

    function renderAtomList(s, changes) {
        if (!els.atomList) return;
        // Empty state.
        if (s.atoms.length === 0) {
            if (_lastRenderMode !== "empty") {
                els.atomList.innerHTML = "";
                _lastRenderMode = "empty";
                _lastAtomsRef = null;
            }
            if (els.listEmpty) els.listEmpty.hidden = false;
            if (els.selectAll) {
                els.selectAll.checked       = false;
                els.selectAll.indeterminate = false;
                els.selectAll.disabled      = true;
            }
            return;
        }
        if (els.listEmpty) els.listEmpty.hidden = true;
        const selected = new Set(s.indices);
        if (els.selectAll) {
            const all  = (selected.size === s.atoms.length);
            const none = (selected.size === 0);
            els.selectAll.disabled      = false;
            els.selectAll.checked       = all;
            els.selectAll.indeterminate = !all && !none;
        }

        // Pick which render path to use.  Force-mode overrides
        // the auto-threshold so demos / tests can exercise the
        // virtual scroller against any structure size.
        const _forced = _resolveForceMode();
        const useVirtual = (_forced === "virtual")
            ? true
            : (_forced === "simple"
                ? false
                : s.atoms.length >= VSCROLL_THRESHOLD);
        // Rebuild rows only when their CONTENT could change (atoms adopted or labels edited)
        // or on the initial render (changes === undefined). A pure selection change (the store
        // marks only CHANGE.SELECTION) falls through to the cheap _diffSelection. The store's
        // dirty-bit set is what makes "did the rows change?" truthful -- the old object-identity
        // check on s.atoms was defeated by the snapshot cloning a fresh array every click (the
        // ~1.2s freeze on a 444-atom structure). Empty/unknown changes -> rebuild (safe).
        const atomsChanged = (changes === undefined)
            || changes.length === 0
            || changes.indexOf("atoms") >= 0
            || changes.indexOf("labels") >= 0;

        if (useVirtual) {
            _renderVirtual(s, selected, atomsChanged);
            _lastRenderMode = "virtual";
        } else if (atomsChanged || _lastRenderMode !== "simple") {
            // Full rebuild: atoms array swapped (new structure or
            // sidecar update) OR we just came back from a virtual /
            // empty render and need fresh DOM.
            _detachVirtualScroller();
            els.atomList.innerHTML = "";
            s.atoms.forEach((a) => {
                els.atomList.appendChild(makeAtomRow(a, selected));
            });
            _lastRenderMode = "simple";
        } else {
            // Selection-only diff: same atoms array; just update
            // the per-row .is-selected class + checkbox state on
            // every existing <tr>.  O(N) but ~50x cheaper than
            // O(N) DOM construction for a ~1000-atom structure.
            _diffSelection(selected);
        }
        _lastAtomsRef = s.atoms;
    }

    function _diffSelection(selected) {
        const rows = els.atomList.querySelectorAll("tr[data-atom-index]");
        for (let i = 0; i < rows.length; i++) {
            const tr = rows[i];
            const idx = parseInt(tr.dataset.atomIndex, 10);
            const want = selected.has(idx);
            const had  = tr.classList.contains("is-selected");
            if (want !== had) {
                tr.classList.toggle("is-selected", want);
            }
            const cb = tr.querySelector('input[type="checkbox"]');
            if (cb && cb.checked !== want) cb.checked = want;
        }
    }

    // ----- Virtual scroller state ----------------------------- //
    // Holds the scroller + spacer refs so subsequent renders
    // (selection diff, scroll events) can update without
    // rebuilding the scroller wrapper.
    let _vscroll = null;     // { scroller, topSpacer, bottomSpacer, scrollHandler }

    function _detachVirtualScroller() {
        if (!_vscroll) return;
        if (_vscroll.scrollHandler) {
            _vscroll.scroller.removeEventListener(
                "scroll", _vscroll.scrollHandler);
        }
        _vscroll = null;
    }

    function _renderVirtual(s, selected, atomsChanged) {
        // Lazy-init the scroller wrapper.  We make the table's
        // PARENT scrollable (the listWrap if present, else the
        // tbody directly).  Heights of spacer <tr>s drive the
        // scrollbar so it reflects the full row count, while
        // only the in-viewport rows + buffer are real <tr>s.
        //
        // Stale-closure guard: the scroll-event handler must
        // see the CURRENT s + selected, not whatever was
        // captured when the scroller was created.  We stash
        // the latest pair in ``_vscroll.latest`` on every
        // render and the handler reads from that container,
        // so a scroll AFTER a selection-toggle paints with
        // the new selection, not the old.
        if (!_vscroll || atomsChanged) {
            _detachVirtualScroller();
            els.atomList.innerHTML = "";

            const scroller = els.atomList.parentElement
                            && els.atomList.parentElement.parentElement
                            || els.atomList;
            // Spacer rows: empty <tr>s with td colSpan=6 + a
            // height style.  Browsers respect td height; this
            // lets the scrollbar position match the full virtual
            // list.
            const topSpacer    = document.createElement("tr");
            const bottomSpacer = document.createElement("tr");
            topSpacer.dataset.role    = "vscroll-top";
            bottomSpacer.dataset.role = "vscroll-bottom";
            // Build the spacer <td> via createElement instead of
            // innerHTML so the XSS audit passes (no innerHTML in
            // selection-panel.js).  Spacer height is set later
            // via inline style on this td.
            const _mkSpacerCell = () => {
                const td = document.createElement("td");
                td.colSpan = 6;
                td.style.padding = "0";
                td.style.border  = "0";
                return td;
            };
            topSpacer.appendChild(_mkSpacerCell());
            bottomSpacer.appendChild(_mkSpacerCell());
            els.atomList.appendChild(topSpacer);
            els.atomList.appendChild(bottomSpacer);

            const latest = { s: s, selected: selected };
            const scrollHandler = () => _paintVirtualWindow(
                latest.s, latest.selected
            );
            scroller.addEventListener("scroll", scrollHandler);
            _vscroll = {
                scroller, topSpacer, bottomSpacer,
                scrollHandler, latest,
            };
        } else {
            // Same atoms; just update the latest-state pointer
            // so the scroll handler picks up the new selection.
            _vscroll.latest.s = s;
            _vscroll.latest.selected = selected;
        }
        _paintVirtualWindow(s, selected);
    }

    function _paintVirtualWindow(s, selected) {
        if (!_vscroll) return;
        const scroller = _vscroll.scroller;
        const total = s.atoms.length;
        const scrollTop = scroller.scrollTop || 0;
        const viewportH = scroller.clientHeight || 300;
        const firstIdx = Math.max(
            0,
            Math.floor(scrollTop / VSCROLL_ROW_HEIGHT) - VSCROLL_BUFFER
        );
        const lastIdx = Math.min(
            total - 1,
            Math.ceil((scrollTop + viewportH) / VSCROLL_ROW_HEIGHT)
                + VSCROLL_BUFFER
        );
        // Spacer heights: rows ABOVE first + rows BELOW last.
        const topH = firstIdx * VSCROLL_ROW_HEIGHT;
        const bottomH = (total - lastIdx - 1) * VSCROLL_ROW_HEIGHT;
        _vscroll.topSpacer.firstElementChild.style.height = topH + "px";
        _vscroll.bottomSpacer.firstElementChild.style.height = bottomH + "px";

        // Remove existing rendered rows (between the spacers).
        let cursor = _vscroll.topSpacer.nextSibling;
        while (cursor && cursor !== _vscroll.bottomSpacer) {
            const next = cursor.nextSibling;
            cursor.remove();
            cursor = next;
        }
        // Insert the visible-window rows just before the bottom
        // spacer.  Order matches s.atoms (already sorted by
        // index from the server).
        for (let i = firstIdx; i <= lastIdx; i++) {
            const a = s.atoms[i];
            if (!a) continue;
            els.atomList.insertBefore(
                makeAtomRow(a, selected),
                _vscroll.bottomSpacer,
            );
        }
    }

    function makeAtomRow(atom, selected) {
        const tr = document.createElement("tr");
        tr.dataset.atomIndex = String(atom.index);
        if (selected.has(atom.index)) tr.classList.add("is-selected");

        const checkTd = document.createElement("td");
        checkTd.className = "col-check";
        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = selected.has(atom.index);
        // The atom list lives in click mode -- checkbox is
        // always interactive here (filter mode hides this
        // whole section).
        checkTd.appendChild(check);
        tr.appendChild(checkTd);

        tr.appendChild(td("col-idx",  String(_toDisplayIndex(atom.index))));
        tr.appendChild(td("col-el",   atom.element || ""));
        tr.appendChild(td("col-name", atom.atomName || ""));
        tr.appendChild(td("col-res",  atom.residueName || ""));

        const labelsTd = document.createElement("td");
        labelsTd.className = "col-labels";
        (atom.labels || []).forEach((label) => {
            labelsTd.appendChild(
                makeLabelTag(label, "tag-region", atom.index));
        });
        if (atom.isFrozen) {
            labelsTd.appendChild(
                makeLabelTag(FROZEN_TAG_LABEL, "tag-frozen", atom.index));
        }
        tr.appendChild(labelsTd);

        onTransient(check, "change", (e) => {
            e.stopPropagation();
            store.toggle(atom.index);
            _anchorIndex = atom.index;
        });
        onTransient(tr, "click", (e) => {
            if (e.target === check) return;
            // Shift-click range selection (2026-06-09): select
            // every atom between the previous click (anchor) and
            // the just-clicked atom, inclusive — added to the
            // current selection rather than replacing.  Standard
            // selection-list semantics; lets the user group +
            // assign region labels (electrode slabs, bridge
            // residues) without clicking each atom individually.
            //
            // Shift held WITHOUT an anchor falls back to plain
            // toggle so the first click in a fresh session is
            // never "do nothing".
            if (e.shiftKey && _anchorIndex !== null
                    && _anchorIndex !== atom.index) {
                const lo = Math.min(_anchorIndex, atom.index);
                const hi = Math.max(_anchorIndex, atom.index);
                const range = [];
                for (let i = lo; i <= hi; i++) range.push(i);
                store.add(range);
                // Anchor stays put — successive shift-clicks
                // extend the range from the same origin.
                return;
            }
            store.toggle(atom.index);
            _anchorIndex = atom.index;
        });
        return tr;
    }

    function td(cls, text) {
        const el = document.createElement("td");
        el.className = cls;
        el.textContent = text;
        return el;
    }

    function makeLabelTag(label, cls, atomIndex) {
        const wrap = document.createElement("span");
        wrap.className = "selection-tag " + cls;
        const textEl = document.createElement("span");
        textEl.textContent = label;
        wrap.appendChild(textEl);
        const x = document.createElement("button");
        x.type = "button";
        x.className = "selection-tag-remove";
        // 1-based for display, matching the row's index column
        // (data-vocabulary.md § 3.1); atomIndex itself stays 0-based.
        x.title = "Remove this label from atom #" + _toDisplayIndex(atomIndex);
        x.textContent = "×";
        onTransient(x, "click", (e) => {
            e.stopPropagation();
            onRemoveSingleLabel(label, atomIndex);
        });
        wrap.appendChild(x);
        return wrap;
    }

    function renderAssignVisibility() {
        if (!els.assignTgt || !els.assignNew) return;
        els.assignNew.hidden = (els.assignTgt.value !== "__new__");
    }

    function renderAssignTarget(state) {
        if (!els.assignTgt) return;
        const seen = new Set();
        const opts = [];
        BUILTIN_TARGETS.forEach((t) => {
            seen.add(t.value);
            opts.push(t);
        });
        const userLabels = new Set();
        (state.atoms || []).forEach((a) => {
            (a.labels || []).forEach((label) => userLabels.add(label));
        });
        Array.from(userLabels).sort().forEach((label) => {
            if (seen.has(label)) return;
            seen.add(label);
            opts.push({ value: label, label: label });
        });
        opts.push({ value: "__new__", label: "+ new region label…" });

        const prev = els.assignTgt.value;
        els.assignTgt.innerHTML = "";
        opts.forEach((o) => {
            const el = document.createElement("option");
            el.value = o.value;
            el.textContent = o.label;
            els.assignTgt.appendChild(el);
        });
        if (seen.has(prev) || prev === "__new__") {
            els.assignTgt.value = prev;
        } else if (prev) {
            // The previously-selected user label is no longer
            // present on any atom (e.g. the sidecar was edited /
            // re-saved elsewhere).  Defaulting the <select> to
            // the first option (BUILTIN_TARGETS[0] = L-electrode)
            // would silently retarget the user's next Assign
            // click; that's a data-loss footgun.  Snap to the
            // "+ new region label..." mode and pre-fill the
            // free-text input with the lost name so the user
            // either re-creates the label deliberately or picks
            // a different target.
            els.assignTgt.value = "__new__";
            if (els.assignNew) {
                els.assignNew.value = prev;
            }
        }
        renderAssignVisibility();
    }

    // ----- static control wiring ------------------------------- //

    on(els.modeClick,       "change", () => store.setMode("click"));
    on(els.modeFilter,      "change", () => store.setMode("filter"));
    on(els.combinator,      "change", () => store.setCombinator(els.combinator.value));
    // 2026-06-12 layout v3 (revised): ``+ Add filter`` is the
    // headline action of the Filter section — accent-coloured,
    // full width, visible ONLY when the user is in filter mode.
    // Click-mode entry is via the mode pill at the top of the
    // panel; Add filter never appears in click mode.  The
    // auto-flip-on-click below stays as a defense-in-depth no-op
    // (the button is hidden in click mode so users can't trigger
    // it, but programmatic callers like a future "make filter
    // from this selection" flow still benefit).
    on(els.addFilterBtn,    "click",  () => {
        store.addFilter({ kind: "by_element", value: "" });
        if (store.getState().mode !== "filter") store.setMode("filter");
    });
    on(els.applyFilterBtn,  "click",  () => store.applyFilter());
    on(els.clearBtn,        "click",  () => store.clear());
    // Invert flips selected ↔ unselected.  Both surfaces (ws.selection +
    // the ephemeral surface) expose ``invert`` -> the ``invertSelection``
    // mutator; no fallback needed.
    on(els.invertBtn, "click", () => store.invert());
    // "Show selected only" (isolate) is a VIEW toggle that
    // now live on the viewer's left toggle RAIL (molview injects them via addViewToggle), by the viewer
    // they affect -- not in this panel.  The isolate auto-clear-on-empty is handled
    // there.  Both still drive the SAME store; the panel just no longer hosts them.

    const _cellStoreUnsub = store.subscribe(function () { renderCell(); });
    cleanups.push(() => { try { _cellStoreUnsub(); } catch (_) { /* ignore */ } });
    // Live-refresh the Cell display on PERIODICITY changes (cell / vacuum / axis_kind)
    // too -- molview.data.subscribe fires on canvas changes; store.subscribe alone misses
    // them (e.g. the Modify Cell op-tab's Update).  Periodicity accessors +
    // subscribe live on molview.data (the carve), NOT the persistence-only workspace.
    const _dm = root.molbuilder && root.molbuilder.molview
              && root.molbuilder.molview.data;
    if (_dm && typeof _dm.subscribe === "function") {
        const _dmUnsub = _dm.subscribe(function () { renderCell(); });
        cleanups.push(() => { try { _dmUnsub(); } catch (_) { /* ignore */ } });
    }

    // --- Cell page (structure-periodicity.md § 3b): page switch + read-only
    // periodicity readout.  Reads ONLY the molview accessors (getUnitCellInfo /
    // getVacuumInfo / getAxisKindInfo) -- no hand-read of the in-memory data --
    // and tags a never-set value "(default)".  Stage 2 is display; edit + Update
    // land in stage 3.
    function _round(n) { return Math.round(Number(n) * 1000) / 1000; }
    function _fmtVec(v) { return "[" + v.map(_round).join(", ") + "]"; }
    function _tag(isDefault) { return isDefault ? "  (default)" : ""; }
    function renderCell() {
        // MolView is DISPLAY-ONLY: mirror the in-memory periodicity via the molview
        // accessors, tagging a never-set parameter "(default)".  Editing lives in the
        // Modify functions (per-group Update), not here.
        const dm = root.molbuilder && root.molbuilder.molview
                 && root.molbuilder.molview.data;
        if (!dm || !els.pageCell) return;
        if (els.cellAxis && dm.getAxisKindInfo) {
            const a = dm.getAxisKindInfo();
            els.cellAxis.textContent =
                ((a.value || []).join(" / ") || "—") + _tag(a.isDefault);
        }
        if (els.cellVacuum && dm.getVacuumInfo) {
            const v = dm.getVacuumInfo();
            els.cellVacuum.textContent =
                _fmtVec(v.value || [0, 0, 0]) + _tag(v.isDefault);
        }
        // § 3c: the drawn box corner (resolved_cell_origin); "(default)" when it is
        // the auto molecule/world corner (no explicit cell_origin override).
        if (els.cellOrigin && dm.getUnitCellOriginInfo) {
            const o = dm.getUnitCellOriginInfo();
            els.cellOrigin.textContent =
                _fmtVec(o.value || [0, 0, 0]) + _tag(o.isDefault);
        }
        if (els.cellMatrix && dm.getUnitCellInfo) {
            const c = dm.getUnitCellInfo();
            _renderMatrix(els.cellMatrix, c.value);
            if (els.cellTag) els.cellTag.textContent = c.isDefault ? "(default)" : "";
        }
    }
    function _renderMatrix(el, m) {
        el.textContent = "";
        if (!m) { el.textContent = "—"; return; }
        for (var r = 0; r < 3; r++) {
            for (var col = 0; col < 3; col++) {
                var cell = document.createElement("span");
                cell.className = "cell-matrix-cell";
                cell.textContent = _round(m[r][col]).toFixed(3);
                el.appendChild(cell);
            }
        }
    }
    function _switchPage(toCell) {
        if (els.pageSel)  els.pageSel.hidden  = toCell;
        if (els.pageCell) els.pageCell.hidden = !toCell;
        if (toCell) renderCell();
    }
    on(els.pageRadioSel,  "change", () => _switchPage(false));
    on(els.pageRadioCell, "change", () => _switchPage(true));
    renderCell();   // initial fill (page is hidden until switched to)
    on(els.assignTgt,       "change", renderAssignVisibility);
    on(els.assignBtn,       "click",  onAssign);
    on(els.addBtn,          "click",  onAddToTarget);
    on(els.removeBtn,       "click",  onRemoveFromTarget);
    on(els.selectAll,       "change", (e) => {
        if (e.target.checked) store.all();
        else                  store.clear();
    });

    // Seed the Assign-target <select> from BUILTIN_TARGETS before
    // any store notify lands -- otherwise a user who opens
    // /modify, clicks the Target dropdown before the first
    // store notify (adopt) lands, sees an empty <select>.  The first
    // store notify will call renderAssignTarget which folds in
    // user-defined labels on top.
    renderAssignTarget({ atoms: [] });
    renderAssignVisibility();

    // ----- subscribe ------------------------------------------- //

    const unsubscribe = store.subscribe((s, changes) => {
        renderStatus(s);
        renderMeasurement(s);
        renderMode(s);
        renderFilters(s);
        renderAtomList(s, changes);   // uses `changes` to rebuild-vs-diff (the ~1.2s fix)
        renderAssignTarget(s);
    });

    function dispose() {
        try { unsubscribe(); } catch (e) { /* ignore */ }
        cleanups.slice().reverse().forEach((fn) => {
            try { fn(); } catch (e) { /* ignore */ }
        });
        cleanups.length = 0;
    }

    return { dispose: dispose };
}

// ``panel.forceRenderMode`` is a writable property tests + demos set to
// "virtual" or "simple" to override the auto-threshold (see VSCROLL_THRESHOLD);
// default null (auto).  Public so external code can flip it without reaching
// into module internals; also resolvable via sessionStorage / URL query.
export const panel = {
    mount:            mount,
    forceRenderMode:  null,
};

// ── Transitional global (removed once every consumer imports this module) ──
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.selection = window.molbuilder.molview.selection || {};
    window.molbuilder.molview.selection.panel = panel;
    // Module-init contract: register with the runtime so consumers can
    // whenReady("selection.panel").
    if (window.molbuilder.runtime
        && typeof window.molbuilder.runtime.register === "function") {
        window.molbuilder.runtime.register("selection.panel", panel);
    }
}
