/* Trajectory inspector core -- 3Dmol viewport + Plotly traces.
 *
 * THE shared trajectory-inspector implementation.  Mounted by the
 * /results tab via the registry adapter (lib/inspectors/trajectory.js),
 * which fetches _trajectory_inspector.html from
 * GET /partials/trajectory-inspector and calls this module's exported
 * ``mount(host, opts)`` against the resulting host element.
 *
 * History note: this module was originally lifted out of
 * static/watch/viewer.js in early 2026-05.  The /watch page route
 * was retired 2026-05-19 (this module became /results-only); the
 * lift is preserved because the core implementation was clean +
 * already root-scoped, so no rewrite was needed.
 *
 * --- How the viewer + plots stay fast --------------------------------
 * Frames are loaded once into a 3Dmol "movie" model (addModelsAsFrames)
 * and the slider / playback simply calls viewer.setFrame(idx), which is
 * fast (no DOM rebuild).  When the server reports a new mtime we rebuild
 * the model with the fresh frame list.
 *
 * Polling cadence: ~15 s (CG steps from SIESTA on a real workload take
 * far longer than that, so the network traffic is negligible).
 *
 * --- DOM scoping convention ------------------------------------------
 * The inspector body lives inside ``mountInspector(rootEl)`` so DOM
 * queries are scoped.  All inside-partial ids (defined in
 * ``_trajectory_inspector.html``) go through ``$()`` (scoped to
 * rootEl, which is the host element on /results).  The single
 * page-level lookup (``status`` banner) is inlined as
 * ``document.getElementById`` in ``setStatus`` — a no-op on
 * /results where that banner doesn't exist.
 */

(function (root) {
    "use strict";

    /**
     * Mount the trajectory inspector inside ``rootEl``.
     *
     * Parameters:
     *   rootEl  -- DOM element (or document) that contains the
     *              trajectory-inspector partial's id'd elements.
     *              On /results this is the #inspector-host element
     *              after the adapter has injected the partial HTML.
     *   opts    -- optional {file?: string}.  If ``file`` is set,
     *              the inspector calls loadByPath(file) once the
     *              UI wiring is in place, so the registry-side
     *              dispatch can mount + auto-load in one call.
     *
     * Returns a handle ``{dispose(), load(path)}``:
     *   dispose() -- stops the polling timer, cancels any in-flight
     *                HTTP request, disposes the embed handle (which
     *                stops its own animation loop + tears down its
     *                3Dmol viewer), and removes window-level
     *                listeners (resize, pagehide).  After dispose()
     *                the rootEl's contents are no longer owned by
     *                the inspector; caller may clear/replace freely.
     *   load(path) -- swap the displayed trajectory to ``path``
     *                 without re-mounting.  Used by the registry-
     *                 side dispatch when the user picks a new
     *                 ``.molwatch.log`` / ``.out`` while the
     *                 inspector is already mounted -- avoids a
     *                 full unmount/remount cycle.
     */
    function mountInspector(rootEl, opts) {
    opts = opts || {};

    const POLL_MS = 15000;

    // Inside-partial lookup: scoped to the inspector's root element
    // (i.e., the #inspector-host on /results after the adapter has
    // injected _trajectory_inspector.html).
    const $ = (id) => rootEl.querySelector("#" + id);

    // ----- Listener bookkeeping ---------------------------------
    //
    // Every element-level addEventListener inside mountInspector
    // goes through _on() so the registered teardown closure is
    // captured in _cleanups[].  dispose() walks _cleanups in
    // reverse + then runs the per-resource teardowns (polling
    // timer, in-flight HTTP request, embed handle, window-level
    // listeners).
    //
    // The window-level listeners (``resize``, ``pagehide`` on the
    // legacy /watch handoff) MUST be tracked here -- they survive
    // the host's innerHTML clear, so without explicit removal
    // every /results mount→dispose→mount cycle would accumulate
    // them.  Element listeners on partial-declared ids are GC'd
    // with their nodes; tracking them is defensive (and matches
    // the spectra core's dispose contract for cross-inspector
    // consistency).
    const _cleanups = [];
    function _on(target, event, handler, opts) {
        if (!target) return;
        target.addEventListener(event, handler, opts);
        _cleanups.push(function () {
            target.removeEventListener(event, handler, opts);
        });
    }

    /* ------------------------------------------------------------------ */
    /*  State                                                              */
    /* ------------------------------------------------------------------ */

    const state = {
        mtime: null,
        data: null,            // {frames, lattice, iterations, energies, max_forces, forces}
        currentFrame: 0,
        pollTimer: null,
        // playTimer removed by #246 A1: the embed owns playback now;
        // the parallel setInterval was racing with the embed loop.
        firstFit: true,        // call handle.refit() once after first load
        // #234 / #236 follow-up: cellShapes / forceShapes / indexLabels
        // / pickShapes arrays gone -- cell wireframe, atom-index
        // labels, force arrows, and pick halos are all owned by the
        // embed now (setCell / setLabels / arrowsPerFrame / pick.halo
        // + setPickedIndices).
        // Inspect-tab pick state lives in the embed (D3 contract);
        // trajectory reads via _handle.getPickedIndices() so there
        // is only one source of truth.  The pickedAtoms mirror was
        // dropped in #246 B3.
        // AbortControllers for in-flight HTTP requests.  Separate
        // load + poll so a new file load supersedes the previous
        // load but doesn't kill the poll cadence, and a poll never
        // cancels a load.  dispose() aborts both.  Without these,
        // rapid file-switching on /results can race responses (late
        // poll arrives after load, late load arrives after dispose,
        // etc.) -- typically harmless because the receiver checks
        // ``state.viewer`` before mutating it, but cleanly aborting
        // is cheaper than checking and matches the adapter-level
        // AbortController pattern in lib/inspectors/trajectory.js.
        loadAbort:  null,
        pollAbort:  null,
    };

    // #205 Part A migration: mount via the standard embed so the
    // knob bar (Style / Labels / Axes / Reset / PNG / Background /
    // Export) appears above the canvas, matching every other tab.
    // ``_viewer3dmol()`` returns the underlying 3Dmol viewer so the
    // existing inspector machinery (cell wireframe, atom indices,
    // force arrows, movie-mode playback via addModelsAsFrames +
    // setFrame, setClickable for picking) continues to work
    // untouched.
    //
    // Part B follow-up: migrate playback to
    // ``animation: { kind: "trajectory", frames, arrowsPerFrame }``
    // + ``appendFrames`` for live polling; drop the raw setFrame /
    // addModelsAsFrames / per-frame redraw loop.
    const _handle = window.molbuilder.viewer.embed($("viewer"), {
        style: { rep: "stick", radiusScale: 1.0 },
        // 2-atom distance pick for the Inspect panel.  ``halo: true``
        // draws the yellow sphere overlay; ``label: false`` keeps the
        // viewer uncluttered (the Inspect panel shows the readout).
        // Atom-list row clicks share the embed's pick state via
        // handle.setPickedIndices (no separate halo path).
        pick:  {
            mode:  "pair",
            halo:  true,    // accepts boolean per § 3.8 after #246 B7
            label: false,
            onPick() {
                // #246 B3: pick state lives in the embed; read
                // live via _handle.getPickedIndices() at each
                // panel refresh.  No host-side mirror.
                updateInspectPanel();
                refreshAtomListHighlights();
            },
        },
        card:  { title:        "Trajectory",
                 showInfoLine: false,
                 height:       "100%" },
        axes:    true,
        export:  { defaultName: "trajectory" },
        onError(err) {
            try { console.warn("[trajectory.inspector]",
                                err.code, err.message); }
            catch (_) {}
        },
    });

    /* ------------------------------------------------------------------ */
    /*  Status banner                                                      */
    /* ------------------------------------------------------------------ */

    /**
     * Translate a raw parser ``error_message`` string into a short
     * human-readable reason tag.  Used by the run-state info-panel
     * to show ``Reason: SCF non-convergence`` etc. instead of the
     * cryptic raw marker.
     *
     * Returns ``null`` when the message is empty (caller should
     * skip the "Reason: ..." line entirely).  Returns the raw
     * message when no classifier matches (fallback: at least the
     * user sees SOMETHING informative).
     *
     * Categories mirror the parser's fatal-marker rule set in
     * ``parsers/siesta.py`` (2026-05-29) so the UI tag tracks the
     * detection logic 1:1.  Adding a new fatal marker on the parser
     * side should add a branch here too.
     */
    function _classifyStopReason(errMsg) {
        if (!errMsg) return null;
        const lower = String(errMsg).toLowerCase();
        if (lower.indexOf("scf_not_conv") >= 0
            || lower.indexOf("scf did not converge") >= 0) {
            return "SCF non-convergence";
        }
        if (lower.indexOf("propor: error") >= 0
            || lower.indexOf("imax = 0") >= 0) {
            return "MPI rank distribution error (propor)";
        }
        if (lower.indexOf("abnormal_termination") >= 0) {
            return "abnormal termination";
        }
        if (lower.indexOf("siesta died") >= 0) {
            return "engine died";
        }
        if (lower.indexOf("siesta: error") >= 0) {
            return "engine error";
        }
        if (lower.indexOf("stopping program from node") >= 0) {
            return "node stop signal";
        }
        // No classifier matched -- fall back to raw text so the
        // user still sees what the parser flagged.  Trim to a
        // reasonable length so the badge doesn't blow up.
        const s = String(errMsg).trim();
        return s.length > 120 ? s.slice(0, 117) + "..." : s;
    }

    function setStatus(msg, kind) {
        // Status banner is page-level (not in the inspector partial),
        // a legacy of the /watch loader bar.  On /results no such
        // element exists, so this is a silent no-op there — error
        // surfacing is the embed's onError + per-inspector renderers.
        const el = document.getElementById("status");
        if (!el) return;
        el.textContent = msg;
        el.className = "status" + (kind ? " " + kind : "");
    }

    /* ------------------------------------------------------------------ */
    /*  3Dmol rendering                                                    */
    /* ------------------------------------------------------------------ */

    function framesToMultiXyz(frames) {
        const out = [];
        for (const frame of frames) {
            out.push(String(frame.length));
            out.push("");
            for (const atom of frame) {
                const [sym, x, y, z] = atom;
                out.push(
                    sym + " " +
                    x.toFixed(6) + " " +
                    y.toFixed(6) + " " +
                    z.toFixed(6)
                );
            }
        }
        return out.join("\n");
    }

    /* In 3Dmol.js a sphere `scale` value is multiplied by the element's
     * van-der-Waals radius, so per-element size differences only become
     * visible at a non-tiny scale.  Defaults below are tuned so that
     * Au / S / C / H look visibly different in every mode that draws
     * atoms.  The user's "radius scale" slider multiplies these.
     *
     * Sizing math lives in molbuilder/web/static/lib/mol-style.js so
     * the Build and Watch viewers stay in lock-step on representation
     * numerics.  The Watch tab additionally exposes a `colorscheme`
     * select, which we forward through the shared helper. */
    // Post-#205/#230 Part A+B: style (rep/radius/background/
    // colorscheme) is fully owned by the embed's standard knob
    // bar.  The trajectory-side applyStyle / styleSpec /
    // _currentRep helpers and the applyStyleAndRewireClicks
    // shim (further down) were removed by #232 review cleanup —
    // handle.setStructure re-applies the embed's current style on
    // every movie swap so no trajectory-side re-apply is needed.

    // Cell line colour is contrast-driven so the box stays visible
    // across every background option.  3Dmol takes integer or
    // string colours; a string passes through unchanged when the
    // user has selected one of the named backgrounds (white / black)
    // and we likewise pass back our integer 0x... constants.
    //
    // Picks a dark grey on light backgrounds (white / light grey)
    // and a light grey on dark backgrounds (black).  An unknown bg
    // value (user-supplied via /bgcolor query string in dev) falls
    // back to mid-grey, which is at least visible on most surfaces
    // even if not optimal.
    function cellLineColor() {
        // Post-#205 the bespoke #bg <select> is gone; the embed's
        // Background popover owns the canvas backdrop.  Read the
        // current background via the public getter (#245 added it
        // precisely so this consumer didn't need to reach into the
        // test affordance surface).  B1, 2026-06-04.
        let bg = "#ffffff";
        try {
            if (_handle && typeof _handle.getBackground === "function") {
                bg = _handle.getBackground() || "#ffffff";
            }
        } catch (_) {}
        bg = String(bg).toLowerCase();
        if (bg === "black" || bg === "#000000" || bg === "#1c1c1c"
            || bg === "0x000000") {
            return 0xcccccc;
        }
        if (bg === "white" || bg === "#ffffff" || bg === "#eeeeee"
            || bg === "0xeeeeee" || bg === "transparent") {
            return 0x444444;
        }
        return 0x888888;       // unknown bg -- safe middle ground
    }

    // ------------------------------------------------------------ //
    // #234 follow-up (2026-06-03): cell wireframe / atom-index      //
    // labels / force vectors all migrated to the embed's            //
    // declarative API.  drawCell / drawIndices / drawForces are     //
    // now thin shims that delegate to applyCell / applyIndexLabels  //
    // / applyForces below; the legacy state.cellShapes /            //
    // .indexLabels / .forceShapes arrays + the HEAD_FRAC / CONE_SEGS//
    // arrowhead constants are dead code but kept until a single     //
    // cleanup commit retires them everywhere (the old draw*         //
    // functions still hold onto the array names).                   //
    // ------------------------------------------------------------ //

    function applyCell() {
        if (!_handle) return;
        const cb = $("show-cell");
        const lat = state.data && state.data.lattice;
        if (cb && cb.checked && lat && lat.length === 3) {
            _handle.setCell({ color: cellLineColor(), radius: 0.04 });
        } else {
            _handle.setCell(false);
        }
    }

    function applyIndexLabels() {
        if (!_handle) return;
        const cb = $("show-indices");
        _handle.setLabels(cb && cb.checked
            ? { atoms: "all", format: "index", fontSize: 9 }
            : false);
    }

    // Build the ArrowSpec array for ONE frame given the current
    // force-knob settings.  Returns [] when forces are off OR the
    // parser didn't capture forces for this frame.
    function _buildArrowsForFrame(frameIdx) {
        if (!$("show-forces") || !$("show-forces").checked) return [];
        if (!state.data) return [];
        const frame  = state.data.frames[frameIdx];
        const forces = state.data.forces && state.data.forces[frameIdx];
        if (!frame || !forces || !forces.length) return [];

        const fscale    = parseFloat($("force-scale").value) || 1.0;
        const fmin      = parseFloat($("force-min").value)   || 0.0;
        const highlight = $("highlight-max").checked;

        const mags = forces.map(function (f) {
            return Math.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        });
        let maxMag = 0, maxIdx = -1;
        for (let i = 0; i < mags.length; i++) {
            if (mags[i] > maxMag) { maxMag = mags[i]; maxIdx = i; }
        }

        const arrows = [];
        for (let i = 0; i < frame.length && i < forces.length; i++) {
            const mag = mags[i];
            if (mag < fmin) continue;
            const a = frame[i], f = forces[i];
            const sx = a[1], sy = a[2], sz = a[3];
            const ex = sx + f[0] * fscale;
            const ey = sy + f[1] * fscale;
            const ez = sz + f[2] * fscale;

            // Gold for the highlighted-max arrow; dim red ->
            // bright orange-red ramp by magnitude for the rest.
            let color;
            if (highlight && i === maxIdx) {
                color = "#ffc400";
            } else {
                const t = maxMag > 0 ? mag / maxMag : 0;
                const r = Math.floor(170 + 85 * t);
                const g = Math.floor( 40 + 60 * t);
                color = "rgb(" + r + "," + g + ",32)";
            }
            arrows.push({
                start:  [sx, sy, sz],
                end:    [ex, ey, ez],
                color:  color,
                radius: 0.05 + 0.04 * (maxMag > 0 ? mag / maxMag : 0),
            });
        }
        return arrows;
    }

    // Build the FULL arrowsPerFrame array for the current
    // trajectory.  Called at setAnimation time AND on every
    // force-knob change (which calls handle.setAnimation with a
    // partial { arrowsPerFrame } update; the embed's #233
    // partial-update path mutates in place + refreshes the current
    // frame without restarting the loop).
    function buildArrowsPerFrame() {
        if (!state.data || !state.data.frames) return null;
        const apf = [];
        for (let i = 0; i < state.data.frames.length; i++) {
            apf.push(_buildArrowsForFrame(i));
        }
        return apf;
    }

    function refreshForcesStatus() {
        const cb = $("show-forces");
        if (!cb || !cb.checked) {
            setForcesStatus("Off — tick to overlay arrows.");
            return;
        }
        if (!state.data) {
            setForcesStatus("No trajectory loaded yet.");
            return;
        }
        const forces = state.data.forces
                       && state.data.forces[state.currentFrame];
        if (!forces || !forces.length) {
            setForcesStatus(
                "No force data on this frame "
              + "(parser did not capture it).");
            return;
        }
        const fmin = parseFloat($("force-min").value) || 0.0;
        const mags = forces.map(function (f) {
            return Math.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        });
        let maxMag = 0;
        for (const m of mags) if (m > maxMag) maxMag = m;
        const drawn = mags.filter(function (m) { return m >= fmin; }).length;
        if (drawn === 0) {
            setForcesStatus(
                "0 arrows shown (all |F| < threshold "
              + fmin.toFixed(3) + " eV/Å; max |F| = "
              + maxMag.toFixed(3) + ").");
        } else {
            setForcesStatus(
                "Showing " + drawn + " arrow"
              + (drawn === 1 ? "" : "s")
              + " (max |F| = " + maxMag.toFixed(3) + " eV/Å).");
        }
    }

    function applyForces() {
        if (!_handle || !state.data) return;
        const apf = buildArrowsPerFrame();
        try { _handle.setAnimation({ arrowsPerFrame: apf }); }
        catch (_) {}
        refreshForcesStatus();
    }

    function drawCell() {
        // Thin shim — see applyCell above.  Kept under the legacy
        // name so existing callers (rebuildModel, knob listeners,
        // applyStructure restore paths) don't need updating in
        // the same commit.
        applyCell();
    }

    /* ------------------------------------------------------------------ */
    /*  Atom-index labels                                                  */
    /* ------------------------------------------------------------------ */

    function drawIndices() {
        // Thin shim — see applyIndexLabels above (#234 follow-up).
        applyIndexLabels();
    }

    /* ------------------------------------------------------------------ */
    /*  Force-vector arrows                                                */
    /* ------------------------------------------------------------------ */

    // #234 follow-up: force arrows are now an embed concern.  The
    // embed's mol-axes.js ArrowSpec primitive owns the shaft + cone
    // geometry; this file just builds an ArrowSpec[] per frame via
    // _buildArrowsForFrame + buildArrowsPerFrame above and hands it
    // to handle.setAnimation({arrowsPerFrame}).  HEAD_FRAC /
    // CONE_SEGS constants gone with the bespoke addCylinder loop.

    function setForcesStatus(msg) {
        // Single point of truth for the diagnostic readout next to
        // the Show-force-vectors checkbox.  Empty string clears the
        // line so the toggle area stays compact when there's
        // nothing to report.
        const el = $("forces-status");
        if (el) el.textContent = msg || "";
    }

    function drawForces() {
        // Thin shim -- see applyForces above (#234 follow-up).
        applyForces();
    }

    /* ------------------------------------------------------------------ */
    /*  Inspect tab: two-atom picking + live distance                      */
    /* ------------------------------------------------------------------ */
    //
    // The embed owns halo rendering via pick.mode = "pair" + pick.halo;
    // _postFramePositionRedraw re-renders halos every frame so they
    // track trajectory motion.  Atom-list row clicks push into the
    // embed's pick state via handle.setPickedIndices.  Single source
    // of truth: the embed's pickedIndices.  Trajectory queries it
    // live via _handle.getPickedIndices() in updateInspectPanel +
    // refreshAtomListHighlights (no host-side mirror per D3
    // contract, fixed in #246 B3).

    function _picks() {
        return _handle ? _handle.getPickedIndices() : [];
    }

    function updateInspectPanel() {
        const pa = _picks();
        const frame = (state.data && state.data.frames[state.currentFrame])
                    || [];
        const fmt = (idx) => {
            const r = frame[idx];
            if (!r) return "—";
            // Display 1-based to match the Overlays tab's index labels
            // and the atom-list table; internal idx stays 0-based.
            return "#" + (idx + 1) + " " + r[0]
                + "  (" + r[1].toFixed(3)
                + ", " + r[2].toFixed(3)
                + ", " + r[3].toFixed(3) + ") Å";
        };
        const aCell = $("inspect-a");
        const bCell = $("inspect-b");
        const dCell = $("inspect-d");
        if (!aCell || !bCell || !dCell) return;
        aCell.textContent = pa[0] != null ? fmt(pa[0]) : "—";
        bCell.textContent = pa[1] != null ? fmt(pa[1]) : "—";
        if (pa.length === 2) {
            const a = frame[pa[0]];
            const b = frame[pa[1]];
            if (a && b) {
                const dx = a[1] - b[1];
                const dy = a[2] - b[2];
                const dz = a[3] - b[3];
                const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
                dCell.textContent = d.toFixed(4) + " Å";
            } else {
                dCell.textContent = "—";
            }
        } else {
            dCell.textContent = "—";
        }
        const hint  = $("inspect-hint");
        const table = $("inspect-table");
        const btn   = $("inspect-clear");
        if (hint)  hint.hidden  = pa.length > 0;
        if (table) table.hidden = pa.length === 0;
        if (btn)   btn.disabled = pa.length === 0;
    }

    // Toggle pick state for ``idx`` (0-based atom index): drop if
    // already picked, append if new; cap at 2 picks total (a third
    // pick drops the oldest).  Drives the atom-list row click path;
    // the embed handles the viewer-click path natively via its
    // pick.mode = "pair" wiring.  Pushes the new state into the
    // embed via setPickedIndices so halos stay in sync.
    function togglePickFromRow(idx) {
        if (!_handle) return;
        const pa = _handle.getPickedIndices();
        const existing = pa.indexOf(idx);
        if (existing !== -1) {
            pa.splice(existing, 1);
        } else {
            if (pa.length >= 2) pa.shift();
            pa.push(idx);
        }
        _handle.setPickedIndices(pa);
        updateInspectPanel();
        refreshAtomListHighlights();
    }

    function clearAtomPicks() {
        if (_handle) _handle.setPickedIndices([]);
        updateInspectPanel();
        refreshAtomListHighlights();
    }

    // Build the atom-list table inside the Inspect panel.  One row
    // per atom: index, element, current-frame (x, y, z).  Rendered
    // once per ``rebuildModel`` (atom identity is fixed across a
    // trajectory); coordinates update on every showFrame() via
    // updateAtomListCoords().  Row click toggles the pick.
    function rebuildInspectAtomList() {
        const tbody = $("inspect-atom-list-body");
        if (!tbody) return;
        tbody.innerHTML = "";
        if (!state.data || !state.data.frames.length) return;
        const frame = state.data.frames[state.currentFrame] || [];
        const frag = document.createDocumentFragment();
        for (let i = 0; i < frame.length; i++) {
            const r = frame[i];
            const tr = document.createElement("tr");
            // dataset.atomIndex stays 0-based (matches the embed's
            // pickedIndices and 3Dmol's atom.serial); the displayed ``#`` column is
            // 1-based to match the overlay labels in the Overlays tab.
            tr.dataset.atomIndex = String(i);

            // Build cells via createElement + textContent.  ``r[0]``
            // (element symbol) comes from the parsed trajectory file;
            // a maliciously-crafted .molwatch.log with an element
            // string like ``<img src=x onerror=...>`` would XSS via
            // innerHTML interpolation.  textContent escapes the value.
            const cellIdx = document.createElement("td");
            cellIdx.className = "col-idx";
            cellIdx.textContent = String(i + 1);
            const cellEl = document.createElement("td");
            cellEl.className = "col-el";
            cellEl.textContent = r[0] || "?";
            const cellX = document.createElement("td");
            cellX.className = "col-coord";
            cellX.textContent = r[1].toFixed(2);
            const cellY = document.createElement("td");
            cellY.className = "col-coord";
            cellY.textContent = r[2].toFixed(2);
            const cellZ = document.createElement("td");
            cellZ.className = "col-coord";
            cellZ.textContent = r[3].toFixed(2);
            tr.appendChild(cellIdx);
            tr.appendChild(cellEl);
            tr.appendChild(cellX);
            tr.appendChild(cellY);
            tr.appendChild(cellZ);

            // Intentional NOT routed through _on(): the row gets GC'd
            // with the tbody clear on the next renderAtomList() call,
            // taking the listener with it.  Tracking these in
            // _cleanups would grow the array unboundedly across
            // re-renders without practical benefit.  A future
            // event-delegation refactor (one tbody listener +
            // event.target.closest("tr")) would let us track via _on.
            tr.addEventListener("click", () => togglePickFromRow(i));
            frag.appendChild(tr);
        }
        tbody.appendChild(frag);
        refreshAtomListHighlights();
    }

    // Per-frame coord refresh: rebuilds only the .col-coord cells of
    // each existing row so the row click handlers stay attached.
    // Called from showFrame().
    function updateAtomListCoords() {
        const tbody = $("inspect-atom-list-body");
        if (!tbody || !state.data || !state.data.frames.length) return;
        const frame = state.data.frames[state.currentFrame] || [];
        const rows  = tbody.children;
        if (rows.length !== frame.length) {
            // Atom count changed (unlikely mid-trajectory) -- rebuild.
            rebuildInspectAtomList();
            return;
        }
        for (let i = 0; i < frame.length; i++) {
            const cells = rows[i].children;
            cells[2].textContent = frame[i][1].toFixed(2);
            cells[3].textContent = frame[i][2].toFixed(2);
            cells[4].textContent = frame[i][3].toFixed(2);
        }
    }

    function refreshAtomListHighlights() {
        const tbody = $("inspect-atom-list-body");
        if (!tbody) return;
        const picked = new Set(_picks());
        for (const tr of tbody.children) {
            tr.classList.toggle(
                "is-selected",
                picked.has(Number(tr.dataset.atomIndex)),
            );
        }
    }

    function rebuildModel() {
        if (!_handle || !state.data || !state.data.frames.length) {
            // No data to mount; the next successful rebuildModel
            // will replace whatever (if anything) is loaded.
            return;
        }
        // #230 Part B: mount the first frame's structure via
        // handle.setStructure + drive playback via handle.setAnimation
        // ({kind: "trajectory", frames}).  Replaces the raw
        // addModelsAsFrames + setFrame loop.  Trajectory-specific
        // overlays (atom indices, force vectors, picked-atom halos,
        // cell wireframe, atom-list coords) re-render per frame via
        // the onFrame callback so they track each animated frame.
        const allFrames = state.data.frames;
        // #234 follow-up: pass the lattice + first-frame xyz to
        // setStructure so the embed owns the cell wireframe via
        // setCell.  Drops the bespoke drawCell + state.cellShapes
        // machinery.
        const firstFrameXyz = framesToMultiXyz([allFrames[0]]);
        _handle.setStructure({
            xyz:     firstFrameXyz,
            lattice: state.data.lattice || undefined,
        });
        // Cell visibility tracks the #show-cell checkbox; applyCell
        // reads the checkbox + lattice and toggles the embed's
        // setCell.
        applyCell();
        // Extract per-frame coordinates ([x, y, z]) for the embed's
        // animation API.  state.data.frames carries each atom as
        // [sym, x, y, z, ...optional cols]; the embed's trajectory
        // animation only needs the position triples.
        const coordFrames = allFrames.map(function (frame) {
            return frame.map(function (atom) {
                return [atom[1], atom[2], atom[3]];
            });
        });
        // Phase 5f B-1: read the Inspect-tab playback knobs at
        // mount so a user who set #speed / #loop BEFORE the file
        // loaded gets their preferences honored (previously the
        // partial-update path was a no-op pre-mount → user's
        // pre-load slider changes silently dropped).  Phase 5g B-4:
        // match the slider-change handler exactly (1000/ms as a
        // float, no Math.round) so the same #speed value produces
        // the same period whether read at mount or after a tick.
        const speedEl = $("speed");
        const loopEl  = $("loop");
        const fpsAtMount = speedEl
            ? 1000 / (parseInt(speedEl.value, 10) || 150)
            : 10;
        _handle.setAnimation({
            kind:           "trajectory",
            frames:         coordFrames,
            // #234 follow-up: force vectors flip per frame via the
            // embed's arrowsPerFrame contract per § 3.9.  Built
            // from the trajectory's parsed forces + current force-
            // knob settings; rebuilt on every knob change via a
            // partial setAnimation update (#233).
            arrowsPerFrame: buildArrowsPerFrame(),
            fps:            fpsAtMount,
            loop:           loopEl ? !!loopEl.checked : true,
            paused:         true,
            onFrame: function (idx, _h) {
                state.currentFrame = idx;
                // Pick halos + atom-index labels follow frames
                // automatically via the embed's
                // _postFramePositionRedraw; the embed's frame
                // strip self-updates its counter / slider.  Only
                // the trajectory's own per-frame DOM bookkeeping
                // (Inspect panel atom list, force overlay status)
                // is wired here.
                updateInspectPanel();
                updateAtomListCoords();
                refreshForcesStatus();
            },
        });
        // Atom-index labels via the embed's setLabels (#234 follow-up).
        applyIndexLabels();
        // Populate the Inspect-tab atom list now that the model is
        // loaded; the list mirrors the per-frame coordinates and is
        // the keyboard-friendly path to selection.
        rebuildInspectAtomList();
        if (state.firstFit) {
            _handle.refit();
            state.firstFit = false;
        }
    }

    // applyStyleAndRewireClicks removed by #232 review cleanup.
    // The embed's standard knob bar preserves clickability via the
    // pickWired flag (#225) when its setStyle re-applies, so no
    // trajectory-side re-arm is needed.  The bespoke dropdowns
    // that used to call this helper are also gone (#205 Part A).

    function showFrame(idx) {
        if (!state.data || !state.data.frames.length) return;
        const n = state.data.frames.length;
        idx = Math.max(0, Math.min(n - 1, idx));
        // #230 Part B: delegate to handle.setAnimationFrame; the
        // embed fires onFrame which runs the per-frame side
        // effects (updateInspectPanel, updateAtomListCoords,
        // refreshForcesStatus).  Pick halos + atom-index labels
        // follow frames natively in the embed via
        // _postFramePositionRedraw.  See rebuildModel() above.
        if (_handle && typeof _handle.setAnimationFrame === "function") {
            _handle.setAnimationFrame(idx);
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Plotly traces                                                      */
    /* ------------------------------------------------------------------ */

    // Build Plotly ``shapes`` + ``annotations`` for stage boundaries
    // when state.data is a multi-stage merged trajectory.  Each
    // stage transition (where a new source .molwatch.log begins)
    // gets a dashed vertical line + a small label at the top of the
    // plot showing the stage name.  Returns {shapes, annotations}
    // (both empty arrays for single-stage runs).
    function stageMarkers() {
        const stages = state.data && state.data.stages;
        if (!Array.isArray(stages) || stages.length < 2) {
            return { shapes: [], annotations: [] };
        }
        const shapes = [];
        const annotations = [];
        for (const s of stages) {
            // Skip the line at frame 0 (start of the first stage --
            // would just be the y-axis).  Always add the label.
            if (s.start_frame > 0) {
                shapes.push({
                    type: "line",
                    xref: "x", yref: "paper",
                    x0: s.start_frame, x1: s.start_frame,
                    y0: 0, y1: 1,
                    line: { color: "#888", width: 1, dash: "dash" },
                });
            }
            const labelX = s.start_frame
                + Math.max(1, Math.floor((s.n_frames - 1) / 2));
            const stageLabel = s.name.replace(/\.molwatch\.log$/, "");
            annotations.push({
                x: labelX,
                y: 1.02,
                xref: "x", yref: "paper",
                text: stageLabel,
                showarrow: false,
                font: { size: 10, color: "#888" },
                xanchor: "center",
                yanchor: "bottom",
            });
        }
        return { shapes, annotations };
    }

    function makePlots() {
        if (!state.data) return;
        const x = state.data.iterations;
        const stageMx = stageMarkers();

        Plotly.react("energy-plot", [{
            x: x,
            y: state.data.energies,
            mode: "lines+markers",
            line: { color: "#1f77b4", width: 1.5 },
            marker: { size: 6 },
            name: "E_KS",
            connectgaps: false,
        }], {
            title: { text: "Total energy", font: { size: 13 } },
            // automargin: true lets Plotly pick the left/bottom
            // margins from the actual tick-label widths -- shorter
            // axis numbers thus claw back lateral space.
            margin: { l: 8, r: 12, t: 32, b: 32 },
            // No fixed dtick: let Plotly pick a sparse number of
            // ticks (~5-7) and skip integer labels at high frame
            // counts.  tickformat=".6~r" gives "shortest unique"
            // numbers (drops trailing zeros), so e.g. an energy
            // like -2073.1000 shows as "-2073.1" instead of
            // "-2073.1000" -- big lateral-space win.
            xaxis: { title: { text: "CG step", standoff: 4 },
                     zeroline: false, automargin: true,
                     nticks: 6 },
            yaxis: { title: { text: "E_KS (eV)", standoff: 4 },
                     tickformat: ".6~r", zeroline: false,
                     automargin: true, nticks: 5 },
            font: { family: "system-ui, sans-serif", size: 10 },
            shapes:      stageMx.shapes,
            annotations: stageMx.annotations,
        }, { displayModeBar: false, responsive: true });

        Plotly.react("force-plot", [{
            x: x,
            y: state.data.max_forces,
            mode: "lines+markers",
            line: { color: "#d62728", width: 1.5 },
            marker: { size: 6 },
            name: "Max |F|",
            connectgaps: false,
        }], {
            title: { text: "Max force", font: { size: 13 } },
            margin: { l: 8, r: 12, t: 32, b: 32 },
            xaxis: { title: { text: "CG step", standoff: 4 },
                     zeroline: false, automargin: true,
                     nticks: 6 },
            yaxis: { title: { text: "Max |F| (eV/\u00C5)", standoff: 4 },
                     tickformat: ".3~r",
                     rangemode: "tozero", zeroline: false,
                     automargin: true, nticks: 5 },
            font: { family: "system-ui, sans-serif", size: 10 },
            shapes:      stageMx.shapes,
            annotations: stageMx.annotations,
        }, { displayModeBar: false, responsive: true });

        renderScfProgress();

        // ----- Post-layout resize fix --------------------------- //
        //
        // The .plots-row CSS uses `grid-template-columns:
        // repeat(auto-fit, minmax(280px, 1fr))`.  When
        // renderScfProgress unhides the two SCF plots, the grid
        // reflows from 2 cells to 4 cells -- each cell goes from
        // ~50% of the row width down to ~25%.  Plotly's
        // `responsive: true` config only listens for window
        // resize, NOT for CSS-grid track changes, so the energy
        // + force plots' SVGs stay at their original (wider) size
        // and overlap into the neighbouring cells.  Same problem
        // in reverse when SCF data goes away later.
        //
        // Fix: explicitly resize every visible plot once the
        // visibility decisions are stable.  Plotly.Plots.resize
        // is a no-op if the container size didn't actually
        // change, so this is cheap to run unconditionally.
        ["energy-plot", "force-plot",
         "scf-energy-plot", "scf-gnorm-plot"].forEach((id) => {
            const el = $(id);
            if (el && !el.hidden) {
                try { Plotly.Plots.resize(el); }
                catch (_) { /* element not yet plotted; ignore */ }
            }
        });
    }

    /*  Render the SCF-iteration progress for the most recent step.
     *  Engine-agnostic: works for PySCF (gnorm/ddm) and SIESTA
     *  (dHmax/dDmax) by checking which keys are present in the
     *  per-cycle dicts.  Hidden when scf_history is empty (e.g.
     *  PySCF without its .log alongside the trajectory, or any
     *  format that doesn't surface per-cycle SCF detail).
     */
    function renderScfProgress() {
        const section = $("scf-section");
        const scfEnergyEl = $("scf-energy-plot");
        const scfGnormEl  = $("scf-gnorm-plot");
        const history = state.data && state.data.scf_history;
        const hideScf = () => {
            section.hidden = true;
            scfEnergyEl.hidden = true;
            scfGnormEl.hidden  = true;
        };
        if (!history || history.length === 0) {
            hideScf();
            return;
        }

        // Walk backwards to find the most recent NON-EMPTY SCF run.
        // The initial-preview block (now emitted before preopt starts,
        // so the file is non-empty from the very first second) carries
        // an intentionally empty `scf_history begin / end` pair -- no
        // SCF has run yet at preview time.  Same shape can appear for
        // any opt step whose SCF detail wasn't captured.  Without this
        // walk, history[length-1] = [] and `current[0].gnorm` throws.
        let current = null, stepIdx = history.length - 1;
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i] && history[i].length > 0) {
                current = history[i];
                stepIdx = i;
                break;
            }
        }
        if (current === null) {
            // No step has SCF detail yet (e.g., file contains only
            // the initial preview).  Hide the SCF panel until the
            // first real SCF block lands.
            hideScf();
            return;
        }
        section.hidden     = false;
        scfEnergyEl.hidden = false;
        scfGnormEl.hidden  = false;

        const cycles   = current.map(c => c.cycle);
        const energies = current.map(c => c.energy);

        // Pick the residual: prefer PySCF's |g|, fall back to
        // SIESTA's dHmax.  Both decrease toward 0 during convergence
        // and look natural on a log y-axis.
        let residual, residualName, residualUnit, residualColor;
        if (current[0].gnorm !== undefined) {
            residual      = current.map(c => c.gnorm);
            residualName  = "|g|";
            residualUnit  = "eV/Å";
            residualColor = "#fbbf24";   // amber
        } else if (current[0].dHmax !== undefined) {
            residual      = current.map(c => c.dHmax);
            residualName  = "dHmax";
            residualUnit  = "eV";
            residualColor = "#4ade80";   // green
        } else {
            residual = null;
        }

        // Engine-aware labels.  state.format is whatever the parser
        // wrote to source_format -- "siesta", "pyscf", or "molwatch"
        // when the file was a unified molwatch.log without an engine
        // header (the molwatch.log emitter normally fills in "siesta"
        // or "pyscf" so this last case is the rare fallback).
        //
        // Banner-title precision rule: be specific where we have
        // certainty, generic where we don't.
        //   * SIESTA only implements DFT (Kohn-Sham), so we can
        //     unambiguously call it "DFT SCF".
        //   * PySCF can do either HF or DFT depending on the method
        //     (RHF/UHF vs RKS/UKS) chosen by the script that wrote
        //     the log.  The parser doesn't extract that today, so we
        //     stay with the generic "SCF" -- which is correct for
        //     either flavour.
        //   * Unknown engine: neutral "SCF" without engine name.
        let bannerTitle, stepLabel;
        if (state.format === "siesta") {
            bannerTitle = "SIESTA DFT SCF progress";
            stepLabel   = "CG/MD step";
        } else if (state.format === "pyscf") {
            bannerTitle = "PySCF SCF progress";
            stepLabel   = "Geom-opt step";
        } else {
            bannerTitle = "SCF progress";
            stepLabel   = "Opt step";
        }
        $("scf-title").textContent = bannerTitle;

        const lastDe = current[current.length - 1].delta_E;
        let statusText = stepLabel + " " + stepIdx
            + " — SCF cycle " + cycles[cycles.length - 1]
            + " (" + current.length + " iters)";
        if (residual !== null) {
            const lastResid = residual[residual.length - 1];
            statusText += ", " + residualName + "="
                       + lastResid.toExponential(2) + " " + residualUnit;
        }
        statusText += ", ΔE=" + lastDe.toExponential(2) + " eV";
        $("scf-status").textContent = statusText;

        // SCF energy convergence within the current step.
        Plotly.react("scf-energy-plot", [{
            x: cycles,
            y: energies,
            mode: "lines+markers",
            line: { color: "#6ba6ff", width: 1.5 },
            marker: { size: 5 },
            name: "E",
        }], {
            title: { text: "SCF energy (current step)", font: { size: 12 } },
            margin: { l: 8, r: 12, t: 28, b: 30 },
            xaxis: { title: { text: "SCF cycle", standoff: 4 },
                     zeroline: false, automargin: true,
                     nticks: 6 },
            yaxis: { title: { text: "E (eV)", standoff: 4 },
                     tickformat: ".6~r", zeroline: false,
                     automargin: true, nticks: 5 },
            font: { family: "system-ui, sans-serif", size: 10 },
        }, { displayModeBar: false, responsive: true });

        // Residual on log y-axis -- spans many decades during SCF.
        const resPlotEl = $("scf-gnorm-plot");
        if (residual !== null) {
            resPlotEl.hidden = false;
            Plotly.react("scf-gnorm-plot", [{
                x: cycles,
                y: residual,
                mode: "lines+markers",
                line: { color: residualColor, width: 1.5 },
                marker: { size: 5 },
                name: residualName,
            }], {
                title: { text: "SCF residual " + residualName,
                         font: { size: 12 } },
                margin: { l: 8, r: 12, t: 28, b: 30 },
                xaxis: { title: { text: "SCF cycle", standoff: 4 },
                         zeroline: false, automargin: true,
                         nticks: 6 },
                yaxis: { title: { text: residualName + " (" + residualUnit + ")",
                                  standoff: 4 },
                         type: "log", zeroline: false, tickformat: ".0e",
                         automargin: true, nticks: 5 },
                font: { family: "system-ui, sans-serif", size: 10 },
            }, { displayModeBar: false, responsive: true });
        } else {
            resPlotEl.hidden = true;
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Polling                                                            */
    /* ------------------------------------------------------------------ */

    async function pollOnce() {
        // Each poll tick supersedes the previous in-flight poll --
        // if a previous tick is still on the wire we don't care
        // about it any more (state.mtime in the URL pins what we're
        // checking against, and an answer to an old mtime is stale
        // by the time it arrives).
        if (state.pollAbort) state.pollAbort.abort();
        state.pollAbort = new AbortController();
        const signal = state.pollAbort.signal;
        try {
            const url = state.mtime !== null
                ? "/api/watch/data?mtime=" + encodeURIComponent(state.mtime)
                : "/api/watch/data";
            const r = await fetch(url, { signal: signal })
                .then(x => x.json());
            if (signal.aborted) return;
            if (!r.ok) {
                setStatus(r.error || "Server error.", "error");
                return;
            }
            if (!r.changed) {
                setStatus(
                    "Up to date \u2014 " + state.data.frames.length
                    + " " + (state.label || "") + " frames "
                    + "(checked " + new Date().toLocaleTimeString() + ").",
                    "ok"
                );
                return;
            }
            applyNewData(r);
        } catch (e) {
            // AbortError: another poll superseded this one, or
            // dispose() ran.  Silent -- the next tick (or the
            // unmount) is the authoritative state.
            if (e.name === "AbortError") return;
            setStatus("Network error: " + e.message, "error");
        }
    }

    // Compact "1h 23m" / "12m 5s" / "45s" formatter for elapsed seconds.
    // Hours-and-minutes for long runs; minutes-and-seconds for medium;
    // bare seconds for short.  Negative inputs (clock skew between
    // server and the file's wall_time) are clamped to 0.
    function fmtElapsed(secs) {
        if (!Number.isFinite(secs) || secs < 0) secs = 0;
        secs = Math.floor(secs);
        if (secs < 60)   return secs + "s";
        if (secs < 3600) return Math.floor(secs/60) + "m " + (secs%60) + "s";
        return Math.floor(secs/3600) + "h " + Math.floor((secs%3600)/60) + "m";
    }

    /* Wall-clock formatter for the run-state badge's "last result at"
     * detail.  Today's results show just HH:MM:SS so the badge stays
     * compact; older results prepend MMM DD so the user can tell the
     * difference between a 12 h-old "Ongoing" (probably stalled) and
     * one from 5 min ago.  Input is a Unix-epoch SECONDS timestamp
     * (matches the wire format of mtime / wall_times). */
    function fmtTimestamp(epochSecs) {
        if (!Number.isFinite(epochSecs)) return "";
        const d   = new Date(epochSecs * 1000);
        const now = new Date();
        const sameDay = d.getFullYear() === now.getFullYear()
            && d.getMonth() === now.getMonth()
            && d.getDate()  === now.getDate();
        const t = d.toLocaleTimeString();
        if (sameDay) return t;
        // "MMM D, HH:MM:SS" -- locale-aware date prefix, no year (the
        // 99% case for "this is from yesterday or last week").
        const date = d.toLocaleDateString(undefined,
            { month: "short", day: "numeric" });
        return date + " " + t;
    }

    function _renderRuntimeInfo(rt) {
        // rt is Trajectory.runtime_info (a {key: value} bag the
        // parser populated from the file's header).  Renders into
        // the compact ``#runtime-summary`` span inside the run-state
        // badge -- one organised line, small font, low-key.  Empty
        // (no runtime_info on the file) -> blank, no visible row.
        const el = $("runtime-summary");
        if (!el) return;
        if (!rt || Object.keys(rt).length === 0) {
            el.textContent = "";
            return;
        }
        const parts = [];
        // GPU first -- it's the question the user actually asks
        // ("did this run use the GPU?").  Strip the "NVIDIA GeForce"
        // prefix that bloats the line without adding info.
        if (rt.gpu_used === true) {
            const name = String(rt.gpu_name || "GPU")
                .replace(/^NVIDIA GeForce /, "")
                .replace(/^NVIDIA /, "");
            let gpu = "GPU " + name;
            if (rt.gpu_compute_capability) gpu += " CC" + rt.gpu_compute_capability;
            if (rt.cuda_version)            gpu += "/CUDA" + rt.cuda_version;
            parts.push(gpu);
        } else if (rt.gpu_requested === true) {
            parts.push("CPU (GPU requested, fell back)");
        } else if (rt.gpu_used === false) {
            parts.push("CPU");
        }
        // Threads.  Compact form: "20T BLAS=1" (T = threads).
        const t = (rt.n_threads_pyscf != null)
            ? rt.n_threads_pyscf
            : rt.n_threads_omp;
        if (t != null) {
            let cpu = t + "T";
            if (rt.n_threads_blas != null) cpu += " BLAS=" + rt.n_threads_blas;
            parts.push(cpu);
        }
        // Memory cap, in GB if >= 1024 MB.
        if (rt.max_memory_mb != null) {
            const mb = Number(rt.max_memory_mb);
            parts.push(mb >= 1024
                ? (mb / 1024).toFixed(mb % 1024 === 0 ? 0 : 1) + " GB"
                : mb + " MB");
        }
        if (rt.hostname) parts.push(rt.hostname);
        el.textContent = parts.join(" · ");
    }

    function _renderParseWarnings(warnings) {
        // Level-3 parser contract (2026-05-28).  Render the
        // non-fatal parse issues into a collapsible panel.  Hidden
        // when there are no issues -- a well-parsed file shows no
        // clutter at all.
        const panel = $("parse-warnings");
        const list  = $("parse-warnings-list");
        const label = $("parse-warnings-count");
        if (!panel || !list) return;
        const ws = Array.isArray(warnings) ? warnings : [];
        if (ws.length === 0) {
            panel.hidden = true;
            list.innerHTML = "";
            return;
        }
        panel.hidden = false;
        if (label) {
            label.textContent = "Parsing notes — "
                + ws.length + (ws.length === 1 ? " issue" : " issues");
        }
        // Build the list.  Each entry: line number, snippet (mono),
        // error (italics).
        list.innerHTML = "";
        for (const w of ws) {
            const li = document.createElement("li");
            li.className = "parse-warning";
            const meta = document.createElement("span");
            meta.className = "parse-warning-meta";
            meta.textContent = "line " + (w.line_no || "?")
                + " · [" + (w.category || "—") + "]";
            const err = document.createElement("span");
            err.className = "parse-warning-error";
            err.textContent = w.error || "(no message)";
            const snip = document.createElement("pre");
            snip.className = "parse-warning-snippet";
            snip.textContent = w.snippet || "";
            li.appendChild(meta);
            li.appendChild(err);
            li.appendChild(snip);
            list.appendChild(li);
        }
    }

    function _latticeEqual(a, b) {
        // Element-wise compare for the 3×3 lattice array (or null on
        // both sides).  Avoids the JSON.stringify proxy which is
        // brittle to parser-shape changes (sparse arrays, key
        // ordering on flattened encodings, etc.).
        if (a === b) return true;
        if (a == null || b == null) return false;
        if (!Array.isArray(a) || !Array.isArray(b)) return false;
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            const ra = a[i];
            const rb = b[i];
            if (!Array.isArray(ra) || !Array.isArray(rb)) return false;
            if (ra.length !== rb.length) return false;
            for (let j = 0; j < ra.length; j++) {
                if (ra[j] !== rb[j]) return false;
            }
        }
        return true;
    }

    function applyNewData(r) {
        // Decide poll path: strict-tail-append (cheap; keeps playback
        // running) vs full rebuild (structure changed or frames
        // shrank).  Strict-tail criteria: existing data present, new
        // frame count > old, first-frame atom count unchanged
        // (proxy for atom-topology unchanged), and the new lattice
        // is equal-or-absent.  Server-side trajectory parsers are
        // monotonic so this catches the common live-watch case.
        const oldData = state.data;
        const oldLen  = oldData ? oldData.frames.length : 0;
        const newLen  = (r.data && r.data.frames && r.data.frames.length) || 0;
        const sameAtomCount = oldData
            && oldLen > 0 && newLen > 0
            && oldData.frames[0].length === r.data.frames[0].length;
        const canAppend = _handle
            && sameAtomCount
            && newLen > oldLen
            && _latticeEqual(oldData && oldData.lattice,
                             r.data    && r.data.lattice);

        const wasAtEnd = !oldData || state.currentFrame >= oldLen - 1;

        state.mtime  = r.mtime;
        state.data   = r.data;
        state.format = r.format || (r.data && r.data.source_format) || "?";
        state.label  = r.label  || state.format;
        _renderRuntimeInfo(state.data && state.data.runtime_info);
        _renderParseWarnings(state.data && state.data.parse_warnings);

        const n = state.data.frames.length;
        if (n === 0) {
            setStatus("File loaded ("+ state.label +") but no frames yet.", "");
            return;
        }
        // Frames now exist -> enable the "Save current frame as XYZ"
        // button (it's disabled-by-default in the template so users
        // can't click it before any data is loaded).  The frame
        // count + slider max are owned by the embed's auto-mounted
        // frame strip (§ 6.3); no host-side DOM update needed.
        $("save-frame").disabled = false;

        if (canAppend) {
            // Strict tail-append: hand the new frames to the embed
            // without resetting the animation loop.  Playback that
            // was running keeps running; arrowsPerFrame for the new
            // frames is recomputed from the updated state.data.
            const wasPlaying = _handle.isAnimationPlaying();
            const newCoords = [];
            for (let i = oldLen; i < newLen; i++) {
                newCoords.push(r.data.frames[i].map(
                    (atom) => [atom[1], atom[2], atom[3]]));
            }
            _handle.appendFrames(newCoords);
            // arrowsPerFrame is a live array on the animation; the
            // embed's renderer reads it per-frame.  setAnimation
            // with only ``arrowsPerFrame`` routes through the
            // trajectory in-place fast path (no setInterval
            // re-arm).  See mol-viewer-embed.js setAnimation merge.
            try {
                _handle.setAnimation({
                    arrowsPerFrame: buildArrowsPerFrame(),
                });
            } catch (_) {}
            // Seek to the new tail if the user was watching the end;
            // otherwise leave the playhead where it is.
            // showFrame() routes through setAnimationFrame which
            // stops the loop as a side effect, so restore playback
            // afterwards if we tore it down.
            if (wasAtEnd) showFrame(n - 1);
            if (wasPlaying && !_handle.isAnimationPlaying()) {
                try { _handle.playAnimation(); } catch (_) {}
            }
        } else {
            rebuildModel();
            const targetIdx = wasAtEnd
                ? n - 1
                : Math.min(state.currentFrame, n - 1);
            showFrame(targetIdx);
        }
        makePlots();

        // Signal to the /results tab-level picker (and any other
        // listener that wants to drop a "Loading…" / "Parsing…"
        // status overlay) that the first render of this file is now
        // visible on screen.  Deferred to two consecutive
        // ``requestAnimationFrame`` ticks so the dispatch fires
        // AFTER the browser has had a chance to paint the new
        // ``frame-tot`` / slider state, run 3Dmol's GPU render, and
        // commit Plotly's plot drawing -- the user's 2026-06-01
        // report was that the picker meta cleared while ``frame-tot``
        // visibly still read "0 / 0" and the viewer was blank, which
        // happens because a synchronous dispatch arrives in the same
        // event-loop tick as the textContent assignment but the
        // browser hasn't painted yet.  Double rAF (rAF inside rAF)
        // is the cheapest pattern that ensures we're past one full
        // paint cycle.  Subsequent polls also dispatch but that's a
        // no-op for the picker (it idempotently clears the parse
        // status; ``parsingFor`` is null on poll-triggered fires).
        try {
            const dispatch = () => document.dispatchEvent(new CustomEvent(
                "molbuilder:inspector:ready",
                { detail: { inspector: "trajectory", frames: n } }
            ));
            if (typeof requestAnimationFrame === "function") {
                requestAnimationFrame(() => requestAnimationFrame(dispatch));
            } else {
                dispatch();
            }
        } catch (_) {
            // CustomEvent / rAF unavailable in some ancient runtimes;
            // the picker's timeout fallback covers it.
        }

        const ts = new Date(r.mtime * 1000).toLocaleTimeString();
        // Elapsed simulation time -- wall_times[] is per-frame Unix
        // epoch.  Total sim time = last_wall - first_wall.  Falls
        // back silently when wall_times is absent.
        const wt = state.data.wall_times || [];
        const firstWall = wt.find(v => Number.isFinite(v));
        let lastWall = null;
        for (let i = wt.length - 1; i >= 0; i--) {
            if (Number.isFinite(wt[i])) { lastWall = wt[i]; break; }
        }
        let elapsed = null;
        if (firstWall != null && lastWall != null) {
            elapsed = lastWall - firstWall;
        }

        // Run-state badge: authoritative when the writer emitted
        // explicit end-of-run markers (PySCF .molwatch.log:
        // "# concluded:" / "# error:"; SIESTA .out: ">> End of run").
        // Defaults to "ongoing" when neither is present.  The badge
        // is the user's primary "is this finished?" signal -- ONE
        // location instead of inferring from various places.
        const runState  = (state.data.run_state || "ongoing").toLowerCase();
        const errMsg    = state.data.error_message || "";
        const badge     = $("run-state-badge");
        const badgeLab  = $("run-state-label");
        const badgeDet  = $("run-state-detail");
        if (badge) {
            badge.classList.remove(
                "run-state-blank", "run-state-finished",
                "run-state-ongoing", "run-state-error",
            );
            badge.hidden = false;
            // "Last result at <time>": prefer the per-frame wall_time
            // from the simulation log (authoritative; this is when the
            // simulation itself produced the result), fall back to
            // the file's mtime (the only timestamp available when the
            // engine doesn't emit per-step wall clocks, e.g. raw SIESTA
            // .out without molwatch hooks).  This is DIFFERENT from
            // "Watch tab last polled at X" -- which is a client-side
            // concern not shown on the badge.
            const lastResultEpoch = Number.isFinite(lastWall)
                ? lastWall
                : (Number.isFinite(state.mtime) ? state.mtime : null);
            const lastResultTs = (lastResultEpoch != null)
                ? fmtTimestamp(lastResultEpoch)
                : "";
            const elapsedTxt = (elapsed != null)
                ? fmtElapsed(elapsed)
                : "";
            const joinParts = (...parts) =>
                parts.filter(s => s && s.length).join(" \u00b7 ");
            if (runState === "finished") {
                badge.classList.add("run-state-finished");
                badgeLab.textContent = "Finished";
                badgeDet.textContent = joinParts(
                    lastResultTs ? "ended " + lastResultTs : "",
                    elapsedTxt ? "total " + elapsedTxt : "",
                );
                badgeDet.removeAttribute("title");
            } else if (runState === "error") {
                // 2026-05-30: "Error" relabelled to "Stopped" per user
                // feedback.  Non-convergence is a STATE, not an error
                // of the viewer / the .out file.  The actual reason
                // (SCF non-convergence, MPI fault, ...) is shown below
                // as a classified tag; the raw parser message is the
                // tooltip so power users can read it without losing
                // the visual hierarchy.
                badge.classList.add("run-state-error");
                badgeLab.textContent = "Stopped";
                const reasonTag = _classifyStopReason(errMsg);
                badgeDet.textContent = joinParts(
                    reasonTag ? "Reason: " + reasonTag : "",
                    lastResultTs ? "stopped " + lastResultTs : "",
                    elapsedTxt ? "total " + elapsedTxt : "",
                );
                // Full raw message available on hover (and for screen
                // readers via aria, since title is announced by most).
                if (errMsg) badgeDet.setAttribute("title", errMsg);
                else        badgeDet.removeAttribute("title");
            } else {
                badge.classList.add("run-state-ongoing");
                badgeLab.textContent = "Running";
                badgeDet.textContent = joinParts(
                    lastResultTs ? "last result " + lastResultTs : "",
                    elapsedTxt ? "sim time " + elapsedTxt : "",
                );
                badgeDet.removeAttribute("title");
            }
        }

        // Bottom status banner keeps the diagnostic detail (mtime,
        // frame count) -- the badge above is the user-facing state,
        // this is the technical readout.
        setStatus(
            "Loaded " + n + " " + state.label + " frames \u2014 mtime " + ts + ".",
            "ok"
        );
    }

    function startPolling() {
        if (state.pollTimer) clearInterval(state.pollTimer);
        state.pollTimer = setInterval(pollOnce, POLL_MS);
    }

    function stopPolling() {
        if (state.pollTimer) {
            clearInterval(state.pollTimer);
            state.pollTimer = null;
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Playback                                                           */
    /* ------------------------------------------------------------------ */

    // Playback control (step / play / pause / state.playTimer)
    // removed by #246 A1.  The embed owns playback now via its
    // animation loop + auto-mounted frame strip (prev/play/next
    // + slider per § 6.3).  The Inspect tab's #speed and #loop
    // controls flow into the embed via setAnimation partial
    // updates per § 3.9.  showFrame() above is still used by the
    // load path (jump to last frame on append) but goes through
    // _handle.setAnimationFrame.

    /* ------------------------------------------------------------------ */
    /*  UI wiring                                                          */
    /* ------------------------------------------------------------------ */

    // Loader-bar wiring (/watch's path-input + Load button) was
    // removed 2026-05-19 along with /watch itself.  /results drives
    // loading via the registry's mount(host, file, ctx) -- the
    // sidebar selection IS the load trigger; no loader-bar UI on
    // /results.  The applyHandoff IIFE (legacy Build→Watch
    // ?path=... query-param pre-fill) is gone for the same reason.

    async function loadByPath(path) {
        // The embed's animation starts paused (initial setAnimation
        // call below uses paused: true) so no host-side pause is
        // needed -- rebuildModel() replaces the active animation.
        // Drop atom picks: a fresh load means a new trajectory and
        // the picked indices may not exist in the new model.
        clearAtomPicks();
        setStatus("Loading\u2026", "");
        // Cancel any previous in-flight load (rapid file-switching
        // case).  Start a new controller; dispose() will abort it.
        if (state.loadAbort) state.loadAbort.abort();
        state.loadAbort = new AbortController();
        const signal = state.loadAbort.signal;
        try {
            const r = await fetch("/api/watch/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: path }),
                signal: signal,
            }).then(x => x.json());

            if (signal.aborted) return;
            if (!r.ok) {
                setStatus(r.error || "Load failed.", "error");
                return;
            }
            state.firstFit = true;
            applyNewData({
                mtime:  r.mtime,
                data:   r.data,
                format: r.format,
                label:  r.label,
            });
            // Directory mode: show the user which file the loader
            // picked, and update the input with the resolved path so
            // the next poll / page-revisit reuses it directly.
            if (r.resolved_from) {
                const baseDir = r.resolved_from.replace(/\/+$/, "");
                const fileNm  = (r.path || "").split("/").pop() || r.path;
                const ts = new Date(r.mtime * 1000).toLocaleTimeString();
                let msg;
                if (Array.isArray(r.stages) && r.stages.length > 1) {
                    // Multi-stage run -- summarise the merge so the
                    // user knows their staged trajectory was joined.
                    const parts = r.stages.map(
                        s => s.name + " (" + s.n_frames + " frame"
                             + (s.n_frames === 1 ? "" : "s") + ")"
                    );
                    msg = "Loaded " + r.stages.length
                        + " stages from " + baseDir + "/  \u2014 "
                        + parts.join(" \u2192 ")
                        + ".  Live polling: " + fileNm + " (mtime " + ts + ").";
                } else {
                    msg = "Loaded \u201c" + fileNm + "\u201d from " + baseDir
                        + "/  \u2014 mtime " + ts + ".";
                }
                setStatus(msg, "ok");
            }
            startPolling();
        } catch (e) {
            // AbortError fires when the user picks another file
            // (or dispose() runs) before this fetch completes.  Not
            // a failure -- the new load (or the dispose) is the
            // authoritative action; surfacing "Network error" here
            // would be misleading.
            if (e.name === "AbortError") return;
            setStatus("Network error: " + e.message, "error");
        }
    }

    // Loader-bar keydown (Enter on path-input → click Load) and the
    // Build→Watch ?path=<file> URL-handoff IIFE were removed
    // 2026-05-19 along with /watch.  On /results the path is set by
    // the registry's mount(host, file, ctx) call -- no URL query
    // pre-fill, no Enter-to-load shortcut.

    // Frame-slider + prev/play/pause/next click wiring removed in
    // #246 A1: those controls now live in the embed's auto-
    // mounted frame strip (§ 6.3).  The Inspect-tab playback
    // configuration (#speed, #loop) flows into the embed via
    // setAnimation partial updates per § 3.9.

    // #205 Part A: #rep / #radius / #bg / #colorscheme are all
    // owned by the embed's standard knob bar now.  Cell wireframe
    // toggle remains a trajectory-specific overlay control.
    _on($("show-cell"),   "change", drawCell);

    /* Overlays */
    _on($("show-indices"), "change", drawIndices);
    _on($("show-forces"),  "change", drawForces);
    _on($("force-scale"), "input", (e) => {
        $("force-scale-val").textContent = parseFloat(e.target.value).toFixed(1);
        drawForces();
    });
    _on($("force-min"),     "input",  drawForces);
    _on($("highlight-max"), "change", drawForces);

    // Playback configuration → embed animation partial-update
    // (§ 3.9).  Speed slider is in ms-per-frame; fps = 1000/ms.
    // Loop checkbox drives the embed's loop opt directly.
    _on($("speed"), "change", () => {
        if (!_handle) return;
        const ms = parseInt($("speed").value, 10) || 150;
        try { _handle.setAnimation({ fps: 1000 / ms }); }
        catch (_) {}
    });
    _on($("loop"), "change", () => {
        if (!_handle) return;
        try { _handle.setAnimation({ loop: !!$("loop").checked }); }
        catch (_) {}
    });

    /* ---- Inspect-tab: Clear-picks button ------------------------- */
    _on($("inspect-clear"), "click", clearAtomPicks);

    /* ---- Save current frame as XYZ ------------------------------- */
    /* Hands the displayed structure off to the next step in the
       user's pipeline -- e.g., dropping the relaxed molecule into a
       tunneling-gap setup as a bridge.  The output is plain XYZ
       (Angstrom), which any chemistry tool reads. */
    _on($("save-frame"), "click", () => {
        if (!state.data || !state.data.frames.length) return;
        const idx = state.currentFrame;
        const frame = state.data.frames[idx];
        if (!frame || !frame.length) return;

        // Comment line: include the engine, step index, and energy
        // when known.  Anything we know stays out of the structural
        // part of the file -- a downstream parser that doesn't
        // recognise the comment just sees a regular XYZ.
        const engine = (state.format && state.format !== "?")
            ? state.format : "molwatch";
        const energyEv = (state.data.energies || [])[idx];
        const fileStem = (state.label && state.label.replace(/[^A-Za-z0-9._-]+/g, "_"))
            || engine;
        // Prefer per-stage step indices when present (multi-stage
        // merge) -- those match the source log's numbering.  Fall
        // back to the global iterations for single-stage runs.
        const stepIdx = ((state.data.step_indices || [])[idx])
                     ?? ((state.data.iterations  || [])[idx]);
        const stepLabel = (stepIdx !== undefined && stepIdx !== null)
            ? "step " + stepIdx : "frame " + idx;

        let comment = "molwatch export from " + engine + " — " + stepLabel;
        if (typeof energyEv === "number" && isFinite(energyEv)) {
            comment += "  energy " + energyEv.toFixed(8) + " eV";
        }

        // Build the XYZ text.  Format: standard 4-column XYZ, fixed
        // 8-decimal coordinates (matches molbuilder's _save_xyz).
        const lines = [String(frame.length), comment];
        for (const row of frame) {
            const el = String(row[0]).padEnd(2);
            const x  = (+row[1]).toFixed(8).padStart(14);
            const y  = (+row[2]).toFixed(8).padStart(14);
            const z  = (+row[3]).toFixed(8).padStart(14);
            lines.push("   " + el + "  " + x + "  " + y + "  " + z);
        }
        const text = lines.join("\n") + "\n";

        // Trigger a browser download.  Filename includes the step
        // index so multiple saves don't collide.
        const filename = fileStem + "_step" + (stepIdx !== undefined
            ? stepIdx : idx) + ".xyz";
        const blob = new Blob([text], { type: "chemical/x-xyz" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        setStatus("Saved " + filename + " (" + frame.length
            + " atoms).", "ok");
    });

    /* ---- Tabs (Style / Overlays / Playback) ---------------------- */
    /* Compact controls: only one panel visible at a time so the
       aside never outgrows the viewer height.  CSS does the visibility
       toggle via `is-active`; we just sync the classes here.
       Queries are scoped to rootEl -- the page may host OTHER
       inspectors (spectra etc.) that also use a .ctab/.ctab-panel
       convention; a document-wide query would cross-fire. */
    rootEl.querySelectorAll(".ctab").forEach((btn) => {
        _on(btn, "click", () => {
            const target = btn.dataset.tab;
            rootEl.querySelectorAll(".ctab").forEach((b) => {
                const active = (b === btn);
                b.classList.toggle("is-active", active);
                // Sync aria-selected so screen readers track the
                // active tab; the visual is-active class alone
                // doesn't reach assistive tech.
                b.setAttribute("aria-selected", active ? "true" : "false");
            });
            rootEl.querySelectorAll(".ctab-panel").forEach(
                (p) => p.classList.toggle(
                    "is-active", p.dataset.panel === target
                )
            );
        });
    });

    // Viewer resize: the embed installs its own ResizeObserver on the
    // canvas host so 3Dmol's WebGL viewport stays in sync as the card
    // resizes (clamp(360px, 52vh, 500px)).  No window-resize wiring
    // needed here.

    // ---- pageshow / visibilitychange: force-refresh on tab re-entry  //
    //
    // The inspector polls every POLL_MS (~15 s) for mtime drift, but
    // setInterval timers are PAUSED while the page sits in the
    // browser's bfcache (Chromium + Firefox default).  After a back/
    // forward restore the next interval fires up to POLL_MS LATER --
    // so a user who generated more frames in another tab can be
    // staring at the old trajectory for up to 15 s before the poll
    // catches up.  Same shape as the 2026-06-02 /results file-picker
    // bug (#192): the polling mechanism keeps state fresh AT STEADY
    // STATE but doesn't react to the "user returned to this tab"
    // event.
    //
    // Fix mirrors the file-picker pattern: hook pageshow (covers
    // bfcache restore + initial load) and visibilitychange ->
    // visible (covers backgrounded-tab re-focus) and call
    // ``pollOnce()`` immediately so an mtime drift surfaces within
    // one network round-trip of the user being back on the tab.
    // ``pollOnce`` is a no-op when no trajectory is loaded
    // (``state.mtime === null``), so wiring these handlers at mount
    // time is safe even before opts.file resolves.
    function _onPageShow(_evt) {
        if (state.mtime !== null) pollOnce();
    }
    function _onVisibilityChange(_evt) {
        if (document.visibilityState === "visible"
            && state.mtime !== null) {
            pollOnce();
        }
    }
    _on(window,   "pageshow",          _onPageShow);
    _on(document, "visibilitychange",  _onVisibilityChange);

    // ``WATCH_PATH_KEY`` sessionStorage persistence (Build ↔ Watch
    // handoff for the legacy /watch loader bar) was removed
    // 2026-05-19 along with /watch.  On /results, file selection is
    // driven by the sidebar's ``projects.onChange`` event + the
    // registry's mount(host, file, ctx) -- no loader-bar input
    // exists to persist.

    // ---- Auto-load + handle assembly --------------------------- //
    //
    // If the caller asked for an initial file (Stage 1D's
    // /results-side mount), load it now.  /watch's auto-bootstrap
    // never passes opts.file -- the user types into the loader bar
    // there.  The applyHandoff block above already handled URL-
    // param pre-fill for the /watch case.
    if (opts.file) {
        loadByPath(opts.file);
    }

    // The handle the caller uses to dispose + control the mounted
    // inspector.  Required for /results' registry-based dispatch
    // (the registry calls dispose() before mounting the next
    // inspector); /watch's bootstrap holds the handle for
    // completeness but never disposes (the tab lives forever).
    return {
        /**
         * Tear down every long-lived resource this mount created:
         * polling timer, in-flight HTTP requests, and the embed
         * handle (the embed's dispose() releases the WebGL
         * context's bookkeeping + its animation loop + its
         * ResizeObserver; the canvas itself is freed when the
         * host's innerHTML is cleared by the caller).  The
         * parallel state.playTimer + the window resize listener
         * were retired in #246 — the embed owns playback +
         * resize.
         */
        dispose() {
            // Walk listener teardowns in reverse so the most recent
            // registration tears down first (LIFO -- matches the
            // order they'd be attached in a remount cycle).  Each
            // teardown is the closure ``_on()`` pushed at register
            // time; covers the window resize listener AND every
            // element-level listener attached during mount.
            // Per-cleanup catch keeps one buggy teardown from
            // blocking the rest.
            while (_cleanups.length) {
                try { _cleanups.pop()(); } catch (_) {}
            }
            // Abort in-flight HTTP requests so their then-handlers
            // don't fire against torn-down DOM (the dispose path is
            // typically triggered by /results' registry mid-fetch
            // when the user picks a different file).
            if (state.loadAbort) { state.loadAbort.abort(); state.loadAbort = null; }
            if (state.pollAbort) { state.pollAbort.abort(); state.pollAbort = null; }
            stopPolling();
            // playTimer cleanup removed in #246 A1 (no parallel
            // playback loop anymore); handle.dispose() below tears
            // down the embed's animation loop on its own.
            // handle.dispose() tears down the embed's animation loop,
            // ResizeObserver, knob bar, models / shapes / labels and
            // releases its references on the 3Dmol viewer.
            if (_handle && typeof _handle.dispose === "function") {
                try { _handle.dispose(); } catch (_) {}
            }
        },
        /**
         * Swap the displayed trajectory without re-mounting the
         * inspector.  Equivalent to typing into the path input on
         * /watch and clicking Load.  Used by /watch's loader-bar
         * Load button + (planned) by the registry-side dispatch
         * when the user picks a new ``.molwatch.log`` while the
         * trajectory inspector is already mounted.
         */
        load(path) { return loadByPath(path); },
    };

    }   // ----- end of mountInspector(rootEl, opts) -----

    // Export for both consumers (watch/viewer.js bootstrap on /watch,
    // lib/inspectors/trajectory.js on /results).  Each consumer is
    // responsible for picking when + where to mount; this module
    // does NOT self-bootstrap on page load.  Loading the script
    // alone is a no-op -- safe to include on any page that might
    // need the inspector later.
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.trajectoryInspector = {
        mount: mountInspector,
    };

})(typeof window !== "undefined" ? window : this);
