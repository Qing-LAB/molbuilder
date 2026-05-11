/* SIESTA live viewer -- 3Dmol viewport + Plotly traces.
 *
 * Frames are loaded once into a 3Dmol "movie" model (addModelsAsFrames)
 * and the slider / playback simply calls viewer.setFrame(idx), which is
 * fast (no DOM rebuild).  When the server reports a new mtime we rebuild
 * the model with the fresh frame list.
 *
 * Polling cadence: ~15 s (CG steps from SIESTA on a real workload take
 * far longer than that, so the network traffic is negligible).
 */

(function () {
    "use strict";

    const POLL_MS = 15000;

    const $ = (id) => document.getElementById(id);

    /* ------------------------------------------------------------------ */
    /*  State                                                              */
    /* ------------------------------------------------------------------ */

    const state = {
        mtime: null,
        data: null,            // {frames, lattice, iterations, energies, max_forces, forces}
        currentFrame: 0,
        pollTimer: null,
        playTimer: null,
        firstFit: true,        // call viewer.zoomTo() once after first load
        // Separate buckets so toggling one overlay doesn't clobber another.
        cellShapes:  [],
        forceShapes: [],
        indexLabels: [],
        // Inspect tab: up to two picked atoms (0-based indices into
        // frame[currentFrame]).  Halo shapes are tracked separately
        // from cell / force / index overlays so toggling any one of
        // those doesn't clobber the picks.
        pickedAtoms: [],
        pickShapes:  [],
    };

    const viewer = $3Dmol.createViewer("viewer", {
        backgroundColor: "white",
        defaultcolors: $3Dmol.elementColors.Jmol,
    });

    /* ------------------------------------------------------------------ */
    /*  Status banner                                                      */
    /* ------------------------------------------------------------------ */

    function setStatus(msg, kind) {
        const el = $("status");
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
    function styleSpec() {
        return molbuilder.style.spec({
            rep:         $("rep").value,
            scale:       parseFloat($("radius").value),
            colorscheme: $("colorscheme").value,
        });
    }

    function applyStyle() {
        viewer.setStyle({}, styleSpec());
        viewer.render();
    }

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
        const bg = ($("bg") || {}).value || "white";
        if (bg === "black" || bg === "0x000000" || bg === "#000000") {
            return 0xcccccc;
        }
        if (bg === "white" || bg === "0xeeeeee" || bg === "#eeeeee") {
            return 0x444444;
        }
        return 0x888888;       // unknown bg -- safe middle ground
    }

    function drawCell() {
        // Remove only the cell shapes -- leave force arrows alone.
        for (const s of state.cellShapes) viewer.removeShape(s);
        state.cellShapes = [];

        if (!$("show-cell").checked) {
            viewer.render();
            return;
        }
        const lat = state.data && state.data.lattice;
        if (!lat || lat.length !== 3) {
            viewer.render();
            return;
        }

        const [a, b, c] = lat;
        const corner = (i, j, k) => ({
            x: i * a[0] + j * b[0] + k * c[0],
            y: i * a[1] + j * b[1] + k * c[1],
            z: i * a[2] + j * b[2] + k * c[2],
        });

        const edges = [
            // edges along a
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 0, 1], [1, 0, 1]],
            [[0, 1, 1], [1, 1, 1]],
            // edges along b
            [[0, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0]],
            [[0, 0, 1], [0, 1, 1]],
            [[1, 0, 1], [1, 1, 1]],
            // edges along c
            [[0, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [1, 0, 1]],
            [[0, 1, 0], [0, 1, 1]],
            [[1, 1, 0], [1, 1, 1]],
        ];

        const lineColor = cellLineColor();
        for (const [u, v] of edges) {
            const s = viewer.addCylinder({
                start:  corner(u[0], u[1], u[2]),
                end:    corner(v[0], v[1], v[2]),
                radius: 0.04,
                color:  lineColor,
                fromCap: 1,
                toCap:   1,
            });
            state.cellShapes.push(s);
        }
        viewer.render();
    }

    /* ------------------------------------------------------------------ */
    /*  Atom-index labels                                                  */
    /* ------------------------------------------------------------------ */

    function drawIndices() {
        for (const l of state.indexLabels) viewer.removeLabel(l);
        state.indexLabels = [];

        if (!$("show-indices").checked) {
            viewer.render();
            return;
        }
        if (!state.data) return;

        const frame = state.data.frames[state.currentFrame];
        if (!frame) return;

        for (let i = 0; i < frame.length; i++) {
            const a = frame[i];
            const lbl = viewer.addLabel(String(i + 1), {
                position:          { x: a[1], y: a[2], z: a[3] },
                backgroundColor:   "black",
                backgroundOpacity: 0.55,
                fontColor:         "white",
                fontSize:          9,
                inFront:           true,
                showBackground:    true,
                alignment:         "centerCenter",
            });
            state.indexLabels.push(lbl);
        }
        viewer.render();
    }

    /* ------------------------------------------------------------------ */
    /*  Force-vector arrows                                                */
    /* ------------------------------------------------------------------ */

    /* `viewer.addArrow` has historically been unreliable across 3Dmol
     * releases (sometimes it silently fails to render).  Instead we
     * assemble each arrow from primitives that we know work in this
     * app (the unit-cell box uses the same `addCylinder` call):
     *   - one cylinder for the shaft;
     *   - a stack of CONE_SEGS cylinders with linearly decreasing
     *     radius, which approximates a true cone for the arrowhead.
     * With ~6 segments the staircase is invisible at typical zoom. */
    const HEAD_FRAC = 0.30;     // last 30% of the arrow length is the head
    const CONE_SEGS = 6;        // radial slices in the cone (more = smoother)
    function setForcesStatus(msg) {
        // Single point of truth for the diagnostic readout next to
        // the Show-force-vectors checkbox.  Empty string clears the
        // line so the toggle area stays compact when there's
        // nothing to report.
        const el = $("forces-status");
        if (el) el.textContent = msg || "";
    }

    function drawForces() {
        for (const s of state.forceShapes) viewer.removeShape(s);
        state.forceShapes = [];

        if (!$("show-forces").checked) {
            setForcesStatus("Off — tick to overlay arrows.");
            viewer.render();
            return;
        }
        if (!state.data) {
            setForcesStatus("No trajectory loaded yet.");
            viewer.render();
            return;
        }

        const frame  = state.data.frames[state.currentFrame];
        const forces = state.data.forces && state.data.forces[state.currentFrame];
        if (!frame || !forces || !forces.length) {
            // The parser couldn't extract forces for this step.
            // Typical for geomeTRIC .xyz trajectories (which carry
            // no forces at all) and for an in-flight CG step that
            // hasn't written its force block yet.
            setForcesStatus(
                "No force data on this frame (parser did not capture it)."
            );
            console.info(
                "[viewer] no per-atom forces for frame",
                state.currentFrame,
                "(forces array length =", (forces || []).length, ")"
            );
            viewer.render();
            return;
        }

        const fscale    = parseFloat($("force-scale").value) || 1.0;
        const fmin      = parseFloat($("force-min").value)   || 0.0;
        const highlight = $("highlight-max").checked;

        const mags = forces.map(([fx, fy, fz]) =>
            Math.sqrt(fx * fx + fy * fy + fz * fz));

        let maxMag = 0, maxIdx = -1;
        for (let i = 0; i < mags.length; i++) {
            if (mags[i] > maxMag) { maxMag = mags[i]; maxIdx = i; }
        }

        let drawn = 0;

        for (let i = 0; i < frame.length && i < forces.length; i++) {
            const a   = frame[i];
            const f   = forces[i];
            const mag = mags[i];
            if (mag < fmin) continue;

            // Colour:
            //   - the largest force (highlight) is gold so it pops;
            //   - others go from dim red to bright orange-red by magnitude.
            let color;
            if (highlight && i === maxIdx) {
                color = 0xffc400;          // gold
            } else {
                const t = maxMag > 0 ? mag / maxMag : 0;
                const r = Math.floor(170 + 85 * t);
                const g = Math.floor( 40 + 60 * t);
                color = (r << 16) | (g << 8) | 0x20;
            }

            const sx = a[1], sy = a[2], sz = a[3];
            const ex = sx + f[0] * fscale;
            const ey = sy + f[1] * fscale;
            const ez = sz + f[2] * fscale;
            // Linear interpolation along the arrow axis.
            const lerp = (t) => ({
                x: sx + (ex - sx) * t,
                y: sy + (ey - sy) * t,
                z: sz + (ez - sz) * t,
            });

            const baseR = 0.05 + 0.04 * (maxMag > 0 ? mag / maxMag : 0);
            const headR = baseR * 2.6;
            const tShaft = 1 - HEAD_FRAC;

            // --- shaft ---
            state.forceShapes.push(viewer.addCylinder({
                start:   { x: sx, y: sy, z: sz },
                end:     lerp(tShaft),
                radius:  baseR,
                color:   color,
                fromCap: 1, toCap: 1,
            }));
            // --- arrowhead: stack of N cylinders tapering to zero ---
            // Segment k spans t1->t2 with radius taken at the midpoint of
            // the linear taper (headR at the base, 0 at the tip).
            for (let k = 0; k < CONE_SEGS; k++) {
                const t1   = tShaft + HEAD_FRAC * (k    ) / CONE_SEGS;
                const t2   = tShaft + HEAD_FRAC * (k + 1) / CONE_SEGS;
                const tmid = (k + 0.5) / CONE_SEGS;
                const r    = headR * (1 - tmid);
                if (r < 0.005) continue;       // skip the vanishing tip
                state.forceShapes.push(viewer.addCylinder({
                    start:   lerp(t1),
                    end:     lerp(t2),
                    radius:  r,
                    color:   color,
                    fromCap: 1, toCap: 1,
                }));
            }
            drawn++;
        }

        console.info(
            "[viewer] drew", drawn, "force arrows on frame",
            state.currentFrame,
            "(maxMag =", maxMag.toFixed(3), "eV/\u00C5 at atom", maxIdx + 1, ")"
        );
        // Surface the same information to the user-visible readout
        // next to the toggle.  When `drawn === 0` and forces are
        // present, fmin is too tight -- tell the user explicitly so
        // they can lower it.
        if (drawn === 0) {
            setForcesStatus(
                "0 arrows shown (all |F| < threshold " +
                fmin.toFixed(3) + " eV/\u00C5; max |F| = " +
                maxMag.toFixed(3) + ")."
            );
        } else {
            setForcesStatus(
                "Showing " + drawn + " arrow" + (drawn === 1 ? "" : "s") +
                " (max |F| = " + maxMag.toFixed(3) + " eV/\u00C5)."
            );
        }
        viewer.render();
    }

    /* ------------------------------------------------------------------ */
    /*  Inspect tab: two-atom picking + live distance                      */
    /* ------------------------------------------------------------------ */

    // Selection halo shared with the Modify tab via static/lib/mol-pick.js
    // (one source of truth for the colour / radius / shape).
    const pick = (window.molbuilder && window.molbuilder.pick) || null;

    function clearPickHighlights() {
        if (pick) pick.clearHalos(viewer, state.pickShapes);
        else state.pickShapes.length = 0;
    }

    function renderPickHighlights() {
        clearPickHighlights();
        if (!state.data || !state.data.frames.length) {
            viewer.render();
            return;
        }
        const frame = state.data.frames[state.currentFrame] || [];
        for (const idx of state.pickedAtoms) {
            const row = frame[idx];
            if (!row || !pick) continue;
            state.pickShapes.push(
                pick.addHalo(viewer, {x: row[1], y: row[2], z: row[3]})
            );
        }
        viewer.render();
    }

    function updateInspectPanel() {
        const pa = state.pickedAtoms;
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
    // pick drops the oldest).  Shared between the viewer click hook
    // and the atom-list row click.
    function togglePick(idx) {
        const pa = state.pickedAtoms;
        const existing = pa.indexOf(idx);
        if (existing !== -1) {
            pa.splice(existing, 1);
        } else {
            if (pa.length >= 2) pa.shift();
            pa.push(idx);
        }
        renderPickHighlights();
        updateInspectPanel();
        refreshAtomListHighlights();
    }

    // 3Dmol click callback.  ``atom.serial`` is 0-based for XYZ-
    // loaded models (every Watch trajectory goes through
    // viewer.addModelsAsFrames(framesToMultiXyz(...), "xyz")).
    function onWatchAtomClick(atom) {
        togglePick(Number(atom.serial));
    }

    function clearAtomPicks() {
        state.pickedAtoms = [];
        renderPickHighlights();
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
            // dataset.atomIndex stays 0-based (matches state.pickedAtoms
            // and 3Dmol's atom.serial); the displayed ``#`` column is
            // 1-based to match the overlay labels in the Overlays tab.
            tr.dataset.atomIndex = String(i);
            tr.innerHTML =
                '<td class="col-idx">' + (i + 1) + '</td>' +
                '<td class="col-el">'  + (r[0] || "?") + '</td>' +
                '<td class="col-coord">' + r[1].toFixed(2) + '</td>' +
                '<td class="col-coord">' + r[2].toFixed(2) + '</td>' +
                '<td class="col-coord">' + r[3].toFixed(2) + '</td>';
            tr.addEventListener("click", () => togglePick(i));
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
        const picked = new Set(state.pickedAtoms);
        for (const tr of tbody.children) {
            tr.classList.toggle(
                "is-selected",
                picked.has(Number(tr.dataset.atomIndex)),
            );
        }
    }

    function rebuildModel() {
        viewer.removeAllModels();
        if (!state.data || !state.data.frames.length) {
            viewer.render();
            return;
        }
        viewer.addModelsAsFrames(framesToMultiXyz(state.data.frames), "xyz");
        applyStyle();
        // Wire the per-atom click hook AFTER applyStyle: 3Dmol's
        // setStyle() rebuilds the per-atom render objects, which on
        // movie-mode models (addModelsAsFrames) drops the clickable
        // flag installed by an earlier setClickable.  Re-installing
        // it last keeps clicks alive across rep / radius /
        // colour-scheme changes.
        viewer.setClickable({}, true, onWatchAtomClick);
        drawCell();
        // Populate the Inspect-tab atom list now that the model is
        // loaded; the list mirrors the per-frame coordinates and is
        // the keyboard-friendly path to selection.
        rebuildInspectAtomList();
        if (state.firstFit) {
            viewer.zoomTo();
            state.firstFit = false;
        }
    }

    // Also re-install setClickable whenever applyStyle re-runs (rep /
    // radius / colour-scheme dropdown changes call applyStyle without
    // a full rebuildModel; without this the click handler silently
    // dies after the first dropdown change).
    function applyStyleAndRewireClicks() {
        applyStyle();
        if (state.data && state.data.frames && state.data.frames.length) {
            viewer.setClickable({}, true, onWatchAtomClick);
        }
    }

    function showFrame(idx) {
        if (!state.data || !state.data.frames.length) return;
        const n = state.data.frames.length;
        idx = Math.max(0, Math.min(n - 1, idx));
        state.currentFrame = idx;
        viewer.setFrame(idx);
        // Labels and force arrows are not animated by 3Dmol's frame system,
        // so we redraw them whenever the active frame changes.  Each of
        // these calls also issues viewer.render(), so no extra render here.
        drawIndices();
        drawForces();
        // Picked-atom halos sit on absolute coordinates and 3Dmol
        // doesn't animate Shape objects with setFrame, so re-render
        // them at every frame change.  Cheap (max 2 spheres).
        renderPickHighlights();
        updateInspectPanel();
        // Refresh the per-row coordinates in the atom list so the
        // user sees per-frame xyz drift.  Cheap: rewrites 3 text
        // nodes per atom, no DOM reshape, no listener churn.
        updateAtomListCoords();
        $("frame-idx").textContent = idx;
        $("frame-slider").value = idx;
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
        try {
            const url = state.mtime !== null
                ? "/api/watch/data?mtime=" + encodeURIComponent(state.mtime)
                : "/api/watch/data";
            const r = await fetch(url).then(x => x.json());
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

    function applyNewData(r) {
        const wasAtEnd = !state.data
            || state.currentFrame >= state.data.frames.length - 1;

        state.mtime  = r.mtime;
        state.data   = r.data;
        state.format = r.format || (r.data && r.data.source_format) || "?";
        state.label  = r.label  || state.format;

        const n = state.data.frames.length;
        if (n === 0) {
            setStatus("File loaded ("+ state.label +") but no frames yet.", "");
            return;
        }
        $("frame-tot").textContent = n - 1;
        $("frame-slider").max      = n - 1;
        // Frames now exist -> enable the "Save current frame as XYZ"
        // button (it's disabled-by-default in the template so users
        // can't click it before any data is loaded).
        $("save-frame").disabled = false;

        rebuildModel();
        const targetIdx = wasAtEnd ? n - 1 : Math.min(state.currentFrame, n - 1);
        showFrame(targetIdx);
        makePlots();

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
            } else if (runState === "error") {
                badge.classList.add("run-state-error");
                badgeLab.textContent = "Error";
                badgeDet.textContent = joinParts(
                    errMsg,
                    lastResultTs ? "stopped " + lastResultTs : "",
                    elapsedTxt ? "total " + elapsedTxt : "",
                );
            } else {
                badge.classList.add("run-state-ongoing");
                badgeLab.textContent = "Ongoing";
                badgeDet.textContent = joinParts(
                    lastResultTs ? "last result " + lastResultTs : "",
                    elapsedTxt ? "sim time " + elapsedTxt : "",
                );
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

    function step(delta) {
        if (!state.data || !state.data.frames.length) return;
        const n = state.data.frames.length;
        let next = state.currentFrame + delta;
        if (next >= n) next = $("loop").checked ? 0     : n - 1;
        if (next < 0)  next = $("loop").checked ? n - 1 : 0;
        showFrame(next);
    }

    function play() {
        if (state.playTimer) clearInterval(state.playTimer);
        const speed = parseInt($("speed").value, 10) || 150;
        state.playTimer = setInterval(() => step(1), speed);
    }

    function pause() {
        if (state.playTimer) {
            clearInterval(state.playTimer);
            state.playTimer = null;
        }
    }

    /* ------------------------------------------------------------------ */
    /*  UI wiring                                                          */
    /* ------------------------------------------------------------------ */

    /*  Load button has two behaviours:
     *    - path field has text -> POST {path:...} as JSON  (live watch)
     *    - path field empty    -> open the hidden file picker;
     *                              the picker's change handler uploads
     *                              the file as multipart and the server
     *                              parses it once (no polling).
     */
    $("load-btn").addEventListener("click", () => {
        const path = $("path-input").value.trim();
        if (path) {
            loadByPath(path);
        } else {
            $("file-picker").click();   // open the native dialog
        }
    });

    $("file-picker").addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        pause();
        clearAtomPicks();
        setStatus("Uploading " + file.name + "\u2026", "");
        const fd = new FormData();
        fd.append("file", file);
        try {
            const r = await fetch("/api/watch/load", { method: "POST", body: fd })
                            .then(x => x.json());
            if (!r.ok) {
                setStatus(r.error || "Upload failed.", "error");
                return;
            }
            // Reflect the upload in the path field so it's clear what
            // was loaded; prefix tells the user it isn't being polled.
            $("path-input").value = "(uploaded) " + file.name;
            state.firstFit = true;
            applyNewData({
                mtime:  r.mtime,
                data:   r.data,
                format: r.format,
                label:  r.label,
            });
            // Uploaded files are one-shot; skip the polling timer
            // because the temp file's mtime never advances.
            if (r.uploaded) {
                stopPolling();
            } else {
                startPolling();
            }
        } catch (err) {
            setStatus("Network error: " + err.message, "error");
        } finally {
            // Reset so picking the same file again still fires "change".
            e.target.value = "";
        }
    });

    async function loadByPath(path) {
        pause();
        // Drop atom picks: a fresh load means a new trajectory and
        // the picked indices may not exist in the new model.
        clearAtomPicks();
        setStatus("Loading\u2026", "");
        try {
            const r = await fetch("/api/watch/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: path }),
            }).then(x => x.json());

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
                $("path-input").value = r.path;
            }
            startPolling();
        } catch (e) {
            setStatus("Network error: " + e.message, "error");
        }
    }

    $("path-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") $("load-btn").click();
    });

    // Honour the "Watch this run" handoff from the Build page:
    // /watch?path=<system_label>.molwatch.log pre-fills the input
    // with the predicted log filename.  We do NOT auto-trigger Load
    // because the user typically still needs to prepend the absolute
    // path of the directory where they'll run the calculation -- the
    // browser-side server has no way to know that.  Surface a hint
    // in the status banner so they know what to do next.
    (function applyHandoff() {
        const params = new URLSearchParams(window.location.search);
        const path = params.get("path");
        if (!path) return;
        $("path-input").value = path;
        setStatus(
            `Path pre-filled from Build (${path}).  Edit to add the absolute ` +
            `directory you'll run in, then press Enter or click Load.`,
            "warn",
        );
        // Move focus into the input and select the filename so the
        // user can type the prefix without manual selection.
        $("path-input").focus();
        try { $("path-input").setSelectionRange(0, 0); } catch (e) { /* ok */ }
    })();

    $("frame-slider").addEventListener("input", (e) => {
        showFrame(parseInt(e.target.value, 10));
    });

    $("rep").addEventListener("change", applyStyleAndRewireClicks);
    $("radius").addEventListener("input", applyStyleAndRewireClicks);
    $("colorscheme").addEventListener("change", applyStyleAndRewireClicks);
    $("show-cell").addEventListener("change", drawCell);
    $("bg").addEventListener("change", (e) => {
        viewer.setBackgroundColor(e.target.value);
        // Cell line colour is bg-contrast-driven; redraw with the
        // new colour so the box stays visible on the new background.
        drawCell();
    });

    /* Overlays */
    $("show-indices").addEventListener("change", drawIndices);
    $("show-forces").addEventListener("change",  drawForces);
    $("force-scale").addEventListener("input", (e) => {
        $("force-scale-val").textContent = parseFloat(e.target.value).toFixed(1);
        drawForces();
    });
    $("force-min").addEventListener("input",      drawForces);
    $("highlight-max").addEventListener("change", drawForces);

    $("play").addEventListener("click",  play);
    $("pause").addEventListener("click", pause);
    $("prev").addEventListener("click",  () => step(-1));
    $("next").addEventListener("click",  () => step(1));
    $("speed").addEventListener("change", () => {
        if (state.playTimer) play();    // restart with new cadence
    });

    /* ---- Inspect-tab: Clear-picks button ------------------------- */
    const inspectClearBtn = $("inspect-clear");
    if (inspectClearBtn) {
        inspectClearBtn.addEventListener("click", clearAtomPicks);
    }

    /* ---- Save current frame as XYZ ------------------------------- */
    /* Hands the displayed structure off to the next step in the
       user's pipeline -- e.g., dropping the relaxed molecule into a
       tunneling-gap setup as a bridge.  The output is plain XYZ
       (Angstrom), which any chemistry tool reads. */
    $("save-frame").addEventListener("click", () => {
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
       toggle via `is-active`; we just sync the classes here. */
    document.querySelectorAll(".ctab").forEach((btn) => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            document.querySelectorAll(".ctab").forEach(
                (b) => b.classList.toggle("is-active", b === btn)
            );
            document.querySelectorAll(".ctab-panel").forEach(
                (p) => p.classList.toggle(
                    "is-active", p.dataset.panel === target
                )
            );
        });
    });

    /* ---- Re-fit 3Dmol on viewport resize ------------------------- */
    /* The viewer height is `clamp(360px, 52vh, 500px)` -- it changes
       when the user resizes the window or rotates a tablet.  3Dmol
       caches canvas size at init, so we have to nudge it to refresh
       its WebGL viewport whenever the container size actually changes;
       debounce with rAF to avoid storming during drag. */
    let _resizeRAF = 0;
    const _onResize = () => {
        cancelAnimationFrame(_resizeRAF);
        _resizeRAF = requestAnimationFrame(() => {
            if (viewer && typeof viewer.resize === "function") {
                viewer.resize();
                viewer.render();
            }
        });
    };
    window.addEventListener("resize", _onResize, { passive: true });

    // ----- Persist file path across Build ↔ Watch navigation -----
    // Navigating to /build and back is a full page reload; save the
    // path-input value so the user doesn't have to retype the path.
    const WATCH_PATH_KEY = "watch-path";
    const pathEl = $("path-input");
    const savedPath = sessionStorage.getItem(WATCH_PATH_KEY);
    if (savedPath && !pathEl.value) {
        pathEl.value = savedPath;
    }
    pathEl.addEventListener("input", () => {
        sessionStorage.setItem(WATCH_PATH_KEY, pathEl.value);
    });
    window.addEventListener("pagehide", () => {
        sessionStorage.setItem(WATCH_PATH_KEY, pathEl.value);
    });
})();
