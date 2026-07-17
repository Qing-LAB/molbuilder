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
    /**
     * Redact the user-identifying prefix from an absolute path so
     * the CSV export header carries enough structure for scientific
     * provenance ("the file lived under projects/BDT/optimization/")
     * without leaking the OS-level username.  Module-level so the
     * IIFE-level export at the bottom of this file can reach it.
     *
     * Patterns redacted:
     *   /home/<user>/...                  -> ~/...
     *   /Users/<user>/...                 -> ~/...        (macOS)
     *   /tmp/pytest-of-<user>/...         -> <tmp>/...
     *   C:\Users\<user>\...               -> ~\...        (Windows)
     *
     * Everything past the user-named segment is preserved verbatim
     * so the reader can still tell where in the user's tree the
     * file lived (project layout / staged-relaxation structure) --
     * only the username segment is replaced with ``~`` or ``<tmp>``.
     */
    function _redactSourcePath(p) {
        if (typeof p !== "string" || !p) return p;
        // POSIX home (Linux + macOS).  ``/home/qqing/foo`` ->
        // ``~/foo``; ``/Users/qqing/foo`` -> ``~/foo``.  The
        // ``[^/]+`` segment is the username.
        p = p.replace(/^\/(home|Users)\/[^/]+\//, "~/");
        // POSIX tmp dirs with embedded username.  pytest writes
        // ``/tmp/pytest-of-qqing/...``; the username-bearing first
        // segment under /tmp is replaced with ``<tmp>``.
        p = p.replace(/^\/tmp\/[^/]*-of-[^/]+\//, "<tmp>/");
        // Windows home.  Case-insensitive on the drive letter +
        // ``Users``; the username segment is everything between
        // ``Users\`` and the next backslash.
        p = p.replace(/^[A-Za-z]:\\Users\\[^\\]+\\/i, "~\\");
        return p;
    }

    function mountInspector(rootEl, opts) {
    opts = opts || {};

    // Poll cadence: 15 s.
    //
    // History: 15 s -> 60 s on 2026-06-14 (commit 6da01ce) to silence
    // ``SSL: UNEXPECTED_EOF_WHILE_READING`` noise from overlapping
    // ticks aborting in-flight TLS connections.  The G4 fix paired
    // that bump with an in-flight guard in ``pollOnce`` (overlapping
    // ticks now SKIP instead of aborting).
    //
    // 2026-06-17 Fix B: drop back to 15 s.  60 s was overcorrection --
    // the in-flight guard alone is sufficient to prevent the SSL
    // noise (the TLS teardown was the symptom, ``abort()`` mid-flight
    // was the cause, and that's gone).  Users watching a live SIESTA
    // SCF run see iterations complete in 13-24 s on the standard
    // workstation; 60 s polling means 1-4 iterations slip between
    // polls and the plot lags the on-disk state.  15 s restores the
    // original responsiveness without re-introducing the SSL bug.
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

    // Bucketed state shape per docs/protocols/results-state-contract.md.
    // The contract partitions inspector state into five disjoint
    // buckets with explicit reset semantics:
    //
    //   fileState  -- replaced atomically each LOADING -> LOADED
    //                 (path, mtime, format, label, parsed data)
    //   viewState  -- per-file user interaction (currentFrame, firstFit, picks)
    //   uiPrefs    -- per-session knobs (hideFrozen etc.)
    //   lifecycle  -- controllers + timers (poll timer, abort controllers, fetchSeq)
    //   derived    -- recomputed from fileState (scfPollHistory)
    //
    // Backward-compat aliases at the end of this block keep the
    // existing ~3000 lines of render code working with the legacy
    // flat ``state.X`` shape while new code reads/writes through
    // the bucketed canonical home.  The legacy aliases are simple
    // getter/setter properties -- no behavior change for callers,
    // only the canonical storage moved.
    //
    // The state machine field is the contract's enforcement point:
    // bucket mutations OUTSIDE transition() are forbidden for
    // fileState / lifecycle / derived (matrix § 3); viewState and
    // uiPrefs CAN be mutated by event handlers (frame scrub,
    // hide-frozen toggle, etc.).
    const state = {
        // Per contract § 2: IDLE / LOADING / LOADED / WATCHING / ERROR.
        machine: "IDLE",

        fileState: {
            path:   null,
            mtime:  null,
            format: null,
            label:  null,
            // data shape: {frames, lattice, iterations, energies,
            //   max_forces, max_forces_constrained, forces,
            //   scf_history, wall_times, in_progress, run_state,
            //   error_message, runtime_info, parse_warnings, ...}
            data:   null,
        },

        viewState: {
            currentFrame: 0,
            firstFit:     true,
        },

        // uiPrefs: contract § 3 reserves this bucket for per-session
        // knobs (sessionStorage-persisted, survives file-switch).
        // Trajectory has NO fields here today: hide-frozen is owned
        // by mol-viewer-embed's own sessionStorage with its own key
        // (mol.viewer.hideFrozen), playback speed/loop live in the
        // embed.  When/if a trajectory-only pref appears (e.g. a
        // plot tab selection that isn't a viewer concern), populate
        // this bucket and wire the sessionStorage roundtrip
        // (key: molbuilder.results.trajectory.uiPrefs.v1).  Keeping
        // the bucket present-but-empty matches the five-bucket
        // contract shape so tests can pin it; the alternative
        // (omit the bucket) would force inspector-specific test
        // matrices.
        uiPrefs: {},

        lifecycle: {
            pollTimer:    null,
            pollInFlight: false,
            loadAbort:    null,
            pollAbort:    null,
            // File-identity guard (contract § 4 Invariant 1).
            // Every fetch resolution checks (response.path,
            // current fetchSeq) before applying.  Late responses
            // from a prior file can never write into the current
            // file's view.
            fetchSeq:     0,
            // 2-consecutive-ticks WATCHING -> LOADED buffer
            // (contract § 2, § 3 matrix row "poll says run
            // finished (2 consecutive ticks)").  A single tick
            // can lie -- the server may see `run_state=finished`
            // while the parser is still flushing trailing data.
            // Incremented in pollOnce when r.data.run_state ===
            // "finished"; reset to 0 on LOADING / WATCHING tick
            // with ongoing state.
            finishedTicks: 0,
        },

        derived: {
            // SCF wall-time progression tracker.  Each entry:
            //   { ts: Date.now() ms, totalIters: int }
            // The SCF status-line builder consumes this to compute a
            // rolling per-iter wall-time estimate.  Per contract § 3
            // reset matrix, derived is cleared on every transition
            // to LOADING (file-switch / Refresh) -- so the buffer
            // never carries stale samples from a prior file.
            scfPollHistory: [],
        },
    };

    // Backward-compat aliases.  Render code throughout this file
    // still reads/writes the legacy flat shape (state.mtime,
    // state.data, state.currentFrame, state.firstFit, ...).  The
    // aliases route through to the bucketed canonical home so the
    // entire body of render code keeps working unchanged; the
    // contract's reset matrix is enforced at transition() and at
    // the four documented external entry-points (loadByPath,
    // Refresh, pollOnce result, dispose).
    //
    // ESLint warns on ``Object.defineProperty(state, ...)`` from
    // inside a closure; we squelch by setting properties first then
    // overwriting with defineProperty.  Same pattern as the
    // workspace dispatcher's compat shims.
    (function _wireBackcompatAliases() {
        function alias(key, bucket) {
            Object.defineProperty(state, key, {
                get: function ()    { return state[bucket][key]; },
                set: function (v)   { state[bucket][key] = v; },
                enumerable: true,
                configurable: true,
            });
        }
        alias("path",         "fileState");
        alias("mtime",        "fileState");
        alias("format",       "fileState");
        alias("label",        "fileState");
        alias("data",         "fileState");
        alias("currentFrame", "viewState");
        alias("firstFit",     "viewState");
        alias("pollTimer",    "lifecycle");
        alias("pollInFlight", "lifecycle");
        alias("loadAbort",    "lifecycle");
        alias("pollAbort",    "lifecycle");
        alias("scfPollHistory", "derived");
    })();

    // Transition orchestrator (contract § 2).  ALL state changes
    // that involve resetting fileState / lifecycle / derived go
    // through here.  Direct mutation of viewState / uiPrefs from
    // event handlers is allowed (frame scrub, hide-frozen toggle).
    //
    // payload shape:
    //   target = 'LOADING'                 -> { path: string }
    //   target = 'LOADED' | 'WATCHING'     -> { data: payload }  (driven by transition itself)
    //   target = 'ERROR'                   -> { message: string }
    //   target = 'IDLE'                    -> {} (dispose path)
    //
    // Reset matrix (contract § 3) enforced here:
    //   -> LOADING: empty fileState, reset viewState (currentFrame=0,
    //               firstFit=true), keep uiPrefs, abort + clear all
    //               in-flight controllers, clear derived, bump fetchSeq.
    //   -> IDLE:    clear fileState, viewState, derived; abort + clear
    //               all controllers; stop poll timer.
    function transition(target, payload) {
        payload = payload || {};
        if (target === "LOADING") {
            // Abort all in-flight requests.  Their .then() handlers
            // still fire, but the file-identity guard (Invariant 1)
            // catches them via the bumped fetchSeq.
            if (state.lifecycle.loadAbort) {
                try { state.lifecycle.loadAbort.abort(); } catch (_) {}
                state.lifecycle.loadAbort = null;
            }
            if (state.lifecycle.pollAbort) {
                try { state.lifecycle.pollAbort.abort(); } catch (_) {}
                state.lifecycle.pollAbort = null;
            }
            state.lifecycle.pollInFlight = false;
            // Stop the poll timer; LOADED/WATCHING will restart it
            // if the run is ongoing.
            stopPolling();
            // Empty fileState.  The new path is the only thing the
            // caller cares about; data populates when the fetch
            // resolves and applyNewData runs under the file-identity
            // guard.
            state.fileState.path   = payload.path || null;
            state.fileState.mtime  = null;
            state.fileState.format = null;
            state.fileState.label  = null;
            state.fileState.data   = null;
            // Reset viewState per matrix: playhead to 0, refit camera
            // on next render.
            state.viewState.currentFrame = 0;
            state.viewState.firstFit     = true;
            // Clear derived caches.
            state.derived.scfPollHistory.length = 0;
            // Reset the 2-tick WATCHING -> LOADED buffer counter.
            // A fresh load is a new ground truth; any stale
            // finishedTicks from a prior file/run must not carry
            // over.  (Pre-PR-2.2: this reset lived in
            // transition('WATCHING') which had the side effect of
            // wiping a just-incremented count when _settlePostLoad
            // transitioned LOADING -> WATCHING with a finished
            // run.  Now the reset is here, where it belongs.)
            state.lifecycle.finishedTicks = 0;
            // Bump fetchSeq AFTER aborts so the new fetch carries a
            // fresh sequence number that no in-flight response can
            // match.
            state.lifecycle.fetchSeq++;
            state.machine = "LOADING";
            return;
        }
        if (target === "IDLE") {
            if (state.lifecycle.loadAbort) {
                try { state.lifecycle.loadAbort.abort(); } catch (_) {}
                state.lifecycle.loadAbort = null;
            }
            if (state.lifecycle.pollAbort) {
                try { state.lifecycle.pollAbort.abort(); } catch (_) {}
                state.lifecycle.pollAbort = null;
            }
            state.lifecycle.pollInFlight = false;
            stopPolling();
            state.fileState.path   = null;
            state.fileState.mtime  = null;
            state.fileState.format = null;
            state.fileState.label  = null;
            state.fileState.data   = null;
            state.viewState.currentFrame = 0;
            state.viewState.firstFit     = true;
            state.derived.scfPollHistory.length = 0;
            state.machine = "IDLE";
            return;
        }
        if (target === "LOADED") {
            // Run is finished (run_state="finished") OR static
            // file (no run_state).  Side effects per contract § 3
            // matrix row "fetch resolved, run finished": stop poll
            // timer if running; clear pollInFlight; reset
            // finishedTicks (we're already in LOADED).
            stopPolling();
            state.lifecycle.pollInFlight = false;
            state.lifecycle.finishedTicks = 0;
            state.machine = "LOADED";
            return;
        }
        if (target === "WATCHING") {
            // Run is ongoing.  Side effects per contract § 3
            // matrix row "fetch resolved, run ongoing": START poll
            // timer.  startPolling() is idempotent so re-entering
            // WATCHING on a mid-poll tick is a no-op for the timer.
            //
            // NOTE: do NOT reset finishedTicks here.  _settlePostLoad
            // owns the counter: it increments on a "finished" tick
            // BEFORE transitioning to WATCHING (if not yet ready to
            // flip to LOADED).  Resetting here would wipe the
            // just-incremented value and effectively turn the
            // 2-tick buffer into a 3-tick one.  The "ongoing tick
            // breaks the consecutive count" reset lives in
            // _settlePostLoad's ongoing branch instead.
            startPolling();
            state.machine = "WATCHING";
            return;
        }
        if (target === "ERROR") {
            // Side effects per contract § 3 row "fetch failed N
            // times": stop poll timer.  Keeps last-good fileState
            // in place so the user still sees what they had.
            stopPolling();
            state.lifecycle.pollInFlight = false;
            state.machine = "ERROR";
            return;
        }
        if (target === "APPLY") {
            // PR 2.3: close deferred gap #1.  applyNewData no longer
            // writes fileState fields directly via the backward-
            // compat aliases; it routes here.  Single canonical
            // writer for fileState matches the contract § 2
            // forbidden list ("Direct mutation of fileState outside
            // a → LOADING → LOADED/WATCHING arc").
            //
            // Two callers (both in applyNewData):
            //   1. noNewContent branch -- payload carries only
            //      {mtime, data} (format/label/path stay because the
            //      file identity didn't change).  Contract § 3 matrix
            //      row "watch tick (new SCF iter same step)" /
            //      "watch tick (new frame)" → "replace .data".
            //   2. full-rebuild branch -- payload carries all five
            //      fields.  This is the LOADING → LOADED/WATCHING
            //      arc.
            //
            // Either way the writes happen atomically inside this
            // single function; render code observes either the
            // pre-update or the post-update view but never a
            // half-updated state.  Machine field is set by
            // _settlePostLoad after the APPLY (it inspects the new
            // data.run_state).
            if (payload.mtime  !== undefined) state.fileState.mtime  = payload.mtime;
            if (payload.data   !== undefined) state.fileState.data   = payload.data;
            if (payload.format !== undefined) state.fileState.format = payload.format;
            if (payload.label  !== undefined) state.fileState.label  = payload.label;
            if (payload.path   !== undefined) state.fileState.path   = payload.path;
            return;
        }
        // Unknown target: silent no-op.  Future targets (the
        // contract reserves room for sub-states) land here.
    }

    // Helper for applyNewData et al: after fileState has been
    // populated, transition to the appropriate post-LOADING state
    // based on run_state.  Centralizes the "finished/ongoing"
    // decision + the 2-consecutive-ticks WATCHING -> LOADED buffer
    // (contract § 2, § 13).
    function _settlePostLoad() {
        const rs = state.fileState.data
                && state.fileState.data.run_state;
        if (rs === "errored") {
            transition("ERROR");
            return;
        }
        if (rs === "finished") {
            // 2-tick buffer: a single "finished" tick may lie if
            // the parser is still flushing trailing data.  Stay in
            // WATCHING until we've seen N consecutive finished
            // ticks; flip to LOADED on the Nth.
            state.lifecycle.finishedTicks++;
            if (state.lifecycle.finishedTicks >= 2) {
                transition("LOADED");
            } else {
                // First finished tick: stay in WATCHING (keep the
                // timer running for one more cycle).
                if (state.machine !== "WATCHING") {
                    transition("WATCHING");
                }
            }
            return;
        }
        // Ongoing (rs === "ongoing" or absent).  Reset the buffer
        // counter: a non-finished tick breaks any consecutive
        // "finished" streak.  Per contract § 13 the buffer counts
        // CONSECUTIVE ticks only.
        state.lifecycle.finishedTicks = 0;
        transition("WATCHING");
    }

    // (Contract § 4 Invariant 3 "render-with-snapshot" is NOT
    // enforced today.  In practice the render functions close over
    // ``state`` directly; applyNewData runs synchronously and calls
    // render functions before yielding to the event loop, so there
    // is no observable "stale-during-tick" race.  A snap() helper
    // existed in PR 2 but went unused and was deleted in PR 2.3 to
    // match implementation -- pulling the dead code prevents the
    // contract from being "aspirational in code, claimed in tests".
    // If Invariant 3 becomes load-bearing later, the snapshot
    // pattern lives at:
    //   docs/protocols/results-state-contract.md § 4 Invariant 3
    // -- bring it back when the conversion is committed.)

    // Per-frame in-progress filter (contract § 4 Invariant 2).
    // The parser tags partial frames with in_progress=true
    // (parse/engines/_helpers.py::trajectory_result_to_legacy_dict
    // emits a per-frame bool array; collapses
    // to [] when no frame is in-progress).  Plot trace builders
    // call plottableFrames() to omit partial frames -- the energy
    // / max_force values for those frames are placeholders (the
    // siesta parser's step_initial_etot fallback) and don't belong
    // in the user-facing plot.  The frame still ships and shows
    // in the inspect list with a "computing..." badge.
    function plottableFrames(data) {
        if (!data || !Array.isArray(data.frames)) return [];
        var inProg = (data.in_progress && data.in_progress.length)
            ? data.in_progress : null;
        var out = [];
        for (var i = 0; i < data.frames.length; i++) {
            if (inProg && inProg[i]) continue;
            out.push(i);
        }
        return out;
    }
    // Expose for tests + future render-function migration.
    state._plottableFrames = plottableFrames;                          // eslint-disable-line camelcase

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
        // Up-to-3 pick for the Inspect panel (task #299): 1 atom →
        // xyz, 2 atoms → distance, 3 atoms → bond angle with the
        // 2nd click as the vertex (the shared
        // lib/selection/measurements.js computes all three from the
        // current frame's coords).  ``halo: true`` draws the
        // yellow sphere overlay; ``label: false`` keeps the
        // viewer uncluttered (the Inspect panel shows the readout).
        // Atom-list row clicks share the embed's pick state via
        // handle.setPickedIndices (no separate halo path).
        pick:  {
            mode:  "triple",
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
        // Axes default OFF (2026-06-13).  Pre-fix every tab mounted
        // the embed with ``axes: true`` so axes always showed on
        // first mount, but a tab that persists axes-state in
        // sessionStorage (Modify) could remember "axes off" from a
        // prior visit and show that on revisit — looking
        // "unpredictable" to the user even though each tab was
        // internally consistent.  Default OFF here matches the
        // user's expectation that axes are an opt-in overlay.
        axes:    false,
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

    /**
     * 2026-06-12: Build a CSV bundling every column drawn across the
     * 4 trajectory plots.  Self-describing header (``#``-comments)
     * carries source file, parser, mtime, and the generation
     * timestamp so the file can be re-traced later without external
     * notes.
     *
     * Layout
     * ------
     * Header block (lines starting with ``#``):
     *   * generation context (timestamp, source path, parser,
     *     source mtime, n_frames)
     *   * column legend pointing at unit + meaning
     *   * a one-line schema reminder for whoever opens the file in
     *     Excel a year later
     *
     * Data block:
     *   step, energy_eV, max_force_eVperA, max_force_constrained_eVperA,
     *   scf_cycle, scf_cycle_energy_eV, scf_cycle_gnorm_eVperA
     *
     * The trailing SCF columns repeat the per-step value for the
     * cycle that's currently shown in the SCF plots (last cycle of
     * that step's SCF run) — same value the SCF gnorm/energy plots
     * graph for that point.  Empty when no SCF history.
     *
     * Numbers use repr-precision (Number.toString()) so the round-
     * trip through CSV is lossless.  Empty cells for missing values
     * (Plotly's connectgaps: false skips them visually).
     */
    function _buildPlotCsv(ctx) {
        var data       = ctx.data || {};
        var sourcePath = _redactSourcePath(
            ctx.sourcePath || "(unknown)");
        var format     = ctx.format     || "(unknown)";
        var label      = ctx.label      || format;
        var mtime      = (typeof ctx.mtime === "number")
            ? new Date(ctx.mtime * 1000).toISOString()
            : "(unknown)";
        var generated  = new Date().toISOString();

        var iterations   = data.iterations    || [];
        var energies     = data.energies      || [];
        var maxForces    = data.max_forces    || [];
        var maxForcesC   = data.max_forces_constrained || [];
        var scfHistory   = data.scf_history   || [];
        var nFrames      = (data.frames || []).length;

        function _esc(v) {
            // Empty string for missing (Plotly's skipped-gap value
            // is null).  Numbers via toString() keep IEEE precision.
            if (v === null || v === undefined) return "";
            if (typeof v === "number") {
                if (!isFinite(v)) return "";
                return v.toString();
            }
            // Quote any text containing commas / quotes / newlines.
            var s = String(v);
            if (/[",\n]/.test(s)) {
                s = '"' + s.replace(/"/g, '""') + '"';
            }
            return s;
        }

        var lines = [];
        lines.push("# molbuilder — trajectory plot data export");
        lines.push("# generated:    " + generated);
        lines.push("# source path:  " + sourcePath);
        lines.push("# parser:       " + format);
        lines.push("# label:        " + label);
        lines.push("# source mtime: " + mtime);
        lines.push("# n_frames:     " + nFrames);
        lines.push("#");
        lines.push("# Column legend:");
        lines.push("#   step                          — CG / opt step index (engine numbering)");
        lines.push("#   energy_eV                     — total energy in eV");
        lines.push("#   max_force_eVperA              — Max |F| across ALL atoms (eV/Å); informational");
        lines.push("#   max_force_constrained_eVperA  — Max |F| EXCLUDING frozen atoms (eV/Å); convergence-gating");
        lines.push("#                                   when constraints exist, otherwise empty.");
        lines.push("#   scf_cycle                     — last SCF iteration index for this step");
        lines.push("#   scf_cycle_energy_eV           — energy at that SCF cycle (eV)");
        lines.push("#   scf_cycle_gnorm_eVperA        — SCF gradient norm at that cycle (eV/Å);");
        lines.push("#                                   PySCF only; SIESTA emits dHmax instead.");
        lines.push("#");
        lines.push("# Empty cells mean the engine didn't emit that value at that step.");
        lines.push("#");
        lines.push([
            "step", "energy_eV",
            "max_force_eVperA", "max_force_constrained_eVperA",
            "scf_cycle", "scf_cycle_energy_eV", "scf_cycle_gnorm_eVperA",
        ].join(","));

        for (var i = 0; i < nFrames; i++) {
            var step = iterations[i];
            if (step === undefined) step = i;
            var scfRun = scfHistory[i] || [];
            var lastCycle = scfRun[scfRun.length - 1] || {};
            lines.push([
                _esc(step),
                _esc(energies[i]),
                _esc(maxForces[i]),
                _esc(maxForcesC[i]),
                _esc(lastCycle.cycle),
                _esc(lastCycle.energy),
                _esc(lastCycle.gnorm),
            ].join(","));
        }
        return lines.join("\n") + "\n";
    }

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

    // applyIndexLabels / drawIndices were retired 2026-06-13.  Atom-
    // index labels are now exclusively controlled by the molview knob
    // bar's Labels popover; the trajectory inspector's #show-indices
    // checkbox was a duplicate that called the same handle.setLabels
    // API with a different default font size, letting the two
    // controls drift out of sync.  Removing the wrapper here + the
    // template control + the listener wire keeps the knob bar as the
    // sole owner per the file-top design note.

    // Frozen-atom indices from runtime_info.frozen_atoms (sidecar-
    // driven; see parsers/_sidecar.py).  Returns a Set<number> for
    // O(1) membership; empty set when no sidecar / no frozen field.
    function _frozenSet() {
        const rt = state.data && state.data.runtime_info;
        const arr = rt && rt.frozen_atoms;
        if (!Array.isArray(arr) || !arr.length) return null;
        return new Set(arr);
    }

    // Toggle the "Hide frozen atoms" overlay on the embed.  Single
    // overlay entry whose ``style.hidden: true`` applies to every
    // frozen index — same primitive selection-panel isolate-mode
    // uses (mol-viewer-embed.js short-circuits empty stylespec to
    // disable every rep + the bonds that touch those atoms).  The
    // trajectory inspector has no other overlay sender, so a full
    // setOverlays replace is safe here.
    function applyHideFrozen() {
        if (!_handle) return;
        // G6 2026-06-14: re-render the Inspect atom-list so frozen
        // rows disappear (when on) / re-appear (when off) in lockstep
        // with the 3D viewer overlay.  Pre-fix the table was rendered
        // once per rebuildModel and stayed atom-complete even after
        // the user toggled hide-frozen on, so the user could pick a
        // frozen atom that was visually hidden in the viewer.
        rebuildInspectAtomList();
        const cb = $("hide-frozen");
        const frozen = _frozenSet();
        if (!frozen || !cb || !cb.checked) {
            _handle.setOverlays({ atoms: [] });
            return;
        }
        _handle.setOverlays({
            atoms: [{
                indices: Array.from(frozen),
                style: { hidden: true },
            }],
        });
    }

    // 2026-06-14: the "Hide frozen atoms" row is ALWAYS visible.
    // Pre-fix this function gated visibility on
    // ``runtime_info.frozen_atoms.size > 0`` and hid the row when
    // the parser failed to surface indices -- which made the
    // toggle silently vanish every time a SIESTA .out failed the
    // filename-pairing heuristic, every time a Results-tab user
    // loaded a constraint-free structure, etc.  The control's
    // PRESENCE is a stable UI affordance; only its EFFECT depends
    // on data.  When there are no frozen atoms, clicking the
    // checkbox is a no-op (applyHideFrozen() returns early because
    // _frozenSet() is empty).  Stable affordance, correct effect,
    // no muscle-memory whiplash.
    //
    // Kept as a named no-op so the existing callsites compile and
    // a future refactor can re-attach availability logic without a
    // wider refactor.  All it does today is keep the checkbox in
    // sync with state when external code resets frozen_atoms:
    // un-check it so the user doesn't see a stale "hidden" filter.
    function refreshHideFrozenAvailability() {
        const frozen = _frozenSet();
        if (frozen && frozen.size > 0) return;
        const cb = $("hide-frozen");
        if (cb && cb.checked) {
            cb.checked = false;
            applyHideFrozen();
        }
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
        // When the "Hide frozen atoms" toggle is on, skip force
        // vectors on frozen atoms — their forces are constraint-
        // balancing artefacts, not physical free-atom forces.
        const hideFrozen = $("hide-frozen") && $("hide-frozen").checked;
        const frozen = hideFrozen ? _frozenSet() : null;

        const mags = forces.map(function (f) {
            return Math.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        });
        // Recompute max excluding frozen atoms when the filter is on
        // — otherwise the "highlight max" arrow would still gold the
        // frozen-atom max and the user wouldn't see the gold marker
        // move to the true free-atom max.
        let maxMag = 0, maxIdx = -1;
        for (let i = 0; i < mags.length; i++) {
            if (frozen && frozen.has(i)) continue;
            if (mags[i] > maxMag) { maxMag = mags[i]; maxIdx = i; }
        }

        const arrows = [];
        for (let i = 0; i < frame.length && i < forces.length; i++) {
            if (frozen && frozen.has(i)) continue;
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
        // Honour the "Hide frozen atoms" toggle the same way
        // _buildArrowsForFrame does: skip frozen indices when the
        // checkbox is on so the reported max-force and arrow count
        // describe what's actually drawn (the unfrozen / free
        // atoms), not the whole frame.  Without this the status
        // line keeps reporting the frozen-atom max even after the
        // user hides those arrows, which made the toggle look
        // like a no-op.
        const hideFrozen = $("hide-frozen") && $("hide-frozen").checked;
        const frozen = hideFrozen ? _frozenSet() : null;
        let maxMag = 0;
        let drawn  = 0;
        for (let i = 0; i < forces.length; i++) {
            if (frozen && frozen.has(i)) continue;
            const f   = forces[i];
            const mag = Math.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
            if (mag > maxMag) maxMag = mag;
            if (mag >= fmin) drawn += 1;
        }
        const suffix = (frozen && frozen.size > 0)
            ? " (frozen atoms hidden)"
            : "";
        if (drawn === 0) {
            setForcesStatus(
                "0 arrows shown (all |F| < threshold "
              + fmin.toFixed(3) + " eV/Å; max |F| = "
              + maxMag.toFixed(3) + ")" + suffix + ".");
        } else {
            setForcesStatus(
                "Showing " + drawn + " arrow"
              + (drawn === 1 ? "" : "s")
              + " (max |F| = " + maxMag.toFixed(3) + " eV/Å)"
              + suffix + ".");
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
    /*  Inspect tab: 1-/2-/3-atom picking + live xyz/distance/angle        */
    /* ------------------------------------------------------------------ */
    //
    // The embed owns halo rendering via pick.mode = "triple" + pick.halo;
    // _postFramePositionRedraw re-renders halos every frame so they
    // track trajectory motion.  Atom-list row clicks push into the
    // embed's pick state via handle.setPickedIndices.  Single source
    // of truth: the embed's pickedIndices.  Trajectory queries it
    // live via _handle.getPickedIndices() in updateInspectPanel +
    // refreshAtomListHighlights (no host-side mirror per D3
    // contract, fixed in #246 B3).  Cap was raised from 2 to 3 in
    // task #299 (2026-06-08); the readout uses the shared
    // lib/selection/measurements.js math + display.

    function _picks() {
        return _handle ? _handle.getPickedIndices() : [];
    }

    // The frame coords are stored as ``[element, x, y, z]`` quads;
    // the shared measurements module wants ``[[x,y,z], …]`` and an
    // optional atomsMeta array with element labels.  Cheap to build
    // per-frame — n_atoms is small (single-digit hundreds for
    // typical inputs) and updateInspectPanel runs at most once per
    // displayed frame.
    function _framePositions(frame) {
        const out = new Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            const r = frame[i];
            out[i] = r ? [r[1], r[2], r[3]] : null;
        }
        return out;
    }
    function _frameAtomsMeta(frame) {
        const out = new Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            const r = frame[i];
            out[i] = { index: i, element: r ? (r[0] || "?") : "?" };
        }
        return out;
    }

    function updateInspectPanel() {
        const pa = _picks();
        const frame = (state.data && state.data.frames[state.currentFrame])
                    || [];
        const readout = $("inspect-measurement");
        if (!readout) return;
        const meas = window.molbuilder
                  && window.molbuilder.selection
                  && window.molbuilder.selection.measurements;
        let display = null;
        if (meas && pa.length > 0) {
            const positions = _framePositions(frame);
            const atomsMeta = _frameAtomsMeta(frame);
            // The embed's pickedIndices array is already in click
            // order — same semantics as the Modify-tab store's
            // pickOrder.  Pass it through both as the selection
            // (the module dedups + bails on 4+) AND as the
            // pickOrder so the angle vertex is the 2nd click.
            const result = meas.compute(pa, atomsMeta, positions, pa);
            if (result) display = result;
        }
        if (display) {
            readout.hidden = false;
            readout.dataset.kind = display.kind;
            readout.textContent = display.display;
        } else {
            readout.hidden = true;
            readout.textContent = "";
        }
        const hint  = $("inspect-hint");
        const btn   = $("inspect-clear");
        if (hint)  hint.hidden  = pa.length > 0;
        if (btn)   btn.disabled = pa.length === 0;
    }

    // Toggle pick state for ``idx`` (0-based atom index): drop if
    // already picked, append if new; cap at 3 picks total (a fourth
    // pick drops the oldest, sliding the FIFO window forward).
    // Drives the atom-list row click path; the embed handles the
    // viewer-click path natively via its pick.mode = "triple"
    // wiring (same cap, same FIFO semantics — kept in lock-step).
    // Pushes the new state into the embed via setPickedIndices so
    // halos stay in sync.
    function togglePickFromRow(idx) {
        if (!_handle) return;
        const pa = _handle.getPickedIndices();
        const existing = pa.indexOf(idx);
        if (existing !== -1) {
            pa.splice(existing, 1);
        } else {
            if (pa.length >= 3) pa.shift();
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

    // 1-based display via the ONE shared rule (data-vocabulary § 3.1), not an
    // ad-hoc i+1, so the trajectory atom-list # column agrees with the selector
    // + viewer overlays.  Falls back to i+1 if _atom-index.js isn't loaded.
    function _toDisplayIndex(i) {
        const m = window.molbuilder && window.molbuilder.atomIndexModel;
        return (m && typeof m.toDisplay === "function") ? m.toDisplay(i) : i + 1;
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
        // G6 2026-06-14: respect the hide-frozen toggle in the
        // Inspect atom list.  Pre-fix the table still rendered one
        // row per frozen atom + the row remained clickable, so the
        // user could pick + measure an atom that was visually
        // hidden in the 3D viewer.  Frozen rows now render as
        // ``.inspect-atom-row--frozen`` (CSS-muted, non-clickable)
        // when hide-frozen is on; this matches the force-arrow
        // filter pattern at line 562.
        const cb = $("hide-frozen");
        const hideFrozen = !!(cb && cb.checked);
        const frozen = hideFrozen ? _frozenSet() : null;
        const frag = document.createDocumentFragment();
        for (let i = 0; i < frame.length; i++) {
            // Skip rendering frozen atoms entirely when hide-frozen
            // is on -- a muted row is still a target for click +
            // would carry a stale picked-state on toggle-off.
            if (frozen && frozen.has(i)) continue;
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
            cellIdx.textContent = String(_toDisplayIndex(i));
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
        // G6 2026-06-14: when hide-frozen is on, rendered rows are
        // a SUBSET of frame atoms — indexing rows[i] against
        // frame[i] would mis-attribute coords to the wrong row.
        // Read each row's dataset.atomIndex to find its source.
        // Defensive: if the count matches frame.length we still use
        // the index path (cheap parity for the common hide-frozen-
        // off case).
        if (rows.length === frame.length) {
            for (let i = 0; i < frame.length; i++) {
                const cells = rows[i].children;
                cells[2].textContent = frame[i][1].toFixed(2);
                cells[3].textContent = frame[i][2].toFixed(2);
                cells[4].textContent = frame[i][3].toFixed(2);
            }
            return;
        }
        // hide-frozen path: rows are a filtered subset, use the
        // per-row dataset.atomIndex to look up the right frame row.
        for (let r = 0; r < rows.length; r++) {
            const tr = rows[r];
            const ai = Number(tr.dataset.atomIndex);
            if (!Number.isInteger(ai) || ai < 0 || ai >= frame.length) {
                // Stale row: atom count shrank, fall back to rebuild.
                rebuildInspectAtomList();
                return;
            }
            const cells = tr.children;
            cells[2].textContent = frame[ai][1].toFixed(2);
            cells[3].textContent = frame[ai][2].toFixed(2);
            cells[4].textContent = frame[ai][3].toFixed(2);
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
        // Atom-index labels: controlled exclusively by the molview
        // knob bar's Labels popover (2026-06-13 — see comment above
        // the retired applyIndexLabels stub).  No initial-render
        // call needed; labels start hidden until the user clicks.
        // Hide-frozen-atoms checkbox visibility + initial overlay
        // sync (2026-06-13).  Reveal the row only when the parser
        // surfaced frozen_atoms; apply the overlay if the user had
        // it checked from a prior load.
        refreshHideFrozenAvailability();
        applyHideFrozen();
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

    // Read theme tokens once per makePlots() so the trace colours
    // track lib/tokens.css.  Falls back to literal hex if the token
    // isn't defined (e.g. a partial CSS load), so the plot never
    // ends up colourless.  The non-themable plot-only colours
    // (the red "all atoms" force trace, the orange SCF-gnorm trace)
    // stay literal because they encode scientific-plot conventions
    // that should be stable across themes — see _trajectory_inspector
    // documentation in docs/protocols/results-tab.md.
    function _themeColors() {
        const cs = getComputedStyle(document.documentElement);
        const get = (name, fb) =>
            (cs.getPropertyValue(name) || "").trim() || fb;
        return {
            // Token-driven (theme-responsive)
            accent:     get("--accent",     "#6ba6ff"),
            success:    get("--success",    "#4ade80"),
            warnSoft:   get("--warn-soft",  "#d8a64b"),
            textMuted:  get("--text-muted", "#6c7280"),
            // Non-themable plot conventions
            forceAllAtoms: "#d62728",   // red — "all atoms" informational trace
            energy:        "#1f77b4",   // blue — single-trace
            scfEnergy:     "#1f77b4",   // blue — single-trace
            scfGnorm:      "#fb923c",   // orange — moved off green to keep
                                        //          the threshold-line green
                                        //          unambiguous
            stageMarker:   "#888",      // gray — stage-boundary dashed line
        };
    }

    // Pull convergence_targets out of runtime_info.  Returns null
    // when the parser didn't find them (older runs / non-molbuilder
    // scripts).  Callers (plots + summary block) decide what to do
    // with absence; the most common is "hide the threshold line +
    // render the hint instead."
    //
    // Two header shapes are normalised to a single flat dict:
    //
    //   FLAT (legacy, single-stage):
    //     {max_force_tol_eV_per_A: 0.023, scf_energy_tol: 1e-9, ...,
    //      source: "molwatch_header"}
    //     -> returned as-is.
    //
    //   NESTED (staged runs, task #534):
    //     {stage1: {max_force_tol_eV_per_A: ..., ...},
    //      stage2: {max_force_tol_eV_per_A: ..., ...}, ...,
    //      source: "molwatch_header"}
    //     -> flattened to the LAST stage's leaf dict (the tightest
    //     tier — the run only stops when the last enabled stage's
    //     targets are met, so that's the "what counts as converged"
    //     reference the plot needs).  Future commit may wire per-
    //     frame stage attribution so threshold lines per stage can
    //     be drawn against the trajectory; today we collapse to the
    //     run's terminal target.
    function _convergenceTargets() {
        const rt = state.data && state.data.runtime_info;
        if (!rt) return null;
        const ct = rt.convergence_targets;
        if (!ct || typeof ct !== "object") return null;
        // Detect nested shape: any top-level value is itself a
        // (non-array) object, excluding the meta ``source`` key.
        let stageNames = [];
        for (const k of Object.keys(ct)) {
            if (k === "source") continue;
            const v = ct[k];
            if (v && typeof v === "object" && !Array.isArray(v)) {
                stageNames.push(k);
            }
        }
        if (stageNames.length === 0) {
            return ct;                 // flat shape — return verbatim
        }
        // Nested.  Pick the LAST stage in insertion order (Object.keys
        // preserves it for non-integer string keys) as the run's
        // tightest tier.  Copy ``source`` across so the summary block
        // still reads "from molwatch header".
        const last = stageNames[stageNames.length - 1];
        const leaf = ct[last] || {};
        const flat = Object.assign({}, leaf);
        if (typeof ct.source === "string") flat.source = ct.source;
        return flat;
    }

    // Render the convergence-summary section above the plots row.
    // Shape + intent documented in docs/protocols/results-tab.md
    // and templates/_trajectory_inspector.html.  Uses textContent /
    // createElement everywhere — no innerHTML interpolation, per
    // the XSS audit's no-unsafe-innerHTML rule.
    function _renderConvergenceSummary() {
        const sec = $("convergence-summary");
        if (!sec) return;
        const ct = _convergenceTargets();
        const targets = $("convergence-summary-targets");
        const current = $("convergence-summary-current");
        const hint = $("convergence-summary-hint");
        const sourceEl = $("convergence-summary-source");

        // Clear scrubbed inner state on each render — the parent
        // section is visibility-toggled below.
        if (targets) targets.replaceChildren();
        if (current) current.replaceChildren();
        if (sourceEl) sourceEl.textContent = "";
        if (hint) hint.hidden = true;
        sec.classList.remove("is-off-target", "is-unknown");

        if (!ct) {
            // No targets — show the "where to put them" hint.
            sec.classList.add("is-unknown");
            sec.hidden = false;
            if (hint) {
                hint.hidden = false;
                hint.textContent = (
                    "Convergence targets not found in source. "
                    + "Load the source .fdf / .py file next to the run, "
                    + "or rerun via molbuilder for self-describing output "
                    + "— the molwatch.log header writes the targets "
                    + "automatically."
                );
            }
            return;
        }

        sec.hidden = false;

        // Source provenance label.
        const sourceMap = {
            "siesta_input_echo": "from SIESTA input echo (redata:)",
            "molwatch_header":   "from molwatch log header",
            "geomeTRIC_log":     "from geomeTRIC log",
        };
        if (sourceEl) {
            sourceEl.textContent =
                sourceMap[ct.source] || (ct.source ? "from " + ct.source : "");
        }

        // Targets list — render only the keys that are populated.
        const rows = [];
        if (typeof ct.max_force_tol_eV_per_A === "number") {
            rows.push(["max |F|",
                ct.max_force_tol_eV_per_A.toFixed(4) + " eV/Å"]);
        }
        if (typeof ct.dm_tolerance === "number") {
            rows.push(["DM tolerance", ct.dm_tolerance.toExponential(2)]);
        }
        if (typeof ct.scf_energy_tol === "number") {
            rows.push(["SCF energy tol",
                ct.scf_energy_tol.toExponential(2) + " Ha"]);
        }
        if (typeof ct.max_scf_iter === "number") {
            rows.push(["MaxSCFIterations", String(ct.max_scf_iter)]);
        }
        if (typeof ct.max_geom_iter === "number") {
            rows.push(["geom steps cap", String(ct.max_geom_iter)]);
        }
        if (typeof ct.max_displ_ang === "number") {
            rows.push(["max displ", ct.max_displ_ang + " Å"]);
        }
        if (targets) {
            for (const [label, val] of rows) {
                const dt = document.createElement("dt");
                dt.textContent = label;
                const dd = document.createElement("dd");
                dd.textContent = val;
                targets.appendChild(dt);
                targets.appendChild(dd);
            }
        }

        // Current step's max force vs target.  Prefers the
        // convergence-gating "free atoms" value when present; falls
        // back to "all atoms" otherwise.  Adds the off-target
        // styling when ratio > 2× so the visual reads "still work
        // to do" without the user having to compute the division.
        const tol = ct.max_force_tol_eV_per_A;
        const forces = state.data && state.data.max_forces;
        const constrained = state.data && state.data.max_forces_constrained;
        const iters = state.data && state.data.iterations;
        if (typeof tol === "number" && Array.isArray(forces)
                && forces.length > 0 && current) {
            const lastIdx = forces.length - 1;
            const allMax = forces[lastIdx];
            const freeMax = (Array.isArray(constrained) && constrained.length)
                ? constrained[constrained.length - 1] : null;
            const primary = (freeMax != null) ? freeMax : allMax;
            if (typeof primary === "number" && isFinite(primary)) {
                const ratio = primary / tol;
                const cls = ratio > 2 ? "off-target"
                          : (ratio <= 1 ? "at-target" : "");
                const ratioText = ratio <= 1
                    ? "at or below target"
                    : ratio.toFixed(1) + "× over target";
                const stepLabel = (Array.isArray(iters) && iters[lastIdx] != null)
                    ? "Step " + iters[lastIdx]
                    : "Latest step";
                const label = (freeMax != null)
                    ? "max |F| (free atoms)"
                    : "max |F|";
                const span = document.createElement("span");
                if (cls) span.className = cls;
                span.textContent = stepLabel + " — " + label + " = "
                    + primary.toFixed(4) + " eV/Å — " + ratioText;
                current.appendChild(span);
                if (freeMax != null && typeof allMax === "number"
                        && isFinite(allMax)) {
                    current.appendChild(document.createElement("br"));
                    const span2 = document.createElement("span");
                    span2.style.color = "var(--text-muted)";
                    span2.textContent = "max |F| (all atoms)  = "
                        + allMax.toFixed(4) + " eV/Å";
                    current.appendChild(span2);
                }
                sec.classList.toggle("is-off-target", ratio > 2);
            }
        }
    }

    // Build Plotly ``shapes`` + ``annotations`` for stage boundaries
    // when state.data is a multi-stage merged trajectory.  Each
    // stage transition (where a new source .molwatch.log begins)
    // gets a dashed vertical line + a small label at the top of the
    // plot showing the stage name.  Returns {shapes, annotations}
    // (both empty arrays for single-stage runs).
    function stageMarkers(themeArg) {
        const stages = state.data && state.data.stages;
        if (!Array.isArray(stages) || stages.length < 2) {
            return { shapes: [], annotations: [] };
        }
        // ``themeArg`` is the theme palette from _themeColors() —
        // makePlots reads it once and passes here so we don't
        // re-walk getComputedStyle for every plot.  Fallback to a
        // neutral hex if a caller forgot, so this function stays
        // safe to call in isolation.
        const stageColor = (themeArg && themeArg.stageMarker) || "#888";
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
                    line: { color: stageColor, width: 1, dash: "dash" },
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
                font: { size: 10, color: stageColor },
                xanchor: "center",
                yanchor: "bottom",
            });
        }
        return { shapes, annotations };
    }

    /**
     * Render the pre-data status banner that explains a blank
     * energy/force plot when the engine hasn't yet written a first
     * data point.  Engine-agnostic: shows "Initializing — first
     * energy estimate pending" for SIESTA, "Optimizing — first
     * step pending" for PySCF, or hides itself once there's
     * usable data.
     *
     * Added 2026-06-14 so a freshly-started run doesn't look like
     * a broken UI to the user.  The "blank plots" symptom on a
     * cold start is correct (engine just hasn't written anything
     * yet) but reads as a bug without explanation.
     */
    // Last rendered banner state -- guards the poll-tick re-render
    // path.  Per ``feedback_no_rewrite_user_ui_state_on_poll`` the
    // banner must NOT stomp on textContent every poll tick once
    // it's settled into "data present + hidden"; that's wasted DOM
    // work and produces unnecessary mutation observers / layout
    // thrash on slow tabs.  ``null`` sentinel means "never rendered
    // yet" so the very first call still writes.
    let _lastEmptyStatus = { hidden: null, text: null };

    function _renderEmptyStatus() {
        const el = $("trajectory-empty-status");
        if (!el) return;
        // Compute the target state first; only commit when it
        // differs from what's already on the DOM.
        const data = state.data;
        let nextHidden, nextText;
        if (!data) {
            nextHidden = false;
            nextText = "Loading run output…";
        } else {
            // Heuristic: usable data when any frame has a finite
            // energy OR any SCF cycle has been parsed.
            const energies = data.energies || [];
            const hasEnergy = energies.some(e =>
                typeof e === "number" && Number.isFinite(e));
            const scfHistory = data.scf_history || [];
            const hasScf = scfHistory.some(s => Array.isArray(s)
                                              && s.length > 0);
            if (hasEnergy || hasScf) {
                nextHidden = true;
                nextText = "";   // not displayed when hidden; sentinel only
            } else {
                nextHidden = false;
                const fmt = data.source_format || "";
                if (fmt === "pyscf") {
                    nextText = (
                        "PySCF initializing — first geomeTRIC step "
                        + "pending.  No energy yet to plot; SCF "
                        + "setup, basis build, and initial guess "
                        + "can take minutes to tens of minutes for "
                        + "large systems."
                    );
                } else if (fmt === "siesta") {
                    nextText = (
                        "SIESTA initializing — first SCF cycle "
                        + "pending.  No energy yet to plot; "
                        + "pseudopotential + basis setup and "
                        + "initial DM construction can take "
                        + "seconds to a few minutes for large "
                        + "systems."
                    );
                } else {
                    nextText = (
                        "Run initializing — no data points yet.  "
                        + "Plots will populate once the engine "
                        + "writes its first energy line."
                    );
                }
            }
        }
        // Early-return on no-op tick.  Stable affordance, no DOM
        // writes when nothing changed.
        if (nextHidden === _lastEmptyStatus.hidden
                && nextText === _lastEmptyStatus.text) {
            return;
        }
        el.hidden = nextHidden;
        if (!nextHidden) {
            // Only touch textContent when we're about to display;
            // when hidden, the text doesn't matter and skipping
            // the write avoids reflow.
            const textEl = el.querySelector(
                ".trajectory-empty-status-text");
            if (textEl) textEl.textContent = nextText;
        }
        _lastEmptyStatus = { hidden: nextHidden, text: nextText };
    }

    function makePlots() {
        if (!state.data) return;
        // Contract § 4 Invariant 2 (in-progress frame filter):
        // ``data.in_progress[i]`` flags a partial frame whose
        // energy / max_force values are placeholders (the parser
        // emits these at EOF when the engine is still writing the
        // step).  Plot trace builders use ``_plottableIdx`` to
        // omit those frames from x / y arrays -- avoids the "odd
        // value disappears on full refresh" bug class.  The frame
        // still ships and shows in the inspect list with a
        // "computing..." badge; only the user-facing plot omits
        // its placeholder y-value.
        const _plottableIdx = plottableFrames(state.data);
        // Derive filtered x / energy / max_force / max_force_C
        // arrays once -- all three plot traces below consume them.
        const x_raw = state.data.iterations || [];
        const energies_raw = state.data.energies || [];
        const max_forces_raw = state.data.max_forces || [];
        const max_forces_c_raw = state.data.max_forces_constrained || [];
        const x = _plottableIdx.map(i => x_raw[i]);
        const energies_plot = _plottableIdx.map(i => energies_raw[i]);
        const max_forces_plot = _plottableIdx.map(i => max_forces_raw[i]);
        const has_constrained =
            Array.isArray(max_forces_c_raw) && max_forces_c_raw.length > 0;
        const max_forces_c_plot = has_constrained
            ? _plottableIdx.map(i => max_forces_c_raw[i])
            : [];
        const theme = _themeColors();
        const ct = _convergenceTargets();
        const stageMx = stageMarkers(theme);

        // Render the empty-state banner BEFORE the plots so it
        // either takes the screen during the brief pre-data window
        // or is hidden in time for Plotly to lay out cleanly.
        _renderEmptyStatus();

        // Render the convergence-summary band above the plots before
        // we touch Plotly — the section's visibility + content drive
        // the eye to the run's progress, then the plots show "where".
        _renderConvergenceSummary();

        Plotly.react("energy-plot", [{
            x: x,
            y: energies_plot,
            mode: "lines+markers",
            line: { color: theme.energy, width: 1.5 },
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

        // 2026-06-12: two traces when the engine reports both.
        //
        //   * ``max_forces``             — across ALL atoms,
        //                                  including frozen ones.
        //                                  Informational.
        //   * ``max_forces_constrained`` — excluding frozen atoms.
        //                                  When the run has frozen
        //                                  atoms, this is what
        //                                  SIESTA actually compares
        //                                  against MD.MaxForceTol
        //                                  for convergence.
        //
        // ``max_forces_constrained`` is an empty list when no frame
        // in the run carried a constrained value (no frozen atoms);
        // in that case we render the single trace as before.
        const constrained = max_forces_c_plot;  // filtered via plottableFrames
        const forceTraces = [{
            x: x,
            y: max_forces_plot,
            mode: "lines+markers",
            line: { color: theme.forceAllAtoms, width: 1.5,
                    dash: (constrained && constrained.length)
                            ? "dash" : "solid" },
            marker: { size: (constrained && constrained.length) ? 4 : 6 },
            name: (constrained && constrained.length)
                    ? "all atoms"
                    : "Max |F|",
            hovertemplate: "step %{x}<br>Max |F| (all atoms): "
                         + "%{y:.4r} eV/A<extra></extra>",
            connectgaps: false,
        }];
        if (constrained && constrained.length) {
            forceTraces.push({
                x: x,
                y: constrained,
                mode: "lines+markers",
                // Theme accent (blue) for the convergence-gating
                // primary signal — moved off green so the green
                // threshold line stays unambiguously "the target".
                line: { color: theme.accent, width: 2 },
                marker: { size: 6 },
                name: "free atoms",
                hovertemplate: "step %{x}<br>Max |F| (free atoms): "
                             + "%{y:.4r} eV/A"
                             + "<extra>convergence-gating</extra>",
                connectgaps: false,
            });
        }
        // ``hasBothTraces`` is used twice — extract for readability.
        var hasBothTraces = !!(constrained && constrained.length);

        // Threshold-line shape + annotation on the force plot when
        // the parser found a convergence target.  Green dashed at
        // y = max_force_tol_eV_per_A; annotation right-anchored so
        // it never collides with the trace at the leading edge.
        // Auto-switch the y-axis to log scale when the trajectory
        // is more than 50× above the target so the target line
        // isn't crushed against the x-axis.
        const forceShapes = stageMx.shapes.slice();
        const forceAnnotations = stageMx.annotations.slice();
        var forceYType = undefined;
        if (ct && typeof ct.max_force_tol_eV_per_A === "number") {
            const tol = ct.max_force_tol_eV_per_A;
            forceShapes.push({
                type: "line", xref: "paper",
                x0: 0, x1: 1,
                yref: "y", y0: tol, y1: tol,
                line: { color: theme.success, width: 1.5, dash: "dash" },
            });
            forceAnnotations.push({
                xref: "paper", x: 1, xanchor: "right",
                yref: "y", y: tol, yanchor: "bottom",
                text: "target " + tol.toFixed(3) + " eV/Å",
                font: { size: 9, color: theme.success },
                showarrow: false,
            });
            // Log y-axis when the run is far above target.
            const allValues = state.data.max_forces || [];
            const maxSeen = allValues.reduce(
                (m, v) => (typeof v === "number" && v > m ? v : m), 0);
            if (tol > 0 && maxSeen / tol > 50) {
                forceYType = "log";
            }
        }
        Plotly.react("force-plot", forceTraces, {
            title: { text: "Max force", font: { size: 13 } },
            margin: { l: 8, r: 12, t: 32,
                      b: hasBothTraces ? 58 : 32 },
            xaxis: { title: { text: "CG step", standoff: 4 },
                     zeroline: false, automargin: true,
                     nticks: 6 },
            yaxis: { title: { text: "Max |F| (eV/\u00C5)", standoff: 4 },
                     tickformat: ".3~r",
                     // rangemode "tozero" anchors the axis at the
                     // x-axis when on a linear scale (the visual
                     // metaphor: "converged means y -> 0").  Log
                     // scale picks its own range so the threshold
                     // line + the trace are both legible at any
                     // magnitude \u2014 auto-engaged when the trajectory
                     // is more than 50x above target.
                     rangemode: forceYType === "log" ? undefined : "tozero",
                     type: forceYType,
                     zeroline: false,
                     automargin: true, nticks: 5 },
            font: { family: "system-ui, sans-serif", size: 10 },
            showlegend: hasBothTraces,
            legend: {
                orientation: "h",
                x: 0.5, xanchor: "center",
                y: -0.22, yanchor: "top",
                font: { size: 9 },
                bgcolor: "rgba(0,0,0,0)",
            },
            shapes:      forceShapes,
            annotations: forceAnnotations,
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
        // Local theme + targets lookups so this can be reasoned
        // about in isolation; both helpers are cheap (one
        // getComputedStyle, one runtime_info read).
        const theme = _themeColors();
        const ct = _convergenceTargets();
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
        // and look natural on a log y-axis.  Both use the orange
        // ``theme.scfGnorm`` so the green threshold line stays the
        // only green element on the plot (2026-06-13 recolor; the
        // pre-fix amber + green choices conflicted with the new
        // threshold-line green).
        let residual, residualName, residualUnit;
        const residualColor = theme.scfGnorm;
        if (current[0].gnorm !== undefined) {
            residual      = current.map(c => c.gnorm);
            residualName  = "|g|";
            residualUnit  = "eV/Å";
        } else if (current[0].dHmax !== undefined) {
            residual      = current.map(c => c.dHmax);
            residualName  = "dHmax";
            residualUnit  = "eV";
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

        // Wall-time annotation.  Two-stage progressive refinement:
        //
        //   STAGE 1 (early, only 1 sample): show SIESTA's first-iter
        //   snapshot from the ``timer: Routine,Calls,Time,% = IterSCF``
        //   line (parsed once at scf:1 -- verified empirically: SIESTA
        //   emits this line exactly once per run, not per iter).
        //
        //   STAGE 2 (>=2 polls with iter increase): use a measured
        //   wall-clock average from the live-poll history.  Each
        //   render pushes (Date.now, total_SCF_iters_across_all_frames)
        //   into state.scfPollHistory.  Two samples that span an
        //   iter increase give us actual per-iter wall time --
        //   including the brief CPU work between iters (DM mixing,
        //   mesh, Etot), which is what the user actually cares about
        //   when judging "is this run going to finish today?".
        //
        // The transition is automatic: as soon as the rolling-window
        // measurement is available it replaces the snapshot.  When the
        // user is mid-iter and the polls are too close together for
        // an iter to have completed, we fall back to STAGE 1 so the
        // display never goes blank.
        //
        // Format helper -- "X.Ys" under 60 s, "Xm Ys" otherwise.  Above
        // 60 s the researcher's mental model switches from "seconds"
        // to "minutes" and the rounded form is more useful.
        const fmtWall = (s) => {
            if (s < 60) return s.toFixed(1) + "s";
            const m = Math.floor(s / 60);
            const r = Math.round(s - m * 60);
            return m + "m " + r + "s";
        };

        // Track total SCF iters across ALL frames for this poll cycle.
        // The trajectory may have N committed frames each with their
        // own scf_history plus an in-progress frame -- iterations are
        // additive across them.
        let totalIters = 0;
        if (state.data && Array.isArray(state.data.frames)) {
            for (const f of state.data.frames) {
                if (f && Array.isArray(f.scf_history)) {
                    totalIters += f.scf_history.length;
                }
            }
        }
        // Push this sample; keep last 32 (~32 polls = ~32 min at the
        // default 60 s poll cadence -- plenty of window for averaging
        // without unbounded growth).
        const nowMs = Date.now();
        state.scfPollHistory.push({ ts: nowMs, totalIters: totalIters });
        if (state.scfPollHistory.length > 32) {
            state.scfPollHistory.shift();
        }

        // STAGE 2a: SERVER-SIDE mtime-delta measurement.  The watch
        // blueprint computes per-iter wall time from the .out's mtime
        // advancement between polls (see watch.py:_attach_iter_walltime).
        // This is the best source: mtime is filesystem state so it
        // survives browser reload, and it's grounded in when the
        // engine ACTUALLY wrote -- not an approximation from the
        // browser's polling clock.  When present, prefer it over
        // the local scfPollHistory.
        const serverPerIter = state.data && state.data.wall_time_per_iter_s;
        const serverWindow  = state.data && state.data.wall_time_per_iter_window;
        let measured = null;
        if (typeof serverPerIter === "number" && isFinite(serverPerIter)
            && serverPerIter > 0) {
            measured = {
                avgPerIter:    serverPerIter,
                itersInWindow: (serverWindow && serverWindow.iters)   || 1,
                windowSeconds: (serverWindow && serverWindow.seconds) || serverPerIter,
                source:        "measured",
            };
        }
        // STAGE 2b: client-side rolling avg as a fallback for the
        // first poll or two before the server has paired samples.
        // Once the server measurement lands, this is shadowed.
        if (measured === null && state.scfPollHistory.length >= 2) {
            const oldest = state.scfPollHistory[0];
            const dtS = (nowMs - oldest.ts) / 1000;
            const di  = totalIters - oldest.totalIters;
            if (di >= 1 && dtS > 1) {
                measured = {
                    avgPerIter:    dtS / di,
                    itersInWindow: di,
                    windowSeconds: dtS,
                    source:        "live",
                };
            }
        }

        if (measured !== null) {
            // Provenance label spelt out so the user can tell at a
            // glance where the number came from:
            //   "from refresh delta" -- server compared two polls'
            //     mtimes (most trustworthy; survives browser reload).
            //   "from poll estimate" -- client-side scfPollHistory
            //     fallback (used during the first 1-2 polls before
            //     the server has paired mtimes).
            const provenanceText = measured.source === "measured"
                ? "from refresh delta"
                : "from poll estimate";
            statusText += ", ~" + fmtWall(measured.avgPerIter)
                       + "/iter (" + provenanceText + ", "
                       + measured.itersInWindow + " iter"
                       + (measured.itersInWindow > 1 ? "s" : "")
                       + " in last " + fmtWall(measured.windowSeconds) + ")";
        } else {
            // STAGE 1: explicit precedence ladder over parser-attached
            // walltime data.  SIESTA emits its ``timer: IterSCF`` line
            // EXACTLY ONCE per run (verified empirically: stage3-run0
            // with 15 geom steps + ~520 SCF cycles had a single timer
            // line at iter 1 of step 1).  But user-facing relevance
            // ranks the available data in a clear order:
            //
            //   1. Current step's most recent cycle (best -- it's
            //      what's happening right now, accounts for any
            //      mid-run cost changes like mesh adaptation)
            //   2. Previous (completed) step's most recent cycle
            //      (good -- a real average over a finished SCF run
            //      at similar problem size)
            //   3. Very first step's iter-1 snapshot (worst -- old,
            //      may include warm-up costs like cache cold-start
            //      and MPS daemon spin-up, but always present in a
            //      well-formed SIESTA .out)
            //   4. Nothing -- no annotation (genuinely no data yet)
            //
            // Each rung also carries a provenance string into the UI
            // so the user can see WHICH source the number came from.
            // Today (3) is the common path because SIESTA only emits
            // once, but (1) and (2) auto-kick-in if a future SIESTA
            // build or PySCF logs richer per-step timing.
            const findLatestCycleWithWalltime = (step) => {
                if (!Array.isArray(step)) return null;
                for (let j = step.length - 1; j >= 0; j--) {
                    const c = step[j];
                    const v = c && c.cumulative_walltime_s;
                    if (typeof v === "number" && isFinite(v)) return c;
                }
                return null;
            };

            let baselineCycle = null;
            let provenance = "";

            // (1) Current step
            const currentHit = findLatestCycleWithWalltime(current);
            if (currentHit) {
                baselineCycle = currentHit;
                provenance = "from current step report";
            }

            // (2) Previous (completed) step
            if (!baselineCycle && stepIdx >= 1) {
                const prevHit = findLatestCycleWithWalltime(history[stepIdx - 1]);
                if (prevHit) {
                    baselineCycle = prevHit;
                    provenance = "from last step report";
                }
            }

            // (3) First IterSCF snapshot anywhere -- SIESTA's
            //     one-and-only ``timer: IterSCF`` line at iter 1 of
            //     step 1.  Used when the live mtime-delta hasn't
            //     produced a fresh measurement yet (e.g. the page
            //     was just opened, the server has only seen 1 poll).
            if (!baselineCycle) {
                for (let i = 0; i < history.length; i++) {
                    const step = history[i];
                    if (!Array.isArray(step)) continue;
                    for (let j = 0; j < step.length; j++) {
                        const c = step[j];
                        const v = c && c.cumulative_walltime_s;
                        if (typeof v === "number" && isFinite(v)) {
                            baselineCycle = c;
                            provenance = "from SIESTA iter-1 timer; "
                                       + "refresh delta pending";
                            break;
                        }
                    }
                    if (baselineCycle) break;
                }
            }

            // (4) Nothing -- leave statusText untouched.
            if (baselineCycle !== null) {
                const cumT = baselineCycle.cumulative_walltime_s;
                const cumN = baselineCycle.cumulative_calls;
                // Prefer per-iter (cumulative/calls).  Fall back to
                // raw cumulative when calls is missing (rare: Fortran
                // column overflow on Calls but not Time -- our
                // overflow-tolerant parser handles each field
                // independently, see test_overflowed_time_drops_only
                // _walltime_keeps_calls).
                const perIter = (typeof cumN === "number" && cumN >= 1)
                    ? cumT / cumN
                    : cumT;
                statusText += ", ~" + fmtWall(perIter)
                           + "/iter (" + provenance + ")";
            }
        }
        $("scf-status").textContent = statusText;

        // SCF energy convergence within the current step.
        Plotly.react("scf-energy-plot", [{
            x: cycles,
            y: energies,
            mode: "lines+markers",
            line: { color: theme.scfEnergy, width: 1.5 },
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
            // SCF-tolerance threshold line, if we know the target.
            // dm_tolerance is dimensionless (DM convergence) and
            // applies to dHmax-style residuals; scf_grad_tol /
            // scf_energy_tol apply to PySCF's |g|.  Pick the one
            // that's relevant to the trace we're plotting; skip if
            // neither was found in the source.
            const scfShapes = [];
            const scfAnnotations = [];
            const scfTol = (residualName === "dHmax")
                ? (ct && typeof ct.dm_tolerance === "number"
                    ? ct.dm_tolerance : null)
                : (ct && typeof ct.scf_grad_tol === "number"
                    ? ct.scf_grad_tol : null);
            if (scfTol != null) {
                scfShapes.push({
                    type: "line", xref: "paper",
                    x0: 0, x1: 1,
                    yref: "y", y0: scfTol, y1: scfTol,
                    line: { color: theme.success, width: 1.5, dash: "dash" },
                });
                scfAnnotations.push({
                    xref: "paper", x: 1, xanchor: "right",
                    yref: "y", y: scfTol, yanchor: "bottom",
                    text: "tol " + scfTol.toExponential(1),
                    font: { size: 9, color: theme.success },
                    showarrow: false,
                });
            }
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
                shapes:      scfShapes,
                annotations: scfAnnotations,
            }, { displayModeBar: false, responsive: true });
        } else {
            resPlotEl.hidden = true;
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Polling                                                            */
    /* ------------------------------------------------------------------ */

    async function pollOnce() {
        // 2026-06-14 in-flight guard: if the previous tick's
        // request is still on the wire, SKIP this tick.  The next
        // tick (POLL_MS later) fires only if the prior one has
        // settled.  Pre-fix every tick UNCONDITIONALLY aborted any
        // in-flight prior poll, tearing down the TLS connection
        // mid-response on slow / large-file polls and surfacing
        // as ``SSL: UNEXPECTED_EOF_WHILE_READING`` in the server
        // log.  The legitimate cancellation path -- the user's
        // dispose() or a file-change supersede -- still aborts
        // via the AbortController held in ``state.pollAbort``;
        // it's only the tick-overlap case the guard prevents.
        //
        // The earlier "stale-mtime-by-arrival" concern in the
        // old comment was always a false worry: ``state.mtime``
        // is the POSTed query param + the server returns
        // ``changed: false`` for any matched mtime, so a slow
        // response with the old mtime is correctly a no-op.  No
        // staleness risk -- just a wasted client wait, which the
        // guard skips entirely.
        if (state.pollInFlight) {
            return;
        }
        state.lifecycle.pollAbort = new AbortController();
        const signal = state.lifecycle.pollAbort.signal;
        state.lifecycle.pollInFlight = true;
        // File-identity guard (contract § 4 Invariant 1): the
        // sequence number this poll is bound to.  If a file-switch
        // / Refresh bumps fetchSeq while this fetch is on the wire,
        // the response is dropped.
        const mySeq = state.lifecycle.fetchSeq;
        try {
            const url = state.mtime !== null
                ? "/api/watch/data?mtime=" + encodeURIComponent(state.mtime)
                : "/api/watch/data";
            const r = await fetch(url, { signal: signal })
                .then(x => x.json());
            if (signal.aborted) return;
            if (state.lifecycle.fetchSeq !== mySeq) return;
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
                // No new data, but a no-change tick still counts
                // for the 2-tick WATCHING -> LOADED buffer if the
                // run was previously reported finished.  Settle
                // re-using the cached fileState.data.run_state.
                _settlePostLoad();
                return;
            }
            applyNewData(r);
            // Contract \u00a7 3 matrix rows "watch tick (new SCF iter)"
            // and "watch tick (new frame)" + the WATCHING->LOADED
            // 2-tick buffer all run through _settlePostLoad.  This
            // is the single canonical site where run_state drives
            // state-machine transitions (Refresh path is the other).
            _settlePostLoad();
        } catch (e) {
            // AbortError: dispose() or a file-change supersede ran.
            // Silent -- the next tick (or the unmount) is the
            // authoritative state.  Pre-fix this also caught the
            // tick-overlap abort case; the in-flight guard above
            // means it no longer fires for plain overlap.
            if (e.name === "AbortError") return;
            setStatus("Network error: " + e.message, "error");
        } finally {
            // Always release the in-flight flag so the NEXT tick
            // (typically 60 s away) can fire.  Even on AbortError
            // / network error, the controller is settled at this
            // point and a new tick is welcome.
            state.pollInFlight = false;
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
        // SIESTA-specific build + diagonalizer info (populated by the
        // parse/engines/siesta.py header probes; absent on PySCF runs
        // and on truncated SIESTA outputs that didn't reach the
        // redata: echo).  See tests/test_siesta_runtime_info_build.py
        // for the schema.
        const sb = rt.siesta_build;
        if (sb && sb.version) {
            let s = "SIESTA " + sb.version;
            if (Array.isArray(sb.parallelisations) && sb.parallelisations.length) {
                s += " · " + sb.parallelisations.join("+");
            }
            parts.push(s);
        }
        const sd = rt.siesta_diag;
        if (sd && sd.algorithm) {
            let s = sd.algorithm;
            if (sd.elpa_gpu === true) s += " GPU";
            parts.push(s);
        }
        if (sd && sd.gpu_device) {
            let g = String(sd.gpu_device)
                .replace(/^NVIDIA GeForce /, "")
                .replace(/^NVIDIA /, "");
            if (sd.gpu_compute_capability) g += " " + sd.gpu_compute_capability;
            parts.push(g);
        }
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

    function _scfFingerprint(data) {
        // Compact stable string over the SCF-history fields that
        // makePlots/renderScfProgress branch on.  Used by
        // applyNewData's noNewContent guard to detect "same geometry
        // but SCF iterations grew within the in-flight step" -- the
        // case where the geometry-only guard would otherwise leave
        // the SCF energy / dDmax / residual plots frozen mid-run.
        //
        // Cheap fingerprint shape: ``<num_steps>/<iters_in_last_step>/
        // <last_cycle_etot>``.  All three pieces flip when the parser
        // appends a new cycle to scf_history[lastStep], so equality
        // means "nothing new to plot" on this axis.
        if (!data || !Array.isArray(data.scf_history)
                  || !data.scf_history.length) {
            return "";
        }
        const hist = data.scf_history;
        const last = hist[hist.length - 1];
        if (!Array.isArray(last) || !last.length) {
            return hist.length + "/0/";
        }
        const tail = last[last.length - 1] || {};
        // Energy field varies by engine: PySCF "etot" / SIESTA
        // "eharris" (early iters before total energy is meaningful)
        // / SIESTA "etot" (steady iters).  Any of them flipping
        // counts as new data.
        const e = (tail.etot != null) ? tail.etot
                : (tail.eharris != null) ? tail.eharris
                : (tail.energy != null) ? tail.energy
                : "";
        return hist.length + "/" + last.length + "/" + e;
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

        // 2026-06-12: live-refresh no-new-frames short-circuit.
        //
        // The /api/watch/data poll fires every N seconds and frequently
        // returns the same trajectory it returned last time (no new
        // frames yet).  Before this guard the ``else`` branch below
        // (which handles the "can't append, do a full rebuild" case)
        // would fire on every same-data poll and the rebuild would:
        //   1. reset the 3Dmol camera (refit + applyCell rebuild),
        //   2. tear down the animation loop and rearm it from frame 0,
        //   3. clobber the user's scroll position on the inspect list.
        // User-visible: the viewer "snaps back" to the default angle
        // every few seconds AND the playback that was running stops.
        //
        // The fix: if the new data carries no new frames AND has the
        // same atom count + lattice (= same file, same parse state),
        // just refresh the per-frame metadata derived from runtime
        // info / parse warnings / runtime state markers — DON'T touch
        // the embed at all.  Anything that needs new data has nothing
        // to consume.
        const sameLength      = oldLen > 0 && newLen === oldLen;
        const sameLatticeNow  = _latticeEqual(
            oldData && oldData.lattice, r.data && r.data.lattice);
        const noNewContent    = sameAtomCount && sameLength && sameLatticeNow;
        if (noNewContent) {
            // Update mtime + run-state markers + parse-warnings list
            // (these can flip from "ongoing" → "finished" /
            // "errored" on a follow-up poll even with no new frames),
            // then bail before touching the model / animation /
            // plots.  Plots rebuilt only if the run-state changed
            // OR the SCF history for the in-flight step grew.
            //
            // 2026-06-17 Fix A: during a CG step, SCF iterations get
            // appended to ``scf_history[lastStep]`` without the frame
            // count, atom count, lattice, or run_state changing.  The
            // geometry-only guard above correctly preserves camera /
            // playback (the original 2026-06-12 fix), but conflating
            // "same geometry" with "nothing to plot" left the SCF
            // energy / dDmax / residual plots frozen until a new CG
            // frame landed.  ``_scfFingerprint`` adds the missing
            // axis: per-step iter count + last cycle's energy.
            const runStateChanged =
                oldData.run_state !== r.data.run_state
             || oldData.error_message !== r.data.error_message;
            const scfChanged =
                _scfFingerprint(oldData) !== _scfFingerprint(r.data);
            // Contract § 2: route fileState writes through
            // transition() so this function is no longer a fileState
            // writer in disguise.  noNewContent only updates the two
            // fields that can change on a same-content tick.
            transition("APPLY", { mtime: r.mtime, data: r.data });
            _renderRuntimeInfo(state.data.runtime_info);
            _renderParseWarnings(state.data.parse_warnings);
            if (runStateChanged || scfChanged) makePlots();
            return;
        }

        const wasAtEnd = !oldData || state.currentFrame >= oldLen - 1;

        // Contract § 2: full-rebuild path -- route the atomic
        // fileState replacement through transition('APPLY') so
        // transition() is the SINGLE entry-point for fileState
        // writes (closes deferred Gap #1).
        //
        // ``r.path`` is the server-resolved absolute path (the input
        // may have been a directory; r.path is the file actually
        // loaded — same value the live-poll URL uses).
        const resolvedFormat = r.format
            || (r.data && r.data.source_format) || "?";
        transition("APPLY", {
            mtime:  r.mtime,
            data:   r.data,
            format: resolvedFormat,
            label:  r.label || resolvedFormat,
            path:   r.path,  // undefined → keep existing per APPLY shape
        });
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
            // H2 2026-06-14 + I1 perf fix: the appended payload
            // may carry a newly-populated ``runtime_info.frozen_atoms``
            // that didn't exist on the prior poll's payload (e.g.
            // SIESTA's constraints echo only appears after the
            // first iteration, or a sidecar landed mid-run).  We
            // call applyHideFrozen() to re-render the atom list +
            // re-apply the embed overlay in one shot.  Pre-I1
            // H2 also called rebuildInspectAtomList() explicitly
            // here — but applyHideFrozen ALREADY calls it
            // unconditionally (per the G6 fix), so the explicit
            // call was a wasted double-rebuild on every appending
            // tick.  Single applyHideFrozen() is the correct shape.
            applyHideFrozen();
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
                window.molbuilder.constants.EVENT_INSPECTOR_READY,
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

    // Refresh-button listener wiring.  Wired ONCE at mount; not
    // re-wired on every load/transition.  Pre-PR-2.1 this lived
    // inside startPolling() which loadByPath called on every load
    // -- meaning the listener was re-attached every load and the
    // old one was only torn down on dispose, multiplying handlers.
    function _wireRefreshListener() {
        const C = (window.molbuilder || {}).constants;
        if (!C || !C.EVENT_REFRESH_REQUESTED) return;
        // Contract § 5: Refresh = file-switch with current path.
        // loadByPath -> transition('LOADING') runs the full reset
        // matrix (scfPollHistory clear, fetchSeq bump, etc.).
        const _onRefresh = () => {
            const p = state.fileState.path;
            if (!p) return;     // not yet loaded; nothing to refresh
            loadByPath(p);
        };
        document.addEventListener(C.EVENT_REFRESH_REQUESTED,
                                   _onRefresh);
        _cleanups.push(() => {
            document.removeEventListener(C.EVENT_REFRESH_REQUESTED,
                                          _onRefresh);
        });
    }

    function startPolling() {
        // Idempotent timer-start.  Called by transition('WATCHING').
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
        // Contract \u00a7 2: file-switch -> transition('LOADING').  This
        // single call runs the reset matrix (matrix \u00a7 3 row 1):
        //   * abort in-flight loadAbort + pollAbort
        //   * stop poll timer + clear pollInFlight
        //   * empty fileState (sets path = new path)
        //   * reset viewState (currentFrame=0, firstFit=true)
        //   * clear derived (scfPollHistory)
        //   * bump fetchSeq for the file-identity guard
        // Refresh button arrives here too -- same code path, same
        // resets.  Eliminates the half-refresh class.
        transition("LOADING", { path: path });
        const mySeq = state.lifecycle.fetchSeq;
        state.lifecycle.loadAbort = new AbortController();
        const signal = state.lifecycle.loadAbort.signal;
        try {
            const r = await fetch("/api/watch/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: path }),
                signal: signal,
            }).then(x => x.json());

            if (signal.aborted) return;
            // Contract § 4 Invariant 1 (file-identity guard): if a
            // newer transition('LOADING') ran while this fetch was
            // on the wire, fetchSeq has been bumped and our
            // response no longer corresponds to the current file.
            // Drop it; the new fetch is already in flight.
            if (state.lifecycle.fetchSeq !== mySeq) return;
            if (!r.ok) {
                setStatus(r.error || "Load failed.", "error");
                return;
            }
            applyNewData({
                mtime:  r.mtime,
                data:   r.data,
                format: r.format,
                label:  r.label,
                path:   r.path,
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
            // Contract § 2: transition to LOADED or WATCHING based
            // on run_state.  _settlePostLoad starts the poll timer
            // iff the run is ongoing (WATCHING branch) -- finished
            // files don't get polled.  Pre-PR-2.1 startPolling()
            // fired unconditionally; finished files would burn a
            // server request every 15 s forever.
            _settlePostLoad();
        } catch (e) {
            // AbortError fires when the user picks another file
            // (or dispose() runs) before this fetch completes.  Not
            // a failure -- the new load (or the dispose) is the
            // authoritative action; surfacing "Network error" here
            // would be misleading.
            if (e.name === "AbortError") return;
            setStatus("Network error: " + e.message, "error");
            transition("ERROR");
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

    /* Overlays — show-indices retired 2026-06-13 (now exclusively
     * the knob bar's Labels popover; see template note). */
    _on($("show-forces"),  "change", drawForces);
    _on($("force-scale"), "input", (e) => {
        $("force-scale-val").textContent = parseFloat(e.target.value).toFixed(1);
        drawForces();
    });
    _on($("force-min"),     "input",  drawForces);
    _on($("highlight-max"), "change", drawForces);
    // Hide-frozen-atoms checkbox (2026-06-13).  Toggling sends a
    // hide overlay to the embed + redraws arrows so the force
    // filter takes effect on the current frame too.
    _on($("hide-frozen"),   "change", function () {
        applyHideFrozen();
        drawForces();
    });

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

    /* ---- Export all plot data as CSV ---------------------------- */
    /* Bundles every column displayed across the 4 trajectory plots
       (energy + max forces + SCF cycle / energy / gnorm) into one
       CSV.  Header carries source attribution + parser + timestamp
       so the file is self-describing — the user can re-trace what
       it came from months later without external bookkeeping.
       Format pinned in lib/trajectory/csv-export.js; this handler
       just gathers state + triggers the browser download. */
    _on($("trajectory-export-csv-btn"), "click", function () {
        if (!state.data || !state.data.frames
                || state.data.frames.length === 0) {
            setStatus("No trajectory loaded — nothing to export.",
                      "error");
            return;
        }
        var fileStem = (state.label
                && state.label.replace(/[^A-Za-z0-9._-]+/g, "_"))
            || state.format
            || "trajectory";
        var csv = _buildPlotCsv({
            data:      state.data,
            sourcePath: state.path || "",
            format:    state.format || "",
            label:     state.label || "",
            mtime:     state.mtime || null,
        });
        var blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement("a");
        a.href     = url;
        a.download = fileStem + "_plots.csv";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setStatus("Exported " + a.download + ".", "ok");
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

    // Refresh-button listener wires here ONCE per mount.  Tears
    // down with the inspector's dispose path via _cleanups.
    // Pre-PR-2.1 this lived inside startPolling() which loadByPath
    // called on every load -- so the listener was re-attached each
    // load and the old ones piled up until dispose.  PR 2.1 audit
    // follow-up makes the wiring strictly per-mount.
    _wireRefreshListener();

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
            // Contract § 2: dispose -> transition('IDLE').  The
            // matrix § 3 row "dispose / unmount" runs the full
            // reset (aborts, timer stop, bucket clears).
            transition("IDLE");
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
        // Exported so tests/test_trajectory_csv_redaction_js.py can
        // pin the redaction patterns without driving a browser.
        // Not part of the inspector's public API; tests-only.
        _redactSourcePath: _redactSourcePath,
    };

})(typeof window !== "undefined" ? window : this);
