/* Trajectory inspector core -- 3Dmol viewport + Plotly traces.
 *
 * THE shared trajectory-inspector implementation.  Mounted by the
 * /results tab via the registry adapter (lib/inspectors/trajectory.js),
 * which fetches _trajectory_inspector.html from
 * GET /partials/trajectory-inspector and calls this module's exported
 * ``mount(host, opts)`` against the resulting host element.
 *
 * --- How the viewer + plots stay fast --------------------------------
 * Frames are loaded once into a 3Dmol "movie" model (addModelsAsFrames)
 * and the slider / playback simply calls viewer.setCurrentFrame(idx), which is
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

import { mount } from "/static/lib/molview/index.js";
import { molviewFiles } from "../projects/molview-doors.js";
(function (root) {
    "use strict";

    /* The run states the server sends, spelled once.
     *
     * The vocabulary is the parser's, defined at
     * parse/engines/_helpers.py:181 and passed through the watch endpoint
     * unchanged (web/blueprints/watch.py:728):
     *
     *     "running" | "ended" | "stopped" | "out_of_memory" | "unknown"
     *
     * -- HOW THE RUN ENDED (`model/parse.md` § 2b), not a grade.  A run
     * that reached its end without converging is "ended"; whether the
     * SCF converged rides beside it in `scf_converged`.
     *
     * The FILE's ending -- read only for a file that belongs to no run (an
     * upload), and for the reason a stopped output states.  How the RUN is
     * doing is a different field: the STATUS state
     * ("pending"|"queued"|"running"|"failed"|"finished", the one door,
     * parse/dirs/job.py), which the server sends with the file as `run` and
     * which the badge and the follow read (`fileState.run`,
     * web/results.md § 4.1). */
    const RUN_STATE = Object.freeze({
        RUNNING:  "running",
        ENDED:    "ended",
        STOPPED:  "stopped",
        OOM:      "out_of_memory",
        UNKNOWN:  "unknown",
    });

    /* THE VIEWER IS THE ONE THIS MODULE MOUNTED, and it is reached through the
     * handle that mounting returned -- `_mv.data` (molview.md § 5.6: a viewer
     * belongs to whoever mounted it; there is no registry to look one up in). */

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
     *                listeners (pageshow).  After dispose()
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
        // POSIX home (Linux + macOS).  ``/home/u/foo`` ->
        // ``~/foo``; ``/Users/u/foo`` -> ``~/foo``.  The
        // ``[^/]+`` segment is the username.
        p = p.replace(/^\/(home|Users)\/[^/]+\//, "~/");
        // POSIX tmp dirs with embedded username.  pytest writes
        // ``/tmp/pytest-of-u/...``; the username-bearing first
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

    // Poll cadence: 15 s.  Overlapping ticks SKIP (the in-flight guard in
    // ``pollOnce``) rather than abort a TLS connection mid-response.
    // Users watching a live SIESTA SCF run see iterations complete in
    // 13-24 s on the standard workstation, so a slower poll lets
    // iterations slip between polls and the plot lags the on-disk state.
    const POLL_MS = 15000;

    // Inside-partial lookup: scoped to the inspector's root element
    // (i.e., the #inspector-host on /results after the adapter has
    // injected _trajectory_inspector.html).
    const $ = (id) => rootEl.querySelector("#" + id);

    // ----- Listener bookkeeping ---------------------------------
    //
    // Every element-level addEventListener inside mountInspector
    // goes through _on(), which registers the teardown at the same moment
    // as the registration; non-listener teardowns (a ResizeObserver to
    // disconnect) go through _listeners.defer().  ONE registry, drained by
    // dispose() in one call, then the per-resource teardowns (polling
    // timer, in-flight HTTP request, embed handle).
    //
    // The window-level listener (``pageshow``) MUST be tracked here -- it survives
    // the host's innerHTML clear, so without explicit removal
    // every /results mount→dispose→mount cycle would accumulate
    // them.  Element listeners on partial-declared ids are GC'd
    // with their nodes; tracking them is defensive (and matches
    // the spectra core's dispose contract for cross-inspector
    // consistency).
    //
    // THE SCOPE IS THE ONLY REGISTRY.
    var _listeners = root.molbuilder.inspectorLifecycle.listeners();
    function _on(target, event, handler, opts) {
        _listeners.on(target, event, handler, opts);
    }

    /* ------------------------------------------------------------------ */
    /*  State                                                              */
    /* ------------------------------------------------------------------ */

    // Bucketed state shape per docs/web/results.md.
    // The contract partitions inspector state into five disjoint
    // buckets with explicit reset semantics:
    //
    //   fileState  -- replaced atomically each LOADING -> LOADED
    //                 (path, mtime, format, label, parsed data)
    //   viewState  -- per-file user interaction (firstFit, picks; NOT the playhead --
    //                 MolView owns that)
    //   uiPrefs    -- per-session knobs (hideFrozen etc.)
    //   lifecycle  -- controllers + timers (poll timer, abort controllers)
    //   derived    -- recomputed from fileState (none today)
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
            //   scf_history, wall_clock_s, elapsed_s, in_progress, run_state,
            //   error_message, runtime_info, parse_warnings, ...}
            data:   null,
            // Per-atom metadata (region labels / frozen tags / annotation
            // channels) the Build tab embedded in this run's input script,
            // recovered by /api/watch/load as the trusted ATOM-METADATA
            // block (a JSON STRING).  The server folds it into the envelope
            // the tab installs; this copy is kept per file (survives polls
            // of the same file; cleared on a fresh load like path/label).
            // null = run had no ATOM-METADATA block.
            atomMetadata: null,
            // The run's periodicity, COMPOSED ON THE SERVER (watch.py: the
            // run's cell, the axis kinds from the run deck's ENGINE-OFFSET
            // record, the engine's origin stated).  The structure arrives in
            // the server's envelope, periodicity included; this copy is read
            // only to say where the box came from (`_cellCameFromTheRun`).
            // null = the run knows nothing.  Same per-file lifecycle as
            // atomMetadata.
            periodicity: null,
            // What this run says ABOUT itself: the free `info` store
            // (molview.md § 8.4a), composed by the server from the run
            // directory -- today the contract its deck records, as
            // `info.calculation`.  THE TAB PROVIDES IT (user, 2026-08-30):
            // MolView has no idea what describes a run.  HELD here, beside
            // its two neighbours, because this viewer is REBUILT on every
            // poll -- the store is re-supplied to each installMolecule or
            // it lasts one tick.  null = the run said nothing.
            info: null,
            // HOW THE RUN THIS FILE BELONGS TO IS DOING -- `{state, detail,
            // live}`, the one door's answer, which the server sends with the
            // file (web/results.md § 4.1).  The viewer follows while `live`,
            // and the badge reads it.  null = the file belongs to no run (an
            // upload).  Keep-on-undefined, like the three above: a poll with
            // new content leaves it out.
            run: null,
        },

        viewState: {
            // NO currentFrame here.  MolView owns the playhead; a tab-side copy could
            // only ever be a stale second answer.  The one place that needs the shown frame asks
            // molview.data.currentFrame() at the moment it needs it.
            firstFit:     true,
        },

        // uiPrefs: contract § 3 reserves this bucket for per-session
        // knobs (sessionStorage-persisted, survives file-switch).
        // Trajectory has NO fields here today: playback speed/loop are
        // knobs on MolView's frame-controls bar.  When/if a
        // trajectory-only pref appears (e.g. a
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
            // the file it was issued for) before applying.  Late responses
            // from a prior file can never write into the current
            // file's view.
        },

        derived: {
            // Empty -- the rate the SCF line shows is the timing
            // instrument's, carried in the file's own data, so nothing is
            // derived here.  Kept
            // present-but-empty, as spectra's is, so the five-bucket shape
            // holds.
        },
    };

    // Backward-compat aliases.  Render code throughout this file
    // still reads/writes the legacy flat shape (state.mtime,
    // state.data, state.firstFit, ...).  The
    // aliases route through to the bucketed canonical home so the
    // entire body of render code keeps working unchanged; the
    // contract's reset matrix is enforced at transition() and at
    // the four documented external entry-points (loadByPath,
    // Refresh, pollOnce result, dispose).
    (function _wireBackcompatAliases() {
        // The shared inspector helper (lib/inspectors/lifecycle.js).
        function alias(key, bucket) {
            root.molbuilder.inspectorLifecycle.alias(state, key, bucket);
        }
        alias("path",         "fileState");
        alias("mtime",        "fileState");
        alias("format",       "fileState");
        alias("label",        "fileState");
        alias("data",         "fileState");
        alias("atomMetadata", "fileState");
        alias("periodicity",  "fileState");
        alias("info",         "fileState");
        alias("structure",    "fileState");
        alias("firstFit",     "viewState");
        alias("pollTimer",    "lifecycle");
        alias("pollInFlight", "lifecycle");
        alias("loadAbort",    "lifecycle");
        alias("pollAbort",    "lifecycle");
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
    //   -> LOADING: empty fileState, reset viewState (firstFit=true),
    //               keep uiPrefs, abort + clear all
    //               in-flight controllers, clear derived.
    //   -> IDLE:    clear fileState, viewState, derived; abort + clear
    //               all controllers; stop poll timer.
    function transition(target, payload) {
        payload = payload || {};
        if (target === "LOADING") {
            // Abort all in-flight requests.  Their .then() handlers
            // still fire, but the file-identity guard (Invariant 1)
            // catches them via the path guard at resolution.
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
            state.fileState.atomMetadata = null;
            state.fileState.periodicity  = null;
            state.fileState.info         = null;
            state.fileState.structure    = null;
            state.fileState.run          = null;
            // Reset viewState per matrix: refit the camera on the next render.  The
            // playhead is NOT reset here -- MolView owns it, and a fresh load resets it
            // there (setData lands on frame 0).
            state.viewState.firstFit     = true;
            // (No sequence counter: the answer carries its own path, and
            // every guard compares against `fileState.path` below.)
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
            state.fileState.atomMetadata = null;
            state.fileState.periodicity  = null;
            state.fileState.info         = null;
            state.fileState.structure    = null;
            state.fileState.run          = null;
            state.viewState.firstFit     = true;
            state.machine = "IDLE";
            return;
        }
        if (target === "LOADED") {
            // The run is no longer live -- finished, or never launched --
            // or the file belongs to no run (`_settlePostLoad`).  Stop the
            // poll timer if running; clear pollInFlight.
            stopPolling();
            state.lifecycle.pollInFlight = false;
            state.machine = "LOADED";
            return;
        }
        if (target === "WATCHING") {
            // The run is live.  START the poll timer; startPolling() is
            // idempotent, so re-entering WATCHING on a mid-poll tick is a
            // no-op for the timer.
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
            // applyNewData does not write fileState fields via the
            // backward-compat aliases; it routes here.  Single canonical
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
            //
            // And the DATA AND THE NAME BELONG TO EACH OTHER: a tick fired
            // for file A can resolve after the user has moved to B.
            //
            // The server already answers the question -- every reply
            // carries `r.path`, the file it actually read -- so the answer
            // is matched against the file on screen HERE, once, and a
            // reply for a file we have moved off is dropped.
            if (payload.path === undefined) {
                throw new Error(
                    "APPLY without a path: data must be written with the "
                    + "file it came from, never onto the current one");
            }
            /* THE ANSWER TO A LOAD NAMES THE FILE.  A POLL MUST NAME THE
             * ONE WE ALREADY HAVE.  Those are different questions.
             *
             * `/api/watch/load` answers with the file the SERVER read after
             * `_resolve_within_roots`: `~` expanded, symlinks followed,
             * `.`/`//`/trailing slash normalised, a DIRECTORY replaced by
             * the newest log inside it, an upload replaced by its /tmp
             * path -- not necessarily what `transition("LOADING")` stored as
             * ASKED for.
             *
             * `state.machine` is the discriminator, and it is exact:
             * `transition("LOADING")` set it and nothing else runs before
             * the answer lands, while a poll only ever fires from WATCHING
             * or LOADED.  So a load ADOPTS the server's name -- which is
             * the authoritative one, and is what the `resolved_from`
             * status message downstream exists to explain -- and a poll,
             * comparing resolved against resolved, still cannot paint one
             * file's frames under another's name. */
            /* NO `path !== null` EXEMPTION: `transition("IDLE")` -- dispose
             * -- sets the path to null, and an answer arriving after the
             * inspector was torn down must not reach a dead panel. */
            if (state.machine !== "LOADING"
                && payload.path !== state.fileState.path) {
                return;          // a POLL answer for a file we are not showing
            }
            if (payload.mtime  !== undefined) state.fileState.mtime  = payload.mtime;
            if (payload.data   !== undefined) state.fileState.data   = payload.data;
            if (payload.format !== undefined) state.fileState.format = payload.format;
            if (payload.label  !== undefined) state.fileState.label  = payload.label;
            if (payload.path   !== undefined) state.fileState.path   = payload.path;
            // Per-file: only the LOAD carries it (undefined on watch ticks →
            // keep existing, so live polls of the same file don't drop the
            // metadata the initial load recovered).
            if (payload.atomMetadata !== undefined)
                state.fileState.atomMetadata = payload.atomMetadata;
            if (payload.periodicity !== undefined)
                state.fileState.periodicity = payload.periodicity;
            if (payload.info !== undefined)
                state.fileState.info = payload.info;
            // Frame 0 as an ENVELOPE -- what the viewer installs.  Same
            // keep-on-undefined rule as the three above, so a watch tick that
            // re-sends frames does not drop it.
            if (payload.structure !== undefined)
                state.fileState.structure = payload.structure;
            // How the run is doing: on the load, and on a poll whose file
            // had nothing new; a poll with new content leaves it out.
            if (payload.run !== undefined)
                state.fileState.run = payload.run;
            return;
        }
        // Unknown target: silent no-op.  Future targets (the
        // contract reserves room for sub-states) land here.
    }

    // After fileState has been written -- a load, or any poll -- move to
    // the state THE RUN calls for: the one door's answer the server sends
    // with the file (`run`, web/results.md § 4.1), never the file's own
    // ending.  Follow while the run is live; a failed run stops as a stopped
    // file always did (ERROR keeps the last data on screen); a finished run,
    // one never launched, and a file that belongs to no run -- an upload --
    // have nothing more to bring.
    function _settlePostLoad() {
        const run = state.fileState.run;
        if (run && run.live) {
            transition("WATCHING");
            return;
        }
        const failed = run ? run.state === "failed" : _fileStopped();
        transition(failed ? "ERROR" : "LOADED");
    }

    // A file read alone -- one that belongs to no run -- stopped by its own
    // ending: a fatal marker, or out of memory.
    function _fileStopped() {
        const rs = state.fileState.data && state.fileState.data.run_state;
        return rs === RUN_STATE.STOPPED || rs === RUN_STATE.OOM;
    }

    // (The render functions close over ``state`` directly; applyNewData
    // runs synchronously and calls them before yielding to the event loop,
    // so there is no observable "stale-during-tick" race.)

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

    // The trajectory inspector mounts the FULL concealed MolView module
    // read-only (web/molview.md § 4, § 8) into the empty #viewer-host and becomes
    // a DATA FEEDER — it hands MolView the parsed coordinate frames + raw
    // per-frame forces (molview.data.reloadFrames / addFrames / setForces; the
    // ENGINE builds + styles the arrows -- § 9.2: a host never hands in a
    // finished appearance).  MolView owns the ENTIRE
    // view: playback + speed +
    // loop (the frame bar), unit-cell display, atom-index labels, selection +
    // measurement + picking, and atom hiding (its render pipeline's
    // isolate/selection).  See docs/web/overview.md
    //
    // mv.mount is async, but the partial-factory calls THIS mount synchronously
    // (it returns {dispose, load}); so we kick the assembly off here and hold
    // the handle in ``_mv`` once ready.  Loads that arrive before the handle
    // resolves are safe: applyNewData sets state.data + the plots regardless,
    // and rebuildModel() awaits ``_mvReady`` before feeding frames.
    let _mv = null;
    /* The one route to this viewer's data (molview.md § 9.3). Read through the
     * handle every time rather than caching `_mv.data`: the handle is null until
     * the mount below resolves, and a cached null would outlive it. */
    const _mvdata = () => (_mv && _mv.ok) ? _mv.data : null;
    const _mvReady = (async function mountMolView() {
        const mb = window.molbuilder || {};
        const ws = mb.workspace;
        const host = $("viewer-host");
        /* NOT GATED ON A VIEWER EXISTING -- this is what creates one. */
        if (!host || typeof mount !== "function" || !ws) {
            setStatus("Viewer unavailable: the MolView module / persistence "
                    + "layer is missing from results.html.", "error");
            return null;
        }
        try {
            _mv = await mount(host, ws, {
                mode:  "readonly",
                owner: "results:trajectory",
                files: molviewFiles,
            });
        } catch (e) {
            setStatus("Viewer failed: "
                + (e && e.message ? e.message : String(e)), "error");
            return null;
        }
        if (!_mv || !_mv.ok) {
            setStatus("Viewer failed: "
                + ((_mv && _mv.error) || "molview.mount failed."), "error");
            return null;
        }
        // Test hook: expose the mount handle so Playwright e2e can drive the
        // read-only trajectory view.
        host.__molview_results_handle = _mv;
        // NOTE: force arrows are NOT re-derived per frame.  The inspector hands the ENGINE
        // the filtered per-frame forces ONCE (buildForcesPerFrame) on a filter-knob change;
        // the engine bakes + styles the arrows into the native animation, so playback draws
        // frame t's arrows with zero per-frame synthesis.
        return _mv;
    })();

    /* ------------------------------------------------------------------ */
    /*  Status banner                                                      */
    /* ------------------------------------------------------------------ */

    /** WHY A STOPPED RUN STOPPED, in the server's words: the file's cause
     * as the one ending reader states it, worded by the SIESTA family's
     * table (`siesta_grammar.CAUSE_WORDS`, `model/parse.md` § 2b) -- the
     * browser keeps no copy of the markers.  Without one, the parser's own
     * message, trimmed so the badge stays a line.  ``null`` when there is
     * neither. */
    function _stopReason(data, errMsg) {
        if (data && data.stop_reason) return data.stop_reason;
        if (!errMsg) return null;
        const s = String(errMsg).trim();
        return s.length > 120 ? s.slice(0, 117) + "..." : s;
    }

    function setStatus(msg, kind) {
        // The inspector's own status line (`_trajectory_inspector.html`
        // #trajectory-status): a refused load, a viewer that could not
        // mount, the cell's provenance, an export.  A mount without the
        // partial has none, and this returns.
        if (!document.getElementById("trajectory-status")) return;
        window.molbuilder.status.set("trajectory-status", msg, kind);
    }

    /* THE LOAD HAS ENDED -- drawn, or refused (the shared inspector door,
     * lib/inspectors/lifecycle.js). */
    function _announceReady(detail) {
        window.molbuilder.inspectorLifecycle.announceReady("trajectory", detail);
    }

    /* ------------------------------------------------------------------ */
    /*  3Dmol rendering                                                    */
    /* ------------------------------------------------------------------ */

    /**
     * Build a CSV bundling every column drawn across the
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
     *   scf_cycle, scf_cycle_energy_eV, scf_cycle_gnorm_eV
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
        lines.push("#   scf_cycle_gnorm_eV            — SCF orbital-gradient norm at that cycle (eV);");
        lines.push("#                                   PySCF only; SIESTA emits dHmax instead.");
        lines.push("#");
        lines.push("# Empty cells mean the engine didn't emit that value at that step.");
        lines.push("#");
        lines.push([
            "step", "energy_eV",
            "max_force_eVperA", "max_force_constrained_eVperA",
            "scf_cycle", "scf_cycle_energy_eV", "scf_cycle_gnorm_eV",
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

    // Style, unit-cell display, atom-index labels, and background are all
    // MolView's (its Cell page + knob bar): the periodicity on the envelope
    // drives the cell box, and the knob bar's Labels popover owns index labels.

    // Frozen-atom indices from runtime_info.frozen_atoms -- as the run's own
    // output states them (model/parse.md 5.3).  Returns a Set<number> for O(1)
    // membership; null when it states none.  Used ONLY to filter the force-arrow
    // overlay now — atom HIDING in the viewer is MolView's job (its
    // selection/isolate render pipeline), not this inspector's.
    function _frozenSet() {
        const rt = state.data && state.data.runtime_info;
        const arr = rt && rt.frozen_atoms;
        if (!Array.isArray(arr) || !arr.length) return null;
        return new Set(arr);
    }

    // (There is no local "which frame is showing?" helper.  MolView owns the playhead and the
    // tab ASKS it at the one place that needs it -- through molview.data, the single door this
    // file uses for every frame read and write.  A tab-side reader would be a second answer to
    // a question that already has an owner, and the tab keeps no copy of the playhead.)

    // Build the per-atom force vectors for ONE frame under the current FILTER knobs.
    // Returns a per-atom array (ORIGINAL atom order) of [fx,fy,fz]; a SUPPRESSED atom is
    // zeroed -- frozen atoms when "hide frozen" is on (their forces are constraint-balancing
    // artefacts, not physical free-atom forces) and sub-threshold magnitudes -- which the
    // engine renders as NO arrow.  This decides WHICH forces show; the
    // ENGINE owns the styling (gold max-highlight, magnitude colour/radius ramp) and the
    // scale (the forceScale flag).  null when the parser captured no forces for this frame.
    function _buildForcesForFrame(frameIdx) {
        if (!state.data) return null;
        const forces = state.data.forces && state.data.forces[frameIdx];
        if (!forces || !forces.length) return null;
        const fmin = parseFloat($("force-min").value) || 0.0;
        const hideFrozen = $("hide-frozen") && $("hide-frozen").checked;
        const frozen = hideFrozen ? _frozenSet() : null;
        return forces.map(function (f, i) {
            if (frozen && frozen.has(i)) return [0, 0, 0];
            const mag = Math.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
            if (mag < fmin) return [0, 0, 0];
            return [f[0], f[1], f[2]];
        });
    }

    /* ------------------------------------------------------------------ */
    /*  Force-vector overlay                                               */
    /* ------------------------------------------------------------------ */
    //
    // The ENGINE builds + styles the force arrows from raw per-frame forces:
    // the inspector hands FILTERED forces (below) + drives the
    // forceScale flag, and MolView owns the gold max-highlight, the magnitude
    // colour/radius ramp, and WHETHER they're drawn (its "show overlay" toggle).

    // The FULL per-frame force set: forcesPerFrame[t] = frame t's filtered per-atom
    // vectors.  O(frames × atoms), rebuilt only when a FILTER knob changes (threshold /
    // hide-frozen) -- scale is a cheap flag, never a rebuild.  null when no forces exist.
    function buildForcesPerFrame() {
        if (!state.data || !Array.isArray(state.data.frames)) return null;
        const fpf = new Array(state.data.frames.length);
        let any = false;
        for (let t = 0; t < state.data.frames.length; t++) {
            fpf[t] = _buildForcesForFrame(t);
            if (fpf[t]) any = true;
        }
        return any ? fpf : null;
    }

    // Re-hand the filtered per-frame forces to MolView after a FILTER knob change
    // (threshold / hide-frozen).  setForces re-bakes the arrow overlay IN PLACE -- no movie
    // reload -- preserving the overlay's on/off visibility.  Scale is separate (the cheap
    // forceScale flag), so it never routes here.
    function drawForces() {
        const d = _mvdata();
        if (d) {
            d.setForces(buildForcesPerFrame());
        }
    }

    // Feed the parsed trajectory to MolView, in ONE call: frame 0 as the
    // envelope that establishes atom identity, every frame beside it, the filtered
    // forces, and the labels the run carried.  MolView's frame bar appears
    // (frameCount > 1) and it owns playback / speed / loop / cell / labels /
    // selection from there.  Async because the load round-trips through the
    // server; awaits _mvReady so it never runs before the mount handle exists.
    // Optional ``seekIdx`` jumps to a frame afterwards (used to keep the
    // playhead near the tail when a live poll forces a full rebuild).
    //
    // THE LATTICE THE RUN REPORTED is handed over as `periodicity` (user
    // decision, 2026-08-03: show the box, mark it as the run's).
    //
    // The cost the user accepted: an isolated molecule that ran in a large
    // SIESTA box gets a box drawn round it -- so the tab says where the box
    // came from (below).
    /* Whether the box on screen came from the run rather than from the user --
     * a fact about the FILE OPERATION this tab performed, so this tab keeps it
     * (molview.md § 6.7: the viewer tracks contents, not where they came from).
     * Read by the status line after a load. */
    function _cellCameFromTheRun() {
        const per = state.periodicity;
        return !!(per && Array.isArray(per.cell) && per.cell.length === 3);
    }

    async function rebuildModel(seekIdx) {
        await _mvReady;
        if (!_mv || !_mvdata() || !state.data
                || !state.data.frames || !state.data.frames.length) {
            return;
        }
        const allFrames = state.data.frames;
        /* FRAME 0 ARRIVES AS AN ENVELOPE, assembled by the server
         * (`watch.py::_frame0_structure`) out of the pieces it already had:
         * the frames from the parsed logs, the labels from the run's input
         * script, the box from its output logs.  `web-api.md` § 1: the
         * browser sends what it holds, and never a document it wrote. */
        const frame0 = state.structure || null;
        if (!frame0) {
            /* The server assembles this from the frames it parsed, so a run
             * with frames and no envelope means the assembly itself failed.
             * Say so rather than writing a document here. */
            setStatus("This run's geometry could not be assembled by the "
                + "server; nothing to show in the viewer.", "error");
            return;
        }
        const coordFrames = allFrames.map(function (frame) {
            return frame.map(function (atom) {
                return [atom[1], atom[2], atom[3]];
            });
        });

        /* Force scale (Å per force unit) is one of the SWITCHES that sit beside
         * the selection (molview.md § 9.5) -- `setSwitch`, the same door as
         * isolate / showForces / showCell.  Set BEFORE the load, so the first
         * arrow bake already uses the right length. */
        const _fscaleEl = $("force-scale");
        if (_fscaleEl) {
            _mvdata().selection.setSwitch(
                "forceScale", parseFloat(_fscaleEl.value) || 1.0);
        }

        /* THE WHOLE RUN GOES IN ONE CALL (molview.md § 9.3).
         *
         * Frame 0 first and the rest after would be the shape the contract
         * names as broken: a subscriber would see a single-frame structure
         * that never existed (§ 6.4), and point 0 would anchor on that one
         * frame -- so a Retract would throw the trajectory away (§ 11.2).
         *
         * The labels ride along the same way: `frame0` is the server's
         * envelope, so the region / frozen / annotation block the Build tab
         * wrote into the input script, the cell and the axis kinds are
         * already on it (/api/watch/load). */
        try {
            await _mvdata().installMolecule({
                structure:    frame0,
                // THE FILE IT CAME FROM names it (`web/molview.md`: an export
                // "writes mine.xyz" for mine.xyz).
                filename:     state.path || "",
                frames:       coordFrames,
                forces:       buildForcesPerFrame(),
            });
        } catch (e) {
            setStatus("Viewer failed to load the run: "
                + (e && e.message ? e.message : String(e)), "error");
            return;
        }
        if (typeof seekIdx === "number"
                && seekIdx > 0 && seekIdx < coordFrames.length) {
            /* `setCurrentFrame` is the viewer's seek. */
            try { _mvdata().setCurrentFrame(seekIdx); } catch (_) {}
        }
    }

    // (No local showFrame(): seeking is `data.setCurrentFrame(i)`, which already range-checks
    // against the frames MolView holds.  A tab-side clamp would re-derive that range from a
    // second count.)

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
    // documentation in docs/web/results.md.
    function _themeColors() {
        const cs = getComputedStyle(document.documentElement);
        const get = (name, fb) =>
            (cs.getPropertyValue(name) || "").trim() || fb;
        return {
            // Token-driven (theme-responsive)
            accent:     get("--accent",     "#6ba6ff"),
            success:    get("--success",    "#4ade80"),
            warnSoft:   get("--warn-soft",  "#d8a64b"),
            textMuted:  get("--text-muted", "#959ba7"),
            // Non-themable plot conventions
            forceAllAtoms: "#d62728",   // red — "all atoms" informational trace
            energy:        "#1f77b4",   // blue — single-trace
            scfEnergy:     "#1f77b4",   // blue — single-trace
            scfGnorm:      "#fb923c",   // orange — moved off green to keep
                                        //          the threshold-line green
                                        //          unambiguous
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
    //     {max_force_tol_eV_per_A: 0.023, max_scf_iter: 100, ...,
    //      source: "molwatch_header"}
    //     -> returned as-is.
    //
    //   NESTED (staged runs, task #534) — keys are the stages' artifact
    //   TOKENS, digit-first (job-contracts.md 6.3), so they are NOT
    //   identifiers and always need quoting:
    //     {"01_coarse": {max_force_tol_eV_per_A: ..., ...},
    //      "02_tight":  {max_force_tol_eV_per_A: ..., ...}, ...,
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

    // THE SCF'S CRITERIA -- `{phase: {residual: {tolerance, unit,
    // required}}}`, one structure for every engine (web/trajectory.md § 3)
    // -- or `{}` when the run states none.
    function _scfCriteria() {
        const rt = state.data && state.data.runtime_info;
        const crit = rt && rt.scf_criteria;
        return (crit && typeof crit === "object") ? crit : {};
    }

    // Render the convergence-summary section above the plots row.
    // Shape + intent documented in docs/web/results.md
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
        // THE SCF'S CRITERIA, every engine's in one shape
        // (`runtime_info.scf_criteria`, web/trajectory.md § 3): one row per
        // residual the SCF had to bring below its tolerance, in that
        // residual's own unit.  The phase is named only when the run has
        // two (a TranSIESTA device).
        const crit = _scfCriteria();
        const critPhases = Object.keys(crit);
        for (const ph of critPhases) {
            for (const [res, c] of Object.entries(crit[ph] || {})) {
                if (!c || typeof c.tolerance !== "number") continue;
                rows.push(["SCF " + res + " tol"
                           + (critPhases.length > 1 ? " (" + ph + ")" : ""),
                    c.tolerance.toExponential(2) + (c.unit ? " " + c.unit : "")
                    + (c.required === false ? " (not required)" : "")]);
            }
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


    /**
     * Render the pre-data status banner that explains a blank
     * energy/force plot when the engine hasn't yet written a first
     * data point.  Engine-agnostic: shows "Initializing — first
     * energy estimate pending" for SIESTA, "Optimizing — first
     * step pending" for PySCF, or hides itself once there's
     * usable data.
     *
     * Shown so a freshly-started run doesn't look like
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
                // WHICH ENGINE, so `state.format` -- the same field the
                // SCF banner reads -- and not `data.source_format`, which
                // is a FORMAT ("molwatch", "siesta-mdnc").
                const fmt = state.format || "";
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
        }, { displayModeBar: false, responsive: true });

        // Two traces when the engine reports both.
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
        // in that case we render the single trace.
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
        const forceShapes = [];
        const forceAnnotations = [];
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
        resizePlots();
    }

    /* THE PLOTS ARE TOLD THEIR SIZE; they never work it out themselves.
     *
     * Plotly draws a fixed-width SVG and `responsive: true` only re-draws on
     * WINDOW resize.  Every other way this card changes width leaves the SVG
     * frozen at whatever it measured at draw time:
     *
     *   * the SCF plots appearing / disappearing reflows the grid from 2 cells
     *     to 4, so each cell halves;
     *   * folding the Projects sidebar changes the column width with no window
     *     resize at all -- `projects-sidebar.js` toggles a body class and
     *     notifies nobody.
     *
     * Measured 2026-08-05 on a live run: drawn folded the SVG was 283 px; after
     * unfolding the container was 550 px and the SVG was still 283.  Drawn
     * unfolded and then folded, the same 267 px goes the other way and the
     * plots overflow their cells.
     *
     * `Plotly.Plots.resize` is a no-op when the size did not actually change,
     * so this is cheap to call unconditionally. */
    const PLOT_IDS = ["energy-plot", "force-plot",
                      "scf-energy-plot", "scf-gnorm-plot"];

    function resizePlots() {
        PLOT_IDS.forEach((id) => {
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
        // Local theme lookup so this can be reasoned about in isolation
        // (one getComputedStyle); the criteria are read where the line is.
        const theme = _themeColors();
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
        // only green element on the plot.
        let residual, residualName, residualUnit;
        const residualColor = theme.scfGnorm;
        if (current[0].gnorm !== undefined) {
            residual      = current.map(c => c.gnorm);
            residualName  = "|g|";
            residualUnit  = "eV";     // an energy: dE over orbital rotations
        } else if (current[0].dHmax !== undefined) {
            residual      = current.map(c => c.dHmax);
            residualName  = "dHmax";
            residualUnit  = "eV";
        } else {
            residual = null;
        }

        // Engine-aware labels.  state.format is WHICH ENGINE RAN --
        // "siesta", "pyscf", or "unknown" -- decided on the server by
        // `parse.contract.engine_of` from what the run directory
        // declares about itself (`running-a-job.md` 4.2).  It is NOT
        // the parser's name and NOT source_format.
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

        /* THE RATE IS THE SCF-TIMING INSTRUMENT'S (`model/parse.md` § 2a P-T4,
         * § 5c): the server reads the run's timing log through the one reader
         * the run record reads it by (`parse.dirs.record.scf_timing_of`), and
         * this shows the figure for the phase the current row is in.  Nothing
         * here computes one.  Each engine's rows are stamped -- the
         * SIESTA tee, a PySCF deck's progress log -- and timed by one rule; an
         * output read alone has none, and no rate: not stated, so not shown. */
        const timing = (state.data && state.data.scf_timing) || null;
        if (timing) {
            const phase = current[current.length - 1].phase;
            const of = (key) => (phase && Number.isFinite(timing[key + "_" + phase]))
                ? timing[key + "_" + phase] : timing[key];
            const perIter = of("s_per_iter");
            const timed = of("iters_measured");
            if (Number.isFinite(perIter)) {
                statusText += ", " + (+perIter.toPrecision(3)) + " s/iter"
                    + (Number.isFinite(timed) ? " (" + timed + " timed)" : "");
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
            // THE PLOTTED RESIDUAL'S OWN CRITERION, in its phase
            // (web/trajectory.md § 3): SIESTA's rows state their phase
            // (`periodic`, `negf`); rows that state none are the run's one
            // phase.  A line only where the run requires the criterion.
            const scfShapes = [];
            const scfAnnotations = [];
            const crit = _scfCriteria();
            const critPhases = Object.keys(crit);
            const rowPhase = current[current.length - 1].phase
                || (critPhases.length === 1 ? critPhases[0] : null);
            const c = rowPhase && crit[rowPhase]
                ? crit[rowPhase][residualName] : null;
            const scfTol = (c && typeof c.tolerance === "number"
                            && c.required !== false) ? c.tolerance : null;
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
        // In-flight guard: if the previous tick's request is still on
        // the wire, SKIP this tick rather than abort it -- an abort
        // tears down the TLS connection mid-response.  The legitimate
        // cancellation path -- the user's dispose() or a file-change
        // supersede -- still aborts via the AbortController held in
        // ``state.pollAbort``.
        //
        // A slow response is not stale: ``state.mtime`` is the query
        // param and the server returns ``changed: false`` for a
        // matched mtime.
        if (state.pollInFlight) {
            return;
        }
        state.lifecycle.pollAbort = new AbortController();
        const signal = state.lifecycle.pollAbort.signal;
        state.lifecycle.pollInFlight = true;
        // THE FILE THIS POLL IS FOR, read once and carried.
        // `/api/watch/data` takes no path -- it answers for whatever the
        // server currently holds -- so the question "is this answer still
        // ours?" can only be asked of the file we were showing when the
        // poll went out.
        const myPath = state.fileState.path;
        try {
            const url = state.mtime !== null
                ? "/api/watch/data?mtime=" + encodeURIComponent(state.mtime)
                : "/api/watch/data";
            const r = await fetch(url, { signal: signal })
                .then(x => x.json());
            if (signal.aborted) return;
            if (myPath !== state.fileState.path) return;
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
                // Nothing new in the file: the news is how the run is
                // doing, which the server sends on such a tick.
                if (r.run !== undefined) {
                    transition("APPLY", { path: myPath, run: r.run });
                    _renderBadge();
                }
                _settlePostLoad();
                return;
            }
            applyNewData(r);
            // A tick with new content: the same settle, by the run's
            // state (kept from the last answer that carried it, or this
            // one's when the run ended with this write).
            _settlePostLoad();
        } catch (e) {
            // AbortError: dispose() or a file-change supersede ran.
            // Silent -- the next tick (or the unmount) is the
            // authoritative state.
            if (e.name === "AbortError") return;
            setStatus("Network error: " + e.message, "error");
        } finally {
            // Always release the in-flight flag so the NEXT tick
            // (POLL_MS away) can fire.  Even on AbortError
            // / network error, the controller is settled at this
            // point and a new tick is welcome.
            state.pollInFlight = false;
        }
    }

    /* Which clock the run-state badge shows, and from which series.
     *
     * TWO CLOCKS, NEITHER SUBSTITUTING FOR THE OTHER (parse.md § 2a):
     * `wall_clock_s` is an absolute epoch and becomes the "last result at"
     * TIMESTAMP; `elapsed_s` counts from the run's start and becomes the
     * DURATION.  Fed to the other's formatter, a six-minute SIESTA run
     * would display as a date.
     *
     * The epoch falls back to the file's `mtime` when the run carries no
     * clock of its own -- a raw SIESTA `.out` without molwatch hooks, whose
     * parser reports null rather than handing over its elapsed seconds.
     * There is
     * NO fallback in the other direction: an elapsed duration cannot be
     * turned into a date, because the file does not contain the missing
     * addend (P-T3).
     *
     * AN ENDED RUN'S "WHEN" IS ITS OWN END, where its output states one:
     * `runtime_info.run_end_local`, SIESTA's `>> End of run` -- the node's
     * clock, with no zone (P-T2).  The mtime is when the FILE last changed,
     * which a copy moves.  One fact, one
     * source; the mtime stays only where nothing in the file says when.
     */
    function badgeClocks(state) {
        const lastFinite = (arr) => {
            for (let i = arr.length - 1; i >= 0; i--) {
                if (Number.isFinite(arr[i])) return arr[i];
            }
            return null;
        };
        const data = (state && state.data) || {};
        // The server already offsets `elapsed_s` to the run's start, so the
        // last value IS the total -- no subtraction here.
        const elapsed  = lastFinite(data.elapsed_s || []);
        const lastWall = lastFinite(data.wall_clock_s || []);
        const endedAt  = (data.runtime_info || {}).run_end_local;
        return {
            elapsed: elapsed,
            lastResultEpoch: Number.isFinite(lastWall)
                ? lastWall
                : (Number.isFinite(state && state.mtime) ? state.mtime : null),
            endedLocal: (typeof endedAt === "string" && endedAt) ? endedAt : null,
        };
    }

    // Compact "1h 23m" / "12m 5s" / "45s" formatter for elapsed seconds.
    // Hours-and-minutes for long runs; minutes-and-seconds for medium;
    // bare seconds for short.  Negative inputs (clock skew between
    // server and the file's clock) are clamped to 0.
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
     * (matches the wire format of mtime / wall_clock_s).  Never
     * pass an elapsed-seconds value here -- see badgeClocks. */
    function fmtTimestamp(epochSecs) {
        if (!Number.isFinite(epochSecs)) return "";
        return _fmtDate(new Date(epochSecs * 1000));
    }

    /* The node's own clock as the output wrote it -- `run_end_local`,
     * "2026-09-24T09:22:18", no zone -- in fmtTimestamp's style.  The digits
     * are shown as written: a time with no zone is not converted, because
     * the file does not say from which (P-T2).  Anything else is shown as
     * it came. */
    function fmtNodeClock(naive) {
        const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})/
            .exec(String(naive || ""));
        if (!m) return String(naive || "");
        return _fmtDate(new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]));
    }

    function _fmtDate(d) {
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
        // SIESTA's build and solver, as its output states them -- the
        // build header and the `diag:` lines, read through the SIESTA
        // family's table (`siesta_grammar`).  Absent on PySCF runs and on
        // an output cut before them.
        const sb = rt.siesta_build;
        if (sb && sb.version) {
            let s = "SIESTA " + sb.version;
            if (Array.isArray(sb.parallelisations) && sb.parallelisations.length) {
                s += " · " + sb.parallelisations.join("+");
            }
            parts.push(s);
        }
        const sd = rt.siesta_diag;
        if (sd && sd.algorithm) parts.push(sd.algorithm);
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

    // Compare frame ``idx`` between two frame arrays (each atom = [element, x, y, z]).
    // Used to GUARD the incremental tail-append: before we append the new frames,
    // the last frame we ALREADY hold (the shared boundary) must be byte-identical in
    // the fresh parse -- proof the server didn't rewrite earlier steps and that our
    // append point is aligned (no wrong / off-by-one / duplicate frame).  Any
    // mismatch => we do NOT append; we fall back to a full atomic rebuild.
    function _frameEqualAt(oldFrames, newFrames, idx) {
        if (idx < 0) return true;   // nothing shared yet -> nothing to check
        if (!Array.isArray(oldFrames) || !Array.isArray(newFrames)) return false;
        const a = oldFrames[idx], b = newFrames[idx];
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            const pa = a[i], pb = b[i];
            if (!pa || !pb) return false;
            if (pa[0] !== pb[0]) return false;               // element identity
            for (let k = 1; k <= 3; k++) {                   // coords (same parse -> exact)
                if (Math.abs(pa[k] - pb[k]) > 1e-9) return false;
            }
        }
        return true;
    }

    // Fingerprint of the EXCLUDED (frozen) atom set -- if it changes between polls
    // (e.g. a reload of a log whose held atoms differ), the excluded arrows change
    // for EVERY frame, so the incremental arrow-append must fall back to a full
    // rebuild.  Stable string so ordering doesn't produce false diffs.
    function _frozenFingerprint(data) {
        const rt = data && data.runtime_info;
        const arr = rt && rt.frozen_atoms;
        return Array.isArray(arr)
            ? arr.slice().sort(function (a, b) { return a - b; }).join(",") : "";
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

    /* THE RUN-STATE BADGE -- the user's primary "is this finished?" signal --
     * drawn from THE RUN the file belongs to: the state the server sends with
     * the file, from the one door (`run`; web/results.md § 4.1,
     * web/trajectory.md § 4), the one the Run panel and `jobset status` read.
     * A file that belongs to no run -- an upload -- is read by its own ending.
     * ONE renderer, called by every answer that can change it: a load, a
     * poll with new content, and a quiet poll that brings only how the run is
     * doing. */
    function _renderBadge() {
        if (!state.data) return;
        // Two clocks, and they are not interchangeable
        // (docs/model/parse.md § 2a).  `elapsed_s[]` counts from the
        // run's start and is the only series that may be shown as a
        // duration; `wall_clock_s[]` is an absolute epoch and the only
        // one that may be shown as a date.  Either may be an all-null
        // series when the engine cannot report it -- no step of a SIESTA
        // .out carries a time of day -- so each is read on its own and
        // neither substitutes for the other.
        const { elapsed, lastResultEpoch, endedLocal } = badgeClocks(state);

        // WHICH OF THE FIVE (`_badgeKind`): the run's, or a lone file's.
        const run       = state.fileState.run;
        const kind      = _badgeKind(run, state.data);
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
            // "Last result at <time>": prefer the per-frame
            // `wall_clock_s` from the simulation log (authoritative;
            // this is when the simulation itself produced the result),
            // fall back to the file's mtime -- the only timestamp
            // available when the engine's steps carry no time of day,
            // e.g. a raw SIESTA .out without molwatch hooks.  This
            // is DIFFERENT from "Watch tab last polled at X" -- a
            // client-side concern not shown on the badge.
            const lastResultTs = (lastResultEpoch != null)
                ? fmtTimestamp(lastResultEpoch)
                : "";
            // A run that has stopped moving is dated by its own end where
            // its output states one (badgeClocks); the fallback is the
            // "last result" time.
            const endedTs = endedLocal ? fmtNodeClock(endedLocal) : lastResultTs;
            const elapsedTxt = (elapsed != null)
                ? fmtElapsed(elapsed)
                : "";
            const joinParts = (...parts) =>
                parts.filter(s => s && s.length).join(" \u00b7 ");
            if (kind === "finished") {
                badge.classList.add("run-state-finished");
                badgeLab.textContent = "Finished";
                badgeDet.textContent = joinParts(
                    endedTs ? "ended " + endedTs : "",
                    elapsedTxt ? "total " + elapsedTxt : "",
                );
                badgeDet.removeAttribute("title");
            } else if (kind === "stopped") {
                // "Stopped", not "Error" (user, 2026-05-30).
                // Non-convergence is a STATE, not an error
                // of the viewer / the .out file.  The actual reason
                // (SCF non-convergence, MPI fault, ...) is shown below
                // as a classified tag; the raw parser message is the
                // tooltip so power users can read it without losing
                // the visual hierarchy.
                badge.classList.add("run-state-error");
                badgeLab.textContent = "Stopped";
                // The output's own stop where it states one; else the run's
                // own words -- a kill its monitor saw states nothing in the
                // output (web/trajectory.md § 4).
                const reasonTag = _fileStopped()
                    ? _stopReason(state.data, errMsg)
                    : ((run && run.detail) || _stopReason(state.data, errMsg));
                badgeDet.textContent = joinParts(
                    reasonTag ? "Reason: " + reasonTag : "",
                    endedTs ? "stopped " + endedTs : "",
                    elapsedTxt ? "total " + elapsedTxt : "",
                );
                // Full raw message available on hover (and for screen
                // readers via aria, since title is announced by most).
                if (errMsg) badgeDet.setAttribute("title", errMsg);
                else        badgeDet.removeAttribute("title");
            } else if (kind === "queued" || kind === "pending") {
                // Launched and silent, or prepped and never launched: the
                // run's own words beneath -- "queued as job 481923".
                badge.classList.add("run-state-ongoing");
                badgeLab.textContent = kind === "queued" ? "Queued"
                                                         : "Not launched";
                badgeDet.textContent = (run && run.detail) || "";
                badgeDet.removeAttribute("title");
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

    }

    /* Which badge: finished | stopped | queued | pending | running -- the
     * run's state when the file belongs to one, else the file's own ending
     * (`run_state`, model/parse.md § 2b). */
    function _badgeKind(run, data) {
        if (run) {
            return ({ finished: "finished", failed: "stopped",
                      queued: "queued", pending: "pending" })[run.state]
                || "running";
        }
        const rs = String((data && data.run_state)
                          || RUN_STATE.RUNNING).toLowerCase();
        if (rs === RUN_STATE.ENDED) return "finished";
        if (rs === RUN_STATE.STOPPED || rs === RUN_STATE.OOM) return "stopped";
        return "running";
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
        // Guards on the incremental tail-append (never add a wrong / extra frame):
        //   - boundary check: the last frame we ALREADY hold is byte-identical in the
        //     fresh parse (the server didn't rewrite history; our append point aligns).
        //   - count-in-sync: the movie's live frame count equals our record (oldLen),
        //     so appending (newLen-oldLen) frames lands us at exactly newLen.
        // Either failing => NOT a provable continuation => full atomic rebuild instead.
        const boundaryOk = _frameEqualAt(oldData && oldData.frames,
                                         r.data && r.data.frames, oldLen - 1);
        const countInSync = !_mv || (_mvdata().frameCount() === oldLen);
        const canAppend = _mv
            && sameAtomCount
            && newLen > oldLen
            && _latticeEqual(oldData && oldData.lattice,
                             r.data    && r.data.lattice)
            && boundaryOk
            && countInSync;

        // Live-refresh no-new-frames short-circuit.
        //
        // The /api/watch/data poll fires every N seconds and frequently
        // returns the same trajectory it returned last time (no new
        // frames yet).  A full rebuild on every such poll would reset
        // the camera, rearm the animation from frame 0 and stop the
        // playback that was running.
        //
        // So if the new data carries no new frames AND has the
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
            // (these can flip from "running" → "ended" / "stopped"
            // on a follow-up poll even with no new frames),
            // then bail before touching the model / animation /
            // plots.  Plots rebuilt only if the run-state changed
            // OR the SCF history for the in-flight step grew.
            //
            // During a CG step, SCF iterations get appended to
            // ``scf_history[lastStep]`` without the frame count, atom
            // count, lattice, or run_state changing, so "same geometry"
            // is not "nothing to plot".  ``_scfFingerprint`` adds that
            // axis: per-step iter count + last cycle's energy.
            const runStateChanged =
                oldData.run_state !== r.data.run_state
             || oldData.error_message !== r.data.error_message;
            const scfChanged =
                _scfFingerprint(oldData) !== _scfFingerprint(r.data);
            // Contract § 2: route fileState writes through
            // transition().  noNewContent only updates the
            // fields that can change on a same-content tick.
            transition("APPLY", { path: r.path, mtime: r.mtime,
                                  data: r.data, run: r.run });
            _renderRuntimeInfo(state.data.runtime_info);
            _renderParseWarnings(state.data.parse_warnings);
            if (runStateChanged || scfChanged) makePlots();
            _renderBadge();
            return;
        }

        // The frame the user is on RIGHT NOW, read before the reload so a full rebuild can
        // restore it / follow the tail.  MolView owns the playhead -- the tab asks it here and
        // keeps no copy of the answer.
        const _mvd = _mvdata();
        const prevFrame = (_mvd && typeof _mvd.currentFrame === "function")
            ? _mvd.currentFrame() : 0;
        const wasAtEnd  = !oldData || prevFrame >= oldLen - 1;

        // Contract § 2: full-rebuild path -- route the atomic
        // fileState replacement through transition('APPLY') so
        // transition() is the SINGLE entry-point for fileState
        // writes.
        //
        // ``r.path`` is the server-resolved absolute path (the input
        // may have been a directory; r.path is the file actually
        // loaded — same value the live-poll URL uses).
        // The server always answers `format`, and it is the only thing
        // entitled to: the engine is a fact about the run DIRECTORY,
        // which the browser cannot see.  Falling back to
        // `data.source_format` here would be a second, client-side answer
        // to a question the server has already answered -- and a wrong one
        // in kind, since source_format is a FORMAT ("siesta-mdnc",
        // "pyscf-geom", "molwatch").  "unknown" is a real answer and
        // renders as the neutral banner.
        const resolvedFormat = r.format || "unknown";
        transition("APPLY", {
            mtime:  r.mtime,
            data:   r.data,
            format: resolvedFormat,
            label:  r.label || resolvedFormat,
            path:   r.path,
            // undefined on watch-data polls → keep existing (metadata is
            // per-file and doesn't change mid-run); set on a fresh load.
            atomMetadata: r.atomMetadata,
            periodicity:  r.periodicity,
            info:         r.info,
            // Same rule, same reason: frame 0's envelope rides with the load
            // and a poll that re-sends frames must not drop it.
            structure:    r.structure,
            // How the run is doing -- on the load always, on a poll only when
            // the run ended with this write (web/results.md § 4.1).
            run:          r.run,
        });
        _renderRuntimeInfo(state.data && state.data.runtime_info);
        _renderParseWarnings(state.data && state.data.parse_warnings);

        const n = state.data.frames.length;
        if (n === 0) {
            setStatus("File loaded ("+ state.label +") but no frames yet.", "");
            return;
        }
        // (Per-frame XYZ export is MolView's Export knob, which is
        // current-frame-correct.  Frame count + slider are MolView's frame bar.)

        if (canAppend) {
            // Strict tail-append: hand MolView ONLY the new frames (addFrames
            // appends without disturbing the current view or MolView's playback
            // timer), so a running animation keeps playing and the camera / the
            // user's frame position don't snap.  MolView's frame bar counter
            // updates itself off the store notification.
            const newCoords = [];
            for (let i = oldLen; i < newLen; i++) {
                newCoords.push(r.data.frames[i].map(
                    (atom) => [atom[1], atom[2], atom[3]]));
            }
            let appendedOk = false;
            try {
                _mvdata().addFrames(newCoords);
                // Post-check: the movie must now hold EXACTLY the server's count.
                // A mismatch (an addFrame threw / dropped / doubled a frame) means
                // the tail is out of sync -> resync via a full atomic rebuild rather
                // than leave a wrong/extra frame on screen.
                appendedOk = _mvdata().frameCount() === newLen;
            } catch (_) { appendedOk = false; }
            if (!appendedOk) {
                rebuildModel(wasAtEnd ? n - 1 : Math.min(prevFrame, n - 1));
            } else {
                // Re-hand the filtered per-frame forces (now including the appended tail).
                // setForces re-bakes the arrow overlay IN PLACE (no movie reload), so ONE call
                // covers both a plain append AND a frozen-set change.
                // (Perf note: this re-bakes every frame's arrows per poll; an incremental
                // force-append is a future optimisation if long live trajectories need it.)
                drawForces();
                // Follow the tail if the user was watching the end; otherwise leave the
                // playhead where it is.  The target comes from MolView's own count, not from
                // `n` (the parsed feed's length): the feed's job is to FEED MolView, never to
                // be the authority on what the viewer is showing.
                if (wasAtEnd) {
                    const shown = _mvdata().frameCount();
                    if (shown > 1) _mvdata().setCurrentFrame(shown - 1);
                }
            }
        } else {
            // Structure changed / frames shrank: full rebuild.  Keep the
            // playhead near where it was (follow the tail if the user was at
            // the end), applied inside rebuildModel after the reload.
            rebuildModel(wasAtEnd ? n - 1 : Math.min(prevFrame, n - 1));
        }
        makePlots();

        // Signal to the /results tab-level picker (and any other
        // listener that wants to drop a "Loading…" / "Parsing…"
        // status overlay) that the first render of this file is now
        // visible on screen.  Deferred to two consecutive
        // ``requestAnimationFrame`` ticks so the dispatch fires
        // AFTER the browser has had a chance to paint the new
        // ``frame-tot`` / slider state, run 3Dmol's GPU render, and
        // commit Plotly's plot drawing -- a synchronous dispatch arrives
        // in the same event-loop tick as the textContent assignment,
        // before the browser has painted.  Double rAF (rAF inside rAF)
        // is the cheapest pattern that ensures we're past one full
        // paint cycle.  Subsequent polls also dispatch but that's a
        // no-op for the picker (it idempotently clears the parse
        // status; ``parsingFor`` is null on poll-triggered fires).
        _announceReady({ frames: n });

        _renderBadge();

        /* WHERE THE BOX CAME FROM, when there is one -- and nothing else on
         * an ordinary load.  A cell drawn round a structure is a claim about
         * its physics, and this one was not made by the person reading it --
         * the run reported it and molbuilder applied it, which for a molecule
         * in a large SIESTA box means a box appears round something isolated.
         * Saying so is the difference between a shown fact and a silent one;
         * the Cell page deliberately answers "is this box mine?" and not
         * "where did it come from" (molview.md \u00a7 9.5), so the tab that did
         * the load is what says it.  The badge carries the run's state and
         * the frame bar its frames; this line carries what neither does. */
        setStatus(_cellCameFromTheRun()
                  ? "Unit cell from the run (its output, or the box its deck"
                    + " placed the atoms in), not set by you."
                  : "", null);
    }

    // Refresh-button listener wiring.  Wired ONCE at mount; not
    // re-wired on every load/transition.
    function _wireRefreshListener() {
        const C = (window.molbuilder || {}).constants;
        if (!C || !C.EVENT_REFRESH_REQUESTED) return;
        // Contract § 5: Refresh = file-switch with current path.
        // loadByPath -> transition('LOADING') runs the full reset
        // matrix.
        const _onRefresh = () => {
            const p = state.fileState.path;
            if (!p) return;     // not yet loaded; nothing to refresh
            loadByPath(p);
        };
        _on(document, C.EVENT_REFRESH_REQUESTED, _onRefresh);

        /* WATCH THE CONTAINER, don't wait to be told -- as the 3-D viewer
         * beside these plots does.  A width change with no re-render (the
         * sidebar fold) reaches no render pass.
         *
         * Observing the row covers every cause -- fold, unfold, window resize,
         * the SCF grid reflow, and any future layout change -- without the
         * sidebar and the plots having to know about each other. */
        const plotsRow = rootEl.querySelector(".plots-row");
        if (plotsRow && typeof ResizeObserver === "function") {
            /* NO requestAnimationFrame COALESCING HERE: rAF does not run in
             * a tab that is not rendering, so a `pending` guard set by an
             * observer fire while the tab is hidden would swallow EVERY later
             * resize for the life of the mount.
             *
             * `Plotly.Plots.resize` already returns early when the size has
             * not changed.  Straight call, no state to get wrong. */
            const ro = new ResizeObserver(() => { resizePlots(); });
            ro.observe(plotsRow);
            _listeners.defer(() => {
                try { ro.disconnect(); } catch (_) {}
            });
        }
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

    // The tab owns NO playback: no step / play / pause / timer, and no #speed or #loop
    // controls.  MolView owns all of it -- the playback timer lives in its mount.js and it
    // renders its own frame-controls bar (prev/play/next + loop + speed + slider + counter),
    // so a tab that hands MolView a trajectory gets the navigation UI for free.  There is
    // exactly one bar and one timer.
    //
    // The only seek this file performs is the follow-the-tail jump after an append, and it
    // goes through `data.setCurrentFrame` like every other frame write -- never through the
    // embed handle.

    /* ------------------------------------------------------------------ */
    /*  UI wiring                                                          */
    /* ------------------------------------------------------------------ */

    // /results drives loading via the registry's mount(host, file, ctx).

    async function loadByPath(path) {
        // A fresh load replaces the whole model (rebuildModel \u2192
        // installMolecule); MolView owns the selection + playhead reset from there.
        setStatus("Loading\u2026", "");
        // Contract \u00a7 2: file-switch -> transition('LOADING').  This
        // single call runs the reset matrix (matrix \u00a7 3 row 1):
        //   * abort in-flight loadAbort + pollAbort
        //   * stop poll timer + clear pollInFlight
        //   * empty fileState (sets path = new path)
        //   * reset viewState (firstFit=true)
        // Refresh button arrives here too -- same code path, same
        // resets.  Eliminates the half-refresh class.
        transition("LOADING", { path: path });
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
            // Contract § 4 Invariant 1, asked of the answer's own
            // identity.  A newer LOADING moved
            // `fileState.path`; dispose set it to null.  The
            // `signal.aborted` check above is the other half: it is the
            // only thing that can tell two loads of the SAME file apart.
            if (path !== state.fileState.path) return;
            if (!r.ok) {
                setStatus(r.error || "Load failed.", "error");
                _announceReady({ error: r.error || "Load failed." });
                return;
            }
            applyNewData({
                mtime:  r.mtime,
                data:   r.data,
                format: r.format,
                label:  r.label,
                path:   r.path,
                // Per-atom metadata recovered from the run's input script
                // (region labels / frozen / annotations); handed to MolView
                // below.  Only the LOAD response carries it -- watch-data
                // polls omit it and keep the value (APPLY keep-existing).
                atomMetadata: r.atom_metadata || null,
                periodicity:  r.periodicity || null,
                // What the run says about itself (the deck's recorded
                // contract), composed by the same server-side answer as
                // the two above -- see watch.py::_run_metadata.
                info:         r.info || null,
                // FRAME 0 AS AN ENVELOPE -- what the viewer installs
                // (watch.py::_frame0_structure).  Forwarded here beside the
                // three above because it arrives the same way and for the same
                // reason: only the LOAD response carries it.
                structure:    r.structure || null,
                // How the run this file belongs to is doing -- the one door's
                // answer, which `_settlePostLoad` follows and the badge reads;
                // null for an upload, which belongs to no run
                // (web/results.md § 4.1).
                run:          r.run !== undefined ? r.run : null,
            });
            // Directory mode: show the user which file the loader
            // picked, and update the input with the resolved path so
            // the next poll / page-revisit reuses it directly.
            if (r.resolved_from) {
                const baseDir = r.resolved_from.replace(/\/+$/, "");
                const fileNm  = (r.path || "").split("/").pop() || r.path;
                // A directory resolves to ONE file (watch.py's discovery
                // chain): stages are separate runs, and the person picks one.
                // No mtime: the badge carries the run's time.
                const msg = "Loaded \u201c" + fileNm + "\u201d from " + baseDir
                    + "/.";
                setStatus(msg, "ok");
            }
            // Contract § 2: transition by the run's state.
            // _settlePostLoad starts the poll timer iff the run is live
            // (WATCHING) -- a run that is over doesn't get polled.
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
            // The load has ended, without a render: say so, as the refused
            // load does, so the cover does not sit over this error.
            _announceReady({ error: e.message });
        }
    }

    // Force-vector PRODUCER parameters — the trajectory-specific controls.
    // The inspector hands the ENGINE filtered raw forces + drives the forceScale
    // flag; the engine builds + styles the arrows (gold max-highlight + magnitude
    // ramp).  Whether they're DRAWN is MolView's "show overlay" view-toggle.
    // SCALE is a cheap flag (in-place length re-bake); the FILTER knobs (min / hide-frozen)
    // re-hand the forces (drawForces → data.setForces, in-place re-bake), and MolView
    // redraws the current frame with the overlay's on/off visibility unchanged.
    // Unit cell, atom-index labels, playback speed / loop, the atom list and
    // per-frame export are MolView's: playback + speed + loop are its frame bar,
    // cell + labels are its knob bar, selection + measurement are its panel, and
    // per-frame structure export is its Export knob (current-frame-correct).
    _on($("force-scale"), "input", (e) => {
        const v = parseFloat(e.target.value) || 1.0;
        $("force-scale-val").textContent = v.toFixed(1);
        // Scale is one of the switches beside the selection (§ 9.5): the engine
        // re-bakes arrow LENGTH in place (no forces rebuild), so dragging the
        // slider is smooth.  Guarded on the VIEWER existing, not on the method:
        // there is no viewer before the mount resolves.
        const d = _mvdata();
        if (d) d.selection.setSwitch("forceScale", v);
    });
    // The FILTER knobs change WHICH forces show (min threshold / exclude frozen), so they
    // re-hand the filtered per-frame forces (drawForces -> setForces, in-place re-bake).
    _on($("force-min"),     "input",  drawForces);
    _on($("hide-frozen"),   "change", drawForces);

    /* ---- Export all plot data as CSV ---------------------------- */
    /* Bundles every column displayed across the 4 trajectory plots
       (energy + max forces + SCF cycle / energy / gnorm) into one
       CSV.  Header carries source attribution + parser + timestamp
       so the file is self-describing — the user can re-trace what
       it came from months later without external bookkeeping.
       This handler gathers state, builds the file through
       _buildPlotCsv above, and triggers the browser download. */
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
    // catches up.
    //
    // Like the file picker: hook pageshow (covers
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

    // ---- Auto-load + handle assembly --------------------------- //
    //
    // If the caller asked for an initial file, load it now.
    if (opts.file) {
        loadByPath(opts.file);
    }

    // Refresh-button listener wires here ONCE per mount.  Tears
    // down with the inspector's dispose path, through the listener scope.
    _wireRefreshListener();

    // The handle the caller uses to dispose + control the mounted
    // inspector.  Required for /results' registry-based dispatch
    // (the registry calls dispose() before mounting the next
    // inspector).
    return {
        /**
         * Tear down every long-lived resource this mount created:
         * polling timer, in-flight HTTP requests, and the embed
         * handle (the embed's dispose() releases the WebGL
         * context's bookkeeping + its animation loop + its
         * ResizeObserver; the canvas itself is freed when the
         * host's innerHTML is cleared by the caller).
         */
        dispose() {
            // Hand the whole scope back (lib/inspectors/lifecycle.js).
            // It tears down in reverse, so the most recent registration
            // goes first -- the order they would be re-attached in on a
            // remount -- and covers every element-level listener from the
            // mount plus the deferred ResizeObserver.  Its per-teardown
            // catch keeps one buggy teardown from blocking the rest.
            _listeners.disposeAll();
            // Contract § 2: dispose -> transition('IDLE').  The
            // matrix § 3 row "dispose / unmount" runs the full
            // reset (aborts, timer stop, bucket clears).
            transition("IDLE");
            // _mv.dispose() (the molview mount handle) tears down the WHOLE
            // fused assembly — the embedded 3Dmol viewer + its animation loop /
            // ResizeObserver / knob bar, the selection panel, the view + frame
            // controls, the overlay controller, and every store subscription.
            if (_mv && typeof _mv.dispose === "function") {
                try { _mv.dispose(); } catch (_) {}
            }
        },
        /**
         * Swap the displayed trajectory without re-mounting the
         * inspector.
         */
        load(path) { return loadByPath(path); },
    };

    }   // ----- end of mountInspector(rootEl, opts) -----

    // Export for the consumer (lib/inspectors/trajectory.js on
    // /results).  The consumer is
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
