/* molbuilder Inspector Registry -- the architectural seam between
 * a sidebar selection and the right file-type-specific UI.
 *
 * Why this exists
 * ===============
 * Before the registry, ``/results``'s dispatch had two hard-coded
 * lists: one in the template (five inspector panels with magic
 * ids) and one in the JS (five match rules in a literal table).
 * Adding a new file type required editing both, in lockstep.
 *
 * With the registry, each inspector is a self-contained module that
 * declares its own match rule, displayName, and how it mounts itself
 * inside a host element.  The template carries ONE host; the JS
 * dispatch is a short "pick inspector for this file; mount it; on
 * next file dispose + mount the next one".  Adding a new file type
 * is one new module + one register call.
 *
 * Inspector contract (the only interface anyone needs to know):
 *
 *   const inspector = {
 *     name:        "trajectory",          // unique registry key
 *     displayName: "Trajectory viewer",   // user-visible label
 *     match:       (filepath) => boolean, // dispatch predicate
 *     mount:       (host, file, ctx) => handle,
 *     isResult:    true,                  // optional, default false.
 *                                         //   When true, this inspector's
 *                                         //   matched files appear in the
 *                                         //   /results tab-level file
 *                                         //   picker dropdown (see
 *                                         //   ``lib/results/file-picker.js``).
 *                                         //   Set to false for catch-all
 *                                         //   viewers (e.g. ``source``)
 *                                         //   that match generic file
 *                                         //   types -- they'd flood the
 *                                         //   dropdown with non-result
 *                                         //   files (configs, READMEs,
 *                                         //   ...).
 *     resultCategory: (file) => string,   // optional, used iff isResult.
 *                                         //   User-facing engine-flavoured
 *                                         //   label the picker groups
 *                                         //   files under (rendered as a
 *                                         //   ``<optgroup>`` header).  An
 *                                         //   inspector that matches more
 *                                         //   than one workflow's outputs
 *                                         //   (e.g. ``trajectory`` claims
 *                                         //   both ``.out`` for SIESTA AND
 *                                         //   ``.molwatch.log`` for PySCF)
 *                                         //   discriminates per-file here.
 *                                         //   Defaults to ``displayName``
 *                                         //   when omitted.
 *   };
 *
 * Handle contract (returned by mount; required cleanup point):
 *
 *   handle = { dispose: () => void };  // required: drop timers,
 *                                      //   subscribers, observers,
 *                                      //   clear host innerHTML
 *
 * Mount context (third arg to ``mount(host, file, ctx)``):
 *
 *   ctx = {
 *     showError(message): void,                        // render error inside host
 *     readFile(file, opts?):  Promise<{ok, text, error?}>,          // -> projects.readFile
 *     readRange(file, offset, maxBytes, opts?): Promise<{ok, text, eof, …}>, // -> projects.readRange
 *   };
 *
 * The framework is high-level DISPATCH; it does NOT implement file reading.  The default
 * context INJECTS the sidebar's file layer (``projects.readFile`` / ``readRange`` -- the
 * ONE byte-access implementation), so handlers read through ``ctx.*`` and never hand-roll
 * an /api/files/* fetch.  We pass ctx EXPLICITLY into mount() so a non-default dispatcher
 * (e.g. a cached reader or a custom error renderer) can swap implementations without
 * patching the registry.
 *
 * Inspectors share the host element exclusively while mounted.  The
 * registry's mount() calls the inspector's mount() and returns its
 * handle directly; the DISPATCHER calls ``handle.dispose()`` BEFORE
 * mounting the next inspector.  Inspectors clean their own DOM in
 * dispose() (e.g., ``host.innerHTML = ""``).
 *
 * Ordering: ``pick()`` evaluates match rules in registration order
 * and returns the first match.  Inspectors with more specific
 * predicates (compound extensions like ``.molwatch.log``) MUST
 * register before more general ones (``.log``) to win the dispatch.
 *
 * Used via ``window.molbuilder.inspectors.{register, pick, mount,
 * list, createDefaultContext}``.
 */
(function (root) {
    "use strict";

    const _inspectors = [];

    /**
     * Register an inspector.  Idempotent: re-registering the same
     * ``name`` replaces the previous entry (lets a placeholder be
     * swapped for a real implementation when it lands).
     */
    function register(inspector) {
        _validateInspector(inspector);
        const ix = _inspectors.findIndex(i => i.name === inspector.name);
        if (ix >= 0) _inspectors[ix] = inspector;
        else         _inspectors.push(inspector);
    }

    /**
     * Pick the first registered inspector whose match() returns true
     * for ``file``.  Returns the inspector or ``null`` if no match.
     * Does not mount.
     *
     * A throwing ``match()`` is a programming error in the inspector.
     * We log + skip rather than abort the entire dispatch (one
     * broken inspector shouldn't break the whole /results page) but
     * the console.warn surfaces the bug so it's noticed in dev.
     */
    function pick(file) {
        if (!file) return null;
        for (const i of _inspectors) {
            try {
                if (i.match(file)) return i;
            } catch (e) {
                console.warn(
                    "[inspectors] match() threw for " + i.name + ":", e
                );
            }
        }
        return null;
    }

    /**
     * Like ``pick()``, but returns ONLY when the matched inspector has
     * ``isResult: true``.  Used by the /results tab-level file picker
     * to enumerate the directory's known result files without
     * including catch-all viewers (the ``source`` inspector matches
     * ``.log`` / ``.json`` / ``.txt`` / ``.md`` / ``.fdf`` / ``.py``,
     * which would flood the picker with configs + READMEs + scripts).
     *
     * Same ordering semantics as ``pick()`` (first-match-wins in
     * registration order); ``isResult`` is checked AFTER ``match`` so
     * a more specific result-inspector still claims the file before a
     * less-specific non-result inspector.
     */
    function pickResult(file) {
        const inspector = pick(file);
        if (!inspector) return null;
        return inspector.isResult ? inspector : null;
    }

    /**
     * Mount the matching inspector inside ``host`` for ``file``,
     * passing ``ctx``.  Returns the inspector's handle, or ``null``
     * if no inspector matched (caller renders a fallback).
     *
     * ``ctx`` is required.  Use ``createDefaultContext(host)`` for
     * the standard one or build your own.  Explicit is better than
     * "registry magically materialises a default" because it makes
     * customisation (caching, telemetry, custom error UI) a one-
     * line change in the dispatcher.
     */
    function mount(host, file, ctx) {
        if (!ctx) {
            throw new TypeError(
                "inspectors.mount(host, file, ctx) requires ctx; "
                + "use createDefaultContext(host) for the standard one"
            );
        }
        const inspector = pick(file);
        if (!inspector) return null;
        try {
            return inspector.mount(host, file, ctx);
        } catch (e) {
            // An inspector that throws on mount is a bug.  We
            // surface the error in the UI (intentional graceful
            // degradation -- one broken inspector shouldn't take
            // down /results) AND log to console for the developer.
            console.error(
                "[inspectors] " + inspector.name + ".mount threw:", e
            );
            ctx.showError(
                inspector.name + " inspector failed to mount: "
                + (e && e.message ? e.message : String(e))
            );
            return { dispose: () => { host.innerHTML = ""; } };
        }
    }

    /**
     * Build the standard mount context.  Dispatchers use this when
     * they don't need to override anything.  The returned object
     * captures ``host`` so callers don't have to pass it twice.
     */
    function createDefaultContext(host) {
        // The framework is high-level DISPATCH; it does NOT own a file reader.  It
        // INJECTS the sidebar's file layer (``projects.readFile`` / ``readRange`` -- the
        // ONE byte-access implementation) so handlers never hand-roll an /api/files/*
        // fetch.  Handlers read through ``ctx.readFile`` / ``ctx.readRange``.
        function _projects() {
            return (root.molbuilder && root.molbuilder.projects) || null;
        }
        return {
            showError(message) {
                host.innerHTML = "";
                const div = document.createElement("div");
                div.className = "inspector-error";
                div.textContent = message;
                host.appendChild(div);
            },
            // Read a file's bytes as text.  ``opts.maxBytes`` raises the read budget
            // (source inspector does this for multi-MB .out/.log); the server caps at 16
            // MB.  Delegates to ``projects.readFile`` -- same envelope ({ok, text, error}).
            readFile(file, opts) {
                const p = _projects();
                if (!p || typeof p.readFile !== "function") {
                    return Promise.resolve({ ok: false,
                        error: "projects file package unavailable" });
                }
                return p.readFile(file, opts);
            },
            // Read a byte WINDOW at ``offset`` (negative = from EOF) capped at
            // ``maxBytes`` -- the paginated source viewer.  Delegates to
            // ``projects.readRange`` (envelope {ok, text, eof, file_size, …}).
            readRange(file, offset, maxBytes, opts) {
                const p = _projects();
                if (!p || typeof p.readRange !== "function") {
                    return Promise.resolve({ ok: false,
                        error: "projects file package unavailable" });
                }
                return p.readRange(file, offset, maxBytes, opts);
            },
        };
    }

    /** Snapshot of registered inspectors (for tests + debugging). */
    function list() {
        return _inspectors.map(i => ({
            name:        i.name,
            displayName: i.displayName,
            isResult:    !!i.isResult,
        }));
    }

    /** Drop ALL registrations.  Tests use this for isolation; do
     *  NOT call from production code (the registry is process-wide
     *  state). */
    function _clear() {
        _inspectors.length = 0;
    }

    // ---- internals -------------------------------------------------- //

    function _validateInspector(inspector) {
        if (!inspector || typeof inspector !== "object") {
            throw new TypeError("inspector must be an object");
        }
        for (const key of ["name", "displayName"]) {
            if (typeof inspector[key] !== "string" || !inspector[key]) {
                throw new TypeError(
                    "inspector." + key + " must be a non-empty string"
                );
            }
        }
        for (const key of ["match", "mount"]) {
            if (typeof inspector[key] !== "function") {
                throw new TypeError(
                    "inspector." + key + " must be a function"
                );
            }
        }
    }

    // ---- export ---------------------------------------------------- //

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.register             = register;
    root.molbuilder.inspectors.pick                 = pick;
    root.molbuilder.inspectors.pickResult           = pickResult;
    root.molbuilder.inspectors.mount                = mount;
    root.molbuilder.inspectors.list                 = list;
    root.molbuilder.inspectors.createDefaultContext = createDefaultContext;
    // Test-only escape hatch; prefix-underscored to discourage
    // production callers (the registry is process-wide state).
    root.molbuilder.inspectors._clear               = _clear;
})(typeof window !== "undefined" ? window : this);
