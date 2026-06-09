/* Shared scaffolding for the "partial-fetch + core.mount" inspector
 * adapters.  Pre-task-#308 lib/inspectors/trajectory.js and
 * lib/inspectors/spectra.js were ~95% identical: same error-card
 * renderer, same AbortController + fetch + innerHandle dance, same
 * cleanup-walk dispose shape.  The factory below captures all of
 * that; the two adapters collapse to ~25-line config blocks each.
 *
 * Future inspectors that follow the same pattern (a server-rendered
 * Jinja partial + a separately-loaded ``core.mount(host, {file})``
 * module) can call ``makePartialInspector(...)`` and get the same
 * scaffold for free.
 *
 * Public API on ``window.molbuilder.inspectors._partialInspectorFactory``:
 *
 *   ``makePartialInspector(opts)`` -> inspector object that the
 *      registry's ``register(inspector)`` accepts.  Opts:
 *
 *        name          string  registry key (e.g. "trajectory")
 *        displayName   string  user-facing label
 *        coreApiKey    string  ``window.molbuilder[coreApiKey]``
 *                              MUST expose ``mount(host, {file})``
 *        partialUrl    string  GET endpoint that returns the partial
 *                              HTML
 *        match         function(file) -> bool
 *        resultCategory function(file) -> string  (optional; used by
 *                              the /results picker's optgroup labels)
 *        isResult      bool   (default true; matches both current
 *                              call-sites)
 *
 * The underscore prefix on ``_partialInspectorFactory`` marks this
 * as an internal helper to the inspectors module; production
 * consumers (the .registers below) live in the per-engine wrappers.
 */
(function (root) {
    "use strict";

    const _basename = (root.molbuilder
                       && root.molbuilder.path
                       && root.molbuilder.path.basename)
                    || ((p) => p || "");

    /**
     * Render an error card into ``host``.  Used when the fetch
     * fails OR the per-engine core module didn't self-register
     * (broken deployment, CDN issue, script-order regression).
     * XSS-safe: createElement + textContent only.
     */
    function _renderError(host, displayName, file, reason) {
        host.innerHTML = "";
        const card = document.createElement("section");
        card.className = "inspector-card error-card";
        const title = document.createElement("h2");
        title.className = "inspector-card-title";
        title.textContent = displayName + " inspector failed to mount";
        card.appendChild(title);
        const fileLine = document.createElement("p");
        fileLine.className = "inspector-card-note";
        fileLine.appendChild(document.createTextNode("File: "));
        const strong = document.createElement("strong");
        strong.textContent = _basename(file);
        fileLine.appendChild(strong);
        card.appendChild(fileLine);
        const reasonLine = document.createElement("p");
        reasonLine.className = "inspector-card-note";
        reasonLine.appendChild(document.createTextNode("Reason: "));
        const code = document.createElement("code");
        code.textContent = reason;
        reasonLine.appendChild(code);
        card.appendChild(reasonLine);
        host.appendChild(card);
    }

    function makePartialInspector(opts) {
        const name          = opts.name;
        const displayName   = opts.displayName;
        const coreApiKey    = opts.coreApiKey;
        const partialUrl    = opts.partialUrl;
        const errorDisplay  = opts.errorDisplayName || displayName;
        const coreScriptHint =
            opts.coreScriptHint
            || ("lib/" + (opts.coreScriptDir || name) + "/core.js");

        const inspector = {
            name:        name,
            displayName: displayName,
            isResult:    (opts.isResult !== false),
            match:       opts.match,
        };
        if (typeof opts.resultCategory === "function") {
            inspector.resultCategory = opts.resultCategory;
        }

        inspector.mount = function (host, file, _ctx) {
            const cleanups = [];

            // Per-engine core must self-register on
            // ``root.molbuilder[coreApiKey]`` before this script
            // runs (see results.html script order).  Missing →
            // visible error card, not a half-mounted UI.
            const api = (root.molbuilder || {})[coreApiKey];
            if (!api || typeof api.mount !== "function") {
                _renderError(host, errorDisplay, file,
                    coreScriptHint
                    + " did not load -- check results.html "
                    + "script order");
                return { dispose() { host.innerHTML = ""; } };
            }

            // Async fetch + mount.  Return the handle synchronously
            // so the registry can dispose() if the user picks
            // another file before the fetch resolves.
            let innerHandle = null;
            let aborted     = false;
            const abortCtl  = new AbortController();
            cleanups.push(() => abortCtl.abort());

            fetch(partialUrl, {
                credentials: "same-origin",
                signal:      abortCtl.signal,
            })
                .then((resp) => {
                    if (!resp.ok) {
                        throw new Error(
                            "HTTP " + resp.status + " from " + partialUrl);
                    }
                    return resp.text();
                })
                .then((partialHtml) => {
                    if (aborted) return;
                    // Trust boundary: partialHtml is a same-origin
                    // Jinja-autoescaped render of a static template
                    // with no user input; same content the per-tab
                    // pages (e.g. /spectra) include via Jinja.
                    host.innerHTML = partialHtml;
                    innerHandle = api.mount(host, { file: file });
                    cleanups.push(() => {
                        try { innerHandle.dispose(); }
                        catch (_) { /* already torn down */ }
                    });
                })
                .catch((err) => {
                    if (err.name === "AbortError") return;
                    _renderError(host, errorDisplay, file,
                        (err && err.message) || String(err));
                });

            return {
                dispose() {
                    aborted = true;
                    // Reverse-walk so the inner handle (last in)
                    // tears down before the AbortController (first
                    // in).
                    for (let i = cleanups.length - 1; i >= 0; i -= 1) {
                        try { cleanups[i](); }
                        catch (_) { /* swallow per-cleanup */ }
                    }
                    host.innerHTML = "";
                },
            };
        };

        return inspector;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors._partialInspectorFactory = {
        makePartialInspector: makePartialInspector,
    };
})(typeof window !== "undefined" ? window : this);
