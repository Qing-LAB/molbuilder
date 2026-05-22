/* Spectra-results inspector -- thin wrapper around lib/spectra/core.js.
 *
 * Match rule: ``*.spectra.json``.
 *
 * Mount flow:
 *   1. Fetch the inspector's DOM markup from
 *      ``GET /partials/spectra-inspector`` (rendered server-side from
 *      ``_spectra_inspector.html`` -- same partial spectra.html
 *      includes via Jinja, single source of truth).
 *   2. Assign the response text to ``host.innerHTML``.  Trusted:
 *      it's a same-origin Jinja-autoescaped render of a static
 *      template with no user input.
 *   3. Call ``window.molbuilder.spectraInspector.mount(host)`` to
 *      wire the 3Dmol mode viewer + Plotly chart + modes table
 *      against the newly-injected DOM.
 *   4. Return a handle whose ``dispose()`` chains the inner handle's
 *      dispose + clears the host.
 *
 * The actual inspector logic lives in lib/spectra/core.js (loaded
 * earlier in results.html).  Both /spectra and /results use that
 * same module; this wrapper is the registry-side adapter only.
 *
 * Auto-load:  the adapter forwards ``opts.file`` to the core's
 * mount() call below; the core honors it by pre-filling watch-path
 * and triggering loadByPath() (parity with the trajectory core,
 * landed 2026-05-19).  Side-effect: picking a .spectra.json in the
 * sidebar on /results auto-mounts the inspector AND auto-loads the
 * file -- no extra click required.
 *
 * Hot-swap:  the spectra core now also exposes ``{ load(path) }``
 * on its handle (same shape as the trajectory core).  A future
 * registry refactor that wants to swap files without a full
 * dispose+remount cycle can call ``handle.load(newPath)`` instead.
 * Today the registry just remounts on each selection change; the
 * load() handle is here for symmetry.
 */
(function (root) {
    "use strict";

    const PARTIAL_URL = "/partials/spectra-inspector";

    const _basename = (root.molbuilder
                       && root.molbuilder.path
                       && root.molbuilder.path.basename)
                    || ((p) => p || "");

    /**
     * Render an error card into ``host`` (used when fetching the
     * partial fails or the core module isn't loaded).  XSS-safe:
     * uses createElement + textContent only.
     */
    function _renderError(host, file, reason) {
        host.innerHTML = "";
        const card = document.createElement("section");
        card.className = "inspector-card error-card";
        const title = document.createElement("h2");
        title.className = "inspector-card-title";
        title.textContent = "Spectra inspector failed to mount";
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

    const inspector = {
        name:        "spectra",
        displayName: "Spectra results",
        match:       (file) => file.toLowerCase().endsWith(".spectra.json"),

        mount(host, file, _ctx) {
            // Cleanup chain.  Walked in reverse by ``dispose``; each
            // entry is a function the wrapper accumulates as it sets
            // up the inner mount.
            const cleanups = [];

            // Defensive: core.js must be loaded BEFORE this script
            // (see results.html's script ordering).  If it's missing
            // (broken deployment, CDN issue, etc.) fail loudly with a
            // visible error instead of leaving a half-mounted UI.
            const api = (root.molbuilder || {}).spectraInspector;
            if (!api || typeof api.mount !== "function") {
                _renderError(host, file,
                    "lib/spectra/core.js did not load -- check "
                    + "results.html script order");
                return { dispose() { host.innerHTML = ""; } };
            }

            // Fetch the partial + mount asynchronously.  We have to
            // return a handle *now* (the registry needs it to dispose
            // if the user picks another file before the fetch
            // completes).  The handle's dispose() captures whatever
            // state has been built so far; abort() any in-flight
            // fetch.
            let innerHandle = null;
            let aborted     = false;
            const abortCtl  = new AbortController();
            cleanups.push(() => abortCtl.abort());

            fetch(PARTIAL_URL, {
                credentials: "same-origin",
                signal:      abortCtl.signal,
            })
                .then((resp) => {
                    if (!resp.ok) {
                        throw new Error(
                            "HTTP " + resp.status + " from "
                            + PARTIAL_URL);
                    }
                    return resp.text();
                })
                .then((partialHtml) => {
                    if (aborted) return;
                    // Trust boundary: partialHtml is the server's
                    // autoescaped render of _spectra_inspector.html.
                    // No user input flows into the template; this is
                    // the same content /spectra sees via Jinja
                    // include.
                    host.innerHTML = partialHtml;
                    // Call into the shared core.  ``mount`` returns
                    // ``{dispose()}``; we keep the handle so dispose()
                    // can chain to its inner dispose().
                    innerHandle = api.mount(host, { file: file });
                    cleanups.push(() => {
                        try { innerHandle.dispose(); }
                        catch (_) { /* already torn down */ }
                    });
                })
                .catch((err) => {
                    if (err.name === "AbortError") return;
                    _renderError(host, file,
                        (err && err.message) || String(err));
                });

            return {
                dispose() {
                    aborted = true;
                    // Walk cleanups in reverse so the inner handle
                    // (added last) tears down before the
                    // AbortController (added first).
                    for (let i = cleanups.length - 1; i >= 0; i -= 1) {
                        try { cleanups[i](); }
                        catch (_) { /* swallow per-cleanup */ }
                    }
                    host.innerHTML = "";
                },
            };
        },
    };

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.spectraInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
