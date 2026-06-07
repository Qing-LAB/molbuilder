/* Trajectory inspector -- thin wrapper around lib/trajectory/core.js.
 *
 * Match rule: ``*.molwatch.log``.
 *
 * Mount flow:
 *   1. Fetch the inspector's DOM markup from
 *      ``GET /partials/trajectory-inspector`` (rendered server-side
 *      from ``_trajectory_inspector.html`` -- same partial /watch
 *      includes via Jinja, single source of truth).
 *   2. Assign the response text to ``host.innerHTML``.  Trusted:
 *      it's a same-origin Jinja-autoescaped render of a static
 *      template with no user input.
 *   3. Call ``window.molbuilder.trajectoryInspector.mount(host, {file})``
 *      to wire the viewer + plots against the newly-injected DOM.
 *   4. Return a handle whose ``dispose()`` chains the inner
 *      handle's dispose + clears the host.
 *
 * The actual inspector logic lives in lib/trajectory/core.js (loaded
 * earlier in results.html).  Both /watch and /results use that same
 * module; this wrapper is the registry-side adapter only.
 */
(function (root) {
    "use strict";

    const PARTIAL_URL = "/partials/trajectory-inspector";

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
        title.textContent = "Trajectory inspector failed to mount";
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
        name:        "trajectory",
        displayName: "Trajectory + SCF history",
        isResult:    true,
        // The match function claims every file extension a backend
        // ``TrajectoryParser`` knows how to parse (see
        // ``molbuilder/parsers/__init__.py::PARSERS``):
        //
        //   * ``.molwatch.log`` -- the canonical molbuilder format
        //     (MolwatchLogParser).  Compound extension, registered
        //     BEFORE plain ``.log`` claimants -- first-match-wins
        //     in the registry guarantees this wins.
        //   * ``.out`` -- SIESTA's redirected-stdout output
        //     (SiestaParser; content-based can_parse, so
        //     non-SIESTA .out files will fail to load with a
        //     clear server-side error rather than mis-render).
        //
        // NOT claimed here (intentional):
        //   * ``.pyscf.log`` -- PySCF wrapper stdout (Phase C,
        //     2026-06-07 rename from .out).  Plain stdout text;
        //     not a trajectory format.  Falls through to the
        //     source inspector (text viewer).  A dedicated
        //     ``pyscf-log`` inspector that parses PySCF stdout +
        //     renders a result panel is on the roadmap; until
        //     then the source viewer is the right surface.
        //   * ``.log`` (plain, non-compound) -- too generic.
        //
        // The trajectory inspector renders frames + energies + force
        // arrows + SCF history from whatever parser detect_parser
        // picks at /api/watch/load time.  The frontend doesn't care
        // which parser ran -- TrajectoryParser is the contract.
        match: (file) => {
            const lower = file.toLowerCase();
            return lower.endsWith(".molwatch.log")
                || lower.endsWith(".out");
        },
        // Two different engine outputs land here; the picker groups
        // them under distinct headers so the user can scan visually.
        // ``.out`` -> SIESTA (siesta wrapper redirects stdout into
        // ``<base>-runN.out``); ``.molwatch.log`` -> the unified
        // molwatch format emitted by the PySCF geometry-opt
        // wrapper.  Both surface in the trajectory inspector but
        // they're semantically distinct workflows.
        resultCategory: (file) => {
            const lower = file.toLowerCase();
            if (lower.endsWith(".molwatch.log")) return "PySCF optimization";
            if (lower.endsWith(".out"))           return "SIESTA optimization";
            return "Trajectory";  // fallback (shouldn't fire given match())
        },

        mount(host, file, _ctx) {
            // Cleanup chain.  Walked in reverse by ``dispose``;
            // each entry is a function the wrapper accumulates as
            // it sets up the inner mount.
            const cleanups = [];

            // Defensive: core.js must be loaded BEFORE this script
            // (see results.html's script ordering).  If it's missing
            // (broken deployment, CDN issue, etc.) fail loudly with a
            // visible error instead of leaving a half-mounted UI.
            const api = (root.molbuilder || {}).trajectoryInspector;
            if (!api || typeof api.mount !== "function") {
                _renderError(host, file,
                    "lib/trajectory/core.js did not load -- check "
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
                    // autoescaped render of _trajectory_inspector.html.
                    // No user input flows into the template; this is
                    // the same content /watch sees via Jinja include.
                    host.innerHTML = partialHtml;
                    // Call into the shared core.  ``mount`` returns
                    // ``{dispose(), load(path)}``; we keep the handle
                    // so dispose() can chain to its inner dispose().
                    innerHandle = api.mount(host, { file: file });
                    cleanups.push(() => {
                        try { innerHandle.dispose(); }
                        catch (_) { /* already torn down */ }
                    });
                    // Result-file picker dropdown lifted to the
                    // /results tab header (2026-06-01); inspectors
                    // no longer mount their own.  See
                    // ``lib/results/file-picker.js``.
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
                    // (added last) tears down before the AbortController
                    // (added first).
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
    root.molbuilder.inspectors.trajectoryInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
