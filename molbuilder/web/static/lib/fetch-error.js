/**
 * fetch-error.js — one sentence for a failed fetch, in one place.
 *
 * The distinction that earns this its own module: a fetch can fail
 * because the NETWORK did (offline, DNS, CORS preflight) or because
 * the SERVER did and answered with something that is not JSON (a 5xx
 * HTML error page, a proxy's plain-text drop, a stubbed 501).  Both
 * arrive at the caller as a rejected promise, and reporting the
 * second as "Network error: Unexpected token <" sends a chemist to
 * check their wifi while the server sits there crashed.
 *
 * Extracted 2026-08-22 (roadmap 7.2).  It had been written twice —
 * `structure-optimization/viewer.js` had it, and `lib/auto-detect.js`
 * gained a copy the same day, while extracting a triplicated renderer.
 * A rule about what a failure MEANS must not be able to hold two
 * opinions.
 *
 * NOT the same thing as `lib/projects/api.js`'s `_fetchEnvelope`, which
 * makes the same distinction while normalising a whole response into
 * `{ok, ...}`.  That is an envelope contract; this is one message.
 * Left as it is rather than folded in: they answer different questions,
 * and merging them would put a response shape and a sentence in one
 * function.
 *
 * Exports (on window.molbuilder.fetchError):
 *   format(e) → string
 */
(function () {
    "use strict";

    var root = (typeof globalThis !== "undefined") ? globalThis
            : (typeof window !== "undefined") ? window : this;

    /**
     * Turn a fetch/parse rejection into something a user can act on.
     *
     * AbortError never reaches here — a supersede is the caller's to
     * filter, and reporting it would flash an error on an ordinary
     * second click.
     */
    function format(e) {
        if (e && e.name === "SyntaxError") {
            return "Server returned non-JSON response "
                 + "(likely a 5xx error page).  Check the server "
                 + "log for the actual failure.";
        }
        return "Network error: "
             + (e && e.message ? e.message : String(e));
    }

    var api = { format: format };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.fetchError = api;
})();
