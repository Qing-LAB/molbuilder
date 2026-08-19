/* Reload the server, from the browser.
 *
 * MODULE: app-reload (classic script; publishes nothing).
 * Contract: docs/archive/2026-08-19-server-reload-plan.md § 3.3.
 *
 * The server restarts by EXITING with a code its supervisor is waiting for, so
 * from here it is three steps: ask, wait for the socket to answer again, reload
 * the page.  There is no websocket and nothing to keep open -- the connection
 * this page has is about to be dropped by design.
 *
 * WHY THE BUTTON IS DRAWN HIDDEN AND REVEALED, never the reverse: a control that
 * appears and then vanishes reads as a permission being taken away.  Almost
 * every session will never see it -- the route exists only when the server runs
 * under a supervisor AND ``rate_limit.admin_emails`` names somebody -- so the
 * quiet case has to be the tidy one.
 */
(function (root) {
    "use strict";

    var BTN_ID   = "app-reload";
    var POLL_MS  = 500;
    /* Long enough to cover a slow import of the whole app, short enough that a
     * server which never comes back stops pretending it might.  A child that
     * fails to import leaves the SUPERVISOR alive but no server listening, and
     * saying so is the only honest thing this page can do about it. */
    var GIVE_UP_MS = 60000;

    function _btn() { return root.document.getElementById(BTN_ID); }

    function _notify(level, message) {
        var n = root.molbuilder && root.molbuilder.notify;
        if (n && typeof n.show === "function") {
            n.show({ id: "server-reload", level: level, message: message });
            return;
        }
        // The notification bar is on every page that has this button, so this
        // is a belt: better a console line than a silent failure.
        if (root.console) root.console.log("[reload] " + message);
    }

    /* Ask the server whether it is still down.  A FAILED fetch is the expected
     * answer while it restarts -- the socket is closed -- so a rejection here is
     * progress, not an error, and must not reach the console as one. */
    function _isUp() {
        return root.fetch("/api/health", { cache: "no-store" })
            .then(function (r) { return r.ok; })
            .catch(function () { return false; });
    }

    function _waitForServer(deadline) {
        return _isUp().then(function (up) {
            if (up) return true;
            if (Date.now() > deadline) return false;
            return new Promise(function (resolve) {
                root.setTimeout(resolve, POLL_MS);
            }).then(function () { return _waitForServer(deadline); });
        });
    }

    function _reload() {
        var btn = _btn();
        /* NAME THE COST BEFORE DOING IT. This disconnects everyone using this
         * server, and a workspace write that is still in flight is lost --
         * `persist` does not wait for the server (workspace.md § 6), so "sent"
         * is not "saved". */
        var ok = root.confirm(
            "Restart the server?\n\n"
            + "Everyone using it is disconnected while it comes back, and "
            + "saves that are still in flight are lost.");
        if (!ok) return;

        if (btn) btn.disabled = true;
        _notify("info", "Restarting the server…");

        root.fetch("/api/admin/reload", { method: "POST" })
            .then(function (r) {
                if (r.status === 403) {
                    throw new Error("this session is not an admin");
                }
                if (!r.ok) {
                    throw new Error("the server refused (HTTP " + r.status + ")");
                }
                return _waitForServer(Date.now() + GIVE_UP_MS);
            })
            .then(function (came_back) {
                if (came_back) {
                    // The page is re-fetched, and every asset revalidates with
                    // it, so new JS and CSS arrive too (server-reload-plan § 2).
                    root.location.reload();
                    return;
                }
                _notify("error",
                    "The server did not come back within a minute. Its "
                    + "supervisor is still running, so the new code most "
                    + "likely fails to import — check the terminal.");
                if (btn) btn.disabled = false;
            })
            .catch(function (err) {
                _notify("error", "Could not restart the server: "
                                 + ((err && err.message) || String(err)));
                if (btn) btn.disabled = false;
            });
    }

    function _start() {
        var btn = _btn();
        if (!btn) return;                       // no auth session, no button
        root.fetch("/api/admin/reload/available", { cache: "no-store" })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (body) {
                if (!body || !body.available) return;   // stays hidden
                btn.hidden = false;
                btn.addEventListener("click", _reload);
            })
            .catch(function () { /* stays hidden, which is the safe state */ });
    }

    if (root.document.readyState === "loading") {
        root.document.addEventListener("DOMContentLoaded", _start);
    } else {
        _start();
    }
})(typeof window !== "undefined" ? window : globalThis);
