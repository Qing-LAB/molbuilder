/* lib/server.js — THE ONE READER of a server's answer.
 *
 * Contract: docs/web/ui-contract.md § 12.
 * Owns:     asking the server and reading what it answers -- the one place
 *           that knows a refusal from an unreadable answer, an unreachable
 *           server, a cancelled question and a person signed out -- and
 *           saying *signed out* once for the page.
 * Called by: every page and module that asks the server a question
 *           (`lib/projects/api.js` among them); never a page's own
 *           ``fetch(...).json()``.
 *
 * NEVER: decide what a page shows for a refusal -- the page says it beside
 *        what asked; paint a failed answer as data.
 *
 * Public API (ES module; mirrored on window.molbuilder.server, and
 * registered with the page runtime as "server"):
 *   ask(url, {method, body, signal, cache, as}) -> Promise<answer>
 *     body    an object (sent as JSON), or a FormData / Blob / string as is;
 *     as      "text" for a question whose yes is not JSON (a partial);
 *             "file" for one whose answer is a file -- its bytes as a Blob,
 *             and the answer's ``headers`` beside them (its name, what the
 *             server said); a refusal is read as any other;
 *     answer  {ok: true, status, body, headers?}
 *             | {ok: false, status, error, body?, aborted?, signedOut?}
 */

let _signedOutSaid = false;

// ...THE SIGN-IN PAGE, by the path the gate sends a browser to
// (ops/access-control.md § 3.2).
function _isSignIn(url) {
    try { return new URL(url).pathname.replace(/\/+$/, "") === "/login"; }
    catch (_) { return false; }
}

function _aborted(err, signal) {
    return (err && err.name === "AbortError") || !!(signal && signal.aborted);
}

/* SIGNED OUT, said once for the page: the notification's link is the page
 * itself in a new tab -- a page request the gate sends through sign-in and
 * back here -- so what this tab holds stays as it is. */
function _signedOut(status, body) {
    if (!_signedOutSaid && typeof window === "object" && window) {
        const notify = window.molbuilder && window.molbuilder.notify;
        if (notify && typeof notify.show === "function") {
            _signedOutSaid = true;
            notify.show({
                id: "signed-out", level: "warn",
                message: "You are signed out. What this page holds is kept: "
                    + "sign in in the new tab, then carry on here.",
                action: { label: "Sign in", newTab: true,
                          href: window.location.pathname
                                + window.location.search },
            });
        }
    }
    return { ok: false, status, signedOut: true, body: body || null,
             error: "you are signed out" };
}

function _unreadable(status) {
    return { ok: false, status,
             error: "the server's answer could not be read (HTTP " + status
                 + ") \u2014 its log says why" };
}

export async function ask(url, opts) {
    const o = opts || {};
    const init = { method: o.method || "GET", cache: o.cache || "no-store",
                   signal: o.signal, headers: {} };
    if (o.body !== undefined && o.body !== null) {
        const raw = (typeof FormData !== "undefined" && o.body instanceof FormData)
            || (typeof Blob !== "undefined" && o.body instanceof Blob)
            || typeof o.body === "string";
        init.body = raw ? o.body : JSON.stringify(o.body);
        if (!raw) init.headers["Content-Type"] = "application/json";
    }
    let r;
    try {
        r = await fetch(url, init);
    } catch (err) {
        if (_aborted(err, o.signal)) {
            return { ok: false, status: 0, aborted: true, error: "cancelled" };
        }
        // ...SAID PLAINLY: the browser's own words ("Failed to fetch") tell
        // a person nothing they can act on (molview.md § 6.9).
        return { ok: false, status: 0,
                 error: "could not reach the server \u2014 check the "
                     + "connection, then try again" };
    }
    // ...SENT TO THE SIGN-IN PAGE: a question the gate answered as a page.
    if (r.redirected && _isSignIn(r.url)) return _signedOut(r.status, null);
    if (o.as === "file" && r.ok) {
        try {
            return { ok: true, status: r.status, body: await r.blob(),
                     headers: r.headers };
        } catch (err) {
            if (_aborted(err, o.signal)) {
                return { ok: false, status: r.status, aborted: true,
                         error: "cancelled" };
            }
            return _unreadable(r.status);
        }
    }
    let text;
    try {
        text = await r.text();
    } catch (err) {
        if (_aborted(err, o.signal)) {
            return { ok: false, status: r.status, aborted: true,
                     error: "cancelled" };
        }
        return _unreadable(r.status);
    }
    // ...A PART OF A PAGE, asked as text: its yes is the text.  A refusal is
    // read as any other, below.
    if (o.as === "text" && r.ok) return { ok: true, status: r.status, body: text };
    let body = null;
    if (text) {
        try { body = JSON.parse(text); }
        catch (_) { return _unreadable(r.status); }
    }
    if (r.status === 401 && body && body.login_url) {
        return _signedOut(401, body);
    }
    if (!r.ok || (body && body.ok === false)) {
        return { ok: false, status: r.status, body,
                 error: (body && body.error) ? String(body.error)
                     : "the server refused (HTTP " + r.status + ")" };
    }
    return { ok: true, status: r.status, body };
}

// ...MIRRORED for a classic script, and REGISTERED with the page's runtime:
// a classic script that asks while the page is still loading -- before a
// module has run -- waits on ``runtime.whenReady("server")``.
if (typeof window === "object" && window) {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.server = { ask };
    const runtime = window.molbuilder.runtime;
    if (runtime && typeof runtime.register === "function") {
        runtime.register("server", window.molbuilder.server);
    }
}
