/* lib/page-prefs.js -- A PAGE'S OWN LAYOUT, KEPT IN THIS BROWSER
 * (`web/runtime.md` § 3).
 *
 * WHERE A PAGE'S STATE LIVES, by what it is (the user, 2026-11-10):
 *   - how you look at a structure or a spectrum lives beside the work, in the
 *     workspace (`workspace.md`; `molview.md` § 11.2b, `spectra.md` § 7);
 *   - a page's own layout -- a width, a panel open or collapsed, a filter, a
 *     reading theme -- lives in this browser, through here: every key
 *     `molbuilder.` and its name, nowhere a key of a page's own.
 *
 * A preference lasts this browser session; `{lasting: true}` keeps it after
 * the browser closes.  Storage the browser refuses (a private window, a full
 * quota) reads as nothing kept and writes nothing: a preference is a
 * convenience, and the page works without it.
 *
 * A CLASSIC SCRIPT, loaded early by the page header (`_app_header.html`), so
 * a page's own classic script can read its preference as it starts; an ES
 * module reads `window.molbuilder.pagePrefs` the same way.
 */
(function (root) {
    "use strict";

    var PREFIX = "molbuilder.";

    function _store(opts) {
        return (opts && opts.lasting) ? root.localStorage : root.sessionStorage;
    }

    /** The preference ``name`` as last written, or null when none is kept. */
    function read(name, opts) {
        try { return _store(opts).getItem(PREFIX + name); }
        catch (_) { return null; }
    }

    /** Keep ``value`` (a string) as the preference ``name``; null or "" drops
     *  it. */
    function write(name, value, opts) {
        try {
            var store = _store(opts);
            if (value === null || value === undefined || value === "") {
                store.removeItem(PREFIX + name);
            } else {
                store.setItem(PREFIX + name, String(value));
            }
        } catch (_) { /* refused by the browser: kept for this page only */ }
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.pagePrefs = Object.freeze({ read: read, write: write });
})(typeof window !== "undefined" ? window : this);
