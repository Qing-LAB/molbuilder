/* lib/page-prefs.js -- WHAT A PAGE KEEPS IN THIS BROWSER: THE ONE DOOR
 * (`web/runtime.md` § 3).
 *
 * Every value a page keeps in the browser goes through here, nowhere else:
 * its layout (a width, a panel open or collapsed, a filter, a reading
 * theme), the sidebar's place, the Structure-optimization tab's forms, and
 * the browser tab's id the workspace names its files with (`workspace.md`
 * § 4).  Every key `molbuilder.` and its name.
 *
 * A value lasts this browser tab's session -- its own, beside no other
 * browser tab's (a tab the browser duplicates starts with a copy);
 * `{lasting: true}` keeps it in this browser after it closes.  Storage the browser
 * refuses (a private window, a full quota) reads as nothing kept, and `write`
 * answers false: the page works without it.
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

    /** ``name`` as last written, or null when nothing is kept. */
    function read(name, opts) {
        try { return _store(opts).getItem(PREFIX + name); }
        catch (_) { return null; }
    }

    /** Keep ``value`` (a string, "" among them) as ``name``; null drops
     *  it.  True when the browser kept it, false when it refused. */
    function write(name, value, opts) {
        try {
            var store = _store(opts);
            if (value === null || value === undefined) {
                store.removeItem(PREFIX + name);
            } else {
                store.setItem(PREFIX + name, String(value));
            }
            return true;
        } catch (_) { return false; }
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.pagePrefs = Object.freeze({ read: read, write: write });
})(typeof window !== "undefined" ? window : this);
