/* molbuilder shared path-string helpers.
 *
 * Tiny utility module so the five inspector modules under
 * static/lib/inspectors/ (and any future viewer) don't each carry
 * their own copy of basename().  Exposed via
 * ``window.molbuilder.path.{basename}``.
 *
 * Why not just use a JS library?  We're vanilla-JS, no bundler.
 * For the 6 lines basename takes, a focused shared module is the
 * right unit -- not a dependency on Node's path module via a
 * polyfill.
 */
(function (root) {
    "use strict";

    /**
     * Last path component of ``p``.  Tolerates both ``/`` and ``\``
     * separators (multi-OS-safe even though molbuilder is Unix-only
     * today: the user could paste a Windows path into the picker).
     * Returns "" for falsy input, ``p`` itself when there's no
     * separator.
     */
    function basename(p) {
        if (!p) return "";
        const ix = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
        return ix >= 0 ? p.slice(ix + 1) : p;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.path = root.molbuilder.path || {};
    root.molbuilder.path.basename = basename;
})(typeof window !== "undefined" ? window : this);
