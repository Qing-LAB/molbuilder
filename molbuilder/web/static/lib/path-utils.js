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

    /**
     * Compute ``target`` expressed relatively to ``fromDir``.
     *
     * The portable form we want to persist into UI state / future
     * sidecar files (privacy + survives copying the whole project
     * tree to another machine).  Example: from
     *   ``/h/u/molbuilder/projects/proj/topic/struct``
     * to
     *   ``/h/u/molbuilder/projects/pseudopotential``
     * returns ``../../../pseudopotential``.
     *
     * Algorithm: split both on "/", strip the common prefix, then
     * emit one ``..`` per remaining fromDir segment + the leftover
     * target segments.  Falsy inputs or a non-absolute target are
     * returned unchanged (caller's preference -- relative inputs
     * are typically the portable form already).
     *
     * Notes:
     *   * Both args must be POSIX-style absolute paths.  No
     *     Windows / drive-letter / scheme handling -- molbuilder
     *     is Unix-only.
     *   * Doesn't touch the filesystem (no ``..`` collapsing
     *     beyond what the prefix-strip naturally does).
     *   * Returns "." when ``target`` IS ``fromDir`` (the empty
     *     relative path is "." in POSIX shells / Python).
     */
    function relativeFromDir(target, fromDir) {
        if (!target || !fromDir) return target || "";
        if (target.charAt(0) !== "/") return target;     // already relative
        const tgtParts = target.replace(/\/+$/, "").split("/");
        const dirParts = fromDir.replace(/\/+$/, "").split("/");
        let i = 0;
        const n = Math.min(tgtParts.length, dirParts.length);
        while (i < n && tgtParts[i] === dirParts[i]) i++;
        const upSegments = dirParts.slice(i).filter(Boolean).length;
        const downSegments = tgtParts.slice(i);
        const parts = [];
        for (let k = 0; k < upSegments; k++) parts.push("..");
        for (const s of downSegments) parts.push(s);
        return parts.length ? parts.join("/") : ".";
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.path = root.molbuilder.path || {};
    root.molbuilder.path.basename = basename;
    root.molbuilder.path.relativeFromDir = relativeFromDir;
})(typeof window !== "undefined" ? window : this);
