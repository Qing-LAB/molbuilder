/* lib/path-utils.js -- THE BROWSER'S ONE PATH DOOR (`web/runtime.md` § 3).
 *
 * The page names files and folders by their full paths, as the server's
 * answers give them; what a page works out of one -- its name, its folder, a
 * name joined to a folder, where it lies in a tree -- is worked out here, and
 * nowhere else.  Path STRINGS only: no file system, no DOM, no server.
 *
 * An ES module, loaded by the page header (`_app_header.html`) and mirrored
 * on `window.molbuilder.path` for a classic script, which reads it when it is
 * called (the pattern of `lib/server.js`).
 */

/** The last part of ``path``: a file's name, a folder's own name.  The
 *  server's paths are POSIX, ``/`` the one separator (a ``\`` is a character
 *  of a name).  "" for nothing given. */
export function basename(path) {
  const p = String(path || "");
  return p.slice(p.lastIndexOf("/") + 1);
}

/** The folder ``path`` lies in: everything before its last part, "" when it
 *  names no folder. */
export function dirname(path) {
  const p = String(path || "");
  const ix = p.lastIndexOf("/");
  return ix >= 0 ? p.slice(0, ix) : "";
}

/** ``name`` in the folder ``folder``: one separator between them, however
 *  the folder ends. */
export function join(folder, name) {
  return String(folder || "").replace(/\/+$/, "") + "/" + String(name || "");
}

/** Where ``path`` lies under the folder ``root``: "" for the folder itself,
 *  its parts below it joined by ``/``, or ``null`` for a path outside it.
 *  INSIDE IS THE FOLDER OR BELOW IT, never a name that merely starts the
 *  same: ``/a/projects2`` is not inside ``/a/projects``. */
export function relative(path, root) {
  const p = String(path || "").replace(/\/+$/, "");
  const r = String(root || "").replace(/\/+$/, "");
  if (!p || !r) return null;
  if (p === r) return "";
  return p.startsWith(r + "/") ? p.slice(r.length + 1) : null;
}

/** Whether ``path`` is the folder ``root`` or lies below it
 *  (:func:`relative`). */
export function isInside(path, root) {
  return relative(path, root) !== null;
}

if (typeof window !== "undefined") {
  window.molbuilder = window.molbuilder || {};
  window.molbuilder.path = { basename, dirname, join, relative, isInside };
}
