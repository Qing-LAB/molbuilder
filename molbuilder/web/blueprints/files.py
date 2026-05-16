"""Files blueprint -- server-side file explorer for the Projects tab.

Routes:

    GET  /api/files/roots               list of allowed root paths
    GET  /api/files/list?path=...&ext=  directory listing
    GET  /api/files/stat?path=...       file metadata
    GET  /api/files/read?path=...       text contents (size-capped)

Full contract:  docs/protocols/web-api.md  §  ``/api/files/*``

Path validation
---------------
Every endpoint that takes a ``path`` query parameter runs it through
:func:`_resolve_within_roots`, which:

  1. Expands ``~`` and ``$VARS``.
  2. Resolves to absolute (follows symlinks).
  3. Rejects raw ``..`` components defense-in-depth (the resolution
     step already prevents escaping the roots, but rejecting ``..``
     in the user-supplied string gives a cleaner error than waiting
     for the equal-or-inside-a-root check to fail).
  4. Requires the resolved path to equal-or-be-inside one of the
     allowed roots, as defined by
     :meth:`Capabilities.file_picker_roots`.

Anything that fails validation gets HTTP 400 with a JSON error.

Design note: this blueprint is intentionally read-only.  No create,
rename, move, delete.  Mutations go through other paths -- generators
(Build / Spectra), the run wrapper, the future derive-job endpoint.
The picker is a *navigation + selection* widget, not a file manager.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request

from molbuilder import diagnostics

bp = Blueprint("files", __name__)


# --------------------------------------------------------------------- #
#  Page route                                                           #
# --------------------------------------------------------------------- #


@bp.route("/projects")
def projects_page():
    """Projects tab -- column-view file explorer."""
    return render_template("projects.html", active_tab="projects")


# Default cap for read_file responses: 1 MB.  Spectra JSONs are
# typically tens of KB; molwatch logs can be much larger but the
# picker preview should not try to render the whole thing.  Callers
# that need more can request explicitly up to MAX_READ_BYTES.
_DEFAULT_READ_BYTES = 1 * 1024 * 1024
_MAX_READ_BYTES     = 16 * 1024 * 1024   # hard ceiling per request


# --------------------------------------------------------------------- #
#  Validation helpers                                                   #
# --------------------------------------------------------------------- #


class _PickerError(Exception):
    """Internal: raised by _resolve_within_roots, caught at the route
    boundary and translated to a JSON 400."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _allowed_roots() -> Tuple[Tuple[Path, str], ...]:
    """Roots the picker is allowed to browse.

    Cached via the Capabilities singleton -- updates only when
    ``diagnostics.initialize()`` is called again (i.e., process
    restart in normal use).  Tests can substitute via
    :func:`diagnostics.set_capabilities`.
    """
    return diagnostics.get_capabilities().file_picker_roots()


def _resolve_within_roots(raw_path: str) -> Path:
    """Resolve ``raw_path`` to an absolute Path that lies inside an
    allowed root.

    Raises :class:`_PickerError` with status 400 if the path is
    missing, contains ``..``, or resolves outside every allowed root.
    Raises with status 404 if the resolved path doesn't exist on disk.
    """
    if not raw_path:
        raise _PickerError(400, "missing 'path' query parameter")

    # Defense in depth: reject .. in the raw string.  Resolution will
    # also normalise it away, but an early reject avoids ambiguity
    # ("did the user think .. was harmless?").
    if ".." in Path(raw_path).parts:
        raise _PickerError(400, f"path may not contain '..': {raw_path!r}")

    expanded = Path(os.path.expandvars(raw_path)).expanduser()
    try:
        resolved = expanded.resolve()
    except (OSError, RuntimeError) as exc:
        raise _PickerError(400, f"could not resolve path: {exc}")

    roots = _allowed_roots()
    if not roots:
        raise _PickerError(
            400,
            "no file-picker roots are configured (Capabilities snapshot "
            "should provide at least projects/ + CWD; check that "
            "diagnostics.initialize() ran at startup).",
        )

    for root_path, _label in roots:
        try:
            resolved.relative_to(root_path)
        except ValueError:
            continue
        # Inside an allowed root.  Existence is checked by the caller
        # (list / stat / read have different expectations -- list-of-
        # missing should 404, but the user might want to stat a file
        # they expect to appear shortly).
        return resolved

    raise _PickerError(
        400,
        f"path {str(resolved)!r} is outside every configured root; "
        f"allowed roots: {[str(p) for p, _ in roots]}",
    )


def _entry_dict(p: Path) -> dict:
    """One entry in a /api/files/list response.

    Calls ``stat`` once and tolerates symlinks-to-nowhere by reporting
    them as ``kind=symlink, size=null``.  Dotfiles are filtered out
    upstream in :func:`_list_entries`.
    """
    try:
        st = p.stat()
    except OSError:
        # Broken symlink or permission denied -- still report the
        # entry so the user can see it exists but in a broken state.
        return {
            "name": p.name, "kind": "other",
            "size": None, "mtime": None,
        }

    if p.is_dir():
        kind = "directory"
        size = None
    elif p.is_file():
        kind = "file"
        size = st.st_size
    elif p.is_symlink():
        kind = "symlink"
        size = st.st_size
    else:
        kind = "other"
        size = None

    return {
        "name":  p.name,
        "kind":  kind,
        "size":  size,
        "mtime": st.st_mtime,
    }


def _list_entries(d: Path, ext_filter: Optional[List[str]]) -> List[dict]:
    """Sorted directory listing.  Hidden entries (leading dot) filtered.

    ``ext_filter`` is a list of lower-cased extensions (with leading
    dot, e.g. ``[".xyz", ".pdb"]``).  Directories are never filtered
    by extension (they need to be reachable so the user can navigate
    into subdirs containing the requested file types).
    """
    out: List[dict] = []
    for entry in d.iterdir():
        if entry.name.startswith("."):
            continue
        if (ext_filter is not None
                and entry.is_file()
                and entry.suffix.lower() not in ext_filter):
            continue
        out.append(_entry_dict(entry))
    # Directories first, then files; each group sorted by name
    # (case-insensitive) so the UI tree is stable across runs.
    out.sort(key=lambda e: (e["kind"] != "directory", e["name"].lower()))
    return out


# --------------------------------------------------------------------- #
#  Routes                                                               #
# --------------------------------------------------------------------- #


@bp.route("/api/files/roots", methods=["GET"])
def api_files_roots():
    """List the absolute root paths the picker is allowed to browse."""
    roots = _allowed_roots()
    return jsonify({
        "ok": True,
        "roots": [
            {
                "path":   str(p),
                "label":  label,
                "exists": p.exists(),
            }
            for p, label in roots
        ],
    })


@bp.route("/api/files/list", methods=["GET"])
def api_files_list():
    """List entries in a directory.

    Query params:
        path   -- absolute or relative path; must resolve inside a root.
        ext    -- optional comma-separated extensions to filter files by
                  (e.g., ``ext=.xyz,.pdb``).  Directories are always shown.
    """
    raw_path = request.args.get("path", "")
    raw_ext  = request.args.get("ext", "")
    try:
        resolved = _resolve_within_roots(raw_path)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not resolved.exists():
        return jsonify({
            "ok": False,
            "error": f"path does not exist: {str(resolved)!r}",
        }), 404
    if not resolved.is_dir():
        return jsonify({
            "ok": False,
            "error": f"path is not a directory: {str(resolved)!r}",
        }), 400

    ext_filter: Optional[List[str]] = None
    if raw_ext:
        # Normalise: lower-case, ensure leading dot.
        ext_filter = []
        for raw in raw_ext.split(","):
            raw = raw.strip().lower()
            if not raw:
                continue
            if not raw.startswith("."):
                raw = "." + raw
            ext_filter.append(raw)

    try:
        entries = _list_entries(resolved, ext_filter)
    except PermissionError as exc:
        return jsonify({
            "ok": False,
            "error": f"permission denied reading directory: {exc}",
        }), 403

    return jsonify({
        "ok":      True,
        "path":    str(resolved),
        "entries": entries,
    })


@bp.route("/api/files/stat", methods=["GET"])
def api_files_stat():
    """Metadata for a single path (file or directory)."""
    raw_path = request.args.get("path", "")
    try:
        resolved = _resolve_within_roots(raw_path)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not resolved.exists():
        return jsonify({
            "ok": False,
            "error": f"path does not exist: {str(resolved)!r}",
        }), 404

    entry = _entry_dict(resolved)
    return jsonify({
        "ok":    True,
        "path":  str(resolved),
        **{k: v for k, v in entry.items() if k != "name"},
    })


@bp.route("/api/files/read", methods=["GET"])
def api_files_read():
    """Read a file as UTF-8 text, capped at ``max_bytes`` (default 1 MB).

    Binary or non-UTF-8 files return a 400 with an explanation -- the
    picker is for previewable text (XYZ, JSON, .py scripts, .molwatch.log,
    .thermo.txt).  Adding base64-for-binary later is a one-line change
    if a real need shows up.
    """
    raw_path = request.args.get("path", "")
    try:
        resolved = _resolve_within_roots(raw_path)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not resolved.exists():
        return jsonify({
            "ok": False,
            "error": f"path does not exist: {str(resolved)!r}",
        }), 404
    if not resolved.is_file():
        return jsonify({
            "ok": False,
            "error": f"path is not a regular file: {str(resolved)!r}",
        }), 400

    try:
        max_bytes = int(request.args.get("max_bytes", _DEFAULT_READ_BYTES))
    except ValueError:
        return jsonify({"ok": False,
                        "error": "max_bytes must be an integer"}), 400
    if max_bytes <= 0 or max_bytes > _MAX_READ_BYTES:
        return jsonify({
            "ok": False,
            "error": f"max_bytes must be in (0, {_MAX_READ_BYTES}]; "
                     f"got {max_bytes}",
        }), 400

    try:
        st = resolved.stat()
    except OSError as exc:
        return jsonify({"ok": False, "error": f"stat failed: {exc}"}), 500

    if st.st_size > max_bytes:
        # Honest 413 -- the caller can re-request with a larger
        # max_bytes (up to _MAX_READ_BYTES) if they really want it.
        return jsonify({
            "ok":   False,
            "error": f"file is {st.st_size} bytes; exceeds max_bytes "
                     f"= {max_bytes}.  Re-request with larger max_bytes "
                     f"(up to {_MAX_READ_BYTES}) if you need more.",
            "size": st.st_size,
        }), 413

    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return jsonify({
            "ok":   False,
            "error": f"file is not valid UTF-8: {str(resolved)!r}.  "
                     f"The picker only previews text files in v1.",
        }), 400
    except PermissionError as exc:
        return jsonify({
            "ok": False,
            "error": f"permission denied: {exc}",
        }), 403

    return jsonify({
        "ok":    True,
        "path":  str(resolved),
        "kind":  "file",
        "size":  st.st_size,
        "mtime": st.st_mtime,
        "text":  text,
    })


__all__ = ["bp"]
