"""Files blueprint -- server-side file explorer for the Projects sidebar.

Routes:

    GET  /api/files/roots               list of allowed root paths
    GET  /api/files/list?path=...&ext=  directory listing
    GET  /api/files/stat?path=...       file metadata
    GET  /api/files/read?path=...       text contents (size-capped)
    POST /api/files/mkdir               create a new subdirectory (validated name)
    POST /api/projects/create           bootstrap a new project with every
                                        canonical subdir (atomic, strict conflict)
    POST /api/files/upload              (stub) multipart upload -- 501 in v1
    POST /api/files/write               (stub) save edited text -- 501 in v1
    DELETE /api/files/delete            (stub) remove file or empty dir -- 501 in v1

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

from flask import Blueprint, jsonify, request

from molbuilder import diagnostics
from molbuilder.projects import (
    CANONICAL_TOPICS, InvalidName, ProjectExists,
    populate_project_skeleton, projects_root,
    validate_name, validate_topic,
)

bp = Blueprint("files", __name__)


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


# --------------------------------------------------------------------- #
#  /api/files/mkdir                                                     #
# --------------------------------------------------------------------- #


def _validate_subdir_name(parent_abs: Path, name: str) -> None:
    """Reject names that violate the projects-hierarchy naming rules.

    The rules depend on the parent path's depth inside ``projects/``:

      * Directly under ``projects/`` -> the name becomes a *project*
        (e.g. ``projects/<name>``).  ``validate_name`` applies the
        ``^[A-Za-z0-9_-]+$`` regex.
      * Under ``projects/<project>/`` -> the name is a *topic* and must
        be one of :data:`molbuilder.projects.CANONICAL_TOPICS`.
        ``validate_topic`` enforces this.
      * Under ``projects/<project>/<topic>/`` -> the name is a
        *structure*; same regex as project.
      * Deeper than that -- inside a structure dir, which is supposed
        to be flat per job-layout v1 -- the name is allowed as an
        ad-hoc subdir (the same name regex), but the design.md
        2026-05-14 row warns against this convention-wise.

    "Depth inside projects/" is computed relative to whichever root
    surfaced the parent path -- the picker's single root from
    ``Capabilities.file_picker_roots()`` IS the projects/ root, so
    we use that here.  This decouples the validator from the
    real-cwd projects_root() and lets tests substitute a tmp tree.

    Raises :class:`molbuilder.projects.InvalidName` on rejection.
    """
    roots = _allowed_roots()
    root = None
    for root_path, _label in roots:
        try:
            parent_abs.relative_to(root_path)
        except ValueError:
            continue
        root = root_path
        break
    if root is None:
        raise InvalidName(
            f"parent {parent_abs!s} is outside the picker's roots; "
            f"the picker shouldn't have surfaced it."
        )
    rel_parts = parent_abs.relative_to(root).parts
    depth = len(rel_parts)
    if depth == 1:
        # Parent is projects/<project>/ -- the new name is a topic.
        validate_topic(name)
    else:
        # All other cases use the same regex (project name, structure
        # name, ad-hoc subdir).
        validate_name(
            name,
            kind=("project" if depth == 0
                  else "structure" if depth == 2
                  else "subdir"),
        )


@bp.route("/api/files/mkdir", methods=["POST"])
def api_files_mkdir():
    """Create a new subdirectory inside an allowed root.

    JSON body: ``{"parent": "<abs-path>", "name": "<new-dir-name>"}``

    Validation:
      1. ``parent`` must resolve inside an allowed root (same check as
         every read endpoint).
      2. ``name`` must satisfy the naming rule for its depth inside
         ``projects/`` -- see :func:`_validate_subdir_name`.
      3. The new path must not already exist (409 Conflict if it does).

    On success: returns ``{ok, path}`` with the absolute path of the
    new directory.  The sidebar's response is to navigate into it.
    """
    body = request.get_json(silent=True) or {}
    parent_raw = body.get("parent", "")
    name = body.get("name", "")

    try:
        parent = _resolve_within_roots(parent_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not parent.is_dir():
        return jsonify({
            "ok": False,
            "error": f"parent path is not a directory: {str(parent)!r}",
        }), 400

    if not isinstance(name, str) or not name:
        return jsonify({
            "ok": False,
            "error": "missing 'name' in request body",
        }), 400

    try:
        _validate_subdir_name(parent, name)
    except InvalidName as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    new_path = parent / name
    if new_path.exists():
        return jsonify({
            "ok":   False,
            "error": f"path already exists: {str(new_path)!r}",
        }), 409

    try:
        new_path.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        # Race condition with another writer; mirrors the 409 above.
        return jsonify({
            "ok":   False,
            "error": f"path already exists: {str(new_path)!r}",
        }), 409
    except PermissionError as exc:
        return jsonify({
            "ok": False,
            "error": f"permission denied: {exc}",
        }), 403
    except OSError as exc:
        return jsonify({
            "ok": False,
            "error": f"mkdir failed: {exc}",
        }), 500

    return jsonify({
        "ok":   True,
        "path": str(new_path),
    })


# --------------------------------------------------------------------- #
#  /api/projects/create -- "+ New project" bootstrap                    #
# --------------------------------------------------------------------- #


@bp.route("/api/projects/create", methods=["POST"])
def api_projects_create():
    """Create ``projects/<name>/`` with every CANONICAL_TOPICS subdir.

    JSON body: ``{"name": "<project-name>"}``

    Validation:
      1. ``name`` must satisfy ``validate_name`` (``^[A-Za-z0-9_-]+$``).
      2. ``projects/<name>/`` must NOT already exist; 409 if it does
         (strict-create -- to add to an existing project, use
         ``/api/files/mkdir`` from inside it).
      3. The picker's root must resolve to the canonical
         ``projects_root()`` -- the new project lives there.  Tests
         that monkey-patch the picker root also see the new project
         appear at THAT root (the picker root IS the projects/ root).

    Atomic: if any subdir create fails after the project root is
    created, the whole project dir is rmtree'd before raising so
    the user doesn't end up with a half-built tree.

    On success: returns ``{ok, path, subdirs}`` -- ``path`` is the
    absolute path of the new project; ``subdirs`` is the list of
    canonical subdir names created (mirror of ``CANONICAL_TOPICS``
    for the convenience of the frontend's success display).
    """
    body = request.get_json(silent=True) or {}
    name = body.get("name", "")
    if not isinstance(name, str) or not name:
        return jsonify({
            "ok": False,
            "error": "missing 'name' in request body",
        }), 400

    # Where the new project should live.  The picker's single root
    # IS projects_root() at runtime; tests can swap it via a monkey-
    # patched Capabilities, in which case the new project lands at
    # the swapped root.  Use whichever root the picker is currently
    # advertising so server + frontend stay consistent.
    roots = _allowed_roots()
    if not roots:
        return jsonify({
            "ok":   False,
            "error": "no file-picker roots configured; cannot create project",
        }), 500
    root_path = roots[0][0]   # we ship single-root in v1

    try:
        # validate_name first so the error message hits the user
        # before we touch disk.
        validate_name(name, kind="project")
    except InvalidName as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    project_path = root_path / name
    if project_path.exists():
        return jsonify({
            "ok":   False,
            "error": (f"project {name!r} already exists at "
                      f"{str(project_path)!r}.  Pick a different name; "
                      f"use '+ New subdir' from inside the existing "
                      f"project to add to it."),
        }), 409

    # Create the empty project root inside the picker's root, then
    # delegate to populate_project_skeleton to bootstrap the canonical
    # subdirs + READMEs.  This split lets us use the picker's resolved
    # root (which may differ from projects_root() in tests) rather than
    # the cwd-derived path the create_project_skeleton convenience
    # wrapper would assume.
    project_path = root_path / name
    try:
        project_path.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        # Race with the existence check above.
        return jsonify({
            "ok":   False,
            "error": f"project {name!r} already exists.",
        }), 409
    except PermissionError as exc:
        return jsonify({
            "ok": False,
            "error": f"permission denied: {exc}",
        }), 403
    except OSError as exc:
        return jsonify({
            "ok": False,
            "error": f"mkdir failed: {exc}",
        }), 500

    try:
        populate_project_skeleton(project_path, project_name=name)
    except OSError as exc:
        # Rollback on partial failure so the user can retry cleanly.
        import shutil
        shutil.rmtree(project_path, ignore_errors=True)
        return jsonify({
            "ok": False,
            "error": (f"failed to populate project {name!r} ({exc}); "
                      f"the partial project tree has been rolled back."),
        }), 500

    return jsonify({
        "ok":      True,
        "path":    str(project_path),
        "subdirs": list(CANONICAL_TOPICS),
    })


# --------------------------------------------------------------------- #
#  Stubs (501 Not Implemented) -- design captured; behaviour deferred   #
# --------------------------------------------------------------------- #
#
# The frontend renders these endpoints' shape today so the design is
# committed to + the UX surface is reviewable.  When the real feature
# lands, swap the body of each route for the actual implementation.
#
# Why 501 and not a silent no-op:
#   * 501 is the standards-correct response for a documented endpoint
#     whose method is not yet implemented; clients can dispatch on it.
#   * Returning {ok: false} in the standard error shape means the
#     existing inline-error UI in the sidebar renders the message
#     immediately, no special-case branch needed.
#   * Tests pin both the status code (501) AND the error text so a
#     future "oops we shipped the stub" lands loudly.


@bp.route("/api/files/upload", methods=["POST"])
def api_files_upload():
    """Stub: actual upload-to-disk for laptop->server file transfer.

    When implemented (task #56), accepts multipart ``file`` + form
    ``target_dir``.  Restrictions (already captured in
    ``docs/protocols/selection.md``):

      * target_dir must resolve inside an allowed picker root
      * target_dir depth must be >= 1 (no upload at projects/ root)
      * filename validated by a *different* regex than ``validate_name``
        (allows dots for extensions; e.g. ``^[A-Za-z0-9_.-]+$``)
      * inside ``user/`` (depth 2+): free-form
      * name conflict at destination -> 409
    """
    return jsonify({
        "ok": False,
        "error": ("/api/files/upload is not implemented yet (task #56).  "
                  "For now, move files into the projects tree via your "
                  "shell (scp / mv) and refresh the sidebar."),
    }), 501


@bp.route("/api/files/write", methods=["POST"])
def api_files_write():
    """Stub: save edited text content back to a file.

    When implemented, accepts JSON ``{path, text, expected_mtime}``.
    Validation:
      * path must resolve inside an allowed root
      * path must already exist as a regular file (no implicit create
        -- that's what ``upload`` is for)
      * ``expected_mtime`` from the prior /api/files/read must match
        the file's current mtime; mismatch -> 409 (someone else edited
        it).  This is the standard concurrent-edit detection pattern.
      * Encoding: UTF-8.  Binary edits not supported (and probably
        never will be from a text-editor UI).
    """
    return jsonify({
        "ok": False,
        "error": ("/api/files/write is not implemented yet.  The "
                  "Preview modal is currently view-only; edit + save "
                  "will land in a follow-up."),
    }), 501


@bp.route("/api/files/delete", methods=["DELETE"])
def api_files_delete():
    """Stub: remove a file or empty directory.

    When implemented, accepts JSON ``{path, recursive}``.
    Validation (same shape as create rules):
      * path must resolve inside an allowed picker root
      * path must NOT be a configured root (no deleting projects/)
      * path must NOT be a CANONICAL_TOPIC subdir directly under a
        project (would orphan the project layout; user goes to the
        shell for that)
      * ``recursive=true`` required for non-empty directories
      * The confirmation modal in the UI is mandatory; the backend
        does NOT add a second confirmation -- one is enough.
    """
    return jsonify({
        "ok": False,
        "error": ("/api/files/delete is not implemented yet.  For "
                  "destructive operations, use your shell (rm) and "
                  "refresh the sidebar."),
    }), 501


__all__ = ["bp"]
