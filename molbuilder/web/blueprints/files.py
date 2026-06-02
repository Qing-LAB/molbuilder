"""Files blueprint -- server-side file explorer for the Projects sidebar.

Routes:

    GET  /api/files/roots               list of allowed root paths
    GET  /api/files/list?path=...&ext=  directory listing
    GET  /api/files/result-list?path=…  trajectory-parseable output files in
                                        a project dir (drives /results dropdown)
    GET  /api/files/stat?path=...       file metadata
    GET  /api/files/read?path=...       text contents (size-capped)
    POST /api/files/mkdir               create a new subdirectory (validated name)
    POST /api/projects/create           bootstrap a new project with every
                                        canonical subdir (atomic, strict conflict)
    POST /api/files/upload              multipart upload into current dir
    POST /api/files/write               save edited text (overwrite + mtime-conflict gated)
    POST /api/files/rename              in-place rename (atomic-no-overwrite +
                                        canonical-topic protection)
    DELETE /api/files/delete            remove file or directory (recursive optional)

Full contract:  docs/protocols/web-api.md  §  ``/api/files/*``  +
                docs/protocols/projects-sidebar.md  §  12

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

Design note: this blueprint is the projects sidebar's full file-system
surface.  The pre-2026-05-30 version was navigation-only ("intentionally
read-only"); the 2026-05-30+ commits added mkdir / upload / write /
delete / rename + the project-bootstrap endpoint to complete capability
C4 in projects-sidebar.md.  Path-safety remains the invariant: every
mutation runs through ``_resolve_within_roots`` + depth + canonical-
topic checks before touching disk.
"""
from __future__ import annotations

import os
import re
import shutil
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


def _depth_inside_root(resolved: Path) -> Optional[int]:
    """Depth of ``resolved`` inside whichever allowed picker root
    contains it; ``None`` when no root contains it.

    Depth 0 = the root itself; depth 1 = a project directory under
    the root; depth 2 = a topic directory under a project; etc.

    Shared by every endpoint whose validation includes a depth
    rule (currently /api/files/write + /api/files/delete + the
    /api/files/upload target_dir check).  Single source so all
    three endpoints stay in lockstep on the security boundary.
    """
    for root_path, _label in _allowed_roots():
        try:
            rel = resolved.relative_to(root_path)
        except ValueError:
            continue
        return len(rel.parts)
    return None


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


# 2026-06-01: ``/api/files/result-list`` endpoint retired.  Its single
# consumer (the in-trajectory dropdown at ``lib/trajectory/result-list.js``)
# was lifted to the tab level at ``lib/results/file-picker.js``, which
# uses the existing ``/api/files/list`` + client-side filtering through
# the inspector registry's ``pickResult`` -- no second endpoint needed,
# and the single source of truth for "what counts as a result" now
# lives entirely in the JS inspectors.


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


# Default per-call window for the range-read endpoint -- 256 KB is
# large enough to feel snappy + render multiple "pages" of text
# without round-tripping for each scroll, small enough that the
# 16 MB hard ceiling is plenty of room for explicit jumps to a
# specific byte range.
_DEFAULT_RANGE_BYTES = 256 * 1024


@bp.route("/api/files/read_range", methods=["GET"])
def api_files_read_range():
    """Range-read endpoint for the source inspector's paginated view
    of arbitrarily-large text files (task #119, 2026-06-02).

    Reads a byte-range of ``path`` and returns it as UTF-8 text plus
    enough metadata for the client to know where it landed + how
    much more there is.  Replaces the source inspector's "read the
    whole file or 413" model with an "I'll fetch what you ask for"
    model.

    Query params:
        path        -- absolute or relative path inside a project root.
        offset      -- byte offset from the start of the file.  Default 0.
                       Negative values are interpreted as "offset bytes
                       before the END of the file", so ``offset=-4096``
                       returns the last 4 KB.  Useful for "show me the
                       tail" without knowing the file size up front.
        max_bytes   -- maximum bytes to return.  Default 256 KB, hard
                       ceiling at ``_MAX_READ_BYTES`` (16 MB).

    UTF-8 boundary handling: a byte range can land in the middle of
    a multi-byte codepoint.  We trim INCOMPLETE codepoints at the
    END of the chunk (so the response is always valid UTF-8) and
    report the actual bytes returned via ``length``.  Trimming at
    the start is NOT done -- if the caller passed an offset mid-
    codepoint they got the byte range they asked for; the trim is a
    polite cleanup for the typical "scrolling forward, next page"
    case where the END boundary is naturally arbitrary.

    Returns ``{ok: True, path, offset, length, file_size, mtime,
    text, eof}`` where:
        offset       -- the absolute byte offset where ``text`` starts
                        (negative-offset requests get RESOLVED here).
        length       -- bytes in ``text`` (after UTF-8 trim).  May be
                        less than ``max_bytes``.
        file_size    -- total size of the file.  Lets the client show
                        a "loaded X of Y" indicator.
        eof          -- True iff ``offset + length == file_size``;
                        client uses this to disable "Load more".
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
        max_bytes = int(request.args.get(
            "max_bytes", _DEFAULT_RANGE_BYTES))
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
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"ok": False,
                        "error": "offset must be an integer"}), 400

    try:
        st = resolved.stat()
    except OSError as exc:
        return jsonify({"ok": False, "error": f"stat failed: {exc}"}), 500

    file_size = st.st_size

    # Resolve negative offset = "from end".  ``offset = -N`` -> read
    # the last N bytes.  Clamp at 0 so an over-large negative offset
    # just means "read from the start".
    if offset < 0:
        offset = max(0, file_size + offset)

    if offset > file_size:
        return jsonify({
            "ok": False,
            "error": f"offset {offset} exceeds file size {file_size}",
        }), 400

    try:
        with open(resolved, "rb") as fh:
            fh.seek(offset)
            raw = fh.read(max_bytes)
    except PermissionError as exc:
        return jsonify({"ok": False,
                        "error": f"permission denied: {exc}"}), 403
    except OSError as exc:
        return jsonify({"ok": False,
                        "error": f"read failed: {exc}"}), 500

    # Trim incomplete UTF-8 codepoints at the END so the response is
    # always valid UTF-8.  Walk back at most 3 bytes (max codepoint
    # length is 4; a complete trailing codepoint ends ≤ 3 bytes
    # before the actual end).
    trimmed = raw
    for _ in range(4):
        try:
            text = trimmed.decode("utf-8")
            break
        except UnicodeDecodeError:
            # Last byte is the start of an incomplete codepoint; drop
            # it and retry.  If we'd already dropped 3 bytes and it's
            # STILL not decodable, the input wasn't valid UTF-8 to
            # begin with -- fall through to the explicit error below.
            if not trimmed:
                break
            trimmed = trimmed[:-1]
    else:
        return jsonify({
            "ok":   False,
            "error": f"file region is not valid UTF-8 at offset "
                     f"{offset}.  read_range only previews text files.",
        }), 400

    length = len(trimmed)
    eof    = (offset + length) >= file_size

    return jsonify({
        "ok":        True,
        "path":      str(resolved),
        "offset":    offset,
        "length":    length,
        "file_size": file_size,
        "mtime":     st.st_mtime,
        "text":      text,
        "eof":       eof,
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


# Upload filename regex.  Allows the dot that ``validate_name``
# forbids (filenames need extensions), plus underscore / hyphen /
# letters / digits.  Deliberately conservative: no spaces, no shell
# metacharacters, no leading dot (so dotfiles can't be uploaded
# accidentally and the list endpoint's hidden-filter still applies).
_UPLOAD_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_upload_filename(name: str) -> Optional[str]:
    """Return an error message if ``name`` is unsuitable as an upload
    target filename; None when OK.

    Separate from ``molbuilder.projects.validate_name`` because that
    one is for directory / project names and forbids the dot that
    filenames need for extensions.
    """
    if not name:
        return "filename is empty"
    if "/" in name or "\\" in name:
        return f"filename may not contain path separators: {name!r}"
    if name in (".", ".."):
        return f"filename {name!r} is not a real filename"
    if not _UPLOAD_FILENAME_RE.match(name):
        return (
            f"filename {name!r} contains unsupported characters; "
            f"allowed: letters, digits, '.', '_', '-' "
            f"(must start with a letter or digit)"
        )
    return None


@bp.route("/api/files/upload", methods=["POST"])
def api_files_upload():
    """Multipart upload into a sidebar-visible directory.

    Form fields:
      * ``target_dir`` -- absolute path of the destination directory.
        Must resolve inside an allowed picker root + be at depth >= 1
        (no uploads directly under ``projects/``) + already exist.
      * ``file``       -- the multipart file part.  ``file.filename``
        is validated against the upload-filename regex below.

    Behaviour:
      * No implicit overwrite: name conflict at destination is 409.
        UI deletes first if the user wants to replace.
      * Max upload size is enforced globally by Flask's
        ``MAX_CONTENT_LENGTH`` (50 MB; the app-level 413 handler
        catches oversize uploads with a clean message).
      * Returns the same shape as ``/api/files/write``
        (``{ok, path, size, mtime}``) so the sidebar refresh code
        path is identical for both write modes.
    """
    target_dir_raw = (request.form.get("target_dir") or "").strip()
    if not target_dir_raw:
        return jsonify({
            "ok": False,
            "error": "missing 'target_dir' form field",
        }), 400

    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({
            "ok": False,
            "error": "missing 'file' multipart part",
        }), 400

    # Validate the destination directory.
    try:
        target_dir = _resolve_within_roots(target_dir_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not target_dir.is_dir():
        return jsonify({
            "ok": False,
            "error": (f"target_dir does not exist or is not a directory: "
                      f"{str(target_dir)!r}.  Use /api/files/mkdir to "
                      f"create it first."),
        }), 400

    # Depth check: target_dir must be at depth >= 1 inside an
    # allowed picker root (matches the /api/files/write rule -- no
    # writes directly into projects/ itself).  Shared helper.
    depth = _depth_inside_root(target_dir)
    if depth is None or depth < 1:
        return jsonify({
            "ok": False,
            "error": (f"cannot upload directly into the picker root; "
                      f"create or pick a subdirectory first.  Got: "
                      f"{str(target_dir)!r}"),
        }), 400

    # Validate the filename.  We use the basename of upload.filename
    # to defang any client-supplied path prefix.
    filename = os.path.basename(upload.filename)
    err = _validate_upload_filename(filename)
    if err is not None:
        return jsonify({"ok": False, "error": err}), 400

    dest = target_dir / filename
    if dest.exists():
        return jsonify({
            "ok": False,
            "error": (f"file already exists: {str(dest)!r}.  "
                      f"Delete it first via the sidebar (or your shell) "
                      f"and re-upload."),
        }), 409

    try:
        upload.save(str(dest))
    except PermissionError as exc:
        return jsonify({
            "ok": False,
            "error": f"permission denied: {exc}",
        }), 403
    except OSError as exc:
        return jsonify({
            "ok": False,
            "error": f"upload write failed: {exc}",
        }), 500

    try:
        st = dest.stat()
    except OSError as exc:
        return jsonify({"ok": False,
                        "error": f"stat after upload failed: {exc}"}), 500

    return jsonify({
        "ok":    True,
        "path":  str(dest),
        "size":  st.st_size,
        "mtime": st.st_mtime,
    })


@bp.route("/api/files/write", methods=["POST"])
def api_files_write():
    """Write text content to a file inside an allowed picker root.

    JSON body: ``{path, text, overwrite?, expected_mtime?}``

    Two distinct call patterns share this endpoint:

      1. **Generate-and-save** (Spectra / Build Generate buttons):
         no ``expected_mtime``; the caller may set ``overwrite: true``
         to clobber an existing file.  Without ``overwrite``, an
         existing file at ``path`` returns 409.

      2. **Edit-and-save** (file-preview modal's Save -- still a
         stub on the UI side): ``expected_mtime`` is the mtime
         captured at /api/files/read time.  If the file's current
         mtime differs, 409 (someone else modified it).

    Validation:
      * ``path`` must resolve inside an allowed root
      * raw ``path`` must not contain ``..``
      * path depth inside the picker root must be >= 1 (no writing
        files directly into ``projects/``; the root stays clean and
        only holds project dirs)
      * parent directory must already exist (no implicit ``mkdir -p``;
        use ``/api/files/mkdir`` for that)
      * ``text`` must be a string (UTF-8 encoded by Flask's JSON
        parser; binary writes are a separate future feature)

    Returns ``{ok, path, size, mtime}`` on success.
    """
    body = request.get_json(silent=True) or {}
    raw_path = body.get("path", "")
    text     = body.get("text", "")
    overwrite      = bool(body.get("overwrite", False))
    expected_mtime = body.get("expected_mtime", None)

    if not isinstance(text, str):
        return jsonify({
            "ok": False,
            "error": "'text' must be a string",
        }), 400

    try:
        resolved = _resolve_within_roots(raw_path)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    # Depth check: the file's PARENT must be at depth >= 1 inside
    # the picker's root.  i.e., write to projects/<proj>/<...>/file
    # but not directly into projects/ itself.  Shared with /upload
    # + /delete via the _depth_inside_root helper.
    parent_depth = _depth_inside_root(resolved.parent)
    if parent_depth is None or parent_depth < 1:
        return jsonify({
            "ok":   False,
            "error": (f"cannot write directly into the picker root; "
                      f"create or pick a subdirectory first.  Got: "
                      f"{str(resolved)!r}"),
        }), 400

    parent = resolved.parent
    if not parent.is_dir():
        return jsonify({
            "ok":   False,
            "error": (f"parent directory does not exist: {str(parent)!r}.  "
                      f"Use /api/files/mkdir to create it first."),
        }), 400

    # Conflict checks (both apply when relevant).
    if resolved.exists():
        if expected_mtime is not None:
            try:
                actual_mtime = resolved.stat().st_mtime
            except OSError as exc:
                return jsonify({"ok": False,
                                "error": f"stat failed: {exc}"}), 500
            # Float comparison with a tolerance; mtime resolutions vary
            # across filesystems but ~ms is a safe threshold for
            # "no concurrent edit".
            if abs(actual_mtime - float(expected_mtime)) > 0.001:
                return jsonify({
                    "ok":   False,
                    "error": (f"file was modified since you opened it "
                              f"(expected mtime {expected_mtime}, "
                              f"found {actual_mtime}).  Re-open the "
                              f"file to see the latest content."),
                    "actual_mtime": actual_mtime,
                }), 409
            # mtime matches: this is a valid edit-save; overwrite OK.
        elif not overwrite:
            return jsonify({
                "ok":   False,
                "error": (f"file already exists: {str(resolved)!r}.  "
                          f"Set overwrite=true to replace it, or pick "
                          f"a different name / directory."),
            }), 409

    try:
        resolved.write_text(text, encoding="utf-8")
    except PermissionError as exc:
        return jsonify({
            "ok": False,
            "error": f"permission denied: {exc}",
        }), 403
    except OSError as exc:
        return jsonify({
            "ok": False,
            "error": f"write failed: {exc}",
        }), 500

    try:
        st = resolved.stat()
    except OSError as exc:
        return jsonify({"ok": False, "error": f"stat after write failed: {exc}"}), 500

    return jsonify({
        "ok":    True,
        "path":  str(resolved),
        "size":  st.st_size,
        "mtime": st.st_mtime,
    })


@bp.route("/api/files/rename", methods=["POST"])
def api_files_rename():
    """Rename a file or directory in place.

    JSON body: ``{path, new_name}``

    Validation contract (mirrors the ``/api/files/delete`` rules so
    the user can't get into a state where rename succeeds where
    delete would refuse):

      * ``path`` must resolve inside an allowed picker root.
      * ``path`` must exist.
      * ``path`` must NOT be the picker root itself (depth 0).
      * ``path`` must NOT be a canonical-topic subdir directly under
        a project (depth 2, name in CANONICAL_TOPICS).  Renaming
        ``projects/<proj>/spectrum/`` to anything else would orphan
        the project layout.
      * ``new_name`` must be a non-empty string with NO path separator,
        NO ``..`` segments, and NOT equal to ``.`` or ``..``.
      * ``new_name`` must not already exist in the same parent
        directory (atomic-no-overwrite policy; explicit overwrite
        would need a separate ``overwrite=true`` flag, not currently
        supported).

    Atomicity: uses :func:`os.rename` so the operation is atomic on
    the same filesystem.  Cross-device renames (e.g. if the path is
    a bind mount from another disk) are rejected with a clear error
    rather than silently falling back to copy+delete (which loses
    inode identity).

    Returns ``{ok, path}`` on success where ``path`` is the NEW
    absolute path.
    """
    body = request.get_json(silent=True) or {}
    raw_path  = body.get("path", "")
    new_name  = body.get("new_name", "")

    if not isinstance(raw_path, str) or not raw_path.strip():
        return jsonify({
            "ok": False,
            "error": "missing 'path' in request body",
        }), 400

    if not isinstance(new_name, str) or not new_name.strip():
        return jsonify({
            "ok": False,
            "error": "missing 'new_name' in request body",
        }), 400

    # Reject path-traversal / separator attempts on new_name.  A
    # legitimate rename only changes the basename; anything that
    # tries to move the entry into another dir is rejected.
    if (
        "/" in new_name
        or "\\" in new_name
        or new_name in (".", "..")
        or ".." in new_name.split("/")    # belt+suspenders for parser
    ):
        return jsonify({
            "ok": False,
            "error": (
                "invalid 'new_name': must be a basename only "
                "(no '/', no '..', no '.')"
            ),
        }), 400

    try:
        resolved = _resolve_within_roots(raw_path)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not resolved.exists():
        return jsonify({
            "ok":   False,
            "error": f"path does not exist: {str(resolved)!r}",
        }), 404

    # Depth check + canonical-topic protection.  Same shape as
    # delete's check; centralising would help but the two endpoints
    # are infrequent enough that the duplication is acceptable.
    rel_parts: Optional[Tuple[str, ...]] = None
    for root_path, _label in _allowed_roots():
        try:
            rel = resolved.relative_to(root_path)
        except ValueError:
            continue
        rel_parts = rel.parts
        break

    if rel_parts is None:
        return jsonify({
            "ok":   False,
            "error": f"path {str(resolved)!r} not inside any picker root",
        }), 400

    depth = len(rel_parts)

    if depth == 0:
        return jsonify({
            "ok":   False,
            "error": ("refusing to rename the picker root itself"),
        }), 400

    if depth == 2 and resolved.is_dir() and rel_parts[1] in CANONICAL_TOPICS:
        return jsonify({
            "ok":   False,
            "error": (f"refusing to rename canonical-topic directory "
                      f"{rel_parts[1]!r} via the UI (would orphan the "
                      f"project layout).  Use your shell if you really "
                      f"need to."),
        }), 400

    # Compute the destination.  resolved.parent is guaranteed to be
    # inside the picker root because resolved is.
    dst = resolved.parent / new_name

    # No-op rename: same name -> short-circuit to success BEFORE the
    # dst-exists check (otherwise the same-name path would hit the
    # 409 branch because dst == resolved and resolved.exists() is True).
    # Matches the typical UX where a user opens the rename dialog,
    # doesn't change anything, and hits Save -- we shouldn't surface
    # that as an error.
    if resolved.name == new_name:
        return jsonify({
            "ok":   True,
            "path": str(resolved),
        })

    # Reject overwrite.  The atomic-no-overwrite policy: a rename to
    # an existing entry would silently replace it (and on POSIX
    # destroy the destination's inode).  The user almost certainly
    # didn't mean that.  An explicit overwrite=true flag could be
    # added later if a real use case shows up.
    if dst.exists():
        return jsonify({
            "ok":    False,
            "error": (f"destination already exists: {new_name!r}.  Pick a "
                      f"different name or delete the existing entry first."),
        }), 409

    try:
        os.replace(str(resolved), str(dst))
    except OSError as exc:
        # Cross-device rename, permission denied, FS quota, ... .
        # The errno + message tells the user what to do.
        return jsonify({
            "ok":    False,
            "error": f"rename failed: {exc.strerror or str(exc)}",
        }), 500

    return jsonify({
        "ok":   True,
        "path": str(dst),
    })


@bp.route("/api/files/delete", methods=["DELETE"])
def api_files_delete():
    """Remove a file or directory inside an allowed picker root.

    JSON body: ``{path, recursive?}``

    Validation contract (matches the stub docstring + the JS
    ``_isDeletableEntry`` gate so the user never sees a control
    they can't use):

      * ``path`` must resolve inside an allowed picker root.
      * ``path`` must NOT be the picker root itself (depth 0:
        refuses to delete ``projects/``).
      * ``path`` must NOT be a canonical-topic subdir directly
        under a project (depth 2 inside the root, name in
        :data:`molbuilder.projects.CANONICAL_TOPICS`).  Deleting
        ``projects/<proj>/spectrum/`` would orphan the project
        layout; users do that explicitly in their shell.
      * ``recursive=true`` is required for non-empty directories.
        Empty directories + plain files can be deleted without it.

    The frontend (sidebar's per-entry × button) is responsible for
    the confirm() dialog; the backend does NOT add a second
    confirmation -- one is enough and double confirms feel patronising.

    Returns ``{ok, path}`` on success.
    """
    body = request.get_json(silent=True) or {}
    raw_path  = body.get("path", "")
    recursive = bool(body.get("recursive", False))

    if not isinstance(raw_path, str) or not raw_path.strip():
        return jsonify({
            "ok": False,
            "error": "missing 'path' in request body",
        }), 400

    try:
        resolved = _resolve_within_roots(raw_path)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not resolved.exists():
        return jsonify({
            "ok":   False,
            "error": f"path does not exist: {str(resolved)!r}",
        }), 404

    # Depth check + canonical-topic protection.  We need both the
    # depth (for the root-protection + recursive-flag rules) AND
    # the path parts (for the canonical-topic name lookup at depth
    # 2).  Walk the roots once for the parts; _depth_inside_root
    # is the same loop but returns just the length.
    rel_parts: Optional[Tuple[str, ...]] = None
    for root_path, _label in _allowed_roots():
        try:
            rel = resolved.relative_to(root_path)
        except ValueError:
            continue
        rel_parts = rel.parts
        break

    if rel_parts is None:
        # _resolve_within_roots already verified containment, so
        # this branch is defensive (could only fire if roots
        # changed mid-request, which the Capabilities snapshot
        # prevents).
        return jsonify({
            "ok":   False,
            "error": f"path {str(resolved)!r} not inside any picker root",
        }), 400

    depth = len(rel_parts)

    if depth == 0:
        return jsonify({
            "ok":   False,
            "error": ("refusing to delete the picker root itself; "
                      "this would wipe every project at once"),
        }), 400

    # Depth 2 = directly under a project.  If the leaf name is a
    # canonical topic AND it's a directory, refuse: deleting the
    # spectrum/ or struct/ subdir orphans the project layout.  Files
    # at depth 2 (e.g. projects/<proj>/README.md) are fine to delete.
    if depth == 2 and resolved.is_dir() and rel_parts[1] in CANONICAL_TOPICS:
        return jsonify({
            "ok":   False,
            "error": (f"refusing to delete canonical-topic directory "
                      f"{rel_parts[1]!r} via the UI (would orphan the "
                      f"project layout).  Use your shell + recreate "
                      f"the project via + New project if needed."),
        }), 400

    # Non-empty dir requires explicit recursive flag.  Empty dirs
    # and plain files can be deleted without it.
    if resolved.is_dir():
        # any() short-circuits as soon as it finds one entry --
        # cheaper than iterating the whole directory.
        try:
            non_empty = any(resolved.iterdir())
        except OSError as exc:
            return jsonify({
                "ok": False,
                "error": f"could not enumerate directory: {exc}",
            }), 500
        if non_empty and not recursive:
            return jsonify({
                "ok":   False,
                "error": (f"directory is not empty: {str(resolved)!r}.  "
                          f"Pass recursive=true to delete it and "
                          f"everything inside."),
            }), 409

    try:
        if resolved.is_dir():
            if recursive:
                shutil.rmtree(resolved)
            else:
                resolved.rmdir()
        else:
            resolved.unlink()
    except PermissionError as exc:
        return jsonify({
            "ok": False,
            "error": f"permission denied: {exc}",
        }), 403
    except OSError as exc:
        return jsonify({
            "ok": False,
            "error": f"delete failed: {exc}",
        }), 500

    return jsonify({
        "ok":   True,
        "path": str(resolved),
    })


__all__ = ["bp"]
