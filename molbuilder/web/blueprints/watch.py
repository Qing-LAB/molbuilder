"""Watch blueprint -- live trajectory viewer for SIESTA / PySCF / future.

Routes (registered with no url_prefix; each carries its own full path):

    GET  /watch                browser UI page
    GET  /api/watch/formats    parser registry summary
    POST /api/watch/load       JSON {"path": "..."} or multipart upload
    GET  /api/watch/data       poll for changes (mtime-based)

The user opens /watch, paste an absolute path to the output file, and
clicks *Load*.  The page polls /api/watch/data roughly every 15
seconds; when the file's mtime advances the parser re-runs and the
viewer + plots refresh.

Format support is plugin-style: see ``molbuilder/parsers/`` for the
registered parsers and the auto-detection registry.

State model: a single global "current file" dict guarded by a Lock.
This is intentional -- the watch app is single-user / single-tab by
design (see docs/design.md "Watch -- live trajectory viewer").
"""

from __future__ import annotations

import os
import tempfile
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request

from molbuilder.parsers import (
    UnknownFormatError,
    detect_parser,
    parser_summary,
    trajectory_to_legacy_dict,
)


bp = Blueprint("watch", __name__)

# Single global "current file" state.  A single user / single tab is
# the expected usage so a plain dict + lock is enough; no need for
# sessions.
_lock = Lock()
_state: Dict[str, Any] = {
    "path":     None,
    "mtime":    None,
    "data":     None,
    "parser":   None,    # the TrajectoryParser class chosen for this file
    "uploaded": False,   # True when the active file was uploaded via
                         # the file-picker (one-shot, no live watching)
}

# Track the last temp file we created from a file-picker upload so
# we can clean it up when a new upload comes in.
_last_temp_upload: Optional[str] = None


def _refresh_if_changed() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Re-parse the current file iff its mtime has advanced.

    Returns ``(state, None)`` on success or ``(None, error_message)`` on
    failure.  Cheap when the file is unchanged.

    Locking strategy: snapshot path/mtime/parser under the lock, then
    drop the lock during the actual parse so other concurrent requests
    aren't blocked for the duration of a multi-MB log re-parse.  After
    parsing we re-acquire and only commit the result if the active file
    hasn't changed under us (defensive against a /api/load racing with
    a /api/data poll).
    """
    # ---- Snapshot under the lock --------------------------------
    with _lock:
        path = _state["path"]
        if not path:
            return None, "No file loaded yet."
        cached_mtime = _state["mtime"]
        parser_cls   = _state["parser"]

    if not os.path.isfile(path):
        return None, f"File not found: {path}"
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        return None, str(exc)

    # ---- Cheap path: nothing changed ----------------------------
    if mtime == cached_mtime:
        with _lock:
            return dict(_state), None

    # ---- Parse OUTSIDE the lock ---------------------------------
    # Parsers return a Trajectory; the JS client consumes the legacy
    # molwatch v1 dict shape, so we adapt at the boundary.
    try:
        traj = parser_cls.parse(path)
        new_data = trajectory_to_legacy_dict(traj)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Parse error: {exc}"

    # ---- Re-acquire to commit (skip if a concurrent /api/load
    #      already swapped to a different file under us) ---------
    with _lock:
        if _state["path"] == path and _state["parser"] is parser_cls:
            _state["data"]  = new_data
            _state["mtime"] = mtime
        return dict(_state), None


@bp.route("/watch")
def watch_page():
    return render_template("watch.html")


@bp.route("/api/watch/formats")
def api_formats():
    """Lightweight: lists registered parsers + their human labels."""
    return jsonify({"ok": True, "formats": parser_summary()})


@bp.route("/api/watch/load", methods=["POST"])
def api_load():
    """Two body shapes:

      * multipart/form-data with a single file field "file" -- file
        is saved to a temp file and parsed (one-shot, no live update);
      * application/json with {"path": "..."} -- server reads the
        absolute path off disk and polls it for live updates.

    The multipart branch is the file-picker fallback for users who
    don't want to type an absolute path.
    """
    # ---- multipart upload (file-picker mode) -----------------------
    if "file" in request.files:
        return _api_load_multipart(request.files["file"])

    # ---- JSON path (live-watch mode) -------------------------------
    body = request.get_json(silent=True) or {}
    path = (body.get("path") or "").strip()
    if not path:
        return jsonify({"ok": False, "error": "Empty path."}), 400
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": f"File not found: {path}"}), 404
    # Auto-detect parser before committing to the new path so an
    # unsupported file doesn't blank out a working one.
    try:
        parser_cls = detect_parser(path)
    except UnknownFormatError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    with _lock:
        _state["path"]     = path
        _state["mtime"]    = None      # force a re-parse next time
        _state["data"]     = None
        _state["parser"]   = parser_cls
        _state["uploaded"] = False

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({
        "ok":       True,
        "path":     state["path"],
        "mtime":    state["mtime"],
        "format":   parser_cls.name,
        "label":    parser_cls.label,
        "data":     state["data"],
        "uploaded": False,
    })


def _api_load_multipart(uploaded_file):
    """Save the uploaded file to a tempdir, parse, and stash the temp
    path on _state.  Future /api/data polls work like always but the
    mtime never advances (we don't write to the temp file again), so
    the data effectively snapshots at upload time.

    Old temp uploads are cleaned up when a new one comes in -- a
    process restart drops the rest.
    """
    global _last_temp_upload

    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    # Keep the original suffix (.xyz / .out / .log) so the parser-
    # detection layer's content sniff isn't fooled by extension-less
    # names.  Sanitise the basename to dodge path-traversal in the
    # temp filename itself.
    safe_name = os.path.basename(uploaded_file.filename) or "upload"
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"molwatch_{int(time.time())}_{safe_name}"
    )
    try:
        uploaded_file.save(tmp_path)
    except OSError as exc:
        return jsonify({"ok": False,
                        "error": f"Failed to write upload: {exc}"}), 500

    try:
        parser_cls = detect_parser(tmp_path)
    except UnknownFormatError as exc:
        # Don't keep an unrecognised upload around.
        try: os.remove(tmp_path)
        except OSError: pass
        return jsonify({"ok": False, "error": str(exc)}), 400

    with _lock:
        # Clean up any previous upload's temp file.
        if _last_temp_upload and _last_temp_upload != tmp_path:
            try: os.remove(_last_temp_upload)
            except OSError: pass
        _last_temp_upload = tmp_path
        _state["path"]     = tmp_path
        _state["mtime"]    = None
        _state["data"]     = None
        _state["parser"]   = parser_cls
        _state["uploaded"] = True

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({
        "ok":               True,
        "path":             tmp_path,
        "mtime":            state["mtime"],
        "format":           parser_cls.name,
        "label":            parser_cls.label,
        "data":             state["data"],
        "uploaded":         True,
        "uploaded_filename": uploaded_file.filename,
    })


@bp.route("/api/watch/data")
def api_data():
    """Return the parsed payload, or just an mtime if nothing changed."""
    client_mtime = request.args.get("mtime", type=float)
    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err})
    if client_mtime is not None and client_mtime == state["mtime"]:
        return jsonify({"ok": True, "changed": False, "mtime": state["mtime"]})
    parser_cls = state["parser"]
    return jsonify({
        "ok":       True,
        "changed":  True,
        "path":     state["path"],
        "mtime":    state["mtime"],
        "format":   parser_cls.name,
        "label":    parser_cls.label,
        "data":     state["data"],
        "uploaded": state.get("uploaded", False),
    })


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def warn_if_remote(host: str) -> None:
    """Emit a stderr warning when the watch app is bound to a non-loopback
    interface.  /api/watch/load reads any file the server can access, so
    exposing it on a network interface is effectively a remote
    arbitrary-file-read endpoint.  Called from molbuilder.cli when the
    user starts the watch server with a non-loopback --host."""
    if host in _LOCAL_HOSTS:
        return
    import sys as _sys
    print(f"WARNING: --host={host} exposes /api/watch/load to the network.",
          file=_sys.stderr)
    print("         The endpoint reads ANY local file the server can",
          file=_sys.stderr)
    print("         access.  Only do this on a trusted single-user",
          file=_sys.stderr)
    print("         machine, or add a reverse-proxy with auth in front.",
          file=_sys.stderr)
