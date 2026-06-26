"""Run-checkpoints HTTP routes -- powers the Projects Sidebar
"Run history" panel (sensor badge + list view + graph viewer).

See docs/protocols/run-checkpoints.md § 8 for the full contract.

Envelope per web-api.md § 1.6 (four-bucket rule):
  * success                            -> HTTP 2xx + ok:true + data
  * scientific advisory -- state-shaped
    refusal (dirty tree, nested-repo
    refusal, legacy MANIFEST refusal)  -> HTTP 200 + ok:false +
                                          error + issues[] +
                                          errors_only[]
                                          (§ 1.1 canonical shape)
  * protocol error (bad path, missing
    param, unknown ref)                -> HTTP 4xx + ok:false +
                                          error: "..."
  * server fault (crash, IO error,
    archive integrity failure)         -> HTTP 5xx + ok:false +
                                          error: "..."

All operations go through the in-process ``Repo`` class (same code
path the CLI exercises).  Real filesystem; no mocks.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from molbuilder.checkpoint import (
    Repo, Checkpoint, RepoState,
    CheckpointError,
    DirtyWorkingTreeError,
    NestedRepoRefusedError,
    NoSuchRefError,
)


_log = logging.getLogger(__name__)

bp = Blueprint("checkpoint", __name__)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _resolve_path(raw: Optional[str]) -> Path:
    """Resolve a user-supplied path to an absolute directory.

    Raises ``ValueError`` (caught by the route handler and surfaced as
    HTTP 400) when ``raw`` is None, empty, or not an existing dir.

    No symlink or directory-traversal handling here -- the molbuilder
    web app is a single-user local server (see web-api.md § 1.3); we
    accept any path the host filesystem grants ``read`` to the user
    running molbuilder.
    """
    if not raw:
        raise ValueError("missing required parameter: path")
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"path is not a directory: {p}")
    return p


def _serialise_checkpoint(cp: Checkpoint) -> Dict[str, Any]:
    """Wire-format a Checkpoint for the JSON response."""
    return {
        "sha":           cp.sha,
        "short_sha":     cp.short_sha,
        "summary":       cp.summary,
        "author_at":     cp.author_at,
        "refs":          list(cp.refs),
        "has_archive":   cp.has_archive,
        "archive_bytes": cp.archive_bytes,
    }


def _serialise_state(st: RepoState) -> Dict[str, Any]:
    """Wire-format a RepoState for the JSON response."""
    return {
        "path":                st.path,
        "initialized":         st.initialized,
        "head":                st.head,
        "current_branch":      st.current_branch,
        "dirty":               st.dirty,
        "untracked":           st.untracked,
        "archive_total_bytes": st.archive_total_bytes,
    }


def _advisory(message: str, where: str = "") -> tuple:
    """Bucket-B response: scientific advisory, HTTP 200 + ok:false.

    Canonical shape per web-api.md § 1.1: a single error-severity
    Issue carried in both ``issues`` and the filtered ``errors_only``
    list, with a short ``error`` banner matching the issue message
    so a generic envelope reader has something to show.
    """
    issue = {"severity": "error", "message": message, "where": where}
    body = {
        "ok":          False,
        "error":       message,
        "issues":      [issue],
        "errors_only": [issue],
    }
    return jsonify(body), 200


def _protocol_error(message: str, code: int = 400) -> tuple:
    """Bucket-C response: protocol error (bad input shape)."""
    return jsonify({"ok": False, "error": message}), code


def _server_fault(message: str, code: int = 500) -> tuple:
    """Bucket-D response: server-side failure (crash, IO, integrity)."""
    return jsonify({"ok": False, "error": message}), code


def _get_body() -> Dict[str, Any]:
    """Extract JSON body or empty dict.  Tolerates missing
    Content-Type so curl-without-headers users get a useful error
    rather than 415."""
    return request.get_json(silent=True) or {}


# --------------------------------------------------------------------- #
#  Read routes                                                          #
# --------------------------------------------------------------------- #


@bp.get("/api/checkpoint/state")
def api_checkpoint_state():
    """Cheap snapshot of the working dir's checkpoint state.

    Called by the sidebar sensor on directory-enter and manual Refresh
    (run-checkpoints.md § 6.2, § 11 decision 7 -- no background
    polling) so the badge can show clean / dirty / N changes.  Cheap
    by contract: a single ``git status`` + ``rev-parse``, with NO walk
    of ``.binsnapshots/`` (archive size is the list/detail surface's
    job, § 6.3 / § 6.6).
    """
    try:
        path = _resolve_path(request.args.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    try:
        repo = Repo(str(path))
        return jsonify({"ok": True,
                        "state": _serialise_state(repo.state())})
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint state failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")


@bp.get("/api/checkpoint/list")
def api_checkpoint_list():
    """List checkpoints, most recent first.  Drives the run-history
    list view + graph viewer."""
    try:
        path = _resolve_path(request.args.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    limit_raw = request.args.get("limit", "50")
    try:
        limit = int(limit_raw)
        if limit <= 0 or limit > 1000:
            raise ValueError("limit out of range (1..1000)")
    except ValueError:
        return _protocol_error(f"invalid limit: {limit_raw!r}")
    try:
        repo = Repo(str(path))
        if not repo.initialized:
            return jsonify({
                "ok":          True,
                "initialized": False,
                "checkpoints": [],
            })
        cps = repo.list_checkpoints(limit=limit)
        return jsonify({
            "ok":          True,
            "initialized": True,
            "checkpoints": [_serialise_checkpoint(c) for c in cps],
        })
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint list failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")


@bp.get("/api/checkpoint/diff")
def api_checkpoint_diff():
    """Unified diff between two refs (default: text-only pathspec).

    Powers the sidebar's "Diff vs HEAD" / "Diff vs working tree"
    context-menu items.
    """
    try:
        path = _resolve_path(request.args.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    ref_a = request.args.get("a")
    ref_b = request.args.get("b")
    if not ref_a or not ref_b:
        return _protocol_error("missing required parameter: a or b")
    pathspec_raw = request.args.get("pathspec", "")
    pathspec: List[str] = (
        [s.strip() for s in pathspec_raw.split(",") if s.strip()]
        or ["*.fdf", "*.out", "*.molwatch.log", "*.parse.log",
            "*.md", "*.molstruct.json", "*.transport.json",
            "*.runtime_info.json", "*.CG", "*.XV"]
    )
    try:
        repo = Repo(str(path))
        if not repo.initialized:
            return _protocol_error(
                "not a checkpoint repo; run init first", code=409)
        diff_text = repo.diff(ref_a, ref_b, pathspec=pathspec)
        return jsonify({"ok": True, "diff": diff_text})
    except NoSuchRefError as exc:
        return _protocol_error(str(exc), code=404)
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint diff failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------- #
#  Write routes                                                         #
# --------------------------------------------------------------------- #


@bp.post("/api/checkpoint/init")
def api_checkpoint_init():
    """Phase 1 -- initialise this dir as a checkpoint repo.

    Refuses on nested working dirs (§ P5) -- bucket B advisory.
    """
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    repo = Repo(str(path))
    if repo.initialized:
        return jsonify({
            "ok":    True,
            "state": _serialise_state(repo.state()),
            "note":  "already initialised; no-op",
        })
    try:
        repo.init()
    except NestedRepoRefusedError as exc:
        return _advisory(str(exc), where="path")
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint init failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({"ok": True, "state": _serialise_state(repo.state())})


@bp.post("/api/checkpoint/commit")
def api_checkpoint_commit():
    """Phase 2 -- explicit user checkpoint.

    Clean tree → ``{"checkpoint": null, "note": "nothing to checkpoint"}``
    (HTTP 200 + ok:true).
    """
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    message = (body.get("message") or "").strip() or None
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint repo; run init first", code=409)
    try:
        cp = repo.checkpoint(message=message)
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint commit failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    if cp is None:
        return jsonify({
            "ok":         True,
            "checkpoint": None,
            "note":       "nothing to checkpoint (working tree clean)",
        })
    return jsonify({
        "ok":         True,
        "checkpoint": _serialise_checkpoint(cp),
    })


@bp.post("/api/checkpoint/tag")
def api_checkpoint_tag():
    """Phase 3 -- annotated tag at HEAD (or at ``at``)."""
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    label   = (body.get("label") or "").strip()
    message = (body.get("message") or "").strip()
    at      = (body.get("at") or "HEAD").strip() or "HEAD"
    if not label:
        return _protocol_error("missing required parameter: label")
    if not message:
        return _protocol_error("missing required parameter: message")
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint repo; run init first", code=409)
    try:
        repo.tag(label, message=message, at=at)
    except NoSuchRefError as exc:
        return _protocol_error(str(exc), code=404)
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint tag failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({"ok": True, "label": label, "at": at})


@bp.post("/api/checkpoint/restore")
def api_checkpoint_restore():
    """Phase 5 -- rewind text + binaries to ``ref``.

    Refuses on dirty working tree (bucket B advisory).  Refuses on
    legacy 2-col MANIFEST with the migrate-manifest hint (bucket B).
    """
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    ref = (body.get("ref") or "").strip()
    if not ref:
        return _protocol_error("missing required parameter: ref")
    include_binaries = bool(body.get("include_binaries", True))
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint repo; run init first", code=409)
    try:
        restored = repo.restore(ref, include_binaries=include_binaries)
    except DirtyWorkingTreeError as exc:
        return _advisory(str(exc), where="working-tree")
    except NoSuchRefError as exc:
        return _protocol_error(str(exc), code=404)
    except CheckpointError as exc:
        # Legacy MANIFEST refusal carries the migrate-manifest hint
        # -- surface as advisory (the user can act on it).
        msg = str(exc)
        if "migrate-manifest" in msg:
            return _advisory(msg, where=".binsnapshots")
        return _server_fault(msg)
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint restore failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({
        "ok":       True,
        "ref":      ref,
        "restored": restored,
    })


@bp.post("/api/checkpoint/migrate-manifest")
def api_checkpoint_migrate_manifest():
    """§ 10.4 -- migrate a legacy 2-column MANIFEST to canonical
    3-column form for the archive at ``ref``.
    """
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    ref = (body.get("ref") or "").strip()
    if not ref:
        return _protocol_error("missing required parameter: ref")
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint repo; run init first", code=409)
    try:
        entries = repo.migrate_manifest(ref)
    except NoSuchRefError as exc:
        return _protocol_error(str(exc), code=404)
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint migrate-manifest failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({
        "ok":      True,
        "ref":     ref,
        "entries": [
            {"name": name, "sha256": sha256, "size_bytes": size}
            for name, (sha256, size) in sorted(entries.items())
        ],
    })


__all__ = ["bp"]
