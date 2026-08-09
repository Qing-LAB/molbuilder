"""Checkpoint HTTP routes -- powers the Projects Sidebar
"Run history" panel (sensor badge + list view + graph viewer).

Contract: docs/execution/checkpointing.md (the rules and their status).
Guide:    docs/execution/running-a-job.md § 6.
Panel:    docs/web/projects.md.

Envelope per web-api.md § 1 (four-bucket rule):
  * success                            -> HTTP 2xx + ok:true + data
  * scientific advisory -- state-shaped
    refusal (dirty tree, nested-repo
    refusal, legacy MANIFEST refusal)  -> HTTP 200 + ok:false +
                                          error + issues[] +
                                          errors_only[]
                                          (web-api.md § 1 canonical shape)
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
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from molbuilder.checkpoint import (
    Repo,
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
    web app is a single-user local server (see web-api.md § 1); we
    accept any path the host filesystem grants ``read`` to the user
    running molbuilder.
    """
    if not raw:
        raise ValueError("missing required parameter: path")
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"path is not a directory: {p}")
    return p


def _advisory(message: str, where: str = "") -> tuple:
    """Bucket-B response: scientific advisory, HTTP 200 + ok:false.

    Canonical shape per web-api.md § 1: a single error-severity
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


def _state_json(state) -> Dict[str, Any]:
    """One state, in the contract's vocabulary and nothing else (§ 5)."""
    return {
        "id":          state.id,
        "short":       state.short,
        "note":        state.note,
        "parent":      state.parent,
        "at":          state.at,
        "archive":     state.archive,
        "calculation": state.calculation,
        "tags":        list(state.tags),
    }


@bp.get("/api/checkpoint/state")
def api_checkpoint_state():
    """Where the folder stands, and what is unsaved (§ 5, A5).

    Called by the sidebar on directory-enter and on manual Refresh -- there is
    no background polling (docs/web/projects.md).  Deliberately does NOT walk
    the archive: that is an O(archived files) sweep for a number this does not
    show.
    """
    try:
        path = _resolve_path(request.args.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    # ``deep=1`` is what the panel's Refresh control asks for: compare content
    # rather than size and timestamp.  The default is the cheap read, because
    # this runs on every directory-enter and hashing gigabytes to draw a badge
    # is a cost the badge does not earn.  Nothing can be lost by being briefly
    # wrong here; the operations that CAN lose something check content
    # themselves (A5).
    deep = request.args.get("deep") in ("1", "true", "yes")
    repo = Repo(str(path))
    try:
        status = repo.status(deep=deep)
        here = repo.standing_at() if status.initialized else None
        return jsonify({
            "ok":            True,
            "path":          str(path),
            "initialized":   status.initialized,
            "standing_at":   _state_json(here) if here else None,
            "clean":         status.clean,
            "changed":       list(status.changed),
            "added":         list(status.added),
            "deleted":       list(status.deleted),
            "unsaved":       list(status.unsaved()),
            "ignore_edited": status.ignore_edited,
            "checked":       "content" if deep else "size+timestamp",
        })
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint state failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")


@bp.get("/api/checkpoint/list")
def api_checkpoint_list():
    """Every state, newest first, each naming the state it came from.

    Two states sharing a parent are alternatives from one point (§ 7.1); the
    panel can draw that without anything having been named.
    """
    try:
        path = _resolve_path(request.args.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    limit_raw = request.args.get("limit")
    limit = None
    if limit_raw is not None:
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            return _protocol_error(f"invalid limit: {limit_raw!r}")
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint folder; run init first", code=409)
    try:
        here = repo.standing_at()
        return jsonify({
            "ok":          True,
            "path":        str(path),
            "standing_at": here.id if here else None,
            "states":      [_state_json(s) for s in repo.states(limit=limit)],
            "tags":        [{"name": t.name, "state": t.state, "note": t.note}
                            for t in repo.tags()],
        })
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint list failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")


@bp.get("/api/checkpoint/config")
def api_checkpoint_config():
    """The classification this folder is saved under -- read only.

    There is no write route.  The classification lives in molbuilder.json and
    has ONE home (S1c); a per-folder editor is what let two folders behave
    differently for no recorded reason, and let somebody change the rules
    between a save and a restore.
    """
    try:
        path = _resolve_path(request.args.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    repo = Repo(str(path))
    try:
        cls = repo._classification()
        return jsonify({
            "ok":               True,
            "path":             str(path),
            "calculation":      repo.calculation() if repo.initialized else None,
            "size_limit_bytes": int(cls["size_limit_bytes"]),
            "always_large":     list(cls["always_large"]),
            "edit_in":          "molbuilder.json",
        })
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint config failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------- #
#  Write routes -- each one is an explicit act by the user (§ 9)        #
# --------------------------------------------------------------------- #


@bp.post("/api/checkpoint/init")
def api_checkpoint_init():
    """Make a folder a checkpoint folder and save its first state."""
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    repo = Repo(str(path))
    if repo.initialized:
        here = repo.standing_at()
        return jsonify({"ok": True, "path": str(path), "already": True,
                        "standing_at": _state_json(here) if here else None})
    try:
        state = repo.init(engine=body.get("engine"),
                          note=body.get("note") or "set up",
                          calculation=body.get("calculation"))
    except NestedRepoRefusedError as exc:
        return _advisory(str(exc), where="path")
    except CheckpointError as exc:
        return _advisory(str(exc), where="calculation")
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint init failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({"ok": True, "path": str(path), "already": False,
                    "state": _state_json(state)})


@bp.post("/api/checkpoint/save")
def api_checkpoint_save():
    """Save the folder as a new state.  The note is required (L3).

    A missing note is a protocol error, not a default: the note is the only
    thing that answers the question you bring to a history a month later, and
    a generated stand-in answers none of it.
    """
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    note = (body.get("note") or "").strip()
    if not note:
        return _protocol_error(
            "a state needs a note saying what happened and what you were "
            "about to do (checkpointing.md L3)")
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint folder; run init first", code=409)
    try:
        state = repo.save(note)
    except CheckpointError as exc:
        return _advisory(str(exc), where="save")
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint save failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({"ok": True, "path": str(path),
                    "changed": state is not None,
                    "state": _state_json(state) if state else None})


@bp.post("/api/checkpoint/tag")
def api_checkpoint_tag():
    """Give a state a name so it can be found again (§ 5, L4)."""
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    name = (body.get("name") or "").strip()
    note = (body.get("note") or "").strip()
    if not name:
        return _protocol_error("missing required parameter: name")
    if not note:
        return _protocol_error(
            "a tag needs a note saying why this state is worth returning to")
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint folder; run init first", code=409)
    try:
        tag = repo.tag(name, note, at=body.get("at"))
    except NoSuchRefError as exc:
        return _protocol_error(str(exc), code=404)
    except CheckpointError as exc:
        return _advisory(str(exc), where="tag")
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint tag failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({"ok": True, "tag": {"name": tag.name, "state": tag.state,
                                        "note": tag.note}})


@bp.post("/api/checkpoint/restore")
def api_checkpoint_restore():
    """Make the folder equal a state exactly (A4, A5).

    **There is no partial restore on this surface either.**  Text and big files
    are one state; returning half of one and half of another produces a folder
    no save ever held.  The old ``include_binaries`` flag did exactly that and
    skipped the archive verification on the way, so it is gone rather than
    defaulted.

    Unsaved work is refused as an advisory naming every file, and ``force``
    accepts the loss -- the decision is the user's, and the surface's job is to
    make it an informed one.
    """
    body = _get_body()
    try:
        path = _resolve_path(body.get("path"))
    except ValueError as exc:
        return _protocol_error(str(exc))
    state = (body.get("state") or body.get("ref") or "").strip()
    if not state:
        return _protocol_error("missing required parameter: state")
    if "include_binaries" in body:
        return _protocol_error(
            "include_binaries is not accepted: a restore returns the whole "
            "folder or it does not happen (checkpointing.md A4)")
    repo = Repo(str(path))
    if not repo.initialized:
        return _protocol_error(
            "not a checkpoint folder; run init first", code=409)
    try:
        target = repo.restore(state, force=bool(body.get("force")))
    except DirtyWorkingTreeError as exc:
        return _advisory(str(exc), where="unsaved")
    except NoSuchRefError as exc:
        return _protocol_error(str(exc), code=404)
    except CheckpointError as exc:
        return _server_fault(str(exc))
    except Exception as exc:                       # noqa: BLE001
        _log.exception("checkpoint restore failed")
        return _server_fault(f"{type(exc).__name__}: {exc}")
    return jsonify({"ok": True, "path": str(path),
                    "state": _state_json(target)})
