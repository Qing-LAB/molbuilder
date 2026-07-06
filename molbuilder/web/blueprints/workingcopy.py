"""``/api/workingcopy/*`` — load / edit (draft) / save for the structure editor.

Dead simple: `open` loads the artifact, `update` keeps a draft (so a reload/crash
doesn't lose edits), `save` writes the files (overwrite the same path, or save-as
a new one).  No gate, no hashing.  Structure + sidecar codec only.

Session (§13.5) is the server-side session: the login when authenticated, else a
stable per-server-run local id for no-auth localhost.
"""
from __future__ import annotations

import uuid

from flask import Blueprint, g, jsonify, request

from molbuilder import persist
from molbuilder import workingcopy as wc
from molbuilder.workingcopy_structure import StructureCodec

from .files import _PickerError, _resolve_within_roots

bp = Blueprint("workingcopy", __name__)
_CODEC = StructureCodec()
_SERVER_SESSION = None


def _session() -> str:
    user = getattr(g, "user", None)
    if isinstance(user, dict):
        ident = user.get("id") or user.get("username")
        if ident:
            return f"user-{ident}"
    global _SERVER_SESSION
    if _SERVER_SESSION is None:
        _SERVER_SESSION = "local-" + uuid.uuid4().hex[:12]
    return _SERVER_SESSION


def _bad(msg, status=400):
    return jsonify({"ok": False, "error": msg}), status


def _body():
    return request.get_json(silent=True) or {}


def _resolve(raw):
    return _resolve_within_roots(raw)


@bp.route("/api/workingcopy/open", methods=["POST"])
def wc_open():
    path = _body().get("path")
    if not isinstance(path, str) or not path:
        return _bad("missing 'path'")
    try:
        src = _resolve(path)
    except _PickerError as e:
        return _bad(e.message, e.status)
    w = wc.WorkingCopy.open(src, _CODEC, session=_session(),
                            project_dir=src.parent)
    return jsonify({"ok": True, "session": w.session, "source": str(w.source),
                    "data": _CODEC.scratch_blob(w.data)})


@bp.route("/api/workingcopy/update", methods=["POST"])
def wc_update():
    b = _body()
    source, blob = b.get("source"), b.get("data")
    if not isinstance(source, str) or blob is None:
        return _bad("missing 'source' or 'data'")
    try:
        src = _resolve(source)
    except _PickerError as e:
        return _bad(e.message, e.status)
    try:
        data = _CODEC.from_scratch(blob)
    except Exception as e:  # noqa: BLE001 -- malformed body -> 400
        return _bad(f"malformed working-copy body: {e}", 400)
    wc.WorkingCopy(codec=_CODEC, session=_session(), project_dir=src.parent,
                   data=data, source=src).update(data)
    return jsonify({"ok": True})


@bp.route("/api/workingcopy/save", methods=["POST"])
def wc_save():
    """Write the artifact (the `.xyz` + `.molstruct.json` pair) to `target` --
    overwrite the same path, or save-as a new one.  `data` is the browser's working
    copy.

    Overwrite gate (save-flow.md §1 / §4.0.1): an existing `target` is refused with
    409 unless ``overwrite: true`` -- the caller confirms first, exactly like
    ``/api/files/write``.  A symlink at the target is always refused.
    """
    b = _body()
    source, blob = b.get("source"), b.get("data")
    if not isinstance(source, str) or blob is None:
        return _bad("missing 'source' or 'data'")
    try:
        src = _resolve(source)
        target = _resolve(b["target"]) if b.get("target") else src
    except _PickerError as e:
        return _bad(e.message, e.status)
    overwrite = bool(b.get("overwrite", False))
    if target.exists() and not overwrite:
        return _bad(f"file already exists: {target}", 409)
    try:
        if target.is_symlink():
            return _bad(f"refusing to write through a symlink at {target}", 400)
    except OSError as e:
        return _bad(f"symlink check failed: {e}", 500)
    try:
        data = _CODEC.from_scratch(blob)
    except Exception as e:  # noqa: BLE001
        return _bad(f"malformed working-copy body: {e}", 400)
    saved = wc.WorkingCopy(codec=_CODEC, session=_session(),
                           project_dir=src.parent, data=data,
                           source=src).save(target)
    return jsonify({"ok": True, "saved": str(saved)})


@bp.route("/api/workingcopy/discard", methods=["POST"])
def wc_discard():
    source = _body().get("source")
    if not isinstance(source, str):
        return _bad("missing 'source'")
    try:
        src = _resolve(source)
    except _PickerError as e:
        return _bad(e.message, e.status)
    wc.WorkingCopy(codec=_CODEC, session=_session(), project_dir=src.parent,
                   data=None, source=src).discard()
    return jsonify({"ok": True})


@bp.route("/api/workingcopy/orphans", methods=["POST"])
def wc_orphans():
    path = _body().get("path")
    if not isinstance(path, str):
        return _bad("missing 'path'")
    try:
        p = _resolve(path)
    except _PickerError as e:
        return _bad(e.message, e.status)
    project = p if p.is_dir() else p.parent
    recs = wc.list_orphans(project, live_sessions=[_session()])
    return jsonify({"ok": True, "orphans": [
        {"scratch": str(r.path), "source": r.source, "session": r.session,
         "ts": r.ts} for r in recs]})


@bp.route("/api/workingcopy/recover", methods=["POST"])
def wc_recover():
    scratch = _body().get("scratch")
    if not isinstance(scratch, str):
        return _bad("missing 'scratch'")
    try:
        sp = _resolve(scratch)
    except _PickerError as e:
        return _bad(e.message, e.status)
    try:
        env = persist.read_json(sp)
        rec = wc.ScratchRecord(path=sp, source=env.get("source"),
                               session=env.get("session"), ts=env.get("ts"),
                               blob=env.get("blob"))
        w = wc.WorkingCopy.recover(rec, _CODEC, project_dir=sp.parent.parent)
    except Exception as e:  # noqa: BLE001
        return _bad(f"could not recover draft: {e}", 400)
    return jsonify({"ok": True,
                    "source": str(w.source) if w.source else None,
                    "data": _CODEC.scratch_blob(w.data)})


@bp.route("/api/workingcopy/clean", methods=["POST"])
def wc_clean():
    path = _body().get("path")
    if not isinstance(path, str):
        return _bad("missing 'path'")
    try:
        p = _resolve(path)
    except _PickerError as e:
        return _bad(e.message, e.status)
    project = p if p.is_dir() else p.parent
    return jsonify({"ok": True, "removed": wc.clean_all(project)})
