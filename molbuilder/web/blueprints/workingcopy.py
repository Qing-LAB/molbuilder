"""``/api/workingcopy/*`` — the web surface over the working-copy core
(``working-copy-persistence.md`` § 6).

**Scratch-backed + stateless:** each request reconstructs a
:class:`~molbuilder.workingcopy.WorkingCopy` from the browser's payload + the
server scratch and delegates to the tested core — there is no server-side
in-memory registry to leak or expire.  Structure + sidecar is the only codec
for now.

Session (§ 13.5) is the *server-side* session: the login when authenticated,
else a stable per-server-run local id for no-auth localhost.
"""
from __future__ import annotations

import uuid
from pathlib import Path

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


def _project_dir(source: Path) -> Path:
    return source.parent


def _resolve(raw):
    """Resolve a path within the allowed roots (raises _PickerError)."""
    return _resolve_within_roots(raw)


# --------------------------------------------------------------------- #
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
                            project_dir=_project_dir(src))
    return jsonify({"ok": True, "session": w.session, "source": str(w.source),
                    "source_hash": w.source_hash,
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
    data = _CODEC.from_scratch(blob)
    w = wc.WorkingCopy(codec=_CODEC, session=_session(),
                       project_dir=_project_dir(src), data=data, source=src,
                       source_hash=b.get("source_hash"))
    w.update(data)
    return jsonify({"ok": True})


@bp.route("/api/workingcopy/commit", methods=["POST"])
def wc_commit():
    b = _body()
    source, blob = b.get("source"), b.get("data")
    if not isinstance(source, str) or blob is None:
        return _bad("missing 'source' or 'data'")
    try:
        src = _resolve(source)
        tgt = _resolve(b["target"]) if b.get("target") else src
    except _PickerError as e:
        return _bad(e.message, e.status)
    data = _CODEC.from_scratch(blob)
    w = wc.WorkingCopy(codec=_CODEC, session=_session(),
                       project_dir=_project_dir(src), data=data, source=src,
                       source_hash=b.get("source_hash"))
    r = w.commit(tgt, on_mismatch=b.get("on_mismatch", "refuse"))
    if r.ok:
        return jsonify({"ok": True, "source": str(r.target),
                        "source_hash": w.source_hash})
    return jsonify({"ok": False, "reason": r.reason,
                    "expected_hash": r.expected_hash,
                    "actual_hash": r.actual_hash})


@bp.route("/api/workingcopy/discard", methods=["POST"])
def wc_discard():
    source = _body().get("source")
    if not isinstance(source, str):
        return _bad("missing 'source'")
    try:
        src = _resolve(source)
    except _PickerError as e:
        return _bad(e.message, e.status)
    wc.WorkingCopy(codec=_CODEC, session=_session(),
                   project_dir=_project_dir(src), data=None, source=src,
                   source_hash=None).discard()
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
        {"scratch": str(r.path), "source": r.source,
         "source_hash": r.source_hash, "session": r.session, "ts": r.ts}
        for r in recs]})


@bp.route("/api/workingcopy/recover", methods=["POST"])
def wc_recover():
    scratch = _body().get("scratch")
    if not isinstance(scratch, str):
        return _bad("missing 'scratch'")
    try:
        sp = _resolve(scratch)
    except _PickerError as e:
        return _bad(e.message, e.status)
    env = persist.read_json(sp)
    rec = wc.ScratchRecord(path=sp, source=env.get("source"),
                           source_hash=env.get("source_hash"),
                           session=env.get("session"), ts=env.get("ts"),
                           blob=env.get("blob"))
    w = wc.WorkingCopy.recover(rec, _CODEC, project_dir=sp.parent.parent)
    return jsonify({"ok": True,
                    "source": str(w.source) if w.source else None,
                    "source_hash": w.source_hash,
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
