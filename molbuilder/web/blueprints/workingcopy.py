"""``/api/workingcopy/*`` — load / edit (draft) / save for the structure editor.

Dead simple: `open` loads the artifact, `update` keeps a draft (so a reload/crash
doesn't lose edits), `save` writes the files (overwrite the same path, or save-as
a new one).  No gate, no hashing.  Structure + sidecar codec only.

Session (§13.5) is the server-side session: the login when authenticated, else a
stable per-server-run local id for no-auth localhost.
"""
from __future__ import annotations

import re
import uuid

from flask import Blueprint, g, jsonify, request

from molbuilder import persist
from molbuilder import workingcopy as wc
from molbuilder.projects import projects_root
from molbuilder.workingcopy import SCRATCH_DIR
from molbuilder.workingcopy_structure import StructureCodec

from .files import _PickerError, _resolve_within_roots

bp = Blueprint("workingcopy", __name__)
_CODEC = StructureCodec()
_SERVER_SESSION = None

# ── State timeline (workspace-contract §4.7) — the FORMAT-BLIND indexed store ──
# A separate concern from the structure-codec working copy above: opaque session
# snapshots filed under ``<workspace_id>.<state_index>.wc.json`` in the SAME dir.
# NEVER run through StructureCodec (those are richer than {xyz, sidecar}); the
# server just moves JSON bytes.  Validate the id (no path traversal) + index.
_WS_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")   # no dots -> unambiguous index parse
_STATE_SUFFIX = ".wc.json"
_STATE_WINDOW = 30                            # rolling window kept on each write


def _default_draft_dir():
    """Draft home for a SOURCELESS workspace (a brand-new molecule not yet
    saved).  The contract persists ANY in-memory data -- a project dir is NOT
    required (workspace-contract.md) -- so a sourceless draft lives at the
    top-level ``projects_root()/.molbuilder_workspace/``, keyed by the client's
    stable workspace id.  ``update`` creates the dir lazily."""
    return projects_root()


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
    blob = _CODEC.scratch_blob(w.data)
    sc = blob.get("sidecar") or {}
    # A6 (molview-migration-plan): the Modify load takes its DATA from HERE in ONE
    # call -- the per-atom rows (element + labels + is_frozen, sidecar already
    # applied by codec.load) + the full periodicity -- so the load needs no
    # /api/selection/atoms disk read.  ``atoms_list`` is the shared wire-shape
    # builder (same rows /api/selection/atoms + every /api/modify/* response emit).
    from ._shared import atoms_list
    # The ONE resolver (§ 3a): struct.resolve_cell() -- surfaced as `resolved_cell` so
    # a cell-less structure still has a box for the render / k-grid + display accessor.
    try:
        _rc = w.data.resolve_cell()
        _resolved_cell = _rc.tolist() if _rc is not None else None
    except Exception:  # noqa: BLE001 -- periodic axis w/o lattice -> no resolved box
        _resolved_cell = None
    try:
        _ro = w.data.resolve_cell_origin()
        _resolved_origin = _ro.tolist() if _ro is not None else None
    except Exception:  # noqa: BLE001
        _resolved_origin = None
    return jsonify({
        "ok": True, "session": w.session, "source": str(w.source),
        "data": blob,
        "atoms": atoms_list(w.data),
        "periodicity": {
            "cell":          sc.get("cell"),
            "resolved_cell": _resolved_cell,
            "resolved_cell_origin": _resolved_origin,   # § 3a box anchor corner
            "axis_kind":     sc.get("axis_kind"),
            "vacuum":        sc.get("vacuum"),
            # (No "kgrid": it's a sampling knob on SiestaConfig / TransportConfig,
            # not geometry -- structure-periodicity.md.)
        },
        # F1: the per-atom annotation channels (atom-annotations.md) ride opaquely
        # through the frontend so a Modify Save doesn't clobber them to {}.
        "annotations": sc.get("annotations") or {},
    })


@bp.route("/api/workingcopy/update", methods=["POST"])
def wc_update():
    """Mirror the in-memory workspace to a transient draft (crash-safety), the
    only automatic disk write.  Two workspace shapes:

    * loaded from a file -> ``{source, data}``; draft next to the file.
    * a brand-new molecule not yet saved -> ``{workspace_id, data}``; draft at
      the top-level ``projects_root()/.molbuilder_workspace/``.  A project dir is
      NOT required -- the contract persists ANY in-memory data.
    """
    b = _body()
    source, blob = b.get("source"), b.get("data")
    if blob is None:
        return _bad("missing 'data'")
    try:
        data = _CODEC.from_scratch(blob)
    except Exception as e:  # noqa: BLE001 -- malformed body -> 400
        return _bad(f"malformed working-copy body: {e}", 400)
    if isinstance(source, str) and source:
        try:
            src = _resolve(source)
        except _PickerError as e:
            return _bad(e.message, e.status)
        wc.WorkingCopy(codec=_CODEC, session=_session(), project_dir=src.parent,
                       data=data, source=src).update(data)
    else:
        ws_id = b.get("workspace_id")
        if not isinstance(ws_id, str) or not ws_id:
            return _bad("missing 'source' or 'workspace_id'")
        wc.WorkingCopy(codec=_CODEC, session=_session(),
                       project_dir=_default_draft_dir(), data=data,
                       source=None, new_id=ws_id).update(data)
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
    blob = b.get("data")
    if blob is None:
        return _bad("missing 'data'")
    source, ws_id = b.get("source"), b.get("workspace_id")
    if b.get("target") is not None and not isinstance(b["target"], str):
        return _bad("'target' must be a string")   # else _resolve() -> 500 (b-nit)
    # Build the working copy under the DRAFT's identity -- {source} for a loaded file,
    # {workspace_id} for a not-yet-saved molecule -- so save() drops the RIGHT draft
    # (review finding b1).  The file is written to ``target`` (the save-as path).
    try:
        if isinstance(source, str) and source:
            src = _resolve(source)
            target = _resolve(b["target"]) if b.get("target") else src
            project_dir, new_id = src.parent, None
        else:
            if not b.get("target"):
                return _bad("missing 'target' (sourceless save)")
            if not isinstance(ws_id, str) or not ws_id:
                return _bad("missing 'source' or 'workspace_id'")
            src = None
            target = _resolve(b["target"])
            project_dir, new_id = _default_draft_dir(), ws_id
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
                           project_dir=project_dir, data=data,
                           source=src, new_id=new_id).save(target)
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


# ─── State timeline endpoints (§4.7) — format-blind indexed snapshots ─────── #
def _state_dir():
    """The state-timeline home — the SAME ``.molbuilder_workspace`` under the
    projects root the sourceless drafts use (``_default_draft_dir()``)."""
    return _default_draft_dir() / SCRATCH_DIR


def _valid_ws_id(ws_id) -> bool:
    return isinstance(ws_id, str) and bool(_WS_ID_RE.match(ws_id))


def _state_index(v):
    """A non-negative int (accept an int or an int-string), else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None


def _state_path(ws_id: str, idx: int):
    return _state_dir() / f"{ws_id}.{idx}{_STATE_SUFFIX}"


def _state_indices(ws_id: str):
    """Sorted list of the state indices on disk for ``ws_id`` (ascending)."""
    d = _state_dir()
    if not d.exists():
        return []
    out = []
    prefix, suffix = ws_id + ".", _STATE_SUFFIX
    for p in d.glob(f"{ws_id}.*{_STATE_SUFFIX}"):
        mid = p.name[len(prefix):-len(suffix)]
        if mid.isdigit():
            out.append(int(mid))
    return sorted(out)


@bp.route("/api/workingcopy/write-state", methods=["POST"])
def wc_write_state():
    """Write ONE opaque session snapshot to ``<workspace_id>.<state_index>.wc.json``
    (FORMAT-BLIND — the bytes are stored verbatim, never through StructureCodec).
    Keeps a rolling window of the most-recent ``_STATE_WINDOW`` indices."""
    b = _body()
    ws_id, idx, data = b.get("workspace_id"), _state_index(b.get("state_index")), b.get("data")
    if not _valid_ws_id(ws_id):
        return _bad("missing or invalid 'workspace_id'")
    if idx is None:
        return _bad("missing or invalid 'state_index' (non-negative int)")
    if data is None:
        return _bad("missing 'data'")
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    persist.write_json(_state_path(ws_id, idx), data)
    # Rolling window: drop the oldest indices beyond the window.
    for old in _state_indices(ws_id)[:-_STATE_WINDOW]:
        try:
            _state_path(ws_id, old).unlink()
        except OSError:
            pass
    return jsonify({"ok": True})


@bp.route("/api/workingcopy/read-state", methods=["POST"])
def wc_read_state():
    """Return the opaque JSON at ``<workspace_id>.<state_index>.wc.json`` (what a
    popState navigating to a history index fetches), or 404 with data=null."""
    b = _body()
    ws_id, idx = b.get("workspace_id"), _state_index(b.get("state_index"))
    if not _valid_ws_id(ws_id):
        return _bad("missing or invalid 'workspace_id'")
    if idx is None:
        return _bad("missing or invalid 'state_index' (non-negative int)")
    p = _state_path(ws_id, idx)
    if not p.exists():
        return jsonify({"ok": True, "data": None}), 404
    try:
        return jsonify({"ok": True, "data": persist.read_json(p)})
    except Exception as e:  # noqa: BLE001 -- corrupt file -> 500
        return _bad(f"could not read state {ws_id}.{idx}: {e}", 500)


@bp.route("/api/workingcopy/prune-states", methods=["POST"])
def wc_prune_states():
    """Tail-delete every state file whose index > ``above_index`` (the abandoned
    tail after a popState).  ``above_index = -1`` clears the whole timeline."""
    b = _body()
    ws_id, above = b.get("workspace_id"), b.get("above_index")
    if not _valid_ws_id(ws_id):
        return _bad("missing or invalid 'workspace_id'")
    if isinstance(above, bool) or not isinstance(above, int) or above < -1:
        return _bad("missing or invalid 'above_index' (int >= -1)")
    removed = 0
    for i in _state_indices(ws_id):
        if i > above:
            try:
                _state_path(ws_id, i).unlink()
                removed += 1
            except OSError:
                pass
    return jsonify({"ok": True, "removed": removed})


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
