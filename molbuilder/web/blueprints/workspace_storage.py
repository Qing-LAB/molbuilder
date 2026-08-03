"""``/api/workspace-storage/*`` — the WORKSPACE'S BYTE STORAGE (server backend).

MODULE: the workspace's on-disk half.  FORMAT-BLIND storage of one opaque JSON
blob per ``{workspace_id, state_index}``, written and read back verbatim.  It
never parses what it stores.

WHAT IT IS NOT — and the old name said otherwise.  This was ``state_timeline.py``
serving ``/api/state-timeline/*``, named after MolView's **timeline**: the
sequence of saved points, the position on it, point 0, the badge, and the policy
of what a save records and what to prune.  **None of that is here.**  That is
``lib/molview/history.js``, MolView's own submodule, and nothing in this file
knows a sequence exists — no order, no position, no notion that index 3 follows
index 2.

The name mattered because it misdirected readers about ownership, repeatedly and
in both directions: once into reading this as a domain module worth promoting out
of the web layer, once into reading it as MolView-private plumbing worth hiding.
It is neither.  It is the **workspace's** storage, and the workspace is a public
module several savers share.

The proof that it is not MolView's: ``lib/inspectors/structure.js`` stores
``{showing: "<path>"}`` under its own ``SHOWING_TAG`` through these exact routes —
a file path, not a history.  Any tag may use it (workspace.md § 4).

**States, not a timeline.**  The word kept in the internals is deliberate: this
stores numbered *states*; the *timeline* is the sequence MolView makes of them.

ROLE: byte storage keyed by ``{workspace_id, state_index}`` under
``<projects_root>/.molbuilder_workspace/states/<workspace_id>.<state_index>.wc.json``.
It owns *where* the bytes live; the consumer owns *what* they are and *when* to
write.  It must never interpret the snapshot.

USED BY:
  - lib/workspace/dispatcher.js (window.molbuilder.workspace) — ``persist`` POSTs
    ``workspace-storage/write``, ``readState`` POSTs ``.../read``,
    ``pruneStatesAbove`` POSTs ``.../prune``.  That dispatcher is the sole client
    of these routes; every other saver reaches them through it.

Contract: docs/web/workspace.md § 9.

History: these routes lived at ``/api/workingcopy/*`` beside an obsolete
structure-editor "door" (open/save/update/…).  The live persistence was extracted,
misnamed ``state_timeline``, and renamed here to what it actually is.
"""
from __future__ import annotations

import logging
import re

from flask import Blueprint, jsonify, request

from molbuilder import persist
from molbuilder.projects import projects_root

bp = Blueprint("workspace_storage", __name__)
_log = logging.getLogger(__name__)

# Opaque session snapshots filed under ``<workspace_id>.<state_index>.wc.json``.
# NEVER parsed as structure; the server just moves JSON bytes.  Validate the id
# (no path traversal) + index.
_WS_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")   # no dots -> unambiguous index parse
_STATE_SUFFIX = ".wc.json"
_STATE_WINDOW = 30                            # rolling window kept on each write

# Residue accounting (workspace-contract.md §4.2).  The rolling window bounds each
# LIVE workspace to <= _STATE_WINDOW files, but a crashed / closed tab leaves its
# whole timeline behind: a new tab mints a fresh workspace_id, so the old id's files
# are never touched again.  Crash RECOVERY is deliberately out of scope, and the
# files are harmless (hidden dot-dir, never listed by the sidebar, globs are per-id
# so they never slow anything).  We do NOT garbage-collect them -- but we WARN once
# per process when the pile crosses this many files so an operator can wipe
# ``<projects_root>/.molbuilder_workspace/states/`` by hand if they care.
_RESIDUE_WARN_FILES = 300                     # ~10 abandoned tabs' worth
_residue_warned = False

#: Top-level workspace home under the projects root (gitignore-able; created
#: lazily).  Same on-disk name the module has always used, so existing state
#: files keep their path ``<projects_root>/.molbuilder_workspace/states/``.
SCRATCH_DIR = ".molbuilder_workspace"


def _default_draft_dir():
    """Home for the workspace state files.  The contract persists ANY in-memory
    data -- a project dir is NOT required (workspace.md) -- so state files live at
    the top-level ``projects_root()``, keyed by ``workspace_id``.

    KNOWN DEFECT, tracked as task #46 and deliberately NOT fixed here: the app
    resolves the projects root TWICE.  The file picker asks
    ``Capabilities.file_picker_roots()`` -- the resolved answer a deployment or a
    test can point elsewhere -- and this asks ``projects_root()``, which is always
    ``Path.cwd()/projects`` and cannot be redirected.  So a test that pins the
    picker at a temp directory still has its SESSION STATE land in the developer's
    live ``projects/`` tree.

    A fix was written and reverted, because it carried its own bug: it fell back
    to ``projects_root()`` when Capabilities answered nothing -- but Capabilities
    answers nothing ONLY when ``projects_root().expanduser().resolve()`` threw, so
    the fallback returns the UNRESOLVED form of a path that has just failed to
    resolve.  A second answer, produced exactly when the first one broke.  The
    real fix resolves once and fails loudly; #46 carries it.
    """
    return projects_root()


def _bad(msg, status=400):
    return jsonify({"ok": False, "error": msg}), status


def _body():
    return request.get_json(silent=True) or {}


def _state_dir():
    """The storage home -- a ``states/`` SUBDIR of the workspace dir."""
    return _default_draft_dir() / SCRATCH_DIR / "states"


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


@bp.route("/api/workspace-storage/write", methods=["POST"])
def ws_write_state():
    """Write ONE opaque session snapshot to ``<workspace_id>.<state_index>.wc.json``
    (FORMAT-BLIND — the bytes are stored verbatim, never parsed as structure).
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
    _warn_if_residue_piling(d)
    return jsonify({"ok": True})


def _warn_if_residue_piling(state_dir) -> None:
    """WARN once per process if abandoned-workspace snapshots have piled up.  We do
    not GC them (crash recovery is out of scope, and they're harmless), but a large
    pile is worth surfacing so an operator can clear the dir by hand."""
    global _residue_warned
    if _residue_warned:
        return
    try:
        total = sum(1 for _ in state_dir.glob(f"*{_STATE_SUFFIX}"))
    except OSError:
        return
    if total >= _RESIDUE_WARN_FILES:
        _residue_warned = True
        _log.warning(
            "workspace storage residue: %d snapshot files under %s -- these "
            "are leftovers from closed/crashed tabs (never garbage-collected; crash "
            "recovery is out of scope). Harmless, but you may delete the directory to "
            "reclaim space.", total, state_dir,
        )


@bp.route("/api/workspace-storage/read", methods=["POST"])
def ws_read_state():
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


@bp.route("/api/workspace-storage/prune", methods=["POST"])
def ws_prune_states():
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
