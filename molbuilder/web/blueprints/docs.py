"""Documents blueprint -- read-only access to the project's ``docs/*.md``.

Powers the **Documents** tab (``/documents``): a convenient in-app reader for
every design doc + guide + README under the repo's ``docs/`` directory,
rendered through the same ``marked`` + ``DOMPurify`` pipeline the Results-tab
markdown inspector uses (``lib/markdown-render.js``).

This is **app-shipped reference content**, a DIFFERENT domain from the
projects file-access framework (``/api/files/*``, which is scoped to the user's
``projects/`` picker roots and supports write/delete).  ``docs/`` is read-only
documentation, so it gets its own tiny read-only endpoint rather than being
bolted onto the picker roots (which would wrongly expose docs to the mutating
file ops).  Path handling mirrors the files blueprint's defence-in-depth
(reject ``..``, resolve, confirm the result stays inside ``docs/``).

Routes:

    GET /api/docs/list              grouped list of every docs/*.md
    GET /api/docs/read?path=<rel>   one doc's raw markdown text
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from flask import Blueprint, jsonify, request

bp = Blueprint("docs", __name__)


def _docs_root() -> Optional[Path]:
    """The repo's ``docs/`` directory, or ``None`` if it isn't present
    (e.g. an installed package that didn't ship the docs tree).

    Located relative to the ``molbuilder`` package: ``<pkg>/..`` is the
    repo root, whose ``docs/`` holds the design docs + guides.  Computed,
    not configured -- the docs live at a fixed spot in the source tree.
    """
    import molbuilder
    root = Path(molbuilder.__file__).resolve().parent.parent / "docs"
    return root if root.is_dir() else None


def _resolve_doc(raw_path: str, root: Path) -> Path:
    """Resolve a ``docs/``-relative path to an absolute ``.md`` file inside
    ``root``, raising ``ValueError`` on any rejection.

    Defence in depth (mirrors files._resolve_within_roots): reject a raw
    ``..`` component, resolve symlinks/relatives, then confirm the resolved
    path is still under ``root`` and is a real ``.md`` file.
    """
    if not raw_path or ".." in raw_path.split("/"):
        raise ValueError("invalid path")
    resolved = (root / raw_path).resolve()
    # Must stay inside docs/ (os.path.commonpath rejects escapes).
    try:
        if os.path.commonpath([resolved, root]) != str(root):
            raise ValueError("path escapes docs/")
    except ValueError:
        raise ValueError("path escapes docs/")
    if resolved.suffix.lower() != ".md" or not resolved.is_file():
        raise ValueError("not a docs .md file")
    return resolved


def _title_of(md_path: Path) -> str:
    """The doc's display title: its first Markdown H1 (``# ...``), falling
    back to the filename stem.  Reads only the head of the file so listing
    ~100 docs stays cheap."""
    try:
        with md_path.open(encoding="utf-8", errors="replace") as fh:
            head = fh.read(2048)
    except OSError:
        return md_path.stem
    for line in head.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip() or md_path.stem
        # Stop at the first non-blank, non-H1 line only if it's clearly
        # body -- but a leading blockquote/frontmatter is common, so just
        # scan the whole head for the first H1.
    return md_path.stem


@bp.route("/api/docs/list", methods=["GET"])
def api_docs_list():
    """Every ``docs/*.md`` (recursive), grouped by its top-level directory.

    Response::

        {ok, groups: [{name, docs: [{path, title}]}]}

    ``path`` is relative to ``docs/`` (what ``/api/docs/read`` takes);
    ``name`` is the directory group ("(root)" for docs sitting directly
    in ``docs/``).  Groups + docs are sorted; the root group is first.
    """
    root = _docs_root()
    if root is None:
        return jsonify({"ok": True, "groups": [],
                        "note": "docs/ directory not found in this install"})

    by_group: Dict[str, List[Dict[str, str]]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".md"):
                continue
            full = Path(dirpath) / fn
            rel = full.relative_to(root)
            group = str(rel.parent) if str(rel.parent) != "." else "(root)"
            by_group.setdefault(group, []).append({
                "path":  str(rel).replace(os.sep, "/"),
                "title": _title_of(full),
            })

    # Root group first, then the rest alphabetically; docs sorted by title.
    def _group_key(name: str):
        return (name != "(root)", name.lower())

    groups = []
    for name in sorted(by_group, key=_group_key):
        docs = sorted(by_group[name], key=lambda d: d["title"].lower())
        groups.append({"name": name, "docs": docs})

    return jsonify({"ok": True, "groups": groups})


@bp.route("/api/docs/read", methods=["GET"])
def api_docs_read():
    """One doc's raw Markdown text (rendered client-side).

    Query: ``path`` -- relative to ``docs/`` (from ``/api/docs/list``).
    Response: ``{ok, path, title, text}`` or ``{ok:false, error}`` (400/404).
    """
    root = _docs_root()
    if root is None:
        return jsonify({"ok": False, "error": "docs/ not available"}), 404
    raw_path = (request.args.get("path") or "").strip()
    try:
        resolved = _resolve_doc(raw_path, root)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return jsonify({"ok": False, "error": f"read failed: {exc}"}), 404
    return jsonify({
        "ok":    True,
        "path":  raw_path,
        "title": _title_of(resolved),
        "text":  text,
    })
