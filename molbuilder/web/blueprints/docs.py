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


def _resolve_label(path: str, root: Path, label: Optional[str] = None) -> str:
    """Return a display label for ``path`` — explicit label wins, else
    the doc's H1 title, else the filename stem."""
    if label:
        return label
    try:
        resolved = root / path
        if resolved.is_file():
            return _title_of(resolved)
    except Exception:
        pass
    return Path(path).stem


def _subtitle_of(path: str, root: Path) -> str:
    """A short, human-readable subtitle for a doc.  Strips the
    `molbuilder — ` prefix from the H1 title so the sidebar shows
    something concise like \"Design\" instead of the full
    \"molbuilder — design\"."""
    try:
        resolved = root / path
        if not resolved.is_file():
            return ""
        title = _title_of(resolved)
        if title.lower().startswith("molbuilder"):
            idx = title.find("—")
            if idx < 0:
                idx = title.find("-")
            if idx >= 0:
                return title[idx + 1:].strip()
        return title
    except Exception:
        return ""


def _build_toc_tree(root: Path) -> List[Dict]:
    """Build the full document tree from ``toc.json``.

    Returns a list of top-level tree nodes.  Each node is either a
    **directory** (has ``children``) or a **document** (has ``path``).
    Archive children are auto-populated from the filesystem.
    """
    import json as _json
    toc_path = root / "toc.json"
    if not toc_path.is_file():
        return []   # fallback: empty tree
    toc = _json.loads(toc_path.read_text(encoding="utf-8"))

    # Collect all paths referenced in the toc tree (for auto-discovery).
    _toc_paths: set = set()
    _new_paths: Dict[str, list] = {}   # domain_label -> [new entries to persist]

    def _collect_paths(node: dict) -> None:
        if "path" in node:
            _toc_paths.add(node["path"])
        for c in node.get("children", []):
            _collect_paths(c)
    for n in toc.get("tree", []):
        _collect_paths(n)

    def _populate(node: dict, parent_rel: str = "") -> dict:
        """Recursively resolve a toc.json node into a render-ready dict."""
        if "path" in node and "children" not in node:
            # Plain leaf — no sub-documents.
            rel = node["path"]
            return {
                "path":     rel,
                "label":    _resolve_label(rel, root, node.get("label")),
                "subtitle": _subtitle_of(rel, root),
            }
        if "path" in node:
            # Master doc with sub-documents — has both path and children.
            rel = node["path"]
            kids = [_populate(c) for c in node.get("children", [])]
            return {
                "path":     rel,
                "label":    _resolve_label(rel, root, node.get("label")),
                "subtitle": _subtitle_of(rel, root),
                "children": kids,
            }
        # Directory node — has label + optional children.
        label = node.get("label", "")
        meta  = node.get("meta", "")
        collapsed = node.get("collapsed", False)
        children = list(node.get("children", []))

        # Auto-discover: for each domain directory, find .md files not
        # listed in toc.json and append them as unlisted children.
        domain_dir = node.get("_dir")
        if not domain_dir:
            # Derive from the first child's path prefix.
            for c in children:
                if "path" in c:
                    prefix = c["path"].split("/")[0] if "/" in c["path"] else ""
                    if prefix and (root / prefix).is_dir():
                        domain_dir = prefix
                        break
        if domain_dir:
            subdir = root / domain_dir
            if subdir.is_dir():
                listed = {c["path"] for c in children if "path" in c}
                # Build stem→path map for R5 prefix-grouping.
                _stem_to_parent: Dict[str, str] = {
                    Path(c["path"]).stem: c["path"]
                    for c in children if "path" in c
                }
                new = []
                for p in sorted(subdir.glob("*.md"), key=lambda p: p.name.lower()):
                    rel = str(p.relative_to(root)).replace(os.sep, "/")
                    if rel in listed or rel in _toc_paths:
                        continue
                    # R5: sub-documents share the master's stem as a prefix.
                    stem = p.stem
                    parent = None
                    for pstem in _stem_to_parent:
                        if stem.startswith(pstem + "-"):
                            parent = pstem
                            break
                    entry = {"path": rel}
                    if parent:
                        entry["_parent"] = parent
                    new.append(entry)
                if new:
                    _new_paths[label] = new
                    children.extend(new)

        # Resolve children, nesting sub-documents (R5 prefix convention)
        # under their parent doc.
        resolved: list = []
        pending_subs: Dict[str, list] = {}   # parent_stem -> [sub-doc children]
        for c in children:
            if "_parent" in c:
                parent_stem = c["_parent"]
                pending_subs.setdefault(parent_stem, []).append(c)
            else:
                resolved.append(c)
        # Now resolve: for each top-level child, if it's a doc node
        # with pending sub-documents, nest them.
        final_children = []
        for c in resolved:
            if "path" in c:
                stem = Path(c["path"]).stem
                if stem in pending_subs:
                    c = dict(c)   # shallow copy
                    c["children"] = pending_subs.pop(stem)
            final_children.append(c)
        # Any leftover subs (parent not found) — append flat.
        for subs in pending_subs.values():
            final_children.extend(subs)

        resolved_children = [_populate(c) for c in final_children] if final_children else []
        return {
            "label":     label,
            "meta":      meta,
            "collapsed": collapsed,
            "children":  resolved_children,
        }

    # Archive auto-population: scan archive/ tree and build a subtree.
    archive_children: list = []
    archive_root = root / "archive"
    if archive_root.is_dir():
        def _scan_dir(dir_path: Path) -> list:
            items = []
            for p in sorted(dir_path.iterdir(), key=lambda p: (p.is_dir(), p.name.lower())):
                if p.is_dir():
                    sub_items = _scan_dir(p)
                    if sub_items:
                        items.append({
                            "label": p.name,
                            "children": sub_items,
                        })
                elif p.suffix.lower() == ".md":
                    rel = str(p.relative_to(root)).replace(os.sep, "/")
                    items.append({
                        "path":     rel,
                        "label":    _resolve_label(rel, root),
                        "subtitle": _subtitle_of(rel, root),
                    })
            return items

        archive_children = _scan_dir(archive_root)

    # Build the tree.
    tree = []
    for node in toc.get("tree", []):
        if node.get("label") == "Archive":
            node["children"] = archive_children
        tree.append(_populate(node))

    # Persist newly-discovered paths back to toc.json.
    if _new_paths:
        _dirty = False
        for tn in toc.get("tree", []):
            label = tn.get("label", "")
            if label in _new_paths:
                existing = tn.setdefault("children", [])
                for entry in _new_paths[label]:
                    parent = entry.get("_parent")
                    if parent:
                        for pc in existing:
                            stem = Path(pc.get("path", "")).stem
                            if stem == parent:
                                pc.setdefault("children", []).append(
                                    {"path": entry["path"]})
                                break
                        else:
                            existing.append({"path": entry["path"]})
                    else:
                        existing.append({"path": entry["path"]})
                _dirty = True
        if _dirty:
            tmp = toc_path.with_suffix(".tmp")
            tmp.write_text(_json.dumps(toc, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            os.replace(str(tmp), str(toc_path))

    return tree


@bp.route("/api/docs/toc", methods=["GET"])
def api_docs_toc():
    """Document tree for the sidebar — built from ``docs/toc.json``.

    Response::

        {ok, tree: [{label, meta?, path?, collapsed?, children?}]}

    Leaf nodes have ``path``; directory nodes have ``children``.
    The root README (``../README.md``) is a special ``readme`` key.
    """
    root = _docs_root()
    if root is None:
        return jsonify({"ok": False, "error": "docs/ not available"}), 404

    toc_path = root / "toc.json"
    if not toc_path.is_file():
        return jsonify({"ok": True, "tree": [], "readme": None})

    import json as _json
    toc = _json.loads(toc_path.read_text(encoding="utf-8"))

    tree = _build_toc_tree(root)

    # Root README
    readme_info = toc.get("root_readme")
    readme = None
    if readme_info:
        repo_readme = (root.parent / "README.md")
        if repo_readme.is_file():
            readme = {
                "path":     "../README.md",
                "label":    "README.md",
                "subtitle": "project root",
            }

    return jsonify({"ok": True, "tree": tree, "readme": readme})


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


@bp.route("/api/docs/img/<path:img_path>", methods=["GET"])
def api_docs_img(img_path: str):
    """Serve an image from ``docs/img/<img_path>``.

    The Documents tab rewrites ``<img>`` src attributes to this
    endpoint so images render regardless of the page's base URL.
    """
    from flask import send_file
    root = _docs_root()
    if root is None:
        return jsonify({"ok": False, "error": "docs/ not available"}), 404
    resolved = root / "img" / img_path
    # Defence: reject .. traversal.
    try:
        resolved = resolved.resolve()
        if os.path.commonpath([resolved, root]) != str(root):
            raise ValueError
    except (ValueError, OSError):
        return jsonify({"ok": False, "error": "invalid path"}), 400
    if not resolved.is_file():
        return jsonify({"ok": False, "error": "not found"}), 404
    return send_file(str(resolved))


@bp.route("/api/docs/read", methods=["GET"])
def api_docs_read():
    """One doc's raw Markdown text (rendered client-side).

    Query: ``path`` -- relative to ``docs/`` (from ``/api/docs/list``).
    Special: ``path=../README.md`` reads the project root README
    (outside ``docs/``; validated separately).
    Response: ``{ok, path, title, text}`` or ``{ok:false, error}`` (400/404).
    """
    root = _docs_root()
    if root is None:
        return jsonify({"ok": False, "error": "docs/ not available"}), 404
    raw_path = (request.args.get("path") or "").strip()

    # Root README: lives outside docs/, validated separately.
    if raw_path == "../README.md":
        repo_readme = root.parent / "README.md"
        if not repo_readme.is_file():
            return jsonify({"ok": False, "error": "README not found"}), 404
        try:
            text = repo_readme.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return jsonify({"ok": False, "error": f"read failed: {exc}"}), 404
        return jsonify({
            "ok":    True,
            "path":  raw_path,
            "title": "molbuilder — Project README",
            "text":  text,
        })

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
