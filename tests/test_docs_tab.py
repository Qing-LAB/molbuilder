"""Documents tab — the read-only docs/*.md reader (blueprints/docs.py).

Pins: the tab page renders + is in the nav; /api/docs/list groups every
docs/*.md; /api/docs/read returns one doc's text + H1 title; and the
path-safety gate rejects traversal / non-.md / outside-docs paths (the same
defence-in-depth class as the files blueprint, on a different, read-only root).

The blueprint serves the ``docs/`` tree plus the explicitly whitelisted root
README and LICENSE; all other paths outside ``docs/`` are rejected.  The
specific docs pinned below are the current tree's spine (the index +
the migration ledger, docs/README.md); update the names as the tree grows.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


# --------------------------------------------------------------------- #
#  Tab page + nav                                                       #
# --------------------------------------------------------------------- #


def test_documents_page_renders(client):
    r = client.get("/documents")
    assert r.status_code == 200
    # the shared nav lists the Documents tab (derived from tabs.TABS)
    assert b"Documents" in r.data


def test_documents_tab_registered_in_tab_order():
    from molbuilder.web.tabs import TABS
    keys = [t["key"] for t in TABS]
    assert "documents" in keys
    doc_tab = next(t for t in TABS if t["key"] == "documents")
    assert doc_tab["path"] == "/documents"


def test_toc_returns_each_document_once(client):
    """A live-edited toc.json must not make duplicate sidebar entries."""
    tree = client.get("/api/docs/toc").get_json()["tree"]
    paths = []

    def collect(nodes):
        for node in nodes:
            if "path" in node:
                paths.append(node["path"])
            collect(node.get("children", []))

    collect(tree)
    assert len(paths) == len(set(paths))


def test_toc_live_update_deduplicates_and_persists(tmp_path):
    """Duplicate and missing entries are removed from a live TOC and file."""
    from molbuilder.web.blueprints.docs import _build_toc_tree

    (tmp_path / "process").mkdir()
    (tmp_path / "process" / "code-audit.md").write_text(
        "# Code audit\n", encoding="utf-8")
    toc_path = tmp_path / "toc.json"
    toc_path.write_text(json.dumps({"tree": [{
        "label": "Process",
        "children": [
            {"path": "process/code-audit.md"},
            {"path": "process/code-audit.md"},
            {"path": "process/missing.md"},
        ],
    }]}), encoding="utf-8")

    tree = _build_toc_tree(tmp_path)
    assert [node["path"] for node in tree[0]["children"]] == [
        "process/code-audit.md"
    ]
    persisted = json.loads(toc_path.read_text(encoding="utf-8"))
    assert persisted["tree"][0]["children"] == [
        {"path": "process/code-audit.md"}
    ]


def test_toc_build_survives_readonly_docs(tmp_path):
    """A read-only docs/ (site-packages install, hardened deploy) must
    not take the sidebar down: the repaired tree is served from memory
    and the persist is silently skipped."""
    import os as _os
    from molbuilder.web.blueprints.docs import _build_toc_tree

    (tmp_path / "process").mkdir()
    (tmp_path / "process" / "code-audit.md").write_text(
        "# Code audit\n", encoding="utf-8")
    toc_path = tmp_path / "toc.json"
    toc_path.write_text(json.dumps({"tree": [{
        "label": "Process",
        "children": [{"path": "process/code-audit.md"},
                     {"path": "process/missing.md"}],   # needs repair
    }]}), encoding="utf-8")
    before = toc_path.read_text(encoding="utf-8")

    _os.chmod(tmp_path, 0o555)
    try:
        tree = _build_toc_tree(tmp_path)      # must not raise
    finally:
        _os.chmod(tmp_path, 0o755)
    assert [n["path"] for n in tree[0]["children"]] == [
        "process/code-audit.md"]              # repaired in memory...
    assert toc_path.read_text(encoding="utf-8") == before   # ...file kept


def test_toc_endpoint_tolerates_corrupt_json(client, tmp_path, monkeypatch):
    """A corrupt toc.json degrades to the empty-tree fallback, not a 500."""
    (tmp_path / "toc.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("molbuilder.web.blueprints.docs._docs_root",
                        lambda: tmp_path)
    r = client.get("/api/docs/toc")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["tree"] == []
    assert "toc.json" in body.get("note", "")


# --------------------------------------------------------------------- #
#  /api/docs/img — containment to docs/img/ + image-only                #
# --------------------------------------------------------------------- #


def test_img_serves_a_real_docs_image(client):
    r = client.get("/api/docs/img/hero-molbuilder.png")
    assert r.status_code == 200
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("bad", [
    "../design.md",                   # docs/ file OUTSIDE img/ (the old hole)
    "../toc.json",                    # ditto — served pre-hardening
    "../../molbuilder/cli.py",        # repo escape
    "../../../etc/passwd",            # filesystem escape
])
def test_img_rejects_paths_outside_docs_img(client, bad):
    r = client.get("/api/docs/img/" + bad)
    assert r.status_code == 400


def test_img_rejects_non_image_files(client, tmp_path, monkeypatch):
    (tmp_path / "img").mkdir(parents=True)
    (tmp_path / "img" / "note.txt").write_text("hi", encoding="utf-8")
    monkeypatch.setattr("molbuilder.web.blueprints.docs._docs_root",
                        lambda: tmp_path)
    r = client.get("/api/docs/img/note.txt")
    assert r.status_code == 400
    assert "not an image" in r.get_json()["error"]


def test_img_missing_file_is_404(client):
    assert client.get("/api/docs/img/nope.png").status_code == 404


# --------------------------------------------------------------------- #
#  /api/docs/list                                                       #
# --------------------------------------------------------------------- #


def test_list_groups_every_md(client):
    d = client.get("/api/docs/list").get_json()
    assert d["ok"] is True
    groups = d["groups"]
    assert groups, "expected at least one doc group"
    # the root group (docs sitting directly in docs/) is first
    assert groups[0]["name"] == "(root)"
    # every entry carries a path + a title
    for g in groups:
        for doc in g["docs"]:
            assert doc["path"].endswith(".md")
            assert doc["title"]


def test_list_finds_the_index_and_migration_audit(client):
    """The documentation index and dated migration audit are discoverable.

    The audit moved to ``archive/`` on 2026-08-22 -- it has nothing open, and
    the spine is for documents that still decide something.  What this test
    is for is unchanged: the listing reaches BOTH the index at the root and a
    dated document nested in a subdirectory, which is the part a flat walk
    silently gets wrong.  So the path is matched where it now lives rather
    than where it was.
    """
    d = client.get("/api/docs/list").get_json()
    all_paths = {doc["path"] for g in d["groups"] for doc in g["docs"]}
    assert "README.md" in all_paths
    assert any(
        path.startswith("archive/") and path.endswith("-document-migration.md")
        for path in all_paths
    )


# --------------------------------------------------------------------- #
#  /api/docs/read                                                       #
# --------------------------------------------------------------------- #


def test_read_returns_text_and_h1_title(client):
    r = client.get("/api/docs/read?path=README.md").get_json()
    assert r["ok"] is True
    assert r["path"] == "README.md"
    # title comes from the first Markdown H1
    assert r["title"] == "molbuilder — documentation"
    assert "## The rules" in r["text"]


def test_read_returns_root_readme_and_license(client):
    """The two whitelisted root-document entries are readable in the tab."""
    readme = client.get("/api/docs/read?path=../README.md").get_json()
    assert readme["ok"] is True
    assert readme["path"] == "../README.md"
    assert readme["title"] == "molbuilder — Project README"
    assert "## Quick start" in readme["text"]

    license_doc = client.get("/api/docs/read?path=../LICENSE").get_json()
    assert license_doc["ok"] is True
    assert license_doc["path"] == "../LICENSE"
    assert license_doc["title"] == "molbuilder license"
    assert "BSD 3-Clause License" in license_doc["text"]


@pytest.mark.parametrize("bad", [
    "../molbuilder/cli.py",           # escape docs/ upward
    "../../etc/passwd",               # deeper escape
    "../molbuilder.json",          # upward escape to a secret-bearing file
    "nope.md",                        # missing file
    "README",                         # no .md suffix
    "",                               # empty
])
def test_read_rejects_unsafe_or_invalid_paths(client, bad):
    r = client.get("/api/docs/read?path=" + bad)
    assert r.status_code in (400, 404)
    assert r.get_json()["ok"] is False


def test_read_rejects_a_real_file_outside_docs_via_traversal(client):
    """A path that resolves to a real, readable file OUTSIDE docs/ must still
    be refused (the escape check, not just existence)."""
    r = client.get("/api/docs/read?path=../pyproject.toml")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False
