"""Documents tab — the read-only docs/*.md reader (blueprints/docs.py).

Pins: the tab page renders + is in the nav; /api/docs/toc groups every
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


# `/api/docs/list` and its two tests stood here.  The route was the tab's
# original flat listing; the commit AFTER the one that added it replaced the
# listing with the `toc.json` tree, and no browser has called it since.
# `/api/docs/toc` also auto-discovers new documents, so it was not even a
# fallback.  Deleted 2026-09-07.


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


def _isolated_docs(tmp_path):
    """A throwaway copy of `docs/`, because the builder PERSISTS.

    `_build_toc_tree` writes the normalised tree back to `toc.json`, so a test
    pointed at the real `docs/` modifies the repository — which is how a suite
    run came back with `docs/toc.json` dirty (2026-09-08).
    """
    import shutil
    from pathlib import Path
    dst = tmp_path / "docs"
    shutil.copytree(Path(__file__).resolve().parents[1] / "docs", dst)
    return dst


def test_a_new_archive_doc_appears_once_on_the_FIRST_render(tmp_path):
    """The render after a new document lands must already be correct.

    **This bug erased its own evidence, which is why it survived.** Adding a
    `docs/archive/*.md` made it render TEN times; the builder then persisted the
    tidied tree, so every later render was clean and nothing was ever visibly
    wrong. Measured 2026-09-08: call 1 gave 230 paths with the new file ×10,
    call 2 gave 221 and clean.

    Two causes, and this test covers both because either alone reproduces it:

    * `_toc_paths` was computed once from the original tree and never updated as
      entries were appended, so every directory node resolving to the same
      directory appended the same unlisted file.
    * `domain_dir` came from `path.split("/")[0]` — the FIRST component — so
      each nested group resolved to its top-level ancestor. Ten sidebar groups
      resolved to `archive`, which was the multiplier.

    Asserting the FIRST call is the whole point: asserting the second passes
    against the bug.
    """
    from molbuilder.web.blueprints.docs import _build_toc_tree
    root = _isolated_docs(tmp_path)
    (root / "archive" / "9999-01-01-brand-new.md").write_text(
        "# Brand new\n", encoding="utf-8")

    def paths_of():
        out = []

        def collect(nodes):
            for n in nodes:
                if "path" in n:
                    out.append(n["path"])
                collect(n.get("children", []))
        collect(_build_toc_tree(root))
        return out

    first = paths_of()
    assert first.count("archive/9999-01-01-brand-new.md") == 1, (
        f"a new archive doc appears "
        f"{first.count('archive/9999-01-01-brand-new.md')}x in the FIRST "
        f"render (should be 1)")
    assert len(first) == len(set(first)), (
        "duplicates in the first render: "
        + repr(sorted({p for p in first if first.count(p) > 1})))
    # ...and it stays right, which the old code also managed.
    assert paths_of() == first


def test_a_nested_archive_group_scans_its_OWN_directory(tmp_path):
    """A group for `archive/old_docs/` must not glob `archive/`.

    The consequence of getting this wrong is not only the duplication above: a
    nested group would surface its ancestor's documents as though they were its
    own children, so the sidebar's shape would stop matching the tree on disk.
    """
    from molbuilder.web.blueprints.docs import _build_toc_tree
    root = _isolated_docs(tmp_path)
    # a doc in the NESTED directory, and one in its ancestor -- each must land
    # in exactly one group, its own.
    (root / "archive" / "old_docs" / "zz-nested-only.md").write_text(
        "# Nested\n", encoding="utf-8")
    (root / "archive" / "zz-top-only.md").write_text("# Top\n",
                                                     encoding="utf-8")

    def group_children(nodes, label):
        for n in nodes:
            if "path" not in n and n.get("label") == label:
                return [c.get("path") for c in n.get("children", [])
                        if "path" in c]
            hit = group_children(n.get("children", []), label)
            if hit is not None:
                return hit
        return None

    tree = _build_toc_tree(root)
    nested = group_children(tree, "old_docs") or []
    assert "archive/old_docs/zz-nested-only.md" in nested, (
        f"the nested group did not discover its own document: {nested[:6]}")
    assert "archive/zz-top-only.md" not in nested, (
        "the nested group surfaced its ANCESTOR's document as its own child")


def test_two_groups_over_one_directory_do_not_each_append_it(tmp_path):
    """Two sidebar groups may curate the SAME folder — a legitimate shape.

    `docs/toc.json` is hand-curated, so splitting one directory across two
    labelled groups ("Recent" / "Older") is an ordinary thing to write. Both
    groups then resolve to the same directory, and an unlisted file in it is
    discovered by each — appearing twice.

    **This is the case the shipped tree does not currently contain**, which is
    why it needs its own fixture: reverting the `_toc_paths.update(...)` guard
    leaves every other test in this file green (measured 2026-09-08). Without
    this test that line would be unproven, and unproven code is either wrong or
    unnecessary.
    """
    import json
    from molbuilder.web.blueprints.docs import _build_toc_tree
    root = tmp_path / "docs"
    (root / "notes").mkdir(parents=True)
    for name in ("alpha.md", "beta.md", "unlisted.md"):
        (root / "notes" / name).write_text(f"# {name}\n", encoding="utf-8")
    (root / "toc.json").write_text(json.dumps({"tree": [
        {"label": "Recent", "children": [{"path": "notes/alpha.md"}]},
        {"label": "Older",  "children": [{"path": "notes/beta.md"}]},
    ]}), encoding="utf-8")

    paths = []

    def collect(nodes):
        for n in nodes:
            if "path" in n:
                paths.append(n["path"])
            collect(n.get("children", []))
    collect(_build_toc_tree(root))

    assert paths.count("notes/unlisted.md") == 1, (
        f"an unlisted file in a directory curated by TWO groups was appended "
        f"{paths.count('notes/unlisted.md')}x — `_toc_paths` is not growing "
        f"as entries are added")
    assert len(paths) == len(set(paths)), f"duplicates: {paths}"
