"""Documents tab — the read-only docs/*.md reader (blueprints/docs.py).

Pins: the tab page renders + is in the nav; /api/docs/list groups every
docs/*.md; /api/docs/read returns one doc's text + H1 title; and the
path-safety gate rejects traversal / non-.md / outside-docs paths (the same
defence-in-depth class as the files blueprint, on a different, read-only root).
"""
from __future__ import annotations

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
    # protocols/ is a known group with many docs
    names = [g["name"] for g in groups]
    assert "protocols" in names
    # every entry carries a path + a title
    for g in groups:
        for doc in g["docs"]:
            assert doc["path"].endswith(".md")
            assert doc["title"]


def test_list_finds_the_overview_and_the_contract(client):
    """The two docs this session authored are discoverable."""
    d = client.get("/api/docs/list").get_json()
    all_paths = {doc["path"] for g in d["groups"] for doc in g["docs"]}
    assert "batch-workflow-overview.md" in all_paths
    assert "protocols/staged-execution.md" in all_paths


# --------------------------------------------------------------------- #
#  /api/docs/read                                                       #
# --------------------------------------------------------------------- #


def test_read_returns_text_and_h1_title(client):
    r = client.get("/api/docs/read?path=batch-workflow-overview.md").get_json()
    assert r["ok"] is True
    assert r["path"] == "batch-workflow-overview.md"
    # title comes from the first Markdown H1
    assert r["title"] == "Running a batch of jobs — the big picture"
    assert "## 1. The one-sentence version" in r["text"]


def test_read_a_nested_doc(client):
    r = client.get("/api/docs/read?path=protocols/staged-execution.md").get_json()
    assert r["ok"] is True
    assert "JobSet" in r["text"]


@pytest.mark.parametrize("bad", [
    "../molbuilder/cli.py",          # escape docs/ upward
    "../../etc/passwd",              # deeper escape
    "protocols/../../setup.py",      # escape via a nested ..
    "nope.md",                       # missing file
    "protocols",                     # a directory, not a .md
    "README",                        # no .md suffix
    "",                              # empty
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
