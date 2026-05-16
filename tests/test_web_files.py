"""Tests for the /api/files/* server-side file picker endpoints.

Covers:
  * /api/files/roots             -- the configured roots are reachable
  * /api/files/list              -- happy path, ext filter, directory ordering
  * /api/files/stat              -- file + directory metadata
  * /api/files/read              -- text content + size cap behaviour
  * Path validation              -- '..' rejection, outside-root rejection
  * Configurable roots           -- molbuilder.json file_picker.roots
                                    extends the default (projects/ + CWD)

Backend contract:  docs/protocols/web-api.md  §  /api/files/*
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder import diagnostics


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


@pytest.fixture
def picker_root(tmp_path: Path):
    """A tmp directory wired in as the *only* file-picker root.

    Substitutes a stand-alone Capabilities snapshot via
    diagnostics.set_capabilities so the picker sees just this tmp
    tree, no projects/ and no CWD.  Test isolation: the conftest's
    autouse diagnostics-reset fixture restores the singleton after.
    """
    # Build a few sample files to browse:
    (tmp_path / "water.xyz").write_text(
        "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
    )
    (tmp_path / "config.json").write_text('{"engine": "pyscf"}\n')
    (tmp_path / "notes.txt").write_text("scratch\n")
    sub = tmp_path / "spectrum" / "BDT"
    sub.mkdir(parents=True)
    (sub / "water_spectra.spectra.json").write_text('{"schema_version": 2}\n')
    (sub / ".hidden").write_text("dotfile\n")

    caps = diagnostics.Capabilities(
        runtime_config={
            "file_picker": {"roots": [str(tmp_path)]},
        },
        conda_binary=None,
        conda_envs=frozenset(),
    )

    # Monkey-patch file_picker_roots to return ONLY the tmp root,
    # bypassing the projects/ + CWD defaults (we want test isolation).
    def _only_tmp_roots(self):
        return ((tmp_path.resolve(), "tmp"),)

    monkey_caps_class = type(caps)  # the frozen Capabilities dataclass
    old = monkey_caps_class.file_picker_roots
    monkey_caps_class.file_picker_roots = _only_tmp_roots
    diagnostics.set_capabilities(caps)
    try:
        yield tmp_path
    finally:
        monkey_caps_class.file_picker_roots = old
        diagnostics.reset_capabilities()


@pytest.fixture
def web(picker_root):
    """Flask test client with the picker_root fixture pre-installed."""
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    app = create_app()
    return app.test_client()


# --------------------------------------------------------------------- #
#  /api/files/roots                                                     #
# --------------------------------------------------------------------- #


class TestFilesRoots:

    def test_roots_lists_configured_root(self, web, picker_root):
        r = web.get("/api/files/roots")
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert len(j["roots"]) == 1
        assert j["roots"][0]["path"] == str(picker_root.resolve())
        assert j["roots"][0]["label"] == "tmp"
        assert j["roots"][0]["exists"] is True


# --------------------------------------------------------------------- #
#  /api/files/list                                                      #
# --------------------------------------------------------------------- #


class TestFilesList:

    def test_list_root_returns_entries(self, web, picker_root):
        r = web.get(f"/api/files/list?path={picker_root}")
        assert r.status_code == 200
        j = r.get_json()
        names = [e["name"] for e in j["entries"]]
        # Directories sort first, then files.
        assert names[0] == "spectrum"   # only dir
        # Files follow, sorted by name.
        assert "config.json" in names
        assert "notes.txt" in names
        assert "water.xyz" in names
        # The hidden file inside spectrum/BDT/.hidden isn't at this level;
        # what matters is the top-level listing didn't expose anything
        # starting with a dot.
        assert all(not e["name"].startswith(".") for e in j["entries"])

    def test_list_filters_hidden_entries(self, web, picker_root):
        r = web.get(
            f"/api/files/list?path={picker_root}/spectrum/BDT"
        )
        assert r.status_code == 200
        j = r.get_json()
        names = [e["name"] for e in j["entries"]]
        assert "water_spectra.spectra.json" in names
        assert ".hidden" not in names

    def test_list_ext_filter(self, web, picker_root):
        r = web.get(
            f"/api/files/list?path={picker_root}&ext=.xyz,.json"
        )
        assert r.status_code == 200
        names = [e["name"] for e in r.get_json()["entries"]]
        # Filter applies to FILES only -- directories must pass through
        # so the user can navigate to find filtered files inside.
        assert "spectrum" in names           # directory: always shown
        assert "water.xyz" in names          # matches .xyz
        assert "config.json" in names        # matches .json
        assert "notes.txt" not in names      # not in filter

    def test_list_ext_filter_normalises_no_dot(self, web, picker_root):
        # ext=xyz (no leading dot) should behave the same as ext=.xyz
        r = web.get(f"/api/files/list?path={picker_root}&ext=xyz")
        names = [e["name"] for e in r.get_json()["entries"]]
        assert "water.xyz" in names
        assert "config.json" not in names

    def test_list_entries_carry_kind_size_mtime(self, web, picker_root):
        r = web.get(f"/api/files/list?path={picker_root}")
        entries = {e["name"]: e for e in r.get_json()["entries"]}
        # Files report size + finite mtime; dirs report size=null.
        assert entries["water.xyz"]["kind"] == "file"
        assert entries["water.xyz"]["size"] > 0
        assert entries["water.xyz"]["mtime"] > 0
        assert entries["spectrum"]["kind"] == "directory"
        assert entries["spectrum"]["size"] is None

    def test_list_missing_path_400(self, web):
        r = web.get("/api/files/list")
        assert r.status_code == 400
        assert "missing 'path'" in r.get_json()["error"]

    def test_list_nonexistent_path_404(self, web, picker_root):
        r = web.get(
            f"/api/files/list?path={picker_root}/nope_no_such_dir"
        )
        assert r.status_code == 404

    def test_list_file_not_directory_400(self, web, picker_root):
        # Pointing list at a file (not a dir) is a usage error.
        r = web.get(
            f"/api/files/list?path={picker_root}/water.xyz"
        )
        assert r.status_code == 400


# --------------------------------------------------------------------- #
#  /api/files/stat                                                      #
# --------------------------------------------------------------------- #


class TestFilesStat:

    def test_stat_file(self, web, picker_root):
        r = web.get(
            f"/api/files/stat?path={picker_root}/water.xyz"
        )
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["kind"] == "file"
        assert j["size"] > 0
        assert j["mtime"] > 0

    def test_stat_directory(self, web, picker_root):
        r = web.get(
            f"/api/files/stat?path={picker_root}/spectrum"
        )
        assert r.status_code == 200
        j = r.get_json()
        assert j["kind"] == "directory"
        assert j["size"] is None

    def test_stat_nonexistent_404(self, web, picker_root):
        r = web.get(
            f"/api/files/stat?path={picker_root}/nope"
        )
        assert r.status_code == 404


# --------------------------------------------------------------------- #
#  /api/files/read                                                      #
# --------------------------------------------------------------------- #


class TestFilesRead:

    def test_read_returns_text(self, web, picker_root):
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz"
        )
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["text"].startswith("3\nwater")
        assert j["size"] == len(j["text"])

    def test_read_respects_max_bytes_with_413(self, web, picker_root):
        # File is ~35 bytes; cap at 5 → 413 with the file's actual size.
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz&max_bytes=5"
        )
        assert r.status_code == 413
        j = r.get_json()
        assert j["ok"] is False
        assert j["size"] > 5

    def test_read_directory_400(self, web, picker_root):
        r = web.get(
            f"/api/files/read?path={picker_root}/spectrum"
        )
        assert r.status_code == 400

    def test_read_rejects_invalid_max_bytes(self, web, picker_root):
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz"
            f"&max_bytes=not_an_int"
        )
        assert r.status_code == 400

    def test_read_rejects_max_bytes_above_ceiling(self, web, picker_root):
        # Hard ceiling is 16 MB.
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz"
            f"&max_bytes=999999999"
        )
        assert r.status_code == 400

    def test_read_non_utf8_400(self, web, picker_root):
        bad = picker_root / "binary.dat"
        bad.write_bytes(b"\xff\xfe\xfd\xfc not valid utf-8")
        r = web.get(f"/api/files/read?path={bad}")
        assert r.status_code == 400
        assert "UTF-8" in r.get_json()["error"]


# --------------------------------------------------------------------- #
#  Path-traversal defense                                               #
# --------------------------------------------------------------------- #


class TestPathTraversalDefense:
    """The picker must never let a request reach outside the configured
    roots, no matter what path the user supplies.  Two layers of
    defense: raw '..' rejection AND resolved-path-must-be-inside-root.
    """

    def test_dot_dot_in_raw_path_rejected(self, web):
        # Even before resolution, a path with .. is rejected.  This
        # avoids ambiguity for users who type '..' assuming it would
        # be normalised harmlessly.
        r = web.get("/api/files/list?path=../../etc")
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_absolute_path_outside_root_rejected(self, web, picker_root):
        # /etc is not inside the tmp picker root → outside-root reject.
        r = web.get("/api/files/list?path=/etc")
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]

    def test_symlink_to_outside_root_rejected(self, web, picker_root):
        # Symlink resolves to /tmp (outside the picker_root tmp).
        # _resolve_within_roots follows symlinks before checking, so
        # the resolved path is what the boundary check sees.
        link = picker_root / "leak"
        link.symlink_to("/etc")
        r = web.get(f"/api/files/list?path={link}")
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]

    def test_empty_path_400(self, web):
        r = web.get("/api/files/stat?path=")
        assert r.status_code == 400


# --------------------------------------------------------------------- #
#  Roots from molbuilder.json                                           #
# --------------------------------------------------------------------- #


class TestProjectsPageAndShim:
    """The Projects tab page renders + the projects-selection shim
    is served + each subscriber tab includes both the shim and the
    banner DOM."""

    def test_projects_page_renders(self, web, picker_root):
        r = web.get("/projects")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert 'id="explorer"' in body
        assert 'id="status-path"' in body
        # Tab nav must include the Projects entry.
        assert ">Projects<" in body
        assert "projects/explorer.js" in body
        assert "projects/explorer.css" in body

    def test_projects_selection_shim_served(self, web, picker_root):
        r = web.get("/static/lib/projects-selection.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # The shim must expose its public init() entry point and
        # subscribe to both event channels (cross-tab storage,
        # same-tab CustomEvent).
        assert "molbuilderProjectsSelection" in body
        assert "init" in body
        assert "molbuilder.current_file" in body
        assert "molbuilder.selection" in body
        assert 'addEventListener("storage"' in body

    @pytest.mark.parametrize("path", ["/spectra", "/modify", "/watch"])
    def test_subscriber_tabs_include_banner_and_shim(
        self, web, picker_root, path,
    ):
        r = web.get(path)
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert 'id="projects-banner"' in body, path
        assert "projects-selection.js" in body, path
        # The shim's CSS classes are required by the banner contract.
        assert 'class="ps-path"' in body, path
        assert 'class="ps-use-btn"' in body, path


class TestRootsFromConfig:
    """file_picker.roots in molbuilder.json adds roots beyond the
    defaults (projects/, CWD)."""

    def test_runtime_config_accepts_file_picker_roots(self, tmp_path):
        # Round-trip through read_config to confirm the new section
        # parses + survives _normalise.
        from molbuilder.runtime_config import (
            read_config, get_file_picker_roots,
        )
        cfg_file = tmp_path / "molbuilder.json"
        cfg_file.write_text('{"file_picker": {"roots": ["~/scratch", '
                            '"/data/shared"]}}')
        cfg = read_config(cfg_file)
        assert get_file_picker_roots(cfg) == ["~/scratch", "/data/shared"]

    def test_runtime_config_rejects_non_list_roots(self, tmp_path):
        from molbuilder.runtime_config import (
            read_config, RuntimeConfigError,
        )
        cfg_file = tmp_path / "molbuilder.json"
        cfg_file.write_text('{"file_picker": {"roots": "not-a-list"}}')
        with pytest.raises(RuntimeConfigError, match="must be a list"):
            read_config(cfg_file)

    def test_runtime_config_rejects_empty_string_root(self, tmp_path):
        from molbuilder.runtime_config import (
            read_config, RuntimeConfigError,
        )
        cfg_file = tmp_path / "molbuilder.json"
        cfg_file.write_text('{"file_picker": {"roots": ["valid", ""]}}')
        with pytest.raises(RuntimeConfigError, match="non-empty strings"):
            read_config(cfg_file)

    def test_capabilities_includes_defaults_plus_config_roots(self, tmp_path):
        # Without monkey-patching: real Capabilities.file_picker_roots
        # should include projects/ + CWD + any existing config roots.
        # Non-existent config roots are silently dropped.
        from molbuilder.diagnostics import Capabilities
        existing = tmp_path / "exists"
        existing.mkdir()
        caps = Capabilities(
            runtime_config={
                "file_picker": {
                    "roots": [str(existing), "/nonexistent/never/here"],
                },
            },
        )
        roots = caps.file_picker_roots()
        paths = [str(p) for p, _ in roots]
        # Defaults always present.
        assert any(p.endswith("/projects") for p in paths)
        # Existing config root added.
        assert str(existing.resolve()) in paths
        # Non-existent config root silently dropped.
        assert not any("nonexistent" in p for p in paths)
