"""Tests for the /api/files/* server-side file picker endpoints.

Covers:
  * /api/files/roots             -- the single projects/ root is reported
  * /api/files/list              -- happy path, ext filter, directory ordering
  * /api/files/stat              -- file + directory metadata
  * /api/files/read              -- text content + size cap behaviour
  * Path validation              -- '..' rejection, outside-root rejection
  * Sidebar partial + JS         -- the persistent sidebar is included in
                                    every tab and the supporting JS / CSS
                                    is served

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
    """A tmp directory wired in as the picker's root.

    Replaces the real ``projects/`` default with this tmp tree by
    monkey-patching :meth:`Capabilities.file_picker_roots`.  Test
    isolation: the conftest's autouse diagnostics-reset fixture
    restores the singleton afterwards.
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
        runtime_config={},
        conda_binary=None,
        conda_envs=frozenset(),
    )

    # Monkey-patch file_picker_roots to return ONLY the tmp root,
    # bypassing the real projects/ default.
    def _only_tmp_roots(self):
        return ((tmp_path.resolve(), "projects"),)

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

    def test_roots_lists_single_projects_root(self, web, picker_root):
        # Single root by design (v1): just projects/.  No CWD, no
        # user-configurable additions.  Plural return shape preserved
        # so future re-addition of multi-root is a one-line change.
        r = web.get("/api/files/roots")
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert len(j["roots"]) == 1
        assert j["roots"][0]["path"] == str(picker_root.resolve())
        assert j["roots"][0]["label"] == "projects"
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


class TestSidebarPartialAndShim:
    """Every tab includes the persistent Projects sidebar partial,
    the supporting JS / CSS are reachable, and subscriber tabs (those
    that also load a file via the selection) include the banner DOM."""

    def test_projects_page_route_removed(self, web):
        # The standalone /projects tab was retired in favour of the
        # persistent sidebar.  Make sure the old route is gone so a
        # bookmark lands on a clean 404 rather than a half-rendered
        # leftover.
        r = web.get("/projects")
        assert r.status_code == 404

    def test_projects_sidebar_js_served(self, web, picker_root):
        r = web.get("/static/lib/projects-sidebar.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # Single-column + breadcrumb design contract.
        assert "openDir" in body
        assert "renderBreadcrumb" in body
        assert "molbuilder.current_dir" in body
        assert "molbuilder.current_file" in body
        # Dynamic top measurement (the v1 hardcoded 3rem caused
        # overlap with the header above the app-tabs nav).
        assert "measureSidebarTop" in body
        assert "offsetHeight" in body
        # Inquire-model public API (replaces the v2 tab-action buttons).
        assert "window.molbuilder.projects" in body
        assert "getCurrentDir" in body
        assert "getCurrentFile" in body
        assert "onChange" in body
        assert "readCurrentFile" in body
        assert "relativeToProjects" in body
        assert "refresh:" in body
        # The "Open in <Tab>" extension-mapping dict was retired -- the
        # sidebar no longer knows about tabs.
        assert "OPEN_TARGETS" not in body
        # File-manipulation: mkdir is the v1 ship.
        assert "submitMkdir" in body
        assert "/api/files/mkdir" in body
        # Cruft from prior pivots that shouldn't return.
        assert "sidebar_collapsed" not in body
        assert "molbuilderTabAutoLoad" not in body

    def test_projects_sidebar_css_served(self, web, picker_root):
        r = web.get("/static/lib/projects-sidebar.css")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # The key layout classes the partial relies on.
        assert ".projects-sidebar" in body
        assert ".ps-breadcrumb" in body
        assert ".ps-actions" in body
        # Body-padding shift for main content.
        assert "padding-left: var(--ps-w)" in body

    def test_projects_selection_shim_removed(self, web, picker_root):
        # The per-tab projects-selection shim was retired -- the sidebar
        # actions section took over (no more "Use this file" banner).
        r = web.get("/static/lib/projects-selection.js")
        assert r.status_code == 404

    @pytest.mark.parametrize("path", ["/", "/spectra", "/modify", "/watch"])
    def test_sidebar_included_in_every_tab(self, web, picker_root, path):
        r = web.get(path)
        assert r.status_code == 200, path
        body = r.get_data(as_text=True)
        # Sidebar partial markup is present.
        assert 'id="projects-sidebar"' in body, path
        assert 'id="ps-breadcrumb"' in body, path
        assert 'id="ps-list"' in body, path
        assert 'id="ps-actions"' in body, path
        # Sidebar JS + CSS included.
        assert "projects-sidebar.js" in body, path
        assert "projects-sidebar.css" in body, path

    @pytest.mark.parametrize("path", ["/spectra", "/modify", "/watch"])
    def test_subscriber_tabs_use_inquire_api(
        self, web, picker_root, path,
    ):
        # Each subscriber tab now pulls from window.molbuilder.projects
        # on its own user-triggered events instead of registering a
        # window.molbuilderTabAutoLoad auto-load shim.  Pin the new
        # contract + the absence of every retired surface.
        r = web.get(path)
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # Inquire API consumption: tab has the Load-from-selection btn
        # AND wires it through window.molbuilder.projects.
        assert 'id="load-from-selection-btn"' in body, path
        assert "window.molbuilder" in body, path
        assert ".projects" in body, path
        # Retired surfaces stay retired.
        assert "molbuilderTabAutoLoad" not in body, path
        assert "projects-selection.js" not in body, path
        assert 'id="projects-banner"' not in body, path

    def test_projects_nav_entry_removed(self, web):
        # The "Projects" app-tab entry was removed from _app_header.html
        # when we pivoted to the sidebar (otherwise users get a dead
        # tab link).  The sidebar's own <h2>Projects</h2> title
        # legitimately contains the word "Projects", so we assert on
        # the app-tab count + the absence of a Projects-href link.
        import re
        body = web.get("/").get_data(as_text=True)
        # Build / Modify / Spectra / Watch -- exactly four app-tab
        # *links*, no Projects entry.  The regex excludes the
        # container <nav class="app-tabs"> (note the trailing s).
        n_tabs = len(re.findall(r'class="app-tab(?: is-active)?"', body))
        assert n_tabs == 4, f"expected 4 app-tabs, found {n_tabs}"
        # No href="/projects" anywhere.
        assert 'href="/projects"' not in body


class TestNoLocalFileInputs:
    """After the sidebar pivot, the browser-local <input type=file>
    pickers were dropped from Spectra / Watch / Modify -- a script
    running on the server can't read a laptop file anyway."""

    @pytest.mark.parametrize("path,absent_id", [
        ("/spectra", 'id="xyz-file"'),
        ("/spectra", 'id="results-file"'),
        ("/watch",   'id="file-picker"'),
        ("/modify",  'id="file-picker"'),
    ])
    def test_file_input_not_emitted(self, web, picker_root, path, absent_id):
        r = web.get(path)
        body = r.get_data(as_text=True)
        assert absent_id not in body, (
            f"{absent_id} unexpectedly present in {path}; "
            f"the sidebar should be the only file-loading path."
        )


class TestFilesMkdir:
    """POST /api/files/mkdir creates a subdirectory inside an allowed
    root, validated against molbuilder.projects naming rules.

    Depth-aware validation:
      * directly under projects/   -> project name; ^[A-Za-z0-9_-]+$
      * under projects/<project>/  -> topic; must be in CANONICAL_TOPICS
      * deeper                     -> structure / ad-hoc subdir; same regex
    """

    def test_mkdir_creates_subdir_inside_root(self, web, picker_root):
        # picker_root is wired as projects/ for these tests.
        r = web.post(
            "/api/files/mkdir",
            json={"parent": str(picker_root), "name": "new_project"},
        )
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["path"] == str((picker_root / "new_project").resolve())
        assert (picker_root / "new_project").is_dir()

    def test_mkdir_rejects_bad_name_at_root_level(self, web, picker_root):
        # ^[A-Za-z0-9_-]+$ disallows spaces, dots, slashes.
        r = web.post(
            "/api/files/mkdir",
            json={"parent": str(picker_root), "name": "bad name"},
        )
        assert r.status_code == 400
        assert "outside [A-Za-z0-9_-]" in r.get_json()["error"]
        assert not (picker_root / "bad name").exists()

    def test_mkdir_rejects_non_canonical_topic_at_topic_depth(
        self, web, picker_root,
    ):
        # Set up projects/<project>/ then try to create a non-canonical
        # topic underneath.  The picker_root acts as projects/.
        (picker_root / "myproj").mkdir()
        r = web.post(
            "/api/files/mkdir",
            json={
                "parent": str(picker_root / "myproj"),
                "name": "Raman",   # not in CANONICAL_TOPICS
            },
        )
        assert r.status_code == 400
        body = r.get_json()
        assert "not one of the canonical six" in body["error"]
        assert not (picker_root / "myproj" / "Raman").exists()

    def test_mkdir_accepts_canonical_topic_at_topic_depth(
        self, web, picker_root,
    ):
        (picker_root / "myproj").mkdir()
        r = web.post(
            "/api/files/mkdir",
            json={
                "parent": str(picker_root / "myproj"),
                "name": "spectrum",   # in CANONICAL_TOPICS
            },
        )
        assert r.status_code == 200
        assert (picker_root / "myproj" / "spectrum").is_dir()

    def test_mkdir_409_when_already_exists(self, web, picker_root):
        (picker_root / "preexisting").mkdir()
        r = web.post(
            "/api/files/mkdir",
            json={"parent": str(picker_root), "name": "preexisting"},
        )
        assert r.status_code == 409
        assert "already exists" in r.get_json()["error"]

    def test_mkdir_400_for_missing_name(self, web, picker_root):
        r = web.post(
            "/api/files/mkdir", json={"parent": str(picker_root)},
        )
        assert r.status_code == 400
        assert "missing 'name'" in r.get_json()["error"]

    def test_mkdir_400_for_parent_outside_root(self, web, picker_root):
        # Reuses the same outside-root rejection as /api/files/list.
        r = web.post(
            "/api/files/mkdir",
            json={"parent": "/etc", "name": "evil"},
        )
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]
        assert not Path("/etc/evil").exists()  # paranoia

    def test_mkdir_400_for_dot_dot_in_parent(self, web, picker_root):
        r = web.post(
            "/api/files/mkdir",
            json={"parent": str(picker_root) + "/..",
                  "name": "anything"},
        )
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_mkdir_400_for_parent_not_a_directory(self, web, picker_root):
        # parent points at a regular file -> 400.
        r = web.post(
            "/api/files/mkdir",
            json={"parent": str(picker_root / "water.xyz"),
                  "name": "child"},
        )
        assert r.status_code == 400
        assert "not a directory" in r.get_json()["error"]


class TestProjectsCreate:
    """POST /api/projects/create bootstraps projects/<name>/ with every
    CANONICAL_TOPICS subdir.  Strict conflict: 409 if the name exists.
    Atomic: any subdir failure rolls back the whole project tree."""

    def test_create_project_bootstraps_full_skeleton(self, web, picker_root):
        r = web.post("/api/projects/create", json={"name": "myproj"})
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["path"] == str((picker_root / "myproj").resolve())
        # Every canonical subdir is created on disk.
        from molbuilder.projects import CANONICAL_TOPICS
        for topic in CANONICAL_TOPICS:
            assert (picker_root / "myproj" / topic).is_dir(), topic
        # Response carries the subdir list verbatim for the UI.
        assert j["subdirs"] == list(CANONICAL_TOPICS)

    def test_create_includes_structure_and_pseudopotential(
        self, web, picker_root,
    ):
        # Both new storage-dir entries land alongside the run-topic
        # dirs as part of the canonical skeleton.
        r = web.post("/api/projects/create", json={"name": "with_storage"})
        assert r.status_code == 200
        assert (picker_root / "with_storage" / "structure").is_dir()
        assert (picker_root / "with_storage" / "pseudopotential").is_dir()

    def test_create_includes_user_freeform_topic(self, web, picker_root):
        # 'user' lands at depth 1 alongside the other canonical topics.
        # Free-form: any subdir name (regex-valid) is accepted inside.
        r = web.post("/api/projects/create", json={"name": "with_user"})
        assert r.status_code == 200
        user_dir = picker_root / "with_user" / "user"
        assert user_dir.is_dir()
        # Verify it's reachable via /api/files/mkdir for an arbitrary
        # name (free-form at depth 2; "free_subdir" passes the regex
        # but is NOT in CANONICAL_TOPICS -- which would have rejected
        # it at depth 1).
        r2 = web.post(
            "/api/files/mkdir",
            json={"parent": str(user_dir), "name": "free_subdir"},
        )
        assert r2.status_code == 200
        assert (user_dir / "free_subdir").is_dir()

    def test_create_writes_readme_in_every_subdir(self, web, picker_root):
        # Each canonical subdir gets a small README.md describing its
        # purpose -- this is the "teaching" hint a new user sees when
        # navigating the tree.
        from molbuilder.projects import CANONICAL_TOPICS
        web.post("/api/projects/create", json={"name": "readme_proj"})
        proj = picker_root / "readme_proj"
        # Project-level README (mentions every canonical topic).
        root_readme = (proj / "README.md").read_text()
        for t in CANONICAL_TOPICS:
            assert t in root_readme, t
        # Per-subdir READMEs (the heading should mention the topic name).
        for t in CANONICAL_TOPICS:
            content = (proj / t / "README.md").read_text()
            assert content.startswith(f"# {t}/"), t

    def test_user_topic_is_canonical(self):
        from molbuilder.projects import CANONICAL_TOPICS
        assert "user" in CANONICAL_TOPICS

    def test_create_returns_409_on_name_conflict(self, web, picker_root):
        # First create succeeds.
        web.post("/api/projects/create", json={"name": "dup"})
        # Second create returns 409 with a clear message.
        r = web.post("/api/projects/create", json={"name": "dup"})
        assert r.status_code == 409
        body = r.get_json()
        assert body["ok"] is False
        assert "already exists" in body["error"]
        # The original project tree is untouched -- the 409 is detection-
        # only, no destructive side-effect.
        assert (picker_root / "dup" / "structure").is_dir()

    def test_create_409_when_project_dir_exists_from_hand(
        self, web, picker_root,
    ):
        # Same 409 path applies when the dir already exists outside
        # the /api/projects/create flow (e.g., user mkdir'd by hand).
        (picker_root / "handmade").mkdir()
        r = web.post("/api/projects/create", json={"name": "handmade"})
        assert r.status_code == 409

    def test_create_400_on_invalid_name(self, web, picker_root):
        # validate_name regex: ^[A-Za-z0-9_-]+$ -- reject spaces, dots.
        for bad in ["my project", "my.proj", "my/proj", "weird*name", ""]:
            r = web.post("/api/projects/create", json={"name": bad})
            assert r.status_code == 400, bad

    def test_create_400_when_name_missing(self, web, picker_root):
        r = web.post("/api/projects/create", json={})
        assert r.status_code == 400
        assert "missing 'name'" in r.get_json()["error"]


class TestSidebarCreateUI:
    """The foldable + New project / + New subdir sections live in the
    partial; the JS wires them to the backend."""

    def test_create_project_form_in_partial(self, web, picker_root):
        body = web.get("/spectra").get_data(as_text=True)
        # + New project section (foldable details + form + error slot)
        assert 'class="ps-create-section"' in body
        assert 'class="ps-create-summary">+ New project</summary>' in body
        assert 'id="ps-newproject-form"' in body
        assert 'id="ps-newproject-input"' in body
        assert 'id="ps-newproject-error"' in body
        assert 'id="ps-newproject-cancel"' in body
        # Subdir-list note (populated by JS at startup)
        assert 'id="ps-newproject-subdirs"' in body

    def test_create_project_uses_canonical_subdir_list_in_js(
        self, web, picker_root,
    ):
        # The JS hard-codes the CANONICAL_TOPICS_DISPLAY list to render
        # the note ("structure/, pseudopotential/, ...").  Pin it so a
        # future Python-side topic addition doesn't drift from the UI
        # without a code review noticing.
        r = web.get("/static/lib/projects-sidebar.js")
        body = r.get_data(as_text=True)
        for needle in ("structure", "pseudopotential", "optimization",
                       "frequency", "spectrum", "transport",
                       "single-point", "scan"):
            assert f'"{needle}"' in body, needle
        # And the POST handler.
        assert "/api/projects/create" in body
        assert "submitNewProject" in body


class TestSidebarMkdirUI:
    """The "+ New subdir" button + inline form is in the partial,
    not the JS."""

    def test_mkdir_form_in_sidebar_partial(self, web, picker_root):
        # Any tab that includes the sidebar partial carries the markup.
        # The form is now inside a <details class="ps-create-section">,
        # so visibility is HTML-controlled (no `hidden` attr).
        body = web.get("/spectra").get_data(as_text=True)
        assert 'id="ps-mkdir-form"' in body
        assert 'id="ps-mkdir-input"' in body
        assert 'id="ps-mkdir-error"' in body
        assert 'id="ps-mkdir-cancel"' in body
        # The + New subdir summary heading lives inside its details.
        assert '+ New subdir</summary>' in body

    def test_mkdir_section_depth_aware_visibility_in_js(
        self, web, picker_root,
    ):
        # Sidebar JS hides the + New subdir <details> when current_dir
        # is at the projects/ root (depth 0).  Pin the toggle logic so
        # a future refactor doesn't accidentally drop it.
        r = web.get("/static/lib/projects-sidebar.js")
        body = r.get_data(as_text=True)
        # The toggle is keyed on closest("details") + hidden = atRoot.
        assert 'closest("details")' in body
        assert "atRoot" in body
        assert "section.hidden" in body


class TestFileOperationStubs:
    """The upload / write / delete endpoints are intentionally 501
    stubs in v1.  The frontend renders the inline error from the
    standard ``{ok:false, error:...}`` shape; design captured in
    ``docs/protocols/selection.md`` for the eventual real impls."""

    def test_upload_returns_501_with_helpful_message(self, web):
        r = web.post("/api/files/upload")
        assert r.status_code == 501
        body = r.get_json()
        assert body["ok"] is False
        assert "not implemented" in body["error"]
        # The message points the user at the manual workaround.
        assert "scp" in body["error"] or "mv" in body["error"]

    def test_write_returns_501_with_helpful_message(self, web):
        r = web.post("/api/files/write")
        assert r.status_code == 501
        body = r.get_json()
        assert body["ok"] is False
        assert "not implemented" in body["error"]
        # The message acknowledges the preview-is-view-only context.
        assert "Preview" in body["error"] or "view-only" in body["error"]

    def test_delete_returns_501_with_helpful_message(self, web):
        r = web.delete("/api/files/delete")
        assert r.status_code == 501
        body = r.get_json()
        assert body["ok"] is False
        assert "not implemented" in body["error"]
        assert "shell" in body["error"] or "rm" in body["error"]


class TestSidebarStubsUI:
    """The stub features ship with their full UI surface so the design
    is reviewable.  Markup checks here; behaviour is exercised at the
    E2E layer (deferred Playwright suite)."""

    def test_upload_section_in_partial(self, web, picker_root):
        body = web.get("/spectra").get_data(as_text=True)
        assert 'id="ps-upload-form"' in body
        assert 'id="ps-upload-input"' in body
        assert 'id="ps-upload-error"' in body
        assert 'class="ps-upload-context"' in body
        # The summary heading is the user-facing label.
        assert '+ Upload file</summary>' in body

    def test_upload_section_depth_aware_visibility_in_js(self, web):
        # Same string-grep pattern as the mkdir-section test; the
        # toggle function name + closest("details") + atRoot all
        # appear in the JS source.  E2E confirmation deferred.
        body = web.get("/static/lib/projects-sidebar.js").get_data(as_text=True)
        assert "updateUploadContext" in body
        # The function reuses the same atRoot pattern.
        assert "elUploadForm" in body
        assert "elUploadContext" in body

    def test_preview_is_per_entry_not_sidebar_bottom_bar(
        self, web, picker_root,
    ):
        # v5.1: Preview moved from a bottom-bar button to a per-entry
        # hover button (alongside the delete ×).  The id="ps-preview-btn"
        # global button is gone; preview elements are now created
        # dynamically by renderList() and use the .ps-entry-preview class.
        body = web.get("/spectra").get_data(as_text=True)
        # No global Preview button in the partial.
        assert 'id="ps-preview-btn"' not in body
        # The dead .ps-actions-row + .ps-action-btn styles are gone too.
        assert 'class="ps-actions-row"' not in body

    def test_preview_per_entry_handler_in_js(self, web):
        # The per-entry button is built by renderList(); pin the
        # class name + the showPreview() call from the entry handler.
        body = web.get(
            "/static/lib/projects-sidebar.js",
        ).get_data(as_text=True)
        assert "ps-entry-preview" in body
        # The button reads "view" (matches the visual pattern -- short
        # text label like the × of delete).
        assert 'view.textContent = "view"' in body
        # File-only: directories don't get a Preview button.
        assert 'if (e.kind === "file")' in body

    def test_preview_per_entry_styles_match_delete_hover_pattern(self, web):
        # Both per-entry buttons inherit .ps-entry-action; the hover-
        # reveal idiom (opacity 0 default, 1 on .ps-entry:hover) is in
        # the shared rule.  Delete + Preview only differ in hover color.
        body = web.get(
            "/static/lib/projects-sidebar.css",
        ).get_data(as_text=True)
        assert ".ps-entry-action {" in body
        assert ".ps-entry:hover .ps-entry-action" in body
        assert ".ps-entry-preview:hover" in body
        assert ".ps-entry-delete:hover" in body

    def test_preview_modal_markup_full(self, web, picker_root):
        body = web.get("/spectra").get_data(as_text=True)
        # Modal scaffolding: backdrop, window, header (title + close),
        # body (pre for text), error slot, footer (Save + Close).
        assert 'id="ps-preview-modal"' in body
        assert 'class="ps-preview-backdrop"' in body
        assert 'id="ps-preview-title"' in body
        assert 'id="ps-preview-meta"' in body
        assert 'id="ps-preview-body"' in body
        assert 'id="ps-preview-error"' in body
        # Save is visible but disabled in v1 (write endpoint stubbed).
        assert 'id="ps-preview-save-btn"' in body
        between = body.split(
            'id="ps-preview-save-btn"', 1,
        )[1].split(">", 1)[0]
        assert "disabled" in between
        assert "coming soon" in between or "not implemented" in between

    def test_preview_modal_starts_hidden(self, web, picker_root):
        # The hidden attribute ensures it doesn't flash on first paint
        # before JS runs.
        body = web.get("/spectra").get_data(as_text=True)
        assert 'id="ps-preview-modal" class="ps-preview-modal" hidden' in body

    def test_preview_uses_existing_read_endpoint_in_js(self, web):
        # Preview is fully functional for view; the JS calls
        # window.molbuilder.projects.readCurrentFile() which wraps
        # /api/files/read.  Pin so a future refactor doesn't switch
        # to a hypothetical new endpoint without notice.
        body = web.get("/static/lib/projects-sidebar.js").get_data(as_text=True)
        assert "readCurrentFile" in body
        assert "showPreview" in body
        assert "openPreviewModal" in body
        assert "closePreviewModal" in body

    def test_delete_button_logic_in_js(self, web):
        # Per-entry × button is rendered for deletable entries.  The
        # eligibility check + confirm flow is in the JS; backend 501s.
        body = web.get("/static/lib/projects-sidebar.js").get_data(as_text=True)
        assert "_isDeletableEntry" in body
        assert "ps-entry-delete" in body
        assert "confirmAndDelete" in body
        assert "_UNDELETABLE_AT_DEPTH_1" in body

    def test_delete_button_has_hover_visibility_css(self, web):
        # The × shows only on hover so it doesn't clutter the list.
        # v5.1: the hover-reveal idiom moved to the shared
        # .ps-entry-action class (preview + delete inherit it); the
        # delete-specific rule only carries the destructive red hover.
        body = web.get(
            "/static/lib/projects-sidebar.css",
        ).get_data(as_text=True)
        assert ".ps-entry-delete" in body
        assert "opacity: 0" in body
        assert ".ps-entry:hover .ps-entry-action" in body
        assert ".ps-entry-delete:hover" in body


class TestRootsContract:
    """Single-root contract: Capabilities.file_picker_roots() returns
    exactly the projects/ entry.  file_picker.roots in molbuilder.json
    was removed; passing it is silently ignored (unknown sections are
    OK per the runtime_config contract)."""

    def test_capabilities_returns_only_projects_root(self):
        from molbuilder.diagnostics import Capabilities
        caps = Capabilities(runtime_config={})
        roots = caps.file_picker_roots()
        assert len(roots) == 1
        path, label = roots[0]
        assert label == "projects"
        assert str(path).endswith("/projects")

    def test_unknown_file_picker_section_ignored(self, tmp_path):
        # The file_picker section is no longer recognised; passing it
        # in molbuilder.json should NOT error (unknown-section graceful
        # ignore), but it also has no effect -- the picker still
        # returns just projects/.
        from molbuilder.runtime_config import read_config
        cfg_file = tmp_path / "molbuilder.json"
        cfg_file.write_text('{"file_picker": {"roots": ["~/scratch"]}}')
        cfg = read_config(cfg_file)
        # The section is dropped during _normalise (unknown sections
        # are silently ignored).
        assert "file_picker" not in cfg

    def test_get_file_picker_roots_removed_from_runtime_config(self):
        # The accessor that v1 added was dropped during the single-
        # root pivot.  Importing it should fail; this test pins the
        # removal so a future revert is caught.
        import molbuilder.runtime_config as rc
        assert not hasattr(rc, "get_file_picker_roots")
