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

    def test_projects_sidebar_entry_is_small_and_bootstraps(
        self, web, picker_root,
    ):
        # v5.3 (split): entry file is ~50 LOC and only does imports +
        # bootstrap.  Behaviour lives in projects/*.js modules.
        r = web.get("/static/lib/projects-sidebar.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # ES module imports for each sub-module.
        assert 'from "./projects/api.js"' in body
        assert 'from "./projects/state.js"' in body
        assert 'from "./projects/list.js"' in body
        assert 'from "./projects/forms.js"' in body
        assert 'from "./projects/preview.js"' in body
        # Mounts the public Inquire API on window.
        assert "window.molbuilder.projects = projects" in body
        # Bootstrap glue only -- behaviour has moved out.
        assert "openDir" in body          # called once for initial nav
        assert "initList" in body
        assert "initForms" in body
        assert "initPreview" in body
        # No more module-level state declarations -- that's state.js's job.
        assert "renderBreadcrumb" not in body
        assert "submitMkdir" not in body
        assert "_isDeletableEntry" not in body

    def test_projects_state_module_exposes_inquire_api(
        self, web, picker_root,
    ):
        r = web.get("/static/lib/projects/state.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # Inquire-API public methods.
        assert "getCurrentDir" in body
        assert "getCurrentFile" in body
        assert "onChange" in body
        assert "readCurrentFile" in body
        assert "relativeToProjects" in body
        assert "refresh" in body
        # writeFile primitive + saveToWorkspace convenience.
        assert "writeFile" in body
        assert "saveToWorkspace" in body
        # sessionStorage keys (the cross-tab contract).
        assert "molbuilder.current_dir" in body
        assert "molbuilder.current_file" in body
        # Retired surfaces stay retired.
        assert "OPEN_TARGETS" not in body
        assert "molbuilderTabAutoLoad" not in body
        assert "sidebar_collapsed" not in body
        assert "measureSidebarTop" not in body

    def test_projects_list_module_owns_rendering(self, web, picker_root):
        r = web.get("/static/lib/projects/list.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert "openDir" in body
        assert "_renderBreadcrumb" in body
        assert "_renderList" in body
        # Delete eligibility + per-entry buttons.
        assert "_isDeletableEntry" in body
        assert "_UNDELETABLE_AT_DEPTH_1" in body
        # Per-entry view + delete buttons (created in renderList).
        assert "ps-entry-preview" in body
        assert "ps-entry-delete" in body

    def test_projects_api_module_is_pure_http(self, web, picker_root):
        r = web.get("/static/lib/projects/api.js")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        for fn in ("apiRoots", "apiList", "apiStat", "apiRead",
                   "apiMkdir", "apiCreateProject", "apiUpload",
                   "apiDelete", "apiWrite"):
            assert "export async function " + fn in body, fn
        # No DOM references in this module.
        assert "document." not in body
        assert "window." not in body

    def test_projects_module_dependency_direction(self, web, picker_root):
        """Module deps form a DAG -- a circular import would still
        work in ES modules but causes init-order subtleties.  Pin
        the allowed direction:

            api.js     -> (nothing in projects/)
            state.js   -> api.js only
            preview.js -> state.js, api.js
            list.js    -> state.js, api.js, preview.js
            forms.js   -> state.js, api.js, list.js

        ``projects-sidebar.js`` (entry) imports all 5 modules and
        is the only file allowed to.
        """
        def imports_from_projects(body):
            import re
            return set(re.findall(
                r'from\s+"\.\/projects\/([a-z]+)\.js"|'
                r'from\s+"\.\/([a-z]+)\.js"',
                body,
            ))
        def flat(matches):
            return {a or b for a, b in matches}

        api    = flat(imports_from_projects(
            web.get("/static/lib/projects/api.js").get_data(as_text=True)
        ))
        state  = flat(imports_from_projects(
            web.get("/static/lib/projects/state.js").get_data(as_text=True)
        ))
        preview = flat(imports_from_projects(
            web.get("/static/lib/projects/preview.js").get_data(as_text=True)
        ))
        list_  = flat(imports_from_projects(
            web.get("/static/lib/projects/list.js").get_data(as_text=True)
        ))
        forms  = flat(imports_from_projects(
            web.get("/static/lib/projects/forms.js").get_data(as_text=True)
        ))

        # api is a leaf -- depends on nothing else in projects/.
        assert api == set(), f"api.js should be a leaf, imports {api}"

        # state depends only on api.
        assert state <= {"api"}, (
            f"state.js may import from api only, found {state}"
        )

        # preview depends on state + api.
        assert preview <= {"state", "api"}, (
            f"preview.js may import from state, api only, found {preview}"
        )

        # list depends on state, api, preview (but NOT forms).
        assert list_ <= {"state", "api", "preview"}, (
            f"list.js may import from state/api/preview only, found {list_}"
        )

        # forms is the top of the per-module stack (besides the entry).
        # It can depend on state, api, list, preview.
        assert forms <= {"state", "api", "list", "preview"}, (
            f"forms.js may import from state/api/list/preview only, "
            f"found {forms}"
        )

        # The crucial negative: state must NOT import from list, forms,
        # or preview (the cycle-breaking discipline).
        assert "list"    not in state, "state.js cannot import from list.js (cycle)"
        assert "forms"   not in state, "state.js cannot import from forms.js"
        assert "preview" not in state, "state.js cannot import from preview.js"

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

    @pytest.mark.parametrize("path", ["/", "/spectra", "/modify", "/watch"])
    def test_body_class_server_side_for_layout(
        self, web, picker_root, path,
    ):
        # `class="has-projects-sidebar"` must be on <body> at server
        # render -- if we waited for the type=module sidebar JS to
        # add the class, the first paint would happen with the
        # WIDER pre-sidebar geometry (no padding-left for the
        # sidebar's 18rem width), and Plotly + CSS-grid-auto-fit
        # layouts would init at the wrong size and look broken
        # until a browser resize fixed them.  This bit users at
        # least once -- pin it.
        body = web.get(path).get_data(as_text=True)
        assert 'class="has-projects-sidebar"' in body, path
        # And the JS should NOT be adding it (avoid double-toggle).
        js = web.get(
            "/static/lib/projects-sidebar.js",
        ).get_data(as_text=True)
        assert 'classList.add("has-projects-sidebar")' not in js

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
        # v5.3: form handlers moved to projects/forms.js + the HTTP
        # call to projects/api.js.  Tested across the two modules.
        forms = web.get(
            "/static/lib/projects/forms.js",
        ).get_data(as_text=True)
        for needle in ("structure", "pseudopotential", "optimization",
                       "frequency", "spectrum", "transport",
                       "single-point", "scan", "user"):
            assert f'"{needle}"' in forms, needle
        assert "_submitNewProject" in forms
        api = web.get("/static/lib/projects/api.js").get_data(as_text=True)
        assert "/api/projects/create" in api


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
        # v5.3.1: forms.js uses the centralised atProjectsRoot helper
        # from state.js (the prior _atRoot duplicate was removed).
        forms = web.get(
            "/static/lib/projects/forms.js",
        ).get_data(as_text=True)
        assert 'closest("details")' in forms
        assert "atProjectsRoot" in forms      # imports + uses the helper
        assert "section.hidden" in forms
        # Helper itself lives in state.js (single source of truth).
        state = web.get(
            "/static/lib/projects/state.js",
        ).get_data(as_text=True)
        assert "export function atProjectsRoot" in state


class TestFilesWrite:
    """POST /api/files/write covers two distinct workflows:

      1. Generate-and-save (Spectra/Build): no expected_mtime; strict
         no-overwrite by default (409 on conflict); the caller may
         opt in with overwrite=true.
      2. Edit-and-save (file-preview modal's Save -- still stubbed on
         the UI side): expected_mtime check (409 on mismatch).

    All cases gated by the same path-validation as the other
    endpoints + a depth >= 1 rule (no writing directly into the
    picker root)."""

    def test_write_happy_path_creates_file(self, web, picker_root):
        sub = picker_root / "myproj" / "topic_a"
        sub.mkdir(parents=True)
        target = str(sub / "out.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "hello world\n"})
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["path"] == target
        assert j["size"] > 0
        assert j["mtime"] > 0
        assert (sub / "out.txt").read_text() == "hello world\n"

    def test_write_409_on_existing_file_no_overwrite(
        self, web, picker_root,
    ):
        sub = picker_root / "myproj" / "topic_a"
        sub.mkdir(parents=True)
        (sub / "out.txt").write_text("original")
        target = str(sub / "out.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "replacement"})
        assert r.status_code == 409
        body = r.get_json()
        assert body["ok"] is False
        assert "already exists" in body["error"]
        # File is untouched -- conflict is detection-only.
        assert (sub / "out.txt").read_text() == "original"

    def test_write_with_overwrite_true_clobbers(self, web, picker_root):
        sub = picker_root / "myproj" / "topic_a"
        sub.mkdir(parents=True)
        (sub / "out.txt").write_text("original")
        target = str(sub / "out.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "new",
                           "overwrite": True})
        assert r.status_code == 200
        assert (sub / "out.txt").read_text() == "new"

    def test_write_mtime_mismatch_returns_409(self, web, picker_root):
        # Edit-and-save flow: write with a wrong expected_mtime.
        sub = picker_root / "myproj" / "topic_a"
        sub.mkdir(parents=True)
        (sub / "out.txt").write_text("original")
        target = str(sub / "out.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "edit",
                           "expected_mtime": 1.0})  # not the real mtime
        assert r.status_code == 409
        body = r.get_json()
        assert body["ok"] is False
        assert "modified since" in body["error"]
        assert "actual_mtime" in body
        # Original content preserved.
        assert (sub / "out.txt").read_text() == "original"

    def test_write_mtime_match_succeeds(self, web, picker_root):
        sub = picker_root / "myproj" / "topic_a"
        sub.mkdir(parents=True)
        f = sub / "out.txt"
        f.write_text("original")
        target = str(f)
        mtime = f.stat().st_mtime
        r = web.post("/api/files/write",
                     json={"path": target, "text": "edit",
                           "expected_mtime": mtime})
        assert r.status_code == 200
        assert f.read_text() == "edit"

    def test_write_at_root_depth_rejected(self, web, picker_root):
        # Cannot write directly into projects/ root; depth >= 1
        # required.  Keeps the root clean (only project dirs there).
        target = str(picker_root / "orphan.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "x"})
        assert r.status_code == 400
        assert "picker root" in r.get_json()["error"]
        assert not (picker_root / "orphan.txt").exists()

    def test_write_outside_root_rejected(self, web, picker_root):
        r = web.post("/api/files/write",
                     json={"path": "/etc/evil", "text": "x"})
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]

    def test_write_dot_dot_rejected(self, web, picker_root):
        r = web.post("/api/files/write",
                     json={"path": str(picker_root) + "/proj/../outside",
                           "text": "x"})
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_write_missing_parent_dir(self, web, picker_root):
        sub = picker_root / "myproj"
        sub.mkdir()
        target = str(sub / "no" / "such" / "dir" / "file.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "x"})
        assert r.status_code == 400
        assert "parent directory does not exist" in r.get_json()["error"]

    def test_write_rejects_non_string_text(self, web, picker_root):
        sub = picker_root / "myproj" / "topic_a"
        sub.mkdir(parents=True)
        r = web.post("/api/files/write",
                     json={"path": str(sub / "out.txt"), "text": 42})
        assert r.status_code == 400
        assert "string" in r.get_json()["error"]


class TestGenerateWritesToWorkspace:
    """Spectra + Build Generate buttons go through the unified
    ``window.molbuilder.projects.saveToWorkspace()`` API after a
    successful render.  Tests pin both layers: the API exists on the
    sidebar JS, and each tab calls it instead of duplicating fetch
    + refresh logic."""

    def test_save_to_workspace_api_exposed(self, web):
        # v5.3: saveToWorkspace lives in projects/state.js; the HTTP
        # call + overwrite gate live in projects/api.js.
        state = web.get(
            "/static/lib/projects/state.js",
        ).get_data(as_text=True)
        assert "saveToWorkspace" in state
        # Returns null on skip (no dir / at root) so callers fall
        # back silently.
        assert "return null" in state
        # Two-tier API: writeFile primitive too.
        assert "writeFile" in state

        api = web.get(
            "/static/lib/projects/api.js",
        ).get_data(as_text=True)
        assert "/api/files/write" in api
        # Strict no-overwrite gate (only sets body.overwrite when
        # explicitly opted in).
        assert "if (opts.overwrite) body.overwrite = true" in api

    def test_spectra_viewer_uses_save_to_workspace(self, web):
        body = web.get(
            "/static/spectra/viewer.js",
        ).get_data(as_text=True)
        # Tab calls the unified API; no direct fetch to /api/files/write.
        assert "proj.saveToWorkspace" in body
        assert "/api/files/write" not in body
        assert ".spectra.py" in body

    def test_build_viewer_uses_save_to_workspace(self, web):
        body = web.get("/static/viewer.js").get_data(as_text=True)
        # Tab calls the unified API.
        assert "proj.saveToWorkspace" in body
        assert "/api/files/write" not in body
        # Both .fdf (FDF generate) and .py (PySCF generate) paths.
        assert '".fdf"' in body
        assert '".py"' in body


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

    # (test_write_returns_501_with_helpful_message retired in v5.2:
    #  POST /api/files/write is now functional -- see TestFilesWrite.
    #  The preview modal's Save button is still UI-disabled
    #  ("coming soon") but the endpoint itself is live for the
    #  Generate-and-save flow.)

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
        # v5.3: lives in projects/forms.js.
        body = web.get(
            "/static/lib/projects/forms.js",
        ).get_data(as_text=True)
        assert "_updateUploadContext" in body
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
        # v5.3: per-entry button is built by renderList() in
        # projects/list.js.
        body = web.get(
            "/static/lib/projects/list.js",
        ).get_data(as_text=True)
        assert "ps-entry-preview" in body
        # The button reads "view" (matches × of delete).
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
        # v5.3: showPreview / openPreviewModal / closePreviewModal
        # live in projects/preview.js; the read backing is in
        # projects/state.js (readCurrentFile wraps /api/files/read).
        preview = web.get(
            "/static/lib/projects/preview.js",
        ).get_data(as_text=True)
        assert "showPreview" in preview
        assert "openPreviewModal" in preview
        assert "closePreviewModal" in preview
        # Preview calls into state for the actual read.
        assert "readCurrentFile" in preview
        state = web.get(
            "/static/lib/projects/state.js",
        ).get_data(as_text=True)
        assert "readCurrentFile" in state
        api = web.get(
            "/static/lib/projects/api.js",
        ).get_data(as_text=True)
        assert "/api/files/read" in api

    def test_delete_button_logic_in_js(self, web):
        # v5.3: eligibility check + confirm flow in projects/list.js;
        # HTTP call in projects/api.js (backend 501 stub).
        body = web.get(
            "/static/lib/projects/list.js",
        ).get_data(as_text=True)
        assert "_isDeletableEntry" in body
        assert "ps-entry-delete" in body
        assert "_confirmAndDelete" in body
        assert "_UNDELETABLE_AT_DEPTH_1" in body
        api = web.get(
            "/static/lib/projects/api.js",
        ).get_data(as_text=True)
        assert "apiDelete" in api
        assert "/api/files/delete" in api

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
