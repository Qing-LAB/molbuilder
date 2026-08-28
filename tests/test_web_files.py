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

Backend contract:  docs/web/web-api.md  §  /api/files/*
"""
from __future__ import annotations

import os
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
    app = create_app(config={})
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
#  /api/files/read_range  (task #119, 2026-06-02)                       #
#                                                                       #
#  Paginated read for the source inspector's virtual-scroll viewer.     #
# --------------------------------------------------------------------- #


class TestFilesReadRange:
    """The range-read endpoint underpins the source inspector's
    arbitrarily-large-text-file viewer.  These tests pin the byte-
    range semantics, the negative-offset "from end" form, UTF-8
    boundary trimming, ``eof`` marker, and the error paths."""

    def test_read_range_default_returns_start_of_file(
            self, web, picker_root):
        """No offset / max_bytes -> 256 KB from offset 0.  For the
        water.xyz fixture (35 bytes) that's the whole file + eof
        true."""
        r = web.get(
            f"/api/files/read_range?path={picker_root}/water.xyz")
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["offset"] == 0
        assert j["length"] == j["file_size"]
        assert j["eof"] is True
        assert j["text"].startswith("3\nwater")

    def test_read_range_explicit_offset_and_max_bytes(
            self, web, picker_root):
        """Caller-specified offset returns exactly those bytes."""
        big = picker_root / "big.log"
        big.write_text("".join(f"line {i:04d}\n" for i in range(200)))
        # Each line is 10 bytes; offset=100 starts mid-line-10.
        r = web.get(
            f"/api/files/read_range?path={big}&offset=100&max_bytes=80")
        j = r.get_json()
        assert r.status_code == 200
        assert j["offset"] == 100
        assert j["length"] == 80
        # The returned text starts at byte 100 which is the start of
        # line 10 ("line 0010\n" starts at offset 100).
        assert j["text"].startswith("line 0010")
        assert j["eof"] is False

    def test_read_range_eof_true_when_chunk_reaches_end(
            self, web, picker_root):
        small = picker_root / "small.log"
        small.write_text("hello world\n")
        # Request more than file size -> get the whole file, eof.
        r = web.get(
            f"/api/files/read_range?path={small}&max_bytes=1000")
        j = r.get_json()
        assert r.status_code == 200
        assert j["eof"] is True
        assert j["text"] == "hello world\n"

    def test_read_range_negative_offset_reads_tail(
            self, web, picker_root):
        """``offset=-N`` returns the last N bytes (tail).  Critical
        UX for "show me the END of this 10 MB log without paging
        through it first"."""
        big = picker_root / "tail.log"
        big.write_text("A" * 1000 + "B" * 500)
        r = web.get(
            f"/api/files/read_range?path={big}&offset=-500")
        j = r.get_json()
        assert r.status_code == 200
        assert j["offset"] == 1000
        assert j["text"] == "B" * 500
        assert j["eof"] is True

    def test_read_range_negative_offset_clamped_to_zero(
            self, web, picker_root):
        """``offset=-99999`` on a 12-byte file becomes offset 0,
        not an error (the caller asked for "more tail than exists"
        which should give them the whole file)."""
        small = picker_root / "tiny.log"
        small.write_text("hello world\n")
        r = web.get(
            f"/api/files/read_range?path={small}&offset=-99999")
        j = r.get_json()
        assert r.status_code == 200
        assert j["offset"] == 0
        assert j["text"] == "hello world\n"

    def test_read_range_offset_past_end_returns_400(
            self, web, picker_root):
        small = picker_root / "short.log"
        small.write_text("12345")
        r = web.get(
            f"/api/files/read_range?path={small}&offset=999")
        assert r.status_code == 400
        body = r.get_json()
        assert "exceeds file size" in body["error"]

    def test_read_range_offset_at_eof_returns_empty_chunk(
            self, web, picker_root):
        """``offset == file_size`` is the canonical "I'm at the end"
        request -- returns empty text + eof:true rather than 400,
        so a client paginating doesn't have to special-case the
        terminal request."""
        small = picker_root / "edge.log"
        small.write_text("abc")
        r = web.get(
            f"/api/files/read_range?path={small}&offset=3")
        j = r.get_json()
        assert r.status_code == 200
        assert j["offset"] == 3
        assert j["text"] == ""
        assert j["length"] == 0
        assert j["eof"] is True

    def test_read_range_invalid_offset_returns_400(
            self, web, picker_root):
        r = web.get(
            f"/api/files/read_range?path={picker_root}/water.xyz"
            f"&offset=not_an_int")
        assert r.status_code == 400

    def test_read_range_invalid_max_bytes_returns_400(
            self, web, picker_root):
        r = web.get(
            f"/api/files/read_range?path={picker_root}/water.xyz"
            f"&max_bytes=zero")
        assert r.status_code == 400

    def test_read_range_max_bytes_above_ceiling_returns_400(
            self, web, picker_root):
        r = web.get(
            f"/api/files/read_range?path={picker_root}/water.xyz"
            f"&max_bytes=99999999999")
        assert r.status_code == 400

    def test_read_range_missing_file_404(self, web, picker_root):
        r = web.get(
            f"/api/files/read_range?path={picker_root}/no-such.log")
        assert r.status_code == 404

    def test_read_range_directory_returns_400(self, web, picker_root):
        d = picker_root / "subdir"
        d.mkdir(exist_ok=True)
        r = web.get(f"/api/files/read_range?path={d}")
        assert r.status_code == 400

    def test_read_range_utf8_boundary_trim(self, web, picker_root):
        """A byte range that lands mid-codepoint MUST not return
        invalid UTF-8.  Construct a file where byte N is the second
        byte of a 2-byte ``é`` (0xC3 0xA9): a request for the first
        N bytes must trim the incomplete leading byte instead of
        returning a 400 or garbled text."""
        # "abcé" -> "abc" (3 bytes) + "é" (2 bytes) = 5 bytes total.
        path = picker_root / "utf8.log"
        path.write_bytes(b"abc\xc3\xa9")
        # max_bytes=4 lands in the MIDDLE of the é codepoint (byte 4
        # is 0xC3, the first byte of é; the second byte would be at
        # position 5).
        r = web.get(
            f"/api/files/read_range?path={path}&max_bytes=4")
        j = r.get_json()
        assert r.status_code == 200
        # The incomplete trailing 0xC3 should have been trimmed.
        assert j["text"] == "abc"
        assert j["length"] == 3
        # eof is False because we trimmed 1 byte off the file's true
        # end (file is 5 bytes; we returned 3).
        assert j["eof"] is False

    def test_read_range_actual_binary_data_returns_400(
            self, web, picker_root):
        """A file region that genuinely isn't UTF-8 (not just a
        truncated codepoint at the edge) MUST return 400 with a
        clear message -- ``read_range`` is text-only like ``read``."""
        bad = picker_root / "binary.bin"
        bad.write_bytes(b"\xff\xfe\xfd\xfc")
        r = web.get(f"/api/files/read_range?path={bad}&max_bytes=4")
        assert r.status_code == 400
        assert "UTF-8" in r.get_json()["error"]

    def test_read_range_file_size_unchanged_across_calls(
            self, web, picker_root):
        """Multiple range reads on the same file must report the
        SAME ``file_size`` -- the client uses it to drive the
        scrollbar / progress indicator."""
        big = picker_root / "stable.log"
        big.write_text("line\n" * 100)
        r1 = web.get(
            f"/api/files/read_range?path={big}&offset=0&max_bytes=50")
        r2 = web.get(
            f"/api/files/read_range?path={big}&offset=50&max_bytes=50")
        assert r1.get_json()["file_size"] == r2.get_json()["file_size"]
        assert r1.get_json()["file_size"] == 500


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

    def test_projects_module_dependency_direction(self, web, picker_root):
        """Module deps form a DAG -- a circular import would still
        work in ES modules but causes init-order subtleties.  Pin
        the allowed direction:

            api.js          -> (nothing in projects/)
            state.js        -> api.js only
            preview.js      -> state.js, api.js
            dialogs.js      -> state.js, api.js
            list.js         -> state.js, api.js, preview.js, dialogs.js
            mutation-bar.js -> state.js, api.js, list.js, dialogs.js

        ``projects-sidebar.js`` (entry) imports the modules and is
        the only file allowed to.

        2026-06-12: ``forms.js`` was renamed to ``mutation-bar.js``
        after the v2 buttons-not-inline-forms refactor.
        """
        def imports_from_projects(body):
            import re
            # Match ./projects/<name>.js or ./<name>.js (hyphen + dot OK).
            return set(re.findall(
                r'from\s+"\.\/projects\/([a-z][a-z0-9_-]*)\.js"|'
                r'from\s+"\.\/([a-z][a-z0-9_-]*)\.js"',
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
        dialogs = flat(imports_from_projects(
            web.get("/static/lib/projects/dialogs.js").get_data(as_text=True)
        ))
        list_  = flat(imports_from_projects(
            web.get("/static/lib/projects/list.js").get_data(as_text=True)
        ))
        mutation_bar = flat(imports_from_projects(
            web.get("/static/lib/projects/mutation-bar.js").get_data(as_text=True)
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

        # dialogs (2026-06-12) is a leaf-ish module: presents modal
        # <dialog>s + handles user input.  May read from api (for the
        # tree-picker's directory listing) + state (for the projects
        # root anchor); never the other way.
        assert dialogs <= {"state", "api"}, (
            f"dialogs.js may import from state, api only, found {dialogs}"
        )

        # list depends on state, api, preview, dialogs (but NOT
        # mutation-bar).
        assert list_ <= {"state", "api", "preview", "dialogs"}, (
            f"list.js may import from state/api/preview/dialogs only, "
            f"found {list_}"
        )

        # mutation-bar (renamed from forms 2026-06-12) is the top of
        # the per-module stack (besides the entry).  Can depend on
        # state, api, list, preview, dialogs.
        assert mutation_bar <= {"state", "api", "list", "preview", "dialogs"}, (
            f"mutation-bar.js may import from state/api/list/preview/"
            f"dialogs only, found {mutation_bar}"
        )

        # The crucial negative: state must NOT import from any
        # downstream module (the cycle-breaking discipline).
        assert "list"         not in state, "state.js cannot import from list.js (cycle)"
        assert "mutation-bar" not in state, "state.js cannot import from mutation-bar.js"
        assert "preview"      not in state, "state.js cannot import from preview.js"
        assert "dialogs"      not in state, "state.js cannot import from dialogs.js"

    def test_projects_selection_shim_removed(self, web, picker_root):
        # The per-tab projects-selection shim was retired -- the sidebar
        # actions section took over (no more "Use this file" banner).
        r = web.get("/static/lib/projects-selection.js")
        assert r.status_code == 404

    @pytest.mark.parametrize("path", [
        "/molbuilder", "/structure-optimization",
        "/spectrum-calculation", "/transport-calculation",
        "/results"])
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

    @pytest.mark.parametrize("path", [
        "/molbuilder", "/structure-optimization",
        "/spectrum-calculation", "/transport-calculation",
        "/results"])
    def test_sidebar_layout_opt_in_is_server_side(
        self, web, picker_root, path,
    ):
        # The app-shell layout opt-in -- ``<body data-sidebars="projects">``
        # -- must be in the SERVER-rendered markup, not added later by the
        # type=module sidebar JS.  page-shell.css keys the whole flex-column
        # shell (nav + [sidebar-rail | content] row) off
        # ``body[data-sidebars]``; if the attribute arrived via JS the first
        # paint would use the pre-shell geometry and layout-sensitive widgets
        # (Plotly, 3Dmol) would init at the wrong size and look broken until a
        # browser resize fixed them.  This bit users at least once under the
        # since-retired ``has-projects-sidebar`` + padding-left shim -- pin the
        # server-side opt-in so the regression can't come back.
        body = web.get(path).get_data(as_text=True)
        assert 'data-sidebars="projects"' in body, path
        # The JS must NOT set the attribute (that would race the first paint);
        # it only toggles the collapsed / mobile-drawer *body classes*.
        js = web.get(
            "/static/lib/projects/projects-sidebar.js",
        ).get_data(as_text=True)
        assert 'setAttribute("data-sidebars"' not in js
        assert "setAttribute('data-sidebars'" not in js
        assert "dataset.sidebars" not in js

    @pytest.mark.parametrize("path", ["/molbuilder"])
    def test_subscriber_tabs_use_inquire_api(
        self, web, picker_root, path,
    ):
        # /molbuilder is the canonical "subscriber tab": it reacts
        # to the Projects-sidebar selection by auto-loading the
        # picked XYZ into the viewer + selection panel.  The wiring
        # lives in modify/selection-bootstrap.js -- the bootstrap
        # subscribes to ``window.molbuilder.projects.onChange`` and
        # forwards changes to the selection store, which loads the
        # file.
        #
        # The legacy "Load from current selection" button (page.js)
        # was retired 2026-05-20 -- the auto-load via the store
        # made it redundant.
        #
        # /spectra is generate-only (no subscriber); /results
        # auto-mounts via the registry dispatch.  /modify is the
        # only remaining subscriber tab; parametrize keeps the seam
        # open for a future tab that adopts the same affordance.
        r = web.get(path)
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # Wires through window.molbuilder.projects (Inquire API).
        assert "window.molbuilder" in body, path
        assert "selection-bootstrap.js" in body, path
        # Retired surfaces stay retired.
        assert 'id="load-from-selection-btn"' not in body, path
        assert "molbuilderTabAutoLoad" not in body, path
        assert "projects-selection.js" not in body, path
        assert 'id="projects-banner"' not in body, path

    def test_projects_nav_entry_removed(self, web):
        # The "Projects" app-tab entry was removed from _app_header.html
        # when we pivoted to the sidebar (otherwise users get a dead
        # tab link).  The sidebar's own <h2>Projects</h2> title
        # legitimately contains the word "Projects", so the actual
        # invariants are: (a) no /projects href anywhere, and (b)
        # every visible app-tab link points at a route we actually
        # serve.  Counting tabs would make this test break every
        # time we add or remove a tab, which is the wrong sensitivity.
        import re
        r = web.get("/structure-optimization")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # No href="/projects" anywhere -- the sidebar replaced the tab.
        assert 'href="/projects"' not in body
        # Each app-tab link points at one of the served routes.  Pull
        # every href from the app-tab class; assert every value is in
        # the served-routes set.  SERVED is derived from
        # ``molbuilder.web.tabs.TABS`` (the canonical tab order) so
        # adding or reordering a tab is a one-place change in
        # ``tabs.py``; no test edit needed.
        from molbuilder.web.tabs import TABS
        SERVED = {t["path"] for t in TABS}
        hrefs = re.findall(
            r'<a[^>]*href="([^"]+)"[^>]*class="app-tab(?: is-active)?"',
            body,
        )
        assert hrefs, "no app-tab links found"
        for h in hrefs:
            assert h in SERVED, (
                f"app-tab link {h!r} points at an unserved route; "
                f"served routes: {sorted(SERVED)}"
            )


# TestNoLocalFileInputs retired 2026-06-10: the class parametrized
# over /spectra (now /spectrum-calculation, but routed via tabs.py)
# and /modify (renamed to /molbuilder in Phase B.5).  After B.5
# the legacy paths /spectra and /modify return 404 by design — the
# tests' ``assert <id> not in body`` checks were silently passing
# against the Flask 404 error page, pinning nothing.  The sidebar-
# is-the-only-file-loader contract still holds at the page level
# (no tab currently emits an <input type=file>); a future tab that
# regressed would be caught by the per-tab Playwright assertions
# in test_molbuilder_e2e / test_build_e2e instead.


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
    """2026-06-12 (v2): three SEPARATE buttons (New project / New
    folder / Upload) in the sidebar header.  Replaces the earlier
    v1 single "+" dropdown which hid the actions behind an extra
    click.  Each button opens its modal dialog directly.  See
    web/projects.md § Mutation UX."""

    def test_create_bar_in_partial(self, web, picker_root):
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        assert 'class="ps-create-bar"' in body

    def test_three_action_buttons_visible(self, web, picker_root):
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        # Three distinct buttons with stable ids for the JS to wire.
        assert 'id="ps-create-project-btn"' in body
        assert 'id="ps-create-folder-btn"' in body
        assert 'id="ps-create-upload-btn"' in body
        # data-action attributes (used by tests + accessibility tools).
        assert 'data-action="new-project"' in body
        assert 'data-action="new-folder"' in body
        assert 'data-action="upload"' in body
        # User-facing labels (verify the wording renders).
        assert "New project" in body
        assert "New folder" in body
        assert "Upload" in body


class TestSidebarMkdirUI:
    """Retired class kept as a marker so future readers can find
    the 2026-06-12 retirement history.  The mkdir form's role
    moved to the + dropdown menu — see TestSidebarCreateUI."""
    pass

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

class TestFileOperationStubs:
    """All three previously-stubbed endpoints (upload, write, delete)
    are now functional.  See TestFilesUpload + TestFilesWrite +
    TestFilesDelete for the real-behaviour tests."""

    # (test_upload_returns_501 / test_write_returns_501 /
    #  test_delete_returns_501 all retired in v5.4: every formerly
    #  stub endpoint is live now.  This class is kept as a marker
    #  so future readers can find the retirement history; remove
    #  when the docstring no longer needs to explain it.)
    pass


# --------------------------------------------------------------------- #
#  Sidecar-pairing helpers                                              #
# --------------------------------------------------------------------- #


def _seed_paired(picker_root: Path, dirname: str = "", stem: str = "water",
                 ext: str = ".xyz") -> tuple[Path, Path]:
    """Drop a structure file + paired .molstruct.json on disk.  Returns
    (structure_path, sidecar_path).  Used by the rename / move / copy
    sidecar-pairing tests."""
    import hashlib
    import json
    parent = picker_root / dirname if dirname else picker_root
    parent.mkdir(parents=True, exist_ok=True)
    struct = parent / f"{stem}{ext}"
    xyz_text = (
        "3\nwater\n"
        "O 0 0 0\n" "H 0.96 0 0\n" "H -0.24 0.93 0\n"
    )
    struct.write_text(xyz_text)
    sidecar = parent / f"{stem}.molstruct.json"
    sidecar.write_text(json.dumps({
        "schema_version": 7,
        "n_atoms_total":  3,
        "structure_hash": hashlib.sha256(xyz_text.encode()).hexdigest(),
        "regions":        {"L-electrode": [0]},
        "frozen_atoms":   [],
        "selection_rules": {},
    }))
    return struct, sidecar


# --------------------------------------------------------------------- #
#  DELETE /api/files/delete  (sidecar pairing)                          #
# --------------------------------------------------------------------- #


class TestFilesDeleteSidecarPairing:
    """2026-07: deleting a .xyz/.pdb file MUST also remove its paired
    .molstruct.json -- the mirror of the rename/move/copy pairing -- else
    the sidecar orphans (labels/cell of a file that no longer exists)."""

    def _delete(self, web, path: Path, **body):
        return web.delete(
            "/api/files/delete",
            json={"path": str(path), **body},
        )

    def test_xyz_delete_removes_sidecar(self, web, picker_root):
        struct, sidecar = _seed_paired(picker_root, stem="water")
        assert struct.exists() and sidecar.exists()
        r = self._delete(web, struct)
        assert r.status_code == 200, r.get_data(as_text=True)
        j = r.get_json()
        assert j["ok"] is True
        assert not struct.exists()
        assert not sidecar.exists()          # the fix: no orphaned sidecar
        assert j["sidecar_removed"] == str(sidecar)

    def test_pdb_delete_removes_sidecar(self, web, picker_root):
        struct, sidecar = _seed_paired(picker_root, stem="prot", ext=".pdb")
        r = self._delete(web, struct)
        assert r.status_code == 200
        assert not struct.exists()
        assert not sidecar.exists()

    def test_delete_without_sidecar_is_fine(self, web, picker_root):
        struct = picker_root / "lonely.xyz"
        struct.write_text("1\nx\nH 0 0 0\n")
        r = self._delete(web, struct)
        assert r.status_code == 200
        assert r.get_json()["sidecar_removed"] is None
        assert not struct.exists()

    def test_deleting_the_sidecar_directly_is_single_file(self, web, picker_root):
        # Deleting the .molstruct.json itself leaves the .xyz untouched (a
        # single-file op, matching rename's "sidecar renamed directly" rule).
        struct, sidecar = _seed_paired(picker_root, stem="water")
        r = self._delete(web, sidecar)
        assert r.status_code == 200
        assert r.get_json()["sidecar_removed"] is None
        assert not sidecar.exists()
        assert struct.exists()


# --------------------------------------------------------------------- #
#  POST /api/files/rename  (sidecar pairing)                            #
# --------------------------------------------------------------------- #


class TestFilesRenameSidecarPairing:
    """2026-06-12: rename of a .xyz/.pdb file MUST move its paired
    .molstruct.json sidecar to match the new stem -- otherwise the
    sidecar orphans (load can't find it; user's labels silently
    disappear).  See web/projects.md § Rename + the
    file-tree-ops contract there."""

    def _rename(self, web, path: Path, new_name: str):
        return web.post(
            "/api/files/rename",
            json={"path": str(path), "new_name": new_name},
        )

    def test_xyz_rename_takes_sidecar(self, web, picker_root):
        struct, sidecar = _seed_paired(picker_root, stem="water")
        r = self._rename(web, struct, "bridge.xyz")
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json()["ok"] is True
        # Source pair gone; destination pair exists with same payload.
        assert not struct.exists()
        assert not sidecar.exists()
        new_struct  = picker_root / "bridge.xyz"
        new_sidecar = picker_root / "bridge.molstruct.json"
        assert new_struct.exists()
        assert new_sidecar.exists()
        import json
        assert json.loads(new_sidecar.read_text())["n_atoms_total"] == 3

    def test_pdb_rename_takes_sidecar(self, web, picker_root):
        struct, sidecar = _seed_paired(picker_root, stem="prot", ext=".pdb")
        r = self._rename(web, struct, "protein.pdb")
        assert r.status_code == 200
        assert not struct.exists()
        assert not sidecar.exists()
        assert (picker_root / "protein.pdb").exists()
        assert (picker_root / "protein.molstruct.json").exists()

    def test_rename_without_sidecar_still_works(self, web, picker_root):
        """A .xyz with NO paired sidecar renames cleanly (no spurious
        404 from the sidecar branch)."""
        struct = picker_root / "lone.xyz"
        struct.write_text("1\nx\nH 0 0 0\n")
        assert not (picker_root / "lone.molstruct.json").exists()
        r = self._rename(web, struct, "renamed.xyz")
        assert r.status_code == 200
        assert (picker_root / "renamed.xyz").exists()

    def test_rename_refuses_when_dst_sidecar_exists(
            self, web, picker_root):
        """If a sidecar already lives at the destination stem, the
        rename refuses BEFORE touching either file.  Without this
        guard, the structure rename would succeed and then the
        sidecar rename would fail mid-way."""
        struct, _ = _seed_paired(picker_root, stem="water")
        # Pre-existing sidecar at the destination stem (no matching
        # structure yet -- a stale orphan).
        (picker_root / "bridge.molstruct.json").write_text("{}")
        r = self._rename(web, struct, "bridge.xyz")
        assert r.status_code == 409
        assert "sidecar already exists" in r.get_json()["error"]
        # Source pair untouched.
        assert struct.exists()
        assert (picker_root / "water.molstruct.json").exists()

    def test_rename_json_sidecar_directly_no_pairing(
            self, web, picker_root):
        """Renaming a .molstruct.json directly is a single-file
        op (pairing only triggers on structure files).  The user
        is on their own for sidecar orphans."""
        _, sidecar = _seed_paired(picker_root, stem="water")
        r = self._rename(web, sidecar, "other.molstruct.json")
        assert r.status_code == 200
        assert not sidecar.exists()
        assert (picker_root / "other.molstruct.json").exists()
        # Structure file untouched (correctly orphaned by the
        # direct rename -- user's choice).
        assert (picker_root / "water.xyz").exists()


# --------------------------------------------------------------------- #
#  POST /api/files/move                                                 #
# --------------------------------------------------------------------- #


class TestFilesMove:
    """Move a file from one allowed-root directory to another.  Same
    sidecar-pairing + atomic-no-overwrite contract as rename."""

    def _move(self, web, path: Path, dest_dir: Path, new_name=None):
        body = {"path": str(path), "dest_dir": str(dest_dir)}
        if new_name is not None:
            body["new_name"] = new_name
        return web.post("/api/files/move", json=body)

    def test_move_file_to_new_dir(self, web, picker_root):
        src = picker_root / "config.json"   # seeded by fixture
        dst_dir = picker_root / "spectrum"  # seeded by fixture
        r = self._move(web, src, dst_dir)
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json()["ok"] is True
        assert not src.exists()
        assert (dst_dir / "config.json").exists()

    def test_move_with_rename(self, web, picker_root):
        src = picker_root / "config.json"
        dst_dir = picker_root / "spectrum"
        r = self._move(web, src, dst_dir, new_name="renamed.json")
        assert r.status_code == 200
        assert (dst_dir / "renamed.json").exists()

    def test_move_xyz_takes_sidecar(self, web, picker_root):
        struct, sidecar = _seed_paired(picker_root, stem="water")
        dst_dir = picker_root / "structures"
        dst_dir.mkdir()
        r = self._move(web, struct, dst_dir)
        assert r.status_code == 200
        assert not struct.exists()
        assert not sidecar.exists()
        assert (dst_dir / "water.xyz").exists()
        assert (dst_dir / "water.molstruct.json").exists()

    def test_move_xyz_with_rename_takes_sidecar(self, web, picker_root):
        """Move-and-rename pairs the sidecar to the NEW stem at the
        new directory."""
        struct, sidecar = _seed_paired(picker_root, stem="water")
        dst_dir = picker_root / "structures"
        dst_dir.mkdir()
        r = self._move(web, struct, dst_dir, new_name="renamed.xyz")
        assert r.status_code == 200
        assert (dst_dir / "renamed.xyz").exists()
        assert (dst_dir / "renamed.molstruct.json").exists()
        # Sidecar's original-stem path no longer exists in either dir.
        assert not sidecar.exists()
        assert not (dst_dir / "water.molstruct.json").exists()

    def test_move_refuses_directory(self, web, picker_root):
        src_dir = picker_root / "spectrum"
        other = picker_root / "other"
        other.mkdir()
        r = self._move(web, src_dir, other)
        assert r.status_code == 400
        assert "directories" in r.get_json()["error"]

    def test_move_refuses_when_dest_missing(self, web, picker_root):
        src = picker_root / "config.json"
        r = self._move(web, src, picker_root / "does-not-exist")
        assert r.status_code in (400, 404)

    def test_move_refuses_overwrite(self, web, picker_root):
        src = picker_root / "config.json"
        dst_dir = picker_root / "spectrum"
        # Pre-existing file at dst with same name.
        (dst_dir / "config.json").write_text("existing\n")
        r = self._move(web, src, dst_dir)
        assert r.status_code == 409

    def test_move_refuses_canonical_topic_dir(self, web, picker_root):
        """Moving a canonical-topic dir would orphan the project
        layout — same protection as rename + delete."""
        proj = picker_root / "proj"
        proj.mkdir()
        (proj / "spectrum").mkdir()
        other_proj = picker_root / "other"
        other_proj.mkdir()
        r = self._move(web, proj / "spectrum", other_proj)
        # Either the canonical-topic guard catches it (400) or the
        # directory-refusal does (400).  Both are correct rejections.
        assert r.status_code == 400

    def test_move_sidecar_failure_rolls_back_structure(
            self, web, picker_root, monkeypatch):
        """2026-06-12 audit follow-up: when the structure leg of a
        sidecar-paired move succeeds but the sidecar leg fails, the
        backend must roll the structure back to its original path so
        the user doesn't end up with an orphaned half-moved pair.

        Setup: real water.xyz + water.molstruct.json in a project
        dir.  Patch ``os.replace`` to throw IOError on the SECOND
        call (the sidecar leg) — first call (structure leg)
        succeeds normally.  Endpoint should return 500 with a
        "rolled back" message and the source files must still exist
        in their original location.
        """
        import os
        struct, sidecar = _seed_paired(picker_root, stem="water")
        dst_dir = picker_root / "structures"
        dst_dir.mkdir()

        real_replace = os.replace
        calls = {"n": 0}
        def _fake_replace(a, b):
            calls["n"] += 1
            # First call = structure leg.  Let it succeed.
            # Second call = sidecar leg.  Throw.
            # Third call (rollback) = structure leg back.  Let it
            # succeed via the real call.
            if calls["n"] == 2:
                raise OSError(28, "no space left on device (simulated)")
            return real_replace(a, b)
        monkeypatch.setattr(os, "replace", _fake_replace)

        r = web.post("/api/files/move", json={
            "path":     str(struct), "dest_dir": str(dst_dir),
        })
        assert r.status_code == 500
        body = r.get_json()
        assert "rolled back" in body["error"].lower(), (
            f"expected rollback message; got {body['error']!r}"
        )
        # Source pair still where it started.
        assert struct.exists()
        assert sidecar.exists()
        # Destination pair NOT present (rollback undid the
        # structure leg; sidecar leg never landed).
        assert not (dst_dir / "water.xyz").exists()
        assert not (dst_dir / "water.molstruct.json").exists()


# --------------------------------------------------------------------- #
#  POST /api/files/copy                                                 #
# --------------------------------------------------------------------- #


class TestFilesCopy:
    """Copy a file inside the picker roots.  Source remains in place;
    sidecar pairs with the copy.  Cross-dir + same-dir-different-name
    are both supported."""

    def _copy(self, web, path: Path, dest_dir: Path, new_name=None):
        body = {"path": str(path), "dest_dir": str(dest_dir)}
        if new_name is not None:
            body["new_name"] = new_name
        return web.post("/api/files/copy", json=body)

    def test_copy_file_to_new_dir(self, web, picker_root):
        src = picker_root / "config.json"
        original = src.read_text()
        dst_dir = picker_root / "spectrum"
        r = self._copy(web, src, dst_dir)
        assert r.status_code == 200, r.get_data(as_text=True)
        assert src.exists()           # source preserved
        assert (dst_dir / "config.json").read_text() == original

    def test_copy_with_rename(self, web, picker_root):
        src = picker_root / "config.json"
        dst_dir = picker_root / "spectrum"
        r = self._copy(web, src, dst_dir, new_name="backup.json")
        assert r.status_code == 200
        assert (dst_dir / "backup.json").exists()
        assert src.exists()

    def test_copy_same_dir_requires_new_name(self, web, picker_root):
        src = picker_root / "config.json"
        r = self._copy(web, src, picker_root)
        # Same path = source.  Refused.
        assert r.status_code == 400

    def test_copy_xyz_takes_sidecar(self, web, picker_root):
        struct, sidecar = _seed_paired(picker_root, stem="water")
        dst_dir = picker_root / "structures"
        dst_dir.mkdir()
        r = self._copy(web, struct, dst_dir, new_name="backup.xyz")
        assert r.status_code == 200
        # Source pair preserved.
        assert struct.exists()
        assert sidecar.exists()
        # Destination pair present.
        assert (dst_dir / "backup.xyz").exists()
        assert (dst_dir / "backup.molstruct.json").exists()
        # Sidecar payload preserved verbatim.
        assert (
            (dst_dir / "backup.molstruct.json").read_text()
            == sidecar.read_text()
        )

    def test_copy_refuses_overwrite(self, web, picker_root):
        src = picker_root / "config.json"
        dst_dir = picker_root / "spectrum"
        (dst_dir / "config.json").write_text("existing\n")
        r = self._copy(web, src, dst_dir)
        assert r.status_code == 409

    def test_copy_refuses_directory(self, web, picker_root):
        src_dir = picker_root / "spectrum"
        other = picker_root / "other"
        other.mkdir()
        r = self._copy(web, src_dir, other)
        assert r.status_code == 400

    def test_copy_sidecar_failure_unlinks_half_copy(
            self, web, picker_root, monkeypatch):
        """2026-06-12 audit follow-up: when the structure leg of a
        sidecar-paired copy succeeds but the sidecar leg fails, the
        backend must unlink the half-copied structure file so the
        user doesn't end up with an orphaned structure-without-its-
        sidecar at the destination.

        Setup: real water.xyz + water.molstruct.json.  Patch
        ``shutil.copy2`` to throw on the SECOND call (the sidecar
        leg) — first call (structure leg) copies normally.
        Endpoint should return 500 mentioning the half-copy
        cleanup AND the destination structure must NOT exist
        afterwards (we ate our own dog food).
        """
        import shutil
        struct, sidecar = _seed_paired(picker_root, stem="water")
        dst_dir = picker_root / "structures"
        dst_dir.mkdir()

        real_copy2 = shutil.copy2
        calls = {"n": 0}
        def _fake_copy2(a, b):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(28, "no space left on device (simulated)")
            return real_copy2(a, b)
        monkeypatch.setattr(shutil, "copy2", _fake_copy2)

        r = web.post("/api/files/copy", json={
            "path":     str(struct), "dest_dir": str(dst_dir),
        })
        assert r.status_code == 500
        body = r.get_json()
        assert "half-paired" in body["error"].lower() \
            or "removed" in body["error"].lower(), (
            f"expected cleanup message; got {body['error']!r}"
        )
        # Source pair preserved.
        assert struct.exists()
        assert sidecar.exists()
        # Destination structure unlinked + sidecar never landed.
        assert not (dst_dir / "water.xyz").exists()
        assert not (dst_dir / "water.molstruct.json").exists()


# --------------------------------------------------------------------- #
#  DELETE /api/files/delete                                             #
# --------------------------------------------------------------------- #


class TestFilesDelete:
    """Validation contract per the endpoint docstring:
      * inside an allowed root + depth >= 1
      * not a canonical-topic dir at depth 2
      * recursive=true required for non-empty directories
    Matches the JS-side ``_isDeletableEntry`` gate so the user
    never sees a UI control that the backend would refuse."""

    def _delete(self, web, path, recursive=False):
        return web.delete(
            "/api/files/delete",
            json={"path": str(path), "recursive": recursive},
        )

    # --- happy paths ---------------------------------------------- #

    def test_delete_file_happy_path(self, web, picker_root):
        target = picker_root / "proj" / "spectrum" / "geom.xyz"
        target.parent.mkdir(parents=True)
        target.write_text("2\nh2\nH 0 0 0\nH 0.74 0 0\n")
        assert target.exists()
        r = self._delete(web, target)
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body["ok"] is True
        assert body["path"] == str(target)
        assert not target.exists()
        # Parent directory untouched.
        assert target.parent.is_dir()

    def test_delete_empty_dir_happy_path(self, web, picker_root):
        target = picker_root / "proj" / "user" / "scratch"
        target.mkdir(parents=True)
        r = self._delete(web, target)
        assert r.status_code == 200
        assert not target.exists()

    def test_delete_recursive_removes_non_empty_dir(self, web, picker_root):
        # Free-form subdir inside user/ so the canonical-topic
        # protection doesn't apply.
        target = picker_root / "proj" / "user" / "scratch"
        target.mkdir(parents=True)
        (target / "a.txt").write_text("x")
        (target / "nested").mkdir()
        (target / "nested" / "b.txt").write_text("y")
        r = self._delete(web, target, recursive=True)
        assert r.status_code == 200
        assert not target.exists()

    # --- rejection paths ----------------------------------------- #

    def test_delete_missing_body_400(self, web):
        # No JSON body at all.
        r = web.delete("/api/files/delete")
        assert r.status_code == 400
        assert "path" in r.get_json()["error"]

    def test_delete_missing_path_400(self, web):
        r = web.delete("/api/files/delete", json={"recursive": True})
        assert r.status_code == 400
        assert "path" in r.get_json()["error"]

    def test_delete_nonexistent_path_404(self, web, picker_root):
        target = picker_root / "proj" / "ghost.xyz"
        # ``ghost.xyz``'s parent ``proj`` doesn't exist either; the
        # resolver still computes a path inside the root, and the
        # existence check returns 404.
        (picker_root / "proj").mkdir(parents=True)
        r = self._delete(web, target)
        assert r.status_code == 404

    def test_delete_outside_root_rejected(self, web, picker_root):
        # Absolute path on a sibling tree the picker root has never
        # heard of.  (Can't use pytest's ``tmp_path`` here -- the
        # ``picker_root`` fixture aliases the SAME tmp directory, so
        # any path under tmp_path resolves inside the root.)
        outside = picker_root.parent.parent / "molbuilder_test_outside"
        r = self._delete(web, outside / "elsewhere.txt")
        assert r.status_code == 400
        err = r.get_json()["error"]
        assert "outside" in err or "root" in err

    def test_delete_dot_dot_in_path_rejected(self, web, picker_root):
        # Defense in depth: ``..`` in the raw string is rejected.
        r = web.delete(
            "/api/files/delete",
            json={"path": str(picker_root) + "/proj/../../etc"},
        )
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_delete_picker_root_itself_rejected(self, web, picker_root):
        # Cannot delete projects/ -- depth-0 protection.
        r = self._delete(web, picker_root, recursive=True)
        assert r.status_code == 400
        err = r.get_json()["error"]
        assert "root" in err.lower()

    def test_delete_canonical_topic_dir_rejected(self, web, picker_root):
        # projects/<proj>/spectrum/ is a canonical topic at depth 2.
        # Refused even with recursive=true -- protect the layout.
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        (target / "run.molwatch.log").write_text("dummy\n")
        r = self._delete(web, target, recursive=True)
        assert r.status_code == 400
        err = r.get_json()["error"]
        assert "spectrum" in err
        assert target.exists(), "target must not have been deleted"

    def test_delete_user_topic_dir_rejected(self, web, picker_root):
        # ``user`` IS a canonical topic too (added 2026-05-16 for the
        # free-form workspace).  Same protection applies.
        target = picker_root / "proj" / "user"
        target.mkdir(parents=True)
        r = self._delete(web, target)
        assert r.status_code == 400
        err = r.get_json()["error"]
        assert "user" in err
        assert target.exists()

    def test_delete_subdir_under_canonical_topic_allowed(self, web,
                                                          picker_root):
        # depth-3 free-form subdir IS deletable, even when its parent
        # is a canonical topic.  This is the canonical user workflow:
        # ``projects/<proj>/spectrum/<run>/`` can be removed.
        target = picker_root / "proj" / "spectrum" / "water_v1"
        target.mkdir(parents=True)
        r = self._delete(web, target)
        assert r.status_code == 200
        assert not target.exists()

    def test_delete_file_named_canonical_topic_allowed(self, web,
                                                        picker_root):
        # The canonical-topic guard fires only for DIRECTORIES.  A
        # plain file at depth 2 named ``spectrum`` (no extension) is
        # deletable -- it's not the layout-orphaning case.
        target = picker_root / "proj" / "spectrum"
        target.parent.mkdir(parents=True)
        target.write_text("not a directory\n")  # plain file
        r = self._delete(web, target)
        assert r.status_code == 200, r.get_data(as_text=True)
        assert not target.exists()

    def test_delete_non_empty_dir_without_recursive_409(self, web,
                                                         picker_root):
        target = picker_root / "proj" / "user" / "scratch"
        target.mkdir(parents=True)
        (target / "f.txt").write_text("x")
        r = self._delete(web, target, recursive=False)
        assert r.status_code == 409
        err = r.get_json()["error"]
        assert "recursive" in err
        assert target.exists()
        assert (target / "f.txt").exists()

    def test_delete_project_dir_with_recursive_allowed(self, web,
                                                        picker_root):
        # depth-1 = a project dir.  Deletable with recursive=true
        # because the user explicitly wants to nuke the project.
        # The canonical-topic guard only fires at depth 2.
        target = picker_root / "doomed_project"
        target.mkdir()
        (target / "spectrum").mkdir()
        (target / "spectrum" / "f.txt").write_text("x")
        r = self._delete(web, target, recursive=True)
        assert r.status_code == 200
        assert not target.exists()


# --------------------------------------------------------------------- #
#  /api/files/upload                                                    #
# --------------------------------------------------------------------- #


class TestFilesUpload:
    """Multipart upload into a sidebar-visible directory.  Same depth
    rules as /api/files/write (no uploads directly into the picker
    root; target_dir must exist as a directory) plus a filename
    regex distinct from validate_name (dots allowed for extensions)."""

    def _post(self, web, target_dir, filename, content=b"hello\n"):
        import io
        return web.post(
            "/api/files/upload",
            data={
                "target_dir": str(target_dir),
                "file": (io.BytesIO(content), filename),
            },
            content_type="multipart/form-data",
        )

    def test_upload_happy_path_writes_file(self, web, picker_root):
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        r = self._post(web, target, "water.spectra.json", b'{"ok":1}\n')
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body["ok"] is True
        assert body["path"] == str(target / "water.spectra.json")
        # File landed with the content we sent.
        assert (target / "water.spectra.json").read_bytes() == b'{"ok":1}\n'
        assert body["size"] == 9
        assert body["mtime"] > 0

    def test_upload_missing_target_dir_400(self, web):
        # Missing target_dir form field.
        import io
        r = web.post(
            "/api/files/upload",
            data={"file": (io.BytesIO(b"x"), "x.txt")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        assert "target_dir" in r.get_json()["error"]

    def test_upload_missing_file_part_400(self, web, picker_root):
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        r = web.post(
            "/api/files/upload",
            data={"target_dir": str(target)},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        assert "'file'" in r.get_json()["error"]

    def test_upload_at_root_depth_rejected(self, web, picker_root):
        # Uploading directly into the picker root (depth 0) is forbidden;
        # parallels the same rule on /api/files/write.
        r = self._post(web, picker_root, "stray.txt")
        assert r.status_code == 400
        assert "subdirectory" in r.get_json()["error"]

    def test_upload_to_missing_dir_400(self, web, picker_root):
        # target_dir resolves inside the root but doesn't exist on disk.
        nonexistent = picker_root / "proj" / "ghost"
        r = self._post(web, nonexistent, "file.txt")
        # /api/files/upload uses the same _resolve_within_roots that
        # treats missing paths as 404; either response indicates the
        # endpoint rejected cleanly.
        assert r.status_code in (400, 404)
        body = r.get_json()
        assert body["ok"] is False

    def test_upload_to_a_file_400(self, web, picker_root):
        # target_dir is a file, not a directory.
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        (target / "blob.bin").write_bytes(b"x")
        r = self._post(web, target / "blob.bin", "file.txt")
        assert r.status_code == 400
        assert "directory" in r.get_json()["error"]

    def test_upload_outside_root_rejected(self, web, tmp_path):
        # Absolute path completely outside the picker root.
        r = self._post(web, tmp_path / "elsewhere", "file.txt")
        assert r.status_code == 400
        assert "outside" in r.get_json()["error"] or "root" in r.get_json()["error"]

    def test_upload_dot_dot_in_target_rejected(self, web, picker_root):
        # Defense in depth: '..' in raw target_dir string is rejected
        # even though the resolution step would also catch it.
        r = self._post(web, str(picker_root) + "/proj/../..", "file.txt")
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_upload_existing_filename_409(self, web, picker_root):
        # No implicit overwrite: clash at destination is 409.  The
        # sidebar's UX is "delete first, then re-upload".
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        (target / "geom.xyz").write_text("existing\n")
        r = self._post(web, target, "geom.xyz", b"replacement\n")
        assert r.status_code == 409
        assert "already exists" in r.get_json()["error"]
        # Original file content is untouched.
        assert (target / "geom.xyz").read_text() == "existing\n"

    def test_upload_overwrite_replaces_existing(self, web, picker_root):
        """Phase 6e: ``overwrite=true`` lets the upload endpoint
        replace an existing file.  Used by the embed's
        save-to-project for animation / image (Blob) exports — the
        text-write path supports overwrite; binary writes route
        through upload, which now does too."""
        import io
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        (target / "movie.gif").write_bytes(b"old-bytes")
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir": str(target),
                "file":       (io.BytesIO(b"new-bytes"), "movie.gif"),
                "overwrite":  "true",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body["ok"] is True
        assert (target / "movie.gif").read_bytes() == b"new-bytes"

    def test_upload_overwrite_false_still_409(self, web, picker_root):
        """Without overwrite (or with overwrite=false), conflict is
        still 409 — same as the no-flag default."""
        import io
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        (target / "movie.gif").write_bytes(b"original")
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir": str(target),
                "file":       (io.BytesIO(b"replacement"), "movie.gif"),
                "overwrite":  "false",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 409
        assert (target / "movie.gif").read_bytes() == b"original"

    def test_upload_auto_rename_picks_unused_name(
            self, web, picker_root):
        """Phase 6e: ``auto_rename=true`` resolves a collision by
        appending ``-2``, ``-3``, ... until a free slot is found.
        Used by the embed's export-params dialog so a re-save of
        the default filename produces a new file rather than
        clobbering."""
        import io
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        (target / "movie.gif").write_bytes(b"first")
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir":  str(target),
                "file":        (io.BytesIO(b"second"), "movie.gif"),
                "auto_rename": "true",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body["ok"] is True
        # Server picked "movie-2.gif"; original is untouched.
        assert body["path"] == str(target / "movie-2.gif")
        assert (target / "movie.gif").read_bytes() == b"first"
        assert (target / "movie-2.gif").read_bytes() == b"second"

    def test_upload_auto_rename_walks_past_multiple_collisions(
            self, web, picker_root):
        """When -2, -3 are also taken, the picker continues to -4."""
        import io
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        for name in ["movie.gif", "movie-2.gif", "movie-3.gif"]:
            (target / name).write_bytes(b"prior")
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir":  str(target),
                "file":        (io.BytesIO(b"new"), "movie.gif"),
                "auto_rename": "true",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body["path"] == str(target / "movie-4.gif")
        assert (target / "movie-4.gif").read_bytes() == b"new"

    def test_upload_auto_rename_no_collision_uses_original_name(
            self, web, picker_root):
        """auto_rename is a no-op when the original name is free —
        the file lands at the requested path."""
        import io
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir":  str(target),
                "file":        (io.BytesIO(b"x"), "fresh.gif"),
                "auto_rename": "true",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        assert r.get_json()["path"] == str(target / "fresh.gif")

    def test_upload_refuses_to_write_through_symlink(
            self, web, picker_root, tmp_path):
        """Phase 6e second-review LANDMINE #18: a symlink at the
        destination must NOT be followed.  Otherwise an attacker
        could plant a dangling symlink pointing at a sensitive
        file and a subsequent upload would clobber it through the
        link."""
        import io, os
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        # Plant a dangling symlink at the upload target.  Use an
        # outside-roots target so we can verify nothing was
        # written there even when the upload succeeds elsewhere.
        outside = tmp_path / "outside-target"
        os.symlink(str(outside), str(target / "movie.gif"))
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir":  str(target),
                "file":        (io.BytesIO(b"replaced"), "movie.gif"),
                "overwrite":   "true",
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 400, r.get_data(as_text=True)
        assert "symlink" in r.get_json()["error"]
        # The link target was never created; the link itself is
        # still where we planted it.
        assert not outside.exists()
        assert (target / "movie.gif").is_symlink()


    def test_upload_filename_with_path_separator_400(self, web, picker_root):
        # ``file.filename`` may carry the client's full path on some
        # browsers; we basename it server-side.  This test sends a
        # bare slash to confirm the validator catches what slips
        # through.  (Browsers normally send just the basename.)
        import io
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        # werkzeug normalises some path prefixes; we test the regex
        # by sending a value that survives basename().
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir": str(target),
                "file": (io.BytesIO(b"x"), "has space.txt"),
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        assert "unsupported" in r.get_json()["error"]

    def test_upload_dotfile_rejected(self, web, picker_root):
        # Leading-dot filenames (.bashrc etc.) are rejected by the
        # ^[A-Za-z0-9] anchor.  Matches the sidebar list endpoint's
        # hidden-filter so we don't upload files that wouldn't show
        # up in the sidebar.
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        r = self._post(web, target, ".bashrc")
        assert r.status_code == 400
        assert "unsupported" in r.get_json()["error"]

    def test_upload_strips_client_path_prefix(self, web, picker_root):
        # Some browsers / curl invocations send the FULL client path
        # as ``file.filename``.  ``os.path.basename`` strips that
        # before validation + write, so the file lands at
        # target_dir/<basename>.
        import io
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        r = web.post(
            "/api/files/upload",
            data={
                "target_dir": str(target),
                "file": (io.BytesIO(b"data\n"), "/tmp/from-client/water.xyz"),
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body["path"] == str(target / "water.xyz")
        assert (target / "water.xyz").read_bytes() == b"data\n"


class TestFilesWriteAutoRename:
    """Phase 6e second-review BOMB #11: the export dialog promises
    auto-rename for ALL kinds; previously only /upload (binary)
    honored auto_rename, so text exports (.xyz/.pdb) 409'd on
    collision after the dialog said they wouldn't.  These tests
    pin the /write parity."""

    def _post(self, web, path, text, **extra):
        body = {"path": str(path), "text": text}
        body.update(extra)
        return web.post(
            "/api/files/write",
            json=body,
        )

    def test_auto_rename_picks_dash_2_on_first_collision(
            self, web, picker_root):
        target = picker_root / "proj"
        target.mkdir(parents=True)
        (target / "structure.xyz").write_text("first\n")
        r = self._post(web, target / "structure.xyz", "second\n",
                       auto_rename=True)
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body["ok"] is True
        assert body["path"] == str(target / "structure-2.xyz")
        # Original untouched.
        assert (target / "structure.xyz").read_text() == "first\n"
        assert (target / "structure-2.xyz").read_text() == "second\n"

    def test_auto_rename_walks_multiple_collisions(
            self, web, picker_root):
        target = picker_root / "proj"
        target.mkdir(parents=True)
        for n in ["structure.xyz", "structure-2.xyz",
                  "structure-3.xyz"]:
            (target / n).write_text("prior\n")
        r = self._post(web, target / "structure.xyz", "n4\n",
                       auto_rename=True)
        assert r.status_code == 200
        assert r.get_json()["path"] == str(target / "structure-4.xyz")

    def test_auto_rename_no_collision_uses_original_path(
            self, web, picker_root):
        target = picker_root / "proj"
        target.mkdir(parents=True)
        r = self._post(web, target / "fresh.xyz", "data\n",
                       auto_rename=True)
        assert r.status_code == 200
        assert r.get_json()["path"] == str(target / "fresh.xyz")

    def test_overwrite_wins_when_both_flags_set(
            self, web, picker_root):
        """overwrite=true wins over auto_rename=true; the request
        is treated as an explicit clobber.  Matches /upload's
        precedence."""
        target = picker_root / "proj"
        target.mkdir(parents=True)
        (target / "x.xyz").write_text("first\n")
        r = self._post(web, target / "x.xyz", "second\n",
                       overwrite=True, auto_rename=True)
        assert r.status_code == 200
        assert r.get_json()["path"] == str(target / "x.xyz")
        assert (target / "x.xyz").read_text() == "second\n"

    def test_write_refuses_symlink_target(
            self, web, picker_root, tmp_path):
        """LANDMINE #18 mirror for the text-write path."""
        import os
        target = picker_root / "proj"
        target.mkdir(parents=True)
        outside = tmp_path / "outside.txt"
        os.symlink(str(outside), str(target / "linky.xyz"))
        r = self._post(web, target / "linky.xyz", "data\n",
                       overwrite=True)
        assert r.status_code == 400
        assert "symlink" in r.get_json()["error"]
        assert not outside.exists()

    def test_write_refuses_directory_target(
            self, web, picker_root):
        """Phase 6e third-review POLISH-3: writing to a directory
        path should 400 with a clean message rather than 500'ing
        on IsADirectoryError (or 200'ing into ``<dirname>-2``
        via auto_rename)."""
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        # Try to write to the directory itself.
        r = self._post(web, target, "data\n", overwrite=True)
        assert r.status_code == 400, r.get_data(as_text=True)
        assert "directory" in r.get_json()["error"]

    def test_write_directory_target_with_auto_rename_still_400(
            self, web, picker_root):
        """auto_rename must NOT turn a directory target into
        ``<dirname>-2`` — that was the worse failure mode the
        is_dir guard prevents."""
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        r = self._post(web, target, "data\n", auto_rename=True)
        assert r.status_code == 400
        # Sibling file did NOT appear.
        sibling = target.parent / "spectrum-2"
        assert not sibling.exists()

    def test_write_rejects_leading_space_filename(
            self, web, picker_root):
        """Phase 6e sixth-review LANDMINE-6: ``/api/files/write``
        must reject the same filenames ``/upload`` does.  Leading
        space is the canonical case the sixth audit flagged."""
        target = picker_root / "proj"
        target.mkdir(parents=True)
        r = self._post(web, target / " foo.xyz", "data\n",
                       overwrite=True)
        assert r.status_code == 400
        assert "unsupported" in r.get_json()["error"].lower()
        # Confirm the file was NOT written.
        assert not (target / " foo.xyz").exists()

    def test_write_rejects_dotfile_leaf(
            self, web, picker_root):
        """LANDMINE-6 mirror: dotfiles (``.bashrc``) rejected by
        /upload; same shape on /write."""
        target = picker_root / "proj"
        target.mkdir(parents=True)
        r = self._post(web, target / ".bashrc", "data\n",
                       overwrite=True)
        assert r.status_code == 400
        assert "unsupported" in r.get_json()["error"].lower()

    def test_write_upload_filename_parity(
            self, web, picker_root):
        """Symmetric assertion: every filename one endpoint accepts
        the other accepts; every filename one rejects the other
        rejects.  Pins the parity invariant."""
        import io
        target = picker_root / "proj"
        target.mkdir(parents=True)
        cases = [
            "good.xyz",       # accepted
            " bad.xyz",       # rejected (leading space)
            ".dotfile",       # rejected (leading dot)
            "with space.xyz", # rejected (space in middle)
            "1-numeric.xyz",  # accepted
        ]
        for name in cases:
            r_w = self._post(
                web, target / name, "x\n", overwrite=True)
            r_u = web.post(
                "/api/files/upload",
                data={
                    "target_dir": str(target),
                    "file":       (io.BytesIO(b"x"), name),
                    "overwrite":  "true",
                },
                content_type="multipart/form-data",
            )
            assert r_w.status_code == r_u.status_code, (
                f"parity drift on {name!r}: write={r_w.status_code} "
                f"upload={r_u.status_code}"
            )

    def test_write_upload_auto_rename_collision_parity(
            self, web, picker_root):
        """Phase 6e seventh-review LANDMINE-8: the auto_rename
        suffix-picker is duplicated across /upload + /write.
        Pin the invariant that both pick the SAME ``<stem>-N<ext>``
        for the same collision state — otherwise a future change
        that tightens one loop's validator won't be mirrored by
        the other.
        """
        import io
        target = picker_root / "proj"
        target.mkdir(parents=True)
        (target / "movie.gif").write_bytes(b"first")

        # Upload with auto_rename: picks movie-2.gif.
        r_u = web.post(
            "/api/files/upload",
            data={
                "target_dir":  str(target),
                "file":        (io.BytesIO(b"u"), "movie.gif"),
                "auto_rename": "true",
            },
            content_type="multipart/form-data",
        )
        assert r_u.status_code == 200
        upload_path = r_u.get_json()["path"]
        # Clean up so write sees the same collision state.
        (target / "movie-2.gif").unlink()

        # Write with auto_rename: should also pick movie-2.gif.
        r_w = self._post(web, target / "movie.gif", "w\n",
                         auto_rename=True)
        assert r_w.status_code == 200
        write_path = r_w.get_json()["path"]
        assert upload_path == write_path, (
            f"auto_rename parity drift: upload picked "
            f"{upload_path!r}; write picked {write_path!r}"
        )


class TestSidebarStubsUI:
    """The stub features ship with their full UI surface so the design
    is reviewable.  Markup checks here; behaviour is exercised at the
    E2E layer (deferred Playwright suite)."""

    def test_upload_action_in_partial(self, web, picker_root):
        """2026-06-12: the upload action lives in its own dedicated
        button alongside New project + New folder (see
        TestSidebarCreateUI for the bar's full anchor set)."""
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        assert 'data-action="upload"' in body
        assert 'id="ps-create-upload-btn"' in body
        assert "Upload" in body

    def test_preview_modal_markup_full(self, web, picker_root):
        r = web.get("/spectrum-calculation")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # Modal scaffolding: backdrop, window, header (title + close),
        # CodeMirror mount point (#ps-preview-cmview replaces the
        # earlier <pre>+<textarea> pair — single editor for both view
        # and edit, virtual scroll caps DOM memory for large files,
        # search + jump-to-line addons handle find/Go-to-line).  Task
        # #310 (2026-06-09) ripped out the <pre>/textarea pair; #302
        # (2026-06-09) wired Edit + Save via /api/files/write +
        # expected_mtime.
        assert 'id="ps-preview-modal"' in body
        assert 'class="ps-preview-backdrop"' in body
        assert 'id="ps-preview-title"' in body
        assert 'id="ps-preview-meta"' in body
        assert 'id="ps-preview-cmview"' in body
        assert 'id="ps-preview-error"' in body
        assert 'id="ps-preview-status"' in body
        # The retired pair must NOT come back — guard against a
        # future refactor that accidentally re-introduces a duplicate
        # body / edit element alongside the CodeMirror mount.
        assert 'id="ps-preview-body"' not in body, (
            "ps-preview-body was retired in task #310 — CodeMirror "
            "owns the view; re-introducing the <pre> means two "
            "elements receive content for the same modal"
        )
        assert 'id="ps-preview-edit"' not in body, (
            "ps-preview-edit textarea was retired in task #310 — "
            "CodeMirror handles edit mode by toggling readOnly off"
        )
        # Edit toggle + Save button both present.  Save starts
        # disabled (no dirty edits on first open) but is no longer
        # the "coming soon" stub from v1.
        assert 'id="ps-preview-edit-btn"' in body
        assert 'id="ps-preview-save-btn"' in body
        save_attrs = body.split(
            'id="ps-preview-save-btn"', 1,
        )[1].split(">", 1)[0]
        assert "disabled" in save_attrs, (
            "Save button must start disabled — it enables when the "
            "editor has unsaved changes"
        )

    def test_preview_modal_starts_hidden(self, web, picker_root):
        # The hidden attribute ensures it doesn't flash on first paint
        # before JS runs.
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        assert 'id="ps-preview-modal" class="ps-preview-modal" hidden' in body

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

    def test_stale_file_picker_section_is_refused_by_name(self, tmp_path):
        # The file_picker section went with the single-root pivot, and
        # since the unknown-keys guard (2026-08-12) a section the loader
        # does not know is REFUSED with its name, not silently dropped:
        # "a key this loader does not know would be silently
        # ineffective" is exactly what happened to a user's stale
        # file_picker block under the old graceful ignore.
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        cfg_file = tmp_path / "molbuilder.json"
        cfg_file.write_text('{"file_picker": {"roots": ["~/scratch"]}}')
        with pytest.raises(RuntimeConfigError, match="file_picker"):
            read_config(cfg_file)

    def test_get_file_picker_roots_removed_from_runtime_config(self):
        # The accessor that v1 added was dropped during the single-
        # root pivot.  Importing it should fail; this test pins the
        # removal so a future revert is caught.
        import molbuilder.runtime_config as rc
        assert not hasattr(rc, "get_file_picker_roots")


class TestDownloadZip:
    """/api/files/download_zip -- the *carry a calculation without ssh*
    door (user, 2026-08-28).  A directory streams as <name>.zip; the
    fence holds; a root refuses; symlinks escaping the root are
    skipped and said, symlinks inside are followed as real bytes."""

    def test_a_directory_streams_as_its_named_zip(self, web, picker_root):
        import io
        import zipfile
        r = web.get("/api/files/download_zip",
                    query_string={"path": str(picker_root / "spectrum" / "BDT")})
        assert r.status_code == 200
        assert r.data[:2] == b"PK", "not a zip"
        assert 'filename=BDT.zip' in r.headers["Content-Disposition"]
        with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
            names = sorted(zf.namelist())
            assert names == ["BDT/.hidden", "BDT/water_spectra.spectra.json"], (
                f"member paths must ride under the folder's name: {names}")
            assert zf.read("BDT/water_spectra.spectra.json") \
                == b'{"schema_version": 2}\n', "bytes must survive verbatim"
        assert r.headers["X-Molbuilder-Skipped"] == "0"

    def test_the_projects_root_itself_is_refused(self, web, picker_root):
        r = web.get("/api/files/download_zip",
                    query_string={"path": str(picker_root)})
        assert r.status_code == 400
        assert "refusing to zip a projects root" in r.get_json()["error"]

    def test_a_file_names_the_single_file_door(self, web, picker_root):
        r = web.get("/api/files/download_zip",
                    query_string={"path": str(picker_root / "water.xyz")})
        assert r.status_code == 400
        assert "/api/files/download" in r.get_json()["error"]

    def test_outside_the_fence_is_refused(self, web, picker_root):
        r = web.get("/api/files/download_zip",
                    query_string={"path": "/etc"})
        assert r.status_code in (400, 403)
        assert r.get_json()["ok"] is False

    def test_missing_is_a_404(self, web, picker_root):
        r = web.get("/api/files/download_zip",
                    query_string={"path": str(picker_root / "spectrum" / "nope")})
        assert r.status_code == 404

    def test_symlinks_escape_skipped_inside_followed(self, web, picker_root):
        import io
        import zipfile
        d = picker_root / "spectrum" / "linked"
        d.mkdir()
        (d / "real.txt").write_text("kept\n")
        (d / "escape").symlink_to("/etc/hostname")          # outside -> skip
        (d / "warm").symlink_to(picker_root / "water.xyz")  # inside -> follow
        r = web.get("/api/files/download_zip",
                    query_string={"path": str(d)})
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
            names = sorted(zf.namelist())
            assert "linked/real.txt" in names
            assert "linked/warm" in names, (
                "an inside-the-tree symlink is a warm file -- the target "
                "machine needs its BYTES, so it must be followed")
            assert "linked/escape" not in names, (
                "a symlink escaping the root must never enter the archive")
            assert zf.read("linked/warm").startswith(b"3\n"), (
                "the followed link must carry the target's real content")
        assert r.headers["X-Molbuilder-Skipped"] == "1"

    def test_the_checkpoint_store_never_enters_an_archive(self, web, picker_root):
        """User 2026-08-28: the zip is a CLEAN run structure -- the
        server's workspace/checkpoint store (.molbuilder_workspace) is
        excluded wherever it appears, by the store module's own name."""
        import io
        import zipfile
        d = picker_root / "spectrum" / "BDT"
        st = d / ".molbuilder_workspace" / "states"
        st.mkdir(parents=True)
        (st / "tab.json").write_text('{"draft": true}\n')
        r = web.get("/api/files/download_zip", query_string={"path": str(d)})
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
            assert not any(".molbuilder_workspace" in n for n in zf.namelist()), (
                "server state leaked into the archive: " + str(zf.namelist()))
