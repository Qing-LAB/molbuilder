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

    def test_projects_selection_shim_removed(self, web, picker_root):
        # The per-tab projects-selection shim was retired -- the sidebar
        # actions section took over (no more "Use this file" banner).
        r = web.get("/static/lib/projects-selection.js")
        assert r.status_code == 404

    @pytest.mark.parametrize("path", [
        "/structure", "/structure-optimization",
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
        "/structure", "/structure-optimization",
        "/spectrum-calculation", "/transport-calculation",
        "/results"])
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

    @pytest.mark.parametrize("path", ["/structure"])
    def test_subscriber_tabs_use_inquire_api(
        self, web, picker_root, path,
    ):
        # /modify is the canonical "subscriber tab": it reacts to the
        # Projects-sidebar selection by auto-loading the picked XYZ
        # into the viewer + selection panel.  The wiring lives in
        # modify/selection-bootstrap.js -- the bootstrap subscribes
        # to ``window.molbuilder.projects.onChange`` and forwards
        # changes to the selection store, which loads the file.
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
        body = web.get("/structure-optimization").get_data(as_text=True)
        # No href="/projects" anywhere -- the sidebar replaced the tab.
        assert 'href="/projects"' not in body
        # Each app-tab link points at one of the served routes.  Pull
        # every href from the app-tab class; assert every value is in
        # the served-routes set.  Adding a new tab updates the served-
        # routes set, not a magic number.
        SERVED = {"/structure", "/structure-optimization",
                  "/spectrum-calculation", "/transport-calculation",
                  "/results"}
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


class TestNoLocalFileInputs:
    """After the sidebar pivot, the browser-local <input type=file>
    pickers were dropped from Spectra / Watch / Modify -- a script
    running on the server can't read a laptop file anyway."""

    @pytest.mark.parametrize("path,absent_id", [
        ("/spectra", 'id="xyz-file"'),
        ("/spectra", 'id="results-file"'),
        ("/modify",  'id="file-picker"'),
        # /watch dropped 2026-05-19; trajectory inspector lives at
        # /results via the registry now (no <input type=file>
        # there either).
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
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        # + New project section (foldable details + form + error slot)
        assert 'class="ps-create-section"' in body
        assert 'class="ps-create-summary">+ New project</summary>' in body
        assert 'id="ps-newproject-form"' in body
        assert 'id="ps-newproject-input"' in body
        assert 'id="ps-newproject-error"' in body
        assert 'id="ps-newproject-cancel"' in body
        # Subdir-list note (populated by JS at startup)
        assert 'id="ps-newproject-subdirs"' in body

class TestSidebarMkdirUI:
    """The "+ New subdir" button + inline form is in the partial,
    not the JS."""

    def test_mkdir_form_in_sidebar_partial(self, web, picker_root):
        # Any tab that includes the sidebar partial carries the markup.
        # The form is now inside a <details class="ps-create-section">,
        # so visibility is HTML-controlled (no `hidden` attr).
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        assert 'id="ps-mkdir-form"' in body
        assert 'id="ps-mkdir-input"' in body
        assert 'id="ps-mkdir-error"' in body
        assert 'id="ps-mkdir-cancel"' in body
        # The + New subdir summary heading lives inside its details.
        assert '+ New subdir</summary>' in body

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

    def test_upload_section_in_partial(self, web, picker_root):
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        assert 'id="ps-upload-form"' in body
        assert 'id="ps-upload-input"' in body
        assert 'id="ps-upload-error"' in body
        assert 'class="ps-upload-context"' in body
        # The summary heading is the user-facing label.
        assert '+ Upload file</summary>' in body

    def test_preview_modal_markup_full(self, web, picker_root):
        body = web.get("/spectrum-calculation").get_data(as_text=True)
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
