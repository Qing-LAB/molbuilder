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
        """A second picker root would put files outside the projects tree one
        click away on every tab.

        `web-api.md` § 4 (`GET /api/files/roots`) and `job-contracts.md` § 2.5:
        `projects/` is the one tree the browser browses. The route also owes
        the sidebar `path` / `label` / `exists` per root -- without `exists`
        the header cannot tell a missing root from an empty one. The root SET
        itself is pinned at its source by
        `TestRootsContract::test_capabilities_returns_only_projects_root`; here
        the roots are monkeypatched, so what this holds is the envelope.
        """
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
        """Files sorted ahead of directories would make the sidebar unnavigable
        -- the folders a user is looking for scroll off under a long file list.

        `web-api.md` § 4 (`GET /api/files/list`) and `projects.md` § 4:
        directories first, then files by name. Honest note (2026-09-09): the
        hidden-entry assertion at the end of this test cannot fail -- the
        fixture plants no dotfile at this level -- so
        `test_list_filters_hidden_entries` is where that rule is actually
        exercised.
        """
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
        """A listing that stopped filtering dotfiles would show `.git`,
        `.binsnapshots` and `.molbuilder_workspace` as ordinary browsable
        folders, inviting a user to delete storage a run depends on.

        `projects.md` § 4 (what the sidebar shows) and `web-api.md` § 4. This
        is the only test that reaches the hidden filter with a dotfile actually
        present.
        """
        r = web.get(
            f"/api/files/list?path={picker_root}/spectrum/BDT"
        )
        assert r.status_code == 200
        j = r.get_json()
        names = [e["name"] for e in j["entries"]]
        assert "water_spectra.spectra.json" in names
        assert ".hidden" not in names

    def test_list_ext_filter(self, web, picker_root):
        """A filter that also hid directories would strand every matching file
        that lives one folder down: the user cannot navigate to what the filter
        is for.

        `web-api.md` § 4 (`GET /api/files/list`, `ext=`): the filter narrows
        FILES only.
        """
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
        """A caller writing `ext=xyz` instead of `ext=.xyz` gets an empty
        folder and no error -- the failure is silent, which is why it is pinned
        rather than left to review.

        `web-api.md` § 4 (`GET /api/files/list`).
        """
        # ext=xyz (no leading dot) should behave the same as ext=.xyz
        r = web.get(f"/api/files/list?path={picker_root}&ext=xyz")
        names = [e["name"] for e in r.get_json()["entries"]]
        assert "water.xyz" in names
        assert "config.json" not in names

    def test_list_entries_carry_kind_size_mtime(self, web, picker_root):
        """A dropped or renamed key blanks the sidebar row with no error, and
        `size: 0` on a directory would render a folder as an empty file.

        `web-api.md` § 4 (`GET /api/files/list`), `projects.md` § 4: each entry
        carries kind, size (null for a directory) and mtime.
        """
        r = web.get(f"/api/files/list?path={picker_root}")
        entries = {e["name"]: e for e in r.get_json()["entries"]}
        # Files report size + finite mtime; dirs report size=null.
        assert entries["water.xyz"]["kind"] == "file"
        assert entries["water.xyz"]["size"] > 0
        assert entries["water.xyz"]["mtime"] > 0
        assert entries["spectrum"]["kind"] == "directory"
        assert entries["spectrum"]["size"] is None

    def test_list_missing_path_400(self, web):
        """A `path`-less request answered with a 500, or with a listing of some
        default directory, instead of a named 400.

        `web-api.md` § 1 (status codes): a malformed request is 400 and the
        message names the missing parameter. The refusal comes from the fence
        (`files._resolve_within_roots`, § 2.1) -- its empty-path branch.
        """
        r = web.get("/api/files/list")
        assert r.status_code == 400
        assert "missing 'path'" in r.get_json()["error"]

    def test_list_nonexistent_path_404(self, web, picker_root):
        """A missing directory answered as a 500 (a server fault) or as a 200
        with an empty list -- the sidebar would show a folder that is not there
        instead of saying it is gone.

        `web-api.md` § 1 status table: 404 is 'no such file / directory'.
        """
        r = web.get(
            f"/api/files/list?path={picker_root}/nope_no_such_dir"
        )
        assert r.status_code == 404

    def test_list_file_not_directory_400(self, web, picker_root):
        """Pointing `list` at a file answered with a 500 from the OS error, or
        with a one-entry listing, instead of the usage error it is.

        `web-api.md` § 1 status table: a bad request is 400, and 5xx is
        reserved for a server fault.
        """
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
        """`stat` losing `kind`, `size` or `mtime` leaves the preview header
        blank and the editor with no mtime to send back as `expected_mtime` --
        which silently disables the lost-update guard on save.

        `web-api.md` § 4 (`GET /api/files/stat`); the save side of that same
        mtime is `test_write_mtime_mismatch_returns_409`.
        """
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
        """A directory reported as `kind: file`, or with `size: 0` instead of
        null, makes the sidebar offer a preview of a folder.

        `web-api.md` § 4 (`GET /api/files/stat`): the same kind / size
        convention the listing uses.
        """
        r = web.get(
            f"/api/files/stat?path={picker_root}/spectrum"
        )
        assert r.status_code == 200
        j = r.get_json()
        assert j["kind"] == "directory"
        assert j["size"] is None

    def test_stat_nonexistent_404(self, web, picker_root):
        """A stat of something that is gone answered 200 or 500 rather than 404
        -- the client cannot tell 'deleted' from 'the server broke'.

        `web-api.md` § 1 status table.
        """
        r = web.get(
            f"/api/files/stat?path={picker_root}/nope"
        )
        assert r.status_code == 404


# --------------------------------------------------------------------- #
#  /api/files/read                                                      #
# --------------------------------------------------------------------- #


class TestFilesRead:

    def test_read_returns_text(self, web, picker_root):
        """`read` reporting a `size` that does not match the text it returned:
        a truncated body reported as whole leaves the viewer believing it holds
        the file when it holds part of it.

        `web-api.md` § 4 (`GET /api/files/read`).
        """
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz"
        )
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["text"].startswith("3\nwater")
        assert j["size"] == len(j["text"])

    def test_read_respects_max_bytes_with_413(self, web, picker_root):
        """Without the cap, one click on a multi-gigabyte `.DM` pulls it
        through the request thread and into the browser; and a 413 that omits
        the file's real `size` leaves the viewer unable to say how big it is or
        to fall back to `read_range`.

        `web-api.md` § 1 status table (413 = payload too large) and § 1a (the
        three-second rule).
        """
        # File is ~35 bytes; cap at 5 → 413 with the file's actual size.
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz&max_bytes=5"
        )
        assert r.status_code == 413
        j = r.get_json()
        assert j["ok"] is False
        assert j["size"] > 5

    def test_read_directory_400(self, web, picker_root):
        """Reading a directory answered as a 500 from `IsADirectoryError`
        instead of the usage error it is.

        `web-api.md` § 1 status table.
        """
        r = web.get(
            f"/api/files/read?path={picker_root}/spectrum"
        )
        assert r.status_code == 400

    def test_read_rejects_invalid_max_bytes(self, web, picker_root):
        """A non-integer `max_bytes` reaching `int()` unguarded is a 500 on a
        request the caller merely mistyped.

        `web-api.md` § 1 status table.
        """
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz"
            f"&max_bytes=not_an_int"
        )
        assert r.status_code == 400

    def test_read_rejects_max_bytes_above_ceiling(self, web, picker_root):
        """Without the 16 MB ceiling a caller can name any size and have the
        server read that much into memory inside one request thread -- a denial
        of service from a single URL.

        `web-api.md` § 1a (the three-second rule) and § 2 (the 50 MB global
        upload cap; this is the read-side ceiling).
        """
        # Hard ceiling is 16 MB.
        r = web.get(
            f"/api/files/read?path={picker_root}/water.xyz"
            f"&max_bytes=999999999"
        )
        assert r.status_code == 400

    def test_read_non_utf8_400(self, web, picker_root):
        """A binary file answered with mojibake, or with a 500 from
        `UnicodeDecodeError`, instead of a refusal that says the file is not
        text -- the viewer would render replacement characters and the user
        would conclude the file is corrupt.

        `web-api.md` § 1 status table; the message must name UTF-8, because
        that message is what the preview shows.
        """
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
        """A chunk that reaches the end without `eof: true` leaves the viewer's
        paginator asking for the same offset forever.

        `web-api.md` § 4 (`GET /api/files/read_range`, task #119, 2026-06-02).
        Honest note (2026-09-09):
        `test_read_range_default_returns_start_of_file` reaches the same `eof`
        computation with the same relation (requested span larger than the
        file), so this is raised as a cut candidate.
        """
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
        """An offset past the end answered with an empty 200 would be
        indistinguishable from a legitimate end-of-file read, so a client
        paging a file that was truncated under it loops instead of reporting
        the change.

        `web-api.md` § 4 (`read_range`). The other side of the boundary is
        `test_read_range_offset_at_eof_returns_empty_chunk`: offset ==
        file_size is 200, offset > file_size is 400.
        """
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
        """A non-integer `offset` reaching `int()` unguarded is a 500 on a
        mistyped request.

        `web-api.md` § 1 status table.
        """
        r = web.get(
            f"/api/files/read_range?path={picker_root}/water.xyz"
            f"&offset=not_an_int")
        assert r.status_code == 400

    def test_read_range_invalid_max_bytes_returns_400(
            self, web, picker_root):
        """The same for `max_bytes`: `read_range` parses two numbers, and a
        guard added to one and not the other is the drift this catches.

        `web-api.md` § 1 status table.
        """
        r = web.get(
            f"/api/files/read_range?path={picker_root}/water.xyz"
            f"&max_bytes=zero")
        assert r.status_code == 400

    def test_read_range_max_bytes_above_ceiling_returns_400(
            self, web, picker_root):
        """`read_range` is the endpoint a viewer calls in a loop, so an
        unbounded `max_bytes` here is the cheapest way to make the server read
        an arbitrary amount per request.

        `web-api.md` § 1a; mirrors the `read` ceiling pinned by
        `test_read_rejects_max_bytes_above_ceiling`.
        """
        r = web.get(
            f"/api/files/read_range?path={picker_root}/water.xyz"
            f"&max_bytes=99999999999")
        assert r.status_code == 400

    def test_read_range_missing_file_404(self, web, picker_root):
        """A range read of a file that is gone answered as an empty 200 or a
        500 rather than 404 -- the viewer cannot tell a deleted file from a
        server fault mid-scroll.

        `web-api.md` § 1 status table.
        """
        r = web.get(
            f"/api/files/read_range?path={picker_root}/no-such.log")
        assert r.status_code == 404

    def test_read_range_directory_returns_400(self, web, picker_root):
        """Ranging over a directory answered as a 500 from the OS error instead
        of a usage 400.

        `web-api.md` § 1 status table.
        """
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
        """A raw `..` reaching path resolution. The ones that resolve outside
        are caught by the containment check, but a `..` that cancels back
        INSIDE the root (`<root>/proj/../secret`) resolves to a legal path and
        would be served -- the raw-string refusal is the only thing that stops
        it.

        `web-api.md` § 2.1 (a path from the browser is fenced at the ROUTE).
        This is the DOOR test for `files._resolve_within_roots`;
        `test-audit-findings.md` § 5 names it as the coverer that let other
        routes' thin traversal wrappers be cut.
        """
        # Even before resolution, a path with .. is rejected.  This
        # avoids ambiguity for users who type '..' assuming it would
        # be normalised harmlessly.
        r = web.get("/api/files/list?path=../../etc")
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_absolute_path_outside_root_rejected(self, web, picker_root):
        """Without the containment check, any absolute path the server process
        can read becomes readable by anyone with a session -- `/etc/shadow` was
        the measured case (`web-api.md` § 2.1, the 2026-06-18 `watch.py` fix).

        `web-api.md` § 2.1. The refusal must also NAME the allowed roots, which
        is what the error-text assertion holds.
        """
        # /etc is not inside the tmp picker root → outside-root reject.
        r = web.get("/api/files/list?path=/etc")
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]

    def test_symlink_to_outside_root_rejected(self, web, picker_root):
        """The fence resolves symlinks BEFORE it checks containment; reverse
        those two steps and a symlink planted inside `projects/` -- by an
        upload, a shared filesystem, or the user's own `ln -s` -- reads
        anything on the machine while every raw-string check still passes.

        `web-api.md` § 2.1. This is the only test that pins the resolve-then-
        check ORDER.
        """
        # Symlink resolves to /tmp (outside the picker_root tmp).
        # _resolve_within_roots follows symlinks before checking, so
        # the resolved path is what the boundary check sees.
        link = picker_root / "leak"
        link.symlink_to("/etc")
        r = web.get(f"/api/files/list?path={link}")
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]

    def test_empty_path_400(self, web):
        """An empty `path` falling through the fence to `Path('')`, which
        resolves to the process CWD -- the caller would get a listing of
        wherever the server was launched.

        `web-api.md` § 2.1 and § 1 status table. It is also the only test in
        this file that drives `/api/files/stat` through the fence at all: the
        audit of 2026-09-09 records that stat / read / read_range have no
        outside-root test of their own.
        """
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
        """A re-added `/projects` page would give the sidebar a second home,
        and the two would disagree about what is selected.

        `projects.md` § 1 (the one door) -- the standalone tab was retired for
        the persistent sidebar. Honest note (2026-09-09): no document records
        the removal (`web-api.md` § 7's removed-routes table does not carry
        it), so this test is the only record of it, which is the weakness
        rather than the strength; raised as a cut candidate.
        """
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
        """The per-tab selection shim coming back would put a second subscriber
        on the selection, so a picked file loads twice -- and, when the two
        disagree, into two different viewers.

        `projects.md` § 3 (opening a molecule -- one door). Like the
        `/projects` route pin above, the retirement is recorded nowhere but
        here; raised as a cut candidate.
        """
        # The per-tab projects-selection shim was retired -- the sidebar
        # actions section took over (no more "Use this file" banner).
        r = web.get("/static/lib/projects-selection.js")
        assert r.status_code == 404

    @pytest.mark.parametrize("path", [
        "/molbuilder", "/structure-optimization",
        "/spectrum-calculation", "/transport-calculation",
        "/results"])
    def test_sidebar_included_in_every_tab(self, web, picker_root, path):
        """A tab shipped without the sidebar partial has NO way to open a file
        -- the sidebar is the only file-picking door, so the tab is unusable
        and nothing else in the suite would say so.

        `projects.md` § 1 (the one door) and § 4. Parametrized over every
        served tab, so a NEW tab that forgets the include is caught by adding
        one line here.
        """
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
        """The shell's layout opt-in arriving from JS instead of the server:
        the first paint uses the pre-shell geometry, so Plotly and 3Dmol
        initialise at the wrong size and stay broken until the user resizes the
        window. This bit users under the since-retired `has-projects-sidebar`
        shim.

        `projects.md` § 4. Honest note (2026-09-09): the second half of this
        test -- that the JS does not spell `setAttribute(data-sidebars)` -- is
        a `testing.md` § 3a source pin on shipped text rather than a behaviour,
        and is raised for redesign, not deletion.
        """
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
        """`/molbuilder` losing its selection bootstrap: picking an `.xyz` in
        the sidebar silently stops loading it into the viewer -- the click just
        does nothing.

        `projects.md` § 3 (opening a molecule -- the one door) and
        `model/structure.md` § 3.1 (`projects.parser.openMolecule`, reached
        through `window.molbuilder.projects`). The negative half (retired ids
        stay absent) is spelling on shipped markup and is the weaker part.
        """
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
        """A tab link pointing at a route the server does not serve -- the user
        clicks a header tab and lands on a 404.

        `tabs.md` (the canonical tab order lives in `web.tabs.TABS`) and
        `web-api.md` § 4. It checks EVERY app-tab link against the served set,
        so it is an artifact lint over the whole class rather than one link --
        the kind `testing.md` § 3b keeps.
        """
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
        """`mkdir` reporting success without creating the directory, or
        creating it somewhere other than the path it echoes back -- the sidebar
        then navigates to a folder that is not there.

        `web-api.md` § 4 (`POST /api/files/mkdir`); `job-contracts.md` § 2.5
        for the name grammar.
        """
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
        """A project directory with a space or a dot in its name breaks
        SIESTA's basename-based file discovery downstream -- the run fails much
        later, inside the engine, with nothing pointing back at the folder
        name.

        `job-contracts.md` § 2.5 (each path segment matches `[A-Za-z0-9_-]+`);
        `projects.validate_name` is the door.
        """
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
        """An ad-hoc topic at depth 2 fragments the project tree: `Raman/`
        beside `spectrum/` means the 'same analysis across structures'
        comparison no longer finds anything.

        `job-contracts.md` § 2.5 (the fixed topic vocabulary). Honest note
        (2026-09-09): the message asserted here says 'canonical six' while the
        set holds NINE topics (`projects.py:115`; the doc corrected six to nine
        on 2026-07-27), so this test currently pins the stale wording --
        reported as a code defect.
        """
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
        """The other half: a depth-2 check that refused everything would make
        the canonical layout uncreatable through the UI, and only a positive
        case separates an over-tight guard from a correct one.

        `job-contracts.md` § 2.5.
        """
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
        """A second `mkdir` on an existing folder answered 200 -- the sidebar
        reports a folder created when it merely found one, so a user who
        retypes an existing project name believes they made a new one.

        `web-api.md` § 1 status table (409 = the destination exists).
        """
        (picker_root / "preexisting").mkdir()
        r = web.post(
            "/api/files/mkdir",
            json={"parent": str(picker_root), "name": "preexisting"},
        )
        assert r.status_code == 409
        assert "already exists" in r.get_json()["error"]

    def test_mkdir_400_for_missing_name(self, web, picker_root):
        """A `name`-less request creating something anyway -- an empty segment,
        or the parent itself -- instead of a named 400.

        `web-api.md` § 1 status table.
        """
        r = web.post(
            "/api/files/mkdir", json={"parent": str(picker_root)},
        )
        assert r.status_code == 400
        assert "missing 'name'" in r.get_json()["error"]

    def test_mkdir_400_for_parent_outside_root(self, web, picker_root):
        """`mkdir` is a WRITE, so a route that skipped the fence lets a session
        create directories anywhere the server user can write.

        `web-api.md` § 2.1. What this test holds is that THIS route reaches the
        fence at all; the fence's own behaviour is `TestPathTraversalDefense`.
        """
        # Reuses the same outside-root rejection as /api/files/list.
        r = web.post(
            "/api/files/mkdir",
            json={"parent": "/etc", "name": "evil"},
        )
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]
        assert not Path("/etc/evil").exists()  # paranoia

    def test_mkdir_400_for_dot_dot_in_parent(self, web, picker_root):
        """A raw `..` in `parent` reaching resolution.

        `web-api.md` § 2.1. Honest note (2026-09-09): the parent used here
        (`<root>/..`) resolves OUTSIDE the root, so containment refuses it even
        with the string check removed, and
        `test_mkdir_400_for_parent_outside_root` already proves this route
        reaches the fence -- raised as a cut candidate.
        """
        r = web.post(
            "/api/files/mkdir",
            json={"parent": str(picker_root) + "/..",
                  "name": "anything"},
        )
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_mkdir_400_for_parent_not_a_directory(self, web, picker_root):
        """A parent that is a regular file answered as a 500 from the OS error
        rather than the usage 400 it is.

        `web-api.md` § 1 status table.
        """
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
        """A project created with a partial tree: the user picks `spectrum` in
        a later tab and it is not there, with nothing saying why.

        `job-contracts.md` § 2.5 (the project tree and its topics). This test
        loops over `CANONICAL_TOPICS` itself, so it proves the ROUTE builds
        what the tuple says -- never what the tuple should contain; the two
        tests below pin the contents.
        """
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
        """The two STORAGE topics dropping out of the canonical set -- which
        the test above cannot catch, because it reads the same tuple it checks.
        Without `structure/` there is nowhere for an uploaded geometry to land,
        and without `pseudopotential/` the project-local `.psml` cache has no
        home.

        `job-contracts.md` § 2.5 (nine topics: two storage, six run, one free-
        form).
        """
        # Both new storage-dir entries land alongside the run-topic
        # dirs as part of the canonical skeleton.
        r = web.post("/api/projects/create", json={"name": "with_storage"})
        assert r.status_code == 200
        assert (picker_root / "with_storage" / "structure").is_dir()
        assert (picker_root / "with_storage" / "pseudopotential").is_dir()

    def test_create_includes_user_freeform_topic(self, web, picker_root):
        """`user/` disappearing, or the depth rule tightening so nothing may be
        created inside it -- the user loses the one place in the tree with no
        naming rules.

        `job-contracts.md` § 2.5 (`user` is free-form: a workspace with no
        rules inside it). The second half proves the topic vocabulary binds at
        depth 1 ONLY -- `free_subdir` is legal inside `user/` and would be
        refused as a topic.
        """
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
        """A topic added to the tuple without its README leaves a new user in
        an empty folder with no idea what belongs there; a project README that
        stops naming a topic hides that topic entirely.

        `job-contracts.md` § 2.5. The loop is over `CANONICAL_TOPICS`, so a
        topic added later is covered without a test edit.
        """
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

    def test_create_returns_409_on_name_conflict(self, web, picker_root):
        """A second create over an existing project rebuilding the skeleton on
        top of it: READMEs rewritten and -- if the bootstrap ever gained a
        clean step -- real results destroyed by a name collision.

        `web-api.md` § 1 status table (409). The last assertion is the load-
        bearing one: the refusal is detection-only and the existing tree is
        untouched.
        """
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
        """A conflict check that consulted its own bookkeeping instead of the
        filesystem: a directory the user made in a shell would be
        indistinguishable from free space, and create would write into it.

        `web-api.md` § 1 status table. Honest note (2026-09-09): this is close
        to the test above and the audit records it as UNSURE rather than a cut
        -- it is the only case where the pre-existing directory was not made by
        the route.
        """
        # Same 409 path applies when the dir already exists outside
        # the /api/projects/create flow (e.g., user mkdir'd by hand).
        (picker_root / "handmade").mkdir()
        r = web.post("/api/projects/create", json={"name": "handmade"})
        assert r.status_code == 409

    def test_create_400_on_invalid_name(self, web, picker_root):
        """A project name carrying a space, a dot or a slash. The slash is the
        dangerous one: `my/proj` would create a nested path from a single
        field.

        `job-contracts.md` § 2.5 (`[A-Za-z0-9_-]+` per segment).
        """
        # validate_name regex: ^[A-Za-z0-9_-]+$ -- reject spaces, dots.
        for bad in ["my project", "my.proj", "my/proj", "weird*name", ""]:
            r = web.post("/api/projects/create", json={"name": bad})
            assert r.status_code == 400, bad

    def test_create_400_when_name_missing(self, web, picker_root):
        """A nameless create answered by making something -- a directory named
        after the empty string, or the root itself treated as the target.

        `web-api.md` § 1 status table.
        """
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
        """The create bar disappearing from the sidebar header, which is where
        all three mutation actions live.

        `projects.md` § 4. Honest note (2026-09-09): if the bar were removed
        `test_three_action_buttons_visible` fails too, so what only this test
        holds is the CONTAINER class that `projects-sidebar.css:232` styles --
        a rename would silently unstyle the header. Raised as a cut candidate
        with that loss stated.
        """
        body = web.get("/spectrum-calculation").get_data(as_text=True)
        assert 'class="ps-create-bar"' in body

    def test_three_action_buttons_visible(self, web, picker_root):
        """A button losing its id leaves the JS unable to wire it: the control
        renders and does nothing when clicked -- the failure mode with no error
        anywhere.

        `projects.md` § 4 (the sidebar's three actions, v2 2026-06-12).
        """
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
        """`write` reporting success without the bytes landing, or landing
        somewhere other than the path it echoes -- the tab says 'saved' and the
        file is not there.

        `web-api.md` § 4 (`POST /api/files/write`), `projects.md` § 3 (the
        server writes; the browser never does). The returned `mtime` is what
        the caller sends back as `expected_mtime` on the next save.
        """
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
        """A generate-and-save that clobbers by default: a second run of the
        same tab silently replaces a file the user had already edited.

        `web-api.md` § 1 status table (409) and `projects.md` § 4.1. The final
        assertion is the one that matters -- the refusal must not touch the
        file.
        """
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
        """The opt-in becoming a no-op: the caller asks for a deliberate
        replacement, gets a 200, and the old bytes stay on disk.

        `web-api.md` § 4 (`POST /api/files/write`, `overwrite`).
        """
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
        """The lost update: two tabs open the same file, both save, and the
        second overwrites the first with no sign anything happened. The
        response must also carry `actual_mtime`, or the editor cannot offer a
        reload.

        `web-api.md` § 1 status table (409); the read side of the mtime is
        `test_stat_file`.
        """
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
        """A guard that refuses even when the file has NOT changed makes
        editing impossible -- every save 409s. Only a positive case separates a
        correct comparison from a broken one.

        `web-api.md` § 4 (`POST /api/files/write`, `expected_mtime`).
        """
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
        """Files written straight into `projects/`, where the sidebar shows
        projects: a loose `.xyz` at that level looks like a project and is not
        one.

        `job-contracts.md` § 2.5 (three levels: project / topic / calculation).
        """
        # Cannot write directly into projects/ root; depth >= 1
        # required.  Keeps the root clean (only project dirs there).
        target = str(picker_root / "orphan.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "x"})
        assert r.status_code == 400
        assert "picker root" in r.get_json()["error"]
        assert not (picker_root / "orphan.txt").exists()

    def test_write_outside_root_rejected(self, web, picker_root):
        """`write` is the most dangerous route in this file -- a skipped fence
        means arbitrary file creation as the server user.

        `web-api.md` § 2.1: what this holds is that the route reaches the
        fence.
        """
        r = web.post("/api/files/write",
                     json={"path": "/etc/evil", "text": "x"})
        assert r.status_code == 400
        assert "outside every configured root" in r.get_json()["error"]

    def test_write_dot_dot_rejected(self, web, picker_root):
        """The `..` that CANCELS. `<root>/proj/../outside` resolves to
        `<root>/outside`, which is INSIDE the root, so the containment check
        passes it; only the raw-string refusal stops the write from landing
        outside the folder the user is looking at.

        `web-api.md` § 2.1. This is the one per-route `..` test whose case is
        not already covered by its outside-root sibling.
        """
        r = web.post("/api/files/write",
                     json={"path": str(picker_root) + "/proj/../outside",
                           "text": "x"})
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_write_missing_parent_dir(self, web, picker_root):
        """A save into a folder that does not exist answered by creating the
        whole chain -- a typo in a path silently makes a tree -- or by a 500
        from the OS error.

        `web-api.md` § 1 status table; `mkdir` is the route that creates
        directories.
        """
        sub = picker_root / "myproj"
        sub.mkdir()
        target = str(sub / "no" / "such" / "dir" / "file.txt")
        r = web.post("/api/files/write",
                     json={"path": target, "text": "x"})
        assert r.status_code == 400
        assert "parent directory does not exist" in r.get_json()["error"]

    def test_write_rejects_non_string_text(self, web, picker_root):
        """A JSON number or object in `text` reaching the writer: either a 500
        mid-write, or a file containing Python's repr of the object.

        `web-api.md` § 1 (the request envelope is strict where a file is
        written).
        """
        sub = picker_root / "myproj" / "topic_a"
        sub.mkdir(parents=True)
        r = web.post("/api/files/write",
                     json={"path": str(sub / "out.txt"), "text": 42})
        assert r.status_code == 400
        assert "string" in r.get_json()["error"]


# A ``TestGenerateWritesToWorkspace`` class stood here saying "Tests pin
# both layers" and holding no tests.  Both layers are pinned in
# `test_projects_public_surface_js.py`: ``saveToWorkspace`` on the sidebar
# surface, and ``safeSave`` wrapping it with the three terminal outcomes.

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
        """Deleting `water.xyz` and leaving `water.molstruct.json` behind: the
        orphan then pairs with the NEXT file that takes the name, handing a
        different structure someone else's region labels and frozen atoms.

        `projects.md` § 4.1 and `model/structure.md` § 2.4 (a structure and its
        sidecar move as one). Measured 2026-07 -- delete was the one operation
        that had not been paired.
        """
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
        """`.pdb` dropping out of the paired-suffix set -- which the `.xyz`
        test above cannot see, because both read the same set.

        `projects.md` § 4.1; the suffix set is the sidecar-pairing helper in
        `web/blueprints/files.py`.
        """
        struct, sidecar = _seed_paired(picker_root, stem="prot", ext=".pdb")
        r = self._delete(web, struct)
        assert r.status_code == 200
        assert not struct.exists()
        assert not sidecar.exists()

    def test_delete_without_sidecar_is_fine(self, web, picker_root):
        """The pairing branch turning a plain delete into a 404 or a 500 when
        there is no sidecar -- the common case broken by the code that handles
        the rare one. `sidecar_removed: null` is what the sidebar reports.

        `projects.md` § 4.1.
        """
        struct = picker_root / "lonely.xyz"
        struct.write_text("1\nx\nH 0 0 0\n")
        r = self._delete(web, struct)
        assert r.status_code == 200
        assert r.get_json()["sidecar_removed"] is None
        assert not struct.exists()

    def test_deleting_the_sidecar_directly_is_single_file(self, web, picker_root):
        """Deleting a sidecar taking its structure with it: the user asked to
        drop the labels and loses the geometry.

        `projects.md` § 4.1 -- pairing triggers on the STRUCTURE file only, so
        naming the sidecar is a single-file operation, matching rename.
        """
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
        """Renaming `water.xyz` to `bridge.xyz` and orphaning
        `water.molstruct.json`: the next load finds no sidecar and the user's
        regions, frozen atoms and cell disappear with no error.

        `projects.md` § 4.1 and `model/structure.md` § 2.4. Measured
        2026-06-12. The payload assertion matters too -- the sidecar must be
        MOVED, not regenerated.
        """
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
        """`.pdb` dropping out of the paired-suffix set on the rename path
        specifically.

        `projects.md` § 4.1.
        """
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
        """A move that copies without removing the source, or removes without
        landing the destination: the first silently duplicates results, the
        second loses them.

        `web-api.md` § 4 (`POST /api/files/move`).
        """
        src = picker_root / "config.json"   # seeded by fixture
        dst_dir = picker_root / "spectrum"  # seeded by fixture
        r = self._move(web, src, dst_dir)
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json()["ok"] is True
        assert not src.exists()
        assert (dst_dir / "config.json").exists()

    def test_move_with_rename(self, web, picker_root):
        """`new_name` ignored on a move: the file lands under its old name, so
        a move-and-rename that was meant to avoid a collision creates one.

        `web-api.md` § 4 (`POST /api/files/move`).
        """
        src = picker_root / "config.json"
        dst_dir = picker_root / "spectrum"
        r = self._move(web, src, dst_dir, new_name="renamed.json")
        assert r.status_code == 200
        assert (dst_dir / "renamed.json").exists()

    def test_move_xyz_takes_sidecar(self, web, picker_root):
        """Moving a structure out from under its sidecar -- the same orphaning
        as rename, on the path that crosses directories.

        `projects.md` § 4.1, `model/structure.md` § 2.4.
        """
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
        """Moving a directory through this route: it would relocate a whole
        calculation (or a topic) with none of the layout checks the delete
        route applies, from a single drag.

        `web-api.md` § 4; `job-contracts.md` § 2.5 (the tree's three levels are
        structural).
        """
        src_dir = picker_root / "spectrum"
        other = picker_root / "other"
        other.mkdir()
        r = self._move(web, src_dir, other)
        assert r.status_code == 400
        assert "directories" in r.get_json()["error"]

    def test_move_refuses_when_dest_missing(self, web, picker_root):
        """A move into a directory that does not exist answered by creating it
        -- a typo in `dest_dir` scatters files into new folders -- or by a 500.

        `web-api.md` § 1 status table. Honest note: the assertion accepts 400
        or 404, so it does not pin WHICH refusal; recorded as a design weakness
        on 2026-09-09.
        """
        src = picker_root / "config.json"
        r = self._move(web, src, picker_root / "does-not-exist")
        assert r.status_code in (400, 404)

    def test_move_refuses_overwrite(self, web, picker_root):
        """A move that silently replaces a file at the destination: the
        overwritten file is gone, and neither name in the request refers to it.

        `web-api.md` § 1 status table (409).
        """
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
        """A copy that MOVES (source gone) or that lands a truncated file. The
        byte comparison is what separates a real copy from a created-and-empty
        one.

        `web-api.md` § 4 (`POST /api/files/copy`).
        """
        src = picker_root / "config.json"
        original = src.read_text()
        dst_dir = picker_root / "spectrum"
        r = self._copy(web, src, dst_dir)
        assert r.status_code == 200, r.get_data(as_text=True)
        assert src.exists()           # source preserved
        assert (dst_dir / "config.json").read_text() == original

    def test_copy_with_rename(self, web, picker_root):
        """`new_name` ignored on copy, which turns every rename-copy into a
        same-name copy and therefore a 409 the user did not ask for.

        `web-api.md` § 4 (`POST /api/files/copy`).
        """
        src = picker_root / "config.json"
        dst_dir = picker_root / "spectrum"
        r = self._copy(web, src, dst_dir, new_name="backup.json")
        assert r.status_code == 200
        assert (dst_dir / "backup.json").exists()
        assert src.exists()

    def test_copy_same_dir_requires_new_name(self, web, picker_root):
        """A copy onto itself: source and destination are one path, so an
        implementation that opens the destination for writing first truncates
        the very file it is copying.

        `web-api.md` § 4 (`POST /api/files/copy`).
        """
        src = picker_root / "config.json"
        r = self._copy(web, src, picker_root)
        # Same path = source.  Refused.
        assert r.status_code == 400

    def test_copy_xyz_takes_sidecar(self, web, picker_root):
        """A copied structure arriving without its sidecar: the copy silently
        loses the regions and frozen atoms, and the difference from the
        original surfaces only in a later calculation. The verbatim payload
        comparison is what proves the sidecar was COPIED rather than rebuilt.

        `projects.md` § 4.1, `model/structure.md` § 2.4.
        """
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
        """A copy that overwrites a file at the destination -- the same silent
        loss as move, with the source still present to make it look harmless.

        `web-api.md` § 1 status table (409).
        """
        src = picker_root / "config.json"
        dst_dir = picker_root / "spectrum"
        (dst_dir / "config.json").write_text("existing\n")
        r = self._copy(web, src, dst_dir)
        assert r.status_code == 409

    def test_copy_refuses_directory(self, web, picker_root):
        """A recursive directory copy from one click: on a run directory that
        duplicates gigabytes of restart files inside a request thread.

        `web-api.md` § 4 and § 1a (the three-second rule).
        """
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
        """Delete reporting success without removing the file -- the sidebar
        row disappears and comes back on refresh -- or taking the parent
        directory with it.

        `web-api.md` § 4 (`DELETE /api/files/delete`).
        """
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
        """An empty directory refused as 'not a file' would leave the user no
        way to remove a folder they made by mistake; the recursive flag exists
        for NON-empty ones.

        `web-api.md` § 4 (`DELETE /api/files/delete`).
        """
        target = picker_root / "proj" / "user" / "scratch"
        target.mkdir(parents=True)
        r = self._delete(web, target)
        assert r.status_code == 200
        assert not target.exists()

    def test_delete_recursive_removes_non_empty_dir(self, web, picker_root):
        """`recursive=true` honoured only one level deep, leaving a half-
        emptied tree the sidebar still shows.

        `web-api.md` § 4 (`DELETE /api/files/delete`).
        """
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
        """A DELETE with no JSON at all reaching `body['path']`: a 500 on a
        request that is merely malformed, and -- if the code defaulted instead
        -- a delete of whatever that default resolves to.

        `web-api.md` § 1 status table; the branch is the route's
        `request.get_json(silent=True) or {}`.
        """
        # No JSON body at all.
        r = web.delete("/api/files/delete")
        assert r.status_code == 400
        assert "path" in r.get_json()["error"]

    def test_delete_missing_path_400(self, web):
        """A body carrying `recursive` but no `path` -- the dangerous shape,
        because everything else a delete needs is present.

        `web-api.md` § 1 status table; the route's `not raw_path.strip()`
        branch, which also covers a whitespace-only path.
        """
        r = web.delete("/api/files/delete", json={"recursive": True})
        assert r.status_code == 400
        assert "path" in r.get_json()["error"]

    def test_delete_nonexistent_path_404(self, web, picker_root):
        """A delete of something already gone answered 200, so the UI reports a
        removal that never happened and a client retrying a failed delete
        cannot tell success from absence.

        `web-api.md` § 1 status table.
        """
        target = picker_root / "proj" / "ghost.xyz"
        # ``ghost.xyz``'s parent ``proj`` doesn't exist either; the
        # resolver still computes a path inside the root, and the
        # existence check returns 404.
        (picker_root / "proj").mkdir(parents=True)
        r = self._delete(web, target)
        assert r.status_code == 404

    def test_delete_outside_root_rejected(self, web, picker_root):
        """The worst case in this file: a delete route that skipped the fence
        removes anything the server user can remove. The comment in the body is
        load-bearing -- `tmp_path` IS the picker root here, so an outside path
        must be built above it.

        `web-api.md` § 2.1.
        """
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
        """A raw `..` in a delete path.

        `web-api.md` § 2.1. Honest note (2026-09-09): the path used here
        resolves outside the root, so containment refuses it without the string
        check, and `test_delete_outside_root_rejected` already proves this
        route reaches the fence -- raised as a cut candidate. The distinct case
        (a `..` that cancels back inside the root) is pinned on `write`, not
        here.
        """
        # Defense in depth: ``..`` in the raw string is rejected.
        r = web.delete(
            "/api/files/delete",
            json={"path": str(picker_root) + "/proj/../../etc"},
        )
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_delete_picker_root_itself_rejected(self, web, picker_root):
        """`recursive=true` on `projects/` itself -- one request removes every
        project on the machine.

        `job-contracts.md` § 2.5 (depth 0 is the tree). The depth rule is the
        route's own decision, not the fence's, which is why it needs its own
        test.
        """
        # Cannot delete projects/ -- depth-0 protection.
        r = self._delete(web, picker_root, recursive=True)
        assert r.status_code == 400
        err = r.get_json()["error"]
        assert "root" in err.lower()

    def test_delete_canonical_topic_dir_rejected(self, web, picker_root):
        """Removing `projects/<proj>/spectrum/` orphans the layout: every later
        tab that resolves a run by topic finds nothing. The refusal must hold
        even with `recursive=true`, which is exactly when the user believes
        they authorised it.

        `job-contracts.md` § 2.5. The final assertion -- the directory still
        exists -- is what separates a refusal from a report.
        """
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
        """`user` being treated as an ordinary folder because it is the free-
        form one. It is a canonical topic, and losing it takes the whole
        workspace with it.

        `job-contracts.md` § 2.5 (nine topics, `user` among them; added
        2026-05-16).
        """
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
        """The topic guard applied to everything BELOW the topic: a user could
        then never delete a run directory (`spectrum/water_v1/`), which is the
        normal workflow.

        `job-contracts.md` § 2.5 -- the protection is at depth 2 only.
        """
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
        """The topic guard matching on NAME alone: a plain file called
        `spectrum` at depth 2 would become undeletable, with a message about
        the project layout that does not apply to it.

        `job-contracts.md` § 2.5 -- the guard is about directories, because
        only a directory can hold the layout.
        """
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
        """A non-empty directory removed without the caller asking for
        recursion: the user meant to delete a folder they believed empty and
        takes its contents with it.

        `web-api.md` § 1 status table (409). Both the directory and its content
        must survive the refusal.
        """
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
        """The depth-2 topic guard leaking up to depth 1: a project could never
        be removed through the UI even when the user explicitly asked to.

        `job-contracts.md` § 2.5 -- depth 1 is a project, and deleting one
        deliberately is legitimate.
        """
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
        """An upload that lands truncated, or under a different name than it
        reports. The byte comparison and the echoed path are what the caller
        uses to confirm the save.

        `web-api.md` § 4 (`POST /api/files/upload`), `projects.md` § 4.
        """
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
        """No `target_dir` answered by writing somewhere -- the CWD, or the
        first root -- instead of a named refusal.

        `web-api.md` § 1 status table.
        """
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
        """A multipart body with no `file` part reaching
        `request.files['file']`: a 500 (KeyError) on a request that is merely
        incomplete, which is what a mis-wired form sends.

        `web-api.md` § 1 status table.
        """
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
        """Uploads landing directly in `projects/`, beside the project
        directories, where nothing later can tell them apart from one.

        `job-contracts.md` § 2.5; the same depth rule `write` obeys.
        """
        # Uploading directly into the picker root (depth 0) is forbidden;
        # parallels the same rule on /api/files/write.
        r = self._post(web, picker_root, "stray.txt")
        assert r.status_code == 400
        assert "subdirectory" in r.get_json()["error"]

    def test_upload_to_missing_dir_400(self, web, picker_root):
        """An upload into a directory that is not there answered by creating
        the path, so a stale `target_dir` silently makes a folder instead of
        reporting that the folder is gone.

        `web-api.md` § 1 status table. Honest note: the assertion accepts 400
        or 404 and so does not pin which refusal fires.
        """
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
        """`target_dir` pointing at a file: without the directory check the
        upload either 500s on the OS error or, worse, writes THROUGH the name
        and destroys the file that was there.

        `web-api.md` § 1 status table.
        """
        # target_dir is a file, not a directory.
        target = picker_root / "proj" / "spectrum"
        target.mkdir(parents=True)
        (target / "blob.bin").write_bytes(b"x")
        r = self._post(web, target / "blob.bin", "file.txt")
        assert r.status_code == 400
        assert "directory" in r.get_json()["error"]

    def test_upload_outside_root_rejected(self, web, tmp_path):
        """Nothing -- and the audit of 2026-09-09 raises this as a CUT
        candidate rather than invent a purpose for it. `tmp_path` is the same
        directory the `picker_root` fixture wires as the root, so `tmp_path /
        'elsewhere'` is INSIDE the fence: the 400 observed here is 'target_dir
        does not exist or is not a directory', the case
        `test_upload_to_missing_dir_400` already covers, and the error-text
        assertion passes only because pytest's temp directory is named after
        this test and therefore contains the words 'outside' and 'root'.
        Measured by replaying the request on 2026-09-09.

        The property is real (`web-api.md` § 2.1) and deserves a test;
        `test_delete_outside_root_rejected` shows how to build a path that is
        genuinely outside the root.
        """
        # Absolute path completely outside the picker root.
        r = self._post(web, tmp_path / "elsewhere", "file.txt")
        assert r.status_code == 400
        assert "outside" in r.get_json()["error"] or "root" in r.get_json()["error"]

    def test_upload_dot_dot_in_target_rejected(self, web, picker_root):
        """A raw `..` in `target_dir` reaching resolution.

        `web-api.md` § 2.1. Honest note (2026-09-09): this target resolves
        outside the root, so containment refuses it without the string check --
        but because the sibling outside-root test cannot fail (see above), this
        is currently the ONLY test proving upload reaches the fence at all.
        Keep it until that one is repaired.
        """
        # Defense in depth: '..' in raw target_dir string is rejected
        # even though the resolution step would also catch it.
        r = self._post(web, str(picker_root) + "/proj/../..", "file.txt")
        assert r.status_code == 400
        assert ".." in r.get_json()["error"]

    def test_upload_existing_filename_409(self, web, picker_root):
        """An upload silently replacing a file of the same name: the user drops
        a second `geom.xyz` into a run folder and the first is gone with no
        prompt.

        `web-api.md` § 1 status table (409). Refusal is the default;
        `overwrite` is the opt-in.
        """
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
        """A filename carrying a SPACE being accepted -- despite this test's
        name, which claims the path-separator case. The body sends 'has
        space.txt'; separators are stripped by `os.path.basename` before
        validation and are covered by `test_upload_strips_client_path_prefix`.

        `web-api.md` § 4 (`POST /api/files/upload`, the filename regex). The
        audit of 2026-09-09 records the name / body mismatch as a design
        defect: nothing in the suite drives a separator that survives
        `basename`, and the space case is already carried by
        `test_write_upload_filename_parity`.
        """
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
        """A dotfile uploaded into a folder whose listing filters dotfiles out:
        the file exists, occupies the name, and the user cannot see it to
        delete it.

        `projects.md` § 4 (hidden entries are never listed); the filename
        anchor is `^[A-Za-z0-9]`.
        """
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
        """A browser sending the full client path as the filename (`/tmp/from-
        client/water.xyz`) either being refused as invalid -- an upload that
        works in one browser and not another -- or reaching the writer with
        separators still in it.

        `web-api.md` § 4 (`POST /api/files/upload`). This is the real path-
        separator case; `test_upload_filename_with_path_separator_400` does not
        cover it despite its name.
        """
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
        """A dialog that promises auto-rename over a `write` that 409s instead:
        the export appears to fail for a reason the user was told would not
        happen. The original must also survive -- an auto-rename that clobbers
        is worse than the 409 it replaced.

        `projects.md` § 5 (`safeSave` confirms the `<stem>-2<ext>` the server
        picked); Phase 6e second review, BOMB #11.
        """
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
        """A suffix picker that tries `-2` once and gives up: the third export
        of the same default filename fails.

        `projects.md` § 5.
        """
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
        """`auto_rename` renaming unconditionally -- every export lands as
        `<stem>-2` even in an empty folder, so the name the dialog showed is
        never the name on disk.

        `projects.md` § 5.
        """
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

    def test_preview_modal_starts_hidden(self, web, picker_root):
        """The preview modal painting over the page for one frame on every
        load, before the JS hides it.

        `projects.md` § 4. Honest note (2026-09-09): the class docstring says
        the behaviour is exercised by 'the deferred Playwright suite' -- a
        suite that does not exist -- which `testing.md` § 3a.1 reads as a
        verdict rather than a caveat: write the e2e that visits the page, then
        retire this markup pin.
        """
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
        """A second root re-entering at the SOURCE -- the CWD, or an operator-
        supplied list -- which every route-level test in this file is blind to,
        because they monkeypatch this method away.

        `job-contracts.md` § 2.5 (`projects/` is the tree) and `web-api.md` §
        4.
        """
        from molbuilder.diagnostics import Capabilities
        caps = Capabilities(runtime_config={})
        roots = caps.file_picker_roots()
        assert len(roots) == 1
        path, label = roots[0]
        assert label == "projects"
        assert str(path).endswith("/projects")

    def test_stale_file_picker_section_is_refused_by_name(self, tmp_path):
        """A retired config section read and silently dropped: an operator's
        `file_picker.roots` block sits in the file looking effective while the
        picker ignores it -- which is what happened to a real config before the
        unknown-keys guard (2026-08-12).

        `configuration.md` § 4 (the `runtime_config._SECTIONS` registry makes
        "known" one total list, so anything outside it is refused). Precisely:
        `file_picker` is NOT a named gravestone the way `secret_key_file` is --
        it falls to the generic unknown-top-level-key refusal, which quotes the
        offending key, which is why matching on the name passes.
        """
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


class TestTheDownloadButtonSaysWhatItIsDoing:
    """The sidebar's Download control (user, 2026-08-29): the message
    lives INSIDE the button while the server compresses, and the button
    is unclickable until the browser's save has started -- which is
    only knowable because the build has its own door."""

    @staticmethod
    def _src(web):
        return web.get(
            "/static/lib/projects/mutation-bar.js").get_data(as_text=True)


class TestDownloadZip:
    """The *carry a calculation without ssh* door (user, 2026-08-28),
    now in two halves (user, 2026-08-29): ``POST /api/files/zip_prepare``
    builds the archive and answers what it is, ``GET
    /api/files/download_zip?token=`` streams it.  The split exists so
    the sidebar button can say *Zipping…* and stay unclickable until
    the save starts -- a plain navigation reports neither end of a
    build that takes minutes.
    """

    @staticmethod
    def _prepare(web, path):
        return web.post("/api/files/zip_prepare", json={"path": str(path)})

    def _fetch(self, web, path):
        """prepare -> stream, the pair the button walks."""
        pre = self._prepare(web, path)
        assert pre.status_code == 200, pre.get_json()
        body = pre.get_json()
        assert body["ok"] is True
        r = web.get("/api/files/download_zip",
                    query_string={"token": body["token"]})
        return body, r

    def test_a_directory_streams_as_its_named_zip(self, web, picker_root):
        """An archive whose members sit at the top level rather than under the
        folder's name -- unzipping on the far machine scatters a run's files
        into the current directory -- and bytes that do not survive the round
        trip, which is a corrupt result at the other end.

        `web-api.md` § 4 (`zip_prepare` / `download_zip`) and `projects.md` § 4
        (user, 2026-08-28: carry a calculation to another machine without ssh).
        """
        import io
        import zipfile
        body, r = self._fetch(web, picker_root / "spectrum" / "BDT")
        assert r.status_code == 200
        assert r.data[:2] == b"PK", "not a zip"
        assert 'filename=BDT.zip' in r.headers["Content-Disposition"]
        with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
            names = sorted(zf.namelist())
            assert names == ["BDT/.hidden", "BDT/water_spectra.spectra.json"], (
                f"member paths must ride under the folder's name: {names}")
            assert zf.read("BDT/water_spectra.spectra.json") \
                == b'{"schema_version": 2}\n', "bytes must survive verbatim"
        assert body["name"] == "BDT.zip"
        assert body["files"] == 2 and body["bytes"] > 0
        assert body["skipped"] == 0

    def test_the_token_is_single_use(self, web, picker_root):
        """The temp file goes with the response, so a replayed token
        must be refused by name rather than fail on a missing file."""
        pre = self._prepare(web, picker_root / "spectrum" / "BDT")
        token = pre.get_json()["token"]
        first = web.get("/api/files/download_zip",
                        query_string={"token": token})
        assert first.status_code == 200
        again = web.get("/api/files/download_zip",
                        query_string={"token": token})
        assert again.status_code == 404
        assert "already downloaded" in again.get_json()["error"]

    def test_an_unknown_token_is_refused(self, web, picker_root):
        """A token the server never issued being answered with anything but a
        refusal: a guessed or ignored token is a read of an arbitrary archive.

        `web-api.md` § 4 -- the archive is streamed BY TOKEN, single-use.
        """
        r = web.get("/api/files/download_zip",
                    query_string={"token": "deadbeef"})
        assert r.status_code == 404
        assert r.get_json()["ok"] is False

    def test_the_projects_root_itself_is_refused(self, web, picker_root):
        """Zipping `projects/` itself: every project on the machine compressed
        inside one request thread -- both a denial of service and an archive
        nobody asked for.

        `web-api.md` § 4 and § 1a (the three-second rule); the message must say
        what was refused.
        """
        r = self._prepare(web, picker_root)
        assert r.status_code == 400
        assert "refusing to zip a projects root" in r.get_json()["error"]

    def test_a_file_names_the_single_file_door(self, web, picker_root):
        """A single file answered with a one-member zip: the user asked for the
        file and gets an archive to unpack. The refusal must NAME
        `/api/files/download`, or the sidebar has nowhere to send them.

        `web-api.md` § 4 (`/api/files/download` is the single-file door).
        """
        r = self._prepare(web, picker_root / "water.xyz")
        assert r.status_code == 400
        assert "/api/files/download" in r.get_json()["error"]

    def test_outside_the_fence_is_refused(self, web, picker_root):
        """`zip_prepare` is a READ of a whole tree, so a skipped fence exports
        an arbitrary directory as a downloadable archive -- the most complete
        exfiltration available in this file.

        `web-api.md` § 2.1. Honest note (2026-09-09): the assertion accepts 400
        or 403 because § 1's status table assigns a path escape to 403 while
        the fence raises 400 -- reported as a doc / code disagreement.
        """
        r = self._prepare(web, "/etc")
        assert r.status_code in (400, 403)
        assert r.get_json()["ok"] is False

    def test_missing_is_a_404(self, web, picker_root):
        """A missing directory answered with an empty archive: the far machine
        receives a valid zip with nothing in it and the user finds out after
        the transfer.

        `web-api.md` § 1 status table.
        """
        r = self._prepare(web, picker_root / "spectrum" / "nope")
        assert r.status_code == 404

    def test_symlinks_escape_skipped_inside_followed(self, web, picker_root):
        """Two failures in opposite directions. A symlink pointing OUT of the
        tree, if followed, exports whatever it names (`/etc/hostname` here); an
        inside-the-tree symlink, if skipped or stored as a link, leaves the far
        machine unpacking a dangling name where a real file's bytes are needed.
        `skipped` is what tells the user something was left behind.

        `web-api.md` § 4 (the zip row) and § 2.1; user, 2026-08-29.
        """
        import io
        import zipfile
        d = picker_root / "spectrum" / "linked"
        d.mkdir()
        (d / "real.txt").write_text("kept\n")
        (d / "escape").symlink_to("/etc/hostname")          # outside -> skip
        (d / "warm").symlink_to(picker_root / "water.xyz")  # inside -> follow
        body, r = self._fetch(web, d)
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
        assert body["skipped"] == 1

    def test_storage_never_enters_an_archive(self, web, picker_root):
        """User 2026-08-28 (*a clean run structure*), extended
        2026-08-29 (*a pure execution dir*): the archive carries the
        folder as it stands NOW.  Three storage subtrees stay out --
        the workspace store, and a checkpoint folder's git repo and
        its large-binary archive -- while ordinary run files, engine
        restart files included, ride along.

        Not through ``git archive``: a checkpoint keeps big files OUT
        of git on purpose (they live in .binsnapshots), so a git export
        would drop exactly the heavy outputs the far machine needs.
        """
        import io
        import zipfile
        d = picker_root / "spectrum" / "BDT"
        (d / ".molbuilder_workspace" / "states").mkdir(parents=True)
        (d / ".molbuilder_workspace" / "states" / "tab.json").write_text("{}\n")
        (d / ".git" / "objects").mkdir(parents=True)
        (d / ".git" / "objects" / "ab12").write_bytes(b"history\n")
        (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (d / ".binsnapshots").mkdir()
        (d / ".binsnapshots" / "big.DM.zst").write_bytes(b"snapshot\n")
        (d / "run.DM").write_bytes(b"restart\n")
        body, r = self._fetch(web, d)
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
            names = zf.namelist()
        for storage in (".molbuilder_workspace", ".git", ".binsnapshots"):
            assert not any(storage in n for n in names), (
                f"{storage} leaked into the archive: {names}")
        assert "BDT/run.DM" in names, (
            "an engine restart file is run data -- the far machine needs it")
        assert "BDT/.hidden" in names, (
            "only STORAGE is excluded, not every dotfile")
        assert sorted(body["excluded"]) == [
            ".binsnapshots", ".git", ".molbuilder_workspace"]

    def test_a_current_big_result_rides_even_when_snapshotted(
            self, web, picker_root):
        """The exclusion is by DIRECTORY, never by size or suffix
        (user, 2026-08-29: *we don't want to lose the big result
        files*).  A checkpointed run has its heavy outputs in TWO
        places -- the live file in the run directory, and copies under
        .binsnapshots keyed by state.  The live one is current result
        and must travel; the copies are history and must not."""
        import io
        import zipfile
        d = picker_root / "spectrum" / "BDT"
        heavy = b"density-matrix bytes\n" * 500
        (d / "siesta.DM").write_bytes(heavy)
        snap = d / ".binsnapshots" / "a1b2c3"
        snap.mkdir(parents=True)
        (snap / "siesta.DM").write_bytes(b"an OLDER density matrix\n")
        _body, r = self._fetch(web, d)
        with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
            names = zf.namelist()
            assert "BDT/siesta.DM" in names, (
                "the CURRENT big result was dropped -- the far machine "
                "would restart from nothing")
            assert zf.read("BDT/siesta.DM") == heavy, (
                "the archived copy shadowed the live file")
        assert not any(".binsnapshots" in n for n in names), (
            "the snapshot history rode along after all")
