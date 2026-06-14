"""End-to-end runtime tests for the v2 paginated source inspector.

The 2026-06-02 (task #119) rewrite replaced the single-shot 16 MB
``ctx.readFile`` with chunked ``/api/files/read_range`` fetches:

  * initial mount loads ``PAGE_BYTES`` (256 KB) of the head.
  * scrolling within ``NEXT_PAGE_TRIGGER_PX`` of the bottom triggers
    the next forward chunk (``mode === "head"``).
  * "Jump to end" switches to ``mode === "tail"`` and loads the
    final ``PAGE_BYTES``; no further auto-load fires in tail mode.

The server side is covered by ``tests/test_web_files.py::
TestFilesReadRange`` (15 tests).  This file pins the client wiring:
the inspector's DOM scaffold, status text, scroll-driven pagination,
and the head→tail mode switch.  Without these tests, a future
refactor (eg. someone hooking ``/api/files/read`` back up "for
simplicity") would silently regress to v1 behaviour.

Approach mirrors ``test_inspector_registry_e2e.py``: a real Flask
server + a tmp_path fixture registered as a picker root so the new
range-read endpoint accepts the path.
"""
from __future__ import annotations

import threading

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Pin ``tmp_path`` as the only Capabilities picker root so
    ``/api/files/read_range`` accepts files written under it.

    Same shape as the helper in ``test_molbuilder_e2e.py`` -- kept local
    so this test module stays self-contained (no cross-file fixture
    import).
    """
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)


@pytest.fixture
def big_log_file(tmp_path, monkeypatch):
    """Write a ~700 KB ``.log`` file -- big enough to need three
    forward pages (256 + 256 + ~188 KB) without being so large it
    slows the test.  Returns the absolute path string.

    Each line is numbered + padded so a reader can visually verify
    which chunk a slice came from (and so the test can grep for known
    sentinels).
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "big.log"
    lines = []
    # 7000 lines * ~100 bytes ~= 700 KB.
    for i in range(7000):
        lines.append(
            f"line {i:06d}  "
            + ("x" * 80)
            + f"  end-{i:06d}"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def small_text_file(tmp_path, monkeypatch):
    """A tiny ``.txt`` file (a few KB) that fits inside a single
    head page -- exercises the eof-on-first-load path."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "small.txt"
    p.write_text("hello world\n" * 50, encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _open_results(page, base_url):
    """Navigate to /results + capture JS errors.  Same shape as
    ``test_inspector_registry_e2e.py::_open_results``."""
    errors = []
    page.on("pageerror", lambda exc: errors.append(("pageerror", str(exc))))
    page.on("console", lambda msg: (
        errors.append(("console.error", msg.text))
        if msg.type == "error" else None
    ))
    page.goto(f"{base_url}/results")
    page.wait_for_selector("#inspector-host", timeout=5000)
    page.wait_for_function(
        "() => window.molbuilder "
        "&& window.molbuilder.inspectors "
        "&& window.molbuilder.inspectors.list().length >= 4"
    )
    return errors


def _mount_source(page, path):
    """Mount the source inspector for ``path`` against
    ``#inspector-host`` via the public registry API.  Stashes the
    handle on window so the test can dispose it cleanly later."""
    page.evaluate(
        """(p) => {
            const host = document.getElementById("inspector-host");
            const reg  = window.molbuilder.inspectors;
            const ctx  = reg.createDefaultContext(host);
            window._handle = reg.mount(host, p, ctx);
        }""",
        path,
    )
    # The inspector's mount() is synchronous (scaffold goes into the
    # host immediately); the first range fetch is async.
    page.wait_for_selector(".source-card .source-scroller", timeout=3000)


def _wait_loaded(page, *, eof=None, mode=None, timeout_ms=5000):
    """Wait for the inspector to finish its in-flight fetch.

    Pre-2026-06-02 there was no client-observable "is the fetch done"
    signal except via the status text -- now we read the live state
    off the DOM: ``.source-status`` flips from "Loading..." to
    "Showing head/tail: ... KB of ... KB (...%)" once the chunk
    lands.  Optionally also gate on the eof/mode contract.
    """
    page.wait_for_function(
        """() => {
            const el = document.querySelector(".source-status");
            if (!el) return false;
            return el.textContent.trim().startsWith("Showing ");
        }""",
        timeout=timeout_ms,
    )
    if eof is True:
        page.wait_for_function(
            """() => document.querySelector(".source-jump-end").disabled""",
            timeout=timeout_ms,
        )
    if mode == "tail":
        page.wait_for_function(
            """() => document.querySelector(".source-status")
                          .textContent.includes("tail")""",
            timeout=timeout_ms,
        )


# --------------------------------------------------------------------- #
#  Tests                                                                #
# --------------------------------------------------------------------- #


class TestInitialChunk:
    """First page rendered correctly + status reflects partial load."""

    def test_initial_load_renders_first_chunk_of_big_file(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        # The pre.source-body now holds the first PAGE_BYTES.
        body_len = page.evaluate(
            "() => document.querySelector('.source-body').textContent.length"
        )
        # PAGE_BYTES is 256 * 1024 = 262144.  ASCII -> 1:1 byte/char.
        # Allow a small UTF-8 trim margin (server may shave up to 3
        # bytes).  Anything >= 262141 is a full page.
        assert 262141 <= body_len <= 262144, (
            f"first chunk length {body_len} is not a full PAGE_BYTES "
            f"(expected ~262144 bytes for ASCII content)"
        )

        # First line should be ``line 000000`` -- forward pagination
        # starts at byte 0.
        first = page.evaluate(
            "() => document.querySelector('.source-body')"
            ".textContent.slice(0, 20)"
        )
        assert first.startswith("line 000000"), (
            f"first chunk does not start at byte 0: {first!r}"
        )

    def test_status_reports_partial_progress_for_big_file(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        status = page.locator(".source-status").inner_text()
        # Sanity: "Showing head: X KB of Y KB (Z%)" -- partial means
        # X < Y and Z < 100.
        assert "head" in status
        assert "KB of" in status
        assert "(100%)" not in status, (
            f"big file should report partial progress, got: {status!r}"
        )

    def test_jump_to_end_button_enabled_when_more_remains(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        disabled = page.evaluate(
            "() => document.querySelector('.source-jump-end').disabled"
        )
        assert disabled is False, (
            "jump-end must be enabled when the head page didn't reach EOF"
        )


class TestSmallFileFitsInOnePage:
    """When the entire file fits in PAGE_BYTES, the inspector should
    mark itself as fully-loaded + disable the Jump-to-end button."""

    def test_small_file_loads_in_one_page(
            self, page, flask_server, small_text_file):
        _open_results(page, flask_server)
        _mount_source(page, small_text_file)
        _wait_loaded(page, eof=True)

        status = page.locator(".source-status").inner_text()
        assert "(100%)" in status, (
            f"small file should report 100%, got: {status!r}"
        )

        # Content shows up verbatim.
        body = page.evaluate(
            "() => document.querySelector('.source-body').textContent"
        )
        assert body.startswith("hello world\n")
        assert body.count("hello world\n") == 50


class TestScrollDrivenPagination:
    """Scrolling to the bottom of the viewport must trigger the next
    page load (head mode)."""

    def test_scroll_to_bottom_loads_next_page(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        before_len = page.evaluate(
            "() => document.querySelector('.source-body').textContent.length"
        )

        # Force the scroller all the way to the bottom + fire scroll.
        # JS programmatic scrollTop = scrollHeight reliably puts us
        # within NEXT_PAGE_TRIGGER_PX of the bottom; the inspector's
        # scroll listener does the rest.
        page.evaluate("""() => {
            const sc = document.querySelector(".source-scroller");
            sc.scrollTop = sc.scrollHeight;
            sc.dispatchEvent(new Event("scroll"));
        }""")

        # Wait for the body to grow.  Allow up to ~5s for the fetch
        # + DOM update.
        page.wait_for_function(
            f"() => document.querySelector('.source-body')"
            f".textContent.length > {before_len + 1000}",
            timeout=5000,
        )

        after_len = page.evaluate(
            "() => document.querySelector('.source-body').textContent.length"
        )
        assert after_len >= before_len + 200_000, (
            f"second page should add ~PAGE_BYTES; "
            f"before={before_len} after={after_len}"
        )

    def test_status_still_head_after_one_scroll_page(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        before_len = page.evaluate(
            "() => document.querySelector('.source-body').textContent.length"
        )
        page.evaluate("""() => {
            const sc = document.querySelector(".source-scroller");
            sc.scrollTop = sc.scrollHeight;
            sc.dispatchEvent(new Event("scroll"));
        }""")
        page.wait_for_function(
            f"() => document.querySelector('.source-body')"
            f".textContent.length > {before_len + 1000}",
            timeout=5000,
        )

        status = page.locator(".source-status").inner_text()
        # Mode must still be "head" (forward pagination); "tail"
        # only happens on the Jump-to-end click.
        assert "head" in status and "tail" not in status, (
            f"mode flipped unexpectedly: {status!r}"
        )


class TestJumpToEndSwitchesToTail:
    """Clicking the Jump-to-end button switches the inspector to
    tail mode and disables further auto-load."""

    def test_click_jump_to_end_loads_tail_chunk(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        page.locator(".source-jump-end").click()
        _wait_loaded(page, mode="tail", eof=True)

        # Last line in big.log is ``line 006999  xxxxxxxxxx...  end-006999``;
        # the tail page must contain it.
        body = page.evaluate(
            "() => document.querySelector('.source-body').textContent"
        )
        assert "line 006999" in body, (
            "tail chunk should contain the very last line of the file"
        )
        assert "end-006999" in body

    def test_tail_disables_jump_to_end_button(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        page.locator(".source-jump-end").click()
        _wait_loaded(page, mode="tail", eof=True)

        disabled = page.evaluate(
            "() => document.querySelector('.source-jump-end').disabled"
        )
        assert disabled is True, (
            "after Jump-to-end the button must be disabled "
            "(we're already at the end)"
        )

    def test_tail_scrolls_to_bottom_of_viewport(
            self, page, flask_server, big_log_file):
        """After Jump-to-end, the scroller should be at the bottom so
        the user lands on the most recent line, not the chunk start.
        """
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        page.locator(".source-jump-end").click()
        _wait_loaded(page, mode="tail", eof=True)

        # Allow a small layout-rounding tolerance: scrollTop +
        # clientHeight should be at scrollHeight (within a few px).
        at_bottom = page.evaluate("""() => {
            const sc = document.querySelector(".source-scroller");
            const slack = (sc.scrollHeight - sc.scrollTop - sc.clientHeight);
            return slack < 10;
        }""")
        assert at_bottom is True, (
            "tail mode must place the scroller at the bottom"
        )


class TestDispose:
    """Regression: dispose() removes the scroll + click listeners so
    a re-mounted host doesn't accumulate handlers."""

    def test_dispose_clears_the_host(
            self, page, flask_server, big_log_file):
        _open_results(page, flask_server)
        _mount_source(page, big_log_file)
        _wait_loaded(page)

        empty_after = page.evaluate("""() => {
            window._handle.dispose();
            return document.getElementById("inspector-host")
                           .innerHTML.length;
        }""")
        assert empty_after == 0, (
            "dispose() must empty #inspector-host"
        )
