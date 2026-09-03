"""The Documents pane reads in the app's theme, not against it.

`docs-render.css` is a deliberately isolated theme: it does not consume the
app's tokens, so a rendered document is a reading surface in its own right and
a markdown palette can be tuned without touching the shell.  That isolation
cut both ways.  The pane defaulted to its LIGHT palette while `tokens.css` has
no light palette at all -- no ``prefers-color-scheme``, no ``data-theme`` --
so every document, and every mermaid diagram in one, opened as a white sheet
inside a dark shell until the reader found the toggle.

Seen 2026-08-23 while confirming the scheduler contract's diagrams render.

The toggle stays: a light reading surface is a real preference, and people
print from this pane.  What is pinned here is which way it STARTS, and that
the answer outlives a browser session -- a reading preference is not session
state, and `sessionStorage` made the reader re-choose every time.
"""
from __future__ import annotations


import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open_docs(page, base_url, doc="README.md"):
    """Open a real document.

    `#docs-render` ships `hidden` and is unhidden once something is rendered
    into it, so a bare `/documents` leaves it attached but invisible -- and an
    empty pane has no background to judge.  Opening a doc is also the path a
    reader actually takes.
    """
    page.goto(f"{base_url}/documents?doc={doc}")
    page.wait_for_selector("#docs-render", state="visible", timeout=10000)


class TestItStartsInTheAppsTheme:

    def test_a_first_visit_reads_dark(self, page, flask_server):
        """No stored preference -- the pane must match the shell around it."""
        _open_docs(page, flask_server)
        assert page.locator("#docs-render.docs-render-dark").count() == 1, \
            "the Documents pane opened light inside a dark app"

    def test_the_background_is_actually_dark(self, page, flask_server):
        """Not just the class -- the painted colour.  A class that no rule
        answers would satisfy the assertion above and still be white."""
        _open_docs(page, flask_server)
        rgb = page.evaluate(
            "() => getComputedStyle(document.querySelector('#docs-render'))"
            ".backgroundColor")
        nums = [int(n) for n in rgb.replace("rgb(", "").replace("rgba(", "")
                .rstrip(")").split(",")[:3]]
        assert max(nums) < 60, f"#docs-render is not dark: {rgb}"


class TestTheReaderCanStillChoose:

    def test_the_toggle_switches_and_is_remembered_across_sessions(
            self, page, flask_server):
        """A reading preference outlives the browser session.  It lived in
        `sessionStorage`, so the reader re-chose in every new one."""
        _open_docs(page, flask_server)
        page.locator("#docs-theme-btn").click()
        assert page.locator("#docs-render.docs-render-dark").count() == 0

        stored = page.evaluate("() => localStorage.getItem('docs-theme')")
        assert stored == "light", (
            "the choice was not written where it survives the session; "
            f"localStorage says {stored!r}")

        # A fresh load honours it -- the toggle is a preference, not a whim.
        _open_docs(page, flask_server)
        assert page.locator("#docs-render.docs-render-dark").count() == 0
