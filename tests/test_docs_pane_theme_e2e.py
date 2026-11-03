"""The Documents pane reads in the app's theme, not against it.

`docs-render.css` is a deliberately isolated theme: it does not consume the
app's tokens, so a rendered document is a reading surface in its own right and
a markdown palette can be tuned without touching the shell.  That isolation
cut both ways.  The pane defaulted to its LIGHT palette while `tokens.css` has
no light palette at all -- no ``prefers-color-scheme``, no ``data-theme`` --
so every document, and every mermaid diagram in one, opened as a white sheet
inside a dark shell until the reader found the toggle.

Seen 2026-08-23 while confirming the scheduler contract's diagrams render.

The rule is `tabs.md` § 5.5: the reading surface starts dark, and the ◑/◐
choice is a reading preference kept in the browser -- a page opened later in
the same browser starts on the surface last chosen.  The toggle stays: a light
reading surface is a real preference, and people print from this pane.  What
is pinned here is which way it STARTS, and that the choice is KEPT --
`sessionStorage` made the reader re-choose in every new tab.
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

    def test_the_background_is_actually_dark(self, page, flask_server):
        """No stored preference: the pane starts dark, as the app around it
        is (`tabs.md` § 5.5) -- the painted colour, not a class, which no rule
        might answer and still paint white.

        THE SILENT FAILURE (2026-08-23): every document, and every mermaid
        diagram in one, opened as a white sheet inside a dark shell."""
        _open_docs(page, flask_server)
        rgb = page.evaluate(
            "() => getComputedStyle(document.querySelector('#docs-render'))"
            ".backgroundColor")
        nums = [int(n) for n in rgb.replace("rgb(", "").replace("rgba(", "")
                .rstrip(")").split(",")[:3]]
        assert max(nums) < 60, f"#docs-render is not dark: {rgb}"


class TestTheReaderCanStillChoose:

    def test_the_toggle_switches_and_is_kept_for_later_pages(
            self, page, flask_server):
        """The ◑/◐ choice is kept for later pages in the same browser: a
        page opened afterwards starts on the surface last chosen
        (`tabs.md` § 5.5).

        THE SILENT FAILURE: the choice held for the one tab only -- it lived
        in `sessionStorage`, so the reader re-chose in every new one.

        MUTATION THIS MUST FAIL AGAINST: the choice written to
        `sessionStorage`, or not read when a page opens."""
        _open_docs(page, flask_server)
        page.locator("#docs-theme-btn").click()
        assert page.locator("#docs-render.docs-render-dark").count() == 0

        # A new page in the same browser, as a later visit opens one.
        later = page.context.new_page()
        try:
            _open_docs(later, flask_server)
            assert later.locator("#docs-render.docs-render-dark").count() == 0, (
                "a page opened later did not start on the surface last chosen")
        finally:
            later.close()
