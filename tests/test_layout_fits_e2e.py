"""Content fits its box, on every page, at every width that matters.

Nothing in the suite measured layout.  Every UI test asserted presence or
behaviour, so a card could clip its own text and stay green -- which is how
the Task-setup facts grid came to declare `minmax(9rem, 1fr)` for values like
``source /home/u/miniconda3/etc/profile.d/conda.sh``, and simply lose the
rest off the edge of the card.

`ui-contract.md` § 3 is the rule: layouts reflow because their CONTENT stops
fitting, and the page body never scrolls sideways.  This measures it.

One correction is built in, and it is the reason a naive version of this test
is worse than none.  **Children of a closed `<details>` keep layout boxes**,
sized to the collapsed summary -- so measuring them reports overflow no user
can ever see.  A first run produced 28 such findings, every one of them a
disclosure nobody had opened.
"""
from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


#: Every served page a person navigates to.
PAGES = ["/structure-optimization", "/spectrum-calculation", "/task-setup",
         "/results", "/molbuilder", "/documents"]

#: Wide, laptop, tablet, phone.  Four rather than every step: the reflow
#: points are what matter, and each width is a full page load.
WIDTHS = [1600, 1024, 768, 480]

#: Overflow that is not a defect, with the reason it is not.  An entry here
#: is a decision; the dict is deliberately awkward to add to.
_ALLOWED = {
    "CodeMirror": "the editor manages its own scrolling and reports the "
                  "document width, not a clipped box",
    "ps-resize-handle": "a 4px drag handle whose grip glyph is 6px; the "
                        "overflow IS the affordance",
}

#: Sub-pixel rounding, not clipping.  Browsers report fractional widths as
#: integers in these APIs and a 1-2px difference is the rounding, not text
#: leaving its box.
_SUBPIXEL_PX = 2

_PROBE = r"""
() => {
  const out = {scrollsX: document.documentElement.scrollWidth > innerWidth + 1,
               over: []};
  const label = (e) => e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") +
    (typeof e.className === "string" && e.className
      ? "." + e.className.trim().split(/\s+/).slice(0, 2).join(".") : "");
  for (const e of document.querySelectorAll("body *")) {
    const s = getComputedStyle(e);
    if (s.display === "none" || s.visibility === "hidden") continue;
    // A closed <details> keeps its children laid out at the summary's width.
    if (e.closest("details:not([open])") &&
        !e.matches("summary, details:not([open])")) continue;
    const r = e.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) continue;
    if (s.overflowX === "auto" || s.overflowX === "scroll") continue;
    const gap = e.scrollWidth - e.clientWidth;
    if (e.clientWidth > 0 && gap > 0) {
      out.over.push({el: label(e), gap: gap,
                     text: (e.textContent || "").trim().slice(0, 60)});
    }
  }
  return out;
}
"""


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _real(findings):
    keep = []
    for f in findings:
        if f["gap"] <= _SUBPIXEL_PX:
            continue
        if any(k in f["el"] for k in _ALLOWED):
            continue
        keep.append(f)
    return keep


@pytest.mark.parametrize("width", WIDTHS)
def test_no_page_scrolls_sideways(page, flask_server, width):
    """§ 3's top-level rule: wide content scrolls inside its own container,
    never by dragging the whole page."""
    page.set_viewport_size({"width": width, "height": 900})
    guilty = []
    for path in PAGES:
        page.goto(f"{flask_server}{path}", wait_until="networkidle")
        page.wait_for_timeout(250)
        if page.evaluate(_PROBE)["scrollsX"]:
            guilty.append(path)
    assert not guilty, f"these pages scroll sideways at {width}px: {guilty}"


@pytest.mark.parametrize("width", WIDTHS)
def test_content_fits_its_box(page, flask_server, width):
    page.set_viewport_size({"width": width, "height": 900})
    found = {}
    for path in PAGES:
        page.goto(f"{flask_server}{path}", wait_until="networkidle")
        page.wait_for_timeout(250)
        real = _real(page.evaluate(_PROBE)["over"])
        if real:
            found[path] = sorted(real, key=lambda f: -f["gap"])[:5]
    assert not found, (
        f"content overflows its container at {width}px:\n"
        + json.dumps(found, indent=2))
