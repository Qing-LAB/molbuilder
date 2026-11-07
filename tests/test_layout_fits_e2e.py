"""Content fits its box, on every tab as it opens empty, at four widths.

`ui-contract.md` § 3 is the rule: layouts reflow because their CONTENT stops
fitting, and the page body never scrolls sideways.  This measures it on each
tab of the tab bar (`web/tabs.TABS`; the `/molview-demo` page is not one and
is not measured) as it opens with nothing in it -- no folder, no file: the
tab bar, the sidebar, the load bars and the forms a tab draws on opening.
A card drawn only for a folder or a structure is hidden on an empty page and
is not measured here -- the Task setup facts grid among them, shown only once
a hand-over has come over (`task-setup/viewer.js` ``renderCameOver``), whose
`minmax(9rem, 1fr)` once lost a conda hook's path off the edge of its card
with every UI test green.

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

from molbuilder.web.tabs import TABS                       # noqa: E402


#: The pages measured: every tab in `web/tabs.TABS`, each as it opens with
#: nothing in it -- a card that waits on a folder or a structure stays
#: hidden.
PAGES = [tab["path"] for tab in TABS]

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
def flask_server(isolated_projects_root_module):
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
def test_every_page_fits_its_width(page, flask_server, width):
    """§ 3, measured once a tab, as it opens empty: the page body never
    scrolls sideways -- wide content scrolls inside its own container, never
    by dragging the whole page -- and a layout reflows because its CONTENT
    stops fitting: no visible element's content runs past its own box where
    nothing scrolls it.

    THE SILENT FAILURE: an empty tab's own furniture -- its tab bar,
    sidebar, load bar or the forms it draws on opening -- clipping its text
    or pushing the page sideways at one of these widths, every UI test
    green.  What a folder or a structure brings onto the page is not
    measured (the module's docstring)."""
    page.set_viewport_size({"width": width, "height": 900})
    sideways, found = [], {}
    for path in PAGES:
        page.goto(f"{flask_server}{path}", wait_until="networkidle")
        measured = page.evaluate(_PROBE)
        if measured["scrollsX"]:
            sideways.append(path)
        real = _real(measured["over"])
        if real:
            found[path] = sorted(real, key=lambda f: -f["gap"])[:5]
    assert not sideways and not found, (
        f"at {width}px, these pages scroll sideways: {sideways}; content "
        f"overflows its container:\n" + json.dumps(found, indent=2))
